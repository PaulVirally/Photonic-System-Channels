# ===========================================================================
# verlan.jl — topology optimisation seeded from a Verlan bound solution.
#
# Architecture
# ────────────
# TopOptData{T,MT64,MT32}
#   Immutable. Holds all static precomputed arrays in both fp64 (source of
#   truth) and fp32 (GEMM shadow). Lives on whatever device use_gpu dictates;
#   the rest of the code never touches the device flag again after construction.
#
# TopOptWork{T,GP,GM,DV,RV}
#   Mutable preallocated buffers. GP is a GEMMPrecision type parameter so that
#   fg! is fully type-stable without any runtime branches. To switch precision
#   call switch_gemm_precision — it rebuilds only the two N×k and k×k GEMM
#   buffers and returns a new TopOptWork, reusing everything else.
#
# make_fg!(data, work, β, tikhonov)
#   Returns a zero-alloc closure compatible with NLSolversBase.only_fg!.
#   All computation stays on device. Only two host↔device transfers per call:
#     • upload:   θ (N Float64, from Optim)           ~4 MB
#     • download: ∇ρ → G (N Float64, to Optim)        ~4 MB
#
# Precision strategy
# ──────────────────
# The dominant cost is the k×k GEMM with K=N inner products:
#   W_inv_gemm = −χ · ρU^H · G₀F,   ρU,G₀F ∈ ℂ^{N×k}
# This is ~256 GFLOPs for N=491520, k=512.
# On an A6000: fp32 ~38.7 TFLOP/s, fp64 ~1.2 TFLOP/s — a 32× gap.
# Strategy: run fp32 GEMM in the early (well-conditioned) Tikhonov stages,
# switch to fp64 GEMM in the final stages where small Tikhonov parameters
# make the system ill-conditioned and rounding errors matter.
# The fp32→fp64 upcast of the k×k result costs ~4 µs; the LU and solves
# (k=512) remain fp64 throughout.
#
# GPU dispatch
# ────────────
# _gemm_adj!(C, A, B, α, β) has a CuMatrix specialization that calls
# CUBLAS.gemm! directly with 'C','N' op flags, bypassing Julia's mul!
# dispatch which was shown to select the align1 slow-path kernel.
# ===========================================================================

using LinearAlgebra
using Optim, NLSolversBase, LineSearches
using Statistics: mean
using JLD2

import CUDA
import CUDA: CuArray, CuMatrix, CuVector
import CUDA.CUBLAS

@inline sigmoid(θ::T, β::T=one(T)) where {T} = one(T) / (one(T) + exp(-β * θ))
@inline inverse_sigmoid(ρ::T, β::T=one(T)) where {T} = log(ρ / (one(T) - ρ)) / β

"""GEMMPrecision trait for dispatching on GEMM precision in the hot loop."""
abstract type GEMMPrecision end

"""Use ComplexF32 for the N×k GEMM. ~32× faster on consumer/workstation GPUs."""
struct SinglePrecisionGEMM <: GEMMPrecision end

"""Use Complex{T} (fp64 for T=Float64) for the N×k GEMM. Needed when the
Tikhonov parameter is small and fp32 rounding corrupts the LU solve."""
struct DoublePrecisionGEMM <: GEMMPrecision end

_gemm_ctype(::Type{T}, ::SinglePrecisionGEMM) where {T<:AbstractFloat} = ComplexF32
_gemm_ctype(::Type{T}, ::DoublePrecisionGEMM) where {T<:AbstractFloat} = Complex{T}

# ---------------------------------------------------------------------------
# Low-level GEMM helper
# C ← α·A^H·B + β·C
#
# Generic CPU path: routes through OpenBLAS/MKL via mul!.
# GPU path: direct CUBLAS.gemm! with 'C','N' op codes.  The direct call is
# required because Julia's mul! dispatch was observed to select the align1
# CUTLASS fallback kernel (~10× slower) for this aspect ratio.
# ---------------------------------------------------------------------------
@inline function _gemm_adj!(C::AbstractMatrix, A::AbstractMatrix,
                             B::AbstractMatrix, α, β)
    mul!(C, A', B, α, β)
end

@inline function _gemm_adj!(C::CuMatrix{CGT}, A::CuMatrix{CGT},
                             B::CuMatrix{CGT}, α, β) where {CGT}
    CUBLAS.gemm!('C', 'N', CGT(α), A, B, CGT(β), C)
end

# ---------------------------------------------------------------------------
# TopOptData — static, immutable after construction
#
# Type parameters
#   T    working float precision (Float64 recommended)
#   MT64 concrete matrix type in working precision (Array or CuArray)
#   MT32 concrete matrix type for fp32 shadows (must live on same device)
# ---------------------------------------------------------------------------
struct TopOptData{T  <: AbstractFloat,
                  MT64 <: AbstractMatrix{Complex{T}},
                  MT32 <: AbstractMatrix{ComplexF32}}
    # Large N×k arrays on device — both precisions
    U_uu_f64  :: MT64   # N×k
    G₀F_f64   :: MT64   # N×k  (= G₀_disjoint * U_uu, precomputed)
    U_s_f64   :: MT64   # N×k  (= Πₛ * U_uu, source-region projection)
    U_uu_f32  :: MT32   # N×k, fp32 shadow of U_uu_f64
    G₀F_f32   :: MT32   # N×k, fp32 shadow of G₀F_f64

    # k×k Hermitian matrices — small enough to live on device without issue.
    # Stored as dense matrices; Hermitian wrapper applied at point of use.
    obj_mat_dev :: MT64   # k×k  objective Hermitian form
    S_sub_dev   :: MT64   # k×k  U_s' * U_s (Gram matrix in source subspace)

    χ     :: Complex{T}
    ζ     :: T
    k     :: Int
    N     :: Int
    bound :: T
end

"""
    TopOptData(U_uu, G₀F, U_s, obj_mat, S_sub, χ, ζ, bound)

Construct TopOptData from fp64 device arrays.  fp32 shadows are built
automatically on the same device.  `obj_mat` and `S_sub` may be CPU matrices;
they are uploaded to device here.
"""
function TopOptData(
    U_uu    :: AbstractMatrix{Complex{T}},
    G₀F     :: AbstractMatrix{Complex{T}},
    U_s     :: AbstractMatrix{Complex{T}},
    obj_mat :: AbstractMatrix,
    S_sub   :: AbstractMatrix,
    χ :: Complex{T}, ζ :: T, bound :: T
) where {T <: AbstractFloat}
    N, k = size(U_uu)
    @assert size(G₀F)  == (N, k) "G₀F must be N×k"
    @assert size(U_s)  == (N, k) "U_s must be N×k"
    @assert size(obj_mat) == (k, k) "obj_mat must be k×k"
    @assert size(S_sub)   == (k, k) "S_sub must be k×k"

    # fp32 shadows: same device (similar propagates device/CPU correctly)
    U32 = similar(U_uu, ComplexF32, N, k);  copyto!(U32, U_uu)
    G32 = similar(G₀F,  ComplexF32, N, k);  copyto!(G32, G₀F)

    # Upload k×k matrices to device
    obj_dev = similar(U_uu, Complex{T}, k, k);  copyto!(obj_dev, Complex{T}.(obj_mat))
    S_dev   = similar(U_uu, Complex{T}, k, k);  copyto!(S_dev,   Complex{T}.(S_sub))

    TopOptData(U_uu, G₀F, U_s, U32, G32, obj_dev, S_dev, χ, ζ, k, N, bound)
end

# Select input matrices in the correct precision for the GEMM
@inline _U_gemm(d::TopOptData, ::SinglePrecisionGEMM) = d.U_uu_f32
@inline _U_gemm(d::TopOptData, ::DoublePrecisionGEMM) = d.U_uu_f64
@inline _G_gemm(d::TopOptData, ::SinglePrecisionGEMM) = d.G₀F_f32
@inline _G_gemm(d::TopOptData, ::DoublePrecisionGEMM) = d.G₀F_f64

# ---------------------------------------------------------------------------
# TopOptWork — mutable preallocated workspace
#
# Type parameters (all inferred from constructor; never set manually)
#   T    working float (Float64)
#   GP   GEMMPrecision trait
#   GM   matrix type for GEMM buffers (e.g. CuMatrix{ComplexF32})
#   DM   matrix type for working-precision device matrices (e.g. CuMatrix{ComplexF64})
#   DV   vector type for working-precision device vectors (e.g. CuVector{ComplexF64})
#   RV   real device vector type (e.g. CuVector{Float64})
#
# Buffer layout
# ─────────────
#   GEMM precision (GM): ρU_gemm (N×k), W_inv_gemm (k×k)
#   Working precision (DM): W_inv_dev (k×k — upcast from W_inv_gemm)
#   Working precision (DV): w_sub, λ_sub, Hw_sub, rhs, q_sub (k), Uλ, Fw (N)
#   Real device (RV): ρ_buf, θ_dev, ∇ρ (N)
#   CPU: G_cpu (N) — gradient output buffer for Optim
#
# Note: q_sub lives on device and is updated by update_optimal_source!, NOT
# inside fg!.  This decoupling is deliberate: source updates are expensive
# (k×k inverse + eigensolver) and should not run every LBFGS step.
# ---------------------------------------------------------------------------
mutable struct TopOptWork{T <: AbstractFloat, GP <: GEMMPrecision,
                           GM  <: AbstractMatrix,
                           DM  <: AbstractMatrix,
                           DV  <: AbstractVector,
                           RV  <: AbstractVector}
    # GEMM-precision device buffers
    ρU_gemm    :: GM   # N×k  (workspace: ρ ⊙ U_gemm)
    W_inv_gemm :: GM   # k×k  (GEMM output before upcast)

    # Working-precision device buffers
    W_inv_dev  :: DM   # k×k  (fp64, upcast from W_inv_gemm; input to lu!)
    w_sub      :: DV   # k    (forward solve: W_inv_dev \ q_sub)
    λ_sub      :: DV   # k    (adjoint solve: W_inv_dev^H \ rhs)
    Hw_sub     :: DV   # k    (obj_mat * w_sub)
    rhs        :: DV   # k    (= 2ζ · Hw_sub, adjoint RHS)
    q_sub      :: DV   # k    (current optimal source in U_uu subspace)
    Uλ         :: DV   # N    (U_uu * λ_sub)
    Fw         :: DV   # N    (G₀F  * w_sub)

    # Real device buffers
    ρ_buf  :: RV   # N    (sigmoid(θ, β))
    θ_dev  :: RV   # N    (θ uploaded from Optim each call)
    ∇ρ     :: RV   # N    (gradient before copy to CPU)

    # CPU-side output (Optim requires CPU vectors)
    G_cpu :: Vector{T}   # N

    _gp :: GP   # precision tag — drives dispatch in make_fg!
end

"""
    TopOptWork(data, gp)

Allocate all workspace buffers on the same device as `data.U_uu_f64`.
"""
function TopOptWork(data::TopOptData{T}, gp::GP) where {T, GP <: GEMMPrecision}
    N, k = data.N, data.k
    CT  = Complex{T}
    CGT = _gemm_ctype(T, gp)

    ref = data.U_uu_f64   # reference array: drives device placement

    TopOptWork(
        similar(ref, CGT, N, k),   # ρU_gemm
        similar(ref, CGT, k, k),   # W_inv_gemm
        similar(ref, CT,  k, k),   # W_inv_dev
        similar(ref, CT,  k),      # w_sub
        similar(ref, CT,  k),      # λ_sub
        similar(ref, CT,  k),      # Hw_sub
        similar(ref, CT,  k),      # rhs
        similar(ref, CT,  k),      # q_sub
        similar(ref, CT,  N),      # Uλ
        similar(ref, CT,  N),      # Fw
        similar(ref, T,   N),      # ρ_buf
        similar(ref, T,   N),      # θ_dev
        similar(ref, T,   N),      # ∇ρ
        zeros(T, N),               # G_cpu (always CPU)
        gp
    )
end

"""
    switch_gemm_precision(work, data, new_gp)

Return a new TopOptWork using `new_gp` for the GEMM step.
All non-GEMM buffers (W_inv_dev, w_sub, … G_cpu) are reused — no reallocation.
The current ρ_buf, θ_dev, q_sub state is preserved across the switch.

Typical usage:
    work = switch_gemm_precision(work, data, DoublePrecisionGEMM())
"""
function switch_gemm_precision(work::TopOptWork{T}, data::TopOptData{T},
                                new_gp::GP) where {T, GP <: GEMMPrecision}
    CGT = _gemm_ctype(T, new_gp)
    ref = data.U_uu_f64
    N, k = data.N, data.k
    TopOptWork(
        similar(ref, CGT, N, k),   # new ρU_gemm
        similar(ref, CGT, k, k),   # new W_inv_gemm
        work.W_inv_dev,            # ─┐
        work.w_sub,                #  │
        work.λ_sub,                #  │
        work.Hw_sub,               #  │ reused
        work.rhs,                  #  │
        work.q_sub,                #  │
        work.Uλ,                   #  │
        work.Fw,                   #  │
        work.ρ_buf,                #  │
        work.θ_dev,                #  │
        work.∇ρ,                   # ─┘
        work.G_cpu,
        new_gp
    )
end

# ---------------------------------------------------------------------------
# Upcast W_inv_gemm → W_inv_dev
# For fp64 GEMM this is a same-type copy (cheap, ~4 µs for k=512).
# For fp32 GEMM this performs the ComplexF32 → ComplexF64 broadcast cast.
# In both cases W_inv_dev is a fresh copy, safe to lu! in place.
# ---------------------------------------------------------------------------
@inline function _upcast_W_inv!(work::TopOptWork)
    work.W_inv_dev .= work.W_inv_gemm
end

# ---------------------------------------------------------------------------
# make_fg! — the optimization closure
#
# Returns fg!(F, G, θ) compatible with NLSolversBase.only_fg!.
# θ is a CPU Vector{T} supplied by Optim.
#
# Hot path (zero allocs after warm-up):
#   1.  copyto!(θ_dev, θ)                — upload θ (N×8 bytes ≈ 4 MB)
#   2.  θ_dev → ρ_buf via sigmoid        — GPU elementwise kernel
#   3.  ρU_gemm = ρ_buf ⊙ U_gemm        — GPU elementwise kernel (N×k)
#   4.  W_inv_gemm = −χ · ρU_gemm^H · G₀F_gemm   — CUBLAS cgemm/zgemm
#   5.  W_inv_dev .= W_inv_gemm          — upcast (fp32→fp64) or copy
#   6.  W_inv_dev[diag] += (1 + tikho)  — GPU in-place
#   7.  W_fac = lu!(W_inv_dev)           — GPU cuSOLVER / CPU LAPACK
#   8.  w_sub = W_fac \ q_sub            — triangular solve
#   9.  Hw_sub = obj_mat_dev * w_sub     — k×k GPU matmul
#  10.  val = ζ · Re(w_sub · Hw_sub)
#  [gradient path]
#  11.  λ_sub = W_fac^H \ (2ζ · Hw_sub)
#  12.  Uλ = U_uu_f64 * λ_sub           — N×k GPU matmul
#  13.  Fw = G₀F_f64 * w_sub            — N×k GPU matmul
#  14.  ∇ρ = −Re(χ·conj(Uλ)·Fw) · β·ρ·(1−ρ)   — GPU broadcast
#  15.  copyto!(G, ∇ρ)                  — download gradient (N×8 bytes ≈ 4 MB)
# ---------------------------------------------------------------------------
function make_fg!(data::TopOptData{T}, work::TopOptWork{T},
                  β::T, tikhonov::T) where {T}
    χ   = data.χ
    ζ   = data.ζ
    neg_χ = -χ
    two_ζ = T(2) * ζ
    tik1  = T(1) + tikhonov

    # Capture concretely typed references (no dynamic dispatch in closure)
    U_gemm  = _U_gemm(data, work._gp)
    G₀_gemm = _G_gemm(data, work._gp)

    function fg!(F, G, θ::AbstractVector)
        # ── 1–2. Upload θ, compute ρ ─────────────────────────────────────
        copyto!(work.θ_dev, θ)
        @. work.ρ_buf = sigmoid(work.θ_dev, β)

        # ── 3. ρU = diag(ρ) * U_gemm  (fused broadcast, GEMM precision) ─
        # Real ρ cast to GEMM float inside broadcast — no separate buffer.
        GT = real(eltype(work.ρU_gemm))
        @. work.ρU_gemm = GT(work.ρ_buf) * U_gemm

        # ── 4. W_inv_gemm = −χ · ρU^H · G₀_gemm ─────────────────────────
        _gemm_adj!(work.W_inv_gemm, work.ρU_gemm, G₀_gemm, neg_χ, zero(eltype(work.W_inv_gemm)))

        # ── 5–6. Upcast + regularised identity ───────────────────────────
        _upcast_W_inv!(work)
        work.W_inv_dev[diagind(work.W_inv_dev)] .+= tik1

        # ── 7–8. LU factorisation + forward solve ─────────────────────────
        W_fac = lu!(work.W_inv_dev)   # in-place; W_inv_dev now holds LU factors
        ldiv!(work.w_sub, W_fac, work.q_sub)

        # ── 9–10. Objective ───────────────────────────────────────────────
        mul!(work.Hw_sub, data.obj_mat_dev, work.w_sub)
        val = ζ * real(dot(work.w_sub, work.Hw_sub))

        if G !== nothing
            # ── 11. Adjoint solve ─────────────────────────────────────────
            @. work.rhs = two_ζ * work.Hw_sub
            ldiv!(work.λ_sub, W_fac', work.rhs)

            # ── 12–13. N-vector matvecs (fp64, both on device) ───────────
            mul!(work.Uλ, data.U_uu_f64, work.λ_sub)
            mul!(work.Fw, data.G₀F_f64,  work.w_sub)

            # ── 14. Gradient + sigmoid chain rule (fused broadcast) ───────
            @. work.∇ρ = -real(χ * conj(work.Uλ) * work.Fw) *
                         (β * work.ρ_buf * (one(T) - work.ρ_buf))

            # ── 15. Download gradient to CPU ──────────────────────────────
            copyto!(work.G_cpu, work.∇ρ)
            G .= work.G_cpu
        end

        F !== nothing && return -val
    end

    return fg!
end

# ---------------------------------------------------------------------------
# update_optimal_source!
#
# Solves the generalised eigenproblem:
#   S·W^H·H·W·S · v = λ · S · v
# where W = inv(W_inv), H = obj_mat, S = U_s'·U_s (all k×k).
# The eigenvector with the largest eigenvalue gives the optimal source
# direction in the subspace, then projected back to full space and
# normalised.
#
# This is expensive (k×k matrix inverse + dense eigensolver) so it should
# NOT be called every fg! evaluation.  Call it every source_update_freq
# Optim iterations at most.
#
# After calling this, work.q_sub is updated in-place on device.
# ---------------------------------------------------------------------------
function update_optimal_source!(work::TopOptWork{T}, data::TopOptData{T}) where {T}
    k = data.k
    CT = Complex{T}

    # Pull k×k matrices to CPU for eigensolver (small, cheap transfer)
    W_inv_cpu = Array(work.W_inv_dev)   # k×k, already LU-factored — need original
    # NOTE: W_inv_dev was modified in-place by lu! so we reconstruct W from
    # the factorisation.  Alternatively, keep a copy.  Here we just recompute
    # from GEMM output which is still valid in W_inv_gemm (not touched by lu!).
    W_gemm_cpu = Array(work.W_inv_gemm)   # k×k, last GEMM output (pre-LU)
    W_f64 = Matrix{CT}(W_gemm_cpu)
    @inbounds for i in 1:k
        W_f64[i,i] += one(T)   # Tikhonov was added before lu!, reflected here
    end
    # W_inv_subspace in the unmodified sense: compute its inverse
    W_cpu = inv(W_f64)

    obj_cpu  = Array(data.obj_mat_dev)
    S_cpu    = Array(data.S_sub_dev)

    B = W_cpu' * (obj_cpu * W_cpu)   # k×k

    # Generalised eigenproblem: S·B·S v = λ S v
    SBS = Hermitian(S_cpu * B * S_cpu)
    S_h = Hermitian(S_cpu)
    eig = eigen(SBS, S_h)
    best = argmax(eig.values)
    v   = eig.vectors[:, best]

    # Map back: q_full = U_s * v, normalise in full space
    # Do this on device to avoid N-vector download
    v_dev = similar(data.U_s_f64, eltype(data.U_s_f64), k)
    copyto!(v_dev, CT.(v))
    mul!(work.Fw, data.U_s_f64, v_dev)   # reuse Fw as temporary N-vector
    n = sqrt(real(dot(work.Fw, work.Fw)))
    work.Fw ./= n

    # q_sub = U_uu^H * q_full
    mul!(work.q_sub, data.U_uu_f64', work.Fw)
    return nothing
end

# ---------------------------------------------------------------------------
# objective_matrix_subspace
#
# Returns the k×k Hermitian matrix H such that the objective is
#   f = ζ · w^H · H · w,   w = W_inv \ q
# accumulated over the top-k singular values.
# ---------------------------------------------------------------------------
function objective_matrix_subspace(k_target::Int, U_uu::AbstractMatrix,
                                    Vur_asym::AbstractMatrix,
                                    Γ::AbstractVector, ζ::T) where {T}
    k = size(U_uu, 2)
    CT = complex(T)
    vdagU = Array(Vur_asym' * U_uu)   # k×k, pull to CPU for accumulation
    Γ_cpu = Array(Γ)                   # pull to CPU for scalar indexing in loop
    H = zeros(CT, k, k)
    tmp = zeros(CT, k, k)
    for i in k:-1:1
        row = view(vdagU, i:i, :)      # 1×k
        mul!(tmp, row', row)           # k×k rank-1 update
        H .+= (ζ * Γ_cpu[i]) .* tmp
        i == k_target && break
    end
    return Hermitian(H)
end

# ---------------------------------------------------------------------------
# run_topopt!
#
# Outer loop: Tikhonov continuation with optional GEMM precision switching.
# Returns (ρ_opt, objective_history) where ρ_opt is a CPU Vector{T}.
#
# Arguments
# ─────────
# ρ_init          Initial density (CPU Vector{T}), e.g. from Verlan seed.
# tikhonov_schedule  Decreasing sequence of Tikhonov parameters.
# iters_per_stage    LBFGS iterations per Tikhonov stage.
# fp32_stages        Indices of tikhonov_schedule for which to use fp32 GEMM.
#                    All other stages use fp64 GEMM.
#                    Default: use fp32 for all but the last two stages.
# source_update_freq Recompute optimal source every this many LBFGS calls.
# β                  Sigmoid sharpness (fixed for now; extend to schedule if needed).
# ---------------------------------------------------------------------------
function run_topopt!(
    data    :: TopOptData{T},
    work    :: TopOptWork{T},
    ρ_init  :: AbstractVector{T};
    tikhonov_schedule  :: Vector{T}  = T.([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 0.0]),
    iters_per_stage    :: Vector{Int} = [10, 20, 30, 50, 100, 200, 500],
    fp32_stages        :: AbstractSet = Set(1:max(1, length(tikhonov_schedule)-2)),
    source_update_freq :: Int         = 10,
    β                  :: T           = one(T),
    inner_iters        :: Int         = 200,
) where {T}

    @assert length(tikhonov_schedule) == length(iters_per_stage)
    N = data.N
    ρ_min, ρ_max = T(1e-6), T(1 - 1e-6)

    stage_history = [(-Inf, zeros(T, N)) for _ in tikhonov_schedule] # (objective, ρ) for best iter in each stage

    ρ = clamp.(Vector{T}(ρ_init), ρ_min, ρ_max)
    prev_ρ = copy(ρ)
    θ = inverse_sigmoid.(ρ, β)
    prev_θ = copy(θ)

    objective_history = T[]
    current_gp = work._gp   # track current precision for logging

    initial_grayness = mean(4 .* ρ .* (one(T) .- ρ))
    initial_fill_ratio = mean(ρ)
    @info string(now()) * " [topopt] Starting optimization with initial grayness = $(round(initial_grayness, sigdigits=6)), fill ratio = $(round(initial_fill_ratio, sigdigits=6))"

    prev_inner_iters = 0
    converged_last = true
    HAGER_ZHANG_SWITCH_ITER = 5  # switch to Hager-Zhang if back tracking converges too quickly

    for (stage, tikhonov) in enumerate(tikhonov_schedule)
        # ── Switch GEMM precision if needed ─────────────────────────────
        target_gp = stage in fp32_stages ? SinglePrecisionGEMM() : DoublePrecisionGEMM()
        if typeof(target_gp) != typeof(current_gp)
            work = switch_gemm_precision(work, data, target_gp)
            current_gp = target_gp
            @info string(now()) * " [topopt] Stage $stage: switched to $(typeof(target_gp))"
        end

        fg! = make_fg!(data, work, β, tikhonov)
        lbfgs_bt = LBFGS(alphaguess  = LineSearches.InitialStatic(),
                      linesearch  = LineSearches.BackTracking())
        lbfgs_hz = LBFGS(alphaguess  = LineSearches.InitialStatic(),
                      linesearch  = LineSearches.HagerZhang())

        @info string(now()) * " [topopt] Stage $stage/$(length(tikhonov_schedule)): " *
              "tikhonov=$(tikhonov), $(iters_per_stage[stage]) outer iters, " *
              "$(typeof(current_gp))"

        call_count = Ref(0)

        # Wrap fg! to trigger source updates and track call count
        # function fg_with_source_update!(F, G, θ_)
        #     call_count[] += 1
        #     # if call_count[] % source_update_freq == 0
        #         # @info string(now()) * " [topopt] Stage $stage: updating optimal source (call count = $(call_count[]))"
        #         # update_optimal_source!(work, data)
        #     # end
        #     return fg!(F, G, θ_)
        # end

        for iter in 1:iters_per_stage[stage]
            clamp!(θ, inverse_sigmoid(ρ_min, β), inverse_sigmoid(ρ_max, β))

            if converged_last && tikhonov > tikhonov_schedule[stage]
                @info string(now()) * " [topopt] Stage $stage iter $iter: objective converged with engourged Tikhonov $(round(tikhonov, sigdigits=3)). Reducing Tikhonov to $(round(tikhonov / 1.1, sigdigits=3))"
                tikhonov /= 1.1
            end

            use_hz = !converged_last ? true : (iter > 1 && prev_inner_iters <= HAGER_ZHANG_SWITCH_ITER)

            res = optimize(
                # NLSolversBase.only_fg!(fg_with_source_update!), θ,
                NLSolversBase.only_fg!(fg!), θ,
                use_hz ? lbfgs_hz : lbfgs_bt,
                Optim.Options(iterations = inner_iters, show_trace = false)
            )

            θ  .= Optim.minimizer(res)
            ρ  .= clamp.(sigmoid.(θ, β), ρ_min, ρ_max)
            obj = -Optim.minimum(res)

            grayness     = mean(4 .* ρ .* (one(T) .- ρ))
            fill_ratio = mean(ρ)
            pct_of_bound = 100 * obj / data.bound
            prev_inner_iters = Optim.iterations(res)

            # TODO: here
            # if obj < data.bound && Optim.converged(res)
            if obj > stage_history[stage][1]
                stage_history[stage] = (obj, copy(ρ))
            end

            if obj > data.bound && !Optim.converged(res)
                ρ .= prev_ρ
                θ .= prev_θ
                tikhonov *= 1.5
                fg! = make_fg!(data, work, β, tikhonov)
                @warn string(now()) * " [topopt] Stage $stage iter $iter: objective $(round(obj, sigdigits=6)) exceeded bound $(round(data.bound, sigdigits=6)) without converging! Reverting with increased Tikhonov ($(round(tikhonov, sigdigits=3)))"
            else
                prev_ρ .= ρ
                prev_θ .= θ
            end
            push!(objective_history, obj)

            @info string(now()) * " [topopt] Stage $stage iter $iter ($(use_hz ? "HZ" : "BT")): " *
                  "f=$(round(obj, sigdigits=6)) " *
                  "($(round(pct_of_bound, sigdigits=3))% of bound), " *
                  "grayness=$(round(grayness, sigdigits=3)), " *
                  "fill_ratio=$(round(fill_ratio, sigdigits=3)), " *
                  "converged=$(Optim.converged(res)), " *
                  "inner_iters=$(Optim.iterations(res))"

            # if iter % source_update_freq == 0
                # @info string(now()) * " [topopt] Stage $stage iter $iter: updating optimal source"
            update_optimal_source!(work, data)
            # end
        end
    end

    return ρ, objective_history, stage_history
end

# ---------------------------------------------------------------------------
# _verlan_sr — I/O, precomputation, entry point for SR systems
# ---------------------------------------------------------------------------
function _verlan_sr(compute_env::ComputeEnvironment, smr::SMRSystem,
                    rsvd_params::RSVDParams)
    gpu = use_gpu(compute_env)
    T   = Float64

    @info string(now()) * " [verlan_sr] Loading Verlan seed"
    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out      = jldopen(jld_out_path, "r")
    ρ_seed       = Float64.(read_array(jld_out, "ρ_verlan_1", false))
    ord_idxs     = jld_out["ordering_idxs"]
    χ            = ComplexF64(jld_out["χ"])
    ζ            = abs2(χ) / imag(χ)
    k_target     = 1
    bound        = Float64(jld_out["μ_subspace"][k_target])
    close(jld_out)

    @info string(now()) * " [verlan_sr] Loading subspace operators"
    jld_in_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_in      = jldopen(jld_in_path, "r")
    U_uu_cpu  = ComplexF64.(read_array(jld_in, "UU_asym/V",  false))
    Vur_cpu   = ComplexF64.(read_array(jld_in, "UR_asym/V",  false))[:, ord_idxs]
    Γ_cpu     = -Float64.( read_array(jld_in, "UR_asym/D",  false))[ord_idxs]
    close(jld_in)
    N, k = size(U_uu_cpu)
    @info string(now()) * " [verlan_sr] N=$N, k=$k"

    # ── Upload to device ──────────────────────────────────────────────────
    to_dev(A) = gpu ? CuArray(A) : A
    U_uu_dev = to_dev(U_uu_cpu)

    @info string(now()) * " [verlan_sr] Precomputing G₀F = G₀_disjoint * U_uu"
    G₀_uu        = load_greens_function(compute_env, smr, Design, Design)
    Πₛ, _, G₀_dis = projected_operators(G₀_uu, smr)
    G₀F_dev      = to_dev(ComplexF64.(opmat(G₀_dis, U_uu_dev)))

    @info string(now()) * " [verlan_sr] Precomputing U_s = Πₛ * U_uu"
    U_s_dev = to_dev(ComplexF64.(opmat(Πₛ, U_uu_dev)))

    @info string(now()) * " [verlan_sr] Computing S_sub = U_s' * U_s"
    S_sub_cpu = Array(U_s_dev' * U_s_dev)

    @info string(now()) * " [verlan_sr] Computing objective matrix"
    obj_mat_cpu = Matrix(objective_matrix_subspace(
        k_target,
        U_uu_dev,
        to_dev(Vur_cpu),
        to_dev(Γ_cpu),
        ζ
    ))

    # ── Construct data and workspace ──────────────────────────────────────
    data = TopOptData(U_uu_dev, G₀F_dev, U_s_dev,
                      obj_mat_cpu, S_sub_cpu,
                      ComplexF64(χ), Float64(ζ), Float64(bound))

    # Start in fp32; switch to fp64 in the final two Tikhonov stages.
    work = TopOptWork(data, SinglePrecisionGEMM())

    # ── Seed the optimal source from initial ρ ────────────────────────────
    # Run one warm-up fg! pass to populate W_inv_dev (needed by update_optimal_source!)
    copyto!(work.ρ_buf, Float64.(clamp.(ρ_seed, 1e-6, 1-1e-6)))
    GT = real(eltype(work.ρU_gemm))
    work.ρU_gemm .= GT.(work.ρ_buf) .* _U_gemm(data, work._gp)
    _gemm_adj!(work.W_inv_gemm, work.ρU_gemm, _G_gemm(data, work._gp),
               -data.χ, zero(eltype(work.W_inv_gemm)))
    _upcast_W_inv!(work)
    work.W_inv_dev[diagind(work.W_inv_dev)] .+= 2.0   # tikhonov=1 for seeding
    update_optimal_source!(work, data)

    # ── Run optimisation ──────────────────────────────────────────────────
    # tikhonov_schedule = Float64.([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 0.0])
    # iters_per_stage   = [10, 20, 30, 50, 100, 200, 500]
    tikhonov_schedule = Float64.([1e1, 1e0, 1e-1, 1e-2, 1e-3])
    iters_per_stage   = [10, 20, 30, 30, 30]
    n_stages = length(tikhonov_schedule)

    ρ_opt, history, stage_history = run_topopt!(
        data, work, Float64.(ρ_seed);
        tikhonov_schedule  = tikhonov_schedule,
        iters_per_stage    = iters_per_stage,
        # fp32_stages        = Set(1:n_stages-2),   # last 2 stages use fp64 GEMM
        fp32_stages        = Set(1:n_stages),
        source_update_freq = 1,
        inner_iters        = 1000,
    )

    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out = jldopen(jld_out_path, "w")
    if !haskey(jld_out, "ρ_verlan_topopt")
        jld_out["ρ_verlan_topopt"] = ρ_opt
    end
    if !haskey(jld_out, "topopt_history")
        jld_out["topopt_history"] = history
    end
    if !haskey(jld_out, "stage_history")
        jld_out["stage_history"] = stage_history
    end
    close(jld_out)

    # @info string(now()) * " [verlan_sr] Completed. Final objective history length: $(length(history))"
    return ρ_opt, history, stage_history
end

function verlan()
    compute_env, smr, rsvd_params = parse_args()
    gpu = use_gpu(compute_env)
    if gpu
        @info string(now()) * " [verlan] GPU mode, device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [verlan] CPU mode"
    end
    isnothing(mediator(smr)) ? _verlan_sr(compute_env, smr, rsvd_params) :
                               error("[verlan] SMR mediator not implemented")
end
