using CUDA
using Dates
using LinearAlgebra
using LinearMaps
using JLD2
using MatrixFreeRandomizedLinearAlgebra
import Funicular

# ---------------------------------------------------------------------------
# The system-design bound on σₙ(Pᵣₛ), Eq. (37) of the paper
#
#     σₙ² ≤ min over μₚ > 0, τ < 1 of μₙ(μₚ, τ) = μₚ σ₁(Πₛ (𝟙(τ) − H)⁻¹ Πₛ),   𝟙(τ) ≻ H,
#
# with 𝟙(τ) = 1 − τΠᵣ, H = ζ(A⁽ⁿ⁾/μₚ − τA − G), A = (−G⁰ᵣᵤ)ᵃ (both signs), G = (G⁰ᵤᵤ)ᵃ and
# A⁽ⁿ⁾ = A without its n−1 largest eigencomponents. Any feasible (μₚ, τ) is a bound. τ < 0 is
# allowed: τ weights the identity ζ⁻¹Πᵣ = (−G⁰ᵣᵤ)ᵃ on t (Eq. (24)), an equality, so any sign works.
#
# H is represented in an orthonormal `basis` P: the kept eigenvectors g of A (all signs,
# `m_kept` columns), augmented with the top `k_uu` eigenvectors of Asym(G⁰ᵤᵤ). The
# augmentation keeps G represented at large separation, where span(g) collapses
# (m = 15 of N_u = 196,608 at 1 λ, s = 357/32 λ); it saturates near k_uu = 512.
#
# In the basis, with W = Pᴴg, S = PₛᴴPₛ, Q(τ) = Pᴴ𝟙(τ)⁻¹P = S + (1−τ)⁻¹(1 − S) and h the
# projected H, Woodbury gives Πₛ(𝟙(τ) − PhPᴴ)⁻¹Πₛ = Πₛ + PₛKPₛᴴ with K = h(1 − Qh)⁻¹.
# Whitening by Q^½ and by R₀ = 1 + ζQ^½(τ(A − A⁽ⁿ⁾)_P + G_P)Q^½ = LLᴴ ⪰ 1 (the operator at
# μₚ = 1/τ; for τ < 0 it can be indefinite) leaves one Cholesky and one eigenproblem per (n, τ),
#
#     L⁻¹ζQ^½A⁽ⁿ⁾_P Q^½L⁻ᴴ = V diag(ν) Vᴴ,   K(μₚ) = Y diag(1/(1 − (1/μₚ − τ)νₖ)) Yᴴ − Q⁻¹,
#
# Y = Q^-½L⁻ᴴV, feasible on the μₚ interval where every 1 − (1/μₚ − τ)νₖ > 0 (the negative Γ
# kept in A⁽ⁿ⁾ make it an interval rather than (ν_max, ∞)). The nonzero eigenvalues of PₛKPₛᴴ
# are those of FᴴKF (FFᴴ = S); sender directions outside span(Pₛ) add the eigenvalue 1.
# ---------------------------------------------------------------------------

const DEFAULT_K_UU = 512
const DEFAULT_GAMMA_RTOL = 1e-12
# Asym(G⁰ᵤᵤ) has a well-separated spectrum, so 4 power iterations converge it
# (residuals at 1e-13 at 1 λ). Oversamples may be cut to the floor to fit the card.
const UU_OVERSAMPLES = 50
const UU_POWER_ITERS = 4
const UU_MIN_OVERSAMPLES = 10
# Below this many Asym(G⁰ᵤᵤ) vectors the augmentation buys almost nothing (the 1 λ
# scan reached the λ/4 reference at 128), so a point that cannot afford it is refused.
const K_UU_FLOOR = 64
# Drops augmentation columns that are numerically in span(g): |R_ii| < rtol · max|R_ii|.
const AUG_QR_RTOL = 1e-10
# Width of the final log μₚ bracket. μₙ is flat at its minimum, so the value is
# converged to about the square of this.
const MU_P_LOG_TOL = 1e-3
# The dual runs on every channel up to DUAL_DENSE_CHANNELS, then on a geometric ladder of
# ratio DUAL_CHANNEL_RATIO; the channels in between take the bound of the sample below them.
const DUAL_DENSE_CHANNELS = 50
const DUAL_CHANNEL_RATIO = 1.1
# τ warm start: walk at most TAU_MAX_STEPS steps from the previous channel's τ, and sweep the
# whole grid instead if the result is more than TAU_RESWEEP_RTOL above the previous channel's.
const TAU_MAX_STEPS = 3
const TAU_RESWEEP_RTOL = 0.01
# τ stays below 1 − TAU_EDGE, where 𝟙(τ) is still comfortably invertible.
const TAU_EDGE = 1e-3
const TAU_PREP_CACHE = 8
# How often the per-n progress is written to the checkpoint, so short (backfilled) jobs lose little.
const CHECKPOINT_SECONDS = 120

"""
    qr_thin_rdiag!(A) -> (Q, d)

The thin `Q` of a tall `A`, in place, and `d = abs.(diag(R))`, the cheap handle an
unpivoted QR offers on column dependence.
"""
function qr_thin_rdiag!(A::Matrix{T}) where {T<:LinearAlgebra.BlasFloat}
    size(A, 1) >= size(A, 2) || throw(ArgumentError("qr_thin_rdiag! wants a tall matrix, got $(size(A))"))
    A, τ = LinearAlgebra.LAPACK.geqrf!(A)
    d = abs.(diag(A))
    LinearAlgebra.LAPACK.orgqr!(A, τ)
    return A, d
end

function qr_thin_rdiag!(A::CuMatrix)
    size(A, 1) >= size(A, 2) || throw(ArgumentError("qr_thin_rdiag! wants a tall matrix, got $(size(A))"))
    n = size(A, 2)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    d = abs.(diag(Array(view(A, 1:n, 1:n))))
    CUDA.CUSOLVER.orgqr!(A, τ)
    return A, d
end

"""
    augmented_basis(gs, U; rtol=AUG_QR_RTOL) -> NamedTuple

`[gs, U]` orthonormalized with the leading `size(gs, 2)` columns left exactly as `gs`
(already orthonormal RSVD output). `span(gs)` is projected out of `U` by two passes of
classical Gram-Schmidt, the rest goes through a Householder QR, and columns with a
vanishing `|R_ii|` are dropped. Returns `(basis, num_uu_kept, num_uu_dropped)`.
"""
function augmented_basis(gs::AbstractMatrix, U::AbstractMatrix; rtol::Real=AUG_QR_RTOL)
    size(U, 1) == size(gs, 1) || throw(DimensionMismatch("the augmentation has $(size(U, 1)) rows but the g basis has $(size(gs, 1))"))
    k = size(U, 2)
    k == 0 && return (basis=gs, num_uu_kept=0, num_uu_dropped=0)
    U1 = copy(U)
    for _ in 1:2
        mul!(U1, gs, gs' * U1, -one(eltype(gs)), one(eltype(gs)))
    end
    U1, d = qr_thin_rdiag!(U1)
    keep = findall(>=(rtol * maximum(d)), d)
    basis = length(keep) == k ? hcat(gs, U1) : hcat(gs, U1[:, keep])
    return (basis=basis, num_uu_kept=length(keep), num_uu_dropped=k - length(keep))
end

# A LinearMap applied to the columns of B one at a time, so the operator only ever
# sees dense vectors on the device B lives on.
function opmat(A::LinearMap, B::AbstractMatrix)
    out = similar(B, promote_type(eltype(A), eltype(B)), size(A, 1), size(B, 2))
    @views for j in axes(B, 2)
        mul!(out[:, j], A, B[:, j])
    end
    return out
end

"""
    fit_augmentation(N_u, m, k_uu, budget_bytes) -> (k_uu, oversamples)

The `k_uu` and Asym(G⁰ᵤᵤ) sketch oversamples that fit `budget_bytes` of device memory
(`nothing`: no limit, as on a CPU run), with `m` the width of the g block (`m_kept`,
both signs of Γ). The dense augmented front end holds
`2·m_aug` tall columns, the augmentation's QR `m + 3k`, and the sketch
`RSVD_PEAK_FUDGE · 3(k + p)`. Oversamples are cut first (down to `UU_MIN_OVERSAMPLES`),
then `k_uu`, with a warning; below `K_UU_FLOOR` this errors.
"""
function fit_augmentation(N_u::Integer, m::Integer, k_uu::Integer, budget_bytes::Union{Nothing,Real})
    p_max = min(UU_OVERSAMPLES, N_u - k_uu)
    isnothing(budget_bytes) && return (k_uu=k_uu, oversamples=p_max)
    columns = floor(Int, budget_bytes / (16 * N_u))
    sketch_cols(p) = floor(Int, columns / (3 * RSVD_PEAK_FUDGE)) - p
    k = min(k_uu, fld(columns - 2m, 2), fld(columns - m, 3), sketch_cols(UU_MIN_OVERSAMPLES))
    k >= min(k_uu, K_UU_FLOOR) || error("fit_augmentation: at N_u = $(N_u), m = $(m) the device " *
        "fits only $(columns) tall columns, i.e. k_uu ≤ $(k), below the useful floor of $(K_UU_FLOOR). " *
        "Run this point on a larger card, or pass --k-uu 0")
    k < k_uu && @warn string(now()) * " [bounds::fit_augmentation] Clipping k_uu $(k_uu) → $(k) to fit the card; this point's bound is not comparable to a k_uu = $(k_uu) one"
    p = min(max(sketch_cols(0) - k, UU_MIN_OVERSAMPLES), min(UU_OVERSAMPLES, N_u - k))
    p < p_max && @warn string(now()) * " [bounds::fit_augmentation] Cutting the Asym(G⁰ᵤᵤ) oversamples to $(p) to fit the card"
    return (k_uu=k, oversamples=p)
end

"""
    uu_eigenbasis(compute_env, G⁰ᵤᵤ_asym, k; oversamples, power_iters) -> NamedTuple

The top `k` eigenpairs of Asym(G⁰ᵤᵤ), in memory, and the residuals `‖Av − λv‖/Λ₁` at
four of them: the evidence that `power_iters` converged the subspace.
"""
function uu_eigenbasis(compute_env::ComputeEnvironment, G⁰ᵤᵤ_asym, k::Int;
                       oversamples::Int=UU_OVERSAMPLES, power_iters::Int=UU_POWER_ITERS)
    sample_vec = use_gpu(compute_env) ? CuArray(zeros(ComplexF64, 0)) : zeros(ComplexF64, 0)
    @info string(now()) * " [bounds::uu_eigenbasis] Top $(k) eigenpairs of Asym(G⁰ᵤᵤ) ($(size(G⁰ᵤᵤ_asym))), $(oversamples) oversamples, $(power_iters) power iterations"
    out = reigen_hermitian(G⁰ᵤᵤ_asym, k; num_oversamples=oversamples,
                           num_power_iterations=power_iters, sample_vec=sample_vec)
    values = Array(out.values)
    idxs = unique(clamp.([1, cld(k, 4), cld(k, 2), k], 1, size(out.vectors, 2)))
    residuals = [norm(G⁰ᵤᵤ_asym * out.vectors[:, i] .- values[i] .* out.vectors[:, i]) / abs(values[1]) for i in idxs]
    @info string(now()) * " [bounds::uu_eigenbasis] Λ[1] = $(values[1]), Λ[end] = $(values[end]); residuals at $(idxs): $(residuals)"
    return (vectors=out.vectors, values=values, residuals=residuals)
end

"""
    load_bounds_inputs(compute_env, smr; gamma_rtol=DEFAULT_GAMMA_RTOL) -> NamedTuple

The RSVD output on scratch, ready for `bounds_from_spectrum`: `Γ` (every eigenvalue,
descending), `kept` (the indices with |Γ| ≥ gamma_rtol · Γ₁: a positive prefix and a
negative suffix), `V` (their `N_u × length(kept)` eigenvectors, on the device for a GPU
run), `Γrs`, `num_pos` (kept positives, the channel count) and `sorted_idxs`.

Past the operator's numerical rank the RSVD returns its own noise floor, whose
directions are not eigenvectors, so `gamma_rtol` keeps it out of the basis.
"""
function load_bounds_inputs(compute_env::ComputeEnvironment, smr::SMRSystem; gamma_rtol::Float64=DEFAULT_GAMMA_RTOL)
    0 <= gamma_rtol <= 1 || throw(ArgumentError("gamma_rtol must lie in [0, 1], got $gamma_rtol"))
    jld_path = rsvd_output_path(compute_env, smr)
    N_u = dof_length(sender_mesh(smr)) + dof_length(receiver_mesh(smr))
    jldopen(jld_path, "r") do jld
        haskey(jld, "UR_asym/num_pos") || error("$(jld_path) has no finished UR_asym; run generate_rsvd first")
        haskey(jld, "UR_asym/V") || haskey(jld, "UR_asym/V_file") || error("$(jld_path) holds only the " *
            "positive-Γ eigenvectors of an older run, and the bound needs all of them; rerun generate_rsvd")
        Γ = Array(jld["UR_asym/D"])
        sorted_idxs = sortperm(Γ, rev=true)
        sorted_idxs == eachindex(Γ) || error("UR_asym/D is not descending, so the stored vectors do not line up with it")
        stored = Int(jld["UR_asym/num_pos"])
        stored == count(>(0), Γ) || error("UR_asym/num_pos is $(stored) but $(count(>(0), Γ)) eigenvalues are positive")
        cut = gamma_rtol * Γ[1]
        num_pos = count(γ -> γ > 0 && γ >= cut, Γ)
        num_neg = count(γ -> γ < 0 && -γ >= cut, Γ)
        num_pos > 0 || error("UR_asym has no positive eigenvalue, so there are no channels to bound")
        ranges = (1:num_pos, (length(Γ) - num_neg + 1):length(Γ))
        kept = vcat(ranges...)
        @info string(now()) * " [bounds::load_bounds_inputs] gamma_rtol = $(gamma_rtol) keeps $(num_pos) of $(stored) positive and $(num_neg) of $(length(Γ) - stored) other Γ"

        V = Matrix{ComplexF64}(undef, N_u, length(kept))
        if haskey(jld, "UR_asym/V_file")
            path = joinpath(dirname(jld_path), jld["UR_asym/V_file"])
            store = Funicular.open_store(path, "r")
            try
                (store.N, store.k) == (N_u, length(Γ)) || error("$(path) holds $(store.N) × $(store.k), expected $(N_u) × $(length(Γ))")
                # Hyperslabs of at most 256 columns, so the staging copy stays small.
                col = 0
                for r in ranges, J in Iterators.partition(r, 256)
                    staged = Matrix{store.stored}(undef, N_u, length(J))
                    Funicular.read_panel!(Funicular.DiskHome(store, first(J):last(J)), staged)
                    copyto!(view(V, :, col .+ (1:length(J))), staged)
                    col += length(J)
                end
            finally
                Funicular.close_store!(store)
            end
        else
            Vall = jld["UR_asym/V"]
            size(Vall) == (N_u, length(Γ)) || error("UR_asym/V is $(size(Vall)), expected $(N_u) × $(length(Γ))")
            copyto!(V, view(Vall, :, kept))
        end
        return (Γ=Γ, kept=kept, V=use_gpu(compute_env) ? CuArray(V) : V,
                Γrs=Array(jld["RS/D"]), num_pos=num_pos, sorted_idxs=sorted_idxs)
    end
end

# U diag(d) Uᴴ, with `d` on U's device.
sandwich(U, d) = (U .* reshape(d, 1, :)) * U'

# The upper Cholesky factor U of a Hermitian positive definite R (R = UᴴU), and op(U)⁻¹X
# for `t` = 'N' or 'C'. Both go to (CU)LAPACK/(CU)BLAS directly: on a CuMatrix,
# LinearAlgebra's `cholesky` falls back to scalar loops and its triangular `\` is ambiguous
# between CUDA and GPUArrays.
function chol_upper(R::AbstractMatrix)
    U, info = R isa CuMatrix ? CUDA.CUSOLVER.potrf!('U', copy(R)) : LinearAlgebra.LAPACK.potrf!('U', copy(R))
    info == 0 || throw(PosDefException(Int(info)))
    return U
end
# The largest eigenpair of a Hermitian M. On the CPU, LAPACK's range driver skips the other
# eigenvectors; CUSOLVER has no such driver here, so the GPU takes the full decomposition.
function top_eigpair(M::AbstractMatrix)
    if M isa CuMatrix
        E = eigen(Hermitian(M))
        return Array(E.values)[end], E.vectors[:, end]
    end
    E = eigen(Hermitian(M), size(M, 1):size(M, 1))
    return E.values[1], E.vectors[:, 1]
end

upper_solve(U, X, t::Char) =
    (X isa CuArray ? CUDA.CUBLAS.trsm! : BLAS.trsm!)('L', 'U', t, 'N', one(eltype(X)), U, copy(X))

"""
    bounds_front_end(gs, basis, Γ_kept, G⁰ᵤᵤ_asym, sender_size) -> NamedTuple

The m_aug × m_aug objects the dual needs, on the basis' device: `W = basisᴴ gs`,
`A_P = W diag(Γ_kept) Wᴴ`, `G_P = basisᴴ (G⁰ᵤᵤ)ᵃ basis` (the only Green applies),
S = PₛᴴPₛ as `U diag(s) Uᴴ` with its factor `F` (FFᴴ = S, numerically zero s dropped),
`Γ` (Γ_kept on the device) and `sender_spanned` (rank S = N_s).
"""
function bounds_front_end(gs, basis, Γ_kept::Vector{Float64}, G⁰ᵤᵤ_asym, sender_size::Int)
    m_aug = size(basis, 2)
    @info string(now()) * " [bounds::bounds_front_end] Projecting into the basis of size $(m_aug)"
    W = basis' * gs
    Bₛ = view(basis, 1:sender_size, :)
    ES = eigen(Hermitian(Bₛ' * Bₛ)) # ascending
    s = clamp.(ES.values, 0, 1)
    first_kept = findfirst(>(m_aug * eps()), Array(s))
    isnothing(first_kept) && error("the basis has no component on the sender")
    # 16 columns at a time, so the N_u × m_aug image is never held.
    G_P = similar(basis, eltype(basis), m_aug, m_aug)
    for j0 in 1:16:m_aug
        J = j0:min(j0 + 15, m_aug)
        G_P[:, J] = basis' * opmat(G⁰ᵤᵤ_asym, view(basis, :, J))
    end
    Γ = similar(W, Float64, length(Γ_kept))
    copyto!(Γ, Γ_kept)
    return (U=ES.vectors, s=s, F=ES.vectors[:, first_kept:end] .* reshape(sqrt.(s[first_kept:end]), 1, :),
            A_P=sandwich(W, Γ), G_P=G_P, W=W, Γ=Γ, sender_spanned=m_aug - first_kept + 1 == sender_size)
end

"""
    tau_prep(front, τ) -> NamedTuple

The matrices of the dual that depend on τ only, shared by every n: `B = 1 + ζQ^½G_PQ^½`,
`Xq = Q^½W`, `FQmh = FᴴQ^-½` and `C₀ = FᴴQ⁻¹F`, with Q = Q(τ). `front` is
`bounds_front_end`'s output plus `ζ`.
"""
function tau_prep(front, τ::Real)
    τ < 1 || throw(ArgumentError("τ must be below 1, got $τ"))
    q = front.s .+ (1 .- front.s) ./ (1 - τ) # the eigenvalues of Q(τ), all > 0
    Qh = sandwich(front.U, sqrt.(q))
    B = front.ζ .* (Qh * front.G_P * Qh)
    view(B, diagind(B)) .+= 1
    return (τ=Float64(τ), B=B, Xq=Qh * front.W, FQmh=front.F' * sandwich(front.U, 1 ./ sqrt.(q)),
            C₀=front.F' * (sandwich(front.U, 1 ./ q) * front.F))
end

# Golden-section search of a unimodal `f` on (lo, hi), down to a bracket of width `tol`.
# Every evaluation is a valid bound, so `f` keeps the running minimum itself.
function golden_section(f, lo::Real, hi::Real, tol::Real)
    invφ = (sqrt(5.0) - 1) / 2
    x₁, x₂ = hi - invφ * (hi - lo), lo + invφ * (hi - lo)
    f₁, f₂ = f(x₁), f(x₂)
    while hi - lo > tol
        if f₁ <= f₂
            hi, x₂, f₂ = x₂, x₁, f₁
            x₁ = hi - invφ * (hi - lo)
            f₁ = f(x₁)
        else
            lo, x₁, f₁ = x₁, x₂, f₂
            x₂ = lo + invφ * (hi - lo)
            f₂ = f(x₂)
        end
    end
end

# Pattern search in τ from the previous channel's optimum: probe ±δ, step downhill while
# that improves, then golden section on the last bracket. Returns the number of steps.
function local_tau_search(dual_at, τ₀::Real, δ::Real, τ_min::Real, τ_max::Real)
    c = clamp(Float64(τ₀), τ_min, τ_max)
    fc = dual_at(c)
    for step in 0:TAU_MAX_STEPS
        l, r = max(c - δ, τ_min), min(c + δ, τ_max)
        fl = l < c ? dual_at(l) : Inf
        fr = r > c ? dual_at(r) : Inf
        if fc <= min(fl, fr)
            golden_section(dual_at, l, r, δ)
            return step
        end
        c, fc = fl < fr ? (l, fl) : (r, fr)
    end
    return TAU_MAX_STEPS + 1
end

"""
    dual_channels(num_pos) -> Vector{Int}

The channels the dual runs on: 1…DUAL_DENSE_CHANNELS, then each about DUAL_CHANNEL_RATIO
times the last, ending at `num_pos`.
"""
function dual_channels(num_pos::Integer)
    ns = collect(1:min(DUAL_DENSE_CHANNELS, num_pos))
    while last(ns) < num_pos
        push!(ns, min(num_pos, max(last(ns) + 1, round(Int, last(ns) * DUAL_CHANNEL_RATIO))))
    end
    return ns
end

"""
    dual_curve(front, prep, n) -> (μₙ, μ_lo, μ_hi), or `nothing` when no μₚ is known to be feasible

At `prep.τ` and index `n`, the function μₚ ↦ (μₙ(μₚ, τ), dμₙ/dlog μₚ), with
μₙ = μₚ σ₁(Πₛ(𝟙(τ) − H)⁻¹Πₛ), and the interval (μ_lo, μ_hi) of μₚ where 𝟙(τ) ≻ H. In the Q^½-whitened basis, 𝟙(τ) − H is
R(s) = 1 + ζQ^½(τA + G − sA⁽ⁿ⁾)Q^½ with s = 1/μₚ. At s = τ this is
R₀ = 1 + ζQ^½(τ(A − A⁽ⁿ⁾) + G)Q^½ ⪰ 1, since A − A⁽ⁿ⁾ is the top n−1 (positive)
components, so R₀ = LLᴴ whitens every τ ≥ 0. For τ < 0, R₀ may be indefinite, and `nothing`
is returned when it is or when the feasible s lie at or below 0. With L⁻¹ζQ^½A⁽ⁿ⁾Q^½L⁻ᴴ = V diag(ν) Vᴴ,

    R(s)⁻¹ = L⁻ᴴ V diag(1/(1 − (s − τ)ν)) Vᴴ L⁻¹,   feasible iff 1 − (s − τ)νₖ > 0 ∀k.

The negative Γ kept in A⁽ⁿ⁾ give negative ν, which bound s from below, so the feasible
μₚ are an interval rather than (ν_max, ∞). Sender directions outside span(Pₛ)
contribute the eigenvalue 1 unless `front.sender_spanned`.
"""
function dual_curve(front, prep, n::Int)
    ζ, τ = front.ζ, prep.τ
    R₀ = n > 1 ? prep.B .+ (ζ * τ) .* sandwich(prep.Xq[:, 1:n-1], front.Γ[1:n-1]) : prep.B
    U = try
        chol_upper(R₀)
    catch err
        # shortcut: skips τ < 0 where R₀ is indefinite but some R(s), s > 0, may be ≻ 0; search s for a whitening point if opt_taus pile up there.
        err isa PosDefException && τ < 0 || rethrow()
        return nothing
    end
    Xn = upper_solve(U, prep.Xq[:, n:end], 'C')
    E = eigen(Hermitian(ζ .* sandwich(Xn, front.Γ[n:end])))
    ν = E.values
    ν_min, ν_max = extrema(Array(ν))
    ν_max > 0 || error("dual_curve: Ñ has no positive eigenvalue at n = $n, so A⁽ⁿ⁾ has no positive part left")
    s_hi = τ + 1 / ν_max
    s_hi > 0 || return nothing # τ < 0 only: every feasible s is ≤ 0, i.e. no μₚ > 0
    s_lo = ν_min < 0 ? max(0.0, τ + 1 / ν_min) : 0.0
    Z = prep.FQmh * upper_solve(U, E.vectors, 'N')
    # With λ, v the top eigenpair of Z diag(1/d) Zᴴ − C₀, Hellmann–Feynman gives
    # dλ/ds = Σₖ |(Zᴴv)ₖ|² νₖ/dₖ², so the slope in log μₚ costs one extra product.
    function μₙ(μ_p)
        s = 1 / μ_p
        d = 1 .- (s - τ) .* ν
        minimum(Array(d)) > 0 || return (Inf, NaN) # outside (μ_lo, μ_hi), 𝟙(τ) − H is not ≻ 0: no bound
        λ, v = top_eigpair(sandwich(Z, 1 ./ d) .- prep.C₀)
        !front.sender_spanned && λ < 0 && return (μ_p, μ_p) # the leftover sender directions' 1 wins
        dλ_ds = sum(Array(abs2.(Z' * v) .* ν ./ d .^ 2))
        return (μ_p * (1 + λ), μ_p * (1 + λ - s * dλ_ds))
    end
    return μₙ, 1 / s_hi, s_lo > 0 ? 1 / s_lo : Inf
end

"""
    system_dual(front, prep, n; μ_start=nothing, log_tol=MU_P_LOG_TOL) -> (μ, μ_p, μ_lo)

`min over μₚ of μₙ(μₚ, τ)` at `prep.τ`. μₙ is convex in μₚ on its feasible interval
(μ_lo, μ_hi), so its slope in x = log μₚ changes sign once. From `μ_start` (the previous
channel's μₚ, if feasible) or 1/τ, the search walks downhill in doubling steps until the
slope changes sign, then closes the bracket by Illinois regula falsi on the slope, until
|dμₙ/dx| ≤ log_tol · μₙ or the bracket is narrower than log_tol. An evaluation outside the
feasible interval counts as a slope pointing back inside. Every evaluation is a valid
bound and the smallest is kept. If no sign change is found, golden section takes over.
"""
function system_dual(front, prep, n::Int; μ_start=nothing, log_tol::Real=MU_P_LOG_TOL)
    curve = dual_curve(front, prep, n)
    isnothing(curve) && return (μ=Inf, μ_p=NaN, μ_lo=NaN) # τ < 0 with no feasible μₚ: no bound
    μₙ, μ_lo, μ_hi = curve
    best = (μ=Inf, μ_p=NaN)
    function slope(x, inward)
        μ, g = μₙ(exp(x))
        μ < best.μ && (best = (μ=μ, μ_p=exp(x)))
        return isfinite(μ) ? (g, μ) : (inward, μ)
    end
    lo, hi = log(μ_lo), log(μ_hi)
    default = prep.τ > 0 ? 1 / prep.τ : 2μ_lo
    x = log(!isnothing(μ_start) && μ_lo < μ_start < μ_hi ? μ_start : default)
    g, μ = slope(x, NaN)
    isfinite(g) || return golden_fallback(front, prep, n, μₙ, μ_lo, μ_hi, best, log_tol)
    dir = g > 0 ? -1 : 1
    a = b = x
    ga = gb = g
    h = 0.25
    for _ in 1:60
        abs(g) <= log_tol * μ && return (μ=best.μ, μ_p=best.μ_p, μ_lo=μ_lo)
        y = dir > 0 ? min(x + h, (x + hi) / 2) : max(x - h, (x + lo) / 2) # never step past an edge
        gy, μ = slope(y, dir > 0 ? Inf : -Inf)
        if sign(gy) == dir # the slope changed sign between x and y
            (a, ga, b, gb) = dir > 0 ? (x, g, y, gy) : (y, gy, x, g)
            break
        end
        x, g, h = y, gy, 2h
    end
    ga < 0 < gb || return golden_fallback(front, prep, n, μₙ, μ_lo, μ_hi, best, log_tol)
    side = 0
    while b - a > log_tol
        t = isfinite(ga) && isfinite(gb) ? (a * gb - b * ga) / (gb - ga) : (a + b) / 2
        gt, μ = slope(t, NaN)
        (!isfinite(gt) || abs(gt) <= log_tol * μ) && break
        if gt < 0
            a, ga = t, gt
            side == -1 && (gb /= 2) # Illinois: halve the stale end so the bracket closes from both sides
            side = -1
        else
            b, gb = t, gt
            side == 1 && (ga /= 2)
            side = 1
        end
    end
    return (μ=best.μ, μ_p=best.μ_p, μ_lo=μ_lo)
end

# Golden section on log μₚ over the whole feasible interval: the slower search the slope
# search falls back to. μₙ ≥ μₚ unless the basis spans the sender, which caps the bracket.
function golden_fallback(front, prep, n, μₙ, μ_lo, μ_hi, best, log_tol)
    @warn string(now()) * " [bounds::system_dual] n=$n τ=$(prep.τ): no slope sign change found, golden section instead"
    f(x) = (μ = μₙ(exp(x))[1]; μ < best.μ && (best = (μ=μ, μ_p=exp(x))); μ)
    μ_start = prep.τ > 0 ? 1 / prep.τ : 2μ_lo
    f(log(μ_start))
    # shortcut: the 1e3 widening is a heuristic, not a proof; raise it if opt_mu_p sits at the cap.
    cap = front.sender_spanned ? 1e3 * max(best.μ, μ_start) : best.μ
    golden_section(f, log(μ_lo), log(min(μ_hi, cap)), log_tol)
    return (μ=best.μ, μ_p=best.μ_p, μ_lo=μ_lo)
end

"""
    build_front(compute_env, smr, V, Γ_kept, ζ, k_uu, G₀_uu) -> (front, augmentation, m_aug)

The augmented basis [g, U_uu] and `bounds_front_end` on it, with `ζ` attached.
"""
function build_front(compute_env::ComputeEnvironment, smr::SMRSystem, V, Γ_kept::Vector{Float64},
                     ζ::Real, k_uu::Int, G₀_uu)
    sender_size = dof_length(sender_mesh(smr))
    N_u = size(V, 1)
    m_kept = size(V, 2)
    G⁰ᵤᵤ_asym = isnothing(G₀_uu) ? asym_green_map(compute_env, smr, Design) : LinearMap(G₀_uu)
    # The augmentation, clamped to what the universe has left and to what the card fits.
    k_uu = min(k_uu, N_u - m_kept)
    augmentation = (k_uu=0, oversamples=0, num_uu_kept=0, num_uu_dropped=0,
                    uu_values=Float64[], uu_residuals=Float64[])
    basis = V
    if k_uu > 0
        budget = use_gpu(compute_env) ? device_budget_bytes() - gila_workspace_bytes(N_u) : nothing
        fit = fit_augmentation(N_u, m_kept, k_uu, budget)
        uu = uu_eigenbasis(compute_env, G⁰ᵤᵤ_asym, fit.k_uu; oversamples=fit.oversamples)
        aug = augmented_basis(V, uu.vectors)
        basis = aug.basis
        augmentation = (k_uu=fit.k_uu, oversamples=fit.oversamples, num_uu_kept=aug.num_uu_kept,
                        num_uu_dropped=aug.num_uu_dropped, uu_values=uu.values, uu_residuals=uu.residuals)
        uu = aug = nothing
        run_gc()
        use_gpu(compute_env) && CUDA.reclaim()
    end
    m_aug = size(basis, 2)
    @info string(now()) * " [bounds::build_front] Basis: m_aug = $(m_aug) ($(m_kept) g + $(augmentation.num_uu_kept) Asym(G⁰ᵤᵤ), $(augmentation.num_uu_dropped) dropped)"

    return (; bounds_front_end(V, basis, Γ_kept, G⁰ᵤᵤ_asym, sender_size)..., ζ=ζ), augmentation, m_aug
end

# Writes through a temporary file, so a job killed mid-write leaves the last checkpoint intact.
function save_atomic(path::AbstractString, pairs::Pair...)
    tmp = "$(path).tmp.$(getpid())"
    jldopen(tmp, "w") do f
        foreach(((k, v),) -> f[k] = v, pairs)
    end
    mv(tmp, path; force=true)
end

# The saved front of an earlier run on the same inputs, or `nothing`.
function load_front_checkpoint(path, key)
    (isnothing(path) || !isfile(path)) && return nothing
    saved = jldopen(f -> (key=f["key"], front=f["front"], augmentation=f["augmentation"], m_aug=f["m_aug"]), path)
    saved.key == key && return saved
    @warn string(now()) * " [bounds::load_front_checkpoint] $(path) was written for different inputs; rebuilding"
    return nothing
end

"""
    bounds_from_spectrum(compute_env, smr, Γ, V, Γrs; kept, k_uu, τs, τ_refine_tol, G₀_uu, checkpoint)

The σₙ(Pᵣₛ) bounds from a loaded Asym(G⁰ᵤᵣ) spectrum: `Γ` every eigenvalue (descending),
`V` the eigenvectors of `Γ[kept]`, column for column.

The dual runs on `channels` (default `dual_channels(num_pos)`). The first, and any channel
whose warm start goes wrong, minimises over μₚ at each τ of the grid `τs` (τ = 1 is
skipped) and refines by golden-section search in τ down to `τ_refine_tol`. The others
search locally from the previous channel's τ and μₚ (`local_tau_search`). `nothing` keeps
the grid minimum at every channel. Every evaluation is a valid bound, and the running
minimum is kept. A channel between samples takes the bound of the sample below it.

The result is `min(dual, two passivity bounds)` per index: `old` from `Γrs` and `new`
from `Γ`, with `which_bounds` saying which one won (1, 2, 3 = dual).

`checkpoint` (a path prefix, or `nothing`) makes the run resumable: the projected
matrices are saved once to `<checkpoint>.front.jld`, and the finished channels every
`CHECKPOINT_SECONDS` to `<checkpoint>.progress.jld`. A rerun on the same inputs skips
the augmentation and the projection, reuses the same basis, and resumes after the
last saved channel. The files are left for the caller to delete.
"""
function bounds_from_spectrum(compute_env::ComputeEnvironment, smr::SMRSystem,
                              Γ::AbstractVector, V, Γrs::AbstractVector;
                              kept::AbstractVector{<:Integer}=eachindex(Γ),
                              k_uu::Int=DEFAULT_K_UU,
                              τs::AbstractVector{<:Real}=range(-1.0, 1.0, length=9),
                              τ_refine_tol::Union{Nothing,Real}=0.05,
                              G₀_uu=nothing,
                              channels=nothing,
                              checkpoint::Union{Nothing,AbstractString}=nothing)
    isempty(τs) && throw(ArgumentError("τs must contain at least one grid point"))
    all(<=(1), τs) && issorted(τs) || throw(ArgumentError("τs must be sorted and ≤ 1, got $(τs)"))
    isnothing(τ_refine_tol) || τ_refine_tol > 0 || throw(ArgumentError("τ_refine_tol must be positive or nothing"))
    k_uu >= 0 || throw(ArgumentError("k_uu must be non-negative, got $k_uu"))
    Γ_kept = Float64.(Γ[kept])
    size(V, 2) == length(kept) || throw(ArgumentError("V has $(size(V, 2)) columns for $(length(kept)) kept Γ"))
    issorted(Γ_kept, rev=true) || throw(ArgumentError("the kept Γ must be descending"))
    num_pos = count(>(0), Γ_kept)
    num_pos >= 1 || throw(ArgumentError("no kept Γ is positive"))
    channels = isnothing(channels) ? dual_channels(num_pos) : collect(Int, channels)
    issorted(channels) && allunique(channels) && first(channels) == 1 && last(channels) <= num_pos ||
        throw(ArgumentError("channels must be increasing, start at 1 and stay ≤ $(num_pos)"))
    m_kept = length(kept)

    sender_size = dof_length(sender_mesh(smr))
    N_u = sender_size + dof_length(receiver_mesh(smr))
    size(V, 1) == N_u || error("the basis has $(size(V, 1)) rows but the universe [sender; receiver] has $(N_u)")
    χ = susceptibility(smr)
    ζ = abs(χ)^2 / imag(χ)
    @info string(now()) * " [bounds::bounds_from_spectrum] χ = $χ, ζ = $ζ, m = $(num_pos) channels, m_kept = $(m_kept), N_u = $(N_u)"

    front_path = isnothing(checkpoint) ? nothing : "$(checkpoint).front.jld"
    progress_path = isnothing(checkpoint) ? nothing : "$(checkpoint).progress.jld"
    key = (Γ_kept=Γ_kept, τs=collect(Float64, τs), τ_refine_tol=τ_refine_tol, k_uu=k_uu, ζ=ζ)
    saved = load_front_checkpoint(front_path, key)
    if isnothing(saved)
        isnothing(progress_path) || rm(progress_path; force=true)
        front, augmentation, m_aug = build_front(compute_env, smr, V, Γ_kept, ζ, k_uu, G₀_uu)
        isnothing(front_path) || save_atomic(front_path, "key" => key, "m_aug" => m_aug, "augmentation" => augmentation,
                                             "front" => map(x -> x isa AbstractArray ? Array(x) : x, front))
    else
        @info string(now()) * " [bounds::bounds_from_spectrum] Resuming from $(front_path)"
        front = map(x -> x isa AbstractArray && use_gpu(compute_env) ? CuArray(x) : x, saved.front)
        augmentation, m_aug = saved.augmentation, saved.m_aug
    end

    usable = [i for i in eachindex(τs) if τs[i] < 1]
    preps = Dict{Float64,Any}()
    prep_at(τ) = (length(preps) >= TAU_PREP_CACHE && empty!(preps); get!(() -> tau_prep(front, τ), preps, Float64(τ)))
    τ_max = 1 - TAU_EDGE

    bounds_dual_basis = fill(Inf, num_pos)
    bounds_dual_by_tau = fill(NaN, num_pos, length(τs))
    opt_taus = fill(NaN, num_pos)
    opt_mu_p = fill(NaN, num_pos)
    k_done = 0
    if !isnothing(saved) && isfile(progress_path)
        jldopen(progress_path) do f
            if haskey(f, "channels") && f["channels"] == channels
                k_done = f["k_done"]
                for (dst, k) in ((bounds_dual_basis, "bounds_dual_basis"), (bounds_dual_by_tau, "bounds_dual_by_tau"),
                                 (opt_taus, "opt_taus"), (opt_mu_p, "opt_mu_p"))
                    copyto!(dst, f[k])
                end
            end
        end
        @info string(now()) * " [bounds::bounds_from_spectrum] $(k_done) of $(length(channels)) channels already done"
    end
    last_save = time()
    for k in (k_done + 1):length(channels)
        n = channels[k]
        prev = k > 1 ? (μ=bounds_dual_basis[channels[k-1]]^2, τ=opt_taus[channels[k-1]], μ_p=opt_mu_p[channels[k-1]]) : nothing
        best = (μ=Inf, τ=NaN, μ_p=NaN)
        dual_at(τ) = try
            d = system_dual(front, prep_at(τ), n; μ_start=isnothing(prev) ? nothing : prev.μ_p)
            d.μ < best.μ && (best = (μ=d.μ, τ=Float64(τ), μ_p=d.μ_p))
            d.μ
        catch err
            @warn string(now()) * " [bounds::bounds_from_spectrum] [$n/$(num_pos)] τ=$(τ) failed; dropping it" exception = err
            Inf
        end
        function grid_sweep()
            grid_μ = [dual_at(τs[i]) for i in usable]
            bounds_dual_by_tau[n, usable] .= sqrt.(grid_μ)
            i = usable[argmin(grid_μ)]
            if !isnothing(τ_refine_tol) && length(τs) > 1
                golden_section(dual_at, Float64(τs[max(i - 1, firstindex(τs))]),
                               min(Float64(τs[min(i + 1, lastindex(τs))]), τ_max), τ_refine_tol)
            end
        end
        swept = isnothing(prev) || isnothing(τ_refine_tol) || length(τs) == 1
        if swept
            grid_sweep()
        else
            # shortcut: τ never goes below τs[1] (−1 by default); widen the grid if opt_taus sit at τs[1].
            walked = local_tau_search(dual_at, prev.τ, τ_refine_tol, first(τs), τ_max)
            # A long walk or a value well above the previous channel's says the warm start lost the optimum.
            if walked > TAU_MAX_STEPS || best.μ > (1 + TAU_RESWEEP_RTOL) * prev.μ
                grid_sweep()
                swept = true
            end
        end
        isfinite(best.μ) || error("every τ tried failed at n=$n")
        @info string(now()) * " [bounds::bounds_from_spectrum] [$n/$(num_pos)] bound $(sqrt(best.μ)) on σₙ(Pᵣₛ) at τ = $(best.τ), μₚ = $(best.μ_p)$(swept ? " (grid)" : "")"
        bounds_dual_basis[n] = sqrt(best.μ)
        opt_taus[n] = best.τ
        opt_mu_p[n] = best.μ_p
        if !isnothing(progress_path) && (time() - last_save > CHECKPOINT_SECONDS || k == length(channels))
            save_atomic(progress_path, "channels" => channels, "k_done" => k, "bounds_dual_basis" => bounds_dual_basis,
                        "bounds_dual_by_tau" => bounds_dual_by_tau, "opt_taus" => opt_taus, "opt_mu_p" => opt_mu_p)
            last_save = time()
        end
    end
    # σₙ ≤ σₘ for m < n, so a channel takes the smallest dual at or below it: this fills the
    # channels between samples and keeps the bound non-increasing.
    accumulate!(min, bounds_dual_basis, bounds_dual_basis)

    # Passivity bounds: from κ = ζ²Γrs² (old form) and κ̃ = ζΓ (new form, Eq. (19)).
    old_form(κ) = κ >= 1 ? 1.0 : sqrt(4κ) / (1 + κ)
    new_form(κ̃) = 2κ̃ >= 1 ? 1.0 : sqrt(4κ̃ * abs(1 - κ̃))
    old_analytical_bounds = old_form.(ζ^2 .* Γrs .^ 2)
    new_analytical_bounds = new_form.(ζ .* max.(Γ, 0))
    old_analytical_bounds[old_analytical_bounds .<= 0] .= NaN
    new_analytical_bounds[new_analytical_bounds .<= 0] .= NaN
    ks = 1:min(length(old_analytical_bounds), length(new_analytical_bounds), num_pos)
    candidates(k) = (old_analytical_bounds[k], new_analytical_bounds[k], bounds_dual_basis[k])
    # argmin returns a NaN if one is present; a NaN candidate is "no bound", so it must lose.
    which_bounds = [argmin(replace(collect(candidates(k)), NaN => Inf)) for k in ks]
    true_bounds = [candidates(k)[which_bounds[k]] for k in ks]

    return (num_pos=num_pos, m_kept=m_kept, bounds_dual_basis=bounds_dual_basis, dual_channels=channels,
            tau_grid=collect(Float64, τs), opt_taus=opt_taus, opt_mu_p=opt_mu_p, bounds_dual_by_tau=bounds_dual_by_tau,
            old_analytical_bounds=old_analytical_bounds, new_analytical_bounds=new_analytical_bounds,
            true_bounds=true_bounds, which_bounds=which_bounds, m_aug=m_aug, augmentation=augmentation)
end

bounds_path(compute_env::ComputeEnvironment, smr::SMRSystem) =
    joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")

function _compute_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem;
                            gamma_rtol::Float64=DEFAULT_GAMMA_RTOL, k_uu::Int=DEFAULT_K_UU)
    out_path = bounds_path(compute_env, smr)
    if isfile(out_path) && jldopen(jld -> haskey(jld, "true_bounds"), out_path, "r")
        @info string(now()) * " [bounds::compute_bounds] $(out_path) is already complete: skipping (delete it to recompute)"
        return nothing
    end
    inputs = load_bounds_inputs(compute_env, smr; gamma_rtol=gamma_rtol)
    checkpoint = "$(out_path).ckpt"
    result = bounds_from_spectrum(compute_env, smr, inputs.Γ, inputs.V, inputs.Γrs;
                                  kept=inputs.kept, k_uu=k_uu, checkpoint=checkpoint)

    # Through a temporary file, so a failed rerun never destroys a finished result.
    tmp_path = "$(out_path).tmp.$(getpid())"
    jldopen(tmp_path, "w") do jld
        jld["χ"] = susceptibility(smr)
        jld["Γ"] = Array(inputs.Γ)
        jld["Γrs"] = Array(inputs.Γrs)
        jld["ordering_idxs"] = inputs.sorted_idxs
        for key in (:bounds_dual_basis, :dual_channels, :tau_grid, :opt_taus, :opt_mu_p, :bounds_dual_by_tau,
                    :old_analytical_bounds, :new_analytical_bounds, :which_bounds, :true_bounds)
            jld[String(key)] = getfield(result, key)
        end
        aug = result.augmentation
        jld["augment/augmented"] = aug.k_uu > 0
        jld["augment/k_uu_requested"] = k_uu
        jld["augment/k_uu_effective"] = aug.k_uu
        jld["augment/m"] = result.num_pos
        jld["augment/m_kept"] = result.m_kept
        jld["augment/m_aug"] = result.m_aug
        jld["augment/num_uu_kept"] = aug.num_uu_kept
        jld["augment/num_uu_dropped"] = aug.num_uu_dropped
        jld["augment/uu_oversamples"] = aug.oversamples
        jld["augment/uu_values"] = aug.uu_values
        jld["augment/uu_residuals"] = aug.uu_residuals
    end
    mv(tmp_path, out_path; force=true)
    rm("$(checkpoint).front.jld"; force=true)
    rm("$(checkpoint).progress.jld"; force=true)
    @info string(now()) * " [bounds::compute_bounds] Saved $(out_path)"
    return result
end

function compute_bounds()
    args = parse_cli()
    select_device(args.compute_env)
    isnothing(mediator(args.smr)) ||
        error("bounds for a system with a mediator are not implemented yet; generate_green and generate_rsvd support it")
    _compute_bounds_sr(args.compute_env, args.smr; gamma_rtol=args.gamma_rtol, k_uu=args.k_uu)
end
