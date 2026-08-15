using GilaElectromagnetics
using CUDA
using Plots
using Dates
using LinearAlgebra
using Roots
using Optim
using LinearMaps
using Base.Threads
using Plots
using KrylovKit
using NLopt
using Random
# using .Projectors
using MatrixFreeRandomizedLinearAlgebra
import Funicular

"""
    bracket_root(f, λs, b; btol=0.0, maxsteps=200)

Find a bracketing interval `((left, f(left)), (right, f(right)))` for

    f(α) = ∑ⱼ |bⱼ|² (α - 2λⱼ) / (α - λⱼ)²

with `f(left) < 0 < f(right)`, both endpoints finite. `f` has a pole at the
largest `λⱼ` with `bⱼ ≠ 0`, where it tends to `-∞` (for a positive pole), and
tends to `0⁺` as `α → ∞`, so such an interval exists whenever that pole is
positive.
"""
function bracket_root(f::Function, λs::AbstractVector{<:Real}, b::AbstractVector;
                      btol::Real=zero(real(eltype(b))), maxsteps::Int=200)
    num_bad = count(!isfinite, λs)
    num_bad == 0 || throw(ArgumentError(
        "bracket_root: $num_bad/$(length(λs)) of the λⱼ are non-finite, which makes " *
        "f(α) NaN at every α — the (α - 2λⱼ)/(α - λⱼ)² factor is Inf/Inf, so the " *
        "term is 0 * NaN even where bⱼ = 0"))
    pole_idxs = findall(j -> abs2(b[j]) > btol, eachindex(λs))
    isempty(pole_idxs) && throw(ArgumentError(
        "bracket_root: every |bⱼ|² is <= btol = $btol, so f ≡ 0"))
    pole = maximum(@view λs[pole_idxs])

    # Left endpoint: f → -∞ as α → pole⁺ (for pole > 0) and f → 0⁺ as α → ∞, so
    # start an offset of order |pole| above the pole and halve until f is finite
    # and negative. Halving rather than growing takes the larges* such offset,
    # which keeps the endpoint away from the pole where f is stiffest. Starting
    # relative to |pole| bounds the number of halvings needed by the ~52 bits of
    # mantissa regardless of how large or small the pole is.
    T = typeof(pole)
    left = nothing
    δ = iszero(pole) ? one(T) : abs(pole)
    α = pole + δ # declared out here so the error below can report the last probe
    for _ in 1:maxsteps
        α = pole + δ
        fα = f(α)
        if isfinite(fα) && fα < zero(fα)
            left = (α, fα)
            break
        end
        α == pole && break # δ has underflowed against pole; halving cannot help
        δ /= 2
    end
    isnothing(left) && throw(ArgumentError(
        "bracket_root: no representable α > λmax = $pole with f(α) < 0 " *
        "(last probe f($α) = $(f(α))). Either λmax ≤ 0, in which case the dual " *
        "has no interior stationary point on (λmax, ∞) and its infimum is 0 as " *
        "α → 0, or the sign change next to λmax cannot be resolved in $T: the " *
        "|bⱼ|² at λmax is negligible against the others, or (α - λmax)² " *
        "under/overflows (eps(λmax) = $(eps(pole)))"))

    # Right endpoint: f → 0⁺ as α → ∞, so walk out until f is finite and positive.
    right = nothing
    step = max(abs(left[1]), one(T))
    for _ in 1:maxsteps
        α = left[1] + step
        fα = f(α)
        if isfinite(fα) && fα > zero(fα)
            right = (α, fα)
            break
        end
        step *= 2
    end
    isnothing(right) && throw(ArgumentError(
        "bracket_root: no α > $(left[1]) with f(α) > 0 after $maxsteps steps"))

    return left, right
end

function qthin!(A::AbstractMatrix)
    F = qr!(A)
    n = min(size(A)...)
    A .= F.Q[:, 1:n]
    return A
end

function qthin!(A::CuMatrix)
    n = min(size(A)...)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    CUDA.CUSOLVER.orgqr!(A, τ)
    return A
end

"""
    psd_pencil_whitener(C; rtol=size(C, 1) * eps(real(eltype(C))))

Eigendecompose the constraint matrix `C` of the generalized problem `Bv = λCv`
and return the pieces needed to solve it on `C`'s numerical range:

- `whitener`: `W = U₊ diag(μ₊)^(-1/2)` over the eigenpairs above the rank
  tolerance, so that `W' C W = 1`. Solving `(W'BW)y = λy` and taking `v = Wy`
  yields the pencil's eigenpairs with `V' C V = 1` which is the normalization
  the dual's `∑ⱼ |bⱼ|²/(α - λⱼ)` resolvent expansion assumes.
- `nullspace`: an orthonormal basis `N` of the numerical null space, for the
  caller to verify that whatever it drops with it is genuinely absent.
- `values`, `tol`, `rank`: the full ascending spectrum of `C` (always a host
  vector) and where it was cut.

Every member of the constraint family `C(τ) = ζ⁻¹(Πₛ + (1−τ)Πᵣ) + τ(−G⁰ᵤᵣ)ᵃ₊ +
(G⁰ᵤᵤ)ᵃ`, `τ ∈ [0, 1]`, is a sum of positive semi-definite terms, so it is never
indefinite in exact arithmetic. An eigenvalue below `-tol` therefore indicates a
wrong sign somewhere upstream, not roundoff, and is reported as an error.

The work stays in `C`'s array space: on the host `eigen` goes through LAPACK,
which raises on a failed factorization, and on the device it goes through
CUSOLVER's `heevd`, which instead silently returns `NaN`/`Inf` eigenvalues.
The explicit non-finite check below covers that case, and the `whitener` and
`nullspace` come back on the same device as `C` so the per-index pencil solves
never round-trip the `m × m` matrices through the host.
"""
function psd_pencil_whitener(C::AbstractMatrix;
                             rtol::Real=size(C, 1) * eps(real(float(eltype(C)))))
    F = eigen(Hermitian(C)) # ascending; LAPACK on the host, heevd on the device
    μ = Array(F.values) # small, and the cut/tol logic is scalar host work
    num_bad = count(!isfinite, μ)
    num_bad == 0 || error("psd_pencil_whitener: eigendecomposition returned " *
        "$num_bad/$(length(μ)) non-finite eigenvalues; CUSOLVER's heevd reports " *
        "failure this way instead of throwing, so the factorization did not converge")
    μmax = maximum(μ)
    μmax > zero(μmax) || error("psd_pencil_whitener: C has no positive eigenvalue " *
        "(largest is $μmax), so the constraint bounds nothing")
    tol = rtol * μmax
    μmin = minimum(μ)
    μmin >= -tol || error("psd_pencil_whitener: C has eigenvalue $μmin, well below the " *
        "roundoff floor of -$tol. C(τ) is a sum of positive semi-definite terms " *
        "(ζ⁻¹Πₛ, (1−τ)ζ⁻¹Πᵣ, τ(−G⁰ᵤᵣ)ᵃ₊, (G⁰ᵤᵤ)ᵃ), so a negative eigenvalue means " *
        "one of them has the wrong sign")
    # μ is ascending, so the kept eigenpairs are a contiguous tail — which also
    # means device indexing below is a plain range, not a gather.
    num_null = searchsortedlast(μ, tol)
    kept = (num_null + 1):length(μ)
    inv_sqrt = similar(F.values, length(kept)) # in C's array space
    copyto!(inv_sqrt, 1 ./ sqrt.(μ[kept]))
    W = F.vectors[:, kept] .* inv_sqrt'
    return (whitener=W, nullspace=F.vectors[:, 1:num_null], values=μ, tol=tol,
            rank=length(kept))
end

"""
    diag_pencil_eigen(d, W, N; btol=1e-8)

Solve `diag(d) v = λ C v` for a diagonal, positive semi-definite `B = diag(d)`,
given the `whitener` and `nullspace` of `psd_pencil_whitener(C)`. Returns
`(values, vectors)` with `vectors' C vectors = 1`. `d` is a host vector; the
dense work happens in `W`'s array space (CUSOLVER/CUBLAS when `W` lives on the
device), `values` come back on the host and `vectors` stay with `W`.

Errors unless `B` is negligible on the numerical null space of `C`, which is the
condition under which discarding that null space is lossless. Both `B` and `C`
are positive semi-definite, so a null direction `v` of `C` splits two ways: if
`v'Bv ≈ 0` then `Bv ≈ 0` as well (for positive semi-definite `B`, `v'Bv = 0`
implies `Bv = 0`), the cross terms vanish too, and dropping `v` changes nothing;
whereas if `v'Bv > 0` the constraint genuinely fails to bound that direction, the
program is unbounded. Silently projecting the second case away would report a
finite bound that the program does not support.
"""
function diag_pencil_eigen(d::AbstractVector{<:Real}, W::AbstractMatrix,
                           N::AbstractMatrix; btol::Real=1e-8)
    all(≥(zero(eltype(d))), d) || throw(ArgumentError(
        "diag_pencil_eigen: B = diag(d) must be positive semi-definite, got " *
        "minimum(d) = $(minimum(d)) (is ζ = |χ|²/ℑχ negative?)"))
    dmax = maximum(d)
    sqrt_d = similar(W, real(eltype(W)), length(d)) # sqrt.(d) in W's array space
    copyto!(sqrt_d, sqrt.(d))
    if size(N, 2) > 0 && dmax > zero(dmax)
        # Column j of this is N[:,j]' * B * N[:,j]; the compression of a positive
        # semi-definite B is positive semi-definite, so its largest diagonal entry
        # also bounds its largest off-diagonal one.
        worst = maximum(sum(abs2, sqrt_d .* N; dims=1))
        worst <= btol * dmax || error("diag_pencil_eigen: B is not negligible on the " *
            "numerical null space of C: max_j N[:,j]'B N[:,j] = $worst vs " *
            "btol * max(d) = $(btol * dmax) over $(size(N, 2)) null direction(s). " *
            "The constraint does not bound those directions, so the true bound for " *
            "this index is +∞ rather than anything this program can report")
    end
    Wd = sqrt_d .* W # Wd' * Wd == W' * diag(d) * W, positive semi-definite by construction
    F = eigen(Hermitian(Wd' * Wd))
    Λ = Array(F.values)
    num_bad = count(!isfinite, Λ)
    num_bad == 0 || error("diag_pencil_eigen: eigendecomposition returned " *
        "$num_bad/$(length(Λ)) non-finite eigenvalues; CUSOLVER's heevd reports " *
        "failure this way instead of throwing, so the factorization did not converge")
    return Λ, W * F.vectors
end

"""
    pencil_probe_duals(pencil, B_diag, ss_basis, n, num_pos; τ=NaN)

Solve the per-probe dual problems for index `n` on one eigendecomposed
constraint pencil (the output of `psd_pencil_whitener`). For each probe
`k ∈ n:num_pos` this finds the stationary dual multiplier `αₖ` by bracketed
root finding and evaluates the dual value `αₖ²/4 ∑ⱼ |bⱼ|²/(αₖ - λⱼ)`, with
`b = V'sₖ` in the pencil's `C`-orthonormal eigenbasis.

Returns `(ks, alphas, duals)`. The bound contribution of this pencil is
`maximum(duals)`. The per-probe records are kept so that `verify_bounds` can
use the same probes in the full space, seeding each full-space
evaluation with the multiplier found here.
"""
function pencil_probe_duals(pencil, B_diag::AbstractVector{<:Real},
                            ss_basis::AbstractMatrix, n::Int, num_pos::Int;
                            τ::Real=NaN)
    Λ_basis, V_basis = diag_pencil_eigen(B_diag, pencil.whitener, pencil.nullspace)

    ks = collect(n:num_pos)
    alphas = zeros(Float64, length(ks))
    duals = zeros(Float64, length(ks))
    for (i, k) in enumerate(ks)
        sₖ_basis = view(ss_basis, :, k)
        if size(pencil.nullspace, 2) > 0
            sₖ_null = norm(pencil.nullspace' * sₖ_basis)
            sₖ_null <= 1e-8 * norm(sₖ_basis) || error("probe k=$k at n=$n, " *
                "τ=$τ has ‖N'sₖ‖ = $sₖ_null of ‖sₖ‖ = $(norm(sₖ_basis)) " *
                "in the numerical null space of C(τ), so the bound is +∞ ")
        end
        b_basis = Array(V_basis' * sₖ_basis)

        fₖ_basis(α) = sum(abs2(bⱼ) * (α - 2λⱼ)/(α - λⱼ)^2 for (bⱼ, λⱼ) in zip(b_basis, Λ_basis))
        ((left, f_left), (right, f_right)) = bracket_root(fₖ_basis, Λ_basis, b_basis)

        αₖ = find_zero(fₖ_basis, (left, right), Roots.Brent())
        alphas[i] = αₖ
        duals[i] = αₖ^2/4 * sum(abs2(bⱼ) / (αₖ - λⱼ) for (bⱼ, λⱼ) in zip(b_basis, Λ_basis))
    end
    return (ks=ks, alphas=alphas, duals=duals)
end

similar_fill(v::AbstractArray{T}, fill_val::T) where T = fill!(similar(v), fill_val)
similar_fill(v::AbstractArray{T}, dims::NTuple{N, Int}, fill_val::T) where {N, T} = fill!(similar(v, dims), fill_val)
Base.:\(::Nothing, x::AbstractArray) = (x, 0)
function lmul_mvp!(y::AbstractVector, ::Nothing, x::AbstractVector)
	copyto!(y, x)
	return y, 0
end

function bicgstab_gpu!(x::AbstractVector, op, b::AbstractVector; preconditioner=nothing, max_iter::Int=size(op, 2), atol::Real=zero(real(eltype(b))), rtol::Real=sqrt(eps(real(eltype(b)))), verbose::Bool=false, initial_zero::Bool=false, print_every::Int=1)
    T = eltype(b)
    mvp = 0
    if initial_zero
        fill!(x, zero(T))
        residual = deepcopy(b)
    else
        residual = b - op * x
        mvp += 1
    end
    atol = max(atol, rtol * norm(residual))
    residual_shadow = deepcopy(residual)
    p = deepcopy(residual)
    v = similar_fill(b, zero(T))
    ρ_prev = one(T); α = one(T); ω = one(T)

    for num_iter in 1:max_iter
        norm(residual) < atol && return x, mvp
        ρ = dot(residual_shadow, residual)
        if num_iter > 1
            β = (ρ / ρ_prev) * (α / ω)
            @. p = residual + β * (p - ω * v)
        end
        p̂, k = preconditioner \ p; mvp += k
        v = op * p̂; mvp += 1
        α = ρ / dot(residual_shadow, v)
        @. residual = residual - α * v
        if norm(residual) < atol
            @. x = x + α * p̂
            return x, mvp
        end
        ŝ, k = preconditioner \ residual; mvp += k
        t = op * ŝ; mvp += 1
        ω = dot(t, residual) / dot(t, t)
        @. x = x + α * p̂ + ω * ŝ
        @. residual = residual - ω * t
        ρ_prev = ρ
        # verbose && println(num_iter, " ", norm(residual), " > ", atol, " (mvp: ", mvp, ")")
        verbose && if num_iter % print_every == 0
            println(num_iter, " ", norm(residual), " > ", atol, " (mvp: ", mvp, ")")
        end
    end
    throw("BiCGStab did not converge after $max_iter iterations.")
end

function bicgstab_gpu(op, b::AbstractVector; preconditioner=nothing, max_iter::Int=length(b), atol::Real=zero(real(eltype(b))), rtol::Real=sqrt(eps(real(eltype(b)))), verbose::Bool=false)
	x = similar_fill(b, zero(eltype(b)))
	return bicgstab_gpu!(x, op, b; preconditioner=preconditioner, max_iter=max_iter, atol=atol, rtol=rtol, verbose=verbose, initial_zero=true)
end

function projected_operators(G₀_uu::AbstractGlaOpr, smr::SMRSystem, env::ComputeEnvironment)
    s = sender(smr)
    r = receiver(smr)
    # G₀_uu.mem.srcVol == G₀_uu.mem.trgVol || error("G₀_uu is not a self operator")
    # union_volume = G₀_uu.mem.srcVol # srcVol == trgVol
    # if union(s, r) != union_volume
    #     @error "union_volume should be union(s, r) but it is not"
    # end
    # sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
    # receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume

    # Zero out the gap between sender and receiver by creating a projector that keeps the sender and receiver regions but zeros out everything else in the union volume (including the gap between s and r)
    # disjoint_union_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    # if use_gpu(env)
    #     disjoint_union_indicator = CuArray(disjoint_union_indicator)
    # end
    # fill!(view(disjoint_union_indicator, sender_mask..., :), one(eltype(disjoint_union_indicator)))
    # fill!(view(disjoint_union_indicator, receiver_mask..., :), one(eltype(disjoint_union_indicator)))
    # disjoint_union_projector_action!(w, v) = begin
    #     w .= vec(disjoint_union_indicator .* reshape(v, size(disjoint_union_indicator)))
    #     return w
    # end
    # u_projector = LinearMap{ComplexF64}(disjoint_union_projector_action!, disjoint_union_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    # sender_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    # if use_gpu(env)
    #     sender_indicator = CuArray(sender_indicator)
    # end
    # fill!(view(sender_indicator, sender_mask..., :), one(eltype(sender_indicator)))
    # s_projector_action!(w, v) = begin
    #     w .= vec(sender_indicator .* reshape(v, size(sender_indicator)))
    #     return w
    # end
    # s_projector = LinearMap{ComplexF64}(s_projector_action!, s_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    sender_size = prod(s.cel)*3
    receiver_size = prod(r.cel)*3
    sender_projector_action!(s_included_in_u::AbstractVector{ComplexF64}, u::AbstractVector{ComplexF64}) = begin
        # our convention is [sender; receiver]
        fill!(s_included_in_u, zero(eltype(s_included_in_u)))
        copyto!(view(s_included_in_u, 1:sender_size), view(u, 1:sender_size))
        return s_included_in_u
    end
    s_projector = LinearMap{ComplexF64}(sender_projector_action!, sender_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    # receiver_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    # if use_gpu(env)
    #     receiver_indicator = CuArray(receiver_indicator)
    # end
    # fill!(view(receiver_indicator, receiver_mask..., :), one(eltype(receiver_indicator)))
    # r_projector_action!(w, v) = begin
    #     w .= vec(receiver_indicator .* reshape(v, size(receiver_indicator)))
    #     return w
    # end
    # r_projector = LinearMap{ComplexF64}(r_projector_action!, r_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)
    #
    # G₀ = LinearMap(G₀_uu)
    # return r_projector, s_projector, u_projector, (u_projector * G₀ * u_projector)

    return s_projector
end

opmat(A::LinearMap, b::AbstractVector) = A*b

function opmat(A::LinearMap, B::AbstractMatrix)
    T = promote_type(eltype(A), eltype(B))
    out = similar(B, T, size(A, 1), size(B, 2))
    tmp = similar(out, T, size(A, 1))
    @views for j in axes(B, 2)
        mul!(tmp, A, B[:, j])
        out[:, j] .= tmp
    end
    return out
end

function opmat(A::LinearMap, vecs::AbstractVector{<:AbstractVector})
    T = promote_type(eltype(A), eltype(vecs))
    out = similar(vecs, T, size(A, 1), length(vecs))
    tmp = similar(out, T, size(A, 1))
    @views for j in axes(vecs, 1)
        mul!(tmp, A, vecs[j])
        out[:, j] .= tmp
    end
    return out
end

function read_array(jld, key::AbstractString, use_gpu::Bool)
    @info string(now()) * " [bounds_bargaining::read_array] Reading array for key '$key' from JLD file"
    if haskey(jld, key)
        arr = jld[key]
        if use_gpu
            @info string(now()) * " [bounds_bargaining::read_array] Moving array for key '$key' to GPU"
            arr = CuArray(arr)
        end
        return arr
    else
        error("Key $key not found in JLD file")
    end
end

function _compute_bounds_smr(::ComputeEnvironment, ::SMRSystem, ::RSVDParams)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_smr] Computing bounds for SMR system"
    throw("Not implemented yet")
end

# Path selection for the N_u-scale front end (FUNICULAR_PLAN.md, workstream C).
#
# The pencil stage is m × m and is left alone. What can outgrow the device is the
# handful of N_u × m matrices in front of it: in the in-memory path, the basis,
# the `ss` probes, and the working matrix `opmat(C, basis)` builds. Those are the
# same three the cost model bills the panel path's host tier for. Above the
# device budget they become `PanelMatrix` objects streamed through the device one
# column panel at a time.
#
# The predicate mirrors `use_panel_path` in rsvd.jl, fudge factor included, so
# that the cost model and the run agree on which regime a job is in.

# basis + ss + one working matrix, the peak of the in-memory front end.
const BOUNDS_FRONT_END_MATRICES = 3

"""
    bounds_footprint_bytes(N_u, m) -> Int

Device bytes the in-memory bounds front end wants: the
`$(BOUNDS_FRONT_END_MATRICES)` `N_u × m` ComplexF64 matrices it holds at its
peak (the basis, the `ss` probes, and `opmat(C, basis)`'s destination). The
`m × m` pencil objects are not counted: they stay dense on the device in both
regimes.
"""
bounds_footprint_bytes(N_u::Integer, m::Integer) =
    BOUNDS_FRONT_END_MATRICES * Int(N_u) * Int(m) * 16

"""
    use_panel_bounds(N_u, m, compute_env) -> Bool

Whether the bounds front end has outgrown the device, in which case the tall
matrices go to Funicular's panel storage. Same shape and same `RSVD_PEAK_FUDGE`
as [`use_panel_path`](@ref), so the two stages of a job agree on the regime.
"""
function use_panel_bounds(N_u::Integer, m::Integer, compute_env::ComputeEnvironment)
    use_gpu(compute_env) || return false
    return RSVD_PEAK_FUDGE * bounds_footprint_bytes(N_u, m) > device_budget_bytes()
end

# A plan that only exists to read an h5 panel matrix back into one host array.
# Nothing is swept through it, so the "device" budget is only there for
# `resolve_panel_width`'s arithmetic (2 staging buffers of one panel each) and
# never allocates. The host budget has to cover the panels themselves, which
# `Matrix` then copies into the array it returns.
function _dense_read_plan(N_u::Integer, m::Integer)
    bytes = Int(N_u) * Int(m) * 16
    return Funicular.ResidencyPlan(backend=Funicular.CPUBackend(),
                                   device_budget=max(4 * bytes, 2^20),
                                   host_budget=max(bytes + (bytes >> 2), 2^20))
end

# Which of the three save formats the RSVD job left behind, in priority order:
#
#   1. `vectors_file`: the basename of an h5 Funicular.save'd `PanelMatrix` next
#      to the JLD (rsvd.jl's panel branch).
#   2. `V_pos`: the N_u × m block inline in the JLD (dense/in-memory and
#      dense-exact branches).
#   3. `V`: the legacy full N_u × k basis. We no longer write it, but existing
#      sweeps on disk still carry it, and the leading `num_pos` columns of the
#      descending ordering are exactly the block the other two formats store.
function _ur_asym_vectors_source(jld, vectors_path::String)
    if haskey(jld, "UR_asym/vectors_file")
        path = joinpath(dirname(vectors_path), jld["UR_asym/vectors_file"])
        isfile(path) || error("UR_asym/vectors_file names $(basename(path)) but there " *
            "is no such file next to the JLD; the RSVD job wrote its eigenvalues but " *
            "not its vectors, so it has to be rerun")
        return (:h5, path)
    end
    haskey(jld, "UR_asym/V_pos") && return (:v_pos, "")
    haskey(jld, "UR_asym/V") && return (:legacy, "")
    error("the JLD has none of UR_asym/vectors_file, UR_asym/V_pos or the legacy " *
          "UR_asym/V, so there is no Asym(G⁰ᵤᵣ) basis to run the bounds on")
end

# The positive-Γ block as a host `Matrix{ComplexF64}`. `cols` are the columns of
# the *sorted* basis that are wanted, that is, `sorted_idxs[1:num_pos]`. The
# positives-only formats already hold that block, so for them the ordering has to
# be the identity: a file whose `D` is not descending has vectors we cannot
# reorder, and we error rather than pair the wrong vector with an eigenvalue.
function _read_ur_asym_dense(jld, source::Symbol, path::String,
                             cols::AbstractVector{Int}, N_u::Integer)
    m = length(cols)
    if source === :h5
        _assert_positive_prefix(cols, "UR_asym/vectors_file")
        @info string(now()) * " [bounds_bargaining::_read_ur_asym_dense] Reading the $(N_u) × $(m) positive block from $(path)"
        plan = _dense_read_plan(N_u, m)
        pm = Funicular.load(Funicular.PanelMatrix, path; plan=plan, readonly=true)
        try
            return Matrix(pm; max_bytes=plan.host_budget)
        finally
            Funicular.free!(pm)
        end
    elseif source === :v_pos
        _assert_positive_prefix(cols, "UR_asym/V_pos")
        @info string(now()) * " [bounds_bargaining::_read_ur_asym_dense] Reading the $(N_u) × $(m) positive block from UR_asym/V_pos"
        return Matrix{ComplexF64}(jld["UR_asym/V_pos"])
    end
    @info string(now()) * " [bounds_bargaining::_read_ur_asym_dense] Reading the legacy full UR_asym/V and taking its leading $(m) sorted columns"
    return Matrix{ComplexF64}(view(jld["UR_asym/V"], :, cols))
end

# The same block as an `N_u × m` `PanelMatrix` on the run's plan. The h5 is opened
# as the matrix's cold tier and its panels stream up as they are swept, so nothing
# dense of that size is ever built. The other two formats have to come through
# host memory once, since that is how they are stored.
function _read_ur_asym_panel(jld, source::Symbol, path::String,
                             cols::AbstractVector{Int}, N_u::Integer, plan)
    if source === :h5
        _assert_positive_prefix(cols, "UR_asym/vectors_file")
        @info string(now()) * " [bounds_bargaining::_read_ur_asym_panel] Opening $(path) as a $(N_u) × $(length(cols)) panel matrix"
        return Funicular.load(Funicular.PanelMatrix, path; plan=plan, readonly=true)
    end
    @info string(now()) * " [bounds_bargaining::_read_ur_asym_panel] The JLD holds the basis densely; cutting it into panels"
    dense = _read_ur_asym_dense(jld, source, path, cols, N_u)
    pm = Funicular.PanelMatrix(dense; plan=plan)
    dense = nothing
    run_gc()
    return pm
end

function _assert_positive_prefix(cols::AbstractVector{Int}, key::AbstractString)
    cols == collect(1:length(cols)) || error("$key stores only the positive-Γ block, " *
        "in the order the RSVD wrote it, but sorting UR_asym/D descending does not " *
        "leave that block as the leading $(length(cols)) columns (wanted columns " *
        "$(first(cols)):… ). The values and the vectors in this file do not line up")
    return nothing
end

"""
    load_bounds_inputs(compute_env, smr; kwargs...)

Reads the `Asym(G⁰ᵤᵣ)` spectrum the RSVD job left in scratch and returns it
ready for [`bounds_from_spectrum`](@ref): `Γ` (every eigenvalue, host, sorted
descending), `Vur_asym` (the `N_u × num_pos` positive block), `Γrs`,
`sorted_idxs` (the descending permutation, saved as `ordering_idxs`), `num_pos`
and the `ResidencyPlan` the basis was built from, if any.

`Vur_asym` comes back as a `Funicular.PanelMatrix` when the front end has
outgrown the device (see [`use_panel_bounds`](@ref)), and otherwise as a plain
matrix, on the device if the run is a GPU run. Only the positive block is staged,
since nothing reads the negative-Γ half of the legacy `V`.

# Keyword arguments
- `plan_override`: use this `ResidencyPlan` and take the panel path regardless
  of the predicate. Mirrors `_save_ur_asym`'s kwarg of the same name, and is how
  a CPU test exercises the panel front end.
- `panel_mode`: `true`/`false` to force the choice, `nothing` (the default) to
  let [`use_panel_bounds`](@ref) make it. `verify_bounds` passes `false`: its
  full-space math wants one dense basis, not panels.
- `to_device`: whether a dense basis is moved to the GPU. Defaults to
  `use_gpu(compute_env)`, and is ignored on the panel path.
"""
function load_bounds_inputs(compute_env::ComputeEnvironment, smr::SMRSystem;
                            plan_override=nothing,
                            panel_mode::Union{Nothing,Bool}=nothing,
                            to_device::Bool=use_gpu(compute_env))
    jld_in_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_in = jldopen(jld_in_path, "r")
    try
        haskey(jld_in, "UR_asym/D") || error("Key UR_asym/D not found in $(jld_in_path)")
        Γ_raw = Array(jld_in["UR_asym/D"])
        # NO MORE SIGN TYPO (this is fixed in rsvd.jl)
        # Γ_raw .*= -one(eltype(Γ_raw)) # Sign typo in the original notes

        # Sort the eigenvalues descending; the vectors are sliced to match below.
        sorted_idxs = Array(sortperm(Γ_raw, rev=true))
        Γ = Γ_raw[sorted_idxs]
        Γrs = Array(jld_in["RS/D"])

        num_pos = haskey(jld_in, "UR_asym/num_pos") ? Int(jld_in["UR_asym/num_pos"]) :
                  count(>(zero(eltype(Γ))), Γ)
        num_pos > 0 || error("UR_asym has no positive eigenvalue, so there is no " *
                             "basis and nothing for the bounds to run on")
        num_pos == count(>(zero(eltype(Γ))), Γ) || error(
            "UR_asym/num_pos is $(num_pos) but $(count(>(zero(eltype(Γ))), Γ)) of the " *
            "$(length(Γ)) saved eigenvalues are positive; the RSVD output is inconsistent")

        sender_size = prod(sender(smr).cel) * 3
        receiver_size = prod(receiver(smr).cel) * 3
        N_u = sender_size + receiver_size

        plan = if plan_override !== nothing
            @info string(now()) * " [bounds_bargaining::load_bounds_inputs] Path: panel front end (plan supplied by the caller)" plan_override
            plan_override
        elseif panel_mode === false
            nothing
        elseif panel_mode === true || use_panel_bounds(N_u, num_pos, compute_env)
            residency_plan(compute_env; workspace_bytes=gila_workspace_bytes(N_u))
        else
            nothing
        end

        cols = sorted_idxs[1:num_pos]
        source, path = _ur_asym_vectors_source(jld_in, ur_asym_vectors_path(compute_env, smr))
        footprint = round(bounds_footprint_bytes(N_u, num_pos) / 2^30; digits=2)

        if plan === nothing
            @info string(now()) * " [bounds_bargaining::load_bounds_inputs] Path: in-memory front end (the $(N_u) × $(num_pos) basis, ss and working matrix want $(footprint) GiB)"
            Vpos = _read_ur_asym_dense(jld_in, source, path, cols, N_u)
            size(Vpos) == (N_u, num_pos) || error(
                "the saved basis is $(size(Vpos, 1)) × $(size(Vpos, 2)) but the universe " *
                "is $(N_u) cells' worth of currents and num_pos is $(num_pos)")
            Vur_asym = to_device ? CuArray(Vpos) : Vpos
            return (Γ=Γ, Vur_asym=Vur_asym, Γrs=Γrs, sorted_idxs=sorted_idxs,
                    num_pos=num_pos, plan=nothing)
        end

        @info string(now()) * " [bounds_bargaining::load_bounds_inputs] Path: panel front end (the $(N_u) × $(num_pos) basis, ss and working matrix want $(footprint) GiB on the device)" plan
        basis = _read_ur_asym_panel(jld_in, source, path, cols, N_u, plan)
        size(basis) == (N_u, num_pos) || error(
            "the saved basis is $(size(basis, 1)) × $(size(basis, 2)) but the universe " *
            "is $(N_u) cells' worth of currents and num_pos is $(num_pos)")
        return (Γ=Γ, Vur_asym=basis, Γrs=Γrs, sorted_idxs=sorted_idxs,
                num_pos=num_pos, plan=plan)
    finally
        close(jld_in)
    end
end

"""
    reverse_gram_schmidt!(ss, gs_pos, s_projector, num_pos) -> ss

The probe vectors `sₖ`, built by modified Gram-Schmidt on the columns of
`A = Πₛ · gs_pos` taken in reverse order: column `i` is `Πₛgᵢ` orthogonalized
against `s_{i+1}, …, s_m` and normalized. Column `i` of the result therefore
spans `Πₛ·span(gᵢ, …, g_m)` together with the later columns, which is what makes
the outer loop's probe set `k ≥ n` shrink with `n`.

This is `O(m²)` BLAS-1 work over `N_u`-vectors. `blocked_reverse_gs_transform`
is its blocked equivalent, which is what the panel front end runs instead.
"""
function reverse_gram_schmidt!(ss::AbstractMatrix, gs_pos::AbstractMatrix,
                               s_projector, num_pos::Int)
    for i in num_pos:-1:1
        gᵢ = view(gs_pos, :, i)
        wᵢ = s_projector * gᵢ
        for j in (i+1):num_pos
            sⱼ = view(ss, :, j)
            cᵢⱼ = dot(sⱼ, wᵢ)
            wᵢ .-= cᵢⱼ * sⱼ
        end
        nrm = norm(wᵢ)
        if nrm < 1e-12
            @warn string(now()) * " [bounds_bargaining::bounds_from_spectrum] Warning: vector $i is nearly linearly dependent on the later vectors, norm after orthogonalization is $nrm (we should stop the basis generation here, but I'm too lazy to fix the code right now; hopfully we never see this warning)"
        end
        ss[:, i] .= wᵢ ./ nrm
    end
    return ss
end

"""
    blocked_reverse_gs_transform(S) -> T

The `m × m` factor with `ss = A · T` for `A = Πₛ · gs_pos` and the `ss` that
[`reverse_gram_schmidt!`](@ref) builds, given only `S = Aᴴ A`.

The loop is modified Gram-Schmidt on the columns of `A` in the order
`m, m−1, …, 1`, normalizing each to unit length. Let `J` be the reversal
permutation, so that `Ã = A J` has those columns in the order the loop visits
them. MGS is a QR factorization: `Ã = Q̃ R̃` with `R̃` upper triangular and
`R̃ⱼⱼ = ‖wⱼ‖ > 0` real, and the loop's `sᵢ` is column `m+1−i` of `Q̃`, i.e.
`ss = Q̃ J`. A QR with a positive real diagonal is unique, so the same `R̃` is
the Cholesky factor of

    ÃᴴÃ = J Aᴴ A J = J S J,

and

    ss = Q̃ J = Ã R̃⁻¹ J = A (J R̃⁻¹ J),   so   T = J R̃⁻¹ J.

`R̃⁻¹` is upper triangular, so `T` is lower triangular: column `i` of `ss` draws
on `aᵢ, …, a_m` and nothing earlier, exactly as the loop does. In panel terms
this replaces `m` orthogonalization passes with one `gram` sweep for `S`, this
`m × m` host factorization, and one `rightmul!` sweep for `A · T`.

The basis is RSVD output and hence near-orthonormal, so squaring its condition
number in the Gram matrix is harmless. A Cholesky that fails outright is the
blocked version of the loop's "nearly linearly dependent" warning, since `Πₛ` can
only have rank `sender_size` and an `m` past that is genuinely singular. It gets
the same shifted retry Funicular's `cholqr2!` uses.
"""
function blocked_reverse_gs_transform(S::AbstractMatrix)
    m = size(S, 1)
    m == size(S, 2) || throw(ArgumentError("the Gram matrix must be square, got $(size(S))"))
    rev = m:-1:1
    T = eltype(S)
    G = Matrix{T}(view(Matrix(S), rev, rev)) # J S J
    F = cholesky(Hermitian(G); check=false)
    if !issuccess(F)
        shift = 11 * m * m * eps(real(T)) * norm(G)
        @warn string(now()) * " [bounds_bargaining::blocked_reverse_gs_transform] the $(m)×$(m) Gram matrix of Πₛ·basis is not numerically positive definite, so some basis vector is nearly linearly dependent on the later ones (the reverse Gram-Schmidt loop would have warned about a vanishing norm here). Retrying the Cholesky with a shift of $shift"
        F = cholesky(Hermitian(G + shift * I); check=false)
        issuccess(F) || error("the $(m)×$(m) Gram matrix of Πₛ·basis is not positive " *
            "definite even after a shift of $shift: the projected basis is rank " *
            "deficient (Πₛ has rank at most sender_size), so the probe vectors are not " *
            "defined. Reduce basis_size below the sender's dimension")
    end
    # κ(A) = κ(R̃), and the Cholesky route squares it before factoring, so the
    # blocked probes drift from the loop's at the κ(A)² · eps level. The diagonal
    # of R̃ bounds κ(R̃) from below and costs O(m) to look at. Measured against the
    # loop: κ(A) ≈ 1e3 drifts by 1e-10, κ(A) ≈ 1e6 by 1e-3. Both remain
    # orthonormal bases of the same reverse-nested spans, so a large ratio only
    # costs us agreement on which representative we get.
    d = abs.(diag(F.U))
    ratio = maximum(d) / max(minimum(d), floatmin(real(T)))
    if ratio > 1e6
        @warn string(now()) * " [bounds_bargaining::blocked_reverse_gs_transform] Πₛ·basis has κ ≥ $(round(ratio; sigdigits=3)); the blocked reverse Gram-Schmidt squares that before factoring, so its probes differ from the O(m²) loop's by roughly κ²·eps ≈ $(round(ratio^2 * eps(real(T)); sigdigits=2)). Both are orthonormal bases of the same nested spans, but the bounds will not reproduce the loop's to full precision"
    end
    Rinv = inv(UpperTriangular(F.U))
    return Matrix{T}(view(Matrix(Rinv), rev, rev)) # J R̃⁻¹ J, lower triangular
end

# The N_u-scale front end, written two ways. Both produce the four objects the
# m × m pencil stage consumes (`ss`, `ss_basis`, `C_basis`, `D_basis`) and the
# three stage timings the cost model is fitted against. Nothing downstream of
# here knows which one ran.

function _bounds_front_end_dense(compute_env::ComputeEnvironment, gs_pos, basis,
                                 Γ_pos, Γ_pos_cpu, ζ, s_projector, G⁰ᵤᵤ_asym,
                                 num_pos::Int, RSVD_BASIS_SIZE::Int, sender_size::Int)
    N_u = size(gs_pos, 1)

    # Reverse Gram-Schmidt
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Performing reverse Gram-Schmidt to construct the ss basis"
    t_gram_schmidt = time_ns()
    ss = similar(gs_pos, N_u, num_pos)
    reverse_gram_schmidt!(ss, gs_pos, s_projector, num_pos)
    t_gram_schmidt = (time_ns() - t_gram_schmidt) / 1e9

    t_ss_basis = time_ns()
    ss_basis = basis' * ss
    t_ss_basis = (time_ns() - t_ss_basis) / 1e9

    B_matvec(n::Int, v::AbstractVector) = begin
        idxs = n:num_pos
        G = view(gs_pos, :, idxs) # N × (num_pos - n + 1)
        weights = (G' * v) .* Γ_pos[idxs]
        return (4/ζ) .* (G * weights) # Computes (4/ζ) ∑  γₗ gₗ gₗ' v for all m ≥ l ≥ n (with m = num_pos)
    end
    B(n) = LinearMap(v -> B_matvec(n, v), N_u, N_u; ishermitian=true)
    B_basis(n) = basis' * opmat(B(n), basis)

    C_matvec(v::AbstractVector) = begin
        out = similar_fill(v, zero(eltype(v)))

        out .+= (1/ζ) * (s_projector * v) # ζ⁻¹ Pₛ action

        G = gs_pos # N × num_pos
        A_weights = (G' * v) .* Γ_pos
        out .+= (G * A_weights) # A₊ action

        out .+= G⁰ᵤᵤ_asym * v # (G⁰ᵤᵤ)ᵃ
        return out
    end
    C = LinearMap(C_matvec, N_u, N_u; ishermitian=true)
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Projecting C into the basis of size $(size(basis, 2))"
    t_c_projection = time_ns()
    C_basis = basis' * opmat(C, basis)
    t_c_projection = (time_ns() - t_c_projection) / 1e9

    Bₛ = view(basis, 1:sender_size, :)
    S_basis = Bₛ' * Bₛ # = basis' Πₛ basis, exact whether or not the basis is orthonormal
    D_basis = (1 / ζ) .* S_basis # −ζ⁻¹Πᵣ = ζ⁻¹Πₛ − ζ⁻¹1 in the basis
    D_basis[diagind(D_basis)] .+= view(Γ_pos, 1:RSVD_BASIS_SIZE) .- (1 / ζ)

    return (ss=ss, ss_basis=ss_basis, C_basis=C_basis, D_basis=D_basis,
            t_gram_schmidt=t_gram_schmidt, t_ss_basis=t_ss_basis,
            t_c_projection=t_c_projection)
end

"""
The panel version of the same front end (FUNICULAR_PLAN.md, workstream C1-C3).
The basis is an `N_u × m` `PanelMatrix`, as is every other `N_u`-scale object.
Every `m × m` object is formed on the host and handed to the pencil stage in the
compute device's array space, exactly as the dense path hands it over.

This takes a fixed number of sweeps, against the dense path's `m`
orthogonalization passes over `N_u`-vectors:

1. `panelmul!(ss, Πₛ, basis)`, then `gram(ss)` for `S = basisᴴΠₛbasis`, the host
   [`blocked_reverse_gs_transform`](@ref), and `rightmul!(ss, T)`.
2. `gram(basis, ss)` for `ss_basis`.
3. `panelmul!(work, (G⁰ᵤᵤ)ᵃ, basis)` and `gram(basis, work)`.

`C(1) = ζ⁻¹Πₛ + (−G⁰ᵤᵣ)ᵃ₊ + (G⁰ᵤᵤ)ᵃ` only needs the Green term swept: with
`basis = gs_pos` the other two are already in hand, since
`basisᴴΠₛbasis = S` and `basisᴴ(−G⁰ᵤᵣ)ᵃ₊basis = (basisᴴbasis) diag(Γ₊)
(basisᴴbasis)ᴴ`. `D_basis` reuses `S` as well, so the whole `τ` family comes out
of the same three sweeps.
"""
function _bounds_front_end_panel(compute_env::ComputeEnvironment, basis,
                                 Γ_pos_cpu, ζ, s_projector, G⁰ᵤᵤ_asym,
                                 num_pos::Int, RSVD_BASIS_SIZE::Int)
    RSVD_BASIS_SIZE == num_pos || error(
        "the panel front end assumes the basis is the whole positive block " *
        "(basis_size = num_pos = $(num_pos)), got RSVD_BASIS_SIZE = $(RSVD_BASIS_SIZE). " *
        "A truncated basis would need basisᴴgs_pos as well, which is a fourth sweep " *
        "nothing in production asks for")
    size(basis, 2) == num_pos || error(
        "the panel basis has $(size(basis, 2)) columns but num_pos is $(num_pos)")
    to_device = use_gpu(compute_env) ? CuArray : identity

    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Performing the blocked reverse Gram-Schmidt (two panel sweeps) to construct the ss basis"
    t_gram_schmidt = time_ns()
    ss = similar(basis)
    Funicular.panelmul!(ss, s_projector, basis) # ss = Πₛ basis
    S_basis = Funicular.gram(ss)                # = basisᴴ Πₛ basis, on the host
    T = blocked_reverse_gs_transform(S_basis)
    Funicular.rightmul!(ss, T)                  # ss = Πₛ basis T
    t_gram_schmidt = (time_ns() - t_gram_schmidt) / 1e9

    t_ss_basis = time_ns()
    ss_basis = Funicular.gram(basis, ss)
    t_ss_basis = (time_ns() - t_ss_basis) / 1e9
    Funicular.free!(ss)

    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Projecting C into the basis of size $(size(basis, 2))"
    t_c_projection = time_ns()
    work = similar(basis)
    Funicular.panelmul!(work, G⁰ᵤᵤ_asym, basis)
    C_basis = Funicular.gram(basis, work) # basisᴴ (G⁰ᵤᵤ)ᵃ basis
    Funicular.free!(work)
    gram_bb = Funicular.gram(basis)       # basisᴴ basis, near the identity
    C_basis .+= (1 / ζ) .* S_basis                             # ζ⁻¹Πₛ
    C_basis .+= gram_bb * (Γ_pos_cpu .* gram_bb')              # (−G⁰ᵤᵣ)ᵃ₊
    t_c_projection = (time_ns() - t_c_projection) / 1e9
    Funicular.free!(basis)

    D_basis = (1 / ζ) .* S_basis # −ζ⁻¹Πᵣ = ζ⁻¹Πₛ − ζ⁻¹1 in the basis
    D_basis[diagind(D_basis)] .+= Γ_pos_cpu .- (1 / ζ)

    # `ss` comes back empty: the probes cannot be held as a dense N_u × m array,
    # which is why this path exists at all. `verify_bounds` is the only consumer,
    # and it runs with panel_mode = false for that reason.
    return (ss=nothing, ss_basis=to_device(ss_basis), C_basis=to_device(C_basis),
            D_basis=to_device(D_basis),
            t_gram_schmidt=t_gram_schmidt, t_ss_basis=t_ss_basis,
            t_c_projection=t_c_projection)
end

"""
    bounds_from_spectrum(compute_env, smr, Γ, Vur_asym, Γrs; kwargs...)

Compute the σₙ(Pᵣₛ) bounds from an already-loaded `Asym(G⁰ᵤᵣ)` spectrum. `Γ` must
be sorted in descending order and `Vur_asym`'s columns must be ordered to match.

`Vur_asym` is either a matrix, holding the whole `N_u × k` basis or just the
`N_u × num_pos` positive block as [`load_bounds_inputs`](@ref) returns it, or a
`Funicular.PanelMatrix` holding that block. In the panel case the `N_u`-scale
front end runs as panel sweeps and the returned `ss` is `nothing`, since the
panel matrices, `Vur_asym` included, are freed before the pencil stage.

# Keyword arguments
- `basis_size`: how many leading eigenvectors to use as the projection basis,
  capped at `m = num_pos`, the number of positive `Γ`. 
- `G₀_uu`: pre-loaded universe operator, loaded here if not supplied.
- `outer_indices`: which `n` of the outer `σₙ` loop to actually evaluate.
  `nothing` (the default) means all of them.
- `on_outer_error`: `:throw` (the default) or `:stop`. With `:stop`, a failure in
  the outer loop is recorded in the returned `outer_error` and the function
  returns with `complete = false` rather than propagating. The benchmark harness
  passes `:stop` so that the setup-stage timings, which are measured before the
  loop and are useful on their own, survive a loop that cannot run on synthetic
  input.
- `τs`: grid of `τ ∈ [0, 1]` over which the power-conservation constraint

      C(τ) = ζ⁻¹(Πₛ + (1−τ)Πᵣ) + τ(−G⁰ᵤᵣ)ᵃ₊ + (G⁰ᵤᵤ)ᵃ

  is scanned. `τ` toggles between the two sides of the receiver-power identity
  `t'ζ⁻¹Πᵣt = t'(−G⁰ᵤᵣ)ᵃt` (Eq. (25) of the paper): at `τ = 0` the power
  reaching the receiver is charged as material absorption inside it (global
  asym power conservation), at `τ = 1` it is charged as radiative transfer
  through `(−G⁰ᵤᵣ)ᵃ₊` (the historical behaviour of this code). Physical
  currents satisfy the constraint at every `τ`, so each grid point yields a
  valid bound. The grid plays two roles: the shared diagnostic table
  `bounds_dual_by_tau`, and the bracketing stage for the per-index refinement
  below. Must be sorted ascending.
- `τ_refine_tol`: after the grid sweep, each index's minimiser is refined by a
  golden-section search on the bracket formed by the best grid point and its
  two neighbours, down to this width in `τ`; `nothing` reports the raw grid
  minimum instead. The search is exact rather than heuristic because the dual
  bound is quasi-convex in `τ`: `C(τ) = C₀ + τC₁` is affine in `τ`, so the
  Lagrangian `h(α, β) = sup_t [t'Bₙt + αℑ(sₖ't) − t'(αC₀ + βC₁)t]` is a
  supremum of functions affine in `(α, β)` (jointly convex) and the dual
  value at `τ` is its infimum along the ray `β = ατ`. A sublevel set
  `{τ : g(τ) ≤ c}` is then the set of slopes of rays meeting the convex set
  `{h ≤ c}`, which is an interval, and the max over probes `k ≥ n` preserves
  quasi-convexity. Each index's bound therefore descends to a single minimum
  and rises again so golden-section converges to the global minimiser. Every
  probe evaluation is a valid bound on its own and the running minimum is kept.
  Each probe point costs one m × m whitening plus one GEVP (roughly twice a
  grid point, whose whitener is shared across all indices). The default grid
  and tolerance add about seven probe points per index; the tolerance costs
  precision only in τ, and the bound is flat near its minimum (relative error
  in σₙ² of roughly `4(Δτ)²` on the semi-analytic model), so `0.05` gives up
  under about one percent of tightness relative to the exact minimiser.

# Returns
A named tuple with the bounds, the bookkeeping needed to save them, and
`stage_times` / `outer_times` for calibration. `bounds_dual_basis` holds the
per-index minimum over all evaluated `τ`, `opt_taus` the `τ` that achieved it
(off-grid when refinement improved on the grid), and `bounds_dual_by_tau` the
grid-only `num_pos × length(τs)` table (`NaN` where an index/grid point was
skipped or failed), with the grid echoed in `tau_grid`.

The basis-side objects `ss` (full-space probe vectors, `N × num_pos`),
`ss_basis`, `C_basis` (the `τ = 1` projected constraint) and `D_basis` are also
returned, in their compute-device array space, so that `verify_bounds` can
rebuild any `C(τ)` pencil and its probes without re-deriving them.
"""
function bounds_from_spectrum(compute_env::ComputeEnvironment, smr::SMRSystem,
                              Γ::AbstractVector, Vur_asym,
                              Γrs::AbstractVector;
                              basis_size::Int=size(Vur_asym, 2),
                              G₀_uu=nothing,
                              outer_indices::Union{Nothing,AbstractVector{Int}}=nothing,
                              on_outer_error::Symbol=:throw,
                              τs::AbstractVector{<:Real}=range(0.0, 1.0, length=5),
                              τ_refine_tol::Union{Nothing,Real}=0.05)
    on_outer_error in (:throw, :stop) ||
        throw(ArgumentError("on_outer_error must be :throw or :stop, got :$on_outer_error"))
    isempty(τs) && throw(ArgumentError("τs must contain at least one grid point"))
    all(τ -> zero(τ) <= τ <= one(τ), τs) || throw(ArgumentError(
        "every τ must lie in [0, 1] — the constraint C(τ) is only a convex " *
        "combination of valid power-conservation statements on that interval, " *
        "got extrema(τs) = $(extrema(τs))"))
    issorted(τs) || throw(ArgumentError(
        "τs must be sorted ascending: the refinement step brackets the minimum " *
        "between the best grid point's neighbours"))
    isnothing(τ_refine_tol) || τ_refine_tol > 0 || throw(ArgumentError(
        "τ_refine_tol must be positive, or `nothing` to disable refinement, " *
        "got $τ_refine_tol"))
    # U_uu = read_array(jld_in, "UU/U", use_gpu(compute_env)) # TODO: could use this as basis too
    if isnothing(G₀_uu)
        G₀_uu = load_green_function(compute_env, smr, [Sender, Receiver], [Sender, Receiver]) # universe -> universe
    end
    # r_projector, s_projector, u_projector, G₀_uu_disjoint = projected_operators(G₀_uu, smr, compute_env)
    s_projector = projected_operators(G₀_uu, smr, compute_env)
    # G⁰ᵤᵤ_asym = u_projector * asym(LinearMap(G₀_uu)) * u_projector
    G⁰ᵤᵤ_asym = asym(LinearMap(G₀_uu))

    GC.gc()
    GC.gc()
    GC.gc()

    # @info "hello" size(Vur_asym) size(G₀_uu) size(s_projector) size(Γ) size(Γrs)

    χ = susceptibility(smr)
    ζ = abs(χ)^2/imag(χ)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Susceptibility χ = $χ, material factor ζ = $ζ"

    num_pos = count(Γ .> zero(eltype(Γ))) # = m, the numerical rank of (−G⁰ᵤᵣ)ᵃ₊
    Γ_pos = Γ[1:num_pos] # These have been sorted in descending order; keep only the positive eigenvalues
    Γ_pos_cpu = Array(Γ_pos) # the per-n diagonal of Bₙ is assembled on the host
    if use_gpu(compute_env)
        Γ_pos = CuArray(Γ_pos)
    end
    # The projection basis is the m-dimensional span of the gₖ
    RSVD_BASIS_SIZE = min(basis_size, num_pos)
    # basis = cat(U_uu, Vur_asym; dims=2)
    # basis = qthin!(basis) # Orthonormalize the basis using QR factorization
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Using RSVD_BASIS_SIZE = $RSVD_BASIS_SIZE (num_pos = $num_pos of $(length(Γ)) RSVD directions)"

    # The C the front end projects is the τ = 1 endpoint of the constraint family
    #
    #     C(τ) = ζ⁻¹(Πₛ + (1−τ)Πᵣ) + τ(−G⁰ᵤᵣ)ᵃ₊ + (G⁰ᵤᵤ)ᵃ,
    #
    # which interpolates between the two sides of the receiver-power identity
    # t'ζ⁻¹Πᵣt = t'(−G⁰ᵤᵣ)ᵃt: τ = 0 charges the power reaching the receiver as
    # material absorption inside it (global asym power conservation), τ = 1 as
    # radiative transfer. Physical currents satisfy every convex combination, so
    # each τ bounds σₙ(Pᵣₛ) on its own and the per-index minimum over a grid is
    # free tightening. Writing C(τ) = C(1) − (1−τ)D needs no further matrix-free
    # projections: D = (−G⁰ᵤᵣ)ᵃ₊ − ζ⁻¹Πᵣ is exact in this basis, because
    # basis'(−G⁰ᵤᵣ)ᵃ₊basis = diag(Γ₊) by the same orthonormality that makes Bₙ
    # diagonal, and basis'Πₛbasis is the Gram matrix of the stored sender rows.
    sender_size = prod(sender(smr).cel) * 3
    receiver_size = prod(receiver(smr).cel) * 3
    size(Vur_asym, 1) == sender_size + receiver_size || error(
        "the universe is not [sender; receiver] ($(size(Vur_asym, 1)) ≠ " *
        "$sender_size + $receiver_size), so Πᵣ ≠ 1 − Πₛ and the τ family " *
        "cannot be assembled from the sender projector alone")

    front = if Vur_asym isa Funicular.PanelMatrix
        _bounds_front_end_panel(compute_env, Vur_asym, Γ_pos_cpu, ζ, s_projector,
                                G⁰ᵤᵤ_asym, num_pos, RSVD_BASIS_SIZE)
    else
        # These have been sorted in descending order of the corresponding Γ values;
        # keep only the eigenvectors with positive eigenvalues. Aliased when the
        # loaded block is already positives-only.
        gs_pos = size(Vur_asym, 2) == num_pos ? Vur_asym : Vur_asym[:, 1:num_pos]
        basis = RSVD_BASIS_SIZE == num_pos ? gs_pos : gs_pos[:, 1:RSVD_BASIS_SIZE] # aliased when full, to avoid a second N × m copy
        _bounds_front_end_dense(compute_env, gs_pos, basis, Γ_pos, Γ_pos_cpu, ζ,
                                s_projector, G⁰ᵤᵤ_asym, num_pos, RSVD_BASIS_SIZE,
                                sender_size)
    end
    ss, ss_basis = front.ss, front.ss_basis
    C_basis, D_basis = front.C_basis, front.D_basis
    t_gram_schmidt, t_ss_basis = front.t_gram_schmidt, front.t_ss_basis
    t_c_projection = front.t_c_projection

    # None of the C(τ) depend on n, so the grid pencils are eigendecomposed once
    # here; the golden-section refinement builds throwaway pencils on demand
    # (those whitenings land in outer_times rather than c_range). 
    build_pencil(τ::Real) = begin
        C_τ = isone(τ) ? C_basis : C_basis .- (1 - τ) .* D_basis
        try
            psd_pencil_whitener(C_τ)
        catch err
            @warn string(now()) * " [bounds_bargaining::bounds_from_spectrum] psd_pencil_whitener failed at τ=$τ; treating this point as unusable" exception = err
            nothing
        end
    end
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Eigendecomposing C(τ) in the basis for τ ∈ $(collect(τs))"
    t_c_range = time_ns()
    pencils = [build_pencil(τ) for τ in τs]
    usable_τ = findall(!isnothing, pencils)
    isempty(usable_τ) && error("psd_pencil_whitener failed at every τ in $(collect(τs))")
    t_c_range = (time_ns() - t_c_range) / 1e9
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] C(τ) numerical ranks: " *
        join(("τ=$(τs[i]) → $(pencils[i].rank)/$(size(C_basis, 1))" for i in usable_τ), ", ")

    # The probes stay in the pencil's array space. Only the small projected
    # b-vectors cross to the host, where the scalar root finds live.
    B_basis_diagonal = zeros(real(eltype(C_basis)), RSVD_BASIS_SIZE)

    bounds_dual_basis = zeros(Float64, num_pos)
    bounds_dual_by_tau = fill(NaN, num_pos, length(τs))
    opt_taus = fill(NaN, num_pos)
    ns = isnothing(outer_indices) ? (1:RSVD_BASIS_SIZE) :
         filter(n -> 1 <= n <= RSVD_BASIS_SIZE, outer_indices)
    complete = length(ns) == num_pos
    outer_times = Tuple{Int,Float64}[]
    outer_error = nothing
    for n in ns # Compute bounds on σₙ(Pᵣₛ)
     try
        t_outer = time_ns()
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] [$n/$(num_pos)] Computing σₙ(Pᵣₛ) bound"

        @info string(now()) * " [$n/$(num_pos)] Projecting Bₙ into the basis of size $(RSVD_BASIS_SIZE)"
        # B_basis_n = B_basis(n)
        fill!(B_basis_diagonal, zero(eltype(B_basis_diagonal)))
        B_basis_diagonal[n:RSVD_BASIS_SIZE] .= (4/ζ) .* Γ_pos_cpu[n:RSVD_BASIS_SIZE] # Bₙ is diagonal in the gs_pos basis, so no projection is needed

        # Solve the GEVP on each C(τ)'s numerical range and keep the tightest τ.
        # Bₙ shrinks with n (Bₙ ⪯ Bₙ₋₁, as they differ by a positive
        # semi-definite rank-one term), but the null directions it is allowed to
        # ignore grow with n too, so the check inside diag_pencil_eigen has to
        # happen per index rather than once up front. Every τ bounds σₙ(Pᵣₛ) on
        # its own, so an evaluation that fails numerically is dropped for this
        # index with a warning.
        pencil_dual(pencil, τ) = maximum(pencil_probe_duals(pencil, B_basis_diagonal, ss_basis, n, num_pos; τ=τ).duals)
        best_dual = Inf
        best_τ = NaN
        best_grid_idx = 0
        last_τ_error = nothing
        eval_dual(pencil, τ) = begin
            isnothing(pencil) && return Inf
            try
                pencil_dual(pencil, τ)
            catch err
                last_τ_error = err
                @warn string(now()) * " [bounds_bargaining::bounds_from_spectrum] [$n/$(num_pos)] τ=$τ failed; dropping this evaluation" exception = err
                Inf
            end
        end

        @info string(now()) * " [$n/$(num_pos)] Solving λⱼ(Bₙ, C(τ)) over $(length(usable_τ)) τ grid point(s)"
        for i in usable_τ
            dual_τ = eval_dual(pencils[i], τs[i])
            isfinite(dual_τ) || continue
            bounds_dual_by_tau[n, i] = sqrt(dual_τ)
            if dual_τ < best_dual
                best_dual, best_τ, best_grid_idx = dual_τ, Float64(τs[i]), i
            end
        end
        isfinite(best_dual) || error("every τ in the grid failed at n=$n" *
            (last_τ_error === nothing ? "" : "; last error: $(sprint(showerror, last_τ_error))"))

        # The dual bound is quasi-convex in τ (see the τ_refine_tol docstring),
        # so the grid minimum brackets the true minimiser between its two
        # neighbours, and golden-section inside that bracket converges to it.
        # The running minimum keeps every evaluation, so noise denting
        # unimodality can only cost tightness, never validity.
        if !isnothing(τ_refine_tol) && length(τs) > 1
            lo = Float64(τs[max(best_grid_idx - 1, firstindex(τs))])
            hi = Float64(τs[min(best_grid_idx + 1, lastindex(τs))])
            invφ = (sqrt(5.0) - 1) / 2
            τ₁ = hi - invφ * (hi - lo)
            τ₂ = lo + invφ * (hi - lo)
            g₁ = eval_dual(build_pencil(τ₁), τ₁)
            g₂ = eval_dual(build_pencil(τ₂), τ₂)
            g₁ < best_dual && ((best_dual, best_τ) = (g₁, τ₁))
            g₂ < best_dual && ((best_dual, best_τ) = (g₂, τ₂))
            refine_iters = 0
            while hi - lo > τ_refine_tol && refine_iters < 200
                refine_iters += 1
                if g₁ <= g₂
                    hi, τ₂, g₂ = τ₂, τ₁, g₁
                    τ₁ = hi - invφ * (hi - lo)
                    g₁ = eval_dual(build_pencil(τ₁), τ₁)
                    g₁ < best_dual && ((best_dual, best_τ) = (g₁, τ₁))
                else
                    lo, τ₁, g₁ = τ₁, τ₂, g₂
                    τ₂ = lo + invφ * (hi - lo)
                    g₂ = eval_dual(build_pencil(τ₂), τ₂)
                    g₂ < best_dual && ((best_dual, best_τ) = (g₂, τ₂))
                end
            end
        end
        @info string(now()) * " [$n/$(num_pos)] Dual is $best_dual at τ = $best_τ, which gives a bound of $(sqrt(best_dual)) on σₙ(Pᵣₛ)"
        bounds_dual_basis[n] = sqrt(best_dual)
        opt_taus[n] = best_τ
        push!(outer_times, (n, (time_ns() - t_outer) / 1e9))
     catch err
        on_outer_error === :throw && rethrow(err)
        # :stop records where it failed and keep whatever has been measured
        frames = stacktrace(catch_backtrace())
        where_str = isempty(frames) ? "unknown" :
                    join(["$(f.func)@$(basename(String(f.file))):$(f.line)"
                          for f in Iterators.take(frames, 3)], "<-")
        outer_error = (n=n, exception=sprint(showerror, err), location=where_str)
        @warn string(now()) * " [bounds_bargaining::bounds_from_spectrum] outer loop failed at n=$n; stopping" exception = err location = where_str
        complete = false
        break
     end
    end

    analytical_bounds_old_form(κ) = ifelse(κ >= one(eltype(κ)), one(eltype(κ)), sqrt(4κ)/(1+κ))
    analytical_bounds_new_form(κ̃) = ifelse(2κ̃ >= one(eltype(κ̃)), one(eltype(κ̃)), sqrt(4κ̃*abs(1-κ̃)))
    κ = ζ^2 .* (Γrs .^ 2)
    κ̃ = ζ .* (max.(Γ, zero(eltype(Γ))))
    old_analytical_bounds = analytical_bounds_old_form.(κ)
    new_analytical_bounds = analytical_bounds_new_form.(κ̃)
    new_analytical_bounds[new_analytical_bounds .<= zero(eltype(new_analytical_bounds))] .= NaN
    old_analytical_bounds[old_analytical_bounds .<= zero(eltype(old_analytical_bounds))] .= NaN

    ks = 1:min(length(old_analytical_bounds), length(new_analytical_bounds), length(bounds_dual_basis))
    which_bounds = [argmin((old_analytical_bounds[k], new_analytical_bounds[k],
                            bounds_dual_basis[k])) for k in ks]
    # true_bounds = min.(old_analytical_bounds[ks], new_analytical_bounds[ks], bounds_dual_basis[ks])
    true_bounds = map(i -> begin # Save which bound is the minimum for each k to see where the flips happen
        if which_bounds[i] == 1
            return old_analytical_bounds[ks[i]]
        elseif which_bounds[i] == 2
            return new_analytical_bounds[ks[i]]
        else
            return bounds_dual_basis[ks[i]]
        end
    end, 1:length(ks))

    # plt = plot(eachindex(bounds_dual_basis), bounds_dual_basis, label="Dual (RSVD basis)")
    # plot!(plt, eachindex(old_analytical_bounds), old_analytical_bounds, label="Passivity [γᵣₛ]")
    # plot!(plt, eachindex(new_analytical_bounds), new_analytical_bounds, label="Passivity [γᵤᵣᵃ]")
    # plot!(plt, eachindex(true_bounds), true_bounds, label="Min", ls=:dash)
    # plot!(plt, xlabel="n", ylabel="Bound on σₙ(Pᵣₛ)", yscale=:log10, minorticks=true, minorgrid=true, legend=:right)
    # display(plt)
    # println("Press Enter to continue...")
    # readline()

    stage_times = (gram_schmidt=t_gram_schmidt, ss_basis=t_ss_basis,
                   c_projection=t_c_projection, c_range=t_c_range,
                   outer_total=sum(last.(outer_times); init=0.0))
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Stage times [s]:" stage_times

    return (num_pos=num_pos, complete=complete, outer_error=outer_error,
            bounds_dual_basis=bounds_dual_basis,
            tau_grid=collect(Float64, τs), opt_taus=opt_taus,
            bounds_dual_by_tau=bounds_dual_by_tau,
            ss=ss, ss_basis=ss_basis, C_basis=C_basis, D_basis=D_basis,
            old_analytical_bounds=old_analytical_bounds,
            new_analytical_bounds=new_analytical_bounds,
            true_bounds=true_bounds, which_bounds=which_bounds, ks=ks,
            basis_size=RSVD_BASIS_SIZE,
            stage_times=stage_times, outer_times=outer_times)
end

function _compute_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams;
                            plan_override=nothing, panel_mode::Union{Nothing,Bool}=nothing)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Computing bounds for SR system"

    inputs = load_bounds_inputs(compute_env, smr; plan_override=plan_override,
                                panel_mode=panel_mode)
    Γ, Vur_asym, Γrs, sorted_idxs = inputs.Γ, inputs.Vur_asym, inputs.Γrs, inputs.sorted_idxs

    # Written up front (truncating any previous run's file) so that the ordering
    # is on disk even if the bounds loop below is cut short by a time limit.
    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out = jldopen(jld_out_path, "w")
    jld_out["Γrs"] = Array(Γrs)
    jld_out["ordering_idxs"] = sorted_idxs
    close(jld_out)

    result = bounds_from_spectrum(compute_env, smr, Γ, Vur_asym, Γrs)
    result.complete || error("bounds_from_spectrum returned an incomplete result; refusing to save partial bounds")

    # Save data to disk
    jld_out = jldopen(jld_out_path, "a")
    if !haskey(jld_out, "χ")
        jld_out["χ"] = susceptibility(smr)
    end
    if !haskey(jld_out, "Γ")
        jld_out["Γ"] = Array(Γ)
    end
    if !haskey(jld_out, "Γrs")
        jld_out["Γrs"] = Array(Γrs)
    end
    if !haskey(jld_out, "bounds_dual_basis")
        jld_out["bounds_dual_basis"] = result.bounds_dual_basis
    end
    if !haskey(jld_out, "tau_grid")
        jld_out["tau_grid"] = result.tau_grid
    end
    if !haskey(jld_out, "opt_taus")
        jld_out["opt_taus"] = result.opt_taus
    end
    if !haskey(jld_out, "bounds_dual_by_tau")
        jld_out["bounds_dual_by_tau"] = result.bounds_dual_by_tau
    end
    if !haskey(jld_out, "old_analytical_bounds")
        jld_out["old_analytical_bounds"] = result.old_analytical_bounds
    end
    if !haskey(jld_out, "new_analytical_bounds")
        jld_out["new_analytical_bounds"] = result.new_analytical_bounds
    end
    if !haskey(jld_out, "true_bounds")
        jld_out["true_bounds"] = result.true_bounds
    end
    if !haskey(jld_out, "which_bounds")
        jld_out["which_bounds"] = result.which_bounds
    end
    close(jld_out)
    return result
end

function compute_bounds()
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using GPU acceleration on device $(gpu_device(compute_env))"
        if !haskey(ENV, "CC_CLUSTER") # This breaks on compute canada
            CUDA.device!(gpu_device(compute_env))
        end
    else
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _compute_bounds_sr(compute_env, smr, rsvd_params)
    else
        _compute_bounds_smr(compute_env, smr, rsvd_params)
    end
end
