using CUDA
using Dates
using LinearAlgebra
using LinearMaps
using Roots
using JLD2
using MatrixFreeRandomizedLinearAlgebra
import Funicular

# ---------------------------------------------------------------------------
# The dual bound on σₙ(Pᵣₛ)
#
# Every operator is projected into an orthonormal `basis`: the kept positive-Γ
# eigenvectors of Asym(G⁰ᵤᵣ) (the "g basis", `m` columns), augmented with the top
# `k_uu` eigenvectors of Asym(G⁰ᵤᵤ). The augmentation is what keeps the projected
# dual a bound at large separation: there the g span collapses (m = 15 of
# N_u = 196,608 at 1 λ, s = 357/32 λ) and cannot represent the constraint, so the
# projected dual under-reports (a Schur complement, b'(P'CP)⁻¹b ≤ b'C⁻¹b). The
# subspaces are nested, so the bound is monotone in k_uu; it saturates near 512.
#
# The constraint family is
#
#     C(τ) = ζ⁻¹(Πₛ + (1−τ)Πᵣ) + τ(−G⁰ᵤᵣ)ᵃ₊ + (G⁰ᵤᵤ)ᵃ,   τ ∈ [0, 1],
#
# which interpolates between the two sides of the receiver-power identity
# t'ζ⁻¹Πᵣt = t'(−G⁰ᵤᵣ)ᵃt. Physical currents satisfy every τ, so each τ bounds σₙ on
# its own and the minimum over τ is free tightening. With D = (−G⁰ᵤᵣ)ᵃ₊ − ζ⁻¹Πᵣ,
# C(τ) = C(1) − (1−τ)D, so one projection of C(1) and D gives the whole family.
#
# In the basis, with W = basisᴴ gs (m_aug × m),
#
#     basisᴴ (−G⁰ᵤᵣ)ᵃ₊ basis = W diag(Γ) Wᴴ,   basisᴴ Bₙ basis = W[:, n:m] diag(4Γ[n:m]/ζ) W[:, n:m]ᴴ,
#
# so only (G⁰ᵤᵤ)ᵃ needs Green applies. The probes sₖ are built from Πₛ·g alone, by a
# reverse Gram-Schmidt, so that index n names the n-th channel; the augmentation
# enlarges the space the dual is solved in, not the objective.
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
# How far B may reach into the numerical null space of C before the bound is +∞.
const AUG_BTOL = 1e-8
# Relative norm a probe may lose to the reverse Gram-Schmidt before it counts as dependent.
const GS_DEPENDENCE_RTOL = 1e-12

"""
    bracket_root(f, λs, b; btol=0, maxsteps=200)

A bracket `((left, f(left)), (right, f(right)))` with `f(left) < 0 < f(right)` for

    f(α) = ∑ⱼ |bⱼ|² (α - 2λⱼ) / (α - λⱼ)²,

which tends to `-∞` at the largest pole `λⱼ` with `bⱼ ≠ 0` and to `0⁺` as `α → ∞`.
"""
function bracket_root(f::Function, λs::AbstractVector{<:Real}, b::AbstractVector;
                      btol::Real=zero(real(eltype(b))), maxsteps::Int=200)
    all(isfinite, λs) || throw(ArgumentError("bracket_root: some λⱼ are not finite"))
    pole_idxs = findall(j -> abs2(b[j]) > btol, eachindex(λs))
    isempty(pole_idxs) && throw(ArgumentError("bracket_root: every |bⱼ|² is <= btol = $btol, so f ≡ 0"))
    pole = maximum(view(λs, pole_idxs))

    # Halve an offset of order |pole| until f is finite and negative: the largest such
    # offset keeps the endpoint away from the pole, where f is stiffest.
    T = typeof(pole)
    left = nothing
    δ = iszero(pole) ? one(T) : abs(pole)
    for _ in 1:maxsteps
        α = pole + δ
        fα = f(α)
        isfinite(fα) && fα < 0 && (left = (α, fα); break)
        α == pole && break # δ has underflowed against the pole
        δ /= 2
    end
    isnothing(left) && throw(ArgumentError(
        "bracket_root: no representable α > λmax = $pole with f(α) < 0. Either λmax ≤ 0, " *
        "or the sign change next to λmax cannot be resolved in $T"))

    right = nothing
    step = max(abs(left[1]), one(T))
    for _ in 1:maxsteps
        α = left[1] + step
        fα = f(α)
        isfinite(fα) && fα > 0 && (right = (α, fα); break)
        step *= 2
    end
    isnothing(right) && throw(ArgumentError("bracket_root: no α > $(left[1]) with f(α) > 0 after $maxsteps steps"))
    return left, right
end

"""
    psd_pencil_whitener(C; rtol=size(C, 1) * eps)

Eigendecompose `C` and return the `whitener` `W = U₊ diag(μ₊)^(-1/2)` over the
eigenpairs above `tol = rtol · μmax` (so `W'CW = 1`), the `nullspace` below it, and
`values`, `tol`, `rank`. Every C(τ) is a sum of positive semidefinite terms, so an
eigenvalue below `-tol` is a sign error upstream and an error here. CUSOLVER reports a
failed factorization as non-finite eigenvalues, which is checked too.
"""
function psd_pencil_whitener(C::AbstractMatrix; rtol::Real=size(C, 1) * eps(real(float(eltype(C)))))
    F = eigen(Hermitian(C)) # ascending
    μ = Array(F.values)
    all(isfinite, μ) || error("psd_pencil_whitener: the eigendecomposition did not converge")
    μmax = maximum(μ)
    μmax > 0 || error("psd_pencil_whitener: C has no positive eigenvalue (largest is $μmax)")
    tol = rtol * μmax
    minimum(μ) >= -tol || error("psd_pencil_whitener: C has eigenvalue $(minimum(μ)), below " *
        "the roundoff floor -$tol. C(τ) is a sum of positive semidefinite terms, so one has the wrong sign")
    num_null = searchsortedlast(μ, tol)
    kept = (num_null + 1):length(μ)
    inv_sqrt = similar(F.values, length(kept))
    copyto!(inv_sqrt, 1 ./ sqrt.(μ[kept]))
    W = F.vectors[:, kept] .* inv_sqrt'
    return (whitener=W, nullspace=F.vectors[:, 1:num_null], values=μ, tol=tol, rank=length(kept))
end

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

"""
    FactoredB(V, c)

`B = V diag(c²) Vᴴ` in factored form, so the pencil never builds `B`. In the basis,
`Bₙ` is this with `V = W[:, n:m]` and `c = sqrt.(4Γ[n:m]/ζ)`.
"""
struct FactoredB{M<:AbstractMatrix,V<:AbstractVector}
    V::M
    c::V
end

# K X for K = diag(c) Vᴴ, so that Xᴴ Kᴴ K X = Xᴴ B X.
apply_factor(F::FactoredB, X::AbstractMatrix) = F.c .* (F.V' * X)

factored_diag(F::FactoredB) = vec(sum(abs2.(F.V) .* reshape(F.c .^ 2, 1, :); dims=2))

"""
    factored_pencil_eigen(F, W, N; btol=AUG_BTOL) -> (values, vectors)

Solve `B v = λ C v` for `B = F.V diag(F.c²) F.Vᴴ`, given the `whitener` `W` and
`nullspace` `N` of `psd_pencil_whitener(C)`, with `vectorsᴴ C vectors = 1`.

Dropping `N` is lossless only if `B` is negligible on it: for positive semidefinite
`B`, `vᴴBv ≈ 0` implies `Bv ≈ 0`, while `vᴴBv > 0` is a direction the constraint does
not bound, i.e. a bound of `+∞` that must not be reported as finite.
"""
function factored_pencil_eigen(F::FactoredB, W::AbstractMatrix, N::AbstractMatrix; btol::Real=AUG_BTOL)
    all(>=(0), F.c) || throw(ArgumentError("FactoredB's weights must be non-negative, got minimum(c) = $(minimum(F.c))"))
    d = Array(factored_diag(F))
    dmax = isempty(d) ? zero(eltype(d)) : maximum(d)
    if size(N, 2) > 0 && dmax > 0
        worst = maximum(sum(abs2, apply_factor(F, N); dims=1))
        worst <= btol * dmax || error("factored_pencil_eigen: B is not negligible on the null " *
            "space of C (max_j N_j'B N_j = $worst > $(btol * dmax)), so the bound for this index is +∞")
    end
    Wd = apply_factor(F, W) # Wdᴴ Wd = Wᴴ B W, positive semidefinite by construction
    E = eigen(Hermitian(Wd' * Wd))
    Λ = Array(E.values)
    all(isfinite, Λ) || error("factored_pencil_eigen: the eigendecomposition did not converge")
    return Λ, W * E.vectors
end

"""
    factored_probe_duals(pencil, F, ss_basis, n, num_pos; τ=NaN) -> NamedTuple

For each probe `k ∈ n:num_pos`, the stationary multiplier `αₖ` of
`∑ⱼ |bⱼ|² (α - 2λⱼ)/(α - λⱼ)² = 0` (`b = V'sₖ` in the pencil's eigenbasis) and the dual
`αₖ²/4 ∑ⱼ |bⱼ|²/(αₖ - λⱼ)`. The pencil's bound for index `n` is the largest dual.
"""
function factored_probe_duals(pencil, F::FactoredB, ss_basis::AbstractMatrix, n::Int, num_pos::Int; τ::Real=NaN)
    Λ, V = factored_pencil_eigen(F, pencil.whitener, pencil.nullspace)
    ks = collect(n:num_pos)
    alphas = zeros(Float64, length(ks))
    duals = zeros(Float64, length(ks))
    for (i, k) in enumerate(ks)
        sₖ = view(ss_basis, :, k)
        if size(pencil.nullspace, 2) > 0
            sₖ_null = norm(pencil.nullspace' * sₖ)
            sₖ_null <= 1e-8 * norm(sₖ) || error("probe k=$k at n=$n, τ=$τ has ‖N'sₖ‖ = $sₖ_null " *
                "of ‖sₖ‖ = $(norm(sₖ)) in the null space of C(τ), so the bound is +∞")
        end
        b = Array(V' * sₖ)
        f(α) = sum(abs2(bⱼ) * (α - 2λⱼ) / (α - λⱼ)^2 for (bⱼ, λⱼ) in zip(b, Λ))
        ((left, _), (right, _)) = bracket_root(f, Λ, b)
        αₖ = find_zero(f, (left, right), Roots.Brent())
        alphas[i] = αₖ
        duals[i] = αₖ^2 / 4 * sum(abs2(bⱼ) / (αₖ - λⱼ) for (bⱼ, λⱼ) in zip(b, Λ))
    end
    return (ks=ks, alphas=alphas, duals=duals)
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
(`nothing`: no limit, as on a CPU run). The dense augmented front end holds
`2·m_aug + m` tall columns, the augmentation's QR `m + 3k`, and the sketch
`RSVD_PEAK_FUDGE · 3(k + p)`. Oversamples are cut first (down to `UU_MIN_OVERSAMPLES`),
then `k_uu`, with a warning; below `K_UU_FLOOR` this errors.
"""
function fit_augmentation(N_u::Integer, m::Integer, k_uu::Integer, budget_bytes::Union{Nothing,Real})
    p_max = min(UU_OVERSAMPLES, N_u - k_uu)
    isnothing(budget_bytes) && return (k_uu=k_uu, oversamples=p_max)
    columns = floor(Int, budget_bytes / (16 * N_u))
    sketch_cols(p) = floor(Int, columns / (3 * RSVD_PEAK_FUDGE)) - p
    k = min(k_uu, fld(columns - 3m, 2), fld(columns - m, 3), sketch_cols(UU_MIN_OVERSAMPLES))
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
descending), `Vur_asym` (the `N_u × num_pos` positive block, on the device for a GPU
run), `Γrs`, `num_pos` and `sorted_idxs`.

`gamma_rtol` keeps only `Γ ≥ gamma_rtol · Γ₁`. Past the operator's numerical rank the
RSVD returns its own noise floor, whose directions are neither eigenvectors nor
independent, and would break the reverse Gram-Schmidt.
"""
function load_bounds_inputs(compute_env::ComputeEnvironment, smr::SMRSystem; gamma_rtol::Float64=DEFAULT_GAMMA_RTOL)
    0 <= gamma_rtol <= 1 || throw(ArgumentError("gamma_rtol must lie in [0, 1], got $gamma_rtol"))
    jld_path = rsvd_output_path(compute_env, smr)
    N_u = dof_length(sender_mesh(smr)) + dof_length(receiver_mesh(smr))
    jldopen(jld_path, "r") do jld
        haskey(jld, "UR_asym/num_pos") || error("$(jld_path) has no finished UR_asym; run generate_rsvd first")
        Γ_raw = Array(jld["UR_asym/D"])
        sorted_idxs = sortperm(Γ_raw, rev=true)
        sorted_idxs[1:jld["UR_asym/num_pos"]] == 1:jld["UR_asym/num_pos"] ||
            error("UR_asym/D is not descending, so the stored positive block does not line up with it")
        Γ = Γ_raw[sorted_idxs]
        stored = Int(jld["UR_asym/num_pos"])
        stored == count(>(0), Γ) || error("UR_asym/num_pos is $(stored) but $(count(>(0), Γ)) eigenvalues are positive")
        num_pos = count(>=(gamma_rtol * Γ[1]), view(Γ, 1:stored))
        num_pos > 0 || error("UR_asym has no positive eigenvalue, so there is no basis to run on")
        num_pos < stored && @info string(now()) * " [bounds::load_bounds_inputs] gamma_rtol = $(gamma_rtol) keeps $(num_pos) of $(stored) positive Γ"

        Vpos = Matrix{ComplexF64}(undef, N_u, num_pos)
        if haskey(jld, "UR_asym/vectors_file")
            path = joinpath(dirname(jld_path), jld["UR_asym/vectors_file"])
            # One hyperslab of the leading columns, never the stored width.
            store = Funicular.open_store(path, "r")
            try
                (store.N, store.k) == (N_u, stored) || error("$(path) holds $(store.N) × $(store.k), expected $(N_u) × $(stored)")
                staged = store.stored === ComplexF64 ? Vpos : Matrix{store.stored}(undef, N_u, num_pos)
                Funicular.read_panel!(Funicular.DiskHome(store, 1:num_pos), staged)
                staged === Vpos || copyto!(Vpos, staged)
            finally
                Funicular.close_store!(store)
            end
        else
            V = jld["UR_asym/V_pos"]
            size(V) == (N_u, stored) || error("UR_asym/V_pos is $(size(V)), expected $(N_u) × $(stored)")
            copyto!(Vpos, view(V, :, 1:num_pos))
        end
        return (Γ=Γ, Vur_asym=use_gpu(compute_env) ? CuArray(Vpos) : Vpos,
                Γrs=Array(jld["RS/D"]), num_pos=num_pos, sorted_idxs=sorted_idxs)
    end
end

"""
    reverse_gram_schmidt!(ss, gs_pos, sender_size, num_pos) -> ss

The probes `sₖ`: modified Gram-Schmidt on the columns of `Πₛ·gs_pos` in reverse order,
so column `i` spans `Πₛ·span(gᵢ, …, g_m)` with the later columns and the probe set
`k ≥ n` shrinks with `n`. `Πₛ` keeps the first `sender_size` rows (`[sender; receiver]`).
A column that loses all but `GS_DEPENDENCE_RTOL` of its norm is an error: the nested
spans would not exist. Raising `--gamma-rtol` keeps the noise floor out of the basis.
"""
function reverse_gram_schmidt!(ss::AbstractMatrix, gs_pos::AbstractMatrix, sender_size::Integer, num_pos::Int)
    fill!(view(ss, (sender_size + 1):size(ss, 1), :), zero(eltype(ss)))
    for i in num_pos:-1:1
        wᵢ = view(ss, 1:sender_size, i)
        copyto!(wᵢ, view(gs_pos, 1:sender_size, i))
        nrm₀ = norm(wᵢ)
        for j in (i+1):num_pos
            sⱼ = view(ss, 1:sender_size, j)
            wᵢ .-= dot(sⱼ, wᵢ) .* sⱼ
        end
        nrm = norm(wᵢ)
        nrm > GS_DEPENDENCE_RTOL * nrm₀ || error("reverse_gram_schmidt!: Πₛg$(i) is numerically in " *
            "the span of the later probes ($(nrm₀) → $(nrm)). Raise --gamma-rtol (default $(DEFAULT_GAMMA_RTOL))")
        wᵢ ./= nrm
    end
    return ss
end

"""
    bounds_front_end(gs_pos, basis, Γ_pos, ζ, G⁰ᵤᵤ_asym, sender_size) -> NamedTuple

The m_aug × m_aug objects the pencil stage needs: the probes in the basis (`ss_basis`),
`C_basis = basisᴴ C(1) basis`, `D_basis` and `W = basisᴴ gs_pos`. Only (G⁰ᵤᵤ)ᵃ takes
Green applies; the Πₛ and (−G⁰ᵤᵣ)ᵃ₊ terms are algebra on `basis` and `W`, and
`−ζ⁻¹Πᵣ = ζ⁻¹Πₛ − ζ⁻¹1` because the basis is orthonormal.
"""
function bounds_front_end(gs_pos, basis, Γ_pos::Vector{Float64}, ζ, G⁰ᵤᵤ_asym, sender_size::Int)
    num_pos = size(gs_pos, 2)
    m_aug = size(basis, 2)
    T = eltype(basis)

    @info string(now()) * " [bounds::bounds_front_end] Reverse Gram-Schmidt on the $(num_pos) g columns"
    ss = similar(gs_pos, size(gs_pos, 1), num_pos)
    reverse_gram_schmidt!(ss, gs_pos, sender_size, num_pos)
    ss_basis = basis' * ss
    ss = nothing # N_u × m, and the projection below is the front end's peak

    @info string(now()) * " [bounds::bounds_front_end] Projecting C into the basis of size $(m_aug)"
    W = basis' * gs_pos
    Bₛ = view(basis, 1:sender_size, :)
    S_basis = Bₛ' * Bₛ
    Γdev = similar(basis, real(T), num_pos)
    copyto!(Γdev, Γ_pos)
    A_basis = (W .* reshape(Γdev, 1, :)) * W'
    # basisᴴ (G⁰ᵤᵤ)ᵃ basis, 16 columns at a time so the N_u × m_aug image is never held.
    Guu_basis = similar(basis, T, m_aug, m_aug)
    for j0 in 1:16:m_aug
        J = j0:min(j0 + 15, m_aug)
        Guu_basis[:, J] = basis' * opmat(G⁰ᵤᵤ_asym, view(basis, :, J))
    end
    Id = similar(basis, T, m_aug, m_aug)
    copyto!(Id, Matrix{T}(I, m_aug, m_aug)) # dense, so no scalar indexing on the device
    C_basis = (1 / ζ) .* S_basis .+ A_basis .+ Guu_basis
    D_basis = (1 / ζ) .* S_basis .+ A_basis .- (1 / ζ) .* Id
    return (ss_basis=ss_basis, C_basis=C_basis, D_basis=D_basis, W=W)
end

"""
    bounds_from_spectrum(compute_env, smr, Γ, Vur_asym, Γrs; num_pos, k_uu, τs, τ_refine_tol, tau_window, pencil_cache_max, G₀_uu)

The σₙ(Pᵣₛ) bounds from a loaded Asym(G⁰ᵤᵣ) spectrum (`Γ` descending, the leading
`num_pos` columns of `Vur_asym` matching it).

For each `n`, the dual is evaluated on the grid `τs` and refined by golden-section
search down to `τ_refine_tol` (`nothing` keeps the grid minimum). The search is exact
because the dual is quasi-convex in τ: C(τ) is affine in τ, the Lagrangian is jointly
convex in the multipliers, and the dual at τ is its infimum along a ray. Every
evaluation is a valid bound, and the running minimum is kept.

The minimising grid point is piecewise constant in `n`, so after the first index only
`±tau_window` grid points around the previous minimum are swept, falling back to the
full grid when the minimum lands on a window edge. Refinement pencils are cached
(`pencil_cache_max` entries, least recently used evicted), since consecutive indices
on a plateau probe identical τ.

The result is `min(dual, two passivity bounds)` per index: `old` from `Γrs` and `new`
from `Γ`, with `which_bounds` saying which one won (1, 2, 3 = dual).
"""
function bounds_from_spectrum(compute_env::ComputeEnvironment, smr::SMRSystem,
                              Γ::AbstractVector, Vur_asym, Γrs::AbstractVector;
                              num_pos::Int=count(>(0), Γ),
                              k_uu::Int=DEFAULT_K_UU,
                              τs::AbstractVector{<:Real}=range(0.0, 1.0, length=5),
                              τ_refine_tol::Union{Nothing,Real}=0.05,
                              tau_window::Int=2,
                              pencil_cache_max::Int=16,
                              G₀_uu=nothing)
    isempty(τs) && throw(ArgumentError("τs must contain at least one grid point"))
    all(τ -> 0 <= τ <= 1, τs) && issorted(τs) || throw(ArgumentError("τs must be sorted and lie in [0, 1], got $(τs)"))
    isnothing(τ_refine_tol) || τ_refine_tol > 0 || throw(ArgumentError("τ_refine_tol must be positive or nothing"))
    k_uu >= 0 || throw(ArgumentError("k_uu must be non-negative, got $k_uu"))
    1 <= num_pos <= length(Γ) && all(>(0), view(Γ, 1:num_pos)) ||
        throw(ArgumentError("the leading num_pos = $(num_pos) Γ must be positive"))

    sender_size = dof_length(sender_mesh(smr))
    N_u = sender_size + dof_length(receiver_mesh(smr))
    size(Vur_asym, 1) == N_u || error("the basis has $(size(Vur_asym, 1)) rows but the universe [sender; receiver] has $(N_u)")
    G⁰ᵤᵤ_asym = isnothing(G₀_uu) ? asym_green_map(compute_env, smr, Design) : LinearMap(G₀_uu)
    χ = susceptibility(smr)
    ζ = abs(χ)^2 / imag(χ)
    Γ_pos = Float64.(Γ[1:num_pos])
    gs_pos = size(Vur_asym, 2) == num_pos ? Vur_asym : Vur_asym[:, 1:num_pos]
    @info string(now()) * " [bounds::bounds_from_spectrum] χ = $χ, ζ = $ζ, m = $(num_pos), N_u = $(N_u)"

    # The augmentation, clamped to what the universe has left and to what the card fits.
    k_uu = min(k_uu, N_u - num_pos)
    augmentation = (k_uu=0, oversamples=0, num_uu_kept=0, num_uu_dropped=0,
                    uu_values=Float64[], uu_residuals=Float64[])
    basis = gs_pos
    if k_uu > 0
        budget = use_gpu(compute_env) ? device_budget_bytes() - gila_workspace_bytes(N_u) : nothing
        fit = fit_augmentation(N_u, num_pos, k_uu, budget)
        uu = uu_eigenbasis(compute_env, G⁰ᵤᵤ_asym, fit.k_uu; oversamples=fit.oversamples)
        aug = augmented_basis(gs_pos, uu.vectors)
        basis = aug.basis
        augmentation = (k_uu=fit.k_uu, oversamples=fit.oversamples, num_uu_kept=aug.num_uu_kept,
                        num_uu_dropped=aug.num_uu_dropped, uu_values=uu.values, uu_residuals=uu.residuals)
        uu = aug = nothing
        run_gc()
        use_gpu(compute_env) && CUDA.reclaim()
    end
    m_aug = size(basis, 2)
    @info string(now()) * " [bounds::bounds_from_spectrum] Basis: m_aug = $(m_aug) ($(num_pos) g + $(augmentation.num_uu_kept) Asym(G⁰ᵤᵤ), $(augmentation.num_uu_dropped) dropped)"

    front = bounds_front_end(gs_pos, basis, Γ_pos, ζ, G⁰ᵤᵤ_asym, sender_size)
    basis = nothing
    C_basis, D_basis, ss_basis, W = front.C_basis, front.D_basis, front.ss_basis, front.W

    build_pencil(τ::Real) = try
        psd_pencil_whitener(isone(τ) ? C_basis : C_basis .- (1 - τ) .* D_basis)
    catch err
        @warn string(now()) * " [bounds::bounds_from_spectrum] psd_pencil_whitener failed at τ=$τ; skipping it" exception = err
        nothing
    end
    pencils = [build_pencil(τ) for τ in τs]
    usable_τ = findall(!isnothing, pencils)
    isempty(usable_τ) && error("psd_pencil_whitener failed at every τ in $(collect(τs))")
    @info string(now()) * " [bounds::bounds_from_spectrum] C(τ) numerical ranks: " *
        join(("τ=$(τs[i]) → $(pencils[i].rank)/$(m_aug)" for i in usable_τ), ", ")

    pencil_cache = Pair{Float64,Any}[] # least recently used first
    cached_pencil(τ_raw::Real) = begin
        τ = Float64(τ_raw)
        grid_hit = findfirst(i -> Float64(τs[i]) == τ, usable_τ)
        isnothing(grid_hit) || return pencils[usable_τ[grid_hit]]
        hit = findfirst(entry -> first(entry) == τ, pencil_cache)
        if !isnothing(hit)
            entry = popat!(pencil_cache, hit)
            push!(pencil_cache, entry)
            return last(entry)
        end
        pencil = build_pencil(τ)
        if pencil_cache_max > 0 && !isnothing(pencil)
            length(pencil_cache) >= pencil_cache_max && popfirst!(pencil_cache)
            push!(pencil_cache, τ => pencil)
        end
        return pencil
    end

    bounds_dual_basis = zeros(Float64, num_pos)
    bounds_dual_by_tau = fill(NaN, num_pos, length(τs))
    opt_taus = fill(NaN, num_pos)
    prev_best_grid_idx = 0
    for n in 1:num_pos
        idx = n:num_pos
        c_n = similar(C_basis, real(eltype(C_basis)), length(idx))
        copyto!(c_n, sqrt.((4 / ζ) .* Γ_pos[idx]))
        B_factor = FactoredB(W[:, idx], c_n)

        best_dual, best_τ, best_grid_idx = Inf, NaN, 0
        eval_dual(pencil, τ) = isnothing(pencil) ? Inf : try
            maximum(factored_probe_duals(pencil, B_factor, ss_basis, n, num_pos; τ=τ).duals)
        catch err
            @warn string(now()) * " [bounds::bounds_from_spectrum] [$n/$(num_pos)] τ=$τ failed; dropping it" exception = err
            Inf
        end
        sweep_grid!(idxs) = for i in idxs
            dual_τ = eval_dual(pencils[i], τs[i])
            isfinite(dual_τ) || continue
            bounds_dual_by_tau[n, i] = sqrt(dual_τ)
            dual_τ < best_dual && ((best_dual, best_τ, best_grid_idx) = (dual_τ, Float64(τs[i]), i))
        end

        windowed = tau_window > 0 && prev_best_grid_idx > 0
        if windowed
            lo_i = max(firstindex(τs), prev_best_grid_idx - tau_window)
            hi_i = min(lastindex(τs), prev_best_grid_idx + tau_window)
            sweep_grid!(i for i in usable_τ if lo_i <= i <= hi_i)
            # A minimum on an interior window edge may hide a lower point outside it.
            windowed = isfinite(best_dual) &&
                       !(best_grid_idx == lo_i && lo_i > firstindex(τs)) &&
                       !(best_grid_idx == hi_i && hi_i < lastindex(τs))
        end
        if !windowed
            best_dual, best_τ, best_grid_idx = Inf, NaN, 0
            sweep_grid!(usable_τ)
        end
        isfinite(best_dual) || error("every τ in the grid failed at n=$n")

        if !isnothing(τ_refine_tol) && length(τs) > 1
            lo = Float64(τs[max(best_grid_idx - 1, firstindex(τs))])
            hi = Float64(τs[min(best_grid_idx + 1, lastindex(τs))])
            invφ = (sqrt(5.0) - 1) / 2
            τ₁, τ₂ = hi - invφ * (hi - lo), lo + invφ * (hi - lo)
            g₁, g₂ = eval_dual(cached_pencil(τ₁), τ₁), eval_dual(cached_pencil(τ₂), τ₂)
            g₁ < best_dual && ((best_dual, best_τ) = (g₁, τ₁))
            g₂ < best_dual && ((best_dual, best_τ) = (g₂, τ₂))
            for _ in 1:200
                hi - lo > τ_refine_tol || break
                if g₁ <= g₂
                    hi, τ₂, g₂ = τ₂, τ₁, g₁
                    τ₁ = hi - invφ * (hi - lo)
                    g₁ = eval_dual(cached_pencil(τ₁), τ₁)
                    g₁ < best_dual && ((best_dual, best_τ) = (g₁, τ₁))
                else
                    lo, τ₁, g₁ = τ₁, τ₂, g₂
                    τ₂ = lo + invφ * (hi - lo)
                    g₂ = eval_dual(cached_pencil(τ₂), τ₂)
                    g₂ < best_dual && ((best_dual, best_τ) = (g₂, τ₂))
                end
            end
        end
        @info string(now()) * " [bounds::bounds_from_spectrum] [$n/$(num_pos)] bound $(sqrt(best_dual)) on σₙ(Pᵣₛ) at τ = $best_τ"
        bounds_dual_basis[n] = sqrt(best_dual)
        opt_taus[n] = best_τ
        prev_best_grid_idx = best_grid_idx
    end

    # Passivity bounds: from κ = ζ²Γrs² (old form) and κ̃ = ζΓ (new form, Eq. (19)).
    old_form(κ) = κ >= 1 ? 1.0 : sqrt(4κ) / (1 + κ)
    new_form(κ̃) = 2κ̃ >= 1 ? 1.0 : sqrt(4κ̃ * abs(1 - κ̃))
    old_analytical_bounds = old_form.(ζ^2 .* Γrs .^ 2)
    new_analytical_bounds = new_form.(ζ .* max.(Γ, 0))
    old_analytical_bounds[old_analytical_bounds .<= 0] .= NaN
    new_analytical_bounds[new_analytical_bounds .<= 0] .= NaN
    ks = 1:min(length(old_analytical_bounds), length(new_analytical_bounds), num_pos)
    candidates(k) = (old_analytical_bounds[k], new_analytical_bounds[k], bounds_dual_basis[k])
    which_bounds = [argmin(candidates(k)) for k in ks]
    true_bounds = [candidates(k)[which_bounds[k]] for k in ks]

    return (num_pos=num_pos, bounds_dual_basis=bounds_dual_basis,
            tau_grid=collect(Float64, τs), opt_taus=opt_taus, bounds_dual_by_tau=bounds_dual_by_tau,
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
    result = bounds_from_spectrum(compute_env, smr, inputs.Γ, inputs.Vur_asym, inputs.Γrs;
                                  num_pos=inputs.num_pos, k_uu=k_uu)

    # Through a temporary file, so a failed rerun never destroys a finished result.
    tmp_path = "$(out_path).tmp.$(getpid())"
    jldopen(tmp_path, "w") do jld
        jld["χ"] = susceptibility(smr)
        jld["Γ"] = Array(inputs.Γ)
        jld["Γrs"] = Array(inputs.Γrs)
        jld["ordering_idxs"] = inputs.sorted_idxs
        for key in (:bounds_dual_basis, :tau_grid, :opt_taus, :bounds_dual_by_tau,
                    :old_analytical_bounds, :new_analytical_bounds, :which_bounds, :true_bounds)
            jld[String(key)] = getfield(result, key)
        end
        aug = result.augmentation
        jld["augment/augmented"] = aug.k_uu > 0
        jld["augment/k_uu_requested"] = k_uu
        jld["augment/k_uu_effective"] = aug.k_uu
        jld["augment/m"] = result.num_pos
        jld["augment/m_aug"] = result.m_aug
        jld["augment/num_uu_kept"] = aug.num_uu_kept
        jld["augment/num_uu_dropped"] = aug.num_uu_dropped
        jld["augment/uu_oversamples"] = aug.oversamples
        jld["augment/uu_values"] = aug.uu_values
        jld["augment/uu_residuals"] = aug.uu_residuals
    end
    mv(tmp_path, out_path; force=true)
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
