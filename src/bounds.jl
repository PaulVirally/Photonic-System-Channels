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
    # and negative. Halving rather than growing takes the largest such offset,
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

Eigendecompose the constraint matrix `C` of `Bv = λCv` and return what is needed
to solve the pencil on `C`'s numerical range: the `whitener`
`W = U₊ diag(μ₊)^(-1/2)` over the eigenpairs above `tol = rtol · μmax`, the
`nullspace` basis `N` below it, and `values`, `tol` and `rank`.

`W' C W = 1`, so solving `(W'BW)y = λy` and taking `v = Wy` gives eigenpairs
normalized as `V' C V = 1`, which is the normalization the dual's
`∑ⱼ |bⱼ|²/(α - λⱼ)` resolvent expansion assumes. `N` comes back so the caller can
check that whatever it drops along with it is genuinely absent.

Every member of the constraint family `C(τ) = ζ⁻¹(Πₛ + (1−τ)Πᵣ) + τ(−G⁰ᵤᵣ)ᵃ₊ +
(G⁰ᵤᵤ)ᵃ`, `τ ∈ [0, 1]`, is a sum of positive semi-definite terms, so it is never
indefinite in exact arithmetic. An eigenvalue below `-tol` is a wrong sign
upstream rather than roundoff, and is an error.

The work stays in `C`'s array space. On the host `eigen` raises on a failed
factorization; on the device CUSOLVER's `heevd` returns `NaN`/`Inf` eigenvalues
instead, which is what the non-finite check below is for.
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
if instead `v'Bv > 0` the constraint fails to bound that direction and the
program is unbounded. Projecting the second case away would report a finite bound
that the program does not support.
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

Solve the per-probe dual problems for index `n` on one eigendecomposed constraint
pencil (the output of `psd_pencil_whitener`). For each probe `k ∈ n:num_pos` this
brackets and solves for the stationary multiplier `αₖ`, then evaluates
`αₖ²/4 ∑ⱼ |bⱼ|²/(αₖ - λⱼ)` with `b = V'sₖ` in the pencil's `C`-orthonormal
eigenbasis.

Returns `(ks, alphas, duals)`; this pencil's contribution to the bound is
`maximum(duals)`. The per-probe records are kept so `verify_bounds` can reuse the
same probes in the full space, seeded with the multipliers found here.
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

# ---------------------------------------------------------------------------
# The augmented projection basis
#
# Production projects every operator into the span of the kept positive-Γ
# eigenvectors of Asym(G⁰ᵤᵣ) (the "g basis", `m` columns). That span collapses
# with separation -- at the 1 λ cube sweep, s = 357/32 λ keeps m = 15 of
# N_u = 196,608 -- and the projected dual then loses validity: it under-reports,
# and the exact λ/4 sweep, whose domain is strictly *contained* in the 1 λ one,
# comes out above it at the same gap (0.042178 against 0.017831). A bound that
# shrinks when the design space grows is not a bound.
#
# The diagnosis is that the *constraint* operators cannot be represented in a
# 15-dimensional span -- above all Asym(G⁰ᵤᵤ), the universe's radiative
# self-interaction, whose rank does not collapse with separation. Once PᵀCP stops
# implying full-space feasibility, the "dual" relaxes nothing in particular.
#
# The repair, proven on production data by bench/augmented_basis_experiment.jl, is
# to augment the projection basis with the top `k_uu` eigenvectors of Asym(G⁰ᵤᵤ):
#
#     basis = orthonormalize([g_kept, U_uu[:, 1:k_uu]])
#
# At the 1 λ, s = 357//32 point (m = 15) that took channel 1 from 0.0178 to 0.2019
# -- crossing the exact λ/4 reference 0.0422 at k_uu = 128 -- and the trace over
# the 15 channels from 0.050 to 0.740, monotonically and saturating. `k_uu = 0`
# reproduces the pure-g production numbers to more than four digits, and this code
# reproduces them bit for bit (the branch is not taken at all).
#
# Monotonicity is arithmetic, not physics: the projected primal is a sup over
# t ∈ range(P), so a larger subspace gives a larger value, and the k_uu sweep uses
# leading subsets of one U_uu, so the subspaces are nested.
#
# Two things production exploits hold *only* in the pure g basis: Bₙ and
# basisᴴ(−G⁰ᵤᵣ)ᵃ₊basis are diagonal there. In the augmented basis they are not, but
# both are built out of the gₗ alone, so everything still reduces to one extra
# small matrix, W = basisᴴ gs_pos (m_aug × m):
#
#     basisᴴ (−G⁰ᵤᵣ)ᵃ₊ basis = W diag(Γ) Wᴴ
#     basisᴴ Bₙ basis        = W[:, n:m] diag(4Γ[n:m]/ζ) W[:, n:m]ᴴ
#
# which is `FactoredB` below, and `factored_pencil_eigen` is `diag_pencil_eigen`
# with the diagonal assumption lifted.
#
# The probes stay the production probes. `ss` is built by `reverse_gram_schmidt!`
# on Πₛ·gs_pos -- the g columns alone, never the augmentation. Column i of `ss`
# spans Πₛ·span(gᵢ, …, g_m) together with the later columns, and the outer loop's
# shrinking probe set k ≥ n is what makes index n mean "the n-th channel". Adding
# the U_uu columns to that construction would change the nested spans, so index n
# would name a different quantity. The augmentation is about representing the
# *constraint* faithfully -- enlarging the feasible set the dual is solved over --
# not about redefining the objective.
# ---------------------------------------------------------------------------

#=
Rank guard on the augmentation, relative to the largest |R_ii| of the
augmentation's own QR. The `U_uu` columns are eigenvectors of a Hermitian
operator, so they are mutually orthonormal to machine precision; the only way one
of them can be numerically dependent after `span(g)` is projected out is if it lay
in `span(g)` to begin with, which can happen for at most `m` of them. So this is
expected to drop nothing, and 1e-10 is far enough above the ~`k·eps` a Householder
QR of 512 near-orthonormal columns loses that a drop means a real dependence
rather than roundoff. Whatever it drops is reported and saved.
=#
const AUG_QR_RTOL = 1e-10

# Matches `diag_pencil_eigen`'s own default, so the null-space admissibility test
# is as strict on the augmented path as it is on the plain one.
const AUG_BTOL = 1e-8

"""
    qr_thin_rdiag!(A) -> (Q, d)

The thin `Q` of a tall `A`, computed in place, together with `d = abs.(diag(R))`
read out of the factored form *before* `orgqr!` overwrites it.

[`qthin!`](@ref) throws `R` away, and `R`'s diagonal is the only cheap handle on
column dependence an unpivoted QR offers: a column that is numerically in the span
of the earlier ones has `|R_ii| ≈ 0`, and the `Q` column generated in its place is
an arbitrary direction orthogonal to the earlier ones rather than anything inside
`span(A)`. [`augmented_basis`](@ref) drops those.
"""
function qr_thin_rdiag!(A::Matrix{T}) where {T<:LinearAlgebra.BlasFloat}
    size(A, 1) >= size(A, 2) || throw(ArgumentError(
        "qr_thin_rdiag! wants a tall matrix, got $(size(A))"))
    A, τ = LinearAlgebra.LAPACK.geqrf!(A)
    d = abs.(diag(A))
    LinearAlgebra.LAPACK.orgqr!(A, τ)
    return A, d
end

function qr_thin_rdiag!(A::CuMatrix)
    size(A, 1) >= size(A, 2) || throw(ArgumentError(
        "qr_thin_rdiag! wants a tall matrix, got $(size(A))"))
    n = size(A, 2)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    # The leading n × n block back on the host (a few MB at n = 512) rather than a
    # device gather along `diagind`: R lives there, and this is a strided copy of a
    # contiguous-column view, which needs nothing clever from CUDA.jl.
    d = abs.(diag(Array(view(A, 1:n, 1:n))))
    CUDA.CUSOLVER.orgqr!(A, τ)
    return A, d
end

"""
    augmented_basis(gs, U; rtol=AUG_QR_RTOL) -> NamedTuple

`[gs, U]` orthonormalized, with the leading `size(gs, 2)` columns left *exactly*
as `gs`.

The `g` block is not touched: it is RSVD output and already orthonormal to about
`1e-14`, which is the level production's own `basisᴴbasis = 1` assumption runs at,
and leaving it alone makes `k_uu = 0` literally the production basis rather than a
rotation of it. `span(g)` is projected out of `U` (classical Gram-Schmidt run
twice, which is as stable as modified Gram-Schmidt and is two GEMMs instead of `m`
passes over `N_u`-vectors), and what is left is orthonormalized by a Householder QR
whose `|diag(R)|` is used to drop numerically dependent columns (see
[`AUG_QR_RTOL`](@ref)).

Returns `(basis, num_uu_kept, num_uu_dropped, dropped_cols, rdiag_min_ratio)`.
`basis` aliases `gs` when `U` is empty.
"""
function augmented_basis(gs::AbstractMatrix, U::AbstractMatrix; rtol::Real=AUG_QR_RTOL)
    N, m = size(gs)
    size(U, 1) == N || throw(DimensionMismatch(
        "the augmentation has $(size(U, 1)) rows but the g basis has $(N)"))
    k = size(U, 2)
    k == 0 && return (basis=gs, num_uu_kept=0, num_uu_dropped=0,
                      dropped_cols=Int[], rdiag_min_ratio=NaN)

    T = eltype(gs)
    U1 = copy(U) # the QR below is in place, and `U` may be a shared block
    # Two classical Gram-Schmidt passes against the g block.
    for _ in 1:2
        coeffs = gs' * U1              # m × k, small
        mul!(U1, gs, coeffs, -one(T), one(T))
    end
    U1, d = qr_thin_rdiag!(U1)

    dmax = maximum(d)
    keep = findall(>=(rtol * dmax), d)
    dropped = setdiff(1:k, keep)
    basis = isempty(dropped) ? hcat(gs, U1) : hcat(gs, U1[:, keep])
    return (basis=basis, num_uu_kept=length(keep), num_uu_dropped=length(dropped),
            dropped_cols=dropped, rdiag_min_ratio=minimum(d) / dmax)
end

"""
    FactoredB(V, c)

`B = V diag(c²) Vᴴ`, held in factored form so that the pencil never builds `B`.

Production's `Bₙ` is diagonal in the `g` basis and [`diag_pencil_eigen`](@ref)
exploits that all the way through. In the augmented basis it is not diagonal, but
it is still a *low-rank congruence of a diagonal*:

    basisᴴ Bₙ basis = W[:, n:m] diag(4Γ[n:m]/ζ) W[:, n:m]ᴴ,   W = basisᴴ gs,

which is this object with `V = W[:, n:m]` and `c = sqrt.(4Γ[n:m]/ζ)`. Everything
[`factored_pencil_eigen`](@ref) needs is a product against the factor
`K = diag(c) Vᴴ`, and `Vᴴ X` is a plain CUBLAS gemm with `transa = 'C'`, so nothing
is ever transposed into a temporary.

With `V = 1` and `c = sqrt.(d)` this reduces to `diag_pencil_eigen`'s
`B = diag(d)` term for term, which `test/augmented_basis.jl` checks.
"""
struct FactoredB{M<:AbstractMatrix,V<:AbstractVector}
    V::M   # m_aug × p
    c::V   # length p, real and non-negative
end

# K * X for K = diag(c) Vᴴ, so that Xᴴ Kᴴ K X = Xᴴ B X. `F.V' * X` is a plain gemm
# with transa = 'C'; nothing is materialized transposed.
apply_factor(F::FactoredB, X::AbstractMatrix) = F.c .* (F.V' * X)

# The diagonal of B = V diag(c²) Vᴴ: Bⱼⱼ = Σᵢ cᵢ² |Vⱼᵢ|². `reshape` rather than `'`
# so the row vector is a plain array on whichever device `c` lives on.
factored_diag(F::FactoredB) =
    vec(sum(abs2.(F.V) .* reshape(F.c .^ 2, 1, :); dims=2))

"""
    factored_pencil_eigen(F, W, N; btol=AUG_BTOL) -> (values, vectors)

Solve `B v = λ C v` for `B = F.V diag(F.c²) F.Vᴴ`, given the `whitener` `W` and
`nullspace` `N` of [`psd_pencil_whitener`](@ref). Returns `(values, vectors)` with
`vectorsᴴ C vectors = 1`, exactly the normalization the dual's
`∑ⱼ |bⱼ|²/(α − λⱼ)` resolvent expansion assumes.

This is [`diag_pencil_eigen`](@ref) with the diagonal assumption lifted, and it is
the same computation term for term:

  * `Wd = K W` here, `sqrt_d .* W` there — identical when `K = diag(sqrt(d))`;
  * `eigen(Hermitian(Wdᴴ Wd))` in both, which is what keeps `B`'s compression
    positive semi-definite by construction rather than by luck;
  * the null-space admissibility test is the same `max_j Nⱼᴴ B Nⱼ ≤ btol · dmax`.
    `dmax` is the largest *diagonal entry* of `B`, which equals `max(d)` in the
    diagonal case and otherwise sits within a factor of `size(B, 1)` of `λmax(B)`;
    using the diagonal makes the test stricter, so it fails loudly rather than
    passing quietly.

Errors unless `B` is negligible on the numerical null space of `C`. The reasoning
is `diag_pencil_eigen`'s: for positive semi-definite `B` and `C`, a null direction
`v` of `C` with `vᴴBv ≈ 0` also has `Bv ≈ 0` and can be dropped, while one with
`vᴴBv > 0` is a direction the constraint fails to bound, i.e. a bound of `+∞` that
no projection may quietly report as finite.
"""
function factored_pencil_eigen(F::FactoredB, W::AbstractMatrix, N::AbstractMatrix;
                               btol::Real=AUG_BTOL)
    all(>=(0), F.c) || throw(ArgumentError(
        "FactoredB's weights must be non-negative (they are square roots), got " *
        "minimum(c) = $(minimum(F.c))"))
    d = Array(factored_diag(F))
    dmax = isempty(d) ? zero(eltype(d)) : maximum(d)
    if size(N, 2) > 0 && dmax > zero(dmax)
        worst = maximum(sum(abs2, apply_factor(F, N); dims=1))
        worst <= btol * dmax || error("factored_pencil_eigen: B is not negligible on " *
            "the numerical null space of C: max_j N[:,j]'B N[:,j] = $worst vs " *
            "btol * max(diag(B)) = $(btol * dmax) over $(size(N, 2)) null direction(s). " *
            "The constraint does not bound those directions, so the true bound for this " *
            "index is +∞ rather than anything this program can report")
    end
    Wd = apply_factor(F, W) # Wdᴴ Wd == Wᴴ B W, positive semi-definite by construction
    E = eigen(Hermitian(Wd' * Wd))
    Λ = Array(E.values)
    num_bad = count(!isfinite, Λ)
    num_bad == 0 || error("factored_pencil_eigen: eigendecomposition returned " *
        "$num_bad/$(length(Λ)) non-finite eigenvalues; CUSOLVER's heevd reports " *
        "failure this way instead of throwing, so the factorization did not converge")
    return Λ, W * E.vectors
end

"""
    factored_probe_duals(pencil, F, ss_basis, n, num_pos; τ=NaN)

[`pencil_probe_duals`](@ref) with [`diag_pencil_eigen`](@ref) swapped for
[`factored_pencil_eigen`](@ref). Same probes, same bracketing, same stationarity
condition, same `αₖ²/4 ∑ⱼ |bⱼ|²/(αₖ − λⱼ)`; only the `B` representation differs.
The root find itself is [`bracket_root`](@ref) plus `Roots.Brent`, reused verbatim.
"""
function factored_probe_duals(pencil, F::FactoredB, ss_basis::AbstractMatrix,
                              n::Int, num_pos::Int; τ::Real=NaN)
    Λ_basis, V_basis = factored_pencil_eigen(F, pencil.whitener, pencil.nullspace)

    ks = collect(n:num_pos)
    alphas = zeros(Float64, length(ks))
    duals = zeros(Float64, length(ks))
    for (i, k) in enumerate(ks)
        sₖ = view(ss_basis, :, k)
        if size(pencil.nullspace, 2) > 0
            sₖ_null = norm(pencil.nullspace' * sₖ)
            sₖ_null <= 1e-8 * norm(sₖ) || error("probe k=$k at n=$n, τ=$τ has " *
                "‖N'sₖ‖ = $sₖ_null of ‖sₖ‖ = $(norm(sₖ)) in the numerical null space " *
                "of C(τ), so the bound is +∞")
        end
        b = Array(V_basis' * sₖ)

        f(α) = sum(abs2(bⱼ) * (α - 2λⱼ) / (α - λⱼ)^2 for (bⱼ, λⱼ) in zip(b, Λ_basis))
        ((left, _), (right, _)) = bracket_root(f, Λ_basis, b)
        αₖ = find_zero(f, (left, right), Roots.Brent())
        alphas[i] = αₖ
        duals[i] = αₖ^2 / 4 * sum(abs2(bⱼ) / (αₖ - λⱼ) for (bⱼ, λⱼ) in zip(b, Λ_basis))
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

    sender_size = dof_length(sender_mesh(smr))
    receiver_size = dof_length(receiver_mesh(smr))
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

# --- Sizing the augmented front end -----------------------------------------

"""
    DEFAULT_K_UU, DEFAULT_AUGMENT_THRESHOLD

The production defaults for the `Asym(G⁰ᵤᵤ)` augmentation, exposed as `--k-uu` and
`--augment-threshold`.

`k_uu = 512` is where the 1 λ sweep's `k_uu` scan saturated (0.1988 at 256 →
0.2019 at 512, a 1.6% last step, against a 11× rise from `k_uu = 0`).

`augment_threshold = 1000` is where the sweep is cut into an augmented far half and
an untouched near half. The boundary is a policy, not a physical edge: augmenting
only ever moves a bound *towards* validity, so the question is not "where does the
pathology start?" but "where is the step in the reported trace small enough that
the two halves can be plotted together?". The q-validation answered it. At the
m = 500 boundary the same point reported a trace 2.5% apart augmented versus not,
and the kept count itself jitters by about ±30 between reruns of the RSVD, which
flips points across the boundary and puts that 2.5% into the sweep as noise. At
m = 1000 the same comparison is well inside the line width, because a basis that
keeps a thousand directions of `Asym(G⁰ᵤᵣ)` already represents the constraint.

It is still bounded well away from the near field. `m_aug² ` pencils at the
near-contact `m = 4000` would grow by 25% for nothing, which is what the threshold
exists to prevent, and 1000 is a quarter of the way there.

Raising the threshold does raise the ceiling on the dense augmented front end,
from `m_aug < 1012` to `m_aug < 1512`, and at the larger universes that ceiling is
not free: see [`max_k_uu_for_budget`](@ref) and [`clip_k_uu`](@ref), which reduce
the effective `k_uu` on a point whose augmented front end would not fit the card.

`--k-uu 0` disables the augmentation entirely and reproduces the pre-augmentation
output bit for bit: the branch below is simply not taken.
"""
const DEFAULT_K_UU = 512
const DEFAULT_AUGMENT_THRESHOLD = 1000

"""
    UU_OVERSAMPLES, UU_POWER_ITERS, UU_MIN_OVERSAMPLES

`reigen_hermitian` parameters for the `Asym(G⁰ᵤᵤ)` solve.

`Asym(G⁰ᵤᵤ)` is the universe's radiation operator, positive semi-definite with a
spectrum that falls off a cliff past the radiative channel count of a two-cube
universe. Well-separated eigenvalues are the easy case for subspace iteration --
the production RSVD's 14 iterations exist for the *clustered* spectrum of
`Asym(G⁰ᵤᵣ)` at large separation, which this operator is not -- so 4 power
iterations is the default here, and the measured 1 λ residuals `‖Av − λv‖/Λ₁` came
back at the 1e-13 level with it.

`UU_MIN_OVERSAMPLES` is the floor [`plan_uu_solve`](@ref) may cut the oversamples
down to when the sketch does not fit the device at the full 50. Below that the
range finder has no slack left at all and the tail of the returned block stops
being an eigenbasis.
"""
const UU_OVERSAMPLES = 50
const UU_POWER_ITERS = 4
const UU_MIN_OVERSAMPLES = 10

"""
    uu_sketch_bytes(N_u, k_uu, oversamples) -> Int

Device bytes the in-memory `reigen_hermitian` for `Asym(G⁰ᵤᵤ)` holds live: the
three `N_u × (k_uu + oversamples)` ComplexF64 matrices the Hermitian range finder
keeps at once (the sketch, its image, and the rotation's destination). Same count
as `rsvd_inmemory_live_bytes` in the cost model and as [`use_panel_path`](@ref)'s,
so the three cannot drift apart.
"""
uu_sketch_bytes(N_u::Integer, k_uu::Integer, oversamples::Integer) =
    3 * Int(N_u) * (Int(k_uu) + Int(oversamples)) * 16

"""
    augmented_footprint_bytes(N_u, m, m_aug) -> Int

Device bytes the *augmented* in-memory front end wants at its peak: the
`N_u × m_aug` basis, the `N_u × m` probes `ss` (built from the `g` columns alone,
so `m` and not `m_aug` wide) and the `N_u × m_aug` destination `opmat(C, basis)`
allocates. Reduces to [`bounds_footprint_bytes`](@ref) when `m_aug == m`.
"""
augmented_footprint_bytes(N_u::Integer, m::Integer, m_aug::Integer) =
    (2 * Int(m_aug) + Int(m)) * Int(N_u) * 16

"""
    K_UU_CLIP_FLOOR

The smallest `k_uu` [`clip_k_uu`](@ref) will clip *to*. Below this there is no
point augmenting: the 1 λ `k_uu` scan reached the λ/4 reference at `k_uu = 128`
and was still an order of magnitude short of it at `k_uu = 32`, so a basis
augmented with 64 directions is paying the whole `Asym(G⁰ᵤᵤ)` solve and the
`m_aug`-wide pencil stage for a bound that is still invalid. A point that cannot
afford 64 is refused, loudly and in the first minute, rather than run to produce a
number nobody can use.
"""
const K_UU_CLIP_FLOOR = 64

"""
    max_k_uu_for_budget(N_u, m, budget_bytes) -> Int

The largest `k_uu` whose augmented front end fits `budget_bytes` on an `N_u`-tall
universe at kept width `m`. May be negative, which means the *unaugmented* front
end does not fit either and no `k_uu` rescues it.

Pure arithmetic on the same three quantities [`plan_uu_solve`](@ref) checks,
solved for `k` instead of tested at a given `k`. With `c = budget / (N_u · 16)`,
the tall columns the budget pays for:

  * the front end, [`augmented_footprint_bytes`](@ref) `= (2 m_aug + m) N_u 16`,
    which at `m_aug = m + k` is `3m + 2k` columns, so `k ≤ (c − 3m) / 2`;
  * the augmentation's QR, `(2 k_uu + m_aug) N_u 16 = m + 3k` columns, so
    `k ≤ (c − m) / 3`;
  * the range finder's fudged peak at the *minimum* oversamples,
    `RSVD_PEAK_FUDGE · 3 (k + p) N_u 16`, so
    `k ≤ c / (3 RSVD_PEAK_FUDGE) − UU_MIN_OVERSAMPLES`.

`UU_MIN_OVERSAMPLES` and not the requested `p` in the third, on purpose: the
oversamples are the range finder's slack and `plan_uu_solve` already spends them
before it gives up, so a `k` that fits at the floor is a `k` that `plan_uu_solve`
can make room for. Cutting `k` is the more expensive repair of the two and goes
second.

Kept as a function of an explicit budget rather than of a `ComputeEnvironment` so
that `bench/cost_model.jl`'s `augment_k_uu_cap` can mirror it line for line and
`test/augmented_basis.jl` can check the two against each other on a grid of
fabricated sizes. If this changes, change that.
"""
function max_k_uu_for_budget(N_u::Integer, m::Integer, budget_bytes::Real)
    column_bytes = Int(N_u) * 16
    columns = floor(Int, budget_bytes / column_bytes)
    k_front = fld(columns - 3 * Int(m), 2)
    k_qr = fld(columns - Int(m), 3)
    k_sketch = floor(Int, columns / (3 * RSVD_PEAK_FUDGE)) - Int(UU_MIN_OVERSAMPLES)
    return min(k_front, k_qr, k_sketch)
end

"""
    clip_k_uu(compute_env, N_u, m, k_uu) -> NamedTuple

The `k_uu` this point can actually afford, given the card it is on. Returns
`(k_uu, requested, clipped, reason, k_fit, budget_bytes)`; errors when not even
[`K_UU_CLIP_FLOOR`](@ref) fits.

Why this exists. `--augment-threshold` caps `m` and therefore caps
`m_aug = m + k_uu`, and at 500 the cap was low enough that the dense augmented
front end fitted everywhere the sweep ran. At 1000 it is not: `m_aug` reaches 1512,
and three `N_u × m_aug` matrices at the 4 λ universe (`N_u = 786,432`, 12 MiB a
column) are tens of gigabytes. The threshold governs *which* points augment, which
is a question about the physics; this governs *how far* one of them can augment,
which is a question about the card, and conflating the two would mean lowering the
threshold globally because one wavelength on one cluster cannot pay for it.

Clipping and not refusing, because the augmentation is worth having partially. The
1 λ scan is monotone and saturating in `k_uu` (0.0178 at 0, 0.1988 at 256, 0.2019
at 512), so a point clipped from 512 to 256 keeps 98% of the repair. A point
clipped below [`K_UU_CLIP_FLOOR`](@ref) keeps almost none of it, and is refused
with the arithmetic in the message, exactly as `plan_uu_solve` refuses a sketch
that cannot be made to fit.

A CPU run is not clipped: it has no device budget to clip against, and the systems
that run on one are the test fixtures.
"""
function clip_k_uu(compute_env::ComputeEnvironment, N_u::Integer, m::Integer,
                   k_uu::Integer)
    k_uu = Int(k_uu)
    unclipped = (k_uu=k_uu, requested=k_uu, clipped=false, reason="",
                 k_fit=k_uu, budget_bytes=0)
    (k_uu <= 0 || !use_gpu(compute_env)) && return unclipped

    budget = device_budget_bytes() - gila_workspace_bytes(N_u)
    budget > 0 || error("clip_k_uu: the Green operator's own device workspace " *
        "($(gila_workspace_bytes(N_u)) bytes) already exceeds 90% of this card " *
        "($(device_budget_bytes()) bytes); nothing can run here, augmented or not")

    k_fit = max_k_uu_for_budget(N_u, m, budget)
    k_fit >= k_uu && return merge(unclipped, (k_fit=k_fit, budget_bytes=budget))

    gib(x) = round(x / 2^30; digits=2)
    k_fit >= K_UU_CLIP_FLOOR || error("clip_k_uu: the augmented front end at " *
        "N_u = $(N_u), kept m = $(m) does not fit this device at any useful " *
        "--k-uu. The budget is $(gib(budget)) GiB (90% of the card, " *
        "$(gib(device_budget_bytes())) GiB, less the Green operator's " *
        "$(gib(gila_workspace_bytes(N_u))) GiB workspace), which buys " *
        "$(fld(budget, Int(N_u) * 16)) columns of $(N_u) × 1 ComplexF64 " *
        "($(gib(Int(N_u) * 16)) GiB each). The three tall matrices are " *
        "2·m_aug + m = 3m + 2k columns, the augmentation's QR is m + 3k, and the " *
        "range finder's fudged sketch at $(UU_MIN_OVERSAMPLES) oversamples is " *
        "3·RSVD_PEAK_FUDGE·(k + $(UU_MIN_OVERSAMPLES)); the largest k satisfying " *
        "all three is $(k_fit), below the K_UU_CLIP_FLOOR of $(K_UU_CLIP_FLOOR) " *
        "at which augmenting stops buying a valid bound. At m_aug = m + " *
        "$(K_UU_CLIP_FLOOR) the front end alone would want " *
        "$(gib(augmented_footprint_bytes(N_u, m, Int(m) + K_UU_CLIP_FLOOR))) GiB. " *
        "Run this point on a larger card, lower --augment-threshold below $(m) so " *
        "it is not augmented at all, or set --k-uu 0")

    @warn string(now()) * " [bounds_bargaining::clip_k_uu] Clipping --k-uu $(k_uu) → $(k_fit) to fit this device: at N_u = $(N_u), kept m = $(m), the budget of $(gib(budget)) GiB (90% of the $(gib(device_budget_bytes())) GiB card less the Green operator's $(gib(gila_workspace_bytes(N_u))) GiB workspace) buys $(fld(budget, Int(N_u) * 16)) tall columns, and k_uu = $(k_uu) would want $(2 * (Int(m) + k_uu) + Int(m)) of them for the front end alone ($(gib(augmented_footprint_bytes(N_u, m, Int(m) + k_uu))) GiB against $(gib(budget)) GiB). At k_uu = $(k_fit) the front end is $(gib(augmented_footprint_bytes(N_u, m, Int(m) + k_fit))) GiB. The k_uu scan is monotone and saturating, so this keeps most of the repair, but the bound at this point is NOT the one a --k-uu $(k_uu) run elsewhere in the sweep reports; augment/k_uu_effective and augment/k_uu_clip_reason in the output record it"

    return (k_uu=k_fit, requested=k_uu, clipped=true, reason="device_budget",
            k_fit=k_fit, budget_bytes=budget)
end

"""
    plan_uu_solve(compute_env, N_u, m, k_uu; oversamples, power_iters) -> NamedTuple

Decide whether the `Asym(G⁰ᵤᵤ)` solve and the basis it produces fit on the device,
and with how many oversamples. Returns `(oversamples, sketch_bytes, peak_bytes,
basis_bytes, budget_bytes)`; errors, loudly and with the arithmetic in the message,
when no admissible configuration exists.

Three quantities are checked against 90% of the card
([`device_budget_bytes`](@ref)), less the Green operator's own workspace
([`gila_workspace_bytes`](@ref)), which the residency machinery holds back
everywhere else for the same reason:

  * the range finder's peak, `RSVD_PEAK_FUDGE` × [`uu_sketch_bytes`](@ref). The
    fudge is the measured ratio of the real high-water to the three nominal
    matrices, and it is the same predicate [`use_panel_path`](@ref) uses, so a
    point the RSVD stage would have called in-memory is called in-memory here too.
  * the augmentation's own QR, which holds `U_uu`, the Gram-Schmidt copy of it and
    the concatenated basis at once, i.e. about `2 k_uu + m_aug` columns.
  * the front end's peak, [`augmented_footprint_bytes`](@ref).

Only the first is negotiable *here*. When it does not fit, the oversamples come
down (the sketch is `k_uu + p` wide and `p` is pure slack for the range finder, so
cutting it costs accuracy in the tail of `U_uu` and nothing else) as far as
[`UU_MIN_OVERSAMPLES`](@ref). Below that, and for the other two, this errors: the
augmented path has no panel front end (see `_bounds_front_end_augmented`), and an
OOM three hours into a bounds job is a worse outcome than a refusal in the first
minute.

Those errors are now a backstop rather than the first line of defence.
[`clip_k_uu`](@ref) runs before this and has already brought `k_uu` down to a value
that satisfies all three checks at `UU_MIN_OVERSAMPLES`, so what reaches here is a
`k_uu` known to fit; the remaining work is spending the oversamples. The checks
stay because `plan_uu_solve` is called directly by tests and by
`bench/augmented_basis_experiment.jl`, and because a guard that is never supposed
to fire is exactly the one worth keeping.

At 4 λ (`N_u = 786,432`) with `k_uu = 512`: the sketch is 7.1 GB per matrix, 21.2
GB for the three, 33.0 GB fudged, against 38.2 GB of budget on an A100-40. It fits,
but with only 5 GB to spare, which is why this function exists rather than a
comment.

A CPU run is not checked: it has no device budget to check against, the systems
that run on one are the test fixtures, and the host side is Slurm's problem rather
than the plan's.
"""
function plan_uu_solve(compute_env::ComputeEnvironment, N_u::Integer, m::Integer,
                       k_uu::Integer; oversamples::Integer=UU_OVERSAMPLES,
                       power_iters::Integer=UU_POWER_ITERS)
    m_aug = Int(m) + Int(k_uu)
    basis_bytes = augmented_footprint_bytes(N_u, m, m_aug)
    qr_bytes = (2 * Int(k_uu) + m_aug) * Int(N_u) * 16
    if !use_gpu(compute_env)
        return (oversamples=Int(oversamples), power_iters=Int(power_iters),
                sketch_bytes=uu_sketch_bytes(N_u, k_uu, oversamples),
                peak_bytes=0, basis_bytes=basis_bytes, qr_bytes=qr_bytes,
                budget_bytes=0)
    end

    budget = device_budget_bytes() - gila_workspace_bytes(N_u)
    budget > 0 || error("plan_uu_solve: the Green operator's own device workspace " *
        "($(gila_workspace_bytes(N_u)) bytes) already exceeds 90% of this card " *
        "($(device_budget_bytes()) bytes); nothing can run here, augmented or not")

    gib(x) = round(x / 2^30; digits=2)
    fits(p) = RSVD_PEAK_FUDGE * uu_sketch_bytes(N_u, k_uu, p) <= budget

    p = Int(oversamples)
    if !fits(p)
        p_ok = nothing
        for candidate in (Int(oversamples) - 1):-1:Int(UU_MIN_OVERSAMPLES)
            fits(candidate) && (p_ok = candidate; break)
        end
        p_ok === nothing && error("plan_uu_solve: the Asym(G⁰ᵤᵤ) solve for k_uu = " *
            "$(k_uu) at N_u = $(N_u) does not fit this device even at the minimum " *
            "$(UU_MIN_OVERSAMPLES) oversamples: the range finder's three " *
            "$(N_u) × $(Int(k_uu) + Int(UU_MIN_OVERSAMPLES)) matrices are " *
            "$(gib(uu_sketch_bytes(N_u, k_uu, UU_MIN_OVERSAMPLES))) GiB, " *
            "$(gib(RSVD_PEAK_FUDGE * uu_sketch_bytes(N_u, k_uu, UU_MIN_OVERSAMPLES))) " *
            "GiB with the measured RSVD_PEAK_FUDGE = $(RSVD_PEAK_FUDGE), against a " *
            "budget of $(gib(budget)) GiB (90% of the card less the Green operator's " *
            "$(gib(gila_workspace_bytes(N_u))) GiB workspace). Lower --k-uu (the " *
            "sketch is 3·N_u·(k_uu + p)·16 bytes, so k_uu ≈ " *
            "$(max(0, floor(Int, budget / (RSVD_PEAK_FUDGE * 3 * Int(N_u) * 16)) - Int(UU_MIN_OVERSAMPLES))) " *
            "would fit), or run this point on a larger card")
        @warn string(now()) * " [bounds_bargaining::plan_uu_solve] The Asym(G⁰ᵤᵤ) sketch at $(oversamples) oversamples wants $(gib(RSVD_PEAK_FUDGE * uu_sketch_bytes(N_u, k_uu, oversamples))) GiB of a $(gib(budget)) GiB budget; cutting the oversamples to $(p_ok) ($(gib(RSVD_PEAK_FUDGE * uu_sketch_bytes(N_u, k_uu, p_ok))) GiB). The oversamples are the range finder's slack, so this costs accuracy in the tail of U_uu and nothing else; the eigenpair residuals in the log say whether it mattered"
        p = p_ok
    end

    qr_bytes <= budget || error("plan_uu_solve: orthonormalizing " *
        "[g_kept, U_uu] at k_uu = $(k_uu), m = $(m) holds U_uu, its Gram-Schmidt " *
        "copy and the $(N_u) × $(m_aug) result at once, $(gib(qr_bytes)) GiB, " *
        "against a budget of $(gib(budget)) GiB. Lower --k-uu or run this point on a " *
        "larger card; the augmented front end is dense by construction and has no " *
        "panel path to fall back to")
    basis_bytes <= budget || error("plan_uu_solve: the augmented front end's three " *
        "tall matrices (the $(N_u) × $(m_aug) basis, the $(N_u) × $(m) probes and " *
        "the $(N_u) × $(m_aug) working matrix) want $(gib(basis_bytes)) GiB against " *
        "a budget of $(gib(budget)) GiB. Lower --k-uu or run this point on a larger " *
        "card; the augmented front end is dense by construction and has no panel " *
        "path to fall back to")

    return (oversamples=p, power_iters=Int(power_iters),
            sketch_bytes=uu_sketch_bytes(N_u, k_uu, p),
            peak_bytes=round(Int, RSVD_PEAK_FUDGE * uu_sketch_bytes(N_u, k_uu, p)),
            basis_bytes=basis_bytes, qr_bytes=qr_bytes, budget_bytes=budget)
end

"""
    uu_eigenbasis(compute_env, G⁰ᵤᵤ_asym, k; oversamples, power_iters) -> NamedTuple

Top-`k` eigenpairs of `Asym(G⁰ᵤᵤ)`, computed in the bounds job itself with the same
`reigen_hermitian` the production RSVD uses, on whichever device the run is on.
Returns `(vectors, values, seconds)`.

In memory, always: [`plan_uu_solve`](@ref) has already established that the sketch
fits, and a `PanelMatrix` here would have to be materialized dense a moment later
anyway, since [`augmented_basis`](@ref)'s QR and the front end that follows are
dense by construction.
"""
function uu_eigenbasis(compute_env::ComputeEnvironment, G⁰ᵤᵤ_asym, k::Int;
                       oversamples::Int=UU_OVERSAMPLES,
                       power_iters::Int=UU_POWER_ITERS)
    sample_vec = zeros(ComplexF64, 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end
    @info string(now()) * " [bounds_bargaining::uu_eigenbasis] Computing the top $(k) " *
          "eigenpairs of Asym(G⁰ᵤᵤ) ($(size(G⁰ᵤᵤ_asym))) with $(oversamples) oversamples " *
          "and $(power_iters) power iterations"
    t = time_ns()
    out = reigen_hermitian(G⁰ᵤᵤ_asym, k; num_oversamples=oversamples,
                           num_power_iterations=power_iters, sample_vec=sample_vec)
    t = (time_ns() - t) / 1e9
    values = Array(out.values)
    @info string(now()) * " [bounds_bargaining::uu_eigenbasis] Done in $(round(t; digits=1)) s: " *
          "$(length(values)) eigenvalues, Λ[1] = $(values[1]), Λ[end] = $(values[end]), " *
          "Λ[end]/Λ[1] = $(values[end] / values[1])"
    return (vectors=out.vectors, values=values, seconds=t)
end

"""
    uu_residuals(G⁰ᵤᵤ_asym, U, Λ, idxs) -> Vector

`‖A uᵢ − Λᵢ uᵢ‖ / Λ₁` at the given column indices. A handful of matvecs, and the
only evidence in the log and in the saved JLD that [`UU_POWER_ITERS`](@ref) was
high enough.

Normalized by the *leading* eigenvalue rather than by `Λᵢ`. `Asym(G⁰ᵤᵤ)`'s spectrum
falls off a cliff, so `‖Av − λv‖/|λ|` at the tail divides a converged residual by a
number near the noise floor and reports a large relative error for a direction
that is, correctly, in the operator's numerical null space. Against `Λ₁` this reads
as what it is: how well the *subspace* has converged.
"""
function uu_residuals(G⁰ᵤᵤ_asym, U::AbstractMatrix, Λ::AbstractVector,
                      idxs::AbstractVector{Int})
    out = fill(NaN, length(idxs))
    scale = isempty(Λ) ? one(eltype(Λ)) : max(abs(Λ[1]), floatmin(eltype(Λ)))
    for (j, i) in enumerate(idxs)
        (1 <= i <= size(U, 2)) || continue
        u = U[:, i]
        Au = G⁰ᵤᵤ_asym * u
        out[j] = norm(Au .- Λ[i] .* u) / scale
    end
    return out
end

# Reading the basis h5 without ever holding the width it was saved at.
#
# `Funicular.save` writes one chunked HDF5 dataset, `N_u × stored_cols`, chunked
# `(N_u, panel width)`, and the columns the spectral cut keeps are its leading `m`
# (`_assert_positive_prefix`). Any leading run of columns is one contiguous
# hyperslab of that dataset, so the file's own reader can fill a destination *we*
# size and the stored width never has to be materialized.
#
# Both readers below used to go through `Funicular.load`, which builds a
# `PanelMatrix` spanning the whole stored width. Its panels are as wide as the
# RSVD's device budget made them, and a `Matrix(pm)` on top of that is a second
# copy of the stored width. At 1 λ (`N_u = 196,608`, 1,951 stored columns, panels
# of ~650) that is up to 7.7 GB of staged panel slabs plus a 6.1 GB dense block to
# keep 38 columns, i.e. ~14 GB against a request the cost model sized at
# `3·N_u·m·16 = 0.36 GB`. That is what OOM-killed the first bounds rerun of the 1 λ
# sweep, and it is what the calibration's own `stage_bounds` peaks are: 15.0-16.2
# GB of RSS at every m from 9 to 1,987, which this accounts for to within 0.7 GB
# at all four separations.
#
# `open_store`, `read_panel!` and `close_store!` are Funicular's disk-tier entry
# points (documented in its `src/io.jl`). Going through them rather than through
# HDF5 directly keeps the knowledge of the file's layout, panel width and storage
# eltype in the one package that writes it.
function _open_ur_asym_store(path::String, N_u::Integer, stored_cols::Integer)
    store = Funicular.open_store(path, "r")
    try
        store.N == N_u || error("$(path) holds $(store.N)-row vectors but the universe " *
            "is $(N_u) cells' worth of currents; this basis was not saved for this system")
        store.k == stored_cols || error("$(path) holds $(store.k) columns but " *
            "UR_asym/num_pos says $(stored_cols); the values and the vectors in this " *
            "file do not line up")
    catch
        Funicular.close_store!(store)
        rethrow()
    end
    return store
end

# Columns `cols` of an opened store, into `dst`. `read_panel!` fills a host array
# of the file's own eltype, so a narrowed cold tier (`ComplexF32` storage under a
# `ComplexF64` compute eltype) is read as it was written and converted on the way
# into `dst`, still one `length(cols)`-wide block at a time.
function _read_store_cols!(dst::AbstractMatrix, store, cols::UnitRange{Int})
    if eltype(dst) === store.stored
        Funicular.read_panel!(Funicular.DiskHome(store, cols), dst)
        return dst
    end
    staged = Matrix{store.stored}(undef, size(dst, 1), length(cols))
    Funicular.read_panel!(Funicular.DiskHome(store, cols), staged)
    copyto!(dst, staged)
    return dst
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

# The positives-only formats hold the whole saved block, `stored_cols` wide, and
# `m` of its columns are wanted. `m < stored_cols` only happens under the
# `gamma_rtol` cut, which drops a tail of a descending spectrum, so the wanted
# columns are the leading `m`. A view, so that the copy the caller makes is the
# only one and it is `N_u × m`.
function _leading_cols(V::AbstractMatrix, m::Int, stored_cols::Integer,
                       key::AbstractString)
    size(V, 2) == stored_cols || error("$key holds $(size(V, 2)) columns but " *
        "UR_asym/num_pos says $(stored_cols); the values and the vectors in this " *
        "file do not line up")
    return m == stored_cols ? V : view(V, :, 1:m)
end

# The block as a host `Matrix{ComplexF64}`, without copying one that already is
# one: JLD2 hands back an array of its own, so there is nothing to alias.
_as_host_complex(V::Matrix{ComplexF64}) = V
_as_host_complex(V::AbstractMatrix) = Matrix{ComplexF64}(V)

# JLD2 hands back whole datasets -- there is no hyperslab read -- so a cut on
# either in-JLD format materializes the *full* stored block before its prefix is
# taken. Both are small-run or legacy paths: `V_pos` is written by the dense-exact
# and in-memory RSVD branches, whose `N_u` is small by construction (the panel
# branch, which is the one that runs at the sizes where this would matter, writes
# the h5 instead), and nothing writes the legacy `V` any more. So this is logged
# rather than worked around -- the cost model bills the `N_u × m` block, and a job
# that lands here under a cut needs the difference on top of its request.
function _warn_full_width_read(key::AbstractString, N_u::Integer, m::Int,
                               width::Integer)
    m == width && return nothing
    gib = round(Int(N_u) * Int(width) * 16 / 2^30; digits=2)
    @warn string(now()) * " [bounds_bargaining::_read_ur_asym_dense] $(key) is a JLD2 dataset and JLD2 has no partial read, so all $(width) stored columns come through host memory ($(gib) GiB for the $(N_u) × $(width) block) before the leading $(m) are taken. The cost model bills the N_u × m block only, so this job needs that much host memory on top of its request"
    return nothing
end

# The positive-Γ block as a host `Matrix{ComplexF64}`. `cols` are the columns of
# the *sorted* basis that are wanted, that is, `sorted_idxs[1:m]`. The
# positives-only formats already hold that block, so for them the ordering has to
# be the identity: a file whose `D` is not descending has vectors we cannot
# reorder, and we error rather than pair the wrong vector with an eigenvalue.
function _read_ur_asym_dense(jld, source::Symbol, path::String,
                             cols::AbstractVector{Int}, N_u::Integer,
                             stored_cols::Integer)
    m = length(cols)
    if source === :h5
        _assert_positive_prefix(cols, "UR_asym/vectors_file")
        @info string(now()) * " [bounds_bargaining::_read_ur_asym_dense] Reading the $(N_u) × $(m) positive block from $(path) as one hyperslab of its $(stored_cols) stored columns"
        # One contiguous read into one N_u × m array. Nothing the stored width
        # sizes is allocated: see `_open_ur_asym_store` for what this replaced.
        store = _open_ur_asym_store(path, N_u, stored_cols)
        try
            V = Matrix{ComplexF64}(undef, Int(store.N), m)
            return _read_store_cols!(V, store, 1:m)
        finally
            Funicular.close_store!(store)
        end
    elseif source === :v_pos
        _assert_positive_prefix(cols, "UR_asym/V_pos")
        @info string(now()) * " [bounds_bargaining::_read_ur_asym_dense] Reading the $(N_u) × $(m) positive block from UR_asym/V_pos"
        _warn_full_width_read("UR_asym/V_pos", N_u, m, stored_cols)
        return _as_host_complex(_leading_cols(jld["UR_asym/V_pos"], m, stored_cols,
                                              "UR_asym/V_pos"))
    end
    @info string(now()) * " [bounds_bargaining::_read_ur_asym_dense] Reading the legacy full UR_asym/V and taking its leading $(m) sorted columns"
    V = jld["UR_asym/V"]
    _warn_full_width_read("UR_asym/V", N_u, m, size(V, 2))
    return Matrix{ComplexF64}(view(V, :, cols))
end

# The same block as an `N_u × m` `PanelMatrix` on the run's plan. With no cut the
# h5 is opened as the matrix's cold tier and its panels stream up as they are
# swept, so nothing dense of that size is ever built; under a cut the kept prefix
# is staged into a matrix of this run's own, one destination panel at a time. The
# other two formats have to come through host memory once, since that is how they
# are stored.
function _read_ur_asym_panel(jld, source::Symbol, path::String,
                             cols::AbstractVector{Int}, N_u::Integer, plan,
                             stored_cols::Integer)
    m = length(cols)
    if source === :h5
        _assert_positive_prefix(cols, "UR_asym/vectors_file")
        if m == stored_cols
            # Nothing to cut: the file is the matrix's cold tier and its panels
            # stream up as the sweeps reach them, so this allocates nothing at all.
            @info string(now()) * " [bounds_bargaining::_read_ur_asym_panel] Opening $(path) as a $(N_u) × $(stored_cols) panel matrix"
            pm = Funicular.load(Funicular.PanelMatrix, path; plan=plan, readonly=true)
            size(pm, 2) == stored_cols || error("$(path) holds $(size(pm, 2)) columns " *
                "but UR_asym/num_pos says $(stored_cols); the values and the vectors " *
                "in this file do not line up")
            return pm
        end
        # The h5 is opened readonly, so the cut cannot narrow it in place, and the
        # kept prefix has to go into a matrix this run owns. Reading it through a
        # `Funicular.load`ed source would stage panels as wide as the *file's*,
        # which the RSVD's device budget chose and which have nothing to do with
        # `m`: 2 GB of host memory per panel at 1 λ, pinned for the length of the
        # copy and so not even spillable, to keep 38 columns. The store is read
        # directly instead, one destination panel at a time, so the peak is one
        # `N_u × panelwidth(kept)` staging block plus the panel it lands in, and
        # `panelwidth(kept) <= m`.
        @info string(now()) * " [bounds_bargaining::_read_ur_asym_panel] Staging the leading $(m) of $(stored_cols) columns of $(path) into a panel matrix this run owns"
        store = _open_ur_asym_store(path, N_u, stored_cols)
        try
            # `store.computed` is the compute eltype the file records, which is what
            # `Funicular.load` would have given the matrix.
            T = store.computed
            kept = Funicular.PanelMatrix{T}(undef, Int(store.N), m; plan=plan,
                                            w=min(store.w, m))
            try
                block = Matrix{T}(undef, Int(store.N), Funicular.panelwidth(kept))
                for j in 1:Funicular.npanels(kept)
                    into = Funicular.panelrange(kept, j)
                    staged = view(block, :, 1:length(into))
                    _read_store_cols!(staged, store, into)
                    Funicular.copycols!(kept, into, staged)
                end
            catch
                Funicular.free!(kept)
                rethrow()
            end
            return kept
        finally
            Funicular.close_store!(store)
        end
    end
    @info string(now()) * " [bounds_bargaining::_read_ur_asym_panel] The JLD holds the basis densely; cutting it into panels"
    dense = _read_ur_asym_dense(jld, source, path, cols, N_u, stored_cols)
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

# The default cut on the positive Asym(G⁰ᵤᵣ) spectrum, exposed as --gamma-rtol.
# `load_bounds_inputs` says why the cut is there.
const DEFAULT_GAMMA_RTOL = 1e-12

# How many of the `num_pos` positive eigenvalues survive the relative cut. `Γ` is
# descending and `Γ[1] > 0`, so the survivors are a prefix and the count names
# them. `gamma_rtol = 0` keeps all of them, since every one is above zero.
_gamma_kept_count(Γ::AbstractVector, num_pos::Integer, gamma_rtol::Real) =
    count(>=(gamma_rtol * Γ[1]), view(Γ, 1:num_pos))

"""
    load_bounds_inputs(compute_env, smr; kwargs...)

Reads the `Asym(G⁰ᵤᵣ)` spectrum the RSVD job left in scratch and returns it
ready for [`bounds_from_spectrum`](@ref): `Γ` (every eigenvalue, host, sorted
descending), `Vur_asym` (the `N_u × num_pos` positive block), `Γrs`,
`sorted_idxs` (the descending permutation of the *saved* spectrum, saved as
`ordering_idxs`), `num_pos` and the `ResidencyPlan` the basis was built from, if
any.

`num_pos` is `size(Vur_asym, 2)`, and the eigenvalues it belongs to are
`Γ[1:num_pos]`. Note that it is not `count(>(0), Γ)`: `gamma_rtol` can cut the
noise floor off the positive block while `Γ` still carries the whole saved
spectrum, so `bounds_from_spectrum` has to be handed this value rather than count
it for itself.

`Vur_asym` comes back as a `Funicular.PanelMatrix` when the front end has
outgrown the device (see [`use_panel_bounds`](@ref)), and otherwise as a plain
matrix, on the device if the run is a GPU run. Only the kept block is staged,
since nothing reads the negative-Γ half of the legacy `V`.

# Keyword arguments
- `gamma_rtol`: keep only the positive eigenvalues with `Γ[i] >= gamma_rtol *
  Γ[1]`. Past the operator's numerical rank the RSVD returns its own noise floor,
  and those directions are neither eigenvectors nor independent of each other. The
  reverse Gram-Schmidt then normalizes vectors that orthogonalized to nothing,
  which ruins every earlier probe as well. `0` keeps the whole positive block.
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
                            gamma_rtol::Float64=DEFAULT_GAMMA_RTOL,
                            plan_override=nothing,
                            panel_mode::Union{Nothing,Bool}=nothing,
                            to_device::Bool=use_gpu(compute_env))
    0 <= gamma_rtol <= 1 || throw(ArgumentError(
        "gamma_rtol must lie in [0, 1] (0 keeps the whole positive block, and a cut " *
        "above Γ[1] would keep nothing), got $gamma_rtol"))
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

        # The spectral cut. Γ is descending and Γ[1] > 0, so the kept positives are
        # a prefix, and from here on `num_pos` is the kept count m: it sizes the
        # read, the plan and everything downstream. `stored_cols` keeps the file's
        # own count, which is what the readers slice against.
        stored_cols = num_pos
        num_pos = _gamma_kept_count(Γ, stored_cols, gamma_rtol)
        if num_pos < stored_cols
            @warn string(now()) * " [bounds_bargaining::load_bounds_inputs] Spectral truncation at gamma_rtol = $(gamma_rtol): keeping $(num_pos) of the $(stored_cols) positive Γ. Γ[$(num_pos)]/Γ[1] = $(Γ[num_pos] / Γ[1]) is kept, Γ[$(num_pos + 1)]/Γ[1] = $(Γ[num_pos + 1] / Γ[1]) is cut. The dropped directions sit at the RSVD's noise floor, so they are neither eigenvectors nor independent of the kept ones"
        end

        sender_size = dof_length(sender_mesh(smr))
        receiver_size = dof_length(receiver_mesh(smr))
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
            Vpos = _read_ur_asym_dense(jld_in, source, path, cols, N_u, stored_cols)
            size(Vpos) == (N_u, num_pos) || error(
                "the saved basis is $(size(Vpos, 1)) × $(size(Vpos, 2)) but the universe " *
                "is $(N_u) cells' worth of currents and num_pos is $(num_pos)")
            Vur_asym = to_device ? CuArray(Vpos) : Vpos
            return (Γ=Γ, Vur_asym=Vur_asym, Γrs=Γrs, sorted_idxs=sorted_idxs,
                    num_pos=num_pos, plan=nothing)
        end

        @info string(now()) * " [bounds_bargaining::load_bounds_inputs] Path: panel front end (the $(N_u) × $(num_pos) basis, ss and working matrix want $(footprint) GiB on the device)" plan
        basis = _read_ur_asym_panel(jld_in, source, path, cols, N_u, plan, stored_cols)
        size(basis) == (N_u, num_pos) || error(
            "the saved basis is $(size(basis, 1)) × $(size(basis, 2)) but the universe " *
            "is $(N_u) cells' worth of currents and num_pos is $(num_pos)")
        return (Γ=Γ, Vur_asym=basis, Γrs=Γrs, sorted_idxs=sorted_idxs,
                num_pos=num_pos, plan=plan)
    finally
        close(jld_in)
    end
end

# How much norm a column of Πₛ·gs_pos may lose to the reverse Gram-Schmidt before
# it counts as dependent on the later ones. Relative rather than absolute, because
# the columns of gs_pos are unit vectors but Πₛ shortens each by an unknown
# amount. Modified Gram-Schmidt over m columns loses about m·eps of relative
# accuracy, so this sits above that for every m the sweep runs.
const GS_DEPENDENCE_RTOL = 1e-12

"""
    reverse_gram_schmidt!(ss, gs_pos, s_projector, num_pos) -> ss

The probe vectors `sₖ`, built by modified Gram-Schmidt on the columns of
`A = Πₛ · gs_pos` taken in reverse order: column `i` is `Πₛgᵢ` orthogonalized
against `s_{i+1}, …, s_m` and normalized. Column `i` of the result therefore
spans `Πₛ·span(gᵢ, …, g_m)` together with the later columns, which is what makes
the outer loop's probe set `k ≥ n` shrink with `n`.

This is `O(m²)` BLAS-1 work over `N_u`-vectors. `blocked_reverse_gs_transform`
is its blocked equivalent, which is what the panel front end runs instead.

A column whose norm falls to `GS_DEPENDENCE_RTOL` of what it was before
orthogonalization is an error. `Πₛgᵢ` in the span of `s_{i+1}, …, s_m` means
`Πₛ·span(gᵢ, …, g_m)` has dimension `m − i`, so the nested spans the outer loop
indexes by `n` do not exist and the probes are undefined. Raise `--gamma-rtol` so
the dependent tail never enters the basis in the first place.
"""
function reverse_gram_schmidt!(ss::AbstractMatrix, gs_pos::AbstractMatrix,
                               s_projector, num_pos::Int)
    for i in num_pos:-1:1
        gᵢ = view(gs_pos, :, i)
        wᵢ = s_projector * gᵢ
        nrm₀ = norm(wᵢ) # what the loss of norm below is measured against
        for j in (i+1):num_pos
            sⱼ = view(ss, :, j)
            cᵢⱼ = dot(sⱼ, wᵢ)
            wᵢ .-= cᵢⱼ * sⱼ
        end
        nrm = norm(wᵢ)
        nrm > GS_DEPENDENCE_RTOL * nrm₀ || error(
            "reverse_gram_schmidt!: Πₛg$(i) is numerically in the span of the later " *
            "probes: its norm went from $(nrm₀) to $(nrm) over the orthogonalization, " *
            "a relative loss below GS_DEPENDENCE_RTOL = $(GS_DEPENDENCE_RTOL). " *
            "Normalizing it would hand every probe s₁…s$(i) an amplified noise " *
            "direction, and dropping it would leave the n ≤ $(i) indices without the " *
            "nested span they are defined on. This is what a basis carried past the " *
            "operator's numerical rank looks like: raise --gamma-rtol (default " *
            "$(DEFAULT_GAMMA_RTOL)) to keep the noise floor out of the basis")
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
number in the Gram matrix is harmless. A Cholesky that fails outright is where
this route sees what the loop sees as a vanishing norm: `Πₛ` has rank at most
`sender_size`, and an `m` past that is genuinely singular. It gets the same
shifted retry Funicular's `cholqr2!` uses.
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
        @warn string(now()) * " [bounds_bargaining::blocked_reverse_gs_transform] the $(m)×$(m) Gram matrix of Πₛ·basis is not numerically positive definite, so some basis vector is nearly linearly dependent on the later ones (this is where the reverse Gram-Schmidt loop errors out on a vanishing norm). A basis carried past the operator's numerical rank does this; --gamma-rtol cuts the noise floor out of it. Retrying the Cholesky with a shift of $shift"
        F = cholesky(Hermitian(G + shift * I); check=false)
        issuccess(F) || error("the $(m)×$(m) Gram matrix of Πₛ·basis is not positive " *
            "definite even after a shift of $shift: the projected basis is rank " *
            "deficient, so the probe vectors are not defined. Either the basis reaches " *
            "past the RSVD's numerical rank, in which case raise --gamma-rtol (default " *
            "$(DEFAULT_GAMMA_RTOL)), or it reaches past Πₛ's own rank, in which case " *
            "reduce basis_size below the sender's dimension")
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
        @warn string(now()) * " [bounds_bargaining::blocked_reverse_gs_transform] Πₛ·basis has κ ≥ $(round(ratio; sigdigits=3)); the blocked reverse Gram-Schmidt squares that before factoring, so its probes differ from the O(m²) loop's by roughly κ²·eps ≈ $(round(ratio^2 * eps(real(T)); sigdigits=2)). Both are orthonormal bases of the same nested spans, but the bounds will not reproduce the loop's to full precision. A κ this large usually means the basis reaches into the RSVD's noise floor; raise --gamma-rtol (default $(DEFAULT_GAMMA_RTOL)) to cut it"
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
    _bounds_front_end_augmented(compute_env, gs_pos, basis, Γ_pos_cpu, ζ,
                                s_projector, G⁰ᵤᵤ_asym, num_pos, sender_size)

The dense front end for an arbitrary orthonormal `basis` -- in production, the
`[g_kept, U_uu]` of [`augmented_basis`](@ref). Returns everything
[`_bounds_front_end_dense`](@ref) does, plus `W = basisᴴ gs_pos`, which is what the
pencil stage needs to build `Bₙ` in a basis where it is no longer diagonal (see
[`FactoredB`](@ref)).

`_bounds_front_end_dense` gets `C_basis` from `basisᴴ opmat(C, basis)`, sweeping the
whole of `C` -- `ζ⁻¹Πₛ`, `(−G⁰ᵤᵣ)ᵃ₊` and `(G⁰ᵤᵤ)ᵃ` -- through the Green operator's
column loop. Only the last of those actually needs matvecs. The other two are
algebra on objects already in hand, exactly as `_bounds_front_end_panel` observes
for the pure `g` basis, and the observation survives the augmentation:

    basisᴴ Πₛ basis         = Bₛᴴ Bₛ,  Bₛ = basis[1:sender_size, :]
    basisᴴ (−G⁰ᵤᵣ)ᵃ₊ basis  = W diag(Γ) Wᴴ

The panel front end can write the second one as `(basisᴴbasis) diag(Γ) (basisᴴbasis)ᴴ`
only because its basis *is* `gs_pos`; `W` is the general form. `D = (−G⁰ᵤᵣ)ᵃ₊ −
ζ⁻¹Πᵣ` is then exact in the basis too, since `−ζ⁻¹Πᵣ = ζ⁻¹Πₛ − ζ⁻¹1` and
`basisᴴbasis = 1` after the QR, so the whole τ family still comes out of one Green
sweep -- now `m_aug` columns wide rather than `m`.

The probes are the production probes: `reverse_gram_schmidt!` on `Πₛ·gs_pos`, the
`g` columns alone. See the section comment above [`augmented_basis`](@ref) for why
that is deliberate.
"""
function _bounds_front_end_augmented(compute_env::ComputeEnvironment, gs_pos, basis,
                                     Γ_pos_cpu, ζ, s_projector, G⁰ᵤᵤ_asym,
                                     num_pos::Int, sender_size::Int)
    N_u = size(gs_pos, 1)
    T = eltype(basis)
    m_aug = size(basis, 2)

    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Performing reverse Gram-Schmidt on the $(num_pos) g columns to construct the ss basis"
    t_gram_schmidt = time_ns()
    ss = similar(gs_pos, N_u, num_pos)
    reverse_gram_schmidt!(ss, gs_pos, s_projector, num_pos)
    t_gram_schmidt = (time_ns() - t_gram_schmidt) / 1e9

    t_ss_basis = time_ns()
    ss_basis = basis' * ss
    t_ss_basis = (time_ns() - t_ss_basis) / 1e9

    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Projecting C into the augmented basis of size $(m_aug) (one (G⁰ᵤᵤ)ᵃ sweep; ζ⁻¹Πₛ and (−G⁰ᵤᵣ)ᵃ₊ are algebra on W)"
    t_c_projection = time_ns()
    W = basis' * gs_pos                                # m_aug × num_pos
    Bₛ = view(basis, 1:sender_size, :)
    S_basis = Bₛ' * Bₛ                                 # basisᴴ Πₛ basis, exact
    Γdev = similar(basis, real(T), num_pos)
    copyto!(Γdev, Γ_pos_cpu)
    A_basis = (W .* reshape(Γdev, 1, :)) * W'          # basisᴴ (−G⁰ᵤᵣ)ᵃ₊ basis
    Guu_basis = basis' * opmat(G⁰ᵤᵤ_asym, basis)       # basisᴴ (G⁰ᵤᵤ)ᵃ basis
    C_basis = (1 / ζ) .* S_basis .+ A_basis .+ Guu_basis
    # −ζ⁻¹Πᵣ = ζ⁻¹Πₛ − ζ⁻¹1, and basisᴴbasis = 1 after the QR. The identity is built
    # as a dense block rather than through `diagind` so that this is one broadcast on
    # the device with no scalar indexing anywhere.
    Id = similar(basis, T, m_aug, m_aug)
    copyto!(Id, Matrix{T}(I, m_aug, m_aug))
    D_basis = (1 / ζ) .* S_basis .+ A_basis .- (1 / ζ) .* Id
    t_c_projection = (time_ns() - t_c_projection) / 1e9

    return (ss=ss, ss_basis=ss_basis, C_basis=C_basis, D_basis=D_basis, W=W,
            t_gram_schmidt=t_gram_schmidt, t_ss_basis=t_ss_basis,
            t_c_projection=t_c_projection)
end

# The panel version of the same front end (FUNICULAR_PLAN.md, workstream C1-C3).
# Every N_u-scale object is a PanelMatrix; every m × m object is formed on the host
# and handed to the pencil stage in the compute device's array space, exactly as
# the dense path hands it over.
#
# Three panel sweeps stand in for the dense path's m orthogonalization passes over
# N_u-vectors. C(1) = ζ⁻¹Πₛ + (−G⁰ᵤᵣ)ᵃ₊ + (G⁰ᵤᵤ)ᵃ only needs the Green term swept:
# with basis = gs_pos the other two are already in hand, since basisᴴΠₛbasis = S
# and basisᴴ(−G⁰ᵤᵣ)ᵃ₊basis = (basisᴴbasis) diag(Γ₊) (basisᴴbasis)ᴴ. D_basis reuses
# S as well, so the whole τ family comes out of the same three sweeps.
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
- `num_pos`: `m`, the number of leading `Γ` the basis spans, which must equal
  `Vur_asym`'s column count when the block is positives-only. Defaults to
  `count(>(0), Γ)`, but [`load_bounds_inputs`](@ref) passes its own value, since
  its `gamma_rtol` cut can leave `Γ` with positives past the columns it staged.
  Counting them here would then disagree with the basis.
- `basis_size`: how many leading eigenvectors to use as the projection basis,
  capped at `m = num_pos`.
- `G₀_uu`: pre-loaded universe operator, loaded here if not supplied.
- `outer_indices`: which `n` of the outer `σₙ` loop to actually evaluate.
  `nothing` (the default) means all of them.
- `nan_unevaluated`: leave `bounds_dual_basis` at `NaN` on the indices the loop
  did not evaluate, instead of the `0.0` the array is allocated with. `false`
  (the default) reproduces the pre-existing output exactly. A zero is the
  tightest bound there is, so an unevaluated index reads as a bound of zero and
  wins every `argmin` in `which_bounds`; that is harmless when the whole loop
  ran and wrong when it did not. The block-parallel bounds path
  (`--outer-range`, `bench/merge_bounds_blocks.jl`) sets this, which is what
  makes `bounds_dual_basis`, `true_bounds` and `which_bounds` sliceable by
  index: outside the evaluated range they are `NaN`, `NaN` and `3`, and the
  merge takes each index from the block that owns it.
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
- `tau_window`: how many grid points either side of the previous index's best `τ`
  to sweep. The minimiser is piecewise constant in `n`, so the neighbour's best
  grid point is nearly always this index's as well. A windowed minimum sitting on
  a window edge that is not also an end of the grid may be hiding a lower point
  just outside, so that index is swept in full instead and the reported bound is
  the same one an unwindowed sweep gives. The skipped grid points stay `NaN` in
  `bounds_dual_by_tau`. `0` sweeps the whole grid at every index. The window only
  applies to an `n` immediately following the last one evaluated, so a sparse
  `outer_indices` sweeps in full throughout.
- `pencil_cache_max`: how many refinement whitenings to keep, evicting the least
  recently used. The grid pencils are already shared across indices, but each
  index's golden-section probes are off-grid. Consecutive indices on a plateau
  open the same bracket and probe the same `τ`, so a handful of entries is enough
  to serve almost all of the refinement's `m × m` eigendecompositions from the
  cache. Each entry is an `m × m` whitener plus null space in the compute device's
  array space, so the memory is `pencil_cache_max` times that; `0` disables the
  cache. Note that on the augmented path the pencils are `m_aug × m_aug`, so the
  entries grow with `k_uu` as well; see the comment on the cache itself.
- `k_uu`: how many leading eigenvectors of `Asym(G⁰ᵤᵤ)` to augment the projection
  basis with, when the point qualifies (see `augment_threshold`). Defaults to
  [`DEFAULT_K_UU`](@ref). `0` disables the augmentation and reproduces the
  pre-augmentation output bit for bit. The section comment above
  [`augmented_basis`](@ref) says what this repairs and why.
- `augment_threshold`: augment only when the kept `m = num_pos` is *below* this.
  Defaults to [`DEFAULT_AUGMENT_THRESHOLD`](@ref), so far-field points -- the ones
  whose projected dual had stopped being a bound -- are augmented and near-field
  ones run exactly as they did before.
- `uu_oversamples`, `uu_power_iters`: `reigen_hermitian` parameters for the
  `Asym(G⁰ᵤᵤ)` solve. [`plan_uu_solve`](@ref) may lower the oversamples to make the
  sketch fit the device, and reports it when it does.

# Returns
A named tuple with the bounds, the bookkeeping needed to save them, and
`stage_times` / `outer_times` for calibration. `bounds_dual_basis` holds the
per-index minimum over all evaluated `τ`, `opt_taus` the `τ` that achieved it
(off-grid when refinement improved on the grid), and `bounds_dual_by_tau` the
grid-only `num_pos × length(τs)` table (`NaN` where an index/grid point was
skipped or failed), with the grid echoed in `tau_grid`. `tau_search` counts what
the search did over the run: refinement pencil cache hits and misses, and how many
indices fell back to a full grid sweep. `evaluated_indices` lists the `n` the loop
actually finished, which is `outer_indices` clipped to `1:min(basis_size, num_pos)`
unless the loop stopped early.

The basis-side objects `ss` (full-space probe vectors, `N × num_pos`),
`ss_basis`, `C_basis` (the `τ = 1` projected constraint) and `D_basis` are also
returned, in their compute-device array space, so that `verify_bounds` can
rebuild any `C(τ)` pencil and its probes without re-deriving them.
"""
function bounds_from_spectrum(compute_env::ComputeEnvironment, smr::SMRSystem,
                              Γ::AbstractVector, Vur_asym,
                              Γrs::AbstractVector;
                              num_pos::Int=count(>(zero(eltype(Γ))), Γ),
                              basis_size::Int=size(Vur_asym, 2),
                              G₀_uu=nothing,
                              outer_indices::Union{Nothing,AbstractVector{Int}}=nothing,
                              nan_unevaluated::Bool=false,
                              on_outer_error::Symbol=:throw,
                              τs::AbstractVector{<:Real}=range(0.0, 1.0, length=5),
                              τ_refine_tol::Union{Nothing,Real}=0.05,
                              tau_window::Int=2,
                              pencil_cache_max::Int=16,
                              k_uu::Int=DEFAULT_K_UU,
                              augment_threshold::Int=DEFAULT_AUGMENT_THRESHOLD,
                              uu_oversamples::Int=UU_OVERSAMPLES,
                              uu_power_iters::Int=UU_POWER_ITERS)
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
    pencil_cache_max >= 0 || throw(ArgumentError(
        "pencil_cache_max must be non-negative (0 disables the refinement pencil " *
        "cache), got $pencil_cache_max"))
    # `num_pos` = m, the numerical rank of (−G⁰ᵤᵣ)ᵃ₊ this run works with. It has to
    # match the columns actually staged, which is why it comes in rather than being
    # counted off Γ.
    1 <= num_pos <= length(Γ) || throw(ArgumentError(
        "num_pos must lie in 1:length(Γ) = 1:$(length(Γ)), got $(num_pos)"))
    all(>(zero(eltype(Γ))), view(Γ, 1:num_pos)) || throw(ArgumentError(
        "the leading $(num_pos) Γ are not all positive, so they are not a positive " *
        "block of a descending spectrum"))
    size(Vur_asym, 2) >= num_pos || throw(ArgumentError(
        "Vur_asym has $(size(Vur_asym, 2)) columns, fewer than num_pos = $(num_pos)"))
    # U_uu = read_array(jld_in, "UU/U", use_gpu(compute_env)) # TODO: could use this as basis too
    if isnothing(G₀_uu)
        G₀_uu = load_green_function(compute_env, smr, [Sender, Receiver], [Sender, Receiver]) # universe -> universe
    end
    # r_projector, s_projector, u_projector, G₀_uu_disjoint = projected_operators(G₀_uu, smr, compute_env)
    s_projector = projected_operators(G₀_uu, smr, compute_env)
    # G⁰ᵤᵤ_asym = u_projector * asym(LinearMap(G₀_uu)) * u_projector
    G⁰ᵤᵤ_asym = asym_self(G₀_uu)

    GC.gc()
    GC.gc()
    GC.gc()

    # @info "hello" size(Vur_asym) size(G₀_uu) size(s_projector) size(Γ) size(Γrs)

    χ = susceptibility(smr)
    ζ = abs(χ)^2/imag(χ)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Susceptibility χ = $χ, material factor ζ = $ζ"

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
    sender_size = dof_length(sender_mesh(smr))
    receiver_size = dof_length(receiver_mesh(smr))
    size(Vur_asym, 1) == sender_size + receiver_size || error(
        "the universe is not [sender; receiver] ($(size(Vur_asym, 1)) ≠ " *
        "$sender_size + $receiver_size), so Πᵣ ≠ 1 − Πₛ and the τ family " *
        "cannot be assembled from the sender projector alone")

    # Whether this point gets the Asym(G⁰ᵤᵤ) augmentation. Both conditions are
    # cheap and both are logged, because "did this point augment?" is the first
    # question anyone reading a bound will ask.
    k_uu >= 0 || throw(ArgumentError("k_uu must be non-negative (0 disables the " *
        "Asym(G⁰ᵤᵤ) augmentation), got $k_uu"))
    N_u = sender_size + receiver_size
    # The universe has N_u directions and the g basis already holds m of them, so
    # there are at most N_u − m left to add; and the sketch `reigen_hermitian`
    # builds is k + p wide, which cannot exceed the operator either. Both clamps
    # only ever bind on the small test systems -- at 1 λ, N_u = 196,608 against
    # k_uu = 512 -- but an unclamped `reigen_hermitian(A, 512)` on a 48 × 48
    # operator is a confusing failure a long way from its cause.
    k_uu_universe = min(k_uu, N_u - num_pos)
    augmenting = k_uu > 0 && num_pos < augment_threshold && k_uu_universe > 0
    if k_uu == 0
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Not augmenting: --k-uu 0"
    elseif num_pos >= augment_threshold
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Not augmenting: the kept m = $(num_pos) is at or above --augment-threshold $(augment_threshold), so the g basis already represents the constraint and this point runs exactly as it did before --k-uu existed"
    elseif k_uu_universe <= 0
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Not augmenting: the g basis already spans all $(N_u) directions of the universe, so there is nothing to augment it with"
    elseif k_uu_universe < k_uu
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Clamping the augmentation to the universe: k_uu $(k_uu) → $(k_uu_universe) (N_u = $(N_u), m = $(num_pos))"
    end
    if augmenting && Vur_asym isa Funicular.PanelMatrix
        error("the Asym(G⁰ᵤᵤ) augmentation was requested (--k-uu $(k_uu), kept m = " *
              "$(num_pos) < --augment-threshold $(augment_threshold)) for a point whose " *
              "front end is panelized. The augmented front end is dense by construction " *
              "-- augmented_basis's Householder QR and the m_aug-wide Green sweep both " *
              "want the whole $(N_u) × $(num_pos + k_uu_universe) block resident -- and the " *
              "panel path exists precisely because that block does not fit, so there is " *
              "nothing sensible to do here. `use_panel_bounds` chose the panel path " *
              "because RSVD_PEAK_FUDGE · 3 · N_u · m · 16 = " *
              "$(round(RSVD_PEAK_FUDGE * bounds_footprint_bytes(N_u, num_pos) / 2^30; digits=2)) " *
              "GiB exceeds this device's budget" *
              (use_gpu(compute_env) ?
               " of $(round(device_budget_bytes() / 2^30; digits=2)) GiB" :
               " (or a plan_override forced it on a CPU run)") *
              ". With the default threshold that only happens on a card far too small for the " *
              "point: the sizer (bench/size_bounds_jobs.jl, with the augmentation on) " *
              "puts an augmenting point on an allocation where the dense front end " *
              "fits. Run this point on a larger card, lower --augment-threshold below " *
              "$(num_pos) so it is not augmented, or set --k-uu 0")
    end

    # What the card can pay for, which is a separate question from what the physics
    # wants. `--augment-threshold` caps m and therefore caps m_aug = m + k_uu, but at
    # the larger universes that cap is above what a dense N_u × m_aug front end fits
    # in, so the effective k_uu comes down here rather than the job dying at its
    # first big allocation. Only evaluated on a point that is actually going to
    # augment: `clip_k_uu` errors when nothing useful fits, and a point that is not
    # augmenting has nothing to refuse. See `clip_k_uu` and `max_k_uu_for_budget`.
    uu_clip = augmenting ? clip_k_uu(compute_env, N_u, num_pos, k_uu_universe) :
              (k_uu=k_uu_universe, requested=k_uu_universe, clipped=false, reason="",
               k_fit=k_uu_universe, budget_bytes=0)
    k_uu_eff = uu_clip.k_uu
    uu_p = min(uu_oversamples, max(0, N_u - k_uu_eff))
    augmenting && uu_p < uu_oversamples &&
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Clamping the sketch to the universe: oversamples $(uu_oversamples) → $(uu_p) (N_u = $(N_u), k_uu = $(k_uu_eff))"

    # The Asym(G⁰ᵤᵤ) eigenbasis, computed here rather than read from the RSVD
    # output: the RSVD job never wrote one (the commented-out `read_array(jld_in,
    # "UU/U", ...)` this replaces was aspirational), and at the sizes that augment
    # the solve is a few minutes against a bounds job that is already tens.
    uu_plan = nothing
    uu_values = Float64[]
    uu_residual_idxs = Int[]
    uu_residual_values = Float64[]
    uu_seconds = 0.0
    k_uu_used = 0
    num_uu_kept = 0
    num_uu_dropped = 0
    dropped_cols = Int[]
    rdiag_min_ratio = NaN
    m_aug = RSVD_BASIS_SIZE

    front = if Vur_asym isa Funicular.PanelMatrix
        _bounds_front_end_panel(compute_env, Vur_asym, Γ_pos_cpu, ζ, s_projector,
                                G⁰ᵤᵤ_asym, num_pos, RSVD_BASIS_SIZE)
    elseif augmenting
        RSVD_BASIS_SIZE == num_pos || error(
            "the augmented front end assumes the g block is the whole kept positive " *
            "block (basis_size = num_pos = $(num_pos)), got RSVD_BASIS_SIZE = " *
            "$(RSVD_BASIS_SIZE). A truncated g block would make W = basisᴴgs_pos and " *
            "the Bₙ factor disagree about which channels the outer loop indexes")
        gs_pos = size(Vur_asym, 2) == num_pos ? Vur_asym : Vur_asym[:, 1:num_pos]
        uu_plan = plan_uu_solve(compute_env, N_u, num_pos, k_uu_eff;
                                oversamples=uu_p, power_iters=uu_power_iters)
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Augmenting the $(num_pos)-column g basis with the top $(k_uu_eff) eigenvectors of Asym(G⁰ᵤᵤ)" oversamples=uu_plan.oversamples power_iters=uu_plan.power_iters sketch_GiB=round(uu_plan.sketch_bytes / 2^30; digits=2) fudged_peak_GiB=round(uu_plan.peak_bytes / 2^30; digits=2) front_end_GiB=round(uu_plan.basis_bytes / 2^30; digits=2) budget_GiB=round(uu_plan.budget_bytes / 2^30; digits=2)

        uu = uu_eigenbasis(compute_env, G⁰ᵤᵤ_asym, k_uu_eff;
                           oversamples=uu_plan.oversamples,
                           power_iters=uu_plan.power_iters)
        U_uu, uu_values, uu_seconds = uu.vectors, uu.values, uu.seconds
        k_uu_used = size(U_uu, 2)
        k_uu_used == k_uu_eff || @warn string(now()) * " [bounds_bargaining::bounds_from_spectrum] reigen_hermitian returned $(k_uu_used) of the $(k_uu_eff) requested Asym(G⁰ᵤᵤ) components; the augmentation is that much smaller"
        uu_residual_idxs = unique(clamp.([1, cld(k_uu_used, 4), cld(k_uu_used, 2),
                                          k_uu_used], 1, max(k_uu_used, 1)))
        uu_residual_values = uu_residuals(G⁰ᵤᵤ_asym, U_uu, uu_values, uu_residual_idxs)
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Asym(G⁰ᵤᵤ): Λ[1] = $(first(uu_values)), Λ[$(k_uu_used)] = $(last(uu_values)), eigenpair residuals ‖Av − λv‖/Λ₁ at $(uu_residual_idxs): $(uu_residual_values). A residual well above 1e-8 at the last index means the subspace has not converged; raise the power iterations"

        aug = augmented_basis(gs_pos, U_uu)
        num_uu_kept, num_uu_dropped = aug.num_uu_kept, aug.num_uu_dropped
        dropped_cols, rdiag_min_ratio = aug.dropped_cols, aug.rdiag_min_ratio
        basis = aug.basis
        m_aug = size(basis, 2)
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Augmented basis: m_aug = $(m_aug) (g: $(num_pos), U_uu kept: $(num_uu_kept), dropped: $(num_uu_dropped) $(dropped_cols), min|R|/max|R| = $(rdiag_min_ratio))"
        # `U_uu` and the QR's working copy are N_u-tall and are done with; the front
        # end is about to want three more matrices of that height.
        U_uu = nothing
        aug = nothing
        run_gc()
        use_gpu(compute_env) && CUDA.reclaim()

        _bounds_front_end_augmented(compute_env, gs_pos, basis, Γ_pos_cpu, ζ,
                                    s_projector, G⁰ᵤᵤ_asym, num_pos, sender_size)
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
    # `W = basisᴴgs_pos`, the one extra small matrix the augmented pencil stage
    # needs; `nothing` on the two paths where Bₙ is diagonal.
    W_aug = augmenting ? front.W : nothing

    # None of the C(τ) depend on n, so the grid pencils are eigendecomposed once
    # here; the golden-section refinement builds its off-grid pencils on demand
    # through the cache below (those whitenings land in outer_times, not c_range).
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

    # Refinement pencils, memoized on the exact Float64 τ. C(τ) does not depend on
    # n, and consecutive indices sitting on the same τ* plateau open the identical
    # golden-section bracket, so they probe the same τ and only the first index of
    # the plateau pays for the whitening. The keys are compared exactly rather than
    # to a tolerance because the probe points are arithmetic on the same bracket
    # endpoints, so they reproduce bit for bit.
    #
    # Each entry holds an m_aug × m_aug whitener plus null space in the compute
    # device's array space, m_aug² complex between them. The two regimes:
    #
    #   * unaugmented, m_aug = m. At the near-field m = 4000 that is 4000² · 16 =
    #     256 MB, so the 16-entry default is 4.1 GB, which fits alongside the rest of
    #     the working set on an A100-40. This is the case the number was chosen for
    #     and it is unchanged.
    #   * augmented, m_aug = m + k_uu with m < augment_threshold = 1000 and
    #     k_uu = 512, so m_aug < 1512 and an entry is under 37 MB: 0.59 GB for all
    #     16. The augmentation cannot make this term the binding one, because the
    #     threshold that lets a point augment is what caps m_aug.
    #
    # The worst case over both is therefore still the unaugmented m = 4000 one, by a
    # factor of seven. Raising the threshold from 500 to 1000 raised the augmented
    # ceiling from 1012 to 1512, which is 2.2× the bytes per entry and still an
    # order of magnitude below the unaugmented case the 16 was chosen against.
    #
    # A --augment-threshold raised far past this would change that: at m = 4000 and
    # k_uu = 512 the entries would be 4512² · 16 = 326 MB and the cache 5.2 GB. The
    # guards for that are `clip_k_uu` and `plan_uu_solve`, which cut or refuse the
    # tall matrices such a point would need long before the cache becomes the
    # problem -- the tall front end at m = 4000, k_uu = 512 is 25× the cache.
    pencil_cache = Pair{Float64,Any}[] # LRU order, least recently used first
    pencil_cache_hits = 0
    pencil_cache_misses = 0
    cached_pencil(τ_raw::Real) = begin
        τ = Float64(τ_raw)
        if pencil_cache_max > 0
            # Golden section brackets the best grid point between its neighbours,
            # so its interior grid point is a τ the sweep already eigendecomposed.
            grid_hit = findfirst(i -> Float64(τs[i]) == τ, usable_τ)
            if !isnothing(grid_hit)
                pencil_cache_hits += 1
                return pencils[usable_τ[grid_hit]]
            end
            hit = findfirst(entry -> first(entry) == τ, pencil_cache)
            if !isnothing(hit)
                entry = popat!(pencil_cache, hit)
                push!(pencil_cache, entry) # touch: move to the MRU end
                pencil_cache_hits += 1
                return last(entry)
            end
        end
        pencil_cache_misses += 1
        pencil = build_pencil(τ)
        # A `nothing` is a failed whitening. It stays out of the cache so that a
        # transient device failure is not remembered as a property of that τ; the
        # cost is a rebuild if the same τ comes back.
        if pencil_cache_max > 0 && !isnothing(pencil)
            length(pencil_cache) >= pencil_cache_max && popfirst!(pencil_cache)
            push!(pencil_cache, τ => pencil)
        end
        return pencil
    end

    # The probes stay in the pencil's array space. Only the small projected
    # b-vectors cross to the host, where the scalar root finds live.
    B_basis_diagonal = zeros(real(eltype(C_basis)), RSVD_BASIS_SIZE)

    bounds_dual_basis = zeros(Float64, num_pos)
    bounds_dual_by_tau = fill(NaN, num_pos, length(τs))
    opt_taus = fill(NaN, num_pos)
    ns = isnothing(outer_indices) ? (1:RSVD_BASIS_SIZE) :
         filter(n -> 1 <= n <= RSVD_BASIS_SIZE, outer_indices)
    complete = length(ns) == num_pos
    # Which indices actually finished, as opposed to which were asked for: an
    # `on_outer_error = :stop` run stops part way through `ns`. `nan_unevaluated`
    # masks against this, and the block path writes it out as `partial/indices`, so
    # a block that died half way cannot be merged as though it had covered its
    # whole range.
    evaluated = falses(num_pos)
    outer_times = Tuple{Int,Float64}[]
    outer_error = nothing
    # The windowed sweep below needs the last index evaluated and where its minimum
    # landed.
    prev_n = 0
    prev_best_grid_idx = 0
    grid_fallbacks = 0
    for n in ns # Compute bounds on σₙ(Pᵣₛ)
     try
        t_outer = time_ns()
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] [$n/$(num_pos)] Computing σₙ(Pᵣₛ) bound"

        @info string(now()) * " [$n/$(num_pos)] Projecting Bₙ into the basis of size $(m_aug)"
        # B_basis_n = B_basis(n)
        # Two representations of the same operator, picked once by `augmenting`:
        #   * the g basis, where Bₙ is diagonal and no projection is needed at all;
        #   * the augmented basis, where it is not, but is still the congruence
        #     W[:, n:m] diag(4Γ[n:m]/ζ) W[:, n:m]ᴴ, which `FactoredB` holds in
        #     factored form so the pencil never builds it.
        B_factor = nothing
        if augmenting
            idx = n:num_pos
            c_n = similar(C_basis, real(eltype(C_basis)), length(idx))
            copyto!(c_n, sqrt.((4 / ζ) .* Γ_pos_cpu[idx]))
            B_factor = FactoredB(W_aug[:, idx], c_n)
        else
            fill!(B_basis_diagonal, zero(eltype(B_basis_diagonal)))
            B_basis_diagonal[n:RSVD_BASIS_SIZE] .= (4/ζ) .* Γ_pos_cpu[n:RSVD_BASIS_SIZE] # Bₙ is diagonal in the gs_pos basis, so no projection is needed
        end

        # Solve the GEVP on each C(τ)'s numerical range and keep the tightest τ.
        # Bₙ shrinks with n (Bₙ ⪯ Bₙ₋₁, as they differ by a positive
        # semi-definite rank-one term), but the null directions it is allowed to
        # ignore grow with n too, so the check inside diag_pencil_eigen has to
        # happen per index rather than once up front. Every τ bounds σₙ(Pᵣₛ) on
        # its own, so an evaluation that fails numerically is dropped for this
        # index with a warning.
        pencil_dual(pencil, τ) = augmenting ?
            maximum(factored_probe_duals(pencil, B_factor, ss_basis, n, num_pos; τ=τ).duals) :
            maximum(pencil_probe_duals(pencil, B_basis_diagonal, ss_basis, n, num_pos; τ=τ).duals)
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

        sweep_grid!(idxs) = begin
            for i in idxs
                dual_τ = eval_dual(pencils[i], τs[i])
                isfinite(dual_τ) || continue
                bounds_dual_by_tau[n, i] = sqrt(dual_τ)
                if dual_τ < best_dual
                    best_dual, best_τ, best_grid_idx = dual_τ, Float64(τs[i]), i
                end
            end
        end

        # The minimising τ is piecewise constant in n, endpoint plateaus with one
        # abrupt transition between them, so the previous index's best grid point is
        # almost always this one's too and a ±tau_window sweep around it does the
        # work of the whole grid. Every τ bounds σₙ on its own, so narrowing the
        # sweep can only cost tightness. A minimum on a window edge that is not also
        # a grid end may be hiding a lower point just outside the window, so the
        # whole grid gets swept instead. `bounds_dual_by_tau` keeps its NaN at the
        # skipped points.
        #
        # Only an immediate predecessor says anything about this Bₙ: verify_bounds
        # passes a handful of spot indices as `outer_indices`, and those n are
        # decades apart.
        windowed = tau_window > 0 && prev_best_grid_idx > 0 && n == prev_n + 1
        if windowed
            window_lo = max(firstindex(τs), prev_best_grid_idx - tau_window)
            window_hi = min(lastindex(τs), prev_best_grid_idx + tau_window)
            @info string(now()) * " [$n/$(num_pos)] Solving λⱼ(Bₙ, C(τ)) over the τ grid window $(window_lo):$(window_hi) of $(firstindex(τs)):$(lastindex(τs))"
            sweep_grid!(i for i in usable_τ if window_lo <= i <= window_hi)
            windowed = isfinite(best_dual) &&
                       !(best_grid_idx == window_lo && window_lo > firstindex(τs)) &&
                       !(best_grid_idx == window_hi && window_hi < lastindex(τs))
            windowed || (grid_fallbacks += 1)
        end
        if !windowed
            # Reset the running minimum so the full sweep resolves ties exactly as
            # an unwindowed run does. The points the window already covered are
            # evaluated again: a few GEVPs, and no new whitening.
            best_dual, best_τ, best_grid_idx = Inf, NaN, 0
            @info string(now()) * " [$n/$(num_pos)] Solving λⱼ(Bₙ, C(τ)) over $(length(usable_τ)) τ grid point(s)"
            sweep_grid!(usable_τ)
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
            g₁ = eval_dual(cached_pencil(τ₁), τ₁)
            g₂ = eval_dual(cached_pencil(τ₂), τ₂)
            g₁ < best_dual && ((best_dual, best_τ) = (g₁, τ₁))
            g₂ < best_dual && ((best_dual, best_τ) = (g₂, τ₂))
            refine_iters = 0
            while hi - lo > τ_refine_tol && refine_iters < 200
                refine_iters += 1
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
        @info string(now()) * " [$n/$(num_pos)] Dual is $best_dual at τ = $best_τ, which gives a bound of $(sqrt(best_dual)) on σₙ(Pᵣₛ)"
        bounds_dual_basis[n] = sqrt(best_dual)
        opt_taus[n] = best_τ
        evaluated[n] = true
        prev_n, prev_best_grid_idx = n, best_grid_idx
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
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] τ search over $(length(ns)) index/indices: refinement pencil cache $(pencil_cache_hits) hit(s) / $(pencil_cache_misses) miss(es) (pencil_cache_max = $(pencil_cache_max)), $(grid_fallbacks) full-grid fallback(s) (tau_window = $(tau_window))"

    # Before the analytical bounds, because `which_bounds` and `true_bounds` are
    # `argmin`s over this array: a NaN here propagates into both (Julia's `argmin`
    # returns the NaN's index), which is exactly what an index nobody computed
    # should look like. A 0.0 would instead read as a perfectly tight bound.
    if nan_unevaluated
        masked = 0
        for n in 1:num_pos
            evaluated[n] && continue
            bounds_dual_basis[n] = NaN
            masked += 1
        end
        masked > 0 && @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Masking $(masked) unevaluated index/indices of $(num_pos) to NaN in bounds_dual_basis (and hence in true_bounds/which_bounds)"
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
            evaluated_indices=findall(evaluated),
            bounds_dual_basis=bounds_dual_basis,
            tau_grid=collect(Float64, τs), opt_taus=opt_taus,
            bounds_dual_by_tau=bounds_dual_by_tau,
            ss=ss, ss_basis=ss_basis, C_basis=C_basis, D_basis=D_basis,
            old_analytical_bounds=old_analytical_bounds,
            new_analytical_bounds=new_analytical_bounds,
            true_bounds=true_bounds, which_bounds=which_bounds, ks=ks,
            basis_size=RSVD_BASIS_SIZE,
            tau_search=(pencil_cache_hits=pencil_cache_hits,
                        pencil_cache_misses=pencil_cache_misses,
                        grid_fallbacks=grid_fallbacks),
            # What the augmentation did, so a plot can tell an augmented point from a
            # plain one and a rerun can tell whether `plan_uu_solve` had to cut the
            # oversamples. `augmented = false` leaves every other field at its
            # do-nothing value, which is what a pre-augmentation run reports.
            augmentation=(augmented=augmenting, k_uu_requested=k_uu,
                          k_uu_effective=k_uu_eff,
                          # What the card, as opposed to the universe, took off the
                          # request. `k_uu_effective` is already the clipped value;
                          # these two say whether the clip is why, so that a point
                          # whose bound was computed at a smaller k_uu than the rest
                          # of the sweep is identifiable from its output alone.
                          k_uu_clipped=uu_clip.clipped,
                          k_uu_clip_reason=uu_clip.reason,
                          k_uu_budget_bytes=uu_clip.budget_bytes,
                          k_uu_returned=k_uu_used, augment_threshold=augment_threshold,
                          m_aug=m_aug, num_uu_kept=num_uu_kept,
                          num_uu_dropped=num_uu_dropped, dropped_cols=dropped_cols,
                          rdiag_min_ratio=rdiag_min_ratio,
                          uu_oversamples=(uu_plan === nothing ? uu_oversamples : uu_plan.oversamples),
                          uu_power_iters=(uu_plan === nothing ? uu_power_iters : uu_plan.power_iters),
                          uu_seconds=uu_seconds, uu_values=uu_values,
                          uu_residual_idxs=uu_residual_idxs,
                          uu_residuals=uu_residual_values),
            stage_times=stage_times, outer_times=outer_times)
end

"""
    partial_bounds_path(project_dir, prefix, tag) -> String

Where a `--partial-suffix <tag>` run writes, and how
[`bench/merge_bounds_blocks.jl`](../bench/merge_bounds_blocks.jl) finds those
files again. One definition, used by the writer and the reader, so the two
cannot drift: `<prefix>_partial_<tag>.jld` beside the final `<prefix>.jld` in
the project directory.

A tag is restricted to word characters, `-` and `.`, so that it cannot smuggle a
path separator into the filename and so that the reader's pattern can recover it
unambiguously from a name that already contains `_`.
"""
partial_bounds_path(dir::AbstractString, prefix::AbstractString, tag::AbstractString) =
    joinpath(dir, "$(prefix)_partial_$(_check_partial_tag(tag)).jld")

const PARTIAL_TAG_PATTERN = r"^[A-Za-z0-9._-]+$"

function _check_partial_tag(tag::AbstractString)
    occursin(PARTIAL_TAG_PATTERN, tag) || throw(ArgumentError(
        "a --partial-suffix tag must match $(PARTIAL_TAG_PATTERN) (word characters, " *
        "'.', '-'), got '$(tag)'. The tag becomes part of a filename and is parsed " *
        "back out of it by bench/merge_bounds_blocks.jl"))
    return tag
end

"""
    _compute_bounds_sr(compute_env, smr, rsvd_params; kwargs...)

The bounds stage of one SR point: read the RSVD's spectrum off scratch, run
[`bounds_from_spectrum`](@ref), write `<prefix>.jld` into the project directory.

# Block-parallel keyword arguments

`outer_range` and `partial_suffix` split the run across independent jobs. The
outer loop over channel indices is embarrassingly parallel -- index `n`'s bound
depends on nothing computed at any other `n` -- so a job can be given a slice of
it and B such jobs can run concurrently. On a queue where short jobs backfill
quickly this turns one 8-hour job into B jobs of an hour that start at once.

- `outer_range`: evaluate only these channel indices. The *whole* front end still
  runs (the Gram-Schmidt, the projections, the `Asym(G⁰ᵤᵤ)` augmentation if the
  point qualifies), because every index needs it; that duplicated work is the
  price of the split, and it is minutes against a loop of hours. The saved arrays
  carry `NaN` outside the range (see `nan_unevaluated` in
  [`bounds_from_spectrum`](@ref)).
- `partial_suffix`: write to `<prefix>_partial_<tag>.jld` instead of
  `<prefix>.jld`, so the blocks do not overwrite each other and so nothing
  downstream mistakes a slice for a finished point. `bench/merge_bounds_blocks.jl`
  assembles the final file from them.

Passing neither reproduces the previous behaviour exactly, down to the file's key
set: the `partial/` group is written only by a partial run.
"""
function _compute_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams;
                            gamma_rtol::Float64=DEFAULT_GAMMA_RTOL,
                            k_uu::Int=DEFAULT_K_UU,
                            augment_threshold::Int=DEFAULT_AUGMENT_THRESHOLD,
                            outer_range::Union{Nothing,UnitRange{Int}}=nothing,
                            partial_suffix::Union{Nothing,AbstractString}=nothing,
                            plan_override=nothing, panel_mode::Union{Nothing,Bool}=nothing)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Computing bounds for SR system"
    partial = !isnothing(outer_range) || !isnothing(partial_suffix)
    if partial
        isnothing(outer_range) && error(
            "--partial-suffix was given without --outer-range. A partial file that " *
            "covers every index is a finished point under a name nothing reads; if " *
            "that is what you want, drop --partial-suffix")
        isnothing(partial_suffix) && error(
            "--outer-range was given without --partial-suffix. Writing a block's " *
            "slice to the point's final <prefix>.jld would leave a file that looks " *
            "finished and is mostly NaN, and the next block would overwrite it")
        isempty(outer_range) && error("--outer-range $(outer_range) is empty")
        first(outer_range) >= 1 || error(
            "--outer-range $(outer_range) starts below 1; channel indices are 1-based")
        # Up here rather than at the first use, which is after `load_bounds_inputs`:
        # a mistyped tag should cost a second, not the minutes it takes to stage the
        # basis off scratch.
        _check_partial_tag(partial_suffix)
    end

    inputs = load_bounds_inputs(compute_env, smr; gamma_rtol=gamma_rtol,
                                plan_override=plan_override, panel_mode=panel_mode)
    Γ, Vur_asym, Γrs, sorted_idxs = inputs.Γ, inputs.Vur_asym, inputs.Γrs, inputs.sorted_idxs

    # Written up front (truncating any previous run's file) so that the ordering
    # is on disk even if the bounds loop below is cut short by a time limit.
    jld_out_path = partial ?
        partial_bounds_path(project_dir(compute_env), file_prefix(smr), partial_suffix) :
        joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    partial && @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Partial run: channel indices $(outer_range) of the kept m = $(inputs.num_pos), writing $(basename(jld_out_path))"
    jld_out = jldopen(jld_out_path, "w")
    jld_out["Γrs"] = Array(Γrs)
    jld_out["ordering_idxs"] = sorted_idxs
    close(jld_out)

    result = bounds_from_spectrum(compute_env, smr, Γ, Vur_asym, Γrs; num_pos=inputs.num_pos,
                                  k_uu=k_uu, augment_threshold=augment_threshold,
                                  outer_indices=isnothing(outer_range) ? nothing :
                                                collect(outer_range),
                                  nan_unevaluated=partial)
    if partial
        # `complete` is false by construction here (it asks whether the loop covered
        # every index), so the check that matters instead is that this block covered
        # everything *it* was asked for. `outer_range` may legitimately hang off the
        # end of the spectrum: it was sliced from the m the sizer measured, and a
        # rerun of the RSVD can move that m by a few counts.
        wanted = intersect(outer_range, 1:inputs.num_pos)
        done = result.evaluated_indices
        isnothing(result.outer_error) && length(done) == length(wanted) || error(
            "block $(outer_range) evaluated $(length(done)) of the $(length(wanted)) " *
            "index/indices it was asked for" *
            (isnothing(result.outer_error) ? "" : "; the loop stopped at n = $(result.outer_error.n): $(result.outer_error.exception)") *
            ". Refusing to save a block that would merge as though it were whole")
        isempty(wanted) && @warn string(now()) * " [bounds_bargaining::_compute_bounds_sr] --outer-range $(outer_range) lies entirely past the kept m = $(inputs.num_pos), so this block evaluated nothing. Its file is still written, so a merge that has coverage from the other blocks succeeds; if the m moved a lot, resize the blocks"
    else
        result.complete || error("bounds_from_spectrum returned an incomplete result; refusing to save partial bounds")
    end

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
    #=
    What the Asym(G⁰ᵤᵤ) augmentation did on this point. The existing keys are
    untouched and keep their meanings -- `bounds_dual_basis` is still the
    per-channel dual, now computed in the richer basis -- so a plot that knows
    nothing about this group reads the file exactly as before. What the group adds
    is the ability to tell an augmented point from a plain one, which matters
    because the two are not comparable: a sweep replotted mid-migration would
    otherwise show a step at `augment_threshold` with nothing in the file to explain
    it.

    `augment/augmented` is the one key a plot needs; the rest is provenance.
    `uu_residuals` is the evidence that `uu_power_iters` was high enough for the
    augmentation to be an eigenbasis rather than an arbitrary spanning set, and
    `uu_oversamples` records what `plan_uu_solve` settled on, which is not
    necessarily what was asked for.

    `k_uu_clipped` / `k_uu_clip_reason` / `k_uu_budget_bytes` are the second thing
    that can make two augmented points incomparable, after `augmented` itself:
    `clip_k_uu` reduces the effective `k_uu` on a point whose dense augmented front
    end would not fit the card it landed on, so a sweep run across a mix of MIG
    slices and whole cards can carry two different `k_uu` without anything else in
    the file saying so. The reason is `""` when nothing was clipped.
    =#
    aug = result.augmentation
    if !haskey(jld_out, "augment/augmented")
        jld_out["augment/augmented"] = aug.augmented
        jld_out["augment/k_uu_requested"] = aug.k_uu_requested
        jld_out["augment/k_uu_effective"] = aug.k_uu_effective
        jld_out["augment/k_uu_clipped"] = aug.k_uu_clipped
        jld_out["augment/k_uu_clip_reason"] = aug.k_uu_clip_reason
        jld_out["augment/k_uu_budget_bytes"] = aug.k_uu_budget_bytes
        jld_out["augment/k_uu_returned"] = aug.k_uu_returned
        jld_out["augment/augment_threshold"] = aug.augment_threshold
        jld_out["augment/m"] = result.num_pos
        jld_out["augment/m_aug"] = aug.m_aug
        jld_out["augment/num_uu_kept"] = aug.num_uu_kept
        jld_out["augment/num_uu_dropped"] = aug.num_uu_dropped
        jld_out["augment/dropped_cols"] = aug.dropped_cols
        jld_out["augment/rdiag_min_ratio"] = aug.rdiag_min_ratio
        jld_out["augment/uu_oversamples"] = aug.uu_oversamples
        jld_out["augment/uu_power_iters"] = aug.uu_power_iters
        jld_out["augment/uu_seconds"] = aug.uu_seconds
        jld_out["augment/uu_values"] = aug.uu_values
        jld_out["augment/uu_residual_idxs"] = aug.uu_residual_idxs
        jld_out["augment/uu_residuals"] = aug.uu_residuals
    end
    #=
    What this file is a slice of. Written only by a partial run, so a normal run's
    key set is bit for bit what it always was, and its presence is how
    bench/merge_bounds_blocks.jl tells a block apart from a finished point.

    `indices` rather than just the range because it is the ground truth the merge
    needs: it is what the loop *finished*, already clipped to the kept m, so a
    merge can check coverage of 1:m by set union without re-deriving anything.
    `range_lo`/`range_hi` are the range as requested, kept for the log and for the
    error message when a block is missing.
    =#
    if partial && !haskey(jld_out, "partial/indices")
        jld_out["partial/indices"] = result.evaluated_indices
        jld_out["partial/range_lo"] = first(outer_range)
        jld_out["partial/range_hi"] = last(outer_range)
        jld_out["partial/tag"] = String(partial_suffix)
        jld_out["partial/num_pos"] = result.num_pos
    end
    close(jld_out)
    return result
end

function compute_bounds()
    compute_env, smr, rsvd_params, gamma_rtol, k_uu, augment_threshold,
        outer_range, partial_suffix = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using GPU acceleration on device $(gpu_device(compute_env))"
        if !haskey(ENV, "CC_CLUSTER") # This breaks on compute canada
            CUDA.device!(gpu_device(compute_env))
        end
    else
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _compute_bounds_sr(compute_env, smr, rsvd_params; gamma_rtol=gamma_rtol,
                           k_uu=k_uu, augment_threshold=augment_threshold,
                           outer_range=outer_range, partial_suffix=partial_suffix)
    else
        isnothing(outer_range) && isnothing(partial_suffix) || error(
            "--outer-range/--partial-suffix are only implemented for the SR path; " *
            "the SMR bounds stage is a stub (_compute_bounds_smr)")
        _compute_bounds_smr(compute_env, smr, rsvd_params)
    end
end
