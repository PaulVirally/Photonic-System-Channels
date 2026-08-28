## The augmented-basis variant front end in bench/augmented_basis_experiment.jl.
##
## Run from the repo root:
##     julia --project=. test/augmented_basis.jl
##
## No GPU. One tiny (2,2,2)+(2,2,2) SR Green function is generated into a
## mktempdir (about a minute) and reused by every run below; the Asym(G⁰ᵤᵣ)
## spectrum is synthetic, as in test/tau_search.jl. Exits nonzero on failure.
##
## Why this test exists. The experiment reimplements the dual in a basis where
## production's two shortcuts (Bₙ and basisᴴ(−G⁰ᵤᵣ)ᵃ₊basis being diagonal) do not
## hold. If that reimplementation is wrong, the k_uu sweep on narval produces a
## curve that rises for the wrong reason and nobody can tell. So the whole point
## of this file is the three anchors:
##
##   (a) with the pure g basis the variant front end reproduces production's
##       bounds_from_spectrum (at --k-uu 0) to roundoff;
##   (b) with N_u = 48 small enough to hold the *whole* space, the variant front
##       end at basis = 1 computes the exact full-space dual, the ground truth
##       the projected bound is supposed to be an upper bound on and is not;
##   (c) a g basis too small to represent the constraint under-reports, and
##       augmenting it with Asym(G⁰ᵤᵤ)'s eigenvectors climbs back, monotonically,
##       to exactly the ground truth once the two spans together cover ℂ⁴⁸.
##
## (c) is the far-field pathology reproduced at 48 dimensions, and it is the
## thing the narval job is looking for at 196,608.
##
## Since the augmentation became a production feature (`--k-uu` /
## `--augment-threshold` on `bounds_from_spectrum`), three more anchors cover the
## *shipped* path rather than the experiment's driver:
##
##   (d) `--k-uu 0` reproduces the pre-augmentation output bit for bit, against
##       values recorded from the code as it stood before the feature landed;
##   (e) the augmented production path recovers the same ground truth (b)
##       establishes, monotonically, on the same fixture;
##   (f) the threshold decides, and a point at or above it runs the k_uu = 0 code
##       to the last bit.
##
## The experiment file is a thin driver over `src/bounds.jl` (`FactoredB`,
## `factored_pencil_eigen`, `augmented_basis`, `uu_eigenbasis` and the front end
## are all production code) so (a) through (c) are testing the shipped
## implementation too, just through a loop that can be handed an arbitrary basis.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using LinearAlgebra
using LinearMaps
using Printf
using Random
using Logging

include(joinpath(@__DIR__, "..", "bench", "augmented_basis_experiment.jl"))

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-62s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

const ROOT = mktempdir(; cleanup=true)
for d in ("preload", "project", "scratch")
    mkpath(joinpath(ROOT, d))
end
println("workspace: ", ROOT)

const ENV_CPU = ComputeEnvironment(joinpath(ROOT, "preload"), joinpath(ROOT, "project"),
                                   joinpath(ROOT, "scratch"), GPUChoice(false, -1))
# The same fixture test/tau_search.jl uses: 48 currents, so the whole universe
# fits in a dense 48 × 48 and the full-space dual is computable exactly.
## `refine_gap=false`: these checks are about the bounds algebra, not the mesh, and
## they are written against the plain uniform universe. The refined mesh of a
## one-cell gap is a different (and much larger) operator; test/gap_refinement.jl
## and test/refined_pipeline.jl cover that side.
const SMR = SMRSystem((2, 2, 2), (1//32, 0//1, 0//1), (2, 2, 2),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im;
                      refine_gap=false)
const N_U = 3 * (prod(sender(SMR).cel) + prod(receiver(SMR).cel))
const SENDER_SIZE = 3 * prod(sender(SMR).cel)
const ζ = abs(PSC.susceptibility(SMR))^2 / imag(PSC.susceptibility(SMR))

# `K` is deliberately small: 3 of 48 directions is the far-field regime (15 of
# 196,608) in miniature, and it is what makes (c) have something to recover.
const K = 3
Random.seed!(0xA06E)
const Γ = collect(exp10.(range(log10(1 / ζ), log10(1e-4 / ζ), length=K)))
const GS = Matrix(qr(randn(ComplexF64, N_U, K)).Q)[:, 1:K]
const Γrs = collect(exp10.(range(log10(1 / ζ), log10(1e-6 / ζ), length=K)))

println("N_u = $(N_U), sender_size = $(SENDER_SIZE), num_pos = $(K), ζ = $(ζ)")

const G_UU = load_green_function(ENV_CPU, SMR, [Sender, Receiver], [Sender, Receiver])
const S_PROJ = PSC.projected_operators(G_UU, SMR, ENV_CPU)
const G_UU_ASYM = PSC.asym(LinearMap(G_UU))

# The production probes. Built once here, exactly as the experiment builds them:
# from Πₛ·gs alone, never from the augmentation.
const SS = begin
    ss = similar(GS, N_U, K)
    reverse_gram_schmidt!(ss, GS, S_PROJ, K)
    ss
end
check("the production probes are orthonormal", opnorm(SS' * SS - I) < 1e-12,
      @sprintf("|ss'ss - I| = %.3e", opnorm(SS' * SS - I)))

const TAUS = range(0.0, 1.0, length=5) # the production grid
quiet(f) = with_logger(f, SimpleLogger(stderr, Logging.Warn))

# The experiment's τ driver over an arbitrary basis. Its N_u-scale work is
# `PSC._bounds_front_end_augmented` and its pencil algebra is `PSC.FactoredB` /
# `PSC.factored_probe_duals`, i.e. production's; the probes it uses are the ones
# that front end builds, which is the same `reverse_gram_schmidt!` on Πₛ·GS that
# `SS` above is.
run_variant(basis; channels=K) = quiet() do
    augmented_dual_bounds(ENV_CPU, basis, GS, Γ, ζ, SENDER_SIZE, S_PROJ, G_UU_ASYM;
                          channels=channels, τs=TAUS, τ_refine_tol=0.05)
end

# =====================================================================
# 1: FactoredB reduces to diag_pencil_eigen term for term
# =====================================================================
#
# The whole generalization off the diagonal rests on
# `basisᴴBₙbasis = KᴴK` with `K = diag(c) Vᴴ`, and on `factored_pencil_eigen`
# being `diag_pencil_eigen` with that `K` substituted in. With `V = 1` the two
# must agree to the last bit that BLAS allows.

println("\n=== FactoredB vs diag_pencil_eigen")

Random.seed!(0xF00D)
let n = 12
    A = randn(ComplexF64, n, n)
    C = A' * A + 0.5I           # positive definite, so the null space is empty
    C = Matrix(Hermitian(C))
    pencil = PSC.psd_pencil_whitener(C)
    d = abs.(randn(n)) .+ 0.1
    Λd, Vd = PSC.diag_pencil_eigen(d, pencil.whitener, pencil.nullspace)
    F = FactoredB(Matrix{ComplexF64}(I, n, n), sqrt.(d))
    Λf, Vf = factored_pencil_eigen(F, pencil.whitener, pencil.nullspace)
    check("factored and diagonal pencils agree on the eigenvalues",
          maximum(abs, Λf .- Λd) <= 1e-11 * maximum(abs, Λd),
          @sprintf("max |ΔΛ| = %.3e", maximum(abs, Λf .- Λd)))
    check("factored_diag reproduces the diagonal of B",
          maximum(abs, Array(factored_diag(F)) .- d) < 1e-14,
          @sprintf("max |Δd| = %.3e", maximum(abs, Array(factored_diag(F)) .- d)))
    # Both normalizations are Vᴴ C V = 1, which is what the resolvent expansion
    # in pencil_probe_duals assumes; check it directly rather than comparing
    # eigenvectors, which are only defined up to phase.
    check("the factored eigenvectors are C-orthonormal",
          opnorm(Vf' * C * Vf - I) < 1e-9,
          @sprintf("|V'CV - I| = %.3e", opnorm(Vf' * C * Vf - I)))
    # And a genuinely non-diagonal B built as a congruence: KᴴK by construction.
    V = randn(ComplexF64, n, 5)
    c = abs.(randn(5)) .+ 0.1
    F2 = FactoredB(V, c)
    B2 = V * Diagonal(c .^ 2) * V'
    Λ2, V2 = factored_pencil_eigen(F2, pencil.whitener, pencil.nullspace)
    resid = opnorm(B2 * V2 - C * V2 * Diagonal(Λ2))
    check("a non-diagonal B solves its own generalized eigenproblem",
          resid < 1e-8 * opnorm(B2), @sprintf("|BV - CVΛ| = %.3e", resid))
end

# =====================================================================
# 2 (anchor a): the pure g basis reproduces production
# =====================================================================

println("\n=== anchor (a): pure g basis vs bounds_from_spectrum")

const PROD = quiet() do
    bounds_from_spectrum(ENV_CPU, SMR, Γ, GS, Γrs; num_pos=K, G₀_uu=G_UU, τs=TAUS,
                         tau_window=0, pencil_cache_max=0, k_uu=0)
end
const VAR_G = run_variant(GS)

println("production bounds = ", PROD.bounds_dual_basis)
println("variant  (k_uu=0) = ", VAR_G.bounds)
const REL_G = maximum(abs.(VAR_G.bounds .- PROD.bounds_dual_basis) ./ PROD.bounds_dual_basis)
check("the variant front end reproduces production on the g basis", REL_G < 1e-10,
      @sprintf("worst relative difference %.3e", REL_G))
check("and it reproduces production's optimal τ",
      all(isapprox(VAR_G.opt_taus[n], PROD.opt_taus[n]; rtol=1e-10, atol=1e-14) for n in 1:K),
      string(VAR_G.opt_taus))
check("m_aug is the g basis's own width", VAR_G.m_aug == K, string(VAR_G.m_aug))

# The bound is a property of the *span*, not of the representative: rotating the
# basis by a unitary transforms C, B and ss_basis by the same congruence, which
# the whole pencil is invariant under. This is what licenses the QR inside
# `augmented_basis`, which may return any orthonormal basis of the same span.
Random.seed!(0xB0A7)
let U = Matrix(qr(randn(ComplexF64, K, K)).Q)
    rotated = run_variant(GS * U)
    rel = maximum(abs.(rotated.bounds .- VAR_G.bounds) ./ VAR_G.bounds)
    check("the bound is invariant under a rotation within the span", rel < 1e-9,
          @sprintf("worst relative difference %.3e", rel))
end

# =====================================================================
# 3 (anchor b): the full space is the ground truth
# =====================================================================
#
# basis = 1 makes every projection the identity, so `augmented_dual_bounds`
# computes the honest full-space dual. Nothing is approximated: C_basis is C,
# Bₙ_basis is Bₙ, ss_basis is ss.

println("\n=== anchor (b): the full-space ground truth")

const FULL_BASIS = Matrix{ComplexF64}(I, N_U, N_U)
const TRUTH = run_variant(FULL_BASIS)
println("full-space bounds = ", TRUTH.bounds)
check("the full basis has m_aug = N_u", TRUTH.m_aug == N_U, string(TRUTH.m_aug))
check("the full-space bounds are finite and positive",
      all(isfinite, TRUTH.bounds) && all(>(0), TRUTH.bounds))
# Non-increasing in n: Bₙ ⪯ Bₙ₋₁ and the probe set shrinks. This holds for the
# true program, so a ground truth that violates it is not one.
check("the ground truth is non-increasing in n",
      all(TRUTH.bounds[n] <= TRUTH.bounds[n-1] * (1 + 1e-8) for n in 2:K),
      string(TRUTH.bounds))

# =====================================================================
# 4 (anchor c): the projected bound under-reports, and the augmentation recovers
# =====================================================================

println("\n=== anchor (c): the drop, and the recovery")

check("the g basis under-reports the ground truth (this is the pathology)",
      VAR_G.bounds[1] < TRUTH.bounds[1] * (1 - 1e-6),
      @sprintf("g basis %.6e vs full space %.6e (%.4f×)", VAR_G.bounds[1],
               TRUTH.bounds[1], VAR_G.bounds[1] / TRUTH.bounds[1]))

# Asym(G⁰ᵤᵤ)'s eigenvectors, exactly. At N_u = 48 a dense eigendecomposition is
# free and gives a *complete* orthonormal set ordered by eigenvalue, which is
# what lets the last point of the sweep close the span exactly. The experiment
# uses reigen_hermitian for the same object at 196,608; check 6 covers that.
const G_UU_DENSE = begin
    M = zeros(ComplexF64, N_U, N_U)
    e = zeros(ComplexF64, N_U)
    for j in 1:N_U
        e[j] = 1
        M[:, j] = G_UU_ASYM * e
        e[j] = 0
    end
    Matrix(Hermitian(M))
end
const UU_EIG = eigen(Hermitian(G_UU_DENSE))
const UU_ORDER = sortperm(UU_EIG.values; rev=true)
const U_UU = UU_EIG.vectors[:, UU_ORDER]
const UU_VALS = UU_EIG.values[UU_ORDER]
@printf("Asym(G⁰ᵤᵤ) spectrum: Λ[1] = %.6e, Λ[%d] = %.6e, positive: %d of %d\n",
        UU_VALS[1], N_U, UU_VALS[end], count(>(0), UU_VALS), N_U)

# K + (N_U - K) = N_U, so the last point closes the span exactly.
const K_UU_SWEEP = [0, 4, 8, 16, 32, N_U - K]
const SWEEP = map(K_UU_SWEEP) do k_uu
    aug = augmented_basis(GS, view(U_UU, :, 1:k_uu))
    res = run_variant(aug.basis)
    (k_uu=k_uu, m_aug=res.m_aug, dropped=aug.num_uu_dropped, bounds=res.bounds)
end

println()
@printf("%6s %6s %5s %14s %14s %8s\n", "k_uu", "m_aug", "drop", "ch1", "ch1/truth", "trace")
for r in SWEEP
    @printf("%6d %6d %5d %14.6e %14.6f %8.4f\n", r.k_uu, r.m_aug, r.dropped,
            r.bounds[1], r.bounds[1] / TRUTH.bounds[1], sum(r.bounds))
end
@printf("%6s %6d %5s %14.6e %14.6f %8.4f\n", "truth", N_U, "-", TRUTH.bounds[1], 1.0,
        sum(TRUTH.bounds))

const CH1 = [r.bounds[1] for r in SWEEP]
const STEPS = diff(CH1) ./ CH1[1:end-1]
check("nothing was dropped by the QR rank guard",
      all(r -> r.dropped == 0, SWEEP),
      "dropped: " * string([r.dropped for r in SWEEP]))
check("m_aug = K + k_uu at every point",
      all(r -> r.m_aug == K + r.k_uu, SWEEP),
      string([r.m_aug for r in SWEEP]))
# Nested spans, so this is arithmetic and not physics. A violation is a bug in
# the front end, which is exactly what makes it a good test.
check("channel 1 is monotone non-decreasing in k_uu", minimum(STEPS) >= -1e-9,
      @sprintf("worst relative step %.3e (negative would be a backward one)",
               minimum(STEPS)))
check("every channel is monotone non-decreasing in k_uu",
      all(SWEEP[i].bounds[n] >= SWEEP[i-1].bounds[n] * (1 - 1e-9)
          for i in 2:length(SWEEP), n in 1:K))
check("the augmentation recovers a real fraction of the gap",
      CH1[end-1] > CH1[1] * 1.05,
      @sprintf("%.6e → %.6e over k_uu = %d → %d", CH1[1], CH1[end-1],
               K_UU_SWEEP[1], K_UU_SWEEP[end-1]))
# The last point spans everything, so the projection is a change of basis and the
# bound must be the ground truth, not merely near it.
const CLOSED = SWEEP[end]
check("a span-closing augmentation reproduces the ground truth exactly",
      maximum(abs.(CLOSED.bounds .- TRUTH.bounds) ./ TRUTH.bounds) < 1e-8,
      @sprintf("worst relative difference %.3e",
               maximum(abs.(CLOSED.bounds .- TRUTH.bounds) ./ TRUTH.bounds)))
check("no augmented bound exceeds the ground truth",
      all(all(r.bounds .<= TRUTH.bounds .* (1 + 1e-8)) for r in SWEEP))

# =====================================================================
# 5: the rank guard
# =====================================================================

println("\n=== the augmentation rank guard")

let dup = hcat(U_UU[:, 1:4], GS[:, 1:2], U_UU[:, 1:2])
    # Columns 5,6 are in span(GS) and columns 7,8 duplicate 1,2, so four of the
    # eight must go and the survivors must still be orthonormal.
    aug = augmented_basis(GS, dup)
    check("columns inside span(g) and duplicates are both dropped",
          aug.num_uu_kept == 4 && aug.num_uu_dropped == 4,
          "kept $(aug.num_uu_kept), dropped $(aug.num_uu_dropped) at $(aug.dropped_cols)")
    check("the surviving basis is orthonormal",
          opnorm(aug.basis' * aug.basis - I) < 1e-11,
          @sprintf("|B'B - I| = %.3e", opnorm(aug.basis' * aug.basis - I)))
    check("the g block is left exactly as it was",
          aug.basis[:, 1:K] == GS)
    # And the dual still runs on it, and agrees with the clean k_uu = 4 point.
    clean = augmented_basis(GS, view(U_UU, :, 1:4))
    dirty_bounds = run_variant(aug.basis).bounds
    clean_bounds = run_variant(clean.basis).bounds
    rel = maximum(abs.(dirty_bounds .- clean_bounds) ./ clean_bounds)
    check("the deduplicated basis gives the clean basis's bound", rel < 1e-9,
          @sprintf("worst relative difference %.3e", rel))
end

# =====================================================================
# 6: uu_eigenbasis, the routine the narval job actually uses
# =====================================================================
#
# The sweep above uses a dense eigendecomposition because it needs a complete
# ordered set. The job uses reigen_hermitian on the same operator, so check that
# it returns eigenpairs of the thing it claims to and that its leading span
# matches the dense one's.

println("\n=== uu_eigenbasis (reigen_hermitian on Asym(G⁰ᵤᵤ))")

# Two identical cubes make Asym(G⁰ᵤᵤ) symmetric under the exchange, so its
# spectrum comes in near-degenerate pairs and a `k` cutting through one of them
# leaves the leading span genuinely ambiguous. Pick a `k` that sits on a real
# gap, so the span comparison below is testing the solver and not the ordering
# of two equal eigenvalues.
const UU_GAPS = [(k, UU_VALS[k] / UU_VALS[k+1]) for k in 4:12]
const UU_K = let hit = findfirst(p -> last(p) > 1.5, UU_GAPS)
    hit === nothing ? first(UU_GAPS[argmax(last.(UU_GAPS))]) : first(UU_GAPS[hit])
end
@printf("leading spectral gaps: %s\n",
        join((@sprintf("Λ%d/Λ%d=%.3f", k, k + 1, r) for (k, r) in UU_GAPS), " "))
@printf("using k = %d (Λ[k]/Λ[k+1] = %.4f)\n", UU_K, UU_VALS[UU_K] / UU_VALS[UU_K+1])

let k = UU_K
    uu = quiet() do
        uu_eigenbasis(ENV_CPU, G_UU_ASYM, k; oversamples=10, power_iters=4)
    end
    check("reigen_hermitian returns k components", size(uu.vectors, 2) == k,
          string(size(uu.vectors)))
    rel = abs.(uu.values .- UU_VALS[1:k]) ./ abs.(UU_VALS[1:k])
    check("its eigenvalues match the dense ones", maximum(rel) < 1e-8,
          @sprintf("worst relative difference %.3e", maximum(rel)))
    res = uu_residuals(G_UU_ASYM, uu.vectors, uu.values, [1, k])
    check("its eigenpair residuals are small", maximum(res) < 1e-7,
          @sprintf("‖Av-λv‖/|λ| = %s", string(res)))
    # The two spans agree, so the sweep's conclusions do not depend on which
    # routine produced the augmentation.
    Ud = UU_EIG.vectors[:, UU_ORDER[1:k]]
    gap = opnorm(uu.vectors * (uu.vectors' * Ud) - Ud)
    check("its span matches the dense leading span", gap < 1e-6,
          @sprintf("subspace residual %.3e", gap))
    br = run_variant(augmented_basis(GS, uu.vectors).basis).bounds
    bd = run_variant(augmented_basis(GS, view(U_UU, :, 1:k)).basis).bounds
    check("and the bound it gives matches the dense basis's",
          maximum(abs.(br .- bd) ./ bd) < 1e-6,
          @sprintf("worst relative difference %.3e", maximum(abs.(br .- bd) ./ bd)))
end

# =====================================================================
# 7: the projected operators are what they claim to be
# =====================================================================
#
# `PSC._bounds_front_end_augmented` builds C_basis and D_basis out of
# `W = basisᴴgs` and one Green sweep instead of sweeping the whole of C the way
# `_bounds_front_end_dense` does. Check that against a literal `basisᴴ C basis` on
# a basis where nothing is diagonal, and check that the probes it builds are the
# production probes.

println("\n=== the augmented front end vs a literal projection")

let aug = augmented_basis(GS, view(U_UU, :, 1:7))
    basis = aug.basis
    proj = quiet() do
        PSC._bounds_front_end_augmented(ENV_CPU, GS, basis, Γ, ζ, S_PROJ, G_UU_ASYM,
                                        K, SENDER_SIZE)
    end
    # The probes are Πₛ·GS's reverse Gram-Schmidt, the g columns alone, whatever the
    # basis is. Same call, so this is exact rather than close.
    check("the front end's probes are the production probes", proj.ss == SS)
    check("ss_basis is basisᴴss",
          opnorm(proj.ss_basis - basis' * SS) < 1e-12,
          @sprintf("|Δ| = %.3e", opnorm(proj.ss_basis - basis' * SS)))
    C_full = zeros(ComplexF64, N_U, N_U)
    e = zeros(ComplexF64, N_U)
    for j in 1:N_U
        e[j] = 1
        Πₛe = S_PROJ * e
        col = (1 / ζ) .* Πₛe .+ GS * ((GS' * e) .* Γ) .+ G_UU_ASYM * e
        C_full[:, j] = col
        e[j] = 0
    end
    C_ref = basis' * C_full * basis
    relC = opnorm(proj.C_basis - C_ref) / opnorm(C_ref)
    check("C_basis matches basisᴴ C(1) basis", relC < 1e-11,
          @sprintf("relative difference %.3e", relC))

    # D = (−G⁰ᵤᵣ)ᵃ₊ − ζ⁻¹Πᵣ, so C(τ) = C(1) − (1−τ)D. Build Πᵣ = 1 − Πₛ literally.
    D_full = GS * Diagonal(Γ) * GS'
    for j in 1:N_U
        e[j] = 1
        D_full[:, j] .-= (1 / ζ) .* (e .- S_PROJ * e)
        e[j] = 0
    end
    D_ref = basis' * D_full * basis
    relD = opnorm(proj.D_basis - D_ref) / opnorm(D_ref)
    check("D_basis matches basisᴴ D basis", relD < 1e-11,
          @sprintf("relative difference %.3e", relD))

    # Bₙ's factored form, against the literal congruence.
    for n in 1:K
        Vn = proj.W[:, n:K]
        c = sqrt.((4 / ζ) .* Γ[n:K])
        F = FactoredB(Vn, c)
        B_ref = basis' * ((4 / ζ) .* (GS[:, n:K] * Diagonal(Γ[n:K]) * GS[:, n:K]')) * basis
        B_fac = (c .* (Vn'))' * (c .* (Vn'))
        rel = opnorm(B_fac - B_ref) / opnorm(B_ref)
        check("Bₙ's factored form matches basisᴴBₙbasis at n = $n", rel < 1e-11,
              @sprintf("relative difference %.3e", rel))
        check("factored_diag(Bₙ) is diag(basisᴴBₙbasis) at n = $n",
              maximum(abs, Array(factored_diag(F)) .- real.(diag(B_ref))) <
                  1e-11 * maximum(abs, real.(diag(B_ref))))
    end
end

# =====================================================================
# 8 (anchor d): --k-uu 0 is bit-for-bit the pre-augmentation output
# =====================================================================
#
# The one property the augmentation must have that is not about physics: a job that
# does not ask for it gets exactly what it got before the feature existed. Not
# "agrees to 1e-12": the k_uu = 0 branch is supposed to be the *same instruction
# sequence*, so the numbers must be identical to the last bit.
#
# GOLDEN below was recorded on the code as it stood immediately before --k-uu
# landed, on this fixture. A mismatch is one of two things and the printed relative
# difference says which: exactly zero means bit-equal (what is expected); tiny but
# nonzero means a different LAPACK/BLAS on this machine rather than a regression,
# and is reported loudly but not failed; anything larger is a regression.

println("\n=== anchor (d): --k-uu 0 reproduces the pre-augmentation output")

# bounds_from_spectrum(...; num_pos=K, G₀_uu=G_UU, τs=TAUS, tau_window=0,
#                      pencil_cache_max=0) on this fixture, before --k-uu existed.
const GOLDEN_BOUNDS = [0.5594540057364662, 0.11052890874517868, 0.006102231764398426]
const GOLDEN_TAUS = [0.7532889043741062, 0.0, 0.0]

function check_bitequal(name, got, want)
    rel = maximum(abs.(got .- want) ./ max.(abs.(want), eps()))
    bit = all(got .== want)
    if bit
        return check(name, true, "bit-identical")
    elseif rel < 1e-12
        return check(name, true,
                     @sprintf("NOT bit-identical but within %.3e", rel) *
                     " (a different LAPACK on this machine, not a regression)")
    end
    return check(name, false, @sprintf("worst relative difference %.3e", rel))
end

check_bitequal("k_uu = 0 reproduces the recorded bounds", PROD.bounds_dual_basis,
               GOLDEN_BOUNDS)
check_bitequal("k_uu = 0 reproduces the recorded optimal τ", PROD.opt_taus, GOLDEN_TAUS)
check("k_uu = 0 reports that it did not augment", !PROD.augmentation.augmented)
check("and reports the plain m as m_aug", PROD.augmentation.m_aug == K,
      string(PROD.augmentation.m_aug))

# The production defaults (windowed sweep, pencil cache) must land on the same
# numbers as the unwindowed reference; test/tau_search.jl establishes that in
# general, and this is the k_uu = 0 instance of it.
const PROD_DEFAULTS = quiet() do
    bounds_from_spectrum(ENV_CPU, SMR, Γ, GS, Γrs; num_pos=K, G₀_uu=G_UU, τs=TAUS,
                         k_uu=0)
end
check("the windowed/cached defaults agree with the unwindowed reference exactly",
      PROD_DEFAULTS.bounds_dual_basis == PROD.bounds_dual_basis,
      string(PROD_DEFAULTS.bounds_dual_basis))

# =====================================================================
# 9 (anchor e): the production augmented path recovers the ground truth
# =====================================================================
#
# Anchor (c) drove the recovery through the experiment's loop with a dense,
# exactly-ordered U_uu. This is the same recovery through `bounds_from_spectrum`
# itself: it computes its own U_uu with `reigen_hermitian`, orthonormalizes,
# switches Bₙ to its factored form, and reports the augmented bound. The
# `k_uu` values are clamped internally to `N_u − m = 45`, so the last two points
# both close the span and must both be the ground truth.

println("\n=== anchor (e): the production --k-uu path on the miniature")

run_prod(k_uu; kw...) = quiet() do
    bounds_from_spectrum(ENV_CPU, SMR, Γ, GS, Γrs; num_pos=K, G₀_uu=G_UU, τs=TAUS,
                         k_uu=k_uu, kw...)
end

const PROD_SWEEP = map(k -> (k_uu=k, res=run_prod(k)), [0, 4, 16, 45, 512])
@printf("\n%6s %6s %6s %5s %14s %14s\n", "k_uu", "k_eff", "m_aug", "drop", "ch1", "ch1/truth")
for r in PROD_SWEEP
    a = r.res.augmentation
    @printf("%6d %6d %6d %5d %14.6e %14.6f\n", r.k_uu, a.k_uu_effective, a.m_aug,
            a.num_uu_dropped, r.res.bounds_dual_basis[1],
            r.res.bounds_dual_basis[1] / TRUTH.bounds[1])
end

const PROD_CH1 = [r.res.bounds_dual_basis[1] for r in PROD_SWEEP]
check("the production sweep is monotone non-decreasing in k_uu",
      minimum(diff(PROD_CH1) ./ PROD_CH1[1:end-1]) >= -1e-8,
      @sprintf("worst relative step %.3e",
               minimum(diff(PROD_CH1) ./ PROD_CH1[1:end-1])))
check("no production bound exceeds the ground truth",
      all(all(r.res.bounds_dual_basis .<= TRUTH.bounds .* (1 + 1e-8)) for r in PROD_SWEEP))
check("k_uu is clamped to the N_u − m directions that exist",
      PROD_SWEEP[end].res.augmentation.k_uu_effective == N_U - K &&
          PROD_SWEEP[end].res.augmentation.m_aug == N_U,
      "k_eff = $(PROD_SWEEP[end].res.augmentation.k_uu_effective), " *
      "m_aug = $(PROD_SWEEP[end].res.augmentation.m_aug)")
let closed = PROD_SWEEP[end].res.bounds_dual_basis,
    rel = maximum(abs.(PROD_SWEEP[end].res.bounds_dual_basis .- TRUTH.bounds) ./ TRUTH.bounds)
    check("a span-closing production run reproduces the ground truth", rel < 1e-8,
          @sprintf("worst relative difference %.3e", rel))
end
check("the augmentation is reported as having happened",
      all(r.res.augmentation.augmented for r in PROD_SWEEP if r.k_uu > 0))
check("the rank guard drops nothing on a converged reigen_hermitian U_uu",
      all(r.res.augmentation.num_uu_dropped == 0 for r in PROD_SWEEP),
      string([r.res.augmentation.num_uu_dropped for r in PROD_SWEEP]))
check("the Asym(G⁰ᵤᵤ) eigenpair residuals are recorded and small",
      all(isempty(r.res.augmentation.uu_residuals) ||
          maximum(r.res.augmentation.uu_residuals) < 1e-7 for r in PROD_SWEEP),
      string(PROD_SWEEP[end].res.augmentation.uu_residuals))

# =====================================================================
# 10 (anchor f): the threshold decides, and says so
# =====================================================================
#
# `augment_threshold` is what keeps the near-field half of a sweep on the code that
# was calibrated for it. A point at or above the threshold has to run the k_uu = 0
# path exactly (same numbers, same `augmented = false`) and to say in the log
# that it chose not to augment, because "why is this point not augmented?" is
# otherwise unanswerable from the output.

println("\n=== anchor (f): the augment_threshold")

let blocked = run_prod(512; augment_threshold=K) # m = K, so m >= threshold
    check("m >= augment_threshold does not augment", !blocked.augmentation.augmented)
    check("and is bit-identical to k_uu = 0",
          blocked.bounds_dual_basis == PROD_DEFAULTS.bounds_dual_basis &&
              blocked.opt_taus == PROD_DEFAULTS.opt_taus,
          string(blocked.bounds_dual_basis))
end
let allowed = run_prod(4; augment_threshold=K + 1) # m = K < threshold
    check("m < augment_threshold augments", allowed.augmentation.augmented &&
          allowed.augmentation.m_aug == K + 4, string(allowed.augmentation.m_aug))
end

# The log line, captured rather than eyeballed: it is the only thing that
# distinguishes "not augmented because of the threshold" from "not augmented
# because --k-uu 0" in a production job's output.
let io = IOBuffer()
    with_logger(SimpleLogger(io, Logging.Info)) do
        bounds_from_spectrum(ENV_CPU, SMR, Γ, GS, Γrs; num_pos=K, G₀_uu=G_UU, τs=TAUS,
                             k_uu=512, augment_threshold=K)
    end
    log = String(take!(io))
    check("the threshold decision is logged",
          occursin("Not augmenting", log) && occursin("augment-threshold", log))
end
let io = IOBuffer()
    with_logger(SimpleLogger(io, Logging.Info)) do
        bounds_from_spectrum(ENV_CPU, SMR, Γ, GS, Γrs; num_pos=K, G₀_uu=G_UU, τs=TAUS,
                             k_uu=0)
    end
    check("--k-uu 0 says so in the log",
          occursin("Not augmenting: --k-uu 0", String(take!(io))))
end

# =====================================================================
# 11: the memory guardrails
# =====================================================================
#
# `plan_uu_solve` is what stands between a 4 λ far-field point and an OOM three
# hours into a bounds job. There is no GPU here, so what is checkable is the
# arithmetic it is built on and the fact that the CPU path does not gate on a
# device budget it does not have.

println("\n=== plan_uu_solve and the footprint arithmetic")

check("uu_sketch_bytes is the three N_u × (k + p) matrices",
      uu_sketch_bytes(196608, 512, 50) == 3 * 196608 * 562 * 16,
      @sprintf("%.2f GiB at the 1 λ point", uu_sketch_bytes(196608, 512, 50) / 2^30))
check("augmented_footprint_bytes reduces to bounds_footprint_bytes at m_aug = m",
      augmented_footprint_bytes(196608, 15, 15) == bounds_footprint_bytes(196608, 15))
check("and counts basis + ss + working matrix when it does not",
      augmented_footprint_bytes(196608, 15, 527) == (2 * 527 + 15) * 196608 * 16)
let p = plan_uu_solve(ENV_CPU, N_U, K, 8)
    check("a CPU run is not gated on a device budget",
          p.oversamples == UU_OVERSAMPLES && p.budget_bytes == 0)
end

# =====================================================================
# 12: the adaptive --k-uu clip, and the cost model's copy of it
# =====================================================================
#
# `--augment-threshold 1000` puts the ceiling at `m_aug = m + k_uu = 1512`, and at
# the larger universes three `N_u × m_aug` matrices are past an A100-40. `clip_k_uu`
# is what keeps that from being an OOM three hours in: it cuts the effective `k_uu`
# to the largest one that fits, and refuses below `K_UU_CLIP_FLOOR`.
#
# Two things are checked. The arithmetic itself, at fabricated production sizes,
# because "does 4 λ fit?" is the question the clip exists to answer and the answer
# has to be in the test rather than in a comment. And the *agreement* between
# `max_k_uu_for_budget` here and `augment_k_uu_cap` in `bench/cost_model.jl`, which
# is the invariant that makes a request and the job it sizes describe the same work:
# the two are separate implementations on purpose (the sizer must start without
# CUDA) and nothing but this stops them drifting.

println("\n=== the adaptive --k-uu clip")

check("the production augment threshold is the one the decision fixed",
      DEFAULT_AUGMENT_THRESHOLD == 1000, string(DEFAULT_AUGMENT_THRESHOLD))

# The three constraints solved for k, checked against the predicates they came
# from: at the returned k everything fits, at k + 1 something does not.
let N_u = 393216, m = 800, budget = 20_000_000_000
    k = max_k_uu_for_budget(N_u, m, budget)
    fits(kk) = augmented_footprint_bytes(N_u, m, m + kk) <= budget &&
               (2 * kk + m + kk) * N_u * 16 <= budget &&
               PSC.RSVD_PEAK_FUDGE * uu_sketch_bytes(N_u, kk, UU_MIN_OVERSAMPLES) <= budget
    check("max_k_uu_for_budget returns a k that fits all three predicates", fits(k),
          "k = $(k)")
    check("and k + 1 does not", !fits(k + 1))
end

# The production table. `N_u` is 3 × 2 × cells for the [sender; receiver] universe:
# 196,608 at 1 λ (32³ a body), and 2× / 4× that at 2 λ and 4 λ. The budget is the
# runtime's: 90% of the card, less the Green operator's own workspace.
const A100_40_TOTAL = 42_505_273_344 # what CUDA.total_memory() reports on an A100-40
budget_for(N_u, total) = floor(Int, 0.9 * total) - PSC.gila_workspace_bytes(N_u)

@printf("\n%10s %10s %8s %10s %12s %12s\n",
        "universe", "N_u", "m", "k_uu eff", "front GiB", "budget GiB")
for (label, N_u) in (("1 λ", 196_608), ("2 λ", 393_216), ("4 λ", 786_432)),
    m in (500, 999)

    b = budget_for(N_u, A100_40_TOTAL)
    k = min(DEFAULT_K_UU, max_k_uu_for_budget(N_u, m, b))
    @printf("%10s %10d %8d %10s %12.2f %12.2f\n", label, N_u, m,
            k < K_UU_CLIP_FLOOR ? "REFUSED" : string(k),
            augmented_footprint_bytes(N_u, m, m + max(k, 0)) / 2^30, b / 2^30)
end

check("1 λ augments at the full k_uu at both ends of the band",
      min(DEFAULT_K_UU, max_k_uu_for_budget(196_608, 500, budget_for(196_608, A100_40_TOTAL))) == DEFAULT_K_UU &&
      min(DEFAULT_K_UU, max_k_uu_for_budget(196_608, 999, budget_for(196_608, A100_40_TOTAL))) == DEFAULT_K_UU)
check("2 λ augments at the full k_uu at both ends of the band",
      min(DEFAULT_K_UU, max_k_uu_for_budget(393_216, 500, budget_for(393_216, A100_40_TOTAL))) == DEFAULT_K_UU &&
      min(DEFAULT_K_UU, max_k_uu_for_budget(393_216, 999, budget_for(393_216, A100_40_TOTAL))) == DEFAULT_K_UU)
check("4 λ still fits the full k_uu at m = 500",
      min(DEFAULT_K_UU, max_k_uu_for_budget(786_432, 500, budget_for(786_432, A100_40_TOTAL))) == DEFAULT_K_UU)
check("4 λ is clipped in the middle of the new band",
      let k = max_k_uu_for_budget(786_432, 700, budget_for(786_432, A100_40_TOTAL))
          K_UU_CLIP_FLOOR <= k < DEFAULT_K_UU
      end,
      string(max_k_uu_for_budget(786_432, 700, budget_for(786_432, A100_40_TOTAL))))
check("and refused at the top of it, where not even the floor fits",
      max_k_uu_for_budget(786_432, 999, budget_for(786_432, A100_40_TOTAL)) < K_UU_CLIP_FLOOR,
      string(max_k_uu_for_budget(786_432, 999, budget_for(786_432, A100_40_TOTAL))))

check("a CPU run is never clipped",
      let c = clip_k_uu(ENV_CPU, 786_432, 999, 512)
          c.k_uu == 512 && !c.clipped && isempty(c.reason)
      end)

# The mirror. Same arithmetic, two implementations, one grid.
isdefined(Main, :CostModel) || include(joinpath(@__DIR__, "..", "bench", "cost_model.jl"))
Main.CostModel.load_coefficients!(joinpath(@__DIR__, "..", "bench"))
let disagreements = Tuple{Int,Int,Float64}[]
    for N_u in (48, 12_288, 196_608, 393_216, 786_432), m in (1, 15, 500, 700, 999, 2000),
        budget in (1.0e9, 5.0e9, 1.8e10, 3.6e10, 7.2e10)

        a = max_k_uu_for_budget(N_u, m, budget)
        b = Main.CostModel.augment_k_uu_cap(N_u, m, budget)
        a == b || push!(disagreements, (N_u, m, budget))
    end
    check("augment_k_uu_cap reproduces max_k_uu_for_budget on every fabricated size",
          isempty(disagreements),
          isempty(disagreements) ? "150 sizes" : string(first(disagreements, 3)))
end
check("the cost model's clip floor is the runtime's",
      Main.CostModel.BOUNDS_K_UU_CLIP_FLOOR == K_UU_CLIP_FLOOR)
check("the cost model's threshold default is the runtime's",
      Main.CostModel.BOUNDS_AUGMENT_THRESHOLD_DEFAULT == DEFAULT_AUGMENT_THRESHOLD)
check("the cost model's k_uu default is the runtime's",
      Main.CostModel.BOUNDS_K_UU_DEFAULT == DEFAULT_K_UU)
check("the cost model's gila workspace is the runtime's",
      Main.CostModel.gila_workspace_bytes(786_432) == PSC.gila_workspace_bytes(786_432))

# End to end through the model: a 4 λ point in the middle of the new band, sized on
# an A100-40, has to come back with the clipped k_uu and not the requested one.
let c = Main.CostModel.with_augmentation(
            Main.CostModel.coefficients_for("narval");
            k_uu=DEFAULT_K_UU, threshold=DEFAULT_AUGMENT_THRESHOLD),
    # The 4 λ universe: 3 · 64 · 64 · 32 = 393,216 currents a body, 786,432 over the
    # [sender; receiver] pair. The cells are what `universe_length` reads.
    pt = Main.CostModel.SRPoint((64, 64, 32), (64, 64, 32); scale=(1//32, 1//32, 1//32),
                                separation=1//2, rank=4000, oversamples=50,
                                power_iters=14, threads=4, num_pos=700)

    Main.CostModel.universe_length(pt) == 786_432 ||
        error("the 4 λ fixture is $(Main.CostModel.universe_length(pt)) currents, not 786,432")
    plain = Main.CostModel.bounds_augment(pt, c, 700)
    sized = Main.CostModel.bounds_augment(pt, c, 700; vram_capacity_bytes=40e9)
    check("with no card named the model charges the unclipped k_uu",
          plain.k_uu == DEFAULT_K_UU && plain.clip == :none, string(plain.k_uu))
    check("named an A100-40 it charges the clipped one",
          sized.clip == :budget && K_UU_CLIP_FLOOR <= sized.k_uu < DEFAULT_K_UU &&
              sized.m_aug == 700 + sized.k_uu,
          "k_uu = $(sized.k_uu), m_aug = $(sized.m_aug)")
    check("and the clip it charges is the one the runtime arithmetic gives",
          sized.k_uu == Main.CostModel.augment_k_uu_cap(
              786_432, 700, Main.CostModel.augment_budget_bytes(786_432, 40e9)),
          string(sized.k_uu))
    infeasible = Main.CostModel.bounds_augment(pt, c, 999; vram_capacity_bytes=40e9)
    check("a point no card can augment is labelled infeasible, not silently sized",
          infeasible.clip == :infeasible && infeasible.k_uu == K_UU_CLIP_FLOOR)
    roomy = Main.CostModel.bounds_augment(pt, c, 999; vram_capacity_bytes=80e9)
    check("and the same point on an H100 is not clipped at all",
          roomy.clip == :none && roomy.k_uu == DEFAULT_K_UU, string(roomy.k_uu))
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
