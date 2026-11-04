## The bounds stage on a tiny CPU system: the low-rank Eq. (37) dual against a dense
## build of 𝟙(τ) − H, its feasibility edge, the sanity properties (a)–(c) of the plan,
## the τ search, and the augmentation.
##
## Run from the repo root:
##     julia --project=. test/bounds_core.jl

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using LinearAlgebra
using LinearMaps
using Printf
using Random
using Logging

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-62s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

const ROOT = mktempdir(; cleanup=true)
const ENV_CPU = ComputeEnvironment(mkpath(joinpath(ROOT, "preload")), mkpath(joinpath(ROOT, "project")),
                                   mkpath(joinpath(ROOT, "scratch")), GPUChoice(false, -1))
const SMR = SMRSystem((2, 2, 2), (1//32, 0//1, 0//1), (2, 2, 2),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im; refine_gap=false)
const N_S = dof_length(sender_mesh(SMR))
const N_U = N_S + dof_length(receiver_mesh(SMR))
const K = 8 # positive Γ, i.e. channels
const ζ = abs(susceptibility(SMR))^2 / imag(susceptibility(SMR))

# A synthetic spectrum at production magnitude (κ̃ = ζΓ from 1 down), with four negative Γ.
Random.seed!(0xBEEF)
const Γ = vcat(exp10.(range(log10(1 / ζ), log10(1e-4 / ζ), length=K)),
               -exp10.(range(log10(1e-3 / ζ), log10(5e-2 / ζ), length=4)))
const V = Matrix(qr(randn(ComplexF64, N_U, length(Γ))).Q)[:, 1:length(Γ)]
const Γrs = collect(exp10.(range(log10(1 / ζ), log10(1e-6 / ζ), length=K)))
const G_UU = load_asym_green_function(ENV_CPU, SMR, Design)
const G_DENSE = PSC._dense_matrix(LinearMap(G_UU), false)
println("N_u = $(N_U), N_s = $(N_S), channels = $(K), m_kept = $(length(Γ))")

# μₚ σ₁(Πₛ(𝟙(τ) − H)⁻¹Πₛ) built densely, with G compressed to the basis P as the low-rank form does.
function dense_operator(P, n, τ, μ_p; Γ=Γ)
    A = V * Diagonal(Γ) * V'
    An = V[:, n:end] * Diagonal(Γ[n:end]) * V[:, n:end]'
    Gc = P * (P' * G_DENSE * P) * P'
    one_τ = Diagonal(vcat(ones(N_S), fill(1 - τ, N_U - N_S)))
    return Hermitian(one_τ + ζ .* (τ .* A .+ Gc .- An ./ μ_p))
end
dense_mu(P, n, τ, μ_p; Γ=Γ) = μ_p * eigmax(Hermitian(inv(dense_operator(P, n, τ, μ_p; Γ=Γ))[1:N_S, 1:N_S]))
front_for(P; Γ=Γ) = (; PSC.bounds_front_end(V, P, Γ, LinearMap(G_DENSE), N_S)..., ζ=ζ)

const P_FULL = augmented_basis(V, randn(ComplexF64, N_U, N_U - length(Γ))).basis
const P_PART = augmented_basis(V, randn(ComplexF64, N_U, 8)).basis # 20 < N_s columns

println("\n=== low-rank dual vs dense 𝟙(τ) − H")
# The weak-coupling case (Γ scaled down, as in the far field) reaches τ < 0 where R(τ) is indefinite.
for (name, P, Γ) in (("full basis", P_FULL, Γ), ("partial basis", P_PART, Γ), ("weak coupling", P_FULL, 1e-3 .* Γ))
    dop(P, n, τ, μ_p) = dense_operator(P, n, τ, μ_p; Γ=Γ)
    dmu(P, n, τ, μ_p) = dense_mu(P, n, τ, μ_p; Γ=Γ)
    front = front_for(P; Γ=Γ)
    check("$name: sender_spanned = $(size(P, 2) >= N_S)", front.sender_spanned == (size(P, 2) >= N_S))
    worst = 0.0
    negative_seen = searched = refused = missed = 0
    for τ in (-1e4, -1e3, -100.0, -30.0, -10.0, -3.0, -1.0, -0.3, 0.0, 0.3, 0.7, 0.95), n in (1, 3, K), t in (0.01, 0.5, 0.99)
        curve = PSC.dual_curve(front, tau_prep(front, τ), n)
        if isnothing(curve)
            τ < 0 || error("dual_curve refused τ = $τ ≥ 0")
            # refused: no μₚ > 0 may make the dense 𝟙(τ) − H ≻ 0
            t == 0.01 && (refused += 1; missed += count(μ -> isposdef(dop(P, n, τ, μ)), exp10.(-12:0.05:16)))
            continue
        end
        τ < 0 && (negative_seen += 1)
        τ < 0 && !isposdef(dop(P, n, τ, 1 / τ)) && (searched += 1) # R(τ) indefinite: whitened at s₀ > 0
        μₙ, μ_lo, μ_hi = curve
        μ_p = μ_lo * (min(μ_hi, 100μ_lo) / μ_lo)^t # across the feasible interval, in log
        worst = max(worst, abs(μₙ(μ_p)[1] - dmu(P, n, τ, μ_p)) / dmu(P, n, τ, μ_p))
    end
    check("$name: μₙ(μₚ, τ) matches the dense build", worst < 1e-10, @sprintf("max rel diff %.2e", worst))
    check("$name: some τ < 0 is feasible and checked", negative_seen > 0, "$(negative_seen) τ < 0 evaluations")
    # The synthetic spectra never make R(τ) indefinite at a feasible τ; the real λ/4 fronts do (verified off-line).
    println("    $name: $(searched) evaluations whitened at s₀ ≠ τ")
    check("$name: a refused τ has no feasible μₚ on a dense grid", missed == 0, "$(refused) refused, $(missed) feasible grid points")

    slope_err, min_err = 0.0, 0.0
    for τ in (-10.0, -0.3, 0.0, 0.5, 0.95), n in (1, 3, K)
        prep = tau_prep(front, τ)
        curve = PSC.dual_curve(front, prep, n)
        isnothing(curve) && continue
        μₙ, μ_lo, μ_hi = curve
        x = (log(μ_lo) + 0.4 * (log(min(μ_hi, 100μ_lo)) - log(μ_lo)), 1e-5) # (point, finite-difference step)
        fd = (μₙ(exp(x[1] + x[2]))[1] - μₙ(exp(x[1] - x[2]))[1]) / 2x[2]
        slope_err = max(slope_err, abs(μₙ(exp(x[1]))[2] - fd) / μₙ(exp(x[1]))[1])
        fine = minimum(μₙ(exp(y))[1] for y in range(log(μ_lo), log(min(μ_hi, 100μ_lo)), length=2001)[2:end-1])
        min_err = max(min_err, PSC.system_dual(front, prep, n).μ / fine - 1)
    end
    check("$name: the slope in log μₚ matches finite differences", slope_err < 1e-5, @sprintf("max rel err %.2e", slope_err))
    check("$name: the slope search reaches the fine-grid minimum", min_err < 1e-5, @sprintf("max excess %.2e", min_err))

    println("--- feasibility edge ($name)")
    edge_ok = guard_ok = true
    upper_seen = false
    for τ in (-30.0, -3.0, -1.0, -0.3, 0.0, 0.5, 0.95), n in (1, K)
        curve = PSC.dual_curve(front, tau_prep(front, τ), n)
        isnothing(curve) && continue
        μₙ, μ_lo, μ_hi = curve
        edge_ok &= isposdef(dop(P, n, τ, μ_lo * (1 + 1e-6))) &&
                   !isposdef(dop(P, n, τ, μ_lo * (1 - 1e-6)))
        guard_ok &= μₙ(μ_lo * (1 - 1e-6))[1] == Inf && (!isfinite(μ_hi) || μₙ(μ_hi * (1 + 1e-6))[1] == Inf)
        if isfinite(μ_hi)
            upper_seen = true
            edge_ok &= isposdef(dop(P, n, τ, μ_hi * (1 - 1e-6))) &&
                       !isposdef(dop(P, n, τ, μ_hi * (1 + 1e-6)))
        end
    end
    check("$name: 𝟙(τ) − H ≻ 0 exactly on (μ_lo, μ_hi)", edge_ok, upper_seen ? "both edges seen" : "no finite μ_hi")
    check("$name: μₙ is Inf outside (μ_lo, μ_hi)", guard_ok)
end

# τ = −1, R(s) = R(τ) − (s − τ)·rest = diag(s, 2 − s): ≻ 0 only on 0 < s < 2, best at s = 1.
let w = PSC.whitening_point(Matrix(Diagonal([-1.0, 3.0])), Matrix(Diagonal([-1.0, 1.0])), -1.0)
    check("whitening_point finds the most positive definite R(s) when R(τ) is indefinite",
          abs(w.s - 1) < 0.15 && abs(w.λ - 1) < 0.15, @sprintf("s₀ = %.3f, λ_min = %.3f", w.s, w.λ))
end

println("\n=== augmentation")
let U = randn(ComplexF64, N_U, 5)
    aug = augmented_basis(V, hcat(U, V[:, 1])) # the last column is already in span(g)
    check("the g block is left untouched", aug.basis[:, 1:length(Γ)] == V)
    check("the basis is orthonormal", opnorm(aug.basis' * aug.basis - I) < 1e-12)
    check("a column already in span(g) is dropped", aug.num_uu_kept == 5 && aug.num_uu_dropped == 1,
          "kept $(aug.num_uu_kept), dropped $(aug.num_uu_dropped)")
end

println("\n=== fitting the augmentation to the device")
# 1000 tall columns at N_u = 10^5: the sketch allows floor(1000 / (3 · 1.554)) = 214 columns.
let N = 100_000, budget = 1000 * 16 * N
    quiet(f) = with_logger(f, SimpleLogger(stderr, Logging.Error))
    free = fit_augmentation(N, 100, 512, nothing)
    check("no budget: k_uu and oversamples untouched", free == (k_uu=512, oversamples=50), string(free))
    small = quiet(() -> fit_augmentation(N, 100, 180, budget))
    check("a k_uu that fits is kept and the oversamples cut to fit", small == (k_uu=180, oversamples=34),
          string(small))
    big = quiet(() -> fit_augmentation(N, 100, 512, budget))
    check("a k_uu that does not fit is clipped to the sketch limit", big.k_uu == 214 - 10 && big.oversamples == 10,
          string(big))
    refused = try quiet(() -> fit_augmentation(N, 600, 512, budget)); false catch; true end
    check("a point that cannot afford K_UU_FLOOR is refused", refused)
end

quiet(f) = with_logger(f, SimpleLogger(stderr, Logging.Error))
const TAUS = range(-1.0, 1.0, length=21)

# (a) ≤ 1, (b) non-increasing in n, (c) σₙ² ≤ Eq. (36) = new_analytical_bounds². For a
# saturated channel (2κ̃ ≥ 1, Eq. (36) = 1) the infimum is the τ → 1 limit, which the τ
# search only approaches, so (a) and that half of (c) are checked at τ = 1 − 1e-6 on `front`.
function properties(name, res, front)
    d = res.bounds_dual_basis
    eq36 = res.new_analytical_bounds[1:res.num_pos]
    saturated = findall(==(1), eq36)
    near_one = tau_prep(front, 1 - 1e-6)
    limit = [sqrt(PSC.system_dual(front, near_one, n).μ) for n in union(1, saturated)]
    check("$name: (a) the dual tends to ≤ 1 as τ → 1", all(limit .<= 1 + 1e-4),
          @sprintf("max %.6f over %d saturated channels (search reached %.4f)", maximum(limit), length(saturated), maximum(d)))
    rise = maximum(diff(d) ./ d[1:end-1]; init=0.0)
    check("$name: (b) non-increasing in n", rise <= 1e-4, @sprintf("worst relative rise %.2e", rise))
    free = setdiff(1:res.num_pos, saturated)
    ratio = d[free] .^ 2 ./ eq36[free] .^ 2
    check("$name: (c) σₙ² ≤ Eq. (36) off saturation", all(ratio .<= 1 + 1e-6),
          @sprintf("max dual²/Eq36 = %.6f over %d channels", maximum(ratio; init=0.0), length(free)))
    @printf("    %s: dual = %s\n", name, string(round.(d; sigdigits=4)))
end

println("\n=== properties on the synthetic spectrum")
run_synthetic(; kwargs...) = quiet() do
    bounds_from_spectrum(ENV_CPU, SMR, Γ, V, Γrs; G₀_uu=G_UU, τs=TAUS, kwargs...)
end
const FULL = run_synthetic(k_uu=N_U)
const PART = run_synthetic(k_uu=8)
check("k_uu ≥ N_u - m_kept spans the universe", FULL.m_aug == N_U, "m_aug = $(FULL.m_aug)")
check("channels are the positive Γ", FULL.num_pos == K)
properties("synthetic, full basis", FULL, front_for(P_FULL))
properties("synthetic, k_uu = 8", PART, front_for(P_PART))
check("true_bounds are positive and passive", all(0 .< FULL.true_bounds .<= 1))

println("\n=== properties on the tiny real system")
let G_rs = green_map(ENV_CPU, SMR, Receiver, Sender)
    A = PSC._dense_matrix(PSC.asym_ur(G_rs, asym_green_map(ENV_CPU, SMR, Receiver), SMR), false)
    F = eigen(Hermitian(A))
    idx = sortperm(F.values, rev=true)
    Γr, Vr = F.values[idx], F.vectors[:, idx]
    kept = findall(γ -> abs(γ) >= DEFAULT_GAMMA_RTOL * Γr[1], Γr)
    Γrs_r = svdvals(PSC._dense_matrix(G_rs, false))
    res = quiet() do
        bounds_from_spectrum(ENV_CPU, SMR, Γr, Vr[:, kept], Γrs_r; kept=kept, k_uu=N_U, G₀_uu=G_UU, τs=TAUS)
    end
    println("    kept $(length(kept)) of $(length(Γr)) Γ, $(res.num_pos) positive; m_aug = $(res.m_aug)")
    front = (; PSC.bounds_front_end(Vr[:, kept], Vr[:, kept], Γr[kept], LinearMap(G_DENSE), N_S)..., ζ=ζ)
    properties("real 2×2×2", res, front)
end

println("\n=== checkpoint and resume")
let ck = joinpath(ROOT, "ckpt")
    whole = run_synthetic(k_uu=8, checkpoint=ck)
    check("a checkpointed run writes its front and progress",
          isfile("$(ck).front.jld") && isfile("$(ck).progress.jld") && all(isfinite, whole.bounds_dual_basis))
    # Pretend the job was killed after three indices.
    PSC.save_atomic("$(ck).progress.jld", "channels" => collect(1:K), "k_done" => 3,
                    "bounds_dual_basis" => [whole.bounds_dual_basis[1:3]; fill(Inf, K - 3)],
                    "bounds_dual_by_tau" => whole.bounds_dual_by_tau, "opt_taus" => whole.opt_taus, "opt_mu_p" => whole.opt_mu_p)
    resumed = run_synthetic(k_uu=8, checkpoint=ck)
    check("a resumed run reproduces the run it resumed", resumed.bounds_dual_basis == whole.bounds_dual_basis &&
          resumed.m_aug == whole.m_aug, string(resumed.bounds_dual_basis ./ whole.bounds_dual_basis))
    stale = run_synthetic(k_uu=8, checkpoint=ck, τs=range(0.0, 1.0, length=6))
    check("a checkpoint for other inputs is not reused", length(stale.tau_grid) == 6 && isfile("$(ck).front.jld"))
end

println("\n=== τ search")
let grid = run_synthetic(k_uu=N_U, τ_refine_tol=nothing)
    worst = maximum(FULL.bounds_dual_basis ./ grid.bounds_dual_basis)
    check("the warm-started τ search stays within 1% of the grid minimum", worst <= 1.01, @sprintf("worst ratio %.4f", worst))
    sparse = run_synthetic(k_uu=N_U, channels=[1, 3, K])
    check("a channel between samples takes the sample below it",
          sparse.bounds_dual_basis[2] == sparse.bounds_dual_basis[1] && sparse.bounds_dual_basis[4] == sparse.bounds_dual_basis[3] &&
          sparse.dual_channels == [1, 3, K])
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
