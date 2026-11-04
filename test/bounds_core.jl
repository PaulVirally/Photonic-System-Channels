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
function dense_operator(P, n, τ, μ_p)
    A = V * Diagonal(Γ) * V'
    An = V[:, n:end] * Diagonal(Γ[n:end]) * V[:, n:end]'
    Gc = P * (P' * G_DENSE * P) * P'
    one_τ = Diagonal(vcat(ones(N_S), fill(1 - τ, N_U - N_S)))
    return Hermitian(one_τ + ζ .* (τ .* A .+ Gc .- An ./ μ_p))
end
dense_mu(P, n, τ, μ_p) = μ_p * eigmax(Hermitian(inv(dense_operator(P, n, τ, μ_p))[1:N_S, 1:N_S]))
front_for(P) = (; PSC.bounds_front_end(V, P, Γ, LinearMap(G_DENSE), N_S)..., ζ=ζ)

const P_FULL = augmented_basis(V, randn(ComplexF64, N_U, N_U - length(Γ))).basis
const P_PART = augmented_basis(V, randn(ComplexF64, N_U, 8)).basis # 20 < N_s columns

println("\n=== low-rank dual vs dense 𝟙(τ) − H")
for (name, P) in (("full basis", P_FULL), ("partial basis", P_PART))
    front = front_for(P)
    check("$name: sender_spanned = $(size(P, 2) >= N_S)", front.sender_spanned == (size(P, 2) >= N_S))
    worst = 0.0
    for τ in (0.0, 0.3, 0.7, 0.95), n in (1, 3, K), t in (0.01, 0.5, 0.99)
        μₙ, μ_lo, μ_hi = PSC.dual_curve(front, tau_prep(front, τ), n)
        μ_p = μ_lo * (min(μ_hi, 100μ_lo) / μ_lo)^t # across the feasible interval, in log
        worst = max(worst, abs(μₙ(μ_p) - dense_mu(P, n, τ, μ_p)) / dense_mu(P, n, τ, μ_p))
    end
    check("$name: μₙ(μₚ, τ) matches the dense build", worst < 1e-10, @sprintf("max rel diff %.2e", worst))

    println("--- feasibility edge ($name)")
    edge_ok = guard_ok = true
    upper_seen = false
    for τ in (0.0, 0.5, 0.95), n in (1, K)
        μₙ, μ_lo, μ_hi = PSC.dual_curve(front, tau_prep(front, τ), n)
        edge_ok &= isposdef(dense_operator(P, n, τ, μ_lo * (1 + 1e-6))) &&
                   !isposdef(dense_operator(P, n, τ, μ_lo * (1 - 1e-6)))
        guard_ok &= μₙ(μ_lo * (1 - 1e-6)) == Inf && (!isfinite(μ_hi) || μₙ(μ_hi * (1 + 1e-6)) == Inf)
        if isfinite(μ_hi)
            upper_seen = true
            edge_ok &= isposdef(dense_operator(P, n, τ, μ_hi * (1 - 1e-6))) &&
                       !isposdef(dense_operator(P, n, τ, μ_hi * (1 + 1e-6)))
        end
    end
    check("$name: 𝟙(τ) − H ≻ 0 exactly on (μ_lo, μ_hi)", edge_ok, upper_seen ? "both edges seen" : "no finite μ_hi")
    check("$name: μₙ is Inf outside (μ_lo, μ_hi)", guard_ok)
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
const TAUS = range(0.0, 1.0, length=11)

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
    PSC.save_atomic("$(ck).progress.jld", "n_done" => 3, "bounds_dual_basis" => [whole.bounds_dual_basis[1:3]; zeros(K - 3)],
                    "bounds_dual_by_tau" => whole.bounds_dual_by_tau, "opt_taus" => whole.opt_taus, "opt_mu_p" => whole.opt_mu_p)
    resumed = run_synthetic(k_uu=8, checkpoint=ck)
    check("a resumed run reproduces the run it resumed", resumed.bounds_dual_basis == whole.bounds_dual_basis &&
          resumed.m_aug == whole.m_aug, string(resumed.bounds_dual_basis ./ whole.bounds_dual_basis))
    stale = run_synthetic(k_uu=8, checkpoint=ck, τs=range(0.0, 1.0, length=6))
    check("a checkpoint for other inputs is not reused", length(stale.tau_grid) == 6 && isfile("$(ck).front.jld"))
end

println("\n=== τ search")
let grid = run_synthetic(k_uu=N_U, τ_refine_tol=nothing)
    check("refinement never loosens the grid minimum",
          all(FULL.bounds_dual_basis .<= grid.bounds_dual_basis .* (1 + 1e-12)))
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
