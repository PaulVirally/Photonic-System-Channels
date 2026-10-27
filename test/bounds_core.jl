## The bounds stage on a tiny CPU system: the factored pencil against a dense
## generalized eigensolve, the reverse Gram-Schmidt probes, the augmentation, and
## the τ search's window and cache against a plain full sweep.
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
const K = 8 # num_pos
const ζ = abs(susceptibility(SMR))^2 / imag(susceptibility(SMR))

# A synthetic spectrum at production magnitude: the root find brackets on κ̃ = ζΓ.
Random.seed!(0xBEEF)
const Γ = collect(exp10.(range(log10(1 / ζ), log10(1e-4 / ζ), length=K)))
const V = Matrix(qr(randn(ComplexF64, N_U, K)).Q)[:, 1:K]
const Γrs = collect(exp10.(range(log10(1 / ζ), log10(1e-6 / ζ), length=K)))
const G_UU = load_asym_green_function(ENV_CPU, SMR, Design)
println("N_u = $(N_U), num_pos = $(K)")

println("\n=== factored pencil vs dense generalized eigensolve")
let m = 6, p = 3
    X = randn(ComplexF64, m, m)
    C = X * X' + I
    Vb = randn(ComplexF64, m, p)
    c = rand(p)
    B = Vb * Diagonal(c .^ 2) * Vb'
    pencil = psd_pencil_whitener(C)
    Λ, Vf = factored_pencil_eigen(FactoredB(Vb, c), pencil.whitener, pencil.nullspace)
    ref = sort(real(eigvals(Hermitian(B), Hermitian(C))))
    d = maximum(abs.(sort(Λ) .- ref)) / maximum(abs.(ref))
    check("eigenvalues match eigvals(B, C)", d < 1e-10, @sprintf("max rel diff %.2e", d))
    check("eigenvectors are C-orthonormal", opnorm(Vf' * C * Vf - I) < 1e-10)
end

println("\n=== reverse Gram-Schmidt probes")
let ss = similar(V)
    reverse_gram_schmidt!(ss, V, N_S, K)
    check("probes are orthonormal", opnorm(ss' * ss - I) < 1e-12)
    check("probes vanish on the receiver", iszero(ss[N_S+1:end, :]))
    # Column i lies in Πₛ·span(gᵢ, …, g_m): its residual against that span is zero.
    worst = maximum(1:K) do i
        P = V[1:N_S, i:K]
        r = ss[1:N_S, i] - P * (P \ ss[1:N_S, i])
        norm(r)
    end
    check("probe i lies in Πₛ·span(gᵢ…g_m)", worst < 1e-10, @sprintf("worst residual %.2e", worst))
end

println("\n=== augmentation")
let U = randn(ComplexF64, N_U, 5)
    aug = augmented_basis(V, hcat(U, V[:, 1])) # the last column is already in span(g)
    check("the g block is left untouched", aug.basis[:, 1:K] == V)
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

quiet(f) = with_logger(f, SimpleLogger(stderr, Logging.Warn))
const TAUS = range(0.0, 1.0, length=11)
run_bounds(; kwargs...) = quiet() do
    bounds_from_spectrum(ENV_CPU, SMR, Γ, V, Γrs; num_pos=K, G₀_uu=G_UU, τs=TAUS, kwargs...)
end

# A larger basis can only raise the projected dual (the primal is a sup over a
# larger subspace), and the whole universe is the full-space dual.
const PLAIN = run_bounds(k_uu=0)
const HALF = run_bounds(k_uu=(N_U - K) ÷ 2)
const FULL = run_bounds(k_uu=N_U)
check("k_uu = 0 uses the g basis alone", PLAIN.m_aug == K)
check("k_uu ≥ N_u - m spans the universe", FULL.m_aug == N_U, "m_aug = $(FULL.m_aug)")
check("the dual grows with the basis",
      all(PLAIN.bounds_dual_basis .<= HALF.bounds_dual_basis .* (1 + 1e-8)) &&
      all(HALF.bounds_dual_basis .<= FULL.bounds_dual_basis .* (1 + 1e-8)),
      string(round.(PLAIN.bounds_dual_basis ./ FULL.bounds_dual_basis; sigdigits=3)))
check("true_bounds are positive and passive", all(0 .< FULL.true_bounds .<= 1))

println("\n=== τ search: window and cache against a full uncached sweep")
const REF = run_bounds(k_uu=0, tau_window=0, pencil_cache_max=0)
same(res) = all(isapprox.(res.bounds_dual_basis, REF.bounds_dual_basis; rtol=1e-10)) &&
            all(isapprox.(res.opt_taus, REF.opt_taus; rtol=1e-10, atol=1e-15))
for (name, kwargs) in (("±1 window", (; tau_window=1, pencil_cache_max=0)),
                       ("±2 window", (; tau_window=2, pencil_cache_max=0)),
                       ("cache only", (; tau_window=0)),
                       ("cache of one", (; tau_window=0, pencil_cache_max=1)),
                       ("defaults", (;)))
    res = run_bounds(; k_uu=0, kwargs...)
    check("$(name) reproduces the full sweep", same(res),
          @sprintf("worst rel diff %.2e", maximum(abs.(res.bounds_dual_basis .- REF.bounds_dual_basis) ./ REF.bounds_dual_basis)))
end
let grid = run_bounds(k_uu=0, τ_refine_tol=nothing)
    check("refinement never loosens the grid minimum",
          all(REF.bounds_dual_basis .<= grid.bounds_dual_basis .* (1 + 1e-12)))
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
