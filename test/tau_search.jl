## The τ search inside bounds_from_spectrum: the windowed grid sweep, its
## edge fallback, and the refinement pencil cache.
##
## Run from the repo root:
##     julia --project=. test/tau_search.jl
##
## No GPU needed. One tiny (2,2,2)+(2,2,2) SR Green function is generated into a
## mktempdir (about a minute) and reused by every run below; the spectrum is
## synthetic. Exits nonzero on failure.
##
## The reference run is `tau_window=0, pencil_cache_max=0`: the whole grid at every
## index, and no memoized pencils. Each τ bounds σₙ on its own, so a narrowed sweep
## can only lose tightness, and every setting checked below has to reproduce the
## reference bounds exactly.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using LinearAlgebra
using Printf
using Random
using Logging

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-58s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

function thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

const ROOT = mktempdir(; cleanup=true)
for d in ("preload", "project", "scratch")
    mkpath(joinpath(ROOT, d))
end
println("workspace: ", ROOT)

const ENV_CPU = ComputeEnvironment(joinpath(ROOT, "preload"), joinpath(ROOT, "project"),
                                   joinpath(ROOT, "scratch"), GPUChoice(false, -1))
const SMR = SMRSystem((2, 2, 2), (1//32, 0//1, 0//1), (2, 2, 2),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im)
const N_U = 3 * (prod(sender(SMR).cel) + prod(receiver(SMR).cel))
const K = 8 # num_pos, and the number of outer indices

# The scale matters: the root find brackets on κ̃ = ζΓ, so a spectrum of the wrong
# magnitude exercises a different branch than production does.
const ζ = abs(PSC.susceptibility(SMR))^2 / imag(PSC.susceptibility(SMR))
Random.seed!(0xBEEF)
const Γ = collect(exp10.(range(log10(1 / ζ), log10(1e-4 / ζ), length=K)))
const V = Matrix(qr(randn(ComplexF64, N_U, K)).Q)[:, 1:K]
const Γrs = collect(exp10.(range(log10(1 / ζ), log10(1e-6 / ζ), length=K)))

println("N_u = $(N_U), num_pos = $(K)")
const G_UU = load_green_function(ENV_CPU, SMR, [Sender, Receiver], [Sender, Receiver])

# Eleven points so that a ±1 and a ±2 window are both narrower than the grid, and
# so the transition below is a jump of six grid points rather than of one.
const TAUS = range(0.0, 1.0, length=11)

# The per-index @info lines are the loop's own noise; only warnings matter here.
run_bounds(; kwargs...) = with_logger(SimpleLogger(stderr, Logging.Warn)) do
    bounds_from_spectrum(ENV_CPU, SMR, Γ, V, Γrs; num_pos=K, G₀_uu=G_UU, τs=TAUS,
                         kwargs...)
end

const REF = run_bounds(tau_window=0, pencil_cache_max=0)
const REF_GRID_ARGMIN = [argmin(REF.bounds_dual_by_tau[n, :]) for n in 1:K]
println("reference bounds       = ", REF.bounds_dual_basis)
println("reference opt_taus     = ", REF.opt_taus)
println("reference grid argmins = ", REF_GRID_ARGMIN)
println("reference refinement whitenings = ", REF.tau_search.pencil_cache_misses)

# The grid minimiser sits at an interior τ for n = 1 and then plateaus at τ = 0,
# which is the shape production output has: endpoint plateaus with one abrupt
# transition. The transition is the case a window cannot see, and the one the edge
# fallback has to recover.
const JUMP = maximum(abs.(diff(REF_GRID_ARGMIN)))
check("the fixture has a plateau after an interior minimum",
      all(==(1), REF_GRID_ARGMIN[2:end]) && REF_GRID_ARGMIN[1] > 3,
      "argmins $(REF_GRID_ARGMIN)")
check("its transition is wider than any window tested here", JUMP > 2,
      "largest jump in grid argmin = $(JUMP)")

# Agreement helpers. The windowed run leaves NaN where it skipped a grid point, and
# that is fine: bounds_dual_by_tau is a diagnostic table, not an input to anything.
same_bounds(res) = all(isapprox(res.bounds_dual_basis[n], REF.bounds_dual_basis[n];
                               rtol=1e-10) for n in 1:K)
worst_rel(res) = maximum(abs(res.bounds_dual_basis[n] - REF.bounds_dual_basis[n]) /
                         REF.bounds_dual_basis[n] for n in 1:K)
same_taus(res) = all(isapprox(res.opt_taus[n], REF.opt_taus[n]; rtol=1e-10,
                              atol=1e-15) for n in 1:K)
table_consistent(res) = all(isnan(res.bounds_dual_by_tau[n, i]) ||
                            res.bounds_dual_by_tau[n, i] == REF.bounds_dual_by_tau[n, i]
                            for n in 1:K, i in 1:length(TAUS))

# --- 1: windowed vs full

println("\n=== windowed sweep vs full sweep")

const WIN = run_bounds(tau_window=2, pencil_cache_max=0)
check("±2 window reproduces the full sweep's bounds", same_bounds(WIN),
      @sprintf("worst relative difference %.3e", worst_rel(WIN)))
check("±2 window reproduces the full sweep's opt_taus", same_taus(WIN),
      string(WIN.opt_taus))
check("the swept grid entries match, the skipped ones are NaN", table_consistent(WIN))
check("some grid entries were left NaN", any(isnan, WIN.bounds_dual_by_tau),
      "$(count(isnan, WIN.bounds_dual_by_tau)) of $(K * length(TAUS)) skipped")

const DEF = run_bounds()
check("the defaults reproduce the full sweep's bounds", same_bounds(DEF),
      @sprintf("worst relative difference %.3e", worst_rel(DEF)))
check("the defaults reproduce the full sweep's opt_taus", same_taus(DEF),
      string(DEF.opt_taus))

# --- 2: the edge fallback

println("\n=== edge fallback")

# n = 1 sweeps the whole grid, so n = 2 inherits an argmin six points away from
# its own. Its windowed minimum lands on the window's low edge, which is not the
# end of the grid, so the sweep has to be redone in full — otherwise n = 2's bound
# would be the one at τ = 0.4 rather than at τ = 0.
check("the transition triggers a fallback at ±2", WIN.tau_search.grid_fallbacks >= 1,
      "$(WIN.tau_search.grid_fallbacks) fallback(s)")
check("n = 2's recovered bound is the full-grid one, not the window's",
      isapprox(WIN.bounds_dual_basis[2], REF.bounds_dual_basis[2]; rtol=1e-10) &&
      WIN.bounds_dual_basis[2] < REF.bounds_dual_by_tau[2, REF_GRID_ARGMIN[1] - 2],
      @sprintf("%.6f vs window-edge %.6f", WIN.bounds_dual_basis[2],
               REF.bounds_dual_by_tau[2, REF_GRID_ARGMIN[1] - 2]))

const WIN1 = run_bounds(tau_window=1, pencil_cache_max=0)
check("±1 window reproduces the full sweep's bounds", same_bounds(WIN1),
      @sprintf("worst relative difference %.3e", worst_rel(WIN1)))
check("±1 window reproduces the full sweep's opt_taus", same_taus(WIN1),
      string(WIN1.opt_taus))
check("±1 falls back at least as often as ±2",
      WIN1.tau_search.grid_fallbacks >= WIN.tau_search.grid_fallbacks,
      "$(WIN1.tau_search.grid_fallbacks) vs $(WIN.tau_search.grid_fallbacks)")

# A window of the grid's own width can never clip anything, so it must never fall
# back and must fill the whole table.
const WIDE = run_bounds(tau_window=length(TAUS), pencil_cache_max=0)
check("a window as wide as the grid never falls back",
      WIDE.tau_search.grid_fallbacks == 0 && same_bounds(WIDE) && same_taus(WIDE) &&
      !any(isnan, WIDE.bounds_dual_by_tau))

# Sparse outer_indices have no neighbour to inherit from: verify_bounds evaluates a
# handful of spot indices that are decades apart, and a window around the previous
# one would be evidence about a different Bₙ.
println("\n=== non-contiguous outer_indices")
const SPOTS = [1, 4, 5, K]
const SPARSE_REF = run_bounds(tau_window=0, pencil_cache_max=0, outer_indices=SPOTS)
const SPARSE_WIN = run_bounds(outer_indices=SPOTS)
check("a sparse index set is unaffected by the window",
      all(isapprox(SPARSE_WIN.bounds_dual_basis[n], SPARSE_REF.bounds_dual_basis[n];
                   rtol=1e-10) for n in SPOTS) &&
      all(isapprox(SPARSE_WIN.opt_taus[n], SPARSE_REF.opt_taus[n]; rtol=1e-10,
                   atol=1e-15) for n in SPOTS),
      "indices $(SPOTS)")
check("only the consecutive pair in it can be windowed",
      !any(isnan, SPARSE_WIN.bounds_dual_by_tau[[1, 4, K], :]),
      "n = 5 follows n = 4, the rest sweep in full")

# --- 3: the refinement pencil cache

println("\n=== refinement pencil cache")

const CACHED = run_bounds(tau_window=0, pencil_cache_max=16)
check("the cache does not move the bounds", same_bounds(CACHED),
      @sprintf("worst relative difference %.3e", worst_rel(CACHED)))
check("the cache does not move opt_taus", same_taus(CACHED), string(CACHED.opt_taus))
check("the full table is still filled", !any(isnan, CACHED.bounds_dual_by_tau))
check("it hits on the plateau", CACHED.tau_search.pencil_cache_hits > 0,
      "$(CACHED.tau_search.pencil_cache_hits) hit(s), " *
      "$(CACHED.tau_search.pencil_cache_misses) miss(es) vs " *
      "$(REF.tau_search.pencil_cache_misses) whitenings uncached")
# The plateau indices open the same bracket and probe the same τ, so every index
# after the first one on the plateau is served entirely from the cache.
check("it removes nearly every refinement whitening",
      CACHED.tau_search.pencil_cache_misses <= REF.tau_search.pencil_cache_misses ÷ 2,
      "$(CACHED.tau_search.pencil_cache_misses) whitenings for $(K) indices, " *
      "down from $(REF.tau_search.pencil_cache_misses)")

const BOTH = run_bounds(tau_window=2, pencil_cache_max=16)
check("window and cache together still reproduce the full sweep",
      same_bounds(BOTH) && same_taus(BOTH),
      @sprintf("worst relative difference %.3e, %d whitenings, %d fallback(s)",
               worst_rel(BOTH), BOTH.tau_search.pencil_cache_misses,
               BOTH.tau_search.grid_fallbacks))

# One entry cannot hold anything useful: an index's own probe sequence visits five
# distinct τ, so whatever the next index asks for first has already been evicted.
# The check is that eviction leaves the results and the accounting alone. Every call
# is still answered, and hits plus misses adds up to the uncached whitening count
# either way.
const TINY_CACHE = run_bounds(tau_window=0, pencil_cache_max=1)
check("eviction pressure changes neither the bounds nor the accounting",
      same_bounds(TINY_CACHE) && same_taus(TINY_CACHE) &&
      TINY_CACHE.tau_search.pencil_cache_hits +
      TINY_CACHE.tau_search.pencil_cache_misses == REF.tau_search.pencil_cache_misses,
      "$(TINY_CACHE.tau_search.pencil_cache_hits) hit(s), " *
      "$(TINY_CACHE.tau_search.pencil_cache_misses) miss(es)")

check("a negative pencil_cache_max is rejected",
      thrown(() -> run_bounds(pencil_cache_max=-1)) isa ArgumentError)

# With refinement off nothing asks for an off-grid pencil, so the cache is never
# consulted at all.
const NO_REFINE = run_bounds(τ_refine_tol=nothing)
check("with refinement disabled the cache is never touched",
      NO_REFINE.tau_search.pencil_cache_hits == 0 &&
      NO_REFINE.tau_search.pencil_cache_misses == 0,
      string(NO_REFINE.tau_search))
check("and the reported τ are grid points",
      all(τ -> any(isapprox(τ), TAUS), NO_REFINE.opt_taus), string(NO_REFINE.opt_taus))

println()
println("whitenings per index: uncached $(REF.tau_search.pencil_cache_misses / K), " *
        "cached $(CACHED.tau_search.pencil_cache_misses / K), " *
        "cached + windowed $(BOTH.tau_search.pencil_cache_misses / K)")

bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
