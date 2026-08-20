#!/usr/bin/env julia
"""
    bench/domain_monotonicity.jl

Why a larger design domain's bound can come out *below* a smaller one's, which it
must never do. Reads bounds output JLDs -- no GPU, no scratch, no RSVD outputs --
and runs in seconds against a project directory you already have locally.

    julia --project=. bench/domain_monotonicity.jl \\
        --small "data analysis/data/narval_Ge1000_arxivV3_0p25x0p25x0p25_3072comps_50oversamples_q3_32scale" \\
        --large "data analysis/data/narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale"

`--small` is the reference the larger domain has to dominate (the exact λ/4 sweep);
`--large` is the sweep under suspicion. Either may be given alone: with only
`--large` you get that sweep's tail table and nothing to compare it against.

Options: `--min-sep 1.0` (skip the near field), `--top-n 9,38,79,150,300`,
`--key true_bounds` (or `bounds_dual_basis` to look past the analytical bounds).

# The two hypotheses this separates

A bounds output stores one number per channel: `true_bounds[n]`, the tightest of
the two analytical bounds and the dual bound for channel `n`. The trace plotted
against separation is their sum. When the large-domain trace falls below the
small-domain one, there are two candidate explanations and they call for
completely different fixes.

  1. **Truncation of the sum.** The large sweep's `--gamma-rtol` cut leaves it
     only a few dozen channels to add up, while the exact small sweep sums many
     hundreds. If the small sweep's trace is carried by channels beyond the large
     sweep's cut-off, the two traces are simply sums over different numbers of
     terms and the crossing is an artefact of where the sum stops. The fix would
     be an analytical tail correction, or a larger rank.
  2. **Restriction of the basis.** The dual is solved over the span of the
     surviving `Asym(G⁰ᵤᵣ)` eigenvectors. Fewer surviving directions is a
     *smaller feasible set*, so the number that comes out is smaller -- and a
     number that shrinks when you restrict the design space is not an upper bound
     on the unrestricted problem at all. This would show up in the *leading*
     channels, not in the tail, and no amount of extra rank fixes it while the
     `gamma_rtol` cut is what collapses the basis.

The `top-N` columns settle (1): they are the fraction of the full trace that the
top `N` channels already account for. The `lead` columns settle (2): they compare
channel 1 against channel 1, where truncation cannot be the explanation.

# Reading the output

  * `topN` near 1.0 at every far separation for the small sweep means its trace
    is *not* in the tail, and hypothesis (1) is dead: adding the large sweep's
    missing channels back would change nothing.
  * `lead(large)/lead(small) < 1` at a separation where the large domain contains
    the small one is hypothesis (2), caught in the act. Channel 1 of a bigger
    domain cannot legitimately be bounded lower than channel 1 of a smaller one.
  * `Γ[m]/Γ[1]` is how far down its own spectrum each sweep got before the cut
    bit. A sweep that stops nine orders of magnitude above the other one is
    solving over a far poorer basis, which is the mechanism behind (2).
  * `dual/ana/oana` is the `which_bounds` histogram: how many channels took the
    dual, the new analytical and the old analytical bound. A far-field point
    where the large sweep takes the dual for every channel while the small sweep
    takes an analytical bound for almost all of them is the same story again --
    the dual is winning because it is spuriously small, not because it is tight.
  * `Σ new_ana` is the analytical (Eq-18-style) trace, computed from `Γrs` alone
    and so independent of the basis. If it is domain-monotone where the plotted
    trace is not, that is the far-field fallback.
"""

using JLD2
using Printf

function parse_cli(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("expected an option starting with --, got '$(argv[i])'")
        key = argv[i][3:end]
        if i + 1 > length(argv) || startswith(argv[i+1], "--")
            opts[key] = "true"; i += 1
        else
            opts[key] = argv[i+1]; i += 2
        end
    end
    return opts
end

"""
    scan(dir, key) -> Dict{Rational{Int},NamedTuple}

Every complete bounds output in `dir`, keyed by the separation in its file name
(`..__<n>ss<d>__RS.jld`). A file missing `key` is a job that did not finish and is
skipped silently; that is the normal state of a partly-run sweep.
"""
function scan(dir::AbstractString, key::AbstractString)
    out = Dict{Rational{Int},NamedTuple}()
    for f in readdir(dir)
        endswith(f, ".jld") || continue
        mm = match(r"__(-?\d+)ss(\d+)__", f)
        mm === nothing && continue
        sep = parse(Int, mm.captures[1]) // parse(Int, mm.captures[2])
        try
            jldopen(joinpath(dir, f), "r") do io
                haskey(io, key) || return
                v = Array{Float64}(io[key])
                isempty(v) && return
                Γ = haskey(io, "Γ") ? Array{Float64}(io["Γ"]) : Float64[]
                wb = haskey(io, "which_bounds") ? Array{Int}(io["which_bounds"]) : Int[]
                na = haskey(io, "new_analytical_bounds") ?
                     Array{Float64}(io["new_analytical_bounds"]) : Float64[]
                out[sep] = (v=sort(v; rev=true), lead=maximum(v), m=length(v),
                            Γ1=isempty(Γ) ? NaN : Γ[1],
                            Γm=isempty(Γ) ? NaN : Γ[min(length(v), length(Γ))],
                            dual=count(==(3), wb), ana=count(==(2), wb),
                            oana=count(==(1), wb),
                            ana_sum=sum(filter(isfinite, na); init=0.0))
            end
        catch err
            println(stderr, "  skip (", sprint(showerror, err), "): ", f)
        end
    end
    return out
end

"Fraction of the full trace already accounted for by the top `n` channels."
topfrac(v::Vector{Float64}, n::Int) = (t = sum(v); t == 0 ? NaN : sum(view(v, 1:min(n, length(v)))) / t)

function report(label, d, tops, minsep)
    println("\n### $label  ($(length(d)) complete output(s))")
    @printf("%-13s %-6s %-11s %s  %-9s %-9s %s\n", "sep", "m", "trace",
            join([@sprintf("top%-6d", n) for n in tops], " "), "Γ[m]/Γ[1]", "Σ new_ana", "dual/ana/oana")
    for sep in sort(collect(keys(d)); by=Float64)
        Float64(sep) >= minsep || continue
        r = d[sep]
        @printf("%-13s %-6d %-11.5g %s  %-9.2e %-9.5g %d/%d/%d\n", string(sep), r.m, sum(r.v),
                join([@sprintf("%-9.4f", topfrac(r.v, n)) for n in tops], " "),
                r.Γm / r.Γ1, r.ana_sum, r.dual, r.ana, r.oana)
    end
end

function compare(small, large, minsep)
    common = sort(collect(intersect(keys(small), keys(large))); by=Float64)
    isempty(common) && (println("\nNo separations in common; nothing to compare."); return)
    println("\n### Domain monotonicity: large / small (must be >= 1 everywhere)")
    @printf("%-13s %-11s %-11s %-8s | %-11s %-11s %-8s | %s\n", "sep",
            "trace small", "trace large", "ratio", "lead small", "lead large", "ratio", "verdict")
    violations = 0
    for sep in common
        Float64(sep) >= minsep || continue
        s, l = small[sep], large[sep]
        ts, tl = sum(s.v), sum(l.v)
        bad = tl < ts
        bad && (violations += 1)
        @printf("%-13s %-11.5g %-11.5g %-8.3f | %-11.5g %-11.5g %-8.3f | %s\n",
                string(sep), ts, tl, tl / ts, s.lead, l.lead, l.lead / s.lead,
                bad ? (l.lead < s.lead ? "VIOLATION (leading channel)" : "VIOLATION (tail only)") : "ok")
    end
    println("\n$violations violating separation(s).")
    println("""
    A violation whose leading-channel ratio is also below 1 is not a truncated
    sum: channel 1 exists in both sweeps, so the only thing that can have made the
    larger domain's channel 1 smaller is the basis the dual was solved over. Check
    the Γ[m]/Γ[1] columns above -- the sweep that stopped higher up its own
    spectrum is the one whose dual is restricted, and its number is not an upper
    bound on the unrestricted problem.

    A violation with a leading-channel ratio at or above 1 *is* a truncated sum,
    and the topN columns say how much of the small sweep's trace sits beyond the
    large sweep's channel count.""")
end

function main(argv::Vector{String})
    opts = parse_cli(argv)
    key = get(opts, "key", "true_bounds")
    minsep = parse(Float64, get(opts, "min-sep", "1.0"))
    tops = parse.(Int, split(get(opts, "top-n", "9,38,79,150,300"), ","))
    small_dir = get(opts, "small", "")
    large_dir = get(opts, "large", "")
    (isempty(small_dir) && isempty(large_dir)) &&
        error("give --small <project dir> and/or --large <project dir>")

    println("key = $key, separations >= $minsep lambda")
    small = isempty(small_dir) ? nothing : scan(small_dir, key)
    large = isempty(large_dir) ? nothing : scan(large_dir, key)
    small === nothing || report("small domain: $(basename(rstrip(small_dir, '/')))", small, tops, minsep)
    large === nothing || report("large domain: $(basename(rstrip(large_dir, '/')))", large, tops, minsep)
    (small === nothing || large === nothing) || compare(small, large, minsep)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
