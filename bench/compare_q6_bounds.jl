#!/usr/bin/env julia
"""
    bench/compare_q6_bounds.jl

Does dropping the production sweeps from `--power-iterations 14` to `6` change
the *bounds*, not just the RSVD spectrum? Reads a `bounds_dual_basis` (the
per-channel bound on σₙ(Pᵣₛ), see `src/bounds.jl`) from a q=6 validation run and
from the production q=14 sweep, at the separations
`jobs/launch_narval_q6_validation.sh` was sized for, and reports whether the two
agree well enough to retire q=14.

    julia --project=. bench/compare_q6_bounds.jl \\
        --q6-project /home/pvirally/projects/def-smolesky/pvirally/Photonic-System-Channels/projects/narval_Ge1000_q6check \\
        --q14-project /home/pvirally/projects/def-smolesky/pvirally/Photonic-System-Channels/projects/narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale \\
        --seps 1//4,9//32,5//16

No GPU, no scratch: this only reads the two projects' bounds JLDs, in the style
of `bench/domain_monotonicity.jl`.

# What is compared, and why

`bounds_dual_basis[n]` is the dual bound on σₙ(Pᵣₛ) for channel `n`, one entry
per surviving `Asym(G⁰ᵤᵣ)` direction (`num_pos`/"kept"). A different
`--power-iterations` changes which directions the RSVD recovers and how well it
recovers them, so the three separations this script is pointed at were chosen
(see the launcher's header) to hold the rest of the pipeline fixed and isolate
that one effect: mid-field points where the kept g-basis is large, so the dual
is not also being reshaped by the `Asym(G⁰ᵤᵤ)` augmentation
(`--k-uu`/`--augment-threshold`) at the same time. Two of the three launcher
picks (9//32, 5//16) augment anyway (there was no clean augmentation-free
second and third point in the 1 λ sweep's kept-count curve, see the launcher's
header) so this script always reports each point's augmentation state
alongside its numbers instead of assuming it off.

Four things are reported per separation:

  * **Per-channel relative deviation**, over channels whose q=14 (reference)
    `bounds_dual_basis` magnitude is above `--min-abs` (default 1e-5; below that
    a bound is numerically noise, and a relative deviation is meaningless). Max
    and median, over the channels common to both runs (`1:min(m6, m14)`).
  * **Trace deviation on the σₙ scale**: `sum(bounds_dual_basis)`, q6 against
    q14, both as computed over each run's own length (this is the same
    trace `bench/domain_monotonicity.jl` reports).
  * **Kept-count difference**: `length(bounds_dual_basis)` (`num_pos`) for q6
    minus q14. A different power-iteration count can recover a different
    number of positive `Asym(G⁰ᵤᵣ)` eigenvalues above `--gamma-rtol`, which
    would shift comparisons even before any per-channel deviation is looked at.
  * **`opt_taus` agreement**: the τ each run's outer loop settled on for a
    channel is a discrete grid pick (`src/bounds.jl`'s `tau_grid`), so "agree"
    means landing on the same grid point, not merely a small numeric gap. The
    fraction of common channels that do, plus the largest `|Δτ|` among the ones
    that do not.

# Verdict

The reporting-floor gates: the trace agrees if it is within 1% relative *or*
1e-5 absolute, whichever is the looser (i.e. it passes if either holds), and the
gated per-channel deviations agree if the max is within 5%. A separation passes
only if both hold. `q=6 passes everywhere this script was pointed at` is the
condition for regenerating the production launchers at

# Per-separation project overrides

`--q6-project-override <sep>=<dir>[,<sep>=<dir>,...]` points one separation's q6
side at a different project directory than `--q6-project`, without touching the
others. This exists because "q=6 vs q=14" is only a fair comparison when both
sides ran the *same* bounds algorithm: `jobs/launch_narval_q6_tiebreak.sh`
reruns q=6's bounds at 9//32 with the `Asym(G⁰ᵤᵤ)` augmentation forced on (the
original q=6 run there kept m=521, above the production `--augment-threshold
500`, so it ran unaugmented while q=14, which kept m=491, augmented), and
writes that rerun to a separate project directory so it does not clobber the
original. `--q6-project-override 9//32=<that directory>` is how this script is
told to read the like-with-like rerun at 9//32 while still reading `--q6-project`
for 1//4 and 5//16.

# Noise floor: a same-q replicate comparison

`--replicate-a <project> --replicate-b <project> --sep <separation>` runs the
exact same four-way report above between two runs at the *same*
`--power-iterations`, instead of q6 against q14. This measures the comparison's
own noise floor: the in-memory RSVD path's random test matrix is unseeded (see
`src/rsvd.jl`: the `--seed` a run is launched with is recorded in its output
but never reaches the sketch on that path), so two runs of the same q at the
same separation still land on slightly different bases, and channels whose
`Asym(G⁰ᵤᵣ)` eigenvalues are nearly degenerate reorder between them. That
reordering is exactly what produced two of the three original q6-vs-q14 FAILs:
per-channel deviations 12x-34x the flat 5% gate with medians around 5e-5, the
signature of a few edge channels swapping rank rather than a systematic q=6
problem. Passing only `--replicate-a`/`--replicate-b`/`--sep` (no
`--q6-project`/`--q14-project`) runs this comparison on its own and stops there.

# Floor-anchored channel gate

`--noise-scale <x>` (default 2.0), given together with `--replicate-a` and
`--replicate-b`, changes what "the gated per-channel deviations agree" means for
the normal q6-vs-q14 comparison: instead of the flat `--channel-rel-gate`
(default 5%), a separation's max and median gated relative deviations must each
be at most `x` times the *replicate*'s max and median: the noise floor just
measured, scaled up by a safety factor, rather than an arbitrary absolute
number. Both verdicts are printed for every separation, raw (flat gate) and
floor-anchored; when floor mode is active (`--replicate-a`/`--replicate-b`
given), the floor-anchored verdict is the one that decides the final per-run
tally and the closing message, since it is the one that accounts for measurement
noise. `q=6 passes everywhere this script was pointed at` is the condition for
regenerating the production launchers at
`--power-iterations 6`; anything short of that keeps them at 14.
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

parse_rational(s::AbstractString) =
    (p = split(strip(s), "//"); length(p) == 2 ? parse(Int, p[1]) // parse(Int, p[2]) : parse(Int, strip(s)) // 1)

"""
    parse_overrides(s) -> Dict{Rational{Int},String}

`--q6-project-override 9//32=<dir>,5//16=<dir2>` into a separation-keyed lookup.
Empty string (the flag not given) is an empty dict, so callers can `get(overrides,
sep, default_dir)` unconditionally.
"""
function parse_overrides(s::AbstractString)
    overrides = Dict{Rational{Int},String}()
    isempty(strip(s)) && return overrides
    for pair in split(s, ",")
        kv = split(pair, "="; limit=2)
        length(kv) == 2 || error("invalid --q6-project-override entry '$pair', expected <sep>=<dir>")
        overrides[parse_rational(kv[1])] = String(strip(kv[2]))
    end
    return overrides
end

# --------------------------------------------------------------------------- #
# Locating and loading one separation's bounds output
# --------------------------------------------------------------------------- #

"""
    find_bounds_jld(dir, sep) -> Union{Nothing,String}

The bounds JLD in `dir` (a project directory, not scratch) whose file name
encodes separation `sep`, or `nothing` if there is none. File names look like
`32x32x32__32x32x32__1ss4__RS.jld`; only the `__<n>ss<d>__` fragment is parsed,
exactly as `bench/domain_monotonicity.jl` and `bench/size_bounds_jobs.jl` do.
"""
function find_bounds_jld(dir::AbstractString, sep::Rational{Int})
    isdir(dir) || return nothing
    for f in readdir(dir)
        endswith(f, ".jld") || continue
        mm = match(r"__(-?\d+)ss(\d+)__", f)
        mm === nothing && continue
        parse(Int, mm.captures[1]) // parse(Int, mm.captures[2]) == sep || continue
        return joinpath(dir, f)
    end
    return nothing
end

"""
    load_bounds(path) -> NamedTuple

`bounds_dual_basis` and `opt_taus`, plus the augmentation provenance
(`src/bounds.jl`'s `augment/*` group) when present. An unaugmented point
never wrote that group, so its absence just means "not augmented", not "job
incomplete".
"""
function load_bounds(path::AbstractString)
    jldopen(path, "r") do io
        haskey(io, "bounds_dual_basis") || error("no bounds_dual_basis in $path (job incomplete?)")
        dual = Array{Float64}(io["bounds_dual_basis"])
        taus = haskey(io, "opt_taus") ? Array{Float64}(io["opt_taus"]) : fill(NaN, length(dual))
        augmented = haskey(io, "augment/augmented") ? Bool(io["augment/augmented"]) : false
        m_aug = haskey(io, "augment/m_aug") ? Int(io["augment/m_aug"]) : length(dual)
        return (dual=dual, opt_taus=taus, m=length(dual), augmented=augmented, m_aug=m_aug)
    end
end

# --------------------------------------------------------------------------- #
# Per-separation comparison
# --------------------------------------------------------------------------- #

"""
    compare_one(sep, q6, q14; min_abs) -> NamedTuple

Everything the report line and the verdict need for one separation, given both
sides already loaded.
"""
function compare_one(q6, q14; min_abs::Real)
    n_common = min(q6.m, q14.m)
    n_common > 0 || error("no channels in common (q6 m=$(q6.m), q14 m=$(q14.m))")
    v6 = view(q6.dual, 1:n_common)
    v14 = view(q14.dual, 1:n_common)

    gated = findall(n -> isfinite(v14[n]) && abs(v14[n]) > min_abs, 1:n_common)
    rel_devs = isempty(gated) ? Float64[] :
        [abs(v6[n] - v14[n]) / abs(v14[n]) for n in gated]
    max_rel = isempty(rel_devs) ? NaN : maximum(rel_devs)
    median_rel = isempty(rel_devs) ? NaN : sort(rel_devs)[cld(length(rel_devs), 2)]

    trace6, trace14 = sum(q6.dual), sum(q14.dual)
    trace_abs_dev = abs(trace6 - trace14)
    trace_rel_dev = trace14 == 0 ? (trace_abs_dev == 0 ? 0.0 : Inf) : trace_abs_dev / abs(trace14)

    tau_common = [n for n in 1:n_common if isfinite(q6.opt_taus[n]) && isfinite(q14.opt_taus[n])]
    tau_matches = [n for n in tau_common if q6.opt_taus[n] == q14.opt_taus[n]]
    tau_agree_frac = isempty(tau_common) ? NaN : length(tau_matches) / length(tau_common)
    tau_mismatches = setdiff(tau_common, tau_matches)
    tau_max_gap = isempty(tau_mismatches) ? 0.0 :
        maximum(abs(q6.opt_taus[n] - q14.opt_taus[n]) for n in tau_mismatches)

    return (n_gated=length(gated), n_common=n_common, max_rel=max_rel, median_rel=median_rel,
            trace6=trace6, trace14=trace14, trace_abs_dev=trace_abs_dev, trace_rel_dev=trace_rel_dev,
            kept_diff=q6.m - q14.m, tau_agree_frac=tau_agree_frac, tau_max_gap=tau_max_gap,
            tau_n_common=length(tau_common))
end

"Trace agrees if within 1% relative OR 1e-5 absolute (whichever is looser); gated channels agree if their max relative deviation is within 5%."
function verdict(c; trace_rel_gate::Real=0.01, trace_abs_gate::Real=1e-5, channel_rel_gate::Real=0.05)
    trace_ok = c.trace_rel_dev <= trace_rel_gate || c.trace_abs_dev <= trace_abs_gate
    channel_ok = isnan(c.max_rel) || c.max_rel <= channel_rel_gate
    return (pass=trace_ok && channel_ok, trace_ok=trace_ok, channel_ok=channel_ok)
end

"""
    floor_verdict(c, floor; noise_scale, trace_rel_gate, trace_abs_gate, channel_rel_gate) -> NamedTuple

The channel gate re-expressed against a measured noise floor (`floor`, a
`compare_one` result from a same-q `--replicate-a`/`--replicate-b` run) instead
of the flat `channel_rel_gate`: a separation's gated max and median relative
deviations must each be at most `noise_scale` times the floor's max and median.
Falls back to `channel_rel_gate` for whichever of the floor's own max/median is
zero or non-finite (an exact-agreement replicate, or too few gated channels to
have one), so an unusually quiet floor measurement can never demand an
unattainably tight gate. The trace gate is unchanged: the floor measures
per-channel reordering noise, not a bias in the summed trace.
"""
function floor_verdict(c, floor; noise_scale::Real=2.0, trace_rel_gate::Real=0.01,
                       trace_abs_gate::Real=1e-5, channel_rel_gate::Real=0.05)
    trace_ok = c.trace_rel_dev <= trace_rel_gate || c.trace_abs_dev <= trace_abs_gate
    max_gate = (isfinite(floor.max_rel) && floor.max_rel > 0) ? noise_scale * floor.max_rel : channel_rel_gate
    median_gate = (isfinite(floor.median_rel) && floor.median_rel > 0) ? noise_scale * floor.median_rel : channel_rel_gate
    channel_ok = isnan(c.max_rel) || (c.max_rel <= max_gate && c.median_rel <= median_gate)
    return (pass=trace_ok && channel_ok, trace_ok=trace_ok, channel_ok=channel_ok,
            max_gate=max_gate, median_gate=median_gate)
end

"""
    print_pair_stats(c; min_abs, label_a="q6", label_b="q14")

The four-part report shared by the q6-vs-q14 comparison and the same-q
`--replicate-a`/`--replicate-b` comparison: per-channel relative deviation
(gated on `label_b`'s magnitude), trace deviation, kept-count difference, and
`opt_taus` agreement. Printing only: callers compute `c = compare_one(...)`
and any verdict separately, so this is shared without coupling the two modes'
pass/fail logic together.
"""
function print_pair_stats(c; min_abs::Real, label_a::AbstractString="q6", label_b::AbstractString="q14")
    @printf("  per-channel rel dev (%d/%d channels above %.0e abs, %s-gated): max %.3g, median %.3g\n",
            c.n_gated, c.n_common, min_abs, label_b, c.max_rel, c.median_rel)
    @printf("  trace (sigma_n scale): %s %.6g, %s %.6g, abs dev %.3g, rel dev %.3g\n",
            label_a, c.trace6, label_b, c.trace14, c.trace_abs_dev, c.trace_rel_dev)
    println("  kept-count difference ($label_a - $label_b): $(c.kept_diff)")
    if c.tau_n_common > 0
        @printf("  opt_taus agreement: %.1f%% of %d common channels exact; max |Δτ| among mismatches: %.4g\n",
                100 * c.tau_agree_frac, c.tau_n_common, c.tau_max_gap)
    else
        println("  opt_taus agreement: no channels with finite τ in both runs")
    end
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

function main(argv::Vector{String})
    opts = parse_cli(argv)
    q6_dir = get(opts, "q6-project", "")
    q14_dir = get(opts, "q14-project", "")
    rep_a_dir = get(opts, "replicate-a", "")
    rep_b_dir = get(opts, "replicate-b", "")
    rep_sep_str = get(opts, "sep", "")
    overrides = parse_overrides(get(opts, "q6-project-override", ""))

    have_q6 = !isempty(q6_dir)
    have_q14 = !isempty(q14_dir)
    have_q6 == have_q14 || error(have_q6 ?
        "--q6-project given without --q14-project" : "--q14-project given without --q6-project")
    normal_mode = have_q6 && have_q14

    have_rep_a = !isempty(rep_a_dir)
    have_rep_b = !isempty(rep_b_dir)
    (have_rep_a || have_rep_b) && have_rep_a != have_rep_b && error(have_rep_a ?
        "--replicate-a given without --replicate-b" : "--replicate-b given without --replicate-a")
    replicate_mode = have_rep_a && have_rep_b
    replicate_mode && isempty(rep_sep_str) &&
        error("--replicate-a/--replicate-b require --sep <separation>")

    normal_mode || replicate_mode || error(
        "nothing to do: pass --q6-project/--q14-project for the q6-vs-q14 comparison, " *
        "and/or --replicate-a/--replicate-b/--sep for a same-q noise-floor comparison")

    seps = parse_rational.(split(get(opts, "seps", "1//4,9//32,5//16"), ","))
    min_abs = parse(Float64, get(opts, "min-abs", "1e-5"))
    trace_rel_gate = parse(Float64, get(opts, "trace-rel-gate", "0.01"))
    trace_abs_gate = parse(Float64, get(opts, "trace-abs-gate", "1e-5"))
    channel_rel_gate = parse(Float64, get(opts, "channel-rel-gate", "0.05"))
    noise_scale = parse(Float64, get(opts, "noise-scale", "2.0"))

    # ----------------------------------------------------------------------- #
    # Noise floor: a same-q replicate comparison. Runs first (and, with neither
    # --q6-project nor --q14-project, alone) so that its result, `floor_c`, is
    # available to anchor the normal comparison's channel gate below.
    # ----------------------------------------------------------------------- #
    floor_c = nothing
    if replicate_mode
        rep_sep = parse_rational(rep_sep_str)
        println("### replicate comparison (noise floor): a=$rep_a_dir  b=$rep_b_dir  sep=$rep_sep")
        a_path = find_bounds_jld(rep_a_dir, rep_sep)
        b_path = find_bounds_jld(rep_b_dir, rep_sep)
        if a_path === nothing || b_path === nothing
            error(a_path === nothing ?
                "no bounds output for separation $rep_sep in --replicate-a ($rep_a_dir)" :
                "no bounds output for separation $rep_sep in --replicate-b ($rep_b_dir)")
        end
        a = load_bounds(a_path)
        b = load_bounds(b_path)
        println("  a: $(basename(a_path))  m=$(a.m)  augmented=$(a.augmented)$(a.augmented ? " (m_aug=$(a.m_aug))" : "")")
        println("  b: $(basename(b_path))  m=$(b.m)  augmented=$(b.augmented)$(b.augmented ? " (m_aug=$(b.m_aug))" : "")")
        floor_c = compare_one(a, b; min_abs=min_abs)
        print_pair_stats(floor_c; min_abs=min_abs, label_a="a", label_b="b")
        println()
        if !normal_mode
            println("Replicate-only run: this is the noise floor a q6-vs-q14 comparison should be read against, not a pass/fail verdict on its own.")
            return nothing
        end
    end

    # ----------------------------------------------------------------------- #
    # The normal q6-vs-q14 comparison, unchanged from before when neither
    # --q6-project-override nor --replicate-a/--replicate-b/--noise-scale are
    # given.
    # ----------------------------------------------------------------------- #
    println("q6 project:  $q6_dir")
    println("q14 project: $q14_dir")
    if !isempty(overrides)
        for (sep, dir) in sort(collect(overrides); by=first)
            println("  override $sep -> $dir")
        end
    end
    println("separations: ", join(string.(seps), ", "))
    gate_msg = "gates: trace within $(trace_rel_gate*100)% rel or $trace_abs_gate abs (looser wins); gated channels within $(channel_rel_gate*100)%"
    if floor_c !== nothing
        gate_msg *= " (raw); floor-anchored gate is $(noise_scale)x the replicate's max/median"
    end
    println(gate_msg)
    println()

    n_pass = 0
    n_fail = 0
    n_skipped = 0
    for sep in seps
        println("### separation $sep")
        this_q6_dir = get(overrides, sep, q6_dir)
        this_q6_dir == q6_dir || println("  (q6 side overridden for this separation: $this_q6_dir)")
        q6_path = find_bounds_jld(this_q6_dir, sep)
        if q6_path === nothing
            println("  q6 output not found in $this_q6_dir. Has the bounds job for this point finished?")
            n_skipped += 1
            println()
            continue
        end
        q14_path = find_bounds_jld(q14_dir, sep)
        if q14_path === nothing
            println("  reference not computed yet, run the boundsonly sweep")
            n_skipped += 1
            println()
            continue
        end

        local q6, q14
        try
            q6 = load_bounds(q6_path)
            q14 = load_bounds(q14_path)
        catch err
            println("  skip (", sprint(showerror, err), ")")
            n_skipped += 1
            println()
            continue
        end

        println("  q6:  $(basename(q6_path))  m=$(q6.m)  augmented=$(q6.augmented)$(q6.augmented ? " (m_aug=$(q6.m_aug))" : "")")
        println("  q14: $(basename(q14_path))  m=$(q14.m)  augmented=$(q14.augmented)$(q14.augmented ? " (m_aug=$(q14.m_aug))" : "")")

        c = compare_one(q6, q14; min_abs=min_abs)
        v = verdict(c; trace_rel_gate=trace_rel_gate, trace_abs_gate=trace_abs_gate, channel_rel_gate=channel_rel_gate)
        print_pair_stats(c; min_abs=min_abs, label_a="q6", label_b="q14")

        local decisive
        if floor_c === nothing
            println("  verdict: ", v.pass ? "PASS" : "FAIL",
                    v.pass ? "" : "  (trace $(v.trace_ok ? "ok" : "FAILS") gate, channels $(v.channel_ok ? "ok" : "FAILS") gate)")
            decisive = v
        else
            println("  verdict (raw, $(channel_rel_gate*100)% flat gate): ", v.pass ? "PASS" : "FAIL",
                    v.pass ? "" : "  (trace $(v.trace_ok ? "ok" : "FAILS") gate, channels $(v.channel_ok ? "ok" : "FAILS") gate)")
            fv = floor_verdict(c, floor_c; noise_scale=noise_scale, trace_rel_gate=trace_rel_gate,
                               trace_abs_gate=trace_abs_gate, channel_rel_gate=channel_rel_gate)
            @printf("  verdict (floor-anchored, gate max %.3g / median %.3g = %.3gx replicate): %s%s\n",
                    fv.max_gate, fv.median_gate, noise_scale, fv.pass ? "PASS" : "FAIL",
                    fv.pass ? "" : "  (trace $(fv.trace_ok ? "ok" : "FAILS") gate, channels $(fv.channel_ok ? "ok" : "FAILS") gate)")
            decisive = fv
        end
        decisive.pass ? (n_pass += 1) : (n_fail += 1)
        println()
    end

    tail = floor_c === nothing ? "." : ", by the floor-anchored verdict."
    println("$n_pass pass, $n_fail fail, $n_skipped skipped (of $(length(seps)) separation(s))$tail")
    if n_skipped > 0
        println("Skipped separations do not count against the verdict; rerun once their inputs exist.")
    end
    if n_fail == 0 && n_skipped == 0 && n_pass == length(seps)
        println("\nq=6 passes at every separation checked. This is the pass condition for regenerating the production launchers at --power-iterations 6.")
    elseif n_fail > 0
        println("\nq=6 does not reproduce the q=14 bounds at every separation checked; keep the production launchers at --power-iterations 14.")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
