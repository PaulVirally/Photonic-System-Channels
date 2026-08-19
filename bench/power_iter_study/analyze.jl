#!/usr/bin/env julia
"""
    bench/power_iter_study/analyze.jl

Read the per-q bounds outputs `run_study.sh` wrote and say, per separation, how
low `--power-iterations` can go before the bounds move.

    julia --project=. bench/power_iter_study/analyze.jl \\
        --root /home/paulv/Projects/Photonic-System-Channels/projects/power_iter_study/k4000

Options:

  --root <dir>        the study's project root, holding `q01/`, `q02/`, ...,
                      `q08/`, `q08r/` and `timings_gpu*.csv` (required)
  --ref 8             the reference q
  --conv-factor 2.0   how far above the sketch-noise floor the second-largest q
                      may sit before the reference is called unconverged
  --out <path>        CSV to write (default `power_iter_summary.csv` next to this
                      script)
  --trace-floor 1e-4  traces below this are not reported at all, so a separation
                      where both q and the reference sit under it passes
                      automatically
  --trace-abs 1e-5    absolute slack on the trace, a decade under the floor
  --trace-rtol 0.01   relative slack on the trace
  --chan-floor 1e-5   channels whose reference bound is under this cannot move a
                      reportable trace, so they are reported but not gated
  --chan-rtol 0.05    relative slack on the gated channels

Reads only JLD2 and a CSV, so it runs anywhere.

# What is compared

`compute_bounds.jl` writes one JLD per experiment into `--project`, named by
`file_prefix(smr)` -- geometry and separation, no q -- which is why each q needs
its own directory. The keys this reads are:

  * `bounds_dual_basis`: the bound on `sigma_n(P_rs)` for each index `n`, length
    `num_pos` *after* the `--gamma-rtol` cut. Already on the singular-value
    scale (`src/bounds.jl` stores `sqrt(best_dual)`), not the squared one, so the
    reporting floors apply to it directly.
  * `opt_taus`: the tau that achieved each of those.
  * `true_bounds`: the per-index minimum of `bounds_dual_basis` and the two
    analytical forms. Reported as a second trace, since that is what a figure
    would actually plot, but the gate runs on `bounds_dual_basis` because that is
    the only one of the three that reads the RSVD basis rather than just its
    eigenvalues.
  * `Gamma`: the whole sorted `Asym(G0_ur)` spectrum, from which the positive
    count comes. The kept count is `length(bounds_dual_basis)`.

The trace bound is the sum over indices. Nothing below `--trace-floor` gets
reported, so a q that only disagrees down there has not cost anything, and the
gate is anchored to the floor rather than to a bare relative tolerance:

  1. both traces under the floor -> pass, there is nothing to be wrong about;
  2. otherwise `|trace_q - trace_ref| <= max(rtol * trace_ref, abs)`;
  3. and, over the channels whose reference bound is at or above
     `--chan-floor`, a relative deviation of at most `--chan-rtol`.

Two sums are printed but not gated, both there to confirm they stay far under the
trace floor: `tail`, the summed absolute deviation over the channels below
`--chan-floor`, and `drop`, the mass of the channels one run kept and the other
cut because `--gamma-rtol` landed in a different place. `drop` is already inside
the trace difference; the channel gate cannot see it, which is why it is printed.

# The reference, and why there are two of it

`--seed` only reaches the sketch on the panel RSVD path
(`MatrixFreeRandomizedLinearAlgebra` throws on a seed without a plan), and at
1/2 lambda with k=4000 the sketch fits an A6000, so the runs take the in-memory
path and each draws its own Gaussian. Two runs at the same q therefore differ.
`run_study.sh` runs the reference twice, into `q08/` and `q08r/`, and the `8r`
row of each table is that difference: the noise floor a low-q row has to be read
against. A q whose deviation sits at the `8r` level is not distinguishable from
the reference, whatever the absolute number looks like.

# Why the reference is checked before it is used

The reference is q=8, not the production q=14, because no run in this study is
allowed to pass about an hour on molering and q=14 at 1/2 lambda is ~72 min. That
makes it a reference of convenience rather than a ground truth, so it has to earn
the role: if q=8 has not itself converged, every deviation measured against it is
measuring the reference's error and the whole table is worthless.

The stand-in for a q=14 ground truth is successive-q convergence. The
second-largest q in the study (q=6 by default) is compared against the reference,
and its deviation is required to sit within `--conv-factor` of the `q8` vs `q8r`
sketch-noise floor on both gated quantities, the trace and the worst gated
channel. If it does, then going from 6 to 8 power iterations changed nothing
distinguishable from redrawing the sketch, and there is no reason to believe 8 to
14 would either. If it does not, the separation gets

    not converged at q=8 -- reference unreliable, raise QS

instead of a verdict. The per-q rows are still printed, because they are the
evidence, but no smallest-acceptable-q is issued off an unconverged reference.

Two cases sidestep the check rather than fail it. A separation whose reference
trace is under `--trace-floor` has nothing reportable to be wrong about, so
convergence is moot and every q passes. And where the noise floor is exactly zero
or undefined -- no replicate, or no gated channel to measure -- the ratio cannot
be formed, and the separation is reported as unassessable rather than either
converged or not.
"""

using JLD2
using Printf
using Statistics

const DEFAULT_REF = 8
const DEFAULT_CONV_FACTOR = 2.0
const DEFAULT_TRACE_FLOOR = 1e-4
const DEFAULT_TRACE_ABS = 1e-5
const DEFAULT_TRACE_RTOL = 0.01
const DEFAULT_CHAN_FLOOR = 1e-5
const DEFAULT_CHAN_RTOL = 0.05

function parse_cli(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("Expected an option starting with --, got '$(argv[i])'")
        key = argv[i][3:end]
        if i + 1 <= length(argv) && !startswith(argv[i + 1], "--")
            opts[key] = argv[i + 1]
            i += 2
        else
            opts[key] = "true"
            i += 1
        end
    end
    return opts
end

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

"""
    find_runs(root) -> Vector{NamedTuple}

One entry per (q directory, JLD) pair found under `root`. `q` is the power
iteration count, `replicate` marks the `q08r`-style second run of the reference,
and `sep` is the `<num>ss<den>` token out of the filename, which is how the
separations are told apart without rebuilding `file_prefix`.
"""
function find_runs(root::AbstractString)
    isdir(root) || error("no such directory: $root")
    runs = NamedTuple[]
    for entry in sort(readdir(root))
        m = match(r"^q(\d+)(r?)$", entry)
        m === nothing && continue
        q = parse(Int, m.captures[1])
        replicate = m.captures[2] == "r"
        dir = joinpath(root, entry)
        isdir(dir) || continue
        for f in sort(readdir(dir))
            endswith(f, ".jld") || continue
            sm = match(r"__(\d+ss\d+)__", f)
            sm === nothing && continue
            push!(runs, (q=q, replicate=replicate, sep=sm.captures[1],
                         label=replicate ? "$(q)r" : string(q),
                         path=joinpath(dir, f)))
        end
    end
    return runs
end

"""
    read_bounds(path) -> NamedTuple or nothing

The arrays the comparison needs, or `nothing` if the bounds loop never got far
enough to write them. `Gamma` is the whole sorted spectrum; `bounds` is only as
long as the kept positive block.
"""
function read_bounds(path::AbstractString)
    return jldopen(path, "r") do io
        haskey(io, "bounds_dual_basis") || return nothing
        bounds = Array{Float64}(io["bounds_dual_basis"])
        Γ = haskey(io, "Γ") ? Array{Float64}(real.(io["Γ"])) : Float64[]
        taus = haskey(io, "opt_taus") ? Array{Float64}(io["opt_taus"]) : fill(NaN, length(bounds))
        true_b = haskey(io, "true_bounds") ? Array{Float64}(io["true_bounds"]) : Float64[]
        return (bounds=bounds, Γ=Γ, opt_taus=taus, true_bounds=true_b)
    end
end

"""
    read_timings(root) -> Dict{(sep, label, stage), Float64}

Seconds per (separation token, q label, stage) out of `timings_gpu*.csv`. Later
rows win, so a rerun after a failure replaces the failure, and rows with a
non-zero exit status are dropped.

The launcher logs the separation as the rational it passed to julia (`1//16`),
while the JLD filenames carry `file_prefix`'s token (`1ss16`); the rational is
translated here so the two line up.
"""
function read_timings(root::AbstractString)
    out = Dict{Tuple{String,String,String},Float64}()
    for f in sort(readdir(root))
        startswith(f, "timings_gpu") && endswith(f, ".csv") || continue
        for (n, line) in enumerate(eachline(joinpath(root, f)))
            n == 1 && startswith(line, "tier,") && continue
            fields = split(strip(line), ',')
            length(fields) == 6 || continue
            sep = replace(fields[2], "//" => "ss")
            label, stage = fields[3], fields[4]
            secs = tryparse(Float64, fields[5])
            status = tryparse(Int, fields[6])
            (secs === nothing || status === nothing) && continue
            status == 0 || continue
            out[(sep, label, stage)] = secs
        end
    end
    return out
end

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

finite_sum(v) = isempty(v) ? 0.0 : sum(x -> isfinite(x) ? x : 0.0, v)

"""
    spectrum_deviation(a, b, n) -> Float64

Largest relative deviation of the leading `n` eigenvalues. Zero where both
entries are exactly equal, so an empty or padded comparison does not manufacture
a disagreement.
"""
function spectrum_deviation(a::AbstractVector, b::AbstractVector, n::Int)
    n = min(n, length(a), length(b))
    n <= 0 && return NaN
    worst = 0.0
    for i in 1:n
        ai, bi = a[i], b[i]
        (isfinite(ai) && isfinite(bi)) || continue
        denom = max(abs(ai), abs(bi))
        denom == 0 && continue
        worst = max(worst, abs(ai - bi) / denom)
    end
    return worst
end

"""
    compare(run, ref; chan_floor) -> NamedTuple

Everything one table row needs. Channels are compared over the common prefix of
the two kept blocks; where the kept counts differ, the extra channels of the
longer one are charged to `tail_abs`, since they are a real difference in what
the run reports.
"""
function compare(run, ref; chan_floor::Float64)
    bq, br = run.bounds, ref.bounds
    n = min(length(bq), length(br))

    gated = [i for i in 1:n if isfinite(br[i]) && br[i] >= chan_floor]
    rels = [abs(bq[i] - br[i]) / br[i] for i in gated if isfinite(bq[i])]
    chan_max = isempty(rels) ? NaN : maximum(rels)
    chan_med = isempty(rels) ? NaN : median(rels)

    # Below the channel floor: reported, not gated. A channel this small cannot
    # move a reportable trace, but the sum over all of them could, so it is worth
    # seeing.
    tail = 0.0
    for i in 1:n
        (isfinite(br[i]) && br[i] >= chan_floor) && continue
        (isfinite(bq[i]) && isfinite(br[i])) || continue
        tail += abs(bq[i] - br[i])
    end
    # Channels one run kept and the other did not, because the gamma-rtol cut fell
    # in a different place. Not a deviation on a shared channel but a difference
    # in what the run reports at all, so it is counted separately. The trace gate
    # already sees this mass; the channel gate cannot.
    dropped = finite_sum(view(bq, (n + 1):length(bq))) +
              finite_sum(view(br, (n + 1):length(br)))

    trace_q, trace_ref = finite_sum(bq), finite_sum(br)
    true_q, true_ref = finite_sum(run.true_bounds), finite_sum(ref.true_bounds)

    tau_devs = [abs(run.opt_taus[i] - ref.opt_taus[i]) for i in gated
                if i <= length(run.opt_taus) && i <= length(ref.opt_taus) &&
                   isfinite(run.opt_taus[i]) && isfinite(ref.opt_taus[i])]

    return (kept=length(bq), kept_ref=length(br), common=n,
            npos=count(>(0.0), run.Γ), npos_ref=count(>(0.0), ref.Γ),
            eig_max=spectrum_deviation(run.Γ, ref.Γ, length(bq)),
            trace=trace_q, trace_ref=trace_ref,
            trace_abs=abs(trace_q - trace_ref),
            trace_rel=trace_ref == 0 ? NaN : abs(trace_q - trace_ref) / trace_ref,
            true_trace=true_q, true_trace_ref=true_ref,
            true_trace_rel=true_ref == 0 ? NaN : abs(true_q - true_ref) / true_ref,
            n_gated=length(gated), chan_max=chan_max, chan_med=chan_med,
            tail_abs=tail, dropped_abs=dropped,
            tau_max=isempty(tau_devs) ? NaN : maximum(tau_devs),
            tau_med=isempty(tau_devs) ? NaN : median(tau_devs))
end

"""
    verdict(c, thresholds) -> (pass::Bool, why::String)

The floor-anchored gate. A trace nobody will report cannot be wrong, so the
first branch passes on it; otherwise the trace has to agree to `trace_rtol` or
to `trace_abs`, whichever is looser, and the channels above `chan_floor` to
`chan_rtol`.
"""
function verdict(c, th)
    if c.trace < th.trace_floor && c.trace_ref < th.trace_floor
        return true, "both traces under the $(th.trace_floor) reporting floor"
    end
    slack = max(th.trace_rtol * c.trace_ref, th.trace_abs)
    if !(c.trace_abs <= slack)
        return false, @sprintf("trace off by %.3e, slack %.3e", c.trace_abs, slack)
    end
    if c.n_gated == 0
        return true, "trace within slack, no channel above the $(th.chan_floor) floor"
    end
    if !(isfinite(c.chan_max) && c.chan_max <= th.chan_rtol)
        return false, @sprintf("worst gated channel off by %.3f (limit %.3f)",
                               c.chan_max, th.chan_rtol)
    end
    return true, @sprintf("trace within %.3e, %d channels within %.3f",
                          slack, c.n_gated, c.chan_max)
end

"""
    convergence(prev, noise, ref_trace, th) -> (status, message)

Has the reference itself converged? `prev` is the second-largest q compared
against the reference, `noise` is the reference's replicate compared against it.
Both are `compare` results.

`status` is one of:

- `:moot` -- the reference trace is under the reporting floor, so nothing here
  would be reported and there is nothing for the reference to be wrong about.
- `:ok` -- on every quantity that can be measured, `prev`'s deviation is within
  `th.conv_factor` of the noise floor. Going from `prev.q` to the reference
  changed nothing distinguishable from redrawing the sketch.
- `:not_converged` -- it is not, so deviations measured against this reference
  are measuring the reference's own error.
- `:unassessable` -- there is no `prev` run, no replicate, or no quantity whose
  noise floor is a usable positive number.

The trace and the worst gated channel are checked independently and both have to
pass. A zero noise floor cannot be scaled, so that quantity is skipped unless
`prev` is zero there too, in which case it agrees exactly and counts as passing.
"""
function convergence(prev, noise, ref_trace, th)
    ref_trace < th.trace_floor &&
        return :moot, @sprintf("reference trace %.3e is under the %.1e reporting floor",
                               ref_trace, th.trace_floor)
    (prev === nothing || noise === nothing) &&
        return :unassessable, "no second-largest q, or no replicate, to compare"

    checks = (("trace", prev.trace_abs, noise.trace_abs),
              ("chan_max", prev.chan_max, noise.chan_max))
    verdicts = String[]
    assessed = 0
    converged = true
    for (name, dev, nz) in checks
        if !isfinite(dev) || !isfinite(nz)
            push!(verdicts, "$(name) n/a")
            continue
        end
        if nz == 0
            if dev == 0
                assessed += 1
                push!(verdicts, "$(name) exact")
            else
                push!(verdicts, @sprintf("%s noise floor is 0, ratio undefined (dev %.2e)", name, dev))
            end
            continue
        end
        assessed += 1
        ratio = dev / nz
        ratio <= th.conv_factor || (converged = false)
        push!(verdicts, @sprintf("%s %.2e vs noise %.2e (%.1fx)", name, dev, nz, ratio))
    end

    msg = join(verdicts, ", ")
    assessed == 0 && return :unassessable, msg
    return (converged ? :ok : :not_converged), msg
end

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

const CSV_HEADER = "separation,q,replicate,operator_passes,rsvd_s,cost_per_pass," *
                   "bounds_s,num_pos,kept,eig_max_rel,trace,trace_ref,trace_abs," *
                   "trace_rel,true_trace,true_trace_rel,n_gated,chan_max_rel," *
                   "chan_median_rel,tail_abs,dropped_abs,tau_max,tau_median,pass"

"Sort key for a `<num>ss<den>` separation token: the separation itself."
sep_value(sep::AbstractString) = (p = split(sep, "ss"); parse(Int, p[1]) / parse(Int, p[2]))
sep_rational(sep::AbstractString) = replace(sep, "ss" => "//")

function report_separation(io_csv, sep, runs, timings, th)
    ref_idx = findfirst(r -> r.q == th.ref && !r.replicate, runs)
    println("\n", "="^104)
    println("separation $(sep)  (x-separation $(sep_rational(sep)) lambda)")
    if ref_idx === nothing
        println("  no q=$(th.ref) reference run in this directory: nothing to compare against")
        return nothing
    end

    ref_data = read_bounds(runs[ref_idx].path)
    if ref_data === nothing
        println("  the q=$(th.ref) reference has no bounds_dual_basis: its bounds job did not finish")
        return nothing
    end

    ref_passes = 2 * th.ref + 2
    ref_rsvd = get(timings, (sep, string(th.ref), "rsvd"), NaN)
    ordered = sort(runs; by=r -> (r.q, r.replicate))

    # Cost. `cost/pass` is the measured wall time per operator pass relative to
    # the reference's: 1.00 means the RSVD scaled exactly with (2q+2), and above
    # 1.00 means the fixed cost (startup, the RS block, the dense algebra on the
    # N x c matrices) is what the low-q run is paying for instead.
    println("\n  cost")
    @printf("  %-5s %7s %10s %10s %10s %10s\n",
            "q", "passes", "rsvd_s", "cost/pass", "bounds_s", "total_h")
    for r in ordered
        passes = 2 * r.q + 2
        rsvd_s = get(timings, (sep, r.label, "rsvd"), NaN)
        bounds_s = get(timings, (sep, r.label, "bounds"), NaN)
        cpp = (isfinite(rsvd_s) && isfinite(ref_rsvd) && ref_rsvd > 0) ?
              (rsvd_s / ref_rsvd) / (passes / ref_passes) : NaN
        @printf("  %-5s %7d %10.0f %10.2f %10.0f %10.2f\n",
                r.label, passes, rsvd_s, cpp, bounds_s,
                (isfinite(rsvd_s) ? rsvd_s : 0.0) / 3600 +
                (isfinite(bounds_s) ? bounds_s : 0.0) / 3600)
    end

    # Quality. `eig_max` is over the kept block, `chan_*` over the gated channels
    # only, `tail` over the channels under the channel floor, and `drop` over the
    # channels one run kept and the other cut.
    println("\n  quality against q=$(th.ref)")
    @printf("  %-5s %6s %6s %10s %11s %10s %9s %10s %9s %9s %9s %8s %5s\n",
            "q", "npos", "kept", "eig_max", "trace", "|dtrace|", "dtrace/t",
            "chan_max", "chan_med", "tail", "drop", "dtau_max", "pass")

    verdicts = Dict{String,Bool}()
    for r in ordered
        data = read_bounds(r.path)
        if data === nothing
            @printf("  %-5s (no bounds_dual_basis: the bounds job did not finish)\n", r.label)
            continue
        end
        c = compare(data, ref_data; chan_floor=th.chan_floor)
        pass, _why = verdict(c, th)
        verdicts[r.label] = pass

        passes = 2 * r.q + 2
        rsvd_s = get(timings, (sep, r.label, "rsvd"), NaN)
        bounds_s = get(timings, (sep, r.label, "bounds"), NaN)
        cpp = (isfinite(rsvd_s) && isfinite(ref_rsvd) && ref_rsvd > 0) ?
              (rsvd_s / ref_rsvd) / (passes / ref_passes) : NaN

        @printf("  %-5s %6d %6d %10.2e %11.4e %10.2e %9.2e %10.2e %9.2e %9.2e %9.2e %8.2e %5s\n",
                r.label, c.npos, c.kept, c.eig_max, c.trace, c.trace_abs,
                c.trace_rel, c.chan_max, c.chan_med, c.tail_abs, c.dropped_abs,
                c.tau_max, pass ? "yes" : "NO")

        println(io_csv, join(string.([
            sep, r.q, r.replicate, passes, rsvd_s, cpp, bounds_s,
            c.npos, c.kept, c.eig_max, c.trace, c.trace_ref, c.trace_abs,
            c.trace_rel, c.true_trace, c.true_trace_rel, c.n_gated,
            c.chan_max, c.chan_med, c.tail_abs, c.dropped_abs,
            c.tau_max, c.tau_med, pass]), ','))
    end

    # Extras that do not fit the table but decide how it should be read.
    ref_c = compare(ref_data, ref_data; chan_floor=th.chan_floor)
    @printf("\n  reference trace %.6e over %d kept channels (%d gated at >= %.1e), true_bounds trace %.6e\n",
            ref_c.trace, ref_c.kept, ref_c.n_gated, th.chan_floor, ref_c.true_trace)
    ref_c.trace < th.trace_floor &&
        @printf("  the reference trace is under the %.1e reporting floor, so nothing here would be reported and every q passes\n",
                th.trace_floor)
    noise_c = nothing
    rep_idx = findfirst(r -> r.q == th.ref && r.replicate, runs)
    if rep_idx === nothing
        println("  no q=$(th.ref) replicate: there is no sketch-noise floor to read the low-q rows against")
    else
        rep = read_bounds(runs[rep_idx].path)
        if rep !== nothing
            noise_c = compare(rep, ref_data; chan_floor=th.chan_floor)
            @printf("  sketch-noise floor (q=%d replicate): |dtrace| %.2e, chan_max %.2e, drop %.2e, dtau_max %.2e\n",
                    th.ref, noise_c.trace_abs, noise_c.chan_max, noise_c.dropped_abs, noise_c.tau_max)
            println("  a row at or under those numbers is not distinguishable from the reference")
        end
    end

    # Has the reference converged? The second-largest q stands in for the q=14
    # ground truth this study cannot afford.
    prev_q = maximum([r.q for r in runs if !r.replicate && r.q < th.ref]; init=0)
    prev_c = nothing
    if prev_q > 0
        prev_run = runs[findfirst(r -> r.q == prev_q && !r.replicate, runs)]
        prev_data = read_bounds(prev_run.path)
        prev_data === nothing || (prev_c = compare(prev_data, ref_data; chan_floor=th.chan_floor))
    end
    status, why = convergence(prev_c, noise_c, ref_c.trace, th)
    @printf("\n  convergence of the q=%d reference (q=%d against it, limit %.1fx the noise floor): %s\n",
            th.ref, prev_q, th.conv_factor, why)

    if status === :not_converged
        println("  not converged at q=$(th.ref) -- reference unreliable, raise QS")
        println("  the rows above are printed as evidence; no verdict is issued off an unconverged reference")
        return nothing
    elseif status === :unassessable
        println("  convergence of the reference could not be assessed, so no verdict is issued")
        println("  run the q=$(th.ref) replicate and at least one q below it, then re-analyse")
        return nothing
    elseif status === :moot
        println("  convergence is moot here: nothing at this separation is reportable")
    else
        println("  q=$(prev_q) to q=$(th.ref) is indistinguishable from redrawing the sketch, so the reference stands")
    end

    # The verdict. Smallest q that passes, ignoring the replicate.
    candidates = sort([r.q for r in runs if !r.replicate && get(verdicts, r.label, false)])
    if isempty(candidates)
        println("  VERDICT $(sep): no q passed. Keep q=$(th.ref).")
    else
        smallest = first(candidates)
        gaps = [r.q for r in runs if !r.replicate && r.q > smallest &&
                                    haskey(verdicts, r.label) && !verdicts[r.label]]
        println("  VERDICT $(sep): smallest acceptable q = $(smallest) " *
                "(trace within $(th.trace_rtol) rel or $(th.trace_abs) abs, " *
                "gated channels within $(th.chan_rtol))")
        isempty(gaps) ||
            println("            NOTE: q = $(join(gaps, ", ")) failed above it, so the " *
                    "pass is not monotone in q; treat $(smallest) as luck, not as convergence")
    end
    return nothing
end

function main(argv::Vector{String})
    opts = parse_cli(argv)
    root = get(opts, "root", "")
    isempty(root) && error("--root is required")
    th = (ref=parse(Int, get(opts, "ref", string(DEFAULT_REF))),
          conv_factor=parse(Float64, get(opts, "conv-factor", string(DEFAULT_CONV_FACTOR))),
          trace_floor=parse(Float64, get(opts, "trace-floor", string(DEFAULT_TRACE_FLOOR))),
          trace_abs=parse(Float64, get(opts, "trace-abs", string(DEFAULT_TRACE_ABS))),
          trace_rtol=parse(Float64, get(opts, "trace-rtol", string(DEFAULT_TRACE_RTOL))),
          chan_floor=parse(Float64, get(opts, "chan-floor", string(DEFAULT_CHAN_FLOOR))),
          chan_rtol=parse(Float64, get(opts, "chan-rtol", string(DEFAULT_CHAN_RTOL))))
    out = get(opts, "out", joinpath(@__DIR__, "power_iter_summary.csv"))

    runs = find_runs(root)
    isempty(runs) && error("no q*/*.jld under $root; has the study run?")
    timings = read_timings(root)

    println("="^104)
    println("Power-iteration quality study")
    println("  root      $root")
    println("  reference q = $(th.ref) (checked for convergence against the next q down, ")
    @printf("                limit %.1fx the q=%d replicate's sketch-noise floor)\n",
            th.conv_factor, th.ref)
    @printf("  trace: reported only above %.1e; agreement to %.1f%% or %.1e absolute\n",
            th.trace_floor, 100 * th.trace_rtol, th.trace_abs)
    @printf("  channels: gated at >= %.1e absolute, to %.1f%% relative\n",
            th.chan_floor, 100 * th.chan_rtol)
    println("  bounds_dual_basis is on the sigma_n(P_rs) scale, so the floors apply to it directly")
    isempty(timings) && println("  NOTE: no timings_gpu*.csv under the root; the wall-time columns will be blank")

    open(out, "w") do io_csv
        println(io_csv, CSV_HEADER)
        for sep in sort(unique(r.sep for r in runs); by=sep_value)
            report_separation(io_csv, sep, filter(r -> r.sep == sep, runs), timings, th)
        end
    end

    println("\n", "="^104)
    println("Wrote $out")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
