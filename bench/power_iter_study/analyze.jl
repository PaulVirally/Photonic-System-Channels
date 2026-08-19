#!/usr/bin/env julia
"""
    bench/power_iter_study/analyze.jl

Read the per-q outputs `run_study.sh` wrote and say, per separation, how low
`--power-iterations` can go before the bounds move.

    julia --project=. bench/power_iter_study/analyze.jl \\
        --root /home/paulv/Projects/Photonic-System-Channels/projects/power_iter_study/k4000

Two modes, picked per file from the keys it holds rather than from a flag:

  * **bounds mode**, on a JLD with `bounds_dual_basis`, i.e. one `compute_bounds.jl`
    wrote into the *project* tree. This is the study's real answer and everything
    below the next section is about it.
  * **spectrum mode**, on a JLD with `UR_asym/D` and no bounds, i.e. one
    `generate_rsvd.jl` wrote into the *scratch* tree. It reports what the `q`
    sweep did to the `Asym(G0_ur)` eigenvalues and forecasts what the bounds half
    would cost, and its verdict is **provisional**: see "Spectrum mode" below.

A root may hold either kind, or a mix, and both roots can be given at once:

    julia --project=. bench/power_iter_study/analyze.jl \\
        --root      /home/paulv/Projects/Photonic-System-Channels/projects/power_iter_study/k4000 \\
        --rsvd-root /home/molering/fatmole/paulv/Photonic-System-Channels/power_iter_study/k4000

Options:

  --root <dir>        a study root holding `q01/`, `q02/`, ..., `q08/`, `q08r/`.
                      The project root also holds `timings_gpu*.csv`. Required
                      unless `--rsvd-root` is given
  --rsvd-root <dir>   a second root to scan, for the case where the RSVD output
                      (scratch) and the bounds output (project) are in different
                      trees. Scanned exactly like `--root`; the split is a
                      convenience, not a meaning
  --ref 8             the reference q
  --conv-factor 2.0   how far above the sketch-noise floor the second-largest q
                      may sit before the reference is called unconverged
  --out <path>        bounds-mode CSV to write (default `power_iter_summary.csv`
                      next to this script)
  --trace-floor 1e-4  traces below this are not reported at all, so a separation
                      where both q and the reference sit under it passes
                      automatically
  --trace-abs 1e-5    absolute slack on the trace, a decade under the floor
  --trace-rtol 0.01   relative slack on the trace
  --chan-floor 1e-5   channels whose reference bound is under this cannot move a
                      reportable trace, so they are reported but not gated
  --chan-rtol 0.05    relative slack on the gated channels

Spectrum mode only:

  --rsvd-out <path>   spectrum-mode CSV (default `power_iter_rsvd_summary.csv`)
  --gamma-rtol 1e-12  the cut whose kept count is recomputed from `UR_asym/D`.
                      Must match what the bounds stage will be run with
  --noise-factor 10   the provisional match threshold is this times the measured
                      sketch-noise floor
  --match-floor 1e-6  ... or this, whichever is larger
  --forecast narval   coefficient set for the bounds-cost forecast, or `none`.
                      Not `molering`: see "The bounds-cost forecast" below
  --cells 16,16,16    geometry the forecast is for (one body)
  --scale 1//32       cell edge in wavelengths, for the forecast
  --rank <k>          sketch rank for the forecast. Defaults to the `k<N>`
                      component of the root path, else 4000
  --oversamples 50    for the forecast

Reads only JLD2 and a CSV, plus (for the forecast alone, and optionally)
`bench/cost_model.jl`, which has no dependencies of its own. So it runs anywhere.

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

# Spectrum mode

What `STAGES=rsvd` leaves behind is `UR_asym/D`, the whole `Asym(G0_ur)` spectrum,
plus `UR_asym/num_pos`, and nothing about the bounds. So spectrum mode reports the
one thing those files can answer -- did the eigenvalues move -- and is explicit
that it is not the study's question.

Per q, against the same reference and the same replicate the bounds mode uses:

  * `npos`, the saved `UR_asym/num_pos`, and `kept`, how many of those survive
    `--gamma-rtol`. `kept` is recomputed here by the rule `_gamma_kept_count` in
    `src/bounds.jl` applies, on the same input: sort `UR_asym/D` descending, take
    the positive prefix `num_pos` long, keep `Gamma[i] >= gamma_rtol * Gamma[1]`.
    It is the `m` the bounds job would run at, and the cost driver.
  * `dev_max` and `dev_med`, the largest and median relative deviation from the
    reference over the *mutually* kept set -- the leading `min(kept, kept_ref)`
    entries of the two descending spectra, matched by index.
  * the sketch-noise floor, the same two numbers between the reference and its
    replicate. Everything above is read against it.

**Index-matching is only meaningful above the noise floor.** Two runs order their
own noise, not a shared spectrum, so at the bottom of the kept block the `i`th
eigenvalue of one run and the `i`th of the other are not the same direction and a
large deviation there is not a disagreement about anything. `dev_med` is over the
whole mutually kept block for exactly this reason: a `dev_max` sitting on the last
few indices while `dev_med` stays at the floor is the signature of that, not of a
spectrum that moved.

The provisional verdict is the smallest q whose mutually-kept eigenvalues match
the reference to `max(--noise-factor * noise floor, --match-floor)`. It is
**spectrum-level only**. `q` reaches the bounds through the RSVD *basis*, and two
runs can agree on `Gamma` to ten digits while their eigenvectors span slightly
different subspaces, which is exactly what `bounds_dual_basis` -- and only
`bounds_dual_basis` -- can see. So a spectrum-level pass is necessary and not
sufficient, and the binding verdict needs the bounds stage.

The convergence gate runs here too, on `dev_max` and `dev_med` instead of the
trace and the worst channel. An unconverged reference suppresses the provisional
verdict exactly as it suppresses the real one. A *missing* replicate does not:
the verdict is still issued, against the bare `--match-floor`, with a note saying
the floor is unmeasured.

# The bounds-cost forecast

Spectrum mode also predicts what the bounds half would cost at the `kept` it just
measured, which is the number that decides whether to run it here at all. It
`include`s `bench/cost_model.jl` (dependency-free by design) and calls
`predict(ComputeBounds, ...)` with `num_pos` set to the measured `kept` and the
coefficient set's `bounds_m_mode` forced to `"fraction"`, so the separation-keyed
truncation power law in `bounds_m` cannot cap a count that is already the
post-truncation one.

The label says **narval-equivalent** because the coefficients are narval's.
molering's own bounds coefficients are known-garbage in the conservative
direction: its fitted `sync_latency` is 6.7 ms against narval's 15 us, the bounds
model spends one device sync per probe over an `O(m^2)` loop, and that single
coefficient then carries the whole estimate -- 147 h for a job whose measured
breakdown extrapolates to about nine minutes. narval's set does reproduce the
request `create_jobs.jl` makes for this exact job. An A6000 is slower than an A100,
but not by an order of magnitude, so read the forecast as a lower bound with the
card difference on top, and see the README's "What this costs".
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

# Spectrum mode. `DEFAULT_GAMMA_RTOL` must stay equal to the same-named constant in
# `src/bounds.jl` and to `GAMMA_RTOL` in `run_study.sh`: the kept count only means
# something if it is the cut the bounds stage will actually apply.
const DEFAULT_GAMMA_RTOL = 1e-12
const DEFAULT_NOISE_FACTOR = 10.0
const DEFAULT_MATCH_FLOOR = 1e-6
const DEFAULT_FORECAST = "narval"
const DEFAULT_CELLS = "16,16,16"
const DEFAULT_SCALE = "1//32"
const DEFAULT_RANK = 4000
const DEFAULT_OVERSAMPLES = 50

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

"`--cells 16,16,16` -> `(16, 16, 16)`."
function parse_cells(s::AbstractString)
    parts = split(strip(s, ['(', ')', ' ']), ',')
    length(parts) == 3 || error("--cells expects three comma-separated integers, got '$s'")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])), parse(Int, strip(parts[3])))
end

"`--scale 1//32` -> `1//32`. A bare integer is allowed and means `n//1`."
function parse_scale(s::AbstractString)
    parts = split(strip(s), "//")
    length(parts) == 1 && return parse(Int, strip(parts[1])) // 1
    length(parts) == 2 || error("--scale expects a rational like 1//32, got '$s'")
    return parse(Int, strip(parts[1])) // parse(Int, strip(parts[2]))
end

"""
    rank_from_roots(roots) -> Int

The sketch rank the forecast should assume, read off the `k<N>` component
`run_study.sh` puts in both tree names (`.../power_iter_study/k4000`), since
`file_prefix` does not encode the rank and nothing in the JLD does either. Falls
back to `DEFAULT_RANK`, and `--rank` overrides both.
"""
function rank_from_roots(roots::AbstractVector{<:AbstractString})
    for root in roots
        for part in reverse(splitpath(abspath(root)))
            m = match(r"^k(\d+)$", part)
            m === nothing || return parse(Int, m.captures[1])
        end
    end
    return DEFAULT_RANK
end

"""
    with_csv(f, path, header, wanted)

`f(io)` on a freshly written `path` carrying `header`, or `f(devnull)` when
`wanted` is false, so that a run with nothing to say in one of the two CSVs leaves
no empty file behind.
"""
function with_csv(f, path::AbstractString, header::AbstractString, wanted::Bool)
    wanted || return f(devnull)
    return open(path, "w") do io
        println(io, header)
        f(io)
    end
end

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

"""
    jld_kinds(path) -> (has_bounds, has_rsvd)

Which analysis a JLD can feed, by the keys it holds rather than by which tree it
was found in: `bounds_dual_basis` for the bounds comparison, `UR_asym/D` for the
spectrum one. Both false is a real state and not an error -- `compute_bounds`
truncates its output file and writes the ordering before the bounds loop starts,
so a job killed part-way leaves a JLD with neither.

An unreadable file is reported and treated as holding neither, so one corrupt
output cannot take the whole run down.
"""
function jld_kinds(path::AbstractString)
    try
        return jldopen(path, "r") do io
            (haskey(io, "bounds_dual_basis"), haskey(io, "UR_asym/D"))
        end
    catch err
        println(stderr, "  cannot read $(path): $(sprint(showerror, err))")
        return (false, false)
    end
end

"""
    find_runs(roots) -> Vector{NamedTuple}

One entry per (q directory, JLD) pair found under any of `roots`. `q` is the power
iteration count, `replicate` marks the `q08r`-style second run of the reference,
and `sep` is the `<num>ss<den>` token out of the filename, which is how the
separations are told apart without rebuilding `file_prefix`. `has_bounds` and
`has_rsvd` come from [`jld_kinds`](@ref).

Several roots are scanned as one, deduplicated by path, so passing the project and
scratch trees together is the same as having found both under one root.
"""
function find_runs(roots::AbstractVector{<:AbstractString})
    runs = NamedTuple[]
    seen = Set{String}()
    for root in roots
        isdir(root) || error("no such directory: $root")
        for entry in sort(readdir(root))
            m = match(r"^q(\d+)(r?)$", entry)
            m === nothing && continue
            q = parse(Int, m.captures[1])
            replicate = m.captures[2] == "r"
            dir = joinpath(root, entry)
            isdir(dir) || continue
            for f in sort(readdir(dir))
                endswith(f, ".jld") || continue
                sm = match(r"__(-?\d+ss\d+)__", f)
                sm === nothing && continue
                path = joinpath(dir, f)
                path in seen && continue
                push!(seen, path)
                has_bounds, has_rsvd = jld_kinds(path)
                push!(runs, (q=q, replicate=replicate, sep=sm.captures[1],
                             label=replicate ? "$(q)r" : string(q),
                             path=path, root=root,
                             has_bounds=has_bounds, has_rsvd=has_rsvd))
            end
        end
    end
    return runs
end

find_runs(root::AbstractString) = find_runs([root])

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
    gamma_kept_count(Γ, num_pos, gamma_rtol) -> Int

`_gamma_kept_count` from `src/bounds.jl`, on the same input `load_bounds_inputs`
gives it: `Γ` sorted descending, `num_pos` the saved positive count, keep the
entries at or above `gamma_rtol * Γ[1]`. That count is the `m` the bounds job runs
at, and it is not the positive count -- the cut is what collapses it.

Duplicated rather than imported so this script keeps starting in a second without
loading CUDA, exactly as `bench/pick_bounds_points.jl` duplicates it. If the cut in
`load_bounds_inputs` ever changes, change both.
"""
function gamma_kept_count(Γ::AbstractVector, num_pos::Integer, gamma_rtol::Real)
    isempty(Γ) && return 0
    Γ[1] > 0 || return 0
    n = clamp(Int(num_pos), 0, length(Γ))
    n == 0 && return 0
    return count(>=(gamma_rtol * Γ[1]), view(Γ, 1:n))
end

"""
    read_rsvd(path, gamma_rtol) -> NamedTuple or nothing

The `UR_asym/` group, or `nothing` if the RSVD never wrote it. `Γ` is sorted
descending here even though `_save_ur_asym_components` says it saves it that way,
because `load_bounds_inputs` sorts it again before using it and this has to agree
with `load_bounds_inputs`, not with the writer.

`num_pos` is the file's own count; `num_pos_counted` is what the saved values
actually say, and the two disagreeing is what `load_bounds_inputs` refuses to run
on, so it is reported rather than reconciled.
"""
function read_rsvd(path::AbstractString, gamma_rtol::Real)
    return jldopen(path, "r") do io
        haskey(io, "UR_asym/D") || return nothing
        Γ = sort(Array{Float64}(real.(Array(io["UR_asym/D"]))); rev=true)
        isempty(Γ) && return nothing
        counted = count(>(0.0), Γ)
        saved = haskey(io, "UR_asym/num_pos") ? Int(io["UR_asym/num_pos"]) : counted
        saved = clamp(saved, 0, length(Γ))
        return (Γ=Γ, num_pos=saved, num_pos_counted=counted, total=length(Γ),
                kept=gamma_kept_count(Γ, saved, gamma_rtol),
                n_rs=haskey(io, "RS/D") ? length(io["RS/D"]) : 0)
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
    isdir(root) || return out
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
    spectrum_devs(a, b, n) -> (max, med, n)

`spectrum_deviation`'s two-number form: the largest *and* the median relative
deviation of the leading `n` entries, plus how many of them could actually be
compared. Both are wanted in spectrum mode because they say different things about
the same block -- a `max` far above the `med` is the bottom of the kept set, where
two runs are ordering their own noise and index-matching has stopped meaning
anything, rather than a spectrum that moved.
"""
function spectrum_devs(a::AbstractVector, b::AbstractVector, n::Int)
    n = min(n, length(a), length(b))
    n <= 0 && return (max=NaN, med=NaN, n=0)
    rels = Float64[]
    for i in 1:n
        ai, bi = a[i], b[i]
        (isfinite(ai) && isfinite(bi)) || continue
        denom = Base.max(abs(ai), abs(bi))
        denom == 0 && continue
        push!(rels, abs(ai - bi) / denom)
    end
    isempty(rels) && return (max=NaN, med=NaN, n=0)
    return (max=maximum(rels), med=median(rels), n=length(rels))
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

    r = noise_ratio_checks((("trace", prev.trace_abs, noise.trace_abs),
                            ("chan_max", prev.chan_max, noise.chan_max)), th.conv_factor)
    r.assessed == 0 && return :unassessable, r.msg
    return (r.converged ? :ok : :not_converged), r.msg
end

"""
    noise_ratio_checks(checks, conv_factor) -> (converged, assessed, msg)

The convergence gate's arithmetic, over `(name, deviation, noise_floor)` triples.
Every triple that can be assessed has to come within `conv_factor` of its floor.

Shared by both modes so that "within 2x the sketch-noise floor" means the same
thing whether the quantity is a bounds trace or an eigenvalue deviation. Only the
triples differ.

A zero noise floor cannot be scaled, so that quantity is skipped unless the
deviation is zero too, in which case it agrees exactly and counts as assessed and
passing. `assessed == 0` means nothing was measurable, which the caller has to
report as unassessable rather than as either answer.
"""
function noise_ratio_checks(checks, conv_factor::Float64)
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
        ratio <= conv_factor || (converged = false)
        push!(verdicts, @sprintf("%s %.2e vs noise %.2e (%.1fx)", name, dev, nz, ratio))
    end
    return (converged=converged, assessed=assessed, msg=join(verdicts, ", "))
end

"""
    spectrum_convergence(prev, noise, conv_factor) -> (status, message)

[`convergence`](@ref) for spectrum mode: the same gate on `dev_max` and `dev_med`
instead of the trace and the worst gated channel. There is no reporting floor on an
eigenvalue, so there is no `:moot` here; the statuses are `:ok`, `:not_converged`
and `:unassessable`.
"""
function spectrum_convergence(prev, noise, conv_factor::Float64)
    (prev === nothing || noise === nothing) &&
        return :unassessable, "no second-largest q, or no replicate, to compare"
    r = noise_ratio_checks((("dev_max", prev.max, noise.max),
                            ("dev_med", prev.med, noise.med)), conv_factor)
    r.assessed == 0 && return :unassessable, r.msg
    return (r.converged ? :ok : :not_converged), r.msg
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

# --------------------------------------------------------------------------- #
# The bounds-cost forecast
# --------------------------------------------------------------------------- #

const COST_MODEL_PATH = normpath(joinpath(@__DIR__, "..", "cost_model.jl"))

#=
Included at load time rather than on demand, for two reasons. Julia 1.12's world
age rules refuse a call to a function whose method was defined by an `include` in
the same frame, so a lazy `include` inside `load_forecast` cannot then call
`load_coefficients!`; and `bench/cost_model.jl` is dependency-free by design, so
including it costs a parse and nothing else. `nothing` if it is not there, which is
the case for a copy of this script sitting outside the repo.
=#
const COST_MODEL = try
    isfile(COST_MODEL_PATH) ? include(COST_MODEL_PATH) : nothing
catch err
    println(stderr, "could not include $(COST_MODEL_PATH): $(sprint(showerror, err))")
    nothing
end

"""
    load_forecast(cluster, cells, scale, rank, oversamples) -> NamedTuple or nothing

The coefficient set and the fixed part of the point the forecast is for.
`nothing` -- with a reason printed -- when the cluster is `none`, `bench/cost_model.jl`
was not there to include, or the coefficients fail to load, since a missing forecast
must not stop the tables that do not need it.

`bounds_m_mode` is forced to `"fraction"` on the loaded coefficients. In
`"truncated"` mode `bounds_m` caps `num_pos` by a separation-keyed power law that
*predicts* the `--gamma-rtol` cut, and the whole point here is that the cut has
been measured: the cap would throw the measurement away and, at a separation where
the law is conservative, replace it with a smaller number. With the mode off,
`bounds_m` is `min(num_pos, N_u)`, i.e. the measured `kept`.
"""
function load_forecast(cluster::AbstractString, cells::NTuple{3,Int},
                       scale::Rational{Int}, rank::Int, oversamples::Int)
    cluster == "none" && return nothing
    if COST_MODEL === nothing
        println("  NOTE: no cost model at $(COST_MODEL_PATH); the bounds-cost forecast is skipped")
        return nothing
    end
    CM = COST_MODEL
    local coeffs
    try
        CM.load_coefficients!(dirname(COST_MODEL_PATH))
        coeffs = CM.coefficients_for(cluster)
        coeffs.calibrated || println("  NOTE: the '$cluster' coefficients are not " *
                                     "calibrated; the forecast is the analytic guess")
        fields = fieldnames(typeof(coeffs))
        as_nt = NamedTuple{fields}(map(f -> getfield(coeffs, f), fields))
        coeffs = CM.Coefficients(; merge(as_nt, (bounds_m_mode="fraction",))...)
    catch err
        println("  NOTE: could not load the cost model ($(sprint(showerror, err))); " *
                "the bounds-cost forecast is skipped")
        return nothing
    end
    return (mod=CM, coeffs=coeffs, cluster=cluster, cells=cells, scale=scale,
            rank=rank, oversamples=oversamples)
end

"""
    bounds_forecast(fc, sep, m) -> NamedTuple or nothing

`predict(ComputeBounds, ...)` at the measured kept `m` for this separation, padded
and raw. `nothing` if there is no forecast context.

`rank` only enters through `effective_num_pos`'s `min(rank, num_pos)` clamp, so it
is held at least as large as `m`: a kept count above the nominal rank would
otherwise be silently truncated to it.
"""
function bounds_forecast(fc, sep::AbstractString, m::Int)
    (fc === nothing || m <= 0) && return nothing
    CM = fc.mod
    parts = split(sep, "ss")
    separation = parse(Int, parts[1]) // parse(Int, parts[2])
    pt = CM.SRPoint(fc.cells, fc.cells; scale=fc.scale, separation=separation,
                    rank=max(fc.rank, m), oversamples=fc.oversamples,
                    num_pos=m, fresh_preload=false)
    padded = CM.predict(CM.ComputeBounds, pt, fc.coeffs; pad=true)
    raw = CM.predict(CM.ComputeBounds, pt, fc.coeffs; pad=false)
    return (padded_s=padded.time_s, raw_s=raw.time_s, mode=padded.mode,
            host_bytes=padded.host_bytes, vram_floor_bytes=padded.vram_floor_bytes)
end

# --------------------------------------------------------------------------- #
# Report: spectrum mode
# --------------------------------------------------------------------------- #

const RSVD_CSV_HEADER = "separation,q,replicate,operator_passes,rsvd_s,num_pos," *
                        "num_pos_counted,total,kept,kept_ref,n_compared," *
                        "eig_max_rel,eig_median_rel,noise_max_rel,noise_median_rel," *
                        "match_threshold,matches,bounds_forecast_s,bounds_forecast_raw_s"

"""
    report_rsvd_separation(io_csv, sep, runs, timings, th, fc, tally) -> nothing

The spectrum-mode table for one separation, plus the bounds-cost forecast. `tally`
accumulates the forecast so the caller can total it over separations.
"""
function report_rsvd_separation(io_csv, sep, runs, timings, th, fc, tally)
    println("\n", "-"^104)
    println("separation $(sep)  (x-separation $(sep_rational(sep)) lambda) -- RSVD spectrum")

    ref_idx = findfirst(r -> r.q == th.ref && !r.replicate, runs)
    if ref_idx === nothing
        qs = sort(unique(r.q for r in runs))
        println("  no q=$(th.ref) reference RSVD output here; the q present are " *
                "$(join(qs, ", ")). Pass --ref $(maximum(qs)) to make the largest of " *
                "them the reference.")
        return nothing
    end
    ref = read_rsvd(runs[ref_idx].path, th.gamma_rtol)
    if ref === nothing
        println("  the q=$(th.ref) reference has no readable UR_asym/D: its RSVD job did not finish")
        return nothing
    end

    ordered = sort(runs; by=r -> (r.q, r.replicate))

    # The sketch-noise floor first: every row below is read against it, and the
    # match threshold is derived from it.
    noise = nothing
    rep_idx = findfirst(r -> r.q == th.ref && r.replicate, runs)
    if rep_idx !== nothing
        rep = read_rsvd(runs[rep_idx].path, th.gamma_rtol)
        rep === nothing ||
            (noise = spectrum_devs(rep.Γ, ref.Γ, min(rep.kept, ref.kept)))
    end
    floor_max = noise === nothing ? NaN : noise.max
    threshold = max(isfinite(floor_max) ? th.noise_factor * floor_max : 0.0, th.match_floor)

    @printf("\n  Gamma = the Asym(G0_ur) eigenvalues, sorted descending; kept = the gamma_rtol %.1e cut,\n",
            th.gamma_rtol)
    println("  which is the m the bounds job would run at. dev_* are over the mutually kept set.")
    @printf("  %-5s %7s %7s %7s %7s %11s %11s %10s\n",
            "q", "npos", "kept", "total", "n_cmp", "dev_max", "dev_med", "rsvd_s")

    matches = Dict{String,Bool}()
    devs = Dict{String,Any}()
    kepts = Dict{String,Int}()
    for r in ordered
        data = read_rsvd(r.path, th.gamma_rtol)
        if data === nothing
            @printf("  %-5s (no readable UR_asym/D: the RSVD job did not finish)\n", r.label)
            continue
        end
        n = min(data.kept, ref.kept)
        d = spectrum_devs(data.Γ, ref.Γ, n)
        devs[r.label] = d
        kepts[r.label] = data.kept
        matches[r.label] = isfinite(d.max) && d.max <= threshold
        rsvd_s = get(timings, (sep, r.label, "rsvd"), NaN)
        @printf("  %-5s %7d %7d %7d %7d %11.3e %11.3e %10.0f\n",
                r.label, data.num_pos, data.kept, data.total, d.n, d.max, d.med, rsvd_s)
        if data.num_pos != data.num_pos_counted
            @printf("        NOTE: UR_asym/num_pos says %d but %d of the saved values are positive; load_bounds_inputs refuses to run on that\n",
                    data.num_pos, data.num_pos_counted)
        end
        fq = bounds_forecast(fc, sep, data.kept)
        println(io_csv, join(string.([
            sep, r.q, r.replicate, 2 * r.q + 2, rsvd_s, data.num_pos,
            data.num_pos_counted, data.total, data.kept, ref.kept, d.n,
            d.max, d.med, floor_max, noise === nothing ? NaN : noise.med,
            threshold, matches[r.label],
            fq === nothing ? "" : fq.padded_s, fq === nothing ? "" : fq.raw_s]), ','))
    end

    @printf("\n  reference q=%d: %d positive of %d saved eigenvalues, %d kept\n",
            th.ref, ref.num_pos, ref.total, ref.kept)
    odd = [r.label for r in ordered if get(kepts, r.label, ref.kept) != ref.kept]
    isempty(odd) ||
        println("  kept differs from the reference at q = $(join(odd, ", ")); a different m " *
                "is a structural difference, not a numerical one, and it resizes the whole " *
                "bounds computation")
    if noise === nothing
        println("  no q=$(th.ref) replicate: the sketch-noise floor is unmeasured, so the " *
                "match threshold below falls back to the bare $(th.match_floor)")
    else
        @printf("  sketch-noise floor (q=%d replicate): dev_max %.3e, dev_med %.3e over %d index(es)\n",
                th.ref, noise.max, noise.med, noise.n)
        println("  a row at or under those numbers is not distinguishable from the reference")
    end
    println("  index-matching only means something above the floor: at the bottom of the kept")
    println("  block two runs order their own noise, so a dev_max there is not a disagreement")

    prev_q = maximum([r.q for r in runs if !r.replicate && r.q < th.ref]; init=0)
    prev_d = prev_q > 0 ? get(devs, string(prev_q), nothing) : nothing
    status, why = spectrum_convergence(prev_d, noise, th.conv_factor)
    @printf("\n  convergence of the q=%d reference (q=%d against it, limit %.1fx the noise floor): %s\n",
            th.ref, prev_q, th.conv_factor, why)

    if status === :not_converged
        println("  not converged at q=$(th.ref) -- reference unreliable, raise QS")
        println("  no provisional verdict is issued off an unconverged reference")
    else
        status === :unassessable &&
            println("  the reference's own convergence could not be assessed, so read the " *
                    "provisional verdict as even weaker than it already is")
        candidates = sort([r.q for r in runs if !r.replicate && get(matches, string(r.q), false)])
        if isempty(candidates)
            @printf("  PROVISIONAL VERDICT %s: no q matches the reference to %.2e. Keep q=%d.\n",
                    sep, threshold, th.ref)
        else
            smallest = first(candidates)
            gaps = [r.q for r in runs if !r.replicate && r.q > smallest &&
                                        haskey(matches, string(r.q)) && !matches[string(r.q)]]
            @printf("  PROVISIONAL VERDICT %s: smallest q whose kept eigenvalues match the reference = %d\n",
                    sep, smallest)
            if isfinite(floor_max)
                @printf("            (threshold %.2e = max(%.0fx the noise floor %.2e, %.1e))\n",
                        threshold, th.noise_factor, floor_max, th.match_floor)
            else
                @printf("            (threshold %.2e, the bare --match-floor: the noise floor is unmeasured)\n",
                        threshold)
            end
            isempty(gaps) ||
                println("            NOTE: q = $(join(gaps, ", ")) failed above it, so the " *
                        "match is not monotone in q; treat $(smallest) as luck")
        end
        println("            SPECTRUM-LEVEL ONLY. q reaches the bounds through the RSVD basis,")
        println("            and two runs can agree on Gamma while their eigenvectors span")
        println("            different subspaces. The binding verdict needs the bounds stage.")
    end

    fc === nothing && return nothing
    println("\n  bounds-cost forecast at the measured kept m, $(fc.cluster)-equivalent coefficients")
    println("  (molering's own bounds coefficients are known-garbage in the conservative")
    println("   direction -- one 6.7 ms sync_latency carries the whole estimate; see the README)")
    @printf("  %-5s %7s %12s %12s %12s %10s\n",
            "q", "kept m", "padded_h", "raw_h", "path", "vram_GiB")
    total_padded = 0.0
    n_runs = 0
    for r in ordered
        haskey(kepts, r.label) || continue
        f = bounds_forecast(fc, sep, kepts[r.label])
        f === nothing && continue
        total_padded += f.padded_s
        n_runs += 1
        @printf("  %-5s %7d %12.2f %12.2f %12s %10.1f\n",
                r.label, kepts[r.label], f.padded_s / 3600, f.raw_s / 3600,
                string(f.mode), f.vram_floor_bytes / 2^30)
    end
    if n_runs > 0
        @printf("  the bounds half of this separation is %d run(s), %.1f h padded in total\n",
                n_runs, total_padded / 3600)
        tally[] = (tally[][1] + total_padded, tally[][2] + n_runs)
    end
    return nothing
end

function main(argv::Vector{String})
    opts = parse_cli(argv)
    root = get(opts, "root", "")
    rsvd_root = get(opts, "rsvd-root", "")
    (isempty(root) && isempty(rsvd_root)) &&
        error("--root is required (or --rsvd-root on its own, for a scratch tree that " *
              "holds only RSVD output)")
    roots = unique(filter(!isempty, [root, rsvd_root]))
    th = (ref=parse(Int, get(opts, "ref", string(DEFAULT_REF))),
          conv_factor=parse(Float64, get(opts, "conv-factor", string(DEFAULT_CONV_FACTOR))),
          trace_floor=parse(Float64, get(opts, "trace-floor", string(DEFAULT_TRACE_FLOOR))),
          trace_abs=parse(Float64, get(opts, "trace-abs", string(DEFAULT_TRACE_ABS))),
          trace_rtol=parse(Float64, get(opts, "trace-rtol", string(DEFAULT_TRACE_RTOL))),
          chan_floor=parse(Float64, get(opts, "chan-floor", string(DEFAULT_CHAN_FLOOR))),
          chan_rtol=parse(Float64, get(opts, "chan-rtol", string(DEFAULT_CHAN_RTOL))),
          gamma_rtol=parse(Float64, get(opts, "gamma-rtol", string(DEFAULT_GAMMA_RTOL))),
          noise_factor=parse(Float64, get(opts, "noise-factor", string(DEFAULT_NOISE_FACTOR))),
          match_floor=parse(Float64, get(opts, "match-floor", string(DEFAULT_MATCH_FLOOR))))
    out = get(opts, "out", joinpath(@__DIR__, "power_iter_summary.csv"))
    rsvd_out = get(opts, "rsvd-out", joinpath(@__DIR__, "power_iter_rsvd_summary.csv"))

    runs = find_runs(roots)
    isempty(runs) && error("no q*/*.jld under $(join(roots, " or ")); has the study run?")
    timings = Dict{Tuple{String,String,String},Float64}()
    for r in roots
        merge!(timings, read_timings(r))
    end

    # A pure-RSVD file is the only thing the bounds report has nothing to say about.
    # Everything else goes to it, including the neither-key case, so that a bounds
    # job killed part-way still reports itself exactly as it did before.
    bounds_runs = filter(r -> r.has_bounds || !r.has_rsvd, runs)
    rsvd_runs = filter(r -> r.has_rsvd, runs)

    println("="^104)
    println("Power-iteration quality study")
    isempty(root) || println("  root      $root")
    isempty(rsvd_root) || println("  rsvd root $rsvd_root")
    println("  reference q = $(th.ref) (checked for convergence against the next q down, ")
    @printf("                limit %.1fx the q=%d replicate's sketch-noise floor)\n",
            th.conv_factor, th.ref)
    if !isempty(bounds_runs)
        @printf("  trace: reported only above %.1e; agreement to %.1f%% or %.1e absolute\n",
                th.trace_floor, 100 * th.trace_rtol, th.trace_abs)
        @printf("  channels: gated at >= %.1e absolute, to %.1f%% relative\n",
                th.chan_floor, 100 * th.chan_rtol)
        println("  bounds_dual_basis is on the sigma_n(P_rs) scale, so the floors apply to it directly")
    end
    if !isempty(rsvd_runs)
        @printf("  spectrum: %d RSVD output(s), kept at gamma_rtol %.1e, matched to max(%.0fx the noise floor, %.1e)\n",
                length(rsvd_runs), th.gamma_rtol, th.noise_factor, th.match_floor)
        println("            those verdicts are PROVISIONAL: the spectrum is not what the study reports")
    end
    isempty(timings) && println("  NOTE: no timings_gpu*.csv under the root; the wall-time columns will be blank")

    fc = isempty(rsvd_runs) ? nothing :
         load_forecast(get(opts, "forecast", DEFAULT_FORECAST),
                       parse_cells(get(opts, "cells", DEFAULT_CELLS)),
                       parse_scale(get(opts, "scale", DEFAULT_SCALE)),
                       parse(Int, get(opts, "rank", string(rank_from_roots(roots)))),
                       parse(Int, get(opts, "oversamples", string(DEFAULT_OVERSAMPLES))))
    fc === nothing || @printf("  forecast: bounds cost at the measured kept m for (%s) at scale %s, rank %d, on %s-equivalent coefficients\n",
                              join(fc.cells, ","), fc.scale, fc.rank, fc.cluster)

    tally = Ref((0.0, 0))
    with_csv(out, CSV_HEADER, !isempty(bounds_runs)) do io_csv
        with_csv(rsvd_out, RSVD_CSV_HEADER, !isempty(rsvd_runs)) do io_rsvd
            for sep in sort(unique(r.sep for r in runs); by=sep_value)
                b = filter(r -> r.sep == sep, bounds_runs)
                v = filter(r -> r.sep == sep, rsvd_runs)
                isempty(b) || report_separation(io_csv, sep, b, timings, th)
                isempty(v) || report_rsvd_separation(io_rsvd, sep, v, timings, th, fc, tally)
            end
        end
    end

    println("\n", "="^104)
    total_s, total_runs = tally[]
    if total_runs > 0
        @printf("bounds-cost forecast over every RSVD output found: %d run(s), %.1f h padded,\n",
                total_runs, total_s / 3600)
        @printf("which is %.1f h of wall time over two GPUs. %s-equivalent coefficients: read it as\n",
                total_s / 7200, fc === nothing ? "" : fc.cluster)
        println("a lower bound with the A6000-vs-A100 difference on top.")
    end
    isempty(bounds_runs) || println("Wrote $out")
    isempty(rsvd_runs) || println("Wrote $rsvd_out")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
