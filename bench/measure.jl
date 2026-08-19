"""
    measure.jl

Measurement utilities for the cost-model calibration harness.

Everything here is dependency-free apart from CUDA (which is already a project
dependency). The only job of this file is to answer three questions about a
piece of work:

  1. How long did it take (wall clock, after compilation)?
  2. What was the peak host resident set size?
  3. What was the peak device (VRAM) usage?

and then to append the answer to a CSV file with a stable schema so that
`bench/fit.jl` can consume rows produced on different clusters at different
times.

Peak host RSS is read from the OS rather than from Julia's allocator, because
the thing we need to predict is what SLURM's cgroup accounting sees. On Linux
`/proc/self/status:VmHWM` is the high-water mark of the resident set and is
exactly the quantity that gets a job killed for exceeding `--mem`. `Sys.maxrss()`
(getrusage) is used as a cross-check and as the fallback on non-Linux hosts.

Peak VRAM is sampled rather than queried, because CUDA.jl's memory pool hides
the true device footprint and because cuFFT/cuSOLVER workspaces are allocated
outside the pool. A watcher task polls `CUDA.memory_info()` and records the
minimum free memory seen. On a dedicated GPU (or a MIG slice) that is the
number that decides whether the job fits.
"""

using Dates
using Printf

# --------------------------------------------------------------------------- #
# Host memory
# --------------------------------------------------------------------------- #

"""
    peak_rss_bytes() -> Int

Peak resident set size of this process in bytes, as the OS reports it.
Prefers `/proc/self/status:VmHWM` (Linux, exactly what cgroups account) and
falls back to `Sys.maxrss()` elsewhere.
"""
function peak_rss_bytes()
    if Sys.islinux() && isfile("/proc/self/status")
        for line in eachline("/proc/self/status")
            if startswith(line, "VmHWM:")
                fields = split(line)
                length(fields) >= 2 || continue
                kb = tryparse(Int, fields[2])
                kb === nothing || return kb * 1024
            end
        end
    end
    return Int(Sys.maxrss())
end

"""
    reset_rss_highwater()

Ask the kernel to reset `VmHWM` to the current RSS. This lets a single process
measure several points without the first (largest) one poisoning the rest.
Silently does nothing where it is unsupported; callers must not rely on it, and
`bench/point.jl` runs one point per process precisely so that it does not have
to.
"""
function reset_rss_highwater()
    try
        if Sys.islinux() && isfile("/proc/self/clear_refs")
            open("/proc/self/clear_refs", "w") do io
                write(io, "5\n") # 5 = reset VmHWM to current RSS
            end
            return true
        end
    catch
    end
    return false
end

# --------------------------------------------------------------------------- #
# Device memory
# --------------------------------------------------------------------------- #

"""
    VramWatcher

Background sampler for device memory. `start_vram_watcher` returns one of these
and `stop_vram_watcher!` returns the peak number of bytes in use on the device
over the watched interval (relative to whatever was already in use when the
watcher started, plus that baseline -- i.e. the absolute device footprint).
"""
#=
Three numbers, because they answer different questions and confusing them makes
the memory model unfittable:

  * `min_free`      -> device bytes unavailable at the worst moment, from the
                       driver's point of view. Includes the CUDA context and any
                       other tenant. This is what decides whether the job fits.
  * `max_pool_used` -> bytes live in CUDA.jl's pool. This is *demand*, and the
                       only one of the three that scales with the problem size.
  * `max_pool_reserved` -> bytes the pool has claimed from the driver. The pool
                       does not shrink, so transient allocation churn (the bounds
                       job allocates a few vectors per basis column, thousands of
                       times) inflates this far above demand.

Fitting a size model against reserved bytes produced slopes 4-33x the analytic
count, anti-correlated with size -- the signature of a size-independent overhead
being attributed to a size-dependent term.
=#
mutable struct VramWatcher
    task::Union{Nothing,Task}
    running::Threads.Atomic{Bool}
    min_free::Threads.Atomic{Int}
    max_pool_used::Threads.Atomic{Int}
    max_pool_reserved::Threads.Atomic{Int}
    total::Int
    baseline_used::Int
    interval_s::Float64
end

const _NO_VRAM_WATCHER = VramWatcher(nothing, Threads.Atomic{Bool}(false),
                                     Threads.Atomic{Int}(typemax(Int)),
                                     Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                                     0, 0, 0.0)

"""
    start_vram_watcher(; interval_s=0.05, enabled=true) -> VramWatcher

Begin sampling `CUDA.memory_info()`. Pass `enabled=false` for CPU-only points
so that callers do not have to branch.
"""
function start_vram_watcher(; interval_s::Real=0.05, enabled::Bool=true)
    enabled || return _NO_VRAM_WATCHER
    free, total = CUDA.memory_info()
    watcher = VramWatcher(nothing, Threads.Atomic{Bool}(true),
                          Threads.Atomic{Int}(free), Threads.Atomic{Int}(0),
                          Threads.Atomic{Int}(0), Int(total), Int(total - free),
                          Float64(interval_s))
    watcher.task = Threads.@spawn begin
        while watcher.running[]
            try
                f, _ = CUDA.memory_info()
                f < watcher.min_free[] && (watcher.min_free[] = Int(f))
                # Both return `missing` on devices without a stream-ordered pool.
                used = CUDA.used_memory()
                used isa Integer && used > watcher.max_pool_used[] &&
                    (watcher.max_pool_used[] = Int(used))
                reserved = CUDA.cached_memory()
                reserved isa Integer && reserved > watcher.max_pool_reserved[] &&
                    (watcher.max_pool_reserved[] = Int(reserved))
            catch
                # A transient CUDA error must not take down the measurement.
            end
            sleep(watcher.interval_s)
        end
    end
    return watcher
end

"""
    stop_vram_watcher!(watcher) -> (peak_used_bytes, peak_used_above_baseline_bytes)

Stop sampling. Returns the absolute device high-water mark and the high-water
mark net of whatever was already resident when the watcher started (CUDA
context, other tenants).
"""
function stop_vram_watcher!(watcher::VramWatcher)
    watcher.task === nothing && return (peak=0, delta=0, live=0, reserved=0)
    watcher.running[] = false
    try
        wait(watcher.task)
    catch
    end
    peak_used = watcher.total - watcher.min_free[]
    return (peak=peak_used, delta=max(0, peak_used - watcher.baseline_used),
            live=watcher.max_pool_used[], reserved=watcher.max_pool_reserved[])
end

# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

"""
    timed(f; device_sync=false) -> (value, seconds)

Wall-clock time `f()`. With `device_sync=true` a `CUDA.synchronize()` runs
before and after so that asynchronous kernel launches are not timed away.
"""
function timed(f; device_sync::Bool=false)
    device_sync && CUDA.synchronize()
    t0 = time_ns()
    value = f()
    device_sync && CUDA.synchronize()
    return value, (time_ns() - t0) / 1e9
end

"""
    repeat_timed(f, reps; device_sync=false, warmup=1) -> (min_s, median_s, mean_s)

Time `f()` `reps` times after `warmup` untimed calls. Used for the cheap
primitives (Green matvecs, dense factorizations) where a single sample is
dominated by launch jitter. The minimum is the honest estimate of the cost of
the work; the mean is what a long run of them will actually take, so both are
recorded.
"""
function repeat_timed(f, reps::Int; device_sync::Bool=false, warmup::Int=1)
    for _ in 1:warmup
        f()
    end
    device_sync && CUDA.synchronize()
    samples = Float64[]
    for _ in 1:reps
        _, dt = timed(f; device_sync=device_sync)
        push!(samples, dt)
    end
    sort!(samples)
    mid = length(samples) == 0 ? 0.0 :
          (isodd(length(samples)) ? samples[(length(samples) + 1) ÷ 2] :
           (samples[length(samples) ÷ 2] + samples[length(samples) ÷ 2 + 1]) / 2)
    return (isempty(samples) ? 0.0 : samples[1], mid,
            isempty(samples) ? 0.0 : sum(samples) / length(samples))
end

# --------------------------------------------------------------------------- #
# CSV output
# --------------------------------------------------------------------------- #

#=
One flat schema for every kind of point. Columns that do not apply to a given
kind are left empty rather than zero, so that the fitter can tell "not
measured" from "measured as zero". Append-only: never reorder or reuse a
column name, only add to the end.
=#
const CSV_COLUMNS = [
    "timestamp",        # when the row was produced (ISO 8601)
    "cluster",          # fir / narval / molering / <hostname>
    "kind",             # see bench/point.jl: g0_self, g0_ext, matvec_self, ...
    "device",           # cpu / gpu
    "gpu_name",         # e.g. NVIDIA A100-SXM4-40GB, or empty on cpu points
    "threads",          # Julia thread count
    "n_x", "n_y", "n_z",# cells of one body
    "n_cells",          # prod of the above
    "scale_num", "scale_den",   # cell size in wavelengths, as a rational
    "sep_num", "sep_den",       # sender-receiver surface separation, in wavelengths
    "contact",          # 1 if the bodies touch (separation == 0), else 0
    "rank",             # RSVD target rank k
    "oversamples",      # p
    "power_iters",      # q
    "sketch_width",     # c = k + p, after clamping to the operator size
    "num_pos",          # positive-eigenvalue count used by the bounds kernel
    "dense_m",          # leading dimension of the dense-primitive point
    "dense_c",          # trailing dimension of the dense-primitive point
    "reps",             # timing repetitions (blank for one-shot points)
    "time_s",           # headline wall time: min for repeated points, total otherwise
    "time_median_s",
    "time_mean_s",
    "peak_rss_bytes",
    "baseline_rss_bytes",   # RSS after package load, before any work
    "peak_vram_bytes",
    "peak_vram_delta_bytes",
    "baseline_vram_bytes",  # device bytes in use before any work
    "peak_vram_live_bytes",     # high-water of live pool allocations (demand)
    "peak_vram_reserved_bytes", # high-water of pool-reserved backing (never shrinks)
    "device_total_bytes",       # device capacity, so censored measurements are detectable
    "bytes_written",    # serialized output size where relevant
    "startup_s",        # process start to first measured work (package load, CUDA init)
    "extra",            # free-form key=value;key=value notes
]

"""
    csv_row(; kwargs...) -> Dict{String,Any}

Build a row with every column present (empty where unset).
"""
function csv_row(; kwargs...)
    row = Dict{String,Any}(col => "" for col in CSV_COLUMNS)
    for (k, v) in kwargs
        key = String(k)
        key in CSV_COLUMNS || error("Unknown CSV column '$key'. Add it to CSV_COLUMNS.")
        row[key] = v
    end
    row["timestamp"] = string(now())
    return row
end

_csv_escape(x) = begin
    s = x isa AbstractFloat ? @sprintf("%.10g", x) : string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return '"' * replace(s, '"' => "\"\"") * '"'
    end
    return s
end

"""
    append_csv_row(path, row)

Append `row` to `path`, writing the header first if the file is new. Opened in
append mode per call so that a crashed or killed point does not lose the rows
that came before it.
"""
function append_csv_row(path::AbstractString, row::AbstractDict)
    mkpath(dirname(abspath(path)))
    write_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        write_header && println(io, join(CSV_COLUMNS, ","))
        println(io, join((_csv_escape(row[col]) for col in CSV_COLUMNS), ","))
    end
    return path
end

"""
    panel_bus_row(; cluster, gpu_name="", threads=Threads.nthreads(),
                  device_total=0, time_s, extras) -> Dict{String,Any}

One `kind="panel_bus"` row, in the shape `bench/fit.jl`'s `fit_panel_bus` reads.

That fitter is the only consumer, and its contract is worth restating here. A row
contributes to `pcie_rate` when `extras` carries `bytes=<moved>` and `time_s` is
the seconds those bytes took, and it contributes to `overlap_factor` when `extras`
carries `overlap=<fraction in (0, 1]>`. A row may carry both, either or neither,
and a row with neither is recorded but ignored by the fit. That is how trial E1's
summary row and its informational rows (the pageable comparison, the overlap
benchmark's own sweep rate) stay in the CSV without dragging the fitted slope
around. Keep an informational byte count under a *different* key
(`bytes_pageable=`, `bytes_sweep=`) for that reason.

Not routed through `bench/point.jl`'s `emit`. `panel_bus` is the one measurement
that produces several independent samples in one process (several transfer sizes,
several compute-to-copy ratios), and `rate_through_origin` wants them as separate
rows so that a fixed per-transfer overhead cannot be amortised into the slope by a
single point. Peak RSS and the device high-water are left empty for the same
reason: they are per-process, so they would be the same number on every row and
describe none of them.
"""
function panel_bus_row(; cluster::AbstractString, gpu_name::AbstractString="",
                       threads::Integer=Threads.nthreads(), device_total::Integer=0,
                       time_s::Real, extras::AbstractVector{<:AbstractString})
    return csv_row(cluster=cluster, kind="panel_bus", device="gpu", gpu_name=gpu_name,
                   threads=threads, device_total_bytes=device_total,
                   time_s=Float64(time_s), extra=join(extras, ";"))
end

"""
    read_csv_rows(path) -> Vector{Dict{String,String}}

Minimal RFC4180-ish reader, enough for the files this harness writes. Unknown
columns are kept; missing ones come back as "".
"""
function read_csv_rows(path::AbstractString)
    rows = Dict{String,String}[]
    isfile(path) || return rows
    lines = readlines(path)
    isempty(lines) && return rows
    header = _csv_split(lines[1])
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = _csv_split(line)
        row = Dict{String,String}(col => "" for col in CSV_COLUMNS)
        for (i, col) in enumerate(header)
            row[col] = i <= length(fields) ? fields[i] : ""
        end
        push!(rows, row)
    end
    return rows
end

function _csv_split(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        ch = line[i]
        if in_quotes
            if ch == '"'
                if i < lastindex(line) && line[nextind(line, i)] == '"'
                    write(buf, '"')
                    i = nextind(line, i)
                else
                    in_quotes = false
                end
            else
                write(buf, ch)
            end
        elseif ch == '"'
            in_quotes = true
        elseif ch == ','
            push!(fields, String(take!(buf)))
        else
            write(buf, ch)
        end
        i = nextind(line, i)
    end
    push!(fields, String(take!(buf)))
    return fields
end

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #

"""
    detect_cluster() -> String

Best-effort cluster label. Compute Canada sets `CC_CLUSTER`; the group server
sets `MOLERING`. Falls back to the hostname.
"""
function detect_cluster()
    haskey(ENV, "PSC_CLUSTER") && return ENV["PSC_CLUSTER"]
    haskey(ENV, "CC_CLUSTER") && return ENV["CC_CLUSTER"]
    haskey(ENV, "MOLERING") && return "molering"
    return gethostname()
end

human_bytes(b::Real) = b >= 1024^3 ? @sprintf("%.2f GiB", b / 1024^3) :
                       b >= 1024^2 ? @sprintf("%.1f MiB", b / 1024^2) :
                       @sprintf("%.0f B", b)

# --------------------------------------------------------------------------- #
# Reading a bounds log back
# --------------------------------------------------------------------------- #

#=
Everything above measures work this process is doing. This section does the
opposite: it reconstructs a measurement from the *log* of a job that has already
finished, or has been killed.

It exists for one case. A bounds job on the backfill queue is capped at three
hours, and the run it is sampling may be longer than that. `bench/point.jl`'s
`--outer-blocks` is the first line of defence -- it bounds the work so the row gets
written -- but a job can still be cut short, and when it is, the process dies
before `emit` and the row is lost even though the log holds the numbers. Every
useful quantity is in there:

  * `bounds_from_spectrum` stamps `string(now())` into every message, so the wall
    time of any stage is the difference between two timestamps, and the per-index
    outer time is the gap between consecutive "Computing" lines. That is the same
    quantity `outer_times` records, measured from outside.
  * the truncation `@warn` in `load_bounds_inputs` prints `gamma_rtol` with the kept
    and stored positive counts, which is the truncation measurement.
  * each index logs which grid points it swept ("the tau grid window LO:HI of A:B"
    when the window applied, "over N grid point(s)" when the whole grid was swept),
    so the per-index grid evaluation count and the window-edge fallback rate are
    both recoverable. An index that logged *both* lines is one that fell back.
  * the run's closing summary line carries the refinement pencil cache's hits and
    misses. A killed run has no summary line, and then the cache counts are simply
    absent from the row rather than guessed at.

Host and device high-water marks are the one thing the log does not carry, because
the process that would have printed them was killed. Slurm's accounting does have
them, so `sacct_peak_rss_bytes` fetches `MaxRSS` for a job id; pass `--jobid` and
the row gets a memory number, omit it and the memory columns stay empty (which is
what "not measured" looks like everywhere else in this schema).
=#

const _LOG_TIMESTAMP = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)"

"First ISO 8601 timestamp on a line, or `nothing`."
function _log_timestamp(line::AbstractString)
    m = match(_LOG_TIMESTAMP, line)
    m === nothing && return nothing
    return tryparse(DateTime, m.captures[1])
end

_seconds_between(a::DateTime, b::DateTime) = (b - a).value / 1000

"""
    parse_bounds_log(path) -> NamedTuple

Reconstruct what a `compute_bounds.jl` (or `bench/point.jl --kind stage_bounds`)
run did, from its log alone. Tolerates truncation everywhere: a field that the run
never got as far as printing comes back `nothing`, and the per-index vectors hold
whatever indices did complete.

Returned fields:

- `num_pos`, `basis_size`, `total_directions`: from the `RSVD_BASIS_SIZE` line.
- `gamma_rtol`, `gamma_kept`, `gamma_stored`: from the truncation warning. Absent
  when nothing was truncated, which is itself information (`gamma_kept ==
  gamma_stored` then, and the caller can say so from `num_pos`).
- `tau_grid_points`: length of the grid the sweep announced.
- `outer_indices`, `outer_seconds`: per-index wall times, by differencing the
  "Computing" timestamps. The *last* started index has no successor to difference
  against, so it is dropped unless the run printed its "Dual is" line, in which
  case that line closes it.
- `grid_evals`: grid points swept per index, from the window/full-sweep lines.
- `fallbacks`: indices that logged a window and then the whole grid, that is, the
  window-edge fallback.
- `cache_hits`, `cache_misses`, `summary_fallbacks`, `tau_window`,
  `pencil_cache_max`: from the closing summary line, or `nothing` if the run was
  cut before it.
- `stage_times`: the `(gram_schmidt = ..., ...)` named tuple as a `Dict`, from the
  "Stage times" line, or from timestamp differences when that line is missing.
- `complete`: whether the closing summary line was reached at all.
- `killed_after_s`: seconds from the first timestamp in the log to the last, which
  is the wall time actually spent when the job was killed.
"""
function parse_bounds_log(path::AbstractString)
    isfile(path) || error("no such log: $path")

    num_pos = basis_size = total_directions = nothing
    gamma_rtol = gamma_kept = gamma_stored = nothing
    tau_grid_points = nothing
    cache_hits = cache_misses = summary_fallbacks = nothing
    tau_window = pencil_cache_max = nothing
    stage_times = Dict{String,Float64}()
    first_stamp = last_stamp = nothing

    # index -> timestamp of its "Computing" line, and index -> grid points swept.
    started = Tuple{Int,DateTime}[]
    closed = Dict{Int,DateTime}()
    grid_evals = Dict{Int,Int}()
    windowed = Set{Int}()
    fell_back = Set{Int}()
    marks = Dict{String,DateTime}()   # front-end stage boundaries

    for line in eachline(path)
        stamp = _log_timestamp(line)
        if stamp !== nothing
            first_stamp === nothing && (first_stamp = stamp)
            last_stamp = stamp
        end

        if (m = match(r"Using RSVD_BASIS_SIZE = (\d+) \(num_pos = (\d+) of (\d+)", line)) !== nothing
            basis_size = parse(Int, m.captures[1])
            num_pos = parse(Int, m.captures[2])
            total_directions = parse(Int, m.captures[3])
        elseif (m = match(r"Spectral truncation at gamma_rtol = ([0-9.eE+-]+): keeping (\d+) of the (\d+)", line)) !== nothing
            gamma_rtol = tryparse(Float64, m.captures[1])
            gamma_kept = parse(Int, m.captures[2])
            gamma_stored = parse(Int, m.captures[3])
        elseif (m = match(r"Eigendecomposing C\(.\) in the basis for .* \[(.*)\]", line)) !== nothing
            tau_grid_points = count(==(','), m.captures[1]) + 1
            stamp === nothing || (marks["c_range_start"] = stamp)
        elseif (m = match(r"grid window (\d+):(\d+) of (\d+):(\d+)", line)) !== nothing
            n = _log_outer_index(line)
            if n !== nothing
                lo, hi = parse(Int, m.captures[1]), parse(Int, m.captures[2])
                grid_evals[n] = hi - lo + 1
                push!(windowed, n)
            end
        elseif (m = match(r"over (\d+) .? ?grid point\(s\)", line)) !== nothing
            n = _log_outer_index(line)
            if n !== nothing
                # A windowed index that also sweeps the whole grid is a window-edge
                # fallback, and the full sweep is what it actually paid for.
                n in windowed && push!(fell_back, n)
                grid_evals[n] = parse(Int, m.captures[1])
            end
        elseif occursin("Computing", line) && occursin("bound", line)
            n = _log_outer_index(line)
            (n === nothing || stamp === nothing) || push!(started, (n, stamp))
        elseif occursin("Dual is", line)
            n = _log_outer_index(line)
            (n === nothing || stamp === nothing) || (closed[n] = stamp)
        elseif (m = match(r"cache (\d+) hit\(s\) / (\d+) miss\(es\) \(pencil_cache_max = (\d+)\), (\d+) full-grid fallback\(s\) \(tau_window = (-?\d+)\)", line)) !== nothing
            cache_hits = parse(Int, m.captures[1])
            cache_misses = parse(Int, m.captures[2])
            pencil_cache_max = parse(Int, m.captures[3])
            summary_fallbacks = parse(Int, m.captures[4])
            tau_window = parse(Int, m.captures[5])
        elseif (m = match(r"stage_times = \((.*)\)", line)) !== nothing
            for field in split(m.captures[1], ',')
                kv = split(field, '='; limit=2)
                length(kv) == 2 || continue
                v = tryparse(Float64, strip(kv[2]))
                v === nothing || (stage_times[strip(kv[1])] = v)
            end
        elseif occursin("reverse Gram-Schmidt", line) && occursin("Performing", line)
            stamp === nothing || (marks["gs_start"] = stamp)
        elseif occursin("Projecting C into the basis", line)
            stamp === nothing || (marks["c_projection_start"] = stamp)
        end
    end

    # Per-index durations. The gap to the next index's "Computing" line is the
    # honest number: it includes everything that index did, the same way
    # `outer_times` does. The last started index has no successor, so it is closed
    # by its own "Dual is" line when there is one and dropped otherwise.
    sort!(started; by=last)
    idxs, secs = Int[], Float64[]
    for (i, (n, t0)) in enumerate(started)
        t1 = i < length(started) ? last(started[i + 1]) : get(closed, n, nothing)
        t1 === nothing && continue
        dt = _seconds_between(t0, t1)
        dt >= 0 || continue
        push!(idxs, n)
        push!(secs, dt)
    end

    # Front-end stage times, when the summary line never printed them.
    if isempty(stage_times) && haskey(marks, "gs_start")
        if haskey(marks, "c_projection_start")
            stage_times["gram_schmidt_plus_ss_basis"] =
                _seconds_between(marks["gs_start"], marks["c_projection_start"])
        end
        if haskey(marks, "c_projection_start") && haskey(marks, "c_range_start")
            stage_times["c_projection"] =
                _seconds_between(marks["c_projection_start"], marks["c_range_start"])
        end
        if haskey(marks, "c_range_start") && !isempty(started)
            stage_times["c_range"] =
                _seconds_between(marks["c_range_start"], last(first(started)))
        end
    end

    killed_after = (first_stamp === nothing || last_stamp === nothing) ? nothing :
                   _seconds_between(first_stamp, last_stamp)

    return (num_pos=num_pos, basis_size=basis_size, total_directions=total_directions,
            gamma_rtol=gamma_rtol, gamma_kept=gamma_kept, gamma_stored=gamma_stored,
            tau_grid_points=tau_grid_points,
            outer_indices=idxs, outer_seconds=secs,
            outer_started=length(started),
            grid_evals=grid_evals, fallbacks=length(fell_back),
            cache_hits=cache_hits, cache_misses=cache_misses,
            summary_fallbacks=summary_fallbacks, tau_window=tau_window,
            pencil_cache_max=pencil_cache_max,
            stage_times=stage_times, complete=cache_hits !== nothing,
            killed_after_s=killed_after)
end

"The `n` of a `[n/m]` progress prefix, or `nothing`."
function _log_outer_index(line::AbstractString)
    m = match(r"\[(\d+)/(\d+)\]", line)
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

"""
    sacct_peak_rss_bytes(jobid) -> Union{Nothing,Int}

`MaxRSS` for a Slurm job, in bytes, as the largest value over its steps. The
memory high-water of a job that was killed before it could read
`/proc/self/status` itself, which is the whole reason this exists.

Returns `nothing` when `sacct` is unavailable or reports nothing, rather than
throwing: a missing memory number must not cost the row its timings.
"""
function sacct_peak_rss_bytes(jobid::AbstractString)
    out = try
        read(`sacct -j $jobid --noheader --parsable2 --format=MaxRSS`, String)
    catch
        return nothing
    end
    best = nothing
    for field in split(out, '\n')
        s = strip(field)
        isempty(s) && continue
        m = match(r"^([0-9.]+)([KMGT]?)$", s)
        m === nothing && continue
        v = tryparse(Float64, m.captures[1])
        v === nothing && continue
        scale = m.captures[2] == "K" ? 1024.0 : m.captures[2] == "M" ? 1024.0^2 :
                m.captures[2] == "G" ? 1024.0^3 : m.captures[2] == "T" ? 1024.0^4 : 1.0
        bytes = round(Int, v * scale)
        best = best === nothing ? bytes : max(best, bytes)
    end
    return best
end

"""
    bounds_log_row(parsed; kwargs...) -> Dict{String,Any}

Turn a `parse_bounds_log` result into a `kind = "stage_bounds"` CSV row in the same
schema `bench/point.jl` writes, so that `bench/fit.jl` cannot tell the two apart.

The geometry cannot be recovered reliably from the log (`parse_args` logs it
through a named tuple whose formatting is Julia's, not ours), so it is passed in.
The launcher that submitted the job knows it; that is where the values come from.

`extra` carries the same `key=value` notes `_bounds_result_notes` emits, plus
`from_log=1` and `log_complete=0/1`, so a row reconstructed from a killed job is
identifiable as one and the tau-shape fit can weight it as it sees fit.
"""
function bounds_log_row(parsed; cluster::AbstractString, cells::NTuple{3,Int},
                        scale::Rational{Int}, separation::Rational{Int},
                        rank::Integer, oversamples::Integer, power_iters::Integer,
                        threads::Integer=0, gpu_name::AbstractString="",
                        peak_rss_bytes::Union{Nothing,Integer}=nothing,
                        peak_vram_bytes::Union{Nothing,Integer}=nothing,
                        device_total_bytes::Union{Nothing,Integer}=nothing,
                        note::AbstractString="")
    notes = ["from_log=1", "log_complete=$(parsed.complete ? 1 : 0)",
             "outer_mode=$(parsed.complete ? "full" : "walltime_cut")"]
    parsed.num_pos === nothing || push!(notes, "num_pos=$(parsed.num_pos)")
    parsed.basis_size === nothing || push!(notes, "basis_size=$(parsed.basis_size)")
    parsed.gamma_rtol === nothing ||
        push!(notes, "gamma_rtol=$(@sprintf("%.6g", parsed.gamma_rtol))")
    parsed.gamma_stored === nothing || push!(notes, "stored_num_pos=$(parsed.gamma_stored)")
    parsed.tau_grid_points === nothing ||
        push!(notes, "tau_grid_points=$(parsed.tau_grid_points)")
    parsed.cache_hits === nothing || push!(notes, "tau_cache_hits=$(parsed.cache_hits)")
    parsed.cache_misses === nothing || push!(notes, "tau_cache_misses=$(parsed.cache_misses)")
    parsed.tau_window === nothing || push!(notes, "tau_window=$(parsed.tau_window)")
    parsed.pencil_cache_max === nothing ||
        push!(notes, "pencil_cache_max=$(parsed.pencil_cache_max)")

    n_eval = length(parsed.outer_indices)
    push!(notes, "outer_indices_done=$n_eval", "outer_indices_started=$(parsed.outer_started)")
    if n_eval > 0
        secs = sort(copy(parsed.outer_seconds))
        total = sum(secs)
        push!(notes, "outer_s_total=$(@sprintf("%.6g", total))",
              "outer_s_mean=$(@sprintf("%.6g", total / n_eval))",
              "outer_s_median=$(@sprintf("%.6g", secs[(n_eval + 1) ÷ 2]))",
              "outer_s_min=$(@sprintf("%.6g", first(secs)))",
              "outer_s_max=$(@sprintf("%.6g", last(secs)))",
              "outer_n_min=$(minimum(parsed.outer_indices))",
              "outer_n_max=$(maximum(parsed.outer_indices))")
        fallbacks = parsed.summary_fallbacks === nothing ? parsed.fallbacks :
                    parsed.summary_fallbacks
        push!(notes, "tau_grid_fallbacks=$fallbacks",
              "tau_grid_fallback_fraction=$(@sprintf("%.6g", fallbacks / n_eval))")
        parsed.cache_misses === nothing ||
            push!(notes, "tau_refine_whitenings_per_index=$(@sprintf("%.6g", parsed.cache_misses / n_eval))")
    end
    evaluated_grid = [parsed.grid_evals[n] for n in parsed.outer_indices
                      if haskey(parsed.grid_evals, n)]
    isempty(evaluated_grid) ||
        push!(notes, "tau_grid_evals_per_index=$(@sprintf("%.6g", sum(evaluated_grid) / length(evaluated_grid)))")
    for (name, seconds) in sort(collect(parsed.stage_times))
        push!(notes, "t_$(name)=$(@sprintf("%.6g", seconds))")
    end
    isempty(note) || push!(notes, note)

    return csv_row(
        cluster=cluster, kind="stage_bounds",
        device=isempty(gpu_name) ? "gpu" : "gpu", gpu_name=gpu_name,
        threads=threads == 0 ? "" : threads,
        n_x=cells[1], n_y=cells[2], n_z=cells[3], n_cells=prod(cells),
        scale_num=numerator(scale), scale_den=denominator(scale),
        sep_num=numerator(separation), sep_den=denominator(separation),
        contact=iszero(separation) ? 1 : 0,
        rank=rank, oversamples=oversamples, power_iters=power_iters,
        sketch_width=rank + oversamples,
        num_pos=parsed.num_pos === nothing ? "" : parsed.num_pos,
        time_s=parsed.killed_after_s === nothing ? NaN : parsed.killed_after_s,
        peak_rss_bytes=peak_rss_bytes === nothing ? "" : peak_rss_bytes,
        peak_vram_bytes=peak_vram_bytes === nothing ? "" : peak_vram_bytes,
        device_total_bytes=device_total_bytes === nothing ? "" : device_total_bytes,
        extra=join(notes, ";"),
    )
end

# --------------------------------------------------------------------------- #
# Standalone entry point: reconstruct a row from a bounds log
# --------------------------------------------------------------------------- #
#=
    julia --project=. bench/measure.jl --parse-bounds-log <log> --out <csv> \
        --cells 32,32,32 --scale 1//32 --sep 1//2 --rank 4000 [--jobid <id>]

Only reachable when this file is *run*, not when it is `include`d, so `point.jl`
and `fit.jl` are unaffected. `--summary` prints what was parsed and writes nothing,
which is the way to check a log before trusting a row built from it.
=#

function _measure_parse_cli(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        arg = argv[i]
        startswith(arg, "--") || error("Expected an option starting with --, got '$arg'")
        key = arg[3:end]
        if i + 1 > length(argv) || startswith(argv[i + 1], "--")
            opts[key] = "true"
            i += 1
        else
            opts[key] = argv[i + 1]
            i += 2
        end
    end
    return opts
end

function _measure_rational(s::AbstractString)
    occursin("//", s) || return Rational{Int}(parse(Int, strip(s)))
    a, b = split(strip(s), "//"; limit=2)
    return parse(Int, a) // parse(Int, b)
end

function _measure_cells(s::AbstractString)
    parts = split(strip(s, ['(', ')', ' ']), ',')
    length(parts) == 3 || error("--cells expects three comma-separated integers")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])), parse(Int, strip(parts[3])))
end

function measure_main(argv::Vector{String})
    opts = _measure_parse_cli(argv)
    log = get(opts, "parse-bounds-log", "")
    isempty(log) && error("bench/measure.jl takes --parse-bounds-log <log>; nothing else is runnable here")
    parsed = parse_bounds_log(log)

    n_eval = length(parsed.outer_indices)
    println("log:                 ", log)
    println("reached the summary: ", parsed.complete ? "yes" : "no (walltime cut or crash)")
    println("num_pos (kept):      ", something(parsed.num_pos, "unknown"))
    println("positives stored:    ", something(parsed.gamma_stored, "not truncated / unknown"))
    println("gamma_rtol:          ", something(parsed.gamma_rtol, "unknown"))
    println("tau grid points:     ", something(parsed.tau_grid_points, "unknown"))
    println("outer indices:       ", parsed.outer_started, " started, ", n_eval, " timed")
    if n_eval > 0
        println("per-index seconds:   mean ",
                @sprintf("%.3f", sum(parsed.outer_seconds) / n_eval),
                "  min ", @sprintf("%.3f", minimum(parsed.outer_seconds)),
                "  max ", @sprintf("%.3f", maximum(parsed.outer_seconds)))
    end
    ge = [parsed.grid_evals[n] for n in parsed.outer_indices if haskey(parsed.grid_evals, n)]
    isempty(ge) || println("grid evals/index:    ",
                           @sprintf("%.3f", sum(ge) / length(ge)),
                           "  (fallbacks: ", parsed.fallbacks, ")")
    parsed.cache_misses === nothing ||
        println("refine whitenings:   ", parsed.cache_misses, " miss(es) / ",
                parsed.cache_hits, " hit(s)")
    isempty(parsed.stage_times) ||
        println("stage times:         ", parsed.stage_times)
    println("wall time in log:    ",
            parsed.killed_after_s === nothing ? "unknown" :
            @sprintf("%.1f s", parsed.killed_after_s))

    out = get(opts, "out", "")
    (isempty(out) || haskey(opts, "summary")) && return parsed

    jobid = get(opts, "jobid", "")
    rss = haskey(opts, "peak-rss-bytes") ? parse(Int, opts["peak-rss-bytes"]) :
          isempty(jobid) ? nothing : sacct_peak_rss_bytes(jobid)
    row = bounds_log_row(parsed;
        cluster=get(opts, "cluster", detect_cluster()),
        cells=_measure_cells(get(opts, "cells", "32,32,32")),
        scale=_measure_rational(get(opts, "scale", "1//32")),
        separation=_measure_rational(get(opts, "sep", "1//32")),
        rank=parse(Int, get(opts, "rank", "4000")),
        oversamples=parse(Int, get(opts, "oversamples", "50")),
        power_iters=parse(Int, get(opts, "power-iters", "14")),
        threads=parse(Int, get(opts, "threads", "0")),
        gpu_name=get(opts, "gpu-name", ""),
        peak_rss_bytes=rss,
        peak_vram_bytes=haskey(opts, "peak-vram-bytes") ?
                        parse(Int, opts["peak-vram-bytes"]) : nothing,
        device_total_bytes=haskey(opts, "device-total-bytes") ?
                           parse(Int, opts["device-total-bytes"]) : nothing,
        note=get(opts, "note", ""))
    append_csv_row(out, row)
    println("-> ", out)
    return parsed
end

if abspath(PROGRAM_FILE) == @__FILE__
    measure_main(ARGS)
end
