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
