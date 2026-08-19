#!/usr/bin/env julia
"""
    bench/point.jl

Run exactly one calibration measurement and append one row to a CSV.

One point per process, always: peak resident set size is a per-process
high-water mark, so measuring several points in one process would report the
largest of them for all of them. `bench/plan.jl` generates the script that
invokes this once per point.

Usage:

    julia --project=. -t <threads> bench/point.jl --kind <kind> [options]

Kinds (see `run_point` at the bottom for the dispatch):

  Host / CPU, no GPU needed
    g0_self         build one self Green block, (R, R)
    g0_ext          build one external Green block, (R, S)
    g0_multiregion  build the 2x2 [S, R] <- [S, R] operator
    stage_greens    run the real `_generate_green_sr` end to end

  Device / GPU
    matvec_self     time G0_rr * v
    matvec_ext      time G0_rs * v
    matvec_uu       time the multi-region [S, R] <- [S, R] operator on a vector
    dense           time qr / gemm / eigh / geigh / svdvals at (m, c)
    bounds_core     time the bounds kernel on a synthetic spectrum
    mem_rsvd        real RSVD memory, with power iterations reduced (memory is
                    q-independent, time is not) -- the cheap way to measure the
                    RSVD's RAM and VRAM without a full run
    stage_rsvd      run the real `_generate_rsvd_sr` end to end
    stage_bounds    run the real `_compute_bounds_sr` end to end
    panel_bus       host-link rate and pipeline overlap for the Funicular panel
                    path: a pinned host-to-device size sweep plus Funicular's own
                    `benchmark/pinned.jl` and `benchmark/overlap.jl`. Writes
                    several `kind="panel_bus"` rows (trial E1)

Options (all have defaults; rationals are written `a//b`):

  --cells 32,32,32        cells of one body (both bodies use this)
  --receiver-cells        override the receiver's cells (defaults to --cells)
  --scale 1//32           cell size in wavelengths; negative uses the anisotropic
                          hack from `SMRSystem` exactly as production does
  --sep 1//32             surface-to-surface separation in wavelengths (0 = contact)
  --chi 13.6+0.05im       susceptibility
  --rank 800              RSVD target rank k
  --oversamples 50        p
  --power-iters 14        q
  --num-pos-frac 0.5      positive-eigenvalue fraction for bounds_core
  --outer-samples 3       how many outer bounds iterations bounds_core evaluates
  --dense-m / --dense-c   dimensions for the dense point (default 6n and k+p)
  --reps 20               timing repetitions for the cheap repeated points
  --gpu 0                 GPU device index, or -1 for CPU
  --preload / --project / --scratch   directories (default under --root)
  --root <dir>            base for the default directories
  --out <csv>             output CSV (default <root>/calibration_<cluster>.csv)
  --cluster <name>        override the cluster label
  --note <text>           free-form text appended to the `extra` column

Funicular-trial options (tier `funicular`; see bench/README.md):

  --seed 0                RSVD seed. The panel path regenerates its Gaussian test
                          matrix from this instead of holding one, so two panel
                          runs at the same seed sketch identically. The in-memory
                          path draws from the global RNG and ignores it, which is
                          why trial E2's parity is to RSVD accuracy rather than
                          bit-for-bit.
  --fresh                 delete the point's `.jld` (and the positive-vector `.h5`
                          it names) before running, so `stage_rsvd` measures the
                          work instead of skipping it because an output exists
  --force-path auto|panel `panel` builds a `Funicular.ResidencyPlan` here and hands
                          it to `_save_ur_asym` / `_run_rsvdvals` as
                          `plan_override`, which is the one hook that bypasses
                          `use_panel_path`. That is how trial E2 runs the panel
                          code on a whole A100 where the in-memory sketch fits, so
                          that the two halves of the parity check differ in the
                          storage path alone. `auto` (the default) leaves the
                          runtime's own predicate alone.
  --host-budget-GB 0      override the plan's `host_budget` (0 = take it from the
                          Slurm request, as production does). A debugging hook:
                          trial E3c forces the NVMe spill through `--mem` instead,
                          so that `residency_plan`'s own reading of
                          `SLURM_MEM_PER_NODE` is what gets exercised.
  --funicular-benchmark   directory holding Funicular's benchmark scripts
                          (default `joinpath(pkgdir(Funicular), "benchmark")`)

Backfill-trial options (tier `backfill`; every job under three hours so that it
rides the backfill queue at low priority):

  --gamma-rtol 1e-12      the bounds job's spectral cut, handed to
                          `load_bounds_inputs`. The positive block shrinks to the
                          directions with `Gamma[i] >= gamma_rtol * Gamma[1]`, which
                          at a far separation is a few dozen columns of two
                          thousand. `stage_bounds` reports both counts, so the
                          truncation itself becomes a fitted quantity rather than
                          an assumption (`bounds_m` in bench/cost_model.jl).
  --outer-blocks 4        run only this many blocks of consecutive outer indices
                          instead of all `m` of them, spread evenly over `1:m`.
                          `0` runs the whole loop, as production does.
  --outer-block-len 24    indices per block. Consecutive on purpose: the windowed
                          tau sweep only narrows for an `n` that immediately
                          follows the last one evaluated, so a block of
                          consecutive indices is the smallest thing that measures
                          the window, the plateau behaviour of the refinement
                          pencil cache, and the full-grid fallback rate at once.
                          The first index of each block pays a full-grid sweep,
                          which is exactly the fallback the fit needs to see.
  --design rs             which order the universe regions are named in, and so
                          which `file_prefix` the point reads and writes. This is
                          the ONE thing that has to match production byte for byte:
                          `src/common.jl` sorts the letters of `--design`, so the
                          production sweeps (`--design rs`) write
                          `<cells>__<cells>__<n>ss<d>__RS.jld`, while this script's
                          historical default builds the system as
                          `[Sender, Receiver]` and looks for `__SR.jld`. Both orders
                          describe the same geometry -- `design_regions` only feeds
                          the filename and a bounding-box union that the SR path
                          never reads -- but a point that wants to reuse a
                          production RSVD output has to spell the name the same way.
                          Default `sr`, which is what every existing tier used.
  --tau-window / --pencil-cache-max / --tau-refine-tol
                          overrides for `bounds_from_spectrum`'s own keywords.
                          Left unset they inherit the production defaults, which is
                          what a calibration point wants.
"""

using PhotonicSystemChannels
using GilaElectromagnetics
using CUDA
using LinearAlgebra
using JLD2
using Dates
using Printf
using Random
import Funicular

include(joinpath(@__DIR__, "measure.jl"))

#=
Captured after the package loads so that `T_ENTRY - PSC_T0` is the process
startup cost: Julia's own boot, precompiled image loading off a shared cluster
filesystem, and CUDA context creation. That is a real per-job cost worth tens of
seconds on Compute Canada, and it belongs in the time estimate. The launcher sets
PSC_T0 to the epoch second at which it invoked us.
=#
const T_ENTRY = time()
const T0 = haskey(ENV, "PSC_T0") ? tryparse(Float64, ENV["PSC_T0"]) : nothing
startup_seconds() = T0 === nothing ? nothing : max(0.0, T_ENTRY - T0)

# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #

function parse_cli(argv::Vector{String})
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

getopt(opts, key, default::AbstractString) = get(opts, key, default)
getopt_int(opts, key, default::Int) = haskey(opts, key) ? parse(Int, opts[key]) : default
getopt_float(opts, key, default::Float64) = haskey(opts, key) ? parse(Float64, opts[key]) : default

function getopt_rational(opts, key, default::Rational{Int})
    haskey(opts, key) || return default
    s = strip(opts[key])
    occursin("//", s) || return Rational{Int}(parse(Int, s))
    num, den = split(s, "//"; limit=2)
    return parse(Int, num) // parse(Int, den)
end

function getopt_cells(opts, key, default::NTuple{3,Int})
    haskey(opts, key) || return default
    parts = split(strip(opts[key], ['(', ')', ' ']), ',')
    length(parts) == 3 || error("--$key expects three comma-separated integers")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])), parse(Int, strip(parts[3])))
end

# --------------------------------------------------------------------------- #
# Point setup
# --------------------------------------------------------------------------- #

"""
    GPU_KINDS

Point kinds that need a device. Everything else defaults to `--gpu -1` so that a
host point never touches CUDA, and so that the Green-function builds it measures
stay on the host exactly as the production CPU job does.
"""
const GPU_KINDS = Set(["matvec_self", "matvec_ext", "matvec_uu", "dense",
                       "bounds_core", "mem_rsvd", "stage_rsvd", "stage_bounds",
                       "panel_bus"])

struct PointSpec
    kind::String
    sender_cells::NTuple{3,Int}
    receiver_cells::NTuple{3,Int}
    scale::Rational{Int}
    separation::Rational{Int}
    chi::ComplexF64
    rank::Int
    oversamples::Int
    power_iters::Int
    num_pos_frac::Float64
    outer_samples::Int
    dense_m::Int
    dense_c::Int
    reps::Int
    gpu::Int
    preload_dir::String
    project_dir::String
    scratch_dir::String
    out_csv::String
    cluster::String
    note::String
    seed::Int
    fresh::Bool
    force_path::String
    host_budget_GB::Float64
    funicular_bench_dir::String
    gamma_rtol::Float64
    outer_blocks::Int
    outer_block_len::Int
    tau_window::Union{Nothing,Int}
    pencil_cache_max::Union{Nothing,Int}
    tau_refine_tol::Union{Nothing,Float64}
    design::String
end

function PointSpec(opts::Dict{String,String})
    kind = getopt(opts, "kind", "")
    isempty(kind) && error("--kind is required")

    cells = getopt_cells(opts, "cells", (32, 32, 32))
    receiver_cells = getopt_cells(opts, "receiver-cells", cells)
    rank = getopt_int(opts, "rank", 800)
    oversamples = getopt_int(opts, "oversamples", 50)

    root = getopt(opts, "root", joinpath(homedir(), "psc-calibration"))
    cluster = getopt(opts, "cluster", detect_cluster())

    n_universe = 3 * (prod(cells) + prod(receiver_cells))

    return PointSpec(kind, cells, receiver_cells,
                     getopt_rational(opts, "scale", 1 // 32),
                     getopt_rational(opts, "sep", 1 // 32),
                     haskey(opts, "chi") ? parse(ComplexF64, opts["chi"]) : 13.6 + 0.05im,
                     rank, oversamples,
                     getopt_int(opts, "power-iters", 14),
                     getopt_float(opts, "num-pos-frac", 0.5),
                     getopt_int(opts, "outer-samples", 3),
                     getopt_int(opts, "dense-m", n_universe),
                     getopt_int(opts, "dense-c", rank + oversamples),
                     getopt_int(opts, "reps", 20),
                     getopt_int(opts, "gpu", kind in GPU_KINDS ? 0 : -1),
                     getopt(opts, "preload", joinpath(root, "preload")),
                     getopt(opts, "project", joinpath(root, "project")),
                     getopt(opts, "scratch", joinpath(root, "scratch")),
                     getopt(opts, "out", joinpath(root, "calibration_$(cluster).csv")),
                     cluster,
                     getopt(opts, "note", ""),
                     getopt_int(opts, "seed", 0),
                     getopt(opts, "fresh", "false") in ("true", "1", "yes"),
                     getopt(opts, "force-path", "auto"),
                     getopt_float(opts, "host-budget-GB", 0.0),
                     getopt(opts, "funicular-benchmark", _default_funicular_bench_dir()),
                     getopt_float(opts, "gamma-rtol", PhotonicSystemChannels.DEFAULT_GAMMA_RTOL),
                     getopt_int(opts, "outer-blocks", 0),
                     getopt_int(opts, "outer-block-len", 24),
                     haskey(opts, "tau-window") ? parse(Int, opts["tau-window"]) : nothing,
                     haskey(opts, "pencil-cache-max") ? parse(Int, opts["pencil-cache-max"]) : nothing,
                     haskey(opts, "tau-refine-tol") ? parse(Float64, opts["tau-refine-tol"]) : nothing,
                     lowercase(getopt(opts, "design", "sr")))
end

"""
    _default_funicular_bench_dir() -> String

Where Funicular keeps `overlap.jl` and `pinned.jl`. Resolved through
`pkgdir(Funicular)` rather than hard-coded, because Funicular comes in by URL and
its depot path carries a content hash that changes with every pin.
"""
function _default_funicular_bench_dir()
    dir = pkgdir(Funicular)
    dir === nothing && return ""
    return joinpath(dir, "benchmark")
end

uses_gpu(spec::PointSpec) = spec.gpu >= 0
n_cells(spec::PointSpec) = prod(spec.receiver_cells)

"""
    design_regions_for(design) -> Vector{SMRVolumeSymbol}

The `design_regions` a `--design` string names, by the same rule `src/common.jl`
uses: uppercase the letters, sort them, map each to its region. So `rs` and `sr`
both give `[Receiver, Sender]` there -- the sort is what makes production's
`file_prefix` end in `RS`.

`sr` here means something different on purpose: the literal `[Sender, Receiver]`
this script has always passed, whose prefix ends in `SR`. Existing tiers wrote
their scratch under that name and have to keep reading it, so it stays the default
and `rs` is the opt-in that matches production.
"""
function design_regions_for(design::AbstractString)
    design == "sr" && return SMRVolumeSymbol[Sender, Receiver]
    return char2volume_symbol.(sort(collect(uppercase(design))))
end

function build_system(spec::PointSpec)
    return SMRSystem(spec.sender_cells,
                     (spec.separation, 0 // 1, 0 // 1),
                     spec.receiver_cells,
                     design_regions_for(spec.design),
                     spec.scale,
                     spec.chi)
end

function build_environment(spec::PointSpec)
    mkpath(spec.preload_dir)
    mkpath(spec.project_dir)
    mkpath(spec.scratch_dir)
    return ComputeEnvironment(spec.preload_dir, spec.project_dir, spec.scratch_dir,
                              GPUChoice(uses_gpu(spec), spec.gpu))
end

gpu_name() = try
    string(CUDA.name(CUDA.device()))
catch
    ""
end

# --------------------------------------------------------------------------- #
# Result plumbing
# --------------------------------------------------------------------------- #

"""
    Measurement

What a point measures. `times` maps a sub-measurement name to seconds; the row's
headline `time_s` is `times[headline]`, and everything else is folded into the
`extra` column so that no information is lost.
"""
struct Measurement
    times::Dict{String,Float64}
    headline::String
    time_median_s::Union{Nothing,Float64}
    time_mean_s::Union{Nothing,Float64}
    bytes_written::Union{Nothing,Int}
    notes::Vector{String}
end

Measurement(; times=Dict{String,Float64}(), headline="total", time_median_s=nothing,
            time_mean_s=nothing, bytes_written=nothing, notes=String[]) =
    Measurement(times, headline, time_median_s, time_mean_s, bytes_written, notes)

function emit(spec::PointSpec, m::Measurement; baseline_rss, baseline_vram, vram,
              device_total=0)
    extras = copy(m.notes)
    for (name, seconds) in sort(collect(m.times))
        name == m.headline && continue
        push!(extras, "t_$(name)=$(@sprintf("%.6g", seconds))")
    end
    isempty(spec.note) || push!(extras, spec.note)

    row = csv_row(
        cluster=spec.cluster,
        kind=spec.kind,
        device=uses_gpu(spec) ? "gpu" : "cpu",
        gpu_name=uses_gpu(spec) ? gpu_name() : "",
        threads=Threads.nthreads(),
        n_x=spec.receiver_cells[1], n_y=spec.receiver_cells[2], n_z=spec.receiver_cells[3],
        n_cells=n_cells(spec),
        scale_num=numerator(spec.scale), scale_den=denominator(spec.scale),
        sep_num=numerator(spec.separation), sep_den=denominator(spec.separation),
        contact=iszero(spec.separation) ? 1 : 0,
        rank=spec.rank, oversamples=spec.oversamples, power_iters=spec.power_iters,
        sketch_width=spec.rank + spec.oversamples,
        dense_m=spec.dense_m, dense_c=spec.dense_c,
        reps=spec.reps,
        time_s=get(m.times, m.headline, NaN),
        time_median_s=m.time_median_s === nothing ? "" : m.time_median_s,
        time_mean_s=m.time_mean_s === nothing ? "" : m.time_mean_s,
        peak_rss_bytes=peak_rss_bytes(),
        baseline_rss_bytes=baseline_rss,
        peak_vram_bytes=vram.peak,
        peak_vram_delta_bytes=vram.delta,
        baseline_vram_bytes=baseline_vram,
        peak_vram_live_bytes=vram.live,
        peak_vram_reserved_bytes=vram.reserved,
        device_total_bytes=device_total,
        bytes_written=m.bytes_written === nothing ? "" : m.bytes_written,
        startup_s=startup_seconds() === nothing ? "" : startup_seconds(),
        extra=join(extras, ";"),
    )
    append_csv_row(spec.out_csv, row)

    println("\n" * "="^78)
    println("point: $(spec.kind)  cells=$(spec.receiver_cells)  sep=$(spec.separation)  ",
            "k=$(spec.rank)  threads=$(Threads.nthreads())")
    for (name, seconds) in sort(collect(m.times))
        marker = name == m.headline ? "*" : " "
        @printf("  %s %-22s %10.3f s\n", marker, name, seconds)
    end
    println("    peak RSS               ", human_bytes(peak_rss_bytes()),
            "  (baseline ", human_bytes(baseline_rss), ")")
    if uses_gpu(spec)
        println("    peak VRAM              ", human_bytes(vram.peak),
                "  (delta ", human_bytes(vram.delta), ")")
        println("    pool live / reserved   ", human_bytes(vram.live), " / ",
                human_bytes(vram.reserved))
    end
    println("  -> $(spec.out_csv)")
    println("="^78)
    return row
end

# --------------------------------------------------------------------------- #
# Host points: Green function block construction
# --------------------------------------------------------------------------- #

#=
Built with `save_to_disk=false` and `force_generate=true` so that the point
measures construction rather than deserialisation, and so that repeated runs do
not depend on what a previous point happened to leave in the preload directory.
=#

function point_g0_self(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    _, dt = timed(() -> load_green_function(env, smr, Receiver, Receiver;
                                            force_generate=true, save_to_disk=false))
    return Measurement(times=Dict("build" => dt), headline="build")
end

function point_g0_ext(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    _, dt = timed(() -> load_green_function(env, smr, Receiver, Sender;
                                            force_generate=true, save_to_disk=false))
    return Measurement(times=Dict("build" => dt), headline="build")
end

function point_g0_multiregion(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    _, dt = timed(() -> load_green_function(env, smr, [Sender, Receiver], [Sender, Receiver];
                                            force_generate=true, save_to_disk=false))
    return Measurement(times=Dict("build" => dt), headline="build")
end

function point_stage_greens(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    before = _preload_bytes(spec.preload_dir)
    _, dt = timed(() -> PhotonicSystemChannels._generate_green_sr(env, smr))
    written = _preload_bytes(spec.preload_dir) - before
    return Measurement(times=Dict("stage" => dt), headline="stage",
                       bytes_written=max(0, written))
end

function _preload_bytes(dir::AbstractString)
    total = 0
    isdir(dir) || return total
    for (root, _, files) in walkdir(dir)
        for f in files
            total += filesize(joinpath(root, f))
        end
    end
    return total
end

# --------------------------------------------------------------------------- #
# Device points: Green function matvecs
# --------------------------------------------------------------------------- #

function _timed_matvec(spec::PointSpec, operator)
    n = size(operator, 2)
    v = CUDA.randn(ComplexF64, n)
    f = () -> operator * v
    tmin, tmed, tmean = repeat_timed(f, spec.reps; device_sync=true, warmup=2)
    return Measurement(times=Dict("matvec" => tmin), headline="matvec",
                       time_median_s=tmed, time_mean_s=tmean,
                       notes=["operator_size=$(size(operator, 1))x$(size(operator, 2))"])
end

function point_matvec_self(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    operator = load_green_function(env, smr, Receiver, Receiver)
    return _timed_matvec(spec, operator)
end

function point_matvec_ext(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    operator = load_green_function(env, smr, Receiver, Sender)
    return _timed_matvec(spec, operator)
end

function point_matvec_uu(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    operator = load_green_function(env, smr, [Sender, Receiver], [Sender, Receiver])
    return _timed_matvec(spec, operator)
end

# --------------------------------------------------------------------------- #
# Device points: dense linear algebra
# --------------------------------------------------------------------------- #

"""
    point_dense(spec)

Time the dense GPU primitives the RSVD and bounds jobs spend real time in, at
the shapes they actually use: a thin QR of `m x c`, the two gemm shapes, a
Hermitian eigendecomposition of `c x c`, the generalized Hermitian
eigendecomposition that dominates the bounds outer loop, and singular values of
`c x c`.

Note `qthin!` is measured on an already-orthonormal matrix on repeat iterations,
which does not change its cost (Householder QR is oblivious to the input).
"""
function point_dense(spec::PointSpec)
    m, c = spec.dense_m, spec.dense_c
    c <= m || error("dense point needs c <= m, got m=$m c=$c")

    free, _ = CUDA.memory_info()
    needed = 3 * m * c * 16 + 6 * c^2 * 16
    needed < 0.8 * free ||
        error("dense point m=$m c=$c needs $(human_bytes(needed)) but only $(human_bytes(free)) is free")

    times = Dict{String,Float64}()
    notes = ["dense_m=$m", "dense_c=$c"]
    reps = max(3, spec.reps ÷ 4)

    # One unsupported primitive must not cost us the whole point: CUDA.jl's
    # coverage of `eigen` / `svdvals!` for Hermitian device matrices has moved
    # around between versions.
    function try_time!(name, f, n; kwargs...)
        try
            tmin, _, _ = repeat_timed(f, n; device_sync=true, kwargs...)
            times[name] = tmin
        catch err
            @warn "dense primitive '$name' failed" exception = err
            push!(notes, "failed_$name=$(typeof(err))")
        end
    end

    A = CUDA.randn(ComplexF64, m, c)
    try_time!("qr", () -> PhotonicSystemChannels.qthin!(A), reps; warmup=1)

    B = CUDA.randn(ComplexF64, m, c)
    try_time!("gemm_TN", () -> A' * B, reps; warmup=1)

    X = CUDA.randn(ComplexF64, c, c)
    try_time!("gemm_NN", () -> A * X, reps; warmup=1)
    CUDA.unsafe_free!(B)

    # A Hermitian matrix, and a Hermitian positive definite one for the second
    # argument of the pencil (`eigen!(Hermitian, Hermitian)` requires it).
    H = X' * X
    P = X' * X
    P[diagind(P)] .+= ComplexF64(c)
    try_time!("eigh", () -> eigen(Hermitian(copy(H))), max(2, reps ÷ 2); warmup=1)
    try_time!("geigh", () -> eigen!(Hermitian(copy(H)), Hermitian(copy(P))),
              max(2, reps ÷ 2); warmup=1)
    try_time!("svdvals", () -> svdvals!(copy(X)), max(2, reps ÷ 2); warmup=1)

    # BLAS-1 bandwidth, kernel launch latency and device-to-host synchronisation,
    # which between them set the cost of the bounds job's O(num_pos^2) loops.
    u = CUDA.randn(ComplexF64, m)
    w = CUDA.randn(ComplexF64, m)
    try_time!("dot", () -> dot(u, w), 4 * reps; warmup=2)
    try_time!("axpy", () -> (w .-= 2.0 * u), 4 * reps; warmup=2)
    tiny = CUDA.randn(ComplexF64, 1)
    try_time!("sync", () -> Array(tiny), 4 * reps; warmup=2)

    times["total"] = sum(values(times))
    push!(notes, "dense_reps=$reps")
    return Measurement(times=times, headline="qr", notes=notes)
end

# --------------------------------------------------------------------------- #
# Device point: the bounds kernel on a synthetic spectrum
# --------------------------------------------------------------------------- #

"""
    synthetic_spectrum(spec, N_u)

Plausible stand-in for an `Asym(G0_ur)` spectrum: `num_pos` positive eigenvalues
decaying logarithmically from `1/zeta`, the rest negative mirror images, and an
orthonormal eigenvector block. The scale matters because the bounds root-find
brackets on `kappa_tilde = zeta * Gamma`, so eigenvalues of the wrong magnitude
would exercise a different branch than production does.
"""
function synthetic_spectrum(spec::PointSpec, N_u::Int)
    k = spec.rank
    num_pos = clamp(round(Int, spec.num_pos_frac * k), 1, k)
    ζ = abs(spec.chi)^2 / imag(spec.chi)

    Γ = zeros(Float64, k)
    Γ[1:num_pos] .= collect(exp10.(range(log10(1 / ζ), log10(1e-6 / ζ), length=num_pos)))
    if num_pos < k
        Γ[(num_pos + 1):end] .= -collect(exp10.(range(log10(1e-6 / ζ), log10(1 / ζ),
                                                      length=k - num_pos)))
    end
    Γrs = collect(exp10.(range(log10(1 / ζ), log10(1e-8 / ζ), length=k)))

    V = CUDA.randn(ComplexF64, N_u, k)
    PhotonicSystemChannels.qthin!(V) # orthonormal columns, as a real eigenbasis would be
    return Γ, V, Γrs, num_pos
end

function point_bounds_core(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    N_u = 3 * (prod(spec.sender_cells) + prod(spec.receiver_cells))

    G₀_uu = load_green_function(env, smr, [Sender, Receiver], [Sender, Receiver])
    Γ, V, Γrs, num_pos = synthetic_spectrum(spec, N_u)

    # Sample the outer loop instead of running all num_pos iterations: the cost
    # of iteration n is affine in (num_pos - n), so a handful of well-spread
    # indices pins it down exactly.
    samples = max(1, min(spec.outer_samples, num_pos))
    outer_indices = unique(round.(Int, range(1, num_pos, length=samples)))

    times = Dict{String,Float64}()
    notes = ["num_pos=$num_pos", "outer_indices=$(join(outer_indices, "|"))"]
    result = nothing
    _, dt = timed(() -> begin
        try
            # `on_outer_error=:stop` so that a loop that cannot run on synthetic
            # input still returns the setup-stage timings, which are measured
            # before it and are the whole point of this measurement.
            result = bounds_from_spectrum(env, smr, Γ, V, Γrs;
                                          G₀_uu=G₀_uu, outer_indices=outer_indices,
                                          on_outer_error=:stop)
        catch err
            frames = stacktrace(catch_backtrace())
            where_str = isempty(frames) ? "unknown" :
                        join(["$(f.func)@$(basename(String(f.file))):$(f.line)"
                              for f in Iterators.take(frames, 3)], "<-")
            @warn "bounds setup failed on synthetic data" exception = err location = where_str
            push!(notes, "setup_ok=0")
            push!(notes, "setup_error=$(typeof(err))")
            push!(notes, "setup_where=$(where_str)")
        end
    end; device_sync=true)
    times["total"] = dt

    if result !== nothing
        st = result.stage_times
        times["gram_schmidt"] = st.gram_schmidt
        times["ss_basis"] = st.ss_basis
        times["c_projection"] = st.c_projection
        times["outer_total"] = st.outer_total
        push!(notes, "outer_ok=$(result.outer_error === nothing ? 1 : 0)")
        if result.outer_error !== nothing
            push!(notes, "outer_fail_n=$(result.outer_error.n)")
            push!(notes, "outer_error=$(replace(result.outer_error.exception, ';' => ',', '=' => ':'))")
            push!(notes, "outer_where=$(result.outer_error.location)")
        end
        for (n, seconds) in result.outer_times
            push!(notes, "outer[$n]=$(@sprintf("%.6g", seconds))")
        end
    end
    push!(notes, "N_u=$N_u")
    return Measurement(times=times, headline="total", notes=notes)
end

# --------------------------------------------------------------------------- #
# Device point: the host link, for the Funicular panel path (trial E1)
# --------------------------------------------------------------------------- #

#=
Two coefficients come out of this point, and nothing else identifies them:
`pcie_rate` (seconds per byte of pinned host-to-device traffic) and
`overlap_factor` (the share of that traffic the double-buffered sweep fails to
hide behind compute). On the panel path the sketch crosses the bus once per sweep,
so between them they are the whole of the panel time model's bus term.

Three sources, in decreasing order of how much the fit leans on them:

  1. an in-process pinned size sweep, through Funicular's own `alloc_host_slab` /
     `h2d!` / `sync_queue`. Four transfer sizes rather than one, because
     `rate_through_origin` fits a slope through the origin and a single point
     cannot tell a slope from a fixed per-transfer overhead.
  2. Funicular's `benchmark/pinned.jl`, which is the same measurement at one size
     plus the pageable comparison that says whether the host tier is page-locked
     at all. The pageable number is recorded under `bytes_pageable=` so that it
     stays in the CSV without being fitted, since a pageable copy is not the rate
     the panel path pays.
  3. Funicular's `benchmark/overlap.jl`, which is where `overlap_factor` comes
     from: at each compute-to-copy ratio it reports the copies alone, the compute
     alone, the sweep at one buffer and the sweep at two.

The scripts run as their own processes. Each wants a clean device and its own CUDA
context, and `benchmark/common.jl` defines `best`, `banner` and `record` at top
level, which would collide with this file's namespace if they were `include`d.
They also write their TSVs next to themselves, so the directory is copied
somewhere writable first: the depot's copy is not ours to scribble in.
=#

# `pinned.jl`'s transfer: N = 1 << 22 rows of W = 8 ComplexF64 columns.
const PINNED_SCRIPT_BYTES = (1 << 22) * 8 * 16
# `overlap.jl`'s serialized pass: N = 1 << 20, K = 16 ComplexF64, up and back.
const OVERLAP_SWEEP_BYTES = 2 * (1 << 20) * 16 * 16
# Rows of the in-process sweep. Column counts, at the same N as `pinned.jl`, so
# each slab is contiguous and the sizes span 64 MiB to 512 MiB.
const PANEL_BUS_SWEEP_WIDTHS = [1, 2, 4, 8]
const PANEL_BUS_SWEEP_ROWS = 1 << 22
#=
`fit_panel_bus` rejects an overlap of zero, since no benchmark can establish that
the bus is free and a zero would delete the term from the model. A sweep whose
copies vanish completely behind compute measures exactly that, so it is recorded
at this floor with the raw value kept alongside under `overlap_raw=`. Five percent
is the resolution `best`'s minimum-of-five sampling can defend at these durations.
=#
const OVERLAP_FLOOR = 0.05

"Minimum of `samples` timings after one warm-up, matching Funicular's own `best`."
function _best(f, samples::Int=5)
    f()
    CUDA.synchronize()
    return minimum(begin
                       _, dt = timed(f; device_sync=true)
                       dt
                   end for _ in 1:samples)
end

"""
    _pinned_sweep_rows(spec, device_total) -> (rows, notes)

Pinned host-to-device timings at several transfer sizes, using the same Funicular
primitives `benchmark/pinned.jl` does. Each size becomes one `panel_bus` row
carrying `bytes=`, which is what `pcie_rate` is fitted from.
"""
function _pinned_sweep_rows(spec::PointSpec, device_total::Int)
    rows = Dict{String,Any}[]
    notes = String[]
    backend = Funicular.cuda_backend()
    queue = Funicular.make_queue(backend)
    name = gpu_name()
    for width in PANEL_BUS_SWEEP_WIDTHS
        dims = (PANEL_BUS_SWEEP_ROWS, width)
        nbytes = prod(dims) * 16
        free, _ = CUDA.memory_info()
        if nbytes > 0.25 * free
            push!(notes, "sweep_skipped_w$(width)=insufficient_free_vram")
            continue
        end
        try
            slab = Funicular.alloc_host_slab(backend, nbytes)
            locked = Funicular.slab_matrix(slab, ComplexF64, dims, 0)
            fill!(locked, one(ComplexF64))
            device = Funicular.alloc_device(backend, ComplexF64, dims)
            dt = _best(() -> begin
                           Funicular.h2d!(device, locked, backend; queue=queue)
                           Funicular.sync_queue(backend, queue)
                       end)
            push!(rows, panel_bus_row(cluster=spec.cluster, gpu_name=name,
                                      device_total=device_total, time_s=dt,
                                      extras=["bytes=$(nbytes)", "source=pinned_sweep",
                                              "panel_cols=$(width)",
                                              "gbps=$(@sprintf("%.4g", nbytes / dt / 2^30))"]))
        catch err
            push!(notes, "sweep_failed_w$(width)=$(typeof(err))")
            @warn "pinned sweep failed" width exception = err
        end
        # The slabs and device blocks go out of scope here; reclaim so the next,
        # larger size sees the room rather than the pool's cached copy of it.
        GC.gc()
        CUDA.reclaim()
    end
    return rows, notes
end

"""
    _stage_funicular_benchmark(spec) -> String

Copy Funicular's `benchmark/` somewhere writable and return the copy's path.

`benchmark/common.jl` writes its TSVs to `joinpath(@__DIR__, "results")`, that is,
inside the package directory. Julia hands out depot package trees read-only often
enough that running in place is a coin flip, and a calibration point should not be
writing into the depot even where it can.
"""
function _stage_funicular_benchmark(spec::PointSpec)
    src = spec.funicular_bench_dir
    isempty(src) && error("could not locate Funicular's benchmark directory; pass --funicular-benchmark")
    isdir(src) || error("no Funicular benchmark directory at '$src'")
    dst = joinpath(spec.scratch_dir, "funicular_benchmark")
    rm(dst; recursive=true, force=true)
    mkpath(dirname(dst))
    cp(src, dst)
    # `cp` preserves the source's modes, and a read-only depot copy would make
    # `record`'s mkpath fail for exactly the reason we copied it out.
    chmod(dst, 0o755; recursive=true)
    return dst
end

"Run one of Funicular's benchmark scripts against the active project, and log it."
function _run_funicular_script(dir::AbstractString, script::AbstractString,
                               log_path::AbstractString)
    path = joinpath(dir, script)
    isfile(path) || return (ok=false, seconds=0.0, log="no such script: $path")
    project = Base.active_project()
    project_dir = project === nothing ? "@." : dirname(project)
    #=
    The main project, not `benchmark/Project.toml`. That file exists for `plot.jl`
    only, and lists DelimitedFiles and Plots, neither of which `pinned.jl` or
    `overlap.jl` touches. Both need CUDA, Funicular and Printf, which this project
    already has, so nothing needs instantiating for them.
    =#
    cmd = `$(Base.julia_cmd()) --project=$(project_dir) --startup-file=no $(path)`
    buf = IOBuffer()
    proc = try
        run(pipeline(ignorestatus(cmd); stdout=buf, stderr=buf); wait=true)
    catch err
        return (ok=false, seconds=0.0, log="failed to launch: $(err)")
    end
    text = String(take!(buf))
    try
        write(log_path, text)
    catch
    end
    print(text)
    return (ok=success(proc), seconds=0.0, log=text)
end

"Header-keyed rows of one of `benchmark/common.jl`'s tab-separated result files."
function _read_tsv(path::AbstractString)
    isfile(path) || return nothing
    lines = filter(!isempty, strip.(readlines(path)))
    length(lines) >= 2 || return nothing
    header = String.(split(lines[1], '\t'))
    out = Dict{String,String}[]
    for line in lines[2:end]
        fields = String.(split(line, '\t'))
        length(fields) == length(header) || continue
        push!(out, Dict(header[i] => fields[i] for i in eachindex(header)))
    end
    return isempty(out) ? nothing : out
end

_tsv_float(row, key) = haskey(row, key) ? tryparse(Float64, row[key]) : nothing

function point_panel_bus(spec::PointSpec)
    mkpath(spec.scratch_dir)
    free, total = CUDA.memory_info()
    device_total = Int(total)
    name = gpu_name()
    notes = String["N_bus_rows=0"]
    rows = Dict{String,Any}[]
    times = Dict{String,Float64}()

    # Three independent sources, and one of them failing must not cost the other
    # two. A point that comes back with the overlap factor and no rate is still
    # half of what E1 was for.
    try
        sweep_rows, sweep_notes = _pinned_sweep_rows(spec, device_total)
        append!(rows, sweep_rows)
        append!(notes, sweep_notes)
    catch err
        push!(notes, "pinned_sweep_failed=$(typeof(err))")
        @warn "the pinned size sweep failed; falling back to Funicular's own scripts" exception = err
    end

    bench_dir = ""
    try
        bench_dir = _stage_funicular_benchmark(spec)
        push!(notes, "funicular_benchmark=$(bench_dir)")
    catch err
        push!(notes, "stage_benchmark_failed=$(typeof(err))")
        @warn "could not stage Funicular's benchmark directory" exception = err
    end

    if !isempty(bench_dir)
        results = joinpath(bench_dir, "results")

        _, dt = timed(() -> _run_funicular_script(bench_dir, "pinned.jl",
                                                  joinpath(spec.scratch_dir, "funicular_pinned.log")))
        times["pinned_script"] = dt
        pinned = _read_tsv(joinpath(results, "pinned.tsv"))
        if pinned === nothing
            push!(notes, "pinned_tsv=missing")
        else
            for row in pinned
                source = get(row, "source", "")
                gbps = _tsv_float(row, "bandwidth_gbps")
                (gbps === nothing || gbps <= 0) && continue
                seconds = PINNED_SCRIPT_BYTES / (gbps * 2^30)
                # Only the page-locked copy is the rate the panel path pays, so
                # only that row gets the `bytes=` key the fit reads.
                key = source == "pinned" ? "bytes" : "bytes_pageable"
                push!(rows, panel_bus_row(cluster=spec.cluster, gpu_name=name,
                                          device_total=device_total, time_s=seconds,
                                          extras=["$key=$(PINNED_SCRIPT_BYTES)",
                                                  "source=script_$(source)",
                                                  "gbps=$(@sprintf("%.4g", gbps))"]))
                push!(notes, "script_$(source)_gbps=$(@sprintf("%.4g", gbps))")
            end
        end

        _, dt = timed(() -> _run_funicular_script(bench_dir, "overlap.jl",
                                                  joinpath(spec.scratch_dir, "funicular_overlap.log")))
        times["overlap_script"] = dt
        overlap = _read_tsv(joinpath(results, "overlap.tsv"))
        if overlap === nothing
            push!(notes, "overlap_tsv=missing")
        else
            for row in overlap
                ratio = _tsv_float(row, "ratio")
                copy_ms = _tsv_float(row, "copy_ms")
                compute_ms = _tsv_float(row, "compute_ms")
                serial_ms = _tsv_float(row, "serial_ms")
                pipeline_ms = _tsv_float(row, "pipeline_ms")
                any(isnothing, (ratio, copy_ms, compute_ms, pipeline_ms)) && continue
                (copy_ms <= 0 || pipeline_ms <= 0) && continue
                extras = ["source=script_overlap", "ratio=$(@sprintf("%.4g", ratio))",
                          "bytes_sweep=$(OVERLAP_SWEEP_BYTES)",
                          "copy_s=$(@sprintf("%.6g", copy_ms / 1e3))",
                          "compute_s=$(@sprintf("%.6g", compute_ms / 1e3))",
                          "pipeline_s=$(@sprintf("%.6g", pipeline_ms / 1e3))"]
                serial_ms === nothing ||
                    push!(extras, "serial_s=$(@sprintf("%.6g", serial_ms / 1e3))")
                #=
                `overlap_factor` is the share of the transfer the pipeline leaves
                exposed, and it is only measurable where compute dominates: below
                a ratio of one the sweep is bus-bound and `pipeline - compute` is
                most of the copy no matter how well the schedule overlaps.
                =#
                if compute_ms >= copy_ms
                    raw = (pipeline_ms - compute_ms) / copy_ms
                    fraction = clamp(raw, OVERLAP_FLOOR, 1.0)
                    push!(extras, "overlap=$(@sprintf("%.6g", fraction))")
                    push!(extras, "overlap_raw=$(@sprintf("%.6g", raw))")
                else
                    push!(extras, "overlap_skipped=bus_bound")
                end
                push!(rows, panel_bus_row(cluster=spec.cluster, gpu_name=name,
                                          device_total=device_total,
                                          time_s=pipeline_ms / 1e3, extras=extras))
            end
        end
    end

    for row in rows
        append_csv_row(spec.out_csv, row)
    end
    notes[1] = "N_bus_rows=$(length(rows))"
    push!(notes, "free_vram_at_start=$(Int(free))")
    times["total"] = sum(values(times); init=0.0)
    length(rows) >= 2 ||
        @warn "panel_bus produced fewer than two rows; pcie_rate will be a single point or absent"
    return Measurement(times=times, headline="total", notes=notes)
end

# --------------------------------------------------------------------------- #
# End-to-end stage points
# --------------------------------------------------------------------------- #

"""
    point_mem_rsvd(spec)

Real RSVD memory at a fraction of the real RSVD cost, by running the actual
`_generate_rsvd_sr` with the power-iteration count knocked down.

The trick is that **peak memory does not depend on `q`**. Looking at
`randomized_hermitian_range_finder`, each power iteration allocates one fresh
`N_u x c` block and drops the previous one, so the live set is the same on
iteration 1 as on iteration 14 -- `Omega`, `Q`, and `operator * Q`. Only the matvec
count, and so the time, scales with `q`. Two iterations rather than zero, because
one full cycle is what lets the allocator reach its steady state; going to `q = 14`
would buy nothing but wall time.

Everything else is the production path: the same operators, the same
`reigen_hermitian` and `rsvdvals` calls, the same host-side `Array(...)` copy and
JLD2 write. So `peak_rss_bytes` and the device high-water are the numbers the real
job would produce.

Needs the Green functions in the preload directory. `bench/plan.jl`'s `memory`
tier submits the `stage_greens` point first and makes this depend on it.
"""
function point_mem_rsvd(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    params = RSVDParams(spec.rank, spec.oversamples, spec.power_iters, spec.seed)
    jld = joinpath(spec.scratch_dir, "$(file_prefix(smr)).jld")
    # A stale JLD makes `_generate_rsvd_sr` skip the work it is here to measure.
    isfile(jld) && rm(jld; force=true)
    _, dt = timed(() -> PhotonicSystemChannels._generate_rsvd_sr(env, smr, params);
                  device_sync=true)
    written = isfile(jld) ? filesize(jld) : 0
    N_u = 3 * (prod(spec.sender_cells) + prod(spec.receiver_cells))
    c = spec.rank + spec.oversamples
    return Measurement(times=Dict("stage" => dt), headline="stage",
                       bytes_written=written,
                       notes=["N_u=$N_u", "sketch_width=$c",
                              "reduced_power_iters=$(spec.power_iters)",
                              "analytic_live_bytes=$(3 * N_u * c * 16)"])
end

"""
    _plan_scratch_dir() -> Union{Nothing,String}

Where `residency_plan` would put Funicular's spill files: node-local NVMe when
Slurm gave us one. Duplicated here rather than reached into, so that the override
path and the production path agree without `bench/` importing an internal.
"""
function _plan_scratch_dir()
    haskey(ENV, "SLURM_TMPDIR") || return nothing
    dir = joinpath(ENV["SLURM_TMPDIR"], "funicular")
    mkpath(dir)
    return dir
end

"""
    _forced_plan(spec, env, N_u) -> Union{Nothing,Funicular.ResidencyPlan}

The `plan_override` for `--force-path panel`, and `nothing` for `--force-path
auto` (which leaves `src/rsvd.jl`'s `use_panel_path` predicate in charge).

Forcing exists for trial E2. The parity check wants the in-memory and panel paths
compared on the *same* card, and the predicate would never choose panel there:
1 lambda at k = 1350 is a 12 GB sketch on a 40 GB A100. The alternative, running
the panel half on a 3g.20gb slice where the predicate flips on its own, would
confound the storage path with three eighths of the streaming multiprocessors, and
the wall-clock ratio the trial is for would mean nothing.

`--host-budget-GB` reconstructs the plan with a smaller host tier instead of
taking it from Slurm, which forces the NVMe spill without a large `--mem`. Trial
E3c does *not* use it: going through `--mem` exercises `residency_plan`'s own
reading of `SLURM_MEM_PER_NODE`, which is the code the production sweep will run.
"""
function _forced_plan(spec::PointSpec, env, N_u::Int)
    spec.force_path in ("auto", "panel") ||
        error("--force-path must be auto or panel, got '$(spec.force_path)'")
    spec.force_path == "panel" || return nothing
    uses_gpu(spec) || error("--force-path panel needs a GPU")
    workspace = gila_workspace_bytes(N_u)
    spec.host_budget_GB > 0 || return residency_plan(env; workspace_bytes=workspace)
    return Funicular.ResidencyPlan(
        backend=Funicular.cuda_backend(),
        device_budget=device_budget_bytes(),
        host_budget=round(Int, spec.host_budget_GB * 2^30),
        workspace_bytes=workspace,
        scratch_dir=_plan_scratch_dir(),
    )
end

"Whatever the run left in the JLD that describes the spectrum, for the row's notes."
function _rsvd_output_notes(jld::AbstractString)
    notes = String[]
    isfile(jld) || return notes
    try
        jldopen(jld, "r") do io
            for (key, label) in (("UR_asym/num_pos", "num_pos"),
                                 ("UR_asym/seed", "out_seed"),
                                 ("UR_asym/exact", "exact"),
                                 ("UR_asym/vectors_file", "vectors_file"))
                haskey(io, key) && push!(notes, "$label=$(io[key])")
            end
            if haskey(io, "UR_asym/D") && haskey(io, "UR_asym/num_pos")
                total = length(io["UR_asym/D"])
                total == 0 ||
                    push!(notes,
                          "positive_fraction=$(@sprintf("%.4g", io["UR_asym/num_pos"] / total))")
            end
        end
    catch err
        push!(notes, "output_read_failed=$(typeof(err))")
    end
    return notes
end

function point_stage_rsvd(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    params = RSVDParams(spec.rank, spec.oversamples, spec.power_iters, spec.seed)
    jld = joinpath(spec.scratch_dir, "$(file_prefix(smr)).jld")
    vectors = ur_asym_vectors_path(env, smr)
    if spec.fresh
        # `_save_ur_asym` skips a complete output, and "complete" includes the h5
        # the JLD names, so both have to go or the point measures a no-op.
        rm(jld; force=true)
        rm(vectors; force=true)
    end
    before = isfile(jld) ? filesize(jld) : 0

    N_u = 3 * (prod(spec.sender_cells) + prod(spec.receiver_cells))
    c = spec.rank + spec.oversamples
    plan = _forced_plan(spec, env, N_u)
    notes = ["N_u=$N_u", "sketch_width=$c", "seed=$(spec.seed)",
             "force_path=$(spec.force_path)", "fresh=$(spec.fresh ? 1 : 0)",
             "workspace_bytes=$(uses_gpu(spec) ? gila_workspace_bytes(N_u) : 0)",
             "predicate_says_panel=$(uses_gpu(spec) && use_panel_path(N_u, spec.rank, env) ? 1 : 0)",
             "slurm_mem_per_node_mb=$(get(ENV, "SLURM_MEM_PER_NODE", ""))",
             "slurm_tmpdir=$(haskey(ENV, "SLURM_TMPDIR") ? 1 : 0)"]
    uses_gpu(spec) && push!(notes, "device_budget=$(device_budget_bytes())")
    plan === nothing || push!(notes, "plan_override=1")
    spec.host_budget_GB > 0 &&
        push!(notes, "host_budget_override_GB=$(@sprintf("%.6g", spec.host_budget_GB))")

    _, dt = timed(() -> begin
                      if plan === nothing
                          PhotonicSystemChannels._generate_rsvd_sr(env, smr, params)
                      else
                          # `_generate_rsvd_sr` has no `plan_override` of its own,
                          # so the two halves it calls are driven directly. Same
                          # calls, same order, one extra keyword.
                          PhotonicSystemChannels._save_ur_asym(env, smr, params;
                                                               plan_override=plan)
                          PhotonicSystemChannels._run_rsvdvals(env, smr, params, "RS/";
                                                               plan_override=plan)
                      end
                  end; device_sync=true)

    written = (isfile(jld) ? filesize(jld) : 0) - before
    isfile(vectors) && (written += filesize(vectors))
    append!(notes, _rsvd_output_notes(jld))
    push!(notes, "jld=$(jld)")
    return Measurement(times=Dict("stage" => dt), headline="stage",
                       bytes_written=max(0, written), notes=notes)
end

"""
    outer_index_blocks(m, nblocks, blocklen) -> Vector{Int}

`nblocks` runs of `blocklen` consecutive outer indices, spread evenly over `1:m`,
deduplicated and sorted.

Consecutive within a block and spread between them, for two different reasons.
Consecutive, because the windowed tau sweep in `bounds_from_spectrum` only narrows
for an `n` that immediately follows the last index evaluated, and the refinement
pencil cache only hits when neighbouring indices sit on the same tau* plateau: a
scattered set of indices would measure neither, and would report the unwindowed
cost as if it were the windowed one. Spread, because the per-index cost falls off
across the loop -- index `n` probes `m - n + 1` vectors -- so a sample taken only
at the top would put the per-index slope at twice its average, and the first index
of each block is also the full-grid fallback the fit needs to count.

Cost, against the whole loop's `m(m+1)/2` probes: `blocklen * m * nblocks(nblocks+1)/2
/ nblocks`, that is, of order `blocklen * m`, so the sample is `2 * blocklen / m`
of the full outer loop. At `m = 2400`, four blocks of 24 is about 5%.
"""
function outer_index_blocks(m::Int, nblocks::Int, blocklen::Int)
    (nblocks <= 0 || blocklen <= 0 || m <= 0) && return Int[]
    idxs = Int[]
    for b in 0:(nblocks - 1)
        start = clamp(round(Int, b * m / nblocks) + 1, 1, m)
        append!(idxs, start:min(m, start + blocklen - 1))
    end
    return sort!(unique!(idxs))
end

"""
    _stored_num_pos(env, smr) -> Union{Nothing,Int}

The positive count the RSVD job wrote, before `--gamma-rtol` cut it down.
`load_bounds_inputs` returns only the kept count, and the ratio of the two is the
whole point of the truncation measurement, so it is read straight out of the JLD.
"""
function _stored_num_pos(env, smr)
    path = joinpath(scratch_dir(env), "$(file_prefix(smr)).jld")
    isfile(path) || return nothing
    try
        return jldopen(path, "r") do io
            haskey(io, "UR_asym/num_pos") ? Int(io["UR_asym/num_pos"]) : nothing
        end
    catch
        return nothing
    end
end

"""
    point_stage_bounds(spec)

Two shapes, chosen by `--outer-blocks`.

`--outer-blocks 0` (the default) runs `_compute_bounds_sr`, which is production
exactly: the whole outer loop, and the output JLD written at the end.

Anything else runs `load_bounds_inputs` followed by `bounds_from_spectrum` with
`outer_indices` restricted to `outer_index_blocks`, and writes no output JLD. That
is the shape the three-hour cap needs. The bounds job's cost is
`front end + sum over n of per-index cost`, the front end is measured once either
way, and the per-index costs come back individually in `outer_times`, so a sample
of the loop identifies the same coefficients the whole loop would -- at a few
percent of the wall time, and without the risk that a walltime kill takes the row
with it. It also reads the *production* scratch directory without writing anything
into the production project directory, which is what lets the backfill tier reuse
RSVD outputs that are already on disk.

`on_outer_error = :stop` throughout, so a numerical failure at one index still
yields the front-end timings and the indices that did run.
"""
function point_stage_bounds(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    params = RSVDParams(spec.rank, spec.oversamples, spec.power_iters, spec.seed)
    N_u = 3 * (prod(spec.sender_cells) + prod(spec.receiver_cells))
    stored = _stored_num_pos(env, smr)
    notes = ["N_u=$N_u", "gamma_rtol=$(@sprintf("%.6g", spec.gamma_rtol))"]
    stored === nothing || push!(notes, "stored_num_pos=$stored")

    if spec.outer_blocks <= 0
        result = nothing
        _, dt = timed(() -> (result = PhotonicSystemChannels._compute_bounds_sr(
                                 env, smr, params; gamma_rtol=spec.gamma_rtol));
                      device_sync=true)
        times = Dict("stage" => dt)
        push!(notes, "outer_mode=full")
        if result !== nothing
            append!(notes, _bounds_result_notes(result, dt))
            st = result.stage_times
            times["gram_schmidt"] = st.gram_schmidt
            times["ss_basis"] = st.ss_basis
            times["c_projection"] = st.c_projection
            times["c_range"] = st.c_range
            times["outer_total"] = st.outer_total
        end
        return Measurement(times=times, headline="stage", notes=notes)
    end

    # Sampled outer loop. The load is timed separately from the loop, because the
    # front end is the term the panel-path coefficients are fitted against and it
    # is the one part of the job a sampled run measures in full.
    inputs = nothing
    _, t_load = timed(() -> (inputs = load_bounds_inputs(env, smr;
                                                        gamma_rtol=spec.gamma_rtol));
                      device_sync=true)
    m = inputs.num_pos
    idxs = outer_index_blocks(m, spec.outer_blocks, spec.outer_block_len)
    isempty(idxs) && (idxs = [1])
    kwargs = Dict{Symbol,Any}(:num_pos => m, :outer_indices => idxs,
                              :on_outer_error => :stop)
    spec.tau_window === nothing || (kwargs[:tau_window] = spec.tau_window)
    spec.pencil_cache_max === nothing || (kwargs[:pencil_cache_max] = spec.pencil_cache_max)
    spec.tau_refine_tol === nothing || (kwargs[:tau_refine_tol] = spec.tau_refine_tol)

    result = nothing
    _, dt = timed(() -> (result = bounds_from_spectrum(env, smr, inputs.Γ, inputs.Vur_asym,
                                                      inputs.Γrs; kwargs...));
                  device_sync=true)
    times = Dict("stage" => dt + t_load, "load" => t_load, "bounds" => dt)
    push!(notes, "outer_mode=sampled", "outer_blocks=$(spec.outer_blocks)",
          "outer_block_len=$(spec.outer_block_len)",
          "outer_indices_requested=$(length(idxs))",
          "panel_front_end=$(inputs.plan === nothing ? 0 : 1)")
    if result !== nothing
        append!(notes, _bounds_result_notes(result, dt))
        st = result.stage_times
        times["gram_schmidt"] = st.gram_schmidt
        times["ss_basis"] = st.ss_basis
        times["c_projection"] = st.c_projection
        times["c_range"] = st.c_range
        times["outer_total"] = st.outer_total
    end
    return Measurement(times=times, headline="stage", notes=notes)
end

"""
    _bounds_result_notes(result, dt) -> Vector{String}

Everything `bench/fit.jl` needs out of a `bounds_from_spectrum` return, as
`key=value` notes.

The tau-search counts are the reason this exists. `result.tau_search` gives the
refinement pencil cache's hits and misses and the number of indices that fell back
to a full grid sweep, and `bounds_dual_by_tau` keeps `NaN` at every grid point an
index did not evaluate, so the number of finite entries per row *is* that index's
grid evaluation count. Together they are the four numbers of `CostModel.TauShape`,
measured rather than assumed:

  * `tau_grid_evals_per_index`  -- mean finite entries per evaluated row. The
    windowed sweep makes this about `min(2 * tau_window + 1, grid)`, and less when
    the minimiser sits at a grid end, where it usually does. A row swept in full
    after a window-edge fallback contributes the whole grid, so the fallback is
    already folded in at its measured rate.
  * `tau_refine_whitenings_per_index` -- cache misses per evaluated index. On a
    plateau this is near zero while the refinement still runs its full complement
    of probes, which is the over-charge the old `TAU_REFINE_EVALS` constant made.
  * `tau_grid_fallbacks` / `tau_cache_hits` / `tau_cache_misses` -- the raw counts,
    kept so the derived means above can be re-derived differently later.

`outer_s_*` are the per-index wall times from `outer_times`. The mean is what a
full loop of this shape would cost per index; the first and last are reported too
because the per-index cost falls off across the loop (index `n` probes `m - n + 1`
vectors) and a single mean over a sampled set hides that.
"""
function _bounds_result_notes(result, dt::Real)
    notes = ["num_pos=$(result.num_pos)", "complete=$(result.complete ? 1 : 0)",
             "basis_size=$(result.basis_size)"]
    ts = result.tau_search
    push!(notes, "tau_grid_points=$(length(result.tau_grid))",
          "tau_cache_hits=$(ts.pencil_cache_hits)",
          "tau_cache_misses=$(ts.pencil_cache_misses)",
          "tau_grid_fallbacks=$(ts.grid_fallbacks)")

    evaluated = [n for (n, _) in result.outer_times]
    n_eval = length(evaluated)
    push!(notes, "outer_indices_done=$n_eval")
    if n_eval > 0
        secs = sort([t for (_, t) in result.outer_times])
        total = sum(secs)
        push!(notes, "outer_s_total=$(@sprintf("%.6g", total))",
              "outer_s_mean=$(@sprintf("%.6g", total / n_eval))",
              "outer_s_median=$(@sprintf("%.6g", secs[(n_eval + 1) ÷ 2]))",
              "outer_s_min=$(@sprintf("%.6g", first(secs)))",
              "outer_s_max=$(@sprintf("%.6g", last(secs)))",
              "outer_n_min=$(minimum(evaluated))", "outer_n_max=$(maximum(evaluated))",
              "tau_refine_whitenings_per_index=$(@sprintf("%.6g", ts.pencil_cache_misses / n_eval))",
              "tau_grid_fallback_fraction=$(@sprintf("%.6g", ts.grid_fallbacks / n_eval))")
        # Finite entries per evaluated row of the grid table: the grid evaluations
        # that index actually made.
        table = result.bounds_dual_by_tau
        finite = [count(isfinite, view(table, n, :)) for n in evaluated
                  if 1 <= n <= size(table, 1)]
        isempty(finite) ||
            push!(notes,
                  "tau_grid_evals_per_index=$(@sprintf("%.6g", sum(finite) / length(finite)))")
    end
    if result.outer_error !== nothing
        push!(notes, "outer_error_n=$(result.outer_error.n)")
    end
    return notes
end

# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

const POINT_KINDS = Dict{String,Function}(
    "g0_self" => point_g0_self,
    "g0_ext" => point_g0_ext,
    "g0_multiregion" => point_g0_multiregion,
    "stage_greens" => point_stage_greens,
    "matvec_self" => point_matvec_self,
    "matvec_ext" => point_matvec_ext,
    "matvec_uu" => point_matvec_uu,
    "dense" => point_dense,
    "bounds_core" => point_bounds_core,
    "mem_rsvd" => point_mem_rsvd,
    "stage_rsvd" => point_stage_rsvd,
    "stage_bounds" => point_stage_bounds,
    "panel_bus" => point_panel_bus,
)

function run_point(spec::PointSpec)
    haskey(POINT_KINDS, spec.kind) ||
        error("Unknown --kind '$(spec.kind)'. Known: $(join(sort(collect(keys(POINT_KINDS))), ", "))")

    if spec.kind in GPU_KINDS
        uses_gpu(spec) || error("Kind '$(spec.kind)' needs a GPU; pass --gpu <index>")
        # `CUDA.device!` breaks under Compute Canada's cgroup GPU binding, which
        # already exposes exactly the requested device as index 0.
        haskey(ENV, "CC_CLUSTER") || CUDA.device!(spec.gpu)
        CUDA.synchronize()
    end

    baseline_rss = peak_rss_bytes()
    baseline_vram = 0
    device_total = 0
    if uses_gpu(spec)
        free, total = CUDA.memory_info()
        baseline_vram = Int(total - free)
        device_total = Int(total)
    end

    watcher = start_vram_watcher(enabled=uses_gpu(spec))
    local measurement
    try
        measurement = POINT_KINDS[spec.kind](spec)
    catch err
        stop_vram_watcher!(watcher)
        rethrow(err)
    end
    vram = stop_vram_watcher!(watcher)

    return emit(spec, measurement; baseline_rss=baseline_rss, baseline_vram=baseline_vram,
                vram=vram, device_total=device_total)
end

if abspath(PROGRAM_FILE) == @__FILE__
    spec = PointSpec(parse_cli(ARGS))
    run_point(spec)
end
