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
    stage_rsvd      run the real `_generate_rsvd_sr` end to end
    stage_bounds    run the real `_compute_bounds_sr` end to end

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
"""

using PhotonicSystemChannels
using GilaElectromagnetics
using CUDA
using LinearAlgebra
using JLD2
using Dates
using Printf
using Random

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
                       "bounds_core", "stage_rsvd", "stage_bounds"])

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
                     getopt(opts, "note", ""))
end

uses_gpu(spec::PointSpec) = spec.gpu >= 0
n_cells(spec::PointSpec) = prod(spec.receiver_cells)

function build_system(spec::PointSpec)
    return SMRSystem(spec.sender_cells,
                     (spec.separation, 0 // 1, 0 // 1),
                     spec.receiver_cells,
                     [Sender, Receiver],
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

function emit(spec::PointSpec, m::Measurement; baseline_rss, baseline_vram,
              peak_vram, peak_vram_delta)
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
        peak_vram_bytes=peak_vram,
        peak_vram_delta_bytes=peak_vram_delta,
        baseline_vram_bytes=baseline_vram,
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
    uses_gpu(spec) && println("    peak VRAM              ", human_bytes(peak_vram),
                              "  (delta ", human_bytes(peak_vram_delta), ")")
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
            result = bounds_from_spectrum(env, smr, Γ, V, Γrs;
                                          G₀_uu=G₀_uu, outer_indices=outer_indices)
            push!(notes, "outer_ok=1")
        catch err
            # A synthetic C_basis is not guaranteed positive definite, so the
            # generalized eigendecomposition can legitimately refuse. The setup
            # stages are still measured; `geigh` comes from the dense point.
            @warn "bounds outer loop failed on synthetic data" exception = err
            push!(notes, "outer_ok=0")
            push!(notes, "outer_error=$(typeof(err))")
        end
    end; device_sync=true)
    times["total"] = dt

    if result !== nothing
        st = result.stage_times
        times["gram_schmidt"] = st.gram_schmidt
        times["ss_basis"] = st.ss_basis
        times["c_projection"] = st.c_projection
        times["outer_total"] = st.outer_total
        for (n, seconds) in result.outer_times
            push!(notes, "outer[$n]=$(@sprintf("%.6g", seconds))")
        end
    end
    push!(notes, "N_u=$N_u")
    return Measurement(times=times, headline="total", notes=notes)
end

# --------------------------------------------------------------------------- #
# End-to-end stage points
# --------------------------------------------------------------------------- #

function point_stage_rsvd(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    params = RSVDParams(spec.rank, spec.oversamples, spec.power_iters)
    jld = joinpath(spec.scratch_dir, "$(file_prefix(smr)).jld")
    before = isfile(jld) ? filesize(jld) : 0
    _, dt = timed(() -> PhotonicSystemChannels._generate_rsvd_sr(env, smr, params);
                  device_sync=true)
    written = (isfile(jld) ? filesize(jld) : 0) - before
    return Measurement(times=Dict("stage" => dt), headline="stage",
                       bytes_written=max(0, written))
end

function point_stage_bounds(spec::PointSpec)
    env = build_environment(spec)
    smr = build_system(spec)
    params = RSVDParams(spec.rank, spec.oversamples, spec.power_iters)
    result = nothing
    _, dt = timed(() -> (result = PhotonicSystemChannels._compute_bounds_sr(env, smr, params));
                  device_sync=true)
    times = Dict("stage" => dt)
    notes = String[]
    if result !== nothing
        push!(notes, "num_pos=$(result.num_pos)")
        st = result.stage_times
        times["gram_schmidt"] = st.gram_schmidt
        times["ss_basis"] = st.ss_basis
        times["c_projection"] = st.c_projection
        times["outer_total"] = st.outer_total
    end
    return Measurement(times=times, headline="stage", notes=notes)
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
    "stage_rsvd" => point_stage_rsvd,
    "stage_bounds" => point_stage_bounds,
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
    if uses_gpu(spec)
        free, total = CUDA.memory_info()
        baseline_vram = Int(total - free)
    end

    watcher = start_vram_watcher(enabled=uses_gpu(spec))
    local measurement
    try
        measurement = POINT_KINDS[spec.kind](spec)
    catch err
        stop_vram_watcher!(watcher)
        rethrow(err)
    end
    peak_vram, peak_vram_delta = stop_vram_watcher!(watcher)

    return emit(spec, measurement; baseline_rss=baseline_rss, baseline_vram=baseline_vram,
                peak_vram=peak_vram, peak_vram_delta=peak_vram_delta)
end

if abspath(PROGRAM_FILE) == @__FILE__
    spec = PointSpec(parse_cli(ARGS))
    run_point(spec)
end
