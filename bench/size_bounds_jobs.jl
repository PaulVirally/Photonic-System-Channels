#!/usr/bin/env julia
"""
    bench/size_bounds_jobs.jl

Size every bounds job in a sweep from the `m` its RSVD output actually produced,
rather than from the truncation envelope's guess. Writes one line per separation:

    <sep as n//d>  <kept m>  <gpu>  <host GB>  <HH:MM:SS>  <mode>

    julia --project=. bench/size_bounds_jobs.jl \\
        --scratch /home/pvirally/scratch/Photonic-System-Channels/<PROJECT>/ \\
        --cells 16,16,16 --scale 1//32 --rank 4000 --oversamples 50 \\
        --power-iters 14 --cluster narval --gamma-rtol 1.0e-12 \\
        --out jobs/bounds_sizes_<PROJECT>.txt

Runs on the login node in seconds and touches no GPU. Separations whose RSVD output
is missing or incomplete are reported on stderr and left out of the table, which is
what makes the generated `submit_bounds_*.sh` safe to run against a partly finished
sweep: a bounds job is emitted only where there is a basis to point it at.

# Why the bounds stage is sized here and not in the launcher

`bounds_m` in `bench/cost_model.jl` estimates the post-`--gamma-rtol` basis size
from a power law in separation, `bounds_m_ref * (sep / BOUNDS_M_REF_SEP)^exponent`,
floored at `bounds_m_floor`. That envelope was fitted on the 1 λ sweep's kept-count
curve, and it does not transfer. The 1/2 λ cube's measured kept counts are 1994 at
1/16 λ, 260 at 5/8 λ and 1351 at 5/2 λ -- not even monotone in separation, let
alone the same power law -- and a production 1/2 λ bounds job was cancelled at the
walltime having been sized for a much smaller basis than the 2049 it ran at.

The cost model's *per-index* cost is fine; the measured 8.3 s/index at m = 2049 is
comfortably inside what the model charges. It is the `m` fed into it that is wrong,
and no refit of a one-size envelope fixes that for every cube at once. So the
bounds stage becomes a second phase: the launcher submits greens and RSVD, and once
those have run, the spectrum on scratch answers the question the envelope was
guessing at. `kept` here is the same cut `load_bounds_inputs` applies, so it is the
`m` the job will really work at, not a proxy for it.

`--time-margin` (default 1.25) is on top of the cost model's own `time_pad`. It
covers the one thing a measured `m` still cannot see: a later change to the RSVD
(a different `q`, a different rank) shifts the kept count by a few percent, and a
bounds job that runs out of walltime loses everything it computed.

The truncation envelope is *disabled* for these predictions -- `bounds_m_mode` is
forced to `"fraction"` on a copy of the coefficients -- because `bounds_m` would
otherwise apply the 1 λ power-law cap on top of the measured count and shrink it
again. Everything else in the coefficient set is used exactly as calibrated.
"""

include(joinpath(@__DIR__, "cost_model.jl"))
using .CostModel
using JLD2
using Printf

# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

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

function parse_cells(s::AbstractString)
    parts = split(strip(s, ['(', ')', ' ']), ',')
    length(parts) == 3 || error("--cells expects three comma-separated integers")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])), parse(Int, strip(parts[3])))
end

parse_rational(s::AbstractString) =
    (p = split(strip(s), "//"); length(p) == 2 ? parse(Int, p[1]) // parse(Int, p[2]) : parse(Int, strip(s)) // 1)

"`--scale 1//32` or `--scale 1//32,1//16,1//16`."
function parse_scale(s::AbstractString)
    parts = split(strip(s, ['(', ')', ' ']), ',')
    length(parts) == 1 && return (parse_rational(parts[1]), parse_rational(parts[1]), parse_rational(parts[1]))
    length(parts) == 3 || error("--scale expects one or three //-rationals")
    return (parse_rational(parts[1]), parse_rational(parts[2]), parse_rational(parts[3]))
end

# --------------------------------------------------------------------------- #
# The GPU ladder, mirrored from create_jobs.jl's select_gpu
# --------------------------------------------------------------------------- #

"(name, capacity GB, compute fraction, host bundle GB), smallest first."
function gpu_options(cluster::AbstractString)
    cluster == "narval" && return [("a100_1g.5gb", 5, 1/8, 17), ("a100_2g.10gb", 10, 2/8, 35),
                                   ("a100_3g.20gb", 20, 3/8, 62), ("a100", 40, 1.0, 124)]
    cluster == "molering" && return [("a6000", 48, 1.0, 128)]
    cluster == "fir" && return [("h100", 80, 1.0, 124)]
    error("no GPU ladder for cluster '$cluster'; add one here to match create_jobs.jl's gpu_options")
end

const MIN_MEMORY_GB = 4
const MIN_TIME_S = 10 * 60

"""
    select_gpu(pt, coeffs, cluster, margin) -> NamedTuple

The smallest allocation the bounds job fits on, with its padded requests. Same
circular-choice loop as `create_jobs.jl`: the prediction sizes the request, the
request picks the card, and the card picks the algorithm, so each candidate is
predicted at its own capacity and the first that fits is taken.
"""
function select_gpu(pt::SRPoint, coeffs::Coefficients, cluster::AbstractString, margin::Real)
    candidate = nothing
    for (name, capacity_GB, fraction, bundle_host_GB) in gpu_options(cluster)
        p = predict(ComputeBounds, pt, coeffs; pad=true, vram_capacity_bytes=capacity_GB * 1e9)
        host_GB = max(MIN_MEMORY_GB, ceil(Int, p.host_bytes / 1e9))
        vram_GB = ceil(Int, p.vram_bytes / 1e9)
        t = (p.time_s - p.device_time_s) + p.device_time_s / fraction
        candidate = (gpu=name, host_GB=host_GB, vram_GB=min(vram_GB, capacity_GB),
                     time_s=max(MIN_TIME_S, ceil(Int, margin * t)), mode=p.mode,
                     fits=(vram_GB <= capacity_GB && host_GB <= bundle_host_GB))
        candidate.fits && return candidate
    end
    return candidate
end

seconds2string(t::Integer) = @sprintf("%02d:%02d:%02d", t ÷ 3600, (t % 3600) ÷ 60, t % 60)

# --------------------------------------------------------------------------- #
# Reading the spectrum
# --------------------------------------------------------------------------- #

"""
    gamma_kept_count(D, stored, gamma_rtol) -> Int

`_gamma_kept_count` from `src/bounds.jl`: sort descending, take the positive
prefix, keep those at or above `gamma_rtol * Γ[1]`. Duplicated rather than
imported, exactly as `bench/pick_bounds_points.jl` duplicates it, so this starts in
a second without loading CUDA. If the cut in `load_bounds_inputs` ever changes,
change it in all three places.
"""
function gamma_kept_count(D::AbstractVector, stored::Integer, gamma_rtol::Real)
    G = sort(Array(D); rev=true)
    isempty(G) && return 0
    G[1] > 0 || return 0
    return count(>=(gamma_rtol * G[1]), view(G, 1:min(Int(stored), length(G))))
end

"`nothing` when the output is not one a bounds job could run against."
function read_kept(jld_path::AbstractString, gamma_rtol::Real)
    vectors = replace(jld_path, r"\.jld$" => "_UR_asym_Vpos.h5")
    inline = false
    kept = stored = 0
    try
        jldopen(jld_path, "r") do io
            (haskey(io, "UR_asym/D") && haskey(io, "UR_asym/num_pos")) ||
                error("UR_asym/D or UR_asym/num_pos missing")
            # The dense-exact and in-memory branches store the block inline; only
            # the panel branch streams it to a sibling h5.
            inline = haskey(io, "UR_asym/V_pos")
            stored = Int(io["UR_asym/num_pos"])
            kept = gamma_kept_count(io["UR_asym/D"], stored, gamma_rtol)
        end
    catch err
        println(stderr, "  skip (unreadable: ", sprint(showerror, err), "): ", basename(jld_path))
        return nothing
    end
    if !inline && !isfile(vectors)
        println(stderr, "  skip (no inline V_pos and no _UR_asym_Vpos.h5): ", basename(jld_path))
        return nothing
    end
    kept > 0 || (println(stderr, "  skip (nothing survives the gamma cut): ", basename(jld_path)); return nothing)
    return (kept=kept, stored=stored)
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

"Same coefficients, with the separation-power-law cap on `m` switched off."
function without_truncation(c::Coefficients)
    fields = fieldnames(Coefficients)
    nt = NamedTuple{fields}(map(f -> getfield(c, f), fields))
    return Coefficients(; merge(nt, (bounds_m_mode="fraction",))...)
end

function main(argv::Vector{String})
    opts = parse_cli(argv)
    scratch = get(opts, "scratch", "")
    isempty(scratch) && error("--scratch <dir> is required")
    isdir(scratch) || error("--scratch is not a directory: $scratch")
    cells = parse_cells(get(opts, "cells", "32,32,32"))
    scale = parse_scale(get(opts, "scale", "1//32"))
    design = uppercase(get(opts, "design", "rs"))
    universe = String(sort(collect(design)))
    gamma_rtol = parse(Float64, get(opts, "gamma-rtol", "1e-12"))
    rank = parse(Int, get(opts, "rank", "4000"))
    oversamples = parse(Int, get(opts, "oversamples", "50"))
    power_iters = parse(Int, get(opts, "power-iters", "14"))
    cores = parse(Int, get(opts, "cores", "2"))
    cluster = get(opts, "cluster", "narval")
    margin = parse(Float64, get(opts, "time-margin", "1.25"))
    out = get(opts, "out", "")

    load_coefficients!(@__DIR__)
    coeffs = without_truncation(coefficients_for(cluster))
    coeffs.calibrated ||
        @warn "cluster '$cluster' has no calibrated cost model; these requests are analytic guesses"

    size_str = join(cells, "x")
    prefix = "$(size_str)__$(size_str)__"
    suffix = "__$(universe).jld"
    println(stderr, "Scanning $scratch for $(prefix)<n>ss<d>$(suffix)")
    names = sort([f for f in readdir(scratch) if startswith(f, prefix) && endswith(f, suffix)])
    isempty(names) && error("""
        No RSVD output matching $(prefix)<n>ss<d>$(suffix) in $scratch.
        Check --cells and --design against the sweep that ran, that this is the
        sweep's *scratch* directory and not its project directory, and that scratch
        has not been purged.""")

    lines = String[]
    println(stderr)
    @printf(stderr, "  %-14s %8s %8s  %-14s %6s %10s %s\n",
            "separation", "kept m", "stored", "gpu", "host", "time", "mode")
    n_over = 0
    for name in names
        mm = match(r"__(-?\d+)ss(\d+)__", name)
        mm === nothing && continue
        sep = parse(Int, mm.captures[1]) // parse(Int, mm.captures[2])
        r = read_kept(joinpath(scratch, name), gamma_rtol)
        r === nothing && continue
        pt = SRPoint(cells, cells; scale=scale, separation=sep, rank=rank,
                     oversamples=oversamples, power_iters=power_iters,
                     threads=cores, num_pos=r.kept)
        # A sanity check on the plumbing: with the envelope off, the model has to be
        # working at exactly the m we measured, or the request means nothing.
        bounds_m(pt, coeffs) == r.kept ||
            error("cost model resolved m = $(bounds_m(pt, coeffs)) for separation $sep but the spectrum says $(r.kept); the truncation cap is still being applied")
        s = select_gpu(pt, coeffs, cluster, margin)
        s.fits || (n_over += 1)
        push!(lines, join([string(numerator(sep)) * "//" * string(denominator(sep)),
                           string(r.kept), s.gpu, string(s.host_GB),
                           seconds2string(s.time_s), string(s.mode)], " "))
        @printf(stderr, "  %-14s %8d %8d  %-14s %5dG %10s %s%s\n",
                string(sep), r.kept, r.stored, s.gpu, s.host_GB,
                seconds2string(s.time_s), s.mode, s.fits ? "" : "  [nothing fits: largest allocation]")
    end

    isempty(lines) && error("found $(length(names)) JLD(s) but none is a complete RSVD output")
    n_over > 0 && @warn "$n_over separation(s) do not fit any allocation on $cluster; they are submitted on the largest one and may run out of memory."

    if isempty(out)
        foreach(println, lines)
    else
        mkpath(dirname(abspath(out)))
        open(out, "w") do io
            foreach(l -> println(io, l), lines)
        end
        println(stderr, "\nWrote $(length(lines)) sized bounds job(s) of $(length(names)) output(s) to $out")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
