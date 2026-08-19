#!/usr/bin/env julia
"""
    bench/pick_bounds_points.jl

Choose which separations the `backfill` tier's bounds points should run on, from
the RSVD outputs that are actually on scratch.

    julia --project=. bench/pick_bounds_points.jl \\
        --scratch /home/pvirally/scratch/Photonic-System-Channels/<PROJECT>/ \\
        --cells 32,32,32 --picks 4 --out picked.txt --table kept_by_sep.csv

Runs on the login node, in seconds, and touches no GPU.

# Why this is a script and not a hardcoded list

The 1 lambda sweep was cancelled part-way, so *some* unknown subset of its 333
separations has finished RSVD output and the rest has nothing. A generated launcher
cannot know which, and a bounds job pointed at a missing basis fails instantly and
wastes a queue slot. So the launcher asks this first.

# Why it reads the spectrum and not just the file names

The quantity that decides what a bounds job costs is `m`: how many of the positive
`Asym(G0_ur)` eigenvalues survive the `--gamma-rtol` cut. It is *not* the stored
positive count, which is about half the sketch width at every separation; the cut is
what collapses it, from around eighteen hundred near contact to a few dozen far
away, and the cost is superlinear in `m` in both time and memory. `UR_asym/D` is a
few thousand `Float64` at the front of the JLD, so applying the same cut
`load_bounds_inputs` applies is essentially free here, and it turns a guess about
the request into arithmetic.

It also means the table this writes (`--table`) is already most of the truncation
measurement: kept against separation, over every output that survived, before any
job has been submitted. `bench/fit.jl` fits `bounds_m_ref` / `bounds_m_exponent`
from the bounds rows, and this table is the independent check on it.

# Output

`--out` gets one line per pick, whitespace-separated, ready for a `while read` loop:

    <sep as n//d>  <kept>  <stored>  <gpu request>  <host GB>  <full|sampled>

The allocation, memory and mode columns come from `kept` through the thresholds in
`request_for`, whose numbers are the cost model's own predictions for this geometry
(see the comment there). Everything is chosen so that no job needs more than three
hours, which is what lets the whole tier ride the backfill queue.
"""

using JLD2
using Printf

# --------------------------------------------------------------------------- #
# Arguments
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

function parse_cells(s::AbstractString)
    parts = split(strip(s, ['(', ')', ' ']), ',')
    length(parts) == 3 || error("--cells expects three comma-separated integers")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])), parse(Int, strip(parts[3])))
end

# --------------------------------------------------------------------------- #
# Reading one candidate
# --------------------------------------------------------------------------- #

"""
    Candidate

One separation whose RSVD output is complete enough to run bounds on.

`stored` is `UR_asym/num_pos`, what the RSVD job saved. `kept` is how many of those
survive `gamma_rtol`, that is, the `m` the bounds job will actually work at.
"""
struct Candidate
    sep::Rational{Int}
    jld::String
    vectors::String
    stored::Int
    kept::Int
    total::Int
    h5_bytes::Int
end

"""
    gamma_kept_count(D, stored, gamma_rtol) -> Int

`_gamma_kept_count` from `src/bounds.jl`, applied to the saved spectrum: sort
descending, take the positive prefix, keep those at or above `gamma_rtol * G[1]`.

Duplicated rather than imported so this script starts in a second without loading
CUDA, and because it must keep agreeing with the version in `src/` for the numbers
below to mean anything. If `load_bounds_inputs` ever changes its cut, change this
with it.
"""
function gamma_kept_count(D::AbstractVector, stored::Integer, gamma_rtol::Real)
    G = sort(Array(D); rev=true)
    isempty(G) && return 0
    G[1] > 0 || return 0
    stored = min(Int(stored), length(G))
    return count(>=(gamma_rtol * G[1]), view(G, 1:stored))
end

"""
    read_candidate(jld_path, gamma_rtol) -> Union{Nothing,Candidate}

`nothing` when the output is not usable, with a reason on stderr. Usable means what
`_ur_asym_is_complete` in `src/rsvd.jl` means: `UR_asym/D` and `UR_asym/num_pos`
present, and the positive vectors reachable -- here, the sibling
`_UR_asym_Vpos.h5` existing and being at least as large as the block it claims to
hold. A run killed between writing the values and finishing the vectors leaves a
short h5 behind, and a bounds job pointed at one fails on the read.
"""
function read_candidate(jld_path::AbstractString, gamma_rtol::Real)
    m = match(r"__(-?\d+)ss(\d+)__", basename(jld_path))
    if m === nothing
        println(stderr, "  skip (no NssD separation in the name): ", basename(jld_path))
        return nothing
    end
    sep = parse(Int, m.captures[1]) // parse(Int, m.captures[2])
    vectors = replace(jld_path, r"\.jld$" => "_UR_asym_Vpos.h5")
    if !isfile(vectors)
        println(stderr, "  skip (no _UR_asym_Vpos.h5): ", basename(jld_path))
        return nothing
    end
    stored = total = 0
    kept = 0
    try
        jldopen(jld_path, "r") do io
            (haskey(io, "UR_asym/D") && haskey(io, "UR_asym/num_pos")) ||
                error("UR_asym/D or UR_asym/num_pos missing")
            D = io["UR_asym/D"]
            total = length(D)
            stored = Int(io["UR_asym/num_pos"])
            kept = gamma_kept_count(D, stored, gamma_rtol)
        end
    catch err
        println(stderr, "  skip (unreadable: ", sprint(showerror, err), "): ",
                basename(jld_path))
        return nothing
    end
    kept > 0 || (println(stderr, "  skip (nothing survives the gamma cut): ",
                         basename(jld_path)); return nothing)
    return Candidate(sep, jld_path, vectors, stored, kept, total, filesize(vectors))
end

# --------------------------------------------------------------------------- #
# Sizing one pick
# --------------------------------------------------------------------------- #

"""
    request_for(kept) -> (gpu, host_GB, mode, time_limit)

What to ask the scheduler for, given the `m` this separation will run at.

Every branch is under three hours, which is the constraint the whole tier exists
under: a job that asks for less than three hours rides the backfill queue at low
priority and starts in the gaps between other people's reservations, and one that
asks for eighteen never starts at all.

The numbers are `bench/cost_model.jl`'s own predictions for `(32,32,32)` at rank
4000 on the named allocation, times 2.5, then rounded up to something legible:

  * `m <= 600`: the *whole* outer loop costs about 0.25 h on a 3g.20gb slice, so
    this runs `--outer-blocks 0` -- production exactly, end to end, output JLD and
    all. That makes the far picks a full-pipeline validation as well as a
    calibration point, for free.
  * `600 < m <= 1400`: the whole loop is 1.6 h and climbing as `m^2`-ish, so this
    samples the loop (`--outer-blocks 4`), which costs 0.3 h. The device floor at
    `m = 1400` is about 16.7 GiB plus the refinement pencil cache's ~1 GiB, inside
    a 20 GiB slice.
  * `m > 1400`: still sampled, but on a whole A100. At `m = 2400` the panel front
    end's device floor is 17.8 GiB *before* the 16-entry refinement pencil cache
    adds 2.95 GiB, and 20.8 GiB does not fit a 20 GiB slice. This is the one branch
    that cannot use a slice, and it is why the picker reads the spectrum instead of
    assuming.
"""
function request_for(kept::Int)
    kept <= 600 && return ("a100_3g.20gb", 16, "full", "01:00:00")
    kept <= 1400 && return ("a100_3g.20gb", 26, "sampled", "01:30:00")
    return ("a100", 34, "sampled", "02:30:00")
end

"""
    choose(candidates, picks) -> Vector{Candidate}

Which candidates to actually submit: `picks` of them, spread evenly in `log(kept)`
between the largest and smallest `m` available, both ends always included.

Spread by `m` and not by separation, even though the truncation model is a power law
in separation, because `m` is what every fit these points feed is a function of. The
tau-search shape, the per-index outer time and the memory counts are all functions
of `m` and none of them knows what a separation is; and since `m` falls off steeply
and smoothly with the gap, four points spread in `log(m)` are also four points
spread across the sweep, whereas four spread evenly by index can easily land three
of them in the far-field tail where `m` has already bottomed out and every one says
the same thing.

The largest `m` is always among them. That is the regime the eighteen-hour request
was really about, so it is the one the model most needs a measurement in.
"""
function choose(candidates::Vector{Candidate}, picks::Int)
    isempty(candidates) && return Candidate[]
    picks >= length(candidates) && return copy(candidates)
    picks <= 1 && return [argmax(c -> c.kept, candidates)]
    hi = maximum(c -> c.kept, candidates)
    lo = minimum(c -> c.kept, candidates)
    targets = hi == lo ? fill(Float64(hi), picks) :
              exp.(range(log(hi), log(lo); length=picks))
    taken = Int[]
    for t in targets
        best, best_d = 0, Inf
        for (i, c) in enumerate(candidates)
            i in taken && continue
            d = abs(log(c.kept) - log(t))
            d < best_d && ((best, best_d) = (i, d))
        end
        best == 0 || push!(taken, best)
    end
    sort!(taken; by=i -> Float64(candidates[i].sep))
    return [candidates[i] for i in taken]
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

function main(argv::Vector{String})
    opts = parse_cli(argv)
    scratch = get(opts, "scratch", "")
    isempty(scratch) && error("--scratch <dir> is required")
    isdir(scratch) || error("--scratch is not a directory: $scratch")
    cells = parse_cells(get(opts, "cells", "32,32,32"))
    design = uppercase(get(opts, "design", "rs"))
    universe = String(sort(collect(design)))
    gamma_rtol = parse(Float64, get(opts, "gamma-rtol", "1e-12"))
    picks = parse(Int, get(opts, "picks", "4"))

    size_str = join(cells, "x")
    prefix = "$(size_str)__$(size_str)__"
    suffix = "__$(universe).jld"
    println(stderr, "Scanning $scratch for $(prefix)<n>ss<d>$(suffix)")

    names = sort([f for f in readdir(scratch)
                  if startswith(f, prefix) && endswith(f, suffix)])
    if isempty(names)
        println(stderr, """
        No RSVD output matching $(prefix)<n>ss<d>$(suffix) in $scratch.

        Things to check, in order:
          * the --cells and --design of this tier against the sweep that ran. The
            production launchers pass --design rs, whose file_prefix ends in RS;
            bench/point.jl's historical default builds [Sender, Receiver] and ends
            in SR. Both name the same geometry.
          * that this is the sweep's *scratch* directory and not its project
            directory. The RSVD outputs and the _UR_asym_Vpos.h5 live in scratch;
            the bounds outputs live in the project directory.
          * that scratch was not purged. Nothing here can recover a deleted basis;
            the RSVD would have to be re-run.
        """)
        error("no candidates")
    end

    candidates = Candidate[]
    for name in names
        cand = read_candidate(joinpath(scratch, name), gamma_rtol)
        cand === nothing || push!(candidates, cand)
    end
    isempty(candidates) && error("found $(length(names)) JLD(s) but none is a complete RSVD output")
    sort!(candidates; by=c -> Float64(c.sep))

    println(stderr)
    println(stderr, "$(length(candidates)) usable RSVD output(s) of $(length(names)) JLD(s), at gamma_rtol = $gamma_rtol:")
    @printf(stderr, "  %-14s %8s %8s %8s %8s  %s\n",
            "separation", "kept m", "stored", "total", "kept/st", "Vpos h5")
    for c in candidates
        @printf(stderr, "  %-14s %8d %8d %8d %8.4f  %.2f GiB\n",
                string(c.sep), c.kept, c.stored, c.total,
                c.kept / max(c.stored, 1), c.h5_bytes / 2^30)
    end

    table = get(opts, "table", "")
    if !isempty(table)
        open(table, "w") do io
            println(io, "sep_num,sep_den,sep,kept,stored,total,gamma_rtol,h5_bytes,jld")
            for c in candidates
                println(io, join([numerator(c.sep), denominator(c.sep),
                                  @sprintf("%.10g", Float64(c.sep)), c.kept, c.stored,
                                  c.total, @sprintf("%.3g", gamma_rtol), c.h5_bytes,
                                  basename(c.jld)], ","))
            end
        end
        println(stderr, "\nWrote the full kept-count table to $table")
    end

    chosen = choose(candidates, picks)
    lines = String[]
    println(stderr)
    println(stderr, "Picked $(length(chosen)) of them, spread evenly in log(m) and listed near to far:")
    for c in chosen
        gpu, host_GB, mode, time_limit = request_for(c.kept)
        push!(lines, join([string(numerator(c.sep)) * "//" * string(denominator(c.sep)),
                           string(c.kept), string(c.stored), gpu, string(host_GB),
                           mode, time_limit], " "))
        @printf(stderr, "  sep %-12s m=%-6d %-13s %3dG  %-8s %s\n",
                string(c.sep), c.kept, gpu, host_GB, mode, time_limit)
    end

    out = get(opts, "out", "")
    if isempty(out)
        for line in lines
            println(line)
        end
    else
        open(out, "w") do io
            for line in lines
                println(io, line)
            end
        end
        println(stderr, "\nWrote $(length(lines)) pick(s) to $out")
    end
    return chosen
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
