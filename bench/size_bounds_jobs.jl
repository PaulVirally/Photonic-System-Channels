#!/usr/bin/env julia
"""
    bench/size_bounds_jobs.jl

Size every bounds job in a sweep from the `m` its RSVD output actually produced,
rather than from the truncation envelope's guess. Writes one line per job:

    <key>  <kept m>  <gpu>  <host GB>  <HH:MM:SS>  <mode>  <m_aug>  <role>  <range>  <tag>

    julia --project=. bench/size_bounds_jobs.jl \\
        --scratch /home/pvirally/scratch/Photonic-System-Channels/<PROJECT>/ \\
        --cells 16,16,16 --scale 1//32 --rank 4000 --oversamples 50 \\
        --power-iters 14 --cluster narval --gamma-rtol 1.0e-12 \\
        --block-target-minutes 60 --out jobs/bounds_sizes_<PROJECT>.txt

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

# The Asym(G⁰ᵤᵤ) augmentation

`bounds_from_spectrum` augments the projection basis on points that keep fewer than
`--augment-threshold` directions, which is where the projected far-field dual had
stopped being a bound (`src/bounds.jl`, and `bench/augmented_basis_experiment.jl`
for the evidence). That costs a `reigen_hermitian` on `Asym(G⁰ᵤᵤ)` at width
`k_uu + 50` and turns every `m × m` pencil into an `m_aug × m_aug` one with
`m_aug = m + k_uu`, so a far-field job of thirteen minutes becomes one of
twenty-something.

This is exactly the question this file exists to answer, since it is the
*measured* `m` that decides whether a point augments, so the augmentation is **on
by default here**, at the same `--k-uu` / `--augment-threshold` the bounds job
defaults to. A sizer run and the job it sizes therefore agree without the launcher
having to pass the flags, so the existing `submit_bounds_*.sh` scripts need no
edits. `--k-uu 0` turns it off and reproduces the pre-augmentation
requests exactly.

`m_aug` is what the pencil stage really ran at: equal to the kept `m` on a point
that did not augment, and `m + k_uu` on one that did.

## The card can cut `k_uu` down, and this has to know

`--augment-threshold` caps `m` and so caps `m_aug = m + k_uu`. At the production
`--k-uu 512` and a threshold of 1000 that ceiling is `m_aug = 1512`, and three
`N_u × m_aug` matrices at the larger universes do not fit every allocation:
`clip_k_uu` in `src/bounds.jl` then reduces the effective `k_uu` to the largest one
that does. So `m_aug` is a property of the *point and the card*, not of the point,
and it is resolved inside `select_gpu`, once per candidate allocation, at that
candidate's capacity. The row's `m_aug` column is the chosen card's.

An allocation where not even `K_UU_CLIP_FLOOR` directions fit is one the job will
refuse to start on, so `select_gpu` treats it as not fitting and moves up the
ladder; a separation with no such allocation anywhere on the cluster is reported
and warned about rather than silently sized onto a card it will die on.

The clip arithmetic here (`augment_k_uu_cap` in `bench/cost_model.jl`) is the
runtime's (`max_k_uu_for_budget` in `src/bounds.jl`) line for line, and
`test/augmented_basis.jl` checks the two against each other. The one residual
difference is the budget they are handed: the ladder below names a card by its
marketing capacity while the job reads `CUDA.total_memory()`, which is about 6%
larger, so this can size a slightly smaller `m_aug` than the job runs at.
`--time-margin` is what covers that.

# Gap refinement, and why the key is what decides it

A separation under `MIN_GAP_CELLS` coarse x-cells can be refined at the facing
surfaces (`src/refinement.jl`), which makes each body a tiling, every Green
operator a block matrix over region pairs, and `N_u` larger (4.75x at the 1/4
lambda sweep's nearest gap, 1.94x at 1/2 lambda). Every `N_u`-scale term in the
request moves with it: the three `N_u x m_aug` matrices of the front end, the
`3m N_u` host block, the panel-path predicate, and the `k_uu` clip. A refined
point sized at the cuboid `N_u` asks for about a fifth of the memory it needs.

The refinement is **read off the scratch key, not assumed**. `file_prefix` in
`src/SMRSystems.jl` appends `__refF<factor>T<thickness>` to a refined point's
key and leaves an unrefined one exactly as it was, so the name on disk records
which mesh the RSVD really ran on. A near separation whose file carries no suffix
came off an ordinary run and has a cuboid basis; sizing it as refined would ask
for five times the memory *and* point the bounds job at a Green key that does not
exist. So a suffixed name is sized refined, an unsuffixed one is sized unrefined,
and the `(factor, thickness)` in the name is checked against the one the cost
model derives: a mismatch means the table has moved since the RSVD ran and the
basis on disk is not the one being sized for.

Refinement is off unless `--refine` is passed, so a suffixed output is the one
that needs the flag repeated on its bounds job. An unsuffixed near point needs no
flag at all.

# Splitting a long point into independent blocks

The outer loop over channel indices is embarrassingly parallel: index `n`'s bound
depends on nothing computed at any other `n`. On a low-priority account short jobs
backfill almost immediately and long ones sit, so a near-contact point predicted
at 8 hours is better run as B jobs of an hour that start at once, which collapses
the wall time to about one block plus the queue.

A point whose predicted request exceeds `--block-target-minutes` (default 60, `0`
disables splitting entirely) is therefore emitted as B block rows plus one merge
row:

  * a block row is `compute_bounds.jl ... --outer-range lo:hi --partial-suffix
    b<i>of<B>`, on the same card and memory as the unsplit job would have used;
  * the merge row is `bench/merge_bounds_blocks.jl`, a CPU job of minutes that
    assembles the blocks into the standard `<prefix>.jld`. It has to be submitted
    with `--dependency=afterok:` every block.

**Blocks are equal in time, not in count.** Index `n` probes `k = n, …, m`, so the
first index costs `m` times what the last one does. At the 1/2 λ, m = 2049 point
the model says 20 s at `n = 1` against 0.7 s at `n = m`. Equal-count blocks would
have the first finishing hours after the last. The split comes out of the cost
model's own decomposition: `bounds_time_s(...; indices=lo:hi)` charges the front
end in full and the per-index terms only over the slice, and it is exactly affine
in the slice, so `F` (front end), `α` (per index) and `β` (per probe) come from
three evaluations of it and the block boundaries are then a cumulative-sum split
of `w(n) = α + β(m − n + 1)`. Each block's `--time` is `F + Σ w(n)` over its own
indices, `--time-margin` applied on top, so every block carries the front-end
overhead it really pays.

B is chosen so that `F + W/B` fits the target rather than as `ceil(total/target)`,
because the front end is duplicated into every block. It is capped by
`--max-blocks` (default 24) and by `m`. A point whose front end alone exceeds the
target is reported and split as far as the cap allows.

Augmenting points are never split, whatever their predicted time: `uu_eigenbasis`
draws an unseeded random sketch, so two blocks of one augmented point would work
in two different bases and their bounds would not belong in the same file.
`bench/merge_bounds_blocks.jl` refuses such a merge; this refuses to create it. In
practice the constraint costs little (a point augments only when
`m < --augment-threshold`, and at `m` in the hundreds that is minutes of work),
but raising the threshold to 1000 does widen the band of points that are held
whole, so `n_aug_unsplit` in the summary is worth reading.

# The row format, and old submit scripts

Column 1 is the key a submit script looks a row up by. It is the bare separation
`1//32` for an unsplit point, and `1//32@b3of7` / `1//32@merge` for the rows of a
split one. That is deliberate: a submit script that predates the split does
`awk '\$1==sep'`, finds nothing for a split separation, and prints "skipped",
rather than matching block 1's row and submitting a whole-point job under a
one-block walltime. Regenerate the submit script, or pass
`--block-target-minutes 0`, to submit those points.

Columns 2-7 are unchanged and mean what they always did. Columns 8-10 are `role`
(`single`, `block` or `merge`), `range` (`lo:hi`, or the whole `1:m` on a `single`
and `merge` row) and `tag` (the `--partial-suffix`, `-` where there is none).

A submit script consuming the block rows looks like this, one `sbatch` per block,
then the merge behind `afterok`:

    blocks=\$(awk -v s="\$SEP" '\$1 ~ "^"s"@b" {print}' "\$SIZES")
    ids=()
    while read -r key m gpu mem time mode maug role range tag; do
        id=\$(sbatch --time=\$time --mem=\${mem}G --gpus=\$gpu:1 ... <<EOF
    srun julia --project=. -t \\\$SLURM_CPUS_PER_TASK compute_bounds.jl <point flags> \\
        --outer-range \$range --partial-suffix \$tag
    EOF
    )
        ids+=("\${id##* }")
    done <<< "\$blocks"
    sbatch --dependency=afterok:\$(IFS=:; echo "\${ids[*]}") --time=00:20:00 --mem=4G \\
           --cpus-per-task=2 <<EOF
    srun julia --project=. bench/merge_bounds_blocks.jl --project <dir> --prefix '<prefix>' --cleanup
    EOF
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
    cluster == "fir" && return [("nvidia_h100_80gb_hbm3_1g.10gb", 10, 1/8, 41),
                                ("nvidia_h100_80gb_hbm3_2g.20gb", 20, 2/8, 82),
                                ("nvidia_h100_80gb_hbm3_3g.40gb", 40, 3/8, 144),
                                ("h100", 80, 1.0, 288)]
    cluster == "nibi" && return [("h100_1g.10gb", 10, 1/8, 35), ("h100_2g.20gb", 20, 2/8, 71),
                                 ("h100_3g.40gb", 40, 3/8, 125), ("h100", 80, 1.0, 250)]
    cluster == "rorqual" && return [("h100_1g.10gb", 10, 1/8, 17), ("h100_2g.20gb", 20, 2/8, 35),
                                    ("h100_3g.40gb", 40, 3/8, 62), ("h100", 80, 1.0, 124)]
    cluster == "molering" && return [("a6000", 48, 1.0, 480)]
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

`capacity_GB` and `fraction` come back with the choice because the block sizer
below has to re-predict the *same* job over a slice of its outer loop, and a
prediction at a different capacity or a MIG stretch at a different fraction would
be a prediction of a different job. `augment` comes back for the same reason: the
effective `k_uu`, and so `m_aug`, is a property of the card, not of the point, so
the row's `m_aug` column has to be the chosen card's.

An allocation on which the augmentation is `:infeasible` (one where not even
`BOUNDS_K_UU_CLIP_FLOOR` directions fit, which is where `clip_k_uu` refuses to
start) does not count as fitting, whatever the memory arithmetic says. Sizing a
job onto a card it will refuse to run on is worse than sizing it onto the next one
up, and if nothing on the cluster is big enough the caller's existing "nothing
fits" warning is exactly the right report.
"""
function select_gpu(pt::SRPoint, coeffs::Coefficients, cluster::AbstractString, margin::Real)
    candidate = nothing
    for (name, capacity_GB, fraction, bundle_host_GB) in gpu_options(cluster)
        capacity_bytes = capacity_GB * 1e9
        p = predict(ComputeBounds, pt, coeffs; pad=true, vram_capacity_bytes=capacity_bytes)
        aug = bounds_augment(pt, coeffs, bounds_m(pt, coeffs);
                             vram_capacity_bytes=capacity_bytes)
        host_GB = max(MIN_MEMORY_GB, ceil(Int, p.host_bytes / 1e9))
        vram_GB = ceil(Int, p.vram_bytes / 1e9)
        t = (p.time_s - p.device_time_s) + p.device_time_s / fraction
        candidate = (gpu=name, host_GB=host_GB, vram_GB=min(vram_GB, capacity_GB),
                     time_s=max(MIN_TIME_S, ceil(Int, margin * t)), mode=p.mode,
                     capacity_GB=capacity_GB, fraction=fraction, augment=aug,
                     fits=(vram_GB <= capacity_GB && host_GB <= bundle_host_GB &&
                           aug.clip != :infeasible))
        candidate.fits && return candidate
    end
    return candidate
end

seconds2string(t::Integer) = @sprintf("%02d:%02d:%02d", t ÷ 3600, (t % 3600) ÷ 60, t % 60)

# --------------------------------------------------------------------------- #
# Splitting the outer loop into equal-time blocks
# --------------------------------------------------------------------------- #

const DEFAULT_BLOCK_TARGET_MINUTES = 60.0
# More blocks than this on one point stops being a queue optimization and starts
# being a way to run out of MaxSubmit: a 100-separation sweep at 24 blocks is
# already 2500 jobs. It is also the point where the duplicated front end starts to
# dominate what is left of each block.
const DEFAULT_MAX_BLOCKS = 24
# The merge is JLD2 reading and writing m-scale arrays. Seconds of work; the
# request is almost entirely Julia's startup and the filesystem.
const DEFAULT_MERGE_MINUTES = 20.0
const MERGE_MEMORY_GB = 4
# Slurm's MaxSubmit on narval is 1000 per account. Warn well short of it, since a
# sweep is usually not the only thing queued.
const JOB_COUNT_WARN = 600

"""
    block_time_s(pt, coeffs, sel, margin, indices) -> Int

What one `--outer-range` block of this point would request, in seconds: the cost
model over that slice, MIG-stretched on the chosen card exactly as `select_gpu`
does for the whole job, `margin` on top, floored at `MIN_TIME_S`.

`indices = 1:0` is the front end on its own, which is what a block pays before it
evaluates anything.
"""
function block_time_s(pt::SRPoint, coeffs::Coefficients, sel, margin::Real,
                      indices::AbstractUnitRange{Int})
    p = predict(ComputeBounds, pt, coeffs; pad=true,
                vram_capacity_bytes=sel.capacity_GB * 1e9, indices=indices)
    t = (p.time_s - p.device_time_s) + p.device_time_s / sel.fraction
    return max(MIN_TIME_S, ceil(Int, margin * t))
end

"Unfloored, unrounded seconds for the same slice; the arithmetic the split uses."
function block_seconds(pt::SRPoint, coeffs::Coefficients, sel, margin::Real,
                       indices::AbstractUnitRange{Int})
    p = predict(ComputeBounds, pt, coeffs; pad=true,
                vram_capacity_bytes=sel.capacity_GB * 1e9, indices=indices)
    return margin * ((p.time_s - p.device_time_s) + p.device_time_s / sel.fraction)
end

"""
    index_weights(pt, coeffs, sel, margin, m) -> (front_s, w)

The front end's seconds and the per-index seconds `w[n]`, read off the cost model
rather than assumed.

`bounds_time_s` is affine in the two counts a slice moves (how many indices it
sweeps, and how many probes those indices do) so the per-index cost is
`w(n) = α + β(m − n + 1)` exactly, and three evaluations pin it down: the front
end alone (`1:0`), the first index, and the last one. The caller checks the
reconstruction against the whole-loop prediction, which is the assertion that this
decomposition is the model's and not a story about it.
"""
function index_weights(pt::SRPoint, coeffs::Coefficients, sel, margin::Real, m::Int)
    front = block_seconds(pt, coeffs, sel, margin, 1:0)
    first_idx = block_seconds(pt, coeffs, sel, margin, 1:1) - front
    last_idx = block_seconds(pt, coeffs, sel, margin, m:m) - front
    β = m > 1 ? (first_idx - last_idx) / (m - 1) : 0.0
    α = last_idx - β
    return (front_s=front, w=[α + β * (m - n + 1) for n in 1:m], α=α, β=β)
end

"""
    equal_time_ranges(w, B) -> Vector{UnitRange{Int}}

`1:m` cut into `B` contiguous ranges of as nearly equal total weight as the
integer boundaries allow. Every block gets at least one index, so `B` is silently
capped at `m`.
"""
function equal_time_ranges(w::Vector{Float64}, B::Int)
    m = length(w)
    B = clamp(B, 1, m)
    B == 1 && return [1:m]
    cum = cumsum(w)
    total = cum[end]
    ranges = UnitRange{Int}[]
    lo = 1
    for b in 1:(B - 1)
        hi = searchsortedfirst(cum, total * b / B)
        hi = clamp(hi, lo, m - (B - b)) # leave one index for each remaining block
        push!(ranges, lo:hi)
        lo = hi + 1
    end
    push!(ranges, lo:m)
    return ranges
end

"""
    plan_blocks(pt, coeffs, sel, m, margin; target_s, max_blocks) -> NamedTuple or nothing

How to cut this point up, or `nothing` if it should stay one job.

`B` is chosen from `F + W/B <= target`, not from `total/target`: each block runs
the whole front end, so `F` is paid `B` times over and a naive `B` would leave
every block over the target by that much. When the front end alone is at or past
the target there is no `B` that reaches it; the point is split as far as the cap
allows and the caller says so.
"""
function plan_blocks(pt::SRPoint, coeffs::Coefficients, sel, m::Int, margin::Real;
                     target_s::Real, max_blocks::Int)
    target_s > 0 || return nothing
    total_s = block_seconds(pt, coeffs, sel, margin, 1:m)
    total_s <= target_s && return nothing
    m > 1 || return nothing

    weights = index_weights(pt, coeffs, sel, margin, m)
    reconstructed = weights.front_s + sum(weights.w)
    isapprox(reconstructed, total_s; rtol=1e-9) || error("""
        the per-index decomposition of bounds_time_s does not reproduce it:
        $(reconstructed) s from F + sum(w) against $(total_s) s from the whole-loop
        prediction. bounds_time_s has stopped being affine in the slice, through a new
        term that is neither front end nor per index, and index_weights here has to
        be rewritten to match before any block walltime can be trusted.""")

    loop_s = total_s - weights.front_s
    headroom = target_s - weights.front_s
    B = headroom > 0 ? ceil(Int, loop_s / headroom) : max_blocks
    B = clamp(B, 2, min(max_blocks, m))
    ranges = equal_time_ranges(weights.w, B)
    times = [block_time_s(pt, coeffs, sel, margin, r) for r in ranges]
    # `B` came from the average block, and the boundaries are integers, so the
    # longest block can land a little over the target where the average sits just
    # under it. One more block fixes that, and the loop cannot run away: it stops
    # at `max_blocks`, which the caller then reports as a block still over target.
    while maximum(times) > target_s && B < min(max_blocks, m)
        B += 1
        ranges = equal_time_ranges(weights.w, B)
        times = [block_time_s(pt, coeffs, sel, margin, r) for r in ranges]
    end
    return (ranges=ranges, times=times, front_s=weights.front_s, total_s=total_s,
            front_over_target=(headroom <= 0), longest_s=maximum(times))
end

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

"""
    output_pattern(cells, universe) -> Regex

What an RSVD output of this sweep is called on scratch, from `file_prefix` in
`src/SMRSystems.jl`. Captures the separation and, when the point was refined, the
`(factor, thickness)` of the mesh it ran on. The refinement group is optional, so
one pattern matches both a finished pre-refinement sweep and a refined near point.
"""
function output_pattern(cells::NTuple{3,Int}, universe::AbstractString)
    size_str = join(cells, "x")
    return Regex("^" * size_str * "__" * size_str * raw"__(-?\d+)ss(\d+)__" *
                 universe * raw"(?:__refF(\d+)T(\d+))?\.jld$")
end

"""
    scan_outputs(scratch, pattern) -> Vector{NamedTuple}

Every RSVD output in `scratch` this sweep wrote, as
`(name, sep, refined, factor, thickness)`, one per separation.

Where a separation has both a refined and an unrefined output (a scratch
directory that has seen a `--refine` run and a plain one) the refined one wins
and the other is reported. They would otherwise be emitted as two rows
under one key, and a submit script keyed on the separation would fire both.
"""
function scan_outputs(scratch::AbstractString, pattern::Regex)
    found = Dict{Rational{Int},Any}()
    shadowed = String[]
    for name in sort(readdir(scratch))
        m = match(pattern, name)
        m === nothing && continue
        sep = parse(Int, m.captures[1]) // parse(Int, m.captures[2])
        refined = m.captures[3] !== nothing
        row = (name=name, sep=sep, refined=refined,
               factor=refined ? parse(Int, m.captures[3]) : 0,
               thickness=refined ? parse(Int, m.captures[4]) : 0)
        prev = get(found, sep, nothing)
        if prev === nothing
            found[sep] = row
        elseif prev.refined == refined
            error("two outputs for separation $sep in $scratch: $(prev.name) and $(name)")
        else
            keep, drop = refined ? (row, prev) : (prev, row)
            found[sep] = keep
            push!(shadowed, drop.name)
        end
    end
    isempty(shadowed) || @warn "$(length(shadowed)) separation(s) have both a refined and an unrefined RSVD output on scratch; the refined one is sized and the other ignored. Delete the stale mesh, or size the two sweeps out of separate scratch directories." shadowed
    return [found[k] for k in sort(collect(keys(found)))]
end

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
    k_uu = parse(Int, get(opts, "k-uu", string(BOUNDS_K_UU_DEFAULT)))
    augment_threshold = parse(Int, get(opts, "augment-threshold",
                                       string(BOUNDS_AUGMENT_THRESHOLD_DEFAULT)))
    block_target_min = parse(Float64, get(opts, "block-target-minutes",
                                          string(DEFAULT_BLOCK_TARGET_MINUTES)))
    block_target_min >= 0 ||
        error("--block-target-minutes must be non-negative (0 disables splitting)")
    max_blocks = parse(Int, get(opts, "max-blocks", string(DEFAULT_MAX_BLOCKS)))
    max_blocks >= 2 || error("--max-blocks must be at least 2")
    merge_time_s = ceil(Int, 60 * parse(Float64, get(opts, "merge-minutes",
                                                     string(DEFAULT_MERGE_MINUTES))))
    out = get(opts, "out", "")

    load_coefficients!(@__DIR__)
    # Truncation envelope off (we measured `m`); augmentation on, at the bounds
    # job's own defaults, so the request matches what the job will do. See the
    # docstring.
    coeffs = with_augmentation(without_truncation(coefficients_for(cluster));
                               k_uu=k_uu, threshold=augment_threshold)
    coeffs.calibrated ||
        @warn "cluster '$cluster' has no calibrated cost model; these requests are analytic guesses"

    size_str = join(cells, "x")
    pattern = output_pattern(cells, universe)
    println(stderr, "Scanning $scratch for $(size_str)__$(size_str)__<n>ss<d>__$(universe)" *
            "[__refF<f>T<t>].jld")
    outputs = scan_outputs(scratch, pattern)
    isempty(outputs) && error("""
        No RSVD output matching $(pattern.pattern) in $scratch.
        Check --cells and --design against the sweep that ran, that this is the
        sweep's *scratch* directory and not its project directory, and that scratch
        has not been purged.""")

    lines = String[]
    println(stderr)
    println(stderr, k_uu > 0 ?
            "Augmentation ON: --k-uu $(k_uu), --augment-threshold $(augment_threshold); " *
            "points with m < $(augment_threshold) are sized at m_aug = m + $(k_uu)" :
            "Augmentation OFF (--k-uu 0): pre-augmentation requests")
    println(stderr, block_target_min > 0 ?
            "Block target: $(block_target_min) min per job; points predicted above it are " *
            "split into --outer-range blocks (max $(max_blocks)) plus a merge job" :
            "Block splitting OFF (--block-target-minutes 0): one job per separation")
    @printf(stderr, "  %-14s %8s %8s %8s %10s  %-14s %6s %10s %s\n",
            "separation", "kept m", "m_aug", "stored", "N_u", "gpu", "host", "time", "mode")
    n_over = 0
    n_refined = 0          # separations whose basis is on a refined mesh
    n_unrefined_near = 0   # ...and near ones whose basis is not
    n_split = 0            # separations emitted as blocks + merge
    n_front_over = 0       # ...of those, ones whose front end alone exceeds the target
    n_still_over = 0       # ...of those, ones whose longest block still exceeds it
    n_aug_unsplit = 0      # augmenting points left whole despite being over the target
    n_clipped = 0          # augmenting points whose k_uu the card cut down
    n_aug_infeasible = 0   # ...and ones where no card on the cluster can augment at all
    for out in outputs
        sep, name = out.sep, out.name
        r = read_kept(joinpath(scratch, name), gamma_rtol)
        r === nothing && continue
        # The mesh the RSVD ran on, off the key. See the docstring.
        pt = SRPoint(cells, cells; scale=scale, separation=sep, rank=rank,
                     oversamples=oversamples, power_iters=power_iters,
                     threads=cores, num_pos=r.kept, refine_gap=out.refined)
        ref = refinement_of(pt)
        if out.refined
            n_refined += 1
            ref === nothing &&
                error("$(name) is keyed as a refined output but the cost model derives no refinement for separation $sep at scale $(scale[1]); GAP_REFINEMENT_TABLE has moved since the RSVD ran and the basis on disk is not the one this would size for")
            (ref.factor == out.factor && ref.thickness == out.thickness) ||
                error("$(name) was built at (factor $(out.factor), thickness $(out.thickness)) but the cost model derives (factor $(ref.factor), thickness $(ref.thickness)) for separation $sep; GAP_REFINEMENT_TABLE has moved since the RSVD ran and the basis on disk is not the one this would size for")
        elseif gap_refinement(sep, scale[1]) !== nothing
            n_unrefined_near += 1
        end
        # A sanity check on the plumbing: with the envelope off, the model has to be
        # working at exactly the m we measured, or the request means nothing.
        bounds_m(pt, coeffs) == r.kept ||
            error("cost model resolved m = $(bounds_m(pt, coeffs)) for separation $sep but the spectrum says $(r.kept); the truncation cap is still being applied")
        s = select_gpu(pt, coeffs, cluster, margin)
        s.fits || (n_over += 1)
        # The chosen card's shape, not the point's: `clip_k_uu` makes the effective
        # k_uu a function of the allocation, so asking again without a capacity
        # would print an m_aug the job will not run at.
        aug = s.augment
        aug.clip == :budget && (n_clipped += 1)
        aug.clip == :infeasible && (n_aug_infeasible += 1)
        sep_str = string(numerator(sep)) * "//" * string(denominator(sep))
        whole = 1:r.kept

        # An augmenting point runs a fresh unseeded Asym(G⁰ᵤᵤ) sketch in every
        # process, so its blocks would not share a basis. See the docstring; the
        # merge refuses these too.
        plan = aug.augmented ? nothing :
               plan_blocks(pt, coeffs, s, r.kept, margin;
                           target_s=60 * block_target_min, max_blocks=max_blocks)
        if isnothing(plan) && aug.augmented && block_target_min > 0 &&
           s.time_s > 60 * block_target_min
            n_aug_unsplit += 1
        end

        row(key, time_s, role, idx_range, tag, gpu, host_GB) =
            join([key, string(r.kept), gpu, string(host_GB), seconds2string(time_s),
                  string(s.mode), string(aug.m_aug), role,
                  "$(first(idx_range)):$(last(idx_range))", tag], " ")

        # The augmentation note on the console line. Silent when the point does not
        # augment, `aug` when it does at the requested k_uu, and the arithmetic when
        # the card took some of it away. A clipped point's bound is not comparable
        # with an unclipped one's, so it has to be visible in the sizer's own output
        # and not only in the job's log.
        aug_note = if !aug.augmented
            ""
        elseif aug.clip == :budget
            "  aug [k_uu clipped $(aug.k_uu_requested) → $(aug.k_uu) by $(s.gpu)]"
        elseif aug.clip == :infeasible
            "  aug [INFEASIBLE: no allocation on $cluster fits k_uu ≥ $(BOUNDS_K_UU_CLIP_FLOOR); this job will refuse to start]"
        else
            "  aug"
        end

        # The mesh note on the console line: silent unless the basis on scratch is
        # refined, in which case the bounds job has to carry --refine to look for
        # the Green blocks that RSVD wrote.
        mesh_note = out.refined ?
                    "  [refined F$(out.factor)T$(out.thickness): submit this one with --refine]" : ""

        if isnothing(plan)
            push!(lines, row(sep_str, s.time_s, "single", whole, "-", s.gpu, s.host_GB))
            @printf(stderr, "  %-14s %8d %8d %8d %10d  %-14s %5dG %10s %s%s%s%s%s\n",
                    sep_str, r.kept, aug.m_aug, r.stored, universe_length(pt), s.gpu,
                    s.host_GB, seconds2string(s.time_s), s.mode, aug_note, mesh_note,
                    s.fits ? "" : "  [nothing fits: largest allocation]",
                    (aug.augmented && block_target_min > 0 && s.time_s > 60 * block_target_min) ?
                    "  [over target, not split: augmented]" : "")
        else
            n_split += 1
            plan.front_over_target && (n_front_over += 1)
            plan.longest_s > 60 * block_target_min && (n_still_over += 1)
            B = length(plan.ranges)
            for (i, rng) in enumerate(plan.ranges)
                tag = "b$(i)of$(B)"
                push!(lines, row("$(sep_str)@$(tag)", plan.times[i], "block", rng, tag,
                                 s.gpu, s.host_GB))
            end
            push!(lines, row("$(sep_str)@merge", merge_time_s, "merge", whole, "-",
                             "cpu", MERGE_MEMORY_GB))
            @printf(stderr, "  %-14s %8d %8d %8d %10d  %-14s %5dG %10s %s%s  [%d blocks of %s..%s + merge, front end %s each]%s\n",
                    sep_str, r.kept, aug.m_aug, r.stored, universe_length(pt), s.gpu,
                    s.host_GB, seconds2string(s.time_s), s.mode, mesh_note, B,
                    seconds2string(minimum(plan.times)), seconds2string(maximum(plan.times)),
                    seconds2string(ceil(Int, plan.front_s)),
                    s.fits ? "" : "  [nothing fits: largest allocation]")
        end
    end

    isempty(lines) && error("found $(length(outputs)) JLD(s) but none is a complete RSVD output")
    n_refined > 0 && println(stderr, "\n$(n_refined) separation(s) are on a refined mesh; their N_u is the tiling's, not the cuboid's, and their bounds jobs need --refine.")
    n_unrefined_near > 0 && println(stderr, "$(n_unrefined_near) separation(s) are inside $(MIN_GAP_CELLS) coarse cells of gap on the plain cuboid mesh, so their gap carries Gila's quadrature error. They are sized for the mesh they ran on.")
    n_over > 0 && @warn "$n_over separation(s) do not fit any allocation on $cluster; they are submitted on the largest one and may run out of memory."
    n_front_over > 0 && @warn "$n_front_over split separation(s) have a front end (Gram-Schmidt, projections, Green sweep) that costs as much as the whole --block-target-minutes on its own, so no number of blocks reaches the target. They are split to --max-blocks $(max_blocks) and each block is front end plus a share of the loop."
    n_still_over > 0 && @warn "$n_still_over split separation(s) still have a block above the target, either because of the --max-blocks $(max_blocks) cap or because of the duplicated front end. Raise --max-blocks, or accept the longer block."
    n_aug_unsplit > 0 && @warn "$n_aug_unsplit augmenting separation(s) are over the target but are left as single jobs: uu_eigenbasis draws an unseeded Asym(G_uu) sketch, so two blocks of one augmented point would work in different bases. If one of these is genuinely hours long, the fix is a seeded sketch, not a split."
    n_clipped > 0 && @warn "$n_clipped augmenting separation(s) have their --k-uu clipped by the card they were sized onto: the dense augmented front end is (2*m_aug + m) * N_u * 16 bytes and at these m it does not fit the allocation at the full --k-uu $(k_uu). clip_k_uu in src/bounds.jl applies the same cut at runtime and records it in augment/k_uu_effective, so the request and the job agree, but the bounds at those separations are computed in a smaller augmented basis than the rest of the sweep. The k_uu scan is monotone and saturating so most of the repair survives; if these points matter, run them on a larger card or lower --augment-threshold below their m."
    n_aug_infeasible > 0 && @warn "$n_aug_infeasible separation(s) cannot augment at all on $cluster: no allocation fits even k_uu = $(BOUNDS_K_UU_CLIP_FLOOR), which is where clip_k_uu refuses to start rather than compute a bound that is still invalid. Those jobs will exit in their first minute. Lower --augment-threshold below their kept m so they are not augmented, run them on a cluster with a larger card, or accept --k-uu 0 for them."
    if n_split > 0
        println(stderr)
        @warn "$n_split separation(s) are split into blocks. Their rows are keyed '<sep>@b<i>of<B>' and '<sep>@merge', not '<sep>', so a submit script written before the split existed will report them as skipped rather than submit a whole-point job under a one-block walltime. Regenerate the submit script (see this file's docstring for the sbatch/afterok shape), or pass --block-target-minutes 0."
    end

    # One row is one sbatch, blocks and merges included, so this is the sweep's
    # job count and the number Slurm's MaxSubmit is counted against.
    n_jobs = length(lines)
    # How many of them are blocks, counted off the rows themselves.
    n_block_rows = count(l -> split(l)[8] == "block", lines)
    println(stderr)
    println(stderr, "Total jobs: $(n_jobs) ($(n_jobs - n_block_rows - n_split) single" *
            (n_split > 0 ? ", $(n_block_rows) block over $(n_split) split separation(s), $(n_split) merge" : "") * ")")
    n_jobs > JOB_COUNT_WARN && @warn "$(n_jobs) jobs is close to narval's MaxSubmit of 1000 per account, and that limit counts everything you have queued, not just this sweep. Submit in batches, raise --block-target-minutes, or lower --max-blocks."

    if isempty(out)
        foreach(println, lines)
    else
        mkpath(dirname(abspath(out)))
        open(out, "w") do io
            foreach(l -> println(io, l), lines)
        end
        println(stderr, "Wrote $(n_jobs) sized bounds job(s) from $(length(outputs)) output(s) to $out")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
