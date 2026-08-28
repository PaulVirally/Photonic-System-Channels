#!/usr/bin/env julia
"""
    bench/plan_refined.jl

Generate the `refined` calibration tier: the points that measure what gap
refinement costs, which `bench/cost_model.jl` predicts and nobody has yet timed.

    julia bench/plan_refined.jl --cluster narval
    julia bench/plan_refined.jl --cluster narval --dry-run

Writes `bench/launch_calibration_<cluster>_refined.sh` and
`bench/manifest_<cluster>_refined.csv`, and prints the copy/run instructions. It
submits nothing.

A separate file from `bench/plan.jl` rather than a sixth tier inside it, because
the four calibration tiers and the two trial tiers there are all fitted into one
`plan_points` dispatch and this asks a narrower question than any of them. It
reuses that file's `ClusterSpec`, `PlannedPoint`, `point_command`,
`resource_lines` and `merge_block`, so a point here submits exactly the way a
point there does and `bench/point.jl` cannot tell the difference.

# What each point identifies

Three provisional coefficients came in with gap refinement, all defaulting to
1.0, all charged only on a refined point (see the `Coefficients` docstring in
`bench/cost_model.jl`). Everything below exists to move one of them.

  * `g0_sandwich_scale` and `g0_partition_scale`, from the `stage_greens` points.
    A refined build is a sum over region-pair blocks of five kinds, and the model
    already knows the exact shape and count of each; what it does not know is
    whether the fitted `g0_contact_*` and `g0_ext_*` triples, measured on
    whole-body blocks, still hold on a two-cell contact layer remeshed at the
    fine scale and on a partitioned quadrature that pays its fixed cost `pairs`
    times over.

    Two unknowns, so what matters is how widely the points spread the ratio of
    the two bracketed work sums. Under narval's own coefficients the six refined
    builds below span `C_sandwich / C_partition` from 3.9 down to 0.11, a
    thirty-five-fold lever arm: `(8,8,8)` at `g = 1` is the one geometry whose
    slab swallows the whole body, so it has six sandwich blocks, six partition
    blocks and no coarse bulk at all, while `(64,32,32)` at `g = 1` is six and
    eighteen on a body that is almost entirely bulk. `C_sandwich` itself barely
    moves across the whole set -- it is six blocks times a fixed cost -- so it is
    the small end that identifies it and the large end that identifies the other.

  * `mv_composite_scale`, from the `matvec_*` points. A composite matvec is a
    loop over region-pair blocks, and the model charges each block's FFT work and
    each block's launch overhead separately, so what is left for the multiplier is
    `_cmpMul`'s own bookkeeping: the output allocation, the per-region reshapes
    and views, and the accumulation over the block row. These are the cheapest
    points here, seconds apiece, and the only ones that measure the multiplier
    without the RSVD's dense work on top.

  * The `stage_rsvd` and `stage_bounds` points are the assembled check. They are
    the only two that exercise a refined `N_u` end to end, including the folded
    `Asym(G⁰ᵤᵤ)` and `Asym(G⁰ᵣᵣ)` applies that `src/rsvd.jl`'s `hasmethod` shim
    now takes on a composite operator (`asym(::GlaCmpOprVac)` landed upstream at
    rev d4c0516). That halving is why `REFINED_ASYM_SELF_APPLIES` is 1 and
    `uu_asym_applies` is 1 on a refined point, and no primitive point can see it:
    it lives in the shim and the apply count, not in the operator.

# Controls

Every geometry here also runs at `g = MIN_GAP_CELLS`, one coarse cell past the
threshold, where `gap_refinement` returns `nothing` and the job builds the plain
cuboids. Those rows carry no cross-scale blocks at all, so `measured - predicted`
at a control is the residual of the *already fitted* part of the model at these
shapes. Without them a systematic miss in the base terms would be absorbed
straight into the two new multipliers, and the tier would be reporting a
difference between two coefficient sets rather than a measurement.

The controls pass `--refine` too. The flag is then not a variable anywhere in the
tier: every job runs production's mesh policy and the separation alone decides
whether anything is refined, which is exactly the production sweep's own
arrangement. `refinement_of` agrees -- it returns `nothing` at `g >= MIN_GAP_CELLS`
whatever `refine_gap` says -- so a control row is predicted with the pre-refinement
arithmetic term for term even though its `refine` extra reads 1.

# Sizes and the three-hour box

Every point is predicted under three hours on narval's own coefficients, which is
the whole reason the RSVD and bounds points run at `k = 800` rather than at the
production `k = 4000`: the multipliers being fitted are per-matvec and per-block,
and neither moves with the rank. `q` is production's 6. The bounds point samples
its outer loop (`--outer-blocks` / `--outer-block-len`) exactly as the backfill
tier does, so its cost is the front end -- the Gram-Schmidt, the projections and
the Green sweep -- which is the part refinement moves.

`REFINED_CHI` is the production sweep's resonant Germanium rather than
`DEFAULT_CHI`. The Green builds and the RSVD do not care: the operators are
vacuum, and `χ` enters `_generate_rsvd_sr` only as the scalar shift
`imag(inv(χ)) * I`, which changes no count. It matters for the bounds point,
whose `num_pos` and `Γ` truncation are properties of that spectrum, and costs
nothing to carry on the others.

# One anisotropic geometry

`(64,32,32)` at `scale = -1//8` is here because the anisotropic cross-scale
jacobian is the path that was broken until d4c0516, and so the least
characterized. An earlier version of this file claimed the tier could not reach it
because `bench/point.jl` takes a single rational `--scale`; that was wrong.
`SMRSystem`'s negative-scale convention is exactly what a single rational is for,
`getopt_rational` parses `-1//8`, `point_command` quotes it, and `row_to_srpoint`
already reads it back. Refinement is x-only and `gap_refinement` reads `scale[1]`,
which stays `1//32`, so these separations refine there as they do everywhere else.

# Fitting the result

`bench/fit.jl` has no reader for this tier yet. The greens rows give a
two-parameter linear least squares against the model's own per-kind counts,

    measured − predicted_without_crossscale
        = s_sandwich · C_sandwich + s_partition · C_partition

with `C_*` the bracketed sums `greens_time_s` already forms
(`g0_contact_fft · sandwich_fft_work + g0_contact_cell · sandwich_cells / eta +
g0_contact_fixed · n_sandwich_blocks`, and its `ext` counterpart), and the control
rows fixing the unrefined intercept. `mv_composite_scale` is one ratio per matvec
row, measured over `rs_apply_work` / `rr_apply_work` / `universe_apply_work`.

Fit the multipliers before reading anything into `rsvd_pass_scale`. That fit
divides a measured `stage_rsvd` time by the model's own per-pass prediction, and
on a refined row the prediction carries `mv_composite_scale`; run it against a
tier where the multiplier is still 1.0 and the composite's surcharge lands in
`rsvd_pass_scale` instead, where every unrefined point then pays it too.
"""

include(joinpath(@__DIR__, "plan.jl"))

using Printf
using Dates

# --------------------------------------------------------------------------- #
# The points
# --------------------------------------------------------------------------- #

# Resonant Germanium, zeta = 225: the production sweep's own susceptibility. See
# the docstring for why the cheap points carry it anyway.
const REFINED_CHI = "18.612+1.5502660841406073im"
const REFINED_SEED = 20260828
# `k` well under production: see the docstring. Neither multiplier being fitted
# moves with the rank, and this is what keeps every point inside the box. `q` is
# production's.
const REFINED_RANK = 800
const REFINED_POWER_ITERS = 6
const REFINED_OVERSAMPLES = 50
const REFINED_GAMMA_RTOL = "1.0e-12"
# The sampled outer loop of the bounds point, the same shape the backfill tier
# uses: four blocks of 24 consecutive indices, spread over the loop.
const REFINED_OUTER_BLOCKS = 4
const REFINED_OUTER_BLOCK_LEN = 24
# Nothing here may ask for more than this. A calibration point that is killed at
# the walltime measures nothing at all.
const REFINED_MAX_TIME_S = 3 * 3600
# The first separation the job leaves alone, in wavelengths at the production
# x scale. Every geometry runs here as its own control.
const REFINED_CONTROL_SEP = MIN_GAP_CELLS // 32

"""
    REFINED_BUILDS

The `stage_greens` points, as `(tag, cells, scale, separation)`.

The separations are production's own: of the hundred-point log grid a sweep runs,
exactly the five under `MIN_GAP_CELLS` coarse cells are refined, and they land on
four distinct `GAP_REFINEMENT_TABLE` entries -- `g = 1` is `(6,6)`, `g = 2` is
`(3,4)`, `g = 3` is `(2,4)`, and `g = 4` and `g = 5` share `(2,2)`. All four are
sampled here, so a structural miscount in any table row would show up as a
residual rather than hide inside the two multipliers.

`0p25` and `1lam` are the small end, where the sandwich bracket is the larger
share; `2lam` is the anisotropic geometry. Each appears once more at
`REFINED_CONTROL_SEP` as its own control, added by `plan_refined_points` rather
than listed, so a geometry cannot arrive here without one.
"""
const REFINED_BUILDS = (
    (:c0p25, (8, 8, 8), 1 // 32, 1 // 32),      # g = 1, table (6,6): slab is the whole body
    (:c0p25, (8, 8, 8), 1 // 32, 3 // 32),      # g = 3, table (2,4)
    (:c1lam, (32, 32, 32), 1 // 32, 1 // 32),   # g = 1, table (6,6)
    (:c1lam, (32, 32, 32), 1 // 32, 1 // 16),   # g = 2, table (3,4)
    (:c2lam, (64, 32, 32), -1 // 8, 1 // 32),   # g = 1, anisotropic
    (:c2lam, (64, 32, 32), -1 // 8, 5 // 32),   # g = 5, table (2,2), anisotropic
)

"The geometries, in the order their controls are emitted."
const REFINED_GEOMETRIES = (
    (:c0p25, (8, 8, 8), 1 // 32),
    (:c1lam, (32, 32, 32), 1 // 32),
    (:c2lam, (64, 32, 32), -1 // 8),
)

# The geometry the device points run on. `1lam` rather than `2lam`: it is a
# production cube, its refined RSVD fits the card with room to spare at k = 800,
# and its composite operators already have the nine-block-per-pair structure that
# `mv_composite_scale` is about. The matvec points run at both the refined and the
# control separation, which is what makes the multiplier a ratio of two measured
# numbers rather than a ratio to a prediction.
const REFINED_DEVICE = (:c1lam, (32, 32, 32), 1 // 32)
const REFINED_DEVICE_SEP = 1 // 32

"`scale` as `SRPoint` wants it, from the single rational `SMRSystem` and the CLI take."
srpoint_scale(s::Rational{Int}) =
    s < 0 ? (1 // 32, abs(s), abs(s)) : (s, s, s)

srpoint(cells, scale, sep, threads) =
    SRPoint(cells, cells; scale=srpoint_scale(scale), separation=sep,
            rank=REFINED_RANK, oversamples=REFINED_OVERSAMPLES,
            power_iters=REFINED_POWER_ITERS, threads=threads, refine_gap=true)

"The `bench/point.jl` flags every point of this tier shares."
common_args(cells, scale, sep) =
    ["--cells", join(cells, ","), "--scale", rat(scale),
     "--chi", REFINED_CHI, "--sep", rat(sep),
     "--rank", string(REFINED_RANK), "--oversamples", string(REFINED_OVERSAMPLES),
     "--power-iters", string(REFINED_POWER_ITERS), "--seed", string(REFINED_SEED),
     "--refine"]

"The label fragment for a separation: its gap in coarse x-cells at the production scale."
gap_cells(sep::Rational{Int}) = numerator(sep * 32)

"""
    matvec_predicted_s(kind, pt, coeffs, reps) -> Float64

What a `matvec_*` point should take, wall clock. `CostModel` has no `predict` for
one -- the enum is the three pipeline jobs -- so the applies are priced from the
same `*_apply_work` regressors `rsvd_time_s` uses and the two startup terms are
added, which on these points is nearly the whole bill: a refined `G⁰ᵣₛ` apply at
`1lam` is seventy milliseconds against a minute of Julia and CUDA coming up.

`--reps` plus `warmup=2` applies, and the multiplier stays at whatever
`mv_composite_scale` currently is, which is the point: a badly wrong 1.0 shows up
here as a badly wrong estimate on a job that is dominated by startup anyway.
"""
function matvec_predicted_s(kind::AbstractString, pt::SRPoint, c::Coefficients,
                            reps::Int)
    mvs = CostModel.composite_mv_scale(pt, c)
    per_apply = if kind == "matvec_ext"
        w = CostModel.rs_apply_work(pt)
        mvs * (c.mv_ext_fft * w.fft + c.mv_ext_fixed * w.blocks)
    elseif kind == "matvec_self"
        w = CostModel.rr_apply_work(pt)
        mvs * (c.mv_self_fft * w.fft + c.mv_self_fixed * w.blocks)
    else
        w = CostModel.universe_apply_work(pt)
        mvs * (2 * (c.mv_self_fft * w.self_fft + c.mv_self_fixed * w.self_blocks) +
               2 * (c.mv_ext_fft * w.ext_fft + c.mv_ext_fixed * w.ext_blocks))
    end
    return c.g0_startup_s + c.gpu_startup_s + (reps + 2) * per_apply
end

"`bench/point.jl`'s own `--reps` default, which this tier does not override."
const REFINED_MATVEC_REPS = 20

"""
    boxed_time(predicted_s, factor) -> Int

A time limit from a prediction: `factor` times it, floored at ten minutes so a
seconds-long point still gets a usable limit, and capped at
`REFINED_MAX_TIME_S`. The cap is a hard requirement of this tier, so a point that
hits it is reported by `refined_main` rather than silently truncated.
"""
boxed_time(predicted_s::Real, factor::Real) =
    clamp(ceil(Int, factor * predicted_s), 600, REFINED_MAX_TIME_S)

function plan_refined_points(cluster::ClusterSpec)
    threads = min(cluster.max_cores, 12)
    coeffs = coefficients_for(cluster.name)
    capacity = cluster.max_vram_GB * 1e9
    points = PlannedPoint[]
    greens_label = Dict{Tuple{Symbol,Rational{Int}},String}()

    # ---- the refined Green builds, and one control per geometry ------------
    builds = vcat(collect(REFINED_BUILDS),
                  [(tag, cells, scale, REFINED_CONTROL_SEP)
                   for (tag, cells, scale) in REFINED_GEOMETRIES])
    for (tag, cells, scale, sep) in builds
        label = "refg_$(tag)_g$(gap_cells(sep))"
        pt = srpoint(cells, scale, sep, threads)
        p = predict(GenerateGreens, pt, coeffs; pad=true)
        # Twice the padded prediction, which is three times the raw one. These are
        # the points whose cost the tier exists to discover, so the limit cannot be
        # tight around a number the model is admitting it does not know.
        push!(points, PlannedPoint(label, "stage_greens",
                                   common_args(cells, scale, sep), threads,
                                   min(cluster.max_host_GB,
                                       max(8, ceil(Int, 3 * p.host_bytes / 2^30))),
                                   boxed_time(p.time_s, 2.0), false, nothing,
                                   nothing, p.time_s, 0.0))
        greens_label[(tag, sep)] = label
    end

    # ---- the composite matvecs, and their unrefined controls ---------------
    #=
    Seconds each: `bench/point.jl` times `--reps` applications of an operator that
    is already on disk. They depend on the Green build of the same geometry and
    gap, so the blocks are there to load rather than rebuilt inside a device job.
    Ten minutes and a whole card is the smallest useful ask.
    =#
    dev_tag, dev_cells, dev_scale = REFINED_DEVICE
    for sep in (REFINED_DEVICE_SEP, REFINED_CONTROL_SEP)
        dep = greens_label[(dev_tag, sep)]
        mv_pt = srpoint(dev_cells, dev_scale, sep, threads)
        for kind in ("matvec_ext", "matvec_self", "matvec_uu")
            short = replace(kind, "matvec_" => "mv_")
            pm = matvec_predicted_s(kind, mv_pt, coeffs, REFINED_MATVEC_REPS)
            push!(points, PlannedPoint("ref$(short)_$(dev_tag)_g$(gap_cells(sep))",
                                       kind, common_args(dev_cells, dev_scale, sep),
                                       threads, 16, 600, true, dep, nothing, pm, 1.0))
        end
    end

    # ---- one refined RSVD, and the bounds job that reads it ----------------
    dev_pt = srpoint(dev_cells, dev_scale, REFINED_DEVICE_SEP, threads)
    dev_dep = greens_label[(dev_tag, REFINED_DEVICE_SEP)]
    rsvd_label = "refrsvd_$(dev_tag)_g$(gap_cells(REFINED_DEVICE_SEP))"
    rsvd_scratch = "\$CAL_ROOT/refined/$(rsvd_label)"
    #=
    1.75 rather than the 1.5 the padded prediction already carries. Roughly half
    this point's raw time is Green matvecs, and on a refined operator every one of
    them is priced through `mv_composite_scale`, which is 1.0 because nobody has
    measured it. The wider box is what keeps the point alive if the true
    multiplier turns out to be three.
    =#
    pr = predict(GenerateRSVD, dev_pt, coeffs; pad=true, vram_capacity_bytes=capacity)
    push!(points, PlannedPoint(rsvd_label, "stage_rsvd",
                               vcat(common_args(dev_cells, dev_scale, REFINED_DEVICE_SEP),
                                    ["--scratch", rsvd_scratch, "--fresh"]),
                               threads,
                               min(cluster.max_host_GB,
                                   max(8, ceil(Int, 1.5 * pr.host_bytes / 2^30))),
                               boxed_time(pr.time_s, 1.75), true, dev_dep,
                               nothing, pr.time_s, 1.0))

    #=
    The bounds point reads the RSVD point's scratch and waits on it, the same
    wiring the funicular tier's E4 chain uses.

    2.5, the widest box here, for a reason that has nothing to do with refinement.
    This point's cost is quadratic in `m`, and `m` is `num_pos`, which the model
    only guesses (`NUM_POS_FRACTION`, 0.6 of the rank) until the RSVD that runs
    just before it reports the real number. A point whose own dependency decides
    its cost cannot have a tight limit.
    =#
    pb = predict(ComputeBounds, dev_pt, coeffs; pad=true, vram_capacity_bytes=capacity)
    push!(points, PlannedPoint("refbounds_$(dev_tag)_g$(gap_cells(REFINED_DEVICE_SEP))",
                               "stage_bounds",
                               vcat(common_args(dev_cells, dev_scale, REFINED_DEVICE_SEP),
                                    ["--scratch", rsvd_scratch,
                                     "--gamma-rtol", REFINED_GAMMA_RTOL,
                                     "--outer-blocks", string(REFINED_OUTER_BLOCKS),
                                     "--outer-block-len", string(REFINED_OUTER_BLOCK_LEN)]),
                               threads,
                               min(cluster.max_host_GB,
                                   max(8, ceil(Int, 1.5 * pb.host_bytes / 2^30))),
                               boxed_time(pb.time_s, 2.5), true, rsvd_label,
                               nothing, pb.time_s, 1.0))

    return points
end

# --------------------------------------------------------------------------- #
# The launch script
# --------------------------------------------------------------------------- #

const REFINED_PREAMBLE = """
#
# What this tier measures
#
#   refg_*        the composite Green build, at the four GAP_REFINEMENT_TABLE
#                 entries a production sweep actually reaches, on three
#                 geometries, plus each of those geometries at g = 6 where
#                 nothing is refined. Fits g0_sandwich_scale and
#                 g0_partition_scale.
#   refmv_*       one composite matvec each of G0_rs, G0_rr and the universe
#                 block operator, and the same three on the unrefined control
#                 geometry. Fits mv_composite_scale as a ratio of two measured
#                 numbers.
#   refrsvd_*     one refined RSVD end to end, at k = 800 so that it fits the
#                 box. Checks the assembled model, and with refbounds_* is the
#                 only place the folded Asym applies of the hasmethod shim in
#                 src/rsvd.jl are visible.
#   refbounds_*   one refined bounds front end, outer loop sampled, reading the
#                 RSVD point's scratch.
#
# Every point carries --refine, controls included: bench/point.jl leaves
# refinement off unless a tier asks, unlike src/common.jl, and holding the flag
# fixed across the tier leaves the separation as the only variable. A control at
# g = 6 builds the plain cuboids anyway, which is what makes it a control.
#
# THE PRELOAD DIRECTORY MUST BE EMPTY. stage_greens measures a build, and
# load_green_function returns a cached block rather than building one when the
# file is already there -- a warm directory turns every greens point into a
# deserialisation timing. Worse, a .glaG0 written before Gila rev d4c0516 no
# longer reads back at all: GlaOprVac now serialises mem, srcMsk and trgMsk where
# it used to write mem alone, so an old file dies with an EOFError partway
# through. This script refuses to submit while any .glaG0 is present; clear them
# first, or pass --force if you know what is there.
#
# Before submitting, two steps on two different kinds of node.
#
# Instantiate on the LOGIN node -- compute nodes have no internet, and a fresh
# clone has no Manifest.toml (it is gitignored), so this resolves from scratch:
#
#   module load StdEnv/2023 julia/1.12.5 cuda/12.2
#   cd <code dir>
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# Precompile on a GPU node, NOT here. CUDA.jl has to be configured for the
# cluster's local toolkit and precompiled against a visible device; doing it off
# a GPU node is what produces the errors. A MIG slice is enough:
#
#   salloc --account=<account> --time=00:45:00 --mem=16G --cpus-per-task=2 \\
#          --gpus=a100_1g.5gb:1 srun --pty bash test/setup_cuda_narval.sh
#
# The refg points are independent of each other and can be submitted on their
# own; the device points depend on their own geometry's refg point having written
# the preload directory, and refbounds depends on refrsvd.
"""

"""
    preload_guard(cluster)

The block that stops a submission into a warm preload directory. See
`REFINED_PREAMBLE` for why: a cached block makes `stage_greens` measure the wrong
thing, and a block cached before Gila rev d4c0516 does not deserialise at all.
"""
function preload_guard(cluster::ClusterSpec)
    return """
    if [ "\${1:-}" != "--force" ]; then
        stale=\$(find \$CAL_ROOT/preload -name '*.glaG0' 2>/dev/null | wc -l)
        if [ "\$stale" -ne 0 ]; then
            echo "\$CAL_ROOT/preload already holds \$stale .glaG0 file(s)."
            echo
            echo "stage_greens measures a build, and load_green_function loads a cached"
            echo "block instead of building one. A block cached before Gila rev d4c0516"
            echo "does not even deserialise -- the on-disk format gained srcMsk/trgMsk."
            echo
            echo "Look at them, then clear them:"
            echo "  ls -R \$CAL_ROOT/preload"
            echo "  find \$CAL_ROOT/preload -name '*.glaG0' -delete"
            echo
            echo "Or re-run this script with --force to submit anyway."
            exit 1
        fi
    fi
    """
end

function refined_script(cluster::ClusterSpec, points::Vector{PlannedPoint})
    cluster.has_slurm ||
        error("bench/plan_refined.jl only emits a SLURM script; '$(cluster.name)' has no scheduler. Run the manifest's commands by hand there.")
    io = IOBuffer()
    println(io, """
    #!/bin/bash
    # Cost-model calibration for $(cluster.name), tier=refined.
    # Generated $(now()) by bench/plan_refined.jl. Do not edit; regenerate instead.
    #
    # Every point is its own job: one point running out of memory or time must
    # not take the rest of the calibration with it. Each writes its own row file,
    # so partial results are always usable.
    #
    # Submit:  bash <this script>
    # Collect: bash <this script> --merge
    $(REFINED_PREAMBLE)
    set -u

    CODE_DIR=$(cluster.code_dir)
    CAL_ROOT=$(cluster.cal_root)
    ROWS=\$CAL_ROOT/rows_refined
    OUT=\$CAL_ROOT/calibration_$(cluster.name)_refined.csv

    mkdir -p \$CAL_ROOT/logs \$CAL_ROOT/preload \$CAL_ROOT/project \$CAL_ROOT/scratch \$ROWS
    cd \$CODE_DIR

    $(merge_block(cluster))
    $(preload_guard(cluster))
    echo "Submitting $(length(points)) calibration points for $(cluster.name) (tier=refined)"
    echo "Each point writes its own row file under \$ROWS"
    """)

    for point in points
        var = "jid_$(replace(point.label, r"[^A-Za-z0-9_]" => "_"))"
        dep_line = point.depends_on === nothing ? "" :
            "    --dependency=afterok:\${jid_$(replace(point.depends_on, r"[^A-Za-z0-9_]" => "_"))} \\\n"
        println(io, """
        $var=\$(sbatch --parsable \\
        $(dep_line)    --job-name=psccal_$(point.label) \\
            --output=\$CAL_ROOT/logs/$(point.label)_%j.out \\
            --account=$(cluster.account) \\
            --time=$(seconds2string(point.time_s)) \\
        $(resource_lines(cluster, point))    --chdir=\$CODE_DIR \\
            --export=ALL \\
            <<EOF
        #!/bin/bash
        $(cluster.modules)
        export PSC_T0=\\\$(date +%s)
        srun $(point_command(cluster, point, :refined))
        EOF
        )
        sleep 0.05
        """)
    end

    println(io, """
    echo
    echo "All points submitted. Watch them with: squeue -u \\\$USER"
    echo
    echo "When they have finished, merge the per-point rows and copy the result back:"
    echo "  bash bench/launch_calibration_$(cluster.name)_refined.sh --merge"
    echo "  scp $(CC_UNAME)@$(cluster.name).alliancecan.ca:\$OUT bench/data/"
    """)
    return String(take!(io))
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

"""
    device_report(cluster, coeffs)

What the GPU points ask of the card, printed alongside the point table.
`PlannedPoint` carries no device-memory field -- `--gpus` takes a whole card and
SLURM has nothing to give it -- but device memory is the one resource a point here
can silently exceed, so the numbers are worth reading before submitting.

The matvec points are sized by the operator they hold, which on a refined point is
the composite's blocks summed; the staged points by `predict`, which already knows
about the card and reports which path it sized for.
"""
function device_report(cluster::ClusterSpec, coeffs::Coefficients)
    _, dev_cells, dev_scale = REFINED_DEVICE
    capacity = cluster.max_vram_GB * 1e9
    gb(bytes) = bytes / 2^30
    println()
    @printf("  device memory on a %d GB %s card, padded:\n",
            cluster.max_vram_GB, cluster.full_gpu)
    for sep in (REFINED_DEVICE_SEP, REFINED_CONTROL_SEP)
        pt = srpoint(dev_cells, dev_scale, sep, min(cluster.max_cores, 12))
        @printf("    g=%-2d %-9s operators: ext %5.2f GB  self %5.2f GB  universe %5.2f GB\n",
                gap_cells(sep), CostModel.is_refined(pt) ? "refined" : "control",
                gb(coeffs.vram_pad * CostModel.ext_operator_bytes(pt)),
                gb(coeffs.vram_pad * CostModel.self_operator_bytes(pt)),
                gb(coeffs.vram_pad * CostModel.universe_operator_bytes(pt)))
    end
    pt = srpoint(dev_cells, dev_scale, REFINED_DEVICE_SEP, min(cluster.max_cores, 12))
    for (job, name) in ((GenerateRSVD, "refrsvd"), (ComputeBounds, "refbounds"))
        p = predict(job, pt, coeffs; pad=true, vram_capacity_bytes=capacity)
        @printf("    %-9s %5.2f GB peak (floor %5.2f GB), path %s\n",
                name, gb(p.vram_bytes), gb(p.vram_floor_bytes), string(p.mode))
    end
    return nothing
end

function refined_main(argv::Vector{String})
    cluster_name = "narval"
    dry_run = false
    i = 1
    while i <= length(argv)
        if argv[i] == "--cluster" && i < length(argv)
            cluster_name = argv[i+1]; i += 2
        elseif argv[i] == "--dry-run"
            dry_run = true; i += 1
        else
            error("unknown argument '$(argv[i])'; usage: julia bench/plan_refined.jl [--cluster <name>] [--dry-run]")
        end
    end

    load_coefficients!(@__DIR__)
    cluster = ClusterSpec(cluster_name)
    coeffs = coefficients_for(cluster_name)
    coeffs.calibrated ||
        @warn "cluster '$cluster_name' has no calibrated cost model, so these time limits are analytic guesses"
    points = plan_refined_points(cluster)

    script_path = joinpath(@__DIR__, "launch_calibration_$(cluster_name)_refined.sh")
    manifest_path = joinpath(@__DIR__, "manifest_$(cluster_name)_refined.csv")
    if !dry_run
        write(script_path, refined_script(cluster, points))
        chmod(script_path, 0o755)
        write_manifest(manifest_path, cluster, points, :refined)
    end

    println("Planned $(length(points)) points for $(cluster_name) (tier=refined)")
    @printf("  %-22s %-14s %3s %6s %10s %10s %s\n",
            "label", "kind", "gpu", "host", "predicted", "limit", "depends on")
    for p in points
        @printf("  %-22s %-14s %3s %5dG %10s %10s %s\n", p.label, p.kind,
                p.gpu ? "yes" : "no", p.host_GB,
                p.predicted_s > 0 ? seconds2string(round(Int, p.predicted_s)) : "-",
                seconds2string(p.time_s),
                p.depends_on === nothing ? "-" : p.depends_on)
    end
    device_report(cluster, coeffs)
    at_cap = count(p -> p.time_s >= REFINED_MAX_TIME_S, points)
    at_cap > 0 && @warn "$at_cap point(s) are at the $(seconds2string(REFINED_MAX_TIME_S)) cap, so their limit is the box and not the prediction. Lower --rank or --power-iters for those, or split them."
    gpu_limit = sum(p.time_s for p in points if p.gpu; init=0)
    @printf("  worst case if every point used its whole limit: %.1f h total, %.1f h of it on a GPU\n",
            sum(p.time_s for p in points) / 3600, gpu_limit / 3600)
    @printf("  predicted, if the model is right: %.1f h total\n",
            sum(p.predicted_s for p in points) / 3600)
    println()
    if dry_run
        println("--dry-run: planned but wrote nothing")
        println("Would write $script_path")
        println("Would write $manifest_path")
    else
        println("Wrote $script_path")
        println("Wrote $manifest_path")
    end
    println()
    println("Copy and run (this script submits nothing itself):")
    println("  scp $script_path $(CC_UNAME)@$(cluster_name).alliancecan.ca:$(cluster.code_dir)bench/")
    println("  ssh $(CC_UNAME)@$(cluster_name).alliancecan.ca 'cd $(cluster.code_dir) && bash bench/$(basename(script_path))'")
    return script_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    refined_main(ARGS)
end
