#!/usr/bin/env julia
"""
    bench/plan_refined.jl

Generate the `refined` calibration tier: the points that measure what gap
refinement costs, which `bench/cost_model.jl` predicts and nobody has yet timed.

`bench/point.jl` leaves gap refinement off unless a tier asks, so every point here
carries `--refine`. That is the flag `src/common.jl` now takes as its default, so
these points measure the meshes a production sweep runs on.

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

  * `g0_sandwich_scale` and `g0_partition_scale`, from the four `stage_greens`
    points. A refined build is a sum over region-pair blocks of five kinds, and
    the model already knows the exact shape and count of each; what it does not
    know is whether the fitted `g0_contact_*` and `g0_ext_*` triples, measured on
    whole-body blocks, still hold on a two-cell contact layer remeshed at the
    fine scale and on a partitioned quadrature that pays its fixed cost `pairs`
    times over. The four points spread the two kinds' shares widely enough to
    separate them: `(8,8,8)` at `g = 1` is 6 sandwich and 6 partition blocks with
    no coarse bulk at all, while `(32,32,32)` at `g = 3` is 6 and 18 on a body
    that is almost entirely bulk. Two unrefined controls at `g = 6`, the same
    geometries one cell past the refinement threshold, turn the comparison into
    a measurement instead of a difference across two fits.

  * `mv_composite_scale`, from the three `matvec_*` points. A composite matvec is
    a loop over region-pair blocks, and the model charges each block's FFT work
    and each block's launch overhead separately, so what is left for the
    multiplier is `_cmpMul`'s own bookkeeping: the output allocation, the
    per-region reshapes and views, and the accumulation over the block row. These
    are the cheapest points here, seconds apiece, and they are the only ones that
    measure the multiplier without the RSVD's dense work on top.

  * The `stage_rsvd` and `stage_bounds` points are the assembled check, the same
    role the `stage_*` points play in the calibration tiers proper. They are the
    only two that exercise a refined `N_u` end to end, including the folded
    `Asym(G⁰ᵣᵣ)` and `Asym(G⁰ᵤᵤ)` applies that `src/rsvd.jl`'s `hasmethod` shim
    picks up, which no primitive point can see, since they live in the shim and
    not in the operator.

# Sizes and the three-hour box

Every point is predicted under three hours on narval's own coefficients, which is
the whole reason the RSVD and bounds points run at `k = 800`, `q = 6` rather than
at the production `k = 4000`, `q = 14`: the multipliers being fitted are
per-matvec and per-block, and neither moves with the rank. The bounds point
samples its outer loop (`--outer-blocks` / `--outer-block-len`) exactly as the
backfill tier does, so its cost is the front end (the Gram-Schmidt, the
projections and the Green sweep) which is the part refinement moves.

The refined geometries are isotropic because `bench/point.jl`'s `--scale` takes
one rational, so the anisotropic production sweeps cannot be measured through it.
That is not a gap in the calibration: the block laws are functions of the region
shapes, and an anisotropic body refines along x exactly as an isotropic one does.

# Fitting the result

`bench/fit.jl` has no reader for these rows yet. The four greens rows give a
two-parameter linear least squares against the model's own per-kind counts,

    measured − predicted_without_crossscale
        = s_sandwich · C_sandwich + s_partition · C_partition

with `C_*` the bracketed sums `greens_time_s` already forms
(`g0_contact_fft · sandwich_fft_work + g0_contact_cell · sandwich_cells / eta +
g0_contact_fixed · n_sandwich_blocks`, and its `ext` counterpart), and the two
control rows fixing the unrefined intercept. `mv_composite_scale` is one ratio
per matvec row. Put the three numbers into `bench/coeffs_<cluster>.jl` by hand
until `fit.jl` learns to read the tier; they are three lines, and the file names
them as provisional.
"""

include(joinpath(@__DIR__, "plan.jl"))

using Printf
using Dates

# --------------------------------------------------------------------------- #
# The points
# --------------------------------------------------------------------------- #

const REFINED_SCALE = 1 // 32
const REFINED_CHI = DEFAULT_CHI
const REFINED_SEED = 20260827
# `k` and `q` well under production: see the docstring. Neither multiplier being
# fitted moves with the rank, and this is what keeps every point inside the box.
const REFINED_RANK = 800
const REFINED_POWER_ITERS = 6
const REFINED_OVERSAMPLES = 50
const REFINED_GAMMA_RTOL = "1.0e-12"
# The sampled outer loop of the bounds point, the same shape the backfill tier
# uses: four blocks of 24 indices, spread over the loop.
const REFINED_OUTER_BLOCKS = 4
const REFINED_OUTER_BLOCK_LEN = 24
# Nothing here may ask for more than this. A calibration point that is killed at
# the walltime measures nothing at all.
const REFINED_MAX_TIME_S = 3 * 3600
# The refined and unrefined ends of the band. `1//32` and `3//32` are two
# different table entries ((6,6) and (2,4)); `6//32` is the first gap the job
# leaves alone, so it is the same geometry with none of the new block kinds.
const REFINED_GAPS = (1 // 32, 3 // 32)
const REFINED_CONTROL_GAP = MIN_GAP_CELLS // 32
const REFINED_BODIES = ((:c8, (8, 8, 8)), (:c32, (32, 32, 32)))
# The geometry the device points run on: the largest of the two, where the
# composite matvec has the most blocks and the most to say.
const REFINED_DEVICE_BODY = (32, 32, 32)
const REFINED_DEVICE_GAP = 1 // 32

# Refinement is the default in `src/common.jl` but not in `bench/point.jl`, so this
# tier spells it out on both sides: `refine_gap` on the predicted point and
# `--refine` on the job.
srpoint(cells, sep, threads) =
    SRPoint(cells, cells; scale=REFINED_SCALE, separation=sep, rank=REFINED_RANK,
            oversamples=REFINED_OVERSAMPLES, power_iters=REFINED_POWER_ITERS,
            threads=threads, refine_gap=true)

"The `bench/point.jl` flags every point of this tier shares."
common_args(cells, sep) =
    ["--cells", join(cells, ","), "--scale", string(REFINED_SCALE),
     "--chi", REFINED_CHI, "--sep", string(sep),
     "--rank", string(REFINED_RANK), "--oversamples", string(REFINED_OVERSAMPLES),
     "--power-iters", string(REFINED_POWER_ITERS), "--seed", string(REFINED_SEED),
     "--refine"]

"""
    boxed_time(predicted_s, factor) -> Int

A time limit from a prediction: `factor` times it, floored at ten minutes so a
seconds-long point still gets a usable limit, and capped at
`REFINED_MAX_TIME_S`. The cap is a hard requirement of this tier, so a point that
hits it is reported by `main` rather than silently truncated.
"""
boxed_time(predicted_s::Real, factor::Real) =
    clamp(ceil(Int, factor * predicted_s), 600, REFINED_MAX_TIME_S)

function plan_refined_points(cluster::ClusterSpec)
    threads = min(cluster.max_cores, 12)
    coeffs = coefficients_for(cluster.name)
    points = PlannedPoint[]
    greens_label = Dict{Tuple{Symbol,Rational{Int}},String}()

    # ---- the four refined Green builds, plus two unrefined controls --------
    for (tag, cells) in REFINED_BODIES,
        sep in (REFINED_GAPS..., REFINED_CONTROL_GAP)

        g = numerator(sep * 32)
        label = "refg_$(tag)_g$(g)"
        pt = srpoint(cells, sep, threads)
        p = predict(GenerateGreens, pt, coeffs; pad=true)
        # Three times the padded prediction: these are the points whose cost the
        # tier exists to discover, so the limit cannot be tight around a number
        # the model is admitting it does not know.
        push!(points, PlannedPoint(label, "stage_greens", common_args(cells, sep),
                                   threads,
                                   max(8, ceil(Int, 3 * p.host_bytes / 1e9)),
                                   boxed_time(p.time_s, 3.0), false))
        greens_label[(tag, sep)] = label
    end

    # ---- the composite matvecs --------------------------------------------
    #=
    Seconds each: `bench/point.jl` times `--reps` applications of an operator that
    is already on disk. They depend on the Green build of the same geometry and
    gap so the blocks are there to load rather than rebuilt inside a device job.
    Ten minutes and a whole card is the smallest useful ask.
    =#
    dev_cells, dev_sep = REFINED_DEVICE_BODY, REFINED_DEVICE_GAP
    dev_pt = srpoint(dev_cells, dev_sep, threads)
    dev_dep = greens_label[(:c32, dev_sep)]
    for kind in ("matvec_ext", "matvec_self", "matvec_uu")
        push!(points, PlannedPoint("ref$(replace(kind, "matvec_" => "mv_"))_c32_g1",
                                   kind, common_args(dev_cells, dev_sep),
                                   threads, 16, 600, true, dev_dep))
    end

    # ---- one refined RSVD, and the bounds job that reads it ----------------
    rsvd_label = "refrsvd_c32_g1"
    rsvd_scratch = "\$CAL_ROOT/refined/$(rsvd_label)"
    capacity = cluster.max_vram_GB * 1e9
    pr = predict(GenerateRSVD, dev_pt, coeffs; pad=true, vram_capacity_bytes=capacity)
    push!(points, PlannedPoint(rsvd_label, "stage_rsvd",
                               vcat(common_args(dev_cells, dev_sep),
                                    ["--scratch", rsvd_scratch, "--fresh"]),
                               threads,
                               min(cluster.max_host_GB, max(8, ceil(Int, 1.5 * pr.host_bytes / 1e9))),
                               boxed_time(pr.time_s, 1.5), true, dev_dep))

    # The bounds point reads the RSVD point's scratch and waits on it, the same
    # wiring the funicular tier's E4 chain uses.
    pb = predict(ComputeBounds, dev_pt, coeffs; pad=true, vram_capacity_bytes=capacity)
    push!(points, PlannedPoint("refbounds_c32_g1", "stage_bounds",
                               vcat(common_args(dev_cells, dev_sep),
                                    ["--scratch", rsvd_scratch,
                                     "--gamma-rtol", REFINED_GAMMA_RTOL,
                                     "--outer-blocks", string(REFINED_OUTER_BLOCKS),
                                     "--outer-block-len", string(REFINED_OUTER_BLOCK_LEN)]),
                               threads,
                               min(cluster.max_host_GB, max(8, ceil(Int, 1.5 * pb.host_bytes / 1e9))),
                               boxed_time(pb.time_s, 2.0), true, rsvd_label))

    return points
end

# --------------------------------------------------------------------------- #
# The launch script
# --------------------------------------------------------------------------- #

const REFINED_PREAMBLE = """
#
# What this tier measures
#
#   refg_*        the composite Green build, at two gaps that land on two
#                 different GAP_REFINEMENT_TABLE entries and on two body sizes,
#                 plus the same two geometries at g = 6 where nothing is refined.
#                 Fits g0_sandwich_scale and g0_partition_scale.
#   refmv_*       one composite matvec each of G0_rs, G0_rr and the universe
#                 block operator. Fits mv_composite_scale.
#   refrsvd_*     one refined RSVD end to end, at k = 800 and q = 6 so that it
#                 fits the box. Checks the assembled model, and is the only point
#                 that sees the folded Asym(G0_rr) applies of the hasmethod shim
#                 in src/rsvd.jl.
#   refbounds_*   one refined bounds front end, outer loop sampled, reading the
#                 RSVD point's scratch.
#
# Every point carries --refine, which is what puts bench/point.jl on the refined
# path; it is off by default there even though src/common.jl now refines.
#
# Before submitting, on the LOGIN node:
#
#   module load StdEnv/2023 julia/1.12.5 cuda/12.2
#   cd <code dir>
#   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
#
# The four refg points and the two controls are independent and can be submitted
# on their own; the device points depend on refg_c32_g1 having written the
# preload directory, and refbounds depends on refrsvd.
"""

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
    coefficients_for(cluster_name).calibrated ||
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
    @printf("  %-22s %-14s %3s %6s %10s %s\n", "label", "kind", "gpu", "host", "time", "depends on")
    for p in points
        @printf("  %-22s %-14s %3s %5dG %10s %s\n", p.label, p.kind, p.gpu ? "yes" : "no",
                p.host_GB, seconds2string(p.time_s),
                p.depends_on === nothing ? "-" : p.depends_on)
    end
    at_cap = count(p -> p.time_s >= REFINED_MAX_TIME_S, points)
    at_cap > 0 && @warn "$at_cap point(s) are at the $(seconds2string(REFINED_MAX_TIME_S)) cap, so their limit is the box and not the prediction. Lower --rank or --power-iters for those, or split them."
    @printf("  worst-case wall time if every point used its whole limit: %.1f h\n",
            sum(p.time_s for p in points) / 3600)
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
