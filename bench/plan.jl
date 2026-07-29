#!/usr/bin/env julia
"""
    bench/plan.jl

Generate the calibration run script for one cluster.

    julia bench/plan.jl --cluster fir      --tier quick
    julia bench/plan.jl --cluster narval   --tier full
    julia bench/plan.jl --cluster molering --tier full

Writes `bench/launch_calibration_<cluster>_<tier>.sh` plus a manifest CSV listing
every planned point, and prints the copy/run instructions.

# Why these points

Each fitted coefficient in `CostModel.Coefficients` needs to be identified by
some point in this plan, and nothing here exists for any other reason:

  * `g0_self` / `g0_ext` across eight body sizes separate the `M log M` (FFT of
    the circulant) term from the `M` (per-cell quadrature) term, and separate
    self blocks from external ones -- external blocks keep eight times as much
    Fourier data.
  * `g0_ext` at five separations, one of them zero, isolates the contact
    surcharge. Cost is expected to be flat in separation otherwise; the
    1000-wavelength point is there to confirm that, because the previous
    estimator's blow-up came from assuming the opposite.
  * The thread scan fixes the parallel efficiency of the quadrature loops, which
    is what decides how many cores are worth requesting.
  * `matvec_self` / `matvec_ext` across sizes give the per-matvec cost that
    dominates the RSVD job, at the shapes it actually uses.
  * `dense` across `(m, c)` gives the QR / gemm / eigen rates. These are *not*
    negligible: at `c ~ 3000` the `q + 1` thin QRs of an `N_u x c` matrix are
    comparable to the entire matvec bill, and the previous estimator omitted
    them entirely.
  * `bounds_core` gives the bounds job's Gram-Schmidt, projection and inner-loop
    costs without needing an RSVD output to exist first.
  * `stage_greens` / `stage_rsvd` / `stage_bounds` chains are the validation:
    they check the assembled model against reality on a handful of real runs.

# Tiers

  * `quick`   -- primitives on the four smallest bodies. Enough to identify every
                 coefficient, and short enough to run in an afternoon. Start here.
  * `full`    -- the same primitives across every body, including the large
                 anisotropic ones, so the fit interpolates rather than
                 extrapolates at the sizes you actually submit.
  * `validate` -- end-to-end `greens -> rsvd -> bounds` chains only. Run these
                 *after* fitting, to check the assembled model against reality.
                 They are the expensive tier: on molering, where points run
                 sequentially, budget days rather than hours.
"""

include(joinpath(@__DIR__, "cost_model.jl"))
using .CostModel
using Printf
using Dates

# --------------------------------------------------------------------------- #
# Cluster configuration
# --------------------------------------------------------------------------- #

const CC_UNAME = "pvirally"
const CC_ACCOUNT = "def-smolesky"
const CC_CODE_DIR = "/home/$(CC_UNAME)/Photonic-System-Channels/"
const CC_CAL_ROOT = "/home/$(CC_UNAME)/scratch/psc-calibration/"

const MOLERING_UNAME = "paulv"
const MOLERING_CODE_DIR = "/home/$(MOLERING_UNAME)/Projects/Photonic-System-Channels/"
const MOLERING_CAL_ROOT = "/home/molering/fatmole/$(MOLERING_UNAME)/psc-calibration/"

struct ClusterSpec
    name::String
    has_slurm::Bool
    code_dir::String
    cal_root::String
    account::String
    modules::String
    full_gpu::String     # SLURM --gpus argument for a whole GPU
    max_cores::Int
    max_host_GB::Int
    max_vram_GB::Int
end

function ClusterSpec(name::AbstractString)
    name == "fir" && return ClusterSpec("fir", true, CC_CODE_DIR, CC_CAL_ROOT, CC_ACCOUNT,
                                        "module load StdEnv/2023 julia/1.12.5 cuda/12.2",
                                        "h100:1", 12, 240, 80)
    name == "narval" && return ClusterSpec("narval", true, CC_CODE_DIR, CC_CAL_ROOT, CC_ACCOUNT,
                                           "module load StdEnv/2023 julia/1.12.5 cuda/12.2",
                                           "a100:1", 12, 240, 40)
    name == "molering" && return ClusterSpec("molering", false, MOLERING_CODE_DIR,
                                             MOLERING_CAL_ROOT, "", "", "a6000", 32, 480, 48)
    error("Unknown cluster '$name'. Known: fir, narval, molering.")
end

# --------------------------------------------------------------------------- #
# The bodies we calibrate on
# --------------------------------------------------------------------------- #

#=
Cell counts and ranks track the runs already in `data analysis/data`, so that the
fit is interpolating over the region the real jobs live in rather than
extrapolating into it. The anisotropic entries use the negative-scale convention
from `SMRSystem`: `scale = -1//8` means cells of (1/32, 1/8, 1/8) wavelengths,
which is how the existing 3-lambda and 4-lambda runs keep their cell counts
manageable.
=#
const BODIES = [
    (label="l0p25", cells=(8, 8, 8), scale=1 // 32, rank=1350, tier=:quick),
    (label="l0p5", cells=(16, 16, 16), scale=1 // 32, rank=1350, tier=:quick),
    (label="l0p75", cells=(24, 24, 24), scale=1 // 32, rank=1350, tier=:quick),
    (label="l1", cells=(32, 32, 32), scale=1 // 32, rank=2750, tier=:quick),
    (label="l2agiso", cells=(64, 32, 32), scale=-1 // 8, rank=1350, tier=:full),
    (label="l3aniso", cells=(96, 32, 32), scale=-1 // 8, rank=800, tier=:full),
    (label="l4aniso", cells=(128, 32, 32), scale=-1 // 8, rank=600, tier=:full),
    (label="l2iso", cells=(64, 64, 64), scale=1 // 32, rank=600, tier=:full),
]

"Separations in wavelengths. Zero is contact; the last one checks the far field."
const SEPARATIONS = [0 // 1, 1 // 32, 8 // 32, 1 // 1, 1000 // 1]
const QUICK_SEPARATIONS = [0 // 1, 8 // 32, 1000 // 1]

const THREAD_SCAN = [1, 2, 4, 8]

"Sketch widths for the dense points, spanning the ranks actually in use."
const DENSE_WIDTHS = [128, 512, 1400, 2800]

const DEFAULT_CHI = "13.6+0.05im"
const DEFAULT_OVERSAMPLES = 50
const DEFAULT_POWER_ITERS = 14

# --------------------------------------------------------------------------- #
# Points
# --------------------------------------------------------------------------- #

"""
    PlannedPoint

One invocation of `bench/point.jl`, with the resources to ask for. `args` are
passed through verbatim, `threads` becomes both `--cpus-per-task` and Julia's
`-t`, and `gpu` says whether to request a whole GPU.
"""
struct PlannedPoint
    label::String
    kind::String
    args::Vector{String}
    threads::Int
    host_GB::Int
    time_s::Int
    gpu::Bool
end

cells_arg(cells::NTuple{3,Int}) = join(cells, ",")
rat(r::Rational{Int}) = "$(numerator(r))//$(denominator(r))"

as_srpoint(body, separation::Rational{Int}, threads::Int) =
    SRPoint(body.cells, body.cells;
            scale=body.scale < 0 ? (1 // 32, abs(body.scale), abs(body.scale)) :
                  (body.scale, body.scale, body.scale),
            separation=separation, rank=body.rank, oversamples=DEFAULT_OVERSAMPLES,
            power_iters=DEFAULT_POWER_ITERS, threads=threads)

"Resources for a Green-block point: three times the analytic peak, floored at 8 GB."
function block_resources(cluster::ClusterSpec, body, separation::Rational{Int}, threads::Int)
    pt = as_srpoint(body, separation, threads)
    counts = greens_counts(pt)
    host_GB = min(cluster.max_host_GB, max(8, ceil(Int, 3 * counts.peak_bytes / 2^30) + 4))
    # Uncalibrated predictions can be badly wrong in either direction, so give
    # calibration jobs a very loose time box: they are cheap and being killed
    # mid-measurement wastes the whole point.
    time_s = clamp(ceil(Int, 10 * predict(GenerateGreens, pt, coefficients_for(cluster.name);
                                          pad=false).time_s), 3600, 12 * 3600)
    return host_GB, time_s
end

function plan_points(cluster::ClusterSpec, tier::Symbol)
    quick = tier == :quick
    micro = tier != :validate
    bodies = quick ? filter(b -> b.tier == :quick, BODIES) : BODIES
    separations = quick ? QUICK_SEPARATIONS : SEPARATIONS
    points = PlannedPoint[]

    common(body) = ["--cells", cells_arg(body.cells),
                    "--scale", rat(body.scale),
                    "--chi", DEFAULT_CHI,
                    "--rank", string(body.rank),
                    "--oversamples", string(DEFAULT_OVERSAMPLES),
                    "--power-iters", string(DEFAULT_POWER_ITERS)]

    # ---- Host: Green function block construction ---------------------------
    for body in (micro ? bodies : ())
        threads = min(4, cluster.max_cores)
        host_GB, time_s = block_resources(cluster, body, 8 // 32, threads)
        # Self blocks do not depend on the separation at all.
        push!(points, PlannedPoint("g0self_$(body.label)", "g0_self",
                                   vcat(common(body), ["--sep", rat(8 // 32)]),
                                   threads, host_GB, time_s, false))
        for sep in separations
            hg, ts = block_resources(cluster, body, sep, threads)
            push!(points, PlannedPoint("g0ext_$(body.label)_sep$(numerator(sep))ss$(denominator(sep))",
                                       "g0_ext", vcat(common(body), ["--sep", rat(sep)]),
                                       threads, hg, ts, false))
        end
        # The multi-region build is where the Green job actually peaks, so it is
        # the point that validates the memory model rather than a coefficient.
        push!(points, PlannedPoint("g0uu_$(body.label)", "g0_multiregion",
                                   vcat(common(body), ["--sep", rat(8 // 32)]),
                                   threads, host_GB, time_s, false))
    end

    # ---- Host: thread scaling ----------------------------------------------
    scan_body = bodies[min(length(bodies), quick ? 3 : 4)]
    for threads in (micro ? filter(<=(cluster.max_cores), THREAD_SCAN) : ())
        host_GB, time_s = block_resources(cluster, scan_body, 8 // 32, threads)
        push!(points, PlannedPoint("g0threads_$(scan_body.label)_t$(threads)", "g0_ext",
                                   vcat(common(scan_body), ["--sep", rat(8 // 32)]),
                                   threads, host_GB, time_s, false))
    end

    # ---- Device: Green matvecs ---------------------------------------------
    for body in (micro ? bodies : ())
        args = vcat(common(body), ["--sep", rat(8 // 32), "--reps", "20"])
        host_GB, time_s = block_resources(cluster, body, 8 // 32, 4)
        for (kind, suffix) in (("matvec_self", "self"), ("matvec_ext", "ext"),
                               ("matvec_uu", "uu"))
            push!(points, PlannedPoint("mv$(suffix)_$(body.label)", kind, args,
                                       min(4, cluster.max_cores), host_GB, time_s, true))
        end
    end

    # ---- Device: dense linear algebra --------------------------------------
    for body in (micro ? bodies : ())
        N_u = 6 * prod(body.cells)
        for c in DENSE_WIDTHS
            c <= N_u || continue
            # Three m-by-c complex matrices plus workspace, kept under 40% of the
            # smallest VRAM we calibrate on so the same plan runs everywhere.
            bytes = 3 * N_u * c * 16 + 6 * c^2 * 16
            bytes < 0.4 * cluster.max_vram_GB * 2^30 || continue
            push!(points, PlannedPoint("dense_$(body.label)_c$(c)", "dense",
                                       vcat(common(body),
                                            ["--sep", rat(8 // 32), "--dense-m", string(N_u),
                                             "--dense-c", string(c), "--reps", "12"]),
                                       min(4, cluster.max_cores),
                                       max(8, ceil(Int, 2 * bytes / 2^30) + 4), 3600, true))
        end
    end

    # ---- Device: the bounds kernel on synthetic spectra --------------------
    for body in (micro ? bodies : ())
        for rank in unique([min(body.rank, 256), min(body.rank, 800), body.rank])
            N_u = 6 * prod(body.cells)
            bytes = 5 * N_u * rank * 16
            bytes < 0.5 * cluster.max_vram_GB * 2^30 || continue
            args = ["--cells", cells_arg(body.cells), "--scale", rat(body.scale),
                    "--chi", DEFAULT_CHI, "--sep", rat(8 // 32),
                    "--rank", string(rank), "--oversamples", string(DEFAULT_OVERSAMPLES),
                    "--power-iters", string(DEFAULT_POWER_ITERS),
                    "--num-pos-frac", "0.5", "--outer-samples", "4"]
            push!(points, PlannedPoint("boundscore_$(body.label)_k$(rank)", "bounds_core",
                                       args, min(4, cluster.max_cores),
                                       max(16, ceil(Int, 3 * bytes / 2^30) + 8), 4 * 3600, true))
        end
    end

    # ---- End-to-end validation chains -------------------------------------
    if tier == :validate
        for body in bodies
            # One chain per body at a mid separation, plus contact on the
            # smallest two where it is cheap, since contact is the case where the
            # model is most likely to be wrong.
            seps = prod(body.cells) <= 4096 ? [8 // 32, 0 // 1] : [8 // 32]
            for sep in seps
                tag = "$(body.label)_sep$(numerator(sep))ss$(denominator(sep))"
                pt = as_srpoint(body, sep, min(4, cluster.max_cores))
                coeffs = coefficients_for(cluster.name)
                args = vcat(common(body), ["--sep", rat(sep)])
                hg, ts = block_resources(cluster, body, sep, min(4, cluster.max_cores))
                push!(points, PlannedPoint("stagegreens_$(tag)", "stage_greens", args,
                                           min(4, cluster.max_cores), hg, ts, false))
                p_rsvd = predict(GenerateRSVD, pt, coeffs; pad=false)
                push!(points, PlannedPoint("stagersvd_$(tag)", "stage_rsvd", args,
                                           min(4, cluster.max_cores),
                                           max(16, ceil(Int, 3 * p_rsvd.host_bytes / 2^30)),
                                           clamp(ceil(Int, 6 * p_rsvd.time_s), 3600, 24 * 3600),
                                           true))
                p_bounds = predict(ComputeBounds, pt, coeffs; pad=false)
                push!(points, PlannedPoint("stagebounds_$(tag)", "stage_bounds", args,
                                           min(4, cluster.max_cores),
                                           max(16, ceil(Int, 3 * p_bounds.host_bytes / 2^30)),
                                           clamp(ceil(Int, 6 * p_bounds.time_s), 3600, 24 * 3600),
                                           true))
            end
        end
    end

    return points
end

# --------------------------------------------------------------------------- #
# Script emission
# --------------------------------------------------------------------------- #

seconds2string(seconds::Real) = @sprintf("%02d:%02d:%02d", seconds ÷ 3600,
                                         (seconds % 3600) ÷ 60, seconds % 60)

function point_command(cluster::ClusterSpec, point::PlannedPoint, tier::Symbol)
    # `--note` is quoted because its value contains a `;`, and `--scale` because
    # anisotropic scales are negative rationals that a bare shell word would be
    # happy to mangle.
    return join(vcat(["julia", "--project=.", "-t", string(point.threads),
                      "bench/point.jl", "--kind", point.kind],
                     [startswith(a, "--") ? a : "'$a'" for a in point.args],
                     ["--gpu", point.gpu ? "0" : "-1",
                      "--root", "\$CAL_ROOT", "--out", "\$OUT",
                      "--cluster", cluster.name,
                      "--note", "'tier=$(tier);label=$(point.label)'"]), " ")
end

function slurm_script(cluster::ClusterSpec, points::Vector{PlannedPoint}, tier::Symbol)
    io = IOBuffer()
    println(io, """
    #!/bin/bash
    # Cost-model calibration for $(cluster.name), tier=$(tier).
    # Generated $(now()) by bench/plan.jl. Do not edit; regenerate instead.
    #
    # Every point is its own job: one point running out of memory or time must
    # not take the rest of the calibration with it. Rows are appended to
    # \$OUT as each job finishes, so partial results are still usable.

    set -u

    CODE_DIR=$(cluster.code_dir)
    CAL_ROOT=$(cluster.cal_root)
    OUT=\$CAL_ROOT/calibration_$(cluster.name).csv

    mkdir -p \$CAL_ROOT/logs \$CAL_ROOT/preload \$CAL_ROOT/project \$CAL_ROOT/scratch
    cd \$CODE_DIR

    echo "Submitting $(length(points)) calibration points for $(cluster.name) (tier=$(tier))"
    echo "Results will accumulate in \$OUT"
    """)

    for point in points
        gpu_line = point.gpu ? "    --gpus=$(cluster.full_gpu) \\\n" : ""
        println(io, """
        sbatch \\
            --job-name=psccal_$(point.label) \\
            --output=\$CAL_ROOT/logs/$(point.label)_%j.out \\
            --account=$(cluster.account) \\
            --time=$(seconds2string(point.time_s)) \\
            --cpus-per-task=$(point.threads) \\
            --mem=$(point.host_GB)G \\
        $(gpu_line)    --chdir=\$CODE_DIR \\
            --export=ALL \\
            <<EOF
        #!/bin/bash
        $(cluster.modules)
        export PSC_T0=\$(date +%s)
        srun $(point_command(cluster, point, tier))
        EOF
        sleep 0.05
        """)
    end

    println(io, """
    echo
    echo "All points submitted. Watch them with: squeue -u \\\$USER"
    echo "When they are done, copy the CSV back:"
    echo "  scp $(CC_UNAME)@$(cluster.name).alliancecan.ca:\$OUT bench/data/"
    """)
    return String(take!(io))
end

function bash_script(cluster::ClusterSpec, points::Vector{PlannedPoint}, tier::Symbol)
    io = IOBuffer()
    println(io, """
    #!/bin/bash
    # Cost-model calibration for $(cluster.name), tier=$(tier).
    # Generated $(now()) by bench/plan.jl. Do not edit; regenerate instead.
    #
    # No scheduler here, so points run one at a time in the foreground. Each is
    # allowed to fail without stopping the run; check the logs afterwards for
    # any point whose row is missing from the CSV.
    #
    # Run it detached, it takes a while:
    #   nohup bash $(basename("launch_calibration_$(cluster.name)_$(tier).sh")) > calibration.log 2>&1 &

    set -u

    CODE_DIR=$(cluster.code_dir)
    CAL_ROOT=$(cluster.cal_root)
    OUT=\$CAL_ROOT/calibration_$(cluster.name).csv

    mkdir -p \$CAL_ROOT/logs \$CAL_ROOT/preload \$CAL_ROOT/project \$CAL_ROOT/scratch
    cd \$CODE_DIR

    export PSC_CLUSTER=$(cluster.name)

    total=$(length(points))
    index=0
    """)

    for point in points
        println(io, """
        index=\$((index + 1))
        echo "[\$index/\$total] $(point.label)"
        export PSC_T0=\$(date +%s)
        $(point_command(cluster, point, tier)) \\
            > \$CAL_ROOT/logs/$(point.label).out 2>&1 \\
            || echo "  FAILED: $(point.label) (see \$CAL_ROOT/logs/$(point.label).out)"
        """)
    end

    println(io, """
    echo
    echo "Done. Copy the CSV back to your laptop:"
    echo "  scp $(MOLERING_UNAME)@molering:\$OUT bench/data/"
    """)
    return String(take!(io))
end

function write_manifest(path::AbstractString, cluster::ClusterSpec,
                       points::Vector{PlannedPoint}, tier::Symbol)
    open(path, "w") do io
        println(io, "label,kind,threads,host_GB,time_limit_s,gpu,args")
        for p in points
            println(io, join([p.label, p.kind, p.threads, p.host_GB, p.time_s,
                              p.gpu ? 1 : 0, "\"$(join(p.args, " "))\""], ","))
        end
    end
    return path
end

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

function main(argv::Vector{String})
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

    cluster_name = get(opts, "cluster", "")
    isempty(cluster_name) && error("--cluster is required (fir, narval or molering)")
    tier = Symbol(get(opts, "tier", "quick"))
    tier in (:quick, :full, :validate) || error("--tier must be quick, full or validate")

    load_coefficients!(@__DIR__)
    cluster = ClusterSpec(cluster_name)
    points = plan_points(cluster, tier)

    script = cluster.has_slurm ? slurm_script(cluster, points, tier) :
             bash_script(cluster, points, tier)
    script_path = joinpath(@__DIR__, "launch_calibration_$(cluster_name)_$(tier).sh")
    write(script_path, script)
    chmod(script_path, 0o755)
    manifest_path = write_manifest(joinpath(@__DIR__,
                                            "manifest_$(cluster_name)_$(tier).csv"),
                                   cluster, points, tier)

    gpu_points = count(p -> p.gpu, points)
    total_time = sum(p.time_s for p in points)
    println("Planned $(length(points)) points for $(cluster_name) (tier=$tier)")
    println("  $(length(points) - gpu_points) host points, $gpu_points device points")
    println("  worst-case wall time if every point used its whole limit: ",
            @sprintf("%.1f h", total_time / 3600))
    println("  (real total is far less; the limits are deliberately loose)")
    by_kind = Dict{String,Int}()
    for p in points
        by_kind[p.kind] = get(by_kind, p.kind, 0) + 1
    end
    for (kind, count) in sort(collect(by_kind))
        println("    ", rpad(kind, 16), count)
    end
    println()
    println("Wrote $script_path")
    println("Wrote $manifest_path")
    println()
    if cluster.has_slurm
        println("Copy and run:")
        println("  scp $script_path $(CC_UNAME)@$(cluster_name).alliancecan.ca:$(cluster.code_dir)bench/")
        println("  ssh $(CC_UNAME)@$(cluster_name).alliancecan.ca 'cd $(cluster.code_dir) && bash bench/$(basename(script_path))'")
    else
        println("Copy and run:")
        println("  scp $script_path $(MOLERING_UNAME)@molering:$(cluster.code_dir)bench/")
        println("  ssh $(MOLERING_UNAME)@molering 'cd $(cluster.code_dir) && nohup bash bench/$(basename(script_path)) > calibration.log 2>&1 &'")
    end
    return script_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
