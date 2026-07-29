using PhotonicSystemChannels
using GilaElectromagnetics
using Dates
using Printf

# create_jobs.jl
# --------------
# Set PROJECT_NAME and the cluster below, define your experiments at the bottom
# of the file, then run
#
#     julia create_jobs.jl
#
# which writes `jobs/launch_<PROJECT_NAME>.sh` and prints the command to copy it
# over. Run that script on the cluster to submit everything.
#
# Time and memory requests come from `bench/cost_model.jl`, calibrated per cluster
# by the harness in `bench/` (see bench/README.md). If a cluster has no
# `bench/coeffs_<cluster>.jl` yet, the model falls back to uncalibrated analytic
# guesses and says so loudly -- treat those requests as placeholders.

include(joinpath(@__DIR__, "bench", "cost_model.jl"))
using .CostModel

const PROJECT_NAME = "metasurface_0p25x0p25x0p25_1350comps_50oversamples_32scale"

# Previous project names:
#   heat-transfer_sep_2x2x0p5_512comps
#   SM_metasurface_2x2x0p5_400comps
#   metasurface_1x2x2_700comps_50oversamples
#   fir_metasurface_1x1x1_2750comps_50oversamples
#   fir_metasurface_2x2x2_1350comps_50oversamples_aniso
#   fir_metasurface_3x3x3_800comps_50oversamples_aniso
#   fir_metasurface_4x4x4_600comps_50oversamples_aniso
#   fir_spherecomp_2x2x2_1350comps_50_oversamples_silica_aniso
#   waveguide_0p25x1x1_@0p125_800comps_50oversamples
#
# Susceptibilities in use:
#   paper: 13.6 + 0.05im (TODO: use 11.098 + 0.05im?)
#   silica: -2.30466271 + 1.478912im

# Molering config
const MOLERING_UNAME = "paulv"
const MOLERING_CODE_DIR = "/home/$(MOLERING_UNAME)/Projects/Photonic-System-Channels/"
const MOLERING_PRELOAD_DIR = "/home/molering/fatmole/greens_functions/"
const MOLERING_PROJECT_DIR = "/home/$(MOLERING_UNAME)/Projects/Photonic-System-Channels/projects/"
const MOLERING_SCRATCH_DIR = "/home/molering/fatmole/$(MOLERING_UNAME)/Photonic-System-Channels/"

# Compute canada (digital research alliance of canada) config
const CC_UNAME = "pvirally"
const CC_DEFAULT_GROUP_NAME = "def-smolesky"
const CC_RRG_NAME = "rrg-smolesky"
const CC_RRG_CLUSTERS = [] # The RRG doesn't work anymore :(
const CC_CODE_DIR = "/home/$(CC_UNAME)/Photonic-System-Channels/"
const CC_PRELOAD_DIR = "/home/$(CC_UNAME)/scratch/preload/"
const CC_SCRATCH_DIR = "/home/$(CC_UNAME)/scratch/Photonic-System-Channels/"

"""
    NUM_POS_FRACTION

Assumed fraction of the computed rank that has a positive `Asym(G⁰ᵤᵣ)` eigenvalue.
The bounds job runs one dense `k x k` generalized eigendecomposition per positive
eigenvalue and an `O(num_pos²)` inner loop on top, so this number matters a lot
and is only known after the RSVD has run.

Measured across `data analysis/data`: 0.22-0.52, clustering near 0.50 for the
larger runs. 0.60 is deliberately on the high side so the bounds job does not run out of memory.
"""
const NUM_POS_FRACTION = 0.60

"""
    MIN_MEMORY_GB, MIN_TIME_S

Floors on every request. 4 GB because nothing useful runs in less and the
scheduler does not reward shaving it; 10 minutes because process startup and
package loading alone can eat several minutes on a cold shared filesystem.
"""
const MIN_MEMORY_GB = 4
const MIN_TIME_S = 10 * 60

"""
    TARGET_WALL_TIME_S

Wall time the core-count heuristic aims to come in under for the CPU job. Alliance
schedulers backfill short jobs aggressively, so a job that fits in a few hours
starts sooner than one that asks for a day. Only used to pick `--cpus-per-task`.
"""
const TARGET_WALL_TIME_S = 3 * 3600

const CORE_CANDIDATES = [1, 2, 4, 8, 16]

@enum JobType begin
    GenerateGreensJob
    GenerateRSVDJob
    ComputeBoundsJob
end
const ORDERED_JOBS = [GenerateGreensJob, GenerateRSVDJob, ComputeBoundsJob]

cost_job(job::JobType) = job == GenerateGreensJob ? CostModel.GenerateGreens :
                         job == GenerateRSVDJob ? CostModel.GenerateRSVD :
                         CostModel.ComputeBounds

function main_file(job::JobType)
    job == GenerateGreensJob && return "generate_green.jl"
    job == GenerateRSVDJob && return "generate_rsvd.jl"
    job == ComputeBoundsJob && return "compute_bounds.jl"
    error("Unknown job type: $job")
end

function job_var_name(job::JobType)
    job == GenerateGreensJob && return "g0_job"
    job == GenerateRSVDJob && return "rsvd_job"
    job == ComputeBoundsJob && return "bounds_job"
    error("Unknown job type: $job")
end

function previous_job(job::JobType)
    job == GenerateGreensJob && return nothing
    job == GenerateRSVDJob && return GenerateGreensJob
    job == ComputeBoundsJob && return GenerateRSVDJob
    error("Unknown job type: $job")
end

function Base.string(job::JobType)
    job == GenerateGreensJob && return "Greens function generation"
    job == GenerateRSVDJob && return "RSVD generation"
    job == ComputeBoundsJob && return "bounds computation"
    error("Unknown job type: $job")
end

"Does this job run on the GPU? The Green function job deliberately does not."
uses_gpu(job::JobType) = job != GenerateGreensJob

struct ClusterConfig
    name::String
    has_slurm::Bool
    max_vram_GB::Int
    max_host_GB::Int
    max_cores::Int
    preload_dir::String
    project_dir::String
    scratch_dir::String
    code_dir::String
end

num_threads(config::ClusterConfig) = config.has_slurm ? "\\\$SLURM_CPUS_PER_TASK" : "auto"

cc_project_dir(server) =
    server in CC_RRG_CLUSTERS ?
    "/home/$(CC_UNAME)/projects/$(CC_RRG_NAME)/$(CC_UNAME)/Photonic-System-Channels/projects/" :
    "/home/$(CC_UNAME)/projects/$(CC_DEFAULT_GROUP_NAME)/$(CC_UNAME)/Photonic-System-Channels/projects/"

function ClusterConfig(server::AbstractString)
    if server == "molering"
        return ClusterConfig(server, false,
                             48,  # NVIDIA A6000
                             480, # host RAM
                             32,
                             MOLERING_PRELOAD_DIR, MOLERING_PROJECT_DIR,
                             MOLERING_SCRATCH_DIR, MOLERING_CODE_DIR)
    elseif server == "narval"
        return ClusterConfig(server, true,
                             40,  # NVIDIA A100-SXM4-40GB
                             240,
                             12,
                             CC_PRELOAD_DIR, cc_project_dir(server),
                             CC_SCRATCH_DIR, CC_CODE_DIR)
    elseif server == "fir"
        return ClusterConfig(server, true,
                             80,  # NVIDIA H100 80GB HBM3
                             240,
                             12,
                             CC_PRELOAD_DIR, cc_project_dir(server),
                             CC_SCRATCH_DIR, CC_CODE_DIR)
    end
    error("Unknown server: $server")
end

"""
    gpu_options(cluster) -> Vector{(name, vram_GB, compute_fraction)}

The GPU allocations available on a cluster, smallest first. MIG slices get a
fraction of the streaming multiprocessors as well as a fraction of the memory.
"""
function gpu_options(cluster::ClusterConfig)
    if cluster.name == "narval"
        return [("a100_1g.5gb", 5, 1 / 8), ("a100_2g.10gb", 10, 2 / 8),
                ("a100_3g.20gb", 20, 3 / 8), ("a100", 40, 1.0)]
    elseif cluster.name == "fir"
        return [("nvidia_h100_80gb_hbm3_1g.10gb", 10, 1 / 8),
                ("nvidia_h100_80gb_hbm3_2g.20gb", 20, 2 / 8),
                ("nvidia_h100_80gb_hbm3_3g.40gb", 40, 3 / 8),
                ("h100", 80, 1.0)]
    elseif cluster.name == "molering"
        return [("a6000", 48, 1.0)]
    end
    error("GPU options not implemented for cluster: $(cluster.name)")
end

"""
    choose_gpu(cluster, vram_GB) -> (name, compute_fraction)

Smallest allocation whose memory fits. A slice is cheaper to schedule but slower,
so `compute_fraction` is used to stretch the time request.
"""
function choose_gpu(cluster::ClusterConfig, vram_GB::Real)
    options = gpu_options(cluster)
    for (name, capacity, fraction) in options
        vram_GB <= capacity && return (name, fraction)
    end
    return (options[end][1], options[end][3])
end

"""
    Experiment

One parameter combination. The sender/receiver form (`mediator === nothing`) is
the one the cost model covers; a mediator is still accepted so old sweeps can be
reproduced, but its resources fall back to a crude heuristic.

`separation` is the surface-to-surface gap along x in wavelengths, which is what
`--rs-sep` takes.
"""
struct Experiment
    sender_cells::NTuple{3,Int}
    mediator_cells::Union{Nothing,NTuple{3,Int}}
    receiver_cells::NTuple{3,Int}
    separation::Union{Nothing,Rational{Int}}
    sm_separation::Union{Nothing,NTuple{3,Rational{Int}}}
    mr_separation::Union{Nothing,NTuple{3,Rational{Int}}}
    scale::Rational{Int}
    chi::ComplexF64
    rank::Int
    oversamples::Int
    power_iters::Int
end

"""
    sr_experiment(; cells, separation, scale, chi, rank, oversamples, power_iters,
                    receiver_cells=cells)

A single sender/receiver experiment. `scale` follows `SMRSystem`'s convention: a
positive rational is the isotropic cell size in wavelengths, and a *negative* one
means anisotropic cells of `(1//32, |scale|, |scale|)`.
"""
sr_experiment(; cells::NTuple{3,Int}, separation::Rational{Int},
              scale::Rational{Int}=1 // 32, chi::ComplexF64=13.6 + 0.05im,
              rank::Int, oversamples::Int=50, power_iters::Int=14,
              receiver_cells::NTuple{3,Int}=cells) =
    Experiment(cells, nothing, receiver_cells, separation, nothing, nothing,
               scale, chi, rank, oversamples, power_iters)

"""
    sr_sweep(; separations, kwargs...)

One `Experiment` per separation, everything else held fixed. This is what replaces
the old `repeat([...], num_experiments)` columns: the sweep variable is written
once and nothing else has to be kept in sync with its length.
"""
sr_sweep(; separations::AbstractVector{Rational{Int}}, kwargs...) =
    [sr_experiment(; separation=sep, kwargs...) for sep in separations]

"""
    smr_experiment(; sender_cells, mediator_cells, receiver_cells,
                     sm_separation, mr_separation, scale, chi, rank, ...)

A sender/mediator/receiver experiment. The calibrated cost model does not cover
these -- only the `mediator === nothing` pipeline was measured -- so their requests
come from `fallback_resources`, which is a deliberately generous guess off the
largest volume. Fine for reproducing an old sweep; do not expect the requests to
be tight.
"""
smr_experiment(; sender_cells::NTuple{3,Int}, mediator_cells::NTuple{3,Int},
               receiver_cells::NTuple{3,Int},
               sm_separation::NTuple{3,Rational{Int}},
               mr_separation::NTuple{3,Rational{Int}},
               scale::Rational{Int}=1 // 32, chi::ComplexF64=13.6 + 0.05im,
               rank::Int, oversamples::Int=50, power_iters::Int=14) =
    Experiment(sender_cells, mediator_cells, receiver_cells, nothing,
               sm_separation, mr_separation, scale, chi, rank, oversamples, power_iters)

is_sr(exp::Experiment) = isnothing(exp.mediator_cells)

function to_smr_system(exp::Experiment)
    return SMRSystem(exp.sender_cells, exp.mediator_cells, exp.receiver_cells,
                     exp.sm_separation, exp.mr_separation,
                     is_sr(exp) ? (exp.separation, 0 // 1, 0 // 1) : nothing,
                     exp.scale, exp.chi)
end

to_rsvd_params(exp::Experiment) = RSVDParams(exp.rank, exp.oversamples, exp.power_iters)

"""
    to_cost_point(exp, threads)

The cost model's view of an experiment. Note what is *not* here: the union of the
sender and receiver volumes. Nothing in the sender/receiver pipeline builds an
operator on that bounding box -- the "universe" is the concatenated
`[sender; receiver]` vector and a four-block multi-region operator -- so the gap
between the bodies contributes nothing to cost. Using `prod(union.cel)` is what
made a 10000-wavelength separation ask for terabytes.
"""
function to_cost_point(exp::Experiment, threads::Int)
    scale = exp.scale < 0 ? (1 // 32, abs(exp.scale), abs(exp.scale)) :
            (exp.scale, exp.scale, exp.scale)
    return SRPoint(exp.sender_cells, exp.receiver_cells;
                   scale=scale, separation=exp.separation,
                   rank=exp.rank, oversamples=exp.oversamples,
                   power_iters=exp.power_iters, threads=threads,
                   num_pos=ceil(Int, NUM_POS_FRACTION * exp.rank))
end

# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #

"""
    Resources

What one job asks the scheduler for, plus the raw predictions it came from so the
summary table can show its work.
"""
struct Resources
    time_s::Int
    host_GB::Int
    vram_GB::Int
    cores::Int
    gpu_name::String
    gpu_fraction::Float64
    over_vram::Bool
end

"""
    GPU_JOB_CORES

Cores for the GPU jobs. They are not CPU-bound, but they are not single-threaded
either: the RSVD job pulls an `N_u x k` eigenvector block back to the host and
writes it through JLD2, and the bounds job reads one back. A couple of cores keeps
that and the garbage collector from serialising against the device work, and asking
for more would burn allocation on idle cores.
"""
const GPU_JOB_CORES = 2

"""
    choose_cores(job, exp, cluster, coeffs) -> Int

For the Green-function job: the fewest cores that bring the predicted wall time
under `TARGET_WALL_TIME_S`. Compute Canada charges core-seconds and the quadrature
loops scale sublinearly, so asking for more cores than the wall-time target needs
spends allocation without buying priority. For the GPU jobs: a fixed small count.
"""
function choose_cores(job::JobType, exp::Experiment, cluster::ClusterConfig,
                      coeffs::Coefficients)
    uses_gpu(job) && return min(GPU_JOB_CORES, cluster.max_cores)
    is_sr(exp) || return min(4, cluster.max_cores)
    candidates = filter(<=(cluster.max_cores), CORE_CANDIDATES)
    isempty(candidates) && return cluster.max_cores
    for cores in candidates
        t = predict(CostModel.GenerateGreens, to_cost_point(exp, cores), coeffs).time_s
        t <= TARGET_WALL_TIME_S && return cores
    end
    return last(candidates)
end

"""
    fallback_resources(job, exp, cluster)

Resources for a mediator system, which the calibrated model does not cover. Scales
off the largest single volume with the old empirical constants; deliberately
generous, and warned about at generation time.
"""
function fallback_resources(job::JobType, exp::Experiment, cluster::ClusterConfig)
    volume = 3 * maximum(prod, filter(!isnothing,
                                      [exp.sender_cells, exp.mediator_cells, exp.receiver_cells]))
    c = exp.rank + exp.oversamples
    if job == GenerateGreensJob
        time_s = ceil(Int, 2.15e-6 * volume * log2(volume) * 4 + 30 * 60)
        host_GB = ceil(Int, (1.7e9 + 800 * volume) * 1e-9 * 1.5)
        vram_GB = 0
    else
        time_s = ceil(Int, 1.5e-8 * (2 + 2 * exp.power_iters) * c * volume * log2(volume) + 3600)
        host_GB = ceil(Int, (2.0e9 + 320 * c * volume) * 1e-9 * 1.5)
        vram_GB = ceil(Int, (1.5e9 + 320 * c * volume) * 1e-9 * 1.5)
    end
    return time_s, host_GB, vram_GB
end

function resources_for(job::JobType, exp::Experiment, cluster::ClusterConfig,
                       coeffs::Coefficients, cores::Int)
    if is_sr(exp)
        p = predict(cost_job(job), to_cost_point(exp, cores), coeffs)
        time_s, host_bytes, vram_bytes = p.time_s, p.host_bytes, p.vram_bytes
        host_GB = ceil(Int, host_bytes / 1e9)
        vram_GB = ceil(Int, vram_bytes / 1e9)
    else
        time_s, host_GB, vram_GB = fallback_resources(job, exp, cluster)
    end

    over_vram = false
    gpu_name, fraction = "", 1.0
    if uses_gpu(job)
        if vram_GB > cluster.max_vram_GB
            over_vram = true
        end
        gpu_name, fraction = choose_gpu(cluster, vram_GB)
        # A MIG slice has a fraction of the compute as well as a fraction of the
        # memory, and the calibration was taken on a whole GPU.
        time_s /= fraction
    end

    host_GB = max(MIN_MEMORY_GB, min(host_GB, cluster.max_host_GB))
    return Resources(max(MIN_TIME_S, ceil(Int, time_s)), host_GB, vram_GB,
                     cores, gpu_name, fraction, over_vram)
end

function seconds2string(seconds::Real)
    hours = floor(Int, seconds / 3600)
    mins = floor(Int, (seconds - hours * 3600) / 60)
    secs = round(Int, seconds - hours * 3600 - mins * 60)
    if secs >= 60
        mins += div(secs, 60)
        secs = mod(secs, 60)
    end
    if mins >= 60
        hours += div(mins, 60)
        mins = mod(mins, 60)
    end
    with_zeros(x) = lpad(string(x), 2, '0')
    return "$(with_zeros(hours)):$(with_zeros(mins)):$(with_zeros(secs))"
end

# --------------------------------------------------------------------------- #
# Naming and command line arguments
# --------------------------------------------------------------------------- #

rational2string(r::Rational, separator="//") = "$(numerator(r))$separator$(denominator(r))"

function experiment_name(smr::SMRSystem)
    s = sender(smr)
    m = mediator(smr)
    r = receiver(smr)
    if isnothing(m)
        sep = rs_separation(smr)[1] # Assume only x-separation for heat transfer
        return "\\($(join(s.cel, ","))\\)_$(rational2string(sep, "ss"))_\\($(join(r.cel, ","))\\)@\\($(join(rational2string.(s.scl, "ss"), ","))\\)"
    end
    sm_sep = sm_separation(smr)
    sm_sep_string = "\\($(rational2string(sm_sep[1])), $(rational2string(sm_sep[2])), $(rational2string(sm_sep[3]))\\)"
    mr_sep = mr_separation(smr)
    mr_sep_string = "\\($(rational2string(mr_sep[1])), $(rational2string(mr_sep[2])), $(rational2string(mr_sep[3]))\\)"
    return "\\($(join(s.cel, ","))\\)_$(sm_sep_string)_\\($(join(m.cel, ","))\\)_$(mr_sep_string)_\\($(join(r.cel, ","))\\)@\\($(join(rational2string.(s.scl, "ss"), ","))\\)"
end

function heat_transfer_args(smr::SMRSystem, params::RSVDParams)
    s = sender(smr)
    r = receiver(smr)
    sender_string = "\\($(join(s.cel, ","))\\)"
    receiver_string = "\\($(join(r.cel, ","))\\)"
    sep = rs_separation(smr)[1] # Assume only x-separation for heat transfer
    rs_sep_string = "\\($(rational2string(sep)),0//1,0//1\\)"
    scale_string = allequal(s.scl) ? rational2string(s.scl[1]) : rational2string(-s.scl[2]) # anisotropic hack
    chi_string = "$(real(χ(smr)))+$(imag(χ(smr)))im"
    name_string = experiment_name(smr)
    design_string = "rs" # Design the entire region
    return "--sender $(sender_string) --receiver $(receiver_string) --rs-sep $(rs_sep_string) --scale $(scale_string) --chi $(chi_string) --design $(design_string) --components $(params.rank) --oversamples $(params.oversamples) --power-iterations $(params.power_iter) --name $(name_string)"
end

function smr_args(smr::SMRSystem, params::RSVDParams)
    s = sender(smr)
    m = mediator(smr)
    r = receiver(smr)
    sender_string = "\\($(join(s.cel, ","))\\)"
    mediator_string = "\\($(join(m.cel, ","))\\)"
    receiver_string = "\\($(join(r.cel, ","))\\)"
    sm_sep = sm_separation(smr)
    sm_sep_string = "\\($(rational2string(sm_sep[1])), $(rational2string(sm_sep[2])), $(rational2string(sm_sep[3]))\\)"
    mr_sep = mr_separation(smr)
    mr_sep_string = "\\($(rational2string(mr_sep[1])), $(rational2string(mr_sep[2])), $(rational2string(mr_sep[3]))\\)"
    scale_string = "$(rational2string(s.scl[1]))"
    chi_string = "$(real(χ(smr)))+$(imag(χ(smr)))im"
    name_string = experiment_name(smr)
    design_string = "m" # Design the mediator region
    return "--sender $(sender_string) --mediator $(mediator_string) --receiver $(receiver_string) --sm-sep $(sm_sep_string) --mr-sep $(mr_sep_string) --scale $(scale_string) --chi $(chi_string) --design $(design_string) --components $(params.rank) --oversamples $(params.oversamples) --power-iterations $(params.power_iter) --name $(name_string)"
end

args(smr::SMRSystem, params::RSVDParams) =
    isnothing(mediator(smr)) ? heat_transfer_args(smr, params) : smr_args(smr, params)

# --------------------------------------------------------------------------- #
# Script generation
# --------------------------------------------------------------------------- #

function slurm_header_footer(job::JobType, cluster::ClusterConfig, smr::SMRSystem,
                             res::Resources, dependency::Union{Nothing,JobType}=nothing)
    var_name = job_var_name(job)
    header = "$var_name=\$(sbatch \\\n"
    if !isnothing(dependency)
        header *= "    --dependency=afterok:\${$(job_var_name(dependency))} \\\n"
    end
    header *= """    --job-name=$(PROJECT_NAME)_$(experiment_name(smr)) \\
    --output=$(cluster.project_dir)/$(PROJECT_NAME)/logs/$(experiment_name(smr))_%j.out \\
    --account=$(cluster.name in CC_RRG_CLUSTERS ? CC_RRG_NAME : CC_DEFAULT_GROUP_NAME) \\
    --time=$(seconds2string(res.time_s)) \\
    --cpus-per-task=$(res.cores) \\
    --mem=$(res.host_GB)G \\
    --chdir=$(cluster.code_dir) \\
"""
    if uses_gpu(job)
        header *= """    --gpus=$(res.gpu_name):1 \\\n"""
    end
    header *= """    --export=ALL \\
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
srun """
    footer = """EOF
)
$var_name=\${$var_name##* }
sleep 0.05
"""
    return header, footer
end

"""
    job_launcher_script(jobs, cluster, experiments) -> (script, plan)

Build the launcher. Returns the script text and the per-experiment resource plan
so the caller can print a summary.
"""
function job_launcher_script(jobs::AbstractVector{JobType}, cluster::ClusterConfig,
                             experiments::AbstractVector{Experiment})
    coeffs = coefficients_for(cluster.name)
    coeffs.calibrated ||
        @warn "Cluster '$(cluster.name)' has no calibrated cost model; requests are analytic guesses. Run the harness in bench/ (see bench/README.md)."

    script = """
#!/bin/bash

# Job launcher generated on $(now()) by create_jobs.jl
# Cost model: $(coeffs.calibrated ? "calibrated for $(coeffs.name)" : "UNCALIBRATED defaults")

echo Running job launcher for $(join(string.(jobs), ", "))
echo There $(length(experiments) > 1 ? "are" : "is") $(length(experiments)) experiment$(length(experiments) > 1 ? "s" : "") to launch
echo We are expecting to be on $(cluster.name)

# Change to code directory
cd $(cluster.code_dir)

# Create scratch, preload, and project directories if they don't exist
mkdir -p $(cluster.scratch_dir)/$(PROJECT_NAME)/
mkdir -p $(cluster.preload_dir)
mkdir -p $(cluster.project_dir)/$(PROJECT_NAME)/
"""
    if cluster.has_slurm
        script *= "mkdir -p $(cluster.project_dir)/$(PROJECT_NAME)/logs/ # Directory for slurm logs\n"
    end
    script *= "\n# Job submission commands follow\n\n"

    plan = Tuple{Experiment,Dict{JobType,Resources}}[]
    for exp in experiments
        smr = to_smr_system(exp)
        params = to_rsvd_params(exp)

        resources = Dict{JobType,Resources}()
        for job in ORDERED_JOBS
            job in jobs || continue
            cores = choose_cores(job, exp, cluster, coeffs)
            resources[job] = resources_for(job, exp, cluster, coeffs, cores)
        end
        push!(plan, (exp, resources))

        for job in ORDERED_JOBS
            job in jobs || continue
            res = resources[job]
            if res.over_vram
                @warn "$(experiment_name(smr)) $(string(job)) needs about $(res.vram_GB) GB of VRAM, more than the $(cluster.max_vram_GB) GB on $(cluster.name). Submitting anyway on a whole GPU; expect an out-of-memory failure."
            end

            header, footer = "", ""
            if cluster.has_slurm
                dependency = previous_job(job) in jobs ? previous_job(job) : nothing
                header, footer = slurm_header_footer(job, cluster, smr, res, dependency)
            end
            script *= header

            job_args = args(smr, params)
            job_args *= uses_gpu(job) ? " --gpu 0" : " --gpu false"
            script *= "julia --project=. -t $(num_threads(cluster)) $(main_file(job)) $job_args --project $(cluster.project_dir)/$(PROJECT_NAME)/ --scratch $(cluster.scratch_dir)/$(PROJECT_NAME)/\n"

            script *= footer
        end
        script *= "\n"
    end
    return script, plan
end

"""
    print_plan(plan, jobs, cluster)

Summary table of what was requested and why. Worth reading before submitting: it
is where a rank that does not fit in VRAM, or a bounds job that has quietly become
the expensive one, becomes obvious.
"""
function print_plan(plan, jobs::AbstractVector{JobType}, cluster::ClusterConfig)
    println(stderr)
    println(stderr, "Resource plan for $(length(plan)) experiments on $(cluster.name):")
    println(stderr, "  ", rpad("experiment", 30), rpad("job", 8), rpad("time", 11),
            rpad("cores", 6), rpad("host", 8), "gpu")
    totals = Dict{JobType,Float64}()
    core_seconds = 0.0
    for (exp, resources) in plan
        label = is_sr(exp) ?
                "$(join(exp.sender_cells, "x")) sep=$(rational2string(exp.separation)) k=$(exp.rank)" :
                "$(join(exp.sender_cells, "x")) +mediator k=$(exp.rank)"
        first_row = true
        for job in ORDERED_JOBS
            job in jobs || continue
            res = resources[job]
            totals[job] = get(totals, job, 0.0) + res.time_s
            core_seconds += res.time_s * res.cores
            short = job == GenerateGreensJob ? "greens" :
                    job == GenerateRSVDJob ? "rsvd" : "bounds"
            gpu = uses_gpu(job) ? "$(res.gpu_name) ($(res.vram_GB) GB)" : "-"
            println(stderr, "  ", rpad(first_row ? label : "", 30), rpad(short, 8),
                    rpad(seconds2string(res.time_s), 11), rpad(res.cores, 6),
                    rpad("$(res.host_GB)G", 8), gpu)
            first_row = false
        end
    end
    println(stderr)
    for job in ORDERED_JOBS
        haskey(totals, job) || continue
        @printf(stderr, "  total requested %-24s %8.1f h\n", string(job), totals[job] / 3600)
    end
    @printf(stderr, "  total core-hours requested %13.1f\n", core_seconds / 3600)
    println(stderr, "  (requests, not predictions: they include the padding factors)")
    return nothing
end

# =========================================================================== #
# Experiment definitions
# =========================================================================== #

load_coefficients!(joinpath(@__DIR__, "bench"))

cluster = ClusterConfig("molering")
# cluster = ClusterConfig("fir")
# cluster = ClusterConfig("narval")

### Metasurface: lambda/4 cubes at lambda/32 cells, swept in separation
experiments = sr_sweep(
    cells=(8, 8, 8),
    separations=[10, 12, 14, 16, 18, 20, 22] .// 1,
    scale=1 // 32,
    chi=13.6 + 0.05im,
    rank=1350,
    oversamples=50,
    power_iters=14,
)

# Other sweeps that have been run, for reference:
#
#   # log-spaced separations from 1/32 to 300 wavelengths, ~250 points
#   separations = unique(round.(Int, logrange(1, 300 * 32, 250))) .// 32
#
#   # every cell separation from touching to 3 wavelengths, then coarse out to 300
#   separations = vcat(collect(0:(3 * 32)) .// 32, Rational.(collect(10:5:300)))
#
#   # 3-lambda cubes with anisotropic cells (1/32, 1/8, 1/8)
#   experiments = sr_sweep(cells=(96, 32, 32), separations=..., scale=-1//8, rank=800)
#
#   # waveguide: sender length swept instead of separation
#   experiments = [sr_experiment(cells=(l, 32, 32), receiver_cells=(8, 32, 32),
#                                separation=1//8, scale=1//32, rank=800)
#                  for l in 8:2:(6 * 32)]

command, plan = job_launcher_script(
    [GenerateGreensJob, GenerateRSVDJob, ComputeBoundsJob],
    cluster,
    experiments,
)

print(command)
print_plan(plan, [GenerateGreensJob, GenerateRSVDJob, ComputeBoundsJob], cluster)

mkpath(joinpath(@__DIR__, "jobs"))
script_path = joinpath(@__DIR__, "jobs", "launch_$(PROJECT_NAME).sh")
open(script_path, "w") do f
    write(f, command)
end
println(stderr, "\nJob launcher script written to $(script_path)")
if cluster.has_slurm
    println(stderr, "Copy it over:  scp \"$(script_path)\" \"$(CC_UNAME)@$(cluster.name).alliancecan.ca:$(cluster.code_dir)jobs/\"")
    println(stderr, "Then run it:   ssh $(CC_UNAME)@$(cluster.name).alliancecan.ca 'cd $(cluster.code_dir) && bash jobs/launch_$(PROJECT_NAME).sh'")
else
    println(stderr, "Copy it over:  scp \"$(script_path)\" \"$(MOLERING_UNAME)@molering:$(cluster.code_dir)jobs/\"")
    println(stderr, "Then run it:   ssh $(MOLERING_UNAME)@molering 'cd $(cluster.code_dir) && bash jobs/launch_$(PROJECT_NAME).sh'")
end
