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
# by the harness in `bench/` (see bench/README.md).

include(joinpath(@__DIR__, "bench", "cost_model.jl"))
using .CostModel

# const PROJECT_NAME = "narval_arxivV3_0p25x0p25x0p25_1350comps_50oversamples_32scale"
# const PROJECT_NAME = "fir_arxivV3_0p5x0p5x0p5_1350comps_50oversamples_32scale"
# const PROJECT_NAME = "nibi_arxivV3_1x1x1_1350comps_50oversamples_32scale"
# const PROJECT_NAME = "narval_arxivV3_1x1x1_1350comps_50oversamples_32scale_2"
# const PROJECT_NAME = "fir_arxivV3_2x2x2_1350comps_50oversamples_aniso-32-n16-n16scale"   # <- should be written 64-n32-n32scale, but I had a typo :( (it did still run at the right sizes though)
# const PROJECT_NAME = "narval_arxivV3_2x2x2_800comps_50oversamples_aniso-32-n16-n16scale" # <- should be written 64-n32-n32scale, but I had a typo :( (it did still run at the right sizes though)
# const PROJECT_NAME = "narval_arxivV3_4x4x4_400comps_50oversamples_aniso-128-n8-n8scale"
# const PROJECT_NAME = "fir_arxivV3_4x4x4_800comps_50oversamples_aniso-32-n8-n8scale"
# const PROJECT_NAME = "narval_arxivV3_sphere_2x2x2_800comps_50oversamples_aniso-64-n16-n16scale"

# const PROJECT_NAME = "narval_Ge_arxivV3_0p25x0p25x0p25_1350comps_50oversamples_32scale"
# const PROJECT_NAME = "narval_Ge_arxivV3_0p5x0p5x0p5_1350comps_50oversamples_32scale"
# const PROJECT_NAME = "narval_Ge500_arxivV3_1x1x1_1350comps_50oversamples_32scale"
# const PROJECT_NAME = "narval_Ge_arxivV3_2x2x2_800comps_50oversamples_aniso-64-n32-n32scale"
# const PROJECT_NAME = "narval_Ge_arxivV3_4x4x4_400comps_50oversamples_aniso-128-n8-n8scale"

# Germanium with ζ = 1000 (χ = 4.250 + 0.0342557im), 4000 components everywhere the
# universe is big enough to hold them. 1/4 λ has N_u = 3072 < 4000, so its "rank" is
# the full spectrum and it goes down the dense-exact path instead.
# const PROJECT_NAME = "narval_Ge1000_arxivV3_0p25x0p25x0p25_3072comps_50oversamples_32scale"
const PROJECT_NAME = "molering_Ge1000_arxivV3_0p25x0p25x0p25_3072comps_50oversamples_32scale"
# const PROJECT_NAME = "narval_Ge1000_arxivV3_0p5x0p5x0p5_4000comps_50oversamples_32scale"
# const PROJECT_NAME = "narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale"
# const PROJECT_NAME = "narval_Ge1000_arxivV3_2x2x2_4000comps_50oversamples_aniso-64-n32-n32scale"
# const PROJECT_NAME = "narval_Ge1000_arxivV3_4x4x4_4000comps_50oversamples_aniso-128-n8-n8scale"


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
    GREENS_LAUNCHER

How the Green-function stage is submitted. `:sbatch` or `:glost`.

- `:sbatch` (the default): one `sbatch` job per separation. A 333-separation sweep
  therefore spends 333 of the 1000 running+queued jobs Alliance's `MaxSubmit`
  allows, which is what pins these sweeps at 333 x 3 = 999 jobs.
- `:glost`: one `sbatch` job runs every separation's Green-function task through
  GLOST (CEA's Greedy Launcher Of Small Tasks). GLOST is an MPI farm: rank 0 is a
  manager, and ranks 1..N-1 pull lines out of a task file as they free up. One
  GLOST job occupies one queue slot no matter how many tasks it holds, so the 333
  greens jobs collapse to 1 and the slot budget goes to the GPU stages. All the
  tasks also share one node and one depot, so Julia's compile cache is warm after
  the first task instead of paying `CostModel.RECOMPILE_OVERHEAD_S` 333 times.

Only the CPU-only Green-function job can be farmed this way. The GPU stages need
one GPU each and stay as ordinary `sbatch` jobs. GLOST is an MPI program from the
Alliance module stack, so `:glost` requires a SLURM cluster. Molering has neither
the module nor a queue to relieve, and `validate_greens_launcher` refuses it there
at generation time rather than emitting a script that cannot run.
"""
const GREENS_LAUNCHER = :sbatch # :glost for the Alliance sweeps; molering refuses :glost

"""
    GLOST_NTASKS, GLOST_CPUS_PER_TASK, GLOST_MEM_GB

Shape of the one GLOST job, sized for a narval base node (64 Rome cores, 249 GB).

`GLOST_NTASKS` counts MPI ranks. Rank 0 is a manager that runs no task when
`size > 1`, so 13 ranks are 1 manager + 12 workers x 4 threads = 52 cores, which
leaves room for the OS on a 64-core node. Tasks are separate processes, so each one
may use `SLURM_CPUS_PER_TASK` threads: "serial" here only means one task per rank,
and 4 threads is where the Green-function job's `@threads` quadrature loops stop
paying (see `MOLERING_THREADS`, and `choose_cores`, which usually lands on 4-8 for
this job).

Memory is what keeps the worker count down: greens tasks measure ~2-4 GB each at
1 lambda and more at 4 lambda, and 12 of them share the node's RAM. `--mem=240G`
asks for essentially the whole node. Note, however, that `ClusterConfig.max_host_GB`
does not apply here: that number is the per-GPU billing bundle, and this job touches
no GPU.
"""
const GLOST_NTASKS = 13
const GLOST_CPUS_PER_TASK = 4
const GLOST_MEM_GB = 240

"Concurrently executing GLOST tasks: every rank but the manager."
const GLOST_WORKERS = GLOST_NTASKS - 1

"""
    GLOST_TIME_SAFETY, GLOST_TIME_SLACK_S, GLOST_MAX_TIME_S, GLOST_DRAIN_LEAD_S

Walltime policy for the farm. The request is

    min(GLOST_MAX_TIME_S,
        GLOST_TIME_SAFETY * sum(predicted greens time) / GLOST_WORKERS
        + GLOST_TIME_SLACK_S)

Dividing the total by the worker count ignores two things, and that is what the
safety factor pays for: the greedy schedule leaves workers idle at the tail (with a
10x spread in task cost, the last task can stretch the farm by a good fraction of
one task), and a task that lands on a busy node runs slower than the calibration.
The slack term covers the serial pre-step, the module loads and the first Julia
compile, which happen once and so do not divide by the worker count.

`GLOST_MAX_TIME_S` is Alliance's 7-day ceiling. Asking for more is a rejection, not
a longer job. If the formula hits the cap, split the task file by separation range
into several GLOST jobs, which are independent, rather than trimming the safety
factor.

`GLOST_DRAIN_LEAD_S` is how early Slurm signals the job so GLOST can finish the
tasks in flight and stop handing out new ones instead of being killed mid-task.
"""
const GLOST_TIME_SAFETY = 1.5
const GLOST_TIME_SLACK_S = 30 * 60
const GLOST_MAX_TIME_S = 7 * 24 * 3600
const GLOST_DRAIN_LEAD_S = 600

"""
    NUM_POS_FRACTION

Assumed fraction of the computed rank that has a positive `Asym(G⁰ᵤᵣ)` eigenvalue.
The bounds job runs ~11 dense `m x m` device pencil solves per positive
eigenvalue (the τ grid plus golden-section refinement, `m = num_pos`) and an
`O(num_pos² x evals)` probe loop on top, so this number matters a lot and is
only known after the RSVD has run.

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

"""
    ClusterConfig

# Fields
- `max_cores`, `max_host_GB`: the Alliance's published per-GPU bundle. Usage is
  billed as the largest of `gpus`, `cores / max_cores` and `host / max_host_GB`,
  so staying inside the bundle keeps a job costing exactly one GPU-equivalent;
  exceeding it costs priority without making the job faster.
"""
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

"""
    MOLERING_THREADS

Julia threads for the unscheduled machine. The molering thread scan measured the
Green-function job peaking at 16 of the 128 available and getting slower beyond:
Gila's FFTs are single-threaded and only the quadrature loops are `@threads`, so
the parallel fraction saturates early.
"""
const MOLERING_THREADS = 16

"""
    num_threads(cluster) -> String

The `-t` argument for Julia. Under SLURM that is whatever the scheduler gave us,
which is `choose_cores`' answer by construction.
"""
num_threads(config::ClusterConfig) =
    config.has_slurm ? "\\\$SLURM_CPUS_PER_TASK" : string(MOLERING_THREADS)

cc_project_dir(server) =
    server in CC_RRG_CLUSTERS ?
    "/home/$(CC_UNAME)/projects/$(CC_RRG_NAME)/$(CC_UNAME)/Photonic-System-Channels/projects/" :
    "/home/$(CC_UNAME)/projects/$(CC_DEFAULT_GROUP_NAME)/$(CC_UNAME)/Photonic-System-Channels/projects/"

function ClusterConfig(server::AbstractString)
    if server == "molering"
        return ClusterConfig(server, false,
                             48,  # NVIDIA A6000
                             480, # host RAM
                             128, # 64-core Threadripper Pro 5995WX with SMT
                             MOLERING_PRELOAD_DIR, MOLERING_PROJECT_DIR,
                             MOLERING_SCRATCH_DIR, MOLERING_CODE_DIR)
    elseif server == "narval"
        # Calcul Quebec. 159 GPU nodes, 4x A100-SXM4-40GB, 48 Milan cores, 498 GB.
        return ClusterConfig(server, true,
                             40,   # NVIDIA A100-SXM4-40GB
                             124,  # bundle: 124.5 GB per A100
                             12,   # bundle: 12 cores per A100
                             CC_PRELOAD_DIR, cc_project_dir(server),
                             CC_SCRATCH_DIR, CC_CODE_DIR)
    elseif server == "fir"
        # Simon Fraser, replaced Cedar. 160 GPU nodes, 4x H100 SXM5, 48 EPYC 9454
        # cores, 1125 GB. MIG is enabled on roughly half the GPU nodes.
        return ClusterConfig(server, true,
                             80,   # NVIDIA H100 80GB HBM3
                             288,  # bundle: 288 GB per H100
                             12,   # bundle: 12 cores per H100
                             CC_PRELOAD_DIR, cc_project_dir(server),
                             CC_SCRATCH_DIR, CC_CODE_DIR)
    elseif server == "nibi"
        # Waterloo, replaced Graham. 10 GPU nodes with 8x H100 SXM 80GB each,
        # 112 Intel Xeon 6 "Granite Rapids" cores, 2000 GB.
        return ClusterConfig(server, true,
                             80,   # NVIDIA H100 80GB
                             250,  # bundle: 250 GB per H100
                             14,   # bundle: 14 cores per H100
                             CC_PRELOAD_DIR, cc_project_dir(server),
                             CC_SCRATCH_DIR, CC_CODE_DIR)
    elseif server == "rorqual"
        # Calcul Quebec, replaced Beluga. GPU nodes are 4x H100 SXM5 80GB,
        # 64 AMD Genoa cores, 498 GB.
        return ClusterConfig(server, true,
                             80,   # NVIDIA H100 80GB
                             124,  # bundle: 124.5 GB per H100
                             16,   # bundle: 16 cores per H100
                             CC_PRELOAD_DIR, cc_project_dir(server),
                             CC_SCRATCH_DIR, CC_CODE_DIR)
    end
    error("Unknown server: $server. Known: molering, narval, fir, nibi, rorqual.")
end

"""
    gpu_options(cluster) -> Vector{(name, vram_GB, compute_fraction, host_GB)}

The GPU allocations available on a cluster, smallest first.

A MIG slice gets a fraction of the streaming multiprocessors as well as a fraction
of the memory, and it comes with a correspondingly smaller bundle of cores and
system RAM: `host_GB` is that bundle, and asking for more RAM than a slice's share
is billed as though several slices had been used, which defeats the point of
taking one.

The names are cluster-specific and a name the cluster does not define is a hard
`sbatch` rejection rather than a bad estimate, so they are transcribed from each
cluster's wiki page rather than guessed. Note that fir spells its H100 slices
`nvidia_h100_80gb_hbm3_*` while nibi and rorqual spell the same partitions
`h100_*`. Verify with `sinfo -o "%G" | sort -u` if a submission is refused.

The core bundle is not tracked here: the Green-function job never touches a GPU,
and the GPU jobs ask for `GPU_JOB_CORES`, which is inside every 2g/3g/whole-GPU
bundle. It does slightly exceed the 1g slice's 1.7 cores on fir and narval, which
costs about 18% more than one slice-equivalent on the smallest jobs in the sweep.
"""
function gpu_options(cluster::ClusterConfig)
    if cluster.name == "narval"
        return [("a100_1g.5gb", 5, 1 / 8, 17), ("a100_2g.10gb", 10, 2 / 8, 35),
                ("a100_3g.20gb", 20, 3 / 8, 62), ("a100", 40, 1.0, 124)]
    elseif cluster.name == "fir"
        return [("nvidia_h100_80gb_hbm3_1g.10gb", 10, 1 / 8, 41),
                ("nvidia_h100_80gb_hbm3_2g.20gb", 20, 2 / 8, 82),
                ("nvidia_h100_80gb_hbm3_3g.40gb", 40, 3 / 8, 144),
                ("h100", 80, 1.0, 288)]
    elseif cluster.name == "nibi"
        return [("h100_1g.10gb", 10, 1 / 8, 35), ("h100_2g.20gb", 20, 2 / 8, 71),
                ("h100_3g.40gb", 40, 3 / 8, 125), ("h100", 80, 1.0, 250)]
    elseif cluster.name == "rorqual"
        return [("h100_1g.10gb", 10, 1 / 8, 17), ("h100_2g.20gb", 20, 2 / 8, 35),
                ("h100_3g.40gb", 40, 3 / 8, 62), ("h100", 80, 1.0, 124)]
    elseif cluster.name == "molering"
        return [("a6000", 48, 1.0, 480)]
    end
    error("GPU options not implemented for cluster: $(cluster.name)")
end

"""
    choose_gpu(cluster, vram_GB, host_GB) -> (name, compute_fraction)

Smallest allocation that fits both the device memory and the host memory the job
needs. A slice is cheaper to schedule but slower, so `compute_fraction` is used to
stretch the time request.

Only the mediator systems reach this. A sender/receiver job goes through
`select_gpu` instead, since its memory demand is not a number we can compute before
naming a card: the card is what decides which storage path the job takes.
"""
function choose_gpu(cluster::ClusterConfig, vram_GB::Real, host_GB::Real)
    options = gpu_options(cluster)
    for (name, capacity, fraction, bundle_host_GB) in options
        vram_GB <= capacity && host_GB <= bundle_host_GB && return (name, fraction)
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

function to_cost_point(exp::Experiment, threads::Int)
    scale = exp.scale < 0 ? (1 // 32, abs(exp.scale), abs(exp.scale)) :
            (exp.scale, exp.scale, exp.scale)
    return SRPoint(exp.sender_cells, exp.receiver_cells;
                   scale=scale, separation=exp.separation,
                   rank=exp.rank, oversamples=exp.oversamples,
                   power_iters=exp.power_iters, threads=threads,
                   num_pos=ceil(Int, NUM_POS_FRACTION * exp.rank))
end

"""
    Resources

What one job asks the scheduler for, plus the raw predictions it came from so the
summary table can show its work.

# Fields beyond the request itself
- `mode`: the storage path the cost model sized this request for. `:in_memory`,
  `:panel` or `:dense_exact` for a GPU job, `:host` for the Green-function job. It is
  not a knob: `src/rsvd.jl` picks the path at run time from the card it finds itself
  on, and the field only records which (card, path) pairing the request was built
  around, so that the plan table can be read against the allocation table in
  `FUNICULAR_PLAN.md`. If the mode on a row is a surprise, the time and memory on that
  row are about a different algorithm than the one you have in mind.
- `vram_floor_GB`: the least device memory the job can be squeezed into, where
  `vram_GB` is what it will take if the card has room. Only the floor decides
  feasibility (see `select_gpu`), so only the floor belongs in the `over_vram`
  warning: the capped request would name a number that is an artifact of the cap.
- `host_uncapped_GB`: what the cost model asked for before `cluster.max_host_GB`
  clamped it, which is the same thing `vram_floor_GB` is for device memory: the
  demand, as opposed to the request. On the panel path the two differ at the top
  of the sweep, and the difference is not an error. Funicular's plan has an NVMe
  tier (`scratch_dir`, `\$SLURM_TMPDIR`; see `residency_plan` in `src/common.jl`),
  so a host budget below the whole sketch means panels spill to node-local disk
  rather than that the job dies. Capping at the bundle and letting it spill is the
  intended behaviour, since exceeding the bundle is billed as several GPUs and buys
  nothing. `print_plan` says so on the rows where it happens, because a silent cap
  looks identical to a request that happened to fit.
"""
struct Resources
    time_s::Int
    host_GB::Int
    vram_GB::Int
    cores::Int
    gpu_name::String
    gpu_fraction::Float64
    over_vram::Bool
    mode::Symbol
    vram_floor_GB::Int
    host_uncapped_GB::Int
end

"""
    mode_label(mode) -> String

Short name for the plan table's `mode` column. `:host` is the Green-function job,
which has no device storage path to report.
"""
mode_label(mode::Symbol) =
    mode == :dense_exact ? "dense" :
    mode == :in_memory ? "inmem" :
    mode == :panel ? "panel" :
    mode == :host ? "-" : string(mode)

"""
    GPU_JOB_CORES

Cores for the GPU jobs. They are not CPU-bound, but they are not single-threaded
either: the RSVD job pulls an `N_u x k` eigenvector block back to the host and
writes it through JLD2, the bounds job reads one back and runs its Brent root
finds in single-threaded host Julia (its pencil eigendecompositions run on the
device). A couple of cores keeps the host share and the garbage collector from
serialising against the device work, and asking for more would burn allocation
on idle cores.
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
    # Without a scheduler there is nothing to ask for, but the prediction still
    # has to match what the job will really run with, since the model divides the
    # quadrature term by the thread efficiency.
    cluster.has_slurm || return MOLERING_THREADS
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
    # This path bypasses `predict`, so the recompile tax is added here instead.
    return time_s + ceil(Int, CostModel.RECOMPILE_OVERHEAD_S), host_GB, vram_GB
end

"""
    select_gpu(job, exp, cluster, coeffs, cores) -> NamedTuple

The allocation for one GPU job, chosen by trying every allocation the cluster
offers, smallest first, and taking the first one the job fits on.

One prediction is not enough, because the question is circular: the prediction sizes
the request, the request picks the card, and the card picks the algorithm.
`src/rsvd.jl` reads `CUDA.total_memory()` and switches between three storage paths on
what it finds (`CostModel.rsvd_mode`): a device-resident sketch when it fits,
Funicular's host-resident panel matrices when it does not, and a dense exact
eigendecomposition when the universe is smaller than the rank. Those paths cost
different amounts, and they spend memory in different places. The same 1 λ, k = 1350
RSVD wants 77 GB of VRAM and 10 GB of host memory in memory, or 18 GB of VRAM and
13 GB of host memory in panels. There is therefore no answer to "how much VRAM does
this job need?" until a card is named. Answer it with the in-memory path's number and
we book a whole A100 for a job that fits a 20 GB slice.

Each candidate is predicted at its own capacity, and accepted when both padded
requests are inside the allocation's bundle:

  * `vram_bytes <= capacity_GB`. On the panel path this is exact: Funicular
    preallocates its staging buffers and nothing inside a sweep allocates, so the
    request and the floor are the same number. On the in-memory path the request
    carries `rsvd_vram_factor`, that is, how large CUDA.jl's pool was measured growing
    while it held garbage, so testing against it is conservative and the job would
    probably also fit a smaller slice under memory pressure. That is deliberate, since
    an OOM costs the whole job while a slice that is too large only costs some
    allocation, and it is why `over_vram` below uses the floor instead.
  * `host_GB <= bundle_host_GB`. The Alliance bills the largest of GPUs,
    `cores / max_cores` and `host / max_host_GB`, so a slice whose host bundle we
    exceed is billed as several slices and there was no point taking one. This is
    the binding constraint at the large end: at 4 λ, k = 4000 the panel path's 121 GB
    of pinned host memory rules out every slice on narval, whatever the VRAM says.

If nothing fits, we return the largest allocation with its request capped at the card
and `over_vram` set from that card's floor, the least memory the job can be squeezed
into. Device memory here is churn-elastic: a job whose comfortable footprint was
137 GB was measured completing in 71 GB on an 80 GB card once pressure forced the
allocator to collect, so warning on the comfortable number would refuse jobs that
demonstrably run.
"""
function select_gpu(job::JobType, exp::Experiment, cluster::ClusterConfig,
                    coeffs::Coefficients, cores::Int)
    pt = to_cost_point(exp, cores)
    candidate = nothing
    for (name, capacity_GB, fraction, bundle_host_GB) in gpu_options(cluster)
        p = predict(cost_job(job), pt, coeffs; vram_capacity_bytes=capacity_GB * 1e9)
        host_uncapped_GB = ceil(Int, p.host_bytes / 1e9)
        host_GB = max(MIN_MEMORY_GB, min(host_uncapped_GB, cluster.max_host_GB))
        vram_GB = ceil(Int, p.vram_bytes / 1e9)
        floor_GB = ceil(Int, p.vram_floor_bytes / 1e9)
        #=
        A MIG slice has a fraction of the SMs as well as a fraction of the memory,
        and the calibration was taken on a whole GPU, but only the device-bound
        share of the work slows down. Stretching the whole prediction
        over-requested by up to 8x on the small-body sweeps, whose bounds job is
        dominated by a single-threaded host-side root find. On the panel path the
        PCIe sweep term is in the host share for the same reason: a slice gets a
        fraction of the SMs, not a fraction of the bus.
        =#
        candidate = (gpu_name=name, fraction=fraction, mode=p.mode,
                     time_s=(p.time_s - p.device_time_s) + p.device_time_s / fraction,
                     host_GB=host_GB, vram_GB=min(vram_GB, capacity_GB),
                     vram_floor_GB=floor_GB, over_vram=floor_GB > capacity_GB,
                     host_uncapped_GB=host_uncapped_GB)
        vram_GB <= capacity_GB && host_GB <= bundle_host_GB && return candidate
    end
    # Nothing fitted: the last candidate is the largest allocation the cluster has,
    # already carrying its own floor-based `over_vram`.
    return candidate
end

function resources_for(job::JobType, exp::Experiment, cluster::ClusterConfig,
                       coeffs::Coefficients, cores::Int)
    if is_sr(exp) && uses_gpu(job)
        s = select_gpu(job, exp, cluster, coeffs, cores)
        return Resources(max(MIN_TIME_S, ceil(Int, s.time_s)), s.host_GB, s.vram_GB,
                         cores, s.gpu_name, s.fraction, s.over_vram, s.mode,
                         s.vram_floor_GB, s.host_uncapped_GB)
    end

    if is_sr(exp)
        # The Green-function job: CPU only, so there is no card to select and no
        # device-bound share to stretch. `predict` reports `mode = :host`.
        p = predict(cost_job(job), to_cost_point(exp, cores), coeffs)
        host_uncapped_GB = ceil(Int, p.host_bytes / 1e9)
        host_GB = max(MIN_MEMORY_GB, min(host_uncapped_GB, cluster.max_host_GB))
        return Resources(max(MIN_TIME_S, ceil(Int, p.time_s)), host_GB, 0, cores,
                         "", 1.0, false, p.mode, 0, host_uncapped_GB)
    end

    #=
    Mediator system: no calibrated model, so no mode-aware selection either.
    `fallback_resources` bypasses `predict` entirely and its numbers are
    in-memory-shaped guesses, which is what the `:in_memory` label below records. It
    reports no host/device split, so the whole prediction is treated as device-bound
    and stretched by the slice fraction.
    =#
    time_s, host_uncapped_GB, vram_GB = fallback_resources(job, exp, cluster)
    host_GB = max(MIN_MEMORY_GB, min(host_uncapped_GB, cluster.max_host_GB))
    uses_gpu(job) || return Resources(max(MIN_TIME_S, ceil(Int, time_s)), host_GB, 0,
                                      cores, "", 1.0, false, :host, 0, host_uncapped_GB)
    over_vram = vram_GB > cluster.max_vram_GB
    floor_GB = vram_GB
    vram_GB = min(vram_GB, cluster.max_vram_GB)
    gpu_name, fraction = choose_gpu(cluster, vram_GB, host_GB)
    return Resources(max(MIN_TIME_S, ceil(Int, time_s / fraction)), host_GB, vram_GB,
                     cores, gpu_name, fraction, over_vram, :in_memory, floor_GB,
                     host_uncapped_GB)
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

"""
    slurm_time_string(seconds) -> String

`--time` in the `D-HH:MM:SS` form once a request runs past a day. Slurm does accept
the `168:00:00` that `seconds2string` gives for a week, but the day form is what the
wiki quotes and is easier to check against the 7-day ceiling. Under 24 h this returns
exactly what `seconds2string` does, so the per-experiment jobs are unaffected.
"""
function slurm_time_string(seconds::Real)
    days = floor(Int, seconds / 86400)
    days == 0 && return seconds2string(seconds)
    return "$(days)-$(seconds2string(seconds - days * 86400))"
end

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
    --chdir=$(cluster.code_dir) \\
"""
    header *= "    --cpus-per-task=$(res.cores) \\\n"
    header *= "    --mem=$(res.host_GB)G \\\n"
    if uses_gpu(job)
        header *= "    --gpus=$(res.gpu_name):1 \\\n"
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
    job_command(job, cluster, smr, params) -> String

The one `julia` invocation a job runs. The GLOST task file has to reproduce it
character for character, since a task line that differs from what the `:sbatch` path
runs is a silently different experiment, and the `--name` in it is what the RSVD job
later looks for.

Note the `-t` argument is `\\\$SLURM_CPUS_PER_TASK` (escaped) under SLURM: these
commands go into an unquoted `<<EOF` heredoc, and the submitting shell, where the
variable is not set, would otherwise expand it. The GLOST task file is not a heredoc,
so `glost_task_line` strips the escape.

`gpu_index` is which CUDA device the GPU jobs pin (`CUDA.device!` in each main
file). Under SLURM it stays 0, since the allocation exposes exactly the card it
granted; on molering it is how two launcher scripts share the two A6000s.
"""
function job_command(job::JobType, cluster::ClusterConfig, smr::SMRSystem,
                     params::RSVDParams; gpu_index::Int=0)
    job_args = args(smr, params)
    job_args *= uses_gpu(job) ? " --gpu $(gpu_index)" : " --gpu false"
    return "julia --project=. -t $(num_threads(cluster)) $(main_file(job)) $job_args --project $(cluster.project_dir)/$(PROJECT_NAME)/ --scratch $(cluster.scratch_dir)/$(PROJECT_NAME)/"
end

"""
    validate_greens_launcher(cluster)

Fail at generation time rather than at submission time. A `:glost` script needs
`sbatch`, `srun` and an MPI-launched `glost_launch`, so on a machine with no
scheduler there is nothing to launch it with and no queue slots to free. Molering is
therefore a hard error here.
"""
function validate_greens_launcher(cluster::ClusterConfig)
    GREENS_LAUNCHER in (:sbatch, :glost) ||
        error("GREENS_LAUNCHER = $(repr(GREENS_LAUNCHER)) is not a launcher. Use :sbatch (one job per separation) or :glost (one farmed job).")
    if GREENS_LAUNCHER == :glost && !cluster.has_slurm
        error("GREENS_LAUNCHER = :glost needs a SLURM cluster: the farm is an MPI " *
              "`srun glost_launch` inside one sbatch job, and glost comes from the " *
              "Alliance module stack. Cluster '$(cluster.name)' has no scheduler " *
              "(and no queue limit to work around), so set GREENS_LAUNCHER = :sbatch, " *
              "which on that machine just runs the jobs in sequence.")
    end
    return nothing
end

"The task file lives next to the launcher it belongs to, and is copied over with it."
glost_tasks_filename() = "greens_tasks_$(PROJECT_NAME).txt"

"""
    glost_greens_time_s(exp, cluster, coeffs) -> Float64

Predicted wall time of one farmed Green-function task, at `GLOST_CPUS_PER_TASK`
threads. That is what the task really gets, and the model divides the quadrature term
by the thread efficiency, so the thread count has to match.

This is conservative on purpose: `predict` adds `RECOMPILE_OVERHEAD_S` to every task,
but under GLOST only the first task on the node pays it. We leave it in, since it
costs nothing but a slightly longer walltime request.
"""
glost_greens_time_s(exp::Experiment, cluster::ClusterConfig, coeffs::Coefficients) =
    is_sr(exp) ?
    predict(CostModel.GenerateGreens, to_cost_point(exp, GLOST_CPUS_PER_TASK), coeffs).time_s :
    Float64(first(fallback_resources(GenerateGreensJob, exp, cluster)))

"""
    glost_walltime_s(experiments, cluster, coeffs) -> Int

`--time` for the farm: the whole sweep's predicted greens time spread over
`GLOST_WORKERS` workers, padded and capped. See `GLOST_TIME_SAFETY` for the policy.
"""
function glost_walltime_s(experiments::AbstractVector{Experiment},
                          cluster::ClusterConfig, coeffs::Coefficients)
    total_s = sum(exp -> glost_greens_time_s(exp, cluster, coeffs), experiments; init=0.0)
    t = GLOST_TIME_SAFETY * total_s / GLOST_WORKERS + GLOST_TIME_SLACK_S
    return min(GLOST_MAX_TIME_S, max(MIN_TIME_S, ceil(Int, t)))
end

"""
    glost_task_line(cluster, exp) -> String

One line of the task file: the same command the `:sbatch` path runs, with its own log
file.

There are two differences from the heredoc form:

1. The `\\\$` escapes come off. GLOST hands each line to a shell inside the job, where
   `SLURM_CPUS_PER_TASK` is set and has to be expanded, and nothing re-escapes it on
   the way there. (The `\\(` escapes around the tuple arguments stay: that same shell
   would otherwise read the parentheses as a subshell, as in the heredoc.)
2. Output is redirected per task. There is one Slurm log for the whole farm, so 333
   tasks interleaving into it would be unreadable. Each task appends to the log file
   the `:sbatch` path would have given it, with `>>` rather than `>` so that a
   resubmitted task does not erase the output of the run that died.
"""
function glost_task_line(cluster::ClusterConfig, exp::Experiment)
    smr = to_smr_system(exp)
    cmd = job_command(GenerateGreensJob, cluster, smr, to_rsvd_params(exp))
    cmd = replace(cmd, "\\\$" => "\$")
    log = "$(cluster.project_dir)/$(PROJECT_NAME)/logs/greens_$(experiment_name(smr)).out"
    return "$(cmd) >> $(log) 2>&1"
end

"""
    glost_tasks_file(cluster, experiments) -> String

The whole task file: one line per experiment, with no header and no comments. GLOST
counts lines as tasks, `glost_filter` reads them back positionally, and the pre-step
below takes line 1 literally.
"""
glost_tasks_file(cluster::ClusterConfig, experiments::AbstractVector{Experiment}) =
    join((glost_task_line(cluster, exp) for exp in experiments), "\n") * "\n"

"""
    glost_sbatch_block(cluster, experiments, walltime_s) -> String

The single farmed Green-function job, written in the same
`g0_job=\$(sbatch ... <<EOF ... EOF)` shape as the per-experiment jobs so that the
RSVD jobs' `--dependency=afterok:\${g0_job}` picks it up unchanged. With `:glost`
there is one greens job id instead of 333, under the same `job_var_name`, so emitting
this block in place of the per-experiment ones is all the dependency rewiring there
is.
"""
function glost_sbatch_block(cluster::ClusterConfig,
                            experiments::AbstractVector{Experiment}, walltime_s::Int)
    var_name = job_var_name(GenerateGreensJob)
    tasks = "jobs/$(glost_tasks_filename())"
    logs_dir = "$(cluster.project_dir)/$(PROJECT_NAME)/logs"
    account = cluster.name in CC_RRG_CLUSTERS ? CC_RRG_NAME : CC_DEFAULT_GROUP_NAME
    #=
    Everything below that must survive to the *job* script has its `$` escaped
    (`\\\$` in this source): the heredoc delimiter is unquoted, so the submitting
    shell expands anything left bare, and none of these variables exist there.
    =#
    return """
# ---------------------------------------------------------------------------
# Green functions, farmed through GLOST: one queue slot for all $(length(experiments)) tasks.
#
# $(GLOST_NTASKS) MPI ranks = 1 GLOST manager (rank 0 runs no task when size > 1)
# + $(GLOST_WORKERS) workers x $(GLOST_CPUS_PER_TASK) threads. Tasks are processes, so each one
# threads freely; "serial" only means one task per rank.
#
# --time=$(slurm_time_string(walltime_s)) = $(GLOST_TIME_SAFETY) x (predicted greens time for the whole
# sweep) / $(GLOST_WORKERS) workers + $(round(Int, GLOST_TIME_SLACK_S / 60)) min, capped at $(slurm_time_string(GLOST_MAX_TIME_S)).
#
# --signal=B:USR1@$(GLOST_DRAIN_LEAD_S): GLOST drains on SIGUSR1. It lets the tasks in flight
# finish and stops handing out new ones, instead of every worker being killed
# mid-task at the walltime edge. NOTE the `B:` prefix delivers the signal to the
# batch shell only. Check in the first smoke test on narval whether it reaches
# the `srun` step, and so glost_launch. If it does not, drop the `B:` so that
# Slurm signals the step's tasks directly.
#
# To resume after a walltime kill or a partial failure: GLOST logs a per-task exit
# code, and glost_filter turns that log plus the original task file into the list
# of tasks that did not finish. Roughly
#     glost_filter -H $(logs_dir)/greens_glost_<jobid>.out $(tasks) > jobs/greens_remaining.txt
# but the flags are version-dependent and unverified here, so check
# `glost_filter -h` on narval. Whatever the flags turn out to be, use glost_filter
# to extract the unfinished tasks and resubmit this same block pointed at the
# remaining list.
# ---------------------------------------------------------------------------
$var_name=\$(sbatch \\
    --job-name=$(PROJECT_NAME)_greens_glost \\
    --output=$(logs_dir)/greens_glost_%j.out \\
    --account=$(account) \\
    --time=$(slurm_time_string(walltime_s)) \\
    --chdir=$(cluster.code_dir) \\
    --nodes=1 \\
    --ntasks=$(GLOST_NTASKS) \\
    --cpus-per-task=$(GLOST_CPUS_PER_TASK) \\
    --mem=$(GLOST_MEM_GB)G \\
    --signal=B:USR1@$(GLOST_DRAIN_LEAD_S) \\
    --export=ALL \\
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5
# glost_launch is an MPI program, so it needs a compiler + MPI loaded first.
# THIS MODULE SET IS NOT VERIFIED ON NARVAL. VERIFY with: module spider glost
# (and check which openmpi it wants under StdEnv/2023) before submitting. A
# wrong module line here fails all $(length(experiments)) tasks at once.
module load gcc openmpi glost

# Every task runs julia -t \\\$SLURM_CPUS_PER_TASK. srun sets it for the farm
# ranks; pin it here too so the serial pre-step below gets the same thread count.
export SLURM_CPUS_PER_TASK=\\\${SLURM_CPUS_PER_TASK:-$(GLOST_CPUS_PER_TASK)}

tasks="$(tasks)"
work="\\\${SLURM_TMPDIR:-/tmp}"

#=== shared-preload pre-step =================================================
# The self Green function is named after geometry alone (self/<cells>_<scale>),
# so the receiver self block is identical for every separation in the sweep, and
# $(GLOST_WORKERS) workers starting at once would all check-then-build it. Build it once,
# serially, by running task line 1 here and farming only lines 2..N.
#
# bench/generate_single_greens.jl cannot do this job as it stands. It calls
# load_greens_function (the exported name is load_green_function, so it throws
# immediately), it passes save_to_disk=false with an empty preload dir, so even
# fixed it would leave no file behind, and it hardcodes cubic (n,n,n) volumes at
# an isotropic 1//32 scale, so it cannot express the anisotropic sweeps
# (scale < 0 meaning (1//32,|s|,|s|)) or non-cubic senders. Running the first
# task line instead builds the shared block through the same code path the
# farmed tasks use, and gets that separation's own blocks done at the same time,
# so no work is duplicated.
head -n 1 "\\\$tasks" > "\\\$work/greens_preload.sh"
tail -n +2 "\\\$tasks" > "\\\$work/greens_farm.txt"
echo "[glost] pre-step: building the shared self blocks via task 1"
bash "\\\$work/greens_preload.sh" || { echo "[glost] pre-step FAILED; refusing to farm into a race"; exit 1; }
echo "[glost] pre-step done"
#============================================================================

nfarm=\\\$(wc -l < "\\\$work/greens_farm.txt")
echo "[glost] farming \\\$nfarm tasks over $(GLOST_WORKERS) workers"
srun glost_launch "\\\$work/greens_farm.txt"
EOF
)
$var_name=\${$var_name##* }
sleep 0.05

"""
end

"""
    job_launcher_script(jobs, cluster, experiments) -> (script, plan, glost)

Build the launcher. Returns the script text, the per-experiment resource plan so the
caller can print a summary, and a named tuple describing the farm, including the task
file the caller has to write next to the launcher. That third value is `nothing`
unless `GREENS_LAUNCHER == :glost` and the Green-function job is being run, and on
that default path nothing else about the output changes either.

`gpu_index` is forwarded to `job_command`: every GPU job in this launcher pins
that CUDA device. See `NUM_GPUS` at the bottom of the file.
"""
function job_launcher_script(jobs::AbstractVector{JobType}, cluster::ClusterConfig,
                             experiments::AbstractVector{Experiment};
                             gpu_index::Int=0)
    validate_greens_launcher(cluster)
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

    #=
    With :glost the Green-function stage is one job, emitted here, ahead of
    everything that depends on it. It binds the same `g0_job` shell variable the
    per-experiment greens jobs would have, so the RSVD jobs' `--dependency=afterok:${g0_job}`
    points at the farm and every RSVD job waits for the whole farm rather than for
    its own greens job. That is coarser, since the slowest task gates the fastest
    RSVD, but greens is the cheap stage. RSVD -> bounds chaining is unaffected.
    =#
    glost = nothing
    glost_active = GREENS_LAUNCHER == :glost && GenerateGreensJob in jobs
    if glost_active
        walltime_s = glost_walltime_s(experiments, cluster, coeffs)
        script *= glost_sbatch_block(cluster, experiments, walltime_s)
        glost = (tasks_filename=glost_tasks_filename(),
                 tasks=glost_tasks_file(cluster, experiments),
                 num_tasks=length(experiments),
                 farmed_tasks=max(0, length(experiments) - 1),
                 nodes=1, ntasks=GLOST_NTASKS, workers=GLOST_WORKERS,
                 cpus_per_task=GLOST_CPUS_PER_TASK, mem_GB=GLOST_MEM_GB,
                 time_s=walltime_s)
    end

    plan = Tuple{Experiment,Dict{JobType,Resources}}[]
    for exp in experiments
        smr = to_smr_system(exp)
        params = to_rsvd_params(exp)

        resources = Dict{JobType,Resources}()
        for job in ORDERED_JOBS
            job in jobs || continue
            glost_active && job == GenerateGreensJob && continue
            cores = choose_cores(job, exp, cluster, coeffs)
            resources[job] = resources_for(job, exp, cluster, coeffs, cores)
        end
        push!(plan, (exp, resources))

        for job in ORDERED_JOBS
            job in jobs || continue
            glost_active && job == GenerateGreensJob && continue
            res = resources[job]
            if res.over_vram
                @warn "$(experiment_name(smr)) $(string(job)) cannot be squeezed below about $(res.vram_floor_GB) GB of VRAM on the $(mode_label(res.mode)) path, more than the $(cluster.max_vram_GB) GB on $(cluster.name). Submitting anyway on a whole GPU, but expect an out-of-memory failure. Reduce the rank or move it to a bigger card."
            end

            header, footer = "", ""
            if cluster.has_slurm
                dependency = previous_job(job) in jobs ? previous_job(job) : nothing
                header, footer = slurm_header_footer(job, cluster, smr, res, dependency)
            end
            script *= header
            script *= job_command(job, cluster, smr, params; gpu_index=gpu_index) * "\n"
            script *= footer
        end
        script *= "\n"
    end
    return script, plan, glost
end

"""
    DISK_WARN_TB

Scratch usage above which `print_plan` complains. The Alliance's default `/scratch`
quota is 20 TB per user, and a sweep that fills it does not fail early: it fails at
the save, after the job has done every matvec. 15 TB leaves room for the
Green-function preloads and for whatever the previous sweep has not had cleaned out
yet.
"""
const DISK_WARN_TB = 15.0

"""
    sweep_disk_bytes(plan) -> (bytes, num_counted)

Scratch the sweep's eigenvector saves will occupy: `N_u * m * 16` bytes per
experiment, where `m = ceil(NUM_POS_FRACTION * k)` is how many columns the
positives-only save writes. That save keeps the positive-Γ prefix only, in
ComplexF64 (the decision recorded in `FUNICULAR_PLAN.md` B5, which drops a
4 λ k=4000 sweep from ~17 TB to ~10 TB).

Both factors come from the cost model (`universe_length`, `effective_num_pos`)
rather than being re-derived here, so this number and the `bytes_written` the RSVD
prediction is built on cannot drift apart. `to_cost_point` is called with one thread
because neither factor depends on the thread count.

Mediator experiments are skipped, since they have no `SRPoint` and the positives-only
save is not what that pipeline writes. The count of what was included comes back with
the total so the caller can say so.
"""
function sweep_disk_bytes(plan)
    total = 0.0
    counted = 0
    for (exp, _) in plan
        is_sr(exp) || continue
        pt = to_cost_point(exp, 1)
        total += CostModel.universe_length(pt) * CostModel.effective_num_pos(pt) * 16
        counted += 1
    end
    return total, counted
end

"""
    print_plan(plan, jobs, cluster; glost=nothing)

Summary table of what was requested and why. Worth reading before submitting: it
is where a rank that does not fit in VRAM, or a bounds job that has quietly become
the expensive one, becomes obvious.

`glost` is the named tuple `job_launcher_script` returns for a farmed Green-function
stage. When it is given there are no greens rows, since there is one job and not one
per separation, and the farm gets a summary block of its own instead.
"""
function print_plan(plan, jobs::AbstractVector{JobType}, cluster::ClusterConfig;
                    glost=nothing)
    println(stderr)
    println(stderr, "Resource plan for $(length(plan)) experiments on $(cluster.name):")
    println(stderr, "  ", rpad("experiment", 30), rpad("job", 8), rpad("time", 11),
            rpad("cores", 6), rpad("host", 8), rpad("mode", 7), "gpu")
    totals = Dict{JobType,Float64}()
    core_seconds = 0.0
    for (exp, resources) in plan
        label = is_sr(exp) ?
                "$(join(exp.sender_cells, "x")) sep=$(rational2string(exp.separation)) k=$(exp.rank)" :
                "$(join(exp.sender_cells, "x")) +mediator k=$(exp.rank)"
        first_row = true
        for job in ORDERED_JOBS
            job in jobs || continue
            haskey(resources, job) || continue # farmed stages have no per-experiment row
            res = resources[job]
            totals[job] = get(totals, job, 0.0) + res.time_s
            core_seconds += res.time_s * res.cores
            short = job == GenerateGreensJob ? "greens" :
                    job == GenerateRSVDJob ? "rsvd" : "bounds"
            gpu = uses_gpu(job) ? "$(res.gpu_name) ($(res.vram_GB) GB)" : "-"
            println(stderr, "  ", rpad(first_row ? label : "", 30), rpad(short, 8),
                    rpad(seconds2string(res.time_s), 11), rpad(res.cores, 6),
                    rpad("$(res.host_GB)G", 8), rpad(mode_label(res.mode), 7), gpu)
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

    #=
    Nothing in the request mentions disk, and running out of it is the worst of the
    failures: over quota, the RSVD job dies writing its output, after having already
    spent every matvec it was going to spend. So it goes in the same summary as the
    memory and the walltime.
    =#
    disk_bytes, disk_counted = sweep_disk_bytes(plan)
    if disk_counted > 0
        disk_TB = disk_bytes / 1e12
        println(stderr)
        @printf(stderr, "  estimated scratch for eigenvector saves %12.2f TB\n", disk_TB)
        @printf(stderr, "    = %d experiments x N_u x ceil(%.2f x k) columns x 16 B (positive-Γ prefix only, ComplexF64)\n",
                disk_counted, NUM_POS_FRACTION)
        println(stderr, "    eigenvalues, singular values and metadata are negligible beside this.")
        if disk_TB > DISK_WARN_TB
            println(stderr)
            @printf(stderr, "  !!! %.1f TB of scratch is over the %.0f TB line: check your scratch quota\n",
                    disk_TB, DISK_WARN_TB)
            println(stderr, "  !!! before submitting, with:  diskusage_report")
            println(stderr, "  !!! Over quota the sweep fails at the save, after doing all of the work.")
        end
    end

    #=
    A host request that hit `cluster.max_host_GB` is the one place where the number
    in the table is not the number the model asked for, and on the panel path the
    gap can be tens of gigabytes. That is by design (see `Resources.host_uncapped_GB`):
    asking past the per-GPU bundle is billed as several GPUs, and Funicular's plan
    spills panels to the node-local NVMe under $SLURM_TMPDIR instead of failing. But
    a cap that says nothing reads exactly like a request that fitted, so say it.
    =#
    capped = [(exp, job, res) for (exp, resources) in plan for job in ORDERED_JOBS
              if haskey(resources, job)
              for res in (resources[job],)
              if res.host_uncapped_GB > cluster.max_host_GB]
    if !isempty(capped)
        println(stderr)
        println(stderr, "  $(length(capped)) job(s) want more host memory than $(cluster.name)'s $(cluster.max_host_GB) GB bundle;")
        println(stderr, "  the request is capped there and the run leans on the NVMe spill tier:")
        shown = Set{Tuple{Symbol,Int,Int}}()
        for (exp, job, res) in capped
            key = (res.mode, res.host_uncapped_GB, exp.rank)
            key in shown && continue
            push!(shown, key)
            short = job == GenerateGreensJob ? "greens" :
                    job == GenerateRSVDJob ? "rsvd" : "bounds"
            println(stderr, "    $(join(exp.sender_cells, "x")) k=$(exp.rank) $(short) [$(mode_label(res.mode))]: ",
                    "wants $(res.host_uncapped_GB) G, asking for $(res.host_GB) G")
        end
        println(stderr, "  Funicular's ResidencyPlan gets scratch_dir = \$SLURM_TMPDIR (src/common.jl,")
        println(stderr, "  residency_plan), so panels beyond the host budget go to node-local disk rather")
        println(stderr, "  than out of memory. This is expected at the top of the sweep, not a mis-sizing.")
        println(stderr, "  It does cost bus time, and it needs SLURM_TMPDIR to be real: if the node has no")
        println(stderr, "  local NVMe the plan has nowhere to spill and the job will die at the sketch.")
    end

    if !isnothing(glost)
        glost_core_hours = glost.time_s * glost.ntasks * glost.cpus_per_task / 3600
        println(stderr)
        println(stderr, "  Green functions: 1 GLOST job (not $(glost.num_tasks) jobs), $(glost.nodes) node, $(glost.ntasks) ranks")
        println(stderr, "    workers            $(glost.workers) x $(glost.cpus_per_task) threads (rank 0 is the manager and runs no task)")
        println(stderr, "    tasks              $(glost.num_tasks) total; 1 run serially as the shared-preload pre-step, $(glost.farmed_tasks) farmed")
        println(stderr, "    walltime           $(slurm_time_string(glost.time_s))  (mem $(glost.mem_GB)G)")
        @printf(stderr, "    core-hours         %.1f (not counted in the total above)\n", glost_core_hours)
        println(stderr, "    queue slots        1 instead of $(glost.num_tasks); every RSVD job depends on this one job id")
    end

    coeffs = coefficients_for(cluster.name)
    if !coeffs.calibrated
        println(stderr)
        println(stderr, "  NOTE: $(cluster.name) has no measured calibration. These requests come from")
        println(stderr, "  bench/coeffs_$(cluster.name).jl, derived from another cluster and derated to")
        println(stderr, "  over-estimate. Measure it with:")
        println(stderr, "      julia bench/plan.jl --cluster $(cluster.name) --tier quick")
    end

    # A MIG slice name the cluster does not define is a hard sbatch rejection, and
    # only about half of fir's GPU nodes carry slices at all, so say which names
    # this plan is about to use.
    slices = unique(res.gpu_name for (_, resources) in plan for (_, res) in resources
                    if occursin("g.", res.gpu_name))
    if !isempty(slices)
        println(stderr)
        println(stderr, "  This plan requests MIG slices: $(join(sort(slices), ", ")).")
        println(stderr, "  If sbatch refuses them, check the names with:  sinfo -o \"%G\" | sort -u")
    end
    return nothing
end

load_coefficients!(joinpath(@__DIR__, "bench"))

cluster = ClusterConfig("molering")
# cluster = ClusterConfig("fir")
# cluster = ClusterConfig("narval")
# cluster = ClusterConfig("nibi")
# cluster = ClusterConfig("rorqual")

"""
    NUM_GPUS

How many launcher scripts to split the sweep across, one CUDA device per script
(`--gpu 0`, `--gpu 1`, ...). Molering has two A6000s and no scheduler, so two
scripts running side by side is how both cards get used. Keep this at 1 on the
SLURM clusters: there the scheduler hands every job its own card and `--gpu 0`
is always right inside an allocation (the loop below enforces this).

The split is round-robin over the experiment list, so whatever run order is set
below survives within each script, and both scripts cover the whole separation
range rather than one taking the cheap half. The scripts share every directory:
outputs are keyed by experiment name, the ext Green blocks are keyed by
separation (disjoint between the scripts), and the one shared self Green block
is written through `serialize_atomic`, so concurrent scripts at worst build it
redundantly once each instead of corrupting it.
"""
const NUM_GPUS = 2

valid_clusters = ["molering", "fir", "narval", "nibi", "rorqual"]
for cluster_name in valid_clusters
    if occursin(cluster_name, PROJECT_NAME) && cluster_name != cluster.name
        @warn "Project name contains '$(cluster_name)' but the cluster is set to '$(cluster.name)'. Did you mean to run on $(cluster_name)?"
        println(stderr, "Press enter to continue anyway, or Ctrl-C to abort...")
        readline()
    end
end


### Metasurface: lambda/4 cubes at lambda/32 cells, swept in separation
# separations = unique(round.(Int, logrange(1, 10000 * 32, 415))) .// 32 # 415 points gives us 333 actual points (times 3 = 999 < 1000 which is the number of points we can submit to the queue at once (× 3 because 3 jobs per experiment))

"""
    unique_log_separations(count) -> Vector{Rational{Int}}

`count` log-spaced separations with the production grid's endpoints, 1/32 λ to
10000 λ. Rounding to integer cells collapses log-spaced points at the small end
(415 requested points give 333 on the production grid), so the argument to
`logrange` is searched for until the unique count comes out exactly right.
"""
function unique_log_separations(count::Int)
    for n in count:20*count
        seps = unique(round.(Int, logrange(1, 10000 * 32, n))) .// 32
        length(seps) == count && return seps
    end
    error("no logrange length gives exactly $count unique separations")
end

# Molering two-GPU 1/4 λ sweep: 100 points total, 50 per card.
separations = unique_log_separations(NUM_GPUS * 50)

# Run the 1 λ to 10 λ decade first, then everything else, both halves in
# increasing separation. The round-robin split preserves this order per script.
separations = vcat(filter(s -> 1 <= s <= 10, separations),
                   filter(s -> !(1 <= s <= 10), separations))

# chi = 13.6 + 0.05im # "Silicon like"
chi = 4.250 + 0.0342557im # Germanium with ζ = 1000
# chi = 4.250 + 0.06854306950164653im # Germanium with ζ = 500

# The Ge ζ = 1000 production sweeps: rank 4000 at every size except 1/4 λ, whose
# universe only has N_u = 3072 columns to give. Uncomment one, match PROJECT_NAME
# at the top of the file, and run. Each of the five writes its own launcher.

# 1/4
experiments = sr_sweep(cells=(8, 8, 8), separations=separations, scale=1 // 32, chi=chi, rank=3072, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(8, 8, 8), separations=separations, scale=1 // 32, chi=chi, rank=1350, oversamples=50, power_iters=14)

# 1/2
# experiments = sr_sweep(cells=(16, 16, 16), separations=separations, scale=1 // 32, chi=chi, rank=4000, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(16, 16, 16), separations=separations, scale=1 // 32, chi=chi, rank=1350, oversamples=50, power_iters=14)

# 1
# experiments = sr_sweep(cells=(32, 32, 32), separations=separations, scale=1 // 32, chi=chi, rank=4000, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(32, 32, 32), separations=separations, scale=1 // 32, chi=chi, rank=1350, oversamples=50, power_iters=14)

# 2
# experiments = sr_sweep(cells=(64, 32, 32), separations=separations, scale=-1 // 16, chi=chi, rank=4000, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(64, 32, 32), separations=separations, scale=-1 // 16, chi=chi, rank=1350, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(64, 32, 32), separations=separations, scale=-1 // 16, chi=chi, rank=800, oversamples=50, power_iters=14)

# 4
# experiments = sr_sweep(cells=(128, 32, 32), separations=separations, scale=-1 // 8, chi=chi, rank=4000, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(128, 32, 32), separations=separations, scale=-1 // 8, chi=chi, rank=800, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(128, 32, 32), separations=separations, scale=-1 // 8, chi=chi, rank=400, oversamples=50, power_iters=14)

# silica 2x2
# experiments = sr_sweep(cells=(64, 32, 32), separations=separations, scale=-1 // 16, chi=-2.30466271 + 1.478912im, rank=1350, oversamples=50, power_iters=14)
# experiments = sr_sweep(cells=(64, 32, 32), separations=separations, scale=-1 // 16, chi=-2.30466271 + 1.478912im, rank=800, oversamples=50, power_iters=14)

# The greens stage is geometry-keyed and chi-independent, so the preload files the
# earlier sweeps left behind make these tasks skip almost immediately. It stays in
# the list anyway: on cold scratch it is the only thing that builds them.
jobs_to_run  = [GenerateGreensJob, GenerateRSVDJob, ComputeBoundsJob]
# jobs_to_run  = [GenerateRSVDJob, ComputeBoundsJob]

NUM_GPUS > 1 && cluster.has_slurm &&
    error("NUM_GPUS > 1 is for unscheduled machines: under SLURM every job gets " *
          "its own allocation and --gpu 0 is always the right device. Set NUM_GPUS = 1.")

# One launcher per GPU, round-robin over the experiment list. NUM_GPUS == 1
# reproduces the old single-launcher flow, same filename and all.
mkpath(joinpath(@__DIR__, "jobs"))
script_paths = String[]
for gpu_index in 0:(NUM_GPUS - 1)
    gpu_experiments = experiments[(gpu_index + 1):NUM_GPUS:end]
    command, plan, glost_summary = job_launcher_script(jobs_to_run, cluster,
                                                       gpu_experiments;
                                                       gpu_index=gpu_index)
    # print(command)
    print_plan(plan, jobs_to_run, cluster; glost=glost_summary)

    suffix = NUM_GPUS == 1 ? "" : "_gpu$(gpu_index)"
    script_path = joinpath(@__DIR__, "jobs", "launch_$(PROJECT_NAME)$(suffix).sh")
    open(script_path, "w") do f
        write(f, command)
    end
    push!(script_paths, script_path)
    println(stderr, "\nJob launcher script written to $(script_path)")
    if !isnothing(glost_summary)
        # The launcher submits the farm, but the task file is what the farm reads, so
        # it has to travel with it. The sbatch block looks for it at
        # <code_dir>jobs/<name>.
        tasks_path = joinpath(@__DIR__, "jobs", glost_summary.tasks_filename)
        open(tasks_path, "w") do f
            write(f, glost_summary.tasks)
        end
        println(stderr, "GLOST task file written to $(tasks_path) ($(glost_summary.num_tasks) tasks)")
        println(stderr, "Copy BOTH over (the launcher is useless without the task file):")
        println(stderr, "  scp \"$(script_path)\" \"$(tasks_path)\" \"$(CC_UNAME)@$(cluster.name).alliancecan.ca:$(cluster.code_dir)jobs/\"")
    end
end

quoted_paths = join(("\"$p\"" for p in script_paths), " ")
if cluster.has_slurm
    println(stderr, "Copy it over:  scp $(quoted_paths) \"$(CC_UNAME)@$(cluster.name).alliancecan.ca:$(cluster.code_dir)jobs/\"")
    println(stderr, "Then run it:   ssh $(CC_UNAME)@$(cluster.name).alliancecan.ca 'cd $(cluster.code_dir) && bash jobs/launch_$(PROJECT_NAME).sh'")
else
    println(stderr, "Copy over:  scp $(quoted_paths) \"$(MOLERING_UNAME)@molering:$(cluster.code_dir)jobs/\"")
    for p in script_paths
        name = basename(p)
        log = replace(name, r"\.sh$" => "") * ".log"
        println(stderr, "Then run:   ssh $(MOLERING_UNAME)@molering 'cd $(cluster.code_dir) && nohup bash jobs/$(name) > jobs/$(log) 2>&1 &'")
    end
end
