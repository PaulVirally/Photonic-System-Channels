using PhotonicSystemChannels
using GilaElectromagnetics
using Dates

# CreateJobs.jl
# -------------
# Change the project name below and your compute canada / molering settings
# first, then go to the bottom of the file to set up your desired parameter
# combinations. Finally, run this script with
# `julia create_jobs.jl > job_launcher.sh` and copy it over to the cluster of
# your choice to run it with `bash job_launcher.sh`.

const PROJECT_NAME = "heat-transfer_sep_4x4x0p5"

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
const CC_RRG_CLUSTERS = ["narval"]
const CC_CODE_DIR = "/home/$(CC_UNAME)/Photonic-System-Channels/"
const CC_PRELOAD_DIR = "/home/$(CC_UNAME)/scratch/preload/"
const CC_SCRATCH_DIR = "/home/$(CC_UNAME)/scratch/Photonic-System-Channels/"

const MEMORY_PADDING = 1.2 # Factor to pad memory estimates by to avoid OOM errors
const TIME_PADDING = 1.5 # Factor to pad time estimates by to avoid time limit exceeded errors

@enum JobType begin
    GenerateGreens
    GenerateRSVD
    ComputeBounds
end
const ORDERED_JOBS = [GenerateGreens, GenerateRSVD, ComputeBounds]

function main_file(job::JobType)
    if job == GenerateGreens
        return "generate_greens.jl"
    elseif job == GenerateRSVD
        return "generate_rsvd.jl"
    elseif job == ComputeBounds
        return "compute_bounds.jl"
    end
    error("Unknown job type: $job")
end

function job_var_name(job::JobType)
    if job == GenerateGreens
        return "g0_job"
    elseif job == GenerateRSVD
        return "rsvd_job"
    elseif job == ComputeBounds
        return "bounds_job"
    end
    error("Unknown job type: $job")
end

function previous_job(job::JobType)
    if job == GenerateGreens
        return nothing
    elseif job == GenerateRSVD
        return GenerateGreens
    elseif job == ComputeBounds
        return GenerateRSVD
    end
    error("Unknown job type: $job")
end

const CPU_TURBO_GHZ_REFERENCE = 4.5 # Benchmarks for CPU workloads were done on an AMD Ryzen Threadripper Pro 5995WX, which turbos to 4.5 GHz
const GPU_FP64_TFLOPS_REFERENCE = 0.6 # Benchmarks for (double precision) GPU workloads were done on an NVIDIA A6000, which (I'm estimating here) can do roughly 0.6 TFLOPS

struct ClusterConfig
    name::String
    has_slurm::Bool
    cpu_turbo_GHz::Float64
    gpu_fp64_TFLOPS::Float64
    max_vram_GB::Int64
    preload_dir::String
    project_dir::String
    scratch_dir::String
end

num_threads(config::ClusterConfig) = config.has_slurm ? "\$SLURM_CPUS_PER_TASK" : "auto"

function cpu2turbo(cpu::String)
    if cpu == "Epyc 7413" return 4.6 end
    return error("Unknown CPU: $cpu")
end

function ClusterConfig(server::AbstractString)
    if server == "molering"
        return ClusterConfig(server,
                             false, # Does not run slurm
                             4.5, # AMD Ryzen Threadripper Pro 5995WX turbo frequency
                             0.6, # NVIDIA A6000 FP64 TFLOPS
                             48, # Max VRAM in GB
                             MOLERING_PRELOAD_DIR,
                             MOLERING_PROJECT_DIR,
                             MOLERING_SCRATCH_DIR)
    elseif server == "narval"
        return ClusterConfig(server,
                             true, # Runs slurm
                             3.6, # AMD EPYC 7413 turbo frequency
                             9.7, # NVIDIA A100SXM4 40GB FP64 TFLOPS
                             40, # Max VRAM in GB
                             CC_PRELOAD_DIR,
                             server in CC_RRG_CLUSTERS ? "/home/$(CC_UNAME)/projects/$(CC_RRG_NAME)/$(CC_UNAME)/Photonic-System-Channels/" : "/home/$(CC_UNAME)/projects/$(CC_DEFAULT_GROUP_NAME)/$(CC_UNAME)/Photonic-System-Channels/",
                             CC_SCRATCH_DIR)
    elseif server == "fir"
        return ClusterConfig(server,
                             true, # Runs slurm
                             3.8, # AMD EPYC 9454 turbo frequency
                             34.0, # NVIDIA A40 FP64 TFLOPS
                             80, # Max VRAM in GB
                             CC_PRELOAD_DIR,
                             server in CC_RRG_CLUSTERS ? "/home/$(CC_UNAME)/projects/$(CC_RRG_NAME)/$(CC_UNAME)/Photonic-System-Channels/" : "/home/$(CC_UNAME)/projects/$(CC_DEFAULT_GROUP_NAME)/$(CC_UNAME)/Photonic-System-Channels/",
                             CC_SCRATCH_DIR)
    end
    error("Unknown server: $server")
end

function gpu_string(cluster::ClusterConfig, memory_GB::Int)
    if memory_GB > cluster.max_vram_GB
        error("Requested memory $(memory_GB) GB exceeds max VRAM $(cluster.max_vram_GB) GB on cluster $(cluster.name)")
    end
    if cluster.name == "narval"
        if memory_GB < 5
            return "a100_1g.5gb"
        elseif memory_GB < 10
            return "a100_2g.10gb"
        elseif memory_GB < 20
            return "a100_3g.20gb"
        end
        return "a100"
    elseif cluster.name == "fir"
        if memory_GB < 10
            return "nvidia_h100_80gb_hbm3_1g.10gb"
        elseif memory_GB < 20
            return "nvidia_h100_80gb_hbm3_2g.20gb"
        elseif memory_GB < 40
            return "nvidia_h100_80gb_hbm3_3g.40gb"
        end
        return "h100"
    elseif cluster.name == "molering"
        return "a6000"
    end
    error("GPU string not implemented for cluster: $(cluster.name)")
end

function gpu_compute_fraction(gpu_str::AbstractString)
    if gpu_str == "a100_1g.5gb"
        return 1/8
    elseif gpu_str == "a100_2g.10gb"
        return 2/8
    elseif gpu_str == "a100_3g.20gb"
        return 3/8
    elseif gpu_str == "a100_4g.20gb"
        return 4/8
    elseif gpu_str == "a100"
        return 1.0
    elseif gpu_str == "nvidia_h100_80gb_hbm3_1g.10gb"
        return 1/8
    elseif gpu_str == "nvidia_h100_80gb_hbm3_2g.20gb"
        return 2/8
    elseif gpu_str == "nvidia_h100_80gb_hbm3_3g.40gb"
        return 3/8
    elseif gpu_str == "h100"
        return 1.0
    elseif gpu_str == "a6000"
        return 1.0
    end
    error("GPU compute fraction not implemented for GPU string: $gpu_str")
end

rational2string(r::Rational, separator="//") = "$(numerator(r))$separator$(denominator(r))"

function experiment_name(smr::SMRSystem)
    s = sender(smr)
    m = mediator(smr)
    r = receiver(smr)
    if isnothing(m)
        sep = rs_separation(smr)[1] # Assume only x-separation for heat transfer
        return "($(join(s.cel, ",")))_$(rational2string(sep, "ss"))_($(join(r.cel, ",")))@($(join(rational2string.(s.scl, "ss"), ",")))"
    end
    sm_sep = sm_separation(smr)
    sm_sep_string = "($(rational2string(sm_sep[1])), $(rational2string(sm_sep[2])), $(rational2string(sm_sep[3])))"
    mr_sep = mr_separation(smr)
    mr_sep_string = "($(rational2string(mr_sep[1])), $(rational2string(mr_sep[2])), $(rational2string(mr_sep[3])))"
    return "($(join(s.cel, ",")))_$(sm_sep_string)_($(join(m.cel, ",")))_$(mr_sep_string)_($(join(r.cel, ",")))@($(join(rational2string.(s.scl, "ss"), ",")))"
end

function heat_transfer_args(smr::SMRSystem, params::RSVDParams)
    s = sender(smr)
    r = receiver(smr)
    sender_string = "($(join(s.cel, ",")))"
    receiver_string = "($(join(r.cel, ",")))"
    sep = rs_separation(smr)[1] # Assume only x-separation for heat transfer
    rs_sep_string = "($(rational2string(sep)),0//1,0//1)"
    scale_string = "($(join(rational2string.(s.scl), ",")))"
    chi_string = "$(real(χ(smr)))+$(imag(χ(smr)))im"
    name_string = experiment_name(smr)
    design_string = "rs" # Design the entire region
    return "--sender $(sender_string) --receiver $(receiver_string) --rs-sep $(rs_sep_string) --scale $(scale_string) --chi $(chi_string) --design $(design_string) --gpu true --components $(params.rank) --oversamples $(params.oversamples) --power-iterations $(params.power_iter) --name $(name_string)"
end

function smr_args(smr::SMRSystem, params::RSVDParams)
    s = sender(smr)
    m = mediator(smr)
    r = receiver(smr)
    sender_string = "($(join(s.cel, ",")))"
    mediator_string = "($(join(m.cel, ",")))"
    receiver_string = "($(join(r.cel, ",")))"
    sm_sep = sm_separation(smr)
    sm_sep_string = "($(rational2string(sm_sep[1])), $(rational2string(sm_sep[2])), $(rational2string(sm_sep[3])))"
    mr_sep = mr_separation(smr)
    mr_sep_string = "($(rational2string(mr_sep[1])), $(rational2string(mr_sep[2])), $(rational2string(mr_sep[3])))"
    scale_string = "($(join(rational2string.(s.scl), ",")))"
    chi_string = "$(real(χ(smr)))+$(imag(χ(smr)))im"
    name_string = experiment_name(smr)
    design_string = "m" # Design the mediator region
    return "--sender $(sender_string) --mediator $(mediator_string) --receiver $(receiver_string) --sm-sep $(sm_sep_string) --mr-sep $(mr_sep_string) --scale $(scale_string) --chi $(chi_string) --design $(design_string) --gpu true --components $(params.rank) --oversamples $(params.oversamples) --power-iterations $(params.power_iter) --name $(name_string)"
end

function args(smr::SMRSystem, params::RSVDParams)
    if isnothing(mediator(smr))
        return heat_transfer_args(smr, params)
    end
    return smr_args(smr, params)
end

# Compute the biggest volume (in voxels cubed) among the source and target regions in the SMR system
function biggest_volume_voxels_cubed(smr::SMRSystem)
    pairs = volume_pairs(smr)
    volume(pair) = 3*max(prod(pair.source.cel), prod(pair.target.cel)) # 3x for polarizations
    return maximum(volume.(pairs))
end

function time_s(job_type::JobType, smr::SMRSystem, params::RSVDParams)
    max_vol = biggest_volume_voxels_cubed(smr)
    num_pairs = length(volume_pairs(smr)) # Number of source-target volume pairs
    if job_type == GenerateGreens
        g0_time_s_A6000 = TIME_PADDING * (max_vol * log2(max_vol) * 2.147823889114151e-6 + 37.68202102148491) # empirical formula
        return g0_time_s_A6000 * num_pairs
    elseif job_type == GenerateRSVD
        q = params.power_iter
        k = params.rank
        p = params.oversamples
        rsvd_time_s_A6000 = TIME_PADDING * (1.4917e-8 * (2 + 2q)*(k+p)*max_vol*log2(max_vol) + 71.3435)
        return rsvd_time_s_A6000 * num_pairs
    elseif job_type == ComputeBounds
        bounds_time_s_A6000 = 0.0 # empirical formula
        return bounds_time_s_A6000
    end
    error("Unknown job type: $job_type")
end

function memory_GB(job_type::JobType, smr::SMRSystem, params::RSVDParams)
    max_vol = biggest_volume_voxels_cubed(smr)
    if job_type == GenerateGreens
        return ceil(Int, MEMORY_PADDING * (1.722879361316178e9 + 798.4151595299923 * max_vol) * 1e-9) # empirical formula
    elseif job_type == GenerateRSVD
        k = params.rank
        p = params.oversamples
        memory_GB = ceil(Int, MEMORY_PADDING * (307.4977255*(k+p)*max_vol + 1.955629444e9) * 1e-9) # empirical formula
        return memory_GB
    elseif job_type == ComputeBounds
        memory_GB = 0.0 # empirical formula
        return memory_GB
    end
    error("Unknown job type: $job_type")
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

function Base.string(job::JobType)
    if job == GenerateGreens
        return "Greens function generation"
    elseif job == GenerateRSVD
        return "RSVD generation"
    elseif job == ComputeBounds
        return "bounds computation"
    end
    error("Unknown job type: $job")
end

function slurm_header_footer(job::JobType, cluster::ClusterConfig, smr::SMRSystem, time_s::Int, cores::Int, memory_GB::Int, dependency::Union{Nothing, JobType}=nothing)
    var_name = job_var_name(job)
    header = "$var_name=\$(sbatch \\\n"
    if !isnothing(dependency)
        header *= "    --dependency=afterok:\${$(job_var_name(dependency))} \\\n"
    end
    header *= """    --job-name=$(PROJECT_NAME)_$(experiment_name(smr)) \\
    --output=$(cluster.project_dir)/$(PROJECT_NAME)/logs/$(experiment_name(smr))_%j.out \\
    --account=$(cluster.name in CC_RRG_CLUSTERS ? CC_RRG_NAME : CC_DEFAULT_GROUP_NAME) \\
    --time=$(seconds2string(time_s)) \\
    --cpus-per-task=$cores \\
    --mem=$(memory_GB)G \\
    --chdir=$CC_CODE_DIR \\
"""
    if job in [GenerateRSVD, ComputeBounds] # Use GPU for RSVD generation and bounds computation
        header *= """    --gres=gpu:$(gpu_string(cluster, memory_GB)):1 \\\n"""
    end
    header *= """    --export=ALL \\
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.11.3 cuda/12.2
srun """
    footer = """EOF
)
$var_name=\${$var_name##* }
sleep 0.05
"""
    return header, footer
end

function job_launcher_script(jobs::AbstractVector{JobType},
        cluster::ClusterConfig,
        sender_cells::AbstractVector{NTuple{3, Int}},
        mediator_cells::AbstractVector{<:Union{Nothing, NTuple{3, Int}}},
        receiver_cells::AbstractVector{NTuple{3, Int}},
        sm_separations::AbstractVector{<:Union{Nothing, NTuple{3, Rational{Int}}}},
        mr_separations::AbstractVector{<:Union{Nothing, NTuple{3, Rational{Int}}}},
        rs_separations::AbstractVector{<:Union{Nothing, NTuple{3, Rational{Int}}}},
        scales::AbstractVector{Rational{Int}},
        chis::AbstractVector{ComplexF64},
        target_ranks::AbstractVector{Int},
        num_oversamples::AbstractVector{Int},
        num_power_iters::AbstractVector{Int})

    script = """
#!/bin/bash

# Job launcher generated on $(now()) by create_jobs.jl

echo Running job launcher for $(join(string.(jobs), ", "))
echo There $(length(sender_cells) > 1 ? "are" : "is") $(length(sender_cells)) job$(length(sender_cells) > 1 ? "s" : "") to launch
echo We are expecting to be on $(cluster.name)

# Create scratch, preload, and project directories if they don't exist
mkdir -p $(cluster.scratch_dir)/$(PROJECT_NAME)/
mkdir -p $(cluster.preload_dir)
mkdir -p $(cluster.project_dir)/$(PROJECT_NAME)/
"""
    if cluster.has_slurm
        script *= "mkdir -p $(cluster.project_dir)/$(PROJECT_NAME)/logs/ # Directory for slurm logs\n"
    end
    script *= "\n# Job submission commands follow\n\n"

    # For each parameter combination, create the corresponding submission command
    for (sender_num_cells, mediator_num_cells, receiver_num_cells, sm_separation, mr_separation, rs_separation, scale, χ, target_rank, num_oversample, num_power_iter) in zip(sender_cells, mediator_cells, receiver_cells, sm_separations, mr_separations, rs_separations, scales, chis, target_ranks, num_oversamples, num_power_iters)
        smr = SMRSystem(sender_num_cells,
            mediator_num_cells,
            receiver_num_cells,
            sm_separation,
            mr_separation,
            rs_separation,
            scale,
            χ)
        rsvd_params = RSVDParams(target_rank, num_oversample, num_power_iter)

        max_job_memory = maximum(max(4, ceil(Int, memory_GB(job_type, smr, rsvd_params))) for job_type in jobs) # Max memory across all jobs for this experiment
        if max_job_memory > cluster.max_vram_GB
            @warn "Skipping experiment $(experiment_name(smr)) because required memory $(max_job_memory) GB exceeds max VRAM $(cluster.max_vram_GB) GB on cluster $(cluster.name)"
            continue
        end

        # For each job type, add the corresponding job submission command
        for job_type in ORDERED_JOBS
            if !(job_type in jobs) continue end

            memory = max(4, ceil(Int, memory_GB(job_type, smr, rsvd_params))) # We should always request at least 4 GB
            time = max(10*60, ceil(Int, time_s(job_type, smr, rsvd_params) * gpu_compute_fraction(gpu_string(cluster, memory)))) # Adjust time based on GPU compute fraction, minimum 10 minutes
            num_cores = cluster.name == "fir" ? 6 : 4 # fir has 6-core nodes, everything else works fine with 4 cores (molering runs with all available cores)

            header, footer = "", ""
            if cluster.has_slurm
                header, footer = slurm_header_footer(job_type, cluster, smr, time, num_cores, memory, previous_job(job_type) in jobs ? previous_job(job_type) : nothing)
            end
            script *= header

            job_args = args(smr, rsvd_params)
            script *= "julia -t $(num_threads(cluster)) $(main_file(job_type)) $job_args \n"

            script *= footer
        end
        script *= "\n"
    end
    return script
end

num_experiments = 32
separations = unique(collect(round.(Int, logrange(8, 8*32, num_experiments)))) .// 32 # from 8//32 λ to 8//1 λ in log-spaced steps
num_experiments = length(separations)
@show separations

print(job_launcher_script(
    [GenerateGreens, GenerateRSVD],
    ClusterConfig("molering"),
    repeat([(16, 64, 64)], num_experiments), # sender cells: 0.5×4×4 λ³
    repeat([nothing], num_experiments), # mediator cells (this is an SR system)
    repeat([(16, 64, 64)], num_experiments), # receiver cells, same as sender
    repeat([nothing], num_experiments), # sm separations (SR system)
    repeat([nothing], num_experiments), # mr separations (SR system)
    [(sep, 0//1, 0//1) for sep in separations], # rs separations
    repeat([1//32], num_experiments), # scales
    repeat([14.6+0.05im], num_experiments), # chis
    repeat([64], num_experiments), # RSVD target ranks
    repeat([15], num_experiments), # RSVD oversamples
    repeat([14], num_experiments) # RSVD power iters
))
