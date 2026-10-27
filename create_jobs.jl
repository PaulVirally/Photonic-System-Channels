#!/usr/bin/env julia
#=
Writes the script that runs a sweep: for every separation, the Green operators, the
RSVD and the bounds, in that order. Edit the sweep block below, then

    julia --project=. create_jobs.jl     # writes jobs/<NAME>.sh
    bash jobs/<NAME>.sh                  # submits the chain (Slurm) or runs it (molering)

On Slurm each point is three jobs chained with afterok; the stages skip work whose
output already exists, so resubmitting a sweep only redoes what is missing.
=#

struct Cluster
    name::String
    slurm::Bool
    account::String
    code_dir::String
    preload_dir::String
    project_dir::String
    scratch_dir::String
    setup::String # shell lines run before Julia
end

const USER = ENV["USER"]
const CLUSTERS = Dict(
    "narval" => Cluster("narval", true, "def-smolesky",
        "/home/$(USER)/Photonic-System-Channels",
        "/home/$(USER)/scratch/preload_gila",
        "/home/$(USER)/projects/def-smolesky/$(USER)/Photonic-System-Channels/projects",
        "/home/$(USER)/scratch/Photonic-System-Channels",
        """module load StdEnv/2023 julia/1.12.5 cuda/12.2
        export JULIA_DEPOT_PATH=/home/$(USER)/scratch/julia_depot_psc:\$HOME/.julia"""),
    "molering" => Cluster("molering", false, "",
        "/home/paulv/Projects/Photonic-System-Channels",
        "/home/molering/fatmole/greens_functions",
        "/home/paulv/Projects/Photonic-System-Channels/projects",
        "/home/molering/fatmole/paulv/Photonic-System-Channels",
        ""),
)

"Slurm request for one stage. `gpu` is a Slurm GPU type (\"a100\", \"a100_3g.20gb\") or `nothing`."
Base.@kwdef struct Stage
    time::String
    cpus::Int
    mem::String
    gpu::Union{Nothing,String} = nothing
end

# --- the sweep ---------------------------------------------------------------
# Example: resonant germanium (ζ = |χ|²/Im χ = 225), λ/4 cubes, log-spaced gaps.

const CLUSTER = CLUSTERS["narval"]
const NAME = "Ge225_0p25"
const CHI = 18.612 + 1.5502660841406073im
const CELLS = (8, 8, 8)
const SCALE = 1 // 32 # or (1//32, 1//16, 1//16) for anisotropic cells
const SEPARATIONS = unique(round.(Int, exp10.(range(0, log10(32 * 1e4), length=20)))) .// 32
const RANK, OVERSAMPLES, POWER_ITERS = 4000, 50, 6
const K_UU = 512
const GAMMA_RTOL = 1e-12
const RESOURCES = (green=Stage(time="00:30:00", cpus=8, mem="16G"),
                   rsvd=Stage(time="00:30:00", cpus=2, mem="32G", gpu="a100"),
                   bounds=Stage(time="01:00:00", cpus=2, mem="32G", gpu="a100"))

# ------------------------------------------------------------------------------

rstr(r::Rational) = "$(numerator(r))//$(denominator(r))"
scale_arg(s::Rational) = rstr(s)
scale_arg(s::NTuple{3,Rational}) = "($(join(rstr.(s), ",")))"
chi_arg(χ) = "$(real(χ))$(imag(χ) < 0 ? "-" : "+")$(abs(imag(χ)))im"
point_name(sep) = "$(NAME)_$(numerator(sep))ss$(denominator(sep))"

function stage_command(script, sep, gpu::Bool)
    args = ["--sender", "'($(join(CELLS, ",")))'", "--receiver", "'($(join(CELLS, ",")))'",
            "--rs-sep", "'($(rstr(sep)),0//1,0//1)'", "--scale", "'$(scale_arg(SCALE))'",
            "--chi", "'$(chi_arg(CHI))'", "--design", "rs", "--name", point_name(sep),
            "--components", string(RANK), "--oversamples", string(OVERSAMPLES),
            "--power-iterations", string(POWER_ITERS), "--k-uu", string(K_UU),
            "--gamma-rtol", string(GAMMA_RTOL), "--gpu", gpu ? "0" : "false",
            "--preload", "'$(CLUSTER.preload_dir)'",
            "--project", "'$(joinpath(CLUSTER.project_dir, NAME))'",
            "--scratch", "'$(joinpath(CLUSTER.scratch_dir, NAME))'"]
    threads = CLUSTER.slurm ? "\$SLURM_CPUS_PER_TASK" : "16"
    return "julia --project=. --threads=$(threads) $(script) $(join(args, " "))"
end

function sbatch_block(var, stage::Stage, sep, script, dependency)
    log = joinpath(CLUSTER.project_dir, NAME, "logs", "$(point_name(sep))_$(var)_%j.out")
    opts = ["--parsable", "--account=$(CLUSTER.account)", "--job-name=$(point_name(sep))_$(var)",
            "--time=$(stage.time)", "--cpus-per-task=$(stage.cpus)", "--mem=$(stage.mem)",
            "--output='$(log)'"]
    isnothing(stage.gpu) || push!(opts, "--gpus=$(stage.gpu):1")
    isnothing(dependency) || push!(opts, "--dependency=afterok:\$$(dependency)")
    return """
    $(var)=\$(sbatch $(join(opts, " ")) <<'EOF'
    #!/bin/bash
    set -euo pipefail
    $(CLUSTER.setup)
    cd '$(CLUSTER.code_dir)'
    $(stage_command(script, sep, !isnothing(stage.gpu)))
    EOF
    )
    [ -n "\$$(var)" ] || { echo "sbatch failed for $(point_name(sep)) $(var)" >&2; exit 1; }
    echo "$(point_name(sep)) $(var): \$$(var)"
    """
end

function write_script()
    lines = ["#!/bin/bash", "set -euo pipefail",
             "mkdir -p '$(joinpath(CLUSTER.project_dir, NAME, "logs"))'"]
    for sep in SEPARATIONS
        if CLUSTER.slurm
            push!(lines, sbatch_block("green", RESOURCES.green, sep, "generate_green.jl", nothing))
            push!(lines, sbatch_block("rsvd", RESOURCES.rsvd, sep, "generate_rsvd.jl", "green"))
            push!(lines, sbatch_block("bounds", RESOURCES.bounds, sep, "compute_bounds.jl", "rsvd"))
        else
            push!(lines, "cd '$(CLUSTER.code_dir)'", CLUSTER.setup,
                  stage_command("generate_green.jl", sep, false),
                  stage_command("generate_rsvd.jl", sep, true),
                  stage_command("compute_bounds.jl", sep, true))
        end
    end
    path = joinpath(mkpath(joinpath(@__DIR__, "jobs")), "$(NAME).sh")
    write(path, join(filter(!isempty, lines), "\n") * "\n")
    chmod(path, 0o755)
    println("Wrote $(path): $(length(SEPARATIONS)) points on $(CLUSTER.name)")
end

write_script()
