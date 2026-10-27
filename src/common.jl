using Dates
using ArgParse
import CUDA
import Funicular
import HDF5 # Funicular's disk tier only exists once HDF5 is loaded

function ArgParse.parse_item(::Type{GPUChoice}, x::AbstractString)
    s = lowercase(String(x))
    s in ("true", "t", "yes", "y") && return GPUChoice(true, 0)
    s in ("false", "f", "no", "n") && return GPUChoice(false, -1)
    id = tryparse(Int, s)
    isnothing(id) && error("Invalid GPU argument '$x'. Use true/false or an integer GPU device index.")
    return GPUChoice(true, id)
end

function parse_3tuple(s::AbstractString, conv::Function)
    parts = split(strip(strip(s), ['(', ')']), ',')
    length(parts) == 3 || error("Expected 3 components, got $(length(parts)) in '$s'")
    return Tuple(conv(strip(p)) for p in parts)
end

function parse_rational(s::AbstractString)
    numden = split(strip(s), "//"; limit=2)
    length(numden) == 2 || error("Invalid rational '$s', expected a//b")
    return parse(Int, numden[1]) // parse(Int, numden[2])
end

ArgParse.parse_item(::Type{NTuple{3, Int}}, s::AbstractString) = parse_3tuple(s, x -> parse(Int, x))
ArgParse.parse_item(::Type{NTuple{3, Rational{Int}}}, s::AbstractString) = parse_3tuple(s, parse_rational)
parse_scale(s::AbstractString) = startswith(strip(s), "(") ? parse_3tuple(s, parse_rational) : parse_rational(s)

# Default directories: the group server (MOLERING), an Alliance cluster (CC_CLUSTER), or a laptop.
function _default_dirs(project_name::AbstractString)
    user = ENV["USER"]
    if haskey(ENV, "MOLERING")
        return (preload=joinpath(ENV["MOLERING"], "fatmole", "greens_functions"),
                project=joinpath("/home", user, "Sender-Mediator-Receiver SVD Bounds", "projects", project_name),
                scratch=joinpath(ENV["MOLERING"], "fatmole", user, "SMR-Bounds", project_name))
    elseif haskey(ENV, "CC_CLUSTER")
        # Not ~/scratch/preload: those files were serialized by older Gila versions.
        return (preload=joinpath("/home", user, "scratch", "preload_gila"),
                project=joinpath("/home", user, "projects", "def-smolesky", user, project_name),
                scratch=joinpath("/home", user, "scratch", project_name))
    end
    desktop = joinpath("/Users", user, "Desktop")
    return (preload=joinpath(desktop, "preload"),
            project=joinpath(desktop, project_name),
            scratch=joinpath(desktop, project_name, "scratch"))
end

_default_gpu() = (haskey(ENV, "MOLERING") || haskey(ENV, "CC_CLUSTER")) ? GPUChoice(true, 0) : GPUChoice(false, -1)

"""
    parse_cli() -> NamedTuple

Parse the command line shared by `generate_green`, `generate_rsvd` and
`compute_bounds`. Returns `(compute_env, smr, rsvd_params, gamma_rtol, k_uu)`;
each stage reads the fields it needs.
"""
function parse_cli()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--sender"
            help = "Sender volume size as (x,y,z) with integer number of cells"
            arg_type = NTuple{3, Int}
            required = true
        "--mediator"
            help = "Mediator volume size as (x,y,z) with integer number of cells"
            arg_type = NTuple{3, Int}
            default = (0, 0, 0)
        "--receiver"
            help = "Receiver volume size as (x,y,z) with integer number of cells"
            arg_type = NTuple{3, Int}
            required = true
        "--sm-sep"
            help = "Sender–mediator separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
        "--mr-sep"
            help = "Mediator–receiver separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
        "--rs-sep"
            help = "Sender–receiver separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
        "--scale"
            help = "Cell size in wavelengths: a//b, or (a//b,c//d,e//f) for anisotropic cells (sender/receiver systems only)"
            arg_type = String
            required = true
        "--chi"
            help = "Complex susceptibility χ = a + bi (pass 'a + bi')"
            arg_type = ComplexF64
            required = true
        "--name"
            help = "Project name"
            arg_type = String
            required = true
        "--design"
            help = "Design regions, e.g. 'rs'"
            arg_type = String
            required = true
        "--preload"
            help = "Directory for cached Green functions"
            arg_type = String
        "--project"
            help = "Project directory (bounds output)"
            arg_type = String
        "--scratch"
            help = "Scratch directory (RSVD output)"
            arg_type = String
        "--gpu"
            help = "Use GPU acceleration: true/false or a device index"
            arg_type = GPUChoice
            default = _default_gpu()
        "--components"
            help = "Number of RSVD components (capped at the operator size)"
            arg_type = Int
            default = 256
        "--oversamples"
            help = "Number of RSVD oversamples"
            arg_type = Int
            default = 20
        "--power-iterations"
            help = "Number of RSVD power iterations"
            arg_type = Int
            default = 14
        "--seed"
            help = "Seed for the panel RSVD's test matrix (0 derives it from --name)"
            arg_type = Int
            default = 0
        "--gamma-rtol"
            help = "Bounds: keep only the positive Asym(G⁰ᵤᵣ) eigenvalues Γ ≥ gamma-rtol · Γ₁ (0 keeps all)"
            arg_type = Float64
            default = DEFAULT_GAMMA_RTOL
        "--k-uu"
            help = "Bounds: number of Asym(G⁰ᵤᵤ) eigenvectors added to the projection basis (0 disables)"
            arg_type = Int
            default = DEFAULT_K_UU
        "--no-refine"
            help = "Mesh the sender and receiver uniformly even when their gap is under $(MIN_GAP_CELLS) cells"
            action = :store_true
    end
    args = ArgParse.parse_args(settings)

    project_name = args["name"]
    dirs = _default_dirs(project_name)
    compute_env = ComputeEnvironment(something(args["preload"], dirs.preload),
                                     something(args["project"], dirs.project),
                                     something(args["scratch"], dirs.scratch),
                                     args["gpu"])
    mkpath.((preload_dir(compute_env), project_dir(compute_env), scratch_dir(compute_env)))
    @info string(now()) * " [common::parse_cli] Working on $project_name" preload_dir(compute_env) project_dir(compute_env) scratch_dir(compute_env) use_gpu(compute_env)

    design_symbols = design_regions_from(args["design"])
    if args["mediator"] == (0, 0, 0)
        isnothing(args["rs-sep"]) && error("No mediator specified, so --rs-sep is required")
        smr = SMRSystem(args["sender"], args["rs-sep"], args["receiver"], design_symbols,
                        parse_scale(args["scale"]), args["chi"]; refine_gap=!args["no-refine"])
        @info string(now()) * " [common::parse_cli] SR system" rs_separation(smr) refinement(smr)
    else
        (isnothing(args["sm-sep"]) || isnothing(args["mr-sep"])) &&
            error("A mediator was specified, so --sm-sep and --mr-sep are required")
        smr = SMRSystem(args["sender"], args["sm-sep"], args["mediator"], args["mr-sep"],
                        args["receiver"], design_symbols, parse_rational(args["scale"]), args["chi"])
        @info string(now()) * " [common::parse_cli] SMR system" ms_separation(smr) rm_separation(smr)
    end

    rsvd_params = RSVDParams(args["components"], args["oversamples"], args["power-iterations"],
                             resolve_seed(args["seed"], project_name))
    @info string(now()) * " [common::parse_cli] RSVD parameters" rsvd_params args["gamma-rtol"] args["k-uu"]
    return (compute_env=compute_env, smr=smr, rsvd_params=rsvd_params,
            gamma_rtol=args["gamma-rtol"], k_uu=args["k-uu"])
end

"""
    resolve_seed(requested, project_name) -> Int

`requested`, or a positive seed derived from the project name when it is 0, so
each point of a sweep sketches differently and a rerun sketches the same way
(within one Julia version: `hash` is not stable across versions).
"""
resolve_seed(requested::Int, project_name::AbstractString) =
    requested == 0 ? Int(hash(project_name) >>> 1) : requested

function select_device(compute_env::ComputeEnvironment)
    if !use_gpu(compute_env)
        @info string(now()) * " [common::select_device] Using the CPU"
        return
    end
    @info string(now()) * " [common::select_device] Using GPU $(gpu_device(compute_env))"
    # Slurm already restricts the visible devices on the Alliance clusters, and selecting one there fails.
    haskey(ENV, "CC_CLUSTER") || CUDA.device!(gpu_device(compute_env))
end

# Host memory kept back from the panel tier for the runtime, the operator's host side and
# the page cache the positive-block h5 write dirties (Slurm charges it to the job). 8 GB is
# the measured shortfall of a 1 λ panel job that was OOM-killed with 6 GB.
const HOST_OVERHEAD_RESERVE_BYTES = 8 * 2^30
const HOST_BUDGET_FLOOR_BYTES = 2^30
# The rest of the card absorbs the driver and the memory pool's cached blocks.
const DEVICE_BUDGET_FRACTION = 0.9

device_budget_bytes() = floor(Int, DEVICE_BUDGET_FRACTION * CUDA.total_memory())

# What Slurm granted, in bytes; the machine's memory off a cluster.
function _slurm_host_bytes()
    units = Dict('K' => 2^10, 'M' => 2^20, 'G' => 2^30, 'T' => 2^40)
    mem_bytes(raw) = begin
        isempty(raw) && return nothing
        unit = get(units, uppercase(last(raw)), nothing)
        value = tryparse(Int, isnothing(unit) ? raw : raw[1:end-1])
        isnothing(value) ? nothing : value * something(unit, 2^20) # bare numbers are MB
    end
    node = mem_bytes(strip(get(ENV, "SLURM_MEM_PER_NODE", "")))
    isnothing(node) || return node
    per_cpu = mem_bytes(strip(get(ENV, "SLURM_MEM_PER_CPU", "")))
    cpus = tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", ""))
    (isnothing(per_cpu) || isnothing(cpus)) || return per_cpu * cpus
    return Int(Sys.total_memory())
end

#=
A Funicular host pool never gives its pinned slabs back to the OS: they go only when the
whole plan is garbage collected. Two panel decompositions in one process (UR_asym, then
RS) would each budget the whole allocation and OOM, so every plan is tracked here and the
next one is budgeted around whatever the earlier ones still hold.
=#
const LIVE_RESIDENCY_PLANS = WeakRef[]

function live_pool_host_bytes()
    filter!(ref -> ref.value !== nothing, LIVE_RESIDENCY_PLANS)
    return sum(ref -> Funicular.host_bytes_reserved(ref.value.hostpool), LIVE_RESIDENCY_PLANS; init=0)
end

"""
    reclaim_host_pools!() -> Int

Collect residency plans that are no longer referenced and return the pinned host bytes
still held. Call it between panel decompositions, from the frame above the one that
built the plan.
"""
function reclaim_host_pools!()
    run_gc()
    held = live_pool_host_bytes()
    held > 0 && @warn string(now()) * " [common::reclaim_host_pools!] $(held) bytes of pinned host pool are still reachable; the next residency plan is budgeted around them"
    return held
end

"""
    residency_plan(compute_env; workspace_bytes=0) -> Funicular.ResidencyPlan or nothing

The plan the panel path streams through, sized from the job's allocation: 90% of the
card, and the Slurm memory request less a reserve and less what earlier plans still
hold. Spills to node-local `\$SLURM_TMPDIR` when there is one. `nothing` on a CPU run.
`workspace_bytes` is the operator's own device footprint (see `gila_workspace_bytes`).
"""
function residency_plan(compute_env::ComputeEnvironment; workspace_bytes::Int=0)
    use_gpu(compute_env) || return nothing
    already_pinned = live_pool_host_bytes()
    host_budget = max(_slurm_host_bytes() - HOST_OVERHEAD_RESERVE_BYTES - already_pinned,
                      HOST_BUDGET_FLOOR_BYTES)
    scratch = haskey(ENV, "SLURM_TMPDIR") ? mkpath(joinpath(ENV["SLURM_TMPDIR"], "funicular")) : nothing
    @info string(now()) * " [common::residency_plan] Building a residency plan" device_budget=device_budget_bytes() host_budget workspace_bytes scratch already_pinned
    plan = Funicular.ResidencyPlan(backend=Funicular.cuda_backend(),
                                   device_budget=device_budget_bytes(),
                                   host_budget=host_budget,
                                   workspace_bytes=workspace_bytes,
                                   scratch_dir=scratch)
    push!(LIVE_RESIDENCY_PLANS, WeakRef(plan))
    return plan
end

function run_gc()
    GC.gc()
    GC.gc()
    GC.gc()
end
