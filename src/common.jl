using Dates
using Serialization
using ArgParse
import CUDA
import Funicular
import HDF5 # Funicular's disk tier only exists once HDF5 is loaded

function ArgParse.parse_item(::Type{GPUChoice}, x::AbstractString)
    s = lowercase(String(x))

    if s in ("true", "t", "yes", "y")
        return GPUChoice(true, 0)
    elseif s in ("false", "f", "no", "n")
        return GPUChoice(false, -1)
    else
        try
            return GPUChoice(true, parse(Int, s))
        catch
            error("Invalid GPU argument '$x'. Use true/false or an integer GPU device index.")
        end
    end
end

function parse_3tuple(s::AbstractString, conv::Function)
    inner = strip(s)
    inner = strip(inner, ['(', ')'])
    parts = split(inner, ',')
    length(parts) == 3 || error("Expected 3 components, got $(length(parts)) in '$s'")
    return (conv(strip(parts[1])),
            conv(strip(parts[2])),
            conv(strip(parts[3])))
end
parse_int_3tuple(s::AbstractString) = parse_3tuple(s, x -> parse(Int, x))
function parse_rational(s::AbstractString)
    numden = split(strip(s), "//"; limit = 2)
    length(numden) == 2 || error("Invalid rational '$s', expected a//b")
    return parse(Int, numden[1]) // parse(Int, numden[2])
end

function Base.parse(::Type{Rational}, s::AbstractString)
    numden = split(strip(s), "//"; limit=2)
    length(numden) == 2 || error("Invalid rational '$s', expected a//b")
    return parse(Int, numden[1]) // parse(Int, numden[2])
end
parse_rational_3tuple(s::AbstractString) = parse_3tuple(s, x -> parse(Rational, x))
ArgParse.parse_item(::Type{NTuple{3, Int}}, s::AbstractString) = parse_int_3tuple(s)
ArgParse.parse_item(::Type{NTuple{3, Rational{Int}}}, s::AbstractString) = parse_rational_3tuple(s)

function _default_preload_dir()
    path = joinpath("/Users", ENV["USER"], "Desktop", "preload")
    if haskey(ENV, "MOLERING")
        # On our group's server
        path = joinpath(ENV["MOLERING"], "fatmole", "greens_functions")
    elseif haskey(ENV, "CC_CLUSTER")
        # On compute Canada
        path = joinpath("/home", ENV["USER"], "scratch", "preload")
    end
    return path
end
function _default_project_dir(project_name::AbstractString)
    path = joinpath("/Users", ENV["USER"], "Desktop", project_name)
    if haskey(ENV, "MOLERING")
        # On our group's server
        path = joinpath("/home", ENV["USER"], "Sender-Mediator-Receiver SVD Bounds", "projects", project_name)
    elseif haskey(ENV, "CC_CLUSTER")
        # On compute Canada
        path = joinpath("/home", ENV["USER"], "projects", "rrg-smolesky", ENV["USER"], project_name)
    end
    return path
end
function _default_scratch_dir(project_name::AbstractString)
    path = joinpath("/Users", ENV["USER"], "Desktop", project_name, "scratch")
    if haskey(ENV, "MOLERING")
        # On our group's server
        path = joinpath(ENV["MOLERING"], "fatmole", ENV["USER"], "SMR-Bounds", project_name)
    elseif haskey(ENV, "CC_CLUSTER")
        # On compute Canada
        path = joinpath("/home", ENV["USER"], "scratch", project_name)
    end
    return path
end
function _default_gpu()
    if haskey(ENV, "MOLERING") || haskey(ENV, "CC_CLUSTER")
        return GPUChoice(true, 0)
    end
    return GPUChoice(false, -1)
end

function ArgParse.parse_args()
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
            default = (1//0, 1//0, 1//0) # To warn if not set and was needed

        "--mr-sep"
            help = "Mediator–receiver separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
            default = (1//0, 1//0, 1//0) # To warn if not set and was needed

        "--rs-sep"
            help = "Sender–receiver separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
            default = (1//0, 1//0, 1//0) # To warn if not set and was needed

        "--scale"
            help = "Mesh scale as a//b in wavelengths per cell"
            arg_type = Rational{Int}
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
            help = "Design region"
            arg_type = String
            required = true

        "--preload"
            help = "Directory for preloaded Green's functions"
            arg_type = String
            default = _default_preload_dir()

        "--project"
            help = "Project directory"
            arg_type = String

        "--scratch"
            help = "Scratch directory"
            arg_type = String

        "--gpu"
            help = "Use GPU acceleration"
            arg_type = GPUChoice
            default = _default_gpu()

        "--components"
            help = "Number of singular value components to compute"
            arg_type = Int
            default = 256

        "--oversamples"
            help = "Number of oversamples to use in RSVD"
            arg_type = Int
            default = 20

        "--power-iterations"
            help = "Number of power iterations to use in RSVD"
            arg_type = Int
            default = 14

        "--seed"
            help = "Seed for the panel path's regenerated test matrix (0 derives it from --name)"
            arg_type = Int
            default = 0
    end
    args = parse_args(settings)

    project_name = args["name"]
    @info string(now()) * " [common::parse_args] Working on $project_name"

    if isnothing(get(args, "project", nothing))
        args["project"] = _default_project_dir(project_name)
    end
    if isnothing(get(args, "scratch", nothing))
        args["scratch"] = _default_scratch_dir(project_name)
    end

    compute_env = ComputeEnvironment(
        get(args, "preload", _default_preload_dir()),
        get(args, "project", _default_project_dir(project_name)),
        get(args, "scratch", _default_scratch_dir(project_name)),
        get(args, "gpu", _default_gpu())
    )
    mkpath(preload_dir(compute_env))
    mkpath(project_dir(compute_env))
    mkpath(scratch_dir(compute_env))
    @info string(now()) * " [common::parse_args] Using compute environment:" preload_dir(compute_env) project_dir(compute_env) scratch_dir(compute_env) use_gpu(compute_env)

    design_symbols = char2volume_symbol.(sort(collect(uppercase(args["design"])))) # sort to ensure consistent naming
    has_mediator = true
    if args["mediator"] == (0, 0, 0)
        any(args["rs-sep"] .== 1//0) && error("No mediator specified, but sender–receiver separation was not set")
        has_mediator = false
    else
        any(args["sm-sep"] .== 1//0) && error("Mediator specified, but sender–mediator separation was not set")
        any(args["mr-sep"] .== 1//0) && error("Mediator specified, but mediator–receiver separation was not set")
    end

    if has_mediator
        smr = SMRSystem(
            args["sender"],
            args["sm-sep"],
            args["mediator"],
            args["mr-sep"],
            args["receiver"],
            design_symbols,
            args["scale"],
            args["chi"]
        )
        @info string(now()) * " [common::parse_args] Using SMR system with mediator" ms_separation(smr) rm_separation(smr) rs_separation(smr)
    else
        smr = SMRSystem(
            args["sender"],
            args["rs-sep"],
            args["receiver"],
            design_symbols,
            args["scale"],
            args["chi"]
        )
        @info string(now()) * " [common::parse_args] Using SR system without mediator" rs_separation(smr)
    end

    rsvd_params = RSVDParams(
        args["components"],
        args["oversamples"],
        args["power-iterations"],
        resolve_seed(args["seed"], project_name)
    )
    @info string(now()) * " [common::parse_args] Using RSVD parameters:" rank(rsvd_params) oversamples(rsvd_params) power_iter(rsvd_params) seed(rsvd_params)

    return compute_env, smr, rsvd_params
end

"""
    resolve_seed(requested::Int, project_name::AbstractString) -> Int

The seed the panel path sketches with. `requested == 0` (the `--seed` default)
derives one from the experiment name, so each separation of a sweep gets its own
seed without us writing 333 of them down; anything else is taken as given. The
result is always positive and depends on the name alone, so a rerun of the same
experiment sketches the same way. Note, however, that this only holds within one
Julia version, since `hash` and `Xoshiro` are implementation details.
"""
function resolve_seed(requested::Int, project_name::AbstractString)
    requested == 0 || return requested
    return Int(hash(project_name) >>> 1) # >>> 1 clears the sign bit, so this fits an Int
end

# Host memory the process needs for itself outside the panel tier: the Julia
# runtime, the Gila operator's host side, the JLD writes, and the page-locking
# slack. Trial E2 pins the real number down; 6 GB is the plan's working estimate.
const HOST_OVERHEAD_RESERVE_BYTES = 6 * 2^30
# Below this there is no point starting, since one row sweep has to hold a whole
# matrix.
const HOST_BUDGET_FLOOR_BYTES = 2^30
# Fraction of the device we hand to Funicular. The rest absorbs the driver, the
# memory pool's cached blocks, and CUDA's own allocations. We read it from
# total_memory rather than free_memory, which under-reports once the pool holds
# cached blocks. On a MIG slice, total_memory reports the slice, not the card.
const DEVICE_BUDGET_FRACTION = 0.9

device_budget_bytes() = floor(Int, DEVICE_BUDGET_FRACTION * CUDA.total_memory())

# Slurm reports memory in MB, but writes the unit out when the request carried
# one ("240G"), so we honour the suffix when there is one.
const SLURM_MEM_UNITS = Dict('K' => 2^10, 'M' => 2^20, 'G' => 2^30, 'T' => 2^40)

function _slurm_mem_bytes(name::String)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return nothing
    unit = get(SLURM_MEM_UNITS, uppercase(last(raw)), nothing)
    digits = unit === nothing ? raw : raw[1:end-1]
    value = tryparse(Int, digits)
    value === nothing && return nothing
    return value * (unit === nothing ? 2^20 : unit) # bare numbers are MB
end

# What Slurm gave us, in bytes. `SLURM_MEM_PER_NODE` is the whole-node request; a
# `--mem-per-cpu` job gets the same number from the per-CPU request times the
# task's CPU count. Off a cluster neither exists, so we fall back to the
# machine's total memory.
function _slurm_host_bytes()
    node = _slurm_mem_bytes("SLURM_MEM_PER_NODE")
    node === nothing || return node
    per_cpu = _slurm_mem_bytes("SLURM_MEM_PER_CPU")
    cpus = tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", ""))
    (per_cpu === nothing || cpus === nothing) || return per_cpu * cpus
    return Int(Sys.total_memory())
end

"""
    residency_plan(compute_env; workspace_bytes=0) -> Funicular.ResidencyPlan or nothing

The `ResidencyPlan` the panel path streams its sketches through, sized from the
allocation this process is running inside. Returns `nothing` on a CPU run: with
no device in the picture and the sketch already in host memory, the panel
machinery only adds bookkeeping, so CPU runs keep the in-memory path
(FUNICULAR_PLAN.md, workstream B1/B3).

Both budgets come from what the job was granted, not from what is momentarily
free:

- `device_budget` is 90% of `CUDA.total_memory()`. The held-back tenth absorbs
  the driver and the memory pool's cached blocks. Nothing else is subtracted
  here. The operator's own device footprint goes in through `workspace_bytes`,
  which the plan holds back from the buffer pool itself.
- `host_budget` is the Slurm memory request minus a $(HOST_OVERHEAD_RESERVE_BYTES >> 30) GB overhead
  reserve for the runtime and the operator's host side, floored at
  $(HOST_BUDGET_FLOOR_BYTES >> 30) GB. Off a cluster it is the machine's total memory, same
  reserve.
- `scratch_dir` is node-local NVMe (`\$SLURM_TMPDIR`) when Slurm gave us one.
  With a scratch dir the host tier no longer has to hold a whole sketch, only
  what a sweep touches at once.

`workspace_bytes` is a keyword rather than a trait because `G₀_ur_asym` is a
LinearMaps composition, which has nowhere to carry one. See
`gila_workspace_bytes` in `rsvd.jl` for the estimate.
"""
function residency_plan(compute_env::ComputeEnvironment; workspace_bytes::Int=0)
    if !use_gpu(compute_env)
        @info string(now()) * " [common::residency_plan] CPU run: no residency plan, the in-memory path is cheaper here"
        return nothing
    end

    device_budget = device_budget_bytes()
    host_budget = max(_slurm_host_bytes() - HOST_OVERHEAD_RESERVE_BYTES, HOST_BUDGET_FLOOR_BYTES)

    scratch = nothing
    if haskey(ENV, "SLURM_TMPDIR")
        scratch = joinpath(ENV["SLURM_TMPDIR"], "funicular")
        mkpath(scratch)
    end

    @info string(now()) * " [common::residency_plan] Building a residency plan" device_budget host_budget workspace_bytes scratch
    return Funicular.ResidencyPlan(
        backend=Funicular.cuda_backend(),
        device_budget=device_budget,
        host_budget=host_budget,
        workspace_bytes=workspace_bytes,
        scratch_dir=scratch
    )
end

function run_gc()
    @info string(now()) * " [common::run_gc] Running garbage collector"
    GC.gc()
    GC.gc()
    GC.gc()
end

asym(X) = (X - X') / (2im)
sym(X) = (X + X') / 2
