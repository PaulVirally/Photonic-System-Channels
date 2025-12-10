using Dates
using Serialization
using ArgParse

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
        path = joinpath("/home", ENV["USERS"], "Sender-Mediator-Receiver SVD Bounds", "projects", project_name)
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
        return true
    end
    return false
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
            arg_type = Bool
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
        args["power-iterations"]
    )
    @info string(now()) * " [common::parse_args] Using RSVD parameters:" rank(rsvd_params) oversamples(rsvd_params) power_iter(rsvd_params)

    return compute_env, smr, rsvd_params
end

function run_gc()
    @info string(now()) * " [common::run_gc] Running garbage collector"
    GC.gc()
    GC.gc()
    GC.gc()
end
