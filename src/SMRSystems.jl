module SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design, char2volume_symbol, volume_symbol2char
export SMRSystem, sender, mediator, receiver, ms_separation, rm_separation, rs_separation, volume, χ, susceptibility, chi, design_regions, universe_regions, universe, design, volume_pairs
export load_green_function, file_prefix, fix_mask, serialize_atomic
export GapRefinement, gap_refinement, refine_body, dof_length, mesh_tag, CmpBlkOprVac, AsyCmpBlkOprVac
export mesh, sender_mesh, receiver_mesh, refinement, is_refined, MIN_GAP_CELLS

using GilaElectromagnetics
using Serialization
using Dates
using ..Params

include("refinement.jl")

"""
    SMRVolumeSymbol

An enumeration representing the four volumes in an SMR system: Sender, Mediator, and Receiver, as well as Design (sometimes the union of all three volumes).
"""
@enum SMRVolumeSymbol begin
    Sender
    Mediator
    Receiver
    Design
end
# const Universe = SMRVolumeSymbol.Design # Alias for Design volume

function char2volume_symbol(c::AbstractChar)
    if c == 'S'
        return Sender
    elseif c == 'M'
        return Mediator
    elseif c == 'R'
        return Receiver
    elseif c == 'D' || c == 'U'
        return Design
    end
    throw(ArgumentError("Invalid volume symbol character: $c"))
end

function volume_symbol2char(symbol::SMRVolumeSymbol)
    if symbol == Sender
        return 'S'
    elseif symbol == Mediator
        return 'M'
    elseif symbol == Receiver
        return 'R'
    elseif symbol == Design
        return 'D'
    end
    throw(ArgumentError("Invalid SMRVolumeSymbol: $symbol"))
end

"""
    SMRSystem

A structure representing 3 volumes, a sender, mediator (optional), and receiver, along with a design region (maybe the union of the three) and the complex susceptibility of we are allowed to design with.

# Fields
- `sender_volume::GlaVol`: The sender volume.
- `mediator_volume::Union{Nothing, GlaVol}`: The mediator volume (can be `nothing` if no mediator is present).
- `receiver_volume::GlaVol`: The receiver volume.
- `design_volume::GlaVol`: The design volume.
- `design_regions::AbstractVector{SMRVolumeSymbol}`: The regions that are part of the design volume.
- `χ::ComplexF64`: The complex susceptibility of the mediator.
- `sender_mesh::GlaCmpVol`: The tiling the sender is actually discretized on.
- `receiver_mesh::GlaCmpVol`: The tiling the receiver is actually discretized on.
- `refinement::Union{Nothing, GapRefinement}`: How the two meshes were refined at
    the gap, or `nothing` when they are the plain volumes.

The volumes stay the coarse cuboids whatever the meshes are, so anything that
names an experiment or measures a separation reads the plain geometry. The Green
operators and the degree of freedom counts are built from the meshes.
"""
struct SMRSystem
    sender_volume::GlaVol
    mediator_volume::Union{Nothing, GlaVol}
    receiver_volume::GlaVol
    design_volume::GlaVol
    design_regions::AbstractVector{SMRVolumeSymbol}
    χ::ComplexF64
    sender_mesh::GlaCmpVol
    receiver_mesh::GlaCmpVol
    refinement::Union{Nothing, GapRefinement}
end

SMRSystem(sender_volume::GlaVol, mediator_volume::Union{Nothing, GlaVol},
          receiver_volume::GlaVol, design_volume::GlaVol,
          design_regions::AbstractVector{SMRVolumeSymbol}, χ::ComplexF64) =
    SMRSystem(sender_volume, mediator_volume, receiver_volume, design_volume,
              design_regions, χ, GlaCmpVol(sender_volume), GlaCmpVol(receiver_volume),
              nothing)

sender(system::SMRSystem) = system.sender_volume
mediator(system::SMRSystem) = system.mediator_volume
receiver(system::SMRSystem) = system.receiver_volume
design(system::SMRSystem) = system.design_volume
universe(system::SMRSystem) = design(system)

design_regions(system::SMRSystem) = system.design_regions
universe_regions(system::SMRSystem) = design_regions(system)

sender_mesh(system::SMRSystem) = system.sender_mesh
receiver_mesh(system::SMRSystem) = system.receiver_mesh
refinement(system::SMRSystem) = system.refinement
is_refined(system::SMRSystem) = !isnothing(system.refinement)

"""
    mesh(system::SMRSystem, symbol::SMRVolumeSymbol)

Get the tiling that the given volume of the system is discretized on.

# Arguments
- `system::SMRSystem`: The SMR system.
- `symbol::SMRVolumeSymbol`: The volume symbol (Sender, Mediator, or Receiver).
    `Design` has no tiling: the universe spans the gap, and a composite volume is
    one solid cuboid.

# Return
- `mesh::GlaCmpVol`: The tiling corresponding to the given symbol. It is the plain
    volume unless the gap was refined.
"""
function mesh(system::SMRSystem, symbol::SMRVolumeSymbol)
    symbol == Sender && return sender_mesh(system)
    symbol == Receiver && return receiver_mesh(system)
    symbol == Mediator && return GlaCmpVol(mediator(system))
    # `CmpBlkOprVac` assembles the universe out of the two meshes instead.
    symbol == Design && throw(ArgumentError("The universe spans the gap between the sender and the receiver, so it has no single tiling. Ask for the sender and receiver meshes instead."))
    throw(ArgumentError("Invalid SMRVolumeSymbol: $symbol"))
end

# The cache-key fragment for one volume of the system, empty when its mesh is the
# plain volume. See `mesh_tag`.
volume_tag(system::SMRSystem, symbol::SMRVolumeSymbol) =
    symbol == Design ? mesh_tag(sender_mesh(system), sender(system)) *
                       mesh_tag(receiver_mesh(system), receiver(system)) :
                       mesh_tag(mesh(system, symbol), volume(system, symbol))

χ(system::SMRSystem) = system.χ
susceptibility(system::SMRSystem) = χ(system)
chi(system::SMRSystem) = χ(system)

ms_separation(system::SMRSystem) = abs.(sender(system).org .- mediator(system).org)
rm_separation(system::SMRSystem) = abs.(mediator(system).org .- receiver(system).org)
function rs_separation(system::SMRSystem)
    rs_dir = (1, 0, 0) # Assume separation along x-axis
    snd, rcv = sender(system), receiver(system)
    center_to_center = abs.(snd.org .- rcv.org)
    snd_size = snd.cel .* snd.scl
    rcv_size = rcv.cel .* rcv.scl
    half_extents = ((snd_size .+ rcv_size) .// 2) .* rs_dir
    return center_to_center .- half_extents
end

function SMRSystem(sender_num_cells::NTuple{3, Int}, sm_separation_wl::NTuple{3, Rational{Int}}, mediator_num_cells::NTuple{3, Int}, mr_separation_wl::NTuple{3, Rational{Int}}, receiver_num_cells::NTuple{3, Int}, design_regions::AbstractVector{SMRVolumeSymbol}, scale::Rational{Int}, χ::ComplexF64)
    sender_center_wl = (0//1, 0//1, 0//1)
    sender_size_wl = sender_num_cells .* scale
    sm_dir = (1, 0, 0) # Assume separation along x-axis

    mediator_size_wl = mediator_num_cells .* scale
    mediator_center_wl = sender_center_wl .+ (sender_size_wl .// 2) .* sm_dir .+ abs.(sm_separation_wl) .+ (mediator_size_wl .// 2) .* sm_dir # Silently ignore negative separations

    mr_dir = (1, 0, 0) # Assume separation along x-axis
    receiver_center_wl = mediator_center_wl .+ (mediator_size_wl .// 2) .* mr_dir .+ abs.(mr_separation_wl) .+ (receiver_num_cells .* scale .// 2) .* mr_dir # Silently ignore negative separations

    sender_volume = GlaVol(sender_num_cells, (scale, scale, scale), sender_center_wl)
    mediator_volume = GlaVol(mediator_num_cells, (scale, scale, scale), mediator_center_wl)
    receiver_volume = GlaVol(receiver_num_cells, (scale, scale, scale), receiver_center_wl)

    isempty(design_regions) && throw(ArgumentError("Design regions cannot be empty. You must specify at least one of Sender, Mediator, or Receiver for the design region."))
    design_volumes = GlaVol[]
    for region in design_regions
        if region == Sender
            push!(design_volumes, sender_volume)
        elseif region == Mediator
            push!(design_volumes, mediator_volume)
        elseif region == Receiver
            push!(design_volumes, receiver_volume)
        end
    end
    design_volume = union(design_volumes...)

    return SMRSystem(sender_volume, mediator_volume, receiver_volume, design_volume, design_regions, χ)
end

# `refine_gap` is on by default. The cross-scale jacobian that made a two-scale
# composite mesh give an indefinite Asym(G⁰ᵤᵤ) is fixed upstream (rev d4c0516), and
# with it the universe operator is positive semidefinite to roundoff again, so a
# near separation is meshed the way its quadrature needs. `--no-refine` turns it
# off and takes the plain uniform volumes. bench/refined_near_field.jl is the
# measurement.
function SMRSystem(sender_num_cells::NTuple{3, Int}, rs_separation_wl::NTuple{3, Rational{Int}}, receiver_num_cells::NTuple{3, Int}, design_regions::AbstractVector{SMRVolumeSymbol}, scale::Rational{Int}, χ::ComplexF64; refine_gap::Bool=true)

    if scale > zero(typeof(scale))
        scale = (scale, scale, scale)
    else
        # This is a hack to give us differently sized cells. I was too lazy to change the code, so a negative scale means anisotropic
        scale = (1//32, abs(scale), abs(scale))
    end

    sender_center_wl = (0//1, 0//1, 0//1)
    sender_size_wl = sender_num_cells .* scale
    rs_dir = (1, 0, 0) # Assume separation along x-axis

    receiver_size_wl = receiver_num_cells .* scale
    receiver_center_wl = sender_center_wl .+ (sender_size_wl .// 2) .* rs_dir .+ abs.(rs_separation_wl) .+ (receiver_size_wl .// 2) .* rs_dir # Silently ignore negative separations

    sender_volume = GlaVol(sender_num_cells, scale, sender_center_wl)
    receiver_volume = GlaVol(receiver_num_cells, scale, receiver_center_wl)

    isempty(design_regions) && throw(ArgumentError("Design regions cannot be empty. You must specify at least one of Sender or Receiver for the design region."))
    design_volumes = GlaVol[]
    for region in design_regions
        if region == Sender
            push!(design_volumes, sender_volume)
        elseif region == Receiver
            push!(design_volumes, receiver_volume)
        else
            @warn "Region $region cannot be part of the design because there is only a sender and receiver (no mediator) in this system. Ignoring region $region for design volume."
        end
    end
    design_volume = union(design_volumes...)

    ref = refine_gap ? gap_refinement(abs(rs_separation_wl[1]), scale[1]) : nothing
    if isnothing(ref)
        return SMRSystem(sender_volume, nothing, receiver_volume, design_volume, design_regions, χ)
    end
    # The sender faces the gap across its high-x surface and the receiver across
    # its low-x one, so the two slabs sit on opposite faces.
    return SMRSystem(sender_volume, nothing, receiver_volume, design_volume,
                     design_regions, χ,
                     refine_body(sender_volume, ref, :high),
                     refine_body(receiver_volume, ref, :low), ref)
end

function SMRSystem(sender_num_cells::NTuple{3, Int}, mediator_num_cells::Union{Nothing, NTuple{3, Int}}, receiver_num_cells::NTuple{3, Int}, sm_separation_wl::Union{Nothing, NTuple{3, Rational{Int}}}, mr_separation_wl::Union{Nothing, NTuple{3, Rational{Int}}}, rs_separation_wl::Union{Nothing, NTuple{3, Rational{Int}}}, scale::Rational{Int}, χ::ComplexF64; refine_gap::Bool=true)
    if isnothing(mediator_num_cells)
        design_regions = [Sender, Receiver]
        return SMRSystem(sender_num_cells, rs_separation_wl, receiver_num_cells, design_regions, scale, χ; refine_gap=refine_gap)
    end
    # Gap refinement is derived for the sender/receiver pair alone, so a system
    # with a mediator keeps the plain meshes it always had.
    design_regions = [Mediator]
    return SMRSystem(sender_num_cells, sm_separation_wl, mediator_num_cells, mr_separation_wl, receiver_num_cells, design_regions, scale, χ)
end

# Generate the filename for the Green's function between the target and source volumes.
# `target_tag` and `source_tag` name each side's refinement and are empty for a
# plain volume, so an unrefined system keys its blocks where a finished sweep's
# preload directory already has them.
function green_fname(target_volume::GlaVol, source_volume::GlaVol,
                     target_tag::AbstractString="", source_tag::AbstractString="")
    rational2str(r::Rational) = string(numerator(r), "ss", denominator(r))
    if target_volume == source_volume && target_tag == source_tag
         # Self green function
         which = "self"
         size = join(target_volume.cel, "x")
         scale = join(map(rational2str, target_volume.scl), "x")
         return "$(which)/$(size)_$(scale)$(target_tag).glaG0"
    end
    # External green's function
    which = "ext"
    size_source = join(source_volume.cel, "x")
    scale_source = join(map(rational2str, source_volume.scl), "x")
    pos_source = join(map(rational2str, source_volume.org), "x")
    size_target = join(target_volume.cel, "x")
    scale_target = join(map(rational2str, target_volume.scl), "x")
    pos_target = join(map(rational2str, target_volume.org), "x")
    return "$(which)/$(size_source)_$(scale_source)@$(pos_source)$(source_tag)_to_$(size_target)_$(scale_target)@$(pos_target)$(target_tag).glaG0"
end

# The same path relative to the preload directory, for two volumes of a system,
# with each side's refinement tag filled in.
green_fname(system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol) =
    green_fname(volume(system, target), volume(system, source),
                volume_tag(system, target), volume_tag(system, source))

"""
    volume(system::SMRSystem, symbol::SMRVolumeSymbol)

Get the volume corresponding to the given SMRVolumeSymbol in the SMRSystem.

# Arguments
- `system::SMRSystem`: The SMR system containing the sender, mediator, and
    receiver volumes.
- `symbol::SMRVolumeSymbol`: The volume symbol (Sender, Mediator, or Receiver).

# Return
- `volume::GlaVol`: The volume corresponding to the given symbol.
"""
function volume(system::SMRSystem, symbol::SMRVolumeSymbol)
    if symbol == Sender
        return sender(system)
    elseif symbol == Mediator
        return mediator(system)
    elseif symbol == Receiver
        return receiver(system)
    elseif symbol == Design
        return design(system)
    end
    throw(ArgumentError("Invalid SMRVolumeSymbol: $symbol"))
end

function volume_pairs(smr::SMRSystem)
    pairs = @NamedTuple{source::GlaVol, target::GlaVol}[]
    if isnothing(mediator(smr))
        # heat transfer: (ur, ru) [TODO: add (su, us)]
        push!(pairs, (source=receiver(smr), target=universe(smr))) # ur
        push!(pairs, (source=universe(smr), target=receiver(smr))) # ru
    else
        # generic smr: (rs, ms, mm, rm)
        push!(pairs, (source=sender(smr), target=receiver(smr))) # rs
        push!(pairs, (source=sender(smr), target=mediator(smr))) # sm
        push!(pairs, (source=mediator(smr), target=mediator(smr))) # mm
        push!(pairs, (source=mediator(smr), target=receiver(smr))) # mr
    end
    return pairs
end

"""
    serialize_atomic(fpath, obj)

Serialize `obj` to `fpath` via a process-unique temporary file plus a rename.

Green function filenames depend only on the geometry, so a self operator is
shared by every separation in a sweep. Launching N jobs at once means N
processes serializing to the same path. This function ensures that only one
process writes to the final path at a time, and that the file is not corrupted
if multiple processes try to write at once.

# Arguments
- `fpath::AbstractString`: The file path to serialize to.
- `obj`: The object to serialize.

# Return
- `fpath::AbstractString`: The file path that was serialized to.
"""
function serialize_atomic(fpath::AbstractString, obj)
    mkpath(dirname(fpath))
    tmppath = "$(fpath).tmp.$(getpid()).$(rand(UInt32))"
    try
        open(tmppath, "w") do io
            serialize(io, obj)
        end
        mv(tmppath, fpath; force=true)
    catch err
        rm(tmppath; force=true)
        rethrow(err)
    end
    return fpath
end

function fix_mask(mask::StepRange{Int})
    if mask.start > 0 && mask.stop > 0
        return mask
    end
    @info string(now()) * " [rsvd::fix_mask] Detected mask with non-positive start or stop: $(mask). Fixing..."
    new_start = max(1, abs(mask.start))
    new_stop = max(1, abs(mask.stop))
    new_step = abs(mask.step)
    @info string(now()) * " [rsvd::fix_mask] New mask: $(new_start):$(new_step):$(new_stop)"
    return new_start:new_step:new_stop
end

"""
    load_green_function(environment::ComputeEnvironment, system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol)

Load or generate the vacuum Green's function operator G₀ between the target and source volumes in the given SMR system, using the specified compute environment for file paths and GPU usage.

# Arguments
- `environment::ComputeEnvironment`: The compute environment containing directory paths and GPU settings.
- `system::SMRSystem`: The SMR system containing the sender, mediator, and
    receiver volumes.
- `target::SMRVolumeSymbol`: The target volume symbol (Sender, Mediator, or Receiver).
- `source::SMRVolumeSymbol`: The source volume symbol (Sender, Mediator, or Receiver).

# Keyword Arguments
- `force_generate::Bool=false`: If true, forces regeneration of the Green's function even if it exists on disk.
- `save_to_disk::Bool=true`: If true, saves the generated Green's function to disk.

# Return
- `G₀::VacuumGreenOperator`: The vacuum Green's function operator between the target and source volumes.
"""
function load_green_function(environment::ComputeEnvironment, system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol; force_generate::Bool=false, save_to_disk::Bool=true)
    is_refined(system) && return _load_green_composite(environment, system, target, source;
                                                       force_generate=force_generate,
                                                       save_to_disk=save_to_disk)
    target_volume = volume(system, target)
    source_volume = volume(system, source)

    fname = green_fname(target_volume, source_volume)
    fpath = joinpath(preload_dir(environment), fname)

    if isfile(fpath) && !force_generate
        volumes_overlap = GilaElectromagnetics.GilaOperators.ovrChk(target_volume, source_volume)
        if volumes_overlap
            source_mask = GilaElectromagnetics.GilaOperators.mskRng(target_volume, source_volume)
            target_mask = GilaElectromagnetics.GilaOperators.mskRng(source_volume, target_volume)
            fixed_source_mask = fix_mask.(source_mask)
            fixed_target_mask = fix_mask.(target_mask)
        else
            source_mask = (0:0, 0:0, 0:0)
            target_mask = (0:0, 0:0, 0:0)
        end
        @info string(now()) * " [SMRSystem::load_green_function] Loading G₀ from $(fpath)"
        io = open(fpath, "r")
        G₀ = VacuumGreenOperator(deserialize(io, VacuumGreenOperator).mem, source_mask, target_mask)
        @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
        close(io)
        if use_gpu(environment)
            @info string(now()) * " [SMRSystem::load_green_function] Moving G₀ to GPU"
            useGpu!(G₀)
        end
        @info string(now()) * " [SMRSystem::load_green_function] Using G₀:" G₀
        return G₀
    end
    @info string(now()) * " [SMRSystem::load_green_function] Generating G₀"
    G₀ = VacuumGreenOperator(target_volume, source_volume)
    @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
    if save_to_disk
        @info string(now()) * " [SMRSystem::load_green_function] Saving G₀ to $(fpath)"
        serialize_atomic(fpath, G₀)
    end
    if use_gpu(environment)
        @info string(now()) * " [SMRSystem::load_green_function] Moving G₀ to GPU"
        useGpu!(G₀)
    end
    @info string(now()) * " [SMRSystem::load_green_function] Using G₀:" G₀
    return G₀
end

# The refined counterpart of the single block path. The mesh of a refined body is
# a tiling rather than a cuboid, so its block is a composite operator.
function _load_green_composite(environment::ComputeEnvironment, system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol; force_generate::Bool=false, save_to_disk::Bool=true)
    target_mesh = mesh(system, target)
    source_mesh = mesh(system, source)

    fpath = joinpath(preload_dir(environment), green_fname(system, target, source))

    if isfile(fpath) && !force_generate
        @info string(now()) * " [SMRSystem::load_green_function] Loading composite G₀ from $(fpath)"
        G₀ = open(io -> deserialize(io, GlaCmpOprVac), fpath, "r")
        @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
    else
        @info string(now()) * " [SMRSystem::load_green_function] Generating composite G₀"
        G₀ = GlaCmpOprVac(target_mesh, source_mesh)
        @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
        if save_to_disk
            @info string(now()) * " [SMRSystem::load_green_function] Saving G₀ to $(fpath)"
            serialize_atomic(fpath, G₀)
        end
    end
    if use_gpu(environment)
        @info string(now()) * " [SMRSystem::load_green_function] Moving G₀ to GPU"
        useGpu!(G₀)
    end
    @info string(now()) * " [SMRSystem::load_green_function] Using G₀:" G₀
    return G₀
end

# The refined counterpart of the multi-region path. Gila's composite volume has to
# be one solid cuboid, so the universe of two separated bodies is assembled here as
# a block matrix over the two meshes. The four blocks go to disk on their own and
# the assembly is rebuilt on load.
function _load_green_block(environment::ComputeEnvironment, system::SMRSystem, targets::Vector{SMRVolumeSymbol}, source::Vector{SMRVolumeSymbol}; force_generate::Bool=false, save_to_disk::Bool=true)
    fpath = joinpath(preload_dir(environment), green_fname(system, Design, Design))

    if isfile(fpath) && !force_generate
        @info string(now()) * " [SMRSystem::load_green_function] Loading block composite G₀ from $(fpath)"
        G₀ = CmpBlkOprVac(open(deserialize, fpath, "r")::Matrix{GlaCmpOprVac})
        @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
    else
        @info string(now()) * " [SMRSystem::load_green_function] Generating block composite G₀"
        target_meshes = GlaCmpVol[mesh(system, t) for t in targets]
        source_meshes = GlaCmpVol[mesh(system, s) for s in source]
        G₀ = CmpBlkOprVac(target_meshes, source_meshes)
        @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
        if save_to_disk
            @info string(now()) * " [SMRSystem::load_green_function] Saving G₀ to $(fpath)"
            serialize_atomic(fpath, G₀.blocks)
        end
    end
    if use_gpu(environment)
        @info string(now()) * " [SMRSystem::load_green_function] Moving G₀ to GPU"
        useGpu!(G₀)
    end
    @info string(now()) * " [SMRSystem::load_green_function] Using G₀:" G₀
    return G₀
end

function load_green_function(environment::ComputeEnvironment, system::SMRSystem, targets::Vector{SMRVolumeSymbol}, source::Vector{SMRVolumeSymbol}; force_generate::Bool=false, save_to_disk::Bool=true)
    if is_refined(system)
        length(targets) > 1 || length(source) > 1 ||
            return _load_green_composite(environment, system, targets[1], source[1];
                                         force_generate=force_generate, save_to_disk=save_to_disk)
        return _load_green_block(environment, system, targets, source;
                                 force_generate=force_generate, save_to_disk=save_to_disk)
    end
    target_is_design = false
    if length(targets) > 1
        target_is_design = true
    end
    source_is_design = false
    if length(source) > 1
        source_is_design = true
    end
    target_volume = target_is_design ? volume(system, Design) : volume(system, targets[1])
    source_volume = source_is_design ? volume(system, Design) : volume(system, source[1])

    fname = green_fname(target_volume, source_volume)
    fpath = joinpath(preload_dir(environment), fname)

    if isfile(fpath) && !force_generate
        @info string(now()) * " [SMRSystem::load_green_function] Loading G₀ from $(fpath)"
        io = open(fpath, "r")
        G₀ = MultiRegionVacuumGreenOperator(deserialize(io))
        @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
        close(io)
        if use_gpu(environment)
            @info string(now()) * " [SMRSystem::load_green_function] Moving G₀ to GPU"
            useGpu!(G₀)
        end
        @info string(now()) * " [SMRSystem::load_green_function] Using G₀:" G₀
        return G₀
    end
    @info string(now()) * " [SMRSystem::load_green_function] Generating G₀"
    target_volumes = [volume(system, t) for t in targets]
    source_volumes = [volume(system, s) for s in source]
    G₀ = MultiRegionVacuumGreenOperator(target_volumes, source_volumes)
    @info string(now()) * " [SMRSystem::load_green_function] Loaded G₀"
    if save_to_disk
        @info string(now()) * " [SMRSystem::load_green_function] Saving G₀ to $(fpath)"
        serialize_atomic(fpath, G₀)
    end
    if use_gpu(environment)
        @info string(now()) * " [SMRSystem::load_green_function] Moving G₀ to GPU"
        useGpu!(G₀)
    end
    @info string(now()) * " [SMRSystem::load_green_function] Using G₀:" G₀
    return G₀
end

function file_prefix(system::SMRSystem)
    sender_volume = sender(system)
    medium_volume = mediator(system)
    receiver_volume = receiver(system)

    sender_size = join(sender_volume.cel, "x")
    if isnothing(medium_volume)
        medium_size = ""
    else
        medium_size = join(medium_volume.cel, "x")
    end
    receiver_size = join(receiver_volume.cel, "x")

    if isnothing(medium_volume)
        spacing = (2*(receiver_volume.org[1] - sender_volume.org[1]) - receiver_volume.cel[1]*receiver_volume.scl[1] - sender_volume.cel[1]*sender_volume.scl[1])//2
    else
        spacing = (2*(medium_volume.org[1] - sender_volume.org[1]) - medium_volume.cel[1]*medium_volume.scl[1] - sender_volume.cel[1]*sender_volume.scl[1])//2
    end

    universe_string = prod(volume_symbol2char.(universe_regions(system)))

    # Empty unless the gap was refined, so an unrefined point keeps its usual
    # scratch key. The table entry alone identifies the refinement: the sizes it is
    # clamped against are already in the prefix.
    ref = refinement(system)
    refinement_string = isnothing(ref) ? "" : "__refF$(ref.factor)T$(ref.thickness)"

    if isnothing(medium_volume)
        prefix = "$(sender_size)__$(receiver_size)__$(numerator(spacing))ss$(denominator(spacing))__$(universe_string)$(refinement_string)"
    else
        prefix = "$(sender_size)__$(medium_size)__$(receiver_size)__$(numerator(spacing))ss$(denominator(spacing))__$(universe_string)$(refinement_string)"
    end
    return prefix
end

end # module
