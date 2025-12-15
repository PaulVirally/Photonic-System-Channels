module SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design, char2volume_symbol, volume_symbol2char
export SMRSystem, sender, mediator, receiver, ms_separation, rm_separation, rs_separation, volume, χ, susceptibility, chi, design_regions, universe_regions, universe, design
export load_greens_function, file_prefix

using GilaElectromagnetics
using Serialization
using Dates
using ..Params

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
"""
struct SMRSystem
    sender_volume::GlaVol
    mediator_volume::Union{Nothing, GlaVol}
    receiver_volume::GlaVol
    design_volume::GlaVol
    design_regions::AbstractVector{SMRVolumeSymbol}
    χ::ComplexF64
end

sender(system::SMRSystem) = system.sender_volume
mediator(system::SMRSystem) = system.mediator_volume
receiver(system::SMRSystem) = system.receiver_volume
design(system::SMRSystem) = system.design_volume
universe(system::SMRSystem) = design(system)

design_regions(system::SMRSystem) = system.design_regions
universe_regions(system::SMRSystem) = design_regions(system)

χ(system::SMRSystem) = system.χ
susceptibility(system::SMRSystem) = χ(system)
chi(system::SMRSystem) = χ(system)

ms_separation(system::SMRSystem) = sender(system).org .- mediator(system).org
rm_separation(system::SMRSystem) = mediator(system).org .- receiver(system).org
rs_separation(system::SMRSystem) = sender(system).org .- receiver(system).org

function SMRSystem(sender_num_cells::NTuple{3, Int}, sm_separation_wl::NTuple{3, Rational{Int}}, mediator_num_cells::NTuple{3, Int}, mr_separation_wl::NTuple{3, Rational{Int}}, receiver_num_cells::NTuple{3, Int}, design_regions::AbstractVector{SMRVolumeSymbol}, scale::Rational{Int}, χ::ComplexF64)
    sender_center_wl = (0//1, 0//1, 0//1)
    sender_size_wl = sender_num_cells .* scale
    sm_dir = (1, 0, 0) # Assume separation along x-axis

    mediator_size_wl = mediator_num_cells .* scale
    mediator_center_wl = sender_center_wl .+ sender_size_wl .* sm_dir .+ sm_separation_wl

    mr_dir = (1, 0, 0) # Assume separation along x-axis
    receiver_center_wl = mediator_center_wl .+ mediator_size_wl .* mr_dir .+ mr_separation_wl

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

function SMRSystem(sender_num_cells::NTuple{3, Int}, rs_separation_wl::NTuple{3, Rational{Int}}, receiver_num_cells::NTuple{3, Int}, design_regions::AbstractVector{SMRVolumeSymbol}, scale::Rational{Int}, χ::ComplexF64)
    sender_center_wl = (0//1, 0//1, 0//1)
    sender_size_wl = sender_num_cells .* scale
    rs_dir = (1, 0, 0) # Assume separation along x-axis

    receiver_center_wl = sender_center_wl .+ sender_size_wl .* rs_dir .+ rs_separation_wl

    sender_volume = GlaVol(sender_num_cells, (scale, scale, scale), sender_center_wl)
    receiver_volume = GlaVol(receiver_num_cells, (scale, scale, scale), receiver_center_wl)

    isempty(design_regions) && throw(ArgumentError("Design regions cannot be empty. You must specify at least one of Sender or Receiver for the design region."))
    design_volumes = GlaVol[]
    for region in design_regions
        if region == Sender
            push!(design_volumes, sender_volume)
        elseif region == Receiver
            push!(design_volumes, receiver_volume)
        end
    end
    design_volume = union(design_volumes...)

    return SMRSystem(sender_volume, nothing, receiver_volume, design_volume, design_regions, χ)
end

# Generate the filename for the Green's function between the target and source volumes.
function greens_fname(target_volume::GlaVol, source_volume::GlaVol)
    rational2str(r::Rational) = string(numerator(r), "ss", denominator(r))
    if target_volume == source_volume
         # Self greens function
         which = "self"
         size = join(target_volume.cel, "x")
         scale = join(map(rational2str, target_volume.scl), "x")
         return "$(which)/$(size)_$(scale).glaG0"
    end
    # External green's function
    which = "ext"
    size_source = join(source_volume.cel, "x")
    scale_source = join(map(rational2str, source_volume.scl), "x")
    pos_source = join(map(rational2str, source_volume.org), "x")
    size_target = join(target_volume.cel, "x")
    scale_target = join(map(rational2str, target_volume.scl), "x")
    pos_target = join(map(rational2str, target_volume.org), "x")
    return "$(which)/$(size_source)_$(scale_source)@$(pos_source)_to_$(size_target)_$(scale_target)@$(pos_target).glaG0"
end

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

"""
    load_greens_function(environment::ComputeEnvironment, system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol)

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
- `G₀::VacuumGreensOperator`: The vacuum Green's function operator between the target and source volumes.
"""
function load_greens_function(environment::ComputeEnvironment, system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol; force_generate::Bool=false, save_to_disk::Bool=true)
    target_volume = volume(system, target)
    source_volume = volume(system, source)

    fname = greens_fname(target_volume, source_volume)
    fpath = joinpath(preload_dir(environment), fname)

    if isfile(fpath) && !force_generate
        volumes_overlap = GilaElectromagnetics.GilaOperators.ovrChk(target_volume, source_volume)
        if volumes_overlap
            source_mask = GilaElectromagnetics.GilaOperators.mskRng(target_volume, source_volume)
            target_mask = GilaElectromagnetics.GilaOperators.mskRng(source_volume, target_volume)
        else
            source_mask = (0:0, 0:0, 0:0)
            target_mask = (0:0, 0:0, 0:0)
        end
        @info string(now()) * " [SMRSystem::load_greens_function] Loading G₀ from $(fpath)"
        io = open(fpath, "r")
        G₀ = VacuumGreensOperator(deserialize(io, VacuumGreensOperator).mem, source_mask, target_mask)
        @info string(now()) * " [SMRSystem::load_greens_function] Loaded G₀"
        close(io)
        if use_gpu(environment)
            @info string(now()) * " [SMRSystem::load_greens_function] Moving G₀ to GPU"
            useGpu!(G₀)
        end
        @info string(now()) * " [SMRSystem::load_greens_function] Using G₀:" G₀
        return G₀
    end
    @info string(now()) * " [SMRSystem::load_greens_function] Generating G₀"
    G₀ = VacuumGreensOperator(target_volume, source_volume)
    @info string(now()) * " [SMRSystem::load_greens_function] Loaded G₀"
    if save_to_disk
        mkpath(dirname(fpath))
        @info string(now()) * " [SMRSystem::load_greens_function] Saving G₀ to $(fpath)"
        io = open(fpath, "w")
        serialize(io, G₀)
        close(io)
    end
    if use_gpu(environment)
        @info string(now()) * " [SMRSystem::load_greens_function] Moving G₀ to GPU"
        useGpu!(G₀)
    end
    @info string(now()) * " [SMRSystem::load_greens_function] Using G₀:" G₀
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

    if isnothing(medium_volume)
        prefix = "$(sender_size)__$(receiver_size)__$(numerator(spacing))ss$(denominator(spacing))__$(universe_string)"
    else
        prefix = "$(sender_size)__$(medium_size)__$(receiver_size)__$(numerator(spacing))ss$(denominator(spacing))__$(universe_string)"
    end
    return prefix
end

end # module
