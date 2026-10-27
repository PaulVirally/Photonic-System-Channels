module SMRSystems

export SMRVolumeSymbol, Sender, Mediator, Receiver, Design, char2volume_symbol, volume_symbol2char
export design_regions_from, design_string
export SMRSystem, sender, mediator, receiver, ms_separation, rm_separation, rs_separation, volume, susceptibility, design_regions, design
export load_green_function, load_asym_green_function, green_map, asym_green_map, universe_mesh, file_prefix, serialize_atomic
export GapRefinement, gap_refinement, refine_body, dof_length, mesh_tag
export mesh, sender_mesh, receiver_mesh, refinement, is_refined, MIN_GAP_CELLS

using GilaElectromagnetics
using LinearAlgebra
using LinearMaps
using Serialization
import CUDA
using Dates
using ..Params

# Storage precision of every operator built here. Gila defaults to Float32 and the
# bounds need double precision, so every constructor names it.
const GILA_PREC = Float64


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
    design_regions_from(design) -> Vector{SMRVolumeSymbol}

The regions a `--design` string names. Uppercased and **sorted**, so the result is
a function of the set of letters and not of the order they were typed in, so `rs`
and `sr` name the same point and the same files (`file_prefix` reads this order).

See also [`design_string`](@ref), the inverse, which is what a generator should
put on a command line.
"""
design_regions_from(design::AbstractString) =
    char2volume_symbol.(sort(collect(uppercase(design))))

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

design_regions(system::SMRSystem) = system.design_regions

"""
    design_string(system) -> String

The `--design` string that rebuilds `system`'s design regions, for a generator to
put on an emitted command line instead of a literal of its own.

"""
design_string(system::SMRSystem) =
    lowercase(prod(volume_symbol2char.(design_regions(system))))

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
    # `universe_mesh` tiles the universe out of the two meshes instead.
    symbol == Design && throw(ArgumentError("The universe spans the gap between the sender and the receiver, so it has no single tiling. Ask for the sender and receiver meshes instead."))
    throw(ArgumentError("Invalid SMRVolumeSymbol: $symbol"))
end

# The cache-key fragment for one volume of the system, empty when its mesh is the
# plain volume. See `mesh_tag`.
volume_tag(system::SMRSystem, symbol::SMRVolumeSymbol) =
    symbol == Design ? mesh_tag(sender_mesh(system), sender(system)) *
                       mesh_tag(receiver_mesh(system), receiver(system)) :
                       mesh_tag(mesh(system, symbol), volume(system, symbol))

susceptibility(system::SMRSystem) = system.χ

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

# Gap refinement is on unless `refine_gap=false` (`--no-refine`).
# `scale` is one cell size for cubic cells, or (x, y, z) for anisotropic ones.
function SMRSystem(sender_num_cells::NTuple{3, Int}, rs_separation_wl::NTuple{3, Rational{Int}}, receiver_num_cells::NTuple{3, Int}, design_regions::AbstractVector{SMRVolumeSymbol}, scale::Union{Rational{Int}, NTuple{3, Rational{Int}}}, χ::ComplexF64; refine_gap::Bool=true)
    scale isa Rational && (scale = (scale, scale, scale))
    all(>(0), scale) || throw(ArgumentError("The mesh scale must be positive, got $(scale)"))

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
    fpath = joinpath(preload_dir(environment), green_fname(target_volume, source_volume))
    return _load_or_build(environment, fpath, "G₀"; force_generate=force_generate, save_to_disk=save_to_disk) do
        GlaOprVac{GILA_PREC}(target_volume, source_volume)
    end
end

# Load the operator at `fpath`, or build it on the CPU with `build` and save it there,
# then move it to the GPU when the environment asks for one.
function _load_or_build(build, environment::ComputeEnvironment, fpath::AbstractString, what::AbstractString;
                        force_generate::Bool=false, save_to_disk::Bool=true)
    if isfile(fpath) && !force_generate
        @info string(now()) * " [SMRSystem::load_green_function] Loading $(what) from $(fpath)"
        G₀ = open(deserialize, fpath, "r")
    else
        @info string(now()) * " [SMRSystem::load_green_function] Generating $(what)"
        G₀ = build()
        if save_to_disk
            @info string(now()) * " [SMRSystem::load_green_function] Saving $(what) to $(fpath)"
            serialize_atomic(fpath, G₀)
        end
    end
    @info string(now()) * " [SMRSystem::load_green_function] Loaded $(what)"
    if use_gpu(environment)
        @info string(now()) * " [SMRSystem::load_green_function] Moving $(what) to GPU"
        G₀ = adapt(CUDA.CUDABackend(), G₀)
    end
    @info string(now()) * " [SMRSystem::load_green_function] Using $(what):" G₀
    return G₀
end

# The refined counterpart of the single block path. The mesh of a refined body is
# a tiling rather than a cuboid, so its block is a composite operator.
function _load_green_composite(environment::ComputeEnvironment, system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol; force_generate::Bool=false, save_to_disk::Bool=true)
    target_mesh = mesh(system, target)
    source_mesh = mesh(system, source)
    fpath = joinpath(preload_dir(environment), green_fname(system, target, source))
    return _load_or_build(environment, fpath, "composite G₀"; force_generate=force_generate, save_to_disk=save_to_disk) do
        CompositeVacuumGreenOperator{GILA_PREC}(target_mesh, source_mesh)
    end
end

"""
    universe_mesh(system::SMRSystem) -> GlaCmpVol

The universe as one tiling: the sender's regions followed by the receiver's.

Gila's composite volume takes separated bodies, so the gap is simply not part of
the tiling and the flat layout is `[sender; receiver]`, the convention `asym_ur`
and the bounds' sender projector are written against. An unrefined body is a
tiling of one region, so refined and unrefined systems take the same path.
"""
universe_mesh(system::SMRSystem) =
    GlaCmpVol(vcat(regions(sender_mesh(system)), regions(receiver_mesh(system))))

# Where the universe operator lives in the preload directory. Named after the
# bounding volume as before, with `uni_` in front so that a file of the old
# bounding-box multi-region layout can never be read as the new one.
function _universe_fname(system::SMRSystem; asym::Bool=false)
    which, rest = split(green_fname(system, Design, Design), "/"; limit=2)
    rest = replace(rest, ".glaG0" => (asym ? ".glaAsyG0" : ".glaG0"))
    return joinpath(which, "uni_" * rest)
end

function load_green_function(environment::ComputeEnvironment, system::SMRSystem, targets::Vector{SMRVolumeSymbol}, source::Vector{SMRVolumeSymbol}; force_generate::Bool=false, save_to_disk::Bool=true)
    length(targets) == 1 && length(source) == 1 &&
        return load_green_function(environment, system, targets[1], source[1];
                                   force_generate=force_generate, save_to_disk=save_to_disk)
    (sort(targets) == [Sender, Receiver] && sort(source) == [Sender, Receiver]) ||
        throw(ArgumentError("only the universe [Sender, Receiver] is supported as a multi-body operator, got $(targets) <- $(source)"))
    cvol = universe_mesh(system)
    fpath = joinpath(preload_dir(environment), _universe_fname(system))
    return _load_or_build(environment, fpath, "universe G₀"; force_generate=force_generate, save_to_disk=save_to_disk) do
        CompositeVacuumGreenOperator{GILA_PREC}(cvol, cvol)
    end
end

"""
    load_asym_green_function(environment, system, which::SMRVolumeSymbol) -> operator

`Asym(G⁰)` of a self operator, built from the volume rather than from `G⁰`.

`which` is `Sender`, `Receiver` or `Design` (the universe). Gila takes the
imaginary part of the kernel before its transform (`hrmPrt=:asy`), which keeps
the digits that `(G - G')/2im` cancels away and keeps the operator positive
semidefinite. Cached next to `G⁰` in the preload directory as a `.glaAsyG0` file.
"""
function load_asym_green_function(environment::ComputeEnvironment, system::SMRSystem, which::SMRVolumeSymbol; force_generate::Bool=false, save_to_disk::Bool=true)
    if which == Design
        cvol = universe_mesh(system)
        fpath = joinpath(preload_dir(environment), _universe_fname(system; asym=true))
        return _load_or_build(environment, fpath, "universe Asym(G₀)"; force_generate=force_generate, save_to_disk=save_to_disk) do
            CompositeVacuumGreenOperator{GILA_PREC}(cvol; hrmPrt=:asy)
        end
    end
    m = mesh(system, which)
    fpath = joinpath(preload_dir(environment), replace(green_fname(system, which, which), ".glaG0" => ".glaAsyG0"))
    return _load_or_build(environment, fpath, "Asym(G₀)"; force_generate=force_generate, save_to_disk=save_to_disk) do
        length(regions(m)) == 1 ? GlaOprVac{GILA_PREC}(only(regions(m)); hrmPrt=:asy) :
                                  CompositeVacuumGreenOperator{GILA_PREC}(m; hrmPrt=:asy)
    end
end

"""
    green_map(environment, system, target, source) -> LinearMap
    asym_green_map(environment, system, which) -> LinearMap

The Green operator, or `Asym(G⁰)` of a self block, in L²-normalised unknowns:
`ΔVₜ^½ G ΔVₛ^-½`. Gila's operators act on cell-average current densities and are
self-adjoint only in the ΔV-weighted inner product, so on a mesh with more than one
cell size the bare operator is not Hermitian in the plain inner product that every
eigensolve and projection here assumes. On a uniform mesh the weights cancel and
this is `LinearMap(op)`. `Design` is the universe, sender then receiver.
"""
green_map(environment::ComputeEnvironment, system::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol) =
    _l2_map(load_green_function(environment, system, target, source),
            _cell_mesh(system, target), _cell_mesh(system, source), use_gpu(environment))

asym_green_map(environment::ComputeEnvironment, system::SMRSystem, which::SMRVolumeSymbol) =
    _l2_map(load_asym_green_function(environment, system, which),
            _cell_mesh(system, which), _cell_mesh(system, which), use_gpu(environment))

_cell_mesh(system::SMRSystem, symbol::SMRVolumeSymbol) = symbol == Design ? universe_mesh(system) : mesh(system, symbol)

function _l2_map(op, trg::GlaCmpVol, src::GlaCmpVol, on_gpu::Bool)
    wt, ws = cellvolumes(trg), cellvolumes(src)
    L = LinearMap(op)
    allequal(wt) && allequal(ws) && wt[1] == ws[1] && return L
    ref = max(maximum(wt), maximum(ws)) # one constant on both sides, so it cancels
    dt, ds = sqrt.(wt ./ ref), sqrt.(ws ./ ref)
    on_gpu && ((dt, ds) = (CUDA.CuArray(dt), CUDA.CuArray(ds)))
    f!(y, x) = (mul!(y, L, x ./ ds); y .*= dt; y)
    fc!(x, y) = (mul!(x, L', y ./ dt); x .*= ds; x) # L' is Gila's ΔV-weighted adjoint
    return LinearMap{eltype(L)}(f!, fc!, size(L)...; ismutating=true)
end

# Names a point's RSVD and bounds files: cell counts, the gap to the first body after the
# sender, the cell size, the design regions, and the gap refinement if any.
function file_prefix(system::SMRSystem)
    snd, med, rcv = sender(system), mediator(system), receiver(system)
    next = isnothing(med) ? rcv : med
    gap = (2 * (next.org[1] - snd.org[1]) - next.cel[1] * next.scl[1] - snd.cel[1] * snd.scl[1]) // 2
    rstr(r) = "$(numerator(r))ss$(denominator(r))"
    sizes = join((join(v.cel, "x") for v in (snd, med, rcv) if !isnothing(v)), "__")
    scale = allequal(snd.scl) ? rstr(snd.scl[1]) : join(rstr.(snd.scl), "x")
    ref = refinement(system)
    refinement_string = isnothing(ref) ? "" : "__refF$(ref.factor)T$(ref.thickness)"
    return "$(sizes)__$(rstr(gap))__at$(scale)__$(prod(volume_symbol2char.(design_regions(system))))$(refinement_string)"
end

end # module
