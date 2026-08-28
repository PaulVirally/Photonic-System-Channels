import GilaElectromagnetics: adjoint!, isadjoint, isselfoperator, isexternaloperator,
                             isgpu, useCpu!, useGpu!, glaSze, asym

# Gila's quadrature only reaches a relative error of 1e-8 once two volumes are
# `MIN_GAP_CELLS` cells of the coarser scale apart, and its operator constructors
# warn below that. A sweep walks the sender/receiver separation down to a single
# cell, so the near points need the two facing surfaces meshed finely enough that
# *every* pair of regions clears the bar, not just the two that face each other.
const MIN_GAP_CELLS = 6

# (factor, thickness) per gap `g` in coarse x-cells, for g = 1 through
# MIN_GAP_CELLS - 1. A body is meshed as a slab of `thickness` coarse cells at the
# gap face refined by `factor` in x, then two coarse cells, then the coarse bulk,
# which makes the three pair rules
#
#   fine   against fine     g * factor        >= MIN_GAP_CELLS
#   fine   against coarse   g + thickness     >= MIN_GAP_CELLS
#   coarse against coarse   g + 2 * thickness >= MIN_GAP_CELLS
#
# each measured in the coarser scale of the pair.
const GAP_REFINEMENT_TABLE = ((6, 6), (3, 4), (2, 4), (2, 2), (2, 2))

"""
    GapRefinement

The x-only refinement a sender/receiver pair needs at its gap.

# Fields
- `gap::Int`: The separation between the two bodies, in coarse x-cells
- `factor::Int`: The refinement applied along x at the two gap faces (1 for no
    refinement)
- `thickness::Int`: The thickness of the refined slab, in coarse x-cells
"""
struct GapRefinement
    gap::Int
    factor::Int
    thickness::Int
end

"""
    gap_refinement(gap_wl, x_scale) -> Union{Nothing, GapRefinement}

The refinement a pair of bodies separated by `gap_wl` needs. A gap off the grid is
rounded down and a gap under one cell is treated as one, so an off-grid separation
is refined at least as hard as the grid point below it.

# Arguments
- `gap_wl::Rational`: The separation between the two bodies, in wavelengths
- `x_scale::Rational`: The coarse cell size along x, in wavelengths

# Return
- A `GapRefinement`, or `nothing` at `MIN_GAP_CELLS` cells or more, where the
    operator is accurate as it stands, and at contact, which runs the exact
    quadrature
"""
function gap_refinement(gap_wl::Rational, x_scale::Rational)
    gap_wl > zero(gap_wl) || return nothing
    cells = gap_wl // x_scale
    cells >= MIN_GAP_CELLS && return nothing
    g = max(1, floor(Int, cells))
    factor, thickness = GAP_REFINEMENT_TABLE[g]
    return GapRefinement(g, factor, thickness)
end

# Outer corners of a volume, the way GlaCmpVol measures its edges.
_lwr_edge(vol::GlaVol) = Tuple(first.(vol.grd) .- (vol.scl .// 2))
_upr_edge(vol::GlaVol) = Tuple(last.(vol.grd) .+ (vol.scl .// 2))

# The box covering `ncel` coarse x-cells at one x face of `vol`, spanning the
# whole cross section so that `refine` carves in x alone.
function _face_box(vol::GlaVol, ncel::Integer, face::Symbol)
    lwr, upr = _lwr_edge(vol), _upr_edge(vol)
    xlo, xhi = face === :high ? (upr[1] - ncel * vol.scl[1], upr[1]) :
                                (lwr[1], lwr[1] + ncel * vol.scl[1])
    org = ((xlo + xhi) // 2, (lwr[2] + upr[2]) // 2, (lwr[3] + upr[3]) // 2)
    dim = (xhi - xlo, upr[2] - lwr[2], upr[3] - lwr[3])
    return (org, dim)
end

"""
    refine_body(vol, ref, face) -> GlaCmpVol

Mesh `vol` as the refined slab, contact layer and coarse bulk of `ref`. A body too
thin to hold the slab and the layer is refined whole.

Gila computes a cross-scale block whose regions touch by remeshing the whole
coarse region at the fine scale, so the two coarse cells between the slab and the
bulk are what keeps the sandwich from remeshing the entire body. The bulk sees the
slab across them through the ordinary partitioned quadrature.

# Arguments
- `vol::GlaVol`: The body to mesh, which must have an even number of cells in
    every dimension
- `ref::GapRefinement`: The refinement to apply
- `face::Symbol`: Which x face carries the slab, `:high` or `:low`

# Return
- A `GlaCmpVol` tiling of `vol`, in flat layout order
"""
function refine_body(vol::GlaVol, ref::GapRefinement, face::Symbol)
    face in (:high, :low) || throw(ArgumentError("A gap face is :high or :low, got :$(face)."))
    any(isodd.(vol.cel)) && throw(ArgumentError("A refined mesh needs an even number of cells in every dimension, and this body has $(vol.cel). Round the geometry to even cells, or drop --refine and accept the quadrature error of the unrefined gap."))
    ncel = Int(vol.cel[1])
    thickness = min(ref.thickness, ncel)
    cvol = GlaCmpVol(vol)
    # Even `ncel` and even `thickness` leave no room between the two cases: the
    # slab either takes the whole body or leaves at least the two-cell layer.
    thickness == ncel &&
        return refine(cvol, _face_box(vol, ncel, face); factor=(ref.factor, 1, 1))
    thickness + 2 < ncel &&
        (cvol = refine(cvol, _face_box(vol, thickness + 2, face); factor=1))
    return refine(cvol, _face_box(vol, thickness, face); factor=(ref.factor, 1, 1))
end

"""
    dof_length(vol) -> Int

Degrees of freedom a volume carries: three per cell, summed over the regions of a
composite volume.
"""
dof_length(vol::GlaVol) = 3 * prod(vol.cel)
dof_length(cvol::GlaCmpVol) = sum(3 * prod(reg.cel) for reg in regions(cvol))

"""
    mesh_tag(cvol, base) -> String

The cache-key fragment naming how `cvol` refines `base`, empty when it does not,
so that an unrefined body keys its Green blocks where a finished sweep's preload
directory already has them.

A refined body spells out its x-partition region by region in flat layout order as
`<x-cells>f<refinement factor>`, then which x face carries the finest region. The
two faces are mirror images and the Green operator is not mirror symmetric, so a
sender and a receiver of the same shape are different operators.
"""
function mesh_tag(cvol::GlaCmpVol, base::GlaVol)
    regs = regions(cvol)
    length(regs) == 1 && regs[1] == base && return ""
    body = join(("$(reg.cel[1])f$(Int(base.scl[1] // reg.scl[1]))" for reg in regs), "-")
    length(regs) == 1 && return "_ref$(body)"
    fine = regs[argmin([reg.scl[1] for reg in regs])]
    face = _upr_edge(fine)[1] == maximum(reg -> _upr_edge(reg)[1], regs) ? "h" : "l"
    return "_ref$(body)$(face)"
end

"""
    CmpBlkOprVac

The universe vacuum Green operator of a refined sender/receiver pair.

Gila's composite volume is one solid cuboid, so the two bodies cannot share one
tiling across the gap and the universe operator has to be assembled here as a
block matrix over the two meshes. The flat layout is the sender's degrees of
freedom followed by the receiver's, which is the convention `asym_ur` and the
bounds' sender projector are written against.

# Fields
- `blocks::Matrix{GlaCmpOprVac}`: The block matrix, `blocks[i, j]` mapping body
    `j` to body `i`, with the rows and the columns in flat layout order
"""
struct CmpBlkOprVac <: AbstractGlaVacOpr
    blocks::Matrix{GlaCmpOprVac}
end

"""
    CmpBlkOprVac(trgCvls, srcCvls; useGpu=false)

Build the block operator between two lists of meshes, one `GlaCmpOprVac` per
target/source pair.

# Arguments
- `trgCvls::AbstractVector{GlaCmpVol}`: The target meshes, in flat layout order
- `srcCvls::AbstractVector{GlaCmpVol}`: The source meshes, in flat layout order
- `useGpu::Bool=false`: Whether to build the blocks on the GPU
"""
CmpBlkOprVac(trgCvls::AbstractVector{GlaCmpVol}, srcCvls::AbstractVector{GlaCmpVol};
             useGpu::Bool=false) =
    CmpBlkOprVac([GlaCmpOprVac(trg, src; useGpu=useGpu) for trg in trgCvls, src in srcCvls])

_row_sizes(opr::CmpBlkOprVac) = [size(opr.blocks[i, 1], 1) for i in axes(opr.blocks, 1)]
_col_sizes(opr::CmpBlkOprVac) = [size(opr.blocks[1, j], 2) for j in axes(opr.blocks, 2)]

Base.size(opr::CmpBlkOprVac) = (sum(_row_sizes(opr)), sum(_col_sizes(opr)))
Base.size(opr::CmpBlkOprVac, dim::Int) = size(opr)[dim]
glaSze(opr::CmpBlkOprVac) = glaSze.(opr.blocks)
glaSze(opr::CmpBlkOprVac, dim::Int) = map(sze -> sze[dim], glaSze(opr))

function Base.:*(opr::CmpBlkOprVac, innVec::AbstractVector{ComplexF64})
    rowOff = cumsum([0; _row_sizes(opr)])
    colOff = cumsum([0; _col_sizes(opr)])
    length(innVec) == colOff[end] || throw(ArgumentError("An input of length $(length(innVec)) does not fit this operator, which has $(colOff[end]) source degrees of freedom."))
    outVec = fill!(similar(innVec, rowOff[end]), zero(eltype(innVec)))
    for i in axes(opr.blocks, 1)
        outBlk = view(outVec, (rowOff[i] + 1):rowOff[i + 1])
        for j in axes(opr.blocks, 2)
            outBlk .+= opr.blocks[i, j] * view(innVec, (colOff[j] + 1):colOff[j + 1])
        end
    end
    return outVec
end

function adjoint!(opr::CmpBlkOprVac)
    adjMat = Matrix{GlaCmpOprVac}(undef, reverse(size(opr.blocks)))
    for i in axes(opr.blocks, 1), j in axes(opr.blocks, 2)
        adjMat[j, i] = adjoint!(opr.blocks[i, j])
    end
    return CmpBlkOprVac(adjMat)
end

function useCpu!(opr::CmpBlkOprVac)
    useCpu!.(opr.blocks)
    return opr
end

function useGpu!(opr::CmpBlkOprVac)
    useGpu!.(opr.blocks)
    return opr
end

GilaElectromagnetics.GilaVacuum.arrTyp(opr::CmpBlkOprVac) = GilaElectromagnetics.GilaVacuum.arrTyp(first(opr.blocks))
isadjoint(opr::CmpBlkOprVac) = all(isadjoint, opr.blocks)
isselfoperator(opr::CmpBlkOprVac) =
    ==(size(opr.blocks)...) && all(isselfoperator(opr.blocks[i, i]) for i in axes(opr.blocks, 1))
isexternaloperator(opr::CmpBlkOprVac) = !isselfoperator(opr)
isgpu(opr::CmpBlkOprVac) = all(isgpu, opr.blocks)

function Base.show(io::IO, opr::CmpBlkOprVac)
    numTrg, numSrc = size(opr.blocks)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "block composite G₀ ($numTrg target bod", numTrg == 1 ? "y" : "ies",
        " × $numSrc source bod", numSrc == 1 ? "y" : "ies", ")")
    print(io, "\n  $(size(opr, 1)) × $(size(opr, 2)) degrees of freedom")
    for blk in opr.blocks[1, :]
        print(io, "\n  ", blk.srcCvl)
    end
end
Base.show(io::IO, ::MIME"text/plain", opr::CmpBlkOprVac) = show(io, opr)

"""
    AsyCmpBlkOprVac

The anti-Hermitian part of a refined pair's universe vacuum Green operator.

# Fields
- `opr::CmpBlkOprVac`: The same block assembly with every block's Fourier
    coefficients folded, so that applying it is applying `Asym(G⁰ᵤᵤ)`
"""
struct AsyCmpBlkOprVac <: AbstractGlaVacOpr
    opr::CmpBlkOprVac
end

# One block of the assembly, folded. Gila's `_hrmOpr` does this and then refuses
# anything but a self operator, because a fold returns the entrywise imaginary
# part, which is the antisymmetrization only where the operator is complex
# symmetric. That premise holds for the assembly as a whole rather than for the
# off diagonal blocks one at a time, so `asym` below carries the check and this
# folds whatever it is handed.
function _fold_block(blk::GlaCmpOprVac)
    blkMat = Matrix{AbstractGlaOpr}(undef, size(blk.blkMat))
    for idx in eachindex(blk.blkMat)
        blkMat[idx] = GilaElectromagnetics.GilaOperators._hrmBlk(blk.blkMat[idx], true)
    end
    return GlaCmpOprVac(blk.trgCvl, blk.srcCvl, blkMat)
end

"""
    asym(opr::CmpBlkOprVac) -> AsyCmpBlkOprVac

`Asym(G⁰ᵤᵤ)` of a refined pair, folded into the Fourier coefficients.

Folding costs one Green apply where `(X - X')/2im` costs two, and Gila offers it
for a self operator alone: the fold returns the entrywise imaginary part of a
block, which is the antisymmetrization only for a complex symmetric block. Every
block of the universe folds even so. Reciprocity makes `G⁰ᵣₛ` the transpose of
`G⁰ₛᵣ`, so the whole assembly is complex symmetric, and the (sender, receiver)
block of `(U - U')/2im`, namely `(G⁰ₛᵣ - G⁰ᵣₛ')/2im`, is the entrywise imaginary
part of `G⁰ₛᵣ` even though `G⁰ₛᵣ` is not square, let alone symmetric.

The two forms differ by the quadrature's complex symmetry defect, and the folded
one is the better of the two: it inherits the positive semidefiniteness of the
continuum `Im G⁰` rather than picking up that defect as a negative eigenvalue,
which is exactly what the bounds' pencil whitener is sensitive to.

The fold is a copy of the Fourier data, as it is upstream, so an assembly and its
`asym` held at once cost twice one assembly's Green memory. The difference form
costs nothing extra and twice the time; this trades the one for the other.

# Arguments
- `opr::CmpBlkOprVac`: The universe operator, which has to be a self operator and
    not in adjoint mode

# Return
- An `AsyCmpBlkOprVac`, Hermitian, costing one Green apply per block
"""
function asym(opr::CmpBlkOprVac)
    isselfoperator(opr) || throw(ArgumentError("AsyCmpBlkOprVac can only be built from a block assembly whose target and source bodies are the same, and this one maps $(size(opr.blocks, 2)) source bodies to $(size(opr.blocks, 1)) target bodies."))
    isadjoint(opr) && throw(ArgumentError("AsyCmpBlkOprVac can only be built from an assembly that is not in adjoint mode. Call adjoint! on it to restore it first."))
    return AsyCmpBlkOprVac(CmpBlkOprVac(_fold_block.(opr.blocks)))
end

Base.size(opr::AsyCmpBlkOprVac) = size(opr.opr)
Base.size(opr::AsyCmpBlkOprVac, dim::Int) = size(opr.opr, dim)
glaSze(opr::AsyCmpBlkOprVac) = glaSze(opr.opr)
glaSze(opr::AsyCmpBlkOprVac, dim::Int) = glaSze(opr.opr, dim)
Base.:*(opr::AsyCmpBlkOprVac, innVec::AbstractVector{ComplexF64}) = opr.opr * innVec
GilaElectromagnetics.GilaVacuum.arrTyp(opr::AsyCmpBlkOprVac) =
    GilaElectromagnetics.GilaVacuum.arrTyp(opr.opr)
# Hermitian, so the adjoint is the operator itself.
adjoint!(opr::AsyCmpBlkOprVac) = opr
isadjoint(::AsyCmpBlkOprVac) = false
isselfoperator(::AsyCmpBlkOprVac) = true
isexternaloperator(::AsyCmpBlkOprVac) = false
isgpu(opr::AsyCmpBlkOprVac) = isgpu(opr.opr)
useCpu!(opr::AsyCmpBlkOprVac) = (useCpu!(opr.opr); opr)
useGpu!(opr::AsyCmpBlkOprVac) = (useGpu!(opr.opr); opr)

function Base.show(io::IO, opr::AsyCmpBlkOprVac)
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "block composite Asym(G₀) ($(size(opr.opr.blocks, 1)) bodies)")
    print(io, "\n  $(size(opr, 1)) × $(size(opr, 2)) degrees of freedom")
end
Base.show(io::IO, ::MIME"text/plain", opr::AsyCmpBlkOprVac) = show(io, opr)
