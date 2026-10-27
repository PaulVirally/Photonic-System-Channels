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
    any(isodd.(vol.cel)) && throw(ArgumentError("A refined mesh needs an even number of cells in every dimension, and this body has $(vol.cel). Round the geometry to even cells, or pass --no-refine and accept the quadrature error of the unrefined gap."))
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
