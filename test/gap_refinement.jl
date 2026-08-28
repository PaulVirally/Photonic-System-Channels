## Gap refinement: the derivation, the cache keys it changes, and the two things
## the refined operators have to get right numerically.
##
## Run from the repo root:
##     julia --project=. test/gap_refinement.jl
##
## Gila's quadrature reaches 1e-8 only once two volumes are six cells of the
## coarser scale apart, and a sweep walks the sender/receiver gap down to one
## cell. `gap_refinement` answers that by refining the two facing surfaces, and
## the bar it has to clear is not just the two regions facing each other: *every*
## pair of regions across the gap has to be six of its own coarser cells apart.
## The first half of this file checks that on the real production geometries, by
## measuring the gaps the same way Gila's proximity check does rather than by
## restating the table.
##
## The second half is about the preload directory. Ninety-five far points of the
## current sweeps are already computed, and their Green blocks are keyed by
## geometry alone. A point that is not refined has to key exactly where it always
## did, or every one of those files becomes a miss.
##
## No GPU needed. Everything lands in a mktempdir. Exits nonzero on failure.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using GilaElectromagnetics
using LinearAlgebra
using LinearMaps
using Printf

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-72s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

const CHI = 13.6 + 0.05im
const DESIGN = SMRVolumeSymbol[Sender, Receiver]

# Refinement is the default, and every system here says so anyway: `refine` is the
# knob the checks turn, and the control for a check that wants it off says so on
# the same line.
system(cells, sep, scale; refine=true, receiver_cells=cells) =
    SMRSystem(cells, (sep, 0//1, 0//1), receiver_cells, DESIGN, scale, CHI;
              refine_gap=refine)

# The five production geometries, as (cells, scale) with `SMRSystem`'s convention
# that a negative scale means anisotropic cells of (1//32, |scale|, |scale|).
const GEOMETRIES = (((8, 8, 8), 1//32), ((16, 16, 16), 1//32), ((32, 32, 32), 1//32),
                    ((64, 32, 32), -1//16), ((128, 32, 32), -1//8))

lwr_edge(vol::GlaVol) = Tuple(first.(vol.grd) .- (vol.scl .// 2))
upr_edge(vol::GlaVol) = Tuple(last.(vol.grd) .+ (vol.scl .// 2))

# The separation of two regions in cells of the coarser scale of the pair, which
# is the number Gila's `prxChk` compares against six. `nothing` when they touch.
function pair_gap_cells(regA::GlaVol, regB::GlaVol)
    sep = max.(lwr_edge(regB) .- upr_edge(regA), lwr_edge(regA) .- upr_edge(regB))
    all(sep .<= 0) && return nothing
    coarse = max.(regA.scl, regB.scl)
    dir = argmax(idx -> max(sep[idx], 0//1) // coarse[idx], 1:3)
    return sep[dir] // coarse[dir]
end

# The tightest pair across the gap, over every region of one mesh against every
# region of the other.
function min_cross_gap(smr::SMRSystem)
    gaps = [pair_gap_cells(a, b) for a in regions(sender_mesh(smr))
                                 for b in regions(receiver_mesh(smr))]
    return minimum(filter(!isnothing, gaps))
end

println("=== the table satisfies the three pair rules it was built from")
for g in 1:(MIN_GAP_CELLS - 1)
    factor, thickness = PSC.SMRSystems.GAP_REFINEMENT_TABLE[g]
    check("g = $g: fine against fine, g * factor = $(g * factor) ≥ $MIN_GAP_CELLS",
          g * factor >= MIN_GAP_CELLS)
    check("g = $g: fine against coarse, g + t = $(g + thickness) ≥ $MIN_GAP_CELLS",
          g + thickness >= MIN_GAP_CELLS)
    check("g = $g: coarse against coarse, g + 2t = $(g + 2thickness) ≥ $MIN_GAP_CELLS",
          g + 2thickness >= MIN_GAP_CELLS)
    check("g = $g: the slab thickness is even", iseven(thickness))
end

println("\n=== every region pair of every production geometry clears the bar")
for (cells, scale) in GEOMETRIES, g in 1:(MIN_GAP_CELLS - 1)
    smr = system(cells, g//32, scale)
    ref = refinement(smr)
    gap = min_cross_gap(smr)
    check("$(cells) at g = $g: closest region pair is $(gap) cells of its coarser scale",
          gap >= MIN_GAP_CELLS, "factor $(ref.factor), thickness $(ref.thickness)")
end

println("\n=== the refined mesh is slab, contact layer and bulk, in that order")
for (cells, scale) in GEOMETRIES, g in 1:(MIN_GAP_CELLS - 1)
    smr = system(cells, g//32, scale)
    ref = refinement(smr)
    regs = regions(sender_mesh(smr))
    coarse_x = sender(smr).scl[1]
    thickness = min(ref.thickness, cells[1])
    want = thickness == cells[1] ? [(thickness * ref.factor, ref.factor)] :
           thickness + 2 == cells[1] ? [(thickness * ref.factor, ref.factor), (2, 1)] :
           [(thickness * ref.factor, ref.factor), (2, 1), (cells[1] - thickness - 2, 1)]
    got = [(Int(reg.cel[1]), Int(coarse_x // reg.scl[1])) for reg in regs]
    check("$(cells) at g = $g: sender regions are $(want)", got == want, string(got))
    # The slab sits on the face each body turns toward the gap.
    check("$(cells) at g = $g: the sender slab is on its high-x face",
          upr_edge(regs[1])[1] == upr_edge(sender(smr))[1])
    check("$(cells) at g = $g: the receiver slab is on its low-x face",
          lwr_edge(regions(receiver_mesh(smr))[1])[1] == lwr_edge(receiver(smr))[1])
end

println("\n=== small bodies clamp instead of overrunning")
# 1/4 λ at 1/32 is eight x-cells, so the six-cell slab and the two-cell layer
# exactly fill it and there is no bulk left.
quarter = system((8, 8, 8), 1//32, 1//32)
check("(8,8,8) at g = 1: slab plus layer, no bulk", nregions(sender_mesh(quarter)) == 2,
      string([Tuple(r.cel) for r in regions(sender_mesh(quarter))]))
# Four x-cells cannot hold the slab at all, so the whole body is refined.
tiny = system((4, 4, 4), 1//32, 1//32)
check("(4,4,4) at g = 1: the whole body is refined", nregions(sender_mesh(tiny)) == 1 &&
      regions(sender_mesh(tiny))[1].cel == (24, 4, 4),
      string(Tuple(regions(sender_mesh(tiny))[1].cel)))
check("(4,4,4) at g = 1: the pair still clears the bar", min_cross_gap(tiny) >= MIN_GAP_CELLS,
      string(min_cross_gap(tiny)))
smallest = system((2, 2, 2), 1//32, 1//32)
check("(2,2,2) at g = 1: the whole body is refined",
      regions(sender_mesh(smallest))[1].cel == (12, 2, 2),
      string(Tuple(regions(sender_mesh(smallest))[1].cel)))

println("\n=== a gap of six cells or more is left alone")
check("gap_refinement returns nothing at six cells",
      gap_refinement(6//32, 1//32) === nothing)
check("gap_refinement returns nothing at a wide gap",
      gap_refinement(1//1, 1//32) === nothing)
check("gap_refinement returns nothing for touching bodies",
      gap_refinement(0//1, 1//32) === nothing)
check("gap_refinement rounds an off-grid gap down",
      gap_refinement(5//64, 1//32) == GapRefinement(2, 3, 4),
      string(gap_refinement(5//64, 1//32)))
check("gap_refinement treats a sub-cell gap as one cell",
      gap_refinement(1//64, 1//32) == GapRefinement(1, 6, 6),
      string(gap_refinement(1//64, 1//32)))
for (cells, scale) in GEOMETRIES
    far = system(cells, 6//32, scale)
    check("$(cells) at g = 6: the system is not refined", !is_refined(far))
    check("$(cells) at g = 6: the sender mesh is the plain volume",
          sender_mesh(far) == GlaCmpVol(sender(far)))
end

println("\n=== refine_gap=false reproduces the unrefined system exactly")
for (cells, scale) in GEOMETRIES, g in 1:(MIN_GAP_CELLS - 1)
    off = system(cells, g//32, scale; refine=false)
    check("$(cells) at g = $g with refine_gap=false is unrefined", !is_refined(off))
end

println("\n=== unrefined cache keys have not moved")
# Hard-coded rather than re-derived: this is the check that a preload directory
# full of far points stays valid, so the expected strings have to come from
# outside the code that generates them.
const FAR = system((8, 8, 8), 1//4, 1//32)
check("unrefined (R,R) key", PSC.SMRSystems.green_fname(FAR, Receiver, Receiver) ==
      "self/8x8x8_1ss32x1ss32x1ss32.glaG0",
      PSC.SMRSystems.green_fname(FAR, Receiver, Receiver))
check("unrefined (R,S) key", PSC.SMRSystems.green_fname(FAR, Receiver, Sender) ==
      "ext/8x8x8_1ss32x1ss32x1ss32@0ss1x0ss1x0ss1_to_8x8x8_1ss32x1ss32x1ss32@1ss2x0ss1x0ss1.glaG0",
      PSC.SMRSystems.green_fname(FAR, Receiver, Sender))
check("unrefined universe key", PSC.SMRSystems.green_fname(FAR, Design, Design) ==
      "self/24x8x8_1ss32x1ss32x1ss32.glaG0",
      PSC.SMRSystems.green_fname(FAR, Design, Design))
check("unrefined scratch key", file_prefix(FAR) == "8x8x8__8x8x8__1ss4__SR",
      file_prefix(FAR))
# The two-argument spelling is what every pre-refinement caller used.
check("the two-argument green_fname is unchanged",
      PSC.SMRSystems.green_fname(receiver(FAR), sender(FAR)) ==
      PSC.SMRSystems.green_fname(FAR, Receiver, Sender))

const ANISO_FAR = system((64, 32, 32), 1//4, -1//16)
check("unrefined anisotropic (R,R) key",
      PSC.SMRSystems.green_fname(ANISO_FAR, Receiver, Receiver) ==
      "self/64x32x32_1ss32x1ss16x1ss16.glaG0",
      PSC.SMRSystems.green_fname(ANISO_FAR, Receiver, Receiver))
check("unrefined anisotropic scratch key",
      file_prefix(ANISO_FAR) == "64x32x32__64x32x32__1ss4__SR", file_prefix(ANISO_FAR))

println("\n=== a shared cache key means a shared operator, and nothing else")
# Two gaps that land on the same table entry give a body the same mesh, and a self
# operator depends on the mesh alone, so g = 4 and g = 5 *should* share the (R,R)
# file. What must never happen is two different operators sharing a key, so the
# check is that a key determines the geometry it was built from rather than that
# keys are all distinct.
mesh_shape(cvol::GlaCmpVol, base::GlaVol) =
    [(Tuple(reg.cel), Tuple(reg.scl), Tuple(reg.org .- base.org)) for reg in regions(cvol)]

function op_signature(smr::SMRSystem, target::SMRVolumeSymbol, source::SMRVolumeSymbol)
    target == Design && return (:universe, mesh_shape(sender_mesh(smr), sender(smr)),
                                mesh_shape(receiver_mesh(smr), receiver(smr)),
                                receiver(smr).org .- sender(smr).org)
    trg, src = volume(smr, target), volume(smr, source)
    return (:pair, mesh_shape(mesh(smr, target), trg), mesh_shape(mesh(smr, source), src),
            trg.org .- src.org)
end

let keys = Dict{String,Any}(), prefixes = Dict{String,Any}(), clashes = String[]
    for (cells, scale) in GEOMETRIES, g in 0:(MIN_GAP_CELLS - 1)
        # g = 0 stands for the unrefined far point of the same geometry.
        smr = system(cells, g == 0 ? 1//4 : g//32, scale)
        label = "$(cells) g=$g"
        for (target, source) in ((Sender, Sender), (Receiver, Receiver),
                                 (Receiver, Sender), (Design, Design))
            key = PSC.SMRSystems.green_fname(smr, target, source)
            signature = op_signature(smr, target, source)
            if haskey(keys, key) && keys[key][2] != signature
                push!(clashes, "$(label) and $(keys[key][1]) are different operators under $key")
            end
            keys[key] = (label, signature)
        end
        prefix = file_prefix(smr)
        signature = (op_signature(smr, Design, Design), refinement(smr))
        if haskey(prefixes, prefix) && prefixes[prefix][2] != signature
            push!(clashes, "$(label) and $(prefixes[prefix][1]) share the scratch key $prefix")
        end
        prefixes[prefix] = (label, signature)
    end
    check("no cache key covers two different operators", isempty(clashes),
          isempty(clashes) ? "$(length(keys)) keys, $(length(prefixes)) scratch keys" : first(clashes))
end

const NEAR = system((8, 8, 8), 1//32, 1//32)
check("a refined (R,R) key is tagged",
      PSC.SMRSystems.green_fname(NEAR, Receiver, Receiver) ==
      "self/8x8x8_1ss32x1ss32x1ss32_ref36f6-2f1l.glaG0",
      PSC.SMRSystems.green_fname(NEAR, Receiver, Receiver))
# The sender turns its high-x face to the gap and the receiver its low-x one, so
# the two meshes are mirror images. The Green operator is not mirror symmetric,
# so they must not share a key.
check("the sender and receiver self keys differ",
      PSC.SMRSystems.green_fname(NEAR, Sender, Sender) !=
      PSC.SMRSystems.green_fname(NEAR, Receiver, Receiver),
      PSC.SMRSystems.green_fname(NEAR, Sender, Sender))
check("a refined scratch key is tagged",
      file_prefix(NEAR) == "8x8x8__8x8x8__1ss32__SR__refF6T6", file_prefix(NEAR))

println("\n=== the refined operators are the operators they claim to be")
const ROOT = mktempdir(; cleanup=true)
const ENVIRONMENT = ComputeEnvironment(joinpath(ROOT, "preload"), joinpath(ROOT, "project"),
                                       joinpath(ROOT, "scratch"), GPUChoice(false, -1))
for dir in (preload_dir(ENVIRONMENT), project_dir(ENVIRONMENT), scratch_dir(ENVIRONMENT))
    mkpath(dir)
end

# (2,2,2) at one cell of gap refines both bodies whole by six in x, so N_r = 144.
const TINY = system((2, 2, 2), 1//32, 1//32)
const G_RS = load_green_function(ENVIRONMENT, TINY, Receiver, Sender)
check("the refined (R,S) block is a composite operator", G_RS isa GlaCmpOprVac,
      string(typeof(G_RS)))
check("the refined (R,S) block is sized by the meshes",
      size(G_RS) == (dof_length(receiver_mesh(TINY)), dof_length(sender_mesh(TINY))),
      string(size(G_RS)))

const M_RS = Matrix(G_RS[:, :])
check("the densified block is finite", all(isfinite, M_RS))
const M_RS_ADJ = Matrix(adjoint(G_RS)[:, :])
let d = norm(M_RS_ADJ - M_RS') / norm(M_RS)
    check("adjoint(G⁰ᵣₛ) densifies to the dense adjoint", d < 1e-12,
          @sprintf("rel diff = %.3e", d))
end
# Applying the operator and reading its columns are two different code paths.
let x = ComplexF64.(1:size(G_RS, 2)) ./ size(G_RS, 2),
    d = norm(G_RS * x - M_RS * x) / norm(M_RS * x)
    check("the matrix-vector product agrees with the densified block", d < 1e-12,
          @sprintf("rel diff = %.3e", d))
end

println("\n--- at six cells of gap the refinement is a no-op, spectrum and all")
const FAR_ON = system((2, 2, 2), 6//32, 1//32)
const FAR_OFF = system((2, 2, 2), 6//32, 1//32; refine=false)
check("the two systems have the same sender mesh", sender_mesh(FAR_ON) == sender_mesh(FAR_OFF))
check("the two systems have the same cache keys",
      PSC.SMRSystems.green_fname(FAR_ON, Receiver, Sender) ==
      PSC.SMRSystems.green_fname(FAR_OFF, Receiver, Sender))

function ur_spectrum(smr)
    A, _ = PSC.asym_ur(load_green_function(ENVIRONMENT, smr, Receiver, Sender),
                       load_green_function(ENVIRONMENT, smr, Receiver, Receiver), smr)
    n = size(A, 1)
    M = zeros(ComplexF64, n, n)
    e = zeros(ComplexF64, n)
    for j in 1:n
        e[j] = 1
        M[:, j] .= A * e
        e[j] = 0
    end
    return sort(eigvals(Hermitian(M)); rev=true)
end

let on = ur_spectrum(FAR_ON), off = ur_spectrum(FAR_OFF),
    d = maximum(abs.(on .- off)) / maximum(abs.(off))
    check("Asym(G⁰ᵤᵣ) spectra agree with refinement on and off", d < 1e-12,
          @sprintf("max rel diff = %.3e", d))
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
