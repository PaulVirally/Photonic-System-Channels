## Narval pre-flight 3: does a Green operator written to the preload directory
## still work when a *different* process reads it back?
##
## Run from the repo root:
##     julia --project=. test/preload_roundtrip.jl
##
## Why a fresh process. Gila's custom Serialization methods are written against
## `io::IO`:
##
##     Serialization.serialize(io::IO, opr::GlaOprVac) = serialize(io, opr.mem)
##     Serialization.serialize(io::IO, opr::MulRegGlaOprVac) = serialize(io, opr.oprMat)
##
## The single-region path therefore round-trips through GlaVacOprMem's own
## serializer, which stores only egoFur plus the volumes and rebuilds the FFTW
## plans with glaOprPrp on load. The multi-region path hands a
## `Matrix{GlaOprVac}` to the generic array serializer, and array elements are
## written through an AbstractSerializer, not an IO, so the `io::IO` methods
## never fire. Each GlaOprVac is then serialized field by field, FFTW plan
## pointers included. Inside one process those raw pointers can still happen to
## address a live plan, so an in-process reload looks fine. A fresh process is
## the only place the breakage shows.
##
## The control is a single-region (R,R) round trip, which is expected to pass.
##
## The refined half of the file asks the same question of the composite operators
## a near point is built from. Those go to disk through the generic serializer as
## well (Gila has no typed deserializer for a composite operator yet) so they
## have exactly as much to lose from a stale plan pointer as the multi-region one
## did, and an in-process reload would hide it just as well.
##
## No GPU needed. Everything lands in a mktempdir. Exits nonzero on failure.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using LinearAlgebra
using LinearMaps
using JLD2
using Printf
using Random
using Serialization
using GilaElectromagnetics

const REPO = dirname(@__DIR__)
const ROOT = mktempdir(; cleanup=true)
const PRELOAD = joinpath(ROOT, "preload")
mkpath(PRELOAD)
mkpath(joinpath(ROOT, "project"))
mkpath(joinpath(ROOT, "scratch"))
println("workspace: ", ROOT)

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-58s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

const ENVIRONMENT = ComputeEnvironment(PRELOAD, joinpath(ROOT, "project"),
                                       joinpath(ROOT, "scratch"), GPUChoice(false, -1))

# The smallest system the pipeline actually uses: (4,4,4) sender and receiver at
# 1/32 λ per cell, 1/32 λ apart. N_r = 3*64 = 192, N_u = 3*(64+64) = 384.
#
# The refined system is (8,2,2) at four cells of gap, whose bodies are wide enough
# in x to hold the whole three-region mesh (refined slab, contact layer, coarse
# bulk) and thin enough in y and z that the six self operators it needs are
# quick. That is the case worth writing to disk: a body refined whole is one plain
# region, while this one carries the cross-scale contact blocks, which are the
# ones built on a remeshed volume of their own.
const SMR = SMRSystem((4, 4, 4), (1//32, 0//1, 0//1), (4, 4, 4),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im;
                      refine_gap=false)
const SMR_REF = SMRSystem((8, 2, 2), (1//8, 0//1, 0//1), (8, 2, 2),
                          SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im;
                          refine_gap=true)
const N_R = dof_length(receiver_mesh(SMR))
const N_U = dof_length(sender_mesh(SMR)) + dof_length(receiver_mesh(SMR))
const N_R_REF = dof_length(receiver_mesh(SMR_REF))
const N_U_REF = dof_length(sender_mesh(SMR_REF)) + dof_length(receiver_mesh(SMR_REF))

Random.seed!(20260814)
const V_SINGLE = randn(ComplexF64, N_R)
const V_MULTI = randn(ComplexF64, N_U)
const V_SINGLE_REF = randn(ComplexF64, N_R_REF)
const V_MULTI_REF = randn(ComplexF64, N_U_REF)

# The child script. It is written into the workspace rather than kept in test/,
# since it is meaningless without the workspace it is handed.
const CHILD = joinpath(ROOT, "child.jl")
write(CHILD, """
## Spawned by test/preload_roundtrip.jl. Loads one Green operator out of a
## preload directory that a previous process wrote, applies it to a vector
## handed over through JLD2, and writes the result back.
using PhotonicSystemChannels
const PSC = PhotonicSystemChannels
using LinearAlgebra
using LinearMaps
using JLD2

root, tag = ARGS[1], ARGS[2]
env = ComputeEnvironment(joinpath(root, "preload"), joinpath(root, "project"),
                         joinpath(root, "scratch"), GPUChoice(false, -1))
refined = endswith(tag, "_refined")
smr = refined ? SMRSystem((8, 2, 2), (1//8, 0//1, 0//1), (8, 2, 2),
                          SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im;
                          refine_gap=true) :
                SMRSystem((4, 4, 4), (1//32, 0//1, 0//1), (4, 4, 4),
                          SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im;
                          refine_gap=false)

v = load(joinpath(root, "vectors.jld2"), "v_\$(tag)")

println("[child \$tag] loading the operator from the preload directory")
flush(stdout)
G = startswith(tag, "single") ? load_green_function(env, smr, Receiver, Receiver) :
                                load_green_function(env, smr, [Sender, Receiver], [Sender, Receiver])
println("[child \$tag] applying it")
flush(stdout)
y = PSC.asym(LinearMap(G)) * v
println("[child \$tag] norm(y) = ", norm(y))
flush(stdout)
jldsave(joinpath(root, "child_\$(tag).jld2"); y=y)
println("[child \$tag] wrote the result")
""")

"""
    run_child(tag) -> (status, y)

Run the child process for `tag` and read back its vector. `status` is `:ok`,
`:crashed` (the process died, which is what the multi-region bug looks like), or
`:no_output` (it exited cleanly but wrote nothing).
"""
function run_child(tag)
    out = joinpath(ROOT, "child_$(tag).jld2")
    rm(out; force=true)
    cmd = `$(Base.julia_cmd()) --project=$(REPO) $(CHILD) $(ROOT) $(tag)`
    try
        run(cmd)
    catch err
        err isa ProcessFailedException || rethrow(err)
        println("child ($tag) died: ", err)
        return (:crashed, nothing)
    end
    isfile(out) || return (:no_output, nothing)
    return (:ok, load(out, "y"))
end

println("\n=== parent: building the operators and applying them")
G_rr = load_green_function(ENVIRONMENT, SMR, Receiver, Receiver)
y_single = PSC.asym(LinearMap(G_rr)) * V_SINGLE
@printf("single-region (R,R): N_r = %d, norm(y) = %.10e\n", N_R, norm(y_single))

G_uu = load_green_function(ENVIRONMENT, SMR, [Sender, Receiver], [Sender, Receiver])
y_multi = PSC.asym(LinearMap(G_uu)) * V_MULTI
@printf("multi-region [S,R]<-[S,R]: N_u = %d, norm(y) = %.10e\n", N_U, norm(y_multi))

check("the refined bodies are meshed in three regions",
      nregions(receiver_mesh(SMR_REF)) == 3,
      string([Tuple(r.cel) for r in regions(receiver_mesh(SMR_REF))]))
G_rr_ref = load_green_function(ENVIRONMENT, SMR_REF, Receiver, Receiver)
check("the refined (R,R) block is a composite operator",
      G_rr_ref isa GlaCmpOprVac, string(typeof(G_rr_ref)))
check("its cross-scale contact blocks are built on a remeshed volume",
      any(blk -> blk isa GilaElectromagnetics.GilaOperators.GlaSndOprVac, G_rr_ref.blkMat),
      string(count(blk -> blk isa GilaElectromagnetics.GilaOperators.GlaSndOprVac, G_rr_ref.blkMat)) * " of " * string(length(G_rr_ref.blkMat)))
y_single_ref = PSC.asym(LinearMap(G_rr_ref)) * V_SINGLE_REF
@printf("refined (R,R): N_r = %d, norm(y) = %.10e\n", N_R_REF, norm(y_single_ref))

G_uu_ref = load_green_function(ENVIRONMENT, SMR_REF, [Sender, Receiver], [Sender, Receiver])
check("the refined universe is a block composite operator",
      G_uu_ref isa CmpBlkOprVac, string(typeof(G_uu_ref)))
check("the refined universe is [sender; receiver]", size(G_uu_ref) == (N_U_REF, N_U_REF),
      string(size(G_uu_ref)))
y_multi_ref = PSC.asym(LinearMap(G_uu_ref)) * V_MULTI_REF
@printf("refined [S,R]<-[S,R]: N_u = %d, norm(y) = %.10e\n", N_U_REF, norm(y_multi_ref))

jldsave(joinpath(ROOT, "vectors.jld2");
        v_single=V_SINGLE, v_multi=V_MULTI,
        v_single_refined=V_SINGLE_REF, v_multi_refined=V_MULTI_REF)

# Drop the parent's own operators before spawning. It changes nothing for the
# child, but it keeps the point clear: the child gets the file, not the objects.
G_rr = nothing
G_uu = nothing
G_rr_ref = nothing
G_uu_ref = nothing
GC.gc()

for (dir, _, files) in walkdir(PRELOAD), f in sort(files)
    path = joinpath(dir, f)
    println("preload: ", relpath(path, PRELOAD), "  (", filesize(path), " B)")
end

println("\n=== control: single-region (R,R) in a fresh process")
status_single, y_single_child = run_child("single")
if status_single == :ok
    d = norm(y_single_child - y_single) / norm(y_single)
    check("control: (R,R) survives a fresh-process reload", d < 1e-12,
          @sprintf("rel diff = %.3e", d))
else
    check("control: (R,R) survives a fresh-process reload", false, string(status_single))
    println("""
    The single-region control failed too. That is NOT the multi-region bug: it
    means the GlaVacOprMem serializer itself is broken here, so nothing in the
    preload directory can be trusted. Fix this before anything else.""")
end

println("\n=== the real check: multi-region [S,R]<-[S,R] in a fresh process")
status_multi, y_multi_child = run_child("multi")
if status_multi == :ok
    d = norm(y_multi_child - y_multi) / norm(y_multi)
    check("multi-region survives a fresh-process reload", d < 1e-12,
          @sprintf("rel diff = %.3e", d))
    if d >= 1e-12
        println("""
    The child ran but disagreed with the parent. The deserialized multi-region
    operator is a different operator, which is the same bug as a crash, only
    quieter: a sweep would produce numbers instead of failing.""")
    end
else
    check("multi-region survives a fresh-process reload", false, string(status_multi))
    println("""
    KNOWN BUG: the multi-region ([Sender,Receiver] <- [Sender,Receiver]) Green
    operator does not survive serialization. Gila serializes MulRegGlaOprVac by
    handing its Matrix{GlaOprVac} to the generic array serializer, which walks
    each GlaOprVac field by field instead of going through the `io::IO` method
    that rebuilds the FFTW plans. The plan pointers written to disk are only
    valid inside the writing process, so the child either dies or computes
    garbage.

    This must be fixed before any narval run reads a universe operator off disk.
    Everything that calls load_green_function(env, smr, [Sender,Receiver],
    [Sender,Receiver]) is affected: src/bounds.jl (_compute_bounds_sr) and
    src/verify_bounds.jl. Under GLOST it is worse, since the greens stage exists
    precisely to write these files for later jobs to read.

    The fix is on the Gila side: give MulRegGlaOprVac a serializer that writes
    each block through GlaVacOprMem's own serializer (and a matching
    deserializer), the way the single-region path already does.""")
end

println("\n=== refined: composite (R,R) in a fresh process")
status_single_ref, y_single_ref_child = run_child("single_refined")
if status_single_ref == :ok
    d = norm(y_single_ref_child - y_single_ref) / norm(y_single_ref)
    check("refined (R,R) survives a fresh-process reload", d < 1e-12,
          @sprintf("rel diff = %.3e", d))
else
    check("refined (R,R) survives a fresh-process reload", false, string(status_single_ref))
end

println("\n=== refined: block composite [S,R]<-[S,R] in a fresh process")
status_multi_ref, y_multi_ref_child = run_child("multi_refined")
if status_multi_ref == :ok
    d = norm(y_multi_ref_child - y_multi_ref) / norm(y_multi_ref)
    check("refined universe survives a fresh-process reload", d < 1e-12,
          @sprintf("rel diff = %.3e", d))
else
    check("refined universe survives a fresh-process reload", false, string(status_multi_ref))
    println("""
    The refined universe operator is serialized as its Matrix{GlaCmpOprVac} of
    blocks and rebuilt on load. If this fails and the unrefined multi-region
    control above passed, the composite blocks are the ones carrying live state
    to disk, not the assembly around them.""")
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
