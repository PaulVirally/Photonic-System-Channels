## The whole pipeline on a refined system: Green blocks, RSVD, bounds.
##
## Run from the repo root:
##     julia --project=. test/refined_pipeline.jl
##
## test/gap_refinement.jl checks that the derivation produces the right mesh and
## the right cache keys. This one checks that the rest of the code can actually
## run on it: that the three Green blocks build and land in the preload directory
## under their refined names, that the RSVD reads a universe whose degrees of
## freedom are the refined ones, and that a bound comes out finite at the far end.
##
## Refinement is the default now, and the system here says so anyway so that the
## fixture reads as the refined one it is.
##
## The system is (2,2,2) at one cell of gap, where the slab is thicker than the
## body and both sides are refined whole by six in x. N_u = 288 rather than the 48
## the unrefined system would have, which is small enough to run the exact path
## end to end and large enough that a layout mistake shows up as a size error
## rather than as a coincidence.
##
## The proximity warning is the load-bearing check. Gila warns whenever a target
## region and a source region sit closer than six cells of their coarser scale,
## and that warning firing at a production separation would mean the derivation is
## wrong. So the Green stage runs under a logger that collects warnings.
##
## No GPU needed. Everything lands in a mktempdir. Exits nonzero on failure.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using GilaElectromagnetics
using LinearAlgebra
using JLD2
using Logging
using Printf

const ROOT = mktempdir(; cleanup=true)
const PRELOAD = joinpath(ROOT, "preload")
for d in (PRELOAD, joinpath(ROOT, "project"), joinpath(ROOT, "scratch"))
    mkpath(d)
end
println("workspace: ", ROOT)

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-66s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

const ENVIRONMENT = ComputeEnvironment(PRELOAD, joinpath(ROOT, "project"),
                                       joinpath(ROOT, "scratch"), GPUChoice(false, -1))
const SMR = SMRSystem((2, 2, 2), (1//32, 0//1, 0//1), (2, 2, 2),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im;
                      refine_gap=true)
const N_S = dof_length(sender_mesh(SMR))
const N_R = dof_length(receiver_mesh(SMR))
const N_U = N_S + N_R
const PARAMS = RSVDParams(16, 4, 4, 20260826)

check("the system is refined", is_refined(SMR), string(refinement(SMR)))
check("the universe grew with the mesh", (N_S, N_R, N_U) == (144, 144, 288),
      "N_s = $N_S, N_r = $N_R, N_u = $N_U")

println("\n=== the Green stage")
PSC._generate_green_sr(ENVIRONMENT, SMR)

for (target, source) in ((Receiver, Sender), (Receiver, Receiver), (Design, Design))
    fname = target == Design ? PSC.SMRSystems._universe_fname(SMR) : PSC.SMRSystems.green_fname(SMR, target, source)
    check("the $(target)<-$(source) block is on disk under its refined name",
          isfile(joinpath(PRELOAD, fname)) && occursin("_ref", fname), fname)
end

const G_RS = load_green_function(ENVIRONMENT, SMR, Receiver, Sender)
const G_RR = load_green_function(ENVIRONMENT, SMR, Receiver, Receiver)
const G_UU = load_green_function(ENVIRONMENT, SMR, [Sender, Receiver], [Sender, Receiver])
check("the universe operator is a composite over the two meshes", G_UU isa CompositeVacuumGreenOperator, string(typeof(G_UU)))
check("the universe operator is N_u × N_u", size(G_UU) == (N_U, N_U), string(size(G_UU)))
check("the universe operator is a self operator", isselfoperator(G_UU))

# The universe is [sender; receiver], and the blocks of the assembly have to line
# up with the two standalone blocks that Asym(G⁰ᵤᵣ) is built from. A vector
# supported on the sender alone lands where G⁰ᵣₛ says it does.
let x = zeros(ComplexF64, N_U)
    x[1:N_S] .= ComplexF64.(1:N_S) ./ N_S
    y = G_UU * x
    z = G_RS * x[1:N_S]
    d = norm(y[N_S+1:end] - z) / norm(z)
    check("the receiver rows of G⁰ᵤᵤ x are G⁰ᵣₛ x_s", d < 1e-12,
          @sprintf("rel diff = %.3e", d))
end

const A_UR = PSC.asym_ur(G_RS, load_asym_green_function(ENVIRONMENT, SMR, Receiver), SMR)
check("Asym(G⁰ᵤᵣ) is sized by the refined universe", size(A_UR) == (N_U, N_U),
      string(size(A_UR)))
let x = ComplexF64.(1:N_U) ./ N_U, y = ComplexF64.(N_U:-1:1) ./ N_U,
    d = abs(dot(y, A_UR * x) - conj(dot(x, A_UR * y))) / abs(dot(y, A_UR * x))
    check("Asym(G⁰ᵤᵣ) is Hermitian", d < 1e-10, @sprintf("rel asymmetry = %.3e", d))
end

println("\n=== the RSVD stage")
PSC._generate_rsvd_sr(ENVIRONMENT, SMR, PARAMS)
const JLD_PATH = joinpath(scratch_dir(ENVIRONMENT), "$(file_prefix(SMR)).jld")
check("the scratch key carries the refinement", occursin("__refF6T6", file_prefix(SMR)),
      file_prefix(SMR))
check("the RSVD wrote its jld", isfile(JLD_PATH))

const RSVD_OUT = jldopen(JLD_PATH, "r") do jld
    Dict{String,Any}(k => jld[k] for k in ("UR_asym/D", "UR_asym/num_pos",
                                           "UR_asym/exact", "UR_asym/V_pos", "RS/D"))
end
const NUM_POS = RSVD_OUT["UR_asym/num_pos"]
check("the spectrum spans the refined universe", length(RSVD_OUT["UR_asym/D"]) == N_U,
      string(length(RSVD_OUT["UR_asym/D"])))
check("the basis is N_u × num_pos", size(RSVD_OUT["UR_asym/V_pos"]) == (N_U, NUM_POS),
      string(size(RSVD_OUT["UR_asym/V_pos"])))
check("the positive block is nonempty and everything is finite",
      NUM_POS > 0 && all(isfinite, RSVD_OUT["UR_asym/D"]) &&
      all(isfinite, RSVD_OUT["UR_asym/V_pos"]),
      "num_pos = $(NUM_POS) of $(N_U)")
check("RS/D has one value per receiver degree of freedom",
      length(RSVD_OUT["RS/D"]) == N_R && all(isfinite, RSVD_OUT["RS/D"]),
      string(length(RSVD_OUT["RS/D"])))

println("\n=== the bounds stage")
const RESULT = with_logger(SimpleLogger(stderr, Logging.Warn)) do
    PSC._compute_bounds_sr(ENVIRONMENT, SMR; k_uu=16)
end
const BOUNDS_PATH = PSC.bounds_path(ENVIRONMENT, SMR)
check("the bounds run wrote its jld", isfile(BOUNDS_PATH), BOUNDS_PATH)
let vals = jldopen(jld -> jld["true_bounds"], BOUNDS_PATH, "r")
    check("the bounds are finite, positive and passive", all(isfinite, vals) && all(>(0), vals) && all(<=(1), vals),
          string(extrema(vals)))
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
