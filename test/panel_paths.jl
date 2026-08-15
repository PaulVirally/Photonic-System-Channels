## Narval pre-flight 2: the three _save_ur_asym branches (dense-exact, in-memory
## RSVD, panel RSVD) and _run_rsvdvals, on a tiny CPU-only SR system.
##
## Run from the repo root:
##     julia --project=. test/panel_paths.jl
##
## No GPU needed. Everything lands in a mktempdir. A few minutes of runtime, most
## of it the Green functions for the (4,4,4) volumes. Exits nonzero on failure.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using LinearAlgebra
using Printf
using JLD2
using Funicular
using MatrixFreeRandomizedLinearAlgebra
using HDF5
using Random

const ROOT = mktempdir(; cleanup=true)
const PRELOAD = joinpath(ROOT, "preload")
mkpath(PRELOAD)
println("workspace: ", ROOT)

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-58s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

env(tag) = ComputeEnvironment(PRELOAD, joinpath(ROOT, tag, "project"),
                              joinpath(ROOT, tag, "scratch"), GPUChoice(false, -1))
for tag in ("dense", "memory", "panel")
    mkpath(joinpath(ROOT, tag, "scratch"))
    mkpath(joinpath(ROOT, tag, "project"))
end

# The system generate_rsvd would build for an SR run: (4,4,4) sender and receiver,
# separation 1/32 λ, scale 1/32 λ per cell, χ = 13.6 + 0.05i. N_u = 3 * (64 + 64).
const SMR = SMRSystem((4, 4, 4), (1//32, 0//1, 0//1), (4, 4, 4),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im)
const N_U = 3 * (prod(sender(SMR).cel) + prod(receiver(SMR).cel))
const SEED = 20260814
const PARAMS = RSVDParams(48, 24, 14, SEED)

println("N_u = $(N_U), rank = $(PSC.rank(PARAMS)), oversamples = $(oversamples(PARAMS)), seed = $(PSC.seed(PARAMS))")

# A CPU-backend plan, which is what lets the panel branch be exercised without a
# GPU: residency_plan returns nothing on a CPU run by design, so the test injects
# one. panel_width is forced small so several panels are actually swept.
const CPU_PLAN = ResidencyPlan(; backend=Funicular.CPUBackend(),
                               device_budget=64 * 2^20,
                               host_budget=256 * 2^20,
                               panel_width=16)

jldpath(tag) = joinpath(ROOT, tag, "scratch", "$(file_prefix(SMR)).jld")

println("\n=== branch 1: dense-exact (natural threshold, N_u = $(N_U) ≤ $(PSC.DENSE_EXACT_MAX_N_U))")
PSC._save_ur_asym(env("dense"), SMR, PARAMS)

println("\n=== branch 2: in-memory RSVD (dense threshold forced to 0)")
Random.seed!(SEED)
PSC._save_ur_asym(env("memory"), SMR, PARAMS; max_dense_N_u=0)

println("\n=== branch 3: panel RSVD (dense threshold forced to 0, CPU plan injected)")
PSC._save_ur_asym(env("panel"), SMR, PARAMS; max_dense_N_u=0, plan_override=CPU_PLAN)

println("\n=== checks")

read_group(tag) = jldopen(jldpath(tag), "r") do jld
    Dict(k => jld["UR_asym/"*k] for k in ("D", "num_pos", "seed", "exact") if haskey(jld, "UR_asym/"*k))
end
has(tag, k) = jldopen(jldpath(tag), "r") do jld
    haskey(jld, "UR_asym/"*k)
end
get_key(tag, k) = jldopen(jldpath(tag), "r") do jld
    jld["UR_asym/"*k]
end

dense = read_group("dense")
mem = read_group("memory")
pan = read_group("panel")

# --- contract keys
for (tag, group) in (("dense", dense), ("memory", mem), ("panel", pan))
    check("$tag: D, num_pos, seed, exact present",
          all(haskey(group, k) for k in ("D", "num_pos", "seed", "exact")),
          string(sort(collect(keys(group)))))
    check("$tag: seed == $(SEED)", group["seed"] == SEED, string(group["seed"]))
    check("$tag: D is a host Vector{Float64}", get_key(tag, "D") isa Vector{Float64},
          string(typeof(get_key(tag, "D"))))
    check("$tag: num_pos::Int in 1:length(D)",
          group["num_pos"] isa Int && 1 <= group["num_pos"] <= length(group["D"]),
          "num_pos = $(group["num_pos"]) of $(length(group["D"]))")
    check("$tag: legacy full V key is gone", !has(tag, "V"))
end
check("dense: exact == true", dense["exact"] === true)
check("memory: exact == false", mem["exact"] === false)
check("panel: exact == false", pan["exact"] === false)
check("dense: D holds the whole spectrum ($(N_U))", length(dense["D"]) == N_U,
      string(length(dense["D"])))
check("memory: D holds rank(params) = $(PSC.rank(PARAMS)) values",
      length(mem["D"]) == PSC.rank(PARAMS), string(length(mem["D"])))
check("panel: D holds rank(params) = $(PSC.rank(PARAMS)) values",
      length(pan["D"]) == PSC.rank(PARAMS), string(length(pan["D"])))

for tag in ("dense", "memory")
    V = get_key(tag, "V_pos")
    m = read_group(tag)["num_pos"]
    check("$tag: V_pos is a host Matrix{ComplexF64} of $(N_U)×$m",
          V isa Matrix{ComplexF64} && size(V) == (N_U, m), string(typeof(V), size(V)))
end
check("panel: V_pos is NOT inline in the jld", !has("panel", "V_pos"))
check("panel: vectors_file recorded", has("panel", "vectors_file"),
      has("panel", "vectors_file") ? get_key("panel", "vectors_file") : "")
vectors_path = PSC.ur_asym_vectors_path(env("panel"), SMR)
check("panel: vectors_file names the h5 next to the jld",
      has("panel", "vectors_file") && get_key("panel", "vectors_file") == basename(vectors_path) && isfile(vectors_path),
      vectors_path)

# --- descending order and the positive prefix
for tag in ("dense", "memory", "panel")
    D = get_key(tag, "D")
    m = read_group(tag)["num_pos"]
    check("$tag: D descending, positives are the leading $m",
          issorted(D; rev=true) && all(>(0), D[1:m]) && (m == length(D) || D[m+1] <= 0))
end

# --- values: dense is exact, so it is the reference
Dd, Dm, Dp = dense["D"], mem["D"], pan["D"]
md, mm, mp = dense["num_pos"], mem["num_pos"], pan["num_pos"]
@printf("num_pos: dense = %d/%d, memory = %d/%d, panel = %d/%d\n",
        md, length(Dd), mm, length(Dm), mp, length(Dp))

reldiff(a, b) = maximum(abs.(a .- b) ./ max.(abs.(a), abs.(b)))
top = 10
check("memory vs dense: top $top positive eigenvalues to 1e-8",
      reldiff(Dd[1:top], Dm[1:top]) < 1e-8, @sprintf("max rel diff = %.3e", reldiff(Dd[1:top], Dm[1:top])))
check("panel vs dense: top $top positive eigenvalues to 1e-8",
      reldiff(Dd[1:top], Dp[1:top]) < 1e-8, @sprintf("max rel diff = %.3e", reldiff(Dd[1:top], Dp[1:top])))
common_pos = min(mm, mp)
check("panel vs memory: top $top eigenvalues to 1e-8",
      reldiff(Dm[1:top], Dp[1:top]) < 1e-8, @sprintf("max rel diff = %.3e", reldiff(Dm[1:top], Dp[1:top])))
@printf("%-58s      max rel diff = %.3e (all %d shared positives)\n",
        "panel vs memory over the whole positive prefix:", reldiff(Dm[1:common_pos], Dp[1:common_pos]), common_pos)
@printf("%-58s      max rel diff = %.3e\n",
        "memory vs dense over memory's positive prefix:", reldiff(Dd[1:mm], Dm[1:mm]))

# --- the h5 round trip
loaded = Funicular.load(PanelMatrix, vectors_path; plan=CPU_PLAN)
Vp = Matrix(loaded)
check("panel: h5 round-trips to a $(N_U)×$(mp) ComplexF64 matrix",
      size(Vp) == (N_U, mp) && eltype(Vp) == ComplexF64, string(eltype(Vp), size(Vp)))
check("panel: h5 columns are orthonormal", opnorm(Vp' * Vp - I) < 1e-10,
      @sprintf("|V'V - I| = %.3e", opnorm(Vp' * Vp - I)))
Funicular.free!(loaded)

# The dense assembly is the operator itself, so the eigen relation is checkable
# directly: this is what says the h5 columns really are the eigenvectors of the
# eigenvalues sitting next to them in the jld.
G_rs = load_green_function(env("dense"), SMR, Receiver, Sender)
G_rr = load_green_function(env("dense"), SMR, Receiver, Receiver)
A_op, _ = PSC.asym_ur(G_rs, G_rr, SMR)
A = PSC._dense_matrix(A_op, false)
resid(V, D) = norm(A * V - V * Diagonal(D)) / norm(A * V)
# The tail of an RSVD's positive prefix is the part the sketch resolved worst, so
# the eigen relation is only tight over the converged block. Both RSVD paths are
# held to the same bar over the leading columns, and the tail residual is printed
# rather than asserted: it is the method's error, not the plumbing's.
Vm = get_key("memory", "V_pos")
Vd = get_key("dense", "V_pos")
r_pan = resid(Vp[:, 1:top], Dp[1:top])
r_mem = resid(Vm[:, 1:top], Dm[1:top])
check("panel: A V_pos ≈ V_pos diag(Γ) over the leading $top (h5)", r_pan < 1e-8,
      @sprintf("rel residual = %.3e", r_pan))
check("memory: A V_pos ≈ V_pos diag(Γ) over the leading $top", r_mem < 1e-8,
      @sprintf("rel residual = %.3e", r_mem))
check("panel matches memory's accuracy over the whole prefix",
      abs(resid(Vp, Dp[1:mp]) - resid(Vm, Dm[1:mm])) < 1e-2,
      @sprintf("panel = %.3e, memory = %.3e (RSVD error, not fp)",
               resid(Vp, Dp[1:mp]), resid(Vm, Dm[1:mm])))
r_dense = resid(Vd, Dd[1:md])
check("dense: A V_pos ≈ V_pos diag(Γ) over all $md", r_dense < 1e-10, @sprintf("rel residual = %.3e", r_dense))

# Eigenvectors are phase (and, in a degenerate block, rotation) ambiguous, so the
# two runs are compared as subspaces: the principal angles between the leading
# top-column spans, which is what σ(V₁' V₂) reports.
angles = svdvals(Vp[:, 1:top]' * Vm[:, 1:top])
check("panel vs memory: leading $top-dim subspaces coincide",
      maximum(abs.(angles .- 1)) < 1e-6, @sprintf("max |σ - 1| = %.3e", maximum(abs.(angles .- 1))))
angles_d = svdvals(Vp[:, 1:top]' * Vd[:, 1:top])
check("panel vs dense: leading $top-dim subspaces coincide",
      maximum(abs.(angles_d .- 1)) < 1e-6, @sprintf("max |σ - 1| = %.3e", maximum(abs.(angles_d .- 1))))

# --- skip-if-exists, on all three branches
println("\n=== skip-if-exists")
for (tag, kwargs) in (("dense", (;)), ("memory", (; max_dense_N_u=0)), ("panel", (; max_dense_N_u=0, plan_override=CPU_PLAN)))
    before = stat(jldpath(tag)).mtime
    PSC._save_ur_asym(env(tag), SMR, PARAMS; kwargs...)
    after = stat(jldpath(tag)).mtime
    check("$tag: rerun skips (jld untouched)", before == after)
end
# And the panel branch has to redo the work if the h5 is gone.
rm(vectors_path)
PSC._save_ur_asym(env("panel"), SMR, PARAMS; max_dense_N_u=0, plan_override=CPU_PLAN)
check("panel: a missing h5 makes the rerun rebuild it", isfile(vectors_path))

# --- RS values, dense branch (N_r = 192 ≤ 6144)
println("\n=== _run_rsvdvals")
PSC._run_rsvdvals(env("dense"), SMR, PARAMS, "RS/")
rs = jldopen(jldpath("dense"), "r") do jld
    jld["RS/D"]
end
G_rs_dense = PSC._dense_matrix(PSC.LinearMap(load_green_function(env("dense"), SMR, Receiver, Sender)), false)
check("RS/D matches dense svdvals exactly",
      length(rs) == 192 && maximum(abs.(rs .- svdvals(G_rs_dense))) < 1e-10,
      @sprintf("length = %d, max abs diff = %.3e", length(rs), maximum(abs.(rs .- svdvals(G_rs_dense)))))
# The in-memory and panel branches of _run_rsvdvals, forced down.
PSC._run_rsvdvals(env("memory"), SMR, PARAMS, "RS/"; max_dense_N_r=0)
rs_mem = jldopen(jldpath("memory"), "r") do jld
    jld["RS/D"]
end
PSC._run_rsvdvals(env("panel"), SMR, PARAMS, "RS/"; max_dense_N_r=0, plan_override=CPU_PLAN)
rs_pan = jldopen(jldpath("panel"), "r") do jld
    jld["RS/D"]
end
check("RS/D in-memory branch: top $top singular values match dense to 1e-8",
      reldiff(rs[1:top], rs_mem[1:top]) < 1e-8, @sprintf("max rel diff = %.3e", reldiff(rs[1:top], rs_mem[1:top])))
check("RS/D panel branch: top $top singular values match dense to 1e-8",
      reldiff(rs[1:top], rs_pan[1:top]) < 1e-8, @sprintf("max rel diff = %.3e", reldiff(rs[1:top], rs_pan[1:top])))

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
