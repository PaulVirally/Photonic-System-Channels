## The --gamma-rtol spectral cut and the reverse Gram-Schmidt's rank backstop.
##
## Run from the repo root:
##     julia --project=. test/gamma_truncation.jl
##
## No GPU and no Green functions needed: the UR_asym group is fabricated in the
## format _save_ur_asym writes, and everything lands in a mktempdir. Seconds of
## runtime. Exits nonzero on failure.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using LinearAlgebra
using Printf
using JLD2
using Funicular
using HDF5
using Random

const ROOT = mktempdir(; cleanup=true)
println("workspace: ", ROOT)

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-58s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

# The exception `f` threw, or `nothing`. A function rather than a top-level
# try/catch because assigning the caught value to a global from a soft scope
# silently makes a new local instead.
function thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

# --- 1: the cut itself, on a clean top block plus a noise tail

println("\n=== the relative cut")

# Six decades of real spectrum, then a noise floor eleven decades further down,
# then the negative half. This is the shape a large separation produces: the RSVD
# underflows everything past the operator's numerical rank.
const CLEAN = [10.0^(-i) for i in 0:5]
const NOISE = [1e-13, 1e-14, 1e-15, 1e-16]
const NEG = -[1e-16, 1e-10, 1e-3, 1.0]
const SPECTRUM = vcat(CLEAN, NOISE, NEG)
const NUM_POS = length(CLEAN) + length(NOISE)

check("spectrum is descending with num_pos = $NUM_POS",
      issorted(SPECTRUM; rev=true) && count(>(0), SPECTRUM) == NUM_POS)
check("default rtol keeps the clean block only",
      PSC._gamma_kept_count(SPECTRUM, NUM_POS, PSC.DEFAULT_GAMMA_RTOL) == length(CLEAN),
      string(PSC._gamma_kept_count(SPECTRUM, NUM_POS, PSC.DEFAULT_GAMMA_RTOL)))
check("rtol = 0 keeps the whole positive block",
      PSC._gamma_kept_count(SPECTRUM, NUM_POS, 0.0) == NUM_POS,
      string(PSC._gamma_kept_count(SPECTRUM, NUM_POS, 0.0)))
check("rtol = 1e-3 keeps four", PSC._gamma_kept_count(SPECTRUM, NUM_POS, 1e-3) == 4,
      string(PSC._gamma_kept_count(SPECTRUM, NUM_POS, 1e-3)))
check("rtol = 1 keeps the leading eigenvalue alone",
      PSC._gamma_kept_count(SPECTRUM, NUM_POS, 1.0) == 1,
      string(PSC._gamma_kept_count(SPECTRUM, NUM_POS, 1.0)))
# The cut is on the ratio, so scaling the whole spectrum cannot move it.
check("the cut is scale invariant",
      PSC._gamma_kept_count(1e-20 .* SPECTRUM, NUM_POS, PSC.DEFAULT_GAMMA_RTOL) == length(CLEAN))

# --- 2: load_bounds_inputs on a fabricated UR_asym group

println("\n=== dense-path load")

const SMR = SMRSystem((2, 2, 2), (1//32, 0//1, 0//1), (2, 2, 2),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im)
const N_U = 3 * (prod(sender(SMR).cel) + prod(receiver(SMR).cel))
println("N_u = $(N_U), num_pos on disk = $(NUM_POS), kept at the default rtol = $(length(CLEAN))")

Random.seed!(0x5EEDCA5E)
# Orthonormal, as the RSVD's would be; nothing in the reader depends on it, but it
# makes the column comparisons below unambiguous.
const V_POS = Matrix(qr(randn(ComplexF64, N_U, NUM_POS)).Q)[:, 1:NUM_POS]
const GAMMA_RS = collect(range(1.0, 0.1; length=NUM_POS))

env(tag) = ComputeEnvironment(joinpath(ROOT, "preload"), joinpath(ROOT, tag, "project"),
                              joinpath(ROOT, tag, "scratch"), GPUChoice(false, -1))
jldpath(tag) = joinpath(ROOT, tag, "scratch", "$(file_prefix(SMR)).jld")

# The two formats the reader has to slice: V_pos inline, and an h5 the JLD names.
function fabricate(tag; inline::Bool)
    mkpath(joinpath(ROOT, tag, "scratch"))
    mkpath(joinpath(ROOT, tag, "project"))
    mkpath(joinpath(ROOT, "preload"))
    vectors_file = nothing
    if !inline
        path = PSC.ur_asym_vectors_path(env(tag), SMR)
        plan = ResidencyPlan(; backend=Funicular.CPUBackend(), device_budget=8 * 2^20,
                             host_budget=32 * 2^20, panel_width=3)
        pm = PanelMatrix(V_POS; plan=plan)
        try
            Funicular.save(pm, path)
        finally
            Funicular.free!(pm)
        end
        vectors_file = basename(path)
    end
    PSC._save_ur_asym_components(jldpath(tag), "UR_asym/", SPECTRUM, NUM_POS, 1, false;
                                 V_pos=inline ? V_POS : nothing,
                                 vectors_file=vectors_file)
    jldopen(jldpath(tag), "a+") do jld
        jld["RS/D"] = GAMMA_RS
    end
    return nothing
end

fabricate("inline"; inline=true)
fabricate("h5"; inline=false)

for tag in ("inline", "h5")
    inputs = load_bounds_inputs(env(tag), SMR; panel_mode=false, to_device=false)
    check("$tag: num_pos is the kept count",
          inputs.num_pos == length(CLEAN), string(inputs.num_pos))
    check("$tag: the basis has exactly num_pos columns",
          size(inputs.Vur_asym) == (N_U, length(CLEAN)), string(size(inputs.Vur_asym)))
    check("$tag: the kept columns are the leading ones",
          Matrix(inputs.Vur_asym) ≈ V_POS[:, 1:length(CLEAN)])
    check("$tag: Γ[1:num_pos] are the kept eigenvalues",
          inputs.Γ[1:inputs.num_pos] == CLEAN)
    # The invariant downstream runs on: bounds_from_spectrum is given num_pos
    # rather than counting it, because Γ still carries the cut positives.
    check("$tag: Γ keeps the whole saved spectrum",
          length(inputs.Γ) == length(SPECTRUM) && count(>(0), inputs.Γ) == NUM_POS,
          "count(>(0), Γ) = $(count(>(0), inputs.Γ)) vs num_pos = $(inputs.num_pos)")

    all_of_it = load_bounds_inputs(env(tag), SMR; gamma_rtol=0.0, panel_mode=false,
                                  to_device=false)
    check("$tag: gamma_rtol = 0 keeps the whole positive block",
          all_of_it.num_pos == NUM_POS && size(all_of_it.Vur_asym, 2) == NUM_POS,
          "num_pos = $(all_of_it.num_pos), $(size(all_of_it.Vur_asym, 2)) columns")

    bad_rtol = thrown(() -> load_bounds_inputs(env(tag), SMR; gamma_rtol=-1.0,
                                               panel_mode=false, to_device=false))
    check("$tag: a negative gamma_rtol is rejected", bad_rtol isa ArgumentError,
          string(typeof(bad_rtol)))
end

# The panel front end sees the same cut: the h5 case copies the kept panels into a
# matrix of its own, the inline case cuts the truncated dense block into panels.
println("\n=== panel-path load")
const CPU_PLAN = ResidencyPlan(; backend=Funicular.CPUBackend(), device_budget=8 * 2^20,
                               host_budget=32 * 2^20, panel_width=3)
for tag in ("inline", "h5")
    inputs = load_bounds_inputs(env(tag), SMR; plan_override=CPU_PLAN)
    check("$tag: the panel basis has exactly num_pos columns",
          inputs.Vur_asym isa PanelMatrix && size(inputs.Vur_asym) == (N_U, length(CLEAN)),
          string(size(inputs.Vur_asym)))
    check("$tag: the panel basis holds the kept columns",
          Matrix(inputs.Vur_asym) ≈ V_POS[:, 1:length(CLEAN)])
    Funicular.free!(inputs.Vur_asym)
end

# --- 3: the num_pos contract bounds_from_spectrum is held to

println("\n=== bounds_from_spectrum's num_pos")

# These are rejected before any Green operator is loaded, so no volumes are needed.
const V_SMALL = V_POS[:, 1:length(CLEAN)]
bad_num_pos(n, V=V_SMALL) = thrown(() -> bounds_from_spectrum(env("inline"), SMR, SPECTRUM,
                                                              V, GAMMA_RS; num_pos=n))
check("num_pos = 0 is rejected", bad_num_pos(0) isa ArgumentError,
      string(typeof(bad_num_pos(0))))
check("num_pos past length(Γ) is rejected",
      bad_num_pos(length(SPECTRUM) + 1) isa ArgumentError)
check("a num_pos reaching into the non-positive tail is rejected",
      bad_num_pos(NUM_POS + 1) isa ArgumentError)
check("a num_pos wider than the basis is rejected",
      bad_num_pos(NUM_POS) isa ArgumentError,
      "basis has $(size(V_SMALL, 2)) columns")

# --- 4: the reverse Gram-Schmidt backstop

println("\n=== reverse Gram-Schmidt rank backstop")

# Πₛg₁ = (Πₛg₂ + Πₛg₃)/√2 exactly, so the loop reaches i = 1 with nothing left
# after orthogonalizing against s₂ and s₃. The probes for n ≤ 1 do not exist.
const P_FULL = Matrix{ComplexF64}(I, 8, 8)
dependent_gs() = begin
    G = zeros(ComplexF64, 8, 3)
    G[1, 3] = 1
    G[2, 2] = 1
    G[1, 1] = G[2, 1] = 1 / sqrt(2)
    G
end

const SS_DEP = zeros(ComplexF64, 8, 3)
const DEP_ERR = thrown(() -> PSC.reverse_gram_schmidt!(SS_DEP, dependent_gs(), P_FULL, 3))
check("a dependent column is an error, not a normalized noise vector",
      DEP_ERR isa ErrorException, string(typeof(DEP_ERR)))
check("the error names --gamma-rtol",
      DEP_ERR !== nothing && occursin("--gamma-rtol", sprint(showerror, DEP_ERR)))
check("no NaN was written where the dependent probe would have gone",
      all(iszero, view(SS_DEP, :, 1)))
const KEPT = view(SS_DEP, :, 2:3)
check("the probes built before the failure are still orthonormal",
      opnorm(KEPT' * KEPT - I) < 1e-14,
      @sprintf("|s's - I| = %.3e", opnorm(KEPT' * KEPT - I)))

# Independent columns whose projections are uniformly tiny used to trip the old
# absolute 1e-12 floor. The test is relative now, so they go through.
const SS_TINY = zeros(ComplexF64, 8, 3)
const TINY_ERR = thrown(() -> PSC.reverse_gram_schmidt!(SS_TINY, 1e-14 .* Matrix{ComplexF64}(I, 8, 3),
                                                        P_FULL, 3))
check("a uniformly tiny but independent block is not cut", TINY_ERR === nothing,
      string(typeof(TINY_ERR)))
check("its probes come out orthonormal",
      TINY_ERR === nothing && opnorm(SS_TINY' * SS_TINY - I) < 1e-12,
      @sprintf("|s's - I| = %.3e", opnorm(SS_TINY' * SS_TINY - I)))

# And the ordinary case still reproduces the reverse-nested spans it always did.
Random.seed!(0xB0A7)
A = randn(ComplexF64, 8, 4)
ss_ref = zeros(ComplexF64, 8, 4)
PSC.reverse_gram_schmidt!(ss_ref, A, Matrix{ComplexF64}(I, 8, 8), 4)
check("an independent block gives an orthonormal ss",
      opnorm(ss_ref' * ss_ref - I) < 1e-12,
      @sprintf("|s's - I| = %.3e", opnorm(ss_ref' * ss_ref - I)))
# Column i draws on aᵢ…a_m and nothing earlier, which is what the outer loop's
# shrinking probe set rests on.
spans = [opnorm(ss_ref[:, i:4] * (ss_ref[:, i:4]' * A[:, i:4]) - A[:, i:4]) for i in 1:4]
check("span(sᵢ…s_m) == span(aᵢ…a_m) for every i", maximum(spans) < 1e-12,
      @sprintf("max residual = %.3e", maximum(spans)))

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
