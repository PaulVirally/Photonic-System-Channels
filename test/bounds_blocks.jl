## Splitting a bounds job across independent --outer-range blocks, and merging
## them back into the file a monolithic run would have written.
##
## Run from the repo root:
##     julia --project=. test/bounds_blocks.jl
##
## No GPU needed. One tiny (2,2,2)+(2,2,2) SR Green function is generated into a
## mktempdir (about a minute) and reused; the UR_asym spectrum is fabricated in the
## format _save_ur_asym writes, exactly as test/gamma_truncation.jl does. Exits
## nonzero on failure.
##
## The claim under test is equality, not approximation: three blocks plus a merge
## must reproduce a monolithic run's bounds_dual_basis, opt_taus and true_bounds
## *exactly*. Nothing in the outer loop couples the indices (index n's bound is a
## GEVP against the shared pencils and the probes k >= n) so the only thing a
## split can disturb is the τ window, which carries the previous index's best grid
## point forward. That carry is guarded by `n == prev_n + 1`, so the first index of
## each block sweeps the whole grid instead, and a full sweep is the reference the
## window is defined against (test/tau_search.jl). Equality is therefore expected
## and is asserted rather than assumed.
##
## k_uu = 0 throughout. num_pos = 8 is far below the production augment_threshold,
## so the default would augment every run here, and an augmented run cannot be
## split at all, because uu_eigenbasis draws an unseeded sketch and two processes
## get two different bases. That is a property of the augmentation, not of the
## split; test/augmented_basis.jl covers the augmented path, and
## bench/size_bounds_jobs.jl refuses to split an augmenting point.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using JLD2
using LinearAlgebra
using Printf
using Random
using Logging

# The merge tool, in its own module: it is a script with its own `main`,
# `parse_cli` and friends, and this file wants none of those names.
module MergeTool
    include(joinpath(@__DIR__, "..", "bench", "merge_bounds_blocks.jl"))
end

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-62s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

function thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

const ROOT = mktempdir(; cleanup=true)
println("workspace: ", ROOT)

## `refine_gap=false`: these checks are about the bounds algebra, not the mesh, and
## they are written against the plain uniform universe. The refined mesh of a
## one-cell gap is a different (and much larger) operator; test/gap_refinement.jl
## and test/refined_pipeline.jl cover that side.
const SMR = SMRSystem((2, 2, 2), (1//32, 0//1, 0//1), (2, 2, 2),
                      SMRVolumeSymbol[Sender, Receiver], 1//32, 13.6 + 0.05im;
                      refine_gap=false)
const N_U = 3 * (prod(sender(SMR).cel) + prod(receiver(SMR).cel))
const K = 8                      # num_pos, and the number of channel indices
const PREFIX = file_prefix(SMR)
const RSVD_PARAMS = RSVDParams(K, 2, 4, 12345)

const ζ = abs(PSC.susceptibility(SMR))^2 / imag(PSC.susceptibility(SMR))
Random.seed!(0xB10CC5)
const Γ = vcat(collect(exp10.(range(log10(1 / ζ), log10(1e-4 / ζ), length=K))),
               -[1e-6 / ζ, 1e-3 / ζ])
const V_POS = Matrix(qr(randn(ComplexF64, N_U, K)).Q)[:, 1:K]
const Γ_RS = collect(exp10.(range(log10(1 / ζ), log10(1e-6 / ζ), length=K)))

"A workspace with the fabricated RSVD output on scratch, ready for a bounds run."
function make_env(tag::AbstractString)
    for d in ("preload", "$tag/project", "$tag/scratch")
        mkpath(joinpath(ROOT, d))
    end
    env = ComputeEnvironment(joinpath(ROOT, "preload"), joinpath(ROOT, tag, "project"),
                             joinpath(ROOT, tag, "scratch"), GPUChoice(false, -1))
    jld = joinpath(scratch_dir(env), "$(PREFIX).jld")
    if !isfile(jld)
        PSC._save_ur_asym_components(jld, "UR_asym/", Γ, K, 1, false; V_pos=V_POS)
        jldopen(jld, "a+") do f
            f["RS/D"] = Γ_RS
        end
    end
    return env
end

"The per-index @info lines are the loop's own noise; only warnings matter here."
run_bounds(env; kwargs...) = with_logger(SimpleLogger(stderr, Logging.Warn)) do
    PSC._compute_bounds_sr(env, SMR, RSVD_PARAMS; k_uu=0, panel_mode=false, kwargs...)
end

read_jld(path) = jldopen(path, "r") do f
    Dict{String,Any}(k => f[k] for k in MergeTool.all_keys(f))
end

# --- 1: a monolithic run is what it always was

println("\n=== the monolithic run")

const MONO_ENV = make_env("mono")
run_bounds(MONO_ENV)
const MONO_PATH = joinpath(project_dir(MONO_ENV), "$(PREFIX).jld")
const MONO = read_jld(MONO_PATH)

check("the monolithic run writes <prefix>.jld", isfile(MONO_PATH))
# The key list is the contract the merge is written against, and the one thing a
# merged file cannot be allowed to differ in.
check("its keys are exactly FINAL_KEYS, in order",
      collect(keys(MONO)) |> (ks -> sort(ks) == sort(MergeTool.FINAL_KEYS)) &&
      jldopen(MergeTool.all_keys, MONO_PATH, "r") == MergeTool.FINAL_KEYS,
      "$(length(MONO)) keys")
check("no partial/ group on a normal run",
      !any(startswith(k, "partial/") for k in keys(MONO)))
check("every index has a bound",
      length(MONO["bounds_dual_basis"]) == K && !any(isnan, MONO["bounds_dual_basis"]),
      string(MONO["bounds_dual_basis"]))
check("the augmentation is off at k_uu = 0", MONO["augment/augmented"] == false)

# The new plumbing has to be inert unless asked for: an explicit full range with the
# masking off must reproduce the default call term for term.
const G_UU = load_green_function(MONO_ENV, SMR, [Sender, Receiver], [Sender, Receiver])
raw_bounds(; kwargs...) = with_logger(SimpleLogger(stderr, Logging.Warn)) do
    bounds_from_spectrum(MONO_ENV, SMR, Γ, V_POS, Γ_RS; num_pos=K, G₀_uu=G_UU, k_uu=0,
                         kwargs...)
end
const PLAIN = raw_bounds()
const EXPLICIT = raw_bounds(outer_indices=collect(1:K))
check("outer_indices = 1:K with masking off is the default call",
      isequal(PLAIN.bounds_dual_basis, EXPLICIT.bounds_dual_basis) &&
      isequal(PLAIN.opt_taus, EXPLICIT.opt_taus) &&
      isequal(PLAIN.true_bounds, EXPLICIT.true_bounds) &&
      isequal(PLAIN.which_bounds, EXPLICIT.which_bounds))
check("and it reports every index as evaluated",
      PLAIN.evaluated_indices == collect(1:K) && PLAIN.complete)

# --- 2: three blocks

println("\n=== three blocks")

# Deliberately unequal, and deliberately not the split the sizer would pick: what
# matters is that the boundaries fall inside the τ plateau, where the window would
# otherwise have carried a grid point across them.
const RANGES = [1:2, 3:5, 6:K]
const BLOCK_ENV = make_env("blocks")
for (i, r) in enumerate(RANGES)
    run_bounds(BLOCK_ENV; outer_range=r, partial_suffix="b$(i)of$(length(RANGES))")
end
const PARTIAL_PATHS = [partial_bounds_path(project_dir(BLOCK_ENV), PREFIX,
                                           "b$(i)of$(length(RANGES))")
                       for i in eachindex(RANGES)]
check("each block wrote its own file", all(isfile, PARTIAL_PATHS),
      join(basename.(PARTIAL_PATHS), ", "))
check("and none of them wrote the point's real filename",
      !isfile(joinpath(project_dir(BLOCK_ENV), "$(PREFIX).jld")))

const PARTIALS = read_jld.(PARTIAL_PATHS)
check("a partial holds FINAL_KEYS plus the partial/ group",
      all(sort(collect(keys(p))) == sort(vcat(MergeTool.FINAL_KEYS, MergeTool.PARTIAL_KEYS))
          for p in PARTIALS))
check("partial/indices is the block's range",
      all(PARTIALS[i]["partial/indices"] == collect(RANGES[i]) for i in eachindex(RANGES)))
check("partial/range_lo|hi and num_pos are recorded",
      all(PARTIALS[i]["partial/range_lo"] == first(RANGES[i]) &&
          PARTIALS[i]["partial/range_hi"] == last(RANGES[i]) &&
          PARTIALS[i]["partial/num_pos"] == K for i in eachindex(RANGES)))
# The point of the masking: outside its range a block says "I do not know", not "0".
check("a block is NaN outside its range, finite inside it",
      all(all(isnan, PARTIALS[i]["bounds_dual_basis"][setdiff(1:K, RANGES[i])]) &&
          all(isfinite, PARTIALS[i]["bounds_dual_basis"][RANGES[i]])
          for i in eachindex(RANGES)))
check("and so are true_bounds, with which_bounds pointing at the dual",
      all(all(isnan, PARTIALS[i]["true_bounds"][setdiff(1:K, RANGES[i])]) &&
          all(==(3), PARTIALS[i]["which_bounds"][setdiff(1:K, RANGES[i])])
          for i in eachindex(RANGES)))
check("the front-end keys agree across the blocks",
      all(isequal(PARTIALS[1][k], p[k]) for p in PARTIALS, k in MergeTool.SHARED_KEYS))

# The equality claim, before the merge is even involved.
const BLOCKWISE = [PARTIALS[findfirst(r -> n in r, RANGES)]["bounds_dual_basis"][n]
                   for n in 1:K]
check("each block's bound equals the monolithic one, exactly",
      isequal(BLOCKWISE, MONO["bounds_dual_basis"]),
      @sprintf("worst |Δ| = %.3e", maximum(abs.(BLOCKWISE .- MONO["bounds_dual_basis"]))))

# --- 3: the merge

println("\n=== the merge")

const MERGED_PATH = MergeTool.main(["--project", project_dir(BLOCK_ENV),
                                    "--prefix", PREFIX, "--quiet"])
const MERGED = read_jld(MERGED_PATH)

check("the merge writes <prefix>.jld", isfile(MERGED_PATH) &&
      MERGED_PATH == joinpath(project_dir(BLOCK_ENV), "$(PREFIX).jld"))
check("the merged file's keys are the monolithic file's, in the same order",
      jldopen(MergeTool.all_keys, MERGED_PATH, "r") ==
      jldopen(MergeTool.all_keys, MONO_PATH, "r"))
for key in ("bounds_dual_basis", "opt_taus", "true_bounds", "which_bounds")
    check("merged $key == monolithic $key, exactly", isequal(MERGED[key], MONO[key]),
          key == "which_bounds" ? string(MERGED[key]) :
          @sprintf("max |Δ| = %.3e",
                   maximum(abs.(Float64.(MERGED[key]) .- Float64.(MONO[key])))))
end
for key in MergeTool.SHARED_KEYS
    check("merged $key == monolithic $key", isequal(MERGED[key], MONO[key]))
end
check("array shapes match the monolithic ones",
      all(size(MERGED[k]) == size(MONO[k]) for k in MergeTool.FINAL_KEYS
          if MONO[k] isa AbstractArray))
check("types match the monolithic ones",
      all(typeof(MERGED[k]) == typeof(MONO[k]) for k in MergeTool.FINAL_KEYS))
# bounds_dual_by_tau is a diagnostic table, and the two runs skip different grid
# points: each block's first index sweeps the whole grid where the monolithic run
# windowed. So the merged table is a superset, and where both filled an entry they
# must agree.
const TAU_TABLE_OK = all(isnan(MONO["bounds_dual_by_tau"][i]) ||
                         isequal(MONO["bounds_dual_by_tau"][i], MERGED["bounds_dual_by_tau"][i])
                         for i in eachindex(MONO["bounds_dual_by_tau"]))
check("the τ table is a superset of the monolithic one, agreeing where both are set",
      TAU_TABLE_OK &&
      count(isnan, MERGED["bounds_dual_by_tau"]) <= count(isnan, MONO["bounds_dual_by_tau"]),
      "$(count(isnan, MONO["bounds_dual_by_tau"])) NaN monolithic, " *
      "$(count(isnan, MERGED["bounds_dual_by_tau"])) merged")

check("merging again refuses to clobber the result",
      thrown(() -> MergeTool.main(["--project", project_dir(BLOCK_ENV),
                                   "--prefix", PREFIX, "--quiet"])) !== nothing)
check("--force overwrites it",
      MergeTool.main(["--project", project_dir(BLOCK_ENV), "--prefix", PREFIX,
                      "--quiet", "--force"]) == MERGED_PATH)

# --- 4: what the merge refuses

println("\n=== merge validation")

"A copy of the blocks in their own directory, so each case starts clean."
function stage_partials(tag; drop=Int[])
    dir = joinpath(ROOT, "case_$(tag)")
    mkpath(dir)
    for (i, p) in enumerate(PARTIAL_PATHS)
        i in drop && continue
        cp(p, joinpath(dir, basename(p)); force=true)
    end
    return dir
end

const BAD_GAMMA = stage_partials("gamma")
let path = joinpath(BAD_GAMMA, basename(PARTIAL_PATHS[2]))
    data = read_jld(path)
    data["Γ"] = data["Γ"] .* 1.01 # a rerun RSVD, or a different point altogether
    rm(path)
    jldopen(path, "w") do f
        for k in vcat(MergeTool.FINAL_KEYS, MergeTool.PARTIAL_KEYS)
            f[k] = data[k]
        end
    end
end
const GAMMA_ERR = thrown(() -> MergeTool.main(["--project", BAD_GAMMA, "--prefix", PREFIX,
                                               "--quiet"]))
check("a partial with a different Γ is refused",
      GAMMA_ERR !== nothing && occursin("Γ", sprint(showerror, GAMMA_ERR)),
      GAMMA_ERR === nothing ? "no error" : first(split(sprint(showerror, GAMMA_ERR), '\n')))
check("and nothing was written", !isfile(joinpath(BAD_GAMMA, "$(PREFIX).jld")))

const GAPPED = stage_partials("gap"; drop=[2])
const GAP_ERR = thrown(() -> MergeTool.main(["--project", GAPPED, "--prefix", PREFIX,
                                             "--quiet"]))
check("a missing block is refused",
      GAP_ERR !== nothing && occursin("covered by no block", sprint(showerror, GAP_ERR)),
      GAP_ERR === nothing ? "no error" : first(split(sprint(showerror, GAP_ERR), '\n')))

const GAP_PATH = MergeTool.main(["--project", GAPPED, "--prefix", PREFIX, "--quiet",
                                 "--allow-gaps"])
const GAPPED_OUT = read_jld(GAP_PATH)
check("--allow-gaps merges the rest and leaves the hole NaN",
      all(isnan, GAPPED_OUT["bounds_dual_basis"][RANGES[2]]) &&
      isequal(GAPPED_OUT["bounds_dual_basis"][setdiff(1:K, RANGES[2])],
              MONO["bounds_dual_basis"][setdiff(1:K, RANGES[2])]),
      string(GAPPED_OUT["bounds_dual_basis"]))
check("the gap's true_bounds is NaN too, not a zero-valued bound",
      all(isnan, GAPPED_OUT["true_bounds"][RANGES[2]]))

# A resubmitted block that overlaps its neighbours: allowed, because the values it
# recomputes are the same values. This is the shape of "block 2 failed, I reran it
# with a slightly wider range".
const OVERLAP_ENV = make_env("overlap")
for (i, r) in enumerate(RANGES)
    run_bounds(OVERLAP_ENV; outer_range=r, partial_suffix="b$(i)of$(length(RANGES))")
end
run_bounds(OVERLAP_ENV; outer_range=2:6, partial_suffix="redo")
const OVERLAP_PATH = MergeTool.main(["--project", project_dir(OVERLAP_ENV),
                                     "--prefix", PREFIX, "--quiet"])
check("overlapping blocks that agree are merged",
      isequal(read_jld(OVERLAP_PATH)["bounds_dual_basis"], MONO["bounds_dual_basis"]))

const NOT_A_BLOCK = joinpath(ROOT, "case_notablock")
mkpath(NOT_A_BLOCK)
cp(MONO_PATH, joinpath(NOT_A_BLOCK, "$(PREFIX)_partial_finished.jld"))
const SHAPE_ERR = thrown(() -> MergeTool.main(["--project", NOT_A_BLOCK, "--prefix", PREFIX,
                                               "--quiet"]))
check("a finished point renamed as a partial is refused",
      SHAPE_ERR !== nothing && occursin("partial/", sprint(showerror, SHAPE_ERR)),
      SHAPE_ERR === nothing ? "no error" : first(split(sprint(showerror, SHAPE_ERR), '\n')))

const NO_BLOCKS = joinpath(ROOT, "case_empty")
mkpath(NO_BLOCKS)
check("an empty directory is an error, not an empty merge",
      thrown(() -> MergeTool.main(["--project", NO_BLOCKS, "--prefix", PREFIX,
                                   "--quiet"])) !== nothing)

# --- 5: --cleanup

println("\n=== cleanup")

const CLEAN_DIR = stage_partials("cleanup")
MergeTool.main(["--project", CLEAN_DIR, "--prefix", PREFIX, "--quiet", "--cleanup"])
check("--cleanup leaves only the merged file",
      isfile(joinpath(CLEAN_DIR, "$(PREFIX).jld")) &&
      isempty(filter(f -> occursin("_partial_", f), readdir(CLEAN_DIR))),
      join(readdir(CLEAN_DIR), ", "))
check("and the merged file is still the monolithic one",
      isequal(read_jld(joinpath(CLEAN_DIR, "$(PREFIX).jld"))["bounds_dual_basis"],
              MONO["bounds_dual_basis"]))

# --- 6: the CLI's own guards

println("\n=== flag guards")

const GUARD_ENV = make_env("guards")
check("--outer-range without --partial-suffix is refused",
      thrown(() -> run_bounds(GUARD_ENV; outer_range=1:3)) !== nothing)
check("--partial-suffix without --outer-range is refused",
      thrown(() -> run_bounds(GUARD_ENV; partial_suffix="lonely")) !== nothing)
check("a tag that is not filename-safe is refused",
      thrown(() -> run_bounds(GUARD_ENV; outer_range=1:3,
                              partial_suffix="../escape")) isa ArgumentError)
check("the range parser rejects hi < lo",
      thrown(() -> PSC.parse_index_range("9:3")) !== nothing)
check("the range parser rejects a 0 lower bound",
      thrown(() -> PSC.parse_index_range("0:3")) !== nothing)
check("the range parser takes a bare index", PSC.parse_index_range("7") == 7:7)
check("the range parser takes lo:hi", PSC.parse_index_range(" 12:40 ") == 12:40)

# A block whose range hangs off the end of a spectrum that moved: legal, and it
# must not be mergeable as though it had covered those indices.
const TAIL_ENV = make_env("tail")
run_bounds(TAIL_ENV; outer_range=1:K, partial_suffix="b1of2")
run_bounds(TAIL_ENV; outer_range=(K + 1):(K + 5), partial_suffix="b2of2")
const TAIL_PATH = MergeTool.main(["--project", project_dir(TAIL_ENV), "--prefix", PREFIX,
                                  "--quiet"])
check("a block past the end of the spectrum evaluates nothing and merges cleanly",
      isequal(read_jld(TAIL_PATH)["bounds_dual_basis"], MONO["bounds_dual_basis"]))

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
