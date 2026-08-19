#!/usr/bin/env julia
"""
    patch_rs_chi.jl -- audit (and where possible repair) the χ dependence of the
                       RSVD-stage JLDs after a susceptibility correction.

Every `Ge1000` sweep up to 2026-08-19 was launched with

    χ_old = 4.25 + 0.0342557im      (ζ = |χ|²/ℑχ ≈ 527.4)

which is germanium's *refractive index*, not its susceptibility. The corrected
value is

    χ_new = 17.06132654701751 + 0.29117345im   (ζ = 1000 exactly)

This script answers, per file, the only question that matters for salvage: does
anything stored in this JLD depend on χ, and if it does, can the correction be
applied in place?

# The answer for this codebase: nothing in the RSVD-stage JLD depends on χ

The RSVD stage for a sender/receiver system is `_generate_rsvd_sr` in
`src/rsvd.jl`. It writes exactly two groups:

  * `UR_asym/` -- `_save_ur_asym` → `reigen_hermitian(G₀_ur_asym, …)`, i.e. the
    eigendecomposition of `Asym(G⁰ᵤᵣ)`, a vacuum Green operator. χ never enters.
    Keys: `D`, `num_pos`, `seed`, `exact`, and one of `V_pos` / `vectors_file`.

  * `RS/` -- `_run_rsvdvals(compute_env, smr, rsvd_params, "RS/")`, i.e. the
    *singular values of the vacuum Green operator* G⁰ from the sender volume to
    the receiver volume (`char2volume_symbol` maps 'R'→Receiver, 'S'→Sender).
    χ never enters here either. Key: `D`.

`RS/D` is therefore **NOT** χ-dependent and must not be shifted. Shifting it
would corrupt 333 valid outputs.

The expression `imag(inv(χ))*I - asym(LinearMap(G₀_rr))`, which *is* a uniform
scalar shift of a χ-free spectrum, lives in `_save_constraint_asym`
(`src/rsvd.jl:477-478`) and writes the `constraint_asym/` group. That function
has been commented out of `_generate_rsvd_sr` (`src/rsvd.jl:494-495`) ever since
`_generate_rsvd_sr` was introduced (commit 8652775, 2026-06-01), so current
sweeps never produce the group at all. If an older file does carry it, this
script can repair it: eigenvalues of `c·I - B` shift by `Δc` and the
eigenvectors are unchanged, so

    Δ = ℑ(1/χ_new) - ℑ(1/χ_old) = -1/ζ_new + 1/ζ_old

added to `constraint_asym/D` is exact.

χ enters the pipeline at the *bounds* stage instead (`src/bounds.jl:1070-1071`
computes ζ = |χ|²/ℑχ and lines 1335-1336 apply it to the χ-free Γ and Γrs it
read from the JLD), so bounds outputs are not salvageable and must be re-run.
Note that the `--gamma-rtol` spectral cut in `load_bounds_inputs` acts on Γ
alone, so the truncated `m` -- and hence every cost-model time and memory
request -- is χ-independent too.

# Usage

    julia --project=. bench/patch_rs_chi.jl <scratch-dir> [options]

    --apply                 write the patch (default is a dry run)
    --old <chi>             χ the files were written with
                            (default 4.25+0.0342557im)
    --new <chi>             χ they should describe
                            (default 17.06132654701751+0.29117345im)
    --keys <g1/,g2/,…>      groups whose `D` is a uniformly shifted spectrum and
                            may be patched (default `constraint_asym/`).
                            `RS/` and `UR_asym/` are refused: see above.
    --quiet                 one line per file instead of a full key listing

# Idempotence

A patched group gets a `<group>chi` scalar recording the χ it now describes. A
group whose recorded `chi` already equals `--new` is skipped; a group with no
`chi` key is treated as having been written by the old code at `--old`. Running
with `--apply` twice is a no-op the second time.

# Rewriting a dataset with JLD2

`_save_component` in `src/rsvd.jl` skips a key that already exists, so opening
with `"a+"` and assigning is *not* enough to change a value -- and JLD2 itself
refuses to overwrite an existing name. The dataset has to be `delete!`d first
and then rewritten, which is what `_rewrite!` below does. `delete!` unlinks
rather than reclaims, so the file grows by the size of the new array; `D` is a
few tens of kilobytes, so this is irrelevant here.
"""

using JLD2
using Printf

# Groups that `_generate_rsvd_sr` writes, and whether χ enters them. Derived by
# reading every `_save_*` call reachable from `generate_rsvd` in src/rsvd.jl.
const CHI_FREE_KEYS = Dict(
    "UR_asym/D"            => "eigenvalues of Asym(G⁰ᵤᵣ), a vacuum Green operator",
    "UR_asym/V_pos"        => "eigenvectors of Asym(G⁰ᵤᵣ), inline",
    "UR_asym/vectors_file" => "basename of the streamed eigenvector block",
    "UR_asym/num_pos"      => "count of positive Γ",
    "UR_asym/seed"         => "RSVD seed",
    "UR_asym/exact"        => "whether the dense-exact path was taken",
    "UR_asym/V"            => "legacy full eigenvector block",
    "RS/D"                 => "singular values of the vacuum G⁰ sender→receiver",
    "RS/U"                 => "right singular vectors of G⁰ sender→receiver",
    "RS/V"                 => "left singular vectors of G⁰ sender→receiver",
    # SMR-system keys, listed so an SMR file does not read as "unknown".
    "SM/D" => "vacuum G⁰ sender→mediator", "SM/U" => "vacuum G⁰ sender→mediator",
    "SM/V" => "vacuum G⁰ sender→mediator",
    "MR/D" => "vacuum G⁰ mediator→receiver", "MR/U" => "vacuum G⁰ mediator→receiver",
    "MR/V" => "vacuum G⁰ mediator→receiver",
    "MM/D" => "vacuum G⁰ mediator→mediator",
    "UU/D" => "vacuum G⁰ universe→universe", "UU/U" => "vacuum G⁰ universe→universe",
    "UU/V" => "vacuum G⁰ universe→universe",
    "UU_asym/D" => "eigenvalues of Asym(G⁰ᵤᵤ)",
    "UU_asym/V" => "eigenvectors of Asym(G⁰ᵤᵤ)",
)

# Groups whose `D` is `ℑ(1/χ)·I` plus a χ-free operator, so a uniform shift of
# `D` (leaving `V` alone) converts one χ to another exactly.
const SHIFTABLE_GROUPS = Dict(
    "constraint_asym/" => "eigenvalues of ℑ(1/χ)I - Asym(G⁰ᵣᵣ) (src/rsvd.jl:477-478)",
)

# Refusing these by name, with the reason, is the whole point of the script.
const REFUSED_GROUPS = Dict(
    "RS/" => "RS/D holds the singular values of the vacuum Green operator G⁰ from " *
             "the sender to the receiver (_run_rsvdvals at src/rsvd.jl:558, called " *
             "from _generate_rsvd_sr at src/rsvd.jl:492). χ never enters it. ζ is " *
             "applied to these values later, at bounds time (src/bounds.jl:1335). " *
             "Shifting RS/D would corrupt correct data.",
    "UR_asym/" => "UR_asym/ holds the eigendecomposition of Asym(G⁰ᵤᵣ), a vacuum " *
                  "Green operator (_save_ur_asym at src/rsvd.jl:~330). χ never " *
                  "enters it.",
)

chi_imag_inv(χ::ComplexF64) = imag(inv(χ))
zeta(χ::ComplexF64) = abs2(χ) / imag(χ)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

struct Options
    dir::String
    apply::Bool
    χ_old::ComplexF64
    χ_new::ComplexF64
    groups::Vector{String}
    quiet::Bool
end

const USAGE = """
usage: julia --project=. bench/patch_rs_chi.jl <scratch-dir> [--apply]
              [--old <chi>] [--new <chi>] [--keys <group/,…>] [--quiet]

Audits every *.jld in <scratch-dir> for χ-dependent data and, with --apply,
shifts the spectra that a χ change shifts uniformly. See the docstring at the
top of this file for why RS/D is not one of them.
"""

function parse_options(argv::Vector{String})
    dir = nothing
    apply = false
    quiet = false
    χ_old = ComplexF64(4.25, 0.0342557)
    χ_new = ComplexF64(17.06132654701751, 0.29117345)
    groups = collect(keys(SHIFTABLE_GROUPS))
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--apply"
            apply = true
        elseif a == "--quiet"
            quiet = true
        elseif a in ("-h", "--help")
            print(USAGE)
            exit(0)
        elseif a == "--old"
            i += 1; i <= length(argv) || error("--old needs a value")
            χ_old = parse(ComplexF64, argv[i])
        elseif a == "--new"
            i += 1; i <= length(argv) || error("--new needs a value")
            χ_new = parse(ComplexF64, argv[i])
        elseif a == "--keys"
            i += 1; i <= length(argv) || error("--keys needs a value")
            groups = String[]
            for g in split(argv[i], ',')
                g = strip(g)
                isempty(g) && continue
                endswith(g, "/") || (g = g * "/")
                if haskey(REFUSED_GROUPS, g)
                    println(stderr, "\nREFUSED: --keys named $(g), which is not χ-dependent.\n")
                    println(stderr, "  " * REFUSED_GROUPS[g])
                    println(stderr, "\nNothing was written. Re-read the docstring at the top of " *
                                    "bench/patch_rs_chi.jl.")
                    exit(2)
                end
                push!(groups, g)
            end
        elseif startswith(a, "-")
            error("unknown option $(a)\n\n" * USAGE)
        else
            isnothing(dir) || error("more than one directory given ($(dir), $(a))\n\n" * USAGE)
            dir = a
        end
        i += 1
    end
    isnothing(dir) && error("no directory given\n\n" * USAGE)
    isdir(dir) || error("not a directory: $(dir)")
    return Options(dir, apply, χ_old, χ_new, sort(groups), quiet)
end

# ---------------------------------------------------------------------------
# JLD2 helpers
# ---------------------------------------------------------------------------

"""
    _rewrite!(jld, key, value)

Replace an existing dataset. JLD2 will not overwrite a name in place, and
`src/rsvd.jl`'s `_save_component` skips keys that already exist, so the old
entry is unlinked with `delete!` before the new one is written. `delete!` does
not reclaim the bytes; the arrays involved here are tens of kilobytes.
"""
function _rewrite!(jld, key::String, value)
    haskey(jld, key) && delete!(jld, key)
    jld[key] = value
    return nothing
end

"""
    _flat_keys(jld) -> Vector{String}

Every leaf path in the file, `group/name`, one level of nesting deep, which is
all the RSVD stage ever writes.
"""
function _flat_keys(jld)
    out = String[]
    for k in keys(jld)
        v = try
            jld[k]
        catch
            nothing
        end
        if v isa JLD2.Group
            for sub in keys(v)
                push!(out, "$(k)/$(sub)")
            end
        else
            push!(out, k)
        end
    end
    return sort(out)
end

_group_of(key::String) = (i = findlast('/', key); isnothing(i) ? "" : key[1:i])

# ---------------------------------------------------------------------------
# per-file work
# ---------------------------------------------------------------------------

mutable struct Tally
    files::Int
    chi_free::Int          # nothing χ-dependent found
    patched::Int           # a shiftable group was (or would be) shifted
    already::Int           # a shiftable group already records χ_new
    unknown::Int           # a key not in the classification table
    failed::Int
end
Tally() = Tally(0, 0, 0, 0, 0, 0)

function process_file(path::String, o::Options, Δ::Float64, t::Tally)
    t.files += 1
    name = basename(path)
    local flat, present_groups
    try
        jldopen(path, "r") do jld
            flat = _flat_keys(jld)
            present_groups = unique(_group_of(k) for k in flat)
        end
    catch err
        t.failed += 1
        @printf("%-60s  UNREADABLE (%s)\n", name, sprint(showerror, err))
        return
    end

    unknown = filter(k -> !haskey(CHI_FREE_KEYS, k) &&
                          !any(g -> startswith(k, g), o.groups) &&
                          !endswith(k, "/chi"), flat)

    # Which of the requested groups are actually here and carry a `D`?
    actionable = String[]
    for g in o.groups
        (g * "D") in flat && push!(actionable, g)
    end

    if !o.quiet
        println()
        println(name)
        for k in flat
            cls = haskey(CHI_FREE_KEYS, k) ? "χ-free   ($(CHI_FREE_KEYS[k]))" :
                  endswith(k, "/chi")      ? "provenance" :
                  any(g -> startswith(k, g), o.groups) ? "shiftable group" :
                  "UNKNOWN -- not in the classification table"
            @printf("    %-26s %s\n", k, cls)
        end
    end

    if !isempty(unknown)
        t.unknown += 1
        for k in unknown
            @printf("    !! %s is not classified; refusing to guess whether χ enters it\n", k)
        end
    end

    if isempty(actionable)
        t.chi_free += 1
        o.quiet && @printf("%-60s  no χ-dependent key; nothing to do\n", name)
        !o.quiet && println("    -> no χ-dependent key present; nothing to do")
        return
    end

    for g in actionable
        jldopen(path, o.apply ? "a+" : "r") do jld
            recorded = haskey(jld, g * "chi") ? ComplexF64(jld[g * "chi"]) : nothing
            if recorded !== nothing && recorded == o.χ_new
                t.already += 1
                @printf("    %-20s already records χ = %s; skipping\n", g, string(recorded))
                return
            end
            from = recorded === nothing ? o.χ_old : recorded
            δ = chi_imag_inv(o.χ_new) - chi_imag_inv(from)
            D = Array(jld[g * "D"])
            @printf("    %-20s %d values, χ %s -> %s, Δ = %+.12g\n",
                    g, length(D), string(from), string(o.χ_new), δ)
            @printf("    %-20s D[1] %.12g -> %.12g   D[end] %.12g -> %.12g\n",
                    "", first(D), first(D) + δ, last(D), last(D) + δ)
            if recorded === nothing
                @printf("    %-20s (no %schi recorded; assuming it was written at χ = %s)\n",
                        "", g, string(o.χ_old))
            end
            if o.apply
                _rewrite!(jld, g * "D", D .+ δ)
                _rewrite!(jld, g * "chi", o.χ_new)
                println("    -> written")
            else
                println("    -> DRY RUN, nothing written (pass --apply)")
            end
            t.patched += 1
        end
    end
    return
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function main(argv::Vector{String})
    o = parse_options(argv)
    Δ = chi_imag_inv(o.χ_new) - chi_imag_inv(o.χ_old)

    println("patch_rs_chi.jl")
    println("  directory : $(o.dir)")
    @printf("  χ_old     : %s   (ζ = %.6g, ℑ(1/χ) = %.12g)\n",
            string(o.χ_old), zeta(o.χ_old), chi_imag_inv(o.χ_old))
    @printf("  χ_new     : %s   (ζ = %.6g, ℑ(1/χ) = %.12g)\n",
            string(o.χ_new), zeta(o.χ_new), chi_imag_inv(o.χ_new))
    @printf("  Δ         : %+.12g\n", Δ)
    println("  groups    : $(join(o.groups, ", "))")
    println("  mode      : $(o.apply ? "APPLY (files will be modified)" : "dry run")")
    println()
    println("  Not patched, by construction: RS/D and UR_asym/* are χ-free in this")
    println("  codebase (see the docstring at the top of this file). If the audit")
    println("  below reports nothing but χ-free keys, the RSVD outputs in this")
    println("  directory are already correct for χ_new and only the bounds stage")
    println("  needs re-running.")

    files = sort(filter(f -> endswith(f, ".jld"), readdir(o.dir; join=true)))
    if isempty(files)
        println("\nNo *.jld files in $(o.dir).")
        return 0
    end
    println("\n$(length(files)) *.jld file(s) found.")

    t = Tally()
    for f in files
        process_file(f, o, Δ, t)
    end

    println()
    println("summary")
    @printf("  files scanned            %d\n", t.files)
    @printf("  χ-free, nothing to do    %d\n", t.chi_free)
    @printf("  %-24s %d\n", o.apply ? "patched" : "would patch", t.patched)
    @printf("  already at χ_new         %d\n", t.already)
    @printf("  with unclassified keys   %d\n", t.unknown)
    @printf("  unreadable               %d\n", t.failed)
    if !o.apply && t.patched > 0
        println("\nRe-run with --apply to write these changes.")
    end
    if t.unknown > 0
        println("\nSome keys are not in the classification table in this script. Read what")
        println("writes them in src/ before assuming they survive the χ correction.")
    end
    return t.failed > 0 ? 1 : 0
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && exit(main(collect(ARGS)))
