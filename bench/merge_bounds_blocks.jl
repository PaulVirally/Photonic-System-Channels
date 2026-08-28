#!/usr/bin/env julia
"""
    bench/merge_bounds_blocks.jl

Assemble the `<prefix>_partial_<tag>.jld` files a block-parallel bounds run leaves
in a project directory into the single `<prefix>.jld` a monolithic run would have
written.

    julia --project=. bench/merge_bounds_blocks.jl \\
        --project /home/pvirally/projects/.../<PROJECT>/ \\
        --prefix '(16,16,16)_1ss32_(16,16,16)@(1ss32,1ss32,1ss32)' \\
        --cleanup

Runs in seconds on a login node or a 4 GB CPU allocation. Loads JLD2 and nothing
else (no CUDA, no Gila) which is the whole reason this is a separate script
rather than a mode of `compute_bounds.jl`: a merge job that spends a minute
loading a GPU stack to copy some vectors around would cost more than the split
saves.

# What a block is

`compute_bounds.jl --outer-range lo:hi --partial-suffix <tag>` runs the *whole*
front end (Gram-Schmidt, projections, the `Asym(G⁰ᵤᵤ)` augmentation when the point
qualifies) and then only the slice `lo:hi` of the outer loop over channel indices.
Index `n`'s bound depends on nothing computed at any other `n`, so B such jobs are
independent and backfill concurrently; `bench/size_bounds_jobs.jl` sizes them.

Each block writes the standard keys with `NaN` outside its range, including
`true_bounds`, since `which_bounds` is an `argmin` that a NaN wins, plus a
`partial/` group recording what it covered. Merging is therefore per-index
assembly and not arithmetic: every value in the output is a value some block
computed.

# What is checked before anything is written

The front-end-derived keys (`Γ`, `Γrs`, `χ`, `ordering_idxs`, `tau_grid`, the two
analytical-bound curves, and the whole `augment/` group) are *compared* across the
blocks rather than taken from the first one. Two blocks that disagree on them did
not run against the same input: a rerun RSVD, a different `--gamma-rtol`, a
different `--k-uu`, a stale partial left over from an earlier attempt. Merging
those would produce a file whose halves mean different things, and nothing
downstream could tell.

One disagreement is expected rather than pathological, and is worth knowing about:
`uu_eigenbasis` calls `reigen_hermitian` **without a seed**, so an augmented point
draws a fresh random sketch in every process and two blocks of it get two
(equally valid, slightly different) bases. The `augment/uu_values` comparison
catches that. Augmenting points are the ones with `m < augment_threshold`, i.e.
minutes of work, so they are never split in the first place
(`bench/size_bounds_jobs.jl` refuses to) and if you split one by hand, this is
the error you get.

# Coverage

The union of the blocks' evaluated indices must be `1:m`. A gap is an error, since
a merged file with a silent hole in it is worse than no file: `--allow-gaps` turns
it into a warning and leaves those indices `NaN`, which is the honest
representation and is what a plot will skip. Overlaps are allowed when the blocks
agree on the overlapping values, which they will, being deterministic given the
same front end.

# Options

    --project <dir>     directory holding the partials (required)
    --prefix <name>     the point's file_prefix, without `.jld` (required)
    --out <path>        write here instead of <project>/<prefix>.jld
    --allow-gaps        merge an incomplete cover, leaving the gaps NaN
    --force             overwrite an existing output file
    --cleanup           delete the partials after a successful write
    --quiet             only warnings and errors
"""

using JLD2
using Printf

# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

"Same flag grammar as bench/size_bounds_jobs.jl: `--key value` or bare `--flag`."
function parse_cli(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("expected an option starting with --, got '$(argv[i])'")
        key = argv[i][3:end]
        if i + 1 > length(argv) || startswith(argv[i+1], "--")
            opts[key] = "true"; i += 1
        else
            opts[key] = argv[i+1]; i += 2
        end
    end
    return opts
end

flag(opts, name) = get(opts, name, "false") == "true"

# --------------------------------------------------------------------------- #
# The file layout
# --------------------------------------------------------------------------- #

#=
`<prefix>_partial_<tag>.jld`, which is `partial_bounds_path` in `src/bounds.jl`.
Duplicated rather than imported, exactly as `bench/size_bounds_jobs.jl` duplicates
`_gamma_kept_count`, so that this script starts in a second without loading CUDA.
If the writer's naming changes, change it here too. `test/bounds_blocks.jl`
writes with the one and reads with the other, so a drift fails there.
=#
const PARTIAL_INFIX = "_partial_"

"The tag of a partial file of `prefix`, or `nothing` if the name is not one."
function partial_tag(name::AbstractString, prefix::AbstractString)
    head = prefix * PARTIAL_INFIX
    (startswith(name, head) && endswith(name, ".jld")) || return nothing
    tag = chop(name; head=length(head), tail=length(".jld"))
    return isempty(tag) ? nothing : tag
end

"""
The keys a finished `<prefix>.jld` holds, in the order `_compute_bounds_sr` writes
them. The merged file has to be one of these and nothing else (same keys, same
order, same shapes) so that a plot cannot tell a merged point from a monolithic
one.

Hardcoded, and checked against every partial before anything is written, so that a
key added to the writer without being added here fails loudly here instead of
being silently dropped from every merged point in the sweep.
"""
const FINAL_KEYS = [
    "Γrs", "ordering_idxs", "χ", "Γ",
    "bounds_dual_basis", "tau_grid", "opt_taus", "bounds_dual_by_tau",
    "old_analytical_bounds", "new_analytical_bounds", "true_bounds", "which_bounds",
    "augment/augmented", "augment/k_uu_requested", "augment/k_uu_effective",
    "augment/k_uu_clipped", "augment/k_uu_clip_reason", "augment/k_uu_budget_bytes",
    "augment/k_uu_returned", "augment/augment_threshold", "augment/m", "augment/m_aug",
    "augment/num_uu_kept", "augment/num_uu_dropped", "augment/dropped_cols",
    "augment/rdiag_min_ratio", "augment/uu_oversamples", "augment/uu_power_iters",
    "augment/uu_seconds", "augment/uu_values", "augment/uu_residual_idxs",
    "augment/uu_residuals",
]

"The `partial/` group a block adds on top of `FINAL_KEYS`."
const PARTIAL_KEYS = ["partial/indices", "partial/range_lo", "partial/range_hi",
                      "partial/tag", "partial/num_pos"]

"""
Keys every block must agree on exactly. Everything the front end derives: the
spectrum it was handed, the material, the ordering, the τ grid, the two analytical
curves (functions of Γ and Γrs alone) and the whole augmentation record.

`bounds_dual_basis`, `opt_taus`, `bounds_dual_by_tau`, `true_bounds` and
`which_bounds` are the per-index ones and are assembled instead; they are exactly
`FINAL_KEYS` minus this list.
"""
const SHARED_KEYS = [
    "Γrs", "ordering_idxs", "χ", "Γ", "tau_grid",
    "old_analytical_bounds", "new_analytical_bounds",
    "augment/augmented", "augment/k_uu_requested", "augment/k_uu_effective",
    "augment/k_uu_clipped", "augment/k_uu_clip_reason", "augment/k_uu_budget_bytes",
    "augment/k_uu_returned", "augment/augment_threshold", "augment/m", "augment/m_aug",
    "augment/num_uu_kept", "augment/num_uu_dropped", "augment/dropped_cols",
    "augment/rdiag_min_ratio", "augment/uu_oversamples", "augment/uu_power_iters",
    "augment/uu_values", "augment/uu_residual_idxs", "augment/uu_residuals",
]

#=
`augment/uu_seconds` is deliberately *not* shared: it is how long that block's
Asym(G⁰ᵤᵤ) solve took, which is a property of the node it landed on. It is
provenance, not input, and blocks legitimately disagree about it. The merged file
takes the first block's, and the log says so.

`augment/k_uu_budget_bytes` is a property of the node too, but it is shared rather
than excepted because it is 0 on every point that can be blocked: `clip_k_uu` only
computes a budget for a point that augments, and an augmenting point is never split
(the sizer refuses to create the blocks and `check_mergeable` refuses to merge
them). If that ever changes, this one moves down here with `uu_seconds`.
=#
const PER_INDEX_KEYS = ["bounds_dual_basis", "opt_taus", "bounds_dual_by_tau",
                        "true_bounds", "which_bounds"]

# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

"Every dataset path in `f`, groups walked, in file order."
function all_keys(f, prefix::AbstractString="")
    out = String[]
    for k in keys(prefix == "" ? f : f[prefix])
        path = prefix == "" ? k : "$(prefix)/$(k)"
        if f[path] isa JLD2.Group
            append!(out, all_keys(f, path))
        else
            push!(out, path)
        end
    end
    return out
end

"One block, read whole. These files are m-scale, so nothing here is large."
function read_block(path::AbstractString)
    data = Dict{String,Any}()
    present = String[]
    jldopen(path, "r") do f
        present = all_keys(f)
        for k in present
            data[k] = f[k]
        end
    end
    missing_keys = setdiff(vcat(FINAL_KEYS, PARTIAL_KEYS), present)
    extra_keys = setdiff(present, vcat(FINAL_KEYS, PARTIAL_KEYS))
    isempty(missing_keys) || error("""
        $(basename(path)) is missing $(length(missing_keys)) key(s) this merge needs: $(join(missing_keys, ", ")).
        A file without the partial/ group is not a block: a finished point, or a
        pre-split file, got caught by the name pattern. A file missing one of the
        standard keys was written by a job that died mid-save.""")
    isempty(extra_keys) || error("""
        $(basename(path)) holds $(length(extra_keys)) key(s) this merge does not know about: $(join(extra_keys, ", ")).
        _compute_bounds_sr has grown a key since FINAL_KEYS in
        bench/merge_bounds_blocks.jl was written. Add it there, to the per-index
        list or the shared list, whichever it is, rather than letting merged points
        quietly lose it.""")
    return (path=path, data=data)
end

# --------------------------------------------------------------------------- #
# Comparing and assembling
# --------------------------------------------------------------------------- #

"A short, readable description of where two values differ."
function describe_mismatch(a, b)
    if a isa AbstractArray && b isa AbstractArray
        size(a) == size(b) || return "shapes $(size(a)) vs $(size(b))"
        bad = findall(i -> !isequal(a[i], b[i]), eachindex(a))
        first_bad = first(bad)
        return "$(length(bad)) of $(length(a)) entries differ, first at $(first_bad): " *
               "$(a[first_bad]) vs $(b[first_bad])"
    end
    return "$(a) vs $(b)"
end

"""
    validate_shared(blocks) -> nothing

Every block must agree on every front-end-derived key. Errors on the first
disagreement with both values in the message; see the file docstring for the
reasons two blocks legitimately might.
"""
function validate_shared(blocks)
    ref = first(blocks)
    # `isequal` rather than `==`, so that NaN equals NaN: `old_analytical_bounds`
    # and friends are full of NaN by construction, and `==` on those arrays
    # returns `false` for two identical files.
    for blk in Iterators.drop(blocks, 1), k in SHARED_KEYS
        isequal(ref.data[k], blk.data[k]) || error("""
            $(basename(ref.path)) and $(basename(blk.path)) disagree on '$k', which the
            front end derives and every block therefore has to reproduce:
              $(describe_mismatch(ref.data[k], blk.data[k]))
            These blocks did not run against the same input. Check that the RSVD output on
            scratch has not been regenerated between them, that they were submitted with the
            same --gamma-rtol / --k-uu / --augment-threshold, and that no partial from an
            earlier attempt is still lying in the project directory. If this point augments
            (augment/augmented is true), the two blocks drew different Asym(G_uu) sketches and
            cannot be merged at all: augmenting points are minutes of work and must run whole.""")
    end
    return nothing
end

"""
    assemble(blocks; allow_gaps) -> (data, coverage)

The merged key/value table, plus what was covered.

Starts from the first block, which is already `NaN` (and `which_bounds == 3`,
the `argmin` of a NaN) everywhere it did not evaluate, and writes each block's
own indices over the top. Gaps therefore keep the NaN a partial run puts there
rather than a sentinel invented here, which is why `--allow-gaps` needs no special
case in the writer.
"""
function assemble(blocks; allow_gaps::Bool=false)
    ref = first(blocks)
    m = Int(ref.data["partial/num_pos"])
    for blk in blocks
        Int(blk.data["partial/num_pos"]) == m || error(
            "$(basename(ref.path)) ran at m = $m and $(basename(blk.path)) at " *
            "m = $(Int(blk.data["partial/num_pos"])). The spectral cut moved between " *
            "them, so their channel indices are not the same channels")
    end

    data = Dict{String,Any}(k => ref.data[k] for k in FINAL_KEYS)
    # Independent copies: `ref.data`'s arrays are about to be written into.
    for k in PER_INDEX_KEYS
        data[k] = copy(ref.data[k])
    end
    length(data["bounds_dual_basis"]) == m || error(
        "bounds_dual_basis has $(length(data["bounds_dual_basis"])) entries but " *
        "partial/num_pos is $m")

    owner = zeros(Int, m) # which block wrote each index, 0 for none yet
    n_true = length(data["true_bounds"])
    overlaps = 0
    for (b, blk) in enumerate(blocks)
        for n in Int.(blk.data["partial/indices"])
            1 <= n <= m || error("$(basename(blk.path)) claims index $n, outside 1:$m")
            if owner[n] != 0
                overlaps += 1
                # Two blocks that evaluated the same index must have got the same
                # answer: the front end is shared and the loop body is deterministic.
                # If they did not, one of them is from a different run and the merge
                # would be a coin toss.
                for k in PER_INDEX_KEYS
                    k in ("true_bounds", "which_bounds") && n > n_true && continue
                    a = k == "bounds_dual_by_tau" ? data[k][n, :] : data[k][n]
                    c = k == "bounds_dual_by_tau" ? blk.data[k][n, :] : blk.data[k][n]
                    # bounds_dual_by_tau is a diagnostic table whose NaN pattern
                    # depends on where the τ window happened to be, so agreement
                    # there is only required where both blocks filled the entry.
                    if k == "bounds_dual_by_tau"
                        all(i -> isnan(a[i]) || isnan(c[i]) || isequal(a[i], c[i]),
                            eachindex(a)) || error(
                            "$(basename(blocks[owner[n]].path)) and $(basename(blk.path)) " *
                            "both evaluated index $n and disagree on '$k'")
                    else
                        isequal(a, c) || error(
                            "$(basename(blocks[owner[n]].path)) and $(basename(blk.path)) " *
                            "both evaluated index $n and disagree on '$k': $a vs $c")
                    end
                end
                continue
            end
            owner[n] = b
            data["bounds_dual_basis"][n] = blk.data["bounds_dual_basis"][n]
            data["opt_taus"][n] = blk.data["opt_taus"][n]
            data["bounds_dual_by_tau"][n, :] = blk.data["bounds_dual_by_tau"][n, :]
            if n <= n_true
                data["true_bounds"][n] = blk.data["true_bounds"][n]
                data["which_bounds"][n] = blk.data["which_bounds"][n]
            end
        end
    end

    gaps = findall(==(0), owner)
    if !isempty(gaps)
        runs = index_runs(gaps)
        msg = "$(length(gaps)) of the $m channel indices are covered by no block: " *
              join(["$(first(r)):$(last(r))" for r in Iterators.take(runs, 10)], ", ") *
              (length(runs) > 10 ? ", ..." : "")
        allow_gaps || error("""
            $msg
            Either a block has not finished (or failed: check its log and resubmit just
            that --outer-range), or the blocks were sized against a different m than the
            run used. --allow-gaps merges anyway and leaves those indices NaN.""")
        @warn "$msg. --allow-gaps: they stay NaN in the merged file"
    end
    return (data=data, m=m, owner=owner, gaps=gaps, overlaps=overlaps)
end

"`[1,2,3,7,8]` -> `[1:3, 7:8]`, for readable gap reporting."
function index_runs(idxs::AbstractVector{Int})
    runs = UnitRange{Int}[]
    isempty(idxs) && return runs
    lo = prev = first(idxs)
    for i in Iterators.drop(idxs, 1)
        if i == prev + 1
            prev = i
        else
            push!(runs, lo:prev); lo = prev = i
        end
    end
    push!(runs, lo:prev)
    return runs
end

# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

"""
    write_merged(path, data)

`FINAL_KEYS` in `_compute_bounds_sr`'s own write order, and nothing else. The
`partial/` group is dropped: the merged file is a finished point and has to be
indistinguishable from one.
"""
function write_merged(path::AbstractString, data::Dict{String,Any})
    mkpath(dirname(abspath(path)))
    jldopen(path, "w") do f
        for k in FINAL_KEYS
            f[k] = data[k]
        end
    end
    return path
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

function main(argv::Vector{String})
    opts = parse_cli(argv)
    project = get(opts, "project", "")
    prefix = get(opts, "prefix", "")
    isempty(project) && error("--project <dir> is required")
    isempty(prefix) && error("--prefix <name> is required (the point's file_prefix, without .jld)")
    isdir(project) || error("--project is not a directory: $project")
    quiet = flag(opts, "quiet")
    say(args...) = quiet ? nothing : println(args...)

    tags = Tuple{String,String}[] # (tag, path)
    for name in sort(readdir(project))
        tag = partial_tag(name, prefix)
        tag === nothing && continue
        push!(tags, (tag, joinpath(project, name)))
    end
    isempty(tags) && error("""
        No $(prefix)$(PARTIAL_INFIX)<tag>.jld in $project.
        Check --prefix against the point's file_prefix (it contains parentheses and
        commas; quote it), and that this is the *project* directory and not scratch.""")

    say("Merging $(length(tags)) block(s) of $(prefix)")
    blocks = [read_block(path) for (_, path) in tags]
    validate_shared(blocks)

    out = get(opts, "out", joinpath(project, "$(prefix).jld"))
    if isfile(out) && !flag(opts, "force")
        error("""
            $out already exists. A monolithic run of this point, or an earlier merge, is
            already there and this would replace it. Pass --force to overwrite.""")
    end

    asm = assemble(blocks; allow_gaps=flag(opts, "allow-gaps"))
    for (i, blk) in enumerate(blocks)
        idxs = Int.(blk.data["partial/indices"])
        owned = count(==(i), asm.owner)
        span = isempty(idxs) ? "(empty)" : "$(minimum(idxs)):$(maximum(idxs))"
        say(@sprintf("  %-40s %-16s %6d index/indices%s", basename(blk.path), span,
                     length(idxs), owned == length(idxs) ? "" : "  ($owned kept)"))
    end
    asm.overlaps > 0 && @warn "$(asm.overlaps) index/indices were evaluated by more than one block; they agree, and the first block's values are kept"

    write_merged(out, asm.data)
    covered = asm.m - length(asm.gaps)
    say("Wrote $out: $(covered)/$(asm.m) channel indices from $(length(blocks)) block(s)" *
        (isempty(asm.gaps) ? "" : ", $(length(asm.gaps)) left NaN"))
    say("  augment/augmented = $(asm.data["augment/augmented"]), m_aug = $(asm.data["augment/m_aug"])")

    if flag(opts, "cleanup")
        for blk in blocks
            rm(blk.path)
        end
        say("Deleted $(length(blocks)) partial file(s)")
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
