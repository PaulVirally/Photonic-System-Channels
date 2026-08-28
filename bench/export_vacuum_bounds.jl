#!/usr/bin/env julia
"""
    bench/export_vacuum_bounds.jl

Pull the *small* per-separation vectors out of the production RSVD/bounds JLDs
and write them to one bundle, so the analytical (Eq18/Eq19) bounds can be
evaluated and plotted off-cluster without moving the multi-GB RSVD outputs.

# Hard constraint

This script runs on the cluster's checkout, which is an older state of the repo
with an older `GilaElectromagnetics` pin. It therefore depends on **JLD2 and the
stdlib only**: no `using PhotonicSystemChannels`, no `include` of anything in
`src/`. Everything it needs to know about the file layout is duplicated here.

# What it reads

Per separation, two files can exist and both are checked:

  * the RSVD output, `<scratch>/<project>/<prefix>.jld`, written by
    `src/rsvd.jl`. Keys taken: `RS/D` (the vacuum sender->receiver singular
    values Γrs), `UR_asym/D` (every eigenvalue of `(-G⁰ᵤᵣ)ᵃ`, unsorted as
    written) and `UR_asym/num_pos`.
  * the bounds output, `<projects>/<project>/<prefix>.jld`, written by
    `src/bounds.jl::_compute_bounds_sr`. It carries its own copies of the two
    spectra under `Γrs` and `Γ`, plus `true_bounds`, `bounds_dual_basis`,
    `which_bounds`, `χ` and the `augment/` group. Bounds exist for some sizes
    and separations and not others; whatever is there is taken and the rest is
    skipped silently.

Both directories are searched for both layouts, because the boundsonly workflow
can leave RS data in either.

# What the analytical bounds need

`src/bounds.jl` computes them from exactly two vectors and the material factor
ζ = |χ|²/ℑχ:

    Eq18 (old form): κ  = ζ² Γrs²          -> κ ≥ 1 ? 1 : sqrt(4κ)/(1+κ)
    Eq19 (new form): κ̃ = ζ max(Γ, 0)       -> 2κ̃ ≥ 1 ? 1 : sqrt(4κ̃|1-κ̃|)

χ is supplied at plot time, so only Γrs and Γ travel. Both are length ~= the
RSVD component count (a few thousand doubles), which is why this is kilobytes a
point instead of gigabytes.

# Usage

    julia --project=. bench/export_vacuum_bounds.jl [options]

    --out PATH             bundle path (default under exports/, dirs created)
    --scratch-root PATH    parent of the per-project scratch dirs
    --project-root PATH    parent of the per-project bounds dirs
    --project SIZE=NAME    override one size's project dir name (repeatable)
    --only SIZE[,SIZE...]  restrict to these size labels
    --verbose              one line per file read

A missing directory is reported and skipped, never fatal: the project names
below are transcribed by hand and may be slightly off. The manifest printed at
the end says, per size, how many separations were found and how many carry
bounds; a size showing 0 found is the signal to re-run with `--project`.
"""

using JLD2
using Printf

# The five cube sizes of the arxivV3 campaign. Keys are the labels used in the
# bundle and in the plot legend; values are the directory basenames under both
# roots.
const DEFAULT_PROJECTS = [
    "0p25" => "narval_Ge1000_arxivV3_0p25x0p25x0p25_3072comps_50oversamples_q6_32scale",
    "0p5"  => "narval_Ge1000_arxivV3_0p5x0p5x0p5_4000comps_50oversamples_q6_32scale",
    "1"    => "narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale",
    "2"    => "narval_Ge1000_arxivV3_2x2x2_4000comps_50oversamples_q6_aniso-64-n32-n32scale",
    "4"    => "narval_Ge1000_arxivV3_4x4x4_4000comps_50oversamples_q6_aniso-128-n8-n8scale",
]

const DEFAULT_SCRATCH_ROOT = "/home/pvirally/scratch/Photonic-System-Channels"
const DEFAULT_PROJECT_ROOT = "/home/pvirally/projects/def-smolesky/pvirally/Photonic-System-Channels/projects"
const DEFAULT_OUT = "/home/pvirally/Photonic-System-Channels/exports/vacuum_bounds_export.jld2"

# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------

# `src/SMRSystems.jl::file_prefix` builds
#     <sender>__<receiver>__<num>ss<den>__<universe>[__refF<f>T<t>]
# for the SR case and inserts a mediator size for SMR. Rather than count the
# fields, find the one that looks like a separation; that is unambiguous either
# way and survives the refinement suffix.
const SEP_RE = r"^(-?\d+)ss(\d+)$"

# `nothing` for anything that is not a sweep point: partial blocks, the augmented
# basis experiment's output, and any file whose name has no `<num>ss<den>` field.
function parse_prefix(prefix::AbstractString)
    occursin("_partial_", prefix) && return nothing
    endswith(prefix, "_augmented_basis") && return nothing
    parts = split(prefix, "__")
    length(parts) >= 3 || return nothing
    idx = findfirst(p -> occursin(SEP_RE, p), parts)
    idx === nothing && return nothing
    m = match(SEP_RE, parts[idx])
    num, den = parse(Int, m.captures[1]), parse(Int, m.captures[2])
    den == 0 && return nothing
    return (sep = String(parts[idx]), num = num, den = den, wl = num / den,
            sender = String(parts[1]),
            receiver = String(parts[idx - 1]),
            tail = join(parts[(idx + 1):end], "__"))
end

# --------------------------------------------------------------------------
# Defensive JLD2 reads
# --------------------------------------------------------------------------

# `haskey` on a nested path and the read itself both go through this: a key that
# is absent, a group that is not there, a half-written file mid-job all come
# back as `nothing` rather than taking the walk down with them.
function tryread(jld, key::AbstractString)
    try
        haskey(jld, key) || return nothing
        return jld[key]
    catch
        return nothing
    end
end

# First key that reads. The alternatives are the historical/output layouts of
# the same quantity: `RS/D` is what `_run_rsvdvals` writes into the scratch
# file, `Γrs` is the copy `_compute_bounds_sr` puts in the bounds file.
function tryread_any(jld, keys::Vector{String})
    for k in keys
        v = tryread(jld, k)
        v === nothing || return (v, k)
    end
    return (nothing, "")
end

asvec(x) = x === nothing ? nothing : collect(Float64, vec(x))
asivec(x) = x === nothing ? nothing : collect(Int, vec(x))

# --------------------------------------------------------------------------
# One file
# --------------------------------------------------------------------------

# Everything of interest a single .jld can hold, whichever of the two kinds it
# is. Fields left `nothing` are merged over from the other kind's file.
mutable struct Point
    sep::String
    num::Int
    den::Int
    wl::Float64
    sender::String
    receiver::String
    gamma_rs::Union{Nothing,Vector{Float64}}
    gamma::Union{Nothing,Vector{Float64}}
    num_pos::Union{Nothing,Int}
    chi::Union{Nothing,ComplexF64}
    true_bounds::Union{Nothing,Vector{Float64}}
    dual::Union{Nothing,Vector{Float64}}
    which::Union{Nothing,Vector{Int}}
    augmented::Union{Nothing,Bool}
    aug_m::Union{Nothing,Int}
    aug_m_aug::Union{Nothing,Int}
    rs_source::String
    bounds_source::String
end

Point(p) = Point(p.sep, p.num, p.den, p.wl, p.sender, p.receiver,
                 nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                 nothing, nothing, nothing, "", "")

set!(pt, field, v) = (v === nothing || getfield(pt, field) !== nothing) ? nothing :
                     setfield!(pt, field, v)

function harvest!(pt::Point, path::AbstractString, verbose::Bool)
    jld = nothing
    try
        jld = jldopen(path, "r")
    catch err
        @warn "could not open $path" err
        return false
    end
    got_rs, got_bounds = false, false
    try
        # The vacuum S->R singular values. `RS/D` in the RSVD output, `Γrs` in
        # the bounds output; identical content, whichever turns up first wins.
        grs, _ = tryread_any(jld, ["RS/D", "Γrs", "Gamma_rs"])
        # Every eigenvalue of the Asym(G⁰ᵤᵣ) decomposition, as written (the
        # descending sort is `load_bounds_inputs`' and is redone at plot time).
        g, _ = tryread_any(jld, ["UR_asym/D", "Γ", "Gamma"])
        np, _ = tryread_any(jld, ["UR_asym/num_pos", "partial/num_pos"])
        chi, _ = tryread_any(jld, ["χ", "chi"])

        set!(pt, :gamma_rs, asvec(grs))
        set!(pt, :gamma, asvec(g))
        np === nothing || set!(pt, :num_pos, Int(np))
        chi === nothing || set!(pt, :chi, ComplexF64(chi))
        got_rs = grs !== nothing && g !== nothing

        tb = asvec(tryread(jld, "true_bounds"))
        du = asvec(tryread(jld, "bounds_dual_basis"))
        wb = asivec(tryread(jld, "which_bounds"))
        set!(pt, :true_bounds, tb)
        set!(pt, :dual, du)
        set!(pt, :which, wb)
        got_bounds = tb !== nothing

        a = tryread(jld, "augment/augmented")
        a === nothing || set!(pt, :augmented, Bool(a))
        am = tryread(jld, "augment/m")
        am === nothing || set!(pt, :aug_m, Int(am))
        ama = tryread(jld, "augment/m_aug")
        ama === nothing || set!(pt, :aug_m_aug, Int(ama))

        # `num_pos` is not a key of the bounds output, but the dual has exactly
        # one entry per kept channel, so it names the same m.
        if pt.num_pos === nothing && du !== nothing
            pt.num_pos = length(du)
        end
    finally
        close(jld)
    end
    got_rs && isempty(pt.rs_source) && (pt.rs_source = path)
    got_bounds && isempty(pt.bounds_source) && (pt.bounds_source = path)
    verbose && @printf("    %-70s rs=%-5s bounds=%-5s\n", basename(path), got_rs, got_bounds)
    return got_rs || got_bounds
end

# --------------------------------------------------------------------------
# One size
# --------------------------------------------------------------------------

function scan_size(label::AbstractString, project::AbstractString,
                   dirs::Vector{String}, verbose::Bool)
    points = Dict{String,Point}()
    seen_dirs = String[]
    for dir in dirs
        if !isdir(dir)
            println("  [miss] $dir")
            continue
        end
        push!(seen_dirs, dir)
        files = try
            sort(filter(f -> endswith(f, ".jld"), readdir(dir)))
        catch err
            @warn "could not list $dir" err
            String[]
        end
        println("  [ok]   $dir  ($(length(files)) .jld)")
        for f in files
            info = parse_prefix(f[1:(end - 4)])
            info === nothing && continue
            pt = get!(points, info.sep, Point(info))
            harvest!(pt, joinpath(dir, f), verbose)
        end
    end
    return (label = label, project = project, dirs = seen_dirs, points = points)
end

# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

function write_bundle(out::AbstractString, results)
    mkpath(dirname(out))
    jldopen(out, "w") do jld
        jld["schema"] = "vacuum_bounds_export/1"
        # Eq18/Eq19 are functions of these two vectors and ζ only; recorded so
        # the plot side cannot silently pair them with the wrong formula.
        jld["eq18"] = "kappa = zeta^2 * Gamma_rs.^2; b = kappa>=1 ? 1 : sqrt(4kappa)/(1+kappa)"
        jld["eq19"] = "kappat = zeta * max(Gamma,0); b = 2kappat>=1 ? 1 : sqrt(4kappat*abs(1-kappat))"
        sizes = String[]
        for r in results
            isempty(r.points) && continue
            push!(sizes, r.label)
            jld["$(r.label)/project"] = r.project
            jld["$(r.label)/dirs"] = r.dirs
            seps = sort(collect(keys(r.points)); by = s -> r.points[s].wl)
            jld["$(r.label)/separations"] = seps
            jld["$(r.label)/sep_wl"] = [r.points[s].wl for s in seps]
            for s in seps
                pt = r.points[s]
                g = "$(r.label)/$(s)/"
                jld[g * "sep_num"] = pt.num
                jld[g * "sep_den"] = pt.den
                jld[g * "sep_wl"] = pt.wl
                jld[g * "sender_cells"] = pt.sender
                jld[g * "receiver_cells"] = pt.receiver
                jld[g * "rs_source"] = pt.rs_source
                jld[g * "bounds_source"] = pt.bounds_source
                pt.gamma_rs === nothing || (jld[g * "Gamma_rs"] = pt.gamma_rs)
                pt.gamma === nothing || (jld[g * "Gamma"] = pt.gamma)
                pt.num_pos === nothing || (jld[g * "num_pos"] = pt.num_pos)
                pt.chi === nothing || (jld[g * "chi"] = pt.chi)
                pt.true_bounds === nothing || (jld[g * "true_bounds"] = pt.true_bounds)
                pt.dual === nothing || (jld[g * "bounds_dual_basis"] = pt.dual)
                pt.which === nothing || (jld[g * "which_bounds"] = pt.which)
                pt.augmented === nothing || (jld[g * "augmented"] = pt.augmented)
                pt.aug_m === nothing || (jld[g * "augment_m"] = pt.aug_m)
                pt.aug_m_aug === nothing || (jld[g * "augment_m_aug"] = pt.aug_m_aug)
            end
        end
        jld["sizes"] = sizes
    end
    return out
end

# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

function manifest(results, out::AbstractString)
    println()
    println("MANIFEST")
    @printf("  %-6s %-6s %-8s %-8s %-10s %-10s %s\n",
            "size", "seps", "with_RS", "with_bd", "sep_min", "sep_max", "pos_frac")
    for r in results
        seps = sort(collect(keys(r.points)); by = s -> r.points[s].wl)
        if isempty(seps)
            @printf("  %-6s %-6d %-8s %-8s %-10s %-10s %s\n",
                    r.label, 0, "-", "-", "-", "-", "NOTHING FOUND (check --project)")
            continue
        end
        pts = [r.points[s] for s in seps]
        n_rs = count(p -> p.gamma_rs !== nothing && p.gamma !== nothing, pts)
        n_bd = count(p -> p.true_bounds !== nothing, pts)
        # The share of Γ that is positive. Eq19 is NaN wherever Γ ≤ 0, so a size
        # whose fraction is ~0 has had its sign flipped somewhere and its Eq19
        # curve will be empty; ~0.6 is the healthy value the RSVD logs report.
        fr = Float64[]
        for p in pts
            p.gamma === nothing && continue
            push!(fr, count(>(0.0), p.gamma) / max(length(p.gamma), 1))
        end
        frs = isempty(fr) ? "-" : @sprintf("%.3f-%.3f", minimum(fr), maximum(fr))
        @printf("  %-6s %-6d %-8d %-8d %-10.4f %-10.4f %s\n",
                r.label, length(seps), n_rs, n_bd,
                minimum(p -> p.wl, pts), maximum(p -> p.wl, pts), frs)
    end
    sz = filesize(out)
    println()
    @printf("  bundle: %s  (%.2f MiB)\n", out, sz / 2^20)
    return nothing
end

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

function main(args::Vector{String})
    out = DEFAULT_OUT
    scratch_root = DEFAULT_SCRATCH_ROOT
    project_root = DEFAULT_PROJECT_ROOT
    overrides = Dict{String,String}()
    only = String[]
    verbose = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out"; out = args[i+1]; i += 2
        elseif a == "--scratch-root"; scratch_root = args[i+1]; i += 2
        elseif a == "--project-root"; project_root = args[i+1]; i += 2
        elseif a == "--project"
            kv = split(args[i+1], "="; limit = 2)
            length(kv) == 2 || error("--project wants SIZE=NAME, got $(args[i+1])")
            overrides[String(kv[1])] = String(kv[2]); i += 2
        elseif a == "--only"; only = String.(split(args[i+1], ",")); i += 2
        elseif a == "--verbose"; verbose = true; i += 1
        elseif a in ("-h", "--help")
            println("usage: julia --project=. bench/export_vacuum_bounds.jl " *
                    "[--out PATH] [--scratch-root PATH] [--project-root PATH] " *
                    "[--project SIZE=NAME]... [--only SIZE,SIZE] [--verbose]")
            println("sizes: ", join(first.(DEFAULT_PROJECTS), ", "))
            return
        else
            error("unknown argument $a")
        end
    end

    println("scratch root: $scratch_root")
    println("project root: $project_root")
    println("output      : $out")
    println()

    results = []
    for (label, default_name) in DEFAULT_PROJECTS
        (isempty(only) || label in only) || continue
        project = get(overrides, label, default_name)
        println("size $label -> $project")
        dirs = [joinpath(scratch_root, project), joinpath(project_root, project)]
        push!(results, scan_size(label, project, dirs, verbose))
    end

    write_bundle(out, results)
    manifest(results, out)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
