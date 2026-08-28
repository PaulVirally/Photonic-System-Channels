#!/usr/bin/env julia
"""
    bench/refined_near_field.jl

What does the gap refinement (`src/refinement.jl`) do to the *physics*, the
per-channel bounds on σₙ(Pᵣₛ), at the separations it was introduced for?

The quarter-wavelength cube, (8,8,8) cells at scale 1//32, is small enough that
the whole pipeline (Green blocks → exact `Asym(G⁰ᵤᵣ)` eigendecomposition →
bounds) runs on a laptop CPU, so the same point can be computed twice, once with
`refine_gap=true` and once with `refine_gap=false`, and the two bound vectors
compared directly. Everything here goes through the production entry points
(`_generate_green_sr`, `_save_ur_asym`/`_run_rsvdvals`, `load_bounds_inputs`,
`bounds_from_spectrum`); nothing re-implements any physics.

    julia --project=. bench/refined_near_field.jl --out <dir>
    julia --project=. bench/refined_near_field.jl --out <dir> --quick
    julia --project=. bench/refined_near_field.jl --out <dir> --report-only

Three measurements, selected with `--gaps` / `--fconv` / `--quick`:

  1. Refined against unrefined at gaps g = 1..5 coarse x-cells (separations
     1//32 … 5//32 λ), plus g = 6 as the no-op anchor, where `gap_refinement`
     returns `nothing` and the two runs must agree to the last bit.
  2. Mesh convergence at g = 1: the table's (factor, thickness) = (6, 6) against
     other factors at the same separation, to see whether f = 6 is converged.
     The doubled f = 12 mesh is built here through the exported refinement
     surface (`GapRefinement` + `refine_body` + the field-wise `SMRSystem`
     constructor) rather than by touching `GAP_REFINEMENT_TABLE`.
  3. The proximity warning: Gila warns when a target and a source region are
     closer than `MIN_GAP_CELLS` cells of their coarser scale. It must fire on
     the unrefined near runs and must not fire on the refined ones.

# The exact path, and why it has to be forced at g = 1

`use_dense_path` caps the exact eigensolve at `DENSE_EXACT_MAX_N_U` = 12,288 and
`DENSE_EXACT_MAX_N_R` = 6,144. Every configuration here is under both except the
g = 1 refined mesh, which is N_u = 14,592 (N_r = 7,296): the f = 6 slab is six of
the body's eight coarse cells, so a body goes from 1,536 to 7,296 degrees of
freedom. An RSVD there would compare an approximate spectrum against an exact
one, so `--max-dense-nu` / `--max-dense-nr` raise the caps for this script alone
and every point on the table is exact. `--rsvd-fallback` gives up on that and
takes the production path instead, for meshes too large to diagonalize.

# Which channels are evaluated

The outer loop of `bounds_from_spectrum` costs one m × m eigendecomposition per
(channel, τ) pair, so a full sweep is O(m⁴) and m here is over a thousand. The
default is therefore a *ladder*: every channel up to `--ladder-head`, then
`--ladder-points` geometrically spaced up to m, with m itself always included.
Both members of a pair are evaluated on the *same* index set, derived from the
smaller of the two m, so the per-channel comparison is like for like. The trace
is then reported two ways: `trace_exact` over the sampled channels only, and
`trace_interp`, the piecewise-linear interpolation of the sampled curve summed
over 1:m, which is what to read as "the trace". `--outer-mode full` evaluates
every channel and makes `trace_exact` the real thing; budget days for it.

`m` here is the count `load_bounds_inputs` keeps after its `--gamma-rtol` cut,
not the stored positive count: at this size the cut is severe (386 kept of 1,536
stored at 3//16 λ), and a ladder built on the stored count would sample almost
nothing.

# --k-uu 0 by default

`DEFAULT_AUGMENT_THRESHOLD` is 1,000 and the kept m here is a few hundred, so
production defaults would switch the `Asym(G⁰ᵤᵤ)` augmentation *on* for these
points. That is wrong for this measurement twice over: the augmentation's sketch
is drawn from the global RNG, so two runs of one point differ at the 1e-9 level
and the no-op anchor can no longer be checked bit for bit; and a refined and an
unrefined mesh whose kept m straddle the threshold would be compared across two
different algorithms rather than across two meshes. This script therefore
defaults to `--k-uu 0`, the documented way to reproduce the pre-augmentation
output exactly, and seeds the RNG regardless so `--k-uu 512` is reproducible too.
"""

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using GilaElectromagnetics
using Dates
using JLD2
using LinearAlgebra
using LinearMaps
using Logging
using Printf
using Random
using Statistics

const CHI = 17.06132654701751 + 0.29117345im # Germanium, ζ = 1000
const CELLS = (8, 8, 8)
const SCALE = 1//32
const DESIGN = SMRVolumeSymbol[Sender, Receiver]

# ---------------------------------------------------------------------------
# CLI

function parse_cli(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("expected an option starting with --, got '$(argv[i])'")
        key = argv[i][3:end]
        if i + 1 > length(argv) || startswith(argv[i+1], "--")
            opts[key] = "true"
            i += 1
        else
            opts[key] = argv[i+1]
            i += 2
        end
    end
    return opts
end

optstr(o, k, d) = get(o, k, d)
optint(o, k, d) = haskey(o, k) ? parse(Int, o[k]) : d
optbool(o, k, d) = haskey(o, k) ? o[k] == "true" : d
optints(o, k, d) = haskey(o, k) ? (o[k] == "none" ? Int[] : parse.(Int, split(o[k], ','))) : d

# ---------------------------------------------------------------------------
# Warning capture. Gila's proximity complaint is the check that says the
# derivation in src/refinement.jl is doing its job, so every stage runs under a
# logger that passes messages through and keeps the warnings.

struct WarnCollector <: AbstractLogger
    inner::AbstractLogger
    warnings::Vector{String}
end
Logging.min_enabled_level(lg::WarnCollector) = Logging.min_enabled_level(lg.inner)
Logging.shouldlog(lg::WarnCollector, args...) = Logging.shouldlog(lg.inner, args...)
Logging.catch_exceptions(lg::WarnCollector) = Logging.catch_exceptions(lg.inner)
function Logging.handle_message(lg::WarnCollector, level, message, _mod, group, id, file, line; kwargs...)
    level >= Logging.Warn && push!(lg.warnings, string(message))
    return Logging.handle_message(lg.inner, level, message, _mod, group, id, file, line; kwargs...)
end

is_proximity(w) = occursin("separated by", w) || occursin("MIN_GAP", w)

# Wall clock and the resident high-water mark either side of a stage. `maxrss` is
# monotone over the process, so the delta is a lower bound on what the stage
# itself took; the absolute value is reported alongside it.
function timed_stage(f, warnings::Vector{String})
    rss0 = Sys.maxrss()
    t0 = time_ns()
    out = with_logger(WarnCollector(global_logger(), warnings)) do
        f()
    end
    secs = (time_ns() - t0) / 1e9
    return (value=out, seconds=secs, rss_delta=Sys.maxrss() - rss0, rss_peak=Sys.maxrss())
end

# ---------------------------------------------------------------------------
# Systems

base_system(gap::Int) = SMRSystem(CELLS, (gap//32, 0//1, 0//1), CELLS, DESIGN, SCALE, CHI;
                                  refine_gap=false)

# A system on a hand-picked (factor, thickness) at the given gap, bypassing
# `GAP_REFINEMENT_TABLE` without editing it. The field-wise `SMRSystem`
# constructor takes the two meshes directly, which is the whole public surface
# this needs: the sender faces the gap across its high-x surface and the
# receiver across its low-x one, exactly as the production constructor does it.
function custom_system(gap::Int, factor::Int, thickness::Int)
    b = base_system(gap)
    ref = GapRefinement(gap, factor, thickness)
    return SMRSystem(sender(b), nothing, receiver(b), design(b), DESIGN, CHI,
                     refine_body(sender(b), ref, :high),
                     refine_body(receiver(b), ref, :low), ref)
end

function system_for(spec::NamedTuple)
    spec.kind === :unrefined && return base_system(spec.gap)
    spec.kind === :refined &&
        return SMRSystem(CELLS, (spec.gap//32, 0//1, 0//1), CELLS, DESIGN, SCALE, CHI;
                         refine_gap=true)
    return custom_system(spec.gap, spec.factor, spec.thickness)
end

n_u(smr) = dof_length(sender_mesh(smr)) + dof_length(receiver_mesh(smr))

# ---------------------------------------------------------------------------
# Stages

function case_dirs(root::String, tag::String)
    dirs = (preload=joinpath(root, tag, "preload"),
            project=joinpath(root, tag, "project"),
            scratch=joinpath(root, tag, "scratch"))
    foreach(mkpath, values(dirs))
    return dirs
end

function run_front(root::String, tag::String, smr::SMRSystem, params::RSVDParams,
                   max_dense_nu::Int, max_dense_nr::Int, gamma_rtol::Float64,
                   spectrum::Bool=true)
    dirs = case_dirs(root, tag)
    env = ComputeEnvironment(dirs.preload, dirs.project, dirs.scratch, GPUChoice(false, -1))
    warnings = String[]

    println("  [$(tag)] Green blocks, N_u = $(n_u(smr))")
    green = timed_stage(warnings) do
        PSC._generate_green_sr(env, smr)
    end
    @printf("  [%s] Green: %.1f s, maxrss %.2f GiB\n", tag, green.seconds, green.rss_peak / 2^30)

    # `--no-spectrum` stops here. The exact eigensolve is the expensive stage and
    # a mesh being probed only for the positivity of its operators does not need
    # it; `kept` comes back as zero and the bounds stage is skipped.
    if !spectrum
        none = (value=nothing, seconds=NaN, rss_delta=UInt64(0), rss_peak=Sys.maxrss())
        return (env=env, warnings=warnings, green=green, ur=none, rs=none,
                meta=(num_pos=0, exact=false, nspec=0, kept=0), jld_path="")
    end

    println("  [$(tag)] Asym(G⁰ᵤᵣ) spectrum")
    ur = timed_stage(warnings) do
        PSC._save_ur_asym(env, smr, params; max_dense_N_u=max_dense_nu)
    end
    PSC.reclaim_host_pools!()
    rs = timed_stage(warnings) do
        PSC._run_rsvdvals(env, smr, params, "RS/"; max_dense_N_r=max_dense_nr)
    end
    @printf("  [%s] RSVD: %.1f s (UR %.1f + RS %.1f), maxrss %.2f GiB\n", tag,
            ur.seconds + rs.seconds, ur.seconds, rs.seconds, rs.rss_peak / 2^30)

    # `kept` is the m the bounds actually run on: `load_bounds_inputs` cuts the
    # stored positive block at `gamma_rtol`, and at these sizes that cut is
    # brutal (1,536 stored positives against 386 kept at 3//16 λ). The channel
    # ladder has to be built against it, not against the stored count.
    jld_path = joinpath(dirs.scratch, "$(file_prefix(smr)).jld")
    meta = jldopen(jld_path, "r") do jld
        stored = Int(jld["UR_asym/num_pos"])
        Γ = sort(Array(jld["UR_asym/D"]); rev=true)
        (num_pos=stored, exact=Bool(jld["UR_asym/exact"]), nspec=length(Γ),
         kept=PSC._gamma_kept_count(Γ, stored, gamma_rtol))
    end
    return (env=env, warnings=warnings, green=green, ur=ur, rs=rs, meta=meta,
            jld_path=jld_path)
end

#=
Is Asym(G⁰ᵤᵤ) still positive semi-definite on this mesh?

It has to be: it is the radiated-power operator of the universe, and the bounds'
constraint C(τ) = ζ⁻¹(Πₛ + (1−τ)Πᵣ) + τ(−G⁰ᵤᵣ)ᵃ₊ + (G⁰ᵤᵤ)ᵃ is a sum of positive
semi-definite terms of which this is the only one that is not positive by
construction. `psd_pencil_whitener` refuses a C with a negative eigenvalue, so a
mesh that loses this loses the bound outright rather than shifting it.

The probe is `reigen_hermitian` on ±the operator (through the exported
`uu_eigenbasis`), which costs a handful of matvecs rather than the dense build the
eigensolve would want. What is reported is λmin/λmax: a mesh in good standing has
λmin at the roundoff floor of λmax, and one that has lost positivity has a ratio
of order one.
=#
function extreme_eigs(env::ComputeEnvironment, op, k::Int, power_iters::Int)
    A = PSC.asym_self(op)
    hi = uu_eigenbasis(env, A, k; power_iters=power_iters).values
    lo = uu_eigenbasis(env, -A, k; power_iters=power_iters).values
    return (lambda_max=first(hi), lambda_min=-first(lo), ratio=-first(lo) / first(hi))
end

# Both the universe and one body on its own. The universe operator is the block
# assembly across the gap; the receiver self block is a single `GlaCmpOprVac` over
# one body's own regions, with no gap and no assembly in it. Reporting the two
# separates a defect in the cross-body assembly from one in the cross-scale mesh
# a refined body carries inside itself.
function psd_probe(env::ComputeEnvironment, smr::SMRSystem, k::Int, power_iters::Int)
    uu = extreme_eigs(env, load_green_function(env, smr, [Sender, Receiver],
                                               [Sender, Receiver]), k, power_iters)
    rr = extreme_eigs(env, load_green_function(env, smr, Receiver, Receiver), k, power_iters)
    return (lambda_max=uu.lambda_max, lambda_min=uu.lambda_min, ratio=uu.ratio,
            rr_lambda_max=rr.lambda_max, rr_lambda_min=rr.lambda_min, rr_ratio=rr.ratio)
end

function run_bounds(front, smr::SMRSystem, idxs::Vector{Int}, gamma_rtol::Float64,
                    k_uu::Int, augment_threshold::Int, rng_seed::Int)
    warnings = String[]
    stage = timed_stage(warnings) do
        # The Asym(G⁰ᵤᵤ) augmentation draws its sketch from the global RNG, so a
        # nonzero --k-uu makes two runs of the same point differ at the 1e-9
        # level. Seeding here is what lets the no-op anchor be checked bit for
        # bit rather than to a tolerance.
        Random.seed!(rng_seed)
        inputs = load_bounds_inputs(front.env, smr; gamma_rtol=gamma_rtol, panel_mode=false)
        ns = filter(n -> 1 <= n <= inputs.num_pos, idxs)
        res = bounds_from_spectrum(front.env, smr, inputs.Γ, inputs.Vur_asym, inputs.Γrs;
                                   num_pos=inputs.num_pos, outer_indices=ns,
                                   nan_unevaluated=true, k_uu=k_uu,
                                   augment_threshold=augment_threshold)
        (inputs=inputs, res=res, ns=ns)
    end
    append!(front.warnings, warnings)
    return stage
end

# ---------------------------------------------------------------------------
# Channel index sets

function ladder_indices(m::Int, head::Int, points::Int)
    idxs = collect(1:min(head, m))
    if m > head
        lo, hi = log(head + 1), log(m)
        for t in range(lo, hi; length=points)
            push!(idxs, clamp(round(Int, exp(t)), 1, m))
        end
        push!(idxs, m)
    end
    return sort!(unique!(idxs))
end

# The sampled curve summed over 1:m, linear between samples. `ns` is sorted and
# ends at m, so every integer in 1:m is covered by exactly one segment or is the
# final sample.
function interp_trace(ns::Vector{Int}, vs::Vector{Float64})
    isempty(ns) && return NaN
    total = 0.0
    for i in 1:(length(ns) - 1)
        a, b, va, vb = ns[i], ns[i+1], vs[i], vs[i+1]
        for n in a:(b-1)
            total += va + (n - a) / (b - a) * (vb - va)
        end
    end
    return total + vs[end]
end

# ---------------------------------------------------------------------------
# One measured point

rtol_of(opts) = haskey(opts, "gamma-rtol") ? parse(Float64, opts["gamma-rtol"]) :
                PSC.DEFAULT_GAMMA_RTOL

function measure(root::String, spec::NamedTuple, params::RSVDParams, opts)
    smr = system_for(spec)
    tag = spec.tag
    front = run_front(root, tag, smr, params,
                      optint(opts, "max-dense-nu", 32_768),
                      optint(opts, "max-dense-nr", 16_384), rtol_of(opts),
                      !optbool(opts, "no-spectrum", false))
    psd = nothing
    if optbool(opts, "psd-probe", true)
        psd = psd_probe(front.env, smr, optint(opts, "psd-k", 4),
                        optint(opts, "psd-power-iters", 6))
        @printf("  [%s] Asym(G⁰ᵤᵤ): λmax = %.6e, λmin = %.6e, ratio = %.3e\n",
                tag, psd.lambda_max, psd.lambda_min, psd.ratio)
        @printf("  [%s] Asym(G⁰ᵣᵣ): λmax = %.6e, λmin = %.6e, ratio = %.3e\n",
                tag, psd.rr_lambda_max, psd.rr_lambda_min, psd.rr_ratio)
    end
    return (spec=spec, smr=smr, front=front, psd=psd)
end

function base_record(point, extra::Dict{String,Any})
    smr, front, spec = point.smr, point.front, point.spec
    prox = filter(is_proximity, front.warnings)
    rec = Dict{String,Any}(
        "tag" => spec.tag, "gap" => spec.gap, "kind" => String(spec.kind),
        "factor" => is_refined(smr) ? refinement(smr).factor : 0,
        "thickness" => is_refined(smr) ? refinement(smr).thickness : 0,
        "N_u" => n_u(smr), "num_pos_stored" => front.meta.num_pos,
        "kept" => front.meta.kept, "exact_spectrum" => front.meta.exact,
        "refinement" => string(refinement(smr)), "file_prefix" => file_prefix(smr),
        "t_green" => front.green.seconds, "t_ur" => front.ur.seconds,
        "t_rs" => front.rs.seconds,
        "rss_green" => front.green.rss_peak,
        "n_warnings" => length(front.warnings), "n_proximity" => length(prox),
        "proximity_first" => isempty(prox) ? "" : first(prox),
        "psd_lambda_max" => get(point, :psd, nothing) === nothing ? NaN : point.psd.lambda_max,
        "psd_lambda_min" => get(point, :psd, nothing) === nothing ? NaN : point.psd.lambda_min,
        "psd_rr_lambda_max" => get(point, :psd, nothing) === nothing ? NaN : point.psd.rr_lambda_max,
        "psd_rr_lambda_min" => get(point, :psd, nothing) === nothing ? NaN : point.psd.rr_lambda_min)
    merge!(rec, extra)
    return rec
end

# A bounds stage that cannot run is a result, not a crash: at these meshes the
# failure mode is `psd_pencil_whitener` refusing a C(τ) that is not positive
# semi-definite, which is exactly what this benchmark is trying to detect. The
# record is written with `failed` set so the sweep continues and the report says
# which points died and why.
function finish(point, idxs::Vector{Int}, opts)
    try
        point.front.meta.kept == 0 && error("--no-spectrum: no Asym(G⁰ᵤᵣ) basis was computed, so this point has Green operators and a positivity probe but no bounds")
        return finish_ok(point, idxs, opts)
    catch err
        msg = sprint(showerror, err)
        @printf("  [%s] bounds FAILED: %s\n", point.spec.tag, first(split(msg, '\n')))
        return base_record(point, Dict{String,Any}(
            "failed" => true, "error" => msg,
            "num_pos" => point.front.meta.kept, "ns" => Int[],
            "dual" => Float64[], "true_bounds" => Float64[],
            "old_analytical" => Float64[], "new_analytical" => Float64[],
            "which_bounds" => Int[], "opt_taus" => Float64[],
            "Gamma" => Float64[], "Gamma_rs" => Float64[], "augmented" => false,
            "trace_exact" => NaN, "trace_interp" => NaN, "trace_dual_interp" => NaN,
            "t_bounds" => NaN, "rss_bounds" => UInt64(0)))
    end
end

function finish_ok(point, idxs::Vector{Int}, opts)
    smr, front, spec = point.smr, point.front, point.spec
    stage = run_bounds(front, smr, idxs, rtol_of(opts),
                       optint(opts, "k-uu", 0),
                       optint(opts, "augment-threshold", PSC.DEFAULT_AUGMENT_THRESHOLD),
                       optint(opts, "seed", 20260827))
    res, ns = stage.value.res, stage.value.ns
    @printf("  [%s] bounds: %.1f s over %d channel(s) of m = %d, maxrss %.2f GiB\n",
            spec.tag, stage.seconds, length(ns), stage.value.inputs.num_pos,
            stage.rss_peak / 2^30)

    dual = res.bounds_dual_basis[ns]
    true_b = res.true_bounds[ns]
    return base_record(point, Dict{String,Any}(
        "failed" => false, "error" => "",
        "num_pos" => stage.value.inputs.num_pos,
        "ns" => ns, "dual" => dual, "true_bounds" => true_b,
        "old_analytical" => res.old_analytical_bounds,
        "new_analytical" => res.new_analytical_bounds,
        "which_bounds" => res.which_bounds[ns],
        "opt_taus" => res.opt_taus[ns],
        "Gamma" => Array(stage.value.inputs.Γ),
        "Gamma_rs" => Array(stage.value.inputs.Γrs),
        "augmented" => res.augmentation.augmented,
        "trace_exact" => sum(filter(isfinite, true_b)),
        "trace_interp" => interp_trace(ns, true_b),
        "trace_dual_interp" => interp_trace(ns, dual),
        "t_bounds" => stage.seconds, "rss_bounds" => stage.rss_peak))
end

save_point(root, rec) = jldopen(joinpath(root, "results", "$(rec["tag"]).jld2"), "w") do jld
    for (k, v) in rec
        jld[k] = v
    end
end

# ---------------------------------------------------------------------------
# Reporting

function rel(a, b)
    d = max(abs(a), abs(b))
    d == 0 && return 0.0
    return abs(a - b) / d
end

# Over the channels both runs evaluated. A mesh whose kept m falls short of the
# shared ladder evaluates a prefix of it, so the comparison runs to the shorter
# of the two rather than indexing past the end.
function channel_stats(ref::Vector{Float64}, alt::Vector{Float64}, floor_abs::Float64)
    n = min(length(ref), length(alt))
    keep = findall(i -> isfinite(ref[i]) && isfinite(alt[i]) &&
                        max(abs(ref[i]), abs(alt[i])) > floor_abs, 1:n)
    isempty(keep) && return (n=0, max=NaN, median=NaN, argmax=0)
    ds = [rel(ref[i], alt[i]) for i in keep]
    j = argmax(ds)
    return (n=length(keep), max=maximum(ds), median=median(ds), argmax=keep[j])
end

load_points(root) = begin
    dir = joinpath(root, "results")
    recs = Dict{String,Dict{String,Any}}()
    for f in sort(readdir(dir))
        endswith(f, ".jld2") || continue
        recs[splitext(f)[1]] = jldopen(joinpath(dir, f), "r") do jld
            Dict{String,Any}(k => jld[k] for k in keys(jld))
        end
    end
    return recs
end

sep_label(g) = string((g//32))

function report(root::String, floor_abs::Float64)
    recs = load_points(root)
    isempty(recs) && (println("no results in $(joinpath(root, "results"))"); return)

    println("\n", "="^100)
    println("TABLE 1  refined vs unrefined, (8,8,8) at 1//32, χ = $(CHI)")
    println("="^100)
    @printf("%-7s %6s %6s %6s %6s  %12s %12s %9s  %9s %9s %5s\n",
            "sep λ", "Nu_un", "Nu_re", "m_un", "m_re", "trace_unref", "trace_ref",
            "rel diff", "chan max", "chan med", "warn")
    traces = Tuple{Int,Float64}[]
    for g in 1:12
        un = get(recs, "g$(g)_unref", nothing)
        re = get(recs, "g$(g)_ref", nothing)
        (isnothing(un) || isnothing(re)) && continue
        if get(un, "failed", false) || get(re, "failed", false)
            @printf("%-7s %6d %6d %6d %6d  %12s %12s %9s  %9s %9s %d/%d\n",
                    sep_label(g), un["N_u"], re["N_u"], un["kept"], re["kept"],
                    get(un, "failed", false) ? "FAILED" : "-",
                    get(re, "failed", false) ? "FAILED" : "-", "-", "-", "-",
                    un["n_proximity"], re["n_proximity"])
            continue
        end
        cs = channel_stats(un["true_bounds"], re["true_bounds"], floor_abs)
        tu, tr = un["trace_interp"], re["trace_interp"]
        push!(traces, (g, tr))
        @printf("%-7s %6d %6d %6d %6d  %12.6f %12.6f %+9.3e  %9.3e %9.3e %d/%d\n",
                sep_label(g), un["N_u"], re["N_u"], un["num_pos"], re["num_pos"],
                tu, tr, (tr - tu) / max(abs(tu), eps()), cs.max, cs.median,
                un["n_proximity"], re["n_proximity"])
    end
    println("\n  trace_* is the sampled curve of true_bounds = min(Eq18, Eq19, dual)")
    println("  interpolated over 1:m. `chan max/med` are per-channel relative")
    println("  differences over sampled channels above $(floor_abs). `warn` is the")
    println("  proximity-warning count, unrefined/refined.")

    # The dual on its own, which is the piece the mesh actually changes.
    println("\n", "-"^100)
    println("TABLE 1b  the raw dual bound (bounds_dual_basis), same channels")
    println("-"^100)
    @printf("%-7s %12s %12s %9s  %9s %9s  %9s\n",
            "sep λ", "dual_unref", "dual_ref", "rel diff", "chan max", "chan med", "chan1 Δrel")
    for g in 1:12
        un = get(recs, "g$(g)_unref", nothing)
        re = get(recs, "g$(g)_ref", nothing)
        (isnothing(un) || isnothing(re)) && continue
        (get(un, "failed", false) || get(re, "failed", false)) && continue
        cs = channel_stats(un["dual"], re["dual"], floor_abs)
        tu, tr = un["trace_dual_interp"], re["trace_dual_interp"]
        d1 = (re["dual"][1] - un["dual"][1]) / abs(un["dual"][1])
        @printf("%-7s %12.6f %12.6f %+9.3e  %9.3e %9.3e  %+9.3e\n",
                sep_label(g), tu, tr, (tr - tu) / max(abs(tu), eps()),
                cs.max, cs.median, d1)
    end

    if length(traces) > 1
        println("\n  domain monotonicity of the refined traces in separation:")
        ok = true
        for i in 1:(length(traces) - 1)
            a, b = traces[i], traces[i+1]
            mono = b[2] <= a[2]
            ok &= mono
            @printf("    %s → %s : %.6f → %.6f  %s\n", sep_label(a[1]), sep_label(b[1]),
                    a[2], b[2], mono ? "decreasing" : "INCREASING")
        end
        println("  ", ok ? "monotone decreasing throughout" : "NOT monotone")
    end

    # Table 2: the f sweep at g = 1.
    fkeys = sort([k for k in keys(recs) if startswith(k, "g1_f")])
    if !isempty(fkeys)
        println("\n", "="^100)
        println("TABLE 2  mesh convergence at g = 1 (sep 1//32 λ)")
        println("="^100)
        ref = get(recs, "g1_ref", nothing)
        @printf("%-14s %6s %6s %6s %6s  %12s %9s  %9s %9s\n",
                "mesh", "f", "t", "N_u", "m", "trace", "vs (6,6)", "chan max", "chan med")
        rows = Any[]
        !isnothing(ref) && push!(rows, ref)
        append!(rows, (recs[k] for k in fkeys))
        for r in rows
            cs = isnothing(ref) ? (max=NaN, median=NaN) :
                 channel_stats(ref["true_bounds"], r["true_bounds"], floor_abs)
            dt = isnothing(ref) ? NaN :
                 (r["trace_interp"] - ref["trace_interp"]) / abs(ref["trace_interp"])
            @printf("%-14s %6d %6d %6d %6d  %12.6f %+9.3e  %9.3e %9.3e\n",
                    r["tag"], r["factor"], r["thickness"], r["N_u"], r["num_pos"],
                    r["trace_interp"], dt, cs.max, cs.median)
        end
    end

    # Table 3: timings.
    println("\n", "="^100)
    println("TABLE 3  wall clock [s] and peak RSS [GiB]")
    println("="^100)
    @printf("%-14s %6s %6s  %9s %9s %9s %9s  %8s %8s\n",
            "tag", "N_u", "m", "green", "ur_asym", "rs", "bounds", "rss_grn", "rss_bnd")
    for k in sort(collect(keys(recs)))
        r = recs[k]
        @printf("%-14s %6d %6d  %9.1f %9.1f %9.1f %9.1f  %8.2f %8.2f\n",
                r["tag"], r["N_u"], r["num_pos"], r["t_green"], r["t_ur"], r["t_rs"],
                r["t_bounds"], r["rss_green"] / 2^30, r["rss_bounds"] / 2^30)
    end

    # The no-op anchor, wherever both members exist at a gap the table does not
    # refine. Bit identity is the claim, not agreement to a tolerance.
    for g in MIN_GAP_CELLS:12
        un = get(recs, "g$(g)_unref", nothing)
        re = get(recs, "g$(g)_ref", nothing)
        (isnothing(un) || isnothing(re)) && continue
        println("\n", "="^100)
        println("NO-OP ANCHOR at sep $(sep_label(g)) λ (g = $g ≥ MIN_GAP_CELLS = $(MIN_GAP_CELLS))")
        println("="^100)
        same_mesh = un["N_u"] == re["N_u"] && un["file_prefix"] == re["file_prefix"]
        bits(a, b) = length(a) == length(b) && all(i -> (isnan(a[i]) && isnan(b[i])) || a[i] === b[i], eachindex(a))
        println("  identical mesh and cache key : ", same_mesh, "  ($(un["file_prefix"]))")
        println("  Γ bit-identical              : ", bits(un["Gamma"], re["Gamma"]))
        println("  Γrs bit-identical            : ", bits(un["Gamma_rs"], re["Gamma_rs"]))
        println("  dual bit-identical           : ", bits(un["dual"], re["dual"]))
        println("  true_bounds bit-identical    : ", bits(un["true_bounds"], re["true_bounds"]))
        println("  max |Δdual|                  : ",
                maximum(abs.(un["dual"] .- re["dual"])))
    end

    # Asym(G⁰ᵤᵤ) is the radiated-power operator of the universe and the only term
    # of C(τ) that is not positive by construction. This is the table that says
    # whether a mesh still has a bound to compute.
    println("\n", "="^100)
    println("ASYM(G⁰ᵤᵤ) POSITIVITY  (λmin/λmax at roundoff = healthy; order one = the")
    println("                       constraint is no longer positive semi-definite)")
    println("="^100)
    @printf("%-14s %6s  %13s %13s %10s %-8s  %13s %10s %s\n",
            "tag", "N_u", "uu λmax", "uu λmin", "uu ratio", "", "rr λmin", "rr ratio", "")
    for k in sort(collect(keys(recs)))
        r = recs[k]
        lmax, lmin = get(r, "psd_lambda_max", NaN), get(r, "psd_lambda_min", NaN)
        isfinite(lmax) || continue
        rmax, rmin = get(r, "psd_rr_lambda_max", NaN), get(r, "psd_rr_lambda_min", NaN)
        ratio, rratio = lmin / lmax, rmin / rmax
        verdict(x) = x > -1e-8 ? "PSD" : "NOT PSD"
        @printf("%-14s %6d  %13.6e %13.6e %10.3e %-8s  %13.6e %10.3e %s\n",
                r["tag"], r["N_u"], lmax, lmin, ratio, verdict(ratio),
                rmin, rratio, isfinite(rratio) ? verdict(rratio) : "")
    end

    failed = [recs[k] for k in sort(collect(keys(recs))) if get(recs[k], "failed", false)]
    if !isempty(failed)
        println("\n", "="^100)
        println("POINTS WHOSE BOUNDS COULD NOT BE COMPUTED")
        println("="^100)
        for r in failed
            println("  ", r["tag"], " (N_u = ", r["N_u"], ", kept m = ", r["kept"], ")")
            println("    ", first(split(r["error"], '\n')))
        end
    end

    println("\n", "="^100)
    println("PROXIMITY WARNINGS")
    println("="^100)
    for k in sort(collect(keys(recs)))
        r = recs[k]
        @printf("%-14s %3d warning(s), %3d proximity  %s\n", r["tag"], r["n_warnings"],
                r["n_proximity"], first(split(r["proximity_first"], '\n')))
    end
end

# ---------------------------------------------------------------------------

function main(argv::Vector{String})
    opts = parse_cli(argv)
    root = optstr(opts, "out", joinpath(pwd(), "refined_near_field_out"))
    mkpath(joinpath(root, "results"))
    floor_abs = haskey(opts, "min-abs") ? parse(Float64, opts["min-abs"]) : 1e-5

    if optbool(opts, "report-only", false)
        report(root, floor_abs)
        return
    end

    quick = optbool(opts, "quick", false)
    gaps = optints(opts, "gaps", quick ? [1, 3] : [1, 2, 3, 4, 5, 6])
    fconv = optbool(opts, "fconv", !quick)
    params = RSVDParams(optint(opts, "rank", 512), optint(opts, "oversamples", 20),
                        optint(opts, "power-iterations", 6), optint(opts, "seed", 20260827))
    head = optint(opts, "ladder-head", 16)
    points = optint(opts, "ladder-points", 40)
    full = optstr(opts, "outer-mode", "ladder") == "full"

    println("bench/refined_near_field.jl  $(now())")
    println("  out      = $root")
    println("  gaps     = $gaps, fconv = $fconv, mode = $(full ? "full" : "ladder($head, $points)")")
    println("  BLAS threads = $(BLAS.get_num_threads()), Julia threads = $(Threads.nthreads())")

    for g in gaps
        println("\n--- gap $g cells (sep $(sep_label(g)) λ) ---")
        specs = [(tag="g$(g)_unref", gap=g, kind=:unrefined),
                 (tag="g$(g)_ref", gap=g, kind=:refined)]
        pts = [measure(root, s, params, opts) for s in specs]
        m = minimum(p.front.meta.kept for p in pts)
        idxs = full ? collect(1:m) : ladder_indices(m, head, points)
        println("  shared channel set: $(length(idxs)) of m = $m " *
                "(kept: $([p.front.meta.kept for p in pts]))")
        for p in pts
            save_point(root, finish(p, idxs, opts))
        end
    end

    if fconv
        # The g = 1 row of GAP_REFINEMENT_TABLE is (6, 6). f = 12 doubles the
        # factor at the same slab thickness; (2, 4) is the g = 3 row applied at
        # g = 1, a deliberately *under*-refined mesh that says which way the
        # answer moves as f falls.
        println("\n--- f convergence at gap 1 ---")
        fspecs = NamedTuple[]
        for tok in split(optstr(opts, "f-meshes", "12/6,3/6,2/4"), ',')
            f, t = parse.(Int, split(tok, '/'))
            push!(fspecs, (tag="g1_f$(f)t$(t)", gap=1, kind=:custom, factor=f, thickness=t))
        end
        ref = get(load_points(root), "g1_ref", nothing)
        isnothing(ref) && error("run --gaps 1 first: the f table is anchored on g1_ref")
        idxs = Vector{Int}(ref["ns"])
        for s in fspecs
            p = measure(root, s, params, opts)
            save_point(root, finish(p, idxs, opts))
        end
    end

    # Arbitrary (gap, factor, thickness) points, as `g/f/t` triples. The one this
    # exists for is a refined mesh at a gap the table would leave alone: if a
    # refined body's operator misbehaves there too, the defect is in the mesh
    # rather than in the proximity the mesh was introduced to fix.
    for tok in split(optstr(opts, "custom", ""), ',')
        isempty(tok) && continue
        g, f, t = parse.(Int, split(tok, '/'))
        s = (tag="g$(g)_f$(f)t$(t)", gap=g, kind=:custom, factor=f, thickness=t)
        println("\n--- custom mesh: gap $g, factor $f, thickness $t ---")
        p = measure(root, s, params, opts)
        m = p.front.meta.kept
        # `--ns-from <tag>` evaluates the same channel indices an existing record
        # did, which is what makes a custom mesh comparable to it channel by
        # channel rather than only in the trace.
        borrowed = optstr(opts, "ns-from", "")
        idxs = if isempty(borrowed)
            full ? collect(1:m) : ladder_indices(m, head, points)
        else
            Vector{Int}(load_points(root)[borrowed]["ns"])
        end
        save_point(root, finish(p, idxs, opts))
    end

    report(root, floor_abs)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
