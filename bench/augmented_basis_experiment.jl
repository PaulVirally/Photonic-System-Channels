#!/usr/bin/env julia
"""
    bench/augmented_basis_experiment.jl

Does augmenting the projection basis with the top eigenvectors of `Asym(G⁰ᵤᵤ)`
restore the far-field dual bound?

# The failure this is aimed at

The production bounds project every operator into the span of the *kept*
positive-Γ eigenvectors of `Asym(G⁰ᵤᵣ)` (the "g basis", `m` columns). That span
collapses with separation: at the 1 λ cube sweep, `s = 357/32 λ ≈ 11.16 λ` keeps
`m = 15` of `N_u = 196,608` directions (`bench/data/kept_by_sep_1l.csv`, row 112).
The projected dual then *loses validity*: it under-reports. The exact λ/4 sweep,
whose domain is strictly *contained* in the 1 λ one, comes out **above** it at the
same gap:

    channel 1:  0.042178  (exact λ/4, 8x8x8__8x8x8__357ss32__RS.jld)
                0.017831  (reduced 1 λ, 32x32x32__32x32x32__357ss32__RS.jld)

A bound that shrinks when the design space grows is not a bound. The diagnosis is
that the *constraint* operators cannot be represented in a 15-dimensional span,
above all `Asym(G⁰ᵤᵤ)`, the universe's radiative self-interaction, whose rank does
*not* collapse with separation. Once `PᵀC P` stops implying full-space
feasibility, the "dual" is an unconstrained relaxation of nothing in particular.

# The hypothesis

Augment the projection basis with the top `k_uu` eigenvectors of `Asym(G⁰ᵤᵤ)`:

    basis = orthonormalize([g_kept, U_uu[:, 1:k_uu]])

and the constraint is representable again, so the dual rises back to a valid
level. There is archaeological support in `src/bounds.jl`: the commented-out
`# U_uu = read_array(jld_in, "UU/U", ...)` and
`# basis = cat(U_uu, Vur_asym; dims=2)` near the top of `bounds_from_spectrum`.

This is monotone by construction: the projected primal is a `sup` over
`t ∈ range(P)`, so a *larger* subspace gives a *larger* value. The `k_uu` sweep
uses leading subsets of one `U_uu`, so the subspaces are nested and channel 1 has
to be non-decreasing in `k_uu`. A non-monotone column is a bug in this file, not
physics.

# What it computes

For one separation of the 1 λ sweep (default the 333-grid point closest to
11.16 λ, which is exactly `357//32`), and for each `k_uu` in the sweep:

  1. Load the production RSVD output from the production scratch dir (read-only)
     via `load_bounds_inputs`: the kept `g` basis and `Γ`.
  2. Compute the top `k_uu_max` eigenvectors of `Asym(G⁰ᵤᵤ)` **once**, with
     `reigen_hermitian`, and nest the sweep inside them.
  3. Build `basis = [g_kept, U_uu[:, 1:k_uu]]`, orthonormalized with a documented
     rank guard (see `augmented_basis`).
  4. Solve the same τ-family dual as production, over the first `--channels`
     indices `n`, in that basis (see `augmented_dual_bounds`).
  5. Print a table and the success criteria, and save everything to `--out`.

# This is production code now

When this experiment was written none of the machinery it needed existed in
`src/`, so it carried its own. It answered its question (at the 1 λ,
`s = 357/32` point, channel 1 went 0.0178 → 0.2019, monotone in `k_uu` and
saturating, crossing the exact λ/4 reference 0.0422 at `k_uu = 128`) and the
machinery was promoted: `augmented_basis`, `FactoredB`, `factored_pencil_eigen`,
`factored_probe_duals`, `uu_eigenbasis`, `uu_residuals` and the augmented front
end are all in `src/bounds.jl`, behind `bounds_from_spectrum`'s `--k-uu` /
`--augment-threshold`.

So this file no longer implements anything the bounds job does. It is a driver:
it loads the production RSVD output, computes one shared `U_uu`, and calls the
*shipped* front end and the *shipped* pencil stage once per `k_uu`. What is left
here is the sweep itself and the two things production cannot be asked for: a
run over an arbitrary basis (`basis = 1`, for `test/augmented_basis.jl`'s
full-space ground truth) and a `k_uu` sweep nested inside one `U_uu`, which is
what makes the monotonicity criterion arithmetic rather than a statement about
six independent random sketches.

Why the augmented basis needs a front end of its own at all: production exploits
two facts that hold *only* in the pure `g` basis, namely that `Bₙ` and
`basisᴴ(−G⁰ᵤᵣ)ᵃ₊basis` are diagonal there. In the augmented basis they are not.
Everything still reduces to one extra small matrix,

    W = basisᴴ · gs_kept        (m_aug × m),

because both operators are built out of the `gₗ` alone:

    basisᴴ (−G⁰ᵤᵣ)ᵃ₊ basis = W diag(Γ) Wᴴ
    basisᴴ Bₙ basis        = W[:, n:m] diag(4Γ[n:m]/ζ) W[:, n:m]ᴴ

`basisᴴΠₛbasis` is the Gram matrix of the stored sender rows, exactly as in
production, and only the `(G⁰ᵤᵤ)ᵃ` term needs matvecs (`opmat(C, basis)`-style,
as the dense front end already does). So this front end costs *one* Green sweep
over `m_aug` columns per `k_uu`, and no new matrix-free work at all in the pencil
stage.

Note that `--k-uu` here is a *list*, the sweep, while `--k-uu` on
`compute_bounds.jl` is the single number a production job augments by. The
production default, 512, is the last point of this sweep.

# The probes stay the production probes

`ss` is built by `reverse_gram_schmidt!` on `Πₛ · gs_kept`, the *`g` columns
alone*, never the augmentation. This is deliberate and it is the one design
decision in here that is not forced.

The probe `sₖ` is what *defines which* `σₙ` is being bounded: column `i` of `ss`
spans `Πₛ·span(gᵢ, …, g_m)` together with the later columns, and the outer loop's
shrinking probe set `k ≥ n` is what makes index `n` mean "the n-th channel". Add
the `U_uu` columns to that construction and the nested spans change, so index `n`
names a different quantity and neither the `k_uu` sweep nor the comparison
against production (0.0178) nor the one against the exact λ/4 run (0.0422) is
apples-to-apples any more.

The augmentation is about representing the *constraint* faithfully, enlarging
the feasible set the dual is solved over, not about redefining the objective.
Keeping the probes fixed is what isolates that one change.

Note that the augmentation still helps the probes, without changing them: the
dual evaluates `α²/4 · sₖᴴ(αC − Bₙ)⁺sₖ` on `Pᵀsₖ`, and a larger `P` captures more
of each `sₖ`. That is part of the mechanism, not a confound: it is the same
`sₖ`, seen less lossily.

# Usage

    julia --project=. bench/augmented_basis_experiment.jl \\
        --sep 357//32 \\
        --scratch /home/pvirally/scratch/Photonic-System-Channels/narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale/ \\
        --out \$SCRATCH/psc-augmented-basis/

`jobs/sbatch_augmented_basis_1l.sh` is the narval wrapper. Everything is written
to `--out`; the production scratch and project directories are only ever read.
"""

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels

using CUDA
using Dates
using JLD2
using LinearAlgebra
using LinearMaps
using MatrixFreeRandomizedLinearAlgebra
using Printf
using Roots

# ---------------------------------------------------------------------------
# Reference numbers, all read out of the saved sweeps in `data analysis/data/`
# on 2026-08-20. They are hard-coded so the job prints its own verdict without
# needing either project directory on the node.
# ---------------------------------------------------------------------------

# The 333-point grid of the 1 λ sweep, closest point to 11.16 λ. Verified against
# jobs/greens_tasks_narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale.txt
# (entry 111 of 333: --rs-sep (357//32,0//1,0//1)) and against
# bench/data/kept_by_sep_1l.csv row 112 (sep 11.15625, kept 15 of 2035 stored).
const DEFAULT_SEP = 357 // 32

# σ₁ from the exact λ/4 sweep at the same gap: the 0.25 λ cubes are *contained*
# in the 1 λ ones, so this is a lower bound on any honest 1 λ bound, and the
# reduced 1 λ dual sitting below it is the whole reason this experiment exists.
#   data analysis/data/narval_Ge1000_arxivV3_0p25x0p25x0p25_3072comps_50oversamples_q3_32scale/
#     8x8x8__8x8x8__357ss32__RS.jld  ["bounds_dual_basis"][1]
const REFERENCE_QUARTER_WAVE_CHANNEL1 = 0.04217836342317775

# What production reports at this point, for the k_uu = 0 column to reproduce.
#   data analysis/data/narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale/
#     32x32x32__32x32x32__357ss32__RS.jld
const REFERENCE_PRODUCTION_CHANNEL1 = 0.017830501477751625
const REFERENCE_PRODUCTION_TRACE_15 = 0.05043478718641561   # sum over all 15 kept channels

# Ge at 1000 K, the arxivV3 sweep's susceptibility, from the generated launchers.
const DEFAULT_CHI = 17.06132654701751 + 0.29117345im

const DEFAULT_SCRATCH = "/home/pvirally/scratch/Photonic-System-Channels/narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale/"

# The sweep this file scans. Named apart from `PhotonicSystemChannels.DEFAULT_K_UU`,
# which is the *production* scalar (512) that `--k-uu` defaults to; this is the list
# of points the experiment measures on the way there.
const DEFAULT_K_UU_SWEEP = [0, 32, 64, 128, 256, 512]

# The rank guard on the augmentation (`AUG_QR_RTOL`) and the null-space
# admissibility tolerance (`AUG_BTOL`) are production constants now; see
# `src/bounds.jl` for what each is and why it is the value it is. They are
# `using`-imported here rather than redeclared, so `--qr-rtol` defaults to exactly
# what the bounds job runs at.

# ---------------------------------------------------------------------------
# What is production code here, and what is local
# ---------------------------------------------------------------------------
#
# `qr_thin_rdiag!`, `augmented_basis`, `FactoredB`, `factored_pencil_eigen`,
# `factored_probe_duals`, `uu_eigenbasis` and `uu_residuals` are all *production*
# code in `src/bounds.jl`, reached here through `PhotonicSystemChannels`:
# `bounds_from_spectrum`'s `--k-uu` path is exactly this experiment. The
# `C_basis`/`D_basis`/`W` construction over an arbitrary basis is
# `PSC._bounds_front_end_augmented`, which additionally builds the probes rather
# than taking them as an argument.
#
# That matters for what the experiment is *evidence about*. A sweep run against a
# second implementation says the second implementation rises with `k_uu`; a sweep
# run against the shipped one says the shipped one does.
#
# One thing is local, and deliberately: `augmented_dual_bounds`, the τ loop over
# an *arbitrary* basis. Production always builds its basis from `gs_pos` and the
# `Asym(G⁰ᵤᵤ)` solve, so it cannot be asked for the `basis = 1` full-space ground
# truth that `test/augmented_basis.jl`'s anchor (b) needs, nor for a `k_uu` sweep
# nested inside one shared `U_uu` (which is what makes the monotonicity criterion
# arithmetic rather than a statement about two random sketches). It also runs the
# unwindowed, plain-`Dict` variant of the τ search on purpose; see below.

# ---------------------------------------------------------------------------
# The τ driver over an arbitrary basis
# ---------------------------------------------------------------------------

"""
    augmented_dual_bounds(compute_env, basis, gs_pos, Γ_pos_cpu, ζ, sender_size,
                          s_projector, G⁰ᵤᵤ_asym; channels, τs, τ_refine_tol)
        -> NamedTuple

The τ-family dual of `bounds_from_spectrum`, solved in an arbitrary orthonormal
`basis` rather than in the one production would build for itself.

The `N_u`-scale work (the probes, `ss_basis`, `W = basisᴴgs_pos`, `C_basis` and
`D_basis`) is `PSC._bounds_front_end_augmented`, the production front end, and
the pencil algebra is production's `PSC.FactoredB` / `PSC.factored_probe_duals`.
What is written out here is only the loop that drives them.

The τ machinery mirrors production's: the grid pencils are whitened once (they do
not depend on `n`), each index takes the minimum over the grid, and the grid
minimum's two neighbours bracket a golden-section refinement down to
`τ_refine_tol`. Every evaluated τ is a valid bound on its own and the running
minimum keeps all of them, so nothing here can make a bound *invalid*, only less
tight.

Two production knobs are deliberately off:

  * `tau_window`: production sweeps a ±2 window around the previous index's best
    grid point to save `m × m` whitenings at `m = 4000`. Here `m_aug ≤ 527`, a
    whitening is milliseconds, and a full sweep at every index removes one place
    where this could differ from production for a reason that is not the basis.
    `test/tau_search.jl` establishes that the windowed and unwindowed sweeps agree
    exactly, so this costs nothing but time.
  * the LRU pencil cache: replaced by a plain `Dict` memo on the exact `Float64`
    τ, which is what the LRU degenerates to when nothing is ever evicted.
"""
function augmented_dual_bounds(compute_env::ComputeEnvironment, basis, gs_pos,
                               Γ_pos_cpu::AbstractVector, ζ::Real, sender_size::Int,
                               s_projector, G⁰ᵤᵤ_asym;
                               channels::Int,
                               τs::AbstractVector{<:Real}=range(0.0, 1.0, length=5),
                               τ_refine_tol::Union{Nothing,Real}=0.05)
    num_pos = size(gs_pos, 2)
    m_aug = size(basis, 2)
    ns = 1:min(channels, num_pos)

    front = PSC._bounds_front_end_augmented(compute_env, gs_pos, basis, Γ_pos_cpu, ζ,
                                            s_projector, G⁰ᵤᵤ_asym, num_pos, sender_size)
    W, C_basis, D_basis = front.W, front.C_basis, front.D_basis
    ss_basis = front.ss_basis
    t_ss = front.t_ss_basis

    build_pencil(τ) = begin
        C_τ = isone(τ) ? C_basis : C_basis .- (1 - τ) .* D_basis
        try
            PSC.psd_pencil_whitener(C_τ)
        catch err
            @warn string(now()) * " [augmented_basis::augmented_dual_bounds] " *
                  "psd_pencil_whitener failed at τ=$τ; treating this point as unusable" exception = err
            nothing
        end
    end

    t_pencils = time_ns()
    pencils = [build_pencil(τ) for τ in τs]
    usable = findall(!isnothing, pencils)
    isempty(usable) && error("psd_pencil_whitener failed at every τ in $(collect(τs))")
    t_pencils = (time_ns() - t_pencils) / 1e9
    ranks = [(Float64(τs[i]), pencils[i].rank) for i in usable]

    memo = Dict{Float64,Any}()
    cached_pencil(τ_raw) = begin
        τ = Float64(τ_raw)
        gi = findfirst(i -> Float64(τs[i]) == τ, usable)
        gi === nothing || return pencils[usable[gi]]
        get!(() -> build_pencil(τ), memo, τ)
    end

    # Bₙ's factor: basisᴴBₙbasis = W[:, n:m] diag(4Γ[n:m]/ζ) W[:, n:m]ᴴ.
    factor(n) = begin
        idx = n:num_pos
        Vn = W[:, idx]
        c = similar(basis, real(eltype(basis)), length(idx))
        copyto!(c, sqrt.((4 / ζ) .* Γ_pos_cpu[idx]))
        FactoredB(Vn, c)
    end

    bounds = fill(NaN, length(ns))
    opt_taus = fill(NaN, length(ns))
    by_tau = fill(NaN, length(ns), length(τs))
    per_index_seconds = zeros(Float64, length(ns))

    t_outer = time_ns()
    for (j, n) in enumerate(ns)
        t_n = time_ns()
        F = factor(n)
        eval_dual(pencil, τ) = begin
            isnothing(pencil) && return Inf
            try
                maximum(factored_probe_duals(pencil, F, ss_basis, n, num_pos; τ=τ).duals)
            catch err
                @warn string(now()) * " [augmented_basis::augmented_dual_bounds] " *
                      "[$n] τ=$τ failed; dropping this evaluation" exception = err
                Inf
            end
        end

        best, best_τ, best_i = Inf, NaN, 0
        for i in usable
            v = eval_dual(pencils[i], τs[i])
            isfinite(v) || continue
            by_tau[j, i] = sqrt(v)
            if v < best
                best, best_τ, best_i = v, Float64(τs[i]), i
            end
        end
        isfinite(best) || error("every τ in the grid failed at n=$n")

        if !isnothing(τ_refine_tol) && length(τs) > 1
            lo = Float64(τs[max(best_i - 1, firstindex(τs))])
            hi = Float64(τs[min(best_i + 1, lastindex(τs))])
            invφ = (sqrt(5.0) - 1) / 2
            τ₁ = hi - invφ * (hi - lo)
            τ₂ = lo + invφ * (hi - lo)
            g₁ = eval_dual(cached_pencil(τ₁), τ₁)
            g₂ = eval_dual(cached_pencil(τ₂), τ₂)
            g₁ < best && ((best, best_τ) = (g₁, τ₁))
            g₂ < best && ((best, best_τ) = (g₂, τ₂))
            iters = 0
            while hi - lo > τ_refine_tol && iters < 200
                iters += 1
                if g₁ <= g₂
                    hi, τ₂, g₂ = τ₂, τ₁, g₁
                    τ₁ = hi - invφ * (hi - lo)
                    g₁ = eval_dual(cached_pencil(τ₁), τ₁)
                    g₁ < best && ((best, best_τ) = (g₁, τ₁))
                else
                    lo, τ₁, g₁ = τ₁, τ₂, g₂
                    τ₂ = lo + invφ * (hi - lo)
                    g₂ = eval_dual(cached_pencil(τ₂), τ₂)
                    g₂ < best && ((best, best_τ) = (g₂, τ₂))
                end
            end
        end

        bounds[j] = sqrt(best)
        opt_taus[j] = best_τ
        per_index_seconds[j] = (time_ns() - t_n) / 1e9
        @info string(now()) * " [augmented_basis::augmented_dual_bounds] " *
              "[$n/$(last(ns))] m_aug=$(m_aug) σₙ ≤ $(bounds[j]) at τ = $(best_τ) " *
              "($(round(per_index_seconds[j]; digits=2)) s)"
    end
    t_outer = (time_ns() - t_outer) / 1e9

    return (ns=collect(ns), bounds=bounds, opt_taus=opt_taus, bounds_by_tau=by_tau,
            tau_grid=collect(Float64, τs), pencil_ranks=ranks, m_aug=m_aug,
            per_index_seconds=per_index_seconds,
            stage_times=(gram_schmidt=front.t_gram_schmidt,
                         c_projection=front.t_c_projection,
                         ss_basis=t_ss, pencils=t_pencils, outer=t_outer))
end

# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

"""
    analytical_bounds(Γ, Γrs, ζ) -> (old, new)

The two semi-analytic per-channel bounds `bounds_from_spectrum` reports, copied
verbatim so the table has the same ceiling production plots against. Neither
depends on the projection basis, so both are constant down every `k_uu` column;
they are here as the level the dual must saturate *below*.
"""
function analytical_bounds(Γ::AbstractVector, Γrs::AbstractVector, ζ::Real)
    old_form(κ) = ifelse(κ >= one(κ), one(κ), sqrt(4κ) / (1 + κ))
    new_form(κ̃) = ifelse(2κ̃ >= one(κ̃), one(κ̃), sqrt(4κ̃ * abs(1 - κ̃)))
    old = old_form.(ζ^2 .* (Γrs .^ 2))
    new = new_form.(ζ .* max.(Γ, zero(eltype(Γ))))
    return old, new
end

"""
    run_experiment(compute_env, smr; kwargs...) -> NamedTuple

Load, sweep, report, save. See the module docstring for what each step does.
"""
function run_experiment(compute_env::ComputeEnvironment, smr::SMRSystem;
                        gamma_rtol::Float64=PSC.DEFAULT_GAMMA_RTOL,
                        k_uu_list::Vector{Int}=copy(DEFAULT_K_UU_SWEEP),
                        uu_oversamples::Int=PSC.UU_OVERSAMPLES,
                        uu_power_iters::Int=PSC.UU_POWER_ITERS,
                        channels::Int=16,
                        qr_rtol::Float64=PSC.AUG_QR_RTOL,
                        τs::AbstractVector{<:Real}=range(0.0, 1.0, length=5),
                        τ_refine_tol::Union{Nothing,Real}=0.05,
                        run_production_reference::Bool=true,
                        save::Bool=true)
    prefix = file_prefix(smr)
    @info string(now()) * " [augmented_basis::run_experiment] System $(prefix), " *
          "gap $(rs_separation(smr)[1]) λ, χ = $(susceptibility(smr))"

    # 1. The production RSVD output, read-only, exactly as the bounds job reads it.
    inputs = load_bounds_inputs(compute_env, smr; gamma_rtol=gamma_rtol, panel_mode=false)
    Γ, gs_pos, Γrs = inputs.Γ, inputs.Vur_asym, inputs.Γrs
    num_pos = inputs.num_pos
    N_u = size(gs_pos, 1)
    sender_size = dof_length(sender_mesh(smr))
    χ = susceptibility(smr)
    ζ = abs(χ)^2 / imag(χ)
    Γ_pos_cpu = Array(Γ[1:num_pos])
    @info string(now()) * " [augmented_basis::run_experiment] N_u = $(N_u), the g basis " *
          "kept m = $(num_pos) directions at gamma_rtol = $(gamma_rtol), ζ = $(ζ)"

    # 2. The universe operator, loaded once and shared by everything below.
    G₀_uu = load_green_function(compute_env, smr, [Sender, Receiver], [Sender, Receiver])
    s_projector = PSC.projected_operators(G₀_uu, smr, compute_env)
    G⁰ᵤᵤ_asym = PSC.asym_self(G₀_uu)

    # 3. The production numbers, recomputed here so the k_uu = 0 column has
    #    something on the same node to be compared against. Cheap at m = 15.
    production = nothing
    if run_production_reference
        @info string(now()) * " [augmented_basis::run_experiment] Reproducing the " *
              "production bounds_from_spectrum for the first $(channels) channels"
        t = time_ns()
        # `k_uu = 0` explicitly: `bounds_from_spectrum` now augments by default, and
        # what this column is for is the *pre-augmentation* number the saved sweep
        # reports (REFERENCE_PRODUCTION_CHANNEL1). The augmented production path is
        # what the k_uu rows below measure.
        res = bounds_from_spectrum(compute_env, smr, Γ, gs_pos, Γrs; num_pos=num_pos,
                                   G₀_uu=G₀_uu, outer_indices=collect(1:min(channels, num_pos)),
                                   τs=τs, τ_refine_tol=τ_refine_tol, k_uu=0)
        production = (bounds=res.bounds_dual_basis[1:min(channels, num_pos)],
                      opt_taus=res.opt_taus[1:min(channels, num_pos)],
                      seconds=(time_ns() - t) / 1e9)
        @info string(now()) * " [augmented_basis::run_experiment] Production reference: " *
              "channel 1 = $(production.bounds[1]) (saved sweep says $(REFERENCE_PRODUCTION_CHANNEL1))"
        # Free the production run's own N_u-scale objects before the sweep.
        res = nothing
        GC.gc()
        use_gpu(compute_env) && CUDA.reclaim()
    end

    # 4. The probes are built inside `PSC._bounds_front_end_augmented`, once per
    #    k_uu, by the same `reverse_gram_schmidt!` on Πₛ·gs_pos that production runs
    #    (from the g columns alone, never the augmentation). That is deterministic
    #    and O(m²) over N_u-vectors at m = 15, so rebuilding it per row costs nothing
    #    and leaves no object for this file to construct on its own.

    # 5. Asym(G⁰ᵤᵤ)'s leading eigenvectors, once, at the largest k_uu asked for.
    k_uu_max = maximum(k_uu_list)
    U_uu = nothing
    uu_values = Float64[]
    uu_resid = (idxs=Int[], values=Float64[])
    uu_seconds = 0.0
    if k_uu_max > 0
        uu = uu_eigenbasis(compute_env, G⁰ᵤᵤ_asym, k_uu_max;
                           oversamples=uu_oversamples, power_iters=uu_power_iters)
        U_uu, uu_values, uu_seconds = uu.vectors, uu.values, uu.seconds
        k_avail = size(U_uu, 2)
        if k_avail < k_uu_max
            @warn string(now()) * " [augmented_basis::run_experiment] reigen_hermitian " *
                  "returned $(k_avail) of the $(k_uu_max) requested components; the " *
                  "sweep is clamped to that"
            k_uu_list = unique(min.(k_uu_list, k_avail))
        end
        ridx = unique(clamp.([1, cld(k_avail, 4), cld(k_avail, 2), k_avail], 1, k_avail))
        uu_resid = (idxs=ridx, values=uu_residuals(G⁰ᵤᵤ_asym, U_uu, uu_values, ridx))
        @info string(now()) * " [augmented_basis::run_experiment] Asym(G⁰ᵤᵤ) eigenpair " *
              "residuals ‖Av − λv‖/Λ₁ at $(ridx): $(uu_resid.values)"
        GC.gc()
        use_gpu(compute_env) && CUDA.reclaim()
    end

    # 6. The sweep.
    old_ana, new_ana = analytical_bounds(Γ, Γrs, ζ)
    nchan = min(channels, num_pos)
    rows = []
    for k_uu in unique(sort(k_uu_list))
        @info string(now()) * " [augmented_basis::run_experiment] ===== k_uu = $(k_uu) ====="
        t = time_ns()
        aug = augmented_basis(gs_pos,
                              k_uu == 0 ? view(gs_pos, :, 1:0) : view(U_uu, :, 1:k_uu);
                              rtol=qr_rtol)
        @info string(now()) * " [augmented_basis::run_experiment] k_uu = $(k_uu): " *
              "m_aug = $(size(aug.basis, 2)) (g: $(num_pos), U_uu kept: $(aug.num_uu_kept), " *
              "dropped: $(aug.num_uu_dropped) $(aug.dropped_cols), " *
              "min|R|/max|R| = $(aug.rdiag_min_ratio))"
        res = augmented_dual_bounds(compute_env, aug.basis, gs_pos, Γ_pos_cpu, ζ,
                                    sender_size, s_projector, G⁰ᵤᵤ_asym;
                                    channels=channels, τs=τs,
                                    τ_refine_tol=τ_refine_tol)
        wall = (time_ns() - t) / 1e9
        push!(rows, (k_uu=k_uu, m_aug=res.m_aug, num_uu_kept=aug.num_uu_kept,
                     num_uu_dropped=aug.num_uu_dropped, bounds=res.bounds,
                     opt_taus=res.opt_taus, bounds_by_tau=res.bounds_by_tau,
                     pencil_ranks=res.pencil_ranks, stage_times=res.stage_times,
                     seconds=wall))
        aug = nothing
        res = nothing
        GC.gc()
        use_gpu(compute_env) && CUDA.reclaim()
    end

    result = (prefix=prefix, N_u=N_u, num_pos=num_pos, ζ=ζ, χ=χ,
              gamma_rtol=gamma_rtol, channels=nchan, Γ=Array(Γ), Γrs=Array(Γrs),
              old_analytical=old_ana, new_analytical=new_ana,
              uu_values=uu_values, uu_residual_idxs=uu_resid.idxs,
              uu_residuals=uu_resid.values, uu_seconds=uu_seconds,
              production=production, rows=rows, tau_grid=collect(Float64, τs))

    report(result)
    save && save_result(compute_env, result)
    return result
end

"""
    report(result)

The table and the verdict. Everything the job is for is in this output; the JLD
next to it is for replotting.
"""
function report(result)
    io = IOBuffer()
    nchan = result.channels
    ncol = min(5, nchan)

    println(io, "\n================ augmented-basis sweep: $(result.prefix) ================")
    @printf(io, "N_u = %d   g basis m = %d (gamma_rtol = %g)   channels reported = %d   ζ = %.6g\n",
            result.N_u, result.num_pos, result.gamma_rtol, nchan, result.ζ)
    if !isempty(result.uu_values)
        @printf(io, "Asym(G⁰ᵤᵤ): Λ[1] = %.6e, Λ[%d] = %.6e (ratio %.3e), %.1f s\n",
                result.uu_values[1], length(result.uu_values), result.uu_values[end],
                result.uu_values[end] / result.uu_values[1], result.uu_seconds)
        @printf(io, "            eigenpair residuals ‖Av−λv‖/Λ₁ at %s: %s\n",
                string(result.uu_residual_idxs),
                join((@sprintf("%.2e", r) for r in result.uu_residuals), ", "))
    end

    header = @sprintf("%6s %6s %5s", "k_uu", "m_aug", "drop")
    for c in 1:ncol
        header *= @sprintf(" %13s", "ch$(c)")
    end
    header *= @sprintf(" %13s %9s", "trace(1..$(nchan))", "wall[s]")
    println(io, "\n", header)
    println(io, "-"^length(header))

    for row in result.rows
        line = @sprintf("%6d %6d %5d", row.k_uu, row.m_aug, row.num_uu_dropped)
        for c in 1:ncol
            line *= @sprintf(" %13.6e", row.bounds[c])
        end
        line *= @sprintf(" %13.6e %9.1f", sum(row.bounds), row.seconds)
        println(io, line)
    end

    if result.production !== nothing
        line = @sprintf("%6s %6d %5s", "prod", result.num_pos, "-")
        for c in 1:ncol
            line *= @sprintf(" %13.6e", result.production.bounds[c])
        end
        line *= @sprintf(" %13.6e %9.1f", sum(result.production.bounds),
                         result.production.seconds)
        println(io, line)
    end

    line = @sprintf("%6s %6s %5s", "analyt", "-", "-")
    for c in 1:ncol
        line *= @sprintf(" %13.6e", min(result.old_analytical[c], result.new_analytical[c]))
    end
    println(io, line)

    println(io, "\nOptimal τ per channel:")
    for row in result.rows
        @printf(io, "    k_uu = %-5d %s\n", row.k_uu,
                join((@sprintf("%.3f", t) for t in row.opt_taus[1:ncol]), " "))
    end
    println(io, "C(τ) numerical ranks (τ → rank/m_aug):")
    for row in result.rows
        @printf(io, "    k_uu = %-5d %s\n", row.k_uu,
                join(("τ=$(t)→$(r)/$(row.m_aug)" for (t, r) in row.pencil_ranks), ", "))
    end

    # --- the verdict
    ch1 = [row.bounds[1] for row in result.rows]
    ks = [row.k_uu for row in result.rows]
    println(io, "\n================ success criteria (channel 1) ================")
    @printf(io, "reference: exact λ/4 at the same gap = %.6e   reduced 1 λ (saved) = %.6e\n",
            REFERENCE_QUARTER_WAVE_CHANNEL1, REFERENCE_PRODUCTION_CHANNEL1)

    # 1. Monotone. Nested subspaces, so this is arithmetic, not physics: a
    #    violation past roundoff means the front end or the basis is wrong.
    steps = length(ch1) > 1 ? diff(ch1) ./ max.(ch1[1:end-1], eps()) : Float64[]
    worst_step = isempty(steps) ? 0.0 : minimum(steps)
    mono = worst_step >= -1e-8
    @printf(io, "1. monotone non-decreasing in k_uu ....... %s (worst relative step %.3e, negative = backward)\n",
            mono ? "YES" : "NO ", worst_step)

    # 2. Crosses the λ/4 value.
    crossed = findfirst(>=(REFERENCE_QUARTER_WAVE_CHANNEL1), ch1)
    detail = if crossed === nothing
        @sprintf(" (best %.6e = %.3f× the λ/4 value, short by %.2f×)", maximum(ch1),
                 maximum(ch1) / REFERENCE_QUARTER_WAVE_CHANNEL1,
                 REFERENCE_QUARTER_WAVE_CHANNEL1 / max(maximum(ch1), eps()))
    else
        @sprintf(" (first at k_uu = %d, %.6e)", ks[crossed], ch1[crossed])
    end
    @printf(io, "2. crosses the exact λ/4 value ........... %s%s\n",
            crossed === nothing ? "NO " : "YES", detail)

    # 3. Saturates below the analytical ceiling. At this separation the analytical
    #    channel-1 bound is 1.0 (κ̃ = ζΓ₁ ≥ 1/2), so this is a sanity check that the
    #    augmentation has not blown the dual past every other bound we hold.
    ceiling = min(result.old_analytical[1], result.new_analytical[1])
    below = ch1[end] <= ceiling * (1 + 1e-6)
    @printf(io, "3. stays below the analytical channel 1 .. %s (analytical = %.6e, last = %.6e)\n",
            below ? "YES" : "NO ", ceiling, ch1[end])

    last_step = isempty(steps) ? NaN : steps[end]
    @printf(io, "4. saturating ............................ last relative increase %.3e (k_uu %d → %d)\n",
            last_step, length(ks) > 1 ? ks[end-1] : ks[end], ks[end])

    println(io, "\nHow to read this. Enlarging the projection basis enlarges the primal feasible")
    println(io, "set, so criterion 1 has to hold identically; if it does not, the bug is here.")
    println(io, "Criterion 2 is the hypothesis: the far-field dual under-reports because the")
    println(io, "constraint operators, Asym(G⁰ᵤᵤ) above all, cannot live in a $(result.num_pos)-dimensional")
    println(io, "span, and restoring them lifts channel 1 back over the value the *contained*")
    println(io, "λ/4 domain already achieves. A NO on 2 with a YES on 4 (saturated well below")
    println(io, "0.0422) falsifies it: the missing directions are not Asym(G⁰ᵤᵤ)'s leading ones,")
    println(io, "and the next suspect is the g basis itself, i.e. the gamma_rtol cut.")
    println(io, "Criterion 3 is a marker, not a requirement: the analytical bound and the dual")
    println(io, "are independent relaxations, and production already takes the min of the two")
    println(io, "per channel, so a dual above the analytical is a channel where the analytical")
    println(io, "one happens to be tighter, not a sign that the augmentation overshot.")
    print(String(take!(io)))
    return nothing
end

"""
    save_result(compute_env, result)

Everything the table is built from, into `--out`. Never touches the production
directories: `project_dir` is the experiment's own output dir by construction.
"""
function save_result(compute_env::ComputeEnvironment, result)
    path = joinpath(project_dir(compute_env), "$(result.prefix)_augmented_basis.jld")
    @info string(now()) * " [augmented_basis::save_result] Saving to $(path)"
    jldopen(path, "w") do jld
        jld["prefix"] = result.prefix
        jld["N_u"] = result.N_u
        jld["num_pos"] = result.num_pos
        jld["gamma_rtol"] = result.gamma_rtol
        jld["channels"] = result.channels
        jld["chi"] = result.χ
        jld["zeta"] = result.ζ
        jld["Gamma"] = result.Γ
        jld["Gamma_rs"] = result.Γrs
        jld["tau_grid"] = result.tau_grid
        jld["old_analytical_bounds"] = result.old_analytical
        jld["new_analytical_bounds"] = result.new_analytical
        jld["uu_values"] = result.uu_values
        jld["uu_residual_idxs"] = result.uu_residual_idxs
        jld["uu_residuals"] = result.uu_residuals
        jld["reference/quarter_wave_channel1"] = REFERENCE_QUARTER_WAVE_CHANNEL1
        jld["reference/production_channel1"] = REFERENCE_PRODUCTION_CHANNEL1
        jld["reference/production_trace_15"] = REFERENCE_PRODUCTION_TRACE_15
        if result.production !== nothing
            jld["production/bounds"] = result.production.bounds
            jld["production/opt_taus"] = result.production.opt_taus
            jld["production/seconds"] = result.production.seconds
        end
        jld["k_uu"] = [row.k_uu for row in result.rows]
        jld["m_aug"] = [row.m_aug for row in result.rows]
        jld["num_uu_dropped"] = [row.num_uu_dropped for row in result.rows]
        jld["seconds"] = [row.seconds for row in result.rows]
        for row in result.rows
            p = "k_uu_$(row.k_uu)/"
            jld[p*"bounds"] = row.bounds
            jld[p*"opt_taus"] = row.opt_taus
            jld[p*"bounds_by_tau"] = row.bounds_by_tau
            jld[p*"m_aug"] = row.m_aug
            jld[p*"num_uu_kept"] = row.num_uu_kept
            jld[p*"num_uu_dropped"] = row.num_uu_dropped
            jld[p*"pencil_rank_taus"] = [t for (t, _) in row.pencil_ranks]
            jld[p*"pencil_ranks"] = [r for (_, r) in row.pencil_ranks]
            jld[p*"seconds"] = row.seconds
            # Scalars rather than the NamedTuple, so nothing depends on JLD2's
            # handling of a type this file defines.
            for (name, secs) in pairs(row.stage_times)
                jld[p*"stage_times/"*String(name)] = secs
            end
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

const USAGE = """
bench/augmented_basis_experiment.jl: does augmenting the projection basis with
the top eigenvectors of Asym(G0_uu) restore the far-field dual bound?

  --sep A//B          separation (gap) of the 1 lambda sweep point [default 357//32]
  --sender x,y,z      sender cell counts [32,32,32]
  --receiver x,y,z    receiver cell counts [32,32,32]
  --scale A//B        cell size in wavelengths [1//32]
  --chi a+bi          susceptibility [17.06132654701751+0.29117345im]
  --scratch DIR       production RSVD scratch dir (read only)
  --preload DIR       Green function cache [\$HOME/scratch/preload on a cluster]
  --out DIR           output dir [\$SCRATCH/psc-augmented-basis]
  --gpu true|false|N  [true on a cluster]
  --gamma-rtol X      the production spectral cut [1e-12]
  --k-uu LIST         comma-separated k_uu sweep [0,32,64,128,256,512]
  --uu-oversamples N  reigen_hermitian oversamples for Asym(G0_uu) [50]
  --uu-power-iters N  reigen_hermitian power iterations for Asym(G0_uu) [4]
  --channels N        how many leading channels to bound [16]
  --qr-rtol X         augmentation rank guard [1e-10]
  --taus LIST         comma-separated tau grid [0,0.25,0.5,0.75,1]
  --tau-refine-tol X  golden-section width, or "none" [0.05]
  --no-production     skip the production bounds_from_spectrum cross-check
  --no-save           do not write the output JLD
  --help
"""

function parse_cli(args::Vector{String})
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 1
    while i <= length(args)
        a = args[i]
        startswith(a, "--") || error("unexpected positional argument '$a'\n\n$USAGE")
        key = a[3:end]
        if key in ("help", "no-production", "no-save")
            push!(flags, key)
            i += 1
            continue
        end
        i < length(args) || error("--$key needs a value\n\n$USAGE")
        opts[key] = args[i+1]
        i += 2
    end
    return opts, flags
end

parse_rat(s::AbstractString) = begin
    parts = split(strip(s), "//")
    length(parts) == 2 || error("expected a rational a//b, got '$s'")
    parse(Int, parts[1]) // parse(Int, parts[2])
end
parse_triple(s::AbstractString) = begin
    parts = split(strip(s, ['(', ')', ' ']), ",")
    length(parts) == 3 || error("expected x,y,z, got '$s'")
    (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])), parse(Int, strip(parts[3])))
end
parse_gpu(s::AbstractString) = begin
    t = lowercase(strip(s))
    t in ("true", "t", "yes", "y") && return GPUChoice(true, 0)
    t in ("false", "f", "no", "n") && return GPUChoice(false, -1)
    return GPUChoice(true, parse(Int, t))
end

function main(args::Vector{String}=copy(ARGS))
    opts, flags = parse_cli(args)
    if "help" in flags
        println(USAGE)
        return nothing
    end

    sep = parse_rat(get(opts, "sep", string(numerator(DEFAULT_SEP), "//", denominator(DEFAULT_SEP))))
    sender_cells = parse_triple(get(opts, "sender", "32,32,32"))
    receiver_cells = parse_triple(get(opts, "receiver", "32,32,32"))
    scale = parse_rat(get(opts, "scale", "1//32"))
    χ = parse(ComplexF64, get(opts, "chi", string(DEFAULT_CHI)))
    gpu = parse_gpu(get(opts, "gpu", haskey(ENV, "CC_CLUSTER") || haskey(ENV, "MOLERING") ? "true" : "false"))

    scratch = get(opts, "scratch", DEFAULT_SCRATCH)
    preload = get(opts, "preload",
                  haskey(ENV, "CC_CLUSTER") ? joinpath("/home", ENV["USER"], "scratch", "preload") :
                  joinpath(homedir(), "scratch", "preload"))
    out = get(opts, "out", joinpath(get(ENV, "SCRATCH", homedir()), "psc-augmented-basis"))
    mkpath(out)

    gamma_rtol = parse(Float64, get(opts, "gamma-rtol", string(PSC.DEFAULT_GAMMA_RTOL)))
    k_uu_list = parse.(Int, split(get(opts, "k-uu", join(DEFAULT_K_UU_SWEEP, ",")), ","))
    uu_oversamples = parse(Int, get(opts, "uu-oversamples", string(PSC.UU_OVERSAMPLES)))
    uu_power_iters = parse(Int, get(opts, "uu-power-iters", string(PSC.UU_POWER_ITERS)))
    channels = parse(Int, get(opts, "channels", "16"))
    qr_rtol = parse(Float64, get(opts, "qr-rtol", string(PSC.AUG_QR_RTOL)))
    τs = sort(parse.(Float64, split(get(opts, "taus", "0.0,0.25,0.5,0.75,1.0"), ",")))
    refine_raw = get(opts, "tau-refine-tol", "0.05")
    τ_refine_tol = lowercase(refine_raw) == "none" ? nothing : parse(Float64, refine_raw)

    # `design_regions` matches the production `--design rs`: char2volume_symbol on
    # the *sorted* characters, which is [Receiver, Sender] and is what puts "RS"
    # (not "SR") in the file prefix the RSVD output is stored under.
    design_symbols = char2volume_symbol.(sort(collect("RS")))
    smr = SMRSystem(sender_cells, (sep, 0 // 1, 0 // 1), receiver_cells,
                    design_symbols, scale, χ)

    compute_env = ComputeEnvironment(preload, out, scratch, gpu)
    if use_gpu(compute_env)
        @info string(now()) * " [augmented_basis::main] GPU device $(gpu_device(compute_env))"
        haskey(ENV, "CC_CLUSTER") || CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [augmented_basis::main] CPU run"
    end
    @info string(now()) * " [augmented_basis::main] Directories" preload scratch out
    @info string(now()) * " [augmented_basis::main] Expecting the RSVD output at " *
          joinpath(scratch, "$(file_prefix(smr)).jld")

    run_experiment(compute_env, smr; gamma_rtol=gamma_rtol, k_uu_list=k_uu_list,
                   uu_oversamples=uu_oversamples, uu_power_iters=uu_power_iters,
                   channels=channels, qr_rtol=qr_rtol, τs=τs,
                   τ_refine_tol=τ_refine_tol,
                   run_production_reference=!("no-production" in flags),
                   save=!("no-save" in flags))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
