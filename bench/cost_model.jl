"""
    CostModel

Analytic time and memory model for the sender/receiver (no mediator) pipeline,
shared by `bench/fit.jl` (which calibrates it) and `create_jobs.jl` (which uses
it to size SLURM requests).

Deliberately free of dependencies: a point is described by plain integers and
rationals, not by an `SMRSystem`, so that this file can be included anywhere.

# How it is put together

Every prediction is `count * primitive_cost`, where the counts are derived by
reading the code and the primitive costs are the only fitted quantities:

  * `GenerateGreens` builds a fixed number of Gila operator blocks. Each block
    is either a *self* block (source volume == target volume, which lets Gila
    exploit reflection symmetry and store 8x less Fourier data) or an
    *external* block. External blocks between touching bodies take a more
    expensive code path (`genEgoCrcExt!` generates a contact volume and uses
    the singular-integral corrections), so they are counted separately.

  * `GenerateRSVD` is dominated by two things: Green-function matrix-vector
    products, whose count follows exactly from `reigen_hermitian` /
    `rsvdvals` in MatrixFreeRandomizedLinearAlgebra v0.2; and dense GPU linear
    algebra on tall `N x c` matrices, which is *not* negligible at the ranks
    actually in use (c ~ 3000) and which the previous estimator ignored
    entirely.

  * `ComputeBounds` is not constant-time. Each of the `m = num_pos` indices
    runs `TAU_GRID_POINTS + TAU_REFINE_EVALS` pencil solves (the shared τ
    grid plus a per-index golden-section refinement), every one an `m x m`
    Hermitian eigendecomposition on the *device* (CUSOLVER heevd), on top of
    an `O(m^2 * evals)` probe loop -- a device gemv, a device-to-host
    transfer, and a host Brent root find per probe -- an `O(m^2)` reverse
    Gram-Schmidt over length-`N_u` vectors, and `4m` self plus `4m` external
    Green matvecs.

# Three storage paths

The tall `N x c` matrices can live in three places, and which one they live in
changes the model itself rather than only its constants:

  * `:in_memory`, a device-resident `CuArray` sketch with a Householder QR and no
    bus traffic. This is what the model above describes, and what every
    prediction with `vram_capacity_bytes === nothing` returns.

  * `:panel`, MatrixFreeRandomizedLinearAlgebra's Funicular path
    (`ext/MFRLAFunicularExt.jl`). The sketch is a `PanelMatrix` cut into column
    panels held in pinned host memory and streamed through staging buffers on
    the device, the test matrix is a `GhostPanels` that costs nothing, and the
    orthonormalization is CholeskyQR2 rather than a thin QR. Device memory stops
    scaling with `c`, host memory starts to, and the bus becomes a cost term.

  * `:dense_exact`, for `N_u <= DENSE_EXACT_MAX_N`: the operator is applied to
    the identity and eigendecomposed exactly, with no sketch at all.

`rsvd_mode` and `bounds_mode` pick between them from the card's capacity, using
the same predicate the runtime uses (`uses_dense_path`, then `uses_panel_path`),
so a prediction and the job it sizes agree on which code will run.

Sizes are always driven by the cell count of **one body**. The separation
between the bodies changes nothing about the cost except at contact. In
particular, `union(sender, receiver)` -- the bounding box that includes the gap
-- must never appear in a cost estimate: no operator is ever built on it. (The
previous estimator used it, which is why a 10000-wavelength separation asked
for terabytes.)

# Units

Times are seconds, memory is bytes, and "flops" are notional real-equivalent
floating point operations used only as regressors; the fitted rates absorb the
constant factors.
"""
module CostModel

export SRPoint, JobKind, GenerateGreens, GenerateRSVD, ComputeBounds
export Coefficients, DEFAULT_COEFFICIENTS, coefficients_for, load_coefficients!
export predict, predict_time_s, predict_host_bytes, predict_vram_bytes
export greens_counts, rsvd_counts, bounds_counts
export rsvd_panel_counts, rsvd_dense_counts, bounds_panel_counts
export uses_panel_path, uses_dense_path, rsvd_mode, bounds_mode
export TauShape, TAU_SHAPE_LEGACY, tau_shape, tau_evals_per_index, bounds_m
export BOUNDS_M_REF_SEP, rsvd_time_parts
export panel_width, panel_staging_bytes
export n_cells, vector_length, universe_length, sketch_width, is_contact
export self_fourier_bytes, ext_fourier_bytes, block_build_peak_bytes
export circulant_cells, fft_work

# --------------------------------------------------------------------------- #
# Points and jobs
# --------------------------------------------------------------------------- #

@enum JobKind GenerateGreens GenerateRSVD ComputeBounds

const BYTES_PER_COMPLEX = 16 # ComplexF64

"""
    SRPoint

One sender/receiver experiment, described by exactly the quantities that drive
its cost.

# Fields
- `sender_cells`, `receiver_cells`: cells per body.
- `scale`: cell edge length in wavelengths, per axis (anisotropic allowed).
- `separation`: surface-to-surface gap along x in wavelengths. Only its being
  zero or not matters for cost.
- `rank`, `oversamples`, `power_iters`: RSVD parameters (`k`, `p`, `q`).
- `threads`: `--cpus-per-task` / Julia threads, used for the CPU job.
- `num_pos`: number of positive `Asym(G0_ur)` eigenvalues, which sets the size
  of the bounds computation. `nothing` means "estimate it from `rank`" via
  `NUM_POS_FRACTION`; it is only known after the RSVD has run.
- `fresh_preload`: whether the shared `self/` Green function still has to be
  built (true on a cold preload directory).
"""
struct SRPoint
    sender_cells::NTuple{3,Int}
    receiver_cells::NTuple{3,Int}
    scale::NTuple{3,Rational{Int}}
    separation::Rational{Int}
    rank::Int
    oversamples::Int
    power_iters::Int
    threads::Int
    num_pos::Union{Nothing,Int}
    fresh_preload::Bool
end

function SRPoint(sender_cells::NTuple{3,Int}, receiver_cells::NTuple{3,Int};
                 scale::Union{Rational{Int},NTuple{3,Rational{Int}}}=1//32,
                 separation::Rational{Int}=0//1,
                 rank::Int=256, oversamples::Int=50, power_iters::Int=14,
                 threads::Int=4, num_pos::Union{Nothing,Int}=nothing,
                 fresh_preload::Bool=true)
    scl = scale isa Rational ? (scale, scale, scale) : scale
    return SRPoint(sender_cells, receiver_cells, scl, separation, rank,
                   oversamples, power_iters, threads, num_pos, fresh_preload)
end

"""
    NUM_POS_FRACTION

Fraction of the computed rank that comes back with a positive `Asym(G0_ur)`
eigenvalue, used when the true `num_pos` is unknown. Measured across the
existing `data analysis/data` outputs it runs 0.22-0.52, clustering near 0.5
for the larger runs; 0.6 is a deliberately pessimistic default because the
bounds cost grows superlinearly in `num_pos`.
"""
const NUM_POS_FRACTION = 0.6

"""
    TAU_GRID_POINTS, TAU_REFINE_EVALS

Shape of the τ optimization in `bounds_from_spectrum` (`src/bounds.jl`): the
power-conservation constraint `C(τ)` is scanned over a `TAU_GRID_POINTS`-point
grid whose whiteners are eigendecomposed once and shared across every index,
then each index runs a golden-section refinement averaging `TAU_REFINE_EVALS`
extra dual evaluations (measured at the default `τ_refine_tol = 0.05`), each of
which builds its own throwaway whitener. Update these if the `τs` /
`τ_refine_tol` defaults in `bounds_from_spectrum` change.
"""
const TAU_GRID_POINTS = 5
const TAU_REFINE_EVALS = 6

"""
    TauShape

How much τ work one outer index actually costs. Four numbers, because the
windowed sweep and the refinement pencil cache in `bounds_from_spectrum`
(`src/bounds.jl`) changed three of them independently and the old model folded
them into two constants:

- `grid_points`: τ grid points whose whiteners are eigendecomposed once, up
  front, and shared by every index. Unchanged by the window: the whole grid is
  still built before the loop.
- `grid_evals`: **GEVP dual evaluations per index over the grid**. The full sweep
  makes this `grid_points`; the windowed sweep makes it about
  `min(2*tau_window + 1, grid_points)`, and less again when the minimiser sits at
  a grid end, which is where it usually sits. No whitening is involved -- the
  window reuses the shared grid pencils -- so this term is pure `diag_pencil_eigen`
  plus probes.
- `refine_evals`: extra dual evaluations per index from the golden-section
  refinement.
- `refine_whitenings`: **new** `m x m` whiteners per index, that is, refinement
  probes that *miss* the pencil cache. Consecutive indices on a τ* plateau open
  the identical bracket and probe the identical τ, so on a plateau this is ~0
  while `refine_evals` stays at its full value. Charging one whitening per
  refinement probe (what the old model did) over-charges the dominant term of the
  whole job by the plateau length.
- `cache_entries`: how many refinement whiteners the LRU cache holds. Each is an
  `m x m` whitener plus its null space, so it is a device-memory term
  (`pencil_cache_max` in `bounds_from_spectrum`). Zero in the legacy shape,
  because the code the legacy rows were measured on had no cache.

`TAU_SHAPE_LEGACY` reproduces the pre-window model exactly, and is the default
everywhere, so a `Coefficients` that has not been refitted -- and every
`coeffs_<cluster>.jl` written before this existed -- predicts precisely what it
predicted before.
"""
struct TauShape
    grid_points::Float64
    grid_evals::Float64
    refine_evals::Float64
    refine_whitenings::Float64
    cache_entries::Float64
end

const TAU_SHAPE_LEGACY = TauShape(TAU_GRID_POINTS, TAU_GRID_POINTS,
                                  TAU_REFINE_EVALS, TAU_REFINE_EVALS, 0.0)

"Total dual evaluations per index: grid sweep plus refinement probes."
tau_evals_per_index(t::TauShape) = t.grid_evals + t.refine_evals

"""
    BOUNDS_M_REF_SEP

Reference separation for the `--gamma-rtol` truncation model (see `bounds_m`), in
wavelengths. The nearest gap in the production sweeps, hence the one where the
kept count is largest and the model is anchored rather than extrapolated.
"""
const BOUNDS_M_REF_SEP = 1 // 32

"""
    RECOMPILE_OVERHEAD_S

Flat per-job tax added to every padded time request. Some clusters invalidate
Julia's compilation cache between jobs (heterogeneous CPU microarchitectures
behind one queue, cleaned scratch caches), and a full recompile of this package
stack costs about 700 s regardless of the job's size. Applied after `time_pad`
so the request grows by exactly this amount, and only on the padded path so
`bench/fit.jl`'s comparisons of raw predictions against measurements are
unaffected.
"""
const RECOMPILE_OVERHEAD_S = 700.0

n_cells(cells::NTuple{3,Int}) = prod(cells)
sender_cells_count(pt::SRPoint) = n_cells(pt.sender_cells)
receiver_cells_count(pt::SRPoint) = n_cells(pt.receiver_cells)

"Length of a polarisation vector on one body (3 components per cell)."
vector_length(cells::NTuple{3,Int}) = 3 * n_cells(cells)

"""
    universe_length(pt)

Length of a vector on the *universe*, which in this pipeline is the
concatenation `[sender; receiver]` -- not the bounding box. See
`asym_ur` in `src/rsvd.jl` and `projected_operators` in `src/bounds.jl`.
"""
universe_length(pt::SRPoint) = vector_length(pt.sender_cells) + vector_length(pt.receiver_cells)

is_contact(pt::SRPoint) = iszero(pt.separation)

effective_num_pos(pt::SRPoint) =
    pt.num_pos === nothing ? max(1, min(pt.rank, ceil(Int, NUM_POS_FRACTION * pt.rank))) :
    max(1, min(pt.rank, pt.num_pos))

"""
    sketch_width(pt; clamp_to=nothing)

Number of random test vectors, `c = k + p`. `reigen_hermitian` never clamps
this to the operator size, but `rsvd`/`rsvdvals` clamp it to
`min(size(operator)...)`, so pass `clamp_to` for those.
"""
function sketch_width(pt::SRPoint; clamp_to::Union{Nothing,Int}=nothing)
    c = pt.rank + pt.oversamples
    return clamp_to === nothing ? c : min(c, clamp_to)
end

# --------------------------------------------------------------------------- #
# Gila block geometry: exact byte counts
# --------------------------------------------------------------------------- #

"""
    circulant_cells(target_cells, source_cells)

Cells in the circulant embedding Gila builds for a block, `prod(trgCel + srcCel)`.
For two equal bodies of `n` cells this is `8n`: the embedding doubles every
axis, which is the single biggest reason Green-function generation is memory
hungry.
"""
circulant_cells(target_cells::NTuple{3,Int}, source_cells::NTuple{3,Int}) =
    prod(target_cells .+ source_cells)

"Work unit for one FFT-based pass over a circulant of `M` cells."
fft_work(M::Real) = M * log2(max(M, 2))

"""
    self_fourier_bytes(cells)

Bytes of Fourier data a *self* block retains (`GlaVacOprMem.egoFur`). Gila keeps
8 parity branches of `truInf` unique entries with 6 unique tensor components;
for a self block `truInf_i = max(ceil(C_i/2) + iseven(C_i), 2)`, i.e. roughly
`n/8` entries per branch.
"""
function self_fourier_bytes(cells::NTuple{3,Int})
    tru = ntuple(i -> max(cld(cells[i], 2) + (iseven(cells[i]) ? 1 : 0), 2), 3)
    return 8 * prod(tru) * 6 * BYTES_PER_COMPLEX
end

"""
    ext_fourier_bytes(target_cells, source_cells)

Bytes of Fourier data an *external* block retains. Here `truInf_i` is the full
half-circulant `(Ct_i + Cs_i) / 2`, so an external block between equal bodies
stores 8x more than a self block: `768 * n` bytes versus `96 * n`.
"""
function ext_fourier_bytes(target_cells::NTuple{3,Int}, source_cells::NTuple{3,Int})
    tot = target_cells .+ source_cells
    tru = ntuple(i -> tot[i] ÷ 2, 3)
    return 8 * prod(tru) * 6 * BYTES_PER_COMPLEX
end

"""
    block_build_peak_bytes(target_cells, source_cells; self)

Peak transient host memory while `GlaVacOprMem(cmpInf, trgVol, srcVol)` builds
one block, from the allocations in `vacuum/glaVacOprMem.jl`:

  * `egoCrc`      `9 * M` complex -- all nine tensor components of the circulant
  * `egoFurPrp`   `6 * M` complex -- its Fourier transform, six unique components
  * `egoFurInt`   `6 * M/8` complex -- parity-branch staging buffer
  * `egoFur`      the retained data (see `self_fourier_bytes` / `ext_fourier_bytes`)

`egoCrc` alone is `144 * M` bytes, which for equal bodies is `1152 * n` -- about
24 polarisation vectors. This is why "the Green function costs about one vector"
underestimates by more than an order of magnitude.
"""
function block_build_peak_bytes(target_cells::NTuple{3,Int}, source_cells::NTuple{3,Int};
                               self::Bool)
    M = circulant_cells(target_cells, source_cells)
    tot = target_cells .+ source_cells
    ego_crc = 9 * M * BYTES_PER_COMPLEX
    ego_fur_prp = 6 * M * BYTES_PER_COMPLEX
    ego_fur_int = 6 * prod(ntuple(i -> max(tot[i] ÷ 2, 2), 3)) * BYTES_PER_COMPLEX
    retained = self ? self_fourier_bytes(target_cells) :
                      ext_fourier_bytes(target_cells, source_cells)
    return ego_crc + ego_fur_prp + ego_fur_int + retained
end

# --------------------------------------------------------------------------- #
# Notional flop counts for dense primitives
# --------------------------------------------------------------------------- #
#=
Only used as regressors; the fitted rates in `Coefficients` absorb every
constant factor, so being off by 2x here is harmless as long as the *shape* in
(m, c) is right.
=#

"Thin QR of a complex `m x c` matrix, including the explicit Q (geqrf + orgqr)."
flops_qr(m::Real, c::Real) = 16 * (m * c^2 - c^3 / 3)

"""
    flops_cholqr2(m, c)

CholeskyQR2 of a complex `m x c` matrix, which is how the panel path
orthonormalizes instead of a thin QR (`Funicular.cholqr2!`, called from
`panel_range_finder` in `ext/MFRLAFunicularExt.jl`).

Two passes of `cholqr_pass!`, each a `gram` (`Y' Y`, `8 m c^2`) followed by an
`rdiv_rows!` against the Cholesky factor (`8 m c^2` again), so `32 m c^2`. The
`c x c` Cholesky inside each pass is dropped, as `flops_qr` drops its `c^3` term:
`m >> c` everywhere this model is used. The work is all gemm, which is why the
panel path rates it at `gemm_rate` and not at the lower `qr_rate`.
"""
flops_cholqr2(m::Real, c::Real) = 32 * m * c^2

"Cholesky factorization of a complex `c x c` matrix (`c^3/3` complex MACs)."
flops_cholesky(c::Real) = 8 * c^3 / 3
"Complex gemm `(m x k) * (k x n)`."
flops_gemm(m::Real, k::Real, n::Real) = 8 * m * k * n
"Hermitian eigendecomposition of a `c x c` matrix."
flops_eigh(c::Real) = 30 * c^3
"Generalized Hermitian eigendecomposition of a `c x c` pencil."
flops_geigh(c::Real) = 60 * c^3
"Singular values only of a `c x c` matrix."
flops_svdvals(c::Real) = 20 * c^3

# --------------------------------------------------------------------------- #
# Coefficients
# --------------------------------------------------------------------------- #

"""
    Coefficients

Fitted primitive costs for one cluster. Everything the model does not derive
from code structure lives here, so recalibrating means replacing one of these
and nothing else.

# Green-function block construction (CPU, host)
- `g0_self_fft`, `g0_ext_fft`: seconds per unit of `M*log2(M)` (FFT of the
  circulant, single-threaded in Gila -- FFTW's threads are never enabled).
- `g0_self_cell`, `g0_ext_cell`: seconds per circulant cell of quadrature work,
  at one thread. Divided by `thread_efficiency`.
- `g0_contact_fft`, `g0_contact_cell`, `g0_contact_fixed`: the same triple for an
  external block between *touching* bodies, which is a different code path rather
  than a surcharge on the ordinary one. `genEgoCrcExt!` detects contact, builds a
  self Green function on a small contact volume (paying the expensive `O(1)`
  Gauss-Legendre setup), and then evaluates cells against that table instead of
  through `egoFunOut!`'s adaptive cubature. Measured consequence: contact costs a
  large fixed amount and very little per cell, so it dominates at small sizes and
  is *cheaper* than the ordinary path by 32 cells a side. Modelling it as
  `ext + surcharge` cannot represent that and mispredicts by 3-5x.
- `g0_self_fixed`, `g0_ext_fixed`: per-block fixed cost. Split by block kind
  because they are not remotely similar: a self block pays for the
  Gauss-Legendre setup and the weakly-singular coincident/adjacent-cell
  corrections (`wekS` / `wekE` / `wekV`), which are `O(1)` in the cell count but
  expensive, so a self block's cost is nearly flat over small sizes while an
  external block's is not.
- `g0_thread_scaling`: `s` in `eta(T) = 1 + s*(T-1)`; `s = 1` is perfect
  scaling of the quadrature loops.
- `g0_startup_s`: process startup, package load and JIT for the CPU job.
- `disk_write_rate`, `disk_read_rate`: bytes/s for serialising operators and
  reading the RSVD output.

# Green matvecs (GPU)
- `mv_self_fft`, `mv_ext_fft`: seconds per unit of `M*log2(M)` for one matvec.
- `mv_self_fixed`, `mv_ext_fixed`: per-matvec launch overhead.

# Dense GPU linear algebra
- `qr_rate`, `gemm_rate`, `eigh_rate`, `geigh_rate`, `svdvals_rate`: notional
  flops/s for the corresponding primitive.
- `bandwidth`: device bytes/s for BLAS-1 traffic.
- `launch_latency`: seconds per kernel launch, which dominates the
  `O(num_pos^2)` BLAS-1 loops.
- `sync_latency`: seconds per device-to-host synchronisation, paid once per
  probe when the projected b-vector crosses to the host for the root find.
- `host_root_find`: seconds per unit of `probes * m` for the host-side Brent
  root find in the bounds inner loop.

# Memory
- `*_mem_factor`: multiplier on the analytic byte count, absorbing allocator
  slack, garbage that has not been collected yet, and library workspaces.
- `*_mem_base`: fixed footprint of a process of that kind (Julia + CUDA +
  package images).

# Funicular panel path
- `panel_host_mem_factor`: multiplier on the analytic *host* byte count when the
  tall matrices live in pinned host memory. Much smaller than
  `rsvd_host_mem_factor` has to be, because the plan owns page-locked slabs and
  hands out one block per panel, so the analytic count is what is actually
  mapped. What is left over is slab doubling, the process baseline already being
  in `rsvd_host_mem_base`.
- `panel_workspace_bytes`: device memory the operator itself takes while it is
  applied, held back from the panel buffer pool (Funicular's `workspace_bytes`).
  `G0_ur_asym` is a `LinearMaps` composition and cannot carry the trait, so the
  number has to be supplied: CUFFT plan work areas plus the composition's
  per-apply `N_u`-vector temporaries. Trial E2 measures it. If it goes
  unreported, the budget arithmetic hands the panel buffers memory the operator
  will then take, and the device overflows mid-sweep.
- `pcie_rate`: achievable pinned host-to-device bytes/s, one direction. Not the
  link's nominal rate: it is what Funicular's own `benchmark/pinned.jl` measures
  end to end (trial E1).
- `overlap_factor`: fraction of the sweep traffic that is *not* hidden behind the
  operator applies. Funicular stages `nbuffers` panels ahead, so where the apply
  dominates most of the transfer disappears into it, and 0.15 says 85% hides. It
  is a single number standing in for a pipeline, so it only means anything where
  compute does dominate. That is the regime the panel path runs in, since it is
  only chosen when the sketch is huge.

# Padding
- `time_pad`, `host_mem_pad`, `vram_pad`: safety factors applied by `predict`.
  Set these from the measured spread (see `bench/fit.jl`), not by taste.
- `panel_host_mem_pad`: `host_mem_pad`'s replacement on the panel path. The two
  are padding different uncertainties. `host_mem_pad` is the p95 of
  measured/predicted host RSS over in-memory runs, where the analytic count is a
  *floor* under a Julia heap the GC may not have swept, with JLD2's buffers on
  top. On the panel path the count is slab arithmetic: Funicular preallocates
  page-locked slabs and hands out one block per panel, and the positives-only
  save never forms a host copy at all. Applying the in-memory p95 on top of the
  already-tight `panel_host_mem_factor` would count the same margin twice, and at
  4 λ, `k = 4000` that difference decides whether the job fits the 124.5 GB
  single-card bundle. 1.05 covers what is left, that is, the slab allocator's
  rounding and its doubling growth.
"""
Base.@kwdef struct Coefficients
    name::String = "uncalibrated"
    calibrated::Bool = false

    g0_self_fft::Float64 = 1.0e-7
    g0_ext_fft::Float64 = 1.2e-7
    g0_self_cell::Float64 = 4.0e-6
    g0_ext_cell::Float64 = 5.0e-6
    g0_contact_fft::Float64 = 0.0
    g0_contact_cell::Float64 = 2.0e-7
    g0_contact_fixed::Float64 = 20.0
    g0_self_fixed::Float64 = 12.0
    g0_ext_fixed::Float64 = 2.0
    g0_thread_scaling::Float64 = 0.7
    g0_startup_s::Float64 = 60.0
    disk_write_rate::Float64 = 2.0e8
    disk_read_rate::Float64 = 4.0e8

    mv_self_fft::Float64 = 4.0e-10
    mv_ext_fft::Float64 = 4.0e-10
    mv_self_fixed::Float64 = 3.0e-4
    mv_ext_fixed::Float64 = 3.0e-4

    qr_rate::Float64 = 2.0e12
    gemm_rate::Float64 = 6.0e12
    eigh_rate::Float64 = 5.0e11
    geigh_rate::Float64 = 4.0e11
    svdvals_rate::Float64 = 4.0e11
    bandwidth::Float64 = 8.0e11
    launch_latency::Float64 = 1.0e-5
    sync_latency::Float64 = 2.0e-5
    host_root_find::Float64 = 5.0e-7

    gpu_startup_s::Float64 = 120.0

    #=
    Memory defaults are deliberately generous, because they are what gets used
    when a cluster has no end-to-end runs yet and because the failure modes are
    wildly asymmetric: an over-request costs some queue time, an under-request
    costs the entire job at the point where it has already done all the work.

    The factors sit at 2x the analytic count (which comes from reading the
    allocations in the code, so it is a floor, not a guess) plus a base that
    covers the Julia process, the package images and the CUDA context. Measured
    Green-job memory landed at 1.36-1.64x analytic + ~1.6 GiB across three
    clusters, so 2x + 4 GB is comfortably above what the one calibrated case
    actually needs.
    =#
    greens_mem_factor::Float64 = 2.0
    greens_mem_base::Float64 = 4.0e9
    rsvd_host_mem_factor::Float64 = 1.5
    rsvd_host_mem_base::Float64 = 4.0e9
    rsvd_vram_factor::Float64 = 2.0
    rsvd_vram_base::Float64 = 3.0e9

    #=
    Smallest observed ratio of real device high-water to the analytic live-array
    count, used for the *feasibility* test rather than for the request.

    Device memory in this pipeline is churn-elastic: CUDA.jl's pool grows to hold
    garbage Julia has not collected, so a job with a 5 GB working set was measured
    taking 37 GB on an idle 80 GB card, while one with a 46 GB working set fitted
    in 71 GB on the same card. The observed peak is "whatever the allocator felt
    like taking", not demand. What it cannot go below is this floor, so this is the
    number that decides whether a card is big enough -- and the request is capped
    at the card, because asking for more than exists just gets the job rejected.
    Measured minimum across fir's mem_rsvd points: 1.56.
    =#
    vram_floor_factor::Float64 = 1.6
    bounds_host_mem_factor::Float64 = 3.0
    bounds_host_mem_base::Float64 = 6.0e9
    bounds_vram_factor::Float64 = 2.0
    bounds_vram_base::Float64 = 3.0e9

    #=
    The τ-search shape (see `TauShape`) and the `--gamma-rtol` truncation model
    (see `bounds_m`). Both default to the pre-window, pre-truncation behaviour, so
    every coefficient file written before they existed keeps predicting exactly
    what it predicted before; `bench/fit.jl` flips the modes only when the
    calibration rows carry the columns that identify them.

    `bounds_tau_mode`:
      "legacy"   -- `TAU_SHAPE_LEGACY`: the whole grid at every index, one throwaway
                    whitening per refinement probe, no pencil cache.
      "measured" -- the four numbers below, taken from `stage_bounds` rows that
                    reported them.

    `bounds_m_mode`:
      "fraction"  -- `m = NUM_POS_FRACTION * rank`, capped at the rank and `N_u`.
      "truncated" -- additionally capped by the power law
                     `bounds_m_ref * (sep / BOUNDS_M_REF_SEP)^bounds_m_exponent`,
                     which is the `--gamma-rtol` cut's kept count as a function of
                     separation. `bounds_m_exponent` is negative: the further apart
                     the bodies, the fewer directions of `Asym(G0_ur)` sit above
                     the RSVD's noise floor, and at the far end of a sweep the
                     positive block shrinks by more than an order of magnitude.
    =#
    bounds_tau_mode::String = "legacy"
    bounds_tau_grid_points::Float64 = TAU_GRID_POINTS
    bounds_tau_grid_evals::Float64 = TAU_GRID_POINTS
    bounds_tau_refine_evals::Float64 = TAU_REFINE_EVALS
    bounds_tau_refine_whitenings::Float64 = TAU_REFINE_EVALS
    bounds_tau_cache_entries::Float64 = 0.0

    bounds_m_mode::String = "fraction"
    bounds_m_ref::Float64 = 0.0
    bounds_m_exponent::Float64 = 0.0

    #=
    Multiplier on the part of the RSVD's predicted time that scales with the power
    iteration count `q`, that is, on the `(2q + 2)`-ish operator passes and the
    per-pass dense work. Identified by running the same geometry at two low `q`
    (the `backfill` tier's B points): the slope in `q` is the per-pass cost and the
    intercept is everything that happens once. 1.0 leaves the fitted matvec and
    gemm rates as the only per-pass estimate, which is what they were before.
    =#
    rsvd_pass_scale::Float64 = 1.0

    #=
    Panel-path coefficients. All four are uncalibrated defaults, replaced by
    workstream E's funicular tier: E1 for the bus pair, E2/E3 for the two memory
    numbers. They are `@kwdef` defaults rather than required fields so that every
    `coeffs_<cluster>.jl` written before they existed still loads.

    `panel_host_mem_factor` is close to 1 on purpose. The analytic host count on
    this path is exact (two `N_u x c` panel matrices' worth of pinned slabs), so
    the only slack is the slab allocator's doubling. On the in-memory path the
    count is a floor under whatever CUDA.jl and the GC decided to hold.
    =#
    panel_host_mem_factor::Float64 = 1.1
    panel_workspace_bytes::Float64 = 1.5e9
    pcie_rate::Float64 = 20.0e9
    overlap_factor::Float64 = 0.15

    time_pad::Float64 = 1.5
    # Small on purpose: the memory margin lives in `*_mem_factor`, which multiplies
    # an analytic count read off the allocations. This is only for run-to-run slop.
    host_mem_pad::Float64 = 1.15
    vram_pad::Float64 = 1.15
    # Smaller still, and not fitted from the in-memory runs: see the field docs.
    panel_host_mem_pad::Float64 = 1.05
end

thread_efficiency(coeffs::Coefficients, threads::Int) =
    1 + coeffs.g0_thread_scaling * (max(1, threads) - 1)

"""
    tau_shape(c) -> TauShape

The τ-search shape these coefficients describe. `TAU_SHAPE_LEGACY` unless the fit
set `bounds_tau_mode = "measured"`.
"""
tau_shape(c::Coefficients) =
    c.bounds_tau_mode == "measured" ?
        TauShape(c.bounds_tau_grid_points, c.bounds_tau_grid_evals,
                 c.bounds_tau_refine_evals, c.bounds_tau_refine_whitenings,
                 c.bounds_tau_cache_entries) :
        TAU_SHAPE_LEGACY

"""
    bounds_m(pt, c) -> Int

The `m` the bounds job actually runs at: the number of positive `Asym(G0_ur)`
eigenvalues that survive `--gamma-rtol`.

`effective_num_pos` is the starting point (a measured `num_pos` when the point
carries one, `NUM_POS_FRACTION * rank` otherwise). In
`bounds_m_mode = "truncated"` a separation-dependent cap is then applied, because
the spectral cut in `load_bounds_inputs` throws away every direction below
`gamma_rtol * Γ[1]` and how many that leaves depends strongly on the gap: near
contact almost the whole positive block survives, and at the far end of a sweep
about fifty columns of eighteen hundred do. The bounds cost is superlinear in `m`
in both time (the outer loop is `O(m^2)` evaluations of `m x m` problems) and
memory, so charging the near-contact `m` at every separation over-requests the
far end of a sweep by more than an order of magnitude. That is what an 18 h
request for a job that runs in minutes looks like.

A cap and not a replacement: whatever the RSVD produced is still the ceiling, and
a measured `num_pos` on a row that was itself produced under the cut is already
the post-truncation count.
"""
function bounds_m(pt::SRPoint, c::Coefficients)
    m = min(effective_num_pos(pt), universe_length(pt))
    c.bounds_m_mode == "truncated" || return m
    c.bounds_m_ref > 0 || return m
    # Contact has no gap to scale by, and is the near end of the family anyway.
    iszero(pt.separation) && return m
    ratio = Float64(pt.separation / BOUNDS_M_REF_SEP)
    ratio > 0 || return m
    cap = c.bounds_m_ref * ratio^c.bounds_m_exponent
    isfinite(cap) || return m
    return clamp(round(Int, cap), 1, m)
end

"""
    DEFAULT_COEFFICIENTS

Per-cluster coefficient registry. Entries start uncalibrated (the analytic
guesses above) and are replaced by `load_coefficients!` once
`bench/coeffs_<cluster>.jl` exists.
"""
const DEFAULT_COEFFICIENTS = Dict{String,Coefficients}(
    "molering" => Coefficients(name="molering"),
    "narval" => Coefficients(name="narval"),
    "fir" => Coefficients(name="fir"),
)

"""
    coefficients_for(cluster) -> Coefficients

Look up a cluster, falling back to an uncalibrated set with a warning so that a
missing calibration is loud rather than silent.
"""
function coefficients_for(cluster::AbstractString)
    haskey(DEFAULT_COEFFICIENTS, cluster) && return DEFAULT_COEFFICIENTS[cluster]
    @warn "No cost-model coefficients for cluster '$cluster'; using uncalibrated defaults."
    return Coefficients(name=cluster)
end

"""
    load_coefficients!(dir=@__DIR__)

Load every `coeffs_<cluster>.jl` in `dir` into `DEFAULT_COEFFICIENTS`. Each such
file is generated by `bench/fit.jl` and must evaluate to a `Coefficients`.
"""
function load_coefficients!(dir::AbstractString=@__DIR__)
    loaded = String[]
    isdir(dir) || return loaded
    for file in sort(readdir(dir))
        m = match(r"^coeffs_(.+)\.jl$", file)
        m === nothing && continue
        cluster = m.captures[1]
        try
            coeffs = include(joinpath(dir, file))
            coeffs isa Coefficients ||
                error("$file must evaluate to a CostModel.Coefficients, got $(typeof(coeffs))")
            DEFAULT_COEFFICIENTS[cluster] = coeffs
            push!(loaded, cluster)
        catch err
            @warn "Failed to load coefficients from $file" exception = err
        end
    end
    return loaded
end

# --------------------------------------------------------------------------- #
# Which code path will run
# --------------------------------------------------------------------------- #

"""
    DENSE_EXACT_MAX_N

Universe length at or below which `src/rsvd.jl` skips the randomized
factorization and computes the spectrum exactly: build dense `Asym(G0_ur)` by
applying the operator to the identity, `eigen!` it on the device, save the
positive prefix.

12,288 covers the 1/4 λ cube (`N_u = 3,072`) with room to spare. It is a policy,
not a threshold that falls out of anything. Past `N_u ~ k` the "rank" is the whole
spectrum, so a rank-`k` sketch of a `3,072 x 3,072` operator at `k = 4,000`
approximates nothing: it is a more expensive and less accurate way to compute a
full eigendecomposition. Keep this in step with `DENSE_EXACT_MAX_N_U` in
`src/rsvd.jl`, which is the same number under a different name.
"""
const DENSE_EXACT_MAX_N = 12_288

"""
    PANEL_PATH_DEVICE_FRACTION, PANEL_PATH_FLOOR_FACTOR

The two constants in the path predicate, mirroring what `src/rsvd.jl` computes.

`PANEL_PATH_DEVICE_FRACTION` is the plan's `device_budget` as a fraction of the
card's total. Ten percent is held back for the driver and the pool's own
bookkeeping, which is the reserve Funicular's residency page recommends, and it is
why the budget is taken from `CUDA.total_memory()` rather than from
`CUDA.free_memory()` (the latter under-reports once the pool holds cached blocks).

`PANEL_PATH_FLOOR_FACTOR` is narval's fitted `vram_floor_factor`, that is, the
*smallest* ratio of real device high-water to the analytic live-array count ever
measured for this pipeline. It appears here rather than the request-sizing
`rsvd_vram_factor` because the predicate asks a feasibility question, can the
in-memory sketch be squeezed onto this card at all, and `vram_floor_factor` is the
number that answers it. Pass a `Coefficients` to the three-argument form to use
that cluster's fitted value instead of this default.
"""
const PANEL_PATH_DEVICE_FRACTION = 0.9
const PANEL_PATH_FLOOR_FACTOR = 1.554

"""
    TARGET_PANEL_BYTES, PANEL_STAGING_BUFFERS

Funicular's panel geometry, from `ResidencyPlan`'s defaults.

`TARGET_PANEL_BYTES` is `target_panel_bytes = 2 GiB`. Left alone, the plan picks
the widest panel whose staging buffers fit the device budget, capped here. Two
gigabytes is the middle of the one-to-four-gigabyte range the design targets:
wider panels use the bus better, but cost more device memory per buffer and give
the pipeline a coarser unit of work to hide.

`PANEL_STAGING_BUFFERS` is the plan's `nbuffers = 2` plus the one extra buffer the
residency page lists for the operations that need a step beyond the sweep's own
(`project`, which is a panel, and the in-place `rightmul!`, which is a row block).
Three is therefore the count for a sweep that hits those, that is, the peak, which
is what a memory request wants.
"""
const TARGET_PANEL_BYTES = 2^31
const PANEL_STAGING_BUFFERS = 3

"""
    panel_width(N, c) -> Int

Columns per panel that `Funicular.choose_panel_width` will land on for an
`N x c` matrix: the widest panel of at most `TARGET_PANEL_BYTES`, at least one
column, never more than the whole matrix.

The real function also clamps against `device_budget / nbuffers`, which is the
binding constraint on a small MIG slice. That is left out because the panel path
is only selected when `N * c * 16` is enormous, and by then
`TARGET_PANEL_BYTES / (N * 16)` is the smaller of the two. Funicular also evens
the width out over the panel count it implies (a `k = 1000` matrix allowing
`w = 999` is cut 500/500 rather than 999/1), which changes the ragged tail, not
the staging-buffer size this is used for.
"""
panel_width(N::Integer, c::Integer; target_panel_bytes::Real=TARGET_PANEL_BYTES) =
    clamp(fld(floor(Int, target_panel_bytes), max(N, 1) * BYTES_PER_COMPLEX), 1, c)

"""
    panel_staging_bytes(N, c; matrices=2)

Device memory the staging buffers take for a sweep over `matrices` panel matrices
of shape `N x c`: `PANEL_STAGING_BUFFERS` buffers per operand, each one panel.

Two matrices is the peak for both jobs of interest here. The Hermitian power
iteration's `panelmul!(Z, operator, Y)` and `gram(Q, Z)` each have two operands,
and so does the bounds front-end's `rightmul!(ss, basis, T)`. This is the whole of
the panel path's `c`-dependence on the device, and it does *not* grow with `c`:
panels get more numerous, not larger.
"""
panel_staging_bytes(N::Integer, c::Integer; matrices::Integer=2) =
    PANEL_STAGING_BUFFERS * matrices * N * panel_width(N, c) * BYTES_PER_COMPLEX

"""
    rsvd_operator_vram_bytes(pt)

Device bytes the Green operators hold for the whole RSVD job, on every path:
`G0_rs`'s external Fourier data plus `G0_rr` and the `Asym` copy `src/rsvd.jl`
builds alongside it.
"""
rsvd_operator_vram_bytes(pt::SRPoint) =
    ext_fourier_bytes(pt.receiver_cells, pt.sender_cells) +
    2 * self_fourier_bytes(pt.receiver_cells)

"""
    rsvd_inmemory_live_bytes(pt)

Live device bytes of the in-memory RSVD: the three `N_u x c` complex matrices the
Hermitian range finder holds at once (`Omega`, `Q`, and `operator * Q`) plus the
operators. This is both `rsvd_counts(pt).vram_bytes` and the quantity the path
predicate tests, written once so the two cannot drift apart.
"""
rsvd_inmemory_live_bytes(pt::SRPoint) =
    3 * universe_length(pt) * sketch_width(pt) * BYTES_PER_COMPLEX +
    rsvd_operator_vram_bytes(pt)

"""
    uses_dense_path(pt) -> Bool

Whether the universe is small enough for the dense-exact branch (`N_u <=
DENSE_EXACT_MAX_N`). Independent of the card: at these sizes everything fits
everywhere, and the reason to take this branch is accuracy, not memory.
"""
uses_dense_path(pt::SRPoint) = universe_length(pt) <= DENSE_EXACT_MAX_N

"""
    uses_panel_path(pt, vram_capacity_bytes; floor_factor=PANEL_PATH_FLOOR_FACTOR)
    uses_panel_path(pt, vram_capacity_bytes, coeffs)

Whether the RSVD will hand MatrixFreeRandomizedLinearAlgebra a Funicular
`ResidencyPlan` instead of keeping the sketch on the device, that is, whether the
in-memory sketch cannot be squeezed into `PANEL_PATH_DEVICE_FRACTION` of the card.

One predicate, evaluated identically here and in `use_panel_path` in `src/rsvd.jl`
(the names differ by a letter; they must not differ in what they compute). The cost
model is what sizes the SLURM request and picks the MIG slice, so guessing a
different path than the job then takes gets the request wrong either way: a
panel-path job given an in-memory VRAM request wastes a whole card, and an
in-memory job given panel-path host memory dies at the first `Array`.

`vram_capacity_bytes === nothing` means no card was named, and returns `false`.
The in-memory path is the historical behaviour, and the one every existing caller
and every fitted coefficient describes.

Note, however, that this does *not* consult `uses_dense_path`. The runtime checks
the dense branch first and never reaches the plan question for a tiny universe,
and `rsvd_mode` reproduces that ordering. Keeping this function to the memory
question alone makes it directly comparable with `use_panel_path` in `src/rsvd.jl`.
"""
function uses_panel_path(pt::SRPoint, vram_capacity_bytes::Union{Nothing,Real};
                         floor_factor::Real=PANEL_PATH_FLOOR_FACTOR)
    vram_capacity_bytes === nothing && return false
    return floor_factor * rsvd_inmemory_live_bytes(pt) >
           PANEL_PATH_DEVICE_FRACTION * vram_capacity_bytes
end

uses_panel_path(pt::SRPoint, vram_capacity_bytes::Union{Nothing,Real},
                coeffs::Coefficients) =
    uses_panel_path(pt, vram_capacity_bytes; floor_factor=coeffs.vram_floor_factor)

"""
    rsvd_mode(pt, vram_capacity_bytes[, coeffs]) -> Symbol
    bounds_mode(pt, vram_capacity_bytes[, coeffs]) -> Symbol

Which of `:in_memory`, `:panel`, `:dense_exact` the job will run, in the order the
runtime tests them: dense-exact first (a tiny universe never wants a sketch), then
the memory predicate.

`bounds_mode` never returns `:dense_exact`. The bounds job has no dense-exact
branch, and a universe small enough to trigger one has an `N_u x m` basis that
fits on any card, which is the in-memory path already. It follows the *RSVD's*
predicate rather than one of its own because the two jobs must agree: the bounds
job reads what the RSVD wrote, and the RSVD's choice of path is what decides
whether that is an h5 panel dataset or a dense JLD2 block.
"""
function rsvd_mode(pt::SRPoint, vram_capacity_bytes::Union{Nothing,Real};
                   floor_factor::Real=PANEL_PATH_FLOOR_FACTOR)
    vram_capacity_bytes === nothing && return :in_memory
    uses_dense_path(pt) && return :dense_exact
    uses_panel_path(pt, vram_capacity_bytes; floor_factor=floor_factor) && return :panel
    return :in_memory
end

rsvd_mode(pt::SRPoint, vram_capacity_bytes::Union{Nothing,Real}, coeffs::Coefficients) =
    rsvd_mode(pt, vram_capacity_bytes; floor_factor=coeffs.vram_floor_factor)

function bounds_mode(pt::SRPoint, vram_capacity_bytes::Union{Nothing,Real};
                     floor_factor::Real=PANEL_PATH_FLOOR_FACTOR)
    mode = rsvd_mode(pt, vram_capacity_bytes; floor_factor=floor_factor)
    return mode == :panel ? :panel : :in_memory
end

bounds_mode(pt::SRPoint, vram_capacity_bytes::Union{Nothing,Real}, coeffs::Coefficients) =
    bounds_mode(pt, vram_capacity_bytes; floor_factor=coeffs.vram_floor_factor)

# --------------------------------------------------------------------------- #
# Job 1: GenerateGreens
# --------------------------------------------------------------------------- #

"""
    greens_counts(pt) -> NamedTuple

Blocks built by `_generate_green_sr`, after the pipeline fix that drops the
never-read `[Receiver] <- [Sender, Receiver]` operator and adds the `(R, R)`
self block that `src/rsvd.jl` actually loads:

  1. `(R, S)` -- one external block, serialised on its own.
  2. `(R, R)` -- one self block, shared by every separation (so only built when
     the preload directory is cold).
  3. `[S, R] <- [S, R]` -- a `MultiRegionVacuumGreenOperator`, i.e. four blocks:
     `ss` and `rr` self, `sr` and `rs` external.

Peak memory comes from the multi-region build, where the blocks already
finished are still resident while the last one is being constructed.
"""
function greens_counts(pt::SRPoint)
    s, r = pt.sender_cells, pt.receiver_cells
    contact = is_contact(pt)

    # (target, source, is_self) for every block that gets built
    blocks = Tuple{NTuple{3,Int},NTuple{3,Int},Bool}[]
    push!(blocks, (r, s, false))                    # (R, S) standalone external
    pt.fresh_preload && push!(blocks, (r, r, true)) # (R, R) shared self block
    multiregion = [(s, s, true), (s, r, false), (r, s, false), (r, r, true)]
    append!(blocks, multiregion)

    # Three block kinds, not two plus a surcharge: an external block between
    # touching bodies runs a different branch of `genEgoCrcExt!` with a large
    # fixed cost and a small per-cell cost.
    self_work = 0.0        # sum of M*log2(M) over self blocks
    self_cells = 0         # sum of M over self blocks
    ext_work = 0.0
    ext_cells = 0
    contact_work = 0.0
    contact_cells = 0
    n_self = 0
    n_ext = 0
    n_contact = 0
    for (trg, src, isself) in blocks
        M = circulant_cells(trg, src)
        if isself
            self_work += fft_work(M)
            self_cells += M
            n_self += 1
        elseif contact
            contact_work += fft_work(M)
            contact_cells += M
            n_contact += 1
        else
            ext_work += fft_work(M)
            ext_cells += M
            n_ext += 1
        end
    end

    # Bytes serialised: the retained Fourier data of every operator written out.
    bytes_written = ext_fourier_bytes(r, s)                              # (R, S)
    pt.fresh_preload && (bytes_written += self_fourier_bytes(r))          # (R, R)
    bytes_written += self_fourier_bytes(s) + self_fourier_bytes(r) +
                     ext_fourier_bytes(s, r) + ext_fourier_bytes(r, s)    # multi-region

    # Peak: everything retained by the multi-region operator except the block
    # currently under construction, plus that block's transient peak. Take the
    # worst ordering.
    retained = [self_fourier_bytes(s), ext_fourier_bytes(s, r),
                ext_fourier_bytes(r, s), self_fourier_bytes(r)]
    peaks = [block_build_peak_bytes(trg, src; self=isself) for (trg, src, isself) in multiregion]
    peak_bytes = 0
    for i in eachindex(multiregion)
        resident = sum(retained[j] for j in eachindex(retained) if j != i; init=0)
        peak_bytes = max(peak_bytes, resident + peaks[i])
    end
    # Serialisation of the finished multi-region operator buffers a copy.
    peak_bytes = max(peak_bytes, sum(retained) * 2)

    return (n_self_blocks=n_self, n_ext_blocks=n_ext, n_contact_blocks=n_contact,
            self_fft_work=self_work, ext_fft_work=ext_work, contact_fft_work=contact_work,
            self_cells=self_cells, ext_cells=ext_cells, contact_cells=contact_cells,
            n_blocks=n_self + n_ext + n_contact,
            bytes_written=bytes_written, peak_bytes=peak_bytes)
end

function greens_time_s(pt::SRPoint, c::Coefficients)
    counts = greens_counts(pt)
    eta = thread_efficiency(c, pt.threads)
    t = c.g0_self_fft * counts.self_fft_work + c.g0_ext_fft * counts.ext_fft_work +
        c.g0_contact_fft * counts.contact_fft_work
    t += (c.g0_self_cell * counts.self_cells + c.g0_ext_cell * counts.ext_cells +
          c.g0_contact_cell * counts.contact_cells) / eta
    t += c.g0_self_fixed * counts.n_self_blocks + c.g0_ext_fixed * counts.n_ext_blocks +
         c.g0_contact_fixed * counts.n_contact_blocks
    t += counts.bytes_written / c.disk_write_rate
    return t + c.g0_startup_s
end

greens_host_bytes(pt::SRPoint, c::Coefficients) =
    c.greens_mem_factor * greens_counts(pt).peak_bytes + c.greens_mem_base

# --------------------------------------------------------------------------- #
# Job 2: GenerateRSVD
# --------------------------------------------------------------------------- #

"""
    rsvd_counts(pt) -> NamedTuple

Work done by `_generate_rsvd_sr`, which is two randomized factorizations.

**`_save_ur_asym`**: `reigen_hermitian(Asym(G0_ur), k; p, q)` on an `N_u x N_u`
operator with sketch width `c = k + p` (never clamped, see `reigen.jl`).
`randomized_hermitian_range_finder` applies the operator once for the initial
sketch and once per power iteration, and `eigen_hermitian_restricted` applies it
once more, so `c*(q + 2)` operator applications. Each application of
`asym(iota_r G0_rs Pi_s) + iota_r Asym(G0_rr) Pi_r` costs **two** external
matvecs (the map and its adjoint) plus **one** self matvec
(`AsymVacuumGreenOperator` is a single operator, not a difference).

**`_run_rsvdvals("RS/")`**: `rsvdvals(G0_rs, k; p, q)` with sketch width
`min(3 n, k + p)`. `randomized_range_finder` does `1 + 2q` applications and
`svdvals_restricted` one more adjoint application, so `c*(2q + 2)` external
matvecs.

Total external matvecs are therefore `c*(2q + 4 + 2q + 2)`-ish -- about 2.5x
what the previous estimator's `(2 + 2q)*(k + p)` accounted for.

Dense work: `q + 1` thin QRs of `N_u x c` in the Hermitian range finder plus
`2q + 2` of `N_r x c` in the SVD one, the `c x c` eigen/svd solves, and the
`N x c x c` gemms.

This is the in-memory path, with the sketch resident on the device. See
`rsvd_panel_counts` for the Funicular panel path and `rsvd_dense_counts` for the
dense-exact branch; `rsvd_mode` picks between the three. All three share the
matvec counts above, which is why they are stated here only.
"""
function rsvd_counts(pt::SRPoint)
    N_s = vector_length(pt.sender_cells)
    N_r = vector_length(pt.receiver_cells)
    N_u = N_s + N_r
    q = pt.power_iters
    k = pt.rank

    c_herm = sketch_width(pt)                      # reigen_hermitian does not clamp
    c_svd = sketch_width(pt; clamp_to=min(N_r, N_s))

    herm_applications = c_herm * (q + 2)
    mv_ext = 2 * herm_applications + c_svd * (2q + 2)
    mv_self = herm_applications

    M_ext = circulant_cells(pt.receiver_cells, pt.sender_cells)
    M_self = circulant_cells(pt.receiver_cells, pt.receiver_cells)

    # Dense flops.
    qr_flops = (q + 1) * flops_qr(N_u, c_herm) + (2q + 2) * flops_qr(N_r, c_svd)
    gemm_flops = 2 * flops_gemm(N_u, c_herm, c_herm) +       # B = Q' (A Q)
                 flops_gemm(N_u, c_herm, min(k, c_herm)) +   # evecs = Q * Vtilde
                 flops_gemm(N_r, c_svd, c_svd)               # B' = A' Q staging
    solve_flops = flops_eigh(c_herm) + flops_svdvals(c_svd)

    # Peak device memory: the Hermitian range finder holds Omega, Q and the
    # freshly applied operator*Q simultaneously, all N_u x c complex, plus the
    # operators (G0_rs, G0_rr and its Asym copy).
    vram_bytes = rsvd_inmemory_live_bytes(pt)

    # Host: `_save_reigen_hermitian` pulls the eigenvectors back with `Array(...)`
    # and JLD2 buffers them on the way out.
    host_dense = N_u * min(k, c_herm) * BYTES_PER_COMPLEX
    bytes_written = host_dense + c_herm * BYTES_PER_COMPLEX + c_svd * BYTES_PER_COMPLEX

    return (mv_ext=mv_ext, mv_self=mv_self,
            ext_fft_work=mv_ext * fft_work(M_ext),
            self_fft_work=mv_self * fft_work(M_self),
            qr_flops=qr_flops, gemm_flops=gemm_flops, solve_flops=solve_flops,
            sketch_width_herm=c_herm, sketch_width_svd=c_svd,
            vram_bytes=vram_bytes, host_dense_bytes=host_dense,
            bytes_written=bytes_written)
end

"""
    RSVD_PANEL_SWEEPS_PER_ITER, RSVD_PANEL_SWEEPS_START, RSVD_PANEL_SWEEPS_RESTRICT

Sweeps, that is, traversals of an `N_u x c` panel matrix, in `reigen_hermitian`'s
panel path, read off `panel_range_finder`, `panel_range_start` and
`panel_restricted` in `ext/MFRLAFunicularExt.jl`. A `panelmul!` is one sweep, and
a `cholqr2!` is four (two `cholqr_pass!`es, each a `gram` and an `rdiv_rows!`).

  * start: `panelmul!(Y, operator, Omega)` then `cholqr2!(Y)`, so 1 + 4. `Omega`
    is a `GhostPanels`, regenerated per column rather than stored or moved, so
    the test matrix contributes no traffic at all.
  * each power iteration: `panelmul!(Z, operator, Y)` then `cholqr2!(Z)`, 1 + 4.
  * the reduced block: `panel_restricted` takes the two-sweep route
    (`panelmul!(Z, operator, Q)` then `gram(Q, Z)`) whenever the host budget holds
    a second `N_u x c` matrix, which on this pipeline it does by construction: the
    host peak in `rsvd_panel_counts` *is* those two matrices.

Left out: the final `rightmul!(V, Q, rotation)`, because the positives-only save
runs with `factored=true` and forms only the `m` positive columns (workstream B5),
and the `rsvdvals` side, whose sweeps are over the half-height `N_r x c_svd`
matrices. Both are absorbed into `overlap_factor`, which is fitted against measured
wall time, so an under-counted sweep makes the fitted factor larger. That is the
right place for the error to land.
"""
const RSVD_PANEL_SWEEPS_PER_ITER = 5
const RSVD_PANEL_SWEEPS_START = 5
const RSVD_PANEL_SWEEPS_RESTRICT = 2

"""
    rsvd_panel_counts(pt) -> NamedTuple

`rsvd_counts` for the Funicular panel path (`reigen_hermitian_panel` /
`rsvdvals_panel` in `ext/MFRLAFunicularExt.jl`). Same algorithm, different storage,
so the parts that count operator applications are *identical*: Funicular applies
the operator column by column, a sketch of `c` columns still costs `c`
applications per pass, and `mv_ext`/`mv_self` here are the same expressions as on
the in-memory path.

What changes:

  * The thin QR becomes CholeskyQR2. `cholqr2!` can be computed one row block at a
    time, which is what makes it panelizable at all. More flops
    (`flops_cholqr2`), but all gemm, hence `gemm_rate`.
  * Device memory stops scaling with `c`. Panels are never resident: the plan
    allocates staging buffers once and reuses them, so the device holds
    `panel_staging_bytes` plus the operator's own workspace plus the `c x c`
    reduced block, and nothing that grows with the sketch.
  * Host memory starts scaling with `c`: two `N_u x c` matrices during the sweeps,
    `Y` and the swapped `Z` of the Hermitian power iteration. Two and not three
    during the sweeps, since the test matrix is a ghost, and not fewer, since
    `gram` and `cholqr2!` traverse *rows* and so hold every panel of their matrix
    in host memory at once. That is the floor Funicular's residency page calls
    out. The *peak* is one matrix higher than the sweep floor, and that extra
    matrix is the whole point of `host_panel_bytes` being
    `(2c + m) * N_u * 16` rather than `2c * N_u * 16`; see the save bullet.
  * The bus becomes a cost. `sweep_bytes` is the total traffic, upload plus
    writeback, over every sweep in `RSVD_PANEL_SWEEPS_*`.
  * The save shrinks but is *not* free. Only the positive-Γ columns are written
    (workstream B5), streamed panel by panel to h5, so `bytes_written` is `m`-wide
    rather than `k`-wide and no host-side `Array` of the eigenvectors is ever
    formed. What still costs is the `N_u x m` `PanelMatrix` itself:
    `_save_ur_asym` calls `materialize_columns(out.vectors, 1:m)`, which allocates
    a third tall matrix out of the plan, and it does so *on top of* the sketch
    high-water rather than in place of it, because `Funicular.free!` hands a block
    back to the plan's pinned slab pool instead of to the OS. Nothing the RSVD
    frees before the save shrinks the process's RSS, so the cgroup sees
    `2c + m` columns even though only `m` of them are live at that instant.
"""
function rsvd_panel_counts(pt::SRPoint)
    N_s = vector_length(pt.sender_cells)
    N_r = vector_length(pt.receiver_cells)
    N_u = N_s + N_r
    q = pt.power_iters

    c_herm = sketch_width(pt)
    c_svd = sketch_width(pt; clamp_to=min(N_r, N_s))

    # Identical to the in-memory path: the panel path does the same applications.
    herm_applications = c_herm * (q + 2)
    mv_ext = 2 * herm_applications + c_svd * (2q + 2)
    mv_self = herm_applications

    M_ext = circulant_cells(pt.receiver_cells, pt.sender_cells)
    M_self = circulant_cells(pt.receiver_cells, pt.receiver_cells)

    cholqr_flops = (q + 1) * flops_cholqr2(N_u, c_herm) +
                   (2q + 2) * flops_cholqr2(N_r, c_svd)
    gemm_flops = 2 * flops_gemm(N_u, c_herm, c_herm) +       # gram(Q, Z) for B
                 flops_gemm(N_u, c_herm, min(pt.rank, c_herm)) +  # rightmul! for the vectors
                 flops_gemm(N_r, c_svd, c_svd)               # the svd side's reduction
    solve_flops = flops_eigh(c_herm) + flops_svdvals(c_svd)

    sweeps = RSVD_PANEL_SWEEPS_PER_ITER * q + RSVD_PANEL_SWEEPS_START +
             RSVD_PANEL_SWEEPS_RESTRICT
    sweep_bytes = sweeps * N_u * c_herm * BYTES_PER_COMPLEX * 2  # up and writeback

    vram_bytes = panel_staging_bytes(N_u, c_herm) +
                 rsvd_operator_vram_bytes(pt) +
                 c_herm^2 * BYTES_PER_COMPLEX
    m = min(effective_num_pos(pt), c_herm)
    #=
    Three `N_u`-tall matrices' worth of pinned slabs, not two: the two `c`-wide
    sweep matrices plus the `m`-wide positive block, which is formed *after* them
    and does not replace them. See the host-memory bullet in the docstring, and
    the narval OOM it was written for.
    =#
    host_panel_bytes = (2 * c_herm + m) * N_u * BYTES_PER_COMPLEX
    bytes_written = N_u * m * BYTES_PER_COMPLEX +
                    c_herm * BYTES_PER_COMPLEX + c_svd * BYTES_PER_COMPLEX

    return (mv_ext=mv_ext, mv_self=mv_self,
            ext_fft_work=mv_ext * fft_work(M_ext),
            self_fft_work=mv_self * fft_work(M_self),
            cholqr_flops=cholqr_flops, gemm_flops=gemm_flops,
            solve_flops=solve_flops,
            sketch_width_herm=c_herm, sketch_width_svd=c_svd,
            panel_width=panel_width(N_u, c_herm), sweeps=sweeps,
            sweep_bytes=sweep_bytes,
            vram_bytes=vram_bytes, host_panel_bytes=host_panel_bytes,
            num_saved=m, bytes_written=bytes_written)
end

"""
    rsvd_dense_counts(pt) -> NamedTuple

`rsvd_counts` for the dense-exact branch (workstream B6, `N_u <=
DENSE_EXACT_MAX_N`): apply the operator to the identity, `eigen!` the result on the
device, save the positive prefix, and get `RS/D` from dense `svdvals` of the same
block.

`N_u` applications, one column of the identity each, counted as one external and
one self matvec per application. That is a simplification: a faithful count of
`asym(iota_r G0_rs Pi_s) + iota_r Asym(G0_rr) Pi_r` is *two* external passes (the
map and its adjoint) plus one self, and the dense `svdvals` adds `N_s` more
external applications on top. At `N_u <= 12,288` the whole branch takes a minute
or two either way, so the model keeps the simpler count.

Memory is exact: three `N_u x N_u` complex matrices on the device (the operator,
LAPACK / cuSOLVER's copy, the eigenvectors), one on the host.
"""
function rsvd_dense_counts(pt::SRPoint)
    N_s = vector_length(pt.sender_cells)
    N_r = vector_length(pt.receiver_cells)
    N_u = N_s + N_r

    mv_ext = N_u
    mv_self = N_u
    M_ext = circulant_cells(pt.receiver_cells, pt.sender_cells)
    M_self = circulant_cells(pt.receiver_cells, pt.receiver_cells)

    solve_flops = flops_eigh(N_u)
    vram_bytes = 3 * N_u^2 * BYTES_PER_COMPLEX + rsvd_operator_vram_bytes(pt)
    host_dense_bytes = N_u^2 * BYTES_PER_COMPLEX
    m = min(effective_num_pos(pt), N_u)
    bytes_written = N_u * m * BYTES_PER_COMPLEX

    return (mv_ext=mv_ext, mv_self=mv_self,
            ext_fft_work=mv_ext * fft_work(M_ext),
            self_fft_work=mv_self * fft_work(M_self),
            solve_flops=solve_flops,
            vram_bytes=vram_bytes, host_dense_bytes=host_dense_bytes,
            num_saved=m, bytes_written=bytes_written)
end

"""
    _rsvd_time_raw(pt, c, mode) -> (total, device_bound)

The unscaled prediction for one storage path. `rsvd_time_s` is this plus
`rsvd_pass_scale`; `rsvd_time_parts` is this evaluated twice to split the
`q`-dependent work out.

See `bounds_time_s` for why the device-bound part is reported separately.

The bus term on the panel path is *not* device-bound. A MIG slice gets a fraction
of the streaming multiprocessors, not a fraction of the PCIe link, so stretching
the transfers by the slice fraction would invent time that does not exist, and in
the wrong direction: a slower slice hides *more* of the transfer behind compute,
not less.
"""
function _rsvd_time_raw(pt::SRPoint, c::Coefficients, mode::Symbol)
    if mode == :dense_exact
        n = rsvd_dense_counts(pt)
        device = c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext
        device += c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self
        device += n.solve_flops / c.eigh_rate
        host = n.bytes_written / c.disk_write_rate + c.gpu_startup_s
        return (host + device, device)
    elseif mode == :panel
        n = rsvd_panel_counts(pt)
        device = c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext
        device += c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self
        device += n.cholqr_flops / c.gemm_rate + n.gemm_flops / c.gemm_rate
        device += n.solve_flops / c.eigh_rate
        host = n.sweep_bytes / c.pcie_rate * c.overlap_factor
        host += n.bytes_written / c.disk_write_rate + c.gpu_startup_s
        return (host + device, device)
    end
    n = rsvd_counts(pt)
    device = c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext
    device += c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self
    device += n.qr_flops / c.qr_rate + n.gemm_flops / c.gemm_rate
    device += n.solve_flops / c.eigh_rate
    host = n.bytes_written / c.disk_write_rate + c.gpu_startup_s
    return (host + device, device)
end

"Same point with the power iterations removed, for the per-pass split below."
_at_zero_power_iters(pt::SRPoint) =
    SRPoint(pt.sender_cells, pt.receiver_cells, pt.scale, pt.separation, pt.rank,
            pt.oversamples, 0, pt.threads, pt.num_pos, pt.fresh_preload)

"""
    rsvd_time_parts(pt, c; vram_capacity_bytes=nothing) -> NamedTuple

The RSVD's predicted time split into the part that scales with the power iteration
count and the part that does not: `pass` and `fixed`, each with its device-bound
share.

Every count in `rsvd_counts` / `rsvd_panel_counts` is affine in `q` -- the sketch
costs `(2q + 2)`-ish operator passes, `q + 1` orthonormalizations, `q` extra sweeps
-- so evaluating the same expression at `q` and at `q = 0` separates the two
exactly. That is also the shape the measurement has: two runs of one geometry at
two low `q` give a slope (the per-pass cost, which extrapolates to any `q`) and an
intercept (the once-only work), for a small fraction of what one production-`q` run
would cost.

The mode is resolved once, from the real point, and reused for the `q = 0`
evaluation, so the split never straddles two storage paths.
"""
function rsvd_time_parts(pt::SRPoint, c::Coefficients;
                         vram_capacity_bytes::Union{Nothing,Real}=nothing)
    mode = rsvd_mode(pt, vram_capacity_bytes, c)
    total, device = _rsvd_time_raw(pt, c, mode)
    fixed, device_fixed = _rsvd_time_raw(_at_zero_power_iters(pt), c, mode)
    return (total=total, device=device, fixed=fixed, device_fixed=device_fixed,
            pass=max(0.0, total - fixed), device_pass=max(0.0, device - device_fixed),
            passes_per_q=2.0)
end

function rsvd_time_s(pt::SRPoint, c::Coefficients;
                     vram_capacity_bytes::Union{Nothing,Real}=nothing)
    mode = rsvd_mode(pt, vram_capacity_bytes, c)
    # Exactly the pre-`rsvd_pass_scale` numbers when the scale is 1, bit for bit:
    # no round trip through the difference at all.
    c.rsvd_pass_scale == 1.0 && return _rsvd_time_raw(pt, c, mode)
    p = rsvd_time_parts(pt, c; vram_capacity_bytes=vram_capacity_bytes)
    return (p.fixed + c.rsvd_pass_scale * p.pass,
            p.device_fixed + c.rsvd_pass_scale * p.device_pass)
end

"""
    rsvd_vram_bytes(pt, c; vram_capacity_bytes=nothing)

Device memory to request. On the in-memory and dense-exact paths this is
`rsvd_vram_factor` times the analytic live count: both hold plain `CuArray`s, so
the real high-water is whatever CUDA.jl's pool grew to while holding garbage Julia
had not collected, and the fitted factor is the measured size of that.

On the panel path the factor is dropped, because the elasticity it models is gone.
Funicular allocates its staging buffers once, at plan construction, and *nothing
inside a sweep loop allocates device memory*. The analytic count is therefore the
demand and the high-water at once, which is why `rsvd_vram_floor_bytes` returns
the same number for that path.
"""
function rsvd_vram_bytes(pt::SRPoint, c::Coefficients;
                         vram_capacity_bytes::Union{Nothing,Real}=nothing)
    mode = rsvd_mode(pt, vram_capacity_bytes, c)
    mode == :panel &&
        return rsvd_panel_counts(pt).vram_bytes + c.panel_workspace_bytes + c.rsvd_vram_base
    counts = mode == :dense_exact ? rsvd_dense_counts(pt) : rsvd_counts(pt)
    return c.rsvd_vram_factor * counts.vram_bytes + c.rsvd_vram_base
end

"""
    rsvd_vram_floor_bytes(pt, c; vram_capacity_bytes=nothing)
    bounds_vram_floor_bytes(pt, c; vram_capacity_bytes=nothing)

Least device memory the job can be squeezed into, as opposed to what it will use
if given room. Compare *this* against a card's capacity to decide whether the job
can run there; see `vram_floor_factor`.

On the panel path the floor and the request coincide. Preallocated staging buffers
have no slack to squeeze out, so there is no smaller configuration to fall back
to; the only way down is a narrower panel, and `panel_width` already reports what
Funicular will choose.
"""
function rsvd_vram_floor_bytes(pt::SRPoint, c::Coefficients;
                               vram_capacity_bytes::Union{Nothing,Real}=nothing)
    mode = rsvd_mode(pt, vram_capacity_bytes, c)
    mode == :panel &&
        return rsvd_panel_counts(pt).vram_bytes + c.panel_workspace_bytes + c.rsvd_vram_base
    counts = mode == :dense_exact ? rsvd_dense_counts(pt) : rsvd_counts(pt)
    return c.vram_floor_factor * counts.vram_bytes + c.rsvd_vram_base
end

function bounds_vram_floor_bytes(pt::SRPoint, c::Coefficients;
                                 vram_capacity_bytes::Union{Nothing,Real}=nothing)
    tau, m = tau_shape(c), bounds_m(pt, c)
    bounds_mode(pt, vram_capacity_bytes, c) == :panel &&
        return bounds_panel_counts(pt; tau=tau, m=m).vram_bytes + c.panel_workspace_bytes +
               c.bounds_vram_base
    return c.vram_floor_factor * bounds_counts(pt; tau=tau, m=m).vram_bytes + c.bounds_vram_base
end

"""
    rsvd_host_bytes(pt, c; vram_capacity_bytes=nothing)

Host memory to request.

The in-memory and dense-exact paths pull the result back with `Array(...)` and let
JLD2 buffer it on the way out, so the count is that one block and the fitted
`rsvd_host_mem_factor` covers the buffering.

The panel path has no such block: the positives-only save streams panels straight
to h5 (workstream B5), never materializing a host `Array` of the eigenvectors.
What it has instead is `(2c + m) * N_u * 16`, that is, the two pinned `N_u x c`
matrices of the power iteration *plus* the `N_u x m` positive block the save
forms. The old `k`-wide `Array` term is dropped rather than added to: the two
sweep matrices dominate it by a wide margin (at 4 λ and `c = 4050` they are 102
GB against 50 GB), but the positive block is not the same thing as that `Array`
and does not go away with it.

# Why the positive block is added and not maximised over

Reading `_save_ur_asym`, the sketch looks finished by the time
`materialize_columns(out.vectors, 1:m)` runs, which would make the peak
`max(2c, m) * N_u * 16`. It is not, and a sweep of 1 λ, `k = 4000` panel jobs on
narval was OOM-killed showing it. Those jobs asked for `--mem=34G` on a whole
A100, which is exactly what the two-matrix count predicted; the plan reported a 28
GiB host budget, the sketch peaked at the predicted `2 * N_u * c * 16 = 25.5` GB,
and the process was then killed at `Forming the 196608 x 1919 positive block and
streaming it`, i.e. after the sketch had completed and while the save was
allocating. 31 of 333 jobs got through, which is a request sitting right at the
edge rather than a different failure. Two things keep the sketch's memory on the
books:

  1. `Funicular.free!` returns a block to the plan's pinned slab pool, not to the
     OS. Slabs are page-locked and are not unmapped, so the sketch's high-water
     is still charged to the cgroup when the positive block is allocated. Unless
     the block is served *entirely* out of freed slabs -- which it is not, since
     the sketch matrices are still live when `materialize_columns` starts -- the
     new pinned pages stack on the old ones.
  2. The h5 stream's dirty page cache is charged to the same cgroup. Slurm's
     memory accounting counts page cache, and pages dirtied faster than
     writeback retires them cannot be reclaimed on demand, so an `m`-wide
     streamed write shows up in the job's memory footprint as well as on disk.

`effective_num_pos` supplies `m`, so this term inherits the pessimistic
`NUM_POS_FRACTION = 0.6` and the cap at `rank` / `N_u`. The killed jobs measured a
positive fraction of 0.48, so the assumed `m` sits about 25% above what they
actually formed. This is the term whose under-count killed them, so the margin
stays.
"""
function rsvd_host_bytes(pt::SRPoint, c::Coefficients;
                         vram_capacity_bytes::Union{Nothing,Real}=nothing)
    mode = rsvd_mode(pt, vram_capacity_bytes, c)
    mode == :panel &&
        return c.panel_host_mem_factor * rsvd_panel_counts(pt).host_panel_bytes +
               c.rsvd_host_mem_base
    counts = mode == :dense_exact ? rsvd_dense_counts(pt) : rsvd_counts(pt)
    return c.rsvd_host_mem_factor * counts.host_dense_bytes + c.rsvd_host_mem_base
end

# --------------------------------------------------------------------------- #
# Job 3: ComputeBounds
# --------------------------------------------------------------------------- #

"""
    bounds_counts(pt) -> NamedTuple

Work done by `_compute_bounds_sr` after the τ-optimized pencil refactor, with
`m = num_pos` (the projection basis is the `m` positive eigenvectors, so every
dense pencil object is `m x m`) and `evals = TAU_GRID_POINTS + TAU_REFINE_EVALS`
dual evaluations per index:

  * Reverse Gram-Schmidt: `m(m-1)/2` (dot, axpy) pairs on length-`N_u` vectors
    on the device. Two kernel launches each, so launch latency matters as much
    as bandwidth.
  * `ss_basis = basis' * ss` and the `C`/`D` projections: `m` applications of
    `C` -- each applies `asym(G0_uu)`, a difference of two multi-region
    operators whose applications run all four blocks, so `4m` self plus `4m`
    external Green matvecs -- plus the `(m x N_u)(N_u x m)` gemms.
  * Pencil whitenings (device heevd, `psd_pencil_whitener`): `TAU_GRID_POINTS`
    shared `m x m` Hermitian eigendecompositions up front, plus
    `TAU_REFINE_EVALS` throwaway ones per index for the golden-section probes.
  * Pencil solves (device, `diag_pencil_eigen`), once per (index, evaluation):
    an `m x m` Hermitian eigendecomposition and two `m x m x m` gemms.
  * Probe loop, `(m - n + 1)` probes per (index, evaluation) -- about
    `evals * m^2/2` in total: an `m x m` device gemv, one device-to-host
    transfer of the projected b-vector, and a host-side Brent root find over
    length-`m` resolvent expansions.
"""
function bounds_counts(pt::SRPoint; tau::TauShape=TAU_SHAPE_LEGACY,
                      m::Union{Nothing,Integer}=nothing)
    N_u = universe_length(pt)
    k = pt.rank
    m = m === nothing ? min(effective_num_pos(pt), N_u) : min(Int(m), N_u)
    pairs = m * (m - 1) / 2
    evals_per_index = tau_evals_per_index(tau)
    probes = evals_per_index * m * (m + 1) / 2

    gs_bytes = pairs * 3 * N_u * BYTES_PER_COMPLEX   # read s_j, read/write w_i
    gs_launches = 2 * pairs

    M_ext = circulant_cells(pt.receiver_cells, pt.sender_cells)
    M_self = circulant_cells(pt.receiver_cells, pt.receiver_cells)
    mv_ext = 4 * m
    mv_self = 4 * m

    # Device-side dense work: the m-wide basis projections, then the pencil
    # stage -- whitenings (grid shared, refinement per index) and one
    # diag_pencil_eigen (eigh + two gemms) per evaluation, all through
    # CUSOLVER/CUBLAS. Only the root finds stay on the host.
    gemm_flops = flops_gemm(m, N_u, m) +            # ss_basis
                 2 * m * flops_gemm(N_u, m, 1) +    # C: G' v and G w per application
                 flops_gemm(m, N_u, m) +            # basis' * (C basis)
                 flops_gemm(m, N_u, m)              # D: Bs' Bs on the sender rows
    whitenings = tau.grid_points + m * tau.refine_whitenings
    pencil_eigh_flops = (whitenings + m * evals_per_index) * flops_eigh(m)
    pencil_gemm_flops = 2 * m * evals_per_index * flops_gemm(m, m, m)
    probe_gemv_flops = probes * flops_gemm(m, m, 1)
    root_work = probes * m

    # The pencil arena lives on the device with the whitenings: C_basis, D, S,
    # ss_basis, the working whitener + eigenvectors, the cached grid whiteners
    # (whitener + nullspace ~ m^2 each) and the refinement pencil cache's entries
    # (the same two objects per entry).
    vram_bytes = (3 * k + 2 * m) * N_u * BYTES_PER_COMPLEX +
                 (2 * tau.grid_points + 2 * tau.cache_entries + 8) * m^2 * BYTES_PER_COMPLEX +
                 2 * self_fourier_bytes(pt.receiver_cells) +
                 2 * ext_fourier_bytes(pt.receiver_cells, pt.sender_cells)
    # One host-side copy of the eigenvector block; JLD2's own buffering and the
    # `CuArray(...)` staging copy are what `bounds_host_mem_factor` absorbs.
    host_bytes = N_u * k * BYTES_PER_COMPLEX
    bytes_read = N_u * k * BYTES_PER_COMPLEX

    return (num_pos=m, gs_bytes=gs_bytes, gs_launches=gs_launches,
            mv_ext=mv_ext, mv_self=mv_self,
            ext_fft_work=mv_ext * fft_work(M_ext),
            self_fft_work=mv_self * fft_work(M_self),
            gemm_flops=gemm_flops,
            pencil_eigh_flops=pencil_eigh_flops,
            pencil_gemm_flops=pencil_gemm_flops,
            probe_gemv_flops=probe_gemv_flops, probes=probes,
            root_work=root_work, vram_bytes=vram_bytes, host_bytes=host_bytes,
            bytes_read=bytes_read)
end

"""
    BOUNDS_PANEL_SWEEPS

Sweeps in the panelized reverse Gram-Schmidt (workstream C2): one `gram` for
`G = basis' basis`, then one `rightmul!` for `ss = basis * T` once the reversed
permutation, Cholesky and triangular inverse of `G` have been done on the host at
`m x m`. Two, where the in-memory version is an `O(m^2)` loop of BLAS-1 pairs.

The blocked form is only legitimate because the basis is RSVD output and hence
near-orthonormal, so `G` is close to the identity and the squared conditioning of
the Cholesky route costs nothing. On a general basis it would not be.
"""
const BOUNDS_PANEL_SWEEPS = 2

"""
    bounds_panel_counts(pt) -> NamedTuple

`bounds_counts` for the panel path. Everything downstream of the `m x m` reduction
is untouched: the pencil stage works on `m x m` objects, about 1.7 GB on the
device at `m = 2400`, and never wanted panels. Only the `N_u`-scale front end
changes.

  * Reverse Gram-Schmidt becomes two sweeps (see `BOUNDS_PANEL_SWEEPS`). The
    `O(m^2)` BLAS-1 term disappears, and with it the launch-latency term that
    dominated it. `gs_gemm_flops` is two `N_u x m x m` gemms, `gs_sweep_bytes` is
    the traffic to feed them, and `gs_cholesky_flops` is the host-side `m x m`
    factorization between them.
  * Host memory holds three `N_u x m` matrices: the basis loaded from the h5
    panel dataset, the orthonormalized `ss`, and one working matrix for the
    projections. Row sweeps (`gram`, `rightmul!`) hold every panel of their matrix
    at once, so this is a floor, not an average.

    Three is one more than the front end's live peak, and that slack stands in for
    the effect that broke the RSVD side (see `rsvd_host_bytes`). Walking
    `_bounds_front_end_panel` in `src/bounds.jl`: the basis is live throughout,
    `ss = similar(basis)` makes two, `free!(ss)` returns its blocks to the plan's
    pinned slab pool, and `work = similar(basis)` then comes back out of that pool.
    The *live* count never exceeds two, and even a pool that reuses nothing at all
    tops out at the three this counts. There is no fourth: `gram_bb`, `S_basis`,
    `ss_basis`, `C_basis` and `D_basis` are all `m x m`.

    There is no save term either. The bounds job writes only `m`-scale results to
    its output JLD, so it never forms an `N_u`-tall matrix at save time the way
    `_save_ur_asym` does with `materialize_columns`. Its one `N_u`-scale file
    operation is a *read*: `_read_ur_asym_panel` opens the basis h5 with
    `readonly=true` as the matrix's cold tier. Read page cache is clean and the
    kernel can drop it under cgroup pressure, unlike the dirty pages an h5 stream
    writes, so it needs no reserve of its own.
  * Device memory is staging buffers plus the pencil arena. The `(3k + 2m) * N_u`
    tall term of the in-memory count is gone, leaving the cached grid whiteners,
    the working whitener and eigenvectors, and the projected `m x m` blocks, that
    is, the `(2 * TAU_GRID_POINTS + 8) * m^2` term, unchanged. The operator's own
    device workspace is charged on top of this count by `bounds_vram_bytes`, since
    it is a coefficient rather than something countable from the code.
  * The read shrinks to the `m` positive vectors, which is all that was saved.

The `m` applications of `C` and their `4m` self plus `4m` external Green matvecs
are unchanged: `panelmul!` applies the operator column by column, exactly as the
resident path does.
"""
function bounds_panel_counts(pt::SRPoint; tau::TauShape=TAU_SHAPE_LEGACY,
                            m::Union{Nothing,Integer}=nothing)
    N_u = universe_length(pt)
    m = m === nothing ? min(effective_num_pos(pt), N_u) : min(Int(m), N_u)
    evals_per_index = tau_evals_per_index(tau)
    probes = evals_per_index * m * (m + 1) / 2

    M_ext = circulant_cells(pt.receiver_cells, pt.sender_cells)
    M_self = circulant_cells(pt.receiver_cells, pt.receiver_cells)
    mv_ext = 4 * m
    mv_self = 4 * m

    # The panelized front end: one gram sweep, an m x m host Cholesky, one
    # rightmul! sweep. Bytes are counted up and back for each sweep.
    gs_gemm_flops = 2 * flops_gemm(N_u, m, m)
    gs_sweep_bytes = BOUNDS_PANEL_SWEEPS * N_u * m * BYTES_PER_COMPLEX * 2
    gs_cholesky_flops = flops_cholesky(m)

    gemm_flops = flops_gemm(m, N_u, m) +            # ss_basis
                 2 * m * flops_gemm(N_u, m, 1) +    # C: G' v and G w per application
                 flops_gemm(m, N_u, m) +            # basis' * (C basis)
                 flops_gemm(m, N_u, m)              # D: Bs' Bs on the sender rows
    whitenings = tau.grid_points + m * tau.refine_whitenings
    pencil_eigh_flops = (whitenings + m * evals_per_index) * flops_eigh(m)
    pencil_gemm_flops = 2 * m * evals_per_index * flops_gemm(m, m, m)
    probe_gemv_flops = probes * flops_gemm(m, m, 1)
    root_work = probes * m

    vram_bytes = panel_staging_bytes(N_u, m) +
                 (2 * tau.grid_points + 2 * tau.cache_entries + 8) * m^2 * BYTES_PER_COMPLEX +
                 2 * self_fourier_bytes(pt.receiver_cells) +
                 2 * ext_fourier_bytes(pt.receiver_cells, pt.sender_cells)
    host_bytes = 3 * N_u * m * BYTES_PER_COMPLEX
    bytes_read = N_u * m * BYTES_PER_COMPLEX

    return (num_pos=m,
            gs_gemm_flops=gs_gemm_flops, gs_sweep_bytes=gs_sweep_bytes,
            gs_cholesky_flops=gs_cholesky_flops,
            mv_ext=mv_ext, mv_self=mv_self,
            ext_fft_work=mv_ext * fft_work(M_ext),
            self_fft_work=mv_self * fft_work(M_self),
            gemm_flops=gemm_flops,
            pencil_eigh_flops=pencil_eigh_flops,
            pencil_gemm_flops=pencil_gemm_flops,
            probe_gemv_flops=probe_gemv_flops, probes=probes,
            root_work=root_work,
            panel_width=panel_width(N_u, m),
            vram_bytes=vram_bytes, host_bytes=host_bytes,
            bytes_read=bytes_read)
end

"""
    bounds_time_s(pt, c; vram_capacity_bytes=nothing) -> (total, device_bound)

Total predicted seconds, and how many of them are actually device throughput.

The split matters because a MIG slice gets a fraction of the streaming
multiprocessors, so only the device-bound part stretches when you ask for one.
The pencil eigendecompositions and gemms run on the device (heevd/CUBLAS) and
stretch; the per-probe Brent root finds are single-threaded host Julia with the
GPU idle, and scaling them by the slice fraction would over-request on exactly
the small-body sweeps that are otherwise cheap enough to fit in a slice.

The panel path adds two more terms to the host bucket for the same reason: the
front end's sweep traffic (PCIe, not SMs) and its `m x m` Cholesky, which
workstream C2 does on the host between the two sweeps.
"""
function bounds_time_s(pt::SRPoint, c::Coefficients;
                       vram_capacity_bytes::Union{Nothing,Real}=nothing)
    tau = tau_shape(c)
    m = bounds_m(pt, c)
    if bounds_mode(pt, vram_capacity_bytes, c) == :panel
        n = bounds_panel_counts(pt; tau=tau, m=m)
        device = n.gs_gemm_flops / c.gemm_rate
        device += c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext
        device += c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self
        device += n.gemm_flops / c.gemm_rate
        device += n.pencil_eigh_flops / c.eigh_rate
        device += (n.pencil_gemm_flops + n.probe_gemv_flops) / c.gemm_rate
        host = n.gs_sweep_bytes / c.pcie_rate * c.overlap_factor
        host += n.gs_cholesky_flops / c.eigh_rate
        host += n.probes * c.sync_latency + n.root_work * c.host_root_find
        host += n.bytes_read / c.disk_read_rate + c.gpu_startup_s
        return (host + device, device)
    end
    n = bounds_counts(pt; tau=tau, m=m)
    device = n.gs_bytes / c.bandwidth + n.gs_launches * c.launch_latency
    device += c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext
    device += c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self
    device += n.gemm_flops / c.gemm_rate
    device += n.pencil_eigh_flops / c.eigh_rate
    device += (n.pencil_gemm_flops + n.probe_gemv_flops) / c.gemm_rate
    # Per-probe D2H syncs and root finds go in the host bucket: neither gets
    # slower on a MIG slice, so neither should be stretched by its fraction.
    host = n.probes * c.sync_latency + n.root_work * c.host_root_find
    host += n.bytes_read / c.disk_read_rate + c.gpu_startup_s
    return (host + device, device)
end

"""
    bounds_vram_bytes(pt, c; vram_capacity_bytes=nothing)
    bounds_host_bytes(pt, c; vram_capacity_bytes=nothing)

As for the RSVD: the panel path's device count is preallocated and so is taken
without `bounds_vram_factor`, while its host count is Funicular's pinned slabs and
so is taken with the tight `panel_host_mem_factor` rather than the in-memory
`bounds_host_mem_factor` (which has to cover a `CuArray(...)` staging copy and
JLD2's buffering that this path does not make).

`panel_workspace_bytes` is charged here too, for the same reason it is charged to
the RSVD. The bounds front end drives the *same* `LinearMaps` composition through
`panelmul!` for the `m` applications of `C`, so the operator's device workspace,
that is, CUFFT plan work areas plus the composition's per-apply `N_u`-vector
temporaries, is just as real on this side and just as invisible to the plan's own
arithmetic unless it is declared.
"""
function bounds_vram_bytes(pt::SRPoint, c::Coefficients;
                           vram_capacity_bytes::Union{Nothing,Real}=nothing)
    tau, m = tau_shape(c), bounds_m(pt, c)
    bounds_mode(pt, vram_capacity_bytes, c) == :panel &&
        return bounds_panel_counts(pt; tau=tau, m=m).vram_bytes + c.panel_workspace_bytes +
               c.bounds_vram_base
    return c.bounds_vram_factor * bounds_counts(pt; tau=tau, m=m).vram_bytes + c.bounds_vram_base
end

function bounds_host_bytes(pt::SRPoint, c::Coefficients;
                           vram_capacity_bytes::Union{Nothing,Real}=nothing)
    tau, m = tau_shape(c), bounds_m(pt, c)
    bounds_mode(pt, vram_capacity_bytes, c) == :panel &&
        return c.panel_host_mem_factor * bounds_panel_counts(pt; tau=tau, m=m).host_bytes +
               c.bounds_host_mem_base
    return c.bounds_host_mem_factor * bounds_counts(pt; tau=tau, m=m).host_bytes +
           c.bounds_host_mem_base
end

# --------------------------------------------------------------------------- #
# Public prediction API
# --------------------------------------------------------------------------- #

"""
    predict(job, pt, coeffs; pad=true, vram_capacity_bytes=nothing) -> NamedTuple

Predicted `(time_s, host_bytes, vram_bytes)` for one job on one point.
`vram_bytes` is zero for the CPU-only Green-function job. With `pad=true` the
coefficient set's safety factors are applied, plus the flat
`RECOMPILE_OVERHEAD_S` on the time (added after `time_pad`, and left out of
`device_time_s` so MIG-slice stretching never multiplies it).

`vram_capacity_bytes` is the total memory of the card (or MIG slice) the job would
run on, which is what decides between the in-memory, panel and dense-exact paths
(see `rsvd_mode`). Leaving it `nothing`, the default, predicts the in-memory path
unconditionally, which is what every caller written before the Funicular
integration means and what every fitted coefficient was calibrated against. Pass
it once the allocation is known. Note, however, that the choice is circular: the
prediction sizes the request, the request picks the card, and the card picks the
path. A caller sizing a job should evaluate candidate cards and take the first
that fits rather than expect one pass to settle it.
"""
function predict(job::JobKind, pt::SRPoint, coeffs::Coefficients=coefficients_for("molering");
                 pad::Bool=true, vram_capacity_bytes::Union{Nothing,Real}=nothing)
    if job == GenerateGreens
        # CPU-only job: no device-bound share to stretch for a GPU slice, and no
        # card, so `vram_capacity_bytes` has nothing to select.
        mode = :host
        t, device_t = greens_time_s(pt, coeffs), 0.0
        host, vram, floor = greens_host_bytes(pt, coeffs), 0.0, 0.0
    elseif job == GenerateRSVD
        mode = rsvd_mode(pt, vram_capacity_bytes, coeffs)
        t, device_t = rsvd_time_s(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
        host = rsvd_host_bytes(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
        vram = rsvd_vram_bytes(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
        floor = rsvd_vram_floor_bytes(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
    elseif job == ComputeBounds
        mode = bounds_mode(pt, vram_capacity_bytes, coeffs)
        t, device_t = bounds_time_s(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
        host = bounds_host_bytes(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
        vram = bounds_vram_bytes(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
        floor = bounds_vram_floor_bytes(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes)
    else
        error("Unknown job kind: $job")
    end
    if pad
        t *= coeffs.time_pad
        device_t *= coeffs.time_pad
        #=
        The host pad is the only path-aware padding factor, because it is the only
        one whose *meaning* changes with the path: `host_mem_pad` measures the
        spread of a GC'd heap around an analytic floor, and the panel path has no
        such heap (see `panel_host_mem_pad`). `time_pad` and `vram_pad` stay as
        they are. Wall time is noisy on every path for the same reasons (node
        contention), and the panel path's device count already dropped its
        churn-elastic factor in `rsvd_vram_bytes`, so `vram_pad` does the same
        run-to-run job there that it does everywhere else.
        =#
        host *= mode == :panel ? coeffs.panel_host_mem_pad : coeffs.host_mem_pad
        vram *= coeffs.vram_pad
        floor *= coeffs.vram_pad
        t += RECOMPILE_OVERHEAD_S
    end
    # `mode` is reported so a caller can log which path it sized for. With one
    # predicate, the log and the job agree.
    return (time_s=t, device_time_s=device_t, host_bytes=host, vram_bytes=vram,
            vram_floor_bytes=floor, mode=mode)
end

predict_time_s(job::JobKind, pt::SRPoint, coeffs::Coefficients; pad::Bool=true,
               vram_capacity_bytes::Union{Nothing,Real}=nothing) =
    predict(job, pt, coeffs; pad=pad, vram_capacity_bytes=vram_capacity_bytes).time_s
predict_host_bytes(job::JobKind, pt::SRPoint, coeffs::Coefficients; pad::Bool=true,
                   vram_capacity_bytes::Union{Nothing,Real}=nothing) =
    predict(job, pt, coeffs; pad=pad, vram_capacity_bytes=vram_capacity_bytes).host_bytes
predict_vram_bytes(job::JobKind, pt::SRPoint, coeffs::Coefficients; pad::Bool=true,
                   vram_capacity_bytes::Union{Nothing,Real}=nothing) =
    predict(job, pt, coeffs; pad=pad, vram_capacity_bytes=vram_capacity_bytes).vram_bytes

end # module
