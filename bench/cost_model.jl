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
    Gram-Schmidt over length-`N_u` vectors, and, per universe apply, `2m` self
    plus `2m` external Green matvecs (doubled on an unrefined point, where
    `Asym(G⁰ᵤᵤ)` has no folded form; see `UU_ASYM_APPLIES`).

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

Sizes are always driven by the cell count of **one body**. In particular,
`union(sender, receiver)`, the bounding box that includes the gap, must never
appear in a cost estimate: no operator is ever built on it. (The previous
estimator used it, which is why a 10000-wavelength separation asked for
terabytes.)

The separation changes nothing about the cost except at its two ends. At contact
the external blocks take the contact quadrature. Under six coarse cells of gap a
job run with `--refine` refines the two facing surfaces (`src/refinement.jl`),
which makes each body a tiling rather than a cuboid, every Green operator a block
matrix over region pairs, and `N_u` larger (by 4.75x at the nearest gap of the
1/4 lambda sweep). That is on by default on both sides, so a point predicts the
refined meshes unless `refine_gap = false` says otherwise, matching a job run with
`--no-refine`. See the gap-refinement section.

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
export bounds_time_s, outer_indices_swept, probe_count
export BOUNDS_M_REF_SEP, rsvd_time_parts
export bounds_augment, with_augmentation, AugmentShape, AUGMENT_OFF
export augment_k_uu_cap, augment_budget_bytes, BOUNDS_K_UU_CLIP_FLOOR
export BOUNDS_K_UU_DEFAULT, BOUNDS_AUGMENT_THRESHOLD_DEFAULT
export BOUNDS_UU_OVERSAMPLES, BOUNDS_UU_POWER_ITERS
export panel_width, panel_staging_bytes
export n_cells, vector_length, universe_length, sketch_width, is_contact
export self_fourier_bytes, ext_fourier_bytes, block_build_peak_bytes
export circulant_cells, fft_work
export MIN_GAP_CELLS, GAP_REFINEMENT_TABLE, GapRefinement, gap_refinement
export refinement_of, is_refined, MeshRegion, body_regions, point_meshes
export GreenBlock, region_block, composite_blocks, green_operators
export sender_length, receiver_length, REFINED_ASYM_SELF_APPLIES
export ext_operator_bytes, self_operator_bytes, universe_operator_bytes

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
- `refine_gap`: whether the job refines the two facing surfaces at a near gap,
  mirroring `SMRSystem`'s keyword of the same name. `true`, the default, is what a
  production job does and predicts the refined meshes; `false` is `--no-refine`
  and predicts the plain cuboid ones.
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
    refine_gap::Bool
end

function SRPoint(sender_cells::NTuple{3,Int}, receiver_cells::NTuple{3,Int};
                 scale::Union{Rational{Int},NTuple{3,Rational{Int}}}=1//32,
                 separation::Rational{Int}=0//1,
                 rank::Int=256, oversamples::Int=50, power_iters::Int=14,
                 threads::Int=4, num_pos::Union{Nothing,Int}=nothing,
                 fresh_preload::Bool=true, refine_gap::Bool=true)
    scl = scale isa Rational ? (scale, scale, scale) : scale
    return SRPoint(sender_cells, receiver_cells, scl, separation, rank,
                   oversamples, power_iters, threads, num_pos, fresh_preload,
                   refine_gap)
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
    PENCIL_CACHE_MAX

Entries the refinement pencil cache holds, from `pencil_cache_max = 16` in
`bounds_from_spectrum` (`src/bounds.jl`). Each entry is one whitener plus its null
space, `m^2` complex between them, so this is a device-memory term and nothing
else. It is here for the same reason `TAU_GRID_POINTS` is: a measurement that did
not record the cache size still ran with it, and the code's declared default is a
better stand-in than zero. Update it if `bounds_from_spectrum`'s default changes.
"""
const PENCIL_CACHE_MAX = 16

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
    BOUNDS_K_UU_DEFAULT, BOUNDS_AUGMENT_THRESHOLD_DEFAULT
    BOUNDS_UU_OVERSAMPLES, BOUNDS_UU_POWER_ITERS

The `Asym(G⁰ᵤᵤ)` augmentation's production defaults, mirrored from `DEFAULT_K_UU`,
`DEFAULT_AUGMENT_THRESHOLD`, `UU_OVERSAMPLES` and `UU_POWER_ITERS` in
`src/bounds.jl`. Duplicated rather than imported for the same reason
`bench/size_bounds_jobs.jl` duplicates `_gamma_kept_count`: the cost model has to
start in a second on a login node without loading CUDA. If the defaults there
change, change them here.

Note that these are the values a job *runs* at; whether the model *charges* for
them is `bounds_augment_mode`, which is `"off"` unless a caller turns it on. That
split is what keeps every prediction made before the augmentation existed
byte-identical.
"""
const BOUNDS_K_UU_DEFAULT = 512
const BOUNDS_AUGMENT_THRESHOLD_DEFAULT = 1000
const BOUNDS_UU_OVERSAMPLES = 50
const BOUNDS_UU_POWER_ITERS = 4

"""
    BOUNDS_K_UU_CLIP_FLOOR, BOUNDS_UU_MIN_OVERSAMPLES
    GILA_N_TEMPORARIES, CUFFT_WORKSPACE_BYTES

The rest of the augmentation's runtime constants, mirrored from `K_UU_CLIP_FLOOR`
and `UU_MIN_OVERSAMPLES` in `src/bounds.jl` and `GILA_N_TEMPORARIES` /
`CUFFT_WORKSPACE_BYTES` in `src/rsvd.jl`, and duplicated for the same reason the
four above are. They exist here only to feed `augment_k_uu_cap`, which has to
reach the same clipped `k_uu` the job will.
"""
const BOUNDS_K_UU_CLIP_FLOOR = 64
const BOUNDS_UU_MIN_OVERSAMPLES = 10
const GILA_N_TEMPORARIES = 6
const CUFFT_WORKSPACE_BYTES = 512 * 2^20

"""
    gila_workspace_bytes(N_u) -> Int

`gila_workspace_bytes` from `src/rsvd.jl`: the Green composition's per-apply
`N_u`-vector temporaries plus Gila's CUFFT plan work areas. Held back from the
device budget everywhere the runtime plans against it, and therefore held back
here too.
"""
gila_workspace_bytes(N_u::Integer) =
    GILA_N_TEMPORARIES * Int(N_u) * BYTES_PER_COMPLEX + CUFFT_WORKSPACE_BYTES

"""
    augment_budget_bytes(N_u, vram_capacity_bytes) -> Float64

The device budget `clip_k_uu` clips against, from a card capacity:
`PANEL_PATH_DEVICE_FRACTION` of the card, less `gila_workspace_bytes`. Exactly
`device_budget_bytes() - gila_workspace_bytes(N_u)` in `src/bounds.jl`, with the
named capacity standing in for `CUDA.total_memory()`.

The one thing that does *not* line up, and cannot: the ladders in
`create_jobs.jl` and `bench/size_bounds_jobs.jl` name a card by its marketing
capacity (`40` for an A100-40), while `CUDA.total_memory()` reports the 42.5 GB it
really has. So this budget is about 6% under the runtime's and the clip here is
correspondingly a little more aggressive. That is the same nominal-versus-reported
gap `uses_panel_path` already lives with, and the direction is the tolerable one:
the sizer can predict a smaller `m_aug` than the job ends up using, which the
`--time-margin` covers, whereas a sizer that predicted a *larger* `k_uu` than the
card can hold would be sizing a job that cannot run. The invariant that is exact,
and that `test/augmented_basis.jl` checks, is that `augment_k_uu_cap` and
`max_k_uu_for_budget` return the same `k` for the same `(N_u, m, budget)`.
"""
augment_budget_bytes(N_u::Integer, vram_capacity_bytes::Real) =
    PANEL_PATH_DEVICE_FRACTION * vram_capacity_bytes - gila_workspace_bytes(N_u)

"""
    augment_k_uu_cap(N_u, m, budget_bytes) -> Int

`max_k_uu_for_budget` from `src/bounds.jl`, line for line: the largest `k_uu`
whose augmented front end, augmentation QR and fudged `Asym(G⁰ᵤᵤ)` sketch all fit
`budget_bytes`. See that function for what the three terms are and why the sketch
is measured at the minimum oversamples. If it changes, change this.
"""
function augment_k_uu_cap(N_u::Integer, m::Integer, budget_bytes::Real)
    column_bytes = Int(N_u) * BYTES_PER_COMPLEX
    columns = floor(Int, budget_bytes / column_bytes)
    k_front = fld(columns - 3 * Int(m), 2)
    k_qr = fld(columns - Int(m), 3)
    k_sketch = floor(Int, columns / (3 * PANEL_PATH_FLOOR_FACTOR)) -
               BOUNDS_UU_MIN_OVERSAMPLES
    return min(k_front, k_qr, k_sketch)
end

"""
    AugmentShape

What the `Asym(G⁰ᵤᵤ)` augmentation costs on one point: whether it happens at all,
how many directions it adds, and the resulting pencil dimension.

# Fields
- `augmented::Bool`: Whether this point augments at all. `false` is the whole
  struct's off switch, and every count that reads it is then zero
- `k_uu::Int`: Directions of `Asym(G⁰ᵤᵤ)` the basis is actually augmented by,
  after every clamp. Zero when `augmented` is false
- `m_aug::Int`: Width of the pencil stage, `m + k_uu` augmented and `m` not
- `oversamples::Int`, `power_iters::Int`: The `reigen_hermitian` sketch's width
  above `k_uu` and its power iteration count, mirrored from `BOUNDS_UU_*`
- `k_uu_requested::Int`: The `k_uu` the caller asked for, before any clamp
- `clip::Symbol`: Which clamp bound, one of the four below

`m_aug` is the width every `m × m` object in the bounds job becomes, and it is the
one number that matters most: the pencil stage is `O(m · evals · m_aug³)`, so a
far-field point that keeps `m = 15` and augments to `m_aug = 527` does not get 35×
more expensive (the outer loop is still 15 indices) but each of those indices now
solves a 527-dimensional generalized eigenproblem instead of a 15-dimensional one.

`AUGMENT_OFF` is what every point gets under `bounds_augment_mode = "off"`, and
its `m_aug == m` makes every count in `bounds_counts` collapse, term for term, to
the expression it had before this existed.

`k_uu_requested` and `clip` are what the card did to the request, so that a caller
sizing a job can tell the four outcomes apart:

  * `:none`: `k_uu == k_uu_requested`, the job runs the augmentation it was asked
    for;
  * `:universe`: clamped to the `N_u − m` directions that exist, which only binds
    on the test fixtures;
  * `:budget`: `clip_k_uu` in `src/bounds.jl` will cut `k_uu` on this card. The
    counts here are for the *clipped* job, which is the one that will run;
  * `:infeasible`: not even `BOUNDS_K_UU_CLIP_FLOOR` fits and the job will refuse
    to start. The shape is still filled in at the floor, so a request built from it
    is an upper bound on anything that could have run, but a caller choosing a card
    should treat this as "does not fit" and try a larger one, which is what
    `bench/size_bounds_jobs.jl`'s `select_gpu` does.
"""
struct AugmentShape
    augmented::Bool
    k_uu::Int
    m_aug::Int
    oversamples::Int
    power_iters::Int
    k_uu_requested::Int
    clip::Symbol
end

# Five-argument form, for the callers that predate the clip: nothing was asked for
# beyond what was granted, and nothing was taken away.
AugmentShape(augmented::Bool, k_uu::Integer, m_aug::Integer, oversamples::Integer,
             power_iters::Integer) =
    AugmentShape(augmented, Int(k_uu), Int(m_aug), Int(oversamples),
                 Int(power_iters), Int(k_uu), :none)

AUGMENT_OFF(m::Integer) = AugmentShape(false, 0, Int(m), 0, 0)

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
    sender_length(pt), receiver_length(pt)

Length of a vector on the *universe*, which in this pipeline is the
concatenation `[sender; receiver]` -- not the bounding box. See
`asym_ur` in `src/rsvd.jl` and `projected_operators` in `src/bounds.jl`.

On a refined point a body is a tiling rather than a cuboid, so its length is
`dof_length` of that tiling (`src/refinement.jl`) and not `3 * prod(cells)`. See
the gap-refinement section below.
"""
universe_length(pt::SRPoint) = sender_length(pt) + receiver_length(pt)

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

`pairs` is the partition-pair count of a cross-scale block (`totParSrc *
totParTrg` in `glaVacOprMem.jl`), which every array above carries as a trailing
dimension. It is one on every same-scale block, which is every block there was
before gap refinement existed.
"""
function block_build_peak_bytes(target_cells::NTuple{3,Int}, source_cells::NTuple{3,Int};
                               self::Bool, pairs::Integer=1)
    M = circulant_cells(target_cells, source_cells)
    tot = target_cells .+ source_cells
    ego_crc = 9 * M * pairs * BYTES_PER_COMPLEX
    ego_fur_prp = 6 * M * pairs * BYTES_PER_COMPLEX
    ego_fur_int = 6 * prod(ntuple(i -> max(tot[i] ÷ 2, 2), 3)) * pairs * BYTES_PER_COMPLEX
    retained = self ? self_fourier_bytes(target_cells) :
                      pairs * ext_fourier_bytes(target_cells, source_cells)
    return ego_crc + ego_fur_prp + ego_fur_int + retained
end

# --------------------------------------------------------------------------- #
# Gap refinement
# --------------------------------------------------------------------------- #
#=
`src/refinement.jl` is the source of truth for everything in this section. It is
mirrored here rather than imported for the reason the k_uu clip and
`_gamma_kept_count` are mirrored: the cost model has to start in a second on a
login node without loading Gila or CUDA. If the table, the slab layout or the
face convention there changes, change them here.

What the mirror has to reproduce, region for region, is *shape*: the cost of a
composite Green operator is a sum over pairs of regions, and which of Gila's
three code paths a pair takes is decided by whether the two regions share a cell
size and whether they touch.
=#

"""
    MIN_GAP_CELLS, GAP_REFINEMENT_TABLE

`src/refinement.jl`'s constants. `GAP_REFINEMENT_TABLE[g]` is the `(factor,
thickness)` a gap of `g` coarse x-cells needs: a slab of `thickness` coarse cells
at the gap face refined by `factor` along x, then a two-cell coarse contact
layer, then the coarse bulk.
"""
const MIN_GAP_CELLS = 6
const GAP_REFINEMENT_TABLE = ((6, 6), (3, 4), (2, 4), (2, 2), (2, 2))

"""
    GapRefinement

`src/refinement.jl`'s struct: the gap in coarse x-cells, and the x-only
refinement factor and slab thickness it calls for.
"""
struct GapRefinement
    gap::Int
    factor::Int
    thickness::Int
end

"""
    gap_refinement(gap_wl, x_scale) -> Union{Nothing, GapRefinement}

`gap_refinement` from `src/refinement.jl`, line for line. `nothing` at contact
and at six or more coarse cells of gap, where the operator is accurate as it
stands and nothing is refined.
"""
function gap_refinement(gap_wl::Rational, x_scale::Rational)
    gap_wl > zero(gap_wl) || return nothing
    cells = gap_wl // x_scale
    cells >= MIN_GAP_CELLS && return nothing
    g = max(1, floor(Int, cells))
    factor, thickness = GAP_REFINEMENT_TABLE[g]
    return GapRefinement(g, factor, thickness)
end

"""
    refinement_of(pt) -> Union{Nothing, GapRefinement}
    is_refined(pt) -> Bool

The refinement this point's job will run, or `nothing` for one that will not.

`refine_gap = false`, which is what `--no-refine` asks for, is unrefined by
construction. So is a body with an odd cell count in any dimension: `refine_body`
throws on one rather than rounding it, so such a point cannot be refined at all
and the unrefined prediction is the one that describes it.
"""
function refinement_of(pt::SRPoint)
    pt.refine_gap || return nothing
    (all(iseven, pt.sender_cells) && all(iseven, pt.receiver_cells)) || return nothing
    return gap_refinement(pt.separation, pt.scale[1])
end

is_refined(pt::SRPoint) = refinement_of(pt) !== nothing

"""
    MeshRegion

One region of a body's tiling, with the two things a block cost needs: its own
cell count, and where it sits on the pair's shared x axis.

- `cells`: cells of the region, at the region's own scale
- `factor`: x-refinement, so the region's x cell is `coarse_x / factor`. One on a
  coarse region, and one everywhere on an unrefined point
- `x_lo`, `x_hi`: the region's x faces, measured in *coarse* cells from the
  sender's low-x face, so that two regions touch exactly when their closed
  intervals meet, which is the test `_cntChk` and `genEgoCrcExt!` make
"""
struct MeshRegion
    cells::NTuple{3,Int}
    factor::Int
    x_lo::Rational{Int}
    x_hi::Rational{Int}
end

"""
    body_regions(cells, ref, face, x0) -> Vector{MeshRegion}

`refine_body` from `src/refinement.jl` as a list of shapes: slab, contact layer
and coarse bulk in flat layout order, with the slab on the `:high` or `:low` x
face and the body's low-x face at `x0` coarse cells.

`ref === nothing` is the plain cuboid, one region, on which every operator of an
unrefined point is built.
"""
function body_regions(cells::NTuple{3,Int}, ref::Union{Nothing,GapRefinement},
                      face::Symbol, x0::Rational{Int})
    nx = cells[1]
    ref === nothing && return [MeshRegion(cells, 1, x0, x0 + nx)]
    f = ref.factor
    t = min(ref.thickness, nx)
    # Even `nx` and even `thickness` leave no room between the two cases: the slab
    # either takes the whole body or leaves at least the two-cell layer.
    spans = t == nx ? [(nx, f)] :
            t + 2 == nx ? [(t, f), (2, 1)] :
                          [(t, f), (2, 1), (nx - t - 2, 1)]
    low_to_high = face === :high ? reverse(spans) : spans
    regs = MeshRegion[]
    off = zero(x0)
    for (width, fac) in low_to_high
        push!(regs, MeshRegion((width * fac, cells[2], cells[3]), fac,
                               x0 + off, x0 + off + width))
        off += width
    end
    return face === :high ? reverse(regs) : regs
end

"""
    point_meshes(pt) -> NamedTuple

The sender's and receiver's tilings on one shared x axis, in coarse cells with
the sender's low-x face at zero. The sender turns its high-x face to the gap and
the receiver its low-x one, so the two are mirror images, which is why they are
different operators and key differently (`mesh_tag`).
"""
function point_meshes(pt::SRPoint)
    ref = refinement_of(pt)
    gap = pt.separation // pt.scale[1]
    return (sender=body_regions(pt.sender_cells, ref, :high, zero(gap)),
            receiver=body_regions(pt.receiver_cells, ref, :low,
                                  pt.sender_cells[1] + gap))
end

sender_length(pt::SRPoint) = is_refined(pt) ?
    sum(3 * prod(reg.cells) for reg in point_meshes(pt).sender) :
    vector_length(pt.sender_cells)
receiver_length(pt::SRPoint) = is_refined(pt) ?
    sum(3 * prod(reg.cells) for reg in point_meshes(pt).receiver) :
    vector_length(pt.receiver_cells)

"""
    GreenBlock

One region-pair block of a composite Green operator, reduced to the shapes its
cost is a function of. `_cmpBlk` in Gila's `src/glaCmpOpr.jl` picks between four
constructions; this reduces all four to the *three* code paths the fitted
coefficients already describe, plus a flag saying which of the two new ones it
came off:

  * a pair of identical regions on the diagonal of a self operator is `:self`;
  * a same-scale pair is `:ext`, or `:contact` if the two touch, exactly the
    split `greens_counts` always made, since `genEgoCrcExt!` branches on contact
    by itself;
  * a **cross-scale touching** pair is Gila's sandwich (`GlaSndOprVac`). It
    remeshes *both* regions at the finer scale and builds one same-scale operator
    between them, and the remeshed pair still touches, so it is a `:contact`
    block on the remeshed shapes. This is what the two-cell contact layer is for:
    without it the sandwich would remesh the whole coarse body;
  * a **cross-scale separated** pair is the partitioned quadrature. Each side is
    cut into `div = lcm(scales) / own scale` sub-lattices, every pair of
    sub-lattices gets its own circulant on the coarser grid, and the block is an
    `:ext` block on the partition shapes with `pairs > 1`.

# Fields
- `kind`: `:self`, `:ext` or `:contact`
- `target`, `source`: cells of *one partition* of each side, already remeshed
- `pairs`: `totParTrg * totParSrc` (`glaVacOprMem.jl`). One on a same-scale block
- `fft_passes`: FFT traversals one *matvec* makes, relative to a same-scale
  block's one. The action transforms each source partition forward and each
  target partition back (`glaVacAct.jl`), so it is `(trgDiv + srcDiv) / 2` where
  the *build* transforms all `pairs` blocks at once
- `crossscale`: whether the block came off one of the two new paths, so the
  provisional scalings can be charged to it and to nothing else
"""
struct GreenBlock
    kind::Symbol
    target::NTuple{3,Int}
    source::NTuple{3,Int}
    pairs::Int
    fft_passes::Float64
    crossscale::Bool
end

# Closed boxes meeting in x. The two bodies share a y/z cross section by
# construction (`SMRSystem` centres them on one axis), so x settles it, which is
# `_cntChk` in `glaCmpOpr.jl` and the `cntChk` in `genEgoCrcExt!`.
_regions_touch(a::MeshRegion, b::MeshRegion) = max(a.x_lo, b.x_lo) <= min(a.x_hi, b.x_hi)

"""
    region_block(trg, src, selfblk) -> GreenBlock

`_cmpBlk` in `glaCmpOpr.jl`, reduced to shapes. See `GreenBlock`.
"""
function region_block(trg::MeshRegion, src::MeshRegion, selfblk::Bool)
    selfblk && return GreenBlock(:self, trg.cells, trg.cells, 1, 1.0, false)
    if trg.factor == src.factor
        kind = _regions_touch(trg, src) ? :contact : :ext
        return GreenBlock(kind, trg.cells, src.cells, 1, 1.0, false)
    end
    if _regions_touch(trg, src)
        fine = max(trg.factor, src.factor)
        remesh(r::MeshRegion) = (r.cells[1] * (fine ÷ r.factor), r.cells[2], r.cells[3])
        return GreenBlock(:contact, remesh(trg), remesh(src), 1, 1.0, true)
    end
    # Refinement is x-only, so only the x axis is ever partitioned.
    g = gcd(trg.factor, src.factor)
    trg_div, src_div = trg.factor ÷ g, src.factor ÷ g
    trg_par = (trg.cells[1] ÷ trg_div, trg.cells[2], trg.cells[3])
    src_par = (src.cells[1] ÷ src_div, src.cells[2], src.cells[3])
    return GreenBlock(:ext, trg_par, src_par, trg_div * src_div,
                      (trg_div + src_div) / 2, true)
end

"""
    composite_blocks(trg, src; selfop) -> Vector{GreenBlock}

Every region-pair block of one `GlaCmpOprVac`. `selfop` is Gila's `slfCmp`: the
two tilings are the same body, so the diagonal takes the self path.

On an unrefined point each tiling is one region, so this is one block: self if
`selfop`, contact if the bodies touch, external otherwise. That is exactly the
inventory `greens_counts` counted before gap refinement existed.
"""
composite_blocks(trg::Vector{MeshRegion}, src::Vector{MeshRegion}; selfop::Bool) =
    vec([region_block(trg[i], src[j], selfop && i == j)
         for i in eachindex(trg), j in eachindex(src)])

"""
    green_operators(pt) -> NamedTuple

The four body-pair operators of a point, each as its list of region-pair blocks.
`_generate_green_sr` builds `rs` on its own, `rr` on its own (the shared `self/`
block), and all four again as the universe operator: a `MultiRegionVacuum-
GreenOperator` unrefined, a `CmpBlkOprVac` refined, four body-pair blocks either
way.
"""
function green_operators(pt::SRPoint)
    meshes = point_meshes(pt)
    s, r = meshes.sender, meshes.receiver
    return (ss=composite_blocks(s, s; selfop=true),
            sr=composite_blocks(s, r; selfop=false),
            rs=composite_blocks(r, s; selfop=false),
            rr=composite_blocks(r, r; selfop=true))
end

"Retained Fourier data of one block (`egoFur`), the part that goes to disk."
block_retained_bytes(b::GreenBlock) =
    b.kind === :self ? self_fourier_bytes(b.target) :
                       b.pairs * ext_fourier_bytes(b.target, b.source)

"`M log2 M` one matvec of the block transforms. See `GreenBlock.fft_passes`."
block_mv_fft(b::GreenBlock) = b.fft_passes * fft_work(circulant_cells(b.target, b.source))

"Transient peak while this one block is built."
block_peak_bytes(b::GreenBlock) =
    block_build_peak_bytes(b.target, b.source; self=(b.kind === :self), pairs=b.pairs)

"""
    REFINED_ASYM_SELF_APPLIES

Green applies `asym_self(G⁰ᵣᵣ)` costs on a refined point. Folding the
antisymmetrization into the Fourier coefficients costs one apply where
`(X - X')/2im` costs two, and upstream now has `asym(::GlaCmpOprVac)` (rev
d4c0516), so `src/rsvd.jl`'s `hasmethod` shim takes the folded path on a composite
operator as it always did on a plain one.

One on an unrefined point too, for the same reason.
"""
const REFINED_ASYM_SELF_APPLIES = 1

asym_self_applies(pt::SRPoint) = is_refined(pt) ? REFINED_ASYM_SELF_APPLIES : 1

"""
    UU_ASYM_APPLIES

Universe applies one `Asym(G⁰ᵤᵤ)` costs, which is where the bounds job spends its
Green time. A refined point assembles the universe as `CmpBlkOprVac`, whose `asym`
folds (`src/refinement.jl`), so one. An unrefined one assembles it as Gila's
multi-region operator, which has no `asym`, so `asym_self` falls back to
`(X - X')/2im` and it is two. Either way one universe apply is 2 self plus 2
external block matvecs.
"""
uu_asym_applies(pt::SRPoint) = is_refined(pt) ? 1 : 2

"""
    rs_apply_work(pt), rr_apply_work(pt), universe_apply_work(pt)

What one application of a Green operator costs, as the regressors the fitted
matvec coefficients multiply: `fft`, the `M log2 M` summed over the operator's
region-pair blocks, and `blocks`, how many per-block matvecs the composite
launches.

`universe_apply_work` reports the pair the bounds job's `4m` self / `4m` external
split is written against: one "self matvec" is one of the universe operator's two
self body-blocks and one "external matvec" one of its two external ones, so the
value is the mean over each pair.

Every one of these is `(fft_work(M), 1.0)` on an unrefined point, so
`mv_ext * ext_fft_work` stays the expression it always was.
"""
function rs_apply_work(pt::SRPoint)
    is_refined(pt) ||
        return (fft=fft_work(circulant_cells(pt.receiver_cells, pt.sender_cells)),
                blocks=1.0)
    bs = composite_blocks(point_meshes(pt).receiver, point_meshes(pt).sender; selfop=false)
    return (fft=sum(block_mv_fft, bs), blocks=Float64(length(bs)))
end

function rr_apply_work(pt::SRPoint)
    is_refined(pt) ||
        return (fft=fft_work(circulant_cells(pt.receiver_cells, pt.receiver_cells)),
                blocks=1.0)
    r = point_meshes(pt).receiver
    bs = composite_blocks(r, r; selfop=true)
    return (fft=sum(block_mv_fft, bs), blocks=Float64(length(bs)))
end

function universe_apply_work(pt::SRPoint)
    if !is_refined(pt)
        return (self_fft=fft_work(circulant_cells(pt.receiver_cells, pt.receiver_cells)),
                ext_fft=fft_work(circulant_cells(pt.receiver_cells, pt.sender_cells)),
                self_blocks=1.0, ext_blocks=1.0)
    end
    ops = green_operators(pt)
    work(bs) = sum(block_mv_fft, bs)
    return (self_fft=(work(ops.ss) + work(ops.rr)) / 2,
            ext_fft=(work(ops.sr) + work(ops.rs)) / 2,
            self_blocks=(length(ops.ss) + length(ops.rr)) / 2,
            ext_blocks=(length(ops.sr) + length(ops.rs)) / 2)
end

"""
    ext_operator_bytes(pt), self_operator_bytes(pt), universe_operator_bytes(pt)

Device bytes the `G⁰ᵣₛ`, `G⁰ᵣᵣ` and universe operators hold: their retained
Fourier data, summed over region-pair blocks. The unrefined values are
`ext_fourier_bytes(r, s)`, `self_fourier_bytes(r)` and `2` of each, the same
thing the RSVD and bounds counts always charged.
"""
ext_operator_bytes(pt::SRPoint) = is_refined(pt) ?
    sum(block_retained_bytes,
        composite_blocks(point_meshes(pt).receiver, point_meshes(pt).sender; selfop=false)) :
    ext_fourier_bytes(pt.receiver_cells, pt.sender_cells)

function self_operator_bytes(pt::SRPoint)
    is_refined(pt) || return self_fourier_bytes(pt.receiver_cells)
    r = point_meshes(pt).receiver
    return sum(block_retained_bytes, composite_blocks(r, r; selfop=true))
end

function universe_operator_bytes(pt::SRPoint)
    if !is_refined(pt)
        return 2 * self_fourier_bytes(pt.receiver_cells) +
               2 * ext_fourier_bytes(pt.receiver_cells, pt.sender_cells)
    end
    ops = green_operators(pt)
    return sum(block_retained_bytes, vcat(ops.ss, ops.sr, ops.rs, ops.rr))
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

    `bounds_m_floor`: a lower bound on the truncated `m`, in the same units. The
    decay is not monotone all the way out: the kept count bottoms out around 30
    wavelengths and then climbs slowly again, because once the coupling is pure
    far field the surviving directions stop being killed by the `gamma_rtol` cut
    and the spectrum flattens against its own top eigenvalue. Measured on the
    1 lambda sweep the minimum is 9 directions near 28 lambda and the count is
    back to ~79 by 10000 lambda, so a pure power law extrapolates to `m = 1`
    where the truth is two orders of magnitude larger. The floor is the largest
    kept count observed beyond the decay minimum; it costs almost nothing,
    because a bounds job at `m = 79` and one at `m = 9` are both dominated by
    load and startup. Default 1.0, which is the old unfloored behaviour.
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
    bounds_m_floor::Float64 = 1.0

    #=
    The `Asym(G⁰ᵤᵤ)` augmentation (`--k-uu` / `--augment-threshold` in
    `src/bounds.jl`), as a mode with an off default, exactly like `bounds_m_mode`
    above and for the same reason: every prediction made before it existed (every
    row in `bench/parity_cost_model.jl`, every `coeffs_<cluster>.jl` written
    earlier) has to keep coming out byte-identical.

    `bounds_augment_mode`:
      "off": `bounds_augment` returns `AUGMENT_OFF(m)`. Nothing is charged, and
             every count collapses to its pre-augmentation expression.
      "on":  points with `bounds_m(pt, c) < bounds_augment_threshold` are charged
             for the `reigen_hermitian` solve on `Asym(G⁰ᵤᵤ)` at width
             `bounds_k_uu + bounds_uu_oversamples`, for the augmentation's own
             QR, for the sketch's device residency, and for a pencil stage at
             `m_aug = m + bounds_k_uu` instead of at `m`. When the caller names a
             card, `bounds_k_uu` is first clipped to what that card's augmented
             front end fits, exactly as `clip_k_uu` clips it at runtime.

    These are not fitted. They are the job's own parameters, and the caller that
    turns the mode on is the caller that knows which flags the job will carry:
    `bench/size_bounds_jobs.jl` (which knows `m` from the spectrum on scratch) and
    `create_jobs.jl` (which writes the flags into the command line). `with_augmentation`
    is the one-liner that flips them together.
    =#
    bounds_augment_mode::String = "off"
    bounds_k_uu::Float64 = 0.0
    bounds_augment_threshold::Float64 = 0.0
    bounds_uu_oversamples::Float64 = BOUNDS_UU_OVERSAMPLES
    bounds_uu_power_iters::Float64 = BOUNDS_UU_POWER_ITERS

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
    PROVISIONAL, gap refinement. Three multipliers, all 1.0, all charged only on
    a refined point, so every prediction made before gap refinement existed is
    untouched and every refined prediction is the structural law with nothing
    fudged on top. They are 1.0 because the *code paths* are ones the fitted
    coefficients already describe (see `GreenBlock`), not because anyone has
    measured them at these shapes. Calibrate with `bench/plan_refined.jl`, which
    emits the tier, and fold the result in through `bench/fit.jl`.

    `g0_sandwich_scale`: on a cross-scale touching block, which Gila builds by
    remeshing both regions at the fine scale and running the *contact* quadrature
    between them (`_sndBlk`). Same code path as `g0_contact_*`, but that triple was
    fitted on whole-body contact where the fixed Gauss-Legendre setup is amortised
    over a large volume; a refined point pays it once per region pair on slabs of a
    few cells, so the fixed term is the one most likely to be wrong here.

    `g0_partition_scale`: on a cross-scale separated block, the partitioned
    quadrature. `egoFunExt!` is the same kernel `g0_ext_cell` was fitted on, and
    the count of cells it visits (`pairs * M`) is exact, so the risk is in the
    per-block fixed cost, which is now paid `pairs` times over on smaller arrays.

    `mv_composite_scale`: on a refined point's Green matvecs. The per-block launch
    overhead is already counted (`mv_*_launches`), so this covers what is left of
    `_cmpMul`: the output allocation, the per-region reshapes and views, and the
    `.+=` accumulation over the block row, none of which the same-scale matvec
    coefficients ever saw.
    =#
    g0_sandwich_scale::Float64 = 1.0
    g0_partition_scale::Float64 = 1.0
    mv_composite_scale::Float64 = 1.0

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

`bounds_m_floor` holds the cap up at the far end, where the power law is the wrong
shape: the measured kept count stops falling around thirty wavelengths and climbs
back, so an unfloored law reaches `m = 1` against a measured 68. See the
`Coefficients` docstring.
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
    cap = max(cap, c.bounds_m_floor)
    return clamp(round(Int, cap), 1, m)
end

"""
    bounds_augment(pt, c[, m]; vram_capacity_bytes=nothing) -> AugmentShape

Whether this point's bounds job augments its projection basis with the leading
eigenvectors of `Asym(G⁰ᵤᵤ)`, and how wide that makes the pencil stage.

Mirrors the decision in `bounds_from_spectrum`: augment when `k_uu > 0` and the
kept `m` is strictly below `augment_threshold`, with `k_uu` clamped to the
`N_u − m` directions the universe actually has left (which only ever binds on the
test fixtures) and the oversamples clamped so the sketch is no wider than the
operator.

`vram_capacity_bytes` names the card, and with it the third clamp: `clip_k_uu` in
`src/bounds.jl` reduces `k_uu` on a point whose dense augmented front end does not
fit the device budget, and this reduces it identically (see `augment_k_uu_cap` and
`augment_budget_bytes`). Without it (`nothing`, the default) there is no card
and no clip, which is the same convention `uses_panel_path` follows and is what
keeps a prediction made without a capacity unchanged.

Sizing it is not optional at the production sizes. `augment_threshold = 1000` and
`k_uu = 512` put the ceiling at `m_aug = 1512`, and three `N_u × m_aug` matrices at
the 4 λ universe are past an A100-40; a model that charged the unclipped `m_aug`
would size a job for work the runtime will refuse to do.

`m` may be passed in to save recomputing `bounds_m`; it defaults to it.

Returns `AUGMENT_OFF(m)` under `bounds_augment_mode = "off"`, which is the default
and is what makes every downstream count reduce to its pre-augmentation form.
"""
function bounds_augment(pt::SRPoint, c::Coefficients,
                        m::Integer=bounds_m(pt, c);
                        vram_capacity_bytes::Union{Nothing,Real}=nothing)
    m = Int(m)
    c.bounds_augment_mode == "on" || return AUGMENT_OFF(m)
    k_req = round(Int, c.bounds_k_uu)
    k_req > 0 || return AUGMENT_OFF(m)
    m < c.bounds_augment_threshold || return AUGMENT_OFF(m)
    N_u = universe_length(pt)
    k = min(k_req, N_u - m)
    k > 0 || return AUGMENT_OFF(m)
    clip = k < k_req ? :universe : :none
    if vram_capacity_bytes !== nothing
        k_fit = augment_k_uu_cap(N_u, m, augment_budget_bytes(N_u, vram_capacity_bytes))
        if k_fit < BOUNDS_K_UU_CLIP_FLOOR
            # `clip_k_uu` errors here. Charge the floor, which is the cheapest
            # augmentation the runtime would have accepted, and label it so the
            # caller can move to a larger card instead of asking for this one.
            k, clip = BOUNDS_K_UU_CLIP_FLOOR, :infeasible
        elseif k_fit < k
            k, clip = k_fit, :budget
        end
    end
    p = min(round(Int, c.bounds_uu_oversamples), max(0, N_u - k))
    return AugmentShape(true, k, m + k, p, round(Int, c.bounds_uu_power_iters),
                        k_req, clip)
end

"""
    with_augmentation(c; k_uu, threshold, oversamples, power_iters) -> Coefficients

`c` with `bounds_augment_mode = "on"` and the augmentation's parameters set.
Everything else is carried over field by field, exactly as
`bench/size_bounds_jobs.jl`'s `without_truncation` does, so a calibrated
coefficient set stays calibrated.

This is the *only* way the augmentation gets charged. A caller that does not call
it predicts what it predicted before the augmentation existed, which is what
`bench/parity_cost_model.jl` asserts.
"""
function with_augmentation(c::Coefficients;
                           k_uu::Integer=BOUNDS_K_UU_DEFAULT,
                           threshold::Integer=BOUNDS_AUGMENT_THRESHOLD_DEFAULT,
                           oversamples::Integer=BOUNDS_UU_OVERSAMPLES,
                           power_iters::Integer=BOUNDS_UU_POWER_ITERS)
    fields = fieldnames(Coefficients)
    nt = NamedTuple{fields}(map(f -> getfield(c, f), fields))
    return Coefficients(; merge(nt, (bounds_augment_mode=(k_uu > 0 ? "on" : "off"),
                                     bounds_k_uu=Float64(k_uu),
                                     bounds_augment_threshold=Float64(threshold),
                                     bounds_uu_oversamples=Float64(oversamples),
                                     bounds_uu_power_iters=Float64(power_iters)))...)
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
    ext_operator_bytes(pt) + 2 * self_operator_bytes(pt)

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

# Refined points

Nothing above changes in *structure* when the gap is refined. The same three
operators are built; each is a composite over region pairs rather than one
circulant, so `green_operators` expands each into a list of `GreenBlock`s and
every sum below runs over that list instead of over four hard-coded tuples. On an
unrefined point every list is one block and the two spellings are the same
arithmetic, term for term.

The two cross-scale block kinds get their own accumulators (`sandwich_*`,
`partition_*`) rather than being folded into `contact_*` and `ext_*`. They run
the same code and are charged the same fitted triples, but at shapes nobody has
measured, so `greens_time_s` puts a named provisional multiplier on each and a
future refit has something to move. See `GreenBlock` and the `Coefficients`
docstring.
"""
function greens_counts(pt::SRPoint)
    ops = green_operators(pt)
    multiregion = vcat(ops.ss, ops.sr, ops.rs, ops.rr)

    blocks = GreenBlock[]
    append!(blocks, ops.rs)                    # (R, S) standalone
    pt.fresh_preload && append!(blocks, ops.rr) # (R, R) shared self operator
    append!(blocks, multiregion)

    # Five block kinds, not two plus a surcharge: an external block between
    # touching bodies runs a different branch of `genEgoCrcExt!` with a large
    # fixed cost and a small per-cell cost, and the two cross-scale kinds are
    # those two paths again at shapes the fit has never seen.
    self_work = 0.0        # sum of M*log2(M) over self blocks
    self_cells = 0         # sum of M over self blocks
    ext_work = 0.0
    ext_cells = 0
    contact_work = 0.0
    contact_cells = 0
    sandwich_work = 0.0
    sandwich_cells = 0
    partition_work = 0.0
    partition_cells = 0
    n_self = 0
    n_ext = 0
    n_contact = 0
    n_sandwich = 0
    n_partition = 0
    for b in blocks
        # A build fills every partition pair of the circulant and transforms them
        # all under one batched plan, so both scale with `pairs`.
        M = circulant_cells(b.target, b.source)
        work, cells = b.pairs * fft_work(M), b.pairs * M
        if b.kind === :self
            self_work += work
            self_cells += cells
            n_self += 1
        elseif b.crossscale && b.kind === :contact
            sandwich_work += work
            sandwich_cells += cells
            n_sandwich += 1
        elseif b.crossscale
            partition_work += work
            partition_cells += cells
            n_partition += 1
        elseif b.kind === :contact
            contact_work += work
            contact_cells += cells
            n_contact += 1
        else
            ext_work += work
            ext_cells += cells
            n_ext += 1
        end
    end

    # Bytes serialised: the retained Fourier data of every operator written out.
    bytes_written = sum(block_retained_bytes, ops.rs)                      # (R, S)
    pt.fresh_preload &&
        (bytes_written += sum(block_retained_bytes, ops.rr))               # (R, R)
    bytes_written += sum(block_retained_bytes, multiregion)                # universe

    # Peak: everything retained by the universe operator except the block
    # currently under construction, plus that block's transient peak. Take the
    # worst ordering. Region-pair granularity, since that is what Gila's
    # constructor loops over.
    retained = block_retained_bytes.(multiregion)
    peaks = block_peak_bytes.(multiregion)
    peak_bytes = 0
    for i in eachindex(multiregion)
        resident = sum(retained[j] for j in eachindex(retained) if j != i; init=0)
        peak_bytes = max(peak_bytes, resident + peaks[i])
    end
    # Serialisation of the finished universe operator buffers a copy.
    peak_bytes = max(peak_bytes, sum(retained) * 2)

    return (n_self_blocks=n_self, n_ext_blocks=n_ext, n_contact_blocks=n_contact,
            n_sandwich_blocks=n_sandwich, n_partition_blocks=n_partition,
            self_fft_work=self_work, ext_fft_work=ext_work, contact_fft_work=contact_work,
            sandwich_fft_work=sandwich_work, partition_fft_work=partition_work,
            self_cells=self_cells, ext_cells=ext_cells, contact_cells=contact_cells,
            sandwich_cells=sandwich_cells, partition_cells=partition_cells,
            n_blocks=n_self + n_ext + n_contact + n_sandwich + n_partition,
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
    # The two cross-scale kinds, guarded so that an unrefined point takes literally
    # the arithmetic it took before gap refinement existed. Each runs the code path
    # its own triple was fitted for, times a provisional scaling; see
    # `Coefficients`.
    if counts.n_sandwich_blocks > 0
        t += c.g0_sandwich_scale *
             (c.g0_contact_fft * counts.sandwich_fft_work +
              c.g0_contact_cell * counts.sandwich_cells / eta +
              c.g0_contact_fixed * counts.n_sandwich_blocks)
    end
    if counts.n_partition_blocks > 0
        t += c.g0_partition_scale *
             (c.g0_ext_fft * counts.partition_fft_work +
              c.g0_ext_cell * counts.partition_cells / eta +
              c.g0_ext_fixed * counts.n_partition_blocks)
    end
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
    N_s = sender_length(pt)
    N_r = receiver_length(pt)
    N_u = N_s + N_r
    q = pt.power_iters
    k = pt.rank

    c_herm = sketch_width(pt)                      # reigen_hermitian does not clamp
    c_svd = sketch_width(pt; clamp_to=min(N_r, N_s))

    herm_applications = c_herm * (q + 2)
    mv_ext = 2 * herm_applications + c_svd * (2q + 2)
    # One apply of `Asym(G⁰ᵣᵣ)` on a refined point, where Gila has no composite
    # `asym` yet and `asym_self` falls back to the difference. See
    # `REFINED_ASYM_SELF_APPLIES`.
    mv_self = asym_self_applies(pt) * herm_applications

    ext_apply = rs_apply_work(pt)
    self_apply = rr_apply_work(pt)

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
            mv_ext_launches=mv_ext * ext_apply.blocks,
            mv_self_launches=mv_self * self_apply.blocks,
            ext_fft_work=mv_ext * ext_apply.fft,
            self_fft_work=mv_self * self_apply.fft,
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

# The `RS/` singular values are a second panel decomposition, and it is counted

`_generate_rsvd_sr` does not stop at `_save_ur_asym`. It then runs
`_run_rsvdvals(..., "RS/")`, which at these sizes takes the panel path too and
builds a residency plan of its own. `rsvdvals_panel` holds two tall matrices at
its peak: the range basis `Q`, `N_r x c_svd`, and the reduction's `Bdag`,
`N_s x c_svd`. They have *different row counts*, so their pool blocks have
different byte sizes and Funicular's size-keyed free list cannot recycle one into
the other; the pool has to hold both. That is `(N_r + N_s) * c_svd * 16`, i.e.
`N_u * c_svd * 16` -- 51 GB at 4 λ, `c = 4050`.

`host_panel_bytes` is the **maximum** of the two phases and not their sum, and
that is a statement about `src/rsvd.jl`, not about Funicular. Funicular has no
API to hand a slab back to the OS, so two coexisting plans really do add; what
`_generate_rsvd_sr` now does is call `reclaim_host_pools!` between the phases so
that the first plan is collected before the second is built, and
`common.jl`'s `residency_plan` subtracts any pool that survived from the next
plan's budget. With that, the process's high-water is one pool at a time and the
honest request is the larger one. Without it -- the state the 4 λ probe RSVD ran
in -- the two pools coexist, 116 GiB is budgeted twice out of a 124 GiB
allocation, and the job is OOM-killed the moment the second pool starts growing.

On every symmetric sender/receiver point the UR term wins (`2c + m` against
`c_svd`, on twice the rows), so this `max` does not move the production requests;
it binds only where the sender and receiver are very different sizes, when
`max(N_r, N_s)` approaches `N_u` and `3 c_svd` can exceed `2 c_herm + m`.
"""
function rsvd_panel_counts(pt::SRPoint)
    N_s = sender_length(pt)
    N_r = receiver_length(pt)
    N_u = N_s + N_r
    q = pt.power_iters

    c_herm = sketch_width(pt)
    c_svd = sketch_width(pt; clamp_to=min(N_r, N_s))

    # Identical to the in-memory path: the panel path does the same applications.
    herm_applications = c_herm * (q + 2)
    mv_ext = 2 * herm_applications + c_svd * (2q + 2)
    mv_self = asym_self_applies(pt) * herm_applications

    ext_apply = rs_apply_work(pt)
    self_apply = rr_apply_work(pt)

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
    host_ur_panel_bytes = (2 * c_herm + m) * N_u * BYTES_PER_COMPLEX
    #=
    `_run_rsvdvals("RS/")`'s own plan: `Q` is `N_r x c_svd` and `Bdag` is
    `N_s x c_svd`, and their blocks are different sizes so neither recycles into
    the other. See the docstring for why the two phases are `max`ed rather than
    summed, and what in `src/rsvd.jl` has to hold for that to be true.
    =#
    host_rs_panel_bytes = (N_r + N_s) * c_svd * BYTES_PER_COMPLEX
    host_panel_bytes = max(host_ur_panel_bytes, host_rs_panel_bytes)
    bytes_written = N_u * m * BYTES_PER_COMPLEX +
                    c_herm * BYTES_PER_COMPLEX + c_svd * BYTES_PER_COMPLEX

    return (mv_ext=mv_ext, mv_self=mv_self,
            mv_ext_launches=mv_ext * ext_apply.blocks,
            mv_self_launches=mv_self * self_apply.blocks,
            ext_fft_work=mv_ext * ext_apply.fft,
            self_fft_work=mv_self * self_apply.fft,
            cholqr_flops=cholqr_flops, gemm_flops=gemm_flops,
            solve_flops=solve_flops,
            sketch_width_herm=c_herm, sketch_width_svd=c_svd,
            panel_width=panel_width(N_u, c_herm), sweeps=sweeps,
            sweep_bytes=sweep_bytes,
            vram_bytes=vram_bytes, host_panel_bytes=host_panel_bytes,
            host_ur_panel_bytes=host_ur_panel_bytes,
            host_rs_panel_bytes=host_rs_panel_bytes,
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
    N_s = sender_length(pt)
    N_r = receiver_length(pt)
    N_u = N_s + N_r

    mv_ext = N_u
    mv_self = asym_self_applies(pt) * N_u
    ext_apply = rs_apply_work(pt)
    self_apply = rr_apply_work(pt)

    solve_flops = flops_eigh(N_u)
    vram_bytes = 3 * N_u^2 * BYTES_PER_COMPLEX + rsvd_operator_vram_bytes(pt)
    host_dense_bytes = N_u^2 * BYTES_PER_COMPLEX
    m = min(effective_num_pos(pt), N_u)
    bytes_written = N_u * m * BYTES_PER_COMPLEX

    return (mv_ext=mv_ext, mv_self=mv_self,
            mv_ext_launches=mv_ext * ext_apply.blocks,
            mv_self_launches=mv_self * self_apply.blocks,
            ext_fft_work=mv_ext * ext_apply.fft,
            self_fft_work=mv_self * self_apply.fft,
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
    mvs = composite_mv_scale(pt, c)
    if mode == :dense_exact
        n = rsvd_dense_counts(pt)
        device = mvs * (c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext_launches)
        device += mvs * (c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self_launches)
        device += n.solve_flops / c.eigh_rate
        host = n.bytes_written / c.disk_write_rate + c.gpu_startup_s
        return (host + device, device)
    elseif mode == :panel
        n = rsvd_panel_counts(pt)
        device = mvs * (c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext_launches)
        device += mvs * (c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self_launches)
        device += n.cholqr_flops / c.gemm_rate + n.gemm_flops / c.gemm_rate
        device += n.solve_flops / c.eigh_rate
        host = n.sweep_bytes / c.pcie_rate * c.overlap_factor
        host += n.bytes_written / c.disk_write_rate + c.gpu_startup_s
        return (host + device, device)
    end
    n = rsvd_counts(pt)
    device = mvs * (c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext_launches)
    device += mvs * (c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self_launches)
    device += n.qr_flops / c.qr_rate + n.gemm_flops / c.gemm_rate
    device += n.solve_flops / c.eigh_rate
    host = n.bytes_written / c.disk_write_rate + c.gpu_startup_s
    return (host + device, device)
end

"Same point with the power iterations removed, for the per-pass split below."
_at_zero_power_iters(pt::SRPoint) =
    SRPoint(pt.sender_cells, pt.receiver_cells, pt.scale, pt.separation, pt.rank,
            pt.oversamples, 0, pt.threads, pt.num_pos, pt.fresh_preload,
            pt.refine_gap)

"""
    composite_mv_scale(pt, c) -> Float64

`mv_composite_scale` on a refined point and exactly `1.0` on every other, so an
unrefined prediction keeps the arithmetic it had before composite operators
existed. See the `Coefficients` docstring for what the multiplier covers.
"""
composite_mv_scale(pt::SRPoint, c::Coefficients) =
    is_refined(pt) ? c.mv_composite_scale : 1.0

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
        return bounds_panel_counts(pt; tau=tau, m=m,
                                   augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes)).vram_bytes +
               c.panel_workspace_bytes + c.bounds_vram_base
    return c.vram_floor_factor *
           bounds_counts(pt; tau=tau, m=m, augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes)).vram_bytes +
           c.bounds_vram_base
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
    outer_indices_swept(m, indices) -> AbstractUnitRange{Int}
    probe_count(m, ns) -> Float64

The slice of the bounds job's outer loop a prediction is for, and how many probe
evaluations that slice costs.

`indices === nothing` means the whole loop, `1:m`, which is what every caller
written before the block split means and what every fitted coefficient was
calibrated against. Anything else is clipped to `1:m`: a block sized from a
measured `m` can hang off the end of a spectrum that has since moved by a few
counts, exactly as it can in `_compute_bounds_sr`.

The probe count is where the outer loop's cost stops being uniform in `n`. Index
`n` probes `k = n, …, m`, so it does `m − n + 1` root finds and gemvs per
τ evaluation, and the first index costs `m` times what the last one does. Summed
over the whole loop that is the familiar `m(m+1)/2`; summed over a block it is
what makes equal-*count* blocks unequal in time, and why
`bench/size_bounds_jobs.jl` sizes them by this instead.
"""
function outer_indices_swept(m::Integer, indices::Union{Nothing,AbstractUnitRange{Int}})
    indices === nothing && return 1:Int(m)
    return max(1, first(indices)):min(Int(m), last(indices))
end

function probe_count(m::Integer, ns::AbstractUnitRange{Int})
    isempty(ns) && return 0.0
    # sum_{n=lo}^{hi} (m - n + 1), in closed form so a 4000-index loop is arithmetic
    lo, hi = first(ns), last(ns)
    return (Float64(m - lo + 1) + Float64(m - hi + 1)) * length(ns) / 2
end

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
    `C` -- each applies `asym(G0_uu)`, whose applications run all four blocks of
    the universe, so `2m` self plus `2m` external Green matvecs on a refined
    point and twice that on an unrefined one (`UU_ASYM_APPLIES`) -- plus the
    `(m x N_u)(N_u x m)` gemms.
  * Pencil whitenings (device heevd, `psd_pencil_whitener`): `TAU_GRID_POINTS`
    shared `m x m` Hermitian eigendecompositions up front, plus
    `TAU_REFINE_EVALS` throwaway ones per index for the golden-section probes.
  * Pencil solves (device, `diag_pencil_eigen`), once per (index, evaluation):
    an `m x m` Hermitian eigendecomposition and two `m x m x m` gemms.
  * Probe loop, `(m - n + 1)` probes per (index, evaluation) -- about
    `evals * m^2/2` in total: an `m x m` device gemv, one device-to-host
    transfer of the projected b-vector, and a host-side Brent root find over
    length-`m` resolvent expansions.

# The augmented path

Under `bounds_augment_mode = "on"` (see `bounds_augment`) a far-field point first
runs `reigen_hermitian` on `Asym(G⁰ᵤᵤ)` at sketch width `k_uu + p`, orthonormalizes
`[g_kept, U_uu]`, and then does everything above at `m_aug = m + k_uu` rather than
at `m`. Three things change and one does not:

  * the pencil stage's dimension, everywhere. `flops_eigh(m_aug)`,
    `flops_gemm(m_aug, m_aug, m_aug)`, the probe gemv, the resolvent length.
  * the Green sweep, which now projects `m_aug` columns rather than `m`, hence
    `2 m_aug` self plus `2 m_aug` external matvecs per universe apply.
  * device residency: `2 m_aug + m` tall columns for the front end, against the
    `3 m` of the plain path, plus the sketch's own three `N_u x (k_uu + p)`
    matrices, which are transient but peak *before* the front end rather than
    alongside it, hence a `max` and not a sum.
  * what does **not** change is the outer loop's length or the probe count. Both
    are `m`: the channels being bounded are still the `g` directions, and the
    probes are still built from `Πₛ·gs_pos` alone. That is why a point at `m = 15`
    and `m_aug = 527` is minutes rather than hours.

The `uu_*` counts are the same expressions `rsvd_counts` uses for
`reigen_hermitian`, evaluated at this width and against this operator: `c(q + 2)`
applications, each of `asym(G⁰ᵤᵤ)` on the `[S, R] <- [S, R]` universe, so `2` self
plus `2` external matvecs apiece per universe apply, `q + 1` thin QRs, the reduction gemms
and one `c x c` eigensolve. Checked against the measured 1 λ point (`N_u = 196,608`,
`k_uu = 512`, `q = 4`): the model says 273 s of device time against 227 s measured,
i.e. 20% high, which is the direction a request should err in.

Every one of these terms is exactly zero, and every `m_aug` is exactly `m`, when
the augmentation is off.
"""
function bounds_counts(pt::SRPoint; tau::TauShape=TAU_SHAPE_LEGACY,
                      m::Union{Nothing,Integer}=nothing,
                      augment::Union{Nothing,AugmentShape}=nothing,
                      indices::Union{Nothing,AbstractUnitRange{Int}}=nothing)
    N_u = universe_length(pt)
    k = pt.rank
    m = m === nothing ? min(effective_num_pos(pt), N_u) : min(Int(m), N_u)
    aug = augment === nothing ? AUGMENT_OFF(m) : augment
    m_aug = min(aug.m_aug, N_u)
    pairs = m * (m - 1) / 2
    evals_per_index = tau_evals_per_index(tau)
    ns = outer_indices_swept(m, indices)
    n_indices = length(ns)
    probes = evals_per_index * probe_count(m, ns)

    gs_bytes = pairs * 3 * N_u * BYTES_PER_COMPLEX   # read s_j, read/write w_i
    gs_launches = 2 * pairs

    uni = universe_apply_work(pt)
    uu_applies = uu_asym_applies(pt)
    mv_ext = 2 * uu_applies * m_aug
    mv_self = 2 * uu_applies * m_aug

    # Device-side dense work: the m_aug-wide basis projections, then the pencil
    # stage -- whitenings (grid shared, refinement per index) and one
    # diag_pencil_eigen / factored_pencil_eigen (eigh + two gemms) per evaluation,
    # all through CUSOLVER/CUBLAS. Only the root finds stay on the host.
    gemm_flops = flops_gemm(m_aug, N_u, m) +            # ss_basis (and W = basis'gs)
                 2 * m_aug * flops_gemm(N_u, m, 1) +    # C: G' v and G w per application
                 flops_gemm(m_aug, N_u, m_aug) +        # basis' * (C basis)
                 flops_gemm(m_aug, N_u, m_aug)          # D: Bs' Bs on the sender rows
    whitenings = tau.grid_points + n_indices * tau.refine_whitenings
    pencil_eigh_flops = (whitenings + n_indices * evals_per_index) * flops_eigh(m_aug)
    pencil_gemm_flops = 2 * n_indices * evals_per_index * flops_gemm(m_aug, m_aug, m_aug)
    probe_gemv_flops = probes * flops_gemm(m_aug, m_aug, 1)
    root_work = probes * m_aug

    # The Asym(G⁰ᵤᵤ) solve and the augmentation's QR. All zero when off.
    c_uu = aug.augmented ? aug.k_uu + aug.oversamples : 0
    q_uu = aug.power_iters
    uu_applications = aug.augmented ? c_uu * (q_uu + 2) : 0
    uu_mv_ext = 2 * uu_applies * uu_applications
    uu_mv_self = 2 * uu_applies * uu_applications
    uu_qr_flops = aug.augmented ? (q_uu + 1) * flops_qr(N_u, c_uu) : 0.0
    uu_gemm_flops = aug.augmented ?
        2 * flops_gemm(N_u, c_uu, c_uu) + flops_gemm(N_u, c_uu, aug.k_uu) : 0.0
    uu_solve_flops = aug.augmented ? flops_eigh(c_uu) : 0.0
    # augmented_basis: two classical Gram-Schmidt passes against the g block (two
    # gemms each) then a Householder QR of the N_u x k_uu remainder.
    aug_qr_flops = aug.augmented ?
        2 * (flops_gemm(m, N_u, aug.k_uu) + flops_gemm(N_u, m, aug.k_uu)) +
        flops_qr(N_u, aug.k_uu) : 0.0
    uu_sketch_bytes = aug.augmented ? 3 * N_u * c_uu * BYTES_PER_COMPLEX : 0

    #=
    Three `N_u x m` device matrices, which is `bounds_footprint_bytes` in
    `src/bounds.jl` exactly: the basis, the orthonormalized `ss = similar(basis)`,
    and the `out = similar(B, N_u, m)` that `opmat` allocates for `C * basis`. That
    is the front end's live peak and the same count the job's own
    `use_panel_bounds` predicate tests, so the two cannot disagree about whether a
    card is big enough.

    Nothing on the device is a function of `k`. `load_bounds_inputs` applies the
    gamma truncation on the host as a count *before* it reads anything
    (`num_pos = _gamma_kept_count(...)`, then `cols = sorted_idxs[1:num_pos]`), and
    the single host-to-device conversion is `CuArray(Vpos)` on an `N_u x m` host
    matrix. The rank-`k` block is a host-side read and a host-side spectrum; it is
    never resident. The `3 * k * N_u` term this replaces was the RSVD sketch's
    footprint (`Omega`, `Q`, `A*Q`) copied onto a job that has no sketch, and the
    calibration refutes it directly: at `k = 4000`, `N_u = 196608` it claims 37.7 GB
    of resident device memory for three bounds jobs that ran to completion inside a
    19.6 GiB MIG slice.

    The pencil arena lives on the device with the whitenings: C_basis, D_basis,
    S_basis, ss_basis, the working whitener + eigenvectors and the per-evaluation
    temporaries (the `8`), plus one `m^2` for each shared grid pencil and each
    refinement pencil the LRU cache holds. One, not two, per pencil: the whitener is
    `m x rank` and the null space `m x num_null` with `rank + num_null == m`, so the
    pair is `m^2` complex between them, which is what this line's own comment always
    said and what the code does.
    =#
    # The tall term is `bounds_footprint_bytes`'s generalization,
    # `augmented_footprint_bytes(N_u, m, m_aug)` in `src/bounds.jl`: the basis, the
    # `ss` probes (`m` wide, because they are built from the g columns alone) and
    # `opmat`'s destination. At `m_aug == m` it is `3 m N_u 16` exactly.
    #
    # `max` and not `+` against the sketch: `reigen_hermitian` finishes, its three
    # matrices are freed and `CUDA.reclaim()`ed, and only then does the front end
    # allocate. The two peaks do not coexist, and charging their sum would push the
    # 4 lambda point off an A100-40 for a peak that never happens.
    vram_bytes = max((2 * m_aug + m) * N_u * BYTES_PER_COMPLEX, uu_sketch_bytes) +
                 (tau.grid_points + tau.cache_entries + 8) * m_aug^2 * BYTES_PER_COMPLEX +
                 universe_operator_bytes(pt)
    # One host-side copy of the eigenvector block; JLD2's own buffering and the
    # `CuArray(...)` staging copy are what `bounds_host_mem_factor` absorbs.
    host_bytes = N_u * k * BYTES_PER_COMPLEX
    bytes_read = N_u * k * BYTES_PER_COMPLEX

    return (num_pos=m, m_aug=m_aug, gs_bytes=gs_bytes, gs_launches=gs_launches,
            mv_ext=mv_ext, mv_self=mv_self,
            mv_ext_launches=mv_ext * uni.ext_blocks,
            mv_self_launches=mv_self * uni.self_blocks,
            ext_fft_work=mv_ext * uni.ext_fft,
            self_fft_work=mv_self * uni.self_fft,
            gemm_flops=gemm_flops,
            pencil_eigh_flops=pencil_eigh_flops,
            pencil_gemm_flops=pencil_gemm_flops,
            probe_gemv_flops=probe_gemv_flops, probes=probes,
            root_work=root_work,
            uu_mv_ext=uu_mv_ext, uu_mv_self=uu_mv_self,
            uu_mv_ext_launches=uu_mv_ext * uni.ext_blocks,
            uu_mv_self_launches=uu_mv_self * uni.self_blocks,
            uu_ext_fft_work=uu_mv_ext * uni.ext_fft,
            uu_self_fft_work=uu_mv_self * uni.self_fft,
            uu_qr_flops=uu_qr_flops, uu_gemm_flops=uu_gemm_flops,
            uu_solve_flops=uu_solve_flops, aug_qr_flops=aug_qr_flops,
            uu_sketch_bytes=uu_sketch_bytes,
            vram_bytes=vram_bytes, host_bytes=host_bytes,
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
  * Device memory is staging buffers plus the pencil arena. The `3 * m * N_u` tall
    term of the in-memory count is gone, leaving the cached grid whiteners,
    the working whitener and eigenvectors, and the projected `m x m` blocks, that
    is, the `(TAU_GRID_POINTS + cache_entries + 8) * m^2` term, unchanged. The
    operator's own device workspace is charged on top of this count by
    `bounds_vram_bytes`, since it is a coefficient rather than something countable
    from the code.
  * The read shrinks to the `m` positive vectors, which is all that was saved.

The `m` applications of `C` and their Green matvecs are unchanged: `panelmul!`
applies the operator column by column, exactly as the resident path does.
"""
function bounds_panel_counts(pt::SRPoint; tau::TauShape=TAU_SHAPE_LEGACY,
                            m::Union{Nothing,Integer}=nothing,
                            augment::Union{Nothing,AugmentShape}=nothing,
                            indices::Union{Nothing,AbstractUnitRange{Int}}=nothing)
    N_u = universe_length(pt)
    m = m === nothing ? min(effective_num_pos(pt), N_u) : min(Int(m), N_u)
    aug = augment === nothing ? AUGMENT_OFF(m) : augment
    m_aug = min(aug.m_aug, N_u)
    evals_per_index = tau_evals_per_index(tau)
    ns = outer_indices_swept(m, indices)
    n_indices = length(ns)
    probes = evals_per_index * probe_count(m, ns)

    uni = universe_apply_work(pt)
    uu_applies = uu_asym_applies(pt)
    mv_ext = 2 * uu_applies * m_aug
    mv_self = 2 * uu_applies * m_aug

    # The panelized front end: one gram sweep, an m x m host Cholesky, one
    # rightmul! sweep. Bytes are counted up and back for each sweep.
    gs_gemm_flops = 2 * flops_gemm(N_u, m, m)
    gs_sweep_bytes = BOUNDS_PANEL_SWEEPS * N_u * m * BYTES_PER_COMPLEX * 2
    gs_cholesky_flops = flops_cholesky(m)

    gemm_flops = flops_gemm(m_aug, N_u, m) +            # ss_basis (and W = basis'gs)
                 2 * m_aug * flops_gemm(N_u, m, 1) +    # C: G' v and G w per application
                 flops_gemm(m_aug, N_u, m_aug) +        # basis' * (C basis)
                 flops_gemm(m_aug, N_u, m_aug)          # D: Bs' Bs on the sender rows
    whitenings = tau.grid_points + n_indices * tau.refine_whitenings
    pencil_eigh_flops = (whitenings + n_indices * evals_per_index) * flops_eigh(m_aug)
    pencil_gemm_flops = 2 * n_indices * evals_per_index * flops_gemm(m_aug, m_aug, m_aug)
    probe_gemv_flops = probes * flops_gemm(m_aug, m_aug, 1)
    root_work = probes * m_aug

    # The Asym(G0_uu) solve and the augmentation's QR; see `bounds_counts` for the
    # counts and for the check against the measured 1 lambda point. All zero when
    # the augmentation is off.
    c_uu = aug.augmented ? aug.k_uu + aug.oversamples : 0
    q_uu = aug.power_iters
    uu_applications = aug.augmented ? c_uu * (q_uu + 2) : 0
    uu_mv_ext = 2 * uu_applies * uu_applications
    uu_mv_self = 2 * uu_applies * uu_applications
    uu_qr_flops = aug.augmented ? (q_uu + 1) * flops_qr(N_u, c_uu) : 0.0
    uu_gemm_flops = aug.augmented ?
        2 * flops_gemm(N_u, c_uu, c_uu) + flops_gemm(N_u, c_uu, aug.k_uu) : 0.0
    uu_solve_flops = aug.augmented ? flops_eigh(c_uu) : 0.0
    aug_qr_flops = aug.augmented ?
        2 * (flops_gemm(m, N_u, aug.k_uu) + flops_gemm(N_u, m, aug.k_uu)) +
        flops_qr(N_u, aug.k_uu) : 0.0
    uu_sketch_bytes = aug.augmented ? 3 * N_u * c_uu * BYTES_PER_COMPLEX : 0

    #=
    One `m^2` per pencil, as in `bounds_counts`: whitener plus null space.

    An augmented point is charged the *dense* tall term on top of the staging
    buffers, because it really does hold three dense `N_u`-tall matrices. This mode
    label is inherited from the RSVD's storage choice (did it write an h5 or an
    inline JLD2 block?) and not from the bounds front end's: `bounds_from_spectrum`
    refuses the augmented/panel combination outright, so an augmenting point reads
    its `m` columns out of the h5 into one dense block and proceeds densely. `max`
    against the sketch for the same reason as in `bounds_counts`: the two peaks do
    not coexist.

    What keeps that refusal from firing is this count and the card selection built
    on it, not slack in `use_panel_bounds`. At `augment_threshold = 1000` the two
    predicates are no longer far apart on a small slice (`use_panel_bounds` at
    1 lambda, `m = 999` wants 13.6 GiB against a 10 GB MIG's 8.4 GiB budget), but
    the dense tall term charged here is larger still, so a caller sizing this point
    rejects that slice before the job ever sees it. A card chosen against a count
    that did *not* include the tall term is what would produce the refusal.

    The `gs_*` terms are left as the panelized ones even when augmenting, where the
    run really does the `O(m^2)` BLAS-1 loop instead. At the `m < 1000` that lets a
    point augment both are seconds against a job of many minutes, and keeping one
    expression here is worth more than the third decimal place.
    =#
    vram_bytes = (aug.augmented ?
                  max(panel_staging_bytes(N_u, m),
                      (2 * m_aug + m) * N_u * BYTES_PER_COMPLEX,
                      uu_sketch_bytes) :
                  panel_staging_bytes(N_u, m)) +
                 (tau.grid_points + tau.cache_entries + 8) * m_aug^2 * BYTES_PER_COMPLEX +
                 universe_operator_bytes(pt)
    host_bytes = 3 * N_u * m * BYTES_PER_COMPLEX
    bytes_read = N_u * m * BYTES_PER_COMPLEX

    return (num_pos=m, m_aug=m_aug,
            gs_gemm_flops=gs_gemm_flops, gs_sweep_bytes=gs_sweep_bytes,
            gs_cholesky_flops=gs_cholesky_flops,
            mv_ext=mv_ext, mv_self=mv_self,
            mv_ext_launches=mv_ext * uni.ext_blocks,
            mv_self_launches=mv_self * uni.self_blocks,
            ext_fft_work=mv_ext * uni.ext_fft,
            self_fft_work=mv_self * uni.self_fft,
            gemm_flops=gemm_flops,
            pencil_eigh_flops=pencil_eigh_flops,
            pencil_gemm_flops=pencil_gemm_flops,
            probe_gemv_flops=probe_gemv_flops, probes=probes,
            root_work=root_work,
            uu_mv_ext=uu_mv_ext, uu_mv_self=uu_mv_self,
            uu_mv_ext_launches=uu_mv_ext * uni.ext_blocks,
            uu_mv_self_launches=uu_mv_self * uni.self_blocks,
            uu_ext_fft_work=uu_mv_ext * uni.ext_fft,
            uu_self_fft_work=uu_mv_self * uni.self_fft,
            uu_qr_flops=uu_qr_flops, uu_gemm_flops=uu_gemm_flops,
            uu_solve_flops=uu_solve_flops, aug_qr_flops=aug_qr_flops,
            uu_sketch_bytes=uu_sketch_bytes,
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

`indices` predicts one *block* of the outer loop (`compute_bounds.jl
--outer-range lo:hi`) rather than the whole of it. Everything outside the loop
(the Gram-Schmidt, the Green sweep, the shared grid whitenings, the `Asym(G⁰ᵤᵤ)`
solve) is charged in full, because a block really does run all of it; only the
per-index terms shrink. Two consequences worth naming:

  * `indices = 1:0` is the front-end cost on its own, which is what a block pays
    before it computes anything. `bench/size_bounds_jobs.jl` reads it off exactly
    that way, and it is why B blocks cost more in total than one job does.
  * the cost is affine in the two counts the slice moves, `length(ns)` and
    `probe_count`, so a block's time is `F + Σ_{n ∈ block} (α + β(m − n + 1))`
    with `F`, `α` and `β` recoverable from three evaluations of this function.
    That is what makes an equal-*time* split a closed-form calculation rather
    than a search.

`nothing`, the default, is the whole loop and reproduces the previous arithmetic
term for term.
"""
function bounds_time_s(pt::SRPoint, c::Coefficients;
                       vram_capacity_bytes::Union{Nothing,Real}=nothing,
                       indices::Union{Nothing,AbstractUnitRange{Int}}=nothing)
    tau = tau_shape(c)
    m = bounds_m(pt, c)
    mvs = composite_mv_scale(pt, c)
    if bounds_mode(pt, vram_capacity_bytes, c) == :panel
        n = bounds_panel_counts(pt; tau=tau, m=m, augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes),
                                indices=indices)
        device = n.gs_gemm_flops / c.gemm_rate
        device += mvs * (c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext_launches)
        device += mvs * (c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self_launches)
        device += n.gemm_flops / c.gemm_rate
        device += n.pencil_eigh_flops / c.eigh_rate
        device += (n.pencil_gemm_flops + n.probe_gemv_flops) / c.gemm_rate
        if n.uu_mv_ext > 0
            device += mvs * (c.mv_ext_fft * n.uu_ext_fft_work + c.mv_ext_fixed * n.uu_mv_ext_launches)
            device += mvs * (c.mv_self_fft * n.uu_self_fft_work + c.mv_self_fixed * n.uu_mv_self_launches)
            device += (n.uu_qr_flops + n.aug_qr_flops) / c.qr_rate
            device += n.uu_gemm_flops / c.gemm_rate
            device += n.uu_solve_flops / c.eigh_rate
        end
        host = n.gs_sweep_bytes / c.pcie_rate * c.overlap_factor
        host += n.gs_cholesky_flops / c.eigh_rate
        host += n.probes * c.sync_latency + n.root_work * c.host_root_find
        host += n.bytes_read / c.disk_read_rate + c.gpu_startup_s
        return (host + device, device)
    end
    n = bounds_counts(pt; tau=tau, m=m, augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes), indices=indices)
    device = n.gs_bytes / c.bandwidth + n.gs_launches * c.launch_latency
    device += mvs * (c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext_launches)
    device += mvs * (c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self_launches)
    device += n.gemm_flops / c.gemm_rate
    device += n.pencil_eigh_flops / c.eigh_rate
    device += (n.pencil_gemm_flops + n.probe_gemv_flops) / c.gemm_rate
    #=
    The Asym(G⁰ᵤᵤ) solve and the augmentation's QR, all of it device work. Guarded
    rather than added as zeros so that an unaugmented point takes literally the
    same arithmetic path it did before this existed, which is what
    bench/parity_cost_model.jl checks.

    At the measured 1 lambda point (N_u = 196,608, k_uu = 512, q = 4, narval's
    coefficients) this block alone comes to 273 s, of which 272 s is the 13,488
    external plus 13,488 self matvecs; the measurement was 227 s. The dense terms
    are under a second, so the whole estimate lives or dies on the matvec rates,
    which are the best-calibrated coefficients in the set.
    =#
    if n.uu_mv_ext > 0
        device += mvs * (c.mv_ext_fft * n.uu_ext_fft_work + c.mv_ext_fixed * n.uu_mv_ext_launches)
        device += mvs * (c.mv_self_fft * n.uu_self_fft_work + c.mv_self_fixed * n.uu_mv_self_launches)
        device += (n.uu_qr_flops + n.aug_qr_flops) / c.qr_rate
        device += n.uu_gemm_flops / c.gemm_rate
        device += n.uu_solve_flops / c.eigh_rate
    end
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
        return bounds_panel_counts(pt; tau=tau, m=m,
                                   augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes)).vram_bytes +
               c.panel_workspace_bytes + c.bounds_vram_base
    return c.bounds_vram_factor *
           bounds_counts(pt; tau=tau, m=m, augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes)).vram_bytes +
           c.bounds_vram_base
end

function bounds_host_bytes(pt::SRPoint, c::Coefficients;
                           vram_capacity_bytes::Union{Nothing,Real}=nothing)
    tau, m = tau_shape(c), bounds_m(pt, c)
    bounds_mode(pt, vram_capacity_bytes, c) == :panel &&
        return c.panel_host_mem_factor *
               bounds_panel_counts(pt; tau=tau, m=m,
                                   augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes)).host_bytes +
               c.bounds_host_mem_base
    return c.bounds_host_mem_factor *
           bounds_counts(pt; tau=tau, m=m, augment=bounds_augment(pt, c, m; vram_capacity_bytes=vram_capacity_bytes)).host_bytes +
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

`indices` (`ComputeBounds` only, `nothing` for the whole loop) predicts one
`--outer-range` block: the front end in full, the outer loop only over those
channel indices. The memory numbers are unchanged by it, since a block holds the
same front end, so only `time_s` and `device_time_s` move. See `bounds_time_s`.
"""
function predict(job::JobKind, pt::SRPoint, coeffs::Coefficients=coefficients_for("molering");
                 pad::Bool=true, vram_capacity_bytes::Union{Nothing,Real}=nothing,
                 indices::Union{Nothing,AbstractUnitRange{Int}}=nothing)
    indices === nothing || job == ComputeBounds || throw(ArgumentError(
        "`indices` slices the bounds job's outer loop over channel indices; it means " *
        "nothing for $(job)"))
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
        t, device_t = bounds_time_s(pt, coeffs; vram_capacity_bytes=vram_capacity_bytes,
                                    indices=indices)
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
