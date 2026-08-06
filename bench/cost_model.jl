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

# Padding
- `time_pad`, `host_mem_pad`, `vram_pad`: safety factors applied by `predict`.
  Set these from the measured spread (see `bench/fit.jl`), not by taste.
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

    time_pad::Float64 = 1.5
    # Small on purpose: the memory margin lives in `*_mem_factor`, which multiplies
    # an analytic count read off the allocations. This is only for run-to-run slop.
    host_mem_pad::Float64 = 1.15
    vram_pad::Float64 = 1.15
end

thread_efficiency(coeffs::Coefficients, threads::Int) =
    1 + coeffs.g0_thread_scaling * (max(1, threads) - 1)

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
    # freshly applied operator*Q simultaneously, all N_u x c complex.
    dense_vram = 3 * N_u * c_herm * BYTES_PER_COMPLEX
    operator_vram = ext_fourier_bytes(pt.receiver_cells, pt.sender_cells) +
                    2 * self_fourier_bytes(pt.receiver_cells) # G0_rr and its Asym copy
    vram_bytes = dense_vram + operator_vram

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

"See `bounds_time_s` for why the device-bound part is reported separately."
function rsvd_time_s(pt::SRPoint, c::Coefficients)
    n = rsvd_counts(pt)
    device = c.mv_ext_fft * n.ext_fft_work + c.mv_ext_fixed * n.mv_ext
    device += c.mv_self_fft * n.self_fft_work + c.mv_self_fixed * n.mv_self
    device += n.qr_flops / c.qr_rate + n.gemm_flops / c.gemm_rate
    device += n.solve_flops / c.eigh_rate
    host = n.bytes_written / c.disk_write_rate + c.gpu_startup_s
    return (host + device, device)
end

rsvd_vram_bytes(pt::SRPoint, c::Coefficients) =
    c.rsvd_vram_factor * rsvd_counts(pt).vram_bytes + c.rsvd_vram_base

"""
    rsvd_vram_floor_bytes(pt, c), bounds_vram_floor_bytes(pt, c)

Least device memory the job can be squeezed into, as opposed to what it will use
if given room. Compare *this* against a card's capacity to decide whether the job
can run there; see `vram_floor_factor`.
"""
rsvd_vram_floor_bytes(pt::SRPoint, c::Coefficients) =
    c.vram_floor_factor * rsvd_counts(pt).vram_bytes + c.rsvd_vram_base
bounds_vram_floor_bytes(pt::SRPoint, c::Coefficients) =
    c.vram_floor_factor * bounds_counts(pt).vram_bytes + c.bounds_vram_base

rsvd_host_bytes(pt::SRPoint, c::Coefficients) =
    c.rsvd_host_mem_factor * rsvd_counts(pt).host_dense_bytes + c.rsvd_host_mem_base

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
function bounds_counts(pt::SRPoint)
    N_u = universe_length(pt)
    k = pt.rank
    m = min(effective_num_pos(pt), N_u)
    pairs = m * (m - 1) / 2
    evals_per_index = TAU_GRID_POINTS + TAU_REFINE_EVALS
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
    whitenings = TAU_GRID_POINTS + m * TAU_REFINE_EVALS
    pencil_eigh_flops = (whitenings + m * evals_per_index) * flops_eigh(m)
    pencil_gemm_flops = 2 * m * evals_per_index * flops_gemm(m, m, m)
    probe_gemv_flops = probes * flops_gemm(m, m, 1)
    root_work = probes * m

    # The pencil arena lives on the device with the whitenings: C_basis, D, S,
    # ss_basis, the working whitener + eigenvectors, and the cached grid
    # whiteners (whitener + nullspace ~ m^2 each).
    vram_bytes = (3 * k + 2 * m) * N_u * BYTES_PER_COMPLEX +
                 (2 * TAU_GRID_POINTS + 8) * m^2 * BYTES_PER_COMPLEX +
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
    bounds_time_s(pt, c) -> (total, device_bound)

Total predicted seconds, and how many of them are actually device throughput.

The split matters because a MIG slice gets a fraction of the streaming
multiprocessors, so only the device-bound part stretches when you ask for one.
The pencil eigendecompositions and gemms run on the device (heevd/CUBLAS) and
stretch; the per-probe Brent root finds are single-threaded host Julia with the
GPU idle, and scaling them by the slice fraction would over-request on exactly
the small-body sweeps that are otherwise cheap enough to fit in a slice.
"""
function bounds_time_s(pt::SRPoint, c::Coefficients)
    n = bounds_counts(pt)
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

bounds_vram_bytes(pt::SRPoint, c::Coefficients) =
    c.bounds_vram_factor * bounds_counts(pt).vram_bytes + c.bounds_vram_base

bounds_host_bytes(pt::SRPoint, c::Coefficients) =
    c.bounds_host_mem_factor * bounds_counts(pt).host_bytes + c.bounds_host_mem_base

# --------------------------------------------------------------------------- #
# Public prediction API
# --------------------------------------------------------------------------- #

"""
    predict(job, pt, coeffs; pad=true) -> NamedTuple

Predicted `(time_s, host_bytes, vram_bytes)` for one job on one point.
`vram_bytes` is zero for the CPU-only Green-function job. With `pad=true` the
coefficient set's safety factors are applied, plus the flat
`RECOMPILE_OVERHEAD_S` on the time (added after `time_pad`, and left out of
`device_time_s` so MIG-slice stretching never multiplies it).
"""
function predict(job::JobKind, pt::SRPoint, coeffs::Coefficients=coefficients_for("molering");
                 pad::Bool=true)
    if job == GenerateGreens
        # CPU-only job: no device-bound share to stretch for a GPU slice.
        t, device_t = greens_time_s(pt, coeffs), 0.0
        host, vram, floor = greens_host_bytes(pt, coeffs), 0.0, 0.0
    elseif job == GenerateRSVD
        t, device_t = rsvd_time_s(pt, coeffs)
        host, vram = rsvd_host_bytes(pt, coeffs), rsvd_vram_bytes(pt, coeffs)
        floor = rsvd_vram_floor_bytes(pt, coeffs)
    elseif job == ComputeBounds
        t, device_t = bounds_time_s(pt, coeffs)
        host, vram = bounds_host_bytes(pt, coeffs), bounds_vram_bytes(pt, coeffs)
        floor = bounds_vram_floor_bytes(pt, coeffs)
    else
        error("Unknown job kind: $job")
    end
    if pad
        t *= coeffs.time_pad
        device_t *= coeffs.time_pad
        host *= coeffs.host_mem_pad
        vram *= coeffs.vram_pad
        floor *= coeffs.vram_pad
        t += RECOMPILE_OVERHEAD_S
    end
    return (time_s=t, device_time_s=device_t, host_bytes=host, vram_bytes=vram,
            vram_floor_bytes=floor)
end

predict_time_s(job::JobKind, pt::SRPoint, coeffs::Coefficients; pad::Bool=true) =
    predict(job, pt, coeffs; pad=pad).time_s
predict_host_bytes(job::JobKind, pt::SRPoint, coeffs::Coefficients; pad::Bool=true) =
    predict(job, pt, coeffs; pad=pad).host_bytes
predict_vram_bytes(job::JobKind, pt::SRPoint, coeffs::Coefficients; pad::Bool=true) =
    predict(job, pt, coeffs; pad=pad).vram_bytes

end # module
