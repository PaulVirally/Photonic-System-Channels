#!/usr/bin/env julia
"""
    bench/plan.jl

Generate the calibration run script for one cluster.

    julia bench/plan.jl --cluster fir      --tier quick
    julia bench/plan.jl --cluster narval   --tier full
    julia bench/plan.jl --cluster molering --tier full

Writes `bench/launch_calibration_<cluster>_<tier>.sh` plus a manifest CSV listing
every planned point, and prints the copy/run instructions.

# Why these points

Each fitted coefficient in `CostModel.Coefficients` needs to be identified by
some point in this plan, and nothing here exists for any other reason:

  * `g0_self` / `g0_ext` across eight body sizes separate the `M log M` (FFT of
    the circulant) term from the `M` (per-cell quadrature) term, and separate
    self blocks from external ones -- external blocks keep eight times as much
    Fourier data.
  * `g0_ext` at five separations, one of them zero, isolates the contact
    surcharge. Cost is expected to be flat in separation otherwise; the
    1000-wavelength point is there to confirm that, because the previous
    estimator's blow-up came from assuming the opposite.
  * The thread scan fixes the parallel efficiency of the quadrature loops, which
    is what decides how many cores are worth requesting.
  * `matvec_self` / `matvec_ext` across sizes give the per-matvec cost that
    dominates the RSVD job, at the shapes it actually uses.
  * `dense` across `(m, c)` gives the QR / gemm / eigen rates. These are *not*
    negligible: at `c ~ 3000` the `q + 1` thin QRs of an `N_u x c` matrix are
    comparable to the entire matvec bill, and the previous estimator omitted
    them entirely.
  * `bounds_core` gives the bounds job's Gram-Schmidt, projection and inner-loop
    costs without needing an RSVD output to exist first.
  * `stage_greens` / `stage_rsvd` / `stage_bounds` chains are the validation:
    they check the assembled model against reality on a handful of real runs.

# Tiers

  * `quick`   -- primitives on the four smallest bodies. Enough to identify every
                 coefficient, and short enough to run in an afternoon. Start here.
  * `full`    -- the same primitives across every body, including the large
                 anisotropic ones, so the fit interpolates rather than
                 extrapolates at the sizes you actually submit.
  * `validate` -- end-to-end `greens -> rsvd -> bounds` chains only. Run these
                 *after* fitting, to check the assembled model against reality.
                 They are the expensive tier: on molering, where points run
                 sequentially, budget days rather than hours.
"""

include(joinpath(@__DIR__, "cost_model.jl"))
using .CostModel
using Printf
using Dates

# --------------------------------------------------------------------------- #
# Cluster configuration
# --------------------------------------------------------------------------- #

const CC_UNAME = "pvirally"
const CC_ACCOUNT = "def-smolesky"
const CC_CODE_DIR = "/home/$(CC_UNAME)/Photonic-System-Channels/"
const CC_CAL_ROOT = "/home/$(CC_UNAME)/scratch/psc-calibration/"

const MOLERING_UNAME = "paulv"
const MOLERING_CODE_DIR = "/home/$(MOLERING_UNAME)/Projects/Photonic-System-Channels/"
const MOLERING_CAL_ROOT = "/home/molering/fatmole/$(MOLERING_UNAME)/psc-calibration/"

struct ClusterSpec
    name::String
    has_slurm::Bool
    code_dir::String
    cal_root::String
    account::String
    modules::String
    full_gpu::String     # SLURM --gpus argument for a whole GPU
    max_cores::Int
    max_host_GB::Int
    max_vram_GB::Int
end

#=
`max_cores` and `max_host_GB` are the Alliance's published per-GPU bundle, so a
calibration point that stays inside them is billed as exactly one GPU-equivalent.
=#
function ClusterSpec(name::AbstractString)
    name == "fir" && return ClusterSpec("fir", true, CC_CODE_DIR, CC_CAL_ROOT, CC_ACCOUNT,
                                        "module load StdEnv/2023 julia/1.12.5 cuda/12.2",
                                        "h100:1", 12, 288, 80)
    name == "narval" && return ClusterSpec("narval", true, CC_CODE_DIR, CC_CAL_ROOT, CC_ACCOUNT,
                                           "module load StdEnv/2023 julia/1.12.5 cuda/12.2",
                                           "a100:1", 12, 124, 40)
    name == "nibi" && return ClusterSpec("nibi", true, CC_CODE_DIR, CC_CAL_ROOT, CC_ACCOUNT,
                                         "module load StdEnv/2023 julia/1.12.5 cuda/12.2",
                                         "h100:1", 14, 250, 80)
    name == "rorqual" && return ClusterSpec("rorqual", true, CC_CODE_DIR, CC_CAL_ROOT, CC_ACCOUNT,
                                            "module load StdEnv/2023 julia/1.12.5 cuda/12.2",
                                            "h100:1", 16, 124, 80)
    # 128 logical threads (64 physical cores with SMT): molering is an AMD Ryzen
    # Threadripper Pro 5995WX, and production jobs there get all of them, so the
    # thread scan has to reach 128. The second hyperthread on each core adds little
    # for FFT and quadrature work; the measured optimum is 16.
    name == "molering" && return ClusterSpec("molering", false, MOLERING_CODE_DIR,
                                             MOLERING_CAL_ROOT, "", "", "a6000", 128, 480, 48)
    error("Unknown cluster '$name'. Known: fir, narval, nibi, rorqual, molering.")
end

# --------------------------------------------------------------------------- #
# The bodies we calibrate on
# --------------------------------------------------------------------------- #

#=
Cell counts and ranks track the runs already in `data analysis/data`, so that the
fit is interpolating over the region the real jobs live in rather than
extrapolating into it. The anisotropic entries use the negative-scale convention
from `SMRSystem`: `scale = -1//8` means cells of (1/32, 1/8, 1/8) wavelengths,
which is how the existing 3-lambda and 4-lambda runs keep their cell counts
manageable.
=#
const BODIES = [
    (label="l0p25", cells=(8, 8, 8), scale=1 // 32, rank=1350, tier=:quick),
    (label="l0p5", cells=(16, 16, 16), scale=1 // 32, rank=1350, tier=:quick),
    (label="l0p75", cells=(24, 24, 24), scale=1 // 32, rank=1350, tier=:quick),
    (label="l1", cells=(32, 32, 32), scale=1 // 32, rank=2750, tier=:quick),
    (label="l2agiso", cells=(64, 32, 32), scale=-1 // 8, rank=1350, tier=:full),
    (label="l3aniso", cells=(96, 32, 32), scale=-1 // 8, rank=800, tier=:full),
    (label="l4aniso", cells=(128, 32, 32), scale=-1 // 8, rank=600, tier=:full),
    (label="l2iso", cells=(64, 64, 64), scale=1 // 32, rank=600, tier=:full),
]

"Separations in wavelengths. Zero is contact; the last one checks the far field."
const SEPARATIONS = [0 // 1, 1 // 32, 8 // 32, 1 // 1, 1000 // 1]
const QUICK_SEPARATIONS = [0 // 1, 8 // 32, 1000 // 1]

#=
Thread counts for the scan, filtered to the cluster's core count. This has to
reach as far as the production jobs actually go, or the fitted efficiency
`eta(T) = 1 + s(T - 1)` gets extrapolated instead of interpolated -- and a linear
efficiency model does not survive that. On fir and narval the filter leaves
[1, 2, 4, 8], which is exactly the range `choose_cores` picks from. On molering
production runs with `-t auto`, i.e. every core, so the scan has to go all the way
up.
=#
const THREAD_SCAN = [1, 2, 4, 8, 16, 32, 64, 128]

"Sketch widths for the dense points, spanning the ranks actually in use."
const DENSE_WIDTHS = [128, 512, 1400, 2800]

const DEFAULT_CHI = "13.6+0.05im"
const DEFAULT_OVERSAMPLES = 50
const DEFAULT_POWER_ITERS = 14

# --------------------------------------------------------------------------- #
# Points
# --------------------------------------------------------------------------- #

"""
    PlannedPoint

One invocation of `bench/point.jl`, with the resources to ask for. `args` are
passed through verbatim, `threads` becomes both `--cpus-per-task` and Julia's
`-t`, and `gpu` says whether to request a whole GPU.
"""
struct PlannedPoint
    label::String
    kind::String
    args::Vector{String}
    threads::Int
    host_GB::Int
    time_s::Int
    gpu::Bool
    depends_on::Union{Nothing,String}   # label of a point that must finish first
    #=
    The three fields below exist for the `funicular` tier and are inert elsewhere.
    Calibration proper always takes a whole GPU, since the point there is to
    measure primitives on undivided hardware. The Funicular trials ask the
    opposite kind of question, whether a *particular* allocation holds a
    particular job, so the slice is part of the trial and cannot be defaulted.
    =#
    gpu_request::Union{Nothing,String}  # `--gpus` value; nothing means the whole GPU
    predicted_s::Float64                # cost model's unpadded wall time, 0 if unknown
    bill_fraction::Float64              # GPU-equivalents this allocation is billed as
end

PlannedPoint(label, kind, args, threads, host_GB, time_s, gpu) =
    PlannedPoint(label, kind, args, threads, host_GB, time_s, gpu, nothing, nothing, 0.0, 0.0)
PlannedPoint(label, kind, args, threads, host_GB, time_s, gpu, depends_on) =
    PlannedPoint(label, kind, args, threads, host_GB, time_s, gpu, depends_on, nothing, 0.0, 0.0)

"""
    GPU_ALLOCATIONS

Per cluster, the GPU allocations a planned point may name, as
`name => (vram_GB, fraction, bundle_host_GB)`. `fraction` is the share of the
card's streaming multiprocessors, which is both what stretches a slice's
device-bound time and what the allocation is billed at. `bundle_host_GB` is the
system RAM that comes with it, and asking for more is billed as though several
slices had been taken.

The same numbers as `choose_gpu`'s table in `create_jobs.jl`, and kept in step
with it, but not shared: `create_jobs.jl` is not includable from here without
dragging the whole package in.
"""
const GPU_ALLOCATIONS = Dict(
    "narval" => Dict("a100_1g.5gb" => (vram_GB=5, fraction=1 / 8, bundle_host_GB=17),
                     "a100_2g.10gb" => (vram_GB=10, fraction=2 / 8, bundle_host_GB=35),
                     "a100_3g.20gb" => (vram_GB=20, fraction=3 / 8, bundle_host_GB=62),
                     "a100" => (vram_GB=40, fraction=1.0, bundle_host_GB=124)),
)

function gpu_allocation(cluster::ClusterSpec, name::AbstractString)
    table = get(GPU_ALLOCATIONS, cluster.name, nothing)
    table === nothing && error("no GPU allocation table for cluster '$(cluster.name)'")
    haskey(table, name) ||
        error("'$name' is not a $(cluster.name) GPU allocation. Known: $(join(sort(collect(keys(table))), ", "))")
    return table[name]
end

cells_arg(cells::NTuple{3,Int}) = join(cells, ",")
rat(r::Rational{Int}) = "$(numerator(r))//$(denominator(r))"

# `refine_gap = false` because the jobs this tier emits do not pass `--refine`,
# and `bench/point.jl` leaves it off. `SEPARATIONS` reaches down to one cell, where
# the production default would refine, so the point has to say which mesh it is
# predicting rather than inherit a default the job does not share.
as_srpoint(body, separation::Rational{Int}, threads::Int) =
    SRPoint(body.cells, body.cells;
            scale=body.scale < 0 ? (1 // 32, abs(body.scale), abs(body.scale)) :
                  (body.scale, body.scale, body.scale),
            separation=separation, rank=body.rank, oversamples=DEFAULT_OVERSAMPLES,
            power_iters=DEFAULT_POWER_ITERS, threads=threads, refine_gap=false)

"Resources for a Green-block point: three times the analytic peak, floored at 8 GB."
function block_resources(cluster::ClusterSpec, body, separation::Rational{Int}, threads::Int)
    pt = as_srpoint(body, separation, threads)
    counts = greens_counts(pt)
    host_GB = min(cluster.max_host_GB, max(8, ceil(Int, 3 * counts.peak_bytes / 2^30) + 4))
    # Uncalibrated predictions can be badly wrong in either direction, so give
    # calibration jobs a very loose time box: they are cheap and being killed
    # mid-measurement wastes the whole point.
    time_s = clamp(ceil(Int, 10 * predict(GenerateGreens, pt, coefficients_for(cluster.name);
                                          pad=false).time_s), 3600, 12 * 3600)
    return host_GB, time_s
end

# --------------------------------------------------------------------------- #
# The `funicular` tier: trials E1-E4 of FUNICULAR_PLAN.md, workstream E
# --------------------------------------------------------------------------- #

#=
This tier is not calibration in the sense the other four are. Those measure
primitives on undivided hardware and let `create_jobs.jl` derate for a slice.
These ask whether particular allocations hold particular jobs, so the allocation
is part of the trial and every point names its own.

Every trial is one separation. Separation does not affect cost except at contact
(see the model's claims in bench/README.md), so one mid-sweep gap is as
informative as thirty-three.

Two conventions the trials depend on:

  * `--fresh` on every RSVD point. `_save_ur_asym` skips a complete output, and a
    trial that skipped its own work would report a startup time as a wall time.
  * a private `--scratch` per RSVD point, so that the parity pair's two JLDs
    survive side by side and each E4 bounds point reads the E3 output it is meant
    to read rather than whichever ran last.
=#

const FUNICULAR_SEPARATION = 16 // 32
const FUNICULAR_SCALE = 1 // 32
const FUNICULAR_OVERSAMPLES = 50
const FUNICULAR_POWER_ITERS = 14
const FUNICULAR_K_PARITY = 1350     # the historical rank, for E2 and E4's back-comparison
const FUNICULAR_K_PRODUCTION = 4000 # the rank the sweep is going to
#=
One seed for every trial. The panel path regenerates its Gaussian test matrix from
it, so the E3 runs are reproducible and re-runnable. The in-memory path draws from
the global RNG and ignores the seed, which is why E2's parity is to RSVD accuracy
rather than bit-for-bit.
=#
const FUNICULAR_SEED = 20260814
#=
`--mem` for trial E3c, chosen so that the spill happens through production code
rather than through a testing hook. `residency_plan` takes `SLURM_MEM_PER_NODE`
and subtracts `HOST_OVERHEAD_RESERVE_BYTES` (6 GiB), so 66 GiB requested is a
60 GiB host budget exactly. The 4 lambda panel peak is ~95 GiB, so the tier below
has to take the difference and the NVMe path gets exercised end to end.
=#
const FUNICULAR_SPILL_MEM_GB = 66

const FUNICULAR_BODIES = (
    l1=(label="l1", cells=(32, 32, 32)),    # 1 lambda, N_u = 196,608
    l2=(label="l2", cells=(64, 32, 32)),    # 2 lambda, N_u = 393,216
    l4=(label="l4", cells=(128, 32, 32)),   # 4 lambda, N_u = 786,432
)

funicular_srpoint(body, rank::Int; threads::Int=4, fresh_preload::Bool=true) =
    SRPoint(body.cells, body.cells; scale=FUNICULAR_SCALE,
            separation=FUNICULAR_SEPARATION, rank=rank,
            oversamples=FUNICULAR_OVERSAMPLES, power_iters=FUNICULAR_POWER_ITERS,
            threads=threads, fresh_preload=fresh_preload)

funicular_common(body, rank::Int) =
    ["--cells", cells_arg(body.cells), "--scale", rat(FUNICULAR_SCALE),
     "--chi", DEFAULT_CHI, "--sep", rat(FUNICULAR_SEPARATION),
     "--rank", string(rank), "--oversamples", string(FUNICULAR_OVERSAMPLES),
     "--power-iters", string(FUNICULAR_POWER_ITERS),
     "--seed", string(FUNICULAR_SEED)]

"""
    funicular_billing(cluster, alloc, host_GB, threads) -> Float64

GPU-equivalents this allocation is billed at: the largest of its share of the
card, its share of the per-GPU RAM bundle, and its share of the core bundle. A
slice's `bundle_host_GB` is by construction its share of the card, so it stands in
for the GPU term.
"""
funicular_billing(cluster::ClusterSpec, alloc, host_GB::Real, threads::Int) =
    max(alloc.bundle_host_GB / cluster.max_host_GB,
        host_GB / cluster.max_host_GB,
        threads / cluster.max_cores)

"""
    funicular_gpu_point(...) -> PlannedPoint

One GPU trial, sized from the cost model on the allocation it names.

`force_panel` passes `vram_capacity_bytes = 0` to `predict`, which makes
`rsvd_mode` return `:panel` whatever the card is. That is the sizing counterpart
of `bench/point.jl`'s `--force-path panel`. The trial is going to run the panel
code on that allocation, so it has to be *sized* for the panel code; asking the
model what the predicate would have chosen would give the in-memory answer for
the very runs where the predicate is being overridden.

Wall time stretches only the device-bound share, by the slice's SM fraction. The
bus term and the writes do not get slower on a slice, and stretching them would
invent time in the wrong direction, since a slower slice hides *more* of a
transfer behind compute, not less.
"""
function funicular_gpu_point(cluster::ClusterSpec, label::AbstractString,
                             kind::AbstractString, pt::SRPoint, args::Vector{String},
                             alloc_name::AbstractString; job::JobKind,
                             force_panel::Bool=false, depends_on=nothing,
                             host_GB::Union{Nothing,Int}=nothing,
                             time_factor::Real=2.5, threads::Int=4)
    coeffs = coefficients_for(cluster.name)
    alloc = gpu_allocation(cluster, alloc_name)
    capacity = force_panel ? 0.0 : alloc.vram_GB * 2^30
    p = predict(job, pt, coeffs; pad=false, vram_capacity_bytes=capacity)
    wall = (p.time_s - p.device_time_s) + p.device_time_s / alloc.fraction
    time_s = clamp(ceil(Int, time_factor * wall), 3600, 24 * 3600)
    if time_s < time_factor * wall
        # 24 h is a queue-behaviour ceiling, not a model output. Say when a point
        # is sitting on it, because the margin over the prediction is then whatever
        # the ceiling leaves rather than the factor asked for.
        @info "$label is limit-bound at the 24 h ceiling" predicted_h=wall / 3600 margin=time_s / wall
    end
    threads = min(threads, cluster.max_cores)
    hg = host_GB === nothing ?
         clamp(ceil(Int, p.host_bytes / 2^30) + 8, 16, alloc.bundle_host_GB) :
         host_GB
    if hg > alloc.bundle_host_GB
        @warn "$label asks for more RAM than $(alloc_name)'s bundle; it will be billed as more than one slice" host_GB=hg bundle=alloc.bundle_host_GB
    end
    if p.vram_floor_bytes > alloc.vram_GB * 2^30
        @warn "$label's predicted device floor does not fit $(alloc_name)" floor_GB=p.vram_floor_bytes / 2^30 capacity_GB=alloc.vram_GB
    end
    return PlannedPoint(label, kind, args, threads, hg, time_s, true, depends_on,
                        alloc_name, wall, funicular_billing(cluster, alloc, hg, threads))
end

function plan_funicular_points(cluster::ClusterSpec)
    cluster.name == "narval" ||
        error("the funicular tier is narval-only for now (workstream E); got '$(cluster.name)'")
    coeffs = coefficients_for(cluster.name)
    threads = min(4, cluster.max_cores)
    points = PlannedPoint[]
    scratch(name) = ["--scratch", "\$CAL_ROOT/funicular/$(name)"]

    # ---- E1: the host link ------------------------------------------------
    #=
    A whole card, because the question is what the link does when nothing else is
    on it, and because Funicular's own `overlap.jl` allocates a 2 GiB device
    budget of its own on top of this process's pinned sweep.
    =#
    push!(points, PlannedPoint("e1_panelbus", "panel_bus",
                               vcat(scratch("e1_panelbus"), ["--reps", "5"]),
                               threads, 32, 3600, true, nothing, "a100", 0.0,
                               funicular_billing(cluster, gpu_allocation(cluster, "a100"),
                                                 32, threads)))

    # ---- Green functions, one per geometry --------------------------------
    #=
    CPU jobs, and dependencies of everything below: no RSVD can start until the
    (R, S) and (R, R) blocks for its geometry are in the shared preload directory.
    They are separate jobs rather than a prologue inside each GPU job so that the
    three geometries build concurrently and nothing burns GPU time on a host
    quadrature loop.
    =#
    greens_label = Dict{Symbol,String}()
    for (key, body) in pairs(FUNICULAR_BODIES)
        pt = funicular_srpoint(body, FUNICULAR_K_PRODUCTION; threads=threads)
        hg, ts = block_resources(cluster, (cells=body.cells, scale=FUNICULAR_SCALE,
                                           rank=FUNICULAR_K_PRODUCTION),
                                 FUNICULAR_SEPARATION, threads)
        label = "fungreens_$(body.label)"
        greens_label[key] = label
        # No `--scratch`: `stage_greens` writes into the shared preload directory
        # under `--root`, which is exactly what every RSVD point below then reads.
        # `bill_fraction` is zero because a CPU job is billed against a CPU
        # allocation; it costs core-hours, not GPU-equivalents.
        push!(points, PlannedPoint(label, "stage_greens",
                                   funicular_common(body, FUNICULAR_K_PRODUCTION),
                                   threads, hg, ts, false, nothing, nothing,
                                   predict(GenerateGreens, pt, coeffs; pad=false).time_s,
                                   0.0))
    end

    # ---- E2: parity, in-memory against panel, on the same card ------------
    l1 = FUNICULAR_BODIES.l1
    parity_pt = funicular_srpoint(l1, FUNICULAR_K_PARITY; threads=threads)
    for (tag, force) in (("inmem", false), ("panel", true))
        args = vcat(funicular_common(l1, FUNICULAR_K_PARITY),
                    scratch("e2_l1_$(tag)"), ["--fresh"],
                    ["--force-path", force ? "panel" : "auto"])
        push!(points, funicular_gpu_point(cluster, "e2_l1_$(tag)", "stage_rsvd",
                                          parity_pt, args, "a100";
                                          job=GenerateRSVD, force_panel=force,
                                          depends_on=greens_label[:l1], threads=threads))
    end

    # ---- E3: the panel path at the production rank ------------------------
    #=
    E3d is not in the plan's table. E4 wants bounds at k = 4000 for 1 lambda and
    the table's E3 rows only produce 2 and 4 lambda outputs, so the 1 lambda
    k = 4000 RSVD has to exist somewhere; here is the cheapest place. At that rank
    the predicate chooses the panel path on its own (the in-memory sketch is 38 GB
    against a 3g.20gb slice's 20), so nothing is forced.
    =#
    l1_prod = funicular_srpoint(l1, FUNICULAR_K_PRODUCTION; threads=threads)
    push!(points, funicular_gpu_point(cluster, "e3d_l1_k4000", "stage_rsvd", l1_prod,
                                      vcat(funicular_common(l1, FUNICULAR_K_PRODUCTION),
                                           scratch("e3d_l1_k4000"), ["--fresh"],
                                           ["--force-path", "panel"]),
                                      "a100_3g.20gb"; job=GenerateRSVD, force_panel=true,
                                      depends_on=greens_label[:l1], threads=threads))

    # E3a: the same 2 lambda job twice, once on the tight bundle and once on the
    # whole card. Same `--mem` on both, so the only variable is the GPU.
    l2 = FUNICULAR_BODIES.l2
    l2_pt = funicular_srpoint(l2, FUNICULAR_K_PRODUCTION; threads=threads)
    for (tag, alloc) in (("slice", "a100_3g.20gb"), ("full", "a100"))
        push!(points, funicular_gpu_point(cluster, "e3a_l2_$(tag)", "stage_rsvd", l2_pt,
                                          vcat(funicular_common(l2, FUNICULAR_K_PRODUCTION),
                                               scratch("e3a_l2_$(tag)"), ["--fresh"],
                                               ["--force-path", "panel"]),
                                          alloc; job=GenerateRSVD, force_panel=true,
                                          depends_on=greens_label[:l2],
                                          host_GB=60, threads=threads))
    end

    # E3b: 4 lambda with room to spare, which is the run that measures the 102 GB
    # peak against the 124.5 GB bundle. E3c: the same job on a 60 GiB host budget,
    # which is under the ~95 GiB the two panel matrices want, so Funicular has to
    # spill to node-local NVMe.
    l4 = FUNICULAR_BODIES.l4
    l4_pt = funicular_srpoint(l4, FUNICULAR_K_PRODUCTION; threads=threads)
    push!(points, funicular_gpu_point(cluster, "e3b_l4_full", "stage_rsvd", l4_pt,
                                      vcat(funicular_common(l4, FUNICULAR_K_PRODUCTION),
                                           scratch("e3b_l4_full"), ["--fresh"],
                                           ["--force-path", "panel"]),
                                      "a100"; job=GenerateRSVD, force_panel=true,
                                      depends_on=greens_label[:l4],
                                      host_GB=118, threads=threads))
    push!(points, funicular_gpu_point(cluster, "e3c_l4_spill", "stage_rsvd", l4_pt,
                                      vcat(funicular_common(l4, FUNICULAR_K_PRODUCTION),
                                           scratch("e3c_l4_spill"), ["--fresh"],
                                           ["--force-path", "panel"]),
                                      "a100"; job=GenerateRSVD, force_panel=true,
                                      depends_on=greens_label[:l4],
                                      host_GB=FUNICULAR_SPILL_MEM_GB,
                                      # The model has no NVMe term, so its estimate
                                      # for this run is the no-spill one. Give the
                                      # limit room for the round trip through disk.
                                      time_factor=4.0, threads=threads))

    # ---- E4: bounds on the E3 outputs -------------------------------------
    #=
    Each bounds point reads the scratch directory its RSVD point wrote and depends
    on it with `afterok`. The k = 1350 run reads the *panel* half of the parity
    pair on purpose: it is the one that compares the panelized front-end against
    the historical numbers in `data analysis/data`, so it has to be reading a
    panel-path basis.
    =#
    for (label, body, rank, alloc, src, dep) in
        (("e4_bounds_l1_k4000", l1, FUNICULAR_K_PRODUCTION, "a100_3g.20gb",
          "e3d_l1_k4000", "e3d_l1_k4000"),
         ("e4_bounds_l4_k4000", l4, FUNICULAR_K_PRODUCTION, "a100",
          "e3b_l4_full", "e3b_l4_full"),
         ("e4_bounds_l1_k1350", l1, FUNICULAR_K_PARITY, "a100_3g.20gb",
          "e2_l1_panel", "e2_l1_panel"))
        pt = funicular_srpoint(body, rank; threads=threads)
        push!(points, funicular_gpu_point(cluster, label, "stage_bounds", pt,
                                          vcat(funicular_common(body, rank), scratch(src)),
                                          alloc; job=ComputeBounds, force_panel=true,
                                          depends_on=dep, threads=threads))
    end

    return points
end

# --------------------------------------------------------------------------- #
# The `backfill` tier: refit the bounds and RSVD costs with nothing over 3 hours
# --------------------------------------------------------------------------- #

#=
Why this tier exists.

Two things about the pipeline changed after the narval coefficients were fitted, and
both of them changed a *count* rather than a rate, which is exactly the kind of
error a fitted rate cannot absorb:

  * `--gamma-rtol` (`load_bounds_inputs` in src/bounds.jl) cuts the positive
    `Asym(G0_ur)` block down to the directions above the RSVD's noise floor. At a far
    separation that is a few dozen columns where the model still charges
    `NUM_POS_FRACTION * rank = 2400`, and the bounds cost is superlinear in `m`.
  * the windowed tau grid sweep and the refinement pencil cache (`bounds_from_spectrum`)
    cut the per-index tau work from `TAU_GRID_POINTS + TAU_REFINE_EVALS = 11`
    evaluations and six fresh `m x m` whitenings down to about three evaluations and,
    on a tau* plateau, near zero new whitenings.

Together they are why a 1 lambda bounds job is requested at 18 h. Nothing in that
request will ever start: narval's backfill window is what a low-priority job gets,
and an 18 h ask does not fit one. So every job here is capped at three hours, most
of them well under, and the tier's whole design follows from that:

  * the bounds points *sample* the outer loop (`--outer-blocks`) instead of running
    it. The front end is measured in full either way, the per-index costs come back
    individually in `outer_times`, and a few percent of the loop identifies the same
    coefficients the whole loop would.
  * the RSVD points run at low `q`. RSVD cost is affine in `q` -- `(2q + 2)`-ish
    operator passes -- so two low-`q` runs give the per-pass slope, which extrapolates
    to any `q`, and a third checks that the line is straight. `rsvd_time_parts` in
    bench/cost_model.jl is the model side of the same split.
  * the bounds points reuse RSVD output that is *already on scratch* from the
    cancelled 1 lambda sweep, so there is no RSVD to pay for at production rank.
    Which separations survived is not knowable from here, so
    `bench/pick_bounds_points.jl` decides on the login node at submit time and sizes
    each job from the `m` it reads out of that separation's own spectrum.
=#

const BACKFILL_MAX_TIME_S = 3 * 3600
const BACKFILL_CHI = "4.25+0.0342557im"   # Ge, zeta = 1000: the arxivV3 production chi
const BACKFILL_SCALE = 1 // 32
const BACKFILL_RANK = 4000
const BACKFILL_OVERSAMPLES = 50
const BACKFILL_GAMMA_RTOL = "1.0e-12"
const BACKFILL_SEED = 20260819
const BACKFILL_PICKS = 4
const BACKFILL_OUTER_BLOCKS = 4
const BACKFILL_OUTER_BLOCK_LEN = 24
#=
`--design rs`, everywhere in this tier, and this is load-bearing. `src/common.jl`
sorts the letters of `--design`, so every production sweep writes
`<cells>__<cells>__<n>ss<d>__RS`; `bench/point.jl` historically built
`[Sender, Receiver]` and so looked for `__SR`. Same geometry, different filename, and
a bounds point that spells it the old way finds none of the outputs it came to read.
=#
const BACKFILL_DESIGN = "rs"
#=
The power-iteration counts. Three, not two: two identify the slope and the third says
whether the line through them is straight, which is the assumption the whole
extrapolation to q = 14 rests on. Low enough that even the 1 lambda panel path
finishes inside an hour.
=#
const BACKFILL_QS = (1, 3, 5)

#=
The production sweeps this tier reads and reuses. The 1 lambda sweep is the one that
ran; its scratch holds the RSVD outputs the bounds points sample, and its preload
directory holds the Green blocks those separations needed. Both are the *defaults*
production used (`_default_preload_dir()` for the preload), written out here so the
generated launcher records what it was pointed at.
=#
const BACKFILL_PROD_PROJECT_1L =
    "narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale"
backfill_prod_scratch(project::AbstractString) =
    "/home/$(CC_UNAME)/scratch/Photonic-System-Channels/$(project)/"
const BACKFILL_PROD_PRELOAD = "/home/$(CC_UNAME)/scratch/preload/"

const BACKFILL_BODIES = (
    l0p5=(label="0p5", cells=(16, 16, 16), alloc="a100_3g.20gb"), # N_u =  24,576, in-memory
    l1=(label="1l", cells=(32, 32, 32), alloc="a100"),            # N_u = 196,608, panel
)

backfill_srpoint(cells::NTuple{3,Int}, q::Int; threads::Int=2) =
    SRPoint(cells, cells; scale=BACKFILL_SCALE, separation=BACKFILL_SEPARATION,
            rank=BACKFILL_RANK, oversamples=BACKFILL_OVERSAMPLES, power_iters=q,
            threads=threads)

#=
The separation the RSVD points are *sized* at. Only contact changes an RSVD's cost,
so any non-contact gap gives the same answer; the launcher then runs them at
`$SEP_1L`, the nearest separation the picker found, because that is a separation
whose Green blocks are guaranteed to be in the production preload already -- the
production RSVD at that separation is what produced the basis the bounds points read.
=#
const BACKFILL_SEPARATION = 1 // 2

backfill_common(cells::NTuple{3,Int}, q::Int) =
    ["--cells", cells_arg(cells), "--scale", rat(BACKFILL_SCALE),
     "--chi", BACKFILL_CHI, "--design", BACKFILL_DESIGN,
     "--rank", string(BACKFILL_RANK), "--oversamples", string(BACKFILL_OVERSAMPLES),
     "--power-iters", string(q), "--seed", string(BACKFILL_SEED)]

"""
    backfill_gpu_point(...) -> PlannedPoint

One GPU point of the backfill tier, sized from the cost model on the allocation it
names and then **hard-capped at three hours**.

`funicular_gpu_point` does the same arithmetic against a 24 h ceiling. This one caps
at `BACKFILL_MAX_TIME_S` and shouts if the capped limit is under the padded
prediction, because a point that has to be truncated to fit the backfill window is a
point that needs its `q` or its sample size reduced, not one that should be submitted
and hoped for.
"""
function backfill_gpu_point(cluster::ClusterSpec, label::AbstractString,
                            kind::AbstractString, pt::SRPoint, args::Vector{String},
                            alloc_name::AbstractString; job::JobKind,
                            depends_on=nothing, host_GB::Union{Nothing,Int}=nothing,
                            time_factor::Real=2.5, threads::Int=2)
    coeffs = coefficients_for(cluster.name)
    alloc = gpu_allocation(cluster, alloc_name)
    capacity = alloc.vram_GB * 2^30
    p = predict(job, pt, coeffs; pad=false, vram_capacity_bytes=capacity)
    wall = (p.time_s - p.device_time_s) + p.device_time_s / alloc.fraction
    time_s = clamp(ceil(Int, time_factor * wall), 1800, BACKFILL_MAX_TIME_S)
    if time_s < time_factor * wall
        @warn "$label had to be capped at the 3 h backfill ceiling; it wants more than $(time_factor)x its prediction" predicted_h = wall / 3600 requested_h = time_s / 3600
    end
    threads = min(threads, cluster.max_cores)
    hg = host_GB === nothing ?
         clamp(ceil(Int, p.host_bytes / 2^30) + 4, 8, alloc.bundle_host_GB) : host_GB
    if p.vram_floor_bytes > capacity
        @warn "$label's predicted device floor does not fit $(alloc_name)" floor_GB = p.vram_floor_bytes / 2^30 capacity_GB = alloc.vram_GB
    end
    return PlannedPoint(label, kind, args, threads, hg, time_s, true, depends_on,
                        alloc_name, wall, funicular_billing(cluster, alloc, hg, threads))
end

"""
    plan_backfill_points(cluster) -> Vector{PlannedPoint}

The seven statically-known points: one Green-function job for the 1/2 lambda
geometry, and the six `stage_rsvd` points that measure the per-pass rate at two sizes
and three power-iteration counts.

The four `stage_bounds` points are *not* here. Their separations are chosen on the
login node by `bench/pick_bounds_points.jl`, which also chooses each one's allocation
and time limit from the `m` it reads out of that separation's spectrum, so they are
emitted by `backfill_bounds_block` as a shell loop over the picker's output rather
than as `PlannedPoint`s with fixed arguments. `write_manifest` lists them as template
rows so the manifest is still a complete account of what gets submitted.
"""
function plan_backfill_points(cluster::ClusterSpec)
    cluster.name == "narval" ||
        error("the backfill tier is narval-only (it reuses narval scratch); got '$(cluster.name)'")
    coeffs = coefficients_for(cluster.name)
    threads = min(2, cluster.max_cores)
    points = PlannedPoint[]

    #=
    Green functions for 1/2 lambda. This is the one geometry whose blocks are not
    already on narval: the 0p5 sweep was generated but never submitted, so its
    preload entries do not exist, while the 1 lambda sweep ran and left its own
    behind. A CPU job, and a dependency of the three 0p5 RSVD points.

    It writes into a preload directory of its own, under the calibration root, and not
    into the production one. Two reasons, and the second is the load-bearing one:

      * a calibration job has no business writing into the tree the production sweeps
        read, and
      * `point_stage_greens` measures what it wrote by walking the whole preload
        directory twice with `_preload_bytes`. Pointed at the production preload --
        which holds every block of the 1 lambda sweep -- that is two full `walkdir`s
        over hundreds of gigabytes on a shared filesystem, before and after, to
        measure a few hundred megabytes. Pointed at its own directory it is instant.

    The three 0p5 RSVD points read the same private directory, so the halves stay
    consistent; only the 1 lambda points, which read blocks they did not build, look
    at the production preload.
    =#
    body = BACKFILL_BODIES.l0p5
    greens_pt = backfill_srpoint(body.cells, 14; threads=4)
    greens_time = clamp(ceil(Int, 4 * predict(GenerateGreens, greens_pt, coeffs;
                                             pad=true).time_s),
                        1800, BACKFILL_MAX_TIME_S)
    push!(points, PlannedPoint("f_greens_0p5", "stage_greens",
                               vcat(backfill_common(body.cells, 14),
                                    ["--preload", "\$PRELOAD_0P5",
                                     "--sep", "\$SEP_RSVD"]),
                               min(4, cluster.max_cores), 12, greens_time, false,
                               nothing, nothing,
                               predict(GenerateGreens, greens_pt, coeffs; pad=false).time_s,
                               0.0))

    #=
    The RSVD points. `--fresh` on every one of them, and a private `--scratch` per
    (geometry, q): `file_prefix` encodes cells, separation and the universe string and
    nothing else, so three runs at three `q` would otherwise write the same file, and
    `_save_ur_asym` would skip the second and third as already done. That is also why
    they must not be pointed at a production scratch directory -- a `--fresh` run
    there would delete a production basis and replace it with a `q = 1` one.

    No `--force-path`: the point of measuring at production sizes is to measure the
    path the predicate really chooses, which is in-memory at 1/2 lambda and Funicular
    panels at 1 lambda.
    =#
    for (key, b) in pairs(BACKFILL_BODIES)
        for q in BACKFILL_QS
            label = "f_rsvd_$(b.label)_q$(q)"
            args = vcat(backfill_common(b.cells, q),
                        ["--scratch", "\$CAL_ROOT/backfill/rsvd_$(b.label)_q$(q)",
                         "--preload", key === :l0p5 ? "\$PRELOAD_0P5" : "\$PROD_PRELOAD",
                         "--fresh", "--sep", "\$SEP_RSVD"])
            push!(points, backfill_gpu_point(cluster, label, "stage_rsvd",
                                             backfill_srpoint(b.cells, q; threads=threads),
                                             args, b.alloc; job=GenerateRSVD,
                                             depends_on=key === :l0p5 ? "f_greens_0p5" : nothing,
                                             threads=threads))
        end
    end
    return points
end

"""
    backfill_pick_block(cluster) -> String

The login-node step that runs before any `sbatch`: choose the bounds separations from
what is actually on scratch, and choose the separation the RSVD points run at.

Idempotent. The pick file is written once and reused, so re-running the script after a
partial submission submits the *same* four separations rather than re-deciding, which
matters because the row files are named after the pick index. Delete the pick file (or
pass `--pick`) to choose again.

`SEP_RSVD` is the nearest pick. Separation does not change an RSVD's cost away from
contact, so the choice is free, and taking it from the pick list guarantees that the
Green blocks for it are already in the production preload -- the production RSVD at
that separation is what built them.
"""
function backfill_pick_block(cluster::ClusterSpec)
    prod_scratch = backfill_prod_scratch(BACKFILL_PROD_PROJECT_1L)
    picker = "julia --project=. bench/pick_bounds_points.jl " *
             "--scratch \"\$PROD_SCRATCH\" --cells $(cells_arg(BACKFILL_BODIES.l1.cells)) " *
             "--design $(BACKFILL_DESIGN) --gamma-rtol $(BACKFILL_GAMMA_RTOL) " *
             "--picks $(BACKFILL_PICKS) --out \"\$PICKS\" --table \"\$KEPT_TABLE\""
    return """

    PROD_SCRATCH=$(prod_scratch)
    PROD_PRELOAD=$(BACKFILL_PROD_PRELOAD)
    PRELOAD_0P5=\$CAL_ROOT/backfill/preload_$(BACKFILL_BODIES.l0p5.label)
    PICKS=\$CAL_ROOT/backfill/picked_$(BACKFILL_BODIES.l1.label).txt
    KEPT_TABLE=\$CAL_ROOT/backfill/kept_by_sep_$(BACKFILL_BODIES.l1.label).csv
    mkdir -p \$CAL_ROOT/backfill \$PRELOAD_0P5

    if [ ! -d "\$PROD_SCRATCH" ]; then
        echo "The 1 lambda sweep's scratch directory does not exist:"
        echo "  \$PROD_SCRATCH"
        echo "The bounds points have nothing to read. Either the sweep never wrote"
        echo "there or scratch was purged; check the path against"
        echo "jobs/launch_$(BACKFILL_PROD_PROJECT_1L).sh and try again."
        exit 1
    fi

    if [ "\${1:-}" = "--pick" ]; then
        rm -f "\$PICKS"
        $(picker)
        echo
        echo "Picked separations are in \$PICKS and the full table in \$KEPT_TABLE."
        echo "Re-run this script without --pick to submit."
        exit 0
    fi

    if [ ! -s "\$PICKS" ]; then
        echo "Choosing bounds separations from the RSVD outputs on scratch..."
        $(picker) || { echo "The picker failed; nothing was submitted."; exit 1; }
    else
        echo "Reusing the existing pick list \$PICKS (delete it, or pass --pick, to choose again):"
    fi
    cat "\$PICKS"
    echo

    # The RSVD points run at the nearest pick, whose Green blocks are already in the
    # preload directory because the production RSVD at that separation built them.
    SEP_RSVD=\$(head -n 1 "\$PICKS" | awk '{print \$1}')
    if [ -z "\$SEP_RSVD" ]; then
        echo "The pick list is empty; nothing was submitted."
        exit 1
    fi
    echo "RSVD points will run at separation \$SEP_RSVD"
    echo
    """
end

"""
    backfill_bounds_block(cluster) -> String

The shell that submits the four `stage_bounds` points: read the picker's output,
one `sbatch` per line, each with the allocation, memory, time limit and outer-loop
mode that line names.

Reads the pick file on file descriptor 3. `sbatch` here has its script on stdin
through a heredoc, so it would not eat the loop's input anyway, but a `while read`
loop around a command that *might* read stdin is a bug waiting for the next edit.

The heredoc is deliberately unquoted so that `\$SEP`, `\$BLOCKS` and `\$LABEL` are
substituted at submission time -- the batch script the scheduler stores has to carry
the literal separation, since the pick file will not be consulted again. `\$(date +%s)`
is escaped for the opposite reason: it has to run inside the job. There are no
backticks anywhere in this file for the same reason: an unquoted heredoc executes
them at submission.
"""
function backfill_bounds_block(cluster::ClusterSpec)
    common = join(vcat(["--kind", "stage_bounds"],
                       backfill_common(BACKFILL_BODIES.l1.cells, DEFAULT_POWER_ITERS),
                       ["--gamma-rtol", BACKFILL_GAMMA_RTOL,
                        "--outer-block-len", string(BACKFILL_OUTER_BLOCK_LEN)]), " ")
    return """
    # ---------------------------------------------------------------------------
    # A: bounds on the RSVD outputs already sitting in the 1 lambda sweep's scratch.
    #
    # One job per pick. Each reads PROD_SCRATCH (never writes to it: a sampled run
    # writes no output JLD at all, and a full one writes only into --project, which
    # points at the calibration tree) and asks for exactly what its own m needs.
    #
    # A "full" pick runs the whole outer loop, production exactly, output JLD and all.
    # A "sampled" pick runs $(BACKFILL_OUTER_BLOCKS) blocks of $(BACKFILL_OUTER_BLOCK_LEN) consecutive
    # indices spread over 1:m, a few percent of the loop, and identifies the same
    # coefficients. Which one a pick gets is decided by its m, by the picker.
    # ---------------------------------------------------------------------------
    pick_index=0
    while read -r SEP KEPT STORED GPU MEM MODE LIMIT <&3; do
        [ -n "\$SEP" ] || continue
        pick_index=\$((pick_index + 1))
        LABEL=f_bounds_1l_p\${pick_index}
        if [ "\$MODE" = "full" ]; then BLOCKS=0; else BLOCKS=$(BACKFILL_OUTER_BLOCKS); fi
        jid=\$(sbatch --parsable \\
            --job-name=psccal_\${LABEL} \\
            --output=\$CAL_ROOT/logs/\${LABEL}_%j.out \\
            --account=$(cluster.account) \\
            --time=\$LIMIT \\
            --cpus-per-task=2 \\
            --mem=\${MEM}G \\
            --gpus=\${GPU}:1 \\
            --chdir=\$CODE_DIR \\
            --export=ALL \\
            <<EOF
    #!/bin/bash
    $(cluster.modules)
    export PSC_T0=\\\$(date +%s)
    srun julia --project=. -t 2 bench/point.jl $(common) \\
        --sep '\$SEP' --outer-blocks \$BLOCKS \\
        --scratch "\$PROD_SCRATCH" --preload "\$PROD_PRELOAD" \\
        --gpu 0 --root \$CAL_ROOT --out \$ROWS/\${LABEL}.csv --cluster $(cluster.name) \\
        --note 'tier=backfill;label=\${LABEL};pick=\${pick_index};picked_m=\${KEPT};picked_stored=\${STORED};mode=\${MODE}'
    EOF
    )
        echo "  \${LABEL}  sep=\${SEP}  m=\${KEPT}  \${GPU}  \${MEM}G  \${LIMIT}  \${MODE}  -> job \${jid}"
        sleep 0.05
    done 3< \$PICKS
    """
end

"""
    backfill_preamble(cluster)

What has to be true before this script is submitted, as comments. Three assumptions,
all of them about the state of narval's filesystems rather than about the code, and
all three cheap to check by hand.
"""
function backfill_preamble(cluster::ClusterSpec)
    return """
    # ---------------------------------------------------------------------------
    # Every job in this tier asks for at most 03:00:00, so that all of them are
    # eligible for narval's backfill window. Nothing here needs a reservation and
    # nothing here should ever sit in the queue behind an 18 h request.
    #
    # Three things this script assumes about narval. It checks the first itself and
    # fails loudly if it is wrong; the other two are worth a glance first.
    #
    # 1. The cancelled 1 lambda sweep left RSVD outputs on scratch:
    #
    #      ls $(backfill_prod_scratch(BACKFILL_PROD_PROJECT_1L)) | grep -c _UR_asym_Vpos.h5
    #
    #    bench/pick_bounds_points.jl runs before anything is submitted, lists what
    #    is actually there, reads each spectrum, and picks $(BACKFILL_PICKS) of them
    #    spread over the range of surviving m. Run it on its own first if you want to
    #    see the table without submitting anything:
    #
    #      bash bench/launch_calibration_$(cluster.name)_backfill.sh --pick
    #
    #    It writes the full kept-count table next to the pick file, which is the
    #    truncation measurement in its own right: how many of the positive
    #    Asym(G0_ur) survive --gamma-rtol, per separation, before any job runs.
    #
    # 2. The 1 lambda Green blocks are in $(BACKFILL_PROD_PRELOAD). They will be,
    #    for every separation the sweep reached -- the RSVD needed them. The bounds
    #    points additionally want the (S, R) block, which the RSVD never applied, so
    #    the first bounds job at a given separation may build one block before it
    #    starts. That is minutes at 1 lambda and the time limits have room for it.
    #
    # 3. The 1/2 lambda blocks are NOT there: that sweep was generated and never
    #    submitted. f_greens_0p5 builds them, into a preload directory of its own
    #    under CAL_ROOT, and the three 0p5 RSVD points depend on it with afterok.
    #
    # The RSVD points run at q = $(join(BACKFILL_QS, ", ")) and each gets its own
    # scratch subdirectory. file_prefix does not encode q, so without that the
    # second and third runs would find the first one's output and skip the work they
    # exist to measure -- and a --fresh run pointed at a production scratch directory
    # would delete a production basis. Neither happens here; check anyway if you edit
    # the --scratch paths.
    # ---------------------------------------------------------------------------
    """
end

"""
    backfill_epilogue(cluster)

The measure/refit/regenerate sequence, printed by the script when it finishes
submitting, including the one step that is specific to this tier: replaying the log
of any bounds job that the three-hour limit cut short.
"""
function backfill_epilogue(cluster::ClusterSpec)
    return """
    echo
    echo "When they have finished:"
    echo
    echo "  1. Any bounds job the 3 h limit cut short wrote no row, but its log holds"
    echo "     the numbers. Replay it (per killed job; --summary first to look):"
    echo
    echo "     julia --project=. bench/measure.jl \\\\"
    echo "         --parse-bounds-log \$CAL_ROOT/logs/f_bounds_1l_p1_<jobid>.out \\\\"
    echo "         --out \$ROWS/f_bounds_1l_p1_fromlog.csv \\\\"
    echo "         --cells 32,32,32 --scale 1//32 --sep <that pick's sep> \\\\"
    echo "         --rank $(BACKFILL_RANK) --cluster $(cluster.name) --jobid <jobid> \\\\"
    echo "         --note 'tier=backfill;label=f_bounds_1l_p1;from_walltime_cut=1'"
    echo
    echo "  2. Merge the rows and copy them back:"
    echo "     bash bench/launch_calibration_$(cluster.name)_backfill.sh --merge"
    echo "     scp $(CC_UNAME)@$(cluster.name).alliancecan.ca:\$OUT bench/data/calibration_$(cluster.name)_backfill.csv"
    echo "     scp $(CC_UNAME)@$(cluster.name).alliancecan.ca:\$KEPT_TABLE bench/data/"
    echo
    echo "  3. Refit. The new rows identify three things the old coefficients had no"
    echo "     measurement for, and bench/fit.jl reports each one by name:"
    echo "       bounds tau shape        grid evals and new whitenings per index"
    echo "       bounds gamma truncation m as a power law in separation"
    echo "       rsvd_pass_scale         measured / predicted per operator pass"
    echo "     Until they are fitted the model keeps its old constants exactly, so a"
    echo "     fit that reports them as 'not calibrated' has changed nothing."
    echo "     julia bench/fit.jl"
    echo
    echo "  4. Regenerate the job scripts and read the new bounds requests:"
    echo "     julia create_jobs.jl"
    """
end

function plan_points(cluster::ClusterSpec, tier::Symbol)
    tier == :funicular && return plan_funicular_points(cluster)
    tier == :backfill && return plan_backfill_points(cluster)
    quick = tier == :quick
    micro = tier in (:quick, :full)
    bodies = quick ? filter(b -> b.tier == :quick, BODIES) : BODIES
    separations = quick ? QUICK_SEPARATIONS : SEPARATIONS
    points = PlannedPoint[]

    common(body) = ["--cells", cells_arg(body.cells),
                    "--scale", rat(body.scale),
                    "--chi", DEFAULT_CHI,
                    "--rank", string(body.rank),
                    "--oversamples", string(DEFAULT_OVERSAMPLES),
                    "--power-iters", string(DEFAULT_POWER_ITERS)]

    # ---- Host: Green function block construction ---------------------------
    for body in (micro ? bodies : ())
        threads = min(4, cluster.max_cores)
        host_GB, time_s = block_resources(cluster, body, 8 // 32, threads)
        # Self blocks do not depend on the separation at all.
        push!(points, PlannedPoint("g0self_$(body.label)", "g0_self",
                                   vcat(common(body), ["--sep", rat(8 // 32)]),
                                   threads, host_GB, time_s, false))
        for sep in separations
            hg, ts = block_resources(cluster, body, sep, threads)
            push!(points, PlannedPoint("g0ext_$(body.label)_sep$(numerator(sep))ss$(denominator(sep))",
                                       "g0_ext", vcat(common(body), ["--sep", rat(sep)]),
                                       threads, hg, ts, false))
        end
        # The multi-region build is where the Green job actually peaks, so it is
        # the point that validates the memory model rather than a coefficient.
        push!(points, PlannedPoint("g0uu_$(body.label)", "g0_multiregion",
                                   vcat(common(body), ["--sep", rat(8 // 32)]),
                                   threads, host_GB, time_s, false))
    end

    # ---- Host: thread scaling ----------------------------------------------
    scan_body = bodies[min(length(bodies), quick ? 3 : 4)]
    for threads in (micro ? filter(<=(cluster.max_cores), THREAD_SCAN) : ())
        host_GB, time_s = block_resources(cluster, scan_body, 8 // 32, threads)
        push!(points, PlannedPoint("g0threads_$(scan_body.label)_t$(threads)", "g0_ext",
                                   vcat(common(scan_body), ["--sep", rat(8 // 32)]),
                                   threads, host_GB, time_s, false))
    end

    # ---- Device: Green matvecs ---------------------------------------------
    for body in (micro ? bodies : ())
        args = vcat(common(body), ["--sep", rat(8 // 32), "--reps", "20"])
        host_GB, time_s = block_resources(cluster, body, 8 // 32, 4)
        for (kind, suffix) in (("matvec_self", "self"), ("matvec_ext", "ext"),
                               ("matvec_uu", "uu"))
            push!(points, PlannedPoint("mv$(suffix)_$(body.label)", kind, args,
                                       min(4, cluster.max_cores), host_GB, time_s, true))
        end
    end

    # ---- Device: dense linear algebra --------------------------------------
    for body in (micro ? bodies : ())
        N_u = 6 * prod(body.cells)
        for c in DENSE_WIDTHS
            c <= N_u || continue
            # Three m-by-c complex matrices plus workspace, kept under 40% of the
            # smallest VRAM we calibrate on so the same plan runs everywhere.
            bytes = 3 * N_u * c * 16 + 6 * c^2 * 16
            bytes < 0.4 * cluster.max_vram_GB * 2^30 || continue
            push!(points, PlannedPoint("dense_$(body.label)_c$(c)", "dense",
                                       vcat(common(body),
                                            ["--sep", rat(8 // 32), "--dense-m", string(N_u),
                                             "--dense-c", string(c), "--reps", "12"]),
                                       min(4, cluster.max_cores),
                                       min(cluster.max_host_GB,
                                           max(8, ceil(Int, 2 * bytes / 2^30) + 4)),
                                       3600, true))
        end
    end

    # ---- Device: the bounds kernel on synthetic spectra --------------------
    for body in (micro ? bodies : ())
        for rank in unique([min(body.rank, 256), min(body.rank, 800), body.rank])
            N_u = 6 * prod(body.cells)
            bytes = 5 * N_u * rank * 16
            bytes < 0.5 * cluster.max_vram_GB * 2^30 || continue
            args = ["--cells", cells_arg(body.cells), "--scale", rat(body.scale),
                    "--chi", DEFAULT_CHI, "--sep", rat(8 // 32),
                    "--rank", string(rank), "--oversamples", string(DEFAULT_OVERSAMPLES),
                    "--power-iters", string(DEFAULT_POWER_ITERS),
                    "--num-pos-frac", "0.5", "--outer-samples", "4"]
            push!(points, PlannedPoint("boundscore_$(body.label)_k$(rank)", "bounds_core",
                                       args, min(4, cluster.max_cores),
                                       min(cluster.max_host_GB,
                                           max(16, ceil(Int, 3 * bytes / 2^30) + 8)),
                                       4 * 3600, true))
        end
    end

    # ---- Memory tier: the real RSVD footprint, cheaply --------------------
    #=
    Green functions on the CPU (they have to exist before the RSVD can load them),
    then the real RSVD with power iterations cut to 2. Memory does not depend on
    `q` -- each power iteration recycles one N_u x c block -- so this measures
    exactly the RAM and VRAM the production job uses, for roughly a sixth of the
    matvecs. This is the only grounded source for RSVD memory; the dense points
    over-state it because their timing loops churn allocations.
    =#
    if tier == :memory
        for body in BODIES
            tag = body.label
            args = vcat(common(body), ["--sep", rat(8 // 32)])
            hg, ts = block_resources(cluster, body, 8 // 32, min(4, cluster.max_cores))
            greens_label = "memgreens_$(tag)"
            push!(points, PlannedPoint(greens_label, "stage_greens", args,
                                       min(4, cluster.max_cores), hg, ts, false))
            # Ask for a whole GPU: we are measuring how much it needs, so capping
            # the allocation in advance would just censor the answer.
            pt = as_srpoint(body, 8 // 32, min(4, cluster.max_cores))
            coeffs = coefficients_for(cluster.name)
            p_rsvd = predict(GenerateRSVD, pt, coeffs; pad=false)
            # `common(body)` already carries --power-iters, so drop it before
            # appending the reduced value rather than passing the flag twice.
            base_args = String[]
            local skip_next = false
            for a in common(body)
                if skip_next; skip_next = false; continue; end
                if a == "--power-iters"; skip_next = true; continue; end
                push!(base_args, a)
            end
            push!(points, PlannedPoint("memrsvd_$(tag)", "mem_rsvd",
                                       vcat(base_args,
                                            ["--sep", rat(8 // 32), "--power-iters", "2"]),
                                       min(4, cluster.max_cores),
                                       min(cluster.max_host_GB,
                                           max(32, ceil(Int, 4 * p_rsvd.host_bytes / 2^30))),
                                       clamp(ceil(Int, 3 * p_rsvd.time_s), 1800, 12 * 3600),
                                       true, greens_label))
        end
    end

    # ---- End-to-end validation chains -------------------------------------
    if tier == :validate
        for body in bodies
            # One chain per body at a mid separation, plus contact on the
            # smallest two where it is cheap, since contact is the case where the
            # model is most likely to be wrong.
            seps = prod(body.cells) <= 4096 ? [8 // 32, 0 // 1] : [8 // 32]
            for sep in seps
                tag = "$(body.label)_sep$(numerator(sep))ss$(denominator(sep))"
                pt = as_srpoint(body, sep, min(4, cluster.max_cores))
                coeffs = coefficients_for(cluster.name)
                args = vcat(common(body), ["--sep", rat(sep)])
                hg, ts = block_resources(cluster, body, sep, min(4, cluster.max_cores))
                push!(points, PlannedPoint("stagegreens_$(tag)", "stage_greens", args,
                                           min(4, cluster.max_cores), hg, ts, false))
                p_rsvd = predict(GenerateRSVD, pt, coeffs; pad=false)
                push!(points, PlannedPoint("stagersvd_$(tag)", "stage_rsvd", args,
                                           min(4, cluster.max_cores),
                                           max(16, ceil(Int, 3 * p_rsvd.host_bytes / 2^30)),
                                           clamp(ceil(Int, 6 * p_rsvd.time_s), 3600, 24 * 3600),
                                           true))
                p_bounds = predict(ComputeBounds, pt, coeffs; pad=false)
                push!(points, PlannedPoint("stagebounds_$(tag)", "stage_bounds", args,
                                           min(4, cluster.max_cores),
                                           max(16, ceil(Int, 3 * p_bounds.host_bytes / 2^30)),
                                           clamp(ceil(Int, 6 * p_bounds.time_s), 3600, 24 * 3600),
                                           true))
            end
        end
    end

    return points
end

# --------------------------------------------------------------------------- #
# Script emission
# --------------------------------------------------------------------------- #

seconds2string(seconds::Real) = @sprintf("%02d:%02d:%02d", seconds ÷ 3600,
                                         (seconds % 3600) ÷ 60, seconds % 60)

function point_command(cluster::ClusterSpec, point::PlannedPoint, tier::Symbol)
    # `--note` is quoted because its value contains a `;`, and `--scale` because
    # anisotropic scales are negative rationals that a bare shell word would be
    # happy to mangle.
    # One CSV per point. Every point is a separate job appending to a shared
    # filesystem, and concurrent appends tear lines in half -- two narval rows were
    # lost that way. `merge_rows` at the end of the script stitches them together.
    # Single quotes everywhere except where the value is meant to be expanded by
    # the shell. The funicular tier's `--scratch` values are written against
    # `$CAL_ROOT`, and a single-quoted `$CAL_ROOT` is a directory called
    # `$CAL_ROOT`. Double quotes still keep the word together, which is what the
    # quoting was for.
    quote_arg(a) = startswith(a, "--") ? a : (occursin('$', a) ? "\"$a\"" : "'$a'")
    return join(vcat(["julia", "--project=.", "-t", string(point.threads),
                      "bench/point.jl", "--kind", point.kind],
                     [quote_arg(a) for a in point.args],
                     ["--gpu", point.gpu ? "0" : "-1",
                      "--root", "\$CAL_ROOT",
                      "--out", "\$ROWS/$(point.label).csv",
                      "--cluster", cluster.name,
                      "--note", "'tier=$(tier);label=$(point.label)'"]), " ")
end

"""
    resource_lines(cluster, point) -> String

The `sbatch` lines that say how much of the machine to take.

Calibration always takes a whole GPU, never a MIG slice: the whole point is to
measure the primitives on undivided hardware, and `create_jobs.jl` then derates
for a slice by its SM fraction.

The `funicular` tier is the exception, and names its own allocation per point
(`gpu_request`). Those trials do not measure primitives. They ask whether a given
bundle holds a given job, so the bundle is the variable under test, and the answer
for a whole card would not be an answer to that question.
"""
function resource_lines(cluster::ClusterSpec, point::PlannedPoint)
    lines = "    --cpus-per-task=$(point.threads) \\\n    --mem=$(point.host_GB)G \\\n"
    if point.gpu
        request = point.gpu_request === nothing ? cluster.full_gpu :
                  "$(point.gpu_request):1"
        lines *= "    --gpus=$(request) \\\n"
    end
    return lines
end

"""
    merge_block(cluster)

The `--merge` mode both generated scripts share: stitch the per-point row files
into one CSV, keeping a single header. Needed because each point writes its own
file -- concurrent appends to one shared-filesystem CSV tear lines in half.
"""
function merge_block(cluster::ClusterSpec)
    return """
    if [ "\${1:-}" = "--merge" ]; then
        n=\$(ls -1 \$ROWS/*.csv 2>/dev/null | wc -l)
        if [ "\$n" -eq 0 ]; then
            echo "No row files in \$ROWS -- nothing to merge."
            exit 1
        fi
        head -n 1 \$(ls -1 \$ROWS/*.csv | head -n 1) > \$OUT
        for f in \$ROWS/*.csv; do tail -n +2 "\$f" >> \$OUT; done
        echo "Merged \$n row file(s) into \$OUT (\$(( \$(wc -l < \$OUT) - 1 )) rows)."
        exit 0
    fi
    """
end

"""
    funicular_preamble(cluster)

Everything the `funicular` tier needs said before the first `sbatch`, as comments
in the generated script. Comments rather than commands: the login-node steps below
need a human to read their output, since an `instantiate` that cannot reach github
otherwise only surfaces three hours later on a compute node, and this script is
submitted from the login node rather than run on it.
"""
function funicular_preamble(cluster::ClusterSpec)
    return """
    # ---------------------------------------------------------------------------
    # Before submitting: three things, on the LOGIN node, in this order.
    #
    # 1. Instantiate. Compute nodes have no internet, and this tier is the first
    #    thing here to need Funicular and HDF5, both of which come in by URL:
    #
    #      $(cluster.modules)
    #      cd $(cluster.code_dir)
    #      julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
    #      julia --project=. -e 'using PhotonicSystemChannels, Funicular, CUDA; println(pkgdir(Funicular))'
    #
    #    The last line is the one that matters for E1. It prints the depot path
    #    whose `benchmark/` subdirectory holds `overlap.jl` and `pinned.jl`, which
    #    is what `bench/point.jl` resolves through `pkgdir(Funicular)`.
    #
    # 2. Funicular's `benchmark/Project.toml` does NOT need instantiating. It
    #    lists DelimitedFiles and Plots, and those are for `benchmark/plot.jl`
    #    only. `overlap.jl` and `pinned.jl` need CUDA, Funicular and Printf, all of
    #    which this project already has, so the E1 point runs them against the main
    #    project environment. It copies the directory under the point's --scratch
    #    first, because `benchmark/common.jl` writes its TSV results next to itself
    #    and the depot is not ours to write into.
    #
    # 3. Check the slice names are still what this script asks for, since a name
    #    the cluster does not define is a hard sbatch rejection:
    #
    #      sinfo -o "%G" | sort -u
    #
    # Trial E3c (`e3c_l4_spill`) spills to node-local NVMe through
    # \$SLURM_TMPDIR. It asks for --mem=$(FUNICULAR_SPILL_MEM_GB)G specifically:
    # `residency_plan` reads SLURM_MEM_PER_NODE and subtracts a 6 GiB overhead
    # reserve, so that request is a 60 GiB host budget exactly, against a ~95 GiB
    # panel peak. No --tmp is requested (narval GPU nodes carry NVMe and the flag
    # is not universally accepted); if the job dies writing spill files, check that
    # \$SLURM_TMPDIR has ~120 GB free on the node it landed on.
    #
    # The three E4 points need workstream C (the panelized bounds front-end in
    # src/bounds.jl). They are written against the CLI that exists today, but at
    # k = 4000 the old in-memory front-end would want an N_u x m basis as one
    # CuArray, ~30 GB at 4 lambda, so submit them only once C has landed. The
    # E1-E3 points do not depend on it; comment the E4 block out to run the rest.
    # ---------------------------------------------------------------------------
    """
end

"""
    funicular_epilogue(cluster, points)

What to run once the trials come back: the parity comparison E2 exists for, and
the refit that turns E1's rows into `pcie_rate` and `overlap_factor`.
"""
function funicular_epilogue(cluster::ClusterSpec, points::Vector{PlannedPoint})
    return """
    echo
    echo "When they have finished:"
    echo
    echo "  1. Trial E2's parity check (login node, no GPU needed). The two paths"
    echo "     use different RNG mechanisms, so this reports the deviation of the"
    echo "     top of the spectrum rather than asserting equality:"
    echo
    echo "     julia --project=. bench/compare_parity.jl \\\\"
    echo "         --a \$CAL_ROOT/funicular/e2_l1_inmem/*.jld \\\\"
    echo "         --b \$CAL_ROOT/funicular/e2_l1_panel/*.jld \\\\"
    echo "         --label-a in-memory --label-b panel --rtol 1e-6"
    echo
    echo "  2. Merge the rows and copy them back:"
    echo "     bash bench/launch_calibration_$(cluster.name)_funicular.sh --merge"
    echo "     scp $(CC_UNAME)@$(cluster.name).alliancecan.ca:\$OUT bench/data/calibration_$(cluster.name)_funicular.csv"
    echo
    echo "  3. Refit. E1's panel_bus rows identify pcie_rate and overlap_factor;"
    echo "     the E2/E3 stage_rsvd rows are what panel_host_mem_factor and"
    echo "     panel_workspace_bytes have been waiting for:"
    echo "     julia bench/fit.jl"
    echo
    echo "  4. Re-run create_jobs.jl and read its print_plan against the capacity"
    echo "     table at the top of FUNICULAR_PLAN.md."
    """
end

function slurm_script(cluster::ClusterSpec, points::Vector{PlannedPoint}, tier::Symbol)
    io = IOBuffer()
    println(io, """
    #!/bin/bash
    # Cost-model calibration for $(cluster.name), tier=$(tier).
    # Generated $(now()) by bench/plan.jl. Do not edit; regenerate instead.
    #
    # Every point is its own job: one point running out of memory or time must
    # not take the rest of the calibration with it. Each writes its own row file,
    # so partial results are always usable.
    #
    # Submit:  bash <this script>
    # Collect: bash <this script> --merge$(tier == :backfill ? "\n# Pick:    bash <this script> --pick   (choose the bounds separations, submit nothing)" : "")
    $(tier == :funicular ? "\n" * funicular_preamble(cluster) :
      tier == :backfill ? "\n" * backfill_preamble(cluster) : "")
    set -u

    CODE_DIR=$(cluster.code_dir)
    CAL_ROOT=$(cluster.cal_root)
    ROWS=\$CAL_ROOT/$(tier == :funicular ? "rows_funicular" : tier == :backfill ? "rows_backfill" : "rows")
    OUT=\$CAL_ROOT/calibration_$(cluster.name)$(tier == :funicular ? "_funicular" : tier == :backfill ? "_backfill" : "").csv

    mkdir -p \$CAL_ROOT/logs \$CAL_ROOT/preload \$CAL_ROOT/project \$CAL_ROOT/scratch \$ROWS
    cd \$CODE_DIR

    $(merge_block(cluster))$(tier == :backfill ? backfill_pick_block(cluster) : "")
    echo "Submitting $(length(points))$(tier == :backfill ? " + $(BACKFILL_PICKS) picked bounds" : "") calibration points for $(cluster.name) (tier=$(tier))"
    echo "Each point writes its own row file under \$ROWS"
    """)

    for point in points
        # Capture the job id so later points can depend on this one, and emit the
        # dependency line when a point needs an earlier one's output (the memory
        # tier's RSVD points need their Green functions on disk first).
        var = "jid_$(replace(point.label, r"[^A-Za-z0-9_]" => "_"))"
        dep_line = point.depends_on === nothing ? "" :
            "    --dependency=afterok:\${jid_$(replace(point.depends_on, r"[^A-Za-z0-9_]" => "_"))} \\\n"
        println(io, """
        $var=\$(sbatch --parsable \\
        $(dep_line)    --job-name=psccal_$(point.label) \\
            --output=\$CAL_ROOT/logs/$(point.label)_%j.out \\
            --account=$(cluster.account) \\
            --time=$(seconds2string(point.time_s)) \\
        $(resource_lines(cluster, point))    --chdir=\$CODE_DIR \\
            --export=ALL \\
            <<EOF
        #!/bin/bash
        $(cluster.modules)
        export PSC_T0=\\\$(date +%s)
        srun $(point_command(cluster, point, tier))
        EOF
        )
        sleep 0.05
        """)
    end

    # The backfill tier's bounds points come from a shell loop over the picker's
    # output rather than from `points`, so they are emitted here, before the "all
    # submitted" line rather than after it.
    tier == :backfill && print(io, "\n" * backfill_bounds_block(cluster))

    # `print`, not `println`: the tier-specific tail below supplies the final
    # newline, and splitting this block in two must not add one of its own.
    print(io, """
    echo
    echo "All points submitted. Watch them with: squeue -u \\\$USER"
    """)
    if tier == :backfill
        println(io, backfill_epilogue(cluster))
    elseif tier == :funicular
        println(io, funicular_epilogue(cluster, points))
    else
        println(io, """
        echo
        echo "When they have finished, merge the per-point rows and copy the result back:"
        echo "  bash bench/$(basename("launch_calibration_$(cluster.name)_$(tier).sh")) --merge"
        echo "  scp $(CC_UNAME)@$(cluster.name).alliancecan.ca:\$OUT bench/data/"
        """)
    end
    return String(take!(io))
end

function bash_script(cluster::ClusterSpec, points::Vector{PlannedPoint}, tier::Symbol)
    io = IOBuffer()
    println(io, """
    #!/bin/bash
    # Cost-model calibration for $(cluster.name), tier=$(tier).
    # Generated $(now()) by bench/plan.jl. Do not edit; regenerate instead.
    #
    # No scheduler here, so points run one at a time in the foreground. Each is
    # allowed to fail without stopping the run; check the logs afterwards for
    # any point whose row is missing from the CSV.
    #
    # Run it detached, it takes a while:
    #   nohup bash $(basename("launch_calibration_$(cluster.name)_$(tier).sh")) > calibration.log 2>&1 &

    set -u

    CODE_DIR=$(cluster.code_dir)
    CAL_ROOT=$(cluster.cal_root)
    ROWS=\$CAL_ROOT/rows
    OUT=\$CAL_ROOT/calibration_$(cluster.name).csv

    mkdir -p \$CAL_ROOT/logs \$CAL_ROOT/preload \$CAL_ROOT/project \$CAL_ROOT/scratch \$ROWS
    cd \$CODE_DIR

    export PSC_CLUSTER=$(cluster.name)

    $(merge_block(cluster))
    total=$(length(points))
    index=0
    """)

    for point in points
        println(io, """
        index=\$((index + 1))
        echo "[\$index/\$total] $(point.label)"
        export PSC_T0=\$(date +%s)
        $(point_command(cluster, point, tier)) \\
            > \$CAL_ROOT/logs/$(point.label).out 2>&1 \\
            || echo "  FAILED: $(point.label) (see \$CAL_ROOT/logs/$(point.label).out)"
        """)
    end

    println(io, """
    echo
    echo "Done. Merge the per-point rows, then copy the result back:"
    echo "  bash bench/$(basename("launch_calibration_$(cluster.name)_$(tier).sh")) --merge"
    echo "  scp $(MOLERING_UNAME)@molering:\$OUT bench/data/"
    """)
    return String(take!(io))
end

"""
    write_manifest(path, cluster, points, tier)

The human-readable list of what is about to be submitted. Read it before
submitting; it is the cheapest place to notice that a point is going to ask for
something silly.

Two schemas. The four calibration tiers keep the original columns

    label,kind,threads,host_GB,time_limit_s,gpu,args

where `gpu` is 0/1 (calibration always takes a whole GPU) and every geometry,
rank and separation lives inside the quoted `args` string exactly as
`bench/point.jl` will receive them. The `funicular` tier appends four columns

    ...,gpu,gpu_request,depends_on,predicted_wall_s,predicted_gpu_h,args

because three of its facts have nowhere else to go: which allocation the point
names (the trials compare allocations, so 0/1 cannot say it), which point must
finish first (the Green functions and the E4 chain), and what the cost model
predicts, which is the number the trial is judged against. The extension is
additive and only on these tiers, so anything reading the older manifests keeps
working.

`backfill` and `refined` take the wide schema for the same reason. Both chain
points -- a device point there reads what a host point wrote -- and both are read
by a human deciding whether the queue cost is worth it, which is a question the
prediction columns answer and the limit column does not.
"""
function write_manifest(path::AbstractString, cluster::ClusterSpec,
                       points::Vector{PlannedPoint}, tier::Symbol)
    open(path, "w") do io
        if tier in (:funicular, :backfill, :refined)
            println(io, "label,kind,threads,host_GB,time_limit_s,gpu,gpu_request," *
                        "depends_on,predicted_wall_s,predicted_gpu_h,args")
            for p in points
                println(io, join([p.label, p.kind, p.threads, p.host_GB, p.time_s,
                                  p.gpu ? 1 : 0,
                                  p.gpu_request === nothing ? "" : p.gpu_request,
                                  p.depends_on === nothing ? "" : p.depends_on,
                                  @sprintf("%.0f", p.predicted_s),
                                  @sprintf("%.3f", p.bill_fraction * p.predicted_s / 3600),
                                  "\"$(join(p.args, " "))\""], ","))
            end
            tier == :backfill && backfill_manifest_templates(io, cluster)
        else
            println(io, "label,kind,threads,host_GB,time_limit_s,gpu,args")
            for p in points
                println(io, join([p.label, p.kind, p.threads, p.host_GB, p.time_s,
                                  p.gpu ? 1 : 0, "\"$(join(p.args, " "))\""], ","))
            end
        end
    end
    return path
end

"""
    backfill_manifest_templates(io, cluster)

The four `stage_bounds` rows of the backfill manifest.

They are templates, not points. Their separation is chosen on the login node by
`bench/pick_bounds_points.jl` from what survived on scratch, and so are their
allocation, memory, time limit and outer-loop mode -- each from the `m` that
separation's own spectrum yields under `--gamma-rtol`. So the columns that would name
those carry `picker` rather than a number, and `args` carries the shell variables the
launcher substitutes at submission time.

Written into the manifest anyway, because the manifest is meant to be a complete
account of what a script submits, and a reader who finds seven rows for an
eleven-job script has been misled. The `--pick` mode of the launcher is what turns
these four rows into concrete ones, before anything is queued.
"""
function backfill_manifest_templates(io::IO, cluster::ClusterSpec)
    common = join(vcat(backfill_common(BACKFILL_BODIES.l1.cells, DEFAULT_POWER_ITERS),
                       ["--gamma-rtol", BACKFILL_GAMMA_RTOL,
                        "--outer-block-len", string(BACKFILL_OUTER_BLOCK_LEN),
                        "--sep", "\$SEP", "--outer-blocks", "\$BLOCKS",
                        "--scratch", "\$PROD_SCRATCH", "--preload", "\$PROD_PRELOAD"]), " ")
    for i in 1:BACKFILL_PICKS
        println(io, join(["f_bounds_1l_p$(i)", "stage_bounds", 2, "picker", "picker", 1,
                          "picker", "", "picker", "picker", "\"$(common)\""], ","))
    end
    return io
end

"""
    print_funicular_cost(cluster, points)

The table the `funicular` tier is judged by before it is submitted: what each
trial asks for, what the cost model thinks it will take, and what that costs in
GPU-equivalent hours. The budget in FUNICULAR_PLAN.md is 30 GPU-hours, and this is
printed so that a plan which has drifted past it gets noticed while the fix is
still an edit rather than a cancelled job.

Two totals, because they answer different questions. The predicted total is the
model's unpadded estimate, that is, what the trials should actually cost. The
total at the limits is what they would cost if every job ran to its wall clock,
which is the number to compare against an allocation's remaining balance. The
limits here are loose on purpose, since a trial killed mid-measurement wastes the
whole trial.

The E3c row's prediction is the known miss in this table. The cost model has no
NVMe term, so it prices that run as though the host tier held everything. It will
be slower, and by how much is the measurement.
"""
function print_funicular_cost(cluster::ClusterSpec, points::Vector{PlannedPoint})
    println()
    println("Predicted cost (cost model, unpadded; GPU-equivalents billed as the")
    println("largest of the GPU, RAM and core shares of the per-GPU bundle):")
    println()
    @printf("  %-20s %-14s %5s %6s   %9s %9s %8s\n",
            "point", "allocation", "cpus", "mem", "predicted", "limit", "GPU-h")
    predicted_gpu_h = 0.0
    limit_gpu_h = 0.0
    for p in points
        alloc = p.gpu ? (p.gpu_request === nothing ? cluster.full_gpu : p.gpu_request) : "cpu"
        gpu_h = p.bill_fraction * p.predicted_s / 3600
        predicted_gpu_h += gpu_h
        limit_gpu_h += p.bill_fraction * p.time_s / 3600
        @printf("  %-20s %-14s %5d %5dG   %8.2fh %8.2fh %8.2f\n",
                p.label, alloc, p.threads, p.host_GB,
                p.predicted_s / 3600, p.time_s / 3600, gpu_h)
    end
    println()
    @printf("  predicted total   %6.1f GPU-hours over %d GPU jobs (+%d CPU jobs)\n",
            predicted_gpu_h, count(p -> p.gpu, points), count(p -> !p.gpu, points))
    @printf("  at the limits     %6.1f GPU-hours\n", limit_gpu_h)
    #=
    E1 has no cost-model prediction, since no SRPoint describes how fast the PCIe
    link is, so its predicted hours read as zero and its limit is the real number.
    Printed rather than fudged: an unexplained zero in a cost table is how a
    budget gets believed.
    =#
    println("  (E1 has no cost-model prediction: no SRPoint describes a bus")
    println("   benchmark, so it contributes 0 to the predicted total and its")
    println("   1 h limit to the other. E3c's prediction omits the NVMe round")
    println("   trip the trial exists to measure, so it will run over.)")
    #=
    The three E4 rows are most of this total, and they are sized at
    NUM_POS_FRACTION = 0.6, which the cost model calls "deliberately pessimistic"
    because bounds time grows as m^4. The existing outputs run 0.22-0.52. The
    sensitivity is printed rather than left for the reader to work out, since it
    decides whether this tier fits the budget.
    =#
    bounds_gpu_h = sum((p.bill_fraction * p.predicted_s / 3600
                        for p in points if p.kind == "stage_bounds"); init=0.0)
    @printf("  of which bounds   %6.1f GPU-hours, at NUM_POS_FRACTION = 0.6\n", bounds_gpu_h)
    @printf("  tier total        %6.1f GPU-hours if num_pos comes back at 0.5k, the top\n",
            predicted_gpu_h - bounds_gpu_h * (1 - (0.5 / 0.6)^4))
    println("                    of the historically measured 0.22-0.52 range")
    predicted_gpu_h <= 30 ||
        @info "predicted past FUNICULAR_PLAN.md's 30 GPU-hour budget at the pessimistic num_pos; see the sensitivity line above" predicted_gpu_h
    return nothing
end

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

function main(argv::Vector{String})
    opts = Dict{String,String}()
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("Expected an option starting with --, got '$(argv[i])'")
        key = argv[i][3:end]
        if i + 1 <= length(argv) && !startswith(argv[i + 1], "--")
            opts[key] = argv[i + 1]
            i += 2
        else
            opts[key] = "true"
            i += 1
        end
    end

    cluster_name = get(opts, "cluster", "")
    isempty(cluster_name) && error("--cluster is required (fir, narval or molering)")
    tier = Symbol(get(opts, "tier", "quick"))
    tier in (:quick, :full, :memory, :validate, :funicular, :backfill) ||
        error("--tier must be quick, full, memory, validate, funicular or backfill")
    dry_run = get(opts, "dry-run", "false") in ("true", "1", "yes")

    load_coefficients!(@__DIR__)
    cluster = ClusterSpec(cluster_name)
    points = plan_points(cluster, tier)

    script = cluster.has_slurm ? slurm_script(cluster, points, tier) :
             bash_script(cluster, points, tier)
    script_path = joinpath(@__DIR__, "launch_calibration_$(cluster_name)_$(tier).sh")
    manifest_path = joinpath(@__DIR__, "manifest_$(cluster_name)_$(tier).csv")
    if dry_run
        # Plan and cost it, write nothing. Useful for reading the resource table
        # before letting it overwrite a script that is already out on a cluster.
        println("--dry-run: planned but wrote nothing\n")
    else
        write(script_path, script)
        chmod(script_path, 0o755)
        write_manifest(manifest_path, cluster, points, tier)
    end

    gpu_points = count(p -> p.gpu, points)
    total_time = sum(p.time_s for p in points)
    println("Planned $(length(points)) points for $(cluster_name) (tier=$tier)")
    if tier == :backfill
        println("  plus $(BACKFILL_PICKS) stage_bounds points whose separation, allocation, memory and")
        println("  time limit are chosen on the login node by bench/pick_bounds_points.jl,")
        println("  from the m that each surviving RSVD output yields under --gamma-rtol")
        println("  ($(BACKFILL_PICKS + length(points)) jobs in total, none over $(seconds2string(BACKFILL_MAX_TIME_S)))")
    end
    println("  $(length(points) - gpu_points) host points, $gpu_points device points")
    println("  worst-case wall time if every point used its whole limit: ",
            @sprintf("%.1f h", total_time / 3600))
    println("  (real total is far less; the limits are deliberately loose)")
    by_kind = Dict{String,Int}()
    for p in points
        by_kind[p.kind] = get(by_kind, p.kind, 0) + 1
    end
    for (kind, count) in sort(collect(by_kind))
        println("    ", rpad(kind, 16), count)
    end
    tier == :funicular && print_funicular_cost(cluster, points)
    println()
    if dry_run
        println("Would write $script_path")
        println("Would write $manifest_path")
    else
        println("Wrote $script_path")
        println("Wrote $manifest_path")
    end
    println()
    if cluster.has_slurm
        println("Copy and run:")
        println("  scp $script_path $(CC_UNAME)@$(cluster_name).alliancecan.ca:$(cluster.code_dir)bench/")
        println("  ssh $(CC_UNAME)@$(cluster_name).alliancecan.ca 'cd $(cluster.code_dir) && bash bench/$(basename(script_path))'")
    else
        println("Copy and run:")
        println("  scp $script_path $(MOLERING_UNAME)@molering:$(cluster.code_dir)bench/")
        println("  ssh $(MOLERING_UNAME)@molering 'cd $(cluster.code_dir) && nohup bash bench/$(basename(script_path)) > calibration.log 2>&1 &'")
    end
    return script_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
