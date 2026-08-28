using Dates
using Serialization
using ArgParse
import CUDA
import Funicular
import HDF5 # Funicular's disk tier only exists once HDF5 is loaded

function ArgParse.parse_item(::Type{GPUChoice}, x::AbstractString)
    s = lowercase(String(x))

    if s in ("true", "t", "yes", "y")
        return GPUChoice(true, 0)
    elseif s in ("false", "f", "no", "n")
        return GPUChoice(false, -1)
    else
        try
            return GPUChoice(true, parse(Int, s))
        catch
            error("Invalid GPU argument '$x'. Use true/false or an integer GPU device index.")
        end
    end
end

function parse_3tuple(s::AbstractString, conv::Function)
    inner = strip(s)
    inner = strip(inner, ['(', ')'])
    parts = split(inner, ',')
    length(parts) == 3 || error("Expected 3 components, got $(length(parts)) in '$s'")
    return (conv(strip(parts[1])),
            conv(strip(parts[2])),
            conv(strip(parts[3])))
end
parse_int_3tuple(s::AbstractString) = parse_3tuple(s, x -> parse(Int, x))
function parse_rational(s::AbstractString)
    numden = split(strip(s), "//"; limit = 2)
    length(numden) == 2 || error("Invalid rational '$s', expected a//b")
    return parse(Int, numden[1]) // parse(Int, numden[2])
end

function Base.parse(::Type{Rational}, s::AbstractString)
    numden = split(strip(s), "//"; limit=2)
    length(numden) == 2 || error("Invalid rational '$s', expected a//b")
    return parse(Int, numden[1]) // parse(Int, numden[2])
end
parse_rational_3tuple(s::AbstractString) = parse_3tuple(s, x -> parse(Rational, x))
ArgParse.parse_item(::Type{NTuple{3, Int}}, s::AbstractString) = parse_int_3tuple(s)
ArgParse.parse_item(::Type{NTuple{3, Rational{Int}}}, s::AbstractString) = parse_rational_3tuple(s)

"""
    parse_index_range(s) -> UnitRange{Int}

`--outer-range lo:hi`, the slice of the bounds job's outer loop over channel
indices this process is responsible for. Inclusive at both ends, 1-based, and the
same convention Julia's own `lo:hi` has, because that is what the flag's value is
read back as everywhere downstream. A bare `n` is `n:n`, which is mostly useful
for a one-index reproduction of a suspicious channel.
"""
function parse_index_range(s::AbstractString)
    parts = split(strip(s), ':')
    lo, hi = if length(parts) == 1
        v = parse(Int, strip(parts[1])); (v, v)
    elseif length(parts) == 2
        (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])))
    else
        error("Invalid index range '$s', expected lo:hi")
    end
    lo >= 1 || error("Invalid index range '$s': channel indices are 1-based, got lo = $lo")
    hi >= lo || error("Invalid index range '$s': hi < lo, which selects no index at all")
    return lo:hi
end
ArgParse.parse_item(::Type{UnitRange{Int}}, s::AbstractString) = parse_index_range(s)

function _default_preload_dir()
    path = joinpath("/Users", ENV["USER"], "Desktop", "preload")
    if haskey(ENV, "MOLERING")
        # On our group's server
        path = joinpath(ENV["MOLERING"], "fatmole", "greens_functions")
    elseif haskey(ENV, "CC_CLUSTER")
        # On compute Canada
        path = joinpath("/home", ENV["USER"], "scratch", "preload")
    end
    return path
end
function _default_project_dir(project_name::AbstractString)
    path = joinpath("/Users", ENV["USER"], "Desktop", project_name)
    if haskey(ENV, "MOLERING")
        # On our group's server
        path = joinpath("/home", ENV["USER"], "Sender-Mediator-Receiver SVD Bounds", "projects", project_name)
    elseif haskey(ENV, "CC_CLUSTER")
        # On compute Canada
        path = joinpath("/home", ENV["USER"], "projects", "rrg-smolesky", ENV["USER"], project_name)
    end
    return path
end
function _default_scratch_dir(project_name::AbstractString)
    path = joinpath("/Users", ENV["USER"], "Desktop", project_name, "scratch")
    if haskey(ENV, "MOLERING")
        # On our group's server
        path = joinpath(ENV["MOLERING"], "fatmole", ENV["USER"], "SMR-Bounds", project_name)
    elseif haskey(ENV, "CC_CLUSTER")
        # On compute Canada
        path = joinpath("/home", ENV["USER"], "scratch", project_name)
    end
    return path
end
function _default_gpu()
    if haskey(ENV, "MOLERING") || haskey(ENV, "CC_CLUSTER")
        return GPUChoice(true, 0)
    end
    return GPUChoice(false, -1)
end

function ArgParse.parse_args()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--sender"
            help = "Sender volume size as (x,y,z) with integer number of cells"
            arg_type = NTuple{3, Int}
            required = true

        "--mediator"
            help = "Mediator volume size as (x,y,z) with integer number of cells"
            arg_type = NTuple{3, Int}
            default = (0, 0, 0)

        "--receiver"
            help = "Receiver volume size as (x,y,z) with integer number of cells"
            arg_type = NTuple{3, Int}
            required = true

        "--sm-sep"
            help = "Sender–mediator separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
            default = (1//0, 1//0, 1//0) # To warn if not set and was needed

        "--mr-sep"
            help = "Mediator–receiver separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
            default = (1//0, 1//0, 1//0) # To warn if not set and was needed

        "--rs-sep"
            help = "Sender–receiver separation as (a//b, c//d, e//f) in wavelengths"
            arg_type = NTuple{3, Rational{Int}}
            default = (1//0, 1//0, 1//0) # To warn if not set and was needed

        "--scale"
            help = "Mesh scale as a//b in wavelengths per cell"
            arg_type = Rational{Int}
            required = true

        "--chi"
            help = "Complex susceptibility χ = a + bi (pass 'a + bi')"
            arg_type = ComplexF64
            required = true

        "--name"
            help = "Project name"
            arg_type = String
            required = true

        "--design"
            help = "Design region"
            arg_type = String
            required = true

        "--preload"
            help = "Directory for preloaded Green's functions"
            arg_type = String
            default = _default_preload_dir()

        "--project"
            help = "Project directory"
            arg_type = String

        "--scratch"
            help = "Scratch directory"
            arg_type = String

        "--gpu"
            help = "Use GPU acceleration"
            arg_type = GPUChoice
            default = _default_gpu()

        "--components"
            help = "Number of singular value components to compute"
            arg_type = Int
            default = 256

        "--oversamples"
            help = "Number of oversamples to use in RSVD"
            arg_type = Int
            default = 20

        "--power-iterations"
            help = "Number of power iterations to use in RSVD"
            arg_type = Int
            default = 14

        "--seed"
            help = "Seed for the panel path's regenerated test matrix (0 derives it from --name)"
            arg_type = Int
            default = 0

        "--gamma-rtol"
            help = "Relative cut on the positive Asym(G⁰ᵤᵣ) spectrum for the bounds: keep only Γ ≥ gamma-rtol * Γ₁ (0 keeps the whole positive block)"
            arg_type = Float64
            default = DEFAULT_GAMMA_RTOL

        "--k-uu"
            help = "How many leading eigenvectors of Asym(G⁰ᵤᵤ) to augment the bounds' projection basis with, on points below --augment-threshold. 0 disables the augmentation and reproduces the pre-augmentation output bit for bit"
            arg_type = Int
            default = DEFAULT_K_UU

        "--augment-threshold"
            help = "Augment only when the kept m = num_pos is below this. Far-field points, whose g basis is too small to represent Asym(G⁰ᵤᵤ) and whose dual therefore under-reports, are augmented; near-field points run exactly as they did before --k-uu existed. Raising it widens the augmented half of a sweep and raises the ceiling on m_aug = m + k_uu, which clip_k_uu then trims back on any point whose dense front end would not fit the card"
            arg_type = Int
            default = DEFAULT_AUGMENT_THRESHOLD

        "--outer-range"
            help = "Bounds only: compute the outer loop over channel indices lo:hi instead of all of them, so that one long bounds job can be split across B independent short ones that backfill concurrently. The full front end still runs in each; only the loop is sliced. Requires --partial-suffix, and the blocks are assembled by bench/merge_bounds_blocks.jl"
            arg_type = UnitRange{Int}

        "--partial-suffix"
            help = "Bounds only: write the output as <prefix>_partial_<tag>.jld in the project directory instead of <prefix>.jld. Goes with --outer-range: it keeps the blocks of one point from overwriting each other, and keeps anything reading the project directory from mistaking a slice for a finished point"
            arg_type = String

        "--refine"
            help = "Refine the two facing surfaces when the sender and the receiver are closer than the six cells Gila's quadrature needs. This is the default; the flag is there so that a launcher can say so explicitly"
            action = :store_true

        "--no-refine"
            help = "Mesh the sender and the receiver as plain uniform volumes whatever their separation, and accept the quadrature error of the unrefined gap at the near separations"
            action = :store_true
    end
    args = parse_args(settings)

    project_name = args["name"]
    @info string(now()) * " [common::parse_args] Working on $project_name"

    if isnothing(get(args, "project", nothing))
        args["project"] = _default_project_dir(project_name)
    end
    if isnothing(get(args, "scratch", nothing))
        args["scratch"] = _default_scratch_dir(project_name)
    end

    compute_env = ComputeEnvironment(
        get(args, "preload", _default_preload_dir()),
        get(args, "project", _default_project_dir(project_name)),
        get(args, "scratch", _default_scratch_dir(project_name)),
        get(args, "gpu", _default_gpu())
    )
    mkpath(preload_dir(compute_env))
    mkpath(project_dir(compute_env))
    mkpath(scratch_dir(compute_env))
    @info string(now()) * " [common::parse_args] Using compute environment:" preload_dir(compute_env) project_dir(compute_env) scratch_dir(compute_env) use_gpu(compute_env)

    design_symbols = char2volume_symbol.(sort(collect(uppercase(args["design"])))) # sort to ensure consistent naming
    has_mediator = true
    if args["mediator"] == (0, 0, 0)
        any(args["rs-sep"] .== 1//0) && error("No mediator specified, but sender–receiver separation was not set")
        has_mediator = false
    else
        any(args["sm-sep"] .== 1//0) && error("Mediator specified, but sender–mediator separation was not set")
        any(args["mr-sep"] .== 1//0) && error("Mediator specified, but mediator–receiver separation was not set")
    end

    # Gap refinement changes the mesh the Green blocks are built on, so it also
    # changes their cache keys and the scratch key of the point. It is on unless
    # `--no-refine` turns it off; `--refine` says the default out loud and is
    # otherwise a no-op.
    args["refine"] && args["no-refine"] &&
        error("--refine and --no-refine ask for two different meshes")
    refine_gap = !args["no-refine"]

    if has_mediator
        smr = SMRSystem(
            args["sender"],
            args["sm-sep"],
            args["mediator"],
            args["mr-sep"],
            args["receiver"],
            design_symbols,
            args["scale"],
            args["chi"]
        )
        @info string(now()) * " [common::parse_args] Using SMR system with mediator" ms_separation(smr) rm_separation(smr) rs_separation(smr)
    else
        smr = SMRSystem(
            args["sender"],
            args["rs-sep"],
            args["receiver"],
            design_symbols,
            args["scale"],
            args["chi"];
            refine_gap=refine_gap
        )
        @info string(now()) * " [common::parse_args] Using SR system without mediator" rs_separation(smr)
        if is_refined(smr)
            @info string(now()) * " [common::parse_args] Refining the gap surfaces" refinement(smr) dof_length(sender_mesh(smr)) dof_length(receiver_mesh(smr))
        end
    end

    rsvd_params = RSVDParams(
        args["components"],
        args["oversamples"],
        args["power-iterations"],
        resolve_seed(args["seed"], project_name)
    )
    @info string(now()) * " [common::parse_args] Using RSVD parameters:" rank(rsvd_params) oversamples(rsvd_params) power_iter(rsvd_params) seed(rsvd_params)

    # `gamma_rtol` belongs to the bounds stage, not the RSVD, so it rides alongside
    # RSVDParams instead of going into it. The RSVD entry points destructure the
    # leading three and let this one fall off the end.
    gamma_rtol = args["gamma-rtol"]
    @info string(now()) * " [common::parse_args] Using gamma_rtol = $(gamma_rtol) for the bounds' spectral cut"

    # Same story as `gamma_rtol`: the Asym(G⁰ᵤᵤ) augmentation belongs to the bounds
    # stage, so these ride alongside `RSVDParams` rather than going into it, and the
    # RSVD and greens entry points let them fall off the end of the tuple.
    k_uu = args["k-uu"]
    augment_threshold = args["augment-threshold"]
    @info string(now()) * " [common::parse_args] Using k_uu = $(k_uu) and augment_threshold = $(augment_threshold) for the bounds' Asym(G⁰ᵤᵤ) augmentation"

    # Same story again: the block split belongs to the bounds stage alone, so it
    # rides on the end of the tuple and every other entry point lets it fall off.
    # Both are `nothing` unless asked for, and `_compute_bounds_sr` refuses one
    # without the other -- a block that writes to the point's real filename, or one
    # that writes every index under a partial name, are both worse than an error.
    outer_range = get(args, "outer-range", nothing)
    partial_suffix = get(args, "partial-suffix", nothing)
    if !isnothing(outer_range) || !isnothing(partial_suffix)
        @info string(now()) * " [common::parse_args] Partial bounds run: outer_range = $(outer_range), partial_suffix = $(partial_suffix)"
    end

    return compute_env, smr, rsvd_params, gamma_rtol, k_uu, augment_threshold,
           outer_range, partial_suffix, refine_gap
end

"""
    resolve_seed(requested::Int, project_name::AbstractString) -> Int

The seed the panel path sketches with. `requested == 0` (the `--seed` default)
derives one from the experiment name, so each separation of a sweep gets its own
seed without us writing 333 of them down; anything else is taken as given. The
result is always positive and depends on the name alone, so a rerun of the same
experiment sketches the same way. Note, however, that this only holds within one
Julia version, since `hash` and `Xoshiro` are implementation details.
"""
function resolve_seed(requested::Int, project_name::AbstractString)
    requested == 0 || return requested
    return Int(hash(project_name) >>> 1) # >>> 1 clears the sign bit, so this fits an Int
end

# Host memory the process needs for itself outside the panel tier: the Julia
# runtime, the Gila operator's host side, the JLD writes, the page-locking slack,
# and the page cache the h5 stream in `_save_ur_asym` dirties on its way out --
# Slurm charges dirty pages to the job's cgroup, and writeback does not retire
# them fast enough for them to be reclaimable on demand. Trial E2 pins the real
# number down. Was 6 GB, the plan's working estimate; raised to 8 GB after a
# sweep of 1 λ, k = 4000 panel jobs on narval was OOM-killed at the positive-block
# save with a 28 GiB budget carved out of a 34 GB request, i.e. with the whole
# margin already spent. The extra 2 GB is the measured shortfall, not a guess at a
# comfortable number: the analytic under-count that caused it is fixed on the
# request side, in `rsvd_host_bytes` in bench/cost_model.jl, and this reserve only
# has to keep the plan from handing the sketch memory the save and the page cache
# will then want.
const HOST_OVERHEAD_RESERVE_BYTES = 8 * 2^30
# Below this there is no point starting, since one row sweep has to hold a whole
# matrix.
const HOST_BUDGET_FLOOR_BYTES = 2^30
# Fraction of the device we hand to Funicular. The rest absorbs the driver, the
# memory pool's cached blocks, and CUDA's own allocations. We read it from
# total_memory rather than free_memory, which under-reports once the pool holds
# cached blocks. On a MIG slice, total_memory reports the slice, not the card.
const DEVICE_BUDGET_FRACTION = 0.9

device_budget_bytes() = floor(Int, DEVICE_BUDGET_FRACTION * CUDA.total_memory())

# Slurm reports memory in MB, but writes the unit out when the request carried
# one ("240G"), so we honour the suffix when there is one.
const SLURM_MEM_UNITS = Dict('K' => 2^10, 'M' => 2^20, 'G' => 2^30, 'T' => 2^40)

function _slurm_mem_bytes(name::String)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return nothing
    unit = get(SLURM_MEM_UNITS, uppercase(last(raw)), nothing)
    digits = unit === nothing ? raw : raw[1:end-1]
    value = tryparse(Int, digits)
    value === nothing && return nothing
    return value * (unit === nothing ? 2^20 : unit) # bare numbers are MB
end

# What Slurm gave us, in bytes. `SLURM_MEM_PER_NODE` is the whole-node request; a
# `--mem-per-cpu` job gets the same number from the per-CPU request times the
# task's CPU count. Off a cluster neither exists, so we fall back to the
# machine's total memory.
function _slurm_host_bytes()
    node = _slurm_mem_bytes("SLURM_MEM_PER_NODE")
    node === nothing || return node
    per_cpu = _slurm_mem_bytes("SLURM_MEM_PER_CPU")
    cpus = tryparse(Int, get(ENV, "SLURM_CPUS_PER_TASK", ""))
    (per_cpu === nothing || cpus === nothing) || return per_cpu * cpus
    return Int(Sys.total_memory())
end

"""
    residency_plan(compute_env; workspace_bytes=0) -> Funicular.ResidencyPlan or nothing

The `ResidencyPlan` the panel path streams its sketches through, sized from the
allocation this process is running inside. Returns `nothing` on a CPU run: with no
device in the picture the sketch is already in host memory, so the panel machinery
would only add bookkeeping and CPU runs keep the in-memory path
(FUNICULAR_PLAN.md, workstream B1/B3).

Both budgets come from what the job was granted, not from what is momentarily
free:

- `device_budget` is 90% of `CUDA.total_memory()`. The held-back tenth absorbs
  the driver and the memory pool's cached blocks. Nothing else is subtracted
  here. The operator's own device footprint goes in through `workspace_bytes`,
  which the plan holds back from the buffer pool itself.
- `host_budget` is the Slurm memory request minus a $(HOST_OVERHEAD_RESERVE_BYTES >> 30) GB overhead
  reserve for the runtime, the operator's host side and the page cache the
  positive-block h5 stream dirties (Slurm charges that to the cgroup), floored at
  $(HOST_BUDGET_FLOOR_BYTES >> 30) GB. Off a cluster it is the machine's total memory, same
  reserve. The reserve is a measured OOM margin, not a round number: see the
  comment on `HOST_OVERHEAD_RESERVE_BYTES`.
- `scratch_dir` is node-local NVMe (`\$SLURM_TMPDIR`) when Slurm gave us one.
  With a scratch dir the host tier no longer has to hold a whole sketch, only
  what a sweep touches at once.

`workspace_bytes` is a keyword rather than a trait because `G₀_ur_asym` is a
LinearMaps composition, which has nowhere to carry one. See
`gila_workspace_bytes` in `rsvd.jl` for the estimate.
"""
#=
Every `ResidencyPlan` this process has handed out and has not yet let go of, held
weakly so that a plan whose last strong reference is gone can still be collected.

This exists because a Funicular host pool is *per plan* and *never shrinks*.
`Funicular.free!` pushes a block onto `HostPool.free`, a size-keyed free list;
`HostPool.reserved`, `HostPool.slabs` and `HostPool.cursors` are only ever grown
(`grow_host_pool!` in Funicular's `src/plan.jl` is the sole writer of `reserved`,
and it only adds). There is no `release!`/`destroy!`/`close` on the pool, the plan
or the slab, and no finalizer: the page-locked `Vector{UInt8}` slabs go back to the
OS only when the whole plan becomes unreachable and Julia's GC collects them.

A plan therefore keeps its high-water mark charged to the cgroup for as long as it
is alive, and two plans built from the same Slurm allocation each believe they own
`allocation - reserve`. That is what killed the 4 λ probe RSVD: `_save_ur_asym`
finished a 786432 × 4050 panel decomposition inside a 116 GiB budget, and
`_run_rsvdvals("RS/")` then built a *second* plan with the same 116 GiB budget
while the first pool was still resident.
=#
const LIVE_RESIDENCY_PLANS = WeakRef[]

"""
    live_pool_host_bytes() -> Int

Host bytes already reserved by residency plans this process still holds. Sums
`Funicular.host_bytes_reserved` over the live entries of `LIVE_RESIDENCY_PLANS`
and drops the dead ones on the way through, so it is also the cheap way to see
whether a previous decomposition's pool has actually gone away.
"""
function live_pool_host_bytes()
    total = 0
    keep = WeakRef[]
    for ref in LIVE_RESIDENCY_PLANS
        plan = ref.value
        plan === nothing && continue
        push!(keep, ref)
        total += Funicular.host_bytes_reserved(plan.hostpool)
    end
    empty!(LIVE_RESIDENCY_PLANS)
    append!(LIVE_RESIDENCY_PLANS, keep)
    return total
end

"""
    reclaim_host_pools!() -> Int

Collect any residency plan the process has finished with and report the pinned
host bytes still held afterwards.

Call this *between* panel decompositions, from the frame above the one that built
the plan. It cannot work from inside that frame: the plan, the `PanelEigen` and the
`PanelFactored` are still live locals there, and a `PanelMatrix` holds its plan as a
field, so nothing is collectable until the frame is gone. `_generate_rsvd_sr` is the
right place, and `_save_ur_asym` / `_run_rsvdvals` help by dropping their own
references before they return.

There is no way to force the pool back to the OS (see `LIVE_RESIDENCY_PLANS`), so
this is GC and nothing more. The number it returns is the honest one: whatever is
left is charged against the next plan's budget by `residency_plan`, which is the
part that does not depend on the GC cooperating.
"""
function reclaim_host_pools!()
    before = live_pool_host_bytes()
    run_gc()
    after = live_pool_host_bytes()
    if before > 0 || after > 0
        @info string(now()) * " [common::reclaim_host_pools!] Residency-plan host pools" reserved_before=before reserved_after=after released=(before - after)
    end
    after > 0 && @warn string(now()) * " [common::reclaim_host_pools!] $(after) bytes of page-locked host pool are still reachable, so the next residency plan is budgeted around them. If a later decomposition spills heavily to scratch, this is why: a plan or a panel matrix from the previous decomposition is still referenced."
    return after
end

function residency_plan(compute_env::ComputeEnvironment; workspace_bytes::Int=0)
    if !use_gpu(compute_env)
        @info string(now()) * " [common::residency_plan] CPU run: no residency plan, the in-memory path is cheaper here"
        return nothing
    end

    device_budget = device_budget_bytes()
    #=
    Host budget for *this* plan: the allocation, less the overhead reserve, less
    whatever earlier plans have already pinned and not given back. Subtracting the
    live pools is what makes a second panel decomposition in one process safe
    without depending on the GC having run: if the previous plan was collected the
    term is zero and this plan gets the whole budget, and if it was not, this plan
    gets what is genuinely left and spills the rest to `scratch_dir` instead of
    being OOM-killed. See `LIVE_RESIDENCY_PLANS`.
    =#
    already_pinned = live_pool_host_bytes()
    host_budget = max(_slurm_host_bytes() - HOST_OVERHEAD_RESERVE_BYTES - already_pinned,
                      HOST_BUDGET_FLOOR_BYTES)

    scratch = nothing
    if haskey(ENV, "SLURM_TMPDIR")
        scratch = joinpath(ENV["SLURM_TMPDIR"], "funicular")
        mkpath(scratch)
    end

    if already_pinned > 0
        @warn string(now()) * " [common::residency_plan] An earlier residency plan still holds $(already_pinned) bytes of page-locked host memory, which this plan's budget is reduced by. Expect more spilling to scratch than the cost model's request assumes." already_pinned host_budget scratch
        scratch === nothing && @error string(now()) * " [common::residency_plan] ...and there is no SLURM_TMPDIR to spill to, so this decomposition may run out of host tier. Run the decompositions in separate jobs, or raise --mem."
    end

    @info string(now()) * " [common::residency_plan] Building a residency plan" device_budget host_budget workspace_bytes scratch already_pinned
    plan = Funicular.ResidencyPlan(
        backend=Funicular.cuda_backend(),
        device_budget=device_budget,
        host_budget=host_budget,
        workspace_bytes=workspace_bytes,
        scratch_dir=scratch
    )
    push!(LIVE_RESIDENCY_PLANS, WeakRef(plan))
    return plan
end

function run_gc()
    @info string(now()) * " [common::run_gc] Running garbage collector"
    GC.gc()
    GC.gc()
    GC.gc()
end

asym(X) = (X - X') / (2im)
sym(X) = (X + X') / 2
