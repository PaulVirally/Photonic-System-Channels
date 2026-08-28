using MatrixFreeRandomizedLinearAlgebra
using LinearMaps
using LinearAlgebra
using JLD2
using CUDA
using GilaElectromagnetics
import Funicular

# Path selection. There are three ways to get the spectrum of Asym(G⁰ᵤᵣ), in
# order of priority (FUNICULAR_PLAN.md, workstream B3/B6):
#
#   1. dense-exact, when the universe is small enough to build and diagonalize
#      the whole operator. No RSVD error, and the "rank" is the full spectrum.
#   2. in-memory RSVD, whenever the sketch fits on the device. A resident
#      CuArray sketch pays no upload per sweep, so it beats the panel path.
#   3. panel RSVD, above that.

# 1/4 λ at x-scale 1/32 is N_u = 3,072, so the cap covers it with room to spare.
# The dense eigensolve is still a few minutes at the top end.
const DENSE_EXACT_MAX_N_U = 12_288
# The RS values come from a rectangular operator, so the dense branch there is
# bounded by the smaller side. Half the Hermitian cap, since a dense N_r × N_s
# block is a full matrix rather than a triangle.
const DENSE_EXACT_MAX_N_R = 6_144
# Measured ratio of the in-memory `reigen_hermitian` device high-water mark to
# the 3 × (N_u × c) ComplexF64 matrices the algorithm nominally holds (the
# sketch, its image, and the rotation's destination). From bench/cost_model.jl.
const RSVD_PEAK_FUDGE = 1.554
# The LinearMaps composition's per-apply temporaries, in N_u-vectors: the two
# inclusions, the two clips, and the two halves of the asymmetrization. Trial E2
# replaces this estimate with a measurement.
const GILA_N_TEMPORARIES = 6
# Gila's CUFFT plans hold work areas the residency plan cannot see, so we have to
# declare them or the buffer pool hands out memory the operator takes for itself
# mid-sweep. Flat, since the work area does not scale with N_u the way the
# temporaries do. Trial E2 replaces this too.
const CUFFT_WORKSPACE_BYTES = 512 * 2^20

"""
    use_dense_path(N_u; max_N_u=DENSE_EXACT_MAX_N_U) -> Bool

Whether the universe is small enough to diagonalize exactly. Takes priority over
both RSVD paths.
"""
use_dense_path(N_u::Integer; max_N_u::Integer=DENSE_EXACT_MAX_N_U) = N_u <= max_N_u

"""
    use_panel_path(N_u, c, compute_env) -> Bool

Whether the `reigen_hermitian` sketch has outgrown the device, in which case the
tall matrices go to Funicular's panel storage. The cost model calls this too, so
that it and the run agree on which regime a job is in.
"""
function use_panel_path(N_u::Integer, c::Integer, compute_env::ComputeEnvironment)
    use_gpu(compute_env) || return false
    return RSVD_PEAK_FUDGE * 3 * Int(N_u) * Int(c) * 16 > device_budget_bytes()
end

"""
    gila_workspace_bytes(N_u) -> Int

Estimated device memory the `Asym(G⁰ᵤᵣ)` composition needs for itself while it is
applied: `$(GILA_N_TEMPORARIES)` N_u-vectors of ComplexF64 for its temporaries,
plus $(CUFFT_WORKSPACE_BYTES >> 20) MiB for Gila's CUFFT plan work areas. A
LinearMaps composition cannot carry `Funicular.workspace_bytes` as a trait, so
this goes to the plan as a keyword instead. Trial E2 measures the real number and
replaces this estimate.
"""
gila_workspace_bytes(N_u::Integer) = GILA_N_TEMPORARIES * Int(N_u) * 16 + CUFFT_WORKSPACE_BYTES

function generate_rsvd()
    @info string(now()) * " [rsvd::generate_rsvd] Starting RSVD generation"
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [generate_green::generate_green] Using GPU acceleration on device $(gpu_device(compute_env))"
        if !haskey(ENV, "CC_CLUSTER") # This breaks on compute canada
            CUDA.device!(gpu_device(compute_env))
        end
    else
        @info string(now()) * " [generate_green::generate_green] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _generate_rsvd_sr(compute_env, smr, rsvd_params)
        @info string(now()) * " [rsvd::generate_rsvd] Completed RSVD generation for SR system"
        return nothing
    end
    _generate_rsvd_smr(compute_env, smr, rsvd_params)
    @info string(now()) * " [rsvd::generate_rsvd] Completed RSVD generation for SMR system"
    return nothing
end

# function asym_ur(G₀_uu::VacuumGreenOperator, smr::SMRSystem)
#     s = sender(smr)
#     r = receiver(smr)
#     G₀_uu.mem.srcVol == G₀_uu.mem.trgVol || error("G₀_uu is not a self operator")
#     union_volume = G₀_uu.mem.srcVol # srcVol == trgVol
#     if union(s, r) != union_volume
#         @error "union_volume should be union(s, r) but it is not"
#     end
#     sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
#     sender_mask = fix_mask.(sender_mask)
#     receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume
#     receiver_mask = fix_mask.(receiver_mask)
#     disjoint_union_projector_action(x_union::AbstractArray{ComplexF64, 4}) = begin
#         x = similar(x_union)
#         fill!(x, zero(eltype(x)))
#         copyto!(view(x, sender_mask..., :), view(x_union, sender_mask..., :))
#         copyto!(view(x, receiver_mask..., :), view(x_union, receiver_mask..., :))
#         # The output x now has nonzero entries only in the sender and receiver regions
#         # that is, we've zeroed out the gap between s and r
#         return x
#     end
#     vec_disjoint_union_projector_action!(w, v) = begin
#         v_tens = reshape(v, glaSze(G₀_uu)[2])
#         out_tens = disjoint_union_projector_action(v_tens)
#         copyto!(w, vec(out_tens))
#         return w
#     end
#     u_projector = LinearMap{ComplexF64}(vec_disjoint_union_projector_action!, vec_disjoint_union_projector_action!, size(G₀_uu)...; ismutating=true)
#     r_projector_action(x_union::AbstractArray{ComplexF64, 4}) = begin
#         x = similar(x_union)
#         fill!(x, zero(eltype(x)))
#         copyto!(view(x, receiver_mask..., :), view(x_union, receiver_mask..., :))
#         # The output x now has nonzero entries only in the receiver region
#         return x
#     end
#     vec_r_projector_action!(w, v) = begin
#         v_tens = reshape(v, glaSze(G₀_uu)[2])
#         out_tens = r_projector_action(v_tens)
#         copyto!(w, vec(out_tens))
#         return w
#     end
#     r_projector = LinearMap{ComplexF64}(vec_r_projector_action!, vec_r_projector_action!, size(G₀_uu)...; ismutating=true)
#     G₀ = LinearMap(G₀_uu)
#     return 1/(2im) * (u_projector * G₀ * r_projector - r_projector * G₀' * u_projector)
# end

"""
    asym_self(G₀) -> LinearMap

`Asym(G₀)` of a self operator, as a `LinearMap`.

Gila's `AsyGlaOprVac` and `AsyGlaCmpOprVac`, and `AsyCmpBlkOprVac` for a refined
pair's block assembly, fold the antisymmetrization into the Fourier coefficients
and cost one Green apply instead of two. Where none of them has a method, this
falls back to `(X - X')/2im`, which is the same operator at twice the cost.

The folded form reads `Asym(G₀)` off as the entrywise imaginary part, which is the
antisymmetrization only for a complex-symmetric `G₀`. That premise is what makes
the operator having to be a self operator load bearing rather than cosmetic, and
every `asym` above throws on an external or adjoint one rather than returning a
wrong answer, so the `hasmethod` check is safe to dispatch on.

The two forms are not the same matrix: they differ by the quadrature's complex
symmetry defect, and the folded one is the better of the two. It keeps the
positive semidefiniteness the continuum operator has, where the difference form
turns that defect into a small negative eigenvalue that the bounds' pencil
whitener is sensitive to.
"""
function asym_self(G₀)
    hasmethod(GilaElectromagnetics.GilaOperators.asym, Tuple{typeof(G₀)}) &&
        return LinearMap(GilaElectromagnetics.GilaOperators.asym(G₀))
    return asym(LinearMap(G₀))
end

function asym_ur(G₀_rs::AbstractGlaOpr, G₀_rr::AbstractGlaOpr, smr::SMRSystem)
    # We want to compute the action of Asym(G⁰ᵣᵤ), but this operator is ambiguously defined. Here we write out it's definition.
    # Let ιᵣ be the inclusion of the receiver region into the universe, then define
    # Ĝ⁰ᵣᵤ = ιᵣ G⁰ᵣᵤ which is an operator that maps from the universe to the universe, but is zero outside of the receiver region:
    # Ĝ⁰ᵣᵤ = [0 0; G⁰ᵣₛ G⁰ᵣᵣ]
    # We can now formally define Asym(G⁰ᵣᵤ) = Asym(Ĝ⁰ᵣᵤ) = (Ĝ⁰ᵣᵤ - Ĝ⁰ᵣᵤ')/(2im):
    # Asym(G⁰ᵣᵤ) = [0 -1/2im G⁰ᵣₛ'; 1/2im G⁰ᵣₛ Asym(G⁰ᵣᵣ)]
    # Here it is in code:
    sender_size = dof_length(sender_mesh(smr))
    receiver_size = dof_length(receiver_mesh(smr))

    # Define ιᵣ: the inclusion of the receiver region into the universe
    receiver_inclusion_action!(r_included_in_u::AbstractVector{ComplexF64}, r_only::AbstractVector{ComplexF64}) = begin
        fill!(r_included_in_u, zero(eltype(r_included_in_u)))
        # our convention is [sender; receiver]
        copyto!(view(r_included_in_u, sender_size .+ (1:receiver_size)), r_only)
        return r_included_in_u
    end

    # Define ιₛ: the inclusion of the sender region into the universe
    sender_inclusion_action!(s_included_in_u::AbstractVector{ComplexF64}, s_only::AbstractVector{ComplexF64}) = begin
        fill!(s_included_in_u, zero(eltype(s_included_in_u)))
        # our convention is [sender; receiver]
        copyto!(view(s_included_in_u, 1:sender_size), s_only)
        return s_included_in_u
    end

    # Define the clipping operator that takes a vector defined on the universe and extracts only the receiver part
    receiver_clip_action!(r_clipped::AbstractVector{ComplexF64}, u_vec::AbstractVector{ComplexF64}) = begin
        # our convention is [sender; receiver]
        copyto!(r_clipped, view(u_vec, sender_size .+ (1:receiver_size)))
        return r_clipped
    end

    # Define the clipping operator that takes a vector defined on the universe and extracts only the sender part
    sender_clip_action!(s_clipped::AbstractVector{ComplexF64}, u_vec::AbstractVector{ComplexF64}) = begin
        # our convention is [sender; receiver]
        copyto!(s_clipped, view(u_vec, 1:sender_size))
        return s_clipped
    end

    receiver_inclusion = LinearMap{ComplexF64}(receiver_inclusion_action!, receiver_clip_action!, sender_size + receiver_size, receiver_size; ismutating=true)
    sender_inclusion = LinearMap{ComplexF64}(sender_inclusion_action!, sender_clip_action!, sender_size + receiver_size, sender_size; ismutating=true)
    receiver_clip = LinearMap{ComplexF64}(receiver_clip_action!, receiver_inclusion_action!, receiver_size, sender_size + receiver_size; ismutating=true)
    sender_clip = LinearMap{ComplexF64}(sender_clip_action!, sender_inclusion_action!, sender_size, sender_size + receiver_size; ismutating=true)

    G₀_rs = LinearMap(G₀_rs) # Maps from sender to receiver
    Π_s = sender_clip # Projects from universe to sender
    Π_r = receiver_clip # Projects from universe to receiver
    ι_r = receiver_inclusion # Includes receiver into universe

    asym_G₀_rr = asym_self(G₀_rr) # half the cost of (G₀_rr - G₀_rr')/(2im) where Gila has the method
    asym_G₀_ru = -(asym(ι_r * G₀_rs * Π_s) + ι_r * asym_G₀_rr * Π_r) # minus sign becuase the paper uses (-G⁰ᵣᵤ)ᵃ everywhere instead of( G⁰ᵣᵤ)ᵃ
    positive_seeder = -asym(ι_r * G₀_rs * Π_s) # -1 for the same reason
    return asym_G₀_ru, positive_seeder
end

function uu_disjoint_union(G₀_uu::VacuumGreenOperator, smr::SMRSystem)
    s = sender(smr)
    r = receiver(smr)
    G₀_uu.mem.srcVol == G₀_uu.mem.trgVol || error("G₀_uu is not a self operator")
    union_volume = G₀_uu.mem.srcVol # srcVol == trgVol
    if union(s, r) != union_volume
        @error "union_volume should be union(s, r) but it is not"
    end
    sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
    sender_mask = fix_mask.(sender_mask)
    receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume
    receiver_mask = fix_mask.(receiver_mask)
    disjoint_union_projector_action(x_union::AbstractArray{ComplexF64, 4}) = begin
        x = similar(x_union)
        fill!(x, zero(eltype(x)))
        copyto!(view(x, sender_mask..., :), view(x_union, sender_mask..., :))
        copyto!(view(x, receiver_mask..., :), view(x_union, receiver_mask..., :))
        # The output x now has nonzero entries only in the sender and receiver regions
        # that is, we've zeroed out the gap between s and r
        return x
    end
    vec_disjoint_union_projector_action!(w, v) = begin
        v_tens = reshape(v, glaSze(G₀_uu)[2])
        out_tens = disjoint_union_projector_action(v_tens)
        copyto!(w, vec(out_tens))
        return w
    end
    projector = LinearMap{ComplexF64}(vec_disjoint_union_projector_action!, vec_disjoint_union_projector_action!, size(G₀_uu)...; ismutating=true)
    G₀ = LinearMap(G₀_uu)
    return projector * G₀ * projector
end

# function crop_to_receiver(v::AbstractVector, smr::SMRSystem, total_volume::GlaVol)
#     r = receiver(smr)
#     receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, total_volume)
#     v_tens = reshape(v, total_volume.cel..., :)
#     cropped_tens = view(v_tens, receiver_mask..., :)
#     return vec(cropped_tens)
# end

"""
    ur_asym_vectors_path(compute_env, smr) -> String

Where the positive-Γ eigenvectors go when they are streamed to disk rather than
written into the JLD. One HDF5 file per experiment, chunked one panel per chunk,
next to the JLD that names it.
"""
ur_asym_vectors_path(compute_env::ComputeEnvironment, smr::SMRSystem) =
    joinpath(scratch_dir(compute_env), "$(file_prefix(smr))_UR_asym_Vpos.h5")

# Complete means `D` (all returned eigenvalues) and `num_pos` are present and the
# positive vectors are reachable, either inline in the JLD or in the h5 file the
# JLD names. A run that wrote the values but died before the vectors landed is
# not complete.
function _ur_asym_is_complete(jld_path::String, jld_key::String, vectors_path::String)
    ispath(jld_path) || return false
    jld = jldopen(jld_path, "r")
    try
        (haskey(jld, jld_key * "D") && haskey(jld, jld_key * "num_pos")) || return false
        haskey(jld, jld_key * "V_pos") && return true
        haskey(jld, jld_key * "vectors_file") || return false
        return isfile(joinpath(dirname(vectors_path), jld[jld_key*"vectors_file"]))
    finally
        close(jld)
    end
end

# The number of positive eigenvalues, which is the `m` everything downstream is
# sized by. `reigen_hermitian` and the dense branch both sort descending, so the
# positives are a prefix and `m` is all bounds needs to slice them out. We check
# that here rather than assume it.
function _num_positive(evals::AbstractVector)
    issorted(evals, rev=true) || error("the eigenvalues are not sorted in descending order, so the positive ones are not a prefix and num_pos does not describe them")
    m = count(>(zero(eltype(evals))), evals)
    m == 0 || all(>(zero(eltype(evals))), view(evals, 1:m)) || error("the $(m) positive eigenvalues are not the leading $(m) entries")
    return m
end

"""
    materialize_columns(f::PanelFactored, cols) -> PanelMatrix

Columns `cols` of the product `f.Q * f.C`, formed without building the whole
product. `MatrixFreeRandomizedLinearAlgebra.materialize` builds all `k` columns,
but we only ever want the positive-Γ prefix, and at 4 λ the difference is tens of
terabytes over a sweep (FUNICULAR_PLAN.md, workstream B5).
"""
function materialize_columns(f::PanelFactored, cols)
    C = f.C[:, cols]
    V = similar(f.Q, eltype(f.Q), (size(f.Q, 1), size(C, 2)))
    Funicular.rightmul!(V, f.Q, C) # V = Q * C[:, cols]
    return V
end

# Assembles the operator densely by applying it to the identity, one column at a
# time. The operator is matrix-free, so this costs `size(op, 2)` matvecs, which is
# only affordable because this branch is gated on a small universe. The assembled
# matrix is a host array either way, since it goes straight to LAPACK.
#
# The columns have to go through one at a time as vectors: LinearMaps has no true
# matrix application for an N-ary FunctionMap composition. It re-composes the
# intermediate result (`op_i * X` is a CompositeMap, not a product) and materializes
# it through `convert(AbstractMatrix, ...)`, which allocates a host `Matrix`
# wherever the operands live. With CuArray-backed blocks inside `op` that mixes
# host and device memory and dies in BLAS. The vector path takes its intermediates
# from `similar(x)`, which is what every other GPU path here already relies on.
function _dense_matrix(op, on_gpu::Bool)
    rows, cols = size(op)
    T = ComplexF64
    M = Matrix{T}(undef, rows, cols)
    e = zeros(T, cols)
    x = on_gpu ? CuVector{T}(undef, cols) : e
    y = on_gpu ? CuVector{T}(undef, rows) : Vector{T}(undef, rows)
    for j in 1:cols
        e[j] = one(T)
        on_gpu && copyto!(x, e)
        mul!(y, op, x)
        copyto!(view(M, :, j), Array(y))
        e[j] = zero(T)
    end
    return M
end

# The exact spectrum of the assembled operator, descending. `eigen!` runs on the
# host: at the sizes this branch takes (1/4 λ is N_u = 3,072) it is seconds, and
# the eigenvectors go into the JLD as a host array anyway. The vectors come back
# as a permuted view, not a permuted copy, so only the positive block the caller
# slices out is materialized.
function _dense_hermitian_eigen(M::Matrix{ComplexF64})
    F = eigen!(Hermitian(M)) # ascending, as LAPACK returns it
    idxs = sortperm(F.values, rev=true)
    return F.values[idxs], view(F.vectors, :, idxs)
end

function _save_ur_asym(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams;
                       plan_override=nothing, max_dense_N_u::Integer=DENSE_EXACT_MAX_N_U)
    jld_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    vectors_path = ur_asym_vectors_path(compute_env, smr)
    jld_key = "UR_asym/"

    if _ur_asym_is_complete(jld_path, jld_key, vectors_path)
        @info string(now()) * " [rsvd::_save_ur_asym] RSVD for $(jld_key) already exists at $(jld_path): skipping"
        return
    end

    @info string(now()) * " [rsvd::_save_ur_asym] Computing RSVD for UR_asym"
    @info string(now()) * " [rsvd::_save_ur_asym] Loading G₀ operators"
    G₀_rs = load_green_function(compute_env, smr, Receiver, Sender) # sender -> receiver
    G₀_rr = load_green_function(compute_env, smr, Receiver, Receiver) # receiver -> receiver
    G₀_ur_asym, _positive_seeder = asym_ur(G₀_rs, G₀_rr, smr)
    # We tried seeding the rSVD with the off-diagonal block's range (`seed_Q`),
    # but it didn't help and just costs us an extra rSVD, so we skip it.

    N_u = size(G₀_ur_asym, 1)
    c = rank(rsvd_params)
    seed_value = seed(rsvd_params)

    if use_dense_path(N_u; max_N_u=max_dense_N_u)
        @info string(now()) * " [rsvd::_save_ur_asym] Path: dense-exact (N_u = $(N_u) ≤ $(max_dense_N_u)). No RSVD error: the whole spectrum is computed exactly"
        M = _dense_matrix(G₀_ur_asym, use_gpu(compute_env))
        evals, V = _dense_hermitian_eigen(M)
        M = nothing
        m = _num_positive(evals)
        _log_positive_fraction(m, length(evals))
        @info string(now()) * " [rsvd::_save_ur_asym] Saving exact eigendecomposition to $(jld_path)"
        _save_ur_asym_components(jld_path, jld_key, evals, m, seed_value, true; V_pos=view(V, :, 1:m))
        run_gc()
        return
    end

    plan = plan_override === nothing ?
           (use_panel_path(N_u, c, compute_env) ? residency_plan(compute_env; workspace_bytes=gila_workspace_bytes(N_u)) : nothing) :
           plan_override

    if plan === nothing
        @info string(now()) * " [rsvd::_save_ur_asym] Path: in-memory RSVD (the $(N_u) × $(c) sketch fits on the device)"
        sample_vec = zeros(ComplexF64, 0)
        if use_gpu(compute_env)
            sample_vec = CuArray(sample_vec)
        end
        @info string(now()) * " [rsvd::_save_ur_asym] Computing $(c) components of a randomized eigen decomposition for a $(size(G₀_ur_asym)) Hermitian operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
        out = reigen_hermitian(G₀_ur_asym, c; num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
        evals = Array(out.values)
        m = _num_positive(evals)
        _log_positive_fraction(m, length(evals))
        @info string(now()) * " [rsvd::_save_ur_asym] Saving reigen to $(jld_path)"
        _save_ur_asym_components(jld_path, jld_key, evals, m, seed_value, false; V_pos=view(out.vectors, :, 1:m))
        run_gc()
        return
    end

    @info string(now()) * " [rsvd::_save_ur_asym] Path: panel RSVD (the $(N_u) × $(c) sketch does not fit on the device)" plan
    @info string(now()) * " [rsvd::_save_ur_asym] Computing $(c) components of a randomized eigen decomposition for a $(size(G₀_ur_asym)) Hermitian operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations, seed $(seed_value)"
    out = reigen_hermitian(G₀_ur_asym, c; num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), plan=plan, seed=seed_value, factored=true, validate=true)
    evals = Array(out.values)
    m = _num_positive(evals)
    _log_positive_fraction(m, length(evals))
    m > 0 || error("no positive eigenvalues of Asym(G⁰ᵤᵣ) were found, so there is no basis to save and the bounds have nothing to run on")

    # `factored=true` left the product Q * rotation unevaluated, so only the
    # positive prefix is ever formed: an N_u × m panel matrix instead of N_u × c.
    @info string(now()) * " [rsvd::_save_ur_asym] Forming the $(N_u) × $(m) positive block and streaming it to $(vectors_path)"
    V_pos = materialize_columns(out.vectors, 1:m)
    try
        Funicular.save(V_pos, vectors_path)
    finally
        Funicular.free!(V_pos)
        Funicular.free!(out.vectors.Q)
    end

    #=
    Drop everything that roots the plan before returning. `free!` only hands the
    blocks back to the plan's own pinned slab pool -- Funicular has no way to give
    a slab back to the OS -- so the pool stays charged to the cgroup for as long as
    anything can still reach the plan. `V_pos` and `out.vectors.Q` are
    `PanelMatrix`es and each holds `plan` as a field, so the plan outlives this
    frame unless they go too. The caller runs `reclaim_host_pools!` once the frame
    is gone; nulling here is what makes that collection possible.
    =#
    V_pos = nothing
    out = nothing
    plan = nothing

    @info string(now()) * " [rsvd::_save_ur_asym] Saving reigen to $(jld_path)"
    _save_ur_asym_components(jld_path, jld_key, evals, m, seed_value, false; vectors_file=basename(vectors_path))
    run_gc()
    return
end

# Both the bounds' time (∝ m⁴) and the sweep's disk usage hang off the positive
# fraction. The cost model assumes 0.6, so we print the measured value: a run that
# comes back well above it shows up in the RSVD log rather than in a blown bounds
# job.
function _log_positive_fraction(m::Int, total::Int)
    @info string(now()) * " [rsvd::_save_ur_asym] num_pos = $(m) of $(total) directions (positive fraction $(round(m / total; digits=3)))"
end

"""
    _save_ur_asym_components(jld_path, jld_key, evals, num_pos, seed_value, exact; V_pos=nothing, vectors_file=nothing)

Writes the `UR_asym/` group. `D` is every eigenvalue the solve returned, in
descending order, and `num_pos` is how many of them are positive, which is the `m`
that slices the basis. Exactly one of `V_pos` (the N_u × m block, inline) and
`vectors_file` (the basename of the h5 the block was streamed to) is given. The
legacy full `V` key is not written: at 4 λ its negative-Γ half is 7 TB of vectors
nothing reads.
"""
function _save_ur_asym_components(jld_path::String, jld_key::String, evals::AbstractVector, num_pos::Int, seed_value::Int, exact::Bool; V_pos=nothing, vectors_file=nothing)
    if jld_path == ""
        @info string(now()) * " [rsvd::_save_ur_asym_components] Empty jld_path provided: skipping save"
        return
    end
    jld = jldopen(jld_path, "a+")
    try
        _save_component(jld, jld_key * "D", Array(evals))
        _save_scalar(jld, jld_key * "num_pos", num_pos)
        _save_scalar(jld, jld_key * "seed", seed_value)
        _save_scalar(jld, jld_key * "exact", exact)
        if V_pos !== nothing
            @info string(now()) * " [rsvd::_save_ur_asym_components] Saving the positive-Γ eigenvectors"
            _save_component(jld, jld_key * "V_pos", V_pos)
        end
        if vectors_file !== nothing
            _save_scalar(jld, jld_key * "vectors_file", vectors_file)
        end
    finally
        close(jld)
    end
    return
end

function _save_constraint_asym(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    jld_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_key = "constraint_asym/"
    if ispath(jld_path)
        jld = jldopen(jld_path, "r")
        if haskey(jld, "$(jld_key)/D") && haskey(jld, "$(jld_key)/V")
            @info string(now()) * " [rsvd::generate_rsvd] RSVDs for constraint_asym already exist at $(jld_path): skipping"
            close(jld)
            return
        else
            close(jld)
        end
    end
    @info string(now()) * " [rsvd::generate_rsvd] Computing eigendecomposition for Asym(χ⁻¹ - G₀_rr)"
    @info string(now()) * " [rsvd::generate_rsvd] Loading G₀_rr operator"
    G₀_rr = load_green_function(compute_env, smr, Receiver, Receiver) # receiver -> receiver
    sample_vec = zeros(ComplexF64, 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end
    χ = susceptibility(smr)
    A = imag(inv(χ))*I - asym(LinearMap(G₀_rr))
    @info string(now()) * " [rsvd::generate_rsvd] Computing $(rank(rsvd_params)) components of a randomized eigen decomposition for a $(size(A)) Hermitian operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
    out = reigen_hermitian(A, rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
    @info string(now()) * " [rsvd::generate_rsvd] Saving constraint asymmetry reigen to $(jld_path)"
    _save_reigen_hermitian(out.vectors, out.values, jld_path, "constraint_asym/")
end

function _generate_rsvd_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [rsvd::generate_rsvd] Starting RSVD generation for SR system"

    @info string(now()) * " [rsvd::generate_rsvd] Computing UR asym RSVD"
    _save_ur_asym(compute_env, smr, rsvd_params)

    #=
    Two panel decompositions run in this one process, and each of them builds its
    own `ResidencyPlan`. A Funicular host pool never returns a slab to the OS, so
    the UR_asym pool is still page-locked and still charged to the cgroup when the
    RS plan is built, and a second plan sized for the whole allocation is a
    guaranteed OOM (the 4 λ probe, `--mem=124G`, died here). `_save_ur_asym` has
    dropped its references by now, so this frame is the first place the plan is
    actually collectable; `residency_plan` budgets around whatever survives.
    =#
    @info string(now()) * " [rsvd::generate_rsvd] Reclaiming the UR_asym residency plan before the next decomposition"
    reclaim_host_pools!()

    @info string(now()) * " [rsvd::generate_rsvd] Computing RSVD for RS"
    _run_rsvdvals(compute_env, smr, rsvd_params, "RS/")

    # @info string(now()) * " [rsvd::generate_rsvd] Computing RSVD for Asym(χ⁻¹ - G₀_rr)"
    # _save_constraint_asym(compute_env, smr, rsvd_params)

    # @info string(now()) * " [rsvd::generate_rsvd] Computing RSVD for G₀_uu"
    # _run_rsvd(compute_env, smr, rsvd_params, "UU/")

    # χ = susceptibility(smr)
    # G₀_uu = load_green_function(compute_env, smr, Design, Design)
    # Ga = asym(I*inv(χ) - LinearMap(G₀_uu))
    # sample_vec = zeros(ComplexF64, 0)
    # if use_gpu(compute_env)
    #     sample_vec = CuArray(sample_vec)
    # end
    # out = reigen_hermitian(Ga, rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
    # _save_reigen_hermitian(out.vectors, out.values, joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld"), "A")

    # @info string(now()) * " [rsvd::generate_rsvd] Computing RSVD for Asym(G₀_uu)"
    # G₀_uu_union = load_green_function(compute_env, smr, Design, Design)
    # G₀_uu_disjoint = uu_disjoint_union(G₀_uu_union, smr) # For the universe -> universe case, we need to zero out the gap to get a basis that is more useful in the bounds
    # sample_vec = zeros(ComplexF64, 0)
    # if use_gpu(compute_env)
    #     sample_vec = CuArray(sample_vec)
    # end
    # out = reigen_hermitian(asym(G₀_uu_disjoint), rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
    # _save_reigen_hermitian(out.vectors, out.values, joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld"), "UU_asym/")
end

function _run_rsvd(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams, jld_key::String)
    fname = file_prefix(smr)
    jld_path = joinpath(scratch_dir(compute_env), "$(fname).jld")
    if ispath(jld_path)
        jld = jldopen(jld_path, "r")
        if haskey(jld, jld_key * "U") && haskey(jld, jld_key * "D") && haskey(jld, jld_key * "V")
            @info string(now()) * " [rsvd::_run_rsvd] RSVD for $(jld_key) already exists at $(jld_path): skipping"
            close(jld)
            return
        else
            close(jld)
        end
    end

    @info string(now()) * " [rsvd::_run_rsvd] Computing RSVD for $(jld_key)"
    @info string(now()) * " [rsvd::_run_rsvd] Loading G₀ operator"
    target = char2volume_symbol(jld_key[1]) # First character indicates target
    source = char2volume_symbol(jld_key[2]) # Second character indicates source
    G₀_ab = load_green_function(compute_env, smr, target, source)
    if target == Design && source == Design
        @info string(now()) * " [rsvd::_run_rsvd] Applying disjoint union projector to G₀_uu for universe -> universe case"
        G₀_ab = uu_disjoint_union(G₀_ab, smr) # For the universe -> universe case, we need to zero out the gap to get a basis that is more useful in the bounds
    end
    sample_vec = zeros(ComplexF64, 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end

    @info string(now()) * " [rsvd::_run_rsvd] Computing $(rank(rsvd_params)) components of a randomized SVD for a $(size(G₀_ab)) operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
    out = rsvd(LinearMap(G₀_ab), rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)

    @info string(now()) * " [rsvd::_run_rsvd] Saving RSVD to $(jld_path)"
    _save_rsvd(out, jld_path, jld_key)

    run_gc()
end

function _run_rsvdvals(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams, jld_key::String;
                       plan_override=nothing, max_dense_N_r::Integer=DENSE_EXACT_MAX_N_R)
    fname = file_prefix(smr)
    jld_path = joinpath(scratch_dir(compute_env), "$(fname).jld")
    if ispath(jld_path)
        jld = jldopen(jld_path, "r")
        if haskey(jld, jld_key * "D")
            @info string(now()) * " [rsvd::_run_rsvdvals] RSVD for $(jld_key) already exists at $(jld_path): skipping"
            close(jld)
            return
        else
            close(jld)
        end
    end

    @info string(now()) * " [rsvd::_run_rsvdvals] Computing RSVD for $(jld_key)"
    @info string(now()) * " [rsvd::_run_rsvdvals] Loading G₀ operator"
    target = char2volume_symbol(jld_key[1]) # First character indicates target
    source = char2volume_symbol(jld_key[2]) # Second character indicates source
    G₀_ab = load_green_function(compute_env, smr, target, source)
    if target == Design && source == Design
        @info string(now()) * " [rsvd::_run_rsvdvals] Applying disjoint union projector to G₀_uu for universe -> universe case"
        G₀_ab = uu_disjoint_union(G₀_ab, smr) # For the universe -> universe case, we need to zero out the gap to get a basis that is more useful in the bounds
    end
    G₀_ab = LinearMap(G₀_ab)
    dims = size(G₀_ab)
    N_r = minimum(dims)
    c = rank(rsvd_params)

    # Same three-way split as the UR_asym branch, but on the smaller side of the
    # operator. The rsvdvals peak is two N × c matrices, half the reigen peak, so
    # this is never the binding path; it uses the same plan anyway.
    if N_r <= max_dense_N_r
        @info string(now()) * " [rsvd::_run_rsvdvals] Path: dense-exact singular values for $(jld_key) (min side $(N_r) ≤ $(max_dense_N_r))"
        out = svdvals!(_dense_matrix(G₀_ab, use_gpu(compute_env)))
        @info string(now()) * " [rsvd::_run_rsvdvals] Saving exact singular values to $(jld_path)"
        _save_rsvd(out, jld_path, jld_key)
        run_gc()
        return
    end

    plan = plan_override === nothing ?
           (use_panel_path(maximum(dims), c, compute_env) ? residency_plan(compute_env; workspace_bytes=gila_workspace_bytes(maximum(dims))) : nothing) :
           plan_override

    if plan === nothing
        @info string(now()) * " [rsvd::_run_rsvdvals] Path: in-memory RSVD for $(jld_key)"
        sample_vec = zeros(ComplexF64, 0)
        if use_gpu(compute_env)
            sample_vec = CuArray(sample_vec)
        end
        @info string(now()) * " [rsvd::_run_rsvdvals] Computing $(c) components of a randomized SVD for a $(dims) operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
        out = rsvdvals(G₀_ab, c; num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
    else
        @info string(now()) * " [rsvd::_run_rsvdvals] Path: panel RSVD for $(jld_key)" plan
        @info string(now()) * " [rsvd::_run_rsvdvals] Computing $(c) components of a randomized SVD for a $(dims) operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations, seed $(seed(rsvd_params))"
        out = rsvdvals(G₀_ab, c; num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), plan=plan, seed=seed(rsvd_params))
        # `rsvdvals_panel` frees both bases itself and returns a plain host vector,
        # so nothing but this local still roots the plan. Drop it so the pinned
        # slabs can be collected once this frame is gone; see `reclaim_host_pools!`.
        plan = nothing
    end

    @info string(now()) * " [rsvd::_run_rsvdvals] Saving RSVD to $(jld_path)"
    _save_rsvd(out, jld_path, jld_key)

    run_gc()
end

function _generate_rsvd_smr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [rsvd::generate_rsvd] Starting RSVD generation for SMR system"
    jld_keys = ["RS/", "SM/", "MR/"]
    for jld_key in jld_keys
        @info string(now()) * " [rsvd::_generate_rsvd_smr] Processing $(jld_key)"
        _run_rsvd(compute_env, smr, rsvd_params, jld_key)
        reclaim_host_pools!()
    end
    _run_rsvdvals(compute_env, smr, rsvd_params, "MM/")
end

function _save_component(jld::JLD2.JLDFile, key::String, component::AbstractArray)
    if haskey(jld, key)
        @info string(now()) * " [rsvd::_save_component] $(key) already exists: skipping"
    else
        @info string(now()) * " [rsvd::_save_component] Saving $(key)"
        jld[key] = Array(component) # Ensure the data is copied to the host
    end
end

# Same skip-and-log behaviour as `_save_component`, for the metadata scalars
# (`num_pos`, `seed`, `exact`, `vectors_file`) that are not arrays.
function _save_scalar(jld::JLD2.JLDFile, key::String, value)
    if haskey(jld, key)
        @info string(now()) * " [rsvd::_save_scalar] $(key) already exists: skipping"
    else
        @info string(now()) * " [rsvd::_save_scalar] Saving $(key) = $(value)"
        jld[key] = value
    end
end

function _save_rsvd(factorization::SVD, jld_path::String, jld_key::String)
    if jld_path == ""
        @info string(now()) * " [rsvd::_save_rsvd] Empty jld_path provided: skipping save"
        return
    end
    U, S, Vt = factorization.U, factorization.S, factorization.Vt
    jld = jldopen(jld_path, "a+")

    @info string(now()) * " [rsvd::_save_rsvd] Saving left singular vectors"
    _save_component(jld, jld_key * "V", U)

    @info string(now()) * " [rsvd::_save_rsvd] Saving singular values"
    _save_component(jld, jld_key * "D", S)

    @info string(now()) * " [rsvd::_save_rsvd] Saving right singular vectors"
    _save_component(jld, jld_key * "U", Vt')

    close(jld)
end

function _save_rsvd(vals::AbstractVector, jld_path::String, jld_key::String)
    if jld_path == ""
        @info string(now()) * " [rsvd::_save_rsvd] Empty jld_path provided: skipping save"
        return
    end
    S = Array(vals)
    jld = jldopen(jld_path, "a+")

    @info string(now()) * " [rsvd::_save_rsvd] Saving singular values"
    _save_component(jld, jld_key * "D", S)

    close(jld)
end

function _save_reigen_hermitian(vectors::AbstractMatrix, values::AbstractVector, jld_path::String, jld_key::String)
    if jld_path == ""
        @info string(now()) * " [rsvd::_save_reigen_hermitian] Empty jld_path provided: skipping save"
        return
    end
    jld = jldopen(jld_path, "a+")

    @info string(now()) * " [rsvd::_save_reigen_hermitian] Saving eigenvectors"
    _save_component(jld, jld_key * "V", vectors)

    @info string(now()) * " [rsvd::_save_reigen_hermitian] Saving eigenvalues"
    _save_component(jld, jld_key * "D", values)

    close(jld)
end
