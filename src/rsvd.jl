using MatrixFreeRandomizedLinearAlgebra
using LinearMaps
using LinearAlgebra
using JLD2
using CUDA
using GilaElectromagnetics
import Funicular

# There are three ways to get the spectrum of Asym(G⁰ᵤᵣ), in order of priority:
#
#   1. dense-exact, when the universe is small enough to build and diagonalize
#      the whole operator. No RSVD error, and the "rank" is the full spectrum.
#   2. in-memory RSVD, whenever the sketch fits on the device.
#   3. panel RSVD (Funicular), above that.

# 1/4 λ at x-scale 1/32 is N_u = 3,072, so the cap covers it with room to spare.
const DENSE_EXACT_MAX_N_U = 12_288
# The RS values come from a rectangular block, which is a full matrix rather than a triangle.
const DENSE_EXACT_MAX_N_R = 6_144
# Measured ratio of the in-memory `reigen_hermitian` device high-water mark to the
# 3 × (N_u × c) ComplexF64 matrices it nominally holds.
const RSVD_PEAK_FUDGE = 1.554
# The operator's per-apply temporaries, in N_u-vectors, and Gila's FFT work areas,
# which the residency plan cannot see and has to be told about.
const GILA_N_TEMPORARIES = 6
const FFT_WORKSPACE_BYTES = 512 * 2^20

use_dense_path(N_u::Integer; max_N_u::Integer=DENSE_EXACT_MAX_N_U) = N_u <= max_N_u

"Whether the `reigen_hermitian` sketch has outgrown the device, so the tall matrices go to Funicular panels."
function use_panel_path(N_u::Integer, c::Integer, compute_env::ComputeEnvironment; n_mats::Integer=3)
    use_gpu(compute_env) || return false
    return RSVD_PEAK_FUDGE * n_mats * Int(N_u) * Int(c) * 16 > device_budget_bytes()
end

"Estimated device memory the Green operator needs for itself while it is applied."
gila_workspace_bytes(N_u::Integer) = GILA_N_TEMPORARIES * Int(N_u) * 16 + FFT_WORKSPACE_BYTES

function generate_rsvd()
    args = parse_cli()
    select_device(args.compute_env)
    if isnothing(mediator(args.smr))
        _generate_rsvd_sr(args.compute_env, args.smr, args.rsvd_params)
    else
        _generate_rsvd_smr(args.compute_env, args.smr, args.rsvd_params)
    end
    @info string(now()) * " [rsvd::generate_rsvd] Done"
end

antiherm(X) = (X - X') / (2im)

"""
    asym_ur(G₀_rs, asym_G₀_rr, smr) -> LinearMap

The action of Asym(G⁰ᵣᵤ) on the universe `[sender; receiver]`. With ιᵣ the inclusion of
the receiver into the universe, Ĝ⁰ᵣᵤ = ιᵣ G⁰ᵣᵤ = [0 0; G⁰ᵣₛ G⁰ᵣᵣ], and

    Asym(G⁰ᵣᵤ) = Asym(Ĝ⁰ᵣᵤ) = [0  -G⁰ᵣₛ'/2i; G⁰ᵣₛ/2i  Asym(G⁰ᵣᵣ)].

Returned with a minus sign, because the paper works with (−G⁰ᵣᵤ)ᵃ throughout.
`G₀_rs` and `asym_G₀_rr` (Asym(G⁰ᵣᵣ) itself) come from `green_map` / `asym_green_map`.
"""
function asym_ur(G₀_rs, asym_G₀_rr, smr::SMRSystem)
    sender_size = dof_length(sender_mesh(smr))
    receiver_size = dof_length(receiver_mesh(smr))
    N_u = sender_size + receiver_size
    s_rows, r_rows = 1:sender_size, sender_size .+ (1:receiver_size)

    include_r!(u, r) = (fill!(u, zero(eltype(u))); copyto!(view(u, r_rows), r); u)
    include_s!(u, s) = (fill!(u, zero(eltype(u))); copyto!(view(u, s_rows), s); u)
    clip_r!(r, u) = copyto!(r, view(u, r_rows))
    clip_s!(s, u) = copyto!(s, view(u, s_rows))

    ι_r = LinearMap{ComplexF64}(include_r!, clip_r!, N_u, receiver_size; ismutating=true)
    Π_s = LinearMap{ComplexF64}(clip_s!, include_s!, sender_size, N_u; ismutating=true)
    Π_r = LinearMap{ComplexF64}(clip_r!, include_r!, receiver_size, N_u; ismutating=true)
    return -(antiherm(ι_r * LinearMap(G₀_rs) * Π_s) + ι_r * LinearMap(asym_G₀_rr) * Π_r)
end

rsvd_output_path(compute_env::ComputeEnvironment, smr::SMRSystem) =
    joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")

"Where the panel path streams the eigenvectors, next to the JLD that names it."
ur_asym_vectors_path(compute_env::ComputeEnvironment, smr::SMRSystem) =
    joinpath(scratch_dir(compute_env), "$(file_prefix(smr))_UR_asym_V.h5")

# Complete means `D` and `num_pos` are present and the vectors are reachable, inline in
# the JLD or in the h5 file the JLD names. The positive-only `V_pos`/`vectors_file` of
# older runs do not count: the bounds need the negative-Γ eigenvectors too.
function _ur_asym_is_complete(jld_path::String, vectors_path::String)
    isfile(jld_path) || return false
    return jldopen(jld_path, "r") do jld
        (haskey(jld, "UR_asym/D") && haskey(jld, "UR_asym/num_pos")) || return false
        haskey(jld, "UR_asym/V") && return true
        haskey(jld, "UR_asym/V_file") &&
            isfile(joinpath(dirname(vectors_path), jld["UR_asym/V_file"]))
    end
end

# `m`, the number of positive eigenvalues. Both solvers sort descending, so the
# positives are a prefix, which is what lets `m` alone slice them out downstream.
function _num_positive(evals::AbstractVector)
    issorted(evals, rev=true) || error("the eigenvalues are not sorted in descending order")
    m = count(>(0), evals)
    @info string(now()) * " [rsvd::_num_positive] num_pos = $(m) of $(length(evals)) (positive fraction $(round(m / length(evals); digits=3)))"
    return m
end

"Columns `cols` of `f.Q * f.C`, without forming the whole product."
function materialize_columns(f::PanelFactored, cols)
    C = f.C[:, cols]
    V = similar(f.Q, eltype(f.Q), (size(f.Q, 1), size(C, 2)))
    Funicular.rightmul!(V, f.Q, C)
    return V
end

# The operator assembled on the host, one column per matvec. LinearMaps has no
# device-safe matrix apply for a FunctionMap composition, so the columns go through
# one at a time as vectors.
function _dense_matrix(op, on_gpu::Bool)
    rows, cols = size(op)
    M = Matrix{ComplexF64}(undef, rows, cols)
    e = zeros(ComplexF64, cols)
    x = on_gpu ? CuVector{ComplexF64}(undef, cols) : e
    y = on_gpu ? CuVector{ComplexF64}(undef, rows) : Vector{ComplexF64}(undef, rows)
    for j in 1:cols
        e[j] = 1
        on_gpu && copyto!(x, e)
        mul!(y, op, x)
        copyto!(view(M, :, j), Array(y))
        e[j] = 0
    end
    return M
end

# Green applies per apply of `asym_ur`: G⁰ᵣₛ and its adjoint, plus Asym(G⁰ᵣᵣ), counted
# as two (its difference form) to keep the tolerance the production runs passed with.
const UR_ASYM_GREEN_APPLIES = 4

"""
    adjoint_check_rtol(N, green_applies) -> Float64

Tolerance for the `⟨y, Gx⟩ = ⟨G'y, x⟩` probe on an operator of size `N` that costs
`green_applies` Green applies. Funicular's default, `64 eps √N`, is calibrated for an
elementary operator; each Green apply is two FFT passes whose rounding adds in
quadrature, hence `√(2 · green_applies)`. The extra 4× covers what was measured on the
gap-refined 2 λ points (misses of up to 1.96× with eleven digits agreeing); a wrong
adjoint disagrees at O(1) and is still caught.
"""
adjoint_check_rtol(N::Integer, green_applies::Integer) =
    4 * 64 * sqrt(2 * green_applies) * eps(Float64) * sqrt(float(N))

"""
    _check_operator_normwise(G; backend, rtol, n=3) -> true

`Funicular.check_operator`'s adjoint, panel and Hermitian probes with the inner products
scaled by `‖y‖‖Gx‖` instead of `|⟨y, Gx⟩|`, which for random probes is about √N
smaller and turns a fixed rounding error into spurious failures. Run here, with
`validate=false` passed on, because Funicular takes no `rtol` from the RSVD calls.
"""
function _check_operator_normwise(G; backend, rtol, n::Integer=3)
    rows, cols = Funicular.operator_size(G)
    T = eltype(G)
    probe(m) = Funicular.todevice(backend, Funicular.randprobe(T, m, n))
    apply(A, Z) = (local AZ = Funicular.alloc_device(backend, T, (Funicular.operator_size(A)[1], n));
                   fill!(AZ, zero(T)); Funicular.columnwise_apply!(AZ, A, Z); AZ)
    X, Y = probe(cols), probe(rows)
    GX = apply(G, X)
    compare(other, what) = for c in 1:n
        y, gx = view(Y, :, c), view(GX, :, c)
        left, right = dot(y, gx), dot(view(other, :, c), view(X, :, c))
        abs(left - right) <= rtol * norm(y) * norm(gx) ||
            throw(ArgumentError("$(what) check failed on probe $c: ⟨y, Gx⟩ = $left, other side = $right, |diff|/(‖y‖‖Gx‖) = $(abs(left - right) / (norm(y) * norm(gx))) > $rtol"))
    end
    compare(apply(adjoint(G), Y), "adjoint")
    if Funicular.panel_capable(G)
        P = Funicular.alloc_device(backend, T, (rows, n)); fill!(P, zero(T))
        mul!(P, G, X)
        isapprox(P, GX; rtol=rtol, atol=rtol) || throw(ArgumentError("the panel apply of G does not match its column apply to rtol $rtol"))
    end
    Funicular.ishermitian_op(G) && compare(apply(G, Y), "Hermitian")
    return true
end

"""
    _save_ur_asym(compute_env, smr, rsvd_params; plan_override=nothing, max_dense_N_u=DENSE_EXACT_MAX_N_U)

The Asym(G⁰ᵤᵣ) eigendecomposition, written to `UR_asym/` in the point's JLD: every
eigenvalue (`D`), the positive count (`num_pos`), and one eigenvector per eigenvalue in
the same descending order, inline (`V`) or streamed to an h5 file (`V_file`). `plan_override` and
`max_dense_N_u` let tests force a path.
"""
function _save_ur_asym(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams;
                       plan_override=nothing, max_dense_N_u::Integer=DENSE_EXACT_MAX_N_U)
    jld_path = rsvd_output_path(compute_env, smr)
    vectors_path = ur_asym_vectors_path(compute_env, smr)
    if _ur_asym_is_complete(jld_path, vectors_path)
        @info string(now()) * " [rsvd::_save_ur_asym] UR_asym already exists at $(jld_path): skipping"
        return
    end

    G₀_ur_asym = asym_ur(green_map(compute_env, smr, Receiver, Sender),
                         asym_green_map(compute_env, smr, Receiver), smr)
    N_u = size(G₀_ur_asym, 1)
    c = min(rank(rsvd_params), N_u)

    if use_dense_path(N_u; max_N_u=max_dense_N_u)
        @info string(now()) * " [rsvd::_save_ur_asym] Path: dense-exact (N_u = $(N_u))"
        F = eigen!(Hermitian(_dense_matrix(G₀_ur_asym, use_gpu(compute_env))))
        idxs = sortperm(F.values, rev=true)
        evals = F.values[idxs]
        m = _num_positive(evals)
        _save_ur_asym_components(jld_path, evals, m, seed(rsvd_params), true; V=F.vectors[:, idxs])
        return
    end

    plan = !isnothing(plan_override) ? plan_override :
           use_panel_path(N_u, c, compute_env) ? residency_plan(compute_env; workspace_bytes=gila_workspace_bytes(N_u)) : nothing
    if plan === nothing
        @info string(now()) * " [rsvd::_save_ur_asym] Path: in-memory RSVD, $(c) components of $(N_u) × $(N_u)"
        sample_vec = use_gpu(compute_env) ? CuArray(zeros(ComplexF64, 0)) : zeros(ComplexF64, 0)
        out = reigen_hermitian(G₀_ur_asym, c; num_oversamples=oversamples(rsvd_params),
                               num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
        evals = Array(out.values)
        m = _num_positive(evals)
        _save_ur_asym_components(jld_path, evals, m, seed(rsvd_params), false; V=out.vectors)
        return
    end

    @info string(now()) * " [rsvd::_save_ur_asym] Path: panel RSVD, $(c) components of $(N_u) × $(N_u), seed $(seed(rsvd_params))" plan
    _check_operator_normwise(G₀_ur_asym; backend=plan.backend,
                             rtol=adjoint_check_rtol(N_u, UR_ASYM_GREEN_APPLIES))
    out = reigen_hermitian(G₀_ur_asym, c; num_oversamples=oversamples(rsvd_params),
                           num_power_iterations=power_iter(rsvd_params), plan=plan,
                           seed=seed(rsvd_params), factored=true, validate=false)
    evals = Array(out.values)
    m = _num_positive(evals)
    m > 0 || error("Asym(G⁰ᵤᵣ) has no positive eigenvalue, so the bounds have no basis to run on")
    V = materialize_columns(out.vectors, 1:length(evals))
    try
        Funicular.save(V, vectors_path)
    finally
        Funicular.free!(V)
        Funicular.free!(out.vectors.Q)
    end
    # The panel matrices hold the plan; drop them so `reclaim_host_pools!` can collect it.
    V = out = plan = nothing
    _save_ur_asym_components(jld_path, evals, m, seed(rsvd_params), false; V_file=basename(vectors_path))
end

# Writes `UR_asym/` from scratch: a group left by a run that died part way through
# could pair new eigenvalues with old vectors.
function _save_ur_asym_components(jld_path::String, evals::AbstractVector, num_pos::Int,
                                  seed_value::Int, exact::Bool; V=nothing, V_file=nothing)
    jldopen(jld_path, "a+") do jld
        haskey(jld, "UR_asym") && delete!(jld, "UR_asym")
        isnothing(V) || (jld["UR_asym/V"] = Array(V))
        isnothing(V_file) || (jld["UR_asym/V_file"] = V_file)
        jld["UR_asym/seed"] = seed_value
        jld["UR_asym/exact"] = exact
        jld["UR_asym/D"] = Array(evals)
        jld["UR_asym/num_pos"] = num_pos # last: `_ur_asym_is_complete` keys on it
    end
end

function _generate_rsvd_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    _save_ur_asym(compute_env, smr, rsvd_params)
    # The UR_asym panel plan's pinned host pool is collectable only now that its frame is gone.
    reclaim_host_pools!()
    _run_rsvdvals(compute_env, smr, rsvd_params, "RS/")
end

# The blocks the mediator bound reads: G⁰ᵣ←[s;m] and G⁰ₘₛ with their vectors, and the
# vacuum G⁰ᵣₛ singular values the bound is compared with.
function _generate_rsvd_smr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    _run_rsvd_vectors(compute_env, smr, rsvd_params, "RU/",
                      () -> green_map(compute_env, smr, [Receiver], [Sender, Mediator]))
    reclaim_host_pools!()
    _run_rsvd_vectors(compute_env, smr, rsvd_params, "MS/", () -> green_map(compute_env, smr, Mediator, Sender))
    reclaim_host_pools!()
    _run_rsvdvals(compute_env, smr, rsvd_params, "RS/")
end

_has_key(jld_path, key) = isfile(jld_path) && jldopen(jld -> haskey(jld, key), jld_path, "r")

# The block a two-letter key names: target first, then source.
_key_operator(compute_env, smr, jld_key) =
    green_map(compute_env, smr, char2volume_symbol(jld_key[1]), char2volume_symbol(jld_key[2]))

const STORE_COLUMNS = 256

"A tall matrix (host, device, or a Funicular `PanelMatrix`) to an h5 store at `path`."
save_tall(V::Funicular.PanelMatrix, path::AbstractString) = Funicular.save(V, path)
function save_tall(V::AbstractMatrix, path::AbstractString)
    store = Funicular.open_store(path, "w"; N=size(V, 1), k=size(V, 2), w=min(STORE_COLUMNS, size(V, 2)))
    try
        for J in Iterators.partition(axes(V, 2), STORE_COLUMNS)
            Funicular.write_panel!(Funicular.DiskHome(store, first(J):last(J)), Array(view(V, :, J)))
        end
    finally
        Funicular.close_store!(store)
    end
    return path
end

"Columns `J` (a unit range) of the tall matrix in the h5 store at `path`, on the host."
function read_columns(path::AbstractString, J::UnitRange{Int})
    store = Funicular.open_store(path, "r")
    try
        staged = Matrix{store.stored}(undef, store.N, length(J))
        Funicular.read_panel!(Funicular.DiskHome(store, J), staged)
        return staged isa Matrix{ComplexF64} ? staged : Matrix{ComplexF64}(staged)
    finally
        Funicular.close_store!(store)
    end
end

store_size(path::AbstractString) = (s = Funicular.open_store(path, "r"); try (s.N, s.k) finally Funicular.close_store!(s) end)

# Gaussian probes of the residual. Halko, Martinsson & Tropp (2011), Eq. (4.3): for any B,
# ‖B‖ ≤ 10√(2/π) maxᵢ ‖Bωᵢ‖ fails with probability at most 10⁻ʳ. Applied to B = (EEᴴ)^q E,
# whose norm is ‖E‖^(2q+1), the factor shrinks to its (2q+1)-th root and the probes see
# the top of the tail rather than its Frobenius mass.
const RESIDUAL_PROBES = 10
const RESIDUAL_POWER = 3

"""
    residual_bound(G, U_path, on_gpu; q=RESIDUAL_POWER) -> Float64

A bound on ‖E‖ = ‖(1 − UUᴴ)G‖ that holds with probability 1 − 10⁻¹⁰, `U` the orthonormal
left vectors in the store at `U_path`. For an RSVD, U diag(S) Vᴴ = UUᴴG, so E is the
residual of the decomposition. Costs `RESIDUAL_PROBES · (2q + 1)` applies of G or Gᴴ.
"""
function residual_bound(G, U_path::AbstractString, on_gpu::Bool; q::Int=RESIDUAL_POWER)
    k = store_size(U_path)[2]
    function project!(Y)
        for J in Iterators.partition(1:k, STORE_COLUMNS)
            UJ = read_columns(U_path, first(J):last(J))
            on_gpu && (UJ = CuArray(UJ))
            Y .-= UJ * (UJ' * Y)
        end
        return Y
    end
    Ω = randn(ComplexF64, size(G, 2), RESIDUAL_PROBES)
    Y = project!(opmat(G, on_gpu ? CuArray(Ω) : Ω))
    for _ in 1:q
        Y = project!(opmat(G, opmat(G', Y)))
    end
    return (10 * sqrt(2 / π) * maximum(norm.(eachcol(Array(Y)))))^(1 / (2q + 1))
end

"""
    _run_rsvd_vectors(compute_env, smr, rsvd_params, jld_key, operator)

Full RSVD of the block `operator()` builds, in memory or in Funicular panels: `D` in the
point's JLD, the left and right vectors in h5 stores beside it (`U_file`, `V_file`), and
`eps`, the [`residual_bound`](@ref) of the decomposition.
"""
function _run_rsvd_vectors(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams,
                           jld_key::String, operator)
    jld_path = rsvd_output_path(compute_env, smr)
    if _has_key(jld_path, jld_key * "eps")
        @info string(now()) * " [rsvd::_run_rsvd_vectors] $(jld_key) already exists at $(jld_path): skipping"
        return
    end
    G = operator()
    dims = size(G)
    c = min(rank(rsvd_params), minimum(dims))
    stem = joinpath(scratch_dir(compute_env), "$(file_prefix(smr))_$(rstrip(jld_key, '/'))")
    U_path, V_path = stem * "_U.h5", stem * "_V.h5"
    plan = use_panel_path(maximum(dims), c, compute_env) ?
           residency_plan(compute_env; workspace_bytes=gila_workspace_bytes(maximum(dims))) : nothing
    if plan === nothing
        @info string(now()) * " [rsvd::_run_rsvd_vectors] Path: in-memory RSVD, $(c) components of $(jld_key) ($(dims))"
        sample_vec = use_gpu(compute_env) ? CuArray(zeros(ComplexF64, 0)) : zeros(ComplexF64, 0)
        out = rsvd(G, c; num_oversamples=oversamples(rsvd_params),
                   num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
        S = Array(out.S)
        save_tall(out.U, U_path)
        save_tall(copy(Array(out.Vt)'), V_path)
    else
        @info string(now()) * " [rsvd::_run_rsvd_vectors] Path: panel RSVD, $(c) components of $(jld_key) ($(dims))" plan
        _check_operator_normwise(G; backend=plan.backend, rtol=adjoint_check_rtol(maximum(dims), 1))
        out = rsvd(G, c; num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params),
                   plan=plan, seed=seed(rsvd_params), validate=false)
        S = Array(out.S)
        try
            save_tall(out.U, U_path)
            save_tall(out.V, V_path)
        finally
            Funicular.free!(out.U)
            Funicular.free!(out.V)
        end
    end
    out = plan = nothing
    run_gc()
    use_gpu(compute_env) && CUDA.reclaim()
    ε = residual_bound(G, U_path, use_gpu(compute_env))
    @info string(now()) * " [rsvd::_run_rsvd_vectors] $(jld_key): σ[1] = $(S[1]), σ[$(length(S))] = $(S[end]), residual bound ε = $(ε)"
    jldopen(jld_path, "a+") do jld
        jld[jld_key * "D"] = S
        jld[jld_key * "U_file"] = basename(U_path)
        jld[jld_key * "V_file"] = basename(V_path)
        jld[jld_key * "eps"] = ε # last: the skip above keys on it
    end
end

# Singular values of one Green block: dense-exact, in-memory or panel, as for UR_asym.
function _run_rsvdvals(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams, jld_key::String;
                       plan_override=nothing, max_dense_N_r::Integer=DENSE_EXACT_MAX_N_R,
                       operator=() -> _key_operator(compute_env, smr, jld_key))
    jld_path = rsvd_output_path(compute_env, smr)
    if _has_key(jld_path, jld_key * "D")
        @info string(now()) * " [rsvd::_run_rsvdvals] $(jld_key) already exists at $(jld_path): skipping"
        return
    end
    G₀_ab = operator()
    dims = size(G₀_ab)
    c = min(rank(rsvd_params), minimum(dims))

    out = if minimum(dims) <= max_dense_N_r
        @info string(now()) * " [rsvd::_run_rsvdvals] Path: dense-exact singular values of $(jld_key) ($(dims))"
        svdvals!(_dense_matrix(G₀_ab, use_gpu(compute_env)))
    else
        plan = !isnothing(plan_override) ? plan_override :
               # a square block keeps every tall matrix at full height: OOMed at 3 for 393216², fine at 196608²
               use_panel_path(maximum(dims), c, compute_env; n_mats=4) ?
               residency_plan(compute_env; workspace_bytes=gila_workspace_bytes(maximum(dims))) : nothing
        if plan === nothing
            @info string(now()) * " [rsvd::_run_rsvdvals] Path: in-memory RSVD, $(c) values of $(jld_key) ($(dims))"
            sample_vec = use_gpu(compute_env) ? CuArray(zeros(ComplexF64, 0)) : zeros(ComplexF64, 0)
            rsvdvals(G₀_ab, c; num_oversamples=oversamples(rsvd_params),
                     num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)
        else
            @info string(now()) * " [rsvd::_run_rsvdvals] Path: panel RSVD, $(c) values of $(jld_key) ($(dims))" plan
            _check_operator_normwise(G₀_ab; backend=plan.backend, rtol=adjoint_check_rtol(maximum(dims), 1))
            rsvdvals(G₀_ab, c; num_oversamples=oversamples(rsvd_params),
                     num_power_iterations=power_iter(rsvd_params), plan=plan,
                     seed=seed(rsvd_params), validate=false)
        end
    end
    jldopen(jld -> (jld[jld_key * "D"] = Array(out)), jld_path, "a+")
    run_gc()
end
