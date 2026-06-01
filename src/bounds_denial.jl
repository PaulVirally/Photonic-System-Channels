using GilaElectromagnetics
using CUDA
using Dates
using LinearAlgebra
using Roots
using Optim
using LinearMaps
using Base.Threads
using Plots
using KrylovKit
using NLopt
using Random
using .Projectors
using MatrixFreeRandomizedLinearAlgebra

function qthin!(A::AbstractMatrix)
    F = qr!(A)
    n = min(size(A)...)
    A .= F.Q[:, 1:n]
    return A
end

function qthin!(A::CuMatrix)
    n = min(size(A)...)
    τ = similar(A, n)
    CUDA.CUSOLVER.geqrf!(A, τ)
    CUDA.CUSOLVER.orgqr!(A, τ)
    return A
end

function ordering_idxs(Γ::AbstractVector{Float64})
    @info string(now()) * " [bounds_denial::ordering_idxs] Forming the indexing set by ordering γ⁰ᵃᵤᵣ"
    sorted_idxs = sortperm(Γ; rev=true)
    return sorted_idxs
end

function read_array(jld, key::AbstractString, use_gpu::Bool)
    @info string(now()) * " [bounds_denial::read_array] Reading array for key '$key' from JLD file"
    if haskey(jld, key)
        arr = jld[key]
        if use_gpu
            @info string(now()) * " [bounds_denial::read_array] Moving array for key '$key' to GPU"
            arr = CuArray(arr)
        end
        return arr
    else
        error("Key $key not found in JLD file")
    end
end

function projected_operators(G₀_uu::VacuumGreensOperator, smr::SMRSystem, env::ComputeEnvironment)
    s = sender(smr)
    r = receiver(smr)
    G₀_uu.mem.srcVol == G₀_uu.mem.trgVol || error("G₀_uu is not a self operator")
    union_volume = G₀_uu.mem.srcVol # srcVol == trgVol
    if union(s, r) != union_volume
        @error "union_volume should be union(s, r) but it is not"
    end
    sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
    receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume

    # Zero out the gap between sender and receiver by creating a projector that keeps the sender and receiver regions but zeros out everything else in the union volume (including the gap between s and r)
    disjoint_union_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    if use_gpu(env)
        disjoint_union_indicator = CuArray(disjoint_union_indicator)
    end
    fill!(view(disjoint_union_indicator, sender_mask..., :), one(eltype(disjoint_union_indicator)))
    fill!(view(disjoint_union_indicator, receiver_mask..., :), one(eltype(disjoint_union_indicator)))
    disjoint_union_projector_action!(w, v) = begin
        w .= vec(disjoint_union_indicator .* reshape(v, size(disjoint_union_indicator)))
        return w
    end
    u_projector = LinearMap{ComplexF64}(disjoint_union_projector_action!, disjoint_union_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    sender_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    if use_gpu(env)
        sender_indicator = CuArray(sender_indicator)
    end
    fill!(view(sender_indicator, sender_mask..., :), one(eltype(sender_indicator)))
    s_projector_action!(w, v) = begin
        w .= vec(sender_indicator .* reshape(v, size(sender_indicator)))
        return w
    end
    s_projector = LinearMap{ComplexF64}(s_projector_action!, s_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    receiver_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    if use_gpu(env)
        receiver_indicator = CuArray(receiver_indicator)
    end
    fill!(view(receiver_indicator, receiver_mask..., :), one(eltype(receiver_indicator)))
    r_projector_action!(w, v) = begin
        w .= vec(receiver_indicator .* reshape(v, size(receiver_indicator)))
        return w
    end
    r_projector = LinearMap{ComplexF64}(r_projector_action!, r_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    G₀ = LinearMap(G₀_uu)
    return r_projector, s_projector, u_projector, (u_projector * G₀ * u_projector)
end

function region_idxs(G₀_uu::VacuumGreensOperator, smr::SMRSystem)
    s = sender(smr)
    r = receiver(smr)
    G₀_uu.mem.srcVol == G₀_uu.mem.trgVol || error("G₀_uu is not a self operator")
    union_volume = G₀_uu.mem.srcVol # srcVol == trgVol
    if union(s, r) != union_volume
        @error "union_volume should be union(s, r) but it is not"
    end
    sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
    receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume
    disjoint_union_mask = ntuple(i -> union(sender_mask[i], receiver_mask[i]), 3) # Mask for the union of sender and receiver regions, excluding the gap between them
    return sender_mask, receiver_mask, disjoint_union_mask
end

opmat(A::LinearMap, b::AbstractVector) = A*b

function opmat(A::LinearMap, B::AbstractMatrix)
    T = promote_type(eltype(A), eltype(B))
    out = similar(B, T, size(A, 1), size(B, 2))
    tmp = similar(out, T, size(A, 1))
    @views for j in axes(B, 2)
        mul!(tmp, A, B[:, j])
        out[:, j] .= tmp
    end
    return out
end

function opmat(A::LinearMap, vecs::AbstractVector{<:AbstractVector})
    T = promote_type(eltype(A), eltype(vecs))
    out = similar(vecs, T, size(A, 1), length(vecs))
    tmp = similar(out, T, size(A, 1))
    @views for j in axes(vecs, 1)
        mul!(tmp, A, vecs[j])
        out[:, j] .= tmp
    end
    return out
end

function _compute_bounds_smr(::ComputeEnvironment, ::SMRSystem, ::RSVDParams)
    @info string(now()) * " [bounds_denial::_compute_bounds_smr] Computing bounds for SMR system"
    throw("Not implemented yet")
end

function global_asym_multiplier_bound(lhs_op::LinearMap, rhs_op::LinearMap, lhs_basis::AbstractMatrix, rhs_basis::AbstractMatrix, basis::AbstractMatrix, ζ::Float64; num_fs_vals::Int=15, initial_regularization::Float64=1e-14, max_regularization::Float64=1e-6)
    # Compute generalized eigenvalues of (lhs, rhs) to get a bound on the global asym multiplier

    # RSVD First
    basis_vals, basis_vecs = eigen(Hermitian(lhs_basis), Hermitian(rhs_basis))
    basis_value = maximum(basis_vals)
    initial_guess = basis * view(basis_vecs, :, argmax(basis_vals))
    basis_multiplier_bound = ζ^2 * basis_value / (ζ * basis_value + 1)
    @info string(now()) * " [bounds_denial::global_asym_multiplier_bound] RSVD predicts α > $(basis_multiplier_bound) for global asym using (L, R)"

    # Full space
    regularization = initial_regularization
    success = false
    fs_multiplier_bound = -Inf
    while regularization < max_regularization
        @info string(now()) * " [bounds_denial::global_asym_multiplier_bound] Solving the (L, R) generalized eigenvalue problem in the full space with regularization $regularization"
        try
            vals, vecs, _ = geneigsolve(v -> (lhs_op*v, rhs_op*v .+ regularization .* v), initial_guess, num_fs_vals, :LR; tol=1e-7, krylovdim=280, maxiter=5001, ishermitian=true, isposdef=true, orth=KrylovKit.ModifiedGramSchmidt2())
            fs_value = maximum(vals)
            fs_multiplier_bound = ζ^2 * fs_value / (ζ * fs_value + 1)
            @info string(now()) * " [bounds_denial::global_asym_multiplier_bound] Full space generalized eigenvalue problem forces α > $(fs_multiplier_bound) using (L, R) for global asym with regularization $regularization"

            success = true

            @info string(now()) * " [bounds_denial::global_asym_multiplier_bound] Enriching the basis with the eigenvectors from the generalized eigenvalue problem"
            basis = hcat(basis, vecs...)

            @error "make this faster"
            basis = qthin!(basis) # Orthogonalize a first time
            basis = qthin!(basis) # Orthgonalize a second time because I am a paranoid little boi
            # @info string(now()) * " [bounds_denial::global_asym_multiplier_bound] Re-forming the dense matrices in the enriched basis"
            # lhs_basis = sym(basis' * opmat(lhs_op, basis))
            # rhs_basis = sym(basis' * opmat(rhs_op, basis))

            break
        catch e
            if !(e isa PosDefException)
                rethrow(e)
            end
            @error string(now()) * " [bounds_denial::global_asym_multiplier_bound] Generalized eigenvalue problem for (L, R) did not converge with regularization $regularization; trying again with higher regularization" e
            if iszero(regularization)
                regularization = 1e-15
            else
                regularization *= 10
            end
        end
    end
    if !success
        fs_multiplier_bound = basis_multiplier_bound
        @warn string(now()) * " [bounds_denial::_global_asym_multiplier_bound] Failed to solve the generalized eigenvalue problem in the full space; using the RSVD-based bound of α > $(fs_multiplier_bound) for global asym instead"
    end
    return fs_multiplier_bound, basis
end

mutable struct ObjectiveCache{T<:Number, MAT<:AbstractMatrix{T}, VEC<:AbstractVector{T}}
    G0_ur_asym::LinearMap{T}
 
    # rSVD eigendecomposition: G^{vac a}_{ur} ≈ Vur_asym * Diagonal(Γ) * Vur_asym^†.
    Γ::Vector{Float64}
    Vur_asym::MAT
    positive_indicator::VEC # length r; 1 in positive eigenspace of G0_ur_asym, 0 elsewhere
 
    # Deflation state, preallocated to capacity `cap`.
    V::MAT # N × cap; first n columns are orthonormal
    Y::MAT # N × cap; Y[:, 1:n] = A * V[:, 1:n]
    H::MAT # cap × cap; H[1:n, 1:n] = V^† Y, Hermitian
    n::Int # current number of deflation directions
    cap::Int
 
    # Matvec workspace (preallocated, length cap).
    a_buf::VEC # holds V^† v
    b_buf::VEC # holds Y^† v
    c_buf::VEC # holds H * (V^† v)
end

function similar_fill(arr::AbstractArray, size, val::T) where T
    new_arr = similar(arr, T, size)
    fill!(new_arr, val)
    return new_arr
end

function ObjectiveCache(G0_ur_asym::LinearMap{T}, Γ::Vector{Float64}, Vur_asym::AbstractMatrix{T}, capacity::Int; support_tol::Float64=1e-12) where T
    N = size(G0_ur_asym, 1)
    r = length(Γ)
    size(Vur_asym) == (N, r) || throw(DimensionMismatch( "Vur_asym is $(size(Vur_asym)); expected ($N, $r)"))
    size(G0_ur_asym, 2) == N || throw(DimensionMismatch( "neg_Ga_ur is not square: $(size(G0_ur_asym))"))
    capacity > 0 || throw(ArgumentError("capacity must be positive"))
 
    V = similar_fill(Vur_asym, (N, capacity), zero(T))
    Y = similar_fill(Vur_asym, (N, capacity), zero(T))
    H = similar_fill(Vur_asym, (capacity, capacity), zero(T))
    a_buf = similar_fill(Vur_asym, capacity, zero(T))
    b_buf = similar_fill(Vur_asym, capacity, zero(T))
    c_buf = similar_fill(Vur_asym, capacity, zero(T))
 
    positive_indicator_host = T.(Γ .> support_tol)
    positive_indicator = similar(Vur_asym, r)
    copyto!(positive_indicator, positive_indicator_host)
 
    return ObjectiveCache{T, typeof(V), typeof(a_buf)}(
        G0_ur_asym, Γ, Vur_asym, positive_indicator,
        V, Y, H, 0, capacity,
        a_buf, b_buf, c_buf,
    )
end

function objective_operator(cache::ObjectiveCache{T}) where T
    # The operator is (I - Pₖ) * G0_ur_asym * (I - Pₖ) where Pₖ = Vₖ * Vₖ^† is the orthogonal projector onto the current deflation subspace
    N = size(cache.G0_ur_asym, 1)
    forward!(out, v) = begin
        mul!(out, cache.G0_ur_asym, v)
        n = cache.n
        if iszero(n)
            # No deflation directions yet, so the operator is just G0_ur_asym
            return out
        end
 
        Vk = view(cache.V, :, 1:n)
        Yk = view(cache.Y, :, 1:n)
        Hk = view(cache.H, 1:n, 1:n)
        a  = view(cache.a_buf, 1:n)
        b  = view(cache.b_buf, 1:n)
        c  = view(cache.c_buf, 1:n)
 
        # Three small matvecs into preallocated buffers.
        mul!(a, Vk', v)
        mul!(b, Yk', v)
        mul!(c, Hk, a)
 
        # out = out - Y a + V (H a - Y^† v)
        mul!(out, Yk, a, -one(T), one(T)) # out -= Y a
        @. c -= b # c = H a - Y^† v
        mul!(out, Vk, c, one(T), one(T)) # out += V c
        return out
    end
    return LinearMap{T}(forward!, forward!, N, N; ismutating=true, ishermitian=true)
end

function dominant_eigenvector_deflated_kernel(cache::ObjectiveCache{T}) where T
    Vur = cache.Vur_asym
    Γ = cache.Γ
    n = cache.n
 
    if iszero(n)
        # operator is G0_ur_asym so dominant direction is the first eigenvector of G0_ur_asym
        # Γ is sorted descending, so this is column 1 of Vur_asym.
        out = similar(Vur, size(Vur, 1))
        out .= view(Vur, :, 1)
        return out
    end
 
    # Coefficients of V_k in eigenbasis: C = Vur^† V_k  (r × n)
    C = Vur' * view(cache.V, :, 1:n)
 
    # Γ is real, so (Diag(Γ) C C^†)^† = C C^† Diag(Γ).
    CCt = C * C'
    DCCt = Diagonal(CuArray(Γ)) * CCt
    M_eig = Diagonal(CuArray(Γ)) - DCCt - DCCt' + CCt * DCCt # M = Diag(Γ) - Diag(Γ) C C^† - C C^† Diag(Γ) + C C^† Diag(Γ) C C^†
 
    # Pull to host for eigen
    M_host = Array(sym(M_eig)) # symmetrize explicitly to be safe
    F = eigen(Hermitian(M_host))
    α_host = F.vectors[:, argmax(F.values)]
 
    α = similar(Vur, size(Vur, 2))
    copyto!(α, α_host)
    return Vur * α # back to N-space
end

function update_projector!(cache::ObjectiveCache{T}, w_candidate::Union{AbstractVector,Nothing}=nothing) where T
    cache.n < cache.cap || error("[bounds_denial::update_projector!] Deflation cache full (n = $(cache.n) / cap = $(cache.cap))")
 
    used_fallback = w_candidate === nothing
    if used_fallback
        @info string(now()) * " [bounds_denial::update_projector!] No optimizer provided; using dominant eigenvector of the deflated kernel as the next deflation direction"
        w = dominant_eigenvector_deflated_kernel(cache)
    else
        w = copy(w_candidate)
    end
 
    # Project onto positive eigenspace of G0_ur_asym.
    coefs = cache.Vur_asym' * w
    coefs .*= cache.positive_indicator # zero out any components in the non-positive eigenspace
    w_proj = cache.Vur_asym * coefs
 
    # Orthogonalize against V[:, 1:n] (modified Gram-Schmidt twice).
    num_gram_schmidt = 2
    if cache.n > 0
        Vk = view(cache.V, :, 1:cache.n)
        for _ in 1:num_gram_schmidt
            β = Vk' * w_proj
            mul!(w_proj, Vk, β, -one(T), one(T)) # w_proj -= Vk * β
        end
    end

    # Normalize (with safeguards against degenerate directions)
    nrm = norm(w_proj)
    if nrm < 1e-12
        if used_fallback
            error("[update_projector!] Fallback produced a degenerate direction (norm $nrm); this should basically never happen... 😢")
        end
        @warn string(now()) * " [bounds::update_projector!] Candidate direction has post-orthogonalization norm $nrm; falling back to dominant eigenvector of the deflated kernel"
        return update_projector!(cache, nothing)
    end
    ŵ = w_proj ./ nrm

    # Update V, Y, H with the new direction ŵ.
    cache.n += 1
    n = cache.n
    cache.V[:, n] .= ŵ
 
    y_new = similar(ŵ)
    mul!(y_new, cache.G0_ur_asym, ŵ)
    cache.Y[:, n] .= y_new
 
    if n > 1
        Vprev = view(cache.V, :, 1:n-1)
        h_off = Vprev' * y_new
        cache.H[1:n-1, n] .= h_off
        cache.H[n, 1:n-1] .= conj.(h_off)
    end
    CUDA.@allowscalar cache.H[n, n] = real(ŵ' * y_new) # Hermitian => diagonal is real

    @info string(now()) * " [bounds::update_projector!] n = $n / $(cache.cap) (used_fallback = $used_fallback)"
    return cache
end

function update_projector_classic!(cache::ObjectiveCache{T}) where T
    cache.n < cache.cap || error("[bounds_denial::update_projector_classic!] Cache full (n = $(cache.n))")
    next_idx = cache.n + 1
    next_idx <= size(cache.Vur_asym, 2) || error("[bounds_denial::update_projector_classic!] No more eigenvectors of G^{0,a}_{ur} to deflate (next_idx = $next_idx)")

    ŵ = view(cache.Vur_asym, :, next_idx)  # already unit-norm and orthogonal to V[:, 1:n]

    cache.n += 1
    n = cache.n
    cache.V[:, n] .= ŵ

    y_new = similar(cache.Y, size(cache.Y, 1))
    mul!(y_new, cache.G0_ur_asym, ŵ)
    cache.Y[:, n] .= y_new

    if n > 1
        Vprev = view(cache.V, :, 1:n-1)
        h_off = Vprev' * y_new
        cache.H[1:n-1, n] .= h_off
        cache.H[n, 1:n-1] .= conj.(h_off)
    end
    CUDA.@allowscalar cache.H[n, n] = real(ŵ' * y_new)

    @info string(now()) * " [bounds_denial::update_projector_classic!] Deflated Vur_asym[:, $next_idx]; n = $n / $(cache.cap)"
    return cache
end

function projector_from_vecs(vecs::Vector{<:AbstractVector})
    @info string(now()) * " [bounds_denial::projector_from_vecs] Building projector from $(length(vecs)) deflation directions"
    mat = stack(vecs)
    ortho_vecs = qthin!(mat)
    forward!(out, v) = begin
        out .= ortho_vecs * (ortho_vecs' * v)
    end
    return LinearMap(forward!, forward!, length(vecs[1]), length(vecs[1]); ismutating=true, ishermitian=true)
end

function gap_overlap(vec::AbstractVector, u_projector::LinearMap)
    # Measure how much the vector overlaps with the gap region
    gap_component = vec - u_projector * vec # Gap region = (I - U)
    return norm(gap_component) / norm(vec)
end

function _compute_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Computing bounds for SR system"

    jld_in_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_in = jldopen(jld_in_path, "r")

    Γ = read_array(jld_in, "UR_asym/D", use_gpu(compute_env))
    Γ .*= -one(eltype(Γ)) # Sign typo in the original notes
    Vur_asym = read_array(jld_in, "UR_asym/V", use_gpu(compute_env))
    jld_out = jldopen(joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld"), "w")

    # Sort the singular values and vectors according to the ordering_idxs
    sorted_idxs = ordering_idxs(Γ)
    Vur_asym = Vur_asym[:, sorted_idxs]
    Γ = Array(Γ[sorted_idxs])

    Γrs = jld_in["RS/D"]
    if !haskey(jld_out, "Γrs")
        jld_out["Γrs"] = Array(Γrs)
    end
    if !haskey(jld_out, "ordering_idxs")
        jld_out["ordering_idxs"] = Array(sorted_idxs)
    end
    close(jld_out)
    Γrs = nothing
    U_uu = read_array(jld_in, "UU/U", use_gpu(compute_env))
    # RSVD_BASIS_SIZE = size(U_uu, 2)
    RSVD_BASIS_SIZE = 256
    @info string(now()) * " [bounds_denial::_compute_bounds_sr] RSVD basis size: $(size(U_uu, 2)); Using RSVD_BASIS_SIZE = $RSVD_BASIS_SIZE"
    U_uu = U_uu[:, 1:RSVD_BASIS_SIZE]

    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Finished reading data from disk; closing JLD file and freeing memory"
    close(jld_in)
    GC.gc()
    GC.gc()
    GC.gc()

    G₀_uu = load_greens_function(compute_env, smr, Design, Design)
    r_projector, s_projector, u_projector, _ = projected_operators(G₀_uu, smr, compute_env)
    G0_ur_asym = asym(u_projector * LinearMap(G₀_uu) * r_projector)

    χ = susceptibility(smr)
    ζ = abs(χ)^2/imag(χ)
    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Susceptibility χ = $χ, material factor ζ = $ζ"

    basis = U_uu

    lhs_op = (imag(inv(χ))*u_projector - u_projector * asym(LinearMap(G₀_uu))) * u_projector # Im(χ⁻¹)I - Asym(G₀)
    rhs_op = s_projector + (ζ^2/4) * u_projector * G₀_uu' * s_projector * G₀_uu * u_projector + ζ * asym(s_projector * G₀_uu * u_projector) # S + (ζ^2/4) G₀' * S * G₀ + ζ * Asym(S * G₀ * U)

    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Forming the dense matrices in the initial RSVD basis"
    lhs_basis = basis' * (opmat(lhs_op, basis))
    rhs_basis = basis' * (opmat(rhs_op, basis))

    # We do not need to do every index. We know that after Γ < 0, the bound is 0, so we add 5 to have a few points to show this
    stop_at_k = min(findfirst(Γ .< zero(eltype(Γ))) + 5, rank(rsvd_params)) # TODO: this was true when we were removing eigenvectors of G0_ur_asym, but is this still true for this strategy?

    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Initializing the objective cache for the deflation process with capacity $stop_at_k"
    cache = ObjectiveCache(G0_ur_asym, Γ, Vur_asym, stop_at_k)

    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Computing a bound on the global asym multiplier using the initial RSVD basis and then refining it with a solve in the full space"
    global_asym_multiplier, basis = global_asym_multiplier_bound(lhs_op, rhs_op, lhs_basis, rhs_basis, basis, ζ; num_fs_vals=50) # From RSVD experiments, the optimal multiplier is super close to the boundary, so we'll just use the boundary since optimization is giga problematic
    global_asym_multiplier *= 0.99999 # Go ever so slightly inside the boundary for numerical stability
    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Using global asym multiplier of $global_asym_multiplier for the bounds computation"

    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Re-forming the dense matrices in the enriched basis"
    @error "make this faster"
    lhs_basis = basis' * opmat(lhs_op, basis) # Re-form the dense matrices in the enriched basis after finding the multiplier bound since the basis has changed
    rhs_basis = basis' * opmat(rhs_op, basis)

    scattering2power = (sqrt(4 * imag(χ)^2)/abs(χ))^2 * ζ # The conversion factor from scattering units to power units for the bounds
    power_bounds = ones(stop_at_k) * NaN

    initial_guess = nothing

    # regularization = 1e-14
    @info string(now()) * " [bounds_denial::_compute_bounds_sr] Starting the iterative deflation process to compute bounds for k ∈ 1:$(stop_at_k)"# with regularization $regularization for the generalized eigenvalue problems"
    for k in 1:stop_at_k
        S = global_asym_multiplier * lhs_op - (global_asym_multiplier^2/(ζ*(ζ - global_asym_multiplier))) * rhs_op # Schur complement
        Ω = objective_operator(cache) # Note that this does not have the ζ factor from bounds.jl, we put this in scattering2power instead to make this as confusing as possible

        S_svdvals = rsvdvals(S, 1; num_oversamples=10, num_power_iterations=3, sample_vec=CUDA.zeros(eltype(S), 0))
        S_norm = CUDA.@allowscalar maximum(S_svdvals)
        @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] ‖S‖₂ ≈ $(S_norm)"

        Ω_svdvals = rsvdvals(Ω, 1; num_oversamples=10, num_power_iterations=3, sample_vec=CUDA.zeros(eltype(Ω), 0))
        Ω_norm = CUDA.@allowscalar maximum(Ω_svdvals)
        @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] ‖Ω‖₂ ≈ $(Ω_norm)"

        # Find directions where S is *too* small which causes numerical instability in the generalized eigenvalue problem.
        # These are evil because they caused me 6 months of debugging
        # We will project these directions out in the generalized eigenvalue problem
        # We do this in the basis and in the full space

        evil_threshold = 1e-6

        S_basis = global_asym_multiplier * lhs_basis - (global_asym_multiplier^2/(ζ*(ζ - global_asym_multiplier))) * rhs_basis
        # S_basis_eigen = eigen(Hermitian(S_basis))
        # S_basis_norm = maximum(abs.(S_basis_eigen.values))

        # Evil directions directly in the basis
        # basis_evil_idxs = abs.(S_basis_eigen.values) .< evil_threshold * S_basis_norm
        # @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Found $(sum(basis_evil_idxs)) basis eigenvalues of S below evil threshold of $(evil_threshold * S_basis_norm); these will be projected out in the generalized eigenvalue solve"
        # basis_evil_directions = collect(eachcol(basis * S_basis_eigen.vectors[:, basis_evil_idxs]))
 
        # Evil directions in the full space (KrylovKit.jl tends to report an "invariant susbpace of dimension 1", i.e., the small eigenvalues are numerically degenerate, which is why we need the basis evil directions.
        # small_eig_initial_guess = basis * S_basis_eigen.vectors[:, argmin(S_basis_eigen.values)]
        # small_eig_initial_guess = CUDA.rand(eltype(S), size(S, 1))
        # small_eig_initial_guess = u_projector * small_eig_initial_guess # Zero out the gap region (which has many zero eigenvalues)
        # num_small_eigvals = 5
        # smallest_eigvals, smallest_eigvecs, _ = eigsolve(v -> S * v, small_eig_initial_guess, num_small_eigvals, :SR; tol=1e-9, krylovdim=280, maxiter=5001, ishermitian=true, orth=KrylovKit.ModifiedGramSchmidt2())
        noise_threshold = evil_threshold * S_norm
        # evil_idxs = abs.(smallest_eigvals) .< noise_threshold
        # @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Smallest eigenvalues of S: $(smallest_eigvals); Found $(sum(evil_idxs)) eigenvalues below noise threshold of $noise_threshold which will be projected out in the generalized eigenvalue solve"
        # evil_directions = smallest_eigvecs[evil_idxs]

        # all_evil_directions = [basis_evil_directions; evil_directions]
        all_evil_directions = Vector{CuArray{eltype(S), 1}}()

        # for (i, vec) in enumerate(all_evil_directions)
        #     overlap = gap_overlap(vec, u_projector)
        #     @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Evil direction $i has gap overlap of $(overlap); this should be close to zero to avoid numerical instability in the generalized eigenvalue problem"
        # end

        # Fullspace deflation loop: keep looking for directions where S is too small
        max_deflation_trials = 100
        vals_per_trial = 3
        for trial in 1:max_deflation_trials
            @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Full space deflation trial $trial/$max_deflation_trials to find small eigenvalues of S after projecting out $(length(all_evil_directions)) evil directions"
            x₀ = CUDA.rand(eltype(S), size(S, 1))
            for _ in 1:4 # Orthogonalize a few times
                x₀ = u_projector * x₀ # Zero out the gap region (which has many zero eigenvalues)
                if !isempty(all_evil_directions)
                    D = stack(all_evil_directions)
                    x₀ .-= D * (D' * x₀)
                end
                nrm = norm(x₀)
                if nrm < 1e-14 # We found a vector in the span of the evil directions, just keep moving on
                    @warn string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Full space deflation trial $trial/$max_deflation_trials produced a random vector that is numerically zero after projecting out $(length(all_evil_directions)) evil directions (norm after orthogonalization = $(nrm)); ignoring this direction"
                    continue
                end
                x₀ ./= nrm
            end
            
            if !isempty(all_evil_directions)
                P = projector_from_vecs(all_evil_directions)
                # deflator = LinearMap(v -> v .- P * v, v -> v .- P * v, size(S)...; ismutating=false, ishermitian=true)
                deflator = u_projector - P
                S_deflated = deflator * S * deflator
            else
                S_deflated = S
            end
            
            floor_threshold = S_norm * 1e-14 # safe margin above machine epsilon to avoid picking up noise
            vals, vecs, info = eigsolve(v -> S_deflated * v, x₀, vals_per_trial, :SR; tol=1e-10, krylovdim=280, maxiter=5001, ishermitian=true, orth=KrylovKit.ModifiedGramSchmidt2())
            valid_idxs = [i for (i, (λ, v, r)) in enumerate(zip(vals, vecs, info.normres)) if abs(λ) >= floor_threshold && r < 1e-3 * abs(λ)]
            vals = vals[valid_idxs]
            vecs = vecs[valid_idxs]
            max_idxs = min(vals_per_trial, length(vals))
            vals = vals[1:max_idxs]
            vecs = vecs[1:max_idxs]
            real_evil_count = count((abs.(vals) .<= noise_threshold) .& (abs.(vals) .>= floor_threshold)) # "Real" here means not numerical noise
            if real_evil_count == 0
                @info "No more real evil directions; terminating deflation"
                break
            end
            # if all(abs.(vals) .> noise_threshold)
            #     @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Full space deflation trial $trial/$max_deflation_trials did not find any new evil directions; proceeding with the generalized eigenvalue solve"
            #     break
            # end
            which = (abs.(vals) .<= noise_threshold) .& (abs.(vals) .>= floor_threshold)
            # which = abs.(vals) .<= noise_threshold
            @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Found evil directions with eigenvalues $(vals[which]); adding these to the projector and trying again"
            new_evil_vecs = collect(vecs[which])
            keep = trues(length(new_evil_vecs))
            for (i, vec) in enumerate(new_evil_vecs)
                overlap = gap_overlap(vec, u_projector)
                if overlap > 1e-15
                    @warn string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] New evil direction $i from trial has gap overlap of $(overlap); dropping"
                    keep[i] = false
                end
                @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] New evil direction $i from trial has gap overlap of $(overlap)"
            end
            new_evil_vecs = new_evil_vecs[keep]
            append!(all_evil_directions, new_evil_vecs)
            if trial == max_deflation_trials
                @warn string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Reached maximum number of deflation trials ($max_deflation_trials) in the full space without finding a non-evil direction; proceeding with the generalized eigenvalue solve but the results may be unreliable"
            end
        end

        # Gather basis and full space evil directions and build the projector to deflate them
        evil_projector = projector_from_vecs(all_evil_directions)
        deflator = u_projector - evil_projector
        # deflator = LinearMap(v -> v .- evil_projector * v, v -> v .- evil_projector * v, size(S)...; ismutating=false, ishermitian=true)

        # Check ‖Ω‖ vs ‖Ωv₀‖
        # v₀ = smallest_eigvecs[argmin(smallest_eigvals)]
        # small_norm = norm(Ω * v₀)
        # @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] ‖Ω v₀‖ ≈ $(small_norm) (should be ≤ ‖Ω‖ ≈ $(Ω_norm))"

        @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Building the full space objective operator"
        if k == 1
            # TODO: Should we do this for every k? Or will the solution from a previous iteration be a better initial guess?
            @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Projecting the Schur complement into the basis"
            S_basis = global_asym_multiplier * lhs_basis - (global_asym_multiplier^2/(ζ*(ζ - global_asym_multiplier))) * rhs_basis
            @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Projecting the objective operator into the basis"
            Ω_basis = basis' * opmat(Ω, basis)

            @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Solving the generalized eigenvalue problem for (Ω, S) in the basis for an initial guess for the full space solve"
            basis_genvals, basis_genvecs = eigen(Hermitian(Ω_basis), Hermitian(S_basis))
            initial_guess = basis * basis_genvecs[:, argmax(basis_genvals)]
        end

        num_genvecs = 1
        try
            @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Solving the generalized eigenvalue problem for (Ω, S) in the full space"
            genvals, genvecs, geninfo = geneigsolve(v -> (sym(deflator * Ω * deflator) * v, sym(deflator * S * deflator) * v), initial_guess, num_genvecs, :LR, tol=1e-8, krylovdim=280, maxiter=20, ishermitian=true, isposdef=true, orth=KrylovKit.ModifiedGramSchmidt2())

            # If the solver did not converge, don't report a bound (leave it as NaN). We don't update the initial guess, but the projector will be updated with a pretty good vector
            if geninfo.converged < num_genvecs
                @warn string(now()) * " [bounds_denial::_compute_bounds_sr] Generalized eigenvalue problem for (Ω, S) in the full space did not fully converge; converged $(geninfo.converged)/$num_genvecs eigenvalues" geninfo

                # update_projector!(cache)
                update_projector_classic!(cache)
                continue
            end

            best_genval = maximum(genvals)
            best_genvec = genvecs[argmax(genvals)]
            power_bound = best_genval * scattering2power
            @info string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Largest generalized eigenvalue of (Ω, S) $(best_genval) gives bound of $power_bound in power units"

            power_bounds[k] = power_bound
            initial_guess = best_genvec # Use this iteration's solution as the initial guess for the next iteration # TODO: maybe there's a smart thing we can do to this vector to have it be a better guess for the next iteration since we know exactly how the projector changes?

            # update_projector!(cache, best_genvec)
            update_projector_classic!(cache)
        catch e
            if !(e isa PosDefException)
                 rethrow(e)
            end
            @error string(now()) * " [bounds_denial::_compute_bounds_sr] [k=$k/$(stop_at_k)] Generalized eigenvalue solve for (Ω, S) failed at k=$k" e

            # Don't report a bound (leave it as NaN) but continue to the next k. We don't update the initial guess, but the projector will be updated with a pretty good vector
            # update_projector!(cache)
            update_projector_classic!(cache)
        end
        @show power_bounds
    end
    
    # Save data to disk
    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out = jldopen(jld_out_path, "a")
    if !haskey(jld_out, "Γ")
        jld_out["Γ"] = Array(Γ)
    end
    if !haskey(jld_out, "χ")
        jld_out["χ"] = susceptibility(smr)
    end
    if !haskey(jld_out, "global_asym_power_bounds")
        jld_out["global_asym_bounds"] = Array(power_bounds)
    end
    close(jld_out)
end

function compute_bounds()
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [bounds_denial::compute_bounds] Using GPU acceleration on device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [bounds_denial::compute_bounds] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _compute_bounds_sr(compute_env, smr, rsvd_params)
    else
        _compute_bounds_smr(compute_env, smr, rsvd_params)
    end
end
