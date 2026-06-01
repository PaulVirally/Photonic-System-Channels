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

function gap_overlap(vec::AbstractVector, u_projector::LinearMap)
    # Measure how much the vector overlaps with the gap region
    gap_component = vec - u_projector * vec # Gap region = (I - U)
    return norm(gap_component) / norm(vec)
end

function similar_fill(arr::AbstractArray, size, val::T) where T
    new_arr = similar(arr, T, size)
    fill!(new_arr, val)
    return new_arr
end

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
    @info string(now()) * " [bounds_bargaining::ordering_idxs] Forming the indexing set by ordering γ⁰ᵃᵤᵣ"
    sorted_idxs = sortperm(Γ; rev=true)
    return sorted_idxs
end

function read_array(jld, key::AbstractString, use_gpu::Bool)
    @info string(now()) * " [bounds_bargaining::read_array] Reading array for key '$key' from JLD file"
    if haskey(jld, key)
        arr = jld[key]
        if use_gpu
            @info string(now()) * " [bounds_bargaining::read_array] Moving array for key '$key' to GPU"
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
    @info string(now()) * " [bounds_bargaining::_compute_bounds_smr] Computing bounds for SMR system"
    throw("Not implemented yet")
end

function modified_gram_schmidt!(v::AbstractVector, basis::AbstractMatrix; num_iterations::Int=1)
    # Orthogonalize v against the columns of basis using modified Gram-Schmidt
    for _ in 1:num_iterations
        for j in axes(basis, 2)
            v .-= dot(view(basis, :, j), v) * view(basis, :, j)
        end
    end
    return v
end

function random_orthogonal_vector(basis; initial_op=I, max_rand_tries=3, mgs_iters=2, norm_threshold=1e-10, sample_vec=zeros(eltype(basis), size(basis, 1)), seed::Union{Nothing, AbstractVector}=nothing)
    v = similar(sample_vec, eltype(basis), size(basis, 1))
    for i in 1:max_rand_tries # Try a few times to find a direction that is not in the span of the previous bad directions; if we fail too many times in a row, we've run out of orthogonal directions to search and we should just give up
        if isnothing(seed)
            rand!(v) # Seed with a random vector
        else
            copyto!(v, seed)
        end
        v .= initial_op * v
        normalize!(v) # Normalize to make modified Gram-Schmidt slightly more stable
        modified_gram_schmidt!(v, basis; num_iterations=mgs_iters) # Orthogonalize against previously found bad directions
        norm_x₀ = norm(v)
        if norm_x₀ < norm_threshold
            if i == max_rand_tries
                return nothing
            end
            continue
        end
        break
    end
    return v
end

# Estimate the 2-norm of a Hermitian operator using a randomized singular value decomposition
function opnorm_2(op::LinearMap; num_oversamples=10, num_power_iterations=3, sample_vec=zeros(eltype(op), 0))
    # @info string(now()) * " [bounds_bargaining::opnorm_2] Estimating the operator 2-norm using randomized SVD with num_oversamples = $num_oversamples and num_power_iterations = $num_power_iterations"
    vals = rsvdvals(op, 1; num_oversamples=num_oversamples, num_power_iterations=num_power_iterations, sample_vec=sample_vec)
    # @info string(now()) * " [bounds_bargaining::opnorm_2] Estimated operator 2-norm: $(maximum(vals))"
    return maximum(vals)
end

# Projector that spans the column space of `ortho_vecs`; Assumes that the columns of `ortho_vecs` are orthonormalized (if not, we can orthonormalize them first using qthin!)
function projector(ortho_vecs::AbstractMatrix)
    forward!(out, v) = begin
        out .= ortho_vecs * (ortho_vecs' * v)
    end
    return LinearMap(forward!, forward!, size(ortho_vecs, 1), size(ortho_vecs, 1); ismutating=true, ishermitian=true)
end

# Remove the bad directions `bad_vecs` from `op`
function deflated_op(op::LinearMap, bad_vecs::AbstractMatrix, u_projector::LinearMap)
    if iszero(size(bad_vecs, 2))
        return u_projector * op * u_projector # TODO: If this is too slow and we are super mega ultra duper sure that op zeros out the gap, we can remove the u_projectors
    end
    Π = projector(bad_vecs)
    deflator = u_projector - Π # The "identity" is the gapless region (because we really only care about operators that act on sender ⊔ receiver, NOT sender ∪ receiver)
    return deflator * op * deflator
end

function bad_directions(op::LinearMap, sign_definiteness::Symbol, u_projector::LinearMap, min_abs_rvalue::Float64=-1.0; max_directions::Int=100, directions_per_iter::Int=3, sample_vec::AbstractVector=zeros(eltype(op), 0), norm_threshold::Float64=1e-10, mgs_iters::Int=2, krylov_rtol=1e-6, krylovdim=280, noise_floor_rtol=1e-14)
    if !(sign_definiteness in [:PSD, :NSD])
        error("sign_definiteness must be :PSD or :NSD")
    end
    spectrum_end = sign_definiteness == :PSD ? :SR : :LR # Search for smallest real part for PSD, largest real part for NSD

    bad_vecs = similar_fill(sample_vec, (size(op, 2), max_directions), zero(eltype(op))) # Will contain the directions the ruin the sign definiteness of the operator
    num_bad_vecs = 0

    # All norms are relative to the operator norm, so compute the operator norm
    op_norm = opnorm_2(op; sample_vec=sample_vec)
    noise_floor = noise_floor_rtol * op_norm # Any eigenvalue λ such that |λ| < noise_floor is considered noise and ignored when determining if a direction is bad or not
    min_abs_value = min_abs_rvalue * op_norm # A direction is also considered bad if its eigenvalue has absolute value less than this threshold. This is useful for the Schur complement to get rid of unphysically small eigenvalues that lead to numerical instability in the generalized eigenvalue problem for the dual bound

    # @info string(now()) * " [bounds_bargaining::bad_directions] Starting search for bad directions with max_directions = $max_directions, directions_per_iter = $directions_per_iter, noise_floor = $noise_floor, and min_abs_value = $min_abs_value"

    while num_bad_vecs < max_directions
        # Random vector orthogonal to previously found bad vectors
        # @info string(now()) * " [bounds_bargaining::bad_directions] Building up a random vector orthogonal to the $num_bad_vecs previously found bad directions"
        x₀ = random_orthogonal_vector(view(bad_vecs, :, 1:num_bad_vecs); initial_op=u_projector, sample_vec=sample_vec, norm_threshold=norm_threshold, mgs_iters=mgs_iters)
        if isnothing(x₀)
            @warn string(now()) * " [bounds_bargaining::bad_directions] Ran out of orthogonal directions to search for bad directions; found $num_bad_vecs bad directions"
            break
        end

        # Deflate the operator to remove the influence of previously found bad directions and find the next bad direction
        deflated = deflated_op(op, view(bad_vecs, :, 1:num_bad_vecs), u_projector)
        # @info string(now()) * " [bounds_bargaining::bad_directions] Looking for $directions_per_iter extremal eigenvalues of the deflated operator"
        vals, vecs, _ = eigsolve(v -> deflated * v, x₀, directions_per_iter, spectrum_end; tol=krylov_rtol*op_norm, krylovdim=krylovdim, maxiter=5001, ishermitian=true, orth=KrylovKit.ModifiedGramSchmidt2()) # Note that length(vals) might be greater than directions_per_iter

        # Filter out eigenvalues below the noise floor
        keep_idxs = abs.(vals) .> noise_floor # Only consider eigenvalues above the noise floor
        vals = vals[keep_idxs]
        vecs = vecs[keep_idxs]

        # Are there any eigenvalues that violate the sign definiteness?
        sign_definiteness_condition = sign_definiteness == :PSD ? (vals .< zero(eltype(vals))) : (vals .> zero(eltype(vals)))
        min_abs_value_condition = abs.(vals) .<= min_abs_value # Consider any eigenvalues that are too small to be bad
        bad_idxs = findall(sign_definiteness_condition .| min_abs_value_condition)
        if isempty(bad_idxs)
            # @info string(now()) * " [bounds_bargaining::bad_directions] No more bad directions found; terminating search with $num_bad_vecs bad directions"
            # @show vals
            break
        end
        num_new_bad_vecs = length(bad_idxs)
        if num_bad_vecs + num_new_bad_vecs > max_directions
            # @warn string(now()) * " [bounds_bargaining::bad_directions] Found $num_new_bad_vecs new bad directions, but only $(max_directions - num_bad_vecs) more can be stored; truncating and terminating search with $max_directions bad directions"
            num_new_bad_vecs = max_directions - num_bad_vecs
        end
        # @info string(now()) * " [bounds_bargaining::bad_directions] Found $num_new_bad_vecs bad directions in this iteration with eigenvalues $(vals[bad_idxs])"
        # for (i, bad_idx) in enumerate(bad_idxs)
            # @info string(now()) * " [bounds_bargaining::bad_directions] Bad direction $i/$(num_new_bad_vecs) with eigenvalue $(vals[bad_idx]) has gap overlap $(gap_overlap(vecs[bad_idx], u_projector))"
        # end

        # Store the bad directions and orthogonalize the basis
        # @info string(now()) * " [bounds_bargaining::bad_directions] Orthogonalizing the bad directions"
        for i in 1:num_new_bad_vecs # Bruh this loop
            CUDA.@allowscalar bad_idx = bad_idxs[i]
            copyto!(view(bad_vecs, :, num_bad_vecs + i), vecs[bad_idx])
        end
        num_bad_vecs += num_new_bad_vecs
        qthin!(view(bad_vecs, :, 1:num_bad_vecs)) # Orthonormalize the bad directions to improve numerical stability in future iterations
        qthin!(view(bad_vecs, :, 1:num_bad_vecs)) # Do it twice just to be safe (should not be necessary, the CUDA implementation of QR is Householder which is numerically stable, but whatever it is cheap relative to the eigsolves)
    end

    # @info string(now()) * " [bounds_bargaining::bad_directions] Finished searching for bad directions; found $num_bad_vecs bad directions"
    return view(bad_vecs, :, 1:num_bad_vecs)
end

function global_asym_multiplier_bound(lhs_op::LinearMap, rhs_op::LinearMap, u_projector::LinearMap, compute_env::ComputeEnvironment, ζ::Float64; max_directions::Int=100)
    # Compute generalized eigenvalues of (lhs, rhs) to get a bound on the global asym multiplier

    # We know that both lhs and rhs must be sign definite. Specifically, we
    # know that lhs is negative definite and rhs is positive definite.
    # Numerically, these might not be perfectly definite, so we will find all
    # the directions (for both lhs and rhs) that ruin this property and project
    # them out when we look for the bound on the global asym multiplier.

    sample_vec = zeros(eltype(lhs_op), 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end

    @info string(now()) * " [bounds_bargaining::global_asym_multiplier_bound] Finding vectors that violate lhs ⪯ 0"
    bad_lhs_directions = bad_directions(lhs_op, :NSD, u_projector; max_directions=(max_directions ÷ 2), sample_vec=sample_vec, noise_floor_rtol=1e-14, krylov_rtol=1e-4) # NSD is inherently harder than PSD, so use a smaller tolerance

    @info string(now()) * " [bounds_bargaining::global_asym_multiplier_bound] Finding vectors that violate rhs ⪰ 0"
    bad_rhs_directions = bad_directions(rhs_op, :PSD, u_projector; max_directions=(max_directions ÷ 2), sample_vec=sample_vec, noise_floor_rtol=1e-14, krylov_rtol=1e-4)

    # Combine TODO: is this valid? Should we be projecting out the bad lhs directions from rhs and vice-versa?
    all_bad_directions = hcat(bad_lhs_directions, bad_rhs_directions)
    for (i, direction) in eachcol(all_bad_directions)
        overlap = gap_overlap(direction, u_projector)
        @info string(now()) * " [bounds_bargaining::global_asym_multiplier_bound] Bad direction $i/$(size(all_bad_directions, 2)) has gap overlap $overlap"
    end

    # Remove the bad directions from the operators
    deflated_lhs = deflated_op(lhs_op, all_bad_directions, u_projector)
    deflated_rhs = deflated_op(rhs_op, all_bad_directions, u_projector)

    # Solve the geneig (lhs, rhs) to find a bound on α_global_asym
    x₀ = random_orthogonal_vector(all_bad_directions; initial_op=u_projector, sample_vec=sample_vec, norm_threshold=1e-10, mgs_iters=2)
    @info string(now()) * " [bounds_bargaining::global_asym_multiplier_bound] Solving the generalized eigenvalue problem (L, R) in the deflated space to get an initial bound on the global asym multiplier"
    vals, _, _ = geneigsolve(v -> (deflated_lhs*v, deflated_rhs*v), x₀, 1, :LR; tol=1e-7, krylovdim=280, maxiter=5001, ishermitian=true, isposdef=true, orth=KrylovKit.ModifiedGramSchmidt2())
    value = maximum(vals)
    multiplier_bound = ζ^2 * value / (ζ * value + 1)
    @info string(now()) * " [bounds_bargaining::global_asym_multiplier_bound] Bound on global asym multiplier from deflated geneig: α > $(multiplier_bound)"
    return multiplier_bound
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

mutable struct OptimizerState{T<:Number, VEC<:AbstractVector{T}, MAT<:AbstractMatrix{T}}
    objective_cache::ObjectiveCache{T, MAT, VEC}
    χ::T
    lhs_op::LinearMap{T}
    rhs_op::LinearMap{T}
    u_projector::LinearMap{T}
    sample_vec::VEC
    prev_solution::Union{Nothing, VEC}
end

function dual!(state::OptimizerState, global_asym_multiplier_arr::Vector{Float64}, ∇_arr::Vector{Float64})
    global_asym_multiplier = only(global_asym_multiplier_arr) # Unwrap the scalar from the array container

    ζ = abs(state.χ)^2/imag(state.χ)
    scattering2power = (sqrt(4 * imag(state.χ)^2)/abs(state.χ))^2 * ζ # The conversion factor from scattering units to power units for the bounds
    S = global_asym_multiplier * state.lhs_op - (global_asym_multiplier^2/(ζ*(ζ - global_asym_multiplier))) * state.rhs_op # Schur complement

    # We need S ⪰ 0, but we also need the nullspace to be physical. Small
    # directions with eigvals of ~1e-20 will be projected out as well as large-ish negative directions
    # @info string(now()) * " [bounds_bargaining::dual!] Finding directions that violate S($(global_asym_multiplier)) ⪰ 0 to project out in the generalized eigenvalue problem"
    min_abs_rvalue = 1e-7 # Any direction with eigenvalue less than this in absolute value will be counted as an evil direction
    noise_floor_rtol = 1e-14 # Any direction with eigenvalue less than this times the operator norm will be considered noise and ignored when determining if a direction is bad or not
    evil_directions = bad_directions(S, :PSD, state.u_projector, min_abs_rvalue; max_directions=20, directions_per_iter=3, sample_vec=state.sample_vec, norm_threshold=1e-10, mgs_iters=2, krylov_rtol=1e-8, noise_floor_rtol=noise_floor_rtol)

    # Deflate operators against these evil directions
    deflated_S = deflated_op(S, evil_directions, state.u_projector)
    deflated_Ω = deflated_op(objective_operator(state.objective_cache), evil_directions, state.u_projector)

    x₀ = random_orthogonal_vector(evil_directions; initial_op=state.u_projector, sample_vec=state.sample_vec, norm_threshold=1e-10, mgs_iters=2, seed=state.prev_solution)
    # @info string(now()) * " [bounds_bargaining::dual!] Solving the generalized eigenvalue problem (deflated_Ω, deflated_S) to get the dual bound for α = $global_asym_multiplier"
    vals, vecs, info = geneigsolve(v -> (deflated_Ω * v, deflated_S * v), x₀, 1, :LR; tol=1e-7, krylovdim=280, maxiter=5001, ishermitian=true, isposdef=true, orth=KrylovKit.ModifiedGramSchmidt2())
    if info.converged < 1
        @warn string(now()) * " [bounds_bargaining::dual!] Generalized eigenvalue solve did not converge for α = $global_asym_multiplier" info
    end
    best_idx = argmax(real.(vals))
    power_bound = real(vals[best_idx]) * scattering2power

    optimal_vector = vecs[best_idx]
    if isnothing(state.prev_solution)
        state.prev_solution = copy(optimal_vector)
    else
        state.prev_solution .= optimal_vector
    end

    # Gradient
    if length(∇_arr) > 0
        coeff = global_asym_multiplier * (2ζ - global_asym_multiplier) / (ζ * (ζ - global_asym_multiplier)^2)
        numerator_op = -power_bound * (state.lhs_op + coeff * state.rhs_op)
        deflated_numerator_op = deflated_op(numerator_op, evil_directions, state.u_projector)
        numerator = real(dot(optimal_vector, deflated_numerator_op * optimal_vector))
        denominator = real(dot(optimal_vector, deflated_S * optimal_vector))
        ∇ = numerator / denominator
        ∇_arr[1] = ∇
        @info string(now()) * " [bounds_bargaining::dual!] Dual($global_asym_multiplier): $power_bound; ∇ = $∇"
    else
        @info string(now()) * " [bounds_bargaining::dual!] Dual($global_asym_multiplier): $power_bound"
    end

    return power_bound
end

function _compute_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Computing bounds for SR system"

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
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] RSVD basis size: $(size(U_uu, 2)); Using RSVD_BASIS_SIZE = $RSVD_BASIS_SIZE"
    U_uu = U_uu[:, 1:RSVD_BASIS_SIZE]

    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Finished reading data from disk; closing JLD file and freeing memory"
    close(jld_in)
    GC.gc()
    GC.gc()
    GC.gc()

    G₀_uu = load_greens_function(compute_env, smr, Design, Design)
    r_projector, s_projector, u_projector, _ = projected_operators(G₀_uu, smr, compute_env)
    G0_ur_asym = asym(u_projector * LinearMap(G₀_uu) * r_projector)

    χ = susceptibility(smr)
    ζ = abs(χ)^2/imag(χ)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Susceptibility χ = $χ, material factor ζ = $ζ"

    # basis = U_uu

    lhs_op = (imag(inv(χ))*u_projector - u_projector * asym(LinearMap(G₀_uu))) * u_projector # Im(χ⁻¹)I - Asym(G₀)
    rhs_op = s_projector + (ζ^2/4) * u_projector * G₀_uu' * s_projector * G₀_uu * u_projector + ζ * asym(s_projector * G₀_uu * u_projector) # S + (ζ^2/4) G₀' * S * G₀ + ζ * Asym(S * G₀ * U)

    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Finding the bound for the global asym mulitiplier"
    multiplier_bound = global_asym_multiplier_bound(lhs_op, rhs_op, u_projector, compute_env, ζ; max_directions=30)
    # multiplier_bound = -0.0008441525923973377 # Pre-computed from a previous run to save 5 mins per debug run
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Initial bound on global asym multiplier: $multiplier_bound"

    # We do not need to do every index. We know that after Γ < 0, the bound is 0, so we add 5 to have a few points to show this
    stop_at_k = min(findfirst(Γ .< zero(eltype(Γ))) + 5, rank(rsvd_params))

    # scattering2power = (sqrt(4 * imag(χ)^2)/abs(χ))^2 * ζ # The conversion factor from scattering units to power units for the bounds
    # power_bounds = ones(stop_at_k) * NaN
    # initial_guess = nothing


    cache = ObjectiveCache(G0_ur_asym, Γ, Vur_asym, 100; support_tol=1e-12) # Cache for the objective operator; capacity of 100 deflation vectors should be more than enough for our purposes
    sample_vec = zeros(eltype(lhs_op), 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end

    power_bounds = ones(stop_at_k) * NaN
    prev_multiplier = multiplier_bound * 0.99
    for k in 1:stop_at_k
        state = OptimizerState(cache, χ, lhs_op, rhs_op, u_projector, sample_vec, nothing)
        # opt = NLopt.Opt(:LD_MMA, 1)
        opt = NLopt.Opt(:LD_LBFGS, 1)
        NLopt.lower_bounds!(opt, multiplier_bound*0.999)
        NLopt.upper_bounds!(opt, 0.0)
        NLopt.xtol_rel!(opt, 1e-2)
        NLopt.ftol_rel!(opt, 1e-4)
        objective(x, grad) = dual!(state, x, grad)
        NLopt.min_objective!(opt, objective)
        try
            @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] [k=$k/$(stop_at_k)] Starting optimization"
            optimal_dual_value, optimal_multiplier, ret = NLopt.optimize(opt, [prev_multiplier]) # Start slightly to the right of the multiplier bound
            prev_multiplier = optimal_multiplier[1]
            power_bounds[k] = optimal_dual_value
            @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] [k=$k/$(stop_at_k)] Optimal global asym multiplier: $optimal_multiplier; optimal dual value: $optimal_dual_value; return code: $ret"
        catch e
            if e isa InterruptException
                rethrow(e)
            end
            @warn string(now()) * " [bounds_bargaining::_compute_bounds_sr] Caught exception when setting up NLopt optimization for k = $k; skipping this k" e
        end
        update_projector_classic!(cache) # Update the projector to deflate the top k-1 eigenvectors of G^{0,a}_{ur}
        @show power_bounds
    end

    # TODO: After each k, we need to run: update_projector_classic!(state.objective_cache)

    # @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Computing dual bounds for k ∈ 1:$(stop_at_k)"# with regularization $regularization for the generalized eigenvalue problems"
    # for k in 1:stop_at_k
    #     global_asym_multiplier = multiplier_bound * 0.8 # Random guess for debugging
    #     @info string(now()) * " [bounds_bargaining] [$k/$(stop_at_k)] Using α = $global_asym_multiplier"
    #
    #     S = global_asym_multiplier * lhs_op - (global_asym_multiplier^2/(ζ*(ζ - global_asym_multiplier))) * rhs_op # Schur complement
    #
    #     # We need S ⪰ 0, but we also need the nullspace to be physical. Small
    #     # directions with eigvals of ~1e-20 will be projected out as well as large-ish negative directions
    #     @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] [k=$k/$(stop_at_k)] Finding directions that violate S ⪰ 0 to project out in the generalized eigenvalue problem"
    #     min_abs_rvalue = 1e-6 # Any direction with eigenvalue less than this in absolute value will be counted as an evil direction
    #     noise_floor_rtol=1e-14 # Any direction with eigenvalue less than this times the operator norm will be considered noise and ignored when determining if a direction is bad or not
    #     evil_directions = bad_directions(S, :PSD, u_projector, min_abs_rvalue; max_directions=100, directions_per_iter=3, sample_vec=sample_vec, norm_threshold=1e-10, mgs_iters=2, krylov_rtol=1e-8, noise_floor_rtol=noise_floor_rtol)
    #
    #     # Deflate operators against these evil directions
    #     deflated_S = deflated_op(S, evil_directions, u_projector)
    #     deflated_Ω = deflated_op(objective_operator(cache), evil_directions, u_projector)
    #
    #     @show opnorm_2(deflated_S; sample_vec=sample_vec)
    #     @show opnorm_2(deflated_Ω; sample_vec=sample_vec)
    #
    #     x₀ = random_orthogonal_vector(evil_directions; initial_op=u_projector, sample_vec=sample_vec, norm_threshold=1e-10, mgs_iters=2, seed=initial_guess)
    #     vals, vecs, info = geneigsolve(v -> (deflated_Ω * v, deflated_S * v), x₀, 1, :LR; tol=1e-7, krylovdim=280, maxiter=5001, ishermitian=true, isposdef=true, orth=KrylovKit.ModifiedGramSchmidt2())
    #     update_projector_classic!(cache) # Remove top eigenvector from Ω for the next k
    #     if info.converged < 1
    #         @warn string(now()) * " [bounds_bargaining::_compute_bounds_sr] [k=$k/$(stop_at_k)] Generalized eigenvalue solve did not converge"
    #         continue
    #     end
    #     best_idx = argmax(real.(vals))
    #     power_bound = real(vals[best_idx]) * scattering2power
    #     power_bounds[k] = power_bound
    #     initial_guess = vecs[best_idx]
    #     @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] [k=$k/$(stop_at_k)] Power bound: $power_bound"
    #     @show power_bounds
    # end

    #
    # # Save data to disk
    # jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    # jld_out = jldopen(jld_out_path, "a")
    # if !haskey(jld_out, "Γ")
    #     jld_out["Γ"] = Array(Γ)
    # end
    # if !haskey(jld_out, "χ")
    #     jld_out["χ"] = susceptibility(smr)
    # end
    # if !haskey(jld_out, "global_asym_power_bounds")
    #     jld_out["global_asym_bounds"] = Array(power_bounds)
    # end
    # close(jld_out)
end

function compute_bounds()
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using GPU acceleration on device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _compute_bounds_sr(compute_env, smr, rsvd_params)
    else
        _compute_bounds_smr(compute_env, smr, rsvd_params)
    end
end
