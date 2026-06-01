using Dates
using JLD2

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
        disjoint_union_indicator = cu(disjoint_union_indicator)
    end
    fill!(view(disjoint_union_indicator, sender_mask..., :), one(eltype(disjoint_union_indicator)))
    fill!(view(disjoint_union_indicator, receiver_mask..., :), one(eltype(disjoint_union_indicator)))
    disjoint_union_projector_action!(w, v) = begin
        w .= vec(disjoint_union_indicator .* reshape(v, size(disjoint_union_indicator)))
        return w
    end
    u_projector = LinearMap{ComplexF64}(disjoint_union_projector_action!, disjoint_union_projector_action!, size(G₀_uu)...; ismutating=true)

    sender_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    if use_gpu(env)
        sender_indicator = cu(sender_indicator)
    end
    fill!(view(sender_indicator, sender_mask..., :), one(eltype(sender_indicator)))
    s_projector_action!(w, v) = begin
        w .= vec(sender_indicator .* reshape(v, size(sender_indicator)))
        return w
    end
    s_projector = LinearMap{ComplexF64}(s_projector_action!, s_projector_action!, size(G₀_uu)...; ismutating=true)

    receiver_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    if use_gpu(env)
        receiver_indicator = cu(receiver_indicator)
    end
    fill!(view(receiver_indicator, receiver_mask..., :), one(eltype(receiver_indicator)))
    r_projector_action!(w, v) = begin
        w .= vec(receiver_indicator .* reshape(v, size(receiver_indicator)))
        return w
    end
    r_projector = LinearMap{ComplexF64}(r_projector_action!, r_projector_action!, size(G₀_uu)...; ismutating=true)

    G₀ = LinearMap(G₀_uu)
    return r_projector, s_projector, u_projector, (u_projector * G₀ * u_projector)
end

function fullspace_global_asym_bound(SOME MORE STUFF HERE, ..., num_enrichment_vecs::Int)
    # TODO: you are here
    lhs_op = (imag(inv(χ))*u_projector - u_projector * asym(LinearMap(G₀_uu))) * u_projector # Im(χ⁻¹)I - Asym(G₀)
    lhs = sym(basis' * opmat(lhs_op, basis)) # Project into the basis of the (low rank) right singular vectors of G⁰ᵤᵤ

    rhs_op = s_projector + (ζ^2/4) * u_projector * G₀_uu' * s_projector * G₀_uu * u_projector + ζ * asym(s_projector * G₀_uu * u_projector) # S + (ζ^2/4) G₀' * S * G₀ + ζ * Asym(S * G₀ * U)
    rhs = sym(basis' * opmat(rhs_op, basis))

    # RSVD First
    basis_vals, basis_vecs = eigen!(Hermitian(lhs), Hermitian(rhs))
    basis_value = maximum(basis_vals)
    initial_guess = basis * cu(basis_vecs[:, argmax(basis_vals)])
    basis_multiplier_bound = ζ^2 * basis_value / (ζ * basis_value + 1)
    @info string(now()) * " [bounds::fullspace_global_asym_bound] RSVD predicts α > $(basis_multiplier_bound) for the global asym multiplier"

    # Full space
    @assert num_enrichment_vecs > 0 "You have to enrich the basis otherwise this duck will be sad 🦆 😢"
    regularization = 1e-14 #zero(real(eltype(lhs_op)))
    success = false
    fs_multiplier_bound = -Inf
    while regularization < 1e-6
        @info string(now()) * " [bounds::fullspace_global_asym_bound] Solving the (lhs, rhs) generalized eigenvalue problem in the full space with regularization $regularization"
        try
            vals, vecs, info = geneigsolve(v -> (lhs_op*v, rhs_op*v .+ regularization .* v), lr_initial_guess, num_enrichment_vecs, :LR; tol=1e-7, krylovdim=280, maxiter=5001, ishermitian=true, isposdef=true, orth=KrylovKit.ModifiedGramSchmidt2())
            fs_value = maximum(lr_vals)
            fs_multiplier_bound = ζ^2 * fs_value / (ζ * fs_value + 1)
            @info string(now()) * " [bounds::fullspace_global_asym_bound] Full space generalized eigenvalue problem forces α > $(lr_fs_multiplier_bound) for global asym with regularization $regularization" lr_info

            success = true

            @info string(now()) * " [bounds::fullspace_global_asym_bound] Enriching the basis with the eigenvectors from the generalized eigenvalue problem"

            # TODO: make this faster
            @error "make orthogonalization faster by using the fact the old basis was already orthogonal"
            basis = hcat(basis, vecs...)
            basis = qthin!(basis) # Orthogonalize
            @info string(now()) * " [bounds::fullspace_global_asym_bound] Re-forming the dense matrices in the enriched basis"
            # TODO: make this faster too
            @error "make this faster by only computing new rows and column"
            lhs = sym(U_uu' * opmat(lhs_op, U_uu))
            rhs = sym(U_uu' * opmat(rhs_op, U_uu))

            break
        catch e
            if e isa InterruptException
                rethrow(e)
            end
            @error string(now()) * " [bounds::fullspace_global_asym_bound] Generalized eigenvalue problem did not converge with regularization $regularization; trying again with higher regularization" e
            if iszero(regularization)
                regularization = 1e-15
            else
                regularization *= 10
            end
        end
    end
    if !success
        fs_multiplier_bound = basis_multiplier_bound
        @warn string(now()) * " [bounds::compute_bounds] Failed to solve the generalized eigenvalue problem in the full space; using the RSVD-based bound of α > $(fs_multiplier_bound) for global asym instead"
    end
    return fs_multiplier_bound, basis
end

function _compute_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [bounds::_compute_bounds_sr] Computing bounds for SR system"
    
    @info string(now()) * " [bounds::_compute_bounds_sr] Loading pre-computed data from JLD2 files"
    jld_in_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")

    jld_in = jldopen(jld_in_path, "r")
    jld_out = jldopen(jld_out_path, "w")

    # Read in pre-computed data
    Γ = read_array(jld_in, "UR_asym/D", use_gpu(compute_env)) # The eigenvalues of I_u Asym(G₀_uu) I_r
    Γ .*= -one(eltype(Γ)) # Sign typo in the original notes
    Vur_asym = read_array(jld_in, "UR_asym/V", use_gpu(compute_env)) # The eigenvectors of I_u Asym(G₀_uu) I_r
    jld_out = jldopen(joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld"), "w")
    sorted_idxs = ordering_idxs(Γ)

    # Sort the singular values vectors according to the ordering_idxs
    Vur_asym = Vur_asym[:, sorted_idxs]
    Γ = Array(Γ[sorted_idxs]) # Send to CPU because we pretty much only scalar index this guy

    Γrs = jld_in["RS/D"] # The eigenvalues of G₀_rs
    if !haskey(jld_out, "Γrs")
        jld_out["Γrs"] = Array(Γrs) # Save these in the output file
    end
    if !haskey(jld_out, "ordering_idxs")
        jld_out["ordering_idxs"] = Array(sorted_idxs) # Save the ordering indices in the output file to see how much reordering we had to do
    end
    close(jld_out) # We'll re-use jld_out, but for now let's close the file in case the program crashes before the end

    Γrs = nothing # We don't need this anymore, so let's free up some memory
    basis = read_array(jld_in, "UU/U", use_gpu(compute_env)) # The RSVD basis: right singular vectors of G₀_uu
    RSVD_BASIS_SIZE = size(basis, 2)
    # RSVD_BASIS_SIZE = 256
    @info string(now()) * " [bounds::_compute_bounds_sr] RSVD basis size: $(size(basis, 2)); Using RSVD_BASIS_SIZE = $RSVD_BASIS_SIZE"
    basis = basis[:, 1:RSVD_BASIS_SIZE]

    close(jld_in)
    GC.gc()
    GC.gc()
    GC.gc() # Reduce VRAM pressure

    G₀_uu = load_greens_function(compute_env, smr, Design, Design)
    r_projector, s_projector, u_projector, G₀_uu_disjoint = projected_operators(G₀_uu, smr, compute_env)

    # Material factors
    χ = susceptibility(smr)
    ζ = abs(χ)^2/imag(χ)
    @info string(now()) * " [bounds::_compute_bounds_sr] Susceptibility χ = $χ, material factor ζ = $ζ"
end

function compute_bounds()
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [bounds::compute_bounds] Using GPU acceleration on device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [bounds::compute_bounds] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _compute_bounds_sr(compute_env, smr, rsvd_params)
    else
        _compute_bounds_smr(compute_env, smr, rsvd_params)
    end
end
