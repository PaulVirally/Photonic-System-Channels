using MatrixFreeRandomizedLinearAlgebra
using LinearMaps
using LinearAlgebra
using JLD2
using CUDA
using GilaElectromagnetics
 
function generate_rsvd()
    @info string(now()) * " [rsvd::generate_rsvd] Starting RSVD generation"
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [generate_green::generate_green] Using GPU acceleration on device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
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

function asym_ur(G₀_rs::VacuumGreenOperator, G₀_rr::VacuumGreenOperator, smr::SMRSystem)
    # We want to compute the action of Asym(G⁰ᵣᵤ), but this operator is ambiguously defined. Here we write out it's definition.
    # Let ιᵣ be the inclusion of the receiver region into the universe, then define
    # Ĝ⁰ᵣᵤ = ιᵣ G⁰ᵣᵤ which is an operator that maps from the universe to the universe, but is zero outside of the receiver region:
    # Ĝ⁰ᵣᵤ = [0 0; G⁰ᵣₛ G⁰ᵣᵣ]
    # We can now formally define Asym(G⁰ᵣᵤ) = Asym(Ĝ⁰ᵣᵤ) = (Ĝ⁰ᵣᵤ - Ĝ⁰ᵣᵤ')/(2im):
    # Asym(G⁰ᵣᵤ) = [0 -1/2im G⁰ᵣₛ'; 1/2im G⁰ᵣₛ Asym(G⁰ᵣᵣ)]
    # Here it is in code:
    s = sender(smr)
    r = receiver(smr)

    sender_size = prod(s.cel)*3
    receiver_size = prod(r.cel)*3

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
        copyto!(r_clipped, view(u_vec, 1:receiver_size))
        return r_clipped
    end

    # Define the clipping operator that takes a vector defined on the universe and extracts only the sender part
    sender_clip_action!(s_clipped::AbstractVector{ComplexF64}, u_vec::AbstractVector{ComplexF64}) = begin
        # our convention is [sender; receiver]
        copyto!(s_clipped, view(u_vec, sender_size .+ (1:receiver_size)))
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

    asym_G₀_rr = LinearMap(AsymVacuumGreenOperator(G₀_rr)) # twice as efficient as (G₀_rr - G₀_rr')/(2im)
    asym_G₀_ru = asym(ι_r * G₀_rs * Π_s) + ι_r * asym_G₀_rr * Π_r
    return asym_G₀_ru
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

function _save_ur_asym(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    fname = file_prefix(smr)
    jld_path = joinpath(scratch_dir(compute_env), "$(fname).jld")

    jld_key = "UR_asym/"
    if ispath(jld_path)
        jld = jldopen(jld_path, "r")
        if haskey(jld, jld_key * "D") && haskey(jld, jld_key * "V")
            @info string(now()) * " [rsvd::generate_rsvd] RSVD for $(jld_key) already exists at $(jld_path): skipping"
            close(jld)
            return
        else
            close(jld)
        end
    end

    @info string(now()) * " [rsvd::generate_rsvd] Computing RSVD for UR_asym"
    @info string(now()) * " [rsvd::generate_rsvd] Loading G₀ operators"
    # G₀_uu = load_green_function(compute_env, smr, Design, Design) # universe -> universe
    # G₀_ur_asym = asym_ur(G₀_uu, smr)
    # G₀_ru = load_green_function(compute_env, smr, [Receiver], [Sender, Receiver]) # universe -> receiver
    G₀_rs = load_green_function(compute_env, smr, Receiver, Sender) # sender -> receiver
    G₀_rr = load_green_function(compute_env, smr, Receiver, Receiver) # receiver -> receiver
    G₀_ur_asym = asym_ur(G₀_rs, G₀_rr, smr)
    sample_vec = zeros(ComplexF64, 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end

    @info string(now()) * " [rsvd::generate_rsvd] Computing $(rank(rsvd_params)) components of a randomized eigen decomposition for a $(size(G₀_ur_asym)) Hermitian operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
    out = reigen_hermitian(G₀_ur_asym, rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)

    @info string(now()) * " [rsvd::generate_rsvd] Saving reigen to $(jld_path)"
    _save_reigen_hermitian(out.vectors, out.values, jld_path, jld_key)
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

function _run_rsvdvals(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams, jld_key::String)
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
    sample_vec = zeros(ComplexF64, 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end

    @info string(now()) * " [rsvd::_run_rsvdvals] Computing $(rank(rsvd_params)) components of a randomized SVD for a $(size(G₀_ab)) operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
    out = rsvdvals(LinearMap(G₀_ab), rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)

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
