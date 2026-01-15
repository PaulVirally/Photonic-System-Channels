using MatrixFreeRandomizedLinearAlgebra
using LinearMaps
using LinearAlgebra
using JLD2
using CUDA
using GilaElectromagnetics
 
function generate_rsvd()
    @info string(now()) * " [rsvd::generate_rsvd] Starting RSVD generation"
    compute_env, smr, rsvd_params = parse_args()
    if isnothing(mediator(smr))
        _generate_rsvd_sr(compute_env, smr, rsvd_params)
        @info string(now()) * " [rsvd::generate_rsvd] Completed RSVD generation for SR system"
        return nothing
    end
    _generate_rsvd_smr(compute_env, smr, rsvd_params)
    @info string(now()) * " [rsvd::generate_rsvd] Completed RSVD generation for SMR system"
    return nothing
end

# function _disjoint_union_hack(G₀_ur::VacuumGreensOperator, G₀_ru_adjoint::VacuumGreensOperator, smr::SMRSystem)
#     s = sender(smr)
#     r = receiver(smr)
#     GilaElectromagnetics.GilaOperators.isoverlappingoperator(G₀_ur) || error("G₀_ur is not an overlapping operator")
#     GilaElectromagnetics.GilaOperators.isoverlappingoperator(G₀_ru_adjoint) || error("G₀_ru_adjoint is not an overlapping operator")
#     union_volume = G₀_ur.mem.srcVol # srcVol == trgVol for overlapping operators
#     sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
#     receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume
#     disjoint_G₀_ur = VacuumGreensOperator(G₀_ur.mem, receiver_mask, sender_mask) 
#     disjoint_G₀_ru_adjoint = VacuumGreensOperator(G₀_ru_adjoint.mem, sender_mask, receiver_mask)
#     return disjoint_G₀_ur, disjoint_G₀_ru_adjoint
# end

function asym_ur(G₀_uu::VacuumGreensOperator, smr::SMRSystem)
    s = sender(smr)
    r = receiver(smr)
    G₀_uu.mem.srcVol == G₀_uu.mem.trgVol || error("G₀_uu is not a self operator")
    union_volume = G₀_uu.mem.srcVol # srcVol == trgVol
    sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
    receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume
    disjoint_union_projector_action(x_union::AbstractArray{ComplexF64, 4}) = begin
        x = similar(x_union)
        fill!(x, zero(eltype(x)))
        copyto!(view(x, sender_mask..., :), view(x_union, sender_mask..., :))
        copyto!(view(x, receiver_mask..., :), view(x_union, receiver_mask..., :))
        # The output x now has nonzero entries only in the sender and receiver regions
        # that is, we've zeroed out the gap between s and r
        return x
    end
    vec_action!(w, v) = begin
        v_tens = reshape(v, glaSze(G₀_uu)[2])
        out_tens = disjoint_union_projector_action(v_tens)
        copyto!(w, vec(out_tens))
        return w
    end
    projector =  LinearMap(ComplexF64, size(G₀_uu)..., false, false, vec_action!, nothing, vec_action!; S=GilaElectromagneitcs.arrType(G₀_uu))
    G₀ = LinearMap(G₀_uu)
    return (G₀ * projector - projector * G₀') / (2im) # Note that projector is orthogonal
end

function crop_to_receiver(v::AbstractVector, smr::SMRSystem, total_volume::GlaVol)
    r = receiver(smr)
    receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, total_volume)
    v_tens = reshape(v, size(total_volume)..., :)
    cropped_tens = view(v_tens, receiver_mask..., :)
    return vec(cropped_tens)
end

function _generate_rsvd_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    fname = file_prefix(smr)
    jld_path = joinpath(scratch_dir(compute_env), "$(fname).jld")
    jld_key = "UR_asym/"
    if ispath(jld_path)
        jld = jldopen(jld_path, "r")
        if haskey(jld, jld_key * "U") && haskey(jld, jld_key * "D") && haskey(jld, jld_key * "V")
            @info string(now()) * " [rsvd::generate_rsvd] RSVD for $(jld_key) already exists at $(jld_path): skipping"
            close(jld)
            return
        else
            close(jld)
        end
    end

    @info string(now()) * " [rsvd::generate_rsvd] Computing RSVD for UR_asym"
    @info string(now()) * " [rsvd::generate_rsvd] Loading G₀ operators"
    G₀_uu = load_greens_function(compute_env, smr, Design, Design) # universe -> universe
    G₀_ur_asym = asym_ur(G₀_uu, smr)
    # G₀_ur = load_greens_function(compute_env, smr, Design, Receiver) # receiver -> universe
    # G₀_ru_adjoint = adjoint(load_greens_function(compute_env, smr, Receiver, Design)) # universe -> receiver (adjoint, so it goes receiver -> universe)
    # G₀_ur, G₀_ru_adjoint = _disjoint_union_hack(G₀_ur, G₀_ru_adjoint, smr)
    # G₀_ur_asym = (LinearMap(G₀_ur) - LinearMap(G₀_ru_adjoint)) / ComplexF64(2im) # Anti-Hermitian part of G₀_ur
    sample_vec = zeros(ComplexF64, 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end

    @info string(now()) * " [rsvd::generate_rsvd] Computing $(rank(rsvd_params)) components of a randomized eigen decomposition for a $(size(G₀_ur_asym)) Hermitian operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
    out = reigen_hermitian(G₀_ur_asym, rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)

    @info string(now()) * " [rsvd::generate_rsvd] Cropping eigenvectors to receiver region"
    cropped_vecs = hcat([Array(crop_to_receiver(v, smr, G₀_uu.mem.srcVol)) for v in eachcol(out.vectors)]...)

    @info string(now()) * " [rsvd::generate_rsvd] Saving reigen to $(jld_path)"
    _save_reigen_hermitian(cropped_vecs, out.values, jld_path, jld_key)

    # @info string(now()) * " [rsvd::generate_rsvd] Computing $(rank(rsvd_params)) components of a randomized SVD for a $(size(G₀_ur_asym)) operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
    # out = rsvd(G₀_ur_asym, rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)

    # @info string(now()) * " [rsvd::generate_rsvd] Saving RSVD to $(jld_path)"
    # _save_rsvd(out, jld_path, jld_key)
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
    G₀_ab = load_greens_function(compute_env, smr, target, source)
    sample_vec = zeros(ComplexF64, 0)
    if use_gpu(compute_env)
        sample_vec = CuArray(sample_vec)
    end

    @info string(now()) * " [rsvd::_run_rsvd] Computing $(rank(rsvd_params)) components of a randomized SVD for a $(size(G₀_ab)) operator using $(oversamples(rsvd_params)) oversamples and $(power_iter(rsvd_params)) power iterations"
    out = rsvd(LinearMap(G₀_ab), rank(rsvd_params); num_oversamples=oversamples(rsvd_params), num_power_iterations=power_iter(rsvd_params), sample_vec=sample_vec)

    @info string(now()) * " [rsvd::_run_rsvd] Saving RSVD to $(jld_path)"
    _save_rsvd(out, jld_path, jld_key)
end

function _generate_rsvd_smr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    jld_keys = ["RS/", "SM/", "MR/", "MM/"]
    for jld_key in jld_keys
        @info string(now()) * " [rsvd::_generate_rsvd_smr] Processing $(jld_key)"
        _run_rsvd(compute_env, smr, rsvd_params, jld_key)
    end
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
    _save_component(jld, jld_key * "V", Vt')

    @info string(now()) * " [rsvd::_save_rsvd] Saving singular values"
    _save_component(jld, jld_key * "D", S)

    @info string(now()) * " [rsvd::_save_rsvd] Saving right singular vectors"
    _save_component(jld, jld_key * "U", U)

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
