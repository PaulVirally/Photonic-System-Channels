using JLD2
using GilaElectromagnetics
using LinearMaps

function _performance_smr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    throw(ArgumentError("SMR systems not implemented"))
end

# function bicgstab!(x::AbstractVector, op, b::AbstractVector, max_iter::Int=size(op, 2)*10, atol::Real=zero(real(eltype(b))), rtol::Real=sqrt(eps(real(eltype(b)))), initial_x_is_zero::Bool=false, verbose::Bool=false)
#     T = eltype(b)
#     r = similar(b)
#     r̂ = similar(b)
#     p = similar(b)
#     v = similar(b)
#     s = similar(b)
#     t = similar(b)
#     if initial_x_is_zero
#         copy!(r, b)
#     else
#         op_mul!(r, op, x)
#         @. r = b - r
#     end
# end

function _performance_smr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out = jldopen(jld_out_path, "w")

    ρ_opt = read_array(jdl_out, "ρ_opt", use_gpu(compute_env))
    stage_history = jld_out["stage_history"]

    G₀_uu = load_greens_function(compute_env, smr, Design, Design)
    s_projector, u_projector, G₀_uu_disjoint = projected_operators(G₀_uu, smr)

    Winv_action(v, X) = v - (X .* (G₀_uu_disjoint * v))
    Winv_dag_action(v, X) = v - (G₀_uu_disjoint' * (conj.(X) .* v))
    W_inv_factory(X) = LinearMap(v -> Winv_action(v, X), v -> Winv_dag_action(v, X), size(G₀_uu_disjoint)...; ismutating=false)

    function objective_action(x, k)
        out = similar(x)
        fill!(out, zero(eltype(x)))
        @views for i in k:rank(rsvd_params)
            out .+= (ζ * Γ[i]) .* (Vur_asym[:, i] * (Vur_asym[:, i]' * x))
        end
        return out
    end
    objective_op(k) = LinearMap(x -> objective_action(x, k), x -> objective_action(x, k), size(Vur_asym, 1), size(Vur_asym, 1); ismutating=false)
end

function performance()
    compute_env, smr, rsvd_params = parse_args()
    gpu = use_gpu(compute_env)
    if gpu
        @info string(now()) * " [verlan] GPU mode, device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [verlan] CPU mode"
    end
    
    if isnothing(mediator(smr))
        return _performance_sr(compute_env, smr, rsvd_params)
    end
    return _performance_smr(compute_env, smr, rsvd_params)
end