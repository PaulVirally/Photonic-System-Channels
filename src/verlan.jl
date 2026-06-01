using GilaElectromagnetics
using LinearMaps
using LinearAlgebra
using Optim
using NLSolversBase
using LineSearches
using StatsBase
using BenchmarkTools
using Random

const COMPLEX_T = ComplexF32
const REAL_T = Float32

function _verlan_smr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [verlan::_verlan_smr] Performing TopOpt with Verlan seed for SMR system"
    
    error("Not implemented yet")
end

# Return (I - XG), the inverse of the scattering operator, in the U_uu basis
W_inv_subspace(ρ::AbstractVector, χ, G₀_uu_disjoint_sub_cols::AbstractMatrix, U_uu::AbstractMatrix) = I - χ * (U_uu' * (ρ .* G₀_uu_disjoint_sub_cols))

# Return the Hermitian quadratic form matrix associated with the objective function in the subspace spanned by U_uu
function objective_matrix_subspace(k::Int, U_uu::AbstractMatrix, Vur_asym::AbstractMatrix, Γ::AbstractVector, ζ)
    Γ_cpu = Array(Γ)
    vdagUmat = Vur_asym' * U_uu
    # if k == 1
    #     return Hermitian(ζ .* (vdagUmat' * (Diagonal(Γ) * vdagUmat)))
    # end
    objective_mat = similar(U_uu, eltype(U_uu), size(U_uu, 2), size(U_uu, 2))
    fill!(objective_mat, zero(eltype(objective_mat)))
    rank1 = similar(U_uu, eltype(U_uu), size(U_uu, 2), size(U_uu, 2))
    for i in reverse(1:size(U_uu, 2))
        row = view(vdagUmat, i:i, :)
        mul!(rank1, row', row) # (v'U)' * v'U = U'vv'U
        objective_mat .+= (ζ * Γ_cpu[i]) .* rank1
        if i == k
            break
        end
    end
    return Hermitian(objective_mat)
end

function adjoint_field_subspace(W_inv_sub::AbstractMatrix, obj_mat_sub::AbstractMatrix, w_sub::AbstractVector, ζ)
    vec = (2ζ) .* (obj_mat_sub * w_sub)
    return W_inv_sub' \ vec
end

function density_gradient(W_inv_sub::AbstractMatrix, obj_mat_sub::AbstractMatrix, w_sub::AbstractVector, χ, G₀_uu_disjoint_sub_cols::AbstractMatrix, U_uu::AbstractMatrix, ζ)
    λ_subs = adjoint_field_subspace(W_inv_sub, obj_mat_sub, w_sub, ζ)
    ∇ = real.(χ .* (conj.(U_uu * λ_subs) .* (G₀_uu_disjoint_sub_cols * w_sub))) # ∂f/∂ρ
    return ∇ # The gradient in the full space
end

function optimal_source(S::AbstractMatrix, obj_mat_sub::AbstractMatrix, W_subs::AbstractMatrix, U_s::AbstractMatrix)
    B = W_subs' * (obj_mat_sub * W_subs) # W_subs = inv(W_inv_sub)
    out = eigen(Hermitian(S * B * S), Hermitian(S)) # S = U_s' * U_s
    max_evec = out.vectors[:, argmax(out.values)]
    q = U_s * max_evec # Map back to the full space
    return q / norm(q) # Source is normalized in full space
end

sigmoid(θ, β=one(eltype(θ))) = one(β) / (one(β) + exp(-β * θ))
inverse_sigmoid(ρ, β=one(eltype(ρ))) = log(ρ / (one(β) - ρ)) / β
deriv_sigmoid(θ, β=one(eltype(θ))) = β * sigmoid(θ, β) * (one(eltype(θ)) - sigmoid(θ, β))

function chunked_gemm!(C, A, B, α, β; chunk=4096)
    # A is N×k, B is N×k, C is k×k
    # We want C = α * A^H * B + β * C
    N, k = size(A)
    # Initialize with β scaling
    C .*= β
    for i in 1:chunk:N
        r = i:min(i+chunk-1, N)
        CUBLAS.gemm!('C', 'N', α, A[r, :], B[r, :], one(eltype(C)), C)
    end
end

function bencher(obj_mat_sub, χ, ζ, q_sub, G₀_uu_disjoint_sub_cols, U_uu, β, bound, tikhonov_parameter=1e-3, max_grad_norm=1e3)
    N, k = size(U_uu)

    # k = 512

    GC.gc()
    GC.gc()
    GC.gc()

    # Workspace
    CT = complex(eltype(obj_mat_sub))
    RT = real(eltype(obj_mat_sub))
    ρU = similar(obj_mat_sub, CT, N, k) # diag(ρ) * U_uu
    W_inv_sub = similar(obj_mat_sub, CT, k, k) # inv(W) in the subspace
    W_inv_sub_copy = similar(W_inv_sub) # Copy for LU factorization
    w_sub = similar(obj_mat_sub, CT, k) # Scattered field in the subspace
    λ_sub = similar(obj_mat_sub, CT, k) # Adjoint field in the subspace
    Hw_sub = similar(obj_mat_sub, CT, k) # Action of objective matrix on w_sub
    Uλ = similar(obj_mat_sub, CT, N) # U_uu * λ_sub
    Fw = similar(obj_mat_sub, CT, N) # G₀_uu_disjoint_sub_cols * w_sub
    rhs = similar(obj_mat_sub, CT, k) # Right-hand side for adjoint solve
    ρ_buf = similar(obj_mat_sub, RT, N) # Sigmoid buffer for ρ
    ∇ρ = similar(obj_mat_sub, RT, N) # Gradient w.r.t ρ

    # Fill with random data for benchmarks
    # obj_mat_sub = CUDA.rand(CT, k, k) # Placeholder random matrix for benchmarks
    # GC.gc()
    # q_sub = CUDA.rand(CT, k) # Placeholder random vector for benchmarks
    # GC.gc()
    # G₀_uu_disjoint_sub_cols = CUDA.rand(CT, N, k)
    # GC.gc()
    # U_uu = CUDA.rand(CT, N, k) # Placeholder random matrix for benchmarks
    # GC.gc()

    neg_χ = -χ
    two_ζ = CT(2ζ)

    # if obj_mat_sub isa CuArray
        θ = CUDA.rand(RT, N)
    # else
        # θ = rand(RT, N)
    # end
    G = similar(obj_mat_sub, RT, N) # Placeholder for gradient output

    #### START OF fg! ####

    @info string(now()) * " [verlan::bencher] Some info"
    @show typeof(ρU)
    @show typeof(G₀_uu_disjoint_sub_cols)
    @show size(ρU), sizeof(G₀_uu_disjoint_sub_cols)
    @show ρU.dims, strides(ρU)

    b = @benchmark begin
        @. $ρ_buf = sigmoid($θ, $β)
        @. $ρU = $ρ_buf * $U_uu
    end
    display(b)
    @. ρ_buf = sigmoid(θ, β)
    @. ρU = ρ_buf * U_uu

    p = CUDA.@profile CUDA.@sync CUBLAS.gemm!('C', 'N', neg_χ, ρU, G₀_uu_disjoint_sub_cols, zero(CT), W_inv_sub)
    display(p)

    p = CUDA.@profile CUDA.@sync chunked_gemm!(W_inv_sub, ρU, G₀_uu_disjoint_sub_cols, -χ, zero(CT))
    display(p)

    @show typeof(ρU)
    @show typeof(G₀_uu_disjoint_sub_cols)
    @show typeof(W_inv_sub)

    b = @benchmark begin
        # CUBLAS.gemm!('C', 'N', $neg_χ, $ρU, $G₀_uu_disjoint_sub_cols, zero($CT), $W_inv_sub)
        chunked_gemm!($W_inv_sub, $ρU, $G₀_uu_disjoint_sub_cols, -$χ, zero($CT))
    end
    # CUBLAS.gemm!('C', 'N', neg_χ, ρU, G₀_uu_disjoint_sub_cols, zero(CT), W_inv_sub)
    chunked_gemm!(W_inv_sub, ρU, G₀_uu_disjoint_sub_cols, -χ, zero(CT))
    display(b)

    @show typeof(W_inv_sub)
    b = @benchmark begin
        $W_inv_sub[diagind($W_inv_sub)] .+= (one($CT) + $tikhonov_parameter)
    end
    W_inv_sub[diagind(W_inv_sub)] .+= (one(CT) + tikhonov_parameter)
    display(b)

    b = @benchmark begin
        copyto!($W_inv_sub_copy, $W_inv_sub)
        W_fac = lu!($W_inv_sub_copy)
        ldiv!($w_sub, W_fac, $q_sub)
    end
    display(b)
    copyto!(W_inv_sub_copy, W_inv_sub)
    W_fac = lu!(W_inv_sub_copy)
    ldiv!(w_sub, W_fac, q_sub)

    b = @benchmark begin
        mul!($Hw_sub, $obj_mat_sub, $w_sub)
        val = $ζ * real(dot($w_sub, $Hw_sub))
    end
    display(b)
    mul!(Hw_sub, obj_mat_sub, w_sub)
    val = ζ * real(dot(w_sub, Hw_sub))

    b = @benchmark begin
        @. $rhs = $two_ζ * $Hw_sub
        ldiv!($λ_sub, $W_fac', $rhs)

        mul!($Uλ, $U_uu, $λ_sub)
        mul!($Fw, $G₀_uu_disjoint_sub_cols, $w_sub)

        @. $G = -real($χ * conj($Uλ) * $Fw) * ($β * $ρ_buf * (one($RT) - $ρ_buf))
    end

    #### END OF fg! ####
end

function topopt_fg!_factory(obj_mat_sub, χ, ζ, q_sub, G₀_uu_disjoint_sub_cols, U_uu, β, bound, tikhonov_parameter=1e-3, max_grad_norm=1e3)
    N, k = size(U_uu)

    # Workspace
    CT = complex(eltype(obj_mat_sub))
    RT = real(eltype(obj_mat_sub))
    ρU_conj = similar(obj_mat_sub, CT, N, k) # diag(ρ) * conj(U_uu)
    W_inv_sub = similar(obj_mat_sub, CT, k, k) # inv(W) in the subspace
    W_inv_sub_copy = similar(W_inv_sub) # Copy for LU factorization
    w_sub = similar(obj_mat_sub, CT, k) # Scattered field in the subspace
    λ_sub = similar(obj_mat_sub, CT, k) # Adjoint field in the subspace
    Hw_sub = similar(obj_mat_sub, CT, k) # Action of objective matrix on w_sub
    Uλ = similar(obj_mat_sub, CT, N) # U_uu * λ_sub
    Fw = similar(obj_mat_sub, CT, N) # G₀_uu_disjoint_sub_cols * w_sub
    rhs = similar(obj_mat_sub, CT, k) # Right-hand side for adjoint solve
    ρ_buf = similar(obj_mat_sub, RT, N) # Sigmoid buffer for ρ
    ∇ρ = similar(obj_mat_sub, RT, N) # Gradient w.r.t ρ

    U_uu_conj = conj.(U_uu)

    function fg!(F, G, θ)
        @. ρ_buf = sigmoid(θ, β)
        @. ρU_conj = ρ_buf * U_uu_conj

        # W_inv_sub = I - χ U_uu' * ρF
        mul!(W_inv_sub, transpose(ρU_conj), G₀_uu_disjoint_sub_cols, -χ, zero(CT))
        W_inv_sub[diagind(W_inv_sub)] .+= (one(eltype(W_inv_sub)) + tikhonov_parameter) # Add (1 + tikhonov_parameter) to the diagonal for Tikhonov regularization

        copyto!(W_inv_sub_copy, W_inv_sub)
        W_fac = lu!(W_inv_sub_copy)
        ldiv!(w_sub, W_fac, q_sub)
        # if any(!isfinite, w_sub) # TODO: remove check?
        #     @warn string(now()) * " [verlan::fg!] Non-finite value encountered in w_sub. Returning Inf for objective and zero gradient."
        #     G !== nothing && fill!(G, zero(eltype(G)))
        #     F !== nothing && return -Inf # -bound
        #     return nothing
        # end

        mul!(Hw_sub, obj_mat_sub, w_sub)
        val = ζ * real(dot(w_sub, Hw_sub))

        # if val < bound
        #     @warn string(now()) * " [verlan::fg!] Objective value $(val) exceeds bound $(bound). This may indicate numerical issues. Returning bound for objective and zero gradient."
        #     G !== nothing && fill!(G, zero(eltype(G)))
        #     F !== nothing && return -bound
        #     return nothing
        # end

        if G !== nothing
            @. rhs = (2ζ) * Hw_sub
            ldiv!(λ_sub, W_fac', rhs)

            mul!(Uλ, U_uu, λ_sub)
            mul!(Fw, G₀_uu_disjoint_sub_cols, w_sub)

            # @. ∇ρ = real(χ * conj(Uλ) * Fw) # Gradient for ρ
            # @. G = -(∇ρ * β * ρ_buf * (one(eltype(G)) - ρ_buf)) # Chain rule for sigmoid: dF/dθ = dF/dρ * dρ/dθ = ∇ρ[i] * β * ρ_buf[i] * (1 - ρ_buf[i]), and negative because we're minimizing -f
            @. G = -real(χ * conj(Uλ) * Fw) * (β * ρ_buf * (one(RT) - ρ_buf))

            # grad_norm = norm(G)
            # if grad_norm > max_grad_norm
            #     G .*= (max_grad_norm / grad_norm)
            # end
        end

        if F !== nothing
            return -val # Negative because we're minimizing -f
        end
    end

    return fg!
end

function _verlan_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [verlan::_verlan_sr] Performing TopOpt with Verlan seed for SR system"

    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out = jldopen(jld_out_path, "r")
    # @info string(now()) * " [verlan::_verlan_sr] Loading verlan seed ρ_avg from $(jld_out_path)"
    which_ρ = "ρ_verlan_1"
    # which_ρ = "ρ_verlan_avg"
    @info string(now()) * " [verlan::_verlan_sr] Loading verlan seed $(which_ρ) from $(jld_out_path)"
    ρ = REAL_T.(read_array(jld_out, which_ρ, use_gpu(compute_env)))
    ordering_idxs = jld_out["ordering_idxs"]
    χ = COMPLEX_T(jld_out["χ"])
    ζ = abs2(χ) / imag(χ)
    k = 1
    bound = REAL_T.(jld_out["μ_subspace"][k])
    close(jld_out)

    @info string(now()) * " [verlan::_verlan_sr] Loading associated data for Verlan seed from $(jld_out_path)"
    jld_in_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_in = jldopen(jld_in_path, "r")

    U_uu = read_array(jld_in, "UU_asym/V", use_gpu(compute_env))
    # temp = similar(U_uu, eltype(U_uu), size(U_uu)[1], 512)
    # fill!(temp, zero(eltype(temp)))
    # copyto!(temp, U_uu[:, ordering_idxs])
    # U_uu = temp

    @info string(now()) * " [verlan::_verlan_sr] Loaded U_uu with size $(size(U_uu)) and type $(eltype(U_uu))"
    Vur_asym = COMPLEX_T.(read_array(jld_in, "UR_asym/V", use_gpu(compute_env))[:, ordering_idxs])
    Vur_asym = CUDA.@allowscalar Vur_asym[:, ordering_idxs]
    Γ = CUDA.@allowscalar -read_array(jld_in, "UR_asym/D", use_gpu(compute_env))[ordering_idxs] # Negative to account for sign error in notes
    Γ = REAL_T.(Γ)
    close(jld_in)

    @info string(now()) * " [verlan::_verlan_sr] Loading Green's function"
    G₀_uu = load_greens_function(compute_env, smr, Design, Design)
    Πₛ, Πᵤ, G₀_uu_disjoint = projected_operators(G₀_uu, smr)

    @info string(now()) * " [verlan::_verlan_sr] Precomputing subspace operators for optimization loops"
    @info string(now()) * " [verlan::_verlan_sr] Precomputing F = G₀_uu_disjoint * U_uu (for gradient)"
    G₀_uu_disjoint_sub_cols = COMPLEX_T.(opmat(G₀_uu_disjoint, U_uu))
    @info string(now()) * " [verlan::_verlan_sr] Precomputing U_s = Πₛ * U_uu (for optimal source)"
    U_s_sub = COMPLEX_T.(opmat(Πₛ, U_uu))
    @info string(now()) * " [verlan::_verlan_sr] Precomputing S = U_s' * U_s (for optimal source)"
    S_sub = U_s_sub' * U_s_sub

    @info string(now()) * " [verlan::_verlan_sr] Converting data to $(COMPLEX_T) for optimization"
    U_uu = COMPLEX_T.(U_uu)
    G₀_uu_disjoint_sub_cols = COMPLEX_T.(G₀_uu_disjoint_sub_cols)
    U_s_sub = COMPLEX_T.(U_s_sub)
    S_sub = COMPLEX_T.(S_sub)

    @info string(now()) * " [verlan::_verlan_sr] Computing objective matrix in subspace"
    # obj_mat_sub = objective_matrix_subspace(1, U_uu, Vur_asym, Γ, ζ)
    obj_mat_sub = CUDA.rand(COMPLEX_T, size(U_uu, 2), size(U_uu, 2)) # Placeholder random matrix for benchmarks

    @info string(now()) * " [verlan::_verlan_sr] Computing optimal source for initial ρ"
    q = optimal_source(S_sub, obj_mat_sub, inv(W_inv_subspace(ρ, χ, G₀_uu_disjoint_sub_cols, U_uu)), U_s_sub)
    q_sub = U_uu' * q

    # Restrict ρ to [0, 1]
    ρ_min, ρ_max = REAL_T(1e-6), REAL_T(1.0 - 1e-6)
    clamp!(ρ, ρ_min, ρ_max) # Avoid exactly 0 or 1 for NaNs and Infs
    ρ_init = copy(ρ) # Save initial ρ for later analysis
    θ = inverse_sigmoid.(ρ) # Initialize θ based on the initial ρ

    inner_density_loops = 5000
    objective_history = REAL_T[]

    # βs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0] # Geometric ramping schedule
    # iters_per_β = [300, 200, 100, 50, 30, 10, 10, 10, 10] # More iterations for smaller β where the landscape is smoother
    βs = [1.0]
    iters_per_β = [1]
    num_β_stages = length(βs)

    tikhonov_parameters = REAL_T.([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0])
    iters_per_tikhonov = [3, 10, 20, 30, 50, 100, 200, 1000] # More iterations for smaller Tikhonov parameters as we get closer to the true optimum
    num_tikhonov_stages = length(tikhonov_parameters)

    @info string(now()) * " [verlan::_verlan_sr] Starting optimization loops with $(num_tikhonov_stages) Tikhonov stages, $(num_β_stages) β stages, and $(inner_density_loops) inner density loops per stage"
    for (tikhonov_stage, tikhonov_parameter) in enumerate(tikhonov_parameters)
        @info string(now()) * " [verlan::_verlan_sr] Starting Tikhonov stage with parameter = $(tikhonov_parameter) and $(iters_per_tikhonov[tikhonov_stage]) iterations"
        for tikhonov_iter in 1:iters_per_tikhonov[tikhonov_stage]
            for (β_stage, β) in enumerate(βs)
                @info string(now()) * " [verlan::_verlan_sr] [Tik $(tikhonov_stage)/$(num_tikhonov_stages) @ $(tikhonov_iter)/$(iters_per_tikhonov[tikhonov_stage])] Starting β stage with β = $(β) and $(iters_per_β[β_stage]) iterations"
                for β_iter in 1:iters_per_β[β_stage]
                    clamp!(ρ, ρ_min, ρ_max) # Ensure ρ stays within bounds to avoid NaNs/Infs
                    @. θ = inverse_sigmoid(ρ, β) # Update θ based on current ρ for the optimizer

                    # Function and gradient for Optim.jl
                    max_val = -Inf # -1.6 * bound
                    max_grad_norm = Inf
                    fg! = topopt_fg!_factory(obj_mat_sub, χ, ζ, q_sub, G₀_uu_disjoint_sub_cols, U_uu, β, max_val, tikhonov_parameter, max_grad_norm)

                    # @info string(now()) * " [verlan::_verlan_sr] Benchmarking different parts of fg!"
                    # bencher(obj_mat_sub, χ, ζ, q_sub, G₀_uu_disjoint_sub_cols, U_uu, β, bound, tikhonov_parameter)

                    # @info string(now()) * " [verlan::_verlan_sr] Benchmarking full fg! execution"
                    # F_ = CUDA.zeros(REAL_T, 1) # Placeholder for objective value
                    # G_ = CUDA.zeros(REAL_T, length(θ)) # Placeholder for gradient
                    # b = @benchmark $fg!($F_, $G_, $θ) seconds=30
                    # display(b)

                    @info string(now()) * " [verlan::_verlan_sr] [Tik $(tikhonov_stage)/$(num_tikhonov_stages) @ $(tikhonov_iter)/$(iters_per_tikhonov[tikhonov_stage])] [β $(β_stage)/$(num_β_stages) @ $(β_iter)/$(iters_per_β[β_stage])] Running $(inner_density_loops) optimization steps with LBFGS"
                    res = optimize(
                        NLSolversBase.only_fg!(fg!), θ,
                        LBFGS(
                                 alphaguess=LineSearches.InitialStatic(),
                                 linesearch=LineSearches.HagerZhang() # HagerZhang()? BackTracking()?
                        ),
                        Optim.Options(
                            iterations=inner_density_loops,
                            show_trace=true,
                        )
                    )
                    θ .= Optim.minimizer(res)
                    @. ρ = sigmoid(θ, β)
                    ρ[θ .== -Inf] .= ρ_min
                    ρ[θ .== Inf] .= ρ_max
                    ρ[.!(isfinite.(ρ))] .= ρ_min # Fix potential NaNs/Infs in ρ
                    objective = -Optim.minimum(res) # Negate because we minimized the negative of the objective
                    push!(objective_history, objective)

                    grayness = mean(4 .* ρ .* (1.0 .- ρ)) # 0 = binary, 1 = gray
                    dist_to_init = norm(ρ - ρ_init) / norm(ρ_init)

                    @info string(now()) * " [verlan::_verlan_sr] [Tik $(tikhonov_stage)/$(num_tikhonov_stages) @ $(tikhonov_iter)/$(iters_per_tikhonov[tikhonov_stage])] [β $(β_stage)/$(num_β_stages) @ $(β_iter)/$(iters_per_β[β_stage])] f = $(objective) ($(round(100*objective/bound, sigdigits=3))% of the bound), grayness = $(grayness), relative distance from Verlan seed = $(dist_to_init)"

                    q = optimal_source(S_sub, obj_mat_sub, inv(W_inv_subspace(ρ, χ, G₀_uu_disjoint_sub_cols, U_uu)), U_s_sub)
                    q_sub = U_uu' * q
                end
            end
        end
    end
    @info string(now()) * " [verlan::_verlan_sr] Completed all optimization loops"
    @warn "Do something with q and ρ"
    @show Array(ρ)
    @show objective_history
    return nothing
end

function verlan()
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [verlan::verlan] Using GPU acceleration on device $(gpu_device(compute_env))"
        CUDA.device!(gpu_device(compute_env))
    else
        @info string(now()) * " [verlan::verlan] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _verlan_sr(compute_env, smr, rsvd_params)
    else
        _verlan_smr(compute_env, smr, rsvd_params)
    end
end
