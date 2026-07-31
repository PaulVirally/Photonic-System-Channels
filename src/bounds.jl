using GilaElectromagnetics
using CUDA
using Plots
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
# using .Projectors
using MatrixFreeRandomizedLinearAlgebra

function midpoint(left::T, right::T) where T <: Real
    φ = (1 + sqrt(5))/2
    if isfinite(left) && isfinite(right)
        return (left + right) / 2
    elseif isfinite(left) && !isfinite(right)
        return left * φ
    elseif !isfinite(left) && isfinite(right)
        return right / φ
    end
    # Both left and right are infinite; return zero if one is negative and the other is positive
    if left * right < zero(T)
        return zero(T)
    end
    @warn "[midpoint] Both left == right == ∞ "
    return left
end

issign(x::T, s::Symbol) where T <: Real = (s == :positive && x > zero(T)) || (s == :negative && x < zero(T))

function bisect_safe(f::Function, left::T, right::T, desired_sign::Symbol; f_left::Union{T, Nothing}=nothing, f_right::Union{T, Nothing}=nothing) where T <: Real
    if isnothing(f_left)
        f_left = f(left)
    end
    if isnothing(f_right)
        f_right = f(right)
    end

    if isfinite(f_left) && issign(f_left, desired_sign)
        return ((left, f_left), (left, f_left), (right, f_right))
    end
    if isfinite(f_left) && issign(f_right, desired_sign)
        return ((right, f_right), (left, f_left), (right, f_right))
    end

    while true
        mid = midpoint(left, right)
        f_mid = f(mid)
        if isfinite(f_mid) && issign(f_mid, desired_sign)
            return ((mid, f_mid), (left, f_left), (right, f_right))
        elseif isfinite(f_mid) && issign(f_left, desired_sign)
            right, f_right = mid, f_mid
        elseif isfinite(f_mid) && issign(f_right, desired_sign)
            left, f_left = mid, f_mid
        else
            @error " [bisect_safe] Neither side has the desired sign; initial interval is not valid"
            return ((mid, f_mid), (left, f_left), (right, f_right))
        end
    end
end

function refine_interval(f, interval)
    left, right = interval
    f_left, f_right = f(left), f(right)
    T = promote_type(eltype(f_left), eltype(f_right))
    φ = T((1 + sqrt(5))/2) 

    # We want to refine (left, right) so that f(left) < 0 and f(right) > 0

    if !isfinite(f_left) || f_left > zero(T)
        (left, f_left), _, _ = bisect_safe(f, left, midpoint(left, right), :negative, f_left=f_left, f_right=f_right)
    end
    if !isfinite(f_right) || f_right < zero(T)
        (right, f_right), _, _ = bisect_safe(f, midpoint(left, right), right, :positive, f_left=f_left, f_right=f_right)
    end
    return ((left, f_left), (right, f_right))
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

similar_fill(v::AbstractArray{T}, fill_val::T) where T = fill!(similar(v), fill_val)
similar_fill(v::AbstractArray{T}, dims::NTuple{N, Int}, fill_val::T) where {N, T} = fill!(similar(v, dims), fill_val)
Base.:\(::Nothing, x::AbstractArray) = (x, 0)
function lmul_mvp!(y::AbstractVector, ::Nothing, x::AbstractVector)
	copyto!(y, x)
	return y, 0
end

function bicgstab_gpu!(x::AbstractVector, op, b::AbstractVector; preconditioner=nothing, max_iter::Int=size(op, 2), atol::Real=zero(real(eltype(b))), rtol::Real=sqrt(eps(real(eltype(b)))), verbose::Bool=false, initial_zero::Bool=false, print_every::Int=1)
    T = eltype(b)
    mvp = 0
    if initial_zero
        fill!(x, zero(T))
        residual = deepcopy(b)
    else
        residual = b - op * x
        mvp += 1
    end
    atol = max(atol, rtol * norm(residual))
    residual_shadow = deepcopy(residual)
    p = deepcopy(residual)
    v = similar_fill(b, zero(T))
    ρ_prev = one(T); α = one(T); ω = one(T)

    for num_iter in 1:max_iter
        norm(residual) < atol && return x, mvp
        ρ = dot(residual_shadow, residual)
        if num_iter > 1
            β = (ρ / ρ_prev) * (α / ω)
            @. p = residual + β * (p - ω * v)
        end
        p̂, k = preconditioner \ p; mvp += k
        v = op * p̂; mvp += 1
        α = ρ / dot(residual_shadow, v)
        @. residual = residual - α * v
        if norm(residual) < atol
            @. x = x + α * p̂
            return x, mvp
        end
        ŝ, k = preconditioner \ residual; mvp += k
        t = op * ŝ; mvp += 1
        ω = dot(t, residual) / dot(t, t)
        @. x = x + α * p̂ + ω * ŝ
        @. residual = residual - ω * t
        ρ_prev = ρ
        # verbose && println(num_iter, " ", norm(residual), " > ", atol, " (mvp: ", mvp, ")")
        verbose && if num_iter % print_every == 0
            println(num_iter, " ", norm(residual), " > ", atol, " (mvp: ", mvp, ")")
        end
    end
    throw("BiCGStab did not converge after $max_iter iterations.")
end

function bicgstab_gpu(op, b::AbstractVector; preconditioner=nothing, max_iter::Int=length(b), atol::Real=zero(real(eltype(b))), rtol::Real=sqrt(eps(real(eltype(b)))), verbose::Bool=false)
	x = similar_fill(b, zero(eltype(b)))
	return bicgstab_gpu!(x, op, b; preconditioner=preconditioner, max_iter=max_iter, atol=atol, rtol=rtol, verbose=verbose, initial_zero=true)
end

function projected_operators(G₀_uu::AbstractGlaOpr, smr::SMRSystem, env::ComputeEnvironment)
    s = sender(smr)
    r = receiver(smr)
    # G₀_uu.mem.srcVol == G₀_uu.mem.trgVol || error("G₀_uu is not a self operator")
    # union_volume = G₀_uu.mem.srcVol # srcVol == trgVol
    # if union(s, r) != union_volume
    #     @error "union_volume should be union(s, r) but it is not"
    # end
    # sender_mask = GilaElectromagnetics.GilaOperators.mskRng(s, union_volume) # Mask for sender region within the union volume
    # receiver_mask = GilaElectromagnetics.GilaOperators.mskRng(r, union_volume) # Mask for receiver region within the union volume

    # Zero out the gap between sender and receiver by creating a projector that keeps the sender and receiver regions but zeros out everything else in the union volume (including the gap between s and r)
    # disjoint_union_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    # if use_gpu(env)
    #     disjoint_union_indicator = CuArray(disjoint_union_indicator)
    # end
    # fill!(view(disjoint_union_indicator, sender_mask..., :), one(eltype(disjoint_union_indicator)))
    # fill!(view(disjoint_union_indicator, receiver_mask..., :), one(eltype(disjoint_union_indicator)))
    # disjoint_union_projector_action!(w, v) = begin
    #     w .= vec(disjoint_union_indicator .* reshape(v, size(disjoint_union_indicator)))
    #     return w
    # end
    # u_projector = LinearMap{ComplexF64}(disjoint_union_projector_action!, disjoint_union_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    # sender_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    # if use_gpu(env)
    #     sender_indicator = CuArray(sender_indicator)
    # end
    # fill!(view(sender_indicator, sender_mask..., :), one(eltype(sender_indicator)))
    # s_projector_action!(w, v) = begin
    #     w .= vec(sender_indicator .* reshape(v, size(sender_indicator)))
    #     return w
    # end
    # s_projector = LinearMap{ComplexF64}(s_projector_action!, s_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    sender_size = prod(s.cel)*3
    receiver_size = prod(r.cel)*3
    sender_projector_action!(s_included_in_u::AbstractVector{ComplexF64}, u::AbstractVector{ComplexF64}) = begin
        # our convention is [sender; receiver]
        fill!(s_included_in_u, zero(eltype(s_included_in_u)))
        copyto!(view(s_included_in_u, 1:sender_size), view(u, 1:sender_size))
        return s_included_in_u
    end
    s_projector = LinearMap{ComplexF64}(sender_projector_action!, sender_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)

    # receiver_indicator = zeros(eltype(G₀_uu), glaSze(G₀_uu)[2])
    # if use_gpu(env)
    #     receiver_indicator = CuArray(receiver_indicator)
    # end
    # fill!(view(receiver_indicator, receiver_mask..., :), one(eltype(receiver_indicator)))
    # r_projector_action!(w, v) = begin
    #     w .= vec(receiver_indicator .* reshape(v, size(receiver_indicator)))
    #     return w
    # end
    # r_projector = LinearMap{ComplexF64}(r_projector_action!, r_projector_action!, size(G₀_uu)...; ismutating=true, ishermitian=true)
    #
    # G₀ = LinearMap(G₀_uu)
    # return r_projector, s_projector, u_projector, (u_projector * G₀ * u_projector)

    return s_projector
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

function _compute_bounds_smr(::ComputeEnvironment, ::SMRSystem, ::RSVDParams)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_smr] Computing bounds for SMR system"
    throw("Not implemented yet")
end

function load_bounds_inputs(compute_env::ComputeEnvironment, smr::SMRSystem)
    jld_in_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_in = jldopen(jld_in_path, "r")

    Γ = read_array(jld_in, "UR_asym/D", use_gpu(compute_env))
    # NO MORE SIGN TYPO (this is fixed in rsvd.jl)
    # Γ .*= -one(eltype(Γ)) # Sign typo in the original notes
    Vur_asym = read_array(jld_in, "UR_asym/V", use_gpu(compute_env))

    # Sort the singular values and vectors in descending order
    sorted_idxs = sortperm(Γ, rev=true)
    Vur_asym = Vur_asym[:, sorted_idxs]
    Γ = Array(Γ[sorted_idxs])
    Γrs = Array(jld_in["RS/D"])

    close(jld_in)
    return (Γ=Γ, Vur_asym=Vur_asym, Γrs=Γrs, sorted_idxs=Array(sorted_idxs))
end

"""
    bounds_from_spectrum(compute_env, smr, Γ, Vur_asym, Γrs; kwargs...)

Compute the σₙ(Pᵣₛ) bounds from an already-loaded `Asym(G⁰ᵤᵣ)` spectrum. `Γ` must
be sorted in descending order and `Vur_asym`'s columns must be ordered to match.

# Keyword arguments
- `basis_size`: how many leading eigenvectors to use as the projection basis.
- `G₀_uu`: pre-loaded universe operator, loaded here if not supplied.
- `outer_indices`: which `n` of the outer `σₙ` loop to actually evaluate.
  `nothing` (the default) means all of them.
- `on_outer_error`: `:throw` (the default) or `:stop`. With `:stop`, a failure in
  the outer loop is recorded in the returned `outer_error` and the function
  returns with `complete = false` rather than propagating. The benchmark harness
  passes `:stop` so that the setup-stage timings, which are measured before the
  loop and are useful on their own, survive a loop that cannot run on synthetic
  input.

# Returns
A named tuple with the bounds, the bookkeeping needed to save them, and
`stage_times` / `outer_times` for calibration.
"""
function bounds_from_spectrum(compute_env::ComputeEnvironment, smr::SMRSystem,
                              Γ::AbstractVector, Vur_asym::AbstractMatrix,
                              Γrs::AbstractVector;
                              basis_size::Int=size(Vur_asym, 2),
                              G₀_uu=nothing,
                              outer_indices::Union{Nothing,AbstractVector{Int}}=nothing,
                              on_outer_error::Symbol=:throw)
    on_outer_error in (:throw, :stop) ||
        throw(ArgumentError("on_outer_error must be :throw or :stop, got :$on_outer_error"))
    # U_uu = read_array(jld_in, "UU/U", use_gpu(compute_env)) # TODO: could use this as basis too
    RSVD_BASIS_SIZE = min(basis_size, size(Vur_asym, 2))
    basis = copy(Vur_asym)
    # basis = cat(U_uu, Vur_asym; dims=2)
    # basis = qthin!(basis) # Orthonormalize the basis using QR factorization
    basis = basis[:, 1:RSVD_BASIS_SIZE] # Restrict the basis to the top RSVD_BASIS_SIZE singular vectors
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Using RSVD_BASIS_SIZE = $RSVD_BASIS_SIZE"

    if isnothing(G₀_uu)
        G₀_uu = load_green_function(compute_env, smr, [Sender, Receiver], [Sender, Receiver]) # universe -> universe
    end
    # r_projector, s_projector, u_projector, G₀_uu_disjoint = projected_operators(G₀_uu, smr, compute_env)
    s_projector = projected_operators(G₀_uu, smr, compute_env)
    # G⁰ᵤᵤ_asym = u_projector * asym(LinearMap(G₀_uu)) * u_projector
    G⁰ᵤᵤ_asym = asym(LinearMap(G₀_uu))

    GC.gc()
    GC.gc()
    GC.gc()

    # @info "hello" size(Vur_asym) size(G₀_uu) size(s_projector) size(Γ) size(Γrs)

    χ = susceptibility(smr)
    ζ = abs(χ)^2/imag(χ)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Susceptibility χ = $χ, material factor ζ = $ζ"

    num_pos = count(Γ .> zero(eltype(Γ)))
    Γ_pos = Γ[1:num_pos] # These have been sorted in descending order; keep only the positive eigenvalues
    if use_gpu(compute_env)
        Γ_pos = CuArray(Γ_pos)
    end
    gs_pos = Vur_asym[:, 1:num_pos] # These have been sorted in descending order of the corresponding Γ values; keep only the eigenvectors with positive eigenvalues

    # Reverse Gram-Schmidt
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Performing reverse Gram-Schmidt to construct the ss basis"
    t_gram_schmidt = time_ns()
    ss = similar(gs_pos, size(gs_pos, 1), num_pos)
    for i in num_pos:-1:1
        gᵢ = view(gs_pos, :, i)
        wᵢ = s_projector * gᵢ
        for j in (i+1):num_pos
            sⱼ = view(ss, :, j)
            cᵢⱼ = dot(sⱼ, wᵢ)
            wᵢ .-= cᵢⱼ * sⱼ
        end
        nrm = norm(wᵢ)
        if nrm < 1e-12
            @warn string(now()) * " [bounds_bargaining::bounds_from_spectrum] Warning: vector $i is nearly linearly dependent on the later vectors, norm after orthogonalization is $nrm (we should stop the basis generation here, but I'm too lazy to fix the code right now; hopfully we never see this warning)"
        end
        ss[:, i] .= wᵢ ./ nrm
    end
    t_gram_schmidt = (time_ns() - t_gram_schmidt) / 1e9

    t_ss_basis = time_ns()
    ss_basis = basis' * ss
    t_ss_basis = (time_ns() - t_ss_basis) / 1e9

    B_matvec(n::Int, v::AbstractVector) = begin
        idxs = n:num_pos
        G = view(gs_pos, :, idxs) # N × (num_pos - n + 1)
        weights = (G' * v) .* Γ_pos[idxs]
        return (4/ζ) .* (G * weights) # Computes (4/ζ) ∑  γₗ gₗ gₗ' v for all m ≥ l ≥ n (with m = num_pos)
    end
    B(n) = LinearMap(v -> B_matvec(n, v), size(G₀_uu)...; ishermitian=true)
    B_basis(n) = basis' * opmat(B(n), basis)

    C_matvec(v::AbstractVector) = begin
        out = similar_fill(v, zero(eltype(v)))

        out .+= (1/ζ) * (s_projector * v) # ζ⁻¹ Pₛ action

        G = gs_pos # N × num_pos
        A_weights = (G' * v) .* Γ_pos
        out .+= (G * A_weights) # A₊ action

        out .+= G⁰ᵤᵤ_asym * v # (G⁰ᵤᵤ)ᵃ
        return out
    end
    C = LinearMap(C_matvec, size(G₀_uu)...; ishermitian=true)
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Projecting C into the basis of size $(size(basis, 2))"
    t_c_projection = time_ns()
    C_basis = basis' * opmat(C, basis)
    t_c_projection = (time_ns() - t_c_projection) / 1e9

    B_basis_diagonal = similar_fill(C_basis, (size(C_basis, 1),), zero(eltype(C_basis)))

    bounds_dual_basis = zeros(Float64, num_pos)
    B_basis_n = similar(C_basis)
    ns = isnothing(outer_indices) ? (1:num_pos) : filter(n -> 1 <= n <= num_pos, outer_indices)
    complete = length(ns) == num_pos
    outer_times = Tuple{Int,Float64}[]
    outer_error = nothing
    for n in ns # Compute bounds on σₙ(Pᵣₛ)
     try
        t_outer = time_ns()
        @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] [$n/$(num_pos)] Computing σₙ(Pᵣₛ) bound"

        @info string(now()) * " [$n/$(num_pos)] Projecting Bₙ into the basis of size $(size(basis, 2))"
        # B_basis_n = B_basis(n)
        fill!(B_basis_diagonal, zero(eltype(B_basis_diagonal)))
        B_basis_diagonal[n:num_pos] .= (4/ζ) .* Γ_pos[n:num_pos]
        B_basis_n .= diagm(B_basis_diagonal) # We can skip the projection step for Bₙ since Bₙ is diagonal in the gs_pos basis and the basis is just a change of basis from gs_pos

        # Solve GEVP
        @info string(now()) * " [$n/$(num_pos)] Solving λⱼ(Bₙ, C) in the basis"
        basis_fact = eigen!(Hermitian(copy(B_basis_n)), Hermitian(copy(C_basis))) # The copies are because CUDA can't do eigen, only eigen! for some reason
        V_basis = basis_fact.vectors
        Λ_basis = Array(basis_fact.values)
        interval_basis = (maximum(Λ_basis), Inf)

        best_dual = -Inf
        for k in n:num_pos
            sₖ_basis = view(ss_basis, :, k)
            b_basis = Array(V_basis' * sₖ_basis)

            fₖ_basis(α) = sum(abs2(bⱼ) * (α - 2λⱼ)/(α - λⱼ)^2 for (bⱼ, λⱼ) in zip(b_basis, Λ_basis))
            ((left, f_left), (right, f_right)) = refine_interval(fₖ_basis, interval_basis)
            # @info string(now()) * " [$n/$(num_pos)] [k=$k/$(num_pos)] Refined bracketing interval for root finding: ($left, $right) ↦  ($f_left, $f_right)"

            αₖ_opt_basis = find_zero(fₖ_basis, (left, right), Roots.Brent())
            dual_basis(α) = α^2/4 * sum(abs2(bⱼ) / (α - λⱼ) for (bⱼ, λⱼ) in zip(b_basis, Λ_basis))
            curr_dual = dual_basis(αₖ_opt_basis)
            if curr_dual > best_dual
                best_dual = curr_dual
            end
            # @info string(now()) * " [$n/$(num_pos)] [k=$k/$(num_pos)] Found root at α = $αₖ_opt_basis with dual value $(curr_dual) $(curr_dual > best_dual ? ">" : "<") $(best_dual) (best dual so far)"
        end
        dual = best_dual
        @info string(now()) * " [$n/$(num_pos)] Dual is $dual, which gives a bound of $(sqrt(dual)) on σₙ(Pᵣₛ)"
        bounds_dual_basis[n] = sqrt(dual)
        push!(outer_times, (n, (time_ns() - t_outer) / 1e9))
     catch err
        on_outer_error === :throw && rethrow(err)
        # :stop records where it failed and keep whatever has been measured
        frames = stacktrace(catch_backtrace())
        where_str = isempty(frames) ? "unknown" :
                    join(["$(f.func)@$(basename(String(f.file))):$(f.line)"
                          for f in Iterators.take(frames, 3)], "<-")
        outer_error = (n=n, exception=sprint(showerror, err), location=where_str)
        @warn string(now()) * " [bounds_bargaining::bounds_from_spectrum] outer loop failed at n=$n; stopping" exception = err location = where_str
        complete = false
        break
     end
    end

    analytical_bounds_old_form(κ) = ifelse(κ >= one(eltype(κ)), one(eltype(κ)), sqrt(4κ)/(1+κ))
    analytical_bounds_new_form(κ̃) = ifelse(2κ̃ >= one(eltype(κ̃)), one(eltype(κ̃)), sqrt(4κ̃*abs(1-κ̃)))
    κ = ζ^2 .* (Γrs .^ 2)
    κ̃ = ζ .* (max.(Γ, zero(eltype(Γ))))
    old_analytical_bounds = analytical_bounds_old_form.(κ)
    new_analytical_bounds = analytical_bounds_new_form.(κ̃)
    new_analytical_bounds[new_analytical_bounds .<= zero(eltype(new_analytical_bounds))] .= NaN
    old_analytical_bounds[old_analytical_bounds .<= zero(eltype(old_analytical_bounds))] .= NaN

    ks = 1:min(length(old_analytical_bounds), length(new_analytical_bounds), length(bounds_dual_basis))
    which_bounds = [argmin((old_analytical_bounds[k], new_analytical_bounds[k],
                            bounds_dual_basis[k])) for k in ks]
    # true_bounds = min.(old_analytical_bounds[ks], new_analytical_bounds[ks], bounds_dual_basis[ks])
    true_bounds = map(i -> begin # Save which bound is the minimum for each k to see where the flips happen
        if which_bounds[i] == 1
            return old_analytical_bounds[ks[i]]
        elseif which_bounds[i] == 2
            return new_analytical_bounds[ks[i]]
        else
            return bounds_dual_basis[ks[i]]
        end
    end, 1:length(ks))

    # plt = plot(eachindex(bounds_dual_basis), bounds_dual_basis, label="Dual (RSVD basis)")
    # plot!(plt, eachindex(old_analytical_bounds), old_analytical_bounds, label="Passivity [γᵣₛ]")
    # plot!(plt, eachindex(new_analytical_bounds), new_analytical_bounds, label="Passivity [γᵤᵣᵃ]")
    # plot!(plt, eachindex(true_bounds), true_bounds, label="Min", ls=:dash)
    # plot!(plt, xlabel="n", ylabel="Bound on σₙ(Pᵣₛ)", yscale=:log10, minorticks=true, minorgrid=true, legend=:right)
    # display(plt)
    # println("Press Enter to continue...")
    # readline()

    stage_times = (gram_schmidt=t_gram_schmidt, ss_basis=t_ss_basis,
                   c_projection=t_c_projection,
                   outer_total=sum(last.(outer_times); init=0.0))
    @info string(now()) * " [bounds_bargaining::bounds_from_spectrum] Stage times [s]:" stage_times

    return (num_pos=num_pos, complete=complete, outer_error=outer_error,
            bounds_dual_basis=bounds_dual_basis,
            old_analytical_bounds=old_analytical_bounds,
            new_analytical_bounds=new_analytical_bounds,
            true_bounds=true_bounds, which_bounds=which_bounds, ks=ks,
            basis_size=RSVD_BASIS_SIZE,
            stage_times=stage_times, outer_times=outer_times)
end

function _compute_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem, rsvd_params::RSVDParams)
    @info string(now()) * " [bounds_bargaining::_compute_bounds_sr] Computing bounds for SR system"

    inputs = load_bounds_inputs(compute_env, smr)
    Γ, Vur_asym, Γrs, sorted_idxs = inputs.Γ, inputs.Vur_asym, inputs.Γrs, inputs.sorted_idxs

    # Written up front (truncating any previous run's file) so that the ordering
    # is on disk even if the bounds loop below is cut short by a time limit.
    jld_out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld")
    jld_out = jldopen(jld_out_path, "w")
    jld_out["Γrs"] = Array(Γrs)
    jld_out["ordering_idxs"] = sorted_idxs
    close(jld_out)

    result = bounds_from_spectrum(compute_env, smr, Γ, Vur_asym, Γrs)
    result.complete || error("bounds_from_spectrum returned an incomplete result; refusing to save partial bounds")

    # Save data to disk
    jld_out = jldopen(jld_out_path, "a")
    if !haskey(jld_out, "χ")
        jld_out["χ"] = susceptibility(smr)
    end
    if !haskey(jld_out, "Γ")
        jld_out["Γ"] = Array(Γ)
    end
    if !haskey(jld_out, "Γrs")
        jld_out["Γrs"] = Array(Γrs)
    end
    if !haskey(jld_out, "bounds_dual_basis")
        jld_out["bounds_dual_basis"] = result.bounds_dual_basis
    end
    if !haskey(jld_out, "old_analytical_bounds")
        jld_out["old_analytical_bounds"] = result.old_analytical_bounds
    end
    if !haskey(jld_out, "new_analytical_bounds")
        jld_out["new_analytical_bounds"] = result.new_analytical_bounds
    end
    if !haskey(jld_out, "true_bounds")
        jld_out["true_bounds"] = result.true_bounds
    end
    if !haskey(jld_out, "which_bounds")
        jld_out["which_bounds"] = result.which_bounds
    end
    close(jld_out)
    return result
end

function compute_bounds()
    compute_env, smr, rsvd_params = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using GPU acceleration on device $(gpu_device(compute_env))"
        if !haskey(ENV, "CC_CLUSTER") # This breaks on compute canada
            CUDA.device!(gpu_device(compute_env))
        end
    else
        @info string(now()) * " [bounds_bargaining::compute_bounds] Using CPU computation"
    end

    if isnothing(mediator(smr))
        _compute_bounds_sr(compute_env, smr, rsvd_params)
    else
        _compute_bounds_smr(compute_env, smr, rsvd_params)
    end
end
