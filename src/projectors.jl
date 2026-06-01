module Projectors

export ProjectiveConstraint, sample_projective_constraint, schur_complement_op, AbstractQuadrature, SymQuadrature, AsymQuadrature, multiplier_boundary, Ablock_op, Bblock_op, Dblock_op, Ablock_diag, full_A_inv_op_factory, full_A_diag, full_D_op, full_sB_op, full_A_inv_op, enumerate_constraints

using GilaElectromagnetics
using Dates
using Distributions
using Random
using LinearMaps
using CUDA

abstract type AbstractQuadrature end
struct SymQuadrature <: AbstractQuadrature end
struct AsymQuadrature <: AbstractQuadrature end

struct ProjectiveConstraint
    idxs::Union{NTuple{3, AbstractVector{Int}}, NTuple{3, Vector{Int}}}
    quadrature::AbstractQuadrature
    level::Int
    which_splits::NTuple{3, Int}
    cells::NTuple{3, Int}
end

Base.:(==)(pc1::ProjectiveConstraint, pc2::ProjectiveConstraint) = pc1.idxs == pc2.idxs && pc1.quadrature == pc2.quadrature

quadrature(::SymQuadrature, z::Number) = real(z)
quadrature(::SymQuadrature, op) = (op + op') / 2
quadrature(::AsymQuadrature, z::Number) = imag(z)
quadrature(::AsymQuadrature, op) = (op - op') / (2im)

quadrature(pc::ProjectiveConstraint, z) = quadrature(pc.quadrature, z)
σ(::SymQuadrature) = 0.5
σ(::AsymQuadrature) = -0.5im
σdag(::SymQuadrature) = 0.5
σdag(::AsymQuadrature) = 0.5im
sign(::SymQuadrature) = 1
sign(::AsymQuadrature) = -1

function indicator_tensor(idxs::NTuple{3, AbstractVector{Int}}, vol::GlaVol, sample_vec::AbstractVector{T}, val::U=one(U)) where {T, U}
    out = similar(sample_vec, vol.cel..., 3)
    fill!(out, zero(eltype(sample_vec)))
    fill!(view(out, idxs..., :), val)
    return out
end

# function Ablock_op(pc::ProjectiveConstraint, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T) where T
#     forward!(out, v) = begin
#         out_tens = similar(out, vol.cel..., 3)
#         fill!(out_tens, zero(eltype(v)))
#         fill!(view(out_tens, intersect.(sender_mask, pc.idxs)..., :), quadrature(pc, inv(χ)))
#         out .= vec(out_tens) # out = [q(χ⁻¹) * I_region * I_sender] * v
#         return out
#     end
#     return LinearMap{ComplexF64}(forward!, forward!, size(vol.cel)...; ismutating=true)
# end

function Ablock_diag(pc::ProjectiveConstraint, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T) where T
    out = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(out, zero(eltype(χ)))
    fill!(view(out, intersect.(sender_mask, pc.idxs)..., :), quadrature(pc, inv(χ))) # out = q(χ⁻¹) * I_region * I_sender
    return vec(view(out, sender_mask..., :)) # only return the entries in the sender
end

function Ablock_op(pc::ProjectiveConstraint, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T) where T
    total_mask = ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3) # I_region * I_sender
    sender_and_region = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(sender_and_region, zero(eltype(χ)))
    fill!(view(sender_and_region, total_mask..., :), quadrature(pc, inv(χ))) # q(χ⁻¹) IₛIᵨ
    forward!(out, v) = begin
        out .= vec(sender_and_region .* reshape(v, size(sender_and_region))) # out = [q(χ⁻¹) * I_region * I_sender] * v
    end
    return LinearMap{ComplexF64}(forward!, forward!, prod(vol.cel) * 3, prod(vol.cel) * 3; ismutating=true)
end

function Bblock_op(pc::ProjectiveConstraint, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, universe_mask::NTuple{3, <:AbstractVector{Int}}, G₀::LinearMap, χ::T) where T
    total_mask = ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3) # I_region * I_sender
    sender_and_region = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(sender_and_region, zero(eltype(χ)))
    fill!(view(sender_and_region, total_mask..., :), one(eltype(χ))) # IₛIᵨ

    universe = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(universe, zero(eltype(χ)))
    fill!(view(universe, universe_mask..., :), one(eltype(χ))) # Iᵤ

    tensor_buffer = CUDA.zeros(eltype(χ), vol.cel..., 3)

    forward!(out, v) = begin
        out .= vec(sender_and_region .* (σ(pc.quadrature) .* reshape(G₀ * v, size(sender_and_region)) .- quadrature(pc, inv(χ)) .* universe .* reshape(v, size(sender_and_region))))
    end
    dagger!(out, v) = begin
        tensor_buffer .= sender_and_region .* reshape(v, size(tensor_buffer))
        out .= σdag(pc.quadrature) .* (G₀' * vec(tensor_buffer)) .- vec(quadrature(pc, inv(χ)) .* tensor_buffer)
    end
    return LinearMap{ComplexF64}(forward!, dagger!, size(G₀)...; ismutating=true)
end

function Dblock_op(pc::ProjectiveConstraint, vol::GlaVol, universe_mask::NTuple{3, <:AbstractVector{Int}}, G₀::LinearMap, χ::T) where T
    region = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(region, zero(eltype(χ)))
    fill!(view(region, pc.idxs..., :), one(eltype(χ))) # I_region

    universe = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(universe, zero(eltype(χ)))
    fill!(view(universe, universe_mask..., :), quadrature(pc, inv(χ))) # q(χ⁻¹) I_universe

    forward!(out, v) = begin
        out .= vec(region .* (universe .* reshape(v, size(region)) .- reshape(quadrature(pc, G₀) * v, size(region))))
    end
    return LinearMap{ComplexF64}(forward!, forward!, size(G₀)...; ismutating=true) # Dblock is self-adjoint, so we can use the same function for both forward and dagger
end

# function full_A_diagonal(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T, sample_vec::AbstractVector) where {T, RT}
#     out_tens = similar(sample_vec, vol.cel..., 3)
#     fill!(out_tens, zero(eltype(sample_vec)))
#     fill!(view(out_tens, sender_mask..., :), one(eltype(sample_vec))) # I_sender
#     for (pc, multiplier) in zip(pcs, multipliers)
#         out_tens .+= indicator_tensor(ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3), vol, sample_vec, multiplier * quadrature(pc, inv(χ))) # out = [I_sender + sum_i (multiplierᵢ * q_i(χ⁻¹) * I_regionᵢ)] * sample_vec
#     end
#     return vec(out_tens)
# end

function multiplier_boundary(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T, sample_vec::AbstractVector) where {T, RT}
    # in each curr inds
    # 1 + a1*q1 + a2*q2 + a3*q3 + a_curr * q_curr >= 0
    # => a_curr * q_curr >= -1 - a1*q1 - a2*q2 - a3*q3
    tens = similar(sample_vec, RT, vol.cel..., 3)
    fill!(tens, zero(RT))
    fill!(view(tens, sender_mask..., :), -one(RT)) # I_sender
    for (pc, multiplier) in zip(pcs[1:end-1], multipliers[1:end-1]) # Go through all previous multipliers and constraints
        tens .+= RT.(indicator_tensor(ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3), vol, sample_vec, multiplier * -quadrature(pc, inv(χ)))) # out = [I_sender + sum_i (multiplierᵢ * q_i(χ⁻¹) * I_regionᵢ)] * sample_vec
    end
    q_curr = RT(quadrature(pcs[end].quadrature, inv(χ)))
    if iszero(q_curr)
        throw(ArgumentError("Current quadrature value is zero, cannot determine multiplier boundary."))
    end
    tens ./= q_curr
    if q_curr > zero(RT)
        return (:greater, minimum(tens[pcs[end].idxs..., :]))
    end
    return (:less, maximum(tens[pcs[end].idxs..., :]))
end

function full_A_inv_op(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T) where {T, RT}
    diag_tensor = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(view(diag_tensor, sender_mask..., :), one(eltype(χ)))
    for (pc, multiplier) in zip(pcs, multipliers)
        mask = ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3)
        view(diag_tensor, mask..., :) .+= multiplier * quadrature(pc, inv(χ))
    end
    inv_diag = vec(diag_tensor)
    inv_diag .= ifelse.(inv_diag .!= 0, 1 ./ inv_diag, 0)
    forward!(out, v) = (out .= inv_diag .* v)
    return LinearMap{ComplexF64}(forward!, forward!, 3*prod(vol.cel), 3*prod(vol.cel); ismutating=true)
end

function full_A_diag(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T) where {T, RT}
    region_buffer = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(region_buffer, zero(eltype(χ)))
    fill!(view(region_buffer, sender_mask..., :), one(eltype(χ))) # I_sender
    sample_vec = CUDA.zeros(eltype(χ), 0)
    for (pc, multiplier) in zip(pcs, multipliers)
        region_buffer .+= indicator_tensor(ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3), vol, sample_vec, multiplier * quadrature(pc, inv(χ))) # out = [I_sender + sum_i (multiplierᵢ * q_i(χ⁻¹) * I_regionᵢ)] * sample_vec
    end
    return vec(view(region_buffer, sender_mask..., :)) # only return the entries in the sender
end

function full_A_inv_op_factory(pcs::AbstractVector{<:ProjectiveConstraint}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T) where {T}
    intersection_masks = [ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3) for pc in pcs] # I_region * I_sender for each constraint
    region_buffer = CUDA.zeros(eltype(χ), vol.cel..., 3)
    sample_vec = CUDA.zeros(eltype(χ), 0)

    function full_A_inv_op(multipliers::AbstractVector{<:RT}) where RT
        fill!(region_buffer, zero(eltype(χ)))
        fill!(view(region_buffer, sender_mask..., :), one(eltype(χ))) # I_sender
        for (pc, multiplier, intersection_mask) in zip(pcs, multipliers, intersection_masks)
            region_buffer .+= indicator_tensor(intersection_mask, vol, sample_vec, multiplier * quadrature(pc, inv(χ))) # out = [I_sender + sum_i (multiplierᵢ * q_i(χ⁻¹) * I_regionᵢ)] * v
        end
        can_invert_mask = region_buffer .!= zero(eltype(region_buffer)) # Only invert where the mask is nonzero to avoid division by zero
        region_buffer[can_invert_mask] .= 1 ./ region_buffer[can_invert_mask] # out = (sum_i [q_i(χ⁻¹) * I_regionᵢ * I_sender])⁻¹ * v
        return LinearMap{ComplexF64}(v -> vec(region_buffer) .* v, v -> vec(region_buffer) .* v, 3*prod(vol.cel), 3*prod(vol.cel); ismutating=false)
    end
    return full_A_inv_op
end

function full_A_inv_diagonal_factory(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, χ::T, sample_vec::AbstractVector{T}) where {T, RT}
    prev_A_tensor = similar(sample_vec, RT, vol.cel..., 3)
    fill!(prev_A_tensor, zero(RT))
    fill!(view(prev_A_tensor, sender_mask..., :), one(RT)) # I_sender
    for (pc, multiplier) in zip(pcs[1:end-1], multipliers[1:end-1])
        prev_A_tensor .+= indicator_tensor(ntuple(i -> intersect(sender_mask[i], pc.idxs[i]), 3), vol, sample_vec, multiplier * quadrature(pc, inv(χ))) # out = [I_sender + sum_i (multiplierᵢ * q_i(χ⁻¹) * I_regionᵢ)] * sample_vec
    end
    curr_rho_indicator_tensor = indicator_tensor(ntuple(i -> intersect(sender_mask[i], pcs[end].idxs[i]), 3), vol, sample_vec, quadrature(pcs[end], inv(χ))) # I_sender * I_region for current constraint
    curr_A_tensor = copy(prev_A_tensor)
    can_invert_mask = similar(sample_vec, Bool, size(prev_A_tensor))

    function full_A_inv_diagonal(μ::RT)
        curr_A_tensor .= prev_A_tensor .+ (curr_rho_indicator_tensor .* μ) # out = [I_sender + sum_i (multiplierᵢ * q_i(χ⁻¹) * I_regionᵢ) + μ * q_curr * I_sender * I_region]
        can_invert_mask .= curr_A_tensor .!= zero(RT)
        # curr_A_tensor[can_invert_mask] .= 1 ./ curr_A_tensor[can_invert_mask]
        curr_A_tensor .= ifelse.(can_invert_mask, inv.(curr_A_tensor), zero(RT))
        return vec(curr_A_tensor)
    end
    return full_A_inv_diagonal
end

function full_B_op(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, universe_mask::NTuple{3, <:AbstractVector{Int}}, G₀::LinearMap, χ::T) where {T, RT}
    return sum(multiplier * Bblock_op(pc, vol, sender_mask, universe_mask, G₀, χ) for (pc, multiplier) in zip(pcs, multipliers))
end

function full_sB_op(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, universe_mask::NTuple{3, <:AbstractVector{Int}}, G₀_uu_disjoint::LinearMap, χ::T) where {T, RT}
    region_tensor_buffer = CUDA.zeros(eltype(χ), vol.cel..., 3)

    universe_buffer = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(view(universe_buffer, universe_mask..., :), one(eltype(χ)))

    sender_region = CUDA.zeros(eltype(χ), vol.cel..., 3)
    fill!(view(sender_region, sender_mask..., :), one(eltype(χ))) # I_sender

    Gv = CUDA.zeros(eltype(χ), size(G₀_uu_disjoint, 1)) # Buffer to store G₀v since it's used in every constraint
    forward!(out, v) = begin
        fill!(out, zero(eltype(out)))
        Gv .= G₀_uu_disjoint * v # precompute G₀v once since it's used in every constraint
        for (μ, pc) in zip(multipliers, pcs)
            fill!(region_tensor_buffer, zero(typeof(χ)))
            fill!(view(region_tensor_buffer, pc.idxs..., :), one(typeof(χ))) # I_region for current constraint

            # χ part
            out .-= vec(region_tensor_buffer .* universe_buffer) .* (μ * quadrature(pc, inv(χ))) .* v

            # G₀ part
            out .+= (vec(region_tensor_buffer) .* (μ * σ(pc.quadrature))) .* Gv
        end
        out .*= vec(sender_region)
        return out
    end

    v_tensor = CUDA.zeros(eltype(χ), vol.cel..., 3)
    act_tensor = CUDA.zeros(eltype(χ), size(G₀_uu_disjoint, 2)) # Accumulate sign(qⱼ)*σ(j)*Iᵨ₍ⱼ₎ * v here to be acted on by G later
    dagger!(out, v) = begin
        fill!(act_tensor, zero(eltype(χ)))
        fill!(out, zero(eltype(out)))
        v_tensor .= sender_region .* reshape(v, size(sender_region)) # I_sender * v
        for (μ, pc) in zip(multipliers, pcs)
            fill!(region_tensor_buffer, zero(typeof(χ)))
            fill!(view(region_tensor_buffer, pc.idxs..., :), one(typeof(χ))) # I_region for current constraint

            # χ part
            out .-= vec(region_tensor_buffer .* universe_buffer .* (μ * quadrature(pc, inv(χ))) .* v_tensor)

            # G₀ part; note that we still have to act with G₀' on act_tensor and add that to the output
            act_tensor .+= vec((μ * σdag(pc.quadrature)) .* region_tensor_buffer .* v_tensor)
        end
        out .+= G₀_uu_disjoint' * vec(act_tensor)
        return out
    end

    return LinearMap{ComplexF64}(forward!, dagger!, size(G₀_uu_disjoint)...; ismutating=true)
end

# function full_D_op(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, universe_mask::NTuple{3, <:AbstractVector{Int}}, G₀::LinearMap, χ::T) where {T, RT}
#     return sum(multiplier * Dblock_op(pc, vol, universe_mask, G₀, χ) for (pc, multiplier) in zip(pcs, multipliers))
# end

function full_D_op(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, universe_mask::NTuple{3, <:AbstractVector{Int}}, G₀_uu_disjoint::LinearMap, χ::T) where {T, RT}
    universe_buffer = CUDA.zeros(typeof(χ), vol.cel..., 3)
    fill!(universe_buffer, zero(typeof(χ)))
    fill!(view(universe_buffer, universe_mask..., :), one(typeof(χ))) # I_universe
    region_buffer = CUDA.zeros(typeof(χ), vol.cel..., 3)
    act_vec = CUDA.zeros(typeof(χ), size(G₀_uu_disjoint, 2)) # Accumulate sign(qⱼ)*σ(j)*Iᵨ₍ⱼ₎ * v here to be acted on by G later
    Gv = CUDA.zeros(typeof(χ), size(G₀_uu_disjoint, 1)) # Buffer to store G₀v since it's used in every constraint
    function forward!(out, v)
        fill!(act_vec, zero(typeof(χ)))
        fill!(out, zero(eltype(out)))
        Gv .= G₀_uu_disjoint * v # precompute G₀v once since it's used in every constraint
        for (μ, pc) in zip(multipliers, pcs)
            fill!(region_buffer, zero(typeof(χ)))
            fill!(view(region_buffer, pc.idxs..., :), one(typeof(χ))) # I_region for current constraint

            # χ part
            out .+= vec(region_buffer .* universe_buffer) .* (μ * quadrature(pc, inv(χ))) .* v

            # G₀ part; note that we still have to act with G₀' on act_vec and add that to the output
            out .-= (μ * σ(pc.quadrature)) .* vec(region_buffer) .* Gv
            act_vec .-= (μ * sign(pc.quadrature) * σ(pc.quadrature)) .* vec(region_buffer) .* v
        end
        out .+= G₀_uu_disjoint' * act_vec
        return out
    end
    return LinearMap{ComplexF64}(forward!, forward!, size(G₀_uu_disjoint)...; ismutating=true) # D is self-adjoint
end

function schur_complement_op(pcs::AbstractVector{<:ProjectiveConstraint}, multipliers::AbstractVector{<:RT}, vol::GlaVol, sender_mask::NTuple{3, <:AbstractVector{Int}}, universe_mask::NTuple{3, <:AbstractVector{Int}}, G₀::LinearMap, χ::T) where {T, RT}
    A_inv = full_A_inv_op(pcs, multipliers, vol, sender_mask, χ)
    B = full_B_op(pcs, multipliers, vol, sender_mask, universe_mask, G₀, χ)
    D = full_D_op(pcs, multipliers, vol, universe_mask, G₀, χ)
    return D - B' * A_inv * B
end

function subdivide(num_cells::Int, level::Int)
    @assert level >= 0 "Subdivision level must be non-negative."
    num_divisions = 1 << level
    if mod(num_cells, num_divisions) != 0
        @error string(now()) * " [projectors::subdivide] Subdivision level $(level) does not evenly divide the number of cells $(num_cells). No subdivision will be performed."
        return UnitRange{Int}[]
    end

    step_size = num_cells ÷ num_divisions
    subdivisions = [((i-1)*step_size + 1 : i*step_size) for i in 1:num_divisions]
    return subdivisions
end

function subdivide(vol::GlaVol, levels::NTuple{3, Int})
    cells = vol.cel
    subdivisions = subdivide.(cells, levels)
    return subdivisions
end

function max_level(vol::GlaVol, max_subdivisions::Int)
    cells = vol.cel
    max_levels = trailing_zeros.(cells) # The number of times we can divide by 2 before we get an odd number (for each dimension)
    subdivisions_done = 0
    max_level_reached = 0
    for maybe_level in 0:maximum(max_levels)
        # Check if the current level of subdivision is possible for all dimensions. If not, use the maximum possible level for that dimension.
        levels = ntuple(i -> maybe_level <= max_levels[i] ? maybe_level : max_levels[i], length(cells))
        curr_num_subdivisions = prod(1 .<< levels)
        subdivisions_done += curr_num_subdivisions
        if subdivisions_done > max_subdivisions
            break
        end
        max_level_reached = maybe_level
    end
    return max_level_reached
end

function level_weights(vol::GlaVol, max_subdivisions::Int)
    cells = vol.cel
    maximum_level = max_level(vol, max_subdivisions)
    weights = Vector{Float64}(undef, maximum_level + 1)
    for maybe_level in 0:maximum_level
        levels = ntuple(i -> maybe_level <= trailing_zeros(cells[i]) ? maybe_level : trailing_zeros(cells[i]), length(cells))
        curr_num_subdivisions = prod(1 .<< levels)
        weights[maybe_level + 1] = inv(curr_num_subdivisions)
    end
    return weights ./ sum(weights) # Normalize weights to sum to 1
end

function sample_level(vol::GlaVol, max_subdivisions::Int; rng=Random.default_rng())
    weights = level_weights(vol, max_subdivisions)
    dist = Categorical(weights)
    max_level = rand(rng, dist) - 1 # Subtract 1 to convert from 1-based indexing to 0-based levels
    return max_level
end

function sample_subdivision(vol::GlaVol, level::Int; rng=Random.default_rng())
    cells = vol.cel
    subdivisions = subdivide(vol, ntuple(i -> level <= trailing_zeros(cells[i]) ? level : trailing_zeros(cells[i]), length(cells)))
    which_splits = rand.(Ref(rng), axes.(subdivisions, 1))
    return which_splits, getindex.(subdivisions, which_splits)
end

sample_quadrature(rng=Random.default_rng()) = rand(rng, [SymQuadrature(), AsymQuadrature()])

function _sample_projective_constraint(vol::GlaVol, max_subdivisions::Int; rng=Random.default_rng())
    level = sample_level(vol, max_subdivisions; rng)
    which_splits, subdivision = sample_subdivision(vol, level; rng)
    quadrature = sample_quadrature(rng)
    return ProjectiveConstraint(subdivision, quadrature, level, which_splits, vol.cel)
end

function sample_projective_constraint(vol::GlaVol, max_subdivisions::Int; avoid::Union{Nothing, ProjectiveConstraint}=nothing, rng=Random.default_rng())
    @error "we need to check if all the indices are in the gap, and sample a new PC is that is the case"
    pc = _sample_projective_constraint(vol, max_subdivisions; rng)
    while avoid !== nothing && pc == avoid
        pc = sample_projective_constraint(vol, max_subdivisions; rng)
    end
    return pc
end

function sample_projective_constraint(vol::GlaVol, max_subdivisions::Int; avoid::AbstractVector{ProjectiveConstraint}, rng=Random.default_rng())
    @error "we need to check if all the indices are in the gap, and sample a new PC is that is the case"
    pc = _sample_projective_constraint(vol, max_subdivisions; rng)
    while any(pc == avoid_pc for avoid_pc in avoid)
        pc = _sample_projective_constraint(vol, max_subdivisions; rng)
    end
    return pc
end

function enumerate_constraints(vol::GlaVol, max_constraints::Int)
    cells = vol.cel
    max_levels = trailing_zeros.(cells)
    constraints = ProjectiveConstraint[]
    sizehint!(constraints, max_constraints)

    for level in 0:maximum(max_levels)
        levels_per_dim = ntuple(i -> min(level, max_levels[i]), 3)
        splits_per_dim = 1 .<< levels_per_dim
        subdivisions = subdivide(vol, levels_per_dim)

        for quad in (SymQuadrature(), AsymQuadrature())
            for ci in CartesianIndices(Tuple(splits_per_dim))
                which_splits = Tuple(ci)
                idxs = ntuple(i -> subdivisions[i][which_splits[i]], 3)
                push!(constraints, ProjectiveConstraint(idxs, quad, level, which_splits, cells))
                length(constraints) >= max_constraints && return constraints
            end
        end
    end

    return constraints
end

function ProjectiveConstraint(vol::GlaVol, idxs::NTuple{3, AbstractVector{Int}}, quadrature::AbstractQuadrature)
    cells = vol.cel
    # Infer per-dimension subdivision level and which-split from the indices
    level_per_dim = ntuple(3) do i
        n_divs = cells[i] ÷ length(idxs[i])
        @assert n_divs * length(idxs[i]) == cells[i] "Indices in dim $i don't evenly tile the volume."
        @assert ispow2(n_divs) "Number of divisions in dim $i ($n_divs) is not a power of 2."
        trailing_zeros(n_divs)
    end
    level = maximum(level_per_dim)
    # Verify consistency: clipping `level` by trailing_zeros must recover per-dim levels
    for i in 1:3
        @assert min(level, trailing_zeros(cells[i])) == level_per_dim[i] "Indices don't correspond to a single valid subdivision level."
    end
    which_splits = ntuple(i -> (first(idxs[i]) - 1) ÷ length(idxs[i]) + 1, 3)
    return ProjectiveConstraint(idxs, quadrature, level, which_splits, cells)
end

function ordinal(n::Int)
    if n % 100 in 11:13
        suffix = "th"
    else
        suffix = if n % 10 == 1
            "st"
        elseif n % 10 == 2
            "nd"
        elseif n % 10 == 3
            "rd"
        else
            "th"
        end
    end
    return string(n, suffix)
end

function level2str(level::Int)
    if level == 0
        return "Global"
    end
    return string(ordinal(level), " splitting level")
end

function quadrature2str(quadrature::AbstractQuadrature)
    if quadrature isa SymQuadrature
        return "sym"
    elseif quadrature isa AsymQuadrature
        return "asym"
    else
        return "Unknown Quadrature"
    end
end

function Base.show(io::IO, pc::ProjectiveConstraint)
    print(io, "$(level2str(pc.level)) $(quadrature2str(pc.quadrature)) constraint")
    if pc.level > 0
        splits_per_level = 1 .<< ntuple(i -> pc.level <= trailing_zeros(pc.cells[i]) ? pc.level : trailing_zeros(pc.cells[i]), length(pc.cells))
        split_str = join(["$(which)/$(split_per_level)" for (which, split_per_level) in zip(pc.which_splits, splits_per_level)], ", ")
        print(io, " @ split ($split_str)")
    end
end

end # module Projectors
