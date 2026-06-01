using LinearAlgebra
using Roots
using Base.Threads

function objective(α::Float64, Λ::AbstractVector{Float64}, b²::AbstractVector{Float64})
    # return α/4 * sum(@. b² * ((3α*Λ - 2) / (1 - α*Λ)^2))
    return α/4 * sum(@. b² * ((2 - α*Λ) / (1 - α*Λ)^2))
end

using Plots
function ξ(Λ_constraint::AbstractVector{Float64}, U_constraint::AbstractMatrix{ComplexF64}, Vur_asym::AbstractMatrix{ComplexF64})
    B² = abs2.(U_constraint' * Vur_asym)

    max_pole = maximum(1 ./ Λ_constraint)
    ξs = zeros(Float64, size(B², 2))
    for k in collect(axes(B², 2)) # TODO: @threads
        b² = @view B²[:, k]
        f(α) = objective(α, Λ_constraint, b²)
        xs = range(1e-3*max_pole, 5*max_pole; length=1000)
        plot(xs, asinh.(f.(xs)); label="Objective function for k=$k")
        readline()
        lo, hi = find_interval(max_pole, f) # TODO
        α = find_zero(f, (lo, hi), Roots.Brent())
        ξs[k] = α^2/4 * sum(@. b² / (1 - α*Λ)^2) # Note that Vur_asym has already been truncated to the receiver region
        break
    end

    return ξs
end

function compute_bounds()
    compute_env, smr, rsvd_params = parse_args()

    jld_path = joinpath(scratch_dir(compute_env), "$(file_prefix(smr)).jld")
    jld = jldopen(jld_path, "r")
    Λ_constraint = jld["constraint_asym/D"]
    U_constraint = jld["constraint_asym/V"]
    Vur_asym = jld["UR_asym/V"]
    close(jld)
    @show ξ(Λ_constraint, U_constraint, Vur_asym)
end

