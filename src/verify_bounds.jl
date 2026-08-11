using KrylovKit
using JLD2
using Printf
using Random

# Indices to verify, as fractions of `num_pos`. Index 1 is always included.
const VERIFY_INDEX_FRACTIONS = (0.35, 0.50, 0.85)

#=
Relative inflation applied to each basis-optimal multiplier before the
full-space evaluation. The basis feasible set is larger than the full-space
one, so the basis α can sit exactly on (or past) the full-space feasibility
boundary, where αC − Bₙ is singular and CG stalls. The dual is stationary at
its minimizer, so evaluating at (1+ε)α costs only O(ε²) in tightness while
making the certificate systems comfortably definite.
=#
const VERIFY_ALPHA_INFLATION = 0.05

#=
Cap on the number of full-space probe solves per index. The certified bound is
a max over probes k ∈ n:num_pos, so a truncated probe set is reported as such
Tthe probes with the largest basis duals are kept, which is where the max is
overwhelmingly likely to live, but it is no longer a strict certificate over
all probes.
=#
const VERIFY_MAX_PROBES = 512

const VERIFY_CG_RTOL = 1e-8
const VERIFY_CG_MAXITER = 1000
const VERIFY_LANCZOS_KRYLOVDIM = 150
const VERIFY_LANCZOS_MAXITER = 20
# Accept λ_min ≥ -VERIFY_FEAS_RTOL * λ_max as feasible
const VERIFY_FEAS_RTOL = 1e-8
# Number of times to double the multiplier floor if the basis α is infeasible.
const VERIFY_MAX_FEAS_INFLATIONS = 6

function verify_indices(num_pos::Int)
    ns = [clamp(round(Int, f * num_pos), 1, num_pos) for f in VERIFY_INDEX_FRACTIONS]
    return unique!(sort!([1; ns]))
end

function full_space_family(gs_pos::AbstractMatrix, Γ_pos::AbstractVector, ζ::Real, s_projector, G⁰ᵤᵤ_asym)
    num_pos = length(Γ_pos)
    apply_C(τ, v) = begin
        Πₛv = s_projector * v
        out = (1/ζ) .* Πₛv .+ ((1 - τ)/ζ) .* (v .- Πₛv) # ζ⁻¹(Πₛ + (1−τ)Πᵣ)
        out .+= τ .* (gs_pos * ((gs_pos' * v) .* Γ_pos)) # τ(−G⁰ᵤᵣ)ᵃ₊
        out .+= G⁰ᵤᵤ_asym * v # (G⁰ᵤᵤ)ᵃ
        return out
    end
    apply_B(n, v) = begin
        G = view(gs_pos, :, n:num_pos)
        return (4/ζ) .* (G * ((G' * v) .* view(Γ_pos, n:num_pos)))
    end
    apply_M(α, τ, n) = v -> α .* apply_C(τ, v) .- apply_B(n, v)
    return (C=apply_C, B=apply_B, M=apply_M)
end

function lanczos_extreme(f, x₀::AbstractVector, which::Symbol; tol::Real)
    vals, _, info = eigsolve(f, x₀, 1, which; ishermitian=true, krylovdim=VERIFY_LANCZOS_KRYLOVDIM, maxiter=VERIFY_LANCZOS_MAXITER, tol=tol)
    return (value=real(vals[1]), converged=info.converged >= 1, normres=first(info.normres))
end

function rsvd_eigenpair_residuals(compute_env::ComputeEnvironment, smr::SMRSystem, gs_pos::AbstractMatrix, Γ_pos_cpu::AbstractVector, ls::AbstractVector{Int})
    @info string(now()) * " [verify_bounds::rsvd_eigenpair_residuals] Loading exact Asym(−G⁰ᵣᵤ) pieces"
    G₀_rs = load_green_function(compute_env, smr, Receiver, Sender)
    G₀_rr = load_green_function(compute_env, smr, Receiver, Receiver)
    A_exact, _ = asym_ur(G₀_rs, G₀_rr, smr)

    rayleighs = zeros(Float64, length(ls))
    residuals = zeros(Float64, length(ls))
    for (i, l) in enumerate(ls)
        g = gs_pos[:, l]
        Ag = A_exact * g
        rayleighs[i] = real(dot(g, Ag))
        residuals[i] = norm(Ag .- Γ_pos_cpu[l] .* g) / abs(Γ_pos_cpu[l])
        @info string(now()) * " [verify_bounds::rsvd_eigenpair_residuals] l=$l: " *
            "Γₗ=$(Γ_pos_cpu[l]), Rayleigh=$(rayleighs[i]), rel. residual=$(residuals[i])"
    end
    return (ls=collect(ls), Γ=Float64.(Γ_pos_cpu[collect(ls)]), rayleighs=rayleighs, residuals=residuals)
end

"""
    certify_index(n, τ, records, family, ss, x₀)

Full-space dual certificate for index `n` at the production-optimal `τ`.
`records` are the basis-side per-probe solutions from `pencil_probe_duals`.

Feasibility logic: `C(τ) ⪰ 0` (certified separately, once, at the τ = 0, 1
endpoints, `C` is affine in τ so the endpoints cover the family) makes
`λ_min(αC − Bₙ)` non-decreasing in α, so one Lanczos check at the smallest
evaluated multiplier certifies every larger one. If the basis multiplier is
infeasible in the full space the floor is doubled until feasible and every
probe is clamped to at least the floor.
"""
function certify_index(n::Int, τ::Real, records, family, ss::AbstractMatrix,
                       x₀::AbstractVector)
    ks, basis_alphas, basis_duals = records.ks, records.alphas, records.duals

    order = sortperm(basis_duals; rev=true)
    truncated = length(order) > VERIFY_MAX_PROBES
    if truncated
        @warn string(now()) * " [verify_bounds::certify_index] n=$n has $(length(order)) probes; " *
            "evaluating only the $(VERIFY_MAX_PROBES) with the largest basis duals. " *
            "The certificate no longer covers every probe — treat it as evidence, not proof."
        order = order[1:VERIFY_MAX_PROBES]
    end

    alphas = (1 + VERIFY_ALPHA_INFLATION) .* basis_alphas

    α_floor = minimum(view(alphas, order))
    Mop(α) = family.M(α, τ, n)
    rough_scale = norm(Mop(α_floor)(x₀)) / norm(x₀)
    λ_max = lanczos_extreme(Mop(α_floor), x₀, :LR; tol=1e-6 * rough_scale)
    feas_tol = VERIFY_FEAS_RTOL * abs(λ_max.value)
    lanczos_tol = 1e-6 * abs(λ_max.value)

    @info string(now()) * " [verify_bounds::certify_index] n=$n, τ=$τ: certifying λ_min(αC − Bₙ) ≥ 0 at α=$α_floor"
    λ_min = lanczos_extreme(Mop(α_floor), x₀, :SR; tol=lanczos_tol)
    num_inflations = 0
    α_infeasible = NaN
    while λ_min.value < -feas_tol && num_inflations < VERIFY_MAX_FEAS_INFLATIONS
        num_inflations += 1
        α_infeasible = α_floor
        α_floor *= 2
        @warn string(now()) * " [verify_bounds::certify_index] n=$n: basis multiplier is " *
            "infeasible in the full space (λ_min = $(λ_min.value) < -$feas_tol) — the sketch " *
            "under-reported the feasibility threshold. Raising the multiplier floor to $α_floor"
        λ_min = lanczos_extreme(Mop(α_floor), x₀, :SR; tol=lanczos_tol)
    end
    feasible = λ_min.converged && λ_min.value >= -feas_tol
    if feasible && num_inflations > 0
        for _ in 1:3
            α_mid = (α_infeasible + α_floor) / 2
            λ_mid = lanczos_extreme(Mop(α_mid), x₀, :SR; tol=lanczos_tol)
            if λ_mid.converged && λ_mid.value >= -feas_tol
                α_floor, λ_min = α_mid, λ_mid
            else
                α_infeasible = α_mid
            end
        end
        @info string(now()) * " [verify_bounds::certify_index] n=$n: bisected the multiplier " *
            "floor down to $α_floor (λ_min ≈ $(λ_min.value))"
    end
    feasible || @warn string(now()) * " [verify_bounds::certify_index] n=$n: could not certify " *
        "dual feasibility (λ_min estimate $(λ_min.value), converged=$(λ_min.converged), " *
        "residual $(λ_min.normres)); the full-space values below are not certificates"

    # λ_min(M(α)) is non-decreasing in α (C ⪰ 0), so the value certified at
    # α_floor lower-bounds it for every evaluated probe; it converts each CG
    # residual r into a rigorous dual error bar |s'M⁺s − s'y| ≤ ‖s‖‖r‖/λ_min.
    λ_floor = feasible ? max(λ_min.value, 0.0) : 0.0

    used_alphas = fill(NaN, length(ks))
    full_duals = fill(NaN, length(ks))
    full_dual_errs = fill(NaN, length(ks))
    solves_converged = fill(false, length(ks))
    for (j, i) in enumerate(order)
        α = max(alphas[i], α_floor)
        s = ss[:, ks[i]]
        y, info = linsolve(Mop(α), s; ishermitian=true, isposdef=true,
                           tol=VERIFY_CG_RTOL * norm(s), maxiter=VERIFY_CG_MAXITER)
        used_alphas[i] = α
        solves_converged[i] = info.converged >= 1
        full_duals[i] = α^2/4 * real(dot(s, y))
        resid = first(info.normres)
        full_dual_errs[i] = λ_floor > 0 ? α^2/4 * norm(s) * resid / λ_floor : NaN
        if j % 25 == 0
            @info string(now()) * " [verify_bounds::certify_index] n=$n: $j/$(length(order)) probe solves done"
        end
    end

    # The certified value pads each probe by its error bar, so a truncated CG
    # solve widens the certificate instead of invalidating it. NaN error bars
    # (no positive λ_floor) leave the point estimate, flagged by feasible=false.
    padded = [isnan(full_dual_errs[i]) ? full_duals[i] : full_duals[i] + full_dual_errs[i]
              for i in eachindex(full_duals)]
    i_best = order[argmax(view(padded, order))]
    return (feasible=feasible, λ_min=λ_min.value, λ_min_converged=λ_min.converged,
            λ_min_normres=λ_min.normres, λ_max=λ_max.value, α_floor=α_floor,
            num_feasibility_inflations=num_inflations, probes_truncated=truncated,
            ks=ks, basis_alphas=basis_alphas, basis_duals=basis_duals,
            used_alphas=used_alphas, full_duals=full_duals, full_dual_errs=full_dual_errs,
            solves_converged=solves_converged,
            all_solves_converged=all(view(solves_converged, order)),
            full_bound=sqrt(max(padded[i_best], 0.0)),
            full_bound_point=sqrt(max(maximum(view(full_duals, order)), 0.0)),
            best_probe=ks[i_best])
end

"""
    production_bounds_checks(jld_path)

Nearly-free internal-consistency checks on the saved production bounds, over
all indices: the dual bounds must be non-increasing in `n` (`Bₙ ⪯ Bₙ₋₁` and
the probe set shrinks), and should not meaningfully exceed the semi-analytic
Eq. (40) curve, which is a further relaxation of the same program. Returns
`nothing` when the production file is absent.
"""
function production_bounds_checks(jld_path::AbstractString)
    ispath(jld_path) || return nothing
    jld = jldopen(jld_path, "r")
    if !haskey(jld, "bounds_dual_basis")
        close(jld)
        return nothing
    end
    dual = jld["bounds_dual_basis"]
    analytic = haskey(jld, "new_analytical_bounds") ? jld["new_analytical_bounds"] : nothing
    close(jld)

    computed = findall(>(0.0), dual) # partial runs leave zeros
    monotonicity_violations = [n for (prev, n) in zip(computed[1:end-1], computed[2:end])
                               if dual[n] > dual[prev] * (1 + 1e-6)]
    analytic_exceedances = isnothing(analytic) ? Int[] :
        [n for n in computed
         if n <= length(analytic) && isfinite(analytic[n]) && dual[n] > analytic[n] * (1 + 1e-3)]
    return (bounds=dual, computed=computed,
            monotonicity_violations=monotonicity_violations,
            analytic_exceedances=analytic_exceedances)
end

function _verify_summary(smr::SMRSystem, result, per_index, rsvd_check, λC0, λC1, production)
    io = IOBuffer()
    println(io, "\n================ verify_bounds summary: $(file_prefix(smr)) ================")

    println(io, "Constraint family PSD check (full space; C is affine in τ, so the endpoints cover it):")
    @printf(io, "    λ_min(C(0)) ≈ %.3e (converged: %s)   λ_min(C(1)) ≈ %.3e (converged: %s)\n",
            λC0.value, λC0.converged, λC1.value, λC1.converged)

    println(io, "RSVD eigenpair fidelity vs exact Asym(−G⁰ᵣᵤ):")
    for i in eachindex(rsvd_check.ls)
        @printf(io, "    l=%-5d Γₗ=%-13.6e Rayleigh=%-13.6e rel. residual=%.3e\n",
                rsvd_check.ls[i], rsvd_check.Γ[i], rsvd_check.rayleighs[i],
                rsvd_check.residuals[i])
    end

    println(io, "Per-index full-space certificates (bounds on σₙ(Pᵣₛ)):")
    for rec in per_index
        if hasproperty(rec, :error)
            @printf(io, "    n=%-5d FAILED: %s\n", rec.n, rec.error)
            continue
        end
        ratio = rec.full_bound / rec.basis_bound
        num_evaluated = count(!isnan, rec.full_duals)
        num_converged = count(rec.solves_converged)
        @printf(io, "    n=%-5d τ*=%-6.3f basis σₙ ≤ %-13.6e full-space σₙ ≤ %-13.6e (full/basis = %.4f)\n",
                rec.n, rec.τ, rec.basis_bound, rec.full_bound, ratio)
        @printf(io, "            production: %-13.6e point estimate: %-13.6e analytic old/new: %.3e / %.3e\n",
                rec.production_bound, rec.full_bound_point,
                get(result.old_analytical_bounds, rec.n, NaN),
                get(result.new_analytical_bounds, rec.n, NaN))
        @printf(io, "            feasible: %-5s (λ_min ≈ %.3e, λ_max ≈ %.3e, floor inflations: %d)   CG converged: %d/%d probes%s   best k: %d\n",
                rec.feasible, rec.λ_min, rec.λ_max, rec.num_feasibility_inflations,
                num_converged, num_evaluated,
                rec.probes_truncated ? " (TRUNCATED)" : "", rec.best_probe)
    end

    if production !== nothing
        println(io, "Production output checks (all $(length(production.computed)) computed indices):")
        if isempty(production.monotonicity_violations)
            println(io, "    monotonicity: OK (non-increasing in n)")
        else
            println(io, "    monotonicity: VIOLATED at n ∈ $(production.monotonicity_violations)")
        end
        if isempty(production.analytic_exceedances)
            println(io, "    vs semi-analytic Eq. (40): OK (never meaningfully above)")
        else
            println(io, "    vs semi-analytic Eq. (40): EXCEEDED at n ∈ $(production.analytic_exceedances)")
        end
    else
        println(io, "Production output file not found; skipped saved-bounds checks")
    end

    println(io, "Reading the ratios: full/basis slightly above 1 is expected (the full dual is")
    println(io, "evaluated at (1+ε)α, ε = $(VERIFY_ALPHA_INFLATION), and each probe is padded by its CG")
    println(io, "error bar ‖s‖‖r‖/λ_min); ratios well")
    println(io, "above 1 mean the sketch under-reported.")
    println(io, "Caveat: with floor inflations > 0 the basis multipliers were infeasible in the full")
    println(io, "space (itself a red flag), and the certificate is then evaluated far from each")
    println(io, "probe's optimum. The sketch's true optimism lies somewhere between 1 and the printed ratio.")
    print(String(take!(io)))
end

function _verify_bounds_sr(compute_env::ComputeEnvironment, smr::SMRSystem)
    Random.seed!(0x0B0DD5) # reproducible Lanczos starts

    inputs = load_bounds_inputs(compute_env, smr)
    Γ, Vur_asym, Γrs = inputs.Γ, inputs.Vur_asym, inputs.Γrs

    χ = susceptibility(smr)
    ζ = abs(χ)^2 / imag(χ)

    G₀_uu = load_green_function(compute_env, smr, [Sender, Receiver], [Sender, Receiver])

    num_pos = count(Γ .> zero(eltype(Γ)))
    ns = verify_indices(num_pos)
    @info string(now()) * " [verify_bounds::_verify_bounds_sr] Verifying indices $ns of num_pos = $num_pos"

    # Reproduce the production basis computation at the selected indices
    # This also returns the probes and projected constraint
    result = bounds_from_spectrum(compute_env, smr, Γ, Vur_asym, Γrs;
                                  G₀_uu=G₀_uu, outer_indices=ns)
    result.basis_size == num_pos || error(
        "verify_bounds assumes the production default basis_size = num_pos, got " *
        "$(result.basis_size) ≠ $num_pos")

    Γ_pos_cpu = Γ[1:num_pos]
    gs_pos = Vur_asym[:, 1:num_pos]
    Γ_pos = similar(gs_pos, real(eltype(gs_pos)), num_pos) # in gs_pos's array space
    copyto!(Γ_pos, Γ_pos_cpu)

    # RSVD spectral fidelity at (and around) the tested indices.
    rsvd_check = rsvd_eigenpair_residuals(compute_env, smr, gs_pos, Γ_pos_cpu,
                                          unique!(sort!(vcat(1, ns, num_pos))))
    run_gc() # release the G₀_rs / G₀_rr device memory before the heavy loop

    # Full-space
    s_projector = projected_operators(G₀_uu, smr, compute_env)
    G⁰ᵤᵤ_asym = asym(LinearMap(G₀_uu))
    family = full_space_family(gs_pos, Γ_pos, ζ, s_projector, G⁰ᵤᵤ_asym)

    x₀ = similar(gs_pos, size(gs_pos, 1))
    copyto!(x₀, randn(ComplexF64, size(gs_pos, 1)))

    scale_C = norm(family.C(1.0, x₀)) / norm(x₀)
    λC0 = lanczos_extreme(v -> family.C(0.0, v), x₀, :SR; tol=1e-6 * scale_C)
    λC1 = lanczos_extreme(v -> family.C(1.0, v), x₀, :SR; tol=1e-6 * scale_C)
    for (τ_end, λC) in ((0.0, λC0), (1.0, λC1))
        λC.value >= -VERIFY_FEAS_RTOL * scale_C ||
            @warn string(now()) * " [verify_bounds::_verify_bounds_sr] C(τ=$τ_end) has " *
                "λ_min ≈ $(λC.value) < 0 in the full space. The PSD constraint-family " *
                "claim fails there and the per-index feasibility logic is unsound"
    end

    # Per-index certificates
    per_index = []
    for n in ns
        rec = try
            τ = result.opt_taus[n]
            isfinite(τ) || error("production run recorded no optimal τ for n=$n")
            C_τ = isone(τ) ? result.C_basis : result.C_basis .- (1 - τ) .* result.D_basis
            pencil = psd_pencil_whitener(C_τ)
            B_diag = zeros(Float64, num_pos)
            B_diag[n:num_pos] .= (4/ζ) .* Γ_pos_cpu[n:num_pos]
            records = pencil_probe_duals(pencil, B_diag, result.ss_basis, n, num_pos; τ=τ)
            cert = certify_index(n, τ, records, family, result.ss, x₀)
            basis_bound = sqrt(maximum(records.duals))
            @info string(now()) * " [verify_bounds::_verify_bounds_sr] n=$n: basis $(basis_bound) " *
                "vs full-space $(cert.full_bound) (ratio $(cert.full_bound / basis_bound))"
            (n=n, τ=τ, basis_bound=basis_bound,
             production_bound=result.bounds_dual_basis[n], cert...)
        catch err
            @warn string(now()) * " [verify_bounds::_verify_bounds_sr] index n=$n failed" exception = (err, catch_backtrace())
            (n=n, error=sprint(showerror, err))
        end
        push!(per_index, rec)
    end

    # Saved-output consistency checks over all production indices
    production = production_bounds_checks(
        joinpath(project_dir(compute_env), "$(file_prefix(smr)).jld"))

    # Save everything needed to re-plot or re-audit without rerunning
    out_path = joinpath(project_dir(compute_env), "$(file_prefix(smr))_verify.jld")
    @info string(now()) * " [verify_bounds::_verify_bounds_sr] Saving verification data to $out_path"
    jldopen(out_path, "w") do jld
        jld["ns"] = ns
        jld["num_pos"] = num_pos
        jld["chi"] = χ
        jld["alpha_inflation"] = VERIFY_ALPHA_INFLATION
        jld["lambda_min_C0"] = λC0.value
        jld["lambda_min_C1"] = λC1.value
        jld["rsvd_residuals/ls"] = rsvd_check.ls
        jld["rsvd_residuals/Gamma"] = rsvd_check.Γ
        jld["rsvd_residuals/rayleighs"] = rsvd_check.rayleighs
        jld["rsvd_residuals/residuals"] = rsvd_check.residuals
        for rec in per_index
            p = "n_$(rec.n)/"
            if hasproperty(rec, :error)
                jld[p * "error"] = rec.error
                continue
            end
            jld[p * "tau"] = rec.τ
            jld[p * "basis_bound"] = rec.basis_bound
            jld[p * "production_bound"] = rec.production_bound
            jld[p * "full_bound"] = rec.full_bound
            jld[p * "feasible"] = rec.feasible
            jld[p * "lambda_min"] = rec.λ_min
            jld[p * "lambda_max"] = rec.λ_max
            jld[p * "alpha_floor"] = rec.α_floor
            jld[p * "num_feasibility_inflations"] = rec.num_feasibility_inflations
            jld[p * "probes_truncated"] = rec.probes_truncated
            jld[p * "ks"] = rec.ks
            jld[p * "basis_alphas"] = rec.basis_alphas
            jld[p * "basis_duals"] = rec.basis_duals
            jld[p * "used_alphas"] = rec.used_alphas
            jld[p * "full_duals"] = rec.full_duals
            jld[p * "full_dual_errs"] = rec.full_dual_errs
            jld[p * "full_bound_point"] = rec.full_bound_point
            jld[p * "solves_converged"] = rec.solves_converged
            jld[p * "best_probe"] = rec.best_probe
        end
        if production !== nothing
            jld["production/monotonicity_violations"] = production.monotonicity_violations
            jld["production/analytic_exceedances"] = production.analytic_exceedances
        end
    end

    _verify_summary(smr, result, per_index, rsvd_check, λC0, λC1, production)
    return per_index
end

function verify_bounds()
    compute_env, smr, _ = parse_args()

    if use_gpu(compute_env)
        @info string(now()) * " [verify_bounds::verify_bounds] Using GPU acceleration on device $(gpu_device(compute_env))"
        if !haskey(ENV, "CC_CLUSTER") # This breaks on compute canada
            CUDA.device!(gpu_device(compute_env))
        end
    else
        @info string(now()) * " [verify_bounds::verify_bounds] Using CPU computation"
    end

    isnothing(mediator(smr)) ||
        error("verify_bounds only handles sender/receiver systems (mirroring _compute_bounds_sr)")
    return _verify_bounds_sr(compute_env, smr)
end
