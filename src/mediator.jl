using LinearAlgebra

# ---------------------------------------------------------------------------
# The mediator design bound on σₙ(Gᵣₛ), Eqs. (31)–(33) of the paper, with the RSVD
# tails inside the dual.
#
# With s̃ = sender ∪ mediator, G⁰ᵣₛ̃ ≈ V Γ [Uₛ; U_d]ᴴ (residual ε_r) and G⁰ₘₛ ≈ V_d Γ_ds Yₛᴴ
# (residual ε_d), the objective tail gives A ⪯ A_k + ε_r²𝟙 (the residual's range is
# orthogonal to the kept one) and the power constraint, relaxed to ≤ (free, μₚ ≥ 0),
# gains ε_d‖t‖ ≤ ε_d(a/2 + ‖t‖²/2a). Both act as a loss reduction: with the solve done at
# B(c) = c + Asym(G⁰ₘₘ), c = (1 − f)ζ⁻¹, the share fζ⁻¹ of the loss absorbs them (half
# each), so the bound is valid for μₚ ≥ 2ζε_r²/f, a = ζε_d/f, and
#
#     σₙ² ≤ μₙ(μₚ) = σ₁(X [E + μₚ⁻¹ D K Dᴴ] Xᴴ) + ε_r² + μₚ ζ ε_d²/(2f),
#     X = [UₛΓ, Yₛ], D = diag(1, μₚ/2i), E = diag(1, 0), K = M + M₍·ᵤ₎(μₚ − Mᵤᵤ)⁻¹M₍ᵤ·₎,
#     M = [U_dΓ, V_dΓ_ds]ᴴ B(c)⁻¹ [U_dΓ, V_dΓ_ds],   feasible for μₚ > λ_max(Mᵤᵤ).
#
# Index n keeps columns n:k_r of the u block, so every n reads principal submatrices
# of the n = 1 matrices `M` and `gram` = XᴴX.
# ---------------------------------------------------------------------------

"""
    mediator_front(Γ, Uₛ, U_d, Γ_ds, V_d, Yₛ, pbp) -> NamedTuple

The n = 1 matrices of the dual: `M` and `gram`, with `k_r = length(Γ)`. `pbp(P)` returns
PᴴB(c)⁻¹P with c = (1 − f)ζ⁻¹ for the loss share `f` the dual is then called with.
"""
mediator_front(Γ, Uₛ, U_d, Γ_ds, V_d, Yₛ, pbp) =
    mediator_front(Γ, Uₛ, Yₛ, hcat(U_d .* transpose(Γ), V_d .* transpose(Γ_ds)), pbp)

# P = [U_dΓ, V_dΓ_ds] already formed, so the caller can build it without copies.
function mediator_front(Γ, Uₛ, Yₛ, P::Matrix, pbp)
    X = hcat(Uₛ .* transpose(Γ), Yₛ)
    gram = Array(Hermitian(X' * X))
    X = nothing
    M = pbp(P)
    return (M=Array(Hermitian((M + M') / 2)), gram=gram, k_r=length(Γ))
end

"""
    mediator_curve(front, n, ζ, ε_r, ε_d, f; ball=nothing) -> (μₙ, μ_lo)

The function μₚ ↦ μₙ(μₚ) of the header at index `n`, and the bottom of its domain:
λ_max(Mᵤᵤ), or the floor 2ζε_r²/f below which B(c) no longer absorbs the tails. `front`
must have been built at c = (1 − f)ζ⁻¹; with ε_r = ε_d = 0, f = 0 is exact.

With `ball` = 1 + ζ²σ₁²(G⁰ₘₛ) and f = 0, the tails are instead the paper's a-posteriori
ball (SM §VII.A): μₙ gains (ε_r² + μₚε_d)·ball, from ‖t‖ ≤ ζσ₁(G⁰ₘₛ) by power conservation.
"""
function mediator_curve(front, n::Int, ζ::Real, ε_r::Real, ε_d::Real, f::Real; dev=identity, ball=nothing)
    0 <= f < 1 || throw(ArgumentError("the loss share f must lie in [0, 1), got $f"))
    isnothing(ball) || f == 0 || throw(ArgumentError("the paper's ball solves at the full loss, f = 0"))
    f > 0 || !isnothing(ball) || ε_r == ε_d == 0 || throw(ArgumentError("nonzero residuals need a loss share f > 0"))
    k_r = front.k_r
    1 <= n <= k_r || throw(ArgumentError("n = $n outside 1:$k_r"))
    idx = vcat(n:k_r, k_r+1:size(front.M, 1))
    nu = k_r - n + 1
    M = dev(front.M[idx, idx])
    Eu = eigen(Hermitian(M[1:nu, 1:nu]))
    MW = M[:, 1:nu] * Eu.vectors
    Eg = eigen(Hermitian(dev(front.gram[idx, idx])))
    Gh = (Eg.vectors .* transpose(sqrt.(max.(Eg.values, 0)))) * Eg.vectors'
    GE = Gh[:, 1:nu] * Gh[:, 1:nu]' # Gh E Gh, the μₚ-independent part
    function μₙ(μ_p)
        K = M + (MW .* transpose(1 ./ (μ_p .- Eu.values))) * MW'
        d = dev(vcat(ones(ComplexF64, nu), fill(μ_p / 2im, length(idx) - nu)))
        H = GE + Gh * ((d .* K .* d') ./ μ_p) * Gh
        isnothing(ball) || return _eigmax(H) + (ε_r^2 + μ_p * ε_d) * ball
        return _eigmax(H) + ε_r^2 + (ε_d == 0 ? 0.0 : μ_p * ζ * ε_d^2 / 2f)
    end
    return μₙ, max(maximum(Eu.values), ε_r == 0 || !isnothing(ball) ? 0.0 : 2ζ * ε_r^2 / f, eps())
end

"""
    mediator_primal(front, n, μ_p) -> (x_coef, c, idx)

The maximizer z = [x; t] behind μₙ(μₚ): x = X·`x_coef` with X = [UₛΓ, Yₛ][:, idx], and
t = B(c)⁻¹ P[:, idx]·`c` with P = [U_dΓ, V_dΓ_ds]. x is the top eigenvector of the Schur
complement and t = (μₚB_dd − A_dd)⁻¹(A_ds − μₚB_ds)x, by Woodbury in the same basis as M.
"""
function mediator_primal(front, n::Int, μ_p::Real)
    k_r = front.k_r
    idx = vcat(n:k_r, k_r+1:size(front.M, 1))
    nu = k_r - n + 1
    M = front.M[idx, idx]
    Eg = eigen(Hermitian(front.gram[idx, idx]))
    s = sqrt.(max.(Eg.values, 0))
    Gh = (Eg.vectors .* transpose(s)) * Eg.vectors'
    Ghp = (Eg.vectors .* transpose([v > 1e-12 * maximum(s) ? 1 / v : 0.0 for v in s])) * Eg.vectors'
    R = (μ_p * I - M[1:nu, 1:nu]) \ M[1:nu, :]
    K = M + M[:, 1:nu] * R
    d = vcat(ones(ComplexF64, nu), fill(μ_p / 2im, length(idx) - nu))
    H = Gh[:, 1:nu] * Gh[:, 1:nu]' + Gh * ((d .* K .* d') ./ μ_p) * Gh
    v = eigen(Hermitian((H + H') / 2)).vectors[:, end]
    w = conj.(d) .* (Gh * v) # Dᴴ Xᴴ x
    c = (w + vcat((μ_p * I - M[1:nu, 1:nu]) \ (M[1:nu, :] * w), zeros(length(idx) - nu))) ./ μ_p
    return (x_coef=Ghp * v, c=c, idx=idx)
end

_eigmax(H::Matrix) = eigmax(Hermitian(H))
_eigmax(H::CuMatrix) = maximum(Array(CUDA.CUSOLVER.heevd!('N', 'U', (H + H') ./ 2)))

"""
    mediator_dual(front, n, ζ, ε_r, ε_d, f; log_tol=1e-4) -> (μ, μ_p)

`min over μₚ` of μₙ(μₚ), so `sqrt(μ)` bounds σₙ(Gᵣₛ). μₙ is convex in μₚ on its domain
and grows like μₚ at the top, so the bracket doubles until μₙ rises and golden section
finishes on log μₚ. Every evaluation is a valid bound; the smallest is kept.
"""
function mediator_dual(front, n::Int, ζ::Real, ε_r::Real, ε_d::Real, f::Real; log_tol::Real=1e-4, dev=identity, ball=nothing)
    μₙ, lo = mediator_curve(front, n, ζ, ε_r, ε_d, f; dev=dev, ball=ball)
    best = (μ=Inf, μ_p=NaN)
    g(t) = (μ_p = exp(t); μ = μₙ(μ_p); μ < best.μ && (best = (μ=μ, μ_p=μ_p)); μ)
    hi = 2lo
    while g(log(2hi)) < g(log(hi))
        (hi *= 2) < 1e100 || error("mediator_dual: μₙ still falls at μₚ = $hi, so the problem is not bounded at n = $n")
    end
    a, b = log(lo) + 1e-12, log(2hi)
    golden_min(g, a, b; iters=golden_iters(a, b, log_tol))
    return best
end

# ---------------------------------------------------------------------------
# The bounds stage for a system with a mediator.
# ---------------------------------------------------------------------------

# The share f of the material loss set aside for the RSVD tails. Larger f tolerates
# larger residuals; the bound pays about f on its own (see test/mediator_dense.jl).
const DEFAULT_LOSS_SHARE = 0.1

"""
    pcg_columns(applyB, precond, P; tol=1e-10, maxiter=5000) -> (Z, relres, iters)

Preconditioned CG on every column of `P` at once for B Z = P, B and the preconditioner
Hermitian positive definite: one `applyB` per iteration for the whole block. `relres` is
the true residual ‖P − BZ‖/‖P‖ per column, recomputed at the end.
"""
function pcg_columns(applyB, precond, P::AbstractMatrix; tol::Real=1e-10, maxiter::Int=5000)
    colnorm(X) = sqrt.(real.(sum(abs2, X; dims=1)))
    cdot(X, Y) = real.(sum(conj.(X) .* Y; dims=1))
    pn = colnorm(P)
    Z, R = zero(P), copy(P)
    Y = precond(R)
    D, ry = copy(Y), cdot(R, Y)
    iters = 0
    while maximum(colnorm(R) ./ pn) > tol && iters < maxiter
        iters += 1
        BD = applyB(D)
        α = ry ./ cdot(D, BD)
        Z .+= α .* D
        R .-= α .* BD
        Y = precond(R)
        ry_new = cdot(R, Y)
        D .= Y .+ (ry_new ./ ry) .* D
        ry = ry_new
    end
    return Z, vec(Array(colnorm(P .- applyB(Z)) ./ pn)), iters
end

"""
    mediator_M(compute_env, smr, P, c, k_mm) -> (M, relres)

M = PᴴB(c)⁻¹P, B(c) = c + Asym(G⁰ₘₘ), solved by [`pcg_columns`](@ref) in column batches
sized to the device, preconditioned by the top `k_mm` eigenpairs (W, λ) of Asym(G⁰ₘₘ):
x/c + W((1/(c + λ) − 1/c) ⊙ Wᴴx), exact on span(W). `k_mm` is clamped to leave room for
the sketch's oversamples.
"""
function mediator_M(compute_env::ComputeEnvironment, smr::SMRSystem, P::Matrix{ComplexF64}, c::Real, k_mm::Int)
    on_gpu = use_gpu(compute_env)
    dev(X) = on_gpu ? CuArray(X) : X
    N_m, k = size(P)
    M, relres = zeros(ComplexF64, k, k), fill(NaN, k)
    Amm = asym_green_map(compute_env, smr, Mediator)
    k_mm = max(0, min(k_mm, N_m - UU_OVERSAMPLES))
    W, λ = if k_mm > 0
        uu = uu_eigenbasis(compute_env, Amm, k_mm)
        uu.vectors, dev(uu.values)
    else
        dev(zeros(ComplexF64, N_m, 0)), dev(zeros(0))
    end
    precond(R) = R ./ c .+ W * ((1 ./ (c .+ λ) .- 1 / c) .* (W' * R))
    applyB(X) = c .* X .+ opmat(Amm, X)
    # Six N_m-column blocks live during the solve (P_J, Z, R, Y, D, BD), plus about four
    # broadcast temporaries (cdot, colnorm, the preconditioner) not yet collected: budget twelve.
    budget = on_gpu ? device_budget_bytes() - gila_workspace_bytes(N_m) - 16 * N_m * (size(W, 2) + STORE_COLUMNS) : 2^36
    width = clamp(floor(Int, budget / (12 * 16 * N_m)), 1, k)
    @info string(now()) * " [mediator::mediator_M] B⁻¹ on $(k) columns of N_m = $(N_m), batches of $(width), k_mm = $(size(W, 2)), c = $(c)"
    for J0 in 1:width:k
        J = J0:min(J0 + width - 1, k)
        t = time()
        Z, rr, iters = pcg_columns(applyB, precond, dev(P[:, J]))
        for I in Iterators.partition(1:k, STORE_COLUMNS)
            M[I, J] = Array(dev(P[:, I])' * Z)
        end
        relres[J] = rr
        @info string(now()) * " [mediator::mediator_M] Columns $(J): $(iters) iterations, max residual $(maximum(rr)), $(round(time() - t; digits=1)) s"
    end
    return M, relres
end

"""
    compute_mediator_bounds(compute_env, smr; k_mm, channels) -> NamedTuple

The mediator design bound on σₙ(Gᵣₛ) for n = 1:`channels`, from the stage-2 RSVDs, written
to `bounds_path`. ε_r, ε_d are the RSVD residual bounds; the solve runs at
c = (1 − `loss_share`)ζ⁻¹. `tails = :paper` swaps that for the paper's ball at the full
loss; `tails = :none` drops the tails at the full loss, which is not a bound but the
baseline both add to. Anything but the default goes beside it with a suffix
(`_paper`, `_none`, `_f<loss_share>`).
"""
function compute_mediator_bounds(compute_env::ComputeEnvironment, smr::SMRSystem; k_mm::Int=DEFAULT_K_UU, channels::Int=100,
                                 loss_share::Real=DEFAULT_LOSS_SHARE, tails::Symbol=:dual)
    tails in (:dual, :paper, :none) || throw(ArgumentError("tails must be :dual, :paper or :none, got $tails"))
    tails == :dual || (loss_share = 0.0)
    suffix = tails != :dual ? "_$(tails)" : loss_share != DEFAULT_LOSS_SHARE ? "_f$(loss_share)" : ""
    out_path = replace(bounds_path(compute_env, smr), r"\.jld2?$" => SubstitutionString(suffix * "\\0"))
    if isfile(out_path) && jldopen(f -> haskey(f, "true_bounds"), out_path, "r")
        @info string(now()) * " [mediator::compute_mediator_bounds] $(out_path) is already complete: skipping (delete it to recompute)"
        return nothing
    end
    χ = susceptibility(smr)
    imag(χ) > 0 || throw(ArgumentError("the mediator bound needs a lossy χ (Im χ > 0), got $χ"))
    ζ = abs(χ)^2 / imag(χ)
    jld_path = rsvd_output_path(compute_env, smr)
    dir = dirname(jld_path)
    inp = jldopen(jld_path, "r") do f
        for key in ("RU/eps", "MS/eps", "RS/D")
            haskey(f, key) || error("$(jld_path) has no $(key); run generate_rsvd first")
        end
        (Γ=f["RU/D"], ε_r=f["RU/eps"], RU_V=joinpath(dir, f["RU/V_file"]),
         Γ_ds=f["MS/D"], ε_d=f["MS/eps"], MS_U=joinpath(dir, f["MS/U_file"]), MS_V=joinpath(dir, f["MS/V_file"]),
         Γrs=f["RS/D"])
    end
    N_s = dof_length(sender_mesh(smr))
    k_r, k_d = length(inp.Γ), length(inp.Γ_ds)
    @info string(now()) * " [mediator::compute_mediator_bounds] χ = $χ, ζ = $ζ, k_r = $(k_r), k_d = $(k_d), ε_r = $(inp.ε_r), ε_d = $(inp.ε_d)"
    # P = [U_dΓ, V_dΓ_ds] filled in place, a store block at a time: at N_m = 3M, whole-store
    # reads and their scaled copies held 3.5 P's (350 GB) on the host
    Uₛ = Matrix{ComplexF64}(undef, N_s, k_r)
    P = Matrix{ComplexF64}(undef, size(read_columns(inp.MS_U, 1:1), 1), k_r + k_d)
    for J in Iterators.partition(1:k_r, STORE_COLUMNS)
        C = read_columns(inp.RU_V, first(J):last(J))
        Uₛ[:, J] = C[1:N_s, :]
        P[:, J] = C[N_s+1:end, :] .* transpose(inp.Γ[J])
    end
    for J in Iterators.partition(1:k_d, STORE_COLUMNS)
        P[:, k_r .+ J] = read_columns(inp.MS_U, first(J):last(J)) .* transpose(inp.Γ_ds[J])
    end
    relres = Float64[]
    pbp(P) = ((M, relres) = mediator_M(compute_env, smr, P, (1 - loss_share) / ζ, k_mm); M)
    front = mediator_front(inp.Γ, Uₛ, read_columns(inp.MS_V, 1:k_d), P, pbp)
    Uₛ = P = nothing
    # σ₁ of the exact G⁰ₘₛ is at most the RSVD's plus its residual
    ball = tails == :paper ? 1 + ζ^2 * (inp.Γ_ds[1] + inp.ε_d)^2 : nothing
    n_max = min(channels, k_r)
    bounds, mu_p = fill(NaN, n_max), fill(NaN, n_max)
    for n in 1:n_max
        d = mediator_dual(front, n, ζ, tails == :none ? 0.0 : inp.ε_r, tails == :none ? 0.0 : inp.ε_d, loss_share; dev=use_gpu(compute_env) ? CuArray : identity, ball=ball)
        bounds[n], mu_p[n] = sqrt(d.μ), d.μ_p
        @info string(now()) * " [mediator::compute_mediator_bounds] [$n/$(n_max)] bound $(bounds[n]) on σₙ(Gᵣₛ) (vacuum $(get(inp.Γrs, n, NaN))), μₚ = $(mu_p[n])"
        flush(stderr) # logs redirected to a file are block-buffered
    end

    tmp_path = "$(out_path).tmp.$(getpid())"
    jldopen(tmp_path, "w") do f
        f["χ"] = χ
        f["true_bounds"] = bounds
        f["opt_mu_p"] = mu_p
        f["Γrs"] = inp.Γrs
        f["Γ_rs̃"] = inp.Γ
        f["Γ_ds"] = inp.Γ_ds
        f["eps_r"] = inp.ε_r
        f["eps_d"] = inp.ε_d
        f["k_mm"] = k_mm
        f["loss_share"] = loss_share
        f["tails"] = String(tails)
        isnothing(ball) || (f["ball"] = ball)
        f["B_relres"] = relres
    end
    mv(tmp_path, out_path; force=true)
    @info string(now()) * " [mediator::compute_mediator_bounds] Saved $(out_path)"
    return (bounds=bounds, mu_p=mu_p, Γrs=inp.Γrs)
end
