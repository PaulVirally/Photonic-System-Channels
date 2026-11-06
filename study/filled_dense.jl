## Dense comparison of the filled-block QCQP bounds of ../filled_blocks_qcqp.pdf on a coarse
## 3D system: sender s and receiver r solid χ blocks, mediator m binary.
##
##     julia --project=. study/filled_dense.jl <wg|ms1> <param> <n> <out.jld2> [scale]
##
## wg: s, r = 1/4 × 1 × 1 (x × y × z, x is the s→r axis), m = ℓ × 1 × 1, gaps 1/8, param = ℓ.
## ms1: s, r, m = 1 × 1 × 1, gaps d, param = d.
## Every (method, n) result is checkpointed to <out.jld2>; a rerun skips what is there.
## Bounds and primal values are on σ₁²(Pᵣₛ Q⊥) with Q = top n−1 right singular vectors of
## the empty-mediator P⁽⁰⁾ (paper Eq. 12); ≥ σₙ²(Pᵣₛ) of every structure.

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels
using LinearAlgebra, Printf, Random, JLD2

const χ = 18.612 + 1.5502660841406073im
const ζ = abs2(χ) / imag(χ)
const c = 4imag(χ) / ζ        # ‖Pᵣₛq‖² = c‖Eᵣᴴw‖²
const κ = 2imag(χ) / χ        # Pᵣₛ = κ Wᵣₛ
BLAS.set_num_threads(Threads.nthreads())

parse_rat(s) = (p = split(s, "//"); length(p) == 2 ? parse(Int, p[1]) // parse(Int, p[2]) : parse(Int, p[1]) // 1)
const FAMILY, PARAM, NS, OUT = ARGS[1], parse_rat(ARGS[2]), [parse(Int, ARGS[3])], ARGS[4]
const SCALE = length(ARGS) >= 5 ? parse_rat(ARGS[5]) : 1 // 8
logp(args...) = (println(args...); flush(stdout))

# ---------------------------------------------------------------- geometry and G⁰
function geometry()
    s, gap, m = filled_geometry(FAMILY, PARAM, SCALE)
    sep = (gap * SCALE, 0 // 1, 0 // 1)
    return s, m, SMRSystem(s, sep, m, sep, s, [Mediator], SCALE, ComplexF64(χ))
end
const SC, MC, SMR = geometry()
const ROOT = mkpath(joinpath(dirname(abspath(OUT)), "env_" * splitext(basename(OUT))[1]))
const ENVC = ComputeEnvironment(mkpath(joinpath(ROOT, "p")), mkpath(joinpath(ROOT, "q")),
                                mkpath(joinpath(ROOT, "s")), GPUChoice(false, -1))
const SYMS = (Sender, Mediator, Receiver)
t0 = time()
const G = let B = [PSC._dense_matrix(green_map(ENVC, SMR, X, Y), false) for X in SYMS, Y in SYMS]
    [B[1, 1] B[1, 2] B[1, 3]; B[2, 1] B[2, 2] B[2, 3]; B[3, 1] B[3, 2] B[3, 3]]
end
const Ns, Nm = 3prod(SC), 3prod(MC)
const Nr, Nu = Ns, size(G, 1)
const IS, IM, IR = 1:Ns, Ns+1:Ns+Nm, Ns+Nm+1:Nu
const IB = vcat(IS, IR)
antiherm(X) = (X - X') / 2im
const GA = Hermitian(antiherm(G))
const ES = Matrix{ComplexF64}(I, Nu, Nu)[:, IS]
logp(@sprintf("%s %s scale %s: Ns %d Nm %d Nu %d, G built in %.0f s", FAMILY, ARGS[2], SCALE, Ns, Nm, Nu, time() - t0))
logp(@sprintf("check: ‖G − Gᵀ‖/‖G‖ = %.1e, λmin Asym(G) = %.2e (λmax %.2e)", norm(G - transpose(G)) / norm(G),
              eigmin(GA), eigmax(GA)))

# ---------------------------------------------------------------- structures (primal)
# Pᵣₛ of the structure with mediator mask ρ ∈ {0,1}^cells: (1 − XG)w = Eₛq, X = diag χ(x).
cellmask(ρ) = repeat(ρ, 3) # dof order: cells (column-major) then component
function prs(ρ)
    x = vcat(fill(χ, Ns), χ .* cellmask(ρ), fill(χ, Nr))
    keep = findall(!iszero, x)  # empty points carry w = 0
    W = (I - x[keep] .* G[keep, keep]) \ Matrix{ComplexF64}(I, length(keep), length(keep))[:, 1:Ns]
    rpos = (length(keep) - Nr + 1):length(keep)
    return κ * W[rpos, :]
end
const P0 = prs(zeros(prod(MC)))
const PF = prs(ones(prod(MC)))
const SV0 = svd(P0)
logp(@sprintf("empty mediator σ₁..₄ = %s; full mediator σ₁..₄ = %s",
              join((@sprintf("%.3e", x) for x in SV0.S[1:4]), " "), join((@sprintf("%.3e", x) for x in svdvals(PF)[1:4]), " ")))
maximum(SV0.S) <= 1 + 1e-8 && maximum(svdvals(PF)) <= 1 + 1e-8 || logp("WARNING: σ(P) > 1, convention check failed")
Qperp(n) = SV0.V[:, n:end]

# ---------------------------------------------------------------- generic dual
# max α,w [α;w]ᴴA[α;w] s.t. [α;w]ᴴBⱼ[α;w] = 0, αᴴα = 1, with blocks (aa, wa, ww).
# g(λ) = λmax(H_aa + H_waᴴ K⁻¹ H_wa), K = −H_ww ≻ 0; ∂g/∂λⱼ = vᴴBⱼv (Danskin).
struct Form
    aa::Matrix{ComplexF64}
    wa::Matrix{ComplexF64}
    ww::Matrix{ComplexF64}
end
Base.:+(a::Form, b::Form) = Form(a.aa + b.aa, a.wa + b.wa, a.ww + b.ww)
Base.:*(t::Real, a::Form) = Form(t * a.aa, t * a.wa, t * a.ww)
quad(F::Form, α, w) = real(dot(α, F.aa, α) + 2real(dot(w, F.wa, α)) + dot(w, F.ww, w))

# β = Inf: the true λmax. Finite β: the soft-max λmax + log Σ exp(β(eᵢ − λmax))/β ≥ λmax,
# smooth where λmax is degenerate (non-radiating sender currents make it so); its gradient
# averages vᵢᴴBⱼvᵢ over the eigenvectors with the soft-max weights.
function gval(A::Form, Bs::Vector{Form}, λ; grad=false, β=Inf)
    H = A
    for (l, B) in zip(λ, Bs)
        H = H + l * B
    end
    C = cholesky(Hermitian(-H.ww); check=false)
    issuccess(C) || return Inf, nothing
    KH = C \ H.wa
    M = Hermitian(H.aa + H.wa' * KH)
    if isinf(β)
        E = eigen(M, size(M, 1):size(M, 1))
        g = E.values[1]
        grad || return g, nothing
        Vα = E.vectors[:, 1:1]
    else
        E = eigen(M)
        e = E.values
        wts = exp.(β .* (e .- e[end]))
        g = e[end] + log(sum(wts)) / β
        grad || return g, nothing
        p = wts ./ sum(wts)
        keep = findall(>(1e-9), p)
        Vα = E.vectors[:, keep] .* transpose(sqrt.(p[keep]))
    end
    Vw = KH * Vα
    q(B) = real(dot(Vα, B.aa * Vα) + 2real(dot(Vw, B.wa * Vα)) + dot(Vw, B.ww * Vw))
    return g, [q(B) for B in Bs]
end

function bfgs(f, λ0; iters=100, tol=1e-7)
    λ = copy(λ0)
    g, ∇ = f(λ)
    isfinite(g) || error("infeasible start")
    reset() = Matrix{Float64}(I, length(λ), length(λ)) * (0.1 * norm(λ) / max(norm(∇), 1e-300))
    Hi = reset()
    stall = 0
    for it in 1:iters
        d = -Hi * ∇
        dot(d, ∇) >= 0 && (Hi = reset(); d = -Hi * ∇)
        t = 1.0
        gn, ∇n, λn = Inf, nothing, λ
        for _ in 1:40
            λn = λ + t * d
            gn, ∇n = f(λn)
            isfinite(gn) && gn <= g + 1e-4 * t * dot(∇, d) && break
            t /= 2
        end
        if !(isfinite(gn) && gn < g)
            stall += 1
            Hi = reset() / 10
            stall >= 3 && break
            continue
        end
        s, y = λn - λ, ∇n - ∇
        if dot(s, y) > 1e-12 * norm(s) * norm(y)
            ρ = 1 / dot(s, y)
            Hi = (I - ρ * s * y') * Hi * (I - ρ * y * s') + ρ * s * s'
        end
        rel = (g - gn) / abs(g)
        λ, g, ∇ = λn, gn, ∇n
        stall = rel < tol ? stall + 1 : 0
        stall >= 5 && break
    end
    return λ
end

# Soft-max continuation; any feasible λ is a valid bound, so the reported value is the
# true λmax at the best multipliers met, and a stall only loosens it.
function minimize_dual(A, Bs, λ0; iters=100)
    λ = copy(λ0)
    best = (gval(A, Bs, λ)[1], λ)
    for b in (20.0, 100.0, 500.0, 2500.0)
        β = b / max(abs(best[1]), 1e-12)
        λ = bfgs(l -> gval(A, Bs, l; grad=true, β=β), λ; iters=iters)
        g = gval(A, Bs, λ)[1]
        g < best[1] && (best = (g, λ))
    end
    return best
end

# 1-D convex case: golden section on log λ over (λlo, ∞).
function minimize_1d(A, B, λlo)
    v, t = golden_min(t -> gval(A, [B], [λlo * (1 + exp(t))])[1], -12.0, 12.0; iters=40)
    return v, λlo * (1 + exp(t))
end

# ---------------------------------------------------------------- forms in (q, w) coordinates
# A Form whose `aa`/`wa` blocks live in q (sender) coordinates; `alpha(F, Q⊥)` maps q = Q⊥α.
alpha(F::Form, Qp) = Form(Qp' * F.aa * Qp, F.wa * Qp, F.ww)

proj(idx) = (p = zeros(Nu); p[idx] .= 1; Diagonal(p))
# Eq. (22): Im(wᴴ Z e(w, q)) = 0, e = (χ⁻¹ − G⁰)w − χ⁻¹Eₛq.
function zform(Z::AbstractMatrix)
    Zs = Z[:, IS]                         # Z Eₛ
    wa = -(1 / χ) / 2im * Zs              # B_wq = −χ⁻¹/(2i) Z Eₛ
    ww = antiherm(Z * (I / χ - G))
    return Form(zeros(ComplexF64, Ns, Ns), wa, ww)
end
const A_T1 = Form(zeros(ComplexF64, Ns, Ns), zeros(ComplexF64, Nu, Ns), Matrix(c * proj(IR)))

# Partition of a region (dof index range `rng`, cell dims `dims`) into blocks ≈ dims ./ parts.
function partition(rng, dims, parts)
    nc = prod(dims)
    lab = zeros(Int, dims)
    edges(n, p) = round.(Int, range(0, n, length=min(p, n) + 1))
    ex, ey, ez = (edges(dims[k], parts[k]) for k in 1:3)
    for (i, j, k) in Iterators.product(1:dims[1], 1:dims[2], 1:dims[3])
        a, b, d = searchsortedfirst(ex, i) - 1, searchsortedfirst(ey, j) - 1, searchsortedfirst(ez, k) - 1
        lab[i, j, k] = a + (length(ex) - 1) * ((b - 1) + (length(ey) - 1) * (d - 1))
    end
    lab = repeat(vec(lab), 3)
    return [rng[findall(==(l), lab)] for l in unique(lab)]
end

# ---------------------------------------------------------------- exact elimination operators (Eq. 47)
const Wbb = inv(I - χ * G[IB, IB])     # W⁽ⁱ⁾_bb
const Js, Jr = 1:Ns, Ns+1:Ns+Nr        # positions of s, r inside b
const Pz = κ * Wbb[Jr, Js]             # P⁽⁰⁾ (equals P0)
const Γ = κ * Wbb[Jr, :] * (χ * G[IB, IM])
const Fq = G[IM, IB] * Wbb[:, Js]
const Gb = G[IM, IM] + G[IM, IB] * Wbb * (χ * G[IB, IM])
const S = Hermitian(I / ζ + antiherm(Gb))
logp(@sprintf("check: ‖P⁽⁰⁾ − P0‖ = %.1e, λmin S·ζ = %.4f, ‖Gb − Gbᵀ‖/‖Gb‖ = %.1e",
              norm(Pz - P0), eigmin(S) * ζ, norm(Gb - transpose(Gb)) / norm(Gb)))
mform(idx) = (R = Diagonal([i in idx ? 1.0 : 0.0 for i in 1:Nm]);
              Form(zeros(ComplexF64, Ns, Ns), -(R * Fq) / 2im, -(R / ζ + antiherm(R * Gb))))
const A_EL = Form(Pz' * Pz, Γ' * Pz, Γ' * Γ)
λ0_el() = maximum(real, eigvals(Hermitian(Γ' * Γ), S)) # K = λS − ΓᴴΓ ≻ 0 for λ > λ0

# ---------------------------------------------------------------- linear fill tests
# βₖᴴ L v = 0 (Eq. 37) for the columns of β (on b): a_kᴴw = c_kᴴ q. Returns the transform
# (α, y) ↦ (α, Wp α + N y) applied to forms already in α coordinates.
function linear_transform(β, Qp)
    Aw = (I / χ - G)' * (zeros(ComplexF64, Nu, size(β, 2)) |> X -> (X[IB, :] = β; X)) # (χ⁻¹ − G)ᴴ E_b β
    C = (β' * vcat(Matrix{ComplexF64}(I, Ns, Ns), zeros(Nr, Ns))) / χ * Qp            # χ⁻¹ βᴴ E_bᴴEₛ Q⊥
    Wp = Aw * ((Aw' * Aw) \ C)
    N = qr(Aw).Q * Matrix{ComplexF64}(I, Nu, Nu)
    N = N[:, size(β, 2)+1:end]
    return F -> Form(F.aa + Wp' * F.wa + F.wa' * Wp + Wp' * F.ww * Wp, N' * (F.wa + F.ww * Wp), N' * F.ww * N)
end

# ---------------------------------------------------------------- run
res = isfile(OUT) ? load(OUT, "res") : Dict{String,Any}()
save_res() = (tmp = OUT * ".tmp"; jldsave(tmp; res); mv(tmp, OUT; force=true))
function run!(f, key)
    haskey(res, key) && return logp(@sprintf("%-34s %s (cached)", key, res[key]))
    t = time()
    v = f()
    res[key] = v
    save_res()
    logp(@sprintf("%-34s %s  [%.0f s]", key, v, time() - t))
end
res["meta"] = (family=FAMILY, param=PARAM, scale=SCALE, Ns=Ns, Nm=Nm, χ=χ, sv0=SV0.S[1:12], svF=svdvals(PF)[1:12])
save_res()

const T1_PARTS = [IS, IM, IR]
for n in NS
    Qp = Qperp(n)
    run!("empty n=$n") do; opnorm(P0 * Qp)^2 end
    run!("full n=$n") do; opnorm(PF * Qp)^2 end
    # global real power: equals 1 (Prop. 1)
    # tier-1 variants all gave 1.000 at wg ℓ = 1/4 (global, sub-blocks, Z2, radiative-mode tests):
    # rerun them only there; elsewhere they cost hours at Nu = 4608 and say nothing new
    const_check = FAMILY == "wg" && PARAM == 1 // 4 && SCALE == 1 // 8
    const_check && run!("global n=$n") do; minimize_dual(alpha(A_T1, Qp), [alpha(zform(I(Nu) * 1.0), Qp)], [2c * ζ])[1] end
    # tier 1: region-resolved real power
    T1 = [alpha(zform(proj(p)), Qp) for p in T1_PARTS]
    A1 = alpha(A_T1, Qp)
    run!("tier1 n=$n") do; minimize_dual(A1, T1, fill(2c * ζ, 3))[1] end
    # tier 1 with each region split 2×2×2
    sub = vcat(partition(IS, SC, (2, 2, 2)), partition(IM, MC, (2, 2, 2)), partition(IR, SC, (2, 2, 2)))
    const_check && run!("tier1-sub222 n=$n") do
        minimize_dual(A1, [alpha(zform(proj(p)), Qp) for p in sub], fill(2c * ζ, length(sub)))[1]
    end
    # tier 1 + Z2 quadratic tests: Z = u uᴴ (and iZ), u = top 3 radiative modes of s and of r
    modes = Vector{ComplexF64}[]
    for idx in (IS, IR)
        E = eigen(Hermitian(Matrix(GA)[idx, idx])); for k in 0:2
            u = zeros(ComplexF64, Nu); u[idx] = E.vectors[:, end-k]; push!(modes, u)
        end
    end
    Z2 = vcat([alpha(zform(u * u'), Qp) for u in modes], [alpha(zform(im * u * u'), Qp) for u in modes])
    const_check && run!("tier1+Z2x12 n=$n") do; minimize_dual(A1, vcat(T1, Z2), vcat(fill(2c * ζ, 3), zeros(length(Z2))))[1] end
    # tier 1 + linear tests on the receiver's radiative modes, then all of r, then all of b
    Er = eigen(Hermitian(Matrix(GA)[IR, IR]))
    for (lab, β) in (("r-rad4", vcat(zeros(Ns, 4), Er.vectors[:, end-3:end])),
                     ("r-rad16", vcat(zeros(Ns, 16), Er.vectors[:, end-15:end])),
                     ("all-r", vcat(zeros(Ns, Nr), Matrix{ComplexF64}(I, Nr, Nr))),
                     ("all-b", Matrix{ComplexF64}(I, Ns + Nr, Ns + Nr)))
        (const_check || lab in ("all-r", "all-b")) && run!("tier1+lin-$lab n=$n") do
            Tf = linear_transform(β, Qp)
            minimize_dual(Tf(A1), [Tf(B) for B in T1], fill(2c * ζ, 3))[1]
        end
    end
    # exact elimination, Eq. (50): one multiplier
    λlo = λ0_el()
    Ael = alpha(A_EL, Qp)
    Bel = alpha(mform(1:Nm), Qp)
    run!("elim n=$n") do; minimize_1d(Ael, Bel, λlo)[1] end
    # exact elimination + real power on subregions of m
    for parts in ((2, 2, 2), (4, 4, 4))
        sub = partition(1:Nm, MC, parts)
        run!("elim-sub$(join(parts)) n=$n") do
            λs = minimize_1d(Ael, Bel, λlo)[2]
            minimize_dual(Ael, [alpha(mform(p), Qp) for p in sub], fill(λs, length(sub)))[1]
        end
    end
    # Remark 1: rank removal on T = [P⁽⁰⁾ Γ], q unrestricted
    run!("elim-joint n=$n") do
        T = hcat(Pz, Γ)
        Ft = svd(T)
        Tn = T - Ft.U[:, 1:n-1] * Diagonal(Ft.S[1:n-1]) * Ft.V[:, 1:n-1]'
        Aj = Form(Tn[:, 1:Ns]' * Tn[:, 1:Ns], Tn[:, Ns+1:end]' * Tn[:, 1:Ns], Tn[:, Ns+1:end]' * Tn[:, Ns+1:end])
        λj = maximum(real, eigvals(Hermitian(Aj.ww), S))
        minimize_1d(Aj, mform(1:Nm), λj)[1]
    end
end

# ---------------------------------------------------------------- primal: adjoint ascent on ρ, then greedy flips
# f(ρ) = σ₁²(Pᵣₛ(ρ)Q⊥); continuous χρ is only a heuristic path, the reported value is binary.
function primal(n; iters=60, seed=1)
    Qp = Qperp(n)
    nc = prod(MC)
    function fgrad(ρ)
        x = vcat(fill(χ, Ns), χ .* cellmask(ρ), fill(χ, Nr))
        Mx = I - x .* G
        Lu = lu(Mx)
        Wcols = Lu \ ES
        P = κ * Wcols[IR, :] * Qp
        F = svd(P)
        q = Qp * F.V[:, 1]
        p = F.U[:, 1] * F.S[1]
        b = G * (Wcols * q)                     # G W Eₛ q
        a = Lu' \ (zeros(ComplexF64, Nu) |> v -> (v[IR] = conj(κ) * p; v)) # Wᴴ Eᵣ κ* p
        gd = 2real.(χ .* conj.(a[IM]) .* b[IM]) # df/dx_dof
        return F.S[1]^2, vec(sum(reshape(gd, nc, 3); dims=2))
    end
    best = (0.0, zeros(nc))
    for start in (zeros(nc), ones(nc), fill(0.5, nc))
        ρ = copy(start)
        f, g = fgrad(ρ)
        step = 0.2 / max(maximum(abs, g), 1e-300)
        for _ in 1:iters
            ρn = clamp.(ρ .+ step .* g, 0, 1)
            fn, gn = fgrad(ρn)
            if fn > f
                ρ, f, g = ρn, fn, gn; step *= 1.2
            else
                step /= 2
            end
        end
        ρb = Float64.(ρ .> 0.5)
        fb, gb = fgrad(ρb)
        # greedy: flip the cells whose gradient favours flipping, in shrinking batches
        k = max(1, nc ÷ 8)
        while k >= 1
            want = [(ρb[i] == 0 ? gb[i] : -gb[i]) for i in 1:nc]
            cand = sortperm(want; rev=true)[1:k]
            cand = filter(i -> want[i] > 0, cand)
            isempty(cand) && break
            ρt = copy(ρb); ρt[cand] .= 1 .- ρt[cand]
            ft, gt = fgrad(ρt)
            if ft > fb
                ρb, fb, gb = ρt, ft, gt
            else
                k ÷= 2
            end
        end
        fb > best[1] && (best = (fb, ρb))
    end
    return best
end
for n in NS
    run!("primal n=$n") do; primal(n)[1] end
end
logp("done")
