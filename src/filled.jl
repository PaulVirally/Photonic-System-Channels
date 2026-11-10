# Filled sender and receiver blocks (filled_blocks_qcqp.pdf, §6): every operator of the exact
# elimination is one solve on the box that encloses s, m and r, with χ on s ∪ r and zero on
# the mediator and the gaps. With A = I − XG⁰ on the box and Eₓ the inclusion of region x,
#
#     W_bb⁽ⁱ⁾ y = Eᵦᴴ A⁻¹ Eᵦ y,        P⁽⁰⁾ q = κ Eᵣᴴ A⁻¹ Eₛ q,        κ = 2 Im χ / χ,
#     𝓕 y = Eₘᴴ G⁰ A⁻¹ Eᵦ y,           Gᵇₘₘ w = Eₘᴴ G⁰ A⁻¹ Eₘ w,
#
# (A⁻¹ leaves a source on m in place, since X vanishes there). All four are complex
# symmetric (G⁰ᵀ = G⁰, χ constant on b), so every adjoint is conj ∘ op ∘ conj.

using GilaElectromagnetics, CUDA, LinearAlgebra

mutable struct SolveStats
    solves::Int
    products::Int
    maxres::Float64
    seconds::Float64
end
SolveStats() = SolveStats(0, 0, 0.0, 0.0)

"""
    FilledBox(s_cells, gap1, m_cells, gap2, r_cells, scale, χ; on_gpu, tol=1e-11)

The box (x is the s → r axis; bodies centred in y, z, the box as wide as the widest; gaps in cells) with its
operator A = I − XG⁰ and the shifted-frequency preconditioner (`ShiftedSolve`). `s`, `m`, `r`
and `b = s ∪ r` index the box vector. Solves stop on the true relative residual `tol`.
"""
struct FilledBox{O, P, I}
    A::O
    pc::P
    cells::NTuple{3, Int}
    s::I; m::I; r::I; b::I
    tol::Float64
    stats::SolveStats
end

function _filled_box(χa, scale, on_gpu, tol, regions)
    cells = size(χa)
    vol = GlaVol(cells, (scale, scale, scale))
    A = on_gpu ? InvSctOpr{Float64}(vol, χa; dev=CUDA.CUDABackend()) : InvSctOpr{Float64}(vol, χa)
    LI = LinearIndices((cells..., 3))
    idx((xx, yy, zz)) = vec([LI[i, j, k, c] for i in xx, j in yy, k in zz, c in 1:3])
    up(v) = on_gpu ? CuArray(v) : v
    s, m, r = idx.(regions)
    return FilledBox(A, ShiftedSolve(A, χa), cells, up(s), up(m), up(r), up(vcat(s, r)), Float64(tol), SolveStats())
end

function FilledBox(sc::NTuple{3, Int}, gap1::Int, mc::NTuple{3, Int}, gap2::Int, rc::NTuple{3, Int},
                   scale::Rational{Int}, χ::Number; on_gpu::Bool, tol::Real=1e-11)
    nx, ny, nz = sc[1] + gap1 + mc[1] + gap2 + rc[1], max(sc[2], mc[2], rc[2]), max(sc[3], mc[3], rc[3])
    mid(n, N) = iseven(N - n) ? ((N - n) ÷ 2 + 1:(N + n) ÷ 2) : throw(ArgumentError("faces $sc $mc $rc can't be centred"))
    reg(xx, c) = (xx, mid(c[2], ny), mid(c[3], nz))
    s, m, r = reg(1:sc[1], sc), reg(sc[1]+gap1+1:sc[1]+gap1+mc[1], mc), reg(nx-rc[1]+1:nx, rc)
    χa = zeros(ComplexF64, nx, ny, nz)
    χa[s...] .= χ
    χa[r...] .= χ
    return _filled_box(χa, scale, on_gpu, tol, (s, m, r))
end

"""
    solo_box(cells, scale, χ; on_gpu, tol=1e-11) -> FilledBox

One filled block alone (χ everywhere in `cells`), so A = I − χG⁰ = χ(χ⁻¹ − G⁰) and
A_rr⁻¹ v = χ · boxsolve(box, v) (mediator_bound.pdf §8). `r` and `b` index the whole box.
"""
solo_box(cells::NTuple{3, Int}, scale::Rational{Int}, χ::Number; on_gpu::Bool, tol::Real=1e-11) =
    _filled_box(fill(ComplexF64(χ), cells), scale, on_gpu, tol, ((1:0, 1:0, 1:0), (1:0, 1:0, 1:0), Base.OneTo.(cells)))

nbox(fb::FilledBox) = 3prod(fb.cells)
vacuum(fb::FilledBox) = GilaElectromagnetics.vac(fb.A)
function embed(fb::FilledBox, v::AbstractVector, idx)
    x = fill!(similar(v, nbox(fb)), 0)
    x[idx] .= v
    return x
end

const GMRES_RESTART, GMRES_MAXIT = 40, 20000

"""
    boxsolve(fb, rhs) -> A⁻¹ rhs

Flexible GMRES (the preconditioner is an inner iteration) to the true relative residual `fb.tol`,
with up to three refinement sweeps on the recomputed residual. Errors if it never gets there.
"""
function boxsolve(fb::FilledBox, rhs::AbstractVector)
    t = time()
    gmres(abstol, reltol) = GMRESSolver(GMRES_RESTART, GMRES_MAXIT, abstol, reltol; preCon=fb.pc, flexible=true)
    log = SlvLog()
    x = solve(fb.A, rhs, gmres(0.0, fb.tol); log)
    nb = norm(rhs)
    res = similar(rhs)
    rel = Inf
    for sweep in 0:3
        mul!(res, fb.A, x)
        res .= rhs .- res
        rel = norm(res) / nb
        rel <= fb.tol && break
        sweep == 3 && error("boxsolve: true residual $rel above $(fb.tol) after refinement")
        l2 = SlvLog()
        x .+= solve(fb.A, res, gmres(nb * fb.tol / 2, 0.0); log=l2)
        log.oprApp += l2.oprApp
    end
    st = fb.stats
    st.solves += 1
    st.products += log.oprApp
    st.maxres = max(st.maxres, rel)
    st.seconds += time() - t
    return x
end

κ_of(χ) = 2imag(χ) / χ

# The maps, on region vectors (device arrays when on the GPU). Each is complex symmetric.
P0_apply(fb, χ, q) = κ_of(χ) .* boxsolve(fb, embed(fb, q, fb.s))[fb.r]
P0_adj(fb, χ, z) = conj.(κ_of(χ) .* boxsolve(fb, embed(fb, conj.(z), fb.r))[fb.s])
F_apply(fb, y) = (vacuum(fb) * boxsolve(fb, embed(fb, y, fb.b)))[fb.m]
function F_adj(fb, z)  # 𝓕ᴴ = conj ∘ W_bb G_bm ∘ conj
    g = (vacuum(fb) * embed(fb, conj.(z), fb.m))[fb.b]
    return conj.(boxsolve(fb, embed(fb, g, fb.b))[fb.b])
end
Gb_apply(fb, w) = (vacuum(fb) * boxsolve(fb, embed(fb, w, fb.m)))[fb.m]

# ---------------------------------------------------------------------------------------------
# Low-rank exact elimination with the truncation inside the relaxation.
#
# U (real, orthonormal, on m) spans the ranges of G_mb, conj G_mb and Asym G⁰ₘₘ; with Π = UUᴴ,
# ranges F, Γᴴ, Asym Gᵇ ⊂ range Π up to tails. For any feasible (q, w), with y = Uᴴw, s = ‖Π⊥w‖,
# A = Asym Gᵇₘₘ ⪰ 0, ε_x = ‖A − ΠÂΠ‖, ε_F ≥ ‖𝓕 − Π𝓕̃‖, ε₀ ≥ ‖P⁽⁰⁾ − P⁽⁰⁾ₖ‖, ε_Γ = 2 Im χ ε_F:
#   (ζ⁻¹ − ε_x)(‖y‖² + s²) + yᴴÂy ≤ Im(qᴴF̂ᴴy) + ε_F‖w‖,
#   ‖Pᵣₛq‖ ≤ ‖P⁽⁰⁾ₖq + Γ̂y‖ + ε₀ + ε_Γ‖w‖,     ‖w‖² = ‖y‖² + s²,
# where Â, F̂, Γ̂ are the approximations actually used. Dropping A (A ⪰ 0) is also valid: Â = 0,
# ε_x = 0. AM-GM (a > 0) and (x + b)² ≤ (1 + t)x² + (1 + 1/t)b² make both homogeneous in
# (q, y, s), so the Lagrange dual with λ ≥ 0 bounds σ₁²(Pᵣₛ Q⊥) for every λ, t, a it is finite at.
# ---------------------------------------------------------------------------------------------

"""
    elim_reduced(; CqCq, CyCq, CyCy, M, UAU, ζ, ε0, εF, εx, εΓ) -> (λ, t, a) ↦ value

The dual function of the reduced, tail-relaxed QCQP, from Gram matrices of the objective:
C_q: α ↦ P⁽⁰⁾ₖX'α and C_y: y ↦ ΓUy (so CqCq = C_qᴴC_q, CyCq = C_yᴴC_q, CyCy = C_yᴴC_y);
M = X'ᴴF̂ᴴ and UAU = Uᴴ(Asym Gᵇ)U. Returns +Inf where the Schur block is not negative definite.
With `wnorm = true` it also returns ‖y‖ of the maximizer (‖q‖ = 1).
"""
function elim_reduced(; CqCq, CyCq, CyCy, M, UAU, ζ, ε0, εF, εx, εΓ)
    CqCq, CyCy = Hermitian(Matrix(CqCq)), Hermitian(Matrix(CyCy))
    function g(λ, t, a; wnorm::Bool=false)
        σs = 1 / ζ - εx - εF / (2a)
        λ > 0 && σs > 0 && λ * σs > 2(1 + 1 / t) * εΓ^2 || return wnorm ? (Inf, NaN) : Inf
        Haa = (1 + t) * CqCq + (2(1 + 1 / t) * ε0^2 + λ * εF * a / 2) * I
        Hya = (1 + t) * CyCq .- (λ / 2im) .* M'
        # ε_Γ‖w‖ with ‖w‖² = ‖y‖² + s²: the tail term sits on y as well as on s
        K = Hermitian(λ * (σs * I + UAU) - (1 + t) * CyCy - 2(1 + 1 / t) * εΓ^2 * I)
        C = cholesky(K; check=false)
        issuccess(C) || return wnorm ? (Inf, NaN) : Inf
        KH = C \ Hya
        H = Hermitian(Haa + Hya' * KH)
        wnorm || return eigvals(H, size(H, 1):size(H, 1))[1]
        E = eigen(H, size(H, 1):size(H, 1))
        return E.values[1], norm(KH * E.vectors[:, 1])
    end
    return g
end

"""
    elim_minimize(g; λ0, tails=true) -> (value, λ, t, a)

min over λ, t, a > 0 of the dual function: golden section in log λ inside a coordinate search
over (log t, log a). Every point evaluated is a valid bound; the smallest is returned. Without
tails (ε = 0) only λ matters.
"""
function elim_minimize(g; λ0::Real, tails::Bool=true)
    bestλ(lt, la) = golden_min(x -> g(exp(x), exp(lt), exp(la)), log(λ0) - 12, log(λ0) + 12; iters=36)
    if !tails
        v, x = bestλ(log(1e-12), 0.0)
        return (value=v, λ=exp(x), t=0.0, a=0.0)
    end
    lt, la = log(1e-3), 0.0
    v, x = bestλ(lt, la)
    for _ in 1:2
        v1, lt1 = golden_min(z -> bestλ(z, la)[1], lt - 8, lt + 8; iters=14)
        if v1 < v
            lt = lt1
            v, x = bestλ(lt, la)
        end
        v1, la1 = golden_min(z -> bestλ(lt, z)[1], la - 10, la + 10; iters=14)
        if v1 < v
            la = la1
            v, x = bestλ(lt, la)
        end
    end
    return (value=v, λ=exp(x), t=exp(lt), a=exp(la))
end

"""
    elim_fast(g; λ0, x0, w0, ε0, εΓ) -> (value, λ, t, a)

One golden search in λ at the AM-GM choices t = (ε₀ + ε_Γ w₀)/x₀, a = w₀, taken from the
ε = 0 solution (value x₀², maximizer ‖y‖ = w₀). Any t, a > 0 gives a valid bound.
"""
function elim_fast(g; λ0::Real, x0::Real, w0::Real, ε0::Real, εΓ::Real)
    t = max((ε0 + εΓ * w0) / max(x0, 1e-300), 1e-12)
    a = max(w0, 1e-12)
    v, x = golden_min(x -> g(exp(x), t, a), log(λ0) - 12, log(λ0) + 12; iters=28)
    return (value=v, λ=exp(x), t=t, a=a)
end
