# The filled-block bound of Fig 2 at scale: the low-rank exact elimination of
# filled_blocks_qcqp.pdf (Eq. 50) with the truncation inside the relaxation (src/filled.jl), and
# the empty-mediator transfer P⁽⁰⁾ from receiver-alone solves (mediator_bound.pdf §8, Eq. 55–56).
#
#   1. vacuum basis on m: G_mb ≈ U_mb S V_bᴴ (rank k_mb) and the top k_a of Asym G⁰ₘₘ, giving the
#      real orthonormal U = orth{Re U_mb, Im U_mb, U_a}                            (no solves)
#   2. Y = W_bbᴴ V_b, so 𝓕̃ᴴU = Y S U_mbᴴU and Gᵇ̃ = G⁰ₘₘ + U_mb S (χ Yᴴ conj V_b) S U_mbᵀ  (k_mb solves)
#   3. P⁽⁰⁾ = (2 Im χ/χ²) A_rr⁻¹ U_r Σ (X conj V_s)ᵀ, G_rs ≈ U_r Σ V_sᴴ (rank k_rs)   (k_rs box + k_rs small solves)
#   4. ε₀, ε_F, ε_x: norms of what 1–3 leave out, on the true operators, by Lanczos  (2 solves per step)
#   5. the reduced dual for each n, with rank removal on P⁽⁰⁾ (:bound) and on [P⁽⁰⁾ Γ] (:joint)

"""
    filled_geometry(family, param, scale; m_face=nothing) -> (s_cells, gap_cells, m_cells)

`"wg"`: s, r = 1/4 × 1 × 1, gaps 1/8, m = param × 1 × 1 (param = ℓ); `"wgt"` (thin): s, r =
1/4 × 1/4 × 1/4, gaps 1/8, m = param × 1/4 × 1/4. `"ms05"`, `"ms1"`, `"ms2"`, `"ms4"`:
s, r, m = 1 × h × h (h = 1/2, 1, 2, 4), gaps = param (= d); `"ms44"`: m = 4 × 4 × 4. Lengths in
wavelengths, x first (the s → r axis); r is the same block as s. `m_face` (λ) replaces the
mediator's y, z side; the box centres every body.
"""
function filled_geometry(family::AbstractString, param::Rational, scale::Rational; m_face=nothing)
    c(x) = (n = x / scale; isinteger(n) || throw(ArgumentError("$x is not a multiple of $scale")); Int(n))
    m_face === nothing || return (g = filled_geometry(family, param, scale); (g[1], g[2], (g[3][1], c(m_face), c(m_face))))
    family == "wg" && return (c(1 // 4), c(1), c(1)), c(1 // 8), (c(param), c(1), c(1))
    q = 1 // 4
    family == "wgt" && return (c(q), c(q), c(q)), c(1 // 8), (c(param), c(q), c(q))
    h = get(Dict("ms05" => 1 // 2, "ms1" => 1 // 1, "ms2" => 2 // 1, "ms4" => 4 // 1, "ms44" => 4 // 1), family, 0 // 1)
    h > 0 || throw(ArgumentError("family must be wg, wgt, ms05, ms1, ms2, ms4 or ms44, not $family"))
    return (c(1), c(h), c(h)), c(param), (c(family == "ms44" ? 4 : 1), c(h), c(h))
end

"Channels evaluated: 1, then 2ᵏ and 3·2ᵏ (1, 2, 3, 4, 6, 8, 12, …) below `nmax`, and `nmax`; 40 gives the original list."
channel_grid(nmax) = sort!(union(filter(<=(nmax), [1; [m << k for m in (2, 3) for k in 0:30]]), nmax))

# Columns of `f` applied to the host columns of X, one device vector at a time.
function colmap(f, X::AbstractMatrix, nout::Int, dev)
    Y = Matrix{ComplexF64}(undef, nout, size(X, 2))
    for j in axes(X, 2)
        Y[:, j] = Array(f(dev(X[:, j])))
    end
    return Y
end

# Randomized SVD (q power iterations, p oversamples) of a matrix-free operator: A ≈ U S Vᴴ.
function rsvd_host(f, fadj, nin, nout, k, dev; p=16, q=2)
    orthc(X) = Matrix(qr(X).Q)
    Q = orthc(colmap(f, randn(ComplexF64, nin, k + p), nout, dev))
    for _ in 1:q
        Q = orthc(colmap(f, orthc(colmap(fadj, Q, nin, dev)), nout, dev))
    end
    F = svd(colmap(fadj, Q, nin, dev))   # (QᴴA)ᴴ = V S Wᴴ  ⇒  A ≈ Q W S Vᴴ
    return Q * F.V[:, 1:k], F.S[1:k], F.U[:, 1:k]
end

# Orthonormal basis of range(M) through the small Gram matrix (twice), directions below 1e-6 σ₁
# dropped: U only has to be orthonormal, and gesdd fails on 393216 × 6344 (32-bit workspace).
function orth_gram(M)
    E = eigen(Symmetric(M' * M))
    keep = E.values .> 1e-12 * maximum(E.values)
    Q = M * (E.vectors[:, keep] ./ transpose(sqrt.(E.values[keep])))
    E2 = eigen(Symmetric(Q' * Q))
    k2 = E2.values .> 0.5
    return Q * (E2.vectors[:, k2] ./ transpose(sqrt.(E2.values[k2])))
end

# Orthonormal coordinates of the basis behind a Gram matrix G = BᴴB: returns R = ZᴴB with Z an
# orthonormal basis of range B, directions below 1e-14 of the largest eigenvalue dropped, and
# √(largest dropped eigenvalue), which bounds ‖(I − ZZᴴ)B‖.
function gram_coords(G)
    E = eigen(Hermitian((G + G') / 2))
    keep = E.values .> 1e-14 * maximum(E.values)
    return sqrt.(E.values[keep]) .* E.vectors[:, keep]', sqrt(max(maximum(E.values[.!keep]; init=0.0), 0.0))
end

"""
    lanczos_norm(T, Tadj, n, dev; k=20, δ=1e-10, real_start=false) -> upper bound on ‖T‖

k Lanczos steps on TᴴT (full reorthogonalization) from a random start give the top Ritz value θ,
and P(λ₁ > θ/(1 − ε)) ≤ 1.648 √n e^{−√ε (2k−1)} (Kuczyński and Woźniakowski, SIAM J. Matrix
Anal. Appl. 13 (1992), Thm. 4.2). With ε set so that this is δ, the result bounds ‖T‖ except with
probability δ. A complex operator on Cⁿ counts as real symmetric on R²ⁿ; `real_start` is for an
operator that acts on real vectors only.
"""
function lanczos_norm(T, Tadj, n::Int, dev; k::Int=20, δ::Real=1e-10, real_start::Bool=false)
    rε = log(1.648 * sqrt(real_start ? n : 2n) / δ) / (2k - 1)
    rε < 1 || throw(ArgumentError("$k Lanczos steps are too few for n = $n, δ = $δ"))
    v = real_start ? complex.(randn(n)) : randn(ComplexF64, n)
    V, α, β = reshape(v ./ norm(v), :, 1), Float64[], Float64[]
    for j in 1:k
        w = Array(Tadj(T(dev(V[:, j]))))
        push!(α, real(dot(V[:, j], w)))
        for _ in 1:2
            w .-= V * (V' * w)
        end
        push!(β, norm(w))
        β[end] < 1e-12 * abs(α[1]) && break   # invariant subspace: θ is exact
        V = hcat(V, w ./ β[end])
    end
    θ = eigmax(SymTridiagonal(α, β[1:length(α)-1]))
    return sqrt(max(θ, 0) / (1 - rε^2))
end

# The ε = 0 value (1-D in λ, not a bound), then the bound at the AM-GM (t, a) from its maximizer.
# Asym Gᵇ projected on U (minus ε_x) is used when ε_x is small, and dropped (A ⪰ 0) otherwise.
function _dual(CqCq, CyCq, CyCy, M, UAU, ζ, ε0, εF, εx, εΓ)
    λ0 = 2ζ * max(opnorm(CyCy), 1e-30)
    function pair(A, ex)
        g0 = elim_reduced(; CqCq, CyCq, CyCy, M, UAU=A, ζ, ε0=0.0, εF=0.0, εx=0.0, εΓ=0.0)
        n0 = elim_minimize(g0; λ0, tails=false)
        _, w0 = g0(n0.λ, 1e-12, 1.0; wnorm=true)
        g = elim_reduced(; CqCq, CyCq, CyCy, M, UAU=A, ζ, ε0, εF, εx=ex, εΓ)
        return elim_fast(g; λ0=n0.λ, x0=sqrt(max(n0.value, 0)), w0, ε0, εΓ), n0
    end
    useA = εx < 0.5 / ζ
    b, n0 = pair(useA ? Matrix(UAU) : zeros(size(UAU)), useA ? εx : 0.0)
    if useA && !isfinite(b.value)
        b, n0 = pair(zeros(size(UAU)), 0.0)
    end
    return b.value, n0.value
end

"""
    filled_bound(family, param, scale; χ, k_mb, k_rs, k_a=200, nmax=40, on_gpu=true,
                 lanczos_steps=20, δ=1e-10, log=println) -> NamedTuple

Upper bounds on σₙ²(Pᵣₛ) over every structuring of the mediator, for n in
`channel_grid(nmax)` (geometry from `filled_geometry`, `m_face` passed on). Returns
`n => (bound, noeps, joint, joint_noeps, empty)`: the bounds with rank removal on P⁽⁰⁾ and on
[P⁽⁰⁾ Γ], the same with every ε set to zero (not bounds), and σₙ²(P⁽⁰⁾); plus the ε's and ranks.
Each bound holds except with probability 3δ (the three Lanczos estimates).
"""
function filled_bound(family::AbstractString, param::Rational, scale::Rational; χ::Number, k_mb::Int, k_rs::Int,
                      k_a::Int=200, nmax::Int=40, m_face=nothing, on_gpu::Bool=true, lanczos_steps::Int=20, δ::Real=1e-10, log=println)
    ζ = abs2(χ) / imag(χ)
    sc, gap, mc = filled_geometry(family, param, scale; m_face)
    fb = FilledBox(sc, gap, mc, gap, sc, scale, χ; on_gpu)
    Gv = vacuum(fb)
    Ns, Nm, Nb = length(fb.s), length(fb.m), length(fb.b)
    dev(x) = on_gpu ? CuArray(ComplexF64.(x)) : ComplexF64.(x)
    cm(f, X, nout) = colmap(f, X, nout, dev)
    # a rank above the operator's dimensions is full rank (small blocks or a short mediator)
    k_mb, k_rs, k_a = min(k_mb, Nb, Nm), min(k_rs, Ns), min(k_a, Nm)
    log("$family $param λ/$(denominator(scale)): box $(fb.cells), Ns $Ns, Nm $Nm; k_mb $k_mb, k_rs $k_rs, k_a $k_a")

    # 1. vacuum basis on m
    Gmb(y) = (Gv * embed(fb, y, fb.b))[fb.m]
    Gmb_adj(z) = conj.((Gv * embed(fb, conj.(z), fb.m))[fb.b])
    Umb, Smb, Vb = rsvd_host(Gmb, Gmb_adj, Nb, Nm, k_mb, dev)
    Am(w) = complex.(imag.((Gv * embed(fb, w, fb.m))[fb.m]))   # Asym G⁰ₘₘ w = Im(G⁰w) for real w
    Qa = Matrix(qr(real(cm(Am, randn(Nm, k_a + 16), Nm))).Q)
    Qa = Matrix(qr(real(cm(Am, Qa, Nm))).Q)
    Ea = eigen(Symmetric(real(Qa' * cm(Am, Qa, Nm))))
    U = orth_gram(hcat(real(Umb), imag(Umb), Qa * Ea.vectors[:, end:-1:end-k_a+1]))
    kU = size(U, 2)
    log("G_mb: σ₁ $(Smb[1]), σ_k $(Smb[end]); dim U $kU")

    # 2. Y = W_bbᴴ V_b (W_bb is complex symmetric: W_bbᴴ v = conj(W_bb conj v))
    Y = cm(v -> conj.(boxsolve(fb, embed(fb, conj.(v), fb.b))[fb.b]), Vb, Nb)
    Bk = Umb' * U
    Ck = Diagonal(Smb) * Bk                   # 𝓕̃ᴴU = Y Ck
    S = Diagonal(Smb)                         # UᴴAsym(Gᵇ̃)U = UᵀAsym(G⁰ₘₘ)U + Im(Bᴴ S (χ Yᴴ conj V_b) S conj B)
    UAm = U' * real(cm(Am, U, Nm))
    UAU = Symmetric((UAm + UAm') / 2 + imag(Bk' * (S * (χ .* (Y' * conj(Vb))) * S) * conj(Bk)))
    Umb = Vb = nothing
    log("Y done: $(fb.stats.solves) solves")

    # 3. P⁽⁰⁾ from G_rs ≈ U_r Σ V_sᴴ: A_blk⁻¹ v = χ (A⁻¹ E_b v)_b and A_rr⁻¹ v = χ A_solo⁻¹ v
    Grs(y) = (Gv * embed(fb, y, fb.s))[fb.r]
    Grs_adj(z) = conj.((Gv * embed(fb, conj.(z), fb.r))[fb.s])
    Ur, Srs, Vs = rsvd_host(Grs, Grs_adj, Ns, Ns, k_rs, dev)
    log("G_rs: σ_k/σ₁ $(Srs[end] / Srs[1]) at k_rs = $k_rs")
    XV = cm(v -> χ .* boxsolve(fb, embed(fb, conj.(v), fb.s))[fb.s], Vs, Ns)   # X conj V_s
    solo = solo_box(sc, scale, χ; on_gpu)
    R = cm(u -> χ .* boxsolve(solo, u), Ur, Ns)                                 # A_rr⁻¹ U_r
    Q1, R1 = qr(R)
    Q2, R2 = qr(XV)
    F = svd((2imag(χ) / χ^2) .* (R1 * Diagonal(Srs) * transpose(R2)))          # P⁽⁰⁾ ≈ Q1 (core) Q2ᵀ
    k0 = count(>(1e-8 * F.S[1]), F.S)
    U0, S0, V0 = Matrix(Q1) * F.U[:, 1:k0], F.S[1:k0], conj.(Matrix(Q2)) * F.V[:, 1:k0]

    # 4. tails on the true operators
    lz(T, Tadj, n; kw...) = lanczos_norm(T, Tadj, n, dev; k=lanczos_steps, δ, kw...)
    U0d, S0d, V0d = dev(U0), dev(S0), dev(V0)
    ε0 = lz(q -> P0_apply(fb, χ, q) .- U0d * (S0d .* (V0d' * q)), z -> P0_adj(fb, χ, z) .- V0d * (S0d .* (U0d' * z)), Ns)
    low(y) = dev(U * (Ck' * (Y' * Array(y))))     # Π𝓕̃ and its adjoint, on the host: Y may not fit the GPU
    low_adj(z) = dev(Y * (Ck * (U' * Array(z))))
    εF = lz(y -> F_apply(fb, y) .- low(y), z -> F_adj(fb, z) .- low_adj(z), Nb)
    D(w) = complex.(imag.(Gb_apply(fb, complex.(real.(w))))) .- dev(U * (UAU * (U' * real.(Array(w)))))
    εx = lz(D, D, Nm; real_start=true)            # D is real symmetric
    log("tails: ε₀ $ε0, ε_F $εF, ε_x $εx (ζ⁻¹ = $(1 / ζ))")

    # 5. the reduced problem from Gram matrices of the s and r blocks of Y (never N × kU):
    #    F̂ᴴ = Y_s Ck,  ΓU = 2 Im χ conj(Y_r) conj(Ck),  ΓUᴴU0 = 2 Im χ Ckᵀ Y_rᵀ U0.
    Ys, Yr = view(Y, 1:Ns, :), view(Y, Ns+1:Nb, :)
    VtYs, YrtU0, YrYr = V0' * Ys, transpose(Yr) * U0, Yr' * Yr
    # q-space Xs = [V0 Xc], Xc orthonormal on the part of span Y_s orthogonal to V0; the part of F̂ᴴ
    # along dropped directions (≤ δXc ‖Ck‖) joins ε_F.
    RXc, δXc = gram_coords(Ys' * Ys - VtYs' * VtYs)
    nXc = size(RXc, 1)
    nXs = k0 + nXc
    XsF = vcat(VtYs * Ck, RXc * Ck)                # XsᴴF̂ᴴ
    εF += δXc * opnorm(Ck)
    εΓ = 2imag(χ) * εF
    ΓtU0 = 2imag(χ) .* (transpose(Ck) * YrtU0)
    CyCy = (2imag(χ))^2 .* (transpose(Ck) * conj(YrYr) * conj(Ck))
    # Joint T = [P⁽⁰⁾ₖ ΓΠ] = Bᵣ Kc with Bᵣ = [U0 conj(Y_r)], in orthonormal coordinates of span Bᵣ;
    # its part along dropped directions (≤ ηT ‖[q; w]‖) joins ε₀ and ε_Γ of the joint variant.
    Rr, δr = gram_coords([I YrtU0'; YrtU0 conj(YrYr)])
    Kc = [Diagonal(S0) zeros(k0, nXc + kU); zeros(size(Ck, 1), nXs) (2imag(χ)) .* conj(Ck)]
    Tc = Rr * Kc
    ηT = δr * opnorm(Kc)
    FT = svd(Tc)

    bounds = Dict{Int, NamedTuple}()
    for n in channel_grid(nmax)
        # rank removal on P⁽⁰⁾: q ⊥ V0[:, 1:n-1], q ∈ span[V0[:, n:end] Xc]
        Cq = hcat(Diagonal(S0)[:, n:end], zeros(k0, nXc))
        b, b0 = _dual(Cq' * Cq, ΓtU0 * Cq, CyCy, XsF[vcat(n:k0, k0+1:nXs), :], UAU, ζ, ε0, εF, εx, εΓ)
        # rank removal on the joint T (Remark 1): σₙ(Pᵣₛ) ≤ σ₁((T − Tₙ₋₁)[q; w])
        Tn = Tc - FT.U[:, 1:n-1] * Diagonal(FT.S[1:n-1]) * FT.V[:, 1:n-1]'
        Tq, Ty = Tn[:, 1:nXs], Tn[:, nXs+1:end]
        j, j0 = _dual(Tq' * Tq, Ty' * Tq, Ty' * Ty, XsF, UAU, ζ, ε0 + ηT, εF, εx, εΓ + ηT)
        bounds[n] = (bound=b, noeps=b0, joint=j, joint_noeps=j0, empty=n <= k0 ? S0[n]^2 : NaN)
        log("n = $n: bound $b (ε = 0: $b0), joint $j (ε = 0: $j0)")
    end
    return (; bounds, ε0, εF, εx, kU, k_mb, k_rs, k0, solves=fb.stats.solves + solo.stats.solves,
            maxres=max(fb.stats.maxres, solo.stats.maxres))
end
