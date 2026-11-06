## The mediator bound on a tiny CPU system, everything dense: the low-rank μₙ(μₚ)
## against the Schur complement of Eq. (33) built from full matrices, the bound against
## σₙ(Gᵣₛ) of real structures (vacuum, solid, random fillings, exact T), and the
## truncated-RSVD bound with its tails in the dual, or in the paper's ball, against the same structures.
##
## Run from the repo root:
##     julia --project=. test/mediator_dense.jl

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels
using LinearAlgebra, Printf, Random

failures = String[]
function check(name, ok, detail="")
    ok || push!(failures, name)
    @printf("%-58s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
end

const ROOT = mktempdir(; cleanup=true)
const ENVC = ComputeEnvironment(mkpath(joinpath(ROOT, "p")), mkpath(joinpath(ROOT, "q")),
                                mkpath(joinpath(ROOT, "s")), GPUChoice(false, -1))
const χ = 13.6 + 0.05im
const ζ = abs(χ)^2 / imag(χ)
const SMR = SMRSystem((2, 8, 8), (1//8, 0//1, 0//1), (4, 8, 8), (1//8, 0//1, 0//1), (2, 8, 8),
                      [Mediator], 1//16, χ)
dense(op) = PSC._dense_matrix(op, false)
const Grs = dense(green_map(ENVC, SMR, Receiver, Sender))
const Gru = dense(green_map(ENVC, SMR, [Receiver], [Sender, Mediator]))
const Gms = dense(green_map(ENVC, SMR, Mediator, Sender))
const Gmm = dense(green_map(ENVC, SMR, Mediator, Mediator))
const Amm = Hermitian(dense(asym_green_map(ENVC, SMR, Mediator)))
const Ns, Nm = size(Gms, 2), size(Gms, 1)
const Grm = Gru[:, Ns+1:end]
check("G_r←[s;m] is [G_rs G_rm]", isapprox(Gru[:, 1:Ns], Grs; rtol=1e-10))
check("Asym(G_mm) matches (G − Gᴴ)/2i", isapprox(Amm, (Gmm - Gmm') / 2im; rtol=1e-8))

# The bound's inputs from (possibly truncated) exact SVDs, with c the loss the solve uses.
function front_from(kr, kd, c)
    F, Fd = svd(Gru), svd(Gms)
    pbp(P) = P' * ((c * I + Amm) \ P)
    front = mediator_front(F.S[1:kr], F.V[1:Ns, 1:kr], F.V[Ns+1:end, 1:kr],
                           Fd.S[1:kd], Fd.U[:, 1:kd], Fd.V[:, 1:kd], pbp)
    ε_r = kr < length(F.S) ? F.S[kr+1] : 0.0
    ε_d = kd < length(Fd.S) ? Fd.S[kd+1] : 0.0
    return front, ε_r, ε_d
end

# 1. Low-rank curve against the dense Schur complement, at ε = 0 (full SVDs, c = ζ⁻¹).
const NCH = 8
full, _, _ = front_from(size(Gru, 1), Ns, 1 / ζ)
A = Hermitian(Gru' * Gru)
EA = eigen(A)
s, d = 1:Ns, Ns+1:Ns+Nm
Bds = Gms / 2im
Bdd = Hermitian(I / ζ + Amm)
for n in (1, 2, 5)
    An = A - EA.vectors[:, end-n+2:end] * Diagonal(EA.values[end-n+2:end]) * EA.vectors[:, end-n+2:end]'
    dense_μ(μ) = eigmax(Hermitian(An[s, s] + (An[s, d] - μ * Bds') * ((μ * Bdd - An[d, d]) \ (An[d, s] - μ * Bds))))
    μₙ, lo = mediator_curve(full, n, ζ, 0, 0, 0)
    errs = [abs(μₙ(f * lo) - dense_μ(f * lo)) / dense_μ(f * lo) for f in (1.5, 4.0, 30.0)]
    check("n=$n: low-rank μₙ(μₚ) = dense Schur", maximum(errs) < 1e-7, @sprintf("max rel err %.1e", maximum(errs)))
    check("n=$n: domain edge = dense λ_max(A_dd, B_dd)", isapprox(lo, maximum(eigvals(Hermitian(An[d, d]), Bdd)); rtol=1e-6))
    μ = 4.0 * lo
    S = Hermitian(An[s, s] + (An[s, d] - μ * Bds') * ((μ * Bdd - An[d, d]) \ (An[d, s] - μ * Bds)))
    xd = eigen(S).vectors[:, end]
    td = (μ * Bdd - An[d, d]) \ ((An[d, s] - μ * Bds) * xd)
    pr = mediator_primal(full, n, μ)
    F, Fd = svd(Gru), svd(Gms)
    X = hcat(F.V[1:Ns, :] .* transpose(F.S), Fd.V)[:, pr.idx]
    P = hcat(F.V[Ns+1:end, :] .* transpose(F.S), Fd.U .* transpose(Fd.S))[:, pr.idx]
    x = X * pr.x_coef
    t = Bdd \ (P * pr.c)
    ph = dot(xd, x) / abs(dot(xd, x)) # eigenvectors agree up to a phase: x ≈ ph·xd
    check("n=$n: primal z = dense Schur maximizer", norm(x - ph * xd) < 1e-6 && norm(t - ph * td) < 1e-6 * norm(td),
          @sprintf("|⟨xd, x⟩| %.8f, ‖Δx‖ %.1e, ‖Δt‖/‖t‖ %.1e", abs(dot(xd, x)), norm(x - ph * xd), norm(t - ph * td) / norm(td)))
end

# 2. Real structures: vacuum, solid, random fillings, T = (χ⁻¹ − G_mm)⁻¹ on the filled voxels.
Random.seed!(1)
function σ_structure(mask)
    f = findall(mask)
    T = inv(I / χ - Gmm[f, f])
    return svdvals(Grs + Grm[:, f] * T * Gms[f, :])[1:NCH]
end
voxel = repeat(1:Nm÷3, outer=3) # Gila stores (cells..., 3): a cell's 3 polarizations are filled together
cells = [falses(Nm ÷ 3), trues(Nm ÷ 3), [rand(Nm ÷ 3) .< p for p in rand(20)]...]
σs = reduce(hcat, [σ_structure(c[voxel]) for c in cells])
bound_full = [sqrt(mediator_dual(full, n, ζ, 0, 0, 0).μ) for n in 1:NCH]
worst = maximum(σs ./ bound_full)
check("full-SVD bound ≥ σₙ of 22 structures", worst <= 1 + 1e-8, @sprintf("max σ/bound %.4f", worst))
@printf("  n  bound      vacuum     solid      best random\n")
for n in 1:NCH
    @printf("  %d  %.4e %.4e %.4e %.4e\n", n, bound_full[n], σs[n, 1], σs[n, 2], maximum(σs[n, 3:end]))
end

# 3. Truncated SVDs, tails in the dual: still a bound for every loss share f. The ratio to
# the full-SVD bound splits into the cost of setting the loss aside (ε = 0 at the same c)
# and the cost of the tails themselves.
for f in (0.5, 0.1, 0.02, 1e-4), (kr, kd) in ((60, 60), (30, 120))
    c = (1 - f) / ζ
    tr, ε_r, ε_d = front_from(kr, kd, c)
    b = [sqrt(mediator_dual(tr, n, ζ, ε_r, ε_d, f).μ) for n in 1:NCH]
    aside, _, _ = front_from(size(Gru, 1), Ns, c)
    b0 = [sqrt(mediator_dual(aside, n, ζ, 0, 0, 0).μ) for n in 1:NCH]
    w = maximum(σs ./ b)
    check("f=$f k_r=$kr k_d=$kd (ε_r=$(round(ε_r; sigdigits=2)), ε_d=$(round(ε_d; sigdigits=2))): bound ≥ σₙ",
          w <= 1 + 1e-8, @sprintf("σ/bound ≤ %.3f; aside/full %.3f..%.3f, bound/aside %.3f..%.3f", w,
                                  extrema(b0 ./ bound_full)..., extrema(b ./ b0)...))
end

# 3b. The paper's a-posteriori ball at the full loss: also a bound, and the comparison with
# the tails in the dual at f = 0.1.
for (kr, kd) in ((60, 60), (30, 120))
    tr, ε_r, ε_d = front_from(kr, kd, 1 / ζ)
    ball = 1 + ζ^2 * opnorm(Gms)^2
    b = [sqrt(mediator_dual(tr, n, ζ, ε_r, ε_d, 0; ball=ball).μ) for n in 1:NCH]
    trd, _, _ = front_from(kr, kd, 0.9 / ζ)
    bd = [sqrt(mediator_dual(trd, n, ζ, ε_r, ε_d, 0.1).μ) for n in 1:NCH]
    w = maximum(σs ./ b)
    check("paper ball k_r=$kr k_d=$kd: bound ≥ σₙ", w <= 1 + 1e-8,
          @sprintf("σ/bound ≤ %.3f; paper/dual %.3f..%.3f", w, extrema(b ./ bd)...))
end

# 4. The production pieces: the h5 store round trip, the residual bound against the exact
# σ_{k+1}, and the batched, preconditioned B⁻¹ against a dense solve.
F = svd(Gru)
upath = joinpath(ROOT, "U.h5")
PSC.save_tall(F.U[:, 1:40], upath)
check("h5 store round trip", PSC.read_columns(upath, 1:40) == F.U[:, 1:40])
ε = PSC.residual_bound(green_map(ENVC, SMR, [Receiver], [Sender, Mediator]), upath, false)
check("residual bound ≥ σ₄₁ and within 3×", F.S[41] <= ε <= 3 * F.S[41], @sprintf("ε/σ₄₁ = %.2f", ε / F.S[41]))
Ptest = hcat(F.V[Ns+1:end, 1:20] .* transpose(F.S[1:20]), svd(Gms).U[:, 1:20])
Mref = Ptest' * ((I / 2ζ + Amm) \ Ptest)
for k_mm in (0, 40)
    M, rr = PSC.mediator_M(ENVC, SMR, Ptest, 1 / 2ζ, k_mm)
    check("B⁻¹ by PCG, k_mm = $(k_mm)", norm(M - Mref) <= 1e-8 * norm(Mref) && maximum(rr) <= 1e-10,
          @sprintf("rel err %.1e, max residual %.1e", norm(M - Mref) / norm(Mref), maximum(rr)))
end

isempty(failures) || (println("FAILED: ", join(failures, "; ")); exit(1))
println("ALL PASS")
