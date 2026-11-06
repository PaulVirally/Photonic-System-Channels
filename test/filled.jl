## The filled-block bound (Fig 2) on a coarse CPU system: the waveguide with ℓ = λ/4 at λ/8.
## `filled_bound` with every ε set to zero must reproduce the exact dual minimum min_λ g₁(λ),
## computed from dense blocks of G⁰ (filled_blocks_qcqp.pdf Eqs. 20–28), and the bound with
## the tails must lie above it. This also checks the receiver-alone P⁽⁰⁾ (mediator_bound.pdf §8).
##
##     julia --project=. test/filled.jl

using PhotonicSystemChannels
const PSC = PhotonicSystemChannels
using LinearAlgebra, Printf

const χ = 18.612 + 1.5502660841406073im
const ζ = abs2(χ) / imag(χ)
const κ = 2imag(χ) / χ
const SCALE = 1 // 8

failures = String[]
function check(name, ok, detail="")
    ok || push!(failures, name)
    @printf("%-62s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
end

# Dense reference: G⁰ on s ∪ m ∪ r (columns of the box's vacuum operator), then Route 1 of
# mediator_bound.pdf at n = 1.
sc, gap, mc = filled_geometry("wg", 1 // 4, SCALE)
fb = PSC.FilledBox(sc, gap, mc, gap, sc, SCALE, χ; on_gpu=false)
Gv = PSC.vacuum(fb)
idx = vcat(fb.s, fb.m, fb.r)
G = Matrix{ComplexF64}(undef, length(idx), length(idx))
for (k, i) in enumerate(idx)
    e = zeros(ComplexF64, PSC.nbox(fb))
    e[i] = 1
    G[:, k] = (Gv * e)[idx]
end
Ns, Nm = length(fb.s), length(fb.m)
println("dense G⁰ built: $(size(G, 1)) × $(size(G, 2))"); flush(stdout)
IS, IM, IR = 1:Ns, Ns+1:Ns+Nm, Ns+Nm+1:size(G, 1)
IB = vcat(IS, IR)
W = inv(I - χ * G[IB, IB])
Js, Jr = 1:Ns, Ns+1:length(IB)
Pe = κ * W[Jr, Js]
T = κ * W[Jr, :] * (χ * G[IB, IM])
Fm = G[IM, IB] * W[:, Js]
Gbg = G[IM, IM] + G[IM, IB] * W * (χ * G[IB, IM])
Πbg = Hermitian(I / ζ + (Gbg - Gbg') / 2im)
function g1(λ)
    K = Hermitian(λ * Πbg - T' * T)
    isposdef(K) || return Inf
    C = T' * Pe - (λ / 2im) * Fm
    return eigmax(Hermitian(Pe' * Pe + C' * (K \ C)))
end
λ0 = eigmax(Hermitian(Matrix(Πbg)^(-1 / 2) * (T' * T) * Matrix(Πbg)^(-1 / 2)))
println("dense reference built"); flush(stdout)
dense, _ = golden_min(t -> g1(λ0 * (1 + exp(t))), -14.0, 14.0; iters=80)

r = filled_bound("wg", 1 // 4, SCALE; χ, k_mb=368, k_rs=128, k_a=64, nmax=2, on_gpu=false, log=msg -> (println("  ", msg); flush(stdout)))
b = r.bounds[1]
check("ε = 0 value equals the dense dual minimum", abs(b.noeps - dense) < 1e-4 * dense, @sprintf("%.8f vs %.8f", b.noeps, dense))
check("bound with tails lies above it, within 0.05", b.noeps <= b.bound < b.noeps + 0.05, @sprintf("%.6f", b.bound))
check("joint rank removal is a bound too", b.joint >= b.joint_noeps && isfinite(b.joint), @sprintf("%.6f", b.joint))
# the §8 factorization is rank-truncated: |σ₁ − σ₁(P⁽⁰⁾ₖ)| ≤ ε₀, the tail the bound relies on
σe = opnorm(Pe)
check("σ₁²(P⁽⁰⁾) from §8 matches dense within the ε₀ tail", abs(b.empty - σe^2) <= r.ε0 * (2σe + r.ε0),
      @sprintf("%.8f vs %.8f (ε₀ %.1e)", b.empty, σe^2, r.ε0))
check("box solves reached the residual target", r.maxres <= 1e-11, @sprintf("%.1e", r.maxres))

isempty(failures) || error("failed: $(join(failures, ", "))")
