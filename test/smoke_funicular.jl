## Narval pre-flight 1: the dependency stack loads and the CPU-backend panel path
## of reigen_hermitian agrees with a dense eigen.
##
## Run from the repo root:
##     julia --project=. test/smoke_funicular.jl
##
## No GPU needed. Nothing is written outside a mktempdir, so this is safe to run
## on a login node. Exits nonzero on the first failed check.

using LinearAlgebra
using Random
using Printf

failures = String[]
function check(name, ok, detail="")
    push!(failures, ok ? "" : name)
    @printf("%-58s %s  %s\n", name, ok ? "PASS" : "FAIL", detail)
    return ok
end

println("--- loading PhotonicSystemChannels")
using PhotonicSystemChannels
check("PhotonicSystemChannels loads", true)

println("--- loading Funicular + MatrixFreeRandomizedLinearAlgebra")
using Funicular
using MatrixFreeRandomizedLinearAlgebra
ext = Base.get_extension(MatrixFreeRandomizedLinearAlgebra, :MFRLAFunicularExt)
check("MFRLAFunicularExt is loaded", ext !== nothing, string(ext))

using HDF5
hdf5_ext = Base.get_extension(Funicular, :FunicularHDF5Ext)
check("HDF5 is available (Funicular disk tier)", true, string(pkgversion(HDF5)))
check("FunicularHDF5Ext is loaded", hdf5_ext !== nothing, string(hdf5_ext))

## A minimal matrix-free Hermitian operator: size / eltype / adjoint / mul! only,
## which is the whole panel-path operator contract.
struct HermOp{T}
    A::Matrix{T}
end
Base.size(op::HermOp) = size(op.A)
Base.size(op::HermOp, i::Int) = size(op.A, i)
Base.eltype(::HermOp{T}) where {T} = T
Base.adjoint(op::HermOp) = op
LinearAlgebra.mul!(y::AbstractVector, op::HermOp, x::AbstractVector) = mul!(y, op.A, x)
LinearAlgebra.mul!(Y::AbstractMatrix, op::HermOp, X::AbstractMatrix) = mul!(Y, op.A, X)
Base.:*(op::HermOp, x::AbstractVecOrMat) = op.A * x
Funicular.ishermitian_op(::HermOp) = true

const n = 200
const k = 10
const p = 5

Random.seed!(0xC0FFEE)

## A well-separated synthetic spectrum, which is what makes the 1e-8 bar fair:
## the sketch resolves a decaying spectrum essentially exactly at this size.
function decaying_hermitian(n)
    Q = qr(randn(ComplexF64, n, n)).Q * Matrix{ComplexF64}(I, n, n)
    λ = [3.0 * 0.45^(i - 1) for i in 1:n]
    H = Q * Diagonal(λ) * Q'
    return (H + H') / 2
end

plan = ResidencyPlan(; backend=Funicular.CPUBackend(),
                     device_budget=64 * 2^20,
                     host_budget=64 * 2^20,
                     panel_width=4)
println("--- plan: ", plan)

H = decaying_hermitian(n)
op = HermOp(H)
λ_ref = sort(eigvals(Hermitian(H)); rev=true)[1:k]

println("--- reigen_hermitian on the CPU panel path")
E = reigen_hermitian(op, k; num_oversamples=p, plan=plan, seed=1)
err = maximum(abs.(E.values .- λ_ref))
V = Matrix(E.vectors)
orth = opnorm(V' * V - I)
res = norm(H * V - V * Diagonal(E.values)) / norm(H * V)
println("    vectors type = ", typeof(E.vectors), ", size = ", size(E.vectors),
        ", npanels = ", Funicular.npanels(E.vectors))
check("panel eigenvalues match dense eigen to 1e-8", err < 1e-8,
      @sprintf("max abs err = %.3e", err))
check("panel eigenvectors are orthonormal", orth < 1e-10,
      @sprintf("|V'V - I| = %.3e", orth))
check("panel eigenpairs satisfy H V = V diag(λ)", res < 1e-10,
      @sprintf("rel residual = %.3e", res))

## The factored form is what the positives-only save in workstream B5 uses.
F = reigen_hermitian(op, k; num_oversamples=p, plan=plan, seed=1, factored=true)
check("factored=true gives the same values", F.values ≈ E.values, string(typeof(F.vectors)))
check("factored vectors materialize to the same matrix", Matrix(F.vectors) ≈ V)

## The in-memory path builds its sketch with `similar(operator, T, n)` unless given
## a sample_vec; the panel path never needs one, since Funicular allocates from the
## plan. That is why the wrapper above gets away without `similar`.
λ_mem = reigvals_hermitian(op, k; num_oversamples=p, sample_vec=zeros(ComplexF64, n))
d_mem = maximum(abs.(λ_mem .- E.values))
check("in-memory reigvals matches the panel run", d_mem < 1e-8,
      @sprintf("max abs diff = %.3e", d_mem))

## The disk tier, with the host budget squeezed so the spill actually happens.
## This is the same valve trial E3c uses through $SLURM_TMPDIR.
println("--- disk tier (scratch_dir, squeezed host_budget)")
mktempdir() do dir
    dplan = ResidencyPlan(; backend=Funicular.CPUBackend(),
                          device_budget=64 * 2^20,
                          host_budget=192 * 2^10,
                          panel_width=2,
                          scratch_dir=dir)
    Ed = reigen_hermitian(op, k; num_oversamples=p, plan=dplan, seed=1)
    files = filter(f -> endswith(f, ".h5"), readdir(dir))
    d_disk = maximum(abs.(Ed.values .- E.values))
    check("disk tier produced h5 scratch", !isempty(files), "$(length(files)) file(s)")
    check("disk-tier eigenvalues match the host-tier run", d_disk < 1e-8,
          @sprintf("max abs diff = %.3e", d_disk))

    path = joinpath(dir, "vectors.h5")
    save(Ed.vectors, path)
    reloaded = load(PanelMatrix, path; plan=dplan)
    check("save/load round-trips the panel matrix",
          size(reloaded) == size(Ed.vectors) && Matrix(reloaded) ≈ Matrix(Ed.vectors),
          string(size(reloaded)))
end

println()
bad = filter(!isempty, failures)
if isempty(bad)
    println("ALL CHECKS PASSED ($(length(failures)) checks)")
else
    println("FAILED: ", join(bad, ", "))
    exit(1)
end
