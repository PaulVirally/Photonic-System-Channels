# A low-memory preconditioner for Gila's `A = I - χG₀` on high-index dielectric bodies several
# material wavelengths across, such as two cuboids at χ = 18.55 + 1.54i separated by vacuum.
#
# M is the same problem at the complex frequency f(1 + 0.1i), which damps the resonances that
# make A hard, solved loosely (to 0.1) by an inner GMRES. That inner solve is preconditioned by
# Chan's optimal circulant of `I - χ̄G₀` built at f(1 + i), inverted by the 3D FFT, with χ̄ the
# mean of χ over its support plus 3i and the identity off the support. The plain circulant at the
# real frequency is near singular on the bodies' resonant modes and stalls the solve.
#
# On two 1-wavelength cubes at 64 cells per wavelength (A6000, Float64, to 1e-8) this takes 830
# to 900 operator products and 20 to 24 s, against 4250 to 4650 and 66 to 81 s unpreconditioned.
# It holds about 1.5 kB per cell of the box, two thirds of it the inner GMRES basis.
#
#     opr = InvSctOpr{Float64}(GlaVol((24, 8, 8), (1//32, 1//32, 1//32)), χ)   # χ: (24, 8, 8)
#     M = ShiftedSolve(opr, χ)
#     x = solve(opr, b, GMRESSolver(20, 5000, 0.0, 1e-8; preCon=M, flexible=true))
#
# M is an inner iteration, so the outer solve must be flexible GMRES; GCRO-DR refuses it.

using FFTW, StaticArrays

# Gila stores the symmetric 3×3 kernel as xx, yy, zz, xy, xz, yz; `SLOT[b, a]` is where entry
# (b, a) lives. `PARITY[slot]` is that component's sign under reflection in x, y and z: odd in
# exactly the axes its two indices name (Groth, Polimeridis and White, arXiv:1903.09802, Table 1).
const SLOT = [1 4 5; 4 2 6; 5 6 3]
const PARITY = ((1, 1, 1), (1, 1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))

# The Green kernel over non-negative displacements, (nx, ny, nz, 6): the field of a unit source
# in the corner cell, once per component.
function greenkernel(vac)
    cel = GilaElectromagnetics.glaSze(vac, 2)[1:3]
    src, fld, g = similar(vac, (cel..., 3)), similar(vac, (cel..., 3)), similar(vac, (cel..., 6))
    for a in 1:3
        fill!(src, 0)
        src[1:1, 1:1, 1:1, a] .= 1   # broadcast: a GPU array refuses a scalar store
        mul!(fld, vac, src)
        for b in 1:a
            g[:, :, :, SLOT[b, a]] .= @view fld[:, :, :, b]
        end
    end
    return g
end

# The circulant nearest the Toeplitz matrix of kernel `c` in Frobenius norm, level by level:
# `c_i = ((l-i)/l)·t_i + (i/l)·t_{i-l}`, with `t_{i-l}` the kernel at `l-i` signed by `PARITY`.
function chan(c)
    for (ax, l) in enumerate(size(c)[1:3])
        w = copyto!(similar(c, Float64, ntuple(d -> d == ax ? l : 1, 4)), (l:-1:1) ./ l)
        sgn = copyto!(similar(c, Float64, (1, 1, 1, 6)), [p[ax] for p in PARITY])
        refl = circshift(reverse(c; dims=ax), ntuple(d -> d == ax ? 1 : 0, 4))   # i ↦ l - i
        c = w .* c .+ (1 .- w) .* sgn .* refl
    end
    return c
end

# `(I - χ̄C)⁻¹` per Fourier mode, C the circulant of `vac`'s kernel; the identity off `support`.
struct Circulant{S,V,F,B,P,Q}
    inv::S
    modes::V
    fld::F
    support::B
    fwd::P
    bwd::Q
end

function Circulant(vac, χ̄, support)
    ĉ = fft(chan(greenkernel(vac)), 1:3)
    blocks = SMatrix{3,3,ComplexF64,9}.(ntuple(k -> @view(ĉ[:, :, :, SLOT[k]]), 9)...)
    fld = similar(ĉ, size(ĉ)[1:3]..., 3)
    return Circulant((s -> inv(I - χ̄ * s)).(blocks), similar(ĉ, SVector{3,ComplexF64}, size(ĉ)[1:3]),
                     fld, support, plan_fft!(fld, 1:3), plan_ifft!(fld, 1:3))
end

function LinearAlgebra.ldiv!(out, M::Circulant, inp)
    copyto!(vec(M.fld), inp)
    M.fwd * M.fld
    f = ntuple(c -> @view(M.fld[:, :, :, c]), 3)
    M.modes .= M.inv .* SVector{3,ComplexF64}.(f...)
    for c in 1:3
        f[c] .= getindex.(M.modes, c)
    end
    M.bwd * M.fld
    out .= ifelse.(M.support, vec(M.fld), inp)
    return out
end

struct ShiftedSolve{O,S}
    opr::O
    solver::S
end

"""
    ShiftedSolve(opr, χ; shift=0.1, δ=1.0, tol=0.1)

`opr` is the problem's `InvSctOpr`, `χ` its susceptibility on the grid.
"""
function ShiftedSolve(opr, χ::AbstractArray; shift=0.1, δ=1.0, tol=0.1)   # typed: an untyped form would overwrite the default constructor
    vac = GilaElectromagnetics.vac(opr)
    T, vol, (; dev, frqPhz) = real(eltype(sus(opr).sus)), vac.mem.srcVol, vac.mem.cmpInf
    shifted = InvSctOpr{T}(GlaOprVac{T}(vol; dev, frqPhz=frqPhz * (1 + im * shift)), χ)
    χ̄ = sum(χ) / count(!iszero, χ) + 3im
    C = Circulant(GlaOprVac{T}(vol; dev, frqPhz=real(frqPhz) * (1 + im * δ)), χ̄, sus(opr).sus .!= 0)
    inner = GMRESSolver(20, 500, 0.0, tol; preCon=C, side=:right)
    return ShiftedSolve{typeof(shifted),typeof(inner)}(shifted, inner)
end

LinearAlgebra.ldiv!(out, M::ShiftedSolve, inp) = copyto!(out, solve(M.opr, inp, M.solver))
GilaElectromagnetics.isvarying(::ShiftedSolve) = true
