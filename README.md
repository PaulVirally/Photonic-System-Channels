# Photonic System Channels

Upper bounds on the singular values σₙ of the sender → receiver transfer operator of a
photonic design, computed by a projected Lagrange dual over the power-conservation
constraints of the design region. Green functions come from
[GilaElectromagnetics.jl](https://github.com/moleskySean/GilaElectromagnetics.jl).

## Pipeline

Each point (a sender, a receiver, their separation, a cell size and a susceptibility χ)
runs three stages, each a script taking the same command line:

| Stage | Script | Writes |
|---|---|---|
| Green operators | `generate_green.jl` | G⁰ᵣₛ, G⁰ᵣᵣ, G⁰ᵤᵤ, Asym(G⁰ᵣᵣ), Asym(G⁰ᵤᵤ), cached in `--preload` |
| Spectra | `generate_rsvd.jl` | Asym(G⁰ᵤᵣ) eigenpairs and G⁰ᵣₛ singular values in `--scratch` |
| Bounds | `compute_bounds.jl` | `true_bounds` and friends in `--project` |

Every stage skips work whose output already exists, so rerunning a point only redoes
what is missing. The directories default to `./data/{preload,project,scratch}`, or to
the group's shared ones when `MOLERING` is set (where `--gpu` also defaults to device 0).

```bash
ARGS="--sender (8,8,8) --receiver (8,8,8) --rs-sep (1//4,0//1,0//1) --scale 1//32 \
      --chi 18.612+1.5502660841406073im --design rs --components 4000 --name example"
julia --project=. generate_green.jl  $ARGS --gpu false
julia --project=. generate_rsvd.jl   $ARGS --gpu 0
julia --project=. compute_bounds.jl  $ARGS --gpu 0
```

`--scale (1//32,1//16,1//16)` gives anisotropic cells. A gap under six cells is meshed
with refined slabs at the facing surfaces unless `--no-refine` is passed. `--k-uu`
(default 512) is the number of Asym(G⁰ᵤᵤ) eigenvectors added to the projection basis,
and `--gamma-rtol` (default 1e-12) cuts the RSVD noise floor off the Asym(G⁰ᵤᵣ)
spectrum (both signs: the bound uses the negative-Γ eigenvectors too).

A system with a mediator (`--mediator`, `--sm-sep`, `--mr-sep`, `--design m`) bounds
σₙ(Gᵣₛ) over every structuring of the mediator (Eqs. (31)–(33)): the RSVD stage takes
G⁰ᵣ←[s;m] and G⁰ₘₛ with their vectors and a residual bound each (`RU/`, `MS/`, h5 stores
beside the JLD) and the vacuum σₙ(G⁰ᵣₛ) (`RS/D`); the bounds stage solves
B = ζ⁻¹/2 + Asym(G⁰ₘₘ) on their mediator blocks by block PCG, deflated by `--k-uu`
eigenpairs of Asym(G⁰ₘₘ), then minimises the dual for
`--channels` indices. The RSVD residuals enter the dual itself (half the material loss
absorbs them), so the result is a bound and not an estimate.

The RSVD stage stores every Asym(G⁰ᵤᵣ) eigenvector it computes, not only the
positive-Γ ones, so its scratch output is about twice what a positive-only run needed
(tens of GB per point at 4 λ with 4000 components). Outputs of older runs hold only the
positive block; `generate_rsvd.jl` recomputes them. The bounds stage checkpoints to
`<project>/<prefix>.jld.ckpt.*` as it goes, so a killed run resumes where it stopped.

## Output

`<project>/<prefix>.jld` holds, per channel n, `true_bounds[n]`: the minimum of the
system-design dual of Eq. (37) (`bounds_dual_basis`, with its minimisers
`opt_taus` and `opt_mu_p`) and the two passivity bounds
(`old_analytical_bounds`, `new_analytical_bounds`), with `which_bounds` naming the
winner (1, 2, 3). The dual runs on the channels in `dual_channels` (all up to 50, then a
geometric ladder); the others take the bound of the sampled channel below them, which is
valid since σₙ ≤ σₘ for m < n. Φ = Σₙ σₙ² is `sum(abs2, true_bounds)`. The `augment/` group records
the basis the dual was solved in.

## Code

- `src/SMRSystems.jl`, `src/refinement.jl`: geometry, gap refinement, Green operator cache.
- `src/rsvd.jl`: Asym(G⁰ᵤᵣ) and the RSVDs, dense-exact, in memory, or in Funicular panels when the sketch outgrows the GPU.
- `src/bounds.jl`: the augmented basis and the Eq. (37) dual, minimised over μₚ and τ.
- `src/mediator.jl`: the mediator design bound, Eqs. (31)–(33), and its B⁻¹ solve.
- `src/common.jl`: command line, device selection, memory budgets, `golden_min`.
- `src/filled.jl`, `src/preconditioner.jl`: the filled-block box solves and the reduced dual (Fig 2).
- `src/elimination.jl`: the Fig 2 pipeline, `filled_bound`.

## Sweeps

A sweep is a loop over points. For example, resonant germanium (ζ = |χ|²/Im χ = 225),
λ/4 cubes, gaps in units of the 1/32 λ cell:

```bash
for g in 1 2 4 8 16 32 64 128 256 512 1024; do
  ARGS="--sender (8,8,8) --receiver (8,8,8) --rs-sep ($g//32,0//1,0//1) --scale 1//32 \
        --chi 18.612+1.5502660841406073im --design rs --components 4000 --oversamples 50 \
        --power-iterations 6 --name Ge225_0p25_${g}ss32"
  julia --project=. --threads=16 generate_green.jl  $ARGS --gpu false
  julia --project=. --threads=16 generate_rsvd.jl   $ARGS --gpu 0
  julia --project=. --threads=16 compute_bounds.jl  $ARGS --gpu 0
done
```

To use both GPUs, split the points between two such loops, one with `--gpu 0` and one
with `--gpu 1`. Points share the Green cache safely (writes are atomic).

## Tests

Standalone scripts, CPU only: `julia --project=. test/<name>.jl`, or all of them with
`julia --project=. test/runtests.jl`.

## Fig 2: filled-block bound

Sender and receiver are solid blocks of germanium (χ = 18.612 + 1.550i); the mediator between
them is designed. `filled_bound` bounds σₙ²(Pᵣₛ) over every structuring of the mediator, for
n = 1 … 40, by exact elimination of the filled blocks with a low-rank vacuum basis on the
mediator, the truncation errors (ε₀, ε_F, ε_x) carried inside the relaxation:

```bash
julia --project=. --threads=16 filled_bound.jl wg 1//2 --k-mb 3072 --k-rs 384 --out results/wg_1ss2.jld2
julia --project=. --threads=16 filled_bound.jl ms1 1//4 --k-mb 1024 --k-rs 256 --out results/ms1_1ss4.jld2 --gpu 1
```

Geometries (`filled_geometry`, lengths in wavelengths, cells of `--scale`, default λ/32):

| family | sender, receiver | mediator | gaps | param |
|---|---|---|---|---|
| `wg` | 1/4 × 1 × 1 | ℓ × 1 × 1 | 1/8 | ℓ |
| `wgt` | 1/4 × 1/4 × 1/4 | ℓ × 1/4 × 1/4 | 1/8 | ℓ |
| `ms05`, `ms1`, `ms2`, `ms4` | 1 × h × h (h = 1/2, 1, 2, 4) | 1 × h × h | d | d |
| `ms44` | 1 × 4 × 4 | 4 × 4 × 4 | d | d |

`--k-mb` is the rank kept of the vacuum coupling G_mb (mediator ← blocks) and `--k-rs` that of
G_rs (receiver ← sender); both truncations are paid for by the ε's, so they trade run time
against tightness, never validity. The output holds, per n, the bound with rank removal on the
empty-mediator transfer P⁽⁰⁾ (`bound`) and on [P⁽⁰⁾ Γ] (`joint`), each also with every ε set to
zero (`noeps`, `joint_noeps`; not bounds), and σₙ²(P⁽⁰⁾) (`empty`). The ε's are Lanczos
estimates that hold except with probability 10⁻¹⁰ each (Kuczyński–Woźniakowski).

The cost is k_mb + 2 k_rs box solves plus about 120 for the ε's, each a flexible GMRES on the
box enclosing all three bodies with the shifted-frequency circulant preconditioner, to a true
relative residual of 10⁻¹¹. `study/filled_dense.jl` compares the bound with dense QCQP methods
on coarse grids.
