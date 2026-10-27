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
what is missing.

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
spectrum. A system with a mediator (`--mediator`, `--sm-sep`, `--mr-sep`) runs the
first two stages; its bounds are not implemented yet.

## Output

`<project>/<prefix>.jld` holds, per channel n, `true_bounds[n]`: the minimum of the
projected dual (`bounds_dual_basis`) and the two passivity bounds
(`old_analytical_bounds`, `new_analytical_bounds`), with `which_bounds` naming the
winner (1, 2, 3). Φ = Σₙ σₙ² is `sum(abs2, true_bounds)`. The `augment/` group records
the basis the dual was solved in.

## Code

- `src/SMRSystems.jl`, `src/refinement.jl`: geometry, gap refinement, Green operator cache.
- `src/rsvd.jl`: Asym(G⁰ᵤᵣ) and the RSVDs, dense-exact, in memory, or in Funicular panels when the sketch outgrows the GPU.
- `src/bounds.jl`: the augmented basis, the τ family of constraints and the dual.
- `src/common.jl`: command line, device selection, memory budgets.

## Sweeps

`create_jobs.jl` writes a submit script for a sweep (edit the block at its top):
on Narval a chain of three Slurm jobs per point, on molering a sequential run.

## Tests

Standalone scripts, CPU only: `julia --project=. test/<name>.jl`. Run them in a job
on a cluster, not on a login node.
