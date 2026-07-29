# Cost-model calibration

`create_jobs.jl` sizes its SLURM requests from a cost model. This directory holds
the model, the harness that measures the numbers it needs, and the fit that turns
those measurements into per-cluster coefficients.

```
cost_model.jl   the model: analytic work counts + fitted primitive costs
plan.jl         generates the run script for one cluster and tier
point.jl        runs one measurement and appends one CSV row
measure.jl      peak RSS / peak VRAM / timing / CSV plumbing
fit.jl          CSVs -> bench/coeffs_<cluster>.jl
data/           put the CSVs you bring back from the clusters here
```

## What to run

Three tiers, in this order:

| tier       | what it measures                                        | when |
|------------|---------------------------------------------------------|------|
| `quick`    | primitives on the four smallest bodies                  | first; identifies every coefficient |
| `full`     | the same primitives on every body, up to 128x32x32       | when you want the fit interpolating rather than extrapolating at your real sizes |
| `validate` | end-to-end `greens -> rsvd -> bounds` chains             | after fitting, to check the model and set the padding factors |

`quick` is roughly 60 short jobs. `validate` is the expensive one: on molering,
where there is no scheduler and points run one at a time, budget days.

## The loop

Generate a launcher:

```bash
julia bench/plan.jl --cluster fir --tier quick
```

That writes `bench/launch_calibration_fir_quick.sh` and a manifest CSV listing
every point with the resources it will ask for, then prints the copy/run commands.
Read the manifest before submitting; it is the cheapest place to notice that a
point is going to ask for something silly.

Copy it over and run it. On the Alliance clusters every point is its own `sbatch`,
so one point running out of memory or time cannot take the rest down:

```bash
scp bench/launch_calibration_fir_quick.sh pvirally@fir.alliancecan.ca:/home/pvirally/Photonic-System-Channels/bench/
```

```bash
ssh pvirally@fir.alliancecan.ca 'cd /home/pvirally/Photonic-System-Channels && bash bench/launch_calibration_fir_quick.sh'
```

On molering there is no scheduler, so run it detached:

```bash
ssh paulv@molering 'cd /home/paulv/Projects/Photonic-System-Channels && nohup bash bench/launch_calibration_molering_quick.sh > calibration.log 2>&1 &'
```

Rows accumulate in `$CAL_ROOT/calibration_<cluster>.csv` as each point finishes,
so a partial run is still usable. When it is done, bring the CSV back:

```bash
scp pvirally@fir.alliancecan.ca:/home/pvirally/scratch/psc-calibration/calibration_fir.csv bench/data/
```

Then fit:

```bash
julia bench/fit.jl
```

This reads every `bench/data/*.csv`, groups by the `cluster` column, and writes
`bench/coeffs_<cluster>.jl`. `create_jobs.jl` loads those automatically; until one
exists for a cluster it warns and uses analytic guesses. Add `--report-only` to
see the fit without writing anything.

## Reading the fit report

Each fitted coefficient prints with the number of points behind it, and each group
prints `measured/predicted  median / min / max`. Two things are worth looking at:

**A coefficient that came out zero.** `W = M log2 M` and `M` are correlated over
the size range sampled, so non-negative least squares will sometimes put all the
weight on one of the pair. That is honest — the data cannot separate them — and the
surviving coefficient still reproduces the measurements. It does mean
extrapolating far past the largest body sampled will drift, which is the argument
for running the `full` tier before submitting big jobs.

**Anything listed under "Not calibrated".** Those kept their defaults. The usual
causes are a tier that skipped the relevant points, or `startup_s` being empty
because `PSC_T0` was not exported (the generated scripts do it; a hand-run point
will not).

The padding factors are set from the 95th percentile of measured/predicted on the
end-to-end runs, so without the `validate` tier they stay at their defaults.

## Running a single point by hand

Useful when a point failed and you want to see why:

```bash
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells 32,32,32 --scale 1//32 --sep 0//1 --rank 800 --gpu -1 --root ~/psc-calibration
```

`--kind` is one of:

| kind | device | measures |
|------|--------|----------|
| `g0_self` | host | one self Green block, `(R, R)` |
| `g0_ext` | host | one external Green block, `(R, S)`; pass `--sep 0//1` for the contact path |
| `g0_multiregion` | host | the 2x2 `[S, R] <- [S, R]` operator, which is where the Green job peaks |
| `stage_greens` | host | the real `_generate_green_sr` |
| `matvec_self` / `matvec_ext` / `matvec_uu` | GPU | one Green matvec |
| `dense` | GPU | QR / gemm / eigh / geigh / svdvals / BLAS-1 at `(m, c)` |
| `bounds_core` | GPU | the bounds kernel on a synthetic spectrum |
| `stage_rsvd` / `stage_bounds` | GPU | the real jobs |

The Green-block points build with `force_generate=true, save_to_disk=false`, so
they measure construction rather than deserialisation and do not depend on what a
previous point left in the preload directory. The `stage_*` and `matvec_*` points
do use the preload directory, which is why the plan runs them after the builds.

`bounds_core` synthesises a plausible `Asym(G⁰ᵤᵣ)` spectrum instead of needing a
real RSVD output, and samples a few iterations of the `O(num_pos²)` outer loop
rather than running all of them. A synthetic `C_basis` is not guaranteed positive
definite, so the generalized eigendecomposition can legitimately refuse; the row
records `outer_ok=0` when that happens and the setup stages are still measured.
`geigh_rate` comes from the `dense` points either way.

## What the model actually claims

Worth knowing before you trust a number:

- **Separation does not affect cost**, except at contact. No operator is ever
  built on the bounding box that contains the gap: the "universe" in this pipeline
  is the concatenated `[sender; receiver]` vector and a four-block multi-region
  operator. A 10000-wavelength separation costs the same as a 1-wavelength one.
- **The Green function job builds six blocks**, three self and three external, and
  peaks during the multi-region build with three finished blocks still resident.
  External blocks retain eight times as much Fourier data as self blocks.
- **The RSVD job does `c(q+2)` applications of `Asym(G⁰ᵤᵣ)`** — each two external
  plus one self matvec — **and `c(2q+2)` external matvecs** for the `RS/` singular
  values. The dense QRs are not a rounding error at `c ~ 3000`.
- **The bounds job is not constant-time.** It runs one dense `k x k` generalized
  eigendecomposition per positive eigenvalue plus an `O(num_pos²)` inner loop, so
  it grows faster in rank than the RSVD does.
- **`num_pos` is guessed.** The bounds cost depends on how many `Asym(G⁰ᵤᵣ)`
  eigenvalues come back positive, which is only known after the RSVD has run.
  `create_jobs.jl` assumes 60% of the rank; the existing outputs in
  `data analysis/data` run 22-52%.
- **MIG slices are approximated.** The calibration runs on a whole GPU, and a
  slice's time is scaled by the inverse of its SM fraction. Memory bandwidth does
  not scale with SM count, so this over-estimates — in the safe direction.
