# Cost-model calibration

`create_jobs.jl` sizes its SLURM requests from a cost model. This directory holds
the model, the harness that measures the numbers it needs, and the fit that turns
those measurements into per-cluster coefficients.

```
cost_model.jl      the model: analytic work counts + fitted primitive costs
plan.jl            generates the run script for one cluster and tier
point.jl           runs one measurement and appends one CSV row
measure.jl         peak RSS / peak VRAM / timing / CSV plumbing
fit.jl             CSVs -> bench/coeffs_<cluster>.jl
derive_coeffs.jl   conservative stopgap coefficients for an uncalibrated cluster
compare_parity.jl  compares two RSVD runs' spectra (trial E2)
data/              put the CSVs you bring back from the clusters here
```

## Clusters

Per-GPU *bundle* below is the Alliance's published ratio: usage is billed as the
largest of `gpus`, `cores / bundle_cores` and `host / bundle_RAM`, so a job that
stays inside the bundle costs exactly one GPU-equivalent.

| cluster | GPU | bundle/GPU | MIG slices | calibrated |
|---------|-----|-----------|------------|------------|
| fir | 4x H100 SXM5 80GB | 12 cores, 288 GB | `nvidia_h100_80gb_hbm3_{1g.10gb,2g.20gb,3g.40gb}` | yes, incl. memory |
| narval | 4x A100 SXM4 40GB | 12 cores, 124 GB | `a100_{1g.5gb,2g.10gb,3g.20gb}` | yes, time only |
| nibi | 8x H100 SXM 80GB | 14 cores, 250 GB | `h100_{1g.10gb,2g.20gb,3g.40gb}` | no — derived from fir |
| rorqual | 4x H100 SXM5 80GB | 16 cores, 124 GB | `h100_{1g.10gb,2g.20gb,3g.40gb}` | no — derived from fir |
| molering | RTX A6000 48GB | 16 useful threads of 128 | — | yes, time only |

Note that fir spells the same three H100 partitions differently from nibi and
rorqual. A slice name the cluster does not define is a hard `sbatch` rejection
rather than a bad estimate, so if a submission is refused, check what the cluster
actually offers — and confirm the `module load` line while you are there, since
available Julia and CUDA versions differ between clusters:

```bash
sinfo -o "%G" | sort -u
```

`create_jobs.jl` picks the smallest slice that fits both the predicted VRAM *and*
the predicted host RAM, since a slice comes with a proportionally smaller RAM
bundle and exceeding it is billed as though several slices had been taken. Its
plan summary lists every slice name the plan is about to request. Only about half
of fir's GPU nodes carry MIG at all, which costs queue breadth but not
correctness.

Calibration itself always takes a whole GPU — the point is to measure the
primitives on undivided hardware — and the model then derates for a slice by its
SM fraction.

A cluster with no calibration uses `bench/coeffs_<name>.jl` written by
`derive_coeffs.jl`: fir's measurements derated to err towards over-estimating
(CPU times x1.5, GPU times x1.25, rates /1.25, memory unchanged since it is the
same code on the same 80 GB card). `create_jobs.jl` says so in its plan summary.
Replace them by calibrating for real — `--tier quick` then `--tier memory` — after
which `fit.jl` overwrites the derived file.

## What to run

Three tiers, in this order:

| tier       | what it measures                                        | when |
|------------|---------------------------------------------------------|------|
| `quick`    | primitives on the four smallest bodies                  | first; identifies every coefficient |
| `full`     | the same primitives on every body, up to 128x32x32       | when you want the fit interpolating rather than extrapolating at your real sizes |
| `validate` | end-to-end `greens -> rsvd -> bounds` chains             | after fitting, to check the model and set the padding factors |
| `funicular` | the Funicular panel path on narval (workstream E)      | before the k=4000 sweep; see its own section below |

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

**Padding.** The time padding comes from the 95th percentile of
measured/predicted on the end-to-end runs when there are any; without them it
falls back to 1.25x the worst primitive-level miss, floored at 1.5. Memory padding
is deliberately *not* derived from timing scatter — memory is not noisy the way
wall time is, so its margin lives in the `*_mem_factor` multiplier on the analytic
count instead. Stacking all three (analytic floor x factor x timing pad) is how a
26 GB rSVD once turned into a 96 GB request.

**Collecting rows.** Each point writes its own CSV under `$CAL_ROOT/rows`, because
concurrent appends to one file on a shared filesystem tear lines in half (two
narval rows were lost that way). Merge them before copying back:

```bash
bash bench/launch_calibration_narval_quick.sh --merge
```

## The `funicular` tier

A fifth tier, narval-only, and not calibration in the sense the other four are.
Those measure primitives on undivided hardware and let `create_jobs.jl` derate for
a slice. These are workstream E of `FUNICULAR_PLAN.md`: eleven GPU trials asking
whether particular allocations hold particular jobs on the Funicular panel path, so
each point names its own allocation rather than taking a whole card.

```bash
julia bench/plan.jl --cluster narval --tier funicular            # write both files
julia bench/plan.jl --cluster narval --tier funicular --dry-run  # cost it, write nothing
```

Every trial is one separation (`16//32`, which Julia normalises to `1//2`),
`chi = 13.6+0.05im`, scale `1//32`, `p = 50`, `q = 14`, and one shared seed.

| point | trial | allocation | what it pins down |
|-------|-------|-----------|-------------------|
| `e1_panelbus` | E1 | whole a100 | `pcie_rate`, `overlap_factor` |
| `fungreens_l{1,2,4}` | (none) | cpu | the Green blocks everything else depends on |
| `e2_l1_{inmem,panel}` | E2 | whole a100 | eigenvalue parity, wall-clock ratio, host/VRAM high-water, `panel_workspace_bytes`, the host overhead reserve |
| `e3d_l1_k4000` | (added) | a100_3g.20gb | E4's 1 lambda input, and the 1 lambda production allocation |
| `e3a_l2_{slice,full}` | E3a | a100_3g.20gb, a100 | whether the 62 GB bundle holds 51 GB + overhead |
| `e3b_l4_full` | E3b | whole a100 | the 102 GB peak against the 124.5 GB bundle; measured positive fraction |
| `e3c_l4_spill` | E3c | whole a100 + NVMe | the spill path end to end |
| `e4_bounds_l{1,4}_k4000`, `e4_bounds_l1_k1350` | E4 | 3g.20gb / a100 / 3g.20gb | panelized front-end correctness and the `m^4` pencil constant |

`e3d_l1_k4000` is not in the plan's table. E4 wants bounds at k = 4000 for 1 lambda
and the table's E3 rows only produce 2 and 4 lambda outputs, so the 1 lambda
k = 4000 RSVD has to exist somewhere.

The tier does three things the other four do not.

It forces the path. `bench/point.jl --force-path panel` builds a
`Funicular.ResidencyPlan` and hands it to `_save_ur_asym` / `_run_rsvdvals` as
`plan_override`, which is the one hook that bypasses `use_panel_path`. Trial E2
needs it: the parity check compares the two storage paths, and at 1 lambda,
k = 1350 the predicate would never choose panel on a 40 GB card. Running the panel
half on a 3g.20gb slice instead, where the predicate does flip on its own, would
confound the storage path with three eighths of the SMs, and the wall-clock ratio
would mean nothing.

It forces the host budget through Slurm rather than through a flag. Trial E3c asks
for `--mem=66G` on purpose: `residency_plan` reads `SLURM_MEM_PER_NODE` and
subtracts a 6 GiB overhead reserve, so that request is a 60 GiB host budget
exactly, against a ~95 GiB panel peak at 4 lambda. There is no environment
override in `src/`, and adding one would have meant E3c exercising a testing hook
instead of the code the production sweep will run. `point.jl` does carry a
`--host-budget-GB` override for debugging; the tier leaves it unset.

Its manifest carries four extra columns. The four calibration tiers keep

    label,kind,threads,host_GB,time_limit_s,gpu,args

with every geometry, rank and separation inside the quoted `args`. `funicular`
appends `gpu_request`, `depends_on`, `predicted_wall_s` and `predicted_gpu_h`
before `args`, because three of its facts have nowhere else to go: which allocation
the point names (0/1 cannot say it when the allocation is the variable under test),
which point must finish first (the Green functions, and the E4 chain), and what the
model predicts, which is the number the trial is judged against. The extension is
additive and only on this tier.

The tier also writes its rows to `$CAL_ROOT/rows_funicular` and merges them into
`calibration_narval_funicular.csv` rather than sharing `rows/` with the other
tiers. Otherwise `--merge` would sweep up an earlier quick run's rows and the fit
would count them twice.

### Trial E1 and the `panel_bus` rows

`fit.jl`'s `fit_panel_bus` reads rows with `kind="panel_bus"`, taking `pcie_rate`
from rows carrying `bytes=<moved>` in `extra` (paired with the row's `time_s`) and
`overlap_factor` from rows carrying `overlap=<fraction>`. The point produces them
from three sources:

- an in-process pinned host-to-device sweep at four transfer sizes, through
  Funicular's own `alloc_host_slab` / `h2d!` / `sync_queue`. Four sizes rather than
  one, because `rate_through_origin` fits a slope through the origin and a single
  point cannot tell a slope from a fixed per-transfer overhead;
- Funicular's `benchmark/pinned.jl`. Its pageable comparison is recorded under
  `bytes_pageable=` so it stays in the CSV without being fitted, since a pageable
  copy is not the rate the panel path pays;
- Funicular's `benchmark/overlap.jl`, one row per compute-to-copy ratio.
  `overlap = (pipeline - compute) / copy`, recorded only where compute dominates:
  below a ratio of one the sweep is bus-bound and that expression is most of the
  copy however well the schedule overlaps. A ratio whose copies vanish entirely is
  recorded at a floor of 0.05 with the raw value kept under `overlap_raw=`, because
  `fit_panel_bus` rejects a zero and a zero would delete the term from the model.

The scripts run as their own processes against the main project environment.
Funicular's `benchmark/Project.toml` does not need instantiating: it lists
DelimitedFiles and Plots, both for `benchmark/plot.jl` only, while `pinned.jl` and
`overlap.jl` need CUDA, Funicular and Printf, which this project already has. The
point copies the directory under its `--scratch` first, because
`benchmark/common.jl` writes its TSV results next to itself and the depot is not
ours to write into. It locates the source through `pkgdir(Funicular)`, so a
re-pinned commit needs no edit here.

### Trial E2 and the parity comparison

```bash
julia --project=. bench/compare_parity.jl \
    --a $CAL_ROOT/funicular/e2_l1_inmem/<prefix>.jld \
    --b $CAL_ROOT/funicular/e2_l1_panel/<prefix>.jld \
    --label-a in-memory --label-b panel --rtol 1e-6
```

This is not an equality check, and it must not become one. The in-memory path takes
its Gaussian sketch from Julia's global RNG, while the panel path has nowhere to
keep one and regenerates blocks from an integer seed. Two panel runs at the same
seed sketch identically; a panel run and an in-memory run never do, whatever seed
either is given. It follows that the two spectra agree only to the accuracy of the
randomized method, and the top of the spectrum is the only part where that accuracy
is many digits. The tail is where the sketch's random subspace has not converged,
and two different random subspaces have no reason to agree there.

The verdict is therefore taken over the leading `--top-fraction` (default 10%) of
the positive spectrum at `--rtol`, and everything below is reported rather than
judged; `--strict` judges the whole positive block. Deviations come out two ways:
per-element (`|a-b| / max(|a|,|b|)`, how many digits this eigenvalue agrees to) and
scaled by the largest eigenvalue (whether the disagreement matters downstream). A
`num_pos` difference between the two runs is called out separately, because it
sizes every object the bounds job then builds. The script reads only JLD2, so it
runs on a login node.

### Budget

`plan.jl` prints a per-point cost table for this tier. As planned it comes to 31.3
GPU-equivalent hours predicted, 72 at the limits. The limits are loose on purpose,
since a trial killed mid-measurement wastes the whole trial. Two caveats the table
prints for itself: E1 has no cost-model prediction (no `SRPoint` describes a bus
benchmark) so it contributes zero to the predicted total, and E3c's prediction
omits the NVMe round trip the trial exists to measure.

The three E4 points are 19.6 of those 31.3, because they are sized at
`NUM_POS_FRACTION = 0.6` and bounds time grows as `m^4`. The existing outputs run
0.22-0.52; at 0.5 the tier comes to 21.2 GPU-hours. The pessimistic figure is kept
for the *limits*, since a bounds job killed at 24 h is a wasted day, and the
sensitivity is printed so the budget is not read off the wrong number.

The two k=4000 bounds points sit on the 24 h ceiling rather than the 2.5x margin
the others get; `plan.jl` says so when it plans them.

### Ordering

E1 to E3 need only workstreams A and B. The three E4 points need workstream C, the
panelized bounds front-end: they are written against the CLI that exists today, but
at k = 4000 the old in-memory front-end would want the `N_u x m` basis as one
`CuArray`, which is ~30 GB at 4 lambda. Submit them only once C has landed. The
generated script says so, and commenting out the E4 block leaves the rest runnable.

## The `backfill` tier

    julia bench/plan.jl --cluster narval --tier backfill

Eleven jobs, none of them asking for more than three hours, whose only purpose is to
refit the two counts that the `--gamma-rtol` truncation and the windowed tau search
changed. A 1 lambda bounds job is currently requested at 18 h; nothing that asks for
18 h gets through narval's backfill window, so the sweep does not run at all.

What the tier measures, and why each piece is cheap:

| | jobs | asks for | measures |
|---|---|---|---|
| A | 4 x `stage_bounds` | <= 02:30:00, `a100` or `a100_3g.20gb` | the tau shape, the truncated `m`, the per-index outer cost, host and device high-water |
| B | 6 x `stage_rsvd` | <= 01:05:00 | the per-operator-pass rate at 1/2 and 1 lambda, at `q = 1, 3, 5` |
| | 1 x `stage_greens` | <= 01:03:00, CPU | the 1/2 lambda blocks, which no sweep has built yet |

Three ideas make it fit in three hours.

**The bounds points sample the outer loop.** `--outer-blocks 4 --outer-block-len 24`
runs four runs of 24 consecutive indices spread over `1:m` instead of all `m`. The
front end is measured in full either way, `outer_times` reports each index
separately, and the sample is a few percent of the loop. Consecutive within a block
because the windowed sweep only narrows for an `n` that follows the last index
evaluated; spread between blocks because index `n` probes `m - n + 1` vectors, so a
sample taken only at the top would put the per-index cost at twice its average. A
pick whose `m` is small enough runs `--outer-blocks 0`, the whole loop, production
exactly.

**The RSVD points run at low `q`.** RSVD cost is affine in the power iteration count,
so two low-`q` runs give the per-pass slope and the third checks the line is straight.
`rsvd_time_parts` in `cost_model.jl` splits the prediction the same way, and
`rsvd_pass_scale` is the ratio between them.

**The bounds points reuse RSVD output already on scratch.** The cancelled 1 lambda
sweep left finished bases behind, so there is no production-rank RSVD to pay for.
Which separations survived is not knowable when the script is generated, so
`bench/pick_bounds_points.jl` runs on the login node first: it lists what is there,
reads each spectrum, applies the same `--gamma-rtol` cut `load_bounds_inputs`
applies, and picks four spread over the range of surviving `m`. It sizes each job
from its own `m` -- allocation, memory, time limit and outer-loop mode -- and writes
the whole kept-count table alongside, which is the truncation measurement in its own
right. Look before submitting:

```bash
bash bench/launch_calibration_narval_backfill.sh --pick
```

### `--design rs`

Every point in this tier passes it, and it is load-bearing. `src/common.jl` sorts the
letters of `--design`, so production sweeps write
`<cells>__<cells>__<n>ss<d>__RS`, while `bench/point.jl` historically built
`[Sender, Receiver]` and looked for `__SR`. Same geometry, different filename; a
bounds point that spells it the old way finds none of the outputs it came to read.
`sr` remains the default so the other tiers keep reading their own scratch.

### When a bounds job is cut short anyway

The three-hour cap can still land on a job whose `m` was larger than the picker's
spectrum read suggested. The process is killed before it can write its row, but the
log holds everything: `bounds_from_spectrum` stamps a timestamp into every message,
so per-index times are differences between consecutive "Computing" lines, the
truncation warning carries the kept and stored counts, and each index logs which grid
points it swept. `bench/measure.jl` reads it back:

```bash
julia --project=. bench/measure.jl --parse-bounds-log <log> --summary
julia --project=. bench/measure.jl --parse-bounds-log <log> --out <row.csv> \
    --cells 32,32,32 --scale 1//32 --sep <sep> --rank 4000 --cluster narval --jobid <id>
```

`--summary` prints what it found and writes nothing. With `--out` it appends a
`stage_bounds` row in the same schema `bench/point.jl` writes, tagged `from_log=1`
and `log_complete=0`, so the fit cannot tell the two apart but a reader can.
`--jobid` fetches `MaxRSS` from `sacct`, since the process that would have read
`/proc/self/status` is gone. It works on production `compute_bounds.jl` logs too.

### What the refit changes, and what it must not

Three coefficient groups, and all three default to leaving the model exactly as it
was: `bounds_tau_mode = "legacy"`, `bounds_m_mode = "fraction"`,
`rsvd_pass_scale = 1.0`. A calibration CSV with none of the new columns -- which is
every CSV written before the windowed sweep existed -- produces byte-identical
predictions. `bench/parity_cost_model.jl` is the check:

```bash
julia bench/parity_cost_model.jl > after.txt
git stash && julia bench/parity_cost_model.jl > before.txt && git stash pop
diff before.txt after.txt
```

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
| `panel_bus` | GPU | pinned host-link rate and pipeline overlap (trial E1) |

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
- **Thread counts are pinned, not `auto`.** Every point runs at a fixed `-t`,
  because the parallel efficiency `eta(T) = 1 + s(T-1)` is fitted from the scan
  and `-t auto` would collapse the scan. The scan has to reach as far as
  production actually runs: on fir and narval that is 8 (`choose_cores` never
  picks more, since `max_cores` is 12), on molering it is every core because
  `create_jobs.jl` emits `-t auto` there. `ClusterSpec` assumes 64 cores on
  molering; if `julia -e 'println(Sys.CPU_THREADS)'` there says more, raise it and
  regenerate, otherwise the largest jobs get an extrapolated efficiency.
