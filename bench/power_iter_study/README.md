# Power-iteration quality study

How low can `--power-iterations` go before the bounds move? Production runs at
`q = 14`, which costs `2q + 2 = 30` operator passes at the sketch width. At `q = 3`
it is 8 passes, and the narval RSVD request drops with it. The point of this
directory is to find out what that costs in the bounds, not in the spectrum,
because the bounds are what gets reported.

```
run_study.sh        the launcher: greens once, then rsvd + bounds per (separation, q)
run_study_gpu0.sh   GPU 0 half
run_study_gpu1.sh   GPU 1 half
analyze.jl          reads the per-q outputs, prints the tables, writes the CSV
```

## Where it runs, and why not at 1/4 lambda

At 1/4 lambda the universe is `N_u = 3,072`, under `DENSE_EXACT_MAX_N_U = 12_288`
in `src/rsvd.jl`, so `_save_ur_asym` builds and diagonalizes the whole operator
and `q` never enters. The study therefore runs at 1/2 lambda: sender and receiver
`(16,16,16)` at scale `1//32`, `N_u = 24,576`, rank 4000, oversamples 50 — the
production point of `jobs/launch_narval_Ge1000_arxivV3_0p5x0p5x0p5_*`, and the
smallest one where the RSVD is real.

Three separations, all of them ones the narval 0p5 sweep actually runs:

| separation | name fragment | why |
|---|---|---|
| `1//16` | `1ss16` | near field. Flattest spectrum, so this is where power iterations matter most |
| `5//8` | `5ss8` | close coupling |
| `5//2` | `5ss2` | the problem region around 2.5 lambda |

`TIER=full` switches to `(32,32,32)` at 1 lambda. There the sketch no longer fits
an A6000 and the RSVD takes Funicular's panel path, which is a different
algorithm with different scaling — worth measuring, but run it `STAGES=rsvd`: see
the cost section for why its bounds stage is not affordable here.

## Directory layout

The RSVD and bounds filenames come from `file_prefix(smr)`, which encodes cells,
separation and the universe string and nothing else. Two runs at different `q`
would write the same file. So each `q` gets its own scratch and project
subdirectory:

```
/home/molering/fatmole/paulv/Photonic-System-Channels/power_iter_study/k4000/q03/   scratch
/home/paulv/Projects/Photonic-System-Channels/projects/power_iter_study/k4000/q03/  project
```

`k<RANK>` is in the path because `file_prefix` does not encode the rank either, so
a rank-1350 rerun would otherwise collide with the rank-4000 one.

Green functions are `q`-independent and go to the preload directory
(`preload_dir` in `src/common.jl`, `/home/molering/fatmole/greens_functions`),
which is separate from `--scratch`. Every `q` reuses the same blocks with no
copying. They do have to exist before either GPU process starts, or the two race
to build the same file, which is why the greens step is its own invocation below.

## What to run

```bash
cd /home/paulv/Projects/Photonic-System-Channels/

# 1. Green functions for the three separations. CPU, ~15 min each, once.
STAGES=greens bash bench/power_iter_study/run_study.sh 0

# 2. Both GPUs. RSVD for every (separation, q) first, then the bounds.
LOGS=/home/paulv/Projects/Photonic-System-Channels/projects/power_iter_study/k4000
nohup bash bench/power_iter_study/run_study_gpu0.sh > $LOGS/gpu0.log 2>&1 &
nohup bash bench/power_iter_study/run_study_gpu1.sh > $LOGS/gpu1.log 2>&1 &

# 3. Read it. Safe to run while the bounds are still going; rows with no bounds
#    output yet just say so.
julia --project=. bench/power_iter_study/analyze.jl \
    --root /home/paulv/Projects/Photonic-System-Channels/projects/power_iter_study/k4000
```

The launcher does all of the RSVD runs before any of the bounds runs. The spectra
are the cheap half and are worth having complete before the expensive half
starts. To stop after them, `STAGES=rsvd`; to come back for the bounds later,
`STAGES=bounds`. Nothing is `set -e`: a failed run logs its exit code to the CSV
and the loop carries on, and rerunning skips whatever already landed, since both
`_save_ur_asym` and `compute_bounds` check for their own output first.

Knobs, all environment variables: `RANK` (4000), `OVERSAMPLES` (50), `QS`
(`"1 2 3 4 6 14"`), `SEED` (20260814), `GAMMA_RTOL` (1.0e-12), `TIER`
(`half`/`full`), `STAGES` (`greens`/`rsvd`/`bounds`/`study`/`all`).

## The seed does not do what it looks like it does

`--seed` only reaches the sketch on the panel RSVD path.
`MatrixFreeRandomizedLinearAlgebra.reigen_hermitian` throws
`seed_without_plan` if it is given a seed with no residency plan, and
`_save_ur_asym` only builds a plan when `use_panel_path` says the sketch has
outgrown the device. At 1/2 lambda with `k = 4000` the sketch is 6.9 GiB against
an A6000's 43 GiB budget, so the in-memory path runs and draws its Gaussian from
the RNG. Two runs at the same `q` are not identical, and `--seed` is recorded in
the JLD without having been used.

Rather than force the panel path — which would measure a different algorithm than
the one production uses — the study runs the reference twice, into `q14/` and
`q14r/`. That difference is the sketch-noise floor, and `analyze.jl` prints it
under each table. A low-`q` row sitting at the `14r` level is not distinguishable
from the reference, whatever its absolute deviation looks like. Read the tables
that way; the alternative (a seeded comparison) is not available without a change
to `src/rsvd.jl`.

## Reading the output

Two tables per separation.

**cost.** `rsvd_s`, `bounds_s`, and `cost/pass`: the measured wall time per
operator pass, relative to `q = 14`'s. `1.00` means the RSVD scaled exactly with
`2q + 2`. Above `1.00` means the fixed cost — startup, the `RS/` block, the dense
algebra on the `N_u x c` matrices — is what the low-`q` run is paying for
instead, which is the number that says how much of the 18 h request actually goes
away.

**quality**, against `q = 14`:

- `npos` positive eigenvalues of `Asym(G0_ur)`, `kept` of them surviving the
  `--gamma-rtol` cut. `kept` is what sizes everything downstream, so a `q` that
  keeps a different number is doing something structural, not numerical.
- `eig_max`, the largest relative eigenvalue deviation over the kept block.
- `trace`, the reported quantity: the sum of `bounds_dual_basis` over all
  indices, i.e. the sum of the bounds on `sigma_n(P_rs)`. `src/bounds.jl` stores
  `sqrt(best_dual)`, so this is already on the singular-value scale, not the
  squared one.
- `|dtrace|`, `dtrace/t`: absolute and relative trace deviation.
- `chan_max`, `chan_med`: per-channel relative deviation, over the channels whose
  reference bound is at or above `1e-5`.
- `tail`: summed absolute deviation over the channels below that floor. Printed,
  not gated — it is here to confirm it stays far under the trace floor.
- `drop`: the mass of channels one run kept and the other cut. Already inside
  `|dtrace|`; the channel gate cannot see it, which is why it is separate.
- `dtau_max`: largest `opt_taus` disagreement over the gated channels. Reported
  rather than gated: two taus either side of a plateau can give the same bound.
- `pass`.

### The floors, and why the thresholds are what they are

A trace below `1e-4` is not reported at all, so it cannot be wrong. That is the
anchor, not a bare relative tolerance:

1. both `trace_q` and `trace_ref` under `1e-4` — pass, nothing there is
   reportable;
2. otherwise `|trace_q - trace_ref| <= max(0.01 * trace_ref, 1e-5)`, i.e. 1 %
   relative or an absolute error a decade under the reporting floor, whichever is
   looser;
3. and, over the channels whose reference bound is at or above `1e-5` — below
   that a single channel cannot materially move a reportable trace — at most 5 %
   relative deviation.

The raw relative deviations stay in the table so the margin on each `q` is
visible, but the gate is the three rules above. All six numbers are flags
(`--trace-floor`, `--trace-abs`, `--trace-rtol`, `--chan-floor`, `--chan-rtol`,
`--ref`).

The verdict line per separation is the smallest `q` that passes. If a larger `q`
failed above it, the line says so: a non-monotone pass is luck, not convergence,
and should not be believed.

`true_bounds` — the per-index minimum of `bounds_dual_basis` and the two
analytical forms — is carried in the CSV as a second trace, since that is what a
figure would plot. The gate runs on `bounds_dual_basis` because it is the only
one of the three that reads the RSVD *basis* rather than just its eigenvalues,
and so the only one where `q` can hurt in a way the spectrum does not show.

## What this costs

From `bench/coeffs_molering.jl` through `bench/cost_model.jl`, at 1/2 lambda,
`k = 4000`, `m = 0.6k = 2400`, padded (so including `time_pad = 2.10` and the
700 s recompile tax) and per run:

| stage | q = 1 | 2 | 3 | 4 | 6 | 14 |
|---|---|---|---|---|---|---|
| greens (CPU, once per separation) | 14 min | — | — | — | — | — |
| rsvd | 22 min | 26 | 30 | 34 | 41 | 72 min |

21 RSVD runs (3 separations x 6 `q` + 3 replicates) over two GPUs is about
**7.5 h wall**, and the unpadded numbers are a third of that. This half is cheap
and it is where the launcher starts.

The bounds stage is the problem. The model puts one bounds run at 1/2 lambda,
`k = 4000` at **147 h** on molering's coefficients, which over 21 runs and two
GPUs is 64 days. Do not take that at face value, but do not ignore it either:

- molering's fitted `sync_latency` is 6.7 ms, against narval's 15 us, and the
  bounds model spends one sync per probe over an `O(m^2)` loop, so that one
  coefficient carries the whole estimate. On the single molering bounds point we
  have a measured breakdown for (`bench/data/calibration_molering.csv`, 1/4
  lambda, `k = 1350`, `num_pos = 675`: `t_c_projection` 48.9 s,
  `t_gram_schmidt` 14.9 s, `t_outer_total` 2.86 s over 4 outer indices) a full
  run extrapolates to about 9 minutes. The model says 613 min for the same point.
  It is off by a factor of tens, in the conservative direction.
- narval's coefficients, which do reproduce the 18:13:45 that
  `create_jobs.jl` requests for this exact job, give 18.2 h padded / 12.0 h raw.
  An A6000 is slower than an A100 but not by an order of magnitude.

So budget somewhere between 20 h and 40 h per bounds run at `k = 4000`, i.e.
**9 to 18 days of wall time** for the bounds half on two GPUs. If that is more
than the study is worth, drop the rank — the bounds cost goes as roughly `m^4`:

| rank | m | bounds/run, narval coeffs | bounds/run, molering coeffs | 21 runs / 2 GPUs, narval coeffs |
|---|---|---|---|---|
| 800 | 480 | 0.35 h | 5.2 h | 3.7 h |
| 1350 | 810 | 0.87 h | 14.7 h | 9.1 h |
| 2048 | 1229 | 2.5 h | 34.4 h | 27 h |
| 4000 | 2400 | 18.2 h | 146.8 h | 191 h |

`RANK=1350 bash bench/power_iter_study/run_study_gpu0.sh` writes to its own
`k1350/` tree and costs about a day, and 1350 is the historical parity rank the
Funicular work back-compares against. It answers a slightly different question
than `k = 4000` does — the spectrum a rank-1350 sketch resolves is a different
spectrum — so if only one tier fits, run the RSVD tier at 4000 and the bounds
tier at 1350, and say so when the number is used.

The 1 lambda stretch tier (`TIER=full`) is about **27 h wall** for its RSVD half
over two GPUs, panel path, and its bounds stage is the same 147 h/run figure at a
larger `N_u`. Run it `STAGES=rsvd` and read the spectra only.
