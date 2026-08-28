# Backfill queue latency probe

Empirical answer to "if I ask narval for T minutes on shape S, how long do I
wait in the queue before it starts", for our (low-priority, no RRG) account.
Not a model of SLURM's backfill scheduler, just a small repeatable
measurement of it, done twice at different times of day so a single lucky or
unlucky sample doesn't get mistaken for the answer.

```
busywork.jl     genuine GPU (CPU-fallback) compute for a fixed number of minutes
probe_job.sh    static sbatch payload: module load + busywork.jl, minutes as $1
submit_probe.sh submits the (requested-time x shape x rep) matrix
collect.sh      reads sacct back into a wait-time table
```

## The trick, and why it matters

SLURM's backfill scheduler plans around the walltime a job *requests*
(`--time`), not what it uses. So the queue wait for a 3-hour request and a
10-minute request differ even if both jobs do the same amount of work: a
short request fits into gaps a long one can't. That's the variable this probe
sweeps: each job asks for one of several `--time` values but only actually
*runs* about `BUSY_MINUTES` (10 by default) of real GPU work, then exits.

The measurement itself, queue wait, `Start - Submit`, is complete the
instant the job starts running. Nothing that happens afterward changes it.
That's what makes the trick sound: a `sleep 10m` job would answer the same
question, but it would also (a) show up in `sacct`/`seff` as a job that used
none of its requested walltime, which is exactly the pattern Alliance's
fairshare accounting and a human auditor both notice, and (b) never actually
touch the GPU, so a broken CUDA stack on some node would go unnoticed.
`busywork.jl` instead runs real, synchronizing 4096x4096 ComplexF32 gemms
(cuBLAS via CUDA.jl; falls back to CPU BLAS with a loud warning if CUDA isn't
functional) and logs an achieved-TFLOPS line every ~30s, so the job's log
looks like, and is, real utilization.

## Efficiency caveat, read before scaling this up

Alliance tracks used-vs-requested walltime per account, and this probe's
whole design is jobs that finish well inside their requested time (a
10-minute request only runs ~8 minutes of work; a 3-hour request also only
runs ~10). That's necessary to sweep `--time` at all, but it is not a
generally acceptable submission pattern: do it too much and the account's
efficiency numbers look bad for real reasons. This batch is deliberately
small (24 jobs, one run of the defaults below) and BUSY_MINUTES defaults to a
real 10 minutes of compute per job, not a token amount. Don't multiply REPS
or add more TIME points without thinking about the same tradeoff.

## Defaults (all env-overridable, see the scripts)

- `TIMES`: 10 20 30 60 120 180 (minutes, becomes `sbatch --time`)
- `SHAPES`: `whole` (`--gpus=a100:1 --mem=124G --cpus-per-task=2`, the 4-lambda
  RSVD shape) and `slice` (`--gpus=a100_3g.20gb:1 --mem=16G --cpus-per-task=2`,
  the bounds-job shape)
- `REPS`: 2
- `BUSY_MINUTES`: 10, capped to `requested - 2` for short requests so the job
  always finishes with margin instead of racing its own time limit

6 times x 2 shapes x 2 reps = 24 jobs. Actual GPU-minutes consumed sum to
about 3.9 hours total across all 24 jobs (8 min at the 10-minute cell, 10 min
everywhere else, per shape per rep), call it "~4 wall-hours", weighted
toward `slice` jobs being cheap and `whole` jobs being a full A100 each for
that same wall-clock time. `--time` requested (not used) ranges up to 3h per
job; nothing here approaches narval's job-count or GPU-hour limits.

## Running it

On narval, from the repo root:

```bash
bash bench/backfill_probe/submit_probe.sh
```

Dry run first if you want to see exactly what would be submitted without
touching the queue:

```bash
DRY=1 bash bench/backfill_probe/submit_probe.sh
```

Collect (works while jobs are still queued or running; pending jobs show up
as "still waiting"):

```bash
bash bench/backfill_probe/collect.sh
```

`collect.sh` needs the same `TIMES`/`SHAPES`/`REPS` you submitted with (to
rebuild the exact job-name list for `sacct --name`); if you changed those,
override them the same way here. `SINCE` defaults to 12 hours ago. Widen it
if you're collecting long after submitting.

## Getting a fair picture: run it twice

Backfill availability depends heavily on what everyone else on the cluster is
doing, which has a strong time-of-day / day-of-week shape. One run tells you
about that one moment. Submit the batch once during a period you'd guess is
busy (weekday daytime) and once when you'd guess it's quiet (weekend or
overnight), and compare the two tables. `collect.sh` prints its own run
timestamp and every job's `Submit` timestamp precisely so this comparison is
possible later, from saved output, without re-running anything.

## Reading the table into a decision

`collect.sh` prints, per (shape, requested-time) cell: how many reps started,
how many are still pending, and the median wait (in minutes) across the ones
that started. A `*` marks a cell where a still-pending job was excluded from
that median. Treat that median as optimistic and the wait as unresolved
until it clears.

The number you actually want out of this is a **time target per shape**: the
largest requested `--time` whose median wait is still one you're willing to
sit through. Walk each shape's row from small `--time` to large; the wait
usually rises (a longer request fits into fewer backfill gaps), but not
always monotonically: narval's scheduler state is not yours to control, and
2 reps is a small sample. If wait is flat or noisy across `--time`, that
itself is useful: it says the request length isn't the lever, queue depth is,
and lowering `--time` on real jobs won't buy much. Do this separately for
`whole` and `slice`. A 3g.20gb slice request typically clears far faster
than a whole-A100 request at the same `--time`, because it fits into far more
of what's actually free.

## Assumptions about narval worth checking yourself

- GPU/gres names: `a100:1` and `a100_3g.20gb:1`. Confirm with
  `sinfo -o "%G" | sort -u`. bench/README.md's cluster table lists narval's
  full MIG slice set as `a100_{1g.5gb,2g.10gb,3g.20gb}`.
- Account: `def-smolesky` (the same one every other launcher in `bench/`
  uses). Override with `ACCOUNT=` if that's changed.
- Module line: `StdEnv/2023 julia/1.12.5 cuda/12.2`, copied from the other
  narval launchers in `bench/`. If those have moved on to a newer stack,
  update `probe_job.sh` to match: a probe run against a stale module set
  measures the wrong queue.
- `$HOME/scratch/psc-backfill-probe/` is created for logs; nothing here
  writes anywhere else.
