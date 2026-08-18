# Funicular integration plan: 4000 components on narval

The goal is to run the full pipeline at k = 4000 components with an x-scale of
1/32, on narval's A100-40GB cards, using MIG slices where they fit and queueing
as few jobs as possible. The capacity math below was done on 2026-08-14 with
the calibrated narval coefficients.

| cube  | cells        | N_u       | RSVD host peak (c=4050) | RSVD time | bounds m≈2400 | allocation           |
|-------|--------------|-----------|--------------------------|-----------|----------------|----------------------|
| 1/4 λ | (8,8,8)      | 3,072     | (dense-exact)            | minutes   | trivial        | smallest MIG slice   |
| 1/2 λ | (16,16,16)   | 24,576    | 3.2 GB (in-memory path)  | ~0.5 h    | ~1.3 h         | a100_2g.10gb         |
| 1 λ   | (32,32,32)   | 196,608   | 26 GB                    | ~0.9 h    | ~1.5 h         | a100_3g.20gb         |
| 2 λ   | (64,32,32)   | 393,216   | 51 GB                    | ~1.6 h    | ~1.8 h         | a100_3g.20gb (62 GB host, tight) or a100 |
| 4 λ   | (128,32,32)  | 786,432   | 102 GB                   | ~2.9 h    | ~2 h           | whole a100 (124.5 GB host) |

On the panel path the peak is 2 host-resident `N_u × c` matrices (the power
iteration in `reigen_hermitian` holds Y and Z at once), and the random test
matrix costs nothing because it is regenerated rather than stored. At 4 λ the
ceiling inside the single-GPU bundle is k ≈ 4,300, so 4000 leaves about 7% of
headroom before we need the NVMe spill valve.

The following decisions are already made:

- We save only the positive-Γ eigenvectors, in ComplexF64. This brings the
  disk usage of the 4 λ sweep from about 17 TB down to about 10 TB.
- Node-local NVMe spill (through `$SLURM_TMPDIR`) is allowed for tight runs.
  During `cholqr2!` only one of the two matrices needs full host residency, so
  spilling the other drops the floor to roughly one matrix plus panels, that
  is, 56 to 60 GB at 4 λ.
- 1/4 λ is computed densely and exactly. N_u = 3,072 there, which is smaller
  than 4000, so the "rank" is the full spectrum and there is no reason to
  sketch at all.
- No warm-starting across separations (`seed_Q` chaining). The separations run
  concurrently, so there is no previous Q to inherit. This is an explicit
  non-goal.
- We run benchmark trials on narval before committing the sweep (workstream E).
- The Green-function jobs go through GLOST (workstream F): one Slurm job runs
  all 333 CPU tasks, which frees queue slots.

---

## Workstream A: dependencies and plumbing

1. Manifest: point MFRLA at the `paul-funicular` branch (v0.4.0) and add
   Funicular.jl by URL (`https://github.com/PaulVirally/Funicular.jl`, not
   registered). MFRLA's own `[sources]` carries Funicular for its test target,
   but this project needs its own entry for the extension to load. Add
   HDF5.jl, which enables Funicular's disk tier and is the write path for the
   streamed vector save. Pin exact commit SHAs.
2. Cluster instantiation: compute nodes have no internet, so we
   `Pkg.instantiate` on the narval login node after every Manifest bump and
   check that the depot's `clones/` carries both git dependencies. The Julia
   module stays `julia/1.12.5` (MFRLA requires 1.11 or newer).
3. The branch is additive (a `plan` keyword), so existing call sites should
   keep compiling. Check the 0.2.0 to 0.4.0 changes from the
   `paul-performance` merge before bumping anyway.

## Workstream B: RSVD panel path (`src/rsvd.jl`)

1. Plan construction (`src/Params.jl` or `common.jl`): a
   `residency_plan(compute_env)` helper that reads
   `device_budget` = 0.9 × `CUDA.total_memory()` (this works on MIG slices
   too, since total_memory reports the slice), `host_budget` =
   `SLURM_MEM_PER_NODE` (falling back to `Sys.total_memory()`) minus a
   measured overhead reserve of about 6 GB (trial E2 pins this down),
   `workspace_bytes` from B2, and `scratch_dir` = `$SLURM_TMPDIR/funicular`
   when that variable exists.
2. The operator contract: `G₀_ur_asym` is a LinearMaps composition, so it
   cannot carry traits, and `workspace_bytes` is passed to the plan instead.
   We measure it once in trial E2 (Gila's FFT plan work areas plus the
   composition's per-apply temporaries, a few N_u-vectors, roughly 0.5 to
   1 GB at 4 λ) and encode it per size in the cost model. `validate=true`
   stays on: `check_operator` costs a handful of matvecs against multi-hour
   jobs and catches adjoint and contiguity mistakes.
3. Path selection: keep the current in-memory `CuArray` path whenever
   `3 N_u c · 16 · 1.554 ≤ device_budget`. It is faster when it fits, which
   covers 1/2 λ at k=4000 and everything during small test runs. Above that,
   pass a plan. One predicate, logged loudly.
4. Seeds: the panel path takes an integer `seed` instead of the global RNG.
   Add `--seed` (default: `hash(experiment_name)` truncated) and store it in
   the JLD output. Results are reproducible up to floating-point rounding
   across panel widths and machines, within one Julia version.
5. Positives-only save, in F64: run `reigen_hermitian(...; factored=true)`.
   The sort is descending, so the positive-Γ vectors are a prefix:
   `m = count(>(0), Γ)`. Form only those columns with
   `rightmul!(V_pos, Q, rotation[:, 1:m])`, an `N_u × m` panel matrix (the
   full k columns are never materialized), and stream its panels to disk. The
   format is a separate `<prefix>_vectors.h5` with one chunked dataset (chunk
   = one panel); the existing `.jld` keeps `UR_asym/D` (all k values), the
   seed, and metadata. The skip-if-exists check moves to the h5 dataset.
6. Dense-exact branch for tiny universes: when `N_u ≤ 12,288` (which covers
   1/4 λ), build dense `Asym(G⁰ᵤᵣ)` by applying the operator to the identity
   (N_u matvecs, cheap at that size), `eigen!` it, save the positive prefix,
   and mark the output `exact=true`. The same branch gives `RS/D` by dense
   `svdvals`. There is no RSVD error at the smallest size, and the job fits
   the smallest MIG slice.
7. `_run_rsvdvals("RS/")`: pass the same plan. The extension already handles
   it, and its peak (2 × `N_r × c`, half the reigen peak) is never binding.

## Workstream C: bounds panel path (`src/bounds.jl`)

The bounds job is the binding constraint at k=4000: in-memory it would want
more than 130 GB of device memory at 4 λ. The pencil stage does not change,
since the `m × m` objects are about 1.7 GB on the device at m=2400. Only the
N_u-scale front end moves to panels:

1. Load the positive vectors as a `PanelMatrix` from the h5 dataset (panels
   stay cold until swept) instead of one `CuArray`.
2. Reverse Gram-Schmidt becomes two panel sweeps: one `gram` sweep for
   `G = basisᴴ basis`, the reverse-ordered Cholesky and triangular inverse of
   the reversed-permuted `G` on the host (an m × m computation), then one
   `rightmul!` sweep `ss = basis · T`. The basis is RSVD output and therefore
   near-orthonormal, so the squared conditioning of the Cholesky route is
   harmless here. We validate this against the existing loop at k=1350
   (trial E4).
3. Projections (`ss_basis`, `C_basis`, `D_basis`): `panelmul!` for the `m`
   applications of `C` (the 8m Green matvecs are unchanged) and `gram` for
   the contractions. Everything downstream of the m × m reduction stays as it
   is.
4. `verify_bounds.jl`: `gs_pos` (N_u × m, about 30 GB at 4 λ) stays
   host-resident dense, since it fits every host bundle we would use, with
   per-probe device staging. This is a small separate change; audit it after
   C1 through C3 land.
5. One assumption to monitor: `NUM_POS_FRACTION = 0.6`. The measured range is
   0.22 to 0.52. If a k=4000 run comes back with m much larger than 2400,
   both the bounds time (which scales as m⁴) and the disk move, so the RSVD
   job log prints the measured fraction where it is visible early.

## Workstream D: scheduling and cost model

1. `bench/cost_model.jl` gains a panel-mode branch in `rsvd_counts` and
   `bounds_counts`, selected by the same predicate as B3. Host bytes are
   `2 N_u c · 16` plus a base for the RSVD, and roughly `3 N_u m · 16` (basis,
   ss, and one working matrix) for bounds. Device bytes are the staging
   buffers (`nbuffers · N_u · w · 16` per swept matrix) plus
   `workspace_bytes` plus the small reduced blocks. Time keeps the existing
   matvec terms and adds the CholeskyQR2 gemm term `(q+1) · 32 · N_u · c²`
   and a PCIe sweep term whose rate comes from trial E1.
2. MIG selection: `choose_gpu` already picks the smallest fitting slice, so it
   only needs the corrected predictions. At k=4000 we expect the allocation
   column of the table above. The two tight cases (1 λ on 2g.10gb versus
   3g.20gb, and 2 λ on 3g.20gb versus the whole card, possibly with spill)
   get decided by the trials.
3. `print_plan` gains a disk line: the positives-only F64 scratch usage per
   sweep, so a sweep that exceeds quota is visible before submission.
4. Queue footprint: with GLOST (workstream F), a 333-separation sweep costs
   1 to 4 greens jobs plus 333 RSVD plus 333 bounds, about 670 slots against
   narval's 1000-job MaxSubmit, versus 999 today. An option to decide after
   the trials: merging RSVD and bounds into one GPU job per separation brings
   this to about 337 slots and removes the inter-job scratch handoff, at the
   cost of coupling the MIG choice to the larger of the two stages.

## Workstream E: narval benchmark trials

A new manifest `bench/manifest_narval_funicular.csv` and a `funicular` tier in
`bench/plan.jl` and `measure.jl`. Every trial is a single separation (a
mid-sweep, non-contact value such as a 1/2 λ gap). The budget target is about
8 jobs and under 30 GPU-hours.

| # | trial | allocation | what it pins down |
|---|-------|-----------|-------------------|
| E1 | Funicular's own `benchmark/overlap.jl` and `pinned.jl` | 1 GPU, 1 h | the achievable pinned PCIe rate and pipeline overlap on narval, that is, the sweep-time coefficient |
| E2 | 1 λ, k=1350, same seed, in-memory vs panel | 2 jobs | eigenvalue parity to floating-point tolerance; the wall-clock ratio (we expect at most 1.3×); host and device high-water against the model; Gila `workspace_bytes`; the host overhead reserve |
| E3a | 2 λ, k=4000, panel RSVD | 3g.20gb and whole card | whether the 62 GB bundle really holds 51 GB plus overhead; the time split between matvec, cholqr2, and bus |
| E3b | 4 λ, k=4000, panel RSVD | whole card | the 102 GB peak against the 124.5 GB bundle; the measured positive fraction |
| E3c | 4 λ, k=4000, `host_budget` forced to 60 GB | whole card + NVMe | the spill path end to end, and the NVMe throughput cost |
| E4 | bounds at k=4000 (1 λ and 4 λ) on E3 outputs | 3g.20gb / whole card | the panelized front end against the old loop (at 1350), and the m⁴ pencil-time constant |

Afterwards we refit: extend `bench/fit.jl` with the panel coefficients,
regenerate `coeffs_narval.jl`, rerun `create_jobs.jl`, and read `print_plan`
against the table above. Local development happens on molering first (an
A6000 with 480 GB of host memory and no queue, which makes it a good
Funicular testbed).

## Workstream F: GLOST for Green-function generation

Findings. Narval caps each user at 1000 jobs running plus queued (Slurm
MaxSubmit), which is what pins sweeps at 333 × 3. GLOST (CEA's Greedy
Launcher Of Small Tasks, available as an Alliance module) is an MPI program:
rank 0 is a manager, and ranks 1 through N−1 greedily execute lines of a task
file, one shell command per line. One GLOST job occupies exactly one queue
slot regardless of task count, which is the property we want. Per-task exit
codes land in the log, `glost_filter` extracts unfinished tasks for a
resubmission list, and `--signal=USR1@600` plus GLOST's SIGUSR1 handler
drains gracefully at the walltime edge. Tasks are processes, so a task may
use `SLURM_CPUS_PER_TASK` threads; "serial" only means one task per rank.
GPU stages do not fit this model, so GLOST applies to the CPU-only greens
job.

The design:

1. `create_jobs.jl` emits, per sweep, a task file
   `jobs/greens_tasks_<PROJECT>.txt` with one line per separation
   (`julia --project=. -t $SLURM_CPUS_PER_TASK generate_green.jl <args> --gpu false ...`)
   and one sbatch script running `srun glost_launch` on a narval CPU node.
   With `--nodes=1 --ntasks=13 --cpus-per-task=4 --mem=240G` we get 12
   workers of 4 threads each plus the manager. Greens tasks measure 2 to
   4 GB each and run 10 to 30 minutes at 4 threads, so 333 tasks are roughly
   one node-day (two nodes if we are impatient). Verify the module line on
   narval first (`module spider glost`); glost_launch needs an MPI loaded.
2. The shared-preload race: the `(R,R)` self block does not depend on the
   separation, and today each of the 333 jobs checks for it and builds it if
   missing. Before farming, the sbatch script builds the shared block once,
   before `glost_launch` starts, so 12 concurrent workers never race on it.
   While touching this, confirm `load_green_function` writes preload files
   through a temp-file-and-rename (atomic) rather than in place.
3. Dependencies: all RSVD jobs take `--dependency=afterok:<glost_job_id>`.
   This is coarser than today's per-experiment chain, since everything waits
   for the slowest greens task, but greens is the cheap stage. If that ever
   matters, split the task file into about 4 GLOST jobs by separation range
   and depend range-wise.
4. A side benefit: all tasks share the node and depot, so Julia recompiles
   once instead of 333 times. That alone recovers most of the
   `RECOMPILE_OVERHEAD_S = 700` seconds the current requests carry per job.
5. Julia's own startup (about 40 s per task) is negligible against 10 to
   30 minute tasks. If the tasks ever get much shorter, revisit this with a
   persistent-worker pattern instead of GLOST.

## Sequencing

1. M1: A and B on molering (panel RSVD, positives-only h5 save, dense 1/4 λ).
2. M2: trials E1 through E3 on narval; refit the coefficients; freeze the MIG
   table.
3. M3: C (the bounds front end) and E4; the cost-model branch of D.
4. M4: F (GLOST greens), which is independent of B and C and can proceed in
   parallel.
5. M5: the full k=4000 sweep, 1 λ first since it is the cheapest
   full-pipeline validation, then 2 λ and 4 λ.

## Risks

- The 4 λ bundle margin (a 102 GB peak plus overhead against 124.5 GB) is
  real but thin. E3b measures it and E3c proves the spill valve. In the worst
  case the 4 λ RSVD requests 2 GPU-equivalents of memory (249 GB) and still
  runs on one card.
- CholeskyQR2 conditioning at c=4050: sketches after the power iterations are
  well conditioned, and the shifted-QR3 fallback exists. E2's parity check is
  the guard.
- If num_pos comes out above 0.6k, the bounds time (m⁴) and the disk both
  move. The logged positive fraction (C5) surfaces this early.
- JLD2 and HDF5 stay in separate files by design (B5), so there is no format
  interop risk.
- GLOST module and MPI quirks on narval: verify with a 5-line smoke-test task
  file before wiring it into `create_jobs.jl`.
