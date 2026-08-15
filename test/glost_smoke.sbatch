#!/bin/bash
# Narval pre-flight 4: does GLOST run at all on narval?
#
# Submit from the repo root, so that the relative task-file path below resolves:
#
#     sbatch test/glost_smoke.sbatch
#
# This runs no Julia and touches no data. It exists to answer three questions
# before create_jobs.jl's :glost launcher is trusted with 333 Green-function
# tasks: does the module set load, does srun glost_launch start, and does the
# manager/worker split come out the way the task file assumes.
#
# What success looks like. 5 tasks over 3 MPI ranks = 1 manager (rank 0 runs no
# task once size > 1) + 2 workers. The log must show all 5 "task N done" lines,
# each preceded by a hostname, and GLOST's own per-task exit-code lines, all
# zero. Two workers take the 5 tasks greedily, so expect roughly 3 rounds of
# 5 s, not 5 tasks in parallel. If only 4 lines appear, or if the log shows one
# rank doing everything, the manager accounting is different from what the
# greens task file assumes and the pre-step / farming split in create_jobs.jl
# has to be rechecked.
#
# The resume path, worth trying once the run above succeeds: kill or shorten a
# run, then extract the tasks that did not finish and resubmit those.
#
#     glost_filter -H glost_smoke_<jobid>.out test/glost_smoke_tasks.txt > remaining.txt
#
# The flags are version-dependent and unverified here, so check glost_filter -h
# on narval first. This is the mechanism the real sweep depends on to survive a
# walltime kill, so it is worth confirming on 5 trivial tasks rather than on a
# node-day of Green functions.

#SBATCH --job-name=glost_smoke
#SBATCH --output=glost_smoke_%j.out
#SBATCH --account=def-smolesky
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# glost_launch is an MPI program, so it needs a compiler + MPI loaded first.
# THIS MODULE SET IS NOT VERIFIED ON NARVAL. VERIFY with: module spider glost
# (and check which openmpi it wants under StdEnv/2023) before submitting. A
# wrong module line here is exactly what this smoke test is for.
module load StdEnv/2023 gcc openmpi glost

echo "[smoke] modules:"
module list 2>&1
echo "[smoke] glost_launch: $(command -v glost_launch)"
echo "[smoke] ntasks = $SLURM_NTASKS, cpus-per-task = $SLURM_CPUS_PER_TASK"

srun glost_launch test/glost_smoke_tasks.txt
