#!/bin/bash
# bench/backfill_probe/probe_job.sh
#
# Static sbatch payload for one backfill-probe job. It is handed straight to
# sbatch as the script itself (not built via a heredoc), with every resource
# request (--time, --gpus, --mem, --cpus-per-task, --account, --job-name,
# --output, ...) coming from the sbatch command line in submit_probe.sh. This
# file only knows how many minutes of busywork to run.
#
#   sbatch --account=... --job-name=... --output=... --time=... --gpus=... \
#       --mem=... --cpus-per-task=... --chdir=... \
#       bench/backfill_probe/probe_job.sh <minutes>
#
# $1: minutes of genuine GPU work to do (passed straight to busywork.jl).
#
# By the time this script's first line runs, the measurement the probe exists
# to make (queue wait, Start minus Submit) is already over; sacct has it.
# Everything below is just making the job look, and be, like real utilization.

set -eu

MINUTES=${1:?"probe_job.sh requires <minutes> as its first argument"}

echo "probe_job.sh: SLURM_JOB_ID=${SLURM_JOB_ID:-unknown} SLURM_JOB_NAME=${SLURM_JOB_NAME:-unknown}"
echo "probe_job.sh: started at $(date -Is), requesting ${MINUTES} minute(s) of busywork"
echo "probe_job.sh: queue wait for this job is Start - Submit, already fixed before this line ran; see collect.sh / sacct"

module load StdEnv/2023 julia/1.12.5 cuda/12.2

cd "${SLURM_SUBMIT_DIR:-.}"

julia --project=. bench/backfill_probe/busywork.jl --minutes "$MINUTES"

echo "probe_job.sh: done at $(date -Is)"
