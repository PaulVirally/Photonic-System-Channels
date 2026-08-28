#!/bin/bash
# bench/backfill_probe/submit_probe.sh
#
# Submits the backfill-probe matrix: every (requested walltime) x (resource
# shape) x rep combination, as one sbatch job each. Each job requests time T
# but only actually runs ~BUSY_MINUTES of real GPU work (see busywork.jl).
# SLURM's backfill scheduler plans around the *requested* walltime, so T is
# the thing this probe varies, and "how long did it wait before Start" is the
# thing it measures. See README.md for the rest of the design.
#
# The job script itself lives in the static bench/backfill_probe/probe_job.sh,
# not in a heredoc here. This repo has been bitten before by a backtick inside
# an unquoted heredoc turning into a command substitution nobody wanted; the
# fix used here is to not put a job script inside a heredoc at all. Every
# resource request is passed to sbatch on the command line instead.
#
# Usage:
#   bash bench/backfill_probe/submit_probe.sh          # submit for real
#   DRY=1 bash bench/backfill_probe/submit_probe.sh     # print sbatch lines, submit nothing
#
# All knobs are environment-overridable, e.g.:
#   TIMES="10 60" SHAPES="slice" REPS=1 bash bench/backfill_probe/submit_probe.sh

set -u

# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #

: "${TIMES:=10 20 30 60 120 180}"          # requested walltime, minutes
: "${SHAPES:=whole slice}"                 # resource shapes to sweep
: "${REPS:=2}"                             # repeats per (shape, time) cell
: "${BUSY_MINUTES:=10}"                    # target minutes of real GPU work per job
: "${ACCOUNT:=def-smolesky}"
: "${CODE_DIR:=$HOME/Photonic-System-Channels/}"
: "${LOG_ROOT:=$HOME/scratch/psc-backfill-probe}"
: "${DRY:=0}"

# Resource shapes. Two are defined by default:
#   whole: one full A100 (the 4-lambda RSVD shape)
#   slice: one 3g.20gb MIG slice of an A100 (the bounds-job shape)
# Override e.g. WHOLE_GRES / WHOLE_MEM / WHOLE_CPUS / SLICE_GRES / SLICE_MEM /
# SLICE_CPUS individually, or add a new shape name to SHAPES and define its
# own <NAME>_GRES / <NAME>_MEM / <NAME>_CPUS (upper-cased) before running this.
: "${WHOLE_GRES:=--gpus=a100:1}"
: "${WHOLE_MEM:=--mem=124G}"
: "${WHOLE_CPUS:=--cpus-per-task=2}"

: "${SLICE_GRES:=--gpus=a100_3g.20gb:1}"
: "${SLICE_MEM:=--mem=16G}"
: "${SLICE_CPUS:=--cpus-per-task=2}"

JOB_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probe_job.sh"
LOGDIR="$LOG_ROOT/logs"

if [ "$DRY" != "1" ]; then
    mkdir -p "$LOGDIR"
fi

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# minutes -> HH:MM:SS for --time
to_hhmmss() {
    local total=$1
    printf '%02d:%02d:00' $((total / 60)) $((total % 60))
}

# shape name -> its three resource flags, via indirect expansion on
# <SHAPE_UPPER>_GRES / _MEM / _CPUS. Kept as plain variables (not an
# associative array) so this also runs under bash 3.2.
shape_flag() {
    local shape=$1 kind=$2
    local upper
    upper=$(printf '%s' "$shape" | tr '[:lower:]' '[:upper:]')
    local varname="${upper}_${kind}"
    printf '%s' "${!varname:?unknown shape '$shape' (define ${varname})}"
}

# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #

total_jobs=0
echo "submit_probe.sh: TIMES=[$TIMES]  SHAPES=[$SHAPES]  REPS=$REPS  BUSY_MINUTES=$BUSY_MINUTES  DRY=$DRY"
echo "submit_probe.sh: logs under $LOGDIR"
echo

for shape in $SHAPES; do
    gres=$(shape_flag "$shape" GRES) || exit 1
    mem=$(shape_flag "$shape" MEM) || exit 1
    cpus=$(shape_flag "$shape" CPUS) || exit 1

    for minutes in $TIMES; do
        time_fmt=$(to_hhmmss "$minutes")

        # Actual busywork = min(BUSY_MINUTES, requested - 2), floored at 1, so
        # even a 10-minute request finishes with margin instead of racing the
        # time limit.
        cap=$((minutes - 2))
        if [ "$cap" -lt 1 ]; then
            cap=1
        fi
        busy=$BUSY_MINUTES
        if [ "$busy" -gt "$cap" ]; then
            busy=$cap
        fi

        rep=1
        while [ "$rep" -le "$REPS" ]; do
            jobname="bfprobe_${shape}_${minutes}m_r${rep}"
            outfile="$LOGDIR/${jobname}_%j.out"

            sbatch_args=(
                --parsable
                --account="$ACCOUNT"
                --job-name="$jobname"
                --output="$outfile"
                --time="$time_fmt"
                "$gres"
                "$mem"
                "$cpus"
                --chdir="$CODE_DIR"
                --export=ALL
                "$JOB_SCRIPT" "$busy"
            )

            if [ "$DRY" = "1" ]; then
                printf 'sbatch'
                printf ' %q' "${sbatch_args[@]}"
                printf '\n'
            else
                jid=$(sbatch "${sbatch_args[@]}")
                echo "  ${jobname}  time=${time_fmt}  busy=${busy}m  ${gres}  ${mem}  -> job ${jid}"
                sleep 0.05
            fi

            total_jobs=$((total_jobs + 1))
            rep=$((rep + 1))
        done
    done
done

echo
echo "submit_probe.sh: $total_jobs job(s) $( [ "$DRY" = "1" ] && echo "would be submitted (DRY=1)" || echo "submitted" )"
