#!/bin/bash
# bench/backfill_probe/collect.sh
#
# Reads back the backfill-probe matrix from sacct and prints queue wait
# (Start - Submit) per job, plus the median per (shape, requested-time) cell.
# Safe to run while jobs are still pending/running; see the PENDING handling
# below. Pure bash/awk, no julia dependency.
#
# Usage (run on narval, same account the jobs were submitted under):
#   bash bench/backfill_probe/collect.sh
#   SINCE=2026-08-20T00:00 bash bench/backfill_probe/collect.sh
#   TIMES="10 60" SHAPES="slice" REPS=1 bash bench/backfill_probe/collect.sh
#
# TIMES/SHAPES/REPS should match whatever submit_probe.sh was actually called
# with, so the exact-name list handed to sacct --name matches what was
# submitted. If you are not sure what was used, widen NAME_PATTERN instead
# (see below) and it will fall back to a prefix match done in awk.

set -u

: "${TIMES:=10 20 30 60 120 180}"
: "${SHAPES:=whole slice}"
: "${REPS:=2}"
: "${SINCE:=}"
: "${NAME_PATTERN:=bfprobe_}"

if [ -z "$SINCE" ]; then
    SINCE=$(date -d '-12 hours' +%Y-%m-%dT%H:%M 2>/dev/null || date -v-12H +%Y-%m-%dT%H:%M 2>/dev/null || echo "1970-01-01T00:00")
fi

NOW_EPOCH=$(date +%s)
echo "collect.sh: run at $(date -Is)"
echo "collect.sh: sacct window --starttime=$SINCE, user=$USER"
echo

# Build the exact job-name list submit_probe.sh would have used, as a comma
# list for sacct --name (sacct matches names exactly, not by glob).
names=""
for shape in $SHAPES; do
    for minutes in $TIMES; do
        rep=1
        while [ "$rep" -le "$REPS" ]; do
            jobname="bfprobe_${shape}_${minutes}m_r${rep}"
            if [ -z "$names" ]; then
                names="$jobname"
            else
                names="$names,$jobname"
            fi
            rep=$((rep + 1))
        done
    done
done

RAW=$(sacct -u "$USER" --starttime "$SINCE" --name="$names" \
    --format=JobName%40,Submit,Start,State,Timelimit,Elapsed --parsable2 --noheader 2>/dev/null)

if [ -z "$RAW" ]; then
    echo "collect.sh: sacct returned nothing for those exact names in this window."
    echo "collect.sh: falling back to a prefix scan for '${NAME_PATTERN}*' (widen SINCE if this is also empty)."
    RAW=$(sacct -u "$USER" --starttime "$SINCE" \
        --format=JobName%40,Submit,Start,State,Timelimit,Elapsed --parsable2 --noheader 2>/dev/null \
        | awk -F'|' -v pat="$NAME_PATTERN" 'index($1, pat) == 1')
fi

if [ -z "$RAW" ]; then
    echo "collect.sh: still nothing. Nothing has been submitted in this window, or SINCE/NAME_PATTERN need adjusting."
    exit 1
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# Per-job table: parse name -> shape/minutes/rep, compute wait in minutes.
# pending flag: 1 if Start is not known yet (job still queued).
trim() { printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }

while IFS='|' read -r name submit start state timelimit elapsed; do
    name=$(trim "$name")
    submit=$(trim "$submit")
    start=$(trim "$start")
    state=$(trim "$state")
    [[ "$name" =~ ^bfprobe_([A-Za-z0-9]+)_([0-9]+)m_r([0-9]+)$ ]] || continue
    shape=${BASH_REMATCH[1]}
    minutes=${BASH_REMATCH[2]}
    rep=${BASH_REMATCH[3]}

    submit_epoch=$(date -d "$submit" +%s 2>/dev/null)
    if [ -z "$submit_epoch" ]; then
        continue
    fi

    if [ "$start" = "Unknown" ] || [ -z "$start" ]; then
        pending=1
        wait_min=$(( (NOW_EPOCH - submit_epoch) / 60 ))
        wait_display="still waiting >= ${wait_min} min (state=${state})"
    else
        pending=0
        start_epoch=$(date -d "$start" +%s 2>/dev/null)
        if [ -z "$start_epoch" ]; then
            continue
        fi
        wait_min=$(( (start_epoch - submit_epoch) / 60 ))
        wait_display="${wait_min} min"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$shape" "$minutes" "$rep" "$pending" "$wait_min" "$submit" "$start" "$state" >> "$TMP"

    printf '  %-6s %5sm rep=%s  submit=%-20s start=%-20s state=%-10s wait=%s\n' \
        "$shape" "$minutes" "$rep" "$submit" "${start:-(pending)}" "$state" "$wait_display"
done <<< "$RAW"

echo
echo "collect.sh: median queue wait per (shape, requested time). '*' means a pending"
echo "job in that cell was excluded from the median (its wait is only a lower bound)."
echo

sort -k1,1 -k2,2n "$TMP" | awk -F'\t' '
{
    key = $1 "\t" $2
    if (!(key in seen)) { order[++nkeys] = key; seen[key] = 1 }
    if ($4 == 0) {
        n[key]++
        vals[key, n[key]] = $5
    } else {
        pend[key]++
    }
}
END {
    printf "  %-8s %10s  %10s %10s %12s\n", "SHAPE", "REQ_MIN", "STARTED", "PENDING", "MEDIAN_WAIT"
    for (i = 1; i <= nkeys; i++) {
        key = order[i]
        split(key, parts, "\t")
        m = n[key] + 0
        for (j = 1; j <= m; j++) arr[j] = vals[key, j]
        for (j = 2; j <= m; j++) {
            v = arr[j]; k = j - 1
            while (k >= 1 && arr[k] > v) { arr[k+1] = arr[k]; k-- }
            arr[k+1] = v
        }
        if (m == 0) {
            median = "n/a"
        } else if (m % 2 == 1) {
            median = arr[(m+1)/2] " min"
        } else {
            median = (arr[m/2] + arr[m/2+1]) / 2 " min"
        }
        p = pend[key] + 0
        mark = (p > 0) ? "*" : ""
        printf "  %-8s %9smin  %10d %10d %12s%s\n", parts[1], parts[2], m, p, median, mark
        delete arr
    }
}
'
