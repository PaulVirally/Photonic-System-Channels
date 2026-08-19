#!/bin/bash
# Cost-model calibration for narval, tier=backfill.
# Generated 2026-08-19T11:13:49.945 by bench/plan.jl. Do not edit; regenerate instead.
#
# Every point is its own job: one point running out of memory or time must
# not take the rest of the calibration with it. Each writes its own row file,
# so partial results are always usable.
#
# Submit:  bash <this script>
# Collect: bash <this script> --merge
# Pick:    bash <this script> --pick   (choose the bounds separations, submit nothing)

# ---------------------------------------------------------------------------
# Every job in this tier asks for at most 03:00:00, so that all of them are
# eligible for narval's backfill window. Nothing here needs a reservation and
# nothing here should ever sit in the queue behind an 18 h request.
#
# Three things this script assumes about narval. It checks the first itself and
# fails loudly if it is wrong; the other two are worth a glance first.
#
# 1. The cancelled 1 lambda sweep left RSVD outputs on scratch:
#
#      ls /home/pvirally/scratch/Photonic-System-Channels/narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale/ | grep -c _UR_asym_Vpos.h5
#
#    bench/pick_bounds_points.jl runs before anything is submitted, lists what
#    is actually there, reads each spectrum, and picks 4 of them
#    spread over the range of surviving m. Run it on its own first if you want to
#    see the table without submitting anything:
#
#      bash bench/launch_calibration_narval_backfill.sh --pick
#
#    It writes the full kept-count table next to the pick file, which is the
#    truncation measurement in its own right: how many of the positive
#    Asym(G0_ur) survive --gamma-rtol, per separation, before any job runs.
#
# 2. The 1 lambda Green blocks are in /home/pvirally/scratch/preload/. They will be,
#    for every separation the sweep reached -- the RSVD needed them. The bounds
#    points additionally want the (S, R) block, which the RSVD never applied, so
#    the first bounds job at a given separation may build one block before it
#    starts. That is minutes at 1 lambda and the time limits have room for it.
#
# 3. The 1/2 lambda blocks are NOT there: that sweep was generated and never
#    submitted. f_greens_0p5 builds them, into a preload directory of its own
#    under CAL_ROOT, and the three 0p5 RSVD points depend on it with afterok.
#
# The RSVD points run at q = 1, 3, 5 and each gets its own
# scratch subdirectory. file_prefix does not encode q, so without that the
# second and third runs would find the first one's output and skip the work they
# exist to measure -- and a --fresh run pointed at a production scratch directory
# would delete a production basis. Neither happens here; check anyway if you edit
# the --scratch paths.
# ---------------------------------------------------------------------------

set -u

CODE_DIR=/home/pvirally/Photonic-System-Channels/
CAL_ROOT=/home/pvirally/scratch/psc-calibration/
ROWS=$CAL_ROOT/rows_backfill
OUT=$CAL_ROOT/calibration_narval_backfill.csv

mkdir -p $CAL_ROOT/logs $CAL_ROOT/preload $CAL_ROOT/project $CAL_ROOT/scratch $ROWS
cd $CODE_DIR

if [ "${1:-}" = "--merge" ]; then
    n=$(ls -1 $ROWS/*.csv 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "No row files in $ROWS -- nothing to merge."
        exit 1
    fi
    head -n 1 $(ls -1 $ROWS/*.csv | head -n 1) > $OUT
    for f in $ROWS/*.csv; do tail -n +2 "$f" >> $OUT; done
    echo "Merged $n row file(s) into $OUT ($(( $(wc -l < $OUT) - 1 )) rows)."
    exit 0
fi

PROD_SCRATCH=/home/pvirally/scratch/Photonic-System-Channels/narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale/
PROD_PRELOAD=/home/pvirally/scratch/preload/
PRELOAD_0P5=$CAL_ROOT/backfill/preload_0p5
PICKS=$CAL_ROOT/backfill/picked_1l.txt
KEPT_TABLE=$CAL_ROOT/backfill/kept_by_sep_1l.csv
mkdir -p $CAL_ROOT/backfill $PRELOAD_0P5

if [ ! -d "$PROD_SCRATCH" ]; then
    echo "The 1 lambda sweep's scratch directory does not exist:"
    echo "  $PROD_SCRATCH"
    echo "The bounds points have nothing to read. Either the sweep never wrote"
    echo "there or scratch was purged; check the path against"
    echo "jobs/launch_narval_Ge1000_arxivV3_1x1x1_4000comps_50oversamples_32scale.sh and try again."
    exit 1
fi

if [ "${1:-}" = "--pick" ]; then
    rm -f "$PICKS"
    julia --project=. bench/pick_bounds_points.jl --scratch "$PROD_SCRATCH" --cells 32,32,32 --design rs --gamma-rtol 1.0e-12 --picks 4 --out "$PICKS" --table "$KEPT_TABLE"
    echo
    echo "Picked separations are in $PICKS and the full table in $KEPT_TABLE."
    echo "Re-run this script without --pick to submit."
    exit 0
fi

if [ ! -s "$PICKS" ]; then
    echo "Choosing bounds separations from the RSVD outputs on scratch..."
    julia --project=. bench/pick_bounds_points.jl --scratch "$PROD_SCRATCH" --cells 32,32,32 --design rs --gamma-rtol 1.0e-12 --picks 4 --out "$PICKS" --table "$KEPT_TABLE" || { echo "The picker failed; nothing was submitted."; exit 1; }
else
    echo "Reusing the existing pick list $PICKS (delete it, or pass --pick, to choose again):"
fi
cat "$PICKS"
echo

# The RSVD points run at the nearest pick, whose Green blocks are already in the
# preload directory because the production RSVD at that separation built them.
SEP_RSVD=$(head -n 1 "$PICKS" | awk '{print $1}')
if [ -z "$SEP_RSVD" ]; then
    echo "The pick list is empty; nothing was submitted."
    exit 1
fi
echo "RSVD points will run at separation $SEP_RSVD"
echo

echo "Submitting 7 + 4 picked bounds calibration points for narval (tier=backfill)"
echo "Each point writes its own row file under $ROWS"

jid_f_greens_0p5=$(sbatch --parsable \
    --job-name=psccal_f_greens_0p5 \
    --output=$CAL_ROOT/logs/f_greens_0p5_%j.out \
    --account=def-smolesky \
    --time=01:02:48 \
    --cpus-per-task=4 \
    --mem=12G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '4.25+0.0342557im' --design 'rs' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260819' --preload "$PRELOAD_0P5" --sep "$SEP_RSVD" --gpu -1 --root $CAL_ROOT --out $ROWS/f_greens_0p5.csv --cluster narval --note 'tier=backfill;label=f_greens_0p5'
EOF
)
sleep 0.05

jid_f_rsvd_0p5_q1=$(sbatch --parsable \
    --dependency=afterok:${jid_f_greens_0p5} \
    --job-name=psccal_f_rsvd_0p5_q1 \
    --output=$CAL_ROOT/logs/f_rsvd_0p5_q1_%j.out \
    --account=def-smolesky \
    --time=00:30:00 \
    --cpus-per-task=2 \
    --mem=10G \
    --gpus=a100_3g.20gb:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '4.25+0.0342557im' --design 'rs' --rank '4000' --oversamples '50' --power-iters '1' --seed '20260819' --scratch "$CAL_ROOT/backfill/rsvd_0p5_q1" --preload "$PRELOAD_0P5" --fresh --sep "$SEP_RSVD" --gpu 0 --root $CAL_ROOT --out $ROWS/f_rsvd_0p5_q1.csv --cluster narval --note 'tier=backfill;label=f_rsvd_0p5_q1'
EOF
)
sleep 0.05

jid_f_rsvd_0p5_q3=$(sbatch --parsable \
    --dependency=afterok:${jid_f_greens_0p5} \
    --job-name=psccal_f_rsvd_0p5_q3 \
    --output=$CAL_ROOT/logs/f_rsvd_0p5_q3_%j.out \
    --account=def-smolesky \
    --time=00:45:16 \
    --cpus-per-task=2 \
    --mem=10G \
    --gpus=a100_3g.20gb:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '4.25+0.0342557im' --design 'rs' --rank '4000' --oversamples '50' --power-iters '3' --seed '20260819' --scratch "$CAL_ROOT/backfill/rsvd_0p5_q3" --preload "$PRELOAD_0P5" --fresh --sep "$SEP_RSVD" --gpu 0 --root $CAL_ROOT --out $ROWS/f_rsvd_0p5_q3.csv --cluster narval --note 'tier=backfill;label=f_rsvd_0p5_q3'
EOF
)
sleep 0.05

jid_f_rsvd_0p5_q5=$(sbatch --parsable \
    --dependency=afterok:${jid_f_greens_0p5} \
    --job-name=psccal_f_rsvd_0p5_q5 \
    --output=$CAL_ROOT/logs/f_rsvd_0p5_q5_%j.out \
    --account=def-smolesky \
    --time=01:04:24 \
    --cpus-per-task=2 \
    --mem=10G \
    --gpus=a100_3g.20gb:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '4.25+0.0342557im' --design 'rs' --rank '4000' --oversamples '50' --power-iters '5' --seed '20260819' --scratch "$CAL_ROOT/backfill/rsvd_0p5_q5" --preload "$PRELOAD_0P5" --fresh --sep "$SEP_RSVD" --gpu 0 --root $CAL_ROOT --out $ROWS/f_rsvd_0p5_q5.csv --cluster narval --note 'tier=backfill;label=f_rsvd_0p5_q5'
EOF
)
sleep 0.05

jid_f_rsvd_1l_q1=$(sbatch --parsable \
    --job-name=psccal_f_rsvd_1l_q1 \
    --output=$CAL_ROOT/logs/f_rsvd_1l_q1_%j.out \
    --account=def-smolesky \
    --time=00:30:00 \
    --cpus-per-task=2 \
    --mem=42G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '4.25+0.0342557im' --design 'rs' --rank '4000' --oversamples '50' --power-iters '1' --seed '20260819' --scratch "$CAL_ROOT/backfill/rsvd_1l_q1" --preload "$PROD_PRELOAD" --fresh --sep "$SEP_RSVD" --gpu 0 --root $CAL_ROOT --out $ROWS/f_rsvd_1l_q1.csv --cluster narval --note 'tier=backfill;label=f_rsvd_1l_q1'
EOF
)
sleep 0.05

jid_f_rsvd_1l_q3=$(sbatch --parsable \
    --job-name=psccal_f_rsvd_1l_q3 \
    --output=$CAL_ROOT/logs/f_rsvd_1l_q3_%j.out \
    --account=def-smolesky \
    --time=00:43:01 \
    --cpus-per-task=2 \
    --mem=42G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '4.25+0.0342557im' --design 'rs' --rank '4000' --oversamples '50' --power-iters '3' --seed '20260819' --scratch "$CAL_ROOT/backfill/rsvd_1l_q3" --preload "$PROD_PRELOAD" --fresh --sep "$SEP_RSVD" --gpu 0 --root $CAL_ROOT --out $ROWS/f_rsvd_1l_q3.csv --cluster narval --note 'tier=backfill;label=f_rsvd_1l_q3'
EOF
)
sleep 0.05

jid_f_rsvd_1l_q5=$(sbatch --parsable \
    --job-name=psccal_f_rsvd_1l_q5 \
    --output=$CAL_ROOT/logs/f_rsvd_1l_q5_%j.out \
    --account=def-smolesky \
    --time=01:00:41 \
    --cpus-per-task=2 \
    --mem=42G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '4.25+0.0342557im' --design 'rs' --rank '4000' --oversamples '50' --power-iters '5' --seed '20260819' --scratch "$CAL_ROOT/backfill/rsvd_1l_q5" --preload "$PROD_PRELOAD" --fresh --sep "$SEP_RSVD" --gpu 0 --root $CAL_ROOT --out $ROWS/f_rsvd_1l_q5.csv --cluster narval --note 'tier=backfill;label=f_rsvd_1l_q5'
EOF
)
sleep 0.05


# ---------------------------------------------------------------------------
# A: bounds on the RSVD outputs already sitting in the 1 lambda sweep's scratch.
#
# One job per pick. Each reads PROD_SCRATCH (never writes to it: a sampled run
# writes no output JLD at all, and a full one writes only into --project, which
# points at the calibration tree) and asks for exactly what its own m needs.
#
# A "full" pick runs the whole outer loop, production exactly, output JLD and all.
# A "sampled" pick runs 4 blocks of 24 consecutive
# indices spread over 1:m, a few percent of the loop, and identifies the same
# coefficients. Which one a pick gets is decided by its m, by the picker.
# ---------------------------------------------------------------------------
pick_index=0
while read -r SEP KEPT STORED GPU MEM MODE LIMIT <&3; do
    [ -n "$SEP" ] || continue
    pick_index=$((pick_index + 1))
    LABEL=f_bounds_1l_p${pick_index}
    if [ "$MODE" = "full" ]; then BLOCKS=0; else BLOCKS=4; fi
    jid=$(sbatch --parsable \
        --job-name=psccal_${LABEL} \
        --output=$CAL_ROOT/logs/${LABEL}_%j.out \
        --account=def-smolesky \
        --time=$LIMIT \
        --cpus-per-task=2 \
        --mem=${MEM}G \
        --gpus=${GPU}:1 \
        --chdir=$CODE_DIR \
        --export=ALL \
        <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind stage_bounds --cells 32,32,32 --scale 1//32 --chi 4.25+0.0342557im --design rs --rank 4000 --oversamples 50 --power-iters 14 --seed 20260819 --gamma-rtol 1.0e-12 --outer-block-len 24 \
    --sep '$SEP' --outer-blocks $BLOCKS \
    --scratch "$PROD_SCRATCH" --preload "$PROD_PRELOAD" \
    --gpu 0 --root $CAL_ROOT --out $ROWS/${LABEL}.csv --cluster narval \
    --note 'tier=backfill;label=${LABEL};pick=${pick_index};picked_m=${KEPT};picked_stored=${STORED};mode=${MODE}'
EOF
)
    echo "  ${LABEL}  sep=${SEP}  m=${KEPT}  ${GPU}  ${MEM}G  ${LIMIT}  ${MODE}  -> job ${jid}"
    sleep 0.05
done 3< $PICKS
echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo
echo "When they have finished:"
echo
echo "  1. Any bounds job the 3 h limit cut short wrote no row, but its log holds"
echo "     the numbers. Replay it (per killed job; --summary first to look):"
echo
echo "     julia --project=. bench/measure.jl \\"
echo "         --parse-bounds-log $CAL_ROOT/logs/f_bounds_1l_p1_<jobid>.out \\"
echo "         --out $ROWS/f_bounds_1l_p1_fromlog.csv \\"
echo "         --cells 32,32,32 --scale 1//32 --sep <that pick's sep> \\"
echo "         --rank 4000 --cluster narval --jobid <jobid> \\"
echo "         --note 'tier=backfill;label=f_bounds_1l_p1;from_walltime_cut=1'"
echo
echo "  2. Merge the rows and copy them back:"
echo "     bash bench/launch_calibration_narval_backfill.sh --merge"
echo "     scp pvirally@narval.alliancecan.ca:$OUT bench/data/calibration_narval_backfill.csv"
echo "     scp pvirally@narval.alliancecan.ca:$KEPT_TABLE bench/data/"
echo
echo "  3. Refit. The new rows identify three things the old coefficients had no"
echo "     measurement for, and bench/fit.jl reports each one by name:"
echo "       bounds tau shape        grid evals and new whitenings per index"
echo "       bounds gamma truncation m as a power law in separation"
echo "       rsvd_pass_scale         measured / predicted per operator pass"
echo "     Until they are fitted the model keeps its old constants exactly, so a"
echo "     fit that reports them as 'not calibrated' has changed nothing."
echo "     julia bench/fit.jl"
echo
echo "  4. Regenerate the job scripts and read the new bounds requests:"
echo "     julia create_jobs.jl"

