#!/bin/bash
# Cost-model calibration for molering, tier=memory.
# Generated 2026-08-05T12:30:19.206 by bench/plan.jl. Do not edit; regenerate instead.
#
# No scheduler here, so points run one at a time in the foreground. Each is
# allowed to fail without stopping the run; check the logs afterwards for
# any point whose row is missing from the CSV.
#
# Run it detached, it takes a while:
#   nohup bash launch_calibration_molering_memory.sh > calibration.log 2>&1 &

set -u

CODE_DIR=/home/paulv/Projects/Photonic-System-Channels/
CAL_ROOT=/home/molering/fatmole/paulv/psc-calibration/
ROWS=$CAL_ROOT/rows
OUT=$CAL_ROOT/calibration_molering.csv

mkdir -p $CAL_ROOT/logs $CAL_ROOT/preload $CAL_ROOT/project $CAL_ROOT/scratch $ROWS
cd $CODE_DIR

export PSC_CLUSTER=molering

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

total=16
index=0

index=$((index + 1))
echo "[$index/$total] memgreens_l0p25"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l0p25.csv --cluster molering --note 'tier=memory;label=memgreens_l0p25' \
    > $CAL_ROOT/logs/memgreens_l0p25.out 2>&1 \
    || echo "  FAILED: memgreens_l0p25 (see $CAL_ROOT/logs/memgreens_l0p25.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l0p25"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l0p25.csv --cluster molering --note 'tier=memory;label=memrsvd_l0p25' \
    > $CAL_ROOT/logs/memrsvd_l0p25.out 2>&1 \
    || echo "  FAILED: memrsvd_l0p25 (see $CAL_ROOT/logs/memrsvd_l0p25.out)"

index=$((index + 1))
echo "[$index/$total] memgreens_l0p5"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l0p5.csv --cluster molering --note 'tier=memory;label=memgreens_l0p5' \
    > $CAL_ROOT/logs/memgreens_l0p5.out 2>&1 \
    || echo "  FAILED: memgreens_l0p5 (see $CAL_ROOT/logs/memgreens_l0p5.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l0p5"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l0p5.csv --cluster molering --note 'tier=memory;label=memrsvd_l0p5' \
    > $CAL_ROOT/logs/memrsvd_l0p5.out 2>&1 \
    || echo "  FAILED: memrsvd_l0p5 (see $CAL_ROOT/logs/memrsvd_l0p5.out)"

index=$((index + 1))
echo "[$index/$total] memgreens_l0p75"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l0p75.csv --cluster molering --note 'tier=memory;label=memgreens_l0p75' \
    > $CAL_ROOT/logs/memgreens_l0p75.out 2>&1 \
    || echo "  FAILED: memgreens_l0p75 (see $CAL_ROOT/logs/memgreens_l0p75.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l0p75"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l0p75.csv --cluster molering --note 'tier=memory;label=memrsvd_l0p75' \
    > $CAL_ROOT/logs/memrsvd_l0p75.out 2>&1 \
    || echo "  FAILED: memrsvd_l0p75 (see $CAL_ROOT/logs/memrsvd_l0p75.out)"

index=$((index + 1))
echo "[$index/$total] memgreens_l1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l1.csv --cluster molering --note 'tier=memory;label=memgreens_l1' \
    > $CAL_ROOT/logs/memgreens_l1.out 2>&1 \
    || echo "  FAILED: memgreens_l1 (see $CAL_ROOT/logs/memgreens_l1.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l1.csv --cluster molering --note 'tier=memory;label=memrsvd_l1' \
    > $CAL_ROOT/logs/memrsvd_l1.out 2>&1 \
    || echo "  FAILED: memrsvd_l1 (see $CAL_ROOT/logs/memrsvd_l1.out)"

index=$((index + 1))
echo "[$index/$total] memgreens_l2agiso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l2agiso.csv --cluster molering --note 'tier=memory;label=memgreens_l2agiso' \
    > $CAL_ROOT/logs/memgreens_l2agiso.out 2>&1 \
    || echo "  FAILED: memgreens_l2agiso (see $CAL_ROOT/logs/memgreens_l2agiso.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l2agiso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l2agiso.csv --cluster molering --note 'tier=memory;label=memrsvd_l2agiso' \
    > $CAL_ROOT/logs/memrsvd_l2agiso.out 2>&1 \
    || echo "  FAILED: memrsvd_l2agiso (see $CAL_ROOT/logs/memrsvd_l2agiso.out)"

index=$((index + 1))
echo "[$index/$total] memgreens_l3aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l3aniso.csv --cluster molering --note 'tier=memory;label=memgreens_l3aniso' \
    > $CAL_ROOT/logs/memgreens_l3aniso.out 2>&1 \
    || echo "  FAILED: memgreens_l3aniso (see $CAL_ROOT/logs/memgreens_l3aniso.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l3aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l3aniso.csv --cluster molering --note 'tier=memory;label=memrsvd_l3aniso' \
    > $CAL_ROOT/logs/memrsvd_l3aniso.out 2>&1 \
    || echo "  FAILED: memrsvd_l3aniso (see $CAL_ROOT/logs/memrsvd_l3aniso.out)"

index=$((index + 1))
echo "[$index/$total] memgreens_l4aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l4aniso.csv --cluster molering --note 'tier=memory;label=memgreens_l4aniso' \
    > $CAL_ROOT/logs/memgreens_l4aniso.out 2>&1 \
    || echo "  FAILED: memgreens_l4aniso (see $CAL_ROOT/logs/memgreens_l4aniso.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l4aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l4aniso.csv --cluster molering --note 'tier=memory;label=memrsvd_l4aniso' \
    > $CAL_ROOT/logs/memrsvd_l4aniso.out 2>&1 \
    || echo "  FAILED: memrsvd_l4aniso (see $CAL_ROOT/logs/memrsvd_l4aniso.out)"

index=$((index + 1))
echo "[$index/$total] memgreens_l2iso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l2iso.csv --cluster molering --note 'tier=memory;label=memgreens_l2iso' \
    > $CAL_ROOT/logs/memgreens_l2iso.out 2>&1 \
    || echo "  FAILED: memgreens_l2iso (see $CAL_ROOT/logs/memgreens_l2iso.out)"

index=$((index + 1))
echo "[$index/$total] memrsvd_l2iso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l2iso.csv --cluster molering --note 'tier=memory;label=memrsvd_l2iso' \
    > $CAL_ROOT/logs/memrsvd_l2iso.out 2>&1 \
    || echo "  FAILED: memrsvd_l2iso (see $CAL_ROOT/logs/memrsvd_l2iso.out)"

echo
echo "Done. Merge the per-point rows, then copy the result back:"
echo "  bash bench/launch_calibration_molering_memory.sh --merge"
echo "  scp paulv@molering:$OUT bench/data/"

