#!/bin/bash
# Cost-model calibration for molering, tier=validate.
# Generated 2026-07-30T13:36:59.938 by bench/plan.jl. Do not edit; regenerate instead.
#
# No scheduler here, so points run one at a time in the foreground. Each is
# allowed to fail without stopping the run; check the logs afterwards for
# any point whose row is missing from the CSV.
#
# Run it detached, it takes a while:
#   nohup bash launch_calibration_molering_validate.sh > calibration.log 2>&1 &

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

total=30
index=0

index=$((index + 1))
echo "[$index/$total] stagegreens_l0p25_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p25_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l0p25_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l0p25_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l0p25_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l0p25_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l0p25_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p25_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l0p25_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l0p25_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l0p25_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l0p25_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l0p25_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p25_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l0p25_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l0p25_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l0p25_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l0p25_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l0p25_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p25_sep0ss1.csv --cluster molering --note 'tier=validate;label=stagegreens_l0p25_sep0ss1' \
    > $CAL_ROOT/logs/stagegreens_l0p25_sep0ss1.out 2>&1 \
    || echo "  FAILED: stagegreens_l0p25_sep0ss1 (see $CAL_ROOT/logs/stagegreens_l0p25_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l0p25_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p25_sep0ss1.csv --cluster molering --note 'tier=validate;label=stagersvd_l0p25_sep0ss1' \
    > $CAL_ROOT/logs/stagersvd_l0p25_sep0ss1.out 2>&1 \
    || echo "  FAILED: stagersvd_l0p25_sep0ss1 (see $CAL_ROOT/logs/stagersvd_l0p25_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l0p25_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p25_sep0ss1.csv --cluster molering --note 'tier=validate;label=stagebounds_l0p25_sep0ss1' \
    > $CAL_ROOT/logs/stagebounds_l0p25_sep0ss1.out 2>&1 \
    || echo "  FAILED: stagebounds_l0p25_sep0ss1 (see $CAL_ROOT/logs/stagebounds_l0p25_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l0p5_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p5_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l0p5_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l0p5_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l0p5_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l0p5_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l0p5_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p5_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l0p5_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l0p5_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l0p5_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l0p5_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l0p5_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p5_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l0p5_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l0p5_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l0p5_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l0p5_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l0p5_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p5_sep0ss1.csv --cluster molering --note 'tier=validate;label=stagegreens_l0p5_sep0ss1' \
    > $CAL_ROOT/logs/stagegreens_l0p5_sep0ss1.out 2>&1 \
    || echo "  FAILED: stagegreens_l0p5_sep0ss1 (see $CAL_ROOT/logs/stagegreens_l0p5_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l0p5_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p5_sep0ss1.csv --cluster molering --note 'tier=validate;label=stagersvd_l0p5_sep0ss1' \
    > $CAL_ROOT/logs/stagersvd_l0p5_sep0ss1.out 2>&1 \
    || echo "  FAILED: stagersvd_l0p5_sep0ss1 (see $CAL_ROOT/logs/stagersvd_l0p5_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l0p5_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p5_sep0ss1.csv --cluster molering --note 'tier=validate;label=stagebounds_l0p5_sep0ss1' \
    > $CAL_ROOT/logs/stagebounds_l0p5_sep0ss1.out 2>&1 \
    || echo "  FAILED: stagebounds_l0p5_sep0ss1 (see $CAL_ROOT/logs/stagebounds_l0p5_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l0p75_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p75_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l0p75_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l0p75_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l0p75_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l0p75_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l0p75_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p75_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l0p75_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l0p75_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l0p75_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l0p75_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l0p75_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p75_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l0p75_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l0p75_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l0p75_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l0p75_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l1_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l1_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l1_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l1_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l1_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l1_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l1_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l1_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l1_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l1_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l1_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l1_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l1_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l1_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l1_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l1_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l1_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l1_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l2agiso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l2agiso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l2agiso_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l2agiso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l2agiso_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l2agiso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l2agiso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l2agiso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l2agiso_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l2agiso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l2agiso_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l2agiso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l2agiso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l2agiso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l2agiso_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l2agiso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l2agiso_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l2agiso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l3aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l3aniso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l3aniso_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l3aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l3aniso_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l3aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l3aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l3aniso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l3aniso_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l3aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l3aniso_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l3aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l3aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l3aniso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l3aniso_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l3aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l3aniso_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l3aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l4aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l4aniso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l4aniso_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l4aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l4aniso_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l4aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l4aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l4aniso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l4aniso_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l4aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l4aniso_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l4aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l4aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l4aniso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l4aniso_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l4aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l4aniso_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l4aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagegreens_l2iso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l2iso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagegreens_l2iso_sep1ss4' \
    > $CAL_ROOT/logs/stagegreens_l2iso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagegreens_l2iso_sep1ss4 (see $CAL_ROOT/logs/stagegreens_l2iso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagersvd_l2iso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l2iso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagersvd_l2iso_sep1ss4' \
    > $CAL_ROOT/logs/stagersvd_l2iso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagersvd_l2iso_sep1ss4 (see $CAL_ROOT/logs/stagersvd_l2iso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] stagebounds_l2iso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l2iso_sep1ss4.csv --cluster molering --note 'tier=validate;label=stagebounds_l2iso_sep1ss4' \
    > $CAL_ROOT/logs/stagebounds_l2iso_sep1ss4.out 2>&1 \
    || echo "  FAILED: stagebounds_l2iso_sep1ss4 (see $CAL_ROOT/logs/stagebounds_l2iso_sep1ss4.out)"

echo
echo "Done. Merge the per-point rows, then copy the result back:"
echo "  bash bench/launch_calibration_molering_validate.sh --merge"
echo "  scp paulv@molering:$OUT bench/data/"

