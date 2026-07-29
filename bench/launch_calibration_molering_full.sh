#!/bin/bash
# Cost-model calibration for molering, tier=full.
# Generated 2026-07-29T12:26:52.685 by bench/plan.jl. Do not edit; regenerate instead.
#
# No scheduler here, so points run one at a time in the foreground. Each is
# allowed to fail without stopping the run; check the logs afterwards for
# any point whose row is missing from the CSV.
#
# Run it detached, it takes a while:
#   nohup bash launch_calibration_molering_full.sh > calibration.log 2>&1 &

set -u

CODE_DIR=/home/paulv/Projects/Photonic-System-Channels/
CAL_ROOT=/home/molering/fatmole/paulv/psc-calibration/
OUT=$CAL_ROOT/calibration_molering.csv

mkdir -p $CAL_ROOT/logs $CAL_ROOT/preload $CAL_ROOT/project $CAL_ROOT/scratch
cd $CODE_DIR

export PSC_CLUSTER=molering

total=125
index=0

index=$((index + 1))
echo "[$index/$total] g0self_l0p25"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l0p25' \
    > $CAL_ROOT/logs/g0self_l0p25.out 2>&1 \
    || echo "  FAILED: g0self_l0p25 (see $CAL_ROOT/logs/g0self_l0p25.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p25_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p25_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l0p25_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p25_sep0ss1 (see $CAL_ROOT/logs/g0ext_l0p25_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p25_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p25_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l0p25_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l0p25_sep1ss32 (see $CAL_ROOT/logs/g0ext_l0p25_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p25_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p25_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l0p25_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l0p25_sep1ss4 (see $CAL_ROOT/logs/g0ext_l0p25_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p25_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p25_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l0p25_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p25_sep1ss1 (see $CAL_ROOT/logs/g0ext_l0p25_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p25_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p25_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l0p25_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p25_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l0p25_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l0p25"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l0p25' \
    > $CAL_ROOT/logs/g0uu_l0p25.out 2>&1 \
    || echo "  FAILED: g0uu_l0p25 (see $CAL_ROOT/logs/g0uu_l0p25.out)"

index=$((index + 1))
echo "[$index/$total] g0self_l0p5"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l0p5' \
    > $CAL_ROOT/logs/g0self_l0p5.out 2>&1 \
    || echo "  FAILED: g0self_l0p5 (see $CAL_ROOT/logs/g0self_l0p5.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p5_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p5_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l0p5_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p5_sep0ss1 (see $CAL_ROOT/logs/g0ext_l0p5_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p5_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p5_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l0p5_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l0p5_sep1ss32 (see $CAL_ROOT/logs/g0ext_l0p5_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p5_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p5_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l0p5_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l0p5_sep1ss4 (see $CAL_ROOT/logs/g0ext_l0p5_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p5_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p5_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l0p5_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p5_sep1ss1 (see $CAL_ROOT/logs/g0ext_l0p5_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p5_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p5_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l0p5_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p5_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l0p5_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l0p5"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l0p5' \
    > $CAL_ROOT/logs/g0uu_l0p5.out 2>&1 \
    || echo "  FAILED: g0uu_l0p5 (see $CAL_ROOT/logs/g0uu_l0p5.out)"

index=$((index + 1))
echo "[$index/$total] g0self_l0p75"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l0p75' \
    > $CAL_ROOT/logs/g0self_l0p75.out 2>&1 \
    || echo "  FAILED: g0self_l0p75 (see $CAL_ROOT/logs/g0self_l0p75.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p75_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p75_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l0p75_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p75_sep0ss1 (see $CAL_ROOT/logs/g0ext_l0p75_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p75_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p75_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l0p75_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l0p75_sep1ss32 (see $CAL_ROOT/logs/g0ext_l0p75_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p75_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p75_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l0p75_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l0p75_sep1ss4 (see $CAL_ROOT/logs/g0ext_l0p75_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p75_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p75_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l0p75_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p75_sep1ss1 (see $CAL_ROOT/logs/g0ext_l0p75_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l0p75_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l0p75_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l0p75_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l0p75_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l0p75_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l0p75"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l0p75' \
    > $CAL_ROOT/logs/g0uu_l0p75.out 2>&1 \
    || echo "  FAILED: g0uu_l0p75 (see $CAL_ROOT/logs/g0uu_l0p75.out)"

index=$((index + 1))
echo "[$index/$total] g0self_l1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l1' \
    > $CAL_ROOT/logs/g0self_l1.out 2>&1 \
    || echo "  FAILED: g0self_l1 (see $CAL_ROOT/logs/g0self_l1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l1_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l1_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l1_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l1_sep0ss1 (see $CAL_ROOT/logs/g0ext_l1_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l1_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l1_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l1_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l1_sep1ss32 (see $CAL_ROOT/logs/g0ext_l1_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l1_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l1_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l1_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l1_sep1ss4 (see $CAL_ROOT/logs/g0ext_l1_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l1_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l1_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l1_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l1_sep1ss1 (see $CAL_ROOT/logs/g0ext_l1_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l1_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l1_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l1_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l1_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l1_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l1' \
    > $CAL_ROOT/logs/g0uu_l1.out 2>&1 \
    || echo "  FAILED: g0uu_l1 (see $CAL_ROOT/logs/g0uu_l1.out)"

index=$((index + 1))
echo "[$index/$total] g0self_l2agiso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l2agiso' \
    > $CAL_ROOT/logs/g0self_l2agiso.out 2>&1 \
    || echo "  FAILED: g0self_l2agiso (see $CAL_ROOT/logs/g0self_l2agiso.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2agiso_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2agiso_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l2agiso_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l2agiso_sep0ss1 (see $CAL_ROOT/logs/g0ext_l2agiso_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2agiso_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2agiso_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l2agiso_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l2agiso_sep1ss32 (see $CAL_ROOT/logs/g0ext_l2agiso_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2agiso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2agiso_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l2agiso_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l2agiso_sep1ss4 (see $CAL_ROOT/logs/g0ext_l2agiso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2agiso_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2agiso_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l2agiso_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l2agiso_sep1ss1 (see $CAL_ROOT/logs/g0ext_l2agiso_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2agiso_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2agiso_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l2agiso_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l2agiso_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l2agiso_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l2agiso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l2agiso' \
    > $CAL_ROOT/logs/g0uu_l2agiso.out 2>&1 \
    || echo "  FAILED: g0uu_l2agiso (see $CAL_ROOT/logs/g0uu_l2agiso.out)"

index=$((index + 1))
echo "[$index/$total] g0self_l3aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l3aniso' \
    > $CAL_ROOT/logs/g0self_l3aniso.out 2>&1 \
    || echo "  FAILED: g0self_l3aniso (see $CAL_ROOT/logs/g0self_l3aniso.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l3aniso_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l3aniso_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l3aniso_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l3aniso_sep0ss1 (see $CAL_ROOT/logs/g0ext_l3aniso_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l3aniso_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l3aniso_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l3aniso_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l3aniso_sep1ss32 (see $CAL_ROOT/logs/g0ext_l3aniso_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l3aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l3aniso_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l3aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l3aniso_sep1ss4 (see $CAL_ROOT/logs/g0ext_l3aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l3aniso_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l3aniso_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l3aniso_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l3aniso_sep1ss1 (see $CAL_ROOT/logs/g0ext_l3aniso_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l3aniso_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l3aniso_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l3aniso_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l3aniso_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l3aniso_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l3aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l3aniso' \
    > $CAL_ROOT/logs/g0uu_l3aniso.out 2>&1 \
    || echo "  FAILED: g0uu_l3aniso (see $CAL_ROOT/logs/g0uu_l3aniso.out)"

index=$((index + 1))
echo "[$index/$total] g0self_l4aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l4aniso' \
    > $CAL_ROOT/logs/g0self_l4aniso.out 2>&1 \
    || echo "  FAILED: g0self_l4aniso (see $CAL_ROOT/logs/g0self_l4aniso.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l4aniso_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l4aniso_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l4aniso_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l4aniso_sep0ss1 (see $CAL_ROOT/logs/g0ext_l4aniso_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l4aniso_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l4aniso_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l4aniso_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l4aniso_sep1ss32 (see $CAL_ROOT/logs/g0ext_l4aniso_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l4aniso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l4aniso_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l4aniso_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l4aniso_sep1ss4 (see $CAL_ROOT/logs/g0ext_l4aniso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l4aniso_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l4aniso_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l4aniso_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l4aniso_sep1ss1 (see $CAL_ROOT/logs/g0ext_l4aniso_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l4aniso_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l4aniso_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l4aniso_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l4aniso_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l4aniso_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l4aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l4aniso' \
    > $CAL_ROOT/logs/g0uu_l4aniso.out 2>&1 \
    || echo "  FAILED: g0uu_l4aniso (see $CAL_ROOT/logs/g0uu_l4aniso.out)"

index=$((index + 1))
echo "[$index/$total] g0self_l2iso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_self --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0self_l2iso' \
    > $CAL_ROOT/logs/g0self_l2iso.out 2>&1 \
    || echo "  FAILED: g0self_l2iso (see $CAL_ROOT/logs/g0self_l2iso.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2iso_sep0ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2iso_sep0ss1' \
    > $CAL_ROOT/logs/g0ext_l2iso_sep0ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l2iso_sep0ss1 (see $CAL_ROOT/logs/g0ext_l2iso_sep0ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2iso_sep1ss32"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2iso_sep1ss32' \
    > $CAL_ROOT/logs/g0ext_l2iso_sep1ss32.out 2>&1 \
    || echo "  FAILED: g0ext_l2iso_sep1ss32 (see $CAL_ROOT/logs/g0ext_l2iso_sep1ss32.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2iso_sep1ss4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2iso_sep1ss4' \
    > $CAL_ROOT/logs/g0ext_l2iso_sep1ss4.out 2>&1 \
    || echo "  FAILED: g0ext_l2iso_sep1ss4 (see $CAL_ROOT/logs/g0ext_l2iso_sep1ss4.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2iso_sep1ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2iso_sep1ss1' \
    > $CAL_ROOT/logs/g0ext_l2iso_sep1ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l2iso_sep1ss1 (see $CAL_ROOT/logs/g0ext_l2iso_sep1ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0ext_l2iso_sep1000ss1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0ext_l2iso_sep1000ss1' \
    > $CAL_ROOT/logs/g0ext_l2iso_sep1000ss1.out 2>&1 \
    || echo "  FAILED: g0ext_l2iso_sep1000ss1 (see $CAL_ROOT/logs/g0ext_l2iso_sep1000ss1.out)"

index=$((index + 1))
echo "[$index/$total] g0uu_l2iso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0uu_l2iso' \
    > $CAL_ROOT/logs/g0uu_l2iso.out 2>&1 \
    || echo "  FAILED: g0uu_l2iso (see $CAL_ROOT/logs/g0uu_l2iso.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t1"
export PSC_T0=$(date +%s)
julia --project=. -t 1 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t1' \
    > $CAL_ROOT/logs/g0threads_l1_t1.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t1 (see $CAL_ROOT/logs/g0threads_l1_t1.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t2"
export PSC_T0=$(date +%s)
julia --project=. -t 2 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t2' \
    > $CAL_ROOT/logs/g0threads_l1_t2.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t2 (see $CAL_ROOT/logs/g0threads_l1_t2.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t4"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t4' \
    > $CAL_ROOT/logs/g0threads_l1_t4.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t4 (see $CAL_ROOT/logs/g0threads_l1_t4.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t8"
export PSC_T0=$(date +%s)
julia --project=. -t 8 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t8' \
    > $CAL_ROOT/logs/g0threads_l1_t8.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t8 (see $CAL_ROOT/logs/g0threads_l1_t8.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t16"
export PSC_T0=$(date +%s)
julia --project=. -t 16 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t16' \
    > $CAL_ROOT/logs/g0threads_l1_t16.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t16 (see $CAL_ROOT/logs/g0threads_l1_t16.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t32"
export PSC_T0=$(date +%s)
julia --project=. -t 32 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t32' \
    > $CAL_ROOT/logs/g0threads_l1_t32.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t32 (see $CAL_ROOT/logs/g0threads_l1_t32.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t64"
export PSC_T0=$(date +%s)
julia --project=. -t 64 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t64' \
    > $CAL_ROOT/logs/g0threads_l1_t64.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t64 (see $CAL_ROOT/logs/g0threads_l1_t64.out)"

index=$((index + 1))
echo "[$index/$total] g0threads_l1_t128"
export PSC_T0=$(date +%s)
julia --project=. -t 128 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=g0threads_l1_t128' \
    > $CAL_ROOT/logs/g0threads_l1_t128.out 2>&1 \
    || echo "  FAILED: g0threads_l1_t128 (see $CAL_ROOT/logs/g0threads_l1_t128.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l0p25"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l0p25' \
    > $CAL_ROOT/logs/mvself_l0p25.out 2>&1 \
    || echo "  FAILED: mvself_l0p25 (see $CAL_ROOT/logs/mvself_l0p25.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l0p25"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l0p25' \
    > $CAL_ROOT/logs/mvext_l0p25.out 2>&1 \
    || echo "  FAILED: mvext_l0p25 (see $CAL_ROOT/logs/mvext_l0p25.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l0p25"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l0p25' \
    > $CAL_ROOT/logs/mvuu_l0p25.out 2>&1 \
    || echo "  FAILED: mvuu_l0p25 (see $CAL_ROOT/logs/mvuu_l0p25.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l0p5"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l0p5' \
    > $CAL_ROOT/logs/mvself_l0p5.out 2>&1 \
    || echo "  FAILED: mvself_l0p5 (see $CAL_ROOT/logs/mvself_l0p5.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l0p5"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l0p5' \
    > $CAL_ROOT/logs/mvext_l0p5.out 2>&1 \
    || echo "  FAILED: mvext_l0p5 (see $CAL_ROOT/logs/mvext_l0p5.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l0p5"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l0p5' \
    > $CAL_ROOT/logs/mvuu_l0p5.out 2>&1 \
    || echo "  FAILED: mvuu_l0p5 (see $CAL_ROOT/logs/mvuu_l0p5.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l0p75"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l0p75' \
    > $CAL_ROOT/logs/mvself_l0p75.out 2>&1 \
    || echo "  FAILED: mvself_l0p75 (see $CAL_ROOT/logs/mvself_l0p75.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l0p75"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l0p75' \
    > $CAL_ROOT/logs/mvext_l0p75.out 2>&1 \
    || echo "  FAILED: mvext_l0p75 (see $CAL_ROOT/logs/mvext_l0p75.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l0p75"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l0p75' \
    > $CAL_ROOT/logs/mvuu_l0p75.out 2>&1 \
    || echo "  FAILED: mvuu_l0p75 (see $CAL_ROOT/logs/mvuu_l0p75.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l1' \
    > $CAL_ROOT/logs/mvself_l1.out 2>&1 \
    || echo "  FAILED: mvself_l1 (see $CAL_ROOT/logs/mvself_l1.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l1' \
    > $CAL_ROOT/logs/mvext_l1.out 2>&1 \
    || echo "  FAILED: mvext_l1 (see $CAL_ROOT/logs/mvext_l1.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l1"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l1' \
    > $CAL_ROOT/logs/mvuu_l1.out 2>&1 \
    || echo "  FAILED: mvuu_l1 (see $CAL_ROOT/logs/mvuu_l1.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l2agiso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l2agiso' \
    > $CAL_ROOT/logs/mvself_l2agiso.out 2>&1 \
    || echo "  FAILED: mvself_l2agiso (see $CAL_ROOT/logs/mvself_l2agiso.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l2agiso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l2agiso' \
    > $CAL_ROOT/logs/mvext_l2agiso.out 2>&1 \
    || echo "  FAILED: mvext_l2agiso (see $CAL_ROOT/logs/mvext_l2agiso.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l2agiso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l2agiso' \
    > $CAL_ROOT/logs/mvuu_l2agiso.out 2>&1 \
    || echo "  FAILED: mvuu_l2agiso (see $CAL_ROOT/logs/mvuu_l2agiso.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l3aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l3aniso' \
    > $CAL_ROOT/logs/mvself_l3aniso.out 2>&1 \
    || echo "  FAILED: mvself_l3aniso (see $CAL_ROOT/logs/mvself_l3aniso.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l3aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l3aniso' \
    > $CAL_ROOT/logs/mvext_l3aniso.out 2>&1 \
    || echo "  FAILED: mvext_l3aniso (see $CAL_ROOT/logs/mvext_l3aniso.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l3aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l3aniso' \
    > $CAL_ROOT/logs/mvuu_l3aniso.out 2>&1 \
    || echo "  FAILED: mvuu_l3aniso (see $CAL_ROOT/logs/mvuu_l3aniso.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l4aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l4aniso' \
    > $CAL_ROOT/logs/mvself_l4aniso.out 2>&1 \
    || echo "  FAILED: mvself_l4aniso (see $CAL_ROOT/logs/mvself_l4aniso.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l4aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l4aniso' \
    > $CAL_ROOT/logs/mvext_l4aniso.out 2>&1 \
    || echo "  FAILED: mvext_l4aniso (see $CAL_ROOT/logs/mvext_l4aniso.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l4aniso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l4aniso' \
    > $CAL_ROOT/logs/mvuu_l4aniso.out 2>&1 \
    || echo "  FAILED: mvuu_l4aniso (see $CAL_ROOT/logs/mvuu_l4aniso.out)"

index=$((index + 1))
echo "[$index/$total] mvself_l2iso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvself_l2iso' \
    > $CAL_ROOT/logs/mvself_l2iso.out 2>&1 \
    || echo "  FAILED: mvself_l2iso (see $CAL_ROOT/logs/mvself_l2iso.out)"

index=$((index + 1))
echo "[$index/$total] mvext_l2iso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvext_l2iso' \
    > $CAL_ROOT/logs/mvext_l2iso.out 2>&1 \
    || echo "  FAILED: mvext_l2iso (see $CAL_ROOT/logs/mvext_l2iso.out)"

index=$((index + 1))
echo "[$index/$total] mvuu_l2iso"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=mvuu_l2iso' \
    > $CAL_ROOT/logs/mvuu_l2iso.out 2>&1 \
    || echo "  FAILED: mvuu_l2iso (see $CAL_ROOT/logs/mvuu_l2iso.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p25_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p25_c128' \
    > $CAL_ROOT/logs/dense_l0p25_c128.out 2>&1 \
    || echo "  FAILED: dense_l0p25_c128 (see $CAL_ROOT/logs/dense_l0p25_c128.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p25_c512"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p25_c512' \
    > $CAL_ROOT/logs/dense_l0p25_c512.out 2>&1 \
    || echo "  FAILED: dense_l0p25_c512 (see $CAL_ROOT/logs/dense_l0p25_c512.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p25_c1400"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p25_c1400' \
    > $CAL_ROOT/logs/dense_l0p25_c1400.out 2>&1 \
    || echo "  FAILED: dense_l0p25_c1400 (see $CAL_ROOT/logs/dense_l0p25_c1400.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p25_c2800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '2800' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p25_c2800' \
    > $CAL_ROOT/logs/dense_l0p25_c2800.out 2>&1 \
    || echo "  FAILED: dense_l0p25_c2800 (see $CAL_ROOT/logs/dense_l0p25_c2800.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p5_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p5_c128' \
    > $CAL_ROOT/logs/dense_l0p5_c128.out 2>&1 \
    || echo "  FAILED: dense_l0p5_c128 (see $CAL_ROOT/logs/dense_l0p5_c128.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p5_c512"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p5_c512' \
    > $CAL_ROOT/logs/dense_l0p5_c512.out 2>&1 \
    || echo "  FAILED: dense_l0p5_c512 (see $CAL_ROOT/logs/dense_l0p5_c512.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p5_c1400"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p5_c1400' \
    > $CAL_ROOT/logs/dense_l0p5_c1400.out 2>&1 \
    || echo "  FAILED: dense_l0p5_c1400 (see $CAL_ROOT/logs/dense_l0p5_c1400.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p5_c2800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '2800' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p5_c2800' \
    > $CAL_ROOT/logs/dense_l0p5_c2800.out 2>&1 \
    || echo "  FAILED: dense_l0p5_c2800 (see $CAL_ROOT/logs/dense_l0p5_c2800.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p75_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p75_c128' \
    > $CAL_ROOT/logs/dense_l0p75_c128.out 2>&1 \
    || echo "  FAILED: dense_l0p75_c128 (see $CAL_ROOT/logs/dense_l0p75_c128.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p75_c512"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p75_c512' \
    > $CAL_ROOT/logs/dense_l0p75_c512.out 2>&1 \
    || echo "  FAILED: dense_l0p75_c512 (see $CAL_ROOT/logs/dense_l0p75_c512.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p75_c1400"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p75_c1400' \
    > $CAL_ROOT/logs/dense_l0p75_c1400.out 2>&1 \
    || echo "  FAILED: dense_l0p75_c1400 (see $CAL_ROOT/logs/dense_l0p75_c1400.out)"

index=$((index + 1))
echo "[$index/$total] dense_l0p75_c2800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '2800' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l0p75_c2800' \
    > $CAL_ROOT/logs/dense_l0p75_c2800.out 2>&1 \
    || echo "  FAILED: dense_l0p75_c2800 (see $CAL_ROOT/logs/dense_l0p75_c2800.out)"

index=$((index + 1))
echo "[$index/$total] dense_l1_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '196608' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l1_c128' \
    > $CAL_ROOT/logs/dense_l1_c128.out 2>&1 \
    || echo "  FAILED: dense_l1_c128 (see $CAL_ROOT/logs/dense_l1_c128.out)"

index=$((index + 1))
echo "[$index/$total] dense_l1_c512"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '196608' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l1_c512' \
    > $CAL_ROOT/logs/dense_l1_c512.out 2>&1 \
    || echo "  FAILED: dense_l1_c512 (see $CAL_ROOT/logs/dense_l1_c512.out)"

index=$((index + 1))
echo "[$index/$total] dense_l1_c1400"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '196608' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l1_c1400' \
    > $CAL_ROOT/logs/dense_l1_c1400.out 2>&1 \
    || echo "  FAILED: dense_l1_c1400 (see $CAL_ROOT/logs/dense_l1_c1400.out)"

index=$((index + 1))
echo "[$index/$total] dense_l2agiso_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '393216' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l2agiso_c128' \
    > $CAL_ROOT/logs/dense_l2agiso_c128.out 2>&1 \
    || echo "  FAILED: dense_l2agiso_c128 (see $CAL_ROOT/logs/dense_l2agiso_c128.out)"

index=$((index + 1))
echo "[$index/$total] dense_l2agiso_c512"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '393216' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l2agiso_c512' \
    > $CAL_ROOT/logs/dense_l2agiso_c512.out 2>&1 \
    || echo "  FAILED: dense_l2agiso_c512 (see $CAL_ROOT/logs/dense_l2agiso_c512.out)"

index=$((index + 1))
echo "[$index/$total] dense_l3aniso_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '589824' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l3aniso_c128' \
    > $CAL_ROOT/logs/dense_l3aniso_c128.out 2>&1 \
    || echo "  FAILED: dense_l3aniso_c128 (see $CAL_ROOT/logs/dense_l3aniso_c128.out)"

index=$((index + 1))
echo "[$index/$total] dense_l3aniso_c512"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '589824' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l3aniso_c512' \
    > $CAL_ROOT/logs/dense_l3aniso_c512.out 2>&1 \
    || echo "  FAILED: dense_l3aniso_c512 (see $CAL_ROOT/logs/dense_l3aniso_c512.out)"

index=$((index + 1))
echo "[$index/$total] dense_l4aniso_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '786432' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l4aniso_c128' \
    > $CAL_ROOT/logs/dense_l4aniso_c128.out 2>&1 \
    || echo "  FAILED: dense_l4aniso_c128 (see $CAL_ROOT/logs/dense_l4aniso_c128.out)"

index=$((index + 1))
echo "[$index/$total] dense_l4aniso_c512"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '786432' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l4aniso_c512' \
    > $CAL_ROOT/logs/dense_l4aniso_c512.out 2>&1 \
    || echo "  FAILED: dense_l4aniso_c512 (see $CAL_ROOT/logs/dense_l4aniso_c512.out)"

index=$((index + 1))
echo "[$index/$total] dense_l2iso_c128"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind dense --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '1572864' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=dense_l2iso_c128' \
    > $CAL_ROOT/logs/dense_l2iso_c128.out 2>&1 \
    || echo "  FAILED: dense_l2iso_c128 (see $CAL_ROOT/logs/dense_l2iso_c128.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p25_k256"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p25_k256' \
    > $CAL_ROOT/logs/boundscore_l0p25_k256.out 2>&1 \
    || echo "  FAILED: boundscore_l0p25_k256 (see $CAL_ROOT/logs/boundscore_l0p25_k256.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p25_k800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p25_k800' \
    > $CAL_ROOT/logs/boundscore_l0p25_k800.out 2>&1 \
    || echo "  FAILED: boundscore_l0p25_k800 (see $CAL_ROOT/logs/boundscore_l0p25_k800.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p25_k1350"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '1350' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p25_k1350' \
    > $CAL_ROOT/logs/boundscore_l0p25_k1350.out 2>&1 \
    || echo "  FAILED: boundscore_l0p25_k1350 (see $CAL_ROOT/logs/boundscore_l0p25_k1350.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p5_k256"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p5_k256' \
    > $CAL_ROOT/logs/boundscore_l0p5_k256.out 2>&1 \
    || echo "  FAILED: boundscore_l0p5_k256 (see $CAL_ROOT/logs/boundscore_l0p5_k256.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p5_k800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p5_k800' \
    > $CAL_ROOT/logs/boundscore_l0p5_k800.out 2>&1 \
    || echo "  FAILED: boundscore_l0p5_k800 (see $CAL_ROOT/logs/boundscore_l0p5_k800.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p5_k1350"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '1350' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p5_k1350' \
    > $CAL_ROOT/logs/boundscore_l0p5_k1350.out 2>&1 \
    || echo "  FAILED: boundscore_l0p5_k1350 (see $CAL_ROOT/logs/boundscore_l0p5_k1350.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p75_k256"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p75_k256' \
    > $CAL_ROOT/logs/boundscore_l0p75_k256.out 2>&1 \
    || echo "  FAILED: boundscore_l0p75_k256 (see $CAL_ROOT/logs/boundscore_l0p75_k256.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p75_k800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p75_k800' \
    > $CAL_ROOT/logs/boundscore_l0p75_k800.out 2>&1 \
    || echo "  FAILED: boundscore_l0p75_k800 (see $CAL_ROOT/logs/boundscore_l0p75_k800.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l0p75_k1350"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '1350' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l0p75_k1350' \
    > $CAL_ROOT/logs/boundscore_l0p75_k1350.out 2>&1 \
    || echo "  FAILED: boundscore_l0p75_k1350 (see $CAL_ROOT/logs/boundscore_l0p75_k1350.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l1_k256"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l1_k256' \
    > $CAL_ROOT/logs/boundscore_l1_k256.out 2>&1 \
    || echo "  FAILED: boundscore_l1_k256 (see $CAL_ROOT/logs/boundscore_l1_k256.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l1_k800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l1_k800' \
    > $CAL_ROOT/logs/boundscore_l1_k800.out 2>&1 \
    || echo "  FAILED: boundscore_l1_k800 (see $CAL_ROOT/logs/boundscore_l1_k800.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l2agiso_k256"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l2agiso_k256' \
    > $CAL_ROOT/logs/boundscore_l2agiso_k256.out 2>&1 \
    || echo "  FAILED: boundscore_l2agiso_k256 (see $CAL_ROOT/logs/boundscore_l2agiso_k256.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l2agiso_k800"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l2agiso_k800' \
    > $CAL_ROOT/logs/boundscore_l2agiso_k800.out 2>&1 \
    || echo "  FAILED: boundscore_l2agiso_k800 (see $CAL_ROOT/logs/boundscore_l2agiso_k800.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l3aniso_k256"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l3aniso_k256' \
    > $CAL_ROOT/logs/boundscore_l3aniso_k256.out 2>&1 \
    || echo "  FAILED: boundscore_l3aniso_k256 (see $CAL_ROOT/logs/boundscore_l3aniso_k256.out)"

index=$((index + 1))
echo "[$index/$total] boundscore_l4aniso_k256"
export PSC_T0=$(date +%s)
julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster molering --note 'tier=full;label=boundscore_l4aniso_k256' \
    > $CAL_ROOT/logs/boundscore_l4aniso_k256.out 2>&1 \
    || echo "  FAILED: boundscore_l4aniso_k256 (see $CAL_ROOT/logs/boundscore_l4aniso_k256.out)"

echo
echo "Done. Copy the CSV back to your laptop:"
echo "  scp paulv@molering:$OUT bench/data/"

