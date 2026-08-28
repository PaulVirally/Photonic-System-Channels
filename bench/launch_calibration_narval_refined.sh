#!/bin/bash
# Cost-model calibration for narval, tier=refined.
# Generated 2026-08-27T10:41:54.177 by bench/plan_refined.jl. Do not edit; regenerate instead.
#
# Every point is its own job: one point running out of memory or time must
# not take the rest of the calibration with it. Each writes its own row file,
# so partial results are always usable.
#
# Submit:  bash <this script>
# Collect: bash <this script> --merge
#
# What this tier measures
#
#   refg_*        the composite Green build, at two gaps that land on two
#                 different GAP_REFINEMENT_TABLE entries and on two body sizes,
#                 plus the same two geometries at g = 6 where nothing is refined.
#                 Fits g0_sandwich_scale and g0_partition_scale.
#   refmv_*       one composite matvec each of G0_rs, G0_rr and the universe
#                 block operator. Fits mv_composite_scale.
#   refrsvd_*     one refined RSVD end to end, at k = 800 and q = 6 so that it
#                 fits the box. Checks the assembled model, and is the only point
#                 that sees the folded Asym(G0_rr) applies of the hasmethod shim
#                 in src/rsvd.jl.
#   refbounds_*   one refined bounds front end, outer loop sampled, reading the
#                 RSVD point's scratch.
#
# Every point carries --refine, which is what puts bench/point.jl on the refined
# path; it is off by default there even though src/common.jl now refines.
#
# Before submitting, on the LOGIN node:
#
#   module load StdEnv/2023 julia/1.12.5 cuda/12.2
#   cd <code dir>
#   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
#
# The four refg points and the two controls are independent and can be submitted
# on their own; the device points depend on refg_c32_g1 having written the
# preload directory, and refbounds depends on refrsvd.

set -u

CODE_DIR=/home/pvirally/Photonic-System-Channels/
CAL_ROOT=/home/pvirally/scratch/psc-calibration/
ROWS=$CAL_ROOT/rows_refined
OUT=$CAL_ROOT/calibration_narval_refined.csv

mkdir -p $CAL_ROOT/logs $CAL_ROOT/preload $CAL_ROOT/project $CAL_ROOT/scratch $ROWS
cd $CODE_DIR

if [ "${1:-}" = "--merge" ]; then
    n=$(ls -1 $ROWS/*.csv 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "No row files in $ROWS, nothing to merge."
        exit 1
    fi
    head -n 1 $(ls -1 $ROWS/*.csv | head -n 1) > $OUT
    for f in $ROWS/*.csv; do tail -n +2 "$f" >> $OUT; done
    echo "Merged $n row file(s) into $OUT ($(( $(wc -l < $OUT) - 1 )) rows)."
    exit 0
fi

echo "Submitting 11 calibration points for narval (tier=refined)"
echo "Each point writes its own row file under $ROWS"

jid_refg_c8_g1=$(sbatch --parsable \
    --job-name=psccal_refg_c8_g1 \
    --output=$CAL_ROOT/logs/refg_c8_g1_%j.out \
    --account=def-smolesky \
    --time=01:06:51 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c8_g1.csv --cluster narval --note 'tier=refined;label=refg_c8_g1'
EOF
)
sleep 0.05

jid_refg_c8_g3=$(sbatch --parsable \
    --job-name=psccal_refg_c8_g3 \
    --output=$CAL_ROOT/logs/refg_c8_g3_%j.out \
    --account=def-smolesky \
    --time=01:31:51 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '3//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c8_g3.csv --cluster narval --note 'tier=refined;label=refg_c8_g3'
EOF
)
sleep 0.05

jid_refg_c8_g6=$(sbatch --parsable \
    --job-name=psccal_refg_c8_g6 \
    --output=$CAL_ROOT/logs/refg_c8_g6_%j.out \
    --account=def-smolesky \
    --time=00:45:41 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c8_g6.csv --cluster narval --note 'tier=refined;label=refg_c8_g6'
EOF
)
sleep 0.05

jid_refg_c32_g1=$(sbatch --parsable \
    --job-name=psccal_refg_c32_g1 \
    --output=$CAL_ROOT/logs/refg_c32_g1_%j.out \
    --account=def-smolesky \
    --time=02:39:40 \
    --cpus-per-task=12 \
    --mem=16G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c32_g1.csv --cluster narval --note 'tier=refined;label=refg_c32_g1'
EOF
)
sleep 0.05

jid_refg_c32_g3=$(sbatch --parsable \
    --job-name=psccal_refg_c32_g3 \
    --output=$CAL_ROOT/logs/refg_c32_g3_%j.out \
    --account=def-smolesky \
    --time=01:58:52 \
    --cpus-per-task=12 \
    --mem=13G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '3//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c32_g3.csv --cluster narval --note 'tier=refined;label=refg_c32_g3'
EOF
)
sleep 0.05

jid_refg_c32_g6=$(sbatch --parsable \
    --job-name=psccal_refg_c32_g6 \
    --output=$CAL_ROOT/logs/refg_c32_g6_%j.out \
    --account=def-smolesky \
    --time=00:50:22 \
    --cpus-per-task=12 \
    --mem=12G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c32_g6.csv --cluster narval --note 'tier=refined;label=refg_c32_g6'
EOF
)
sleep 0.05

jid_refmv_ext_c32_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c32_g1} \
    --job-name=psccal_refmv_ext_c32_g1 \
    --output=$CAL_ROOT/logs/refmv_ext_c32_g1_%j.out \
    --account=def-smolesky \
    --time=00:10:00 \
    --cpus-per-task=12 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind matvec_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_ext_c32_g1.csv --cluster narval --note 'tier=refined;label=refmv_ext_c32_g1'
EOF
)
sleep 0.05

jid_refmv_self_c32_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c32_g1} \
    --job-name=psccal_refmv_self_c32_g1 \
    --output=$CAL_ROOT/logs/refmv_self_c32_g1_%j.out \
    --account=def-smolesky \
    --time=00:10:00 \
    --cpus-per-task=12 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind matvec_self --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_self_c32_g1.csv --cluster narval --note 'tier=refined;label=refmv_self_c32_g1'
EOF
)
sleep 0.05

jid_refmv_uu_c32_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c32_g1} \
    --job-name=psccal_refmv_uu_c32_g1 \
    --output=$CAL_ROOT/logs/refmv_uu_c32_g1_%j.out \
    --account=def-smolesky \
    --time=00:10:00 \
    --cpus-per-task=12 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind matvec_uu --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_uu_c32_g1.csv --cluster narval --note 'tier=refined;label=refmv_uu_c32_g1'
EOF
)
sleep 0.05

jid_refrsvd_c32_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c32_g1} \
    --job-name=psccal_refrsvd_c32_g1 \
    --output=$CAL_ROOT/logs/refrsvd_c32_g1_%j.out \
    --account=def-smolesky \
    --time=02:33:42 \
    --cpus-per-task=12 \
    --mem=26G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --scratch "$CAL_ROOT/refined/refrsvd_c32_g1" --fresh --gpu 0 --root $CAL_ROOT --out $ROWS/refrsvd_c32_g1.csv --cluster narval --note 'tier=refined;label=refrsvd_c32_g1'
EOF
)
sleep 0.05

jid_refbounds_c32_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refrsvd_c32_g1} \
    --job-name=psccal_refbounds_c32_g1 \
    --output=$CAL_ROOT/logs/refbounds_c32_g1_%j.out \
    --account=def-smolesky \
    --time=00:49:31 \
    --cpus-per-task=12 \
    --mem=26G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_bounds --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260827' --refine --scratch "$CAL_ROOT/refined/refrsvd_c32_g1" --gamma-rtol '1.0e-12' --outer-blocks '4' --outer-block-len '24' --gpu 0 --root $CAL_ROOT --out $ROWS/refbounds_c32_g1.csv --cluster narval --note 'tier=refined;label=refbounds_c32_g1'
EOF
)
sleep 0.05

echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo
echo "When they have finished, merge the per-point rows and copy the result back:"
echo "  bash bench/launch_calibration_narval_refined.sh --merge"
echo "  scp pvirally@narval.alliancecan.ca:$OUT bench/data/"

