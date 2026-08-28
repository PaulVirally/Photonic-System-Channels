#!/bin/bash
# Cost-model calibration for narval, tier=refined.
# Generated 2026-08-28T17:20:20.926 by bench/plan_refined.jl. Do not edit; regenerate instead.
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
#   refg_*        the composite Green build, at the four GAP_REFINEMENT_TABLE
#                 entries a production sweep actually reaches, on three
#                 geometries, plus each of those geometries at g = 6 where
#                 nothing is refined. Fits g0_sandwich_scale and
#                 g0_partition_scale.
#   refmv_*       one composite matvec each of G0_rs, G0_rr and the universe
#                 block operator, and the same three on the unrefined control
#                 geometry. Fits mv_composite_scale as a ratio of two measured
#                 numbers.
#   refrsvd_*     one refined RSVD end to end, at k = 800 so that it fits the
#                 box. Checks the assembled model, and with refbounds_* is the
#                 only place the folded Asym applies of the hasmethod shim in
#                 src/rsvd.jl are visible.
#   refbounds_*   one refined bounds front end, outer loop sampled, reading the
#                 RSVD point's scratch.
#
# Every point carries --refine, controls included: bench/point.jl leaves
# refinement off unless a tier asks, unlike src/common.jl, and holding the flag
# fixed across the tier leaves the separation as the only variable. A control at
# g = 6 builds the plain cuboids anyway, which is what makes it a control.
#
# THE PRELOAD DIRECTORY MUST BE EMPTY. stage_greens measures a build, and
# load_green_function returns a cached block rather than building one when the
# file is already there -- a warm directory turns every greens point into a
# deserialisation timing. Worse, a .glaG0 written before Gila rev d4c0516 no
# longer reads back at all: GlaOprVac now serialises mem, srcMsk and trgMsk where
# it used to write mem alone, so an old file dies with an EOFError partway
# through. This script refuses to submit while any .glaG0 is present; clear them
# first, or pass --force if you know what is there.
#
# Before submitting, two steps on two different kinds of node.
#
# Instantiate on the LOGIN node -- compute nodes have no internet, and a fresh
# clone has no Manifest.toml (it is gitignored), so this resolves from scratch:
#
#   module load StdEnv/2023 julia/1.12.5 cuda/12.2
#   cd <code dir>
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# Precompile on a GPU node, NOT here. CUDA.jl has to be configured for the
# cluster's local toolkit and precompiled against a visible device; doing it off
# a GPU node is what produces the errors. A MIG slice is enough:
#
#   salloc --account=<account> --time=00:45:00 --mem=16G --cpus-per-task=2 \
#          --gpus=a100_1g.5gb:1 srun --pty bash test/setup_cuda_narval.sh
#
# The refg points are independent of each other and can be submitted on their
# own; the device points depend on their own geometry's refg point having written
# the preload directory, and refbounds depends on refrsvd.

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
        echo "No row files in $ROWS -- nothing to merge."
        exit 1
    fi
    head -n 1 $(ls -1 $ROWS/*.csv | head -n 1) > $OUT
    for f in $ROWS/*.csv; do tail -n +2 "$f" >> $OUT; done
    echo "Merged $n row file(s) into $OUT ($(( $(wc -l < $OUT) - 1 )) rows)."
    exit 0
fi

if [ "${1:-}" != "--force" ]; then
    stale=$(find $CAL_ROOT/preload -name '*.glaG0' 2>/dev/null | wc -l)
    if [ "$stale" -ne 0 ]; then
        echo "$CAL_ROOT/preload already holds $stale .glaG0 file(s)."
        echo
        echo "stage_greens measures a build, and load_green_function loads a cached"
        echo "block instead of building one. A block cached before Gila rev d4c0516"
        echo "does not even deserialise -- the on-disk format gained srcMsk/trgMsk."
        echo
        echo "Look at them, then clear them:"
        echo "  ls -R $CAL_ROOT/preload"
        echo "  find $CAL_ROOT/preload -name '*.glaG0' -delete"
        echo
        echo "Or re-run this script with --force to submit anyway."
        exit 1
    fi
fi

echo "Submitting 17 calibration points for narval (tier=refined)"
echo "Each point writes its own row file under $ROWS"

jid_refg_c0p25_g1=$(sbatch --parsable \
    --job-name=psccal_refg_c0p25_g1 \
    --output=$CAL_ROOT/logs/refg_c0p25_g1_%j.out \
    --account=def-smolesky \
    --time=00:44:34 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c0p25_g1.csv --cluster narval --note 'tier=refined;label=refg_c0p25_g1'
EOF
)
sleep 0.05

jid_refg_c0p25_g3=$(sbatch --parsable \
    --job-name=psccal_refg_c0p25_g3 \
    --output=$CAL_ROOT/logs/refg_c0p25_g3_%j.out \
    --account=def-smolesky \
    --time=01:01:14 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '3//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c0p25_g3.csv --cluster narval --note 'tier=refined;label=refg_c0p25_g3'
EOF
)
sleep 0.05

jid_refg_c1lam_g1=$(sbatch --parsable \
    --job-name=psccal_refg_c1lam_g1 \
    --output=$CAL_ROOT/logs/refg_c1lam_g1_%j.out \
    --account=def-smolesky \
    --time=01:46:27 \
    --cpus-per-task=12 \
    --mem=15G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c1lam_g1.csv --cluster narval --note 'tier=refined;label=refg_c1lam_g1'
EOF
)
sleep 0.05

jid_refg_c1lam_g2=$(sbatch --parsable \
    --job-name=psccal_refg_c1lam_g2 \
    --output=$CAL_ROOT/logs/refg_c1lam_g2_%j.out \
    --account=def-smolesky \
    --time=01:25:32 \
    --cpus-per-task=12 \
    --mem=13G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c1lam_g2.csv --cluster narval --note 'tier=refined;label=refg_c1lam_g2'
EOF
)
sleep 0.05

jid_refg_c2lam_g1=$(sbatch --parsable \
    --job-name=psccal_refg_c2lam_g1 \
    --output=$CAL_ROOT/logs/refg_c2lam_g1_%j.out \
    --account=def-smolesky \
    --time=02:27:38 \
    --cpus-per-task=12 \
    --mem=19G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c2lam_g1.csv --cluster narval --note 'tier=refined;label=refg_c2lam_g1'
EOF
)
sleep 0.05

jid_refg_c2lam_g5=$(sbatch --parsable \
    --job-name=psccal_refg_c2lam_g5 \
    --output=$CAL_ROOT/logs/refg_c2lam_g5_%j.out \
    --account=def-smolesky \
    --time=01:37:04 \
    --cpus-per-task=12 \
    --mem=14G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '18.612+1.5502660841406073im' --sep '5//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c2lam_g5.csv --cluster narval --note 'tier=refined;label=refg_c2lam_g5'
EOF
)
sleep 0.05

jid_refg_c0p25_g6=$(sbatch --parsable \
    --job-name=psccal_refg_c0p25_g6 \
    --output=$CAL_ROOT/logs/refg_c0p25_g6_%j.out \
    --account=def-smolesky \
    --time=00:30:27 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c0p25_g6.csv --cluster narval --note 'tier=refined;label=refg_c0p25_g6'
EOF
)
sleep 0.05

jid_refg_c1lam_g6=$(sbatch --parsable \
    --job-name=psccal_refg_c1lam_g6 \
    --output=$CAL_ROOT/logs/refg_c1lam_g6_%j.out \
    --account=def-smolesky \
    --time=00:33:35 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c1lam_g6.csv --cluster narval --note 'tier=refined;label=refg_c1lam_g6'
EOF
)
sleep 0.05

jid_refg_c2lam_g6=$(sbatch --parsable \
    --job-name=psccal_refg_c2lam_g6 \
    --output=$CAL_ROOT/logs/refg_c2lam_g6_%j.out \
    --account=def-smolesky \
    --time=00:36:49 \
    --cpus-per-task=12 \
    --mem=11G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '18.612+1.5502660841406073im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu -1 --root $CAL_ROOT --out $ROWS/refg_c2lam_g6.csv --cluster narval --note 'tier=refined;label=refg_c2lam_g6'
EOF
)
sleep 0.05

jid_refmv_ext_c1lam_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c1lam_g1} \
    --job-name=psccal_refmv_ext_c1lam_g1 \
    --output=$CAL_ROOT/logs/refmv_ext_c1lam_g1_%j.out \
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
srun julia --project=. -t 12 bench/point.jl --kind matvec_ext --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_ext_c1lam_g1.csv --cluster narval --note 'tier=refined;label=refmv_ext_c1lam_g1'
EOF
)
sleep 0.05

jid_refmv_self_c1lam_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c1lam_g1} \
    --job-name=psccal_refmv_self_c1lam_g1 \
    --output=$CAL_ROOT/logs/refmv_self_c1lam_g1_%j.out \
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
srun julia --project=. -t 12 bench/point.jl --kind matvec_self --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_self_c1lam_g1.csv --cluster narval --note 'tier=refined;label=refmv_self_c1lam_g1'
EOF
)
sleep 0.05

jid_refmv_uu_c1lam_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c1lam_g1} \
    --job-name=psccal_refmv_uu_c1lam_g1 \
    --output=$CAL_ROOT/logs/refmv_uu_c1lam_g1_%j.out \
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
srun julia --project=. -t 12 bench/point.jl --kind matvec_uu --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_uu_c1lam_g1.csv --cluster narval --note 'tier=refined;label=refmv_uu_c1lam_g1'
EOF
)
sleep 0.05

jid_refmv_ext_c1lam_g6=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c1lam_g6} \
    --job-name=psccal_refmv_ext_c1lam_g6 \
    --output=$CAL_ROOT/logs/refmv_ext_c1lam_g6_%j.out \
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
srun julia --project=. -t 12 bench/point.jl --kind matvec_ext --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_ext_c1lam_g6.csv --cluster narval --note 'tier=refined;label=refmv_ext_c1lam_g6'
EOF
)
sleep 0.05

jid_refmv_self_c1lam_g6=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c1lam_g6} \
    --job-name=psccal_refmv_self_c1lam_g6 \
    --output=$CAL_ROOT/logs/refmv_self_c1lam_g6_%j.out \
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
srun julia --project=. -t 12 bench/point.jl --kind matvec_self --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_self_c1lam_g6.csv --cluster narval --note 'tier=refined;label=refmv_self_c1lam_g6'
EOF
)
sleep 0.05

jid_refmv_uu_c1lam_g6=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c1lam_g6} \
    --job-name=psccal_refmv_uu_c1lam_g6 \
    --output=$CAL_ROOT/logs/refmv_uu_c1lam_g6_%j.out \
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
srun julia --project=. -t 12 bench/point.jl --kind matvec_uu --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '3//16' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --gpu 0 --root $CAL_ROOT --out $ROWS/refmv_uu_c1lam_g6.csv --cluster narval --note 'tier=refined;label=refmv_uu_c1lam_g6'
EOF
)
sleep 0.05

jid_refrsvd_c1lam_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refg_c1lam_g1} \
    --job-name=psccal_refrsvd_c1lam_g1 \
    --output=$CAL_ROOT/logs/refrsvd_c1lam_g1_%j.out \
    --account=def-smolesky \
    --time=02:29:48 \
    --cpus-per-task=12 \
    --mem=24G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --scratch "$CAL_ROOT/refined/refrsvd_c1lam_g1" --fresh --gpu 0 --root $CAL_ROOT --out $ROWS/refrsvd_c1lam_g1.csv --cluster narval --note 'tier=refined;label=refrsvd_c1lam_g1'
EOF
)
sleep 0.05

jid_refbounds_c1lam_g1=$(sbatch --parsable \
    --dependency=afterok:${jid_refrsvd_c1lam_g1} \
    --job-name=psccal_refbounds_c1lam_g1 \
    --output=$CAL_ROOT/logs/refbounds_c1lam_g1_%j.out \
    --account=def-smolesky \
    --time=00:52:40 \
    --cpus-per-task=12 \
    --mem=24G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 12 bench/point.jl --kind stage_bounds --cells '32,32,32' --scale '1//32' --chi '18.612+1.5502660841406073im' --sep '1//32' --rank '800' --oversamples '50' --power-iters '6' --seed '20260828' --refine --scratch "$CAL_ROOT/refined/refrsvd_c1lam_g1" --gamma-rtol '1.0e-12' --outer-blocks '4' --outer-block-len '24' --gpu 0 --root $CAL_ROOT --out $ROWS/refbounds_c1lam_g1.csv --cluster narval --note 'tier=refined;label=refbounds_c1lam_g1'
EOF
)
sleep 0.05

echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo
echo "When they have finished, merge the per-point rows and copy the result back:"
echo "  bash bench/launch_calibration_narval_refined.sh --merge"
echo "  scp pvirally@narval.alliancecan.ca:$OUT bench/data/"

