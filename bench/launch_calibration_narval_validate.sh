#!/bin/bash
# Cost-model calibration for narval, tier=validate.
# Generated 2026-08-05T12:30:14.419 by bench/plan.jl. Do not edit; regenerate instead.
#
# Every point is its own job: one point running out of memory or time must
# not take the rest of the calibration with it. Each writes its own row file,
# so partial results are always usable.
#
# Submit:  bash <this script>
# Collect: bash <this script> --merge

set -u

CODE_DIR=/home/pvirally/Photonic-System-Channels/
CAL_ROOT=/home/pvirally/scratch/psc-calibration/
ROWS=$CAL_ROOT/rows
OUT=$CAL_ROOT/calibration_narval.csv

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

echo "Submitting 30 calibration points for narval (tier=validate)"
echo "Each point writes its own row file under $ROWS"

jid_stagegreens_l0p25_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l0p25_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l0p25_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p25_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l0p25_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l0p25_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l0p25_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l0p25_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p25_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l0p25_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l0p25_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l0p25_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l0p25_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p25_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l0p25_sep1ss4'
EOF
)
sleep 0.05

jid_stagegreens_l0p25_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l0p25_sep0ss1 \
    --output=$CAL_ROOT/logs/stagegreens_l0p25_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p25_sep0ss1.csv --cluster narval --note 'tier=validate;label=stagegreens_l0p25_sep0ss1'
EOF
)
sleep 0.05

jid_stagersvd_l0p25_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l0p25_sep0ss1 \
    --output=$CAL_ROOT/logs/stagersvd_l0p25_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p25_sep0ss1.csv --cluster narval --note 'tier=validate;label=stagersvd_l0p25_sep0ss1'
EOF
)
sleep 0.05

jid_stagebounds_l0p25_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l0p25_sep0ss1 \
    --output=$CAL_ROOT/logs/stagebounds_l0p25_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p25_sep0ss1.csv --cluster narval --note 'tier=validate;label=stagebounds_l0p25_sep0ss1'
EOF
)
sleep 0.05

jid_stagegreens_l0p5_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l0p5_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l0p5_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p5_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l0p5_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l0p5_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l0p5_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l0p5_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p5_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l0p5_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l0p5_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l0p5_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l0p5_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p5_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l0p5_sep1ss4'
EOF
)
sleep 0.05

jid_stagegreens_l0p5_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l0p5_sep0ss1 \
    --output=$CAL_ROOT/logs/stagegreens_l0p5_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p5_sep0ss1.csv --cluster narval --note 'tier=validate;label=stagegreens_l0p5_sep0ss1'
EOF
)
sleep 0.05

jid_stagersvd_l0p5_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l0p5_sep0ss1 \
    --output=$CAL_ROOT/logs/stagersvd_l0p5_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p5_sep0ss1.csv --cluster narval --note 'tier=validate;label=stagersvd_l0p5_sep0ss1'
EOF
)
sleep 0.05

jid_stagebounds_l0p5_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l0p5_sep0ss1 \
    --output=$CAL_ROOT/logs/stagebounds_l0p5_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p5_sep0ss1.csv --cluster narval --note 'tier=validate;label=stagebounds_l0p5_sep0ss1'
EOF
)
sleep 0.05

jid_stagegreens_l0p75_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l0p75_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l0p75_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l0p75_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l0p75_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l0p75_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l0p75_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l0p75_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:09:35 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l0p75_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l0p75_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l0p75_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l0p75_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l0p75_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l0p75_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l0p75_sep1ss4'
EOF
)
sleep 0.05

jid_stagegreens_l1_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l1_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l1_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l1_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l1_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l1_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l1_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l1_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=03:54:25 \
    --cpus-per-task=4 \
    --mem=31G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l1_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l1_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l1_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l1_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l1_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=04:10:36 \
    --cpus-per-task=4 \
    --mem=31G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l1_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l1_sep1ss4'
EOF
)
sleep 0.05

jid_stagegreens_l2agiso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l2agiso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l2agiso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:16:19 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l2agiso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l2agiso_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l2agiso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l2agiso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l2agiso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=03:17:42 \
    --cpus-per-task=4 \
    --mem=31G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l2agiso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l2agiso_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l2agiso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l2agiso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l2agiso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=31G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l2agiso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l2agiso_sep1ss4'
EOF
)
sleep 0.05

jid_stagegreens_l3aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l3aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l3aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:43:28 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l3aniso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l3aniso_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l3aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l3aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l3aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=02:50:25 \
    --cpus-per-task=4 \
    --mem=28G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l3aniso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l3aniso_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l3aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l3aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l3aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=28G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l3aniso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l3aniso_sep1ss4'
EOF
)
sleep 0.05

jid_stagegreens_l4aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l4aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l4aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=02:10:40 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l4aniso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l4aniso_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l4aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l4aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l4aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=02:49:53 \
    --cpus-per-task=4 \
    --mem=28G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l4aniso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l4aniso_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l4aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l4aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l4aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=28G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l4aniso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l4aniso_sep1ss4'
EOF
)
sleep 0.05

jid_stagegreens_l2iso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagegreens_l2iso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l2iso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=03:59:48 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/stagegreens_l2iso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagegreens_l2iso_sep1ss4'
EOF
)
sleep 0.05

jid_stagersvd_l2iso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagersvd_l2iso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l2iso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=05:34:36 \
    --cpus-per-task=4 \
    --mem=49G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagersvd_l2iso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagersvd_l2iso_sep1ss4'
EOF
)
sleep 0.05

jid_stagebounds_l2iso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_stagebounds_l2iso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l2iso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=49G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $ROWS/stagebounds_l2iso_sep1ss4.csv --cluster narval --note 'tier=validate;label=stagebounds_l2iso_sep1ss4'
EOF
)
sleep 0.05

echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo
echo "When they have finished, merge the per-point rows and copy the result back:"
echo "  bash bench/launch_calibration_narval_validate.sh --merge"
echo "  scp pvirally@narval.alliancecan.ca:$OUT bench/data/"

