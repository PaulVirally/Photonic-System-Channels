#!/bin/bash
# Cost-model calibration for fir, tier=validate.
# Generated 2026-07-29T12:26:43.146 by bench/plan.jl. Do not edit; regenerate instead.
#
# Every point is its own job: one point running out of memory or time must
# not take the rest of the calibration with it. Rows are appended to
# $OUT as each job finishes, so partial results are still usable.

set -u

CODE_DIR=/home/pvirally/Photonic-System-Channels/
CAL_ROOT=/home/pvirally/scratch/psc-calibration/
OUT=$CAL_ROOT/calibration_fir.csv

mkdir -p $CAL_ROOT/logs $CAL_ROOT/preload $CAL_ROOT/project $CAL_ROOT/scratch
cd $CODE_DIR

echo "Submitting 30 calibration points for fir (tier=validate)"
echo "Results will accumulate in $OUT"

sbatch \
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
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l0p25_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l0p25_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l0p25_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l0p25_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l0p25_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l0p25_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:05:47 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l0p25_sep1ss4'
EOF
sleep 0.05

sbatch \
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
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l0p25_sep0ss1'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l0p25_sep0ss1 \
    --output=$CAL_ROOT/logs/stagersvd_l0p25_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l0p25_sep0ss1'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l0p25_sep0ss1 \
    --output=$CAL_ROOT/logs/stagebounds_l0p25_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:05:47 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l0p25_sep0ss1'
EOF
sleep 0.05

sbatch \
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
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l0p5_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l0p5_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l0p5_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l0p5_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l0p5_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l0p5_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:06:09 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l0p5_sep1ss4'
EOF
sleep 0.05

sbatch \
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
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l0p5_sep0ss1'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l0p5_sep0ss1 \
    --output=$CAL_ROOT/logs/stagersvd_l0p5_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l0p5_sep0ss1'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l0p5_sep0ss1 \
    --output=$CAL_ROOT/logs/stagebounds_l0p5_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:06:09 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l0p5_sep0ss1'
EOF
sleep 0.05

sbatch \
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
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l0p75_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l0p75_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l0p75_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=19G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l0p75_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l0p75_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l0p75_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:07:13 \
    --cpus-per-task=4 \
    --mem=21G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l0p75_sep1ss4'
EOF
sleep 0.05

sbatch \
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
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l1_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l1_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l1_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:41:46 \
    --cpus-per-task=4 \
    --mem=57G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l1_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l1_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l1_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=12:09:49 \
    --cpus-per-task=4 \
    --mem=69G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l1_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagegreens_l2agiso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l2agiso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l2agiso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l2agiso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l2agiso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:21:53 \
    --cpus-per-task=4 \
    --mem=56G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l2agiso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l2agiso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l2agiso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:13:12 \
    --cpus-per-task=4 \
    --mem=68G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l2agiso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagegreens_l3aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l3aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l3aniso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l3aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l3aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:09:02 \
    --cpus-per-task=4 \
    --mem=51G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l3aniso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l3aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l3aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=62G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l3aniso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagegreens_l4aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l4aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l4aniso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l4aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l4aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:07:58 \
    --cpus-per-task=4 \
    --mem=51G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l4aniso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l4aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l4aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=62G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l4aniso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagegreens_l2iso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagegreens_l2iso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagegreens_l2iso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagersvd_l2iso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagersvd_l2iso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=02:06:39 \
    --cpus-per-task=4 \
    --mem=93G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagersvd_l2iso_sep1ss4'
EOF
sleep 0.05

sbatch \
    --job-name=psccal_stagebounds_l2iso_sep1ss4 \
    --output=$CAL_ROOT/logs/stagebounds_l2iso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=114G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu 0 --root $CAL_ROOT --out $OUT --cluster fir --note 'tier=validate;label=stagebounds_l2iso_sep1ss4'
EOF
sleep 0.05

echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo "When they are done, copy the CSV back:"
echo "  scp pvirally@fir.alliancecan.ca:$OUT bench/data/"

