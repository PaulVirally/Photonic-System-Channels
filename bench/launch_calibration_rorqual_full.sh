#!/bin/bash
# Cost-model calibration for rorqual, tier=full.
# Generated 2026-08-05T12:30:30.556 by bench/plan.jl. Do not edit; regenerate instead.
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
OUT=$CAL_ROOT/calibration_rorqual.csv

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

echo "Submitting 128 calibration points for rorqual (tier=full)"
echo "Each point writes its own row file under $ROWS"

jid_g0self_l0p25=$(sbatch --parsable \
    --job-name=psccal_g0self_l0p25 \
    --output=$CAL_ROOT/logs/g0self_l0p25_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l0p25.csv --cluster rorqual --note 'tier=full;label=g0self_l0p25'
EOF
)
sleep 0.05

jid_g0ext_l0p25_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p25_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p25_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:23:58 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p25_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p25_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l0p25_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p25_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l0p25_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p25_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p25_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l0p25_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p25_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l0p25_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p25_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p25_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l0p25_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p25_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p25_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p25_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p25_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l0p25_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p25_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p25_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p25_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p25_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l0p25=$(sbatch --parsable \
    --job-name=psccal_g0uu_l0p25 \
    --output=$CAL_ROOT/logs/g0uu_l0p25_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l0p25.csv --cluster rorqual --note 'tier=full;label=g0uu_l0p25'
EOF
)
sleep 0.05

jid_g0self_l0p5=$(sbatch --parsable \
    --job-name=psccal_g0self_l0p5 \
    --output=$CAL_ROOT/logs/g0self_l0p5_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l0p5.csv --cluster rorqual --note 'tier=full;label=g0self_l0p5'
EOF
)
sleep 0.05

jid_g0ext_l0p5_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p5_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p5_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:24:42 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p5_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p5_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l0p5_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p5_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l0p5_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p5_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p5_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l0p5_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p5_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l0p5_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p5_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p5_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l0p5_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p5_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p5_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p5_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p5_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l0p5_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p5_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p5_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p5_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p5_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l0p5=$(sbatch --parsable \
    --job-name=psccal_g0uu_l0p5 \
    --output=$CAL_ROOT/logs/g0uu_l0p5_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l0p5.csv --cluster rorqual --note 'tier=full;label=g0uu_l0p5'
EOF
)
sleep 0.05

jid_g0self_l0p75=$(sbatch --parsable \
    --job-name=psccal_g0self_l0p75 \
    --output=$CAL_ROOT/logs/g0self_l0p75_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l0p75.csv --cluster rorqual --note 'tier=full;label=g0self_l0p75'
EOF
)
sleep 0.05

jid_g0ext_l0p75_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p75_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p75_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:26:40 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p75_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p75_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l0p75_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p75_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l0p75_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p75_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p75_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l0p75_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p75_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l0p75_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p75_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p75_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l0p75_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p75_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p75_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p75_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p75_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l0p75_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l0p75_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l0p75_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l0p75_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l0p75_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l0p75=$(sbatch --parsable \
    --job-name=psccal_g0uu_l0p75 \
    --output=$CAL_ROOT/logs/g0uu_l0p75_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l0p75.csv --cluster rorqual --note 'tier=full;label=g0uu_l0p75'
EOF
)
sleep 0.05

jid_g0self_l1=$(sbatch --parsable \
    --job-name=psccal_g0self_l1 \
    --output=$CAL_ROOT/logs/g0self_l1_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l1.csv --cluster rorqual --note 'tier=full;label=g0self_l1'
EOF
)
sleep 0.05

jid_g0ext_l1_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l1_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l1_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:30:31 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l1_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l1_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l1_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l1_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l1_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l1_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l1_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l1_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l1_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l1_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l1_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l1_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l1_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l1_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l1_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l1_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l1_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l1_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l1_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l1_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l1_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l1_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l1=$(sbatch --parsable \
    --job-name=psccal_g0uu_l1 \
    --output=$CAL_ROOT/logs/g0uu_l1_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l1.csv --cluster rorqual --note 'tier=full;label=g0uu_l1'
EOF
)
sleep 0.05

jid_g0self_l2agiso=$(sbatch --parsable \
    --job-name=psccal_g0self_l2agiso \
    --output=$CAL_ROOT/logs/g0self_l2agiso_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l2agiso.csv --cluster rorqual --note 'tier=full;label=g0self_l2agiso'
EOF
)
sleep 0.05

jid_g0ext_l2agiso_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2agiso_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l2agiso_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:37:11 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2agiso_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l2agiso_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l2agiso_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2agiso_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l2agiso_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2agiso_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l2agiso_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l2agiso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2agiso_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l2agiso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2agiso_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l2agiso_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l2agiso_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2agiso_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l2agiso_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2agiso_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l2agiso_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l2agiso_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2agiso_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l2agiso_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2agiso_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l2agiso_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l2agiso=$(sbatch --parsable \
    --job-name=psccal_g0uu_l2agiso \
    --output=$CAL_ROOT/logs/g0uu_l2agiso_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l2agiso.csv --cluster rorqual --note 'tier=full;label=g0uu_l2agiso'
EOF
)
sleep 0.05

jid_g0self_l3aniso=$(sbatch --parsable \
    --job-name=psccal_g0self_l3aniso \
    --output=$CAL_ROOT/logs/g0self_l3aniso_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l3aniso.csv --cluster rorqual --note 'tier=full;label=g0self_l3aniso'
EOF
)
sleep 0.05

jid_g0ext_l3aniso_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l3aniso_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l3aniso_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:43:51 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l3aniso_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l3aniso_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l3aniso_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l3aniso_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l3aniso_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l3aniso_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l3aniso_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l3aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l3aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l3aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l3aniso_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l3aniso_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l3aniso_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l3aniso_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l3aniso_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l3aniso_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l3aniso_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l3aniso_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l3aniso_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l3aniso_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l3aniso_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l3aniso_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l3aniso=$(sbatch --parsable \
    --job-name=psccal_g0uu_l3aniso \
    --output=$CAL_ROOT/logs/g0uu_l3aniso_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l3aniso.csv --cluster rorqual --note 'tier=full;label=g0uu_l3aniso'
EOF
)
sleep 0.05

jid_g0self_l4aniso=$(sbatch --parsable \
    --job-name=psccal_g0self_l4aniso \
    --output=$CAL_ROOT/logs/g0self_l4aniso_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l4aniso.csv --cluster rorqual --note 'tier=full;label=g0self_l4aniso'
EOF
)
sleep 0.05

jid_g0ext_l4aniso_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l4aniso_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l4aniso_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=01:50:31 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l4aniso_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l4aniso_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l4aniso_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l4aniso_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l4aniso_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l4aniso_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l4aniso_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l4aniso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l4aniso_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l4aniso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l4aniso_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l4aniso_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l4aniso_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l4aniso_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l4aniso_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l4aniso_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l4aniso_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l4aniso_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l4aniso_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l4aniso_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l4aniso_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l4aniso_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l4aniso=$(sbatch --parsable \
    --job-name=psccal_g0uu_l4aniso \
    --output=$CAL_ROOT/logs/g0uu_l4aniso_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l4aniso.csv --cluster rorqual --note 'tier=full;label=g0uu_l4aniso'
EOF
)
sleep 0.05

jid_g0self_l2iso=$(sbatch --parsable \
    --job-name=psccal_g0self_l2iso \
    --output=$CAL_ROOT/logs/g0self_l2iso_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_self --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0self_l2iso.csv --cluster rorqual --note 'tier=full;label=g0self_l2iso'
EOF
)
sleep 0.05

jid_g0ext_l2iso_sep0ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2iso_sep0ss1 \
    --output=$CAL_ROOT/logs/g0ext_l2iso_sep0ss1_%j.out \
    --account=def-smolesky \
    --time=02:17:09 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '0//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2iso_sep0ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l2iso_sep0ss1'
EOF
)
sleep 0.05

jid_g0ext_l2iso_sep1ss32=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2iso_sep1ss32 \
    --output=$CAL_ROOT/logs/g0ext_l2iso_sep1ss32_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//32' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2iso_sep1ss32.csv --cluster rorqual --note 'tier=full;label=g0ext_l2iso_sep1ss32'
EOF
)
sleep 0.05

jid_g0ext_l2iso_sep1ss4=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2iso_sep1ss4 \
    --output=$CAL_ROOT/logs/g0ext_l2iso_sep1ss4_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2iso_sep1ss4.csv --cluster rorqual --note 'tier=full;label=g0ext_l2iso_sep1ss4'
EOF
)
sleep 0.05

jid_g0ext_l2iso_sep1ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2iso_sep1ss1 \
    --output=$CAL_ROOT/logs/g0ext_l2iso_sep1ss1_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2iso_sep1ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l2iso_sep1ss1'
EOF
)
sleep 0.05

jid_g0ext_l2iso_sep1000ss1=$(sbatch --parsable \
    --job-name=psccal_g0ext_l2iso_sep1000ss1 \
    --output=$CAL_ROOT/logs/g0ext_l2iso_sep1000ss1_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1000//1' --gpu -1 --root $CAL_ROOT --out $ROWS/g0ext_l2iso_sep1000ss1.csv --cluster rorqual --note 'tier=full;label=g0ext_l2iso_sep1000ss1'
EOF
)
sleep 0.05

jid_g0uu_l2iso=$(sbatch --parsable \
    --job-name=psccal_g0uu_l2iso \
    --output=$CAL_ROOT/logs/g0uu_l2iso_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_multiregion --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0uu_l2iso.csv --cluster rorqual --note 'tier=full;label=g0uu_l2iso'
EOF
)
sleep 0.05

jid_g0threads_l1_t1=$(sbatch --parsable \
    --job-name=psccal_g0threads_l1_t1 \
    --output=$CAL_ROOT/logs/g0threads_l1_t1_%j.out \
    --account=def-smolesky \
    --time=02:17:53 \
    --cpus-per-task=1 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 1 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0threads_l1_t1.csv --cluster rorqual --note 'tier=full;label=g0threads_l1_t1'
EOF
)
sleep 0.05

jid_g0threads_l1_t2=$(sbatch --parsable \
    --job-name=psccal_g0threads_l1_t2 \
    --output=$CAL_ROOT/logs/g0threads_l1_t2_%j.out \
    --account=def-smolesky \
    --time=01:50:19 \
    --cpus-per-task=2 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 2 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0threads_l1_t2.csv --cluster rorqual --note 'tier=full;label=g0threads_l1_t2'
EOF
)
sleep 0.05

jid_g0threads_l1_t4=$(sbatch --parsable \
    --job-name=psccal_g0threads_l1_t4 \
    --output=$CAL_ROOT/logs/g0threads_l1_t4_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0threads_l1_t4.csv --cluster rorqual --note 'tier=full;label=g0threads_l1_t4'
EOF
)
sleep 0.05

jid_g0threads_l1_t8=$(sbatch --parsable \
    --job-name=psccal_g0threads_l1_t8 \
    --output=$CAL_ROOT/logs/g0threads_l1_t8_%j.out \
    --account=def-smolesky \
    --time=01:29:39 \
    --cpus-per-task=8 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 8 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0threads_l1_t8.csv --cluster rorqual --note 'tier=full;label=g0threads_l1_t8'
EOF
)
sleep 0.05

jid_g0threads_l1_t16=$(sbatch --parsable \
    --job-name=psccal_g0threads_l1_t16 \
    --output=$CAL_ROOT/logs/g0threads_l1_t16_%j.out \
    --account=def-smolesky \
    --time=01:26:13 \
    --cpus-per-task=16 \
    --mem=8G \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 16 bench/point.jl --kind g0_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/g0threads_l1_t16.csv --cluster rorqual --note 'tier=full;label=g0threads_l1_t16'
EOF
)
sleep 0.05

jid_mvself_l0p25=$(sbatch --parsable \
    --job-name=psccal_mvself_l0p25 \
    --output=$CAL_ROOT/logs/mvself_l0p25_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l0p25.csv --cluster rorqual --note 'tier=full;label=mvself_l0p25'
EOF
)
sleep 0.05

jid_mvext_l0p25=$(sbatch --parsable \
    --job-name=psccal_mvext_l0p25 \
    --output=$CAL_ROOT/logs/mvext_l0p25_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l0p25.csv --cluster rorqual --note 'tier=full;label=mvext_l0p25'
EOF
)
sleep 0.05

jid_mvuu_l0p25=$(sbatch --parsable \
    --job-name=psccal_mvuu_l0p25 \
    --output=$CAL_ROOT/logs/mvuu_l0p25_%j.out \
    --account=def-smolesky \
    --time=01:15:45 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l0p25.csv --cluster rorqual --note 'tier=full;label=mvuu_l0p25'
EOF
)
sleep 0.05

jid_mvself_l0p5=$(sbatch --parsable \
    --job-name=psccal_mvself_l0p5 \
    --output=$CAL_ROOT/logs/mvself_l0p5_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l0p5.csv --cluster rorqual --note 'tier=full;label=mvself_l0p5'
EOF
)
sleep 0.05

jid_mvext_l0p5=$(sbatch --parsable \
    --job-name=psccal_mvext_l0p5 \
    --output=$CAL_ROOT/logs/mvext_l0p5_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l0p5.csv --cluster rorqual --note 'tier=full;label=mvext_l0p5'
EOF
)
sleep 0.05

jid_mvuu_l0p5=$(sbatch --parsable \
    --job-name=psccal_mvuu_l0p5 \
    --output=$CAL_ROOT/logs/mvuu_l0p5_%j.out \
    --account=def-smolesky \
    --time=01:17:56 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l0p5.csv --cluster rorqual --note 'tier=full;label=mvuu_l0p5'
EOF
)
sleep 0.05

jid_mvself_l0p75=$(sbatch --parsable \
    --job-name=psccal_mvself_l0p75 \
    --output=$CAL_ROOT/logs/mvself_l0p75_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l0p75.csv --cluster rorqual --note 'tier=full;label=mvself_l0p75'
EOF
)
sleep 0.05

jid_mvext_l0p75=$(sbatch --parsable \
    --job-name=psccal_mvext_l0p75 \
    --output=$CAL_ROOT/logs/mvext_l0p75_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l0p75.csv --cluster rorqual --note 'tier=full;label=mvext_l0p75'
EOF
)
sleep 0.05

jid_mvuu_l0p75=$(sbatch --parsable \
    --job-name=psccal_mvuu_l0p75 \
    --output=$CAL_ROOT/logs/mvuu_l0p75_%j.out \
    --account=def-smolesky \
    --time=01:24:08 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l0p75.csv --cluster rorqual --note 'tier=full;label=mvuu_l0p75'
EOF
)
sleep 0.05

jid_mvself_l1=$(sbatch --parsable \
    --job-name=psccal_mvself_l1 \
    --output=$CAL_ROOT/logs/mvself_l1_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l1.csv --cluster rorqual --note 'tier=full;label=mvself_l1'
EOF
)
sleep 0.05

jid_mvext_l1=$(sbatch --parsable \
    --job-name=psccal_mvext_l1 \
    --output=$CAL_ROOT/logs/mvext_l1_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l1.csv --cluster rorqual --note 'tier=full;label=mvext_l1'
EOF
)
sleep 0.05

jid_mvuu_l1=$(sbatch --parsable \
    --job-name=psccal_mvuu_l1 \
    --output=$CAL_ROOT/logs/mvuu_l1_%j.out \
    --account=def-smolesky \
    --time=01:36:33 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l1.csv --cluster rorqual --note 'tier=full;label=mvuu_l1'
EOF
)
sleep 0.05

jid_mvself_l2agiso=$(sbatch --parsable \
    --job-name=psccal_mvself_l2agiso \
    --output=$CAL_ROOT/logs/mvself_l2agiso_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l2agiso.csv --cluster rorqual --note 'tier=full;label=mvself_l2agiso'
EOF
)
sleep 0.05

jid_mvext_l2agiso=$(sbatch --parsable \
    --job-name=psccal_mvext_l2agiso \
    --output=$CAL_ROOT/logs/mvext_l2agiso_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l2agiso.csv --cluster rorqual --note 'tier=full;label=mvext_l2agiso'
EOF
)
sleep 0.05

jid_mvuu_l2agiso=$(sbatch --parsable \
    --job-name=psccal_mvuu_l2agiso \
    --output=$CAL_ROOT/logs/mvuu_l2agiso_%j.out \
    --account=def-smolesky \
    --time=01:58:26 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l2agiso.csv --cluster rorqual --note 'tier=full;label=mvuu_l2agiso'
EOF
)
sleep 0.05

jid_mvself_l3aniso=$(sbatch --parsable \
    --job-name=psccal_mvself_l3aniso \
    --output=$CAL_ROOT/logs/mvself_l3aniso_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l3aniso.csv --cluster rorqual --note 'tier=full;label=mvself_l3aniso'
EOF
)
sleep 0.05

jid_mvext_l3aniso=$(sbatch --parsable \
    --job-name=psccal_mvext_l3aniso \
    --output=$CAL_ROOT/logs/mvext_l3aniso_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l3aniso.csv --cluster rorqual --note 'tier=full;label=mvext_l3aniso'
EOF
)
sleep 0.05

jid_mvuu_l3aniso=$(sbatch --parsable \
    --job-name=psccal_mvuu_l3aniso \
    --output=$CAL_ROOT/logs/mvuu_l3aniso_%j.out \
    --account=def-smolesky \
    --time=02:20:38 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l3aniso.csv --cluster rorqual --note 'tier=full;label=mvuu_l3aniso'
EOF
)
sleep 0.05

jid_mvself_l4aniso=$(sbatch --parsable \
    --job-name=psccal_mvself_l4aniso \
    --output=$CAL_ROOT/logs/mvself_l4aniso_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l4aniso.csv --cluster rorqual --note 'tier=full;label=mvself_l4aniso'
EOF
)
sleep 0.05

jid_mvext_l4aniso=$(sbatch --parsable \
    --job-name=psccal_mvext_l4aniso \
    --output=$CAL_ROOT/logs/mvext_l4aniso_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l4aniso.csv --cluster rorqual --note 'tier=full;label=mvext_l4aniso'
EOF
)
sleep 0.05

jid_mvuu_l4aniso=$(sbatch --parsable \
    --job-name=psccal_mvuu_l4aniso \
    --output=$CAL_ROOT/logs/mvuu_l4aniso_%j.out \
    --account=def-smolesky \
    --time=02:43:02 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l4aniso.csv --cluster rorqual --note 'tier=full;label=mvuu_l4aniso'
EOF
)
sleep 0.05

jid_mvself_l2iso=$(sbatch --parsable \
    --job-name=psccal_mvself_l2iso \
    --output=$CAL_ROOT/logs/mvself_l2iso_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_self --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvself_l2iso.csv --cluster rorqual --note 'tier=full;label=mvself_l2iso'
EOF
)
sleep 0.05

jid_mvext_l2iso=$(sbatch --parsable \
    --job-name=psccal_mvext_l2iso \
    --output=$CAL_ROOT/logs/mvext_l2iso_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_ext --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvext_l2iso.csv --cluster rorqual --note 'tier=full;label=mvext_l2iso'
EOF
)
sleep 0.05

jid_mvuu_l2iso=$(sbatch --parsable \
    --job-name=psccal_mvuu_l2iso \
    --output=$CAL_ROOT/logs/mvuu_l2iso_%j.out \
    --account=def-smolesky \
    --time=04:13:50 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind matvec_uu --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --reps '20' --gpu 0 --root $CAL_ROOT --out $ROWS/mvuu_l2iso.csv --cluster rorqual --note 'tier=full;label=mvuu_l2iso'
EOF
)
sleep 0.05

jid_dense_l0p25_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l0p25_c128 \
    --output=$CAL_ROOT/logs/dense_l0p25_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p25_c128.csv --cluster rorqual --note 'tier=full;label=dense_l0p25_c128'
EOF
)
sleep 0.05

jid_dense_l0p25_c512=$(sbatch --parsable \
    --job-name=psccal_dense_l0p25_c512 \
    --output=$CAL_ROOT/logs/dense_l0p25_c512_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p25_c512.csv --cluster rorqual --note 'tier=full;label=dense_l0p25_c512'
EOF
)
sleep 0.05

jid_dense_l0p25_c1400=$(sbatch --parsable \
    --job-name=psccal_dense_l0p25_c1400 \
    --output=$CAL_ROOT/logs/dense_l0p25_c1400_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p25_c1400.csv --cluster rorqual --note 'tier=full;label=dense_l0p25_c1400'
EOF
)
sleep 0.05

jid_dense_l0p25_c2800=$(sbatch --parsable \
    --job-name=psccal_dense_l0p25_c2800 \
    --output=$CAL_ROOT/logs/dense_l0p25_c2800_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '3072' --dense-c '2800' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p25_c2800.csv --cluster rorqual --note 'tier=full;label=dense_l0p25_c2800'
EOF
)
sleep 0.05

jid_dense_l0p5_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l0p5_c128 \
    --output=$CAL_ROOT/logs/dense_l0p5_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p5_c128.csv --cluster rorqual --note 'tier=full;label=dense_l0p5_c128'
EOF
)
sleep 0.05

jid_dense_l0p5_c512=$(sbatch --parsable \
    --job-name=psccal_dense_l0p5_c512 \
    --output=$CAL_ROOT/logs/dense_l0p5_c512_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p5_c512.csv --cluster rorqual --note 'tier=full;label=dense_l0p5_c512'
EOF
)
sleep 0.05

jid_dense_l0p5_c1400=$(sbatch --parsable \
    --job-name=psccal_dense_l0p5_c1400 \
    --output=$CAL_ROOT/logs/dense_l0p5_c1400_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p5_c1400.csv --cluster rorqual --note 'tier=full;label=dense_l0p5_c1400'
EOF
)
sleep 0.05

jid_dense_l0p5_c2800=$(sbatch --parsable \
    --job-name=psccal_dense_l0p5_c2800 \
    --output=$CAL_ROOT/logs/dense_l0p5_c2800_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=12G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '24576' --dense-c '2800' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p5_c2800.csv --cluster rorqual --note 'tier=full;label=dense_l0p5_c2800'
EOF
)
sleep 0.05

jid_dense_l0p75_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l0p75_c128 \
    --output=$CAL_ROOT/logs/dense_l0p75_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p75_c128.csv --cluster rorqual --note 'tier=full;label=dense_l0p75_c128'
EOF
)
sleep 0.05

jid_dense_l0p75_c512=$(sbatch --parsable \
    --job-name=psccal_dense_l0p75_c512 \
    --output=$CAL_ROOT/logs/dense_l0p75_c512_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p75_c512.csv --cluster rorqual --note 'tier=full;label=dense_l0p75_c512'
EOF
)
sleep 0.05

jid_dense_l0p75_c1400=$(sbatch --parsable \
    --job-name=psccal_dense_l0p75_c1400 \
    --output=$CAL_ROOT/logs/dense_l0p75_c1400_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=15G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p75_c1400.csv --cluster rorqual --note 'tier=full;label=dense_l0p75_c1400'
EOF
)
sleep 0.05

jid_dense_l0p75_c2800=$(sbatch --parsable \
    --job-name=psccal_dense_l0p75_c2800 \
    --output=$CAL_ROOT/logs/dense_l0p75_c2800_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=27G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '82944' --dense-c '2800' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l0p75_c2800.csv --cluster rorqual --note 'tier=full;label=dense_l0p75_c2800'
EOF
)
sleep 0.05

jid_dense_l1_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l1_c128 \
    --output=$CAL_ROOT/logs/dense_l1_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=8G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '196608' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l1_c128.csv --cluster rorqual --note 'tier=full;label=dense_l1_c128'
EOF
)
sleep 0.05

jid_dense_l1_c512=$(sbatch --parsable \
    --job-name=psccal_dense_l1_c512 \
    --output=$CAL_ROOT/logs/dense_l1_c512_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=14G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '196608' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l1_c512.csv --cluster rorqual --note 'tier=full;label=dense_l1_c512'
EOF
)
sleep 0.05

jid_dense_l1_c1400=$(sbatch --parsable \
    --job-name=psccal_dense_l1_c1400 \
    --output=$CAL_ROOT/logs/dense_l1_c1400_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=29G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '196608' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l1_c1400.csv --cluster rorqual --note 'tier=full;label=dense_l1_c1400'
EOF
)
sleep 0.05

jid_dense_l1_c2800=$(sbatch --parsable \
    --job-name=psccal_dense_l1_c2800 \
    --output=$CAL_ROOT/logs/dense_l1_c2800_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=55G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '196608' --dense-c '2800' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l1_c2800.csv --cluster rorqual --note 'tier=full;label=dense_l1_c2800'
EOF
)
sleep 0.05

jid_dense_l2agiso_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l2agiso_c128 \
    --output=$CAL_ROOT/logs/dense_l2agiso_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=9G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '393216' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l2agiso_c128.csv --cluster rorqual --note 'tier=full;label=dense_l2agiso_c128'
EOF
)
sleep 0.05

jid_dense_l2agiso_c512=$(sbatch --parsable \
    --job-name=psccal_dense_l2agiso_c512 \
    --output=$CAL_ROOT/logs/dense_l2agiso_c512_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=23G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '393216' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l2agiso_c512.csv --cluster rorqual --note 'tier=full;label=dense_l2agiso_c512'
EOF
)
sleep 0.05

jid_dense_l2agiso_c1400=$(sbatch --parsable \
    --job-name=psccal_dense_l2agiso_c1400 \
    --output=$CAL_ROOT/logs/dense_l2agiso_c1400_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=54G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '393216' --dense-c '1400' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l2agiso_c1400.csv --cluster rorqual --note 'tier=full;label=dense_l2agiso_c1400'
EOF
)
sleep 0.05

jid_dense_l3aniso_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l3aniso_c128 \
    --output=$CAL_ROOT/logs/dense_l3aniso_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=11G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '589824' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l3aniso_c128.csv --cluster rorqual --note 'tier=full;label=dense_l3aniso_c128'
EOF
)
sleep 0.05

jid_dense_l3aniso_c512=$(sbatch --parsable \
    --job-name=psccal_dense_l3aniso_c512 \
    --output=$CAL_ROOT/logs/dense_l3aniso_c512_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=32G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '589824' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l3aniso_c512.csv --cluster rorqual --note 'tier=full;label=dense_l3aniso_c512'
EOF
)
sleep 0.05

jid_dense_l4aniso_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l4aniso_c128 \
    --output=$CAL_ROOT/logs/dense_l4aniso_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=14G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '786432' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l4aniso_c128.csv --cluster rorqual --note 'tier=full;label=dense_l4aniso_c128'
EOF
)
sleep 0.05

jid_dense_l4aniso_c512=$(sbatch --parsable \
    --job-name=psccal_dense_l4aniso_c512 \
    --output=$CAL_ROOT/logs/dense_l4aniso_c512_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=41G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '786432' --dense-c '512' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l4aniso_c512.csv --cluster rorqual --note 'tier=full;label=dense_l4aniso_c512'
EOF
)
sleep 0.05

jid_dense_l2iso_c128=$(sbatch --parsable \
    --job-name=psccal_dense_l2iso_c128 \
    --output=$CAL_ROOT/logs/dense_l2iso_c128_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=23G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind dense --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --dense-m '1572864' --dense-c '128' --reps '12' --gpu 0 --root $CAL_ROOT --out $ROWS/dense_l2iso_c128.csv --cluster rorqual --note 'tier=full;label=dense_l2iso_c128'
EOF
)
sleep 0.05

jid_boundscore_l0p25_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p25_k256 \
    --output=$CAL_ROOT/logs/boundscore_l0p25_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p25_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p25_k256'
EOF
)
sleep 0.05

jid_boundscore_l0p25_k800=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p25_k800 \
    --output=$CAL_ROOT/logs/boundscore_l0p25_k800_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p25_k800.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p25_k800'
EOF
)
sleep 0.05

jid_boundscore_l0p25_k1350=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p25_k1350 \
    --output=$CAL_ROOT/logs/boundscore_l0p25_k1350_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '1350' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p25_k1350.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p25_k1350'
EOF
)
sleep 0.05

jid_boundscore_l0p5_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p5_k256 \
    --output=$CAL_ROOT/logs/boundscore_l0p5_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p5_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p5_k256'
EOF
)
sleep 0.05

jid_boundscore_l0p5_k800=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p5_k800 \
    --output=$CAL_ROOT/logs/boundscore_l0p5_k800_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p5_k800.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p5_k800'
EOF
)
sleep 0.05

jid_boundscore_l0p5_k1350=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p5_k1350 \
    --output=$CAL_ROOT/logs/boundscore_l0p5_k1350_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '1350' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p5_k1350.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p5_k1350'
EOF
)
sleep 0.05

jid_boundscore_l0p75_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p75_k256 \
    --output=$CAL_ROOT/logs/boundscore_l0p75_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=16G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p75_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p75_k256'
EOF
)
sleep 0.05

jid_boundscore_l0p75_k800=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p75_k800 \
    --output=$CAL_ROOT/logs/boundscore_l0p75_k800_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=23G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p75_k800.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p75_k800'
EOF
)
sleep 0.05

jid_boundscore_l0p75_k1350=$(sbatch --parsable \
    --job-name=psccal_boundscore_l0p75_k1350 \
    --output=$CAL_ROOT/logs/boundscore_l0p75_k1350_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=34G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '1350' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l0p75_k1350.csv --cluster rorqual --note 'tier=full;label=boundscore_l0p75_k1350'
EOF
)
sleep 0.05

jid_boundscore_l1_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l1_k256 \
    --output=$CAL_ROOT/logs/boundscore_l1_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=20G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l1_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l1_k256'
EOF
)
sleep 0.05

jid_boundscore_l1_k800=$(sbatch --parsable \
    --job-name=psccal_boundscore_l1_k800 \
    --output=$CAL_ROOT/logs/boundscore_l1_k800_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=44G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l1_k800.csv --cluster rorqual --note 'tier=full;label=boundscore_l1_k800'
EOF
)
sleep 0.05

jid_boundscore_l2agiso_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l2agiso_k256 \
    --output=$CAL_ROOT/logs/boundscore_l2agiso_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=31G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l2agiso_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l2agiso_k256'
EOF
)
sleep 0.05

jid_boundscore_l2agiso_k800=$(sbatch --parsable \
    --job-name=psccal_boundscore_l2agiso_k800 \
    --output=$CAL_ROOT/logs/boundscore_l2agiso_k800_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=79G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l2agiso_k800.csv --cluster rorqual --note 'tier=full;label=boundscore_l2agiso_k800'
EOF
)
sleep 0.05

jid_boundscore_l2agiso_k1350=$(sbatch --parsable \
    --job-name=psccal_boundscore_l2agiso_k1350 \
    --output=$CAL_ROOT/logs/boundscore_l2agiso_k1350_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=124G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '1350' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l2agiso_k1350.csv --cluster rorqual --note 'tier=full;label=boundscore_l2agiso_k1350'
EOF
)
sleep 0.05

jid_boundscore_l3aniso_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l3aniso_k256 \
    --output=$CAL_ROOT/logs/boundscore_l3aniso_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=42G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l3aniso_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l3aniso_k256'
EOF
)
sleep 0.05

jid_boundscore_l3aniso_k800=$(sbatch --parsable \
    --job-name=psccal_boundscore_l3aniso_k800 \
    --output=$CAL_ROOT/logs/boundscore_l3aniso_k800_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=114G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '800' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l3aniso_k800.csv --cluster rorqual --note 'tier=full;label=boundscore_l3aniso_k800'
EOF
)
sleep 0.05

jid_boundscore_l4aniso_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l4aniso_k256 \
    --output=$CAL_ROOT/logs/boundscore_l4aniso_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=53G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l4aniso_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l4aniso_k256'
EOF
)
sleep 0.05

jid_boundscore_l4aniso_k600=$(sbatch --parsable \
    --job-name=psccal_boundscore_l4aniso_k600 \
    --output=$CAL_ROOT/logs/boundscore_l4aniso_k600_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=114G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --sep '1//4' --rank '600' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l4aniso_k600.csv --cluster rorqual --note 'tier=full;label=boundscore_l4aniso_k600'
EOF
)
sleep 0.05

jid_boundscore_l2iso_k256=$(sbatch --parsable \
    --job-name=psccal_boundscore_l2iso_k256 \
    --output=$CAL_ROOT/logs/boundscore_l2iso_k256_%j.out \
    --account=def-smolesky \
    --time=04:00:00 \
    --cpus-per-task=4 \
    --mem=98G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind bounds_core --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --sep '1//4' --rank '256' --oversamples '50' --power-iters '14' --num-pos-frac '0.5' --outer-samples '4' --gpu 0 --root $CAL_ROOT --out $ROWS/boundscore_l2iso_k256.csv --cluster rorqual --note 'tier=full;label=boundscore_l2iso_k256'
EOF
)
sleep 0.05

echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo
echo "When they have finished, merge the per-point rows and copy the result back:"
echo "  bash bench/launch_calibration_rorqual_full.sh --merge"
echo "  scp pvirally@rorqual.alliancecan.ca:$OUT bench/data/"

