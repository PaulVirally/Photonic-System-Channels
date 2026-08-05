#!/bin/bash
# Cost-model calibration for rorqual, tier=memory.
# Generated 2026-08-05T12:30:32.175 by bench/plan.jl. Do not edit; regenerate instead.
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

echo "Submitting 16 calibration points for rorqual (tier=memory)"
echo "Each point writes its own row file under $ROWS"

jid_memgreens_l0p25=$(sbatch --parsable \
    --job-name=psccal_memgreens_l0p25 \
    --output=$CAL_ROOT/logs/memgreens_l0p25_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l0p25.csv --cluster rorqual --note 'tier=memory;label=memgreens_l0p25'
EOF
)
sleep 0.05

jid_memrsvd_l0p25=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l0p25} \
    --job-name=psccal_memrsvd_l0p25 \
    --output=$CAL_ROOT/logs/memrsvd_l0p25_%j.out \
    --account=def-smolesky \
    --time=00:30:00 \
    --cpus-per-task=4 \
    --mem=32G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '8,8,8' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l0p25.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l0p25'
EOF
)
sleep 0.05

jid_memgreens_l0p5=$(sbatch --parsable \
    --job-name=psccal_memgreens_l0p5 \
    --output=$CAL_ROOT/logs/memgreens_l0p5_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l0p5.csv --cluster rorqual --note 'tier=memory;label=memgreens_l0p5'
EOF
)
sleep 0.05

jid_memrsvd_l0p5=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l0p5} \
    --job-name=psccal_memrsvd_l0p5 \
    --output=$CAL_ROOT/logs/memrsvd_l0p5_%j.out \
    --account=def-smolesky \
    --time=00:30:00 \
    --cpus-per-task=4 \
    --mem=32G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '16,16,16' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l0p5.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l0p5'
EOF
)
sleep 0.05

jid_memgreens_l0p75=$(sbatch --parsable \
    --job-name=psccal_memgreens_l0p75 \
    --output=$CAL_ROOT/logs/memgreens_l0p75_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l0p75.csv --cluster rorqual --note 'tier=memory;label=memgreens_l0p75'
EOF
)
sleep 0.05

jid_memrsvd_l0p75=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l0p75} \
    --job-name=psccal_memrsvd_l0p75 \
    --output=$CAL_ROOT/logs/memrsvd_l0p75_%j.out \
    --account=def-smolesky \
    --time=00:30:20 \
    --cpus-per-task=4 \
    --mem=32G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '24,24,24' --scale '1//32' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l0p75.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l0p75'
EOF
)
sleep 0.05

jid_memgreens_l1=$(sbatch --parsable \
    --job-name=psccal_memgreens_l1 \
    --output=$CAL_ROOT/logs/memgreens_l1_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l1.csv --cluster rorqual --note 'tier=memory;label=memgreens_l1'
EOF
)
sleep 0.05

jid_memrsvd_l1=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l1} \
    --job-name=psccal_memrsvd_l1 \
    --output=$CAL_ROOT/logs/memrsvd_l1_%j.out \
    --account=def-smolesky \
    --time=01:29:11 \
    --cpus-per-task=4 \
    --mem=42G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --rank '2750' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l1.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l1'
EOF
)
sleep 0.05

jid_memgreens_l2agiso=$(sbatch --parsable \
    --job-name=psccal_memgreens_l2agiso \
    --output=$CAL_ROOT/logs/memgreens_l2agiso_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l2agiso.csv --cluster rorqual --note 'tier=memory;label=memgreens_l2agiso'
EOF
)
sleep 0.05

jid_memrsvd_l2agiso=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l2agiso} \
    --job-name=psccal_memrsvd_l2agiso \
    --output=$CAL_ROOT/logs/memrsvd_l2agiso_%j.out \
    --account=def-smolesky \
    --time=01:10:58 \
    --cpus-per-task=4 \
    --mem=41G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '64,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '1350' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l2agiso.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l2agiso'
EOF
)
sleep 0.05

jid_memgreens_l3aniso=$(sbatch --parsable \
    --job-name=psccal_memgreens_l3aniso \
    --output=$CAL_ROOT/logs/memgreens_l3aniso_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l3aniso.csv --cluster rorqual --note 'tier=memory;label=memgreens_l3aniso'
EOF
)
sleep 0.05

jid_memrsvd_l3aniso=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l3aniso} \
    --job-name=psccal_memrsvd_l3aniso \
    --output=$CAL_ROOT/logs/memrsvd_l3aniso_%j.out \
    --account=def-smolesky \
    --time=00:59:41 \
    --cpus-per-task=4 \
    --mem=38G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '96,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '800' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l3aniso.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l3aniso'
EOF
)
sleep 0.05

jid_memgreens_l4aniso=$(sbatch --parsable \
    --job-name=psccal_memgreens_l4aniso \
    --output=$CAL_ROOT/logs/memgreens_l4aniso_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l4aniso.csv --cluster rorqual --note 'tier=memory;label=memgreens_l4aniso'
EOF
)
sleep 0.05

jid_memrsvd_l4aniso=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l4aniso} \
    --job-name=psccal_memrsvd_l4aniso \
    --output=$CAL_ROOT/logs/memrsvd_l4aniso_%j.out \
    --account=def-smolesky \
    --time=00:58:34 \
    --cpus-per-task=4 \
    --mem=38G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '128,32,32' --scale '-1//8' --chi '13.6+0.05im' --rank '600' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l4aniso.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l4aniso'
EOF
)
sleep 0.05

jid_memgreens_l2iso=$(sbatch --parsable \
    --job-name=psccal_memgreens_l2iso \
    --output=$CAL_ROOT/logs/memgreens_l2iso_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --power-iters '14' --sep '1//4' --gpu -1 --root $CAL_ROOT --out $ROWS/memgreens_l2iso.csv --cluster rorqual --note 'tier=memory;label=memgreens_l2iso'
EOF
)
sleep 0.05

jid_memrsvd_l2iso=$(sbatch --parsable \
    --dependency=afterok:${jid_memgreens_l2iso} \
    --job-name=psccal_memrsvd_l2iso \
    --output=$CAL_ROOT/logs/memrsvd_l2iso_%j.out \
    --account=def-smolesky \
    --time=01:51:25 \
    --cpus-per-task=4 \
    --mem=66G \
    --gpus=h100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind mem_rsvd --cells '64,64,64' --scale '1//32' --chi '13.6+0.05im' --rank '600' --oversamples '50' --sep '1//4' --power-iters '2' --gpu 0 --root $CAL_ROOT --out $ROWS/memrsvd_l2iso.csv --cluster rorqual --note 'tier=memory;label=memrsvd_l2iso'
EOF
)
sleep 0.05

echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo
echo "When they have finished, merge the per-point rows and copy the result back:"
echo "  bash bench/launch_calibration_rorqual_memory.sh --merge"
echo "  scp pvirally@rorqual.alliancecan.ca:$OUT bench/data/"

