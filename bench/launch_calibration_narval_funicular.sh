#!/bin/bash
# Cost-model calibration for narval, tier=funicular.
# Generated 2026-08-14T19:41:58.015 by bench/plan.jl. Do not edit; regenerate instead.
#
# Every point is its own job: one point running out of memory or time must
# not take the rest of the calibration with it. Each writes its own row file,
# so partial results are always usable.
#
# Submit:  bash <this script>
# Collect: bash <this script> --merge

# ---------------------------------------------------------------------------
# Before submitting: three things, on the LOGIN node, in this order.
#
# 1. Instantiate. Compute nodes have no internet, and this tier is the first
#    thing here to need Funicular and HDF5, both of which come in by URL:
#
#      module load StdEnv/2023 julia/1.12.5 cuda/12.2
#      cd /home/pvirally/Photonic-System-Channels/
#      julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
#      julia --project=. -e 'using PhotonicSystemChannels, Funicular, CUDA; println(pkgdir(Funicular))'
#
#    The last line is the one that matters for E1. It prints the depot path
#    whose `benchmark/` subdirectory holds `overlap.jl` and `pinned.jl`, which
#    is what `bench/point.jl` resolves through `pkgdir(Funicular)`.
#
# 2. Funicular's `benchmark/Project.toml` does NOT need instantiating. It
#    lists DelimitedFiles and Plots, and those are for `benchmark/plot.jl`
#    only. `overlap.jl` and `pinned.jl` need CUDA, Funicular and Printf, all of
#    which this project already has, so the E1 point runs them against the main
#    project environment. It copies the directory under the point's --scratch
#    first, because `benchmark/common.jl` writes its TSV results next to itself
#    and the depot is not ours to write into.
#
# 3. Check the slice names are still what this script asks for, since a name
#    the cluster does not define is a hard sbatch rejection:
#
#      sinfo -o "%G" | sort -u
#
# Trial E3c (`e3c_l4_spill`) spills to node-local NVMe through
# $SLURM_TMPDIR. It asks for --mem=66G specifically:
# `residency_plan` reads SLURM_MEM_PER_NODE and subtracts a 6 GiB overhead
# reserve, so that request is a 60 GiB host budget exactly, against a ~95 GiB
# panel peak. No --tmp is requested (narval GPU nodes carry NVMe and the flag
# is not universally accepted); if the job dies writing spill files, check that
# $SLURM_TMPDIR has ~120 GB free on the node it landed on.
#
# The three E4 points need workstream C (the panelized bounds front-end in
# src/bounds.jl). They are written against the CLI that exists today, but at
# k = 4000 the old in-memory front-end would want an N_u x m basis as one
# CuArray, ~30 GB at 4 lambda, so submit them only once C has landed. The
# E1-E3 points do not depend on it; comment the E4 block out to run the rest.
# ---------------------------------------------------------------------------

set -u

CODE_DIR=/home/pvirally/Photonic-System-Channels/
CAL_ROOT=/home/pvirally/scratch/psc-calibration/
ROWS=$CAL_ROOT/rows_funicular
OUT=$CAL_ROOT/calibration_narval_funicular.csv

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

echo "Submitting 14 calibration points for narval (tier=funicular)"
echo "Each point writes its own row file under $ROWS"

jid_e1_panelbus=$(sbatch --parsable \
    --job-name=psccal_e1_panelbus \
    --output=$CAL_ROOT/logs/e1_panelbus_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=32G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind panel_bus --scratch "$CAL_ROOT/funicular/e1_panelbus" --reps '5' --gpu 0 --root $CAL_ROOT --out $ROWS/e1_panelbus.csv --cluster narval --note 'tier=funicular;label=e1_panelbus'
EOF
)
sleep 0.05

jid_fungreens_l1=$(sbatch --parsable \
    --job-name=psccal_fungreens_l1 \
    --output=$CAL_ROOT/logs/fungreens_l1_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --gpu -1 --root $CAL_ROOT --out $ROWS/fungreens_l1.csv --cluster narval --note 'tier=funicular;label=fungreens_l1'
EOF
)
sleep 0.05

jid_fungreens_l2=$(sbatch --parsable \
    --job-name=psccal_fungreens_l2 \
    --output=$CAL_ROOT/logs/fungreens_l2_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '64,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --gpu -1 --root $CAL_ROOT --out $ROWS/fungreens_l2.csv --cluster narval --note 'tier=funicular;label=fungreens_l2'
EOF
)
sleep 0.05

jid_fungreens_l4=$(sbatch --parsable \
    --job-name=psccal_fungreens_l4 \
    --output=$CAL_ROOT/logs/fungreens_l4_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_greens --cells '128,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --gpu -1 --root $CAL_ROOT --out $ROWS/fungreens_l4.csv --cluster narval --note 'tier=funicular;label=fungreens_l4'
EOF
)
sleep 0.05

jid_e2_l1_inmem=$(sbatch --parsable \
    --dependency=afterok:${jid_fungreens_l1} \
    --job-name=psccal_e2_l1_inmem \
    --output=$CAL_ROOT/logs/e2_l1_inmem_%j.out \
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
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '1350' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e2_l1_inmem" --fresh --force-path 'auto' --gpu 0 --root $CAL_ROOT --out $ROWS/e2_l1_inmem.csv --cluster narval --note 'tier=funicular;label=e2_l1_inmem'
EOF
)
sleep 0.05

jid_e2_l1_panel=$(sbatch --parsable \
    --dependency=afterok:${jid_fungreens_l1} \
    --job-name=psccal_e2_l1_panel \
    --output=$CAL_ROOT/logs/e2_l1_panel_%j.out \
    --account=def-smolesky \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=20G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '1350' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e2_l1_panel" --fresh --force-path 'panel' --gpu 0 --root $CAL_ROOT --out $ROWS/e2_l1_panel.csv --cluster narval --note 'tier=funicular;label=e2_l1_panel'
EOF
)
sleep 0.05

jid_e3d_l1_k4000=$(sbatch --parsable \
    --dependency=afterok:${jid_fungreens_l1} \
    --job-name=psccal_e3d_l1_k4000 \
    --output=$CAL_ROOT/logs/e3d_l1_k4000_%j.out \
    --account=def-smolesky \
    --time=06:09:13 \
    --cpus-per-task=4 \
    --mem=37G \
    --gpus=a100_3g.20gb:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e3d_l1_k4000" --fresh --force-path 'panel' --gpu 0 --root $CAL_ROOT --out $ROWS/e3d_l1_k4000.csv --cluster narval --note 'tier=funicular;label=e3d_l1_k4000'
EOF
)
sleep 0.05

jid_e3a_l2_slice=$(sbatch --parsable \
    --dependency=afterok:${jid_fungreens_l2} \
    --job-name=psccal_e3a_l2_slice \
    --output=$CAL_ROOT/logs/e3a_l2_slice_%j.out \
    --account=def-smolesky \
    --time=10:39:50 \
    --cpus-per-task=4 \
    --mem=60G \
    --gpus=a100_3g.20gb:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e3a_l2_slice" --fresh --force-path 'panel' --gpu 0 --root $CAL_ROOT --out $ROWS/e3a_l2_slice.csv --cluster narval --note 'tier=funicular;label=e3a_l2_slice'
EOF
)
sleep 0.05

jid_e3a_l2_full=$(sbatch --parsable \
    --dependency=afterok:${jid_fungreens_l2} \
    --job-name=psccal_e3a_l2_full \
    --output=$CAL_ROOT/logs/e3a_l2_full_%j.out \
    --account=def-smolesky \
    --time=04:03:09 \
    --cpus-per-task=4 \
    --mem=60G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '64,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e3a_l2_full" --fresh --force-path 'panel' --gpu 0 --root $CAL_ROOT --out $ROWS/e3a_l2_full.csv --cluster narval --note 'tier=funicular;label=e3a_l2_full'
EOF
)
sleep 0.05

jid_e3b_l4_full=$(sbatch --parsable \
    --dependency=afterok:${jid_fungreens_l4} \
    --job-name=psccal_e3b_l4_full \
    --output=$CAL_ROOT/logs/e3b_l4_full_%j.out \
    --account=def-smolesky \
    --time=07:38:11 \
    --cpus-per-task=4 \
    --mem=118G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '128,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e3b_l4_full" --fresh --force-path 'panel' --gpu 0 --root $CAL_ROOT --out $ROWS/e3b_l4_full.csv --cluster narval --note 'tier=funicular;label=e3b_l4_full'
EOF
)
sleep 0.05

jid_e3c_l4_spill=$(sbatch --parsable \
    --dependency=afterok:${jid_fungreens_l4} \
    --job-name=psccal_e3c_l4_spill \
    --output=$CAL_ROOT/logs/e3c_l4_spill_%j.out \
    --account=def-smolesky \
    --time=12:13:06 \
    --cpus-per-task=4 \
    --mem=66G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_rsvd --cells '128,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e3c_l4_spill" --fresh --force-path 'panel' --gpu 0 --root $CAL_ROOT --out $ROWS/e3c_l4_spill.csv --cluster narval --note 'tier=funicular;label=e3c_l4_spill'
EOF
)
sleep 0.05

jid_e4_bounds_l1_k4000=$(sbatch --parsable \
    --dependency=afterok:${jid_e3d_l1_k4000} \
    --job-name=psccal_e4_bounds_l1_k4000 \
    --output=$CAL_ROOT/logs/e4_bounds_l1_k4000_%j.out \
    --account=def-smolesky \
    --time=24:00:00 \
    --cpus-per-task=4 \
    --mem=34G \
    --gpus=a100_3g.20gb:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e3d_l1_k4000" --gpu 0 --root $CAL_ROOT --out $ROWS/e4_bounds_l1_k4000.csv --cluster narval --note 'tier=funicular;label=e4_bounds_l1_k4000'
EOF
)
sleep 0.05

jid_e4_bounds_l4_k4000=$(sbatch --parsable \
    --dependency=afterok:${jid_e3b_l4_full} \
    --job-name=psccal_e4_bounds_l4_k4000 \
    --output=$CAL_ROOT/logs/e4_bounds_l4_k4000_%j.out \
    --account=def-smolesky \
    --time=24:00:00 \
    --cpus-per-task=4 \
    --mem=104G \
    --gpus=a100:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '128,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '4000' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e3b_l4_full" --gpu 0 --root $CAL_ROOT --out $ROWS/e4_bounds_l4_k4000.csv --cluster narval --note 'tier=funicular;label=e4_bounds_l4_k4000'
EOF
)
sleep 0.05

jid_e4_bounds_l1_k1350=$(sbatch --parsable \
    --dependency=afterok:${jid_e2_l1_panel} \
    --job-name=psccal_e4_bounds_l1_k1350 \
    --output=$CAL_ROOT/logs/e4_bounds_l1_k1350_%j.out \
    --account=def-smolesky \
    --time=01:18:12 \
    --cpus-per-task=4 \
    --mem=19G \
    --gpus=a100_3g.20gb:1 \
    --chdir=$CODE_DIR \
    --export=ALL \
    <<EOF
#!/bin/bash
module load StdEnv/2023 julia/1.12.5 cuda/12.2
export PSC_T0=\$(date +%s)
srun julia --project=. -t 4 bench/point.jl --kind stage_bounds --cells '32,32,32' --scale '1//32' --chi '13.6+0.05im' --sep '1//2' --rank '1350' --oversamples '50' --power-iters '14' --seed '20260814' --scratch "$CAL_ROOT/funicular/e2_l1_panel" --gpu 0 --root $CAL_ROOT --out $ROWS/e4_bounds_l1_k1350.csv --cluster narval --note 'tier=funicular;label=e4_bounds_l1_k1350'
EOF
)
sleep 0.05

echo
echo "All points submitted. Watch them with: squeue -u \$USER"
echo
echo "When they have finished:"
echo
echo "  1. Trial E2's parity check (login node, no GPU needed). The two paths"
echo "     use different RNG mechanisms, so this reports the deviation of the"
echo "     top of the spectrum rather than asserting equality:"
echo
echo "     julia --project=. bench/compare_parity.jl \\"
echo "         --a $CAL_ROOT/funicular/e2_l1_inmem/*.jld \\"
echo "         --b $CAL_ROOT/funicular/e2_l1_panel/*.jld \\"
echo "         --label-a in-memory --label-b panel --rtol 1e-6"
echo
echo "  2. Merge the rows and copy them back:"
echo "     bash bench/launch_calibration_narval_funicular.sh --merge"
echo "     scp pvirally@narval.alliancecan.ca:$OUT bench/data/calibration_narval_funicular.csv"
echo
echo "  3. Refit. E1's panel_bus rows identify pcie_rate and overlap_factor;"
echo "     the E2/E3 stage_rsvd rows are what panel_host_mem_factor and"
echo "     panel_workspace_bytes have been waiting for:"
echo "     julia bench/fit.jl"
echo
echo "  4. Re-run create_jobs.jl and read its print_plan against the capacity"
echo "     table at the top of FUNICULAR_PLAN.md."

