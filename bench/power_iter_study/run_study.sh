#!/bin/bash

# Power-iteration quality study, molering. No scheduler: plain sequential julia
# invocations, one process per GPU, same command-line idiom as
# jobs/launch_molering_Ge1000_arxivV3_*_gpu[01].sh.
#
#   bash bench/power_iter_study/run_study.sh <gpu-index>
#
# Environment overrides:
#   STAGES=greens|rsvd|bounds|study|all   default all (study = rsvd + bounds)
#   TIER=half|full                        default half (1/2 lambda); full is the
#                                         1 lambda stretch tier, RSVD only
#   RANK=4000  OVERSAMPLES=50  QS="1 2 3 4 6 8"  SEED=20260814
#   GAMMA_RTOL=1.0e-12
#
# Every q gets its own scratch and project subdirectory. The RSVD and bounds
# filenames come from file_prefix(smr), which encodes geometry and separation and
# nothing else, so runs at different q would otherwise overwrite each other.
#
# G0 is q-independent and lands in the shared preload directory (preload_dir in
# src/common.jl, not --scratch), so every q reuses the same blocks. Generate them
# once, before either GPU process starts, or the two will race to build the same
# file:
#
#   STAGES=greens bash bench/power_iter_study/run_study.sh 0
#
# then start run_study_gpu0.sh and run_study_gpu1.sh together.
#
# The reference is q=8, not the production q=14: no single run on molering is
# allowed to pass about an hour, and q=14 at 1/2 lambda is ~72 min. q=8 is ~50
# min. That makes the reference something to be checked rather than trusted, so
# the analysis gates on the q=6 vs q=8 deviation sitting at the sketch-noise
# floor before it will issue a verdict at all. See the README.
#
# NOTE on --seed: the seed only reaches the sketch on the panel RSVD path. At
# 1/2 lambda with k=4000 the sketch fits an A6000, so src/rsvd.jl takes the
# in-memory path, which draws from the RNG and rejects a seed. The reference is
# therefore run twice (q08/ and q08r/, see QS_REPLICATE below) so the analysis
# has a sketch-noise floor to compare the low-q deviations against. Read the
# README before changing that.

GPU=${1:?usage: run_study.sh <gpu-index>}

# --- fixed by the production sweep --------------------------------------------
CHI="4.25+0.0342557im"
DESIGN="rs"
SCALE="1//32"
THREADS=16

# --- knobs -------------------------------------------------------------------
RANK=${RANK:-4000}
OVERSAMPLES=${OVERSAMPLES:-50}
# Top out at 8. At 1/2 lambda that is ~50 min; q=14 is ~72 min and over the
# one-hour-per-run ceiling this study is held to.
QS=${QS:-"1 2 3 4 6 8"}
SEED=${SEED:-20260814}
GAMMA_RTOL=${GAMMA_RTOL:-1.0e-12}
STAGES=${STAGES:-all}
TIER=${TIER:-half}
# Second run of the reference q. Same q, its own directory, so the analysis can
# separate sketch noise from the effect of dropping power iterations, and can
# check that the reference itself has converged. Must stay equal to the largest
# entry in QS. Set to an empty string to skip it, and lose both checks.
QS_REPLICATE=${QS_REPLICATE:-8}

# The reference is the largest q in QS, which is what analyze.jl defaults --ref
# to. Derived rather than written twice so an overridden QS cannot disagree with
# it.
REF_Q=$(for q in $QS; do echo "$q"; done | sort -n | tail -1)
if [ -n "$QS_REPLICATE" ] && [ "$QS_REPLICATE" != "$REF_Q" ]; then
    echo "QS_REPLICATE=${QS_REPLICATE} is not the largest q in QS (${REF_Q}); the" >&2
    echo "convergence check needs the replicate to be at the reference." >&2
    exit 2
fi

# --- molering paths ----------------------------------------------------------
CODE_DIR=/home/paulv/Projects/Photonic-System-Channels
SCRATCH_ROOT=/home/molering/fatmole/paulv/Photonic-System-Channels/power_iter_study/k${RANK}
PROJECT_ROOT=${CODE_DIR}/projects/power_iter_study/k${RANK}
PRELOAD=/home/molering/fatmole/greens_functions

# --- tiers -------------------------------------------------------------------
# "<x-separation as a rational>:<the name fragment the narval 0p5 sweep uses>".
# All three separations are ones jobs/greens_tasks_narval_Ge1000_arxivV3_0p5*.txt
# actually runs: 1/16 near-field, 5/8 close-coupling, 5/2 the problem region.
HALF_CELLS="16,16,16"
HALF_SEPS="1//16:1ss16 5//8:5ss8 5//2:5ss2"
# Stretch tier. At 1 lambda, k=4000 the sketch no longer fits an A6000, so the
# RSVD takes the panel path and the bounds stage is out of reach on this machine
# (README, "What this costs"). Run it as STAGES=rsvd and read the spectra only.
FULL_CELLS="32,32,32"
FULL_SEPS="1//16:1ss16 5//8:5ss8 5//2:5ss2"

case "$TIER" in
    half) TAG=0p5; CELLS=$HALF_CELLS; SEPS=$HALF_SEPS ;;
    full) TAG=1x1; CELLS=$FULL_CELLS; SEPS=$FULL_SEPS ;;
    *) echo "unknown TIER '$TIER' (half|full)" >&2; exit 2 ;;
esac

CSV=${PROJECT_ROOT}/timings_gpu${GPU}.csv

mkdir -p "$SCRATCH_ROOT" "$PROJECT_ROOT" "$PRELOAD"
if [ ! -f "$CSV" ]; then
    echo "tier,separation,q,stage,seconds,status" > "$CSV"
fi

cd "$CODE_DIR" || exit 1

echo "Power-iteration study on molering, GPU ${GPU}"
echo "  tier ${TIER} (${TAG}), cells (${CELLS}), scale ${SCALE}"
echo "  rank ${RANK}, oversamples ${OVERSAMPLES}, seed ${SEED}, gamma-rtol ${GAMMA_RTOL}"
echo "  q values: ${QS} (reference q=${REF_Q}, replicate q=${QS_REPLICATE:-none})"
echo "  stages: ${STAGES}"
echo "  scratch ${SCRATCH_ROOT}"
echo "  project ${PROJECT_ROOT}"
echo "  timings ${CSV}"

want_stage() {
    case "$STAGES" in
        all)    return 0 ;;
        greens) [ "$1" = greens ] ;;
        study)  [ "$1" = rsvd ] || [ "$1" = bounds ] ;;
        rsvd)   [ "$1" = rsvd ] ;;
        bounds) [ "$1" = bounds ] ;;
        *) echo "unknown STAGES '$STAGES' (greens|rsvd|bounds|study|all)" >&2; exit 2 ;;
    esac
}

stamp() { date +%Y-%m-%dT%H:%M:%S; }

# run_stage <stage> <separation> <q> <command...>: time it, log one CSV line.
run_stage() {
    local stage=$1 sep=$2 q=$3
    shift 3
    echo "[$(stamp)] START ${stage} sep=${sep} q=${q}"
    local t0 t1 rc
    t0=$(date +%s)
    "$@"
    rc=$?
    t1=$(date +%s)
    echo "${TAG},${sep},${q},${stage},$((t1 - t0)),${rc}" >> "$CSV"
    echo "[$(stamp)] END   ${stage} sep=${sep} q=${q} $((t1 - t0))s rc=${rc}"
    return $rc
}

name_for() { echo "(${CELLS})_$1_(${CELLS})@(1ss32,1ss32,1ss32)"; }
qdir() { printf 'q%02d' "$1"; }

# --- greens: once per separation, CPU, into the shared preload ----------------
# --power-iterations is irrelevant here; it is passed only because parse_args
# logs the whole RSVDParams and the sweep's launchers pass it too.
if want_stage greens && [ "$GPU" = "0" ]; then
    for entry in $SEPS; do
        sep=${entry%%:*}
        tag=${entry##*:}
        name=$(name_for "$tag")
        run_stage greens "$sep" 0 \
            julia --project=. -t $THREADS generate_green.jl \
            --sender "($CELLS)" --receiver "($CELLS)" \
            --rs-sep "($sep,0//1,0//1)" --scale "$SCALE" --chi "$CHI" \
            --design "$DESIGN" --components "$RANK" --oversamples "$OVERSAMPLES" \
            --power-iterations "$REF_Q" --seed "$SEED" \
            --name "$name" --gpu false \
            --preload "$PRELOAD" \
            --project "${PROJECT_ROOT}/greens" --scratch "${SCRATCH_ROOT}/greens"
    done
fi

# --- the (separation, q) task list -------------------------------------------
# Alternating assignment, like the generated gpu0/gpu1 pair. The bounds stage
# costs the same at every q, so alternating balances the two GPUs better than
# splitting by q would.
ALL_QS="$QS"
if [ -n "$QS_REPLICATE" ]; then
    # The replicate shares its q with the reference but needs its own directory,
    # so it is carried as "<q>r" and the r is stripped before it reaches julia.
    ALL_QS="$QS ${QS_REPLICATE}r"
fi

tasks=""
for entry in $SEPS; do
    for q in $ALL_QS; do
        tasks="$tasks $entry|$q"
    done
done

# RSVD for every task this GPU owns first, then the bounds. The spectra are the
# cheap half of the study and are worth having complete before the expensive half
# starts.
for stage in rsvd bounds; do
    want_stage "$stage" || continue
    i=-1
    for task in $tasks; do
        i=$((i + 1))
        [ $((i % 2)) -eq "$GPU" ] || continue

        entry=${task%%|*}
        qraw=${task##*|}
        sep=${entry%%:*}
        tag=${entry##*:}
        q=${qraw%r}
        name=$(name_for "$tag")
        sub=$(qdir "$q")
        [ "$q" = "$qraw" ] || sub="${sub}r"
        scratch="${SCRATCH_ROOT}/${sub}"
        project="${PROJECT_ROOT}/${sub}"
        mkdir -p "$scratch" "$project"

        if [ "$stage" = rsvd ]; then
            run_stage rsvd "$sep" "$qraw" \
                julia --project=. -t $THREADS generate_rsvd.jl \
                --sender "($CELLS)" --receiver "($CELLS)" \
                --rs-sep "($sep,0//1,0//1)" --scale "$SCALE" --chi "$CHI" \
                --design "$DESIGN" --components "$RANK" --oversamples "$OVERSAMPLES" \
                --power-iterations "$q" --seed "$SEED" \
                --name "$name" --gpu "$GPU" \
                --preload "$PRELOAD" --project "$project" --scratch "$scratch"
        else
            run_stage bounds "$sep" "$qraw" \
                julia --project=. -t $THREADS compute_bounds.jl \
                --sender "($CELLS)" --receiver "($CELLS)" \
                --rs-sep "($sep,0//1,0//1)" --scale "$SCALE" --chi "$CHI" \
                --design "$DESIGN" --components "$RANK" --oversamples "$OVERSAMPLES" \
                --power-iterations "$q" --seed "$SEED" \
                --name "$name" --gpu "$GPU" --gamma-rtol "$GAMMA_RTOL" \
                --preload "$PRELOAD" --project "$project" --scratch "$scratch"
        fi
    done
done

echo "[$(stamp)] GPU ${GPU} done. Timings in ${CSV}"
echo "Analyse with:"
echo "  julia --project=. bench/power_iter_study/analyze.jl --root ${PROJECT_ROOT}"
