#!/bin/bash
# ==============================================================
# AffectGPT corruption-based MELD evaluation entry point.
#
# Two-phase flow per condition:
#   1. preprocess_corruption.py  → meld-process-{cond}-{preset}/
#   2. cache_outputs.py          → cache/{cond}/meld.jsonl  (raw responses)
#   3. compute_metrics.py        → cache/{cond}/meld.metrics.json
#
# Designed for manual / tmux runs (not slurm). Run from inside the AffectGPT
# repo root (the dir containing setup_downloads.sh, missing_modality_eval.py,
# cache_outputs.py, compute_metrics.py).
#
# MELD source can live anywhere; pass via MELD_RAW (defaults to ./MELD.Raw to
# match setup_downloads.sh).
#
# Conditions evaluated (8 total):
#     none, text, audio, video, text_audio, text_video, audio_video,
#     text_audio_video
#
# Usage:
#     ./run_corruption_eval.sh                                         # all 8 conds, strong
#     PRESET=mild ./run_corruption_eval.sh
#     MELD_RAW=/mnt/palmerla/MELD.Raw ./run_corruption_eval.sh
#     CONDITIONS="none text audio_video" ./run_corruption_eval.sh
#     MAX_SAMPLES=20 ./run_corruption_eval.sh                          # quick smoke test
#     MAX_SAMPLES=20 CONDITIONS="none audio" ./run_corruption_eval.sh
# ==============================================================
set -eo pipefail

# ----- Configuration (override via env vars) -----
PRESET="${PRESET:-strong}"
SEED="${SEED:-42}"
# GPU: prefer CUDA_VISIBLE_DEVICES if the caller already set it; otherwise GPU=0
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU="${GPU:-${CUDA_VISIBLE_DEVICES}}"
else
    GPU="${GPU:-0}"
fi
CFG_PATH="${CFG_PATH:-train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml}"
TEST_EPOCH="${TEST_EPOCH:-60}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
LIMIT="${LIMIT:-0}"

ROOT_DIR="$(pwd)"
DATASET_DIR="${ROOT_DIR}/../dataset"
MELD_PROC="${DATASET_DIR}/meld-process"
MELD_PROC_CLEAN="${DATASET_DIR}/meld-process-clean"

# Source raw MELD (where setup_downloads.sh expects it). Make symlink from
# user-supplied location into the expected ./MELD.Raw before calling setup.
MELD_RAW="${MELD_RAW:-${ROOT_DIR}/MELD.Raw}"
MELD_RAW_LINK="${ROOT_DIR}/MELD.Raw"

# Cache + metrics output root
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/output/corruption_eval_${PRESET}}"

# Conditions (name → space-separated modalities to corrupt)
ALL_CONDITIONS=(
    "none"
    "text"
    "audio"
    "video"
    "text_audio"
    "text_video"
    "audio_video"
    "text_audio_video"
)
declare -A CORRUPT_MODS=(
    ["none"]=""
    ["text"]="text"
    ["audio"]="audio"
    ["video"]="video"
    ["text_audio"]="text audio"
    ["text_video"]="text video"
    ["audio_video"]="audio video"
    ["text_audio_video"]="text audio video"
)

if [ -n "${CONDITIONS:-}" ]; then
    read -r -a CONDITIONS_TO_RUN <<< "${CONDITIONS}"
else
    CONDITIONS_TO_RUN=("${ALL_CONDITIONS[@]}")
fi

echo "=============================================="
echo " AffectGPT MELD Corruption Eval"
echo "=============================================="
echo "ROOT_DIR:    ${ROOT_DIR}"
echo "MELD_RAW:    ${MELD_RAW}"
echo "DATASET_DIR: ${DATASET_DIR}"
echo "PRESET:      ${PRESET}"
echo "SEED:        ${SEED}"
echo "GPU:         ${GPU}"
echo "Conditions:  ${CONDITIONS_TO_RUN[*]}"
echo "Cache root:  ${RESULTS_ROOT}"
echo
mkdir -p "${RESULTS_ROOT}"

# ==============================================================
# Step 1 — Ensure base setup is in place
# ==============================================================
# Symlink user-supplied MELD.Raw into the location setup_downloads.sh expects
if [ ! -e "${MELD_RAW_LINK}" ]; then
    if [ -d "${MELD_RAW}" ]; then
        echo "Linking ${MELD_RAW} -> ${MELD_RAW_LINK}"
        ln -s "${MELD_RAW}" "${MELD_RAW_LINK}"
    fi
fi

if [ ! -d "${MELD_PROC}" ] && [ ! -d "${MELD_PROC_CLEAN}" ]; then
    echo "Running setup_downloads.sh (first time)..."
    bash setup_downloads.sh
fi

if [ ! -e "${MELD_PROC_CLEAN}" ]; then
    if [ -L "${MELD_PROC}" ]; then
        :  # already linked, nothing to back up
    elif [ -d "${MELD_PROC}" ]; then
        echo "Backing up clean MELD-process to ${MELD_PROC_CLEAN}..."
        mv "${MELD_PROC}" "${MELD_PROC_CLEAN}"
    else
        echo "ERROR: ${MELD_PROC_CLEAN} not found. Run setup_downloads.sh first." >&2
        exit 1
    fi
fi

# ==============================================================
# Step 2 — Per-condition: preprocess, cache, then compute metrics
# ==============================================================
restore_clean() {
    if [ -L "${MELD_PROC}" ] || [ -e "${MELD_PROC}" ]; then
        rm -f "${MELD_PROC}"
    fi
    ln -sfn "$(basename ${MELD_PROC_CLEAN})" "${MELD_PROC}"
}
trap restore_clean EXIT

for cond in "${CONDITIONS_TO_RUN[@]}"; do
    if [ -z "${CORRUPT_MODS[$cond]+x}" ]; then
        echo "WARNING: unknown condition '${cond}', skipping." >&2
        continue
    fi
    mods="${CORRUPT_MODS[$cond]}"

    if [ -z "${mods}" ]; then
        DST="${MELD_PROC_CLEAN}"
    else
        DST="${DATASET_DIR}/meld-process-${cond}-${PRESET}"
        if [ ! -d "${DST}" ]; then
            echo
            echo "----------------------------------------------"
            echo " [${cond}] Corrupting modalities: ${mods}"
            echo "----------------------------------------------"
            python preprocess_corruption.py \
                --src "${MELD_PROC_CLEAN}" \
                --dst "${DST}" \
                --corrupt_modalities ${mods} \
                --preset "${PRESET}" \
                --seed "${SEED}" \
                --limit "${LIMIT}"
        else
            echo "[${cond}] Corrupted dir exists — reusing: ${DST}"
        fi
    fi

    SAVE_DIR="${RESULTS_ROOT}/${cond}"
    JSONL="${SAVE_DIR}/meld.jsonl"
    METRICS="${SAVE_DIR}/meld.metrics.json"
    mkdir -p "${SAVE_DIR}"

    if [ -f "${METRICS}" ]; then
        echo "[${cond}] Metrics already computed, skipping inference."
        continue
    fi

    # Repoint meld-process at this condition
    if [ -L "${MELD_PROC}" ] || [ -e "${MELD_PROC}" ]; then
        rm -f "${MELD_PROC}"
    fi
    ln -sfn "$(basename ${DST})" "${MELD_PROC}"

    echo
    echo "----------------------------------------------"
    echo " [${cond}] Caching outputs"
    echo "         data    = ${DST}"
    echo "         jsonl   = ${JSONL}"
    echo "----------------------------------------------"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u cache_outputs.py \
        --cfg-path "${CFG_PATH}" \
        --datasets MELD \
        --save_dir "${SAVE_DIR}" \
        --max_samples "${MAX_SAMPLES}" \
        --resume \
        --options "inference.test_epoch=${TEST_EPOCH}"

    echo
    echo "----------------------------------------------"
    echo " [${cond}] Computing metrics"
    echo "----------------------------------------------"
    python compute_metrics.py "${JSONL}"
done

echo
echo "=============================================="
echo " ALL DONE"
echo "=============================================="
echo "Cache + metrics root: ${RESULTS_ROOT}"
ls -1 "${RESULTS_ROOT}/" 2>/dev/null | sed 's/^/  /'
