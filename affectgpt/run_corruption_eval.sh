#!/bin/bash
set -eo pipefail

PRESET="${PRESET:-strong}"
SEED="${SEED:-42}"
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

MELD_RAW="${MELD_RAW:-${ROOT_DIR}/MELD.Raw}"
MELD_RAW_LINK="${ROOT_DIR}/MELD.Raw"

RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/output/corruption_eval_${PRESET}}"

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

mkdir -p "${RESULTS_ROOT}"

if [ ! -e "${MELD_RAW_LINK}" ]; then
    if [ -d "${MELD_RAW}" ]; then
        ln -s "${MELD_RAW}" "${MELD_RAW_LINK}"
    fi
fi

if [ ! -d "${MELD_PROC}" ] && [ ! -d "${MELD_PROC_CLEAN}" ]; then
    bash setup_downloads.sh
fi

if [ ! -e "${MELD_PROC_CLEAN}" ]; then
    if [ -L "${MELD_PROC}" ]; then
        :
    elif [ -d "${MELD_PROC}" ]; then
        mv "${MELD_PROC}" "${MELD_PROC_CLEAN}"
    else
        echo "ERROR: ${MELD_PROC_CLEAN} not found. Run setup_downloads.sh first." >&2
        exit 1
    fi
fi

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

    SAVE_DIR="${RESULTS_ROOT}/${cond}"
    JSONL="${SAVE_DIR}/meld.jsonl"
    METRICS="${SAVE_DIR}/meld.metrics.json"
    mkdir -p "${SAVE_DIR}"

    if [ -f "${METRICS}" ]; then
        continue
    fi

    if [ -L "${MELD_PROC}" ] || [ -e "${MELD_PROC}" ]; then
        rm -f "${MELD_PROC}"
    fi
    ln -sfn "$(basename ${MELD_PROC_CLEAN})" "${MELD_PROC}"

    CUDA_VISIBLE_DEVICES="${GPU}" python -u cache_outputs.py \
        --cfg-path "${CFG_PATH}" \
        --datasets MELD \
        --save_dir "${SAVE_DIR}" \
        --max_samples "${MAX_SAMPLES}" \
        --resume \
        --corruption_preset "${PRESET}" \
        ${mods:+--corrupt_modalities ${mods}} \
        --options "inference.test_epoch=${TEST_EPOCH}"

    python compute_metrics.py "${JSONL}"
done
