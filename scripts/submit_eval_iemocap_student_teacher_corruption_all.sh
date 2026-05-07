#!/bin/bash
# Submit strong-corruption IEMOCAP student-teacher eval jobs for every modality combo.
#
# Usage:
#   ./scripts/submit_eval_iemocap_student_teacher_corruption_all.sh
#   ./scripts/submit_eval_iemocap_student_teacher_corruption_all.sh --run_label my_label
#
# Extra args are forwarded through sbatch to src.evaluate.evaluate_corruption.
# Defaults below can be overridden by passing the same flag later on the command line.

set -euo pipefail

cd "$(dirname "$0")/.."

COMBOS=(
    "text"
    "audio"
    "video"
    "text audio"
    "text video"
    "audio video"
    "text audio video"
)

SBATCH_SCRIPT="scripts/run_eval_student_teacher_corruption.sbatch"
MANIFEST="/project2/robinjia_875/lijc/data/IEMOCAP_full_release/manifests/iemocap_utterance_labels.csv"
ADAPTER_PATH="/project2/robinjia_875/lijc/MLLM-Emotion-and-Affect-Recognition/ckpts/iemocap_finetuned_distill_clean_teacher_base_lk0.03_strong_weighted/lora_adapter"
RUN_LABEL="iemocap_student_teacher_strong_weighted"
OUTPUT_DIR="results/iemocap/strong_corruption"

EXTRA_ARGS=("$@")

for combo in "${COMBOS[@]}"; do
    # shellcheck disable=SC2086
    echo "Submitting: IEMOCAP student_teacher corruption=strong modalities='${combo}'"
    sbatch \
        --job-name="eval_iemocap_st_corrupt_strong_${combo// /+}" \
        "$SBATCH_SCRIPT" \
        --dataset iemocap \
        --manifest "$MANIFEST" \
        --adapter_path "$ADAPTER_PATH" \
        --run_label "$RUN_LABEL" \
        --corruption_preset strong \
        --output_dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}" \
        --modalities $combo
done
