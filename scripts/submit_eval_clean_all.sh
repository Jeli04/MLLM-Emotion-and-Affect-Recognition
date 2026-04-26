#!/bin/bash
# Submit clean-eval jobs for every modality combo, for both base and finetuned models.
#
# Usage:
#   ./scripts/submit_eval_clean_all.sh                # MELD test split
#   ./scripts/submit_eval_clean_all.sh --split dev    # forward extra args to the eval script
#
# Any args you pass are forwarded through sbatch -> the eval script.
# The hardcoded `--modalities audio video` inside each .sbatch is overridden
# by the `--modalities ...` we append last (argparse keeps the last value).

set -euo pipefail

cd "$(dirname "$0")/.."

# Per-model modality lists. Comment out combos that already have a result file.
BASE_COMBOS=(
    # "text"               # done: results_test_text_base.json
    # "audio"              # done: results_test_audio_base.json
    # "video"              # done: results_test_video_base.json
    # "text audio"         # done: results_test_audio+text_base.json
    # "text video"         # done: results_test_text+video_base.json
    # "audio video"        # done: results_test_audio+video_base.json
    # "text audio video"   # done: results_test_audio+text+video_base.json
)

FINETUNED_COMBOS=(
    # "text"               # done: results_test_text_finetuned.json
    # "audio"              # done: results_test_audio_finetuned.json
    # "text audio"         # done: results_test_audio+text_finetuned.json
    "video"
    "text video"
    "audio video"
    "text audio video"
)

BASE_SBATCH="scripts/run_eval_base_clean.sbatch"
FINETUNED_SBATCH="scripts/run_eval_finetune_clean.sbatch"

EXTRA_ARGS=("$@")

submit_combo() {
    local name="$1"
    local sbatch_script="$2"
    local combo="$3"
    # shellcheck disable=SC2086
    echo "Submitting: model=${name} modalities='${combo}'"
    sbatch \
        --job-name="eval_clean_${name}_${combo// /+}" \
        "$sbatch_script" \
        "${EXTRA_ARGS[@]}" \
        --modalities $combo
}

for combo in "${BASE_COMBOS[@]}"; do
    submit_combo "base" "$BASE_SBATCH" "$combo"
done

for combo in "${FINETUNED_COMBOS[@]}"; do
    submit_combo "finetuned" "$FINETUNED_SBATCH" "$combo"
done
