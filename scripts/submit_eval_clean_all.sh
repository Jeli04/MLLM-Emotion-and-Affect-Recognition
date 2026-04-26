#!/bin/bash
# Submit medium-corruption eval jobs for every modality combo, for both base and finetuned models.
#
# Usage:
#   ./scripts/submit_eval_clean_all.sh                # MELD test split, medium corruption
#   ./scripts/submit_eval_clean_all.sh --split dev    # forward extra args to the eval script
#
# Any args you pass are forwarded through sbatch -> the eval script.
# The hardcoded `--modalities audio video` inside each .sbatch is overridden
# by the `--modalities ...` we append last (argparse keeps the last value).

set -euo pipefail

cd "$(dirname "$0")/.."

# Per-model modality lists. Comment out combos that already have a result file.
BASE_COMBOS=(
    "text"
    "audio"
    "video"
    "text audio"
    "text video"
    "audio video"
    "text audio video"
)

FINETUNED_COMBOS=(
    "text"
    "audio"
    "video"
    "text audio"
    "text video"
    "audio video"
    "text audio video"
)

BASE_SBATCH="scripts/run_eval_base_corruption.sbatch"
FINETUNED_SBATCH="scripts/run_eval_finetune_corruption.sbatch"

EXTRA_ARGS=("$@")

submit_combo() {
    local name="$1"
    local sbatch_script="$2"
    local combo="$3"
    # shellcheck disable=SC2086
    echo "Submitting: model=${name} corruption=medium modalities='${combo}'"
    sbatch \
        --job-name="eval_corrupt_medium_${name}_${combo// /+}" \
        "$sbatch_script" \
        "${EXTRA_ARGS[@]}" \
        --corruption_preset medium \
        --output_dir results/meld \
        --modalities $combo
}

for combo in "${BASE_COMBOS[@]}"; do
    submit_combo "base" "$BASE_SBATCH" "$combo"
done

# for combo in "${FINETUNED_COMBOS[@]}"; do
#     submit_combo "finetuned" "$FINETUNED_SBATCH" "$combo"
# done
