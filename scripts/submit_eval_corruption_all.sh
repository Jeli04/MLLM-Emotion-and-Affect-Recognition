#!/bin/bash
# Submit strong-corruption eval jobs for every modality combo, for both base and finetuned models.
#
# Usage:
#   ./scripts/submit_eval_clean_all.sh                # MELD test split, strong corruption
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

DPO_COMBOS=(
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

run_label_from_extra_args() {
    local label="$1"
    local i=0
    while (( i < ${#EXTRA_ARGS[@]} )); do
        case "${EXTRA_ARGS[$i]}" in
            --run_label)
                if (( i + 1 < ${#EXTRA_ARGS[@]} )); then
                    label="${EXTRA_ARGS[$((i + 1))]}"
                fi
                ((i += 2))
                ;;
            --run_label=*)
                label="${EXTRA_ARGS[$i]#--run_label=}"
                ((i += 1))
                ;;
            *)
                ((i += 1))
                ;;
        esac
    done
    printf '%s\n' "$label"
}

submit_combo() {
    local name="$1"
    local sbatch_script="$2"
    local combo="$3"
    shift 3
    local combo_extra_args=("$@")
    # shellcheck disable=SC2086
    echo "Submitting: model=${name} corruption=strong modalities='${combo}'"
    sbatch \
        --job-name="eval_corrupt_strong_${name}_${combo// /+}" \
        "$sbatch_script" \
        "${EXTRA_ARGS[@]}" \
        "${combo_extra_args[@]}" \
        --corruption_preset strong \
        --output_dir results/meld \
        --modalities $combo 
}

# for combo in "${BASE_COMBOS[@]}"; do
#     submit_combo "base" "$BASE_SBATCH" "$combo"
# done

for combo in "${FINETUNED_COMBOS[@]}"; do
    submit_combo "finetuned" "$FINETUNED_SBATCH" "$combo"
done

# dpo_run_label="dpo_$(run_label_from_extra_args "finetune")"
# for combo in "${DPO_COMBOS[@]}"; do
#     submit_combo "dpo" "$FINETUNED_SBATCH" "$combo" --run_label "$dpo_run_label"
# done
