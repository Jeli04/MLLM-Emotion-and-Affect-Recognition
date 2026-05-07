#!/bin/bash
# Rerun DPO rebalance: keep all real minority pairs, keep 20% of neutral/joy
# pairs as anchors, no synthesized anti-majority pairs. Result is the natural
# class distribution with neutral/joy capped so they don't dominate.
set -euo pipefail

cd "$(dirname "$0")/.."

SRC=results/dpo/dpo_samples_train_audio+text+video_corrupt_strong_base.json
OUT=results/dpo/dpo_samples_train_audio+text+video_corrupt_strong_base_rebalanced_v2.json

PYTHONPATH=. uv run --active --no-sync python -m src.dpo.rebalance_dpo_dataset \
    --input "$SRC" \
    --output "$OUT" \
    --minority anger disgust fear sadness surprise \
    --anti_majority_targets \
    --anchor_majority_fraction 0.2 \
    --anchor_seed 42

echo
echo "Wrote $OUT"
