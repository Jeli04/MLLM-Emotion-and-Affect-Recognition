#!/bin/bash
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -euo pipefail

mkdir -p slurm_logs

module purge
module load gcc/13.3.0
module load cuda/12.4.0

cd $SLURM_SUBMIT_DIR

module load ffmpeg
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7}"

# Trains DPO on the MELD dataset produced by
# scripts/run_build_dpo_finetune_weighted_v3.sbatch.
# Starts from the same SFT LoRA adapter used to generate the pairs
# (finetuned_corrupt_strong_weighted), so DPO refines on top of it.
# PYTHONPATH=. uv run --active --no-sync python -m src.dpo.dpo_training \
#     --dpo_data_path /project2/robinjia_875/lijc/MLLM-Emotion-and-Affect-Recognition/results/dpo/dpo_samples_train_audio+text+video_corrupt_strong_base_rebalanced_v4.json \
#     --output_dir ./ckpts/dpo/dpo_base_rebalanced_v4 \
#     --modalities text audio video \
#     --learning_rate 1e-6 \
#     --num_epochs 1 \
#     --eval_ratio 0.1 \
#     --beta 0.5 \
#     --wandb_run_name dpo_base_rebalanced_v4 \
#     "$@"
# PYTHONPATH=. uv run --active --no-sync python -m src.dpo.dpo_training \
#     --dpo_data_path /project2/robinjia_875/lijc/MLLM-Emotion-and-Affect-Recognition/results/dpo/dpo_samples_train_audio+text+video_corrupt_strong_base_rebalanced_v4.json \
#     --output_dir ./ckpts/dpo/dpo_finetune_weighted_rebalanced_v4 \
#     --adapter_path ./ckpts/finetuned_corrupt_strong_weighted/lora_adapter \
#     --modalities text audio video \
#     --learning_rate 1e-6 \
#     --num_epochs 1 \
#     --eval_ratio 0.1 \
#     --beta 0.5 \
#     --wandb_run_name dpo_finetune_weighted_rebalanced_v4 \
#     "$@"
PYTHONPATH=. uv run --active --no-sync python -m src.dpo.dpo_training \
    --dpo_data_path /project2/robinjia_875/lijc/MLLM-Emotion-and-Affect-Recognition/results/dpo/dpo_samples_train_audio+text+video_corrupt_strong_student_teacher_weighted_rebalanced_v4.json \
    --output_dir ./ckpts/dpo/dpo_student_teacher_weighted_rebalanced_v4 \
    --adapter_path ./ckpts/finetuned_distill_clean_teacher_base_lk0.03_strong_weighted2 \
    --modalities text audio video \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --eval_ratio 0.1 \
    --beta 0.5 \
    --wandb_run_name dpo_student_teacher_weighted_rebalanced_v4 \
    "$@"
status=$?
if [ "$status" -eq 135 ]; then
    echo "ERROR: Python terminated with SIGBUS (bus error). This is a native-code/OS signal, so Python try/except cannot recover from it. Check the faulthandler traceback above for the active Python stack." >&2
fi
exit "$status"
