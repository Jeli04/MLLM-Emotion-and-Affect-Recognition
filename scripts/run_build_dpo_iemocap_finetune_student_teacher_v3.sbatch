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
mkdir -p results/dpo

module purge
module load gcc/13.3.0
module load cuda/12.4.0

cd $SLURM_SUBMIT_DIR

source ./.venv/bin/activate
module load ffmpeg

# IEMOCAP DPO data on top of iemocap_finetuned_distill_clean_teacher_base_lk0.03_strong_weighted,
# using the new expanded confusion graph + per-class cap (rebalanced_v3 convention).
# Writes results/dpo/dpo_samples_iemocap_train_audio+text+video_corrupt_strong_finetune_student_teacher_weighted_rebalanced_v3.json
# PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
#     --dataset iemocap \
#     --split train \
#     --manifest /project2/robinjia_875/lijc/data/IEMOCAP_full_release/manifests/iemocap_utterance_labels.csv \
#     --model_path ./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4 \
#     --adapter_path ./ckpts/iemocap_finetuned_distill_clean_teacher_base_lk0.03_strong_weighted/lora_adapter \
#     --corrupt \
#     --corruption_preset strong \
#     --modalities text audio video \
#     --correct_sample_ratio 0.0 \
#     --max_chosen_per_class 150 \
#     --output_name_suffix finetune_student_teacher_weighted_rebalanced_v3 \
#     "$@"
PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
    --dataset iemocap \
    --split train \
    --manifest /project2/robinjia_875/lijc/data/IEMOCAP_full_release/manifests/iemocap_utterance_labels.csv \
    --model_path ./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4 \
    --adapter_path /project2/robinjia_875/lijc/MLLM-Emotion-and-Affect-Recognition/ckpts/iemocap_sft_balanced/ckpts_SFT_weightedsample/lora_adapter \
    --corrupt \
    --corruption_preset strong \
    --modalities text audio video \
    --correct_sample_ratio 0.0 \
    --max_chosen_per_class 150 \
    --output_name_suffix finetune_weighted_rebalanced_v3 \
    "$@"



