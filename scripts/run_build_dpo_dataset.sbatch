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
# PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
#     --split train --data_root /project2/robinjia_875/lijc/data/MELD.Raw \
#     --adapter_path ./ckpts/finetuned_corrupt_strong_weighted/lora_adapter \
#     --corruption_preset strong \
#     --corrupt \
#     --modalities text audio video \
#     --output_name_suffix finetune_weighted_rebalanced_v3 \
#     --max_chosen_per_class 150
PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
    --split train --data_root /project2/robinjia_875/lijc/data/MELD.Raw \
    --corruption_preset strong \
    --corrupt \
    --modalities text audio video \
    --output_name_suffix base_rebalanced_v4 \
    --correct_sample_ratio 0.4 \
    --max_chosen_per_class 150
# PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
#     --split train \
#     --data_root /project2/robinjia_875/lijc/data/MELD.Raw \
#     --model_path ./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4 \
#     --adapter_path ./ckpts/finetuned_distill_clean_teacher_base_lk0.03_strong_weighted2/lora_adapter \
#     --correct_sample_ratio 0.0 \
#     --corruption_preset strong \
#     --output_name_suffix student_teacher_weighted \
#     --corrupt \
#     --modalities text audio video \
#     "$@"
# PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
#     --split train \
#     --data_root /project2/robinjia_875/lijc/data/MELD.Raw \
#     --model_path ./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4 \
#     --adapter_path ./ckpts/finetuned_medium/lora_adapter \
#     --correct_sample_ratio 0.2 \
#     --corruption_preset strong \
#     --output_name_suffix student_teacher \
#     --corrupt \
#     --modalities text audio video \
#     "$@"
# PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
#     --split train \
#     --data_root /project2/robinjia_875/lijc/data/MELD.Raw \
#     --model_path ./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4 \
#     --adapter_path ./ckpts/finetuned_base/lora_adapter \
#     --correct_sample_ratio 0.2 \
#     --corruption_preset medium \
#     --corrupt \
#     --modalities text audio video \
#     "$@"
# PYTHONPATH=. python -m src.dpo.build_dpo_dataset \
#     --split train \
#     --data_root /project2/robinjia_875/lijc/data/MELD.Raw \
#     --model_path ./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4 \
#     --adapter_path 
#     --correct_sample_ratio 0.2 \
#     --corruption_preset medium \
#     --corrupt \
#     --modalities text audio video \
#     "$@"
