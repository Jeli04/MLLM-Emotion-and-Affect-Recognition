#!/bin/bash
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

module purge
module load gcc/13.3.0
module load cuda/12.4.0
module load ffmpeg

export PATH="$HOME/.local/bin:$PATH"

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
# PYTHONPATH=. uv run --active --no-sync python src/finetune.py \
#     --modalities text audio video \
#     --distill \
#     --no_modality_mask \
#     --clean_teacher \
#     --teacher_logits_dir ./cache/teacher_logits/finetuned_t+a+v \
#     --teacher_adapter_path ./ckpts/finetuned_base/lora_adapter \
#     --full_ce_weight 0.0 \
#     --mask_ce_weight 1.0 \
#     --lambda_kl 0.03 \
#     --output_dir ./ckpts/finetuned_distill_clean_teacher_adapter_lk0.03 \
#     --wandb_run_name distill_clean_teacher_adapter_lk0.03
PYTHONPATH=. uv run --active --no-sync python src/distillation_finetune.py \
    --modalities text audio video \
    --distill \
    --no_modality_mask \
    --clean_teacher \
    --base_teacher \
    --teacher_logits_dir ./cache/teacher_medium_logits/base_t+a+v  \
    --full_ce_weight 0.0 \
    --mask_ce_weight 1.0 \
    --lambda_kl 0.03 \
    --output_dir ./ckpts/finetuned_distill_clean_teacher_base_lk0.03_medium \
    --wandb_run_name distill_clean_teacher_base_lk0.03_medium