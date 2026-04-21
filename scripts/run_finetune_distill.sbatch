#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-task=a100:1
#SBATCH --time=12:00:00
#SBATCH --job-name=affectgpt_eval
#SBATCH --output=slurm-eval-%j.out

module purge
module load gcc/13.3.0
module load cuda/12.4.0
module load ffmpeg

export PATH="$HOME/.local/bin:$PATH"

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
PYTHONPATH=. uv run --active --no-sync python src/finetune.py \
    --data_root /home1/palmerla/eval_affectgpt/AffectGPT/AffectGPT/MELD.Raw \
    --modalities text audio video \
    --distill \
    --lambda_kl 1.0 \
    --output_dir ./ckpts/finetuned_distill \
    --wandb_run_name distill_lk1.0
