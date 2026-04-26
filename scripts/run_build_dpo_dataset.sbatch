#!/bin/bash
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
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

source ./.venv-qwen/bin/activate
module load ffmpeg
uv run python -m src.dpo.build_dpo_dataset \
    --split train \
    --data_root /project2/robinjia_875/lijc/data/MELD.Raw \
    --model_path ./ckpts/Qwen2.5-Omni-7B-GPTQ-Int4 \
    --adapter_path ./ckpts/finetuned_base/lora_adapter \
    --correct_sample_ratio 0.2 \
    --corrupt \
    --modalities text audio video \
    "$@"
