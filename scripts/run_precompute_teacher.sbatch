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

# Precompute base-teacher logits for all three modalities.
# Override --teacher_adapter_path / --base_teacher / --output_dir / etc. via "$@".
PYTHONPATH=. uv run --active --no-sync python scripts/precompute_teacher_logits.py \
    --modalities text audio video \
    --splits train dev \
    --teacher_adapter_path  /project2/robinjia_875/lijc/MLLM-Emotion-and-Affect-Recognition/ckpts/finetuned_base/lora_adapter/ \
    --output_dir ./cache/teacher_logits/finetuned_t+a+v \
    "$@"
