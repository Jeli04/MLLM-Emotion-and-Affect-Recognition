#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=affgpt_corr_eval
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

set -euo pipefail

mkdir -p slurm_logs

module purge
module load gcc/13.3.0
module load cuda/12.4.0
module load ffmpeg

cd "$SLURM_SUBMIT_DIR"

source ./.venv/bin/activate

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-Omni-7B-GPTQ-Int4}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATA_ROOT="${DATA_ROOT:-/project2/robinjia_875/lijc/data/MELD.Raw}"
SPLIT="${SPLIT:-test}"
CORRUPTION_PRESET="${CORRUPTION_PRESET:-medium}"
RUN_LABEL="${RUN_LABEL:-add_affgpt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/meld}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

CMD=(
  python -u src/evaluate/evaluate_corruption.py
  --model_path "$MODEL_PATH"
  --data_root "$DATA_ROOT"
  --split "$SPLIT"
  --corruption_preset "$CORRUPTION_PRESET"
  --output_dir "$OUTPUT_DIR"
  --run_label "$RUN_LABEL"
  --run_all_modalities
)

if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter_path "$ADAPTER_PATH")
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
