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

mkdir -p slurm_logs

module purge
module load gcc/13.3.0
module load cuda/12.4.0

cd $SLURM_SUBMIT_DIR

source ./.venv/bin/activate
module load ffmpeg
PYTHONPATH=. python src/finetune.py \
    --modalities text audio video \
    --corruption_preset medium \
    --output_dir ./ckpts/finetuned_corrupt_medium \
    --resume_from_checkpoint ./ckpts/finetuned_corrupt_medium/checkpoint-600
    --num_epochs 1 \
    "$@"
status=$?
if [ "$status" -eq 135 ]; then
    echo "ERROR: Python terminated with SIGBUS (bus error). This is a native-code/OS signal, so Python try/except cannot recover from it. Check the faulthandler traceback above for the active Python stack." >&2
fi
exit "$status"
# PYTHONPATH=. python src/finetune.py \
#     --modalities text audio video \
#     "$@"
