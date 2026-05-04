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

mkdir -p slurm_logs

module purge
module load gcc/13.3.0
module load cuda/12.4.0

cd $SLURM_SUBMIT_DIR

source ./.venv/bin/activate
module load ffmpeg
PYTHONPATH=. python -m src.evaluate.evaluate_clean \
    --modalities audio video \
    "$@"
