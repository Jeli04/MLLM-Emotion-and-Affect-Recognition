#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=affectgpt_extract
#SBATCH --output=slurm-extract-%j.out

export PATH="/home1/palmerla/.conda/envs/affgpt/bin:$PATH"
export MESA_GL_VERSION_OVERRIDE=3.3
export MEDIAPIPE_DISABLE_GPU=1

module load ver/2506 gcc/14.3.0
module load ffmpeg/7.1.1

echo "Python: $(which python)"

cd /home1/palmerla/eval_affectgpt/AffectGPT/AffectGPT

source setup_downloads.sh
