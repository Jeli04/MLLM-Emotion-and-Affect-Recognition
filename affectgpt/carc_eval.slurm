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

export PATH="/home1/palmerla/.conda/envs/affgpt/bin:$PATH"

cd /home1/palmerla/eval_affectgpt/AffectGPT/AffectGPT

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

CFG_PATH="train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml"

CUDA_VISIBLE_DEVICES=0 python -u missing_modality_eval.py \
	--cfg-path "${CFG_PATH}" \
	--datasets MELD \
	--masking_conditions none mask_audio mask_video mask_text mask_audio_video mask_audio_text mask_video_text \
	--save_dir /scratch1/palmerla/output/missing_modality_results \
	--max_samples 0 \
	--options "inference.test_epoch=60"

