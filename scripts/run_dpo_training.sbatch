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

mkdir -p slurm_logs

module purge
module load gcc/13.3.0
module load cuda/12.4.0

cd $SLURM_SUBMIT_DIR

module load ffmpeg
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7}"
PYTHONPATH=. uv run --active --no-sync python -m src.dpo.dpo_training \
    --dpo_data_path results/dpo/dpo_samples_train_audio+text+video_corrupt_strong_student_teacher.json \
    --output_dir ./ckpts/dpo/dpo_samples_train_audio+text+video_corrupt_strong_student_teacher_lk0.03 \
    --adapter_path ./ckpts/finetuned_distill_clean_teacher_base_lk0.03_medium/lora_adapter \
    --modalities text audio video \
    "$@"
status=$?
if [ "$status" -eq 135 ]; then
    echo "ERROR: Python terminated with SIGBUS (bus error). This is a native-code/OS signal, so Python try/except cannot recover from it. Check the faulthandler traceback above for the active Python stack." >&2
fi
exit "$status"
