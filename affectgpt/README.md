# Affect GPT Evaluation

Contains slurm files only for running on CARC. Note the original paper uses OpenFaces, but we use mediapipe to extract faces as inputs to the model. We also just use pure accuracy rather than hitrate which was used in the original paper. 

## Setup
Clone AffectGPT from this link [link](https://github.com/zeroQiaoba/AffectGPT). Places the files in this direcotry 
at the root of the AffectGPT directory at `AffectGPT/AffectGPT`. Put the zipped MELD dataset in the same directory. Then, put the files on CARC. 

Once the files are on CARC, activate your conda environment and install the requirements.txt. 

## Running (Missing Modality Masking)
Replace all paths in the .sh files and the .slurm files with appropriate output directories. You can do it with these sed commands: 

```bash
sed -i 's|/home1/palmerla/.conda/envs/affgpt/bin|/home1/USER/.conda/envs/affgpt/bin|g' carc_download.slurm carc_eval.slurm

sed -i 's|/home1/palmerla/eval_affectgpt/AffectGPT/AffectGPT|/path/to/your/AffectGPT|g' carc_download.slurm carc_eval.slurm 

sed -i 's|msoleyma_1026|your_account_id|g' carc_eval.slurm carc_download.slurm
```

Then, just run `sbatch carc_download.slurm` to extract all the inputs to the model and `sbatch carc_eval.slurm` to run the evaluations over the 7 masking conditions. 

## Running (Modality Corruption)
This runs the model over 8 corruption conditions (none, T, A, V, TA, TV, AV, TAV) at the strong preset. Corruption is applied in-memory at runtime, so no extra disk prep is needed beyond `setup_downloads.sh`.

Files needed: `run_corruption_eval.sh`, `cache_outputs.py`, `corruption_lib.py`, `compute_metrics.py`, `aggregate_metrics.py`, `setup_downloads.sh`, `extract_faces.py`, `requirements.txt`.

First time, run setup once:
```bash
bash setup_downloads.sh
```

Then run the corruption eval:
```bash
./run_corruption_eval.sh
```

Override with env vars:
- `CONDITIONS="text audio_video"` runs a subset.
- `MAX_SAMPLES=100` for a smoke test.
- `PRESET=mild` or `medium` for weaker corruption (default is `strong`).
- `CUDA_VISIBLE_DEVICES=0` to pin a GPU.
- `RESULTS_ROOT=/path/to/out` to override the output dir.

Per-condition metrics land in `output/corruption_eval_strong/<cond>/meld.metrics.json`. For fair cross-condition numbers over the intersection of samples that succeeded in every condition, run:

```bash
python aggregate_metrics.py output/corruption_eval_strong
```

That writes `aggregate.json` and `aggregate.csv` next to the per-condition dirs.
