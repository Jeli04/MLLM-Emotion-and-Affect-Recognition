# Affect GPT Evaluation

Contains slurm files only for running on CARC. Note the original paper uses OpenFaces, but we use mediapipe to extract faces as inputs to the model. We also just use pure accuracy rather than hitrate which was used in the original paper. 

## Setup
Clone AffectGPT from this link [link](https://github.com/zeroQiaoba/AffectGPT). Places the files in this direcotry 
at the root of the AffectGPT directory at `AffectGPT/AffectGPT`. Put the zipped MELD dataset in the same directory. Then, put the files on CARC. 

Once the files are on CARC, activate your conda environment and install the requirements.txt. 

## Running
Replace all paths in the .sh files and the .slurm files with appropriate output directories. You can do it with these sed commands: 

```bash
sed -i 's|/home1/palmerla/.conda/envs/affgpt/bin|/home1/USER/.conda/envs/affgpt/bin|g' carc_download.slurm carc_eval.slurm

sed -i 's|/home1/palmerla/eval_affectgpt/AffectGPT/AffectGPT|/path/to/your/AffectGPT|g' carc_download.slurm carc_eval.slurm 

sed -i 's|msoleyma_1026|your_account_id|g' carc_eval.slurm carc_download.slurm
```

Then, just run `sbatch carc_download.slurm` to extract all the inputs to the model and `sbatch carc_eval.slurm` to run the evaluations over the 7 masking conditions. 
