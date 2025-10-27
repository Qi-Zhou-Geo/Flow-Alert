#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS)
#SBATCH --job-name=v39model         # job name, "Qi_run"

#SBATCH --ntasks=1                  # each individual task in the job array will have single task associated with it
#SBATCH --array=1-3                # job array id 1-16

#SBATCH --mem-per-cpu=64G            # Memory Request (per CPU; can use on GLIC)
#SBATCH --begin=2023-10-20T09:00:00  # job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p "logs"
project_path="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert"

# load environment
source /home/qizhou/miniforge3/bin/activate
conda activate flow-alert

# Define model types array
models=("RF" "XGB" "LSTM")
model_type="${models[$SLURM_ARRAY_TASK_ID - 1]}"

echo "Model type: $model_type"
echo "Start time: $(date)"

# Run the Python script
python "${project_path}/demo/Jiangjia/jiangjia_2026.py" --model_type "$model_type"

echo "End time: $(date)"
echo "Job completed for model: $model_type"

