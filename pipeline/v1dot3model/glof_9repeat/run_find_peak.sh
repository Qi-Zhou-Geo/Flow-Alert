#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D$feature_typeH:MM:SS)
#SBATCH --job-name=GLOF            # job name, "Qi_run"

#SBATCH --ntasks=1                  # each individual task in the job array will have single task associated with it
#SBATCH --array=1-1                 # job array id

#SBATCH --mem-per-cpu=64G            # Memory Request (per CPU; can use on GLIC)

#SBATCH --begin=2023-10-20T09:00:00  # job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p "logs"

source /home/qizhou/miniforge3/bin/activate
conda activate flow-alert

srun python find_the_peak_f.py