#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS)
#SBATCH --job-name=YM_test         # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-1                # job array id

#SBATCH --mem-per-cpu=32G          # Memory Request (per CPU; can use on GLIC)

#SBATCH --begin=2023-10-20T09:00:00  # job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p "logs"


# load environment
source /home/qizhou/miniforge3/bin/activate
conda activate flow-alert

# Run the script with the selected parameter
srun python Yanmen_main.py
