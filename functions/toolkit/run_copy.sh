#!/bin/bash
#SBATCH -t 00:10:00                # time limit: (D-HH:MM:SS)
#SBATCH --job-name=copy            # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-1               # job array id

#SBATCH --mem-per-cpu=2G            # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p "logs"

source /home/qizhou/miniforge3/bin/activate
conda activate flow-alert

# 2025-08-12
path_in="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/v2model/2.2/LSTM"
path_out="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/v2model"
ML_name="LSTM"
batch_size=128
seq_len=32
station="ILL02"
feature_type="E"
num_repeat=9

srun python copy_trained_model.py \
     --path_in "$path_in" \
     --path_out "$path_out" \
     --ML_name "$ML_name" \
     --batch_size "$batch_size" \
     --seq_len "$seq_len" \
     --station "$station" \
     --feature_type "$feature_type" \
     --num_repeat "$num_repeat"


# Check if both logs exist and are empty
out_file="logs/out_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"
err_file="logs/err_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"

if [ -f "$out_file" ] && [ -f "$err_file" ] && [ ! -s "$out_file" ] && [ ! -s "$err_file" ]; then
    echo "Logs empty. Removing logs folder."
    rm -rf logs
else
    echo "Logs not empty or missing, keeping logs folder."
fi