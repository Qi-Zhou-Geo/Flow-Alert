#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step3           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have single task associated with it
#SBATCH --array=1-1                # job array id

#SBATCH --mem-per-cpu=8G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs

source /home/qizhou/miniforge3/bin/activate
conda activate seismic

# Define arrays for parameters1, parameters2, and parameters3
catchment_name="Museum_Fire"
seismic_network="1A"
parameters1=(2021)
parameters2=("E19A")
parameters3="CHZ"
id1=175
id2=238


# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}

# Print the current combination
echo "Year: $current_parameters1, Station: $current_parameters2, Component: $current_parameters3"

# Run your Python script using srun with the parameters
srun python ../../s3_merge_single_julday.py \
    --catchment_name "$catchment_name" \
    --seismic_network "$seismic_network" \
    --input_year "$current_parameters1" \
    --input_station "$current_parameters2" \
    --input_component "$current_parameters3" \
    --id1 "$id1" \
    --id2 "$id2"

# delete the empty logout file
OUT_FILE="logs/out_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"
ERR_FILE="logs/err_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"

[ -e "$OUT_FILE" ] && [ ! -s "$OUT_FILE" ] && rm "$OUT_FILE"
[ -e "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ] && rm "$ERR_FILE"