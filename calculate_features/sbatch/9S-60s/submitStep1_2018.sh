#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step1           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have single task associated with it
#SBATCH --array=1-106              # job array id

#SBATCH --mem-per-cpu=8G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/step1_out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/step1_err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

source /home/qizhou/miniforge3/bin/activate
conda activate seismic


# Define arrays for parameters1, parameters2, and parameters3
parameters1=(2018)
parameters2=("ILL12")
parameters3=($(seq 145 250)) # 106 = 250 - 145 + 1
catchment_name="Illgraben"
seismic_network="9S"
input_component="EHZ"
input_window_size=60


# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}

# Print the current combination
echo "Year: $current_parameters1, Station: $current_parameters2, Julday $current_parameters3"

srun python ../../s1_cal_TypeA_TypeB.py \
    --catchment_name "$catchment_name" \
    --seismic_network "$seismic_network" \
    --input_year "$current_parameters1" \
    --input_station "$current_parameters2" \
    --input_component "$input_component" \
    --input_window_size "$input_window_size" \
    --id "$current_parameters3"
