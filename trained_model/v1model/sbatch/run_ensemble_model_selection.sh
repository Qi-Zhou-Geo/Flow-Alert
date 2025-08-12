#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=ensemble        # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-6               # job array id

#SBATCH --mem-per-cpu=128G		       # Memory Request (per CPU; can use on GLIC)
#SBATCH --begin=2023-10-20T09:00:00# job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs

parameters1=("ILL18" "ILL12" "ILL13") # input station
parameters2=("XGBoost" "Random_Forest") # input model
parameters3=("C") # input features

# Create parameters4 with integers from 1 to 80 excluding [34, 26, 52]
exclude_list=()  # You can leave this empty or define the values to exclude
parameters4=()

for i in {0..79}; do
    if [[ ${#exclude_list[@]} -eq 0 || ! " ${exclude_list[@]} " =~ " ${i} " ]]; then
        parameters4+=($i)
    fi
done

# Print the list of indexes
echo "Indexes: ${parameters4[@]}"

# Convert parameters4 to a comma-separated string
parameters4_string=$(IFS=,; echo "${parameters4[*]}")

# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}


# Run your Python script using srun with the parameters
srun python ../../../functions/tree_ensemble_main_1by1.py \
     --input_station "$current_parameters1" \
     --model_type "$current_parameters2" \
     --feature_type "$current_parameters3" \
     --input_component "EHZ" \
     --indexes "$parameters4_string"
