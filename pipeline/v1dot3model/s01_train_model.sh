#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS)
#SBATCH --job-name=train_m         # job name, "Qi_run"

#SBATCH --ntasks=1                  # each individual task in the job array will have single task associated with it
#SBATCH --array=1-24                # job array id 1-16

#SBATCH --mem-per-cpu=64G            # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A40:1             # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU            # reserve the GPU

#SBATCH --begin=2023-10-20T09:00:00  # job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p "logs"
project_path="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert"


# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
# load environment
source /home/qizhou/miniforge3/bin/activate
conda activate flow-alert

start_time=$(date)
echo "Job started at: $start_time"

#parameters1=(64 128 256 512) # batch_size
#parameters2=(16 32 48 64 80 96 128) # seq_length

# variables
model_type="LSTM"
feature_type="H" # feature_type
parameters1=(64 128 256 512) # batch_size
parameters2=(16 32 48 64 96 128) # seq_length
parameters3=(2) # station type, "ILL12" "ILL13" "ILL18"
temp=(
"Illgraben-9S-2017-ILL0*-EHZ-H-training-True"
"Illgraben-9S-2018-ILL1*-EHZ-H-training-True"
"Illgraben-9S-2019-ILL1*-EHZ-H-training-True"
"Illgraben-9S-2020-ILL1*-EHZ-H-testing-True"
)
num_repeat=9
sub_folder="pipeline/v1dot3model"
output_dir="${project_path}/${sub_folder}"


# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}


# redifine the parameters, replave * in parameters4
params=()
for i in "${!temp[@]}"; do
  params[$i]="${temp[$i]//\*/$current_parameters3}"
done


# print the current variables
echo -e "$model_type, $feature_type, $current_parameters1, $current_parameters2, $current_parameters3, $output_dir \n"
echo "${params[@]}"

# Run your Python script using srun with the parameters
srun --gres=gpu:A40:1 python v1dot3main.py \
     --model_type "$model_type" \
     --feature_type "$feature_type" \
     --batch_size "$current_parameters1" \
     --seq_length "$current_parameters2" \
     --params "${params[@]}" \
     --num_repeat "$num_repeat" \
     --output_dir "$output_dir"

end_time=$(date)
echo "Job finished at: $end_time"
echo "Total running time: $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) seconds"



# delete the empty logout file
OUT_FILE="logs/out_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"
ERR_FILE="logs/err_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"

[ -e "$OUT_FILE" ] && [ ! -s "$OUT_FILE" ] && rm "$OUT_FILE"
[ -e "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ] && rm "$ERR_FILE"