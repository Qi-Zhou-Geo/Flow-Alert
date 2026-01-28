#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D$feature_typeH:MM:SS)
#SBATCH --job-name=GLOF            # job name, "Qi_run"

#SBATCH --ntasks=1                  # each individual task in the job array will have single task associated with it
#SBATCH --array=1-1                 # job array id

#SBATCH --mem-per-cpu=64G            # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A40:1             # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU            # reserve the GPU

#SBATCH --begin=2023-10-20T09:00:00  # job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p "logs"

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
# load environment
source /home/qizhou/miniforge3/bin/activate
conda activate flow-alert

project_root="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert"

start_time=$(date)
echo "Job started at: $start_time"

# "NEP08", "NEP07", "NEP06", "NEP10", "NEP05", "NEP04"

model_version="v1dot3model"
num_repeat=9
feature_type="H"
batch_size=128
seq_length=64
output_path="$project_root/pipeline/$model_version/glof_${num_repeat}repeat/output"


params="Bothekoshi-XN-2016-STA-HHZ-$feature_type-testing-False"
seismic_sta_list=("NEP08" "NEP07" "NEP06" "NEP10" "NEP05" "NEP04")

for sta in "${seismic_sta_list[@]}"; do
  # Replace STA with the current station
  current_params="${params//STA/$sta}"

  echo "Running with params: $current_params"
  srun --gres=gpu:A40:1 python "$project_root/pipeline/cross_catchments_test/by_single_catchment_main.py" \
       --params "$current_params" \
       --model_version "$model_version" \
       --num_repeat "$num_repeat" \
       --feature_type "$feature_type" \
       --batch_size "$batch_size" \
       --seq_length "$seq_length" \
       --output_path "$output_path"
done


# delete the empty logout file
OUT_FILE="logs/out_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"
ERR_FILE="logs/err_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"

[ -e "$OUT_FILE" ] && [ ! -s "$OUT_FILE" ] && rm "$OUT_FILE"
[ -e "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ] && rm "$ERR_FILE"


