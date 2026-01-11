#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D$feature_typeH:MM:SS)
#SBATCH --job-name=cross_test      # job name, "Qi_run"

#SBATCH --ntasks=1                  # each individual task in the job array will have train_test_2017-2020_9repeat single task associated with it
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

start_time=$(date)
echo "Job started at: $start_time"


model_version="v1dot3model"
num_repeat=9
feature_type="H"
parameters1=(128) # batch_size
parameters2=(64) # seq_length

params_list=(
"Illgraben-9S-2022-ILL12-EHZ-$feature_type-testing-True"
)


for params in "${params_list[@]}"; do
  for batch_size in "${parameters1[@]}"; do
    for seq_length in "${parameters2[@]}"; do

      echo "Running with params: $params"
      srun --gres=gpu:A40:1 python by_single_catchment_main.py \
          --params "$params" \
          --model_version "$model_version" \
          --num_repeat "$num_repeat" \
          --feature_type "$feature_type" \
          --batch_size "$batch_size" \
          --seq_length "$seq_length"

    done
  done
done

# delete the empty logout file
OUT_FILE="logs/out_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"
ERR_FILE="logs/err_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.txt"

[ -e "$OUT_FILE" ] && [ ! -s "$OUT_FILE" ] && rm "$OUT_FILE"
[ -e "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ] && rm "$ERR_FILE"