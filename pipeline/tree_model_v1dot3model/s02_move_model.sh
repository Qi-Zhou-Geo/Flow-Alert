#!/bin/bash
#SBATCH --output=/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/v1dot3model/ReadMe4Tree_model_details.txt
#SBATCH --error=/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/v1dot3model/slurm-%j.err

num_repeat=3
model_version="v1dot3model"
feature_type="H" # feature_type

path_in="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/pipeline/tree_model_${model_version}/train_test_2017-2020_${num_repeat}repeat_tree"
path_out="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/${model_version}"

# Create output directory if it doesn't exist
mkdir -p "${path_out}"

for repeat in $(seq 1 $num_repeat); do  # Changed 'sec' to 'seq'

  # XGB
  format_in="Model=XGB_STA=ILL02_Feature=${feature_type}_repeat=${repeat}.ubj"
  format_out="Model=XGB_STA=ILL02_Feature=${feature_type}_repeat=${repeat}.ubj"

  time_now=$(date)
  echo "${time_now}, Copy model"
  echo "from: ${path_in}/${format_in}"
  echo "to: ${path_out}/${format_out}"
  echo ""

  cp "${path_in}/${format_in}" "${path_out}/${format_out}"


  # RF
  format_in="Model=RF_STA=ILL02_Feature=${feature_type}_repeat=${repeat}.joblib"
  format_out="Model=RF_STA=ILL02_Feature=${feature_type}_repeat=${repeat}.joblib"

  time_now=$(date)
  echo "${time_now}, Copy model"
  echo "from: ${path_in}/${format_in}"
  echo "to: ${path_out}/${format_out}"
  echo ""

  cp "${path_in}/${format_in}" "${path_out}/${format_out}"

done

echo "Done! Processed ${num_repeat} files."