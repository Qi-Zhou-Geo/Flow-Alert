#!/bin/bash
#SBATCH --output=/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/v1dot3model/ReadMe4LSTM_model_details.txt
#SBATCH --error=/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/v1dot3model/slurm-%j.err

num_repeat=9
model_version="v1dot3model"
feature_type="H" # feature_type
batch_size=128 # batch_size
seq_length=64 # seq_length

path_in="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/pipeline/${model_version}/train_test_2017-2020_${num_repeat}repeat/LSTM"
path_out="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/trained_model/${model_version}"

# Create output directory if it doesn't exist
mkdir -p "${path_out}"

for repeat in $(seq 1 $num_repeat); do  # Changed 'sec' to 'seq'

  format_in="Illgraben-9S-2017-ILL02-EHZ-${feature_type}-training-True-repeat-${repeat}-LSTM-${feature_type}-b${batch_size}-s${seq_length}.pt"
  format_out="Model=LSTM_STA=ILL02_Feature=${feature_type}_repeat=${repeat}.pt"

  time_now=$(date)
  echo "${time_now}, Copy model"
  echo "from: ${path_in}/${format_in}"
  echo "to: ${path_out}/${format_out}"
  echo ""

  cp "${path_in}/${format_in}" "${path_out}/${format_out}"

done

echo "Done! Processed ${num_repeat} files."