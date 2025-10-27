#!/bin/bash

project_path="/Users/qizhou/#python/Flow-Alert"
model_version="v1dot3model"

# Load conda into this shell
# find the local conda -> conda info | grep 'base environment'
source /usr/local/Caskroom/mambaforge/base/etc/profile.d/conda.sh
source /usr/local/Caskroom/mambaforge/base/etc/profile.d/mamba.sh
mamba activate flow-alert

model_type="XGB"
python "${project_path}/pipeline/tree_model_${model_version}/tree_main.py" --model_type "$model_type"
