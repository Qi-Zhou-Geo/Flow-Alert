#!/bin/bash
#SBATCH -t 00:20:00                # time limit: (D-HH:MM:SS)
#SBATCH --job-name=create_label    # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-1                # job array id

#SBATCH --mem-per-cpu=16G		       # Memory Request (per CPU; can use on GLIC)


#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs


source /home/qizhou/miniforge3/bin/activate
conda activate flow-alert


# Run your Python script using srun with the parameters
# ILL02
srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2017 \
     --input_station "ILL02" \
     --input_component "EHZ" \
     --usecols 5 6

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2018 \
     --input_station "ILL12" \
     --input_component "EHZ" \
     --usecols 5 6

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2019 \
     --input_station "ILL12" \
     --input_component "EHZ" \
     --usecols 5 6

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2020 \
     --input_station "ILL12" \
     --input_component "EHZ" \
     --usecols 5 6

# ILL03
srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2017 \
     --input_station "ILL03" \
     --input_component "EHZ" \
     --usecols 7 8

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2018 \
     --input_station "ILL13" \
     --input_component "EHZ" \
     --usecols 7 8

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2019 \
     --input_station "ILL13" \
     --input_component "EHZ" \
     --usecols 7 8

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2020 \
     --input_station "ILL13" \
     --input_component "EHZ" \
     --usecols 7 8

# ILL08
srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2017 \
     --input_station "ILL08" \
     --input_component "EHZ" \
     --usecols 2 3

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2018 \
     --input_station "ILL18" \
     --input_component "EHZ" \
     --usecols 2 3

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2019 \
     --input_station "ILL18" \
     --input_component "EHZ" \
     --usecols 2 3

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2020 \
     --input_station "ILL18" \
     --input_component "EHZ" \
     --usecols 2 3
