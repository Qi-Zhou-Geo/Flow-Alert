#!/bin/bash

#SBATCH --job-name=combined_steps    # Job name
#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

# Submit jobs for 2017 (sequential)
job1_2017=$(sbatch submitStep1_2017.sh)
job2_2017=$(sbatch --dependency=afterok:${job1_2017##* } submitStep2_2017.sh)
sbatch --dependency=afterok:${job2_2017##* } submitStep3_2017.sh

# Submit jobs for 2018/2019 (sequential)
job4_2018_2019=$(sbatch submitStep1_2018_2019.sh)
job5_2018_2019=$(sbatch --dependency=afterok:${job4_2018_2019##* } submitStep2_2018_2019.sh)
sbatch --dependency=afterok:${job5_2018_2019##* } submitStep3_2018_2019.sh

# Submit jobs for 2020 (sequential)
job7_2020=$(sbatch submitStep1_2020.sh)
job8_2020=$(sbatch --dependency=afterok:${job7_2020##* } submitStep2_2020.sh)
sbatch --dependency=afterok:${job8_2020##* } submitStep3_2020.sh

