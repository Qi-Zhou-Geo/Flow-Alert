#!/bin/bash

#SBATCH --job-name=combined_steps    # Job name
#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

# Submit jobs for 2017
job_2017_step1=$(sbatch submitStep1_2017.sh)
sbatch --dependency=afterok:${job_2017_step1##* } submitStep3_2017.sh

# Submit jobs for 2018
job_2018_step1=$(sbatch submitStep1_2018.sh)
sbatch --dependency=afterok:${job_2018_step1##* } submitStep3_2018.sh

# Submit jobs for 2019
job_2019_step1=$(sbatch submitStep1_2019.sh)
sbatch --dependency=afterok:${job_2019_step1##* } submitStep3_2019.sh

# Submit jobs for 2020
job_2020_step1=$(sbatch submitStep1_2020.sh)
sbatch --dependency=afterok:${job_2020_step1##* } submitStep3_2020.sh

# Submit jobs for 2022
job_2022_step1=$(sbatch submitStep1_2022.sh)
sbatch --dependency=afterok:${job_2022_step1##* } submitStep3_2022.sh

# Delete the logs if it is zero
find logs/ -type f -name "*.txt" -size 0
find logs/ -type f -name "*.txt" -size 0 -delete