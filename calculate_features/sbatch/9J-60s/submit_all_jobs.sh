#!/bin/bash

#SBATCH --job-name=combined_steps    # Job name
#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

# Submit jobs for 2013 (sequential)
job1=$(sbatch submitStep1_2013.sh)
sbatch --dependency=afterok:${job1##* } submitStep3_2013.sh

# Submit jobs for 2014 (sequential)
job2=$(sbatch submitStep1_2014.sh)
sbatch --dependency=afterok:${job2##* } submitStep3_2014.sh

# Delete the logs if it is zero
find logs/ -type f -name "*.txt" -size 0
find logs/ -type f -name "*.txt" -size 0 -delete