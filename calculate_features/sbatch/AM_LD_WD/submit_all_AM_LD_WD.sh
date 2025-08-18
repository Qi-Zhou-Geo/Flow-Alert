#!/bin/bash

#SBATCH --job-name=combined_steps    # Job name
#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

# Submit jobs for 2024 AM (sequential)
job1=$(sbatch submitStep1_AM_2024.sh)
#job2=$(sbatch --dependency=afterok:${job1##* } submitStep2_AM_2024.sh)
#sbatch --dependency=afterok:${job2##* } submitStep3_AM_2024.sh
sbatch --dependency=afterok:${job1##* } submitStep3_AM_2024.sh

# Submit jobs for 2023 LD (sequential)
job1=$(sbatch submitStep1_LD_2023.sh)
#job2=$(sbatch --dependency=afterok:${job1##* } submitStep2_LD_2023.sh)
#sbatch --dependency=afterok:${job2##* } submitStep3_LD_2023.sh
sbatch --dependency=afterok:${job1##* } submitStep3_LD_2023.sh

# Submit jobs for 2023 WD (sequential)
job1=$(sbatch submitStep1_WD_2023.sh)
#job2=$(sbatch --dependency=afterok:${job1##* } submitStep2_WD_2023.sh)
#sbatch --dependency=afterok:${job2##* } submitStep3_WD_2023.sh
sbatch --dependency=afterok:${job1##* } submitStep3_WD_2023.sh

# Delete the logs if it is zero
find logs/ -type f -name "*.txt" -size 0
find logs/ -type f -name "*.txt" -size 0 -delete