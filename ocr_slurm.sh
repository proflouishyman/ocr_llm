#!/bin/bash

#definitely double check everything. i mess with this a lot. 

# This is how you run this:
# sbatch ocr_slurm.sh

#SBATCH --job-name=ocr-job
#SBATCH --output=./slurm_logging/ocr_job_%A_%a.out  # Output file, where %A is the job ID and %a is the array index
#SBATCH --error=./slurm_logging/ocr_job_%A_%a.err   # Error file
#SBATCH --ntasks=1                  # Each task is a single process (good practice for array jobs)
#SBATCH --nodes=1                     # Each job array task runs on one node
#SBATCH --ntasks-per-node=1           # One task per node

#SBATCH --time=2:00:00              # Time limit for the job
#SBATCH --array=0-199                # Creates  tasks in the array
#SBATCH --cpus-per-task=4           # Number of CPU cores per task

#SBATCH --account=lhyman6               # Adjust to your account
#SBATCH --export=ALL 


# Activate the conda environment

ml anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ocrenv

# Run the Python script
srun python /data/lhyman6/OCR/scripts/ocr_images_pyte_slurm_nocheck_3 $SLURM_ARRAY_TASK_ID
