#!/bin/bash -l

# This is how you run this:
# sbatch run_bart_slurm.sh

#SBATCH --job-name=run_bart_slurm3
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks=2  # Number of tasks should match the number of GPUs
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:2  # Number of GPUs requested
#SBATCH --qos=qos_gpu
#SBATCH --account lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=./slurm_logging/ocr_training_%j.out
#SBATCH --error=./slurm_logging/ocr_training_%j.err

# Activate conda environment
ml anaconda
module load cuda/12.1 
ml gcc

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ocrenv

# Set environment variables for PyTorch distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)  # Random free port
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"


srun --ntasks=$SLURM_NTASKS python /data/lhyman6/OCR/scripts/bart_over_data_modeltest_debug_2.py  --limit 100
