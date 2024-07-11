#!/bin/bash -l

#SBATCH --job-name=BART_train
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:2
#SBATCH --qos=qos_gpu
#SBATCH --account lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=./slurm_logging/ocr_training_%j.out
#SBATCH --error=./slurm_logging/ocr_training_%j.err

# Activate conda environment
module load anaconda
module load cuda/12.1 
module load gcc

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ocrenv

# Set environment variables for PyTorch distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)  # Random free port
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Variables for dataset and model training
TRAIN_DATASET=$1
VALIDATION_DATASET=$2
TEST_DATASET=$3
OUTPUT_DIR=$4
SCRIPT_PATH="/data/lhyman6/OCR/scripts/train_bart.py"

# Running the training script using the transformers Trainer
srun python $SCRIPT_PATH --run_locally --train_dataset $TRAIN_DATASET --validation_dataset $VALIDATION_DATASET --test_dataset $TEST_DATASET --output_dir $OUTPUT_DIR
