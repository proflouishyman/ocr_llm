#!/bin/bash -l

#SBATCH --job-name=BART_train_silver_10000
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
#SBATCH --output=./slurm_logging/ocr_training_silver_10000_%j.out
#SBATCH --error=./slurm_logging/ocr_training_silver_10000_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

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

TRAIN_DATASET="/scratch4/lhyman6/OCR/OCR/ocr_llm/work/silver_10000/train"
VALIDATION_DATASET="/scratch4/lhyman6/OCR/OCR/ocr_llm/work/silver_10000/val"
TEST_DATASET="/scratch4/lhyman6/OCR/OCR/ocr_llm/work/silver_10000/test"
OUTPUT_DIR="/scratch4/lhyman6/OCR/OCR/ocr_llm/work/tuning_results_silver_10000"
SCRIPT_PATH="/data/lhyman6/OCR/scripts/ocr_llm/train_bart.py"

echo "Training dataset: $TRAIN_DATASET"
echo "Validation dataset: $VALIDATION_DATASET"
echo "Test dataset: $TEST_DATASET"
echo "Output directory: $OUTPUT_DIR"

# Running the training script using the transformers Trainer
srun python $SCRIPT_PATH --run_locally --train_dataset $TRAIN_DATASET --validation_dataset $VALIDATION_DATASET --test_dataset $TEST_DATASET --output_dir $OUTPUT_DIR

echo "Job finished on $(date)"
