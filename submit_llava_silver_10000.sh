#!/bin/bash -l

#SBATCH --job-name=s_10k_l
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:2
#SBATCH --qos=qos_gpu
#SBATCH --account lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=./slurm_logging/llava_16_training_silver_10000_%j.out
#SBATCH --error=./slurm_logging/llava_16_training_silver_10000_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Function to find a random available port
find_free_port() {
  while true; do
    PORT=$(shuf -i 20000-29999 -n 1)
    ss -lpn | grep -q ":$PORT " || break
  done
  echo $PORT
}

# Set a unique port for distributed training
export MASTER_PORT=$(find_free_port)

# Set CUTLASS environment variable
export CUTLASS_PATH=/data/lhyman6/OCR/scripts/ocr_llm/cutlass/cutlass

# Ensure DeepSpeed is on the PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate conda environment
module load anaconda
module load cuda/12.1 
module load gcc

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llavaenv

# Running the training script
srun bash train_llava_silver_10000.sh
