#!/bin/bash -l

#SBATCH --job-name=BART_train_{dataset}_{size}
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu
#SBATCH --account lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=./slurm_logging/ocr_training_{dataset}_{size}_%j.out
#SBATCH --error=./slurm_logging/ocr_training_{dataset}_{size}_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Activate conda environment
module load anaconda
module load cuda/12.1 
module load gcc

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llavaenv





# Running the training script using the transformers Trainer
sbatch THENAME OF THE TRAINING SH FILE