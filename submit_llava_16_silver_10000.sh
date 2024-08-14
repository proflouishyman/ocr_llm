#!/bin/bash -l

#SBATCH --job-name=s_10k_l16_2
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu
#SBATCH --account=lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=/data/lhyman6/OCR/scripts/ocr_llm/slurm_logging/ocr_job/llava_16_training_silver_10000_%j.out
#SBATCH --error=/data/lhyman6/OCR/scripts/ocr_llm/slurm_logging/ocr_job/llava_16_training_silver_10000_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Load modules and activate conda environment
module load anaconda
module load cuda/12.1 
module load gcc

source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate llavaenv

cd /data/lhyman6/OCR/scripts/ocr_llm

# Running the training script
bash train_llava_16_silver_10000.sh
