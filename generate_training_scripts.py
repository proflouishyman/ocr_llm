import os

# Current working directory
current_dir = os.getcwd()

# Directory paths
base_dir = "/scratch4/lhyman6/OCR/OCR/ocr_llm/work"
script_path = "/data/lhyman6/OCR/scripts/ocr_llm/train_bart.py"

# Dataset names and sizes
datasets = ["gold", "silver"]
sizes = [100, 1000, 10000]

# SLURM job script template
template = """#!/bin/bash -l

#SBATCH --job-name=BART_train_{dataset}_{size}
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
#SBATCH --output=./slurm_logging/ocr_training_{dataset}_{size}_%j.out
#SBATCH --error=./slurm_logging/ocr_training_{dataset}_{size}_%j.err

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

TRAIN_DATASET="{train_dataset}"
VALIDATION_DATASET="{validation_dataset}"
TEST_DATASET="{test_dataset}"
OUTPUT_DIR="{output_dir}"
SCRIPT_PATH="{script_path}"

echo "Training dataset: $TRAIN_DATASET"
echo "Validation dataset: $VALIDATION_DATASET"
echo "Test dataset: $TEST_DATASET"
echo "Output directory: $OUTPUT_DIR"

# Running the training script using the transformers Trainer
srun python $SCRIPT_PATH --run_locally --train_dataset $TRAIN_DATASET --validation_dataset $VALIDATION_DATASET --test_dataset $TEST_DATASET --output_dir $OUTPUT_DIR

echo "Job finished on $(date)"
"""

# Loop through each dataset and size to create submission scripts
for dataset in datasets:
    for size in sizes:
        train_dataset = f"{base_dir}/{dataset}_{size}/train"
        validation_dataset = f"{base_dir}/{dataset}_{size}/val"
        test_dataset = f"{base_dir}/{dataset}_{size}/test"
        output_dir = f"{base_dir}/tuning_results_{dataset}_{size}"
        script_name = f"{current_dir}/train_bart_{dataset}_{size}.sh"

        # Replace placeholders in the template
        script_content = template.format(
            dataset=dataset,
            size=size,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            output_dir=output_dir,
            script_path=script_path
        )

        # Write the script to a file
        with open(script_name, 'w') as file:
            file.write(script_content)

        # Make the script executable
        os.chmod(script_name, 0o755)

        print(f"Generated script {script_name}")
