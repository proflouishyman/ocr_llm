import os

# Base template for the fine-tuning file
training_template = """#!/bin/bash

# Function to find a random available port
find_free_port() {{
  while true; do
    PORT=$(shuf -i 20000-29999 -n 1)
    if ! ss -lpn | grep -q ":$PORT "; then
      echo $PORT
      return
    fi
  done
}}

# Set a unique port for distributed training
export MASTER_PORT=$(find_free_port)

# Debugging: Print the chosen port to verify it's set correctly
echo "Chosen MASTER_PORT: $MASTER_PORT"

# Verify the port is actually free
if ss -lpn | grep -q ":$MASTER_PORT "; then
  echo "Port $MASTER_PORT is already in use. Exiting."
  exit 1
fi

# Set CUTLASS environment variable
export CUTLASS_PATH=/data/lhyman6/OCR/scripts/ocr_llm/cutlass/cutlass
export PATH="$HOME/.local/bin:$PATH"
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate llavaenv
cd /data/lhyman6/OCR/scripts/ocr_llm

export WANDB_MODE=offline
export WANDB_SILENT=true

deepspeed train_mem.py \\
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \\
    --deepspeed /data/lhyman6/OCR/scripts/ocr_llm/zero3.json \\
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \\
    --version v1 \\
    --data_path /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/{data_type}_{size}/{data_type}_{size}.json \\
    --image_folder ./data/lhyman6/OCR/data/images/ \\
    --vision_tower ./clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --group_by_modality_length True \\
    --fp16 False \\
    --bf16 True \\
    --output_dir /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16/{data_type}_{size}/checkpoints/llava-hf/llava-v1.6-mistral-7b-hf-task-lora \\
    --num_train_epochs 15 \\
    --per_device_train_batch_size 16 \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps 1 \\
    --evaluation_strategy "no" \\
    --save_strategy "epoch" \\
    --save_steps 1 \\
    --save_total_limit 10 \\
    --learning_rate 2e-4 \\
    --weight_decay 0. \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 5 \\
    --tf32 False \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 4 \\
    --lazy_preprocess True \\
    #--report_to wandb
"""



# Base template for the submission Slurm script
slurm_template = """#!/bin/bash -l

#SBATCH --job-name={job_name}
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:2
#SBATCH --qos=qos_gpu
#SBATCH --account lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=./slurm_logging/llava_16_training_{data_type}_{size}_%j.out
#SBATCH --error=./slurm_logging/llava_16_training_{data_type}_{size}_%j.err

echo "Job started on $(date)"
echo "Running on node $(hostname)"

# Function to find a random available port
find_free_port() {{
  while true; do
    PORT=$(shuf -i 20000-29999 -n 1)
    ss -lpn | grep -q ":$PORT " || break
  done
  echo $PORT
}}

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

source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate llavaenv

cd /data/lhyman6/OCR/scripts/ocr_llm

# Running the training script
srun bash train_llava_16_{data_type}_{size}.sh
"""



# Types and sizes
types = ["gold", "silver"]
sizes = [100, 1000, 10000]

# Function to generate the job name
def generate_job_name(data_type, size):
    size_map = {100: "100", 1000: "1k", 10000: "10k"}
    prefix = "g" if data_type == "gold" else "s"
    return f"{prefix}_{size_map[size]}_l16_1"

# Create the scripts
# Create the scripts

# Create the scripts
for data_type in types:
    for size in sizes:
        # Create training script
        training_script_content = training_template.format(data_type=data_type, size=size)
        training_script_name = f"train_llava_16_{data_type}_{size}.sh"
        with open(training_script_name, "w") as script_file:
            script_file.write(training_script_content)
        print(f"Created training script: {training_script_name}")

        # Generate job name
        job_name = generate_job_name(data_type, size)
        
        # Create submission Slurm script
        slurm_script_content = slurm_template.format(data_type=data_type, size=size, job_name=job_name)
        slurm_script_name = f"submit_llava_16_{data_type}_{size}.sh"
        with open(slurm_script_name, "w") as slurm_file:
            slurm_file.write(slurm_script_content)
        print(f"Created Slurm script: {slurm_script_name}")