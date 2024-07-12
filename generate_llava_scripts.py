import os

# Base template for the fine-tuning file
training_template = """#!/bin/bash

export WANDB_MODE=offline
export WANDB_SILENT=true

deepspeed train_mem.py \\
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \\
    --deepspeed /data/lhyman6/OCR/scripts/ocr_llm/zero3.json \\
    --model_name_or_path liuhaotian/llava-v1.5-13b \\
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
    --output_dir /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/{data_type}_{size}/checkpoints/llava-v1.5-13b-task-lora \\
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
    --logging_steps 1 \\
    --tf32 False \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 4 \\
    --lazy_preprocess True \\
    #--report_to wandb
"""

# Base template for the submission Slurm script
slurm_template = """#!/bin/bash -l

#SBATCH --job-name=LLaVA_train_{data_type}_{size}
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu
#SBATCH --account lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=./slurm_logging/llava_training_{data_type}_{size}_%j.out
#SBATCH --error=./slurm_logging/llava_training_{data_type}_{size}_%j.err

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

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llavaenv

# Running the training script
srun bash train_llava_{data_type}_{size}.sh
"""

# Types and sizes
types = ["gold", "silver"]
sizes = [100, 1000, 10000]

# Create the scripts
for data_type in types:
    for size in sizes:
        # Create training script
        training_script_content = training_template.format(data_type=data_type, size=size)
        training_script_name = f"train_llava_{data_type}_{size}.sh"
        with open(training_script_name, "w") as script_file:
            script_file.write(training_script_content)

        # Create output directory if it doesn't exist
        output_dir = f"/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/{data_type}_{size}/checkpoints/llava-v1.5-13b-task-lora"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create submission Slurm script
        slurm_script_content = slurm_template.format(data_type=data_type, size=size)
        slurm_script_name = f"submit_llava_{data_type}_{size}.sh"
        with open(slurm_script_name, "w") as slurm_file:
            slurm_file.write(slurm_script_content)

print("Training and submission scripts generated successfully.")
