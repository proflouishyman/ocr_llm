import os

# Create logging directory if it doesn't exist
os.makedirs("/data/lhyman6/OCR/scripts/ocr_llm/slurm_logging/ocr_job", exist_ok=True)

# Base template for the fine-tuning file
training_template = """#!/bin/bash

# Set the number of GPUs
NUM_GPUS=4

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

# Set environment variables
export CUTLASS_PATH=/data/lhyman6/OCR/scripts/ocr_llm/cutlass/cutlass
export PATH="$HOME/.local/bin:$PATH"
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER

# Load modules and activate conda environment
module load anaconda
module load cuda/12.1 
module load gcc

source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate llavaenv

cd /data/lhyman6/OCR/scripts/ocr_llm

export WANDB_MODE=offline
export WANDB_SILENT=true

# Calculate per-GPU batch size
TOTAL_BATCH_SIZE=12  # Reduced from 16
PER_GPU_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NUM_GPUS))

# Debug information
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Function to log system metrics
log_metrics() {{
    LOG_FILE="/data/lhyman6/OCR/scripts/ocr_llm/slurm_logging/ocr_job/system_metrics_{data_type}_{size}.log"
    echo "Timestamp,CPU_Usage,RAM_Usage,Swap_Usage,Disk_Usage,GPU_Usage,GPU_Memory" > $LOG_FILE
    
    while true; do
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{{print $2 + $4}}')
        RAM_USAGE=$(free -m | awk '/Mem:/ {{ print $3 }}')
        SWAP_USAGE=$(free -m | awk '/Swap:/ {{ print $3 }}')
        DISK_USAGE=$(df -h | awk '$NF=="/data/lhyman6/OCR/scripts/ocr_llm" {{print $5}}')
        GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk '{{print $1","$2}}')
        
        echo "$TIMESTAMP,$CPU_USAGE,$RAM_USAGE,$SWAP_USAGE,$DISK_USAGE,$GPU_STATS" >> $LOG_FILE
        sleep 60  # Log every minute
    done
}}

# Start logging in the background
log_metrics &
LOG_PID=$!

# Add debugging section to check model layers
python3 << END
from transformers import AutoModelForCausalLM
import torch

print("Starting model layer check...")

# Load the model
model = AutoModelForCausalLM.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Check if model is in training mode
print(f"Model is in training mode: {{model.training}}")

# Check if all parameters require gradients
all_require_grad = all(p.requires_grad for p in model.parameters())
print(f"All parameters require gradients: {{all_require_grad}}")

# Print details for each named parameter
for name, param in model.named_parameters():
    print(f"{{name}}:")
    print(f"  Requires grad: {{param.requires_grad}}")
    print(f"  In training mode: {{param.training}}")
    print(f"  Shape: {{param.shape}}")
    print(f"  Data type: {{param.dtype}}")
    print("---")

# Check total number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {{total_params}}")
print(f"Trainable parameters: {{trainable_params}}")

print("Model layer check completed.")
END

echo "Starting DeepSpeed training..."

# Run DeepSpeed directly (without srun)
deepspeed --num_gpus=$NUM_GPUS train_mem.py \\
    --deepspeed /data/lhyman6/OCR/scripts/ocr_llm/zero3.json \\
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \\
    --version v1 \\
    --data_path /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/{data_type}_{size}/{data_type}_{size}.json \\
    --image_folder /data/lhyman6/OCR/data/images/ \\
    --vision_tower /data/lhyman6/OCR/scripts/ocr_llm/clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --group_by_modality_length True \\
    --fp16 False \\
    --bf16 True \\
    --output_dir /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16/{data_type}_{size}/checkpoints/llava-hf/llava-v1.6-mistral-7b-hf-task \\
    --num_train_epochs 15 \\
    --per_device_train_batch_size $PER_GPU_BATCH_SIZE \\
    --per_device_eval_batch_size 4 \\
    --gradient_accumulation_steps 2 \\
    --evaluation_strategy "no" \\
    --save_strategy "epoch" \\
    --save_steps 1 \\
    --save_total_limit 10 \\
    --learning_rate 5e-5 \\
    --weight_decay 0.01 \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 5 \\
    --tf32 False \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --dataloader_num_workers 4 \\
    --lazy_preprocess True \\
    --run_name "llava_16_{data_type}_{size}_training"


# Stop the background logging
kill $LOG_PID

echo "DeepSpeed training completed."


"""

# Base template for the submission Slurm script
slurm_template = """#!/bin/bash -l

#SBATCH --job-name={job_name}
#SBATCH --time=24:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu
#SBATCH --account=lhyman6_gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhyman6@jhu.edu
#SBATCH --output=/data/lhyman6/OCR/scripts/ocr_llm/slurm_logging/ocr_job/llava_16_training_{data_type}_{size}_%j.out
#SBATCH --error=/data/lhyman6/OCR/scripts/ocr_llm/slurm_logging/ocr_job/llava_16_training_{data_type}_{size}_%j.err

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
bash train_llava_16_{data_type}_{size}.sh
"""

# Types and sizes
types = ["gold", "silver"]
sizes = [100, 1000, 10000]

# Function to generate the job name
def generate_job_name(data_type, size):
    size_map = {100: "100", 1000: "1k", 10000: "10k"}
    prefix = "g" if data_type == "gold" else "s"
    return f"{prefix}_{size_map[size]}_l16_2"

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