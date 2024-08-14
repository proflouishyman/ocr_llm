#!/bin/bash

# Set the number of GPUs
NUM_GPUS=2

# Function to find a random available port
find_free_port() {
  while true; do
    PORT=$(shuf -i 20000-29999 -n 1)
    if ! ss -lpn | grep -q ":$PORT "; then
      echo $PORT
      return
    fi
  done
}

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
TOTAL_BATCH_SIZE=8  # Reduced from 16
PER_GPU_BATCH_SIZE=$((TOTAL_BATCH_SIZE / NUM_GPUS))

# Debug information
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Add debugging section to check model layers
python3 << END
from transformers import AutoModelForCausalLM
import torch

print("Starting model layer check...")

# Load the model
model = AutoModelForCausalLM.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Check if model is in training mode
print(f"Model is in training mode: {model.training}")

# Check if all parameters require gradients
all_require_grad = all(p.requires_grad for p in model.parameters())
print(f"All parameters require gradients: {all_require_grad}")

# Print details for each named parameter
for name, param in model.named_parameters():
    print(f"{name}:")
    print(f"  Requires grad: {param.requires_grad}")
    print(f"  In training mode: {param.training}")
    print(f"  Shape: {param.shape}")
    print(f"  Data type: {param.dtype}")
    print("---")

# Check total number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

print("Model layer check completed.")
END

echo "Starting DeepSpeed training..."

# Start GPU memory monitoring
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > gpu_usage.log &

# Run DeepSpeed directly (without srun)
deepspeed --num_gpus=$NUM_GPUS train_mem.py \
    --deepspeed /data/lhyman6/OCR/scripts/ocr_llm/zero3.json \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --version v1 \
    --data_path /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/gold_1000/gold_1000.json \
    --image_folder /data/lhyman6/OCR/data/images/ \
    --vision_tower /data/lhyman6/OCR/scripts/ocr_llm/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 False \
    --bf16 True \
    --output_dir /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16/gold_1000/checkpoints/llava-hf/llava-v1.6-mistral-7b-hf-task \
    --num_train_epochs 15 \
    --per_device_train_batch_size $PER_GPU_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name "llava_16_gold_1000_training"

echo "DeepSpeed training completed."
