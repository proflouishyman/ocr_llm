#!/bin/bash

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

deepspeed train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /data/lhyman6/OCR/scripts/ocr_llm/zero3.json \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --version v1 \
    --data_path /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/gold_1000/gold_1000.json \
    --image_folder ./data/lhyman6/OCR/data/images/ \
    --vision_tower ./clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 False \
    --bf16 True \
    --output_dir /scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16/gold_1000/checkpoints/llava-hf/llava-v1.6-mistral-7b-hf-task-lora \
    --num_train_epochs 15 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    #--report_to wandb
