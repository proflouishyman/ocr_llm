import os
from llava.train.train import train

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    # Ensure necessary directories exist
    ensure_dir('/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/gold_100')
    ensure_dir('/data/lhyman6/OCR/data/images/')
    ensure_dir('./checkpoints/llava-v1.5-13b-task-lora')
    
    train(
        attn_implementation="flash_attention_2",
        #lora_enable=True,
        #lora_r=128,
        #lora_alpha=256,
        #mm_projector_lr=2e-5,
        deepspeed='./zero3.json',  # Adjust this path if needed
        model_name_or_path='liuhaotian/llava-v1.5-13b',
        version='v1',
        data_path='/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/gold_100/gold_100.json',
        image_folder='/data/lhyman6/OCR/data/images/',
        vision_tower='./clip-vit-large-patch14-336',  # Use the local path if downloaded manually
        mm_projector_type='mlp2x_gelu',
        mm_vision_select_layer=-2,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        image_aspect_ratio='pad',
        group_by_modality_length=True,
        bf16=True,
        output_dir='./checkpoints/llava-v1.5-13b-task-lora',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        evaluation_strategy='no',
        save_strategy='steps',
        save_steps=50000,
        save_total_limit=1,
        learning_rate=2e-4,
        weight_decay=0.,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        logging_steps=1,
        tf32=True,
        model_max_length=2048,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        lazy_preprocess=True,
        report_to='wandb'
    )
