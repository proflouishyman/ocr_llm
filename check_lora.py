import os
import torch
from transformers import LlavaNextForConditionalGeneration
from peft import PeftModel, PeftConfig

# Configuration
untuned_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model_base_path = '/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16'
model_dirs = [
    'gold_100',
    'gold_1000',
    'gold_10000',
    'silver_100',
    'silver_1000',
    'silver_10000'
]

def get_latest_checkpoint(model_dir):
    checkpoints = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None

def load_base_model():
    return LlavaNextForConditionalGeneration.from_pretrained(
        untuned_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    )

def load_lora_model(base_model, adapter_model_path):
    return PeftModel.from_pretrained(base_model, adapter_model_path)

def check_lora_weights(base_model, fine_tuned_model):
    base_params = dict(base_model.named_parameters())
    fine_tuned_params = dict(fine_tuned_model.named_parameters())
    
    for name, param in fine_tuned_params.items():
        if 'lora' in name.lower():
            print(f"Checking LoRA parameter: {name}")
            if name in base_params:
                diff = torch.norm(param - base_params[name])
                print(f"  Difference (L2 norm): {diff.item()}")
            else:
                print(f"  New LoRA parameter. L2 norm: {torch.norm(param).item()}")

def main():
    print("Loading base model...")
    base_model = load_base_model()

    for model_name in model_dirs:
        print(f"\nChecking LoRA weights for {model_name}")
        model_dir = os.path.join(model_base_path, model_name, 'checkpoints', 'llava-hf', 'llava-v1.6-mistral-7b-hf-task-lora')
        latest_checkpoint = get_latest_checkpoint(model_dir)
        
        if latest_checkpoint:
            print(f"Loading LoRA from {latest_checkpoint}")
            fine_tuned_model = load_lora_model(base_model, latest_checkpoint)
            check_lora_weights(base_model, fine_tuned_model)
        else:
            print(f"No checkpoint found for {model_name}")

if __name__ == "__main__":
    main()