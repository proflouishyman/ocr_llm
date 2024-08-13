import os
import shutil
import pandas as pd
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel, PeftConfig
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image

#uses llavaenv
#8 13 2024

# Set environment variable to avoid warnings about parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration variables
csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/complete_testing_csv.csv'
output_csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/processed_llava_16_testing_csv.csv'
model_base_path = '/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava_16'
image_data_directory = '/data/lhyman6/OCR/scripts/data/second_images'
untuned_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# List of directories containing fine-tuned models
model_dirs = [
    'gold_100',
    'gold_1000',
    'gold_10000',
    'silver_100',
    'silver_1000',
    'silver_10000'
]

# Mapping of model names to output column names in the CSV
model_output_columns = {
    'gold_100': 'LLAVA_gold_100',
    'gold_1000': 'LLAVA_gold_1000',
    'gold_10000': 'LLAVA_gold_10000',
    'silver_100': 'LLAVA_silver_100',
    'silver_1000': 'LLAVA_silver_1000',
    'silver_10000': 'LLAVA_silver_10000'
}

# Separate column for untrained model
untrained_column = 'LLAVA_untuned'

# Base prompt template for OCR correction
base_prompt = "[INST] <image>\nCorrect this OCR: {ocr_text}[/INST]"
save_interval = 100  # Save progress every 100 rows
process_row_limit = 2  # Limit the number of rows to process (for testing)

# Helper function to get the latest checkpoint in a directory

def get_latest_checkpoint(model_dir):
    checkpoints = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None

def load_lora_model(base_model_id, adapter_model_path):
    # Load the base model
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    )
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    
    # Merge the LoRA weights with the base model
    model = model.merge_and_unload()
    
    return model

# New function to copy config.json to checkpoint directory
def copy_config_to_checkpoint(model_dir, checkpoint_dir):
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        shutil.copy(config_path, checkpoint_dir)
        print(f"Copied config.json to {checkpoint_dir}")
    else:
        print(f"Warning: config.json not found in {model_dir}")

# Helper function to load an image file
def load_image(image_filename):
    try:
        return Image.open(os.path.join(image_data_directory, image_filename))
    except IOError:
        return None  # Return None if image cannot be opened

# Function to generate text from the model given an image and prompt
def load_and_generate(model, processor, device, text, image):
    inputs = processor(text, image, return_tensors="pt").to(device)
    with autocast():
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    is_blank = not generated_text.strip()
    return generated_text if not is_blank else "ERROR: Blank response generated", is_blank




# Main function
def main():
    print("Starting processing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = LlavaNextProcessor.from_pretrained(untuned_model_id)
    
    # Load and preprocess the dataset
    df = pd.read_csv(csv_path)
    df = df.head(process_row_limit)

    # Ensure all output columns are of object type to store string data
    all_columns = list(model_output_columns.values()) + [untrained_column]
    for column in all_columns:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype('object')

    total_blank_count = 0
    scaler = GradScaler()

    # Load the base model once
    print(f"\nLoading base model: {untuned_model_id}")
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        untuned_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    ).to(device)
    print("Base model loaded successfully.")

    # Outer progress bar for tracking overall progress across all models
    with tqdm(total=(len(model_dirs) + 1) * len(df), desc="Overall Progress") as pbar:
        # Process untrained model first
        print("\nProcessing with untrained model...")

        untrained_blank_count = 0
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing untrained model", leave=False):
            if pd.isna(row[untrained_column]):
                image = load_image(f"{row['id']}.jpg")
                if image is None:
                    df.at[index, untrained_column] = "ERROR: Image not found"
                    pbar.update(1)
                    continue
                text_prompt = base_prompt.format(ocr_text=row['pyte_ocr'])
                result, is_blank = load_and_generate(base_model, processor, device, text_prompt, image)
                df.at[index, untrained_column] = result
                if is_blank:
                    untrained_blank_count += 1
                    total_blank_count += 1
            if (index + 1) % save_interval == 0:
                df.to_csv(output_csv_path, index=False)
                print(f"Progress saved at row {index + 1}")
            pbar.update(1)
        
        print(f"Untrained model processing complete. Blank responses: {untrained_blank_count}")

        # Process with fine-tuned models
        for model_name in model_dirs:
            print(f"\nProcessing with fine-tuned model: {model_name}")
            model_dir = os.path.join(model_base_path, model_name, 'checkpoints', 'llava-hf', 'llava-v1.6-mistral-7b-hf-task-lora')
            latest_checkpoint = get_latest_checkpoint(model_dir)
            if latest_checkpoint:
                try:
                    # Load only the LoRA adapter
                    peft_config = PeftConfig.from_pretrained(latest_checkpoint)
                    model = PeftModel.from_pretrained(base_model, latest_checkpoint)
                    print(f"LoRA adapter loaded for: {model_name}")
                    print(f"Checkpoint path: {latest_checkpoint}")
                except Exception as e:
                    print(f"Error loading LoRA adapter for {model_name}: {str(e)}")
                    pbar.update(len(df))
                    continue
            else:
                print(f"No checkpoint found for {model_name}, skipping...")
                pbar.update(len(df))
                continue

            column_name = model_output_columns[model_name]
            
            identical_count = 0
            model_blank_count = 0
            max_identical_threshold = 1  # Stop after 5 identical results
            
            # Inner progress bar for tracking progress within each model
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {model_name}", leave=False):
                if pd.isna(row[column_name]):
                    image = load_image(f"{row['id']}.jpg")
                    if image is None:
                        df.at[index, column_name] = "ERROR: Image not found"
                        pbar.update(1)
                        continue
                    
                    text_prompt = base_prompt.format(ocr_text=row['pyte_ocr'])
                    result, is_blank = load_and_generate(model, processor, device, text_prompt, image)
                    
                    # Compare with untrained model result
                    if result == row[untrained_column]:
                        identical_count += 1
                        if identical_count >= max_identical_threshold:
                            print(f"\nWARNING: {model_name} produced {max_identical_threshold} consecutive identical results to the untrained model.")
                            print("LoRA may not be applied correctly. Skipping to next model.")
                            break
                    else:
                        identical_count = 0  # Reset counter if results differ
                    
                    df.at[index, column_name] = result
                    if is_blank:
                        model_blank_count += 1
                        total_blank_count += 1
                
                if (index + 1) % save_interval == 0:
                    df.to_csv(output_csv_path, index=False)
                    print(f"Progress saved at row {index + 1}")
                
                pbar.update(1)  # Update overall progress bar

            print(f"Results saved for {model_name}. Identical responses: {identical_count}, Blank responses: {model_blank_count}")

            # Unload the LoRA adapter
            model = base_model

    df.to_csv(output_csv_path, index=False)
    print(f"\nAll processing complete. Total blank responses across all models: {total_blank_count}")
if __name__ == "__main__":
    main()