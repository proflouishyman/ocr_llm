import os
import pandas as pd
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
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
process_row_limit = 5  # Limit the number of rows to process (for testing)

# Helper function to get the latest checkpoint in a directory
def get_latest_checkpoint(model_dir):
    checkpoints = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None

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

    blank_count = 0
    scaler = GradScaler()

    # Outer progress bar for tracking overall progress across all models
    with tqdm(total=(len(model_dirs) + 1) * len(df), desc="Overall Progress") as pbar:
        # Process untrained model first
        print("Processing untrained model...")
        untrained_model = LlavaNextForConditionalGeneration.from_pretrained(untuned_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False).to(device)
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing untrained model", leave=False):
            if pd.isna(row[untrained_column]):
                image = load_image(f"{row['id']}.jpg")
                if image is None:
                    df.at[index, untrained_column] = "ERROR: Image not found"
                    pbar.update(1)
                    continue
                text_prompt = base_prompt.format(ocr_text=row['pyte_ocr'])
                result, is_blank = load_and_generate(untrained_model, processor, device, text_prompt, image)
                df.at[index, untrained_column] = result
                if is_blank:
                    blank_count += 1
            if (index + 1) % save_interval == 0:
                df.to_csv(output_csv_path, index=False)
                print(f"Progress saved at row {index + 1}")
            pbar.update(1)
        
        print(f"Untrained model processing complete. Blank responses: {blank_count}")

        # Process with fine-tuned models
        for model_name in model_dirs:
            model_dir = os.path.join(model_base_path, model_name, 'checkpoints', 'llava-hf', 'llava-v1.6-mistral-7b-hf-task-lora')
            latest_checkpoint = get_latest_checkpoint(model_dir)
            if latest_checkpoint:
                model = LlavaNextForConditionalGeneration.from_pretrained(latest_checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=False)
            else:
                print(f"No checkpoint found for {model_name}, skipping...")
                pbar.update(len(df))  # Update overall progress bar
                continue

            model.to(device)
            column_name = model_output_columns[model_name]
            
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
                    df.at[index, column_name] = result
                    if is_blank:
                        blank_count += 1
                
                if (index + 1) % save_interval == 0:
                    df.to_csv(output_csv_path, index=False)
                    print(f"Progress saved at row {index + 1}")
                
                pbar.update(1)  # Update overall progress bar

            print(f"Results saved for {model_name}, current total blank responses: {blank_count}")

    df.to_csv(output_csv_path, index=False)
    print(f"All processing complete. Final total blank responses: {blank_count}")

if __name__ == "__main__":
    main()