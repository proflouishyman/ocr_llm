import os
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from tqdm import tqdm
from PIL import Image
#completely untested.

# Variables
csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/complete_testing_csv.csv'
output_csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/processed_1000_testing_csv.csv'
model_base_path = '/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava'
image_data_directory = '/data/lhyman6/OCR/scripts/data/second_images'  # Assuming image data may be used
untuned_model_id = "llava-hf/llava-1.5-7b-hf"
quantization_config = {}  # Define this if needed for deploying quantized models

model_dirs = [
    'gold_100',
    'gold_1000',
    'gold_10000',
    'silver_100',
    'silver_1000',
    'silver_10000'
]
model_output_columns = {
    'gold_100': 'LLAVA_gold_100',
    'gold_1000': 'LLAVA_gold_1000',
    'gold_10000': 'LLAVA_gold_10000',
    'silver_100': 'LLAVA_silver_100',
    'silver_1000': 'LLAVA_silver_1000',
    'silver_10000': 'LLAVA_silver_10000',
    'untuned': 'LLAVA_untuned'
}

base_prompt = "Correct this OCR:"
save_interval = 100
process_row_limit = 2000

# Helper functions
def get_latest_checkpoint(model_dir):
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None

def load_image(image_filename):
    try:
        return Image.open(os.path.join(image_data_directory, image_filename))
    except IOError:
        return None  # Return None if image cannot be opened

def load_and_generate(model, processor, device, text, image=None):
    if image:
        inputs = processor(text, images=image, return_tensors='pt').to(device)
    else:
        inputs = processor(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=1000,
            min_length=50,
            num_beams=5, 
            length_penalty=2.0,
            early_stopping=True
        )
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    return generated_text if generated_text.strip() else "ERROR: Blank response generated"

# Main function
def main():
    print("Starting processing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(untuned_model_id)
    model = LlavaForConditionalGeneration.from_pretrained(untuned_model_id, device_map="auto")
    df = pd.read_csv(csv_path)
    df = df.head(process_row_limit)

    for column in model_output_columns.values():
        df[column] = df[column].astype('object')  # Ensure column is ready for string data

    blank_count = 0

    # Process with untuned model
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing untuned LLAVA"):
        image = load_image(f"{row['id']}.jpg")  # Assumes 'id' column for image filenames
        text_prompt = f"{base_prompt} {row['pyte_ocr']}"
        if pd.isna(row['LLAVA_untuned']):
            result = load_and_generate(model, processor, device, text_prompt, image)
            df.at[index, 'LLAVA_untuned'] = result if result != "ERROR: Blank response generated" else pd.NA
            if result == "ERROR: Blank response generated":
                blank_count += 1
        if (index + 1) % save_interval == 0:
            df.to_csv(output_csv_path, index=False)
            print(f"Progress saved at row {index + 1}")

    df.to_csv(output_csv_path, index=False)
    print("Results saved back to the new CSV, total blank responses: ", blank_count)

if __name__ == "__main__":
    main()
