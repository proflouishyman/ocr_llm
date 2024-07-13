import os
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm

# Variables
csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/complete_testing_csv.csv'
output_csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/processed_1000_testing_csv.csv'  # New output CSV path
model_base_path = '/scratch4/lhyman6/OCR/OCR/ocr_llm/work'
model_dirs = [
    'tuning_results_gold_100',
    'tuning_results_gold_1000',
    'tuning_results_gold_10000',
    'tuning_results_silver_100',
    'tuning_results_silver_1000',
    'tuning_results_silver_10000'
]
model_output_columns = {
    'tuning_results_gold_100': 'BART_gold_100',
    'tuning_results_gold_1000': 'BART_gold_1000',
    'tuning_results_gold_10000': 'BART_gold_10000',
    'tuning_results_silver_100': 'BART_silver_100',
    'tuning_results_silver_1000': 'BART_silver_1000',
    'tuning_results_silver_10000': 'BART_silver_10000'
}

base_prompt = "Correct this OCR:"
save_interval = 100  # Save after processing every 50 rows
process_row_limit = 2000  # Number of rows to process

# Helper functions
def get_latest_checkpoint(model_dir):
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    return max(checkpoints, key=lambda x: int(x.split('-')[-1])) if checkpoints else None

def load_and_generate(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=1000,  # Increased max_length
            min_length=50,  # Added min_length
            num_beams=5, 
            length_penalty=2.0,  # Added length_penalty
            early_stopping=True
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not generated_text.strip():
        return "ERROR: Blank response generated", True
    return generated_text, False


# Main function
def main():
    print("Starting processing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    df = pd.read_csv(csv_path)
    df = df.head(process_row_limit)  # Limit the DataFrame to the first 'process_row_limit' rows

    # Set column data types to object for string handling
    for column in model_output_columns.values():
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype('object')

    # Ensure the untrained BART column is correctly formatted
    untrained_column = 'BART_untuned'
    if untrained_column not in df.columns:
        df[untrained_column] = pd.NA
    df[untrained_column] = df[untrained_column].astype('object')

    blank_count = 0  # Initialize counter for blank responses

    print("Loading the untrained BART model...")
    untrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing untrained BART model"):
        if pd.isna(row[untrained_column]):
            result, is_blank = load_and_generate(untrained_model, tokenizer, device, f"{base_prompt}: {row['pyte_ocr']}")
            if is_blank:
                blank_count += 1
            df.at[index, untrained_column] = result
        # Save periodically
        if (index + 1) % save_interval == 0:
            df.to_csv(output_csv_path, index=False)
            print(f"Progress saved at row {index + 1}")
    print(f"Untrained BART model processed successfully, {blank_count} blank responses detected.")

    # Process with trained models
    for dir_name in model_dirs:
        model_dir = os.path.join(model_base_path, dir_name)
        latest_checkpoint = get_latest_checkpoint(model_dir)
        if latest_checkpoint:
            full_path = os.path.join(model_dir, latest_checkpoint)
            model = BartForConditionalGeneration.from_pretrained(full_path).to(device)
            model_column = model_output_columns[dir_name]
            print(f"Processing with model from {full_path}...")
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {dir_name}"):
                if pd.isna(row[model_column]):
                    result, is_blank = load_and_generate(model, tokenizer, device, f"{base_prompt}: {row['pyte_ocr']}")
                    if is_blank:
                        blank_count += 1
                    df.at[index, model_column] = result
                # Save periodically
                if (index + 1) % save_interval == 0:
                    df.to_csv(output_csv_path, index=False)
                    print(f"Progress saved at row {index + 1}")
            print(f"Completed processing for {model_column}, {blank_count} blank responses detected.")

    # Saving results back to the new CSV
    df.to_csv(output_csv_path, index=False)
    print("Results saved back to the new CSV, total blank responses: ", blank_count)

if __name__ == "__main__":
    main()
