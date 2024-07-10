import argparse
import os
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import gc
import json
import logging

#runs on one GPU and works perfectly.

# Setup logging
logger = logging.getLogger('SimplifiedLogger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('/data/lhyman6/OCR/debug/simplified_process.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

class TextDataset(Dataset):
    def __init__(self, tokenizer, df, max_length=1024):
        self.data = []
        self.original_indices = []
        self.part_numbers = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        for i, text in enumerate(df['ocr'].tolist()):
            parts = self.split_text(text)
            for part_num, part in enumerate(parts):
                self.data.append(tokenizer(part, truncation=True, padding='max_length', max_length=max_length))
                self.original_indices.append(i)
                self.part_numbers.append(part_num)
        logger.info(f"Dataset initialized with {len(self.data)} parts")

    def split_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.max_length:
            return [text]
        parts = []
        for i in range(0, len(tokens), self.max_length):
            parts.append(self.tokenizer.convert_tokens_to_string(tokens[i:i+self.max_length]))
        return parts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.data[idx].items()}
        item['original_index'] = self.original_indices[idx]
        item['part_number'] = self.part_numbers[idx]
        return item

    def __len__(self):
        return len(self.data)

def load_model(model_path, device):
    logger.info("Loading model...")
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    logger.info("Model loaded successfully")
    return model, tokenizer

def generate_text_with_logging(model, dataloader, device, tokenizer, df, output_path):
    model.eval()
    corrected_texts = {}
    reasons = {}  # Dictionary to store reasons for outputs
    detailed_outputs = {}  # Dictionary to store detailed outputs for each index
    status_updates = []  # List to store status updates

    for batch_num, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        logger.debug(f"Processing batch {batch_num}")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        original_indices = batch['original_index'].tolist()
        part_numbers = batch['part_number'].tolist()
        max_length = input_ids.size(1) + 50  # Increase max length to avoid truncation

        with torch.no_grad():
            with autocast():
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=5, early_stopping=True)
            batch_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            for i, (idx, part_num, text, output) in enumerate(zip(original_indices, part_numbers, batch_texts, outputs)):
                logger.debug(f"Index {idx}, Part {part_num}: Text generated")
                if idx not in corrected_texts:
                    corrected_texts[idx] = {}
                    reasons[idx] = {}
                    detailed_outputs[idx] = {}
                corrected_texts[idx][part_num] = text
                detailed_outputs[idx][part_num] = {
                    "input_ids": input_ids[i].tolist(),
                    "attention_mask": attention_mask[i].tolist(),
                    "output_ids": output.tolist(),
                    "output_text": text
                }
                if text.strip() == "":
                    reasons[idx][part_num] = "Model produced empty output"
                    status_updates.append(f"Empty output for index {idx}, part {part_num}")
                else:
                    reasons[idx][part_num] = "Model produced valid output"

            logger.debug(f"Batch {batch_num} processed with indices: {original_indices}")

        # Clear cache and run garbage collection to free up memory
        torch.cuda.empty_cache()
        gc.collect()

    assembled_texts = [
        " ".join([corrected_texts.get(idx, {}).get(part_num, "") for part_num in sorted(corrected_texts.get(idx, {}).keys())])
        for idx in range(len(df))
    ]
    notes = [
        " | ".join([reasons.get(idx, {}).get(part_num, "") for part_num in sorted(reasons.get(idx, {}).keys())])
        for idx in range(len(df))
    ]
    df['model_corrected'] = assembled_texts
    df['notes'] = notes

    # Save the detailed outputs and status updates to a JSON file
    output_json = {
        "corrected_texts": corrected_texts,
        "detailed_outputs": detailed_outputs,
        "status_updates": status_updates
    }

    with open(output_path.replace('.csv', '_detailed_outputs.json'), 'w') as f:
        json.dump(output_json, f, indent=4)

    return assembled_texts, detailed_outputs, status_updates  # Return detailed outputs for further analysis if needed

def main():
    logger.info("Starting main function")
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50, help='Number of rows to process')
    parser.add_argument('--model_path', type=str, default='/scratch4/lhyman6/OCR/work/tuning_results_long/checkpoint-15500', help='Path to the model')
    parser.add_argument('--data_path', type=str, default='/data/lhyman6/OCR/1919_ocr_loaded_sorted.csv', help='Path to the data')
    parser.add_argument('--output_path', type=str, default='/data/lhyman6/OCR/debug/1919_debug4.csv', help='Path to the output')


    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, tokenizer = load_model(args.model_path, device)

    logger.info("Loading data...")
    df = pd.read_csv(args.data_path)
    logger.info(f"Data loaded: {len(df)} rows")

    if args.limit is not None:
        df = df.head(args.limit)
        logger.info(f"Limiting data to {args.limit} rows")

    if 'model_corrected' in df.columns:
        df['model_corrected'] = df['model_corrected'].fillna("")
    else:
        df['model_corrected'] = ""

    # Filter out already processed data
    unprocessed_df = df[df['model_corrected'] == ""]
    logger.info(f"Unprocessed data: {len(unprocessed_df)} rows")
    
    if unprocessed_df.empty:
        logger.info(f"All rows in {args.data_path} have already been processed.")
        return

    dataset = TextDataset(tokenizer, unprocessed_df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    logger.info("Starting text generation...")
    corrected_texts, detailed_outputs, status_updates = generate_text_with_logging(model, dataloader, device, tokenizer, df, args.output_path)
    logger.info("Text generation completed")

    df['model_corrected'] = corrected_texts
    df.to_csv(args.output_path, index=False)
    logger.info(f"Final output saved to {args.output_path}")

    # Save detailed outputs for further analysis
    detailed_output_path = args.output_path.replace('.csv', '_detailed_outputs.json')
    with open(detailed_output_path, 'w') as f:
        json.dump(detailed_outputs, f, indent=4)
    logger.info(f"Detailed outputs saved to {detailed_output_path}")

if __name__ == "__main__":
    logger.info("Executing main")
    main()
    logger.info("Main execution finished")
