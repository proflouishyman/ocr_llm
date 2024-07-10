# Created on: 2024-05-14
# Purpose: Run BART model for text generation with SLURM support for distributed training

import argparse
import os
import socket
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import torch.distributed as dist
import gc
import json

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
        print(f"Dataset initialized with {len(self.data)} parts")

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
    print("Loading model...")
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    print("Model loaded successfully")
    return model, tokenizer

import json

# Function to generate text with logging and detailed outputs
def generate_text(model, dataloader, device, tokenizer, df, output_path, save_interval=10):
    model.eval()
    corrected_texts = {}
    reasons = {}  # Dictionary to store reasons for outputs
    detailed_outputs = {}  # Dictionary to store detailed outputs for each index
    status_updates = []  # List to store status updates

    for batch_num, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        original_indices = batch['original_index'].tolist()
        part_numbers = batch['part_number'].tolist()
        max_length = input_ids.size(1) + 50  # Increase max length to avoid truncation

        with torch.no_grad():
            with autocast():
                outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=5, early_stopping=True)
            batch_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            for i, (idx, part_num, text, output) in enumerate(zip(original_indices, part_numbers, batch_texts, outputs)):
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

        # Save progress and status updates periodically
        if batch_num % save_interval == 0 and dist.get_rank() == 0:
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
            df.to_csv(output_path, index=False)
            status_updates.append(f"Saved progress at batch {batch_num}")

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

    # Return detailed outputs and status updates for further analysis
    return assembled_texts, detailed_outputs, status_updates

# Example of using the function
corrected_texts, detailed_outputs, status_updates = generate_text_with_logging(model, dataloader, device, tokenizer, df, output_path)

# Save the detailed outputs and status updates to a JSON file
output_json = {
    "corrected_texts": corrected_texts,
    "detailed_outputs": detailed_outputs,
    "status_updates": status_updates
}

with open(output_path.replace('.csv', '_detailed_outputs.json'), 'w') as f:
    json.dump(output_json, f, indent=4)


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
    return assembled_texts, detailed_outputs  # Return detailed outputs for further analysis if needed

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup_distributed_environment(rank, world_size, master_port):
    print(f"Setting up distributed environment for rank {rank}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    print(f"Distributed environment set up for rank {rank}")
    return device

def print_setup_details(rank, world_size):
    if rank == 0:
        print(f"Total GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"World size: {world_size}")

def log_memory_usage(rank):
    print(f"Rank {rank} memory usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(rank) / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(rank) / 1024**2:.2f} MB")

def main_worker(rank, world_size, args, model_path, data_path, output_path, master_port):
    try:
        print(f"Starting main_worker on rank {rank}")
        device = setup_distributed_environment(rank, world_size, master_port)
        print(f"Rank {rank} using device: {device}")

        if rank == 0:
            print_setup_details(rank, world_size)

        log_memory_usage(rank)

        model, tokenizer = load_model(model_path, device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        log_memory_usage(rank)

        print("Loading data...")
        df = pd.read_csv(data_path)
        print(f"Data loaded: {len(df)} rows")

        if 'model_corrected' in df.columns:
            df['model_corrected'] = df['model_corrected'].fillna("")
        else:
            df['model_corrected'] = ""

        if args.limit is not None:
            df = df.head(args.limit)
            print(f"Limiting data to {args.limit} rows")

        # Filter out already processed data
        unprocessed_df = df[df['model_corrected'] == ""]
        print(f"Unprocessed data: {len(unprocessed_df)} rows")
        
        if unprocessed_df.empty:
            print(f"All rows in {data_path} have already been processed.")
            return

        dataset = TextDataset(tokenizer, unprocessed_df)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, sampler=sampler, pin_memory=True, num_workers=4)

        log_memory_usage(rank)

        print("Starting text generation...")
        corrected_texts, detailed_outputs = generate_text(model, dataloader, device, tokenizer, df, output_path)
        print("Text generation completed")

        log_memory_usage(rank)

        if rank == 0:
            df['model_corrected'] = corrected_texts
            df.to_csv(output_path, index=False)
            print(f"Final output saved to {output_path}")

            # Save detailed outputs for further analysis
            detailed_output_path = output_path.replace('.csv', '_detailed_outputs.json')
            with open(detailed_output_path, 'w') as f:
                json.dump(detailed_outputs, f, indent=4)
            print(f"Detailed outputs saved to {detailed_output_path}")
    except Exception as e:
        print(f"An error occurred on rank {rank}: {str(e)}")

def main():
    print("Starting main function")
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50, help='Number of rows to process')
    parser.add_argument('--model_path', type=str, default='/scratch4/lhyman6/OCR/work/tuning_results_long/checkpoint-15500', help='Path to the model')
    parser.add_argument('--data_path', type=str, default='/data/lhyman6/OCR/1919_ocr_loaded_sorted.csv', help='Path to the data')
    parser.add_argument('--output_path', type=str, default='/data/lhyman6/OCR/debug/1919_debug2.csv', help='Path to the output')

    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    master_port = int(os.environ.get('MASTER_PORT', find_free_port()))
    print(f"World size: {world_size}, Master port: {master_port}")

    mp.spawn(main_worker, args=(world_size, args, args.model_path, args.data_path, args.output_path, master_port), nprocs=world_size, join=True)

if __name__ == "__main__":
    print("Executing main")
    main()
    print("Main execution finished")
