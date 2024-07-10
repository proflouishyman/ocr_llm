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
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import socket

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

def generate_text_with_logging(rank, world_size, model, dataloader, device, tokenizer, df, output_dir):
    model.eval()
    corrected_texts = {}
    reasons = {}  # Dictionary to store reasons for outputs
    detailed_outputs = {}  # Dictionary to store detailed outputs for each index
    status_updates = []  # List to store status updates

    for batch_num, batch in enumerate(tqdm(dataloader, desc=f"Processing batches on GPU {rank}")):
        logger.debug(f"Processing batch {batch_num} on GPU {rank}")
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
                logger.debug(f"Index {idx}, Part {part_num}: Text generated on GPU {rank}")
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

            logger.debug(f"Batch {batch_num} processed with indices: {original_indices} on GPU {rank}")

        # Clear cache and run garbage collection to free up memory
        torch.cuda.empty_cache()
        gc.collect()

    # Write the results for this GPU to a separate CSV file
    output_path = os.path.join(output_dir, f'output_rank_{rank}.csv')
    df['original_index'] = df.index  # Ensure original_index is preserved
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
    logger.info(f"Output for GPU {rank} saved to {output_path}")

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

def cleanup():
    dist.destroy_process_group()

def process_ocr_data(limit, model_path, data_path, output_dir):
    logger.info("Starting process_ocr_data function")

    def main_worker(rank, world_size, master_port):
        setup_distributed_environment(rank, world_size, master_port)
        logger.info(f"Starting main_worker on rank {rank}")

        device = torch.device(f"cuda:{rank}")
        logger.info(f"Using device: {device}")

        model, tokenizer = load_model(model_path, device)
        model = DDP(model, device_ids=[rank])

        logger.info(f"Loading data on rank {rank}...")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {len(df)} rows on rank {rank}")

        if limit is not None:
            df = df.head(limit)
            logger.info(f"Limiting data to {limit} rows on rank {rank}")

        if 'model_corrected' in df.columns:
            df['model_corrected'] = df['model_corrected'].fillna("")
        else:
            df['model_corrected'] = ""

        # Split data evenly among GPUs
        split_size = len(df) // world_size
        start_idx = rank * split_size
        end_idx = (rank + 1) * split_size if rank != world_size - 1 else len(df)
        df_split = df.iloc[start_idx:end_idx].reset_index(drop=True)
        logger.info(f"Data split: rows {start_idx} to {end_idx} for GPU {rank}")

        dataset = TextDataset(tokenizer, df_split)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

        logger.info(f"Starting text generation on rank {rank}...")
        generate_text_with_logging(rank, world_size, model, dataloader, device, tokenizer, df_split, output_dir)
        logger.info(f"Text generation completed on rank {rank}")

        cleanup()

    world_size = torch.cuda.device_count()
    logger.info(f"World size: {world_size}")
    master_port = find_free_port()

    mp.spawn(main_worker, args=(world_size, master_port), nprocs=world_size, join=True)

    # Combine the outputs from all GPUs
    combined_df = pd.concat([pd.read_csv(os.path.join(output_dir, f'output_rank_{rank}.csv')) for rank in range(world_size)])
    combined_df = combined_df.sort_values(by=['original_index']).reset_index(drop=True)

    # Save the combined and sorted output
    combined_output_path = os.path.join(output_dir, 'combined_output.csv')
    combined_df.to_csv(combined_output_path, index=False)
    logger.info(f"Combined output saved to {combined_output_path}")

if __name__ == "__main__":
    logger.info("Executing main")

    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50, help='Number of rows to process')
    parser.add_argument('--model_path', type=str, default='/scratch4/lhyman6/OCR/work/tuning_results_long/checkpoint-15500', help='Path to the model')
    parser.add_argument('--data_path', type=str, default='/data/lhyman6/OCR/1919_ocr_loaded_sorted.csv', help='Path to the data')
    parser.add_argument('--output_dir', type=str, default='/data/lhyman6/OCR/debug/', help='Directory for output files')

    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")

    process_ocr_data(args.limit, args.model_path, args.data_path, args.output_dir)

    logger.info("Main execution finished")
