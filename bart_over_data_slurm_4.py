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

class TextDataset(Dataset):
    def __init__(self, tokenizer, df, max_length=512):
        self.data = []
        self.original_indices = []
        for i, text in enumerate(df['ocr'].tolist()):
            parts = self.split_text(text, max_length, tokenizer)
            for part in parts:
                self.data.append(tokenizer(part, truncation=True, padding='max_length', max_length=max_length))
                self.original_indices.append(i)

    def split_text(self, text, max_length, tokenizer):
        tokens = tokenizer.tokenize(text)
        if len(tokens) <= max_length:
            return [text]
        parts = []
        for i in range(0, len(tokens), max_length):
            parts.append(tokenizer.convert_tokens_to_string(tokens[i:i+max_length]))
        return parts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.data[idx].items()}
        item['original_index'] = self.original_indices[idx]
        return item

    def __len__(self):
        return len(self.data)

def load_model(model_path, device):
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    return model, tokenizer

def generate_text(model, dataloader, device, tokenizer, df, output_path, save_interval=10):
    model.eval()
    corrected_texts = ["" for _ in range(len(df))]

    batch_results = {}

    for batch_num, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        original_indices = batch['original_index'].tolist()
        max_length = input_ids.size(1) + 3

        with torch.no_grad():
            with autocast():
                outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=5, early_stopping=True)
            batch_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            for idx, text in zip(original_indices, batch_texts):
                if idx in batch_results:
                    batch_results[idx].append(text)
                else:
                    batch_results[idx] = [text]

        if batch_num % save_interval == 0 and dist.get_rank() == 0:
            for idx, texts in batch_results.items():
                corrected_texts[idx] = " ".join(texts)
            df['model_corrected'] = corrected_texts
            df.to_csv(output_path, index=False)
            print(f"Saved progress at batch {batch_num}")

        # Clear cache and run garbage collection to free up memory
        torch.cuda.empty_cache()
        gc.collect()

    for idx, texts in batch_results.items():
        corrected_texts[idx] = " ".join(texts)
    return corrected_texts

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup_distributed_environment(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    return torch.device("cuda", rank)

def main_worker(rank, world_size, args, model_path, data_path, output_path, master_port):
    device = setup_distributed_environment(rank, world_size, master_port)

    model, tokenizer = load_model(model_path, device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    df = pd.read_csv(data_path)
    if 'model_corrected' in df.columns:
        df['model_corrected'] = df['model_corrected'].fillna("")
    else:
        df['model_corrected'] = ""

    if args.limit is not None:
        df = df.head(args.limit)

    # Filter out already processed data
    unprocessed_df = df[df['model_corrected'] == ""]
    
    if unprocessed_df.empty:
        print(f"All rows in {data_path} have already been processed.")
        return

    dataset = TextDataset(tokenizer, unprocessed_df)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, sampler=sampler, pin_memory=True, num_workers=4)

    corrected_texts = generate_text(model, dataloader, device, tokenizer, df, output_path)

    if rank == 0:
        df['model_corrected'] = corrected_texts
        df.to_csv(output_path, index=False)
        print(f"Final output saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--model_path', type=str, default='/scratch4/lhyman6/OCR/work/tuning_results_long/checkpoint-15500', help='Path to the model')
    parser.add_argument('--data_path', type=str, default='/data/lhyman6/OCR/1919_ocr_loaded.csv', help='Path to the data')
    parser.add_argument('--output_path', type=str, default='/data/lhyman6/OCR/1919_15500_data_ocr_loaded_4.csv', help='Path to the output')

    args = parser.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    master_port = int(os.environ.get('MASTER_PORT', find_free_port()))
    
    mp.spawn(main_worker, args=(world_size, args, args.model_path, args.data_path, args.output_path, master_port), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
