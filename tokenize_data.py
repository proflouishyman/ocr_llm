# Date and time of creation
# 2024-07-10

# Purpose: Tokenize string data from a CSV file, split into training, validation, and testing sets, and save for later training in specified subfolders
#totally untested


import pandas as pd
from transformers import BartTokenizer
import torch
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_data(file_path, target_column, size=None):
    # Load the data from the CSV file
    print(f"Loading data from {file_path} for target column '{target_column}'...")
    data = pd.read_csv(file_path)
    if size:
        data = data.sample(n=size, random_state=42)
    return data['pyte'], data[target_column]

def tokenize_data(tokenizer, inputs, targets, max_length=1024):
    print("Tokenizing data...")
    input_ids = []
    target_ids = []
    for input_text, target_text in tqdm(zip(inputs, targets), total=len(inputs)):
        input_id = tokenizer.encode(input_text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        target_id = tokenizer.encode(target_text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        input_ids.append(input_id)
        target_ids.append(target_id)
    input_ids = torch.cat(input_ids)
    target_ids = torch.cat(target_ids)
    return input_ids, target_ids

def save_tokenized_data(input_ids, target_ids, folder_path):
    print(f"Saving tokenized data to {folder_path}...")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(input_ids, os.path.join(folder_path, 'inputs.pt'))
    torch.save(target_ids, os.path.join(folder_path, 'targets.pt'))
    print(f"Data saved to {folder_path}")

def split_and_save_data(input_ids, target_ids, base_folder):
    print("Splitting data into training, validation, and testing sets...")
    train_inputs, temp_inputs, train_targets, temp_targets = train_test_split(
        input_ids, target_ids, test_size=0.2, random_state=42
    )
    val_inputs, test_inputs, val_targets, test_targets = train_test_split(
        temp_inputs, temp_targets, test_size=0.5, random_state=42
    )
    
    splits = {'train': (train_inputs, train_targets), 
              'val': (val_inputs, val_targets), 
              'test': (test_inputs, test_targets)}
    
    for split, (split_inputs, split_targets) in splits.items():
        folder_path = os.path.join(base_folder, split)
        save_tokenized_data(split_inputs, split_targets, folder_path)

def main(file_path, target_column, sizes, base_folder):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    for size in sizes:
        print(f"\nProcessing size {size} for target column '{target_column}'")
        inputs, targets = load_data(file_path, target_column, size=size)
        
        # Use tqdm for progress indication during tokenization
        print("Starting tokenization...")
        input_ids, target_ids = tokenize_data(tokenizer, inputs, targets, max_length=1024)
        
        folder_name = f"{target_column}_{size}"
        base_folder_path = os.path.join(base_folder, folder_name)
        split_and_save_data(input_ids, target_ids, base_folder_path)

if __name__ == "__main__":
    file_path = '/scratch4/lhyman6/OCR/ocr_llm/complete_bart_training_data.csv'  # replace with your file path
    base_folder = '/scratch4/lhyman6/OCR/ocr_llm/tokenized_data'  # base folder to save tokenized data
    sizes = [100, 1000, 10000]

    for target_column in ['gold', 'silver']:
        main(file_path, target_column, sizes, base_folder)
