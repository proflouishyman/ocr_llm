# Date and time of creation: 2024-07-10
# Purpose: Tokenize string data from a CSV file, split into training, validation, and testing sets, and save for later training in specified subfolders

import pandas as pd
from transformers import BartTokenizer
import torch
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*is*', category=SyntaxWarning)

def load_data(file_path, target_column, size=None):
    # Load the data from the CSV file in order
    print(f"Loading data from {file_path} for target column '{target_column}'...")
    data = pd.read_csv(file_path)
    if size:
        data = data.iloc[:size]  # Select the first 'size' rows
    # Ensure all values are strings
    data['pyte'] = data['pyte'].astype(str)
    data[target_column] = data[target_column].astype(str)
    return data['pyte'], data[target_column]

def tokenize_function(examples, tokenizer, max_length=1024):
    inputs = tokenizer(examples['input'], max_length=max_length, truncation=True, padding='max_length')
    targets = tokenizer(examples['target'], max_length=max_length, truncation=True, padding='max_length')
    inputs['labels'] = targets['input_ids']
    return inputs

def save_dataset(dataset, base_folder, split_name):
    folder_path = os.path.join(base_folder, split_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataset.save_to_disk(folder_path)
    print(f"Dataset saved to {folder_path}")

def split_and_save_data(dataset, base_folder):
    print("Splitting data into training, validation, and testing sets...")
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
    
    datasets = DatasetDict({
        'train': train_test_split['train'],
        'val': test_valid_split['train'],
        'test': test_valid_split['test']
    })

    for split in ['train', 'val', 'test']:
        save_dataset(datasets[split], base_folder, split)

def main(file_path, target_column, sizes, base_folder):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    for size in sizes:
        print(f"\nProcessing size {size} for target column '{target_column}'")
        inputs, targets = load_data(file_path, target_column, size=size)
        
        data = {'input': inputs, 'target': targets}
        dataset = Dataset.from_dict(data)

        print("Starting tokenization...")
        tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        
        folder_name = f"{target_column}_{size}"
        base_folder_path = os.path.join(base_folder, folder_name)
        split_and_save_data(tokenized_dataset, base_folder_path)

if __name__ == "__main__":
    file_path = '/data/lhyman6/OCR/scripts/ocr_llm/complete_bart_training_data.csv'  # replace with your file path
    base_folder = '/scratch4/lhyman6/OCR/OCR/ocr_llm/work'  # base folder to save tokenized data
    sizes = [100, 1000, 10000]

    for target_column in ['gold', 'silver']:
        main(file_path, target_column, sizes, base_folder)
