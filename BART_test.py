import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler
import random

# Function to load the model
def load_model(model_path):
    print("Loading model...")
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    print("Model loaded successfully.")
    return model, tokenizer

# Function to prepare data loader
def prepare_data(data_path, num_samples=3):
    print("Preparing data...")
    df = pd.read_csv(data_path)
    df_sample = df.sample(n=num_samples, random_state=42)  # Randomly pick samples
    print("Data prepared.")
    return df_sample

# Function to generate corrections using the model
def generate_text(model, tokenizer, texts, device):
    print("Generating corrections...")
    model.to(device)
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
    corrected_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print("Corrections generated.")
    return corrected_texts

# Main function to run the script
def main():
    print("Initializing...")
    model_path = '/scratch4/lhyman6/OCR/work/tuning_results_long/checkpoint-15500'
    data_path = '/data/lhyman6/OCR/data/training_data.csv'
    output_path = '/data/lhyman6/OCR/data/corrected_data.csv'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU for computation.")
    else:
        print("CUDA not detected. Using CPU for computation.")
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)
    
    # Prepare data
    data = prepare_data(data_path)
    
    # Generate corrections
    corrected_texts = generate_text(model, tokenizer, data['ocr'].tolist(), device)
    
    # Add corrected texts to the DataFrame and save to CSV
    data['model_corrected'] = corrected_texts
    data.to_csv(output_path, index=False)

    # Print out the comparisons
    for index, row in data.iterrows():
        print(f"Original: {row['ocr']}")
        print(f"Ground Truth: {row['corrected']}")
        print(f"Model Corrected: {row['model_corrected']}")
        print("-------------------------------")

if __name__ == "__main__":
    main()
