# 2024-07-10
# This script loads the merged CSV file with gold, silver, and pyte columns,
# tokenizes the text in these columns using the BART tokenizer, calculates token lengths,
# displays statistics, plots distributions, and counts rows with tokens exceeding the BART limit.

import pandas as pd
import matplotlib.pyplot as plt
from transformers import BartTokenizer
from tqdm import tqdm

# Load BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# Load data
csv_file_path = 'complete_bart_training_data.csv'
print(f"Loading data from {csv_file_path}...")
data = pd.read_csv(csv_file_path)
print("Data loaded successfully.")

# Ensure all data in 'gold', 'silver', and 'pyte' are strings, replacing NaNs with an empty string
print("Converting 'gold', 'silver', and 'pyte' columns to strings and filling NaNs with empty strings...")
data['gold'] = data['gold'].fillna('').astype(str)
data['silver'] = data['silver'].fillna('').astype(str)
data['pyte'] = data['pyte'].fillna('').astype(str)

# Tokenize and calculate token lengths using tqdm for progress indication
print("Tokenizing 'gold' column and calculating token lengths...")
tqdm.pandas(desc="Tokenizing Gold")
data['gold_token_length'] = data['gold'].progress_apply(lambda x: len(tokenizer.tokenize(x)))

print("Tokenizing 'silver' column and calculating token lengths...")
tqdm.pandas(desc="Tokenizing Silver")
data['silver_token_length'] = data['silver'].progress_apply(lambda x: len(tokenizer.tokenize(x)))

print("Tokenizing 'pyte' column and calculating token lengths...")
tqdm.pandas(desc="Tokenizing Pyte")
data['pyte_token_length'] = data['pyte'].progress_apply(lambda x: len(tokenizer.tokenize(x)))

# Display some statistics
print("Gold Token Lengths:")
print(data['gold_token_length'].describe())

print("\nSilver Token Lengths:")
print(data['silver_token_length'].describe())

print("\nPyte Token Lengths:")
print(data['pyte_token_length'].describe())

# Plot the distribution of token lengths for Gold, Silver, and Pyte
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
plt.hist(data['gold_token_length'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Gold Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
plt.hist(data['silver_token_length'], bins=50, color='green', alpha=0.7)
plt.title('Distribution of Silver Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
plt.hist(data['pyte_token_length'], bins=50, color='red', alpha=0.7)
plt.title('Distribution of Pyte Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate how many rows have a token length greater than 1024 (which is the limit for BART token size)
num_rows_exceeding_gold = data[data['gold_token_length'] > 1024].shape[0]
num_rows_exceeding_silver = data[data['silver_token_length'] > 1024].shape[0]
num_rows_exceeding_pyte = data[data['pyte_token_length'] > 1024].shape[0]

print(f"Number of rows with more than 1024 tokens in 'gold': {num_rows_exceeding_gold}")
print(f"Number of rows with more than 1024 tokens in 'silver': {num_rows_exceeding_silver}")
print(f"Number of rows with more than 1024 tokens in 'pyte': {num_rows_exceeding_pyte}")
