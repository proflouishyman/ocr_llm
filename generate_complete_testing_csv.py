# 2024-07-13
# This script updates a CSV file with OCR content from .jpg.txt and .txt.pyte files, extracting specific parts
# and adding new columns for is_printed and is_legible.

import pandas as pd
import os
from tqdm import tqdm

# Define file paths
csv_file = '/data/lhyman6/OCR/data/gompers_corrections/by-the-people-afl-campaign-20240506 - 2of2.csv'
output_csv = '/data/lhyman6/OCR/scripts/ocr_llm/complete_testing_csv.csv'
data_directory = '/data/lhyman6/OCR/scripts/data/second_images'

# Define template columns
template_columns = ['id', 'transcription', 'pyte_ocr', 'chatgpt_ocr', 'is_printed', 'is_legible', 'BART_untuned',
                    'BART_gold_100', 'BART_gold_1000', 'BART_gold_10000', 'BART_silver_100', 'BART_silver_1000',
                    'BART_silver_10000', 'LLAVA_untuned', 'LLAVA_gold_100', 'LLAVA_gold_1000', 'LLAVA_gold_10000',
                    'LLAVA_silver_100', 'LLAVA_silver_1000', 'LLAVA_silver_10000']

# Print initial information
print(f'Reading CSV file from: {csv_file}')
print(f'Saving rearranged and templated CSV to: {output_csv}')

# Read the CSV into a DataFrame and specify dtype as str to avoid type issues
df = pd.read_csv(csv_file, dtype=str)

# Keep only 'Asset' and 'Transcription' columns and rename them
df = df[['Asset', 'Transcription']]
df.columns = ['id', 'transcription']

# Create a template DataFrame with all columns and initialize with blank data
template_df = pd.DataFrame(columns=template_columns)
template_df = pd.concat([df, template_df], axis=0)

# Ensure all columns in template are present and fill missing with empty strings
final_df = template_df.reindex(columns=template_columns).fillna('')

# Save the modified DataFrame back to CSV
final_df.to_csv(output_csv, index=False)
print('Initial CSV file successfully rearranged and templated.')

# Read the newly created templated CSV into a DataFrame for updating
df = pd.read_csv(output_csv, dtype=str)

# Function to extract data from a file
def extract_data(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Split the content by markers
        ocr_text = extract_field(content, "OCR Text:", "Summary:")
        summary = extract_field(content, "Summary:", "Date:")
        date = extract_field(content, "Date:", "Printed:")
        printed = "True" in extract_field(content, "Printed:", "Legible:")
        legible = "True" in extract_field(content, "Legible:", None)

        return ocr_text, printed, legible
    except Exception as e:
        raise Exception(f"Error processing file {file_path}: {str(e)}\nContent: {content}")

# Helper function to extract field content between markers
def extract_field(content, start_marker, end_marker):
    try:
        start_idx = content.index(start_marker) + len(start_marker)
        if end_marker:
            end_idx = content.index(end_marker, start_idx)
            return content[start_idx:end_idx].strip()
        else:
            return content[start_idx:].strip()
    except ValueError:
        return ""

# Variables to track missing files
missing_files_count = 0
missing_files_examples = []

# Iterate over the DataFrame using tqdm for progress tracking
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Updating CSV"):
    id_value = row['id']
    pyte_file = f"{data_directory}/{id_value}.txt.pyte"  # Construct the file path for the .jpg.txt.pyte file
    txt_file = f"{data_directory}/{id_value}.jpg.txt"  # Construct the file path for the .jpg.txt file
    
    # Check if the .jpg.txt.pyte file exists and read the content
    if os.path.exists(pyte_file):
        with open(pyte_file, 'r') as file:
            pyte_content = file.read()
        df.at[index, 'pyte_ocr'] = pyte_content  # Update the DataFrame with pyte_ocr content
    else:
        missing_files_count += 1
        if len(missing_files_examples) < 5:  # Collect up to 5 examples to display
            missing_files_examples.append(id_value)
    
    # Check if the .jpg.txt file exists and read the content
    if os.path.exists(txt_file):
        try:
            ocr_text, is_printed, is_legible = extract_data(txt_file)
            df.at[index, 'chatgpt_ocr'] = ocr_text  # Update the DataFrame with chatgpt_ocr content
            df.at[index, 'is_printed'] = is_printed  # Update the DataFrame with is_printed content
            df.at[index, 'is_legible'] = is_legible  # Update the DataFrame with is_legible content
        except Exception as e:
            print(f"Error processing file {txt_file}: {str(e)}")
    else:
        missing_files_count += 1
        if len(missing_files_examples) < 5:  # Collect up to 5 examples to display
            missing_files_examples.append(id_value)

# Save the updated DataFrame back to CSV
df.to_csv(output_csv, index=False)
print('CSV file has been updated successfully with pyte_ocr, chatgpt_ocr, is_printed, and is_legible content.')

# Report missing files
if missing_files_count > 0:
    print(f"Failed to find {missing_files_count} '.jpg.txt' files for some entries.")
    print("Examples of missing files:", missing_files_examples)
else:
    print("All expected '.jpg.txt' files were found.")

# Check for any remaining blank entries in 'pyte_ocr' and 'chatgpt_ocr'
blank_pyte_entries = df[df['pyte_ocr'].isnull() | (df['pyte_ocr'] == '')]
blank_chatgpt_entries = df[df['chatgpt_ocr'].isnull() | (df['chatgpt_ocr'] == '')]
print(f"Number of blank entries remaining in 'pyte_ocr': {len(blank_pyte_entries)}")
print(f"Number of blank entries remaining in 'chatgpt_ocr': {len(blank_chatgpt_entries)}")

if not blank_pyte_entries.empty:
    print("Some entries are still blank in 'pyte_ocr'. Here are a few examples:")
    print(blank_pyte_entries.head())  # Show a few examples of rows with blank pyte_ocr

if not blank_chatgpt_entries.empty:
    print("Some entries are still blank in 'chatgpt_ocr'. Here are a few examples:")
    print(blank_chatgpt_entries.head())  # Show a few examples of rows with blank chatgpt_ocr

print(df.head())
print(df.columns.tolist())
