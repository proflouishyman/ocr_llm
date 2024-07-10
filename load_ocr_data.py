# Created on: 2024-05-10
# Purpose: This script loads every text file in a directory into a DataFrame, assigns IDs using filenames, puts the text into a column labeled 'ocr', and saves the data as a CSV file. It includes progress indicators.

import os
import pandas as pd
from tqdm import tqdm

def load_text_files(directory):
    """
    Load every text file in a directory into a DataFrame, ensuring all text is treated as string.

    Args:
    - directory: The directory path containing the text files.

    Returns:
    - df: DataFrame containing the loaded text files, with 'ocr' as strings.
    """
    file_texts = []
    file_ids = []

    # Get list of text files
    text_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    # Use tqdm for progress indication
    for filename in tqdm(text_files, desc="Loading text files"):
        # Read the text from the file
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            file_texts.append(text)
            # Create an ID using the filename (dropping the .txt extension)
            file_id = os.path.splitext(filename)[0]
            file_ids.append(file_id)

    # Create a DataFrame and ensure 'ocr' column is treated as strings
    df = pd.DataFrame({'id': file_ids, 'ocr': file_texts})
    df['ocr'] = df['ocr'].astype(str) 
    return df


def save_as_csv(dataframe, filename):
    """
    Save DataFrame as a CSV file.

    Args:
    - dataframe: The DataFrame to be saved.
    - filename: The filename (with path) for the CSV file.
    """
    dataframe.to_csv(filename, index=False)

# Example usage:
directory_path = '/scratch4/lhyman6/1919/1919/images_pyte'
csv_filename = '/data/lhyman6/OCR/1919_ocr_loaded.csv'
dataframe = load_text_files(directory_path)
save_as_csv(dataframe, csv_filename)
