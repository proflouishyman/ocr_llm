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

def validate_and_clean_data(dataframe):
    """
    Perform checks on DataFrame and clean data by removing empty strings in 'ocr' column.

    Args:
    - dataframe: DataFrame to validate and clean.

    Returns:
    - dataframe: Cleaned DataFrame.
    - num_empty_strings: Number of rows with empty strings in 'ocr' column.
    - num_non_strings: Number of rows with non-string entries in 'ocr' column.
    - empty_string_filenames: List of filenames with empty strings in 'ocr' column.
    - non_string_filenames: List of filenames with non-string entries in 'ocr' column.
    """
    # Check for non-string entries and convert to string if any
    num_non_strings = dataframe['ocr'].apply(lambda x: not isinstance(x, str)).sum()
    non_string_filenames = dataframe.loc[dataframe['ocr'].apply(lambda x: not isinstance(x, str)), 'id'].tolist()
    if num_non_strings > 0:
        print(f"Found {num_non_strings} non-string entries in 'ocr' column. Converting to strings.")

    # Remove rows with empty strings
    num_empty_strings = (dataframe['ocr'].str.strip() == '').sum()
    empty_string_filenames = dataframe.loc[dataframe['ocr'].str.strip() == '', 'id'].tolist()
    if num_empty_strings > 0:
        print(f"Found {num_empty_strings} rows with empty strings in 'ocr' column. Dropping them.")

    dataframe = dataframe[dataframe['ocr'].str.strip() != '']
    dataframe.loc[:, 'ocr'] = dataframe['ocr'].astype(str)

    return dataframe, num_empty_strings, num_non_strings, empty_string_filenames, non_string_filenames

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
dataframe, num_empty_strings, num_non_strings, empty_string_filenames, non_string_filenames = validate_and_clean_data(dataframe)
save_as_csv(dataframe, csv_filename)
print("CSV file saved successfully.")

print(f"Number of empty strings: {num_empty_strings}")
print(f"Empty string filenames: {empty_string_filenames}")
print(f"Number of non-string entries: {num_non_strings}")
print(f"Non-string filenames: {non_string_filenames}")
