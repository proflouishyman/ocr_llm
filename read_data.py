# Date and time of creation: [current date and time]
# Purpose of the program: Read in a CSV file, drop unnecessary columns, add OCR data from text files if available, and save as a new CSV file

import pandas as pd
import os
from tqdm import tqdm

# Function to read OCR data from text files
def read_ocr_from_file(asset_name):
    ocr_file_path = f"//scratch4/lhyman6/OCR/work/text/{asset_name}.txt"
    if os.path.isfile(ocr_file_path):
        with open(ocr_file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return None

def main():
    # File paths
    input_csv_path = "/data/lhyman6/OCR/data/bythepeople1.csv"
    output_csv_path = "/data/lhyman6/OCR/data/training_data.csv"

    # Read the CSV file
    print("Reading the CSV file...")
    df = pd.read_csv(input_csv_path)

    # Rename columns
    df = df.rename(columns={'Asset': 'id', 'Transcription': 'corrected', 'OCR': 'ocr'})

    # Set the initial value of the 'ocr' column to blank
    df['ocr'] = ""

    # Add OCR data from text files to the DataFrame
    print("Adding OCR data to the DataFrame...")
    with tqdm(total=len(df['id'])) as pbar:
        df['ocr'] = df['id'].apply(lambda x: read_ocr_from_file(x))
        pbar.update()

    # Keep only the necessary columns
    df = df[['id', 'ocr', 'corrected']]


    # Count the number of values read in and how many remain blank
    num_values_read = df['ocr'].count()
    num_blank_values = df['ocr'].isnull().sum()

    
    #convert to strings for tokenization
    df['ocr'] = df['ocr'].fillna('').astype(str)
    df['corrected'] = df['corrected'].fillna('').astype(str)


    # Print the head of the DataFrame
    print("Printing the head of the DataFrame:")
    print(df.head())

    # Save the DataFrame to a new CSV file
    print("Saving the modified DataFrame to a new CSV file...")
    df.to_csv(output_csv_path, index=False)

    print(f"CSV file with OCR data saved at: {output_csv_path}")
    print(f"Number of values read in: {num_values_read}")
    print(f"Number of blank values remaining: {num_blank_values}")

if __name__ == "__main__":
    main()
