# 2024-05-15
# Script to sort a CSV file by the "id" column
# This script reads a CSV file, sorts it by the "id" column, and writes the sorted data to a new CSV file.

import pandas as pd

# Read the CSV file
input_file = '/data/lhyman6/OCR/1919_ocr_loaded.csv'  # Replace with the path to your input CSV file
output_file = '/data/lhyman6/OCR/1919_ocr_loaded_sorted.csv'  # Replace with the desired path for the sorted output CSV file

# Load the CSV data into a DataFrame, ensuring the 'id' column is read as a string
df = pd.read_csv(input_file, dtype={'id': str})

# Sort the DataFrame by the "id" column
df_sorted = df.sort_values(by='id')

# Write the sorted DataFrame to a new CSV file
df_sorted.to_csv(output_file, index=False)

print(f"Sorted CSV file saved as {output_file}")
