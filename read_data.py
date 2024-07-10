# 2024-07-10
# This script reads two CSV files, processes the first to keep only Asset and Transcription columns,
# renames them to id and gold, and merges it with the second CSV file after removing the .jpg extension
# from the ID column. The resulting DataFrame contains the gold and silver columns and provides statistics
# on the merge operation.

import pandas as pd

# Paths to the CSV files
gold_csv_path = '/data/lhyman6/OCR/data/gompers_corrections/bythepeople1.csv'
silver_csv_path = '/data/lhyman6/OCR/scripts/ocr_llm/silver_ocr_data.csv'
merged_csv_path = 'complete_bart_training_data.csv'

# Read the first CSV file (gold data)
print("Reading the gold CSV file...")
gold_df = pd.read_csv(gold_csv_path)
print("Gold CSV file read successfully.")

# Keep only the Asset and Transcription columns
print("Filtering the gold DataFrame to keep only the Asset and Transcription columns...")
gold_df_filtered = gold_df[['Asset', 'Transcription']]

# Rename the columns to id and gold
print("Renaming the columns to id and gold...")
gold_df_filtered.rename(columns={'Asset': 'id', 'Transcription': 'gold'}, inplace=True)

# Convert the 'gold' column to strings for tokenization
print("Converting the 'gold' column to strings for tokenization...")
gold_df_filtered['gold'] = gold_df_filtered['gold'].fillna('').astype(str)

# Print the head of the gold DataFrame
print("Printing the head of the gold DataFrame:")
print(gold_df_filtered.head())

# Read the second CSV file (silver data)
print("Reading the silver CSV file...")
silver_df = pd.read_csv(silver_csv_path)
print("Silver CSV file read successfully.")

# Remove the .jpg extension from the ID column
print("Removing the .jpg extension from the ID column...")
silver_df['ID'] = silver_df['ID'].str.replace('.jpg', '')

# Sort the silver DataFrame
print("Sorting the silver DataFrame...")
silver_df = silver_df.sort_values(by='ID')

# Print the head of the silver DataFrame
print("Printing the head of the silver DataFrame:")
print(silver_df.head())

# Merge the DataFrames on the id column
print("Merging the gold and silver DataFrames...")
merged_df = pd.merge(gold_df_filtered, silver_df, how='left', left_on='id', right_on='ID')

# Rename the ocr column to silver
print("Renaming the 'ocr' column to 'silver'...")
merged_df.rename(columns={'ocr': 'silver'}, inplace=True)

# Drop the extra ID column from the silver DataFrame
print("Dropping the extra 'ID' column from the merged DataFrame...")
merged_df.drop(columns=['ID'], inplace=True)

# Identify and drop rows where the merge failed (silver column is NaN)
print("Dropping rows with failed merges...")
initial_row_count = len(merged_df)
merged_df.dropna(subset=['silver'], inplace=True)
rows_dropped = initial_row_count - len(merged_df)

# Save the merged DataFrame as a new CSV file
print(f"Saving the merged DataFrame to {merged_csv_path}...")
merged_df.to_csv(merged_csv_path, index=False)
print("Merged DataFrame saved successfully.")

# Print the head of the merged DataFrame
print("Printing the head of the merged DataFrame:")
print(merged_df.head())

# Provide statistics on the merge operation
total_rows_gold = len(gold_df_filtered)
total_rows_silver = len(silver_df)
total_rows_merged = len(merged_df)

print("\nMerge Operation Statistics:")
print(f"Total rows in gold DataFrame: {total_rows_gold}")
print(f"Total rows in silver DataFrame: {total_rows_silver}")
print(f"Total rows in merged DataFrame: {total_rows_merged}")
print(f"Number of rows dropped due to failed merges: {rows_dropped}")
print(f"Number of successful merges: {total_rows_merged}")
print(f"Merged DataFrame saved as: {merged_csv_path}")
