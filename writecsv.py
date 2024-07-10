# Date: 2024-07-10
# Purpose: Extract data from text files, clean and reformat it to handle irregular line breaks,
# ensure consistency for training an LLM, and count correctly formatted lines.

import os
import csv

# Define the directory containing the text files
input_directory = '/data/lhyman6/data/images/'
# Path to the cleaned output CSV file
output_csv = 'cleaned_output4.csv'
# Path to the error log file
error_log = 'error_log.txt'

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
        
        # Return data if both printed and legible are true
        if printed and legible:
            return ocr_text
        else:
            return None
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

# List to store rows for the CSV
rows = []
errors = []
skipped_count = 0

# Iterate over files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_directory, filename)
        try:
            ocr_text = extract_data(file_path)
            if ocr_text:
                # Get the ID from the filename (stripping to the jpg portion)
                file_id = filename.replace('.txt', '')
                rows.append([file_id, ocr_text])
            else:
                skipped_count += 1
        except Exception as e:
            errors.append(str(e))

# Sort rows by the file ID
rows.sort(key=lambda x: x[0])

# Write cleaned data to the CSV file
def clean_and_write_csv(rows, output_path):
    correctly_formatted_count = 0

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['ID', 'ocr'])

        # Initialize a variable to hold the current entry
        current_entry = None

        for row in rows:
            if len(row) == 2:
                if current_entry:
                    # Join the previous entry with newlines replaced by \n
                    full_row = current_entry[1].replace('\n', '\\n').strip()
                    csvwriter.writerow([current_entry[0], full_row])
                    correctly_formatted_count += 1
                # Add the new entry
                current_entry = row
            else:
                # Append the current line to the previous entry
                current_entry[1] += ' ' + ' '.join(row).strip()

        # Write the last entry if exists
        if current_entry:
            full_row = current_entry[1].replace('\n', '\\n').strip()
            csvwriter.writerow([current_entry[0], full_row])
            correctly_formatted_count += 1

    return correctly_formatted_count

correctly_formatted_count = clean_and_write_csv(rows, output_csv)

# Write errors to the log file
if errors:
    with open(error_log, 'w', encoding='utf-8') as log_file:
        log_file.write('\n'.join(errors))

# Output summary
print(f'Data successfully written to {output_csv}')
print(f'Number of files skipped: {skipped_count}')
print(f'Number of errors: {len(errors)}')
if errors:
    print(f'Error details can be found in {error_log}')
print(f"Number of correctly formatted lines: {correctly_formatted_count}")
