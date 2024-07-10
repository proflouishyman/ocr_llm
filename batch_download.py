# batch_download.py
# 2024-06-12
# Script to check the status of batches, download the output files if the batches are completed,
# extract each line from JSONL result files into individual JSON files named by custom_id,
# and extract the "content" field from each JSON file into text files next to the original image files.
# Revision July 10 2024

import os
import json
import time
from openai import OpenAI
import re

# Variables
API_KEY_FILE = "/data/lhyman6/api_key.txt"
BATCH_DIR = "batch"
COMPLETED_DIR = os.path.join(BATCH_DIR, "completed")
OUTPUT_DIR = "batch_return"
EXTRACTION_BASE_DIR = "batch_json_results"  # Base output directory for individual JSON files
RETRY_LIMIT = 5
RETRY_DELAY = 60  # in seconds

def read_api_key(api_key_file):
    """Read the API key from a file."""
    with open(api_key_file, "r") as file:
        return file.read().strip()

def get_batch_status(api_key, batch_id):
    """Get the status of a batch."""
    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)
    return batch

def download_file(api_key, file_id, output_dir, filename):
    """Download a file from OpenAI and save it to the specified directory with the given filename."""
    client = OpenAI(api_key=api_key)
    response = client.files.content(file_id)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file_path = os.path.join(output_dir, f"{filename}.jsonl")
    with open(output_file_path, "wb") as output_file:
        for chunk in response.iter_bytes():
            output_file.write(chunk)
    
    return output_file_path

def serialize_batch(batch):
    """Serialize the batch object into a JSON-serializable dictionary."""
    return {
        "id": batch.id,
        "object": batch.object,
        "endpoint": batch.endpoint,
        "errors": str(batch.errors) if batch.errors else None,
        "input_file_id": batch.input_file_id,
        "completion_window": batch.completion_window,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "created_at": batch.created_at,
        "in_progress_at": batch.in_progress_at,
        "expires_at": batch.expires_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "expired_at": batch.expired_at,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed
        },
        "metadata": batch.metadata
    }

def read_batch_ids(batch_dir):
    """Read batch IDs from filenames in the specified directory."""
    batch_ids = []
    for filename in os.listdir(batch_dir):
        if filename.endswith(".txt"):
            batch_id = filename.replace(".txt", "")
            batch_ids.append(batch_id)
    return batch_ids

def extract_json_lines(result_file, output_dir):
    """Extract each line from the result file into individual JSON files named by custom_id."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(result_file, "r") as file:
        for line in file:
            data = json.loads(line)
            custom_id = data.get("custom_id", "unknown_id")
            output_file_path = os.path.join(output_dir, f"{custom_id}.json")
            with open(output_file_path, "w") as output_file:
                json.dump(data, output_file, indent=2)
            print(f"Extracted {custom_id} to {output_file_path}")



def extract_content_to_text(json_dir):
    """Extract specific fields from JSON files and save as text files next to the original image files."""
    failed_files = []

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_file_path = os.path.join(json_dir, filename)
            with open(json_file_path, "r") as json_file:
                try:
                    data = json.load(json_file)
                    method_used = "JSON Parsing"
                    success_message = "and it was successful"
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON content for {json_file_path}. Using regex extraction instead.")
                    method_used = "Regex Extraction"
                    success_message = ""

                    # Set data to empty dict to prevent further errors
                    data = {}

                response_body = data.get('response', {}).get('body', {})
                choices = response_body.get('choices', [])

                if choices:
                    message_content = choices[0].get('message', {}).get('content', '')

                    # Remove the enclosing ```json and ``` markers
                    if message_content.startswith("```json") and message_content.endswith("```"):
                        message_content = message_content[7:-3].strip()

                    try:
                        message_json = json.loads(message_content)
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON content for {json_file_path}. Using regex extraction instead.")
                        method_used = "Regex Extraction"
                        success_message = ""

                        # Set message_json to empty dict to prevent further errors
                        message_json = {}

                    # Regex patterns to extract fields
                    ocr_text_pattern = r'"ocr_text": "([^"]+)"'
                    summary_pattern = r'"summary": "([^"]+)"'
                    date_pattern = r'"date": "([^"]+)"'
                    is_printed_pattern = r'"is_printed": (true|false)'
                    is_legible_pattern = r'"is_legible": (true|false)'

                    # Find matches using regex
                    ocr_text_match = re.search(ocr_text_pattern, message_content)
                    summary_match = re.search(summary_pattern, message_content)
                    date_match = re.search(date_pattern, message_content)
                    is_printed_match = re.search(is_printed_pattern, message_content)
                    is_legible_match = re.search(is_legible_pattern, message_content)

                    # Extract fields from regex matches or use defaults
                    ocr_text = ocr_text_match.group(1) if ocr_text_match else ""
                    summary = summary_match.group(1) if summary_match else ""
                    date = date_match.group(1) if date_match else "No Date Provided"
                    is_printed = is_printed_match.group(1) == "true" if is_printed_match else False
                    is_legible = is_legible_match.group(1) == "true" if is_legible_match else False

                    # Prepare content for output
                    content = (f"OCR Text: {ocr_text}\n"
                               f"Summary: {summary}\n"
                               f"Date: {date}\n"
                               f"Printed: {is_printed}\n"
                               f"Legible: {is_legible}")

                    custom_id = data.get("custom_id", "unknown_id")
                    original_path = custom_id.replace("|", os.sep)
                    original_dir = os.path.dirname(original_path)
                    original_filename = original_path.split(os.sep)[-1]

                    # Save the content to a text file
                    text_filename = f"{original_filename}.txt"
                    text_file_path = os.path.join(original_dir, text_filename)
                    os.makedirs(original_dir, exist_ok=True)
                    with open(text_file_path, "w") as text_file:
                        text_file.write(content)
                    
                    print(f"Extracted content of {custom_id} to {text_file_path} using {method_used}. {success_message}")
                else:
                    print(f"No valid choices found for {json_file_path}.")
                    failed_files.append(json_file_path)

            if not choices:
                failed_files.append(json_file_path)
    
    if failed_files:
        print("\nFiles that failed:")
        for file in failed_files:
            print(file)


def process_result_files(result_dir, extraction_base_dir, batch_id):
    """Process all JSONL files in the specified directory and extract them to individual JSON files."""
    extraction_output_dir = os.path.join(extraction_base_dir, batch_id)
    if not os.path.exists(extraction_output_dir):
        os.makedirs(extraction_output_dir)

    for filename in os.listdir(result_dir):
        if filename.endswith(".jsonl") and f"batch_{batch_id}" in filename:
            result_file_path = os.path.join(result_dir, filename)
            print(f"Processing file: {result_file_path}")
            extract_json_lines(result_file_path, extraction_output_dir)
            extract_content_to_text(extraction_output_dir)


def replace_escape_sequences(content):
    """Replace escape sequences with placeholders."""
    return content.replace('\\', '|')

def process_text_files(directory):
    """Process text files in the specified directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            if lines[0].strip() == "```json":
                lines = lines[1:]
            
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            
            # Delete everything until the first '{' in the entire content
            content = ''.join(lines)
            content = content[content.find('{'):]
            lines = content.splitlines(True)  # Split the content back into lines, keeping line breaks
            
            # Process only the first 5 lines for replacing escape sequences
            for i in range(min(5, len(lines))):
                lines[i] = replace_escape_sequences(lines[i])
            
            # Combine the lines back to a single string
            content = ''.join(lines)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)

def clean(directory):
    """Placeholder function for cleaning text files."""
    process_text_files(directory)

if __name__ == "__main__":
    print("Reading API key from file...")
    API_KEY = read_api_key(API_KEY_FILE)
    print("API key successfully read.")

    print("Reading batch IDs from filenames in the batch directory...")
    batch_ids = read_batch_ids(BATCH_DIR)
    print(f"Found batch IDs: {batch_ids}")

    attempts = {batch_id: 0 for batch_id in batch_ids}

    while batch_ids:
        for batch_id in batch_ids[:]:  # Use a copy of the list to allow modifications
            print(f"Checking status of batch: {batch_id}")

            batch = get_batch_status(API_KEY, batch_id)
            batch_details = serialize_batch(batch)
            print(json.dumps(batch_details, indent=2))

            if batch.status == "completed":
                print(f"Batch {batch_id} completed successfully.")
                if batch.output_file_id:
                    print(f"Downloading output file {batch.output_file_id}...")
                    output_file_path = download_file(API_KEY, batch.output_file_id, OUTPUT_DIR, f"batch_{batch_id}")
                    print(f"Output file downloaded to {output_file_path}")
                    
                    # Move the corresponding .txt file to the completed directory
                    if not os.path.exists(COMPLETED_DIR):
                        os.makedirs(COMPLETED_DIR)
                    
                    batch_txt_file = os.path.join(BATCH_DIR, f"{batch_id}.txt")
                    completed_txt_file = os.path.join(COMPLETED_DIR, f"{batch_id}.txt")
                    if os.path.exists(batch_txt_file):
                        os.rename(batch_txt_file, completed_txt_file)
                        print(f"Moved batch file {batch_txt_file} to {completed_txt_file}")
                    
                    # Extract JSON lines to individual files and content to text files next to the original image files
                    process_result_files(OUTPUT_DIR, EXTRACTION_BASE_DIR, batch_id)

                # Remove the completed batch from the list
                batch_ids.remove(batch_id)
                del attempts[batch_id]

            elif batch.status == "failed":
                print(f"Batch {batch_id} failed.")
                batch_ids.remove(batch_id)
                del attempts[batch_id]

            elif batch.status == "expired":
                print(f"Batch {batch_id} expired.")
                batch_ids.remove(batch_id)
                del attempts[batch_id]

            else:
                print(f"Batch {batch_id} is in status: {batch.status}")
                if attempts[batch_id] >= RETRY_LIMIT:
                    print(f"Reached maximum retry limit for batch {batch_id}. Exiting.")
                    batch_ids.remove(batch_id)
                    del attempts[batch_id]
                else:
                    attempts[batch_id] += 1

        if batch_ids:
            print(f"Waiting for {RETRY_DELAY} seconds before next check...")
            time.sleep(RETRY_DELAY)
    
    # Clean the directory
    print("Cleaning")
    clean('/data/lhyman6/OCR/data/borr/test/rolls')
