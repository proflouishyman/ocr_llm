import base64
import json
import os
from openai import OpenAI
from tqdm import tqdm

# Variables
API_KEY_FILE = "api_key.txt"
IMAGE_DIR = "/data/lhyman6/data/images/"  # Directory containing image files
PROMPT_FILE = "/data/lhyman6/OCR/scripts/ocr_llm/gompers_ocr_prompt.txt"  # File containing the prompt
API_URL = "https://api.openai.com/v1/chat/completions"
BATCH_DIR = "batch"
DESCRIPTION = "test gompers upload"
VALID_IMAGE_TYPES = [".jpg", ".jpeg", ".png"]
MAX_BATCH_SIZE_BYTES = 95 * 1024 * 1024  # slightly less than 100 MB
MAX_BATCHES = 1  # Set the maximum number of batches to process
JSONL_FILE_BASE = "batchinput"

def read_prompt(prompt_file):
    """Read the prompt from a file."""
    with open(prompt_file, "r") as file:
        return file.read().strip()

def create_jsonl_entry(image_path, prompt, custom_id, model="gpt-4o"):
    """Create a JSONL entry for batch processing."""
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    img_data_size = len(img_data)
    file_format = image_path.split('.')[-1].lower()
    mime_type = f"image/{'jpeg' if file_format in ['jpg', 'jpeg'] else file_format}"
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": {"type": "text", "text": prompt}},
            {"role": "user", "content": {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_data}"}}}
        ],
        "max_tokens": 4000
    }
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}, img_data_size

def get_files(image_dir):
    """Get all files in the specified directory."""
    return [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

def get_primary_basename(file_path):
    """Return the primary base name of the file, cutting off at the first period."""
    return file_path.split('/')[-1].split('.', 1)[0]

def create_custom_id(image_path):
    """Create a custom ID by replacing problematic characters in the image path."""
    return image_path.replace("/", "|").replace("\\", "|")

def write_jsonl_file(entries, output_file):
    """Write multiple JSONL entries to a file."""
    with open(output_file, "w") as jsonl_file:
        for entry in entries:
            jsonl_file.write(json.dumps(entry, separators=(',', ':')) + "\n")  # Compress JSON

def upload_jsonl_file(api_key, jsonl_file_path):
    """Upload the JSONL file to OpenAI for batch processing."""
    client = OpenAI(api_key=api_key)
    with open(jsonl_file_path, "rb") as jsonl_file:
        batch_input_file = client.files.create(
            file=jsonl_file,  # Use the file object directly
            purpose="batch"
        )
    return batch_input_file

def create_batch(api_key, batch_input_file_id, description):
    """Create a batch using the uploaded JSONL file ID."""
    client = OpenAI(api_key=api_key)
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )
    return batch

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


if __name__ == "__main__":
    print("Reading API key from file...")
    with open(API_KEY_FILE, "r") as file:
        API_KEY = file.read().strip()
    print("API key successfully read.")

    print("Reading prompt from file...")
    PROMPT = read_prompt(PROMPT_FILE)
    print(f"Prompt: {PROMPT}")

    print(f"Collecting files from directory: {IMAGE_DIR}")
    all_files = get_files(IMAGE_DIR)
    all_basenames = {get_primary_basename(f) for f in tqdm(all_files, desc="Processing files")}

    print(f"Found {len(all_files)} files.")

    # Filter to keep only image files without corresponding non-image files
    filtered_image_files = [f for f in all_files if get_primary_basename(f) in all_basenames and f.endswith(tuple(VALID_IMAGE_TYPES))]

    print(f"Filtered image files to {len(filtered_image_files)} for processing.")

    batch_index = 0
    while batch_index < MAX_BATCHES:
        current_batch_size = 0
        valid_batch_files = []
        batch_entries = []

        for file in tqdm(filtered_image_files, desc=f"Processing batch {batch_index + 1}"):
            entry, img_data_size = create_jsonl_entry(file, PROMPT, create_custom_id(file))
            if current_batch_size + img_data_size > MAX_BATCH_SIZE_BYTES:
                break
            valid_batch_files.append(file)
            batch_entries.append(entry)
            current_batch_size += img_data_size

        if not valid_batch_files:
            print("No files fit into the batch size limit. Exiting.")
            break

        jsonl_file = f"{JSONL_FILE_BASE}_batch_{batch_index + 1}.jsonl"
        write_jsonl_file(batch_entries, jsonl_file)
        print(f"JSONL file {jsonl_file} created and uploaded successfully.")

        batch_input_file = upload_jsonl_file(API_KEY, jsonl_file)
        batch = create_batch(API_KEY, batch_input_file.id, DESCRIPTION)
        print("Batch created successfully. Batch details:")
        batch_details = serialize_batch(batch)
        print(json.dumps(batch_details, indent=2))

        # Clean up
        if os.path.exists(jsonl_file):
            os.remove(jsonl_file)
        batch_index += 1

