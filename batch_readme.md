# OpenAI Batch Processing Scripts

## Overview
This repository contains scripts to upload image files to the OpenAI API for batch processing, download the processed results, and extract relevant data from the results.

### Scripts Included
1. `batch_upload.py`: Script to create JSONL files, upload them, and create batches for the OpenAI API.
2. `batch_download.py`: Script to check the status of batches, download the output files if the batches are completed, extract each line from JSONL result files into individual JSON files named by `custom_id`, and extract the "content" field from each JSON file into text files next to the original image files.
3. bigclean.py cleans all the files and labels the ones that need to be sent to the api for deep cleaning which get the .api suffix.
5. test_json.py validates files for json format and deletes the text files if they exist.


## Setup
1. Ensure you have Python and the `openai` package installe
2. Place your OpenAI API key in a file named `api_key.txt`.
3. Ensure your image files and the prompt file are in the specified directories.

## batch_upload.py

### Description
This script creates JSONL files from a directory of image files, uploads them to the OpenAI API, and creates batches for processing. Each batch contains a maximum of 400 images.

### Variables
- `API_KEY_FILE`: Path to the file containing your OpenAI API key.
- `IMAGE_DIR`: Directory containing image files to be processed.
- `PROMPT_FILE`: Path to the file containing the prompt text.
- `JSONL_FILE_BASE`: Base name of the JSONL file to be created.
- `API_URL`: The OpenAI API endpoint URL.
- `BATCH_DIR`: Directory to store batch-related files.
- `DESCRIPTION`: Description for the batch.
- `VALID_IMAGE_TYPES`: List of valid image file extensions.
- `MAX_IMAGES_PER_BATCH`: Maximum number of images per batch (default is 400).

### Usage
1. Place your image files in the directory specified by `IMAGE_DIR`.
2. Create a text file containing your prompt and set the `PROMPT_FILE` variable to its path.
3. Run the script:
    ```bash
    python batch_upload.py
    ```

### Notes
- The `custom_id` for each JSONL entry will include the full path of the image file, with directory separators replaced by the `|` character.
- Each batch will have a separate JSONL file named with a counter to distinguish between them.
- The JSONL files will be deleted after they are successfully uploaded.

## batch_download.py

### Description
This script checks the status of batches, downloads the output files if the batches are completed, extracts each line from JSONL result files into individual JSON files named by `custom_id`, and extracts the "content" field from each JSON file into text files next to the original image files.

### Variables
- `API_KEY_FILE`: Path to the file containing your OpenAI API key.
- `BATCH_DIR`: Directory containing batch-related `.txt` files.
- `COMPLETED_DIR`: Subdirectory of `BATCH_DIR` for storing completed batch `.txt` files.
- `OUTPUT_DIR`: Directory to store downloaded JSONL result files.
- `EXTRACTION_BASE_DIR`: Base directory to store extracted JSON files.

### Usage
1. Ensure that the `batch_upload.py` script has been run and the batch `.txt` files are present in the `BATCH_DIR`.
2. Run the script:
    ```bash
    python batch_download.py
    ```

### Processed Output
- Downloaded JSONL files are stored in the `OUTPUT_DIR` directory.
- Extracted JSON files are stored in subdirectories of `EXTRACTION_BASE_DIR` named after the batch IDs.
- Extracted text files are saved next to the original image files. The filenames for the text files are derived from the `custom_id` with `.txt` appended.

## Example Directory Structure

├── api_key.txt
├── batch
│ ├── batch_abc123.txt
│ └── completed
├── batch_download.py
├── batch_upload.py
├── batch_return
│ └── batch_abc123.jsonl
├── batch_json_results
│ └── batch_abc123
│ ├── data|lhyman6|OCR|data|borr|test|rolls|tray_1_roll_11_page3986_img1.png.json
│ └── data|lhyman6|OCR|data|borr|test|rolls|tray_1_roll_11_page3980_img1.png.json
├── data
│ └── lhyman6
│ └── OCR
│ └── data
│ └── borr
│ └── test
│ └── rolls
│ ├── tray_1_roll_11_page3986_img1.png
│ ├── tray_1_roll_11_page3986_img1.png.txt
│ ├── tray_1_roll_11_page3980_img1.png
│ └── tray_1_roll_11_page3980_img1.png.txt
└── rolls_prompt.txt

