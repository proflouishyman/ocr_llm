# Date: 2024-07-11
# This script processes images and associated text files from a directory to prepare for training a LLaVA model using the datasets library.
# It reads from a CSV file to get the IDs and associated ground truth text, loads the corresponding images, and processes the data.
# It also creates directories and generates separate JSON files for the gold and silver data with sizes of 100, 1000, and 10000.

from PIL import Image
import os
import json
import pandas as pd
from tqdm import tqdm

def load_local_dataset(csv_file, image_folder, data_column='gold'):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    dataset = []
    print(f"Loading {data_column} dataset...")

    for index, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
        image_id = row['id']
        image_path = os.path.join(image_folder, f"{image_id}.jpg")
        groundtruth = row[data_column]

        # Ensure the image exists
        if os.path.exists(image_path):
            # Add to dataset
            dataset.append({
                'image': image_path,
                'groundtruth': groundtruth,
                'id': image_id
            })

    print(f"Loaded {len(dataset)} entries into the {data_column} dataset.")
    return pd.DataFrame(dataset)

def save_json(dataset, output_folder, subset_name, data_column):
    subset_folder = os.path.join(output_folder, f"{data_column}_{subset_name}")

    if not os.path.exists(subset_folder):
        os.makedirs(subset_folder)

    json_data_list = []

    for index, row in dataset.iterrows():
        image_path = row['image']
        groundtruth = row['groundtruth']
        unique_id = row['id']

        # Structure for LLaVA JSON
        json_data = {
            "id": unique_id,
            "image": image_path,  # Original path to the image
            "conversations": [
                {
                    "from": "human",
                    "value": "What is the OCR of this image?"
                },
                {
                    "from": "gpt",
                    "value": groundtruth
                }
            ]
        }

        # Append to list
        json_data_list.append(json_data)

    # Save the JSON data list to a file
    json_output_path = os.path.join(subset_folder, f"{data_column}_{subset_name}.json")
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data_list, json_file, indent=4)
    print(f"Saved {subset_name} {data_column} dataset to {json_output_path}.")

def save_datasets(csv_file, image_folder, output_folder, sizes, data_columns):
    for data_column in data_columns:
        dataset = load_local_dataset(csv_file, image_folder, data_column)

        for size in sizes:
            if len(dataset) >= size:
                subset_dataset = dataset.head(size)
                save_json(subset_dataset, output_folder, f"{size}", data_column)

# Usage example
csv_file = '/data/lhyman6/OCR/scripts/ocr_llm/complete_bart_training_data.csv'
image_folder = '/data/lhyman6/data/images'
output_folder = '/scratch4/lhyman6/OCR/OCR/ocr_llm/work/llava/'
sizes = [100, 1000, 10000]  # Sizes for testing
data_columns = ['gold', 'silver']  # Columns to be used

save_datasets(csv_file, image_folder, output_folder, sizes, data_columns)
