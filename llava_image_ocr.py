from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os
from tqdm import tqdm  # For progress indicators

# Date: 2024-07-14
# Purpose: Extract text from every JPG in a directory and save the text files in a different directory, with progress indicators and checks for already processed files, and an option to process only a specified number of images.

# Model path
model_path = "llava-hf/llava-v1.6-mistral-7b-hf"

# Initialize processor and model
processor = LlavaNextProcessor.from_pretrained(model_path)
model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

# Directories
input_directory = "/data/lhyman6/OCR/scripts/data/second_images"  # Update with the path to your input directory
output_directory = "/data/lhyman6/OCR/scripts/ocr_llm/test"  # Update with the path to your output directory

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Get a list of all JPG files in the input directory
jpg_files = [f for f in os.listdir(input_directory) if f.lower().endswith(".jpg")]

# Specify the maximum number of images to process (set to None to process all images)
max_images = 10  # Change this value as needed

# Iterate over all JPG files with progress indicators
processed_count = 0
for filename in tqdm(jpg_files, desc="Processing images"):
    if max_images is not None and processed_count >= max_images:
        tqdm.write("Reached the maximum number of images to process.")
        break
    
    input_path = os.path.join(input_directory, filename)
    
    # Determine the corresponding output file path
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_directory, output_filename)
    
    # Check if the file has already been processed
    if os.path.exists(output_path):
        tqdm.write(f"Skipping {filename} (already processed)")
        continue
    
    # Open image
    image = Image.open(input_path)
    prompt = "[INST] <image>\n Use OCR and extract text from this letter. Take your time.[/INST]"
    
    # Prepare input
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    
    # Generate output
    output = model.generate(**inputs, max_new_tokens=1024)
    
    # Decode and save the text
    text = processor.decode(output[0], skip_special_tokens=True)
    
    with open(output_path, "w") as f:
        f.write(text)
    
    tqdm.write(f"Processed {filename} and saved text to {output_filename}")
    processed_count += 1
