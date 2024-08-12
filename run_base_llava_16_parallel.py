# 2024-08-12. Code to use mixed precision with LlavaNext model.
# Purpose: Utilize mixed precision to optimize memory usage and increase inference speed.
#works August 12 2024
import os
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# Set environment variable to avoid warnings about parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the model and processor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=False)

# Move model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define local image file path as a variable
image_file = "/data/lhyman6/OCR/scripts/data/second_images/mss511850149-610.jpg"
image = Image.open(image_file)
prompt = "[INST] <image>\nGive me an OCR of this image[/INST]"

# Enable mixed precision
scaler = GradScaler()

# Verify the model using mixed precision
with autocast():
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200)

print(processor.decode(output[0], skip_special_tokens=True))
