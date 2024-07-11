from transformers import LlavaNextProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os
from tqdm import tqdm  # For progress indicators

# Date: 2024-05-14
# Purpose: Extract text from every JPG in a directory and save the text files in a different directory, with progress indicators and checks for already processed files.
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb



# Initialize processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")


#i am not sure whether to use below or above.
#processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf"D)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right