# from transformers import AutoTokenizer, AutoModelForCausalLM
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path

# # Define the model path
# model_path = "liuhaotian/llava-v1.6-mistral-7b"

# # Load the tokenizer and model using Hugging Face transformers
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)

# # Load additional components if necessary using custom LLaVA utilities
# # This example assumes that load_pretrained_model is a utility to load a pre-trained model
# # and its components (e.g., image processor, context length, etc.)
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path),
#     offload_folder="/content/llava_model"
# )

# # Example input text
# input_text = "Hello, how are you?"

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors="pt")

# # Generate output using the model
# outputs = model.generate(**inputs)

# # Decode the generated tokens
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(generated_text)

import importlib.util
import sys
import os
from llava.train.train import train

# Define the path to the llava module
module_path = '/data/lhyman6/OCR/scripts/ocr_llm/llava/LLaVA/llava'

# Add the module path to the system path
sys.path.append(module_path)

# Import the llava module
spec = importlib.util.spec_from_file_location("llava", os.path.join(module_path, "__init__.py"))
llava = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llava)

# Inspect the llava module
print(dir(llava))


