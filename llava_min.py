# llava_local_no_gradio.py
# Date: YYYY-MM-DD
# Time: HH:MM
# Purpose: Run LLaVA locally using HuggingFace without Gradio.

"""pip install tokenizers==0.15.1
pip install torch==2.1.2 torchvision==0.16.2
pip install transformers==4.37.2
pip install llava==1.2.2.post1"""


from transformers import pipeline
from PIL import Image
import base64
from io import BytesIO
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load the LLaVA model
model_id = "llava-hf/llava-1.5-7b-hf"
llava_pipeline = pipeline("image-to-text", model=model_id, device=device)

# Define the function to interact with the model
def ask_llava(image_path, question):
    # Open and process the image
    with Image.open(image_path).convert("RGB") as img:
        # Resize the image if it's too large
        img.thumbnail((1000, 1000))
        
        # Convert the image to bytes
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        image_bytes = buf.getvalue()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Create the input dictionary
    input_data = {
        "image": image_base64,
        "text": question
    }
    
    # Get the response from the model
    response = llava_pipeline(input_data)
    return response


# Example usage
image_path = '/data/lhyman6/OCR/scripts/data/second_images/mss511850261-1068.jpg'
question = "What is written in this letter?"

response = ask_llava(image_path, question)
print(response)

# Example usage
image_path = '/data/lhyman6/OCR/scripts/data/second_images/mss511850261-1068.jpg'
question = "What is written in this letter?"

response = ask_llava(image_path, question)
print(response)
