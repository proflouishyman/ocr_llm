from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
import torch
from PIL import Image

# Disable torch init
disable_torch_init()

# Load model and tokenizer
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()

# Load CLIP vision model and image processor
vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").cuda().half()
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

def ask_llava(image_path, question):
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_processor(image, return_tensors='pt')['pixel_values'].half().cuda()

    # Get image features
    with torch.no_grad():
        image_features = vision_tower(image_tensor).last_hidden_state

    # Prepare conversation
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Generate response
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_features,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria]
        )

    # Decode and print response
    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return output

# Example usage
image_path = '/data/lhyman6/OCR/scripts/data/second_images/mss511850261-1068.jpg'
question = "What is written in this letter?"

response = ask_llava(image_path, question)
print(response)
