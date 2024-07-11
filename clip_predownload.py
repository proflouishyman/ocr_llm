from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-large-patch14-336"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Save the model and processor locally
model.save_pretrained('./clip-vit-large-patch14-336')
processor.save_pretrained('./clip-vit-large-patch14-336')
