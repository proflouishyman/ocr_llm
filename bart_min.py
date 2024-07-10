from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def main():
    model_path = '/scratch4/lhyman6/OCR/work/tuning_results_long/checkpoint-15500'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    print("Model loaded successfully")

    sample_text = "This is a test sentence."
    inputs = tokenizer(sample_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", result)

if __name__ == "__main__":
    main()
