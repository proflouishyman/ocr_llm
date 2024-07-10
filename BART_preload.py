from transformers import BartForConditionalGeneration, BartTokenizer

# Load pre-trained model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


print("BART loaded")