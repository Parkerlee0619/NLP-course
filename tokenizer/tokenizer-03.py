from transformers import pipeline

# token_classifier = pipeline("token-classification")
# token_classifier.save_pretrained("pipeline/token-classification")
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))