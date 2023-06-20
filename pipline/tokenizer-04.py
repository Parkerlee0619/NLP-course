import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1 = "I've been waiting for a HuggingFace course my whole life."
sequence2 = "I hate this so much!"


tokens1 = tokenizer.tokenize(sequence1)
sequence1_ids = tokenizer.convert_tokens_to_ids(tokens1)
tokens2 = tokenizer.tokenize(sequence2)
sequence2_ids = tokenizer.convert_tokens_to_ids(tokens2)

print("Input Sequence1_ids:", torch.tensor([sequence1_ids]))
print("Input Sequence2_ids:", torch.tensor([sequence2_ids]))

padding_id = tokenizer.pad_token_id
batched_ids = [
    [ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
          2026,  2878,  2166,  1012],
    [1045, 5223, 2023, 2061, 2172,  999, padding_id, padding_id, padding_id, padding_id, padding_id, padding_id, padding_id, padding_id]     
]

attention_mask = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
]

print("Sequence1 logits:", model(torch.tensor([sequence1_ids])).logits)
print("Sequence2 logits:", model(torch.tensor([sequence2_ids])).logits)
print("Batched logits:", model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask)))
