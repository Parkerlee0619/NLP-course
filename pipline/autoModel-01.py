from transformers import BertConfig, BertModel
import torch

model = BertModel.from_pretrained("models/bert-base-cased")

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences)
output = model(model_inputs)
print(output)