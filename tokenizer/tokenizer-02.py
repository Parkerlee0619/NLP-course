from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
start, end = encoding.word_to_chars(3)
print(encoding.word_ids())
print(example[start:end])
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# example = "81s"
# encoding = tokenizer(example)
# print(encoding.tokens())