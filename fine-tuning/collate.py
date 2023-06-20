from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})