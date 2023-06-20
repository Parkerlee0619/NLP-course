from datasets import load_dataset

raw_datasets = load_dataset("squad")
raw_datasets.save_to_disk("datasets/squad")
print(raw_datasets)