from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer

# This can take a few minutes to load, so grab a coffee or tea while you wait!
# raw_datasets = load_dataset("code_search_net", "python")
# raw_datasets.save_to_disk("datasets/CoedSearchNet")
raw_datasets = load_from_disk("datasets/CoedSearchNet")

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )



training_corpus = get_training_corpus()
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

# Save tokenizer
tokenizer.save_pretrained("code-search-net-tokenizer")


example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = tokenizer.tokenize(example)
print(tokens)