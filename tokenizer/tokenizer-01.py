from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("code-search-net-tokenizer")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = tokenizer.tokenize(example)
print(tokens)