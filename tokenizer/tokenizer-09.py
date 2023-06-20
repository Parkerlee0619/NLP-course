from transformers import AutoTokenizer
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("gpt2")


corpus =  [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]
word_freqs = defaultdict(int)

vocab =  ["<|endoftext|>"] + ["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]

for word, freq in corpus:
        word_freqs[word] += freq

splits = {word: [c for c in word] for word in word_freqs.keys()}

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    total = 0
    for pair in vocab:
        for word, freq in word_freqs.items():
            split = splits[word]
            i = 0
            while i <= len(split) - len(pair):
                j = 0
                sub = ''
                while j < len(pair):
                    sub = sub + split[i+j]
                    j += 1
                print(sub)
                if sub == pair:
                    pair_freqs[pair] += freq
                    total += freq
                i += 1
    return pair_freqs, total

pair_freqs, total = compute_pair_freqs(splits)

def compute_pair_score(split):
     if len(split) == 1 :
          return pair_freqs[split] / total
     

def tokenize(text):
     splits = {word: [c for c in word] for word in text}
   

