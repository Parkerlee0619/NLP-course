from datasets import load_dataset, load_from_disk
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
# dataset.save_to_disk("datasets/wikitext")
dataset = load_from_disk("datasets/wikitext")

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))



