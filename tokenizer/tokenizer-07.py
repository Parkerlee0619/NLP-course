from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import numpy as np

# question_answerer = pipeline("question-answering")

long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
# outputs = question_answerer(question=question, context=context)
# print(outputs)

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")

outputs = model(**inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

start_logits[mask] = -10000
end_logits[mask] = -10000

start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

candidates = []
# for start_probs, end_probs in zip(start_probabilities, end_probabilities):
#     scores = start_probs[:, None] * end_probs[None, :]
#     idx = torch.triu(scores).argmax().item()

#     start_idx = idx // scores.shape[1]
#     end_idx = idx % scores.shape[1]
#     score = scores[start_idx, end_idx].item()
#     candidates.append((start_idx, end_idx, score))

# for candidate, offset in zip(candidates, offsets):
#     start_token, end_token, score = candidate
#     start_char, _ = offset[start_token]
#     _, end_char = offset[end_token]
#     answer = long_context[start_char:end_char]
#     result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
#     print(result)

for start_probs, end_probs, offset in zip(start_probabilities, end_probabilities, offsets):
    scores = start_probs[:, None] * end_probs[None, :]
    scores = torch.triu(scores)

    # top 5 anwser in each block
    top_five_idx = torch.topk(scores.flatten(), 5).indices

    for idx in top_five_idx:
        start_idx = idx // scores.shape[1]
        end_idx = idx % scores.shape[1]
        score = scores[start_idx, end_idx].item()
        start_char, _ = offset[start_idx]
        _, end_char = offset[end_idx]
        candidates.append((start_char, end_char, score))

# top 5 anwser in all candidates
candidates = sorted(candidates, key=lambda x:x[2], reverse=True)[:5]

for candidate in candidates:
    start_char, end_char, score = candidate
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)
    

