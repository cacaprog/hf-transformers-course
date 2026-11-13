https://huggingface.co/learn/llm-course/en/chapter2/4

translate text into data that can be processed by model (numbers)
a representation

word-based
character-based
	- differ according languages, ex.: chinese character carries more information than in a latin language
subword tokenization

- Byte-level BPE, as used in GPT-2
- WordPiece, as used in BERT
- SentencePiece or Unigram, as used in several multilingual models

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

encoding -> tokenization -> decoding

encoding
- tokenization
- conversion to input IDs

decoding
from vocabulary indices, we want to get a string


### Batch

When you’re trying to batch together two (or more) sentences, they might be of different lengths. If you’ve ever worked with tensors before, you know that they need to be of rectangular shape, so you won’t be able to convert the list of input IDs into a tensor directly. To work around this problem, we usually _pad_ the inputs.

```python
# The following list cannot be converted to a tensor
batched_ids = [
    [200, 200, 200],
    [200, 200]
]

# using pad to make our tensor a rectangular shpae
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

Padding makes sure all our sentences have the same length by adding a special word called the _padding token_ to the sentences with fewer values.

### Attention masks

the key feature of Transformer models is attention layers that _contextualize_ each token

_Attention masks_ are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, and 0s indicate the corresponding tokens should not be attended to (i.e., they should be ignored by the attention layers of the model).

---

`"pt"` returns PyTorch tensors and `"np"` returns NumPy arrays

---

>[!warning ] The tokenizer and model should always be from the same checkpoint.
>Look the wrong example
>

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("gpt2")  # model checkpoint is different from tokenizer

encoded = tokenizer("Hey!", return_tensors="pt")
result = model(**encoded)
```


