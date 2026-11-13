[[References#HuggingFace Pipeline]]

https://huggingface.co/learn/llm-course/en/chapter2/1

![The full NLP pipeline: tokenization of text, conversion to IDs, and inference through the Transformer model and the model head.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline-dark.svg)

Transformer models only accept _tensors_ as input

return_tensors="pt"
seria o pt o PyTorch?

hidden states, also knows as features


### Vector output by the Transformer
- batch size - The number of sequences processed at a time
- sequence length - The length of the numerical representation of the sequence
- hidden size - The vector dimension of each model input (768 for smaller models, 3072 or more for larger)

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

```markdown
torch.Size([2, 16, 768])
```

2 - batch size
16 - sequence length
768 - hidden size

### Model heads

![A Transformer network alongside its head.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head-dark.svg)

The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension.

There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task. Here is a non-exhaustive list:

- `*Model` (retrieve the hidden states)
- `*ForCausalLM`
- `*ForMaskedLM`
- `*ForMultipleChoice`
- `*ForQuestionAnswering`
- `*ForSequenceClassification`
- `*ForTokenClassification`
- and others 🤗

## Models

https://huggingface.co/learn/llm-course/en/chapter2/3

> 🤗 Transformers library is the ability to easily share models and tokenizers with the community.

`AutoModel`

checkpoint name - specific model architecture and weights
ex: bert-base-cased

When we save a model, two files are created:
config.json - model configuration, architecture
model.safetensors - model weights, parameters of the model


Hub API:
- Push a model from a local repository
- Update specific files without re-uploading everything
- Add model cards to document the model’s abilities, limitations, known biases, etc.

Encoding text
Arrays and tensors need to be rectangular, so we can’t simply convert these lists to a PyTorch tensor (or NumPy array). The tokenizer provides an option for that: padding.

If you have sequences longer than the model can handle, you’ll need to truncate them with the `truncation` parameter

By combining the padding and truncation arguments, you can make sure your tensors have the exact size you need

```python
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)
```

```python
{'input_ids': tensor([[  101,  1731,  1132,  1128,   102],
         [  101,  1045,  1005,  1049,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]])}
```

