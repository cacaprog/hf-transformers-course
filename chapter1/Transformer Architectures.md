https://huggingface.co/learn/llm-course/en/chapter1/6

Transformer models use one of three architectures: encoder-only, decoder-only, or encoder-decoder (sequence-to-sequence)

## Choosing the right architecture

When working on a specific NLP task, how do you decide which architecture to use? Here’s a quick guide:

|Task|Suggested Architecture|Examples|
|---|---|---|
|Text classification (sentiment, topic)|Encoder|BERT, RoBERTa|
|Text generation (creative writing)|Decoder|GPT, LLaMA|
|Translation|Encoder-Decoder|T5, BART|
|Summarization|Encoder-Decoder|BART, T5|
|Named entity recognition|Encoder|BERT, RoBERTa|
|Question answering (extractive)|Encoder|BERT, RoBERTa|
|Question answering (generative)|Encoder-Decoder or Decoder|T5, GPT|
|Conversational AI|Decoder|GPT, LLaMA|

> When in doubt about which model to use, consider:
> 
>1. What kind of understanding does your task need? (Bidirectional or unidirectional)
>2. Are you generating new text or analyzing existing text?
>3. Do you need to transform one sequence into another?
>
>The answers to these questions will guide you toward the right architecture.

## [](https://huggingface.co/learn/llm-course/en/chapter1/6#attention-mechanisms)Attention mechanisms

- LSH attention - used by Reformer
- Local attention
- Axial positional encodings

