![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/whisper_architecture.png)https://huggingface.co/learn/llm-course/en/chapter1/4

- GPT-like - auto-regressive transformer models
- BERT-like - auto-encoding transformer models
- T5-like - sequence-to-sequence transformer models

[[Encoder, decoder and encoder-decoder]]

causal language modeling - the output (predict the next word) depends on the past and present inputs

masked kabguage modeling - model predicts a masked word in the sentence

sharing the trained weights and building on top of already trained weights reduces the overall compute cost and carbon footprint of the community.

pretaining - act of training a model from scratch

![The pretraining of a language model is costly in both time and money.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/pretraining-dark.svg)

fine-tuning - training after a model has been pretrained

![The fine-tuning of a language model is cheaper than pretraining in both time and money.](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/finetuning-dark.svg)

always try to leverage a pretrained model — one as close as possible to the task you have at hand — and fine-tune it

- encoder-only models - good for tasks that require understanding of the input (sentence classification, ner)
- decoder-only models - good for generative tasks such as text generation
- encoder-decoder models or sequento-to-sequence - good for generative tasks that require an input, such as translation or summarization


Attention layers
a word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied

Attention mask
used in the encoder/decoder to prevent the model from paying attention to some special words

Architecture x checkpoint

- Architecture: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
- Checkpoints: These are the weights that will be loaded in a given architecture.
- Model: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both.

For example, BERT is an architecture while `bert-base-cased`, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say “the BERT model” and “the `bert-base-cased` model.”

---

# [](https://huggingface.co/learn/llm-course/en/chapter1/5#how--transformers-solve-tasks)How 🤗 Transformers solve tasks

https://huggingface.co/learn/llm-course/en/chapter1/5

Language models

approaches to train a transformer model:
- masked language modeling (MLM) - used by encoder models like BERT, randomly mask some tokens in the input and trains the model to predict the original tokens based on the surroinding context
- causal language modeling (CLM) - used by decoder models like GPT, predicts the next token based on all previous tokens in the sequence.

- **Encoder-only models** (like BERT): These models use a bidirectional approach to understand context from both directions. They’re best suited for tasks that require deep understanding of text, such as classification, named entity recognition, and question answering.

- **Decoder-only models** (like GPT, Llama): These models process text from left to right and are particularly good at text generation tasks. They can complete sentences, write essays, or even generate code based on a prompt.

- **Encoder-decoder models** (like T5, BART): These models combine both approaches, using an encoder to understand the input and a decoder to generate output. They excel at sequence-to-sequence tasks like translation, summarization, and question answering.

![transformer-models-for-language](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers_architecture.png)

transfer learning - allows models to adapt to many different NPL tasks wity relative small amounts of task-specific data

## Tasks

Text classification
- sentiment analysis
- topic classification
- spam detection

text classification guide to learn how to finetune DistilBERT and use it for inference! - https://huggingface.co/docs/transformers/tasks/sequence_classification


Token classification
assigning a label to each token in a sequence
- ner (named entity recognition)
- part-of-speech tagging

Check out our complete [token classification guide](https://huggingface.co/docs/transformers/tasks/token_classification) to learn how to finetune DistilBERT and use it for inference!


Question answering
finding the answer to a question within a given context or passage

Check out our complete [question answering guide](https://huggingface.co/docs/transformers/tasks/question_answering) to learn how to finetune DistilBERT and use it for inference!

Summarization
condensing a longer text into a shorter version while preserving its key information and meaning

use encoder-decoder models like BART and T5

Check out our complete [summarization guide](https://huggingface.co/docs/transformers/tasks/summarization) to learn how to finetune T5 and use it for inference!


Translation
converting text from one language to another while preserving its meaning
another example of a sequence-to-sequence task

Check out our complete [translation guide](https://huggingface.co/docs/transformers/tasks/translation) to learn how to finetune T5 and use it for inference!

### Beyond text
Speech and audio
Whisper is a encoder-decoder (sequence-to-sequence) transformer pretrained on 680,000 hours of labeled audio data

![](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/whisper_architecture.png)


ASR - automatic speech recognition

Check out our complete [automatic speech recognition guide](https://huggingface.co/docs/transformers/tasks/asr) to learn how to finetune Whisper and use it for inference!


Computer Vision
understanding and interpreting visual information from images or videos.
- image classification
- object detection
- segmentation
- depth estimation

two ways to do that:
- Split an image into a sequence of patches and process them in parallel with a Transformer.
- Use a modern CNN, like [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext), which relies on convolutional layers but adopts modern network designs.

> A third approach mixes Transformers with convolutions (for example, [Convolutional Vision Transformer](https://huggingface.co/docs/transformers/model_doc/cvt) or [LeViT](https://huggingface.co/docs/transformers/model_doc/levit)).

Image classification
ViT and ConvNeXT can both be used for image classification; the main difference is that ViT uses an attention mechanism while ConvNeXT uses convolutions.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg)

Check out our complete [image classification guide](https://huggingface.co/docs/transformers/tasks/image_classification) to learn how to fine-tune ViT and use it for inference!

