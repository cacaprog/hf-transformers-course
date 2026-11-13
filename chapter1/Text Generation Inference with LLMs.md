https://huggingface.co/learn/llm-course/en/chapter1/8

The role of attention
- attention mechanism: enables LLMs to understand context and generate coherent responses
- relevant information: focuses on crucial words to predict the next token
- advancements: improved scalling and efficiency in handling longer sequences

Two-phase inference process
- prefill phase: tokenization, embedding conversion, and initial processing
- decode phase: autoregressive text generation, involving attention computation, probability calculation, token selection, and continuation check

### [](https://huggingface.co/learn/llm-course/en/chapter1/8#context-length-and-attention-span)Context Length and Attention Span

attention span

> The context length is the maximum number of tokens the model can consider at once when generating a response.

These capabilities are limited by several practical factors:
- The model’s architecture and size
- Available computational resources
- The complexity of the input and desired output

## Sampling Strategies

### Token Selection: From Probabilities to Token Choices

![image](https://huggingface.co/reasoning-course/images/resolve/main/inference/1.png)

### Controlling Generation Length: Setting Boundaries

We can control generation length in several ways:

1. **Token Limits**: Setting minimum and maximum token counts
2. **Stop Sequences**: Defining specific patterns that signal the end of generation
3. **End-of-Sequence Detection**: Letting the model naturally conclude its response

![image](https://huggingface.co/reasoning-course/images/resolve/main/inference/3.png)

### [](https://huggingface.co/learn/llm-course/en/chapter1/8#beam-search-looking-ahead-for-better-coherence)Beam Search: Looking Ahead for Better Coherence

Instead of committing to a single choice at each step, it explores multiple possible paths simultaneously - like a chess player thinking several moves ahead.

![image](https://huggingface.co/reasoning-course/images/resolve/main/inference/4.png)

## [](https://huggingface.co/learn/llm-course/en/chapter1/8#practical-challenges-and-optimization)Practical Challenges and Optimization

### Key Performance Metrics

When working with LLMs, four critical metrics will shape your implementation decisions:

1. **Time to First Token (TTFT)**: How quickly can you get the first response? This is crucial for user experience and is primarily affected by the prefill phase.
2. **Time Per Output Token (TPOT)**: How fast can you generate subsequent tokens? This determines the overall generation speed.
3. **Throughput**: How many requests can you handle simultaneously? This affects scaling and cost efficiency.
4. **VRAM Usage**: How much GPU memory do you need? This often becomes the primary constraint in real-world applications.

### [](https://huggingface.co/learn/llm-course/en/chapter1/8#the-context-length-challenge)The Context Length Challenge

One of the most significant challenges in LLM inference is managing context length effectively. Longer contexts provide more information but come with substantial costs:

- **Memory Usage**: Grows quadratically with context length
- **Processing Speed**: Decreases linearly with longer contexts
- **Resource Allocation**: Requires careful balancing of VRAM usage

### The KV Cache Optimization

To address these challenges, one of the most powerful optimizations is KV (Key-Value) caching. This technique significantly improves inference speed by storing and reusing intermediate calculations. This optimization:

- Reduces repeated calculations
- Improves generation speed
- Makes long-context generation practical

The trade-off is additional memory usage, but the performance benefits usually far outweigh this cost.