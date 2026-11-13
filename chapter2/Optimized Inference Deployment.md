https://huggingface.co/learn/llm-course/en/chapter2/8

fazer este exercício, que reúne toda a teoria e prática com modelos e seus controles

---
### Framework selection
Text Generation Inference (TGI)
vLLM
llama.cpp


How to deploy these frameworks in production

Flash Attention

---

| Feature | TGI (Text Generation Inference) | vLLM | llama.cpp |
| --- | --- | --- | --- |
| **Core Architecture** | Rust/Python server built around Hugging Face tooling | Python/CUDA engine built around **PagedAttention** & continuous batching | Pure **C/C++** inference engine built for zero-dependency portability |
| **Primary Hardware** | Datacenter NVIDIA GPUs | Datacenter/Consumer GPUs (NVIDIA, AMD ROCm, Intel, TPU) | **CPUs**, Apple Silicon (Metal), consumer GPUs (CUDA/Vulkan/ROCm) |
| **Quantization** | GPTQ, AWQ, EETQ, bitsandbytes, FP8 | FP8, AWQ, GPTQ, INT8/INT4, GGUF | **GGUF** (Q2 through Q8, K-quants, IQ-quants) |
| **Concurrency** | High (continuous/dynamic batching) | **Highest** (dynamic batching + paged KV-cache) | Low to Moderate (slot-based batching) |
| **API Compatibility** | Custom REST/gRPC + OpenAI Messages API | Full **OpenAI-compatible** API out of the box | Built-in `llama-server` with OpenAI-compatible endpoints |
| **Best For** | Turnkey production setups inside the Hugging Face ecosystem | High-traffic production APIs & multi-GPU cluster serving | Local dev, edge devices, CPU inference, and VRAM-constrained setups |

---

## 1. Text Generation Inference (TGI)

Developed by Hugging Face, TGI is an opinionated, production-ready Dockerized inference server engineered for stability and seamless integration with the Hugging Face Hub.

* **Recommended For:**
* Teams already deeply invested in the Hugging Face ecosystem (`transformers`, Hub models, Inference Endpoints).
* Rapid, standardized deployments via official Docker containers without heavy custom engineering.
* User-facing APIs prioritizing tightly clustered, predictable latency distributions and out-of-the-box observability.


* **Pros:**
* **Turnkey Production Observability:** Built-in Prometheus metrics, OpenTelemetry distributed tracing, and health checks out of the box.
* **Zero-Friction HF Integration:** Point it at a Hugging Face model ID and deploy immediately with native support for safetensors, token streaming, and watermarks.
* **Predictable Latency:** Strong SLA reliability with tightly clustered token latency distributions.
* **Guardrails & Structured Output:** Native support for JSON schema constraints and grammar-guided generation.


* **Cons:**
* **Lower Peak Throughput than vLLM:** Trails vLLM under heavy concurrent load due to differences in KV-cache memory paging and prefill/decode handling.
* **Rigid LoRA Handling:** Dynamically swapping multiple LoRA adapters on the fly is more cumbersome than in vLLM.
* **Heavier Footprint:** Primarily geared toward Linux + NVIDIA GPU environments; poorly suited for consumer laptops or CPU-only servers.



---

## 2. vLLM

Originally created at UC Berkeley and now governed under the PyTorch Foundation, vLLM is the industry standard for high-throughput, multi-user enterprise LLM serving.

* **Recommended For:**
* High-concurrency production APIs (16+ simultaneous users/agents) where tokens-per-second throughput directly dictates infrastructure cost.
* Multi-GPU and multi-node deployments requiring Tensor or Pipeline Parallelism for massive models (e.g., 70B–400B+ parameters).
* Agentic workflows and RAG pipelines with heavy system-prompt reuse.


* **Pros:**
* **PagedAttention & Prefix Caching:** Eliminates KV-cache memory fragmentation and reuses cached prompt prefixes across requests—dramatically cutting latency and compute in multi-turn chat and agent loops.
* **Massive Throughput:** Continuous batching allows new requests to join mid-generation, maximizing GPU utilization.
* **Day-Zero Model Support:** New open-weight architectures are typically supported within days of release.
* **Rich Serving Features:** Native support for speculative decoding, FP8 KV-cache quantization, multi-LoRA serving, and full OpenAI API parity (including tool/function calling).


* **Cons:**
* **Requires Full VRAM Fit:** Unlike `llama.cpp`, vLLM cannot easily split layers between system RAM and GPU VRAM; if a model doesn't fit in GPU memory, it won't load.
* **Operational Overhead:** Acts as a raw serving engine rather than a managed platform—you must build your own autoscaling, monitoring, and GPU recovery wrappers.
* **Fast Release Churn:** Frequent updates mean CLI flags and behaviors occasionally break between minor versions.



---

## 3. llama.cpp

Written in pure C/C++ with zero Python runtime dependencies, `llama.cpp` (and its `GGUF` file format) brought efficient LLM execution to consumer hardware, CPUs, and edge devices.

* **Recommended For:**
* Local development, offline/air-gapped apps, desktop assistants, and edge/on-device deployments.
* **VRAM-constrained setups** where a model must be split across GPU VRAM and system RAM (hybrid CPU+GPU offloading).
* Low-concurrency workloads (1 to ~15 users) where single-stream latency matters more than massive parallel batching.


* **Pros:**
* **Unmatched Hardware Portability:** Runs natively on x86/ARM CPUs (AVX/NEON), Apple Silicon (Metal), NVIDIA (CUDA), AMD (ROCm), Vulkan, and mobile NPUs.
* **Granular GGUF Quantization:** Single-file `GGUF` bundles let you pick exact precision levels (from 2-bit to 8-bit) to squeeze large models onto modest hardware.
* **Layer-Level CPU/GPU Offloading:** Can offload as many transformer layers to the GPU as VRAM allows and run the rest in system RAM.
* **Blazing Single-User Latency:** Minimal runtime overhead yields fast Time-to-First-Token (TTFT) and generation speeds for single streams.


* **Cons:**
* **Scales Poorly Under High Concurrency:** Uses static slot-based batching rather than true dynamic PagedAttention, causing throughput to plateau quickly above ~16–32 concurrent users.
* **Manual Compilation & Tuning:** Getting peak performance often requires compiling from source with hardware-specific flags (CMake) and manually tuning thread counts, batch sizes, and GPU layer splits.
* **GGUF Conversion Step:** Models from Hugging Face usually need to be converted and quantized into `.gguf` format before running (though pre-quantized GGUFs are widely hosted on the Hub).



---

## Quick Decision Rules

1. **Model fits completely in datacenter/high-end VRAM + serving many concurrent users?** $\rightarrow$ **vLLM**
2. **Deploying on NVIDIA GPUs + want a plug-and-play Docker container with Hugging Face tooling & metrics?** $\rightarrow$ **TGI**
3. **Running on a laptop, CPU server, edge device, or a GPU with too little VRAM for the full model?** $\rightarrow$ **llama.cpp**


## Questions

How do I know how many layers I could set for GPU and CPU, considering my local machine?

To determine how many layers you can offload to your GPU versus keeping on the CPU in `llama.cpp` (using the `--n-gpu-layers` or `-ngl` flag), you need to balance your model’s VRAM footprint against your available hardware limits.

Modern versions of `llama.cpp` feature an `auto` or fitted default, but you can precisely calculate or test the ideal split using the framework below.

---

### Step 1: Identify Your Core Variables

To figure out your budget, check three things:

1. **Total Model Layers:** Found in the model card on Hugging Face or printed in the terminal when you first attempt to load a GGUF model (e.g., standard 8B models usually have 32 transformer blocks + 1 output layer = 33 total layers; 14B models often have 40+ blocks).
2. **File Size & Per-Layer Size:** Divide the GGUF file size by the total layer count to get a rough **VRAM cost per layer**.
* *Example:* A 5 GB model split across 33 layers costs roughly **~150 MB per layer**.


3. **Available VRAM:** Check your GPU’s free VRAM (remember to subtract ~1 to 2 GB for your desktop environment/OS overhead and the model's KV cache context window buffer).

---

### Step 2: Choose Your Offload Strategy

#### Option A: Full Offload (Best Performance)

If your available VRAM is greater than `Model File Size + KV Cache Overhead` (typically needing 1–2 GB extra headroom), offload everything.

* **Setting:** `--n-gpu-layers 99` (or `-1`, or `all`).
* *Note:* Setting a number higher than the model's actual layer count automatically caps out safely at the maximum layer count.

#### Option B: Partial Offload (Hybrid CPU + GPU)

If the model file is larger than your VRAM, you can split the load. The GPU handles as many layers as it can fit, and the remaining layers fall back to system RAM and your CPU.

* **The Math Formula:**

$$\text{Target GPU Layers} = \frac{\text{Free VRAM (in MB)} - \text{Headroom (1500 MB)}}{\text{Size per Layer (in MB)}}$$


* **Setting:** Pass that calculated integer to the flag (e.g., `--n-gpu-layers 20`).
* *Performance Trade-off:* Partial offloading is significantly faster than running purely on the CPU, but slower than a full GPU offload because data must shuttle back and forth over the PCIe bus.

---

### Step 3: Use Built-In Testing Tools

Instead of guessing, you can use `llama.cpp`'s native tooling to benchmark different layer counts and find your hardware's breaking point:

1. **Let `llama.cpp` decide automatically:**
Run your command with `--n-gpu-layers auto` (the default in current builds) and watch the startup logs. The console will explicitly print how many layers were successfully offloaded to your device (e.g., `llm_load_tensors: offloaded 33/33 layers to GPU`).
2. **Benchmark performance steps:**
Use `llama-bench` to test tokens-per-second throughput across incremental layer jumps to see where performance drops:
```bash
llama-bench -m your_model.gguf -ngl 0,10,20,33

```