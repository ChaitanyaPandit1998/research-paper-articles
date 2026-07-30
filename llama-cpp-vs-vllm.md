# llama.cpp vs. vLLM, Demystified: Two Answers to "How Do You Actually Run a Trained Model?"

**Pull quotes:**
- "Training answers 'how do you make the weights good.' Inference engines answer a completely different question: once the weights are good, how do you get a token out the other end before the user gets bored?"
- "llama.cpp's entire design center is a laptop with 16GB of RAM and no GPU. vLLM's entire design center is a datacenter GPU serving a thousand people at once. Neither is a worse version of the other — they optimized for opposite scarcities."
- "PagedAttention didn't speed up attention. It sped up everything *around* attention, by admitting that a KV cache is just memory management wearing a machine-learning costume — and operating systems solved memory management decades ago."

---

A trained model is just a folder of numbers until something loads those weights, runs the forward pass token by token, and manages everything a forward pass needs to stay fast — the KV cache, the batch of concurrent requests, the precision of every multiply. llama.cpp and vLLM are the two most widely used answers to that problem, and they disagree about almost every design decision, because they were built to solve different scarcities: llama.cpp exists so a large model can run on hardware that doesn't have much of anything to spare; vLLM exists so a GPU that has plenty to spare doesn't waste a single byte of it while serving hundreds of people at once. This article works through what an inference engine actually has to do, how each project solves it, the numbers behind why their tradeoffs diverge so sharply, and where each one is — and isn't — the right tool.

---

## Table of contents

1. [How they came to be](#1-how-they-came-to-be)
2. [What problem an inference engine actually solves](#2-what-problem-an-inference-engine-actually-solves)
3. [The mechanisms in active use](#3-the-mechanisms-in-active-use)
4. [The numbers: memory and throughput](#4-the-numbers-memory-and-throughput)
5. [A runnable comparison](#5-a-runnable-comparison)
6. [What neither one solves](#6-what-neither-one-solves)
7. [Alternatives and relatives](#7-alternatives-and-relatives)
8. [Summary table](#8-summary-table)
9. [Key takeaways](#9-key-takeaways)
10. [Further reading](#10-further-reading)

---

## 1. How they came to be

**llama.cpp (March 2023).** Days after Meta released the original LLaMA weights — as a research-only download, runnable at the time only through a multi-gigabyte PyTorch/CUDA stack — Georg Gerganov ported the model to plain C/C++ with no external dependencies, built on top of his own tensor library, `ggml`. The pitch was almost provocative in its narrowness: run a 7-billion-parameter language model on a MacBook, in reasonable time, on the CPU alone. That required two things PyTorch inference at the time didn't prioritize — aggressive weight quantization (packing each parameter into 4 bits instead of 16) and a runtime with none of the memory and dependency overhead of a full Python/CUDA stack. It worked, spawned an entire ecosystem (GGUF as a portable model format, `llama-cpp-python` bindings, Ollama built directly on top of it), and did more than any single project to make "run a real LLM on your own laptop" an ordinary thing to do rather than a research demo.

**vLLM (June 2023).** A team at UC Berkeley's Sky Computing Lab, led by Woosung Kwon and Zhuohan Li, published *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* — the paper that both introduced vLLM and diagnosed why existing GPU serving stacks left so much throughput on the table: the KV cache (the running memory of every previous token's attention keys and values) was being allocated in large contiguous chunks per request, the same way early operating systems allocated memory before virtual memory and paging existed, and that led to the same disease — fragmentation and waste, in this case up to 60–80% of GPU memory sitting unused because it had been reserved for a worst-case sequence length that most requests never reached. vLLM's answer borrowed the OS solution directly: page the KV cache into small fixed-size blocks, the same way an operating system pages RAM, and let requests share and reclaim blocks dynamically instead of reserving a maximum up front.

> **Note.** Both projects are a response to the same underlying fact — a trained transformer is architecturally identical whether it runs on a phone or a GPU cluster — but they picked opposite constraints to optimize against. llama.cpp assumes compute and memory are scarce and a single user is waiting. vLLM assumes compute is provisioned and expensive, and the job is to never let it sit idle while dozens of users wait concurrently. Everything else in this article follows from that one fork in priorities.

---

## 2. What problem an inference engine actually solves

Training produces a static set of weights. Turning those weights into something that answers a prompt requires solving three problems that have nothing to do with backpropagation, loss functions, or activation functions (see the companion [backpropagation](backpropagation.md), [loss functions](loss-functions.md), and [activation functions](activation-functions.md) articles for that earlier half of the model's life) — inference is a different problem, with different bottlenecks entirely.

**Getting the weights into memory at all.** A 70-billion-parameter model at 16-bit precision is 140GB — more RAM or VRAM than most single machines have, let alone the RAM left over after the operating system and everything else. The weights have to either be shrunk (quantization) or spread across multiple devices (parallelism), or both, before a forward pass is even possible.

**Generating one token at a time, fast.** Unlike training, where a whole batch of sequences is scored in parallel against known targets, autoregressive generation is inherently sequential: token $t{+}1$ can't be predicted until token $t$ has been sampled and fed back in. Every generated token requires a full pass through every layer of the network, and unlike training's compute-bound matrix multiplications over large batches, decoding one token for one sequence is memory-bandwidth-bound — the GPU or CPU spends most of its time waiting to read weights from memory, not computing with them. This is precisely why quantization (shrinking how many bytes have to move) and batching (amortizing that memory traffic over many sequences at once) are the two levers every inference engine pulls.

**Not recomputing the past on every step.** Naively, generating token 500 of a sequence would require re-running the attention computation over all 499 prior tokens from scratch. Every practical inference engine avoids this by caching each prior token's attention keys and values — the **KV cache** — so each new token only computes attention against a cache lookup, not a full recomputation. That cache is the single largest consumer of memory during generation besides the weights themselves, and how an engine manages it is close to the whole story of what makes vLLM and llama.cpp different from each other.

$$\text{KV cache size (bytes)} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seqLen} \times \text{batch} \times \text{bytesPerValue}$$

The factor of 2 is for storing both keys *and* values; everything else scales linearly with how long the conversation has gotten and how many conversations are running at once — which is exactly why cache management, not raw FLOPs, is the resource both engines spend the most design effort protecting.

---

## 3. The mechanisms in active use

Three mechanisms, and the two engines make a different call on every one of them.

### Quantization: how small can a weight get before the model breaks

**Intuition first.** A weight trained and stored in 16-bit floating point carries far more precision than the model actually needs to produce a coherent next-token distribution. Quantization rounds each weight to a lower-precision representation — 8 bits, 4 bits, even lower — trading a small, usually-tolerable accuracy loss for a large, very-tolerable memory reduction.

llama.cpp built its entire identity around this trade. Its GGUF format ships models in a family of quantization schemes — Q8_0 (8-bit, near-lossless), Q4_K_M (4-bit, the most commonly used default, mixing precision per-tensor to protect the layers most sensitive to rounding), down to Q2_K (2-bit, aggressive, noticeably lossier) — computed once, offline, and baked into the file a user downloads. A 7B model that needs 14GB at 16-bit precision needs roughly 4GB at Q4_K_M, which is the entire reason a MacBook with 16GB of unified memory can run it at all.

vLLM supports quantization too (AWQ, GPTQ, and FP8 are all common in production vLLM deployments), but it isn't the project's center of gravity — a GPU serving deployment usually has enough VRAM that quantization is a throughput optimization, not the difference between "runs" and "doesn't run." The emphasis inverts: llama.cpp treats quantization as existential, vLLM treats it as one lever among several.

### KV cache management: contiguous allocation vs. paging

**Intuition first.** Every serving engine has to decide, before it knows how long a conversation will actually run, how much memory to set aside for that conversation's KV cache. Guess too high across many concurrent requests and most of that memory sits reserved and unused; guess too low and a long conversation runs out of cache mid-generation.

Pre-PagedAttention serving stacks (and llama.cpp's traditional design, built for a single interactive user rather than hundreds of concurrent strangers) allocate each sequence's KV cache as one contiguous block sized for a maximum sequence length. For a single local user, that waste is a rounding error against 16GB of RAM; for a GPU serving thousands of concurrent, wildly variable-length requests, Kwon et al. measured 60–80% of allocated KV-cache memory going unused this way.

**PagedAttention**, vLLM's core contribution, borrows the fix directly from operating-system virtual memory: split the KV cache into small fixed-size blocks (pages), maintain a per-sequence table mapping logical cache positions to physical blocks, and allocate physical blocks on demand as a sequence grows rather than reserving a worst-case span up front. Blocks aren't even required to be contiguous in physical memory — exactly the trick that lets an OS run more processes than it has contiguous free RAM for. The practical payoff is that far more concurrent sequences fit in the same GPU memory, which is the single biggest lever behind vLLM's throughput numbers.

```
Traditional (contiguous) KV cache          PagedAttention KV cache
──────────────────────────────────         ──────────────────────────────────
[seq A: reserved for max_len    ]          [seq A blocks: 3, 7, 12       ]
[seq B: reserved for max_len    ]          [seq B blocks: 1, 4           ]
[   ...mostly empty, unusable   ]          [free blocks: 0,2,5,6,8,9,... ]
by any other sequence           ]          shared/reclaimed on demand
```

### Batching: one request at a time vs. continuous batching

**Intuition first.** Because decoding is memory-bandwidth-bound, running the model for one sequence at a time wastes most of the GPU's or CPU's throughput — the same weights get streamed from memory to compute one token's worth of work, when they could serve many sequences' worth of work per pass at almost the same memory-traffic cost. Batching multiple sequences together amortizes that fixed memory-read cost over more useful compute.

llama.cpp's original design point was a single interactive session — one user, one conversation, request-response — and its architecture reflects that; `llama-server` has since added request batching, but it is not the axis the project was built to maximize. vLLM's defining scheduling trick is **continuous batching** (also called iteration-level scheduling): instead of waiting for an entire batch of requests to finish generating before starting the next batch (static batching, which stalls on whichever sequence in the batch happens to run longest), vLLM re-evaluates the batch composition at every single decoding step, injecting newly arrived requests and evicting completed ones mid-stream. No GPU cycle is spent waiting on a straggler, and a request that arrives mid-stream doesn't wait for the entire current batch to drain first.

> **Note.** These three mechanisms compound. Quantization determines how much memory the weights occupy, which determines how much memory is left over for KV cache pages, which determines how many concurrent sequences continuous batching can actually schedule at once. A production vLLM deployment tunes all three together; llama.cpp deliberately spends most of that budget on the first one, because for a single local user, the other two barely matter.

---

## 4. The numbers: memory and throughput

Two worked examples — why quantization is existential for local inference, and why paging changes concurrent-request capacity by an order of magnitude.

### Example 1 — Fitting a 7B Model on a Laptop

**Setup:** Llama-2-7B, compared at three precisions.

```
Precision      Bytes/param    Model size    Fits in 16GB unified memory?
──────────────────────────────────────────────────────────────────────
FP16           2              ~13.5 GB      Barely — no room for KV cache or OS
Q8_0           1              ~7.2 GB       Yes, comfortably
Q4_K_M         ~0.56          ~4.1 GB       Yes, with room to spare for context
```

At FP16, a 7B model alone consumes nearly all of a 16GB MacBook's unified memory, leaving nothing for the KV cache, the operating system, or any other running application — in practice, unusable. At Q4_K_M, the same model needs less than a third of that, which is the concrete, arithmetic reason "download a GGUF file and run it locally" became possible at all. This is not a marginal optimization; it's the difference between a model that runs on consumer hardware and one that categorically doesn't.

### Example 2 — Concurrent Requests Under Contiguous vs. Paged KV Cache

**Setup:** a GPU with 40GB free for KV cache, serving requests with a 2048-token maximum sequence length but a **512-token average actual length** — a realistic gap, since most conversations don't reach the maximum the server has to plan for. Assume each token's KV cache entry costs roughly 0.5MB for a mid-sized model (a simplified, representative figure).

```
Allocation strategy      Memory reserved per request      Requests that fit in 40GB
──────────────────────────────────────────────────────────────────────────────────
Contiguous (max_len)     2048 × 0.5MB = 1024 MB            ~39 concurrent requests
Paged (actual usage)     512  × 0.5MB = 256 MB              ~156 concurrent requests
```

Because contiguous allocation reserves for the worst case every request *might* reach, it wastes roughly 75% of the memory it holds, matching the 60–80% figure the PagedAttention paper measured empirically. Paging reclaims that waste directly into more concurrent capacity — a **4× increase in requests served from the same GPU memory**, with no change to the model, the hardware, or the quantization. This is the single number that explains why vLLM's throughput claims (the original paper reported 2–4× higher throughput than the serving stacks of the time) are structural, not a micro-optimization — they come from memory accounting, not from a faster kernel.

---

## 5. A runnable comparison

Both engines are typically driven through an OpenAI-compatible HTTP API, which makes the client-side code nearly identical — the difference is entirely in how each server is launched and what it optimizes underneath that shared interface.

```bash
# --- llama.cpp: single machine, local weights, quantized GGUF file ---
# Download a quantized model once, then serve it directly from disk.
./llama-server \
  -m llama-2-7b.Q4_K_M.gguf \
  -c 4096 \        # context window
  -ngl 32 \        # offload 32 layers to GPU if available, rest stays on CPU
  --port 8080

# --- vLLM: GPU server, full/quantized HF weights, throughput-tuned ---
# Downloads weights from the Hugging Face Hub and serves with continuous
# batching and PagedAttention on by default.
vllm serve meta-llama/Llama-2-7b-hf \
  --gpu-memory-utilization 0.90 \   # how much VRAM to dedicate to KV cache pages
  --max-num-seqs 256 \              # concurrent sequences continuous batching may schedule
  --port 8080
```

```python
# Both servers expose the same OpenAI-compatible chat completions endpoint —
# this client code is identical regardless of which engine is running underneath.
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="llama-2-7b",
    messages=[{"role": "user", "content": "Explain KV caching in one sentence."}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

```python
# vLLM also exposes a direct Python engine API for offline batch inference —
# useful when there's no need for a long-running server at all.
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf", quantization="awq")
params = SamplingParams(temperature=0.7, max_tokens=100)

# vLLM's continuous batching applies even here: passing a list lets it
# schedule all prompts together rather than looping one at a time.
prompts = ["Explain KV caching in one sentence.", "What is PagedAttention?"]
outputs = llm.generate(prompts, params)
for out in outputs:
    print(out.outputs[0].text)
```

The client-visible surface — an OpenAI-shaped chat completions call — is deliberately interchangeable; what differs is invisible from that vantage point: `-ngl 32` is llama.cpp deciding how to split a quantized model across a CPU/GPU boundary a laptop actually has, while `--max-num-seqs` is vLLM deciding how aggressively to pack PagedAttention's block table with concurrent strangers' requests. Same API contract, opposite resource assumptions underneath it.

---

## 6. What neither one solves

Objectivity requires noting that an inference engine, however well built, doesn't fix problems that live upstream of it.

**They don't fix a bad or under-trained model.** llama.cpp and vLLM serve whatever weights they're handed, quantized or not, exactly as faithfully as they were trained — an engine can make a bad model answer *faster*, never *better*. The loss function used to train it (see the companion [loss functions article](loss-functions.md)) already decided what "good" means for that model, long before either engine is involved.

**Quantization is not free — it's a measured, bounded accuracy loss.** Q4 quantization on most models loses a small amount of benchmark accuracy relative to FP16, and that loss grows as the bit-width drops further; below roughly 3 bits it becomes noticeable on harder tasks. Neither engine can quantize its way out of that tradeoff — it's a genuine cost, not an optimization with no downside, and the right operating point depends on the task, not a fixed rule.

**Neither one fixes a context-length or attention-complexity problem.** Both engines manage the KV cache efficiently, but neither changes the underlying $O(n^2)$ cost of attention over a sequence of length $n$ — a genuinely long context is expensive to serve under either engine, for the same architectural reason it's expensive to train over (see the companion [flash-attention notes](flash-attention.md) for the training/kernel-level version of this same cost).

**Throughput and latency are frequently in tension, and neither engine erases that tradeoff.** vLLM's continuous batching maximizes aggregate throughput across many concurrent users, which can (under heavy load) increase the latency any single user experiences relative to being served alone. llama.cpp's single-stream design gives one user the GPU or CPU's full, undivided attention, at the cost of serving exactly one user. Picking an engine is partly picking which side of that tradeoff a given deployment actually needs.

**Neither replaces correct hardware provisioning.** Quantization and paging both stretch existing memory further, but they don't manufacture memory that doesn't exist — a 400B-parameter model still needs either extreme quantization, multiple GPUs, or both, regardless of which engine loads it.

---

## 7. Alternatives and relatives

llama.cpp and vLLM are the two most widely used open-source inference engines, but the space is broader, and the differences between the alternatives track the same fork in priorities described in [Section 1](#1-how-they-came-to-be).

**Hugging Face Text Generation Inference (TGI).** A production serving stack built and maintained by Hugging Face, occupying similar territory to vLLM — continuous batching, tight integration with the Hub's model formats, tensor parallelism for multi-GPU serving. It predates vLLM's PagedAttention paper and has since adopted similar paged-cache ideas; the two are close competitors for the same GPU-serving use case rather than solving different problems.

**NVIDIA TensorRT-LLM.** A serving stack built directly on NVIDIA's TensorRT compiler, trading portability for the largest possible throughput on NVIDIA hardware specifically — kernels are compiled and fused ahead of time for a target GPU, which buys speed at the cost of losing the flexibility of a general PyTorch-based stack. It's the choice when the deployment is NVIDIA-only and every remaining percent of throughput is worth engineering effort to extract.

**SGLang.** A newer serving engine (2024) built around structured generation and complex prompting patterns (agent loops, constrained decoding, tool calls) as a first-class concern, with its own paged-cache and batching design in the same family as vLLM's — it targets workloads with more structure and reuse across requests than a simple chat completion.

**Ollama.** Built directly on top of llama.cpp (using it as the inference backend under a friendlier CLI and model-management layer) rather than a competing engine — it inherits llama.cpp's quantization-first, local-first design entirely, and mostly changes the distribution and packaging story, not the underlying mechanism.

**MLC-LLM / Apple MLX.** Compiler-based approaches that generate hardware-specific inference kernels (for mobile GPUs, WebGPU, Apple Silicon) rather than shipping one general-purpose runtime — closer in spirit to llama.cpp's "run anywhere with modest resources" goal, but achieved by compiling per-target instead of by a single portable C++ codebase.

**None of these have displaced the llama.cpp/vLLM split** because it isn't really a technology gap — it's two different deployment scenarios (one user with limited hardware; many users with provisioned hardware) that will keep needing genuinely different engineering answers regardless of how good either engine gets at the other's job.

---

## 8. Summary table

| Concept | llama.cpp | vLLM |
|---|---|---|
| Design center | Single user, limited/no GPU, local hardware | Many concurrent users, provisioned GPU, datacenter |
| Origin | Gerganov, March 2023 — C/C++ port of LLaMA inference | Kwon et al., UC Berkeley, June 2023 — PagedAttention paper |
| Primary lever pulled | Aggressive quantization (GGUF, down to 2–4 bit) | KV cache paging + continuous batching |
| KV cache strategy | Contiguous per-sequence allocation | Paged, block-based, OS-style virtual memory |
| Batching | Originally single-stream; server mode adds basic batching | Continuous (iteration-level) batching by default |
| Typical hardware | CPU, Apple Silicon (Metal), consumer GPU | Datacenter GPU (NVIDIA, AMD), multi-GPU tensor parallel |
| What it optimizes for | Fits in limited memory, runs with no dependencies | Maximum requests served per GPU-hour |
| Model format | GGUF (quantized, self-contained file) | Hugging Face safetensors, optionally quantized |

---

## 9. Key takeaways

- **Both engines run the same trained transformer — they differ entirely in the resource they assume is scarce.** llama.cpp assumes memory and compute are scarce and a single user is waiting; vLLM assumes GPU memory is provisioned and the job is to never let it idle.
- **Quantization is llama.cpp's central bet, and it's an existential one.** A 7B model shrinks from ~13.5GB at FP16 to ~4GB at Q4_K_M — the difference between "runs on a laptop" and "doesn't fit at all," not a marginal speedup.
- **PagedAttention is a memory-management fix borrowed from operating systems, not a faster attention kernel.** By paging the KV cache instead of reserving worst-case contiguous blocks, vLLM reclaims the 60–80% of memory that contiguous allocation wastes, directly turning into more concurrent requests served per GPU.
- **Continuous batching is what makes that reclaimed memory actually useful.** Re-scheduling the batch at every decoding step, rather than waiting for the slowest sequence in a static batch, is what converts "more memory available" into "more throughput delivered."
- **Neither engine changes what the model knows or how it was trained.** They serve whatever weights they're handed, faster or leaner — the model's actual capability was decided upstream, by the architecture, the loss function, and the training data.
- **The right choice is a deployment question, not a quality question.** "Which engine is better" doesn't have a fixed answer; "one user on a laptop" and "a thousand users hitting an API" are different problems, and each engine is close to optimal for the one it was built to solve.

---

## 10. Further reading

- **"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (Kwon et al., 2023) — the paper that introduced vLLM and PagedAttention, and measured the 60–80% KV-cache waste under contiguous allocation referenced in Section 1 and Section 4: arxiv.org/abs/2309.06180

- **The llama.cpp GitHub repository** — the C/C++ inference engine and `ggml` tensor library described in Section 1, including the GGUF format specification and the k-quant quantization schemes covered in Section 3: github.com/ggml-org/llama.cpp

- **The vLLM GitHub repository and documentation** — the reference implementation of continuous batching and PagedAttention, including the `vllm serve` CLI and Python `LLM` engine API used in Section 5: github.com/vllm-project/vllm, docs.vllm.ai

- **"How is LLaMA.cpp possible?"** (Finbarr Timbers, 2023) — a widely cited technical breakdown of why quantized CPU inference of a multi-billion-parameter model is arithmetically feasible at all: finbarr.ca/how-is-llama-cpp-possible

- **The GGUF format specification** — the file format llama.cpp uses to package quantized weights, metadata, and tokenizer into one portable file: github.com/ggml-org/ggml/blob/master/docs/gguf.md

- **"Orca: A Distributed Serving System for Transformer-Based Generative Models"** (Yu et al., OSDI 2022) — the earlier paper that first introduced iteration-level (continuous) batching, which vLLM's scheduler builds directly on: usenix.org/conference/osdi22/presentation/yu

---

Training and inference are the same model's two lives, and they barely resemble each other. Training spends its effort on backpropagation and a loss function, chasing a gradient through a graph it's allowed to hold entirely in memory because it already knows every target in advance. Inference gets none of that luxury — it has to produce one token at a time, from a model it can't change, on hardware it usually can't upgrade, for users who are actually waiting. llama.cpp and vLLM are what that second problem looks like once it's taken seriously: one engine shrinking a model down to fit where almost nothing fits, the other stretching a GPU's memory to serve almost everyone at once — two different scarcities, two different disciplines, the same weights underneath both.
