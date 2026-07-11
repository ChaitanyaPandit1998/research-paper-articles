# Flash Attention, Demystified: Why the Fastest Attention Kernel Does Exactly the Same Math

**Pull quotes:**
- "Flash Attention doesn't approximate anything. It computes the exact same numbers as standard attention — it just refuses to write them to slow memory along the way."
- "The attention bottleneck was never the matrix multiplication. It was 16 million numbers being written to memory and then immediately read back."
- "SRAM is 100x smaller than HBM and 6-7x faster. Flash Attention's entire trick is refusing to leave that fast, tiny neighborhood until the answer is done."

---

You double your training sequence length from 4K to 8K tokens — same GPU, same model, same batch size — and training crashes with a CUDA out-of-memory error. Not 2x the memory pressure you braced for. Closer to 4x. That's not a leak or a misconfiguration; it's the $n \times n$ attention score matrix growing quadratically while everything else about your job barely changed. Flash Attention exists because of exactly that moment.

Flash Attention is one of the few systems-level ideas in deep learning that changed what was practically buildable — not by inventing new math, but by refusing to waste memory bandwidth on old math. This article works through what the standard attention bottleneck actually is, why it's an I/O problem rather than a compute problem, how tiling and online softmax solve it exactly (not approximately), and where Flash Attention v1 through v3 and Sliding Window Attention sit in the models you use today.

---

## Table of contents

1. [What is it?](#1-what-is-it)
2. [Use case](#2-use-case)
3. [Different attention implementations actively used](#3-different-attention-implementations-actively-used)
4. [Core mechanics for each](#4-core-mechanics-for-each)
5. [Explanation for each](#5-explanation-for-each)
6. [Python example](#6-python-example)
7. [Pros and cons](#7-pros-and-cons)
8. [Summary](#8-summary)
9. [Key takeaways](#9-key-takeaways)
10. [Appendix: the details behind the claims](#10-appendix-the-details-behind-the-claims)
11. [Further reading](#11-further-reading)

---

## 1. What is it?

Flash Attention is an implementation of self-attention that produces numerically identical outputs to standard attention, but runs several times faster and uses a fraction of the memory — by changing *how* the computation touches GPU memory, not *what* it computes.

To see why this was necessary, look at what standard attention does. For queries $Q$, keys $K$, and values $V$, each of shape $(n, d)$ for sequence length $n$ and head dimension $d$:

$$S = \frac{QK^T}{\sqrt{d}}, \qquad P = \text{softmax}(S), \qquad O = PV$$

The problem is $S$. It's an $n \times n$ matrix — every token's score against every other token. For a sequence of 4,096 tokens, that's 16 million numbers. For 32K tokens, it's over a billion. A standard implementation materializes this full matrix: writes it out to GPU memory, reads it back to run softmax, writes the result out, reads it back again to multiply by $V$.

Here's the part that surprises people the first time they see it: none of that is expensive because of the *arithmetic*. Modern GPUs can do trillions of floating-point operations per second — the multiplications and additions in $QK^T$ and $PV$ finish almost instantly. What's expensive is moving $S$ back and forth between the GPU's two kinds of memory:

- **HBM (High Bandwidth Memory)** — the large pool of GPU RAM (40-80GB on an A100/H100). Big, but relatively slow to access (~1.5-3 TB/s).
- **SRAM** — a tiny on-chip cache (tens of megabytes) sitting right next to the compute cores. Small, but roughly an order of magnitude faster (~19 TB/s).

Standard attention reads and writes the full $n \times n$ matrix to HBM multiple times over. The GPU's compute cores spend most of their time idle, waiting for data to arrive from HBM rather than actually computing. This is the textbook definition of a **memory-bandwidth-bound** operation, as opposed to a **compute-bound** one — the bottleneck is *I/O*, not FLOPs. Flash Attention's entire contribution is restructuring the computation so the full $n \times n$ matrix is never written to HBM at all.

---

## 2. Use case

Flash Attention lives in exactly one place in a transformer: the self-attention (or cross-attention) computation inside each transformer block, right where $Q$, $K$, and $V$ get combined into an output. It's a drop-in replacement — same inputs, same outputs, same math — for the `softmax(QK^T/√d)V` step. Everything else in the block (layer norm, the feed-forward network, residual connections) is untouched.

Why it matters in practice:

- **Training speed.** Because the GPU is no longer stalled waiting on HBM traffic, wall-clock training time drops substantially — commonly 2-4x faster attention, translating into meaningfully faster end-to-end training for attention-heavy models.
- **Memory usage.** Standard attention needs $O(n^2)$ memory just to hold the score matrix (and its gradient, during backprop). Flash Attention needs $O(n)$ memory, because it never materializes the full matrix — it recomputes small pieces on the fly during the backward pass instead of storing them.
- **Longer context lengths.** This is the big unlock. Because memory no longer scales quadratically with sequence length, models can be trained and run at context lengths (16K, 32K, 128K+ tokens) that would simply run out of GPU memory with standard attention. Long-context LLMs are, in large part, a Flash-Attention-shaped consequence.

**Prefill vs. decode: two different bottlenecks.** Everything above describes *training* and *prefill* (processing a full prompt at once) — many query tokens against many keys, so the $n \times n$ score matrix is the problem. Autoregressive *decoding* (generating one token at a time) is a different regime: a single query token attends against a growing cache of all previous keys/values. There's no large score matrix to avoid — the bottleneck shifts to reading the entire KV cache from HBM for every single new token. This is why `flash-attn` ships a separate `flash_attn_with_kvcache` function: it's still about minimizing HBM traffic, just against a cache instead of a freshly-computed block.

**Pairing with GQA/MQA.** Grouped-query attention (GQA) and multi-query attention (MQA) — used in Llama 3, Mistral, and Qwen — shrink the KV cache itself by sharing key/value heads across multiple query heads. This is a complementary optimization, not a competing one: GQA/MQA reduces *how much* KV data has to be read during decode, while Flash Attention reduces the *cost per byte* of reading it. Nearly every modern open-weight LLM ships with both simultaneously.

**The memory savings in actual numbers.** Both approaches need $O(nd)$ memory for $Q$, $K$, $V$, and the output — that part is identical. The difference is the *extra* memory each one needs on top of that. Standard attention's extra cost is the $n \times n$ score matrix; Flash Attention's extra cost is just two running statistics ($m$, $\ell$) per query row, i.e. $O(n)$. For a single attention head, fp16, head dim $d=64$:

| Sequence length $n$ | Standard attention's extra memory ($n^2$ score matrix) | Flash Attention's extra memory ($m,\ell$ running stats) |
|---|---|---|
| 4,096 | ~32 MB | ~32 KB |
| 32,768 | ~2 GB | ~256 KB |
| 131,072 (128K) | ~32 GB | ~1 MB |

That's *per head, per batch item* — a real model with, say, 32 heads and a batch of 8 multiplies the standard-attention column by 256, which is exactly why 128K-context standard attention isn't just slow, it's usually impossible to fit in GPU memory at all, while Flash Attention's extra cost stays in the kilobytes-to-megabytes range regardless of context length.

**When you might not need it.** For short sequences (a few hundred tokens) the $n^2$ score matrix is small enough that it fits comfortably in memory and the HBM round-trips it causes are cheap in absolute terms — the naive implementation is simpler to read, debug, and modify, and the speed difference versus Flash Attention is negligible. The crossover where Flash Attention's advantage becomes decisive is once $n$ climbs into the low thousands and beyond; below that, optimizing this particular op usually isn't where your time is best spent.

---

## 3. Different attention implementations actively used

The lineage, roughly in order of adoption:

| Implementation | Introduced | Idea |
|---|---|---|
| **Standard / vanilla attention** | Original transformer (2017) | Materialize the full $n \times n$ score matrix, softmax it, multiply by $V$. |
| **Memory-efficient attention** | 2021 (Rabe & Staats) | Avoid storing the full matrix using a chunked, sequential softmax — a precursor to Flash Attention's core idea, without the GPU-specific kernel engineering. |
| **Flash Attention (v1)** | 2022 (Dao et al.) | Tiling + online softmax, fused into a single custom CUDA kernel. First implementation to make the HBM-avoidance idea fast in practice. |
| **Flash Attention 2** | 2023 | Better parallelization (across sequence length, not just batch/heads) and reduced non-matmul work. ~2x faster than v1. |
| **Flash Attention 3** | 2024 | Redesigned for Hopper (H100) GPUs: asynchronous data movement overlapped with compute, low-precision (FP8) tensor core paths, warp specialization. |
| **Sliding Window Attention (SWA)** | Used in Mistral, Longformer, BigBird | Not a competing kernel — a masking pattern layered *on top of* Flash Attention. Each token only attends within a fixed local window, trading global context for further memory and compute savings at long sequence lengths. |

Who uses what, roughly:

- **GPT-2** and other older/reference implementations: standard attention (no fused kernel).
- **Llama 2/3, Mistral, Qwen, most modern open-weight LLMs**: Flash Attention 2 as the default training and inference kernel (via the `flash-attn` package or PyTorch's `scaled_dot_product_attention`).
- **Mistral 7B specifically**: Flash Attention *combined with* Sliding Window Attention — most layers use a local window, letting the model handle long sequences cheaply while information still propagates across the full context over multiple layers.
- **H100-class deployments (e.g., recent frontier-lab training stacks)**: increasingly Flash Attention 3, to exploit Hopper's asynchronous copy engines and FP8 tensor cores.
- **PyTorch itself**: `torch.nn.functional.scaled_dot_product_attention` auto-dispatches to a Flash-Attention-style fused kernel when the hardware and dtypes support it, falling back to a memory-efficient or math kernel otherwise.

---

## 4. Core mechanics for each

### Standard attention

$$S = \frac{QK^T}{\sqrt{d}} \in \mathbb{R}^{n \times n}$$
$$P_{ij} = \frac{e^{S_{ij} - \max_j S_{ij}}}{\sum_j e^{S_{ij} - \max_j S_{ij}}} \quad \text{(row-wise softmax)}$$
$$O = PV$$

Steps, with memory traffic made explicit:

1. Load $Q, K$ from HBM → compute $S = QK^T/\sqrt{d}$ → **write $S$ to HBM**.
2. **Read $S$ back from HBM** → compute row max and softmax → **write $P$ to HBM**.
3. **Read $P$ and $V$ back from HBM** → compute $O = PV$ → write $O$ to HBM.

Total HBM traffic scales as $O(n^2)$ — dominated by writing and re-reading the $n \times n$ intermediate matrices.

### Flash Attention (v1/v2/v3 share the same core algorithm)

The trick: never form $S$ or $P$ in full. Split $K$ and $V$ into blocks along the sequence dimension, and for each block of $Q$, sweep through the $K$/$V$ blocks, maintaining a **running (online) softmax** that is mathematically equivalent to computing softmax over the full row — without ever holding the full row in memory.

For a query block against key/value block $j$, keep three running statistics per query row: the running max $m$, the running sum of exponentials $\ell$, and the running weighted output $O$. On seeing a new block:

$$S^{(j)} = \frac{Q K_j^T}{\sqrt{d}}$$
$$m^{\text{new}} = \max\!\left(m^{\text{old}},\ \text{rowmax}(S^{(j)})\right)$$
$$P^{(j)} = e^{S^{(j)} - m^{\text{new}}}$$
$$\ell^{\text{new}} = e^{m^{\text{old}} - m^{\text{new}}} \ell^{\text{old}} + \text{rowsum}(P^{(j)})$$
$$O^{\text{new}} = e^{m^{\text{old}} - m^{\text{new}}} O^{\text{old}} + P^{(j)} V_j$$

After the last block, normalize: $O_{\text{final}} = O^{\text{new}} / \ell^{\text{new}}$.

The rescaling terms $e^{m^{\text{old}} - m^{\text{new}}}$ are what let the running sum and running output be corrected retroactively every time a new block reveals a larger max — this is the "online softmax" trick, and it's what makes the result *exact*, not approximate.

All of this — loading $Q_i, K_j, V_j$ blocks, computing $S^{(j)}$, updating $m, \ell, O$ — happens **inside SRAM**, for one $(Q_i, K_j)$ block pair at a time, fused into a single kernel. Only the final normalized output block gets written back to HBM. The full $n \times n$ matrix never exists in HBM.

**Backward pass:** rather than storing $P$ (which would cost $O(n^2)$ memory), Flash Attention stores only $O$, $m$, and $\ell$ (each $O(n)$), and *recomputes* the needed blocks of $S$ and $P$ on the fly during backprop — a modest increase in FLOPs in exchange for a large reduction in memory, a good trade since the operation is memory-bound, not compute-bound.

*(Want to see the tiling grid drawn out, the HBM-access lower-bound proof, the exact FLOPs overhead this recomputation costs, and a fully worked numeric example that checks the math by hand? They're in the [Appendix](#10-appendix-the-details-behind-the-claims) — split out so the core algorithm above stays readable in one pass.)*

**What changes across v1 → v2 → v3** is not this core math — it's engineering around it:

- **v2**: parallelizes across the sequence dimension in addition to batch/heads (better GPU utilization when batch size or head count is small), reduces the number of non-matmul operations (rescaling, bookkeeping), and reduces communication between warps *within* a thread block so less time is spent on shared-memory reads/writes relative to actual compute.
- **v3**: targets Hopper (H100) specifically — overlaps HBM-to-SRAM data movement with compute using asynchronous copy instructions (so the GPU is never just waiting), adds an FP8 low-precision path through the tensor cores for further throughput, and uses warp specialization (dedicating different groups of GPU threads to loading vs. computing vs. softmax bookkeeping, running concurrently rather than sequentially).

### Sliding Window Attention (layered on top)

Same tiling/online-softmax machinery, but the block sweep is truncated: query position $i$ only visits key/value blocks that fall within $[i - w, i]$ instead of $[0, i]$. This turns per-token attention cost from $O(n)$ into $O(w)$, and total attention cost from $O(n^2)$ into $O(n \cdot w)$.

---

## 5. Explanation for each

### Standard attention — the intuition

Imagine grading a class of $n$ students against $n$ possible essay prompts, by writing out the entire $n \times n$ grade table on a whiteboard before doing anything else — even though you only ever need one row summarized at a time. Most of the effort goes into writing and re-reading that giant table, not the actual grading arithmetic.

Formally: this is exactly the $S = QK^T$, softmax, $PV$ pipeline above — precise and simple, but wasteful, because the full table gets written to slow storage (HBM) and read back multiple times before you get your answer.

### Flash Attention — the intuition

Imagine instead reading a very long book to answer "what's this book about?" — but you have a tiny sticky note (SRAM) instead of a full desk to spread pages out on (HBM). You read a chunk of pages, update a short running summary on the sticky note, discard the pages, read the next chunk, update the summary again — and so on to the end. At no point do you have the whole book laid out at once, yet your final summary is exactly what you'd get from reading everything at once, *if* you're careful about how you merge each new chunk into the running summary.

That "being careful" is the online-softmax rescaling step ($e^{m^{\text{old}} - m^{\text{new}}}$ above): every time a new chunk reveals a bigger extreme value than anything seen so far, you retroactively adjust your running total so the final answer comes out mathematically identical to having seen everything at once. Nothing is approximated — the running summary is just built incrementally instead of all at once.

**Why the GPU memory hierarchy is the real reason this is faster:** SRAM is ~10-20x faster than HBM but also roughly 100-1000x smaller. Standard attention doesn't fit in SRAM, so it's forced to stage data through HBM repeatedly — and HBM bandwidth, not the GPU's raw FLOP throughput, becomes the limiting factor. Flash Attention's tiling is sized specifically so each block of $Q$, $K$, $V$ *does* fit in SRAM. The arithmetic performed is the same (in fact, tiling means a little redundant recomputation happens in the backward pass) — but because almost everything now happens in the fast, on-chip memory instead of bouncing through HBM, the GPU's compute cores stay busy instead of idle. Same FLOPs, far less waiting.

---

## 6. Python example

The easiest way to get Flash Attention today is PyTorch's `scaled_dot_product_attention` (SDPA), which dispatches to a fused Flash-Attention-style kernel automatically when hardware and dtypes allow it. Compare it to a naive, manual implementation of the same math:

```python
import torch
import torch.nn.functional as F

B, H, N, D = 2, 8, 1024, 64  # batch, heads, seq_len, head_dim
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

# Naive: materializes the full (N, N) score matrix
def naive_attention(q, k, v):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale       # (B, H, N, N) — the bottleneck
    probs = torch.softmax(scores, dim=-1)
    return probs @ v

# Flash Attention: same math, fused kernel, no (N, N) materialization
def flash_attention(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

out_naive = naive_attention(q, k, v)
out_flash = flash_attention(q, k, v)
print(torch.allclose(out_naive, out_flash, atol=1e-2))  # True — same result
```

If the `flash-attn` package is installed directly (common for Llama/Mistral-style training stacks), the equivalent call is:

```python
from flash_attn import flash_attn_func

# q, k, v shaped (batch, seq_len, num_heads, head_dim) — note the layout differs from SDPA
out = flash_attn_func(q, k, v, causal=True)
```

Both calls compute identical attention outputs to `naive_attention`; the difference is entirely in memory traffic and speed, which becomes visible once $N$ grows into the thousands and the naive version starts allocating gigabytes for the score matrix alone.

In practice, most engineers never call `flash_attn_func` directly — they enable it through the model library they're already using. In Hugging Face `transformers`, it's a single constructor argument:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    attn_implementation="flash_attention_2",   # or "sdpa" for PyTorch's built-in kernel
    torch_dtype="bfloat16",
).to("cuda")
```

`attn_implementation="flash_attention_2"` requires the `flash-attn` package to be installed and swaps in the fused kernel for every attention layer in the model; `"sdpa"` uses PyTorch's built-in dispatch instead (no extra install, slightly less optimized than a hand-tuned `flash-attn` build, but portable). Both replace every attention call in the model with the mechanics described above — nothing else about the model changes.

---

## 7. Pros and cons

**Standard / vanilla attention**
- ✅ Simple to implement and reason about; easy to debug, inspect, or modify (e.g., custom masks, attention visualization).
- ✅ No special hardware or kernel dependencies.
- ❌ $O(n^2)$ memory for the score matrix — becomes prohibitive past a few thousand tokens.
- ❌ Memory-bandwidth bound: GPU compute sits idle waiting on HBM traffic.

**Flash Attention v1**
- ✅ Exact attention — identical output to standard attention, no approximation.
- ✅ $O(n)$ memory instead of $O(n^2)$; substantial wall-clock speedup.
- ❌ Requires the custom CUDA kernel (via `flash-attn` package); not pure PyTorch, so portability and installation friction exist.
- ❌ Limited GPU utilization when batch size × heads is small, since parallelization is only across batch/heads.
- ❌ Only supports fp16/bf16 inputs, not fp32 — models running in full precision need a cast before calling in.

**Flash Attention v2**
- ✅ Better parallelization (adds sequence-length parallelism) and less bookkeeping overhead than v1 — roughly 2x faster.
- ✅ Same exactness and memory guarantees as v1.
- ❌ Still a specialized kernel with GPU/driver version requirements; benefits are smaller on older GPUs (pre-Ampere).

**Flash Attention v3**
- ✅ Substantial further speedup on H100/Hopper via async compute-memory overlap, FP8 support, and warp specialization.
- ✅ Exact attention preserved (FP8 path trades some numerical precision for throughput, by design, when enabled).
- ❌ Hopper-specific — gains don't transfer to older GPU architectures; narrower hardware support than v2.
- ❌ Most complex implementation of the three; fewer frameworks have fully integrated it yet.

**Sliding Window Attention (on top of Flash Attention)**
- ✅ Reduces attention cost from $O(n^2)$ to $O(n \cdot w)$, enabling very long sequences with a fixed local compute/memory budget.
- ✅ Composes cleanly with Flash Attention's tiling — no separate kernel needed, just a restricted mask.
- ❌ Not globally exact in a single layer — a token literally cannot attend beyond its window in that layer (long-range dependencies must propagate through depth, across multiple layers).
- ❌ Choosing window size is a tradeoff that needs tuning per task; too small a window can hurt tasks that need genuine long-range recall.

---

## 8. Summary

| Implementation | Memory complexity | Exact attention? | Typical use case | Key tradeoff |
|---|---|---|---|---|
| Standard attention | $O(n^2)$ | Yes | Reference implementations, short sequences, education/debugging (GPT-2-era) | Simple, but doesn't scale — quadratic memory and HBM-bound. |
| Memory-efficient attention | $O(n)$ | Yes | Precursor / fallback when fused kernels unavailable | Avoids materializing full matrix, but no GPU-specific kernel fusion — slower than Flash Attention. |
| Flash Attention v1 | $O(n)$ | Yes | Early adopters, Ampere-class GPUs | First exact, fused, HBM-avoiding kernel — big win, limited parallelism. |
| Flash Attention v2 | $O(n)$ | Yes | Default for Llama, Mistral, Qwen and most modern LLM training/inference | ~2x over v1 via better parallelization; still needs kernel support. |
| Flash Attention v3 | $O(n)$ | Yes (FP8 path trades precision) | H100/Hopper training and inference | Fastest, but hardware-specific (Hopper only). |
| Sliding Window Attention | $O(n \cdot w)$ | Exact within window, not globally | Mistral, long-context models needing cheap local attention | Trades global per-layer context for further memory/compute savings. |

---

## 9. Key takeaways

- Flash Attention computes **exactly** the same numbers as standard attention — it is purely an I/O and memory-layout optimization, not an approximation.
- The bottleneck it solves is **memory bandwidth**, not FLOPs: standard attention wastes most of its time writing and re-reading the $n \times n$ score matrix to slow HBM instead of computing.
- The core mechanism is **tiling + online softmax**: process $K$/$V$ in small blocks that fit in fast on-chip SRAM, maintaining a running max, running sum, and running output that get exactly rescaled as each new block arrives.
- Memory drops from $O(n^2)$ to $O(n)$, which is what actually enables today's long-context models — the win isn't just speed, it's that training and inference at 32K-128K+ tokens becomes feasible at all.
- v1 → v2 → v3 is a story of engineering, not new math: better parallelization (v2), then hardware-specific tricks like async data movement, FP8, and warp specialization for Hopper GPUs (v3).
- Sliding Window Attention is a complementary technique, not a competing one — it composes with Flash Attention's tiling to push memory/compute down further, at the cost of needing multiple layers for information to travel beyond the local window.

---

## 10. Appendix: the details behind the claims

This section exists for readers who want the receipts behind section 4's claims — skippable on a first read.

**The tiling grid, visually.** Split $Q$ into row-blocks and $K$/$V$ into column-blocks; the kernel sweeps one $(Q_i, K_j)$ tile at a time, entirely inside SRAM:

```
                K_1     K_2     K_3     K_4
              ┌───────┬───────┬───────┬───────┐
        Q_1   │ visit │ visit │ visit │ visit │  → running (m, ℓ, O) updated 4x, then O_1 written to HBM once
              ├───────┼───────┼───────┼───────┤
        Q_2   │ visit │ visit │ visit │ visit │  → same, independently, for row-block 2
              ├───────┼───────┼───────┼───────┤
        Q_3   │ visit │ visit │ visit │ visit │
              ├───────┼───────┼───────┼───────┤
        Q_4   │ visit │ visit │ visit │ visit │
              └───────┴───────┴───────┴───────┘

Causal masking: tiles strictly above the diagonal are skipped entirely
(not computed then masked) — a free speedup FA2 added on top of v1.
```

Each cell is a small matmul that fits in SRAM; only the four output row-blocks ($O_1$–$O_4$) ever get written back to HBM — never the 4×4 grid of raw scores. Tile size itself — how many tokens fit in one block — is chosen by the kernel at compile/launch time based on the GPU's SRAM capacity and the head dimension $d$; it isn't something you set by hand when calling `flash_attn_func` or SDPA.

**Why this is provably optimal, not just empirically faster.** The FA1 paper doesn't just show speedups — it proves a lower bound. Standard attention needs $\Theta(nd + n^2)$ HBM accesses (dominated by the $n^2$ term for long sequences). Flash Attention needs $\Theta(n^2d^2/M)$ accesses, where $M$ is the SRAM size — and the paper shows no exact attention algorithm can do asymptotically better across all values of $M$. In other words, tiling to the SRAM budget isn't a heuristic that happens to help; it's close to the theoretical floor for how little data an exact attention computation can move through HBM.

**How much extra compute the backward pass costs, roughly.** A standard backward pass reuses the $P$ matrix already sitting in memory from the forward pass, so its cost is on the order of the usual "backward is ~2x forward" rule of thumb — call forward $1\times$ and backward $2\times$, for $3\times$ total. Flash Attention's backward pass has to redo the $QK^T$-and-softmax work to reconstruct $P$ before it can use it, adding roughly one extra forward-sized pass on top of that same $2\times$ backward cost — around $1\times + 2\times + 1\times = 4\times$ total, versus $3\times$ for standard attention. That's roughly 30% more total FLOPs across the forward and backward pass combined — a derived estimate, not a quoted benchmark figure. It's a deliberate trade: extra arithmetic is nearly free on a GPU that's otherwise waiting on HBM, so paying ~30% more compute to avoid ever storing an $O(n^2)$ matrix is a clear net win once $n$ is large.

**A worked example with real numbers.** Take one query row with two key/value blocks of size 2, unscaled scores $S^{(1)} = [1, 3]$ for block 1 and $S^{(2)} = [2, 5]$ for block 2 (values chosen small so the arithmetic is easy to follow by hand):

*Block 1:* $m^{\text{new}} = \max(-\infty, 3) = 3$. $P^{(1)} = e^{[1,3]-3} = [e^{-2}, e^{0}] = [0.135,\ 1.0]$. $\ell^{\text{new}} = 0 + (0.135+1.0) = 1.135$. $O^{\text{new}} = P^{(1)}V_1$ (say $V_1 = [10, 20]$ elementwise-paired) $= 0.135(10) + 1.0(20) = 21.35$.

*Block 2:* rowmax of $S^{(2)}$ is $5 > 3$, so $m^{\text{new}} = 5$. Rescale factor $e^{m^{\text{old}}-m^{\text{new}}} = e^{3-5} = e^{-2} = 0.135$. $P^{(2)} = e^{[2,5]-5} = [e^{-3}, e^{0}] = [0.050,\ 1.0]$. $\ell^{\text{new}} = 0.135(1.135) + (0.050+1.0) = 0.153 + 1.050 = 1.203$. With $V_2 = [30, 40]$: $O^{\text{new}} = 0.135(21.35) + (0.050(30) + 1.0(40)) = 2.88 + 41.50 = 44.38$.

*Final:* $O_{\text{final}} = 44.38 / 1.203 \approx 36.9$.

Check against computing softmax over all four scores at once — $[1,3,2,5]$, max $=5$: $P = e^{[1,3,2,5]-5} = [0.018,\ 0.135,\ 0.050,\ 1.0]$, $\sum P = 1.203$, weighted sum against $[10,20,30,40] = 0.018(10)+0.135(20)+0.050(30)+1.0(40) = 0.18+2.70+1.50+40.0 = 44.38$, divided by $1.203 \approx 36.9$. Same answer — the block-by-block version with rescaling reproduces the single-pass softmax exactly, which is the entire point.

---

## 11. Further reading

- Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* — the original paper; introduces tiling, online softmax, and the HBM-access lower bound.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*
- Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.*
- Rabe, M. N., & Staats, C. (2021). *Self-attention Does Not Need $O(n^2)$ Memory.* — the memory-efficient attention precursor.
- Jiang, A. Q., et al. (2023). *Mistral 7B.* — introduces the Sliding Window Attention + rolling KV cache combination used alongside Flash Attention.
- Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* — the grouped-query attention technique paired with Flash Attention in Llama 3, Mistral, and Qwen.
