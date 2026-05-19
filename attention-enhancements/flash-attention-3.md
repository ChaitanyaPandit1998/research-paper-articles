# Flash Attention 3 (FA3)

**What it is:** A highly optimised implementation of attention that keeps data in fast GPU memory (SRAM) instead of slow memory (HBM), dramatically speeding up attention computation on H100 GPUs.

**Code:** `gpt.py:104-121`

---

## The Memory Bottleneck in Standard Attention

### Where GPU memory lives

A GPU has two types of memory:

```
HBM (High Bandwidth Memory)  — the large GPU RAM (e.g. 80 GB on H100)
  → Slow to access: ~3 TB/s bandwidth
  → Far from compute cores

SRAM (Static RAM / on-chip cache) — tiny, on the chip itself (e.g. 20 MB)
  → Fast to access: ~20 TB/s bandwidth
  → Right next to compute cores (6-7× faster than HBM)
```

### What standard attention does (the slow way)

For attention `softmax(Q·Kᵀ/√d) · V`:

```
Step 1: Load Q from HBM → SRAM                 ← slow HBM read
Step 2: Load K from HBM → SRAM                 ← slow HBM read
Step 3: Compute S = Q·Kᵀ                       ← fast, in SRAM
Step 4: Write S back to HBM                    ← slow HBM write
Step 5: Load S from HBM again                  ← slow HBM read
Step 6: Compute A = softmax(S)                 ← fast, in SRAM
Step 7: Write A back to HBM                    ← slow HBM write
Step 8: Load A from HBM, Load V from HBM       ← slow HBM reads
Step 9: Compute output = A·V                   ← fast, in SRAM
Step 10: Write output to HBM                   ← slow HBM write
```

The compute (steps 3, 6, 9) is fast. But the constant loading/saving to HBM makes it **memory-bandwidth bound** — the GPU spends most of its time waiting for data transfers.

For a sequence of 4096 tokens, the S matrix is `4096 × 4096 = 16M` numbers. Writing and reading that back from HBM repeatedly is extremely expensive.

---

## Flash Attention: The Core Idea

Instead of computing the full `Q·Kᵀ` matrix at once, Flash Attention tiles the computation into small blocks that fit entirely in SRAM.

### Tiling

```
Sequence: 4096 tokens
Block size: 64 tokens (fits in SRAM)

Process block 1 of K and V (tokens 1-64):
  Load Q_block1 into SRAM
  Load K_block1 into SRAM
  Compute partial scores → partial softmax
  Accumulate partial output

Process block 2 of K and V (tokens 65-128):
  Load K_block2 into SRAM
  Update softmax (online softmax algorithm)
  Accumulate output

...continue until all blocks processed...

Write final output to HBM once
```

The key insight: using the **online softmax** algorithm, you can compute the final correct softmax without ever materializing the full S matrix in HBM.

### Memory access comparison

```
Standard attention:
  HBM reads/writes: ~5 × (n × n × d) operations
  For n=4096, d=64: ~5 × 1 billion = huge

Flash Attention:
  HBM reads/writes: ~4 × (n × d) operations
  For n=4096, d=64: ~4 × 262K = manageable

Speedup: roughly n/block_size × (some factor) faster
For long sequences: 5-10× less memory bandwidth used
```

---

## Flash Attention 3: What's New for H100

FA1 and FA2 were designed for Ampere GPUs (A100). FA3 is specifically optimized for **Hopper architecture (H100)** with three key additions:

### 1. Asynchronous pipeline

H100 has dedicated hardware for async data transfers:

```
FA2:    Load K_block → Compute → Load next K_block → Compute → ...
        (sequential: wait for load before computing)

FA3:    Load K_block_2  (happening in background)
          while
        Compute with K_block_1  (happening simultaneously)

Result: computation and data loading overlap → GPU never idles waiting for data
```

### 2. FP8 / BF16 tensor core utilisation

H100 has specialised matrix multiply units (tensor cores) for low-precision arithmetic:

```
FA3 uses H100 tensor cores with BF16 inputs
→ 2-4× higher throughput on matrix multiplies
```

### 3. Warp specialization

FA3 assigns different GPU "warps" (groups of threads) to different tasks:

```
Warp group 1: handling data loads
Warp group 2: doing matrix multiplications
Warp group 3: applying softmax and accumulating

These run concurrently instead of sequentially
```

---

## How the Code Uses FA3

```python
# gpt.py:104-121 — training vs inference split, with window_size for SSSL layers

# Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
# window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
if kv_cache is None:
    # Training: causal attention with optional sliding window
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    # Inference: use flash_attn_with_kvcache which handles cache management
    k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
    y = flash_attn.flash_attn_with_kvcache(
        q, k_cache, v_cache,
        k=k, v=v,
        cache_seqlens=kv_cache.cache_seqlens,
        causal=True,
        window_size=window_size,
    )
    # Advance position after last layer processes
    if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)

# Re-assemble the heads and project back to residual stream
y = y.contiguous().view(B, T, -1)
y = self.c_proj(y)
```

### The two masks FA3 handles

**Causal mask (training):**
```
"The cat sat on the mat"

"mat" can attend to:  The, cat, sat, on, the, mat   ← all previous + self
"on"  can attend to:  The, cat, sat, on              ← can't peek at future
"The" can attend to:  The                            ← only itself
```

**Sliding window mask (for S-type layers):**
```
"mat" (pos 5) with window=2 can attend to: on(3), the(4), mat(5)
Even though it's causal, it's further restricted to the local window
```

FA3 handles both simultaneously in one fused kernel — no separate mask materialisation.

---

## KV Cache at Inference

During text generation (one token at a time), FA3 also handles the incremental KV cache:

```
Step 1: Generate "The"
  KV cache: {The: (K, V)}

Step 2: Generate "cat"  
  Q = query for "cat"
  K, V = load cached K/V for "The" + compute new K/V for "cat"
  Attention over 2 tokens

Step 3: Generate "sat"
  Q = query for "sat"
  K, V = load cached K/V for "The", "cat" + new for "sat"
  Attention over 3 tokens

...and so on
```

FA3's tiling makes this efficient even for long cached sequences.

---

## Fallback: SDPA on Non-H100 Hardware

On CPUs, Apple M-series, or older NVIDIA GPUs:

```
torch.nn.functional.scaled_dot_product_attention (SDPA)
→ Uses Flash Attention 2 on CUDA devices that support it
→ Falls back to standard attention on other hardware
→ Handles causal masking and sliding window via attention_bias

Works everywhere, just not as fast as FA3 on H100
```

---

## Summary

| | Standard Attention | Flash Attention 2 | Flash Attention 3 |
|---|---|---|---|
| Writes full n×n matrix to HBM | Yes ❌ | No ✅ | No ✅ |
| Tiled / blocked computation | No | Yes | Yes |
| Async pipeline | No | No | Yes ✅ |
| H100 tensor core optimized | No | Partial | Yes ✅ |
| Handles causal + sliding window | Separate passes | Yes | Yes |
| Handles KV cache inference | No | Yes | Yes |

> **One-line summary:** FA3 keeps attention computation inside fast on-chip SRAM by tiling the calculation into small blocks, and on H100 GPUs overlaps data loading with computation — making attention 2-10× faster with no change to the math.
