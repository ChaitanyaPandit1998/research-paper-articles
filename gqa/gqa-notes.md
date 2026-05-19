# GQA: Grouped-Query Attention

**Paper:** GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
**Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai (Google Research)
**Published:** EMNLP 2023
**arXiv:** https://arxiv.org/abs/2305.13245

---

## Background: What Happens Inside an Attention Layer?

Every Transformer attention layer works by computing three things for each token:
- **Query (Q)** — "What am I looking for?"
- **Key (K)** — "What do I contain?"
- **Value (V)** — "What do I give out if matched?"

The model computes Q·K scores to decide which tokens to attend to, then uses those scores to blend V vectors into the output.

In **Multi-Head Attention (MHA)** — the standard since the original Transformer — this is done in parallel across multiple "heads". If there are 32 heads, you get 32 separate Q, K, and V matrices working simultaneously, each looking for different patterns.

---

## The Problem: KV Cache at Inference Time

During training, all tokens are processed at once — fast.

During inference (generating text one token at a time), the model needs to attend to **every previous token** to generate the next one. To avoid recomputing K and V for every past token on every new step, the model stores them in a **KV cache**:

```
Generating token 100:
  → need K and V for tokens 1–99
  → store them in memory (KV cache)
  → just compute Q for token 100 and look up the cache

Generating token 101:
  → need K and V for tokens 1–100
  → cache grows by one row
```

With Multi-Head Attention and 32 heads, you store **32 K vectors + 32 V vectors** per token. For long sequences and large batches this becomes huge:

```
Example: LLaMA-scale model
  32 heads × 128 dims × 2 (K+V) × 4 bytes × 4096 tokens × 32 layers
  ≈ several GB just for the KV cache

The GPU spends most of its time reading/writing this cache,
not doing actual computation → slow inference
```

This bottleneck is called **memory bandwidth bound** inference.

---

## Solution 1: Multi-Query Attention (MQA)

Proposed by Shazeer (2019). The fix is simple:

> Keep all 32 Query heads. But use only **1 Key head and 1 Value head**, shared by all queries.

```
MHA:  Q₁ Q₂ Q₃ ... Q₃₂    K₁ K₂ K₃ ... K₃₂    V₁ V₂ V₃ ... V₃₂
                            ↑ 32 separate K heads  ↑ 32 separate V heads

MQA:  Q₁ Q₂ Q₃ ... Q₃₂    K (just 1)             V (just 1)
                            ↑ all queries share it  ↑ all queries share it
```

**Result:**
- KV cache shrinks by 32× — massive memory saving
- Inference is much faster
- **Problem:** Sharing one K and V across all heads loses expressive power → quality drops

---

## Solution 2: Grouped-Query Attention (GQA) — The Paper's Contribution

GQA sits between MHA and MQA. Instead of 1 or 32 KV heads, use **G groups** (e.g. G=8):

```
32 query heads split into 8 groups of 4:

Group 1:  Q₁  Q₂  Q₃  Q₄   →  share  K₁  V₁
Group 2:  Q₅  Q₆  Q₇  Q₈   →  share  K₂  V₂
Group 3:  Q₉  Q₁₀ Q₁₁ Q₁₂  →  share  K₃  V₃
...
Group 8:  Q₂₉ Q₃₀ Q₃₁ Q₃₂  →  share  K₈  V₈
```

Each group of queries still has its own dedicated K and V — so you keep more diversity than MQA — but you only cache 8 K/V heads instead of 32.

```
KV cache size comparison (32 heads → 8 KV heads):
  MHA: 32 K + 32 V = 64 vectors per token
  GQA: 8 K  + 8 V  = 16 vectors per token  (4× smaller)
  MQA: 1 K  + 1 V  = 2 vectors per token   (32× smaller)
```

### GQA unifies MHA and MQA

```
G = H  (groups = heads)  →  GQA becomes MHA  (original, full quality)
G = 1  (one group)       →  GQA becomes MQA  (fastest, lowest quality)
G = H/4                  →  sweet spot (near-MHA quality, near-MQA speed)
```

---

## The Uptraining Method

The paper also solves a practical problem: you already have a trained MHA model — how do you convert it to GQA cheaply, without retraining from scratch?

### Step 1 — Mean pool the KV heads

For each group of K heads that will be merged into one, average them:

```
Group 1 has K heads: K₁, K₂, K₃, K₄

New single K head = (K₁ + K₂ + K₃ + K₄) / 4

Do the same for V heads.
```

This gives you a sensible starting point — not random, not biased toward one head.

### Step 2 — Continue pre-training briefly

Run the model for ~5% of the original pre-training compute to let it adapt:

```
Original pre-training:  100% compute   (e.g. 1 trillion tokens)
Uptraining:               5% compute   (e.g. 50 billion tokens)

Cost: 20× cheaper than training GQA from scratch
```

The model quickly learns to make good use of the pooled KV heads.

---

## Results

Tested on T5 and other large models:

| Method | Quality | Inference Speed | KV Cache Size |
|---|---|---|---|
| MHA (32 heads) | Best | Slowest | Largest |
| GQA (8 heads) | Near-MHA | Near-MQA | 4× smaller than MHA |
| MQA (1 head) | Drops noticeably | Fastest | Smallest |

GQA hits the sweet spot: you give up almost nothing in quality, but get most of the speed benefit of MQA.

---

## Why This Matters

Before GQA, the choice was binary:
- Use MHA → great quality, slow inference
- Use MQA → fast inference, quality hit

GQA makes it a dial you can tune. It also lets you **reuse existing checkpoints** rather than train new models — which is hugely valuable given the cost of training large models.

---

## Real-World Adoption

GQA is now the standard in most modern large language models:

| Model | Attention Type |
|---|---|
| LLaMA 2 (70B) | GQA |
| LLaMA 3 (all sizes) | GQA |
| Mistral 7B | GQA |
| Mixtral 8x7B | GQA |
| Gemma | GQA |
| Falcon 40B | MQA (predecessor) |
| GPT-3 | MHA (older) |

---

## One-Line Summary

> GQA gives each small group of query heads its own shared Key and Value — fewer heads to cache means faster inference, while keeping enough diversity to match full multi-head quality.
