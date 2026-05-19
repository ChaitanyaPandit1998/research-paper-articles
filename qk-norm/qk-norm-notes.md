# QK Norm

**What it is:** Normalizing the Query (Q) and Key (K) vectors with RMSNorm after positional encoding (RoPE) but before the attention dot product.

**Used in:** Gemma, PaLM 2, LLaMA variants, and other large-scale modern Transformers.

---

## The Problem It Solves: Attention Entropy Collapse

### What is "entropy" in attention?

When attention weights are spread across many tokens, entropy is **high** — the model is gathering information from many places:

```
Healthy attention for "cat":
  "The"  → 0.25
  "cat"  → 0.30   ← spread out, model learns from multiple tokens
  "sat"  → 0.25
  "on"   → 0.10
  "mat"  → 0.10
```

When attention collapses onto a single token, entropy is **low** — the model effectively ignores everything else:

```
Collapsed attention for "cat":
  "The"  → 0.97   ← almost everything goes here ("attention sink")
  "cat"  → 0.01
  "sat"  → 0.01
  "on"   → 0.01
  "mat"  → 0.00
```

This is **attention entropy collapse** — and it makes the model nearly useless because it can't aggregate information broadly.

### Simple analogy: a broken election

Imagine attention is a vote across 6 candidates. Healthy: each candidate gets 10–30% of votes. Collapsed: one candidate gets 97%, nobody else matters. The "winner" carries all information; everyone else is silenced. The model loses the ability to learn from context.

---

### How Does Collapse Happen?

The attention score between token A (Query) and token B (Key) is:

```
score = Q · K  /  √d
```

If Q and K vectors grow large in magnitude during training, their dot product explodes:

```
Early training:
  Q = [0.1, 0.2, 0.1]  K = [0.1, 0.3, 0.2]
  Q·K = 0.01 + 0.06 + 0.02 = 0.09   ← small, softmax stays spread

After many training steps (weights grow):
  Q = [3.1, 4.2, 2.8]  K = [2.9, 3.7, 3.1]
  Q·K = 8.99 + 15.54 + 8.68 = 33.21  ← huge, softmax spikes
```

When you feed large numbers into softmax, the biggest one dominates completely:

```
Softmax([33.21, 5.1, 3.2, 2.8]):
  e^33.21 = 2.7 × 10¹⁴   ← astronomically large
  e^5.1   = 164
  e^3.2   = 24
  e^2.8   = 16

After normalising: [≈0.9999, ≈0.0001, ≈0.0000, ≈0.0000]

One token gets essentially ALL the attention. Collapse complete.
```

This happens more at **long context lengths** (more tokens to spike on) and gets worse the longer you train without a fix.

---

## The Fix: QK Norm

### Standard attention flow

```
Input x
  │
  ├──→  x × W_q  →  Q  →  RoPE(Q)  ──────────────────→  Q · Kᵀ / √d
  │                                                               │
  └──→  x × W_k  →  K  →  RoPE(K)  ──────────────────→          │
                                                                  ↓
                                                              Softmax → weights → blend V
```

### QK Norm attention flow

```
Input x
  │
  ├──→  x × W_q  →  Q  →  RoPE(Q)  →  RMSNorm(Q)  ──→  Q · Kᵀ / √d
  │                                                               │
  └──→  x × W_k  →  K  →  RoPE(K)  →  RMSNorm(K)  ──→          │
                                                                  ↓
                                                              Softmax → weights → blend V
```

The only addition: **RMSNorm after RoPE, before the dot product.**

RMSNorm rescales every Q and K vector to a controlled magnitude — so no matter how large the weights grow, the dot products stay bounded. The softmax never collapses.

---

## What is RMSNorm?

### The formula

```
RMSNorm(x) = (x / RMS(x)) × γ

where:
  RMS(x) = √( mean(x²) )   ← root of the mean of squared values
  γ       = learnable scale parameter (one per dimension)
```

### Step-by-step with real numbers

Say Q for "cat" is the vector `[3.0, −4.0, 2.0, −1.0]`

**Step 1 — Square each value:**
```
[3.0², (-4.0)², 2.0², (-1.0)²]
= [9.0, 16.0, 4.0, 1.0]
```

**Step 2 — Take the mean:**
```
(9.0 + 16.0 + 4.0 + 1.0) / 4 = 30.0 / 4 = 7.5
```

**Step 3 — Take the square root (this is RMS):**
```
√7.5 ≈ 2.74
```

**Step 4 — Divide original vector by RMS:**
```
[3.0, −4.0, 2.0, −1.0]  /  2.74
= [1.09, −1.46, 0.73, −0.36]
```

**Step 5 — Multiply by learnable scale γ** (say γ = [1.2, 1.2, 1.2, 1.2]):
```
[1.09 × 1.2, −1.46 × 1.2, 0.73 × 1.2, −0.36 × 1.2]
= [1.31, −1.75, 0.88, −0.43]
```

The vector now has a controlled magnitude. No matter how large `[3.0, -4.0, 2.0, -1.0]` was to begin with — whether weights had grown to `[30, -40, 20, -10]` — after RMSNorm the scale is always governed by γ.

### Why RMSNorm and not LayerNorm?

```
LayerNorm(x) = (x − mean(x)) / std(x) × γ + β

RMSNorm(x)  = x / RMS(x) × γ
```

| | LayerNorm | RMSNorm |
|---|---|---|
| Subtracts mean? | Yes | No |
| Extra parameter β (bias)? | Yes | No |
| Computation cost | Higher | Lower (~10% faster) |
| Preserves direction? | No (shifts the vector) | Better (only rescales) |

For Q and K, direction matters — it encodes what the token is "looking for" and "containing". RMSNorm rescales without shifting, so the directional information (and the positional signal from RoPE) is preserved.

The learnable γ also gives the model a way to tune the **effective temperature** of attention — if γ is large, dot products are larger and attention is sharper; if γ is small, attention is softer and more spread out.

---

## Walkthrough: "The cat sat on the mat"

### Without QK Norm — weights have grown large after training

```
Q for "cat":  [3.1, 4.2, −2.8,  3.5]   ← large magnitude
K for "sat":  [2.9, 3.7,  3.1, −2.4]   ← large magnitude
K for "The":  [4.1, 3.9, −3.0,  2.8]   ← large magnitude

Dot products:
  "cat"Q · "sat"K  = (3.1×2.9) + (4.2×3.7) + (−2.8×3.1) + (3.5×−2.4)
                   = 8.99 + 15.54 − 8.68 − 8.40  = 7.45

  "cat"Q · "The"K  = (3.1×4.1) + (4.2×3.9) + (−2.8×−3.0) + (3.5×2.8)
                   = 12.71 + 16.38 + 8.40 + 9.80  = 47.29  ← huge spike

Softmax([47.29, 7.45, small, small, small, small]):
  "The" gets ≈ 0.9999   ← attention sink
  "sat" gets ≈ 0.0001
  Everything else ≈ 0

"cat" is now attending almost exclusively to "The" — useless.
```

### With QK Norm — after RMSNorm is applied

```
RMS("cat"Q)  = √(mean([9.61, 17.64, 7.84, 12.25])) = √11.84 ≈ 3.44
RMS("The"K)  = √(mean([16.81, 15.21, 9.0, 7.84]))  = √12.22 ≈ 3.49

Normalised:
  "cat"Q_norm  = [0.90, 1.22, −0.81, 1.02]   (γ=1 for simplicity)
  "The"K_norm  = [1.18, 1.12, −0.86, 0.80]

Dot product:
  "cat"Q_norm · "The"K_norm
  = (0.90×1.18) + (1.22×1.12) + (−0.81×−0.86) + (1.02×0.80)
  = 1.06 + 1.37 + 0.70 + 0.82  = 3.95   ← much smaller

  "cat"Q_norm · "sat"K_norm  ≈ 2.10

Softmax([3.95, 2.10, 1.5, 0.8, 0.6, 0.5]):
  "The" → 0.39
  "sat" → 0.25
  "on"  → 0.15
  "cat" → 0.10
  ...

Spread out. "cat" learns from multiple tokens. Healthy attention.
```

---

## Relationship to Attention Temperature

Softmax "temperature" controls how sharp or spread the output is:
- **High temperature** → flat, uniform distribution (attends to everything equally)
- **Low temperature** → peaked, spike distribution (attends to one thing)

Without QK Norm, temperature is uncontrolled — dot products grow with weights, making attention sharper and sharper until it collapses.

With QK Norm, the learnable γ acts as a **controlled temperature dial**:

```
Small γ → small dot products → flatter softmax → attend broadly
Large γ → larger dot products → sharper softmax → attend selectively

The model learns the right γ from data, per layer, per head.
Temperature is now a learned property, not an accident of weight magnitude.
```

---

## Where It's Used

| Model | Uses QK Norm? |
|---|---|
| Gemma (Google) | ✅ Yes |
| PaLM 2 (Google) | ✅ Yes |
| LLaMA variants | ✅ Some variants |
| GPT-3 | ❌ No (older architecture) |
| BERT | ❌ No (older architecture) |

Particularly valuable at **long context lengths** (16K+ tokens) where the risk of attention entropy collapse is highest — more tokens means more chances for one to become a dominant sink.

---

## One-Line Summary

> QK Norm applies RMSNorm to Query and Key vectors before the attention dot product, preventing the softmax from collapsing onto a single token as weights grow large during training.
