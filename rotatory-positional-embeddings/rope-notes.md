# RoPE: Rotary Position Embedding

**Paper:** RoFormer: Enhanced Transformer with Rotary Position Embedding
**Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
**Year:** 2021
**Link:** https://arxiv.org/abs/2104.09864

---

## The Problem RoPE Solves

Transformers have no built-in sense of word order — the attention mechanism treats all tokens as an unordered set. Position encodings fix this, but existing approaches have trade-offs:

| Approach | How it works | Problem |
|---|---|---|
| **Absolute (Sinusoidal)** | Add fixed sine/cosine values to token embeddings | Poor generalisation beyond training length; model doesn't naturally reason about *distance* between tokens |
| **Learned Absolute** | Learn a position embedding per slot | Hard cap at training length; can't handle longer sequences at all |
| **Relative** | Modify attention scores to include distance | Complex to implement cleanly; adds overhead; doesn't compose nicely with all architectures |

**RoPE's goal:** encode position in a way that is elegant, efficient, and makes *relative distance* fall out naturally from the math.

---

## The Core Idea

Instead of *adding* position information to token vectors, RoPE **rotates** them.

> Rotate the Query and Key vectors by an angle that depends on their position in the sequence. Then when Q and K are multiplied together (the attention dot product), the rotation angles subtract — and what's left encodes only the *relative distance* between the two tokens.

Think of it like clock hands:
- Token at position 3 has been rotated 3 steps.
- Token at position 7 has been rotated 7 steps.
- When you compare them, the difference (7 − 3 = 4 steps) is all that remains.

---

## How It Works (Step by Step)

### Step 1 — Split the vector into 2D pairs
Each query or key vector (dimension D) is split into D/2 pairs of numbers. Each pair lives in a 2D plane and can be rotated independently.

### Step 2 — Assign a rotation angle per pair and position
For a token at position `m`, dimension pair `d` gets rotated by:

```
angle = m × θ_d
where θ_d = 10000^(−2d / D)
```

- Position `m` scales the angle — further along = more rotation.
- Dimension index `d` controls the *frequency* — early pairs rotate slowly (low frequency, captures long-range structure), late pairs rotate fast (high frequency, captures local structure). Same idea as sinusoidal encodings.

### Step 3 — Apply the rotation
Each 2D pair `[x₁, x₂]` at position `m` becomes:

```
[x₁ cos(mθ) − x₂ sin(mθ),
 x₁ sin(mθ) + x₂ cos(mθ)]
```

This is a standard 2D rotation matrix applied per pair.

### Step 4 — Compute attention as normal
The dot product of rotated Q (position m) and rotated K (position n) gives:

```
Q_m · K_n  =  f(q, k, m−n)
```

The result depends only on `m − n` — the **relative distance** — not on the absolute positions. Relative position awareness comes for free from the math.

---

## Key Properties

### Relative position awareness
The model sees how far apart two tokens are, not where they sit in an absolute sense. This is more useful for language understanding.

### Long-term decay
Tokens that are far apart naturally produce smaller dot products (the high-frequency rotation pairs oscillate quickly, averaging toward zero over long distances). The model pays less attention to distant tokens without any explicit rule — it emerges from the geometry.

### Works beyond training length
Because position is encoded as an angle (not a learned slot), the model can handle sequences longer than those seen during training. Trained on 2K tokens → still reasonable at 4K–8K.

### Zero extra parameters
RoPE is a mathematical transformation applied at attention time. No new weights, no embedding tables. Completely free in terms of model size.

### Compatible with linear attention
Works with efficient attention variants (e.g. Performer), unlike some relative encoding schemes.

---

## Results

- Outperforms absolute sinusoidal, learned absolute, and relative position encodings on long-text classification benchmarks.
- Trained on 2K-length sequences → gracefully handles 4K–8K at inference.
- Consistent gains across RoBERTa-style and GPT-style architectures.

---

## Real-World Adoption

RoPE has become the **de-facto standard** position encoding for modern large language models:

| Model | Uses RoPE? |
|---|---|
| LLaMA 1 / 2 / 3 | ✅ Yes |
| Mistral | ✅ Yes |
| GPT-NeoX | ✅ Yes |
| Falcon | ✅ Yes |
| PaLM 2 | ✅ Yes |
| Gemma | ✅ Yes |
| Qwen | ✅ Yes |

Extensions like **YaRN** and **LongRoPE** further stretch RoPE to handle 128K+ context windows by rescaling the rotation frequencies.

---

## Where Does the Rotation Formula Come From?

The formula used in RoPE:
```
x_new = x·cos(θ) − y·sin(θ)
y_new = x·sin(θ) + y·cos(θ)
```

This has nothing to do with RoPE specifically. It is the standard **2D rotation formula** from geometry, known for centuries. RoPE simply borrows it.

### Step 1 — Describe any point using distance and angle

Any point **(x, y)** can be described as a distance `r` from the origin and an angle `φ`:

```
x = r · cos(φ)
y = r · sin(φ)

        y
        │     * (x, y)
        │    /
        │   /  r
        │  /
        │ / φ
        └───────────  x
```

### Step 2 — Rotating means adding θ to the angle

Rotating by θ keeps the distance `r` the same — you just spin around the origin.
The angle goes from `φ` to `(φ + θ)`:

```
x_new = r · cos(φ + θ)
y_new = r · sin(φ + θ)
```

### Step 3 — Expand using the angle addition identity

A standard trig identity:
```
cos(φ + θ) = cos(φ)·cos(θ) − sin(φ)·sin(θ)
sin(φ + θ) = sin(φ)·cos(θ) + cos(φ)·sin(θ)
```

Substitute in:
```
x_new = r·cos(φ)·cos(θ) − r·sin(φ)·sin(θ)
```

Since `r·cos(φ) = x` and `r·sin(φ) = y`:
```
x_new = x·cos(θ) − y·sin(θ)   ✓

y_new = r·sin(φ)·cos(θ) + r·cos(φ)·sin(θ)
      = y·cos(θ) + x·sin(θ)
      = x·sin(θ) + y·cos(θ)   ✓
```

The formula is nothing more than "add θ to the angle" written in (x, y) coordinates.

### Quick sanity check

Point **(1, 0)** rotated by **90°**:
```
cos(90°) = 0,  sin(90°) = 1

x_new = 1·0 − 0·1 = 0
y_new = 1·1 + 0·0 = 1

Result: (0, 1)  ✓ — moved straight up, as expected
```

Point **(1, 0)** rotated by **45°**:
```
cos(45°) = 0.707,  sin(45°) = 0.707

x_new = 1·0.707 − 0·0.707 = 0.707
y_new = 1·0.707 + 0·0.707 = 0.707

Result: (0.707, 0.707)  ✓ — halfway between x-axis and y-axis
```

### Why RoPE uses it

RoPE treats each pair of numbers in a word embedding as an (x, y) point and rotates it by `position × θ`. The rotation formula is just standard geometry. RoPE's actual contribution was realising that applying this rotation to Q and K vectors causes their dot product to encode only the relative distance between two tokens — not the formula itself.

---

## Design Decisions — How Were These Choices Made?

### Why rotation matrices? Was it derived or guessed?

**Fully derived from first principles.** The paper starts by writing down a precise requirement:

> *Find a function f(q, m) such that the dot product ⟨f(q, m), f(k, n)⟩ depends only on the word content q, k and the relative gap (m − n) — never on the absolute positions alone.*

The authors then work through the algebra. Treating each 2D pair as a complex number, the only function that satisfies the requirement is multiplying by `e^(imθ)` — which is exactly a rotation in the complex plane. Rotation wasn't a creative guess or an aesthetic choice; it is what the constraint equation forces you to arrive at. The math leaves no other option.

In plain terms: they asked "what operation preserves relative distance in a dot product?" and rotation was the answer the algebra gave back.

---

### Why 10000? Was it derived?

**No — it was borrowed.** The 10000 base comes from the original Transformer paper (Vaswani et al. 2017, "Attention is All You Need"), which used the same constant in its sinusoidal positional encoding. The RoPE authors carried it over without a new derivation.

The Vaswani paper also gives no formal proof for 10000 — it was chosen because it produces a useful practical spread:

```
With θᵢ = 10000^(−2i/d), the wavelengths range from:
  2π  (fastest rotating dimension)
  to
  10000 × 2π  (slowest rotating dimension)

This covers short-range and long-range patterns
across typical training lengths of 512–4096 tokens.
```

**What if you used a different base?**

| Base | Effect |
|---|---|
| Too small (e.g. 100) | Wavelengths too short — slow dimensions rotate too fast, can't capture long-range structure |
| 10000 | Good default for sequences up to ~4K tokens |
| Too large (e.g. 10,000,000) | Slow dimensions barely move — wasted capacity, poor short-range resolution |

The fact that later work changes this constant confirms it was always a practical default:
- **YaRN** uses base ≈ 500,000 for 32K–128K context windows
- **LongRoPE** rescales frequencies dynamically for 2M+ token contexts

So 10000 is not a fundamental truth — it is a default that works well for the sequence lengths models were trained on in 2021.

**Note — this implementation uses base=100,000:**

```python
# gpt.py:263
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
    # TODO: bump base theta more? e.g. 100K is more common more recently
```

The 10× larger base (100K vs 10K) means the slowest rotating dimension completes one full cycle over a 100× longer sequence. This extends the model's positional resolution for longer contexts — consistent with more recent practice.

---

## One-Line Summary

> RoPE rotates Query and Key vectors by position-dependent angles so that their dot product automatically encodes relative distance — giving Transformers a natural, parameter-free sense of word order that generalises beyond training length.
