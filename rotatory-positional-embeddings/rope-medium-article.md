# From Fixed Signals to Rotating Vectors: Why RoPE Is Replacing Positional Embeddings

**Pull quotes:**
- "Sinusoidal positional encoding is a patch bolted onto a model that has no native sense of order. RoPE weaves position into the attention mechanism itself."
- "The Transformer doesn't know that 'dog' comes before 'bites' unless you tell it. Positional encoding is how you tell it."
- "RoPE doesn't ask the model to memorise absolute positions. It encodes relative distance - and that turns out to be exactly what attention needs."

---

When the "Attention Is All You Need" paper introduced the Transformer in 2017, it quietly acknowledged a problem in a single paragraph: self-attention has no concept of word order. Feed it "dog bites man" and "man bites dog," and - without additional signals - it sees the same three words in the same bag. The fix the authors proposed, sinusoidal positional encoding, was elegant and effective enough to ship. It was not, as the years since have demonstrated, the final answer.

This article explains what positional encoding does, why the sinusoidal version has limits, and how Rotary Position Embedding (RoPE) - now the dominant approach in modern language models including Llama, Mistral, and Qwen - addresses those limits without complicating the architecture.

---

## The Problem: Self-Attention Is Order-Blind

Self-attention computes relationships between tokens by comparing their Query and Key vectors. The computation is a set operation - it does not care which token came first. This is both the architecture's great strength (full parallelism, no sequential bottleneck) and its structural blind spot.

For language, order is not decorative. "The cat sat on the mat" and "The mat sat on the cat" contain identical tokens with opposite meanings. A model that cannot distinguish position cannot parse the difference.

Positional encoding is the solution: inject an order-aware signal into each token's representation so that even though the attention computation itself is order-blind, the inputs it receives carry position information. The question every approach to positional encoding tries to answer is: what signal, and how?

---

## Sinusoidal Positional Encoding: The Original Fix

The 2017 paper added a fixed vector to each token embedding before it entered the attention layers. That vector was computed from sine and cosine functions at different frequencies:

$$PE_{(pos,\, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos,\, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where `pos` is the token's position in the sequence, `i` is the dimension index, and `d_model` is the embedding dimension.

**The intuition.** Think of it like a clock with many hands, each moving at a different speed. The fastest hand (high-frequency dimensions) ticks every few positions - distinguishing nearby tokens. The slowest hand (low-frequency dimensions) completes a full cycle over thousands of positions - encoding long-range order. Every position gets a unique combination of hand positions, and that combination is the positional signal.

**Why sinusoids specifically?** The authors noted that the relative position between any two tokens can be expressed as a linear transformation of their encodings. This means the model can, in principle, learn to attend based on distance rather than absolute position - without being explicitly trained to do so.

### What Works Well

- **No parameters.** The encoding is purely mathematical, computed once and fixed. It adds no trainable weight to the model.
- **Extrapolation by design.** Because the frequencies are continuous functions, the model can, in theory, generalise to positions it never saw during training.
- **Computationally trivial.** Adding a vector to an embedding costs nothing compared to the rest of the forward pass.

### Where It Falls Short

**1. Position information degrades through the layers.** Positional encoding adds a signal to the token embedding at the input layer. By the time that embedding has passed through six or more layers of attention and feed-forward transformations, the positional signal has been mixed, overwritten, and diluted. The model must somehow preserve position information through transformations it was not specifically designed to propagate.

**2. Absolute position is not what attention needs.** What matters in language is not that "dog" is at position 4 in an absolute sense. What matters is that "dog" is two positions before "bites." Self-attention computes a dot product between a Query and a Key - a relative comparison. Injecting absolute position at the input and hoping the model infers relative distance from it is an indirect route.

**3. Poor length generalisation in practice.** While sinusoidal encoding is theoretically continuous, models trained on sequences up to length 512 consistently degrade on sequences of length 1024 or longer. The model has seen position vectors for positions 0–511 during training; the positions 512–1023 are mathematically defined but contextually unfamiliar. Empirically, this generalisation often does not hold (Press et al., 2021; Su et al., 2021).

**4. Learned embeddings are even more brittle.** Some early Transformer variants replaced sinusoidal encodings with a learned embedding table - one vector per position, trained from data. These perform comparably or marginally better within the training distribution but extrapolate even worse: there is simply no learned vector for positions the model never saw.

---

## Rotary Position Embedding (RoPE)

RoPE, introduced by Su et al. in the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021), takes a fundamentally different approach. Rather than adding a position signal to token embeddings at the input, RoPE encodes position by rotating the Query and Key vectors inside the attention computation itself.

### The Core Idea

In 2D, rotating a vector $(x_1, x_2)$ by angle $\theta$ gives:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

RoPE applies this idea across the full head dimension by treating embedding dimensions in pairs - (dim 0, dim 1), (dim 2, dim 3), and so on - and rotating each pair by an angle that depends on the token's position:

$$\theta_i = \frac{1}{10000^{2i/d}}$$

Token at position $m$ gets its Query and Key vectors rotated by $m \times \theta_i$ for each dimension pair $i$.

**What this means geometrically.** Every token's Query and Key are rotated by an amount proportional to their position. A token at position 3 is rotated three times as far as a token at position 1. Tokens at position 0 are not rotated at all.

### Where the Formula Comes From

The rotation formula is not a definition — it follows from three lines of geometry.

**Step 1 — Write (x, y) in polar form.**
Any 2D point can be described by its length $r$ and direction $\alpha$ instead of its coordinates:
$$x = r\cos\alpha \qquad y = r\sin\alpha$$

**Step 2 — Rotating by $\theta$ means adding $\theta$ to the direction.**
The length stays the same; only the angle changes:
$$x_\text{new} = r\cos(\alpha + \theta) \qquad y_\text{new} = r\sin(\alpha + \theta)$$

**Step 3 — Expand using the angle-addition identity.**
$$\cos(\alpha + \theta) = \cos\alpha\cos\theta - \sin\alpha\sin\theta$$
$$\sin(\alpha + \theta) = \sin\alpha\cos\theta + \cos\alpha\sin\theta$$

Substitute $r\cos\alpha = x$ and $r\sin\alpha = y$:

$$\boxed{x_\text{new} = x\cos\theta - y\sin\theta \qquad y_\text{new} = x\sin\theta + y\cos\theta}$$

This is the rotation matrix shown above — derived, not assumed. RoPE applies it to each dimension pair in the Query and Key vectors with $\theta = m \times \theta_i$, position $m$ scaled by the pair's base frequency. The same angle-addition identity is what makes the relative distance property work: when you dot-product two rotated vectors, the absolute positions cancel and only $(m - n)$ survives.

### Why Rotation Encodes Relative Distance

The critical property of RoPE emerges when you compute the attention score - the dot product between a rotated Query and a rotated Key.

For a query at position $m$ and a key at position $n$, their dot product depends only on their content vectors and the rotation angle $m - n$:

$$q_m \cdot k_n = f(x_m, x_n, m - n)$$

The absolute positions $m$ and $n$ disappear. What survives is their difference - the relative distance. This is exactly the information self-attention needs. The model does not need to learn to infer relative position from absolute signals; the mechanism provides it directly.

This property is sometimes called relative position encoding by construction - it is a mathematical consequence of the rotation, not a learned behaviour.

---

## Side-by-Side Comparison

| | Sinusoidal PE | Learned PE | RoPE |
|---|---|---|---|
| Where applied | Input embedding (additive) | Input embedding (additive) | Inside attention (multiplicative rotation) |
| Trainable parameters | None | Yes (one vector per position) | None |
| Encodes | Absolute position | Absolute position | Relative position (by construction) |
| Length generalisation | Limited in practice | Poor — no vector beyond training length | Strong — continuous rotation scales naturally |
| Attention score dependency | Indirect (model must learn to infer distance) | Indirect | Direct — score is a function of m−n only |
| Used in | Original Transformer (2017), BERT | GPT-2 | Llama, Mistral, Qwen, Falcon, and most modern LLMs |
| Compute overhead | Negligible (one addition) | Negligible (one lookup) | Negligible (rotation per dim pair — trivial vs attention cost) |

---

## Why Modern LLMs Chose RoPE

RoPE is not merely a theoretical improvement. The shift to RoPE in production models reflects empirical results that consistently favour it over the alternatives.

**Long-context performance.** Models using RoPE generalise better to sequence lengths beyond their training distribution. Extensions such as YaRN (Peng et al., 2023) and Dynamic NTK scaling further extend RoPE-based models to context lengths of 128K tokens and beyond - capabilities that sinusoidal encoding cannot reach without significant degradation.

**Position information stays in the attention layer.** Because RoPE is applied to Q and K rather than to the input embeddings, the positional signal does not need to survive multiple layers of transformation. It is injected at the point of use. This makes the positional information structurally robust across model depth.

**No interference with token semantics.** Sinusoidal PE adds a fixed vector to the token embedding, mixing positional and semantic signals at the representation level. RoPE keeps them separate - the token embedding carries meaning, the rotation carries position. The attention mechanism sees both without conflating them.

**Compatibility with GQA and Flash Attention.** RoPE is applied per-head to Q and K before the attention dot product, which means it composes naturally with Grouped Query Attention (GQA) and hardware-optimised attention kernels like Flash Attention. The rotation is just a pair of matrix operations; it does not change the attention computation's structure.

---

## A Concrete Example

Suppose you have a sentence: "The cat sat on the mat."

Under sinusoidal PE, each token receives a fixed positional vector added to its embedding at the input layer. By the time the attention computation runs six layers later, "cat" (position 1) and "mat" (position 5) carry the memory of their original signals - but that memory has been processed through six rounds of mixing. The model must learn, from data, to use whatever positional trace remains.

Under RoPE, when the attention layer computes the score between "cat" (position 1) and "mat" (position 5), the query vector for "cat" has been rotated by $1 \times \theta$ and the key vector for "mat" has been rotated by $5 \times \theta$. Their dot product is a function of $5 - 1 = 4$ - the relative distance - regardless of their absolute positions. A model fine-tuned on sequences of length 512 and deployed on a sequence of length 2048 still sees relative distances expressed as the same angular differences. The mechanism extrapolates continuously.

---

## Worked Examples

### Example 1 — Sinusoidal PE in Numbers

**Setup:** sentence "The cat sat on the mat", `d_model = 4`, positions start at 0.

With `d_model = 4`, there are two frequency pairs:
- Pair 0 (`i = 0`): divisor = $10000^{0/4} = 1$ → frequency = 1.0 (fast hand)
- Pair 1 (`i = 1`): divisor = $10000^{2/4} = 100$ → frequency = 0.01 (slow hand)

The positional vector added to each token embedding:

```
Token     Position   dim 0         dim 1         dim 2         dim 3
                     sin(pos×1.0)  cos(pos×1.0)  sin(pos×0.01) cos(pos×0.01)
────────────────────────────────────────────────────────────────────────────
"cat"     1          sin(1)= 0.841 cos(1)= 0.540 sin(0.01)=0.010 cos(0.01)=1.000
"sat"     2          sin(2)= 0.909 cos(2)=-0.416 sin(0.02)=0.020 cos(0.02)=1.000
"mat"     5          sin(5)=-0.959 cos(5)= 0.284 sin(0.05)=0.050 cos(0.05)=0.999
```

**What to notice:**

Dims 0–1 (high frequency, fast hand) change dramatically between "cat" and "mat": `0.841` → `-0.959`. These dimensions distinguish nearby tokens sharply.

Dims 2–3 (low frequency, slow hand) barely move: `0.010` → `0.050`. A full cycle takes thousands of positions. These dimensions encode coarse, long-range order.

Every position gets a unique fingerprint across the four dimensions — that fingerprint is what gets added to the token embedding at the input layer, before attention ever runs.

**The limitation in plain sight:** "cat" at position 1 always gets `[0.841, 0.540, 0.010, 1.000]`, regardless of the sentence it appears in or which other tokens it needs to relate to. The signal is fixed and absolute. The model must then learn, from training data, to infer from these absolute signals what the relative distance between "cat" and "mat" actually is.

---

### Example 2 — RoPE Rotation, Step by Step

**Setup:** `head_dim = 2` (one dimension pair, the simplest possible case). Highest frequency pair: $\theta_0 = 1/10000^0 = 1.0$.

**The query and key vectors** (the 2D content the model has learned for these words):
```
q ("cat") = [0.8, 0.6]
k ("mat") = [0.6, 0.8]
```

**Step 1 — Before rotation (no position information)**

```
q · k = 0.8×0.6 + 0.6×0.8 = 0.96
```

A score of 0.96. The model sees "cat" and "mat" as very similar — because their content vectors are similar. No position information is present. The model cannot tell that they are four positions apart.

**Step 2 — Rotate q by "cat"'s position angle (position 1)**

Rotation angle: $m \times \theta_0 = 1 \times 1.0 = 1.0$ radian.

$$\begin{pmatrix} q_1' \\ q_2' \end{pmatrix} = \begin{pmatrix} \cos(1.0) & -\sin(1.0) \\ \sin(1.0) & \cos(1.0) \end{pmatrix} \begin{pmatrix} 0.8 \\ 0.6 \end{pmatrix}$$

```
q' = [0.8×0.540 − 0.6×0.841,   0.8×0.841 + 0.6×0.540]
   = [0.432 − 0.505,             0.673 + 0.324]
   = [−0.073,  0.997]
```

**Step 3 — Rotate k by "mat"'s position angle (position 5)**

Rotation angle: $n \times \theta_0 = 5 \times 1.0 = 5.0$ radians.

```
k' = [0.6×cos(5.0) − 0.8×sin(5.0),   0.6×sin(5.0) + 0.8×cos(5.0)]
   = [0.6×0.284 − 0.8×(−0.959),       0.6×(−0.959) + 0.8×0.284]
   = [0.170 + 0.767,                    −0.575 + 0.227]
   = [0.937,  −0.348]
```

**Step 4 — Compute the attention score**

```
q' · k' = (−0.073)(0.937) + (0.997)(−0.348)
        = −0.068 − 0.347
        = −0.415
```

The score dropped from **0.96** (no position) to **−0.42** (with position). The rotation has injected the information that "cat" and "mat" are not at the same place — their angular difference (5.0 − 1.0 = 4.0 radians) has been folded directly into the score.

---

### Example 3 — The Relative Distance Property

This is the central claim of RoPE: the attention score after rotation depends only on *how far apart* the tokens are, not on *where* they are in the sentence. Here it is in numbers.

Same content vectors throughout: q = [0.8, 0.6], k = [0.6, 0.8], θ₀ = 1.0.

**The clock-hand analogy.** Before looking at the table, consider two clock hands. Rotate one by 30° and another by 50°. The angle *between* them is always 20°, regardless of where they started. RoPE does exactly this: it rotates each token's vector by an angle proportional to its position. When you take the dot product of two rotated vectors, you are measuring alignment — and alignment only cares about the angle *between* the vectors, not their absolute orientation.

**Three different pairs, all exactly 2 positions apart:**

```
Pair          m    n    gap   q rotated by m     k rotated by n     score
──────────────────────────────────────────────────────────────────────────
positions 1,3  1    3    2    [−0.073,  0.997]   [−0.707, −0.707]   −0.65
positions 5,7  5    7    2    [ 0.802, −0.597]   [−0.073,  0.997]   −0.65
positions 10,12 10  12   2    [−0.345, −0.939]   [ 0.936,  0.353]   −0.65
```

The rotated vectors look completely different at each pair of positions — but the dot product is the same every time: **−0.65**.

At positions 1 and 3, q has been rotated by 1θ and k by 3θ. At positions 5 and 7, q by 5θ and k by 7θ. The individual vectors point in entirely different directions in 2D space — but the *relative angle between them* is always 2θ. That relative angle is what the dot product measures, so the score is always the same.

The absolute positions 1, 5, 10 have disappeared. Only the gap of 2 survives in the score.

**Why this happens — one line of algebra:**

$$q' \cdot k' = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\bigl((n-m)\theta\bigr)\, k$$

Since $R(m\theta)^\top = R(-m\theta)$ (rotation matrices are orthogonal), composing the two rotations gives $R(-m\theta) \cdot R(n\theta) = R((n-m)\theta)$. Undoing a rotation by $m$ and then applying $n$ is the same as just applying $n - m$. The absolute positions cancel exactly; only the gap remains. The score is a function of the content vectors and the *gap* — nothing else.

**Why sinusoidal PE cannot do this.** Sinusoidal PE *adds* position to the token embedding before attention runs:

$$\text{token at position } m \;=\; \text{content} + \sin(m)$$

The dot product then expands into four terms — including cross-terms like $\text{content}_q \cdot \sin(n)$ and $\text{content}_k \cdot \sin(m)$ — where absolute positions are entangled with semantic content. There is no algebraic cancellation. The model must learn from data that the two absolute signals together encode a relative distance. RoPE delivers that distance directly, by construction.

---

## What RoPE Does Not Solve

Objectivity requires noting that RoPE is not the end of the story.

**Attention is still O(n²).** RoPE improves how position is encoded; it does not change the cost of computing attention over long sequences. At 100K tokens, quadratic complexity remains a hard practical constraint.

**Rotational symmetry has limits.** RoPE encodes position as a rotation angle. Very long sequences require large angles, and eventually the rotation wraps around - a phenomenon called frequency aliasing at very long ranges. YaRN, LongRoPE, and similar extensions address this at the cost of additional complexity.

**The relative position property holds exactly only in the full-attention case.** With certain attention variants (sliding window, sparse attention), the clean $m - n$ dependency is disrupted. The interaction between RoPE and non-standard attention patterns is an active area of research.

---

## Key Takeaways

- **Self-attention is order-blind.** Positional encoding is the mechanism that gives Transformer models a sense of sequence order - without it, word order is invisible to the model.
- **Sinusoidal PE works by addition.** It injects a fixed signal into token embeddings before the attention layers. It is parameter-free and theoretically continuous, but encodes absolute position and generalises poorly to lengths beyond training.
- **Learned PE trades flexibility for brittleness.** It can fit the training distribution well but fails entirely beyond the vocabulary of positions it saw during training.
- **RoPE works by rotation.** It encodes position inside the attention mechanism by rotating Query and Key vectors, not by modifying embeddings. The resulting dot product depends only on relative distance - the information attention actually needs.
- **RoPE is now the industry standard for good reason:** better length generalisation, no additional parameters, and a clean separation between semantic and positional information.

---

## Further Reading

- **"Attention Is All You Need"** (Vaswani et al., 2017) — the paper that introduced sinusoidal positional encoding and the Transformer architecture. The positional encoding section (Section 3.5) is brief but worth reading directly: arxiv.org/abs/1706.03762

- **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (Su et al., 2021) — the original RoPE paper. The derivation of the relative position property from first principles is clear and accessible: arxiv.org/abs/2104.09864

- **"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"** (Press et al., 2021) — a systematic study of length generalisation failures in standard positional encodings, and the ALiBi alternative: arxiv.org/abs/2108.12409

- **"YaRN: Efficient Context Window Extension of Large Language Models"** (Peng et al., 2023) — the extension that pushed RoPE-based models to 128K context windows. Required reading if you are working on long-context fine-tuning: arxiv.org/abs/2309.00071

---

The 2017 paper that introduced sinusoidal positional encoding described it as "a patch" - something to handle the order-blindness of self-attention without complicating the architecture. Four years later, RoPE proved that the patch could be replaced by something structurally cleaner. The core lesson is the same one the Transformer itself taught: the right abstraction, even for a seemingly minor problem, compounds.
