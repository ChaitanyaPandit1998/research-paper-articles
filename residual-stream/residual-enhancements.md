# Residual Stream Enhancements

**What it is:** Three learned tricks that fine-tune how information flows through the residual stream — controlling how strongly each layer updates the representation, how much the original embedding is preserved, and what gets stripped before the final prediction.

**Code:** `gpt.py:180-186`, `gpt.py:235-239`, `gpt.py:447-459`

```python
# gpt.py:180-186 — parameter declarations
self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
self.smear_gate = Linear(24, 1, bias=False)
self.smear_lambda = nn.Parameter(torch.zeros(1))
self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))

# gpt.py:235-239 — init: decaying pattern per layer
for i in range(n_layer):
    self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
for i in range(n_layer):
    self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

# gpt.py:447-459 — forward pass: lambda scaling, x0 blend, backout subtraction
x0 = x  # save initial normalized embedding for x0 residual
n_layer = self.config.n_layer
backout_layer = n_layer // 2  # cache at halfway point
x_backout = None
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0   # ← scale + blend
    ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
    if i == backout_layer:
        x_backout = x
# Subtract mid-layer residual to remove low-level features before logit projection
if x_backout is not None:
    x = x - self.backout_lambda.to(x.dtype) * x_backout       # ← backout subtraction
x = norm(x)
```

---

## Background: What Is the Residual Stream?

In a Transformer, every layer adds its output to the current representation rather than replacing it:

```
x = x + attention(x)    ← residual connection in attention
x = x + mlp(x)          ← residual connection in MLP
```

This `x` flowing through all layers is the **residual stream** — a running "sum" that accumulates information from every layer.

```
Start:    x = embedding("cat")              = [0.4, 0.6, -0.2, 0.8]
Layer 0:  x = x + attn_0(x) + mlp_0(x)    = [0.7, 0.5,  0.1, 0.6]
Layer 1:  x = x + attn_1(x) + mlp_1(x)    = [0.5, 0.8,  0.4, 0.3]
...
Layer N:  Final x → lm_head → next token prediction
```

The three enhancements below are all learned modifications to this flow.

---

## Enhancement 1: resid_lambdas — Per-Layer Residual Scaling

### What it does

Instead of a plain addition:
```
x = x + sublayer(x)
```

Scale the residual stream before adding:
```
x = λ × x + sublayer(x)

λ = resid_lambda (a single learned scalar, one per layer)
```

### Why this helps

Without λ, every layer has equal "power" to update the stream. With λ, early layers can be cautious (λ close to 1 → preserve the residual), while later layers can be bolder (λ < 1 → more of the update dominates).

### Walkthrough with "The cat sat on the mat"

```
Layer 0 (early): λ₀ = 0.95
  x = 0.95 × [0.4, 0.6, -0.2, 0.8]  +  [0.1, 0.2, 0.3, -0.1]
    = [0.38, 0.57, -0.19, 0.76]      +  [0.1, 0.2,  0.3, -0.1]
    = [0.48, 0.77,  0.11, 0.66]

  The original embedding is mostly preserved (scaled by 0.95).
  The layer adds a small correction.

Layer 5 (later): λ₅ = 0.7
  x = 0.7 × [current x]  +  sublayer(x)

  More of the update replaces the residual.
  Later layers overwrite more aggressively.
```

The model learns these λ values during training — it figures out naturally that early layers should be conservative and late layers should be transformative.

---

## Enhancement 2: x0_lambdas — Blending the Initial Embedding

### What it does

At every layer, blend the **original input embedding** (x₀ from layer 0) back into the residual stream:

```
Standard residual:
  x = x + sublayer(x)

With x0_lambda:
  x = x + sublayer(x) + α × x0

α = x0_lambda (learned scalar, one per layer)
x0 = the original token embedding, frozen from layer 0
```

### Why this helps

Deep in the network, x has been updated so many times it no longer looks like the original embedding. Adding x₀ back provides a "where did I start?" signal — a direct link to token identity at each layer.

Think of it like a GPS re-sync: every few steps you check your original position to make sure you haven't drifted too far.

### Walkthrough

```
"cat" original embedding x0 = [0.4, 0.6, -0.2, 0.8]  ← constant throughout

Layer 2: α₂ = 0.4  (strong blend early)
  Current x = [0.5, 0.7, 0.3, 0.2]
  x = x + sublayer(x) + 0.4 × x0
                        ↑ pulls x back toward original

Layer 8: α₈ = 0.05  (faint blend late)
  x = x + sublayer(x) + 0.05 × x0
  Almost no influence from original embedding
  The model has built up rich context and doesn't need reminding of basics
```

The α values are learned — naturally larger for early layers (where grounding matters) and fade toward zero in later layers (where abstract reasoning dominates).

---

## Enhancement 3: backout_lambda — Stripping Low-Level Features from Logits

### What it does

The final residual stream before the LM head contains **all** the accumulated information — some high-level (abstract reasoning, prediction-relevant) and some low-level (raw syntax, position signals).

backout_lambda subtracts the **mid-layer** residual from the **final** residual before applying the output norm:

```
Standard:
  logits = lm_head(RMSNorm(x_final))

With backout_lambda:
  logits = lm_head(RMSNorm(x_final - β × x_mid))

β = backout_lambda (learned scalar)
x_mid = the residual at the middle layer
x_final = the residual at the last layer
```

### Why this helps

```
x_mid carries: low-level features (position, surface syntax, raw token patterns)
               high-level features (some, but not fully developed)

x_final - x_mid captures:
  = what the second half of the network added
  = the high-level reasoning, predictions, semantic content
  ≈ "what the model figured out in the final layers"

By stripping x_mid, you remove low-level noise from logits.
The output distribution is based more on high-level reasoning.
```

### Walkthrough

```
After all layers, "mat" final residual x_final = [0.5, 0.4, 0.8, 0.1]
At middle layer, "mat" mid residual x_mid     = [0.3, 0.6, 0.2, 0.7]

backout_lambda β = 0.3

Adjusted residual = x_final - β × x_mid
                  = [0.5, 0.4, 0.8, 0.1] - 0.3 × [0.3, 0.6, 0.2, 0.7]
                  = [0.5, 0.4, 0.8, 0.1] - [0.09, 0.18, 0.06, 0.21]
                  = [0.41, 0.22, 0.74, -0.11]

This adjusted vector goes through RMSNorm → lm_head → logits
Low-level features in x_mid have been partially subtracted out.
```

---

## How the Three Work Together

```
x = x0  (original embedding)

For each layer:
  1. Scale the residual:      x = λ × x        ← resid_lambda
  2. Add sublayer output:     x = x + sublayer(x)
  3. Blend original back in:  x = x + α × x0   ← x0_lambda

After all layers:
  4. Strip low-level:         x = x_final - β × x_mid  ← backout_lambda
  5. Normalize and predict:   logits = lm_head(RMSNorm(x))
```

Each trick is a single scalar per layer — negligible parameter cost, measurable training benefit.

---

## How Is x0_lambda Different from Value Embeddings?

Both use the original token embedding — but they solve different problems in different places.

| | x0_lambdas | Value Embeddings |
|---|---|---|
| **Where it acts** | The residual stream, between layers | Inside the V vector, during attention |
| **Who benefits** | The token itself | Other tokens attending to this token |
| **Question answered** | "Am I still remembering what I am?" | "When others look at me, do they know what I am?" |

**x0_lambdas** fix an **internal drift** problem: as "cat"'s own representation evolves through layers, it can drift far from its original meaning. Adding `x0` back nudges the token's own running state back toward its identity.

**Value Embeddings** fix an **outward signal** problem: when "sat" attends to "cat" and retrieves its V, that V should clearly identify the token being retrieved. Without this, the drifted hidden state produces a murky V.

```python
# x0_lambdas — affects a token's own evolving state (residual stream loop)
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0   # gpt.py:452

# Value Embeddings — affects what a token sends to others (inside attention)
v = v + gate.unsqueeze(-1) * ve   # gpt.py:95
```

Fixing one doesn't fix the other — a token can have a well-grounded internal state but still send a muddy V to others, and vice versa.

---

## One-Line Summaries

| Enhancement | What it does |
|---|---|
| **resid_lambdas** | Scales the residual before each addition — early layers are cautious, later layers overwrite more |
| **x0_lambdas** | Blends the original token embedding back into each layer — keeps the model grounded in token identity |
| **backout_lambda** | Subtracts the mid-point residual from the final residual — strips low-level noise before predicting the next token |
