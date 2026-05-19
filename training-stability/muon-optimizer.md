# Muon Optimizer + AdamW: Two Optimizers for Two Jobs

**What it is:** Weight matrices in the transformer (attention, MLP) use the **Muon** optimizer, which applies orthogonalized gradient updates. Embeddings, the LM head, and scalar parameters use **AdamW** with tuned per-group learning rates.

**Code:** `gpt.py:369-409`

---

## Background: What Is an Optimizer?

An optimizer decides how to update model weights given the gradient from backpropagation.

```
Basic gradient descent:
  weight_new = weight_old - learning_rate × gradient

The gradient tells us: "which direction makes the loss go up?"
We step in the opposite direction.
```

Different optimizers handle this update differently — AdamW adapts learning rates per parameter, Muon orthogonalizes the update direction.

---

## AdamW: The Standard Optimizer

AdamW tracks two running statistics per parameter:
- **m** (first moment) — smoothed average of past gradients
- **v** (second moment) — smoothed average of past squared gradients

```
m = β₁ × m + (1 - β₁) × gradient         (momentum)
v = β₂ × v + (1 - β₂) × gradient²        (adaptive scale)

update = m / (√v + ε)
weight_new = weight_old - lr × update
```

The effect: parameters with large noisy gradients get smaller updates (√v is large), parameters with consistent small gradients get larger updates (√v is small).

### Why AdamW struggles for large matrices

For a large weight matrix W (e.g., 4096 × 4096), each of the 16M entries gets its own m and v. The update direction for each entry is considered independently.

**Problem: redundancy.** Many gradient directions are nearly parallel — they carry the same information. The optimizer wastes steps moving in correlated directions.

Think of it like this: if 100 neurons all learn "is this a noun?", most of their updates are redundant. AdamW doesn't prevent this.

---

## Muon: Orthogonalized Gradient Updates

Muon stands for **M**omentum + **O**rthogonalization **U**sing Newton-Schulz (the algorithm used internally).

### The core idea

Before applying the update, **orthogonalize** the gradient matrix using the Newton-Schulz algorithm:

```
gradient G  →  Newton-Schulz iteration  →  orthogonalized G_orth

G_orth has the property:
  - Same shape as G
  - All update directions are mutually perpendicular (orthogonal)
  - No two neurons update in the same direction
```

### What orthogonalization means

Imagine a weight matrix where each row is a direction for one neuron to update:

```
AdamW updates (can be correlated):
  Row 1: [0.8, 0.6, 0.0, 0.1]   ← similar to row 2
  Row 2: [0.7, 0.5, 0.1, 0.2]   ← almost the same direction as row 1
  Row 3: [0.0, 0.1, 0.9, 0.8]
  Row 4: [-0.1, 0.2, 0.8, 0.7]

Muon (after orthogonalization):
  Row 1: [1.0, 0.0, 0.0, 0.0]   ← points along dim 1
  Row 2: [0.0, 1.0, 0.0, 0.0]   ← points along dim 2, perpendicular to row 1
  Row 3: [0.0, 0.0, 1.0, 0.0]   ← points along dim 3
  Row 4: [0.0, 0.0, 0.0, 1.0]   ← points along dim 4
```

Each neuron moves in a unique direction — no wasted effort on redundant updates. The weight matrix evolves more efficiently.

### Newton-Schulz Iteration (simplified)

```
Input: G (gradient matrix)
Goal: find O where O is "as close as possible to G but orthonormal"

Step 1: Normalize G
Step 2: Iterate:  O ← 1.5 × O - 0.5 × O × Oᵀ × O
Step 3: Repeat ~5 times until convergence

Result: O_orth — an orthogonalized version of G
```

Each iteration brings O closer to having orthogonal rows while preserving the "direction" of G. 5 iterations is typically enough.

---

## Why Not Use Muon for Everything?

Muon is designed for **matrix** parameters. It doesn't work well for:

### 1. Embeddings (wte, lm_head)

```
wte has shape: (vocab_size=50K, d_model=1024)

Muon orthogonalizes rows — each vocabulary token would get a unique update direction.
But vocabulary tokens appear at very different frequencies:
  "the" appears millions of times → strong gradient
  "sesquipedalian" appears rarely → near-zero gradient

Orthogonalizing these together doesn't make sense.
The rare token rows would be forced to update in directions
that have nothing to do with their actual gradient.
```

### 2. Scalar parameters (biases, gate values, lambdas)

```
resid_lambda, smear_gate, x0_lambda — these are single numbers (scalars).

Orthogonalization is a matrix operation.
A 1×1 "matrix" can only be orthogonalized to +1 or -1 — useless.
AdamW's adaptive learning rate is exactly right for scalars.
```

---

## The Split: What Uses What

```python
# gpt.py:374-409 — actual setup_optimizer() code

# Separate out all parameters into groups
matrix_params = list(self.transformer.h.parameters())    # attention + MLP weights → Muon
value_embeds_params = list(self.value_embeds.parameters())
embedding_params = list(self.transformer.wte.parameters())
lm_head_params = list(self.lm_head.parameters())
resid_params = [self.resid_lambdas]
x0_params = [self.x0_lambdas]
smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]

# AdamW groups: embeddings, lm_head, scalars — each with carefully tuned lr
param_groups = [
    dict(kind='adamw', params=lm_head_params,       lr=unembedding_lr * dmodel_lr_scale, ...),
    dict(kind='adamw', params=embedding_params,     lr=embedding_lr * dmodel_lr_scale, ...),
    dict(kind='adamw', params=value_embeds_params,  lr=embedding_lr * dmodel_lr_scale * 0.5, ...),
    dict(kind='adamw', params=resid_params,         lr=scalar_lr * 0.01, ...),
    dict(kind='adamw', params=x0_params,            lr=scalar_lr, ...),
    dict(kind='adamw', params=smear_params,         lr=0.2, ...),
]

# Muon groups: matrix params, grouped by shape for stacking (Newton-Schulz needs same shape)
for shape in sorted({p.shape for p in matrix_params}):
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(
        kind='muon', params=group_params, lr=matrix_lr,
        momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
    ))

optimizer = MuonAdamW(param_groups)
```

Different learning rates per group because:
- Matrix weights can take larger steps (Muon's orthogonalization prevents them from diverging)
- Embeddings need smaller steps (sparse updates — each token only updates when it appears)
- Scalars (lambdas, gates) are sensitive and need careful tuning
- All AdamW learning rates are scaled by `∝1/√(d_model/768)` to stay proportional as model size changes

---

## Walkthrough: One Training Step on "The cat sat on the mat"

### Forward pass

```
Input: "The cat sat on the mat" → predict "."
Model processes sequence → outputs logits → compute cross-entropy loss
```

### Backward pass (compute gradients)

```
dL/dW_q = gradient for the Query weight matrix   (4096×4096)
dL/dW_v = gradient for the Value weight matrix   (4096×4096)
dL/d(wte) = gradient for embedding matrix        (50K×1024, sparse — only 6 rows nonzero)
dL/d(lambda_0) = gradient for layer 0 resid_lambda (scalar)
```

### Optimizer step

```
For W_q (Muon):
  1. Compute momentum: m = 0.9×m_prev + dL/dW_q
  2. Orthogonalize m via Newton-Schulz (5 iterations)
  3. Update: W_q -= lr × orthogonalized_m

For wte (AdamW):
  1. Compute m = 0.9×m + gradient (sparse — only 6 of 50K rows updated)
  2. Compute v = 0.999×v + gradient²
  3. Update: wte -= 0.001 × m / (√v + ε)

For lambda_0 (AdamW):
  1. Gradient: single number, e.g. 0.0023
  2. Adaptive update as usual
  3. lambda_0 -= 0.005 × adapted_gradient
```

---

## Why This Works Better Than AdamW Alone

```
Empirical finding (Kosson et al., 2024, introducing Muon):
  For weight matrices, Muon reaches the same loss 20-30% faster than AdamW
  with equivalent wall-clock time.

Why:
  AdamW: many steps wasted on correlated gradient directions
  Muon:  each step is maximally informative — no redundant updates
         The weight matrix "explores" the loss landscape more efficiently
```

---

## Summary

| Parameter type | Optimizer | Why |
|---|---|---|
| Attention W_q, W_k, W_v, W_o | **Muon** | Matrices → orthogonalized updates → efficient, no redundancy |
| MLP W_up, W_down | **Muon** | Same reason |
| wte (token embeddings) | **AdamW** | Sparse, frequency-varied → needs adaptive lr |
| lm_head | **AdamW** | Output projection, different statistics from inner matrices |
| Scalars (lambdas, gates) | **AdamW** | Not matrices → orthogonalization meaningless |

> **One-line summary:** Muon orthogonalizes gradient updates for weight matrices so each neuron moves in a unique direction — no wasted steps on redundant updates. AdamW handles everything else that doesn't fit the matrix structure.
