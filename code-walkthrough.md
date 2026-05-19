# Code Walkthrough — "cat drinks milk" through the GPT Architecture

We trace the string `"cat drinks milk"` through every step of `train_gpt_new.py`.
To keep numbers readable, we use a tiny model config throughout:

```python
config = GPTConfig(
    block_size = 8,
    vocab_size = 50257,
    n_layer    = 4,       # 4 transformer blocks
    n_head     = 4,       # 4 Q heads
    n_kv_head  = 2,       # 2 KV heads — GQA
    n_embd     = 16,      # 16-dim embedding
)
# derived:
# head_dim  = n_embd // n_head = 16 // 4 = 4
# kv_repeat = n_head // n_kv_head = 4 // 2 = 2
```

---

## Step 0 — Tokenisation

The GPT-2 BPE tokenizer splits the string into subword tokens and maps each to an integer ID.

```python
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("cat drinks milk")
# → [9246, 15263, 8336]
#     cat   drinks  milk

idx = torch.tensor([[9246, 15263, 8336]])  # shape: (B=1, T=3)
```

`idx` is the only input the model ever sees — just a sequence of integers.

---

## Step 1 — Token Embedding + Cast + RMSNorm

```python
# GPT.forward lines 515–517
x = self.transformer.wte(idx)   # lookup: each integer → a 16-dim vector
x = x.to(COMPUTE_DTYPE)         # fp32 → bf16 (fast matmuls)
x = norm(x)                     # RMSNorm each token vector independently
```

`wte` is a table of shape `(vocab_size, 16)`. Each token ID is used as a row index.

After the lookup, `x` has shape `(1, 3, 16)`:
```
x[0, 0] = [0.31, -0.52, 0.14, ...]   ← embedding for "cat"
x[0, 1] = [-0.08, 0.77, -0.33, ...]  ← embedding for "drinks"
x[0, 2] = [0.55, 0.22, -0.61, ...]   ← embedding for "milk"
```

`norm(x)` rescales each 16-dim vector to have RMS ≈ 1. B and T are never mixed —
each token is normalised independently:

```
rms("cat") = sqrt(mean of all 16 values²)
x[0, 0]   = x[0, 0] / rms("cat")
```

Shape after Step 1: `(1, 3, 16)` — unchanged.

---

## Step 2 — Smear Gate

```python
# GPT.forward lines 523–528
gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
# gate shape: (1, 2, 1) — one scalar per token except the first

x = torch.cat([
    x[:, :1],                         # "cat"    — no predecessor, unchanged
    x[:, 1:] + gate * x[:, :-1],      # "drinks" gets fraction of "cat" blended in
                                       # "milk"   gets fraction of "drinks" blended in
], dim=1)

x0 = x   # save post-smear embedding — used as anchor in every layer
```

Before any attention happens, each token gets a free peek at the token before it.
This is a cheap O(T) bigram signal — no attention computation needed.

```
"cat"    → unchanged
"drinks" → drinks + 0.12 * cat    (gate=0.12)
"milk"   → milk   + 0.09 * drinks (gate=0.09)
```

`smear_lambda` starts at 0 so no smearing happens at the start of training.
The model learns the right amount over time.

---

## Step 3 — Transformer Loop (4 layers)

The following steps repeat for `i = 0, 1, 2, 3`.

---

### 3a — Residual Stream Scaling

```python
# GPT.forward line 548
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

Before each block, the stream is rescaled and re-anchored to the original embedding.

```
layer 0: x = 1.15 * x + 0.20 * x0   # preserve stream, strong anchor
layer 1: x = 1.12 * x + 0.15 * x0
layer 2: x = 1.08 * x + 0.10 * x0
layer 3: x = 0.95 * x + 0.05 * x0   # deep layer — weaker anchor, more freedom
```

`resid_lambdas` and `x0_lambdas` are learnable per-layer scalars, initialised to these
schedules and trained from there.

---

### 3b — Pre-norm Before Attention

```python
# Block.forward line 348
x = x + self.attn(norm(x), ve, cos_sin, window_size)
```

`norm(x)` runs before attention sees the input — this is "pre-norm". It ensures
attention always receives unit-scale vectors, regardless of how large the residual
stream has grown.

---

### 3c — Q, K, V Projections (GQA)

Starting from `x` shape `(1, 3, 16)` — batch=1, 3 tokens, 16-dim vectors.

**Q projection:**
`c_q` is `Linear(16, 16)` — maps each 16-dim token vector to a 16-dim Q output.
`.view(1, 3, 4, 4)` splits the last dimension into 4 heads × 4 dims each:
```
(1, 3, 16) → (1, 3, 4, 4)
              B  T  n_head  head_dim
```
Each token now has 4 separate Q vectors — one per head, each asking its own question.

**K and V projections:**
`c_k` and `c_v` are `Linear(16, 8)` — maps 16-dim input to only 8-dim (2 heads × 4 dims).
`.view(1, 3, 2, 4)` splits into 2 KV heads:
```
(1, 3, 8) → (1, 3, 2, 4)
              B  T  n_kv_head  head_dim
```

```python
# CausalSelfAttention.forward lines 259–261
q = self.c_q(x).view(1, 3, 4, 4)   # 4 Q heads, head_dim=4
k = self.c_k(x).view(1, 3, 2, 4)   # 2 KV heads
v = self.c_v(x).view(1, 3, 2, 4)   # 2 KV heads
```

**Side by side:**
```
x:  (1, 3, 16)   — input

Q:  (1, 3, 4, 4) — 4 heads, each head asks its own question
K:  (1, 3, 2, 4) — 2 heads, each head provides a key to match against
V:  (1, 3, 2, 4) — 2 heads, each head provides a value to return
```

Q has 4 heads but K/V only have 2 — that's GQA.
Q heads 0 and 1 share K/V head 0. Q heads 2 and 3 share K/V head 1:
```
Q heads:  [q_head0, q_head1, q_head2, q_head3]   — 4 heads
K heads:  [k_head0,           k_head1          ]   — 2 heads
           ↑ shared by q0,q1   ↑ shared by q2,q3
```
The K/V cache at inference is 2× smaller because you only store 2 K/V heads instead of 4.

**What `.view()` does here:**

`.view()` reshapes a tensor without changing any of the underlying numbers — only how the dimensions are grouped.

Simple example:
```python
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])  # shape (8,)

x.view(2, 4)
# [[1, 2, 3, 4],
#  [5, 6, 7, 8]]   shape (2, 4) — same 8 numbers, grouped into 2 rows of 4

x.view(4, 2)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]   shape (4, 2) — same 8 numbers, grouped into 4 rows of 2
```

Applied to Q: `c_q(x)` outputs `(1, 3, 16)`. The 16 is really `4 heads × 4 dims`:
```python
# "cat" Q output — 16 numbers flat:
[0.1, 0.5, -0.3, 0.8,  0.2, -0.1, 0.9, 0.4,  -0.5, 0.7, 0.1, 0.3,  0.6, -0.2, 0.4, 0.8]

# .view(1, 3, 4, 4) groups into 4 heads of 4 dims each:
head 0: [0.1,  0.5, -0.3,  0.8]
head 1: [0.2, -0.1,  0.9,  0.4]
head 2: [-0.5, 0.7,  0.1,  0.3]
head 3: [0.6, -0.2,  0.4,  0.8]
```

Applied to K/V: `c_k(x)` outputs `(1, 3, 8)`. The 8 is `2 heads × 4 dims`:
```python
# .view(1, 3, 2, 4) groups into 2 heads of 4 dims each:
head 0: [0.3,  0.1, -0.4,  0.7]
head 1: [-0.2, 0.9,  0.5, -0.1]
```

No computation happens in `.view()` — it is purely a reinterpretation of the same memory.

---

### 3d — Value Embeddings (alternating layers: 1 and 3 for n_layer=4)

```python
# CausalSelfAttention.forward lines 267–271
ve = self.value_embeds[str(i)](idx)              # (1, 3, 8) — raw token embed
ve = ve.view(1, 3, 2, 4)                         # reshape to (B, T, n_kv_head, head_dim)
gate = 3 * torch.sigmoid(self.ve_gate(x[...,:12]))  # (1, 3, 2) — one gate per KV head
v = v + gate.unsqueeze(-1) * ve
```

**Why `.view()` here:**
`value_embeds` is an `Embedding(vocab_size, 8)` — it returns a flat 8-dim vector per token.
But `v` has shape `(1, 3, 2, 4)` — split into 2 KV heads of 4 dims each.
To add `ve` into `v`, they must have the same shape.
`.view(1, 3, 2, 4)` splits the flat 8-dim into 2 heads × 4 dims — matching `v` exactly:
```
ve before: (1, 3, 8)      — flat: [e0, e1, e2, e3, e4, e5, e6, e7]
ve after:  (1, 3, 2, 4)   — head 0: [e0,e1,e2,e3]  head 1: [e4,e5,e6,e7]
v shape:   (1, 3, 2, 4)   — same shape, so v + gate * ve works directly
```

In layer 3, `x` has been transformed by 3 blocks and no longer looks much like the
original "cat" / "drinks" / "milk" embeddings. But `ve` is a fresh lookup of the
raw token — it always carries a clean identity signal.

Mixing it into `v` means: when "milk" attends to "cat", the value it retrieves still
clearly says "I am cat", even in deep layers.

```
gate for "cat",   KV head 0 = 2.1   → strong mix-in
gate for "drinks",KV head 0 = 0.8   → weak mix-in
gate for "milk",  KV head 1 = 1.6   → moderate mix-in
```

---

### 3e — RoPE on Q and K

```python
# CausalSelfAttention.forward lines 277–278
cos, sin = cos_sin           # precomputed tables, sliced to T=3
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
```

`apply_rotary_emb` splits each head vector (dim=4) into two halves and rotates:

```
head vector: [h0, h1, h2, h3]
split:  x1 = [h0, h1]   x2 = [h2, h3]

pair (h0, h2) rotated by θ₀ × position
pair (h1, h3) rotated by θ₁ × position
```

Position 0 ("cat"):    no rotation (angles = 0)
Position 1 ("drinks"): rotated by θ × 1
Position 2 ("milk"):   rotated by θ × 2

After rotation, the dot product `Q[milk] · K[cat]` naturally encodes
the relative distance (2 − 0 = 2). The model never needs to learn
"position 2 is far from position 0" — it falls out of the math.

---

### 3f — QK Norm

```python
# CausalSelfAttention.forward lines 283–285
q, k = norm(q), norm(k)   # RMSNorm each head vector
q = q * 1.2
k = k * 1.2
```

Without this, as weights grow over training, Q·K dot products grow unboundedly.
A very large dot product → softmax collapses to a spike on one token → the model
stops attending broadly. QK Norm prevents this by keeping Q and K at unit scale.
The ×1.2 restores a little sharpness after normalisation.

---

### 3g — Attention (SDPA + sliding window)

```python
# CausalSelfAttention.forward lines 296–314
# GQA: expand K,V from 2 heads to 4 to match Q
k = k.repeat_interleave(2, dim=2)   # (1, 3, 2, 4) → (1, 3, 4, 4)
v = v.repeat_interleave(2, dim=2)

q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
# all now: (1, 4, 3, 4) = (B, n_head, T, head_dim)
```

**Query and Key — what they mean:**

- **Query** = the token asking the question: *"which other tokens are relevant to me?"*
- **Key**   = the token being asked: *"what information do I have?"*

A score `Q[query] · K[key]` is computed for every (query, key) pair.
High score → attend more. Low score → mostly ignore.
Softmax over the allowed scores gives the final attention weights.

---

**Sliding window rule: `query_position − key_position ≤ window`**

The window allows a query to attend to a key only if the key is at most `window` positions behind.

With our 3-token sequence (cat=0, drinks=1, milk=2) and window=2:
```
milk   (pos 2) → milk   (pos 2):  2 - 2 = 0  ≤ 2  ✓
milk   (pos 2) → drinks (pos 1):  2 - 1 = 1  ≤ 2  ✓
milk   (pos 2) → cat    (pos 0):  2 - 0 = 2  ≤ 2  ✓  (just within window)
```

With only 3 tokens the window doesn't exclude anything — all tokens are within
distance 2 of each other. Sliding window only starts cutting tokens off in longer
sequences. Example with 5 tokens `"the cat drinks cold milk"` (window=2):

```
milk (pos 4) → cold   (pos 3):  4 - 3 = 1  ≤ 2  ✓
milk (pos 4) → drinks (pos 2):  4 - 2 = 2  ≤ 2  ✓
milk (pos 4) → cat    (pos 1):  4 - 1 = 3  > 2  ✗  excluded
milk (pos 4) → the    (pos 0):  4 - 0 = 4  > 2  ✗  excluded
```

**Layer 3 (full context, L layer) — no window restriction:**
```
             cat  drinks  milk
cat    →  [  ✓     ✗      ✗  ]   causal: can't see future
drinks →  [  ✓     ✓      ✗  ]
milk   →  [  ✓     ✓      ✓  ]   sees all three — full context
```

The softmax over allowed positions gives attention weights, e.g. for "milk" in layer 3:
```
milk attends to:  cat=0.20,  drinks=0.35,  milk=0.45
```
The output for "milk" is a weighted sum of V vectors at those positions.

---

### 3h — Output Projection + Residual

```python
y = self.c_proj(y)   # (1, 3, 16) — project back to n_embd
x = x + y            # add attention output to residual stream
```

`c_proj` is initialised to zero, so at the very start of training each block
acts as a pure pass-through. The blocks learn to contribute gradually.

---

### 3i — MLP (with ReLU²)

```python
# Block.forward line 349 + MLP.forward
x = x + self.mlp(norm(x))

# inside MLP:
x = self.c_fc(x)          # (1, 3, 16) → (1, 3, 64)  expand 4×
x = F.relu(x).square()    # ReLU²
x = self.c_proj(x)        # (1, 3, 64) → (1, 3, 16)  back to n_embd
```

ReLU² before/after for one token:
```
after c_fc:      [-0.3,  0.8, -1.2,  0.4,  0.0,  1.1, ...]   64 values
after relu:      [ 0.0,  0.8,  0.0,  0.4,  0.0,  1.1, ...]   negatives killed
after .square(): [ 0.0,  0.64, 0.0,  0.16, 0.0,  1.21, ...]  positives amplified
```

Most values become zero (sparse) — the MLP only fires on features it's confident about.

---

### 3j — Cache x_backout at Layer N//2 = 2

```python
# GPT.forward lines 556–557
if i == 2:
    x_backout = x   # snapshot the residual stream at the halfway point
```

After layer 2, the stream holds a mixture of low-level features (syntax, token position,
surface form) and high-level features (meaning, context). This snapshot is used in Step 4.

---

## Step 4 — Backout

```python
# GPT.forward lines 562–563
x = x - self.backout_lambda * x_backout   # backout_lambda ≈ 0.2
```

`x_backout` (from layer 2) still carries the low-level surface features that were
built up in the first half of the network. Subtracting 20% of it nudges the final
representation away from those features, leaving the LM head with a cleaner
high-level signal.

```
x_final = x_layer4 - 0.2 * x_layer2
```

---

## Step 5 — Final RMSNorm + LM Head

```python
# GPT.forward lines 565–568
x = norm(x)                  # (1, 3, 16) — normalise before the output projection
logits = self.lm_head(x)     # (1, 3, 50257) — one score per vocab token
logits = logits.float()      # back to fp32 for numerical stability
```

`lm_head` is a `Linear(16, 50257)` — it maps each 16-dim token representation to
a score for every word in the vocabulary.

The model produces logits at every position:
```
logits[0, 0, :] — "what word follows 'cat'?"          (answer should be "drinks")
logits[0, 1, :] — "what word follows 'cat drinks'?"   (answer should be "milk")
logits[0, 2, :] — "what word follows 'cat drinks milk'?" (some continuation)
```

---

## Step 6 — Logit Softcapping

```python
# GPT.forward lines 574–575
softcap = 15.0
logits = softcap * torch.tanh(logits / softcap)
```

Without this, logits can grow very large as training progresses:
```
raw logit = 40.0  → 15 * tanh(40/15) = 15 * tanh(2.67) ≈ 14.9   (capped)
raw logit =  2.0  → 15 * tanh(2/15)  = 15 * 0.133      ≈  2.0   (unchanged)
raw logit = -8.0  → 15 * tanh(-8/15)                   ≈ -7.1   (mild squeeze)
```

All logits are bounded to `(−15, +15)`. `tanh` is smooth so gradients always flow —
unlike a hard clip which kills gradients outside the range.

---

## Step 7 — Loss (Cross-Entropy)

```python
# GPT.forward line 580
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

Targets are the input tokens shifted by one position — the model must predict the
next token at every position:

```
position 0: input="cat",    target="drinks" (id=15263)
position 1: input="drinks", target="milk"   (id=8336)
position 2: input="milk",   target=<next>   (whatever came after in the training data)
```

Cross-entropy measures how much probability the model assigned to the correct next token.
`loss.backward()` then computes gradients for every parameter, and the optimizer
(Muon for matrices, AdamW for embeddings/scalars) updates them.

---

## Step 8 — Optimizer Setup (setup_optimizer)

### What are scalar params and why are they different?

```python
scalar_params = [
    self.resid_lambdas,     # shape (4,)  — one blend scale per layer
    self.x0_lambdas,        # shape (4,)  — one anchor scale per layer
    self.smear_gate.weight, # shape (1, 24)
    self.smear_lambda,      # shape (1,)  — single number
    self.backout_lambda,    # shape (1,)  — single number
]
```

These are called "scalar" not because they're all single numbers, but because they're
**not the main weight matrices**. They're small control parameters — gates, blend factors,
scale factors. Three reasons they get AdamW instead of Muon:

- `resid_lambdas` is 1D `(4,)` — Newton-Schulz only works on 2D matrices
- `smear_lambda` and `backout_lambda` are single numbers — nothing to orthogonalise
- These parameters directly control the scale of the residual stream — aggressive
  Muon updates could destabilise training

### Why each parameter group gets different lr, betas, weight_decay

There's no formula — these are engineering judgements based on what each parameter does:

**Learning rates:**
```python
matrix_lr    = 0.02     # transformer weight matrices  → Muon
embedding_lr = 0.001    # wte, lm_head, value_embeds  → AdamW
scalar_lr    = 0.005    # resid_lambdas, smear_lambda  → AdamW
```
- `matrix_lr=0.02` is high because Muon's orthogonalised updates are already
  well-scaled — they don't need to be conservative
- `embedding_lr=0.001` is lower — embedding rows are sparse (most tokens don't
  appear in a batch), so you want smaller but more precise steps
- `scalar_lr=0.005` is in between — these numbers directly control stream scale,
  so careful updates are needed

**Betas (momentum decay rates):**
```python
betas=(0.8, 0.995)   # wte, value_embeds
betas=(0.8, 0.96)    # lm_head
betas=(0.8, 0.95)    # scalar params
```
- `beta1=0.8` (lower than standard 0.9) — shorter gradient memory. Embeddings
  get sparse updates so you don't want old stale gradients dominating
- `beta2` controls how fast the per-param learning rate adapts. Higher = more
  stable but slower to react to gradient magnitude changes

**Weight decay:**
```python
weight_decay=0.01    # lm_head, value_embeds
weight_decay=0.001   # wte — very small, tokens have very different frequencies
weight_decay=0.0     # scalar params — no decay at all
```
Weight decay shrinks parameters toward zero each step to prevent overfitting.
But scalar params like `resid_lambdas` must not be decayed — shrinking them toward
zero would break the residual stream scaling entirely.

### LR scaling across model sizes

```python
lr_scale = (n_embd / 768) ** -0.5
```

`768` is the embedding size of the original GPT-2 — used as the reference baseline.
`x ** -0.5` = `1 / sqrt(x)`, so this is:

```python
lr_scale = 1 / sqrt(n_embd / 768)
```

Concrete examples:
```python
n_embd = 768   → lr_scale = 1 / sqrt(1.0) = 1.0   # same as GPT-2, no change
n_embd = 3072  → lr_scale = 1 / sqrt(4.0) = 0.5   # 4× wider  → lr halved
n_embd = 192   → lr_scale = 1 / sqrt(0.25) = 2.0  # 4× narrower → lr doubled
```

Why: when you make the model wider, each layer has more parameters contributing to
the output, so the gradient signal per individual parameter is proportionally smaller.
Without compensation, a wider model would learn more slowly.
Multiplying lr by `lr_scale` keeps the effective update size consistent across widths.
All AdamW learning rates are multiplied by `lr_scale` before use.

### Grouping Muon params by shape

```python
for shape in sorted({p.shape for p in matrix_params}):
    group = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(
        kind='muon', params=group, lr=matrix_lr,
        momentum=0.95, ns_steps=5, weight_decay=weight_decay,
    ))
```

**Line 1 — collect all unique shapes:**

`{p.shape for p in matrix_params}` loops over every matrix parameter and collects
unique shapes. For our small config (n_embd=16), the transformer blocks have:
```
c_q:    (16, 16)   n_embd → n_head * head_dim
c_k:    (8,  16)   n_embd → n_kv_head * head_dim
c_v:    (8,  16)   same as c_k
c_proj: (16, 16)   attention output projection
c_fc:   (64, 16)   MLP expand 4×
c_proj: (16, 64)   MLP contract
```
Unique shapes = `{(16,16), (8,16), (64,16), (16,64)}`. `sorted()` gives consistent ordering.

**Line 2 — group all matrices of the same shape together:**

```python
group = [p for p in matrix_params if p.shape == shape]
```
For shape `(16, 16)` this collects `c_q` from layer 0, `c_q` from layer 1,
`c_proj` from layer 0, `c_proj` from layer 1... — all matrices of that shape
across all 4 layers in one list.

**Why group by shape?**

Newton-Schulz orthogonalises each matrix independently, but matrices of the same
shape can be processed as a batch in one GPU operation — faster than handling each
matrix one at a time. Grouping by shape is what enables that batching.

**Line 3 — append as a Muon param group:**

One param group per unique shape, all with `kind='muon'`. When `MuonAdamW.step()`
runs, it sees `kind='muon'` and routes to `_muon_step` for orthogonalised updates.

---

### How Newton-Schulz is applied inside `_muon_step`

```python
if update.ndim >= 2:
    shape = update.shape
    update = newton_schulz_orthogonalize(update.view(shape[0], -1), steps=ns_steps)
    update = update.view(shape)
```

**`if update.ndim >= 2`**
Newton-Schulz only works on 2D matrices. This skips orthogonalisation for any
1D parameter that might have ended up in the group.

**`shape = update.shape`**
Save the original shape so we can restore it after — e.g. `(64, 16)` for the MLP expand matrix.

**`update.view(shape[0], -1)`**
Reshapes to 2D — `shape[0]` keeps the first dimension, `-1` flattens everything
else. For most weight matrices this is already 2D so nothing changes:
```
(64, 16) → view(64, -1) → (64, 16)   unchanged
```
But if a weight were 3D:
```
(4, 16, 4) → view(4, -1) → (4, 64)   flattened to 2D
```
This ensures Newton-Schulz always receives a 2D matrix regardless of original shape.

**`newton_schulz_orthogonalize(..., steps=5)`**
Runs 5 iterations on the 2D update matrix. After this, each row points in a unique
direction — no two neurons are moving the same way.

**`update.view(shape)`**
Reshapes back to the original shape. Values have changed (now orthogonal) but the
shape is restored to match the weight it will be applied to.

Full flow for the MLP expand matrix:
```
momentum buffer: (64, 16)
     ↓ view(64, -1)
2D matrix:       (64, 16)
     ↓ newton_schulz (5 steps)
orthogonal:      (64, 16)  ← rows now point in unique directions
     ↓ view(64, 16)
update:          (64, 16)  ← applied to the weight with p.add_(update, alpha=-lr)
```

---

### Weight decay and gradient update

```python
if wd != 0.0:
    p.mul_(1.0 - lr * wd)   # weight decay first

p.add_(update, alpha=-lr)   # then apply gradient update
```

**Weight decay** (`p.mul_`): multiplies the weight in-place by a number slightly below 1.
With `lr=0.02` and `wd=0.01`:
```
factor = 1.0 - 0.02 * 0.01 = 0.9998
weight = weight * 0.9998    ← shrinks 0.02% each step toward zero
```

**Gradient update** (`p.add_`): `p = p + alpha * update` with `alpha=-lr`:
```
p = p - 0.02 * update       ← descend the loss
```

**Why weight decay is needed separately from orthogonalisation:**

Orthogonalised updates control the **direction** of each step — making every neuron
move in a unique direction so no steps are wasted. But they say nothing about how
large the weights grow over time.

Without weight decay, a weight that keeps receiving updates in the same direction
accumulates indefinitely:
```
step 1:    w = 0.5  → 0.8
step 2:    w = 0.8  → 1.1
step 1000: w = 50.3   ← keeps growing
```

Very large weights cause logits to explode, gradients to become unstable, and the
model to become brittle. Weight decay adds a counter-force that creates an equilibrium:
```
step 1000: w = 50.3 * 0.9998 + update
```
The gradient update pushes the weight up; weight decay pulls it back down.
The weight stabilises at a size where both forces balance.

One full step:
```
w = 1.5234
w = w * 0.9998        = 1.5231   ← decay first
w = w - 0.02 * update = 1.5225   ← then gradient step
```

---

### AdamW weight decay and gradient update

```python
p.data.mul_(1.0 - lr * wd)          # weight decay
p.data.addcdiv_(m, denom, value=-step_size)  # gradient update
```

**Weight decay** — identical concept to Muon:
```python
p.data.mul_(1.0 - lr * wd)
```
`.data` accesses the raw tensor directly, bypassing autograd — the optimizer
update itself shouldn't be tracked as a computation.

**`p.data.addcdiv_(m, denom, value=-step_size)`**

`addcdiv_` means: `p = p + value * (m / denom)`. With `value=-step_size`:
```
p = p - step_size * (m / denom)
```

- `m`     = running average of gradient direction (which way to move)
- `denom` = `sqrt(v) + eps` = running average of gradient magnitude

So `m / denom` = direction divided by recent magnitude.

**Why divide by magnitude — the adaptive part:**

```
param A: large recent gradients → denom is large → smaller step  (careful)
param B: small recent gradients → denom is small → larger step   (compensate)
```

Each parameter gets its own effective learning rate based on its gradient history.
Rare tokens in the embedding get large steps (they rarely update).
Parameters with consistently large gradients get small, careful steps.

**Muon vs AdamW — side by side:**
```
Muon:   p = p - lr * orthogonalise(momentum)      adjusts direction
AdamW:  p = p - step_size * (m / denom)           adjusts step size per parameter
```
Muon makes every neuron move in a unique direction.
AdamW makes the step size self-adjust based on each parameter's gradient history.

---

## Summary — Shape Flow

```
idx                      (1, 3)           token IDs
wte(idx)                 (1, 3, 16)       token embeddings, fp32
x.to(bf16)               (1, 3, 16)       cast to bf16
norm(x)                  (1, 3, 16)       RMSNorm
smear gate               (1, 3, 16)       bigram blend
── × 4 transformer blocks ──
  resid scale            (1, 3, 16)       λ·x + α·x₀
  norm + attention       (1, 3, 16)       GQA + RoPE + QK Norm + SDPA
  residual add           (1, 3, 16)       x = x + attn_out
  norm + MLP             (1, 3, 16)       Linear → ReLU² → Linear
  residual add           (1, 3, 16)       x = x + mlp_out
──────────────────────────────
backout                  (1, 3, 16)       x = x - 0.2 * x_layer2
norm(x)                  (1, 3, 16)       final normalisation
lm_head(x)               (1, 3, 50257)    logits, fp32
softcap                  (1, 3, 50257)    bounded to ±15
cross_entropy            scalar           loss
```
