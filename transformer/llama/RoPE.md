# RoPE (Rotary Position Embeddings) in Llama — explained

## What this code does, at a high level

This implements **Rotary Position Embeddings (RoPE)** — the mechanism Llama uses to tell attention "where" each token is in the sequence, by rotating query/key vectors based on position instead of adding a positional vector.

## The pieces, and how they connect

**1. `LlamaRotaryEmbedding.__init__`**
- Reads `rope_theta` (base) and `rope_type` ("default", "linear", "dynamic", "yarn", etc.) from config.
- Picks a function `rope_init_fn` based on `rope_type` — for vanilla Llama this is `compute_default_rope_parameters`.
- Calls that function once to get `inv_freq` (a vector of inverse frequencies) and `attention_scaling` (a scalar multiplier, usually `1.0` for default RoPE, different for scaled variants like YaRN).
- Stores `inv_freq` as a buffer (not a learned parameter, but moves with `.to(device)`/`.to(dtype)` calls). `original_inv_freq` is kept as a pristine copy so dynamic RoPE variants can recompute from the same base later.

**2. `compute_default_rope_parameters`** (static method, the actual math)
- `dim` = size of each attention head.
- Computes `inv_freq[i] = 1 / theta^(2i/dim)` for `i = 0, 2, 4, ... dim-2`. This gives a geometric range of frequencies — early indices rotate fast (capture local/short-range position info), later indices rotate slowly (capture long-range position info).
- Returns `(inv_freq, attention_factor=1.0)`.

**3. `forward(x, position_ids)`** — uses `inv_freq` to build per-position `cos`/`sin` tables
- `inv_freq_expanded`: reshapes `inv_freq` to `[batch, dim/2, 1]`.
- `position_ids_expanded`: reshapes positions to `[batch, 1, seq_len]`.
- Matrix-multiplying these gives `freqs`: shape `[batch, seq_len, dim/2]` — basically `position * inv_freq` for every (position, frequency) pair.
- `emb = cat(freqs, freqs)` doubles it to `dim` width (because `rotate_half` later splits the vector into two halves and needs matching frequencies on both halves).
- `cos`/`sin` of that, scaled by `attention_scaling`, are the final outputs — one cos/sin pair per token position, shared across all attention heads/layers that call this module.
- Computed in float32 inside `maybe_autocast(..., enabled=False)` for numerical precision, then cast back to the model's dtype.

So: `__init__` sets up *constants* (`inv_freq`), `forward` turns those constants + *actual positions* into per-token `cos`/`sin` tables.

**4. `rotate_half(x)`**
- Splits a vector in half: `x1` (first half), `x2` (second half).
- Returns `[-x2, x1]` — this is the 90°-rotation trick: RoPE treats pairs of dimensions as 2D vectors and rotates them by the position-dependent angle. Pairing dim `i` with dim `i + dim/2` (rather than adjacent `i, i+1`) is just Llama's interleaving convention — mathematically equivalent to true complex-pair rotation.

**5. `apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)`** — where it all comes together
- `cos`/`sin` come in shaped `[batch, seq_len, head_dim]`; `unsqueeze` inserts a heads-dimension so they broadcast against `q`/`k` shaped `[batch, heads, seq_len, head_dim]`.
- Standard 2D rotation formula applied elementwise:
  `q_rotated = q * cos + rotate_half(q) * sin`
  This is exactly `[x*cosθ - y*sinθ, y*cosθ + x*sinθ]` for each (x, y) dimension pair, i.e. rotating the embedding vector by angle `θ = position * inv_freq`.
- Same rotation applied to both `q` and `k`. Because rotation is applied identically to both, the dot product `q·k` in attention ends up depending only on the *relative* position `(pos_q - pos_k)`, not absolute position — that's RoPE's key property.

## `rotate_half` and `apply_rotary_pos_emb` — full code walkthrough

```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

- `x.shape[-1]` is `head_dim` (e.g. 8 in the worked example). `x.shape[-1] // 2` is the midpoint (4).
- `x1` = first half of the last dimension (the "x" half of every pair, in split-half layout).
- `x2` = second half (the "y" half of every pair).
- `torch.cat((-x2, x1), dim=-1)` builds: negated second half first, then the original first half — i.e. `[-y0,-y1,-y2,-y3, x0,x1,x2,x3]`.
- `...` (ellipsis) means "keep whatever leading dimensions exist (batch, heads, seq_len), only slice the last one" — so this works regardless of tensor rank, as long as `head_dim` is the last axis.

```python
@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**The decorator** `@use_kernel_func_from_hub("rotary_pos_emb")` lets this function be transparently swapped for a faster, hand-written CUDA/Triton kernel pulled from the Hub when available for the current hardware/dtype, falling back to this pure-PyTorch implementation otherwise. It's purely a performance hook — the math is unchanged.

**Why `unsqueeze_dim` is needed (the shape problem):**

`cos`/`sin` come out of `LlamaRotaryEmbedding.forward` with shape `[batch, seq_len, head_dim]` — they don't know about "heads" because the same rotation angle applies identically to every head (heads don't have separate positions). But `q`/`k` have shape `[batch, heads, seq_len, head_dim]` — there's an extra "heads" axis in the middle.

To compute `q * cos` elementwise, the shapes must broadcast, and `[batch, seq_len, head_dim]` doesn't line up against `[batch, heads, seq_len, head_dim]` positionally. So:

```python
cos = cos.unsqueeze(unsqueeze_dim)   # [batch, seq_len, head_dim] → [batch, 1, seq_len, head_dim]
```

With `unsqueeze_dim=1`, a size-1 axis is inserted at position 1. Now `cos` is `[batch, 1, seq_len, head_dim]`, which broadcasts cleanly against `q`'s `[batch, heads, seq_len, head_dim]` — the size-1 "heads" axis is automatically repeated across all real heads. (If `q`/`k` instead use the layout `[batch, seq_len, heads, head_dim]`, pass `unsqueeze_dim=2` so the inserted axis lands in the right spot.)

**The rotation itself**, applied identically to both tensors:
```python
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```
This is the elementwise rotation formula derived in the worked example above. Both `q` and `k` must be rotated — that's the entire point of RoPE: rotating both with the same angle-per-position scheme makes their later dot product (`q_embed @ k_embed.T` in attention) depend only on relative position, not absolute position. The returned `(q_embed, k_embed)` flow straight into `softmax(q_embed @ k_embed.T / sqrt(d)) @ v`.

## The call chain in a real forward pass

```
LlamaModel.forward
  └─ rotary_emb(x, position_ids)        # computes cos, sin once per layer-stack pass
       └─ uses self.inv_freq (built once at init via compute_default_rope_parameters)

LlamaAttention.forward (per layer)
  └─ apply_rotary_pos_emb(q, k, cos, sin)   # rotates this layer's q/k before attention dot product
       └─ rotate_half(q), rotate_half(k)
```

`cos`/`sin` are computed **once** per forward pass (depend only on position_ids, not on layer), then reused by every layer's attention module to rotate that layer's own `q`/`k`.

## Hierarchical structure of all components

```
LlamaRotaryEmbedding (nn.Module)                         ← created once, lives on LlamaModel
│
├── __init__(config, device)
│   ├── reads config.rope_parameters["rope_type"]         ("default" / "linear" / "dynamic" / "yarn" / ...)
│   ├── selects rope_init_fn
│   │     ├── "default" → compute_default_rope_parameters (static method, below)
│   │     └── other     → ROPE_INIT_FUNCTIONS[rope_type]   (external dict, not shown above)
│   ├── calls rope_init_fn(config, device)
│   │     └── returns (inv_freq, attention_scaling)
│   ├── register_buffer("inv_freq", inv_freq)              ← geometric frequency schedule
│   └── register_buffer("original_inv_freq", inv_freq.clone())  ← pristine copy for dynamic RoPE
│
├── compute_default_rope_parameters(config, device, seq_len)   [staticmethod — the math]
│   ├── reads base = rope_theta, dim = head_dim                     e.g. base=10000, dim=8
│   ├── inv_freq[i] = 1 / theta^(2i/dim)               for i=0..3 → [1.0, 0.1, 0.01, 0.001]  ← θ₀..θ₃
│   └── returns (inv_freq, attention_factor=1.0)
│
└── forward(x, position_ids)                               ← called once per model forward pass
    │                                                          e.g. position_ids = [0,1,2,3,4,5]  ("The cat sat on the mat")
    ├── inv_freq_expanded  = inv_freq reshaped to [batch, dim/2, 1]        → [1.0, 0.1, 0.01, 0.001] per batch
    ├── position_ids_expanded = position_ids reshaped to [batch, 1, seq_len]  → [0,1,2,3,4,5]
    ├── freqs = inv_freq_expanded @ position_ids_expanded   (matmul → [batch, seq_len, dim/2])
    │     └── outer product = angle table: freqs[pos] = position × θᵢ
    │           "cat" (pos=1) → [1.0, 0.1, 0.01, 0.001]
    │           "mat" (pos=5) → [5.0, 0.5, 0.05, 0.005]
    ├── emb = cat(freqs, freqs)                              (doubled to full head_dim)
    │     └── "cat" → [1.0,0.1,0.01,0.001, 1.0,0.1,0.01,0.001]   (8 dims, duplicated halves)
    ├── cos = emb.cos() * attention_scaling                  e.g. cos("cat") = [0.540,0.995,0.9999,1.0, 0.540,0.995,0.9999,1.0]
    ├── sin = emb.sin() * attention_scaling                  e.g. sin("cat") = [0.841,0.0998,0.0100,0.001, 0.841,0.0998,0.0100,0.001]
    └── returns (cos, sin)                                   ← shared across all layers, one row per word

rotate_half(x)                                             ← standalone helper function
    ├── x1 = first half of x                                e.g. "cat" split-half q = [0.4,0.2,0.3,0.7 | 0.6,0.8,0.5,0.1]
    ├── x2 = second half of x                                       x1=[0.4,0.2,0.3,0.7], x2=[0.6,0.8,0.5,0.1]
    └── returns cat(-x2, x1)                                 (90° rotation trick) → [-0.6,-0.8,-0.5,-0.1, 0.4,0.2,0.3,0.7]

apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)         ← standalone function, called per attention layer
    ├── cos, sin  ← unsqueeze to broadcast over the heads dimension
    ├── q_embed = q * cos + rotate_half(q) * sin             (uses rotate_half)
    │     └── "cat" dim0: 0.4·cos(1.0) + (−0.6)·sin(1.0) = −0.289   (matches hand-computed Pair 0 x_new)
    ├── k_embed = k * cos + rotate_half(k) * sin             (uses rotate_half)
    └── returns (q_embed, k_embed)                           ← fed into attention's softmax(QK^T)
          └── q_embed("cat") · k_embed("mat") ≈ cos(gap × θᵢ) summed over pairs → encodes "4 positions apart", not absolute position
```

### Ownership / call hierarchy across the model

```
LlamaModel
└── self.rotary_emb = LlamaRotaryEmbedding(config)          ← one instance, shared by all layers
    │
    └── forward pass:
        cos, sin = self.rotary_emb(hidden_states, position_ids)   [computed once]
        │
        for each LlamaDecoderLayer:
            └── LlamaAttention.forward(hidden_states, ..., cos, sin)
                ├── q, k, v = projections(hidden_states)
                ├── q, k = apply_rotary_pos_emb(q, k, cos, sin)    [reused cos/sin, per-layer q/k]
                │         └── rotate_half(q), rotate_half(k)
                └── attn_output = softmax(q @ k.T / sqrt(d)) @ v
```

**Key takeaway on hierarchy:** `LlamaRotaryEmbedding` is instantiated once per model and computes `cos`/`sin` once per forward pass; `apply_rotary_pos_emb` + `rotate_half` are stateless functions called independently inside *every* attention layer, reusing the same `cos`/`sin` but rotating that layer's own `q`/`k` tensors.

## Worked numeric example: mapping the code to actual numbers

Setup: sentence `"The cat sat on the mat"` at positions `0..5`, `dim = 8` (so 4 frequency pairs), `theta = 10000`.

### Stage 1 → `compute_default_rope_parameters`

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
```

`torch.arange(0, 8, 2) = [0, 2, 4, 6]`, divided by `dim=8` gives `[0, 0.25, 0.5, 0.75]`:

```
inv_freq = 1/10000^[0, 0.25, 0.5, 0.75] = [1.0, 0.1, 0.01, 0.001]
```

These are the θᵢ values — early indices (θ₀=1.0) rotate fast (local detail), later indices (θ₃=0.001) rotate slowly (long-range structure). Computed once in `__init__`, stored as `self.inv_freq`.

### Stage 2 → `forward`'s matmul

```python
inv_freq_expanded      # [1, 4, 1]  → [1.0, 0.1, 0.01, 0.001] per batch
position_ids_expanded  # [1, 1, 6] → [0,1,2,3,4,5]
freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [1, 6, 4]
```

The matmul is an outer product of positions × frequencies — exactly the angle table:

```
freqs[0] (pos 0, "The") = [0, 0, 0, 0]
freqs[1] (pos 1, "cat") = [1.0, 0.1, 0.01, 0.001]
freqs[2] (pos 2, "sat") = [2.0, 0.2, 0.02, 0.002]
freqs[5] (pos 5, "mat") = [5.0, 0.5, 0.05, 0.005]
```

After `transpose`, shape is `[batch, seq_len, dim/2] = [1, 6, 4]` — row = word, column = pair, matching the angle table directly.

### Stage 2.5 → `emb = cat(freqs, freqs)`, then `cos`/`sin`

```python
emb = torch.cat((freqs, freqs), dim=-1)   # [1, 6, 4] → [1, 6, 8]
cos = emb.cos()
sin = emb.sin()
```

For "cat" (row 1): `freqs[1] = [1.0, 0.1, 0.01, 0.001]`, so `emb[1] = [1.0, 0.1, 0.01, 0.001, 1.0, 0.1, 0.01, 0.001]` — the 4 angles duplicated to fill all 8 dims. `cos[1]`/`sin[1]` then contain `cos(1.0)=0.5403`, `sin(1.0)=0.8415`, etc., repeated across both halves.

### Stage 3/4 → `apply_rotary_pos_emb`

Manual per-pair formula:
```
x_new = x·cos(α) − y·sin(α)
y_new = x·sin(α) + y·cos(α)
```

Code's vectorized formula:
```python
q_embed = (q * cos) + (rotate_half(q) * sin)
```

These are the same operation applied to all 4 pairs at once. Note: real Llama code uses a *split-half* layout (`[x0,x1,x2,x3, y0,y1,y2,y3]`), not an interleaved one — `rotate_half` splits into first-half/second-half rather than alternating pairs, but the math is equivalent.

"cat"'s embedding in split-half layout: `q = [0.4, 0.2, 0.3, 0.7, 0.6, 0.8, 0.5, 0.1]`.

`rotate_half(q) = [-0.6, -0.8, -0.5, -0.1, 0.4, 0.2, 0.3, 0.7]`.

Computing `q*cos + rotate_half(q)*sin` elementwise across all 8 dims reproduces each pair's rotation:

- dim 0 (`x0`): `0.4·cos(1.0) + (−0.6)·sin(1.0) = 0.4·0.5403 − 0.6·0.8415 = −0.289`
- dim 4 (`y0`): `0.6·cos(1.0) + 0.4·sin(1.0) = 0.6·0.5403 + 0.4·0.8415 = 0.661`

Same pattern for dims 1/5, 2/6, 3/7. This is exactly why `cos`/`sin` had to be duplicated (`cat(freqs,freqs)`) in Stage 2.5 — dim `i` and its rotation partner `i+4` need the *same* angle, and the duplication guarantees `cos[0]==cos[4]`, `cos[1]==cos[5]`, etc.

#### How the manual formula and the vectorized formula relate

The manual formula rotates **one 2D pair** `(x, y)` by angle `α`. The vectorized formula rotates **all pairs in the vector at once** by packing every pair's `x` into the first half and every pair's `y` into the second half, then doing one elementwise operation across the whole vector.

##### The starting point

"cat" has an 8-number embedding, treated as 4 pairs `(x_i, y_i)`. Llama's layout puts all 4 `x`'s first, then all 4 `y`'s:

```
q = [x0, x1, x2, x3,  y0, y1, y2, y3]
   = [0.4, 0.2, 0.3, 0.7,  0.6, 0.8, 0.5, 0.1]
```

So `x0=0.4, y0=0.6` is pair 0, `x1=0.2, y1=0.8` is pair 1, etc.

##### What we want to compute

For each pair independently, rotate by its own angle — 8 small calculations total (2 per pair × 4 pairs):

```
pair 0, angle α0=1.0:   x0_new = x0·cos(α0) − y0·sin(α0)
                        y0_new = x0·sin(α0) + y0·cos(α0)
... and so on for pairs 1, 2, 3
```

The question is how one line, `q*cos + rotate_half(q)*sin`, computes all 8 at once.

##### Building `rotate_half(q)`

`rotate_half` mechanically takes the second half, negates it, and swaps it to the front; the first half moves to the back:

```
q              = [x0, x1, x2, x3,  y0, y1, y2, y3]
rotate_half(q) = [-y0, -y1, -y2, -y3,  x0, x1, x2, x3]
```

With numbers:
```
q              = [0.4, 0.2, 0.3, 0.7,  0.6, 0.8, 0.5, 0.1]
rotate_half(q) = [-0.6, -0.8, -0.5, -0.1,  0.4, 0.2, 0.3, 0.7]
```

Position 0 of `rotate_half(q)` is `-y0` (negated partner from position 4). Position 4 is `x0` (copy from position 0). This swap is the entire trick.

##### Building `cos` and `sin`

Each pair has its own angle (`α0=1.0, α1=0.1, α2=0.01, α3=0.001`). Both `x_i` and `y_i` need the *same* angle, so `cos`/`sin` are built by duplicating the 4 angle values into 8 slots:

```
cos = [cos(α0), cos(α1), cos(α2), cos(α3),  cos(α0), cos(α1), cos(α2), cos(α3)]
sin = [sin(α0), sin(α1), sin(α2), sin(α3),  sin(α0), sin(α1), sin(α2), sin(α3)]
```

So `cos[0] == cos[4]` (both `cos(α0)`), `cos[1] == cos[5]` (both `cos(α1)`), etc.

##### Computing `q*cos + rotate_half(q)*sin` slot by slot

This is 8 independent multiply-and-add operations, one per index. Index 0 and index 4 (pair 0's two slots):

**Index 0:**
```
q[0]=x0, cos[0]=cos(α0), rotate_half(q)[0]=-y0, sin[0]=sin(α0)

result[0] = x0·cos(α0) + (-y0)·sin(α0) = x0·cos(α0) − y0·sin(α0)
```
Identical to the manual `x_new = x·cos(α) − y·sin(α)`.
Numbers: `0.4·0.5403 − 0.6·0.8415 = 0.2161 − 0.5049 = −0.289` ✓

**Index 4:**
```
q[4]=y0, cos[4]=cos(α0) (same angle, thanks to duplication),
rotate_half(q)[4]=x0 (where x0 got moved to), sin[4]=sin(α0)

result[4] = y0·cos(α0) + x0·sin(α0) = x0·sin(α0) + y0·cos(α0)
```
Identical to the manual `y_new = x·sin(α) + y·cos(α)`.
Numbers: `0.6·0.5403 + 0.4·0.8415 = 0.3242 + 0.3366 = 0.661` ✓

##### Why it works — the one key insight

`rotate_half` does two things simultaneously:

1. **Negation**: puts `-y_i` where `x_i` used to be (paired with `sin`), giving the `−y·sin(α)` term needed for `x_new`.
2. **Swap**: puts `x_i` where `y_i` used to be (paired with `sin` again), giving the `+x·sin(α)` term needed for `y_new`.

Combined with `cos`/`sin` being duplicated so the *same angle* appears at both an `x` slot and its partner `y` slot, one elementwise multiply-add silently performs 4 independent 2D rotations — one per pair — in a single vectorized instruction, instead of looping over pairs. Indices `1/5`, `2/6`, `3/7` repeat this exact pattern for pairs 1, 2, 3 with their own angles `α1, α2, α3`.

| Manual (per-pair, scalar) | Vectorized (code, whole tensor) |
|---|---|
| `x` | `q[i]` (first-half slot) |
| `y` | `q[i + dim/2]` (second-half slot, same pair) |
| `−y·sin(α)` term | comes from `rotate_half(q)[i] = -y`, multiplied by `sin[i]` |
| `+x·sin(α)` term | comes from `rotate_half(q)[i+dim/2] = x`, multiplied by `sin[i+dim/2]` |
| one pair, one angle `α` | all pairs at once, because `cos`/`sin` hold every pair's angle and the duplication (`cat(freqs,freqs)`) lines each angle up with both halves of its pair |

So `rotate_half` is the trick that lets `q*cos + rotate_half(q)*sin` apply the manual 2-line rotation formula to *every* pair simultaneously in a single elementwise multiply-add, instead of looping over pairs.

### Stage 5 → the dot product / attention

This part isn't inside `apply_rotary_pos_emb` — it happens later in `LlamaAttention.forward` when it computes `q @ k.T`. The code never explicitly proves the identity `cos(A)cos(B)+sin(A)sin(B) = cos(B−A)`; it just relies on it being mathematically true. Rotating both `q` and `k` with this code, then dotting them, makes the attention score depend only on relative position (`gap × θᵢ`) — that's the payoff the code is engineered to deliver, even though the code itself only performs the rotation, not the proof.

### Mapping summary

| Walkthrough concept | Code |
|---|---|
| θᵢ values | `inv_freq` (from `compute_default_rope_parameters`) |
| angle table (position × θ) | `freqs` (the matmul in `forward`) |
| cos/sin per pair | `cos`, `sin` (after `emb = cat(freqs,freqs)`) |
| rotate [x,y] by α | `q*cos + rotate_half(q)*sin` in `apply_rotary_pos_emb` |
| dot product → cos(gap·θ) | happens later, in attention's `q @ k.T` |
