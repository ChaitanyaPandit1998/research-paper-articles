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
│   ├── reads base = rope_theta, dim = head_dim
│   ├── inv_freq[i] = 1 / theta^(2i/dim)
│   └── returns (inv_freq, attention_factor=1.0)
│
└── forward(x, position_ids)                               ← called once per model forward pass
    ├── inv_freq_expanded  = inv_freq reshaped to [batch, dim/2, 1]
    ├── position_ids_expanded = position_ids reshaped to [batch, 1, seq_len]
    ├── freqs = inv_freq_expanded @ position_ids_expanded   (matmul → [batch, seq_len, dim/2])
    ├── emb = cat(freqs, freqs)                              (doubled to full head_dim)
    ├── cos = emb.cos() * attention_scaling
    ├── sin = emb.sin() * attention_scaling
    └── returns (cos, sin)                                   ← shared across all layers

rotate_half(x)                                             ← standalone helper function
    ├── x1 = first half of x
    ├── x2 = second half of x
    └── returns cat(-x2, x1)                                 (90° rotation trick)

apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)         ← standalone function, called per attention layer
    ├── cos, sin  ← unsqueeze to broadcast over the heads dimension
    ├── q_embed = q * cos + rotate_half(q) * sin             (uses rotate_half)
    ├── k_embed = k * cos + rotate_half(k) * sin             (uses rotate_half)
    └── returns (q_embed, k_embed)                           ← fed into attention's softmax(QK^T)
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
