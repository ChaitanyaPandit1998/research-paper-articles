# Llama Model — Tensor Shape Tracking

Concrete example: **Llama 3 8B**
- `B` = batch size = 2
- `S` = sequence length = 512  
- `H` = hidden size = 4096
- `n_heads` = 32 (Q heads)
- `n_kv_heads` = 8 (K/V heads)
- `head_dim` = 128  (H / n_heads = 4096 / 32)
- `n_kv_groups` = 4  (n_heads / n_kv_heads = 32 / 8)
- `intermediate_size` = 14336
- `vocab_size` = 128256
- `num_layers` = 32

---

## Stage 0 — Input

```
input_ids
  [B, S]  =  [2, 512]
  dtype: int64
  values: token IDs, e.g. [[101, 42, 7, ...], [101, 99, ...]]
```

---

## Stage 1 — Token Embedding

```
embed_tokens: nn.Embedding(vocab_size=128256, hidden_size=4096)

inputs_embeds = embed_tokens(input_ids)
  [B, S]  →  [B, S, H]
  [2, 512] →  [2, 512, 4096]
  dtype: bfloat16

hidden_states = inputs_embeds
  [2, 512, 4096]   ← enters the decoder stack
```

---

## Stage 2 — RoPE Tables (computed ONCE, shared by all 32 layers)

```
position_ids = [0, 1, 2, ..., 511]  →  unsqueeze  →  [B, S]  =  [2, 512]

LlamaRotaryEmbedding.forward(hidden_states, position_ids):

  inv_freq                    [head_dim/2]        = [64]
  inv_freq_expanded           [B, head_dim/2, 1]  = [2, 64, 1]
  position_ids_expanded       [B, 1, S]           = [2, 1, 512]

  freqs = inv_freq_expanded @ position_ids_expanded
                              [B, head_dim/2, S]  = [2, 64, 512]
  freqs.transpose(1,2)        [B, S, head_dim/2]  = [2, 512, 64]

  emb = cat(freqs, freqs, dim=-1)
                              [B, S, head_dim]    = [2, 512, 128]

  cos = emb.cos()             [2, 512, 128]
  sin = emb.sin()             [2, 512, 128]
```

---

## Stage 3 — Causal Mask (computed ONCE, shared by all 32 layers)

```
causal_mask                   [B, 1, S, S]        = [2, 1, 512, 512]
  dtype: bfloat16
  values: 0.0 at allowed positions, -inf at masked (future) positions
  the 1 in dim-1 broadcasts over all n_heads automatically
```

---

## Stage 4 — Decoder Layer (×32)
*Everything below repeats for each of the 32 LlamaDecoderLayer blocks.
Shapes are identical across layers — only the weights differ.*

### 4a. Pre-Attention RMSNorm

```
residual = hidden_states      [2, 512, 4096]   ← saved for residual add

LlamaRMSNorm(hidden_size=4096):
  upcast to float32           [2, 512, 4096]
  variance = mean(x², dim=-1) [2, 512, 1]
  x_norm = x / sqrt(var + ε) [2, 512, 4096]
  cast back to bfloat16
  × self.weight               [4096]  (broadcast)
  output                      [2, 512, 4096]   ← same shape as input
```

### 4b. QKV Projections

```
q_proj: Linear(4096  →  32 × 128 = 4096)
k_proj: Linear(4096  →   8 × 128 = 1024)
v_proj: Linear(4096  →   8 × 128 = 1024)

hidden_states [2, 512, 4096]  →  q_proj  →  [2, 512, 4096]
                               →  k_proj  →  [2, 512, 1024]
                               →  v_proj  →  [2, 512, 1024]

Reshape + transpose to put heads on dim-1:
  query_states  [B, n_heads,    S, head_dim]  = [2, 32, 512, 128]
  key_states    [B, n_kv_heads, S, head_dim]  = [2,  8, 512, 128]
  value_states  [B, n_kv_heads, S, head_dim]  = [2,  8, 512, 128]
```

### 4c. Apply RoPE

```
cos, sin each [2, 512, 128]
  unsqueeze(1) →  [2, 1, 512, 128]   (broadcasts over heads)

q_embed = (query_states × cos) + (rotate_half(query_states) × sin)
  [2, 32, 512, 128]   ← shape unchanged, values rotated

k_embed = (key_states × cos) + (rotate_half(key_states) × sin)
  [2,  8, 512, 128]   ← shape unchanged, values rotated
```

### 4d. KV Cache Update (inference only)

```
past_key_values.update(key_states, value_states, layer_idx)

On token 1 (prompt phase, S=512):
  key cache    [2,  8, 512, 128]
  value cache  [2,  8, 512, 128]

On token 513 (generation phase, S=1):
  new key_states   [2,  8,   1, 128]
  appended →       [2,  8, 513, 128]   ← grows by 1 each step
  
  query_states stays [2, 32, 1, 128]   (only the new token)
  key/value cache   [2,  8, T, 128]    (full history, T grows)
```

### 4e. GQA — repeat_kv

```
n_kv_groups = n_heads / n_kv_heads = 32 / 8 = 4

repeat_kv(key_states, n_rep=4):
  input    [2,  8, 512, 128]
  expand   [2,  8,   4, 512, 128]   (each KV head repeated 4×)
  reshape  [2, 32, 512, 128]        (merged back)

key_states    [2, 32, 512, 128]   ← now matches Q head count
value_states  [2, 32, 512, 128]
```

### 4f. Scaled Dot-Product Attention

```
scaling = head_dim ** -0.5 = 128 ** -0.5 ≈ 0.0884

attn_weights = query_states @ key_states.T × scaling
  [2, 32, 512, 128] @ [2, 32, 128, 512]  →  [2, 32, 512, 512]
  dtype: bfloat16

+ causal_mask [2, 1, 512, 512]   (broadcast over 32 heads)
  [2, 32, 512, 512]

softmax(dim=-1) in float32, cast back to bfloat16
  [2, 32, 512, 512]

attn_output = attn_weights @ value_states
  [2, 32, 512, 512] @ [2, 32, 512, 128]  →  [2, 32, 512, 128]

transpose(1, 2).contiguous()
  [2, 512, 32, 128]
```

### 4g. Output Projection + First Residual

```
reshape to merge heads:
  [2, 512, 32, 128]  →  [2, 512, 4096]   (32 × 128 = 4096)

o_proj: Linear(4096 → 4096)
  [2, 512, 4096]  →  [2, 512, 4096]

residual add:
  hidden_states = residual + attn_output
  [2, 512, 4096] + [2, 512, 4096]  =  [2, 512, 4096]
```

### 4h. Pre-MLP RMSNorm

```
residual = hidden_states      [2, 512, 4096]   ← saved again

LlamaRMSNorm(hidden_size=4096)
  input   [2, 512, 4096]
  output  [2, 512, 4096]      ← same shape
```

### 4i. SwiGLU MLP

```
gate_proj: Linear(4096 → 14336)
up_proj:   Linear(4096 → 14336)
down_proj: Linear(14336 → 4096)

gate = gate_proj(x)           [2, 512, 14336]
up   = up_proj(x)             [2, 512, 14336]

gate after SiLU activation    [2, 512, 14336]
gate × up  (elementwise)      [2, 512, 14336]

down_proj(gate × up)          [2, 512, 14336]  →  [2, 512, 4096]
```

### 4j. Second Residual

```
hidden_states = residual + mlp_output
  [2, 512, 4096] + [2, 512, 4096]  =  [2, 512, 4096]

→ passed to the next decoder layer (back to 4a)
  shape stays [2, 512, 4096] through all 32 layers
```

---

## Stage 5 — Final RMSNorm

```
hidden_states = norm(hidden_states)
  [2, 512, 4096]  →  [2, 512, 4096]   ← same shape
```

---

## Stage 6 — LM Head

```
lm_head: Linear(4096 → 128256, bias=False)
  weights tied to embed_tokens

logits = lm_head(hidden_states[:, -logits_to_keep:, :])
  full case:        [2, 512, 4096]  →  [2, 512, 128256]
  generation case:  [2,   1, 4096]  →  [2,   1, 128256]
                                       ↑ only last token needed
```

---

## Stage 7 — Loss (training only)

```
labels  [B, S]  =  [2, 512]   (input_ids shifted by 1)

cross_entropy(
  logits.view(-1, vocab_size)  [B×S, V]    = [1024, 128256]
  labels.view(-1)              [B×S]        = [1024]
)
→ scalar loss
```

---

## Shape Summary Table

| Tensor | Shape | Notes |
|--------|-------|-------|
| `input_ids` | [2, 512] | int64 token IDs |
| `inputs_embeds` | [2, 512, 4096] | after embed_tokens |
| `position_ids` | [2, 512] | 0…511 per batch |
| `inv_freq` | [64] | head_dim/2 frequencies |
| `cos / sin` | [2, 512, 128] | RoPE tables, shared across layers |
| `causal_mask` | [2, 1, 512, 512] | upper-triangular -inf mask |
| `hidden_states` (per layer) | [2, 512, 4096] | invariant across all 32 layers |
| `query_states` | [2, 32, 512, 128] | post-RoPE |
| `key_states` (pre-GQA) | [2, 8, 512, 128] | 8 KV heads |
| `key_states` (post-GQA) | [2, 32, 512, 128] | after repeat_kv ×4 |
| `attn_weights` | [2, 32, 512, 512] | before softmax |
| `attn_output` (pre-reshape) | [2, 512, 32, 128] | after transpose |
| `attn_output` (post-reshape) | [2, 512, 4096] | after merge heads |
| `gate / up` | [2, 512, 14336] | MLP intermediate |
| `mlp_output` | [2, 512, 4096] | after down_proj |
| `logits` (full) | [2, 512, 128256] | training |
| `logits` (generation) | [2, 1, 128256] | inference |

---

## Generation Phase — How Shapes Change Token by Token

```
Prompt phase (prefill):
  input_ids       [2, 512]
  hidden_states   [2, 512, 4096]
  query_states    [2, 32, 512, 128]
  key cache       [2,  8, 512, 128]
  attn_weights    [2, 32, 512, 512]
  logits          [2,   1, 128256]   (only last token, logits_to_keep=1)

Generation step T (T = 513, 514, ...):
  input_ids       [2,   1]           (only the newly sampled token)
  hidden_states   [2,   1, 4096]
  query_states    [2,  32,  1, 128]  (S=1, attending to all past)
  key cache       [2,   8,  T, 128]  (grows every step)
  attn_weights    [2,  32,  1,  T]   (1 query vs T keys)
  logits          [2,   1, 128256]
```
