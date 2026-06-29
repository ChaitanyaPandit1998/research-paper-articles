# Text Decoder Attention vs Vision Encoder Attention — Side by Side

Covers `Llama4TextAttention` (text decoder) vs `Llama4VisionAttention` (vision encoder).
Both live in `modeling_llama4.py`. The foundational attention mechanics are shared; the differences reflect the different demands of language modelling vs image patch encoding.

---

## Dimensions at a Glance

| | Text (`Llama4TextAttention`) | Vision (`Llama4VisionAttention`) |
|---|---|---|
| Hidden size | 5120 | 768 |
| Q heads | 40 | 16 |
| KV heads | 8 | 16 |
| head_dim | 128 | 48 (768/16) |
| KV groups | 5 (GQA) | 1 (MHA — no grouping) |

---

## Step 1 — QKV Projections

**Text:**
```python
q_proj: Linear(5120 → 40*128=5120, bias=False)
k_proj: Linear(5120 →  8*128=1024, bias=False)
v_proj: Linear(5120 →  8*128=1024, bias=False)
```
K and V are smaller than Q because of GQA — only 8 KV heads vs 40 Q heads.

**Vision:**
```python
q_proj: Linear(768 → 16*48=768, bias=True)
k_proj: Linear(768 → 16*48=768, bias=True)
v_proj: Linear(768 → 16*48=768, bias=True)
```
All three are the same size — full MHA, every head has its own K/V. Biases are on, which they never are in the text decoder.

**Similarity:** both split the output into `[B, seq, heads, head_dim]` then transpose to `[B, heads, seq, head_dim]` before attention.

---

## Step 2 — Positional Encoding (RoPE)

This is the biggest structural difference.

**Text (RoPE layers):**
- 1D RoPE — position is a single integer per token
- Complex-valued: `freqs_cis = torch.polar(ones, freqs)` → unit complex number per frequency
- Applied via `torch.view_as_complex` multiply, equivalent to rotating each (x,y) pair by its angle
- Computed once per forward pass and shared across all 48 layers
- NoPE layers skip this entirely and use temperature tuning instead

**Vision:**
- 2D RoPE — every patch has an (x, y) grid coordinate, not a 1D position
- X and Y frequencies are computed separately then concatenated
- Pre-computed once at `__init__` as a persistent buffer, never recomputed per call
- CLS token gets zero frequencies → identity rotation → no positional encoding for CLS
- Frequency table shape is `[num_patches, num_patches, head_dim/2]` — encodes cross-patch relationships, not just per-token position

```python
# Text: 1D, complex multiply
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)         # e^(i*theta)
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)

# Vision: 2D, same complex multiply but frequencies encode (row, col) separately
freqs_x = ((col_index + 1) * rope_freq).repeat_interleave(2)
freqs_y = ((row_index + 1) * rope_freq).repeat_interleave(2)
freqs   = cat([freqs_x, freqs_y])   # both spatial dimensions baked into one freq tensor
freq_cis = view_as_complex(stack([cos(freqs), sin(freqs)]))
```

---

## Step 3 — QK Norm

**Text (RoPE layers only, when `use_qk_norm=True`):**
```python
query_states = self.qk_norm(query_states)   # Llama4TextL2Norm — no learnable weight
key_states   = self.qk_norm(key_states)
```
Applied after RoPE rotation to keep attention logits in a stable range. Not present on NoPE layers.

**Vision:** no QK norm at all.

---

## Step 4 — Attention Temperature Tuning

**Text (NoPE layers only):**
```python
attn_scales = log1p(floor((positions + 1) / floor_scale)) * attn_scale + 1.0
query_states = query_states * attn_scales
```
Scales query magnitude logarithmically with token position to keep attention discriminative at long context without position encoding.

| Position | scale |
|---|---|
| 0 – 8191 | 1.000 (no change) |
| 8192 – 16383 | 1.069 |
| 65535 | 1.208 |
| 131071 | 1.277 |

**Vision:** no temperature tuning. Vision attention always runs over a fixed 32×32 patch grid (1024+1 tokens), so long-context degradation isn't a concern.

---

## Step 5 — Causal Mask

**Text:**
```python
# RoPE layers: standard causal (upper triangular -inf)
create_causal_mask(...)           # [B, 1, S, S] — each token sees only past tokens

# NoPE layers: chunked causal
create_chunked_causal_mask(...)   # [B, 1, S, S] — block-diagonal within attention_chunk_size window
```
Every text attention layer is causal — tokens cannot attend to future tokens.

**Vision:**
```python
attention_mask = None   # no mask passed to attention
is_causal = False       # explicitly enforced in the attention_interface call
```
Vision attention is **fully bidirectional** — every patch sees every other patch. This makes sense: patch 5 knowing about patch 800 is fine for understanding an image, unlike language where future tokens must be hidden.

---

## Step 6 — GQA Expansion (`repeat_kv`)

**Text:**
```python
n_kv_groups = 40 / 8 = 5
repeat_kv(key_states, n_rep=5)    # [B, 8, S, 128] → [B, 40, S, 128]
```
5 query heads share each KV head. Saves KV-cache memory and KV projection compute.

**Vision:**
```python
self.num_key_value_groups = 1
repeat_kv(key_states, n_rep=1)    # no-op — returns unchanged
```
`n_rep=1` hits the early-return branch in `repeat_kv` — no memory expansion happens. Vision uses full MHA. The call is still there for interface consistency with the text path.

---

## Step 7 — Attention Score Computation

**Text (`eager_attention_forward`):**
```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
# scaling = head_dim**-0.5 = 128**-0.5 ≈ 0.0884

if attention_mask is not None:
    attn_weights = attn_weights + attention_mask   # add causal mask (-inf at future positions)

attn_weights = nn.functional.softmax(attn_weights, dim=-1)   # no fp32 upcast (Llama4 change)
```

**Vision (`vision_eager_attention_forward`):**
```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * module.head_dim**-0.5
# = 48**-0.5 ≈ 0.144 — larger scaling because head_dim is smaller

# attention_mask is None — mask addition step is skipped entirely

attn_weights = nn.functional.softmax(attn_weights, dim=-1)   # also no fp32 upcast
```

Two differences here:
- The mask addition is skipped in vision (bidirectional — nothing to mask)
- The scaling constant differs because `head_dim` differs (128 vs 48)

Note: Llama3's text attention **does** upcast to float32 for softmax (`dtype=torch.float32`) to avoid bfloat16 overflow on large logits. Both Llama4 text and Llama4 vision skip this upcast — an intentional Llama4 change, noted in the code as "llama4 doesn't cast attn weights to fp32."

---

## Step 8 — Output Projection + Residual

Both are identical in structure:
```python
attn_output = attn_output.transpose(1, 2).contiguous()   # [B, heads, S, head_dim] → [B, S, heads, head_dim]
attn_output = attn_output.reshape(*input_shape, -1)       # merge heads → [B, S, hidden_size]
attn_output = self.o_proj(attn_output)                    # project back to hidden_size
hidden_states = residual + attn_output                    # residual connection
```

**Text `o_proj`:** `Linear(5120 → 5120, bias=False)`
**Vision `o_proj`:** `Linear(768 → 768, bias=True)` — same bias pattern as the other vision projections.

---

## Step 9 — KV Cache

**Text:**
```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```
Essential for autoregressive generation — stores all K/V vectors seen so far, appends the new token's K/V, and returns the full history so the new token can attend to all previous context. Without it, generating token N+1 would require re-running all N past tokens through every layer.

**Vision:** no KV cache. Images are always processed as a complete 2D grid in a single forward pass. There is no token-by-token generation loop over patches.

---

## Step 10 — The FFN After Attention

**Text (SwiGLU):**
```python
gate = silu(gate_proj(x))    # SiLU activation on gate branch
out  = gate * up_proj(x)     # elementwise multiply — gated
return down_proj(out)
```

**Vision (plain GELU MLP):**
```python
x = fc1(x)      # Linear, bias=True
x = gelu(x)     # GELU activation — no gate
x = fc2(x)      # Linear, bias=True
```

No gating in vision. This reflects the broader architectural split between ViTs (plain GELU MLPs) and decoder LLMs (SwiGLU/GeGLU gated variants). Both achieve similar capacity, but the gated variant gives the model a learned per-element switch on information flow.

---

## Step 11 — Normalisation

**Text:** `Llama4TextRMSNorm`
- No learnable bias, no mean centering
- Normalises by `1/RMS(x)` then multiplies by a learned per-dimension weight
- Upcast to float32 for computation, cast back after

**Vision:** `nn.LayerNorm`
- Subtracts mean, divides by standard deviation
- Has learnable γ (scale) **and** β (bias)
- Standard for ViT-family architectures — richer normalisation with centering

---

## Full Comparison Table

| | Text Attention | Vision Attention |
|---|---|---|
| Architecture | GQA (40Q / 8KV) | MHA (16Q / 16KV) |
| head_dim | 128 | 48 |
| Projection biases | No (`bias=False`) | Yes (`bias=True`) |
| Positional encoding | 1D RoPE (complex) or NoPE | 2D RoPE (x+y, pre-computed buffer) |
| QK norm | L2 norm after RoPE (when enabled) | None |
| Temperature tuning | Yes (NoPE layers only) | None |
| Attention mask | Causal or chunked causal | None — fully bidirectional |
| `is_causal` | True | **False** |
| KV cache | Yes (inference) | No |
| Softmax fp32 upcast | No (Llama4) / Yes (Llama3) | No |
| FFN type | SwiGLU (gated) | GELU (no gate) |
| Normalisation | RMSNorm (no bias) | LayerNorm (with bias) |
| MoE FFN | Yes (most layers) | No — always dense |
| When RoPE computed | Once per forward pass | Once at `__init__` |

---

## What Is Shared

- `repeat_kv` — called in both paths (no-op in vision since `n_rep=1`, but same function)
- `ALL_ATTENTION_FUNCTIONS` backend dispatch — both use the same registry to pick eager / SDPA / FlexAttention
- Core score computation — `Q @ K.T * scaling → softmax → @ V`
- `transpose + reshape` to merge heads back into hidden_size
- Residual connection pattern — `residual = x; x = sublayer(norm(x)); x = residual + x`
- `GradientCheckpointingLayer` base class on the encoder/decoder layer wrappers
- `apply_rotary_emb` function signature and complex-multiply logic — text and vision both use `view_as_complex` + complex multiply + `view_as_real`, the 2D vs 1D difference is only in how `freqs_cis` is constructed beforehand
