# Llama4 Attention — What's New vs Llama3

The core attention mechanism (GQA, KV cache, backend dispatch, `repeat_kv`) is the same as Llama3. See `../llama/Attention.md` for those. This file covers the four things that are structurally different in Llama4.

---

## 1. RoPE vs NoPE — Per-Layer Attention Type

Llama4 alternates between two types of attention layers, controlled by `config.no_rope_layers[layer_idx]`:

```python
self.use_rope = config.no_rope_layers[layer_idx]   # 1 = RoPE, 0 = NoPE
```

**RoPE layers** (`use_rope == 1`): standard causal attention with position encoding — same as Llama3.

**NoPE layers** (`use_rope == 0`): attention with **no positional encoding at all**. The Q and K vectors are never rotated. Instead, two mechanisms compensate:
1. **Chunked attention mask** — restricts which tokens a query can attend to (see below).
2. **Attention temperature tuning** — scales query magnitude based on position (see section 3).

The mask selection happens in `Llama4TextModel.forward`:
```python
causal_mask_mapping = {
    "full_attention":    create_causal_mask(**mask_kwargs),
    "chunked_attention": create_chunked_causal_mask(**mask_kwargs),
}
# Per layer, pick the mask type matching this layer's layer_type:
decoder_layer(
    attention_mask=causal_mask_mapping[self.config.layer_types[i]],
    ...
)
```

Both masks are pre-computed **once per forward pass** (not once per layer), then dispatched to each layer by index. `create_chunked_causal_mask` builds a block-diagonal mask where each block is `attention_chunk_size` tokens wide — tokens outside the local chunk are masked to `-inf`. This gives NoPE layers a local attention window without any position encoding, similar in spirit to sliding-window attention.

---

## 2. QK L2 Norm — Only on RoPE Layers

RoPE layers optionally apply an L2 normalisation to Q and K **after RoPE rotation**, using `Llama4TextL2Norm`:

```python
if self.config.use_qk_norm and self.use_rope:
    self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)
```

```python
class Llama4TextL2Norm(torch.nn.Module):
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)
```

This is mathematically identical to `LlamaRMSNorm` (same formula), but:
- **No learnable scale** — `Llama4TextL2Norm` has no `self.weight` parameter. It just normalises; the magnitude is fixed at 1.
- **Only on RoPE layers** — NoPE layers never apply this. The condition `self.use_rope` guards the `qk_norm` init too, so NoPE-layer attention objects don't even have the attribute.
- **Applied after RoPE** — Q/K are rotated first, then normalised:
  ```python
  query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)
  if hasattr(self, "qk_norm"):
      query_states = self.qk_norm(query_states)
      key_states   = self.qk_norm(key_states)
  ```

**Why only on RoPE layers?** RoPE rotation can change Q/K magnitudes unevenly across positions and frequency bands. L2 normalising after rotation keeps attention logits (`Q·Kᵀ / sqrt(d)`) in a stable range regardless of position. NoPE layers skip rotation entirely and use temperature tuning instead, so the magnitude problem doesn't arise in the same way.

The comment in the code — "the 128E model does not use qk_norm" — means when `use_qk_norm=False` in the config (which is the case for the 128-expert Maverick variant), this entire block is skipped.

---

## 3. Attention Temperature Tuning — Only on NoPE Layers

NoPE layers have no position encoding, so very long sequences can cause attention to degenerate — all token pairs look equally similar to the attention function. Temperature tuning addresses this by scaling query magnitude *up* as position grows, making the model more "selective" at long positions.

```python
if self.attn_temperature_tuning and not self.use_rope:   # only NoPE layers
    past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens

    attn_scales = (
        torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale)) * self.attn_scale + 1.0
    )
    # attn_scales shape: [S] → broadcast to [B, S, 1, 1] for the query tensor [B, S, heads, head_dim]
    attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand((*input_shape, 1, 1))
    query_states = (query_states * attn_scales).to(query_states.dtype)
```

**The formula unpacked:**

```
floor_scale = 8192  (default)
attn_scale  = 0.1   (default)

for position p:
  scale(p) = log1p(floor((p + 1) / 8192)) × 0.1 + 1.0
```

| Position `p` | `floor((p+1)/8192)` | `log1p(...)` | `scale(p)` |
|---|---|---|---|
| 0 – 8191 | 0 | 0 | 1.0 (no scaling) |
| 8192 – 16383 | 1 | 0.693 | 1.069 |
| 16384 – 24575 | 2 | 1.099 | 1.110 |
| 65535 | 7 | 2.079 | 1.208 |
| 131071 | 15 | 2.773 | 1.277 |

The scale stays at exactly 1.0 for the first 8192 tokens (no change), then grows logarithmically — slowly, so the model isn't destabilised, but enough to keep attention discriminative at very long context lengths. The log growth means doubling the context length adds a roughly constant increment to the scale rather than growing without bound.

**Interaction with KV cache:** `past_seen_tokens` is read from the cache, so during generation token 8200 correctly uses `scale ≈ 1.069` rather than computing its position relative to the current batch input alone.

---

## 4. Complex-Valued RoPE — `torch.polar` Instead of `rotate_half`

Llama4's text RoPE uses a different implementation from Llama3's `rotate_half` + `apply_rotary_pos_emb`.

**Llama3 approach** (real-valued, `rotate_half`):
```python
# cos/sin tables, shape [B, S, head_dim]
cos = emb.cos() * attention_scaling
sin = emb.sin() * attention_scaling

# Apply: rotate each (x, y) pair via cos/sin directly
q_embed = (q * cos) + (rotate_half(q) * sin)
```

**Llama4 approach** (complex-valued, `torch.polar`):
```python
# freqs_cis: complex tensor, shape [B, S, head_dim/2]
freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)   # e^(i·θ) for each freq/position
freqs_cis = freqs_cis * self.attention_scaling

# Apply: view q/k as complex, multiply by freqs_cis, view back as real
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**What `torch.polar(r, θ)` does:** creates a complex tensor `r * e^(iθ) = r * (cos θ + i·sin θ)`. With `r = ones_like(freqs)`, it's just `e^(iθ)` — a unit complex number encoding rotation by angle `θ`.

**What `torch.view_as_complex` does:** interprets each pair of adjacent real floats `(a, b)` as a complex number `a + bi`. So `xq.reshape(*shape, -1, 2)` reshapes `head_dim` → `head_dim/2` pairs, then `view_as_complex` makes each pair a complex scalar. Result shape: `[B, S, head_dim/2]` complex.

**The rotation:** complex multiplication `(a + bi) * (cosθ + i·sinθ) = (a·cosθ − b·sinθ) + i·(a·sinθ + b·cosθ)` — this is exactly the 2D rotation formula from Llama3, but expressed in one complex multiply instead of two real-valued operations (`q * cos` + `rotate_half(q) * sin`). The math is identical; the implementation is different.

**Key difference from Llama3's interleaving:** Llama3 uses adjacent-pairs layout (pairs are interleaved: dim 0 and dim `head_dim/2` form a pair). Llama4 uses `reshape(..., -1, 2)` before `view_as_complex`, so **pairs are consecutive**: dim 0 and dim 1 form a pair, dim 2 and dim 3 form the next pair, etc. These are mathematically equivalent but the physical memory ordering differs — which matters if you're comparing weight tensors between the two models directly.

**Also: no `emb = cat(freqs, freqs)` step.** Llama3 doubles `freqs` to full `head_dim` before taking `cos`/`sin`. Llama4 keeps `freqs_cis` at `head_dim/2` (complex) and uses complex multiplication directly — the duplication step is implicitly handled by the complex number representation.

---

## Summary: Per-Layer Attention Decision Tree

```
For layer i:

  use_rope = no_rope_layers[i]   (1 or 0)

  if use_rope:
    mask = full_attention (standard causal, full context)
    apply RoPE rotation to Q, K
    if use_qk_norm:
      L2-normalise Q and K after rotation
    run attention normally

  else:  # NoPE layer
    mask = chunked_attention (local window of attention_chunk_size tokens)
    skip RoPE rotation entirely
    if attn_temperature_tuning:
      scale Q by log-growing factor based on token position
    run attention normally
```
