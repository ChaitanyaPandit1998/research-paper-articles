# Llama3 vs Llama4 Text Attention — Side by Side

Covers `LlamaAttention` (Llama3, `modeling_llama.py`) vs `Llama4TextAttention` (Llama4, `modeling_llama4.py`).
Both are decoder-only causal attention. The shared foundation is identical; the differences are additive features layered on top in Llama4.

---

## Dimensions at a Glance

Using **Llama3 8B** and **Llama4 Scout 17B-16E** as concrete references.

| | Llama3 8B | Llama4 Scout 17B-16E |
|---|---|---|
| Hidden size | 4096 | 5120 |
| Q heads | 32 | 40 |
| KV heads | 8 | 8 |
| head_dim | 128 | 128 |
| KV groups | 4 | 5 |
| Vocab size | 128256 | 202048 |
| Num layers | 32 | 48 |
| Max context | 8192 | 131072 (128K) |

`head_dim` is 128 in both, so the attention scaling (`head_dim**-0.5 ≈ 0.0884`) is identical.

---

## Step 1 — QKV Projections

**Llama3:**
```python
q_proj: Linear(4096 → 32*128=4096, bias=attention_bias)   # default False
k_proj: Linear(4096 →  8*128=1024, bias=attention_bias)
v_proj: Linear(4096 →  8*128=1024, bias=attention_bias)
```

**Llama4:**
```python
q_proj: Linear(5120 → 40*128=5120, bias=attention_bias)   # default False
k_proj: Linear(5120 →  8*128=1024, bias=attention_bias)
v_proj: Linear(5120 →  8*128=1024, bias=attention_bias)
```

Structure identical — both use GQA with 8 KV heads, both default `bias=False`. The only difference is the larger hidden size and more Q heads in Llama4.

---

## Step 2 — Positional Encoding: RoPE Implementation

This is the deepest implementation difference. Both compute RoPE, but the math is expressed differently.

**Llama3 — real-valued, `rotate_half` trick:**
```python
# LlamaRotaryEmbedding.forward:
freqs = inv_freq_expanded @ position_ids_expanded   # [B, head_dim/2, S]
emb = torch.cat([freqs, freqs], dim=-1)             # duplicate to full head_dim
cos = emb.cos() * attention_scaling                 # [B, S, head_dim]
sin = emb.sin() * attention_scaling                 # [B, S, head_dim]

# apply_rotary_pos_emb:
cos = cos.unsqueeze(1)    # [B, 1, S, head_dim] — broadcast over heads
sin = sin.unsqueeze(1)
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]   # first half
    x2 = x[..., x.shape[-1] // 2 :]   # second half
    return torch.cat((-x2, x1), dim=-1)
```

**Llama4 — complex-valued, `torch.polar`:**
```python
# Llama4TextRotaryEmbedding.forward:
freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, S, head_dim/2]
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)               # complex: e^(i*theta)
freqs_cis = freqs_cis * self.attention_scaling

# apply_rotary_emb:
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
```

**The math is identical** — both implement `x_new = x·cosθ − y·sinθ, y_new = x·sinθ + y·cosθ` for each (x,y) pair. The difference is representation:
- Llama3 stays in real numbers, uses `rotate_half` to rearrange the vector so one elementwise multiply-add covers both halves simultaneously
- Llama4 encodes the same rotation as complex multiplication: `(x + iy) * e^(iθ) = (x·cosθ − y·sinθ) + i(x·sinθ + y·cosθ)`

**Memory layout also differs:**
- Llama3: split-half layout — all `x` values first, all `y` values second: `[x0,x1,x2,x3, y0,y1,y2,y3]`
- Llama4: consecutive-pair layout — `reshape(...,-1,2)` before `view_as_complex` means pairs are adjacent: `[(x0,y0), (x1,y1), (x2,y2), (x3,y3)]`

Both are equivalent for attention (dot products are invariant to this layout), but weight tensors are not directly interchangeable between the two implementations.

**`freqs_cis` shape:**
- Llama3: `cos`/`sin` each `[B, S, head_dim]` — real tensors, already doubled via `cat(freqs, freqs)`
- Llama4: `freqs_cis` is `[B, S, head_dim/2]` complex — no doubling needed, complex multiply handles both halves implicitly

---

## Step 3 — RoPE vs NoPE per Layer (Llama4 only)

**Llama3:** every layer uses RoPE, no exceptions.

**Llama4:** layers alternate between RoPE and NoPE based on `config.no_rope_layers[layer_idx]`:

```python
self.use_rope = config.no_rope_layers[layer_idx]   # 1=RoPE, 0=NoPE

# In forward:
if self.use_rope:
    query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)
# else: skip entirely — no rotation applied
```

With default `no_rope_layer_interval=4`, every 4th layer is a NoPE layer (e.g. layers 3, 7, 11, ... are NoPE). NoPE layers skip the RoPE rotation completely and rely on two compensating mechanisms instead: chunked causal masking (Step 5) and attention temperature tuning (Step 6).

---

## Step 4 — QK L2 Norm (Llama4 only)

**Llama3:** none.

**Llama4:** on RoPE layers, optionally applies `Llama4TextL2Norm` to Q and K after rotation:

```python
if self.config.use_qk_norm and self.use_rope:
    self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

# In forward, after RoPE:
if hasattr(self, "qk_norm"):
    query_states = self.qk_norm(query_states)
    key_states   = self.qk_norm(key_states)
```

`Llama4TextL2Norm` is identical in formula to `LlamaRMSNorm` but has **no learnable weight** — it only normalises to unit RMS, it doesn't rescale. This prevents RoPE rotation from pushing Q/K magnitudes unevenly across positions, which would destabilise attention logits at long context lengths.

`use_qk_norm=True` is the default for Scout 17B-16E; `use_qk_norm=False` for the 128-expert Maverick variant.

---

## Step 5 — Causal Mask

**Llama3:** one mask, computed once, used by all layers:
```python
causal_mask = create_causal_mask(
    config, inputs_embeds, attention_mask, past_key_values, position_ids
)
# passed to every decoder layer unchanged
```

**Llama4:** two masks computed once, dispatched per layer type:
```python
causal_mask_mapping = {
    "full_attention":    create_causal_mask(**mask_kwargs),
    "chunked_attention": create_chunked_causal_mask(**mask_kwargs),
}

# Each layer picks its mask:
decoder_layer(
    attention_mask=causal_mask_mapping[self.config.layer_types[i]],
    ...
)
```

`create_chunked_causal_mask` builds a block-diagonal mask where each block covers `attention_chunk_size=8192` tokens. Within a block, attention is standard causal; tokens in different blocks cannot attend to each other at all. This gives NoPE layers a bounded local window without position encoding — a NoPE layer at token 50000 doesn't see token 1.

RoPE layers always get `full_attention` (standard causal, unlimited lookback). NoPE layers always get `chunked_attention`. The dispatch is by `config.layer_types[i]`, a list of `"full_attention"` / `"chunked_attention"` strings derived from `no_rope_layers` at config init.

---

## Step 6 — Attention Temperature Tuning (Llama4 NoPE layers only)

**Llama3:** none.

**Llama4:** on NoPE layers, query states are scaled by a position-dependent factor before the attention dot product:

```python
if self.attn_temperature_tuning and not self.use_rope:
    past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens

    attn_scales = (
        torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale)) * self.attn_scale + 1.0
    )
    query_states = (query_states * attn_scales.view(1, -1, 1, 1)).to(query_states.dtype)
```

Formula: `scale(p) = log1p(floor((p+1) / 8192)) × 0.1 + 1.0`

The scale is 1.0 (no change) for the first 8192 tokens, then grows logarithmically. This keeps NoPE attention selective at long positions where, without position encoding, all token pairs look increasingly similar. The `past_seen_tokens` offset ensures generation step N correctly uses position N's scale, not position 0.

---

## Step 7 — Attention Score Computation

**Llama3:**
```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# Upcast to float32 for softmax — avoids bfloat16 overflow on large logits
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
attn_output  = torch.matmul(attn_weights, value_states)
```

**Llama4:**
```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# NO float32 upcast — runs softmax in whatever dtype attn_weights is (bfloat16)
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
attn_output  = torch.matmul(attn_weights, value_states)
```

The only difference is the float32 upcast in softmax. Llama3 upcasts because bfloat16 can overflow on `e^x` when attention scores are large (bfloat16 max ≈ 38912, `e^11` already overflows). Llama4 drops the upcast — a deliberate choice noted in the code comment "llama4 doesn't cast attn weights to fp32." This saves memory and speeds up the softmax step at the cost of a small risk of overflow on extreme logit values.

---

## Step 8 — Output Projection + Residual

Structurally identical in both:
```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
hidden_states = residual + attn_output
```

Dimensions differ (`hidden_size` 4096 vs 5120), everything else the same.

---

## Step 9 — KV Cache

Both use `DynamicCache` with the same `past_key_values.update(key_states, value_states, layer_idx)` call. No difference in caching mechanism, only in what's cached (K/V are `[B, 8, T, 128]` in both, growing by 1 per generation step).

---

## Step 10 — Feed-Forward Network

**Llama3 (always dense SwiGLU):**
```python
# LlamaMLP — every layer, same structure
gate = silu(gate_proj(x))    # [B, S, 4096] → [B, S, 11008]
out  = gate * up_proj(x)
return down_proj(out)        # [B, S, 11008] → [B, S, 4096]
```

**Llama4 (dense or MoE, per layer):**
```python
# Dense layers — Llama4TextMLP
gate = silu(gate_proj(x))    # [B, S, 5120] → [B, S, 16384]
out  = gate * up_proj(x)
return down_proj(out)        # [B, S, 16384] → [B, S, 5120]

# MoE layers — Llama4TextMoe
# router → 16 experts (batched bmm) + shared_expert → sum
# see MoE.md for full detail
```

This is the second biggest structural change after RoPE/NoPE. Llama3 has one FFN type; Llama4 has two, selected per layer at construction. With default `interleave_moe_layer_step=1`, every layer in Llama4 is a MoE layer.

---

## Step 11 — Normalisation

Both use the same RMSNorm formula. Different class names, identical implementation:

```python
# Llama3: LlamaRMSNorm
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)

# Llama4: Llama4TextRMSNorm (same math, different name)
output = self._norm(x.float()).type_as(x)
return output * self.weight
```

The `@use_kernel_forward_from_hub("RMSNorm")` decorator on `LlamaRMSNorm` allows the kernel to be swapped for a fused CUDA/Triton implementation. `Llama4TextRMSNorm` lacks this decorator — it always runs the plain PyTorch path.

---

## Step 12 — Flash Attention Support

**Llama3:** `_supports_flash_attn = True` — FlashAttention-2 can be used as the attention backend.

**Llama4:** `_supports_flash_attn = False` — explicitly disabled. SDPA and FlexAttention are supported; FlashAttention-2 is not. The reasons aren't documented inline, but likely relate to the non-standard attention patterns (chunked masks, temperature-scaled Q) in NoPE layers that FlashAttention-2 kernels don't natively support.

---

## What Is Identical Between the Two

- GQA with 8 KV heads and `repeat_kv`
- `ALL_ATTENTION_FUNCTIONS` backend dispatch pattern
- `attention_bias=False` default on all projections
- SwiGLU activation in dense MLP layers
- `DynamicCache` KV caching mechanism
- Residual connection pattern around attention and FFN
- `logits_to_keep` slicing in the LM head for inference efficiency
- `GenerationMixin` for `.generate()` — greedy, sampling, beam search
- `create_causal_mask` from `masking_utils.py` (Llama4 also adds `create_chunked_causal_mask`)

---

## Full Comparison Table

| | Llama3 | Llama4 |
|---|---|---|
| RoPE implementation | real `rotate_half` + `cat(freqs, freqs)` | complex `torch.polar` + `view_as_complex` |
| Memory layout | split-half (`[x0..xN, y0..yN]`) | consecutive pairs (`[(x0,y0)..]`) |
| `position_embeddings` type passed to layers | `(cos, sin)` tuple | `freqs_cis` complex tensor |
| NoPE layers | No — all layers use RoPE | Yes — every `no_rope_layer_interval`-th layer |
| Attention mask | one mask, all layers | two masks dispatched by `layer_types[i]` |
| QK L2 norm | None | Yes, on RoPE layers (no learnable weight) |
| Temperature tuning | None | Yes, on NoPE layers (log-scale with position) |
| Softmax fp32 upcast | Yes (`dtype=torch.float32`) | **No** |
| Flash attention | Supported | **Disabled** |
| FFN | Always dense SwiGLU | Dense or MoE per layer |
| RMSNorm kernel hook | `@use_kernel_forward_from_hub` | Not decorated |
| Config structure | Single `LlamaConfig` | `Llama4Config` → `Llama4TextConfig` + `Llama4VisionConfig` |
| Multimodal | No | Yes (`Llama4ForConditionalGeneration`) |
