# `modeling_llama.py` — Class-by-Class Notes

Reminder of the big picture (see `NOTES.md`): LLaMA has no modular file — this *is* the original, self-contained implementation that Gemma and friends modularize against. Below is what each piece does, in plain terms.

---

## `LlamaRMSNorm`

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

**What it does:** RMSNorm — rescales a vector by its root-mean-square magnitude (`x / sqrt(mean(x²) + eps)`), no mean-subtraction (unlike LayerNorm, which is why it's cheaper). The whole computation is done in `float32` (upcast at the start) for numerical stability, then the result is multiplied by the learnable `weight` and **downcast back to the input dtype afterward**.

**Init:** `weight` starts at all-ones, so at initialization the norm is `1 * normalized_x` — a clean identity scale. (Contrast with Gemma's `(1 + weight)` parameterization, where `weight` starts at zero — see `../gemma/modular_gemma_notes.MD`.)

**`@use_kernel_forward_from_hub("RMSNorm")` decorator:** lets this class transparently swap in a faster, hardware-optimized RMSNorm kernel from the Hub (e.g. a fused CUDA kernel) when available, falling back to this plain PyTorch implementation otherwise.

---

## `LlamaRotaryEmbedding`

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        ...
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn = self.compute_default_rope_parameters if self.rope_type == "default" else ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
```

**What it does:** precomputes the RoPE inverse-frequency vector once at construction time, then on every `forward(x, position_ids)` call turns position indices into `cos`/`sin` tables that encode position.

**`compute_default_rope_parameters`:** the standard RoPE formula —
```python
inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
```
`base` is `rope_theta` (commonly `10000.0`); larger `base` means frequencies decay more slowly across dimensions, which lets the model represent longer-range positions without the angle wrapping around too fast. `attention_factor` is `1.0` here (unused for plain RoPE) — it exists as a hook for scaled RoPE variants (linear scaling, YaRN, etc.) that *do* need to rescale attention magnitudes when extending context length beyond what the model was trained on.

**`original_inv_freq` buffer:** kept as an unmodified copy of the initial frequencies. This matters for **dynamic** RoPE scaling types — algorithms that need to recompute scaled frequencies on the fly as sequence length grows past the training length (see `dynamic_rope_update` decorator on `forward`) need the *original* values to scale from, not whatever `inv_freq` was last mutated to.

**`forward`:**
```python
freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos, sin = emb.cos() * attention_scaling, emb.sin() * attention_scaling
```
Computed in forced `float32` (`maybe_autocast(..., enabled=False)`) regardless of the model's working dtype, then cast back down — RoPE angles are precision-sensitive (small errors compound over long sequences), so this avoids doing the trig in lower precision.

---

## `rotate_half` / `apply_rotary_pos_emb`

```python
def rotate_half(x):
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**What it does:** the actual rotation. Treats each pair of dimensions in `q`/`k` as a 2D vector and rotates it by an angle proportional to its position — `rotate_half` implements the "swap-and-negate" trick that turns a plain elementwise multiply-and-add into a true rotation (mathematically equivalent to multiplying by a block-diagonal rotation matrix, without materializing one).

**Why this encodes *relative* position:** after rotating both `q` and `k` by their absolute positions, the dot product `q · k` in attention ends up depending only on the *difference* between their positions — not their absolute values. This is the core trick that lets RoPE generalize to sequence lengths longer than what cos/sin tables were literally built for in many cases.

**`@use_kernel_func_from_hub("rotary_pos_emb")`:** same Hub-kernel-swap pattern as RMSNorm — lets a fused kernel replace this when available.

---

## `LlamaMLP`

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**What it does:** SwiGLU-style gated feed-forward block. `gate_proj(x)` passes through `act_fn` (SiLU by default) to produce a *gate* signal; `up_proj(x)` produces the actual content; they're multiplied elementwise (the "gating"), then `down_proj` projects back down to `hidden_size`. This gated structure (vs. a plain two-layer MLP) is what most modern LLMs use instead of GPT-2-style `Linear → GELU → Linear`.

---

## `repeat_kv`

```python
def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

**What it does:** for Grouped-Query Attention, duplicates each K/V head `n_rep` times so the head count matches the (larger) number of query heads, without actually copying memory until the final `reshape` (the `expand` step is a zero-copy broadcast view). See `NOTES.md` for the full GQA explanation. No-ops when `n_rep == 1` (no GQA).

---

## `eager_attention_forward`

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights
```

**What it does:** the textbook scaled dot-product attention, written in plain PyTorch (no fused kernel). `repeat_kv` happens first to expand K/V to match Q's head count (GQA). Softmax is forced to `float32` for stability, then cast back. This is the fallback path — `ALL_ATTENTION_FUNCTIONS.get_interface(...)` will swap in a faster backend (SDPA, FlashAttention, FlexAttention) when `config._attn_implementation` requests one; this function exists as the simple, always-correct reference implementation.

---

## `LlamaAttention`

```python
@use_kernelized_func(apply_rotary_pos_emb)
class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.head_dim = ...
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=config.attention_bias)
```

**Projection shapes are the GQA tell:** `q_proj` outputs `num_attention_heads * head_dim`, but `k_proj`/`v_proj` output the smaller `num_key_value_heads * head_dim` — this is *where* the GQA memory savings actually live (smaller K/V weight matrices, smaller KV-cache entries), not just in `repeat_kv`'s runtime broadcast.

**`forward` walkthrough:**
1. Project `hidden_states` through `q/k/v_proj`, reshape each into `(batch, heads, seq, head_dim)`.
2. Apply RoPE to `q`/`k` via `apply_rotary_pos_emb` (using the precomputed `cos`/`sin` passed in as `position_embeddings`) — **not** to `v`, since RoPE only needs to affect the similarity score between queries and keys, not the actual content being retrieved.
3. If a KV cache exists, append the new `key_states`/`value_states` to it (`past_key_values.update(...)`) — this is what makes incremental decoding fast (don't recompute K/V for already-seen tokens).
4. Dispatch to whichever attention backend is configured (`eager`, SDPA, FlashAttention, ...) via `ALL_ATTENTION_FUNCTIONS.get_interface`.
5. Reshape the multi-head output back to `(batch, seq, hidden)` and project through `o_proj`.

**`is_causal = True`:** unlike Gemma, LLaMA has no bidirectional-attention switch — it's always autoregressive (causal masking only).

---

## `LlamaDecoderLayer`

```python
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), ...)[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states
```

**What it does:** one transformer block, **pre-norm** style (normalize *before* the sublayer, not after) — `input_layernorm` runs before attention, `post_attention_layernorm` runs before the MLP (its name describes its *position* — "after attention, before MLP" — not that it normalizes attention's output in-place). Two residual connections, one around each sublayer. This pre-norm + residual pattern is what makes very deep transformers trainable without vanishing/exploding gradients.

**`GradientCheckpointingLayer` base class:** lets this layer's forward pass be wrapped in PyTorch's gradient checkpointing automatically (recompute activations during backward instead of storing them) when `gradient_checkpointing=True` — trades compute for memory during training.

---

## `LlamaPreTrainedModel`

```python
class LlamaPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _can_record_outputs = {"hidden_states": LlamaDecoderLayer, "attentions": LlamaAttention}
```

**What it does:** the shared base for every LLaMA task head — declares capability flags the framework checks (which attention backends it supports, whether it's safe to `torch.compile` with `fullgraph=True`, that `LlamaDecoderLayer` is the atomic unit that shouldn't be split across devices during sharding) and how to capture intermediate outputs (`hidden_states` per layer, `attentions` per attention module) when requested.

---

## `LlamaModel`

```python
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(num_hidden_layers)])
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
```

**`forward` walkthrough:**
1. **Input validation** — exactly one of `input_ids`/`inputs_embeds` must be given.
2. **Embed tokens** — plain `nn.Embedding` lookup (no scaling, unlike Gemma's `√hidden_size` multiply).
3. **Set up the KV cache** — fresh `DynamicCache` if `use_cache=True` and none was passed.
4. **Compute position IDs** — simple increasing range, offset by however many tokens are already cached.
5. **Build the causal attention mask** — `create_causal_mask` constructs the mask that blocks attending to future tokens.
6. **Compute RoPE cos/sin once** — `self.rotary_emb(...)`, reused by every decoder layer below (not recomputed per-layer).
7. **Run through every decoder layer** — threading `hidden_states` through each, along with the mask, position ids, cache, and the shared RoPE tables.
8. **Final norm** — `self.norm` (RMSNorm) on the last layer's output.
9. **Return** — `BaseModelOutputWithPast` with the final hidden states and updated cache.

This is the forward pass Gemma's `GemmaModel.forward` copies almost verbatim (see `../gemma/modular_gemma_notes.MD`) — the only differences there are the scaled embedding lookup and the (optionally) non-causal mask.

---

## `LlamaForCausalLM`

```python
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

**What it does:** wraps `LlamaModel` with a language-modeling head (`lm_head`) that projects final hidden states to vocabulary logits.

**`_tied_weights_keys`:** declares that `lm_head.weight` *can* be tied to `model.embed_tokens.weight` (sharing the same matrix for input embedding and output projection) — but recall `tie_word_embeddings` defaults to `False` for LLaMA, so by default they're **separate** matrices, unlike Gemma which ties them.

**`forward`:**
1. Run `self.model(...)` to get hidden states.
2. **`logits_to_keep`** — slices hidden states to only the last `logits_to_keep` positions before the `lm_head` projection (or all of them if `0`/unspecified). During generation, you usually only need logits for the *next* token, so this avoids computing a `[batch, full_seq, vocab_size]` tensor when only `[batch, 1, vocab_size]` is needed.
3. If `labels` are provided, compute the loss via `self.loss_function` (standard cross-entropy for causal LM).
4. Return `CausalLMOutputWithPast`.

---

## `LlamaForSequenceClassification` / `LlamaForQuestionAnswering` / `LlamaForTokenClassification`

```python
class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...
class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`
class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...
```

**What they do:** zero custom logic — each just combines a generic task-head mixin (`GenericForSequenceClassification`, etc., shared across many model families) with `LlamaPreTrainedModel` to get a classification/QA/token-tagging head on top of LLaMA's backbone. These are the classes Gemma's equivalents (`GemmaForSequenceClassification(LlamaForSequenceClassification): pass`) inherit from directly.

**`base_model_prefix = "transformer"` on `LlamaForQuestionAnswering`:** a backward-compatibility quirk — this particular head used `self.transformer` as the attribute name for the backbone model historically (before the codebase standardized on `self.model` elsewhere), so it overrides the prefix used when loading/saving checkpoints to stay compatible with old saved weights.

---

## `__all__`

```python
__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
```

Public API surface for lazy-loading (see `../gemma/INIT_PY_EXPLAINED.md`). Note `LlamaRMSNorm`, `LlamaAttention`, `LlamaMLP`, `LlamaDecoderLayer`, `LlamaRotaryEmbedding` are internal building blocks, not exported — but they're exactly the classes Gemma and other LLaMA-family models import directly (`from .modeling_llama import LlamaAttention` etc. inside their own `modular_*.py` files) even though they're not part of the public `transformers` top-level API.

---

## Summary Table — What's Distinctive About LLaMA's Implementation

| Class | Key design choice |
|---|---|
| `LlamaRMSNorm` | Plain `weight` (not `1+weight`); init to ones; multiply happens after downcasting to input dtype |
| `LlamaRotaryEmbedding` | Keeps `original_inv_freq` around to support dynamic RoPE scaling re-derivation |
| `apply_rotary_pos_emb` | `rotate_half` trick — encodes *relative* position via rotation, not additive embeddings |
| `LlamaMLP` | SwiGLU gated FFN: `down(act(gate(x)) * up(x))` |
| `repeat_kv` / GQA | `num_key_value_heads` can be `< num_attention_heads`; K/V projections themselves are smaller, not just broadcast at attention time |
| `LlamaAttention` | Always causal (`is_causal = True`, no bidirectional switch) |
| `LlamaDecoderLayer` | Pre-norm transformer block, two residual connections |
| `LlamaModel` | Plain (unscaled) embedding lookup; otherwise identical control flow to what Gemma copies |
| `LlamaForCausalLM` | `tie_word_embeddings=False` by default — separate input/output embedding matrices |

This is the architecture every LLaMA-family modular file (Gemma, and many others in `transformers`) starts from and selectively overrides — see `../gemma/modular_gemma_notes.MD`'s summary table for exactly what Gemma changes relative to everything documented here.
