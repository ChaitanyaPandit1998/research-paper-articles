# LLaMA Model — Notes

## Files in `src/transformers/models/llama/`

### `modeling_llama.py` — The actual model code (no modular file)
Unlike Gemma, LLaMA has no `modular_llama.py` — it's the **base architecture** that other models (Gemma, Mistral, Qwen, etc.) inherit from and override. Everything here is written out directly:
- **`LlamaRMSNorm`** — Root-Mean-Square normalization, no mean-subtraction (unlike LayerNorm)
- **`LlamaRotaryEmbedding`** — Computes RoPE (Rotary Position Embeddings) cos/sin tables
- **`LlamaMLP`** — SwiGLU-style gated feed-forward block
- **`LlamaAttention`** — Multi-head attention with RoPE + grouped-query attention (GQA) support
- **`LlamaDecoderLayer`** — One transformer block: attention + MLP with residual connections
- **`LlamaModel`, `LlamaForCausalLM`, etc.** — Full model stack and task heads

### `configuration_llama.py` — `LlamaConfig`
The settings/hyperparameters dataclass for the model. Also defines the default tensor-parallel (`base_model_tp_plan`) and pipeline-parallel (`base_model_pp_plan`) sharding plans, inherited by Gemma and most other LLaMA-family models unchanged.

### `tokenization_llama.py` — `LlamaTokenizer`
BPE (Byte-Pair Encoding) tokenizer built on the `tokenizers` library (`BPE` model class):
- Spaces replaced by `▁` (Metaspace pre-tokenizer)
- Byte fallback: unknown characters encoded as raw bytes instead of `<unk>`
- Pads on the left side (standard for decoder-only generation)
- `LlamaTokenizerFast = LlamaTokenizer` — kept as a backward-compatible alias; there's only one tokenizer class now, not separate slow/fast implementations

### `convert_llama_weights_to_hf.py` — One-time migration script
Converts Meta's original LLaMA checkpoint format into HuggingFace format. Only used when importing a new LLaMA release from Meta.

### `__init__.py` — Package entry point
Same lazy-loading pattern as every other model — see [[INIT_PY_EXPLAINED]] in `../gemma/` (the mechanism is generic, not LLaMA-specific: `_LazyModule` + `define_import_structure` defer importing `modeling_llama.py`/`configuration_llama.py`/`tokenization_llama.py` until a name is actually accessed).

> **Key difference from Gemma:** there's no "source of truth" modular file to edit here, because LLaMA *is* the source of truth other models modularize against. Edit `modeling_llama.py` / `configuration_llama.py` directly.

---

## `LlamaConfig` — Fields Explained

### Architecture size

| Field | Default (7B) | Meaning |
|---|---|---|
| `vocab_size` | 32,000 | Number of tokens the model knows |
| `hidden_size` | 4096 | Width of the internal representation vectors |
| `intermediate_size` | 11,008 | Width of the feed-forward (MLP) layers |
| `num_hidden_layers` | 32 | Number of transformer blocks stacked on top of each other |
| `num_attention_heads` | 32 | Number of query attention heads per layer |
| `num_key_value_heads` | `None` → defaults to `num_attention_heads` | For GQA — fewer KV heads than query heads means multiple query heads share the same K/V (saves memory/compute); `None` means no GQA, every head has its own K/V |
| `head_dim` | `None` → defaults to `hidden_size / num_attention_heads` | Size of each attention head's vector |

### Behavior flags

| Field | Default | Meaning |
|---|---|---|
| `hidden_act` | `"silu"` | Activation function used in MLP layers (SiLU/Swish, paired with the gate for SwiGLU) |
| `max_position_embeddings` | 2048 | Maximum sequence length (tokens) the model can handle |
| `rms_norm_eps` | 1e-6 | Small constant to prevent division by zero in normalization |
| `use_cache` | `True` | Whether to cache key/value states during generation (speeds up inference) |
| `attention_bias` | `False` | Whether Q/K/V/O projection layers have a bias term |
| `attention_dropout` | 0.0 | Dropout probability in attention (0 = disabled) |
| `mlp_bias` | `False` | Whether MLP projection layers have a bias term |
| `tie_word_embeddings` | `False` | Whether input and output token embeddings share the same weights — LLaMA does **not** tie them (Gemma does) |
| `pretraining_tp` | 1 | Legacy field recording how many shards the model was originally pretrained with (for an older, now mostly unused, manual tensor-parallel rescaling path) |

### Special tokens

| Field | Default | Meaning |
|---|---|---|
| `pad_token_id` | `None` | Token ID used to pad shorter sequences in a batch |
| `bos_token_id` | 1 | "Beginning of sequence" token |
| `eos_token_id` | 2 | "End of sequence" token — model stops generating here |

### Training

| Field | Default | Meaning |
|---|---|---|
| `initializer_range` | 0.02 | Standard deviation for weight initialization |
| `rope_parameters` | `None` | Config for RoPE (Rotary Position Embeddings) — e.g. `rope_theta`, scaling type |

### `validate_architecture`
Enforces `hidden_size % num_attention_heads == 0` — every head needs an equal, whole slice of the hidden dimension.

---

## Multi-GPU Parallelism

LLaMA's `base_model_tp_plan` and `base_model_pp_plan` (in `configuration_llama.py`) are the **original definitions** that Gemma's config (and most other LLaMA-family models) reuses unchanged:

```python
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
base_model_pp_plan = {
    "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    "norm": (["hidden_states"], ["hidden_states"]),
}
```

The full reasoning for *why* `colwise`/`rowwise` are assigned this way (Tensor Parallelism vs Pipeline Parallelism, the `AllReduce` math, worked examples with concrete matrix shapes) is written up in detail in `../gemma/NOTES.md` under "Multi-GPU Parallelism" — it's the same plan, just first defined here in LLaMA's config and inherited by Gemma's. The short version:

- **`colwise`** (`q/k/v_proj`, `gate_proj`, `up_proj`): these *expand* hidden → a larger dimension; each GPU can independently own a slice of output columns (e.g. a subset of attention heads), no communication needed mid-layer.
- **`rowwise`** (`o_proj`, `down_proj`): these *contract* a larger dimension back to hidden; each GPU computes a partial sum over its rows, then an `AllReduce` combines all partials into the final output.
- They're always paired (`colwise → rowwise`) so the column-split output of one layer exactly matches the row-split input the next layer needs — only **one** `AllReduce` per pair, not one per layer.
- `base_model_pp_plan` instead splits *whole layers* sequentially across GPUs (assembly-line style) rather than splitting individual weight matrices — see `../gemma/NOTES.md` for the full GPU1→GPU2→GPU3 walkthrough.

---

## Grouped-Query Attention (GQA) — `repeat_kv`

LLaMA's attention supports GQA via `num_key_value_heads < num_attention_heads`. Fewer K/V heads means less KV-cache memory and less compute for the K/V projections, while still giving every query head its own attention pattern.

```python
def repeat_kv(hidden_states, n_rep):
    # (batch, num_key_value_heads, seqlen, head_dim)
    #   → (batch, num_key_value_heads * n_rep, seqlen, head_dim)
    ...
```

`n_rep = num_attention_heads // num_key_value_heads` — each K/V head is duplicated `n_rep` times (via `expand`, not actual copy, until the final `reshape`) so that every query head has a matching K/V head to attend against. If `num_key_value_heads == num_attention_heads` (the default, no GQA), `n_rep == 1` and this is a no-op.

This happens **inside `eager_attention_forward`**, right before the `Q @ Kᵗ` matmul — the K/V projections themselves stay at the smaller `num_key_value_heads` size (that's where the memory savings come from); they're only logically expanded at attention-compute time.

---

## Attention Backend Dispatch — `ALL_ATTENTION_FUNCTIONS`

`LlamaAttention.forward` doesn't hardcode which attention algorithm runs — it looks one up by name:

```python
attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
```

**What `_attn_implementation` is:** a string set on the config (`"eager"`, `"sdpa"`, `"flash_attention_2"`, `"flex_attention"`, ...), either chosen explicitly when loading a model (`from_pretrained(..., attn_implementation="sdpa")`) or auto-selected by the framework based on what's installed/supported on the current hardware.

**`ALL_ATTENTION_FUNCTIONS` is a global registry**, not a hardcoded if/else — every backend registers itself under its string name (e.g. `sdpa_attention_forward` under `"sdpa"`, a FlashAttention wrapper under `"flash_attention_2"`). `get_interface(name, default)` looks up that name and falls back to `eager_attention_forward` (the plain-PyTorch reference implementation documented above) if the name isn't found or isn't requested.

**Why this matters in practice:**
- `eager_attention_forward` is the only one that *materializes* the full `[batch, heads, seq, seq]` attention-weight matrix — useful for debugging/inspecting attention patterns (`output_attentions=True` requires it), but `O(seq²)` memory.
- `sdpa` calls `torch.nn.functional.scaled_dot_product_attention`, which PyTorch internally fuses/optimizes (and can itself dispatch to a flash-attention-style kernel) without ever returning attention weights — faster and lower-memory, but `output_attentions=True` isn't available through this path.
- `flash_attention_2` calls the actual FlashAttention CUDA kernel — fastest and most memory-efficient for long sequences, but requires the `flash-attn` package and a supported GPU.
- All of these compute the *same* numerical result (up to floating point ordering) — the swap is purely a performance/feature tradeoff, not a behavior change. This is why `_supports_flash_attn` / `_supports_sdpa` / `_supports_flex_attn` flags exist on `LlamaPreTrainedModel`: they declare which of these registry entries this model architecture is actually compatible with.

`repeat_kv` (the GQA broadcast) happens *inside* `eager_attention_forward` specifically — other backends like `flash_attention_2` and `sdpa` handle GQA internally/natively (they accept the smaller K/V head count directly), so `repeat_kv` is not a step every backend repeats; it's an artifact of the plain-PyTorch fallback needing matching head counts for `torch.matmul`.

---

## RoPE Scaling Variants — `rope_type` and `ROPE_INIT_FUNCTIONS`

`LlamaRotaryEmbedding.__init__` picks its frequency-generation function based on `config.rope_parameters["rope_type"]`:

```python
self.rope_type = self.config.rope_parameters["rope_type"]
rope_init_fn = self.compute_default_rope_parameters
if self.rope_type != "default":
    rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
```

**The problem this solves:** a model trained with `max_position_embeddings = 2048` produces garbage attention patterns if you naively feed it a 16k-token sequence — the RoPE angles for positions beyond what it saw in training are out-of-distribution. The scaling variants are different strategies for extending usable context length beyond the original training length, registered in `ROPE_INIT_FUNCTIONS` (in `modeling_rope_utils.py`) and selected purely by the `rope_type` string in config — `LlamaRotaryEmbedding` itself doesn't know the difference, it just calls whichever function the registry hands back.

**Common `rope_type` values:**

| `rope_type` | What it does | Tradeoff |
|---|---|---|
| `"default"` | Standard RoPE, `inv_freq = 1/base^(i/dim)`, `attention_scaling = 1.0` | Only reliable up to `max_position_embeddings` |
| `"linear"` | Divides position ids by a fixed `factor` before computing angles — effectively compresses a longer sequence into the range the model was trained on | Extends context cheaply, but degrades resolution/precision at every position (it's literally squeezing more positions into the same angular range) |
| `"dynamic"` (NTK-aware) | Recomputes `inv_freq` on the fly, scaling `base` upward as the actual sequence length exceeds the original training length (only kicks in past that point) | No quality loss for sequences within the original length; gracefully extends beyond it instead of degrading immediately at the boundary |
| `"yarn"` | Frequency-dependent interpolation — scales different frequency bands differently (high frequencies barely touched, low frequencies stretched more) rather than one uniform `factor` | Better quality extension than plain linear/dynamic scaling, used by most modern long-context LLaMA-family fine-tunes; more complex to compute |
| `"llama3"` | Meta's own scaling scheme introduced for Llama 3.1, a refinement on top of NTK-style scaling tuned specifically for their long-context release | Model-family-specific; same dynamic-rescaling spirit as `"dynamic"`/`"yarn"` |

**What `attention_scaling` is for:** some of these strategies (e.g. YaRN) don't just change the *angles*, they also need to rescale the resulting attention *logits* slightly to compensate for the interpolation — `attention_scaling` is a multiplier applied to `cos`/`sin` in `LlamaRotaryEmbedding.forward` (`cos = emb.cos() * self.attention_scaling`) for exactly this correction. For `"default"` and most variants it's `1.0` (no-op); it only differs from `1.0` for scaling strategies where the math actually demands a logit correction.

**Why `original_inv_freq` is kept around (`dynamic_rope_update` decorator):** strategies like `"dynamic"` need to *recompute* `inv_freq` mid-generation as the sequence grows past `max_position_embeddings` — `dynamic_rope_update` wraps `forward` to check the current `seq_len` against the cached value and re-derive `inv_freq` from `original_inv_freq` (the untouched original) when needed, rather than compounding repeated rescalings on top of an already-rescaled buffer.

---

## KV Cache — `DynamicCache` and `past_key_values.update()`

Every reference to `past_key_values` across this model boils down to one mechanism: avoid recomputing attention keys/values for tokens that were already processed in a previous forward call.

**Without a cache:** generating token N+1 after already generating tokens `1..N` would require re-running the *entire* sequence `1..N` through every layer's `k_proj`/`v_proj` again, just to get back the same K/V vectors computed last step — wasted compute that grows quadratically over a generation.

**With the cache:**
1. **First call** (the prompt) — `use_cache=True` and no `past_key_values` passed in, so `LlamaModel.forward` creates a fresh `DynamicCache(config=self.config)`. The prompt runs through normally; inside each `LlamaAttention.forward`, `past_key_values.update(key_states, value_states, self.layer_idx)` stores that layer's K/V for every prompt position, keyed by `layer_idx` (each layer keeps its own independent cache — layer 0's keys are unrelated to layer 5's).
2. **Subsequent calls** (one new token at a time) — the caller passes the same `past_key_values` object back in. `update()` **appends** the new token's K/V onto what's already stored for that layer and returns the **full concatenated** K/V (old + new) — that full tensor is what attention actually runs against, so the new token can still attend to everything before it.
3. Only the **new** token's hidden state goes through `q_proj`/`k_proj`/`v_proj` on each subsequent call — `q` is computed only for the new position, but it attends against the *entire* cached `key_states`/`value_states` history. This is the actual compute saving: O(1) new work per step instead of O(seq_len).

**Why position IDs need the cache's length:** `LlamaModel.forward` computes `position_ids` as `arange(new_tokens) + past_seen_tokens`, where `past_seen_tokens = past_key_values.get_seq_length()` — this is *why* the cache has to be threaded through, not just for K/V storage: it's also how the model knows token N+1 is at position N+1 and not position 0 again.

**Why "Dynamic":** `DynamicCache` grows its underlying tensors on demand (concatenating new K/V onto the existing buffer each step) rather than pre-allocating a fixed-size buffer up front — simple and correct, though other cache implementations (e.g. a static, pre-allocated cache) trade that flexibility for being more `torch.compile`/CUDA-graph friendly, since a fixed buffer shape doesn't trigger recompilation as the sequence grows.

**Tying it back to GQA:** the cache stores K/V at the *smaller* `num_key_value_heads` size (before `repeat_kv` broadcasts them up to match query heads at attention-compute time) — this is the actual mechanism by which GQA reduces memory: a smaller cache, not less compute work per se.

---

## Shared Infrastructure (Used by LLaMA, Not Specific to It)

The two pieces below — `create_causal_mask` and `GenerationMixin` — are called directly from `LlamaModel.forward` and `LlamaForCausalLM`, but the actual logic lives in generic, model-agnostic files (`masking_utils.py`, `generation/utils.py`) shared by virtually every decoder-only model in `transformers`. Worth understanding since they're load-bearing for how LLaMA actually runs, even though there's nothing LLaMA-specific about them.

### `create_causal_mask` — sliding-window support and merging with padding

`LlamaModel.forward` calls this once per forward pass:
```python
causal_mask = create_causal_mask(
    config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
    past_key_values=past_key_values, position_ids=position_ids,
)
```

**The core mask is built from small boolean "mask functions"**, not a hand-rolled triangular matrix. `causal_mask_function(batch_idx, head_idx, q_idx, kv_idx) → kv_idx <= q_idx` is the base rule ("a query position can see any key position at or before it"). These functions get composed:

- **`and_masks(*fns)` / `or_masks(*fns)`** — combine multiple boolean mask functions via intersection/union. This is the general mechanism, not special-cased per feature.
- **Sliding window**: `sliding_window_overlay(window)` returns `kv_idx > q_idx - window` (i.e. "not too far in the past"); `sliding_window_causal_mask_function` is just `and_masks(sliding_window_overlay(window), causal_mask_function)` — causal *and* within the window. Plain LLaMA doesn't use this (no `sliding_window` in `LlamaConfig`), but it's the same mask-building machinery models like Mistral's local-attention layers plug into.
- **Padding mask merge**: `padding_mask_function(padding_mask)` turns the 2D `[batch, seq]` boolean attention mask (`True` = real token, `False` = padding) into the same 4-argument signature (`padding_mask[batch_idx, kv_idx]`), so it composes via `and_masks` with the causal rule exactly like the sliding-window overlay does — padding is just another overlay, not a separate code path.

**Why this matters for LLaMA specifically:** `LlamaAttention` only ever sees the *final* materialized mask tensor (or `BlockMask` for FlexAttention) — it never expresses sliding-window/padding logic itself. If a future LLaMA-family model variant needed sliding window attention, it would only need to set the right field in its config and point at this same machinery; no change to `LlamaAttention`'s forward signature.

**Backend-aware shortcuts:** `_ignore_causal_mask_sdpa` checks whether the mask can be skipped entirely and replaced with SDPA's own `is_causal=True` flag (cheaper — lets PyTorch dispatch straight to a flash-attention-style kernel instead of materializing and adding an explicit mask tensor). This is only safe when there's no padding and the query/kv lengths align in specific ways (e.g. decoding one token at a time, or a full unpadded prefill) — `create_causal_mask` checks these conditions and returns `None` (meaning "use the implicit causal flag") rather than an actual tensor whenever it can.

**KV-cache interaction:** `_preprocess_mask_arguments` pulls `kv_length`/`kv_offset` from `past_key_values.get_mask_sizes(...)` when a cache is present — this is how the mask correctly accounts for already-cached tokens (see the KV Cache section above) without the mask-building code needing to know anything about `DynamicCache` internals itself.

### `GenerationMixin` — `.generate()` and sampling strategies

`LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin)` — the second base class is what gives every LLaMA causal-LM instance its `.generate(...)` method; `LlamaForCausalLM.forward` itself only computes one step's logits, it has no loop. `GenerationMixin.generate()` is the actual decoding loop, and it dispatches to one of several internal strategies based on the `GenerationConfig` passed in (or the model's default):

| Strategy | Internal method | What it does |
|---|---|---|
| Greedy | `_sample` (with `do_sample=False`) | Always picks the single highest-probability next token |
| Sampling | `_sample` (with `do_sample=True`) | Draws from the next-token distribution, shaped by `temperature` (flattens/sharpens the distribution), `top_k` (restrict to k most likely tokens), `top_p`/nucleus (restrict to the smallest set of tokens whose cumulative probability ≥ p) |
| Beam search | `_beam_search` | Tracks `num_beams` candidate sequences in parallel, expanding and pruning by cumulative log-probability, rather than committing to one token at a time |

**Why this is generic, not LLaMA-specific:** every strategy above operates purely on the `logits` tensor `LlamaForCausalLM.forward` returns plus whatever `past_key_values` cache it returns — `GenerationMixin` doesn't know or care that the underlying model is LLaMA. The loop is: call `forward()` for the new token(s), pick/sample the next token id(s) from the returned logits, append to the sequence, feed the cache + new token back in next iteration, repeat until `eos_token_id` or `max_length`. This is exactly why the KV cache and `logits_to_keep` machinery documented earlier exist — they're the hooks `GenerationMixin`'s loop relies on to avoid redoing work every step.

**The one place LLaMA *does* customize generation behavior:** indirectly, through config — `eos_token_id`, `pad_token_id`, `max_position_embeddings` (generation stops or needs RoPE-scaling intervention once a sequence approaches this) are all read by `GenerationMixin` from `LlamaConfig`, but the generation *algorithm* itself is untouched.

---

## RoPE (Rotary Position Embeddings) — How It's Applied

1. **`LlamaRotaryEmbedding.__init__`** — precomputes `inv_freq`, a vector of `head_dim / 2` inverse frequencies, geometrically spaced based on `rope_theta` (the RoPE base, e.g. `10000.0`). Different RoPE *scaling* strategies (linear, dynamic NTK, YaRN, etc.) are selected via `config.rope_parameters["rope_type"]` and looked up in `ROPE_INIT_FUNCTIONS`.

2. **`forward(x, position_ids)`** — for each position, computes `freqs = inv_freq ⊗ position_ids`, then `cos`/`sin` of those frequencies (duplicated across the full `head_dim` via `torch.cat((freqs, freqs))`). Computed **once per forward pass** in `LlamaModel.forward`, then reused by every decoder layer (not recomputed per-layer).

3. **`apply_rotary_pos_emb(q, k, cos, sin)`** — actually rotates the query/key vectors:
   ```python
   q_embed = (q * cos) + (rotate_half(q) * sin)
   ```
   `rotate_half` splits a vector in half and does `(-x2, x1)` — this is the standard 2D-rotation-per-pair trick that encodes *relative* position into the dot product between any two tokens' Q and K vectors, without needing an explicit positional embedding added to the input.

This same mechanism is inherited unchanged by `GemmaRotaryEmbedding(LlamaRotaryEmbedding): pass` (see `../gemma/modular_gemma_notes.MD`).
