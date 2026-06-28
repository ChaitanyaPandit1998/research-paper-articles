# Llama4 Model — Notes

These notes cover **only what is new or different in Llama4 vs Llama3**. For the foundational machinery — RoPE math, RMSNorm, residual connections, KV cache, GQA, attention backend dispatch, GenerationMixin — see the parallel files in `../llama/`.

---

## Files in `src/transformers/models/llama4/`

| File | Purpose |
|---|---|
| `configuration_llama4.py` | Three config classes: `Llama4Config` (top-level multimodal), `Llama4TextConfig` (text decoder), `Llama4VisionConfig` (vision encoder) |
| `modeling_llama4.py` | Full model: vision encoder, text decoder (with MoE), projector, multimodal assembly |
| `processing_llama4.py` | Multimodal processor — interleaves image tokens into the text sequence |
| `image_processing_llama4.py` | Tiling/resizing/normalisation of input images before the vision encoder |
| `convert_llama4_weights_to_hf.py` | One-time script to import Meta's original checkpoint format |

Unlike Llama3, Llama4 has **no `modular_llama4.py`** — the entire architecture is written out directly in `modeling_llama4.py`.

---

## Three Config Classes

Llama4 splits its config into three tightly nested classes — this is different from Llama3's single flat `LlamaConfig`.

```
Llama4Config  (top-level, model_type="llama4")
├── text_config: Llama4TextConfig   (model_type="llama4_text")
└── vision_config: Llama4VisionConfig  (model_type="llama4_vision_model")
```

`Llama4Config.__post_init__` instantiates `Llama4TextConfig` and `Llama4VisionConfig` from dicts (or defaults) and stores them as sub-configs. This means `model.config.text_config.hidden_size` and `model.config.vision_config.hidden_size` are independently set and can differ (they do: text=5120, vision=768 for Scout 17B-16E).

### Key `Llama4TextConfig` fields (new vs Llama3)

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 202048 | Much larger vocabulary than Llama3's 128256 |
| `hidden_size` | 5120 | Up from 4096 |
| `intermediate_size` | 8192 | MoE expert intermediate size |
| `intermediate_size_mlp` | 16384 | Dense-layer intermediate size (larger than MoE experts) |
| `num_hidden_layers` | 48 | Up from 32 |
| `num_attention_heads` | 40 | — |
| `num_key_value_heads` | 8 | GQA: 5 Q heads per KV head |
| `head_dim` | 128 | Explicit; no longer inferred |
| `max_position_embeddings` | 131072 (4096×32) | 128K context |
| `num_experts_per_tok` | 1 | Top-1 routing (sparse MoE) |
| `num_local_experts` | 16 | Experts per MoE layer (Scout 17B-16E) |
| `moe_layers` | `None` → derived | List of layer indices that use MoE; defaults to every layer via `interleave_moe_layer_step=1` |
| `interleave_moe_layer_step` | 1 | Spacing between MoE layers (1 = every layer is MoE) |
| `use_qk_norm` | `True` | L2-normalize Q/K before attention on RoPE layers |
| `no_rope_layers` | `None` → derived | Per-layer flag: `1` = use RoPE, `0` = NoPE |
| `no_rope_layer_interval` | 4 | Insert a NoPE layer every N layers if `no_rope_layers` is unset |
| `attention_chunk_size` | 8192 | Token window for chunked (NoPE) attention |
| `attn_temperature_tuning` | `True` | Dynamically scale query magnitude for NoPE layers at long positions |
| `floor_scale` | 8192 | Base sequence length for temperature tuning |
| `attn_scale` | 0.1 | Strength of temperature scaling |
| `layer_types` | `None` → derived | `"full_attention"` (RoPE) or `"chunked_attention"` (NoPE) per layer |

### `__post_init__` derives `layer_types` and `no_rope_layers`

```python
# If no_rope_layers not set, generate it: 0 (NoPE) every `no_rope_layer_interval` layers
default_no_rope_layers = [
    int((layer_idx + 1) % self.no_rope_layer_interval != 0)
    for layer_idx in range(self.num_hidden_layers)
]
# Result for interval=4, 48 layers:
# [1,1,1,0, 1,1,1,0, 1,1,1,0, ...] — every 4th layer is NoPE (0)

self.layer_types = [
    "chunked_attention" if no_rope else "full_attention"
    for no_rope in self.no_rope_layers
]
```

`moe_layers` is derived similarly: `range(interleave_moe_layer_step-1, num_hidden_layers, interleave_moe_layer_step)`. With default `interleave_moe_layer_step=1`, every layer is MoE.

---

## Two Parallelism Plans (TP and EP)

Llama4's `Llama4TextConfig` defines both a **tensor-parallel (TP)** plan and an **expert-parallel (EP)** plan — the first time a transformers model config ships with EP built in.

```python
base_model_tp_plan = {
    "layers.*.feed_forward.experts.gate_up_proj": "packed_rowwise",  # note: not colwise
    "layers.*.feed_forward.experts.down_proj":    "colwise",         # note: not rowwise
    "layers.*.feed_forward.router":               — (absent, not sharded in TP)
    ...
}

base_model_ep_plan = {
    "layers.*.feed_forward.experts.gate_up_proj": "grouped_gemm",
    "layers.*.feed_forward.experts.down_proj":    "grouped_gemm",
    "layers.*.feed_forward.router":               "ep_router",
    ...
}
```

**Why TP and EP for experts differ:** under TP, experts are split *within a single expert* across GPUs (each GPU holds a slice of every expert's weights). Under EP, *whole experts* are distributed — each GPU owns a subset of experts entirely. EP is typically more efficient for large MoE layers because it avoids cross-GPU communication during the expert computation itself (only the routing dispatch crosses GPU boundaries), but requires `AlltoAll` operations to send tokens to their assigned experts.

---

## Special Image Tokens

Llama4 introduces three image-related token IDs in `Llama4Config`:

| Token | ID | Meaning |
|---|---|---|
| `boi_token_index` | 200080 | Beginning-of-image marker |
| `eoi_token_index` | 200081 | End-of-image marker |
| `image_token_index` | 200092 | Placeholder for an image patch token |

During `Llama4ForConditionalGeneration.forward`, placeholder `image_token_index` positions in the text embedding sequence are overwritten with the actual projected vision features. The `boi`/`eoi` tokens delimit the image region within the text for the language model.

---

## What Llama4 Does NOT Have (Intentional Regressions from Llama3)

- **No `mlp_bias`** — there is no bias term on MLP projections (same as Llama3, but Llama4's `Llama4TextMLP` makes this unconditional rather than config-controlled).
- **No fp32 upcast for softmax** — Llama3's `eager_attention_forward` upcasts `attn_weights` to float32 before softmax (see `../llama/Attention.md`). Llama4's equivalent does **not**:
  ```python
  # Llama3: attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
  # Llama4: attn_weights = nn.functional.softmax(attn_weights, dim=-1)   ← no upcast
  ```
  This is an intentional choice, noted in the code comment: "llama4 doesn't cast attn weights to fp32."
- **Flash attention is disabled** — `_supports_flash_attn = False` on `Llama4PreTrainedModel`. SDPA and FlexAttention are supported instead.
