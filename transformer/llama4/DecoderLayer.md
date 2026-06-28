# Llama4 Decoder Layer — What's New vs Llama3

The residual connection pattern (save residual → normalise → sublayer → add residual) is identical to Llama3. See `../llama/DecoderLayer.md` for that explanation. This file covers only the structural differences.

---

## Per-Layer Feed-Forward Selection

The most important difference: each decoder layer independently decides whether its FFN is a **dense MLP** or a **MoE block** at construction time, based on `layer_idx`:

```python
class Llama4TextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.is_moe_layer = layer_idx in config.moe_layers

        if self.is_moe_layer:
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)
```

The attribute is named `feed_forward` (not `mlp` as in Llama3). The two variants have different intermediate sizes:
- `Llama4TextMoE` uses `config.intermediate_size = 8192` for its per-expert FFN
- `Llama4TextMLP` (dense fallback) uses `config.intermediate_size_mlp = 16384`

---

## MoE Output Unpacking

Because `Llama4TextMoe.forward` returns a tuple `(hidden_states, router_logits)` while `Llama4TextMLP.forward` returns just `hidden_states`, the decoder layer unpacks conditionally:

```python
def forward(self, hidden_states, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    attention_states, _ = self.self_attn(hidden_states, ...)
    hidden_states = residual + attention_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.feed_forward(hidden_states)

    if self.is_moe_layer:
        hidden_states, _ = hidden_states   # unpack (output, router_logits)

    hidden_states = residual + hidden_states.view(residual.shape)
    return hidden_states
```

The `router_logits` (discarded with `_` here) are captured separately by the model's `_can_record_outputs` infrastructure for computing the auxiliary load-balancing loss.

The `.view(residual.shape)` call is a safety reshape — MoE output may have been flattened to `[B*S, H]` during the expert computation and needs to be reshaped back to `[B, S, H]` before the residual add.

---

## Attention Forward Signature Change

`Llama4TextAttention.forward` receives `position_embeddings` as a **complex tensor** (`freqs_cis`), not a `(cos, sin)` tuple as in Llama3:

```python
# Llama3:
def forward(self, hidden_states, position_embeddings: tuple[Tensor, Tensor], ...):
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

# Llama4:
def forward(self, hidden_states, position_embeddings: Tensor, ...):
    # position_embeddings is freqs_cis, a complex tensor
    if self.use_rope:
        q, k = apply_rotary_emb(q, k, position_embeddings.to(q.device))
```

`Llama4TextModel.forward` computes `freq_cis = self.rotary_emb(hidden_states, position_ids)` once and passes it down to all layers — same shared-computation pattern as Llama3, but the tensor type differs.

---

## `GradientCheckpointingLayer` Base Class

Both `Llama4TextDecoderLayer` and `Llama4VisionEncoderLayer` inherit from `GradientCheckpointingLayer` (a utility base class in transformers, not specific to Llama4). This adds activation checkpointing support: during training with `gradient_checkpointing=True`, the layer's `forward` outputs are discarded after the forward pass and recomputed during the backward pass rather than being stored in memory. This trades compute for memory, allowing training with longer sequences or larger batches. Llama3's `LlamaDecoderLayer` achieves the same via a different mechanism (`_gradient_checkpointing_func`), so the practical outcome is identical.
