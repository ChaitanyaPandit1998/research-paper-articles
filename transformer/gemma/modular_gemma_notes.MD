# `modular_gemma.py` — Class-by-Class Notes

Reminder of the big picture (see `NOTES.md`): Gemma is architecturally almost identical to LLaMA. This file inherits LLaMA's classes and only writes the bits that are actually different. Below is what each piece does, in plain terms.

---

## `GemmaTextScaledWordEmbedding`

```python
class GemmaTextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
```

**What it does:** a normal token-embedding lookup table (`nn.Embedding` — turns a token ID into a vector), but every embedding it returns gets **multiplied by a constant scale factor** (`√hidden_size`, e.g. `√3072 ≈ 55.4`).

**Why:** Gemma's authors found that scaling the embeddings up at the input keeps the magnitude of values flowing through the network in a numerically nicer range, given Gemma's other architectural choices. LLaMA doesn't do this — its embeddings are used as-is.

**Implementation detail worth noting:**
- `embed_scale` is stored as a **buffer** (`register_buffer`, `persistent=False`) — a tensor that moves with the model to GPU/CPU and changes dtype with `.to()` calls, but isn't a trainable parameter and isn't saved in the checkpoint.
- `scalar_embed_scale` keeps the *plain Python float* around too — this matters because of a subtle bug class: if you only ever stored the scale as a `bfloat16` tensor, repeated `.to(dtype)` casts could introduce tiny rounding drift (e.g. `√3072 = 55.4256` rounds to `55.5` in `bfloat16`). Keeping the original float lets the model re-derive the buffer correctly during weight initialization (see `GemmaPreTrainedModel._init_weights` below) instead of compounding rounding error.

---

## `GemmaRMSNorm`

```python
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
```

**What it does:** RMSNorm (Root-Mean-Square Normalization) — a lighter alternative to LayerNorm that just rescales a vector by its RMS magnitude, without subtracting the mean. `_norm` does the actual rescaling: divide by `sqrt(mean(x²) + eps)`.

**The Gemma-specific twist:** the learnable weight is applied as `(1 + weight)` instead of just `weight`, and `weight` is **initialized to zeros** (see `_init_weights` below) — so at initialization, the norm scales by exactly `1`, i.e. it starts as a no-op. LLaMA initializes its norm weight to `1` directly and multiplies by `weight` (no `+1`). Both reach the same starting point (`weight_effective = 1`), but Gemma's parameterization makes it easier to regularize the weight toward `0` during training (a common trick — keeping trainable scale factors centered at 0 rather than 1 plays nicer with weight decay).

**The comment about float16 ordering:**
```python
# Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
```
This is about **order of operations and numerical precision**. LLaMA casts the normalized values down to `float16`/`bfloat16` *before* multiplying by the weight. Gemma does the multiplication in full `float32` precision, and only casts down to the original dtype (`type_as(x)`) at the very end. This avoids extra rounding error from doing the multiply in lower precision — a subtle but deliberate fix (linked PR #29402 in the source).

---

## `GemmaMLP`

```python
class GemmaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
```

**What it does:** inherits LLaMA's entire MLP (the SwiGLU-style gated feed-forward block: `down_proj(activation(gate_proj(x)) * up_proj(x))`), then just **redeclares the three linear layers explicitly**.

**Why redeclare if it's identical to LLaMA's?** Functionally this is the same shape and same `bias=False` as LLaMA already sets up in `super().__init__(config)` — this override exists mainly so this model's code is explicit/self-documenting about its projection layers, and to make it easy to later tweak Gemma's MLP independently (e.g. different bias, different activation) without touching LLaMA's class. As of this snippet it's effectively a no-op rewrite of what the parent already does.

---

## `GemmaRotaryEmbedding`

```python
class GemmaRotaryEmbedding(LlamaRotaryEmbedding):
    pass
```

**What it does:** nothing — Gemma uses RoPE (Rotary Position Embeddings) exactly the same way LLaMA does, so this class exists only so that `GemmaRotaryEmbedding` is a distinct, importable name (for clarity, type-checking, and the auto-generation tooling), even though all the logic is inherited unchanged.

---

## `GemmaAttention`

```python
class GemmaAttention(LlamaAttention):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)
```

**What it does:** inherits all of LLaMA's multi-head attention math (Q/K/V projections, RoPE application, scaled dot-product, output projection), and adds exactly one behavior switch: `is_causal`.

**`is_causal` in plain terms:** causal attention means each token can only "look at" tokens before it (standard for autoregressive text generation — you can't see the future). If `use_bidirectional_attention=True` in the config, this flips `is_causal` to `False`, meaning every token can attend to every other token, both before and after — needed for use cases like encoding a full sentence for embeddings/classification rather than generating text left-to-right.

---

## `GemmaPreTrainedModel`

```python
class GemmaPreTrainedModel(LlamaPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if "RMSNorm" in module.__class__.__name__:
            init.zeros_(module.weight)
        elif isinstance(module, GemmaTextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
```

**What it does:** defines how weights get randomly initialized when you create a fresh (untrained) Gemma model. It first runs the generic initialization that all models share (`PreTrainedModel._init_weights`), then patches two Gemma-specific cases:

1. **Any RMSNorm weight → initialized to all zeros.** This is what makes `GemmaRMSNorm`'s `(1 + weight)` start out equal to `1` (a no-op scale), as described above.
2. **The embedding's `embed_scale` buffer → reset to the exact float value (`scalar_embed_scale`).** This re-derives the buffer from the original Python float rather than trusting whatever dtype-cast tensor might currently be sitting there — guarding against the `bfloat16` rounding issue mentioned earlier (e.g. `55.4256` truncating to `55.5`).

---

## `GemmaModel`

```python
class GemmaModel(LlamaModel):
    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.embed_tokens = GemmaTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        ...
```

**The `__init__` override:** swaps out LLaMA's plain `nn.Embedding` for Gemma's `GemmaTextScaledWordEmbedding`, passing `embed_scale = √hidden_size` — this is where the scaling constant from the very first class actually gets wired in.

**The `forward` override — walking through what it does, step by step:**

1. **Input validation** — exactly one of `input_ids` (raw token IDs) or `inputs_embeds` (pre-computed embeddings) must be given, not both, not neither.
2. **Embed tokens** — if you passed raw `input_ids`, run them through `self.embed_tokens` (the scaled embedding from above) to get vectors.
3. **Set up the KV cache** — if `use_cache=True` and no cache was passed in, create a fresh `DynamicCache` (this is what lets generation reuse previously computed attention keys/values instead of recomputing them every new token — standard inference speedup).
4. **Compute position IDs** — if not provided, build them as a simple increasing range `[0, 1, 2, ...]`, offset by however many tokens are already in the cache (so generation continues numbering positions correctly across calls).
5. **Build the causal attention mask** — `create_causal_mask` constructs the actual mask tensor that prevents attending to future tokens (or skips this restriction entirely under bidirectional mode, per `GemmaAttention.is_causal` above).
6. **Compute RoPE position embeddings once** — `self.rotary_emb(...)` precomputes the rotary embedding values for all positions, which every decoder layer below will reuse (computed once, not per-layer, for efficiency).
7. **Run through every decoder layer** — loop over `self.layers`, threading `hidden_states` through each one along with the mask, position ids, cache, and the precomputed rotary embeddings.
8. **Final norm** — apply `self.norm` (a `GemmaRMSNorm`) to the last layer's output.
9. **Return** — package the final hidden states (and the updated cache, if caching was used) into a `BaseModelOutputWithPast`.

This is essentially LLaMA's forward pass, copied here because Gemma needed to override `__init__` (for the scaled embedding) — when you override `__init__` in a modular file, you typically need to also redeclare `forward` so the generated `modeling_gemma.py` is self-contained.

---

## `GemmaForCausalLM`

```python
class GemmaForCausalLM(LlamaForCausalLM):
    def forward(**super_kwargs):
        r"""... docstring with a usage example ..."""
        return super().forward(**super_kwargs)
```

**What it does:** literally just calls LLaMA's existing `forward` unchanged (`super().forward(**super_kwargs)`). The entire point of overriding it here is to attach a **Gemma-specific docstring** (showing `google/gemma-7b` in the example) instead of LLaMA's docstring — purely a documentation override, no behavior change. This is a common modular-file pattern: override a method just to swap its example/docs while keeping the same logic.

---

## `GemmaForSequenceClassification` / `GemmaForTokenClassification`

```python
class GemmaForSequenceClassification(LlamaForSequenceClassification):
    pass

class GemmaForTokenClassification(LlamaForTokenClassification):
    pass
```

**What they do:** nothing extra — full inheritance, zero overrides. These exist purely to give Gemma its own class names for sequence classification (e.g. sentiment analysis — one label per whole input) and token classification (e.g. named entity recognition — one label per token), reusing LLaMA's classification head logic wholesale.

---

## `__all__`

```python
__all__ = [
    "GemmaConfig",
    "GemmaModel",
    "GemmaForCausalLM",
    "GemmaForSequenceClassification",
    "GemmaForTokenClassification",
    "GemmaPreTrainedModel",
]
```

**What it does:** declares which names are the "public API" of this module — these are the names that `from transformers.models.gemma.modeling_gemma import *` would import, and (relevant to [[INIT_PY_EXPLAINED]]) these are the names `define_import_structure` picks up when building the lazy-loading map. Note `GemmaRMSNorm`, `GemmaAttention`, `GemmaMLP`, etc. are *not* in this list — they're internal building blocks, not meant to be imported directly by end users.

---

## Summary Table — What Gemma Actually Changes vs LLaMA

| Class | Change from LLaMA |
|---|---|
| `GemmaTextScaledWordEmbedding` | New class: multiplies embeddings by `√hidden_size` |
| `GemmaRMSNorm` | `(1 + weight)` instead of `weight`; weight init to 0; multiply in float32 before downcasting |
| `GemmaMLP` | No real change (explicit re-declaration of identical layers) |
| `GemmaRotaryEmbedding` | No change at all (`pass`) |
| `GemmaAttention` | Adds optional bidirectional (non-causal) attention mode |
| `GemmaPreTrainedModel` | Custom weight init for RMSNorm (zeros) and embed_scale buffer |
| `GemmaModel` | Uses the scaled embedding; forward pass otherwise same as LLaMA |
| `GemmaForCausalLM` | No behavior change, only a docstring/example override |
| `GemmaForSequenceClassification` / `GemmaForTokenClassification` | No change at all |

This matches the summary already in `NOTES.md`: Gemma really is "LLaMA + embedding scaling + tweaked RMSNorm + optional bidirectional attention," and everything else is inherited as-is.
