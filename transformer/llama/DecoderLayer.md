# Llama Decoder Layer: residual connections — explained

A Llama decoder layer wraps the attention block (see `RoPE.md` and `Attention.md`) and the MLP block, each with its own pre-norm + residual connection. This file covers the residual connection pattern itself.

## `hidden_states = residual + hidden_states`

This is the **residual connection** (skip connection) — it adds the attention block's output back onto the original input that went into the block, rather than just replacing it.

**Where it sits:** In a Llama decoder layer, the typical flow is:

```python
residual = hidden_states              # save the input before attention
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(hidden_states, ...)   # everything in Attention.md happens here
hidden_states = residual + hidden_states               # ← this line
```

So `residual` is a copy of `hidden_states` captured *before* normalization and attention were applied. After attention finishes and produces its output, that output is **added to** (not used to replace) the original `residual`.

## Why this matters

- **Gradient flow:** In deep networks (Llama has dozens of layers), gradients can vanish as they backpropagate through many transformations. A residual connection gives gradients a direct path (`d(output)/d(residual) = 1`) straight back to earlier layers, bypassing the attention computation entirely. This is what makes training very deep transformers tractable.
- **What attention is actually learning:** Because of the residual, `self_attn(...)` doesn't need to learn to preserve/reconstruct the original token representation — it only needs to learn the *correction* or *update* to add on top of it (e.g. "mix in some context from other tokens"). If attention output were `0` for some token, that token's representation would just pass through unchanged.
- **Identity at initialization:** If a layer's weights start near-zero, the whole block initially behaves close to an identity function (`output ≈ residual`), which is a much easier starting point to optimize from than a network where every layer drastically transforms its input.

## Simple example

Say `residual = [1.0, 2.0, 3.0]` (the token's representation going into attention), and the attention sub-layer computes `hidden_states = [0.1, -0.2, 0.05]` (a small "context update" learned from looking at other tokens).

```
hidden_states = residual + hidden_states
              = [1.0+0.1, 2.0+(-0.2), 3.0+0.05]
              = [1.1, 1.8, 3.05]
```

The token's representation moves only slightly from its original value — it's "the same token, nudged by what it learned from attention," not a full replacement.

This same pattern (`residual = x; x = sublayer(x); x = residual + x`) repeats a second time later in the decoder layer around the MLP/feed-forward block too.
