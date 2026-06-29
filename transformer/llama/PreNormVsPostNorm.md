# Pre-Norm vs Post-Norm — explained

Llama uses **pre-norm**: normalisation happens *before* each sublayer (attention, MLP), not after. This is a deliberate departure from the original "Attention Is All You Need" transformer, which used post-norm. The choice has significant consequences for training stability and depth.

---

## The Two Patterns

### Post-Norm (original transformer, BERT, GPT-2)
```python
# Attention block
hidden_states = LayerNorm(hidden_states + self_attn(hidden_states))

# MLP block
hidden_states = LayerNorm(hidden_states + mlp(hidden_states))
```
Normalisation is applied *after* adding the sublayer output to the residual.

### Pre-Norm (Llama, GPT-3, PaLM, most modern LLMs)
```python
# Attention block
residual = hidden_states
hidden_states = residual + self_attn(RMSNorm(hidden_states))

# MLP block
residual = hidden_states
hidden_states = residual + mlp(RMSNorm(hidden_states))
```
Normalisation is applied *before* passing into the sublayer. The residual path bypasses normalisation entirely.

In Llama's actual code:
```python
def forward(self, hidden_states, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)       # ← norm BEFORE attention
    hidden_states, _ = self.self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states                  # ← residual add AFTER

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)  # ← norm BEFORE MLP
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
```

---

## Why Pre-Norm Trains More Stably

### The gradient problem in post-norm

In post-norm, the normalisation sits *on the output* of the residual connection:
```
output = Norm(x + sublayer(x))
```

During backpropagation, the gradient must flow through the `Norm` at every layer on its way back to early layers. `Norm` rescales activations but doesn't have a guaranteed straight path for gradients. In deep networks (32+ layers), this repeated rescaling can cause gradients to vanish (become tiny) or explode (become huge) before they reach the early layers — making those early layers essentially unable to learn.

### Pre-norm gives gradients a free highway

In pre-norm:
```
output = x + sublayer(Norm(x))
```

The gradient of the loss with respect to `x` has two paths:
1. Through `sublayer(Norm(x))` — the "learned" path, can vanish/explode
2. Directly through the `+` connection: `d(output)/d(x) = 1` — always exactly 1, independent of depth

This direct gradient path (`= 1`) means that *even if* the sublayer path's gradient vanishes, gradients still flow freely back through every residual connection to every layer. Early layers always get a signal. The model doesn't need to learn to propagate gradients — the architecture guarantees it.

### Concrete depth comparison

With 32 Llama layers:
- **Post-norm:** gradient from the last layer to the first passes through 32 LayerNorm operations, each potentially distorting it. In practice, post-norm transformers require careful learning rate warm-up and are notoriously difficult to train beyond ~24 layers without instability.
- **Pre-norm:** gradient always has the `+1` path. Llama trains 70B+ parameter models (80 layers) without any special gradient tricks.

---

## The Trade-Off: Representation Collapse at Later Layers

Pre-norm has one known downside: because each sublayer's input is normalised independently, the *unnormalised* residual stream (the raw `hidden_states` before any norm is applied) can grow very large in magnitude over depth. Later layers see a well-normalised input, but the residual itself may have high variance. This can cause the model to effectively "ignore" the small sublayer outputs relative to the large residual, reducing the effective depth — a phenomenon sometimes called **representation collapse** or **rank collapse** in deep pre-norm networks.

Post-norm doesn't have this problem because the normalisation keeps the residual stream bounded at every layer.

In practice, pre-norm's training stability advantage far outweighs this concern for LLMs — the collapse is manageable and doesn't prevent models from training well. Techniques like QK-norm (which Llama4 adds) partially address the magnitude growth problem inside the attention heads specifically.

---

## What "Pre-Norm" Means for Llama Specifically

Llama uses **RMSNorm** rather than LayerNorm in the pre-norm position. This removes the mean-centering step (no `x - mean(x)`) and the bias term from the normalisation entirely, making it cheaper and slightly less expressive but sufficient in practice.

Two norms per decoder layer:
- `self.input_layernorm` — before attention
- `self.post_attention_layernorm` — before MLP

One norm at the end of the whole stack:
- `self.norm` in `LlamaModel` — final RMSNorm before the LM head

There is **no normalisation after the residual add**, and **no normalisation on the residual path itself** — the bypass path is completely unmodified, which is what preserves the gradient highway.

---

## Visual Summary

```
Post-Norm (original transformer):
  x ──────────────────────────────┐
       ↓                          ↓
  sublayer(x) ──────────────→  x + out  → Norm → next layer
                                    ↑ gradient must pass through Norm every layer

Pre-Norm (Llama):
  x ──────────────────────────────┐
  ↓                               ↓
  Norm → sublayer → out ────→  x + out → next layer
                                    ↑ gradient has direct path (+1) AND sublayer path
```
