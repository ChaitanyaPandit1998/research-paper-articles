# The Residual Stream — A Model-Wide View

Every transformer (Llama, Llama4, GPT, Gemma) can be understood through two lenses:
1. **Layer-by-layer:** what each block does to its input
2. **Residual stream:** what the entire model does to a single shared tensor over depth

The residual stream view is more useful for reasoning about how information flows, why depth helps, and what goes wrong when models fail.

---

## What the Residual Stream Is

At every position in the sequence, there is one tensor — `hidden_states[b, s, :]`, a vector of shape `[hidden_size]` — that persists from the embedding layer all the way to the LM head. Each decoder layer does not *replace* this vector; it *adds a delta to it*:

```
h₀ = embed_tokens(input_ids)           # initial value — token identity

h₁ = h₀ + Δattn₁(RMSNorm(h₀))        # layer 1 attention adds its contribution
h₁ = h₁ + Δmlp₁(RMSNorm(h₁))         # layer 1 MLP adds its contribution

h₂ = h₁ + Δattn₂(RMSNorm(h₁))
h₂ = h₂ + Δmlp₂(RMSNorm(h₂))

...

h₃₂ = h₃₁ + Δattn₃₂(...) + Δmlp₃₂(...)

logits = lm_head(RMSNorm(h₃₂))
```

Every `Δ` is a residual contribution — a correction, an update, a piece of new information — being written into a shared accumulator. The same `h` vector is passed forward, layer after layer, each layer reading it and writing back a small change.

---

## Each Sublayer Writes a Residual, Not a Replacement

```python
# This is what actually runs in LlamaDecoderLayer.forward:
residual = hidden_states                     # READ current stream
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)       # COMPUTE delta
hidden_states = residual + hidden_states     # WRITE delta back to stream
```

Key implication: if a layer's attention output were zero, `hidden_states = residual + 0 = residual` — the stream passes through unchanged. **No layer is forced to do anything.** Each layer has the option to contribute nothing, and the model trains to make contributions only where useful.

At initialisation, weights are small (`initializer_range=0.02`), so all deltas start near zero. Early in training, the model is approximately an identity function — token embeddings flow through mostly unchanged. Training gradually teaches each layer what delta to add and when.

---

## What Each Layer Contributes

Different layers specialise on different kinds of updates:

**Attention sublayers** write information *from other token positions* into the current position's stream. When token `"Paris"` attends to `"capital of France"`, the attention delta carries contextual information about capital cities into `"Paris"`'s residual vector. Before this layer ran, `"Paris"` only knew about itself (from its embedding); after, its residual stream contains a blend of itself and its context.

**MLP sublayers** write information *from the model's parametric knowledge* — facts learned from training data — into the stream. MLPs don't look at other positions; they only look at the current stream vector and ask "given this representation, what else do I know?" A token vector that's been enriched by attention to `"capital of France"` might cause the MLP to add information like `"→ European city, population ~2M"` that wasn't in the original embedding or the attention context.

This specialisation is approximate and not absolute — attention layers can also trigger stored knowledge, and MLP layers can be influenced by context through the accumulated stream — but as a first approximation it's useful: **attention = routing/lookup across positions; MLP = lookup in parametric memory at the current position**.

---

## The Stream Grows Richer With Depth

Shallow in the network, the stream contains mainly token identity — what word this is. The attention and MLP deltas from early layers are small because they're starting from raw token embeddings.

Deep in the network, the stream contains a rich mixture: the original token, context from many other tokens (accumulated across many attention layers), stored facts triggered by that context, syntactic/semantic annotations added by earlier MLP layers, and corrections on top of corrections. The LM head reads this fully enriched vector and converts it to a vocabulary distribution.

This is why depth matters: not because late layers are "smarter" than early layers, but because late layers operate on a richer starting point built up by all previous layers. A 32-layer model has 32 opportunities to read the current stream state and add relevant information; a 1-layer model has only one.

---

## Implications for the KV Cache

The KV cache (see `../llama/NOTES.md`) stores the K/V projections from each past token's residual stream at each layer. This means the cache stores representations of past tokens *at a specific depth* — the stream state when that layer ran, not the final enriched representation.

When a new token attends to a cached position at layer 12, it's attending to that past token's layer-12 stream state — the sum of its embedding and all the attention/MLP deltas that layers 1–11 wrote into it. This is the right thing to attend to: by layer 12, the past token's representation has been enriched by 12 layers of context, which is exactly what the new token needs to contextualise against.

---

## Residual Stream in Llama4 MoE

In Llama4's MoE layers, the residual stream logic is unchanged — the MoE block still produces a delta that's added to the stream:

```python
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.feed_forward(hidden_states)   # MoE block
if self.is_moe_layer:
    hidden_states, _ = hidden_states
hidden_states = residual + hidden_states.view(residual.shape)   # delta added to stream
```

The MoE block decides *which expert* writes the delta, but the delta is still added to the same shared stream. From the stream's perspective, it doesn't know or care whether the delta came from a dense MLP or 16 sparse experts — it just accumulates the result.

---

## Why the Residual Stream View Is Useful

**Debugging:** if a model produces wrong outputs, you can inspect the stream at each layer to see where the wrong information was added or the right information was missed. "Which layer turned `Paris → London` in the stream?" is a well-posed question with this framing.

**Mechanistic interpretability:** researchers study which heads/MLPs write which features into the stream. Some heads reliably write "subject is a city" information; some MLPs reliably write "capitals of European countries" information. The residual stream is the communication channel through which all these components exchange information.

**Intuition for width vs depth:** width (`hidden_size`) determines how many features the stream can hold simultaneously. Depth (num layers) determines how many rounds of read-and-update the stream goes through. Wider streams can represent more simultaneously; deeper networks can build more complex features by composing simpler ones across layers. Both matter — a wide, shallow model can't build hierarchical representations; a deep, narrow model runs out of room to store intermediate features.

**Understanding pre-norm:** pre-norm (used in Llama) normalises the stream *before* reading it in each sublayer, but the raw unnormalised stream is what gets the delta added back to it. This means the stream accumulates in raw magnitude over depth — layer 32 reads a stream that may have 10× the magnitude of the embedding layer output. The final `RMSNorm` before the LM head re-normalises the accumulated stream before converting it to logits. The magnitude growth is a known property of pre-norm deep networks (see `PreNormVsPostNorm.md`).
