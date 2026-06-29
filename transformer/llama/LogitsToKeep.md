# `logits_to_keep` — the LM Head Optimisation

During generation, computing the full `[B, S, vocab_size]` logit tensor is massively wasteful — only the *last* token's logits are needed to decide what to generate next. `logits_to_keep` is the mechanism Llama uses to avoid that waste.

---

## Where It Lives

In `LlamaForCausalLM.forward`:

```python
def forward(
    self,
    ...
    logits_to_keep: int | torch.Tensor = 0,
    ...
):
    hidden_states = outputs[0]   # [B, S, hidden_size]

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
```

`self.lm_head` is `Linear(hidden_size → vocab_size, bias=False)`.

---

## What It Does

`logits_to_keep` controls **how many trailing token positions** get projected to vocabulary logits.

| `logits_to_keep` | `slice_indices` | `hidden_states[:, slice_indices, :]` | Logits shape |
|---|---|---|---|
| `0` | `slice(0, None)` = all | `[B, S, H]` — all tokens | `[B, S, vocab_size]` |
| `1` | `slice(-1, None)` | `[B, 1, H]` — last token only | `[B, 1, vocab_size]` |
| `5` | `slice(-5, None)` | `[B, 5, H]` — last 5 tokens | `[B, 5, vocab_size]` |

**Default is `0`** (all tokens). `GenerationMixin` sets it to `1` during generation steps, so only the last token's hidden state is passed through the LM head.

---

## Why This Matters

The LM head is `Linear(hidden_size → vocab_size)`:
- Llama3 8B: `Linear(4096 → 128256)` — weight matrix is 4096 × 128256 ≈ **524M parameters**, ~1GB in bfloat16
- Llama4: `Linear(5120 → 202048)` — weight matrix is 5120 × 202048 ≈ **1.03B parameters**, ~2GB in bfloat16

The LM head is one of the largest single operations in the model. Computing it for all S tokens means:

```
S=512 tokens, Llama3 8B:
  full:  [B, 512, 4096] @ [4096, 128256]  →  [B, 512, 128256]
  with logits_to_keep=1:  [B, 1, 4096] @ [4096, 128256]  →  [B, 1, 128256]
  savings: 512× fewer FLOPs for this one matmul
```

At `S=2048`, skipping 2047 unnecessary projections is a significant saving — especially since the LM head matmul is bandwidth-bound (the weight matrix barely fits in GPU memory, so loading it once per step is the bottleneck, not arithmetic).

---

## Training vs Inference

**Training:** `logits_to_keep=0` (all tokens). During training, the loss is computed over all positions simultaneously (`labels` covers the full sequence), so all S token logits are needed. Skipping any would mean missing the gradient signal from those positions.

**Inference / generation:** `logits_to_keep=1`. The generation loop only ever samples from the last position's logits — positions `0..S-2` have already been committed to, and their logits would just be thrown away. `GenerationMixin` sets this automatically.

**Speculative decoding or beam search with lookahead:** `logits_to_keep=K` for K > 1. Some generation strategies need to see the last few token logits simultaneously (e.g. to verify or score K candidate continuations). The parameter accommodates this without falling back to full-sequence projection.

---

## The Tensor Slice Mechanics

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

`slice(-1, None)` in Python indexing means "from the last element to the end" — i.e. just the last element. `hidden_states[:, -1:, :]` keeps the last token across all batches while preserving the sequence dimension (shape `[B, 1, H]` not `[B, H]`), so the subsequent matmul and output shape are consistent regardless of `logits_to_keep`.

When `logits_to_keep` is a `torch.Tensor` (e.g. an index tensor for speculative decoding), it's passed directly as the slice index — this allows non-contiguous or non-trailing position selection.

---

## Relationship to KV Cache

`logits_to_keep=1` and the KV cache work together:
- The KV cache means only the new token's hidden states are computed in each generation step (`[B, 1, H]` flows through the entire decoder stack)
- `logits_to_keep=1` means only that one token's hidden state is projected to logits

So in steady-state generation, the entire path is `[B, 1, *]` throughout — O(1) compute per step regardless of how many tokens have already been generated.
