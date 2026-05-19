# Model Improvements from Architecture Enhancements

The enhancements don't all pull in the same direction — they target five distinct problems.

---

## 1. Training Speed

### Faster convergence per step

- **Muon optimizer** — orthogonalised gradients mean each step is maximally informative. No wasted steps on correlated directions. Empirically ~20–30% faster to reach the same loss.
- **Smear Gate** — bigram signal is free at layer 0. The model doesn't need to spend early training steps learning "what came before me?" — it can immediately focus on harder patterns.
- **resid_lambdas + x0_lambdas** — better gradient flow through the residual stream. Early layers don't overwrite the stream aggressively, so gradients propagate cleanly during backprop.

### Faster compute per step

- **ReLU²** — no `exp()` call unlike GELU/SiLU. Each MLP forward pass is cheaper.
- **bf16 activations** — 2× tensor core throughput on H100 for matmuls.
- **Flash Attention 3** — attention is no longer memory-bandwidth bound. The GPU compute units stop waiting for data transfers.

---

## 2. Model Quality at the Same Parameter Count

Same number of parameters, lower loss:

- **Untied embeddings** — `wte` and `lm_head` each specialise. Input encoding and output prediction are genuinely different jobs; sharing forced a compromise.
- **Value Embeddings** — deep layers retain token identity in the V signal. Without this, what "sat" retrieves when attending to "cat" is a blurry context-heavy blob rather than a clear identity signal.
- **Backout** — the LM head sees a cleaner signal. Low-level surface features (position, syntax) built up in the first half of the network get partially subtracted, leaving more high-level reasoning.
- **QK Norm** — prevents attention entropy collapse. The model continues to attend broadly rather than spiking all weight onto one token.
- **RoPE** — relative distance is explicit in the dot product. The model doesn't need to dedicate capacity to learning positional relationships from scratch.

---

## 3. Training Stability

Less likely to diverge or require careful hyperparameter tuning:

- **Logit softcapping** — as weights grow over long training runs, logits can explode. Softcapping bounds them to ±15, preventing overconfident gradients late in training.
- **QK Norm** — attention scores can't explode, so the softmax distribution stays well-behaved throughout training.
- **Explicit bf16** — unlike autocast, the precision boundary is deterministic. No surprise numerical issues from an op silently switching dtype.
- **resid_lambdas** — the residual stream is scaled before each block. Early layers can't accidentally overwrite the stream too aggressively.

---

## 4. Memory Efficiency

More can fit in GPU memory, enabling larger batch sizes or longer sequences:

- **GQA** — KV cache is proportional to `n_kv_head`, not `n_head`. If Q has 8× more heads than K,V, the KV cache shrinks 8×. Critical at inference.
- **Flash Attention** — attention uses O(n) memory instead of O(n²). For a 4096-token sequence, the difference is ~16M vs ~4K numbers in HBM.
- **Sliding Window** — most layers only attend to a local window. Memory and compute for those layers scales with window size, not full sequence length.
- **No biases, no RMSNorm γ** — small savings (~150K params for a 24-layer 1024-dim model), but removes parameters that earned nothing.

---

## 5. Generalisation to Longer Contexts

The model handles sequences longer than it was trained on more gracefully:

- **RoPE with base=100,000** — the slowest rotating dimension completes one cycle over a 100,000-token sequence. Positions interpolate smoothly beyond training length.
- **Sliding Window** — local patterns are learned on short windows. The model naturally composes them at longer contexts through the chain of layers.
- **GQA** — KV cache grows slower at inference. At 10K tokens, a GQA model with 8 KV heads uses 8× less cache than full MHA.

---

## Summary

| What you care about | Improvements |
|---|---|
| **Loss at same compute budget** | Lower — Muon, smear gate, value embeddings, untied heads, backout |
| **Training wall-clock time** | Faster — FA3, bf16, ReLU², Muon |
| **Training stability** | Better — logit softcap, QK norm, explicit precision |
| **Memory at inference** | Lower — GQA, sliding window, Flash Attention |
| **Long-context quality** | Better — RoPE base, sliding window, GQA cache |
| **Risk of training crash** | Lower — softcap, QK norm, resid scaling |

---

## One-Line Takeaway

> For the same number of parameters and the same training compute budget, these enhancements collectively produce a lower loss, a more stable training run, and a model that handles longer sequences more efficiently than a vanilla GPT-2.
