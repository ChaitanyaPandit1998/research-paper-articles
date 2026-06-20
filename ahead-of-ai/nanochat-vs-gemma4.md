# nanochat vs Gemma 4 — Architecture Comparison

> nanochat source: `/Users/chaitanya/Development/AI/nanochat/nanochat/gpt.py`
> Gemma 4 reference: Sebastian Raschka's LLM Architecture Review (May 2026)

---

## Quick Specs

| Property | nanochat | Gemma 3/4 (270M) |
|---|---|---|
| Layers | 12 | 18 |
| Embedding dim | 768 | 640 |
| Query heads | 6 | 4 |
| KV heads | 6 (configurable) | 1 (multi-query) |
| MLP expansion | 4× (→ 3072) | 3.2× (→ 2048) |
| Context length | 2048 | 128K |
| Vocab size | 32,768 | 262,144 |
| Sliding window ratio | 3:1 (SSSL) | 5:1 (local:global) |
| Positional encoding | RoPE (base 100K) | RoPE |
| MLP activation | ReLU² | GeGLU |
| Normalization | Pre-RMSNorm only | Pre + Post RMSNorm (sandwich) |
| Logit softcapping | Yes (15·tanh) | Yes (Gemma 4) |
| Untied embeddings | Yes | Yes |

---

## Feature-by-Feature Comparison

### 1. Normalization Strategy

**nanochat:** Pre-norm only. A single `RMSNorm` before each attention and MLP block, with no learnable scale/shift parameters.

```
x → RMSNorm → Attention → + residual
x → RMSNorm → MLP       → + residual
```

**Gemma 4:** Sandwich norm — both Pre-RMSNorm (before the block) and Post-RMSNorm (after the block, before the residual add).

```
x → Pre-RMSNorm → Attention → Post-RMSNorm → + residual
x → Pre-RMSNorm → MLP       → Post-RMSNorm → + residual
```

**Verdict:** Gemma's sandwich norm provides more training stability at scale. nanochat's approach is simpler and faster. For small-scale experiments, pre-norm alone works fine; at 10B+ parameters, the extra norm matters.

---

### 2. Sliding Window Attention Pattern

**nanochat:** `SSSL` pattern — 3 short-context layers (quarter of sequence = ~512 tokens) followed by 1 full-context layer. The final layer always uses full context.

```
Layer 1  → SHORT (512 tokens)
Layer 2  → SHORT (512 tokens)
Layer 3  → SHORT (512 tokens)
Layer 4  → LONG  (2048 tokens, full)
... (pattern repeats)
Layer 12 → LONG  (always full, hardcoded)
```

**Gemma 4:** `5:1` ratio — 5 local sliding-window layers for every 1 global (full-context) layer.

```
[ SW ][ SW ][ SW ][ SW ][ SW ][ G ] ...
```

**Verdict:** Both use the same philosophy. nanochat's 3:1 ratio is slightly more aggressive with global attention; Gemma's 5:1 gives fewer global layers, saving more compute. At nanochat's small context (2048), this barely matters. At Gemma's 128K context, getting that ratio right is critical.

---

### 3. Group-Query Attention (GQA)

**nanochat:** Configurable. Default is `n_head=6, n_kv_head=6` — effectively standard multi-head attention (MHA). Can be set to fewer KV heads.

**Gemma 4:** `4 query heads, 1 KV head` — this is Multi-Query Attention (MQA), the most aggressive form of GQA. Every query head shares the same single K and V.

```
nanochat (MHA):    Q1 Q2 Q3 Q4 Q5 Q6
                   K1 K2 K3 K4 K5 K6   ← one K per Q head

Gemma 4 (MQA):     Q1 Q2 Q3 Q4
                          K1            ← one K shared by ALL query heads
                          V1
```

**Verdict:** nanochat leaves significant KV cache savings on the table by defaulting to 6 KV heads. Switching to `n_kv_head=1` or `n_kv_head=2` is a free win. Gemma's MQA is optimal for long contexts where KV cache size dominates memory.

---

### 4. MLP Activation Function

**nanochat:** `ReLU²` — applies ReLU then squares the result. Simple, sparse, fast.

```python
x = F.relu(x).square()   # nanochat MLP
```

**Gemma 4:** `GeGLU` — a gated linear unit variant where one branch uses GeLU activation and is element-wise multiplied with a second branch.

```
Linear(x) → split into [A, B]
output = GeLU(A) × B       # Gemma MLP
```

**Verdict:** GeGLU is generally considered superior to ReLU² — the gating mechanism allows the MLP to be more selective. However, GeGLU requires 3 weight matrices (gate + up + down) vs. nanochat's 2, making it ~50% more parameter-heavy in the MLP. ReLU² is faster and produces naturally sparse activations, which can be beneficial. This is a legitimate trade-off, not a clear winner.

---

### 5. KV Sharing Across Layers *(Gemma 4 exclusive)*

**nanochat:** Every layer computes its own fresh K and V projections.

**Gemma 4:** Later layers reuse K and V from an earlier layer (e.g., layers 16–35 reuse the KV from layer 15 in the 35-layer E2B model).

```
nanochat:         Gemma 4 E2B:

Layer 1  K V      Layer 1   K V  ← computed
Layer 2  K V      Layer 2   K V  ← computed
...               ...
Layer 12 K V      Layer 15  K V  ← last to compute its own KV
                            │
                  Layer 16  K─┘  ← shared (only Q is new)
                  Layer 17  K─┘  ← shared
                  ...
                  Layer 35  K─┘  ← shared
```

**Verdict:** nanochat could adopt this easily. At a 2048 context window it saves very little. But if nanochat ever scales to 32K+ contexts, adding KV sharing for the top half of layers would be a near-free memory win.

---

### 6. Per-Layer Embeddings *(Gemma 4 exclusive)*

**Gemma 4:** Each transformer layer receives a layer-specific embedding slice (looked up from a PLE table using the token ID) that is gated and added to the hidden state.

```
Token ID → PLE Table → Layer-N Slice → learned gate → added to hidden state
```

**nanochat (closest analog — Value Embeddings):** nanochat has `value_embeds` — separate embedding tables injected into the V (value) projection of alternating attention layers.

```python
# nanochat value embeddings (ResFormer-style)
ve = self.value_embeds[str(i)](idx)          # look up token embedding
gate = 3 * torch.sigmoid(self.ve_gate(...))  # per-head gate
v = v + gate.unsqueeze(-1) * ve             # add to V vectors
```

**Verdict:** These are conceptually similar — both inject token identity information deep into the network via embedding lookups. nanochat's approach is more targeted (only affects V in attention, only in alternating layers). Gemma 4's PLE is more general (affects the full hidden state, every layer). The PLE's payoff is the parameter count decoupling (5.1B stored, 2.3B active). nanochat's value embeddings don't achieve that but do improve output quality.

---

### 7. nanochat-Exclusive Features *(not in Gemma 4)*

These are things nanochat has that Gemma 4 does not:

#### a. Smear Gate (cheap bigram signal)

Mixes the previous token's embedding into the current token's representation before the transformer layers begin.

```python
gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]])
```

This is a cheap way to give each token a "hint" about what came before, without full causal attention. Gemma has no equivalent.

#### b. Residual Stream Scaling (resid_lambdas + x0_lambdas)

Per-layer learned scalars that:
- `resid_lambdas[i]`: scale the residual stream before each block (initialized stronger at early layers, weaker at deep layers)
- `x0_lambdas[i]`: blend the original input embedding back in at each layer (initialized to decay with depth)

```python
x = resid_lambdas[i] * x + x0_lambdas[i] * x0   # before each block
```

This gives the optimizer fine-grained control over how strongly each layer contributes. Gemma 4 uses fixed residual adds.

#### c. Backout Subtraction

Saves the hidden state at the halfway point of the network, then subtracts a fraction of it from the final hidden state before the output head.

```python
x_backout = x  # cached at layer 6 (of 12)
...
x = x - backout_lambda * x_backout   # before final norm
```

The idea: early layers capture low-level surface features; subtracting a fraction of the mid-point representation encourages the final output to focus on higher-level abstractions. Gemma has no equivalent.

#### d. Muon Optimizer for Matrix Parameters

nanochat uses the Muon optimizer (a Nesterov momentum optimizer with Newton-Schulz orthogonalization) for all weight matrices, and AdamW for embeddings and scalars.

Gemma uses standard AdamW throughout. Muon often converges faster and to better minima for transformer weight matrices.

---

## Summary Scorecard

| Feature | nanochat | Gemma 4 | Winner |
|---|---|---|---|
| Sandwich normalization | No | Yes | Gemma 4 (at scale) |
| Sliding window attention | SSSL 3:1 | 5:1 | Gemma 4 (longer context) |
| GQA aggressiveness | MHA (6:6) | MQA (4:1) | Gemma 4 (KV cache) |
| MLP activation | ReLU² | GeGLU | Gemma 4 (quality) |
| KV sharing | No | Yes | Gemma 4 |
| Per-layer embeddings | No (value embeds only) | Yes | Gemma 4 |
| Smear gate | Yes | No | nanochat |
| Residual stream scaling | Yes | No | nanochat |
| Backout subtraction | Yes | No | nanochat |
| Muon optimizer | Yes | No | nanochat |
| Value embeddings | Yes | No | nanochat |
| QK Norm | Yes | Yes | Tie |
| RoPE | Yes | Yes | Tie |
| Logit softcapping | Yes | Yes | Tie |
| Untied embeddings | Yes | Yes | Tie |

---

## What nanochat Should Borrow from Gemma 4

1. **Reduce KV heads** — change `n_kv_head` from 6 to 1 or 2. Zero-cost memory saving at inference.
2. **Switch to GeGLU** — replace the 2-matrix `ReLU²` MLP with a 3-matrix GeGLU. Quality improvement, small parameter cost.
3. **Try sandwich norm** — add a Post-RMSNorm after each attention/MLP block. More stable at longer training runs.
4. **KV sharing** — if ever scaling context beyond 32K, share KV from a cutoff layer downward.

## What Gemma 4 Could Borrow from nanochat

1. **Muon optimizer** — faster convergence on matrix parameters.
2. **x0 blending + resid_lambdas** — fine-grained per-layer residual control.
3. **Backout subtraction** — encourages higher-level final representations.
4. **Smear gate** — trivially cheap bigram signal.

---

## Architecture Diagram Side-by-Side

```
nanochat (12 layers)                 Gemma 4 (18 layers, simplified)
─────────────────────────            ──────────────────────────────────
Token IDs                            Token IDs
    │                                    │
    ▼                                    ▼
 wte (32K vocab)                      wte (262K vocab)
    │                                    │
  norm(x)                              (no pre-embedding norm)
    │                                    │
 Smear Gate                          ── ── (no smear gate)
    │                                    │
    ├── save as x0                    ── ── (no x0 blending)
    │                                    │
 ┌──────── × 12 layers ─────────┐    ┌──────── × 18 layers ──────────┐
 │                               │    │                                │
 │  x = resid_λ·x + x0_λ·x0    │    │  (fixed residual add)          │
 │                               │    │                                │
 │  ┌── Pre-RMSNorm             │    │  ┌── Pre-RMSNorm               │
 │  │                           │    │  │                             │
 │  │  GQA (6Q / 6KV)          │    │  │  GQA (4Q / 1KV = MQA)      │
 │  │  + Value Embeddings       │    │  │  + PLE slice gating         │
 │  │    (alternating layers)   │    │  │    (every layer)            │
 │  │  + RoPE on Q,K            │    │  │  + RoPE on Q,K              │
 │  │  + QK Norm                │    │  │  + QK Norm                  │
 │  │  + SWA (SSSL pattern)     │    │  │  + SWA (5:1 pattern)        │
 │  │  + Flash Attn 3           │    │  │  + KV sharing (later layers)│
 │  └─────────────────          │    │  └─────────────────            │
 │  + residual                  │    │  ┌── Post-RMSNorm              │
 │                              │    │  + residual                    │
 │  ┌── Pre-RMSNorm            │    │                                │
 │  │                          │    │  ┌── Pre-RMSNorm               │
 │  │  MLP: 4×, ReLU²          │    │  │                             │
 │  └─────────────             │    │  │  MLP: 3.2×, GeGLU           │
 │  + residual                  │    │  └─────────────────            │
 │                              │    │  ┌── Post-RMSNorm              │
 │  [cache at layer 6]          │    │  + residual                    │
 └──────────────────────────────┘    └────────────────────────────────┘
    │                                    │
 x -= backout_λ · x_mid              (no backout)
    │                                    │
 Final RMSNorm                        Final RMSNorm
    │                                    │
 lm_head (untied)                     lm_head (untied)
    │                                    │
 15·tanh(logits/15)                   15·tanh(logits/15)
    │                                    │
 Output logits                         Output logits
```
