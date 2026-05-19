# GPT-2 vs This Model — Architecture Comparison

Both follow the same skeleton: embed → N transformer blocks → LM head.
The differences are in almost every detail inside that skeleton.

---

## Component-by-Component

| Component | GPT-2 (original) | This model |
|---|---|---|
| **Position encoding** | Learned absolute embeddings `wpe` — one vector per slot | RoPE — rotate Q,K by position, no separate embedding |
| **Attention heads** | Full MHA — same number of Q, K, V heads | GQA — fewer K,V heads, shared across Q heads |
| **Q,K stability** | Nothing — dot products can grow | QK Norm — RMSNorm on Q and K after rotation |
| **Attention compute** | Standard O(n²) | Flash Attention 3 — tiled in SRAM |
| **Attention window** | Full context every layer | Sliding window (SSSL) — local on most layers, full on last |
| **MLP activation** | GELU | ReLU² — `F.relu(x).square()` |
| **Normalization** | LayerNorm with learnable γ and β | RMSNorm with no learnable params |
| **Biases** | Yes — all linear layers have bias | No — `bias=False` everywhere |
| **Residual stream** | Plain `x = x + sublayer(x)` | Scaled: `λ×x + α×x₀ + sublayer(x)` |
| **Token identity** | Fades with depth, nothing done about it | Value Embeddings + x0_lambdas fight drift |
| **Bigram signal** | Layer 0 attention must learn it | Smear Gate injects it before layer 0 |
| **Output cleanup** | Nothing | Backout subtracts mid-layer residual |
| **Logit range** | Unbounded | Softcapped to ±15 |
| **wte / lm_head** | Tied (same matrix) | Untied (separate matrices) |
| **Optimizer** | Adam (one optimizer for all) | Muon for weight matrices, AdamW for everything else |
| **Precision** | fp32 or autocast | Explicit fp32 weights, bf16 activations |

---

## Side-by-Side Flow

```
GPT-2                              This model
──────────────────────             ──────────────────────────────
Input Tokens                       Input Tokens
↓                                  ↓
Token Embedding (wte)              Token Embedding (wte)
+ Position Embedding (wpe)         ↓
↓                                  RMSNorm + Smear Gate  ← NEW
                                   ↓
╔══ × N Layers ══╗                ╔══ × N Layers ══════════════╗
║ LayerNorm      ║                ║ λ·x + α·x₀  ← NEW         ║
║ ↓              ║                ║ ↓                           ║
║ MHA            ║                ║ RMSNorm  (no γ)             ║
║ (all heads)    ║                ║ ↓                           ║
║ ↓              ║                ║ GQA + Value Embed  ← NEW    ║
║ Residual +     ║                ║ RoPE + QK Norm  ← NEW       ║
║ LayerNorm      ║                ║ Flash Attn + Window  ← NEW  ║
║ ↓              ║                ║ ↓                           ║
║ MLP  (GELU)    ║                ║ Residual +                  ║
║ ↓              ║                ║ RMSNorm  (no γ)             ║
║ Residual +     ║                ║ ↓                           ║
╚════════════════╝                ║ MLP  (ReLU²)  ← NEW         ║
↓                                 ║ ↓                           ║
LayerNorm                         ║ Residual +                  ║
↓                                 ╚═════════════════════════════╝
LM Head  (tied to wte)            ↓
↓                                 Backout subtraction  ← NEW
Output Logits                     ↓
                                  Final RMSNorm
                                  ↓
                                  LM Head  (untied)  ← NEW
                                  ↓
                                  Logit Softcapping  ← NEW
                                  ↓
                                  Output Logits
```

---

## What Each Addition Fixes

| Addition | Problem it solves in GPT-2 |
|---|---|
| **RoPE** | Learned `wpe` can't generalise beyond training length; relative distance isn't explicit |
| **GQA** | Full MHA replicates K,V heads wastefully — KV cache grows with every head |
| **QK Norm** | Dot products grow unboundedly → attention entropy collapse (all weight on 1 token) |
| **Flash Attention 3** | Standard attention writes full n×n matrix to slow HBM — memory bandwidth bottleneck |
| **Sliding Window** | Every layer attending full context wastes compute on long-range patterns early on |
| **ReLU²** | GELU needs expensive `exp()` calls; less sparse than ReLU² |
| **RMSNorm (no γ)** | LayerNorm's γ/β are redundant — downstream weight matrix already handles scale |
| **No biases** | Biases are redundant in deep networks with normalisation after each layer |
| **resid_lambdas** | All layers have equal influence — no way to control early vs late layer contribution |
| **x0_lambdas** | Token identity fades with depth as context accumulates |
| **Value Embeddings** | V vectors drift from original token — other tokens lose clear identity signal when attending |
| **Smear Gate** | Layer 0 wastes attention capacity rediscovering "what came immediately before me?" |
| **Backout** | Final residual contains low-level noise (syntax, position) mixed into high-level output |
| **Logit softcapping** | Unbounded logits → overconfident softmax → exploding gradients late in training |
| **Untied embeddings** | Shared wte/lm_head forces input encoding and output prediction to compromise |
| **Muon optimizer** | AdamW wastes steps on correlated gradient directions in weight matrices |
| **Explicit bf16** | Autocast is unpredictable; explicit casting gives full control over where precision matters |

---

## One-Line Summary

> GPT-2 is a clean minimal baseline. This model keeps the same skeleton but patches every known weakness — positional generalisation, attention instability, memory efficiency, residual drift, training stability, and compute cost — each with a targeted, well-understood fix.
