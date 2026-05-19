# GPT / Transformer Architecture — Enhancement Map

> Each numbered circle maps to the reference table below.

---

```
                    ┌─────────────────────────────────────────┐
                    │             Input Tokens                │
                    └─────────────────────────────────────────┘
                                        ↓
                    ┌─────────────────────────────────────────┐
                    │   ①  Token Embedding  (wte)             │
                    │   AdamW optimizer · fp32 weights        │
                    │   activations cast to bf16  [⑮]         │
                    └─────────────────────────────────────────┘
                                        ↓
                    ┌─────────────────────────────────────────┐
                    │   ②  RMSNorm  +  Smear Gate             │
                    │   Blends prev token embedding in        │
                    └─────────────────────────────────────────┘
                                        ↓
╔═════════════════════════════════════════════════════════════════╗
║                    × N  Transformer Layers                      ║
║                                                                  ║
║      ┌───────────────────────────────────────────────┐          ║
║      │   ③ ④   λ · x  +  α · x₀                    │          ║
║      │   resid_lambdas · x0_lambdas                  │          ║
║      │   both decay with depth                       │          ║
║      └───────────────────────────────────────────────┘          ║
║                              ↓                                   ║
║      ┌───────────────────────────────────────────────┐          ║
║      │           RMSNorm  (pre-norm)  [⑯]            │          ║
║      └───────────────────────────────────────────────┘          ║
║                              ↓                                   ║
║   ╔═══════════════════════════════════════════════╗             ║
║   ║  A T T E N T I O N                           ║             ║
║   ║                                               ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │  ⑤ ⑥   Q proj | K proj | V proj        │ ║             ║
║   ║  │  GQA: fewer K,V heads  ·  Muon optim.   │ ║             ║
║   ║  │  bias=False  [⑯]                         │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ║                        ↓                      ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │  ⑦   V = V + gate × raw_embed           │ ║             ║
║   ║  │  Value Embeddings  (alternating layers)  │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ║                        ↓                      ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │  ⑧   RoPE  — rotate Q, K by position    │ ║             ║
║   ║  │  ⑨   QK Norm  — RMSNorm on Q, K         │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ║                        ↓                      ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │  ⑩   Flash Attention 3                  │ ║             ║
║   ║  │  Tiled in SRAM · causal mask             │ ║             ║
║   ║  │  ⑪   Sliding window  (SSSL pattern)      │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ║                        ↓                      ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │       Output Projection  W_o             │ ║             ║
║   ║  │       bias=False  ·  Muon optimizer      │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ╚═══════════════════════════════════════════════╝             ║
║                              ↓                                   ║
║                       Residual  +                                ║
║                              ↓                                   ║
║      ┌───────────────────────────────────────────────┐          ║
║      │           RMSNorm  (pre-norm)  [⑯]            │          ║
║      └───────────────────────────────────────────────┘          ║
║                              ↓                                   ║
║   ╔═══════════════════════════════════════════════╗             ║
║   ║  M L P                                        ║             ║
║   ║                                               ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │      Linear expand   d → 4d              │ ║             ║
║   ║  │      bias=False  ·  Muon optimizer  [⑯]  │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ║                        ↓                      ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │  ⑫   ReLU²  activation                  │ ║             ║
║   ║  │  F.relu(x).square()                      │ ║             ║
║   ║  │  Sparse · amplified · cheaper than GELU  │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ║                        ↓                      ║             ║
║   ║  ┌─────────────────────────────────────────┐ ║             ║
║   ║  │      Linear shrink   4d → d              │ ║             ║
║   ║  │      bias=False  ·  Muon optimizer  [⑯]  │ ║             ║
║   ║  └─────────────────────────────────────────┘ ║             ║
║   ╚═══════════════════════════════════════════════╝             ║
║                              ↓                                   ║
║                       Residual  +                                ║
╚═════════════════════════════════════════════════════════════════╝
                                        ↓
                    ┌─────────────────────────────────────────┐
                    │   ⑬  Backout:  x  =  x  −  β · x_mid   │
                    │   backout_lambda · strips low-level feat │
                    └─────────────────────────────────────────┘
                                        ↓
                    ┌─────────────────────────────────────────┐
                    │            Final RMSNorm                │
                    └─────────────────────────────────────────┘
                                        ↓
                    ┌─────────────────────────────────────────┐
                    │   ①  LM Head  (untied from wte)         │
                    │   AdamW optimizer                       │
                    └─────────────────────────────────────────┘
                                        ↓
                    ┌─────────────────────────────────────────┐
                    │   ⑭  Logit Softcapping                  │
                    │   15 × tanh(logits / 15)                │
                    └─────────────────────────────────────────┘
                                        ↓
                    ┌─────────────────────────────────────────┐
                    │            Output Logits                │
                    └─────────────────────────────────────────┘
```

---

## Enhancement Reference

| # | Enhancement | Where it applies | What it does |
|---|---|---|---|
| ① | **AdamW Optimizer** | `wte`, `lm_head`, scalars | Adaptive learning rate for embeddings and scalar/gate params. `lm_head` is untied (separate matrix from `wte`). |
| ② | **Smear Gate** | Before transformer layers | Blends previous token's embedding into current token — cheap bigram signal in O(n), before any attention. |
| ③ | **resid_lambdas** | Before each block | λ scales the residual stream before the block's update is added. Starts ~1.15 at layer 0, decays to ~0.95. |
| ④ | **x0_lambdas** | Before each block | α blends the original token embedding (x₀) back in at every layer. Keeps the stream grounded in token identity. Starts ~0.20, decays to ~0.05. |
| ⑤ | **GQA** (Grouped Query Attention) | Q / K / V projections | Q has more heads than K and V. Multiple Q heads share one K,V head — reduces KV cache size at inference. |
| ⑥ | **Muon Optimizer** | All attention + MLP weights | Orthogonalises gradient updates via Newton-Schulz (5 iters). Each neuron moves in a unique direction — ~20–30% faster convergence than AdamW for matrix params. |
| ⑦ | **Value Embeddings** | V vector (alternating layers) | Raw token embedding injected into V: `V = V + gate × raw_embed`. Gate range (0, 3). Keeps token identity clear when attended to. |
| ⑧ | **RoPE** | Q, K vectors | Rotates Q and K by position-dependent angles. Relative distance falls out naturally in the dot product. `base = 100,000`. |
| ⑨ | **QK Norm** | Q, K after RoPE | RMSNorm on Q and K prevents dot products from exploding → avoids attention entropy collapse. Scale ×1.2 after norm. |
| ⑩ | **Flash Attention 3** | Attention compute | Tiles Q·Kᵀ in fast on-chip SRAM — never writes full n×n matrix to HBM. 2–10× less memory bandwidth on H100. |
| ⑪ | **Sliding Window Attention** | Flash Attention mask | SSSL: most layers attend only to a local window (~n/4). Final layer always attends full context. |
| ⑫ | **ReLU² Activation** | MLP middle step | `F.relu(x).square()` — kills negatives, squares positives. Higher sparsity, stronger gradient for active neurons, no `exp` needed. |
| ⑬ | **backout_lambda** | After all layers | Subtracts cached mid-layer residual: `x = x − β × x_mid`. Strips low-level features so LM head sees high-level reasoning. |
| ⑭ | **Logit Softcapping** | After LM Head | `15 × tanh(logits/15)` — bounds logits to ±15. Prevents overconfident softmax spikes. Smooth squeeze, gradients always flow. |

---

## Cross-Cutting Concerns

| Enhancement | Applies to | What it does |
|---|---|---|
| ⑮ **Precision Management** | Entire forward pass | Weights stored in `fp32` so tiny gradient updates aren't lost. Activations cast to `bf16` for fast tensor core matmuls (2× throughput on H100). Custom `Linear` class handles the cast — no `autocast`. |
| ⑯ **Lean Design** | All `Linear` layers + all `RMSNorm` | `bias=False` everywhere. RMSNorm has no learnable γ (`F.rms_norm` only). Removes redundant params — downstream weight matrices already handle scale and offset. |
