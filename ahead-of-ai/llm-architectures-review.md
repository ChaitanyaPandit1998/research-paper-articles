# Review: "Recent Developments in LLM Architectures" by Sebastian Raschka

> Source: https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures

## What Is This Article About?

Sebastian Raschka surveys the most important architectural changes happening inside modern large language models (LLMs) as of April–May 2026. The central theme is simple: **AI models are being used in longer and longer conversations** (for reasoning tasks, agents, etc.), and that is expensive. So researchers are redesigning the internal machinery of these models to handle long contexts more cheaply.

---

## The Core Problem Being Solved

When a model processes text, it needs to store "memory" of what it has seen — this is called the **KV (Key-Value) cache**. The longer the conversation, the bigger this cache gets, and the more memory and compute it consumes. At 1 million tokens (think: a full novel), this becomes a serious bottleneck.

All the innovations in this article are essentially different ways of asking: *Can we be smarter about what we store and how we process it?*

---

## The Six Key Innovations

### 1. KV Sharing Across Layers (Gemma 4)

**What it is:** Instead of every layer computing its own memory, later layers borrow the memory computed by earlier layers.

**Analogy:** Imagine a relay race where only the first few runners carry the baton; the rest just hand it along without picking up a new one.

**Result:** Google's Gemma 4 saves ~2.7 GB of memory at long contexts with almost no quality loss. The trade-off is slightly less flexibility in how later layers attend to information.

**Diagram:**

```
Standard Transformer (every layer computes its own KV):

  Layer 1  │ Q  K  V │ ──► Attention
  Layer 2  │ Q  K  V │ ──► Attention
  Layer 3  │ Q  K  V │ ──► Attention
     ...        ...
  Layer N  │ Q  K  V │ ──► Attention
                ↑
           (each layer stores its own KV in cache = large memory)


Gemma 4 KV Sharing (only early layers compute KV):

  Layer 1   │ Q  K  V │ ──► Attention  ← computes & stores KV
  Layer 2   │ Q  K  V │ ──► Attention  ← computes & stores KV
     ...
  Layer 15  │ Q  K  V │ ──► Attention  ← last layer to compute KV
             └────┬────┘
                  │  KV values shared downward
                  ▼
  Layer 16  │ Q  ·  · │ ──► Attention  ← borrows KV from Layer 15
  Layer 17  │ Q  ·  · │ ──► Attention  ← borrows KV from Layer 15
     ...
  Layer 35  │ Q  ·  · │ ──► Attention  ← borrows KV from Layer 15

  Result: only 15 KV caches stored instead of 35  →  ~2.7 GB saved
```

---

### 2. Per-Layer Embeddings (Gemma 4)

**What it is:** Each layer of the model gets its own small "flavor" of the input token — a tiny extra signal that helps the layer specialize.

**Analogy:** Think of reading the same sentence but putting on different colored glasses for each chapter of analysis — each pair slightly changes what stands out.

**Result:** Gemma 4's E2B model achieves the active compute of a 2.3B parameter model while storing 5.1B parameters — a clever way to pack in knowledge without paying full compute costs every step.

**Diagram:**

```
For each token at each layer:

  Token ID
     │
     ├──────────────────────────────────────────► Standard Token Embedding
     │                                                      │
     └──► PLE Table Lookup                                  │
               │                                            │
               ▼                                            │
     ┌─────────────────────┐                                │
     │  Layer-1 Slice      │──► learned gate ──► residual ──┤
     └─────────────────────┘                                │
                                                            ▼
                                                    Layer 1 Input
                                                    (token embedding
                                                    + layer flavor)

  (Same process repeated — different PLE slice — for every layer)

  Layer 1  ──► its own PLE slice  ──► unique "color" for this layer
  Layer 2  ──► its own PLE slice  ──► unique "color" for this layer
     ...
  Layer N  ──► its own PLE slice  ──► unique "color" for this layer

  Effect: 5.1B total parameters, but only 2.3B active during compute
```

---

### 3. Layer-Wise Attention Budgeting (Laguna XS.2)

**What it is:** Not all layers need the same amount of attention. Some layers do "local" processing (nearby tokens only), others do "global" processing (all tokens). This technique gives each type a different budget.

**Analogy:** In a large company, analysts handle day-to-day details locally; executives only meet to make big-picture decisions. You don't give every meeting the same agenda length.

**Result:** Expensive global-attention layers get fewer query heads (saving cost); cheaper local-attention layers get more. This balances quality and efficiency without a uniform approach.

**Diagram:**

```
Laguna XS.2 — 40 layers total, two types:

  ┌──────────────────────────────────────────────────────────────┐
  │  GLOBAL ATTENTION LAYER  (10 layers, e.g. layer 4, 8, ...)  │
  │                                                              │
  │  Sees ALL tokens in sequence                                 │
  │  8 KV heads   ×   6 Query heads each  =  48 total queries   │
  │  (expensive to run → fewer query heads to save cost)         │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  SLIDING-WINDOW LAYER  (30 layers, most of the stack)        │
  │                                                              │
  │  Sees only the nearest 512 tokens                            │
  │  8 KV heads   ×   8 Query heads each  =  64 total queries   │
  │  (cheap to run → more query heads are affordable)            │
  └──────────────────────────────────────────────────────────────┘

  Layer stack (simplified):

  [ SW ][ SW ][ SW ][ G ][ SW ][ SW ][ SW ][ G ] ...
   local local local GLOBAL local local local GLOBAL

  SW = sliding window (512-token window, 8 Q/KV head ratio)
  G  = global attention (full sequence, 6 Q/KV head ratio)
```

---

### 4. Compressed Convolutional Attention (ZAYA1-8B)

**What it is:** Instead of attending over all tokens at full size, this method compresses the queries, keys, and values first — then runs attention in that smaller space. It also mixes in local context using convolutions (a technique borrowed from image processing).

**Analogy:** Instead of comparing every word in a document to every other word, you first summarize paragraphs, then compare summaries.

**Result:** Reduces both the memory cache AND the compute cost during training and inference — more aggressive than methods that only compress the stored cache.

**Diagram:**

```
Standard Attention (operates at full token resolution):

  Input tokens  [T1] [T2] [T3] ... [TN]   (full size)
                  │    │    │         │
                  Q    K    V         │    (full-size projections)
                  └────┴────┘         │
                       │              │
                  Attention ──────────┘
                       │
                    Output


Compressed Convolutional Attention (ZAYA1-8B):

  Input tokens  [T1] [T2] [T3] ... [TN]
                  │
                  ▼
           ┌─── Compress ───┐          (reduce to smaller latent space)
           │                │
          [q1] [q2]...    [k1][k2]...  [v1][v2]...
           │                │
           ▼                ▼
     Channel-mix       Channel-mix     (convolution: adds local context)
     Conv on Q         Conv on K
           │                │
           └───────┬─────────┘
                   ▼
             Attention in                (attend in compressed space)
            compressed space             cheaper: fewer tokens × smaller dim
                   │
                   ▼
             Up-project                  (expand back to full size)
                   │
                   ▼
                Output

  Savings: smaller KV cache  +  fewer attention FLOPs
```

---

### 5. Manifold-Constrained Hyper-Connections (DeepSeek V4)

**What it is:** Normal transformers have a single "highway" (residual stream) that carries information layer-to-layer. This technique adds multiple parallel highways with controlled mixing between them. The "manifold constraint" (doubly stochastic matrix) ensures information flows stably rather than chaotically.

**Analogy:** Instead of one pipe carrying water through a building, you have four pipes that can share water with each other — but the valves are designed so pressure stays balanced.

**Result:** ~6.7% training overhead but models reach the same performance in roughly half the training tokens — a significant efficiency gain.

**Diagram:**

```
Standard Residual Stream (one highway):

  Input
    │
    ├────────────────────────────────────────────────────────────┐
    │                                                            │
    ▼                                                            │
  [ Attention / FFN Layer ]                                      │
    │                                                            │
    └──────────────────────────────────────────────► + ──► Output


Hyper-Connections with Manifold Constraint (4 parallel streams):

  Input
    │
    ├──► Stream 1 ──┐
    ├──► Stream 2 ──┤
    ├──► Stream 3 ──┤
    └──► Stream 4 ──┘
              │
              ▼
    ┌─────────────────────┐
    │  Doubly Stochastic  │  ← each row and column sums to 1
    │  Mixing Matrix      │    (stable, balanced redistribution)
    └─────────────────────┘
              │
    ┌─────────────────────┐
    │ Stream 1'           │
    │ Stream 2'  (mixed)  │
    │ Stream 3'           │
    │ Stream 4'           │
    └─────────────────────┘
              │
    [ Attention / FFN Layer ]
              │
    ┌─────────────────────┐
    │  Mix again & merge  │
    └─────────────────────┘
              │
            Output

  Benefit: information redistribution is stable (no exploding/vanishing)
           same quality as baseline at ~half the training tokens
```

---

### 6. Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) (DeepSeek V4)

**What it is:** Instead of compressing each token's representation (like MLA does), these techniques compress along the sequence — meaning many tokens get merged into fewer cached entries.

- **CSA** (moderate): Every 4 tokens merge into 1. Uses smart sparse selection to retain the most important ones.
- **HCA** (aggressive): Every 128 tokens merge into 1. Extremely cheap but less detailed.

DeepSeek V4 uses both side-by-side, always keeping a small window of recent uncompressed tokens for fine-grained local attention.

**Analogy:** CSA is like keeping highlights every paragraph; HCA is like keeping only chapter summaries. Both are used together, with the last page always kept in full.

**Result (at 1-million-token context vs. DeepSeek V3.2):**
- V4-Pro: only **27% of the compute** and **10% of KV cache**
- V4-Flash: only **10% of the compute** and **7% of KV cache**

**Diagram:**

```
Full sequence of 1,000 tokens (each box = 1 token):

  [T1][T2][T3][T4][T5][T6][T7][T8]...[T869]...[T997][T998][T999][T1000]


CSA — Compressed Sparse Attention  (m = 4, mild compression):

  Every 4 tokens → 1 compressed entry (keep most important via sparse select)

  [  C1  ][  C2  ][  C3  ] ... [  C249  ] │ [T997][T998][T999][T1000]
   4→1      4→1     4→1                      └─── recent window (uncompressed)

  Cache size: 249 compressed + 4 recent  vs.  1000 original tokens


HCA — Heavily Compressed Attention  (m = 128, aggressive compression):

  Every 128 tokens → 1 compressed entry (dense merge, very cheap)

  [        H1        ][        H2        ] ... [H7] │ [T997..T1000]
   128→1               128→1                         └── recent window

  Cache size: 7 compressed + 4 recent  vs.  1000 original tokens


DeepSeek V4 uses ALL THREE in parallel for each attention layer:

  Full Sequence
       │
       ├──► CSA Branch      ──► attention over ~250 entries
       │
       ├──► HCA Branch      ──► attention over ~7 entries
       │
       └──► Sliding Window  ──► attention over last 128 tokens (full detail)
                │
                └──► Combine all three outputs  ──► final attention result

  Net effect vs. DeepSeek V3.2 at 1M-token context:
  V4-Pro:   27% compute,  10% KV cache
  V4-Flash: 10% compute,   7% KV cache
```

---

## The Big Picture

| Innovation | What It Reduces | Model |
|---|---|---|
| KV Sharing | KV cache size | Gemma 4 |
| Per-Layer Embeddings | Compute vs. parameter count | Gemma 4 |
| Attention Budgeting | Per-layer attention cost | Laguna XS.2 |
| Convolutional Attention | Cache + compute | ZAYA1-8B |
| Hyper-Connections | Training token cost | DeepSeek V4 |
| CSA / HCA | Sequence cache + FLOPs | DeepSeek V4 |

---

## An Honest Warning From the Author

Raschka notes that a basic transformer used to take ~50–100 lines of PyTorch code. Modern variants with all these tricks take **10x more code**. Each piece is understandable on its own, but putting them together is increasingly complex. This has implications for:

- **Reproducibility** — harder for outside researchers to replicate
- **Debugging** — more moving parts means more places things can go wrong
- **Accessibility** — the gap between frontier labs and academic researchers is widening

---

## What Hasn't Changed

The **transformer architecture itself** remains dominant. There's no revolution here — just intelligent engineering. Progress is coming from:

1. Smarter use of existing compute and memory
2. Better training data and recipes
3. Specialized designs for long-context scenarios

The evolution from GPT-2 (2019) to DeepSeek V4 (2026) is continuous refinement, not replacement.

---

## Overall Assessment

**Strengths of the article:**
- Technically precise without being impenetrable
- Covers a wide range of models and labs (Google, DeepSeek, smaller players)
- Frames each technique around a real problem it solves

**Who should read it:**
- ML engineers curious about what's inside frontier models
- Researchers thinking about long-context or efficient inference
- Anyone building on top of these models and wondering why costs differ

**One-sentence summary:** Modern LLMs are getting dramatically better at long contexts not by changing the fundamental transformer idea, but by being increasingly clever about what they store, what they compute, and how information flows between layers.
