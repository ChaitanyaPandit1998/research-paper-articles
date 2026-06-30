# Byte Latent Transformer: Patches Scale Better Than Tokens

**Paper:** https://arxiv.org/abs/2412.09871
**Authors:** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer
**Affiliation:** Meta FAIR
**Code:** https://github.com/facebookresearch/blt
**Submitted:** December 2024

---

## The Central Question

Every language model you've ever heard of — GPT, Llama, Gemini — runs on *tokens*. Not words. Not characters. Tokens: chunks of text determined by a vocabulary learned before training begins. "unbelievable" might become ["un", "believ", "able"]. The word "Nairobi" in English probably tokenizes fine; in Swahili it might become letter-by-letter fragments.

Tokenization is a *pre-committed* decision baked into the model before it sees any data. It allocates the same compute to every token regardless of whether the token is surprising or trivially predictable.

BLT asks: **what if we skipped the tokenizer entirely and worked directly from raw bytes — but used dynamic, variable-length patches to control how much compute we spend per chunk of text?**

---

## The Problem with Tokenization

### The Fixed-Vocabulary Problem

A tokenizer is trained once on a corpus, builds a vocabulary of 32K–128K subword units, and that vocabulary is frozen. Every model trained with it is stuck with those choices forever. This creates three compounding issues:

**1. Unequal treatment across languages.** English text tokenizes efficiently (~4 characters/token). Languages written in non-Latin scripts (Chinese, Arabic, Thai) often tokenize to 1-2 characters per token — the same compute budget represents far less semantic content. A model trained with equal FLOPs per token is effectively undertrained on those languages.

**2. Brittleness to noise.** The token "hello" is one unit. The token "h3llo" (a simple character swap) maps to something entirely different — often a fragmented sequence the model has barely seen. BPE models are fundamentally character-blind because characters aren't the unit of computation.

**3. Fixed compute per token.** The word "the" and the word "photosynthesis" each get one token and therefore roughly one forward pass. But "the" is almost completely predictable from context — it barely needs compute. "photosynthesis" is informationally dense. A smarter system would spend more compute on the hard stuff and less on the easy stuff.

### The Byte-Level Naivety Trap

The obvious alternative — just process raw bytes — was tried and failed. Byte sequences are ~3-4× longer than token sequences for the same text. A standard transformer attending over byte sequences is 9-16× more expensive (quadratic attention) and much slower to converge. Every byte gets the same expensive treatment regardless of how predictable it is.

---

## BLT's Solution: Dynamic Patches with Three Specialized Transformers

BLT keeps the byte-level input (no vocabulary, no tokenizer) but introduces **patches** — variable-length groups of bytes that become the primary unit of computation.

The key insight: **patch boundaries should fall at high-entropy points** — places where the next byte is hard to predict. Predictable stretches of text (common words, repeated patterns) get bundled into large patches. Surprising, complex, or rare text gets smaller patches (more compute per character).

### The Three-Part Architecture

```
Input bytes ──► [ Local Encoder ] ──► patch representations ──► [ Global Latent Transformer ] ──► patch representations ──► [ Local Decoder ] ──► output bytes
     (lightweight)                                                      (heavy, expensive)                                         (lightweight)
```

**Local Encoder (ℰ)**
- A small, lightweight transformer (few layers)
- Sees raw bytes in a sliding window of 512 bytes
- Uses *cross-attention* to pool byte representations into patch representations
  - Patch queries attend to the bytes within that patch
  - Each patch "summarises" its constituent bytes
- Also uses **hash n-gram embeddings**: for each byte, compute n-grams of length 3–8, hash them to a fixed embedding table, and add those embeddings to the byte representation. This gives the encoder rich subword-level context without a vocabulary.

**Global Latent Transformer (𝒢)**
- The main, expensive transformer
- Operates on *patch sequences* (much shorter than byte sequences)
- This is where most of the model's parameters and FLOPs live
- Because patches are larger than tokens on average (6–8 bytes vs. 4.4 bytes for BPE), the sequence the global transformer sees is *shorter* than what a token-based model would see — cheaper to run
- Initialized from pretrained Llama 3.1 weights in the paper's transfer learning experiments

**Local Decoder (𝒟)**
- Another lightweight transformer
- Takes patch representations from the global transformer
- Uses reversed cross-attention: byte queries attend to patch keys/values
- Generates output bytes one at a time, conditioned on the patch context

### The Three Components Solve Three Different Problems

| Component | Handles | Cost |
|---|---|---|
| Local Encoder | Byte-level context, subword structure | Cheap (few layers, local window) |
| Global Transformer | Long-range reasoning, semantic understanding | Expensive (most parameters) |
| Local Decoder | Byte generation from patch context | Cheap (few layers, local window) |

The expensive global transformer runs on *patches*, not bytes. The cheap local modules handle the byte-level details. This is the core efficiency gain.

---

## Entropy Patching: How Patch Boundaries Are Chosen

This is the most technically interesting part. BLT uses a separate **100M-parameter entropy model** — a small byte-level language model trained on the same corpus — to estimate the entropy (uncertainty) of predicting the next byte at each position.

```
For each byte position i:
  H(xᵢ) = entropy of P(next byte | x₁...xᵢ)   ← computed by the 100M entropy model

Patch boundary at position i  ←→  H(xᵢ) exceeds a global threshold θ
                                    (i.e., the next byte is hard to predict)
```

**Intuition:** If you're in the middle of a common English word like "the", the next byte is nearly certain — low entropy, no patch boundary. If you've just written "the name of the company was" and the next character starts a proper noun, the entropy spikes — a patch boundary appears, meaning the global transformer will step in and pay attention.

Two variants:
- **Global threshold:** boundary whenever entropy > θ (absolute)
- **Relative/monotonic:** boundary whenever entropy *increases* relative to previous byte (detects "inflection points" of uncertainty)

Average patch size: **6–8 bytes** (vs. 4.4 bytes for BPE tokenizers). BLT's patches are bigger on average, meaning shorter sequences for the global transformer.

### Simpler Fallback: Space Patching

For languages/domains where whitespace reliably marks word boundaries, just split on spaces. Easier to compute, no entropy model needed. Slightly worse on scaling curves but achieves even larger average patches (6.1 bytes) and therefore even cheaper inference.

---

## Key Results

### Matching BPE at Equal FLOPs (Compute-Optimal)

Trained at equal FLOP budgets (1B to 8B parameter scale, up to 4T training bytes):

> BLT-Entropy **matches or outperforms** BPE (Llama 3-style) models across all sizes.

This is described as the "first byte-level transformer" to achieve this.

### Inference Efficiency at Parity

Once trained to equal performance, BLT can use **up to 50% fewer FLOPs at inference** compared to an equivalent BPE model — because its patch-based global transformer processes shorter sequences.

### Scaling Beyond Compute-Optimal

At fixed inference FLOP budgets, BLT has a new degree of freedom: simultaneously increase patch size and model capacity. This creates an **additional scaling dimension** that tokenizer-based models simply don't have.

| Benchmark | Llama 3 (BPE) | BLT-Entropy |
|---|---|---|
| MMLU | 58.1% | 57.4% |
| HumanEval | 31.1% | **35.4%** |
| Average (7 tasks) | 60.0% | **61.1%** |

### Robustness: Where BLT Dominates

This is where BLT's byte-level awareness really shines.

**CUTE Benchmark** (character-level tasks like spelling, character counting):
- BLT: **54.1%**
- Llama 3: **27.5%**
- Spelling specifically: BLT **99.9%** vs. Llama 3 **1.1%**

**Noisy HellaSwag** (character-level noise injected into inputs):
- BLT: **64.3%**
- Llama 3.1 (trained on 16× more data): **56.9%**

**Low-Resource Machine Translation** (27 language pairs):
- BLT: **+2 BLEU** over BPE models

BPE models are fundamentally character-blind — the token "hello" and the token "h3llo" are nearly unrelated in the model's embedding space. BLT sees both as sequences of bytes, making character-level variation natural to handle.

### Transfer Learning Bonus

Initializing BLT's global transformer from pretrained Llama 3.1 weights (same transformer backbone, different outer modules) and then continuing training **outperforms both BLT from scratch and Llama 3 from scratch** at the same FLOP budget. This suggests the transformer backbone learns universal computations that transfer cleanly from token-space to patch-space.

---

## The Story: The Post Office and the Smart Sorting Machine

### Chapter 1 — The Old Sorting System

Imagine the world's largest post office. Every day, millions of letters arrive. To process them, the sorting machine reads the address on each letter and routes it accordingly.

The old system works like this: before any letters arrive, a committee spends six months deciding on exactly 32,000 "address patterns" — standard abbreviations, common city names, known zip codes. Every incoming address gets matched to the nearest pattern. This is **tokenization**: a fixed vocabulary decided in advance.

It works well for major cities. Letters to "New York, NY" and "Los Angeles, CA" get processed fast. But letters to small villages in rural Nepal? The system breaks them into individual characters, letter by letter. The same machine that handles "NYC" in one slot now has to process "K", "a", "t", "h", "m", "a", "n", "d", "u" in nine separate slots. Same compute, less efficiency.

And if someone writes "N3w York" with a typo? The machine has never seen that pattern. It fails.

---

### Chapter 2 — The Byte Approach (and Why It Failed)

Someone suggested: "What if we just process every character individually? No pre-committed vocabulary — pure characters."

They built a prototype. It worked, in theory. No more vocabulary problem. Every character, every language, treated fairly.

But it was catastrophically slow. The sorting machine now had to examine every single character — and the hallway connecting the character-reading room to the routing brain was bottlenecked. Processing "New York" as 8 individual characters meant 8 trips down the hallway instead of 1.

The hallway was the expensive part. Raw bytes meant 3-4× more hallway trips than tokens. The prototype was abandoned.

---

### Chapter 3 — The Smart Bundler

A new engineer arrived and watched the sorting machines for a week. She noticed something: most of the hallway trips were for completely predictable stuff. After "New Yor", the next character is almost always "k". After "Dear Mr.", a name is coming. After "Sincerely your", a comma is nearly certain.

"We're wasting the expensive hallway," she said, "on things that are completely obvious."

She proposed a new machine: the **Smart Bundler**. Instead of sending each character individually, the Bundler would watch the incoming characters and decide when to bundle and when to send:

- If the next character is predictable → keep bundling. Don't send yet.
- If the next character is *surprising* — entropy spikes — bundle what you have and **send the bundle**.

A long, predictable phrase like "with kind regards" becomes one bundle. A rare technical term — "photolithography" — might generate several smaller bundles, one per cluster of uncertainty.

This is **entropy patching**. The bundles are patches.

---

### Chapter 4 — Three Rooms, Not One

The new architecture had three rooms, not one.

**Room 1: The Prep Room (Local Encoder)**
Every incoming character first enters the Prep Room. A small, fast team works here. They read the raw characters, compute n-gram patterns (what three-character sequences appear? what five-character sequences?), and summarise each bundle into a single card — a **patch representation**. This room is cheap to run: small team, local focus.

**Room 2: The Brain (Global Latent Transformer)**
The cards — patch representations — are sent down the hallway to the Brain. This is the expensive, powerful room. It reads the sequence of cards, understands the long-range structure ("this letter is from a lawyer, this section is about a contract dispute, the next section will probably request payment"), and annotates each card with its deep understanding.

The Brain runs on *cards*, not characters. Because each card represents 6–8 characters on average, the Brain sees a much shorter sequence than under the old system — and runs much faster.

**Room 3: The Writing Room (Local Decoder)**
Annotated cards go to the Writing Room. Another small, fast team takes each card and writes out the individual characters it represents, guided by the Brain's annotations.

Together, the three rooms read raw characters, reason at the level of meaning, and write raw characters back out — all without ever committing to a fixed vocabulary.

---

### Chapter 5 — The Entropy Oracle

But how does the Smart Bundler know *when* to close a bundle? It needs to answer: "Is the next character surprising?"

BLT uses a **100M-parameter oracle model** — a tiny language model trained to predict the next character at every position. It's not the main model; it's a fast specialist that just estimates uncertainty.

When the oracle says "high uncertainty ahead" — a new bundle begins. When it says "completely predictable" — the bundle grows. The oracle is the gatekeeper between cheap, obvious text and the expensive Brain.

This is why BLT is *dynamic* in a way tokens never are. A boring legal boilerplate might generate one patch per 10 characters. A table of chemical compound names might generate one patch per 3. The compute flows where it's needed.

---

### Chapter 6 — Why Character Blindness Breaks BPE Models

Back to the token-based post office for a moment. Here's what breaks it:

The token "misspelled" was learned as one unit. The token "m1sspelled" has never been seen — it decomposes into fragments the routing system barely knows. The connection between the two is completely lost.

A person reading either version knows instantly: same word, small noise. The BPE machine treats them as near-strangers.

BLT's machine never had this problem. It always read individual characters. "misspelled" and "m1sspelled" are 90% the same byte sequence — and the machine's character-level prep room captures this naturally.

This is why BLT scores 99.9% on spelling tasks and Llama 3 scores 1.1%. Not because Llama 3 is bad at reasoning — it's because BPE architecturally cannot "see" individual characters the way BLT can. The tokenizer blinded it before training even began.

---

### Chapter 7 — The New Scaling Axis

Here's the final insight that the paper emphasises: BLT opens a **new way to scale** that doesn't exist for token-based models.

For a token-based model, inference cost is fixed: sequence length depends on the vocabulary, and you can't change that after training. More model = more cost per token, always.

For BLT, inference cost has a free variable: **patch size**. You can choose to bundle more characters per patch (use a higher entropy threshold) to make the global transformer run faster — at a small accuracy cost. Or bundle less (more patches) for higher accuracy at higher cost. You can tune this *after training* to hit any point on the accuracy-cost curve.

Better yet: within a fixed inference FLOP budget, you can increase *both* patch size (fewer patches per sequence) *and* model capacity (more parameters per patch). A BPE model at fixed inference cost can only increase parameters. BLT can increase parameters *and* reduce sequence length together — a new dimension of scaling.

---

### Chapter 8 — The Universality Finding

The most surprising result in the paper: when researchers took pretrained Llama 3.1 weights (trained entirely on *tokens*) and used them to initialize BLT's global transformer, then continued training with the byte-patch architecture, the resulting model outperformed both:
- BLT trained from scratch
- Llama 3 trained from scratch

Same FLOP budget. Better model.

This suggests something deep: **the transformer backbone doesn't actually care whether it's processing tokens or patches**. The computations it learns — attention patterns, reasoning chains, world knowledge representation — are universal. They transfer cleanly from token-space to patch-space because both are just sequences of learned representations.

If that's true, every model trained on tokens today could be adapted to byte-patch inference — getting the efficiency and robustness benefits of BLT without paying the full training cost of starting over.

---

## Core Insight

> Tokenization is a shortcut that trades robustness and flexibility for efficiency. BLT shows the shortcut isn't necessary: dynamic patches built from bytes can match tokenized models at equal compute, use less compute at inference, and — crucially — see the text the way a human sees it: one character at a time.

---

## The Math — Step by Step

### Entropy of the Next Byte

The entropy oracle is a small language model `p(· | x₁...xᵢ)` over the 256 possible next bytes.

```
H(xᵢ) = -∑_{b ∈ {0..255}} p(b | x₁...xᵢ) · log p(b | x₁...xᵢ)
```

High entropy → many bytes are plausible → model is uncertain → patch boundary.
Low entropy → one byte is almost certain → model is confident → no boundary, keep growing current patch.

### N-gram Hash Embeddings

For each byte position `i`, compute all n-grams of lengths 3–8 ending at position `i`:

```
n-grams = {x_{i-2}x_{i-1}x_i, x_{i-3}...x_i, ..., x_{i-7}...x_i}
```

Each n-gram is hashed to an index in a fixed embedding table (size 400K per n-gram length). The corresponding embeddings are summed and added to the byte's base embedding. This gives the byte context about surrounding character patterns — essentially a learned subword representation without a fixed vocabulary.

### Cross-Attention: Bytes → Patches (Encoder)

For each patch `p` containing bytes `b_s ... b_e`:

```
Patch query   q_p = linear(mean(h_{b_s}...h_{b_e}))   # pooled byte representations
Byte keys     K = [h_{b_s}, ..., h_{b_e}]
Byte values   V = [h_{b_s}, ..., h_{b_e}]

patch_repr_p = softmax(q_p · K^T / √d) · V           # standard attention
```

The patch "reads" all the bytes it contains and compresses them into a single representation.

### Cross-Attention: Patches → Bytes (Decoder)

Reversed role — the byte asks the patch:

```
Byte query    q_b = h_b
Patch keys    K_p = [patch_repr_p]
Patch values  V_p = [patch_repr_p]

byte_repr_b = softmax(q_b · K_p^T / √d) · V_p        # byte attends to its patch
```

### FLOP Comparison at Inference

For a sequence of N bytes forming P patches (P < N):

```
Token model FLOPs  ∝  N² / 2                  # attention over N tokens
BLT FLOPs          ∝  P² / 2   +   N · w      # global transformer on P patches +
                                               # local encoder/decoder on N bytes with window w
```

Since P ≈ N/6, the global transformer (expensive part) is ~36× cheaper in attention. The local modules are linear in N. Net result: 50% fewer FLOPs at inference for equivalent performance.

---

## Key Numbers

| Metric | Value |
|---|---|
| Average patch size (entropy) | 6–8 bytes |
| Average BPE token size | 4.4 bytes |
| Entropy oracle size | 100M parameters |
| Entropy oracle architecture | 14 layers, 512 hidden dim, 512-token window |
| N-gram lengths | 3–8 bytes |
| Embedding table size | 400K entries per n-gram length |
| Models trained in scaling study | 400M, 1B, 2B, 4B, 8B |
| Training bytes | Up to 4T bytes |
| Inference FLOP savings | Up to 50% vs. equivalent BPE model |
| CUTE benchmark (BLT vs Llama 3) | 54.1% vs. 27.5% |
| Spelling specifically | 99.9% vs. 1.1% |
| Noisy HellaSwag | 64.3% vs. 56.9% (vs. Llama 3.1 trained on 16× more data) |
| Low-resource MT improvement | +2 BLEU over BPE |
| HumanEval | 35.4% (BLT) vs. 31.1% (Llama 3) |

---

## Relation to Other Concepts

- **Tokenization (BPE, WordPiece)** — directly replaced. BLT shows BPE is a pragmatic shortcut, not a fundamental requirement. The vocabulary commitment problem (multilingual inequality, noise brittleness) is structural to BPE and eliminated by BLT.
- **Perceiver / Perceiver IO** — BLT's cross-attention pooling (bytes → patches in the encoder) is architecturally similar to Perceiver's latent bottleneck. BLT extends this to autoregressive generation with a decoder.
- **Compute-adaptive inference** — the idea that compute should scale with input complexity. BLT does this at the architectural level (high-entropy regions get more patches = more global transformer steps); similar in spirit to Mixture of Experts routing.
- **Mamba / state space models** — alternative approach to handling long byte sequences with linear-time recurrence instead of quadratic attention. BLT instead uses the patch compression to achieve sub-quadratic cost while keeping full attention in the global transformer.
- **MegaByte** (Yu et al., 2023) — the closest prior work. Also used local encoder/decoder + global transformer on patches. BLT improves by: entropy-based dynamic patching (vs. fixed patch size), hash n-gram embeddings, and scale (8B vs. smaller models in MegaByte).
- **Byte-pair encoding history** — BPE was originally a text compression algorithm (Gage, 1994), adapted for NLP by Sennrich et al. (2016). BLT is in some ways a return to pre-BPE character-level models, but with the compute management that makes them practical at scale.
- **Prismatic Synthesis** (see `../reasoning/prismatic-synthesis/`) — both papers care about compute allocation. Prismatic Synthesis allocates training-data diversity to underrepresented gradient regions; BLT allocates inference compute to high-entropy byte regions. Different domain, same principle: compute should flow where it's most needed.

---

Sources:
- [arXiv abstract](https://arxiv.org/abs/2412.09871)
- [HTML version of paper](https://arxiv.org/html/2412.09871v1)
- [Meta AI Research page](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/)
- [Gonzo ML technical summary](https://gonzoml.substack.com/p/blt-byte-latent-transformer)
- [Code — facebookresearch/blt](https://github.com/facebookresearch/blt)
