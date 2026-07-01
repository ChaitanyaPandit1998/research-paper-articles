# The Paper That Killed RNNs: How "Attention Is All You Need" Rewired AI

> **Medium tags:** Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, Transformers
>
> **Hero image:** A dark-background visualization of an attention matrix — a grid of glowing connection lines between words in a sentence, with "animal" and "it" connected by a bright, thick arc while other connections fade into the background. The visual should feel like a neural constellation map, not a dry diagram.
>
> **Pull quotes:**
> 1. *"The Transformer doesn't play telephone. It puts every word in the same room and lets them all talk to each other simultaneously."*
> 2. *"Better results, less compute. That combination does not happen often."*
> 3. *"The eight researchers who wrote this paper did not build a better translation system. They built the substrate on which almost everything that followed would run."*

---

Every AI system that has impressed, unsettled, or employed you in the last four years runs on an idea from a 15-page paper published on a Friday in June 2017. Eight researchers wrote it. None of them expected it to age this well.

---

## The World Before 2017

If you wanted a machine to understand language in 2016, you fed it one word at a time.

**Recurrent Neural Networks** — RNNs, and their smarter cousin **LSTMs** — processed sequences like a person reading with amnesia: left to right, one token at a time, compressing everything seen so far into a single fixed-size vector called a **hidden state**. That hidden state had to carry the entire context of the sentence forward. For short sentences, this worked. For long ones, early information simply bled away — a phenomenon called the **vanishing gradient problem**.

The deeper issue was architectural: because each step depended on the previous one, you could not parallelise training. A sequence of 500 words required 500 sequential operations. GPUs — the hardware that makes modern AI possible — are built for parallelism. RNNs wasted most of that hardware. Researchers knew it. Nobody had a clean fix.

Attention mechanisms existed as a *patch* for RNNs, letting the model peek back at earlier parts of the input. The Transformer paper asked the obvious question that turned out to be radical: what if attention was the whole model?

### What You'll Learn

- Why sequential computation was the hidden bottleneck killing language model scale
- How **self-attention** lets every word in a sentence simultaneously consider every other word
- What the Transformer encoder–decoder architecture actually does, layer by layer
- The intuition behind the scaled dot-product attention equation — no linear algebra degree required
- Why this architecture, with almost no modifications to its core design, still dominates in 2025

---

## The Room vs. The Telephone Chain

Imagine you need to translate this sentence from English to French:

*"The animal didn't cross the street because it was too tired."*

What does "it" refer to? The animal, not the street. You know this instantly — because your brain didn't read that sentence left-to-right and forget the beginning. It held the whole sentence in view and resolved the ambiguity by comparing "it" against every other word at once.

An RNN reads that sentence like a game of telephone. The word "animal" gets whispered to "didn't," which whispers to "cross," and so on. By the time the model reaches "it," the signal from "animal" is faint. The Transformer doesn't play telephone. It puts every word in the same room and lets them all talk to each other simultaneously.

That is **self-attention**.

### Self-Attention

**What it is:** A mechanism that lets every token in a sequence compute a weighted relationship with every other token in a single step.

**Why it matters:** It replaces the sequential bottleneck of RNNs entirely — every relationship is computed in parallel.

**Analogy:** Think of a group of detectives sharing a case file. Each detective reads every other detective's notes at the same time and decides whose information is most relevant to their own. Nobody waits for the person to their left to finish before reading.

### Queries, Keys, and Values

Self-attention works through three learned projections of each token: a **Query**, a **Key**, and a **Value**.

**What they are:** For each token, the model learns three different representations — what it's looking for (Query), what it contains (Key), and what it will contribute if attended to (Value).

**Why it matters:** This separation lets the model learn *what to ask* independently of *what to offer* — making attention flexible and composable.

**Analogy:** Think of a library. Your Query is the search term you type. Every book has a Key — a catalog entry describing its contents. The library scores your query against all the keys, then hands you the actual books (Values) weighted by how well they matched. You don't read every book equally — you read the most relevant ones most carefully.

Each token asks: *"Given what I am, which other tokens should I pay attention to?"* The answer is a weighted blend of all the Values, where the weights come from how well each token's Key matched the token's Query.

### Multi-Head Attention

The Transformer doesn't run self-attention once. It runs it several times in parallel, with different learned projections each time. These are called **attention heads**.

**What it is:** Running multiple self-attention operations simultaneously, each learning to attend to different kinds of relationships.

**Why it matters:** One head might track grammatical agreement; another might track coreference (what "it" refers to); another might track semantic similarity. A single attention pass can only optimise for one view at a time.

**Analogy:** Reading a contract with three people simultaneously — a lawyer looking for liability clauses, an accountant looking for payment terms, and a manager looking for deadlines. Each reads the same document but extracts a different signal. The Transformer concatenates all their findings.

### Positional Encoding

Self-attention has a problem: it treats a sentence as a *set*, not a *sequence*. "Dog bites man" and "Man bites dog" would look identical to a pure attention layer because the same words appear — just in a different order.

**Positional encoding** is a fixed mathematical signal added to each token's embedding that encodes its position in the sequence. Without it, the model has no concept of word order — which is catastrophic for language.

**Analogy:** Imagine a conference where every attendee wears a name badge with their seat number. The people (tokens) are the same regardless of where they sit, but their seat number tells you something about their relationship to others in the room.

The paper uses sine and cosine functions at different frequencies — a choice that lets the model generalise to sequences longer than those it trained on.

### The Encoder–Decoder Structure

The full Transformer stacks these components into two halves.

The **Encoder** reads the input sequence — say, an English sentence — and builds a rich contextual representation of it. Each of the six encoder layers refines that representation, passing it to the next.

The **Decoder** generates the output sequence — say, the French translation — one token at a time. It attends to its own partial output via **masked self-attention** (which prevents it from cheating by looking at future tokens) and also attends to the encoder's representation via a second attention layer called **cross-attention**.

**Analogy:** The encoder is a skilled analyst who reads the source document and writes a dense briefing. The decoder is a writer who reads that briefing and composes the output, one sentence at a time, consulting the briefing at every step.

```mermaid
flowchart TD
    subgraph InputSide["Input Side"]
        A[Source Tokens] --> B[Input Embedding]
        B --> C[Positional Encoding]
    end

    subgraph Encoder["Encoder — repeated 6x"]
        C --> D[Multi-Head Self-Attention]
        D --> E[Add & Norm]
        E --> F[Feed-Forward Network]
        F --> G[Add & Norm]
    end

    subgraph OutputSide["Output Side"]
        H[Target Tokens - shifted right] --> I[Output Embedding]
        I --> J[Positional Encoding]
    end

    subgraph Decoder["Decoder — repeated 6x"]
        J --> K[Masked Multi-Head Self-Attention]
        K --> L[Add & Norm]
        L --> M[Multi-Head Cross-Attention]
        G --> M
        M --> N[Add & Norm]
        N --> O[Feed-Forward Network]
        O --> P[Add & Norm]
    end

    P --> Q[Linear Projection]
    Q --> R[Softmax]
    R --> S[Output Token Probabilities]
```

Each encoder and decoder layer follows the same two-step rhythm: attend, then transform. The attention sublayer lets tokens communicate. The **feed-forward sublayer** — a simple two-layer network applied independently to each token — processes each token's updated representation. A **residual connection** wraps each sublayer, adding the input back to the output, which keeps gradients healthy during training. **Layer normalisation** stabilises the activations.

Six of these layers, stacked. That's the encoder. The decoder is the same, with one extra cross-attention sublayer inserted between them. The whole architecture is almost aggressively modular. That modularity is part of why it survived.

---

## The Math, Demystified

The Transformer's architecture is elegant in prose, but the paper's real contribution lives in three equations. They are not decorative — they are the entire mechanism. Understanding them tells you *why* self-attention works, not just *that* it does. None of them require anything beyond matrix multiplication and a function you already know intuitively: ranking by relevance.

### Equation 1: Scaled Dot-Product Attention

> 📌 *Insert equation image here — render at [codecogs.com/latex/eqneditor](https://editor.codecogs.com) and upload to Medium*

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

**Every symbol unpacked:**

- **Q** — the Query matrix: a stack of "what am I looking for?" vectors, one per token
- **K** — the Key matrix: a stack of "what do I contain?" vectors, one per token
- **V** — the Value matrix: a stack of "what will I contribute?" vectors, one per token
- **Q·Kᵀ** — a dot product between every query and every key, producing a score for every token pair
- **d_k** — the dimension of the key vectors
- **√d_k** — a scaling factor to stop the dot products from exploding in magnitude
- **softmax(·)** — converts raw scores into probabilities, weights that sum to 1

**What it's doing:** For each token, it scores its query against every other token's key, scales those scores to keep them numerically stable, converts them to weights, then uses those weights to compute a weighted average of all the values.

**Why it matters:** This single equation replaces the entire recurrence mechanism of an RNN. Every token's output is now a function of *all* other tokens simultaneously — computed in one matrix operation.

The √d_k term is easy to skip over. It earns its place. When d_k is large — say, 64 — the dot products between queries and keys grow large in magnitude. Large inputs to softmax push the output into regions where gradients are nearly zero, which kills learning. Dividing by √d_k keeps the scores in a well-behaved range. One line with an outsized effect on training stability.

### Equation 2: Multi-Head Attention

> 📌 *Insert equation image here — render and upload to Medium*

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · Wᴼ
where head_i = Attention(Q·W_i^Q,  K·W_i^K,  V·W_i^V)
```

**Every symbol unpacked:**

- **h** — the number of attention heads (8 in the base model)
- **W_i^Q, W_i^K, W_i^V** — learned projection matrices; each head projects Q, K, V into its own lower-dimensional subspace
- **head_i** — the output of one full attention computation in that subspace
- **Concat(·)** — stack all head outputs side by side into one wide vector
- **Wᴼ** — a learned output projection that compresses the concatenated heads back to model dimension

**What it's doing:** It runs h independent attention operations on learned projections of the same input, then merges their outputs through a final linear layer.

**Why it matters:** Each head can specialise in a different type of dependency — syntax, coreference, semantics — without any of them being explicitly trained to do so. This emergent specialisation is one reason the Transformer generalises so well.

### Equation 3: Positional Encoding

> 📌 *Insert equation image here — render and upload to Medium*

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

**Every symbol unpacked:**

- **pos** — the position of the token in the sequence (0, 1, 2, …)
- **i** — the dimension index within the encoding vector
- **d_model** — the model's embedding dimension (512 in the base model)

**What it's doing:** It assigns each position a unique vector built from sine and cosine waves at different frequencies — lower dimensions oscillate slowly (capturing long-range position), higher dimensions oscillate quickly (capturing fine-grained position).

**Why it matters:** The relative position between any two tokens can be expressed as a linear function of their encodings — meaning the model can learn to attend based on *distance*, not just absolute position.

```mermaid
flowchart LR
    A[Token Embeddings] -->|+ positional signal| B[Positional Encoding]
    B --> C[Query / Key / Value Projections]
    C -->|dot product + scale| D[Attention Scores]
    D -->|softmax| E[Attention Weights]
    E -->|weighted sum of Values| F[Single Head Output]
    F -->|repeat h times| G[Multi-Head Outputs]
    G -->|concat + linear| H[Multi-Head Attention Output]
    H -->|add input + normalise| I[Add & Norm]
    I --> J[Feed-Forward Network]
    J -->|add input + normalise| K[Layer Output]
```

Every token asks a question (Query), every token posts an answer (Key + Value), and the model learns — entirely from data — which questions and answers matter for the task at hand.

---

## The Numbers That Changed Everything

The paper's authors tested the Transformer on machine translation — the standard benchmark for sequence-to-sequence models in 2017. The results were not a marginal improvement. They rewrote the leaderboard.

**WMT 2014 Translation Benchmarks**

> 📌 *Paste this into a Google Sheet and embed as an image, or use [tablesgenerator.com](https://tablesgenerator.com) to export as an image for Medium*

```
Model                   EN→DE BLEU   EN→FR BLEU   Training Cost
──────────────────────────────────────────────────────────────────
ByteNet                   23.75          —              —
ConvS2S (single)          25.16        40.46       ~1.5 × 10¹⁹
ConvS2S (ensemble)        26.36        41.29            —
MoE (ensemble)            26.03        40.56       ~1.2 × 10²⁰
Transformer (base)        27.3         38.1        ~3.3 × 10¹⁸  ✓
Transformer (big)         28.4         41.0        ~2.3 × 10¹⁹  ✓
```

The big Transformer beat all prior single models and most ensembles on English-to-German — by more than 2 BLEU points over the best ensemble. On English-to-French, it matched the best ensemble result with a *single model* trained for a fraction of the cost.

That last column is the quiet bombshell. Better results, less compute. That combination does not happen often.

### What Changed in the Field

The Transformer did not just win a benchmark. It ended an era.

Within 18 months, every major NLP architecture had either adopted attention or been abandoned:

- **2018 — BERT** (Google): encoder-only Transformer, pre-trained on masked language modelling. Redefined NLP benchmarks across the board.
- **2018 — GPT** (OpenAI): decoder-only Transformer, pre-trained autoregressively. The direct ancestor of every GPT model since.
- **2019 — T5** (Google): reframed every NLP task as text-to-text, using the original encoder-decoder structure almost unchanged.
- **2020 — GPT-3**: 175 billion parameters, still the same architecture. Proved that scale applied to the Transformer produced emergent capabilities nobody predicted.
- **2020 — AlphaFold 2** (DeepMind): used attention to solve protein structure prediction — a biology problem, not an NLP one.
- **2021 — Vision Transformer (ViT)**: applied the same architecture to image patches, beating convolutional networks on image classification.
- **2022–present**: Stable Diffusion, Whisper, Codex, PaLM, Llama, Gemini — all Transformers at their core.

The paper's own authors scattered across the industry. Several founded companies — Adept, Cohere, Character.ai — that built directly on this work. The architecture they published on a Friday in June 2017 now runs on hundreds of millions of devices daily.

### What the Paper Did Not Solve

Honest accounting matters here. The Transformer introduced new problems as it closed old ones.

**Quadratic attention complexity.** Self-attention computes a score for every token pair. Double the sequence length and you quadruple the compute. At 512 tokens this is fine. At 100,000 tokens — a legal document, a codebase, a book — it becomes prohibitive. Seven years of follow-on work (Longformer, FlashAttention, linear attention variants) have attacked this problem without fully solving it.

**No structural inductive bias.** RNNs naturally understood that nearby tokens are more related than distant ones. The Transformer treats position purely through learned positional encodings — a patch, not a principle. Later work (RoPE, ALiBi) replaced the paper's sinusoidal scheme with better approaches precisely because the original choice had limits at long range.

**Data hunger.** The Transformer's parallelism is a training-time advantage. It also means the model sees no inherent structure — it learns everything from data. That requires enormous datasets. Applying this architecture to low-resource languages or domains without large corpora remained hard.

**Positional generalisation.** Models trained on sequences up to length 512 degrade on sequences of length 1024. The architecture has no guaranteed ability to extrapolate position — a limitation that still drives active research.

---

## What Comes Next

### Key Takeaways

- The Transformer replaced sequential processing with parallel attention — letting every token in a sequence relate to every other token simultaneously, in a single matrix operation.
- Queries, Keys, and Values are not metaphors. They are learned projections that give the model a structured way to ask questions, post answers, and weight contributions across the entire input.
- Multi-head attention is what turns a clever mechanism into a powerful one — different heads learn to track different kinds of relationships without being told to.
- The architecture's modularity is not incidental. Stack the same two-step block — attend, then transform — and almost any sequence problem becomes tractable.
- The paper's limitations were real: quadratic complexity, data hunger, positional encoding as a patch. Every major advance in the years since has been, at some level, an attempt to fix one of these without breaking the rest.

### The Bigger Picture

What "Attention Is All You Need" actually proved was not that attention is better than recurrence. It proved that the right abstraction, applied consistently, scales. The Transformer did not win because it was maximally clever — it won because it was parallelisable, composable, and simple enough to improve. Every architecture that followed — BERT, GPT, ViT, AlphaFold, Whisper — inherited that simplicity and then pushed one dimension of it further: more data, more parameters, more modalities, longer context. We are still in that expansion phase. The limits of the Transformer are not yet in sight, and the researchers who find them will almost certainly do so by building on top of this paper, not by replacing it.

### Go Deeper

1. **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** (Devlin et al., 2018) — the paper that took the Transformer encoder and turned it into the dominant approach for language understanding tasks. Read it to see how fine-tuning became the new paradigm.

2. **"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"** (Dosovitskiy et al., 2020) — the paper that proved the Transformer was not an NLP-specific tool. If you work in vision or multimodal systems, this is the fork in the road.

3. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022) — the paper that made long-context Transformers practical by rethinking how attention is computed at the hardware level. The math is unchanged; the engineering insight is everything.

---

The eight researchers who wrote this paper did not build a better translation system. They built the substrate on which almost everything that followed would run — and then, apparently, moved on to their next problem.

---

## Final Word Count Breakdown

| # | Section | Target | Actual (approx.) |
|---|---------|--------|-----------------|
| 1 | Hook & Introduction | 300 | ~310 |
| 2 | The Problem with Sequences | 400 | ~280 (merged into intro) |
| 3 | The Big Idea: Attention + Architecture | 1,100 | ~1,050 |
| 4 | The Math, Demystified | 500 | ~520 |
| 5 | Results & Real-World Impact | 400 | ~430 |
| 6 | What the Paper Didn't Solve | 250 | ~240 |
| 7 | Conclusion & What's Next | 250 | ~280 |
| — | **Total** | **3,200** | **~3,110** |
