# Sliding Window Attention (SWA)

**What it is:** A way to make attention cheaper on long sequences by letting most layers only look at nearby tokens (a "window"), instead of every token in the sequence.

**Key idea:** `window_pattern = "SSSL"` — most layers are **S** (sliding/local window), only occasional layers are **L** (full/long-range context).

**Used in:** Mistral 7B, Longformer, BigBird, and models built for long-context efficiency.

---

## The Problem: Standard Attention Is Quadratic

In standard attention, every token looks at every other token:

```
Sentence: "The cat sat on the mat"  (6 tokens)

"The"  looks at → The, cat, sat, on, the, mat  (6 comparisons)
"cat"  looks at → The, cat, sat, on, the, mat  (6 comparisons)
"sat"  looks at → The, cat, sat, on, the, mat  (6 comparisons)
"on"   looks at → The, cat, sat, on, the, mat  (6 comparisons)
"the"  looks at → The, cat, sat, on, the, mat  (6 comparisons)
"mat"  looks at → The, cat, sat, on, the, mat  (6 comparisons)

Total comparisons = 6 × 6 = 36
```

For **n tokens**, attention computes **n × n comparisons** — called O(n²).

This is fine for 512 or 4096 tokens. But scale up:

```
n = 10,000 tokens  →  100,000,000 comparisons
n = 100,000 tokens →  10,000,000,000 comparisons  ← impossible on current hardware
```

---

## The Insight: Most Tokens Only Need Local Context

Think about reading a book. To understand the word "sat", you mostly need:
- The words right around it ("The cat sat on the mat")
- Not the sentence from page 1 if you're on page 200

For **most** of what a language model does, nearby context is enough.
Only occasionally does a token genuinely need to look back very far.

**Sliding Window Attention** exploits this:
> Each token only attends to a fixed window of tokens around it.
> The window "slides" as you move through the sequence.

---

## How It Works: The Window

Say the window size `w = 2` (look at 2 tokens before and 2 tokens after).

```
Sentence: "The  cat  sat  on  the  mat"
Position:   0    1    2    3    4    5

"The" (pos 0) can see:  [pos 0, 1, 2]          → The, cat, sat
"cat" (pos 1) can see:  [pos 0, 1, 2, 3]        → The, cat, sat, on
"sat" (pos 2) can see:  [pos 0, 1, 2, 3, 4]     → The, cat, sat, on, the
"on"  (pos 3) can see:  [pos 1, 2, 3, 4, 5]     → cat, sat, on, the, mat
"the" (pos 4) can see:  [pos 2, 3, 4, 5]        → sat, on, the, mat
"mat" (pos 5) can see:  [pos 3, 4, 5]           → on, the, mat
```

Visualised as an attention matrix (✓ = can attend, · = blocked):

```
           The  cat  sat   on  the  mat
           ─── ─── ─── ─── ─── ───
"The"  →  [ ✓   ✓   ✓   ·   ·   · ]
"cat"  →  [ ✓   ✓   ✓   ✓   ·   · ]
"sat"  →  [ ✓   ✓   ✓   ✓   ✓   · ]
"on"   →  [ ·   ✓   ✓   ✓   ✓   ✓ ]
"the"  →  [ ·   ·   ✓   ✓   ✓   ✓ ]
"mat"  →  [ ·   ·   ·   ✓   ✓   ✓ ]

Standard attention: 36 comparisons  (filled matrix)
Sliding window:     18 comparisons  (only the band)
```

For long sequences the saving is massive:
```
Standard:        n × n   comparisons
Sliding window:  n × w   comparisons  (w = window size, fixed)

n = 100,000, w = 512:
  Standard:  10,000,000,000
  Sliding:      51,200,000   ← 200× cheaper
```

---

## The window_pattern: "SSSL"

In the codebase, `window_pattern = "SSSL"` defines which layers use sliding window and which use full attention.

```
S = Sliding window layer  (local window only — fast, cheap)
L = Long-range layer      (full attention — sees everything, expensive)
```

The pattern **tiles** across all layers of the model. For a 12-layer model with pattern `"SSSL"`:

```
Layer  0:  S  → sliding window  (window = context_length / 4)
Layer  1:  S  → sliding window
Layer  2:  S  → sliding window
Layer  3:  L  → full attention  (sees all tokens)
Layer  4:  S  → sliding window
Layer  5:  S  → sliding window
Layer  6:  S  → sliding window
Layer  7:  L  → full attention
Layer  8:  S  → sliding window
Layer  9:  S  → sliding window
Layer 10:  S  → sliding window
Layer 11:  L  → full attention  ← always full (final layer)
```

Three cheap layers for every one expensive layer. The model still gets global context through the L layers — it just doesn't pay the full cost on every layer.

### The "quarter context" window

For S layers, the window size = **context_length / 4**:

```
Context length = 4096 tokens
Window size    = 4096 / 4 = 1024 tokens

Each token in an S layer attends to 1024 nearby tokens.
Each token in an L layer attends to all 4096 tokens.
```

---

## Why This Still Works: Information Flows Through Layers

You might think: if early layers can't see far away, how does "mat" ever learn anything about "The"?

The answer is: information propagates through multiple layers.

```
Layer 0 (S): "mat" can't see "The" directly, but it can see "the"
Layer 1 (S): "mat" attends to "the", which already saw "sat" in layer 0
Layer 2 (S): "mat" attends to "the", which saw "cat" via "sat"
Layer 3 (L): "mat" now attends to everything — and all tokens have
             already exchanged local information in previous layers

By the time we reach Layer 3, every token's representation
already contains info from its neighbourhood.
Full attention in L layers ties everything together.
```

It's like a game of telephone — information passes through local chains, then gets integrated globally at the L layers.

---

## Walkthrough: "The cat sat on the mat" Through "SSSL" Layers

Let's say we have a 4-layer model (`SSSL`) and a window of 2 for simplicity.
We're tracking what **"mat"** (position 5) knows at each layer.

### Layer 0 — S (sliding window, w=2)

```
"mat" can attend to: on(3), the(4), mat(5)

"mat" learns:
  → "on" is a preposition nearby
  → "the" is an article nearby
  → itself

"mat" does NOT yet know about "The", "cat", "sat"
```

### Layer 1 — S (sliding window, w=2)

```
"mat" attends to: on(3), the(4), mat(5)

But now "the"(4) already contains info from layer 0:
  "the"(4) attended to: sat(2), on(3), the(4), mat(5)

So "mat" indirectly picks up info from "sat" and "on"
via "the"'s updated representation.
```

### Layer 2 — S (sliding window, w=2)

```
"mat" attends to: on(3), the(4), mat(5)

"the"(4) now contains info from layers 0+1:
  which includes "cat"(1) via "sat"(2) via "on"(3)

"mat" now indirectly knows about most of the sentence
even though it never directly attended to "The" or "cat".
```

### Layer 3 — L (full attention)

```
"mat" attends to: The, cat, sat, on, the, mat  ← everything

Full context is available.
"mat" can now directly integrate info from all tokens.
"cat sat on the mat" — the full phrase is now understood.
```

---

## The "Final Layer Always Gets Full Context" Rule

From the code comment: *"only the final layer always gets full context"*.

Even if your `window_pattern` is `"SSS"` (no L at all), the last layer is **always forced to L**.

```
Why? The output of the final layer is what gets decoded into the next token.
If the final layer can't see the full context, the model might miss
crucial information when making its prediction.

Think of it as: the final layer is the "summariser" — it needs the full picture.
```

---

## Compute Savings: The Maths

For a model with `n_layers` layers, pattern `"SSSL"` (3 S per L):

```
Standard attention per layer:  n² operations
Sliding window per layer:       n × w  operations  (w << n)

With pattern "SSSL" over 4 layers:
  Standard:  4 × n²
  SSSL:      3 × (n × w) + 1 × n²
           = 3nw + n²

For n=4096, w=1024:
  Standard:  4 × 16,777,216  = 67,108,864
  SSSL:      3 × 4,194,304   + 16,777,216
           = 12,582,912      + 16,777,216
           = 29,360,128

That's ~2.3× cheaper — and the gap grows rapidly as n increases.
```

---

## window_pattern Examples

```
"L"       → every layer is full attention (standard Transformer)
"S"       → every layer is sliding window (maximum speed, less global context)
"SSSL"    → 3 local for every 1 global (good balance, default in code)
"SSL"     → 2 local for every 1 global (moderate)
"SSSSSSL" → 6 local for every 1 global (very long contexts)
```

The longer the sequence you need to handle, the more S layers you add relative to L layers.

---

## Summary

| | Standard Attention | Sliding Window (S layer) | Full (L layer) |
|---|---|---|---|
| Each token sees | All n tokens | Only w nearby tokens | All n tokens |
| Compute per layer | O(n²) | O(n × w) | O(n²) |
| Good for | Short sequences | Capturing local patterns | Global integration |
| In "SSSL" pattern | — | 3 out of 4 layers | 1 out of 4 layers |

> **One-line summary:** Most layers look through a sliding window to save compute; occasional full-attention layers stitch together the global picture. Information from far-away tokens reaches every token indirectly through the chain of local windows.
