# MHA vs MQA vs GQA — Step-by-Step Walkthrough

Sentence: **"The cat sat on the mat"**

We'll use **4 attention heads** (simplified from the real 32) to keep numbers manageable.

---

## PART 0: What Are Q, K, V — and Why Do We Need Them?

Before comparing the three approaches, let's understand what Q, K, and V actually do.

When a Transformer reads a sentence, every token needs to decide:
> *"Which other tokens should I pay attention to?"*

It does this through three vectors per token:

```
Q (Query)  = "What am I looking for?"
K (Key)    = "What do I contain?"
V (Value)  = "What information do I actually give out?"
```

### Applied to "The cat sat on the mat"

```
Token    Q (What am I looking for?)              K (What do I contain?)
───────  ──────────────────────────────────────  ──────────────────────────────
"The"    Who do I modify? What noun follows?      I am an article
"cat"    What is my verb? What describes me?      I am a subject noun
"sat"    Who is my subject? Any object?           I am a past-tense verb
"on"     What connects me to surroundings?        I am a preposition
"the"    Who do I modify?                         I am an article
"mat"    What describes me? What verb uses me?    I am an object noun
```

**Attention score** = Q · K (dot product)
→ High score = these two tokens are strongly related
→ The model attends to the tokens with the highest scores

**V (Value)** = The actual content passed forward once attention is decided.
Attending to "sat" means you receive "sat"'s V vector — the information it carries.

---

## PART 1: Multi-Head Attention (MHA)

### The Idea

Run attention **4 times in parallel**, each time with different Q, K, V matrices.
Each "head" learns to look for different relationships.

```
Head 1 might learn: subject-verb relationships
  "cat" → pays attention to → "sat"

Head 2 might learn: verb-object relationships
  "sat" → pays attention to → "mat" (via "on")

Head 3 might learn: article-noun relationships
  "The" → pays attention to → "cat"
  "the" → pays attention to → "mat"

Head 4 might learn: positional proximity
  every token → pays attention to → its neighbours
```

### For Each Token, We Compute

```
                Head 1        Head 2        Head 3        Head 4
                ──────        ──────        ──────        ──────
"The"      Q1  K1  V1    Q2  K2  V2    Q3  K3  V3    Q4  K4  V4
"cat"      Q1  K1  V1    Q2  K2  V2    Q3  K3  V3    Q4  K4  V4
"sat"      Q1  K1  V1    Q2  K2  V2    Q3  K3  V3    Q4  K4  V4
"on"       Q1  K1  V1    Q2  K2  V2    Q3  K3  V3    Q4  K4  V4
"the"      Q1  K1  V1    Q2  K2  V2    Q3  K3  V3    Q4  K4  V4
"mat"      Q1  K1  V1    Q2  K2  V2    Q3  K3  V3    Q4  K4  V4

Each token produces:  4 Queries + 4 Keys + 4 Values
                      = 12 vectors per token
```

### How Attention Works for "cat" in Head 1

```
Step 1 — "cat" sends its Query (Q₁) out to every other token's Key (K₁):

  "cat"Q₁ · "The"K₁  = 0.3   (weak match)
  "cat"Q₁ · "cat"K₁  = 0.9   (strong — self-attention)
  "cat"Q₁ · "sat"K₁  = 0.8   (strong — subject matches its verb)
  "cat"Q₁ · "on"K₁   = 0.1   (weak)
  "cat"Q₁ · "the"K₁  = 0.2   (weak)
  "cat"Q₁ · "mat"K₁  = 0.2   (weak)

Step 2 — Convert scores to weights (softmax, must sum to 1):
  [0.05, 0.40, 0.35, 0.07, 0.06, 0.07]

Step 3 — Blend the Value vectors using those weights:
  output = 0.05×"The"V₁ + 0.40×"cat"V₁ + 0.35×"sat"V₁ + ...

Result: "cat"'s new representation, enriched with info from "sat"
```

This happens independently in all 4 heads, then the 4 outputs are concatenated.

### The KV Cache Problem During Inference

When the model **generates** text (predicts the next token one at a time), it needs
to attend to every previous token. Instead of recomputing K and V every step, it
caches them.

```
Generating "The" (step 1):
  Cache: [K₁V₁, K₂V₂, K₃V₃, K₄V₄] for "The"
  = 4 K vectors + 4 V vectors = 8 vectors stored

Generating "cat" (step 2):
  Cache: ... for "The" + ... for "cat"
  = 16 vectors stored

Generating "sat" (step 3):
  Cache grows again...
  = 24 vectors stored

After all 6 tokens:
  Cache = 6 tokens × 4 heads × 2 (K+V) = 48 vectors
```

Now imagine doing this for **thousands of tokens**, with **32 heads**, across **32 layers**:

```
4096 tokens × 32 heads × 2 × 32 layers × 128 dims × 4 bytes ≈ several GB

The GPU spends most of its time reading/writing this huge cache
→ bottleneck is MEMORY BANDWIDTH, not computation
→ inference is SLOW
```

---

## PART 2: Multi-Query Attention (MQA)

### The Idea

Keep all 4 Query heads. But collapse to **just 1 Key head and 1 Value head**, shared
by every query.

```
                Head 1        Head 2        Head 3        Head 4
                ──────        ──────        ──────        ──────
"The"      Q1             Q2            Q3            Q4
           K (shared)     K (shared)    K (shared)    K (shared)
           V (shared)     V (shared)    V (shared)    V (shared)

Each token produces:  4 Queries + 1 Key + 1 Value
                      = 6 vectors per token  (was 12 in MHA)
```

### How Attention Works for "cat" in MQA

```
All 4 heads look at the SAME Keys:

Head 1 — "cat"Q₁ · "sat"K  = 0.8  → attends to "sat"  (subject-verb pattern)
Head 2 — "cat"Q₂ · "sat"K  = 0.7  → also attends to "sat"
Head 3 — "cat"Q₃ · "sat"K  = 0.7  → also attends to "sat"
Head 4 — "cat"Q₄ · "sat"K  = 0.7  → also attends to "sat"

Problem: all 4 heads are asking different questions (different Q matrices)
but looking at the same K — they all end up with similar attention patterns.

Head 1 wanted to find the verb.          ✓ K is general enough to find it
Head 2 wanted to find article modifiers. ✗ But K isn't specialised for that
Head 3 wanted positional patterns.       ✗ K isn't specialised for that either

→ Less diverse attention = quality drops
```

### The KV Cache After MQA

```
After all 6 tokens:
  Cache = 6 tokens × 1 head × 2 (K+V) = 12 vectors  (was 48 in MHA)

That's 4× smaller → GPU reads/writes 4× less → much faster inference

But the quality tradeoff is real — all heads share the same K and V,
so the model can't capture as many different types of relationships.
```

---

## PART 3: Grouped-Query Attention (GQA)

### The Idea

The middle ground. Split 4 query heads into **2 groups of 2**.
Each group gets its **own** K and V head.

```
Group 1: Q₁  Q₂   →  share  K₁  V₁
Group 2: Q₃  Q₄   →  share  K₂  V₂
```

Two K heads and two V heads — more diverse than MQA's one, cheaper than MHA's four.

### For Each Token, We Compute

```
                Group 1               Group 2
                ───────────────       ───────────────
"The"      Q1  Q2  |  K₁  V₁    Q3  Q4  |  K₂  V₂
"cat"      Q1  Q2  |  K₁  V₁    Q3  Q4  |  K₂  V₂
"sat"      Q1  Q2  |  K₁  V₁    Q3  Q4  |  K₂  V₂
"on"       Q1  Q2  |  K₁  V₁    Q3  Q4  |  K₂  V₂
"the"      Q1  Q2  |  K₁  V₁    Q3  Q4  |  K₂  V₂
"mat"      Q1  Q2  |  K₁  V₁    Q3  Q4  |  K₂  V₂

Each token produces:  4 Queries + 2 Keys + 2 Values
                      = 8 vectors per token  (MHA was 12, MQA was 6)
```

### How Attention Works for "cat" in GQA

```
Group 1 (Q₁ and Q₂ share K₁):
  Head 1 — "cat"Q₁ · "sat"K₁  = 0.8  → attends strongly to "sat"
  Head 2 — "cat"Q₂ · "sat"K₁  = 0.7  → also attends to "sat"
             ↑ both use K₁, but their Queries are different
             ↑ so they weight the V differently even though K is shared

Group 2 (Q₃ and Q₄ share K₂):
  Head 3 — "cat"Q₃ · "The"K₂  = 0.7  → attends to "The" (article-noun)
  Head 4 — "cat"Q₄ · "The"K₂  = 0.6  → also finds "The"

Key insight:
  Group 1 (K₁) might specialise in verb/subject relationships
  Group 2 (K₂) might specialise in modifier/noun relationships
  → More diverse than MQA, where 1 K can't specialise at all
  → Near MHA quality, because most important patterns are covered
```

### The KV Cache After GQA

```
After all 6 tokens:
  Cache = 6 tokens × 2 heads × 2 (K+V) = 24 vectors
                                          (MHA was 48, MQA was 12)

2× smaller than MHA → meaningfully faster inference
2× larger than MQA  → slightly slower than MQA, but better quality
```

---

## PART 4: Side-by-Side Comparison

### Vectors Produced Per Token

```
              Queries    Keys    Values    Total vectors
              ───────    ────    ──────    ─────────────
MHA (4 heads)    4         4        4           12
MQA (1 KV head)  4         1        1            6
GQA (2 KV heads) 4         2        2            8
```

### KV Cache Size for the Full Sentence (6 tokens)

```
              K+V vectors    Relative size
              ───────────    ─────────────
MHA              48           ████████████  (baseline)
GQA              24           ██████        (2× smaller)
MQA              12           ███           (4× smaller)
```

### What "cat" Attends To in Each Approach

```
              Head 1         Head 2         Head 3         Head 4
              ──────         ──────         ──────         ──────
MHA           "sat" (0.8)    "The"  (0.7)   "mat"  (0.6)   "on"   (0.5)
              verb pattern   article pat.   object pat.    prep. pat.
              (own K₁)       (own K₂)       (own K₃)       (own K₄)

MQA           "sat" (0.8)    "sat"  (0.7)   "sat"  (0.7)   "sat"  (0.7)
              all heads find similar patterns — diversity lost
              (all share 1 K)

GQA           "sat" (0.8)    "sat"  (0.7)   "The"  (0.7)   "The"  (0.6)
              group 1 finds verb pattern     group 2 finds article pattern
              (share K₁)                    (share K₂)
              → two distinct patterns preserved
```

### Quality vs Speed Tradeoff

```
Quality
  │
  │  MHA ●
  │       \
  │    GQA ●
  │          \
  │       MQA ●
  │
  └────────────────────────  Inference Speed (faster →)
       Slow    Medium   Fast
```

---

## PART 5: The Uptraining Method (Converting MHA → GQA Cheaply)

Say you already have a trained MHA model. Retraining from scratch as GQA would be expensive.
The paper's trick:

### Step 1 — Group the existing KV heads

```
You have 4 trained K heads from MHA:   K₁  K₂  K₃  K₄

Split into 2 groups:
  Group 1: K₁, K₂
  Group 2: K₃, K₄
```

### Step 2 — Mean pool each group into one head

```
New K for Group 1 = (K₁ + K₂) / 2   ← average of the two
New K for Group 2 = (K₃ + K₄) / 2

Do the same for V heads.

Why average? It's the best neutral starting point.
Better than picking one (biased) or random init (wasteful).
```

Visualised:

```
Before (MHA):   K₁   K₂   K₃   K₄
                 \   /       \   /
                  avg         avg
After (GQA):    K_g1         K_g2
```

### Step 3 — Uptrain for 5% of original compute

```
Original training:  100 billion tokens  (full pre-train)
Uptraining:           5 billion tokens  (just 5%)

The model quickly learns to make good use of the pooled heads.
Quality recovers to near-MHA levels.

Result: 20× cheaper than training GQA from scratch.
```

---

## PART 6: Final Summary

```
┌────────────────────┬─────────────────────────────┬──────────────────────────────┐
│                    │ What happens to K and V?     │ Effect                       │
├────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ MHA                │ Every query head has its own │ Best quality.                │
│ (Multi-Head)       │ K and V.                     │ Largest KV cache. Slow.      │
├────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ MQA                │ All query heads share ONE    │ Fastest inference.           │
│ (Multi-Query)      │ K and V.                     │ Smallest cache. Quality dip. │
├────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ GQA                │ Query heads split into G     │ Near-MHA quality.            │
│ (Grouped-Query)    │ groups. Each group shares    │ Near-MQA speed.              │
│                    │ one K and V.                 │ The practical sweet spot.    │
└────────────────────┴─────────────────────────────┴──────────────────────────────┘

GQA is the answer to: "Can we have most of the quality AND most of the speed?"
Answer: Yes — by sharing K and V within groups, not across all heads.
```
