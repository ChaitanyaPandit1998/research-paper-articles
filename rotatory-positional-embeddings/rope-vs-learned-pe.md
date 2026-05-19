# RoPE vs Learned Positional Embeddings — Side-by-Side Walkthrough

Same sentence throughout: **"The cat sat on the mat"**

---

## What Problem Are Both Solving?

A Transformer processes all words at the same time (not left to right like a human).
Without position encoding, it sees a bag of words — no order at all.

```
Without position info, these look IDENTICAL to the model:
"The cat sat on the mat"
"mat the on sat cat The"
```

Both approaches fix this. But they fix it very differently.

---

# PART 1: Learned Positional Embeddings

## The Core Idea

Create a **lookup table** with one row per position.
Each row is a vector of numbers that the model **learns during training**.
Add that row to the word's embedding.

```
Lookup table (learned, one row per position slot):

Position 0:  [ 0.12,  0.85, -0.33,  0.67,  0.44, -0.21,  0.93,  0.11 ]
Position 1:  [-0.45,  0.32,  0.78, -0.56,  0.23,  0.88, -0.14,  0.67 ]
Position 2:  [ 0.91, -0.44,  0.23,  0.81, -0.67,  0.35,  0.52, -0.29 ]
Position 3:  [-0.23,  0.67, -0.89,  0.34,  0.78, -0.56,  0.21,  0.44 ]
Position 4:  [ 0.55, -0.78,  0.44, -0.23,  0.91,  0.12, -0.67,  0.83 ]
Position 5:  [-0.88,  0.23,  0.56, -0.91,  0.34,  0.77, -0.43,  0.19 ]

These numbers start random and get adjusted during training.
The model figures out "useful" values through backpropagation.
```

---

## Step-by-Step: Processing "cat" at Position 1

### Step 1 — Look up the word embedding for "cat"
```
Word embedding for "cat" (from the word embedding table):
[ 0.40,  0.60,  0.20,  0.80,  0.30,  0.50,  0.70,  0.10 ]

This encodes what "cat" MEANS — a small furry animal, etc.
It has nothing to do with position yet.
```

### Step 2 — Look up the positional embedding for position 1
```
Position 1 row from the lookup table:
[-0.45,  0.32,  0.78, -0.56,  0.23,  0.88, -0.14,  0.67 ]

This encodes "second slot in the sequence."
```

### Step 3 — ADD them together
```
Word embedding:       [ 0.40,  0.60,  0.20,  0.80,  0.30,  0.50,  0.70,  0.10 ]
Positional embedding: [-0.45,  0.32,  0.78, -0.56,  0.23,  0.88, -0.14,  0.67 ]
                       ─────────────────────────────────────────────────────────
Sum (new "cat"):      [-0.05,  0.92,  0.98,  0.24,  0.53,  1.38,  0.56,  0.77 ]
```

That's it. "cat" is now encoded as "the word cat, sitting at position 1."

---

## Step-by-Step: Processing "mat" at Position 5

### Step 1 — Word embedding for "mat"
```
[ 0.30,  0.70,  0.40,  0.50,  0.20,  0.60,  0.80,  0.30 ]
```

### Step 2 — Positional embedding for position 5
```
[-0.88,  0.23,  0.56, -0.91,  0.34,  0.77, -0.43,  0.19 ]
```

### Step 3 — ADD them together
```
Word embedding:       [ 0.30,  0.70,  0.40,  0.50,  0.20,  0.60,  0.80,  0.30 ]
Positional embedding: [-0.88,  0.23,  0.56, -0.91,  0.34,  0.77, -0.43,  0.19 ]
                       ─────────────────────────────────────────────────────────
Sum (new "mat"):      [-0.58,  0.93,  0.96, -0.41,  0.54,  1.37,  0.37,  0.49 ]
```

---

## The Dot Product: Comparing "cat" and "mat"

The attention mechanism computes a dot product between the modified embeddings:

```
"cat" (after adding pos embedding): [-0.05,  0.92,  0.98,  0.24,  0.53,  1.38,  0.56,  0.77]
"mat" (after adding pos embedding): [-0.58,  0.93,  0.96, -0.41,  0.54,  1.37,  0.37,  0.49]

Multiply each pair:
  -0.05 × -0.58 =  0.029
   0.92 ×  0.93 =  0.856
   0.98 ×  0.96 =  0.941
   0.24 × -0.41 = -0.098
   0.53 ×  0.54 =  0.286
   1.38 ×  1.37 =  1.891
   0.56 ×  0.37 =  0.207
   0.77 ×  0.49 =  0.377

Sum = 0.029 + 0.856 + 0.941 − 0.098 + 0.286 + 1.891 + 0.207 + 0.377
    = 4.489
```

**The problem:** That number 4.489 mixes together:
- What "cat" means as a word
- What position 1 means
- What "mat" means as a word
- What position 5 means

The model has to **learn from millions of examples** that positions 1 and 5 are 4 apart.
There is nothing in the math that tells it directly. The relative distance is not built in.

---

## The 3 Big Problems with Learned Positional Embeddings

### Problem 1 — Hard cap at training length

```
If your lookup table has 512 rows (positions 0–511):

  "The cat sat on the mat"  → fine (6 words, well within 512)

  A 600-word document       → BROKEN
                              Position 512 doesn't exist in the table.
                              Model has no idea what to do.
```

You simply cannot use the model on longer text than it was trained for.

### Problem 2 — No built-in sense of distance

```
The model sees positional embedding for position 1:
  [-0.45,  0.32,  0.78, -0.56, ...]

And positional embedding for position 5:
  [-0.88,  0.23,  0.56, -0.91, ...]

These are just two independent vectors.
There is NOTHING in their values that says "these are 4 apart."

The model has to figure out the relationship between every pair of positions
purely from seeing enough training examples.
Compare: position 0 vs 1, vs 2, vs 3... for a 512-length model,
that's 512 × 512 = 262,144 pairs to implicitly learn.
```

### Problem 3 — Extra parameters to train and store

```
For a model with:
  max sequence length = 512
  embedding dimension = 768

Lookup table size = 512 × 768 = 393,216 parameters

These add to model size and require training data to learn well.
Rare positions (e.g. 490, 491, 492...) appear in very few training
examples so their embeddings are poorly learned.
```

---

# PART 2: RoPE — How It Fixes All Three Problems

## The Core Idea

Don't add a position vector. Instead, **rotate the word's Query and Key vectors**
by an angle that depends on position. The relative distance then falls out of the
dot product automatically — no learning needed.

---

## Step-by-Step: Processing "cat" at Position 1 with RoPE

### Step 1 — Start with the word embedding for "cat"
```
[ 0.40,  0.60,  0.20,  0.80,  0.30,  0.50,  0.70,  0.10 ]
```

No positional table. No addition. Instead:

### Step 2 — Calculate rotation angles using θ values

```
θ₀ = 1.0,  θ₁ = 0.1,  θ₂ = 0.01,  θ₃ = 0.001

Angles for position 1:
  Pair 0: 1 × 1.0   = 1.000 rad
  Pair 1: 1 × 0.1   = 0.100 rad
  Pair 2: 1 × 0.01  = 0.010 rad
  Pair 3: 1 × 0.001 = 0.001 rad
```

### Step 3 — ROTATE each pair (instead of adding)

```
Rotation formula: [x, y] → [x·cos(α) − y·sin(α),  x·sin(α) + y·cos(α)]

Pair 0 [0.40, 0.60] rotated by 1.0 rad → [−0.289,  0.661]
Pair 1 [0.20, 0.80] rotated by 0.1 rad → [ 0.119,  0.816]
Pair 2 [0.30, 0.50] rotated by 0.01 rad→ [ 0.295,  0.503]
Pair 3 [0.70, 0.10] rotated by 0.001rad → [ 0.700,  0.101]

"cat" after RoPE: [-0.289,  0.661,  0.119,  0.816,  0.295,  0.503,  0.700,  0.101]
```

---

## Step-by-Step: Processing "mat" at Position 5 with RoPE

### Step 1 — Word embedding for "mat"
```
[ 0.30,  0.70,  0.40,  0.50,  0.20,  0.60,  0.80,  0.30 ]
```

### Step 2 — Rotation angles for position 5
```
  Pair 0: 5 × 1.0   = 5.000 rad
  Pair 1: 5 × 0.1   = 0.500 rad
  Pair 2: 5 × 0.01  = 0.050 rad
  Pair 3: 5 × 0.001 = 0.005 rad
```

### Step 3 — Rotate each pair
```
Pair 0 [0.30, 0.70] rotated by 5.0 rad → [ 0.756, −0.089]
Pair 1 [0.40, 0.50] rotated by 0.5 rad → [ 0.111,  0.631]
Pair 2 [0.20, 0.60] rotated by 0.05 rad→ [ 0.170,  0.609]
Pair 3 [0.80, 0.30] rotated by 0.005rad → [ 0.799,  0.304]

"mat" after RoPE: [0.756, −0.089,  0.111,  0.631,  0.170,  0.609,  0.799,  0.304]
```

---

## The Dot Product: Comparing "cat" and "mat" with RoPE

```
"cat" rotated:  [-0.289,  0.661,  0.119,  0.816,  0.295,  0.503,  0.700,  0.101]
"mat" rotated:  [ 0.756, -0.089,  0.111,  0.631,  0.170,  0.609,  0.799,  0.304]

Multiply each pair:
  -0.289 ×  0.756 = −0.219
   0.661 × -0.089 = −0.059
   0.119 ×  0.111 =  0.013
   0.816 ×  0.631 =  0.515
   0.295 ×  0.170 =  0.050
   0.503 ×  0.609 =  0.306
   0.700 ×  0.799 =  0.559
   0.101 ×  0.304 =  0.031

Sum = 1.196
```

For one pair, we can prove this only depends on the gap:

```
Dot product of pair 0:
= cos(1.0) × cos(5.0)  +  sin(1.0) × sin(5.0)
= cos(5.0 − 1.0)           ← trig identity: cos(A)cos(B) + sin(A)sin(B) = cos(B−A)
= cos(4.0)                 ← only the GAP (5−1=4) survives
```

**The model doesn't need to learn this. The math guarantees it.**

---

# PART 3: Side-by-Side Comparison

## How Position Gets Into the Embedding

```
LEARNED PE                          ROPE
──────────────────────────────      ──────────────────────────────────────────
Look up a row in a table            Calculate angles = position × θ
Add the row to word embedding       Rotate the word embedding by those angles
Result: word + position mixed in    Result: word rotated by position amount
```

## How Relative Distance Is Encoded

```
LEARNED PE                          ROPE
──────────────────────────────      ──────────────────────────────────────────
Not encoded directly.               Encoded mathematically.

The model sees:                     The dot product always gives:
  pos 1 embedding = [...]             cos((pos_A − pos_B) × θ)
  pos 5 embedding = [...]
  (just two independent vectors)    The gap falls out of the trig identity.
  
The model must LEARN from data      The model gets it for FREE from the math.
that these are 4 apart.
```

## What Happens with Sequences Longer Than Training Length

```
LEARNED PE                          ROPE
──────────────────────────────      ──────────────────────────────────────────
Trained on max 512 positions?       No lookup table — just multiply position × θ

  Position 600 → NOT IN TABLE       Position 600 → 600 × 1.0 = 600 rad ✓
               → model crashes                  → 600 × 0.1 = 60 rad  ✓
                                                → works fine
```

## Parameters Required

```
LEARNED PE                          ROPE
──────────────────────────────      ──────────────────────────────────────────
512 positions × 768 dims            ZERO extra parameters
= 393,216 numbers to store          θ values are computed from a formula,
  and train                         not learned
```

## Full Comparison Table

| | Learned PE | RoPE |
|---|---|---|
| How position is added | Add a learned vector | Rotate the existing vector |
| Relative distance | Must be learned from data | Built into the math |
| Max sequence length | Hard cap (e.g. 512) | No cap — any length works |
| Extra parameters | Yes (pos table) | None |
| Works beyond training length | No | Yes |
| Position 0 behaviour | Adds a learned vector | No rotation (angle = 0) |
| Adopted in modern LLMs | BERT, early GPT | LLaMA, Mistral, Gemma, most modern models |

---

## The Best Analogy

**Learned PE** is like giving each seat in a cinema a **name badge**.
Seat 1 gets badge "A", seat 2 gets badge "B", etc.
You can tell seats apart, but if someone asks "how far is seat 1 from seat 5?"
you have to count manually — the badges don't tell you.
And if you add a new seat 513? There's no badge for it.

**RoPE** is like a **compass bearing**.
Position 1 = face 1° North. Position 5 = face 5° North.
The difference between any two seats is just arithmetic: 5° − 1° = 4°.
You can always calculate it. And position 600? Just face 600° — no problem.
