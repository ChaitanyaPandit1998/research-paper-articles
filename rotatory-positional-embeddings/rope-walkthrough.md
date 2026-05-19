# RoPE Formula: Complete Step-by-Step Walkthrough

## Our Setup

```
Sentence:   "The  cat  sat  on  the  mat"
Position:     0    1    2    3    4    5

Embedding dimension: d = 8
(means each word is represented by 8 numbers)
(we work in pairs, so we have 4 pairs)
```

---

## STAGE 1: Calculate the θ (Theta) Values

**The formula:**
```
θᵢ = 10000^(−2i / d)

i = which pair (0, 1, 2, 3)
d = total dimensions = 8
```

Work through each pair one at a time:

**Pair 0 (i=0):**
```
θ₀ = 10000^(−2×0 / 8)
   = 10000^(0 / 8)
   = 10000^0
   = 1.0
```

**Pair 1 (i=1):**
```
θ₁ = 10000^(−2×1 / 8)
   = 10000^(−2/8)
   = 10000^(−0.25)
   = 1 / 10000^0.25
   = 1 / 10
   = 0.1
```

**Pair 2 (i=2):**
```
θ₂ = 10000^(−2×2 / 8)
   = 10000^(−4/8)
   = 10000^(−0.5)
   = 1 / 10000^0.5
   = 1 / 100
   = 0.01
```

**Pair 3 (i=3):**
```
θ₃ = 10000^(−2×3 / 8)
   = 10000^(−6/8)
   = 10000^(−0.75)
   = 1 / 10000^0.75
   = 1 / 1000
   = 0.001
```

**Result — θ values (computed once, never change):**
```
Pair 0: θ₀ = 1.0      ← spins fast
Pair 1: θ₁ = 0.1      ← spins slower
Pair 2: θ₂ = 0.01     ← even slower
Pair 3: θ₃ = 0.001    ← barely moves
```

---

## STAGE 2: Calculate Rotation Angles for Every Word

**The formula:**
```
angle = position × θᵢ
```

One row per word, one column per pair:

```
          Pair 0 (θ=1.0)      Pair 1 (θ=0.1)      Pair 2 (θ=0.01)     Pair 3 (θ=0.001)
          ───────────────     ────────────────     ─────────────────    ─────────────────
"The" (0) 0 × 1.0   = 0.000  0 × 0.1   = 0.000   0 × 0.01  = 0.000   0 × 0.001 = 0.000
"cat" (1) 1 × 1.0   = 1.000  1 × 0.1   = 0.100   1 × 0.01  = 0.010   1 × 0.001 = 0.001
"sat" (2) 2 × 1.0   = 2.000  2 × 0.1   = 0.200   2 × 0.01  = 0.020   2 × 0.001 = 0.002
"on"  (3) 3 × 1.0   = 3.000  3 × 0.1   = 0.300   3 × 0.01  = 0.030   3 × 0.001 = 0.003
"the" (4) 4 × 1.0   = 4.000  4 × 0.1   = 0.400   4 × 0.01  = 0.040   4 × 0.001 = 0.004
"mat" (5) 5 × 1.0   = 5.000  5 × 0.1   = 0.500   5 × 0.01  = 0.050   5 × 0.001 = 0.005
```

> These angles are in **radians** (not degrees).

---

## STAGE 3: Apply the Rotation to "cat"

**"cat" is at position 1. Its embedding (8 numbers, 4 pairs):**
```
[ 0.4,  0.6,  |  0.2,  0.8,  |  0.3,  0.5,  |  0.7,  0.1 ]
 └──── pair 0 ──┘  └──── pair 1 ──┘  └──── pair 2 ──┘  └──── pair 3 ──┘
```

**The rotation formula for each pair:**
```
Given a pair [x, y] and an angle α:

x_new = x × cos(α) − y × sin(α)
y_new = x × sin(α) + y × cos(α)
```

---

### Pair 0 — angle = 1.0 radian

```
cos(1.0) = 0.5403
sin(1.0) = 0.8415

Input: x = 0.4,  y = 0.6

x_new = 0.4 × 0.5403  −  0.6 × 0.8415
      = 0.2161         −  0.5049
      = −0.289

y_new = 0.4 × 0.8415  +  0.6 × 0.5403
      = 0.3366         +  0.3242
      =  0.661
```

---

### Pair 1 — angle = 0.1 radian

```
cos(0.1) = 0.9950
sin(0.1) = 0.0998

Input: x = 0.2,  y = 0.8

x_new = 0.2 × 0.9950  −  0.8 × 0.0998
      = 0.1990         −  0.0798
      =  0.119

y_new = 0.2 × 0.0998  +  0.8 × 0.9950
      = 0.0200         +  0.7960
      =  0.816
```

---

### Pair 2 — angle = 0.01 radian

```
cos(0.01) = 0.9999
sin(0.01) = 0.0100

Input: x = 0.3,  y = 0.5

x_new = 0.3 × 0.9999  −  0.5 × 0.0100
      = 0.2999         −  0.0050
      =  0.295   ← barely changed

y_new = 0.3 × 0.0100  +  0.5 × 0.9999
      = 0.0030         +  0.4999
      =  0.503   ← barely changed
```

---

### Pair 3 — angle = 0.001 radian

```
cos(0.001) = 1.0000
sin(0.001) = 0.0010

Input: x = 0.7,  y = 0.1

x_new = 0.7 × 1.0000  −  0.1 × 0.0010
      = 0.7000         −  0.0001
      =  0.700   ← almost identical

y_new = 0.7 × 0.0010  +  0.1 × 1.0000
      = 0.0007         +  0.1000
      =  0.101   ← almost identical
```

---

### "cat" before and after RoPE

```
Original:           [ 0.400,  0.600,  0.200,  0.800,  0.300,  0.500,  0.700,  0.100 ]
After RoPE (pos 1): [-0.289,  0.661,  0.119,  0.816,  0.295,  0.503,  0.700,  0.101 ]

Pair 0 changed a LOT  → big angle (1.0 rad)
Pair 1 changed a bit  → smaller angle (0.1 rad)
Pair 2 barely changed → tiny angle (0.01 rad)
Pair 3 almost same    → near-zero angle (0.001 rad)
```

---

## STAGE 4: Apply the Rotation to "mat"

**"mat" is at position 5. Its embedding:**
```
[ 0.3,  0.7,  |  0.4,  0.5,  |  0.2,  0.6,  |  0.8,  0.3 ]
```

Angles for position 5: `[5.0, 0.5, 0.05, 0.005]`

---

### Pair 0 — angle = 5.0 radians

```
cos(5.0) =  0.2837
sin(5.0) = −0.9589

Input: x = 0.3,  y = 0.7

x_new = 0.3 ×  0.2837  −  0.7 × (−0.9589)
      = 0.0851           +  0.6712
      =  0.756

y_new = 0.3 × (−0.9589) +  0.7 ×  0.2837
      = −0.2877          +  0.1986
      = −0.089
```

### Pair 1 — angle = 0.5 radian

```
cos(0.5) = 0.8776
sin(0.5) = 0.4794

Input: x = 0.4,  y = 0.5

x_new = 0.4 × 0.8776  −  0.5 × 0.4794
      = 0.3510         −  0.2397
      =  0.111

y_new = 0.4 × 0.4794  +  0.5 × 0.8776
      = 0.1918         +  0.4388
      =  0.631
```

### Pair 2 — angle = 0.05 radian

```
cos(0.05) = 0.9988
sin(0.05) = 0.0500

Input: x = 0.2,  y = 0.6

x_new = 0.2 × 0.9988  −  0.6 × 0.0500
      = 0.1998         −  0.0300
      =  0.170

y_new = 0.2 × 0.0500  +  0.6 × 0.9988
      = 0.0100         +  0.5993
      =  0.609
```

### Pair 3 — angle = 0.005 radian

```
cos(0.005) = 1.0000
sin(0.005) = 0.0050

Input: x = 0.8,  y = 0.3

x_new = 0.8 × 1.0000  −  0.3 × 0.0050
      = 0.8000         −  0.0015
      =  0.799

y_new = 0.8 × 0.0050  +  0.3 × 1.0000
      = 0.0040         +  0.3000
      =  0.304
```

### "mat" before and after RoPE

```
Original:           [ 0.300,  0.700,  0.400,  0.500,  0.200,  0.600,  0.800,  0.300 ]
After RoPE (pos 5): [ 0.756, −0.089,  0.111,  0.631,  0.170,  0.609,  0.799,  0.304 ]
```

---

## STAGE 5: The Dot Product — Where the Magic Happens

The model now computes attention between **"cat" and "mat"** by taking the dot product of their rotated embeddings.

**Dot product = multiply matching numbers, then add them all up:**

```
"cat" rotated:  [-0.289,  0.661,  0.119,  0.816,  0.295,  0.503,  0.700,  0.101]
"mat" rotated:  [ 0.756, -0.089,  0.111,  0.631,  0.170,  0.609,  0.799,  0.304]

Multiply each position:
  -0.289 × 0.756  = −0.219
   0.661 × −0.089 = −0.059
   0.119 × 0.111  =  0.013
   0.816 × 0.631  =  0.515
   0.295 × 0.170  =  0.050
   0.503 × 0.609  =  0.306
   0.700 × 0.799  =  0.559
   0.101 × 0.304  =  0.031

Sum = −0.219 − 0.059 + 0.013 + 0.515 + 0.050 + 0.306 + 0.559 + 0.031
    = 1.196
```

That number **1.196 encodes the fact that "cat" and "mat" are 4 positions apart** — not that "cat" is at position 1 or "mat" is at position 5. The absolute positions dissolved into the result. Only the gap survived.

---

## Why the Gap Survives — The Trig Proof (Simple Version)

For a single pair, if we rotate by angle `α` (cat) and `β` (mat), the dot product is:

```
cos(α) × cos(β)  +  sin(α) × sin(β)
```

There is a trig identity that says:

```
cos(A)cos(B) + sin(A)sin(B) = cos(B − A)
```

So the dot product becomes:

```
cos(β − α)
= cos(5.0 − 1.0)
= cos(4.0)          ← depends ONLY on the gap (4), not on 1 or 5
```

The individual positions (1 and 5) cancel out. The distance (4) is all that's left.

---

## Full Sentence Summary

```
WORD      POSITION   ANGLES (radians)                BEHAVIOUR
────────  ────────   ────────────────────────────    ──────────────────────────────────
"The"        0       [0.000, 0.000, 0.000, 0.000]   No rotation. Embedding unchanged.
"cat"        1       [1.000, 0.100, 0.010, 0.001]   Pair 0 spins a lot. Others barely.
"sat"        2       [2.000, 0.200, 0.020, 0.002]   Pair 0 spins 2× more than "cat".
"on"         3       [3.000, 0.300, 0.030, 0.003]   And so on...
"the"        4       [4.000, 0.400, 0.040, 0.004]
"mat"        5       [5.000, 0.500, 0.050, 0.005]

When any two words are compared in attention:
  → their angles subtract
  → result = (position gap) × θᵢ
  → model knows how far apart they are
  → absolute positions are never seen directly
```

---

## Key Takeaways

| What | Why it matters |
|---|---|
| θ decreases with dimension | Early pairs spin fast (local detail), late pairs spin slow (long-range structure) |
| Position 0 has zero angles | First word is the reference — no rotation applied |
| Same word at different positions → different embedding | Model can tell "The" at pos 0 from "the" at pos 4 |
| Dot product of two rotated vectors = cos(gap × θ) | Relative distance encoded automatically, no extra parameters |
| θ values never change | Computed once at model init, baked into the math forever |
