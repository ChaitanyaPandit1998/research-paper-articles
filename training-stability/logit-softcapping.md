# Logit Softcapping

**What it is:** After the final linear layer, logits are squashed through `softcap × tanh(logits / softcap)` with `softcap = 15` — preventing any token's score from growing unboundedly large.

**Code:** `gpt.py:463-467`

```python
# gpt.py:463-467
softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
logits = self.lm_head(x)                        # (B, T, padded_vocab_size)
logits = logits[..., :self.config.vocab_size]   # slice to remove vocab padding
logits = logits.float()                          # switch to fp32 before softcap + loss
logits = softcap * torch.tanh(logits / softcap) # squash the logits
```

---

## What Are Logits?

After the transformer finishes processing, the LM head produces a **logit** for every token in the vocabulary:

```
Vocabulary: ["The", "cat", "sat", "on", "mat", "dog", "ran", ...]
                                          (50,000+ tokens total)

After processing "The cat sat on the ___":
  logit("mat")   = 8.5    ← high score, model thinks "mat" is likely
  logit("floor") = 4.2
  logit("chair") = 3.8
  logit("dog")   = -2.1   ← low score, unlikely
  logit("the")   = 1.4
  ...
```

These logits are fed into softmax to produce probabilities:

```
P(token) = exp(logit) / sum(exp(all logits))
```

---

## The Problem: Logit Explosion

During training, nothing stops logits from growing very large:

```
Early training:     logits ≈ [-2, 1, 3, 8, -1, 2, ...]   ← reasonable range
After many steps:   logits ≈ [-80, 5, 12, 150, -40, ...]  ← exploded!
```

When one logit is 150 and others are in single digits:

```
exp(150) = 1.4 × 10⁶⁵    ← astronomically large
exp(12)  = 162,755
exp(5)   = 148

P("mat") ≈ 1.4×10⁶⁵ / 1.4×10⁶⁵ ≈ 1.000000
P(any other token) ≈ 0.000000
```

The model becomes completely confident about one token. This is **logit explosion** and it causes:
- Training instability (extreme gradients)
- No calibration (model is never uncertain even when it should be)
- Overfitting to memorised outputs

---

## The Fix: Softcapping

### The formula

```
capped_logit = softcap × tanh(logit / softcap)

Where softcap = 15
```

### What tanh does

tanh is an S-shaped function that squashes any input into the range (-1, +1):

```
tanh(-100) ≈ -1.0
tanh(-3)   ≈ -0.995
tanh(-1)   ≈ -0.762
tanh(0)    =  0
tanh(1)    ≈  0.762
tanh(3)    ≈  0.995
tanh(100)  ≈  1.0
```

### Applied to logits with softcap=15

```
capped = 15 × tanh(logit / 15)

logit = 2:    15 × tanh(2/15)   = 15 × tanh(0.133)  = 15 × 0.133  = 1.99
logit = 8:    15 × tanh(8/15)   = 15 × tanh(0.533)  = 15 × 0.487  = 7.31
logit = 15:   15 × tanh(15/15)  = 15 × tanh(1.0)    = 15 × 0.762  = 11.43
logit = 30:   15 × tanh(30/15)  = 15 × tanh(2.0)    = 15 × 0.964  = 14.46
logit = 100:  15 × tanh(100/15) ≈ 15 × 1.0          ≈ 14.99
logit = 150:  15 × tanh(150/15) ≈ 15 × 1.0          ≈ 15.00
```

The capped logit **can never exceed ±15**, no matter how large the raw logit grows.

---

## Step-by-Step Walkthrough: "The cat sat on the ___"

### Without softcapping (after many training steps, weights have grown)

```
Raw logits:
  "mat"   = 150   ← extreme spike
  "floor" = 12
  "chair" = 9
  "dog"   = -80
  "cat"   = 4

After softmax:
  P("mat")   = exp(150) / (exp(150) + exp(12) + ...) ≈ 1.000
  P("floor") ≈ 0.000
  P("chair") ≈ 0.000

The model is 100% certain. Any mistake causes enormous loss.
Gradients explode. Training destabilises.
```

### With softcapping (softcap = 15)

```
Raw logits → Capped logits:
  "mat"   = 150  → 15 × tanh(10)  ≈ 15.00
  "floor" = 12   → 15 × tanh(0.8) ≈ 10.65
  "chair" = 9    → 15 × tanh(0.6) ≈ 8.36
  "dog"   = -80  → 15 × tanh(-5.3) ≈ -14.99
  "cat"   = 4    → 15 × tanh(0.27) ≈ 3.96

After softmax on capped logits:
  P("mat")   = exp(15.00) / total ≈ 0.989
  P("floor") = exp(10.65) / total ≈ 0.010
  P("chair") = exp(8.36)  / total ≈ 0.001
  ...

Still strongly prefers "mat" — model isn't confused.
But P("mat") is 0.989, not 1.000 — some calibrated uncertainty remains.
Gradients are stable. Training continues smoothly.
```

---

## Key Property: Smooth, Not Hard-Clip

There's an alternative approach: hard clipping, `logit = min(max(logit, -15), 15)`.

```
Hard clip at ±15:
  logit = 8  → 8   (unchanged below cap)
  logit = 14 → 14  (unchanged)
  logit = 15 → 15  (clipped, exact boundary)
  logit = 16 → 15  (clipped, same as 15)
  logit = 30 → 15  (clipped, same as 15)
```

At the clipping boundary, the gradient suddenly drops to 0 — a sharp discontinuity that can cause training instability.

Softcapping with tanh is smooth:
```
The gradient of tanh(x) = 1 - tanh(x)² — always positive, never 0

logit = 15: gradient ≈ 0.42   ← still nonzero, gradients flow
logit = 30: gradient ≈ 0.07   ← small but nonzero
logit = 150: gradient ≈ 0.000 ← essentially 0, but smooth approach
```

Training never hits a hard wall — gradients shrink gracefully toward very large logits.

---

## Relationship to Gemma 2

Gemma 2 (Google, 2024) uses the same technique with the same formula and `softcap = 30` for logits. The paper showed it's particularly important at:
- **Long context lengths** — more tokens = more chances for a few logits to spike
- **Large model sizes** — more parameters = more ability to grow extreme logits

`softcap = 15` in this model is more aggressive (tighter cap) than Gemma 2's 30.

---

## One-Line Summary

> Softcapping squashes final logits through a tanh-scaled function so no token can ever receive an unboundedly large score — keeping softmax outputs calibrated and training gradients stable, even when weights grow large.
