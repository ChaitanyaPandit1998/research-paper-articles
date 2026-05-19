# ReLU² Activation

**What it is:** The activation function in the MLP (feed-forward) layers is `relu(x)²` — ReLU applied first, then squared.

**Code:** `nanochat/gpt.py:137`

```python
# gpt.py:129-139
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()   # ← ReLU², line 137
        x = self.c_proj(x)
        return x
```

---

## What Is an Activation Function and Why Does It Matter?

A Transformer's MLP layer looks like this:

```
input → Linear (expand) → Activation → Linear (shrink) → output
```

The activation function sits in the middle. Its job is to introduce **non-linearity** — without it, stacking linear layers is just one big linear layer, and the model can't learn complex patterns.

The activation also decides which neurons "fire" (pass information forward) and which stay silent.

---

## The Activation Function Family

```
Input x = [-2, -1, 0, 0.5, 1, 2, 3]

ReLU(x)     = max(0, x)
            = [0, 0, 0, 0.5, 1, 2, 3]
              Negative → dead. Positive → pass through unchanged.

GELU(x)     ≈ x × Φ(x)    (Φ = Gaussian CDF)
            = [-0.05, -0.16, 0, 0.35, 0.84, 1.95, 2.99]
              Smooth, never exactly zero. Used in GPT-2, BERT.

SiLU(x)     = x × sigmoid(x)
            = [-0.24, -0.27, 0, 0.31, 0.73, 1.76, 2.86]
              Also smooth. Used in LLaMA, Mistral.

ReLU²(x)    = max(0, x)²
            = [0, 0, 0, 0.25, 1, 4, 9]
              Negative → dead. Positive → squared (grows faster).
```

---

## Step-by-Step: What ReLU² Does Differently

### Step 1 — Apply ReLU (same as standard ReLU)

```
Input neuron value: x = 2.5

ReLU(2.5) = max(0, 2.5) = 2.5     ← positive, passes through
ReLU(-1.3) = max(0, -1.3) = 0     ← negative, killed
```

### Step 2 — Square the result

```
ReLU²(2.5) = 2.5² = 6.25
ReLU²(-1.3) = 0² = 0              ← still dead
```

### The difference in the MLP for "cat"

Say the MLP expands to 4 neurons for "cat":

```
Neuron values after linear: [-0.5,  0.3,  1.2,  2.1]

After ReLU:                  [0,    0.3,  1.2,  2.1]   ← 1 neuron dead
After ReLU²:                 [0,    0.09, 1.44, 4.41]  ← same 1 dead,
                                                           but active neurons
                                                           have amplified contrast
```

ReLU² makes **strong signals stronger** (2.1 → 4.41) and **weak signals weaker** (0.3 → 0.09). It amplifies the gap between "this neuron matters" and "this neuron doesn't."

---

## Three Reasons ReLU² Trains Faster

### 1. More sparsity

Sparsity = fraction of neurons outputting exactly 0.

```
ReLU:   exactly 0 for all x < 0
ReLU²:  exactly 0 for all x ≤ 0 (same boundary, but...)

Why does it matter more? Because ReLU² also effectively kills
near-zero positives (0.01² = 0.0001 ≈ irrelevant).
More neurons are "practically dead" → sparser, cleaner representations.
Sparser = more efficient learning signal.
```

### 2. Stronger gradient for confident neurons

The gradient (how much the neuron updates during backprop) is:

```
ReLU gradient:   1 if x > 0,  0 if x ≤ 0
ReLU² gradient:  2·ReLU(x) = 2x if x > 0,  0 if x ≤ 0
```

When x = 2.1:
```
ReLU  gradient = 1       ← always 1, no matter how confident
ReLU² gradient = 2×2.1 = 4.2  ← larger gradient for large activations
```

Neurons that are "very confident" update more aggressively → faster learning.

### 3. Cheaper to compute than GELU/SiLU

```
GELU needs:  exp(), erf() — expensive transcendental functions
SiLU needs:  sigmoid() = 1/(1+exp()) — also needs exp()
ReLU² needs: max(0,x), x*x — just comparisons and multiplication

On modern hardware: ReLU² is noticeably faster per layer.
```

---

## Why Not Just ReLU?

Plain ReLU passes large values through unchanged. ReLU² amplifies them.

```
For token "cat" at neuron detecting "is a subject noun":
  Raw score: 3.0

  ReLU output:  3.0   → fed to next linear layer with weight 1.0 → 3.0
  ReLU² output: 9.0   → fed to next linear layer with weight 1.0 → 9.0
```

ReLU² gives the model a stronger, clearer signal when a feature is strongly present. This leads to lower loss at the same number of training steps.

---

## Quick Comparison

| | ReLU | GELU | SiLU | **ReLU²** |
|---|---|---|---|---|
| Negative inputs | Zero | Near-zero | Near-zero | **Zero** |
| Positive inputs | Pass through | Slightly gated | Slightly gated | **Squared** |
| Sparsity | Medium | Low | Low | **High** |
| Compute cost | Low | High | High | **Low** |
| Training speed at scale | Baseline | Slower | Slower | **Faster** |

---

## One-Line Summary

> ReLU² kills negative neurons (like ReLU), then squares the positive ones — amplifying strong signals, suppressing weak ones, and making the MLP cheaper and faster to train than GELU or SiLU at the same scale.
