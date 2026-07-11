# Activation Functions, Demystified

The single line of nonlinear math that separates a neural network from a spreadsheet — traced from sigmoid to the gated units running inside GPT, Llama, and Gemini.

## Table of contents

1. [What are they?](#1-what-are-they)
2. [Where they live in a network](#2-where-they-live-in-a-network)
3. [The functions in active use](#3-the-functions-in-active-use)
4. [A runnable PyTorch comparison](#4-a-runnable-pytorch-comparison)
5. [Summary table](#5-summary-table)
6. [Key takeaways](#6-key-takeaways)

---

## 1. What are they?

A network without activation functions is just linear algebra pretending to be deep.

An activation function is a small, fixed, elementwise function that a layer applies to its output before passing it to the next layer. Every neuron computes a weighted sum of its inputs — `z = Wx + b` — and the activation function decides how that raw sum gets translated into a signal the rest of the network actually uses. It takes one number in and returns one number out, applied independently to every element of the layer's output.

The reason this matters more than its modest description suggests: without it, depth is an illusion. Stack any number of purely linear layers and the result collapses algebraically into a single linear layer. Watch it happen with two layers:

```
y = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = W′x + b′
```

Two matrix multiplications compose into one matrix multiplication. A hundred linear layers still compose into one. No amount of stacking linear transformations lets a network learn anything a single linear regression couldn't already express — it can't bend a decision boundary, can't separate an XOR pattern, can't approximate a curve. Insert a nonlinear function between the layers, and that collapse stops happening: `W₂ f(W₁x + b₁) + b₂` cannot be reduced to a single matrix multiply, so each additional layer genuinely adds representational power. This is the entire reason activation functions exist — everything else in this article is about which nonlinearity to pick and why it matters.

## 2. Where they live in a network

Activation functions sit between the linear projections inside every feed-forward block: a multi-layer perceptron (MLP), a transformer's feed-forward network (FFN), a convolutional filter bank. The pattern is always *linear → nonlinear → linear*: project the input into a (usually wider) hidden space, bend it with an activation, then project back down.

```
input x  →  linear (W₁x + b₁)  →  activation f(·)  →  linear (W₂h + b₂)  →  output
```

In a transformer, this block — self-attention, then an FFN with an activation in the middle — is what actually holds most of the parameters. Attention decides *which* tokens to look at; the FFN, activation included, is where the model does the bulk of its per-token computation and is generally believed to store much of what a large language model "knows."

Two separate things ride on the choice of activation function, and it's worth keeping them distinct:

**Expressiveness.** A network's ability to approximate complicated functions comes entirely from the nonlinearities between its linear layers. Different activation shapes carve up input space differently — a smooth S-curve saturates gently, a hinge function creates a sharp fold — and that shape becomes part of what the network can represent.

**Gradient flow.** Every activation function's derivative becomes a multiplicative factor in backpropagation. A network might be twenty, fifty, or a hundred layers deep, and the gradient signal for the earliest layers has to survive being multiplied by the derivative of the activation function at *every* layer it passes back through. If that derivative is routinely small (as with sigmoid) or exactly zero (as with ReLU on negative inputs), gradients shrink or vanish before they ever reach the early layers, and training stalls. This is not a minor implementation detail — it is the reason the field moved from sigmoid/tanh to ReLU, and later to the smoother, gated variants covered below.

## 3. The functions in active use

Seven functions, three eras: the saturating classics, the ReLU family that made deep networks trainable, and the smooth gated units that now run inside every major LLM.

### Sigmoid *(logistic function, 1980s–2000s)*

**Intuition first.** Think of a dimmer switch with a soft floor and a soft ceiling: no matter how hard you push the input, the output only ever eases toward 0 or toward 1. Historically it was used to model something like a probability, or a biological neuron easing from "off" to "firing."

```
σ(x) = 1 / (1 + e^(−x))
```

The shape is a smooth S: flat near both extremes, steepest at `x = 0`, output squeezed permanently into `(0, 1)`. For large positive `x` it saturates at 1; for large negative `x` it saturates at 0. That squeezing was the whole appeal — it turns any real number into something that reads like a probability — but the same squeezing is its downfall: the derivative `σ(x)(1−σ(x))` maxes out at just 0.25 and shrinks toward zero at both tails, so gradients flowing backward through several sigmoid layers shrink geometrically. That's the vanishing-gradient problem that made deep sigmoid networks nearly untrainable.

*Used in:* still standard as an **output-layer** gate for binary classification and inside LSTM/GRU gates — just rarely used to activate hidden layers anymore.

- **Pros:** output bounded in (0,1), reads naturally as a probability; smooth and differentiable everywhere.
- **Cons:** vanishing gradients away from x = 0; not zero-centered, which slows optimization; involves an exponential — costlier than a hinge function.

### Tanh *(hyperbolic tangent, 1990s–2010s)*

**Intuition first.** Tanh is sigmoid's better-balanced sibling — the same soft S-curve, but stretched and shifted so it eases between −1 and 1 instead of 0 and 1, passing through the origin. That "zero-centered" property sounds cosmetic but isn't: when a layer's outputs are centered around zero rather than all positive, the gradient updates to the next layer's weights don't get stuck pushing consistently in one direction.

```
tanh(x) = (e^x − e^(−x)) / (e^x + e^(−x)) = 2σ(2x) − 1
```

Same S-shape as sigmoid, same saturating tails, same vanishing-gradient failure mode for deep stacks — it's a rescaled sigmoid, so the math inherits both the strength and the weakness. It became the default hidden-layer activation for RNNs (and still is, inside LSTM/GRU cell states) precisely because zero-centering made optimization noticeably more stable than sigmoid, even though the saturation problem never went away.

*Used in:* dominant inside **recurrent** architectures (LSTM, GRU cell/hidden states); largely superseded by ReLU-family functions in feed-forward and transformer blocks.

- **Pros:** zero-centered output — better-behaved gradients than sigmoid; smooth, well-understood, bounded.
- **Cons:** still saturates and still vanishes gradients at the tails; exponential-based — more expensive than ReLU.

### ReLU *(rectified linear unit, 2011–present)*

**Intuition first.** A one-way valve: anything positive passes through completely unchanged, anything negative is clamped to exactly zero. No easing, no curve — just a hinge at the origin.

```
ReLU(x) = max(0, x)
```

The shape is two straight line segments meeting at a corner: flat at zero for all `x ≤ 0`, then a perfect 45° line for `x > 0`. It doesn't saturate on the positive side at all — the gradient there is a constant 1, so it doesn't shrink no matter how deep the stack goes, which is exactly why it unlocked training networks dozens of layers deep. The tradeoff is bluntness: any neuron that lands in negative territory outputs zero and has zero gradient, so if its weights get pushed into a regime where it's *always* negative, it stops updating forever — the "dying ReLU" problem.

*Used in:* the default for CNNs and most feed-forward networks throughout the 2010s; still common today for its simplicity and speed.

- **Pros:** no vanishing gradient for positive inputs; trivially cheap — one comparison, no exponential; induces sparsity (many neurons output exactly 0).
- **Cons:** dying neurons — a permanently-negative unit stops learning; not zero-centered; non-differentiable kink at x = 0 (rarely an issue in practice).

### Leaky ReLU *(2013–present)*

**Intuition first.** The same one-way valve as ReLU, but with a pinhole leak on the closed side — negative inputs aren't zeroed out, they're just heavily throttled, scaled down by a small constant instead of clamped flat.

```
LeakyReLU(x) = x        if x > 0
             = αx        if x ≤ 0    (typically α = 0.01)
```

Visually it's almost indistinguishable from ReLU — the same hinge at the origin — except the left arm isn't flat, it's a very shallow downward line instead of sitting at zero. That tiny slope is the entire fix: a neuron that lands in negative territory still has a (small) nonzero gradient, so it can recover instead of dying permanently. It directly targets ReLU's single biggest failure mode without changing the cost or the positive-side behavior at all.

*Used in:* common in GANs and CNNs where dying units are a known problem; largely bypassed in modern LLMs in favor of GELU/SiLU-family functions.

- **Pros:** fixes dying-ReLU by keeping gradient flow alive for x < 0; just as cheap as ReLU.
- **Cons:** α is an extra hyperparameter to tune (or learn, as in PReLU); empirical gains over plain ReLU are inconsistent.

### GELU *(Gaussian Error Linear Unit, 2016–present)*

**Intuition first.** Where ReLU is a hard valve that's either fully open or fully shut, GELU is a valve with judgment: it weights each input by how likely a standard Gaussian random variable would be to fall below it, so instead of a sharp on/off cutoff at zero, small negative inputs still leak through a little and the transition into "on" is a smooth curve, not a corner.

```
GELU(x) = x · Φ(x)  ≈  0.5x(1 + tanh[√(2/π) · (x + 0.044715x³)])
```

where Φ is the standard normal cumulative distribution function. The shape tracks ReLU's overall trend — near zero for very negative `x`, roughly linear for large positive `x` — but everything around the origin is rounded off: there's a small dip slightly below zero for negative inputs (unlike ReLU's flat clamp) and a smooth curved onset instead of a kink. That smoothness (the function is infinitely differentiable) removes the exact-zero dead zone that causes dying neurons, while the probabilistic framing — "gate each input by how large it is, stochastically" — was designed to combine the regularizing feel of dropout with the shape of an activation function.

*Used in:* the activation used in **GPT-2, BERT**, and most of the first wave of transformer language models.

- **Pros:** smooth everywhere — no dead zone, no hard kink; strong empirical performance in transformers; small negative outputs preserved instead of clamped.
- **Cons:** more expensive than ReLU (erf or tanh approximation); mostly superseded by SiLU/SwiGLU in newer LLMs.

### SiLU / Swish *(Sigmoid Linear Unit, 2017–present)*

**Intuition first.** Take the input and scale it by its own sigmoid — a self-gating valve, where the input decides how open it should be. Large positive numbers gate themselves almost fully open (behaving like ReLU); numbers near zero and slightly negative gate themselves partly closed instead of being cut off outright.

```
SiLU(x) = x · σ(x) = x / (1 + e^(−x))
```

Nearly identical in shape to GELU — both dip slightly below zero just left of the origin, both curve smoothly into a near-linear rise on the right — which is no accident, since both are smoothed relatives of the same ReLU trend. SiLU is a touch cheaper to compute (one sigmoid vs. an erf/tanh approximation) and was discovered largely via automated search over candidate activation functions, then confirmed by hand — a rare case of a simple closed-form function beating hand-designed alternatives at scale.

*Used in:* the gate activation inside **SwiGLU**, used in **Llama, Mistral, Qwen**, and most current open-weight LLM feed-forward blocks.

- **Pros:** smooth, non-monotonic near zero, no dead zone; cheaper than GELU, comparable or better empirical results; self-gating — naturally suited to being used inside gated units.
- **Cons:** not zero-centered; marginally more expensive than plain ReLU.

### The gated variants: SwiGLU and GeGLU

Every function above is a single curve applied to a single number. The gated units used in modern LLM feed-forward blocks are a different construction entirely: instead of one linear projection passed through one activation, the input is projected *twice* by two independent weight matrices, one branch is passed through an activation to act as a gate, and the two branches are multiplied together elementwise before a final projection back down.

```
x → V x  (linear, "content")  ⊗  f(Wx)  (linear + activation, "gate")  →  W₂(·)  → output
```

This is a GLU — Gated Linear Unit — and the activation used for the gate `f` is what names the variant. Swap in SiLU and you get **SwiGLU**; swap in GELU and you get **GeGLU**.

#### SwiGLU *(gated SiLU, 2020–present)*

**Intuition first.** Instead of one pathway deciding "how much of myself to let through" (self-gating, as in plain SiLU), SwiGLU splits the work: one projection of the input computes *content*, a second, independent projection computes a *gate* over that content, and the two are multiplied together. It's the difference between a valve that reads its own pressure versus a valve controlled by a separate sensor — the second design can learn a much richer notion of "how much to pass."

```
SwiGLU(x) = (SiLU(xW) ⊗ xV) W₂
```

There's no single fixed "shape" to draw for SwiGLU the way there is for SiLU — it's a function of two independent linear projections of the full input vector, not a 1-D curve of a scalar. What carries over from SiLU is the smoothness and the absence of a hard dead zone; what's added is expressiveness, since the gate and the content are no longer forced to be the same signal. The FFN's hidden dimension is typically shrunk by roughly two-thirds to keep the parameter count comparable to a non-gated FFN, since SwiGLU needs three weight matrices (`W, V, W₂`) instead of two.

*Used in:* the FFN activation in **Llama (all versions), Mistral, Qwen, PaLM**, and most current-generation open LLMs.

- **Pros:** consistently outperforms plain ReLU/GELU FFNs at equal compute in published ablations; gate and content are learned separately — more expressive per FFN block.
- **Cons:** three weight matrices instead of two — more parameters per block; extra elementwise multiply — marginally more compute than a plain FFN.

#### GeGLU *(gated GELU, 2020–present)*

**Intuition first.** Exactly the SwiGLU idea — two independent projections, one gating the other — with GELU standing in for SiLU as the gate. Since GELU and SiLU are near-identical smoothed relatives of ReLU, GeGLU and SwiGLU behave very similarly in practice; the choice between them is closer to a house style than a principled tradeoff.

```
GeGLU(x) = (GELU(xW) ⊗ xV) W₂
```

Same structural diagram as SwiGLU, same parameter-count consideration, same reason for existing: separating "how much content" from "how much to let through" gives the FFN more room to represent complex per-token functions than a single activation curve can.

*Used in:* the FFN activation in Google's **Gemma** model family (and T5's later variants).

- **Pros:** same expressiveness gains as SwiGLU over non-gated FFNs; GELU's smoothness carried into the gate.
- **Cons:** same extra-parameter, extra-multiply cost as SwiGLU; no consistent empirical edge over SwiGLU — largely an architectural choice.

## 4. A runnable PyTorch comparison

The snippet below applies every activation covered above to the same sample tensor, so the numbers can be compared directly. `nn.SiLU` and `nn.GELU` ship in PyTorch directly; SwiGLU/GeGLU are shown as the small gated block they actually are inside a real FFN.

```python
import torch
import torch.nn as nn

x = torch.tensor([-3.0, -1.0, -0.1, 0.0, 0.5, 2.0, 5.0])

fns = {
    "sigmoid":    nn.Sigmoid(),
    "tanh":       nn.Tanh(),
    "relu":       nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.01),
    "gelu":       nn.GELU(),
    "silu":       nn.SiLU(),   # a.k.a. Swish
}

for name, fn in fns.items():
    print(f"{name:>11}: {fn(x).detach().numpy().round(3)}")

# SwiGLU as it's actually used inside an FFN block
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w, self.v = nn.Linear(dim, hidden), nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w(x)) * self.v(x))

layer = SwiGLU(dim=8, hidden=16)
sample = torch.randn(2, 8)
print("SwiGLU out:", layer(sample).shape)  # torch.Size([2, 8])
```

Running this on `x = [-3, -1, -0.1, 0, 0.5, 2, 5]` makes the shapes concrete: sigmoid and tanh squeeze everything toward their bounds; ReLU zeroes the entire negative half exactly; GELU and SiLU leave a small negative dip around `-1` instead of clamping to zero, then converge toward the identity line as `x` grows.

## 5. Summary table

| Function | Formula | Typical use | Key tradeoff |
|---|---|---|---|
| Sigmoid | `1 / (1 + e^(−x))` | Output-layer probabilities, LSTM gates | Vanishing gradients, not zero-centered |
| Tanh | `(e^x − e^(−x)) / (e^x + e^(−x))` | RNN / LSTM hidden states | Zero-centered, but still saturates |
| ReLU | `max(0, x)` | CNNs, general FFNs | Cheap and non-saturating, but neurons can die |
| Leaky ReLU | `x if x>0 else αx` | GANs, CNNs prone to dead units | Fixes dying ReLU, adds a tuned constant |
| GELU | `x · Φ(x)` | GPT-2, BERT | Smooth and strong, costlier than ReLU |
| SiLU / Swish | `x · σ(x)` | Gate for SwiGLU FFNs | Smooth, self-gating, near-GELU cost |
| SwiGLU | `(SiLU(xW) ⊗ xV) W₂` | Llama, Mistral, Qwen FFNs | More expressive, three weight matrices |
| GeGLU | `(GELU(xW) ⊗ xV) W₂` | Gemma FFNs | Same as SwiGLU, GELU-flavored gate |

## 6. Key takeaways

- Nonlinearity is the entire point. Without an activation function between layers, any stack of linear layers collapses algebraically into one linear layer — depth would buy nothing.
- The field moved from saturating functions (sigmoid, tanh) to non-saturating ones (ReLU) specifically to fix vanishing gradients, then moved again to smooth, non-saturating functions (GELU, SiLU) to fix ReLU's dead-neuron problem without giving up ReLU's trainability.
- Modern LLMs favor SiLU/SwiGLU over plain ReLU/GELU because gating — multiplying a content projection by a separately-learned gate — adds real expressiveness to the FFN block, and empirically outperforms a single-activation FFN at matched compute.
- Gated variants (SwiGLU, GeGLU) cost more parameters (three weight matrices instead of two) — teams compensate by shrinking the FFN's hidden dimension so total parameter count stays comparable.
- The choice of activation is architecture-specific, not universal: sigmoid/tanh still make sense inside LSTM gates, ReLU is still the right default for many CNNs, and SwiGLU/GeGLU are specifically a transformer-FFN choice.
- Which gated activation a model uses is a legible fingerprint of its lineage: SwiGLU points to the Llama/Mistral/Qwen family, GeGLU points to Gemma, and plain GELU points to an earlier-generation model like GPT-2 or BERT.
