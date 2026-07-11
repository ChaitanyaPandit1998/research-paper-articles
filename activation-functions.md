# Activation Functions, Demystified: From Sigmoid to the Gated Units Inside Every Modern LLM

**Pull quotes:**
- "Stack a hundred linear layers and you still have one linear layer. The activation function is the only thing standing between a neural network and a very expensive linear regression."
- "ReLU didn't win because it was elegant. It won because its gradient doesn't shrink — and for a while, that was the only thing standing between 'deep learning' and 'a network that refuses to train past ten layers.'"
- "SwiGLU isn't a new activation function so much as an admission that one number deciding 'how much to let through' was never quite enough — better to let a second, independently-learned projection make that call."

---

An activation function is a small, fixed, elementwise function applied to a layer's output before it moves to the next layer — a single line of code that quietly decides whether a network is capable of learning anything beyond straight lines. This article works through what activation functions do, why non-linearity is non-negotiable, and how the field moved from sigmoid through ReLU to the smooth, gated units — SwiGLU, GeGLU — running inside Llama, Mistral, Qwen, and Gemma today.

---

## Table of contents

1. [What are they?](#1-what-are-they)
2. [Where they live in a network](#2-where-they-live-in-a-network)
3. [The functions in active use](#3-the-functions-in-active-use)
4. [Gradients and shapes, in numbers](#4-gradients-and-shapes-in-numbers)
5. [A runnable PyTorch comparison](#5-a-runnable-pytorch-comparison)
6. [What activation functions don't solve](#6-what-activation-functions-dont-solve)
7. [Summary table](#7-summary-table)
8. [Key takeaways](#8-key-takeaways)
9. [Further reading](#9-further-reading)

---

## 1. What are they?

A network without activation functions is just linear algebra pretending to be deep.

An activation function is a small, fixed, elementwise function that a layer applies to its output before passing it to the next layer. Every neuron computes a weighted sum of its inputs:

$$z = Wx + b$$

The activation function decides how that raw sum gets translated into a signal the rest of the network actually uses. It takes one number in and returns one number out, applied independently to every element of the layer's output.

The reason this matters more than its modest description suggests: without it, depth is an illusion. Stack any number of purely linear layers and the result collapses algebraically into a single linear layer. Watch it happen with two layers:

$$y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$$

Two matrix multiplications compose into one matrix multiplication. A hundred linear layers still compose into one. No amount of stacking linear transformations lets a network learn anything a single linear regression couldn't already express — it can't bend a decision boundary, can't separate an XOR pattern, can't approximate a curve.

**Where the collapse breaks.** Insert a non-linear function $f$ between the layers and the algebra above simply doesn't go through anymore:

$$y = W_2\, f(W_1 x + b_1) + b_2$$

There is no matrix $W'$ for which this equals $W'x + b'$ in general, because $f$ is not a linear operation — you cannot pull it apart into a matrix multiplication. Each additional `linear → f → linear` block therefore genuinely adds representational power that wasn't there before. This is the entire reason activation functions exist; everything else in this article is about which non-linearity to pick and why it matters.

---

## 2. Where they live in a network

Activation functions sit between the linear projections inside every feed-forward block: a multi-layer perceptron (MLP), a transformer's feed-forward network (FFN), a convolutional filter bank. The pattern is always *linear → non-linear → linear*: project the input into a (usually wider) hidden space, bend it with an activation, then project back down.

```
input x  →  linear (W₁x + b₁)  →  activation f(·)  →  linear (W₂h + b₂)  →  output
```

In a transformer, this block — self-attention, then an FFN with an activation in the middle — is what actually holds most of the parameters. Attention decides *which* tokens to look at; the FFN, activation included, is where the model does the bulk of its per-token computation, and is generally believed to store much of what a large language model "knows."

Two separate things ride on the choice of activation function, and it's worth keeping them distinct.

**Expressiveness.** A network's ability to approximate complicated functions comes entirely from the non-linearities between its linear layers. Different activation shapes carve up input space differently — a smooth S-curve saturates gently, a hinge function creates a sharp fold — and that shape becomes part of what the network can represent.

**Gradient flow.** Every activation function's derivative becomes a multiplicative factor in backpropagation. A network might be twenty, fifty, or a hundred layers deep, and the gradient signal for the earliest layers has to survive being multiplied by the derivative of the activation function at *every* layer it passes back through:

$$\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{\ell=2}^{L} f'(z_\ell) \cdot W_\ell$$

If $f'$ is routinely small (as with sigmoid) or exactly zero (as with ReLU on negative inputs), gradients shrink or vanish before they ever reach the early layers, and training stalls. This is not a minor implementation detail — it is the reason the field moved from sigmoid/tanh to ReLU, and later to the smoother, gated variants covered below.

---

## 3. The functions in active use

Seven functions, three eras: the saturating classics, the ReLU family that made deep networks trainable, and the smooth gated units that now run inside every major LLM.

### Sigmoid *(logistic function, 1980s–2000s)*

**Intuition first.** Think of a dimmer switch with a soft floor and a soft ceiling: no matter how hard you push the input, the output only ever eases toward 0 or toward 1. Historically it was used to model something like a probability, or a biological neuron easing from "off" to "firing."

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

The shape is a smooth S: flat near both extremes, steepest at $x=0$, output squeezed permanently into $(0,1)$. For large positive $x$ it saturates at 1; for large negative $x$ it saturates at 0. That squeezing was the whole appeal — it turns any real number into something that reads like a probability — but the same squeezing is its downfall: the derivative

$$\sigma'(x) = \sigma(x)\big(1-\sigma(x)\big)$$

maxes out at just 0.25 (at $x=0$) and shrinks toward zero at both tails, so gradients flowing backward through several sigmoid layers shrink geometrically. That's the vanishing-gradient problem that made deep sigmoid networks nearly untrainable — worked out in numbers in [Section 4](#4-gradients-and-shapes-in-numbers).

**Worked example.** Take $x = 1.5$:

$$\sigma(1.5) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.2231} = 0.8176$$

A moderately positive input is squashed to a number close to, but never reaching, 1.

*Used in:* still standard as an **output-layer** gate for binary classification and inside LSTM/GRU gates — just rarely used to activate hidden layers anymore.

- **Pros:** output bounded in (0,1), reads naturally as a probability; smooth and differentiable everywhere.
- **Cons:** vanishing gradients away from $x=0$; not zero-centered, which slows optimization; involves an exponential — costlier than a hinge function.

### Tanh *(hyperbolic tangent, 1990s–2010s)*

**Intuition first.** Tanh is sigmoid's better-balanced sibling — the same soft S-curve, but stretched and shifted so it eases between −1 and 1 instead of 0 and 1, passing through the origin. That "zero-centered" property sounds cosmetic but isn't: when a layer's outputs are centered around zero rather than all positive, the gradient updates to the next layer's weights don't get stuck pushing consistently in one direction.

$$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} = 2\sigma(2x) - 1$$

Same S-shape as sigmoid, same saturating tails, same vanishing-gradient failure mode for deep stacks — it's a rescaled sigmoid, so the math inherits both the strength and the weakness. It became the default hidden-layer activation for RNNs (and still is, inside LSTM/GRU cell states) precisely because zero-centering made optimization noticeably more stable than sigmoid, even though the saturation problem never went away.

**Worked example.** Take the same $x = 1.5$ used for sigmoid, so the two are directly comparable:

$$\tanh(1.5) = \frac{e^{1.5} - e^{-1.5}}{e^{1.5} + e^{-1.5}} = \frac{4.4817 - 0.2231}{4.4817 + 0.2231} = 0.9051$$

Notice it's pushed much closer to its ceiling (0.9051 out of a max of 1) than sigmoid was (0.8176 out of a max of 1) — tanh's steeper slope near the origin saturates faster for the same input.

*Used in:* dominant inside **recurrent** architectures (LSTM, GRU cell/hidden states); largely superseded by ReLU-family functions in feed-forward and transformer blocks.

- **Pros:** zero-centered output — better-behaved gradients than sigmoid; smooth, well-understood, bounded.
- **Cons:** still saturates and still vanishes gradients at the tails; exponential-based — more expensive than ReLU.

### ReLU *(rectified linear unit, 2011–present)*

**Intuition first.** A one-way valve: anything positive passes through completely unchanged, anything negative is clamped to exactly zero. No easing, no curve — just a hinge at the origin.

$$\text{ReLU}(x) = \max(0, x)$$

The shape is two straight line segments meeting at a corner: flat at zero for all $x \le 0$, then a perfect 45° line for $x > 0$. It doesn't saturate on the positive side at all — the gradient there is a constant 1, so it doesn't shrink no matter how deep the stack goes, which is exactly why it unlocked training networks dozens of layers deep. The tradeoff is bluntness: any neuron that lands in negative territory outputs zero and has zero gradient, so if its weights get pushed into a regime where it's *always* negative, it stops updating forever — the "dying ReLU" problem.

**Worked example.** Take $x = -2$ and $x = 3$:

$$\text{ReLU}(-2) = \max(0, -2) = 0 \qquad \text{ReLU}(3) = \max(0, 3) = 3$$

The negative input is discarded completely; the positive input passes through untouched, unscaled.

*Used in:* the default for CNNs and most feed-forward networks throughout the 2010s; still common today for its simplicity and speed.

- **Pros:** no vanishing gradient for positive inputs; trivially cheap — one comparison, no exponential; induces sparsity (many neurons output exactly 0).
- **Cons:** dying neurons — a permanently-negative unit stops learning; not zero-centered; non-differentiable kink at $x=0$ (rarely an issue in practice).

### Leaky ReLU *(2013–present)*

**Intuition first.** The same one-way valve as ReLU, but with a pinhole leak on the closed side — negative inputs aren't zeroed out, they're just heavily throttled, scaled down by a small constant instead of clamped flat.

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases} \qquad (\text{typically } \alpha = 0.01)$$

Visually it's almost indistinguishable from ReLU — the same hinge at the origin — except the left arm isn't flat, it's a very shallow downward line instead of sitting at zero. That tiny slope is the entire fix: a neuron that lands in negative territory still has a (small) nonzero gradient, so it can recover instead of dying permanently. It directly targets ReLU's single biggest failure mode without changing the cost or the positive-side behavior at all.

**Worked example.** Same inputs as ReLU above, $\alpha = 0.01$:

$$\text{LeakyReLU}(-2) = 0.01 \times (-2) = -0.02 \qquad \text{LeakyReLU}(3) = 3$$

Where plain ReLU would have output exactly $0$ for $x=-2$, Leaky ReLU outputs $-0.02$ — small, but nonzero, which is enough to keep a gradient flowing back through that neuron.

*Used in:* common in GANs and CNNs where dying units are a known problem; largely bypassed in modern LLMs in favor of GELU/SiLU-family functions.

- **Pros:** fixes dying-ReLU by keeping gradient flow alive for $x<0$; just as cheap as ReLU.
- **Cons:** $\alpha$ is an extra hyperparameter to tune (or learn, as in PReLU); empirical gains over plain ReLU are inconsistent.

### GELU *(Gaussian Error Linear Unit, 2016–present)*

**Intuition first.** Where ReLU is a hard valve that's either fully open or fully shut, GELU is a valve with judgment: it weights each input by how likely a standard Gaussian random variable would be to fall below it, so instead of a sharp on/off cutoff at zero, small negative inputs still leak through a little and the transition into "on" is a smooth curve, not a corner.

$$\text{GELU}(x) = x \cdot \Phi(x)$$

$$\Phi(x) = P(Z \le x),\qquad Z \sim \mathcal{N}(0,1)$$

**Where the approximation comes from.** $\Phi(x)$ has no closed form — it's an integral of the Gaussian density — so in practice it's evaluated either through the error function,

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}(x/\sqrt{2})\right]$$

or through the cheaper tanh-based approximation used in most deep learning libraries:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}\,\left(x + 0.044715\,x^3\right)\right]\right)$$

The shape tracks ReLU's overall trend — near zero for very negative $x$, roughly linear for large positive $x$ — but everything around the origin is rounded off: there's a small dip slightly below zero for negative inputs (unlike ReLU's flat clamp) and a smooth curved onset instead of a kink. That smoothness (the function is infinitely differentiable) removes the exact-zero dead zone that causes dying neurons, while the probabilistic framing — "gate each input by how large it is, stochastically" — was designed to combine the regularizing feel of dropout with the shape of an activation function.

**Worked example.** Take the same $x = 1.5$ used for sigmoid and tanh, using the tanh-based approximation:

$$\text{GELU}(1.5) \approx 0.5 \times 1.5 \times \big(1 + \tanh[0.7979 \times (1.5 + 0.044715 \times 1.5^3)]\big) = 1.3996$$

Compare to $\text{ReLU}(1.5) = 1.5$ exactly — GELU pulls the output in slightly, even for a positive input, because $\Phi(1.5) = 0.933$ rather than $1$.

*Used in:* the activation used in **GPT-2, BERT**, and most of the first wave of transformer language models.

- **Pros:** smooth everywhere — no dead zone, no hard kink; strong empirical performance in transformers; small negative outputs preserved instead of clamped.
- **Cons:** more expensive than ReLU (erf or tanh approximation); mostly superseded by SiLU/SwiGLU in newer LLMs.

### SiLU / Swish *(Sigmoid Linear Unit, 2017–present)*

**Intuition first.** Take the input and scale it by its own sigmoid — a self-gating valve, where the input decides how open it should be. Large positive numbers gate themselves almost fully open (behaving like ReLU); numbers near zero and slightly negative gate themselves partly closed instead of being cut off outright.

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Nearly identical in shape to GELU — both dip slightly below zero just left of the origin, both curve smoothly into a near-linear rise on the right — which is no accident, since both are smoothed relatives of the same ReLU trend. SiLU is a touch cheaper to compute (one sigmoid vs. an erf/tanh approximation) and was discovered largely via automated search over candidate activation functions, then confirmed by hand — a rare case of a simple closed-form function beating hand-designed alternatives at scale.

**Worked example.** Take the same $x = 1.5$ again, reusing $\sigma(1.5) = 0.8176$ from the sigmoid example:

$$\text{SiLU}(1.5) = 1.5 \times \sigma(1.5) = 1.5 \times 0.8176 = 1.2264$$

Slightly below GELU's $1.3996$ at the same input — the two track each other closely but are not identical.

*Used in:* the gate activation inside **SwiGLU**, used in **Llama, Mistral, Qwen**, and most current open-weight LLM feed-forward blocks.

- **Pros:** smooth, non-monotonic near zero, no dead zone; cheaper than GELU, comparable or better empirical results; self-gating — naturally suited to being used inside gated units.
- **Cons:** not zero-centered; marginally more expensive than plain ReLU.

### The gated variants: SwiGLU and GeGLU

Every function above is a single curve applied to a single number. The gated units used in modern LLM feed-forward blocks are a different construction entirely: instead of one linear projection passed through one activation, the input is projected *twice* by two independent weight matrices, one branch is passed through an activation to act as a gate, and the two branches are multiplied together elementwise before a final projection back down.

```
x → V x  (linear, "content")  ⊗  f(Wx)  (linear + activation, "gate")  →  W₂(·)  → output
```

This is a GLU — Gated Linear Unit — and the activation used for the gate $f$ is what names the variant. Swap in SiLU and you get **SwiGLU**; swap in GELU and you get **GeGLU**.

#### SwiGLU *(gated SiLU, 2020–present)*

**Intuition first.** Instead of one pathway deciding "how much of myself to let through" (self-gating, as in plain SiLU), SwiGLU splits the work: one projection of the input computes *content*, a second, independent projection computes a *gate* over that content, and the two are multiplied together. It's the difference between a valve that reads its own pressure versus a valve controlled by a separate sensor — the second design can learn a much richer notion of "how much to pass."

$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW) \otimes xV\big)\,W_2$$

There's no single fixed "shape" to draw for SwiGLU the way there is for SiLU — it's a function of two independent linear projections of the full input vector, not a 1-D curve of a scalar. What carries over from SiLU is the smoothness and the absence of a hard dead zone; what's added is expressiveness, since the gate and the content are no longer forced to be the same signal. The FFN's hidden dimension is typically shrunk by roughly two-thirds to keep the parameter count comparable to a non-gated FFN, since SwiGLU needs three weight matrices ($W, V, W_2$) instead of two.

**Worked example.** Take a toy input vector $x = [1, -1]$, with tiny $2\times2$ weight matrices $W = \begin{pmatrix}1&1\\1&-1\end{pmatrix}$ and $V = \begin{pmatrix}2&0\\0&2\end{pmatrix}$ (skip $W_2$ by treating it as identity, to isolate the gating step):

$$xW = [1{\times}1 + (-1){\times}1,\ 1{\times}1 + (-1){\times}(-1)] = [0,\ 2]$$

$$xV = [1{\times}2 + (-1){\times}0,\ 1{\times}0 + (-1){\times}2] = [2,\ -2]$$

$$\text{SiLU}(xW) = [\text{SiLU}(0),\ \text{SiLU}(2)] = [0,\ 1.7616]$$

$$\text{SwiGLU}(x) = \text{SiLU}(xW) \otimes xV = [0 \times 2,\ 1.7616 \times (-2)] = [0,\ -3.5232]$$

The first output dimension is gated shut entirely ($\text{SiLU}(0)=0$ kills it regardless of what $xV$ was); the second passes through, flipped in sign and scaled, because its gate value was large. This is the mechanism in miniature: the gate decides per-dimension how much of the content survives.

*Used in:* the FFN activation in **Llama (all versions), Mistral, Qwen, PaLM**, and most current-generation open LLMs.

- **Pros:** consistently outperforms plain ReLU/GELU FFNs at equal compute in published ablations; gate and content are learned separately — more expressive per FFN block.
- **Cons:** three weight matrices instead of two — more parameters per block; extra elementwise multiply — marginally more compute than a plain FFN.

#### GeGLU *(gated GELU, 2020–present)*

**Intuition first.** Exactly the SwiGLU idea — two independent projections, one gating the other — with GELU standing in for SiLU as the gate. Since GELU and SiLU are near-identical smoothed relatives of ReLU, GeGLU and SwiGLU behave very similarly in practice; the choice between them is closer to a house style than a principled tradeoff.

$$\text{GeGLU}(x) = \big(\text{GELU}(xW) \otimes xV\big)\,W_2$$

Same structural diagram as SwiGLU, same parameter-count consideration, same reason for existing: separating "how much content" from "how much to let through" gives the FFN more room to represent complex per-token functions than a single activation curve can.

**Worked example.** Same $x$, $W$, $V$ as the SwiGLU example above, so the two can be compared directly. Reusing $xW = [0, 2]$ and $xV = [2, -2]$:

$$\text{GELU}(xW) = [\text{GELU}(0),\ \text{GELU}(2)] = [0,\ 1.9546]$$

$$\text{GeGLU}(x) = \text{GELU}(xW) \otimes xV = [0 \times 2,\ 1.9546 \times (-2)] = [0,\ -3.9092]$$

Nearly the same result as SwiGLU's $[0, -3.5232]$ — the first dimension is gated shut identically, the second differs only because $\text{GELU}(2) = 1.9546$ is slightly larger than $\text{SiLU}(2) = 1.7616$. This is the SwiGLU/GeGLU relationship in one example: same structure, marginally different numbers.

*Used in:* the FFN activation in Google's **Gemma** model family (and T5's later variants).

- **Pros:** same expressiveness gains as SwiGLU over non-gated FFNs; GELU's smoothness carried into the gate.
- **Cons:** same extra-parameter, extra-multiply cost as SwiGLU; no consistent empirical edge over SwiGLU — largely an architectural choice.

### Pros and cons at a glance

*Condensed from the bullet lists above, for quick side-by-side scanning.*

| Function | Pros | Cons |
|---|---|---|
| Sigmoid | Output bounded in (0,1), reads as a probability; smooth and differentiable everywhere | Vanishing gradients away from x=0; not zero-centered; costs an exponential |
| Tanh | Zero-centered output — better-behaved gradients than sigmoid; smooth, well-understood, bounded | Still saturates and vanishes at the tails; exponential-based, more expensive than ReLU |
| ReLU | No vanishing gradient for x>0; trivially cheap (one comparison); induces sparsity | Dying neurons (permanently-negative units stop learning); not zero-centered; non-differentiable kink at 0 |
| Leaky ReLU | Fixes dying-ReLU by keeping gradient alive for x<0; as cheap as ReLU | α is an extra hyperparameter to tune; empirical gains over ReLU are inconsistent |
| GELU | Smooth everywhere, no dead zone; strong empirical performance in transformers; preserves small negative signal | More expensive than ReLU (erf/tanh approximation); mostly superseded by SiLU/SwiGLU in newer LLMs |
| SiLU / Swish | Smooth, non-monotonic near zero, no dead zone; cheaper than GELU; self-gating | Not zero-centered; marginally more expensive than plain ReLU |
| SwiGLU | Outperforms plain ReLU/GELU FFNs at equal compute; gate and content learned separately — more expressive | Three weight matrices instead of two; extra elementwise multiply adds compute |
| GeGLU | Same expressiveness gains as SwiGLU; GELU's smoothness carried into the gate | Same extra-parameter/extra-multiply cost as SwiGLU; no consistent empirical edge over it |

---

## 4. Gradients and shapes, in numbers

Two examples worth sitting with — why sigmoid networks stall, and where ReLU, GELU, and SiLU actually disagree.

### Example 1 — Why Sigmoid Networks Stop Learning

This is the vanishing-gradient problem in actual numbers, not just as a name.

**Setup:** a 5-layer network where every layer uses sigmoid, and every neuron happens to sit at a pre-activation of $z=2$ (a reasonably "active" neuron, not stuck far out in the tails). Backprop multiplies the incoming gradient by $\sigma'(z)$ at every layer it passes through.

$$\sigma(2) = 0.881 \qquad \sigma'(2) = 0.881 \times (1-0.881) = 0.105$$

Propagate a gradient that starts at 1.0 back through five such layers:

```
Layer          Multiply by σ'(2)      Running gradient
──────────────────────────────────────────────────────
start          —                      1.000
layer 5 → 4    × 0.105                0.105
layer 4 → 3    × 0.105                0.0110
layer 3 → 2    × 0.105                0.00116
layer 2 → 1    × 0.105                0.000122
layer 1 → 0    × 0.105                0.0000128
```

By the time the gradient reaches the first layer, it has shrunk by a factor of roughly 78,000 — and this is the *optimistic* case, where every neuron sits at its most favorable point ($z=2$, not deep in the saturated tails where $\sigma'$ is closer to 0.01). The earliest layers effectively stop receiving a usable training signal. Swap sigmoid for ReLU with every neuron active (positive), and each layer multiplies the gradient by exactly 1 — the running gradient stays at 1.000 all the way back. This single multiplicative difference is why deep sigmoid networks were considered nearly untrainable and why ReLU's arrival was treated as a turning point rather than an incremental tweak.

### Example 2 — The Shape Difference Between ReLU, GELU, and SiLU at the Same Inputs

The three functions agree almost everywhere for large $|x|$ and disagree most right around the origin — which is exactly where it matters, since that's where a unit is deciding whether to turn on or off.

```
x       ReLU(x)   GELU(x)   SiLU(x)
─────────────────────────────────────
-2.0     0.000    -0.045    -0.238
-1.0     0.000    -0.159    -0.269
-0.5     0.000    -0.154    -0.189
 0.0     0.000     0.000     0.000
 0.5     0.500     0.346     0.311
 1.0     1.000     0.841     0.731
 2.0     2.000     1.955     1.762
```

Three things to notice:

**ReLU is exactly zero for every negative input** — a hard, information-destroying cutoff. GELU and SiLU are both *negative* (not zero) in that range: a small amount of the original signal survives, just flipped and attenuated, rather than being discarded outright.

**GELU and SiLU nearly agree with each other everywhere**, and both converge toward ReLU as $|x|$ grows — at $x=2$, all three are within about 12% of each other. The difference between them is concentrated in a narrow band around the origin; it is a difference of *degree*, not of *kind*.

**Both curves bottom out and come back up** rather than diving to $-\infty$: GELU's minimum is around $x \approx -0.75$ (value $\approx -0.17$), and it rises back toward 0 for more negative $x$ — the exact opposite of what a straight line would do. This non-monotonicity is the visual signature of a "smoothed ReLU": there's no dead zone, but there also isn't unbounded negative output.

---

## 5. A runnable PyTorch comparison

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

---

## 6. What activation functions don't solve

Objectivity requires noting that no activation function is a complete fix, and picking a better one does not make other training problems disappear.

**They don't replace normalization.** Layer norm and batch norm exist because even a non-saturating activation like ReLU doesn't prevent activations from drifting to extreme scales across a deep network. Activation choice and normalization solve adjacent but distinct problems — swapping ReLU for SiLU is not a substitute for normalizing.

**Smoother is not automatically better.** GELU and SiLU fix ReLU's exact dead zone, but the empirical gap between them and ReLU is small at moderate model scale, and the theoretical case for *why* smoothness helps large transformers specifically is still argued from empirical ablations more than from a settled first-principles account.

**Gating adds real compute, not just parameters.** SwiGLU and GeGLU need three matrix multiplications and an elementwise product where a plain FFN needs two multiplications and one activation call. At inference time, on memory-bandwidth-bound hardware, that extra projection is not free — it's a deliberate trade of throughput for expressiveness, made because the accuracy gain has held up empirically at scale.

**No activation function fixes a bad initialization or learning rate.** Dying ReLU, for instance, is frequently a symptom of an initialization or learning rate that pushes too many neurons negative early in training — Leaky ReLU treats the symptom, but the underlying instability can still show up elsewhere.

**The "right" choice keeps moving.** GELU looked settled after BERT and GPT-2; SwiGLU displaced it in the next generation of open-weight LLMs largely on the strength of one influential ablation study (Shazeer, 2020) rather than a theoretical proof of superiority. Treat every "modern LLMs use X" claim in this article as a snapshot of current practice, not a law of nature.

---

## 7. Summary table

*Formulas below are compressed to fit a table row — see [Section 3](#3-the-functions-in-active-use) for each one written out on its own line, with a worked example.*

| Function | Formula | Typical use | Key tradeoff |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | Output-layer probabilities, LSTM gates | Vanishing gradients, not zero-centered |
| Tanh | $\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | RNN / LSTM hidden states | Zero-centered, but still saturates |
| ReLU | $\max(0,x)$ | CNNs, general FFNs | Cheap and non-saturating, but neurons can die |
| Leaky ReLU | $x$ if $x{>}0$ else $\alpha x$ | GANs, CNNs prone to dead units | Fixes dying ReLU, adds a tuned constant |
| GELU | $x\cdot\Phi(x)$ | GPT-2, BERT | Smooth and strong, costlier than ReLU |
| SiLU / Swish | $x\cdot\sigma(x)$ | Gate for SwiGLU FFNs | Smooth, self-gating, near-GELU cost |
| SwiGLU | $(\text{SiLU}(xW)\otimes xV)W_2$ | Llama, Mistral, Qwen FFNs | More expressive, three weight matrices |
| GeGLU | $(\text{GELU}(xW)\otimes xV)W_2$ | Gemma FFNs | Same as SwiGLU, GELU-flavored gate |

---

## 8. Key takeaways

- **Non-linearity is the entire point.** Without an activation function between layers, any stack of linear layers collapses algebraically into one linear layer — depth would buy nothing.
- **The field moved from saturating to non-saturating to smooth non-saturating.** Sigmoid/tanh saturate and vanish; ReLU fixed vanishing but introduced dying neurons; GELU/SiLU fixed dying neurons without giving up ReLU's non-saturating gradient — each generation targeted the previous one's specific failure mode.
- **Modern LLMs favor SiLU/SwiGLU over plain ReLU/GELU** because gating — multiplying a content projection by a separately-learned gate — adds real expressiveness to the FFN block, and empirically outperforms a single-activation FFN at matched compute.
- **Gated variants cost more parameters, not more magic.** SwiGLU and GeGLU need three weight matrices instead of two; teams compensate by shrinking the FFN's hidden dimension so total parameter count stays comparable to a non-gated design.
- **The choice of activation is architecture-specific, not universal.** Sigmoid/tanh still make sense inside LSTM gates, ReLU is still a defensible default for many CNNs, and SwiGLU/GeGLU are specifically a transformer-FFN choice.
- **Which gated activation a model uses is a legible fingerprint of its lineage:** SwiGLU points to the Llama/Mistral/Qwen family, GeGLU points to Gemma, and plain GELU points to an earlier-generation model like GPT-2 or BERT.

---

## 9. Further reading

- **"Gaussian Error Linear Units (GELUs)"** (Hendrycks & Gimpel, 2016) — the paper that introduced GELU and its tanh-based approximation, later adopted by GPT and BERT: arxiv.org/abs/1606.08415

- **"Searching for Activation Functions"** (Ramachandran, Zoph & Le, 2017) — the paper that discovered Swish/SiLU via automated search over candidate functions and validated it against ReLU and GELU at scale: arxiv.org/abs/1710.05941

- **"GLU Variants Improve Transformer"** (Shazeer, 2020) — the ablation study that introduced SwiGLU and GeGLU and established gated FFNs as an improvement over single-activation FFNs: arxiv.org/abs/2002.05202

- **"Deep Sparse Rectifier Neural Networks"** (Glorot, Bordes & Bengio, 2011) — the paper that established ReLU as a practical default and analyzed the sparsity and gradient-flow properties that made deep networks trainable: proceedings.mlr.press/v15/glorot11a

- **"Rectifier Nonlinearities Improve Neural Network Acoustic Models"** (Maas, Hannun & Ng, 2013) — introduced Leaky ReLU and the dying-ReLU diagnosis that motivated it.

---

Every generation of activation function was a response to a specific, diagnosable failure of the previous one: sigmoid's vanishing gradients gave way to ReLU's non-saturating slope; ReLU's dead neurons gave way to GELU and SiLU's smooth curves; a single self-gated curve gave way to SwiGLU and GeGLU's two-projection gate. None of these were aesthetic upgrades — each is a targeted fix, discoverable in the failure mode of what came before it. The activation function is one line in a forward pass, but it is the line that decides whether the rest of the network's depth was worth building at all.
