# SwiGLU MLP — explained

Every Llama decoder layer has a feed-forward block (`LlamaMLP`) that runs after attention. This file explains what SwiGLU is, why it works, and how it differs from simpler MLP designs.

---

## The Code

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = ACT2FN["silu"]   # SiLU = Swish = x * sigmoid(x)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

For Llama3 8B: `hidden_size=4096`, `intermediate_size=11008`.

Three projections, not two. The forward pass in one line:
```
x → gate_proj → SiLU → ⟩
                        ⊗ → down_proj → output
x → up_proj   -------→ ⟩
```

---

## Building Up From a Plain MLP

### Vanilla MLP (original transformer FFN)
```python
x = Linear(hidden → intermediate)(x)
x = ReLU(x)
x = Linear(intermediate → hidden)(x)
```
One projection up, one activation, one projection back down. ReLU zeroes out negative activations — a hard, input-independent threshold.

### GLU — Gated Linear Unit (Dauphin et al., 2017)
```python
gate = sigmoid(W1 @ x)
up   = W2 @ x
output = gate * up
```
Split the projection into two halves. One half (`gate`) goes through sigmoid to produce values in (0,1). The other (`up`) is the actual content. They're multiplied elementwise — the gate *modulates* how much of each dimension of `up` passes through.

**Key insight:** the gate is input-dependent. For the same weight matrix, different tokens produce different gate values, so the MLP can selectively suppress or amplify different dimensions based on *what the token is*, not just *which dimension it is*. A plain MLP with ReLU makes a fixed, position-independent cut; GLU makes a content-aware, soft cut.

### SwiGLU — GLU with SiLU instead of sigmoid (Noam Shazeer, 2020)
```python
gate = silu(W_gate @ x)     # SiLU instead of sigmoid
up   = W_up @ x
output = gate * up
```

**SiLU (Swish):** `silu(x) = x * sigmoid(x)`

| x | sigmoid(x) | silu(x) = x·σ(x) |
|---|---|---|
| −3 | 0.047 | −0.14 |
| −1 | 0.269 | −0.27 |
| 0 | 0.5 | 0.0 |
| 1 | 0.731 | 0.73 |
| 3 | 0.953 | 2.86 |

Unlike ReLU (hard zero below 0) or sigmoid (always (0,1)), SiLU:
- Is **smooth** everywhere — differentiable, including at 0. No kink for gradients to get stuck on.
- Has a **small negative regime** (slightly below zero for slightly negative inputs) rather than a hard cut — the network can represent "slightly suppress this dimension" rather than just "zero or not zero."
- Is **unbounded above** — large positive activations pass through nearly unchanged (SiLU(x) ≈ x for large x), so there's no saturation ceiling.

Replacing sigmoid with SiLU in the gate makes the gating signal itself richer — rather than a flat (0,1) probability, the gate carries the magnitude of the input, weighted by how confident it is that dimension should fire.

---

## Why Three Matrices Instead of Two

A plain MLP with one intermediate projection has `2 * hidden * intermediate` parameters.

SwiGLU has three projections — `gate_proj`, `up_proj`, `down_proj` — which would naively be `3 * hidden * intermediate`. To keep parameter count equivalent to a 2-matrix MLP at the same capacity, `intermediate_size` is reduced. Llama uses `intermediate_size ≈ 2/3 * 4 * hidden_size`, rounded to a multiple of 256:

For Llama3 8B: `hidden=4096`, `intermediate=11008 ≈ (2/3) * 4 * 4096`.

So the three-matrix SwiGLU uses roughly the same total parameters as a two-matrix plain MLP at a wider intermediate size — you're trading one wide projection for two narrower ones plus a gate.

---

## What the Gate Actually Learns

After training, `gate_proj` learns to detect *features* in the token's representation, and `sigmoid(gate)` learns to express *how relevant each feature is for this token*. The elementwise multiply with `up_proj` (which also detects features, but in a parallel space) then suppresses the dimensions that the gate says are irrelevant.

Concrete example: token `"Paris"` in the context `"The capital of France is Paris"`. The gate might learn to:
- Fire high on "geographic/capital" dimensions → pass through, contributing to the model's understanding of location
- Fire low on "temporal/past-tense" dimensions → suppress them (they're not relevant for a place name)

A plain ReLU MLP would apply the same suppression pattern regardless of context. GLU/SwiGLU makes the suppression pattern a function of the input itself.

---

## Relationship to Llama4's MoE

In Llama4, MoE layers use the same SwiGLU structure inside each expert (`Llama4TextExperts` fuses `gate_proj` and `up_proj` into one `gate_up_proj` parameter for batched efficiency), and the shared expert (`Llama4TextMLP`) is the same SwiGLU structure as Llama3's `LlamaMLP`. The gate mechanism is identical — only the routing layer on top is new.
