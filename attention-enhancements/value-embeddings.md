# Value Embeddings (ResFormer-style)

**What it is:** On alternating layers, the raw token embedding is mixed back into the Value (V) vectors via a learned gate — letting the model "remember" what the original token was, deep in the network.

**Code:** `gpt.py:53-55`, `gpt.py:79-95`

```python
# gpt.py:53-55 — which layers get value embeddings
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

# gpt.py:79-80 — gate is a small Linear over the first 12 input channels
self.ve_gate_channels = 12
self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

# gpt.py:91-95 — mixing value embedding into V in forward()
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
    v = v + gate.unsqueeze(-1) * ve
```

---

## The Problem: Token Identity Gets Lost in Deep Networks

In a standard Transformer, a token's representation evolves through every layer:

```
Layer 0:  "cat" = [0.4, 0.6, -0.2, 0.8]    ← original embedding
Layer 1:  "cat" = [0.7, 0.2,  0.5, 0.3]    ← after attending to neighbours
Layer 2:  "cat" = [0.1, 0.9,  0.3, 0.6]    ← context mixed in
Layer 3:  "cat" = [0.5, 0.4,  0.8, 0.1]    ← even more context
...
Layer 12: "cat" = [0.3, 0.7,  0.2, 0.9]    ← almost unrecognisable
```

Each layer blends in context from surrounding tokens. This is good — it's how the model builds understanding. But the **original token identity** ("this token is specifically 'cat'") can fade as you go deeper.

In the attention mechanism, the V (Value) vector is what gets mixed into the output. If V has drifted far from the original token, the attention output may lose track of which token it's attending to.

---

## The Fix: Mix the Raw Embedding Back into V

### Standard attention value computation

```
V = hidden_state × W_v

hidden_state = the token's current representation (evolved through layers)
W_v          = learned weight matrix
```

### Value Embedding (ResFormer-style)

```
V = hidden_state × W_v  +  gate × raw_embedding

raw_embedding = the original token embedding from wte
gate          = a single learned scalar (or small vector)
```

The raw embedding is added directly into V before attention aggregation.

---

## Step-by-Step Walkthrough: "The cat sat on the mat"

Say we're in layer 6 (an alternating layer where value embeddings are active).

### Step 1 — Compute standard V

```
"cat" hidden state at layer 6: [0.5, 0.4, 0.8, 0.1]
W_v = learned weight matrix

V_standard = [0.5, 0.4, 0.8, 0.1] × W_v = [0.3, 0.7, 0.2, 0.9]
```

### Step 2 — Get the raw token embedding

```
"cat" raw embedding from wte (same as layer 0):
raw_emb = [0.4, 0.6, -0.2, 0.8]

This never changes — it's the fixed initial identity of "cat"
```

### Step 3 — Mix in via gate

```
gate = 0.3   (learned scalar — the model decides how much to blend)

V_final = V_standard  +  gate × raw_emb
        = [0.3, 0.7, 0.2, 0.9]  +  0.3 × [0.4, 0.6, -0.2, 0.8]
        = [0.3, 0.7, 0.2, 0.9]  +  [0.12, 0.18, -0.06, 0.24]
        = [0.42, 0.88, 0.14, 1.14]
```

V now carries both:
- The contextual representation from hidden state (who "cat" is in this sentence)
- A direct signal of the original token identity (the word "cat" specifically)

### Step 4 — Attention uses this enriched V

When "sat" attends to "cat" and retrieves its V, it gets both the contextual meaning AND a direct reminder that the token it attended to was specifically "cat".

---

## Why "Alternating Layers"?

The feature is active on every other layer (e.g., layers 0, 2, 4, 6... or 1, 3, 5, 7...).

```
Even layers: V = W_v(h) + gate × raw_emb   ← value embeddings active
Odd layers:  V = W_v(h)                    ← standard attention

Why not every layer?
→ Each injection adds compute cost
→ Alternating is a balance: frequent enough to help, not so frequent
  that it dominates
→ Empirically found to be effective
```

---

## Why the Gate Matters

The gate is learned — it's not fixed at 0.3. During training, each layer learns its own gate value.

```
If gate → 0:   "I don't need raw identity here, context is enough"
               V ≈ W_v(hidden_state)    ← standard attention

If gate → 1:   "Raw token identity is very important here"
               V carries strong original token signal

If gate → 0.3: "A mild reminder of token identity is helpful"
```

This lets the model adapt: early layers might rely more on raw identity, later layers might suppress it as abstract representations dominate.

---

## The "Cheap" Part

Adding the raw embedding to V costs:
- One multiply: `gate × raw_emb`  (just scalar × vector)
- One add: `V_standard + gate × raw_emb`

No new weight matrices. No extra attention heads. Just a gated addition using the embedding that was already computed.

```
Cost: O(d_model) per token per layer   ← negligible
Benefit: model retains token identity deep in the network
```

---

## ResFormer Connection

The name "ResFormer-style" refers to the ResFormer paper, which showed that injecting raw embeddings into attention values helps models maintain representational quality over many layers. The idea is similar to a **residual connection** but applied specifically to the Value stream rather than the main residual stream.

---

## How Is This Different from x0_lambdas?

Both use the original token embedding — but they solve different problems in different places.

| | Value Embeddings | x0_lambdas |
|---|---|---|
| **Where it acts** | Inside the V vector, during attention | The residual stream, between layers |
| **Who benefits** | Other tokens attending to this token | This token itself |
| **Question answered** | "When others look at me, do they know what I am?" | "Am I still remembering what I am?" |

**Value Embeddings** fix an **outward signal** problem: when "sat" attends to "cat" and retrieves its V, that V should clearly identify the token being retrieved. Without this, the drifted hidden state produces a murky V.

**x0_lambdas** fix an **internal drift** problem: as "cat"'s own representation evolves through layers, it can drift far from its original meaning. Adding `x0` back nudges the token's own running state back toward its identity.

```python
# Value Embeddings — affects what a token sends to others (inside attention)
v = v + gate.unsqueeze(-1) * ve   # gpt.py:95

# x0_lambdas — affects a token's own evolving state (residual stream loop)
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0   # gpt.py:452
```

**Analogy:** Think of "cat" as a person in a long meeting.
- **x0_lambda** is "cat" re-reading their own name badge to remind *themselves* who they are.
- **Value Embeddings** is putting the name badge on the outside of their folder, so *others* who look them up get a clear signal.

Same source of information. Different directions. Both needed.

---

## One-Line Summary

> On alternating layers, the original token embedding is added back into the V vectors via a learned gate — a cheap way to keep the model grounded in token identity as it builds up deep, context-rich representations.
