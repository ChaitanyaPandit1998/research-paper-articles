# Lean Design: No Biases, No Learnable RMSNorm Params, Untied Embeddings

**Topics covered:**
- No bias terms in Linear layers (topic 6a)
- RMSNorm with no learnable scale/shift (topic 6b)
- Untied embedding & LM head weights (topic 7)

**Code:** `gpt.py:42-50`, `gpt.py:75-78`, `gpt.py:172-175`

```python
# gpt.py:42-50
def norm(x):
    return F.rms_norm(x, (x.size(-1),))  # no learnable params — pure normalisation

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))

# gpt.py:75-78 — all attention projections, bias=False
self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

# gpt.py:172-175 — untied embeddings and LM head
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),  # input
    ...
})
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)  # output — separate matrix
```

---

## Part 1: No Bias Terms in Linear Layers

### What is a bias?

Every standard linear layer computes:

```
output = input × W + b

Where:
  W = weight matrix (the main parameters)
  b = bias vector (one extra number per output dimension)
```

A bias lets the layer shift the output up or down regardless of the input.

**Example — without bias:**
```
Linear layer for "cat" embedding [0.4, 0.6]:
  W = [[0.5, 0.2],
       [0.3, 0.8]]

output = [0.4×0.5 + 0.6×0.3,  0.4×0.2 + 0.6×0.8]
       = [0.38,  0.56]
```

**Example — with bias [0.1, -0.2]:**
```
output = [0.38 + 0.1,  0.56 + (-0.2)]
       = [0.48,  0.36]
```

### Why remove it?

**1. Redundancy at scale.**
In a deep network with many layers, each layer can learn to shift activations through its weight matrix. The bias is doing work that the weights can already do — it's a free parameter that doesn't earn its keep.

**2. Fewer parameters = more room for useful ones.**
If your model has `d_model = 1024` and `n_layers = 24`, removing biases from all linear layers saves:
```
~24 layers × ~4 linear projections × 1024 dims = ~100K parameters

That budget can instead go toward a larger W or more layers.
```

**3. RMSNorm compensates.**
RMSNorm after each sublayer already rescales activations. If activations are constantly rescaled, a learned offset (bias) is largely redundant.

```
With bias-free layers:
  output = input × W    → simple, no offset
  → RMSNorm rescales it anyway
  → bias would've been absorbed and undone
```

---

## Part 2: RMSNorm With No Learnable Parameters

### Standard RMSNorm (with learnable scale)

```
RMSNorm(x) = (x / RMS(x)) × γ

γ = learnable scale per dimension (one number per hidden dim)
```

γ lets the model choose "how large should the normalized output be?"

### This model's RMSNorm (no γ)

```
RMSNorm(x) = x / RMS(x)

That's it. No scale, no shift.
```

### Why remove γ?

**1. Even leaner parameter count.**
For `d_model = 1024` and `n_layers = 24`, each RMSNorm normally has 1024 γ values:
```
24 layers × 2 RMSNorms per layer × 1024 = 49,152 parameters saved

Small, but contributes to the "lean" philosophy.
```

**2. The next layer handles scaling.**
After RMSNorm, the output goes into a Linear layer with its own weight matrix W. That W already controls scale — γ would be competing/redundant.

```
RMSNorm (no γ) → x / RMS(x)  →  Linear (W)  →  x/RMS(x) × W

The W columns already act as a learnable scale per dimension.
γ would be absorbed into W during training anyway.
```

**3. Forces the model to rely on weights.**
By removing free scaling parameters, every bit of expressive power must come from the weight matrices. This can lead to better-utilised weights.

---

## Part 3: Untied Embedding & LM Head Weights

### What is weight tying?

In classic GPT-2 and many language models:

```
wte = token embedding matrix     (vocab_size × d_model)
lm_head = output projection       (d_model × vocab_size)

Weight tying: lm_head.weight = wte.weight   ← same matrix, shared
```

The idea: "the best representation of a token as input should also be the best way to predict it as output."

### What this model does instead

```
wte    = nn.Embedding(vocab_size, d_model)    ← initialized independently
lm_head = nn.Linear(d_model, vocab_size)      ← initialized independently

wte.weight ≠ lm_head.weight   ← two separate matrices
```

### Why untie them?

**1. Different jobs, different optimal representations.**

```
wte (input embedding):
  Maps token IDs → rich representations
  "cat" → [0.4, 0.6, -0.2, 0.8, ...]
  Needs to capture: semantics, syntax, what the token IS

lm_head (output projection):
  Maps final hidden states → probability over vocabulary
  Needs to capture: what token should come NEXT given this context
  "The cat sat on the ___" → high probability for "mat"
```

These are related but different. A single shared matrix must compromise.

**2. The hidden state going into lm_head is NOT the same as an input embedding.**

```
Input to wte:    token ID → raw embedding (no context)
Input to lm_head: final hidden state (processed through all transformer layers,
                  has attended to the whole sequence, encoded context)

Forcing them to share a matrix is like using the same key for both
"encoding what this word means" and "predicting what comes next."
```

**3. More parameters, more capacity.**

```
Tied:    1 matrix of vocab_size × d_model
Untied:  2 matrices of vocab_size × d_model

Cost: doubles the embedding parameter count
Benefit: each matrix can specialise fully
```

---

## How These Three Work Together

Think of it as a philosophy: **"every parameter should earn its place."**

```
Bias removed          → W already handles offsets, bias is redundant
RMSNorm γ removed     → downstream W already handles scale, γ is redundant
Embeddings untied     → input encoding and output prediction need different
                        representations, sharing forces a compromise
```

The result is a model that:
- Has fewer "filler" parameters
- Forces each weight matrix to work harder
- Spends its parameter budget on genuine representational capacity

---

## Parameter Count Impact (example: d_model=1024, vocab=50K, 24 layers)

| What | Removed? | Params saved |
|---|---|---|
| Biases in all linear layers | Yes | ~100K |
| RMSNorm γ parameters | Yes | ~49K |
| Weight tying | No (untied adds params) | −50M (adds 50M more) |

Untying adds parameters but improves quality. Removing biases and γ reduces parameters without hurting quality — freeing that budget for more layers or larger matrices.
