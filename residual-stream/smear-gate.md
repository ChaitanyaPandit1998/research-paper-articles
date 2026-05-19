# Smear Gate

**What it is:** Before the transformer layers begin, each token's embedding is blended with its immediate predecessor's embedding via a learned gate — injecting a cheap bigram signal right at the start.

**Code:** `gpt.py:183-184`, `gpt.py:427-444`

```python
# gpt.py:183-184 — parameter declarations
self.smear_gate = Linear(24, 1, bias=False)   # reads only the first 24 channels
self.smear_lambda = nn.Parameter(torch.zeros(1))

# gpt.py:427-444 — forward pass (training path shown)
# Smear: mix previous token's embedding into current position (cheap bigram info)
if kv_cache is None:
    # Training / naive generate: full sequence available, use fast slice
    assert T > 1, "Training forward pass should have T > 1"
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
else:
    # KV cache inference: read prev embedding from cache, store current for next step
    x_pre_smear = kv_cache.prev_embedding
    kv_cache.prev_embedding = x[:, -1:, :]
    if T > 1:
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    elif x_pre_smear is not None:
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
        x = x + gate * x_pre_smear
```

---

## What Is a "Bigram Signal"?

A **bigram** is a pair of adjacent tokens. Bigram statistics are among the most powerful predictors in language:

```
"New ___"     → very likely "York" or "Zealand"
"United ___"  → very likely "States" or "Kingdom"
"The ___"     → likely a noun
"sat ___"     → likely "on", "down", "up"
```

In the sentence **"The cat sat on the mat"**:
- Knowing the previous token is "sat" tells you a lot about what "on" should know
- Knowing the previous token is "The" helps "cat" know it's likely a subject noun

Transformers can learn bigrams through attention, but it takes many layers. The Smear Gate injects this signal **before layer 0** — for free.

---

## How It Works

### Standard embedding (no smear gate)

```
Tokens:    The    cat    sat    on    the    mat
Positions:  0      1      2      3      4      5

x[0] = embed("The")   = [0.5, 0.3, 0.8, 0.2]
x[1] = embed("cat")   = [0.4, 0.6, -0.2, 0.8]
x[2] = embed("sat")   = [0.7, 0.5, 0.3, 0.6]
x[3] = embed("on")    = [0.2, 0.4, 0.9, 0.1]
x[4] = embed("the")   = [0.5, 0.3, 0.8, 0.2]  ← same as "The"
x[5] = embed("mat")   = [0.3, 0.7, 0.4, 0.5]

Each token only knows about itself. No context yet.
```

### With Smear Gate

```
gate = learned scalar (e.g. 0.3)

x[0] = embed("The")   + 0    (no previous token, no smear)
x[1] = embed("cat")   + gate × embed("The")
     = [0.4, 0.6, -0.2, 0.8] + 0.3 × [0.5, 0.3, 0.8, 0.2]
     = [0.4, 0.6, -0.2, 0.8] + [0.15, 0.09, 0.24, 0.06]
     = [0.55, 0.69, 0.04, 0.86]

x[2] = embed("sat")   + gate × embed("cat")
     = [0.7, 0.5, 0.3, 0.6] + 0.3 × [0.4, 0.6, -0.2, 0.8]
     = [0.82, 0.68, 0.24, 0.84]

x[3] = embed("on")    + gate × embed("sat")
x[4] = embed("the")   + gate × embed("on")
x[5] = embed("mat")   + gate × embed("the")
```

Before any transformer layer runs, "cat" already has a trace of "The" baked in, "sat" has a trace of "cat", and so on.

---

## What the Gate Controls

```
gate → 0:   No smearing. Pure token embeddings, standard behaviour.
gate → 0.5: Moderate blend. Each token is noticeably influenced by its predecessor.
gate → 1:   Heavy blend. Each token is half itself, half its predecessor.
```

The gate is **learned** — the model figures out during training how much to smear. In practice, it learns a small-but-nonzero value: "just enough bigram signal to help, not so much that token identity is lost."

---

## Why "Before the Transformer Layers"?

Transformer attention can learn bigrams in layer 0, but it must:
1. Compute Q, K, V for every token
2. Compute all attention scores
3. Weight and aggregate

That costs O(n²) or O(n × w) operations just to capture "what came before me."

The Smear Gate does the same thing in O(n) — one multiply and one add per token. It's a direct lookup of the previous embedding, not a learned attention pattern.

```
Transformer attention to learn "previous token":
  Cost:    O(n × d)  — still proportional to sequence length
  Layers:  Takes at least 1 full attention layer
  Parameters: Attention weights W_q, W_k, W_v, W_o

Smear Gate:
  Cost:    O(n)       — one operation per token pair
  Layers:  0 (runs before layer 0)
  Parameters: 1 scalar (gate)
```

The transformer layers are then freed to focus on higher-order patterns (trigrams, syntax, semantics), because the cheap first-order signal is already handled.

---

## Causal Safety

The gate only looks at the **previous** token, never the next:

```
x[t] = embed(token[t]) + gate × embed(token[t-1])

At position t=3 ("on"):
  Allowed: embed("sat")  ← token at t-2=2, already processed
  Blocked: embed("the")  ← token at t+1=4, can't look ahead
```

This is causal — safe for autoregressive generation, where the model can only see what came before.

---

## Smear Gate vs Attention — What Each Is Good At

```
Smear Gate (position t-1 only):
  "sat on" — what's the word right before me?
  Cheap, always active, great for local function words and stop words

Attention (full window or sliding window):
  "The cat sat on the ___" — who is the subject? what's the verb?
  Expensive, but can find long-range patterns

They are complementary. Smear Gate handles the trivial local case for free,
freeing attention capacity for harder relationships.
```

---

## Walkthrough: How "on" Benefits from the Smear Gate

Without smear gate, at layer 0, "on" only knows it is the word "on":
```
x["on"] = [0.2, 0.4, 0.9, 0.1]   ← only encodes "on"
```

The first attention layer must figure out that "sat" came before "on".

With smear gate, before layer 0:
```
x["on"] = embed("on") + 0.3 × embed("sat")
        = [0.2, 0.4, 0.9, 0.1] + 0.3 × [0.7, 0.5, 0.3, 0.6]
        = [0.41, 0.55, 0.99, 0.28]
```

"on" already carries a trace of "sat" into layer 0. The transformer doesn't need to spend attention capacity rediscovering this — it can immediately build on it.

---

## One-Line Summary

> The Smear Gate blends each token's embedding with its predecessor's embedding using a single learned scalar — injecting a cheap "what came right before me?" signal before any transformer layer runs, freeing attention to focus on longer-range patterns.
