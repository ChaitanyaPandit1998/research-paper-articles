# Llama4 Mixture of Experts (MoE)

Llama4's MoE is entirely new — Llama3 has only dense MLP layers. This file covers the MoE architecture from router to expert computation.

---

## Overview: Interleaved Dense and MoE Layers

Each `Llama4TextDecoderLayer` is either a **dense MLP layer** or a **MoE layer**, determined at construction time:

```python
self.is_moe_layer = layer_idx in config.moe_layers
if self.is_moe_layer:
    self.feed_forward = Llama4TextMoe(config)
else:
    self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)
```

The two MLP types use different intermediate sizes:
- Dense MLP: `intermediate_size_mlp = 16384` (larger)
- MoE expert: `intermediate_size = 8192` (smaller, but there are 16 experts)

With default `interleave_moe_layer_step = 1`, every layer is a MoE layer (`moe_layers = [0, 1, 2, ..., 47]` for a 48-layer model). Changing `interleave_moe_layer_step = 2` would make every other layer MoE, etc.

---

## `Llama4TextMoe` — The MoE Block

```python
class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok   # = 1 (top-1 routing)
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts  # = 16
        self.experts = Llama4TextExperts(config)     # all 16 experts, fused
        self.router = Llama4Router(config)           # routing linear
        self.shared_expert = Llama4TextMLP(config)   # always-active dense expert
```

`Llama4TextMoe.forward`:
```python
def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)   # [B*S, H]

    router_scores, router_logits = self.router(hidden_states)    # [B*S, num_experts]

    # Expand each token n_top_k times (here: top_k=1 so just 1 copy)
    routed_in = hidden_states.repeat(router_scores.shape[1], 1)  # [B*S*num_experts, H]
    routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)   # scale by routing weights

    routed_out = self.experts(routed_in)                         # [B*S*num_experts, H]

    out = self.shared_expert(hidden_states)                      # [B*S, H]
    out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0))

    return out, router_logits
```

The forward pass always computes **two paths**:
1. **Sparse path** via the router + experts (only the top-k expert(s) contribute)
2. **Dense path** via `shared_expert` (always active, every token, every step)

The outputs are summed: every token's final representation is `shared_expert(x) + expert_k(x)`.

---

## `Llama4Router` — Sigmoid Routing, Not Softmax

```python
class Llama4Router(nn.Linear):
    def forward(self, hidden_states):
        router_logits = super().forward(hidden_states)   # [B*S, num_experts]
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

        # Zero out non-top-k logits with -inf, then apply sigmoid
        router_scores = torch.full_like(router_logits, float("-inf"))
        router_scores = router_scores.scatter_(1, router_indices, router_top_value)
        router_scores = torch.nn.functional.sigmoid(router_scores.float()).to(router_scores.dtype)

        return router_scores, router_logits
```

**Sigmoid instead of softmax — why it matters:**

Most MoE implementations (e.g. Mixtral) use softmax over all experts to produce routing weights, making weights sum to 1. Llama4 uses **sigmoid applied independently to each expert's logit** after zeroing non-top-k entries:

- With softmax: `weight_i = exp(logit_i) / sum(exp(logit_j))` — all experts compete, weights sum to 1.
- With sigmoid: `weight_i = 1 / (1 + exp(-logit_i))` — each expert is scored independently, weights do NOT sum to 1, range (0, 1).

This means the router can express "strong confidence in this expert" or "weak confidence but still routing to it" without it affecting the weights of other experts. It also means the total contribution of the routed expert varies in magnitude — sometimes contributing more, sometimes less — which gives the model more expressive capacity at the cost of less explicit load balancing by the routing weights alone.

The non-selected experts get `-inf` before sigmoid → `sigmoid(-inf) = 0`, so they contribute exactly 0 to `routed_in` after scaling.

---

## `Llama4TextExperts` — Batched Expert Computation

All 16 experts are stored as **a single batched parameter tensor**, not as 16 separate `nn.Linear` modules:

```python
class Llama4TextExperts(nn.Module):
    def __init__(self, config):
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, hidden_size, 2 * expert_dim))
        self.down_proj    = nn.Parameter(torch.empty((num_experts, expert_dim, hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]   # SiLU
```

Shape: `gate_up_proj` is `[num_experts=16, hidden=5120, 2*expert_dim=16384]`. Each "slice" along dim 0 is one expert's fused gate+up projection.

**Why fused gate+up?** Instead of separate `gate_proj` (5120→8192) and `up_proj` (5120→8192), Llama4 fuses them into one `[5120, 16384]` matrix per expert, then chunks along the last dim:

```python
def forward(self, hidden_states):
    # hidden_states: [B*S, H], but pre-sorted into [num_experts, tokens_per_expert, H]
    hidden_states = hidden_states.view(self.gate_up_proj.shape[0], -1, self.hidden_size)
    # → [num_experts, T, H]

    gate_up = torch.bmm(hidden_states, self.gate_up_proj)
    # → [num_experts, T, 2*expert_dim]

    gate, up = gate_up.chunk(2, dim=-1)
    # gate: [num_experts, T, expert_dim]
    # up:   [num_experts, T, expert_dim]

    next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
    # SwiGLU activation, then project back to hidden_size
    # → [num_experts, T, H]

    return next_states.view(-1, self.hidden_size)   # → [num_experts*T, H]
```

**`torch.bmm` for expert dispatch:** the key operation is `torch.bmm(hidden_states, self.gate_up_proj)`, a **batched matrix multiply** where the batch dimension is `num_experts`. This is computationally equivalent to running each expert's linear layer separately, but expressed as one BLAS call, which GPUs execute much more efficiently. For this to work, the input `hidden_states` must be **pre-sorted** so tokens assigned to expert 0 are in slice 0, tokens for expert 1 are in slice 1, etc.

**How sorting happens (implicit, not explicit):** In `Llama4TextMoe.forward`, the trick is:
```python
routed_in = hidden_states.repeat(router_scores.shape[1], 1)
```
This repeats the *entire* `[B*S, H]` tensor `num_experts` times, giving `[B*S*num_experts, H]`. Then routing weights (which are 0 for non-selected experts) zero out the copies that don't belong to each expert. The experts kernel then reshapes this into `[num_experts, B*S, H]` — every expert sees all tokens, but non-assigned tokens are multiplied by 0 (effectively masked). This is a **dense dispatch** approach, not sparse.

**Trade-off:** Dense dispatch wastes compute (all experts process all tokens, most scaled by 0), but it avoids the complexity of sparse routing (token sorting, variable-length batches per expert) and is much more friendly to `torch.compile` and tensor parallelism. The comment in the code: "This should really not be run on a single machine, as we are reaching compute bound."

---

## `shared_expert` — Always-Active Expert

Every MoE layer has one dense `Llama4TextMLP` that runs on **every token regardless of routing**:

```python
self.shared_expert = Llama4TextMLP(config)   # uses config.intermediate_size = 8192
```

This is the same SwiGLU MLP as in Llama3, just with a smaller `intermediate_size` (8192 vs 16384 for the dense fallback layer). Its output is **added** to the sparse expert output unconditionally:

```python
out = self.shared_expert(hidden_states)
out.add_(routed_out.sum(dim=0))
```

The shared expert serves as a "base layer" that every token always passes through, while the sparse expert provides token-specific specialisation on top. This pattern is similar to DeepSeek's "shared + routed expert" design and differs from Mixtral/Mixtral-8x7B which has no shared expert (all experts are routed).

---

## Summary: MoE Data Flow

```
hidden_states [B*S, H]
       │
       ├──→ shared_expert (dense MLP, always runs)    → [B*S, H]
       │
       ├──→ router (Linear + topk + sigmoid)           → router_scores [B*S, num_experts]
       │         (top_k=1: only 1 expert selected per token)
       │
       └──→ routed_in = hidden * router_scores (broadcast + scale)  → [B*S*num_experts, H]
                 │
                 └──→ Llama4TextExperts (batched bmm across 16 experts)   → [B*S*num_experts, H]
                           │
                           └──→ sum across expert dim                     → [B*S, H]

output = shared_expert_out + experts_out    [B*S, H]
```

**Router auxiliary loss:** `router_logits` are returned from `Llama4TextMoe` and collected by `Llama4TextModel` (via `_can_record_outputs = {"router_logits": Llama4TextMoe}`). During training, a load-balancing auxiliary loss (coefficient `router_aux_loss_coef = 0.001`) encourages equal utilisation across experts, preventing all tokens from routing to a single expert.

---

## Plain-English Walkthrough

### The Big Idea

In a regular transformer, **every token passes through the same FFN** in every layer. MoE changes this: instead of one large FFN, you have **multiple smaller ones called "experts"**, and each token only uses **one (or a few) of them**. The model has more total parameters, but each token only touches a fraction — making it **cheaper to run than its size suggests**.

---

### 1. Not Every Layer Is MoE — They're Interleaved

`configuration_llama4.py:185-194` — the list of MoE layer indices is built at config init:

```python
self.moe_layers = list(
    range(
        self.interleave_moe_layer_step - 1,  # start
        self.num_hidden_layers,               # end
        self.interleave_moe_layer_step,       # step
    )
)
```

`modeling_llama4.py:419-423` — each decoder layer checks this at construction:

```python
self.is_moe_layer = layer_idx in config.moe_layers
if self.is_moe_layer:
    self.feed_forward = Llama4TextMoe(config)
else:
    self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)
```

```
┌─────────────────────────────────────────────────────────┐
│                   48 Decoder Layers                     │
│                                                         │
│  interleave_moe_layer_step = 1  →  ALL layers are MoE  │
│                                                         │
│  interleave_moe_layer_step = 2:                         │
│  Layer 0 ──► Dense MLP (Llama4TextMLP)                 │
│  Layer 1 ──► MoE FFN   (Llama4TextMoe)                 │
│  Layer 2 ──► Dense MLP (Llama4TextMLP)                 │
│  Layer 3 ──► MoE FFN   (Llama4TextMoe)  ...            │
└─────────────────────────────────────────────────────────┘
```

---

### 2. The MoE Layer Has Three Parts

`modeling_llama4.py:157-165`

```python
class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        self.experts       = Llama4TextExperts(config)  # 16 expert FFNs, batched
        self.router        = Llama4Router(config)       # decides which expert gets each token
        self.shared_expert = Llama4TextMLP(config)      # always-on, every token uses it
```

```
                  ┌──────────────────────────────────────┐
                  │         Llama4TextMoe                │
                  │                                      │
   token          │    ┌──────────┐                      │
   hidden_state ──┼───►│  Router  │──► picks 1 expert   │
                  │    └──────────┘         │            │
                  │                         ▼            │
                  │    ┌────────────────────────────┐    │
                  │    │  Llama4TextExperts (x16)   │    │
                  │    │  Only the selected one     │    │
                  │    │  actually contributes      │    │
                  │    └────────────────────────────┘    │
                  │              │                       │
                  │    ┌─────────┴──────┐                │
                  │    │  Shared Expert │ ◄── always runs│
                  │    │ (Llama4TextMLP)│                │
                  │    └────────────────┘                │
                  │              │                       │
                  │   output = shared + routed           │
                  └──────────────────────────────────────┘
```

The unique thing in LLaMA 4: a **shared expert** processes **every token**, on top of the routed expert. All tokens get a common base computation; the routed expert adds specialisation on top.

---

### 3. The Router — "Which Expert Should Handle This Token?"

`modeling_llama4.py:142-153`

```python
class Llama4Router(nn.Linear):
    def forward(self, hidden_states):
        router_logits = super().forward(hidden_states)            # [tokens, 16]
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = torch.full_like(router_logits, float("-inf"))
                            .scatter_(1, router_indices, router_top_value)
        router_scores = torch.sigmoid(router_scores.float())
        return router_scores, router_logits
```

The router is a **linear classifier** projecting each token from `hidden_size` to `num_experts`:

```
token vector [5120]
       │
       ▼
  Linear layer  (5120 → 16)
       │
       ▼
 16 scores, one per expert
       │
       ▼
  pick top-1  →  expert index + score

  e.g.  Expert 7: 0.93  ◄── this token routes here
        Expert 3: 0.71      (zeroed out, not selected)
        Expert 11: 0.65     (zeroed out, not selected)
```

Non-selected experts get `-inf` → `sigmoid(-inf) = 0`, so they contribute nothing. The selected expert's score is an independent confidence value (sigmoid, not softmax) — it does not compete with other experts.

---

### 4. The Experts — Batched FFNs

`modeling_llama4.py:56-85`

Rather than 16 separate `nn.Linear` layers, all expert weights live in **one 3D tensor**:

```python
self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, hidden_size, 2 * expert_dim))
# shape: [16, 5120, 16384]

self.down_proj = nn.Parameter(torch.empty((num_experts, expert_dim, hidden_size)))
# shape: [16, 8192, 5120]
```

Each expert is a **SwiGLU FFN**, same as LLaMA's standard MLP:

```
token ──► [gate_proj, up_proj] ──► gate * silu(up) ──► down_proj ──► output
```

One `torch.bmm` handles all 16 experts at once — much more GPU-efficient than 16 separate matmuls.

---

### 5. Full MoE Forward Pass

`modeling_llama4.py:167-175`

```python
def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)          # [T, H]
    router_scores, router_logits = self.router(hidden_states)           # [T, 16]
    routed_in = hidden_states.repeat(router_scores.shape[1], 1)         # [T*16, H]
    routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)# scale by score (0 for non-selected)
    routed_out = self.experts(routed_in)                                # [T*16, H]
    out = self.shared_expert(hidden_states)                             # [T, H]
    out.add_(routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0))
    return out, router_logits
```

```
Input tokens  [T, 5120]
      │
      ├──────────────────────────────────────►  Shared Expert (MLP)  ──► out_shared [T, 5120]
      │
      ▼
   Router  →  scores [T, 16]  (mostly 0, 1 non-zero per token)
      │
      ▼
   Scale each token copy by its router score
      │
      ▼
   Experts.forward()  →  out_routed [T, 5120]
      │
      ▼
   out_shared + out_routed  =  final output [T, 5120]
```

---

### 6. Key Config Numbers (LLaMA 4 Scout 17B-16E)

`configuration_llama4.py:139-194`

| Param | Value | Meaning |
|---|---|---|
| `num_local_experts` | 16 | 16 experts per MoE layer |
| `num_experts_per_tok` | 1 | each token routed to exactly 1 expert |
| `hidden_size` | 5120 | token vector size |
| `intermediate_size` | 8192 | expert FFN width |
| `intermediate_size_mlp` | 16384 | dense layer FFN width |
| `num_hidden_layers` | 48 | total layers |
| `interleave_moe_layer_step` | 1 | all layers are MoE by default |

---

### 7. Why This Design?

```
Traditional dense model:
  Every token ──► 1 huge FFN (all params, every time)

MoE model:
  Every token ──► 1 small expert (1/16 of routed params) + shared expert
  Total stored params = 16× a single expert → much larger model
  Active params per token = much smaller → cheaper per forward pass

e.g. "17B active params, 109B total params"
```

Over training, experts tend to **specialise** — different experts handle different token types (code, math, language, etc.). This is emergent, not hardcoded.

---

## Line-by-Line Walkthrough with a Concrete Example

I'll use a tiny example throughout so every shape change is visible:

```
Batch = 1, Sequence = 2 tokens  (T = 2)
hidden_size  = 4    (real: 5120)
num_experts  = 4    (real: 16)
expert_dim   = 6    (real: 8192)
top_k        = 1
```

Input arriving at the MoE layer: shape `[1, 2, 4]` — 1 batch, 2 tokens, each a vector of size 4.

---

### `Llama4TextExperts.__init__`

```python
self.num_experts      = config.num_local_experts    # 4
self.intermediate_size = config.intermediate_size   # 6
self.hidden_size      = config.hidden_size          # 4
self.expert_dim       = self.intermediate_size      # 6  (alias)
```

**Why:** Storing config values as instance attributes so the forward pass can reference them without touching config again.

```python
self.gate_up_proj = nn.Parameter(
    torch.zeros(self.num_experts, self.hidden_size, 2 * self.expert_dim)
)
# shape: [4, 4, 12]
#         ^  ^   ^
#         |  |   └── 2×expert_dim: gate and up projections fused into one matrix
#         |  └────── input size (hidden_size)
#         └───────── one weight matrix per expert
```

**Why:** All 4 experts' weights are packed into one 3D tensor instead of 4 separate `nn.Linear` layers. This allows a single `torch.bmm` to run all experts in parallel — far more efficient on GPU than 4 sequential matmuls. The `2×` is because gate and up projections are fused: first half = gate, second half = up.

```python
self.down_proj = nn.Parameter(
    torch.empty((self.num_experts, self.expert_dim, self.hidden_size))
)
# shape: [4, 6, 4]
```

**Why:** Same batched design as `gate_up_proj` — one down-projection matrix per expert, all stacked together so `bmm` can project all experts back to `hidden_size` in one call.

```python
self.act_fn = ACT2FN[config.hidden_act]   # SiLU
```

**Why:** Storing the activation function as an attribute so the forward pass can call it without string lookups each time.

---

### `Llama4Router.__init__`

```python
class Llama4Router(nn.Linear):
    def __init__(self, config):
        super().__init__(config.hidden_size, config.num_local_experts, bias=False)
        # This IS an nn.Linear(4, 4, bias=False)
        # Weight matrix shape: [4, 4]  (out=num_experts, in=hidden_size)

        self.num_experts = config.num_local_experts   # 4
        self.top_k       = config.num_experts_per_tok # 1
```

**Why:** Subclassing `nn.Linear` directly means we get a trained weight matrix for free and can call `super().forward(x)` to get scores. No extra layers — the router is intentionally minimal (just one linear transform) so routing decisions don't dominate compute.

---

### `Llama4TextMoe.__init__`

```python
self.top_k        = config.num_experts_per_tok    # 1
self.hidden_dim   = config.hidden_size            # 4
self.num_experts  = config.num_local_experts      # 4
self.experts      = Llama4TextExperts(config)     # the 4 batched experts
self.router       = Llama4Router(config)          # the linear classifier
self.shared_expert = Llama4TextMLP(config)        # normal MLP, always runs
```

**Why:** Three independent sub-modules. Nothing is computed in `__init__` — just wiring up the pieces so `forward` can call them.

---

### `Llama4TextMoe.forward` — line by line

**Input:** `hidden_states` shape `[1, 2, 4]`

---

**Line 1**
```python
hidden_states = hidden_states.reshape(-1, self.hidden_dim)
# [1, 2, 4]  →  [2, 4]
```
**Why:** The router and experts work on individual tokens, not on (batch, sequence) pairs. Collapsing those two dimensions means the rest of the forward pass treats every token identically regardless of which batch item or position it came from.

---

**Line 2**
```python
router_scores, router_logits = self.router(hidden_states)
```

**Why:** Before we can send tokens to experts, we need to know which expert each token should go to. The router does this by scoring every token against every expert.

Stepping inside `Llama4Router.forward`:

```python
router_logits = super().forward(hidden_states)
# Why: Run the learned linear layer — produces a raw score per expert per token.
# Linear [2,4] @ [4,4].T  =  [2, 4]

# token 0: [ 0.9,  0.2, -0.1,  0.5]
# token 1: [-0.3,  0.8,  0.7,  0.1]
```

```python
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
# Why: We only want each token to use top_k=1 expert. topk picks the
#      highest-scoring expert index and its score for each token.

# router_top_value: [[0.9],  [0.8]]
# router_indices:   [[0],    [1]]    ← token 0 picks expert 0, token 1 picks expert 1
```

```python
router_scores = torch.full_like(router_logits, float("-inf"))
# Why: Create a blank slate of -inf so that after sigmoid, non-selected
#      experts will become exactly 0 (sigmoid(-inf) = 0).

router_scores = router_scores.scatter_(1, router_indices, router_top_value)
# Why: Place each token's top score back at its chosen expert position.
#      Every other position stays -inf.
# [[ 0.9, -inf, -inf, -inf],   ← token 0 chose expert 0
#  [-inf,  0.8, -inf, -inf]]   ← token 1 chose expert 1

router_scores = torch.sigmoid(router_scores.float())
# Why: Convert the chosen expert's score to a weight in (0, 1) that will
#      scale the expert's output. sigmoid instead of softmax means experts
#      are scored independently — the weight expresses confidence, not competition.
# sigmoid(-inf) = 0.0,  sigmoid(0.9) ≈ 0.71,  sigmoid(0.8) ≈ 0.69

# [[0.71, 0.0,  0.0,  0.0],
#  [0.0,  0.69, 0.0,  0.0]]
```

`router_scores [2, 4]` — mostly zeros, one non-zero weight per token.  
`router_logits [2, 4]` — raw pre-sigmoid values, returned for training loss.

---

**Line 3 — The Repeat (why we copy tokens)**
```python
routed_in = hidden_states.repeat(router_scores.shape[1], 1)
# router_scores.shape[1] = num_experts = 4
# .repeat(4, 1) stacks the token matrix 4 times along rows
# [2, 4]  →  [8, 4]
```

**Why we need to copy at all:**

`Llama4TextExperts.forward` runs a single batched matrix multiply (`bmm`) across all experts at once. For that to work, the input must be shaped `[num_experts, T, hidden]` — a fixed rectangular block where every expert has the same number of token slots.

The problem: routing is sparse and uneven. Token 0 goes to expert 0, token 1 goes to expert 1. Experts 2 and 3 received nothing. We can't give each expert a different-length list of tokens — `bmm` needs a uniform shape.

The solution used here is called **dense dispatch**:
- Give every expert a copy of every token → uniform shape guaranteed
- Multiply by routing scores in the next line → zero out the tokens that don't belong to each expert
- Experts 2 and 3 will multiply their copy by 0 → their output is zeros, wasted compute but correct

The alternative (**sparse dispatch**) would sort tokens by their chosen expert and pack only the real assignments. That's more efficient but much harder to implement and breaks `torch.compile`. The code comment even says: *"This should really not be run on a single machine, as we are reaching compute bound."*

```
# [2, 4]  →  [8, 4]

# row 0: token0_vec  ← copy for expert 0
# row 1: token1_vec  ← copy for expert 0
# row 2: token0_vec  ← copy for expert 1
# row 3: token1_vec  ← copy for expert 1
# row 4: token0_vec  ← copy for expert 2
# row 5: token1_vec  ← copy for expert 2
# row 6: token0_vec  ← copy for expert 3
# row 7: token1_vec  ← copy for expert 3
```

---

**Line 4**
```python
routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)
```

**Why:** Zero out every copy that wasn't selected. This is the masking step — it turns the naive "give every expert every token" trick into a correct sparse operation where only the chosen expert gets a non-zero input.

```python
router_scores.transpose(0,1)   # [2,4] → [4,2]: now (experts × tokens)
# [[0.71, 0.0 ],   ← expert 0's weight for token0, token1
#  [0.0,  0.69],   ← expert 1's weight
#  [0.0,  0.0 ],   ← expert 2 (no token chose it)
#  [0.0,  0.0 ]]   ← expert 3

.reshape(-1, 1)    # [4,2] → [8,1]: flatten to match routed_in's row layout
# [[0.71], [0.0], [0.0], [0.69], [0.0], [0.0], [0.0], [0.0]]
```

Multiply with `routed_in [8, 4]`:
```
row 0: token0_vec × 0.71  ← expert 0 gets token 0, scaled by router confidence
row 1: token1_vec × 0.0   ← expert 0 gets token 1 zeroed (token 1 didn't pick expert 0)
row 2: token0_vec × 0.0   ← expert 1 gets token 0 zeroed
row 3: token1_vec × 0.69  ← expert 1 gets token 1, scaled by router confidence
row 4–7: all zeros         ← experts 2 and 3 receive nothing
```

Shape still `[8, 4]` — only 2 rows carry real signal.

---

**Line 5**
```python
routed_out = self.experts(routed_in)
```

**Why:** Now that each expert's slot has the right tokens (others zeroed), we run all 4 experts through their FFN in one batched call.

Stepping inside `Llama4TextExperts.forward`:

```python
hidden_states = hidden_states.view(self.gate_up_proj.shape[0], -1, self.hidden_size)
# Why: Reshape from flat [8,4] into [4,2,4] so that dimension 0 indexes
#      the expert, making bmm process each expert's tokens independently.
# [8, 4]  →  [4, 2, 4]

# Slice [0]: [token0×0.71,  token1×0.0 ]   ← expert 0's inputs
# Slice [1]: [token0×0.0,   token1×0.69]   ← expert 1's inputs
# Slice [2]: [zeros,        zeros       ]   ← expert 2
# Slice [3]: [zeros,        zeros       ]   ← expert 3
```

```python
gate_up = torch.bmm(hidden_states, self.gate_up_proj)
# Why: Project each token up to a larger space. bmm handles all 4 experts
#      at once — [4,2,4] @ [4,4,12] → [4,2,12].
#      The last dim (12) holds gate (first 6) and up (last 6) fused.
```

```python
gate, up = gate_up.chunk(2, dim=-1)
# Why: Split the fused projection into its two halves.
#      gate [4,2,6] controls how much of the content passes through.
#      up   [4,2,6] is the actual content to be gated.
```

```python
next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
# Why: Apply the SwiGLU activation — SiLU(gate) creates a smooth on/off
#      valve, multiplied element-wise with up to selectively pass content.
#      bmm with down_proj projects the result back to hidden_size.
# [4,2,6] → [4,2,4]
```

```python
next_states = next_states.view(-1, self.hidden_size)
# Why: Flatten back to [8,4] to match the shape that Llama4TextMoe.forward
#      expects when it reshapes and sums across experts.
```

`routed_out [8, 4]` — only 2 of 8 rows are non-zero.

---

**Line 6**
```python
out = self.shared_expert(hidden_states)
```

**Why:** Every token should pass through a common dense FFN regardless of routing. This gives the model a guaranteed base computation that doesn't depend on which expert was chosen — stabilises training and ensures no token is ever "stranded" with a poor expert assignment.

`hidden_states` here is still the **original** `[2, 4]` (we never modified it, only made copies for `routed_in`).

```python
# Inside Llama4TextMLP.forward:
result = down_proj(activation_fn(gate_proj(x)) * up_proj(x))
# Input [2, 4]  →  Output [2, 4]
```

---

**Line 7**
```python
out.add_(
    routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1]).sum(dim=0)
)
```

**Why:** Combine the two paths — shared expert (which ran on all tokens) and the routed expert (which ran on only the selected tokens) — by summing them. Each token's final representation is `shared_output + routed_output`.

```python
routed_out.reshape(router_scores.shape[1], -1, routed_out.shape[-1])
# Why: Go from flat [8,4] back to [4,2,4] so we can sum along the expert axis.

.sum(dim=0)
# Why: Collapse the expert dimension [4,2,4] → [2,4].
#      For each token, we're summing contributions from all experts.
#      Since only one expert per token is non-zero, this is effectively
#      just picking that one expert's output.

out.add_(...)
# Why: In-place addition avoids allocating a new tensor — saves memory.
```

Each token's final value:
```
token_out = shared_expert(token) + router_score × chosen_expert(token)
```

---

**Line 8**
```python
return out, router_logits
```

**Why:** Return both values. `out` is what the decoder layer needs for the residual connection. `router_logits` are needed by the training loop to compute the auxiliary load-balancing loss (`router_aux_loss_coef = 0.001`) — without it, all tokens would collapse onto one or two "easy" experts.

---

### Full Picture in One Diagram

```
Input [1, 2, 4]
  │
  │  reshape(-1, hidden_dim)          WHY: flatten batch×seq so we work token-by-token
  ▼
[2, 4]
  │
  ├────────────────────────────────────────► shared_expert (MLP, always runs)
  │                                          WHY: guaranteed base computation for every token
  │                                              └─► out [2, 4]
  │
  ▼
Router (Linear 4→4)
  WHY: score each token against each expert to decide routing
  └─► logits [2, 4]
        └─► topk(k=1)  →  indices [[0],[1]]
              WHY: pick the single best expert per token
              └─► scatter + sigmoid
                    WHY: zero out non-winners; sigmoid gives independent confidence weight
                    └─► scores [2, 4]
                          [[0.71, 0,    0, 0],
                           [0,    0.69, 0, 0]]
  │
  ▼
.repeat(4, 1)  →  routed_in [8, 4]
  WHY: dense dispatch — give every expert a copy of every token so bmm
       can run all experts in one rectangular batch (zeros mask the extras)
  │
  ▼
× scores.T.reshape(-1,1) [8,1]
  WHY: zero out the copies that don't belong to each expert
  │
  ▼
Llama4TextExperts.forward:
  .view(4, 2, 4)              WHY: separate expert dimension so bmm indexes per-expert
  bmm with gate_up_proj       WHY: project all experts' tokens up in one GPU call
  .chunk(2, dim=-1)           WHY: split fused gate+up into their two halves
  up * SiLU(gate)             WHY: SwiGLU gating — gate controls how much content passes
  bmm with down_proj          WHY: project back to hidden_size
  .view(-1, 4)                WHY: flatten back for the reshape+sum that follows
  │
  ▼
.reshape(4, 2, 4).sum(dim=0)
  WHY: collapse expert dimension — each token picks up only its one expert's output
  →  [2, 4]
  │
  ▼
out (shared) + routed_sum  =  final output [2, 4]
  WHY: every token gets: stable base (shared) + specialised signal (routed)
```
