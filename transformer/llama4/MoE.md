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
