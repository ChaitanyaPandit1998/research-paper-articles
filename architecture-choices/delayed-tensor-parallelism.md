# Delayed Tensor Parallelism (DTP) — Explained Simply

**Related files:** `laneformer-2b.md`, `laneformer-2b-pseudocode.md`

DTP is the core architectural innovation in Laneformer 2B. It solves the problem that standard tensor parallelism is too slow at low batch sizes because of blocking cross-GPU synchronization after every layer.

---

## The Old Way: The Town Meeting Problem

Imagine a small town that bakes bread for the whole country. The recipe is so complex that it takes **8 specialist bakers**, each responsible for one part of the process:

- Baker 1 handles flour measurement
- Baker 2 handles yeast mixing
- Baker 3 handles water temperature
- ...and so on up to Baker 8

The recipe has **15 steps** (like a transformer has 15 layers). After each step, something important happens: **all 8 bakers must stop, walk to the town square, share what they learned, and agree on adjustments before anyone can start the next step.**

```
Step 1: All bakers work          ████████
Town meeting (sync):                      ████████   ← everyone waits
Step 2: All bakers work                            ████████
Town meeting (sync):                                         ████████   ← everyone waits
...
```

This works, but the town meetings eat up half the day. The bread gets made, but slowly.

This is **standard tensor parallelism** — the "town meetings" are what's called an **all-reduce**: every GPU must stop, broadcast its partial result, collect everyone else's, and sum them up before the next layer can begin.

---

## The Problem Gets Worse at Low Batch Size

Here's the cruel part. When the bakery gets a rush order (high batch = many loaves), the actual baking takes a long time — long enough that the meeting overhead is a small fraction of the day. The bakers don't notice it much.

But when there's a single custom order (low batch = one loaf), each step takes only a few minutes. Now the town meeting takes *longer than the baking itself*. The bakers spend more time walking to the square than actually baking.

```
High batch (many loaves):   ████████████████ [baking]  ████ [meeting]   ← meeting is small
Low batch (one loaf):            ████ [baking]          ████ [meeting]   ← meeting dominates
```

This is exactly what happens with LLM inference at low batch sizes. The GPU arithmetic is fast. The inter-GPU communication (the "meeting") is the bottleneck.

---

## The DTP Solution: The Postcard System

One clever baker proposes an idea: **what if instead of a town meeting, each baker just mailed a postcard to the baker who needs their information two steps from now?**

Here's the rule:
- At the end of Step 1, each baker writes their notes on a postcard and drops it in the mailbox addressed to *Step 3*.
- Then they immediately start Step 2. No meeting. No waiting.
- The postcard travels through the postal system while Steps 2 and 3 are being baked.
- At Step 3, the postcard arrives. Each baker reads it and folds the notes in — a simple adjustment, not a full re-do.

```
Step 1 baking:       ████████
Step 1 postcard:              ────────────────────────→ arrives at Step 3
Step 2 baking:       ████████   (no waiting — starts immediately)
Step 3 baking:                ████████  (postcard arrives, fold in the notes)
Step 4 baking:                          ████████
...
```

The postal delivery happens in the background while the bakers keep working. The "meeting cost" disappears from the critical path.

This is **Delayed Tensor Parallelism**. The "postcard" is `future_attention` / `future_mlp` — tensors saved at layer N and consumed at layer N+2. The "folding in the notes" is the `reduce_lanes(PAST)` call — a simple addition.

---

## What the Postcards Actually Say

One detail matters: the postcard doesn't say *"here is the final agreed answer."* It says *"here is what **I** computed — add it to what the others already gave you."*

```python
# At layer N (writing the postcard):
future_attention = hidden_states   # full (B, S, L, D) — all 8 lanes' raw outputs

# At layer N+2 (reading the postcard):
sum_of_others = sum(past, dim=lanes) - past   # what every OTHER lane said, from 2 steps ago
hidden_states = hidden_states + sum_of_others  # fold it in as an addition
```

It's not a full sync. It's each lane getting a *delayed echo* of what the other lanes computed two steps ago. The model is trained from scratch to work correctly with this delay — it learns that two-step-old information is still useful enough to guide computation.

---

## Why You Can't Just Add This to an Existing Model

Here's the catch the story reveals: this only works if the bakers were **trained from childhood to expect delayed postcards**.

If you took bakers trained under the old town-meeting system and suddenly switched to postcards, they'd be confused. Their whole intuition was built on having *current* information from all colleagues at every step. Two-step-old notes would throw off their judgment.

Laneformer was trained from scratch with the postcard system baked in. The model's weights learned to function correctly with stale cross-lane information — because that's all they ever saw during training.

This is why DTP cannot be retrofitted onto Llama, Mistral, or any existing pretrained model. The architecture is the same shape, but the *learned behavior* is fundamentally different.

---

## The Full Picture

```
Standard tensor parallelism (8 GPUs):
  Layer 1: GPU_1..8 compute  →  [ALL-REDUCE: everyone stops]  →  Layer 2 starts
  Layer 2: GPU_1..8 compute  →  [ALL-REDUCE: everyone stops]  →  Layer 3 starts
  ...× 15 layers = 15 blocking stops

DTP (8 lanes, 1 model, broadcast_delay=2):
  Layer 1: all lanes compute  →  save postcard  →  Layer 2 starts immediately
  Layer 2: all lanes compute  →  save postcard  →  Layer 3 starts immediately
  Layer 3: all lanes compute  →  receives Layer 1's postcard  →  cheap addition  →  Layer 4 starts
  Layer 4: all lanes compute  →  receives Layer 2's postcard  →  cheap addition  →  Layer 5 starts
  ...× 15 layers = 0 blocking stops
```

The postcards travel in parallel with computation. By the time they're needed, they've already arrived. The bakers never stop to wait — and the bread gets made at 3,000 tokens per second.

---

## Where the All-Reduce Actually Happens (Under the Hood)

The story simplified it to "one town meeting per layer." In reality there are **two all-reduces per layer** — one inside the attention block and one inside the MLP block. Both happen at the same structural point: the boundary between a column-parallel and a row-parallel linear layer.

### The Column → Row → All-Reduce Pattern

In standard tensor parallelism, weight matrices are split across GPUs in two complementary ways:

**Column-parallel** — each GPU gets a vertical slice of the weight. It takes the full input and produces a partial output independently. No sync needed yet.

**Row-parallel** — each GPU gets a horizontal slice of the weight. It takes a partial input and produces a partial result that must be *summed* with every other GPU's result to get the correct full output. This is where the all-reduce fires.

```
Column-parallel (no sync needed):
  Full input X → GPU_0 computes X @ W_col0 → partial output (its shard)
               → GPU_1 computes X @ W_col1 → partial output (its shard)
               → ...each GPU works independently

Row-parallel (sync required):
  GPU_0: partial_0 = X_shard0 @ W_row0
  GPU_1: partial_1 = X_shard1 @ W_row1
  ...
  ★ ALL-REDUCE: full_output = partial_0 + partial_1 + ... + partial_7 ★
  ← no GPU can proceed until every GPU has contributed its piece
```

### All-Reduce #1 — Inside Attention

```
Q, K, V projections (column-parallel):
  All 8 GPUs receive the same hidden_states
  GPU_0 → owns heads 0–3    (its shard of W_q, W_k, W_v)
  GPU_1 → owns heads 4–7
  ...
  No sync — each GPU works on its own heads

Attention scores + softmax:
  Each GPU runs attention only over its own heads
  No sync — heads are fully independent of each other

Output projection W_o (row-parallel):
  GPU_0 has attention output for heads 0–3  → partial_0
  GPU_1 has attention output for heads 4–7  → partial_1
  ...
  ★ ALL-REDUCE #1 ★  sum partial_0..7 → full attention output
  ← everyone stops here
```

### All-Reduce #2 — Inside the MLP

```
W1 / W3 gate+value projections (column-parallel):
  GPU_0 → ffn_dim shard 0
  GPU_1 → ffn_dim shard 1
  ...
  No sync

SiLU gating: silu(W1) * W3
  Each GPU applies this to its own intermediate shard
  No sync

W2 down-projection (row-parallel):
  GPU_0: partial_0 = ffn_shard_0 @ W2_row0
  GPU_1: partial_1 = ffn_shard_1 @ W2_row1
  ...
  ★ ALL-REDUCE #2 ★  sum partial_0..7 → full MLP output
  ← everyone stops here
```

### Full Per-Layer Picture: 2 Blocking Stops

```
  hidden_states (identical on all 8 GPUs)
        │
        ▼
  Q, K, V projections  [column-parallel, no sync]
        │
        ▼
  Attention scores + softmax  [per-GPU heads, no sync]
        │
        ▼
  Output projection W_o  [row-parallel]
        │
        ▼
  ★ ALL-REDUCE #1 ★  ← GPU_0..7 sum partial results, everyone waits
        │
        ▼
  Residual + RMSNorm
        │
        ▼
  W1, W3 projections  [column-parallel, no sync]
        │
        ▼
  SiLU gate  [no sync]
        │
        ▼
  W2 down-projection  [row-parallel]
        │
        ▼
  ★ ALL-REDUCE #2 ★  ← GPU_0..7 sum partial results, everyone waits
        │
        ▼
  Residual → next layer (same pattern repeats)
```

For a 15-layer model: **30 all-reduces total** per forward pass (2 per layer × 15 layers). Each one sends `hidden_size × 2 bytes` (in bfloat16) across NVLink/InfiniBand and blocks until every GPU has the full result.

### Why This Hurts at Low Batch Size

The all-reduce latency is roughly **fixed** — it costs the same whether you're processing 1 token or 128 tokens, because the same amount of data needs to cross the interconnect regardless of batch size.

```
High batch (128 sequences):
  GPU compute per layer:  ████████████████████  (long — lots of arithmetic)
  All-reduce:                                ██  (small fraction of total time)

Low batch (1 sequence):
  GPU compute per layer:  ████                  (short — almost no arithmetic)
  All-reduce:                 ████              (same fixed cost — now dominates)
```

This is exactly the bottleneck DTP eliminates. Instead of 30 blocking all-reduces, Laneformer uses 0 — replacing each one with a delayed addition using a result that was already computed two layers ago and is sitting in memory, free of charge.

---

## How It Maps to the Code

| Story element | Code equivalent |
|---|---|
| 8 bakers | 8 lanes (`num_lanes=8`) |
| One baking step | One `LaneformerDecoderLayer.forward` |
| Town meeting | All-reduce (blocking cross-GPU sync) |
| Postcard | `future_attention` / `future_mlp` tensor |
| Postcard delay of 2 steps | `broadcast_delay=2` in `LaneformerModel` |
| Mailing the postcard | `past_attentions.append(future_attention)` |
| Reading and folding in | `reduce_lanes(hidden_states, past=past_attention)` with `ReduceMode.PAST` |
| The fold-in arithmetic | `x + (sum(past, dim=lanes) - past)` |
| Bakers trained on postcards | Model trained from scratch with DTP — not finetuned |
