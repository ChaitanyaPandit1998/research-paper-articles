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
