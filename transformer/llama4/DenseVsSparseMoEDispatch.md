# Dense Dispatch vs Sparse Dispatch — Llama4 MoE vs Mixtral-style MoE

Llama4's MoE uses a **dense dispatch** strategy. Most other MoE implementations (Mixtral, DeepSeek, Switch Transformer) use **sparse dispatch**. This file explains the difference, the trade-offs, and why Llama4 made this choice.

---

## What Both Approaches Are Solving

In a top-1 MoE layer with 16 experts and a batch of `N` tokens, the router assigns each token to exactly one expert. The challenge: how do you efficiently run 16 separate expert networks, each processing a different subset of tokens, on a GPU?

---

## Sparse Dispatch (Mixtral, DeepSeek, Switch Transformer)

```
Tokens:   [t0, t1, t2, t3, t4, t5, t6, t7]
Router:    expert=[2,  0,  2,  1,  0,  2,  1,  0]

Sort/gather by expert:
  Expert 0 gets: [t1, t4, t7]   → [3, H]
  Expert 1 gets: [t3, t6]       → [2, H]
  Expert 2 gets: [t0, t2, t5]   → [3, H]

Run each expert on its tokens independently.

Scatter results back to original positions.
```

Only the assigned tokens go to each expert. Expert 0 does 3 forward passes, Expert 1 does 2, Expert 2 does 3. Experts with no assigned tokens do nothing.

**Compute:** proportional to actual token assignments — with top-1 routing, exactly `N * expert_cost` total work, spread across experts.

**Memory:** each expert only holds tokens assigned to it — variable-length batches.

**Problem:** the sort/gather step produces **dynamic shapes** — the number of tokens per expert varies by input. Dynamic shapes break `torch.compile` graph tracing and CUDA kernel fusion. They also complicate tensor parallelism because the sizes of tensors being communicated across GPUs are not known at compile time.

---

## Dense Dispatch (Llama4)

```python
# In Llama4TextMoe.forward:
hidden_states = hidden_states.reshape(-1, self.hidden_dim)   # [N, H]

router_scores, _ = self.router(hidden_states)                # [N, 16]
# router_scores: sigmoid output, 15 entries are ~0 (non-selected), 1 is non-zero per row

routed_in = hidden_states.repeat(router_scores.shape[1], 1)  # [N*16, H]  — ALL tokens, 16 copies
routed_in = routed_in * router_scores.transpose(0, 1).reshape(-1, 1)  # scale by routing weights
# non-selected expert copies are multiplied by ~0 → effectively zeroed

routed_out = self.experts(routed_in)   # [N*16, H]  — ALL experts process ALL tokens

# experts internally does: view as [16, N, H] → bmm with 16 weight matrices
# each expert's output for non-assigned tokens is ~0 (because input was ~0)

routed_out.reshape(16, -1, H).sum(dim=0)   # [N, H]  — sum contributions (most are ~0)
```

Every token is sent to every expert, but non-assigned tokens are scaled to ~0 before the expert computation. The experts run on a fixed-shape `[N*16, H]` tensor every time regardless of routing decisions.

**Compute:** `16 * N * expert_cost` — **16× more FLOPs than sparse** for a top-1 model. All 16 experts process all N tokens, even though 15 of those per token are multiplied by ~0 and contribute nothing meaningful.

**Memory:** fixed shapes throughout — `[N*16, H]` always, no sorting, no variable-length batches.

**No dynamic shapes:** the tensor sizes are fully determined by `N` and `num_experts` alone, independent of routing decisions. Fully `torch.compile`-friendly and CUDA-graph compatible.

---

## Why Llama4 Chose Dense Dispatch

### 1. `torch.compile` and CUDA graphs

Sparse dispatch requires `topk` results to determine which tokens go where, then uses those results to index/gather — dynamic indexing that breaks static graph compilation. Llama4's codebase leans heavily on `torch.compile` (`_can_compile_fullgraph = True` on `Llama4PreTrainedModel`). Dense dispatch never has dynamic shapes and compiles cleanly into a single fused CUDA graph.

### 2. Tensor parallelism is simpler

Under TP, the `experts.gate_up_proj` tensor `[16, H, 2*expert_dim]` is sharded. With dense dispatch, the shard boundaries are always the same; with sparse dispatch, the variable number of tokens per expert makes sharding the compute, not just the weights, more complex.

### 3. The code comment is honest

The code says explicitly:
```
# This should really not be run on a single machine, as we are reaching compute bound
```

Dense dispatch is **designed for multi-machine expert parallelism (EP)**. Under EP:
- Each machine owns a subset of experts (e.g. 2 experts per machine for 8 machines × 2 experts = 16)
- The `AlltoAll` communication sends each token to the machine owning its assigned expert
- Each machine then runs its 2 experts on all tokens sent to it — and since it only holds 2 of the 16 experts, the "dense" dispatch within one machine is actually only 2× wasteful, not 16×
- The wasted compute (tokens assigned to another expert) becomes zero because those tokens were never sent to this machine

Dense dispatch's wastefulness collapses under proper EP: the routing happens at the dispatch communication layer (only send a token to the machine holding its expert), and within each machine, the experts see only the tokens they should process. The `base_model_ep_plan` in `Llama4TextConfig` reflects this — experts get `"grouped_gemm"` and the router gets `"ep_router"` treatment.

---

## The `sigmoid` Routing Choice Compounds This

Recall from `MoE.md`: Llama4's router uses `sigmoid` not `softmax`. Under sigmoid, a non-selected expert's score is `sigmoid(-inf) = 0` — exactly zero, not a small positive number.

This means the "wasted" compute in dense dispatch is genuinely zero-valued:
```
routed_in = all_tokens * router_scores   # non-selected experts: token * 0.0 = 0.0
expert_output = expert(routed_in)        # expert(0.0) = down_proj(act(gate_up @ 0.0)) = 0.0
```

The expert forward pass on zeroed inputs produces zeroed outputs (assuming no bias terms — which Llama4's experts don't have). So the 15× "wasted" expert calls per token multiply the zero input through all weight matrices and produce zero output. The `sum(dim=0)` at the end then sums zero contributions from non-selected experts and the real contribution from the selected expert — correct result.

This is not true for softmax routing: non-selected experts would have small but non-zero scores, so zeroing them via scatter (as Mixtral does) would be slightly lossy. Sigmoid's clean zero for non-selected experts makes dense dispatch exact.

---

## Summary: Trade-Off Table

| | Sparse dispatch (Mixtral) | Dense dispatch (Llama4) |
|---|---|---|
| FLOPs per token | `expert_cost` (only assigned expert) | `16 * expert_cost` (all experts) |
| Wasted compute (single machine) | None | 15/16 = 93.75% |
| Wasted compute (with EP) | Near zero | Near zero |
| Tensor shapes | Dynamic (varies with routing) | Static (always `N*16`) |
| `torch.compile` compatibility | Difficult | Full (`_can_compile_fullgraph`) |
| CUDA graph compatible | No | Yes |
| TP/EP integration | Complex | Straightforward |
| Practical use case | Single/few GPU inference | Large-scale distributed training/inference |
