# Laneformer 2B — Pseudocode Summary

Quick-reference pseudocode for every module in `modeling_laneformer.py`. Each block shows `__init__` state and `forward` logic in plain steps. The central theme throughout: the hidden state carries an extra **lane dimension** `L`, giving tensors shape `(B, S, L, D)` instead of the usual `(B, S, D)`.

Speed callouts are marked **⚡** throughout. A summary of all speed factors is at the bottom.

---

## `ReduceMode` (Enum)
*Controls how a module collapses its lane outputs after computation.*

```
NO_REDUCE  → lanes stay independent; scale output by sqrt(num_lanes) to keep variance stable
PRESENT    → sum all L lanes immediately into one (keepdim) → (B, S, 1, D)
PAST       → delayed: add the sum of lanes from a *previous* layer's broadcast instead
```

`PRESENT` is used in early layers and when `use_early_comm=True`.
`PAST` is the core of Delayed Tensor Parallelism — the sync from layer N arrives at layer N + broadcast_delay.

> **⚡ Speed — the DTP modes eliminate blocking all-reduces:**
> Standard tensor parallelism forces GPUs to stop and sync (all-reduce) after *every* layer.
> `NO_REDUCE` runs each lane fully independently — zero sync cost.
> `PAST` replaces a blocking sync with an *addition* using a result that was computed 2 layers ago and is already sitting in memory. The expensive communication happened while compute was running — not while everyone was waiting.

---

## `LaneModule` (Base class)
*Mixin that gives attention and MLP modules a `reduce_lanes` method.*

```
__init__(no_reduce_scale, reduce_mode):
  1. Store no_reduce_scale = sqrt(num_lanes)   (compensates for variance in NO_REDUCE mode)
  2. Store reduce_mode and map it to an integer code (0, 1, 2) for fast dispatch

reduce_lanes(x, past):
  x shape: (B, S, L, D)

  if NO_REDUCE:
    return x * no_reduce_scale         (keep L lanes, just rescale)

  if PRESENT:
    return x.sum(dim=L, keepdim=True)  (immediate lane aggregation → (B, S, 1, D))

  if PAST:
    sum_past = sum(past, dim=L, keepdim=True) - past
                                        (sum of all OTHER lanes from a past layer)
    return x + sum_past                 (delayed cross-lane information)
```

> **⚡ Speed — integer dispatch, not branching strings:**
> `reduce_mode` is stored as an integer code (0, 1, 2) at init time, not as an enum string. At inference, the hot `reduce_lanes` path does a single integer comparison per call — no string lookup, no isinstance checks.

---

## `compute_rope_freqs`
*Computes per-dimension RoPE frequencies, with optional long-context scaling.*

```
compute_rope_freqs(dim, theta, scaling_args):
  1. Base freqs = 1 / (theta ^ (2i / dim))  for i in 0..dim/2
     → lower i = higher frequency (rotates fast), higher i = lower frequency (rotates slow)

  2. If scaling_args provided (for extended context):
       wavelen = 2π / freqs
       For each frequency:
         if wavelen > low_freq_wavelen:  divide by scaling_factor   (scale down low freqs)
         if wavelen < high_freq_wavelen: keep as-is                 (high freqs unchanged)
         else (medium band):             smooth interpolation between scaled and original

  3. Return freqs   shape: [dim/2]
```

---

## `precompute_freqs_cis`
*Builds a complex-valued rotation table for all positions up to `end`.*

```
precompute_freqs_cis(dim, end, theta, scaling_args):
  1. freqs = compute_rope_freqs(dim, theta, scaling_args)   → [dim/2]
  2. t = [0, 1, 2, ..., end-1]                              → [end]
  3. outer_product = t ⊗ freqs                              → [end, dim/2]  (angles per position)
  4. freqs_cis = polar(ones, outer_product)                  → [end, dim/2]  complex64
                                                               (unit circle points at each angle)
  5. Return freqs_cis
```

The complex representation encodes rotation as multiplication: `x * e^(iθ)` rotates `x` by angle θ.

> **⚡ Speed — computed once, shared across all layers:**
> `freqs_cis` is computed a single time in `LaneformerModel.forward` and passed as an argument to every `LaneformerDecoderLayer`. Standard implementations recompute or look up RoPE tables inside each attention module. Here it is one tensor allocation for the entire forward pass.

---

## `apply_rotary_emb`
*Applies RoPE rotation to query and key tensors using complex multiplication.*

```
apply_rotary_emb(xq, xk, freqs_cis):
  1. Reinterpret last dim of xq as complex pairs:
       xq_  = view_as_complex(xq.reshape(..., -1, 2))   → [..., dim/2] complex
  2. Same for xk:
       xk_  = view_as_complex(xk.reshape(..., -1, 2))   → [..., dim/2] complex

  3. Apply rotation via complex multiply:
       xq_out = view_as_real(xq_ * freqs_cis).flatten(last 2 dims)
       xk_out = view_as_real(xk_ * freqs_cis).flatten(last 2 dims)

  4. Cast back to original dtype and return (xq_out, xk_out)
```

> **⚡ Speed — complex multiply vs rotate_half:**
> Llama and most transformers implement RoPE via a `rotate_half` trick: split the vector in two halves, negate one, concatenate, then do two separate multiplications and an addition.
> Laneformer uses `view_as_complex` + a single complex multiply — mathematically identical but expressed as one fused operation. Hardware (especially on MI300X/H200) can execute complex multiply as a single instruction, avoiding the split-negate-cat overhead.

---

## `LaneLinear` / `LaneColumnLinear` / `LaneRowLinear`
*Lane-aware linear layers. All share one einsum; they differ in how weights are initialized.*

```
LaneLinear.__init__(in_features, out_features, num_lanes):
  1. weight shape = get_shape()   (defined by subclass)
  2. Call reset_parameters()      (defined by subclass)

LaneLinear.forward(x):
  x shape: (B, S, L, in_features/L)
  1. return einsum("bsli, loi -> bslo", x, self.weight)
     b=batch, s=seq, l=lane, i=in_features, o=out_features
     → Each lane L applies its own (in → out) weight matrix independently

LaneColumnLinear.reset_parameters():
  "Column parallel" — splits output features across lanes
  1. Init a flat [out_features, in_features] weight using kaiming_uniform
  2. Reshape to [out_features, num_lanes, in_features // num_lanes]
  3. Permute to [num_lanes, out_features, in_features // num_lanes]  (lane-first)
  → Each lane gets a slice of the input features, projects to full output

LaneRowLinear.reset_parameters():
  "Row parallel" — splits input features across lanes
  1. Init a flat [out_features, in_features] weight using kaiming_uniform
  2. Reshape directly to weight.shape (lanes already in the shape)
  → Each lane sees its slice of input, produces its slice of output
```

Together: `LaneRowLinear` fans out (input → per-lane outputs), `LaneColumnLinear` fans in (per-lane inputs → shared output).

> **⚡ Speed — all-GPU parallelism with zero sync:**
> In standard tensor parallelism, each GPU holds a shard of the weight matrix and must all-reduce results across GPUs before the next layer can start.
> Here, lanes are a *model-internal* concept — all 8 lanes live in one model and one forward pass. The einsum `"bsli,loi->bslo"` runs each lane's matmul independently in the same kernel with no inter-GPU communication at all. The GPU's SIMD width does the parallelism; the network card stays idle.
>
> The weight layout (lane-first: `[L, out, in/L]`) is chosen so each lane's weight slice is contiguous in memory — no gather/scatter overhead when indexing into per-lane weights.

---

## `LaneformerAttention`
*Multi-head attention with GQA, RoPE, KV cache, and lane-aware projections.*

```
__init__(config, layer_idx, reduce_mode):
  1. Inherit LaneModule with no_reduce_scale=sqrt(num_lanes), reduce_mode
  2. Set n_heads=32, n_kv_heads=16, head_dim=hidden_size//n_heads, scaling=head_dim^-0.5
  3. n_rep = n_heads // n_kv_heads = 2   (GQA repeat factor)
  4. wq = LaneRowLinear(hidden_size → n_heads * head_dim,   num_lanes)
  5. wk = LaneRowLinear(hidden_size → n_kv_heads * head_dim, num_lanes)
  6. wv = LaneRowLinear(hidden_size → n_kv_heads * head_dim, num_lanes)
  7. wo = LaneColumnLinear(n_heads * head_dim → hidden_size,  num_lanes)

forward(x, freqs_cis, attention_mask, past_key_values, cache_position):
  x shape: (B, S, L, D)

  # Lane-wise projections
  1. xq = wq(x)   → (B, S, L, n_heads * head_dim / L)   [lane-local query features]
  2. xk = wk(x)   → (B, S, L, n_kv_heads * head_dim / L)
  3. xv = wv(x)   → (B, S, L, n_kv_heads * head_dim / L)

  # Merge lanes into head dimension for attention
  4. Reshape xq → (B, S, n_heads,    head_dim)
  5. Reshape xk → (B, S, n_kv_heads, head_dim)
  6. Reshape xv → (B, S, n_kv_heads, head_dim)

  # Apply rotary position embeddings
  7. xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

  # Transpose to (B, heads, S, head_dim) for attention compute
  8. xq = xq.transpose(1, 2)
  9. xk = xk.transpose(1, 2)
  10. xv = xv.transpose(1, 2)

  # KV cache update
  11. If past_key_values:
        xk, xv = past_key_values.update(xk, xv, layer_idx, cache_kwargs)
        → xk, xv now contain full history: (B, n_kv_heads, past+S, head_dim)

  # Attention computation (GQA handled by backend: n_kv_heads repeated n_rep times)
  12. output, _ = attention_interface(self, xq, xk, xv, mask, scaling=head_dim^-0.5)
      → output shape: (B, S, n_heads, head_dim)

  # Re-introduce lane dimension
  13. output = output.view(B, S, num_lanes, n_heads * head_dim // num_lanes)
      → (B, S, L, D/L)

  # Lane-wise output projection
  14. output = wo(output)   → (B, S, L, D)

  15. Return output
  (caller calls reduce_lanes(output, past=...) after this)
```

> **⚡ Speed — Grouped Query Attention (GQA) halves KV cache bandwidth:**
> `n_kv_heads=16` vs `n_heads=32` means the KV cache is **2× smaller** than standard MHA.
> During autoregressive decode, the bottleneck is reading KV cache from HBM memory, not compute. A smaller KV cache = fewer bytes transferred per token = directly faster decode.
> Standard MHA on a 2B model at 4K context would require 32 heads × head_dim × 4K tokens per layer in cache. GQA cuts that to 16 × head_dim × 4K — same quality, half the memory traffic.
>
> **⚡ Speed — pluggable attention backend:**
> `attention_interface` is resolved at runtime from the config (`eager`, `sdpa`, or `flash`). On MI300X/H200, Flash Attention or SDPA fuses the softmax+dropout+matmul into a single kernel pass, saving multiple round-trips to HBM. The model doesn't hardcode the kernel.

---

## `LaneformerMLP`
*Gated feed-forward block (SwiGLU-style), operating lane-wise.*

```
__init__(config, reduce_mode):
  1. Inherit LaneModule with no_reduce_scale=sqrt(num_lanes), reduce_mode
  2. w1 = LaneRowLinear(hidden_size → ffn_dim, num_lanes)   (gate projection)
  3. w3 = LaneRowLinear(hidden_size → ffn_dim, num_lanes)   (value projection)
  4. w2 = LaneColumnLinear(ffn_dim → hidden_size, num_lanes) (down projection)

forward(x):
  x shape: (B, S, L, D)
  1. gate   = w1(x)             → (B, S, L, ffn_dim/L)
  2. value  = w3(x)             → (B, S, L, ffn_dim/L)
  3. hidden = silu(gate) * value (element-wise gating)
  4. return w2(hidden)          → (B, S, L, D)

  (caller calls reduce_lanes(output, past=...) after this)
```

`silu(x) = x * sigmoid(x)` — the gating nonlinearity from Llama's SwiGLU.

> **⚡ Speed — lane-local MLP, no sync between w1/w3/w2:**
> Each of the three projections (w1, w3, w2) operates independently per lane.
> In a tensor-parallel MLP across real GPUs, you'd need an all-reduce between the column-parallel (w1/w3) and row-parallel (w2) halves. Here that boundary is internal to the lane — w1, w3, and w2 share no cross-lane state until `reduce_lanes` is called after the whole MLP block, meaning the entire MLP forward runs without a single sync point.

---

## `LaneformerDecoderLayer`
*One transformer block: attention → lane-reduce → residual → MLP → lane-reduce → residual.*

```
__init__(config, layer_idx, attention_reduce_mode, mlp_reduce_mode,
         broadcast_attention_to_future, broadcast_mlp_to_future):
  1. attention       = LaneformerAttention(config, layer_idx, attention_reduce_mode)
  2. feed_forward    = LaneformerMLP(config, mlp_reduce_mode)
  3. attention_norm  = RMSNorm or LaneRMSNorm (depending on config.replicated_rmsn_scale)
  4. ffn_norm        = same
  5. Store broadcast_attention_to_future, broadcast_mlp_to_future flags

forward(hidden_states, attention_mask, freqs_cis, past_key_values,
        past_attention, past_mlp, cache_position, ...):
  hidden_states shape: (B, S, L, D)

  # Attention sub-block
  1. residual = hidden_states
  2. hidden_states = attention_norm(hidden_states)
  3. hidden_states = attention(hidden_states, freqs_cis, attention_mask, ...)

  # DTP: capture this layer's output BEFORE reducing, to broadcast to a future layer
  4. future_attention = hidden_states  if broadcast_attention_to_future  else None

  # Lane reduction (mode set per-layer by LaneformerModel.__init__)
  5. hidden_states = attention.reduce_lanes(hidden_states, past=past_attention)
     → collapses or combines the L lane dimension based on reduce_mode

  6. hidden_states = residual + hidden_states    (residual connection)

  # MLP sub-block
  7. residual = hidden_states
  8. hidden_states = ffn_norm(hidden_states)
  9. hidden_states = feed_forward(hidden_states)

  10. future_mlp = hidden_states  if broadcast_mlp_to_future  else None

  11. hidden_states = feed_forward.reduce_lanes(hidden_states, past=past_mlp)

  12. hidden_states = residual + hidden_states    (residual connection)

  13. Return (hidden_states, future_attention, future_mlp)
      ↑ future_* tensors flow forward to layer (this + broadcast_delay)
```

> **⚡ Speed — the DTP pipeline in one place:**
> Step 4 (`future_attention = hidden_states`) is where DTP's broadcast originates. The tensor is saved *before* `reduce_lanes` — i.e., it still has the full `(B, S, L, D)` lane structure.
> Two layers later, this tensor arrives as `past_attention` and is consumed in step 5 via `reduce_lanes(..., past=past_attention)` as a cheap addition (`x + sum_of_other_lanes`).
> The key insight: the "communication" between GPUs in a real tensor-parallel system is replaced here by a *time-shifted addition* — the model learned to tolerate stale cross-lane information because it was trained that way from scratch.
>
> **⚡ Speed — sliding window attention on 10 of 15 layers:**
> `attention_norm` selects between a full causal mask and a sliding-window mask per layer (set at init from `config.swa_layers`). 10 layers use sliding-window attention, which is O(S·W) instead of O(S²). For S=4096 and a window W=512, that's 8× fewer attention scores computed in those layers — directly reducing memory bandwidth and compute per token.

---

## `LaneformerModel`
*Full decoder stack. Manages embeddings, lane expansion, DTP broadcast queues, masks, and RoPE.*

```
__init__(config):
  1. tok_embeddings = nn.Embedding(vocab_size, hidden_size)
  2. For each layer_id in 0..num_hidden_layers:

       if layer_id < broadcast_delay:
         attention_reduce_mode = PRESENT  if use_early_comm  else NO_REDUCE
         mlp_reduce_mode       = PRESENT  if use_early_comm  else NO_REDUCE
         broadcast_*_to_future = False    (too early to broadcast)
       else:
         attention_reduce_mode = PAST     if use_attention_comm  else NO_REDUCE
         mlp_reduce_mode       = PAST     if use_mlp_comm        else NO_REDUCE
         broadcast_*_to_future = (num_layers - broadcast_delay > layer_id)
                                          (True until last broadcast_delay layers)

       layers[layer_id] = LaneformerDecoderLayer(
           config, layer_id, attention_reduce_mode, mlp_reduce_mode,
           broadcast_attention_to_future, broadcast_mlp_to_future)

  3. norm = final RMSNorm or LaneRMSNorm

forward(input_ids, attention_mask, position_ids, past_key_values, ...):

  # Embedding + lane expansion
  1. inputs_embeds = tok_embeddings(input_ids)    → (B, S, D)
  2. hidden_states = inputs_embeds[:, :, None, :].expand(-1, -1, num_lanes, -1)
                                                   → (B, S, L, D)
     (replicate embedding into L identical lanes — DTP diverges them through training)

  # Cache setup
  3. If use_cache and no past_key_values: past_key_values = DynamicCache()
  4. cache_position = arange(past_seen_tokens, past_seen_tokens + S)
  5. position_ids   = cache_position.unsqueeze(0)   → (1, S)

  # Attention masks
  6. Build causal_mask for full-attention layers
  7. If any sliding-window layers: build sliding_window_causal_mask separately

  # RoPE table (shared across all layers)
  8. freqs_cis = precompute_freqs_cis(position_ids)   → complex rotation table

  # DTP broadcast queues
  9. past_attentions = []   (stores future_attention from each layer)
     past_mlps       = []   (stores future_mlp from each layer)

  # Decoder stack
  10. For i in 0..num_hidden_layers:
        past_attention = past_attentions[i - broadcast_delay]  if i >= broadcast_delay  else None
        past_mlp       = past_mlps[i - broadcast_delay]        if i >= broadcast_delay  else None

        hidden_states, future_attention, future_mlp = layers[i](
            hidden_states,
            attention_mask = causal_mask_mapping[layer_type[i]],
            freqs_cis      = freqs_cis,
            past_key_values = past_key_values_for_forward,
            past_attention = past_attention,
            past_mlp       = past_mlp,
            ...
        )
        past_attentions.append(future_attention)   (queued for layer i + broadcast_delay)
        past_mlps.append(future_mlp)

  # Final lane aggregation + norm
  11. if pre_norm_lane_agg:
        hidden_states = hidden_states.sum(dim=L)   → (B, S, D)
        hidden_states = norm(hidden_states)
      elif lm_head_type == "replicate":
        hidden_states = norm(hidden_states)
        hidden_states = hidden_states.mean(dim=L)  → (B, S, D)
      else:
        hidden_states = norm(hidden_states)         → stays (B, S, L, D) for lane/vocab_parallel head

  12. Return hidden_states, past_key_values
```

> **⚡ Speed — `expand` not `repeat` for lane replication:**
> Step 2 uses `.expand()` (not `.repeat()`). `expand` is a zero-copy view — no new memory is allocated, no data is copied. All L lanes initially point to the same embedding data. Only after the first write (first linear layer) do they diverge. This keeps the embedding step essentially free.
>
> **⚡ Speed — KV cache avoids recomputing past tokens:**
> Step 3 initializes a `DynamicCache` if none exists. During autoregressive generation, every new token only runs through the model once — past K/V states are read from cache, not recomputed. Without this, generating token N would require re-attending over all N-1 previous tokens from scratch each step. The cache reduces per-token cost from O(N²) to O(N).
>
> **⚡ Speed — reduce mode baked in at init, not decided at runtime:**
> The per-layer `attention_reduce_mode` and `mlp_reduce_mode` are assigned once in `__init__` and stored as integer codes in each layer's `LaneModule`. At inference, there is no conditional logic choosing how to reduce — the mode is fixed per layer at construction time.

---

## `LaneformerForCausalLM`
*Adds the language model head. Three head types to match how lane aggregation was done.*

```
__init__(config):
  1. model = LaneformerModel(config)
  2. lm_head depends on lm_head_type:
       "replicate"      → nn.Linear(hidden_size → vocab_size)
                          input: (B, S, D)    (lanes already averaged by model)
       "lane"           → LaneLMHead         (lane-distributed linear)
                          input: (B, S, L, D)
       "vocab_parallel" → LaneRowLinear(hidden_size → vocab_size, num_lanes)
                          input: (B, S, L, D) → (B, S, L, vocab/L) → reshape (B, S, vocab)

forward(input_ids, attention_mask, position_ids, past_key_values, labels, logits_to_keep, ...):

  1. outputs = model(input_ids, attention_mask, position_ids, past_key_values, ...)
     hidden_states = outputs.last_hidden_state

  2. Optionally slice to last logits_to_keep tokens (efficient for generation)

  3. if lm_head_type == "replicate":
       logits = lm_head(hidden_states)             → (B, S, vocab_size)
     else:
       logits = lm_head(hidden_states)             → (B, S, L, vocab_size/L)  [lane or vocab_parallel]
       if vocab_parallel:
         logits = logits.reshape(B, S, L * vocab_chunk)  → (B, S, vocab_size)

  4. If labels provided:
       loss = cross_entropy(logits, labels)

  5. Return CausalLMOutputWithPast(loss, logits, past_key_values)
```

> **⚡ Speed — `logits_to_keep` avoids a full vocab projection at every step:**
> The LM head projects `hidden_size → vocab_size` (e.g., 2048 → 32000). During generation, you only need the logits for the *last* token to sample the next one, not all S positions.
> `logits_to_keep=1` slices `hidden_states[:, -1:, ...]` before the LM head — reducing the projection cost by a factor of S. For S=4096 that is a 4096× cheaper LM head call per decode step.

---

## Full Forward Pass Call Chain

```
LaneformerForCausalLM.forward
  └─ LaneformerModel.forward
       │
       ├─ tok_embeddings(input_ids)             → (B, S, D)
       ├─ expand to lanes [zero-copy]           → (B, S, L, D)          ⚡ no alloc
       ├─ precompute_freqs_cis(position_ids)    → complex rotation table ⚡ once for all layers
       │
       └─ LaneformerDecoderLayer.forward  (× num_hidden_layers)
            │
            ├─ RMSNorm / LaneRMSNorm             (attention_norm)
            │
            ├─ LaneformerAttention.forward
            │    ├─ LaneRowLinear (wq, wk, wv)  → lane-wise projections  ⚡ no cross-lane sync
            │    ├─ apply_rotary_emb             → rotated q, k           ⚡ complex multiply (1 op)
            │    │    └─ precomputed freqs_cis
            │    ├─ past_key_values.update       → full K/V history        ⚡ cache: O(N) not O(N²)
            │    └─ attention_interface          → sdpa/flash backend      ⚡ fused kernel
            │         (GQA: 16 KV heads, 2× smaller KV cache)             ⚡ half the memory traffic
            │    └─ LaneColumnLinear (wo)        → lane-wise output proj   ⚡ no cross-lane sync
            │
            ├─ save future_attention             → queued for layer+2      ⚡ DTP: async broadcast
            ├─ attention.reduce_lanes(PAST)      → cheap addition          ⚡ no blocking all-reduce
            ├─ residual + attn_output
            │
            ├─ RMSNorm / LaneRMSNorm             (ffn_norm)
            │
            ├─ LaneformerMLP.forward
            │    ├─ LaneRowLinear (w1, w3)       → gate+value projections  ⚡ no cross-lane sync
            │    ├─ silu(w1(x)) * w3(x)          → SwiGLU gating
            │    └─ LaneColumnLinear (w2)         → down projection
            │
            ├─ save future_mlp                   → queued for layer+2      ⚡ DTP: async broadcast
            ├─ feed_forward.reduce_lanes(PAST)   → cheap addition          ⚡ no blocking all-reduce
            └─ residual + mlp_output
       │
       ├─ Final norm + lane aggregation (mean/sum)
       └─ Return hidden_states
  │
  └─ lm_head(hidden_states[:, -1:, :])  → logits   ⚡ only last token projected
```

---

## DTP Data Flow — One Example (broadcast_delay = 2)

```
Layer 0:  reduce_mode = NO_REDUCE   → broadcast future_attention_0, future_mlp_0
Layer 1:  reduce_mode = NO_REDUCE   → broadcast future_attention_1, future_mlp_1
Layer 2:  reduce_mode = PAST        → receives past_attention = future_attention_0
                                       past_mlp      = future_mlp_0
                                       (layer 0's lane outputs arrive here, 2 layers late)
Layer 3:  reduce_mode = PAST        → receives future_attention_1, future_mlp_1
...

Timeline:
  Layer 0 compute:      ████████
  Layer 0 broadcast:             ──────────────────→ arrives at Layer 2
  Layer 1 compute:      ████████
  Layer 1 broadcast:             ──────────────────→ arrives at Layer 3
  Layer 2 compute:               ████████  (overlaps with Layer 0 broadcast in-flight)

Cross-lane sync is hidden inside compute — not paid as a blocking stall.
```

---

## Speed Factor Summary

| Factor | Where | What it avoids |
|---|---|---|
| **Delayed Tensor Parallelism** | `ReduceMode.PAST` + `reduce_lanes` | Blocking all-reduce after every layer |
| **No-sync lane matmuls** | `LaneLinear` einsum | Any cross-GPU communication during projections |
| **Lane-first weight layout** | `LaneColumnLinear.reset_parameters` | Gather/scatter overhead; weights are contiguous per lane |
| **Grouped Query Attention** | `LaneformerAttention` (16 KV heads) | 2× KV cache memory bandwidth during decode |
| **Sliding window attention** | 10 of 15 layers | O(S²) → O(S·W) attention cost in most layers |
| **Fused attention kernel** | `attention_interface` dispatch | Multiple HBM round-trips in naive softmax+matmul |
| **Complex RoPE multiply** | `apply_rotary_emb` | Extra split-negate-concat ops from `rotate_half` |
| **Single shared `freqs_cis`** | `LaneformerModel.forward` | Redundant RoPE table allocation per layer |
| **Zero-copy lane expand** | `.expand()` in embedding step | Memory allocation for L copies of embeddings |
| **KV cache** | `DynamicCache` | Recomputing past token attention at each decode step |
| **`logits_to_keep` slicing** | `LaneformerForCausalLM.forward` | Full S-token LM head projection when only last token needed |
| **Reduce mode as int code** | `LaneModule.__init__` | Runtime string/enum lookup on the hot inference path |
