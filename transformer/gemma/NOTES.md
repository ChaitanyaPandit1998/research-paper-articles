# Gemma Model — Notes

## Files in `src/transformers/models/gemma/`

### `modular_gemma.py` — Source of truth (edit this one)
The master file where Gemma's actual code lives. Gemma is very similar to LLaMA, so this file imports LLaMA's classes and only overrides what's different:
- **`GemmaRMSNorm`** — Uses `1 + weight` instead of just `weight` (subtle math difference from LLaMA)
- **`GemmaTextScaledWordEmbedding`** — Scales embeddings by `√hidden_size` (e.g., `√3072 ≈ 55.4`), which LLaMA doesn't do
- **`GemmaAttention`** — Mostly LLaMA's attention, but adds support for bidirectional attention (no causal mask)
- **`GemmaModel`, `GemmaForCausalLM`, etc.** — Thin wrappers over LLaMA equivalents with the above customizations applied

### `configuration_gemma.py` — Auto-generated, do not edit
Generated from `modular_gemma.py` by `make fix-repo`. Contains `GemmaConfig`: the settings/hyperparameters for the model.

### `modeling_gemma.py` — Auto-generated, do not edit
Also generated from `modular_gemma.py`. Contains the actual PyTorch model classes (`GemmaModel`, `GemmaForCausalLM`, etc.) fully expanded without inheritance — a standalone, self-contained implementation.

### `tokenization_gemma.py` — Converts text ↔ token IDs
Implements `GemmaTokenizer` using BPE (Byte-Pair Encoding) with:
- Spaces replaced by `▁` (Unicode character, common in SentencePiece tokenizers)
- Byte fallback: unknown characters encoded as raw bytes instead of `<unk>`
- Pads on the left side (standard for decoder-only models)

### `convert_gemma_weights_to_hf.py` — One-time migration script
Converts Google's original Gemma checkpoint format into HuggingFace format. Only used when importing a new Gemma release from Google.

### `__init__.py` — Package entry point
Wires everything together so you can write `from transformers import GemmaModel`. Uses lazy loading so importing `transformers` doesn't load all heavy PyTorch code upfront.

> **Key rule:** Always edit `modular_gemma.py`. The `configuration_` and `modeling_` files are auto-generated and will be overwritten by `make fix-repo`.

---

## `GemmaConfig` — Fields Explained

### Architecture size

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 256,000 | Number of tokens the model knows (its "vocabulary") |
| `hidden_size` | 3072 | Width of the internal representation vectors |
| `intermediate_size` | 24,576 | Width of the feed-forward (MLP) layers — typically 8× hidden |
| `num_hidden_layers` | 28 | Number of transformer blocks stacked on top of each other |
| `num_attention_heads` | 16 | Number of attention heads per layer |
| `num_key_value_heads` | 16 | For GQA — here equal to attention heads, meaning no GQA |
| `head_dim` | 256 | Size of each attention head's vector |

### Behavior flags

| Field | Default | Meaning |
|---|---|---|
| `hidden_act` | `"gelu_pytorch_tanh"` | Activation function used in MLP layers |
| `max_position_embeddings` | 8192 | Maximum sequence length (tokens) the model can handle |
| `rms_norm_eps` | 1e-6 | Small constant to prevent division by zero in normalization |
| `use_cache` | `True` | Whether to cache key/value states during generation (speeds up inference) |
| `attention_bias` | `False` | Whether attention projection layers have a bias term |
| `attention_dropout` | 0.0 | Dropout probabilaity in attention (0 = disabled) |
| `tie_word_embeddings` | `True` | Input and output token embeddings share the same weights (saves memory) |
| `use_bidirectional_attention` | `None` | If `True`, model sees all tokens (no causal mask) — used for encoding tasks |

### Special tokens

| Field | Default | Meaning |
|---|---|---|
| `pad_token_id` | 0 | Token ID used to pad shorter sequences in a batch |
| `bos_token_id` | 2 | "Beginning of sequence" token |
| `eos_token_id` | 1 | "End of sequence" token — model stops generating here |

### Training

| Field | Default | Meaning |
|---|---|---|
| `initializer_range` | 0.02 | Standard deviation for weight initialization |
| `rope_parameters` | `None` | Config for RoPE (Rotary Position Embeddings) |

---

## Multi-GPU Parallelism

### The Problem
Gemma-7B has billions of parameters. A single GPU might not have enough memory. Two strategies are used to split the model across multiple GPUs.

---

### Tensor Parallelism (TP) — Split a single layer across GPUs

All GPUs work **simultaneously** on the same layer, each holding a slice of the weight matrix.

#### `colwise` — split by columns

Weight matrix `[3072 × 24576]` (hidden → intermediate) split across 3 GPUs:

```
         GPU 1               GPU 2               GPU 3
  [ col 0..8191 ]    [ col 8192..16383 ]   [ col 16384..24575 ]
```

Each GPU multiplies the **full input** by its **slice of columns** independently.
Used for: `gate_proj`, `up_proj`, `q_proj`, `k_proj`, `v_proj` — layers that expand the hidden state outward.

#### `rowwise` — split by rows

Weight matrix `[24576 × 3072]` (intermediate → hidden) split across 3 GPUs:

```
         GPU 1               GPU 2               GPU 3
  [ row 0..8191 ]    [ row 8192..16383 ]   [ row 16384..24575 ]

  Each GPU: rows × input_slice → partial output [3072]
                                         ↓ AllReduce (sum)
                                  final output [3072]
```

Each GPU computes a partial result, then they are **summed (AllReduce)** to get the final output.
Used for: `down_proj`, `o_proj` — layers that compress back to hidden size.

#### Why alternate colwise → rowwise?

They pair up naturally:
```
Input → [colwise: gate_proj] → each GPU has partial activations
      → [rowwise: down_proj] → GPUs sum results → full output
```
Only one AllReduce sync is needed at the end, not between every layer.

---

### Pipeline Parallelism (PP) — Split layers sequentially across GPUs

Entire layers are assigned to different GPUs. Data flows through them one after another like an assembly line:

```
  GPU 1            GPU 2                    GPU 3
  embed_tokens  →  layers[0..13]  →  layers[14..27] → norm

  Token IDs     →  Hidden states  →  Hidden states  →  Output
```

The `base_model_pp_plan` in config describes this:
```python
base_model_pp_plan = {
    "embed_tokens": (["input_ids"], ["inputs_embeds"]),
    "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    "norm": (["hidden_states"], ["hidden_states"]),
}
```
The tuples define **what tensor comes in** and **what tensor goes out** at each stage, so the framework knows what to send over the network between GPUs.

#### Reading the structure

Each entry is `"module_name": (input_arg_names, output_arg_names)`:

- **Key** = the submodule name in the model (matches `self.embed_tokens`, `self.layers`, `self.norm` in `GemmaModel`)
- **First list** = names of the tensors this stage *needs as input*
- **Second list** = names of the tensors this stage *produces as output*

#### How it maps onto the 3-GPU pipeline

```
GPU1                          GPU2                                  GPU3
embed_tokens                  layers[0..N]                          norm
─────────────                 ─────────────                         ─────────────
in:  input_ids                in:  hidden_states, attention_mask    in:  hidden_states
out: inputs_embeds            out: hidden_states                    out: hidden_states
```

1. **`embed_tokens` (GPU1)** — takes raw `input_ids` (token IDs), looks them up in the embedding table, and produces `inputs_embeds`. This is the first stage, so its only input is the thing the user gave the model.

2. **`layers` (GPU2)** — its declared input is `hidden_states`, not `inputs_embeds`. The framework knows (from naming convention inside `GemmaModel.forward`) that `inputs_embeds` becomes the *first* `hidden_states` before entering the decoder blocks. It also needs `attention_mask` — unlike `embed_tokens`, attention requires this side-channel tensor too, so the plan declares it as a second required input that must also be shipped to GPU2. It runs all the decoder blocks and outputs the final `hidden_states` after the last layer.

3. **`norm` (GPU3)** — takes that `hidden_states`, applies the final `RMSNorm`, and outputs `hidden_states` again (same name in and out — it transforms in place conceptually).

#### Why it's declared this way instead of just chaining blindly

- It lets the framework **automatically insert the right send/recv ops** between stages: GPU1 sends `inputs_embeds` → GPU2 relabels/accepts it as `hidden_states`; GPU2 sends `hidden_states` → GPU3 receives it as `hidden_states`.
- It also flags **extra tensors that must be forwarded alongside the main activation** — `attention_mask` doesn't originate from the previous stage's output, so the plan tells the framework "also send this raw input through to whichever stage needs it," not just `hidden_states`.
- This is exactly the assembly-line picture above: each stage's contract is "I need X, I hand off Y" — and the plan is what lets `transformers` build that pipeline automatically instead of manually wiring `.to(device)` calls and send/recv between every layer.

---

### TP vs PP — Key Difference

| | Tensor Parallelism | Pipeline Parallelism |
|---|---|---|
| What's split | A single weight matrix | Whole layers |
| GPUs work | Simultaneously on the same layer | Sequentially, one after another |
| Communication | `AllReduce` after each layer | Pass activations between stages |
| Best for | Very wide layers (large matrices) | Very deep models (many layers) |

In practice, large models use **both together**: TP within a node (fast NVLink) and PP across nodes (slower network).

---

### Tensor Parallelism — Pros and Cons

**Pros**
- No "bubble"/idle time — all GPUs compute at once on every forward pass, so there's no warm-up or drain period.
- Reduces per-GPU memory for very wide layers (huge `hidden_size`/`intermediate_size`), which is exactly the bottleneck PP can't fix on its own.
- Latency per token is low for a single request, since the layer's output is ready as soon as one `AllReduce` completes — there's no waiting on several sequential stages.

**Cons**
- Needs an `AllReduce` after every split layer (twice per transformer block — once for attention, once for MLP) — this is frequent, latency-sensitive communication.
- Only works well with very fast interconnects (NVLink/NVSwitch within a node); over slower networks (e.g. across nodes/Ethernet) the `AllReduce` cost dominates and throughput collapses.
- Doesn't reduce the *number* of layers stored per GPU — every GPU still holds a slice of *every* layer, so it doesn't help when the model is deep rather than wide.

**Where to use it:** within a single multi-GPU node (NVLink-connected), to shard very wide layers (large `hidden_size` / `intermediate_size` / many attention heads) when a single GPU can't fit one layer's weights or activations.

---

### Pipeline Parallelism — Pros and Cons

**Pros**
- Communication is just point-to-point activation hand-off between adjacent stages — far less frequent and far less bandwidth-hungry than TP's `AllReduce`, so it tolerates slow interconnects (e.g. across nodes, Ethernet/InfiniBand at rack scale).
- Scales naturally with model *depth* — each GPU only needs to hold a contiguous chunk of layers, so very deep models (many `num_hidden_layers`) fit even when no single GPU could hold the whole stack.
- Combines cleanly with micro-batching to keep all stages busy concurrently (different micro-batches at different stages), recovering much of the idle-time cost.

**Cons**
- Introduces a "pipeline bubble": early stages sit idle waiting for the first micro-batch to arrive, and late stages sit idle until the last one drains — wasted compute that grows with the number of stages.
- End-to-end latency for a single request is higher than TP, since the input must pass through every stage sequentially before the output is ready.
- Uneven layer-to-GPU partitioning (e.g. an embedding/lm_head stage that's cheaper than a stage full of transformer blocks) can leave some GPUs waiting on others — balancing stage cost requires care.

**Where to use it:** across nodes (slower network), to split very deep models (many layers) when the model doesn't fit on the GPUs available within one node, or when wide layers alone aren't the bottleneck.

---

### Putting It Together

| Scenario | Use |
|---|---|
| Single node, multiple GPUs, layer too wide to fit on one GPU | TP |
| Multiple nodes, model too deep to fit even after TP within a node | PP |
| Both — model is huge in both width and depth | TP within each node + PP across nodes (the common large-LLM setup) |

---

## How `colwise` vs `rowwise` is Determined

### The Core Rule

Any linear layer does: `output = input × Weight`

Two ways to split `Weight` across GPUs:

```
colwise: split Weight by columns → each GPU produces a PARTIAL OUTPUT (different columns)
rowwise: split Weight by rows   → each GPU takes a PARTIAL INPUT  (different rows) → sum results
```

They must always be **paired**: `colwise → rowwise`.

### Why the pairing works without extra communication

```
Input (full, on all GPUs)
    │
    ▼
[colwise layer]  ← each GPU gets different output columns
    │
    ▼
GPU1: partial_out_1    GPU2: partial_out_2    GPU3: partial_out_3
    │                       │                       │
    ▼                       ▼                       ▼
[rowwise layer]  ← each GPU already has the right partial input for its rows
    │
    ▼
AllReduce (sum) → full output
```

The colwise output **naturally feeds** into the rowwise input — no communication needed in between. Only **one AllReduce** at the end of the pair.

---

### Applied to Attention (`q, k, v, o`)

Attention has `num_heads = 16` heads. Each head is an independent unit:

```
q_proj: [hidden=3072] → [num_heads × head_dim = 16×256 = 4096]
```

**`q/k/v_proj` → colwise**: split by output columns = each GPU handles a subset of heads

```
GPU1: heads 0-5      GPU2: heads 6-10     GPU3: heads 11-15
[3072 → 1365 cols]   [3072 → 1365 cols]   [3072 → 1366 cols]
```

Each GPU independently computes attention for its subset of heads. This works because attention heads don't interact with each other.

**`o_proj` → rowwise**: takes the concatenated head outputs and projects back to hidden size

```
o_proj: [num_heads × head_dim = 4096] → [hidden=3072]
```

Each GPU already has its heads' outputs (from colwise), handles its rows of `o_proj` → partial result → AllReduce sums them.

---

### Applied to MLP (`gate, up, down`)

Gemma's MLP uses a gated structure (SwiGLU):

```python
output = down_proj( gate_proj(x) * up_proj(x) )
#                  └──── intermediate=24576 ────┘
#        └──────────── back to hidden=3072 ──────┘
```

**`gate_proj` + `up_proj` → colwise**: both expand hidden → intermediate, split by output columns

```
GPU1: cols 0..8191    GPU2: cols 8192..16383    GPU3: cols 16384..24575
```

The elementwise multiply `gate * up` happens locally on each GPU with its own slice — no communication needed.

**`down_proj` → rowwise**: contracts intermediate → hidden, each GPU handles its rows (matching its colwise slice)

```
GPU1: rows 0..8191    × local_activations → partial [3072]
GPU2: rows 8192..16383 × local_activations → partial [3072]
GPU3: rows 16384..24575 × local_activations → partial [3072]
                                                    ↓ AllReduce
                                             final output [3072]
```

---

### The Decision Rule

| If the layer... | Use |
|---|---|
| **Expands** hidden → larger dim, output can be independently split | `colwise` |
| **Contracts** larger dim → hidden, takes the split output as input | `rowwise` |
| Is the **first** in a pair | `colwise` |
| Is the **last** in a pair (needs AllReduce) | `rowwise` |

`q/k/v/gate/up` are colwise because they expand outward and their outputs are independent slices.
`o/down` are rowwise because they contract inward and need to sum across GPUs to produce the full output.

---

## How Colwise Output Feeds into Rowwise Input (this is Tensor Parallelism)

Both GPUs work on the *same pair of layers* simultaneously, each holding a different slice of the *same weight matrices*, synchronizing via `AllReduce`. That's what makes this TP rather than PP.

### Setup

Two layers in sequence with 2 GPUs:
- **Layer 1** (colwise): `W1` shape `[4 × 4]` — input dim 4, output dim 4
- **Layer 2** (rowwise): `W2` shape `[4 × 2]` — input dim 4, output dim 2
- **Input** `X` shape `[1 × 4]`

Full computation (no parallelism):
```
Y = X @ W1        # [1×4] @ [4×4] = [1×4]
Z = Y @ W2        # [1×4] @ [4×2] = [1×2]
```

---

### Step 1 — Colwise split of W1

Split `W1` by columns, half to each GPU:

```
W1 = [ col0  col1  col2  col3 ]
       └─── GPU1 ───┘ └─GPU2─┘
       W1_A = [col0, col1]    W1_B = [col2, col3]
       shape: [4×2]            shape: [4×2]
```

Both GPUs receive the **full input X**. Each computes:

```
GPU1:  Y_A = X @ W1_A   →  shape [1×2]   (left half of Y)
GPU2:  Y_B = X @ W1_B   →  shape [1×2]   (right half of Y)
```

If concatenated: `[Y_A | Y_B] = Y` — the full output. But we **don't concatenate**. Each GPU holds its slice.

---

### Step 2 — Rowwise split of W2

Split `W2` by rows, half to each GPU:

```
W2 = [ row0 ]  ← GPU1 holds W2_A (rows 0-1), shape [2×2]
     [ row1 ]
     ──────
     [ row2 ]  ← GPU2 holds W2_B (rows 2-3), shape [2×2]
     [ row3 ]
```

The full computation `Z = Y @ W2` expands as:

```
Z = [Y_A | Y_B] @ [W2_A]
                   [W2_B]
  = Y_A @ W2_A  +  Y_B @ W2_B
```

**GPU1 already has `Y_A`** (from colwise) and holds `W2_A`:
```
GPU1:  Z_partial_1 = Y_A @ W2_A   →  shape [1×2]
```

**GPU2 already has `Y_B`** (from colwise) and holds `W2_B`:
```
GPU2:  Z_partial_2 = Y_B @ W2_B   →  shape [1×2]
```

AllReduce sums them:
```
Z = Z_partial_1 + Z_partial_2     →  shape [1×2]  ✓
```

---

### Why It Fits Perfectly

```
Colwise splits W1 by COLUMNS  →  output Y is split by COLUMNS
Rowwise splits W2 by ROWS     →  needs input split by ROWS (= same dimension as columns of Y)
```

Columns of `Y` and rows of `W2` are the **same intermediate dimension**. So the colwise output slice on each GPU is exactly the rowwise input slice that GPU needs — no data shuffling between GPUs.

```
GPU1:  has Y columns [0,1]  →  needs W2 rows [0,1]  ✓
GPU2:  has Y columns [2,3]  →  needs W2 rows [2,3]  ✓
```

This is the mathematical reason colwise and rowwise are always paired — the output partitioning of one exactly matches the input partitioning of the other.

---

## Same Example, but as Pipeline Parallelism

### Setup
- **Layer 1** (full): `W1` shape `[4 × 4]` — lives entirely on **GPU1**
- **Layer 2** (full): `W2` shape `[4 × 2]` — lives entirely on **GPU2**
- **Input** `X` shape `[1 × 4]`

Full computation (no parallelism):
```
Y = X @ W1        # [1×4] @ [4×4] = [1×4]
Z = Y @ W2        # [1×4] @ [4×2] = [1×2]
```

### Step 1 — GPU1 holds the whole first layer

```
GPU1:  W1  [4×4]   (the entire matrix, not split)
```

GPU1 receives the input `X` and computes the **full** output `Y`:
```
GPU1:  Y = X @ W1   →  shape [1×4]   (complete result, no partial slices)
```

### Step 2 — Send `Y` over the network to GPU2

```
GPU1 ──── Y [1×4] ────►  GPU2
```
This is a **point-to-point transfer** of an activation tensor — not an `AllReduce`. GPU1 is now idle (or, with pipelining across multiple micro-batches, starts processing the *next* input).

### Step 3 — GPU2 holds the whole second layer

```
GPU2:  W2  [4×2]   (the entire matrix, not split)
```

GPU2 receives `Y` and computes the **full** final output:
```
GPU2:  Z = Y @ W2   →  shape [1×2]   ✓  (complete result, no summing needed)
```

### Why It's Different From TP

```
TP:  both GPUs compute simultaneously, each on a SLICE of the same layer's weights
     → needs AllReduce to combine partial results into one full output

PP:  each GPU computes the FULL result for its own layer, one after another
     → needs a data transfer (not a reduce) to hand the activation to the next stage
```

| | TP (colwise/rowwise example above) | PP (this example) |
|---|---|---|
| Who holds `W1`/`W2` | Both GPUs, each a slice | One GPU each, full matrix |
| Computation timing | Simultaneous | Sequential (assembly line) |
| Communication | `AllReduce` (sum partials) | Point-to-point send of activations |
| GPU utilization | Both busy at once, per layer | One idle while the other computes (unless micro-batched) |

This matches `base_model_pp_plan` in Gemma's config (see above): `embed_tokens` on GPU1, a chunk of `layers` on GPU2, the rest of `layers` + `norm` on GPU3 — each stage computes its full sub-network and ships the resulting hidden state to the next GPU.
