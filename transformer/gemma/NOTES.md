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
"embed_tokens": (["input_ids"], ["inputs_embeds"]),    # GPU 1
"layers":       (["hidden_states", ...], [...]),        # GPU 2
"norm":         (["hidden_states"], ["hidden_states"])  # GPU 3
```
The tuples define **what tensor comes in** and **what tensor goes out** at each stage, so the framework knows what to send over the network between GPUs.

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

## How Colwise Output Feeds into Rowwise Input

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
