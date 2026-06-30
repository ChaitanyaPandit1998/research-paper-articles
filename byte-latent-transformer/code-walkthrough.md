# BLT Codebase Walkthrough

**Repository:** https://github.com/facebookresearch/blt
**Local path:** `/Users/chaitanya/Development/AI/blt`

The codebase divides into **5 phases**: a one-time offline preprocessing phase, then four runtime phases that fire every training step or inference call.

---

## Hierarchical Diagram

```
BLT Codebase
│
├── PHASE 0 — Offline Preprocessing
│   ├── preprocess/preprocess_entropies.py   ← CLI: run once on raw corpus
│   ├── preprocess/parallel_entropies.py     ← Slurm-parallel version
│   ├── preprocess/data_pipeline.py          ← Pipeline orchestration
│   ├── entropy_model.py                     ← Loads the 100M oracle LM
│   └── [OUTPUT] Apache Arrow .arrow files   ← (sample_id, text, entropies[])
│
├── PHASE 1 — Data Loading & Batching
│   ├── data/data_types.py                   ← BltExample, BltSequence, Batch
│   ├── data/iterators/
│   │   ├── arrow_iterator.py                ← Reads Arrow files (tokens + entropies)
│   │   ├── sampling_iterator.py             ← Weighted multi-source sampling
│   │   ├── multiprocess_iterator.py         ← Background worker processes
│   │   ├── packing_iterator.py              ← Packs seqs into fixed-length batches
│   │   ├── preprocess_iterator.py           ← Tokenizes raw text on-the-fly
│   │   ├── looping_iterator.py              ← Epoch-looping wrapper
│   │   └── limit_iterator.py               ← Hard-stop wrapper
│   └── data/ngram_processor.py             ← Computes n-gram IDs from byte arrays
│
├── PHASE 2 — Patching
│   └── data/patcher.py
│       ├── PatcherArgs / Patcher            ← Config + main entry point
│       ├── calculate_entropies()            ← Oracle LM → per-byte entropies
│       ├── entropy()                        ← Shannon entropy of logit distribution
│       ├── find_entropy_patch_start_ids()   ← Threshold → patch boundary positions
│       ├── patch_start_mask_from_entropy_with_monotonicity()
│       ├── patch_start_mask_global_and_monotonicity()
│       ├── find_space_patch_start_ids()     ← Word-boundary patching
│       ├── find_bpe_delim_patch_start_ids() ← BPE-delimiter patching
│       ├── patch_lengths_from_start_ids()   ← start positions → lengths tensor
│       └── [OUTPUT] patch_lengths: [B, P]   ← how many bytes each patch contains
│
├── PHASE 3 — Forward Pass (BLT Model)
│   ├── model/blt.py — ByteLatentTransformer  ← Main orchestrator
│   │   ├── get_blt_input()                  ← Prepares encoder/decoder token seqs
│   │   ├── patch_ids_from_lengths()         ← patch_lengths → per-byte patch index
│   │   ├── cross_attn_mask()                ← FlexAttention block mask
│   │   ├── byte_group_hash_function()       ← Rolling polynomial hash on byte n-grams
│   │   └── compute_hash_embeddings()        ← Hash → embedding table lookup + sum
│   │
│   ├── [3a] model/local_models.py — LocalEncoder
│   │   ├── tok_embeddings (nn.Embedding)    ← 256-vocab byte embeddings
│   │   ├── rope (RotaryEmbedding)           ← RoPE positional encoding
│   │   ├── layers (TransformerBlock ×N)     ← Local causal self-attention
│   │   │   └── attn_bias = local_block_causal   ← no cross-doc attention
│   │   └── cross_attn_layers (CrossAttention)   ← bytes→patches pooling
│   │       └── apply_cross_attention()          ← optional pooling init
│   │
│   ├── [3b] model/utils.py — downsample()
│   │   └── aggregates byte reps → one rep per patch (mean/max/min pool)
│   │
│   ├── [3c] model/latent_transformer.py — GlobalTransformer
│   │   ├── token_embedding_projection       ← project patch dim → global dim
│   │   ├── layers (TransformerBlock ×N)     ← Full causal self-attention on patches
│   │   └── CrossAttention (shared class)    ← used by local encoder/decoder
│   │
│   └── [3d] model/local_models.py — LocalDecoder
│       ├── patch_embedding_projection       ← project global dim → decoder dim
│       ├── cross_attn_layers (CrossAttention)   ← patches→bytes injection
│       ├── layers (TransformerBlock ×N)     ← Local causal self-attention
│       ├── norm (RMSNorm)                   ← final normalisation
│       └── output (nn.Linear)              ← logits over 256 bytes
│
├── PHASE 4 — Training Loop
│   ├── train.py                             ← Main training entry point
│   │   ├── TrainState                       ← step, acc_step, data iterator state
│   │   ├── loss = cross_entropy(logits, targets)
│   │   ├── loss.backward()
│   │   └── optimizer.step()
│   ├── optim.py                             ← AdamW + LR scheduling
│   ├── distributed.py                       ← FSDP, tensor parallel, device mesh
│   ├── checkpoint.py                        ← Save/load distributed checkpoints
│   ├── norms.py                             ← Gradient clipping
│   └── metrics.py                           ← GPU memory monitor, metric logger
│
└── PHASE 5 — Inference / Generation
    ├── generate_blt.py                      ← BLT-specific generation entry point
    │   ├── generate_nocache()               ← Token-by-token autoregressive loop
    │   │   ├── patcher.patch(tokens[:curr_pos])  ← re-patch on every step
    │   │   └── model(tokens, patch_lengths)[:, -1]  ← last byte logit
    │   └── launch_generate()               ← Load model + run prompts
    ├── generate.py                          ← load_consolidated_model_and_tokenizer()
    └── eval.py                              ← Evaluation harness
```

---

## Phase 0 — Offline Preprocessing

**Files:** `preprocess/preprocess_entropies.py`, `entropy_model.py`

This runs **once** on the raw training corpus, before any model training begins.

### What it does

Raw text documents (JSONL) go in. Apache Arrow files (with per-byte entropy values baked in) come out.

```
JSONL docs  →  tokenize  →  entropy oracle LM  →  per-byte entropies  →  Arrow files
```

### Key files

**`entropy_model.py` — `load_entropy_model()`**

The entropy oracle is a **100M-parameter `LMTransformer`** (same architecture as the full model, but tiny). It has:
- 14 layers, 512 hidden dim
- A 512-token sliding window (local causal attention)
- Trained on the same corpus as the main model

It is loaded from a checkpoint and frozen (`requires_grad=False`). Its only job is to predict `P(next_byte | context)`.

**`preprocess/preprocess_entropies.py` — `main()`**

Reads JSONL line by line. For each document:
1. Tokenizes text → list of byte IDs (using `BltTokenizer`)
2. Calls `calculate_entropies(tokens, entropy_model)` → gets `[seq_len]` entropy floats
3. Buffers 1,000 records then flushes to Arrow (IPC format) with schema: `{sample_id, text, entropies[]}`

The Arrow output stores entropies as `float16` to save disk space. At training time, these are read back and used directly — no need to run the oracle again.

### Why offline?

Running the oracle during training would add latency per batch. Precomputing and storing means the training data loader reads entropy values the same way it reads tokens: from disk, already computed.

---

## Phase 1 — Data Loading & Batching

**Files:** `data/data_types.py`, `data/iterators/`

### Data model

`data/data_types.py` defines the data contract that flows through the pipeline:

```python
class BltExample:
    sample_id: str
    text: str
    tokens: list[int] | None          # byte IDs (0–255 + specials)
    entropies: list[float] | None     # per-byte entropy from oracle
    patch_lengths: list[int] | None   # computed at batch-build time
    mask: list[bool] | None           # which positions are real (vs padding)

class Batch:
    x: np.ndarray          # [B, S]  input byte sequences
    y: np.ndarray          # [B, S]  target byte sequences (x shifted by 1)
    patch_lengths: np.ndarray  # [B, P]  patch lengths per sequence
    ngram_ids: np.ndarray | None  # [num_ngrams, B, S] precomputed n-gram indices
    mask: np.ndarray | None    # [B, S] loss mask
```

### Iterator stack

The iterators compose like a pipeline. Each wraps the one below it:

```
MultiprocessIterator           ← spawns background worker processes for I/O
  └── PackingIterator          ← packs variable-length seqs into B×S batches
        └── SamplingIterator   ← weighted mix of data sources
              └── ArrowIterator / PreprocessIterator
                                ← reads Arrow files (tokens + entropies) OR
                                   tokenizes raw text on-the-fly
```

**`arrow_iterator.py`** — Reads the Arrow files produced in Phase 0. Each record has `(sample_id, text, entropies)`. It yields `BltExample` objects with entropies already loaded.

**`sampling_iterator.py`** — Takes multiple source iterators with weights (e.g., 70% web, 20% code, 10% books). Draws from each source according to the weight.

**`packing_iterator.py`** — Packs sequences into fixed-length `[B, S]` tensors with no wasted padding mid-sequence. Multiple short documents are concatenated into one row; EOS tokens mark document boundaries. Returns `Batch` objects.

**`multiprocess_iterator.py`** — Wraps any iterator and runs it in a background process, prefetching data so GPU never waits. Serializes iterator state for resuming after crashes.

**`ngram_processor.py`** — If byte n-gram embeddings are enabled, precomputes n-gram IDs (for n=3..8) from the byte array using sliding window + lookup table. Returns `ngram_ids: [num_ngrams, B, S]`.

---

## Phase 2 — Patching

**File:** `data/patcher.py`

The `Patcher` class takes a `[B, S]` batch of byte IDs and returns `patch_lengths: [B, P]` — a tensor describing how many bytes belong to each patch.

### Five patching modes

| Mode | Mechanism | Avg patch size |
|---|---|---|
| `entropy` | Oracle entropy threshold | 6–8 bytes |
| `space` | Split on whitespace/word-boundary bytes | 6+ bytes |
| `bpe` | Split on a BPE delimiter token | ~token size |
| `bpe_patcher` | A small model predicts BPE boundaries | ~token size |
| `static` | Fixed K bytes per patch | K bytes |

### Entropy patching in detail

This is the primary mode used in the paper. The key functions in `patcher.py`:

**`calculate_entropies(tokens, entropy_model, ...)`**
- Runs the 100M oracle LM over the byte sequence in chunks
- Returns `[B, S]` float tensor of per-byte entropies
- Either uses precomputed values (training) or runs the oracle live (inference)

**`entropy(scores)` — Shannon entropy**
```python
log_probs = F.log_softmax(scores, dim=-1)   # [B, S, 256]
probs = torch.exp(log_probs)
entropy = -(log_probs * probs).sum(dim=-1)  # [B, S]
```

**`find_entropy_patch_start_ids(entropies, threshold, ...)`**
- Two modes:
  - **Global threshold:** `patch_start_mask = entropies > threshold` — a new patch begins at any position where entropy exceeds the threshold (default: 1.335)
  - **Monotonicity:** `differences = entropies[:, 1:] - entropies[:, :-1]; mask = differences > threshold` — a new patch begins only when entropy *increases*, detecting the inflection point

**`patch_lengths_from_start_ids(patch_start_ids, seq_len)`**
- Converts start positions to lengths: e.g., starts at [0, 3, 7] with seq_len=10 → lengths [3, 4, 3]
- Output: `[B, P]` tensor, right-padded with zeros for batching

### Output format

```
tokens:         [h, e, l, l, o,  ,  w, o, r, l, d]
entropy:        [0.1, 0.2, 0.1, 0.1, 0.3, 2.1, 0.2, 0.1, 0.1, 0.3, 0.4]
patch_start:    [T,  F,   F,   F,   F,   T,   F,   F,   F,   F,   F  ]
patch_lengths:  [5, 6]    (bytes 0-4 are patch 0, bytes 5-10 are patch 1)
```

---

## Phase 3 — Forward Pass

**Files:** `model/blt.py`, `model/local_models.py`, `model/latent_transformer.py`, `base_transformer.py`

`ByteLatentTransformer.forward(tokens, patch_lengths, ngram_ids)` orchestrates 7 steps.

### Step 1 — Prepare input sequences (`get_blt_input`)

`blt.py:get_blt_input()` prepares three separate token sequences from the same input:

- **`local_encoder_tokens`** — the raw byte sequence, possibly prepended with BOE (beginning-of-encoder) tokens for the static patching case
- **`local_decoder_tokens`** — the same raw byte sequence (no BOE prefix in dynamic mode)
- The "global tokens" aren't a separate sequence; they're derived from patch structure later

The lag/alignment of these sequences is carefully managed so that the encoder sees bytes `[0..N]` when predicting byte `[N+1]`, avoiding label leakage.

### Step 2 — N-gram / hash embeddings (`compute_hash_embeddings`)

`blt.py:compute_hash_embeddings()` enriches each byte's embedding with subword context:

```python
# For each (hash_function, byte_group_size) pair:
hash_ids = byte_group_hash_function(tokens, group_size, hash_func_nb, max_hash)
#   → rolls a polynomial hash over windows of `group_size` bytes
#   → maps each window to an index in [0, 30000]

local_encoder_embeds += hash_tok_embedding[i](hash_ids)
```

Multiple hash functions (default: 3) over multiple group sizes (e.g., 3,4,5,6,7,8 bytes) are summed into the base byte embedding. This is the implementation of the paper's "n-gram hash embeddings" — giving the encoder knowledge of multi-byte patterns without a fixed vocabulary.

### Step 3 — Local Encoder (`LocalEncoder.forward`)

`local_models.py:LocalEncoder`

```
input:  local_encoder_tokens [B, N+nb_boe]  + optional hash embeds
output: (h_encoder [B, N, dim], h_cross [B, P, dim])
```

Internally:
1. `tok_embeddings(tokens)` — looks up 256-vocab byte embeddings
2. Adds hash embeddings (if provided)
3. Applies RoPE positional encoding
4. Runs `N_local_encoder_layers` of `TransformerBlock` with **local block-causal attention** — each token can only attend to tokens within the same document and within the sliding window (`local_attention_window_len`)
5. After the final (or every) layer, applies `CrossAttention` in **encoder mode**:
   - Queries = patch representations (initialised by pooling byte reps over the patch)
   - Keys/Values = byte representations from the transformer
   - This pools each patch's constituent bytes into a single patch vector

The local block-causal mask (`attn_bias_type="local_block_causal"`) prevents attention across document boundaries (EOS tokens act as hard stops).

### Step 4 — Downsample to Patch Representations

`model/utils.py:downsample()`

If cross-attention is disabled (simpler setup), byte representations are pooled into patch reps by simple aggregation (mean, max, or min over bytes in the patch).

If cross-attention is enabled, `h_cross` from Step 3 is already the patch representation — no additional pooling needed.

Output: `h: [B, P, dim_global]` — one vector per patch.

### Step 5 — Global Latent Transformer (`GlobalTransformer.forward`)

`latent_transformer.py:GlobalTransformer`

```
input:  h [B, P, dim_global]  — patch representations
output: h [B, P, dim_global]  — contextualised patch representations
```

This is the heavy, expensive component. It's a standard `BaseTransformer`:
- Full causal self-attention (patch P attends to patches 0..P-1)
- No cross-attention
- Runs on patch sequence of length P ≈ N/6 — much shorter than N

EOS positions are identified and injected into the `global_tokens` tensor so the global transformer knows document boundaries.

A `token_embedding_projection` linear layer (if `dim_token_emb != dim_global`) projects the patch vectors into the global transformer's hidden dimension before the first layer.

### Step 6 — Unpatch (scatter global reps back to bytes)

`blt.py:ByteLatentTransformer.forward` (lines 1011–1040)

Each byte needs to "know" which patch's global representation it belongs to, so it can condition its prediction on the global context.

Two modes:
- **Gather (default):** `h_gathered = h[batch, decoder_patch_ids]` — simply copies the patch vector to every byte in that patch. Fast, no parameters.
- **Cross-attention decoder:** patch vectors are kept as-is, and the decoder uses a `CrossAttention` layer to let each byte query its patch. More expressive, more compute.

Also extracted: `dec_embeds = h_encoder[:, nb_boe : nb_boe + N, :]` — the local encoder's byte-level representations, which are passed directly to the decoder as initial byte embeddings (bypass the global stage for local byte context).

### Step 7 — Local Decoder (`LocalDecoder.forward`)

`local_models.py:LocalDecoder`

```
input:  dec_embeds [B, N, dim_decoder]  ← byte reps from local encoder
        patch_embeds [B, N, dim_global] ← global patch reps (gathered or cross-attended)
output: logits [B, N, 256]              ← per-byte prediction over 256 possible bytes
```

Internally:
1. If cross-attention is disabled: `h = dec_embeds + patch_embeds` — add global context directly to byte reps
2. Runs `N_local_decoder_layers` of `TransformerBlock` with local block-causal attention
3. Before the first layer (or every layer if `cross_attn_all_layers_decoder`), applies `CrossAttention` in **decoder mode**:
   - Queries = byte representations
   - Keys/Values = patch representations from global transformer
4. Final `RMSNorm` + `nn.Linear(dim_decoder → 256)` → byte logits

The output logits are returned to the training loop or generation loop.

---

## Phase 4 — Training Loop

**File:** `train.py`, `optim.py`, `distributed.py`, `checkpoint.py`

### Training loop structure (`train.py`)

```python
for batch in data_iterator:
    x, y = batch.x, batch.y                    # [B, S] byte sequences
    patch_lengths = batch.patch_lengths         # [B, P] precomputed patches
    ngram_ids = batch.ngram_ids                 # [num_ngrams, B, S] optional

    logits = model(x, patch_lengths, ngram_ids) # [B, S, 256]

    loss = cross_entropy(logits, y, mask)       # scalar
    loss.backward()

    if step % grad_acc_steps == 0:
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

**Gradient accumulation** (`acc_step`) lets you simulate large batch sizes across multiple forward passes.

### Distributed training (`distributed.py`)

- **FSDP (Fully Sharded Data Parallel):** model parameters are sharded across GPUs. Each GPU holds a fraction of the weights; they're gathered for forward/backward and re-sharded after.
- **Tensor Parallel:** optionally splits individual matrix multiplications across GPUs.
- `get_device_mesh(distributed_args)` builds the 2D mesh: `[dp_shard, tp]`
- `parallelize_model(model, world_mesh)` wraps the model in FSDP/TP

### Optimizer (`optim.py`)

AdamW with:
- Separate learning rates for embedding layers vs. transformer layers
- Cosine LR schedule with warmup
- Weight decay applied only to non-bias, non-norm parameters

### Checkpointing (`checkpoint.py`)

- `CheckpointManager` saves: model weights, optimizer state, data iterator state (so training can resume from the exact batch)
- Uses PyTorch distributed checkpoint format (each rank saves its shard)
- Supports saving to S3 via `fsspec`

---

## Phase 5 — Inference / Generation

**Files:** `generate_blt.py`, `generate.py`

### `generate_nocache()` — autoregressive loop

```python
for curr_pos in range(start_pos, end_pos):
    current_tokens = tokens[:, :curr_pos]           # grow by 1 each step

    patch_lengths, _ = patcher.patch(
        current_tokens, include_next_token=True      # re-patch every step
    )

    logits = model(current_tokens, patch_lengths)   # full forward pass
    next_byte_logit = logits[:, -1]                 # only last position matters

    next_token = sample(next_byte_logit)            # greedy / top-k / top-p
    tokens[:, curr_pos] = next_token
```

**Key point:** there is **no KV cache** in this implementation. Every step runs the full forward pass on the growing sequence from scratch. This is expensive compared to token-based models with KV caching, and is a known limitation noted in the codebase (the variable patch boundaries make caching non-trivial).

The patcher runs in **realtime mode** at inference: it loads the 100M entropy oracle and calls it live at each step to compute patch boundaries for the current (growing) byte sequence.

### `load_consolidated_model_and_tokenizer()` (`generate.py`)

Loads the model from a checkpoint directory containing:
- `params.json` — the `ByteLatentTransformerArgs` config
- `consolidated.pth` — the full model state dict (merged from distributed shards)
- `tokenizer.model` — the BPE tokenizer (used as byte encoder, not for vocabulary)

---

## Supporting Infrastructure

| File | Role |
|---|---|
| `args.py` | `TrainArgs`, `EvalArgs` — top-level config objects |
| `config_parser.py` | YAML → Pydantic model parsing with inheritance and overrides |
| `base_transformer.py` | `BaseTransformerArgs`, `TransformerBlock`, `RotaryEmbedding`, `CrossAttention` — shared primitives |
| `transformer.py` | `LMTransformer` — the standard token-based LM (used for the entropy oracle and comparison runs) |
| `norms.py` | RMSNorm + gradient clipping utilities |
| `float8.py` | Float8 quantization support (experimental) |
| `tokenizers/blt_tokenizer.py` | `BltTokenizer` — maps raw bytes to IDs (0–255 + 4 special tokens: BOS, EOS, BOE, PAD) |
| `tokenizers/sentence_piece_tokenizer.py` | BPE tokenizer wrapper (used to pre-tokenize text before byte-encoding) |
| `stool.py` | Slurm job launcher |
| `probe.py` | Activation probing for debugging |
| `plotting/` | Entropy figure and scaling figure generators |

---

## Key Identifiers Cheat Sheet

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `N` or `S` | Sequence length in bytes |
| `P` | Number of patches (≈ N/6 for entropy patching) |
| `dim_local_encoder` | Hidden dim of LocalEncoder (e.g., 512) |
| `dim_global` | Hidden dim of GlobalTransformer (e.g., 4096) |
| `dim_local_decoder` | Hidden dim of LocalDecoder (e.g., 512) |
| `n_layers_local_encoder` | Layers in encoder (typically 1–6, cheap) |
| `n_layers_global` | Layers in global transformer (most of the model) |
| `n_layers_local_decoder` | Layers in decoder (typically 9, medium) |
| `patch_lengths` | `[B, P]` int tensor: how many bytes in each patch |
| `patch_ids` | `[B, N]` int tensor: which patch index each byte belongs to |
| `cross_attn_k` | Number of latent vectors per patch in cross-attention |
| `BOE_ID` | Beginning-of-encoder special token (used for static patch alignment) |
| `BOS_ID` | Beginning-of-sequence |
| `EOS_ID` | End-of-sequence / document boundary marker |
| `OFFSET` | 4 — number of special tokens, byte IDs start at OFFSET+0=4 |

---

## Addition A — Tensor Shape Flow Through Phase 3

Using a concrete small example to show how shapes transform at each sub-step.

**Setup:**
```
B = 2  (batch of 2 sequences)
N = 11 (bytes per sequence: "Hello world")
P = 2  (patches: ["Hello"=5 bytes, " world"=6 bytes])

dim_local_encoder = 512
dim_global        = 2048
dim_local_decoder = 512
vocab_size        = 260  (256 bytes + 4 special tokens)
```

**Shape trace:**

```
Step                         Tensor                    Shape
─────────────────────────────────────────────────────────────────────────
INPUT
  tokens                                               [2, 11]

STEP 1 — get_blt_input()
  local_encoder_tokens                                 [2, 11]
  local_decoder_tokens                                 [2, 11]
  (same as input in dynamic/entropy mode, no BOE prepend)

STEP 2 — compute_hash_embeddings()
  tok_embeddings(tokens)                               [2, 11, 512]
  + hash_embedding[0](hash_ids_grp3)                  [2, 11, 512]
  + hash_embedding[1](hash_ids_grp4)                  [2, 11, 512]
    ...  (3 functions × 6 group sizes = 18 tables, all summed)
  → local_encoder_embeds                               [2, 11, 512]

STEP 3 — LocalEncoder.forward()
  h = local_encoder_embeds                             [2, 11, 512]
  for each TransformerBlock (N_enc layers):
    h = SelfAttention(h, mask=local_block_causal)      [2, 11, 512]
    h = FFN(h)                                         [2, 11, 512]
  CrossAttention (encoder mode):
    patch_embeds (queries, init by pooling)            [2,  2, 512]   ← P=2 patches
    h (keys and values)                                [2, 11, 512]
    → h_cross                                          [2,  2, 512]

STEP 4 — downsample()
  (if pooling, not cross-attn)
    scatter_reduce(h, patch_ids, "mean")               [2,  2, 512]
  token_embedding_projection (if dim_enc != dim_global):
    Linear(512 → 2048)                                 [2,  2, 2048]
  → h (patch representations)                          [2,  2, 2048]

STEP 5 — GlobalTransformer.forward()
  h (input patch reps)                                 [2,  2, 2048]
  for each TransformerBlock (N_global layers):
    h = SelfAttention(h, mask=causal)                  [2,  2, 2048]
    h = FFN(h)                                         [2,  2, 2048]
  → h (contextualised patch reps)                      [2,  2, 2048]

STEP 6 — unpatch (gather)
  decoder_patch_ids  (which patch each byte belongs to) [2, 11]
  h_gathered = h[batch, decoder_patch_ids]              [2, 11, 2048]
  dec_embeds = h_encoder[:, :N, :]  (from Step 3)       [2, 11,  512]

STEP 7 — LocalDecoder.forward()
  h = dec_embeds                                        [2, 11,  512]
  patch_embeds = h_gathered                             [2, 11, 2048]
  patch_embedding_projection: Linear(2048 → 512)        [2, 11,  512]
  h = h + patch_embeds  (if no cross-attn)              [2, 11,  512]
  for each TransformerBlock (N_dec layers):
    h = SelfAttention(h, mask=local_block_causal)        [2, 11,  512]
    h = FFN(h)                                           [2, 11,  512]
  h = RMSNorm(h)                                         [2, 11,  512]
  logits = Linear(512 → 260)                             [2, 11,  260]

OUTPUT
  logits                                                 [2, 11,  260]
  ↓ cross_entropy with targets (shifted by 1)
  loss                                                   scalar
```

The global transformer receives a sequence of length **2** instead of **11** — that's where the FLOP savings come from. Everything expensive (QK^T attention, which is O(P²)) runs on P=2, not N=11.

---

## Addition B — The BOE / Lag Alignment (Hardest Part of the Code)

**File:** `model/blt.py:get_blt_input()` and the block comment starting at line 289

This is the most subtle alignment problem in the entire codebase. It's about ensuring the model cannot "see its own answer" when predicting the next byte.

### The problem

In a standard language model, training is causal: at position i, the model sees bytes `[0..i-1]` and predicts byte `i`. The target `y = x[1:]` — just shift by one.

In BLT, the global transformer operates on patches, not individual bytes. A patch covers multiple bytes. If patch P covers bytes `[4..7]`, the global transformer's representation of patch P was computed from those very bytes — including byte 7. But the decoder at position 7 needs to *predict* byte 8. Can it use the patch P representation? Yes — byte 8 is not in patch P. But what about position 4? The decoder at position 4 needs to predict byte 5. Patch P's representation was built from byte 5. **That would be label leakage.**

The fix is called the **lag correction**: the encoder's view of the sequence is shifted forward by one patch boundary, so a patch's global representation never contains any bytes it's supposed to help predict.

### The fix in dynamic (entropy) mode

```
Sequence:  [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]
Patches:   [P0=b0] [P1=b1,b2,b3,b4] [P2=b5,b6] [P3=b7,b8,b9,b10]

X_encoder: [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]

Global sees patches as:
  P0 rep = f(b0)            ← built from b0 only
  P1 rep = f(b0,b1,b2,b3,b4) ← built from b0..b4
  P2 rep = f(b0..b6)
  P3 rep = f(b0..b10)

X_decoder: [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]
decoder_patch_ids shift by one patch:
  b0 uses P0 rep   → P0 = f(b0)          → predicts b1  ✓ (P0 doesn't contain b1)
  b1 uses P0 rep   → P0 = f(b0)          → predicts b2  ✓ (P0 doesn't contain b2)
  ...
  b4 uses P0 rep   → P0 = f(b0)          → predicts b5  ✓
  b5 uses P1 rep   → P1 = f(b0..b4)      → predicts b6  ✓ (P1 doesn't contain b6)
  ...
```

The key line in `blt.py` is `decoder_patch_ids_from_lengths()` — it removes the *first* patch from the decoder patch assignments and shifts everything by one. Every byte in the decoder uses the *previous* patch's global representation, never its own patch's representation.

This is why the first patch is **forced to be a single byte** in dynamic mode: the first byte in the sequence has no "previous patch" to attend to, so it attends only to itself — a degenerate patch of size 1. The comment in `get_blt_input` refers to this as the *"force single byte first patch"* variant.

### In static mode: BOE tokens

With fixed patch size (e.g., 4 bytes per patch), the same alignment is achieved by **prepending BOE (Beginning-of-Encoder) tokens** to the encoder sequence:

```
patch_size = 4
tokens:   [b0, b1, b2, b3, b4, b5, b6, b7]
encoder:  [BOE, BOE, BOE, b0, b1, b2, b3, b4, b5, b6, b7]
                           ↑ 3 BOE tokens prepended (patch_size - 1)
patches:  [BOE,BOE,BOE,b0] [b1,b2,b3,b4] [b5,b6,b7,PAD]
Global P0 = f(BOE,BOE,BOE,b0)  → decoder uses P0 to predict b1,b2,b3,b4 ✓
Global P1 = f(BOE,BOE,BOE,b0,b1,b2,b3,b4) → decoder uses P1 to predict b5,b6,b7,b8 ✓
```

The BOE tokens act as "empty context" — they shift the encoder's view of the sequence left by exactly one patch, achieving the same lag correction without needing a dedicated first-patch rule.

---

## Addition C — The `local_block_causal` Attention Mask

**File:** `model/utils.py:create_causal_mask()` and `tokens_to_seqlen()`

During training, multiple documents are packed into one row to avoid wasted padding. A row might look like:

```
row: [doc1_byte0 ... doc1_byteK EOS doc2_byte0 ... doc2_byteM EOS doc3_byte0 ...]
```

Standard causal attention would let `doc2_byte0` attend to `doc1_byteK` — leaking information across documents. BLT uses `local_block_causal` to prevent this.

### How it works

**`tokens_to_seqlen(batch, eos_id)`** scans each row for EOS tokens and returns the length of each document:

```python
# Row: [h, e, l, l, o, EOS, w, o, r, l, d, EOS]
#                   ^3            ^9   ← EOS positions (0-indexed)
tokens_to_seqlen → [6, 6]         ← two docs of length 6 each
```

**`create_causal_mask()`** then builds an xformers `BlockDiagonalCausalMask` from those lengths and optionally adds a local sliding window on top:

```python
fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
    q_seqlen=tokens_to_seqlen(tokens, eos_id)
).make_local_attention(sliding_window)
```

The resulting mask looks like:

```
            doc1                  doc2
     h  e  l  l  o  EOS  |  w  o  r  l  d  EOS
h  [ 1  0  0  0  0   0       0  0  0  0  0   0 ]
e  [ 1  1  0  0  0   0       0  0  0  0  0   0 ]
l  [ 1  1  1  0  0   0       0  0  0  0  0   0 ]
l  [ 1  1  1  1  0   0       0  0  0  0  0   0 ]
o  [ 1  1  1  1  1   0       0  0  0  0  0   0 ]
EOS[ 1  1  1  1  1   1       0  0  0  0  0   0 ]
────────────────────────────────────────────────
w  [ 0  0  0  0  0   0       1  0  0  0  0   0 ]  ← hard stop at boundary
o  [ 0  0  0  0  0   0       1  1  0  0  0   0 ]
...
```

The EOS token creates a **hard barrier** — bytes in doc2 cannot attend to anything in doc1, regardless of document content.

The `sliding_window` constraint adds one more restriction: even within a document, a byte can only attend to the last `W` bytes (default 512). This keeps the local encoder/decoder cheap even for very long documents.

The **global transformer** does NOT use `local_block_causal` — it uses plain causal masking over patches. At the patch level, documents are still separated because EOS byte positions propagate into the `global_tokens` tensor (line 1001-1003 of `blt.py`), but cross-document patch attention IS allowed at the global level.

---

## Addition D — The `cross_attn_k` Parameter (Multiple Latent Slots per Patch)

**Files:** `model/blt.py:create_local_encoder()`, `model/local_models.py:LocalEncoder.apply_cross_attention()`

In the basic encoder cross-attention, each patch gets **one** query vector that reads from all the bytes in that patch. `cross_attn_k` multiplies this: each patch gets `k` query vectors instead of 1.

### Why this helps

One vector per patch must compress everything a patch "means" into a single point in `dim_local_encoder`-dimensional space. For a patch covering 8 bytes, that's a lot of compression. Multiple latent slots let different aspects of the patch (phonetic structure, semantic content, positional role) be captured independently.

This is the same idea as **Perceiver IO's latent array** — a fixed-size bank of query vectors that reads from a long input sequence via cross-attention.

### How it's implemented

```python
# In apply_cross_attention():
# patch_embeds has shape [B, P, dim] initially (from pooling)

# project to k slots per patch:
patch_embeds = self.patch_embedding_projection(patch_embeds)
# Linear(dim → dim * k) → [B, P, dim*k]

patch_embeds = patch_embeds.reshape(bs, P * cross_attn_k, dim)
# → [B, P*k, dim]  each of the P patches now has k query slots

# CrossAttention: queries = patch slots, keys/values = byte reps
patch_embeds = cross_attn_layer(x=patch_embeds, kv=h)
# → [B, P*k, dim]

# The cross_attn_mask repeats each patch's byte assignments k times:
cross_mask = create_patch_mask_from_ids(...).repeat_interleave(cross_attn_k, dim=1)
# Each of the k slots for patch P attends to the same bytes as patch P
```

When these `P*k` vectors reach the global transformer, they're reshaped back to `[B, P, dim*k]` and projected to `[B, P, dim_global]`. So the global transformer still sees P patch positions — the k slots are merged into a richer per-patch representation.

The default in the paper's reported results is `cross_attn_k = 1` (one slot per patch). Increasing k trades compute for expressiveness — each patch costs k times as much in the cross-attention but the global transformer is unchanged.

---

## Addition E — Why There Is No KV Cache at Inference

**File:** `generate_blt.py:generate_nocache()`

The function is literally named `generate_nocache`. Here's why caching doesn't work for BLT, and why it's a hard structural problem, not a missing feature.

### How KV cache works in standard models

At step t, you generate token `t`. The K and V matrices for all previous positions `[0..t-1]` are cached from the previous step. You only compute K, V for the new token at position `t`, then attend over `[cache_0..cache_{t-1}, kv_t]`. Cost per step: O(t) instead of O(t²).

This works because token positions are **stable** — token at position 3 is always token at position 3, regardless of how long the sequence gets.

### Why BLT breaks this

At step t, the byte sequence `[b_0, ..., b_t]` is re-patched from scratch:

```
Step t:    bytes [H, e, l, l, o,  , w, o] → patches [Hello][' 'wor][l][d]... wait, 'd' not yet
Step t+1:  bytes [H, e, l, l, o,  , w, o, r]
```

Adding byte `r` might:
- Extend the last patch (if `r` is low entropy): patch structure `[Hello][' 'wor]` becomes `[Hello][' 'work]`... no, actually `[' 'wo][r]` if 'r' was high entropy at the previous step

The critical issue: **a byte's patch assignment can change as new bytes are appended**. At step t, byte 5 (`' '`) might be the start of patch 1. At step t+3, byte 5 is still in patch 1 but patch 1 is now longer. The K and V tensors cached for step t's patch 1 were computed over a different sequence than step t+3's patch 1.

In formulas: at step t, patch 1's key = `f([' ',w,o])`. At step t+1, patch 1's key = `f([' ',w,o,r])`. These are different vectors. The cache is stale.

### The cost

Every inference step runs the **full forward pass** from scratch:
- LocalEncoder over all N bytes: O(N × W) where W = sliding window
- GlobalTransformer over P patches: O(P²) — this is the cheap part
- LocalDecoder over all N bytes: O(N × W)

The dominant cost at inference is actually the local encoder/decoder (O(N)), not the global transformer (O(P²) ≈ O(N²/36)). This is worse than token-based models with KV caching, which achieve O(t) per step.

### Possible future fix

A patch-aware KV cache would need to:
1. Detect when a new byte changes the patch boundary of an old byte
2. Invalidate only the affected patch's cached K/V
3. Recompute only that patch

This is feasible for the global transformer (P is small) but harder for the local encoder/decoder (their sliding window crosses patch boundaries). It's an open engineering challenge that the paper acknowledges.

---

## Addition F — Worked Example: "Hello world" Through All Phases

Let's trace the string **`"Hello world"`** through the entire codebase, from raw text to output logits.

---

### Phase 0 output (already computed offline)

The entropy oracle (100M-param `LMTransformer`) ran over the training corpus and produced this for a document containing "Hello world":

```json
{
  "sample_id": "doc_0042",
  "text": "Hello world",
  "entropies": [2.1, 1.2, 0.4, 0.3, 0.2, 0.5, 1.9, 1.1, 0.6, 0.3, 0.4]
}
```

Entropies are per-byte: H=2.1, e=1.2, l=0.4, l=0.3, o=0.2, ' '=0.5, w=1.9, o=1.1, r=0.6, l=0.3, d=0.4

Stored in an Apache Arrow file, schema: `{sample_id: str, text: str, entropies: list<float16>}`.

---

### Phase 1 output (data loading)

The `ArrowIterator` reads the record. The `PackingIterator` tokenizes and packs it into a `Batch`:

**Tokenization via `BltTokenizer.encode()`:**
```
"Hello world"
→ bytes: H(72)  e(101)  l(108)  l(108)  o(111)  ' '(32)  w(119)  o(111)  r(114)  l(108)  d(100)
→ + OFFSET=4:  76     105     112     112     115      36     123     115     118     112     104
→ + BOS=1 prepended, EOS=2 appended
→ final ids: [1, 76, 105, 112, 112, 115, 36, 123, 115, 118, 112, 104, 2]
              BOS H    e    l    l    o   ' '  w    o    r    l    d   EOS
```

`Batch` object (single-sequence batch, B=1, S=11, ignoring BOS/EOS for model input):
```python
Batch(
    x            = [[76, 105, 112, 112, 115, 36, 123, 115, 118, 112, 104]],  # [1, 11]
    y            = [[105, 112, 112, 115, 36, 123, 115, 118, 112, 104, 2]],   # [1, 11] (shifted)
    patch_lengths = None,   ← will be filled in Phase 2
    entropies    = [[2.1, 1.2, 0.4, 0.3, 0.2, 0.5, 1.9, 1.1, 0.6, 0.3, 0.4]]
)
```

---

### Phase 2 output (patching)

**`Patcher.patch(tokens, entropies=precomputed, threshold=1.335)`**

Step-by-step entropy comparison (threshold = 1.335):
```
Position:  0    1    2    3    4    5    6    7    8    9    10
Byte:      H    e    l    l    o    ' '  w    o    r    l    d
Entropy:   2.1  1.2  0.4  0.3  0.2  0.5  1.9  1.1  0.6  0.3  0.4
> thresh?  YES  NO   NO   NO   NO   NO   YES  NO   NO   NO   NO
```

Positions 0 and 1 are always forced as patch starts (first_ids = [0, 1]).
Position 6 (byte 'w') also triggers a boundary because entropy at position 6 = 1.9 > 1.335.

```
patch_start_mask:  [T, T, F, F, F, F, T, F, F, F, F]
patch_start_ids:   [0, 1, 6]
patch_lengths:     [1, 5, 5]    ← patch 0: 'H' (1 byte)
                                   patch 1: 'ello ' (5 bytes)
                                   patch 2: 'world' (5 bytes)
```

Output: `patch_lengths = torch.tensor([[1, 5, 5]])` shape `[1, 3]` — B=1, P=3 patches.

---

### Phase 3 output (forward pass)

Using small model dimensions: `dim_local_encoder=512, dim_global=1024, dim_local_decoder=512`

**Step 1 — `get_blt_input()`**
```
local_encoder_tokens = [[76, 105, 112, 112, 115, 36, 123, 115, 118, 112, 104]]  [1, 11]
local_decoder_tokens = [[76, 105, 112, 112, 115, 36, 123, 115, 118, 112, 104]]  [1, 11]
```
Dynamic mode: no BOE prepend. First patch forced to 1 byte.

**`patch_ids_from_lengths([1, 5, 5], seq_len=11)`**
```
patch_lengths: [1, 5, 5]
cumsum:        [0, 1, 6, 11]

byte position:  0  1  2  3  4  5  6  7  8  9  10
patch_ids:     [0, 1, 1, 1, 1, 1, 2, 2, 2, 2,  2]
               (H)(e  l  l  l  o)(w  o  r  l   d)
                ^                 ^
               P0               P2
```

**Step 2 — Hash embeddings**
```
tokens [1, 11] → tok_embeddings → [1, 11, 512]
+ rolling hash of 3-byte windows → embedding table → [1, 11, 512]
+ rolling hash of 4-byte windows → embedding table → [1, 11, 512]
  ... (18 hash tables total, summed in-place)
→ local_encoder_embeds [1, 11, 512]
```

**Step 3 — LocalEncoder**
```
h = local_encoder_embeds                               [1, 11, 512]
  ↓ N layers of local-block-causal self-attention
h = encoder hidden states                              [1, 11, 512]
  ↓ cross-attention: patches query bytes
patch query init (mean pool over each patch):
  P0 = mean(h[:, 0:1, :])                             [1,  1, 512]
  P1 = mean(h[:, 1:6, :])                             [1,  1, 512]
  P2 = mean(h[:, 6:11,:])                             [1,  1, 512]
  → patch_queries                                      [1,  3, 512]
cross-attention output:
  queries = patch_queries [1, 3, 512]
  keys/values = h         [1, 11, 512]
  mask: P0 attends to bytes[0], P1 to bytes[1-5], P2 to bytes[6-10]
  → h_cross                                           [1,  3, 512]  ← P=3 patch reps
```

**Step 4 — Downsample + project**
```
h_cross                                               [1,  3,  512]
token_embedding_projection Linear(512 → 1024):        [1,  3, 1024]
```

**Step 5 — GlobalTransformer**
```
h (patch reps)                                        [1,  3, 1024]
  ↓ N_global layers of full causal self-attention
  (P0 → P1 → P2, each attending to all previous patches)
h (contextualised patch reps)                         [1,  3, 1024]

Patch meaning after global attention:
  P0: "sequence starts with a capitalised word"
  P1: "it's 'ello ' — likely part of 'Hello'" (attends to P0)
  P2: "it's 'world' — completing 'Hello world'" (attends to P0, P1)
```

**Step 6 — Unpatch (gather)**
```
decoder_patch_ids (shifted by 1 patch from encoder):
  byte:  0   1   2   3   4   5   6   7   8   9  10
  pid:  [0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2]
         ← uses P0's rep →  ← uses P1's rep →   ← P2 →

h_gathered = h[0, decoder_patch_ids[0], :]           [1, 11, 1024]
  byte 0 (H) gets P0 = f("H")                        ← P0 knows nothing after H
  byte 5 (' ') gets P1 = f("H","ello ")              ← P1 knows "Hello"
  byte 6 (w) gets P1 too!                            ← 'w' uses "Hello" context
  byte 10 (d) gets P2 = f("H","ello ","world")        ← P2 has full context

dec_embeds = h_encoder[:, :11, :]                    [1, 11,  512]  (local byte context)
```

**Step 7 — LocalDecoder**
```
h = dec_embeds                                        [1, 11,  512]
patch_embedding_projection Linear(1024 → 512)
patch_embeds                                          [1, 11,  512]
h = h + patch_embeds                                  [1, 11,  512]
  ↓ N_dec layers of local-block-causal self-attention
h = RMSNorm(h)                                        [1, 11,  512]
logits = Linear(512 → 260)                            [1, 11,  260]
```

**Training loss:**
```
logits [1, 11, 260]    (model's predictions)
y      [1, 11]         (targets: [105, 112, 112, 115, 36, 123, 115, 118, 112, 104, 2])
                                    e    l    l    o   ' '  w    o    r    l    d   EOS

loss = cross_entropy(logits.view(11, 260), y.view(11))  ← scalar
```

At position 0 (byte 'H'), the model must predict byte 'e' (ID 105). At position 5 (byte ' '), it must predict 'w' (ID 123). At position 10 (byte 'd'), it must predict EOS (ID 2).

---

### Phase 5 — One inference step

At generation step 6 (we've generated "Hello " and need to predict the 7th byte):

```python
current_tokens = [[76, 105, 112, 112, 115, 36]]  # "Hello " so far, [1, 6]

# Re-patch the 6-byte sequence
patch_lengths, _ = patcher.patch(current_tokens, include_next_token=True)
# oracle runs on tokens, computes entropies → boundaries → [[1, 5, 1]]
# last patch length=1 accounts for include_next_token

logits = model(current_tokens, patch_lengths=patch_lengths)  # [1, 6, 260]
next_byte_logit = logits[0, -1, :]                           # [260]

# Top candidates (after softmax):
# ID 123 (w) → 0.31  ← "Hello w..." very likely
# ID 116 (t) → 0.12  ← "Hello t..."
# ID 121 (y) → 0.08  ← "Hello y..."

next_token = argmax(next_byte_logit)  # → 123 = 'w'
tokens[:, 6] = 123                    # extend sequence
```

The oracle runs again at step 7, this time on 7 bytes `"Hello w"`, re-patches from scratch, and the cycle repeats until EOS.

---

## How the Five Phases Connect

```
Phase 0                Phase 1                    Phase 2         Phase 3
─────────────────      ──────────────────────     ──────────────  ──────────────────────────────
Raw JSONL corpus  →    Arrow Iterator        →    Patcher    →    ByteLatentTransformer.forward()
  + entropy oracle       (reads tokens +          (patch_          ├─ LocalEncoder
  → Arrow files          precomputed entropies)    lengths)        ├─ GlobalTransformer
                         → Sampling                               └─ LocalDecoder
                         → Packing                                      │
                         → Batch {x, y,                                 ↓ logits [B, N, 256]
                             patch_lengths,                        Phase 4
                             ngram_ids}                       ─────────────────────
                                                               cross_entropy(logits, y)
                                                               → loss.backward()
                                                               → optimizer.step()
                                                                         │
                                                               Phase 5 (inference only)
                                                             ─────────────────────────
                                                              generate_nocache()
                                                              → re-patch each step
                                                              → sample next byte
```
