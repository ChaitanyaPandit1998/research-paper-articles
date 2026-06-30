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
