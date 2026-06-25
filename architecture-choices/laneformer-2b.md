# Kog Laneformer 2B: The Latency-First Model

**Source:** https://huggingface.co/blog/kogai/kog-laneformer-2b-the-latency-first-model

**Topics covered:**
- Delayed Tensor Parallelism (DTP)
- Hardware-aware architecture design
- Phased training with deliberate data specialization
- Latency as a first-class model constraint

---

## Core Philosophy

Most models are designed to maximize benchmark scores and then optimized for speed afterward. Laneformer inverts this:

```
Standard approach:
  Train for accuracy → optimize for speed later (often not possible without quality loss)

Laneformer's approach:
  Design the architecture around speed constraints → train from scratch → hit both goals
```

The key insight: **at low batch sizes, decode speed is not just a FLOPs problem.** Inter-GPU communication overhead becomes the bottleneck, and no amount of post-hoc optimization can remove it if the architecture wasn't designed to hide it.

---

## Key Innovation: Delayed Tensor Parallelism (DTP)

### The Problem It Solves

Tensor parallelism splits a model's weight matrices across multiple GPUs. After each layer, GPUs must **synchronize** — they exchange partial results via an all-reduce operation before the next layer can proceed.

```
Standard Tensor Parallelism:

  Layer N:   GPU_0 computes partial output
             GPU_1 computes partial output
                     ↓
             [ALL-REDUCE: sync across GPUs]   ← overhead paid every layer
                     ↓
  Layer N+1: GPUs continue...
```

At high batch sizes (many requests at once), GPU compute takes long enough that this sync overhead is hidden. But at **low batch sizes** (single requests, low-latency serving):

```
GPU compute time per layer:   ████░░░░░░░░░░░░   (short — small batch)
All-reduce sync overhead:              ████████   (fixed — doesn't shrink with batch)

→ Sync dominates total time
```

### How DTP Fixes It

DTP hides the sync cost by **delaying** it — the all-reduce for layer N is overlapped with the computation of layer N+1, using a 2-layer pipeline:

```
DTP Execution Timeline:

  Layer N   compute:  ████████
  Layer N   all-reduce:        ████████
  Layer N+1 compute:           ████████   ← starts while N's sync is in flight
  Layer N+1 all-reduce:                ████████
  Layer N+2 compute:                   ████████

Result: sync cost is hidden, not paid
```

This requires the architecture to have a specific **8-lane structure** where the communication delay is structurally built into how layers are organized — which is why DTP must be trained from scratch. You cannot graft this onto an existing model.

---

## Architecture Specifications

| Property | Value |
|---|---|
| Parameters | 2.3B |
| Layers | 15 total |
| Attention type | Grouped-Query Attention |
| Query heads | 32 |
| KV heads | 16 |
| Sliding-window layers | 10 of 15 |
| Context length | 4,096 tokens |
| Lane structure | 8 lanes (supports DTP with 2-layer delay) |
| Precision | BF16 |

### Grouped-Query Attention (GQA)

GQA reduces the KV cache size, which matters for inference speed and memory:

```
Standard Multi-Head Attention:
  32 query heads → 32 KV heads
  KV cache size: 32 × head_dim per token

GQA (Laneformer):
  32 query heads → 16 KV heads
  Each KV head is shared by 2 query heads
  KV cache size: 16 × head_dim per token   ← 2× smaller
```

### Sliding-Window Attention

10 of 15 layers use sliding-window attention — each token only attends to the last W tokens instead of the full sequence:

```
Full attention (5 of 15 layers):
  Token 100 attends to:  tokens 1–100    (long-range reasoning)

Sliding-window attention (10 of 15 layers):
  Token 100 attends to:  tokens (100-W)–100    (local context, cheaper)
```

This reduces memory and compute per forward pass at the cost of long-range attention coverage in those layers.

---

## Training Pipeline

Training ran across ~6 trillion tokens in three deliberate phases:

```
Phase 1: Pre-training (~4T tokens)
  ├── Broad, general data (web text, books, code, etc.)
  └── Goal: build a wide generalist base

Phase 2: Mid-training (~2T tokens)
  ├── Heavily shifted toward code and reasoning data
  └── Goal: specialise for coding benchmarks
        → +10 percentage points on coding evals
        → some degradation in general tasks (accepted trade-off)

Phase 3: Post-training (~210M tokens)
  ├── Instruction-following data
  └── Goal: instruction tuning for chat/code use
```

**Infrastructure:** 192 H100 GPUs, 24 nodes, ~21 days.

### Data Trade-off

The team made an explicit choice: concentrate specialization in phase 2 rather than blending coding data evenly across all phases.

```
Blended approach (common):
  All phases get ~30% code data
  → Moderate coding performance everywhere
  → No phase "owns" the specialization

Concentrated approach (Laneformer):
  Phase 2 gets ~70–80% code data
  → Phase 2 does the heavy lifting for coding skills
  → Phase 1 stays general (good base)
  → +10 pts on HumanEval+/MBPP+ vs blended
  → Costs some broad general capability
```

---

## Benchmark Performance

Evaluation used **greedy decoding** (temperature=0). Only relevant code blocks were extracted from outputs before scoring — no full-response pass/fail.

| Benchmark | Score |
|---|---|
| HumanEval+ | 45.1% |
| MBPP+ | 51.6% |

- Described as "extremely competitive for a 2B-class model."
- Pass@N sampling (N=2,4,8,16) shows consistent gains — useful as a practical accuracy-latency knob:

```
Pass@1:  one attempt, fastest
Pass@2:  two attempts, pick best  → higher accuracy, 2× cost
Pass@8:  eight attempts           → near ceiling accuracy, 8× cost
```

---

## Inference Speed

The headline result — single-request decode speed on standard datacenter GPUs:

| Hardware | Speed |
|---|---|
| AMD MI300X (8 GPUs) | **3,000 tokens/second** |
| NVIDIA H200 (8 GPUs) | **2,100 tokens/second** |

Claimed to be the fastest publicly demonstrated single-request decoding for a 2B-class model on standard datacenter hardware.

```
Why "single-request" matters:
  Throughput-optimized systems batch many requests together to amortize GPU idle time.
  But for latency-sensitive applications (code autocomplete, real-time chat),
  you care about how fast ONE request completes — and that's what Laneformer targets.
```

Full DTP benefits require Kog's custom inference engine. Standard Hugging Face `from_pretrained` loading works but runs without DTP acceleration.

---

## Limitations

- **Context length capped at 4,096 tokens** — long-context extension is in progress.
- **General capability trade-off** — deliberate data specialization toward code reduces broad question-answering quality.
- **DTP requires custom runtime** — the architecture's speed advantage is only fully realized with Kog's inference engine, not standard Transformers.

---

## What's Released

Available at `kogai/laneformer-2b-it` on Hugging Face:

- Instruction-tuned BF16 checkpoint
- Custom Hugging Face architecture implementation (needed for DTP structure)
- Full configuration and training recipe

**License:** Apache 2.0 for weights and custom code; Llama 2 Community License for tokenizer.

---

## Key Takeaway

Laneformer's contribution is less about a new attention mechanism and more about a **design methodology**: co-design the architecture and the hardware execution model together, before training begins. Speed cannot be fully optimized in post — the model's structure must make room for it.

```
The broader lesson:
  Latency is a model design constraint, not a serving concern.
  If you wait until after training to think about inference speed,
  you've already foreclosed the best options.
```

The project also demonstrates that focused teams with moderate budgets (~$200K compute) can build competitive small models by making deliberate, honest trade-offs rather than chasing balanced leaderboard numbers.
