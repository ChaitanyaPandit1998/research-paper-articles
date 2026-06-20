# LLM Training Cost Estimation — Notes

## Worked example: custom 8B model, 500B tokens, 16× H100 SXM cluster

### Assumptions

| Parameter | Value |
|---|---|
| Model params | 8B |
| Training tokens | 500B (~3× Chinchilla-optimal of 160B — modest overtraining, not extreme like Llama/Qwen) |
| Hardware | 16× H100 SXM |
| Cluster cost | $63.17/hr (≈ $3.95/hr per GPU) |
| H100 SXM peak BF16 (dense, no sparsity) | 989 TFLOPS/GPU |

### Step 1 — Total compute required

| Quantity | Value |
|---|---|
| Formula | FLOPs ≈ 6 × params × tokens |
| Calculation | 6 × 8×10⁹ × 500×10⁹ |
| **Total FLOPs** | **2.4 × 10²²** |

### Step 2 — Cluster theoretical peak throughput

| Quantity | Value |
|---|---|
| Per-GPU peak | 989 TFLOPS |
| Cluster size | 16 GPUs |
| **Cluster peak throughput** | **15,824 TFLOPS = 1.58 × 10¹⁶ FLOP/s** |

### Step 3 — Apply realistic MFU (Model FLOPs Utilization)

No real run hits 100% of peak (communication overhead, non-matmul ops, data loading, checkpointing). Good dense 8B training typically lands 30–50% MFU; less-optimized setups can be ~20%.

| MFU | Achieved throughput | Time to train | Wall-clock | Cost @ $63.17/hr |
|---|---|---|---|---|
| 20% (conservative) | 3.16 × 10¹⁵ FLOP/s | 2,107 hours | ~88 days | ~$133,000 |
| 35% (realistic/good) | 5.54 × 10¹⁵ FLOP/s | 1,204 hours | ~50 days | ~$76,000 |
| 50% (well-optimized) | 7.91 × 10¹⁵ FLOP/s | 843 hours | ~35 days | ~$53,000 |

### Headline estimate

| | |
|---|---|
| Cost range | ~$50,000 – $135,000 |
| Realistic midpoint (35% MFU) | ~$76,000 |
| Wall-clock time | ~5–12 weeks on a 16-GPU cluster (cluster size is the bottleneck, not the dollar cost) |

### What this estimate excludes

| Excluded item | Why it matters |
|---|---|
| Failed/restarted runs, hyperparameter search, ablations | Adds extra GPU-hours beyond the "clean" run modeled here |
| Data curation/cleaning pipeline cost | Separate cost center, not part of GPU compute |
| Storage, networking, orchestration overhead | Beyond raw GPU-hours billed |
| Evaluation, post-training (SFT/RLHF), salaries | Often comparable to or exceeding pretraining compute cost |

So ~$76K is just the pretraining compute; real end-to-end project cost would run higher.

---

## Reference formulas

```
Training FLOPs ≈ 6 × N × D          (N = params, D = training tokens)

Time (seconds) = Total FLOPs / (achieved FLOP/s)
achieved FLOP/s = num_GPUs × peak_FLOPs_per_GPU × MFU

Cost = Time (hours) × cluster $/hour
```

## See also
- [[modular_gemma_notes]] / `gemma/NOTES.md` — multi-GPU parallelism (TP/PP) mechanics, relevant to how a cluster like the one above would actually be split across GPUs for an 8B run.
