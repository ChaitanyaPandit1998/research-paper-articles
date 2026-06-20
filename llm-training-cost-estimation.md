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

---

## Inference serving cost: custom 8B model, 1,000 users

Training is a one-time cost; serving the model afterward is a recurring cost — this section sizes that recurring cost separately.

### Usage assumptions

| Parameter | Value |
|---|---|
| Active users | 1,000 |
| Prompts/user/day | 30 |
| Input tokens/request | 60 |
| Output tokens/request | 2,500 |
| Avg tokens/request | 2,560 |

### Token volume

| | Per day | Per month (×30) |
|---|---|---|
| Requests | 30,000 | 900,000 |
| Input tokens | 1,800,000 | 54,000,000 |
| Output tokens | 75,000,000 | 2,250,000,000 |
| **Total tokens** | **76,800,000** | **~2.3 billion** |

Output tokens (autoregressive decode — the expensive part of inference) make up ~97.7% of total volume.

### GPU-hours needed per day (decode-bound)

| GPU | Sustained decode throughput | GPU-hours/day needed | % of a 24h day |
|---|---|---|---|
| RTX 4090 | ~2,000 tok/s | ~10.4 hours | 43% |
| A100 | ~3,000 tok/s | ~6.9 hours | 29% |
| H100 SXM | ~5,000 tok/s | ~4.2 hours | 17% |

A single RTX 4090 is tight if traffic concentrates into business hours rather than spreading across 24h — A100/H100 have more headroom.

### Dedicated, always-on GPU cost — RunPod Secure Cloud pricing (verified against runpod.io/pricing)

RunPod has two tiers: **Community Cloud** (cheaper, instances can be preempted/reclaimed at any time — not suitable for live production traffic) and **Secure Cloud** (reliable, enterprise-grade — used below).

| GPU | Community Cloud $/hr | Secure Cloud $/hr (used for estimate) | Monthly (24/7) | Annual | Headroom |
|---|---|---|---|---|---|
| RTX 4090 (×1) | $0.34 | $0.69 | ~$497 | ~$5,964 | Tight — consider 2× for safety margin |
| RTX 4090 (×2, recommended) | — | $0.69 | ~$994 | ~$11,927 | Comfortable |
| A100 80GB (×1) | $0.89 | $1.49 | ~$1,073 | ~$12,876 | Comfortable |
| H100 SXM (×1) | — | $2.69–$3.29 | ~$1,937–$2,369 | ~$23,245–$28,427 | Very comfortable, room to grow |

Note: Spheron's H100 SXM on-demand rate (~$2.50/hr) is cheaper than RunPod's for this GPU — worth comparing providers per-GPU rather than assuming one provider is cheapest across the board.

### Pay-per-token API alternative

```
2,304M tokens/month × ~$1.74/M tokens ≈ $4,009/month ≈ $48,100/year
```

### Comparison

| Option | Annual cost |
|---|---|
| Pay-per-token API | ~$48,100 |
| Dedicated H100 SXM (24/7, RunPod) | ~$23,200–$28,400 |
| Dedicated A100 (24/7, RunPod) | ~$12,876 |
| Dedicated 2× RTX 4090 (24/7, RunPod) | ~$11,927 |

**Takeaway:** with long outputs (2,500 tokens/response), API billing scales linearly with volume and becomes expensive fast — self-hosting wins by 2–4×. A100 on RunPod Secure Cloud is the best value of the options checked, beating both H100 (overkill, pricier here) and a single RTX 4090 (tight on headroom).

### Why this matters relative to training cost

Training (~$76K) is a one-time cost; inference (~$12K–$48K/year depending on hosting choice) is recurring. Over a multi-year deployment, or as user count grows, cumulative inference cost can equal or exceed the original training spend — this is the same dynamic that justifies "overtraining" a smaller model in the first place (see Chinchilla discussion): paying more upfront in training to keep the permanent per-query cost low.

## See also
- [[modular_gemma_notes]] / `gemma/NOTES.md` — multi-GPU parallelism (TP/PP) mechanics, relevant to how a cluster like the one above would actually be split across GPUs for an 8B run.
