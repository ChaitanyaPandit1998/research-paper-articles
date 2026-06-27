# LLM Cost Estimation — 500M Param Model (50B tokens)

Same hardware and inference-workload assumptions as [[8b-param-cost-estimation]] and [[4b-param-cost-estimation]], but with **1/16th the parameters (500M vs 8B) and 1/10th the training tokens (50B vs 500B)** — total training compute is 1/160th of the 8B scenario.

---

# Part 1 — Training Cost

## Worked example: custom 500M model, 50B tokens, 16× H100 SXM cluster

### Assumptions

| Parameter | Value |
|---|---|
| Model params | 500M |
| Training tokens | 50B (~5× Chinchilla-optimal of 10B — moderate overtraining to improve inference quality) |
| Hardware | 16× H100 SXM |
| Cluster cost | $63.17/hr (≈ $3.95/hr per GPU) |
| H100 SXM peak BF16 (dense, no sparsity) | 989 TFLOPS/GPU |

### Step 1 — Total compute required

| Quantity | Value |
|---|---|
| Formula | FLOPs ≈ 6 × params × tokens |
| Calculation | 6 × 5×10⁸ × 5×10¹⁰ |
| **Total FLOPs** | **1.5 × 10²⁰ (1/160th of the 8B model's 2.4 × 10²²)** |

### Step 2 — Cluster theoretical peak throughput

Unchanged from the 8B and 4B scenarios — hardware doesn't change with model size.

| Quantity | Value |
|---|---|
| Per-GPU peak | 989 TFLOPS |
| Cluster size | 16 GPUs |
| **Cluster peak throughput** | **15,824 TFLOPS = 1.58 × 10¹⁶ FLOP/s** |

### Step 3 — Apply realistic MFU (Model FLOPs Utilization)

| MFU | Achieved throughput | Time to train | Wall-clock | Cost @ $63.17/hr |
|---|---|---|---|---|
| 20% (conservative) | 3.16 × 10¹⁵ FLOP/s | ~13.2 hours | **< 1 day** | ~$832 |
| 35% (realistic/good) | 5.54 × 10¹⁵ FLOP/s | ~7.5 hours | **< 1 day** | ~$475 |
| 50% (well-optimized) | 7.91 × 10¹⁵ FLOP/s | ~5.3 hours | **< 1 day** | ~$333 |

At this scale, even the conservative estimate completes in a single working day. The 16-GPU cluster is dramatically over-provisioned — a single H100 would finish the job in under 2 weeks at 35% MFU, making the full cluster wasteful unless you're running many concurrent experiments.

### Headline estimate

| | |
|---|---|
| Cost range | ~$333 – $832 |
| Realistic midpoint (35% MFU) | ~$475 |
| Wall-clock time | ~5–14 hours on a 16-GPU cluster |

Training cost at this scale is effectively a rounding error relative to engineering time and data curation. Iteration speed — not compute budget — becomes the main constraint.

### Reference formulas

```
Training FLOPs ≈ 6 × N × D          (N = params, D = training tokens)

Time (seconds) = Total FLOPs / (achieved FLOP/s)
achieved FLOP/s = num_GPUs × peak_FLOPs_per_GPU × MFU

Cost = Time (hours) × cluster $/hour
```

---

# Part 2 — Inference Cost

## Worked example: custom 500M model, 1,000 users

Usage assumptions are identical to the 8B and 4B scenarios. What changes again is **GPU throughput** — a 500M model does ~1/16th the FLOPs per token of the 8B model, so each GPU can push through tokens roughly **16× faster**.

An additional factor unique to this model size: a 500M model in BF16 weighs only **~1 GB** of VRAM. A single RTX 4090 (24 GB) could host ~20 replicas simultaneously, meaning multi-tenant hosting is viable even on consumer hardware.

### Usage assumptions (unchanged)

| Parameter | Value |
|---|---|
| Active users | 1,000 |
| Prompts/user/day | 30 |
| Input tokens/request | 60 |
| Output tokens/request | 2,500 |
| Avg tokens/request | 2,560 |

### Token volume (unchanged — workload is model-size-independent)

| | Per day | Per month (×30) |
|---|---|---|
| Requests | 30,000 | 900,000 |
| Input tokens | 1,800,000 | 54,000,000 |
| Output tokens | 75,000,000 | 2,250,000,000 |
| **Total tokens** | **76,800,000** | **~2.3 billion** |

### GPU-hours needed per day (decode-bound, ~16× throughput vs. 8B model)

| GPU | Sustained decode throughput (500M, ~16×) | GPU-hours/day needed | % of a 24h day |
|---|---|---|---|
| RTX 4090 | ~32,000 tok/s | ~0.67 hours | 2.8% |
| A100 SXM | ~48,000 tok/s | ~0.44 hours | 1.9% |
| H100 SXM | ~80,000 tok/s | ~0.27 hours | 1.1% |

Note: throughput scaling becomes less linear below ~1B params due to CUDA kernel and framework scheduling overhead at small model sizes. These figures reflect the theoretical ~16× ceiling; real-world numbers may land 20–30% below for single-replica serving, though batching multiple replicas on one GPU (feasible at 1 GB model size) largely recovers this.

### How many GPUs are actually needed (peak load, not daily average)

| Traffic window | Peak tok/sec needed | RTX 4090 (32,000 tok/s) | A100 (48,000 tok/s) | H100 (80,000 tok/s) |
|---|---|---|---|---|
| Spread over 24h | 868 | 1 GPU (2.7% util) | 1 GPU (1.8% util) | 1 GPU (1.1% util) |
| 12h business day | 1,736 | 1 GPU (5.4% util) | 1 GPU (3.6% util) | 1 GPU (2.2% util) |
| 8h concentrated window | 2,604 | 1 GPU (8.1% util) | 1 GPU (5.4% util) | 1 GPU (3.3% util) |

Even in the worst-case concentrated traffic scenario, a single GPU of any tier runs at under 10% utilization. A **single GPU is sufficient** for throughput at this scale; a second GPU is warranted only for availability/redundancy, not capacity.

### Dedicated, always-on GPU cost — RunPod on-demand pricing

GPU rental price is model-size-independent — these figures are **identical to the 8B and 4B scenarios**; only utilization differs.

| GPU | $/hr (RunPod, verified) | Monthly (730 hrs) | Annual (8,760 hrs) |
|---|---|---|---|
| RTX 4090 24GB (×1) | $0.69 | ~$504 | ~$6,044 |
| A100 SXM 80GB (×1) | $1.49 | ~$1,088 | ~$13,052 |
| H100 SXM 80GB (×1) | $3.29 | ~$2,402 | ~$28,820 |

At 1–3% GPU utilization, 24/7 dedicated rental wastes 97–99% of billed capacity. It is the worst possible billing model for a workload this light.

### Pay-per-use model — billed only for GPU-hours actually consumed

| GPU | $/hr | GPU-hours/day | GPU-hours/year (×365) | Monthly (usage-based) | Annual (usage-based) |
|---|---|---|---|---|---|
| RTX 4090 | $0.69 | 0.67 | 244 | ~$14 | ~$168 |
| A100 SXM | $1.49 | 0.44 | 161 | ~$20 | ~$240 |
| H100 SXM | $3.29 | 0.27 | 98 | ~$27 | ~$323 |

### Comparison — 24/7 dedicated vs. pay-per-use (1-year basis)

| GPU | 24/7 dedicated (always-on) | Pay-per-use (usage only) | Savings |
|---|---|---|---|
| RTX 4090 | ~$6,044/yr | ~$168/yr | ~97% cheaper |
| A100 SXM | ~$13,052/yr | ~$240/yr | ~98% cheaper |
| H100 SXM | ~$28,820/yr | ~$323/yr | ~99% cheaper |

**Takeaway:** at under 3% utilization, dedicated rental wastes ~97–99% of every dollar spent. Pay-per-use is no longer just the better option — dedicated rental is economically indefensible at this workload scale.

### Recurring vs. one-time cost — RunPod rental vs. buying hardware

| Model | Cost type | Notes |
|---|---|---|
| Renting (RunPod) | Recurring (OpEx) | Modeled above — ongoing bill, every month |
| Buying hardware outright | One-time (CapEx) + small recurring | Retail (2026): RTX 4090 ~$2,600, A100 80GB ~$11,500, H100 SXM ~$35,000 — plus electricity (~$0.15/kWh, 24/7) |

### 5-year total cost: buy vs. 24/7 rental vs. pay-per-use

| GPU | Buy outright (HW + 5yr electricity) | 5yr rental (24/7) | 5yr pay-per-use | Cheapest option |
|---|---|---|---|---|
| RTX 4090 | ~$5,557 | ~$30,220 | ~$840 | **Pay-per-use** — ~6.6× cheaper than buying outright |
| A100 SXM | ~$14,128 | ~$65,260 | ~$1,200 | **Pay-per-use** — ~11.8× cheaper than buying |
| H100 SXM | ~$39,600 | ~$144,100 | ~$1,615 | **Pay-per-use** — ~24.5× cheaper than buying |

**Takeaway:** this is the first scenario in this series where **pay-per-use wins over buying hardware across all GPU tiers** — including the RTX 4090, which favored ownership in both the 8B and 4B scenarios. At only 1–3% utilization, no hardware purchase can pay itself back fast enough to beat $14–27/month in usage billing. Buying an H100 for this workload would cost ~$24.5× more over 5 years than just paying for the GPU-minutes you actually use.

### Why this matters relative to training cost

Training (~$475 at 35% MFU) is now cheaper than a single month of even pay-per-use A100 inference, and dramatically cheaper than dedicated hosting. At this model scale, the cost conversation flips entirely: **inference economics dominate from day one**, and the correct hosting strategy is unambiguously pay-per-use or serverless — not owned or dedicated hardware. The main cost drivers shift to traffic growth, not model size or training budget.
