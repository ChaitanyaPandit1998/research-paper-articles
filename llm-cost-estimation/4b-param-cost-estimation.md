# LLM Cost Estimation — 4B Param Model (500B tokens)

Same training token count (500B) and same hardware/usage assumptions as [[8b-param-cost-estimation]], but with **half the parameters (4B vs 8B)** — isolates the effect of model size alone on cost.

---

# Part 1 — Training Cost

## Worked example: custom 4B model, 500B tokens, 16× H100 SXM cluster

### Assumptions

| Parameter | Value |
|---|---|
| Model params | 4B |
| Training tokens | 500B (unchanged from the 8B scenario) |
| Hardware | 16× H100 SXM |
| Cluster cost | $63.17/hr (≈ $3.95/hr per GPU) |
| H100 SXM peak BF16 (dense, no sparsity) | 989 TFLOPS/GPU |

### Step 1 — Total compute required

| Quantity | Value |
|---|---|
| Formula | FLOPs ≈ 6 × params × tokens |
| Calculation | 6 × 4×10⁹ × 500×10⁹ |
| **Total FLOPs** | **1.2 × 10²² (exactly half of the 8B model's 2.4 × 10²²)** |

### Step 2 — Cluster theoretical peak throughput

Unchanged from the 8B scenario — hardware doesn't change with model size.

| Quantity | Value |
|---|---|
| Per-GPU peak | 989 TFLOPS |
| Cluster size | 16 GPUs |
| **Cluster peak throughput** | **15,824 TFLOPS = 1.58 × 10¹⁶ FLOP/s** |

### Step 3 — Apply realistic MFU (Model FLOPs Utilization)

| MFU | Achieved throughput | Time to train | Wall-clock | Cost @ $63.17/hr |
|---|---|---|---|---|
| 20% (conservative) | 3.16 × 10¹⁵ FLOP/s | 1,055 hours | ~44 days | ~$66,600 |
| 35% (realistic/good) | 5.54 × 10¹⁵ FLOP/s | 602 hours | ~25 days | ~$38,000 |
| 50% (well-optimized) | 7.91 × 10¹⁵ FLOP/s | 421 hours | ~18 days | ~$26,600 |

All figures are exactly half of the 8B model's, since training FLOPs scale linearly with params for a fixed token count.

### Headline estimate

| | |
|---|---|
| Cost range | ~$26,600 – $66,600 |
| Realistic midpoint (35% MFU) | ~$38,000 |
| Wall-clock time | ~2.5–6.5 weeks on a 16-GPU cluster |

### Reference formulas

```
Training FLOPs ≈ 6 × N × D          (N = params, D = training tokens)

Time (seconds) = Total FLOPs / (achieved FLOP/s)
achieved FLOP/s = num_GPUs × peak_FLOPs_per_GPU × MFU

Cost = Time (hours) × cluster $/hour
```

---

# Part 2 — Inference Cost

## Worked example: custom 4B model, 1,000 users

Usage assumptions are identical to the 8B scenario (workload is determined by users/traffic, not model size). What changes is **GPU throughput** — a 4B model does roughly half the FLOPs per token of the 8B model, so each GPU can push through tokens roughly **2× faster**.

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

### GPU-hours needed per day (decode-bound, ~2× throughput vs. 8B model)

| GPU | Sustained decode throughput (4B, ~2×) | GPU-hours/day needed | % of a 24h day |
|---|---|---|---|
| RTX 4090 | ~4,000 tok/s | ~5.2 hours | 22% |
| A100 SXM | ~6,000 tok/s | ~3.5 hours | 14% |
| H100 SXM | ~10,000 tok/s | ~2.1 hours | 9% |

### How many GPUs are actually needed (peak load, not daily average)

Peak tok/sec needed is unchanged (workload-driven), but per-GPU capacity has roughly doubled:

| Traffic window | Peak tok/sec needed | RTX 4090 (4,000 tok/s) | A100 (6,000 tok/s) | H100 (10,000 tok/s) |
|---|---|---|---|---|
| Spread over 24h | 868 | 1 GPU (22% util) | 1 GPU (14% util) | 1 GPU (9% util) |
| 12h business day | 1,736 | 1 GPU (43% util) | 1 GPU (29% util) | 1 GPU (17% util) |
| 8h concentrated window | 2,604 | 1 GPU (65% util) | 1 GPU (43% util) | 1 GPU (26% util) |

Unlike the 8B model (which needed 2× RTX 4090 in the worst case), a **single GPU of any tier comfortably handles peak load** for a 4B model at this usage volume — 2 GPUs is still recommended for redundancy, not throughput.

### Dedicated, always-on GPU cost — RunPod on-demand pricing

GPU rental price doesn't depend on the model run on it, so these figures are **identical to the 8B scenario** — only utilization (above) differs.

| GPU | $/hr (RunPod, verified) | Monthly (730 hrs) | Annual (8,760 hrs) |
|---|---|---|---|
| RTX 4090 24GB (×1) | $0.69 | ~$504 | ~$6,044 |
| A100 SXM 80GB (×1) | $1.49 | ~$1,088 | ~$13,052 |
| H100 SXM 80GB (×1) | $3.29 | ~$2,402 | ~$28,820 |

### Pay-per-use model — billed only for GPU-hours actually consumed

Because the 4B model needs roughly half the GPU-hours/day of the 8B model, pay-per-use cost is roughly **half** of the 8B scenario's.

| GPU | $/hr | GPU-hours/day | GPU-hours/year (×365) | Monthly (usage-based) | Annual (usage-based) |
|---|---|---|---|---|---|
| RTX 4090 | $0.69 | 5.21 | 1,901 | ~$109 | ~$1,311 |
| A100 SXM | $1.49 | 3.47 | 1,267 | ~$157 | ~$1,888 |
| H100 SXM | $3.29 | 2.08 | 760 | ~$209 | ~$2,502 |

### Comparison — 24/7 dedicated vs. pay-per-use (1-year basis)

| GPU | 24/7 dedicated (always-on) | Pay-per-use (usage only) | Savings |
|---|---|---|---|
| RTX 4090 | ~$6,044/yr | ~$1,311/yr | ~78% cheaper |
| A100 SXM | ~$13,052/yr | ~$1,888/yr | ~86% cheaper |
| H100 SXM | ~$28,820/yr | ~$2,502/yr | ~91% cheaper |

**Takeaway:** with a smaller model finishing the same workload faster, 24/7 dedicated rental wastes even more capacity than in the 8B case — pay-per-use savings jump from 57–83% (8B) to 78–91% (4B).

### Recurring vs. one-time cost — RunPod rental vs. buying hardware

Hardware retail prices and electricity draw are unchanged from the 8B scenario (same physical GPUs, same power cost regardless of model size):

| Model | Cost type | Notes |
|---|---|---|
| Renting (RunPod) | Recurring (OpEx) | Modeled above — ongoing bill, every month |
| Buying hardware outright | One-time (CapEx) + small recurring | Retail (2026): RTX 4090 ~$2,600, A100 80GB ~$11,500, H100 SXM ~$35,000 — plus electricity (~$0.15/kWh, 24/7) |

### 5-year total cost: buy vs. 24/7 rental vs. pay-per-use

| GPU | Buy outright (HW + 5yr electricity) | 5yr rental (24/7) | 5yr pay-per-use | Cheapest option |
|---|---|---|---|---|
| RTX 4090 | ~$5,557 | ~$30,220 | ~$6,555 | **Buy** — but pay-per-use is now nearly as cheap (lower utilization than the 8B case) |
| A100 SXM | ~$14,128 | ~$65,260 | ~$9,440 | **Pay-per-use** — A100 utilization (14%) is too low to justify buying |
| H100 SXM | ~$39,600 | ~$144,100 | ~$12,510 | **Pay-per-use** — H100 utilization (9%) makes buying clearly the wrong call |

**Takeaway:** shrinking the model from 8B to 4B (same workload) tips the buy-vs-pay-per-use balance further toward pay-per-use across the board, because each GPU now finishes the daily workload even faster, leaving more idle capacity that a purchased GPU can't monetize. Only the RTX 4090 — the cheapest hardware, so the easiest to pay back — still favors buying, and even there the margin over pay-per-use has narrowed substantially versus the 8B scenario.

### Why this matters relative to training cost

Training (~$38K at 35% MFU) is roughly half the 8B model's ~$76K, exactly tracking the halved parameter count. Inference cost is also lower in absolute terms, but the bigger story is architectural: a smaller model shifts the *optimal hosting strategy* itself — toward pay-per-use over dedicated/owned hardware — not just the dollar figures.
