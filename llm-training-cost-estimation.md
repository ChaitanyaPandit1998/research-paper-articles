# LLM Cost Estimation — Notes

---

# Part 1 — Training Cost

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

### Reference formulas

```
Training FLOPs ≈ 6 × N × D          (N = params, D = training tokens)

Time (seconds) = Total FLOPs / (achieved FLOP/s)
achieved FLOP/s = num_GPUs × peak_FLOPs_per_GPU × MFU

Cost = Time (hours) × cluster $/hour
```

---

# Part 2 — Inference Cost

## Worked example: custom 8B model, 1,000 users

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

### How many GPUs are actually needed (peak load, not daily average)

| Traffic window | Peak tok/sec needed | RTX 4090 | A100 | H100 |
|---|---|---|---|---|
| Spread over 24h | 868 | 1 GPU (43% util) | 1 GPU (29% util) | 1 GPU (17% util) |
| 12h business day | 1,736 | 1 GPU, 87% util — no margin | 1 GPU (58% util) | 1 GPU (35% util) |
| 8h concentrated window | 2,604 | **2 GPUs needed** | 1 GPU (87% util) | 1 GPU (52% util) |

**1–2 GPUs** depending on traffic concentration and GPU choice. Production services also need redundancy (no single point of failure) — so **2 GPUs is the realistic minimum** even where 1 clears the average load.

### Dedicated, always-on GPU cost — RunPod on-demand pricing (verified directly against runpod.io/pricing)

Monthly uses 730 hours (365×24/12, the standard average-month convention) and annual uses 8,760 hours (24×365), so the two columns are internally consistent — not monthly×12, which understates the year by ~5 days' worth of runtime.

| GPU | $/hr (RunPod, verified) | Monthly (730 hrs) | Annual (8,760 hrs) | Headroom |
|---|---|---|---|---|
| RTX 4090 24GB (×1) | $0.69 | ~$504 | ~$6,044 | Tight — consider 2× for safety margin |
| RTX 4090 24GB (×2, recommended) | $0.69 | ~$1,007 | ~$12,089 | Comfortable |
| A100 SXM 80GB (×1) | $1.49 | ~$1,088 | ~$13,052 | Comfortable |
| H100 SXM 80GB (×1) | $3.29 | ~$2,402 | ~$28,820 | Very comfortable, room to grow |

For reference, RunPod also lists A100 PCIe at $1.39/hr and H100 PCIe at $2.89/hr — PCIe variants are slightly cheaper than SXM but with lower interconnect bandwidth (not a concern for single-GPU inference serving).

### Pay-per-use model — RunPod, billed only for GPU-hours actually consumed

Instead of renting a GPU 24/7, this model assumes RunPod's pay-per-use/Serverless billing: you pay only for the GPU-hours actually needed to process the workload (from the GPU-hours/day table above), not for idle time.

| GPU | $/hr (RunPod, verified) | GPU-hours/day | GPU-hours/year (×365) | Monthly (usage-based) | Annual (usage-based) |
|---|---|---|---|---|---|
| RTX 4090 | $0.69 | 10.42 | 3,802 | ~$219 | ~$2,623 |
| A100 SXM | $1.49 | 6.94 | 2,535 | ~$315 | ~$3,777 |
| H100 SXM | $3.29 | 4.17 | 1,521 | ~$417 | ~$5,004 |

### Comparison — 24/7 dedicated vs. pay-per-use

| GPU | 24/7 dedicated (always-on) | Pay-per-use (usage only) | Savings |
|---|---|---|---|
| RTX 4090 | ~$6,044/yr | ~$2,623/yr | ~57% cheaper |
| A100 SXM | ~$13,052/yr | ~$3,777/yr | ~71% cheaper |
| H100 SXM | ~$28,820/yr | ~$5,004/yr | ~83% cheaper |

**Takeaway:** paying only for GPU-seconds actually consumed cuts cost by 57–83% versus renting a GPU around the clock. Under pay-per-use billing, H100 becomes the cheapest option in absolute terms despite its higher hourly rate, since it clears the workload fastest and racks up the fewest billed hours.

**Caveat:** this assumes pure usage-based billing with no idle/cold-start overhead and no concurrency penalty. In practice, serverless workers are typically kept "warm" for a short idle window after each request to avoid cold-start latency, which adds some billed time beyond pure compute-seconds — actual cost will land somewhere between this estimate and the 24/7 dedicated figure. If many of the 1,000 users hit the service concurrently rather than spread across the day, you may also need ≥2 workers during peak hours regardless of billing model.

### Recurring vs. one-time cost — RunPod rental vs. buying hardware

All GPU cost figures above (24/7 dedicated and pay-per-use) are **recurring (OpEx)** — RunPod rental is billed continuously for as long as the instance runs; there's no point where it's "paid off."

| Model | Cost type | Notes |
|---|---|---|
| Renting (RunPod) | Recurring (OpEx) | Modeled above — ongoing bill, every month |
| Buying hardware outright | One-time (CapEx) + small recurring | Retail (2026): RTX 4090 ~$2,600, A100 80GB ~$11,500, H100 SXM ~$35,000 — plus electricity (~$0.15/kWh, 24/7), much smaller than rental rates |

### 5-year total cost: buy vs. 24/7 rental vs. pay-per-use

Assumes hardware bought once, electricity at ~$0.15/kWh 24/7 (RTX 4090 450W, A100 400W, H100 700W TDP).

| GPU | Buy outright (HW + 5yr electricity) | 5yr rental (24/7) | 5yr pay-per-use | Cheapest option |
|---|---|---|---|---|
| RTX 4090 | ~$5,557 | ~$30,220 | ~$13,115 | **Buy** (5.4× cheaper than 24/7 rental) |
| A100 SXM | ~$14,128 | ~$65,260 | ~$18,885 | **Buy** (4.6× cheaper than 24/7 rental) |
| H100 SXM | ~$39,600 | ~$144,100 | ~$25,020 | **Pay-per-use** — H100 is only ~17% utilized at this workload, so its high purchase price doesn't pay back fast enough to beat usage-based billing |

**Takeaway:** buying wins decisively over 24/7 rental at every GPU tier (4.6–5.4× cheaper over 5 years) — 24/7 rental is the worst option whenever utilization is well below 100%, which it is here for all three GPUs. Between buying and pay-per-use specifically, buying wins for RTX 4090 and A100 (high relative utilization, ~30–43%, makes ownership pay off), but **pay-per-use wins for H100** because its purchase price is too high to justify against only ~17% utilization — the breakeven point depends on utilization, not just hardware cost.

### Why this matters relative to training cost

Training (~$76K) is a one-time cost; inference (~$12,089–$48,100/year depending on hosting choice) is recurring. Over a multi-year deployment, or as user count grows, cumulative inference cost can equal or exceed the original training spend — this is the same dynamic that justifies "overtraining" a smaller model in the first place (see Chinchilla discussion): paying more upfront in training to keep the permanent per-query cost low.
