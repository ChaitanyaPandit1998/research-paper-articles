# LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
**Year:** 2021
**Paper:** https://arxiv.org/abs/2106.09685

---

## The Problem

Fine-tuning large language models (e.g. GPT-3 with 175B parameters) requires updating *all* weights — extremely expensive in GPU memory and storage. Deploying multiple task-specific versions means storing multiple full model copies.

---

## Core Idea

When fine-tuning, the model learns a weight update `ΔW` on top of frozen pre-trained weights `W₀`:

```
new_weights = W₀ + ΔW
```

LoRA's insight: **ΔW is intrinsically low-rank** — the meaningful signal fits in a very low-dimensional space. So instead of learning the full update, decompose it into two small matrices:

```
ΔW = B × A

where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and rank r ≪ min(d, k)
```

The forward pass becomes:

```
h = W₀x + BAx
```

- `W₀` stays **frozen** during training
- Only `A` and `B` are trained (initialized: A ~ random Gaussian, B = 0)
- At inference, merge back: `W = W₀ + BA` — **no extra latency**

---

## Key Insight

Weight updates during adaptation have surprisingly low intrinsic dimensionality. Even for GPT-3 (175B params), ranks as small as **r = 1–4** often work well. Higher rank adds noise, not signal.

> "The top singular-vector directions of A are the most useful, while other directions potentially contain mostly random noise accumulated during training."

---

## Results

| | Full Fine-tuning | LoRA (r=4) |
|---|---|---|
| Trainable params (GPT-3) | 175B | ~4.7M |
| GPU memory reduction | baseline | ~3x less |
| Inference latency | none | **none** (weights merged) |
| Task-specific storage | full model | tiny A, B matrices |
| Performance | baseline | comparable or better |

Tested on: RoBERTa, DeBERTa, GPT-2, GPT-3 — matches or beats full fine-tuning.

---

## Why It Matters

- **Democratizes fine-tuning** — large models can be adapted on consumer hardware
- **Multi-task efficiency** — store only small adapter matrices (MBs) per task, not full model copies
- **No inference overhead** — adapters merge into base weights at deployment
- **Foundation for modern PEFT** — spawned QLoRA, AdaLoRA, DoRA, and most parameter-efficient fine-tuning methods used today

---

## Where LoRA is Applied

LoRA is typically injected into the **attention weight matrices** of each Transformer layer:
- Query projection (`W_q`)
- Value projection (`W_v`)
- (Optionally: key `W_k`, output `W_o`, MLP layers)

---

## Hyperparameters

| Param | Meaning | Typical values |
|---|---|---|
| `r` | Rank of decomposition | 1, 2, 4, 8, 16 |
| `α` | Scaling factor (`α/r` scales ΔW) | 16, 32 |
| Target modules | Which weight matrices to adapt | `q_proj`, `v_proj` |
