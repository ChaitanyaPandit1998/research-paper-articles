# GRPO: Group Relative Policy Optimization — Implementation Notes

**Paper**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (arXiv: 2402.03300)  
**Code**: `huggingface/trl` → `trl/trainer/grpo_trainer.py`

---

## The Full Formula

```
L_GRPO(θ) = (1/G) Σᵢ (1/|oᵢ|) Σₜ min[ r_t^i · Âᵢ, g(ε, Âᵢ) ] − β · D_KL[π_θ || π_ref]
```

Where:
- `G` = number of completions sampled per prompt
- `|oᵢ|` = length of completion `i` (number of tokens)
- `r_t^i = π_θ(oᵢ,ₜ | q, oᵢ,<ₜ) / π_θ_old(oᵢ,ₜ | q, oᵢ,<ₜ)` = per-token probability ratio
- `Âᵢ` = group-relative advantage for completion `i`
- `g(ε, Â)` = clipped surrogate: `clip(r, 1−ε, 1+ε) · Â`
- `β` = KL penalty coefficient
- `π_ref` = frozen reference model

---

## Formula → Code Mapping

### 1. `(1/G) Σᵢ` — Average over G completions

```python
self.num_generations = args.num_generations  # = G   (line 598)
```

The `RepeatSampler` repeats each prompt G times so that G completions land in the same batch. The outer `.mean()` at line 2761 averages across all G×batch_size rows:

```python
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
#                                                                        ^^^^^^ = (1/G) Σᵢ
```

---

### 2. `(1/|oᵢ|) Σₜ` — Average loss per token within each completion

```python
# line 2761
(per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# sum over tokens t                divide by |oᵢ| (non-padding tokens)
```

`mask` is the `completion_mask` — `1` for real tokens, `0` for padding. Dividing by `mask.sum(-1)` gives each completion equal contribution regardless of length.

---

### 3. `r_t^i = π_θ / π_θ_old` — Probability ratio

```python
# line 2690
log_ratio = per_token_logps - old_per_token_logps   # log(π_θ / π_θ_old) per token

# line 2702
coef_1 = torch.exp(log_importance_weights)           # actual ratio r_t^i
```

- `per_token_logps` — computed fresh from the **current** model each step
- `old_per_token_logps` — stored snapshot from when the completions were **generated**

---

### 4. `Âᵢ` — Group-relative advantage (no critic needed)

```python
# line 2310-2311 — group mean: average reward across the G siblings
mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)

# line 2315-2316 — group std
std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)

# line 2330-2332 — Â = (R - mean) / (std + ε)
advantages = rewards - mean_grouped_rewards
advantages = advantages / (std_rewards + 1e-4)
```

**Key insight**: GRPO eliminates the value/critic network entirely. The baseline is simply the mean reward of the G sibling completions for the same prompt — that's the "Group Relative" in the name.

---

### 5. `g(ε, Â)` — Clipped surrogate (same as PPO clip)

```python
# line 2720 — clamp ratio to [1−ε_low, 1+ε_high]
coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

per_token_loss1 = coef_1 * advantages    # unclipped: r_t · Â
per_token_loss2 = coef_2 * advantages    # clipped:   clip(r_t) · Â

# line 2727 — take min, negate (we minimize loss = maximize objective)
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
```

When the ratio `r_t` drifts too far from 1 (policy changed a lot), the clipped term dominates and kills the gradient — preventing destructively large updates.

---

### 6. `β · D_KL[π_θ || π_ref]` — KL divergence penalty

```python
# line 2707-2709 — Schulman approximation: exp(x) - x - 1 ≥ 0 always
per_token_kl = (
    torch.exp(ref_per_token_logps - per_token_logps)
    - (ref_per_token_logps - per_token_logps) - 1
)

# line 2757 — add to per-token loss
per_token_loss = per_token_loss + self.beta * per_token_kl
```

- `ref_per_token_logps` — log probs from the frozen reference model (`self.ref_model`)
- When `beta=0.0`, the reference model is never loaded at all (line 754) — clean optimization

---

### Full Assembly (default `loss_type="grpo"`)

```python
# _compute_loss(), lines 2719–2763
coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)  # PPO surrogate
per_token_loss = per_token_loss + self.beta * per_token_kl              # + β·KL
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
#        └──── (1/|oᵢ|) Σₜ ──────────┘   └──── (1/G) Σᵢ ─────────┘
```

---

## Multiple Update Iterations (μ)

```python
self.num_iterations = args.num_iterations  # = μ in the GRPO paper  (line 716)
```

GRPO reuses the same generated completions for `μ` gradient steps before regenerating. This is managed via `_buffered_inputs` (line 723). The cadence is:

```
Generate completions → run μ × steps_per_generation gradient updates → regenerate
```

---

## Key Hyperparameters

| Config Param | Paper Notation | Meaning |
|---|---|---|
| `num_generations` | G | Completions sampled per prompt |
| `num_iterations` | μ | Gradient steps before regeneration |
| `epsilon` / `epsilon_high` | ε | PPO clip bounds (low / high) |
| `beta` | β | KL penalty coefficient |
| `loss_type` | — | Variant: `grpo`, `dapo`, `bnpo`, `dr_grpo`, `cispo`, etc. |
| `scale_rewards` | — | `"group"` (default), `"batch"`, or `"none"` |

---

## Loss Type Variants

All variants share the same group-relative advantage computation but differ in how the policy ratio is clipped and the loss is normalised:

| `loss_type` | Normalisation | Clipping style |
|---|---|---|
| `grpo` | Mean per-sequence, then mean over batch | Two-sided PPO clip |
| `bnpo` | Divide by total valid tokens in batch | Two-sided PPO clip |
| `dr_grpo` | Divide by `batch_size × max_completion_length` | Two-sided PPO clip |
| `dapo` | Divide by total completion tokens globally | Two-sided PPO clip |
| `cispo` | Global token count | Upper-only clip, multiply logprob directly |
| `luspo` | Sequence-level IS weight | Two-sided PPO clip |
| `sapo` | Mean per-sequence | Soft sigmoid clipping |
| `vespo` | Global token count | Gamma-weighted logprob |

---

## GRPO vs PPO: Key Difference

| | PPO | GRPO |
|---|---|---|
| Advantage estimate | Value network (critic) | Group mean reward (no extra model) |
| Memory overhead | ~2× model size (actor + critic) | ~1× (or ~2× with ref model) |
| Baseline quality | Learned, dense signal | Simple mean, but reward-shaped |
| Main use case | General RL | LLM fine-tuning with verifiable rewards |
