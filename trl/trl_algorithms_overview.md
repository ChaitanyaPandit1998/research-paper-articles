# TRL Algorithms Overview

**Repo**: `huggingface/trl` → `/Users/chaitanya/Development/AI/trl`

A survey of the key training algorithms implemented in TRL, with core formulas and code references.

---

## Algorithm Map

| Algorithm | Trainer File | Data Needed | Reward Model | Core Idea |
|---|---|---|---|---|
| SFT | `trainer/sft_trainer.py` | Demonstrations | None | Next-token prediction |
| Reward Modeling | `trainer/reward_trainer.py` | Preference pairs | This IS it | Bradley-Terry ranking loss |
| PPO | `experimental/ppo/ppo_trainer.py` | Rollouts | Separate model | Clipped PG + value function |
| GRPO | `trainer/grpo_trainer.py` | Prompts only | Reward fn(s) | Group-relative advantage, no critic |
| RLOO | `trainer/rloo_trainer.py` | Prompts only | Reward fn(s) | Leave-one-out PG baseline |
| DPO | `trainer/dpo_trainer.py` | Preference pairs | No (implicit) | Implicit reward via log-ratios |
| KTO | `trainer/kto_trainer.py` | Single labels | No (implicit) | Prospect-theoretic, unpaired data |

---

## 1. SFT — Supervised Fine-Tuning

**File**: `trl/trainer/sft_trainer.py`  
**Paper**: Standard next-token prediction; optional DFT variant (arXiv:2508.05629)

### Idea
The simplest baseline: train the model to imitate human demonstrations via cross-entropy loss on the completion tokens only (prompt tokens are masked out with `-100`).

### Loss Formula

**Default (NLL)**:
```
L_SFT = -Σₜ log p_θ(yₜ | x, y_{<t}) / num_tokens
```

**Optional DFT (Data-weighted)**:
```
L_DFT = Σₜ -p_θ(yₜ).detach() · log p_θ(yₜ) / num_tokens
```
DFT down-weights high-confidence predictions, addressing overconfidence in imitation learning.

### Key Code
```python
shift_labels = labels[..., 1:]
loss_mask = shift_labels != -100
logprobs = selective_log_softmax(outputs.logits, shift_labels)
per_token_loss = -logprobs.exp().detach() * logprobs   # DFT variant
loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
```

---

## 2. Reward Modeling

**File**: `trl/trainer/reward_trainer.py`  
**Paper**: Christiano et al. RLHF (arXiv:1706.03762)

### Idea
Train a scalar reward model (sequence classifier with `num_labels=1`) using the **Bradley-Terry** ranking model: the chosen completion should score higher than the rejected one. The learned reward is later used to drive PPO, GRPO, RLOO, etc.

### Loss Formula

```
L_RM = -log σ(r_chosen - r_rejected - margin)
```

Optional regularisation to prevent reward hacking via score drift:
```
L_reg = center_coef · mean((r_chosen + r_rejected)²)
```

### Key Code
```python
rewards_chosen, rewards_rejected = torch.chunk(outputs.logits.squeeze(-1), chunks=2)

if "margin" in inputs:
    loss = -F.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
else:
    loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

if self.args.center_rewards_coefficient is not None:
    loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)
```

---

## 3. PPO — Proximal Policy Optimization

**File**: `trl/experimental/ppo/ppo_trainer.py`  
**Paper**: Fine-Tuning Language Models from Human Preferences (arXiv:1909.08593)

### Idea
On-policy RL with a **learned value function (critic)**. Generates rollouts, computes advantages via GAE, then updates both policy (actor) and value (critic) with clipped objectives. The clip prevents destructively large policy updates.

### Loss Formula

```
r_t = π_θ(t) / π_θ_old(t)   (probability ratio)

L_policy = -min(r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t)

L_value  = 0.5 · max((V - R)², (clip(V, V_old−c, V_old+c) − R)²)

L_PPO = L_policy + c_vf · L_value − c_entropy · H(π_θ)
```

Where `A_t` comes from a **separate learned value network** (GAE estimate).

### Key Code
```python
ratio = torch.exp(logprobs_diff)
pg_losses  = -mb_advantage * ratio
pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
pg_loss = torch.max(pg_losses, pg_losses2).mean()
loss = pg_loss + args.vf_coef * vf_loss
```

### PPO vs GRPO
| | PPO | GRPO |
|---|---|---|
| Advantage source | Learned value network (GAE) | Group mean reward (no extra model) |
| Memory | ~2× model size | ~1× (ref model optional) |
| Complexity | Higher (actor + critic + reward model) | Lower (reward fn only) |

---

## 4. GRPO — Group Relative Policy Optimization

**File**: `trl/trainer/grpo_trainer.py`  
**Paper**: DeepSeekMath (arXiv:2402.03300)

### Idea
Same PPO-style clipped objective but **eliminates the value/critic network**. Instead, for each prompt, sample G completions and use the **within-group mean reward as the baseline**. See [`grpo_implementation_notes.md`](grpo_implementation_notes.md) for full deep-dive.

### Loss Formula

```
L_GRPO(θ) = (1/G) Σᵢ (1/|oᵢ|) Σₜ min[ r_t^i · Âᵢ, clip(r_t^i, 1−ε, 1+ε) · Âᵢ ]
            − β · D_KL[π_θ || π_ref]

Âᵢ = (Rᵢ − mean({Rⱼ})) / (std({Rⱼ}) + ε)
```

### Key Code
```python
# Advantage: group-relative normalisation (lines 2330-2332)
advantages = rewards - mean_grouped_rewards
advantages = advantages / (std_rewards + 1e-4)

# Clipped surrogate (lines 2720-2727)
coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)

# Final loss (line 2761)
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
```

---

## 5. RLOO — Reinforce Leave-One-Out

**File**: `trl/trainer/rloo_trainer.py`  
**Paper**: Back to Basics: Revisiting REINFORCE Style Optimization for LLMs (arXiv:2402.14740)

### Idea
Returns to the REINFORCE policy gradient algorithm with a **leave-one-out (LOO) baseline**: for each of the G completions, the baseline is the mean reward of the *other* G−1 siblings (not including itself). This reduces variance while avoiding the complexity of a value network.

No clipping — unlike PPO/GRPO, RLOO uses unclipped gradients, making it closer to vanilla policy gradient.

### Loss Formula

```
KL_i = Σₜ log π_θ(t) - log π_ref(t)         (token-level KL)

R̃ᵢ = Rᵢ − β · KL_i                          (KL-adjusted reward)

Aᵢ = R̃ᵢ − (1/(G−1)) Σⱼ≠ᵢ R̃ⱼ              (leave-one-out baseline)

L_RLOO = -Σᵢ Σₜ log π_θ(t) · Aᵢ / |oᵢ|
```

### Key Code
```python
# KL penalty applied to rewards (not loss)
per_token_kl = old_per_token_logps - ref_per_token_logps
kl = (per_token_kl * completion_mask).sum(-1)
rewards = rewards - self.beta * kl

# Leave-one-out baseline
grouped_rewards = rewards.view(-1, num_generations)     # (num_prompts, G)
baseline = (grouped_rewards.sum(1, keepdim=True) - grouped_rewards) / (num_generations - 1)
advantages = (grouped_rewards - baseline).view(-1)
```

### RLOO vs GRPO
| | RLOO | GRPO |
|---|---|---|
| Baseline | Leave-one-out mean (LOO) | Full group mean |
| KL penalty | Applied to reward before advantage | Added to loss term |
| Clipping | No clipping | PPO-style clip |
| Bias | Slightly lower bias (LOO is unbiased) | Slightly higher bias |

---

## 6. DPO — Direct Preference Optimization

**File**: `trl/trainer/dpo_trainer.py`  
**Paper**: Direct Preference Optimization: Your LM is Secretly a Reward Model (arXiv:2305.18290)

### Idea
Skips reward model training entirely. DPO shows that the optimal RLHF policy has a **closed-form relationship** with the preference data — you can directly optimise the policy using preference pairs without ever training or querying a separate reward model.

The key insight: the reward is implicitly represented as `r(x,y) = β log(π_θ(y|x) / π_ref(y|x))`.

### Loss Formula

```
L_DPO(θ) = -E_{(x, y_w, y_l)} [ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x)
                                             - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

Expanded:
```
chosen_logratio   = log π_θ(y_w|x) − log π_ref(y_w|x)
rejected_logratio = log π_θ(y_l|x) − log π_ref(y_l|x)

L_DPO = -log σ( β · (chosen_logratio − rejected_logratio) )
```

### Key Code
```python
chosen_logratios  = chosen_logps  - ref_chosen_logps
rejected_logratios = rejected_logps - ref_rejected_logps

delta_score = chosen_logratios - rejected_logratios      # margin between chosen/rejected
per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
loss = per_sequence_loss.mean()
```

### Variants supported in TRL
| `loss_type` | Description |
|---|---|
| `sigmoid` (default) | Standard DPO |
| `hinge` | SLiC-style hinge loss |
| `ipo` | IPO: avoids overconfidence issues in DPO |
| `robust` | cDPO: handles noisy preference labels |
| `bco_pair` | BCO paired variant |
| `nca_pair` | NCA paired variant |
| `aot` / `aot_pair` | AOT: optimise over sorted advantages |
| `sppo_hard` | SPPO hard variant |

---

## 7. KTO — Kahneman-Tversky Optimization

**File**: `trl/trainer/kto_trainer.py`  
**Paper**: KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)

### Idea
Inspired by **Kahneman-Tversky prospect theory** (humans feel losses more acutely than equivalent gains). KTO works with **unpaired data** — each sample is just a completion labelled "good" (desirable) or "bad" (undesirable), with no paired comparisons needed.

Chosen and rejected completions are handled **asymmetrically**, with different KL correction directions for each.

### Loss Formula

```
KL = mean over unmatched pairs of [ log π_θ(y|x) - log π_ref(y|x) ]

For desirable (chosen):
  L_chosen   = 1 − σ( β · (log π_θ(y_w|x)/π_ref(y_w|x) − KL) )

For undesirable (rejected):
  L_rejected = 1 − σ( β · (KL − log π_θ(y_l|x)/π_ref(y_l|x)) )

L_KTO = λ_w · mean(L_chosen) + λ_u · mean(L_rejected)
```

Where `λ_w` and `λ_u` are desirable/undesirable weights.

### Key Code
```python
chosen_logratios  = policy_chosen_logps  - ref_chosen_logps
rejected_logratios = policy_rejected_logps - ref_rejected_logps

# KL is computed on mismatched (not same-prompt) pairs
chosen_losses   = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))

loss = (self.desirable_weight   * chosen_losses.nanmean()
      + self.undesirable_weight * rejected_losses.nanmean())
```

### KTO vs DPO
| | DPO | KTO |
|---|---|---|
| Data format | Paired (chosen + rejected) | Unpaired (single label) |
| KL role | Implicit (in log-ratio margin) | Explicit separate KL term |
| Weighting | Symmetric | Asymmetric (λ_w ≠ λ_u) |
| Use case | Paired preference datasets | Binary feedback / scalar labels |

---

## RLHF Pipeline: How the Algorithms Connect

```
Raw data
   │
   ▼
[SFT Trainer]  ──────────────────────────────►  SFT model (π_SFT)
                                                      │
   ┌──────────────────────────────────────────────────┤
   │                                                  ▼
   │                                     [Reward Trainer]  ──► Reward model r(x,y)
   │                                                  │
   │              ┌───────────────────────────────────┤
   │              │            │            │         │
   ▼              ▼            ▼            ▼         ▼
[PPO]          [GRPO]       [RLOO]       [DPO]     [KTO]
uses r(x,y)  uses reward   uses reward  no r(x,y)  no r(x,y)
+ value net    fn(s)         fn(s)       (implicit) (unpaired)
```

**Key gradient**:
- Need a reward model + value net → **PPO**
- Need a reward model, want simplicity → **GRPO** or **RLOO**
- Have preference pairs, want offline training → **DPO**
- Have only binary labels (good/bad) → **KTO**
