# GRPO: How DeepSeek Trains Reasoning Models Without a Critic

### A deep dive into Group Relative Policy Optimization for LLM post-training

---

## Table of Contents

1. [What is it?](#what-is-it)
2. [Use case](#use-case)
3. [Different RL post-training approaches actively used](#different-rl-post-training-approaches-actively-used)
4. [Memory footprint compared](#memory-footprint-compared)
5. [Glossary of terms](#glossary-of-terms)
6. [Formula/objective for each](#formulaobjective-for-each)
7. [Line-by-line: what each variable means](#line-by-line-what-each-variable-means)
8. [Worked example: GRPO by the numbers](#worked-example-grpo-by-the-numbers)
9. [Explanation for each](#explanation-for-each)
10. [GRPO in practice: pitfalls & follow-ups](#grpo-in-practice-pitfalls--follow-ups)
11. [Python example](#python-example)
12. [Hyperparameter cheat sheet](#hyperparameter-cheat-sheet)
13. [Pros and cons](#pros-and-cons)
14. [Summary](#summary)
15. [Key takeaways](#key-takeaways)

---

## What is it?

GRPO (**Group Relative Policy Optimization**) is a reinforcement learning algorithm for fine-tuning large language models, introduced by DeepSeek in the DeepSeek-Math paper and later popularized as the workhorse behind DeepSeek-R1. It's the technique that turned a merely fluent language model into one that can grind through a multi-step math proof or debug its own reasoning mid-generation.

To understand why GRPO exists, it helps to remember what pretraining and SFT *don't* do.

Pretraining teaches a model to predict the next token over a massive corpus of text, using plain cross-entropy loss. This gives the model broad knowledge and fluency, but it optimizes for "what token is statistically likely here," not "what response is actually good, correct, or aligned with what a human wants." Supervised fine-tuning (SFT) narrows this by training on curated (prompt, ideal-response) pairs — but it's still cross-entropy loss, still just mimicking a fixed set of examples.

The problem is that cross-entropy loss has no notion of *degrees* of quality. It treats every token in the human-written reference as equally correct and every deviation as equally wrong. It can't tell the model "this answer is 80% as good as the reference" or "this answer is wrong for a subtle reason not represented in your training set." It also can't optimize for things that are easy to *evaluate* but hard to *demonstrate* — like "is this proof logically valid," "is this response harmless," or "did this code actually pass the test suite." SFT can only imitate; it can't be told when its own generations are good or bad.

This is exactly the gap reinforcement learning fills. Instead of matching a fixed target, RL lets you define a **reward function** — a scoring signal for how good a generated response is — and then optimizes the model's *own* outputs against that signal. GRPO is one specific, increasingly popular way to do this optimization efficiently, without needing a separate critic network to estimate value.

---

## Use case

GRPO sits at the very end of the modern LLM training pipeline:

```
Pretraining  →  SFT  →  RL fine-tuning (GRPO / PPO / DPO)
(next-token)   (imitate    (optimize against
                examples)   a reward signal)
```

It's used specifically for tasks where you can *score* a response more reliably than you can *hand-write* the ideal response — which is precisely the situation with:

- **Mathematical and logical reasoning** — you can check if the final answer is correct (or run a verifier), even if writing a perfect step-by-step solution by hand is tedious to scale.
- **Code generation** — you can run unit tests and use pass/fail as reward.
- **Alignment and preference optimization** — you can use a trained reward model or human preference data to score helpfulness/harmlessness, even without a single "correct" response.

GRPO matters because it made RL fine-tuning dramatically cheaper. The dominant RLHF recipe used by OpenAI for InstructGPT and early ChatGPT relies on PPO (Proximal Policy Optimization), which requires training and holding in memory a separate **value/critic model** roughly the same size as the policy model — doubling GPU memory and adding training instability. GRPO's key insight is that you can estimate how good a response is by comparing it to *other responses sampled for the same prompt*, entirely sidestepping the need for a critic. This made large-scale RL fine-tuning of reasoning models tractable for teams without OpenAI-scale infrastructure, and it's a big part of why DeepSeek-R1 was able to reach frontier-level reasoning performance at a fraction of the reported training cost.

---

## Different RL post-training approaches actively used

Three approaches dominate the current landscape, each representing a different point on the complexity/cost/stability tradeoff curve:

| Approach | Origin | Notable users |
|---|---|---|
| **RLHF with PPO** | Schulman et al. 2017 (PPO); Ouyang et al. 2022 (InstructGPT) | OpenAI's InstructGPT, early ChatGPT/GPT-4 RLHF stages |
| **DPO** (Direct Preference Optimization) | Rafailov et al. 2023 | Many open-source post-training recipes: Zephyr, Llama alignment recipes, Mistral-instruct variants |
| **GRPO** | Shao et al. 2024 (DeepSeek-Math) | DeepSeek-Math, DeepSeek-R1, increasingly adopted in open reasoning-model recipes (e.g., via TRL, Unsloth) |

The lineage roughly goes: **PPO-based RLHF** established the paradigm of "train a reward model, then optimize a policy against it with a KL constraint." **DPO** asked "do we even need the RL machinery?" and reformulated preference optimization as a single supervised-style loss directly on preference pairs, skipping the reward model and policy-gradient sampling loop entirely. **GRPO** went a different direction — it kept the RL/reward-model paradigm (useful when you have a scalar reward, not just pairwise preferences) but eliminated the expensive critic network by using intra-group comparisons instead.

---

## Memory footprint compared

The clearest way to see why labs moved off PPO is to count how many full model copies have to sit in GPU memory at once during training.

| Method | Trainable models | Frozen models | Total in memory | Notes |
|---|---|---|---|---|
| **PPO** | Policy + Critic | Reference + Reward model | 4 | Critic is typically ~the same size as the policy — this is what doubles cost |
| **DPO** | Policy | Reference | 2 | No reward model or critic at train time at all |
| **GRPO** | Policy | Reference + Reward (model or rule-based function) | 2–3 | Reward can be a lightweight verifier, not a full network — often cheaper than PPO's reward model too |

For a 7B-parameter model, PPO's critic alone adds roughly 14GB+ of weights (fp16), plus its own optimizer states (Adam roughly doubles that again) — on top of the policy, reference, and reward model already in memory. That extra model is the single biggest line item PPO carries that GRPO and DPO don't.

---

## Glossary of terms

A few RL terms get used throughout without a formal definition — here they are in one place.

- **Policy (π)** — the model being trained. Given a prompt, it outputs a probability distribution over next tokens — sampling from that distribution is how a response gets generated.
- **Rollout** — one full generated response, sampled from the policy for a given prompt. GRPO samples `G` rollouts per prompt per step.
- **Reference policy (π_ref)** — a frozen snapshot of the model, usually the SFT checkpoint before any RL, used purely as an anchor so the trained policy doesn't drift too far during optimization.
- **Reward model** — a separate trained network that scores a (prompt, response) pair with a scalar quality estimate. Used when there's no automatic ground truth (e.g., "how helpful was this reply").
- **Reward function** — a programmatic or rule-based scorer — "did the code pass its tests," "does the final numeric answer match" — cheaper and more reliable than a learned reward model whenever the task allows it.
- **Advantage (A)** — an estimate of how much better or worse a given action (token or response) was than expected. It's what tells the update direction whether to reinforce or suppress something.
- **Critic / value model** — a learned network that predicts expected future reward from a given state. PPO uses it to compute advantages; GRPO and DPO have no equivalent.
- **KL divergence** — a measure of how different two probability distributions are. Here: how far the trained policy's output distribution has drifted from the reference policy's, for the same prompt.

---

## Formula/objective for each

### PPO (as used in RLHF)

The clipped surrogate objective, maximized with respect to policy parameters θ:

```
L_PPO(θ) = E_t [ min( r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]

where:
  r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)      (probability ratio)
  A_t    = advantage estimate at token t (from a learned value/critic model, via GAE)
```

The full RLHF objective adds a KL penalty against the reference (SFT) policy:

```
Objective = E [ L_PPO(θ) ]  -  β · KL[ π_θ(·|x)  ||  π_ref(·|x) ]
```

### DPO

DPO reframes RLHF's KL-constrained reward maximization as a single closed-form classification loss over preference pairs (y_w = preferred/winning response, y_l = dispreferred/losing response):

```
L_DPO(θ) = -E_(x, y_w, y_l) [
    log σ( β · log( π_θ(y_w|x) / π_ref(y_w|x) )
         - β · log( π_θ(y_l|x) / π_ref(y_l|x) ) )
]
```

where σ is the sigmoid function and β controls how sharply the model deviates from the reference policy.

### GRPO

For a prompt *x*, sample a **group** of G responses {y_1, ..., y_G} from the old policy, each scored by a reward function r_i. Compute group-relative advantages by normalizing within the group:

```
A_i = ( r_i - mean(r_1, ..., r_G) ) / std(r_1, ..., r_G)
```

The GRPO objective (per token t in response i), with clipping like PPO but no learned critic:

```
L_GRPO(θ) = E [ (1/G) Σ_i (1/|y_i|) Σ_t
    min( ρ_i,t(θ) · A_i,  clip(ρ_i,t(θ), 1-ε, 1+ε) · A_i )
  ]  -  β · D_KL[ π_θ || π_ref ]

where:
  ρ_i,t(θ) = π_θ(y_i,t | x, y_i,<t) / π_θ_old(y_i,t | x, y_i,<t)
```

The KL term is typically estimated with the unbiased low-variance estimator:

```
D_KL[π_θ || π_ref] ≈ π_ref(y|x)/π_θ(y|x) - log(π_ref(y|x)/π_θ(y|x)) - 1
```

applied per-token and averaged, rather than a Monte Carlo KL sample.

---

## Line-by-line: what each variable means

The formulas above pack a lot into a few symbols. Here's every variable unpacked, formula by formula.

### PPO clipped surrogate objective

- **θ** — the parameters of the policy network being trained (the LLM's weights) — what gradient descent updates.
- **L_PPO(θ)** — the objective (loss, but maximized) as a function of those parameters — the number PPO's optimizer pushes up.
- **E_t [ … ]** — "expectation over t": average this quantity over every token position *t* in the sampled response(s). In practice, a mean over a batch of tokens.
- **r_t(θ)** — the probability ratio at token *t*:

  ```
  r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
  ```

  `π_θ(a_t | s_t)` is the probability the *current* policy assigns to the token actually generated (`a_t`), given everything generated so far (`s_t` — the prompt plus prior tokens). `π_θ_old(a_t | s_t)` is the probability the policy assigned to that same token *before* this update step began (the snapshot used for sampling). The ratio answers: "has the model become more or less likely to say this since the update started?" `r_t = 1` means unchanged; `r_t = 1.5` means 50% more likely now.

- **A_t** — the advantage at token *t*: how much better (positive) or worse (negative) this token/response turned out versus what the critic expected at that point. This is the actual signal driving the update direction. In PPO it comes from a separately trained value/critic network, combined with the observed reward via Generalized Advantage Estimation (GAE).
- **clip(r_t(θ), 1−ε, 1+ε)** — the ratio clamped to stay within `[1−ε, 1+ε]` (commonly ε = 0.2, roughly `[0.8, 1.2]`) — caps how far the ratio can move in one update.
- **min(unclipped, clipped)** — PPO takes the *smaller* of the raw and clipped terms — the "proximal" part of the algorithm. It stops any single update from moving the policy's probability on a token too far in one step, whether that move would help (A_t > 0) or hurt (A_t < 0) the objective.

### Full RLHF objective (adds the KL term)

- **E_x [ … ]** — average over prompts *x* in the batch — an outer expectation on top of the token-level one above.
- **β** — scalar weight controlling how strongly the KL penalty is enforced. Higher β keeps the policy closer to the reference model; lower β lets it drift further toward whatever maximizes reward.
- **KL_x[π_θ(·|x) ‖ π_ref(·|x)]** — the KL divergence between the current policy's output distribution and the frozen reference policy π_ref (typically the SFT model before RL), both conditioned on the same prompt. It measures how far current behavior has drifted from where it started.
- **why subtract it** — without this term, the policy can chase reward however it likes, including exploiting quirks in the reward model (reward hacking) instead of genuinely improving. Subtracting β·KL penalizes drifting too far from the reference distribution, anchoring the model's fluency and general behavior to what SFT already established.

### DPO loss

- **θ** — same as above: the policy's parameters being trained.
- **E_(x, y_w, y_l) [ … ]** — average over a dataset of triples: a prompt *x* paired with a winning (preferred) response *y_w* and a losing (dispreferred) response *y_l* — exactly what a human-preference dataset looks like.
- **π_θ(y_w|x), π_ref(y_w|x)** — the probability the current policy, and separately the frozen reference model, assigns to generating the whole winning response given the prompt (in practice, the sum of per-token log-probabilities across the response).
- **log(π_θ(y_w|x) / π_ref(y_w|x))** — the log-ratio of how much more (or less) likely the current policy is to produce the winning response compared to the reference model — DPO's implicit stand-in for "reward." It never trains an explicit reward model; it uses this ratio directly. The same ratio is computed for the losing response y_l.
- **β** — scales both log-ratios. Plays the same conceptual role as the KL coefficient in PPO: controls how sharply the model separates winners from losers relative to the reference. Higher β means more aggressive updates away from the reference.
- **the subtraction** — `β·log(π_θ(y_w)/π_ref(y_w)) − β·log(π_θ(y_l)/π_ref(y_l))` — the core comparison: how much more has the policy increased its relative preference for the winner versus the loser, compared to where the reference model started. Large and positive means the policy has learned to favor y_w over y_l.
- **σ( … )** — the sigmoid function, squashing that difference into a probability between 0 and 1 — "the probability the model now correctly prefers the winner," analogous to a binary classifier's output.
- **log σ( … )** — log of that probability — the standard trick for turning a probability into something smooth to run gradient descent on (same idea as binary cross-entropy).
- **leading −** — DPO is written as a loss to *minimize* (unlike PPO's objective, which is maximized), so the whole expression is negated: maximizing log σ(…) is the same as minimizing −log σ(…).

**In one line:** DPO never scores a response in isolation — it only asks whether the policy moved its relative probability mass toward the preferred response *more* than the frozen reference model already had. That relative-to-reference framing is what implicitly encodes the KL constraint, without ever computing a KL term explicitly.

### GRPO group-relative advantage

- **x** — the prompt. GRPO samples multiple responses to this *same* prompt in one step.
- **G** — the group size: how many independent responses y_1, …, y_G are sampled from the current policy for this one prompt (commonly 4–64).
- **r_i** — the scalar reward assigned to response *i* by the reward function — e.g., "1 if the math answer is correct else 0," "unit tests passed," or a learned reward model's score.
- **mean(r_1, …, r_G)** — the average reward across all G sampled responses to this prompt. This is GRPO's substitute for a critic's value prediction — instead of a network estimating "what reward should I expect here," the group's own empirical average serves that role.
- **std(r_1, …, r_G)** — the standard deviation of rewards across the group, used to normalize scale, so advantages stay roughly unit-scale regardless of whether the reward function outputs {0, 1} or {−50, 50}.
- **A_i** — the resulting group-relative advantage for response *i*: how many standard deviations above or below the group's average it scored. Exactly at the average → A_i = 0 (no update signal). Above average → positive A_i, tokens reinforced. Below average → pushed down.

### GRPO objective

- **(1/G) Σ_i** — average over all G responses in the group.
- **(1/|y_i|) Σ_t** — for each response *i*, average over all tokens *t* in that response (`|y_i|` is its length) — this normalizes so longer responses don't dominate the gradient just by having more tokens.
- **ρ_i,t(θ)** — the same kind of probability ratio as PPO's r_t(θ), indexed per-response-per-token:

  ```
  ρ_i,t(θ) = π_θ(y_i,t | x, y_i,<t) / π_θ_old(y_i,t | x, y_i,<t)
  ```

  `y_i,t` is the token at position t in response i; `y_i,<t` is all tokens before it in that same response (context so far). Same meaning as PPO: how much has the model's probability on this specific token shifted since sampling began.

- **min(…, clip(…))** — identical mechanism to PPO: clip the ratio to `[1−ε, 1+ε]` and take the smaller of the clipped/unclipped term, preventing any single token's update from moving the policy too far in one step.
- **A_i (reused per token)** — crucially, this is the *same* advantage value for every token in response i — it doesn't vary by t, unlike PPO where A_t can vary token-by-token from the critic. GRPO assigns one group-relative score per whole response and applies it uniformly across that response's tokens.
- **β · D_KL[π_θ ‖ π_ref]** — same role as in PPO: an explicit KL penalty against the frozen reference policy, keeping the model from drifting into reward-hacking territory. Unlike DPO (which folds the KL constraint implicitly into its loss), GRPO keeps it as an explicit separate term, just like PPO does.

### GRPO's per-token KL estimator

This isn't the textbook KL formula — it's a specific low-variance estimator (from Schulman's "Approximating KL Divergence" note) used instead of a naive Monte Carlo KL sample, because standard sampling-based KL estimates are noisy at the token level.

- **π_ref(y|x) / π_θ(y|x)** — the inverse probability ratio between the reference and current policy for the full response y.
- **− log(π_ref(y|x) / π_θ(y|x))** — the log of that same ratio, negated.
- **− 1** — a constant offset. Together, the ratio, its negated log, and the constant combine so the estimator is always non-negative and has zero expected gradient bias — a more stable training signal than a plain single-sample KL estimate.

---

## Worked example: GRPO by the numbers

Prompt: *"What is 17 × 24?"* — a case where correctness is programmatically checkable, so the reward function is just an answer-checker. Sample `G = 4` responses from the current policy:

| i | Response | Correct? | r_i |
|---|---|---|---|
| 1 | "408" | Yes | 1 |
| 2 | "398" | No | 0 |
| 3 | "17×24 = 17×20 + 17×4 = 340+68 = 408" | Yes | 1 |
| 4 | "17 × 24 = 391" | No | 0 |

Group statistics: `mean(r) = (1+0+1+0)/4 = 0.5`, and `std(r) = 0.5` (each value is exactly 0.5 away from the mean). Now compute each advantage:

```
A_1 = (1 - 0.5) / 0.5 = +1.0      A_2 = (0 - 0.5) / 0.5 = -1.0
A_3 = (1 - 0.5) / 0.5 = +1.0      A_4 = (0 - 0.5) / 0.5 = -1.0
```

Both correct responses (1 and 3) get reinforced equally with `A = +1.0`, regardless of the fact that response 3 spelled out its reasoning and response 1 didn't — reward and advantage here only look at the final answer. Both incorrect responses get suppressed with `A = -1.0`. This is the entire "how good was this?" signal for the update — no critic forward pass involved.

**Degenerate case.** Now suppose all four sampled responses had been correct: `rewards = [1, 1, 1, 1]`. Then `mean = 1` and `std = 0` — the denominator vanishes. Real implementations add a small epsilon (e.g., `std + 1e-4`) to avoid a literal division by zero, but the effect is the same either way: every `A_i ≈ 0`, so this prompt contributes essentially no learning signal despite having spent G full rollouts generating it. The same collapse happens if all four are wrong. This is why prompt difficulty matters for GRPO — a training set with too many trivially-easy or hopelessly-hard prompts wastes a large fraction of its sampling budget.

---

## Explanation for each

### PPO — "don't overcorrect on a single grade"

Imagine a teacher who, after each essay, tells you not just "good" or "bad" but a *precise numeric score* from a rubric (this is the reward model), and also compares that score to what an "average" essay at this point in your ability would score (this is the critic/value model, predicting expected reward). PPO nudges you to write more like the essays that scored above that expectation and less like the ones below it — but it clips how big a step you're allowed to take in any direction per update, so one unusually high or low score doesn't cause you to wildly overhaul your writing style. The KL penalty is a second leash: it keeps your writing from drifting so far from your original voice (the reference/SFT policy) that you start gaming the rubric in ways that don't reflect genuine quality.

Formally, PPO needs a **critic** — a learned model that estimates the *expected* future reward from any partial state, so it can compute an advantage (how much better this action was than expected). Training this critic accurately, especially for long sequences with sparse terminal rewards (like "the final answer was right"), is one of the hardest and most unstable parts of RLHF.

### DPO — "skip the rubric, just tell me which essay you liked better"

DPO's mental model: instead of grading essays on an absolute numeric scale and training a policy to chase that score, just show the model *pairs* of essays and say "this one was better." DPO shows mathematically that if your preference data follows the standard Bradley-Terry preference model, the optimal RLHF policy has a closed form — meaning you can skip training a reward model and skip the RL sampling loop entirely, and just directly increase the log-probability gap between preferred and dispreferred responses (relative to a frozen reference model). It turns RLHF into something that looks and trains like ordinary supervised learning, which is a huge simplification.

The tradeoff: DPO only works when you have pairwise preference data, and it's less natural for tasks like math/code where the signal is a clean scalar (correct/incorrect, passed/failed tests) rather than "A vs B, which is better."

### GRPO — "grade a group of essays relative to each other"

This is the analogy the paper itself leans on. Instead of an absolute rubric score *and* a separate model predicting what score you should expect (PPO's critic), the teacher has each student write G different attempts at the same essay prompt, grades all G with the reward function, and then simply says "you did better than your own average attempt" or "worse than your own average attempt" for this specific batch. The group's own mean and standard deviation become the baseline — no separate critic network is needed to estimate "expected reward," because the group itself provides that estimate empirically.

This is the crux of why GRPO removes the critic: in PPO, the critic's whole job is to answer "was this response better or worse than expected, given this prompt?" GRPO answers that question directly and cheaply by sampling multiple responses to the *same* prompt and comparing them to each other — the group mean acts as the value baseline. The **tradeoff** is that this requires generating many samples (typically 4–64) per prompt at each training step, which is more inference compute per update, and the advantage estimate is noisier/coarser than a well-trained critic's smooth estimate — especially with small group sizes. You're trading a persistent, learned model for a per-step, empirical, sampling-based estimate.

---

## GRPO in practice: pitfalls & follow-ups

GRPO's simplicity comes with sharp edges that later work has tried to file down:

- **Length bias from the per-response normalization.** The `1/|y_i|` term averages the loss over each response's token count. Combined with std-normalized advantages, this interacts with response length in ways that can implicitly reward being longer or shorter independent of actual quality. Follow-up work — DAPO (ByteDance) and Dr. GRPO (Sea AI Lab) — identified this and proposed removing or adjusting the length normalization.
- **Degenerate groups.** As shown in the worked example above, when every sampled response in a group gets the same reward, the advantage signal collapses to ~0 and that prompt's rollouts are wasted. This gets worse for training sets skewed toward trivially easy or hopelessly hard prompts — difficulty calibration/filtering of the prompt set is a real practical lever.
- **Reward hacking is still possible without a critic.** Removing the critic doesn't remove the risk of an exploitable reward function. A reward that only checks "does a number matching the answer appear anywhere in the output" can be gamed by a model that pads its response with several candidate numbers rather than genuinely solving the problem.

---

## Python example

This is deliberately minimal pseudocode showing the core GRPO loop — sampling a group, scoring, normalizing advantages, and applying a clipped + KL-penalized update. In practice you'd use TRL's `GRPOTrainer` or Unsloth's GRPO integration rather than hand-rolling this.

```python
import torch
import torch.nn.functional as F

def grpo_step(policy, ref_policy, tokenizer, prompt, reward_fn,
              group_size=8, eps=0.2, beta=0.04):
    # 1. Sample a group of G responses for the same prompt
    prompts = [prompt] * group_size
    responses, old_logps = policy.generate_with_logprobs(prompts)

    # 2. Score each response with the reward function
    rewards = torch.tensor([reward_fn(prompt, r) for r in responses])

    # 3. Group-relative advantage: normalize within the group
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # 4. Recompute log-probs under current policy (post prior updates)
    new_logps = policy.logprobs(prompts, responses)
    ref_logps = ref_policy.logprobs(prompts, responses)

    # 5. PPO-style clipped objective, advantage broadcast per-token
    ratio = torch.exp(new_logps - old_logps)
    unclipped = ratio * advantages.unsqueeze(-1)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages.unsqueeze(-1)
    policy_loss = -torch.min(unclipped, clipped).mean()

    # 6. KL penalty against the frozen reference policy
    kl = torch.exp(ref_logps - new_logps) - (ref_logps - new_logps) - 1
    loss = policy_loss + beta * kl.mean()

    loss.backward()
    return loss.item()
```

Note the absence of any `value_head` or critic forward/backward pass — the only two models involved are the policy and the frozen reference. All of the "how good was this?" signal comes from `rewards.mean()` and `rewards.std()` computed across the sampled group.

---

## Hyperparameter cheat sheet

Rough starting ranges seen in GRPO recipes (DeepSeek-Math/R1-style, TRL, Unsloth) — always tune against your own reward scale and task:

| Hyperparameter | Symbol | Typical range | What it controls |
|---|---|---|---|
| Group size | `G` | 8 – 64 | Responses sampled per prompt. Larger G gives a less noisy advantage estimate but costs more inference compute per step. |
| Clip range | `ε` | 0.1 – 0.3 (commonly 0.2) | How far the probability ratio is allowed to move in one update before clipping kicks in. |
| KL coefficient | `β` | 0.001 – 0.04 | Strength of the anchor back to the reference policy. Higher = safer but slower to improve. |
| Learning rate | — | 1e-6 – 1e-5 | Typically an order of magnitude below SFT learning rates, since RL updates are more fragile. |
| Prompts per step | — | task-dependent | Distinct prompts per training step, each expanded into G rollouts (so total rollouts = prompts × G). |

---

## Pros and cons

### PPO

**Pros**
- Well-established, extensively studied, works for any reward signal (scalar reward model, rule-based reward, etc.)
- Clipping mechanism gives strong empirical stability for a first-principles policy-gradient method
- Doesn't require paired preference data — works with any per-response scalar reward

**Cons**
- Requires training and hosting a separate critic model (often same size as the policy) — roughly doubles memory/compute
- Critic training is notoriously unstable for long-horizon, sparse-reward text generation
- Many hyperparameters to tune (clip range, KL coefficient, GAE λ, critic learning rate)
- High implementation complexity relative to DPO

### DPO

**Pros**
- No reward model, no critic, no online sampling loop — trains like ordinary supervised fine-tuning
- Simple to implement, stable to train, cheap in compute and memory
- Very sample-efficient given a fixed preference dataset

**Cons**
- Needs pairwise preference data, which is awkward to produce for tasks with objective scalar correctness (math, code)
- No online exploration — it only reshapes probabilities of responses already in the dataset, so it can't discover genuinely new better completions the way sampling-based RL can
- Known to sometimes reduce output diversity and can overfit to superficial cues that correlate with "preferred" in the training pairs (a form of reward hacking without the RL loop)

### GRPO

**Pros**
- No critic model needed — significant memory and implementation savings vs. PPO
- Works naturally with rule-based/verifiable rewards (correct answer, passing tests) as well as learned reward models
- Online sampling means the model can discover better responses beyond what's in any fixed dataset, unlike DPO
- Empirically very effective for reasoning tasks (DeepSeek-R1's headline results)

**Cons**
- Requires generating G rollouts per prompt at every training step — significant inference compute overhead
- Advantage estimates from a small group are noisier than a well-trained critic's estimate; small group sizes can be high-variance
- If all G responses in a group get the same reward (all right or all wrong), the advantage signal collapses to ~0 and that batch teaches the model nothing — wasted compute
- Reward function design is critical and hard: a poorly specified or hackable reward function will be exploited regardless of algorithm elegance

---

## Summary

| Method | Objective type | Needs critic model? | Typical use case | Key tradeoff |
|---|---|---|---|---|
| **PPO** | Clipped policy-gradient RL vs. learned reward model, KL-constrained | Yes | General-purpose RLHF (instruction-following, chat alignment) | Strong but expensive — critic doubles memory and adds instability |
| **DPO** | Closed-form supervised loss over preference pairs | No | Offline alignment from preference datasets | Simple & cheap, but no online exploration, needs pairwise data |
| **GRPO** | Clipped policy-gradient RL vs. group-normalized reward, KL-constrained | No | Reasoning, math, code — verifiable/scalar rewards | Cheaper than PPO per-model, but pays in inference compute (many samples per prompt) |

---

## Key takeaways

- **GRPO trades a critic model for more samples per prompt.** Instead of learning to *predict* the expected reward, it *measures* it empirically by sampling a group of responses to the same prompt and normalizing against the group's own mean and standard deviation.
- **The KL penalty is what keeps the policy from reward-hacking.** Without it, the model can drift arbitrarily far from the reference policy to exploit quirks in the reward function; the KL term anchors it to what the SFT model already knows is reasonable.
- **Reward function design matters as much as the RL algorithm itself.** GRPO, PPO, and DPO all fail the same way if the reward signal (or preference data) is poorly specified — the algorithm optimizes exactly what you measure, including its flaws.
- **GRPO's group-relative advantage collapses to zero when all sampled responses get the same reward.** This makes prompt/task difficulty calibration and group size important practical knobs — too-easy or too-hard prompts waste rollouts.
- **DPO and GRPO solve different problems, not competing versions of the same one.** DPO is the right tool when you have (or can easily collect) pairwise preferences and want a cheap offline method; GRPO is the right tool when you have a scalar/verifiable reward and want online exploration.
- **The RL-post-training landscape is converging on "critic-free" methods** for good reason: at LLM scale, the memory and stability cost of a full second critic model is often the single biggest obstacle to running RL fine-tuning at all — which is why both DPO and GRPO, despite being very different algorithms, share the design goal of eliminating it.
