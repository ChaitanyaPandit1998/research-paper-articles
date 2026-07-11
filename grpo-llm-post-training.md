# GRPO: How DeepSeek Trains Reasoning Models Without a Critic

### A deep dive into Group Relative Policy Optimization for LLM post-training

---

## Table of Contents

1. [What is it?](#what-is-it)
2. [Use case](#use-case)
3. [Different RL post-training approaches actively used](#different-rl-post-training-approaches-actively-used)
4. [Formula/objective for each](#formulaobjective-for-each)
5. [Explanation for each](#explanation-for-each)
6. [Python example](#python-example)
7. [Pros and cons](#pros-and-cons)
8. [Summary](#summary)
9. [Key takeaways](#key-takeaways)

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
