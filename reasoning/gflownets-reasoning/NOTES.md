# Building a Reasoning Machine — Edward Hu's PhD Thesis

**Source:** https://edwardjhu.com/thesis/
**Author:** Edward Hu
**Date:** April 7, 2026
**Committee:** Yoshua Bengio, Aaron Courville, Jason Eisner, Dhanya Sridhar

---

## The Central Question

Why are modern AI systems — despite being superhuman at chess, image recognition, and language — still surprisingly bad at deliberate, multi-step reasoning? Tasks like solving a hard math proof, debugging a complex system, or planning a coherent strategy across many steps remain difficult.

This thesis argues the answer is structural: current ML systems search in the **wrong space** and learn from the **wrong data**. The thesis proposes GFlowNets as a principled fix — a framework that lets neural networks search in the kind of discrete, structured spaces where real reasoning happens.

---

## The Three Gaps Between Human and Machine Intelligence

**1. When compute is spent**
Humans spend most of their cognitive effort *at inference time* — deliberating before answering. ML systems spend 99% of compute *at training time* — learning weights — and very little thinking at inference time. A human solving a hard puzzle pauses, backtracks, tries alternatives. A neural network runs a fixed forward pass.

**2. What space is searched**
Humans consciously search structured, low-dimensional spaces: the next move in a chess game (discrete, bounded), the next step in a proof (symbolic, compositional). Neural networks search high-dimensional, continuous weight spaces via gradient descent. These spaces are fundamentally different in shape, size, and navigability.

**3. How composition is handled**
Humans naturally use structured representations — decision trees, causal chains, knowledge graphs — to break complex problems into parts. Deep learning has no principled general mechanism for this compositional structure.

---

## A Brief History of AI as Search

The thesis frames all of AI history as increasingly efficient ways of converting compute into intelligence through search.

### GOFAI — Good Old-Fashioned AI
Search directly in the solution space: enumerate chess moves, enumerate proof steps, use algorithms like MCMC (Markov Chain Monte Carlo) to explore possibilities.

**Strength:** Operates exactly where human reasoning lives — discrete, structured spaces.
**Weakness:** No learning between problems. Solves each problem from scratch. Terrible at scale. Can't generalise. Imagine re-deriving every chess opening from first principles every game.

### Deep Learning
Search in neural network *parameter* space via gradient descent. Instead of searching for the answer directly, learn the mapping from inputs to answers by optimising millions of weights.

**Strength:** Scales incredibly well. Gradient descent + inductive biases make this tractable. Excellent at discriminative tasks (classification) and generative tasks (language, images).
**Weakness:** Only searches parameter space. Doesn't do conscious, deliberate search over structured solutions at inference time. The network sees the input and immediately produces an output — no deliberation, no backtracking.

### The Missing Middle
GOFAI: explicit search in the right space, no learning.
Deep Learning: learning in the wrong space, no deliberate search.

What's needed: **learned, amortized search in discrete, structured spaces** — combining the best of both.

---

## System 1 vs System 2 — Reframed

The classic distinction:
- **System 1:** Fast, automatic, intuitive. "What is 2+2?" — answer is immediate.
- **System 2:** Slow, deliberate, effortful. "What is 17 × 24?" — requires conscious steps.

The common assumption is that these require fundamentally different architectures. The thesis disagrees.

**Hu's claim:** System 2 is not a separate module. It is System 1 performing search over a discrete, low-dimensional latent space.

When you solve 17 × 24 deliberately, your fast pattern-matching system (System 1) is running repeatedly — but over a *sequence of intermediate representations* (17×20=340, 17×4=68, 340+68=408). Each step is fast and intuitive; the deliberation comes from the search structure connecting those steps.

The implication: you don't need a new architecture for reasoning. You need a way to make your existing fast system search through the right compositional space.

---

## GFlowNets — The Bridge

GFlowNets (Generative Flow Networks) are the thesis's core technical contribution. They are a framework for training neural networks to **sample from distributions over compositional objects**.

### What That Means

Given a reward function `R(x)` that says how good a structured object `x` is (a molecule, a proof step, a causal graph, a reasoning chain), GFlowNets learn a policy that generates samples proportional to that reward. Higher reward objects get sampled more often; lower reward objects less often.

Crucially, `x` is **compositional** — built up piece by piece through a sequence of decisions. A molecule is built atom by atom. A proof is built step by step. A reasoning chain is built thought by thought.

### How It's Different From Alternatives

| Approach | What it does | Problem |
|---|---|---|
| GOFAI search (MCMC) | Explore the space directly | Slow, no amortization, starts from scratch each time |
| RL (e.g. policy gradient) | Learn to find the *best* object | Collapses to one mode — finds one good answer and stops exploring |
| GFlowNets | Learn to *sample* proportionally to reward | Maintains diversity, covers all good solutions, amortized |

**The diversity point is critical.** RL finds the peak of the reward mountain and stays there. GFlowNets explore the entire mountain, returning samples from every high-reward region. For reasoning, you often want many diverse plausible hypotheses — not just the single most confident one. Bayesian reasoning demands this.

### The Amortization Benefit

Like deep learning vs GOFAI, GFlowNets learn from experience. After training, generating a sample is cheap — a single forward pass through a neural network. The search happens implicitly via learned parameters, not explicitly step by step. This is *amortized inference*: pay the cost once at training, benefit at every inference.

---

## Knowledge vs Inference Separation

A subtle but important design principle in the thesis.

Deep learning conflates two things:
- **Knowledge:** What the model knows about the world (weights, parameters, inductive biases)
- **Inference:** How the model uses that knowledge to solve a specific problem

These two have **conflicting requirements**:

| | Knowledge | Inference |
|---|---|---|
| Should be | Compact (Occam's razor) | Rich, high-capacity |
| Should have | Strong inductive biases for sample efficiency | Sufficient capacity to handle complex structures |
| Goal | Generalise across problems | Solve this specific problem deeply |

When you mix them into one neural network, you get a compromise that's mediocre at both. The model is too large to be truly sample-efficient; too rigid to reason flexibly.

The thesis (particularly the GFlowNet-LLM chapter) explores separating these: a compact knowledge model paired with a separate, powerful inference mechanism. This is analogous to how a doctor has both *medical knowledge* (compact, generalisable) and a *diagnostic reasoning process* (flexible, problem-specific) — and the two are not the same thing.

---

## The Data Perspective

### Why Scaling Alone Isn't Enough

Scaling laws suggest more data + more compute = better models. But this only holds if the data covers the full distribution of intelligence we want. The internet — the primary training source for LLMs — is biased:

- Overrepresents easy, surface-level reasoning
- Underrepresents deep, multi-step deliberate thought processes
- Has almost no examples of "showing your complete work" — the internal deliberation that leads to answers

If you train on the internet, you learn the internet's distribution — which is a biased slice of intelligence, not the full spectrum.

### The Fix: Strategic Data Augmentation

Rather than just collecting more data, the thesis proposes generating *better* data through:

**Inductive biases as infinite augmentation**
Convolutional layers don't just learn — they embed spatial invariance as a permanent inductive bias, equivalent to augmenting training data with all spatial translations infinitely. The right architectural choices can have the same effect on reasoning data.

**Self-play**
AlphaZero didn't need human game records — it played itself and generated high-quality training data. The thesis applies this to reasoning: generate reasoning chains through self-play with a reward model, creating a data augmentation loop that improves reasoning quality without human annotation.

**The GFlowNet-LLM connection**
GFlowNet-LLM uses inference-time compute to sample chain-of-thought reasoning traces from Bayesian posteriors — treating inference as self-play with an unnormalized reward model. The generated reasoning chains become training data that improves future reasoning. A virtuous cycle.

---

## The Story: The Detective Agency

*A story that covers all the key ideas.*

### Chapter 1 — The Filing Cabinet (GOFAI)

The old detective agency solved crimes the old-fashioned way. Every case started fresh. Pull out the filing cabinet, flip through every folder, read every document, check every suspect against every clue. Systematic, exhaustive, thorough.

For simple cases — three suspects, one crime scene — this worked fine. But as cases grew complex — a thousand suspects, overlapping alibis, international connections — the detectives spent months in the filing room before reaching any conclusion. Each case took exactly as long to start, no matter how many similar cases they'd solved before. Nothing learned. Nothing reused. Just the filing cabinet, again and again.

This was GOFAI: explicit search in the right space, but no learning and no scale.

---

### Chapter 2 — The Genius Intern (Deep Learning)

Then a new kind of detective arrived — the Genius Intern. She hadn't been trained by flipping through filing cabinets. She'd been trained by reading ten million crime novels, news articles, court transcripts, and police reports. Pattern after pattern after pattern.

Show her a crime scene photo and she'd immediately — without hesitation, without deliberation — say "This was an inside job. The perpetrator was left-handed. Check the business partner." And she was right, astonishingly often.

But ask her *how* she knew, and she couldn't tell you. Ask her to solve a crime that required careful step-by-step deduction across twenty pieces of evidence — where each conclusion unlocked the next — and she'd struggle. She'd give you her best guess immediately. No backtracking. No deliberation. No "wait, if that's true, then this can't be, so let me revise."

She searched inside her own head — the parameter space of her learned patterns. Brilliant for pattern matching. Limited for deliberate reasoning chains.

This is deep learning: amortized, fast, scalable — but locked inside parameter space.

---

### Chapter 3 — The Problem With Brilliant Guesses

The agency noticed a gap. The Genius Intern was great at recognising patterns, but some cases required something different: generating *multiple competing theories* and reasoning through all of them simultaneously.

Standard detective work (and standard RL) would commit to the most likely theory and pursue it. If wrong, start over. Winner-takes-all.

But a great detective holds many hypotheses at once — weighted by evidence — and updates all of them as new facts arrive. This is Bayesian reasoning. And neither the filing cabinet approach nor the Genius Intern's pattern matching quite captured it.

---

### Chapter 4 — The Hypothesis Machine (GFlowNets)

The thesis introduces a new kind of detective tool: the Hypothesis Machine.

Tell the Hypothesis Machine what a good theory looks like — the reward function. Give it a crime, a pile of evidence, and a definition of "a satisfying explanation." It will generate theories for you — not one, but many, proportional to how good each theory is.

The best theories come out most often. Decent theories come out sometimes. Bad theories rarely appear. The distribution of output theories matches the distribution of reward.

And crucially: the Hypothesis Machine *learns*. After solving a thousand cases, it gets faster and smarter at generating good hypotheses for new cases. The search is amortized. The first case took all night; after training, a new similar case takes minutes.

This is GFlowNets: learned, amortized sampling over compositional structures — building theories piece by piece (clue by clue, step by step) proportional to how well each finished theory explains the evidence.

---

### Chapter 5 — The Casebook and the Reasoning (Knowledge vs Inference)

A senior detective walks into the agency and observes something strange. The Genius Intern has everything in one head — her medical knowledge, her reasoning process, her biases, her instincts, all tangled together. When she gets something wrong, it's impossible to know whether she lacks knowledge or is reasoning poorly from the knowledge she has.

The senior detective separates them. The **Casebook** holds everything known about criminal patterns, forensic science, human psychology — compact, well-structured, general. The **Reasoning Process** is the live investigation — flexible, case-specific, high-capacity.

Now when the Intern gets something wrong, you can check: is it in the Casebook (knowledge gap) or in the Reasoning (logic error)? And you can improve each independently.

This is the knowledge-inference separation: compact knowledge model + powerful separate inference mechanism. Conflating them creates a mediocre compromise. Separating them lets each excel at what it's designed for.

---

### Chapter 6 — The Biased Library (The Data Problem)

The agency's training library — ten million crime novels — turns out to have a problem. It was written mostly by authors who prefer tidy, fast resolutions. The novels rarely show the detective sitting in confusion for three hours before a breakthrough. They rarely show false starts, dead ends, complete theory reversals.

The Genius Intern learned from this library. So she's great at the tidy patterns but struggles with the messy, non-linear reasoning that real hard cases require.

You could find more books. But if all books come from the same biased library, more books doesn't fix the bias — it just reinforces it.

---

### Chapter 7 — Playing Cases Against Yourself (Self-Play and Data Augmentation)

The solution: stop waiting for real cases and start *generating* training cases.

The agency sets up a simulation room. Two detective interns play against each other — one generates theories, the other tries to poke holes. Back and forth, endlessly. Each round produces a record of reasoning: how a good theory was built, how it was challenged, how it was revised.

This is self-play. AlphaZero did it for chess — no human game records needed, just two versions of itself playing millions of games. The thesis does it for reasoning: generate chain-of-thought reasoning traces by having the model play against itself, using a reward model as a judge.

The generated reasoning chains are *better* than anything in the original biased library. They show genuine deliberation — backtracking, hypothesis revision, multi-step deduction — because the self-play process demands it.

More importantly: this loop is self-improving. Better reasoning generates better training data. Better training data produces better reasoning. Compute converts directly into reasoning quality, at inference time, not just training time.

---

### The Moral

The thesis's answer to "why can't machines reason well" is not "they need more parameters" or "they need more internet data."

The answer is:
1. They're searching in **the wrong space** — parameter space instead of discrete structured space
2. They're not **amortizing inference** over compositional objects — starting from scratch each time instead of learning from experience
3. They're conflating **knowledge and inference** — making both worse
4. They're training on **biased data** — internet text that underrepresents genuine deliberate thought

GFlowNets address point 1 and 2. Knowledge-inference separation addresses point 3. Self-play data augmentation addresses point 4.

Together: a framework for building machines that don't just pattern-match, but genuinely reason.

---

## Chapter Map

| Chapter | Perspective | Contribution |
|---|---|---|
| GFlowNet-EM | Search + Data | Builds discrete latent spaces from data; amortized inference over compositional structures |
| GFlowNet-LLM | Search + Data | Uses inference-time compute to sample chain-of-thought from Bayesian posteriors; self-play analogue |
| Blackboard | Search | Embeds symbolic objects in vector space; gradient-guided search for symbolic manipulation; strong tree-operation biases (`car`, `cdr`, `cons`); exceptional compositional generalisation |
| GFlowNet-CD | Data | Uses causal discovery action/state constraints as inductive biases for sample efficiency |

---

## Key Papers

- *Amortizing Intractable Inference in Large Language Models*
- *GFlowNet-EM for Learning Compositional Latent Variable Models*
- *Differentiable Tree Operations Promote Compositional Generalization*
- *GFlowNets for Causal Discovery: an Overview*
- *GFlowNets and Variational Inference*
- *GFlowNet Foundations*

---

## Relation to Other Concepts

- **Chain-of-thought prompting** — a heuristic approximation of System 2 reasoning; GFlowNets provide the principled underpinning for *why* and *how* to generate good chains
- **RLHF / RLAIF** — RL-based fine-tuning collapses to one mode; GFlowNets maintain diversity, making them more suitable for Bayesian posterior sampling
- **Mixture of Experts** — parallel to knowledge-inference separation at the architecture level; each expert is a knowledge component, the router is an inference mechanism
- **AlphaZero / self-play** — direct inspiration for the data augmentation perspective; the thesis generalises self-play from games to open-ended reasoning tasks
- **Variational Inference** — GFlowNets can be understood as a generalisation of VI to non-differentiable, compositional structured spaces
