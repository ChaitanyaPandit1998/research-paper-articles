# Mixture of Experts: How Modern LLMs Get Bigger Without Getting Slower

**Pull quotes:**
- "A dense model asks every single neuron to weigh in on every single word. MoE asks: why not just call the specialist?"
- "MoE doesn't make a model smaller. It makes a model that only *uses* a small part of itself at a time."
- "DeepSeek didn't invent Mixture of Experts. It made the experts sharper."

---

Large language models have a simple problem: the more you want them to know, the more parameters you need, and the more parameters you have, the slower and more expensive every single word becomes to generate. Mixture of Experts (MoE) is the architecture that broke that trade-off. It's the reason models like DeepSeek-V3, Llama 4, Mixtral, and Qwen3 can carry hundreds of billions of parameters while running each token through only a small fraction of them.

This article covers what MoE is, why it exists, what it looks like inside a model, who's using it, and where the research is headed.

---

## 1. What Is Mixture of Experts?

Picture a hospital. When a patient walks in, you don't send them to every doctor in the building — a general practitioner looks at the case and refers them to the right specialist: a cardiologist, a dermatologist, whoever fits. The hospital as a whole knows a huge amount, but any single patient only interacts with one or two of its doctors.

MoE applies this idea to a neural network. In a normal ("dense") transformer, every token passes through one feed-forward network (FFN), and every parameter in that FFN does some work on every token. In an MoE layer, that single FFN is replaced by:

- A set of **experts** — many smaller FFNs (could be 8, could be 256), each structurally identical but with its own separately trained weights.
- A **router** (sometimes called a gate) — a small learned network that looks at each token and decides which expert(s) should handle it.

For each token, the router picks a small number of experts (commonly 1, 2, or 8 out of hundreds), sends the token only to those, and combines their outputs — usually weighted by how confident the router was in each pick.

The result: the model's **total parameter count** can be enormous (all the experts combined), but the **active parameter count** for any given token — and therefore the compute cost — stays small. This is usually described as the difference between a model's total size and its active size, e.g. "671B total, 37B active."

> **First-principles summary:** MoE trades a network that always does a medium amount of work for a network that almost always does a small amount of work, but occasionally has access to a huge amount of specialized knowledge.

---

## 2. Why Would You Want This? (Use Cases)

**Scaling knowledge without scaling cost.** The main reason labs adopt MoE is simple: you can grow a model's capacity (how much it can "know" or represent) without a proportional growth in the cost of running it. This is why frontier-scale models today routinely have total parameter counts in the hundreds of billions to low trillions, while only activating tens of billions per token — something that would be computationally unaffordable in a dense model of the same total size.

**Specialization across domains.** Because different experts can end up specializing — some picking up more math-heavy patterns, others more attuned to code, dialogue, or specific languages — MoE models can, in principle, handle a broader range of tasks well without one generalist FFN having to do everything adequately but nothing brilliantly.

**Cheaper training and serving at a given quality bar.** For a fixed compute budget, MoE models tend to reach a given quality level faster than dense models of equivalent active size, because the extra (inactive-per-token) parameters still contribute useful capacity during training even though they're cheap to run at inference.

**Multi-task and multi-modal systems.** Because routing can be learned per token (or even per modality), MoE is a natural fit for models that need to handle text, code, and other modalities without one shared block being a bottleneck.

---

## 3. What Problem Does This Solve? (Shortcomings of Dense FFNs)

To see why MoE exists, it helps to look at what it's replacing.

In a standard ("dense") transformer, every token that flows through a layer is processed by the **same** feed-forward network — the same weights, every time, for every token, regardless of whether the token is "the," "photosynthesis," or a line of Python. This has real costs:

**1. Compute scales linearly with parameters.** If you want a dense model to know more, you make the FFN bigger. But because *every* token uses the *entire* FFN, doubling the parameters roughly doubles the compute cost of every forward pass. There's no way to add capacity without also paying for it on every single token, useful or not.

**2. One-size-fits-all weights.** A dense FFN has to represent everything it has learned — grammar, code syntax, chemistry, poetry — inside one shared set of weights that treats every token the same way. It can't easily dedicate a chunk of itself to "handles chemistry vocabulary" and a different chunk to "handles Python syntax," because there is no mechanism to selectively activate different parts of the network for different inputs.

**3. Diminishing returns from raw scale.** Beyond a certain point, simply making a dense FFN bigger yields smaller and smaller quality improvements per unit of added compute — you are paying full price for capacity that any individual token barely uses.

MoE directly targets all three: it lets total capacity grow (fixing #2 by giving specialization somewhere to live) largely independent of per-token compute (fixing #1), and it lets you keep adding useful capacity — more experts — without every added expert taxing every token (fixing #3).

The catch, which is most of what MoE *research* is about, is that this only works if the router does its job well: sending tokens to the experts best suited for them, and spreading load evenly enough that experts don't sit idle or get overwhelmed.

---

## 4. Architecture Diagram

Here's what a single MoE layer looks like, replacing the FFN inside a transformer block:

```mermaid
flowchart TD
    A["Token embedding<br/>(output of attention)"] --> R["Router<br/>(small linear layer + softmax)"]
    R -->|routing scores| S{Select top-k experts}

    S -.not selected.-> E1["Expert 1 (FFN)"]
    S -->|weight 0.6| E2["Expert 2 (FFN)"]
    S -.not selected.-> E3["Expert 3 (FFN)"]
    S -.not selected.-> E4["Expert 4 (FFN)"]
    S -->|weight 0.3| E5["Expert 5 (FFN)"]

    A -.always active.-> SE["Shared Expert<br/>(FFN, optional)"]

    E2 --> C["Weighted sum"]
    E5 --> C
    SE --> C

    C --> O["Output token<br/>(passed to next layer)"]

    style E1 fill:#eee,stroke:#999,color:#999
    style E3 fill:#eee,stroke:#999,color:#999
    style E4 fill:#eee,stroke:#999,color:#999
```

*(Experts 1, 3, and 4 exist in the layer but weren't picked for this token — they simply don't run, so they cost nothing for this forward pass.)*

**Reading the diagram:**
1. A token's vector arrives at the MoE layer.
2. The **router** scores every expert for this token and picks the top-k (e.g. top-2 out of 128).
3. The token is sent *only* to those chosen experts — the rest do zero work for this token.
4. Each chosen expert's output is scaled by the router's confidence score and summed.
5. Some architectures (DeepSeekMoE, Llama 4, Qwen3) also add a **shared expert** that runs on every token unconditionally, capturing common knowledge so the routed experts don't have to re-learn it repeatedly.

---

## 5. Which LLMs Are Using MoE?

MoE went from a research curiosity to the default architecture for frontier-scale open models within a few years. As of mid-2026:

| Model | Total params | Active params | Experts |
|---|---|---|---|
| **DeepSeek-V3 / R1** | ~671B | ~37B | 256 routed + 1 shared |
| **DeepSeek-V4** (Pro / Flash) | 1.6T / 284B | 49B / 13B | fine-grained routed + shared |
| **Mixtral 8x22B** | 141B | 39B | 8 experts, top-2 |
| **Qwen3-235B-A22B** | 235B | ~22B | 128 experts, top-8 |
| **Llama 4 Scout / Maverick** | 109B / ~400B | 17B / 17B | 16 experts + shared expert |
| **Grok-1 / Grok-2** | 314B+ | ~70–80B | MoE (xAI) |
| **Mistral Large / Kimi K2 / GPT-OSS** | varies | varies | MoE |

A few things worth noting:
- The naming convention "A22B," "17B active," etc. refers to **active** parameters per token — the number that actually determines inference cost, not the flashier total parameter count.
- The trend across 2025–2026 has been toward **more, smaller experts** (fine-grained routing — e.g. 128–256 experts) rather than fewer, larger ones (Mixtral's original 8), because fine-grained experts specialize more precisely and reduce redundant knowledge across experts.
- Not every major lab has publicly confirmed using MoE for their flagship models — architecture details for some closed models (e.g. Anthropic's Claude family) haven't been disclosed, so they're excluded from confident claims here.

---

## 6. Current Research Directions

MoE looks simple in a diagram, but making the router behave well in practice is an active, ongoing research problem. A few threads as of 2026:

**Load balancing without hurting quality.** Early MoE models added an extra "auxiliary loss" term during training to stop the router from dumping all tokens onto a handful of favorite experts (a failure mode called "routing collapse"). The problem: that auxiliary loss creates its own gradient pressure that can fight against the model's actual objective. DeepSeek-V3 popularized an **auxiliary-loss-free** approach instead — adjusting each expert's routing bias directly based on how overloaded or underused it's been, without touching the training loss at all.

**Finer-grained expert segmentation.** DeepSeekMoE's core contribution (building on Shazeer et al.'s original sparsely-gated MoE and Google's GShard) was splitting experts into many smaller, more specialized units, plus adding always-on **shared experts** to absorb common knowledge — so routed experts don't waste capacity re-learning things every expert needs anyway. This is now close to standard practice.

**Better specialization signals.** Auxiliary-loss balancing tends to push routing toward being *too* uniform, which can cause experts to overlap in what they learn rather than truly specializing. Newer proposals add extra objectives — like orthogonality losses (push experts to be different from each other) or variance losses (push routing decisions to be more decisive) — to encourage sharper specialization without sacrificing balance.

**Rethinking who chooses.** Most MoE designs let the router pick the experts (token-choice routing). Some recent work — like Autonomy-of-Experts — flips this: experts themselves signal how well they'd handle a given token, and routing follows that signal, aiming to avoid routers making picks disconnected from what experts can actually do well.

**MoE beyond text.** Researchers are extending sparse expert routing into multimodal models — routing not just by token content but by modality or task, so a single model can allocate distinct expert capacity to, say, vision versus language versus audio.

---

## 7. Further Reading

**Foundational papers:**
- Shazeer et al., 2017 — [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — the original sparsely-gated MoE.
- Lepikhin et al., 2020 — [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — scaling MoE across many devices.
- Fedus et al., 2021 — [Switch Transformer](https://arxiv.org/abs/2101.03961) — simplified top-1 routing at scale.

**DeepSeek's contributions:**
- Dai et al., 2024 — [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — fine-grained expert segmentation + shared experts.
- DeepSeek-AI, 2024 — [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — auxiliary-loss-free load balancing at production scale.
- Wang et al., 2024 — [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664).

**Newer research directions:**
- [Autonomy-of-Experts Models](https://arxiv.org/abs/2501.13074) — expert-driven routing.
- Jiang et al., 2024 — [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — the Mixtral 8x7B/8x22B architecture.

**In this repo:**
- [`transformer/llama4/MoE.md`](../transformer/llama4/MoE.md) — code-level walkthrough of Llama 4's MoE implementation.
- [`transformer/llama4/DenseVsSparseMoEDispatch.md`](../transformer/llama4/DenseVsSparseMoEDispatch.md) — dispatch mechanics for sparse vs. dense MoE.
