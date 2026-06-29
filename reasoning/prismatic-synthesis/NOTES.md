# Prismatic Synthesis: Gradient-based Data Diversification for LLM Reasoning

**Paper:** https://arxiv.org/abs/2505.20161
**Authors:** Jaehun Jung, Seungju Han, Ximing Lu, Skyler Hallinan, David Acuna, Shrimai Prabhumoye, Mostafa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi
**Affiliation:** NVIDIA + University of Washington
**Venue:** NeurIPS 2025
**Dataset:** https://huggingface.co/datasets/nvidia/Nemotron-PrismMath

---

## The Central Question

When generating synthetic training data for LLMs, more data is assumed to be better. But this paper asks a sharper question: **what kind of diversity in training data actually makes a model generalise to new, unseen problems?**

The answer is not semantic diversity (different topics), not lexical diversity (different words), not perplexity diversity (surprising text). The answer is **gradient diversity** — training samples that push the model's weights in genuinely different directions. And the paper introduces both a way to measure it (G-Vendi) and a way to generate it (Prismatic Synthesis).

---

## The Problem: Diversity Metrics That Don't Predict Generalization

Standard approaches to measuring training data diversity include:
- **N-gram entropy** — count how many unique word sequences appear
- **Embedding dissimilarity** — measure how far apart sentence embeddings are
- **Perplexity** — how surprising is the text to a base model?
- **Skill set entropy** — how many different topics/skills does the data cover?

All of these measure surface-level diversity — how different the texts *look*. None of them reliably predict whether training on the data will generalise to unseen benchmarks.

The core problem: two math problems can use completely different words, cover different topics, and have high embedding dissimilarity — yet teach the model exactly the same reasoning step. Surface diversity ≠ learning diversity.

**Measured result:** All of the above metrics show weak or no correlation with out-of-distribution (OOD) test performance. G-Vendi achieves Spearman's ρ ≈ 0.9.

---

## G-Vendi: Measuring Diversity Through Gradients

### The Intuition

When you train on a data sample, the model's weights shift in some direction — the gradient direction. If all your training samples shift the weights in roughly the same direction, the model is essentially learning one thing over and over. If samples shift the weights in many different directions, the model is learning many genuinely different things.

**G-Vendi measures how spread out the gradient directions are** across a dataset. High G-Vendi = diverse learning signals. Low G-Vendi = redundant, same-lesson data.

### The Three Steps

**Step 1 — Collect gradients**
For each training sample `(x, y)`, compute the loss gradient with respect to model parameters using a small proxy model:
```
g(x, y) = -∇ log P(y|x; θ) / ||∇ log P(y|x; θ)||
```
The gradient tells you: "if the model trained on this one example, which direction would its weights move?" Normalising by magnitude ensures the direction is what matters, not the scale.

A small off-the-shelf instruction-tuned proxy model is used — not the student model being trained. This is key: you don't need to train anything to measure G-Vendi.

**Step 2 — Reduce dimensionality**
Gradients are high-dimensional (one value per model parameter — millions of dimensions). Random projection (Rademacher projections) compresses them into a much smaller space while **preserving the dot products** between gradient vectors. The geometry is maintained; the compute is tractable.

**Step 3 — Compute entropy of the gradient covariance matrix**
Build a covariance matrix from all gradient vectors. Compute its eigenvalues. High diversity = eigenvalues spread across many dimensions (many distinct learning directions). Low diversity = one or a few dominant eigenvalues (all gradients bunched in the same direction).

G-Vendi = exponentiated entropy of the normalised eigenvalue distribution. Intuitively: how many effectively independent learning directions are represented in this dataset?

**Computational complexity:** O(d²|D|), tractable for datasets of millions of samples.

---

## The Saturation Problem with Vanilla Data Generation

When you generate synthetic training data naively (prompt a model, generate problems, add them all), performance improves quickly at first and then **plateaus around 50,000–100,000 samples**.

Why? Because naive generation floods the dataset with variations of the same reasoning patterns. The first 50K problems cover the common, easy-to-generate problem types. The next 50K cover the same types again, slightly rephrased. In gradient space, all these new samples land in the same already-crowded clusters — they're redundant in terms of what they teach the model.

**Beyond 100K samples, vanilla and persona-guided approaches stop improving.** Prismatic Synthesis keeps improving with scale because it systematically avoids redundant regions.

---

## Prismatic Synthesis: Filling the Uncharted Regions

### Core Idea

Instead of generating data randomly and keeping everything, Prismatic Synthesis specifically targets the **underrepresented regions of gradient space** — the reasoning patterns the current dataset barely covers.

It's iterative: each round adds data that fills the gaps left by all previous rounds.

### The Three-Step Loop

```
Repeat until desired dataset size:
  1. CLUSTER  — project all current samples into gradient space, cluster them
  2. GENERATE — prompt an LLM to create new synthetic samples
  3. FILTER   — keep only new samples whose gradients land in sparse clusters
               (e.g., the least-populated 20% of clusters)
               discard samples that land in already-dense clusters
```

**Step 1: Cluster in gradient space**
Map every existing training sample to its gradient vector. Run clustering (e.g. k-means) on the projected gradient space. Each cluster represents a "type of learning" — a region of the model's parameter space that certain kinds of problems update.

Dense cluster = this type of reasoning is well-represented in the training data.
Sparse cluster = this type of reasoning is barely covered.

**Step 2: Generate new samples**
Prompt the generator LLM to produce new problems. This can be seeded from existing samples in sparse clusters to nudge generation toward underrepresented patterns.

**Step 3: Filter to sparse clusters only**
For each newly generated sample, compute its gradient vector (using the proxy model) and check which cluster it lands in. Keep it only if it falls in a sparse cluster. Discard it if it falls in an already-dense cluster.

This greedy selection continuously pushes the frontier of gradient space outward, ensuring no region is flooded while others remain empty.

### Why "Prismatic"

A prism splits white light into its component wavelengths — revealing the full spectrum. Prismatic Synthesis splits the training data into its component learning signals and ensures the full spectrum is represented, not just the brightest (most common) frequencies.

---

## Key Results

### PrismMath-7B

**Dataset:** 1 million math problem-solution pairs generated via Prismatic Synthesis using a **32B parameter model** as the generator.

**Student model trained:** Qwen2.5-7B (7 billion parameters)

| Benchmark | PrismMath-7B | R1-Distill-Qwen-7B |
|---|---|---|
| AIME24 | **57.08%** | 54.66% |
| AIME25 | **38.33%** | 33.33% |
| AMC23 | **93.75%** | lower |
| Average (7 benchmarks) | **75.25** | lower on 6 of 7 |

R1-Distill-Qwen-7B was trained on data from **DeepSeek-R1 (671B parameters)**. PrismMath-7B uses data from a **32B model** — 20× smaller. Yet it outperforms on 6 of 7 benchmarks.

**The reason:** The 671B model's data, while higher quality per problem, covers redundant gradient space. The 32B model's data, selected for gradient diversity via Prismatic Synthesis, covers a wider range of reasoning directions.

### PrismNLI

**Dataset:** 515K natural language inference samples generated via Prismatic Synthesis.

- Outperforms prior data mixtures by **8% on OOD benchmarks**
- HANS: 92.44%, ANLI-r3: 57.00%, Diagnostics: 86.13%
- No human annotation required

### The Diversity vs. Scale Finding

Training on a **smaller, high-G-Vendi dataset can outperform a 10× larger dataset** drawn from the same source pool. This is the central empirical finding of the paper: **diversity of learning signal > quantity of training samples** for out-of-distribution generalisation.

---

## Core Insight

> What matters for generalization is not how much data you have or how different it looks — it's whether the data teaches the model to update its weights in genuinely different directions. Gradient space is the right space to measure diversity. Filling sparse regions of that space is the right strategy for generating data.

---

## The Story: The Cartographers' Guild

### Chapter 1 — The Map Room

The kingdom's greatest library kept a vast Map Room. Any scholar who wanted to understand the world would come here and study maps — maps of terrain, maps of cities, maps of trade routes. The more maps you studied, the better you understood how the world worked.

When the kingdom decided to train a new generation of navigators, the obvious strategy was: give them as many maps to study as possible. More maps, better navigators.

The training academy set to work printing maps. Thousands of them. Then tens of thousands. The navigators studied dutifully.

But something strange happened. When the navigators were sent out into unfamiliar territory — regions they hadn't studied — many of them failed. They could navigate the regions on their maps beautifully. But new territory confused them.

A junior cartographer named G raised her hand. "I think I know the problem," she said. "Look at the maps we've been printing."

---

### Chapter 2 — The Redundant Maps

The academy had printed 80,000 maps. G laid them out. Every single one was of the capital city or the three main trade routes. There were maps at different scales, maps in different seasons, maps with different artistic styles. They all *looked* different. But they all showed the same territory.

"We have 80,000 maps," G said. "But we've covered only 3 regions of the kingdom. A navigator who studies all 80,000 of these maps knows those 3 regions in extraordinary detail — and knows nothing else."

The academy director shrugged. "But the maps *are* diverse. Look — different words, different symbols, different projections."

"Diverse in appearance," G replied. "Not diverse in what they teach. A navigator who studies map 47 learns the same roads as a navigator who studies map 48. They just see it drawn differently."

This was the central problem. **Surface diversity ≠ learning diversity.**

---

### Chapter 3 — The Compass of Learning

G proposed a new tool: instead of measuring how different maps *look*, measure what each map *teaches*.

She invented a special compass — the **G-Vendi compass**. You hand it a map, and it doesn't measure the map's size or colour or topic. It measures something deeper: *if a navigator studied only this map, in which direction would their understanding grow?*

The compass needle pointed in a direction in a mental space — the space of everything a navigator could possibly learn. Two maps pointing the same direction would teach the same lesson. Two maps pointing in different directions would teach genuinely different things.

Now G could measure the collection of 80,000 maps properly. She held the compass over each map in turn, recorded the direction it pointed, and plotted all 80,000 directions on a great chart.

They were all clustered in a tiny corner of the chart. Every single one of those 80,000 maps, no matter how visually different, pointed in nearly the same direction. The navigators had been studying 80,000 maps but learning from only a handful of genuinely distinct lessons.

The vast remainder of the chart — the space of everything else a navigator could learn — was blank.

---

### Chapter 4 — The Blank Regions

G called the dense corner of the chart the **crowded territory**. She called everything else the **frontier**.

The frontier represented kinds of understanding the academy's navigators had never encountered. Mountain pass navigation. Coastal tidal patterns. Desert landmark reasoning. River delta crossings. Every skill a navigator might need in unfamiliar territory — and the training maps covered none of it.

The academy director was alarmed. "We have 80,000 maps! How can our navigators be unprepared?"

"Because," G said, "preparing for unfamiliar territory requires maps *of* unfamiliar territory. We spent all our effort making more detailed maps of the capital. The frontier stayed blank."

---

### Chapter 5 — Prismatic Synthesis: Mapping the Frontier

G designed a new map-making process. Instead of printing maps freely and hoping for diversity, she built a system that deliberately targeted the blank regions.

The process ran in cycles:

**Step 1 — Survey:** Take all existing maps. Use the G-Vendi compass to plot every map's learning direction on the great chart. Divide the chart into clusters — dense clusters (already well-covered) and sparse clusters (barely covered).

**Step 2 — Dispatch:** Send cartographers out to generate new maps. Encourage them to explore broadly.

**Step 3 — Select:** As new maps arrived, test each one with the compass. If the compass pointed toward a dense cluster — territory already well-mapped — discard that map. Only keep maps whose compass pointed toward a sparse cluster, toward the frontier.

Repeat. Every cycle, the frontier shrank. Every cycle, the navigators' training became more complete.

Like a prism splitting white light into every colour of the spectrum, the system forced the training library to cover the full spectrum of learning — not just the brightest, most common colours.

---

### Chapter 6 — The Small Kingdom vs. The Large Kingdom

Across the mountains, a rival kingdom had a different approach. They hired the greatest master cartographer in the world — a legendary figure who had mapped every corner of the continent over 60 years. His maps were extraordinarily detailed, beautifully accurate, the finest anyone had ever seen.

The rival academy trained their navigators on hundreds of thousands of the master's maps.

G's kingdom used a much younger cartographer — competent, but nothing special. Nowhere near the master's reputation. But they used Prismatic Synthesis to select maps: only the ones that covered blank frontier regions, only the ones that pointed the G-Vendi compass in a new direction.

When the navigators from both kingdoms were tested on a journey through entirely unknown territory, G's navigators — trained on the younger cartographer's frontier-diverse maps — performed better on 6 of 7 tests.

The rival kingdom's navigators knew the master's terrain in extraordinary depth. But it was always the same terrain. G's navigators had been trained to think across the full spectrum of navigation challenges. When the unknown territory didn't resemble anything they'd studied, they could still reason through it.

**A smaller map-maker covering diverse territory beat a master map-maker covering familiar territory.**

---

### Chapter 7 — Why More Maps Stop Helping

The academy director noticed that before G's system, performance had stopped improving around map number 80,000. Adding more maps made no difference.

G explained with a simple picture. Imagine the chart of learning directions as a field. The first 10,000 maps planted seeds densely in the corner. The next 10,000 maps planted seeds in the same corner, on top of the old ones. And the next 10,000 on top of those.

The corner was saturated. New seeds couldn't take root — there was nowhere left to grow. The field beyond the corner was still empty.

Vanilla map generation hit a ceiling because all maps landed in already-saturated territory. You could print a million more maps and the navigators would learn nothing new — just variations of lessons they'd already mastered.

Prismatic Synthesis broke the ceiling by always planting new seeds in the untouched parts of the field. Each cycle opened new ground. Each new map taught something genuinely novel.

---

### Chapter 8 — Diversity Over Scale

The final lesson was the most counterintuitive. G showed that a collection of 100,000 frontier-diverse maps trained better navigators than a collection of 1,000,000 maps drawn from the same redundant pool.

Ten times more maps. Worse navigators.

Because the million maps covered the same territory, over and over. The hundred thousand frontier maps covered ten times the terrain.

**For generalisation, coverage matters more than count. A diverse 100K beats a redundant 1M.**

---

### The Moral

The Prismatic Synthesis paper answers a question that seems obvious but isn't: what does "diverse training data" actually mean?

Not diverse in topic. Not diverse in wording. Not diverse in the style of presentation. Diverse in what the model *learns* — in which directions of the parameter space get updated.

The gradient is the right measuring stick because it directly encodes what a training sample teaches. G-Vendi measures how much of the learning landscape is covered. Prismatic Synthesis is the algorithm for actively filling the gaps.

> More data is not better data. More directions of learning is better data.

---

## The Math — Step by Step

### Step 1 — The Gradient of a Training Sample

When a model with parameters `θ` sees a training sample `(x, y)` — a math problem `x` and its solution `y` — it computes a loss (how wrong its prediction is). The **gradient** of that loss tells you: in which direction should the model's weights move to get better on this sample?

```
raw gradient = -∇ log P(y | x; θ)
```

This is the standard backprop gradient — a vector of partial derivatives of the loss with respect to every parameter. For a 0.5B proxy model that's ~500 million numbers.

We **normalise** it to unit length so only direction matters, not magnitude:

```
g(x, y) = raw_gradient / ||raw_gradient||
```

`g(x, y)` is now a unit vector in 500M-dimensional space. Two samples that would teach the model the same thing have gradients pointing in roughly the same direction. Two samples that teach genuinely different things point in very different directions.

**Why direction, not magnitude?** A hard problem and an easy problem about the same topic both push the weights the same direction — the hard one just pushes harder. Normalising strips out difficulty, keeping only the *kind* of learning.

---

### Step 2 — Random Projection: From 500M to Manageable

You can't build a matrix from 500M-dimensional vectors for a million samples — that's 10¹⁵ numbers. **Random projection** solves this.

The Johnson-Lindenstrauss lemma guarantees: you can project any set of vectors from a high-dimensional space to a much lower-dimensional space using a random matrix, and the **dot products between vectors are preserved** (approximately). Direction relationships survive the compression.

The paper uses **Rademacher projections** — the random matrix `R` has entries of `+1/√k` or `-1/√k` with equal probability, where `k` is the target dimension (~4096). Cheap to apply, strong theoretical guarantees:

```
g_compressed = R @ g          # R is [k × 500M], g_compressed is [k]
```

After this step, each sample is a short vector instead of 500M values. The geometry — which samples are similar, which are different — is preserved.

---

### Step 3 — The Kernel Matrix (Dot Products of All Gradients)

Build a matrix `K` of pairwise dot products between all compressed gradient vectors:

```
K[i][j] = g_compressed_i · g_compressed_j
```

`K` is `[N × N]` where N is the number of samples. Entry `K[i][j]` measures how similar the learning signals of sample `i` and `j` are. Close to 1 = they teach the same thing. Close to 0 = they teach orthogonal things.

Normalise: `K_norm = K / N` — this is the **density matrix**.

---

### Step 4 — Eigenvalues: Finding the Independent Directions

Decompose `K_norm` into its eigenvalues `λ₁, λ₂, ..., λ_N`.

Each eigenvalue corresponds to one independent "learning direction" in parameter space. Its magnitude tells you how much of the dataset's total learning signal points that way.

**Two extremes:**

All samples teach the same thing → one dominant eigenvalue near 1, all others near 0:
```
eigenvalues: [0.97, 0.01, 0.01, 0.01, ...]   ← concentrated, rank-1
```

All samples teach different things → eigenvalues spread evenly:
```
eigenvalues: [0.05, 0.05, 0.05, 0.05, ...]   ← flat spectrum, full rank
```

---

### Step 5 — Shannon Entropy of the Eigenvalue Distribution

```
H = -∑ᵢ λᵢ · log(λᵢ)        (0·log(0) = 0 by convention)
```

Low entropy = concentrated spectrum = homogeneous dataset.
High entropy = flat spectrum = diverse dataset covering many independent learning directions.

---

### Step 6 — G-Vendi: Exponentiated Entropy

The raw entropy `H` is hard to interpret. The **Vendi Score** (Friedman & Dieng, 2022) proposes exponentiating it:

```
G-Vendi = exp(H) = exp(-∑ᵢ λᵢ · log(λᵢ))
```

**What `exp(H)` means physically:** the **effective number of distinct training signals** in the dataset. It answers: "how many genuinely different examples does this dataset behave like?"

- 1M identical samples → G-Vendi ≈ 1 (behaves like 1 unique example)
- 1M perfectly orthogonal samples → G-Vendi ≈ 1M (every sample is independent)
- Real datasets land in between

This is the **effective rank** of the kernel matrix — a concept from signal processing (Roy & Vetterli, 2007). A 1M-sample dataset with G-Vendi of 10,000 behaves like 10,000 genuinely distinct training signals, regardless of how many physical samples it contains.

**Full pipeline:**
```
Dataset D = {(x₁,y₁), ..., (x_N, y_N)}

1. g_i       = -∇ log P(y_i|x_i; θ_proxy) / ||...||    # unit gradient [500M]
2. g_i_c     = R @ g_i                                   # compressed [k ≈ 4096]
3. K[i][j]   = g_i_c · g_j_c  ;  K_norm = K / N        # density matrix [N×N]
4. λ₁...λ_N  = eigenvalues(K_norm)                      # spectrum
5. H         = -∑ λᵢ log λᵢ                             # entropy
6. G-Vendi   = exp(H)                                    # effective unique signals
```

---

### Why This Predicts Generalisation — The Deep Reason

There is a classical result in learning theory: the **influence of a training sample on a test sample** is proportional to the dot product of their gradients:

```
influence(train_i → test_j) ≈ g_train_i · g_test_j
```

A diverse training set (many gradient directions) has at least *some* gradient aligned with almost any test sample encountered. A homogeneous training set (few gradient directions) aligns well with similar test samples but poorly with anything else — exactly the OOD failure the paper documents.

High G-Vendi → many gradient directions → aligned with many test distributions → strong OOD performance. This is why the correlation with OOD benchmarks is Spearman's ρ ≈ 0.9, while surface metrics (n-gram, embedding) are near zero.

---

## The Codebase

**Repository:** [github.com/jaehunjung1/prismatic-synthesis](https://github.com/jaehunjung1/prismatic-synthesis)

### Directory Structure

```
prismatic-synthesis/
├── g-vendi/                          ← G-Vendi metric
│   ├── gradient_computer.py          ← backprop logic, gradient extraction + normalisation
│   ├── collect_gradients.py          ← iterates dataset, calls gradient_computer, batches
│   ├── gradient_vendi.py             ← eigendecomposition + entropy → G-Vendi score
│   ├── compute_g-vendi.py            ← CLI entry point
│   └── data/
│       ├── datasets/                 ← input JSONL files
│       └── gradient_storage/         ← saved .safetensors gradient files
│
├── prismatic-synthesis/              ← the data generation algorithm
│   ├── *_modules/                    ← clustering, filtering, LLM prompting logic
│   └── data/
│       ├── generated/                ← raw LLM-generated candidates
│       ├── gradient_storage/         ← gradients of candidates
│       └── complete/                 ← final filtered dataset batches
│
└── requirements.txt
```

### What Each Key File Does

| File | What it does |
|---|---|
| `gradient_computer.py` | Forward+backward pass through proxy model, extracts loss gradient, normalises, applies Rademacher projection, saves to `.safetensors` |
| `collect_gradients.py` | Iterates over dataset JSONL, calls `gradient_computer` for each sample with batching |
| `gradient_vendi.py` | Loads compressed gradients, builds kernel matrix `K`, eigendecomposes, computes `exp(-∑ λ log λ)` |
| `compute_g-vendi.py` | CLI wrapper — takes dataset file + gradient storage path, prints G-Vendi score |
| `*_modules/` | k-means clustering in gradient space, sparse-cluster filter (bottom 20%), LLM generation prompting |

The proxy model used is **Qwen2.5-0.5B-Instruct** — tiny and fast. Gradients from this scale correlate well with gradients from larger models, so you don't need to run backprop through a 7B+ model.

### Running G-Vendi on a Dataset

```bash
pip install -r requirements.txt

# Step 1: collect gradients for each sample
cd g-vendi/scripts
bash collect_gradients.sh
# → writes .safetensors files to g-vendi/data/gradient_storage/

# Step 2: compute the G-Vendi score
python compute_g-vendi.py \
  --dataset_filename=./data/datasets/seed.jsonl \
  --gradient_storage=./data/gradient_storage/train--qwen2.5-0.5b-instruct
# → prints G-Vendi (effective number of unique learning signals)
```

### Running Prismatic Synthesis (Full Data Generation Loop)

```bash
cd prismatic-synthesis/scripts

# Step 1: gradients for seed dataset
bash collect_seed_set_gradients.sh

# Step 2: generate candidates with an LLM
bash generate_problem.sh      # generate problem statements
bash generate_solution.sh     # generate solutions

# Step 3: compute gradients for new candidates
bash collect_new_gradient.sh

# Step 4: cluster + keep only sparse-cluster samples
bash cluster_filter.sh
# → data/complete/new-batch-1.jsonl  (frontier-only samples)

# Repeat steps 2-4 for the next batch
```

Each `cluster_filter.sh` run keeps only samples whose gradients land in the least-populated 20% of clusters. All others are discarded as redundant in gradient space.

### Pre-Built Datasets (No Pipeline Needed)

The PrismMath and PrismNLI datasets are publicly available on HuggingFace:
- **[nvidia/Nemotron-PrismMath](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath)** — 1M diverse math problem-solution pairs
- Use directly for fine-tuning without running the generation pipeline

---

## Technical Reference

### G-Vendi Formula
```
For each sample (x, y):
  g(x,y) = -∇ log P(y|x; θ) / ||∇ log P(y|x; θ)||   # normalized gradient

Project all gradients to lower dimension via Rademacher random projection
Build covariance matrix K where K_ij = g_i · g_j
Compute eigenvalues λ_1 ... λ_n of normalized K
G-Vendi = exp(H(λ))  where H is Shannon entropy of the eigenvalue distribution
```

Higher G-Vendi = eigenvalues spread across many dimensions = many distinct learning directions = more diverse.

### Prismatic Synthesis Algorithm
```
dataset D = initial seed data
while |D| < target_size:
    gradients = compute_gradients(D, proxy_model)
    clusters  = cluster(gradients)                    # find dense and sparse clusters
    candidates = generate(LLM, D)                    # generate new samples
    new_grads  = compute_gradients(candidates, proxy_model)
    keepers    = [c for c in candidates
                  if cluster(c) in sparse_clusters]  # keep only frontier samples
    D = D ∪ keepers
```

### Key Numbers

| Metric | Value |
|---|---|
| G-Vendi correlation with OOD performance | Spearman's ρ ≈ 0.9, R² > 0.8 |
| Training runs in empirical study | 300+ |
| PrismMath dataset size | 1 million samples |
| Generator model size | 32B (Qwen) |
| Competing approach generator size | 671B (DeepSeek-R1) |
| Size advantage | 20× smaller generator |
| Benchmarks won | 6 of 7 (AIME24, AIME25, AMC23, others) |
| PrismMath-7B AIME24 score | 57.08% |
| PrismNLI OOD improvement | +8% over prior mixtures |
| Vanilla generation saturation point | ~50K–100K samples |

---

## Relation to Other Concepts

- **Artificial Hivemind** (see `../artificial-hivemind/`) — direct connection: the hivemind paper shows LLMs generate homogeneous outputs; Prismatic Synthesis is one approach to *create* the missing diversity in training data by targeting underexplored gradient regions
- **Curriculum learning** — related idea of choosing *what* to learn in a deliberate order; Prismatic Synthesis is curriculum learning guided by gradient space coverage
- **Active learning** — selecting the most informative data points to label; Prismatic Synthesis is active *generation* — generating the most informative data rather than selecting from existing data
- **Coreset selection** — finding a small subset of data that represents the full dataset; G-Vendi provides the metric for what "represents" means in learning terms
- **Data flywheel / synthetic data loops** — Prismatic Synthesis is a disciplined synthetic data flywheel that avoids the mode collapse problem by enforcing gradient diversity at each cycle
- **Model collapse** (Shumailov et al.) — vanilla synthetic data generation leads to model collapse because each generation concentrates on the same regions; Prismatic Synthesis prevents this by design
- **GFlowNets** (see `../gflownets-reasoning/`) — GFlowNets learn to sample proportionally from a reward distribution; Prismatic Synthesis samples synthetic data proportionally to how underrepresented each gradient region is — similar objective, different domain

---

Sources:
- [Project page — NVIDIA Labs](https://nvlabs.github.io/prismatic-synthesis/)
- [arXiv abstract](https://arxiv.org/abs/2505.20161)
- [NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/hash/82a34882f560db982d1257b5af461605-Abstract-Conference.html)
- [Nemotron-PrismMath dataset — HuggingFace](https://huggingface.co/datasets/nvidia/Nemotron-PrismMath)
- [Moonlight review](https://www.themoonlight.io/en/review/prismatic-synthesis-gradient-based-data-diversification-boosts-generalization-in-llm-reasoning)
