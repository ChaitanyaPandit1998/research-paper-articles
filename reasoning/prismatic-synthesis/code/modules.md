# `*_modules/` and Pipeline Scripts — How They Work

**Sources:**
- `prismatic-synthesis/cluster_filter.py`
- `prismatic-synthesis/generate_problem.py`
- `prismatic-synthesis/cluster_modules/` — k-means in gradient space
- `prismatic-synthesis/generation_modules/` — vLLM-based problem/solution generation
- `prismatic-synthesis/gradient_modules/` — gradient loading utilities (mirrors `g-vendi/`)

---

## The Prismatic Synthesis Loop

These files implement the iterative data generation algorithm. One full cycle:

```
┌─────────────────────────────────────────────────────────────┐
│                    One Prismatic Synthesis Cycle             │
│                                                             │
│  Current pool (seed + previous batches)                     │
│          │                                                  │
│          ▼                                                  │
│  [cluster_filter.py Step 1]                                 │
│  K-means on pool's gradient space → cluster labels          │
│          │                                                  │
│          ▼                                                  │
│  [generate_problem.py + generate_solution.py]               │
│  LLM generates N candidate problems+solutions               │
│          │                                                  │
│          ▼                                                  │
│  [collect_gradients.py — prismatic version]                 │
│  Compute gradients for new candidates                       │
│          │                                                  │
│          ▼                                                  │
│  [cluster_filter.py Step 2]                                 │
│  Assign candidates to clusters → keep only sparse ones      │
│          │                                                  │
│          ▼                                                  │
│  New batch added to pool → repeat                           │
└─────────────────────────────────────────────────────────────┘
```

---

## `cluster_filter.py` — The Gatekeeper

This is the most important file in the synthesis pipeline. It decides which generated samples get added to the dataset.

### Full Code Walk-Through

```python
def filter_cluster(sample_gradients, ratio, current_cluster_centroids, current_cluster_labels):
    sample_gradients = F.normalize(sample_gradients, dim=1)

    # assign each new sample to its nearest centroid (cosine similarity)
    similarity_with_centroids = sample_gradients @ current_cluster_centroids.T   # [N_new, k]
    sample_cluster_labels = torch.argmax(similarity_with_centroids, dim=-1).tolist()

    # find the smallest `ratio` clusters in the current pool
    small_clusters = ClusterManager.smallest_clusters(current_cluster_labels, ratio)

    # keep only new samples that land in small clusters
    return [label in small_clusters for label in sample_cluster_labels]
```

**`ClusterManager.smallest_clusters(labels, ratio=0.5)`** — finds which cluster IDs contain the fewest samples. With `ratio=0.5`, it returns the 50% of clusters with the lowest sample counts. These are the "sparse" or "underrepresented" regions.

**Why cosine similarity for assignment?** Gradient vectors are unit-normalised, so cosine similarity = dot product. The centroid that has the highest dot product with a new sample's gradient is the one whose cluster the sample belongs to.

### The `__main__` Block Step by Step

```python
# Step 1: load gradients for the current pool (seed + all previous batches)
_, orig_sample_gradients = GradientManager.load_gradients_for_sample_ids(
    args.gradient_dir, orig_sample_ids
)

# Step 2: k-means on the current pool to find cluster structure
cluster_labels, cluster_centroids = ClusterManager.cluster_kmeans(
    orig_sample_gradients,
    k=int(orig_sample_gradients.size(0) * 0.1),   # k = 10% of pool size
    num_iter=20,
    use_tqdm=True
)
```

`k = 10% of pool` is a key hyperparameter. If pool has 10K samples, k=1000 clusters. The cluster structure reflects the gradient space of the current pool — dense clusters = well-covered reasoning patterns, sparse clusters = gaps.

```python
# Step 3: load new candidates and their gradients
_, new_sample_gradients = GradientManager.load_gradients_for_sample_ids(
    args.gradient_dir, new_sample_ids
)

# Step 4: filter
filter_results = filter_cluster(
    new_sample_gradients, ratio=0.5,
    current_cluster_centroids=cluster_centroids,
    current_cluster_labels=cluster_labels
)
```

`ratio=0.5` means: find the 50% of clusters with the fewest pool samples. Only keep new samples that fall into those sparse clusters.

**What "landing in a sparse cluster" means:** a new sample's gradient is most similar (cosine) to a centroid that represents an underrepresented region. The sample, if added to the pool, would fill a gap in gradient coverage.

**What happens to dense-cluster samples:** discarded. They would add a new version of something already well-covered. No matter how high-quality the sample is, if it's gradient-redundant with the existing pool, it gets dropped.

```python
# Step 5: collect and save kept samples
out_samples = [s for result, s in zip(filter_results, new_samples) if result]
save_to_file(out_samples, args.save_filename)
```

Typically 30–50% of generated candidates pass the filter (the bottom 50% of clusters by count don't necessarily contain 50% of random new samples — most random samples land in the dense clusters).

---

### Why `ratio=0.5` Not `ratio=0.2`?

The paper experiments with different ratios. `ratio=0.5` means "bottom half by cluster size." This is more aggressive than `ratio=0.2` (bottom 20%) — it accepts samples from a wider range of underrepresented clusters, increasing diversity but potentially accepting some noisier samples. The paper found `ratio=0.5` gave the best balance between diversity and quality.

---

## `generate_problem.py` — The Problem Generator

```python
model = VLLMGenerator(args.full_model_name, max_model_len=8192, max_gen_len=8192)
```

Uses **vLLM** for fast batched generation. vLLM's continuous batching makes generating thousands of problems efficiently feasible — it can process `batch_size=512` requests in parallel.

### Few-Shot Seeding with Difficulty Weighting

```python
sample_levels = np.array([s["level"] if "level" in s else 1 for s in samples])
sample_levels = sample_levels / np.sum(sample_levels)   # normalise to probability distribution

batched_fewshot_indices = [
    np.random.choice(range(len(samples)), size=args.num_fewshot_samples,
                     replace=False, p=sample_levels)
    for _ in range(args.batch_size)
]
```

For each generation request, randomly sample `num_fewshot_samples=5` problems from the current pool as few-shot examples. The sampling probability is **weighted by difficulty level** (`sample["level"]`).

**Why weight by difficulty?** Without weighting, the generator would mostly see easy problems (they're more common in any pool) and generate easy problems. Difficulty weighting gives harder problems a proportionally higher chance of being selected as examples, nudging the generator toward harder problem types. This is a soft form of diversity injection at the generation stage, before the gradient-based filter does the hard selection.

### Generation Loop

```python
while num_generated < args.target_size:
    batched_fewshot_samples = [sample_from_pool(5) for _ in range(batch_size)]
    batch_out_samples = model.batch_prompt_problem(batched_fewshot_samples, args.num_new_problems)
    
    for fewshot, out in zip(batched_fewshot_samples, batch_out_samples):
        for sample in out:
            sample["prompt_id"] = f"gen.{input_stem}.{uuid.uuid4().hex}"
        save_to_file(out, args.out_filename, save_mode="a")
        num_generated += len(out)
```

- `batch_size=512` generation requests per round
- Each request generates `num_new_problems=2` new problems
- Ideally produces `512 × 2 = 1024` problems per round
- Target: `target_size=3000` problems per script invocation

Each generated sample gets a UUID-based `prompt_id` for traceability (`gen.{seed_stem}.{uuid}`). The `seed_stem` records which seed dataset spawned this generation round.

### `batch_prompt_problem`

This calls the vLLM generator with a prompt like:

```
Here are some example math problems:

Problem 1: [fewshot_sample_1]
Problem 2: [fewshot_sample_2]
...
Problem 5: [fewshot_sample_5]

Now generate 2 new, distinct math problems that are similar in style and difficulty.
```

The exact prompt format is inside `generation_modules/vllm_model.py` (not fetched, but this is the standard few-shot generation pattern).

---

## Module Directory Roles

### `cluster_modules/`

Contains `ClusterManager` — wraps `GradientVendi.cluster_kmeans` with the same interface but adapted for the synthesis pipeline. The key method:

```python
ClusterManager.cluster_kmeans(data, k, num_iter, use_tqdm)  → (labels, centroids)
ClusterManager.smallest_clusters(labels, ratio)             → set of cluster IDs with fewest samples
```

`smallest_clusters` counts how many pool samples fall in each cluster (via `Counter(labels.tolist())`), then returns the IDs of the bottom `ratio` fraction by count.

### `gradient_modules/`

Contains `GradientManager` — mirrors `GradientVendi`'s gradient loading functions but adapted for the synthesis pipeline's directory layout (which is separate from the `g-vendi/` directory). Functionally identical to `GradientVendi.load_gradients_for_sample_ids`.

### `generation_modules/`

Contains:
- `VLLMGenerator` — wraps vLLM's `LLM` class with convenience methods `batch_prompt_problem` and `batch_prompt_solution`
- `generate_model_util.py` — `save_to_file` helper that appends JSONL records to an output file

---

## Full Pipeline Command Sequence

```bash
cd prismatic-synthesis/scripts

# 0. Start with a seed dataset in data/datasets/seed.jsonl

# 1. Compute gradients for the seed dataset
bash collect_seed_set_gradients.sh
# → data/gradient_storage/seed--qwen2.5-0.5b-instruct/*.safetensors

# 2. Generate candidate problems with Qwen2.5-72B
bash generate_problem.sh
# → data/generated/problems-batch1.jsonl  (3000 raw problems)

# 3. Generate solutions for those problems
bash generate_solution.sh
# → data/generated/batch1.jsonl  (3000 problems with solutions)

# 4. Compute gradients for the new candidates
bash collect_new_gradient.sh
# → data/gradient_storage/batch1--qwen2.5-0.5b-instruct/*.safetensors

# 5. Filter: keep only sparse-cluster candidates
bash cluster_filter.sh
# → data/complete/new-batch-1.jsonl  (~1000-1500 filtered samples)

# 6. Merge new batch into pool and repeat from step 2
```

---

## Key Design Decisions

| Decision | Value | Reasoning |
|---|---|---|
| `k = 10% of pool` | k-means clusters | Large enough to find structure, small enough for tractable clustering |
| `ratio = 0.5` | Sparse cluster threshold | Keep bottom 50% by size — aggressive diversity without over-filtering |
| `num_fewshot = 5` | Few-shot examples | Enough context for the generator to understand style/difficulty |
| `target_size = 3000` | Problems per round | Generates ~3× more than needed (expecting ~33–50% pass rate after filtering) |
| `batch_size = 512` | vLLM batch | Maximises GPU utilisation during generation |
| Generator model | Qwen2.5-72B | Large enough to generate high-quality diverse problems, smaller than competing 671B approach |
| Proxy model | Qwen2.5-0.5B | Tiny — fast gradient computation, correlates well with larger model gradients |
