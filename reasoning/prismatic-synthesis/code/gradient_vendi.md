# `gradient_vendi.py` — How It Works

**Source:** `g-vendi/gradient_vendi.py`
**Role:** Contains all the maths — loads compressed gradients from disk, builds the kernel matrix, eigendecomposes it, computes the G-Vendi score, and provides k-means clustering in gradient space (used by both score computation and `cluster_filter.py`).

---

## Two Classes

The file has two classes with distinct responsibilities:

- **`Vendi`** — pure math: entropy functions and the Vendi score formula. Stateless, all static methods.
- **`GradientVendi`** — gradient-specific operations: loading from disk, clustering, and the top-level G-Vendi computation. Also all static methods.

---

## Class: `Vendi`

### `entropy_q` — Rényi Entropy

```python
@staticmethod
def entropy_q(p, q=1):
    p_ = p[p > 0]              # ignore zero eigenvalues
    if q == 1:
        return -(p_ * np.log(p_)).sum()    # Shannon entropy
    if q == "inf":
        return -np.log(np.max(p))          # min-entropy
    return np.log((p_ ** q).sum()) / (1 - q)  # Rényi entropy order q
```

This computes **Rényi entropy** — a family of entropy measures parameterised by `q`. G-Vendi uses `q=1` which is standard Shannon entropy:

```
H(p) = -∑ pᵢ · log(pᵢ)
```

The `q=inf` case gives min-entropy (only the largest eigenvalue matters). The general case gives a tunable sensitivity between these extremes. G-Vendi always uses `q=1`.

**Why filter `p[p > 0]`?** The kernel matrix has rank at most `min(N, proj_dim)`. If N > proj_dim, some eigenvalues will be exactly zero (the matrix is rank-deficient). `0 · log(0)` is mathematically defined as 0 by convention, but `np.log(0)` would throw `-inf`. The filter avoids this.

---

### `compute_reverse_similarity_matrix` — The Efficiency Trick

```python
@staticmethod
def compute_reverse_similarity_matrix(data, normalize=True):
    if normalize:
        data = torch.Tensor(preprocessing.normalize(data.cpu(), axis=1)).cuda()
    return data.T @ data
```

This computes `X^T X` (shape `[proj_dim × proj_dim]`) instead of `X X^T` (shape `[N × N]`).

**Why this matters:**
- `N` = number of training samples (could be millions)
- `proj_dim` = 1024 (always fixed)
- `N × N` matrix at N=1M would be 10¹² entries — completely infeasible
- `proj_dim × proj_dim` = 1024 × 1024 = 1M entries — trivially small

**Why is this valid?** A fundamental result in linear algebra: the non-zero eigenvalues of `X X^T` and `X^T X` are identical. Since entropy only depends on the eigenvalue distribution, computing the Vendi score on `X^T X` gives exactly the same result as on `X X^T`.

This is the key trick that makes G-Vendi scalable to datasets of any size. The computation cost is `O(proj_dim² × N)` to build `X^T X`, then `O(proj_dim³)` to eigendecompose it — both are independent of N at large scales.

---

### `compute_vendi_score`

```python
@staticmethod
def compute_vendi_score(sim_matrix, n):
    w = scipy.linalg.eigvalsh(sim_matrix.cpu() / n)
    return np.exp(Vendi.entropy_q(w, q=1)).item()
```

**`eigvalsh`** — eigenvalue decomposition for symmetric (Hermitian) matrices. `eigvalsh` is faster and numerically more stable than general `eig` because it exploits the symmetry. `X^T X` is always symmetric positive semi-definite, so this is always valid.

Divides by `n` (number of samples) before decomposing — this normalises the matrix so eigenvalues sum to 1, making them interpretable as a probability distribution.

The result is `exp(H(λ))` — the **effective number of unique learning signals** in the dataset.

---

## Class: `GradientVendi`

### `load_all_gradients`

```python
@staticmethod
def load_all_gradients(gradient_files, device=0):
    if isinstance(gradient_files, Path):
        gradient_files = list(gradient_files.glob("*.safetensors"))

    all_gradient_dict = {}
    for gradient_file in gradient_files:
        all_gradient_dict.update(load_file(gradient_file, device=device))

    sample_ids, sample_gradients = [], []
    for sample_id, sample_gradient in all_gradient_dict.items():
        sample_ids.append(sample_id)
        sample_gradients.append(sample_gradient)

    sample_gradients = torch.stack(sample_gradients, dim=0)
    return sample_ids, sample_gradients
```

Loads all `.safetensors` files from the gradient storage directory. Each file is a dict `{sample_id: gradient_tensor}`. After loading all files, stacks all gradient tensors into a single matrix `[N, proj_dim]`.

The `device=0` parameter loads tensors directly onto GPU 0 — avoids the CPU→GPU transfer cost of loading to CPU first.

---

### `load_gradients_for_sample_ids`

```python
@staticmethod
def load_gradients_for_sample_ids(gradient_files, sample_ids, device=0):
    set_sample_ids = set(sample_ids)
    all_gradient_dict = {}
    for gradient_file in gradient_files:
        loaded = load_file(gradient_file, device=device)
        all_gradient_dict.update({s_id: g for s_id, g in loaded.items()
                                   if s_id in set_sample_ids})

    sample_gradients = [all_gradient_dict[sid] for sid in sample_ids]
    sample_gradients = torch.stack(sample_gradients, dim=0)
    return sample_ids, sample_gradients
```

Same as `load_all_gradients` but filters to a specific set of sample IDs. Uses a `set` for O(1) lookup. The final list comprehension `[all_gradient_dict[sid] for sid in sample_ids]` preserves the original ordering of `sample_ids` — important for `cluster_filter.py` where gradient order must match sample order.

---

### `cluster_kmeans` — GPU k-Means with Cosine Similarity

```python
@staticmethod
def cluster_kmeans(data, k, num_iter, use_tqdm=False):
    data = F.normalize(data, dim=1)        # unit-normalise all gradient vectors
    centroids = data[:k, :].clone()        # initialise: first k samples as centroids
    labels = None

    for _ in range(num_iter):
        # E step: assign each point to nearest centroid (cosine similarity)
        labels = GradientVendi._calculate_sim_matrix_and_label(data, centroids)

        # M step: update centroids to normalised cluster mean
        centroids.zero_()
        for cluster_idx in range(k):
            centroids[cluster_idx] = torch.sum(data[labels == cluster_idx], dim=0)
        centroids = F.normalize(centroids, dim=1)

    return labels, centroids
```

Standard Lloyd's algorithm but with **cosine similarity** instead of Euclidean distance. This makes sense for gradient vectors — what matters is the *direction* of the gradient, not its magnitude (magnitudes were already normalised in `gradient_computer.py`).

**Centroid initialisation:** takes the first `k` samples as initial centroids. Naive but fast. More sophisticated initialisation (k-means++) would give better convergence but is slower.

**M step detail:** instead of `mean()`, computes `sum()` then normalises. Normalising after averaging ensures centroids stay on the unit sphere (consistent with cosine similarity geometry). A simple `mean()` of unit vectors is not itself a unit vector.

**The `k = N / 10` choice** (used in `compute_gradient_vendi`): 10% of sample count. Large enough to capture structure, small enough to be tractable. For 1M samples, k=100K centroids — the k-means itself becomes expensive at this scale. For smaller datasets, it's fast.

---

### `_calculate_sim_matrix_and_label` — VRAM-Efficient Centroid Assignment

```python
@staticmethod
def _calculate_sim_matrix_and_label(data, centroids):
    max_batch_num_centroids = 90     # process 90 centroids at a time

    max_values_list, max_indices_list = [], []
    for batch_start in range(0, centroids.size(0), max_batch_num_centroids):
        batch_sim = data @ centroids[batch_start:batch_start+90].T   # (N, 90)
        max_values, max_indices = batch_sim.max(dim=1)
        max_indices += batch_start        # correct index offset
        max_values_list.append(max_values)
        max_indices_list.append(max_indices)
        del batch_sim                     # free immediately

    # find which batch had the best centroid for each point
    max_values_stacked  = torch.stack(max_values_list, dim=1)   # (N, num_batches)
    max_indices_stacked = torch.stack(max_indices_list, dim=1)
    best_batch = max_values_stacked.argmax(dim=-1)              # (N,)
    labels = max_indices_stacked[torch.arange(N), best_batch]   # (N,)
    return labels
```

The naive approach would compute `data @ centroids.T` = `[N × k]` all at once. With N=1M and k=100K, that's 10¹¹ floats — impossible to hold in VRAM.

The batched approach processes 90 centroids at a time, producing a `[N, 90]` similarity matrix, then immediately deletes it (`del batch_sim`). This keeps peak VRAM usage at `O(N × 90)` instead of `O(N × k)`.

The two-stage argmax (best within batch, then best across batches) is a clean way to find the global argmax without materialising the full matrix.

---

### `compute_gradient_vendi` — Top-Level Score

```python
@staticmethod
def compute_gradient_vendi(gradients):
    k = int(gradients.size(0) / 10)
    _, centroids = GradientVendi.cluster_kmeans(gradients, k, 20, use_tqdm=gradients.size(0) > 1e5)

    reverse_sim_matrix = Vendi.compute_reverse_similarity_matrix(F.normalize(gradients, dim=1))
    score = Vendi.compute_vendi_score(reverse_sim_matrix, n=gradients.shape[0])
    return score
```

**Note on the k-means call:** `cluster_kmeans` is called here but both return values — `labels` (captured as `_`) and `centroids` — are unused in the subsequent Vendi score computation. The Vendi score is computed on all `gradients`, not on the k-means centroids.

The comment says "empirically found that this mitigates the effect of noise from outliers while being more efficient." One interpretation: the k-means call normalises the gradient tensor in-place via `F.normalize` inside `cluster_kmeans`. However, `F.normalize` creates a new tensor (not in-place), so the original `gradients` is not modified. The k-means call here appears to be either vestigial code from an earlier version that used centroids for the Vendi computation, or there for a warming/caching effect on the GPU. The actual G-Vendi score is correctly computed from the full normalised gradient set.

**The actual score computation:**
1. `F.normalize(gradients, dim=1)` — normalise all gradient vectors to unit length
2. `compute_reverse_similarity_matrix` → `X^T X` of shape `[1024 × 1024]`
3. `compute_vendi_score` → eigenvalues → entropy → exp → G-Vendi

---

## Full Math Flow

```
Compressed gradients: G = [N × 1024]  (from .safetensors files)

Normalise:        G_norm = G / ||G||_row           [N × 1024], each row is a unit vector

Reverse sim:      K = G_norm.T @ G_norm            [1024 × 1024]  (= X^T X trick)

Normalise:        K_norm = K / N                   makes eigenvalues sum to 1

Eigendecompose:   λ₁,...,λ₁₀₂₄ = eigvalsh(K_norm) λᵢ ≥ 0, ∑λᵢ = 1

Entropy:          H = -∑ λᵢ log λᵢ                (Shannon entropy of spectrum)

G-Vendi:          score = exp(H)                   effective # of unique learning signals
```

---

## Dependencies

| Library | Used for |
|---|---|
| `scipy.linalg` | `eigvalsh` — symmetric eigendecomposition |
| `sklearn.preprocessing` | `normalize` — row-normalisation of gradient matrix (CPU fallback) |
| `torch.nn.functional` | `F.normalize` — GPU row-normalisation |
| `safetensors.torch` | `load_file` — load `.safetensors` gradient files |
