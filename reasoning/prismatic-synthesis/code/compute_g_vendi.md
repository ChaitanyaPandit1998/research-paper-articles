# `compute_g-vendi.py` — How It Works

**Source:** `g-vendi/compute_g-vendi.py`
**Role:** The CLI entry point. Ties together `GradientVendi.load_all_gradients` and `GradientVendi.compute_gradient_vendi` into a single command you can run to get the G-Vendi score for any dataset.

---

## The Whole File

```python
from gradient_vendi import GradientVendi

def parse_args():
    parser.add_argument("--dataset_filename", type=str)
    parser.add_argument("--gradient_storage_dir", type=str)
    ...

if __name__ == "__main__":
    args = parse_args()

    sample_ids, sample_gradients = GradientVendi.load_all_gradients(args.gradient_storage_dir)

    with jsonlines.open(args.dataset_filename) as f:
        dataset_sample_ids = set(s['id'] for s in list(f))
    assert dataset_sample_ids == set(sample_ids), "dataset_sample_ids != set(sample_ids)."

    g_vendi = GradientVendi.compute_gradient_vendi(sample_gradients)
    print(f"G-Vendi for {args.dataset_filename}: {g_vendi}")
```

This is deliberately thin. All the real work happens in `gradient_vendi.py`. This file's only job is to:
1. Parse two CLI arguments
2. Load gradients
3. Validate completeness
4. Print the score

---

## The Two Arguments

**`--dataset_filename`** — path to the original dataset JSONL (e.g. `./data/datasets/seed.jsonl`). Used only for validation — to check that every sample in the dataset has a corresponding gradient. The actual data content is not used.

**`--gradient_storage_dir`** — path to the directory of `.safetensors` files produced by `collect_gradients.py` (e.g. `./data/gradient_storage/train--qwen2.5-0.5b-instruct`).

---

## The Completeness Check

```python
dataset_sample_ids = set(s['id'] for s in jsonlines.open(args.dataset_filename))
assert dataset_sample_ids == set(sample_ids), "dataset_sample_ids != set(sample_ids)."
```

Strict equality check — every sample in the dataset must have exactly one gradient, and no extra gradients may exist. This catches:

- **Partial runs** — if `collect_gradients.py` was interrupted before finishing, some samples will be missing gradients. The assert fires, telling you to finish the collection before scoring.
- **Stale gradients** — if the dataset was updated (samples added or removed) after gradients were collected, the sets won't match.
- **Wrong directory** — if you point `--gradient_storage_dir` at a different dataset's gradients.

Failing loudly here is much better than silently computing a G-Vendi score on an incomplete set of gradients — which would give a misleadingly low score (fewer samples = fewer gradient directions = artificially low entropy).

---

## Running It

```bash
python compute_g-vendi.py \
  --dataset_filename ./data/datasets/seed.jsonl \
  --gradient_storage_dir ./data/gradient_storage/train--qwen2.5-0.5b-instruct
```

Output:
```
G-Vendi for ./data/datasets/seed.jsonl: 1842.3
```

The number is the **effective number of unique learning signals** — interpretable as "this dataset of 50,000 samples behaves like 1,842 genuinely distinct training examples." Higher is more diverse. There is no fixed "good" threshold; G-Vendi is most useful for comparing two datasets drawn from the same pool (before vs. after Prismatic Synthesis filtering).

---

## Where This Fits in the Pipeline

```
collect_gradients.py     ← produces gradient_storage/
         │
         ▼
compute_g-vendi.py       ← reads gradient_storage/, prints score
         │
         (score is used to compare datasets, not needed for cluster_filter.py)
```

`compute_g-vendi.py` is a **measurement tool**, not part of the synthesis loop. You run it to evaluate the diversity of a dataset. The Prismatic Synthesis loop itself (`cluster_filter.py`) uses the cluster structure from `gradient_vendi.py` directly — it doesn't go through this CLI.

---

## Practical Use Cases

**Comparing seed vs filtered dataset:**
```bash
# score the original seed data
python compute_g-vendi.py --dataset_filename seed.jsonl \
  --gradient_storage_dir gradient_storage/seed--qwen2.5-0.5b-instruct
# G-Vendi: 1842

# score after one round of Prismatic Synthesis
python compute_g-vendi.py --dataset_filename seed+batch1.jsonl \
  --gradient_storage_dir gradient_storage/seed+batch1--qwen2.5-0.5b-instruct
# G-Vendi: 2941   ← more diverse after filtering
```

**Comparing two source datasets:**
```bash
python compute_g-vendi.py --dataset_filename r1_671b_data.jsonl ...
# G-Vendi: 2100

python compute_g-vendi.py --dataset_filename prismmath_data.jsonl ...
# G-Vendi: 8700   ← PrismMath is more diverse despite smaller generator
```

This is exactly the comparison the paper makes to justify that gradient diversity (not teacher model size) drives generalisation.
