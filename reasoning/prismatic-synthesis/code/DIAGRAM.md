# Codebase Diagram — Prismatic Synthesis

---

## 1. Two Pipelines at a Glance

```
┌─────────────────────────────────┐     ┌─────────────────────────────────────────┐
│      G-VENDI PIPELINE           │     │        PRISMATIC SYNTHESIS PIPELINE      │
│      (Measurement)              │     │        (Data Generation Loop)            │
│                                 │     │                                          │
│  Dataset (JSONL)                │     │  Seed Dataset (JSONL)                   │
│       │                         │     │       │                                  │
│       ▼                         │     │       ▼                                  │
│  collect_gradients.py           │     │  generate_problem.py                    │
│       │                         │     │       │                                  │
│       ▼                         │     │       ▼                                  │
│  .safetensors files             │     │  generate_solution.py                   │
│       │                         │     │       │                                  │
│       ▼                         │     │       ▼                                  │
│  compute_g-vendi.py             │     │  collect_gradients.py (prismatic)       │
│       │                         │     │       │                                  │
│       ▼                         │     │       ▼                                  │
│  G-Vendi Score (float)          │     │  cluster_filter.py                      │
│                                 │     │       │                                  │
└─────────────────────────────────┘     │       ▼                                  │
                                        │  new-batch-N.jsonl → back to top ↑      │
                                        └─────────────────────────────────────────┘
```

---

## 2. File Dependency Graph (who imports whom)

```
                    ┌─────────────────────┐
                    │  collect_gradients  │◄──── shell: collect_gradients.sh
                    │  .py  (g-vendi/)    │
                    └────────┬────────────┘
                             │ imports
                             ▼
                    ┌─────────────────────┐       ┌──────────────────────┐
                    │  gradient_computer  │──────►│  trak.projectors     │
                    │  .py               │       │  (CudaProjector /    │
                    └─────────────────────┘       │   BasicProjector)    │
                             │                    └──────────────────────┘
                             │ writes
                             ▼
                    ┌─────────────────────┐
                    │  .safetensors files │◄──────────────────────────────┐
                    │  + .txt index files │                               │
                    └──────┬──────────────┘                               │
                           │                                              │
              ┌────────────┴───────────┐                                  │
              │                        │                                  │
              ▼                        ▼                                  │
  ┌───────────────────┐    ┌───────────────────────┐                      │
  │ compute_g-vendi   │    │   cluster_filter.py   │◄── shell:            │
  │ .py               │    │   (prismatic/)        │    cluster_filter.sh │
  └─────────┬─────────┘    └────────┬──────────────┘                      │
            │ imports                │ imports                             │
            ▼                        ▼                                     │
  ┌───────────────────┐    ┌───────────────────────┐                      │
  │  gradient_vendi   │    │  cluster_modules/     │                      │
  │  .py              │    │  cluster_manager.py   │                      │
  │                   │    └───────────────────────┘                      │
  │  ┌─────────────┐  │             │ wraps                               │
  │  │  Vendi      │  │             ▼                                     │
  │  │  (entropy,  │  │    ┌───────────────────────┐                      │
  │  │   eigvalsh) │  │    │  gradient_vendi.py    │ ◄───────────────────-┘
  │  └─────────────┘  │    │  cluster_kmeans()     │ (shared math layer)
  │  ┌─────────────┐  │    └───────────────────────┘
  │  │ GradientV-  │  │
  │  │ endi        │  │    ┌───────────────────────┐
  │  │ (load,      │  │    │  gradient_modules/    │
  │  │  cluster,   │  │    │  gradient_manager.py  │
  │  │  score)     │  │    └──────────┬────────────┘
  │  └─────────────┘  │               │ mirrors
  └───────────────────┘               ▼
                              ┌───────────────────────┐
                              │  gradient_vendi.py    │
                              │  load_gradients_for   │
                              │  _sample_ids()        │
                              └───────────────────────┘

  ┌────────────────────────┐       ┌────────────────────────────┐
  │  generate_problem.py   │──────►│  generation_modules/       │
  │  (prismatic/)          │       │  vllm_model.py             │
  └────────────────────────┘       │  (VLLMGenerator)           │
  ┌────────────────────────┐       │                            │
  │  generate_solution.py  │──────►│  batch_prompt_problem()    │
  │  (prismatic/)          │       │  batch_prompt_solution()   │
  └────────────────────────┘       └────────────────────────────┘
```

---

## 3. Data Flow — What Each File Reads and Writes

```
                          INPUT                    FILE                      OUTPUT
                    ──────────────────────────────────────────────────────────────────

 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  G-VENDI MEASUREMENT PIPELINE                                                     │
 ├────────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                    │
 │  seed.jsonl ──────────────► collect_gradients.py ──────────► *.safetensors        │
 │  {id, prompt, completion}   (gradient_computer.py inside)     {id: tensor[1024]}   │
 │                                                               *.txt (index)        │
 │                                                                                    │
 │  seed.jsonl                                                                        │
 │  *.safetensors ───────────► compute_g-vendi.py ────────────► G-Vendi Score        │
 │                             (gradient_vendi.py inside)        (printed float)      │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘

 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │  PRISMATIC SYNTHESIS LOOP  (repeats N times)                                      │
 ├────────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                    │
 │  seed.jsonl ──────────────► generate_problem.py ───────────► problems-batch.jsonl │
 │  (few-shot examples)        (Qwen2.5-72B via vLLM)           {problem, level, id}  │
 │                                                                                    │
 │  problems-batch.jsonl ────► generate_solution.py ──────────► batch.jsonl          │
 │                             (Qwen2.5-72B via vLLM)           {prompt, completion,  │
 │                                                                id, level}          │
 │                                                                                    │
 │  batch.jsonl ─────────────► collect_gradients.py ──────────► *.safetensors        │
 │  (candidates)               (Qwen2.5-0.5B proxy)             {id: tensor[1024]}    │
 │                                                                                    │
 │  seed.jsonl (pool)          ┌─────────────────┐                                   │
 │  batch.jsonl (candidates)   │                 │                                   │
 │  *.safetensors (all) ──────►│ cluster_filter  │──────────► new-batch-N.jsonl      │
 │                             │ .py             │            (kept candidates only)  │
 │                             │                 │                                   │
 │                             └────────┬────────┘                                   │
 │                                      │ uses                                       │
 │                             cluster_modules/ ── k-means on pool gradients         │
 │                             gradient_modules/ ── load gradients by sample ID      │
 │                                                                                    │
 │  new-batch-N.jsonl ────────► merge into pool ──────────────► seed+batch1+...jsonl │
 │                              (manual / script)                (grows each cycle)   │
 │                                                                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Class and Method Hierarchy

```
gradient_computer.py
└── GradientComputer
    ├── __init__(model_name, model, tokenizer)
    │     sets up: proj_dim=1024, project_interval=4, save_interval=500
    │     creates: TRAK Rademacher projector (CudaProjector or BasicProjector)
    │
    ├── get_gradient_vector_size(model) ──────── counts trainable params → grad_dim
    ├── get_trak_projector(device) ───────────── tries CudaProjector, falls back to Basic
    ├── prepare_model_input(prompt, completion) ─ chat template + mask prompt in labels
    ├── obtain_gradient(batch) ──────────────── forward+backward → vectorised [grad_dim]
    ├── project_gradients(gradients_dict) ───── batch Rademacher project → [1024] / √proj_dim
    ├── save_projected_gradients(grads, path) ─ write .safetensors + .txt index
    └── compute_project_store_gradients(samples, dir, start_idx)
          └── main loop: obtain → buffer(4) → project → buffer(500) → save

gradient_vendi.py
├── Vendi  (pure math, all static)
│   ├── entropy_q(p, q=1) ─────────── Shannon / Rényi entropy of eigenvalue array
│   ├── compute_reverse_similarity_matrix(data) ── X^T @ X  trick [proj_dim × proj_dim]
│   └── compute_vendi_score(sim_matrix, n) ─────── eigvalsh → entropy → exp(H)
│
└── GradientVendi  (gradient-specific, all static)
    ├── load_all_gradients(dir) ──────────────── load all .safetensors → [N, 1024]
    ├── load_gradients_for_sample_ids(dir, ids) ─ load filtered subset → [N, 1024]
    ├── cluster_kmeans(data, k, num_iter) ──────── cosine-sim Lloyd's algorithm on GPU
    │     └── _calculate_sim_matrix_and_label() ── batched centroid assignment (90 at a time)
    └── compute_gradient_vendi(gradients) ──────── normalise → X^T X → eigvalsh → exp(H)

cluster_filter.py  (prismatic/)
├── filter_cluster(new_grads, ratio, centroids, labels)
│     ├── cosine-assign new samples to nearest centroid
│     ├── ClusterManager.smallest_clusters(labels, ratio=0.5) → sparse cluster IDs
│     └── return [True if in sparse cluster else False]
│
└── __main__
      ├── load pool gradients (GradientManager)
      ├── ClusterManager.cluster_kmeans(pool, k=10%, iter=20)
      ├── load candidate gradients (GradientManager)
      ├── filter_cluster(candidates, ratio=0.5, centroids, labels)
      └── save kept samples

generate_problem.py  (prismatic/)
└── __main__
      ├── VLLMGenerator(Qwen2.5-72B, max_len=8192)
      ├── difficulty-weighted few-shot sampling from pool
      └── batch_prompt_problem(fewshot_samples, num_new=2) × batches
            until target_size=3000 problems generated

cluster_modules/
└── ClusterManager
    ├── cluster_kmeans()          ← wraps GradientVendi.cluster_kmeans
    └── smallest_clusters(labels, ratio) ← Counter(labels) → bottom ratio% IDs

gradient_modules/
└── GradientManager
    └── load_gradients_for_sample_ids() ← mirrors GradientVendi method

generation_modules/
├── VLLMGenerator
│   ├── batch_prompt_problem(fewshot_samples, num_new)
│   └── batch_prompt_solution(problems)
└── generate_model_util
    └── save_to_file(samples, filename, save_mode)
```

---

## 5. Execution Order (One Full Cycle)

```
 Shell scripts drive the pipeline. Each box = one script invocation.

 ┌──────────────────────────────────────────────────────────────────────┐
 │ SETUP (once)                                                         │
 │                                                                      │
 │  collect_seed_set_gradients.sh                                       │
 │  └─► collect_gradients.py ──► gradient_computer.py                  │
 │       (8 GPUs in parallel)         (Qwen2.5-0.5B proxy)             │
 │       writes: seed_gradients/*.safetensors                           │
 └──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ LOOP (repeat for each batch)                                         │
 │                                                                      │
 │  1. generate_problem.sh                                              │
 │     └─► generate_problem.py ──► VLLMGenerator (72B)                 │
 │          writes: generated/problems-batchN.jsonl                     │
 │                          │                                           │
 │                          ▼                                           │
 │  2. generate_solution.sh                                             │
 │     └─► generate_solution.py ──► VLLMGenerator (72B)                │
 │          writes: generated/batchN.jsonl                              │
 │                          │                                           │
 │                          ▼                                           │
 │  3. collect_new_gradient.sh                                          │
 │     └─► collect_gradients.py ──► gradient_computer.py               │
 │          (8 GPUs in parallel)       (Qwen2.5-0.5B proxy)            │
 │          writes: new_gradients/*.safetensors                         │
 │                          │                                           │
 │                          ▼                                           │
 │  4. cluster_filter.sh                                                │
 │     └─► cluster_filter.py                                           │
 │          ├── ClusterManager.cluster_kmeans (pool gradients)          │
 │          ├── GradientManager.load (candidate gradients)              │
 │          └── filter_cluster (keep bottom 50% sparse clusters)        │
 │          writes: complete/new-batch-N.jsonl                          │
 │                          │                                           │
 │                          ▼                                           │
 │         Merge new-batch-N.jsonl into pool → go to step 1            │
 └──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ MEASURE (anytime)                                                    │
 │                                                                      │
 │  compute_g-vendi.py                                                  │
 │  └─► GradientVendi.load_all_gradients()                              │
 │  └─► GradientVendi.compute_gradient_vendi()                          │
 │       └─► Vendi.compute_vendi_score()                                │
 │  prints: G-Vendi = {score}                                           │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Shared Layer — `gradient_vendi.py` as the Common Foundation

```
                         gradient_vendi.py
                         ─────────────────
                    GradientVendi + Vendi classes
                              │
              ┌───────────────┼───────────────────┐
              │               │                   │
              ▼               ▼                   ▼
   compute_g-vendi.py   cluster_modules/    gradient_modules/
   (G-Vendi score)      ClusterManager      GradientManager
                        .cluster_kmeans()   .load_gradients_
                        wraps               for_sample_ids()
                        GradientVendi       mirrors
                        .cluster_kmeans()   GradientVendi
                                            .load_gradients_
                                            for_sample_ids()
```

`gradient_vendi.py` is the **shared math layer** used by both pipelines. The `*_modules/` directories in `prismatic-synthesis/` are thin wrappers that re-expose the same functions under a slightly different interface suited to the synthesis loop's directory conventions.
