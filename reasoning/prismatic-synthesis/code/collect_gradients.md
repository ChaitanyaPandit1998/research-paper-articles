# `collect_gradients.py` — How It Works

**Source:** `g-vendi/collect_gradients.py`
**Role:** The orchestrator. Distributes gradient computation across multiple GPUs, handles resumable checkpointing, and calls `GradientComputer` on the right slice of data for each GPU.

---

## The Big Picture

This file solves the logistics problem: given a dataset with potentially millions of samples and 8 GPUs, how do you distribute the work, avoid duplication, and resume cleanly if a GPU crashes mid-run?

```
Dataset (1M samples)
        │
        ▼
   Split into 8 slices (one per GPU)
        │
        ▼
   Each GPU: find where it left off (resume from checkpoint)
        │
        ▼
   Load proxy model on this GPU
        │
        ▼
   GradientComputer.compute_project_store_gradients(my_slice)
        │
        ▼
   Saves to: ./data/gradient_storage/{dataset}--{model}/
```

---

## `parse_args`

```python
args.save_directory = (Path("./data/gradient_storage") /
                       f"{args.dataset_filename.stem}--{args.model_name_or_path.split('/')[-1].lower()}")
```

The save directory name encodes both the dataset and the proxy model — so gradients from different proxy models are stored separately and never mixed. Example: `train--qwen2.5-0.5b-instruct`.

After setting up the directory, immediately calls `find_start_and_end_idx` — the resume logic.

---

## `find_start_and_end_idx` — The Resume Logic

This is the most complex function in the file. It answers: "given that this GPU has already computed some gradients in a previous run, where should it start today?"

```python
def find_start_and_end_idx(args):
    cuda_visible_device = int(os.getenv("CUDA_VISIBLE_DEVICES"))   # which GPU am I?

    # each GPU gets 1/8 of the dataset
    per_device_size = len(dataset) / 8
    start_idx = cuda_visible_device * per_device_size
    end_idx   = start_idx + per_device_size
```

**Step 1: Assign slice.** GPU 0 gets samples 0–124999, GPU 1 gets 125000–249999, etc.

**Step 2: Check what's already done.**

The `.txt` index files written by `GradientComputer.save_projected_gradients` act as progress markers. Each `.txt` file is named with the global index of the first sample it contains. By scanning all `.txt` filenames in the save directory that fall within this GPU's range:

```python
precomputed_start_indices = [
    int(filename.stem) for filename in save_directory.glob("*.txt")
    if start_idx <= int(filename.stem) < end_idx
]
```

**Step 3: Find the last computed sample.**

```python
last_filename = save_directory / f"{max(precomputed_start_indices)}.txt"
# read the sample IDs inside this file
last_ids = [s["id"] for s in jsonlines.open(last_filename)]
# find their indices in the full dataset
last_ids_indices = [dataset_sample_ids.index(sid) for sid in last_ids]
last_idx = max(last_ids_indices)
start_idx = last_idx + 1   # resume from here
```

If no `.txt` files exist for this GPU's range, it starts from the beginning of its slice.

**Why use `.txt` files as checkpoints rather than counting `.safetensors` files?**
Each `.safetensors` file contains 500 samples (the `save_interval`). The `.txt` companion file lists exactly which sample IDs are in it. This allows precise recovery — even if the last `.safetensors` write was partial, the `.txt` file reflects the actual written content.

---

## `get_model_and_tokenizer`

```python
def get_model_and_tokenizer(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto").cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer
```

- **`torch_dtype="auto"`** — loads in the model's native dtype (bfloat16 for Qwen2.5), avoiding unnecessary fp32 casting
- **`pad_token_id = eos_token_id`** — Qwen2.5 doesn't have a dedicated pad token; using EOS as padding is standard
- **`padding_side = "left"`** — for decoder-only models, left padding ensures the generation suffix always starts at the same position in the batch

---

## `get_dataset`

```python
def get_dataset(filename, start_idx, end_idx):
    with jsonlines.open(filename) as f:
        samples = list(f)[start_idx:end_idx]
    assert all(key in samples[0] for key in ["prompt", "completion", "id"])
    return samples
```

Simple slice load. Validates that every sample has the three required fields: `prompt`, `completion`, and `id`. The `id` field is critical — it's the key used to associate gradients with samples across all downstream steps.

---

## `__main__` — Tying It Together

```python
args = parse_args()                                         # setup + resume detection
model, tokenizer = get_model_and_tokenizer(model_name)     # load proxy model on this GPU
samples = get_dataset(filename, args.start_idx, args.end_idx)  # load this GPU's slice

collector = GradientComputer(model_name, model, tokenizer)
collector.compute_project_store_gradients(samples, args.save_directory, args.start_idx)
```

Clean and linear. All the complexity is hidden inside `parse_args` (resume logic) and `GradientComputer` (the actual computation).

---

## Multi-GPU Execution Pattern

The script is designed to be launched once per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python collect_gradients.py --model_name_or_path ... --dataset_filename ...
CUDA_VISIBLE_DEVICES=1 python collect_gradients.py --model_name_or_path ... --dataset_filename ...
...
CUDA_VISIBLE_DEVICES=7 python collect_gradients.py --model_name_or_path ... --dataset_filename ...
```

Each process reads `CUDA_VISIBLE_DEVICES` to know which GPU it is, which determines which slice of the dataset it processes. The 8 processes run independently — no inter-process communication needed, since each writes to different files.

**Default split:** `--device_split_size 8`. Change this to match the number of available GPUs.

---

## File Naming Convention

The save directory contains files named by the global start index of their contents:

```
gradient_storage/train--qwen2.5-0.5b-instruct/
├── 0.safetensors       ← samples 0–499    (GPU 0's first batch)
├── 0.txt               ← index for above
├── 500.safetensors     ← samples 500–999
├── 500.txt
├── 125000.safetensors  ← samples 125000–125499  (GPU 1's first batch)
├── 125000.txt
...
```

This naming allows any downstream code to load all `.safetensors` files from the directory and reconstruct the complete gradient dataset without needing a separate manifest file.

---

## Data Flow

```
CUDA_VISIBLE_DEVICES=N
        │
        ▼
parse_args() → determine slice [start_idx, end_idx] for GPU N
        │       check .txt files → find last computed sample → set new start_idx
        ▼
get_model_and_tokenizer() → proxy model on GPU N
        │
        ▼
get_dataset() → samples[start_idx:end_idx] from JSONL
        │
        ▼
GradientComputer.compute_project_store_gradients()
        │
        ▼
{start_idx}.safetensors + {start_idx}.txt  (written every 500 samples)
```
