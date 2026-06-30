# `gradient_computer.py` — How It Works

**Source:** `g-vendi/gradient_computer.py`
**Role:** The core computation unit. For each training sample it runs a forward+backward pass through the proxy model, extracts the full gradient vector, projects it down to 1024 dimensions, and saves it to disk.

---

## The Big Picture

Everything this file does can be summarised as:

```
(prompt, completion)
       ↓
   forward pass → loss
       ↓
   backward pass → full gradient vector [500M floats]
       ↓
   random projection → compressed gradient [1024 floats]
       ↓
   save to .safetensors
```

The result is a 1024-dimensional unit vector for every training sample — a compact fingerprint of "what direction would this sample push the model's weights."

---

## Class: `GradientComputer`

### `__init__`

```python
def __init__(self, model_name, model, tokenizer):
    self.block_size = 128
    self.projector_batch_size = 16
    self.proj_dim = 1024          # compressed gradient size
    self.project_interval = 4     # project every 4 samples
    self.save_interval = 500      # write to disk every 500 samples
```

Key hyperparameters and what they control:

| Parameter | Value | Effect |
|---|---|---|
| `proj_dim` | 1024 | Size of compressed gradient. Lower = faster, less precise. Higher = more accurate, more memory. |
| `project_interval` | 4 | How many raw gradients to collect before batch-projecting. Projecting in batches is faster than one-by-one. |
| `save_interval` | 500 | How many projected gradients to buffer before writing to disk. Prevents holding everything in GPU memory. |
| `block_size` | 128 | Internal TRAK projector block size for CUDA kernel efficiency. |

After setting hyperparameters, the constructor computes `grad_dim` (total number of trainable parameters) and creates the **TRAK projector** — the random matrix that does the compression.

---

### `get_gradient_vector_size`

```python
@staticmethod
def get_gradient_vector_size(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
```

Counts every trainable parameter in the proxy model. For Qwen2.5-0.5B this is ~500 million. This becomes the input dimension for the random projection matrix.

The `if p.requires_grad` filter is important — if LoRA adapters are used, only the adapter parameters count, not the frozen base model parameters. This dramatically shrinks the gradient vector when LoRA is active.

---

### `get_trak_projector`

```python
@staticmethod
def get_trak_projector(device):
    try:
        import fast_jl
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        return CudaProjector
    except RuntimeError:
        return BasicProjector
```

Tries to use the fast CUDA implementation (`CudaProjector` via the `fast_jl` library). Falls back to `BasicProjector` (pure PyTorch) if the CUDA kernel is unavailable or fails on this GPU. The CUDA version is significantly faster for large gradient vectors.

---

### `obtain_gradient`

```python
def obtain_gradient(self, batch):
    self.model.zero_grad()
    loss = self.model(**batch).loss
    loss.backward()

    vectorized_gradient = torch.cat(
        [p.grad.view(-1) for n, p in self.model.named_parameters() if p.grad is not None],
        dim=0
    )
    return vectorized_gradient
```

The core backprop step:

1. **`zero_grad()`** — clear any leftover gradients from previous samples. Critical: without this, gradients accumulate across samples.
2. **`model(**batch).loss`** — forward pass. The batch contains `input_ids`, `attention_mask`, and `labels`. The loss is computed only on `labels` tokens (the completion), not the prompt — see `prepare_model_input` below.
3. **`loss.backward()`** — backprop. PyTorch fills `p.grad` for every trainable parameter.
4. **`torch.cat([p.grad.view(-1) ...], dim=0)`** — flatten every parameter's gradient and concatenate them into a single long vector. The order is deterministic (same as `named_parameters()` iteration), so sample comparisons are meaningful.

**Output:** a single float tensor of shape `[grad_dim]` — the full gradient vector for this sample.

---

### `project_gradients`

```python
def project_gradients(self, gradients: Dict[str, torch.Tensor]):
    sample_ids = list(gradients.keys())
    gradients = torch.stack([gradients[s] for s in sample_ids]).to(torch.float16)
    projected = self.projector.project(gradients, model_id=0) / np.sqrt(self.proj_dim)
    return {k: v for k, v in zip(sample_ids, projected)}
```

Takes a dict of `{sample_id: full_gradient_vector}` and compresses them all at once.

**Steps:**
1. Stack all gradient vectors into a batch matrix `[batch_size, grad_dim]`
2. Cast to float16 — the TRAK projector works in fp16 for speed
3. `self.projector.project(gradients, model_id=0)` — applies the Rademacher random matrix, outputting `[batch_size, proj_dim=1024]`
4. Divide by `sqrt(proj_dim)` — the standard Johnson-Lindenstrauss normalisation. This ensures that the expected value of the dot product `(Rg_i) · (Rg_j)` equals the original dot product `g_i · g_j`. Without this, the projected dot products would be systematically inflated by a factor of `proj_dim`.

**Why batch projection?** Applying the random projection matrix to 4 gradients at once is faster than applying it 4 times individually — GPU matrix multiply is more efficient when given larger batches. The `project_interval=4` setting controls this batch size.

---

### `prepare_model_input`

```python
def prepare_model_input(self, prompt: str, completion: str):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    encoding = self.tokenizer.apply_chat_template(messages, return_dict=True, return_tensors="pt")
    encoding['labels'] = self.collator(encoding['input_ids'])['labels']
    return encoding
```

Formats input for a causal LM fine-tuning loss:

- **`apply_chat_template`** — wraps prompt+completion in the model's chat format (e.g. `<|im_start|>user\n...<|im_start|>assistant\n...`)
- **`DataCollatorForCompletionOnlyLM`** — masks the prompt tokens in `labels` with `-100` (PyTorch's ignore index). This means the cross-entropy loss is only computed over the completion tokens.

**Why mask the prompt?** The gradient should reflect what the model learns from the *answer*, not from re-reading the question. The question tokens are given context, not supervised signal. Using loss only on completion tokens gives a gradient that captures "how would this answer change the model" rather than "how would this question change the model."

---

### `compute_project_store_gradients` — The Main Loop

```python
def compute_project_store_gradients(self, samples, save_directory, global_start_idx):
    full_grads = {}
    projected_grads = {}
    
    for sample_idx, sample in enumerate(samples):
        encoding = self.prepare_model_input(sample["prompt"], sample["completion"])
        sample_full_grad = self.obtain_gradient(encoding)
        full_grads[sample["id"]] = sample_full_grad

        # batch-project every `project_interval` samples
        if (sample_idx + 1) % self.project_interval == 0:
            projected_grads.update(self.project_gradients(full_grads))
            full_grads = {}

        # save to disk every `save_interval` samples
        if (sample_idx + 1) % self.save_interval == 0:
            save_filename = Path(save_directory) / f"{global_start_idx + sample_idx + 1 - len(projected_grads)}.safetensors"
            self.save_projected_gradients(projected_grads, save_filename)
            projected_grads = {}

    # flush remaining
    if len(full_grads) > 0:
        projected_grads.update(self.project_gradients(full_grads))
    self.save_projected_gradients(projected_grads, save_filename)
```

Two interleaved buffers:
- `full_grads`: accumulates raw (uncompressed) gradients. Flushed to projector every 4 samples.
- `projected_grads`: accumulates compressed gradients. Flushed to disk every 500 samples.

**Memory flow:**
```
sample 1  → full_grads (500M floats in GPU memory)
sample 2  → full_grads
sample 3  → full_grads
sample 4  → full_grads → project → projected_grads (4 × 1024 floats) → full_grads cleared
...
sample 500 → projected_grads → save .safetensors → projected_grads cleared
```

Raw gradients are large (500M × fp32 = 2GB per sample); they're only kept for 4 samples at a time. Compressed gradients are tiny (1024 × fp16 = 2KB per sample); 500 of them fit easily in memory.

---

### `save_projected_gradients`

```python
@staticmethod
def save_projected_gradients(projected_gradients, save_filename):
    save_file(projected_gradients, save_filename)            # → .safetensors
    
    sample_ids = list(projected_gradients.keys())
    data = [json.dumps({"id": sample_id}) for sample_id in sample_ids]
    with open(save_filename.with_suffix(".txt"), "w") as f:
        f.write("\n".join(data) + "\n")
```

Writes two files for every batch of 500 samples:
- **`.safetensors`** — the compressed gradient tensors, keyed by sample ID. Safetensors is a fast, safe format for tensor storage (no pickle, memory-mappable).
- **`.txt`** — a JSONL index of sample IDs in this file. Used by `collect_gradients.py` to detect which samples have already been processed (for resumable computation).

The filename is the global start index of the first sample in this file, enabling `collect_gradients.py` to reconstruct where computation left off.

---

## Data Flow Summary

```
samples (JSONL)
    │
    ▼
prepare_model_input()   → {input_ids, attention_mask, labels}  (prompt masked in labels)
    │
    ▼
obtain_gradient()       → gradient vector [grad_dim ≈ 500M]    (full backprop)
    │  (buffer 4 at a time)
    ▼
project_gradients()     → compressed vector [1024]              (Rademacher projection / √proj_dim)
    │  (buffer 500 at a time)
    ▼
save_projected_gradients() → {sample_id: tensor}.safetensors + index.txt
```

---

## Dependencies

| Library | Used for |
|---|---|
| `trak` | Rademacher random projection (`CudaProjector`, `BasicProjector`) |
| `safetensors` | Fast tensor serialisation to disk |
| `trl` | `DataCollatorForCompletionOnlyLM` — prompt masking in labels |
| `transformers` | `AutoModelForCausalLM`, `PreTrainedModel` |
| `fast_jl` | CUDA-accelerated Johnson-Lindenstrauss projection (optional) |
