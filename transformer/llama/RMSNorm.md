# LlamaRMSNorm — explained

```python
@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```

This normalizes a token's hidden-state vector to a consistent magnitude before each attention/MLP sub-layer (the `self.input_layernorm` call referenced in `DecoderLayer.md`).

## `super().__init__()`

`LlamaRMSNorm` inherits from `nn.Module`. `super().__init__()` calls `nn.Module`'s own constructor before doing anything else. This is mandatory boilerplate for any custom PyTorch module — `nn.Module.__init__` sets up internal bookkeeping dictionaries (`_parameters`, `_buffers`, `_modules`, hooks, etc.) that PyTorch relies on to track registered parameters/buffers/submodules, move them across devices (`.to(device)`), and include them in `.state_dict()`. Skipping this call and trying `self.weight = nn.Parameter(...)` afterward would actually error, since `nn.Module.__setattr__` (which intercepts parameter/buffer registration) depends on that bookkeeping already existing.

So: nothing RMSNorm-specific happens here — it's "initialize me as a proper PyTorch module before adding my own state."

## `@use_kernel_forward_from_hub("RMSNorm")`

A decorator that lets this module's `forward` be swapped out for an optimized kernel pulled from the Hugging Face Hub's kernel registry — the same pattern as `@use_kernel_func_from_hub("rotary_pos_emb")` on `apply_rotary_pos_emb` (see `RoPE.md`), except here it decorates a whole class rather than a standalone function.

- `"RMSNorm"` is the registry key — it tells the mechanism which kernel implementation to look up (e.g. a fused CUDA/Triton RMSNorm kernel optimized for the current hardware/dtype, instead of running plain PyTorch ops for `mean(x^2)`, `rsqrt`, multiply).
- At call time, if a matching faster kernel is available and applicable, `forward` is transparently replaced by that kernel's implementation; otherwise it falls back to the pure-PyTorch `forward` written in this class.
- This doesn't change the math RMSNorm computes — purely a performance optimization hook.

## The rest of `__init__`

```python
self.weight = nn.Parameter(torch.ones(hidden_size))
self.variance_epsilon = eps
```
- `self.weight`: a **learnable** scale vector, one value per dimension of `hidden_size`, initialized to all `1.0`s. Because it's an `nn.Parameter`, it's registered automatically (thanks to `super().__init__()`) and updated by gradient descent during training. Starting at `1.0` means at initialization, RMSNorm doesn't shrink or grow anything — it starts as a no-op scale, and training adjusts it per-dimension as needed.
- `self.variance_epsilon = eps`: a small constant (`1e-6`) added before taking a square root, purely to avoid dividing by zero if variance happens to be ~0.

## `forward` — the actual math

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
```

**1. `input_dtype = hidden_states.dtype` / `hidden_states.to(torch.float32)`** — remembers the original dtype (e.g. `bfloat16`), then upcasts to `float32` for the actual normalization math. Squaring and averaging in low precision can lose accuracy or overflow/underflow, so the numerically sensitive part is forced to run in float32 — the same precision-safety pattern seen in `RoPE.md`'s `forward`.

**2. `variance = hidden_states.pow(2).mean(-1, keepdim=True)`** — for each token, square every value in its `hidden_size`-wide vector, then average them. This is the **mean square** (note: not centered — RMSNorm doesn't subtract the mean like LayerNorm does, hence "Root Mean Square" rather than "standard deviation"). `keepdim=True` keeps the reduced dimension as size-1 (shape `[..., 1]`) instead of dropping it, so it can broadcast back against the original tensor in the next step.

#### Breaking this line down further

- **`hidden_states.pow(2)`** — squares every single element elementwise. If `hidden_states` has shape `[batch, seq_len, hidden_size]`, the output has the same shape, just every value squared (and now non-negative).
- **`.mean(-1, keepdim=True)`** — averages along the **last** dimension (`hidden_size`): for each token, collapse its `hidden_size`-wide vector of squared values down to a single number, the average. `-1` means "the last axis," regardless of how many leading axes (`batch`, `seq_len`, etc.) exist.
- **Why `keepdim=True` matters:** without it, averaging over the last axis would *remove* that axis entirely (`[batch, seq_len, hidden_size]` → `[batch, seq_len]`). With `keepdim=True`, the axis is kept but shrunk to size 1 (`[batch, seq_len, 1]`). This is required for the very next line, `hidden_states * torch.rsqrt(variance + eps)`: `hidden_states` is `[batch, seq_len, hidden_size]` and `variance` is `[batch, seq_len, 1]` — these shapes broadcast cleanly (the size-1 axis repeats across all `hidden_size` positions). If `keepdim` were `False`, `variance` would be `[batch, seq_len]`, which does **not** broadcast against `[batch, seq_len, hidden_size]` and would error.

**Concrete example:** say `hidden_size = 4`, and one token's vector (already upcast to float32) is `[2.0, -2.0, 4.0, -4.0]`.

Square each element: `[4.0, 4.0, 16.0, 16.0]`. Average them: `(4.0+4.0+16.0+16.0)/4 = 10.0`. So `variance = 10.0` for this token.

It's called `variance` here, but it's really the mean of squares, not the statistical variance (which would center first: `mean((x - mean(x))²)`). RMSNorm deliberately skips centering — it only cares about overall magnitude/scale, not whether the vector is shifted up or down. That's exactly why this is "Root Mean **Square**" Norm, not Standard Deviation Norm.

This `variance` then feeds into `torch.rsqrt(variance + eps)` — here, `1/sqrt(10.0 + 1e-6) ≈ 0.316` — the scaling factor multiplied back into every element of `hidden_states` to normalize the vector's magnitude to ≈1 RMS.

**3. `hidden_states * torch.rsqrt(variance + eps)`** — `rsqrt(v) = 1/sqrt(v)`. Multiplying each value by `1/sqrt(mean_square + eps)` rescales the whole vector so its root-mean-square becomes ≈1. This is the "normalize" step: it controls the *magnitude* of activations without touching their *direction* or relative proportions.

**4. `self.weight * hidden_states.to(input_dtype)`** — cast back to the original dtype, then apply the learnable per-dimension scale. This lets the model learn to amplify or dampen specific dimensions after normalization, rather than being stuck with magnitude exactly 1 everywhere.

## `extra_repr`

```python
def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```
Purely cosmetic — controls what shows up when you `print(model)`, displaying the weight shape and epsilon value in the module's string representation. Doesn't affect computation.

## Why "equivalent to T5LayerNorm"

The docstring notes RMSNorm is equivalent to T5's LayerNorm because T5 already used a simplified LayerNorm that skips mean-centering (no `x - mean(x)` step) — just like this implementation. Standard LayerNorm (e.g. in BERT/GPT-2) centers *and* scales; RMSNorm only scales, which is cheaper to compute and works well in practice for transformer hidden states.
