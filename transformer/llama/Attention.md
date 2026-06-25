# Llama Attention: `repeat_kv` and `eager_attention_forward` — explained

These run immediately after `apply_rotary_pos_emb` (see `RoPE.md`): the rotated `q_embed`/`k_embed` become `query`/`key` here. RoPE rotation happens *before* this code — `eager_attention_forward` doesn't know or care that `q`/`k` were rotated; it just performs standard scaled-dot-product attention. The relative-position property RoPE provides (`q·k ≈ cos(gap·θ)`) is already baked into `query`/`key` by the time they arrive here.

## KV cache update and picking the attention backend

These two lines run inside `LlamaAttention.forward`, right after `query`/`key` have been rotated and right before the actual attention computation:

```python
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
```

### `past_key_values.update(...)` — the KV cache

During autoregressive generation, you don't recompute K/V for every previously-generated token on every step — that would be wasteful. Instead, `past_key_values` is a cache object that stores K/V tensors from all previous forward passes.

- `past_key_values.update(key_states, value_states, self.layer_idx)`:
  - Appends this step's freshly computed `key_states`/`value_states` (just for the new token(s)) onto whatever was already cached for *this specific layer* (`self.layer_idx` — each decoder layer has its own cache slot, since each layer computes different K/V).
  - Returns the **full** K/V sequence so far (old + new), not just the new piece. So after this line, `key_states`/`value_states` cover every token generated up to now, even though the model in this forward pass only just computed K/V for the newest token(s).
- If there's no cache (`past_key_values is None` — e.g. plain training or a one-shot forward pass with no generation), this step is skipped entirely and `key_states`/`value_states` are simply whatever was just computed for the full input sequence.

This is why decoding is fast: layer `i`'s attention only ever needs to do work proportional to the new token(s), while still attending over the entire growing history via the cache.

### `attention_interface` — picking which attention implementation to run

`transformers` supports multiple backends for actually computing attention — the plain PyTorch `eager_attention_forward` documented below, but also faster kernels like FlashAttention-2, SDPA (PyTorch's fused `scaled_dot_product_attention`), or custom Hub kernels.

- `self.config._attn_implementation` is a string set on the model config (e.g. `"eager"`, `"sdpa"`, `"flash_attention_2"`), chosen either by the user (`attn_implementation=...` at model load time) or auto-detected based on what's installed/supported on the current hardware.
- `ALL_ATTENTION_FUNCTIONS` is a registry (dict-like) mapping those implementation-name strings to their corresponding callables.
- `.get_interface(name, default)`: looks up the function registered under `name`; if that name isn't found in the registry, falls back to `eager_attention_forward` (the default arg) — i.e. eager is always the safe fallback.
- The result, `attention_interface`, is just a function reference — assigned but not yet called here. The next line (not shown) would call it: `attention_interface(self, query, key, value, attention_mask, scaling=..., ...)`, matching the exact signature of `eager_attention_forward`.

So this is a pluggable-backend pattern: every attention backend implements the same function signature, and this line swaps in whichever one is fastest/available without changing any other code in `LlamaAttention.forward`.

## `repeat_kv` — why it exists: Grouped Query Attention (GQA)

In GQA, the number of **key/value heads** (`num_key_value_heads`) is smaller than the number of **query heads** (`num_attention_heads`) — multiple query heads share the same K/V head, to save memory/compute. `n_rep = num_attention_heads // num_key_value_heads` is how many query heads share each K/V head.

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

- `batch, num_key_value_heads, slen, head_dim = hidden_states.shape` — unpack shape, e.g. `[batch, num_kv_heads, slen, head_dim]`.
- `if n_rep == 1: return hidden_states` — standard multi-head attention (no grouping); nothing to do.
- `hidden_states[:, :, None, :, :]` inserts a new size-1 axis right after the heads dimension: shape goes from `[batch, num_kv_heads, slen, head_dim]` to `[batch, num_kv_heads, 1, slen, head_dim]`.
- `.expand(batch, num_key_value_heads, n_rep, slen, head_dim)` broadcasts that size-1 axis to size `n_rep` *without copying memory* — each of the `n_rep` "slots" is a view onto the same underlying K/V head.
- `.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)` collapses the `[num_kv_heads, n_rep]` axes into one axis of size `num_kv_heads * n_rep = num_attention_heads`. The result: each K/V head has been duplicated `n_rep` times consecutively, so it now lines up 1-to-1 with query heads.

This is functionally identical to `torch.repeat_interleave(x, dim=1, repeats=n_rep)` (per the docstring), but expand+reshape avoids `repeat_interleave`'s extra copy in some cases — though `.reshape` after `.expand` does force a copy here since the expanded dim isn't contiguous.

**Example:** `num_kv_heads=2`, `n_rep=4` → query heads 0,1,2,3 all read from KV-head 0; query heads 4,5,6,7 all read from KV-head 1.

## `eager_attention_forward` — the actual attention computation

```python
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

**Step 1 — expand K/V to match query heads:**
```python
key_states = repeat_kv(key, module.num_key_value_groups)
value_states = repeat_kv(value, module.num_key_value_groups)
```
Every query head now has its own matching K/V head (even though under the hood several query heads share the same underlying data).

**Step 2 — raw attention scores:**
```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
```
- `key_states.transpose(2, 3)`: swaps `seqlen` and `head_dim` axes, turning K from `[batch, heads, seqlen, head_dim]` into `[batch, heads, head_dim, seqlen]` so it can be matrix-multiplied against `query`.
- `query @ key_states.T`: for every pair of positions `(i, j)`, computes the dot product `q_i · k_j` — the raw, unnormalized attention score. Shape: `[batch, heads, seqlen_q, seqlen_k]`.
- `* scaling`: typically `1/sqrt(head_dim)`, the standard scaled-dot-product-attention scaling to keep gradients stable (without it, dot products grow with `head_dim` and softmax saturates).

**Step 3 — apply mask:**
```python
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
```
Adds the mask (e.g. causal mask: positions can't attend to future tokens) as large negative numbers (`-inf`-ish) at disallowed positions, so after softmax their probability becomes ~0.

**Step 4 — normalize to probabilities:**
```python
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```
Softmax over the last axis (`seqlen_k`) turns raw scores into a probability distribution — for each query position, how much attention to pay to each key position.

### Why float32? The overflow problem

Models run in **bfloat16** by default — 16-bit floats with only 7 bits of mantissa precision and a maximum representable value of ~38,912. The softmax formula is:

```
softmax(x_i) = e^(x_i) / sum(e^(x_j))
```

The danger is the `e^x` step. Attention scores can be large, especially for long sequences or early in training:

```
If one score = 89.0:
  e^89 ≈ 4.5 × 10^38

bfloat16 max ≈ 3.4 × 10^38

→ e^89 overflows to inf in bfloat16
→ softmax output becomes nan
→ nan propagates through the entire forward pass silently
```

**float32** can represent up to ~3.4 × 10^38 with 23 bits of mantissa, so it handles these exponentials safely. Running softmax in float32 eliminates the overflow risk entirely.

### What the single line actually does in sequence

```python
# 1. upcast attn_weights from bfloat16 → float32  (dtype=torch.float32 argument)
# 2. run softmax in float32                        (safe from overflow)
# 3. cast probabilities back to bfloat16           (.to(query.dtype))
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

### Why cast back to bfloat16 after?

The probabilities produced by softmax are all between 0 and 1 — no overflow risk. The next operation (`attn_weights @ value_states`) is a plain weighted sum of value vectors, which bfloat16 handles safely. Staying in float32 for the matmul would use 2× the memory and lose the speed advantage of bfloat16 tensor cores (e.g. H100 runs bfloat16 matmuls much faster than float32). The upcast/downcast is a precision firewall around the one dangerous operation, not a wholesale dtype upgrade.

**Step 5 — dropout:**
```python
attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
```
Standard dropout regularization on the attention weights, only active during training (`module.training`).

**Step 6 — weighted sum of values:**
```python
attn_output = torch.matmul(attn_weights, value_states)
```
Weighted sum of value vectors: for each query position, blend together all value vectors according to the attention probabilities. Shape: `[batch, heads, seqlen_q, head_dim]`.

**Step 7 — reshape back:**
```python
attn_output = attn_output.transpose(1, 2).contiguous()
```
Swaps `heads` and `seqlen` axes back to `[batch, seqlen, heads, head_dim]` (the layout expected downstream, before heads get merged back into one `hidden_size` dimension). `.contiguous()` forces a real memory copy after the transpose, since later ops (like `.view`/`.reshape`) require contiguous memory.

**Step 8 — return:**
```python
return attn_output, attn_weights
```
Returns the actual attention output (fed forward through the rest of the layer) and the raw attention weights (useful for visualization/analysis, e.g. `output_attentions=True`).

## Merging heads back: `attn_output.reshape(*input_shape, -1).contiguous()`

This runs back in `LlamaAttention.forward`, right after the chosen attention backend (eager/SDPA/FlashAttention-2) returns `attn_output`. It turns the per-head attention output back into the model's standard `[batch, seq_len, hidden_size]` shape.

```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
```

Going into this line, `attn_output` has shape `[batch, seq_len, num_heads, head_dim]` (heads were already moved back into this order by `.transpose(1, 2)` inside `eager_attention_forward`, Step 7 above). `input_shape` is `(batch, seq_len)`, captured earlier in `forward` from the original hidden-states input — so `*input_shape, -1` unpacks to `(batch, seq_len, -1)`, and the `-1` tells PyTorch to infer that last axis by flattening `[num_heads, head_dim]` into one axis of size `num_heads * head_dim == hidden_size`.

### Concrete example

Say `batch=1`, `seq_len=2` (2 tokens, e.g. "cat", "sat"), `num_heads=2`, `head_dim=2`, so `hidden_size = 2*2 = 4`.

After attention, `attn_output` (shape `[1, 2, 2, 2]`) holds, for each token, 2 heads each producing a 2-number output:

```
attn_output =
  token "cat":  head0 = [1, 2]    head1 = [3, 4]
  token "sat":  head0 = [5, 6]    head1 = [7, 8]
```

`attn_output.reshape(1, 2, -1)` flattens the last two axes (`num_heads, head_dim` → `2*2=4`) while keeping `batch` and `seq_len` untouched. For "cat", `head0=[1,2]` and `head1=[3,4]` concatenate into one flat vector `[1,2,3,4]`. Same for "sat" → `[5,6,7,8]`.

Result, shape `[1, 2, 4]`:
```
[[ [1,2,3,4],     # "cat" — single hidden_size=4 vector
   [5,6,7,8] ]]   # "sat" — single hidden_size=4 vector
```

Each token went from "2 heads × 2 numbers each" to "1 vector of 4 numbers" — exactly `hidden_size`. This undoes the head-splitting done earlier when `hidden_size` was split into `num_heads × head_dim` for the Q/K/V projections.

**Why `.contiguous()` after:** the reshape here forces a real data copy (heads/head_dim aren't the last contiguous memory block after the earlier `.transpose`), and `.contiguous()` guarantees the result is laid out contiguously in memory — required because the very next step (`self.o_proj(attn_output)`, a linear layer) needs contiguous memory for an efficient matmul.
