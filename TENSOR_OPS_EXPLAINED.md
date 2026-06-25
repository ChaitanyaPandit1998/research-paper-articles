# PyTorch Tensor Shape Operations

Practical examples for `transpose`, `view`, `reshape`, and `contiguous`.

---

## 1. `transpose`

Swaps exactly two dimensions. Does **not** move data in memory — only changes how the tensor is indexed (non-contiguous result).

### Basic swap

```python
import torch

x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])   # shape: [2, 3]

x.transpose(0, 1)
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
# shape: [3, 2]
```

### Swapping inner dims of a 4-D tensor (attention use case)

```python
# After splitting hidden into heads: [batch, seq, heads, head_dim]
x = torch.zeros(2, 10, 8, 64)   # [batch=2, seq=10, heads=8, head_dim=64]

x = x.transpose(1, 2)           # swap dim-1 (seq) and dim-2 (heads)
print(x.shape)                   # torch.Size([2, 8, 10, 64])
#                                    [batch, heads, seq, head_dim]  ← attention-ready
```

### Transposing K for Q @ K^T (dot-product attention)

```python
Q = torch.zeros(2, 8, 10, 64)   # [batch, heads, seq_q, head_dim]
K = torch.zeros(2, 8, 10, 64)   # [batch, heads, seq_k, head_dim]

scores = Q @ K.transpose(-2, -1)
print(scores.shape)              # torch.Size([2, 8, 10, 10])
#                                    [batch, heads, seq_q, seq_k]
```

`transpose(-2, -1)` is a common idiom: swap the last two dims regardless of total rank.

---

## 2. `view`

Reshapes a tensor by merging or splitting dimensions. **Requires the tensor to be contiguous in memory.** Raises a `RuntimeError` otherwise.

### Split one dim into two

```python
x = torch.arange(24)            # shape: [24]

x = x.view(4, 6)                # split into 4 rows, 6 columns
print(x.shape)                   # torch.Size([4, 6])

x = x.view(2, 3, 4)             # split into [2, 3, 4]
print(x.shape)                   # torch.Size([2, 3, 4])
```

### Merge two dims into one

```python
x = torch.zeros(2, 8, 10, 64)   # [batch, heads, seq, head_dim]

# Merge heads and head_dim → hidden
x = x.view(2, 8, 10 * 64)
print(x.shape)                   # torch.Size([2, 8, 640])
```

### Using `-1` to infer a dim

```python
x = torch.zeros(2, 8, 10, 64)

x = x.view(2, -1)               # -1 means "figure it out": 8*10*64 = 5120
print(x.shape)                   # torch.Size([2, 5120])
```

### When `view` fails

```python
x = torch.zeros(2, 8, 10, 64)
x = x.transpose(1, 2)           # now non-contiguous
x.view(2, 10, 512)               # RuntimeError: view size is not compatible ...
                                 # Fix: call .contiguous() first (see section 4)
```

---

## 3. `reshape`

Same as `view` — but works on **non-contiguous** tensors too. If the tensor is already contiguous it returns a view (no copy); otherwise it returns a copy with rearranged memory.

### Same syntax as view

```python
x = torch.arange(24)
x = x.reshape(4, 6)
print(x.shape)                   # torch.Size([4, 6])
```

### Works after transpose (unlike view)

```python
x = torch.zeros(2, 8, 10, 64)
x = x.transpose(1, 2)           # non-contiguous, shape: [2, 10, 8, 64]

x = x.reshape(2, 10, 512)       # succeeds — makes a copy internally if needed
print(x.shape)                   # torch.Size([2, 10, 512])
```

### Merging heads back after attention (full sequence)

```python
batch, heads, seq, head_dim = 2, 8, 10, 64

out = torch.zeros(batch, heads, seq, head_dim)  # after attention
out = out.transpose(1, 2)                        # [2, 10, 8, 64]
out = out.reshape(batch, seq, heads * head_dim)  # [2, 10, 512]
print(out.shape)                                  # torch.Size([2, 10, 512])
```

---

## 4. `contiguous`

Returns a tensor with its data laid out sequentially in memory (C-contiguous order). After operations like `transpose` or `permute`, the tensor's internal strides are rearranged but the data is not moved — the tensor is then non-contiguous. Some operations (notably `view`) require contiguous memory.

### Checking contiguity

```python
x = torch.zeros(3, 4)
print(x.is_contiguous())         # True

y = x.transpose(0, 1)
print(y.is_contiguous())         # False  ← data not moved, strides changed
```

### Making a tensor contiguous

```python
x = torch.zeros(2, 8, 10, 64)
x = x.transpose(1, 2)           # non-contiguous

x = x.contiguous()              # copies data into a new contiguous block
print(x.is_contiguous())         # True

x = x.view(2, 10, 512)          # now safe to use view
print(x.shape)                   # torch.Size([2, 10, 512])
```

### What are strides?

Strides tell PyTorch how many elements to skip in memory to move one step along each dimension.

```python
x = torch.zeros(3, 4)
print(x.stride())                # (4, 1)  — move 4 elements to go to next row,
                                 #            move 1 element to go to next column

y = x.transpose(0, 1)
print(y.stride())                # (1, 4)  — strides swapped, data unchanged
print(y.is_contiguous())         # False
```

After `.contiguous()`, strides are reset to the natural C-order `(cols, 1)`.

---

## When to Use What

| Need | Use |
|---|---|
| Swap two axes | `transpose` or `permute` |
| Merge / split dims, tensor is contiguous | `view` (zero-copy, fastest) |
| Merge / split dims, tensor may not be contiguous | `reshape` (handles both cases) |
| Make non-contiguous tensor contiguous before `view` | `contiguous` |
| Both reorder + merge | `permute` / `transpose` → then `reshape` or `contiguous` + `view` |

### `view` vs `reshape` — which to prefer?

- Use `view` when you know the tensor is contiguous and you want to be explicit that no copy occurs.
- Use `reshape` when you don't want to worry about contiguity — it does the right thing automatically.
- In transformer code, `reshape` after `transpose` is the most common pattern because `transpose` always leaves the tensor non-contiguous.

---

## Full Attention Example (all four ops together)

```python
import torch

batch, seq, hidden = 2, 10, 512
num_heads, head_dim = 8, 64     # hidden = num_heads * head_dim

x = torch.zeros(batch, seq, hidden)

# --- Project and split into heads ---
# (q_proj output has shape [batch, seq, hidden])
q = x.view(batch, seq, num_heads, head_dim)   # view: split hidden
q = q.transpose(1, 2)                          # transpose: [b, heads, seq, d]

k = x.view(batch, seq, num_heads, head_dim)
k = k.transpose(1, 2)

# --- Attention scores ---
scores = q @ k.transpose(-2, -1)              # transpose: K^T for dot product
# scores: [batch, heads, seq, seq]

# --- Merge heads back ---
out = torch.zeros(batch, num_heads, seq, head_dim)
out = out.transpose(1, 2)                      # [batch, seq, heads, head_dim]
out = out.reshape(batch, seq, hidden)          # reshape: merge heads*head_dim

print(out.shape)                               # torch.Size([2, 10, 512])
```
