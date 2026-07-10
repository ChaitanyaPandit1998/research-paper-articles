# Tensors in PyTorch: A Structured Guide for LLM Work

---

## 1. Tensor Basics

### What a tensor is

A scalar is a single number. A vector is a list. A matrix is a grid. A tensor is the generalisation — any number of dimensions.

```python
scalar = torch.tensor(3.14)          # shape: []        — 0 dimensions
vector = torch.tensor([1, 2, 3])     # shape: [3]       — 1 dimension
matrix = torch.tensor([[1,2],[3,4]]) # shape: [2, 2]    — 2 dimensions
cube   = torch.zeros(2, 3, 4)        # shape: [2, 3, 4] — 3 dimensions
```

In LLM work you constantly deal with 4D tensors: `[batch, heads, seq_len, head_dim]`. That's just a tensor with 4 dimensions — the math is identical to the 2D case.

---

### `torch.tensor()` vs `torch.Tensor()` vs factory functions

```python
# torch.tensor() — copies data, infers dtype from input
a = torch.tensor([1.0, 2.0])     # float32 (Python float → float32)
b = torch.tensor([1, 2])         # int64   (Python int → int64)

# torch.Tensor() — always float32, no dtype inference
c = torch.Tensor([1, 2])         # float32 regardless

# Factory functions — preferred in practice
torch.zeros(3, 4)         # all zeros, float32
torch.ones(3, 4)          # all ones
torch.randn(3, 4)         # random, standard normal
torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8] — like Python range()
torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0] — evenly spaced
```

Rule of thumb: use `torch.tensor()` when you have existing data, factory functions when you need a fresh tensor.

---

### `dtype`

| dtype | bits | used for |
|---|---|---|
| `float32` | 32 | default training, most ops |
| `bfloat16` | 16 | mixed-precision training (modern GPUs, TPUs) |
| `float16` | 16 | inference on older GPUs |
| `int64` | 64 | token IDs, indices |
| `bool` | 1 | attention masks, padding masks |

```python
token_ids = torch.tensor([101, 2003, 1037], dtype=torch.int64)
mask      = torch.tensor([True, True, False], dtype=torch.bool)
weights   = torch.randn(768, 768, dtype=torch.bfloat16)
```

`bfloat16` has the same exponent range as `float32` (so it doesn't overflow) but half the precision — that's why it's preferred for training over `float16`.

---

### `device`

A tensor lives on exactly one device. Operations between tensors on different devices fail.

```python
cpu_tensor  = torch.randn(3, 3)                        # on CPU
gpu_tensor  = torch.randn(3, 3, device="cuda")         # on GPU
gpu_tensor2 = torch.randn(3, 3, device="cuda:1")       # on second GPU

# This will error:
cpu_tensor + gpu_tensor   # RuntimeError: expected all tensors on same device
```

---

### `shape`, `ndim`, `numel()`

```python
x = torch.randn(2, 8, 512, 64)  # batch=2, heads=8, seq=512, head_dim=64

x.shape      # torch.Size([2, 8, 512, 64])
x.ndim       # 4
x.numel()    # 2 × 8 × 512 × 64 = 524288 — total number of elements
```

---

## 2. Indexing and Slicing

### Basic slicing

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

x[0]        # tensor([1, 2, 3])         — first row
x[1:3]      # tensor([[4,5,6],[7,8,9]]) — rows 1 and 2
x[:, 0]     # tensor([1, 4, 7])         — first column, all rows
x[1, 2]     # tensor(6)                 — row 1, col 2
```

The `:` means "all of this dimension". `[:, 0]` reads as "every row, column 0".

---

### Boolean masking

```python
x = torch.tensor([3, -1, 4, -1, 5])

mask = x > 0          # tensor([True, False, True, False, True])
x[mask]               # tensor([3, 4, 5]) — only positive values

# In-place zeroing of negatives (common in attention masking):
x[x < 0] = 0         # tensor([3, 0, 4, 0, 5])
```

---

### `torch.where()`

`torch.where(condition, x, y)` picks from `x` where condition is True, from `y` where False.

```python
scores = torch.tensor([0.8, -1e9, 0.3, -1e9, 0.5])
mask   = torch.tensor([True, False, True, False, True])

# Replace masked positions with -inf before softmax (causal masking)
result = torch.where(mask, scores, torch.tensor(float('-inf')))
# tensor([0.8, -inf, 0.3, -inf, 0.5])
```

This is exactly how causal attention masks work — positions the token cannot attend to become `-inf`, so softmax drives them to zero.

---

### Advanced indexing with integer tensors

```python
vocab = torch.randn(50000, 768)   # embedding table: 50k tokens, dim 768
token_ids = torch.tensor([101, 2003, 1037])

# Index with a tensor of indices — retrieves 3 rows
embeddings = vocab[token_ids]     # shape: [3, 768]
```

This is the embedding lookup every transformer does on input token IDs.

---

### View (shared memory) vs copy

```python
x = torch.arange(6)          # tensor([0, 1, 2, 3, 4, 5])
y = x.view(2, 3)             # reshape — shares x's memory

y[0, 0] = 99
print(x)                     # tensor([99, 1, 2, 3, 4, 5]) — x changed too!

z = x.clone().view(2, 3)     # clone first → independent copy
z[0, 0] = 0
print(x)                     # x is unchanged
```

Most shape operations (`view`, `transpose`, slices) return views. Surprising mutations are the classic bug when you forget this.

---

## 3. Shape Manipulation

This section is the most important for LLM work. Almost every transformer operation is a sequence of reshapes.

---

### `view()` vs `reshape()`

Both change shape without moving data — when possible. The difference is what happens when it's *not* possible.

```python
x = torch.arange(12)

x.view(3, 4)      # works — returns a view (zero copy)
x.view(4, 3)      # works
x.reshape(3, 4)   # also works — but if a view isn't possible, copies
```

After a `transpose()`, the tensor is no longer contiguous in memory (more on this in section 6). `view()` will error; `reshape()` will silently copy.

```python
x = torch.randn(3, 4)
t = x.transpose(0, 1)    # shape [4, 3], but memory layout is still [3, 4]

t.view(12)               # RuntimeError: not contiguous
t.reshape(12)            # works — makes a copy internally
t.contiguous().view(12)  # explicit: make contiguous, then view
```

**Rule:** use `view()` intentionally when you know the tensor is contiguous and you want a guaranteed zero-copy operation. Use `reshape()` when you don't need that guarantee.

---

### `transpose()` and `permute()`

`transpose()` swaps exactly two dimensions. `permute()` reorders all dimensions at once.

```python
x = torch.randn(2, 8, 512, 64)  # [batch, heads, seq, head_dim]

# Swap seq and head_dim
x.transpose(2, 3)      # shape: [2, 8, 64, 512]

# Reorder all dims
x.permute(0, 2, 1, 3)  # shape: [2, 512, 8, 64]
```

In Llama's attention code, after computing Q, K, V from the projection:

```python
# q shape after projection: [batch, seq_len, num_heads * head_dim]
q = q.view(batch, seq_len, num_heads, head_dim)
q = q.transpose(1, 2)   # → [batch, num_heads, seq_len, head_dim]
```

This is the head-splitting pattern in `modeling_llama.py`. The `view()` splits the last dimension into `(heads, head_dim)`, then `transpose()` moves heads forward so each head's data is contiguous.

---

### `squeeze()` and `unsqueeze()`

`unsqueeze(dim)` adds a dimension of size 1. `squeeze(dim)` removes it.

```python
x = torch.randn(512, 64)     # [seq_len, head_dim]

x.unsqueeze(0)    # [1, 512, 64]   — add batch dim
x.unsqueeze(1)    # [512, 1, 64]   — add mid dim

y = torch.randn(1, 512, 64)
y.squeeze(0)      # [512, 64]      — remove the size-1 batch dim
y.squeeze()       # removes ALL size-1 dims
```

`unsqueeze` is everywhere in broadcasting setups — you add a dim so PyTorch can broadcast across it.

---

### `expand()` vs `repeat()`

```python
x = torch.tensor([[1], [2], [3]])   # shape: [3, 1]

x.expand(3, 4)    # shape: [3, 4] — zero copy, just changes strides
x.repeat(1, 4)    # shape: [3, 4] — allocates new memory, copies data
```

`expand()` is preferred when you're about to use the result in a computation. `repeat()` is for when you genuinely need a concrete copy — rare.

In GQA (Grouped Query Attention), K and V have fewer heads than Q. `expand()` broadcasts K and V to match Q's head count without allocating memory for the repeated heads:

```python
# Q: [batch, 32, seq, head_dim]  (32 query heads)
# K: [batch,  4, seq, head_dim]  (4 KV heads, grouped)

k = k.unsqueeze(2)                       # [batch, 4, 1, seq, head_dim]
k = k.expand(-1, -1, 8, -1, -1)         # [batch, 4, 8, seq, head_dim]
k = k.reshape(batch, 32, seq, head_dim) # merge group dim back
```

---

### `contiguous()`

After `transpose()` or `permute()`, the tensor's memory layout doesn't match its shape. `contiguous()` makes a copy that does.

```python
x = torch.randn(3, 4)
t = x.transpose(0, 1)

t.is_contiguous()               # False
t.contiguous().is_contiguous()  # True
```

You need this before `view()` — see section 6 for why.

---

### `flatten()` and `unflatten()`

```python
x = torch.randn(2, 8, 512, 64)

x.flatten(1)              # flatten dims 1 onward → [2, 262144]
x.flatten(2, 3)           # flatten only dims 2–3 → [2, 8, 32768]

y = torch.randn(2, 32768)
y.unflatten(1, (512, 64)) # [2, 512, 64] — split dim 1 into (512, 64)
```

---

## 4. Broadcasting

### The rules

PyTorch aligns shapes from the right, then expands any dimension that is 1.

```
Shape A:    [   8, 512,  64]
Shape B:    [      1,   64]   ← aligned from right
Result:     [   8, 512,  64]  ← B's size-1 dims expand to match A
```

```python
scores  = torch.randn(8, 512, 512)   # [heads, seq, seq]
mask    = torch.zeros(1, 512, 512)   # [1, seq, seq]

scores + mask   # mask broadcasts across the 8 heads — no data copy
```

---

### When broadcasting copies data vs when it's zero-copy

`expand()` is zero-copy — it just adjusts strides so the same memory element is read multiple times. But once you pass a broadcasted tensor into most ops (like `+` or `matmul`), PyTorch will materialise the expanded tensor to perform the computation. The *view* is zero-copy; the *result of the op* is not.

---

### Common broadcasting bugs

```python
a = torch.randn(512, 64)
b = torch.randn(64, 512)

a + b   # RuntimeError — shapes don't align from the right
        # Right-align: [512, 64] vs [64, 512] → last dims 64 ≠ 512
```

The silent version is worse — shapes that *almost* match:

```python
a = torch.randn(8, 1, 512)
b = torch.randn(1, 512, 8)

a + b   # shape: [8, 512, 8] — no error, but probably not what you wanted
```

Always check `.shape` after operations when debugging unexpected results.

---

### How GQA uses broadcasting

In Grouped Query Attention, K and V heads are shared across groups of Q heads. Broadcasting expands them without copying:

```python
# Q: [batch, 32, seq, head_dim]
# K: [batch,  4, seq, head_dim]

k = k.unsqueeze(2).expand(-1, -1, 8, -1, -1)
# K is now virtually [batch, 4, 8, seq, head_dim] — same memory, different strides
```

---

## 5. Math Operations

### Elementwise

```python
x = torch.tensor([1.0, 4.0, 9.0])

x + 1          # [2., 5., 10.]
x ** 2         # [1., 16., 81.]
torch.sqrt(x)  # [1., 2., 3.]
torch.exp(x)   # [e¹, e⁴, e⁹]
```

---

### Reductions — `dim=` and `keepdim=`

```python
x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])   # shape: [2, 3]

x.sum()                       # tensor(21.) — all elements
x.sum(dim=0)                  # tensor([5., 7., 9.]) — sum along rows → shape [3]
x.sum(dim=1)                  # tensor([6., 15.])    — sum along cols → shape [2]
x.sum(dim=1, keepdim=True)    # tensor([[6.],[15.]]) — shape [2,1], keeps dims
```

`keepdim=True` matters for broadcasting — without it, the reduced tensor loses a dimension and may not align correctly in subsequent operations.

```python
# Softmax by hand (why keepdim matters)
x = torch.randn(4, 512)
x_max = x.max(dim=1, keepdim=True).values   # [4, 1] — broadcasts correctly
x = x - x_max                               # [4, 512] - [4, 1] → fine
```

---

### Matrix operations

```python
a = torch.randn(512, 64)
b = torch.randn(64, 128)

torch.matmul(a, b)   # [512, 128]
a @ b                # same — @ is the matmul operator

# Batched matmul — @ works on 3D/4D tensors too
q = torch.randn(2, 8, 512, 64)   # [batch, heads, seq, head_dim]
k = torch.randn(2, 8, 64, 512)   # [batch, heads, head_dim, seq]

scores = q @ k    # [2, 8, 512, 512] — matmul over last two dims, batch over first two
```

`torch.bmm()` is the older batched matmul — only handles 3D tensors. The `@` operator is more general and preferred.

---

### `einsum`

Einstein summation — expresses any contraction or reordering in one string.

```python
# Attention: scores[b,h,s,S] = sum over d of q[b,h,s,d] * k[b,h,S,d]
scores = torch.einsum("bhsd,bhSd->bhsS", q, k)

# Outer product
a = torch.randn(3)
b = torch.randn(4)
torch.einsum("i,j->ij", a, b)   # [3, 4]

# Batch matrix multiply
torch.einsum("bik,bkj->bij", x, y)   # equivalent to x @ y for 3D
```

Read the string as: name the dims of each input, name the dims of the output. Any dim that appears in inputs but not the output gets summed over.

---

### Softmax and numerical stability

```python
x = torch.tensor([1.0, 2.0, 3.0])
torch.softmax(x, dim=0)    # [0.09, 0.24, 0.67]
```

Why numerical stability matters:

```python
x = torch.tensor([1000.0, 1001.0, 1002.0])
torch.softmax(x, dim=0)            # works — PyTorch subtracts max internally

# Naive implementation:
torch.exp(x) / torch.exp(x).sum()  # inf/inf → nan
```

PyTorch's `softmax` subtracts `max(x)` before exponentiating — this doesn't change the result mathematically but prevents overflow. `log_softmax` is additionally preferred for loss computation because `log(softmax(x))` has worse numerical properties than computing `log_softmax` directly.

---

### In-place ops and autograd

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

x.add_(1)   # in-place — modifies x directly
# RuntimeError: a leaf Variable that requires grad has been used in an in-place operation
```

PyTorch's autograd records operations to compute gradients. In-place ops destroy the original value that the backward pass needs to use. Avoid them on any tensor that's part of a computation graph.

---

## 6. Memory and Storage

### Storage and strides

Every tensor is a view into a flat 1D block of memory called storage. The tensor's `strides` tell PyTorch how many elements to skip in storage to advance one step in each dimension.

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])   # shape: [2, 3]

x.storage()    # [1, 2, 3, 4, 5, 6] — flat block
x.stride()     # (3, 1) — move 3 to go down a row, move 1 to go right a col
```

To find element `x[i, j]`: `storage[i * 3 + j * 1]`.

---

### Why `transpose()` doesn't move data

```python
t = x.transpose(0, 1)   # shape: [3, 2]

t.storage()   # still [1, 2, 3, 4, 5, 6] — same memory, unchanged
t.stride()    # (1, 3) — strides are swapped
```

`t[i, j]` = `storage[i * 1 + j * 3]`. The data didn't move — PyTorch just changed the recipe for navigating it. This is why `transpose()` is cheap and why `view()` fails afterward: `view()` requires elements to be laid out consecutively in storage, and after transposing they aren't.

---

### `is_contiguous()`

```python
x = torch.randn(3, 4)
x.is_contiguous()               # True — row-major, as expected

t = x.transpose(0, 1)
t.is_contiguous()               # False — strides no longer match shape

t.contiguous().is_contiguous()  # True — made a fresh copy in memory
```

---

### `clone()` vs `detach()` vs `detach().clone()`

```python
x = torch.randn(3, requires_grad=True)
y = x * 2   # y is in the computation graph

y.clone()            # new tensor, same graph — gradient still flows through y
y.detach()           # same storage as y, removed from graph — no copy
y.detach().clone()   # new tensor, removed from graph — fully independent
```

Use `.detach().clone()` when you want a plain tensor you can inspect or log, with no graph attachment and no shared memory.

---

## 7. Autograd — How Gradients Flow Through Tensors

### `requires_grad=True`

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1

y.backward()
x.grad   # tensor([7.]) — dy/dx = 2x + 3 = 2(2) + 3 = 7
```

Only leaf tensors (those you created, not computed) accumulate `.grad`. Intermediate tensors don't by default.

---

### The computational graph

PyTorch builds a graph dynamically as you do operations. Each tensor stores a reference to the function that created it.

```python
x = torch.randn(3, requires_grad=True)
y = x.sum()

y.grad_fn                  # <SumBackward0>
y.grad_fn.next_functions   # points back to x
```

`backward()` walks this graph in reverse, applying the chain rule at each node.

---

### `torch.no_grad()`

```python
with torch.no_grad():
    output = model(input)   # no graph built — saves memory and compute
```

During inference you don't need gradients. `no_grad()` prevents PyTorch from building the graph, which reduces memory usage and speeds up the forward pass.

---

### `detach()`

```python
# Stop gradient flowing through a particular path
target = output.detach()   # treat as a constant, not a trainable output
loss = F.mse_loss(prediction, target)
```

Common in RL and contrastive learning where you want one branch of a computation to not receive gradients.

---

### `retain_graph=True`

```python
loss.backward(retain_graph=True)   # keep the graph alive
loss.backward()                    # use it again
```

By default, PyTorch frees the graph after `backward()` to save memory. `retain_graph=True` keeps it — needed when you call backward multiple times (e.g. multiple loss terms or meta-learning).

---

## 8. Type and Device Movement

```python
x = torch.randn(3, 3)

x.to("cuda")                          # move to GPU
x.to("cpu")                           # move to CPU
x.to(torch.bfloat16)                  # cast dtype
x.to("cuda", dtype=torch.bfloat16)   # both at once — most efficient, one copy

# Shorthands
x.cuda()      # to GPU
x.cpu()       # to CPU
x.float()     # float32
x.half()      # float16
x.bfloat16()  # bfloat16
```

**Common bug:** two tensors on different devices or with different dtypes error at the operation, not at creation. The fix is to cast/move before the op:

```python
a = torch.randn(3, device="cpu")
b = torch.randn(3, device="cuda")

a + b               # RuntimeError: expected all on same device

# Fix:
a.to(b.device) + b
```

---

## 9. Tensor Operations Critical for Transformer Internals

Everything in sections 1–8 feeds into these patterns.

---

### Batched matmul for attention scores

```python
q = torch.randn(2, 8, 512, 64)   # [batch, heads, seq_q, head_dim]
k = torch.randn(2, 8, 512, 64)   # [batch, heads, seq_k, head_dim]

# Compute all attention scores in one op
scores = q @ k.transpose(-2, -1)  # [2, 8, 512, 512]
# transpose(-2, -1) swaps the last two dims: [batch, heads, head_dim, seq_k]
```

`@` on 4D tensors batches over all leading dims and does matmul over the last two.

---

### `view()` + `transpose()` head-splitting pattern

This is the pattern in every transformer implementation:

```python
batch, seq_len = 2, 512
num_heads, head_dim = 8, 64
hidden_dim = num_heads * head_dim   # 512

# After linear projection: flat hidden dim
q = torch.randn(batch, seq_len, hidden_dim)  # [2, 512, 512]

# Split into heads
q = q.view(batch, seq_len, num_heads, head_dim)  # [2, 512, 8, 64]
q = q.transpose(1, 2)                            # [2, 8, 512, 64]
```

`view()` reinterprets the last 512 elements as 8 groups of 64 (no data movement). `transpose()` reorders dims so each head's queries are grouped together.

---

### `torch.cat()` vs `torch.stack()`

```python
a = torch.randn(4, 64)
b = torch.randn(4, 64)

torch.cat([a, b], dim=0)    # [8, 64]   — join along existing dim 0
torch.cat([a, b], dim=1)    # [4, 128]  — join along existing dim 1

torch.stack([a, b], dim=0)  # [2, 4, 64] — new dim 0
torch.stack([a, b], dim=1)  # [4, 2, 64] — new dim 1
```

`cat` joins along an existing dimension; `stack` creates a new one. Use `stack` when assembling a batch from individual items, `cat` when appending to an existing dimension.

---

### `torch.split()` and `torch.chunk()`

```python
x = torch.randn(4, 12)

# split into pieces of specified size
torch.split(x, 4, dim=1)    # three tensors of [4, 4]

# split into N equal chunks
torch.chunk(x, 3, dim=1)    # three tensors of [4, 4]
```

In QKV projections, a single linear layer outputs all three:

```python
qkv = linear(x)                              # [batch, seq, 3 * hidden]
q, k, v = torch.split(qkv, hidden, dim=-1)  # each [batch, seq, hidden]
```

---

### `einsum` for attention

```python
q = torch.randn(2, 8, 512, 64)
k = torch.randn(2, 8, 512, 64)
v = torch.randn(2, 8, 512, 64)

# Attention scores — d is summed over
scores = torch.einsum("bhsd,bhSd->bhsS", q, k)   # [2, 8, 512, 512]
# b=batch, h=head, s=query_seq, S=key_seq, d=head_dim

attn = torch.softmax(scores / 64**0.5, dim=-1)

# Weighted sum of values
out = torch.einsum("bhsS,bhSd->bhsd", attn, v)   # [2, 8, 512, 64]
```

Read the string as: name the dims of each input, name the dims of the output. Any dim that appears in inputs but not the output gets summed over.

---

### `masked_fill()` for causal masks

```python
scores = torch.randn(4, 512, 512)   # [batch, seq, seq]

# Upper triangle mask — position i cannot attend to j > i
mask = torch.triu(torch.ones(512, 512), diagonal=1).bool()

scores = scores.masked_fill(mask, float('-inf'))
attn   = torch.softmax(scores, dim=-1)
```

After softmax, all `-inf` positions become 0 — the token effectively ignores those positions. This is how causal (autoregressive) attention is implemented.

---

## 10. Performance and Memory Efficiency

### Tracking VRAM

```python
print(torch.cuda.memory_allocated() / 1e9, "GB")     # currently in use
print(torch.cuda.max_memory_allocated() / 1e9, "GB") # peak since last reset
torch.cuda.reset_peak_memory_stats()
```

---

### Gradient checkpointing

Normally PyTorch saves all intermediate activations for the backward pass. At large sequence lengths or batch sizes, this exhausts GPU memory.

Gradient checkpointing discards activations during the forward pass and recomputes them during backward — trading ~30% extra compute for a large memory saving.

```python
from torch.utils.checkpoint import checkpoint

output = checkpoint(transformer_block, hidden_states)
```

---

### `pin_memory=True`

```python
loader = DataLoader(dataset, batch_size=32, pin_memory=True)
```

Pinned memory is page-locked on CPU — the GPU can fetch it directly via DMA without an extra CPU copy. Makes CPU→GPU transfer noticeably faster when the GPU is the bottleneck.

---

### Tensor parallelism reshaping patterns

In column-parallel linear (split weight across GPUs by output columns):

```python
# Each GPU holds a slice of the output dim
weight_slice = full_weight[:, start:end]   # [in, out/N]
output_slice = input @ weight_slice        # [batch, seq, out/N]
# All-gather across GPUs to reconstruct full output
```

In row-parallel linear (split by input dim):

```python
input_slice  = full_input[:, :, start:end]  # [batch, seq, in/N]
output_slice = input_slice @ weight_slice   # partial sum
# All-reduce across GPUs to sum partial results
```

The reshaping is just a `view()` or slice on the weight tensor — the parallelism comes from distributing those slices across devices.

---

### `torch.compile()`

```python
model = torch.compile(model)   # JIT-traces the model, fuses ops
```

`torch.compile()` (introduced in PyTorch 2.0) traces your model and applies kernel fusion — for example, merging the matmul + scale + softmax in attention into a single kernel. The biggest gains are in models with many small ops that are individually memory-bandwidth-bound.

---

## The Learning Path

The sections connect like this:

- **Basics (1) + Indexing (2)** give you the vocabulary.
- **Shape manipulation (3)** is where the Llama patterns live — `view()` + `transpose()` makes sense once you see that they're just adjusting strides over a flat storage block.
- **Memory and storage (6)** explains *why* contiguity matters after transpose and *why* view is zero-copy.
- **Broadcasting (4)** explains how masks and GQA head expansion work without extra memory.
- **Autograd (7)** explains why in-place ops and detach patterns exist.
- **Transformer internals (9)** is all of the above applied — every line in `modeling_llama.py` maps back to a concept from sections 1–8.
