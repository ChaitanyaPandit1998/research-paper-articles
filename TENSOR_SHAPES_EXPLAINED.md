# Tensor Shape Operations — When to Reshape vs Transpose

A practical guide for understanding shape transformations in transformer attention code.

---

## The Two Operations

| Operation | What it does | When to use it |
|---|---|---|
| `reshape` / `view` | Merges or splits dimensions | Combine `[heads, n_rep]` → `[heads*n_rep]`, or split `[hidden]` → `[heads, head_dim]` |
| `transpose` / `permute` | Reorders dimensions | When the axis order is wrong for the next operation |

**Key rule:** `reshape` never reorders data in memory; `permute` does.

---

## A Practical Process

1. Write out the current shape with labels.
2. Write out what the next operation *needs*.
3. The gap tells you what to do.

---

## Transformer Examples

### Example 1 — Splitting `hidden` into heads (after linear projection)

```
After q_proj:    [batch, seq, hidden]           ← what we have
                         hidden = heads * head_dim
Attention needs: [batch, heads, seq, head_dim]  ← what we need
```

- `hidden → heads, head_dim` is a **split** → `reshape`
- `seq` and `heads` are in the wrong order → `transpose`

```python
q = q.reshape(batch, seq, num_heads, head_dim)  # split hidden
q = q.transpose(1, 2)                           # [batch, heads, seq, head_dim]
```

---

### Example 2 — Transposing K for attention scores

```
Q shape: [batch, heads, seq_q, head_dim]
K shape: [batch, heads, seq_k, head_dim]

Q @ K requires: [..., seq_q, head_dim] @ [..., head_dim, seq_k]
                                                  ↑ K must be transposed
```

Nothing is being merged or split — just reordered → **transpose only**.

```python
scores = Q @ K.transpose(-2, -1)  # [batch, heads, seq_q, seq_k]
```

---

### Example 3 — Merging heads back after attention

```
After attention: [batch, heads, seq, head_dim]
o_proj needs:    [batch, seq, heads * head_dim]
```

- `heads` and `seq` are in wrong order → `transpose`
- Then `heads, head_dim` must merge → `reshape`

```python
out = out.transpose(1, 2)                        # [batch, seq, heads, head_dim]
out = out.reshape(batch, seq, heads * head_dim)  # [batch, seq, hidden]
```

---

### Example 4 — `repeat_kv` (Grouped Query Attention head expansion)

```
Have: [batch, kv_heads, seq, head_dim]
Need: [batch, kv_heads * n_rep, seq, head_dim]
```

`kv_heads` must expand into `kv_heads * n_rep` — that's a split then merge
(`kv_heads → kv_heads, n_rep → kv_heads*n_rep`). No axis reordering needed → **reshape only**
(with an intermediate `expand` to broadcast without copying memory).

See [`GEMMA_FILES_EXPLAINED.md`](GEMMA_FILES_EXPLAINED.md) → `repeat_kv` section for the full step-by-step.

---

## Quick Decision Tree

```
Do the axes need to be in a different order?
    Yes → permute / transpose
    No  → skip

Do axes need to be merged or split?
    Yes → reshape / view
    No  → skip

Need both? → permute FIRST, then reshape
```

### Why order matters: permute before reshape

Reshape interprets the tensor as a flat array of numbers and re-slices it.
If the axes are in the wrong order when you reshape, the numbers get assigned
to the wrong slots silently — no error, wrong result.

```python
# WRONG — reshaping before permuting scrambles data
x = x.reshape(batch, seq, heads * head_dim)  # seq and heads still swapped!
x = x.permute(...)                            # too late

# CORRECT
x = x.transpose(1, 2)                        # fix axis order first
x = x.reshape(batch, seq, heads * head_dim)  # now safe to merge
```

---

## Summary

| Goal | Operation | Example |
|---|---|---|
| Split one dim into two | `reshape` | `[b, seq, hidden]` → `[b, seq, heads, head_dim]` |
| Merge two dims into one | `reshape` | `[b, heads, n_rep, ...]` → `[b, heads*n_rep, ...]` |
| Swap axis order | `transpose` / `permute` | `[b, seq, heads, d]` → `[b, heads, seq, d]` |
| Both reorder + merge | `permute` then `reshape` | heads → output projection |
