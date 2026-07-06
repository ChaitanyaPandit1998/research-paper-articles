# Llama3 vs Llama4 Text Attention — Side by Side

Covers `LlamaAttention` (Llama3, `modeling_llama.py`) vs `Llama4TextAttention` (Llama4, `modeling_llama4.py`).
Both are decoder-only causal attention. The shared foundation is identical; the differences are additive features layered on top in Llama4.

---

## Dimensions at a Glance

Using **Llama3 8B** and **Llama4 Scout 17B-16E** as concrete references.

| | Llama3 8B | Llama4 Scout 17B-16E |
|---|---|---|
| Hidden size | 4096 | 5120 |
| Q heads | 32 | 40 |
| KV heads | 8 | 8 |
| head_dim | 128 | 128 |
| KV groups | 4 | 5 |
| Vocab size | 128256 | 202048 |
| Num layers | 32 | 48 |
| Max context | 8192 | 131072 (128K) |

`head_dim` is 128 in both, so the attention scaling (`head_dim**-0.5 ≈ 0.0884`) is identical.

---

## Step 1 — QKV Projections

**Llama3:**
```python
q_proj: Linear(4096 → 32*128=4096, bias=attention_bias)   # default False
k_proj: Linear(4096 →  8*128=1024, bias=attention_bias)
v_proj: Linear(4096 →  8*128=1024, bias=attention_bias)
```

**Llama4:**
```python
q_proj: Linear(5120 → 40*128=5120, bias=attention_bias)   # default False
k_proj: Linear(5120 →  8*128=1024, bias=attention_bias)
v_proj: Linear(5120 →  8*128=1024, bias=attention_bias)
```

Structure identical — both use GQA with 8 KV heads, both default `bias=False`. The only difference is the larger hidden size and more Q heads in Llama4.

---

## Step 2 — Positional Encoding: RoPE Implementation

This is the deepest implementation difference. Both compute the same rotation math — they differ in *how* they express it.

---

### Part A — What is identical

**`__init__`** is the same except the config type:

```python
# Llama3                                  # Llama4
class LlamaRotaryEmbedding(nn.Module):    class Llama4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, ...)   def __init__(self, config: Llama4TextConfig, ...)
```

Every line inside `__init__` is identical: `max_seq_len_cached`, `original_max_seq_len`, rope_type dispatch via `ROPE_INIT_FUNCTIONS`, both `register_buffer` calls. The only difference is the config class because Llama4 uses a nested config (`Llama4Config` → `Llama4TextConfig` + `Llama4VisionConfig`) to separate text and vision settings.

**`compute_default_rope_parameters`** is byte-for-byte identical in both:

```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(..., dtype=torch.float) / dim))
```

The base RoPE frequency schedule did not change between Llama3 and Llama4. What changed is what the frequencies are used for downstream.

---

### Part B — `forward`: four concrete differences

```
Llama3                                          Llama4
─────────────────────────────────────────────── ──────────────────────────────────────────────────────────
inv_freq_expanded = (                           inv_freq_expanded = (
    self.inv_freq[None, :, None]                    self.inv_freq[None, :, None]
    .float()                                        .float()
    .expand(position_ids.shape[0], -1, 1)           .expand(position_ids.shape[0], -1, 1)
    .to(x.device)                            ①  )                                              ①
)
position_ids_expanded = position_ids[:, None, :].float()    (same)

freqs = (                                       freqs = (
    inv_freq_expanded.float()            ②          inv_freq_expanded.to(x.device)             ②
    @ position_ids_expanded.float()      ②          @ position_ids_expanded                    ②
).transpose(1, 2)                               ).transpose(1, 2)

emb = torch.cat((freqs, freqs), dim=-1)  ③      (no emb step)                                  ③
cos = emb.cos() * self.attention_scaling  ④      freqs_cis = torch.polar(                       ④
sin = emb.sin() * self.attention_scaling  ④          torch.ones_like(freqs), freqs)             ④
                                                freqs_cis = freqs_cis * self.attention_scaling  ④

return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)   return freqs_cis                         ④
```

#### Difference ① — where `.to(x.device)` sits

Llama3 chains `.to(x.device)` onto `expand` — the expanded tensor lands on the right device before the matmul. Llama4 moves the `.to(x.device)` call inside the matmul expression itself: `inv_freq_expanded.to(x.device) @`. The result is identical; Llama4's placement is a minor code readability refactor — keeping the device move visually close to where the tensor is actually consumed.

#### Difference ② — `.float()` on both matmul operands vs only on one

Llama3: `inv_freq_expanded.float() @ position_ids_expanded.float()` — both sides are explicitly cast to float32.

Llama4: `inv_freq_expanded.to(x.device) @ position_ids_expanded` — only `inv_freq_expanded` is float (from `.float()` two lines above). `position_ids_expanded` is not re-cast here. The `maybe_autocast(..., enabled=False)` context block surrounding both still forces float32 arithmetic, so the precision is equivalent. Llama4 just avoids the redundant second `.float()` call.

#### Difference ③ — `cat(freqs, freqs)` in Llama3, absent in Llama4

This is the central structural difference in `forward`.

```python
# Llama3
emb = torch.cat((freqs, freqs), dim=-1)   # [B, S, head_dim/2] → [B, S, head_dim]
```

Why it exists: Llama3's `rotate_half` splits the vector into two halves — all `x` values first, all `y` values second. For the elementwise multiply `q * cos` to work correctly, `cos[i]` must equal `cos[i + head_dim/2]` (same angle at both the `x` slot and its partner `y` slot). The `cat(freqs, freqs)` duplication guarantees exactly that.

Why Llama4 doesn't need it: `torch.polar(ones, freqs)` creates a complex number `e^(iθ)` for each angle in `freqs`. One complex number already encodes both the `cos(θ)` (real part) and `sin(θ)` (imaginary part) needed for the rotation. When you multiply a complex number `(a + bi) * e^(iθ)`, you get both rotation components implicitly — no duplication required.

In short: `cat(freqs, freqs)` exists to make real-valued arithmetic simulate what complex arithmetic does natively.

#### Difference ④ — what `forward` returns

```python
# Llama3 returns two real tensors:
return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
# cos: [B, S, head_dim]   sin: [B, S, head_dim]

# Llama4 returns one complex tensor:
return freqs_cis
# freqs_cis: [B, S, head_dim/2]  complex
```

Llama3 returns separate `cos` and `sin` tables at full `head_dim` — caller uses both explicitly. Llama4 returns a single complex tensor at half the size — `e^(iθ) = cos(θ) + i·sin(θ)` encodes both in one value. The complex tensor is half the size not because information was lost, but because each complex float64 equivalent (two float32s) holds one (cos, sin) pair.

The type signature of the callers changes accordingly:

```python
# Llama3 caller (LlamaAttention):
position_embeddings = self.rotary_emb(hidden_states, position_ids)   # returns (cos, sin)
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# Llama4 caller (Llama4TextAttention):
position_embeddings = self.rotary_emb(hidden_states, position_ids)   # returns freqs_cis
query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)
```

---

### Part C — applying the rotation: `rotate_half` + `apply_rotary_pos_emb` vs `apply_rotary_emb`

#### Llama3 — two real multiply-adds

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]   # first half:  [x0, x1, x2, x3]
    x2 = x[..., x.shape[-1] // 2 :]   # second half: [y0, y1, y2, y3]
    return torch.cat((-x2, x1), dim=-1)
    # result: [-y0, -y1, -y2, -y3,  x0, x1, x2, x3]

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)   # [B, S, head_dim] → [B, 1, S, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)   # broadcast over heads: q is [B, heads, S, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

`rotate_half` mechanically constructs the vector layout that makes one elementwise operation cover both components of all rotation pairs simultaneously. The `unsqueeze(1)` inserts a size-1 head axis so `[B, S, head_dim]` broadcasts against `[B, heads, S, head_dim]`.

The rotation formula, written out for pair 0:

```
dim 0: q[0] * cos[0] + rotate_half(q)[0] * sin[0]
     = x0 * cos(θ0) + (-y0) * sin(θ0)
     = x0·cosθ − y0·sinθ   ← x component of rotation ✓

dim 4: q[4] * cos[4] + rotate_half(q)[4] * sin[4]
     = y0 * cos(θ0) + x0 * sin(θ0)
     = x0·sinθ + y0·cosθ   ← y component of rotation ✓
```

This works because `cat(freqs, freqs)` made `cos[0] == cos[4]` (same angle at both slots) and `rotate_half` moved `-y0` to slot 0 and `x0` to slot 4.

#### Llama4 — one complex multiply

```python
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Step by step for `xq` (same for `xk`):

**`xq.float().reshape(*xq.shape[:-1], -1, 2)`**

Takes every consecutive pair of floats in the last dimension and groups them:
```
xq:     [B, S, heads, head_dim]         e.g. [..., 128]
reshape: [B, S, heads, head_dim/2, 2]   e.g. [..., 64, 2]
```
Pairs are `(dim0, dim1)`, `(dim2, dim3)`, etc. — consecutive, not split-half. This is the pairing convention choice.

**`torch.view_as_complex(...)`**

Interprets each `(a, b)` pair as a complex number `a + bi`:
```
[B, S, heads, head_dim/2, 2]  →  [B, S, heads, head_dim/2] complex
```

**`freqs_cis[:, :, None, :]`**

Inserts a size-1 axis at position 2 to broadcast over the heads dimension:
```
freqs_cis:             [B, S, head_dim/2] complex
freqs_cis[:,:,None,:]: [B, S, 1, head_dim/2] complex  → broadcasts against [B, S, heads, head_dim/2]
```

**`xq_ * freqs_cis[:, :, None, :]`**

Complex multiplication:
```
(a + bi) * (cosθ + i·sinθ) = (a·cosθ − b·sinθ) + i·(a·sinθ + b·cosθ)
```
This is the identical 2D rotation formula, in one operation:

```
real part → a·cosθ − b·sinθ  ← same as x_new = x·cosθ − y·sinθ
imag part → a·sinθ + b·cosθ  ← same as y_new = x·sinθ + y·cosθ
```

**`torch.view_as_real(...).flatten(3)`**

Unpacks complex back to pairs of reals, then flattens the last two dims back to `head_dim`:
```
[B, S, heads, head_dim/2] complex
  → view_as_real → [B, S, heads, head_dim/2, 2]
  → flatten(3)   → [B, S, heads, head_dim]
```

---

### Part D — memory layout: the one breaking difference

```
Llama3 split-half layout:                Llama4 consecutive-pair layout:
[x0, x1, x2, x3,  y0, y1, y2, y3]      [x0, y0,  x1, y1,  x2, y2,  x3, y3]
 ──── first half ────  ── second half ──  pair0    pair1    pair2    pair3
```

Both layouts encode the same 4 rotation pairs — they just arrange them differently in memory. The math is equivalent because the attention dot product `Q·Kᵀ` is a sum over all dimensions, which is invariant to the ordering of dimension pairs.

**Why this matters:** if you load Llama3 Q/K weight matrices directly into a Llama4 model without reordering, the pairing is wrong — `dim0` would incorrectly pair with `dim1` (a different frequency index) instead of `dim head_dim/2` (its Llama3 partner). The rotations would be corrupted. Model-conversion code must permute the Q/K weight dimensions accordingly.

---

### Part E — full data flow, side by side

```
                    Llama3                              Llama4
────────────────────────────────────────────────────────────────────────────
inv_freq            [head_dim/2]                        [head_dim/2]
                    same formula, same values            same formula, same values

inv_freq_expanded   [B, head_dim/2, 1]                  [B, head_dim/2, 1]

position_ids_exp    [B, 1, S]                           [B, 1, S]

freqs (matmul)      [B, head_dim/2, S] → transpose      [B, head_dim/2, S] → transpose
                    → [B, S, head_dim/2]                → [B, S, head_dim/2]

emb (cat)           [B, S, head_dim]   ← doubled        (skipped)
                    via cat(freqs, freqs)

frequency output    cos [B, S, head_dim]  real           freqs_cis [B, S, head_dim/2]  complex
                    sin [B, S, head_dim]  real            (cos + i·sin encoded together)

in apply_rotary_*:
  Q reshape         [B, heads, S, head_dim]  (unchanged) [B, S, heads, head_dim]
                                                          → reshape → [B, S, heads, head_dim/2, 2]
                                                          → view_as_complex → [B, S, heads, head_dim/2]

  broadcast trick   cos.unsqueeze(1)                     freqs_cis[:, :, None, :]
                    [B, 1, S, head_dim]                  [B, S, 1, head_dim/2]
                    → broadcasts over heads              → broadcasts over heads

  rotation          2 ops: q*cos + rotate_half(q)*sin   1 op: complex multiply
  formula           real-valued                          (a+bi)*(cosθ+i·sinθ)

  output reshape    already [B, heads, S, head_dim]      view_as_real + flatten(3)
                    (no reshape needed)                  → [B, S, heads, head_dim]
────────────────────────────────────────────────────────────────────────────
```

---

### Part F — why Llama4 made these changes

**Why switch to complex numbers?**

The `rotate_half` approach is a workaround for real-valued arithmetic. To rotate pair `(x, y)` by angle θ you need:
```
x_new = x·cosθ − y·sinθ
y_new = y·cosθ + x·sinθ
```
In real space, this requires touching `x` and `y` together. `rotate_half` achieves this by rearranging the whole vector and doing an elementwise multiply-add — effective, but indirect. The `cat(freqs, freqs)` duplication is also indirect — it exists only so the same angle appears at both slots of every pair.

Complex multiplication `(x + iy) * e^(iθ)` performs this rotation directly — the math is baked into the definition of complex multiplication. No rearrangement, no duplication. The code is shorter and the intent is clearer.

**Why the consecutive-pair layout?**

`torch.view_as_complex` requires that the two floats forming a complex number be adjacent in memory — i.e. consecutive pairs. It has no "split-half" mode. So switching to `view_as_complex` forces consecutive-pair layout. If Llama4 had kept split-half layout, it would need an extra permute step before `view_as_complex` to move `y` values next to their `x` partners — defeating the simplification.

**Why does the return type change from `(cos, sin)` to `freqs_cis`?**

Returning two separate real tensors is the natural interface when the caller applies rotation manually via `q * cos + rotate_half(q) * sin`. Returning a single complex tensor is the natural interface when the caller does `q_ * freqs_cis`. The interface follows the implementation: switching from two real multiplications to one complex multiply changes what the caller needs to receive.

**Why no float32 upcast on `position_ids_expanded` in Llama4?**

Llama3 casts both operands to float32 explicitly: `inv_freq_expanded.float() @ position_ids_expanded.float()`. Llama4 omits the second cast. Both are inside `maybe_autocast(..., enabled=False)`, which suppresses autocast and ensures float32 arithmetic regardless. The extra `.float()` on `position_ids_expanded` in Llama3 is a defensive belt-and-suspenders cast that Llama4 dropped as unnecessary given the context manager already guarantees float32.

---

### Part G — Numeric Trace: How Llama3 and Llama4 Tie Out

Both approaches rotate the same pairs by the same angles and produce the same `x_new`, `y_new` numbers. The only difference is the order those numbers sit in memory at the end.

**Setup:** `head_dim = 4` (2 pairs), position `p = 1`, frequencies `θ₀ = 1.0`, `θ₁ = 0.1`.

Query vector — same content, different layout conventions:
```
Llama3 (split-half):      q = [x₀,  x₁,  y₀,  y₁]  = [0.5,  0.3,  0.8,  0.6]
                               ←first half→  ←second half→
                               (all x's)     (all y's)

Llama4 (consecutive):     q = [x₀,  y₀,  x₁,  y₁]  = [0.5,  0.8,  0.3,  0.6]
                               pair 0        pair 1
```

---

#### Llama3 — step by step

**1. Build the angle table and double it:**
```
freqs     = [θ₀, θ₁]         = [1.0, 0.1]           shape [2]

emb = cat(freqs, freqs)       = [1.0, 0.1, 1.0, 0.1]  shape [4]   ← angle repeated for each half
```
Why the duplication: dim 0 `(x₀)` and dim 2 `(y₀)` are partners — they need the same angle `θ₀`. The `cat` puts `θ₀` at both positions 0 and 2.

**2. Compute cos and sin at full head_dim:**
```
cos = [cos(1.0), cos(0.1), cos(1.0), cos(0.1)]
    = [0.5403,   0.9950,   0.5403,   0.9950  ]

sin = [sin(1.0), sin(0.1), sin(1.0), sin(0.1)]
    = [0.8415,   0.0998,   0.8415,   0.0998  ]
```

**3. rotate_half(q):**
```
q             = [ x₀,  x₁,   y₀,   y₁]  = [ 0.5,  0.3,  0.8,  0.6]
rotate_half   = [-y₀, -y₁,   x₀,   x₁]  = [-0.8, -0.6,  0.5,  0.3]
```
This moves `-y` into the `x` slots and `x` into the `y` slots — setting up the subtraction and addition that the rotation formula needs.

**4. q × cos + rotate_half(q) × sin, slot by slot:**
```
dim 0 (x₀):  0.5 × 0.5403  +  (-0.8) × 0.8415  =  0.2702 − 0.6732  = −0.4030
dim 1 (x₁):  0.3 × 0.9950  +  (-0.6) × 0.0998  =  0.2985 − 0.0599  =  0.2386
dim 2 (y₀):  0.8 × 0.5403  +   0.5  × 0.8415   =  0.4322 + 0.4208  =  0.8530
dim 3 (y₁):  0.6 × 0.9950  +   0.3  × 0.0998   =  0.5970 + 0.0299  =  0.6269
```

**Llama3 output (split-half layout):**
```
[−0.4030,  0.2386,  0.8530,  0.6269]
  x₀_new   x₁_new   y₀_new   y₁_new
```

---

#### Llama4 — step by step (same input, different path)

**1. Build freqs_cis via torch.polar:**
```
freqs     = [1.0, 0.1]           shape [2]   ← same angle table as Llama3, NOT doubled

freqs_cis = e^(i·1.0),  e^(i·0.1)
          = cos(1.0)+i·sin(1.0),  cos(0.1)+i·sin(0.1)
          = 0.5403+0.8415i,       0.9950+0.0998i       shape [2] complex
```
One complex number per pair. It carries both cos and sin inside itself — no duplication needed.

**2. Reshape q into consecutive pairs:**
```
q (consecutive) = [0.5, 0.8, 0.3, 0.6]
reshape(-1, 2)  = [(0.5, 0.8), (0.3, 0.6)]     ← pair 0 and pair 1 now explicit
```

**3. view_as_complex — treat each (a, b) pair as a + bi:**
```
xq_ = [0.5+0.8i,   0.3+0.6i]   shape [2] complex
```

**4. Complex multiply xq_ × freqs_cis, pair by pair:**

Pair 0: `(0.5 + 0.8i) × (0.5403 + 0.8415i)`
```
= 0.5×0.5403  +  0.5×0.8415i  +  0.8i×0.5403  +  0.8i×0.8415i
= 0.2702       +  0.4208i       +  0.4322i       +  0.6732·i²
= 0.2702       +  0.4208i       +  0.4322i       −  0.6732      (since i²=−1)

real part:  0.2702 − 0.6732 = −0.4030   ← x₀_new ✓
imag part:  0.4208 + 0.4322 =  0.8530   ← y₀_new ✓
```

Pair 1: `(0.3 + 0.6i) × (0.9950 + 0.0998i)`
```
= 0.3×0.9950  +  0.3×0.0998i  +  0.6i×0.9950  +  0.6i×0.0998i
= 0.2985       +  0.0299i       +  0.5970i       −  0.0599

real part:  0.2985 − 0.0599 =  0.2386   ← x₁_new ✓
imag part:  0.0299 + 0.5970 =  0.6269   ← y₁_new ✓
```

**5. view_as_real → flatten:**
```
[(−0.4030, 0.8530),  (0.2386, 0.6269)]   → flatten →

[−0.4030,  0.8530,  0.2386,  0.6269]
  x₀_new   y₀_new   x₁_new   y₁_new
```

**Llama4 output (consecutive layout):**
```
[−0.4030,  0.8530,  0.2386,  0.6269]
  x₀_new   y₀_new   x₁_new   y₁_new
```

---

#### The tie-out

```
Llama3 output: [−0.4030,  0.2386,  0.8530,  0.6269]   split-half:   all x_new | all y_new
Llama4 output: [−0.4030,  0.8530,  0.2386,  0.6269]   consecutive:  pair0     | pair1

Same 4 numbers. Different order.
```

The rotation values `x₀_new = −0.4030`, `y₀_new = 0.8530`, `x₁_new = 0.2386`, `y₁_new = 0.6269` are identical in both. The memory layout is the only difference.

---

#### Why the mechanisms are equivalent — the structural map

Every step in Llama3 has a direct counterpart in Llama4:

```
Llama3 step                               Llama4 counterpart
─────────────────────────────────────────────────────────────────────────────
cat(freqs, freqs)                         not needed — torch.polar encodes both
  duplicates θ so the same angle            cos(θ) and sin(θ) in one complex
  appears at both the x-slot and y-slot     number; no need to duplicate

cos = emb.cos()                           freqs_cis = torch.polar(ones, freqs)
sin = emb.sin()                             = e^(iθ) = cos(θ) + i·sin(θ)
                                            both in one value, not two tensors

q layout [x₀,x₁,y₀,y₁]                  q layout [x₀,y₀,x₁,y₁]
  x and y split to halves so                x and y consecutive so
  rotate_half can swap them                 reshape(-1,2) can pair them

rotate_half:                              reshape(-1,2) + view_as_complex:
  [-y₀,-y₁, x₀,x₁]                        [(x₀+iy₀), (x₁+iy₁)]
  rearranges to create the                  groups x and y as a
  subtraction and addition                  single complex number
  terms needed for rotation

q*cos + rotate_half(q)*sin                xq_ * freqs_cis
  two elementwise ops                       one complex multiply
  x_new = x·cosθ − y·sinθ                  real part  = x·cosθ − y·sinθ
  y_new = x·sinθ + y·cosθ                  imag part  = x·sinθ + y·cosθ

output stays flat [x₀_new,x₁_new,...]    view_as_real + flatten
  no reshape needed                         converts complex result back
                                            to real flat tensor
─────────────────────────────────────────────────────────────────────────────
```

The short version: `cat(freqs, freqs)` + `rotate_half` is Llama3 manually doing in real arithmetic what `torch.polar` + complex multiplication does in one step. The problem they both solve is the same: **how do you multiply `x` by `cosθ` and subtract `y·sinθ` in the same vectorized operation**, when `x` and `y` are stored as separate numbers in a flat array?

- Llama3's answer: duplicate the angles, rearrange the vector so `-y` lands where `x` was, then multiply-add.
- Llama4's answer: group `x` and `y` into a complex number, then use the definition of complex multiplication (which is the rotation formula).

---

### Part H — Which method is more efficient, and why

The short answer: **FLOPs are identical. Llama4 wins on memory allocations. Llama3 wins in practice via fused CUDA kernels.**

---

#### FLOPs — identical

For one rotation pair `(x, y)` by angle `θ`, both methods do exactly:
```
4 multiplications  (x·cosθ, y·sinθ, x·sinθ, y·cosθ)
2 additions        (x·cosθ − y·sinθ,  x·sinθ + y·cosθ)
= 6 FLOPs per pair
```

Llama3: `(q * cos) + (rotate_half(q) * sin)` — 2 elementwise multiplies + 1 elementwise add across head_dim values.
Llama4: `(a + bi) * (cosθ + i·sinθ)` — complex multiply, which PyTorch expands to the same 4 multiplies + 2 adds.

Neither approach reduces the arithmetic. The differences are entirely in the surrounding overhead.

---

#### Memory allocations — Llama4 is cleaner

Count the tensors that must be **newly allocated** (not views) in the hot path:

```
Llama3                                      Llama4
─────────────────────────────────────────── ────────────────────────────────────────────
cat(freqs, freqs)      → allocates [B,S,D]  torch.polar(ones, freqs) → allocates [B,S,D/2] complex
cos = emb.cos()        → allocates [B,S,D]  (no separate cos tensor)
sin = emb.sin()        → allocates [B,S,D]  (no separate sin tensor)
rotate_half:                                reshape(-1,2)              → view, free
  cat((-x2, x1))       → allocates [B,H,S,D]  view_as_complex         → view, free
q * cos                → allocates [B,H,S,D]  xq_ * freqs_cis         → allocates [B,S,H,D/2]
rotate_half(q) * sin   → allocates [B,H,S,D]  view_as_real            → view, free
final add              → allocates [B,H,S,D]  flatten                 → view, free
─────────────────────────────────────────── ────────────────────────────────────────────
~6 allocations                              ~2 allocations
```

`reshape`, `view_as_complex`, `view_as_real`, and `flatten` are all **zero-copy views** in PyTorch — they reinterpret existing memory without copying. `cat` and elementwise ops always allocate new tensors.

Fewer allocations means less pressure on the GPU memory allocator and fewer round-trips through memory bandwidth — which matters more than FLOPs at the sizes typical of LLM inference.

---

#### The duplication cost — Llama3's biggest overhead

`cat(freqs, freqs)` doubles the frequency table from `[B, S, head_dim/2]` to `[B, S, head_dim]`. This is not a free view — it copies data. The cos and sin tables are then each `[B, S, head_dim]`, twice the size they need to be.

Llama4 never duplicates. `freqs_cis` stays at `[B, S, head_dim/2]` complex. The two float32 values per complex number are the cos and sin — same total bytes as one `[B, S, head_dim]` real tensor, but without the copy or the separate sin tensor.

```
Llama3 frequency memory footprint:
  cos tensor: B × S × head_dim floats
  sin tensor: B × S × head_dim floats
  total:      2 × B × S × head_dim floats

Llama4 frequency memory footprint:
  freqs_cis:  B × S × head_dim/2 complex  =  B × S × head_dim floats (one complex = 2 floats)
  total:      1 × B × S × head_dim floats
```

Llama4 uses half the memory for the frequency tables while storing the same information.

---

#### The rotate_half cost — a hidden memory write

`rotate_half` constructs `[-y₀, -y₁, ..., x₀, x₁, ...]` by calling `torch.cat((-x2, x1), dim=-1)`. That `cat` allocates a full new `[B, heads, S, head_dim]` tensor — which is the largest tensor in the whole attention block. It then gets discarded after the multiply-add. This is a write-then-immediately-discard pattern that burns memory bandwidth.

Llama4's `reshape(-1, 2)` + `view_as_complex` does the same conceptual grouping of pairs with zero allocation — they are just different interpretations of the same memory.

---

#### Where Llama3 wins — the fused kernel hook

Despite more allocations, Llama3's `apply_rotary_pos_emb` has this decorator:

```python
@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    ...
```

This allows the entire function to be replaced at runtime with a **fused CUDA or Triton kernel** from the Hub. Fused kernels matter because they eliminate every intermediate allocation:

```
Without fusion (what we've been counting):
  Step 1: allocate rotate_half output → write to GPU memory
  Step 2: allocate q*cos output       → write to GPU memory
  Step 3: allocate rotate_half*sin    → write to GPU memory
  Step 4: allocate final sum          → write to GPU memory

With a fused kernel:
  One kernel reads q, cos, sin once
  Computes all four steps in registers
  Writes the final result once
  Zero intermediate allocations
```

Fused kernels are 2–4× faster than the naive PyTorch sequence in practice because they avoid the round-trips through GPU memory. Libraries like FlashAttention and xFormers ship exactly these fused RoPE kernels.

Llama4's `apply_rotary_emb` has **no such decorator** — it always runs the Python/PyTorch path. PyTorch's `torch.compile` can fuse it, but there is no explicit hook for a hand-written kernel.

---

#### Summary

| | Llama3 (`rotate_half`) | Llama4 (complex) |
|---|---|---|
| FLOPs per pair | 6 (4 mul + 2 add) | 6 (4 mul + 2 add) |
| Intermediate allocations | ~6 | ~2 |
| Frequency table memory | 2 × `[B,S,head_dim]` real | 1 × `[B,S,head_dim/2]` complex |
| `cat` copies in hot path | 2 (`emb` + `rotate_half`) | 0 |
| Fused kernel hook | Yes — `@use_kernel_func_from_hub` | No |
| With a fused kernel | Fastest in practice | Not applicable |
| Without fusion (PyTorch only) | More allocations, more memory | Fewer allocations, half the freq memory |

**Bottom line:** in pure PyTorch, Llama4's complex approach is cleaner and uses less memory. In a production deployment where a fused CUDA kernel is loaded (which is the common case for Llama3-family models via FlashAttention), Llama3's real-valued approach wins decisively because the kernel eliminates all intermediate allocations and does everything in registers. Llama4's complex approach trades the fused-kernel advantage for cleaner code and a smaller memory footprint on the frequency tables.

---

## Step 3 — RoPE vs NoPE per Layer (Llama4 only)

**Llama3:** every layer uses RoPE, no exceptions.

**Llama4:** layers alternate between RoPE and NoPE based on `config.no_rope_layers[layer_idx]`:

```python
self.use_rope = config.no_rope_layers[layer_idx]   # 1=RoPE, 0=NoPE

# In forward:
if self.use_rope:
    query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)
# else: skip entirely — no rotation applied
```

With default `no_rope_layer_interval=4`, every 4th layer is a NoPE layer (e.g. layers 3, 7, 11, ... are NoPE). NoPE layers skip the RoPE rotation completely and rely on two compensating mechanisms instead: chunked causal masking (Step 5) and attention temperature tuning (Step 6).

---

## Step 4 — QK L2 Norm (Llama4 only)

**Llama3:** none.

**Llama4:** on RoPE layers, optionally applies `Llama4TextL2Norm` to Q and K after rotation:

```python
if self.config.use_qk_norm and self.use_rope:
    self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

# In forward, after RoPE:
if hasattr(self, "qk_norm"):
    query_states = self.qk_norm(query_states)
    key_states   = self.qk_norm(key_states)
```

`Llama4TextL2Norm` is identical in formula to `LlamaRMSNorm` but has **no learnable weight** — it only normalises to unit RMS, it doesn't rescale. This prevents RoPE rotation from pushing Q/K magnitudes unevenly across positions, which would destabilise attention logits at long context lengths.

`use_qk_norm=True` is the default for Scout 17B-16E; `use_qk_norm=False` for the 128-expert Maverick variant.

---

## Step 5 — Causal Mask

**Llama3:** one mask, computed once, used by all layers:
```python
causal_mask = create_causal_mask(
    config, inputs_embeds, attention_mask, past_key_values, position_ids
)
# passed to every decoder layer unchanged
```

**Llama4:** two masks computed once, dispatched per layer type:
```python
causal_mask_mapping = {
    "full_attention":    create_causal_mask(**mask_kwargs),
    "chunked_attention": create_chunked_causal_mask(**mask_kwargs),
}

# Each layer picks its mask:
decoder_layer(
    attention_mask=causal_mask_mapping[self.config.layer_types[i]],
    ...
)
```

`create_chunked_causal_mask` builds a block-diagonal mask where each block covers `attention_chunk_size=8192` tokens. Within a block, attention is standard causal; tokens in different blocks cannot attend to each other at all. This gives NoPE layers a bounded local window without position encoding — a NoPE layer at token 50000 doesn't see token 1.

RoPE layers always get `full_attention` (standard causal, unlimited lookback). NoPE layers always get `chunked_attention`. The dispatch is by `config.layer_types[i]`, a list of `"full_attention"` / `"chunked_attention"` strings derived from `no_rope_layers` at config init.

---

## Step 6 — Attention Temperature Tuning (Llama4 NoPE layers only)

**Llama3:** none.

**Llama4:** on NoPE layers, query states are scaled by a position-dependent factor before the attention dot product:

```python
if self.attn_temperature_tuning and not self.use_rope:
    past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device) + past_seen_tokens

    attn_scales = (
        torch.log1p(torch.floor((positions.float() + 1.0) / self.floor_scale)) * self.attn_scale + 1.0
    )
    query_states = (query_states * attn_scales.view(1, -1, 1, 1)).to(query_states.dtype)
```

Formula: `scale(p) = log1p(floor((p+1) / 8192)) × 0.1 + 1.0`

The scale is 1.0 (no change) for the first 8192 tokens, then grows logarithmically. This keeps NoPE attention selective at long positions where, without position encoding, all token pairs look increasingly similar. The `past_seen_tokens` offset ensures generation step N correctly uses position N's scale, not position 0.

---

## Step 7 — Attention Score Computation

**Llama3:**
```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# Upcast to float32 for softmax — avoids bfloat16 overflow on large logits
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
attn_output  = torch.matmul(attn_weights, value_states)
```

**Llama4:**
```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# NO float32 upcast — runs softmax in whatever dtype attn_weights is (bfloat16)
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
attn_output  = torch.matmul(attn_weights, value_states)
```

The only difference is the float32 upcast in softmax. Llama3 upcasts because bfloat16 can overflow on `e^x` when attention scores are large (bfloat16 max ≈ 38912, `e^11` already overflows). Llama4 drops the upcast — a deliberate choice noted in the code comment "llama4 doesn't cast attn weights to fp32." This saves memory and speeds up the softmax step at the cost of a small risk of overflow on extreme logit values.

---

## Step 8 — Output Projection + Residual

Structurally identical in both:
```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
hidden_states = residual + attn_output
```

Dimensions differ (`hidden_size` 4096 vs 5120), everything else the same.

---

## Step 9 — KV Cache

Both use `DynamicCache` with the same `past_key_values.update(key_states, value_states, layer_idx)` call. No difference in caching mechanism, only in what's cached (K/V are `[B, 8, T, 128]` in both, growing by 1 per generation step).

---

## Step 10 — Feed-Forward Network

**Llama3 (always dense SwiGLU):**
```python
# LlamaMLP — every layer, same structure
gate = silu(gate_proj(x))    # [B, S, 4096] → [B, S, 11008]
out  = gate * up_proj(x)
return down_proj(out)        # [B, S, 11008] → [B, S, 4096]
```

**Llama4 (dense or MoE, per layer):**
```python
# Dense layers — Llama4TextMLP
gate = silu(gate_proj(x))    # [B, S, 5120] → [B, S, 16384]
out  = gate * up_proj(x)
return down_proj(out)        # [B, S, 16384] → [B, S, 5120]

# MoE layers — Llama4TextMoe
# router → 16 experts (batched bmm) + shared_expert → sum
# see MoE.md for full detail
```

This is the second biggest structural change after RoPE/NoPE. Llama3 has one FFN type; Llama4 has two, selected per layer at construction. With default `interleave_moe_layer_step=1`, every layer in Llama4 is a MoE layer.

---

## Step 11 — Normalisation

Both use the same RMSNorm formula. Different class names, identical implementation:

```python
# Llama3: LlamaRMSNorm
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)

# Llama4: Llama4TextRMSNorm (same math, different name)
output = self._norm(x.float()).type_as(x)
return output * self.weight
```

The `@use_kernel_forward_from_hub("RMSNorm")` decorator on `LlamaRMSNorm` allows the kernel to be swapped for a fused CUDA/Triton implementation. `Llama4TextRMSNorm` lacks this decorator — it always runs the plain PyTorch path.

---

## Step 12 — Flash Attention Support

**Llama3:** `_supports_flash_attn = True` — FlashAttention-2 can be used as the attention backend.

**Llama4:** `_supports_flash_attn = False` — explicitly disabled. SDPA and FlexAttention are supported; FlashAttention-2 is not. The reasons aren't documented inline, but likely relate to the non-standard attention patterns (chunked masks, temperature-scaled Q) in NoPE layers that FlashAttention-2 kernels don't natively support.

---

## What Is Identical Between the Two

- GQA with 8 KV heads and `repeat_kv`
- `ALL_ATTENTION_FUNCTIONS` backend dispatch pattern
- `attention_bias=False` default on all projections
- SwiGLU activation in dense MLP layers
- `DynamicCache` KV caching mechanism
- Residual connection pattern around attention and FFN
- `logits_to_keep` slicing in the LM head for inference efficiency
- `GenerationMixin` for `.generate()` — greedy, sampling, beam search
- `create_causal_mask` from `masking_utils.py` (Llama4 also adds `create_chunked_causal_mask`)

---

## Full Comparison Table

| | Llama3 | Llama4 |
|---|---|---|
| RoPE implementation | real `rotate_half` + `cat(freqs, freqs)` | complex `torch.polar` + `view_as_complex` |
| Memory layout | split-half (`[x0..xN, y0..yN]`) | consecutive pairs (`[(x0,y0)..]`) |
| `position_embeddings` type passed to layers | `(cos, sin)` tuple | `freqs_cis` complex tensor |
| NoPE layers | No — all layers use RoPE | Yes — every `no_rope_layer_interval`-th layer |
| Attention mask | one mask, all layers | two masks dispatched by `layer_types[i]` |
| QK L2 norm | None | Yes, on RoPE layers (no learnable weight) |
| Temperature tuning | None | Yes, on NoPE layers (log-scale with position) |
| Softmax fp32 upcast | Yes (`dtype=torch.float32`) | **No** |
| Flash attention | Supported | **Disabled** |
| FFN | Always dense SwiGLU | Dense or MoE per layer |
| RMSNorm kernel hook | `@use_kernel_forward_from_hub` | Not decorated |
| Config structure | Single `LlamaConfig` | `Llama4Config` → `Llama4TextConfig` + `Llama4VisionConfig` |
| Multimodal | No | Yes (`Llama4ForConditionalGeneration`) |
