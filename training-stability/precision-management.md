# Precision Management

**What it is:** Weights stay in full 32-bit precision (fp32) for optimizer accuracy, while activations and computation run in 16-bit brain float (bf16) for speed — applied explicitly via a global COMPUTE_DTYPE rather than PyTorch's automatic mixed precision.

**Code:** `gpt.py:45-50`, `gpt.py:424`

```python
# gpt.py:45-50 — the custom Linear class that replaces autocast
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))

# gpt.py:424 — activations explicitly cast to COMPUTE_DTYPE at the start of forward()
x = self.transformer.wte(idx)        # embed current token
x = x.to(COMPUTE_DTYPE)             # ensure activations are in compute dtype
x = norm(x)
```

---

## Number Formats: fp32 vs bf16

### fp32 (32-bit float)

```
Bits: 1 sign | 8 exponent | 23 mantissa
Range: ~1.2×10⁻³⁸ to 3.4×10³⁸
Precision: ~7 significant decimal digits

Example: 3.14159265  (stored accurately)
         0.00000001  (stored accurately)
```

### bf16 (Brain Float 16)

```
Bits: 1 sign | 8 exponent | 7 mantissa
Range: same as fp32 (same exponent bits)
Precision: ~2-3 significant decimal digits

Example: 3.14159265  →  stored as  3.140625  (less precise)
         0.00000001  →  stored as  0.0        (rounds to zero!)
```

bf16 keeps the same exponent range as fp32 (so no overflow), but halves the mantissa — meaning less precision for small differences.

### Why care about precision?

```
fp32 weight update example:
  weight = 0.123456789
  gradient = -0.000001234
  new weight = 0.123456789 - 0.000001234 = 0.123455555

In bf16:
  weight ≈ 0.1234
  gradient ≈ 0.0  ← rounds to zero! gradient is lost
  new weight = 0.1234  ← no update happened
```

Small gradient updates (which are common late in training) get lost in bf16. This is why **weights must stay in fp32**.

---

## The Standard Approach: torch.autocast

PyTorch's `torch.autocast` automatically switches between fp32 and bf16:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
```

PyTorch decides internally which operations run in bf16 and which stay in fp32.

### Problems with autocast

```
1. Unpredictable:
   Different ops get different dtypes depending on PyTorch version.
   Hard to reason about exactly where precision is maintained.

2. Implicit:
   You can't easily inspect or control where casting happens.
   Debugging numerical issues is harder.

3. Overhead:
   The context manager adds a small overhead to every operation
   as it checks whether to cast.
```

---

## This Model's Approach: Explicit COMPUTE_DTYPE

```python
# gpt.py:45-50

COMPUTE_DTYPE = torch.bfloat16   # on H100
# COMPUTE_DTYPE = torch.float32  # on CPU / Apple MPS

# Weights are created in fp32:
self.linear = nn.Linear(in, out)          # fp32 weights

# Activations are explicitly cast to COMPUTE_DTYPE at use time:
x = x.to(COMPUTE_DTYPE)
y = self.linear(x)                         # input is bf16, output is bf16
```

### What this means step by step

```
Storage (in memory):
  Weight matrix W    → fp32  (precise, full 32 bits)
  Bias (if any)      → fp32

At compute time:
  Input activation x → cast to bf16
  Matrix multiply: x_bf16 × W_fp32
  PyTorch/CUDA handles the mixed precision multiply natively on H100
  Output → bf16

Optimizer step (after backward pass):
  Gradients are accumulated in fp32
  Weight update: W_fp32 -= lr × grad_fp32  ← full precision update
```

---

## Walkthrough: Processing "cat" Through a Linear Layer

### Stored values (fp32)

```
Input activation x (coming from previous layer):
  fp32: [0.40000001, 0.60000002, -0.19999999, 0.80000001]

Weight matrix W (fp32):
  [[ 0.51234567, -0.23456789],
   [ 0.78901234,  0.12345678],
   [-0.34567890,  0.56789012],
   [ 0.90123456, -0.45678901]]
```

### Compute step (cast to bf16 first)

```
x_bf16 = x.to(bf16):
  [0.3984375, 0.59375, -0.203125, 0.796875]
  (precision lost, but close enough for forward pass)

output = x_bf16 × W  (W stays fp32 internally, CUDA handles mixed multiply)
output ≈ [0.5078125, -0.296875]   ← bf16 result
```

### Gradient and weight update (back to fp32)

```
Gradient for this layer (fp32):
  dL/dW = [[0.00012345, -0.00023456],
            [0.00034567,  ...]]

Weight update:
  W_new = W - lr × dL/dW
        = 0.51234567 - 0.001 × 0.00012345
        = 0.51234567 - 0.00000012345
        = 0.51234554655   ← full precision preserved

In bf16 this would be:
  W ≈ 0.5117  (too coarse to represent 0.51234554655)
  gradient ≈ 0.0  (rounds to zero → no update!)
```

This is why weight storage in fp32 is critical.

---

## Hardware-Dependent Dtype

```python
if device == "cuda" and is_hopper_gpu():
    COMPUTE_DTYPE = torch.bfloat16   # H100: use bf16 for speed
elif device == "cpu" or device == "mps":
    COMPUTE_DTYPE = torch.float32    # CPU/Apple: no bf16 benefit, stay fp32
```

On H100, bf16 matmuls run on tensor cores at roughly **2× the throughput** of fp32 matmuls. On CPU or Apple MPS, bf16 doesn't have the same hardware acceleration, so fp32 is used throughout.

---

## Why bf16 Instead of fp16?

```
fp16 (half precision):
  Bits: 1 sign | 5 exponent | 10 mantissa
  Range: much smaller! Max value ≈ 65,504
  Problem: activations can easily exceed 65,504 → overflow → NaN → training crash

bf16 (brain float 16):
  Bits: 1 sign | 8 exponent | 7 mantissa
  Range: same as fp32 (max ≈ 3.4×10³⁸) ← no overflow risk
  Problem: less precise, but doesn't explode

bf16 was designed specifically for deep learning:
  "Give me the range of fp32, just with less precision"
  It almost never overflows, making it much safer than fp16.
```

---

## Summary

| | Weights (stored) | Activations (compute) | Optimizer |
|---|---|---|---|
| Standard autocast | fp32 internally | Mixed (auto) | fp32 |
| **This model** | **fp32** | **bf16 explicitly** | **fp32** |
| Why | Small gradient updates need full precision | bf16 matmuls are 2× faster on H100 | Accurate weight updates |

> **One-line summary:** Weights live in fp32 so tiny gradient updates aren't lost, activations are explicitly cast to bf16 for fast tensor core computation on H100 — with no autocast magic, just explicit dtype management at every step.
