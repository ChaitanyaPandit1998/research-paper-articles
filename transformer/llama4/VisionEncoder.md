# Llama4 Vision Encoder

Llama4 is a vision-language model. The vision side is a ViT-style encoder that processes images and produces token sequences that get merged into the text embedding stream.

---

## Overall Pipeline

```
pixel_values [B*tiles, C, H, W]
       │
       ▼
Llama4VisionModel
  ├── Llama4UnfoldConvolution      patch embedding (unfold + linear)
  ├── class_embedding (CLS token)  appended to patches
  ├── positional_embedding_vlm     learned absolute position embeddings
  ├── LayerNorm (pre)
  ├── Llama4VisionEncoder          34× ViT transformer blocks
  │     └── each block: VisionAttention (RoPE) + VisionMLP + LayerNorm (no RMSNorm)
  ├── LayerNorm (post)
  └── Llama4VisionPixelShuffleMLP  pixel shuffle token reduction + projection
       │
       ▼
image_features [B*tiles, reduced_patches, vision_output_dim=7680]
       │
       ▼
Llama4MultiModalProjector (single Linear)
       │
       ▼
projected_features [B*tiles, reduced_patches, text_hidden_size=5120]
       │
       ▼
merged into text inputs_embeds at image_token_index positions
```

---

## 1. Patch Embedding — `Llama4UnfoldConvolution`

Most ViT implementations use a `Conv2d` with `kernel_size=patch_size, stride=patch_size` to extract patches. Llama4 uses `torch.nn.Unfold` instead:

```python
class Llama4UnfoldConvolution(nn.Module):
    def __init__(self, config):
        kernel_size = config.patch_size   # 14×14
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        self.linear = nn.Linear(
            config.num_channels * kernel_size[0] * kernel_size[1],  # 3 * 14 * 14 = 588
            config.hidden_size,   # 768
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.unfold(hidden_states)   # [B, C*k*k, num_patches]
        hidden_states = hidden_states.permute(0, 2, 1)   # [B, num_patches, C*k*k]
        return self.linear(hidden_states)            # [B, num_patches, hidden_size=768]
```

---

### Line-by-line breakdown

#### `__init__`

---

**`kernel_size = config.patch_size`**

`patch_size` is 14, so `kernel_size = 14`. This defines the size of the window that slides over the image — each patch is a 14×14 pixel tile. In practice `kernel_size` is used as a tuple `(14, 14)` by `nn.Unfold`.

---

**`self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)`**

`torch.nn.Unfold` is the core of this class. It is worth understanding what it does precisely.

Given an image tensor of shape `[B, C, H, W]`, Unfold slides a `kernel_size × kernel_size` window across the image with the given `stride`, and for every window position it flattens all pixel values — across all channels — into a single vector.

With `kernel_size=14` and `stride=14` (stride equals kernel, so windows do not overlap):
```
Image:     [B, 3, 448, 448]
Windows:   32 × 32 = 1024 non-overlapping 14×14 tiles
Per tile:  3 channels × 14 × 14 pixels = 588 values flattened
Output:    [B, 588, 1024]
              ↑    ↑
              │    └── one column per patch position
              └── all pixel values for that patch (flattened)
```

The key thing to notice: Unfold puts patches in the **last** dimension (`1024`), not the second. Each column of the output is one patch. This is the opposite of what the transformer encoder expects (which wants patches as rows in dimension 1), which is why `permute` is needed in `forward`.

Setting `stride = kernel_size` is what makes patches non-overlapping. If stride were smaller than kernel_size, patches would overlap and the patch count would be larger.

---

**`self.linear = nn.Linear(3 * 14 * 14, 768, bias=False)`**

```
Input features:   3 * 14 * 14 = 588   (all pixel values in one patch, all 3 RGB channels)
Output features:  768                  (the model's hidden dimension)
bias=False                             (standard in ViT-style patch projections)
```

This is the **patch projection** layer — it maps each raw 588-dim patch vector into the 768-dim space that the transformer encoder operates in. Every patch gets the same linear transformation applied independently.

`bias=False` is conventional in vision transformers following ViT. The positional embeddings added immediately after provide sufficient bias signal, so a separate bias term here is redundant.

**Why not `Conv2d`?** A `Conv2d` with `kernel_size=14, stride=14, out_channels=768` would do mathematically identical work — PyTorch even implements `Conv2d` internally using `Unfold`. Llama 4 uses explicit `Unfold + Linear` to make the two steps (patch extraction and projection) clearly separate and independently inspectable.

---

#### `forward`

---

**`hidden_states = self.unfold(hidden_states)`**

Input shape coming in: `[B, 3, 448, 448]`

After Unfold:
```
[B, 588, 1024]
  ↑    ↑    ↑
  │    │    └── 1024 patch positions (32 × 32 grid)
  │    └── 588 values per patch (3 × 14 × 14 pixels, flattened)
  └── batch size
```

Think of the output as a table: 1024 columns (one per patch), 588 rows (one per flattened pixel value). Each column is the raw pixel content of one 14×14 tile.

---

**`hidden_states = hidden_states.permute(0, 2, 1)`**

`permute(0, 2, 1)` swaps dimensions 1 and 2, leaving dimension 0 (batch) unchanged:

```
[B, 588, 1024]  →  [B, 1024, 588]
     ↑    ↑              ↑    ↑
  pixels patches      patches pixels
```

After permute, each **row** is one patch (1024 rows), and each row contains its 588 pixel values. This is the shape the transformer expects — a sequence of vectors, one per token.

**Why `permute` and not `transpose`?**

For this specific case they are identical — you could write either:

```python
hidden_states.permute(0, 2, 1)   # equivalent
hidden_states.transpose(1, 2)    # equivalent
```

Both return a non-contiguous view of the same data with dimensions 1 and 2 swapped. No copy, no difference in output.

The real difference is scope:

- `transpose(dim0, dim1)` — swaps exactly two dimensions, nothing else
- `permute(*dims)` — reorders ALL dimensions at once in any order

`permute` becomes necessary when you need to move more than two dimensions in one step. In attention code, shapes like `[B, heads, seq, depth]` are common and often need multi-dimension reordering:

```python
x = torch.randn(B, heads, seq, depth)   # [B, H, S, D]

# Want [B, S, H, D] — two dims moving simultaneously

x.permute(0, 2, 1, 3)     # one clean call
x.transpose(1, 2)          # also works here, but chains get messy with more dims
```

`permute` is the convention throughout ViT-style code because the same codebase already uses it heavily for multi-head attention reshapes — consistency favours it over `transpose`. There is no performance difference between the two.

---

**`return self.linear(hidden_states)`**

`nn.Linear` always operates on the **last dimension**. Applied to `[B, 1024, 588]`, it transforms 588 → 768 independently for each of the 1024 patches:

```
[B, 1024, 588]  →  [B, 1024, 768]
                          ↑
                      hidden_size
```

Each patch is now a 768-dim vector in the transformer's embedding space. The 1024 patches form a sequence that gets passed to the CLS token step and then the encoder blocks.

---

### Shape summary

```
Input image          [B, 3, 448, 448]
after unfold         [B, 588, 1024]     ← patches in last dim
after permute        [B, 1024, 588]     ← patches in sequence dim
after linear         [B, 1024, 768]     ← projected to hidden size
```

---

## 2. CLS Token and Positional Embeddings

### The code

```python
self.num_patches = (image_size // patch_size) ** 2 + 1   # 1024 + 1 = 1025 (includes CLS)
self.scale = config.hidden_size ** -0.5                    # 1/sqrt(768) ≈ 0.036

self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
self.positional_embedding_vlm = nn.Parameter(self.scale * torch.randn(self.num_patches, self.hidden_size))
```

In `forward`:
```python
# Append CLS token
class_embedding = self.class_embedding.expand(batch, 1, hidden_dim)
hidden_state = torch.cat([hidden_state, class_embedding], dim=1)   # [B, 1025, 768]

# Add absolute positional embeddings
hidden_state = hidden_state + positional_embedding_vlm   # learned, [1025, 768] → broadcast
```

---

### Line-by-line breakdown

**`self.num_patches = (image_size // patch_size) ** 2 + 1`**

The image is 448×448. Each patch is 14×14.
```
448 // 14 = 32        → 32 patches fit along each side
32 ** 2   = 1024      → 1024 patches tile the whole image
+ 1                   → one extra slot reserved for the CLS token
= 1025
```
This number sizes every buffer that operates over the full token sequence.

---

**`self.scale = config.hidden_size ** -0.5`**

`hidden_size ** -0.5` is `1 / √hidden_size`. With `hidden_size = 768`:
```
1 / √768 ≈ 1 / 27.7 ≈ 0.036
```
This is a weight initialisation scale factor. `torch.randn` produces values with variance 1. Multiplying by `1/√d` shrinks the variance to `1/d`, preventing initial dot products in attention from being too large — the same reasoning as the `√d_k` denominator in the attention equation.

---

**`self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))`**

```
torch.randn(768)        → random vector of shape [768], values ~ N(0, 1)
* self.scale            → multiply by 0.036, shrinking variance to 1/768
nn.Parameter(...)       → register as a trainable parameter
```
This is the CLS token's initial value — a single vector of length 768. It has no semantic content at initialisation. It learns its role entirely through backpropagation. Because it is an `nn.Parameter`, PyTorch includes it in `model.parameters()` and updates it with every gradient step.

---

**`self.positional_embedding_vlm = nn.Parameter(self.scale * torch.randn(self.num_patches, self.hidden_size))`**

Same pattern, but a 2D matrix:
```
torch.randn(1025, 768)  → one 768-dim vector per token slot (1024 patches + 1 CLS)
* self.scale            → same initialisation scaling
nn.Parameter(...)       → fully learnable
```
Each of the 1025 positions gets its own learned positional vector. Unlike the sinusoidal encoding in the original Transformer paper, these are not computed from a formula — they are free parameters trained to encode whatever positional signal is useful.

---

**`class_embedding = self.class_embedding.expand(batch, 1, hidden_dim)`**

`self.class_embedding` has shape `[768]` — a single vector. `hidden_state` is `[B, 1024, 768]`. To append CLS to each image in the batch it must be reshaped:
```
self.class_embedding            →  [768]
.expand(batch, 1, hidden_dim)   →  [B, 1, 768]
```
`expand` is memory-efficient — it does not copy the data B times. It creates a view that appears to have batch size B while pointing to the same underlying storage.

---

**`hidden_state = torch.cat([hidden_state, class_embedding], dim=1)`**

Concatenate along `dim=1` — the sequence dimension:
```
hidden_state      [B, 1024, 768]   ← 1024 patch tokens
class_embedding   [B,    1, 768]   ← 1 CLS token
─────────────────────────────────
result            [B, 1025, 768]   ← CLS appended at the end
```
Note: most vision models (CLIP, ViT) *prepend* CLS at position 0. Llama 4 appends it at the end. The attention mechanism doesn't care — CLS still attends to every patch regardless. What changes is how the positional encoding handles it, which is why Llama 4 marks CLS with a special ID `-2` and assigns it zero RoPE frequency (no spatial position, since CLS doesn't correspond to any image region).

---

**`hidden_state = hidden_state + positional_embedding_vlm`**

`positional_embedding_vlm` has shape `[1025, 768]`. `hidden_state` has shape `[B, 1025, 768]`. PyTorch broadcasts across the batch dimension automatically:
```
[B, 1025, 768]   ← patch + CLS token vectors (content only so far)
+  [1025, 768]   ← one positional vector per slot (broadcast across B)
=  [B, 1025, 768] ← each token now carries both content and position
```
Every token in every image in the batch gets the same positional signal added to it.

---

After the encoder runs, the CLS token is stripped before further processing:
```python
hidden_state = hidden_state[:, :-1, :]   # remove CLS, keep the 1024 patch tokens
```
The downstream multimodal model works with the 1024 context-enriched patch tokens. CLS served its purpose inside the encoder — giving the self-attention layers a global aggregation slot — and is then discarded. The patch tokens carry its influence even after it is removed, because they attended to it across all layers.

---

### What the CLS token actually is — and where it came from

**Invented for BERT (2018).** BERT processes every token in a sentence and produces one contextual vector per token. For classification tasks you need a *single* summary vector — but which token do you pick? The solution: prepend a special learnable token `[CLS]` with no semantic content. Because BERT uses full bidirectional attention, by the final layer CLS has attended to every other token and absorbed information from the whole sequence. Its output becomes the natural classification summary.

```
Input:  [CLS]  The   company   reported   strong   earnings
           ↕     ↕      ↕         ↕          ↕        ↕
Output:  h_CLS  h_1   h_2       h_3        h_4      h_5
                                                    
→ h_CLS is the single summary vector passed to the classification head
```

**Copied into vision by ViT (2020).** ViT treats images as sequences of patch tokens. The problem is identical to BERT's — one vector per patch, but you need one vector for the whole image. The CLS token was transplanted unchanged: prepend a learnable slot, run bidirectional attention across all patches, use the CLS output as the image representation.

```
[CLS]  patch_1  patch_2  ... patch_1024
  ↕       ↕        ↕             ↕
h_CLS   h_1      h_2           h_1024

→ h_CLS → classification head → "this is a cat"
```

**What CLS actually learns in vision.** In early layers it attends broadly across all patches. By the final layer its attention sharpens onto the most semantically informative regions — the foreground object, not the background. The DINO paper (2021) demonstrated that these CLS attention maps produce clean object outlines with no segmentation supervision — a global summary token accidentally became a free saliency map.

**Alternatives that emerged.**
- **Global Average Pooling (GAP)** — average all patch representations instead of using a special token. Simpler, no special token needed. Many modern ViT variants use this.
- **Register tokens (2023, Meta/FAIR)** — large ViTs develop "artifacts" where CLS and certain patches accumulate disproportionate global attention, corrupting the spatial structure of other tokens. The fix: add dedicated learnable register tokens alongside CLS to give the model explicit global-information slots. Now standard in DINOv2 and subsequent vision encoders.

**The lineage:**
```
BERT (2018)       ViT (2020)          DINO (2021)           Register Tokens (2023)
CLS for text  →   CLS for image   →   CLS learns        →   Multiple registers replace
classification    classification       segmentation           corrupted CLS artifacts
                                       for free
```

**One-line summary:** CLS is a learnable "summary slot" with no initial meaning — bidirectional attention forces it to collect information from every other token by the final layer, making it the natural single-vector representation for the whole sequence. Invented for text in BERT, transplanted into vision in ViT, refined as researchers discovered both its power (free saliency maps) and its failure mode (attention artifacts at scale).

---

## 3. Vision RoPE — 2D Positional Encoding for Patches

The vision encoder uses a **different RoPE implementation** from the text decoder. Patch tokens are 2D (they have x and y coordinates in the image grid), so the vision RoPE encodes **both x and y positions separately** and concatenates them.

```python
class Llama4VisionRotaryEmbedding(nn.Module):
    @staticmethod
    def _compute_freqs_ci(config):
        idx = config.image_size // config.patch_size   # 448 / 14 = 32
        img_idx = torch.arange(idx**2).reshape(idx**2, 1)  # [1024, 1] — flat patch indices
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)  # [1025, 1] — extra row for CLS
        img_idx[-1, -1] = -2   # mark CLS with ID -2

        frequencies_x = img_idx % idx    # x coordinate per patch: [0..31]
        frequencies_y = img_idx // idx   # y coordinate per patch: [0..31]

        freq_dim = config.hidden_size // config.num_attention_heads // 2  # 768/16/2 = 24
        rope_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, 2)[:freq_dim//2] / freq_dim))

        # Build freqs for x and y separately
        freqs_x = ((frequencies_x + 1) * rope_freq).repeat_interleave(2, dim=-1)  # [1025, 24]
        freqs_y = ((frequencies_y + 1) * rope_freq).repeat_interleave(2, dim=-1)  # [1025, 24]
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]  # [1025, 24]

        # CLS positions get zero frequency (no position encoding)
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)

        # Convert to complex representation
        freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
        return freq_cis   # [1025, 1025, freq_dim//2] complex
```

---

### Line-by-line breakdown

---

**`idx = config.image_size // config.patch_size`**

```
448 // 14 = 32
```

Number of patches along one side of the image. The full grid is 32×32 = 1024 patches. `idx` is reused throughout as the grid dimension for both x and y.

---

**`img_idx = torch.arange(idx**2).reshape(idx**2, 1)`**

```
torch.arange(1024)      → [0, 1, 2, ..., 1023]   flat index for every patch
.reshape(1024, 1)       → [[0], [1], ..., [1023]]  column vector, shape [1024, 1]
```

Each row is the flat index of one patch in the 32×32 grid (reading left-to-right, top-to-bottom). The reshape to `[1024, 1]` is deliberate — it lets the `%` and `//` operations later broadcast correctly against `rope_freq` (which is `[12]`), producing `[1024, 12]` frequency matrices.

---

**`img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)`**

```
img_idx[:1]             → [[0]]        first row, shape [1, 1], placeholder for CLS
cat along dim=0         → [1025, 1]    appends one extra row at the end
```

Adds a slot for the CLS token. The value `[[0]]` is a placeholder — the next line immediately overwrites it.

---

**`img_idx[-1, -1] = -2`**

Sets the last element to `-2` — a sentinel value that means "CLS, not a real patch position". The value `-2` was chosen specifically to be negative, which triggers `img_idx < 0` in the `masked_fill` step later. CLS has no spatial location in the image grid, so its position frequencies must be zeroed out rather than computed.

---

**`frequencies_x = img_idx % idx`**

```
% 32  →  column index (x-coordinate) for each patch
```

For a flat index `i` in a 32×32 grid, `i % 32` gives the column it sits in:

```
patch 0   → 0 % 32 = 0   (leftmost column)
patch 1   → 1 % 32 = 1
patch 31  → 31 % 32 = 31  (rightmost column)
patch 32  → 32 % 32 = 0   (back to leftmost, second row)
```

Shape: `[1025, 1]`. CLS: `-2 % 32 = 30` in Python — a junk value that gets zeroed out later by `masked_fill`.

---

**`frequencies_y = img_idx // idx`**

```
// 32  →  row index (y-coordinate) for each patch
```

`i // 32` gives the row:

```
patches 0–31   → y = 0  (top row)
patches 32–63  → y = 1
patches 992–1023 → y = 31  (bottom row)
```

Shape: `[1025, 1]`. Together, `frequencies_x` and `frequencies_y` recover the 2D grid coordinates from the flat index — equivalent to `(i % 32, i // 32)`.

---

**`freq_dim = config.hidden_size // config.num_attention_heads // 2`**

```
768 // 16 = 48    head dimension (each attention head operates on 48 values)
48  // 2  = 24    split in half — 24 dims for x-position, 24 dims for y-position
```

RoPE encodes position by rotating pairs of dimensions. With 48 dims per head and 2D position, half the dims encode x and the other half encode y. `freq_dim = 24` is the number of frequency components needed for one spatial axis.

---

**`rope_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, 2)[:freq_dim//2] / freq_dim))`**

Breaking this apart step by step:

```
torch.arange(0, 24, 2)          → [0, 2, 4, 6, ..., 22]       12 elements
[:freq_dim//2] = [:12]          → same 12 elements
/ freq_dim = / 24               → [0/24, 2/24, ..., 22/24]
                                   = [0.0, 0.083, 0.167, ..., 0.917]

rope_theta ** [0.0, 0.083, ...]  → 10000 ** [0.0, 0.083, ...]
                                   = [1.0, 1.96, 3.83, ..., 7499]

1.0 / [1.0, 1.96, ..., 7499]    → [1.0, 0.51, 0.26, ..., 0.000133]
```

Result shape: `[12]`. This is the same geometric sequence of base frequencies as standard text RoPE — high frequencies (≈1.0) for precise local position, low frequencies (≈0.0001) for coarse global position. Each of the 12 values will be used to rotate one pair of dimensions.

---

**`freqs_x = ((frequencies_x + 1) * rope_freq).repeat_interleave(2, dim=-1)`**

Three operations fused:

**`frequencies_x + 1`** — shift x-coordinates from 0-based to 1-based:
```
x=0 → 1,  x=1 → 2,  ...,  x=31 → 32
```
Without `+1`, the first column (x=0) would produce `0 * rope_freq = 0` for all frequencies. `sin(0) = 0` and `cos(0) = 1` for every dimension — an identity rotation that encodes no positional information. Shifting by 1 ensures every column gets a meaningful, distinct encoding.

**`* rope_freq`** — multiply each position's shifted coordinate by each base frequency:
```
[1025, 1] × [12]  →  [1025, 12]   (broadcasting)
```
Row `i` of the result: the x-coordinate of patch `i`, scaled by each of the 12 base frequencies. This is the angle each dimension pair will rotate by for this patch's x-position.

**`.repeat_interleave(2, dim=-1)`** — duplicate each frequency value:
```
[1025, 12] → [1025, 24]
[a, b, c, ...] → [a, a, b, b, c, c, ...]
```
RoPE applies each frequency to a *pair* of dimensions (real and imaginary parts of a complex rotation). Duplicating ensures each frequency appears twice — once for each element of the pair.

---

**`freqs_y = ((frequencies_y + 1) * rope_freq).repeat_interleave(2, dim=-1)`**

Identical to `freqs_x` but using y-coordinates. Result shape: `[1025, 24]`. Each patch now has both its x and y frequency vectors ready.

---

**`freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]`**

Four chained operations:

**`cat([freqs_x, freqs_y], dim=-1)`** — concatenate x and y frequencies side by side:
```
[1025, 24] + [1025, 24]  →  [1025, 48]
 ← x freqs →   ← y freqs →
```

**`.float()`** — ensure float32. Frequency computation can sometimes produce float16 or bfloat16 depending on config; `view_as_complex` requires float32.

**`.contiguous()`** — ensures memory layout is contiguous. Required for `view_as_complex`, which needs the last dimension to be stored adjacently in memory.

**`[..., ::2]`** — take every other element along the last dimension:
```
[1025, 48]  →  [1025, 24]
indices 0, 2, 4, ..., 46
```

Because `repeat_interleave(2)` doubled each value before the cat, the even-indexed positions `[0, 2, 4, ..., 22]` are the de-duplicated x-frequencies and `[24, 26, ..., 46]` are the de-duplicated y-frequencies. The `[::2]` step strips the duplicates, leaving one copy of each frequency for x and one for y.

---

**`freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)`**

```
img_idx.reshape(-1, 1, 1)  →  [1025, 1, 1]
< 0                        →  boolean mask, True only for CLS row (where img_idx = -2)
masked_fill(..., 0)        →  set those positions to 0.0
```

CLS has no spatial position. Zeroing its frequencies means:
```
cos(0) = 1,  sin(0) = 0  →  complex number (1 + 0j)  →  identity rotation
```
Multiplying any vector by the identity rotation leaves it unchanged — so CLS receives no positional bias from RoPE.

---

**`freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))`**

Three steps:

**`torch.cos(freqs)`, `torch.sin(freqs)`** — compute the cosine and sine of every frequency value. These are the real and imaginary parts of the unit-circle rotation that RoPE applies.

**`torch.stack([cos, sin], dim=-1)`** — pair each cos with its corresponding sin along a new last dimension:
```
[..., 24] + [..., 24]  →  [..., 24, 2]
                              ↑
                    (cos_val, sin_val) pairs
```

**`torch.view_as_complex(...)`** — interpret each `(cos, sin)` pair as a complex number `cos + i·sin`:
```
[..., 24, 2]  →  [..., 24] complex
```

This is the unit complex number `e^(iθ)` for each frequency angle θ. When applied to a query or key vector (via element-wise complex multiplication), it rotates the vector by angle θ — which is exactly what RoPE does to inject positional information. Tokens that are far apart in x or y will have their vectors rotated by very different angles, making the attention dot product naturally sensitive to spatial distance.

---

### Step-by-step: input → output at each line

```
Step | Line                                              | Shape in        | Shape out   | Values
─────────────────────────────────────────────────────────────────────────────────────────────────────
1    | idx = image_size // patch_size                   | scalar          | scalar      | 448//14 = 32

2    | torch.arange(idx**2)                             | —               | [1024]      | [0, 1, 2, ..., 1023]
     | .reshape(idx**2, 1)                              | [1024]          | [1024, 1]   | [[0],[1],...,[1023]]

3    | img_idx[:1]                                      | [1024, 1]       | [1, 1]      | [[0]]  (placeholder)
     | torch.cat([img_idx, img_idx[:1]], dim=0)         | [1024,1]+[1,1]  | [1025, 1]   | [[0],[1],...,[1023],[0]]

4    | img_idx[-1, -1] = -2                             | [1025, 1]       | [1025, 1]   | last row → [[-2]]

5    | frequencies_x = img_idx % 32                    | [1025, 1]       | [1025, 1]   | col index per patch [0..31], CLS→30*
6    | frequencies_y = img_idx // 32                   | [1025, 1]       | [1025, 1]   | row index per patch [0..31], CLS→0*

7    | freq_dim = 768 // 16 // 2                        | scalar          | scalar      | 24

8    | torch.arange(0, 24, 2)[:12] / 24                | —               | [12]        | [0.0, 0.083, ..., 0.917]
     | rope_theta ** [...]                              | [12]            | [12]        | [1.0, 1.96, ..., 7499]
     | 1.0 / [...]  →  rope_freq                        | [12]            | [12]        | [1.0, 0.51, ..., 0.000133]

9    | frequencies_x + 1                               | [1025, 1]       | [1025, 1]   | col shifted 1-based [1..32]
     | * rope_freq                                      | [1025,1]×[12]   | [1025, 12]  | angle per patch per frequency
     | .repeat_interleave(2, dim=-1)  →  freqs_x       | [1025, 12]      | [1025, 24]  | [a,a,b,b,...] each freq duplicated

10   | (same for y)  →  freqs_y                        | [1025, 1]       | [1025, 24]  | same structure, row coords

11   | torch.cat([freqs_x, freqs_y], dim=-1)           | [1025,24]+[1025,24] | [1025, 48] | x-freqs first 24, y-freqs last 24
     | .float().contiguous()                            | [1025, 48]      | [1025, 48]  | (dtype + memory layout)
     | [..., ::2]  →  freqs                             | [1025, 48]      | [1025, 24]  | de-duplicated: 12 x-freqs + 12 y-freqs

12   | img_idx.reshape(-1, 1, 1) < 0                   | [1025, 1]       | [1025,1,1]  | True only for CLS row (val=-2)
     | freqs.masked_fill(..., 0)                        | [1025, 24]      | [1025, 24]  | CLS row → all 0.0, patches unchanged

13   | torch.cos(freqs), torch.sin(freqs)              | [1025, 24]      | [1025, 24]  | cos and sin of each angle
     | torch.stack([cos, sin], dim=-1)                 | [1025,24]+[1025,24] | [1025,24,2] | (cosθ, sinθ) pairs
     | torch.view_as_complex(...)  →  freq_cis         | [1025, 24, 2]   | [1025, 24]† | e^(iθ) — complex rotation per patch
```

`*` CLS junk values (30 and 0) are overwritten to 0.0 by `masked_fill` in step 12 — they never affect the output.
`†` `view_as_complex` collapses the last dim (size 2 = real, imag) into one complex number.

---

**Key differences from text RoPE:**
- **2D, not 1D** — `frequencies_x` (column index) and `frequencies_y` (row index) are encoded separately into the frequency tensor, then concatenated. Each patch gets a position that encodes both where it is horizontally and vertically.
- **Pre-computed at init, not per forward call** — the frequency table is stored as a non-persistent buffer (`register_buffer("freqs_ci", ...)`). It never changes across calls (unlike text RoPE, which recomputes for different `position_ids` each forward pass).
- **CLS token gets zero frequency** — `img_idx[-1, -1] = -2` marks the CLS slot; `masked_fill(img_idx < 0, 0)` zeros out its frequencies. Zero frequency means `cos(0) = 1, sin(0) = 0` → identity rotation → no position encoding for CLS.

The frequency table has shape `[1025, 1025, freq_dim//2]` because the vision encoder performs full bidirectional attention (not causal), and the position encoding is indexed as `freqs_ci[query_position, key_position, :]` rather than just `freqs_ci[position, :]`. This is 2D cross-position encoding, not per-position encoding.

---

### Why these differences exist

#### Why 2D instead of 1D

A flat patch index (0..1023) does not preserve spatial structure. Consider two pairs of patches that are both "1 step apart" in flat index:
- patch 5 and patch 6 — adjacent horizontally (same row, next column)
- patch 32 and patch 33 — also adjacent horizontally, one row below

But patch 5 and patch 37 have flat-index distance 32 yet are directly above each other — the closest vertical neighbours. A 1D RoPE treats flat-index-1 pairs as "near" and flat-index-32 pairs as "far", making vertical neighbours appear distant even though they are spatially close. Two patches can be far apart in flat-index terms but directly adjacent in the image.

2D RoPE encodes x and y independently, so the attention dot product is sensitive to both horizontal and vertical proximity simultaneously. A patch directly above another (distance 32 in flat index, distance 1 in y) correctly reads as spatially close in the y-frequency dimensions even though 1D RoPE would have treated it as far away. This is essential for any attention pattern that should respect spatial locality — detecting edges, object parts, textures.

---

#### Why pre-computed at init (not per forward pass)

Text RoPE recomputes frequencies every forward pass because `position_ids` vary — different sequences have different lengths, and during generation positions shift as tokens are added to the KV cache.

For vision, the image resolution is fixed by the config (`image_size = 448`, `patch_size = 14` → always a 32×32 grid). The grid coordinates never change between inputs. Recomputing 1025 patch positions and their 24 frequency values on every forward pass would be pure waste. Storing the result as a buffer at init costs a small amount of memory in exchange for eliminating the computation entirely at inference time.

---

#### Why CLS gets zero frequency (identity rotation)

CLS is a global aggregation token — it collects information from all patches via attention but does not correspond to any region of the image. Assigning it a real spatial coordinate (e.g. x=0, y=0) would introduce a false spatial bias: CLS would appear "near" the top-left patches and "far" from the bottom-right patches in the rotated Q/K dot products, even though CLS should be equidistant from every patch.

Zero frequency means `cos(0) = 1, sin(0) = 0` — the complex number `1 + 0j` — which is the identity rotation. Multiplying any vector by the identity leaves it unchanged. CLS gets no positional bias from RoPE at all, so attention between CLS and any patch is unaffected by their "relative position". This lets CLS attend uniformly to every patch based purely on content, not on a fictitious spatial distance.

---

#### Why Llama4 text RoPE uses `torch.polar` instead of `rotate_half`

Llama3's `rotate_half` is real-valued: it computes cos and sin tables at full `head_dim` (by duplicating frequencies with `cat(freqs, freqs)`), then applies two elementwise multiplies and an add. It works, but the duplication step (`cat`) and the `rotate_half` rearrangement are both indirect — they exist only to simulate complex multiplication using real arithmetic.

Llama4 uses `torch.polar` + `view_as_complex` to perform the same rotation as a direct complex multiply. The reasons:

1. **No duplication needed** — `freqs_cis` stays at `head_dim/2` (complex), not `head_dim` (real). One complex multiply replaces two real multiplies plus an add, with no intermediate `cat`.
2. **Cleaner pairing convention** — Llama3 uses a split-half layout (dim `i` pairs with dim `i + head_dim/2`), which is a non-obvious layout that `rotate_half` exists to handle. Llama4 uses consecutive pairs (dim 0 pairs with dim 1) — the natural layout for `view_as_complex`, matching how complex numbers are stored in memory.
3. **Mathematical equivalence** — the complex multiply `(a + bi)(cosθ + i·sinθ) = (a·cosθ − b·sinθ) + i·(a·sinθ + b·cosθ)` is exactly the 2D rotation formula. No new math; just expressing it more directly.

The tradeoff: consecutive-pair layout means Llama3 and Llama4 weight tensors are **not interchangeable** — if you load Llama3 Q/K weights into Llama4 without reordering the dimensions, the pairing is wrong and the rotations are corrupted. This is why model-loading code for Llama4 typically includes a permutation step for the attention weight matrices.

---

#### Why vision RoPE uses `view_as_complex` but text RoPE uses `torch.polar`

Both arrive at the same complex `e^(iθ)` representation, but the construction path differs because of an intermediate step unique to vision RoPE: the `masked_fill` that zeros CLS frequencies.

Vision RoPE must zero out frequencies **before** converting to complex, because the masking operates on the angle values (real floats). It manually computes `cos(freqs)` and `sin(freqs)` — after masking — then stacks and calls `view_as_complex`.

Text RoPE has no such masking step (every token has a real position). It can go directly from angles to complex in one call: `torch.polar(ones, freqs)`. `torch.polar` is slightly more concise but requires the magnitude and angle to already be ready in real form — the vision RoPE's need to interleave masking between the angle computation and the complex conversion is what forces the manual `stack` + `view_as_complex` path instead.

---

## 4. Vision Encoder Blocks — `Llama4VisionEncoderLayer`

Unlike the text decoder's `Llama4TextDecoderLayer`, the vision encoder layers use:
- **`nn.LayerNorm`** (not `LlamaRMSNorm`) — with learnable γ/β, the standard vision transformer choice
- **Biased QKV projections** — `q_proj`, `k_proj`, `v_proj`, `o_proj` all use `bias=True` in `Llama4VisionAttention`
- **No causal mask** — `is_causal=False` — vision attention is bidirectional (each patch sees all other patches)
- **No KV cache** — `past_key_values` is accepted in the signature but not used in vision attention; images are always processed as a full 2D grid
- **GQA disabled** — `num_key_value_groups = 1` (MHA, not GQA); every Q head has its own K/V head
- **Full GELU MLP** (not SwiGLU) — `Llama4VisionMLP` uses `fc1 → GELU → fc2`, no gate branch

Layer norm placement matches standard ViT: pre-norm (normalise before attention and MLP, not after).

---

## 5. Pixel Shuffle — Token Reduction

After the 34-layer vision encoder, the 1024 patch tokens are still 1024 — too many to merge into a text sequence efficiently. Pixel shuffle spatially downsamples them before the final projection:

```python
def pixel_shuffle(input_tensor, shuffle_ratio=0.5):
    batch_size, num_patches, channels = input_tensor.shape  # [B, 1024, C]
    patch_size = int(math.sqrt(num_patches))                # 32

    # Reshape to 2D grid
    input_tensor = input_tensor.view(batch_size, patch_size, patch_size, -1)  # [B, 32, 32, C]

    # Rearrange: fold spatial dimensions into channel dimension
    reshaped = input_tensor.view(batch_size, patch_size, int(patch_size * ratio), int(channels / ratio))
    reshaped = reshaped.permute(0, 2, 1, 3).contiguous()
    reshaped = reshaped.view(batch_size, int(patch_size * ratio), int(patch_size * ratio), int(channels / ratio**2))
    reshaped = reshaped.permute(0, 2, 1, 3).contiguous()

    return reshaped.view(batch_size, -1, reshaped.shape[-1])  # [B, reduced_patches, wider_channels]
```

With `pixel_shuffle_ratio = 0.5`:
- Input: `[B, 1024, C]` = `[B, 32×32, C]`
- Output: `[B, 256, 4C]` = `[B, 16×16, 4C]`

Token count drops from 1024 → 256 (4× reduction), while channel width increases 4× to preserve information. The `Llama4VisionPixelShuffleMLP` then projects this back:
```python
class Llama4VisionPixelShuffleMLP(nn.Module):
    def forward(self, encoded_patches):
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)   # Llama4VisionMLP2: fc1→GELU→fc2
```

`Llama4VisionMLP2` projects `4C → projector_input_dim (4096) → projector_output_dim (4096)`, and the whole vision model outputs `[B, 256, vision_output_dim=7680]`. (Note: `vision_output_dim = 7680` is the output of the combined transformer encoder + adapter, larger than the raw hidden size of 768.)

---

## 6. Multimodal Projector — `Llama4MultiModalProjector`

The vision features need to be mapped into the text model's embedding space. This is done by a single linear layer:

```python
class Llama4MultiModalProjector(nn.Module):
    def __init__(self, config):
        self.linear_1 = nn.Linear(
            config.vision_config.vision_output_dim,  # 7680
            config.text_config.hidden_size,          # 5120
            bias=False,
        )
```

Simple — one weight matrix, no activation, no multi-layer. The simplicity is intentional: the ViT encoder already extracts rich features via 34 layers of attention; the projector's only job is to adapt the output dimensionality.

---

## 7. Merging Vision Features into Text

In `Llama4ForConditionalGeneration.forward`:

```python
if pixel_values is not None:
    # 1. Run vision encoder
    image_features = self.get_image_features(pixel_values, ...).last_hidden_state
    # → [B*tiles, 256, 7680]

    # 2. Project to text hidden size
    vision_flat = image_features.view(-1, image_features.size(-1))   # [B*tiles*256, 7680]
    projected = self.multi_modal_projector(vision_flat)               # [B*tiles*256, 5120]

    # 3. Find placeholder positions in text embeddings
    special_image_mask = (input_ids == self.config.image_token_index)
    # → boolean [B, S], True where image_token_index appears

    # 4. Overwrite placeholders with image features
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask.unsqueeze(-1), projected)
```

`masked_scatter` writes the projected image tokens sequentially into the positions where `image_token_index` appears in the text sequence. The text/image token count must match exactly — the processor ensures the right number of `image_token_index` placeholders are inserted for the number of image patch tokens produced.

After this merge, the language model processes the combined sequence with no distinction between text and image tokens — they're all just `[B, S, 5120]` hidden states.

---

## Vision Config Key Values (Scout 17B-16E)

| Field | Value | Meaning |
|---|---|---|
| `hidden_size` | 768 | ViT hidden dimension |
| `num_hidden_layers` | 34 | Encoder depth |
| `num_attention_heads` | 16 | Attention heads |
| `intermediate_size` | 5632 | MLP intermediate size |
| `image_size` | 448 | Input image resolution |
| `patch_size` | 14 | Patch size in pixels |
| `vision_output_dim` | 7680 | Final output dim (after pixel shuffle adapter) |
| `pixel_shuffle_ratio` | 0.5 | 4× token reduction |
| `projector_input_dim` | 4096 | Adapter intermediate dim |
| `projector_output_dim` | 4096 | Adapter output dim (same) |
