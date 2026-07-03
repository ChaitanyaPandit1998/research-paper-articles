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

**Key differences from text RoPE:**
- **2D, not 1D** — `frequencies_x` (column index) and `frequencies_y` (row index) are encoded separately into the frequency tensor, then concatenated. Each patch gets a position that encodes both where it is horizontally and vertically.
- **Pre-computed at init, not per forward call** — the frequency table is stored as a non-persistent buffer (`register_buffer("freqs_ci", ...)`). It never changes across calls (unlike text RoPE, which recomputes for different `position_ids` each forward pass).
- **CLS token gets zero frequency** — `img_idx[-1, -1] = -2` marks the CLS slot; `masked_fill(img_idx < 0, 0)` zeros out its frequencies. Zero frequency means `cos(0) = 1, sin(0) = 0` → identity rotation → no position encoding for CLS.

The frequency table has shape `[1025, 1025, freq_dim//2]` because the vision encoder performs full bidirectional attention (not causal), and the position encoding is indexed as `freqs_ci[query_position, key_position, :]` rather than just `freqs_ci[position, :]`. This is 2D cross-position encoding, not per-position encoding.

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
