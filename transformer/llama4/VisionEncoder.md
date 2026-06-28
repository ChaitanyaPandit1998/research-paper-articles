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

**`torch.nn.Unfold`** extracts sliding-window patches from an image, outputting all pixel values within each patch as a flat vector. For a 448×448 image with patch size 14:
- `num_patches = (448/14)² = 32² = 1024`
- Each patch vector has length `3 * 14 * 14 = 588`
- Output: `[B, 588, 1024]`, then transposed to `[B, 1024, 588]`

This is mathematically equivalent to a stride-matching `Conv2d` (which PyTorch itself implements via unfold internally), but the explicit unfold makes the patch extraction step visible and separable from the projection. The projection linear `588 → 768` is the same as a `Conv2d`'s weight matrix would be.

---

## 2. CLS Token and Positional Embeddings

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

Note: the CLS token is **appended at the end**, not prepended. This is different from CLIP/ViT convention of prepending `[CLS]`. The CLS position is tracked in the vision RoPE tables (see below) using a special ID `-2`.

After the encoder, the CLS token is stripped before further processing:
```python
hidden_state = hidden_state[:, :-1, :]   # remove CLS, keep the 1024 patch tokens
```

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
