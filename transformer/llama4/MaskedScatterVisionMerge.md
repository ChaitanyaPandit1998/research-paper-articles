# `masked_scatter` — How Vision Tokens Replace Text Placeholders

When Llama4 processes an image alongside text, it needs to splice the projected image features into the text embedding sequence at the right positions. This is done in one operation: `masked_scatter`. This file explains the mechanics step by step.

---

## The Setup

Before `Llama4ForConditionalGeneration.forward` runs the language model, it needs to produce a single `inputs_embeds` tensor of shape `[B, S, hidden_size]` that contains both text embeddings and image embeddings in the right order.

The processor (`Llama4Processor`) has already prepared `input_ids` such that every image patch position is marked with a special placeholder token ID:
```
input_ids = [101, 200092, 200092, ..., 200092, 42, 99, ...]
                  ↑─────── image_token_index (200092) ───────↑  ← one per patch token
```

For a single image producing 256 patch tokens after pixel shuffle, there are exactly 256 consecutive `200092` tokens in `input_ids`.

---

## Step 1 — Embed All Tokens (Including Placeholders)

```python
inputs_embeds = self.get_input_embeddings()(input_ids)
# [B, S, 5120]
```

The image placeholder tokens get embedded like any other token — `embed_tokens[200092]` is a real embedding vector. These are throwaway values that will be overwritten, but embedding them is necessary because the operation below needs a correctly-shaped target tensor to write into.

---

## Step 2 — Run the Vision Encoder

```python
image_features = self.get_image_features(pixel_values, ...).last_hidden_state
# [B * num_tiles, 256, 7680]

vision_flat = image_features.view(-1, image_features.size(-1))
# [B * num_tiles * 256, 7680]  →  flat list of all patch feature vectors

projected_vision_flat = self.multi_modal_projector(vision_flat)
# [B * num_tiles * 256, 5120]  →  projected to text hidden size
```

The vision encoder produces one 5120-dimensional vector per patch token, flattened into a single list.

---

## Step 3 — Build the Placeholder Mask

```python
special_image_mask = (input_ids == self.config.image_token_index)
# [B, S]  →  boolean, True at every image_token_index position

special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
# [B, S, 5120]  →  True at every (batch, seq, hidden_dim) position that belongs to an image token
```

The mask starts as a 2D boolean `[B, S]`, then gets expanded along the hidden dimension so it covers all 5120 values of each image placeholder token's embedding vector. Every value at an image token position is `True`; every value at a text token position is `False`.

---

## Step 4 — `masked_scatter`: Write Vision Features Into the Placeholder Slots

```python
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected_vision_flat)
```

`torch.Tensor.masked_scatter(mask, source)` works as follows:
- Iterate through `inputs_embeds` in row-major order
- Wherever `mask` is `True`, replace the value with the next value from `source` (consuming `source` sequentially)
- Wherever `mask` is `False`, leave the value unchanged

Concretely, for a single batch entry with 256 image placeholder positions and a 5120-wide hidden dimension:
- There are `256 * 5120 = 1,310,720` True positions in the mask for this batch entry
- `projected_vision_flat` has exactly `256 * 5120 = 1,310,720` values to write
- `masked_scatter` writes them in order: patch 0's 5120 values into the first placeholder's slot, patch 1's 5120 values into the second placeholder's slot, and so on

Result: `inputs_embeds` now has genuine image patch vectors where the placeholder tokens used to be, and unchanged text embeddings everywhere else.

---

## The Correctness Guarantee

For `masked_scatter` to produce the right result, the number of `True` positions in the mask must exactly equal the number of values in `source`. The code enforces this:

```python
torch_compilable_check(
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
    f"Image features and image tokens do not match: "
    f"tokens: {n_image_tokens}, features: {image_features.shape[0]}"
)
```

If the processor inserted the wrong number of placeholder tokens (e.g. 255 placeholders for 256 patch tokens), this check raises an error before `masked_scatter` runs. The count match is the processor's responsibility — `Llama4Processor` computes exactly how many patch tokens the vision encoder will produce given the image size and pixel shuffle ratio, and inserts exactly that many `image_token_index` placeholders.

---

## Why `masked_scatter` Instead of Indexing

An alternative approach would be explicit index-based assignment:
```python
image_positions = (input_ids == image_token_index).nonzero()
inputs_embeds[image_positions] = projected_vision_flat
```

`masked_scatter` is preferred because:
- It's a single fused PyTorch operation, more `torch.compile`-friendly than a `nonzero()` + index assignment
- `nonzero()` is a dynamic shape operation (the number of True positions isn't known at trace time), which breaks graph compilation
- `masked_scatter` only needs the boolean mask — no dynamic indexing, no shape that depends on image content

The downside: `masked_scatter` writes values in row-major order, so the source tensor must also be ordered to match. The processor ensures this by inserting placeholders in left-to-right sequence and the vision encoder produces patches in the same spatial order.

---

## After the Merge

Once `inputs_embeds` has vision features spliced in, the language model sees a flat `[B, S, 5120]` tensor. It has no explicit signal about which positions are "image" vs "text" — the `boi_token_index` (200080) and `eoi_token_index` (200081) surrounding the image region serve as delimiter tokens that the model learns to interpret during training. The attention mechanism treats image patch embeddings and text embeddings identically from this point forward.
