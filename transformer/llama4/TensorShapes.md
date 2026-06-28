# Llama4 Model — Tensor Shape Tracking

Concrete example: **Llama4 Scout 17B-16E** (multimodal, MoE)

### Text config
- `B` = batch size = 2
- `S` = sequence length = 512 (includes image placeholder tokens)
- `H` = hidden size = 5120
- `n_heads` = 40 (Q heads)
- `n_kv_heads` = 8 (K/V heads)
- `head_dim` = 128
- `n_kv_groups` = 5 (40 / 8)
- `intermediate_size` = 8192 (MoE expert FFN)
- `intermediate_size_mlp` = 16384 (dense layer FFN)
- `vocab_size` = 202048
- `num_layers` = 48
- `num_local_experts` = 16
- `attention_chunk_size` = 8192

### Vision config
- `V_H` = vision hidden size = 768
- `n_patches` = 1024 (32×32 for 448×448 image, 14×14 patch)
- `n_patches_after_shuffle` = 256 (pixel_shuffle_ratio=0.5 → 4× reduction)
- `vision_output_dim` = 7680
- `n_vision_layers` = 34
- `n_vision_heads` = 16

---

## Vision Path

### Stage V1 — Input image

```
pixel_values
  [B * num_tiles, C, H_img, W_img]  =  [2, 3, 448, 448]
  dtype: float32 / bfloat16
```

### Stage V2 — Patch Embedding (`Llama4UnfoldConvolution`)

```
unfold(pixel_values)
  [2, C*k*k, num_patches]  =  [2, 588, 1024]   (588 = 3*14*14)

permute(0, 2, 1)
  [2, 1024, 588]

linear(588 → 768)
  [2, 1024, 768]   ← patch embeddings
```

### Stage V3 — Append CLS Token + Positional Embeddings

```
class_embedding.expand → [2, 1, 768]
cat([patches, cls], dim=1)
  [2, 1025, 768]   (1024 patches + 1 CLS)

+ positional_embedding_vlm [1025, 768]  (broadcast)
  [2, 1025, 768]

layernorm_pre
  [2, 1025, 768]   ← same shape
```

### Stage V4 — Vision Encoder (×34 layers)

Each `Llama4VisionEncoderLayer` (full bidirectional attention, LayerNorm, GELU MLP):

```
input                        [2, 1025, 768]
layernorm                    [2, 1025, 768]

q_proj, k_proj, v_proj (bias=True, MHA: num_kv_groups=1):
  q [2, 16, 1025, 48]    (768/16=48 per head)
  k [2, 16, 1025, 48]
  v [2, 16, 1025, 48]

apply vision RoPE (complex, 2D x+y):
  freqs_ci [1025, 1025, 24] complex
  q, k rotated → same shape

attn_weights = q @ k.T     [2, 16, 1025, 1025]
  no causal mask (is_causal=False)
softmax                    [2, 16, 1025, 1025]
attn_output = weights @ v  [2, 16, 1025, 48]

transpose + merge heads    [2, 1025, 768]
o_proj                     [2, 1025, 768]
residual add               [2, 1025, 768]

layernorm
fc1 (768 → 5632)           [2, 1025, 5632]
GELU
fc2 (5632 → 768)           [2, 1025, 768]
residual add               [2, 1025, 768]
```

### Stage V5 — Post-norm + Remove CLS

```
layernorm_post               [2, 1025, 768]
remove CLS token ([:, :-1])  [2, 1024, 768]
```

### Stage V6 — Pixel Shuffle + Adapter MLP

```
pixel_shuffle(ratio=0.5):
  input  [2, 1024, 768]  = [2, 32×32, 768]
  output [2,  256, 3072] = [2, 16×16, 3072]   (4× fewer tokens, 4× wider channels)

Llama4VisionMLP2:
  fc1 (3072 → 4096)        [2, 256, 4096]
  GELU
  dropout
  fc2 (4096 → 4096)        [2, 256, 4096]

vision_output_dim = 7680 is the configured output — here the adapter outputs 4096 channels
(vision_output_dim is used as the projector input, shape may vary by checkpoint)

image_features               [2, 256, 7680]   ← final vision encoder output
```

### Stage V7 — Multimodal Projector

```
image_features.view(-1, 7680)    [2*256, 7680]  =  [512, 7680]
linear_1 (7680 → 5120)           [512, 5120]
projected_vision_flat            [512, 5120]     ← ready to merge into text sequence
```

---

## Text Path (with image tokens merged)

### Stage T1 — Input IDs

```
input_ids
  [B, S]  =  [2, 512]   (includes image_token_index=200092 placeholders for image patches)
  dtype: int64
```

### Stage T2 — Token Embedding

```
embed_tokens: nn.Embedding(vocab_size=202048, hidden_size=5120)
inputs_embeds  [2, 512, 5120]

Image token positions are overwritten with projected vision features:
  special_image_mask [2, 512]  ← True at image_token_index positions
  inputs_embeds.masked_scatter → replaces 256 image placeholders per image with projected_vision_flat
  inputs_embeds  [2, 512, 5120]   ← text + image embeddings merged
```

### Stage T3 — Position IDs and RoPE (computed ONCE, shared by all 48 layers)

```
position_ids  [2, 512]   (0…511)

Llama4TextRotaryEmbedding.forward:
  inv_freq  [head_dim/2]  =  [64]
  freqs (complex)  [2, 512, 64]   using torch.polar
  freqs_cis  [2, 512, 64]  complex  ← shared across all layers
```

### Stage T4 — Dual Causal Masks (computed ONCE)

```
causal_mask_mapping = {
  "full_attention":    [2, 1, 512, 512]   standard upper-triangular causal mask
  "chunked_attention": [2, 1, 512, 512]   block-diagonal, blocks of attention_chunk_size=8192
                                          (at S=512 < chunk_size, equivalent to full attention here)
}
```

---

## Stage T5 — Decoder Layer (×48)

*Shapes shown for a MoE layer (most layers). Dense layers replace the MoE FFN with a 16384-wide MLP.*

### T5a. Pre-Attention RMSNorm

```
residual = hidden_states  [2, 512, 5120]
Llama4TextRMSNorm(5120)   [2, 512, 5120]   ← same shape, no learnable bias
```

### T5b. QKV Projections

```
q_proj (5120 → 40*128=5120)  [2, 512, 5120]
k_proj (5120 →  8*128=1024)  [2, 512, 1024]
v_proj (5120 →  8*128=1024)  [2, 512, 1024]

Reshape to [B, seq, heads, head_dim] then transpose to [B, heads, seq, head_dim]:
  query_states  [2, 40,  512, 128]
  key_states    [2,  8,  512, 128]
  value_states  [2,  8,  512, 128]   ← transposed already in forward
```

### T5c. Apply RoPE (RoPE layers only, `use_rope=1`)

```
freqs_cis [2, 512, 64] complex  →  unsqueeze  →  [2, 512, 1, 64]  (broadcasts over heads)

apply_rotary_emb (complex multiply):
  view_as_complex(q.reshape(..., -1, 2))  →  [2, 40, 512, 64] complex
  multiply by freqs_cis                   →  [2, 40, 512, 64] complex
  view_as_real.flatten(3)                 →  [2, 40, 512, 128]

query_states  [2, 40, 512, 128]   ← rotated
key_states    [2,  8, 512, 128]   ← rotated
```

### T5c'. Attention Temperature Tuning (NoPE layers only, `use_rope=0`)

```
positions = [0, 1, ..., 511] + past_seen_tokens
attn_scales = log1p(floor((positions+1)/8192)) * 0.1 + 1.0   [512]   all 1.0 at S<8192
attn_scales reshaped → [1, 512, 1, 1]
query_states = query_states * attn_scales   [2, 40, 512, 128]   ← no change for S<8192
```

### T5d. QK Norm (RoPE layers only)

```
Llama4TextL2Norm applied independently to each Q and K head vector:
  query_states  [2, 40, 512, 128]   ← L2 normalised
  key_states    [2,  8, 512, 128]   ← L2 normalised
```

### T5e. KV Cache Update

```
past_key_values.update(key_states, value_states, layer_idx)

Generation step T (S=1 new token):
  new key_states  [2,  8,   1, 128]
  appended cache  [2,  8,   T, 128]   (T = prompt_length + generated_tokens)
  query_states stays [2, 40, 1, 128]
```

### T5f. GQA — repeat_kv

```
n_kv_groups = 40 / 8 = 5

repeat_kv(key_states, n_rep=5):
  [2,  8, 512, 128]  →  [2, 40, 512, 128]

key_states    [2, 40, 512, 128]
value_states  [2, 40, 512, 128]
```

### T5g. Attention

```
scaling = 128**-0.5 ≈ 0.0884

attn_weights = query_states @ key_states.T * scaling
  [2, 40, 512, 128] @ [2, 40, 128, 512]  →  [2, 40, 512, 512]

+ causal_mask [2, 1, 512, 512]  (broadcast over 40 heads)

softmax (NO float32 upcast — Llama4 difference from Llama3)
  [2, 40, 512, 512]

attn_output = attn_weights @ value_states
  [2, 40, 512, 512] @ [2, 40, 512, 128]  →  [2, 40, 512, 128]

transpose(1,2) + merge heads:
  [2, 512, 40, 128]  →  reshape  →  [2, 512, 5120]

o_proj (5120 → 5120)   [2, 512, 5120]
residual add           [2, 512, 5120]
```

### T5h. Pre-FFN RMSNorm

```
residual = hidden_states  [2, 512, 5120]
Llama4TextRMSNorm         [2, 512, 5120]
```

### T5i. MoE FFN (MoE layers)

```
hidden_states.reshape(-1, 5120)   [1024, 5120]   (B*S flattened)

Router (Linear: 5120 → 16):
  router_logits  [1024, 16]
  topk(k=1) → router_scores [1024, 16] (15 entries are -inf → sigmoid ≈ 0)
  router_scores after sigmoid [1024, 16]  (one non-zero per row)

Dense dispatch (repeat all tokens × 16, then zero out non-selected):
  routed_in = hidden.repeat(16, 1)               [1024*16, 5120]  =  [16384, 5120]
  routed_in *= router_scores.T.reshape(-1, 1)    [16384, 5120]

Llama4TextExperts (batched bmm):
  routed_in.view(16, -1, 5120)                   [16, 1024, 5120]
  gate_up = bmm(routed_in, gate_up_proj)         [16, 1024, 16384]
  gate, up = chunk(2) → each [16, 1024, 8192]
  SwiGLU: up * silu(gate)                        [16, 1024, 8192]
  bmm(swiglu_out, down_proj)                     [16, 1024, 5120]
  view(-1, 5120)                                 [16384, 5120]

routed_out.reshape(16, -1, 5120).sum(dim=0)      [1024, 5120]

shared_expert (always active):
  gate_proj (5120 → 8192)   [1024, 8192]
  up_proj   (5120 → 8192)   [1024, 8192]
  SwiGLU: up * silu(gate)   [1024, 8192]
  down_proj (8192 → 5120)   [1024, 5120]

moe_output = shared_expert + routed_experts  [1024, 5120]
moe_output.view(2, 512, 5120)                [2, 512, 5120]

residual add   [2, 512, 5120]
→ next layer (back to T5a)
```

### T5i'. Dense FFN (dense layers)

```
gate_proj (5120 → 16384)   [2, 512, 16384]
up_proj   (5120 → 16384)   [2, 512, 16384]
SwiGLU: up * silu(gate)    [2, 512, 16384]
down_proj (16384 → 5120)   [2, 512, 5120]
residual add               [2, 512, 5120]
```

---

## Stage T6 — Final RMSNorm

```
Llama4TextRMSNorm(5120)
  [2, 512, 5120]  →  [2, 512, 5120]
```

## Stage T7 — LM Head

```
lm_head: Linear(5120 → 202048, bias=False)
  full:       [2, 512, 5120]  →  [2, 512, 202048]
  generation: [2,   1, 5120]  →  [2,   1, 202048]
```

## Stage T8 — Loss (training only)

```
labels [2, 512]  (input_ids shifted by 1)

cross_entropy(
  logits.view(-1, 202048)   [1024, 202048]
  labels.view(-1)            [1024]
)
→ scalar CE loss + router_aux_loss_coef * load_balancing_loss
```

---

## Shape Summary Table

| Tensor | Shape | Notes |
|---|---|---|
| `pixel_values` | [2, 3, 448, 448] | input image |
| `patch_embeds` | [2, 1024, 768] | after unfold + linear |
| `vision_hidden` (encoder) | [2, 1025, 768] | with CLS, across 34 layers |
| `vision_attn_weights` | [2, 16, 1025, 1025] | full bidirectional |
| `post_encoder` | [2, 1024, 768] | CLS removed |
| `after_pixel_shuffle` | [2, 256, 3072] | 4× fewer tokens |
| `image_features` | [2, 256, 7680] | vision encoder output |
| `projected_vision` | [512, 5120] | after multimodal projector |
| `input_ids` | [2, 512] | int64 |
| `inputs_embeds` (merged) | [2, 512, 5120] | text + image |
| `freqs_cis` (text RoPE) | [2, 512, 64] complex | computed once per forward |
| `full_attention mask` | [2, 1, 512, 512] | for RoPE layers |
| `chunked_attention mask` | [2, 1, 512, 512] | for NoPE layers |
| `hidden_states` (per layer) | [2, 512, 5120] | invariant through all 48 layers |
| `query_states` | [2, 40, 512, 128] | after RoPE + QK norm |
| `key_states` (pre-GQA) | [2, 8, 512, 128] | 8 KV heads |
| `key_states` (post-GQA) | [2, 40, 512, 128] | after repeat_kv ×5 |
| `attn_weights` | [2, 40, 512, 512] | no fp32 upcast |
| `router_logits` | [1024, 16] | per-token, per-expert |
| `router_scores` | [1024, 16] | after topk + sigmoid |
| `routed_in` (dispatch) | [16384, 5120] | all tokens × 16 experts |
| `expert bmm input` | [16, 1024, 5120] | sorted by expert |
| `gate_up output` | [16, 1024, 16384] | fused gate+up |
| `expert output` | [16, 1024, 5120] | after down_proj |
| `moe_output` | [2, 512, 5120] | shared + sparse |
| `logits` (full) | [2, 512, 202048] | training |
| `logits` (generation) | [2, 1, 202048] | inference |
