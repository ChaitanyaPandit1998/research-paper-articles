# Llama Model — Pseudocode Summary

Quick-reference pseudocode for every module covered in the session. Each block shows `__init__` state and `forward` logic in plain steps, without PyTorch syntax noise.

---

## `LlamaRMSNorm`
*Normalizes hidden states to a consistent magnitude before each sub-layer.*

```
__init__(hidden_size, eps):
  1. Call super().__init__() to register this as a proper nn.Module
  2. Initialize self.weight as nn.Parameter of shape [hidden_size], filled with ones
  3. Store eps as self.variance_epsilon

forward(hidden_states):
  1. Save original dtype (e.g. bfloat16)
  2. Upcast hidden_states to float32 for numerical stability
  3. Compute variance = mean of squared values along last axis (keepdim=True)
  4. Normalize: hidden_states = hidden_states * (1 / sqrt(variance + eps))
  5. Cast back to original dtype
  6. Apply learnable scale: return self.weight * hidden_states
```

---

## `LlamaRotaryEmbedding`
*Computes cos/sin rotation tables for RoPE positional encoding.*

```
__init__(config):
  1. Call rope_init_fn (default: compute_default_rope_parameters) to get inv_freq
  2. Register inv_freq as a non-persistent buffer (not saved to checkpoint)
  3. Register original_inv_freq as a non-persistent buffer (clone of inv_freq, for reset)

compute_default_rope_parameters(config):
  1. Compute base theta (e.g. 10000.0) from config
  2. Compute inv_freq = 1 / (theta ^ (2i / dim)) for i in 0..dim/2
     → This is the frequency for each pair of dimensions
  3. Return inv_freq

forward(x, position_ids):
  1. Expand inv_freq:    [1, dim/2, 1]      → [batch, dim/2, 1]   (broadcast over batch)
  2. Expand position_ids: [batch, seq_len]  → [batch, 1, seq_len] (broadcast over freq dims)
  3. Compute outer product: freqs = inv_freq @ position_ids → [batch, dim/2, seq_len]
  4. Transpose freqs to [batch, seq_len, dim/2]
  5. Concatenate freqs with itself along last axis → [batch, seq_len, dim]  (full angles)
  6. Return cos(freqs), sin(freqs)   (the rotation tables)
```

---

## `rotate_half`
*Rotates a tensor by 90° using the split-half layout — building block of RoPE.*

```
rotate_half(x):
  1. Split x along last axis into two halves: x1 = first half, x2 = second half
  2. Return concatenation of [-x2, x1]   (negate second half, swap order)
```

---

## `apply_rotary_pos_emb`
*Applies RoPE rotation to query and key tensors using the cos/sin tables.*

```
apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim):
  1. Unsqueeze cos and sin along unsqueeze_dim to align with q/k head dimension
  2. Apply rotation to q:  q_embed = (q * cos) + (rotate_half(q) * sin)
  3. Apply rotation to k:  k_embed = (k * cos) + (rotate_half(k) * sin)
  4. Return q_embed, k_embed
```

---

## `repeat_kv`
*Expands KV heads to match the number of query heads (Grouped Query Attention).*

```
repeat_kv(hidden_states, n_rep):
  1. If n_rep == 1, return hidden_states unchanged  (standard MHA, no grouping)
  2. Insert a new size-1 axis after the heads dimension
     → shape: [batch, num_kv_heads, 1, seq_len, head_dim]
  3. Expand that axis to n_rep (zero-copy broadcast)
     → shape: [batch, num_kv_heads, n_rep, seq_len, head_dim]
  4. Reshape to merge kv_heads and n_rep into one axis
     → shape: [batch, num_kv_heads * n_rep, seq_len, head_dim]
  5. Return result   (each KV head is now repeated n_rep times to match query heads)
```

---

## `eager_attention_forward`
*The default pure-PyTorch scaled dot-product attention computation.*

```
eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout):
  1. Expand K/V heads to match query heads via repeat_kv(key/value, n_rep)
  2. Compute raw scores: attn_weights = (query @ key.T) * scaling
     → shape: [batch, heads, seq_len_q, seq_len_k]
  3. Add attention_mask (large negatives at masked positions, e.g. future tokens)
  4. Softmax over last axis (in float32 for stability), cast back to model dtype
  5. Apply dropout to attn_weights (training only)
  6. Weighted sum of values: attn_output = attn_weights @ value_states
     → shape: [batch, heads, seq_len_q, head_dim]
  7. Transpose back: attn_output.transpose(1, 2).contiguous()
     → shape: [batch, seq_len_q, heads, head_dim]
  8. Return attn_output, attn_weights
```

---

## `LlamaAttention`
*Multi-head attention with GQA, KV cache, and pluggable backends.*

```
forward(hidden_states, position_ids, past_key_values, attention_mask, ...):
  1. Project hidden_states into query, key, value via linear layers (q_proj, k_proj, v_proj)
  2. Reshape each into [batch, num_heads, seq_len, head_dim]
  3. Compute cos/sin tables via LlamaRotaryEmbedding.forward(x, position_ids)
  4. Rotate query and key via apply_rotary_pos_emb(q, k, cos, sin)
  5. If past_key_values exists:
       Append current key/value to cache → get full K/V history back
  6. Pick attention backend:
       attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
         config._attn_implementation, default=eager_attention_forward)
  7. Call attention_interface(query, key, value, attention_mask, scaling, ...)
  8. Merge heads: attn_output.reshape(*input_shape, -1).contiguous()
     → [batch, seq_len, num_heads * head_dim]  ==  [batch, seq_len, hidden_size]
  9. Apply output projection (o_proj linear layer)
  10. Return attn_output, attention_weights
```

---

## `LlamaDecoderLayer`
*Single transformer block: norm → attention → residual → norm → MLP → residual.*

```
forward(hidden_states, position_ids, past_key_values, attention_mask, ...):

  # Attention sub-block
  1. residual = hidden_states                             (save input)
  2. hidden_states = input_layernorm(hidden_states)       (RMSNorm before attention)
  3. hidden_states = self_attn(hidden_states, ...)        (full LlamaAttention forward)
  4. hidden_states = residual + hidden_states             (residual connection)

  # MLP sub-block
  5. residual = hidden_states                             (save input again)
  6. hidden_states = post_attention_layernorm(hidden_states) (RMSNorm before MLP)
  7. hidden_states = mlp(hidden_states)                   (feed-forward block)
  8. hidden_states = residual + hidden_states             (second residual connection)

  9. Return hidden_states
```

---

## `LlamaModel`
*Full stack of decoder layers with embedding, KV cache, and position management.*

```
forward(input_ids, position_ids, past_key_values, attention_mask, ...):

  # Embedding
  1. Embed input_ids → hidden_states via embed_tokens

  # Cache setup
  2. If use_cache=True and no past_key_values provided:
       past_key_values = DynamicCache()   (empty container, grows token-by-token)

  # Position IDs
  3. If position_ids not provided:
       past_seen_tokens = past_key_values.get_seq_length()  (0 if no cache)
       position_ids = torch.arange(seq_len) + past_seen_tokens
       position_ids = position_ids.unsqueeze(0)             (add batch dim)

  # Decoder stack
  4. For each decoder_layer in self.layers:
       hidden_states = decoder_layer(hidden_states, position_ids,
                                     past_key_values, attention_mask, ...)

  # Final norm
  5. hidden_states = norm(hidden_states)   (LlamaRMSNorm over final hidden states)

  6. Return hidden_states, past_key_values
```

---

## Full forward pass call chain

```
LlamaModel.forward
  └─ LlamaDecoderLayer.forward  (× num_layers)
       ├─ LlamaRMSNorm           (input_layernorm)
       ├─ LlamaAttention.forward
       │    ├─ LlamaRotaryEmbedding.forward  → cos/sin tables
       │    ├─ apply_rotary_pos_emb          → rotated q, k
       │    │    └─ rotate_half              (called inside)
       │    ├─ past_key_values.update        → full K/V history
       │    └─ eager_attention_forward
       │         └─ repeat_kv               → expanded K/V for GQA
       ├─ residual + attn_output
       ├─ LlamaRMSNorm           (post_attention_layernorm)
       ├─ MLP.forward
       └─ residual + mlp_output
```
