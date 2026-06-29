# Weight Tying — embed_tokens vs lm_head

Llama sets `tie_word_embeddings = False`, meaning the input embedding table and the output LM head matrix are **separate, independently learned weights**. This is different from many smaller models (GPT-2, Gemma, T5) that share them. Understanding why the choice exists — and what each option costs — clarifies a significant design decision.

---

## What Weight Tying Is

The model has two large matrices that both map between `vocab_size` and `hidden_size`:

```python
embed_tokens = nn.Embedding(vocab_size, hidden_size)    # input:  token_id → hidden vector
lm_head      = nn.Linear(hidden_size, vocab_size, bias=False)  # output: hidden vector → logit per token
```

`embed_tokens.weight` has shape `[vocab_size, hidden_size]`.
`lm_head.weight` has shape `[vocab_size, hidden_size]` (Linear stores it transposed, but same data).

**Tied:** `lm_head.weight = embed_tokens.weight` — one tensor, shared in both directions. Updating one updates both.

**Untied (Llama):** two independent tensors. The model has `2 * vocab_size * hidden_size` parameters for these matrices instead of one copy.

---

## The Argument For Tying

**Parameter efficiency.** For Llama3 8B with `vocab_size=128256`, `hidden_size=4096`:

```
one copy:   128256 × 4096 = 525M parameters ≈ 1GB in bfloat16
tied saves: another 525M parameters / 1GB
```

For smaller models where 1GB is a meaningful fraction of total size (GPT-2 117M, for instance, has ~117M total parameters — the embedding table alone is as large as the whole model), tying is clearly worthwhile.

**Theoretical symmetry.** The hypothesis: a token's input embedding (how it looks when it arrives) and its output logit direction (how you'd point at it from a hidden state) should be related. Tying enforces this — the same vector is both "what this token means as input" and "what direction to point to predict this token as output."

---

## The Argument Against Tying (Why Llama Doesn't)

**At large scale, the saving is small relative to total model size.**

Llama3 8B has ~8 billion parameters total. 525M for an untied embedding is ~6.5% of the model. The 32-layer decoder stack's attention and MLP weights dwarf the embedding table. The parameter saving is meaningful for a 1B model; it's minor for an 8B+ model.

**Input and output embeddings serve different roles.**

When `embed_tokens` maps token 42 to a hidden vector, it needs to encode everything useful about token 42 as context for understanding the sequence — syntax, semantics, subword structure.

When `lm_head` predicts which token to output, it needs to distinguish token 42 from token 43 in a way that maximises cross-entropy loss — a different optimisation objective. The optimal geometry for "represent this token in context" may not be the optimal geometry for "score this token as the next prediction."

Untying lets each matrix specialise independently. Empirically, larger models tend to benefit more from untied embeddings because they have the capacity to learn genuinely different representations for input vs output.

**Tied weights create training interference.**

Every gradient update computed for `lm_head` also directly modifies `embed_tokens`, and vice versa. At 128K+ vocabulary size, most tokens are rare — gradients for rare output predictions inadvertently perturb the input embeddings of those rare tokens in ways that may not improve their input representation quality. Separate matrices isolate these update streams.

---

## The `tie_word_embeddings` Field

In `LlamaConfig`:
```python
tie_word_embeddings: bool = False
```

In `LlamaForCausalLM`:
```python
self._tied_weights_keys = []   # empty — nothing is tied
```

If `tie_word_embeddings=True` were set, the framework would add `"lm_head.weight"` to `_tied_weights_keys`, which signals the weight-saving/loading infrastructure to share the tensor rather than storing it twice.

For comparison, Gemma sets `tie_word_embeddings = True` — its LM head and embedding table are the same matrix. Gemma is designed to be a smaller, more efficient model family where this saving matters more.

---

## Memory Implications at Inference

With `vocab_size=128256`, `hidden_size=4096`, bfloat16:
- Untied (Llama): embed_tokens (1GB) + lm_head (1GB) = 2GB just for these two matrices
- Tied: 1GB shared

At deployment scale this matters, but Llama's inference optimisation answer is `logits_to_keep=1` (see `LogitsToKeep.md`) — only computing the LM head projection for the last token — rather than tying weights. The LM head weights still need to be loaded from memory each step regardless of tying, so the bandwidth cost is the same either way.
