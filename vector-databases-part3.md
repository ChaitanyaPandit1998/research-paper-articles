# Vector Databases, Part 3: Quantization and Hybrid Search

**Pull quotes:**
- "Quantization is the same trade the whole field keeps making, applied to memory instead of latency: give up some precision per vector, in exchange for fitting orders of magnitude more vectors in the same RAM."
- "Binary quantization turning a 768-dimensional float32 vector into 768 bits sounds like it should destroy the signal. It mostly doesn't, for the same reason a blurry thumbnail still lets you recognize a face — most of the information was redundant to begin with."
- "RRF never looks at the underlying scores, only at where each thing landed on each list."

---

[Part 1](vector-databases.md) covered the anatomy of a dense-vector database. [Part 2](vector-databases-part2.md) covered sparse retrieval — BM25 and SPLADE — as the complement dense embeddings can't structurally replace. This article covers the two remaining pieces that make both of those practical and combinable at production scale: shrinking vectors so they fit in memory, and merging dense and sparse results into a single ranking. It works through:

- why quantization exists, and the memory problem it's solving
- scalar, product, binary, and rotation-based (TurboQuant) quantization, compared
- reciprocal rank fusion, the mechanism that merges dense and sparse results into one ranking
- what none of this actually solves — uneven quantization loss, RRF's blind spots, and doubled maintenance
- a worked example where hybrid search finds something dense-only search misses

---

## Table of contents

1. [Quantization: why compress a vector at all](#1-quantization-why-compress-a-vector-at-all)
2. [Quantization techniques, compared](#2-quantization-techniques-compared)
3. [Hybrid search and reciprocal rank fusion](#3-hybrid-search-and-reciprocal-rank-fusion)
4. [What this doesn't solve](#4-what-this-doesnt-solve)
5. [A worked example: where hybrid search wins](#5-a-worked-example-where-hybrid-search-wins)
6. [Summary table](#6-summary-table)
7. [Key takeaways](#7-key-takeaways)
8. [Further reading](#8-further-reading)

---

## 1. Quantization: why compress a vector at all

- **The memory problem, concretely.** A single 768-dimensional embedding stored as float32 takes 768 × 4 bytes = 3,072 bytes. At 10 million vectors, that's **roughly 30 GB just for the raw vectors** — before accounting for the HNSW graph's own overhead (each vector's neighbor lists, stored per layer), which Part 1 noted is typically kept fully in RAM for query speed. At 100 million or a billion vectors, the numbers stop being a hosting inconvenience and start being a hard infrastructure constraint.
  - *In plain terms:* it's the difference between one person's music collection and a hundred million people's — a slightly-too-big MP3 folder is an inconvenience; a slightly-too-big-per-file collection at planet scale is a server bill someone has to justify.

- **Quantization is the general answer: store a lower-precision approximation of each vector instead of the full-precision original.** The same tradeoff Part 1 made central to ANN search — give up a small, controllable amount of accuracy for a large, predictable gain — reappears here, but the resource being traded for is memory (and, often, speed) rather than latency alone.

- **The mechanism is always some version of "round to fewer possible values, then store less."** A float32 has roughly 4 billion distinguishable values per dimension; quantization schemes instead pick a much smaller set of representative values (256 for one common scheme, all the way down to 2 for the most aggressive) and store which representative value each dimension is closest to, plus, at search time, either compare directly against the compressed representation or use it to shortlist candidates that then get rescored against the original full-precision vectors.

  **Quantization, in plain terms.** Think of describing someone's height. "5 feet 10.375 inches" is maximally precise but overkill for most purposes. "5'10"" is almost as useful and takes a fraction of the words. "Tall" is even shorter and still lets you rule out most of a crowd when you're searching for someone — you just might have to look a little closer once you've narrowed it down to the tall people. Quantization is choosing how much precision you actually need to store, given that you're about to compare millions of these descriptions against each other and memory (or disk) is the thing running out, not accuracy in the abstract.

- **Rescoring is the safety net that makes aggressive quantization usable.** Rather than trusting the compressed vectors' distances outright, many production systems use quantized vectors to cheaply narrow millions of candidates down to a few hundred, then compute exact distances against the original float32 vectors only for that small shortlist — recovering most of the accuracy lost to quantization while still doing the bulk of the work over the compressed, memory-cheap representation. This pattern (often called oversampling and rescoring) shows up explicitly in binary quantization (§2) and is available, in some form, for most quantization schemes.

> **Worth remembering.** Quantization and ANN indexing solve two different problems that happen to compound: HNSW/IVF (Part 1) reduce how many vectors get *compared* against a query; quantization reduces how much *memory* each stored vector costs and, often, how expensive each individual comparison is. A production system at scale typically uses both simultaneously — an ANN index built over quantized vectors — not one instead of the other.

---

## 2. Quantization techniques, compared

- **Scalar quantization** compresses each dimension independently: instead of a 32-bit float, each dimension is stored as an 8-bit integer, mapped linearly onto the observed range of values for that dimension. This is the least aggressive common scheme — **roughly a 4x memory reduction** (32 bits → 8 bits) — and the simplest to reason about, since each dimension is quantized on its own with no interaction between dimensions.
  - *In plain terms:* like rounding every price on a menu to the nearest dollar — $12.99 becomes $13. You lose the cents, but you can still tell a $13 dish from a $30 one just fine.

- **Product quantization (PQ)**, previewed in Part 1 as a component of "IVF-PQ," takes a different approach: split each vector into several sub-vectors (say, a 768-dimension vector split into 96 sub-vectors of 8 dimensions each), and separately learn a small codebook of representative sub-vectors (via something like k-means) for each split. Each original vector is then stored as a sequence of codebook indices — one small integer per sub-vector — rather than as raw numbers at all.
  - This achieves much higher compression than scalar quantization, tunable via codebook size and sub-vector count — 16x is a commonly cited example configuration, and Qdrant's own documentation states product quantization can reach up to 64x compression when memory is prioritized over accuracy — because it exploits correlation *between* dimensions within a sub-vector, not just the range of each dimension independently.
  - The cost is a more expensive index-build step (training the codebooks) and a less direct relationship between the stored code and the original vector, which makes exact-distance rescoring slightly more involved than with scalar quantization.
  - *In plain terms:* like a paint store's color-matching swatches. Instead of describing your wall's exact RGB value, the clerk holds up the closest swatch from a fixed set of a few hundred and says "it's this one" — a short code ("aisle 4, swatch 112") replaces a precise-but-verbose description, and it's close enough for almost every purpose.

- **Binary quantization** is the most aggressive of the widely-deployed schemes: each dimension collapses to a single bit — typically, whether the original value was above or below zero (or the dimension's mean). A 768-dimension float32 vector (3,072 bytes) becomes 768 bits (96 bytes), a **32x memory reduction**, and because bitwise operations (Hamming distance, effectively XOR-and-popcount) are extremely cheap compared to floating-point dot products, search over binary-quantized vectors can run **up to roughly 40x faster** than over the original float32 vectors.
  - This level of compression throws away enough information that oversampling and rescoring (§1) generally **isn't optional** — production use of binary quantization typically retrieves a larger candidate set using the cheap binary comparison, then rescoring against original vectors to recover accuracy, rather than trusting binary distances as the final ranking.
  - It works best when a vector's dimensions carry roughly symmetric, somewhat redundant information — which is generally true of embeddings from well-trained models, less reliably true of arbitrary numeric data.
  - *In plain terms:* like reducing a photo to pure black-and-white silhouettes. You can still recognize a face's rough outline in a stack of a million silhouettes far faster than flipping through a million full-color photos — you just want to pull up the original color photo before making a final call on which face it actually is.

- **TurboQuant**, a more recent rotation-based method from Google Research, takes a different angle from all three schemes above: instead of quantizing the original vector's dimensions directly, it first applies a random rotation to the vector (a transformation that preserves distances but spreads information more evenly across dimensions), then quantizes the rotated result at a chosen bit width.
  - The rotation step matters because it makes the subsequent quantization more accurate — spreading a vector's signal evenly across dimensions gives low-bit quantization less "important" information concentrated in a single dimension to lose.
  - **Reported results position it favorably against the other schemes at matched compression:** its 4-bit variant achieves roughly the same recall as scalar quantization (8-bit) while using half the memory, and its 2-bit and 1-bit variants outperform binary quantization at those more aggressive compression levels (roughly 16x and 32x reduction respectively) on recall.
  - *In plain terms:* like shuffling a deck before dealing it into piles, so no single pile ends up all one suit. If a vector's important signal happened to be concentrated in just a few dimensions, aggressive rounding would wipe it out; the rotation spreads that signal around first, so rounding afterward loses a little bit of everything instead of losing a lot of one important thing.
  - **A maturity caveat worth keeping in mind:** scalar, product, and binary quantization have years of production use behind them across many vector databases. TurboQuant is a recent Google Research technique — it's included here because it's directly comparable and because Qdrant's source article covers it, not because it has the same multi-year adoption track record as the other three. Treat its numbers as reported results worth watching, not yet as a default choice with the same battle-tested confidence.

![Bar chart of memory per 768-dimension vector across quantization schemes: float32 at 3072 bytes, scalar quantization at 768 bytes (4x), TurboQuant 4-bit at 384 bytes (8x), product quantization at 192 bytes in a 16x example configuration (up to 64x possible), TurboQuant 2-bit also at 192 bytes (16x), and binary quantization at 96 bytes (32x)](vector-databases-assets/quantization-compression.svg)

*The float32 baseline dwarfs everything else — even scalar quantization's "modest" 4x reduction is a 75% memory cut, before reaching for the more aggressive schemes.*

- **Comparison table:**

| Technique | Compression vs. float32 | Typical recall impact | When to reach for it |
|---|---|---|---|
| Scalar quantization | ~4x (32-bit → 8-bit) | Small, well-understood | Default first step; simple, cheap to build and rescore |
| Product quantization | 16x typical example, up to 64x (codebook-dependent) | Moderate, tunable via codebook size | Large collections where scalar quantization isn't enough, and codebook training cost is acceptable |
| Binary quantization | ~32x (1 bit/dimension) | Significant without rescoring; recoverable with oversampling | Very large collections prioritizing memory and raw speed, paired with a rescoring step |
| TurboQuant (4-bit) | ~8x | Comparable to scalar quantization, at half the memory | Same use case as scalar quantization, when the extra rotation step's cost is acceptable |
| TurboQuant (2-bit/1-bit) | ~16x/32x | Better than binary quantization at matched compression | Same use case as binary quantization, when the accuracy gap matters |

> **Note.** None of these techniques change what gets stored *permanently* in a well-designed system — most databases keep (or can regenerate) the original full-precision vectors somewhere, using the quantized version purely as a faster, cheaper index for the bulk of the search, and falling back to full precision for final rescoring. Quantization is a search-time and memory optimization, not a decision to permanently discard information.

---

## 3. Hybrid search and reciprocal rank fusion

![Diagram of hybrid search: a query fans out into a dense ANN search and a sparse BM25/SPLADE search, each producing its own ranked list, which reciprocal rank fusion then combines into a single final ranking using 1/(k+rank)](vector-databases-assets/hybrid-search-rrf.svg)

*"Resolving error E-4471" ranks 15th in the dense-only list but 1st in the sparse list — RRF's rank-based sum is enough to pull it to the top of the fused ranking without either list's raw scores ever being compared.*

- **The setup.** A hybrid search runs a dense ANN search and a sparse search (BM25 or SPLADE, from [Part 2](vector-databases-part2.md)) against the same query, independently, and needs to merge two ranked lists into one final ranking. This is harder than it sounds because the two lists' scores aren't comparable — a cosine similarity of 0.82 and a BM25 score of 14.3 don't live on the same scale, and naively averaging them either requires ad-hoc normalization or produces meaningless results.

- **How this actually gets wired up in a real system.** "Dense search" and "sparse search" aren't two separate databases bolted together — in a system like Qdrant, a single point (§3 of Part 1's anatomy) can carry both a dense vector and a sparse vector as two differently-named vectors on the same record. One write populates both indexes; one query can ask for results from both named vectors and fuse them, either inside the database itself or in the application layer that calls it. The two-searches-then-fuse picture in this section is the logical shape of hybrid search — the infrastructure underneath it is one collection, not two.

- **Reciprocal rank fusion (RRF) sidesteps the scale problem entirely by discarding scores and using rank instead.** For each document, RRF computes a fused score as the sum, across all the ranked lists it appears in, of `1 / (k + rank)`, where `rank` is that document's position in a given list (1st, 2nd, 3rd...) and `k` is a small constant (commonly 60) that dampens the impact of very high ranks so the method isn't dominated by whichever single list ranked something first.
  - A document that ranks highly in *both* the dense and sparse lists accumulates a high fused score from both terms; a document that ranks well in only one list still contributes, just less.
  - Because the formula only ever looks at rank position, not the underlying score's magnitude or units, it works identically well fusing two dense searches, a dense and a sparse search, or three or more ranked lists from entirely different retrieval methods — this generality is the main reason it became the default fusion method rather than a learned or hand-tuned score-combination rule.

  **RRF, in plain terms.** Two friends each hand you a ranked list of their favorite restaurants in town, but one grades on a 5-star scale and the other on a 100-point scale — the numbers themselves aren't comparable, so averaging "4.5 stars" and "82 points" is meaningless. What you can compare is *position*: restaurant X was friend A's #1 and friend B's #3, restaurant Y was #6 and #1. Instead of trying to reconcile stars and points, you just add up how good each restaurant's *positions* were across both lists, giving more weight to a #1 than a #6. That's RRF — it never looks at the underlying scores, only at where each thing landed on each list.

- **Why rank-based fusion beats naive score blending in practice.** A weighted-sum approach (`0.5 * cosine_score + 0.5 * bm25_score`) requires the two scores to be normalized onto compatible ranges, and that normalization is itself dataset- and query-dependent — a BM25 score's range shifts with corpus size and term rarity, while a cosine similarity is bounded but its practically-observed range for "good matches" varies by embedding model. RRF needs none of that tuning, which is a meaningful operational advantage even though it's a comparatively simple statistical method.

- **The cost of hybrid search is running two retrieval paths per query instead of one** — both the dense ANN search and the sparse inverted-index search have to execute, and their results have to be merged, before a response is ready. Part 1 flagged this same tradeoff in its "Alternatives and relatives" section; RRF is the specific mechanism that makes paying that cost worthwhile, by making the merge step cheap and robust once both searches have already run.

- **RRF in code — small enough to run by hand.** The whole method is a handful of lines; there's no library required to see how it behaves on the two ranked lists from the diagram above.

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    # highest fused score first
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

# dense search ranks the general match first and buries the specific one at rank 8
dense_results = ["troubleshooting_transactions", "refund_policy", "reset_password",
                  "shipping_address", "account_settings", "billing_history",
                  "two_factor_auth", "resolving_e4471"]

# sparse search does the opposite: the exact rare-term match wins outright,
# and the general article — sharing only common terms like "payment" — ranks 9th
sparse_results = ["resolving_e4471", "payment_gateway_codes", "declined_card_reasons",
                   "chargeback_process", "refund_timeline", "invoice_errors",
                   "duplicate_charges", "currency_conversion", "troubleshooting_transactions"]

for doc_id, score in reciprocal_rank_fusion([dense_results, sparse_results]):
    print(f"{score:.5f}  {doc_id}")
```

- Running this (verified by executing it, not hand-computed) prints, top of the list:
  ```
  0.03110  resolving_e4471
  0.03089  troubleshooting_transactions
  0.01613  refund_policy
  0.01613  payment_gateway_codes
  ...
  ```
  - `resolving_e4471`: `1/(60+8) + 1/(60+1) = 0.01471 + 0.01639 = 0.03110`
  - `troubleshooting_transactions`: `1/(60+1) + 1/(60+9) = 0.01639 + 0.01449 = 0.03089`
  - The two land within three thousandths of each other, and `resolving_e4471` edges ahead specifically because it won its list outright (rank 1) while `troubleshooting_transactions` never did (rank 1 and rank 9) — RRF rewards a strong, unanimous signal from one list about as much as a decent showing in both, which is the actual, unglamorous shape of the tradeoff, not a landslide win for either document.
  - Every document appearing in only one list (`refund_policy`, `payment_gateway_codes`, and everything below them) scores noticeably lower than either document that appeared in both, regardless of how high its single rank was — this is the "reward documents both signals agree on" behavior described above, visible directly in the output rather than asserted.

> **Worth remembering.** RRF fuses *rankings*, not *relevance judgments* — it has no notion of whether a document is actually a good answer, only where each retrieval method placed it. Fusion improves robustness by combining two different notions of "close" (semantic and lexical), but, like everything else in this pair of articles, it doesn't substitute for evaluating whether the final results are actually useful for real queries.

---

## 4. What this doesn't solve

Part 1 closed its anatomy section with a reality check — a vector database is a search mechanism, not a relevance guarantee. Part 2 raised the same point about sparse retrieval alone. Quantization and hybrid search deserve the same treatment: each closes a real gap, but each has its own blind spot.

- **Quantization's accuracy loss isn't as uniform as "small and controllable" makes it sound.** The recall numbers cited in §2 assume a roughly well-behaved, redundant embedding space — which is generally true of major off-the-shelf embedding models, but not guaranteed for a narrowly fine-tuned or unusual embedding space. A vector space where the meaningful signal really is concentrated in a handful of dimensions can lose disproportionately more to scalar or binary quantization than the general-case numbers suggest, which is exactly the failure mode TurboQuant's rotation step (§2) is designed to blunt — but blunting it isn't the same as it never happening.

- **RRF's `k=60` is a popular empirical default, not a principled constant**, and the formula has no way to weight one retrieval leg over the other for a given query. A query that's obviously an exact SKU or error-code lookup and a query that's obviously conversational both get the same blind rank-summing treatment — RRF doesn't know, and can't be told without extra machinery on top, that one query would be better served leaning almost entirely on the sparse list.

- **Hybrid search doubles the maintenance surface, not just the query-time cost already noted in §3.** There are now two indexes — an inverted index and an ANN index — that both have to stay in sync with the live corpus. A document that gets updated and re-embedded but not re-tokenized (or vice versa) can silently drift so that its dense and sparse representations describe two different versions of the content, a staleness risk Part 1 flagged for a single index that's now doubled here.

- **Filtering has to be pushed into both legs, not applied only at the end.** Part 1 §3 raised the pre-filter-versus-post-filter tradeoff for a single ANN search; hybrid search doubles it. If a metadata filter (say, "only articles updated in the last 90 days") is applied only after fusion, both the dense and sparse legs waste work ranking documents that get discarded anyway, and a document that would have passed the filter can get pushed out of each list's top-*k* by documents that ultimately don't qualify. The filter generally needs to reach both the ANN search and the inverted-index lookup *before* RRF ever runs, not as a final cleanup step.

> **Worth remembering.** Sparse retrieval, quantization, and hybrid fusion each fix a specific, real gap left by dense-only search — but none of them is a free lunch, and stacking all three doesn't add up to a system that no longer needs evaluation on real queries. The same closing point from Part 1 and Part 2 applies with more moving parts, not fewer: a search mechanism, however well-engineered, is not a relevance guarantee.

---

## 5. A worked example: where hybrid search wins

Return to Part 1's support-site scenario: 40,000 help articles, dense-embedded and indexed with HNSW. Now a user searches for **"error code E-4471 payment failure."**

- **What the dense-only search does.** The query embeds into the same space as before, and the ANN walk correctly surfaces articles that are *semantically* about payment failures — likely including *"Troubleshooting failed transactions,"* the same article Part 1's example found for a differently-worded query. But nothing about the embedding space treats "E-4471" as special: the embedding model has almost certainly never seen that exact string during training, and dense embeddings generally blur specific identifiers into the general semantic neighborhood of "an error code," rather than preserving them exactly. If there's a specific article titled *"Resolving error E-4471"* buried among the 40,000, dense search alone gives it no structural advantage over a dozen other payment-related articles — it has to win purely on overall semantic similarity, and specific identifiers are exactly what dense embeddings are weakest at preserving.

- **What the sparse search (BM25 or SPLADE, from Part 2) does in parallel.** The token "E-4471" (or "e" / "4471" depending on tokenization) is treated as an exact, rare term. Inverse document frequency means a term that appears in only one or two of the 40,000 articles gets weighted very heavily — the article containing that exact code will score dramatically higher than any article that merely discusses payment failures in general, because BM25's scoring mechanism is built precisely to reward rare, exact matches.

- **What RRF does with the two lists.** *"Resolving error E-4471"* likely ranks 1st or 2nd in the sparse list (exact rare-term match) but might rank 15th or lower in the dense list (semantically relevant but not distinctively so). *"Troubleshooting failed transactions"* likely ranks highly in both — it's a strong general semantic match, and if it happens to mention error codes at all it also picks up some sparse-list credit. RRF's `1/(k+rank)` sum rewards both: the specific article gets pulled up by its dominant sparse-list rank even though the dense list alone would have buried it, while the general troubleshooting article stays near the top because both signals agree on it.

- **The takeaway from the example.** Dense-only search would very plausibly return the *directionally* correct but *not most specific* article, missing the one that mentions the user's exact error code by name. **Hybrid search recovers the specific match without sacrificing the general one** — which is exactly the failure mode Part 1 flagged in its "What vector databases don't solve" section: semantic search doesn't understand exact-match requirements, and this is what fixing that gap actually looks like in a concrete query.

---

## 6. Summary table

| Concept | What it means | Where it lives |
|---|---|---|
| Scalar quantization | Per-dimension float→8-bit-int compression | ~4x memory reduction, simplest scheme |
| Product quantization | Sub-vector clustering into codebooks | 16x typical, up to 64x possible, more index-build cost |
| Binary quantization | Per-dimension single-bit compression | ~32x reduction, ~40x speed, needs rescoring |
| TurboQuant | Rotation-then-quantize, at 4/2/1-bit widths | Matches or beats scalar/binary quantization's recall at the same compression |
| Reciprocal rank fusion (RRF) | Rank-based (not score-based) merge of multiple ranked result lists | Combines dense + sparse search results into one ranking |

---

## 7. Key takeaways

- **Quantization trades precision for memory, the same trade ANN search makes for latency** — the techniques differ mainly in how aggressively they compress and how much of that loss gets recovered via rescoring against original vectors.
- **Binary and rotation-based (TurboQuant) quantization make the most aggressive compression usable specifically because of oversampling and rescoring** — the compressed representation narrows the field cheaply, and full-precision vectors settle the final ranking.
- **Reciprocal rank fusion works because it never has to compare scores across incompatible scales** — it fuses positions in a ranked list, which is why it generalizes cleanly to combining any number of retrieval methods, not just exactly two.
- **Hybrid search directly closes the gap Part 1 identified as unsolved** — "semantic search doesn't understand exact-match requirements" is answered, in practice, by running lexical/sparse search alongside dense search and fusing the results, not by making the embedding model try harder.
- **None of it is a free lunch.** Quantization loss isn't uniform across embedding spaces, RRF can't tell a SKU lookup from a conversational query, and running two indexes means two things that can drift out of sync — stacking techniques closes gaps, it doesn't remove the need to evaluate the whole pipeline on real queries.

---

## 8. Further reading

- **"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"** (Cormack, Clarke & Buettcher, 2009) — the paper that introduced RRF, notable for how simple the method is relative to how well it performs: [dl.acm.org/doi/10.1145/1571941.1572114](https://dl.acm.org/doi/10.1145/1571941.1572114)

- **Qdrant's ["What is a Vector Database?"](https://qdrant.tech/articles/what-is-a-vector-database/)** — the article this piece draws its quantization framing from, including TurboQuant, binary quantization, and named vectors, in the context of a real production system's design choices.

- **"Billion-scale similarity search with GPUs"** (Johnson, Douze & Jégou, 2017) — already cited in Part 1 for IVF, also the standard reference for product quantization at scale: arxiv.org/abs/1702.08734

- **The companion [vector databases, Part 1](vector-databases.md)** — for the embedding model, ANN indexing (HNSW/IVF), distance metrics, and metadata filtering this article builds on top of.

- **[Vector databases, Part 2](vector-databases-part2.md)** — for BM25 and SPLADE, the sparse retrieval methods this article's hybrid search section fuses with dense search.

---

Compression and fusion don't make for as clean a story as "embeddings understand meaning" — there's no single elegant idea here the way there is with HNSW's layered graph. What ties this article's topics together instead is that each one exists because dense vector search, however good the embedding model, leaves something on the table: memory at scale, or a ranking that has to reconcile two genuinely different notions of "relevant." None of that is a flaw in the dense-vector approach so much as a reminder that Part 1's closing point cuts both ways — a search mechanism is not a relevance guarantee, and the mechanisms that get you closer to one, in production, are usually more than one mechanism at all.
