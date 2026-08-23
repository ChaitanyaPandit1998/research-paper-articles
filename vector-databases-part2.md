# Vector Databases, Part 2: Sparse Retrieval — BM25, SPLADE, and Why Exact Terms Still Matter

**Pull quotes:**
- "Dense vectors buy you meaning. Sparse vectors buy you exactness. A production retrieval system doesn't pick one — it runs both and merges the results."
- "BM25 isn't a legacy technique being phased out by embeddings — it's the other half of a two-part system."
- "SPLADE is the clearest illustration that 'sparse' and 'semantic' aren't opposites."

---

Part 1 covered the anatomy of a vector database built around dense embeddings — HNSW, IVF, distance metrics, filtering. This article covers the representation dense embeddings structurally can't replace: sparse vectors, and the two dominant ways of producing them. It works through:

- dense vectors versus sparse vectors, and why a database needs to support both
- BM25, the lexical scoring method dense embeddings never fully replaced
- SPLADE, a learned sparse model that sits between BM25 and dense embeddings
- what sparse retrieval alone still doesn't solve

Compression (quantization) and combining dense with sparse search (hybrid search, reciprocal rank fusion) are covered in the companion [Part 3](vector-databases-part3.md), once this article has established what a sparse vector actually is.

---

## Table of contents

1. [Dense vs. sparse vectors](#1-dense-vs-sparse-vectors)
2. [BM25: lexical scoring that never went away](#2-bm25-lexical-scoring-that-never-went-away)
3. [SPLADE: learned sparse retrieval](#3-splade-learned-sparse-retrieval)
4. [What sparse retrieval alone doesn't solve](#4-what-sparse-retrieval-alone-doesnt-solve)
5. [Summary table](#5-summary-table)
6. [Key takeaways](#6-key-takeaways)
7. [Further reading](#7-further-reading)

---

## 1. Dense vs. sparse vectors

![Diagram comparing dense and sparse search: a query splits into a dense embedding that walks an HNSW graph to find approximate nearest neighbors, and a sparse vector that looks up only the postings lists for its non-zero terms in an inverted index](vector-databases-assets/sparse-vs-dense-index.svg)

*Same query, two structurally different index lookups — a graph walk on one side, a postings-list lookup on the other. Neither is a special case of the other.*

- **Dense vectors, recapped from Part 1.** An embedding model maps text into a fixed-length vector — typically 300 to a few thousand dimensions — where almost every dimension carries some value, and distance between vectors approximates semantic similarity. Search over dense vectors uses an ANN index (HNSW, IVF) because comparing a query against every stored vector doesn't scale.

- **Sparse vectors work on a different principle entirely.** Instead of a fixed number of dense dimensions, a sparse vector has one dimension per term in a (often huge — tens or hundreds of thousands of terms) vocabulary, and almost all of those dimensions are zero for any given piece of text.
  - A document about "payment declined" might have non-zero weights only on the dimensions for "payment," "declined," "transaction," and a handful of related terms — everything else in the vocabulary is exactly zero, not just small.
  - Each non-zero dimension carries a weight representing how relevant that term is to this piece of text — how that weight gets computed is exactly what distinguishes BM25 from SPLADE, covered in the next two sections.

- **Sparse vectors are searched with an inverted index, not an ANN graph.** Because almost every dimension is zero, there's no need for HNSW's graph-walking trick — instead, the database maintains, for every vocabulary term, a list of which documents have a non-zero weight on that dimension. A query only needs to score documents that share at least one non-zero dimension with it, via a dot product over just those shared dimensions.
  - This is the same underlying data structure a classic full-text search engine like Elasticsearch or Lucene already uses — sparse-vector search is best understood as inverted-index search wearing vector-database vocabulary, not a new mechanism.
  - It's also why sparse search doesn't have an approximation knob the way HNSW has `ef_search`: the inverted-index lookup is already exact and already fast, because zero-weight dimensions are never touched at all rather than approximately skipped.

- **Why bother with both.** Dense vectors capture meaning across a vocabulary gap ("dog" and "canine" land near each other). Sparse vectors capture exact terms with no vocabulary gap at all — an exact SKU, an error code, a rare technical term — because they don't try to generalize away from the literal tokens present. **Neither representation subsumes the other**; [Part 3](vector-databases-part3.md) covers how a database combines results from both.

  **Dense vs. sparse, in plain terms.** Think of two different ways to find a book in a library. A dense vector is like describing the *vibe* of a book to a well-read librarian — "something like a mystery, but cozier, set in a small town" — and trusting them to walk over to roughly the right shelf, even though you never said a single word that's actually printed on the cover. A sparse vector is like using the card catalog: you look up the exact words on the spine, and you either find an exact match or you don't — no vibes, no nearby-shelf browsing, just "does this word appear, and how important is it." Neither approach is strictly better at finding books; a good library search desk uses both, which is exactly what hybrid search does.

> **Note.** "Sparse" and "dense" describe the *shape* of the vector — mostly-zero versus mostly-nonzero — not which model produced it. A sparse vector isn't automatically simpler or less "learned": SPLADE (§3) produces sparse vectors using a full neural network, while a dense embedding model can, in principle, produce a vector that happens to be sparse. In practice, though, the two shapes are strongly associated with two different families of technique, which is why this article treats them as a natural pairing.

---

## 2. BM25: lexical scoring that never went away

- **BM25 (Best Matching 25) is a scoring function, not a model.** Given a query and a document, it computes a relevance score from three ingredients, with no training and no neural network involved:
  - **Term frequency (TF)** — how often a query term appears in the document, with diminishing returns (a term appearing 10 times isn't 10x as relevant as it appearing once).
  - **Inverse document frequency (IDF)** — how rare the term is across the whole collection; a term that appears in nearly every document (like "the") contributes almost nothing to the score, while a rare term is weighted heavily.
  - **Length normalization** — a term match in a short document counts for more than the same match in a long one, since matches in long documents are partly a function of the document just containing more words.

  **BM25, in plain terms.** Imagine grading how well a book matches your search by three simple rules any librarian would intuit without a computer. First, if the book mentions your search word a lot, it's probably relevant — but the 10th mention doesn't excite you as much as the 1st did, so the score for repetition levels off (**term frequency**). Second, if your search word is "the," it's in every book, so it tells you nothing — but if it's "photosynthesis," and only three books in the whole library use that word, finding it is a strong signal (**inverse document frequency**). Third, a one-page pamphlet that happens to say your word once is a stronger match than a 900-page encyclopedia that says it once, just by proportion (**length normalization**). BM25 is just those three intuitions, turned into a formula and run automatically over every document.

- **This is exactly the "weight" that turns a document into a sparse vector.** Each vocabulary term is a dimension; BM25 provides the formula for the non-zero weight on the terms actually present in a document. Framed this way, BM25 isn't a separate system bolted onto vector search — it's one specific, unlearned way of producing the sparse vectors described in §1.

- **Why it's still standard infrastructure decades after Word2vec.** BM25 has none of the failure modes a dense embedding has:
  - **It's exact** — a query for "invoice #4471" cannot accidentally match "invoice #4472," because BM25 has no notion of "close enough," only "present or absent."
  - **It's interpretable** — a relevance score can be explained per term, which matters for debugging why a result ranked where it did, in a way a 768-number dense vector cannot be explained.
  - **It requires no training data and no embedding model to keep in sync with the corpus** — re-indexing a changed document is just re-tokenizing it, with none of the staleness risk Part 1 flagged for dense embeddings.
  - **It's cheap** — no GPU inference at index time or query time, just tokenization and arithmetic.

- **What it structurally cannot do.** BM25 only ever matches literal tokens (or near-literal, depending on stemming/tokenization choices) — it has no mechanism for "canine vaccination schedules" to match "when should I take my dog to the vet," because "canine" and "dog" are different tokens with no shared vocabulary dimension. This is precisely the vocabulary-gap problem Part 1 opened with, and it's the reason BM25 alone is not a substitute for dense or learned-sparse retrieval, only a complement to it.

> **Worth remembering.** BM25 isn't a legacy technique being phased out by embeddings — it's the other half of a two-part system. Most production hybrid search stacks run BM25 (or a learned-sparse variant) and dense vector search in parallel over the same query, precisely because each covers the other's structural blind spot.

---

## 3. SPLADE: learned sparse retrieval

- **SPLADE (SParse Lexical AnD Expansion) sits between BM25 and dense embeddings.** Like BM25, it produces a sparse, vocabulary-dimensioned vector that can be indexed and searched with the same inverted-index machinery. Unlike BM25, the weights come from a trained neural network rather than a term-frequency formula — and, crucially, the network isn't restricted to only weighting terms that are literally present in the text.

- **Term expansion is the key idea.** Given a document about "the car wouldn't start," a SPLADE model can assign a non-zero weight to "automobile" or "ignition" even if neither word appears in the text, because the model has learned that those terms are relevant to the same underlying content. This directly attacks BM25's vocabulary-gap weakness while keeping the sparse representation's exactness and inverted-index speed for whatever terms *are* literally present.

  **SPLADE, in plain terms.** Go back to the card-catalog librarian from §1 — but now imagine a librarian who, when a book says "the car wouldn't start," quietly also files a card under "automobile" and "ignition," because they've read enough books to know those words are practically synonyms in context. You still search the card catalog the same exact way — flip to a word, see which books are filed under it — but the catalog itself has been filled in more generously than a strict word-for-word index would allow. That's SPLADE: same fast, exact lookup mechanism as BM25, but a smarter (and much more expensive to train) process decided which cards to file where.

- **How the weights get learned.** SPLADE is trained (typically on query-document relevance pairs, similar to how dense embedding models are trained) to produce sparse activations over a language model's vocabulary — using a sparsity-inducing regularization term during training so that most dimensions land at exactly zero rather than merely small, which is what keeps the resulting vectors compatible with fast inverted-index search rather than degrading into a dense vector in disguise.

- **Where it sits relative to the other two options:**
  - **Versus BM25** — SPLADE replaces a hand-designed statistical formula with learned weights and adds term expansion, at the cost of needing a trained model and GPU inference at index/query time, where BM25 needs neither.
  - **Versus dense embeddings** — SPLADE keeps the interpretability advantage (a SPLADE vector's non-zero dimensions correspond to actual vocabulary terms, so a match can still be explained per term) and the exact inverted-index search structure, while giving up some of the fluid, whole-sentence semantic reach a dense embedding gets from compressing meaning into a small number of dense dimensions.

- **Why it's worth knowing about even if you don't deploy it.** SPLADE is the clearest illustration that "sparse" and "semantic" aren't opposites — the field converged on it specifically because pure lexical matching (BM25) and pure dense embedding search each left something on the table, and SPLADE was one of the more influential attempts to take the better parts of both without needing a full hybrid pipeline.

> **Note.** SPLADE and hybrid dense+BM25 search are not mutually exclusive strategies — a system can run SPLADE as its sparse leg of a hybrid search instead of BM25, trading BM25's simplicity and zero training cost for SPLADE's term-expansion advantage. Which one to use is an engineering tradeoff (inference cost and infrastructure versus retrieval quality), not a settled question with one right answer.

---

## 4. What sparse retrieval alone doesn't solve

- **BM25 and SPLADE still don't handle typos or genuinely unseen vocabulary.** A query for "recieve" (misspelled) or a brand-new product name that's never appeared in the corpus has no vocabulary dimension to score against at all — stemming and light fuzzy matching paper over some of this, but it's a bolt-on, not something either scoring method was built to do. Dense embeddings, by contrast, degrade more gracefully here: a misspelled word still lands *somewhere* in embedding space, often close enough to be useful.

- **Neither BM25 nor SPLADE closes the vocabulary gap the way a dense embedding does — at best, they work around a narrow slice of it.** BM25 doesn't bridge the gap at all; SPLADE bridges a *learned* slice of it (the specific term-expansion pairs its training data covered), which is narrower than a dense model's continuous notion of similarity across the whole embedding space.

- **Neither is a complete retrieval strategy on its own in most production systems.** Both are usually deployed as one leg of a hybrid search rather than a sole retrieval method — which is why this article's companion, [Part 3](vector-databases-part3.md), covers exactly how that combination works, after first covering the compression techniques that make storing both representations at scale affordable.

> **Worth remembering.** Sparse retrieval fixes a real, structural gap in dense-only search — exact terms, rare identifiers, zero-vocabulary-gap matching — but it isn't a general-purpose replacement for semantic search either. The two are complementary tools, not competing ones, and the next article covers what happens when you actually run both.

---

## 5. Summary table

| Concept | What it means | Where it lives |
|---|---|---|
| Sparse vector | Mostly-zero, vocabulary-dimensioned vector; weight per present (or expanded) term | Scored via inverted index, not ANN graph |
| BM25 | Unlearned term-frequency / inverse-document-frequency scoring formula | Produces the weights for a classic sparse vector |
| SPLADE | Learned model producing sparse vectors with term expansion beyond literal input tokens | Trained neural network; still indexed as an inverted index |

---

## 6. Key takeaways

- **Dense and sparse vectors solve different problems, not competing versions of the same problem** — dense vectors bridge vocabulary gaps via learned semantic similarity, sparse vectors preserve exact terms via inverted-index matching, and production systems increasingly use both rather than choosing.
- **BM25 is a formula, not a model** — no training, fully interpretable, and still the right tool whenever exact-match precision matters more than semantic reach.
- **SPLADE shows sparse and semantic aren't opposites** — a learned model can produce a sparse, inverted-index-compatible vector that still expands beyond a document's literal tokens, splitting the difference between BM25 and dense embeddings.
- **Sparse retrieval alone still has real gaps** — typos, unseen vocabulary, and no continuous notion of similarity — which is exactly why it's normally paired with dense search rather than used alone.

---

## 7. Further reading

- **"Okapi at TREC-3"** (Robertson, Walker, Jones, Hancock-Beaulieu & Gatford, 1994) — the paper that introduced the BM25 ranking function still in wide production use today: [trec.nist.gov/pubs/trec3/papers/city.ps.gz](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)

- **"SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"** (Formal, Piwowarski & Clinchant, 2021) — the original SPLADE paper: arxiv.org/abs/2107.05720
  - A follow-up, **"SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval"** (Formal et al., 2021), refines the training approach: arxiv.org/abs/2109.10086

- **"Dense Passage Retrieval for Open-Domain Question Answering"** (Karpukhin et al., 2020) — already cited in Part 1, worth revisiting here for its direct comparison of dense retrieval against BM25: arxiv.org/abs/2004.04906

- **The companion [vector databases, Part 1](vector-databases.md)** — for the embedding model, ANN indexing (HNSW/IVF), distance metrics, and metadata filtering this article builds on top of.

- **[Vector databases, Part 3](vector-databases-part3.md)** — the direct continuation of this article: quantization (scalar, product, binary, TurboQuant) for shrinking vectors at scale, and reciprocal rank fusion for combining the dense and sparse search this article introduced into one ranking.

---

Sparse vectors don't make dense embeddings obsolete, any more than dense embeddings made keyword search obsolete when they arrived. What they do is close a specific, structural gap — exact terms, rare identifiers, zero-vocabulary-gap matching — that no amount of better embedding-model training fully closes, because it isn't a modeling problem, it's a representational one. The next article picks up from here: what it costs to store either representation at scale, and what it actually looks like to run both at once.
