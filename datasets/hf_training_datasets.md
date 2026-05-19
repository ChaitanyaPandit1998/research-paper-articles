# Hugging Face High-Quality Training Datasets

> Comprehensive reference of high-quality datasets on Hugging Face, covering pretraining, instruction tuning, math, code, and multimodal domains.

---

## Part 1 — Large-Scale Web Pretraining

| Dataset | Size | Languages | Source(s) | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **FineWeb** | ~350B tokens (sample) | English | Common Crawl (114 snapshots, 2013–2025) | Language detection, deduplication, datatrove pipeline | Best-in-class English web pretraining; outperforms C4, Dolma, Pile on benchmarks | ODC-BY |
| **FineWeb-Edu** | 1.3T tokens (score ≥ 3) | English | FineWeb → educational filter | Llama-3-70B classifier scores 0–5; removes 92% of FineWeb | Knowledge-intensive benchmarks (MMLU, ARC); best for education-heavy pretraining | ODC-BY |
| **FineWeb-2** | 3T+ tokens | 1,000+ languages | Common Crawl (96 snapshots) | Per-language global dedup, hundreds of ablation experiments | Brings FineWeb quality to 1,000+ languages; beats mC4, CulturaX on most evaluated languages | ODC-BY |
| **ClimbMix** | 400B tokens | English | NVIDIA ClimbLab (Nemotron-CC + SmolLM corpus) | 1,000-topic clustering, ad detector, educational scorer, CLIMB iterative mixture optimization | Optimized topic-balanced pretraining mix; 1B model exceeds Llama-3.2-1B by 2.0% at 400B tokens | Apache 2.0 |
| **DCLM-Baseline-1.0** | 4T tokens / 3B docs | English | Common Crawl (240T token pool) | Heuristic cleaning, Bloom filter dedup, fastText classifier (OpenHermes + ELI5 signals) | Research baseline for DataComp-LM benchmark; 7B model hits 63.7 MMLU | CC-BY-4.0 |
| **Dolma v1.7** | 3T tokens | English | CC, C4, StarCoder, Reddit, S2ORC, arXiv, Wikipedia, StackExchange | Fuzzy dedup, fastText lang-ID, PII removal, toxicity filtering | Diverse composition (web + code + academic + books); transparent open-source pipeline | ODC-BY |
| **Dolma v3 Pool** | 9T+ tokens | English | Extended Dolma v1.7 sources | Extended Dolma pipeline | 3× larger than v1.7; same diverse composition philosophy | ODC-BY |
| **RefinedWeb (Falcon)** | ~600B tokens | English | Common Crawl | MacroData Refinement: URL blocklist, trafilatura extraction, strict exact + fuzzy dedup | Web-only data matching curated corpora; powers Falcon-7B/40B; multimodal-friendly (image URLs) | ODC-BY |
| **RedPajama-Data-1T** | 1.2T tokens | English + 20-lang Wikipedia | CC, C4, GitHub (permissive), arXiv, Wikipedia, StackExchange | CCNet pipeline, per-source dedup, quality filtering | Open-source LLaMA training data reproduction; 370+ models trained on it | Multi-license |
| **RedPajama-Data-V2** | 30.4T tokens (5-lang deduped) | EN, DE, FR, IT, ES | 84 CC snapshots (2018–2023) | 40+ quality annotations (perplexity, toxicity, n-gram dedup, MinHash at 4 thresholds) | Rich quality-signal dataset enabling fully custom filtering; streaming-ready | CC Foundation ToU |
| **SlimPajama-627B** | 627B tokens | English-primary | RedPajama-1T (deduplicated) | MinHashLSH (Jaccard 0.8), removes docs < 200 chars; 49.6% bytes removed | Cleaner, deduplicated RedPajama; reduces memorization; open dedup infrastructure | Apache 2.0 |
| **C4 / mC4** | 13.8 TB total | 108 languages | Common Crawl | langdetect/CLD3, boilerplate removal, optional badwords filter | Foundational cleaned web text; mC4 variant covers 108 languages | ODC-BY |
| **Cosmopedia** | 25B tokens / 30M files | English | Web clusters, Stanford/OpenStax/Khan Academy/WikiHow/AutoMathText | Mixtral-8x7B synthesis, 110 curated topics, MinHash dedup, benchmark decontamination | Largest open synthetic dataset; maps world knowledge into textbook/blogpost/story formats | Apache 2.0 |

---

## Part 2 — Multilingual Pretraining

| Dataset | Size | Languages | Source(s) | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **CulturaX** | 6.3T tokens | 167 languages | mC4 v3.1.0 + OSCAR (4 versions) | Language ID, URL filtering, KenLM-based metric cleaning, MinHash fuzzy dedup | >50% non-English; best multilingual pretraining corpus after FineWeb-2 | mC4/OSCAR terms |
| **mC4** | 9.7 TB | 108 languages | Common Crawl | CLD3 language detect, boilerplate removal, dedup | Foundational multilingual web text for 100+ languages | ODC-BY |
| **HPLT v2.0** | 8T tokens (mono) + 380M sentence pairs (parallel) | 191 languages | Common Crawl + WIDE web archives | Quality scoring 0–10, robots.txt compliance, PII indicators, per-segment lang-ID | CC0 public domain; parallel data for 51 languages; covers endangered languages | CC0-1.0 |
| **Glot500** | 1B–10B tokens | 500 low-resource languages | 150+ datasets + multilingual web crawls | Multi-stage dataset curation | Extreme multilingual coverage; focus on endangered and low-resource languages | CC0-1.0 |
| **ROOTS** | 1.6 TB | 59 languages | Compiled for BLOOM training | Ethical data governance pipeline | Ethically-curated multilingual corpus; used to train BLOOM-176B | Varied |

---

## Part 3 — Instruction Tuning & Alignment

| Dataset | Size | Generation Method | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|
| **Tulu 3 SFT Mixture** | 939k samples | Multi-source (18 datasets: human + synthetic) | Domain coverage (math/code/safety/multilingual), persona-based synthesis | Best overall SFT mix; SFT→DPO→RLVR pipeline; 67+ languages | ODC-BY-1.0 |
| **UltraFeedback** | 64k prompts / 256k responses | 17 diverse LLMs + GPT-4 annotation | 4-aspect rating (instruction-following, truthfulness, honesty, helpfulness), 1–5 scale | Gold-standard RLHF/DPO; richest annotation schema for preference learning | MIT |
| **UltraChat 200k** | 515k examples | ChatGPT conversation generation (1.4M → filtered) | Truecasing correction, non-committal response removal | High-quality filtered SFT chat data; train/test splits ready | MIT |
| **Magpie Ultra v0.1** | 50k pairs | Self-synthesis from Llama-3.1-405B (no human seeds) | ArmoRM scoring, Llama Guard 3 safety, Faiss diversity filtering | Best automatic quality control; scores usable as DPO chosen/rejected pairs | Llama 3.1 |
| **SmolTalk** | 1.1M samples | Mixture of Magpie-Ultra + curated public datasets | distilabel curation, IFEval decontamination, 1.7B-scale data ablations | Optimized for small models (1B–7B); editing, summarization, reasoning | Apache 2.0 |
| **OpenHermes 2.5** | ~1M examples | GPT-4 distillation + WizardLM, GPTeacher, airoboros | Multi-source compilation, ShareGPT conversation format | General instruction following, roleplay, coding, creative writing at scale | Mixed |
| **WizardLM Evol-Instruct V2** | ~196k examples | Iterative Evol-Instruct (In-Depth + In-Breadth evolution) | Step-by-step complexity increase: constraints, reasoning depth, concretization | Complex multi-step instruction following and reasoning | MIT |
| **Orca AgentInstruct 1M** | 1.05M examples | Azure GPT-4 agentic framework | Diverse task coverage (editing, coding, reading comprehension) | 40% gain on AGIEval, 54% on GSM8K vs. baselines | CDLA-Permissive-2.0 |
| **Anthropic HH-RLHF** | 169k pairs | Iterative human preference annotation | Human preference pairs, red-team annotations, separate helpfulness/harmlessness splits | Industry-standard RLHF preference data; reward model training and DPO | MIT |
| **WildChat 1M** | 838k conversations | Real ChatGPT user interactions (200+ countries, 68 languages) | OpenAI Moderation API, Detoxify, Presidio PII de-identification | Real-world conversation diversity; multilingual; used in Tulu 3 | ODC-BY |
| **OpenAssistant OASST2** | 135k messages / 13.8k trees | Human-written conversations | Multi-stage human review, toxicity + quality + helpfulness scoring | 24-language multilingual chat; human-verified conversation trees | Apache 2.0 |
| **No Robots** | 10k examples | 100% human-written by skilled annotators | 10 task categories, multi-turn format, zero synthetic contamination | Highest-quality human-verified SFT; benchmarked on MT-Bench and AlpacaEval | CC BY-NC 4.0 |
| **Aya Dataset** | 204k pairs | Human crowd annotation (119 countries) | Peer review sampling, platform safeguards | 65-language multilingual instruction tuning; only major human-annotated multilingual set | Apache 2.0 |
| **FLAN v2** | ~378M rows | Templated instruction format across 1,800+ NLP tasks | Task balancing, zero/few-shot/CoT variations, input inversion | Massive multi-task instruction tuning; zero-shot and few-shot learning at scale | CC-BY-4.0 |
| **Stanford Alpaca** | 52k examples | GPT-3 text-davinci-003 Self-Instruct | Simplified Self-Instruct pipeline | Pioneering open SFT dataset; widely used but has known model-generated biases | CC BY-NC 4.0 |
| **CoCoNot** | 13.8k examples | Human-curated taxonomy + GPT-4 responses | 5-category noncompliance taxonomy, contrasting query pairs | Structured safety/refusal training; balanced helpfulness vs. harm avoidance | ODC-BY |

---

## Part 4 — Mathematics

| Dataset | Size | Source(s) | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|
| **NuminaMath-1.5** | 896k problems | Olympiads, CN K-12, Orca-Math, AoPS Forum, synthetic | OCR from PDFs, CoT formatting, answer validation, 11 curated sources | Competition-level math; NuminaMath-7B won AIMO prize (29/50) | Apache 2.0 |
| **OpenMathInstruct-2** | 14M problem-solution pairs | GSM8K + MATH (augmented via Llama-3.1-405B) | Solution + problem augmentation, code-execution verification | Largest open math instruction dataset | NVIDIA |
| **MetaMathQA** | 395k examples | GSM8K + MATH training sets only (no test leakage) | Answer augmentation, rephrasing, variable substitution, fill-in-the-blank | MetaMath-Mistral-7B: 77.7% GSM8K; bootstrap math questions safely | MIT |
| **Orca-Math 200K** | 200k word problems | Lila + DMath seeds → GPT-4 Turbo | GPT-4 Turbo generation, grade-school focus | Orca-Math-7B: 86.81% on GSM8K; grade-school math word problems | MIT |
| **OpenWebMath** | 6.3M docs / 14.7B tokens | 130k+ domains from Common Crawl (LaTeX-heavy pages) | LaTeX extraction, perplexity filter, SimHash dedup, math content scoring | High-quality math web text; superior to 20× general data volume in LLM training | ODC-BY |
| **MathPile** | ~9.5B tokens | Textbooks, arXiv, Wikipedia, ProofWiki, StackExchange, CC | Language detection, stop-word + formula quality checks | K-12 through postgraduate competition-level math corpus | CC BY-NC-SA 4.0 |
| **Proof-Pile-2** | 55B tokens | arXiv (29B) + OpenWebMath (15B) + AlgebraicStack (11B) | Domain-specific content filtering | Formal + informal theorem proving; autoformalization; Lean/Isabelle/Coq support | Source licenses |
| **NuminaMath CoT** | 859k problems | Olympiads, CN K-12, Orca-Math, GSM8K, AMC/AIME | OCR, translation to English, CoT formatting | Chain-of-thought formatted math; wide difficulty range | Apache 2.0 |
| **GSM8K** | 8.5k problems | Human-created (Upwork + Surge AI) | Multi-worker validation, 2–8 step problems, ~1.7% error rate | Grade-school math reasoning; foundational evaluation benchmark | MIT |
| **DeepMind Mathematics** | Synthetic (procedural) | Procedurally generated | Module-organized, 3 difficulty levels (easy/medium/hard) | Systematic math reasoning: algebra, arithmetic, calculus, measurement | DeepMind Terms |

---

## Part 5 — Code

| Dataset | Size | Prog. Languages | Source(s) | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **The Stack v2** | 67.53 TB / 3.28B files | 600+ | Software Heritage (104M GitHub repos) | Permissive-license only, duplicate removal, opt-out process | Largest open code pretraining dataset; most comprehensive language coverage | Permissive (varied) |
| **StarCoder Data** | ~250B tokens (783 GB code) | 86 | The Stack v1.2 + GitHub Issues + Jupyter notebooks + Git commits | Near-dedup, PII removal, decontamination across languages | Code LLM training with diverse supplementary sources (issues, notebooks, commits) | Custom ToU |
| **CodeParrot GitHub Code** | 115M files / ~873 GB | 32 | GitHub repositories | Exact + near-dedup, malware filtering, 15+ license types tracked | Large-scale GitHub code; filterable by language and license | Varied |
| **The Stack Smol** | 3.1 TB | 30 | The Stack (curated subset) | Permissive-license filter | Smaller curated Stack for resource-constrained code training | Permissive |

---

## Part 6 — Multimodal / Vision-Language

| Dataset | Size | Source(s) | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|
| **LAION-5B** | 5.85B image-text pairs (2.32B English) | Common Crawl image-text pairs | CLIP similarity filtering, aesthetic scoring (v2 ≥ 6.5), face detection | Powers Stable Diffusion, FLUX; largest open vision-language dataset | Varied |
| **DataComp-1B** | 1.39B pairs / 8.86 TB | Web-scraped image-text pairs | CLIP similarity (B32 + L14), face detection, quality annotations | Discriminative (CLIP) + generative (diffusion) vision-language training | CC-BY-4.0 (metadata) |
| **DataComp-12M** | 12M pairs | DataComp-1B best-pool subset | Best-performing subset selection | Efficient vision-language training; superior performance-per-sample | CC-BY-4.0 |

---

## Quick Selection Guide

| Goal | Recommended Datasets |
|---|---|
| English LLM pretraining | FineWeb-Edu → FineWeb → DCLM-Baseline → Dolma v1.7 |
| Multilingual pretraining | FineWeb-2 → CulturaX → HPLT v2.0 → mC4 |
| Topic-optimized pretraining mix | ClimbMix |
| General SFT (instruction tuning) | Tulu 3 SFT Mixture → SmolTalk → OpenHermes 2.5 |
| RLHF / DPO preference training | UltraFeedback → HH-RLHF → Magpie Ultra |
| Mathematics reasoning (SFT) | OpenMathInstruct-2 → NuminaMath-1.5 → MetaMathQA |
| Math pretraining corpus | OpenWebMath → MathPile → Proof-Pile-2 |
| Code pretraining | The Stack v2 → StarCoder Data |
| Multilingual instruction tuning | Aya → Tulu 3 SFT → WildChat 1M → OASST2 |
| Small model fine-tuning (1B–7B) | SmolTalk → Magpie Ultra |
| Vision-language / multimodal | LAION-5B → DataComp-1B |
| Safety / refusal training | CoCoNot → HH-RLHF |
| Human-verified quality (no synthetic) | No Robots → OASST2 |
