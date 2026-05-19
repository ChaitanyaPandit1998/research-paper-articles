# Finance & Financial NLP Datasets on Hugging Face

> Comprehensive reference of high-quality finance datasets on Hugging Face, covering pretraining corpora, sentiment analysis, QA/reasoning, SEC filings, time-series, NER, instruction tuning, risk/credit, and multimodal financial data.

---

## Part 1 — Financial Question Answering & Numerical Reasoning

| Dataset | Size | Source(s) | Task Type | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **FinQA** | 8,281 QA pairs | S&P 500 earnings reports (1999–2019) | Numerical reasoning over tables + text | Expert-written questions, gold reasoning programs annotated | Multi-step numerical calculations over structured/unstructured evidence | MIT |
| **TAT-QA** | 16,552 questions / 2,757 hybrid contexts | Real-world financial reports (tables + paragraphs) | Hybrid QA (tables + text) | Diverse answer forms (single span, multi-span, free-form); derivation paths + answer scales | Complex multi-step reasoning over semi-structured financial reports | CC-BY-4.0 |
| **FinanceBench** | 150 annotated examples | SEC filings (10-K, 10-Q, 8-K; 50+ companies) | Open-book QA (extraction + numerical + logical) | 2,400+ manual review instances; 16 model configs tested for hallucinations | Enterprise-grade hallucination evaluation; GICS sector classification | CC-BY-NC-4.0 |
| **ConvFinQA** | 1,490 rows | Financial reports | Multi-turn conversational QA | Progressive information extraction | Conversational context understanding across multiple dialogue turns | Varies |
| **FiQA** | 17,072 rows (14.5k train / 2.56k test) | Financial domain knowledge base | Aspect-based opinion mining + QA | — | Aspect-based sentiment + QA; sentence similarity; feature extraction | CC BY-NC 3.0 |
| **FinDER** | 5,703 query-evidence-answer triplets | Real 10-K SEC filings; queries from financial professionals | RAG evaluation | Expert-annotated; acronym-heavy real-world queries | Evaluating RAG systems on complex financial documents | CC-BY-NC-4.0 |
| **Fino1 Reasoning Path FinQA** | 5,499 examples | FinQA + GPT-4o reasoning paths | Chain-of-thought financial QA | GPT-4o-generated reasoning explanations | Enhanced reasoning paths for transfer learning; used to train 21+ models | CC-BY-4.0 |
| **FinCoT** | 9,186 rows (7,690 SFT + 1,500 RL) | FinQA, TAT-QA, DocMath-Eval, ConvFinQA, BizBench-QA | Financial QA with CoT | GPT-4o reasoning paths aligned to expert workflows | Expert-aligned chain-of-thought for financial reasoning | Apache 2.0 / MIT / CC |
| **FinTextQA** | 1,262 QA pairs | Finance textbooks (81%) + government/regulatory docs (19%) | Long-form QA with source attribution | High-quality curation with RAG benchmark capabilities | Comprehensive well-sourced long-form answers from authoritative sources | CC-BY-NC-SA-4.0 |
| **Quantitative Finance Reasoning** | 128 QA pairs | Quant interview books + practical guides | Quantitative problem-solving | Gemini-validated, scored 0–10 | Black-Scholes, derivatives pricing, Itô's Lemma, stochastic calculus | Unspecified |

---

## Part 2 — Sentiment Analysis

| Dataset | Size | Source(s) | Task Type | Key Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **Financial PhraseBank** | 4,840 sentences | Financial news (OMX Helsinki listed companies, LexisNexis) | 3-class sentiment (positive / negative / neutral) | 16 expert annotators; 4 configurations by agreement level (50%, 66%, 75%, 100%) | Gold-standard financial sentiment benchmark; investor-centric perspective | CC-BY-NC-SA-3.0 |
| **FiQA Sentiment (TheFinAI)** | 235 rows | FiQA 2018 challenge (financial microblogs + news) | Aspect-based sentiment analysis | FiQA challenge curation | Social media and microblog sentiment in finance | CC BY-NC 3.0 |
| **Twitter Financial News Sentiment** | 11,932 documents | Finance-related tweets | 3-class sentiment (Bearish / Bullish / Neutral) | — | Real-time market sentiment from social media | Varies |
| **FinGPT Sentiment Train** | 76,772 examples | Financial news and sentiment corpus | Text classification (3-class and 7-class) | Instruction-response format; supports federated learning | Fine-grained sentiment; supports both centralized and federated training | MIT |
| **NOSIBLE Financial Sentiment** | 100,000 examples | Financial news | 3-class sentiment | Cleaned + deduplicated | Large-scale industry-standard financial sentiment | Varies |
| **Financial Classification (nickmuchi)** | 5,057 rows | Financial PhraseBank + Kaggle financial texts | Text classification (sentiment) | COVID-era financial text coverage | Covers pandemic-period financial language | Varies |

---

## Part 3 — Named Entity Recognition & Information Extraction

| Dataset | Size | Source(s) | Task Type | Entity Types | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **FiNER-139** | 1.12M rows (900k train / 112k val / 108k test) | 10,000+ SEC 10-K and 10-Q filings (2016–2020) | XBRL numeric entity recognition (IOB2) | 139 XBRL entity types (GAAP accounting concepts); 279 IOB2 labels | Numeric tokens from financial statements; professional auditor annotations | CC-BY-SA-4.0 |
| **FiNER-ORD** | 116,721 tokens / 201 articles | Financial news (webz.io; multiple sources) | Named entity recognition | PER, LOC, ORG (7 BIO classes) | Manually annotated (Doccano); representative financial NER benchmark | CC-BY-NC-4.0 |
| **FinRED** | 32,670 examples (27.6k train / 5.1k test) | Financial news + earnings call transcripts | Relation extraction + classification | Founder, CEO, employer, industry, manufacturer, etc. | Multi-task instruction tuning for financial relationship types | Multiple |

---

## Part 4 — SEC Filings & Financial Reports

| Dataset | Size | Coverage | Source(s) | Key Features | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **PleIAs/SEC** | 245,211 docs / 21.2 GB / 7.2B+ words | 1993–2024 (32 years) | SEC EDGAR Form 10-K | Parquet by year; EDGAR-Crawler extraction | Most comprehensive public SEC corpus; CC0 public domain | CC0-1.0 |
| **EDGAR-Corpus** | 40.7 GB / 220,375 annual reports | 1993–2020 | SEC EDGAR annual filings | 15+ structured sections; year-specific configs; billions of tokens | Large-scale pretraining corpus for financial document understanding | Apache 2.0 |
| **Financial Reports SEC (JanosAudran)** | 10M–100M rows (multiple subsets) | 1993–2020 | SEC EDGAR 10-K filings | Future returns data (1d, 5d, 30d windows); train/val/test splits | Masked LM + text classification; includes market reaction labels | Apache 2.0 |
| **S&P 500 EDGAR 10-K** | 6,282 docs / 964 MB | 2010–2022 | SEC EDGAR Form 10-K | All 15 10-K items; future returns at 12 intervals (1–252 days); CIK + SIC metadata | Removes survivorship bias (includes delisted companies); rich temporal return data | MIT |
| **TeraflopAI/SEC-EDGAR** | 43B clean tokens | Full EDGAR history | SEC EDGAR (HTML/XML parsed) | Accession number, filing date, documents, filer info; LLM-ready format | Largest clean SEC token corpus; comprehensive parsing | Varies |
| **Financial Reports SEC (khaihernlow)** | ~1.9M sentences | 1993–2020 | SEC EDGAR 10-K filings | Sentence-level segmentation; sentiment labels from market reaction around filing dates | Sentence-level sentiment tied to actual stock market response | Varies |

---

## Part 5 — Financial News

| Dataset | Size | Time Period | Source(s) | Task Type | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **Reuters Financial News** | 105,359 articles / 122 MB | 2006–2013 | Reuters financial newswire | News classification, summarization, trend analysis | 8-year historical Reuters coverage with summaries | Apache 2.0 |
| **Reuters-21578** | Classic benchmark | 1987 | Reuters newswire | Text categorization | Foundational text categorization research benchmark | Varies |
| **Financial News Multisource** | 57.1M+ rows | 2006–2013+ | Bloomberg, Reuters + 24 other datasets | Sentiment, classification | Largest multi-source aggregation: Bloomberg (2006–2013) + Reuters (2007–2013) + 24 sources | Varies |
| **FNSPID** | 29.7M stock prices + 15.7M time-aligned news records | 1999–2023 | Yahoo Finance API (prices); NASDAQ, Bloomberg, Reuters, Benzinga (news) | News-informed time-series forecasting | Combines quantitative price signals with qualitative news at scale; 4,775 S&P 500 companies | Varies |

---

## Part 6 — Earnings Call Transcripts

| Dataset | Size | Coverage | Source(s) | Key Features | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **S&P 500 Earnings Transcripts (kurry)** | 33,362 transcripts / 1.82 GB | 2005–2025 (20 years) | S&P 500 + large-caps (685 companies, all 11 GICS sectors) | Ticker, company name, date metadata | Most comprehensive earnings transcript dataset; 20-year S&P 500 coverage | MIT |
| **Earnings Call Transcripts (IBM)** | 188 transcripts + 11,970 stock prices + 1,196 sector index values | — | IBM earnings calls | Integrated price + sector context | Multimodal: text + associated stock prices + sector indices | Varies |
| **FINOS Earnings Call Transcript** | 153 segmented audio/text pairs | Alphabet Q1 2025 (example) | Public earnings calls | Timestamps, duration, transcription quality metrics (Voxtral model) | Audio + text modality; ASR + speaker diarization for financial analysis | Varies |
| **Lamini Earnings Calls QA** | — | — | Earnings call transcripts | Structured QA pairs | QA format for earnings performance analysis | Varies |

---

## Part 7 — Financial Time-Series & Market Data

| Dataset | Size | Coverage | Source(s) | Features | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **CryptoData** | 15.2 MB | Crypto markets | Crypto exchanges | OHLCV, RSI, SMA, EMA, close price + volume sequences | Multiple time-series formats for crypto price forecasting | Apache 2.0 |
| **Trading Dataset V2** | 157,016 rows / 47 MB | 2023–2025 | Market data | EMA20/50, Bollinger Bands, MACD, RSI, CCI, Stochastic; instruction-tuning format | Binary Buy/Sell signal classification; LLM-ready instruction format | MIT |
| **FinGPT Forecaster DOW30** | 1,530 rows (1.23k train / 300 test) | May 2023 – Apr 2024 (weekly) | DOW 30 stocks | Analyst prompts with news, financials, market context; weekly movement labels | Stock price movement forecasting with structured analyst report context | Varies |
| **S&P 500 News Time-Series** | 4,589 articles / 17.6 MB | — | Kaggle financial news (~469 S&P 500 companies) | Article ID, ticker, company name, title, body, publish date | Article text aligned to timestamps for time-series text analysis | MIT |
| **Stock Market Tweets** | — | Real-time | Twitter/X | Social sentiment signals | Real-time social sentiment for market signal analysis | Varies |

---

## Part 8 — Instruction Tuning & Alignment

| Dataset | Size | Source(s) | Generation Method | Quality Methods | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **Finance-Instruct-500k** | 518,185 rows / 580 MB | 37 financial + general sources (FinanceBench, FPB, FinRED, BAAI, etc.) | Multi-source compilation | Deduplication (60k+ duplicates removed), cleaned, preprocessed; 7 task categories | Largest comprehensive financial instruction dataset; EN + ZH; covers QA, reasoning, sentiment, NER, RAG | Apache 2.0 |
| **Finance-Alpaca** | 68,912 examples / 42.9 MB | Stanford Alpaca + FiQA + 1,300 GPT-3.5 pairs | GPT-3.5 synthetic + existing datasets | Community-contributed curation | Stocks, personal finance, taxes, loans, options, crypto, real estate; 35+ fine-tuned models | MIT |
| **AdaptLLM Finance-Tasks** | 23,340 rows / 30.8 MB | ConvFinQA, FPB, FiQA_SA, Headlines, NER | Templated evaluation format | 5 standardized subsets | Domain-specific model evaluation; zero-shot financial classification + QA | Varies |
| **FinMA / PIXIU FIT** | 136k+ instruction samples | Multiple financial NLP datasets | Multi-task instruction synthesis | Covers NLP tasks + prediction tasks | Trained ChanceFocus/finma-7b models; purpose-built for financial task diversity | Varies |

---

## Part 9 — Risk, Credit & Compliance

| Dataset | Size | Source(s) | Task Type | Key Features | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **FinBench** | 332k instances (209k train / 23k val / 99k test) / 729 MB | 10 datasets across 3 risk types | Tabular + text classification | Credit card default (2), loan default (3), credit fraud (2), customer churn (3); LLM-generated profiles | Standardized risk prediction benchmark; multimodal (tables + profiles); profile tuning | CC-BY-NC-4.0 |
| **Gretel Financial Risk Analysis** | 1,034 samples (827 train / 207 test) | 14,306 SEC filings (2023–2024; 10-K/10-Q/8-K) | Risk extraction + classification | Differential privacy (ε=8); 10 risk categories; severity levels HIGH/MEDIUM/LOW/NONE | Privacy-preserving synthetic risk analysis; Phi-3 fine-tuned | Apache 2.0 |
| **Credit Scoring Training Dataset** | — | Credit bureau / financial records | Credit scoring classification | — | Credit risk scoring | Varies |
| **Home Loan Approval Dataset** | — | Loan application records | Loan approval prediction | Creditworthiness metrics | Mortgage/loan approval prediction | Varies |

---

## Part 10 — Multimodal & Document Understanding

| Dataset | Size | Source(s) | Modality | Task Type | Specialty / Purpose | License |
|---|---|---|---|---|---|---|
| **Sujet-Finance-Vision-10k** | 9,819 document images | Invoices, budget sheets, financial reports | Images | Vision-language model training | Financial document layout understanding; image-based financial documents | Varies |
| **PleIAs/AMF-PDF** | 633,244 docs / 321 GB | French Authority for Financial Markets (AMF) | PDF + metadata | Multimodal document analysis | Largest financial document collection; French regulatory corpus | Varies |
| **FC-AMF-OCR** | Text-rich docs with OCR annotations | French AMF documents | PDF + JSON-GZ OCR | OCR + document understanding | Multi-page French financial document OCR | Varies |
| **FinMME** | — | Financial charts, tables, reports | Text + tables + charts | Multimodal reasoning | Visual financial data understanding: charts, tables, mixed-modality reasoning | Varies |

---

## Part 11 — DeFi & Smart Contract Security

| Dataset | Size | Source(s) | Task Type | Specialty / Purpose | License |
|---|---|---|---|---|---|
| **Smart Contract Vulnerability Dataset** | 2,000 entries | DeFi protocols | Vulnerability classification | 15 DeFi vulnerability categories; security analysis | Varies |
| **Solidity DeFi Vulnerabilities** | 270 examples | DeFi attack incidents | Vulnerability detection | Attack scenarios, test cases, historical loss data | Varies |
| **Verified Smart Contracts (andstor)** | 100k–200k contracts | Etherscan verified contracts | Code analysis / pretraining | Large-scale verified Solidity code | Varies |
| **DISL Dataset** | 514,506 contracts | Real-world Ethereum blockchain | Code analysis | Decomposed real-world contracts; no duplication | Varies |

---

## Part 12 — Benchmarks & Evaluation

| Dataset | Size | Coverage | Task Types | Specialty / Purpose | License |
|---|---|---|---|---|---|
| **FLUE-FiQA** | 17,110 rows / 48.9 MB | FiQA 2018 challenge | Sentiment + QA (train/val/test splits) | Part of FLUE benchmark (5 financial datasets); corpus + queries + relevance judgments | CC-BY-3.0 |
| **AdaptLLM Finance-Tasks** | 23,340 rows | ConvFinQA, FPB, FiQA_SA, Headlines, NER | Multi-task NLP evaluation | Comprehensive finance NLP benchmark for domain adaptation research | Varies |
| **Open FinLLM Leaderboard** | Dynamic | Information extraction, sentiment, credit scoring, forecasting | Zero-shot evaluation | Real-world financial AI readiness; continuously updated; Linux Foundation partnership | Open |

---

## Quick Selection Guide

| Goal | Recommended Datasets |
|---|---|
| Financial QA / numerical reasoning | FinQA → TAT-QA → FinanceBench → ConvFinQA |
| Financial sentiment analysis | Financial PhraseBank → FinGPT Sentiment → Twitter Financial News |
| Financial NER | FiNER-139 → FiNER-ORD → FinRED |
| SEC filings / document pretraining | PleIAs/SEC → EDGAR-Corpus → TeraflopAI/SEC-EDGAR |
| Financial news corpus | Reuters Financial News → Financial News Multisource → FNSPID |
| Earnings call analysis | S&P 500 Earnings Transcripts (kurry) → IBM Earnings Call |
| Financial instruction tuning (SFT) | Finance-Instruct-500k → Finance-Alpaca → FinMA FIT |
| Financial reasoning (CoT) | FinCoT → Fino1 Reasoning Path → FinQA |
| Time-series / market data | FNSPID → Trading Dataset V2 → FinGPT Forecaster DOW30 |
| Risk / credit scoring | FinBench → Gretel Financial Risk |
| Financial document vision | Sujet-Finance-Vision-10k → FinMME → PleIAs/AMF-PDF |
| DeFi / smart contracts | DISL → Verified Smart Contracts → Solidity DeFi Vulnerabilities |
| Evaluation / benchmarking | Open FinLLM Leaderboard → FLUE-FiQA → AdaptLLM Finance-Tasks |
| Financial RAG evaluation | FinDER → FinanceBench |
