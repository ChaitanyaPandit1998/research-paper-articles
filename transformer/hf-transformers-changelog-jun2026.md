# HuggingFace Transformers — Recent Changes (Jun 2–6, 2026)

Source: `git log` of `huggingface/transformers`, last 50 commits as of 2026-06-28.
Branch is up to date with upstream main.

---

## New Models

### Cosmos3 Reasoner (NVIDIA) — `f677e3a6`
- Adds `Cosmos3ReasonerForConditionalGeneration` — NVIDIA's Cosmos3 model for conditional generation tasks.
- Went through several naming iterations during review (`cosmos3` → `cosmos3_reasoner` → final class name).
- Includes a dedicated `Cosmos3Processor` and is registered in auto-mappings.
- PR #46146, authored by MaciejBalaNV (NVIDIA).

### Sapiens2 — `e820678256`
- Adds Meta's Sapiens2 human-body foundation model with support for **five distinct task heads** on a shared ViT backbone:
  - Semantic segmentation
  - Surface normal estimation
  - Pointmap (3D) estimation
  - Matting (foreground extraction, including composite output)
  - Depth estimation
- Architecture details:
  - Uses `LlamaRMSNorm` for normalization (re-used from Llama rather than reimplementing).
  - Per-layer KV group handling (`num_key_value_heads_per_layer`).
  - Inherits `DINOv3RopePositionEmbedding` for RoPE.
  - Scale head output size controlled via `head_scale_final_input_size` in config.
  - A `head_config` dict in the main config routes task-specific head parameters — cleaner than stuffing everything in a flat config.
- Image processing handles masks preprocessing, crop-and-resize (batched), and flip augmentation (`flip_pairs`).
- Post-processing is batched for performance; pose bboxes returned in standardised format.
- PR #45919.

### Gemma4 Unified + Gemma4 Unified Assistant — `1423d22f7a`
- Commit message "who needs encoders?" signals a key architectural choice: Gemma4 Unified is a **decoder-only multimodal model** — vision tokens are handled inline without a separate vision encoder tower.
- Adds two new namespaces:
  - `gemma4_unified` — the base unified model
  - `gemma4_unified_assistant` — fine-tuned assistant variant
- Includes image, video, and feature extraction processors for each.
- A conversion script (`convert_gemma4_unified_weights.py`) is included to load from the original checkpoint format.
- PR #46385, Google authorship (Douglas Reid, Sara Smoot, Pablo Montalvo-Leroux + HF team).

---

## Major Features & Improvements

### DeepGEMM BF16 + Mixed FP8/FP4 + MegaMoE — `fa6c8308e2`
A large, high-impact PR (#45634) touching DeepSeek V4 inference performance deeply.

**DeepGEMM kernel extensions:**
- Adds BF16 support to DeepGEMM (previously FP8-only).
- Mixed FP8/FP4 grouped linear kernels — allows different precision per expert in MoE layers.
- Dynamic TP/EP plan swap: the tensor-parallel and expert-parallel plan can be swapped at runtime without a full model reload.
- Retrieves alignment requirements from DeepGEMM and caches them to avoid repeated lookups.

**MegaMoE support:**
- Introduces `MegaMoE` — a fused MoE kernel variant for very large expert counts.
- Custom EP/TP dispatch plan for MegaMoE processed after weight loading.
- `IndexerScorer` abstraction replaces the previous router/expert pre-post processing pair.

**FP8 correctness fixes:**
- `WeightConverter.reverse_transform` now appends a `(?=\.|$)` token-boundary lookahead to reversed source patterns, fixing substring-match bugs where FP8 scale companions (`mlp.experts.gate_up_proj_scale_inv`) were incorrectly routed into shape-only ops (Chunk, SplitModulelist) on save.
- Introduces `ConversionOps.supports_round_trip` opt-in flag — only `Fp8Quantize`/`Fp8Dequantize` opt in for now; other quant backends stay opt-out until audited.
- Gate computation through softmax now stays in FP32 (dropped the `.to(chunk_gate.dtype)` cast in DeepSeek V4 compressors HCA/Indexer/CSA).
- `attention_factor=1.0` is only set when `rope_type="yarn"`, suppressing spurious "Unrecognized keys in rope_parameters" warnings on the second `post_init` pass.

**DTensor support:** grouped linear ops now work with PyTorch DTensor for distributed training/inference.
**Bias support** in grouped linear layers was also fixed.

### Quantization for Small Models (Gemma-specific) — `a921b4d886`
- Adds `GemmaQuantizer` (`src/transformers/quantizers/quantizer_gemma.py`) and `GemmaQuantConfig` in `utils/quantization_config.py` — a quantization path tuned for Gemma-family models.
- Integrates with `gemma_quant.py` in the integrations module.
- Updates the auto-quantizer router so Gemma configs resolve to the new quantizer automatically.
- Also touches `modeling_gguf_pytorch_utils.py` and `cli/serving/model_manager.py` — suggests this is wired into the serving CLI.
- PR #46449, Google/HF collaboration (Sara Smoot, Marc Sun, Phil Culliton, Ryan Mullins).

### Modular: `no_inherit_decorators` — `bcd12e3679`
- Adds a `no_inherit_decorators` sentinel to the modular converter framework.
- Problem it solves: kernel decorators (e.g. `@torch.compile`, `@custom_kernel`) applied to a base class method were being incorrectly propagated to subclasses generated via `modular_*.py` — causing compile failures or wrong kernel dispatch on child models.
- The sentinel marks a method as "do not carry parent decorators into subclass"; the converter checks for it before propagating.
- Simultaneously fixes incorrect RoPE-related inheritance chains across multiple models — these had silently inherited wrong RoPE implementations through decorator propagation.
- Includes a regression test.
- PR #46440.

---

## Bug Fixes

### DSV4 dequant + TP/EP — `50eb20a24f`
- Fixes dequantisation correctness in DeepSeek V4 when tensor-parallel and expert-parallel are both active.
- Related to the larger DeepGEMM refactor but landed as a separate targeted fix.
- PR #46378.

### fbgemm_fp8 — Device Alignment — `dcf0c7c8c3`
- `fbgemm_fp8` was not keeping the current CUDA device aligned with the input tensor's device, causing silent cross-device errors.
- Fix: set the current device to match the input tensor before calling into fbgemm_fp8 kernels.
- PR #46403.

### Fix flip_back Graph Break — `18bd9d1042`
- `flip_back` (used in pose estimation postprocessing) was causing a `torch.compile` graph break.
- PR #46344.

### convert_tokens_to_ids Performance Regression — `d3f05911ab`
- Slow (`PreTrainedTokenizer`) tokenizers were resolving added-vocab tokens through `added_tokens_encoder` (a property), which **rebuilds and re-sorts the full mapping on every access** — called twice per token.
- Made `convert_tokens_to_ids` O(T × N × log N) for N added tokens, a regression from the v5 tokenizer refactor (#40936).
- Fix: read from `_added_tokens_encoder` (the maintained cache) instead, restoring O(1) per-lookup behaviour — consistent with how every other method in the file already worked.
- PR #46315 / #46323.

### Qwen VL Model Parallel Bug — `b07d99be86`
- Fixes a model-parallel bug in the Qwen series VL (vision-language) models.
- PR #46316.

### Fix flip_back Graph Break — `18bd9d1042`
- `flip_back` in pose estimation caused a `torch.compile` graph break.
- PR #46344.

### Path Traversal in Bark Voice Preset Save — `1b8ec344fb`
- **Security fix (CWE-22 path traversal).** `BarkProcessor.save_pretrained` wrote speaker embedding `.npy` files using untrusted keys from `speaker_embeddings_path.json` verbatim as filenames via `os.path.join`, allowing a malicious repo to write files outside the target directory (e.g. key `../../foo` → file at `../../foo_<inner>.npy`).
- Fix: lexical containment check using `os.path.abspath` / `os.path.commonpath` before `np.save`. Symlink-safe (does not call `Path.resolve()`, which would break legitimate HF cache symlink layouts like `snapshots/<rev> → blobs/`).
- Also guards the read path: `_load_voice_preset` now rejects `../`-style values in per-prompt path entries before passing to `cached_file`.
- Allows legitimately nested presets (e.g. `v2/en_speaker_0`) while blocking escapes.
- PR #46237.

### DeepSeek V4 XPU Test Extension — `effde20942`
- Extends the DeepSeek V4 test suite to run on Intel XPU (via `require_xpu` decorator).
- Makes `require_cuda_capability_at_least` **composable** (can be combined with other hardware markers) rather than mutually exclusive.
- PR #46366, Intel authorship.

### Gemma4 Typos + Unified Config Bugs — `41200c1ea0` / `ece3b9a353`
- Small docs typo sweep for Gemma4 (`#46351`).
- `gemma4_unified` conversion script and config bugs fixed (`#46398`): catches config initialization ordering issue that caused incorrect default values on first load.

### CI / Developer Tooling — Various
- Fix Slack report job to use AWS runner + authenticated GitHub API calls (`b1ac534932`, `94246e689c`).
- Add CI Grafana dashboard link workflow to post on PRs (`77ed250a36`).
- Improve CI dashboard comment format (`2fc531657b`).
- DeepSpeed Docker fix (`1316cd72c0`).
- CLIP model conversion fix (`bb1cfda217`).
- Fix missing f-string prefixes in error messages (`e85760e223`).
- `_is_package_available` was falsely reporting packages as available when they had no version — fixed (`18d845c9ad`).
- `torch<=2.7` compatibility fix (`10ed83ffb9`).
- Raise `tqdm` minimum to 4.60 to match `tqdm.contrib.logging` import (`9c7c911c87`).
- Pass `library_name`/`version` to Hub calls via a shared `HfApi` instance (`595721c44c`).
- Fix incorrect attribute mapping in GLM MoE DSA Config (`31637182fb`).
- Fix transitive relative imports when loading models from a local directory (`908f67e864`).

---

## Documentation
- Padding-free training docs (`eca43b8010`, #46333).
- XPU continuous batching docs (`15bb519bd4`, #46334).
- Romanian translations: pipeline tutorial, tokenizer summary, image/video processors, contributing guide, modular transformers, multimodal processing, add_new_model, testing (`032db9c8d6`, `5a55007830`, `ad0323529a`, `a46a732528`).
- Remove sparsity from compressed-tensors docs (`03dbff6cce`).
- Update `num_items_in_batch` docs for causal LMs (`9aed235868`).
- Compressed tensors minimum version bump (`4ae05b0fba`).

---

## Shape / Scope Summary

| Category | Count (approx) |
|---|---|
| New models | 3 (Cosmos3 Reasoner, Sapiens2, Gemma4 Unified) |
| Major features | 3 (DeepGEMM/MegaMoE, Gemma quant, Modular no_inherit_decorators) |
| Bug fixes | 10+ |
| Security fixes | 1 (Bark path traversal) |
| CI / tooling | 8+ |
| Docs | 7+ |

The most architecturally significant changes in this window are **DeepGEMM BF16 + MegaMoE** (advances FP8/FP4 mixed precision for large MoE inference), **Sapiens2** (multi-task human body analysis with five heads on a shared backbone), and **Gemma4 Unified** (encoder-free decoder-only multimodal design).
