# Review: `docs/research/02-eagle4/{overview,architecture,build-and-kernels}.md` (phase 1, EAGLE-4 architecture)

Reviewer role: P1 speed/inference and P2 heads.

Sources:
- `~/research-src/EAGLE-4` @ `7b7b5a5`. The paper `EAGLE4_Paper.pdf` (12 pp.) was extracted with `pdftotext`; page numbers are PDF pages.
- HF `EAGLE-Corp/Qwen3.8-27B-EAGLE4` @ `f58a598`: API, `config.json` and the safetensors header (by range request). No weights were downloaded. The README's link `EAGLE-corp/EAGLE4-Qwen3.8-27B` returns a 307 to this canonical name.
- HF `Qwen/Qwen3.8-27B` `config.json`.
- `~/vllm`, tag `v0.31.0` (`git show` / `git grep`) and HEAD `v0.32.1rc0-68-g4bd75e952a`.
- Our repo.

No GPU was used. The recomputation script is `phase1-eagle4-architecture-calc.py`; its output is `phase1-eagle4-architecture-calc.out`. Both are in this directory.

Labels: **measured** means read from a file or the API. **paper-reported** means taken from the PDF. **estimated** means arithmetic or static reading of code.

## Verdict

| File | Verdict | Checked | Correct | Wrong | Unsupported |
|---|---|---|---|---|---|
| `overview.md` | **PASS WITH CORRECTIONS** | 9 | 7 | 1 | 1 |
| `architecture.md` | **PASS WITH CORRECTIONS** | 10 | 9 (2 with context fixes) | 1 | 0 |
| `build-and-kernels.md` | **PASS WITH CORRECTIONS** | 11 | 6 | 5 | 0 |

**overview.md and architecture.md:**
- Every paper number I checked matches its page, and Table 2 matches cell for cell.
- The HF config and parameter breakdown match the header exactly.
- The licence quotes and line ranges are right.
- The traffic estimate reproduces: the efficient head cuts per-step draft traffic by 70.7%.
- The architecture doc's 3090 section used the target config's fp32 GDN state. Our stack runs that state in fp16, so the row cost is half what the doc gives.

**build-and-kernels.md has the most errors**, but none of them reverses its conclusions:
- The HOOK_MAP list is wrong.
- "MRV2 risks our patches" is wrong: MRV2 is already the vLLM 0.31.0 default, and our patches edit it.
- The L2-prefetch gate is wrong: it is SGLang-only.
- The patch count is wrong.
- The loader mechanism for E-20 is wrong, though its conclusion holds.

**The biggest gap is E-20** ("serve the drafter as a plain EAGLE-3 chain on 0.31.0"). It is missing two steps:
- **Off-by-one aux layer ids.** Without the fix, the draft is silently fed the wrong layers.
- **Embedding sharing.** Without it, the draft costs 6.18 GB instead of 3.64 GB.

## Checked claims

### overview.md

| # | Claim | Doc location | Evidence | Result | Note |
|---|---|---|---|---|---|
| O1 | Table 2 speedups and throughputs | §Claims, Table 2 | paper p. 9, Table 2 | CORRECT | Matches exactly |
| O2 | Nemotron Table 3: 647 (2.74×), 33,480 (1.69×), 563 (2.40×), 29,633 (1.51×) | l. 102ff | paper p. 9 | CORRECT | The paper names the model "Nemotron-3.5-Super-VL-120B-A12B"; the doc drops "VL" (cosmetic, not edited) |
| O3 | Efficient head: top-1 within 128 in 98.2% of steps (SVD init 91.5%); SVD init costs 0.48 tokens; draft time 9.1 → 4.8 ms | l. 140-142 | paper p. 6 | CORRECT | |
| O4 | Phone: 22.8 → 82.6 tok/s (3.62×); 356 → 166 mJ/token; 8.7 → 3.9 ms; +0.27 GB | l. 147-152 | paper p. 10, Table 6 and text | CORRECT, note added | The same page also says 0.26 GB. The paper is inconsistent with itself, not the doc |
| O5 | Flat 5.57–6.04 ms round at 32K–250K context "credited to SWA (§3.3.1, p. 6)" | l. 128 | paper p. 10 (Table 7 text gives no cause); p. 6 §3.3 (motivates SWA) | UNSUPPORTED | This is the doc's inference. Marked inline |
| O6 | The release engine lacks `suspend_drafting`/`verify_budget`, "so native-OFF batch sizes do not run" | l. 198 | `eagle_harness/vllm/speculator.py:69`; `vllm/config.py:242-245` ("the 0.1.0 release engine"); `vllm/batch_policy.py:16-17` | WRONG (second half) | Native-OFF still runs through `SpeculativeConfig.native_decode_batch_sizes`. Only per-batch trimming and suspension are lost |
| O7 | v0.1.0 = `7543671` at 2026-10-20 12:11 −0500; HEAD is 17 commits later | l. 194 | `git rev-list --count v0.1.0..HEAD` = 17; author date 12:11:29 | CORRECT | The commit date is 12:33:57; the doc uses the author date |
| O8 | Licence §1(e), §3(a)–(b), §6(a) quotes and line ranges | §Licence | `LICENSE:73-75, 122-131, 256-262` | CORRECT | See missed item M7 for §9(b) |
| O9 | HF repo metadata: not gated, licence other/eagle-4, 4 files | §What EAGLE-4 is | HF API (measured 2026-10-21) | CORRECT | The README's lowercase alias redirects (307) |

### architecture.md

| # | Claim | Doc location | Evidence | Result | Note |
|---|---|---|---|---|---|
| A1 | Draft config: 1 layer; sliding_attention W=4096; 40/40 heads × 128; rope 1e7; intermediate 16384; aux [3,31,59]; `fc_norm`/`norm_output`; low-rank head r=128, K=128; vocab 248320 | §1 table | HF `config.json`. It is byte-identical to `eagle_harness/core/profiles/mi355x-qwen3.8-27b-tp1/draft_config.json` | CORRECT | |
| A2 | Parameter breakdown: 3089.1M total, embed 1271.4M, fc 78.6M, factors 32.4M, layer 435.2M = the paper's "0.44B" | §1, l. 28-34 | safetensors header (3,089,075,200 params); paper p. 3, Table 1 | CORRECT | Recomputed in `.out` |
| A3 | Quote at `models.py:146-147`, "the embedding is the target's (vLLM shares it)" | l. 35 | `eagle_harness/vllm/models.py:146-147` | CORRECT, context added | Stock vLLM shares only when `torch.equal` holds (`v1/spec_decode/llm_base_proposer.py:1416-1560`; `v1/worker/gpu/spec_decode/eagle/utils.py:15-27`). It cannot hold against our int8 embedding |
| A4 | Resident draft ≈ 1.82B params ≈ 3.64 GB BF16 | l. 36 | calc `.out`: 3.635 GB | CORRECT | Only if the embedding is actually shared (see A3) |
| A5 | ≈1.02 GB per step, a ≈1.1 ms floor; efficient head cuts traffic ~71% | §2, §7 l. 242 | calc `.out`: 1.024 GB, 1.094 ms; full head 3.497 GB; cut 70.7% | CORRECT | |
| A6 | Depth-6 round ≈ 6.2 GB, ≈ 6.6 ms | l. 243 | calc `.out`: 6.30 GB, 6.73 ms | CORRECT | Within rounding (the doc omits the window KV on prefill) |
| A7 | MTP drafter "roughly 0.2 GB per step" | l. 244 | ours `single-user/start_qwen.sh:14-16` (215 MB module; 40k-row int4 head) | WRONG | ≈0.32 GB (estimated) |
| A8 | GDN state 144 MiB/row at fp32; width 16 ≈ 2.25 GiB | l. 248-249 | target config `mamba_ssm_dtype: float32`; ours `start_qwen.sh:1036` (`--mamba-ssm-cache-dtype float16`); `gdn_tree_verify.py:245` | CORRECT arithmetic, context fixed | On our stack: 72 MiB/row, ≈1.1–1.2 GiB |
| A9 | Tree verify refuses quantized KV | l. 252 | `vllm/tree_attention.py:69-70` | CORRECT | |
| A10 | `CUSTOM` clash with pthfa: "cannot coexist without renaming one" | l. 255 | ours `pthfa/python/pth_flash_attn/plugin.py:13-18, 38-44`; `eagle_harness/vllm/plugin.py:31-67` | CORRECT, remedy incomplete | `CUSTOM` is the only plugin slot, so a rename needs a vLLM enum patch. EAGLE-4 overwrites the slot; pthfa yields |

The following were also spot-checked and are correct:
- `draft_tree.py:35-36` (ROOT_SCORE 1.001, DEPTH_DECAY 0.9999)
- `vllm/config.py:68-69, 104-105` (K=8, depth 12; width 16 for chain, 32 otherwise)
- `quickstart-vllm.md:49-50`
- `depth_stop.py:4-27`
- `tree_drafter.py:172-207`
- `gdn_tree_verify.py:4-26`
- `tree_accept.py:4-22` (the losslessness quote)
- `draft_attention.py:4-13, 39-44`

### build-and-kernels.md

| # | Claim | Doc location | Evidence | Result | Note |
|---|---|---|---|---|---|
| B1 | Our image: vLLM 0.31.0, torch 2.13, cu130, Triton 3.7.1, "50-patch series (`Dockerfile:2-5`)" | l. 24, 118 | ours `Dockerfile:1-5` (no count); `patches/series`: 51 entries; 51 `.patch` files | WRONG (count) | 51 (measured) |
| B2 | HOOK_MAP includes `suspend_drafting`, `verify_budget`, `native_decode_batch_sizes`; `missing_hooks` refuses hosts without them | l. 86 | `vllm/adapter.py:30-71, 74-76`; `vllm/batch_policy.py:4-17, 32-35`; `vllm/config.py:73, 217, 242-245` | WRONG | HOOK_MAP has 11 paths (listed inline). The three named hooks are optional. On v0.31.0, 9 of the 11 are missing |
| B3 | None of the fork's hooks exist in `~/vllm` v0.31.0 or HEAD; `CUSTOM` and `num_speculative_tokens_per_batch_size` do | l. 87-91 | `git grep` at both refs; `v1/attention/backends/registry.py:154`; `config/speculative.py:492` | CORRECT | `BaseSpeculator`/`DraftModelSpeculator` also exist (`v1/worker/gpu/spec_decode/speculator.py:59,103`) |
| B4 | `flashinfer_cutedsl` is gated to the SM100 family | l. 95 | `~/vllm` `model_executor/layers/utils.py:104-111`; `utils/flashinfer.py:194-206` | CORRECT | |
| B5 | `candidate.cu` traps below sm_90; JIT/Triton fallback keeps sm86 working; arch tiers | §2 | `csrc/candidate.cu:200-201`; `kernels/native.py:4-8, 48-60, 74-88`; `kernels/arch.py:11-17, 96-107, 127-145`; `build.py:38-42, 61-63` | CORRECT | `_jit` builds for the visible GPU's sm, so an sm_86 build is possible |
| B6 | No FP8, WGMMA, tcgen05, TMA or cluster use in the kernels | §4 | grep for `fp8\|float8\|e4m3\|wgmma\|tcgen\|tensor_descriptor\|cluster_dims\|tma` in `eagle_harness/`: no hits | CORRECT | |
| B7 | `l2_prefetch.py` is gated by `side_stream_prelaunch()` (`arch.py:160-165`) | l. 58 | `sglang/l2_warm.py:36,149`; `core/config.py:70,122`; `sglang/worker.py:231` | WRONG | SGLang-only, enabled by `l2_warm`. E-23 is a new port |
| B8 | Stock `llama_eagle3.py` reads `fc_norm`, `norm_output` and `layer_types: sliding_attention` | l. 97 | `git show v0.31.0:…/llama_eagle3.py:231-240, 252`; `llama.py:185-219` | CORRECT | |
| B9 | The loader prefixes unknown tensors with `model.`, so `efficient_lm_head*` are unexpected | l. 98 | `llama_eagle3.py:420-470`: `elif "lm_head" not in name: name = "model." + name` | WRONG (mechanism) | The names are not prefixed, because they contain "lm_head". They still fail to load, so the conclusion stands |
| B10 | "MRV2 required … our patches target the V1 runner path" | l. 120 | `~/vllm` `config/vllm.py:700-760` (defaults to True); ours `patches/*` edit `v1/worker/gpu/model_runner.py`, `gpu/spec_decode/dflash2/speculator.py` and `gpu/cudagraph_utils.py`; ours `docs/gotchas.md:189` | WRONG | MRV2 is already in use. This is not a new risk |
| B11 | Tree verify refuses quantized KV; only bf16 KV (`start_qwen.sh:213`) qualifies | l. 121 | `tree_attention.py:69-70`; ours `start_qwen.sh:213, 218-250` | CORRECT | |

## Recomputed estimates

All values below are **estimated** bandwidth floors at 936 GB/s, unless marked otherwise. They come from `phase1-eagle4-architecture-calc.out`.

| Quantity | Doc | Recomputed | Status |
|---|---|---|---|
| Total draft params | 3089.1M | 3089.04M from shapes (header 3,089,075,200 incl. norms, measured) | OK |
| Embedding + lm_head share of params | n/a | 82.3% | OK |
| Resident draft, embedding shared | 3.64 GB | 3.635 GB | OK |
| Resident draft, nothing shared | n/a | **6.178 GB** | New. This is the E-20 default on our stack (see M1) |
| Draft KV per token | 20 KiB | 20,480 B, which is 31.25% of the target's 65,536 B | OK |
| SWA window KV | n/a | 80 MiB (0.084 GB) | OK |
| Per step, efficient head | ≈1.02 GB / 1.1 ms | 1.024 GB / 1.094 ms | OK |
| Per step, full lm_head | n/a | 3.497 GB / 3.74 ms | New. This is E-20's per-step cost |
| Traffic cut | ~71% | 70.7% | OK |
| Depth-6 round | 6.2 GB / 6.6 ms | 6.30 GB / 6.73 ms | OK (rounding) |
| W4A16 draft layer step | ≈0.22 GB + scales, ≈0.4 ms | 0.370 GB incl. factors + window, 0.40 ms | OK |
| Our MTP step | ≈0.2 GB | **≈0.32 GB** (0.215 + 0.108) | Corrected |
| GDN state per verify row | 144 MiB fp32 | 144 MiB fp32 / **72 MiB fp16 (our stack)** | Context corrected |
| Width-16 tree GDN state | 2.25 GiB | 2.25 GiB fp32 / **1.12 GiB fp16**; 1.20 GiB with 17 align-mode slots | Context corrected |

## Missed items

1. **M1. E-20 embedding sharing.**
   - The checkpoint ships its own `embed_tokens` (2.54 GB).
   - Stock vLLM shares the target embedding only if the draft has none, or if the two are `torch.equal`. That test cannot pass against our int8 `weight_packed` embedding, and it may raise when it reads `.weight` (estimated).
   - So E-20 as written loads a 6.18 GB draft.
   - Dropping `embed_tokens` is valid only if it equals the original BF16 target embedding. That is unverified.
2. **M2. E-20 aux-id off-by-one.**
   - Stock `qwen3_next.py:711-724` records the input of layer i.
   - `gpu_model_runner.py:5473-5490` passes `eagle_aux_hidden_state_layer_ids` through unchanged (only DFlash adds +1).
   - The EAGLE-4 plugin adds the +1 itself (`eagle_harness/vllm/config.py:199-206`).
   - E-20 must therefore edit the ids to `[4, 32, 60]`. Otherwise acceptance is measured on the wrong features.
3. **M3. E-20 per-draft overrides.**
   - Without `"kv_cache_dtype":"bfloat16"` and `"attention_backend":"FLASH_ATTN"` in `--speculative-config`, the draft inherits the target's quantized KV (`config/speculative.py:376-384, 425-430`).
4. **M4. E-20 is not a speed path.**
   - It reads the full lm_head: 3.5 GB/step, ≈3.7 ms on a 3090.
   - It is an acceptance baseline only.
5. **M5. SWA semantics differ.**
   - Stock vLLM counts draft tokens inside the 4096 window. EAGLE-4 uses "W context + d chain tokens" (`eagle4_config.py:9-12`).
   - This matters only past 4K context.
6. **M6. Engine-level gaps the docs do not raise.**
   - CUDA-graph conditional nodes (E-22) are untested on our driver and container.
   - The hybrid KV allocator would have to unify a 20 KiB/token MHA draft layer with the target's attention and GDN pages. Its padding cost was not examined.
7. **M7. Licence §9(b) survival clause.**
   - The clause lists "12", but the licence ends at §11 (`LICENSE:319-320, 354`).
   - This is a drafting defect; flag it for E-01, not a blocker.

## Corrections made

All of these are inline `[review: …]` notes; the original text is kept.

**overview.md**
1. l. 128: SWA attribution marked as unsupported inference (O5).
2. l. 152: the paper's 0.26/0.27 GB inconsistency noted (O4).
3. l. 198: native-OFF on the release engine corrected (O6).

**architecture.md**
1. l. 35: the embedding-sharing condition and our int8 embedding added (A3).
2. l. 37: lm_head sharing marked moot on our stack (`has_own_lm_head = True`; target lm_head requantized).
3. l. 244: MTP step corrected from 0.2 to ≈0.32 GB (A7).
4. l. 249: fp16 GDN state on our stack, 72 MiB/row, ≈1.1–1.2 GiB (A8).
5. l. 255: `CUSTOM` is the only plugin slot, so a rename needs a vLLM patch; moot under `PTH_FA` int8 KV (A10).

**build-and-kernels.md**
1. l. 24 and l. 118: 51 patches, not 50 (B1).
2. l. 58: L2-prefetch gate corrected to SGLang `l2_warm` (B7).
3. l. 86: HOOK_MAP contents corrected (B2).
4. l. 98: loader mechanism corrected (B9).
5. l. 120: MRV2 is already the default and already patched (B10).
6. l. 124 (item 6): fp16 GDN state, 72 MiB/row.
7. E-20: per-draft KV dtype and backend overrides.
8. E-23: SGLang-only.
9. A new **Reviewer additions** section covers M1–M6.

## Open questions

- Does the draft's `embed_tokens` equal Qwen3.8-27B's BF16 embedding? This needs a hash of the target shard: a download that was not done here.
- Does vLLM 0.31.0's `torch.equal` sharing path raise or fall through on a `weight_packed` embedding? Settle it with a one-line CPU test once the image is available.
- Is the `torch.equal` comparison in MRV2 `_should_share` reached when `has_own_embed_tokens` is set by `process_eagle_weight` (`models/utils.py:1099-1119`)? Static reading says yes, but nothing was run.
