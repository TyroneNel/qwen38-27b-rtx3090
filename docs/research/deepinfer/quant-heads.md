# Quants, heads and numerics

Lane: quantization and heads (Opus, high effort), verified by the lead. Part of the [DeepInfer expedition](README.md).

Citation forms:
- HyperQwen paths are repo-relative.
- `DeepInfer/...` refers to deepseek-ai/DeepInfer at `774ce24`; `DeepGEMM/...` is its `vendor/DeepGEMM` submodule.
- `DeepSpec/...` refers to deepseek-ai/DeepSpec at `005e03b`.
- `vllm@v0.31.0:...` refers to the pinned vLLM tag. `marlin_template.h` and `marlin_utils.py` are vLLM's Marlin files at that tag.
- **(inferred)** marks arithmetic or reasoning by the lane or the lead.

## Verdict

1. **The part of DeepSeek's low-precision recipe that transfers to sm86 is fine-grained activation scaling, done in int8.** DeepInfer quantizes activations per token × per 32 channels for a block-scaled GEMM (`DeepInfer/deepinfer/model_executor/layers/linear.py:13-24, 150, 165`; `DeepGEMM/docs/scaling-factor-format.md:15-40`). HyperQwen's INT8_ACT uses one scale per token row, and on heavy-tailed activations per-128 group scales cut the activation error 4–7× (CPU check a). vLLM 0.31 can already serve this through the Humming kernels, behind a routing trap (quant-1).
2. **The LSB that `marlin-int8-negative-scales` drops is likely not "extremely rare".** The patch already expects about half of all groups to have a negative scale, and says the clamp of code −8 is extremely rare. If AutoRound picks the scale sign so the extreme weight lands on code −8, as the lane's reconstruction does, then every negative group has a −8, and the fold drops an LSB on its largest weight. A bit-exact remap exists and runs on a kernel HyperQwen already builds (quant-2). The real checkpoint settles it in one CPU run.
3. **HyperQwen's own GPTQ leaves easy accuracy on the table.** Signed amax/8 scales, the existing `--mse-clip` and static-group act-order together cut synthetic output error 55% (quant-3). That feeds the int4 DFlash2 drafter's sampling-acceptance gap.
4. **DeepSeek's other choices don't transfer:** fp8 and power-of-two (UE8M0) scales, fp32 logits and bf16 heads. HyperQwen's quantized heads are a deliberate, measured divergence.

## What DeepSeek's recipe is, and what carries over

| Piece | DeepInfer | On sm86 / HyperQwen |
|---|---|---|
| Dense linears: fp8 e4m3 weights in 32×32 blocks; activations per token × per 32 channels with power-of-two scales | `linear.py:13-24` (granularity 32, `round_sf True`), `:150`, `:165` (`deep_gemm.fp8_fp4_gemm_nt(..., recipe=(1,1,32))`) | No fp8 MMA on sm86. Fine-grained activation scaling transfers in int8 (quant-1). Power-of-two scales cost ~40% more error than fp32 scales (check a). |
| Partial sums promoted per K block | DeepGEMM recipe | Marlin's int8 path already promotes int32 per-group partials by an int16 group scale (`marlin_template.h` ~1127-1157). |
| Heads, embeddings, router and small projections in bf16 | `DeepInfer/deepinfer/model_executor/models/v4_modules/containers.py:116, 122` ("Heads use BF16") | HyperQwen quantizes the heads on purpose, backed by KL and perplexity data. Its small projections are already bf16. |
| lm_head as a bf16 GEMM into fp32 logits | `layers/sampler_utils.py:20-26` | Worth very little (check c). Rejected. |
| RMSNorm fused with the activation quant cast | `models/decoder_layer.py:43-44`, `layers/tilelang_ops/norm.py:65-130` | quant-6 (small). |
| Norm + RoPE + Hadamard + cast + KV store, fused | `layers/tilelang_ops/fused_transform.py:37-106, 189` | KV side in [attention-kv.md](attention-kv.md); the Hadamard-before-quant idea for linears is quant-4. |

## CPU checks

All run in under a minute and 1.5 GB RSS. They need torch, which on the research box lives in the user site-packages, so run them from `docs/research/deepinfer/cpu-checks/` with `python3 -E -P <script>` (`-I` would hide the user site). The lead re-ran all five from the repo copies and the outputs match the lane's.

**(a) Activation quant error** (`quant_a_act_quant.py`, `quant_a2_gauss.py`). Synthetic activations, 256 tokens. Metric: relative error of X·W with the weight-quant error excluded, as `bench/act_calib.py` measures it.

| Scheme | Gaussian, K=5120 | t (df=8) | in_proj-like (t5, 6 outlier channels ×40) | down_proj-like, K=17408 (t2.5, 16 outliers ×30) |
|---|---|---|---|---|
| per-token int8 (**today**) | 0.88% | 2.38% | 10.2% | 13.7% |
| per-token × per-128 int8, fp32 scale | 0.65% | 1.10% | 2.08% | 1.96% |
| per-token × per-128 int8, pow2 scale | – | – | 2.96% | 2.81% |
| per-token × per-32 int8, pow2 scale | – | – | 1.57% | 1.54% |
| per-token × per-32 fp8 e4m3, pow2 (DeepSeek's exact recipe) | – | – | 2.53% | 2.76% |
| Hadamard-128 + per-token int8 | 0.87% | 0.90% | 1.26% | 1.75% |
| Hadamard-128 + per-128 int8 | – | – | 0.46% | 0.47% |

The Gaussian row matches HyperQwen's own 0.9–1.05% from `bench/test_marlin_int8_asym.py` (`patches/marlin-int8-asym-zp.patch`, header lines 13-16), which anchors the setup. Real layers' absolute numbers depend on how heavy-tailed real activations are; the ranking doesn't.

**(b) The sign fold** (`quant_b_signfold.py`). AutoRound's `quant_tensor_sym` is **reconstructed from memory (inferred; auto_round isn't installed here)** as scale = ±max|w|/8, with the sign chosen so the extreme element maps to code −8. Under that model:
- 49.3–49.5% of groups have a negative scale, and 100% of those use code −8;
- 1.1–1.7 clamped elements per negative group, always the largest-magnitude weight;
- HyperQwen's fold (clamp +8 → +7) adds **+4.8 to +8.6% weight-error energy** on every int8-activation layer.

In a negative group, code′ = 15 − code with zero point 7 reproduces the weights bit-exactly (max |diff| 0). The int16 4,096-level scale requant (`marlin_utils.py:544-548`) is negligible when max/min |scale| ≤ 64 (mean 0.09%) and only reaches 0.9% mean (11% worst) at a ratio of 1,024.

**(c) bf16 vs fp32 logits** (`quant_c_logits_bf16.py`). Vocabulary 248,320, synthetic Qwen-like logits with top logits 12–40. At T=1, KL(fp32‖bf16) is 1e-4 to 6e-4 for top logits ≤ 28 and 2e-3 at 40; at T=0.6, 0.4e-3 to 1.4e-3; zero argmax flips. For comparison, the shipped int4 GPTQ head's KL is 0.0024–0.0029 against a bf16-logit reference (`docs/third-party-checkpoints.md:115-116`). `drafter/gptq_lm_head.py:53` computes that reference in bf16, so it hides this term.

**(d) GPTQ variants** (`quant_d_gptq.py`, which imports `drafter/gptq_utils.py` from the repo). Synthetic W [256, 5120] with t4 tails; Hessian from heavy-tailed activations with 12 outlier channels. Metric: tr(E H Eᵀ) / tr(W H Wᵀ), two seeds.

| Variant | Output error vs shipped |
|---|---|
| shipped `gptq_quantize` (amax/7, natural order) | 0 (8.8e-3 to 9.3e-3) |
| + `--mse-clip` (in the code, unused by the shipped build, `drafter/README.md:51`) | −27 to −30% |
| signed amax/8 scale (AutoRound's 16th level) | −25% |
| signed /8 + MSE clip | −34 to −36% |
| amax/7 + act-order with static groups | −32 to −35% |
| **signed /8 + MSE clip + act-order (static groups)** | **−55 / −56%** |
| per-channel int4 + act-order | +139 to +146% |
| RTN g128 | +310 to +336% |

The synthetic Hessian has strong outlier channels, which flatters act-order. Expect less on real Hessians.

**Real-checkpoint check** (`quant_real_ckpt_check.py`, for the GPU box; CPU only, ~2 GB). It reports per tensor the share of negative groups, how many use code −8, clamped elements and fold error energy. `quant_mk_fake_ckpt.py` writes a one-layer `fake_ckpt/` into the current directory for a self-test; don't commit it. On the fake checkpoint it reports 49.9% negative groups, 100% using −8, and fold error energy 0.06% of weight energy (int4 RTN is typically 1–2%).

## Candidates (ranked)

| ID | Title | Modes | Gain | Effort | Confidence |
|---|---|---|---|---|---|
| quant-1 | Per-token × per-128 int8 activation scales | A; prefill in B–E | A 942 → up to 1,042 tok/s; B–E prefill +27–30% (if quality allows wider INT8_ACT) | M (route A), L (route B) | Medium |
| quant-2 | Lossless negative-scale fold | A, and INT8_ACT prefill | quality only; removes an error on every INT8_ACT layer | S (fix A), M (fix B) | Medium (hinges on AutoRound's sign rule) |
| quant-3 | Better GPTQ for HyperQwen's own int4 requants | B, C, E (DFlash2); D (MTP); fast variant (lm_head) | DFlash2 +2–3% at default sampling; lm_head PPL +0.6% → ~+0.3% | S | Medium |
| quant-4 | Block-Hadamard-128 rotation of linear inputs | as quant-1 | as quant-1, on the unchanged Marlin kernel | L | Medium |
| quant-5 | Batch-mode lm_head as W4A8 | A | +2–3% | S–M | Low-medium |
| quant-6 | Fuse the int8 activation quant into its producer | prefill | ~0.5% of a 4k prefill | M | Medium |
| quant-7 | int4 embedding where `EMBED_UVA` can't be used | D, E (pool) | pool +8.4% | S | High |

### quant-1: per-token × per-128 int8 activation scales

**Measured (2026-10-21):** INFO (quant-1-calib). The quality gate passes: per-128 cuts down_proj error 5.0× and GDN in_proj 5.7× on every layer. Route A is slow: Humming int8 g128 runs at 0.26–0.41× of Marlin W4A8-int8 at M ≤ 64 (+19% only at M ≥ 256), and vLLM's flag silently falls back to W4A16. [gpu-results/quant-1-calib.md](gpu-results/quant-1-calib.md)

- **HyperQwen today.** One per-token int8 scale per row (`marlin_utils.py:742-745`, `marlin_quant_input` → `per_token_quant_int8`). On by default in batch with `INT8_ACT=int8 INT8_LAYERS=mlp` (`batch/start_qwen.sh:154-155`), opt-in for single-user (`single-user/start_qwen.sh:130-142`). Measured cost: batch +0.9% perplexity (gate_up), +2.2% (MLP, the default), +3.7% (all layers) (`docs/quality.md:30-36`); single-user all layers +4.1% perplexity and GSM8K 95.0 against 96.5 (`docs/optimizations.md:140-143`).
- **Mechanism.** The int8 MMA is unchanged. The int32 group partial is promoted in float by s_w[n,g] · s_a[m,g], like Marlin's existing fp8 branch (`marlin_template.h` ~1176-1185), instead of today's int16 integer promotion. Extra cost: about 2 fp32 ops on CUDA cores per output element per 128-K group, against 128 int8 MACs on tensor cores, so ≤ ~6% of MMA time **(inferred)**.
- **Route A: config and flags only, try first.** vLLM 0.31 already ships it. `compressed_tensors.py:710-742` (`_is_wNaM_int`) accepts `input_activations` with `strategy: group`, int8, dynamic, symmetric → `CompressedTensorsWNA8Int` (`:798`) → `schemes/compressed_tensors_wNa8.py:170` sets `_humming_input_quant_config` → `kernels/linear/mixed_precision/humming.py:50` (min capability 75) → Humming's `DynamicGroup` / `DynamicGroupToken` input modes (`humming/schema.py:160-216`).
  - **The trap.** `choose_mp_linear_kernel` tries Marlin before Humming (`vllm@v0.31.0:vllm/model_executor/kernels/linear/__init__.py:500-509, 841-870`). If Marlin wins, it silently ignores the input-quant config and runs W4A16. `--linear-backend humming` (or the per-quant `--kernel-config`) is required, and it's global: every W4A16 linear moves to Humming.
  - `humming-kernels` is a dependency of `vllm==0.31.0` (`requirements/cuda.txt`), so it's probably already in HyperQwen's venv **(inferred)**. HyperQwen has never evaluated it.
- **Route B.** Change Marlin itself: an activation-scale tensor [M, K/128] plus float promotion. Effort L, local build.
- **Gain, if the quality gate passes so int8 can cover more or all layers:**
  - A: 942 → up to 1,042 tok/s (+10.6%), the `INT8_LAYERS=.` row (`docs/quality.md:33-35`), minus any Humming-vs-Marlin slowdown (unknown). The all-layers row needs `GPU_UTIL` < 0.972 (`docs/quality.md:44-46`).
  - B–E: prefill 1,494 → 1,937 tok/s at 4k (+30%, TTFT ×0.77) and +19% at 51k (`docs/optimizations.md:133-137`), if INT8_ACT becomes the single-user default. Decode unchanged.
- **Risk.** Lossy, so it needs the quality gate (and [evals.md](evals.md)'s decode KL probe). Humming's speed on sm86 is unknown, and route A moves the tuned Marlin layers too.
- **Porting plan.**
  1. Extend `bench/act_calib.py` with per-128 group int8 and Hadamard-128 + per-token hooks, using the quantizers in `quant_a_act_quant.py`. Real per-layer numbers, no kernel risk.
  2. A `prepare/` script that adds `input_activations: {num_bits: 8, type: int, symmetric: true, dynamic: true, strategy: group, group_size: 128}` to the MLP config group of a copied checkpoint.
  3. `INT8_ACT=humming-g128` in `launcher_common.sh` → `--linear-backend humming`, with `VLLM_MARLIN_INPUT_DTYPE` unset.
- **GPU validation.** `bench/act_calib.py` with the new modes: down_proj and GDN in_proj errors fall ≥3×. `bench/quality_battery.py`: `INT8_LAYERS=.` costs ≤ +1.0% perplexity (today +3.7%) and GSM8K stays at or above the W4A16 row. `bench/run_benchmarks.sh` at 64 concurrency: ≥942 tok/s with all layers on Humming. `prefill_ab.sh`: Humming prefill within 5% of Marlin INT8_ACT.

### quant-2: make the negative-scale fold lossless

**Measured (2026-10-20):** INFO, check (b) confirmed on the real checkpoint: 50.4% of groups have a negative scale and 100% of those use code −8; the fold adds 0.069% of weight energy as error. [gpu-results/quant-2-check.md](gpu-results/quant-2-check.md)

- **HyperQwen.** `patches/marlin-int8-negative-scales.patch:36-38`: negative scales are common ("symmetric exports have ~50%"), so the patch folds the sign into the codes, and says "q == -8 loses one LSB, extremely rare". The underlying cause is that the kernel reads int16 scales as `uint16_t` (`marlin_template.h:1132-1136`).
- **Gap.** If check (b)'s reconstruction is right, the fold hits about half of all groups, always on the largest weight, adding +5–9% weight-error energy on every INT8_ACT layer. `bench/act_calib.py` can't see this: it measures activation error on W4A16 kernels. So some of the measured INT8_ACT perplexity cost may be this fold rather than int8 activations **(inferred, size unknown)**.
- **Fix A (load-time Python, no rebuild).** Keep the codes and emit a uint4-plus-zero-point layer: positive groups get zp 8; negative groups get code′ = 15 − code, zp 7 and |s|. Bit-exact in the CPU check. It runs on the `kS8 × kU4` kernel that `patches/marlin-int8-asym-zp.patch` already compiles and admits.
- **Fix B (rebuild).** `uint16_t` → `int16_t` at `marlin_template.h:1132-1136`, and `s.abs().max()` in `marlin_act_int8_process_scales` (`marlin_utils.py:544-548`).
- **Gain.** Quality only, compounding with quant-1: a smaller INT8_ACT perplexity cost makes wider int8 coverage acceptable. Speed 0, or slightly negative if the zero-point path is slower.
- **GPU validation.**
  1. `python3 -E -P quant_real_ckpt_check.py <checkpoint dir>` on the GPU box (CPU only). If negative groups are rare, close this item.
  2. Tests in the style of `bench/test_marlin_int8_asym.py`: remapped and original dequant match bit-exactly.
  3. Perplexity for the `INT8_ACT=int8` rows before and after. Win: any perplexity drop with zero-point-path throughput within 1%.

### quant-3: better GPTQ for HyperQwen's own int4 requants

- **HyperQwen.** `drafter/gptq_utils.py:29, 51` uses scale = amax/7, so code −8 is wasted. No act-order. `mse_clip` exists but the shipped build doesn't pass it (`drafter/README.md:51`). It requants the fast variant's lm_head, the MTP head and the DFlash2 drafter.
- **Change.** Signed ±amax/8 scales, MSE clip and act-order with static groups: −55% output error in check (d). Static-group act-order needs no `g_idx`, so it survives vLLM ≥ 0.30 dropping g_idx (`docs/vllm-0.30.md:53`). Negative scales already work on the W4A16 Marlin path, since the AutoRound body ships them.
- **Gain (inferred).** The int4 DFlash2 drafter loses ~5% acceptance at default sampling (3.2 against 3.4 tokens/step), because "noise in q hurts the acceptance probability" (`drafter/README.md:89-92`). Recovering half of that is +2–3% tok/s in B, C and E at default sampling, 0 at greedy. MTP ≤ +1%. Fast-variant lm_head: KL below 0.0024, perplexity +0.6% → about +0.3%.
- **Effort** S: about 40 lines in `gptq_utils.py`, then rerun `gptq_lm_head.py`, `requant_mtp_gptq.py` and `quant_dflash2.py`. **Risk:** drafter and MTP changes are lossless because verification is exact; the lm_head change is lossy but KL-gated.
- **GPU validation.** `gptq_lm_head.py` prints KL: lower than 0.0024. DFlash2 acceptance on the drafter README protocol (8 prompts × 1,024 tokens): default-sampling tokens/step ≥ 3.3, greedy not below today's. Perplexity and GSM8K for the fast variant.
- **Pairs with** drafter-3 in [drafter.md](drafter.md), which attacks the same 3.2-vs-3.4 gap by training; drafter-2 measures both.

### quant-4: block-Hadamard-128 rotation of linear inputs

Rotate the bf16 weights along K offline (W′ = blockdiag(H₁₂₈) · W), requantize the body with AutoRound or GPTQ from the 54 GB bf16 source, and add an online FWHT-128 fused into `per_token_quant_int8` for the in_proj and gate_up inputs and into the SiLU-mul kernel for down_proj. Check (a): per-token error falls from 10–14% to 1.3–1.8% on heavy-tailed inputs, and to 0.46% combined with per-128. Same gain as quant-1, but on the unchanged per-token Marlin kernel. Effort L; lossy through the requant. **Do it only if quant-1's route A is slow on sm86.** The attention lane found the same split for KV: Hadamard helps integer grids and hurts float grids ([attention-kv.md](attention-kv.md), attn-5).

### quant-5: batch-mode lm_head as W4A8

At 64 concurrency the int8-weight lm_head is compute-bound: 2 × 64 × 5120 × 248,320 = 162.7 GFLOP ≈ 2.7 ms at ~60 TFLOPS against ~1.4 ms of weight reads. W4A8 (the fast variant's int4 GPTQ head plus `lm_head` in `VLLM_MARLIN_INT8_INCLUDE_RE`) could save ~1.3–2 ms of a ~68 ms step: **+2–3% in mode A (inferred)**. It stacks two lossy steps (int4 head +0.6% perplexity, int8 final hidden state). Gate on KL with `drafter/data/hidden.npy`; check the logits path accepts the include regex.

### quant-6: fuse the int8 activation quant into its producer

DeepInfer fuses it into RMSNorm (`decoder_layer.py:43-44`, `norm.py:65-130`). HyperQwen quantizes in `marlin_quant_input`, a separate read and write. A 4k prefill moves about 25 GB of activations over 64 layers, ~28 ms at 900 GB/s, of which fusion saves about half: **~0.5% of a 2.7 s 4k prefill (inferred)**. Marlin would have to accept pre-quantized input. Low priority.

### quant-7: int4 embedding where `EMBED_UVA` can't be used

Community numbers: pool 201,275 → 218,112 tokens (+8.4%) with no perplexity or GSM8K change (`docs/third-party-checkpoints.md:150-163`); the change is BITS 8 → 4 in `prepare/quant_embed.py` / `quant_heads_stream.py:43, 207`. `EMBED_UVA=1` gives +15.7% (`single-user/README.md:443`) and dominates, so this matters only on D/E boxes that can't spare 2.1 GiB of pinned host RAM.

## Already in HyperQwen

- int4 weights with int8 per-token activations on int8 tensor cores (batch default, single-user opt-in).
- Per-group promotion on Marlin's int8 path, in integer arithmetic.
- Small or sensitive projections in bf16: `in_proj_a/b` (`docs/third-party-checkpoints.md:35-60`); DFlash2's kernel_projection, candidate_selector and hidden_projection (`drafter/quant_dflash2.py`).
- A shared, frozen target head for the drafter, as in DeepSpec (`DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py:270-290`): DFlash2's `_maybe_share_lm_head` (`vllm@v0.31.0:vllm/v1/spec_decode/llm_base_proposer.py:1508-1547`).
- Calibrated quantized heads: int8 RTN, and int4 GPTQ with KL 0.0024 in the fast variant (`docs/quality.md:54-58`).
- The MTP draft vocabulary (40,960 rows, `prepare/build_draft_vocab.py`) and embedding UVA offload (`PATCHES.md:75`).
- Marlin tile tuning, a measured negative (+0.4% end to end, `docs/optimizations.md:195-202`).

## Rejected

- **fp32 logits:** KL 1–6e-4 at T=1 with no argmax flips (check c), against 2.4e-3 for the int4 head, and it needs an fp32 Marlin epilogue. Revisit only if quant-3 pushes the head's KL below ~1e-3.
- **Fused lm_head + top-k:** DeepInfer doesn't do it either; logits are ~1 MB per row against a 636 MB weight read.
- **Draft-vocab-restricted DFlash2 candidates:** reads 16.5% of the head and saves ~0.57 ms of 25 ms, but 97.5% per-token coverage costs ~5% tokens/step at k=15. Net negative **(inferred)**. The drafter lane reached the same answer.
- **W8A16 or bf16 DFlash2 drafter:** +1.27 GB of reads per step ≈ +1.36 ms (+5.4%) for +6% acceptance at default sampling and 0 at greedy, plus ~1 GB of pool. quant-3 is the cheaper route.
- **Per-channel int4 lm_head:** +139–146% output error against g128 GPTQ to save 20 MB of scales.
- **Power-of-two (UE8M0) scales:** +40% error against fp32 scales; they exist to suit SM100 DeepGEMM and TMA.
- **fp8 e4m3 activations (DeepSeek's exact recipe):** no fp8 MMA on sm86, and per-32 fp8 (2.5–2.8%) is worse than per-128 int8 (2.0%).
- **Humming `VLLM_HUMMING_USE_F16_ACCUM`:** fp16 accumulation needs fp16 operands; the model runs bf16, and the int8 path already uses the faster int8 MMA.
- **Sub-4-bit body (Humming uint2/3):** DeepSeek uses 4-bit only on sparse MoE experts and keeps dense paths at 8-bit. High quality risk.
- **MXFP4 experts, Mega MoE, DeepEP, EP, hand-SASS FlashMLA, packed UE8M0/TMA scale layouts:** MoE, multi-node or SM100 only.
- **Markov logit-bias head, AcceptRatePredictor, top-k-space rejection:** not numerics; see [drafter.md](drafter.md) and [spec-decode.md](spec-decode.md).

## Open questions (GPU box)

1. Is `humming` importable in HyperQwen's venv, does int8 `DynamicGroup` run on sm86, and how fast is it against tuned Marlin? This picks quant-1's route A or route B/quant-4.
2. What are the real fold cost and scale-ratio spread? `python3 -E -P quant_real_ckpt_check.py models/Qwen3.8-27B-W4A16-AutoRound`. It assumes compressed-tensors contiguous packing and was self-tested on the fake checkpoint only.
3. What are the real per-layer errors for per-128 and Hadamard-128 against per-token? quant-1 step 1.
4. Does single-user `INT8_ACT=int8` also catch the DFlash2 drafter's linears? If so, quant-2 and quant-3 interact there, because signed scales would go through the fold.
5. Does `--linear-backend humming` also capture the int8-weight lm_head and embedding, and handle them correctly?
