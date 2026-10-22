# Attention, KV cache and long context

Lane: attention and KV cache (Opus, high effort), verified by the lead. Part of the [DeepInfer expedition](README.md).

Citation forms:
- HyperQwen paths are repo-relative.
- `DeepInfer/...` refers to deepseek-ai/DeepInfer at `774ce24`; `FlashMLA/...` is its `vendor/FlashMLA` submodule.
- `vllm@v0.31.0:...` refers to the pinned vLLM tag.
- **(inferred)** marks arithmetic or reasoning by the lane or the lead. Unmarked numbers are quoted from a file.

Modes: A batch; B DFlash2 chat at 64k, bf16 KV, T=8; C quoting, T=16; D `SPEC=mtp CTX=long`, fp8, k=3 so T=4 (`single-user/start_qwen.sh:221-224`); E `SPEC=dflash2 CTX=huge`, KVarN, T=8. T is the verify block (draft tokens + 1).

## Verdict

1. **Long-context verify on GA102 is bound by the tensor pipe, not by DRAM and not by dequant.** FlashMLA's Hopper deep-dive budgets SM cycles per KV token for MMA, dequant and DRAM, then builds the kernel around the largest (`FlashMLA/docs/20250929-hopper-fp8-sparse-deep-dive.md:13-25`). Applied to pthfa at T=8, the MMA term (fp16 inputs, fp32 accumulate) is 128 clk per KV token per KV head, DRAM is 77 and dequant about 4–8. pthfa's own header reaches the same conclusion from ncu: "the HMMA pipe is the busiest unit, and under the 220 W cap the SM clock drops to ~1.1 GHz, so the MMA count, not the memory system, sets the time" (`pthfa/csrc/pth_attn/attn_decode.cuh:13-16`, citing ADR 0012 Amendment 2, which isn't in this repo).
2. **So the deep-context lever is fewer or cheaper MMAs, not fewer bytes.** In order of loss: remove the row padding at T=8 (lossless), accumulate PV in fp16 (near-lossless), run QK on int8 (lossy, about SageAttention's level). Moving KVarN's Triton verify onto the same CUDA kernel is the largest single item for E at 200k+.
3. **DeepInfer's KV formats don't port to sm86.** MXFP8 per-32 gives bit-identical error to per-tensor e4m3 on this hardware, NVFP4 per-16 is no better than the Hadamard-int4 recipe KVarN already uses, and neither has an MMA on sm86 (CPU check below).
4. **DSA sparse attention is the large upside at 240k, and the riskiest.** A trained indexer is out of reach on a 3090, and a training-free selector has never been validated on Qwen3.8.

## Cycle budget per KV token per KV head

Rates per SM per clock **(inferred)** from GA102 whitepaper peaks at 1.695 GHz and 82 SMs: fp16 MMA with fp32 accumulate 511 FLOP; fp16 accumulate 1,022; int8 2,044; DRAM 6.73 B. Geometry: 16 full-attention layers, 4 KV heads, G = 6 query heads per KV head, head_dim 256.

| Case | MMA (QK + PV) | DRAM | Bound |
|---|---|---|---|
| T=8 today: 48 rows padded to 64, fp32 acc | 128 clk | int8 K+V 520 B → 77 clk | MMA |
| same, PV fp16 acc | 64 + 32 = 96 | 77 | MMA |
| same, QK int8 (s32 acc) + PV fp16 acc | 16 + 32 = 48 | 77 | DRAM (1.66× faster) |
| T=8, 48 rows unpadded (lead addition, below), fp32 acc | 96 | 77 | MMA |
| T=8, 48 rows unpadded, PV fp16 acc | 48 + 24 = 72 | 77 | DRAM, losslessly |
| T=4 (D), 24 rows, swap-AB kernel | 48 | 77 | DRAM |
| T=8 on KVarN, 210 B/token/head | 128 | 31 | MMA, up to 2.7× headroom |

Two corrections to the budget:
- **Dequant** is the bottleneck on H800 (50 clk against 34 for MMA, deep-dive:17-25). On GA102, pthfa's magic-number int8→fp16 (`pthfa/csrc/pth_attn/dequant.cuh`) costs roughly 4–8 clk **(inferred)**. FlashMLA's remedy (DSM crossover, deep-dive:27-48) is Hopper-only and targets the wrong unit here.
- **Clock (lead).** The DRAM column assumes boost clock. At the ~1.1 GHz the header reports under a power cap, DRAM delivers about 10.4 B/clk/SM, so int8 K+V costs about 50 clk while MMA stays at 128. The kernel is then *more* MMA-bound than the table shows, and even "unpadded + PV fp16" (72 clk) stays MMA-bound **(inferred)**.

**Calibration.** On a 3080 TP=2 rank at 128k, T=8, int8 KV: padded-MMA floor 4.62 ms, real-MMA 3.46 ms, DRAM 2.83 ms, measured 5.631 ms (`docs/long-context.md:538`, `cpu-checks/attn_roofline.out`). The kernel runs at about 82% of the padded fp32-accumulate MMA peak and about 50% of DRAM peak **(inferred)**.

### Roofline per engine step

Milliseconds for the target's 16 full-attention layers at peak 3090 rates (`cpu-checks/attn_roofline.out`):

| Case | T | DRAM | MMA real | MMA padded | PV fp16 acc | QK int8 + PV fp16 |
|---|---|---|---|---|---|---|
| B: DFlash2, 64k, bf16 | 8 | 4.59 | 2.90 | 3.87 | 2.90 | 1.45 |
| C: T=16, 64k, bf16 | 16 | 4.59 | 5.81 | 7.74 | 5.81 | 2.90 |
| D: MTP k=3, 150k, int8/fp8 | 4 | 5.25 | 3.32 | 4.43 | 3.32 | 1.66 |
| D′: DFlash2 + pthfa int8, 150k | 8 | 5.25 | 6.65 | 8.86 | 6.65 | 3.32 |
| E: DFlash2 KVarN k4v2, 240k | 8 | 3.15 → **3.45** | 10.63 | 14.18 | 10.63 | 5.32 |

**Lead correction (E).** The script charges KVarN 6 bits per element and ignores scales. `kvarn/README.md:74-75` gives 840 B/token/layer including scales, so the DRAM floor at 240k is 240,000 × 16 × 840 B / 936 GB/s = **3.45 ms**. Every E conclusion still holds, because E is MMA-bound by a factor of 3–4.

**Lead correction (D).** The D row's "padded" column assumes the 16-row-tile kernel. At T=4 (24 rows) pthfa uses the swap-AB decode kernel, which tiles rows in groups of 8 and has no padding (`attn_decode.cuh:9-22`; `instantiate_d256.cu:5-8`). D's real MMA floor is 3.32 ms, under its 5.25 ms DRAM floor, so D is DRAM-bound on any kernel.

## Candidates (ranked)

| ID | Title | Modes | Gain | Effort | Confidence |
|---|---|---|---|---|---|
| attn-1 | Cheaper MMAs in pthfa verify: unpadded rows at T=8, PV fp16 accumulate, then QK int8 | D′, and E via attn-2 | −4.6 ms/step at 150k T=8 | M | Medium |
| attn-2 | KVarN verify on a pthfa-family CUDA kernel | E | ~+21% tok/s at 240k, ~+2% at 20k | L | Medium-low |
| attn-3 | int8-QK prefill on the quantized-KV arms | D, E cold TTFT | −8 to −9% cold TTFT at 150k | M | Medium |
| attn-4 | Training-free top-k sparse attention (DSA-lite) | E | up to +30% at 240k, 0 at 20k | L | Low |
| attn-5 | Hadamard-rotated int8/int4 per-token-head KV | D (quality, maybe int4) | quality only; int4 would halve D's DRAM floor | S–M | Medium |
| attn-6 | Device-side balanced split scheduler | none as configured | ~0 | M | High (that it's ~0) |

E's 67 tok/s was measured at `--ctx 20000`: 3.15 tokens per step, so about 47 ms per step (`docs/long-context.md:567-571`). Attention is small at that depth. Every gain in this lane grows with depth, and E's per-step attention cost at 200k+ on a 3090 has **not been measured** (the profile in `docs/long-context.md:530-545` is from a 3080 TP=2).

### attn-1: cheaper MMAs in pthfa verify

- **DeepInfer.** The cycle-budget method transfers. The instruction choice doesn't: FlashMLA uses bf16 MMA with fp32 output (deep-dive:11).
- **HyperQwen.**
  - Both GEMMs use `mma.sync.m16n8k16.f32.f16.f16.f32` (`pthfa/csrc/pth_attn/ptx.cuh:334-341`).
  - int8 K and V are dequantised to fp16 with the magic-number trick (`dequant.cuh`). The V scale is folded per 64-token tile as an exponent (`attn_decode.cuh:54-60, 342-368`).
  - int8 MMA appears only in prefill on the bf16 arm (`patches/prefill-attn-int8.patch`, `docs/optimizations.md:148-160`). There is no `s8.s8` or fp16-accumulate MMA in `pthfa/` or `fa2/`.
- **Mechanism, in three stages.**
  1. **Unpadded rows at T=8 (lead addition, lossless).** The swap-AB decode kernel puts rows on the MMA's N dimension in 8-row n-tiles and is limited to row tiles of at most 32 rows, i.e. T ≤ 5 at G = 6 (`attn_decode.cuh:4-22`). T=6–10 falls through to the 16-row-tile kernel, where 48 rows pad to 64 (`launch.cu:363-385`). Extending the decode kernel to six n-tiles (48 rows) removes 25% of the MMA work and the duplicated dequant across warps at T=8. Risk: register pressure from 6 n-tiles of fp32 accumulators; the header already notes ptxas's occupancy budget (`attn_kernel.cuh:674`).
  2. **PV with fp16 accumulate.** Accumulate P×V with `m16n8k16.f16.f16.f16.f16` inside each 64-token tile (four k16 steps), then add the tile's partial into the existing fp32 O accumulator. Overflow is bounded: P ≤ 1, |V| ≤ 127 × v_scale, 64 terms.
  3. **QK on int8.** Quantise the Q rows once per launch (per-row absmax). Feed the cache's int8 K straight into `mma.m16n8k32.s32.s8.s8.s32` with no K dequant, and multiply by q_scale × k_scale in the existing scale step (`attn_decode.cuh:365-368`).
- **CPU check.** `cpu-checks/attn_mma_precision.py` (synthetic: 48 rows × 8,192 tokens; reference is fp64 on the same int8 cache):

  | Arm | Logit RMSE | Output rel-L2 |
  |---|---|---|
  | today | 0.0006 | 0.00042 |
  | PV fp16 acc | 0.0006 | 0.00049 |
  | QK fp16 acc (not proposed) | 0.0023 | 0.0043 |
  | QK int8 (s32) | 0.0107 | 0.0079 |

  int8 KV quantisation itself costs 0.016–0.069 output rel-L2 on the same generators (`cpu-checks/attn_kvq_error.out`). PV fp16 accumulate adds roughly nothing. int8 Q adds 10–50% to the existing KV error.
- **Gain (inferred).**
  - **D′** (`SPEC=dflash2 CTX=long PTH_FA=1`, T=8, 150k). Today: padded floor 8.86 ms, about 10.8 ms at the calibrated 82%. After stage 3: about DRAM 5.25 / 0.85 ≈ 6.2 ms. **About −4.6 ms per step at 150k**, about −0.5 ms at 20k. Stages 1 and 2 alone give about 5.0 ms of MMA (48 rows, PV fp16: 6.65 × 48/64), just under the 5.25 ms DRAM floor at boost clock, with no lossy step.
  - **D:** about 0. T=4 runs on the unpadded decode kernel and is DRAM-bound.
  - **A, B:** 0. B is bf16 on FA2, which is DRAM-bound at T=8 (4.59 vs 3.87 ms at 64k).
  - **C:** MMA-bound on paper (7.74 vs 4.59 ms at 64k), but it runs bf16 FA2 and `PTH_FA` is refused at `DFLASH_TOKENS≠7` (`start_qwen.sh:313-316`). Not quantified.
  - **E:** through attn-2.
- **Effort.** M. Two MMA wrappers in `ptx.cuh`, the s8 k32 B-fragment layout from the int8 smem tile, Q quantisation in `stage_q`, one Q-scale vector. Stage 1 is a separate kernel change in `attn_decode.cuh` and `instantiate_d256.cu`.
- **Risk.** PV fp16 accumulate is near-lossless but not bit-identical, so the proven tolerance in `bench/test_pthfa_attn.py` (header lines 6-8, `2^-8·(p@|v|)+…`) has to be re-derived. int8 QK is lossy: the prefill version reports cos > 0.99999 at 4–51k *with* K mean-smoothing (`docs/optimizations.md:152-156`), and the verify path has none. Needle and GSM8K gates are required for stage 3.
- **Porting plan.** Compile-time template flags (`PV_F16ACC`, `QK_S8`) in `attn_decode.cuh` and `ptx.cuh`; an env knob `PTHFA_MMA=f32|pv16|s8pv16` read in `pthfa/python/pth_flash_attn`; re-export as a pthfa revision. No vLLM patch, since the CUSTOM backend interface is unchanged. Ship stages 1 and 2 first.
- **GPU validation.**
  1. `bench/test_pthfa_attn.py` with the re-derived tolerance.
  2. A microbenchmark in the style of `bench/spec_attn_ctx_scan.py`: µs per layer at 64k, 128k and 150k, T=8. **Win: ≥1.3× per layer at 150k.**
  3. ncu: tensor-pipe utilisation falls, `dram__throughput` rises.
  4. End to end with the `docs/long-context.md:530-545` torch-profile protocol on a 3090. Win: target attention −35% per step at 128k, needles at 50% and 90% of 128k, GSM8K within noise.

### attn-2: KVarN verify on a pthfa-family CUDA kernel (mode E)

**Measured (2026-10-21):** INFO (e-profile), gate open: attention is 78% of E's step at ~170k (117.7 of 150.8 ms), `_kvarn_fused_decode_stage1` at ~40× its DRAM floor; E decodes 11.5 tok/s there. [gpu-results/e-profile.md](gpu-results/e-profile.md)

**Measured (2026-10-22):** WIN: a Triton shared-dequant verify kernel (`KVARN_VERIFY_SHARED=1`) runs 4.7× faster per layer at 240k, T=8; E decodes +38% at 20k and +125% at 170k warm (16.0 → 36.0 tok/s) on a spill-free pool; GSM8K n=1,600 within 1 point of stock, needle 3/3, KL and perplexity unchanged. [gpu-results/attn-2.md](gpu-results/attn-2.md)

- **DeepInfer.** Stores the main KV at 4 bits in production: e2m1 per 16 with an e4m3 scale (`DeepInfer/deepinfer/model_executor/layers/tilelang_ops/cache_common.py:119-127`, `DeepInfer/doc/inference_deployment_qa.md:213-221`). It reads those bytes with one hand-tuned kernel (FlashMLA, SM90/100 only). The lesson is the kernel, not the format.
- **HyperQwen.**
  - E reads KVarN k4v2 through Triton `tl.dot` with an fp32 accumulator (`kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py:525, 538, 1245`). Decode is "~20% slower than fp8 at 100k" (`kvarn/README.md:113`).
  - The shared-dequant verify kernel is off because it corrupted MTP proposals (`kvarn/README.md:104-106`). The fp16-dequant knob exists because register spills made some configs "2.5-24x slower" (`kvarn/README.md:87-103`).
- **Gap.** A 6.6-bit cache read by the slowest MMA mode. E at 240k is the most MMA-bound cell in the roofline: 14.18 ms padded MMA against a 3.45 ms DRAM floor.
- **Mechanism.** Reuse pthfa's planner, paging and split-KV. Swap the tile loader for KVarN's per-128-token tile layout (4-bit K codes with per-channel scale and zero point, 2-bit V with per-row scale). Fold the per-channel K scale into Q per tile, quantise Q to int8 and run the s8 MMA on unpacked K codes; the zero point becomes a per-tile row bias q·zp (**inferred**, algebra only). V uses fp16 dequant and PV with fp16 accumulate.
- **Format check.** `cpu-checks/attn_kvq_error.out`, output rel-L2 for query regimes A / B:
  - int4 per-token-head 0.84 / 0.22; with Hadamard 0.47 / 0.12.
  - NVFP4 per 16: 0.38 / 0.22; with Hadamard 0.42 / 0.13.
  - int4 per 16 with Hadamard: 0.26 / 0.09.

  NVFP4 isn't clearly better than the Hadamard-plus-int recipe KVarN already uses. Keep KVarN's format: perplexity +0.16% and needles from 4k to 240k are measured (`kvarn/README.md:111-113`).
- **Gain (inferred).** At 240k, attention ≈ 14.18 / 0.82 ≈ 17 ms now. New floor ≈ max(DRAM 3.45, MMA 14.18 × 48/128 = 5.3) / 0.85 ≈ 6.3 ms, so **about −11 ms per step**. The 47 ms step at 20k grows to roughly 63 ms at 240k, so that's about 63 → 52 ms, **about +21% tok/s at 240k** and about +2% at 20k. **Lead caveat:** the 17 ms assumes the Triton kernel matches pthfa's 82% of padded peak and pads the same way. Neither is measured; the GPU profile below comes first.
- **Effort.** L. A new loader with the stores unchanged, wired as a KVarN backend option. It must reproduce the KVarN tail pool (`Tail_K_pool_ptr`, `triton_kvarn_decode.py:1190-1215`).
- **Risk.** Moderate. Correctness against an fp64 reference is easy to test. The mamba-align, recycled-page and tail-pool interactions are where KVarN has broken before (`kvarn/README.md:104-106`).
- **Porting plan.** `pthfa/csrc/pth_attn/attn_decode_kvarn.cuh`, a `KVARN_BACKEND=pthfa` switch in the verify dispatcher of `triton_kvarn_decode.py`, and `bench/test_pthfa_kvarn.py` modelled on `bench/test_pthfa_attn.py`.
- **GPU validation.** First, a torch profile of E at 200k on a 3090 to measure the real attention share. Then µs per layer at 64k, 128k and 240k (T=8) against the Triton verify: **win ≥2× per layer at 240k**. End to end: `bench/labd_bench.py` at `--ctx 20000` and at 200k depth; win +15% tok/s at 200k, GSM8K n=600 within ±1 point of 95.2% (`docs/long-context.md:574`), needle at 100k.

### attn-3: int8-QK prefill on the quantized-KV arms

**Measured (2026-10-22):** WIN in mode D (`PTHFA_PREFILL_QK=s8`): cold TTFT −8.9% at 148k and −7.5% at 112k, with KL, perplexity, GSM8K and needle unchanged; mode E runs pending. [gpu-results/attn-3.md](gpu-results/attn-3.md)

Not a DeepInfer idea; it applies attn-1's reasoning to prefill.
- **HyperQwen.** `PREFILL_ATTN=int8` is bf16-arm-only and "falls through to FA2 otherwise" (`docs/optimizations.md:171`). The int8 Triton arm takes "251 s to load a 112k document against FLASH_ATTN's ~112 s" (`start_qwen.sh:232`).
- **Arithmetic (inferred), cold 150k prompt.** Causal attention is 16 layers × 24 heads × 4 × 256 × L²/2 = 4.4e15 FLOP; the GEMMs are 2 × 27e9 × 1.5e5 = 8.1e15. Attention is about 35% of cold prefill compute. The measured kernel gain is 1.27–1.35× on attention (`docs/optimizations.md:157`), giving **about −8 to −9% cold TTFT** at 150k. On the int8 cache arms K is already int8 per token, so K mean-smoothing must happen before the cache write, or as a per-head mean subtracted at store time and added back as q·mean.
- **Effort** M. **Risk** as attn-1 stage 3. **Validation:** `bench/prefill_ab.sh` at 51k, 112k and 150k on the D arm; win ≥5% TTFT.
- **Rank.** Low-medium: D and E are prefix-cached, so only the first turn pays.

### attn-4: training-free top-k sparse attention (DSA-lite)

- **DeepInfer.** A lightning indexer `relu(q@k)·weights` on FP4 index keys (`DeepInfer/deepinfer/model_executor/v4_modules/index_attn.py:57`), top-k through DeepSelect (`index_attn.py:12`), candidate-block pre-selection (`index_attn.py:19-22, 97-102`), cross-layer sharing of KV, index-K and top-k (`v4_modules/cross_layer.py:22-38`), and sparse decode via `indices` (`layers/attention_backend.py:77-137`).
- **HyperQwen.** Dense everywhere.
- **Training-free only with a proxy selector**, e.g. per-page min/max bounds over 64-token pages: 16 B/token/head, against 210 for KVarN and 520 for int8 **(inferred)**. DSA's indexer is trained; the DeepSeek-V3.2 report (not in these repos, quoted from memory) uses a dense KL warm-up of about 2B tokens and then about 0.94T tokens of sparse training. The warm-up alone is about 1.1e20 forward FLOPs, about 21 days on one 3090 **(inferred)**.
- **Upside (inferred, upper bound).** Top-k 2,048 per verify token, union over T=8 ≤ 16k tokens (6.8% of 240k) plus 7.6% bounds traffic: attention falls to about 15% of dense, 17 → ~2.5 ms, **about +30% at 240k** and about 0 at 20k.
- **Effort** L. **Risk:** lossy. Needle and long-context QA quality, and verify acceptance changes.
- **Plan.** An offline study first: dump real Q/K from a 128k run, measure the attention mass that bounds-selected top-k pages capture per layer, and gate on ≥0.99. Only then a kernel: pthfa with a page list, like FlashMLA's `indices` and `topk_length` (`FlashMLA/README.md`).

### attn-5: Hadamard-rotated int8/int4 per-token-head KV

- **DeepInfer** applies Hadamard only to the compressor's index-K; the main KV is fp8 without rotation (`tilelang_ops/fused_transform.py:726-732`; `v4_modules/attn.py:956, 986`).
- **The CPU check explains why** (`cpu-checks/attn_kvq_error.out`). Hadamard *hurts* float grids: e4m3 output error 0.047 → 0.078 and logit RMSE 0.09 → 0.32 in regime A. It *helps* uniform int grids: int8 per-token-head output 0.069 → 0.027, int4 per-token-head 0.84 → 0.47. The quant lane reached the same split for weights and activations ([quant-heads.md](quant-heads.md)).
- **HyperQwen.** int8 and int4 per-token-head are unrotated (no `hadamard` in `patches/` or `pthfa/`); KVarN already rotates.
- **Mechanism.** Rotate k and v by a 256-point orthonormal FWHT at cache store, and q before attention; un-rotate the output. Lossless in exact arithmetic, and one FWHT per token per layer is tiny.
- **Gain.** No speed by itself. It improves the int4 per-token-head profile (`single-user/alternative.sh`, `patches/int4-kv-per-token-head.patch`) and might let D move from fp8 to int4 (pool 2×), which would halve D's DRAM floor at 150k (5.25 → ~2.6 ms) **(inferred)**.
- **Effort** S–M. **Risk:** low for int8; int4 needs perplexity, needle and GSM8K gates.

### attn-6: device-side balanced split scheduler

FlashMLA partitions all requests' KV blocks across SM parts in one warp on device (`FlashMLA/csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu:5, 33-78`), so it's graph-safe and length-aware. pthfa's planner deliberately takes no sequence length and sizes only the workspace (`launch.cu:376-379`, `pthfa/python/pth_flash_attn/planner.py`). It would pay only with many mixed-length streams on pthfa; A uses FA2 and D/E run one or two streams. Recorded so nobody re-derives it.

## Other findings

- **pthfa's row-tile steps matter for the verify length** (handed to [spec-decode.md](spec-decode.md)). T ≤ 5 runs the unpadded decode kernel. T = 6–10 runs one 64-row tile with padding. T ≥ 11 needs two row tiles and so two passes over the KV **(inferred; L2 may absorb part of the second)**. Separately, `docs/gotchas.md:253-262` records the GEMM stair at 16 query tokens and the FA2 spec-attention stair at 21.
- **E's 2,048-token hybrid block** (`kvarn/README.md:76-77`) makes the prefix-reuse grain 2,048 tokens, so a multi-turn E session recomputes about 1k tokens per turn on average **(inferred)**.
- **GDN state.** DeepInfer has no linear-attention state; its fp32 rings and rollback slack are for SWA and compressors (`DeepInfer/deepinfer/worker/cache_engine.py:86-108`). HyperQwen already halves the GDN state with fp16 (`docs/optimizations.md:55-60`). Bounded replay in place of stored states would re-read the 48 GDN layers' weights per rejection, about 3/4 of a step **(inferred)**.
- **TileLang on sm86.** DeepInfer's CUDA TileLang ops nearly all set `TL_DISABLE_TMA_LOWER: True`, so the elementwise and store kernels aren't Hopper-locked, but the e4m3/e2m1 casts need sm89+ (`tilelang_ops/cuda/load_paged_kv_cache.py:28-38`). Adopting it adds a dependency for kernels HyperQwen already writes in Triton or CUDA.

## Already in HyperQwen

- Split-KV verify for quantized caches: "1.3 ms … against 7.4 ms … and 10.1 ms for FA2" (`patches/spec-decode-int8-kv.patch`, `docs/long-context.md:142-146`).
- GQA row packing, a graph-safe planner, magic-number int8 dequant and fused scale folding (pthfa).
- int8-QK SageAttention prefill on the bf16 arm (`patches/prefill-attn-int8.patch`).
- Hadamard plus variance-normalised 4/2-bit KV (KVarN); int4 per-token-head with the mq3d verify (`patches/int4-kv-per-token-head.patch`, `patches/spec-decode-int4-kv-mq3d.patch`).
- Hybrid prefix-cache alignment equivalent to DeepInfer's hit policy (mamba align, `patches/mamba-align-checkpoint-order.patch`; DeepInfer's is `cache_engine.py:32-33, 86-108`).
- Known negatives: PREFILL_ATTN alone measured "1-6% negative" on a WSL2 3090 (`docs/optimizations.md:165`); FA2 is already at "85% of the card's practical fp16 mma rate" (`docs/optimizations.md:150-151`); the int8 kernel's 260 B head stride costs about 13% (`docs/long-context.md:218-220`).

## Rejected

- YOCO/CED prefill truncation: needs consumer layers that share a hidden state (`DeepInfer/deepinfer/environ.py:188-193`); Qwen3.8 has none.
- FlashMLA's DSM crossover: Hopper clusters, and it attacks a dequant bound GA102 doesn't have.
- FlashMLA kernels: SM90/SM100 and CUDA 13.1+ only.
- MXFP8 KV on sm86: same precision as per-tensor e4m3, 3% more bytes, no fp8 MMA.
- NVFP4 main KV: no e2m1 MMA or cvt on sm86, and no edge over Hadamard-int4 or KVarN in the CPU check.
- DeepSelect top-k: sm100 only (`vendor/DeepSelect/setup.py:121-126`, sm80 commented out).
- SWA rings and bounded replay: the target has no SWA, and GDN replay costs a weight re-read.
- `_pad_attention_heads` to 64/128 (`attention_backend.py:8`): a Hopper wgmma shape artefact.
- Internal SASS-tuned FlashMLA (+5–10%, `DeepInfer/doc/inference_deployment_qa.md:223-237`): not released.
- Wide-EP, DeepEP, DeepGEMM MoE, 3FS, PD disaggregation: multi-node or MoE.
- TileLang adoption: a new dependency with no capability HyperQwen lacks.

## CPU checks

numpy only; run from `docs/research/deepinfer/cpu-checks/` with `python3 -I <script>`. The lead re-ran all three from the repo copy and the outputs match the committed `.out` files byte for byte.

| Script | What it measures |
|---|---|
| `attn_kvq_error.py` | K error and attention-output error of 7 KV formats, with and without Hadamard, under two synthetic query regimes (L = 8,192, D = 256, 6 query heads) |
| `attn_mma_precision.py` | logit and output error of the MMA accumulate modes in attn-1 |
| `attn_roofline.py` | per-step DRAM and MMA floors per mode, plus the 3080 calibration point |

The generators are synthetic. Real Qwen3.8 q/k channel statistics would settle the format and Hadamard questions properly (open question 5).

## Open questions

1. ncu on a **3090** at 150k, T=8: tensor-pipe vs DRAM utilisation. This decides attn-1; the 3080 calibration and the pthfa header are the only evidence so far.
   **Measured (2026-10-21):** ncu blocked: no performance-counter access under WSL2. A microbenchmark gives T=8 at 150k as 1,398 µs/layer, 24% of DRAM bandwidth, against 47–54% for the T ≤ 4 decode kernel on the same bytes. So it is on-chip-bound, not DRAM-bound; the pipe-level split is unmeasured. [gpu-results/attn-ncu.md](gpu-results/attn-ncu.md)
2. `docs/optimizations.md:152` says int8 runs at "2x the fp16 rate". The GA102 whitepaper gives 284 TOPS int8 against 71 TFLOPS for fp16 with fp32 accumulate (4×) and 142 with fp16 accumulate (2×). Which baseline did the doc mean?
3. Is D's fp8 FA2/FlashInfer decode near its 5.25 ms DRAM floor at 150k? If not, D's gain is in that backend, not in the MMA mode.
4. E's per-step attention cost at 200k+ on a 3090. attn-2's +21% rests on it.
5. Real Qwen3.8 q/k statistics from one capture, to replace the synthetic generators in `attn_kvq_error.py`.
