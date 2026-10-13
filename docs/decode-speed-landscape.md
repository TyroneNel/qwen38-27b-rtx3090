# The decode-speed landscape

Who else serves this model on this class of card, what they have that we do not, and the experiments that
would close the gaps. Surveyed 2026-10-13 at vLLM 0.31.0 (our pin).

**How to read this page.** Nothing here was measured by us: there is no GPU in the environment this was
written in, so every number is quoted from the repo that published it, with that repo's protocol attached. A
quoted row is a lead, not a result — the point of the page is the *protocol* column, which says what to run on
the 3090 to find out whether the lead survives contact with our stack. Our own published rows are in
[benchmarks.md](benchmarks.md); this page is the other half of that comparison.

## Where we stand

| our row | ours | the comparable row | their config | why it is not like-for-like |
|---|---|---|---|---|
| C1 decode, DFlash2 | 127 tok/s | **162.9 tok/s** — [r0b0tlab](https://github.com/r0b0tlab/qwen38-exl3-dflash2) | 1×3090, EXL3 4.00 bpw + native DFlash2 (ExLlamaV3 `community@355c6ee`), GSM8K greedy, ≤512 new, ctx 8192 | ExLlamaV3, not vLLM; GSM8K greedy (their acceptance 5.66, ours ~3.3); 4.00 bpw trellis, not W4A16 Marlin |
| C1 decode, DFlash2 | 127 tok/s | 141.1 tok/s code / 96.2 prose — [trellis-serve](https://github.com/sybil-solutions/trellis-serve) | 1×3090, EXL3 3.0 bpw + MTP, SGLang 0.5.20 plugin, 16k, fp8 KV | code-shaped workload; their prose row is *below* ours; 3.0 bpw |
| 150k context decode | ~95-100 tok/s (MTP) | 25.3 tok/s at 150k (r0b0tlab, acceptance 1.82, prefill 594) | same as above | **we win this one** — their engine holds 262k on 24 GB but pays far more at depth |
| C64 aggregate | ~1,035 tok/s | nothing found | — | no published repo matches this row |
| quoting its own prompt | 381 tok/s | nothing found | — | no published repo matches the lookup drafter either |

Two things follow. First, the only same-card row that clearly beats us is an **EXL3 + native DFlash2** stack
whose acceptance (5.66) is roughly 1.7× ours — the gap is largely acceptance, and acceptance is
workload-dependent, so the honest next step is to measure ours on *their* workload before believing the gap.
Second, our long-context and aggregate rows are not behind anyone's published number; do not trade them away
to chase a short-context row.

## The comparables

| repo | engine | why it matters here |
|---|---|---|
| [noonghunna/club-3090](https://github.com/noonghunna/club-3090) | vLLM, SGLang, llama.cpp, EXL3 | The densest source of measured 3090 data for this exact model, with per-row rig/power/date discipline and a vendored-patch ledger (`scripts/lib/profiles/patches.yml`) that names the upstream PRs. Our drafter appears in their drafter catalogue as `syvai-qwen38-dflash2`. |
| [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) | NInfer (C++/CUDA) | Already our published baseline; still the only other engine with a measured C1-C8 ladder on this card. |
| [r0b0tlab/qwen38-exl3-dflash2](https://github.com/r0b0tlab/qwen38-exl3-dflash2) | ExLlamaV3 | Same card, same model, native DFlash2, 262k context, and the acceptance arithmetic written down. |
| [z-lab/dflash](https://github.com/z-lab/dflash) | library + drafter author | The DFlash/DFlash2 drafter we requantize: acceptance tables for MTP vs DSpark vs DFlash2 on this model, and the selector's own headroom. |
| [EurekaZang/qwen-a100-fastpath](https://github.com/EurekaZang/qwen-a100-fastpath) | vLLM plugin | Same model, single-user long-context, and a GQA-packed split-KV verify attention with published per-layer timings. |
| [thc1006/qwen3.8-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.8-speculative-decoding-rtx3090) | llama.cpp | Preregistered, DOI-archived n-sweep on one 3090 — the only source that prices each extra verify position. |
| [sybil-solutions/trellis-serve](https://github.com/sybil-solutions/trellis-serve), [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) | SGLang / vLLM fork | EXL3 trellis kernels with an sm_86 path — the artifact our [#103](https://github.com/syv-ai/HyperQwen/issues/103) EXL3 route would start from. |
| [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant), [0xSero/turboquant](https://github.com/0xSero/turboquant) | vLLM + Triton | TurboQuant KV with an explicitly supported `RTX A6000 / SM86` path (`--kv-cache-dtype turboquant25/35`) — the alternative to KVarN 4/2 on the 16 full-attention layers. |
| [JakeATX/llamAmpere](https://github.com/JakeATX/llamAmpere), [JakeATX/sj-kvarn-codec](https://github.com/JakeATX/sj-kvarn-codec) | llama.cpp | An Ampere-tuned fork holding 262k on 24 GB, and a staged/journaled KVarN codec (MIT) next to the one we ported. |
| [L-Forster/open-jet](https://github.com/L-Forster/open-jet) | bespoke CUDA megakernel | 140 tok/s code / 107 greedy on a 3090 — the upper bound a persistent kernel reaches when launch bubbles are gone. |
| [halt95/qwen38-27b-3090s](https://github.com/halt95/qwen38-27b-3090s), [tonyd2wild/Qwen3.8-27B-3090-Cookbook](https://github.com/tonyd2wild/Qwen3.8-27B-3090-Cookbook) | vLLM forks | vLLM-side patch series (FlashInfer verify tiles, int4 lm_head + int4 drafter) and a decision matrix of the 3090 vLLM paths. |

## The gaps

Ordered by expected tokens per second per unit of risk. "Cheapest experiment" is written as an A/B on one
3090 with the metric to read; each one is a candidate for the benchmark harness we already have.

### G1. Verification width is static, and batch mode refuses to speculate at all

Our two modes are a hard switch: `single` speculates with a fixed `num_speculative_tokens`, `batch` sets
speculation aside entirely and says so on startup. That was the right call when nothing could vary K at
runtime, but the pin now can — twice.

- **Dynamic SD** (`"num_speculative_tokens_per_batch_size": [[1,8,7],[9,32,3],[33,512,0]]`), documented in
  our own pin at `docs/features/speculative_decoding/dynamic_speculative_decoding.md` (v0.31.0). K is a
  function of concurrency, including K=0. The field is present in the v0.31.0 tag's
  `vllm/config/speculative.py` and enforced in `vllm/v1/core/sched/scheduler.py` with a generic validator
  (`vllm/v1/spec_decode/dynamic/utils.py`), so it is a scheduler-level feature rather than a method-specific
  one — the published examples use eagle/eagle3 only because that is the common case. Confirm on our own
  ladder before relying on it for dflash/mtp.
- **Adaptive verification** (`"enable_adaptive_verification": true`), same directory: per-(request, position)
  survival scores compete for a global budget, so position 5 of a confident request can outrank position 1 of
  a doubtful one. Requires full cudagraphs (`--enforce-eager` is rejected at startup), no LoRA, no pipeline
  parallelism, and — today — **only DSpark with a confidence head**.

Why it matters: [ieslab](https://github.com/WorkWith-Research/ieslab-better-vllm-speculative-decoding)
measured on 2×3090 that the optimal K is load-dependent — K=8 wins at C=8, ties autoregressive at C=32, and
loses to it by ~21% at C=96; a 2k-token prefill flips "K=8 ≈ AR" to "AR +13%". Our own
[optimizations.md](optimizations.md) says the same thing from the other end: speculation wins below ~8
concurrent users, plain batching above. Two modes is one way to encode that; a K schedule is a better one,
because it removes the cliff between them and gives the DFlash2 residency ceiling
([benchmarks.md](benchmarks.md), C8 ≈ 324 vs MTP 407) somewhere to go.

**Cheapest experiment.** A: today's `SPEC=dflash2` fixed K=7. B: same, plus
`num_speculative_tokens_per_batch_size` with the knee read off a C1-C8-C32 ladder. Metric: aggregate decode
tok/s at C=8/C=16/C=32 and single-stream at C1 (the schedule must not cost C1 anything). Expect the C8+ rows to
stop falling off. Then, separately: `RedHatAI/Qwen3.8-27B-speculator.dspark` (5 layers, max draft 8, aux
layers 4/12/20/28/36/44/52/60, **confidence head enabled**, 262144 configured) with
`enable_adaptive_verification: true` against DFlash2 — we already carry `patches/dspark-draft-quant-config.patch`
for exactly this checkpoint family, so the config is the only new work. Risk: adaptive verification constrains
the attention backend; verify exactness on the quality battery.

### G2. We do not know our acceptance on the workloads where acceptance is high

Published DFlash2 acceptance for this model (`z-lab/Qwen3.8-27B-DFlash2`, SGLang, H200, block 8, `xhigh`,
temp 1.0/top-p 0.95/top-k 20): **5.46 GSM8K, 5.28 MATH-500, 4.79 MBPP, 4.39 HumanEval, 4.10 MT-Bench**;
against MTP 5.02/4.72/3.99/3.91/3.74 and DSpark 4.36/3.92/3.51/3.30/3.01. r0b0tlab measures 5.66 on GSM8K
greedy. Ours is ~3.1-3.65 tokens per step, but on our own 8-prompt mixed chat protocol with model-default
sampling, which is not the same workload at all. Until we have a per-workload acceptance table we cannot tell
"our drafter is worse" from "our prompts are harder", and every tuning decision downstream inherits that
blindness.

Two known drivers of the number, both cheap to control:

- **Reasoning effort.** club-3090 measured, A/B/A/B over four boots, n=12/arm, ~2.6k context, thinking on:
  `low` 5.362 (CV 2.9%) vs `xhigh` 3.872 (CV 4.7%) — **−27.8% against ~2.8% boot noise**. Their shipped
  configs now pin `low`. Our single-user mode runs thinking off, so this is not the same measurement, but it
  says acceptance is a function of the output distribution we ask for, not a property of the checkpoint.
- **The drafter's own quantisation.** Our [drafter/README.md](../drafter/README.md) already answers this
  internally: int4 GPTQ keeps greedy acceptance (3.34-3.65 vs 3.54 bf16) and loses **~5% at default sampling**
  (3.2 vs 3.4). So the requant is not the explanation for a 3.3-vs-5.5 gap — do not spend the extra ~2.7 GiB
  of VRAM on the bf16 drafter expecting one.

**Cheapest experiment.** Run the same server against GSM8K and a code benchmark (both already in
`bench/quality-data/`) and publish acceptance per workload next to our chat row, then re-run with
`reasoning_effort` pinned at `low` vs the template default. Metric: mean tokens per step and decode tok/s, per
workload. This is a measurement we should own regardless of what it says.

### G3. Our own reproduction says attention backend and KV dtype are worth 1.7× on the same box

[reproductions/README.md](reproductions/README.md) already contains the sharpest number on this page: on
2×3090 at 420 W, setup D as shipped reads 90.9 tok/s, and the *same profile* with
`EXTRA_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"` reads **156.0** — at the
same tokens per step (2.59/2.57), so "the gap is the fp8/FlashInfer path's step time, not the drafter (~31 to
~18 ms per step)". club-3090's engine matrix says the same from outside ("use FlashInfer wherever possible"
for 3090). Meanwhile our single-user default is `FLASH_ATTN` + bf16 KV and `CTX=long` moves to FlashInfer/fp8 —
the fast and slow configurations are both things we ship, and which one you get depends on the mode.

**Cheapest experiment.** A 2×2 on one card: {`FLASH_ATTN`+bf16, `TRITON_ATTN`+`int8_per_token_head`,
FlashInfer+fp8, FlashInfer+bf16} × {C1, C8} at fixed context. Metric: ms per step and decode tok/s, plus
acceptance (the KV dtype must not move it). The 2×3090 row is uncapped and one box; confirm the ordering at
250 W on our reference card before changing a default.

### G4. An engine A/B says SGLang is +43% at C1 on an otherwise identical config

club-3090's [PR #1226](https://github.com/noonghunna/club-3090/pull/1226) row is an engine comparison with
everything else held: same model, quant, DFlash2, FP8 KV, TP=4, 262144 context, 220 W — only engine and
drafter depth (7 vs 8) differ. SGLang 144.6/244.8 vs vLLM 100.8/185.0 decode at c=1 (**+43%**), ITL p50
21.9 ms vs ~31 ms. The win narrows as concurrency rises (+4-6% by c=8, +11% at c=16), and their prefill raw
TPS is slightly lower, but TTFT is *lower* at 4K+ (1.5 s vs 2.1 s at 4K).

This is the largest single-stream number anyone has published for this model on 3090-class hardware, and it is
an engine difference rather than a kernel we can copy. Two ways to take it: run SGLang as a second mode for
interactive traffic (its RadixAttention is also the better prefix-cache story for agents), or treat the delta
as a bug report against our vLLM configuration and hunt the cause — step-time overhead, scheduler behaviour,
attention backend, V1/V2 runner. The second is cheaper to start and does not fork our deployment story.

**Cheapest experiment.** Reproduce the c=1 shape (one request, ~1000-token answer, DFlash2, FP8 KV, long
context) and compare our ms/step with the published vLLM arm at the same depth. If we are already at their
vLLM arm, the gap is engine-level; if we are closer to SGLang's arm, theirs was mis-configured and there is
nothing to chase.

### G5. The verify attention still leaves SMs idle

Our `spec-decode-attn.patch` replaced FA2's one-query-token split with a Triton split-KV kernel because
multi-query decode ran 24 CTAs on 82 SMs. [EurekaZang/qwen-a100-fastpath](https://github.com/EurekaZang/qwen-a100-fastpath)
goes further on the same model shape: `plugin/qwen_fastpath/spec_attn.py` packs the Q×G query rows that share a
KV head into one tile and splits the cached prefix over **64 CTAs**, with a suffix kernel for the current step
and a log-sum-exp merge, fixed grids and no host sync (so it stays graph-safe). Published on A100/W8A8/MTP
k=4: **1751 → 126 µs per layer at 38K (13.9×)**, 4590 → 294 µs at 100K (15.6×), decode at 38K 35.0 → 74.4
tok/s (2.13×), error ≤ FA2. A100 is not our card, but the layer-time ratio is architecture-independent enough
to be worth porting behind an env switch, and our split-KV patch is the natural place to add the packing and
the wider grid.

**Cheapest experiment.** Port the packing into `spec-decode-attn.patch` behind an env knob, then re-run the
verify-attention microbenchmark we already have at 1.5k/16k/38k and the C1 decode row. Metric: µs per layer and
decode tok/s; the acceptance must not move.

### G6. The EXL3 route is now buildable (issue #103)

Three artifacts exist that did not when [#103](https://github.com/syv-ai/HyperQwen/issues/103) was filed:
[trellis-serve](https://github.com/sybil-solutions/trellis-serve) implements EXL3 in stock SGLang and vLLM with
an sm_86 path and claims bit-exactness with ExLlamaV3;
[vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) exposes `--quantization exl3` with mixed per-layer
bitrates; and [r0b0tlab](https://github.com/r0b0tlab/qwen38-exl3-dflash2) publishes 4.00 bpw target + drafter
quants that hold 262k context on one 3090 at 162.9 tok/s C1 (GSM8K greedy). Their budget arithmetic is the
useful part: weights ~15.5 GiB target + ~1.2 GiB draft, 16 full-attention layers at 3-bit KV ≈ 3.0 GiB at
262k, GDN verify history ≈ 1.22 GiB per slot with a draft attached.

**Cheapest experiment.** Serve the published EXL3 4.00 bpw target + DFlash2 draft through `vllm-exl3` (or
SGLang, where trellis-serve is further along), run our own 8-prompt C1 protocol and the GSM8K acceptance probe
from G2. Metric: decode tok/s at C1 and the acceptance on the same prompts. This is the largest-effort item on
the page; the point of the experiment is to find out whether the W4A16-Marlin path is the thing holding C1
back, before committing to a second quantization path.

### G7. Recurrent-state residency is what ends speculation at concurrency

Three independent measurements land on the same mechanism:

- Our own: DFlash2 at C8 falls to ~324 tok/s while MTP holds 407 ([benchmarks.md](benchmarks.md)), and
  [long-context.md](long-context.md) traces it to a speculating request reserving recurrent-state pages the
  pool has few of.
- r0b0tlab's budget: GDN recurrent states ≈ **1.22 GiB per sequence slot with a draft attached** vs ≈ 0.15 GiB
  without — an 8× difference, from 48 layers × 8 fp32 history rows.
- deathwing475/vllm-030win, on a 24 GB-class card: **64 MiB of extra residency costs −30% decode**.

ninfer-3090's "ReplaySSM" (cheap save/restore of spec-decode state so MTP survives C8) is the concept to look
at, though their engine is not ours. In our tree the levers are the verify-history dtype and count
(`hybrid-kv-groups-v2-cudagraph`, `mamba-align-checkpoint-order`), and the fp16 GDN state we already ship.

**Cheapest experiment.** Measure the pool and the preemption count while a C1-C8 ladder runs with DFlash2
attached, then repeat with the verify history at fp16 vs bf16 vs int8 and with `--mamba-block-size` raised
(see G10). Metric: concurrent sequences held, preemptions, and aggregate decode tok/s at C8 — plus acceptance,
since shrinking history is where a silent quality bug would live.

### G8. TurboQuant KV has an explicit sm86 path

[mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) documents `RTX A6000 / SM86` as a supported
target for `--kv-cache-dtype turboquant25/35` (source build, Triton kernels, per-layer metadata from
`benchmarks/generate_turboquant_metadata.py`), and [0xSero/turboquant](https://github.com/0xSero/turboquant)
ships score-from-packed-data kernels that avoid materialising the dequantised KV. Our KVarN port covers the
same ground with 4/2-bit; the question is which is faster per decode step and cheaper in KL on the 16
full-attention layers, where our [long-context.md](long-context.md) already measures the int4 route's
trade-offs.

**Cheapest experiment.** A/B TurboQuant 3.5 vs KVarN 4/2 at 150k context: metric decode tok/s, KV tokens
available, and the perplexity/NIAH rungs we already run. Do not hand-write a new int4-KV attention kernel for
this — a community 4-bit-KV HIP attention lost to the Triton packed path (598 vs 510 µs).

### G9. The CUDA-graph capture set is probably costing us memory and speed

[riceharvest/b70-qwen38-cookbook](https://github.com/riceharvest/b70-qwen38-cookbook) reports a capture set of
`[1..40]` instead of the default ladder: **0.3 GiB vs 3.7 GiB of graph memory, and decode +5-19%**. On a 24 GB
card that is ~3.4 GiB back for the KV pool *and* a faster step. We already carry graph-related patches
(`hybrid-kv-groups-v2-cudagraph`, `spec-sampler-prewarm`, `dflash2-prewarm`), and club-3090 separately warns
that drafter depth can silently downgrade cudagraph mode (vllm#35387: at 29K, no-spec 57.0 → n=1 40.6 → n=2
36.0 → n=3 32.7 tok/s, profiled to `FULL_AND_PIECEWISE → PIECEWISE`).

**Cheapest experiment.** Read the actual capture set and graph memory our V1/V2 runners allocate at boot, then
narrow the ladder and re-run C1 and C8. Metric: boot resident MiB, graph MiB, decode tok/s, and the
cudagraph-mode line in the log. Lowest effort on this page.


### G10. Two pool-accounting findings that buy context, not decode

- **Mamba checkpoint pages come out of the KV pool.** club-3090 measured the pool filling at ~80% of nominal
  because checkpoint pages are not counted in vLLM's "GPU KV cache size"; `MAMBA_BLOCK_SIZE=8192` cuts
  checkpoints 5×, which let two 242K requests run concurrently with **0 preemptions**, prefix cache intact
  (29× TTFT), and no per-turn cost (75.3/106.7 decode, 1837 @10K / 1380 @90K). We expose
  `--mamba-cache-mode align` but not the block size; our own [long-context.md](long-context.md) counts
  checkpoint pages, so this is a small change to our reservation arithmetic.
- **Boot residency.** club-3090 backports vllm#35975 (skip the `inputs_embeds` GPU buffer for text-only models)
  and measures **−444 MiB at boot on Qwen3.6-27B**; the upstream PR is closed, so the effect may differ on our
  pin. We already have the UVA embedding knob for a related reason.

**Cheapest experiment.** A: shipped pool accounting. B: `--mamba-block-size 8192`. Metric: reported KV tokens,
preemptions and follow-up TTFT with two long concurrent requests. Then read boot resident MiB with and without
an `inputs_embeds`-optional change, if our pin still allocates it.

### G11. Four upstream items where our tree may already be the fix

Worth doing because each is a silent speed or correctness loss, and because we would be upstreaming work we
have already paid for:

- **vllm#51581 (open issue)** — "DFlash fused-KV projection calls `F.linear` on a sliced `qkv_proj` weight —
  breaks (and can silently corrupt) any weight-quantized drafter". Our
  `patches/dflash2-lookup-drafting.patch` carries `_dense_kv_rows()` for exactly this; club-3090 carries a
  second copy. Two independent fixes and an open issue is an upstream PR waiting to happen.
- **vllm#48905 (open issue)** — "Marlin W4A8 (int8 activations) corrupts on checkpoints with negative group
  scales". That is our `patches/marlin-int8-negative-scales.patch`, found independently.
- **vllm#48904 (open issue)** — "`VLLM_MARLIN_INPUT_DTYPE=int8` silently ignored for auto-round (INC)
  checkpoints". If our checkpoint routes through the INC path, our per-layer int8 selection may be a no-op on
  some layers — i.e. int8 prefill not actually on. Check by asserting the loaded `LinearMethod` and its
  `input_dtype` per layer.
- **vllm#48375 / #43559 (open)** — `MambaManager.find_longest_cache_hit` receives `drop_eagle_block` and (as of
  v0.31.0) still does not read it; the coordinator instead sets `manager.drop_eagle_checkpoint_block = True`
  under `use_eagle`, and its own comment says "no margin for mamba: its finder never drops". Whether that
  instance flag closes the original hole on our pin is not something to assume — our
  `mamba-align-checkpoint-order.patch` sits in the same interaction, and the failure mode is a resumed state
  that includes draft tokens the verifier later rejected (~20% acceptance loss upstream). **Verify with a
  probe**: MTP + prefix caching, assert `prefix_cache_hits_total` grows, and compare acceptance against the
  same run with `--no-enable-prefix-caching`.


### G12. Smaller decode-path items

- **Draft head cost.** [b70](https://github.com/riceharvest/b70-qwen38-cookbook) quantises only the *draft* LM
  head (the target verifies on the full vocab, so it stays lossless): 4.3 → **2.2 ms per verify step**, +28K KV
  tokens, KL +0.0006. We already requantise the main lm_head; the draft head is the one still read at full
  width every step.
- **Verify-depth knee.** [thc1006](https://github.com/thc1006/qwen3.8-speculative-decoding-rtx3090), on one
  3090 in llama.cpp, prices each extra verify position at **0.25-0.29 of a decode step** and finds
  `--spec-draft-n-max 2` best (+59.8% over baseline, [+57.0, +62.8]). Our knee is k=4 on a 7-draft block and we
  cite club-3090's "n=4 eventually dies, n=3 stable". A one-card n-sweep on our stack, on a fixed workload,
  settles whether our shipped n is still right at 250 W.
- **Suffix decoding.** Our pin ships `vllm/v1/spec_decode/suffix_decoding.py` and `vocab_mapping.py`
  (`draft_to_target_ids`), with knobs (`suffix_decoding_max_tree_depth`, `max_spec_factor`, `min_token_prob`).
  The lookup drafter is our answer to the same problem and beats published n-gram numbers, so this is a
  last-priority A/B on a code-edit trace.
- **Draft-vocabulary sanity.** `vocab_mapping.py` gives us the machinery to assert that our 40k truncating
  draft vocabulary is a superset of every token the target actually emits on a workload; a nonzero OOV rate is
  a correctness bug that presents as a speed bug.

## The backlog

Ranked by expected tokens per second per unit of risk. "Config" means no patch is needed on our pin.

| # | change | artifact | metric | expected | risk |
|---|---|---|---|---|---|
| 1 | Load-aware K (dynamic SD) instead of the single/batch switch | config, `num_speculative_tokens_per_batch_size` | aggregate tok/s at C8/C16/C32; C1 unchanged | stops the C8+ collapse; removes the mode cliff | low |
| 2 | Per-workload acceptance table + `reasoning_effort` A/B | our harness + `bench/quality-data/` | tokens/step and tok/s per workload | makes every other row interpretable; up to −28% acceptance recovered if effort is mis-set | none |
| 3 | Narrow CUDA-graph capture set | V1/V2 runner | graph MiB, boot MiB, C1/C8 tok/s | +5-19% decode and ~3.4 GiB of pool | low |
| 4 | Attention backend × KV dtype 2×2 | config, `EXTRA_ARGS` | ms/step, C1/C8 tok/s | our own 2×3090 row suggests up to 1.7× between our own configs | low |
| 5 | GQA-packed split-KV verify attention | `spec-decode-attn.patch` | µs/layer at 1.5k/16k/38k | 13.9-15.6× attention-layer time on A100, less here | med |
| 6 | DSpark + `enable_adaptive_verification` vs DFlash2 | `RedHatAI/Qwen3.8-27B-speculator.dspark` | C1-C32 tok/s, exactness | one config across the load range | med |
| 7 | Draft-head quantisation | `prepare/`, b70 pattern | ms/verify step, KV tokens | ~2 ms/step, +28K tokens | low |
| 8 | TurboQuant vs KVarN at 150k | `mitkox/vllm-turboquant` | tok/s, KV tokens, KL/NIAH | context per byte at equal decode | med |
| 9 | Upstream the four fixes in G11 | our patches | — | no local speed, but they are correctness bugs others are hitting | low |
| 10 | `--mamba-block-size` and boot residency | config | KV tokens, preemptions, TTFT | ~20% more usable pool, fewer preemptions | low |
| 11 | EXL3 second quantization path | `vllm-exl3` / `trellis-serve` | C1 tok/s on our prompts | up to +28% C1 on a rival engine's row; largest effort | high |
| 12 | SGLang as a second engine | club-3090 PR #1226 row | ms/step at C1 | +43% C1 on an identical config (their measurement) | high |

## What we already lead on

Worth stating so this page does not become a reason to churn: no published repo matches our 381 tok/s
context-quoting row or our ~1,035 tok/s C64 aggregate, our long-context decode at 150k-240k is ahead of the
one published 3090 long-context row (25.3 tok/s at 150k, acceptance 1.82), and the MTP draft-vocabulary work
(a 40k head counted over the model's own outputs) has no equivalent in any of these repos — club-3090's
catalogue still points at our drafter as the quantized option.

## Provenance and limits

- **Verified directly** while writing this: every repo and Hugging Face model named here exists and carries the
  description quoted; the upstream vLLM documents and files cited (`adaptive_verification.md`,
  `dynamic_speculative_decoding.md`, `suffix_decoding.py`, `vocab_mapping.py`, `dynamic/utils.py`) are present
  in the **v0.31.0** tag we pin, not only on main; issues #48904, #48905, #51581 are open and #48375 is an open
  PR; the DFlash2 acceptance and throughput tables are from the published model card; the reasoning-effort and
  engine A/B numbers are club-3090's, with their protocols attached.
- **Not verified, treat as leads**: the A100 fastpath's per-layer timings, the b70 cookbook's capture-set and
  draft-head numbers, open-jet's megakernel rows, ninfer's ReplaySSM mechanism, and any figure quoted from a
  repo whose README is the only source. None of them were reproduced on our card.
- **Nothing here was measured on a 3090 by us.** There is no GPU in the environment this survey was written
  in, so every "expected" column is an inference from someone else's measurement and should be treated as a
  hypothesis to falsify with `bench/run_benchmarks.sh`.

