The printed bound holds at the long end on all four boots and fails at the short end on both band profiles: it prints 6 where k=7 seats 7 (−1) and 11 where k=3 seats 13 (−2) at ~1k-token prompts.

# Batch 5 summary (2026-10-15, RTX 3090 at 250 W, image `hyperqwen:exp-844cd90`)

## Q2: the launcher seat print on hardware (`2026-10-15-Q2-*/report.md`)

All four boots ran at `MAX_SEQS=16` and `MAX_LEN=65536`.

| arm | printed ~1k / `MAX_LEN` | `num_gpu_blocks` | observed ~1k / ~32k / ~61k | error ~1k | error at depth (vs ~32k / ~61k) |
|---|---|---|---|---|---|
| a: int8, k=7, `KV_MEM=5583457484` | 6 / 1 | 382 | **7** / 2 / 1 | **−1** | −1 / 0 |
| b: int8, k=3 (band 2) | 11 / 2 | **404** | **13** / 3 / 2 | **−2** | −1 / 0 |
| c: int8, k=7, `KV_MEM` −1 GiB | 5 / 1 | 308 | 5 / 2 / 1 | 0 | −1 / 0 |
| d: shipped KV, k=7 (supplementary) | 6 / 1 | 380 | 6 / 1 / 1 | 0 | 0 / 0 |

- **The bias is one-sided, low at the short end, from two named sources:**
  - **No attention pages at ~1k.** The engine's own accounting shows exactly 6(k+2) pages per request at ~1k
    (54.00 and 30.00), against the print's +4.
  - **The pool at k=3.** It is 404 pages, not `KV_MEM`/14,601,709 = 382, because the engine picks an 816-token block
    there. The launcher comment says that reading is "a little high below" k=7; measured, it is low.
- **The print tracks the pool.** Q2-c's 308 printed = 308 `num_gpu_blocks`.
- **The constants are an int8 calibration as much as a k=7 one.** On the shipped KV the engine's block is 448 tokens,
  and pages per request are 54 + 2·⌈ctx/448⌉ exactly (204 at 32.8k, 332 at 61.5k).
- **Computed, not booted:** a default-KV user at `MAX_LEN=32768` would be printed 2 against a measured fit of 1.
- **No preemption, no measurement spill**, on any of the 12 waves.

## Q1: no mamba knob engages on this model, and none helps (`2026-10-15-Q1-*/report.md`)

- **Profile:** band 2 at the full pin, `MAX_SEQS=16`. Order a1 → b → c → d → bf → cf → a2.
- **`--mamba-ssu-algorithm horizontal|vertical` (b, c): refused at validation.**
  `ValueError: Mamba SSU algorithm selection is only supported with the FlashInfer backend.`
- **`--mamba-backend flashinfer`, alone (d) or with either algorithm (bf, cf): applied to nothing.**
  - The args show `'mamba_backend': 'flashinfer'`.
  - The kernel lines are identical to the baseline: `GDN decode kernel: triton`, 132 JIT compile keys.
  - In the 0.31.0 source only Mamba2 layers and ReplaySSM read these settings; the GDN layers never do.
- **Speed:**
  - C1 T=0 decode is 91.5-91.8 on all five booted arms.
  - N=8 is 342.5 (d) against 336.9 / 338.4 (a1 / a2), +1.4%, from tokens/step, at a *higher* 63.8 ms/pass.
  - No arm reaches the 3% bar, so there was no GSM8K smoke.

## Q3: the `CG=40` acceptance gain is trajectory drift (`2026-10-15-Q3-*/report.md`)

- **The free-run gap reproduces exactly:** C1 T=0 tok/step 3.34 → 3.48 on two boots per arm.
- **The step is unchanged:** N=1 32.3 → 32.1 ms.
- **On one frozen target replayed by both boots the gap vanishes.**
  - GSM8K: 39 vs 39 steps.
  - real4: 182 vs 184, so `CG=40` is 1.1% *lower*.
  - The harness's own free-run: 4.59 vs 4.57.
- **What `CG` does move is greedy numerics.** The boots fork from the frozen continuation at different points, and the
  forked text accepts differently.
- **Consequences:** `CG` is not an acceptance knob, and X-1 is not an upstream acceptance report.

## Q5: the 16-seat C1 loss reproduces at 3.9%, and it is acceptance, not step time (`2026-10-15-Q5-*/report.md`)

- **C1 T=0 decode:** 102.8 / 106.8 at `MAX_SEQS=16` against 108.6 / 109.6 at 8, −3.9% on the means. Batch 4's −7.8%
  used batch 3's 112.1 as the 8-seat side.
- **Tokens/step:** 3.17 against 3.34.
- **`ms/pass` at N=1:** 32.6 against 32.3. X-3's "32.7 vs 29.5" does not reproduce on matched boots.
- **On the same frozen tokens:** GSM8K takes 39 steps on both, and real4 takes 187 vs 182 (−2.5%).
- **The buffer is named:** the int4 3D scratch, `capacity=128` → `256` query tokens (48.38 + 64.50 → 96.75 + 129.00
  MiB). It is the only `MAX_SEQS`-scaled allocation that differs between the boot logs.

## Q4: IFBench, band 2 against the shipped profile: stopped, no result (`2026-10-15-Q4-band2/report.md`)

- **Stopped on the owner's instruction:** "the IfBench is hitting token limit. results are void, stop skip IFBench".
- **What had run:** the band-2 arm, with 204 requests finished (180 `stop`, 24 `length` at `max_tokens` 16384) and
  0 preemptions at 4 threads.
- **Nothing scored**, and the shipped arm never booted.
- **Band 2's precision gate** is still GSM8K n=200 plus perplexity.

## Method notes

- **Steps 0-1:** preflight in `2026-10-15-batch5-preflight/preflight.txt`: 250.00 W, both GPUs `display_active`
  Disabled, idle Shared Usage 0.25 MiB, identity TyroneNel / noreply.
- **Rebuild:** Step 1's diff listed `single-user/start_qwen.sh`, so the image was rebuilt as `hyperqwen:exp-844cd90`
  (`2026-10-15-batch5-image/image.txt`). The pip layers were cached, so the image holds vLLM 0.31.0. No `.env*` is
  baked in (O-2 confirmed).
- **Tools:** `2026-10-15-batch5-tools/` holds batch 4's `arm.sh`, `spillwatch.sh`, `memsum.sh`, `sampler.ps1` and
  `running_probe.py`, copied with TOOLS repointed, plus the batch drivers.
- **The spill floor** is taken between the engine's KV-cache line and the first request.
