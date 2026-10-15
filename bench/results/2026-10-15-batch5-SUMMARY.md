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

## Q1: the mamba knobs

Pending at the time of the Q2 push.

## Q3, Q4, Q5

Pending at the time of the Q2 push.

## Method notes

- **Steps 0-1:** preflight in `2026-10-15-batch5-preflight/preflight.txt`: 250.00 W, both GPUs `display_active`
  Disabled, idle Shared Usage 0.25 MiB, identity TyroneNel / noreply.
- **Rebuild:** Step 1's diff listed `single-user/start_qwen.sh`, so the image was rebuilt as `hyperqwen:exp-844cd90`
  (`2026-10-15-batch5-image/image.txt`). The pip layers were cached, so the image holds vLLM 0.31.0. No `.env*` is
  baked in (O-2 confirmed).
- **Tools:** `2026-10-15-batch5-tools/` holds batch 4's `arm.sh`, `spillwatch.sh`, `memsum.sh`, `sampler.ps1` and
  `running_probe.py`, copied with TOOLS repointed, plus the batch drivers.
- **The spill floor** is taken between the engine's KV-cache line and the first request.
