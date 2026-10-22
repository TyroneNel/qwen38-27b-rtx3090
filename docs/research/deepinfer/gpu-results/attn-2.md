# attn-2: shared-dequant KVarN verify kernel (mode E)

| Field | Value |
|---|---|
| Date | 2026-10-22 14:27–17:28 SAST |
| Commit | `240d914` + `experiments/patches/attn2-kvarn-verify-shared.patch` (Triton; `bench/test_kvarn_verify.py`, `bench/kvarn_verify_bench.py`) |
| Image | `hq:exp-attn2` (`sha256:db030749438c…`), built on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…` (main `ce9615c`) |
| GPU | RTX 3090, 250 W enforced, driver 610.88; busy median SM clock 1440–1455 MHz |
| `.env` knobs | `SPEC=dflash2 CTX=huge PREFIX_CACHE=1 KV_MEM=4000000000 DFLASH_MAX_LEN=180224` (spill-free E pool, see below). On: `KVARN_VERIFY_SHARED=1`. Variants: `KVARN_VERIFY_SHARED_PV16=1`, `DFLASH_TOKENS=3`. |
| Spill | Every speed boot judged by PCIe (`ENVIRONMENT.md`, spill policy): job-window median 22–24 MB/s, maximum 78–165 MB/s, no second above 500 MB/s in any of the 9 speed boots |
| Context | `bench/labd_bench.py` at 20k (six tasks, 512 tokens), then 170k `qa` cold (64 tokens) and warm (256 tokens) on the long corpus, thinking off |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\attn-2\` (`test_*`, `bench*`, `s_{off,on}_{1,2}`, `s_onpv16`, `s_{off,on}_k3`, `kl_{off,on}`; spilled first look: `off_1`, `on_1`, `stockpool_s_off_1`) |

## What it is

KVarN's stock verify runs one program per (query token, KV head, split). At a verify width of 8, every int4 tile is dequantized 8 times, and each dot pads 8 rows to the MMA's 16. The patch adds `_kvarn_verify_shared_stage1/2`: one program per (request, KV head, split) dequantizes each chunk once for all `QLEN × Q_PER_KV` rows (48 in mode E, packed into a 64-row tile) and runs fp16-operand MMAs with fp32 accumulation. With the knob unset, no new kernel is compiled or launched. The pthfa CUDA variant (`attn2-kvarn-pthfa.patch`, `PTHFA_KVARN=1`) was built but not timed: the Triton kernel already clears the bar.

## Correctness

`bench/test_kvarn_verify.py`: 40/40 OK in f32 and with `KVARN_FP16_DEQUANT=1`. Against an fp64 reference, the new route's rel-L2 is 3.7–4.3e-4, against 6.3–9.1e-4 for the stock route.

## Microbenchmark: µs per layer-call (`kvarn_verify_bench.py`)

| Context | T | Stock | Old `KVARN_SHARED_VERIFY` | **attn-2** | attn-2 + pv16 |
|---|---|---|---|---|---|
| 131k | 4 | 3,252 | 1,780 | **904 (3.6×)** | 876 |
| 131k | 8 | 6,572 | 3,464 | **1,412 (4.7×)** | 1,316 |
| 240k | 1 | 1,916 | – | 1,411 (1.4×) | 1,336 |
| 240k | 4 | 6,717 | 3,777 | **1,638 (4.1×)** | 1,647 |
| 240k | 8 | 12,291 | 7,460 | **2,625 (4.7×)** | 2,452 (5.0×) |

The appendix bar is ≥2× per layer at 240k, T=8: **4.7×**. The config sweep confirms the default `(32, 8, 1)` with 41 splits; the next best (`32, 4, 1`) is 2.0× slower.

## End to end: decode tok/s, two boots each (spill-free)

| Arm | 20k (six tasks) | 170k cold | 170k warm |
|---|---|---|---|
| stock (k=7) | 58.8 / 59.4 | 12.3 / 12.5 | 16.0 / 16.1 |
| **attn-2 (k=7)** | **81.2 / 81.7 (+38%)** | **34.3 / 33.1 (+172%)** | **36.7 / 35.4 (+125%)** |
| attn-2 + pv16 (×1) | 82.9 | 33.6 | 35.3 |
| stock, `DFLASH_TOKENS=3` (×1) | 57.4 | 17.2 | 17.4 |
| attn-2, `DFLASH_TOKENS=3` (×1) | 71.9 | 38.9 | 39.2 |

Tokens per step barely move: at 20k 2.73 → 2.65, at 170k warm 2.53 → 2.14 (one task, 256 tokens; cold has identical steps). The gain is the verify kernel. The appendix bar is +15% at depth: **+125% at 170k warm**. 200k is out of reach on the spill-free pool, which caps max length at 180,224.

**Stacking with spec-0.** With attn-2 on, k=7 is better at 20k (81.5 vs 71.9) and k=3 is better at 170k (+9%: 39.2 vs 36.0). The verify width that was costing E at depth is mostly paid down by this kernel, so `DFLASH_TOKENS=3` stops being a large lever once attn-2 is in.

## Quality

| Check | Stock | attn-2 |
|---|---|---|
| decode KL k3, 4k (8 prompts, two repeats, identical) | 1.122e-3 | 8.77e-4 |
| decode KL k3, 16k (4 prompts) | 7.49e-4 | 6.86e-4 |
| perplexity, all (32,645 tokens) | 8.4016 | 8.4015 (−0.00%) |
| GSM8K greedy, n=200 | 0.965 | 0.955 (b=3, c=1, p=0.63) |
| GSM8K seed 1 / seed 2, n=200 each | 0.930 / 0.960 | 0.925 / 0.930 (sampled b=9, c=4, p=0.27) |
| GSM8K pooled, n=600 | **95.2%** (571) | **93.7%** (562) |
| needle at 100k, depths 0.1 / 0.5 / 0.9 | 3/3 | 3/3 |

**Second pass** (fresh boots, greedy plus 4 sampled seeds per arm, `attn-2\q2_on`, `attn-2\q2_off`):

| GSM8K, n=200 each | greedy | seed 1 | seed 2 | seed 3 | seed 4 | pooled n=1,000 |
|---|---|---|---|---|---|---|
| stock | 0.970 | 0.930 | 0.955 | 0.935 | 0.935 | **94.5%** |
| attn-2 | 0.960 | 0.935 | 0.925 | 0.950 | 0.930 | **94.0%** |

Paired test PASS: greedy b=3, c=1, p=0.63; 4 sampled seeds b=10, c=6, p=0.45; perplexity 8.4015 vs 8.4016. Seed 2 loses in both passes (b=7, c=1, then b=6, c=0) while seeds 1, 3 and 4 are even or ahead, so the first pass's −1.5 points was mostly that one seed.

**Over both passes (n=1,600 per arm): stock 94.75%, attn-2 93.9% (−0.9 points).** That is within ±1 point of stock measured the same way on this box. Against the appendix's fixed 95.2% it is 1.3 points under, because stock itself reads 94.75% here. KL is no worse than stock, needle 3/3 at 100k, perplexity identical. Numerics-only boots, so spill does not apply (`attn-2\kl_*`, `attn-2\q_*`, `attn-2\q2_*`).

## Verdict

**WIN.** 4.7× per layer at 240k T=8 (bar 2×). +125% decode at 170k warm and +38% at 20k, end to end and spill-free (bar +15% at depth). Quality within ±1 point of stock on GSM8K n=1,600, needle 3/3, KL and perplexity unchanged. The knob is `KVARN_VERIFY_SHARED=1`; pv16 adds nothing measurable end to end.

## Against the appendix

- The appendix proposed a pthfa CUDA kernel. A Triton kernel that dequantizes once per request clears both bars by a wide margin; the CUDA kernel stays unmeasured.
- The stock E numbers at depth from before 2026-10-22 15:30 ran on the stock KV pool, which spills during long prefills on this box. Spill-free stock at 170k warm is 16.0 tok/s, against 14.3 on the stock pool.
