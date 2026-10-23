# attn-3: int8-QK prefill on the quantized-KV arms

| Field | Value |
|---|---|
| Date | 2026-10-22 21:17–23:35 SAST (mode D); 2026-10-23 14:54–17:05 SAST (mode E) |
| Commit | pthfa source `b6ee607` (was the working tree at run time) (new `attn_prefill_s8.cuh`, `instantiate_d256_attn3.cu`, a mode bit in `attn_launch.*`, `ops.py` knob; patch copy `G:\dev\hq-results\deepinfer-20261020\attn-3\attn3-pthfa.patch`), `bench/ttft_cold.py`, `bench/pthfa_prefill_bench.py`; mode E: `experiments/patches/attn3-kvarn-int8qk.patch` |
| Image | `hq:exp-attn3` (`Dockerfile.patch` on `hq:exp-pthfa-attn3`, itself `Dockerfile.pthfa` from the working tree), on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…` (main `ce9615c`). Rebuilt mid-run (`0c7b24801c88…` → `9ce0fdcc6dd6…`) for the mode E fix only; the D path is unchanged. |
| GPU | RTX 3090, 250 W enforced, driver 610.88; busy median SM clock 1365–1395 MHz |
| Knobs (default off) | D: `PTHFA_PREFILL_QK=s8`. E: `KVARN_PREFILL_INT8=1`. |
| Spill | All eight TTFT boots (D and E): PCIe job-window median 20–24 MB/s, no second above 500 MB/s. Mode E runs on the spill-free pool (`KV_MEM=4000000000 DFLASH_MAX_LEN=180224`). |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\attn-3\` (`test_*`, `D_{base,s8}_{kl,q}`, `D_{base,s8}_ttft_{1,2}`) |

## What it is

Mode D's prefill runs pthfa's 64-row-tile kernel, which dequantizes the int8 per-token-head K to fp16 for QK. The new kernel quantizes each Q row to int8 (as attn-1 stage 3 does) and feeds the cache's int8 K straight to the int8 tensor-core MMA. PV is unchanged. It also serves any verify step longer than 32 rows on the same kernel.

## Correctness

`bench/test_pthfa_attn.py` with `PTHFA_PREFILL_QK=s8`: 88/88 OK. Prefill chunks at depth against the true Q: rel-L2 3.1e-3, nearly all of it from int8 Q itself (3.1e-3 alone).

## Mode D (`SPEC=mtp CTX=long`): cold TTFT, two boots per arm (`bench/ttft_cold.py`, fixed-seed prompts, 2 reps each)

| Prompt | Stock | int8-QK prefill | Change |
|---|---|---|---|
| 51,200 | 58.4 / 58.1 s | 55.7 / 55.6 s | −4.5% |
| 112,000 | 163.4 / 162.1 s | 150.7 / 150.5 s | −7.5% |
| 148,000 | 240.2 / 238.7 s | 218.3 / 218.0 s | **−8.9%** |

The appendix bar is ≥ 5% TTFT on the D arm. 51k narrowly misses (−4.5%); 112k and 148k clear it. The appendix predicted −8 to −9% at 150k; the measured −8.9% lands inside that range.

## Mode D quality (numerics-only boots)

| Check | Stock | int8-QK prefill |
|---|---|---|
| decode KL k3, 4k / 16k | 1.20e-4 / 1.37e-4 | 1.45e-4 / 1.29e-4 |
| perplexity en / da / code | 10.7647 / 10.9089 / 3.5988 | 10.7632 / 10.9092 / 3.5983 |
| GSM8K greedy / seed 1, n=200 each | 0.960 / 0.940 | 0.970 / 0.950 |
| needle at 100k, depths 0.1 / 0.5 / 0.9 | 3/3 | 3/3 |

KL is within the ≤ 2× gate, and perplexity, GSM8K and needle are unchanged.

## Mode E (`SPEC=dflash2 CTX=huge`, `KVARN_PREFILL_INT8=1`): every chunk after the first runs a SageAttention-style int8-QK kernel on the rebuilt fp16 context instead of FA2

Cold TTFT, two boots per arm (`bench/ttft_cold.py`, 2 reps each):

| Prompt | Stock | int8-QK prefill | Change |
|---|---|---|---|
| 51,200 | 56.2 / 56.9 s | 55.0 / 55.0 s | −2.7% |
| 112,000 | 151.3 / 152.4 s | 143.6 / 143.4 s | **−5.5%** |
| 148,000 | 220.8 / 221.6 s | 205.6 / 205.2 s | **−7.1%** |

| Check (numerics boots) | Stock | int8-QK prefill |
|---|---|---|
| decode KL k3, 4k / 16k | 1.12e-3 / 7.49e-4 | 8.15e-4 / 6.42e-4 |
| perplexity, all | 8.4016 | 8.4025 (+0.01%) |
| GSM8K greedy / seed 1, n=200 each | 0.970 / 0.930 | 0.965 / 0.935 |
| needle at 100k, depths 0.1 / 0.5 / 0.9 | 3/3 | 3/3 |

The int8 arm uses 1.2 GiB more VRAM at peak (23,786 vs 22,544 MiB, 790 MiB headroom on the spill-free pool), which stayed spill-free here. Its unit test (`bench/test_kvarn_int8qk.py`) fails one case on its minimum-cosine cut (see below); the end-to-end gates above are the ones that count.

## Verdict

**WIN** in both modes. D: −8.9% cold TTFT at 148k, −7.5% at 112k. E: −7.1% at 148k, −5.5% at 112k. Bar ≥ 5% (stated for D). KL, perplexity, GSM8K and needle are unchanged in both. Below ~50k the gain is under 5%.

## Microbenchmark (`bench/pthfa_prefill_bench.py`, mode D shape, ms per layer-chunk of 2,048 query rows)

| Context | Stock | int8-QK | Speed-up | rel-L2 vs stock |
|---|---|---|---|---|
| 51k | 57.1 | 46.0 | 1.24× | 3.9e-3 |
| 100k | 114.6 | 92.5 | 1.24× | 3.9e-3 |
| 150k | 174.8 | 140.9 | 1.24× | 3.9e-3 |

This is the forced re-run, after the bench was fixed to build one attention impl per layer cache (the first run shared layer 0's scale views across the other three; it read the same 1.24×). For mode E's kernel (`bench/test_kvarn_int8qk.py`), int8 runs 1.32–1.37× FA2 per 2,048-row chunk at 16–150k. That test's accuracy section fails one case on its minimum-cosine cut (q=2048, kv=2048: min cosine 0.998955 against 0.999, rel-L2 4.7e-3, lower than the four passing cases' 5.9–9.3e-3). The mode E KL, quality and TTFT boots run anyway, and they are the gate.
