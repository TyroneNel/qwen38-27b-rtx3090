# attn-3: int8-QK prefill on the quantized-KV arms

| Field | Value |
|---|---|
| Date | 2026-10-22 21:17–23:35 SAST (mode D); mode E queued |
| Commit | `906aedf` + working-tree pthfa (new `attn_prefill_s8.cuh`, `instantiate_d256_attn3.cu`, a mode bit in `attn_launch.*`, `ops.py` knob; patch copy `G:\dev\hq-results\deepinfer-20261020\attn-3\attn3-pthfa.patch`), `bench/ttft_cold.py`, `bench/pthfa_prefill_bench.py`; mode E: `experiments/patches/attn3-kvarn-int8qk.patch` |
| Image | `hq:exp-attn3` (`Dockerfile.patch` on `hq:exp-pthfa-attn3`, itself `Dockerfile.pthfa` from the working tree), on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…` (main `ce9615c`). Rebuilt mid-run (`0c7b24801c88…` → `9ce0fdcc6dd6…`) for the mode E fix only; the D path is unchanged. |
| GPU | RTX 3090, 250 W enforced, driver 610.88; busy median SM clock 1365–1395 MHz |
| Knobs (default off) | D: `PTHFA_PREFILL_QK=s8`. E: `KVARN_PREFILL_INT8=1`. |
| Spill | All four TTFT boots: PCIe job-window median 20–23 MB/s, no second above 500 MB/s |
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

## Verdict (mode D)

**WIN** for D: −8.9% cold TTFT at 148k and −7.5% at 112k (bar ≥ 5%), with no quality cost. Mode E (`KVARN_PREFILL_INT8=1`, which swaps the FA2 call on the rebuilt fp16 context for a SageAttention kernel) failed its first unit test on shared memory (131,592 B against sm86's 101,376). The fix (2 pipeline stages, 82,436 B) is queued for its test, KL, quality and TTFT runs.

## Not valid

The first `bench_prefill` microbenchmark (1.24× per layer-chunk at 50k) shared one layer's scale views across four layer caches, a bug in the bench. Its forced re-run is queued.
