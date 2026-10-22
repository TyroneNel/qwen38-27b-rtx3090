# attn-1: pthfa verify stages (48-row tiles, PV fp16 accumulate, int8 QK)

| Field | Value |
|---|---|
| Date | 2026-10-22 19:17–19:22 SAST (tests, microbenchmarks); end to end queued |
| Commit | `1d83b12` + working-tree pthfa (`attn_decode.cuh`, `ptx.cuh`, `instantiate_d256_attn1.cu`, `ops.py` knobs; vendor check OK), `bench/test_pthfa_attn.py`, `bench/pthfa_verify_bench.py` |
| Image | `hq:exp-pthfa` (`sha256:dc0ceb8330a6…`, `experiments/docker/Dockerfile.pthfa`), built on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…` (main `ce9615c`) |
| GPU | RTX 3090, 250 W enforced, driver 610.88, ~1,710 MHz at the end of the timed runs |
| Knobs (default off) | stage 1 `PTHFA_TILE48=1`; stage 2 `+ PTHFA_MMA=pv16`; stage 3 `+ PTHFA_MMA=s8pv16` |
| Shape | Hq/Hkv/D = 24/4/256 (the model's full-attention layers), batch 1, int8 per-token-head KV, 21 splits |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\attn-1\` (`test_*`, `bench_T8`, `bench_sweep`) |

## Correctness (`bench/test_pthfa_attn.py`, 84 cases each)

| Mode | Result | Worst err/tol |
|---|---|---|
| stock (`0x0`) | 84/84 OK | 0.51 |
| stage 1 (`0x1`) | 84/84 OK | 0.51, identical to stock (lossless) |
| stage 2 (`0x3`) | 84/84 OK | 0.255 against the re-derived pv16 tolerance |
| stage 3 (`0x5`) | 84/84 OK | 0.238; against the true (unquantized) Q over 40 decode cases: worst rel-L2 3.8e-3 (int8 Q alone 3.4e-3), min cosine 0.999962 |

## Microbenchmark: µs per layer at T=8 (`pthfa_verify_bench.py`, 3 rounds, median)

| Context | Stock | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| 64k | 592 | 487 (1.22×) | 443 (1.34×) | 356 (1.66×) |
| 128k | 1,189 | 972 (1.22×) | 878 (1.35×) | 701 (1.70×) |
| **150k** | 1,407 | 1,147 (1.23×) | **1,042 (1.35×)** | **835 (1.69×)** |

DRAM use rises from 24% to 32% (stage 2) and 40% (stage 3) of the 936 GB/s peak. The appendix bar is ≥ 1.3× per layer at 150k: stage 1 alone misses it (1.23×), and stages 2 and 3 clear it.

## Verify-width sweep (2 rounds, speed-up vs stock)

| T | 64k: s1 / s2 / s3 | 150k: s1 / s2 / s3 |
|---|---|---|
| 1 | 0.98 / 0.97 / 0.97 | 0.99 / 0.98 / 0.98 |
| 4 | 1.00 / 1.03 / 1.02 | 1.00 / 1.01 / 1.02 |
| 5 | 1.00 / 1.06 / 1.20 | 1.00 / 1.07 / 1.20 |
| 6 | 1.39 / 1.51 / 1.76 | 1.39 / 1.50 / 1.76 |
| 7 | 1.22 / 1.34 / 1.69 | 1.23 / 1.35 / 1.69 |
| 8 | 1.22 / 1.33 / 1.67 | 1.23 / 1.35 / 1.67 |

The stages remove the stock kernel's cost step between 5 and 6 query rows (runtime-2's stair). They pay only at T ≥ 6, which in serving is D′ (DFlash2 verify width 8). D (MTP, T=4) gains nothing, as the appendix predicted. Single-token decode steps lose 1–3%.

## Quality of stage 3 (lossy int8 QK), mode D′ (`SPEC=dflash2 CTX=long`, numerics-only boots)

| Check | Stock | Stage 3 (`PTHFA_TILE48=1 PTHFA_MMA=s8pv16`) |
|---|---|---|
| perplexity, all | 8.4029 | 8.4029 |
| GSM8K greedy, n=200 | 0.960 | 0.960 (b=1, c=1) |
| GSM8K seeds 1 / 2, n=200 each | 0.940 / 0.940 | 0.940 / 0.945 (b=6, c=6, p=1) |
| GSM8K pooled, n=600 | 94.7% | 94.8% |
| needle at 128k, depths 0.5 / 0.9 | 2/2 | 2/2 |

`bench/compare_quality.py` PASS. The appendix's stage 3 gates (needles at 50% and 90% of 128k, GSM8K within noise) are met (`attn-1\q_base`, `attn-1\q_s3`).

## End to end

The first end-to-end attempt at 128k failed: D′ serves `max_model_len` 131,072, and labd's 128k prompt overshoots it (HTTP 400). Stock vs stage 2 vs stage 3 at labd 110k (qa + summary) is queued.
