# quant-2: lossless negative-scale fold (fix A)

| Field | Value |
|---|---|
| Date | 2026-10-21 (tests, perplexity) and 2026-10-23 09:20–12:11 SAST (throughput) |
| Commit | `experiments/patches/quant2-lossless-fold.patch`, `bench/test_marlin_int8_fold.py`, `bench/test_marlin_int8_asym.py` (has_g_idx fix) |
| Image | `hq:exp-quant2` (`sha256:d3dd059cd6b2…` for the throughput rounds), `Dockerfile.patch` on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…` |
| GPU | RTX 3090, 250 W enforced, driver 610.88 |
| Knob (default off) | `VLLM_MARLIN_INT8_LOSSLESS_FOLD=1`: at load, negative-scale groups are remapped to uint4 + zero point (code′ = 15 − code, zp 7, \|s\|) and served on the `kS8 × kU4` zero-point kernel instead of folding the sign into the codes |
| Mode | A (`batch`, `GPU_UTIL=0.91`, `INT8_ACT=int8`, `INT8_LAYERS=mlp`) |
| Spill | Throughput rounds are speed-only boots, spill-free by PCIe (median 41–44 MB/s, no second above 500 MB/s). The first throughput reading (−1.2%) shared its boot with the perplexity pass, which spilled. |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\quant-2\` (`tests`, `{off,on}_{1..3}` perplexity, `tp_{off,on}_{1..3}` throughput) |

## Correctness

`bench/test_marlin_int8_fold.py` on the GPU: the remap is bit-identical to a direct asymmetric dequant (max |d| 0) on every shape. Kernel rel-L2 against the true weights falls from 0.0264–0.0266 (sign fold) to 0.0090–0.0091 (remap); the worst remap error over all cases is 0.0112. PASS.

## Perplexity (mode A, 32,645 tokens, three boots per arm, identical across boots)

| | Stock fold | Lossless remap |
|---|---|---|
| en | 10.9619 | 10.8851 |
| da | 11.1327 | 11.0136 |
| code | 3.7011 | 3.6531 |
| **all** | **8.5831** | **8.4976 (−1.0%)** |

This confirms quant-2-check: the sign fold costs about 1% of INT8_ACT perplexity, and the remap recovers it.

## Throughput (64 concurrent, 128 in / 512 out, aggregate e2e tok/s; second `run_benchmarks.sh batch` run of each boot)

| Round | Stock | Remap | Change | Busy SM clock (stock / remap) |
|---|---|---|---|---|
| 1 | 856.8 | 848.0 | −1.0% | 1035 / 1005 MHz |
| 2 | 861.6 | 806.4 | −6.4% (−3.8% against the mean of the neighbouring stock boots) | 1035 / 1005 |
| 3 | 813.8 | 802.8 | −1.4% | 1035 / 990 |

256 in / 256 out: −1.1%, −6.7%, −1.7%. The stock arm itself drifted 862 → 814 between rounds 2 and 3 at the same clock, so round 2 is inflated by drift. The remap arm runs 30–45 MHz (~3%) lower in every boot at the same 250 W draw: the zero-point path spends more energy per token, and at the power cap that becomes clock.

## Verdict: LOSS (against the bar)

The appendix's win is any perplexity drop with zero-point-path throughput within 1%. Perplexity drops 1.0%, but throughput lands at −1.0 to −1.4% in the two clean rounds (−1.4% median across all three), just outside the bar, and the clock gap says the cost is structural, not noise. As a trade it is close: −1% perplexity for about −1.4% mode A throughput. Fix B (a kernel rebuild reading the scales as `int16_t`) would remove the fold without the zero-point path and was not tried.
