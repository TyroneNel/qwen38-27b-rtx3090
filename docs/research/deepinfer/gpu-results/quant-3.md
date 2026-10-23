# quant-3: better GPTQ for the int4 DFlash2 drafter

| Field | Value |
|---|---|
| Date | 2026-10-23 18:35–20:12 SAST |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`sha256:0cf71b42d301…`, main `ce9615c`), stock |
| Change | `drafter/gptq_utils.py` `--q3` = signed ±amax/8 scales + MSE clip + act-order with static groups (no `g_idx`; same stored layout); `drafter/quant_dflash2.py --q3`. Unit check `bench/test_gptq_q3.py` (−50% synthetic output error) |
| Hessians | fresh capture with `drafter/capture_dflash2.py` on the `-fast` target, 400 prompts (`gen.jsonl` stand-in), `--ignore-ctx-kv` (the shipped recipe), 10.2 GB `hessians.pt` |
| Arms | **w4a16**: the shipped drafter. **q3ctl**: today's GPTQ on the new Hessians (isolates the Hessian source). **q3**: `--q3` on the new Hessians. Both new drafters 1.2 GB, under `G:\models\Qwen3.8-27B-DFlash2-W4A16-{q3,q3ctl}` |
| Measurement | drafter-2's offline gate on the same capture (9 domains, 94 sequences per mode, ~45k anchors), paired per sequence against w4a16 |
| Scope | the drafter only. The fast variant's lm_head and the MTP head were not requantized (the drafter is the case with the measured gap; see below) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\quant-3\{hess,q3ctl,q3,eval}\` |

## Results

| Arm | mean rel weight error | tau_chain greedy | tau_chain sampling | paired ratio vs w4a16, sampling (CI95) | greedy (CI95) |
|---|---|---|---|---|---|
| w4a16 (shipped) | 0.147 | 3.618 | 3.394 | 1 | 1 |
| q3ctl | 0.1470 | 3.610 | 3.391 | 0.9989 [0.9980, 0.9999] | 0.9979 [0.9940, 1.0015] |
| q3 | **0.1283** | 3.622 | 3.399 | **1.0012 [1.0001, 1.0024]** | 1.0011 [0.9965, 1.0056] |

Per domain, q3/w4a16 at sampling runs from 0.996 (labd) to 1.004 (danish); no domain moves by more than 0.5%.

Against q3ctl (same Hessians), q3 is +0.23% at sampling. The method works as designed: the weight error drops 13% and acceptance moves the right way. The effect is small because there is little left to recover: drafter-2 measured the whole W4A16-vs-bf16 gap at 0.6% on this capture, not the ~5% the appendix's estimate assumed.

## Verdict: LOSS (against the bar)

The bar was +2–3% at default sampling. q3 gives +0.12% (significant offline, invisible in serving), about a fifth to a third of the 0.6% that even a perfect requant could recover. No serving A/B: the queue's A/B gate now requires an offline ratio ≥ 1.02. The `--q3` option is harmless (default-off, same layout) and remains available for requants where a larger gap is measured.
