# quant-5: batch-mode lm_head as W4A8

| Field | Value |
|---|---|
| Date | 2026-10-22 12:01 – 2026-10-23 08:00 SAST |
| Commit | no code: launcher knob `INT8_LAYERS` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`sha256:0cf71b42d301…`, main `ce9615c`), stock |
| GPU | RTX 3090, 250 W enforced, driver 610.88; busy median SM clock 975–1080 MHz |
| `.env` knobs | mode A (`batch`), `GPU_UTIL=0.91`, `MODEL=…-fast` (int4 GPTQ head). A1: stock `INT8_LAYERS=mlp`. A2: `INT8_LAYERS=mlp\|lm_head` (the head joins the int8-activation Marlin path) |
| Protocol | `run_benchmarks.sh batch` twice per boot (second run kept), A1/A2 interleaved. Rounds 1–2 ran the KL probe and quality battery in the same boot, and both arms spilled ~300 MiB to system RAM there. Rounds 3–5 are speed-only boots, spill-free by PCIe (median 42–43 MB/s). |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\quant-5\A{1,2}_{1..5}` |

## Throughput: 64 concurrent, 128 in / 512 out (aggregate e2e tok/s)

| Round | A1 stock | A2 int8 lm_head | Change |
|---|---|---|---|
| 1 (probe in boot) | 819.3 | 819.0 | 0.0% |
| 2 (probe in boot) | 821.6 | 848.0 | +3.2% (A2 boot ran 67 MHz faster) |
| 3 | 833.9 | 841.8 | +0.9% |
| 4 | 844.5 | 851.6 | +0.8% |
| 5 | 859.7 | 853.9 | −0.7% |

The spill-free rounds give +0.8% median, inside the ±1% boot-to-boot spread (the stock arm alone moves 834 → 860 across rounds). 256 in / 256 out follows the same pattern (+1.0 / +0.8 / −0.5%).

## Quality (round 1 boots)

| Check | A1 | A2 |
|---|---|---|
| perplexity, all | 8.6065 | 8.5903 (−0.19%) |
| decode KL k3, 4k | 3.39e-3 | 3.52e-3 |
| GSM8K greedy / seed 1, n=200 | 0.955 / 0.935 | 0.965 / 0.950 |

`compare_quality.py` PASS (greedy b=2, c=4; sampled b=1, c=4). The int8 path is active (perplexity and KL both move), and it costs nothing measurable in quality.

## Verdict: NEUTRAL

The appendix estimated +2–3% in mode A, from the head's 2.7 ms of compute against 1.4 ms of weight reads at 64 concurrency. The measured change is +0.8% at the median, inside noise. Quality is unchanged, so `INT8_LAYERS=mlp|lm_head` is harmless but not worth a default change.
