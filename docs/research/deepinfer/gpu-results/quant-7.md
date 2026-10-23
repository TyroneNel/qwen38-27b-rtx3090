# quant-7: int4 token embedding where `EMBED_UVA` can't be used

| Field | Value |
|---|---|
| Date | 2026-10-22 09:49 – 2026-10-23 09:45 SAST |
| Commit | `experiments/quant7_emb4.py` (builds the checkpoint: embedding BITS 8 → 4, all other tensors hardlinked from `-fast`) |
| Model | `G:\models\Qwen3.8-27B-W4A16-AutoRound-fast-emb4` against `…-fast` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`sha256:0cf71b42d301…`, main `ce9615c`), stock |
| GPU | RTX 3090, 250 W enforced, driver 610.88; busy median SM clock 1035 MHz (D), 1410–1440 MHz (E) |
| Spill | Speed numbers below come only from spill-free boots: D `Ds_*` (PCIe median 46 MB/s), E `E_*_{2,3}` (28–29 MB/s). The first D and E boots shared their boot with the KL probe and quality battery and are used for quality only. |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\quant-7\` |

## Mode D (`SPEC=mtp CTX=long`), `run_benchmarks.sh single`, two boots per arm

| | Stock | int4 embedding | Change |
|---|---|---|---|
| KV pool (tokens) | 207,286 | 223,869 | **+8.0%** |
| C1 greedy tok/s | 98.2 / 98.3 | 97.3 / 98.2 | −0.5% |
| C1 default temperature | 95.2 / 95.5 | 96.3 / 97.1 | +1.3% |
| C4 greedy tok/s | 269.4 / 268.8 | 263.7 / 264.8 | −1.9% |
| tok/step, greedy | 2.64 | 2.62 | −0.8% |

## Mode E (`SPEC=dflash2 CTX=huge`), labd 20k, two boots per arm, spill-free pool

The int4 arm gets the freed embedding memory as extra KV: `KV_MEM` 4,000,000,000 → 4,635,699,208, the same increment as on the stock pool.

| | Stock | int4 embedding | Change |
|---|---|---|---|
| KV pool (tokens) | 182,903 | 212,382 | **+16.1%** |
| decode tok/s, six tasks | 56.9 / 57.0 | 56.3 / 55.9 | −1.4% |
| tok/step | 2.73 | 2.69 | −1.5% |

The drafters (MTP in D, DFlash2 in E) read the same embedding, and their acceptance drops ~1%; that accounts for most of the speed cost.

## Quality (mode D, round 1 boots)

| Check | Stock | int4 embedding |
|---|---|---|
| perplexity, all | 8.4018 | 8.4134 (+0.14%) |
| decode KL k3, 4k | 1.20e-4 | 1.41e-4 |
| GSM8K greedy / seed 1, n=200 | 0.960 / 0.930 | 0.950 / 0.925 (paired PASS: b=3, c=1; b=5, c=4) |

## Verdict: WIN (capacity)

The appendix's claim, from community numbers, was +8.4% pool with no perplexity or GSM8K change. Measured: +8.0% pool in D and +16% in E, perplexity +0.14%, GSM8K within noise, KL within the 2× gate. The cost is 0.5–2% decode speed from slightly lower drafter acceptance. As the appendix says, `EMBED_UVA=1` (+15.7% pool, decode unchanged) dominates where 2.1 GiB of pinned host RAM is available; this is the option when it is not.
