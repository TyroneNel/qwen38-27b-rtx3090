# evals-3: paired, auditable quality battery

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | tools in this result's commit: `bench/quality_battery.py` (`--jsonl`, `--seeds`, `--temperature`), `bench/compare_quality.py` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88; busy median SM clock ~1425 MHz; no spill |
| `.env` knobs | A: `SPEC=dflash2 PREFIX_CACHE=1` (mode B). B: the same plus `INT8_ACT=int8 INT8_LAYERS=.` (all linears int8 activations) |
| Data | `QUALITY_DATA=/app/models/quality-data` (wikitext-2, fineweb-2 dan_Latn, gsm8k test, the repo's sets) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\evals-3\{B_stock,B_int8all}\quality.jsonl`, `compare_stock_vs_int8all.json` |

## Commands

```bash
# inside each arm's container
venv/bin/python bench/quality_battery.py B_stock --jsonl /out/quality.jsonl --seeds 2
# CPU afterwards
python bench/compare_quality.py B_stock/quality.jsonl B_int8all/quality.jsonl --out compare.json
```

The validation pair is the appendix's "known pair": single-user all-layer int8 activations, documented at +4.1% perplexity and GSM8K 95.0 against 96.5 (`docs/optimizations.md:140-143`).

## Numbers

| | Stock (A) | All-linear int8 (B) | Paired test |
|---|---|---|---|
| Perplexity, all (en / da / code) | 8.401 (10.77 / 10.91 / 3.60) | 8.737 (11.04 / 11.41 / 3.78) | **+4.00%** (+2.6% / +4.6% / +5.0%) |
| GSM8K greedy, n=200 | 95.5% [91.7, 97.6] | 96.0% [92.3, 98.0] | McNemar b=2, c=3, p=1.00 |
| GSM8K sampled, seed 1 | 94.5% | 93.5% | b=6, c=4, p=0.75 |
| GSM8K sampled, seed 2 | 95.5% | 91.5% | b=11, c=3, p=0.057 |
| GSM8K sampled, pooled per item | 95.0% | 92.5% | sign test b=14, c=6, **p=0.115** |
| Items hitting `finish_reason=length` (greedy) | 8 of 200 | 7 of 200 | |

Wilson 95% intervals are in brackets; b counts items only A got right, c items only B got right.

## Reading

- **The perplexity side reproduces the documented pair**: +4.0% here against +4.1% in the docs.
- **The greedy GSM8K side does not separate the arms.** The docs' 96.5 → 95.0 drop is three questions, inside one run's noise; here the int8 arm happens to win by one question. That is the appendix's point: n=200 greedy GSM8K cannot see a 1–2-point change.
- **The sampled arms point the right way.** Pooled over two seeds, int8 loses 14 items against 6 gained (p=0.115). That is not significant at n=200 × 2 seeds, but it is the only GSM8K view that leans the same way as perplexity. More seeds would tighten it: at this effect size, about 6 seeds would reach p<0.05 (inferred).
- **The old battery hid two things that the JSONL now shows.** 4% of greedy answers stop at `max_tokens` without a "Final answer:" line and are scored by the last-number fallback. Item 12, 87, 93 and 119 do so in every arm and seed: these are budget problems, not model regressions. `compare_quality.py` lists them as warnings.

## Verdict

INFO. The tool works and gives a verdict on the known pair that matches the docs (perplexity +4.0%, PASS on the greedy GSM8K gate, a non-significant sampled-arm loss). Use perplexity as the gate for small lossy changes, the sampled pooled arm as the GSM8K signal, and greedy GSM8K only as a garbage detector.
