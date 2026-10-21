# evals-10: throughput against per-user speed, per mode

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | `bench/conc_ladder.py` with `--levels` and `--jsonl` (committed with evals-4) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88; A: busy median SM clock 1125 MHz, 99% busy, WDDM non-local flat (370 → 378 MiB) |
| `.env` knobs | A: `batch`, `GPU_UTIL=0.91` (largest spill-free value, baseline.md); B: `SPEC=dflash2 PREFIX_CACHE=1`; D: `SPEC=mtp CTX=long` |
| Context | ~4,096-token distinct prompts, 256 output tokens, greedy, thinking off, `--salt di7`, 2 reps per N (best kept) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\evals-10\A\pareto.{txt,jsonl}`, `…\evals-4\{B,D}\ladder.jsonl` |

## Commands

```bash
venv/bin/python bench/conc_ladder.py --levels 1,2,4,8,16,32,64 --salt di7 --label A_pareto --jsonl /out/pareto.jsonl   # A
venv/bin/python bench/conc_ladder.py --levels 1,2,4,8 --salt di7 --label B_nothink --jsonl /out/ladder.jsonl         # B, D (MAX_SEQS 8)
```

## Numbers

Per-stream decode tok/s, with the decode aggregate (sum over streams while all are decoding) in parentheses:

| N | A (batch, no speculation) | B (DFlash2, 64k) | D (MTP, 150k) |
|---|---|---|---|
| 1 | 44.6 (44) | 117.6 (118) | 105.3 (105) |
| 2 | 40.2 (83) | 74.6 (182) | 75.8 (192) |
| 4 | 30.1 (155) | 42.2 (253) | 43.9 (308) |
| 8 | 20.0 (274) | 24.1 (236), 19 preemptions | 22.9 (423) |
| 16 | 12.2 (450) | – | – |
| 32 | 6.8 (647), 2 preemptions | – | – |
| 64 | 4.8 (586*), 5 preemptions | – | – |

\* At N=64 the 227k-token pool can't hold 64 × ~4.4k tokens, so there is never an instant with all 64 decoding; that row is a tail measurement. A's step time rises from 22.5 ms (N=1) to 49.4 ms (N=32).

## Reading

- **The Pareto frontier belongs to D, not B, from N=2 up.** At 2–8 users, MTP at `CTX=long` gives more aggregate throughput and equal or better per-user speed than DFlash2: at N=8, 423 against 236 tok/s. B's 64k pool (68,605 tokens) holds only ~7 such requests at full speculative state, so it preempts at N=8. This is the residency limit spec-3 targets.
- **B wins only at N=1** (117.6 against 105.3).
- **A's knee is at N=16–32** with 4k prompts: 450 → 647 tok/s aggregate, at 12 → 7 tok/s per user. Above that the pool, not compute, limits it. The headline 64-concurrency number (814 e2e, baseline.md) uses 128-token prompts, which fit.
- A at N=1 (44.6 tok/s) is what a single user gets on the batch profile without speculation: 2.6× slower than B.

## Verdict

INFO. The 8–64 range now has a curve. The few-user range (N=2–8) favours mode D's MTP over B's DFlash2, which bears on the README's open question 1 (multi-seat speculation): today's best multi-seat profile is `SPEC=mtp CTX=long`.
