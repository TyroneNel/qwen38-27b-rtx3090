# spec-0: context-aware fixed verify width in mode E

| Field | Value |
|---|---|
| Date | 2026-10-21 18:38–20:19 SAST |
| Commit | `6b92a3a` (no code: launcher knob `DFLASH_TOKENS` only) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88; busy median SM clock 1425–1440 MHz in every boot; WDDM non-local growth 14–142 MiB (no spill effect) |
| `.env` knobs | `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` plus `DFLASH_TOKENS=3`, `5` or `7` (7 is the default) |
| Context | labd `qa` + `summary` on the long corpus at `--ctx 90000 / 150000 / 235000` (prompts 152,877 and 227,261 tokens at the two deeper points), 256 tokens each |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\spec-0\k{3,5,7}_{1,2}\labd_*.txt` |

## Protocol

Arms rotated per round (7 5 3, then 5 3 7). There were two complete rounds: the third round was stopped by the user partway through (`k3_3` cut mid-run, `k7_3` and `k5_3` never started), so round 3 is excluded. The spread inside each arm is ≤ 1%, far below the effect, so two boots meet the rule (three boots only for speed claims under 10%).

```bash
venv/bin/python bench/labd_bench.py k3_90000 --ctx 90000 --tasks qa,summary --max-tokens 256 --corpus /cache/bench/labd_corpus_long.txt
# then --ctx 150000 and --ctx 235000 in the same boot (prefix cache reuses the shared document prefix)
```

## Numbers: TOTAL decode tok/s (two boots each)

| `DFLASH_TOKENS` | ~90k | ~150k | ~235k |
|---|---|---|---|
| 7 (default) | 18.5 / 18.6 | 13.2 / 13.3 | 9.4 / 9.5 |
| 5 | 22.2 / 22.2 (+20%) | 17.2 / 17.2 (+30%) | 11.7 / 11.8 (+24%) |
| **3** | **30.5 / 30.7 (+65%)** | **20.6 / 20.7 (+56%)** | **16.1 / 16.1 (+70%)** |

Steps per 512 output tokens hardly move with k (e.g. at 150k: k=7 135 + 129 steps, k=3 135 + 139). At these depths DFlash2 rarely gets past the third draft position, so dropping drafts 4–7 costs almost no acceptance. It saves 4 verify tokens at 14–22 ms each (runtime-2).

## Against the appendix

- The simulation predicted k=3 at −2.5 to +12.5% at 240k, depending on a cost per position c_tok of 1–2 ms. The measured gain is +70%, because KVarN's real cost per verify token is 13.9–22.3 ms at 150–240k (runtime-2), an order of magnitude above the simulated range.
- No refusal: a *fixed* lower k is allowed at `CTX=huge` with prefix caching, as the appendix said.

## Verdict

**WIN.** The appendix's criterion is ≥ +5% at 240k with no loss beyond 2% at 90k. Measured: +70% at ~235k, +56% at ~150k and +65% at ~90k for `DFLASH_TOKENS=3` against the default 7. Two interleaved boots per arm, spread ≤ 1%.

Not yet measured: the 20k headline depth, where long drafts pay on copy-heavy text (E's `copy` task runs at 7.6 tok/step). A k=3/5/7 run at 20k is queued, so the user can choose between a flat default of 3 and a context-aware default (3 above some depth, 7 below). The knob already exists, so the change is a launcher default, not code.
