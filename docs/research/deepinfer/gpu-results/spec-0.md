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

## Addendum 2026-10-22: spill-free re-check and the 20k depth

The runs above used the stock E pool (`KV_MEM` 4.9 GiB), which pages GPU memory over PCIe during long cold prefills on this box ([ENVIRONMENT.md](ENVIRONMENT.md)). Both checks below ran on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…` (main `ce9615c`) with `KV_MEM=4000000000 DFLASH_MAX_LEN=180224`. Every boot was spill-free by PCIe: job-window median 20–34 MB/s, no second above 500 MB/s. Raw data: `spec-0\s20k_k{3,5,7}_{1,2}`, `spec-0\s150k_k{3,7}_{1,2}`.

**~150k (qa + summary, 256 tokens each), two boots per arm:** k=7 12.9 / 13.0, k=3 19.9 / 19.8 tok/s, **+54%**. The original +56% stands; spill had not inflated it.

**20k (six tasks, 512 tokens), two boots per arm:**

| Task | k=7 | k=5 | k=3 | k=3 vs k=7 |
|---|---|---|---|---|
| copy | 155.2 | 133.5 | 99.5 | −36% |
| code | 92.8 | 93.5 | 82.5 | −11% |
| edit | 52.5 | 51.3 | 56.2 | +7% |
| quote | 48.6 | 42.2 | 47.9 | −2% |
| summary | 40.0 | 50.3 | 52.5 | +31% |
| qa | 38.5 | 43.6 | 49.6 | +29% |
| **TOTAL** | **55.9** (55.7 / 56.1) | **58.2** (58.3 / 58.2) | **60.2** (60.0 / 60.5) | **+7.8%** |

At 20k a flat k=3 still wins the six-task total, but it is a workload trade, not a free default: copy-heavy and code generation lose 11–36%, while QA and summarising gain ~30%.

**With attn-2 in** ([attn-2.md](attn-2.md)), the cheaper verify takes most of this lever away: k=7 beats k=3 at 20k (81.5 vs 71.9), and k=3 keeps only +9% at 170k (39.2 vs 36.0). Whether to pick k=3 at depth (with or without attn-2), or keep k=7 everywhere, is the user's call.
