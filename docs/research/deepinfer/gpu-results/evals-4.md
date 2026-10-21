# evals-4: thinking-on speed and acceptance

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | tools in this result's commit: `bench/conc_ladder.py` (`--think`, `--levels`, `--jsonl`), `bench/labd_accept_think.py` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88; busy median SM clock 1065 MHz (B), 1350 MHz (D); 98% busy; no VRAM spill (non-local flat at 92 / 106 MiB) |
| `.env` knobs | B: `SPEC=dflash2 PREFIX_CACHE=1`; D: `SPEC=mtp CTX=long` (pthfa default). Chat template default effort (xhigh) when thinking is on |
| Context | conc_ladder: ~4,096-token prompts; labd_accept: 20k (thinking off), 8k (thinking on) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\evals-4\{B,D}\` (`ladder.jsonl`, `ladder_*.txt`, `accept_*.txt`, `accept_think.json`) |

## Commands (inside each mode's container, after `bench/warmup.sh`)

```bash
venv/bin/python bench/conc_ladder.py --levels 1,2,4,8 --salt di7 --label B_nothink --jsonl /out/ladder.jsonl
venv/bin/python bench/conc_ladder.py --levels 1,2,4 --salt di7 --think --out 1024 --label B_think --jsonl /out/ladder.jsonl
venv/bin/python bench/labd_accept.py B_acc --ctx 20000 --block 7          # D: --block 3
venv/bin/python bench/labd_accept_think.py B_think --ctx 8000 --block 7   # D: --block 3
```

One boot per mode; within a boot the off and on arms use the same prompts (`--salt di7`), so the ladder comparison is paired.

## Numbers

Paired ladder (same prompts, thinking off vs on), per-stream decode tok/s:

| Mode | N | Thinking off | Thinking on | Δ | tok/step off → on |
|---|---|---|---|---|---|
| B (DFlash2) | 1 | 117.6 | **100.2** | **−14.8%** | 3.49 → 3.14 |
| B | 2 | 74.6 | 83.3 | +11.7% | 3.47 → 3.45 |
| B | 4 | 42.2 | 50.0 | +18.5% | 3.52 → 3.10 |
| D (MTP) | 1 | 105.3 | **95.1** | **−9.7%** | 2.80 → 2.67 |
| D | 2 | 75.8 | 81.7 | +7.8% | 2.84 → 2.64 |
| D | 4 | 43.9 | 57.5 | +31.0% | 2.90 → 2.59 |

At default (xhigh) effort **every request spent all of its 1,024 tokens reasoning**: 0 of the streams reached `</think>`. The labd_accept_think runs (4,096-token targets) also never reached the answer phase.

Teacher-forced acceptance (labd_accept method, different prompt sets for off and on, so not paired):

| Mode | Thinking off, 6 labd tasks @20k: tok/step (decode) | Thinking on, 4 tasks @8k, reasoning only: tok/step (decode) | accept@pos 0..3, thinking on |
|---|---|---|---|
| B | 3.02 (91.5 tok/s) | 4.42 (140.5) | 0.82 / 0.65 / 0.52 / 0.44 |
| D | 3.02 (103.5) | 3.22 (114.2) | 0.86 / 0.73 / 0.62 |

Per task with thinking on (B): qa 3.18, math 4.93, plan 3.92 tok/step. Math reasoning drafts best.

## Reading

- **The headline numbers are optimistic for a default client at C1.** The repo measures C1 with thinking off. A default request (thinking on, xhigh) decodes 15% slower on B and 10% slower on D at one stream, because the reasoning text drafts worse on these 4k prompts (3.49 → 3.14 tok/step on B).
- **At 2–4 streams thinking-on is faster per stream.** Thinking-on requests run 1,024 tokens against 256 for thinking-off. Longer requests keep the batch full for longer, so less of the run is spent on the ragged tail and on prefill. That is a protocol effect, not an acceptance effect: tok/step is lower or equal with thinking on at every N.
- **Acceptance depends on the content more than on the phase.** On labd's long-context tasks, reasoning text accepts better than answer text (B 4.42 vs 3.02 tok/step), and math reasoning best of all (4.93). On the ladder's 4k prompts it accepts worse. A per-domain panel (evals-9) is the right instrument for this. The 45% thinking-on share of the drafter's training data (drafter.md) fits the strong reasoning-phase acceptance on labd.
- **No answer-phase number was obtainable at xhigh with ≤4,096 tokens.** The split columns are therefore reasoning only. Measuring the answer phase needs `reasoning_effort` set lower or a much larger budget. Neither matches what a default client sends, so it was not run.

## Verdict

INFO. The appendix's suspicion is confirmed in direction, and it is moderate in size. At C1 the speed a default (thinking-on) client sees is 10–15% under the thinking-off headline, on prompts like the ladder's.
