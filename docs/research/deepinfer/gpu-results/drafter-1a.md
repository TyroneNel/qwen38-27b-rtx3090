# drafter-1a: host-side confidence trim of the DFlash2 verify length (stage 0)

| Field | Value |
|---|---|
| Date | 2026-10-22 08:35–09:47 SAST |
| Commit | `49b6d9c` + `experiments/patches/drafter1a-conf-trim.patch` |
| Image | `hq:exp-drafter1a` (`sha256:f7570c26e2be…`), built on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…` (main `ce9615c`) |
| GPU | RTX 3090, 250 W enforced, driver 610.88; busy median SM clock 1395–1410 MHz, temperature 81–82 °C median in every boot; VRAM peak 23.3–23.4 GiB, no spill |
| `.env` knobs | `SPEC=dflash2 CTX=long` (D′). A0: stock. T1: `VLLM_DFLASH2_CONF_TRIM=1 VLLM_DFLASH2_VERIFY_MS_PER_TOKEN=1.9` (fresh confidence, draft counts 2–7). T1s: T1 plus `VLLM_DFLASH2_CONF_TRIM_LENS=4,7` (two widths only). A1: stock with `ASYNC_SCHED=0`, the synchronous-scheduling control the trim needs. |
| Context | `bench/labd_bench.py --ctx 100000` on the long corpus (prompts ~108.7k tokens), all six tasks, 512 tokens, thinking off |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\drafter-1a\{A0,T1,T1s}_{1,2},A1_1\` |

## Protocol

Arms rotated A0, T1, T1s for two rounds, then one A1 boot. The slope of 1.9 ms per verify token comes from runtime-2's measurement at ~112k (q 4→8 stair), not the 3.6 ms the appendix assumed. The trim engages as designed: at ~109k, T1 spread its steps over 2–7 drafts (`{2: 226, 3: 172, 4: 143, 5: 98, 6: 74, 7: 371}`), and T1s split them 4 vs 7 at about 2:3.

## Numbers: decode tok/s (median of boots; tok/step in brackets)

| Task | A0 stock (×2) | A1 sync (×1) | T1 (×2) | T1s (×2) |
|---|---|---|---|---|
| copy | 172.0 (7.64) | 168.7 (7.64) | 164.4 (7.64) | 162.1 (7.64) |
| code | 103.5 (4.67) | 103.4 (4.67) | 102.0 (4.67) | 101.3 (4.67) |
| edit | 60.1 (2.67) | 60.4 (2.68) | 61.0 (2.67) | 61.3 (2.67) |
| quote | 46.3 (2.06) | 44.8 (2.00) | 47.0 (2.01) | 46.2 (2.00) |
| summary | 42.8 (1.92) | 45.6 (2.05) | 44.1 (1.92) | 45.4 (1.97) |
| qa | 45.2 (2.02) | 44.8 (2.00) | 44.5 (1.91) | 48.0 (2.08) |
| **TOTAL** | **60.65** (60.6 / 60.7) | **61.0** | **60.9** (60.9 / 60.9) | **62.1** (62.4 / 61.8) |

Against A0: T1 +0.4%, T1s +2.4%. Against the sync control A1: T1 −0.2%, T1s +1.8%. Copy loses 4–6% under the trim because synchronous scheduling and the fresh-confidence `.cpu()` cost host overlap, and copy steps (lookup blocks) are never trimmed.

## Verdict: LOSS

The appendix's win needs ≥ +8% on the all-six TOTAL at 100k. The best arm reaches +1.8–2.4%: outside the A0 spread (0.2%), but a quarter of the bar. The greedy hash and 8k regression checks were not run, because the speed criterion already fails.

## Against the appendix

- The appendix's +11–20% at 112k assumed a 3.6 ms verify slope. runtime-2 measured 1.2 ms per token for q ≤ 4 and a 1.9 ms stair from 4 to 8 at 112k, so the saved verify tokens are worth about half what the simulation priced, and the step-count cost of trimming eats most of that. The ~64k row's +3.8–4.8% is the closer prediction, and the measured +2% sits just under it.
- Two widths (4, 7) beat the full 2–7 menu. Trimming below 4 saves nothing (verify cost is flat for q ≤ 4) and still loses accepted tokens.
- The trim needs synchronous scheduling, which costs ~2% on copy-heavy text by itself. A device-side trim (spec-2) would avoid that, but spec-1, which spec-2 depends on, came out INVALID.

## Consequence

drafter-1b (learned confidence head) is gated on a drafter-1a or spec-2 win. Neither happened, so drafter-1b is skipped.
