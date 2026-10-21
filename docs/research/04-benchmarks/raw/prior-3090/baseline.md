# baseline: headline numbers per mode on this box

| Field | Value |
|---|---|
| Dates | 2026-10-20 21:45 – 2026-10-21 03:39 (SAST) |
| Commit | `88549e7` / `e6a3ca0` (docs-only between them) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` = `sha256:c696d22e80b4…` (main `2cf50a0`), stock, re-pulled before each phase |
| GPU | RTX 3090, 250 W enforced, driver 610.88, WSL2 (Docker Desktop) |
| Clocks under load | throttle reason `0x4` (SW power cap) only, never thermal; busy-sample median SM clock 1020–1065 MHz in A, B and D, 1365–1455 MHz in C, D′ and E; GPU 70–83 °C |
| `.env` knobs that differ from defaults | the box `.env` minus `VISION`/`CTX`/`SPEC` (documented numbers are text-only), plus each mode's knobs below; `GPU_UTIL=0.93` for A in rounds 1–2 (the box's `.env.bench` value), `PREFIX_CACHE=1` as `.env.example` |
| Context | per mode, below |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\baseline\<mode>_<round>\` (container log, `gpu.csv` 1 s, `gpumem.csv` WDDM 2 s, bench logs, `summary.txt`) |

## Protocol

The repo's own protocol per mode (bench/README.md, run_benchmarks.sh header, single-user/README.md). One fresh container per boot from the stock image, pinned to the 3090 by UUID, gated on 250 W and ≤ 55 °C, `bench/warmup.sh` after `/health`, then:

| Mode | Launch (entrypoint + knobs) | Bench | Headline |
|---|---|---|---|
| A | `batch`, `GPU_UTIL` (see spill below) | `bash bench/run_benchmarks.sh batch`, twice, keep the 2nd | `ROW 64conc 128in/512out` e2e |
| B | `single`, `SPEC=dflash2 PREFIX_CACHE=1` | `bash bench/run_benchmarks.sh single`, twice, keep the 2nd | `ROW cohort C1 … T=0` e2e |
| C | B + `DFLASH_TOKENS=15` | `labd_bench.py C --ctx 20000` (frozen corpus) | `copy` decode |
| D | `single`, `SPEC=mtp CTX=long` (resolves to `PTH_FA=1` on this card) | run_benchmarks single ×2 | `C1 T=0` decode |
| D′ | `single`, `SPEC=dflash2 CTX=long` (resolves to `PTH_FA=1`) | `labd_bench.py --ctx 53000 / 85000 --tasks qa,summary --corpus labd_corpus_long.txt` (t4_battery's d60/d90 cells) | TOTAL decode |
| E | `single`, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` | `labd_bench.py E --ctx 20000` | TOTAL decode, copy decode |

Driver: `G:\dev\hq-results\deepinfer-20261020\tools\di.sh` (`di_boot`), plan `phase1_baseline.sh`, jobs `job_rb.sh` / `job_labd.sh`. Rounds 1 and 2 ran in rotated mode order (B C E D′ D A, then A D D′ E C B).

## Numbers

Two boots per mode. "Doc" is the repo's number; the user confirmed the docs were measured on native Linux, so this box's numbers are the reference for every A/B below.

| Mode | Metric | Round 1 | Round 2 | Median (spread) | Doc (native) | vs doc | WSL2 reference in the repo |
|---|---|---|---|---|---|---|---|
| A | 64conc 128/512 e2e tok/s | 822.9 | 805.6 | **814.3** (805.6–822.9) | 942 | −13.6% | this box's 2026-09-18 campaign 781–858 |
| A | 64conc 128/512 decode (C/medTPOT) | 912 | 896 | 904 | ~1,035 | −12.7% | |
| B | C1 greedy e2e tok/s | 109.2 | 111.6 | **110.4** (109.2–111.6) | 127.1 | −13.1% | 110.7 decode, WSL2 3090 (`docs/reproductions/README.md:64`) |
| B | C1 greedy decode / tok/step | 113.1 / 3.36 | 115.6 / 3.36 | 114.4 | 131.2 | −12.8% | |
| B | C1 default sampling e2e | 107.5 | 110.5 | 109.0 | 118.6 | −8.1% | |
| C | `copy` decode tok/s (tok/step) | 353.2 (14.22) | 361.9 (14.22) | **357.6** | 381 (14.97) | −6.1% | none |
| C | six-task TOTAL | 94.6 | 96.9 | 95.8 | 108 | −11.3% | |
| D | C1 greedy decode (tok/step) | 99.3 (2.64) | 100.2 (2.64) | **99.8** | 100.3 (fp8 arm) | −0.5% | 95.39 (old fp8 arm, `docs/docker.md:153`) |
| D | C1 default decode | 96.7 | 97.3 | 97.0 | 95.3 | +1.8% | |
| D′ | TOTAL decode at ~60k (`--ctx 53000`) | 55.4 | 56.0 | **55.7** | none at TP=1 on a 3090 | – | FA2-fp8 54.0 at ~66k native (README) |
| D′ | TOTAL decode at ~95k (`--ctx 85000`) | 48.4 | 49.2 | **48.8** | none | – | FA2-fp8 48.2 at ~100k native |
| E | TOTAL decode (tok/step) | 57.2 (2.71) | 58.4 (2.71) | **57.8** | 67 (3.15) | −13.7% | 53 TOTAL / copy 130 (`docs/long-context.md:582-584`) |
| E | `copy` decode | 156.7 | 160.4 | 158.6 | 164 | −3.3% | 130 |

KV pools (tokens): A 242,010 at `GPU_UTIL=0.93`, 227,319 at 0.91; B 68,605; C 57,669; D 206,532–207,286; D′ 136,429; E 268,169. All equal the documented pools where one exists.

## Looking for the cause of the >5% gaps (A, B, C, E)

| Suspect | Finding |
|---|---|
| Power limit | 250 W enforced on every boot (gate refuses otherwise); power median 245–250 W under load. |
| Clocks / thermals | Only `SW power cap` throttling (`clocks_event_reasons 0x4`); no thermal or HW slowdown in any boot. Clocks are what 250 W buys. |
| WSL VM clock | WSL clock vs Windows over 124 s: −0.02% (`tools/clockdrift.sh`). The bench README's 8.3% drift does not apply on this box today. |
| VRAM spill (WDDM) | **A at `GPU_UTIL=0.93` spills**: the WSL VM's non-local GPU memory grew 370 → 698 MiB during the bench (A_2). At 0.91 and 0.89 it stays at 370–378 MiB (a static floor in batch mode; single-user modes sit at 92–130 MiB). Throughput is the same at 0.93, 0.91 and 0.89 (805.6 / 812.4 / 810.9), so the spill is not the cause of A's gap, but every later A arm runs at **`GPU_UTIL=0.91`**, the largest spill-free value. B, C, D, D′, E: no growth (≤ 64 MiB). |
| GPU busy fraction | Mean utilisation while clocked up: 92–97% in every mode, so the engine is GPU-bound. |

Conclusion: the gap is the native-vs-WSL2 difference. Where the repo has a WSL2 number for the same arm, this box matches or beats it: B's C1 greedy decode is 114 against 110.7, E's TOTAL is 57.8 against 53 and its copy 158.6 against 130. D matches the native number because its default arm moved to the pthfa kernel, which the native 100.3 (fp8 FlashInfer) did not use.

## Side finding: `SPEC=mtp CTX=long PTH_FA=0` (fp8 FlashInfer) is host-bound on WSL2

Running D's documented arm explicitly (`PTH_FA=0`, fp8 KV on FlashInfer) gives **63–75 tok/s C1** instead of 95–100, with the GPU only **58–64% busy** while clocked up. It is not a spill: at `GPU_UTIL=0.90` (189,795-token pool, 1.2 GB headroom, zero non-local growth) it is the same 64% and 67–70 tok/s. Three boots (Dfp8_1, Dfp8_2, Dfp8_gu90). The pthfa default arm runs the same model at 92% busy and 100 tok/s, so on WSL2 the launcher's `PTH_FA=1` default is worth about +40% for D. Root cause not chased (outside the expedition); the likely suspect is FlashInfer's per-step host planning.

## Verdict

INFO. Baselines for this box are recorded with a ±1–2% round-to-round spread. A, B, C and E sit 6–14% under the native-Linux docs, explained by WSL2 (not power, clocks, clock drift or spill). Mode A's later arms use `GPU_UTIL=0.91`.
