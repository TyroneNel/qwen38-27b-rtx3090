# Baseline: RTX 3090, prior measured runs

Every number in the tables is **measured (prior, GPU box)**: copied from the prior DeepInfer expedition's write-ups, which are stored verbatim in [raw/prior-3090/](raw/prior-3090/) (provenance: [PROVENANCE.md](raw/prior-3090/PROVENANCE.md), branch `origin/research/deepinfer` commit `86ce6ed`). This lane measured nothing; the research box has no GPU. "Source" links point at the raw copy with `file:line`. Anything marked **estimated** or **inferred** is the prior author's label, kept. The harness in [harness/](harness/README.md) fills the gaps listed at the end.

## Method and environment (prior, as recorded)

| Item | Value (measured, prior, GPU box) | Source |
|---|---|---|
| GPU | RTX 3090 24 GB, 250 W enforced, driver 610.88, WSL2 + Docker Desktop (Engine 29.8.2) | [ENVIRONMENT.md:9-16](raw/prior-3090/ENVIRONMENT.md) |
| Image | stock `ghcr.io/syv-ai/hyperqwen:latest` = `sha-2cf50a0` (main `2cf50a0`), launchers baked in | [baseline.md:7](raw/prior-3090/baseline.md), [ENVIRONMENT.md:24](raw/prior-3090/ENVIRONMENT.md) |
| Clocks | throttle `0x4` (SW power cap) only; busy SM median 1020-1065 MHz in A/B/D, 1365-1455 MHz in C/D'/E; 70-83 C | [baseline.md:9](raw/prior-3090/baseline.md) |
| Protocol | one fresh container per boot, gated on 250 W and 55 C or cooler, `bench/warmup.sh`, two boots per mode (rounds rotated), the repo's own commands | [baseline.md:16-27](raw/prior-3090/baseline.md) |
| Repeats | 2 boots per mode, spread 1-2% round to round; ladder runs 2 reps, best kept | [baseline.md:31,69](raw/prior-3090/baseline.md), [evals-10.md:10](raw/prior-3090/evals-10.md) |
| Prompts | `run_benchmarks.sh` cohorts (128/512 at 64 conc for A; C1 cohort for B and D), `labd_bench` frozen corpus at 20k (C, E), 53k and 85k (D'), conc_ladder 4k distinct prompts | [baseline.md:20-25](raw/prior-3090/baseline.md), [evals-10.md:10](raw/prior-3090/evals-10.md) |

### Timing validity (orchestrator question)

Did the prior runs use the fixed WSL2 timing (373a695 / #351, `time.perf_counter`)? **Most likely yes (estimated from commit order and the box setup), not proven by a raw log.** [review: upgraded by git, not by a box log. The commits this baseline records (`88549e7` / `e6a3ca0`, [baseline.md:6](raw/prior-3090/baseline.md)) contain 373a695 (`git merge-base --is-ancestor 373a695 88549e7` = true), and `git show e6a3ca0:bench/labd_bench.py` has `time.perf_counter()` at lines 85, 95, 97. The stock image's `2cf50a0` also contains 373a695. Only "was the box's working tree at that commit" is still unproven (ledger row `prior-timing-check`).]

- The box ran the stock image (launchers baked in) and bind-mounted only `bench/` from `research/deepinfer` over `/app/bench` ([ENVIRONMENT.md:26](raw/prior-3090/ENVIRONMENT.md)). That branch merged upstream `main` (`63f7581`, which contains 373a695) in `88549e7` on 2026-10-20, before the baseline phase (dates [baseline.md:5-6](raw/prior-3090/baseline.md); ancestry per [upstream-diff.md](../01-project/upstream-diff.md) review note at line 128).
- The tools behind the headline numbers use `perf_counter`: `bench/conc_ladder.py` (our checkout, `conc_ladder.py:124,131,161,174-183`), `warmup_probe.py` (`vendor/warmup_probe.py:43,49,65,70`), and `vllm bench serve` client durations. `labd_bench.py` in our checkout at `95ac251` still uses `time.time()` (`bench/labd_bench.py:85,95,97`); the prior box's bind-mounted copy came from the branch that has the fix.
- Cross-check from the prior data itself: the WSL VM clock against Windows over 124 s was -0.02%, so the 8.3% fast-clock drift in `bench/README.md` [review: upstream's copy, `~/research-src/HyperQwen/bench/README.md:8`; not in our 95ac251 `bench/README.md`] did not apply on that day ([baseline.md:57](raw/prior-3090/baseline.md)). Clock steps are separate (a step-back of ~2.3 s about every 28 s); that is exactly what perf_counter removes.
- Caveat: the "8.3%" effect is box and day dependent; the harness interleaves arms and records the WSL-vs-host clock drift instead of assuming it. [review (phase 5): `snapshot.sh` does not record drift (harness gap 2, `gpu-agent-prompt.md` harness-gaps list); the queue's `clockdrift()` stand-in in `lib.sh` does, in bench-1 and after bench-4/5/6. Interleaving is the real control.]

If the GPU agent can still reach `G:\dev\hq-bench` and `G:\dev\hq-results\deepinfer-20261020\`, `git -C /mnt/g/dev/HyperQwen log -1 --format=%H -- bench/labd_bench.py` and `grep -c perf_counter bench/labd_bench.py` settles it (ledger row `prior-timing-check`).

## Per-mode table

Columns: TTFT and peak VRAM were never recorded in the prior committed files (see gaps), so they read `not measured`. KV pool is from the engine log, not peak VRAM. All rows: **measured (prior, GPU box)** unless stated.

| Mode | Arm and context | Decode tok/s (tok/step) | Throughput | Acceptance | TTFT | Peak VRAM / KV pool | Source |
|---|---|---|---|---|---|---|---|
| A batch, no spec | 64 conc, 128 in / 512 out, `GPU_UTIL=0.93` r1-2 then 0.91 | 904 (C / median TPOT) | **814.3** e2e (805.6-822.9) | n/a (no spec) | not measured | not measured; KV pool 242,010 tok at 0.93, 227,319 at 0.91; WDDM non-local 370 -> 698 MiB at 0.93 (spill), flat 370-378 MiB at 0.91 | [baseline.md:20,35-36,49,58](raw/prior-3090/baseline.md) |
| A | N=1 / 2 / 4 / 8 / 16 / 32, 4k prompts, 256 out | per stream 44.6 / 40.2 / 30.1 / 20.0 / 12.2 / 6.8 | aggregate (decode window) 44 / 83 / 155 / 274 / 450 / 647 | n/a | not measured | pool 227k; N=32 had 2 preemptions | [evals-10.md:26-32](raw/prior-3090/evals-10.md) |
| B `SPEC=dflash2 PREFIX_CACHE=1` | C1 greedy, run_benchmarks cohort | **114.4** decode (3.36 tok/step); e2e 110.4 (109.2-111.6) | 110.4 e2e | 3.36 tok/step | not measured | not measured; KV pool 68,605 tok; WDDM non-local 92-130 MiB, no growth | [baseline.md:37-38,49,58](raw/prior-3090/baseline.md) |
| B | C1 default sampling | 109.0 e2e | | | | | [baseline.md:39](raw/prior-3090/baseline.md) |
| B | N=1 / 2 / 4 / 8, 4k prompts, 256 out, thinking off | per stream 117.6 / 74.6 / 42.2 / 24.1 | aggregate 118 / 182 / 253 / 236 (N=8: 19 preemptions, pool-limited) | 3.49 / 3.47 / 3.52 tok/step at N=1/2/4 | not measured | | [evals-10.md:26-29](raw/prior-3090/evals-10.md), [evals-4.md:30-32](raw/prior-3090/evals-4.md) |
| B | C1, thinking on (xhigh), 1,024 out | 100.2 (-14.8% vs off; 3.14 tok/step); N=2 83.3, N=4 50.0 | | 0 of 1,024-token streams reached `</think>` | not measured | | [evals-4.md:30-31,37](raw/prior-3090/evals-4.md) |
| B | acceptance by position, T=0.7, thinking off (8 prompts, 1,117 steps) | | | pos 0 / 1 / 2 / 3 / 6 realised 0.687 / 0.511 / 0.404 / 0.325 / 0.190; 3.611 tok/step (1.58-5.55 across prompts); T=0: 3.486 tok/step, 597 steps; thinking on 3.341 | not measured | | [spec-5.md:30-44,51](raw/prior-3090/spec-5.md) |
| B | teacher-forced, labd 6 tasks at 20k, thinking off / on at 8k [review: thinking on is 4 tasks, reasoning phase only, [evals-4.md:41](raw/prior-3090/evals-4.md)] | 91.5 / 140.5 | | 3.02 / 4.42 tok/step; thinking-on accept@pos 0-3 0.82 / 0.65 / 0.52 / 0.44 | not measured | | [evals-4.md:41-43](raw/prior-3090/evals-4.md) |
| B | warm-up: C1 / C4 per stream, runs 1-4 | 114.4 / 109.8 / 105.7 / 103.6 (C1); 61.4 / 60.0 / 58.9 / 58.2 (C4) | | | | thermal settle -9.4% over ~4 min, 62 -> 83 C | [evals-11.md:24,27,31](raw/prior-3090/evals-11.md) |
| B quality | stock, evals-3 | | | | | ppl all 8.401 (en 10.77, da 10.91, code 3.60); GSM8K greedy n=200 95.5% | [evals-3.md:28-29](raw/prior-3090/evals-3.md) |
| C B + `DFLASH_TOKENS=15` | labd 20k `copy` | **357.6** (14.22 tok/step; 353.2 / 361.9) | six-task TOTAL 95.8 [review: a decode TOTAL, not a throughput] | 14.22 tok/step | not measured | KV pool 57,669 tok | [baseline.md:40-41,49](raw/prior-3090/baseline.md) |
| D `SPEC=mtp CTX=long` (resolved `PTH_FA=1`) | C1 greedy, run_benchmarks cohort | **99.8** (2.64 tok/step; 99.3 / 100.2); default sampling 97.0 | | 2.64 tok/step | not measured | KV pool 206,532-207,286 tok | [baseline.md:42-43,49](raw/prior-3090/baseline.md) |
| D | N=1 / 2 / 4 / 8, 4k prompts, 256 out | per stream 105.3 / 75.8 / 43.9 / 22.9 | aggregate 105 / 192 / 308 / 423 | 2.80 / 2.84 / 2.90 tok/step at N=1/2/4 | not measured | no preemptions reported | [evals-10.md:26-29](raw/prior-3090/evals-10.md), [evals-4.md:33-35](raw/prior-3090/evals-4.md) |
| D | C1, thinking on (xhigh) | 95.1 (-9.7%; 2.67 tok/step); N=2 81.7, N=4 57.5 | | | not measured | | [evals-4.md:33-35](raw/prior-3090/evals-4.md) |
| D | teacher-forced, labd 20k thinking off / 8k thinking on | 103.5 / 114.2 | | 3.02 / 3.22 tok/step; accept@pos 0-2 0.86 / 0.73 / 0.62 | not measured | | [evals-4.md:44](raw/prior-3090/evals-4.md) |
| D | warm-up C1 / C4 per stream, runs 1-3 | 103.6 / 101.2 / 98.5 (C1); 77.6 / 74.5 / 73.3 (C4) | | | | | [evals-11.md:25](raw/prior-3090/evals-11.md) |
| D' `SPEC=dflash2 CTX=long` (resolved `PTH_FA=1`) | labd TOTAL, ~60k (`--ctx 53000`) | **55.7** (55.4 / 56.0) | | not measured | not measured | KV pool 136,429 tok | [baseline.md:44,49](raw/prior-3090/baseline.md) |
| D' | labd TOTAL, ~95k (`--ctx 85000`) | **48.8** (48.4 / 49.2) | | | not measured | | [baseline.md:45](raw/prior-3090/baseline.md) |
| E `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` | labd 20k TOTAL | **57.8** (2.71 tok/step; 57.2 / 58.4) | `copy` 158.6 [review: `copy`-task decode, not a throughput] | 2.71 tok/step | not measured | KV pool 268,169 tok | [baseline.md:46-47,49](raw/prior-3090/baseline.md) |
| E | ~170k (169,895 prompt tokens), qa | 11.5 cold (1.86 tok/step), 12.5 profiled (2.01) | | 1.86 / 2.01 tok/step | not measured | WDDM non-local 92 -> 162 MiB | [e-profile.md:8,31-32](raw/prior-3090/e-profile.md) |
| E | per step at ~170k, 151 ms wall | | | | | attention 117.7 ms (78.0%), Marlin GEMMs 17.7 ms (11.7%), **drafter 3.6 ms (2.4%)**, norms 2.2, GDN 1.3, lm_head 0.8, other 1.7 | [e-profile.md:41-48](raw/prior-3090/e-profile.md) [review: the 151 ms is the mean step wall, 150.8 ms, at [e-profile.md:34](raw/prior-3090/e-profile.md)] |
| D `PTH_FA=0` (side finding) | C1 on fp8 FlashInfer, 3 boots | 63-75, GPU 58-64% busy (host-bound on WSL2); same at `GPU_UTIL=0.90` | | | not measured | pool 189,795 tok at 0.90 | [baseline.md:63-65](raw/prior-3090/baseline.md) |

Native-Linux doc figures (not measured here, from the repo docs via the prior write-up): A 942, B 127.1, C copy 381, D 100.3, E 67 TOTAL ([baseline.md:35-46](raw/prior-3090/baseline.md)). The prior author attributes the 6-14% gap to WSL2 (power, clocks, clock drift and spill ruled out, [baseline.md:51-61](raw/prior-3090/baseline.md)).

## Notes that change how to read the table

1. **Mode D already ran with `PTH_FA=1`.** The prior boots resolved `SPEC=mtp CTX=long` to `PTH_FA=1` because the stock image has b4b97ec (#345) ([baseline.md:23](raw/prior-3090/baseline.md)). Our checkout's launchers (`95ac251`) lack that default, so a boot from our checkout with no `PTH_FA` runs fp8 + FlashInfer and would read about 63-75 tok/s C1 ([baseline.md:65](raw/prior-3090/baseline.md)), not ~100. The harness pins `PTH_FA` in every arm and records it.
2. **Upstream's +9% number is not this box's.** Upstream reports +9% decode at ~66k on a native Ubuntu 3090, one boot per arm; the prior WSL2 A/B at C1 found about +40% for D (63-75 -> ~100) ([upstream-diff.md](../01-project/upstream-diff.md) section 6.1; [baseline.md:65](raw/prior-3090/baseline.md)). Those are different contexts and the PTH_FA=0 side was never interleaved against PTH_FA=1; the harness does it (ledger row `pthfa-ab`).
3. **TTFT is absent.** The prior tools print decode rates and tok/step; `conc_ladder` measures TTFT internally but no TTFT value is recorded in any copied file.
4. **Short vs deep context.** D' and E rows are TOTAL decode of mixed labd tasks, not a controlled per-length sweep. Only E at ~170k and D' at ~60k and ~95k give depth points.
5. **Pool limits shape the matrix.** B's 68,605-token pool preempted 19 times at N=8 with 4k prompts; the aggregate 236 tok/s at B N=8 is pool-limited, not a compute ceiling ([evals-10.md:29,38](raw/prior-3090/evals-10.md)).
6. **Thermal drift.** Cold-card short runs read up to ~9% high ([evals-11.md:32](raw/prior-3090/evals-11.md)); the harness carries the warm-up probe verdict with every boot.

## Gaps the harness must fill (metrics never measured)

| # | Gap | Evidence it is missing | Harness piece | Ledger row |
|---|---|---|---|---|
| 1 | TTFT per mode, per concurrency and context | no TTFT value in any copied file; baselines report decode only | `sweep.py` (mean TTFT, cold prefix cache via salt) | `base-*` |
| 2 | Peak VRAM per cell (nvidia-smi `memory.used`) | only KV pool tokens and WDDM non-local MiB recorded; `gpu.csv` stayed on `G:` | `vram.py` inside `sweep.py` and around `conc_ladder` | `base-*` |
| 3 | Decode and throughput at 8k and 32k context for every mode | only D' ~60k/~95k and E ~170k depth points | `sweep.py --set long_context --ctx 8000,32000` | `base-*` |
| 4 | Spec acceptance length and rate with mean and spread over repeats; acceptance for A-E at the same prompts | acceptance by position only for B (spec-5, T=0.7 and T=0); D/E only as tok/step | `sweep.py` from `/metrics` deltas | `base-*` |
| 5 | Greedy T=0 equality, spec vs non-spec, with near-tie classification | never run | `greedy_equal.py` | `greedy-eq` |
| 6 | Draft time vs verify time at normal context | only E at ~170k (drafter 3.6 ms / 151 ms); no metric exists | profile phase (`trace_breakdown.py`); metric patch described in the harness README | `profile-dv` |
| 7 | `PTH_FA=0` vs `1` interleaved for mtp and dflash2 at `CTX=long` | PTH_FA=0 side only as a 3-boot side finding, not paired | pinned arms, interleaved | `pthfa-ab` |
| 8 | A quality guard on every candidate arm | one stock reference (B, evals-3) only | `quality_guard.py` vs `quality_ref_prior.json` | `quality` |
| 9 | Repeats: >= 3 timed rounds per cell with spread | prior used 2 boots / 2 reps | `--reps 3 --warmup 1` | `base-*` |
| 10 | Per-boot env + nvidia-smi snapshot in the results directory | environment recorded once in ENVIRONMENT.md | `snapshot.sh` | `env`, `smoke` |
| 11 | Whether the prior timing code was the perf_counter version | inferred from commit order only | `prior-timing-check` (read-only on the box) | `prior-timing-check` |
| 12 | Raw jsonl/csv behind the prior numbers | not committed, on `G:\dev\hq-results\deepinfer-20261020\` | copy into `raw/prior-3090/` if still present | `prior-raw` |
