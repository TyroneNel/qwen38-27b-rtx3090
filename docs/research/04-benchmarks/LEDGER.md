# GPU benchmark ledger

One row per GPU job for the benchmark harness ([harness/README.md](harness/README.md)). The GPU agent keeps this file current and commits it with every result. The queue entries are in [../00-meta/gpu-queue.md](../00-meta/gpu-queue.md). Prior numbers are in [baseline.md](baseline.md). Same vocabulary as the prior ledger ([raw/prior-3090/LEDGER.md:5](raw/prior-3090/LEDGER.md)).

Status: `todo` · `running` · `done` · `blocked` · `skipped`. Verdict: `WIN` (clears the item's win criterion and the baseline spread) · `LOSS` · `NEUTRAL` (inside noise) · `INVALID` (correctness or lossless check failed) · `INFO` (measurement only, no win criterion).

Rules for every row: pin `PTH_FA` (harness README, Arm rules); fresh container per boot; 250 W and 55 C or cooler at boot; keep `docs/research/04-benchmarks/results/<label>-<ts>/` as the evidence; every number written here is "measured (this box, date)" and carries its spread. Result directories are not committed unless small; commit the summary row and the sweep/greedy JSON.

## Phase 0: environment and plumbing

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| env | Env snapshot: `snapshot.sh` in container and on WSL host; image digest, driver, power limit, `.env` incl. pinned `PTH_FA`, `[effective-config]` line | – | todo | | | |
| smoke | Harness smoke run: `run_suite.sh smoke` on mode B with `REPS=1 LC_CELLS="" GREEDY_TOKENS=64 PHASES="snap probe sweep greedy spill"` with `GPUMEM_CSV` set (README "VRAM and spill"), then a second greedy record on the same boot and `greedy_equal.py compare`; confirms real `/metrics` names, nvidia-smi parsing, the `gpumem_sample.ps1` csv unit, salt, `warmup_probe`, logprobs under the V2 runner | env | todo | | | |
| prior-timing-check | Read-only: does the box's bind-mounted `bench/labd_bench.py` use `perf_counter` (`grep -c perf_counter`, `git log -1 -- bench/labd_bench.py`)? Settles baseline.md "Timing validity" | – | todo | | | |
| prior-raw | Copy `G:\dev\hq-results\deepinfer-20261020\{baseline,evals-4,evals-10,e-profile}\` jsonl/csv into `raw/prior-3090/data/` if still present | – | todo | | | |

## Phase 1: baseline re-run per mode (the harness suite, 3 timed rounds per cell)

Priority order B, D, E (the modes EAGLE-4 compares against); then D', C, A. Compare every row with [baseline.md](baseline.md); a difference beyond the prior 1-2% spread plus this run's CV needs an explanation (thermal, pinned `PTH_FA`, image).

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| base-B | `SPEC=dflash2 PREFIX_CACHE=1 VISION=0 PTH_FA=0`, full suite (sweep N=1,2,4,8 on short_chat and code, 8k and 32k, ladder, greedy record, probe) | smoke | todo | | | |
| base-D | `SPEC=mtp CTX=long VISION=0`, `PTH_FA=1` (the prior default arm). Raise `LC_CELLS` to include 64k | smoke | todo | | | |
| base-E | `SPEC=dflash2 CTX=huge PREFIX_CACHE=1 VISION=0`, `PTH_FA` pinned (record which), `LC_CELLS="8000:1,2 32000:1 64000:1 120000:1"` | smoke | todo | | | |
| base-Dp | `SPEC=dflash2 CTX=long PTH_FA=1` (D'), `LC_CELLS` to 64k and 90k | smoke | todo | | | |
| base-C | B + `DFLASH_TOKENS=15`, `PTH_FA=0`; plus the repo's `labd_bench.py C --ctx 20000` for the `copy` row | smoke | todo | | | |
| base-A | `batch`, `GPU_UTIL=0.91`, `PTH_FA=0`; ladder to N=64; throughput row | smoke | todo | | | |

## Phase 2: correctness and quality

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| greedy-eq | T=0 token-id equality, `SPEC=off` vs `SPEC=dflash2` (and vs `SPEC=mtp`) on `short_chat` and `code`, 256 tokens; `greedy_equal.py compare` | smoke | todo | | | |
| quality | `quality_battery.py` + `quality_guard.py` on stock B at `GPU_UTIL` near 0.93: ppl within 1% of 8.401, GSM8K within 3 pts of 95.5% (reference [evals-3.md:28-29](raw/prior-3090/evals-3.md)). Re-run for every candidate arm later | smoke | todo | | | |

## Phase 3: attribution probes

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| pthfa-ab | `PTH_FA=0` vs `1` interleaved (0 1 0 1) for `SPEC=mtp CTX=long` and `SPEC=dflash2 CTX=long`; same `SALT`, same corpus; reports decode and TTFT at ~8k and ~66k | base-D, base-Dp | todo | | | |
| profile-dv | Torch profile on a profiler boot of mode B at C1 short context: drafter ms vs target ms per step (`trace_breakdown.py`); ceiling-check against E at 170k (drafter 3.6 of 151 ms) | smoke | todo | | | |
