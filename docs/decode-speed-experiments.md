# Experiments to close the decode-speed gaps

The run list for [decode-speed-landscape.md](decode-speed-landscape.md): seven arms, each an A/B on one RTX
3090, each with the metric and the pass/fail rule decided **before** the run. Written to be executed by an
agent on the GPU box with no judgement calls; see [gpu-agent-brief.md](gpu-agent-brief.md) for that agent's
operating rules.

**Ground rules, because a 3090 at 250 W has 3-8% boot-to-boot spread and every arm here is smaller than that
in places.**

1. **One image, many arms.** Nothing in E1-E5 changes engine code — they are env knobs in `.env` — so one
   image built once from the branch serves every arm. Rebuild only for E7.
2. **Discard the first measurement after every boot.** The first bench run after a start reads 30-50% low
   (JIT warmup). Boot, run the arm once, throw it away, run it again, keep that.
3. **Pin the variables you are not testing.** Any arm that changes KV dtype or attention backend pins
   `MAX_LEN` identically on both sides (the repo's own warning: the pool size moves with the dtype, and a
   context difference masquerades as a kernel difference).
4. **Noise discipline.** A delta under 3% is noise: record it as "no difference". A delta between 3% and 10%
   needs A-B-A (three boots) before it is a result. Above 10% one boot per arm is enough.
5. **No soak, no verify-full, no acceptance battery on every arm.** Speed first. Quality is a one-minute smoke
   (`quality_battery.py --gsm-only --gsm-n 50`) on any arm that changes a kernel, dtype or backend — never a
   full run.
6. **Any crash is a result.** Record the traceback and the exact command, move on. Do not fix the engine to
   make an arm pass.

## Common setup for every arm

```bash
# once
git clone <fork> HyperQwen && cd HyperQwen
git checkout cline/gf59cdmh                      # the branch results are committed to
printf 'MODELS_DIR=G:/models\nVLLM_WSL2_ENABLE_PIN_MEMORY=1\n' > .env
docker compose build single                      # one build, reuse for every arm
docker image inspect <tag> --format '{{.Id}}'    # record the digest in every report

# per arm: set the knobs in .env, then
docker compose --profile single up -d
docker compose exec single bash bench/run_benchmarks.sh single   # discard
docker compose exec single bash bench/run_benchmarks.sh single   # keep these ROW lines
```

The three measurement tools, all inside the image:

| tool | what it gives | used by |
|---|---|---|
| `bench/run_benchmarks.sh single` | the headline ROW lines: e2e, decode (C/medTPOT), medTPOT, TTFT | every arm, C1 |
| `bench/conc_ladder.py --n 1,2,4,8,16,32 --json <f>` | per-stream and aggregate decode per N, **plus** tok/pass, ms/pass and preemptions — which separate "takes turns" from "batching" from "pool too small" | E1, E2, E5 |
| `bench/labd_accept.py` | teacher-forced tokens per step on a frozen sequence: the same continuation replayed on every config, so acceptance is not polluted by trajectory divergence | E3, E4 |

Results go to `bench/results/<YYYY-MM-DD>-<arm>/`: raw logs verbatim, the exact `.env` used, the image digest,
the commit SHA, and a `report.md` with the table. Commit and push them; do not open a PR.

## Memory headroom: the rule that makes every number valid

Under WSL2 and Docker Desktop the card's usable dedicated memory is about half a gigabyte smaller than the
same card on bare metal, and the failure mode when a working set crosses that line is **not** an error: the
remainder lands in host shared memory and the step runs **two to six times slower**, with nothing in vLLM's
log. This is gotcha 53 (`gotchas.md`) and it is the easiest way to produce a wrong answer on this machine,
because `nvidia-smi` shows the card full either way.

Measured, same card, over the line and with room (gotcha 53):

| working set | over the line | with room |
|---|---|---|
| `DFLASH_TOKENS=15` (about 420 MiB over) | 56.5 ms/step | 23.4 |
| verify-kernel query maximum 16 at width 7 | 63.5 | 23.2 |
| a boot that recompiles the drafter's graph while loading the backbone's | 45.0 | 23.2 |
| a 4 GiB bf16 drafter at the shipped pin | 150-264 | 39 |

**The rules, and they are not optional:**

1. **Leave at least 1.3 GB of free VRAM after boot.** Free VRAM after boot is the engine's own figure minus the
   pool; `spec-decode-scratch-token-units.md` records 319 MiB as the "burst zone" and 1,323 MiB as out of
   it, for about 22k of 230k KV tokens. Read it from the boot log and record it in every arm's report.
2. **Read the counters, not the log.** Sample the Windows counters in a second PowerShell window for the whole
   run: `Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage'` every 10 s.
   Shared Usage at its idle baseline (~86 MiB on the reference box) through a run means the working set is on
   the card. Anything above baseline means part of it is not.
3. **Acceptance survives a spill; tok/s does not.** A spilled run's tokens-per-step column stands and its tok/s
   is void. If a spill is detected, discard the timing and re-run with `KV_MEM` and/or `MAX_LEN` reduced until
   Shared Usage stays at baseline — do not report the slow number, and never compare a spilled arm against a
   clean one.
4. **Pin the pool by bytes, not by utilization, in every arm.** `GPU_UTIL` moves the pool with compile-cache
   state and with the KV dtype — precisely what E2 varies. `KV_MEM` is the pinned door (gotcha 31: it assumes a
   headless card, so drop it by whatever the desktop is holding). `KV_MEM` and `MAX_LEN` are two doors into the
   same shortfall: set both explicitly, identically across the arms being compared, and change them only to
   clear a spill.
5. **Match the boot class before comparing to a published row.** A container boot and a native venv boot of the
   same config differ by ~500 MiB of residency on this repo's own measurements, and a WSL2 row and a bare-metal
   row of the same card differ by ~0.5 GB of usable memory. Every comparison in the report says which class
   each side is.
6. **KVarN spills at boot**, when its KV cache is allocated, before any request — so on `CTX=huge` the check
   happens at startup, not during the run, and a clean-looking run can have been slow from the first token.

This is why E2 pins `MAX_LEN` on all four arms, and why every arm's report carries free VRAM after boot and the
Shared Usage trace: without them a 2-6× spill looks like a 2-6× kernel result.


## E1 — Load-aware K instead of the single/batch switch (G1)

**Why:** our two modes are a hard switch. `single` speculates at a fixed K, `batch` turns speculation off
entirely, and the DFlash2 residency ceiling (C8 ≈ 324 tok/s against MTP's 407) has nowhere to go. The pin
supports a K schedule per batch size, so the question is whether it recovers the C8+ rows without costing C1.

**Arms** (both `SPEC=dflash2 PREFIX_CACHE=1 CTX=fast`, identical otherwise):

- **A** — shipped: `DFLASH_TOKENS=7`, no schedule.
- **B** — A plus a schedule in the spec config: `[[1,8,7],[9,32,3],[33,512,0]]`.

**How to apply B without editing the repo.** The launcher builds its own `--speculative-config` from
`SPEC_CFG` and passes it in `SPEC_ARGS`; `EXTRA_ARGS` is appended to the same command line. Whether a second
`--speculative-config` in `EXTRA_ARGS` wins is **not assumed** — the first thing to do is confirm it:

```bash
# which spec config did the engine actually get?
docker compose exec single bash -lc 'tr "\0" " " < /proc/$(pgrep -f "vllm serve" | head -1)/cmdline'
```

If the launcher's config wins (the schedule is absent from that line), do **not** hack the engine: add the
schedule to `SPEC_CFG` behind a new `DYNAMIC_K` env knob in `single-user/start_qwen.sh`, keep it off by
default, and say in the report that arm B required a launcher change (with the diff).

**Metric:** `conc_ladder.py --n 1,2,4,8,16,32 --json` → `decode agg` per N, `preempt`, `ms/pass`; plus
`run_benchmarks.sh single` for the C1 ROW line.

**Pass:** decode agg at N=16 and N=32 improves by more than 10% over A, and C1 decode stays within 3%. **Fail
but interesting:** C1 holds and N=8 improves 3-10% — call it a win only after A-B-A.

**Budget:** ~10 min per boot per ladder, 4 boots.

## E2 — Attention backend × KV dtype at equal context (G3)

**Why:** our own 2×3090 reproduction reads 90.9 tok/s on the shipped fp8/FlashInfer path and 156.0 on
`TRITON_ATTN` + `int8_per_token_head` at the same tokens per step — a 1.7× gap between two configurations we
already ship, in different modes. Find out which one is fastest on *this* card at 250 W.

**Arms** — all `SPEC=dflash2 PREFIX_CACHE=1`, all with `MAX_LEN=65536` pinned so the dtype is the only
variable:

| arm | `EXTRA_ARGS` |
|---|---|
| A | *(shipped `CTX=fast`: `--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16`)* |
| B | `--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head` |
| C | `--attention-backend FLASHINFER --kv-cache-dtype fp8` |
| D | `--attention-backend FLASHINFER --kv-cache-dtype bfloat16` |

**Pin the pool in bytes, and shrink it until the margin holds.** Set `KV_MEM` explicitly on all four arms
(identical value, so the dtype is the only variable) rather than leaving the pool to `GPU_UTIL`, and after
each boot check free VRAM after boot against the 1.3 GB rule above. The three dtypes do not need the same
pool, so if one arm cannot hold the pinned `KV_MEM` without spilling, lower it for **all** arms and say so —
a dtype that only looks fast because its pool was smaller is not a result.

**Verify the backend actually engaged** before trusting any arm — read the engine's args line and the boot log
for the selected backend; on sm86 an fp8 KV request can silently route off `FLASH_ATTN`. An arm whose backend
did not engage is reported as "did not engage", not as a slow arm.

**Metric:** `conc_ladder.py` `ms/pass` at N=1 and N=8 (the step time is the thing being tested), C1 decode
from `run_benchmarks.sh single`, tokens/step from `bench/harness.py spec`, and `quality_battery.py --gsm-only
--gsm-n 50` on any arm that wins.

**Pass:** a winning arm beats A by more than 5% on C1 decode at the same `MAX_LEN`, with tokens/step within 2%
and the GSM8K smoke within 1 point. **Budget:** ~10 min per arm; 4 arms, 1 boot each, A-B-A on the winner.

## E3 — Acceptance by workload (G2)

**Why:** published DFlash2 acceptance for this model is 5.46 GSM8K / 4.39 HumanEval; our published row is ~3.3
on an 8-prompt mixed chat protocol. We cannot tell "our drafter is worse" from "our prompts are harder", and
every tuning decision inherits that. This arm answers it with a number per workload class.

**Arms:** one server (`SPEC=dflash2 CTX=fast`, shipped config), three frozen continuations:

1. a GSM8K-style math answer (from `bench/quality-data/gsm8k` if present),
2. a code answer (a long function with comments),
3. the chat prompt from `bench/prompts_real.jsonl`.

Capture each continuation once with `labd_accept.py`, freeze it to disk, then replay it — that is what makes
the three numbers comparable.

**Metric:** tokens per step and accept rate per workload, plus C1 decode per workload.

**Pass:** the table exists. If any workload's tokens/step is ≥1.3× the chat row, name it in the report as the
workload this stack should be tuned and quoted for. **Budget:** ~15 min total.

## E4 — Reasoning effort, if the served template accepts it (G2)

**Why:** club-3090 measured acceptance 5.362 at `reasoning_effort=low` against 3.872 at `xhigh` (−27.8%,
against ~2.8% boot noise). If our served template exposes the knob, this is the cheapest acceptance in the
repo.

**Arms:** A = default; B = `reasoning_effort=low` (or whatever the served template names it) on the same
teacher-forced sequences as E3.

**First check, and do not assume:** whether the template accepts the value at all. Our stack does not ship
club-3090's `qwen38-reasoning-effort-template` patch, so a `high`-style rung may return 400/500. Send one
request, read the status. If it is rejected, record the error verbatim, mark E4 "not available on this
template", and move on — **do not** add a chat-template patch in this run.

**Pass:** tokens/step at `low` is more than 10% above default with unchanged output quality on the same
sequence. **Budget:** ~10 min, or 2 min to establish it is unavailable.


## E5 — CUDA-graph capture size and mode (G9)

**Why:** the community report is 0.3 GiB against 3.7 GiB of graph memory and +5-19% decode from a smaller
capture set — on a 24 GB card that is pool *and* speed. Our launcher already exposes both knobs
(`CG` → `max_cudagraph_capture_size`, `CUDAGRAPH_MODE` → `cudagraph_mode`), so this costs one `.env` line.

**Arms:** A = shipped (`CG` default, `CUDAGRAPH_MODE` unset); B = `CG=40`; C = `CG=40 CUDAGRAPH_MODE=FULL`.
Read the default from the boot log first — it is computed as `MAX_SEQS*(DRAFT_TOKENS+1)`, capped at 64, so
"the default" is 64 on the shipped profile.

**Metric:** graph MiB and boot resident MiB (from the engine's own memory line), the reported KV pool size,
decode at C1 and C8, and the cudagraph-mode line in the log.

**Pass:** the winning arm gains more than 5% decode at C1 or C8 with a larger pool and no mode downgrade in
the log. **Budget:** ~10 min per arm, 3 boots.

## E6 — Mamba block size and checkpoint pages (G10)

**Why:** checkpoint pages come out of the KV pool and are not counted in the reported pool size; raising the
mamba block size is reported to cut checkpoints ~5×, which bought two concurrent 242K requests with zero
preemptions elsewhere.

**Do not assume the flag name.** First, inside the container:

```bash
docker compose exec single venv/bin/vllm serve --help | grep -i mamba
```

Use whatever block-size flag exists on the pin; if there is none, mark E6 "no flag on this pin" and stop —
that is a valid, useful result.

**Arms:** A = shipped; B = A plus the block-size flag at 8192, `MAX_LEN` pinned equal on both sides.

**Metric:** reported KV tokens, `preempt` from `conc_ladder.py --n 16` on two long concurrent requests, and
follow-up TTFT on a repeated long prompt (the prefix-cache benefit is TTFT, not decode — do not report it as a
decode win).

**Pass:** more usable pool tokens or fewer preemptions at equal or better decode, with no acceptance change.
**Budget:** ~15 min, 2 boots.

## E7 — Draft-head quantisation (G12) — needs a new image, last

**Why:** the draft LM head is read at full width every verify step; quantising only the *draft* head is
lossless by construction (the target verifies on the full vocabulary) and is reported at 4.3 → 2.2 ms per
verify step.

**This is the one arm that changes files under `prepare/`, so it needs its own image** (tag it with the arm
name so it is never confused with the shared one). Skip it unless E1-E5 are done.

**Arms:** A = shipped head; B = draft head quantised, target head unchanged.

**Metric:** ms per verify step (from the `ms/pass` column at fixed K, not from wall clock), C1 decode, and a
GSM8K smoke.

**Pass:** more than 3% decode gain with the smoke unchanged. **Budget:** one build plus ~15 min.

## What not to do

- No `verify.sh` full runs, no acceptance batteries, no soak loops. A smoke per kernel-touching arm only.
- No engine patches to make an arm pass. The one permitted code change is the launcher knob in E1, and only
  if the report includes the diff and it is off by default.
- No model re-preparation unless the model directories are genuinely absent — record what was found in
  `G:/models` first, and if a needed artifact is missing, say so instead of downloading 20 GB.
- No results without raw logs. Every number in a report must be traceable to a log file committed beside it.

