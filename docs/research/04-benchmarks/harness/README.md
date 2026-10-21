# Benchmark harness (RTX 3090 box)

Built and self-checked on the research box (no GPU). The GPU agent runs it. Every number the harness prints is **measured on the box that ran it**; the numbers in `selfcheck.out` come from a mock server and prove plumbing only.

It reuses the repo's tools unchanged and adds thin wrappers only where nothing existed:

| Need | Tool | Origin |
|---|---|---|
| warm-up | `bench/warmup.sh`, `vendor/warmup_probe.py` (WARM when two runs agree within 3%) | repo; vendored from `origin/research/deepinfer` 86ce6ed (`vendor/PROVENANCE.md`) |
| TTFT, decode tok/s, throughput at N=1/2/4/8, three prompt sets, acceptance, peak VRAM | `sweep.py` | new |
| N-stream scaling with ms/pass, preemptions, KV% | `bench/conc_ladder.py` | repo, unchanged |
| context 200 / 8k / 32k / long | `sweep.py --set long_context --ctx`, `conc_ladder.py --ctx-tokens` | new / repo |
| spec-decode acceptance length and rate, per position | `sweep.py` (from `/metrics` deltas) | new |
| greedy T=0 equality spec vs non-spec | `greedy_equal.py record` and `compare` | new |
| quality guard | `bench/quality_battery.py` + `quality_guard.py` (pass/fail against `quality_ref_prior.json`) | repo + new |
| draft vs verify time | `run_suite.sh` phase `profile` + `vendor/trace_breakdown.py` | vendored; see "Draft vs verify" |
| env + GPU snapshot | `snapshot.sh` | new |
| orchestration, one boot | `run_suite.sh` | new |
| CPU self-check | `selfcheck.sh`, `mock_server.py` | new |

Fixed prompt sets are in `prompts/` (checked in): `short_chat.jsonl`, `code.jsonl`, `long_context.jsonl` (the four labd tasks copy/quote/summary/qa, the same as `bench/labd_bench.py` `ALL_TASKS`). Long context is built from the labd corpus (`--corpus`, `$HARNESS_CORPUS`, `/cache/bench/labd_corpus_long.txt`); use the same corpus file on every boot of an A/B.

## Arm rules (read before measuring)

1. **Pin `PTH_FA` explicitly (`0` or `1`) in the arm's `.env`.** Upstream b4b97ec (#345, in the stock image `sha-2cf50a0`, not in our `95ac251` launchers) makes an unset `PTH_FA` resolve to 1 for `SPEC=mtp CTX=long` on cc 8.6 (`resolve_config.sh:105-107` upstream), about +9% decode at ~66k (upstream-reported, `../../01-project/upstream-diff.md` section 6.1). An unpinned arm therefore depends on which launcher copy booted. `run_suite.sh` exits 2 if `PTH_FA` is unset in the container (`ALLOW_UNSET_PTH_FA=1` overrides and marks the run). `snapshot.sh` records the value in every run.
   - The default only fires with `CTX=long`, `VISION` not 1, `DRAFT_TOKENS` unset or 3, no `DYNAMIC_K`, no `--speculative-config` in `EXTRA_ARGS`. The box `.env` has `VISION=1`; set `VISION=0` or the arm silently runs fp8 + FlashInfer. Read the `[effective-config]` line in the container log and keep it in the results.
2. **Same `SALT` on every boot of an A/B** (default: the label, so give both arms the same label prefix via `SALT=...`). Acceptance depends on the prompt. A second run on the same boot needs a new salt, or it hits the prefix cache (sweep prints `!`).
3. **Interleave arms** (A B A B), and start each boot with the GPU at 55 C or cooler. The thermal settle is about -9% over ~4 minutes (`../raw/prior-3090/evals-11.md:31`); a short run on a cold card reads high.
4. **Gate on 250 W** (`nvidia-smi -q -d POWER`, recorded in the snapshot). A run at another limit is not comparable with the baseline.
5. **One GPU job at a time**, GPU 0 pinned by UUID (`ENVIRONMENT.md` in `raw/prior-3090/`).
6. **WSL2 clock.** Docker Desktop/WSL2 steps the VM wall clock back about 2.3 s every ~28 s, and it can run fast against the host (8.3% on one 3090, `bench/README.md` [review: upstream's, `~/research-src/HyperQwen/bench/README.md:8`, added with 373a695; our 95ac251 copy lacks it]). Any `time.time()` window can read short. See "Timing code" below.

## Commands on the GPU box

From WSL, in the repo checkout (conventions from `../raw/prior-3090/ENVIRONMENT.md:26-27`). `MODELS_DIR` must be the WSL spelling `/mnt/g/models`.

```bash
# 0. once per session: chain the harness mounts after the box's own override files
export COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml:docs/research/04-benchmarks/harness/docker-compose.harness.yml
mkdir -p docs/research/04-benchmarks/results

# 1. arm .env (append or replace; keep the box's GPU-pin lines). Example: mode B
cat >> .env <<'EOF'
MODELS_DIR=/mnt/g/models
VLLM_WSL2_ENABLE_PIN_MEMORY=1
SPEC=dflash2
PREFIX_CACHE=1
PTH_FA=0
VISION=0
EOF
# PTH_FA is pinned on purpose (Arm rules 1); dflash2 CTX=long: pick 0 or 1, run both. VLLM_WSL2_ENABLE_PIN_MEMORY is required on WSL2 for SPEC=dflash2.
# Keep .env values bare: no inline comments after a value.

# 2. boot (fresh container per arm), wait for health
docker compose --profile single up -d --force-recreate
until curl -sf http://127.0.0.1:18020/health >/dev/null; do sleep 5; done

# 3. the suite, inside the container (harness is mounted read-only at /harness, results at /results)
docker compose exec -T single bash /harness/run_suite.sh B-pth0
#    results: docs/research/04-benchmarks/results/B-pth0-<timestamp>/

# 4. host-side snapshot (docker image digest, compose config, KV pool log lines)
bash docs/research/04-benchmarks/harness/snapshot.sh docs/research/04-benchmarks/results/B-pth0-<timestamp> host
docker compose down
```

If the compose chaining is awkward on the box: `docker cp docs/research/04-benchmarks/harness <container>:/harness` and `docker cp <container>:/results ./results` after the run (the suite writes under `$OUT`, default `/results/<label>-<ts>`; set `OUT=/tmp/res` if `/results` does not exist).

Per-boot sequence in `run_suite.sh` (default `PHASES="snap warm probe sweep ladder greedy"`): pre-snapshot; `bench/warmup.sh`; `warmup_probe` (WARM or NOT-WARM verdict, not-warm results are suspect); `sweep.py` on short_chat and code at N=1,2,4,8 and long_context at 8k (N=1,2,4) and 32k (N=1), 1 warm-up round then 3 timed rounds per cell, mean/stdev/min/max/CV; `conc_ladder.py` at 200 and 8k tokens with ms/pass, preemptions, KV%; peak VRAM per cell (`vram.py`, nvidia-smi, secondary) and the WDDM spill reading (`gpumem_summary.py`, primary, if `GPUMEM_CSV` is set); greedy ids recorded; post-snapshot. A failed phase is logged to `status.txt` and the rest still run. Exit code is the number of failed phases > 0.

Sizing. Defaults fit mode B's 68,605-token KV pool (N=8 at 4k preempted 19 times in the prior run, `../raw/prior-3090/evals-10.md:29`). For `CTX=long`/`huge` arms raise them, for example `LC_CELLS="8000:1,2,4,8 32000:1,2,4 64000:1" LADDER_CELLS="200:1,2,4,8 8000:1,2,4,8 32000:1,2"`. `LC_CELLS=""` (set but empty) skips the long-context cells; `SMOKE=1` lets `REPS` go below 3 (smoke runs only, no spread). Other knobs: `REPS WARMUP MAXTOK CONC SALT LC_MAXTOK LADDER_OUT GREEDY_TOKENS PROBE_TOKENS PHASES`.

## Spec vs non-spec greedy equality

Speculative decoding is exact at T=0, so token ids must match a non-speculative boot. Record on both boots with the same prompts, then compare off-box or in either container:

```bash
# boot 1: SPEC=off (single-user/start_qwen.sh:193; unset means mtp), keep PTH_FA pinned (the suite requires it), PHASES="snap greedy"
docker compose exec -T single env PHASES="snap greedy" bash /harness/run_suite.sh NS
# boot 2: SPEC=dflash2 (or mtp) with the same GPU/ctx settings (PTH_FA only where the arm allows it: it is refused on SPEC=off, so pin 0 on boot 1 and the arm's value on boot 2)
docker compose exec -T single env PHASES="snap greedy" bash /harness/run_suite.sh SPEC
python3 docs/research/04-benchmarks/harness/greedy_equal.py compare results/NS-*/greedy_NS.json results/SPEC-*/greedy_SPEC.json
```

Exit 0 all identical; 1 divergence, with the verdict line saying `near-tie` (both top-1/top-2 margins under `--near-tie`, default 0.1 nats: different kernel shapes round differently, not a bug) or `real` (large margin: a defect, verdict INVALID); 2 invalid (different prompt sets or `GREEDY_TOKENS`, failed request, no token ids). Record runs at concurrency 1 so batch shape does not add noise. Both arms must use the same `GREEDY_TOKENS`. [review: and the same KV dtype and attention backend: `SPEC=off` at `CTX=fast` (bf16 KV) against `SPEC=mtp CTX=long PTH_FA=1` (int8 KV) diverges for KV reasons, not speculation. Compare spec vs non-spec at the same `CTX` with `PTH_FA=0` on both. A `PTH_FA=1` arm has no non-spec twin (`single-user/start_qwen.sh:299-304` refuses `PTH_FA=1` off mtp/dflash2 `CTX=long`), so read its divergences with the KV-dtype confound in mind.]

## Quality guard

`PHASES="snap quality" bash /harness/run_suite.sh LABEL` runs `bench/quality_battery.py` (perplexity on wikitext-2 en, fineweb-2 da, vLLM source code; GSM8K n=200 greedy) then `quality_guard.py`: perplexity may not rise more than 1% (all and per corpus), GSM8K may not fall more than 3 points (a garbage detector only; one question is 0.5 pt). Reference `quality_ref_prior.json` is the prior stock mode B run: ppl all 8.401 (en 10.77, da 10.91, code 3.60), GSM8K 0.955 (`../raw/prior-3090/evals-3.md:28-29`, measured prior). Needs `QUALITY_DATA=/app/models/quality-data` (default in the suite) and `prompt_logprobs` headroom: boot with `GPU_UTIL` near 0.93 or the logprob pass can OOM. For batch mode keep `GPU_UTIL=0.91` for the throughput arms (A spills at 0.93 on this box) and use a separate boot for quality.

## Draft vs verify time

**No draft-time or verify-time metric exists** in the stack: `/metrics` has only the spec counters (`vllm:spec_decode_num_drafts_total`, `num_draft_tokens_total`, `num_accepted_tokens_total`, `num_accepted_tokens_per_pos_total{position}`), which `sweep.py` turns into tokens per step (1 + accepted/drafts) and acceptance rate (accepted/draft tokens). `conc_ladder.py` reports ms per forward pass but not its split.

What does exist is the profile route, used for `e-profile` (`../raw/prior-3090/e-profile.md:9,17-23`: drafter 3.6 ms of a 151 ms step at E, 170k). Boot with the profiler enabled, then run only the profile phase:

```bash
# .env addition for the profile boot only (profiling slows the server; never mix with timing arms)
# [review: fixed; the line lacked the flag itself and vLLM would get a bare JSON argument. Form as e-profile.md:9 and run_suite.sh:71; keep the comment off the value line]
EXTRA_ARGS=--profiler-config {"profiler":"torch","torch_profiler_dir":"/results/prof"}
docker compose exec -T single env PHASES="snap profile" PROF_DIR=/results/prof bash /harness/run_suite.sh LABEL-prof
```

`profile` posts `/start_profile`, runs a short sweep, posts `/stop_profile`, then `vendor/trace_breakdown.py` writes `breakdown.json` (ms per step by category, with a Drafter row).

A permanent timing metric would need a patch: wrap the drafter's `propose()` in the spec-decode proposer path with a `vllm:spec_decode_draft_time_seconds` histogram (and the rejection-sampler/verify forward with a matching one), using CUDA events or a sync because launches are asynchronous, so the host timer alone would read the launch cost only. The prior `spec5-accept-telemetry.patch` (`origin/research/deepinfer`, `experiments/patches/`) is the template for an inert-unless-enabled hook but records acceptance probabilities, not time. Until such a patch exists, draft time comes from the profile phase only.

## Timing code (WSL2 clock)

Upstream 373a695 (#351) moved every `bench/` duration from `time.time()` to `time.perf_counter()` because the WSL2 VM wall clock steps backwards.

| Code | Clock | Evidence |
|---|---|---|
| `sweep.py`, `vendor/warmup_probe.py` | `time.perf_counter()` | `sweep.py:81,91,94,110,116` [review: renumbered after the finish-chunk fix]; `vendor/warmup_probe.py:43,49,65,70` |
| `greedy_equal.py`, `vram.py` | no latency timing (token ids and memory samples only) | |
| `bench/conc_ladder.py` | `time.perf_counter()` for all windows (`time.time()` only in a salt stamp) | `conc_ladder.py:124,131,161,174-183,227` |
| `bench/labd_bench.py` at our `95ac251` | **`time.time()`** | `labd_bench.py:85,95,97`. Fixed upstream in 373a695 and on the prior branch's `bench/` [review: and in the stock image: `git merge-base --is-ancestor 373a695 2cf50a0` is true, and the baseline commit `e6a3ca0` has `perf_counter` at `bench/labd_bench.py:85,95,97`]. |
| `vllm bench serve` | client durations use `perf_counter` | upstream vLLM |

So: do not bind-mount our `bench/` over `/app/bench` for `labd_bench.py` replays until #351 is adopted. Use the image's copy, or the prior branch's `bench/` (the prior box bind-mounted `research/deepinfer`'s `bench/`, which contains 373a695; `../raw/prior-3090/ENVIRONMENT.md:26`). The harness itself imports `bench/harness.py` and `conc_ladder.py` from `/app/bench`; both are safe. The 8.3% fast-clock caveat is not removed by perf_counter: interleave arms.

## Replaying the prior baseline protocol

The harness measures the same modes with a fixed prompt set; it does not replace the prior protocol numbers. For apples-to-apples with `baseline.md`, run the prior commands per mode (`../raw/prior-3090/baseline.md:18-25`): `bash bench/run_benchmarks.sh single` (modes B, D, twice, keep the second), `bench/run_benchmarks.sh batch` (A, `GPU_UTIL=0.91`), `labd_bench.py C|E --ctx 20000`, `labd_bench.py --ctx 53000|85000 --tasks qa,summary --corpus labd_corpus_long.txt` for D'.

## CPU self-check

`bash selfcheck.sh` (no GPU, stdlib python3 only) starts `mock_server.py` (an OpenAI-shaped server with `/metrics` spec counters) and runs: the whole suite (spec on, acceptance counters move); the PTH_FA refusal (exit 2); [review: the decode-rate window with an EOS-only finish chunk (1c);] greedy compare on identical, real-divergence and near-tie inputs; `quality_guard.py` on synthetic results; `vram.py` without nvidia-smi. It ends with `SELFCHECK PASS`. Saved output: `selfcheck.out` (mock numbers, plumbing only). It cannot check anything GPU-specific: real `/metrics` names on the image, the profiler route and nvidia-smi parsing are first exercised by the smoke run (ledger row `smoke`).

## Output files per run

`snapshot_{pre,post}_<host>.txt`, `warmup_probe.json`, `sweep_<label>_<set>.json` (+ markdown tables in the phase logs), `ladder_<ctx>.json` and `ladder_<ctx>_vram.json`, `greedy_<label>.json`, `suite.log`, `status.txt`, per-phase `<phase>.log`; with `quality`: `result_<label>.json`; with `profile`: `breakdown.json`.

## VRAM and spill (WDDM sampler is the primary source)

nvidia-smi `memory.used` is near-flat per cell on this stack (vLLM preallocates the KV pool) and cannot see memory the WSL VM demoted to system RAM, which is what happened to mode A at `GPU_UTIL=0.93` (370 -> 698 MiB, `../raw/prior-3090/baseline.md:58`). So the box's own spill sampler is the VRAM/spill source; `vram.py` stays as a secondary reading (`vram` in each sweep cell, `ladder_*_vram.json`).

Per boot, from WSL (sampler path, `-Match 1144f` and the "Non Local Usage" counter from `../raw/prior-3090/ENVIRONMENT.md:28`; prior cadence 2 s, `gpumem.csv`):

```bash
LABEL=B-pth0; R=docs/research/04-benchmarks/results
powershell.exe -NoProfile -File 'G:\dev\hq-bench\gpumem_sample.ps1' -Match 1144f > $R/gpumem_$LABEL.csv &  SAMPLER=$!   # start BEFORE the suite
docker compose exec -T -e GPUMEM_CSV=/results/gpumem_$LABEL.csv single bash /harness/run_suite.sh $LABEL
kill $SAMPLER                                   # the suite has already summarised what was written so far
```

The `spill` phase copies the csv to `$OUT/gpumem.csv` and writes `$OUT/spill.json` (first / max / last / growth, verdict `flat` or `SPILL`; growth over 64 MiB is SPILL, the prior bound). Decision D15 (GPU runs): record the growth for every run, flag growth over 64 MiB as an anomaly (the harness verdict `SPILL`, and `spill rc=1` in `status.txt`, is that flag), abort only on OOM, growth over 256 MiB (`gpumem_summary.py ... --tol 256`), or C1 decode more than 10% below the prior measured value for the mode. A flagged boot is still reported with its growth. Without `GPUMEM_CSV` the suite says so in `suite.log` and records no spill reading. The summary assumes the last numeric field of each line is MiB: the sampler's output format is not in the repo, so the first real run must check one `gpumem.csv` against it (smoke row) and adjust `gpumem_summary.py` if the unit differs. The sampler is host-side and cheap; nvidia-smi polling every 0.5 s is the only extra GPU query the harness makes.

## Post-review changes

After the phase-2 review (`../../00-meta/reviews/phase2-harness-baseline.md`), on top of the reviewer's own edits:

1. F3: WDDM "Non Local Usage" sampler documented above and wired in (`gpumem_summary.py`, `spill` phase, `spill.json`); nvidia-smi peak is marked secondary in `vram.py` output and sweep JSON.
2. F5: the smoke run includes the greedy check. Smoke command: `SMOKE=1 PHASES="snap probe sweep greedy spill" REPS=1 LC_CELLS="" GREEDY_TOKENS=64 bash /harness/run_suite.sh smoke`, then in the same boot `python3 /harness/greedy_equal.py record smoke2 --max-tokens 64 --out /results` and `greedy_equal.py compare /results/smoke-*/greedy_smoke.json /results/greedy_smoke2.json` (expect identical or near-tie only; this exercises logprobs and `return_tokens_as_token_ids` under the V2 runner before bench-7).
3. F4: `vram.py` takes the GPU from `HARNESS_GPU`, else the first `CUDA_VISIBLE_DEVICES` entry (the pinned UUID), else 0.
4. `selfcheck.sh` gained 4b (spill flat vs SPILL) and 4c (GPU from env); `selfcheck.out` re-run, SELFCHECK PASS.
