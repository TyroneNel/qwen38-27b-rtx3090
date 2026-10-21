# X-17: draft vs verify vs accept GPU timing (decision D12)

Variant-image patch that logs, per engine step, GPU time spent in the target verify forward, the
rejection sampler (accept) and the drafter (DFlash2 and MTP). It is a measurement tool for a
separately tagged image. It is not in `patches/series`, the project `Dockerfile` or the stock image.

Status labels used below: **measured** = observed on a run; **estimated** = reasoning, no GPU run.
Nothing in this folder is measured on a GPU. Every CPU check below is a measured CPU result. Every
overhead number is estimated. The parser demo in `checks.out` uses synthetic (fake) event times.

## Files

| file | what |
|---|---|
| `spec-timing.patch` | the patch, `-p1` relative to the vllm package dir (adds `v1/worker/gpu/spec_timing.py`, edits `v1/worker/gpu/model_runner.py`, 24 added lines) |
| `Dockerfile.variant` | `FROM ghcr.io/syv-ai/hyperqwen:sha-2cf50a0`, applies the patch to the installed vllm |
| `parse_timing.py` | log lines to a draft/verify/accept summary per run (stdlib only) |
| `test_timing.py` | CPU unit test of the aggregator and parser with fake event times |
| `checks.sh` | the CPU checks; `bash checks.sh > checks.out` (added by the review) |
| `checks.out` | output of every CPU check (re-run on the final files) |
| `series-apply.out` | which of `patches/series` applied to v0.31.0 |

## Reuse decision: upstream StepTimingCollector

Upstream already has a `StepTimingCollector`. Checked, not reused as-is.

- v0.31.0 with our patches applied: `vllm/v1/worker/gpu/async_utils.py` has `StepTimingSample` (line 28),
  `StepTimingEvents` (41), `StepTimingCollector` (48), `collect()` (62; one sync at block exit,
  `timed[-1][0].drafter_end.synchronize()`). In `vllm/v1/worker/gpu/model_runner.py` it is created at 296.
  `forward_end()` (891), `drafter_start()` (896), `drafter_end()` (940) sit in the `_dummy_run` path only.
  `collect()` wraps the capture-time dummy runs (1074). In `execute_model` the real-step calls
  (`record_batch` 1976, `forward_start` 1979) are no-ops because `_collecting` is False.
  Its only consumer is `spec_decode/adaptive_verification.py` (20, 191, 212), which builds
  AdaptiveVerification cost curves.
- `~/vllm` HEAD `4bd75e952a`: same class (`async_utils.py:28/48`), same hook layout in
  `model_runner.py` (created 324, `forward_end` 935, `drafter_start` 940, `drafter_end` 976,
  `collect` 1121, `record_batch`/`forward_start` 2008/2011). Nothing newer exists to backport.
- Coverage: forward and drafter only, only in capture-time dummy runs. No accept phase, not wired
  into real `sample_tokens`/`propose`, no aggregation or logging, and it syncs.

Choice: reuse the idea and the hook placement (the new `begin()` sits right after the existing
`forward_start()` line), in a separate module `spec_timing.py`. The upstream class is untouched.
Extending it would change code AdaptiveVerification depends on, and would still need new real-path
hooks, a rejection-sampler interval, a streaming (no-sync) resolution mode and aggregation.

## Design

Six `torch.cuda.Event` marks per step, recorded on the current stream (GPU timeline, not host timers):

| mark | where (patched tree, `model_runner.py`) |
|---|---|
| 0 | `begin()` after `self.step_timing.forward_start()` in `execute_model` (1989), skipped for `dummy_run` |
| 1 | before `kv_connector.finish_forward()` (2054), after the target forward / graph replay |
| 2 | `sample()` after `compute_logits`, before the sampler (1618) |
| 3 | `sampled()` after the sampler branches, before the TP shard gather (1639) |
| 4 | `sample_tokens`, before `speculator.propose` (2250). `end()` records it if no speculator ran |
| 5 | after `propose` (2268); `end()` at 2302 just before `return async_output` |

Other edits: import at 90, `self._spec_timer = make_spec_timer(self.max_num_reqs)` at 299.

Phases (each is the GPU-timeline span between two marks):

| phase | span | meaning |
|---|---|---|
| verify | 0-1 | target forward (full graph replay or piecewise) |
| logits | 1-2 | `compute_logits` |
| accept | 2-3 | sampler + rejection sampling + acceptance |
| other | 3-4 | everything between sampler and `propose` |
| draft | 4-5 | whole `speculator.propose()` |
| total | 0 to last mark | sum of the above |
| gap | previous step's last mark to this step's mark 0 | idle/host time between steps (scheduling, host work). Valid only for contiguous steps |

Hot-path rules:

- No sync. Steps sit in a fixed ring of slots (default 64, `HQ_SPEC_TIMING_SLOTS`). A slot resolves only
  when `Event.query()` on its last mark is true, checked in `end()` (after `propose` is queued, so the
  bookkeeping overlaps GPU work instead of delaying the next forward launch; `begin()` resolves only
  when the ring is full). A full ring drops the new step and counts it in `dropped`; no `gap` is
  measured across a dropped step. A `begin()` while a step is open counts
  `abandoned`. Exit-time `flush()` syncs once.
- Token counts: `sampled()` copies `num_sampled` into a pinned int buffer with an async copy. This needs
  pinned memory (`is_pin_memory_available()`); otherwise token counts are absent (`tokens:false`).
  Pinned buffers for all slots are allocated once, on the first `sampled()` call (warm-up traffic),
  because a `cudaHostAlloc` on the hot path would sync the device. Consequence: the first spec step's
  `other` phase is inflated once. If the dtype changes or `n > max_reqs`, that step's token count is skipped.
- Off unless `HQ_SPEC_TIMING` is set and not `0`. `make_timer()` returns None and imports neither torch
  nor vllm in that case. Each hook is one `is not None` check. The env vars are read from `os.environ`
  with an `HQ_` prefix and are not registered in `vllm.envs`, so they stay out of the torch.compile cache
  key and the compile cache is shared with the stock baseline (same approach as
  `compile-key-runtime-knobs.patch`).

Aggregation (cumulative, never reset), keyed `kind/rN/G|E`: `kind` is `spec` if `num_draft_tokens > 0`
else `plain`; `rN` is the request count (capped at 17, so `r17` = 17 or more); `G` = FULL cudagraph
mode, `E` = anything else (piecewise or eager, e.g. mixed prefill+decode). Per key: `n`, `gap_n`, `tok`,
`drafts`, `emit`, `ms{phase}` sums and `sq{phase}` sums of squares. Histograms are per (kind, phase) with
edges `EDGES_MS = (0.05,0.1,0.2,0.5,1,2,3,4,5,7.5,10,15,20,30,50,100,250)` plus an overflow bucket.

Emission: one log line `HQ_SPEC_TIMING {json}` every `HQ_SPEC_TIMING_EVERY` resolved steps (default 200),
and at least every 30 s while steps keep arriving. One `HQ_SPEC_TIMING_START` line at creation. JSON fields:
`v` (schema, 1), `run` (8 hex, one per timer instance), `pid`, `ts` (unix s), `mono`, `steps` (resolved),
`dropped`, `abandoned`, `tokens` (bool), `keys`, `hist`. All numbers are cumulative, so a run window is
(last line) minus (baseline line); `parse_timing.py` does that.

Prometheus (optional, cheap): `HQ_SPEC_TIMING_PROM_PORT=<port>` starts a separate
`prometheus_client.start_http_server` inside the worker, exporting `hq_spec_timing_seconds_bucket{kind,phase}`
and `hq_spec_timing_emitted_tokens_total`. Publish that port in compose. Tested by hand on the research box
with a temporary pip install (CPU, fake events), not in `checks.out`. It breaks if `PROMETHEUS_MULTIPROC_DIR`
is set (not set by the launcher unless `api_server_count > 1`). The JSON log line is the primary output.

## CUDA graphs

- Events are recorded on the stream outside captured regions: around the FULL target-graph replay and
  around `propose`. Nothing is recorded in `_dummy_run` or during capture, and `begin()` checks
  `is_current_stream_capturing()`.
- Work inside a captured graph cannot be split. For DFlash2 the FULL graph holds
  `precompute_context_kv` + `_generate_draft` (drafter forward, `compute_candidates`, selector walk,
  `_apply_lookup`), so `draft` is the whole `propose()` span, not the drafter forward alone. For MTP
  (`AutoRegressiveSpeculator.propose`) it is the prefill pass plus k-1 decode graph replays and the host
  metadata building between them.
- Spans are GPU-timeline spans. For piecewise or eager steps (key flag `E`) they include idle launch gaps.
- Async scheduling is the launcher default (`start_qwen.sh:851-852`, `ASYNC_SCHED` defaults to 1). It is off
  only for the DFlash2 lookup with more than 7 draft tokens (`start_qwen.sh:430-435`). Record `ASYNC_SCHED`
  per arm. Without async scheduling, host work is on the critical path and shows up in `gap` and, for eager
  parts, inside the spans. Steps stay correctly attributed in both modes: the runner keeps one
  `execute_model_state`, so `execute_model`/`sample_tokens` always run in pairs.
- A step with no speculator (SPEC=off) has no `draft` phase and kind `plain`. With SPEC on, `plain`
  steps (prefill chunks, the first step after a prefill) still run `propose` and do have a `draft` phase.
- `acc/draft` assumes every request in the step emits 1 + accepted tokens. A non-final prefill chunk in a
  mixed step emits 0, so mixed (`E`) spec keys read low; use the `G` keys for acceptance.
- Model Runner V2 only (the default for DFlash2/MTP). If the V1 runner is picked, no `HQ_SPEC_TIMING_START`
  line appears.
- WSL2/WDDM may delay `query()` becoming true. The ring absorbs this; `dropped` shows any loss.

## Overhead (estimated, not measured)

- Host per step about 100-150 us: 7 `Event.record` (6 marks + the host allocator's event for the pinned
  copy), 1-2 `query`, one small async memcpy, 7 `elapsed_time` calls (5 phases, total, gap) and dict
  updates. WSL2 driver calls may cost 2-5x more.
- If all of it were serial with the GPU, at about 25-30 ms steps: about 0.4-0.6%, upper estimate 0.7%, up
  to about 1% on WSL2. After the review only `begin()` (one capture query, one record, about 10-20 us)
  is on the launch path. The rest runs in `sample_tokens` and `end()` while the step's work is still
  queued, so the expected cost is lower than that bound. Both numbers are estimates.
- WDDM/WSL2: `Event.query()` may flush the driver's batched submissions, which can shift timing either
  way. The A/B below is the only check that covers this.
- GPU side under 0.1% (event records are tiny stream entries). Log volume about 3 KB per 200 steps.
- Verify on the box: A/B at C1 of stock image vs variant with `HQ_SPEC_TIMING` unset, and variant with `=1`.
  Expect the difference within the greedy +-3% noise. A built-in cross-check: the parser's `implied_tok_s`
  should match `sweep.py` tok/s within a few percent.

## Build and run on the GPU box

1. Build (never `docker compose --build`):
   ```
   cd docs/research/04-benchmarks/experiments/X-17-timing
   docker build -f Dockerfile.variant -t hyperqwen:sha-2cf50a0-spectiming .
   ```
   The build prints `spec_timing: inert when HQ_SPEC_TIMING is unset` and fails if the patch does not apply.
2. Compose override (the base compose pins `image: ghcr.io/syv-ai/hyperqwen:latest`); chain it after the box's
   own override and the harness compose file:
   ```yaml
   services:
     single:
       image: hyperqwen:sha-2cf50a0-spectiming
       pull_policy: never
   ```
3. `.env`: add `HQ_SPEC_TIMING=1` for variant arms, leave it out for baseline arms. Optional:
   `HQ_SPEC_TIMING_EVERY=200`, `HQ_SPEC_TIMING_SLOTS=64`, `HQ_SPEC_TIMING_PROM_PORT=<port>` (also publish it).
4. Run the harness suite as usual: pin `PTH_FA`, same SALT, interleave arms.
5. Parse:
   ```
   docker compose logs single | python3 parse_timing.py --label X [--t0 UNIX --t1 UNIX] --out spectiming_X.json
   python3 parse_timing.py container.log --windows     # one row per emitted window
   ```
   `--t0/--t1` are unix seconds matched against each line's `ts`. Without them the whole boot is summarised,
   warm-up included.
6. Smoke test first: the `HQ_SPEC_TIMING_START` line appears with `"tokens":true` (on WSL2 that needs
   `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, already required by `.env.example:45`), `dropped` stays 0, and
   `implied_tok_s` matches `sweep.py`. The entrypoint's `verify.sh --no-server` gives the same patch result
   as on the stock image (`checks.out` section 3b).
7. Then the stock-vs-variant overhead A/B above.

## Parser output fields

Per kind (`spec`/`plain`, "all") and per key row: `n` timed steps; mean ms of `gap verify logits accept other
draft total`; `tok/step` (emitted tokens per step); `acc/draft` (accepted drafts per draft token, from
`emit - requests*n` over `drafts`; shown only when all keys have fewer than 17 requests); `impl tok/s` =
emitted tokens / (total + gap), the decode rate implied by the event timeline. The footer gives each phase's
share of `total` and p50/p95 as the upper edge of the histogram bucket. The `--out` JSON also has `sd_ms`,
`sum_ms`, `emitted_tokens`, `drafts_per_step`.

## Patch baseline and apply results

- Tree: pristine v0.31.0 export, then `patches/series` in order (`series-apply.out`): 51 of 51 applied, 4 with
  hunk offsets only (engine-stall-sentinel, triton-spec-attn-fp8-kv, compile-key-runtime-knobs,
  qwen3_5-embed-uva), none failed. The four KVarN patches (`kvarn/install.sh`) also applied (offsets: kvarn-0.31.0 2 hunks,
  kvarn-v2-runner 1, kvarn-recycled-pages 1, kvarn-fp16-dequant 2).
- Image tree check: applying upstream's `patches/` + `kvarn/` at commit `2cf50a0` (read-only `git archive`)
  to the pristine export gives a tree with no `diff -rq` differences from our baseline, and the patch
  dry-ran cleanly on it. The review re-check (`checks.out` section 1b) is narrower: `patches/` and `kvarn/` are
  byte-identical between `2cf50a0` and this branch, so the image tree equals the baseline. This is evidence,
  not a look inside the pulled image.
- `checks.out`: `git apply --check` OK, `patch --fuzz 0` dry-run OK, real apply byte-identical to the scratch
  tree, second apply fails, `verify.sh`'s patch checks give the same result as on the baseline (3b), byte-compile
  OK (python 3.13 here, image uses 3.12), `test_timing.py` prints OK, parser demo on synthetic lines.

## Open questions

- Whether `draft` must be split into drafter-forward vs the rest for DFlash2. It cannot be inside the FULL
  graph; it would need a non-graph (eager) run or a profiler pass.
- The first spec step per process has an inflated `other` (pinned-buffer allocation); use `--t0` after warm-up.
- The `r17` cap makes `acc/draft` unavailable for large batches.
- The image-tree equivalence is inferred from upstream sources, not verified against the pulled image. The
  Dockerfile build step fails loudly if the patch does not apply.

Re-run by: X-17 reviewer, 2026-10-21, PASS after fixes. All CPU checks re-run (`checks.sh` > `checks.out`). Fixed:
no `gap` across a dropped step, resolution moved from `begin()` to `end()`, and the `LABEL` ARG scope plus a
package-path guard in `Dockerfile.variant`. Review: `docs/research/00-meta/reviews/phase4-x17-timing.md`.
