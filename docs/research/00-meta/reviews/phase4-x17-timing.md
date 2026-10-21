# Phase 4 review: X-17 draft/verify/accept GPU timing (D12)

Reviewer: X-17 reviewer (Opus), 2026-10-21. Scope: `docs/research/04-benchmarks/experiments/X-17-timing/` and the
scratch tree `~/research-wt/vllm-0.31.0-timing` (D14). No GPU was used. Every result below is a **measured CPU
result** on the research box, or **estimated** where marked.

## Verdict

**READY FOR GPU.** The reviewer's fixes are applied, the patch is regenerated, and every check is re-run.

The patch is correct on the real hot path. It does not sync the host. It changes nothing when `HQ_SPEC_TIMING` is
unset. It never records inside a CUDA-graph capture, and it attributes steps correctly with or without async
scheduling. I found and fixed two code bugs (F1, F2) and one Dockerfile bug (F3). I also corrected one wrong claim in
the README that affects how results are read (F4). The other findings are documentation or known limits.

## Re-run outputs (measured, CPU)

`bash checks.sh > checks.out`. `checks.sh` is new and holds the commands, so the checks can be re-run. Summary of
`checks.out` on the final files:

| check | result |
|---|---|
| `git apply --check -p1` against scratch commit `20c7625` (v0.31.0 + 51 series + 4 KVarN) | OK |
| `patch -p1 --fuzz 0 --dry-run` | exit 0 |
| `patches/` + `kvarn/` at `2cf50a0` (the stock image commit) vs this branch | byte-identical (`git diff --quiet`), so the image tree equals the baseline |
| real apply with the `Dockerfile.variant` flags, compared with scratch `HEAD` `0bc5600` | both files identical |
| second apply (`--forward`) | exit 1, as expected |
| verify.sh's patch checks (the entrypoint runs `verify.sh --no-server` on every boot), baseline vs variant tree | same result on all 55 patches (45 exact reverse, 9 content check, 1 "FAIL" = `spec-decode-scratch-token-units.patch`, which verify.sh accepts as superseded by `spec-decode-scratch-within-budget.patch`, the same on stock) |
| byte-compile `model_runner.py`, `spec_timing.py`, `parse_timing.py`, `test_timing.py` (py 3.13.7; the image uses 3.12) | OK |
| `test_timing.py` | OK (two new cases, 3b and 3c) |
| `parse_timing.py` on a synthetic log (fake events) | renders. spec: verify 14.00, accept 1.80, draft 3.90 ms, 4.00 tok/step, acc/draft 0.429. These equal the inputs. |

Compared with the original `checks.out`, every pre-fix check still passes. The only difference in the parser demo
is that the plain row's `gap` is now `-`: the first step has no predecessor, and the synthetic log differs.

Extra checks, not in `checks.out` (measured, CPU):
- **Dockerfile RUN simulation.** I ran the same commands on a local copy of the baseline tree with `PYTHONPATH` set:
  the `SP` derivation, the guard, `patch --forward --fuzz 0`, and the inert import (`make_timer() is None`). All
  passed. `bash -n` on the RUN body passed.
- **Enabled path imports.** `HQ_SPEC_TIMING=1 make_timer(8)` imports `vllm.logger` and
  `vllm.utils.platform_utils.is_pin_memory_available` on the patched tree. It returns a `SpecTimer` (64 slots,
  `every` read from the env).
- **Mutation check.** I restored the pre-fix `_seq` placement and ran test 3b. It fails with `gap_n 2` and `gap` 57 ms
  where 1 ms is correct.
- **Original `checks.out`.** Before any edit, I reproduced it exactly: apply-check OK, dry-run 0, byte-identical
  apply, second apply 1, py_compile OK, tests OK.

## Findings

Severity: High = wrong numbers or a broken boot; Medium = misleading numbers in a plausible case; Low = cosmetic,
docs, or a minor bias; Info = verified correct or a known limit.

| # | sev | where | finding | status |
|---|---|---|---|---|
| F1 | Medium | `spec_timing.py` `begin()`, pre-fix :110-118 | `_seq` was incremented after the ring-full check, so a dropped step took no sequence number. The next timed step then looked contiguous with the step before the drop, and `gap` absorbed the whole dropped step. Reproduced: 57 ms where 1 ms is correct. This happens only when the ring is full, which is the WSL2 `query()` lag case the README names. | **Fixed**: `_seq += 1` before the drop check (`spec_timing.py:112`). Regression test 3b. |
| F2 | Low-Med | `spec_timing.py` `begin()`, pre-fix :103 and :122 | `_resolve()`, then `_consume()` (7 `elapsed_time` calls, a pinned-buffer `sum`, dict updates), and the 30 s `emit()` all ran at the top of `begin()`. That is on the forward-launch path, where the GPU can be idle (sync scheduling, or when the previous `propose` has finished). Estimated 100-150 us per step, serial. | **Fixed**: resolution and the timed emit moved to `end()` (`spec_timing.py:157-169`). At that point the step's own `propose` is still queued, so the host work overlaps the GPU. `begin()` resolves only as a fallback when the ring is full (:113-117). Test 3c covers it, and tests 1-9 now drain with `end()`. |
| F3 | Low | `Dockerfile.variant:10-11` | `ARG BASE` was declared only before `FROM`, so `${BASE}` was empty in the `LABEL` after it. Also, an empty `SP` (vllm import failure masked by `| tail`) only failed indirectly. | **Fixed**: `ARG BASE` is re-declared after `FROM`. An explicit `[ -f "$SP/v1/worker/gpu/model_runner.py" ]` guard now stops the build with a message. |
| F4 | Medium (doc) | X-17 `README.md`, CUDA graphs section | The README said "the launcher uses `--no-async-scheduling` (`start_qwen.sh:851`)". Wrong. `start_qwen.sh:851-852` makes async scheduling the default (`ASYNC_SCHED` defaults to 1). It is off only for the DFlash2 lookup with k>7 (`start_qwen.sh:430-435`). This changes how `gap` and overhead should be read. | **Fixed** in the README; record `ASYNC_SCHED` per arm. Attribution is correct in both modes. The runner holds one `execute_model_state` (`model_runner.py:349`, set at `:2074` at the end of `execute_model`, cleared at `:2113` at the start of `sample_tokens`), so `execute_model`/`sample_tokens` always alternate and `begin..end` never interleave. |
| F5 | Low (doc) | README, test 2 comment | The docs said `plain` steps have no `draft` phase. With SPEC on, `propose` also runs on prefill steps (`model_runner.py:2250-2268` runs whenever `self.speculator` is set), so `plain/…` keys do have a `draft` phase. Only SPEC=off lacks it. | **Fixed** in the README and the test comment |
| F6 | Low | `parse_timing.py` `accept_rate` | `(emit - reqs*n)/drafts` assumes every request emits 1 + accepted tokens. A non-final prefill chunk in a mixed step emits 0 (`get_num_sampled_and_rejected`, `input_batch.py:535`), so `acc/draft` on mixed `spec/…/E` keys reads low. | Documented. Use `G` keys, or `sweep.py`'s `/metrics` rate, for acceptance. |
| F7 | Info | `spec_timing.py:312` `pin=is_pin_memory_available()` | On WSL2, vLLM returns `envs.VLLM_WSL2_ENABLE_PIN_MEMORY` (`platforms/cuda.py:304-322`). Without it, `tokens:false` and there is no tok/step, acc/draft or implied tok/s. `.env.example:45` already sets it, and DFlash2 requires it. | README smoke step now checks `"tokens":true` |
| F8 | Info | entrypoint `verify.sh --no-server` | An added hunk inside another patch's context would fail verify.sh's exact reverse dry-run, and the entrypoint would abort the boot. Simulated on both trees: the result is identical (table above). | Verified, no change |
| F9 | Info | `model_runner.py:299` | The timer exists only in Model Runner V2 (`v1/worker/gpu/model_runner.py`). V2 is the default unless a feature is unsupported (`config/vllm.py:701-761`). On a V1 fallback, nothing is logged. | README: no START line means the V1 runner |
| F10 | Info | `model_runner.py:2250/2268` | Limit: `draft` is the whole `propose()`. For DFlash2 that is the host chain state machine, `precompute_context_kv` and the `_generate_draft` FULL graph. For MTP it is the prefill pass plus k-1 decode replays with host metadata between them. GPU-timeline spans include launch gaps. This is enough for the P1 draft-vs-verify split. A drafter-forward-only number needs the profile phase. | Accepted, documented |
| F11 | Low | `spec_timing.py` `EDGES_MS` | p50/p95 are histogram bucket upper edges. Buckets are coarse where verify lives (10/15/20/30 ms), so a verify p50 of "15" means somewhere in 10-15 ms. Means and SDs are exact sums. | Not changed. If percentiles matter, add edges (12.5, 17.5, 25) to both `EDGES_MS` and `DEFAULT_EDGES`; the test asserts they are equal. |
| F12 | Info | WSL2/WDDM | `Event.query()` can flush the driver's batched submissions, which may move step timing either way. Only the stock-vs-variant A/B can show it. | README overhead section notes it |

### Hot path (task item 2), verified

- **No host sync when enabled.**
  - Hooks only record events: `mark` → `Event.record` on the current stream.
  - `sampled()` (`spec_timing.py:131-145`) does a `non_blocking` D2H copy into pinned memory, and only when pinned
    memory is available. Without it, there is no copy, because an unpinned `non_blocking` copy would sync.
  - `num_sampled.shape[0]` is host metadata.
  - `int(buf.sum())` runs on a host tensor, and only after `query()` reports the slot's last mark complete. All
    marks are on one stream, and the same-stream copy precedes the last mark.
  - `elapsed_time` is only called on completed events (`:193-197`). The gap pairs the previous slot's last mark
    (complete) with this slot's mark 0, which precedes this slot's complete last mark.
  - No `.item()`, `.cpu()` or `synchronize` in hooks. `synchronize` is used only in exit-time `flush()` (`:259-267`).
  - The only device-syncing allocation is the one-time pinned allocation on the first `sampled()` (`:147-155`). It
    is documented, and the first window should be excluded with `--t0`.
  - No stream switch was found: `main_stream` is the current stream (`model_runner.py:564-566`), and
    `use_workspace_lane` and the speculators do not change it (only `spec_decode/utils.py` and
    `adaptive_verification.py` use their own copy streams).
- **Zero change when unset.**
  - `make_timer` returns `None` before importing torch or vllm (`spec_timing.py:295-298`).
  - The module import (`model_runner.py:90`) loads only stdlib.
  - Each hook is one `is not None` test.
  - The `HQ_*` variables are not in `vllm.envs`, so the compile-cache key is unchanged.
- **CUDA graphs.**
  - `begin` is skipped for `dummy_run` (`model_runner.py:1989`). Capture and warm-up go through
    `_dummy_run → execute_model(dummy_run=True)` (`:868-878`) and `_dummy_sampler_run` (`:950-961`, which calls the
    sampler directly, not `sample()`), so no step is open during capture. The other hooks do nothing without an
    open step, and `begin` also checks `is_current_stream_capturing()`.
  - Replay happens on the current stream (`cudagraph_utils.py:561-574`), so marks 0 and 1 bracket the replay on the
    GPU timeline. The verify span includes `get_offloader().sync_prev_onload()` when offload is active, which is
    correct.
- **Async / overlapped steps.**
  - Steps are paired as described in F4.
  - With async scheduling, step N+1's mark 0 is queued before step N completes. The ring holds both, and `gap`
    becomes the true GPU idle time between them.
- **DFlash2 and MTP.** Both use the single real `self.speculator.propose` call (`model_runner.py:2252`). The dummy
  `propose` (`:921`) is not timed.
- **Non-spec steps.** `end()` records mark 4 when no speculator ran, so the step has `other` but no `draft`.
- **Prefill-only steps.** These are `plain`. With SPEC on they have a `draft` phase (F5). With chunked prefill they
  are mostly `E`.
- **Placement.** Mark 0 comes after `prepare_inputs`/`prepare_attn`, so input staging and attention-metadata
  building count in `gap`, not in `verify`. `compute_logits` and the grammar bitmask are in `logits`.
  `prompt_logprobs`, `postprocess_sampled` and the async output copy are in `other`.

### StepTimingCollector claim (task item 3), verified

The claim is correct for both trees:

| | v0.31.0 + patches (scratch) | `~/vllm` `4bd75e952a` |
|---|---|---|
| classes in `async_utils.py` | `:28/41/48` | `:28/41/48` |
| `collect()` | `:62` | `:62` |
| sync at block exit | `:78` | `:78` |
| `forward_end` / `drafter_start` / `drafter_end` (dummy path) | `model_runner.py:894/899/943` (patched numbering) | `model_runner.py:935/940/976` |
| `collect()` around capture | `:1077` | `:1121` |
| real-path `record_batch` / `forward_start` | `:1985/1988`, no-ops outside `collect()` | `:2008/2011` |

The only consumer is `adaptive_verification.py:20,191,212`. The collector has no accept phase and no real-path
draft hooks, and it syncs. Upstream has no draft-time metric to backport.

### Dockerfile (task item 4), verified

- **Install path.** The base image installs `vllm==0.31.0` into `/app/venv` (`Dockerfile:23-26`,
  `docker/requirements.txt:5`). The patch step finds the package directory with the same `SP=$(… vllm.__file__ …)`
  as the variant (`Dockerfile:49`), then runs `patches/apply.sh "$SP"` and `kvarn/install.sh`.
- **Variant.** `Dockerfile.variant` uses `/app/venv/bin/python` and the same `SP` derivation. GNU `patch` is in the
  base image (`Dockerfile:20`), and the build runs as root.
- **Fails loudly.** The variant RUN uses `set -e`, the new `SP` guard, and `patch --forward --fuzz 0`. `py_compile`
  and the inert-import assert must also pass.
- **Residual risk.** I did not look inside the pulled image. The `2cf50a0` equivalence is source-level evidence.
  The build itself is the check: it fails if the patch does not apply.

### Overhead (task item 5)

The original 0.4-0.7% (estimated) is a fair upper bound if all the host work is serial with the GPU: about
100-150 us per step against 25-30 ms steps. The README counted about 12 `elapsed_time` calls; the code makes 7
(corrected). After F2, only `begin()` is on the launch path: one capture query and one record, about 10-20 us
(estimated). So the expected cost is below the bound. GPU-side event cost is under 0.1% (estimated). The README's
A/B (stock vs variant with the variable unset vs `=1`, at C1, within the ±3% greedy noise) is the right gate,
especially given F12.

### Output fields (task item 6)

Per `kind/rN/G|E` key and per kind, the output gives:
- mean, SD and bucket p50/p95 for `verify`, `accept`, `draft`, plus `logits`, `other`, `total` and `gap`;
- emitted tokens per step (summed over requests) and draft tokens per step;
- `acc/draft` (caveat F6);
- `implied_tok_s`, to cross-check against `sweep.py`.

This covers the P1 metrics that `harness/README.md:93-106` says are missing: draft ms, verify ms and accept ms per
step, and tokens per step. It also fills the gap in `gpu-agent-prompt.md:540`. For r>1, divide tok/step by N to get
tokens per request-step. Percentiles are coarse (F11).

## Every edit

Each edit is a fix to a verified bug or a correction of verified-wrong text.

1. Scratch repo `~/research-wt/vllm-0.31.0-timing`:
   - commit `7be5728` records the patch as drafted, before review;
   - commit `0bc5600` holds F1 and F2 in `vllm/v1/worker/gpu/spec_timing.py`: `begin()`, `end()` and the module
     docstring. `model_runner.py` is unchanged, so every hook line number in the README still holds.
2. `X-17-timing/spec-timing.patch`: regenerated with `git diff --relative=vllm/ 20c7625`. The header is kept and
   `spec_timing.py` grows from 319 to 326 lines. The diff body equals scratch `HEAD`.
3. `X-17-timing/test_timing.py`:
   - drains now use `end()` (tests 1, 2, 3, 6, 7);
   - the test 2 comment is fixed;
   - new 3b (no gap across a dropped step) and 3c (`begin()` does not resolve; `end()` does).
4. `X-17-timing/Dockerfile.variant`: `ARG BASE` re-declared after `FROM`, and the `SP` guard added (F3).
5. `X-17-timing/README.md`:
   - `checks.sh` added to the file table;
   - hot-path rules updated (F1, F2);
   - async-scheduling claim corrected (F4);
   - the `plain` draft phase (F5), `acc/draft` bias (F6) and V2-only notes (F9) added;
   - overhead section corrected (7 `elapsed_time` calls, the post-F2 launch-path cost, WDDM);
   - smoke step gains `"tokens":true` and the verify.sh note (F7, F8);
   - the 1b and check-summary text updated;
   - "Re-run by" line appended.
6. `X-17-timing/checks.sh`: new, the script behind `checks.out`.
7. `X-17-timing/checks.out`: refreshed from `checks.sh`.

Not touched: `~/vllm`, `~/research-src`, the project `patches/` and `Dockerfile`, `series-apply.out`,
`parse_timing.py`. No commits in the main project.
