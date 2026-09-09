# Project Investigation

Snapshot: 2026-09-08, commit `231592a7009d0d7c876189f7d1f72d8cc791c76c`, including the uncommitted working-tree changes present during inspection.

This is a source-based engineering assessment, not a fresh GPU qualification. The companion operating process is [self-improvement-loop.md](self-improvement-loop.md).

## Executive conclusion

This repository is an ambitious, evidence-rich vLLM distribution for serving Qwen3.8-27B on a 24 GB RTX 3090. Its strongest work is in narrow runtime optimization, failure archaeology, and GPU-specific correctness oracles. It records negative findings instead of hiding them and often provides safe fallbacks.

The largest gap is that the project can discover subtle defects but cannot yet reliably prevent their publication. Build, preparation, benchmark, and promotion checks use different result contracts. Several checks print failure and exit successfully, benchmark identity is incompletely recorded, and some headline metrics measure a proxy rather than the property their label implies. There are also confirmed current blockers and safety defects in patching, model preparation, authentication, and an optional KVarN path.

The next improvement should not be another optimization. First make every experiment attributable, every invalid run fail closed, every model publication transactional, and every promoted claim pass explicit correctness, quality, latency, memory, and reproducibility gates.

## Finding classes

- **Reproduced**: exercised with a safe local command or isolated mock during this investigation.
- **Confirmed**: follows directly from the current source, but the affected runtime path was not executed.
- **Conditional**: a confirmed defect on an optional path or under a stated configuration.
- **Historical evidence**: supported by retained artifacts, not re-measured against the current stack.
- **Hypothesis**: plausible from source and important enough to test, but not yet established as a defect.

## Priority summary

| ID | Priority | Class | Finding | Immediate action |
|---|---|---|---|---|
| F01 | P0 | Reproduced | The dirty `hybrid-sw-block-promote.patch` is malformed and cannot be installed | Regenerate the patch while preserving its intended logic |
| F02 | P0 | Confirmed | Benchmark and correctness tools can print failure and exit zero | Introduce one fail-closed result contract |
| F03 | P0 | Confirmed | The main cohort does not explicitly disable model thinking as documented | Version the workload and freeze rendered prompts |
| F04 | P0 | Confirmed | Model preparation and variant publication are non-transactional and can mutate source artifacts | Prepare under a lock into a new generation, validate, then publish atomically |
| F05 | P0 | Confirmed | Benchmark provenance is insufficient to reproduce or compare a run | Write an immutable run manifest before measurement |
| F06 | P1 | Reproduced/confirmed | Optional authentication conflicts with the new mandatory Headroom service and broad binds can be unauthenticated | Separate Headroom into its own profile and validate exposure policy |
| F07 | P1 | Conditional | KVarN materialized decode can exceed its scratch allocation | Add a capacity guard and tested fallback |
| F08 | P1 | Confirmed | Calibration code can consume evaluation test data | Pin disjoint splits and enforce sample-hash separation |
| F09 | P1 | Confirmed | `C / TPOT` is presented as decode throughput without proving concurrent decode residency | Rename the proxy and promote measured throughput instead |
| F10 | P1 | Confirmed | Launch, A/B, patch, and shared-memory ownership is unsafe or ambiguous | Use per-run ownership, locks, bounded probes, and owned teardown |
| F11 | P1 | Confirmed | Quality and acceptance scripts produce scores, not enforceable gates | Require coverage, per-item evidence, matched baselines, and nonzero failure exits |
| F12 | P1 | Confirmed | Dependency, model, and server identities are not bound to retained results | Pin inputs and record client/server identities separately |
| F13 | P2 | Confirmed | Several launcher controls are silently ignored or resolve unexpectedly | Build one validated effective-configuration resolver |
| F14 | P2 | Confirmed | Training and export can accept incomplete data or attach the wrong vocabulary IDs | Publish complete datasets and bundle row identity with weights |
| F15 | P2 | Confirmed | CI validates installation but not benchmark logic or current GPU behavior | Add CPU fixture CI and a separate release GPU gate |

## Critical findings

### F01: Current patch series is not installable

**Reproduced.** `git apply --numstat -- patches/hybrid-sw-block-promote.patch` reports `corrupt patch` at line 213. The dirty edit added logic without updating its hunk length. Docker applies all patches with failure propagation, so a build from this exact working tree stops in [Dockerfile](../Dockerfile#L28-L36).

The committed version parses; this is specifically a working-tree blocker. The intended granularity fix should be retained when regenerating the patch.

**Gate:** every patch parses strictly, the complete ordered series applies to a disposable checkout of the pinned vLLM tag, and the resulting tree passes semantic verification.

### F02: Failure does not consistently mean a failing process

**Confirmed; selected paths reproduced with offline mocks.** The repository has several executable checks that only print a failure:

- [api_smoke.py](api_smoke.py#L37-L43) catches request failures, prints a summary at [lines 100-106](api_smoke.py#L100-L106), and does not exit nonzero. An offline mock made all 12 requests fail; it printed `0 / 12 passed` and returned normally.
- [needle_test.py](needle_test.py#L65-L68) prints `MISSED` without failing.
- [test_spec_decode_attn.py](test_spec_decode_attn.py#L65-L89) prints numerical `FAIL` and continues to timing.
- [residue_sweep.py](residue_sweep.py#L99-L107) and [bugb_sweep.py](bugb_sweep.py#L96-L105) report broken cases without making failure machine-readable.
- [run_benchmarks.sh](run_benchmarks.sh#L14) uses only `set -u`; benchmark calls and expected success counts are not validated before rows are published at [lines 53-90](run_benchmarks.sh#L53-L90).

This makes CI integration unsafe and allows invalid evidence to look complete.

**Gate:** every tool ends in exactly one of `PASS`, `FAIL`, `SKIP`, or `INVALID`; `FAIL` and `INVALID` exit nonzero; expected request and result counts are mandatory; no required measurement may be empty, nonfinite, or silently skipped.

### F03: The documented cohort workload is not actually fixed to thinking-off

**Confirmed.** The custom cohort command in [run_benchmarks.sh](run_benchmarks.sh#L61-L70) does not pass chat-template kwargs. The local vLLM benchmark client renders the custom dataset client-side, while the checked-in model template enables thinking when `enable_thinking` is absent. The README describes the corresponding workload as thinking-off at [README.md lines 560-565](../README.md#L560-L565).

The same omission exists in `real_rep.sh` and `prefill_ab.sh`. Consequently, historical custom-cohort output can include reasoning tokens. It cannot safely be relabeled after the fact.

**Gate:** create a new workload protocol version, explicitly set `enable_thinking=false`, save hashes of exact rendered prompt token IDs, and preserve old results under their old identity.

### F04: Model preparation is not an atomic publication pipeline

**Confirmed.** Preparation modifies model directories in place and publishes shards, indexes, and configuration separately. For example, [quant_lm_head.py](../prepare/quant_lm_head.py#L59-L90) replaces these components in separate operations, while [docker/prepare.sh](../docker/prepare.sh#L23-L36) infers readiness from a small set of files and keys. There is no model-directory lock.

Two concrete data-integrity hazards compound this:

- [gptq_lm_head.py](../drafter/gptq_lm_head.py#L69-L76) hardlinks variant artifacts that [build_draft_vocab.py](../prepare/build_draft_vocab.py#L119-L126) later rewrites. The documented variant pipeline can therefore mutate its source inode.
- [quant_heads_stream.py](../prepare/quant_heads_stream.py#L166-L202) can overwrite the pristine `.bak-orig` with an already modified shard when MTP and head tensors share one physical shard.

An interrupted preparation can leave a directory that looks ready enough to be selected but is internally inconsistent.

**Gate:** acquire a model lock, prepare in a new generation directory, preflight all source layout assumptions before mutation, avoid writable hardlinks, verify every indexed tensor and quantization declaration, hash outputs, then atomically publish a completion manifest or pointer.

### F05: A benchmark result does not identify the system that produced it

**Confirmed.** The main benchmark uses `venv/bin/vllm` as its client at [run_benchmarks.sh lines 21-25](run_benchmarks.sh#L21-L25), even when the server is a container or remote process. This workspace's local client metadata reports vLLM 0.27.1, while [docker/requirements.txt](../docker/requirements.txt#L1-L5) pins server image vLLM 0.28.0. [verify.sh](../verify.sh#L26-L29) only warns about this mismatch.

Saved result rows do not bind the actual server version, image digest, applied patch tree, effective launch settings, model hashes, GPU identity, driver, power cap, cache state, or dirty diff. Fixed filenames are overwritten by subsequent runs at [run_benchmarks.sh lines 57-88](run_benchmarks.sh#L57-L88).

**Gate:** each run gets a unique immutable directory and manifest containing commit, dirty-diff hash, client and server versions, image digest, patch hashes, model/drafter/tokenizer/template/data hashes, effective arguments, GPU UUID, driver, power cap, cache lineage, timestamps, and artifact checksums. Secrets must be redacted.

## High-priority correctness and safety

### F06: Authentication and Compose profiles conflict

**Reproduced.** `docker compose --env-file .env.example -f docker-compose.yml --profile single config --quiet` fails because Headroom requires `VLLM_API_KEY` at [docker-compose.yml lines 100-107](../docker-compose.yml#L100-L107). Headroom shares the `single` profile at [lines 87-93](../docker-compose.yml#L87-L93), although the example key is documented as optional for local use at [.env.example lines 14-18](../.env.example#L14-L18).

Additional exposure gaps:

- Compose publishes the server port on all host interfaces at [docker-compose.yml lines 43-46](../docker-compose.yml#L43-L46), while keyless serving is supported.
- Headroom's upstream is fixed to port 18020 at [lines 103-106](../docker-compose.yml#L103-L106), even though `PORT` is configurable.
- [single-user/alternative.sh](../single-user/alternative.sh#L11-L23) can replace an injected key with an empty value if `api_key.txt` is absent, then bind broadly at [lines 81-83](../single-user/alternative.sh#L81-L83).

**Opportunity:** make Headroom opt-in under a separate profile; default published ports to loopback for keyless use; require explicit unauthenticated broad-bind opt-in; preserve environment credentials; make the proxy target follow the actual internal port.

### F07: Optional KVarN materialized decode lacks a scratch bound

**Conditional, source-confirmed.** KVarN caps and allocates scratch storage in [config.py](../kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py#L291-L316) and [kvarn_attn.py](../kvarn/files/vllm/v1/attention/backends/kvarn_attn.py#L1490-L1537). With `KVARN_FUSED_DECODE=0`, the materialization kernel writes rows for the sum of request context lengths at [triton_kvarn_decode.py lines 743-856](../kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py#L743-L856), without checking that this sum fits the allocation. The cached multi-query path already performs an equivalent capacity check at [kvarn_attn.py lines 2587-2591](../kvarn/files/vllm/v1/attention/backends/kvarn_attn.py#L2587-L2591).

**Gate:** prove `required_rows <= allocated_rows` before launch or use a bounded fallback. Add boundary, over-capacity, multi-request, eager, and captured tests.

### F08: Calibration and evaluation data are not isolated

**Confirmed route; historical contamination not established.** [collect_prompts.py](../drafter/collect_prompts.py#L114-L121) labels its GSM8K source as training data, but first recursively reads any parquet under the local GSM8K directory using [lines 18-25](../drafter/collect_prompts.py#L18-L25). That directory contains the same test parquet consumed by [quality_battery.py](quality_battery.py#L108-L113). Generated sequences then feed hidden-state capture and target `lm_head` calibration through the pipeline documented in [drafter/README.md](../drafter/README.md#L43-L53).

[act_calib.py](act_calib.py#L86-L94) also selects calibration text from evaluation domains.

The source proves an unsafe leakage route, not that the currently shipped weights were built through it.

**Gate:** pin dataset repository revision and explicit split files; produce a calibration manifest; hash normalized sample identities; fail if calibration, tuning, and evaluation sets intersect.

### F09: A latency-derived proxy is labeled as throughput

**Confirmed; historical artifact comparison.** The project computes `C / mean TPOT` in [run_benchmarks.sh](run_benchmarks.sh#L67-L70). This does not establish that all `C` streams decoded simultaneously and is not aggregate output tokens divided by wall time.

The retained C8 default-sampling rows illustrate the risk:

| Profile label | E2E output throughput | `C / mean TPOT` | Mean TTFT |
|---|---:|---:|---:|
| fast | 274.49 tok/s | 384.1 | 3.530 s |
| long | 273.31 tok/s | 643.6 | 7.605 s |
| Change | -0.43% | +67.56% | +115.45% |

Sources: local retained artifacts `bench/results/rerun2_stdout.log:5` and `bench/results/long2_stdout.log:5`. These paths are gitignored and are not durable published evidence. The proxy increases 68% while observed end-to-end throughput is flat and queue latency doubles. These cohorts also contain short prompts, so `fast` and `long` describe server context configurations, not tested prompt lengths.

**Opportunity:** retain this only as a clearly named cohort latency score. Report aggregate throughput as actual output tokens over wall time, per-request TPOT as latency, and steady-state decode only from a validated interval with simultaneous resident decoders.

### F10: Process and workspace ownership is ambiguous

**Confirmed.** [prefill_ab.sh](prefill_ab.sh#L26-L45) reuses any healthy server on a port without proving its configuration. It uses a persistent PID file and broad teardown at [lines 86-88](prefill_ab.sh#L86-L88), with no ownership-scoped trap. Different ports do not isolate one GPU.

The patch-series helper resolves a supplied package path to its Git root, resets tracked files, and cleans untracked files in [check_vllm_series.sh](../patches/check_vllm_series.sh#L24-L39). Its CI caller is disposable, but the helper does not enforce that precondition. It must not be run against this workspace.

Both launchers also infer stale shared-memory ownership by scanning `/proc` and delete when no mapping is observed, conflating absence with incomplete visibility: [single-user/start_qwen.sh](../single-user/start_qwen.sh#L57-L67) and [batch/start_qwen.sh](../batch/start_qwen.sh#L32-L42).

**Gate:** allocate a per-run directory and GPU lock, verify process start identity and effective configuration, bound all probes, reuse only by explicit opt-in, and tear down only processes created by that run. Patch validation must create or require a disposable worktree.

### F11: Quality and acceptance checks are descriptive, not gates

**Confirmed.** [quality_battery.py](quality_battery.py#L60-L65) silently loses the code lane outside a hardcoded Python 3.12 layout and changes the test corpus when the installed vLLM source changes. Missing prompt logprobs are skipped at [lines 72-89](quality_battery.py#L72-L89), with no expected token coverage or finite-value check. GSM8K retains only aggregate accuracy at [lines 108-121](quality_battery.py#L108-L121), preventing paired error analysis.

The September 8 artifacts do support similar short-window behavior: all three domain token counts match, aggregate PPL is `8.09123683` versus `8.09127665` (+0.00049%), and GSM8K is 192/200 versus 193/200. One additional answer at n=200 is not meaningful evidence of superiority, and these are not long-context quality tests.

[labd_accept.py](labd_accept.py#L249-L285) can have an empty target or zero clean chunks and still emit a nominal one token per step at [lines 303-315](labd_accept.py#L303-L315). [labd_soak.py](labd_soak.py#L109-L145) checks repeatability and nonempty output, not correspondence to the source.

**Gate:** freeze inputs, require complete domains and token counts, reject missing or nonfinite values, retain per-item predictions and finish reasons, compare a common clean set across arms, and enforce predeclared matched-baseline thresholds.

### F12: Inputs and releases are only partially pinned

**Confirmed.** The base image uses a mutable tag, apt and the complete Python resolution are not locked, and model/dataset downloads do not consistently specify immutable revisions. The weekly Docker workflow intentionally rebuilds dependencies without a commit at [.github/workflows/docker-image.yml lines 13-27](../.github/workflows/docker-image.yml#L13-L27), yet emits the same commit-derived SHA tag at [lines 60-66](../.github/workflows/docker-image.yml#L60-L66). That tag can therefore identify different image contents across scheduled rebuilds.

Compose uses `latest` with `pull_policy: missing` at [docker-compose.yml lines 24-32](../docker-compose.yml#L24-L32), so an existing local `latest` is not guaranteed to become current. The README statement that `latest` is always the current stack is stronger than these mechanics support.

**Opportunity:** pin the base image by digest, lock dependencies and artifact revisions, publish image digests, use content-addressed model manifests, and distinguish immutable commit builds from moving security-refresh builds.

## Additional gaps and opportunities

### Configuration should have one resolver

Several settings have surprising behavior:

- Entrypoint positional arguments are forwarded but omitted from final launcher command arrays.
- Unknown `CTX` and `KV` values fall into real profiles instead of failing.
- `LOOKUP_ADAPTIVE=0` suppresses a launcher safeguard without setting the engine variable the runtime reads.
- Empty `INT8_ACT` may leave an inherited engine variable active.
- `BASE_MODEL_DIR` affects preparation while launchers can independently select another model.

Relevant sources include [docker/entrypoint.sh](../docker/entrypoint.sh#L17-L24), [single-user/start_qwen.sh](../single-user/start_qwen.sh#L155-L190), [single-user/start_qwen.sh lines 264-270](../single-user/start_qwen.sh#L264-L270), and [batch/start_qwen.sh](../batch/start_qwen.sh#L59-L79).

Build one resolver that validates enum values and incompatible combinations, defines precedence once, emits a redacted effective configuration, and supplies the final argument array.

### Metric parsing needs a shared implementation

[run_benchmarks.sh](run_benchmarks.sh#L29-L42) assumes positional Prometheus series and does not account for multiple engines, missing metrics, counter resets, unrelated traffic, or labels. Several Python tools overwrite series rather than summing them, including [labd_bench.py](labd_bench.py#L52-L60). `real_rep.sh` also has a shell expansion bug: `/tmp/rr_$TAG_$i.log` expands `$TAG_`, so different tags overwrite the same paths.

Use one label-aware parser that distinguishes missing from zero, validates monotonic deltas and bounds, aggregates intended series, and records the scrape windows. Speculative counters require an exclusive server or request-scoped instrumentation.

### Cache-state protocols are under-specified

[run_benchmarks.sh](run_benchmarks.sh#L11-L12) instructs the operator to run twice and retain the second pass, but its deterministic seed resets between invocations at [lines 45-53](run_benchmarks.sh#L45-L53). The second invocation can repeat prefixes from the first. Distinct seeds within one invocation do not solve cross-invocation reuse.

Separate engine warmup from prefix-cache state. Cold lanes need unique, recorded prefixes or an isolated cache namespace. Warm lanes need explicit priming and observed hit counters.

### LABD needs a corpus contract

If the historical `~/qwen-serving` path is absent, [labd_bench.py](labd_bench.py#L63-L77) and [labd_accept.py](labd_accept.py#L208-L216) extend an empty source with newlines until reaching 200,000 characters. This terminates, but silently creates a whitespace workload. Context size is estimated in characters, not tokens.

Streaming logic in [labd_bench.py](labd_bench.py#L139-L157) can use chunk count as token count when usage is missing, produce a negative rate for empty output, mix token numerators, and discard generated text.

Require a nonempty content corpus with source and hash, tokenize before submission, require usage and valid stream termination, use monotonic timestamps and consistent numerators, and retain output text or token IDs.

### Demo claims need captured identity

[demo_capture.py](demo_capture.py#L9-L15) documents `--no-spec-baseline`, but the launcher does not parse or forward it and defaults to MTP. [demo_render.py](demo_render.py#L328-L329) then hardcodes `Stock vLLM` and `no speculative decoding`. Missing source-lane recordings prevent proving how the existing video was captured.

Use explicit `SPEC=off`, a matched checkpoint and image, a captured effective configuration, exact token counts, and lane identity validation. A patched no-spec server is not automatically stock vLLM.

### Microbenchmarks need reached-path evidence

[conc_ladder.py](conc_ladder.py#L182-L210) infers overlap from polling observations. An offline fixture with two short, non-overlapping streams was misclassified as a full-overlap window. It also keeps only the best repetition at [lines 243-255](conc_ladder.py#L243-L255).

[spec_attn_ctx_scan.py](spec_attn_ctx_scan.py#L65-L81) labels an arm forced 3D without proving dispatch, allocates scratch in the timed function, and does not check output correctness. Kernel tests do not comprehensively cover int8 padded strides, captured execution, ragged multi-request isolation, or current runtime fingerprints.

Every microbenchmark should assert which path executed, compare an independent reference, retain all repetitions, preallocate timed resources, and include a negative control.

### Training artifacts need completion and row identity

[drafter/capture.py](../drafter/capture.py#L47-L52) publishes its sequence manifest before capture completes and only prints incomplete counts at [lines 109-115](../drafter/capture.py#L109-L115). [train_mtp.py](../drafter/train_mtp.py#L238-L242) can reserve all eligible sequences for validation and leave no training microbatch. Accumulated mean losses are not normalized by contributing token count at [lines 463-471](../drafter/train_mtp.py#L463-L471).

Training accepts custom draft IDs, but the saved head does not bundle them; [export_mtp.py](../drafter/export_mtp.py#L104-L130) can later attach IDs copied from another model.

Capture into a staging generation, validate completeness and finiteness, atomically publish, preflight split sizes, normalize by actual tokens, and bundle vocabulary row IDs plus tokenizer/model fingerprints with every trained head.

### CI should test policy, not only installation

The patch-integrity workflow applies the complete patch series to pinned vLLM at [.github/workflows/patch-integrity.yml](../.github/workflows/patch-integrity.yml#L19-L28), which is valuable. The image build runs install verification but has no GPU, while benchmark helpers and result validation have no CPU fixture suite.

Add fast PR checks for shell syntax, Python compilation, patch format, parser fixtures, API failure behavior, stream truncation, missing metrics, overlap timing, result schemas, and negative controls. Keep GPU qualification as a separate release gate tied to exact artifacts.

### Local artifact hygiene is inconsistent

The untracked `docker-compose.override.yml` says it matches a gitignored pattern, but [.gitignore](../.gitignore) contains no such pattern. The override also requires `GPU_UUID`, causing implicit Compose loading to fail on machines without that local value.

Decide whether the override is a tracked portable example or a truly ignored host-local file, then make its comment and behavior match that decision.

## Strengths to preserve

- The repository preserves null results, retractions, and detailed failure histories in `docs/gotchas.md` and optimization notes.
- Runtime patches often gate unsupported geometry and retain reference fallbacks instead of replacing behavior universally.
- KVarN explicitly distinguishes committed tokens from speculative writes before irreversible packing.
- Memory accounting is treated as a first-class constraint, including profiled scratch and recurrent-state capacity.
- [mq3d_capacity_property.py](mq3d_capacity_property.py) has meaningful mutation-based negative controls rather than a vacuous green property.
- The MQ3D operator oracle imports production dispatch, poisons scratch, compares an independent reference, and checks breach state.
- Teacher-forced acceptance works with exact token IDs and documents selection bias and chunk-boundary limitations.
- The residue work tests all 128 nominal residues, and the verbatim classifier covers multiple corruption shapes.
- The project distinguishes end-to-end throughput, prefill, decode, acceptance, and context capacity more carefully than most inference repositories, even though some remaining labels need correction.

## Recommended work packages

### WP1: Restore trustworthy green and red states

Scope: F01, F02, patch validation, benchmark result validation.

Acceptance criteria:

- The current patch series parses and applies strictly to a disposable vLLM 0.28.0 tree.
- Every executable check has `PASS/FAIL/SKIP/INVALID` semantics and a matching exit code.
- Offline fixtures prove HTTP failure, stream truncation, missing usage, empty output, multi-series metrics, counter reset, and malformed result files all go red.
- Existing arithmetic property tests and their negative controls remain green.

### WP2: Make artifacts attributable and safe

Scope: F04, F05, F10, F12.

Acceptance criteria:

- Preparation runs under a lock and publishes only validated complete generations.
- Variant construction leaves source hashes and inodes unchanged for writable artifacts.
- Every benchmark writes an immutable manifest and unique output directory.
- A/B tooling proves ownership of the server and GPU and only tears down owned processes.
- Release images and model inputs resolve to immutable digests or revisions.

### WP3: Rebaseline the advertised workloads

Scope: F03, F08, F09, F11.

Acceptance criteria:

- Workload protocol v2 freezes rendered token IDs, thinking mode, sampling, data splits, output policy, and cache lane.
- Calibration and evaluation manifests are disjoint by sample hash.
- Aggregate throughput uses output tokens over wall time; latency proxies are labeled as proxies.
- Quality retains per-item evidence and requires complete finite coverage.
- Historical rows remain labeled with their original protocol and are not silently replaced.

### WP4: Qualify optional and high-risk paths

Scope: F06, F07, configuration resolution, KVarN/DFlash2 hypotheses.

Acceptance criteria:

- Compose's default single profile validates without a key, while public unauthenticated exposure requires explicit consent.
- KVarN materialized decode has a proven capacity guard and boundary tests.
- Every launcher prints one validated effective configuration and rejects unknown values.
- GPU qualification covers eager and captured execution, prefix-cache hit and miss, cancellation, staggered completion, int8/fp8/KVarN paths, and multiple resident requests.

## Hypotheses requiring GPU or runtime investigation

These are not confirmed defects and should not be presented as current failures:

- DFlash2 agreement-dependent lookup fusion may need a rejection-sampling oracle when custom agreement thresholds make proposal selection sample-dependent.
- Adaptive and chain host decisions may rely on asynchronous copies without an explicit completion event.
- KVarN side pools and external KV offload may disagree about which representation is authoritative before packed-tile flush.
- Global KVarN registries and shared scratch may alias under dual-batch overlap, multiple engines, or repeated in-process lifecycle.
- Long-block DFlash2 with lookup disabled may misalign grouped-convolution boundaries.
- Split-KV attention may admit dtype combinations beyond the bf16 assumptions in its Triton implementation.

Each hypothesis should enter the loop with a minimal oracle and a negative control before any performance experiment.

## Evidence and limits

Safe checks performed during this investigation:

- Inspected root configuration, CI, Docker, launchers, preparation, drafter, KVarN, patches, benchmark scripts, retained logs, quality JSON, and MQ3D verdict files.
- Reproduced the dirty patch parse failure with `git apply --numstat`.
- Reproduced the default Compose interpolation failure with `.env.example` and the `single` profile.
- Ran shell syntax checks over the benchmark drivers, verifier, launchers, and patch-series helper; they parsed successfully.
- Ran `verbatim.py`; all nine classifier fixtures passed.
- Ran the MQ3D capacity property: 978,978 vectors, 238,276 eligible, zero violations.
- Ran both MQ3D mutation controls; they detected 129,498 and 140,008 expected violations.
- Used offline mocks for API failure exits, needle misses, short-stream overlap, filename expansion, and selected metric arithmetic.
- Read both committed MQ3D verdict files; each contains ten passing rows for its recorded eager, block-16, capacity-32 scope.

Limits:

- No model was loaded, no service was started, and no live GPU benchmark was run.
- The local venv is vLLM 0.27.1 and was not treated as proof of target vLLM 0.28.0 GPU behavior.
- Existing quality and performance artifacts were interpreted only within the workload and provenance they retain.
- Raw IFBench, LABD target/result recordings, demo source lanes, and full residue-sweep outputs were not present for independent review.
- The working tree was already dirty. Existing edits were treated as current source and were not reverted or modified.
- No secret file was read; `.env` contents were not inspected.
