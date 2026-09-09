# Self-Improvement Loop

Use this loop to improve this project without mistaking faster output, a green log line, or a one-off GPU run for a safe release. It is written for both a human maintainer and a coding agent.

The current investigation and starting backlog are in [project-investigation.md](project-investigation.md).

## Objective

Optimize the serving stack under fixed constraints:

1. Preserve correctness for the declared supported workload and configuration.
2. Stay within quality, latency, memory, reliability, and exposure budgets.
3. Improve one predeclared user-visible metric by more than measured noise.
4. Leave reproducible evidence that another machine or future commit can interpret.

An optimization is promoted only when all four are true. A faster invalid run is a failed experiment, not a partial win.

## Loop state

Every experiment moves through this state machine:

`IDEA -> DEFINED -> ORACLED -> BASELINED -> MEASURED -> CHALLENGED -> PROMOTED`

Any failed gate moves it to `REJECTED` or back to `DEFINED`. An infrastructure or evidence failure moves it to `INVALID`, never to `REJECTED`: invalid evidence says nothing about the hypothesis.

Allowed terminal states:

| State | Meaning |
|---|---|
| `PROMOTED` | All gates passed and the change is now the supported default or documented option |
| `REJECTED` | Valid evidence disproved the hypothesis or violated a declared constraint |
| `INVALID` | The experiment could not answer the question because its identity, harness, or execution was defective |
| `PARKED` | Evidence is sound but the gain is too small or the implementation cost is currently unjustified |

## Operating rules

- Change one causal variable per comparison. If two changes are inseparable, name the bundle as the variable.
- Write the hypothesis, workload, constraints, and decision rule before implementing or measuring.
- Build the cheapest oracle before the optimization. Make the oracle go red with a mutation or known-bad fixture.
- Treat exact configuration, model, data, prompt, and cache state as experiment inputs.
- Separate correctness, quality, performance, memory, and reliability evidence. One metric cannot stand in for another.
- Keep every valid repeat. Never select only the best run.
- Compare paired boots or randomized arms, not unrelated historical bests.
- Record negative results and invalid runs. They prevent repeated dead ends.
- Promote a protocol version, not mutable prose. Historical results keep their original protocol identity.
- Run destructive patch checks only in disposable worktrees. Teardown only processes created by the current run.

## Step 0: Choose the next constraint

Read the latest evidence and choose one bottleneck, not one interesting knob.

Selection order:

1. A correctness, security, data-integrity, or false-success defect.
2. A missing gate that prevents trusting later work.
3. A user-visible reliability or compatibility failure.
4. A measured latency, throughput, memory, or context-capacity bottleneck.
5. A speculative optimization with a plausible model and cheap oracle.

Score candidate work from 0 to 3 on each dimension:

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| User impact | None | Niche | Important mode | Default path or data loss |
| Evidence | Guess | Anecdote | Repeated symptom | Reproduction or source proof |
| Risk reduced | None | Local | Cross-mode | Correctness/security/publication |
| Learning value | None | Narrow | Reusable insight | Removes a major unknown |
| Effort | Very large | Large | Medium | Small |

Use `impact + evidence + risk + learning + effort`. Start with the highest score unless a P0 blocks all other evidence.

**Completion criterion:** one constraint is selected, its evidence is linked, and all other candidates remain explicitly out of scope.

## Step 1: Define the experiment

Create a short experiment note before editing code:

```markdown
# EXP-YYYYMMDD-short-name

State: DEFINED
Owner:
Related finding or issue:

## Hypothesis
If <one change>, then <primary metric> improves by <minimum useful amount>
for <workload>, because <mechanism>, while all constraints remain within budget.

## Arms
Control:
Treatment:
Only intended difference:

## Workload identity
Protocol version:
Endpoint and template mode:
Model, drafter, tokenizer:
Prompt/data manifest:
Input/output lengths:
Sampling and thinking mode:
Concurrency and arrival pattern:
Cache lane:

## Gates
Correctness:
Quality:
Primary performance metric:
Latency:
Memory and residency:
Reliability:

## Decision rule
Promote when:
Reject when:
Invalidate when:
```

The mechanism matters. It predicts where the gain should appear and where a regression might hide. For example, an attention optimization predicts a context-length-dependent gain; a scheduler change predicts queueing or residency changes; a drafter change predicts acceptance and verify-step changes.

**Completion criterion:** another engineer can identify the single intended arm difference and compute the decision without asking what the experiment meant.

## Step 2: Freeze identity and provenance

Allocate a unique run root such as:

```text
bench/results/EXP-YYYYMMDD-short-name/
  experiment.md
  control/
    manifest.json
  treatment/
    manifest.json
  decision.md
```

Each arm manifest must record:

- Repository commit and dirty-diff hash.
- Client vLLM version and server vLLM version separately.
- Container image digest or environment lock hash.
- Ordered patch file hashes and resulting installed-tree fingerprint.
- Model, drafter, tokenizer, chat-template, draft-vocabulary, and quantization hashes.
- Dataset revision, split, sample hashes, and rendered prompt token-ID hashes.
- Redacted effective launch arguments and environment variables.
- GPU UUID and name, driver, CUDA, torch, Triton, power cap, clock policy, and platform.
- Cache policy, cache namespace or prefix salt, server boot identity, and warmup lineage.
- Wall-clock start/end, monotonic measurement bounds, and tool versions.
- Expected artifact list and checksums after completion.

Never store secrets. Record only whether authentication was enabled and which credential source won precedence.

**Completion criterion:** arm identity differs only in the intended variable, or every additional difference is declared as a confound before measurement.

## Step 3: Build the oracle first

Choose the smallest oracle that can disprove correctness.

| Change type | Required oracle examples |
|---|---|
| Arithmetic or capacity | Exhaustive/property test over accepted configuration space |
| Kernel | Independent numerical reference, reached-path counter, boundary and ragged shapes |
| Cache or capture | Hit/miss lanes, all relevant residues, eager/captured comparison, source-based output check |
| Scheduler or concurrency | Recorded stream intervals, residency, preemption, cancellation, staggered finish |
| Quantization | Frozen-domain PPL, task outputs, finite checks, calibration/evaluation disjointness |
| Drafter/speculation | Target-distribution oracle, exact token IDs, acceptance counters, no-spec reference |
| Parser/harness | Fixtures for failure, missing values, resets, multi-series labels, malformed streams |
| Preparation | Crash injection, source hash preservation, idempotence, atomic publication |
| Authentication/config | Precedence table, local/public bind cases, unknown-value rejection |

Add a negative control:

- Mutate the corrected formula back to the defect.
- Inject a wrong token, NaN, truncated stream, reset counter, or missing usage field.
- Force the reference path while claiming the optimized path.
- Remove one output artifact before publication.
- Swap one calibration sample into the evaluation set.

The oracle must fail for the negative control. A test that has never gone red has not proved it can detect the defect.

**Completion criterion:** the valid implementation is green and at least one representative known-bad mutation is red with the expected diagnosis.

## Step 4: Validate the harness offline

Before using GPU time, exercise the result pipeline with fixtures.

Required fixture classes:

- HTTP connection failure, timeout, 4xx, and 5xx.
- SSE error event, truncated stream, missing terminal event, and multi-token chunks.
- Missing usage, empty output, invalid finish reason, and fewer responses than requested.
- Missing metric, zero metric, duplicate labels, multiple engines, counter reset, and unrelated traffic.
- Nonfinite duration, tokens, PPL, acceptance, or memory values.
- Unsupported context length and insufficient resident overlap.
- Filename collision and repeated tag.
- Missing corpus, empty target, zero clean chunks, and incomplete capture manifest.

Result contract:

```json
{
  "status": "PASS|FAIL|SKIP|INVALID",
  "reason": "machine-readable-code",
  "expected_records": 8,
  "observed_records": 8,
  "measurements": {},
  "artifacts": []
}
```

`FAIL` means a tested product property failed. `INVALID` means the run cannot test the property. Both exit nonzero. `SKIP` is permitted only for a declared unsupported cell and must not produce a zero-valued measurement.

**Completion criterion:** every bad fixture reaches the intended non-success state and every valid fixture produces a complete schema.

## Step 5: Establish safe ownership

Before booting a service:

1. Acquire a GPU lock containing run ID, PID, process start time, and GPU UUID.
2. Refuse an occupied GPU unless reuse was explicitly requested.
3. If reusing a server, verify authenticated completion plus exact effective configuration and boot identity.
4. Use bounded health, metrics, and completion probes.
5. Write owned process IDs under the run directory, not a shared PID file.
6. Install cleanup traps that stop only those owned process identities.
7. Keep patch application and model preparation in disposable or generation-scoped directories.

Do not use a different TCP port as proof of GPU isolation.

**Completion criterion:** the run can prove which server and GPU it owns, and cleanup cannot match an unrelated process or repository.

## Step 6: Separate warmup from cache state

Declare one cache lane per measurement:

| Lane | Required setup | Required evidence |
|---|---|---|
| Cold engine | New process, no prior kernels or graphs | Boot ID and compile events |
| Warm engine, cold prefix | Shapes prewarmed; unique recorded prefix salt | Zero cached-input tokens for measured requests |
| Warm shared prefix | Prefix explicitly primed once | Expected cached-token count and shared hash |
| Warm conversation | Prior turn retained by protocol | Exact turn lineage and cache-hit evidence |
| Eviction/reuse | Controlled competing prefixes | Occupancy, eviction, and recompute counters |

Warm all shapes that will be timed. A generic short warmup does not warm a captured long-context or different-batch shape. Never solve JIT variance by keeping an unidentifiable second run.

**Completion criterion:** measured requests have the declared compile and cache state, confirmed by counters or trace evidence rather than seed intent alone.

## Step 7: Run correctness and quality gates

Run these before performance measurement so speed cannot bias interpretation.

### Correctness gate

- All requests accounted for.
- No engine death, OOM, stream error, unexpected fallback, scratch breach, or structural corruption.
- Optimized path execution is positively observed.
- Independent oracle matches within the predeclared tolerance.
- Relevant residues, boundaries, ragged batches, cache lanes, and capture modes pass.
- Output identity or source-based semantics pass for tasks where exact identity is expected.

### Quality gate

- Evaluation inputs are frozen and disjoint from calibration/training.
- Every required domain is present with expected item and token counts.
- Logprobs and aggregate values are finite; no missing token is silently skipped.
- Per-item predictions, token counts, truncation, and finish reasons are retained.
- Treatment is compared to a matched control using paired items.
- Lossy profiles use an explicit quality budget; exact speculative changes use the no-spec distributional oracle.

Initial policy while better empirical budgets are learned:

- Investigate any repeatable per-domain PPL regression above 1%.
- Do not call a one-question GSM8K difference at n=200 an improvement.
- Require zero known structural corruption in supported modes.
- Require zero unexpected degraded-state or scratch-breach events.

**Completion criterion:** every declared correctness and quality constraint passes with complete retained evidence. Otherwise stop before timing.

## Step 8: Measure performance in paired runs

Use ABBA or randomized paired boot order:

```text
control, treatment, treatment, control
```

Repeat at the boot level, not only within one long-lived server. Keep all valid repetitions.

Report distinct properties separately:

| Property | Measurement |
|---|---|
| End-to-end aggregate throughput | Actual completed output tokens / cohort wall time |
| Per-request decode latency | TPOT distribution for valid output tokens |
| Time to first token | TTFT distribution including queue and prefill |
| Prefill throughput | Uncached input tokens / validated prefill interval |
| Steady-state decode | Tokens during an independently verified simultaneous-decode window |
| Acceptance | Accepted draft tokens / draft attempts, using label-aware deltas |
| Residency | Actual simultaneous resident requests and overlap duration |
| Capacity | Supported context and concurrency before preemption/OOM |
| Memory | Peak allocated/reserved/device memory under the measured shape |

Rules:

- Label `C / TPOT` as a cohort latency score if retained; it is not measured aggregate throughput.
- Reject a steady-state row if overlap is too short or inferred only from scrape timing.
- Save raw client JSON, server logs, metrics snapshots, output text/token IDs, and finish reasons.
- Report median and tails where user latency matters; report all repeats and confidence intervals for comparisons.
- Measure A/A variability before deciding a useful A/B margin.

**Completion criterion:** the primary metric and every constraint metric are computed from valid raw records, and the claimed gain exceeds both the predeclared useful margin and measured A/A noise.

## Step 9: Challenge the result

Assume the apparent win is wrong and try to explain it away.

Challenge checklist:

- Did thinking mode, tokenizer, chat template, output length, or EOS behavior change?
- Did one arm receive prefix-cache hits, compile less, or reuse an autotune artifact?
- Did requests actually overlap and remain resident?
- Did failed or short responses reduce denominator work?
- Did acceptance counters include unrelated traffic, reset, or omit an engine?
- Did the treatment change prompt tokens or generation distribution?
- Is the gain only a best-run selection or boot-order effect?
- Is aggregate throughput flat while a latency-derived proxy rises?
- Did queueing, TTFT tails, memory, preemptions, or context capacity regress?
- Did the optimized path execute, or did a fallback produce the timing?
- Does the effect follow the mechanism across context, batch, and output-length sweeps?

Run one adversarial cell aimed at the most plausible alternative explanation. Examples include tied logits for top-k, over-capacity scratch, all 128 cache residues, staggered cancellation, a counter reset, or a cold-prefix salt reused across boots.

**Completion criterion:** the strongest alternative explanation is tested and does not account for the result, or the experiment returns to `DEFINED` with a narrower claim.

## Step 10: Decide

Apply the rule written in Step 1 without moving the threshold after seeing results.

### Promote

Set `PROMOTED` only when:

- Harness and provenance are complete.
- Correctness and quality gates pass.
- The primary gain exceeds useful margin and A/A noise.
- Latency, memory, reliability, context, and exposure constraints pass.
- The relevant supported matrix is tested.
- A rollback or safe fallback exists for risky runtime changes.

### Reject

Set `REJECTED` when valid evidence shows no useful gain, a mechanism is wrong, or a constraint fails.

### Invalidate

Set `INVALID` when identity differs unexpectedly, evidence is incomplete, a tool silently skips data, the server configuration is unknown, requests fail, or cache/warmup state is not established.

### Park

Set `PARKED` when evidence is valid but the gain is smaller than its complexity, maintenance, or compatibility cost.

Write `decision.md`:

```markdown
# Decision

State: PROMOTED|REJECTED|INVALID|PARKED
Protocol:
Control manifest:
Treatment manifest:

## Result
Primary effect and uncertainty:
Constraint outcomes:

## Mechanism
What the evidence supports:
What remains unknown:

## Decision
Why this state follows from the predeclared rule:

## Follow-up
One next question, if any:
```

**Completion criterion:** the state follows mechanically from the predeclared rule and links to complete immutable evidence.

## Step 11: Integrate and prevent regression

For a promoted change:

1. Keep the smallest implementation that explains the gain.
2. Add the oracle and negative control to the fastest appropriate test layer.
3. Add configuration validation and safe fallback behavior.
4. Update claims with protocol ID, hardware, workload, and evidence path.
5. Mark old results historical rather than overwriting them.
6. Add a release note describing supported and untested boundaries.
7. Schedule the smallest GPU release check that catches the discovered failure class.

For a rejected change:

1. Preserve the hypothesis, evidence, and reason for rejection.
2. Remove abandoned code unless it remains behind an explicitly experimental switch with a concrete owner.
3. Add the learned constraint to `docs/gotchas.md` only when it is durable and reproducible.

**Completion criterion:** the lesson survives after the experiment branch and local result directory disappear.

## Step 12: Feed the next loop

After each decision, spend ten minutes on a loop retrospective:

```markdown
## Loop retrospective
What surprised us?
Which gate caught it?
Which gate should have caught it earlier?
What manual step can become a fixture or manifest field?
What evidence is still expensive or ambiguous?
What is the next highest constraint?
```

Prefer improvements to the loop that move detection earlier:

`production symptom -> GPU release test -> local GPU oracle -> CPU fixture -> static validation`

Earlier detection is usually more valuable than broader documentation after the fact.

**Completion criterion:** one concrete loop improvement is applied or the retrospective states why the current gates were sufficient.

## Required test matrix

Use the smallest applicable subset, but declare every omitted cell.

| Axis | Minimum cells |
|---|---|
| Speculation | off, MTP, DFlash2 where affected |
| Context profile | fast, long, huge where supported |
| Cache | disabled, cold miss, warm shared hit |
| Capture | eager/reference and shipped captured mode |
| Prompt | short, 4k, 16k, boundary/deep where affected |
| Concurrency | 1, first batched level, capacity edge |
| Finish behavior | aligned, staggered, cancellation |
| Content | prose, code, copy/quote, adversarial structure |
| Sampling | greedy and shipped default when sampling semantics can change |
| Platform | reference Linux RTX 3090; WSL2 only for platform-sensitive paths |

Do not run the full Cartesian product blindly. Use the mechanism to select interactions, then retain one broad release smoke matrix.

## Promotion dashboard

Every candidate should fit one row:

| Experiment | State | Correctness | Quality | Primary gain | TTFT/TPOT | Memory/capacity | Reproducible | Decision |
|---|---|---|---|---|---|---|---|---|
| EXP-ID | DEFINED | pending | pending | pending | pending | pending | manifest link | pending |

No green cell may be inferred from another column. In particular, good output throughput does not imply acceptable TTFT, good PPL does not imply long-context retrieval, and high acceptance does not imply target-distribution correctness.

## Automation backlog

Implement these in order because each makes later optimization evidence cheaper and safer:

1. Shared `PASS/FAIL/SKIP/INVALID` result schema and exit behavior.
2. Offline fixtures for API, stream, metric, overlap, and missing-artifact failure modes.
3. Immutable run directories and automatic provenance manifests.
4. Label-aware Prometheus parser with reset and multi-engine handling.
5. Transactional model preparation with locks and completion manifests.
6. One validated launcher configuration resolver and redacted effective-config output.
7. Paired-run driver with randomized order, cache-lane control, and all-repeat retention.
8. Per-item quality records and calibration/evaluation disjointness checks.
9. GPU release matrix tied to exact image and model digests.

## First three loops for this repository

### Loop A: Trustworthy failure semantics

Hypothesis: a shared result contract and fixture suite will turn all known silent failures into deterministic nonzero exits without changing valid measurements.

Primary evidence: F02 in [project-investigation.md](project-investigation.md#f02-failure-does-not-consistently-mean-a-failing-process).

Useful result: all mocked HTTP failures, needle misses, numerical mismatches, broken residues, empty outputs, and missing records produce the intended failure state; valid fixtures remain green.

### Loop B: Workload protocol v2

Hypothesis: freezing rendered token IDs, thinking mode, data identity, cache lane, and actual throughput math will remove the current ambiguity while preserving reproducible performance rankings.

Primary evidence: F03, F05, F08, F09, and F11.

Useful result: repeated v2 A/A runs produce a measured noise floor; every result binds client/server/model/data identities; no v1 result is relabeled.

### Loop C: Atomic preparation

Hypothesis: generation-scoped preparation under a lock will survive injected interruption and leave source and last-known-good model generations unchanged.

Primary evidence: F04.

Useful result: crash injection at every publication boundary leaves either the old complete generation or the new complete generation selectable, never a mixed model; writable source artifact hashes and inodes remain unchanged.

## Compact daily checklist

Before editing:

- One constraint selected.
- Hypothesis and mechanism written.
- Control, treatment, protocol, and thresholds fixed.

Before GPU measurement:

- Oracle green and negative control red.
- Offline harness fixtures green.
- Manifest complete and arms differ only as declared.
- GPU and server ownership proven.
- Warmup and cache lane observed.

Before claiming a win:

- Every request and token accounted for.
- Correctness and quality gates pass.
- All valid repetitions retained.
- Gain exceeds useful margin and A/A noise.
- TTFT, TPOT, memory, capacity, and reliability remain in budget.
- Strongest alternative explanation challenged.

Before merging or publishing:

- Decision follows the predeclared rule.
- Regression oracle and negative control retained.
- Exact evidence and protocol linked.
- Historical claims remain historically labeled.
- Supported and untested boundaries are explicit.
