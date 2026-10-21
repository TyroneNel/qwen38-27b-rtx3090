# Speculative-decoding runtime

Lane: speculative decoding (Opus, high effort), verified by the lead. Part of the [DeepInfer expedition](README.md).

Citation forms:
- HyperQwen paths are repo-relative. `PLAN:` is `docs/decode-perf-150k-240k-plan.md` on branch `docs/decode-perf-150k-240k-plan`.
- `DeepInfer/...` refers to deepseek-ai/DeepInfer at `774ce24`.
- `vllm@v0.31.0:...` refers to the pinned vLLM tag; `vllm@main:` to `4bd75e952a`.
- Every gain here is an estimate from the lane's simulation or arithmetic. No benchmark number is new; measured figures cite the HyperQwen doc they come from.

Modes: A batch; B DFlash2 chat (T=8); C quoting (`DFLASH_TOKENS=15`); D `SPEC=mtp CTX=long`, k=3; E `SPEC=dflash2 CTX=huge`, KVarN, k=7. "C4" is DFlash2 at four concurrent streams.

## Verdict

1. **DeepInfer's confidence-scheduled verify (MTP_VARLEN) is already in the pin, and HyperQwen already tried it.** It ships as vLLM 0.31 `enable_adaptive_verification` (#52228, `vllm@v0.31.0:vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`). It refused to boot on GDN (PLAN:1139-1215, commit 585c632), and the plan scored it "~0 as shipped; +5–15% if patched", effort M (PLAN:662).
2. **The unblock is small.** Kimi-K3's KDA backend subclasses `GDNAttentionBackend` and opts in for pure spec-decode batches in about 17 lines (#52988). Qwen's GDN pure-spec branch already reads its offsets from the device, which is the property that opt-in relies on (spec-1).
3. **At one stream, adaptive verification as vLLM and DeepInfer build it is worth about zero.** Both choose the total budget from the *previous* step's confidences (`adaptive_verification.py:302-305`; `DeepInfer/csrc/engine/mtp_varlen.hpp:150-267`). At batch 1 the budget is the step's length, so that's a stale per-step k: B −1.3 to +0.7%, D −2.3 to +0.1% in the simulation. The wins are at a few streams (C4 +6–7% over the shipped `DYNAMIC_K`), and at long context only if the trim uses *fresh* confidences on the device (spec-2).
4. **The biggest structural lever is RecoverSSM, not anything DeepInfer ships.** DeepInfer never rolls back state: its SWA and compressor state is a ring indexed by logical position (`DeepInfer/deepinfer/model_executor/layers/tilelang_ops/state_cache.py:12-83`), and rejected slots are overwritten before anyone reads them (`models/mtp/dflash.py:198-228`). GDN state isn't position-indexed, so that can't port. vLLM's RecoverSSM (#51855, for KDA) is the GDN-shaped equivalent: verify from one checkpoint plus a small activation cache, then recompute the accepted state. On GDN it would cut DFlash2's 0.88 GiB per resident to about 0.2 GiB (spec-3).

## Candidates (ranked)

| ID | Title | Gain (estimate) | Effort | Confidence |
|---|---|---|---|---|
| spec-0 | Context-aware fixed verify width at E (`DFLASH_TOKENS` 3 or 5 at 240k) | E −2.5 to +12.5%; frees 0.2–0.4 GiB | S | Medium-low |
| spec-1 | Opt GDN into vLLM's adaptive verification (mirror #52988) | C4 +6–7% over `DYNAMIC_K`; B and D ~0; E +0.7 to +15% over k=7 | S patch, M validation | Medium |
| spec-2 | Fresh-confidence device trim inside a fixed-width graph | B +1 to +4%; D ~0 to +1.6%; E +4 to +10% over the best fixed k | M | Low-medium |
| spec-3 | RecoverSSM for GDN: one checkpoint plus an activation ring instead of 1+k state slots | DFlash2 residents ~6 → ~20+; CTX=long reach ~114k → ~132k; −0.45 ms/step | L | Medium (memory), low (step) |
| spec-4 | Lookup-aware confidences in the verify budget | C1 0 to +3%; C2–C4 positive, unestimated | M | Low |
| spec-5 | Expected-acceptance telemetry Σ min(p, q) | 0 tok/s; lower-variance A/Bs | S | High (as a metric) |

**Lead caveat on every gain in this file.** The simulation charges each verify position a constant c_tok (ms). Real costs come in stairs: the Marlin GEMMs are flat up to 16 query tokens (`docs/gotchas.md:253-262`), and pthfa's attention is flat from T=6 to T=10 because those widths share one padded 64-row tile, while T ≤ 5 runs an unpadded kernel ([attention-kv.md](attention-kv.md)). On pthfa, trimming from k=7 to k=5 saves no attention at all; k ≤ 4 does. The GDN state writes (about 0.08 ms per position, spec-3) are the only part that is truly linear. E runs KVarN's Triton kernel, whose row tiling hasn't been read. Measure the real cost per width (runtime-2 in [runtime.md](runtime.md)) before trusting any row below.

### spec-0: context-aware fixed verify width at 240k (mode E)

- **DeepInfer** picks the verify length from a cost curve profiled at boot plus acceptance (`mtp_varlen.hpp:150-267`; `DeepInfer/deepinfer/worker/model_runner.py:562-622`). The lesson: the right k depends on the per-position cost, and that cost grows with context.
- **HyperQwen** verifies 1+7 at every depth (`single-user/start_qwen.sh:382`, `DRAFT_TOKENS=${DFLASH_TOKENS:-7}`). A k that changes step to step is refused at `CTX=huge` with prefix caching, because a moving verify length corrupts a KVarN prefix-cache hit (`start_qwen.sh:399-428`, `:680-682`). A *fixed* lower k isn't refused.
- **Cost per position (inferred).** Target attention over 16 layers at 128k is 5.63 ms per step at T=8 on a 3080 TP=2 rank (`docs/long-context.md:534-536`). Scaled to one 3090 (twice the KV heads, 68/82 SMs): about 9.3 ms per step, 1.2 ms per position at 128k, about 2.2 ms at 240k if linear in context. KVarN uses a different kernel, so this is a bracket.
- **Simulation** (`cpu-checks/spec_varlen_sim.out`, E rows; the step at k=7 is calibrated to 67 tok/s and 2.4 tokens/step from the 2.0–3.1 measured at 112k, `docs/long-context.md:199-207`):

  | c_tok (ms/position) | k=7 | k=5 | k=3 |
  |---|---|---|---|
  | 1.0 | 76.5 tok/s | 78.2 (+2.2%) | 74.6 (−2.5%) |
  | 2.0 | 75.4 | 82.2 (+9.0%) | 84.8 (+12.5%) |

- **Pool.** Each resident holds 6 GDN groups × (2+k) state pages (`start_qwen.sh:572-573`), about 98 MB per k step. k=5 frees ~0.2 GiB, k=3 ~0.39 GiB.
- **Effort** S: a launcher default in the `SPEC=dflash2 CTX=huge` arm, plus an A/B. **Risk:** lossless (the rejection sampler is unchanged). Copy-heavy work loses its long-block upside, so `DFLASH_TOKENS=15` stays opt-in.
- **GPU validation.** First measure c_tok directly: ms per step at k=3 and k=7 at fixed context. Then `bench/real_rep.sh` at T=0 and PLAN's 240k ladder with `DFLASH_TOKENS=3/5/7`, three boots per arm, at 90k, 150k and 240k. Win: ≥+5% at 240k with no loss beyond 2% at 90k.

### spec-1: opt GDN into vLLM 0.31's adaptive verification

**Measured (2026-10-21):** INVALID. With the opt-in patch, greedy output departs from the adaptive-off reference at token 0–1 on all six labd tasks (0/23 clean chunks, against 18/23 with an identical off-arm control), and acceptance length falls to 1.75. No speed run was made. [gpu-results/spec-1.md](gpu-results/spec-1.md)

- **DeepInfer.** Sigmoid confidence → cumulative survival → global sort → argmax of (num_seqs + Σ survival) / (baseline(graph) + slope·graph + sampler) (`mtp_varlen.hpp:150-267`; `csrc/engine/scheduler.hpp:648-665, 1239-1258`). A TileLang kernel then recomputes per-request lengths on the GPU (`tilelang_ops/schedule_mtp_varlen.py:35-98`). DeepInfer's own table: −1.5% at 16 sequences, +4–7% around 60 (`DeepInfer/doc/inference_deployment_qa.md:249-256`).
- **vLLM pin.** The same policy: the budget comes from stale confidences, and fresh ones distribute it (`adaptive_verification.py:282-350`). Cost tables come from profiled draft and verify curves (lines 80-128). DFlash2 has no confidence head, so it uses the online logistic estimator (`spec_decode/acceptance_estimator.py:313`; `dflash2/speculator.py:246`).
- **Why the patch is S, not M.**
  - The refusal is `supports_device_cpu_query_lens_mismatch()`, which returns `not is_ssm()` (`vllm@v0.31.0:vllm/v1/attention/backend.py:215-228`). The varlen graph check needs `ALWAYS` or a bound (`vllm/v1/worker/gpu/attn_utils.py:334-364`), and GDN declares `UNIFORM_BATCH` (`gdn_attn.py:86`).
  - Qwen GDN's pure-spec branch already takes `spec_query_start_loc = query_start_loc[:n+1]` from the device (`gdn_attn.py:343-365`). Both consumers read device offsets with a fixed (k+1) state row: `causal_conv1d_update(..., max_query_len=k+1)` and `fused_sigmoid_gating_delta_rule_update(cu_seqlens=..., ssm_state_indices=..., num_accepted_tokens=...)` (`qwen_gdn_linear_attn.py:1335-1346, 1448-1463`).
  - HyperQwen already carries the conv-state bound for "a step that verified fewer tokens than the previous step accepted" (`patches/vllm-pr50021-gdn-spec-bounds.patch:1-6`).
  - KDA, a GDN subclass, opts in in #52988 (commit `88aa0d287d`): mismatch → True ("Pure spec-decode reads its plan from DEVICE offsets"), `_cudagraph_support = ALWAYS` ("one k+1 graph replays any 1..k+1 mix"), and a filter for the profiler's over-long dummy rows (`kda_metadata.py:305-311, 411-419, 760-768`). The mamba-hybrid half of #52988 is already in v0.31.0.
  - **Lead caveat.** KDA's metadata also carries `RecoverSSMMetadata`, so its opt-in was written for a backend whose speculative state may be RecoverSSM-shaped. Qwen GDN keeps one state slot per verified token. A trimmed row writes fewer slots and `num_accepted_tokens` indexes within them, which looks safe, but that's an argument, not a test. Validate it with the greedy bit-identity check below before any speed run.
- **Gain** (simulation; the cost model is calibrated so that at C4 a fixed k=3 beats k=7 by +7.9%, matching the measured `DYNAMIC_K` +7.7/+8.0/+9.2% at `start_qwen.sh:678-679`):
  - A: 0 (no speculation; residency blocks it, see spec-3).
  - B: −1.3 to +0.7%. Leave it off.
  - D: −2.3 to +0.1%. Leave it off.
  - E: +0.7 to +15% over k=7, but only −1.4 to +2.5% over the best fixed k, which spec-0 gets for free.
  - **C4, DFlash2 at 4k: +6.0 to +6.7% over `DYNAMIC_K`'s k=3, +15 to +26% over k=7.**
- **Risk.** Lossless, since trimming only removes drafts. The mixed decode-plus-spec branch still assumes CPU totals (#52988's comment), so gate to pure-spec batches as KDA does. **Don't enable at `CTX=huge`** until the KVarN verify-length corruption is root-caused. The FA2-fp8 target backend (D's default) must also accept the mismatch; pthfa says it does (`pthfa/python/pth_flash_attn/vllm_backend.py:38-42`).
- **Porting plan.** `patches/gdn-adaptive-verify-optin.patch`, about 30 lines in `vllm/v1/attention/backends/gdn_attn.py`:
  - `GDNAttentionBackend.supports_device_cpu_query_lens_mismatch()` returns True;
  - `_cudagraph_support = ALWAYS`, or a varlen max query length of `num_spec + 1`;
  - `spec_sequence_masks_cpu &= query_lens_cpu <= num_spec + 1` to drop the profiler's dummy rows.

  Launcher: an opt-in `ADAPTIVE_VERIFY=1` that adds `"enable_adaptive_verification":true` to `SPEC_CFG`, refused at `CTX=huge` with prefix caching on, like `DYNAMIC_K` (`start_qwen.sh:684-693`).
- **GPU validation.** Boot with no refusal. Greedy bit-identity against adaptive-off on `bench/labd_accept.py` (teacher-forced, so the trajectory is fixed), plus the residue sweep, GSM8K and a soak. Speed: `bench/conc_ladder.py` at C1, C2 and C4, three boots per arm, against `DYNAMIC_K`. Win: ≥+4% at C4 and within 1% at C1.
- **Same item as runtime-1** in [runtime.md](runtime.md); this lane's evidence moves its effort from M to S.

### spec-2: fresh-confidence device trim at batch 1

**Measured (2026-10-21):** skipped, because its gate (spec-1) failed. [gpu-results/spec-1.md](gpu-results/spec-1.md)

- **DeepInfer** recomputes per-request lengths on the GPU because async scheduling leaves the host one step stale (`schedule_mtp_varlen.py:35-98`). The *total* is still chosen on the host from older rates. vLLM splits it the same way.
- **HyperQwen.** The lookup long block reads pinned flags one step stale and needs `--no-async-scheduling` (`docs/optimizations.md:287-299`).
- **Mechanism.** Replay the k+1-wide graph every step, but let the drafter's tail kernel write a device `query_start_loc` with n = argmax over the fresh survival prefix of (1 + ΣS) / cost[n], using a cost table in device memory. pthfa and the GDN spec kernels read device lengths, so attention and per-token state writes shrink. Marlin runs padded, which is free below 16 rows. Trimmed positions count as rejected. No host sync.
- **Gain** (simulation, fresh minus stale, σ = 1 estimator noise): B +1.1 to +4.2% over k=7; D −0.2 to +1.1%; E +3.6 to +7.8% over the best fixed k; C4 +2%. It depends on step-to-step autocorrelation (ρ = 0.8 assumed) and estimator quality, neither measured. The stair caveat above applies with full force: on pthfa, a trim that stays in T=6–10 buys only GDN state writes.
- **Effort** M: a fixed-width `fresh_budget` path in `adaptive_verification.py` for num_reqs ≤ R (an env knob in `patches/speed-knobs-envs.patch`), a device cost table, and a small Triton argmax. **Risk:** lossless. Untested whether FA2-fp8 reads device lengths. The padded graph must not read garbage rows; pthfa computes its tile count on the device (`pthfa/csrc/pth_attn/launch.cu:376-380`).
- **GPU validation.** `bench/real_rep.sh` and PLAN's 150k/240k ladder at C1: stale vs fresh vs fixed k. Win: ≥+4% at 240k with ≤1% loss at 4k.
- **Overlaps** drafter-1 in [drafter.md](drafter.md), which reaches the same per-step trim from the drafter side.

### spec-3: RecoverSSM for GDN

**Measured (2026-10-21):** skipped, because its gate (spec-1 WIN) failed. [gpu-results/spec-1.md](gpu-results/spec-1.md)

- **vLLM pin.** `kda_recoverssm_verify` runs the window "without modifying its checkpoint" and fills an fp32 correction cache plus a key/gate cache of k+1 positions (`vllm@v0.31.0:vllm/models/kimi_k3/nvidia/ops/recoverssm.py:585-640`). `_commit_kda_state_kernel` replays the accepted prefix into the state, including align-mode boundary states (lines 399-420). `num_speculative_blocks = 0` when it's on (`layers/mamba/abstract.py:77-83`).
- **HyperQwen** uses vLLM's baseline: the GDN spec kernel stores the state after every token into slot i_t (`fused_sigmoid_gating.py`, `INPLACE_FINAL_STATE` inside the T loop). Each resident holds 6 groups × (2+k) pages, 0.88 GiB for DFlash2 at k=7 (`start_qwen.sh:566-573`; `docs/gotchas.md:325-336`; `docs/optimizations.md:349-352`). That's 15.8% of the `CTX=fast` pool per resident, so 6–7 fit (MTP: 8). Not tried; nothing in the repo mentions RecoverSSM.
- **Arithmetic (inferred).**
  - One GDN layer's SSM state is 48 × 128 × 128 × fp16 = 1.57 MB (`start_qwen.sh:567-568`). At k=7 the baseline writes 8 of them per step: 8 × 1.57 MB × 48 layers = 603 MB, about 0.64 ms at 936 GB/s.
  - RecoverSSM writes an activation ring instead: per layer, a correction of 48 × 8 × 128 × 4 B = 197 KB plus keys and a scalar gate (GDN's gate is per head, KDA's per channel) ≈ 33 KB, so about 11 MB per step over 48 layers. The commit reads and writes one state: 2 × 75 MB ≈ 0.16 ms.
  - Net about −0.45 ms per step per resident: +1.7% on a 25.9 ms C1 step, +3.7% on 49.1 ms at C4.
- **Gain.** Memory per DFlash2 resident 0.88 → ~0.2 GiB. `CTX=fast` residents with empty context 6–7 → ~20–25; B's pool +0.68 GiB ≈ +8.9k tokens. `CTX=long` DFlash2 +0.68 GiB ≈ +18k tokens, so reach ~114k → ~132k (`start_qwen.sh:226`). MTP k=3: 5 → 2 pages, ~0.29 GiB. **Mode A:** this is the prerequisite for any batch speculation; 1+k slots × 64 residents don't fit.
- **Effort** L: port the verify and commit Triton kernels to GDN's scalar gate (g = −exp(A_log)·softplus(a + dt_bias), β = sigmoid(b)); KDA-style `spec_state_slots` metadata (`kda_metadata.py:560-575`); an extended conv window; the mamba-hybrid commit hook; HyperQwen's align-mode patches; and the seat print at `start_qwen.sh:560-600`, whose (2+k) formula changes.
- **Risk.** Exact in real arithmetic with the same number of fp16 roundings, but not bitwise identical, so it needs the full quality battery. It interacts with prefix-cache align mode; KDA handles boundary states, so there is a precedent.
- **GPU validation.** A kernel test against an fp32 recurrent reference, as `vllm@v0.31.0:tests/models/kimi_k3/test_kda.py` does. `bench/conc_ladder.py` at `CTX=fast`: ≥15 residents with 128-token prompts and no preemption. C1 and C4 within noise or better. GSM8K within 1 point, perplexity +≤0.05%.
- **Cheap next step (not run).** A float64 CPU check that per-token GDN states equal checkpoint-plus-replay.

### spec-4: lookup-aware confidences in one verify budget

**Measured (2026-10-21):** skipped, because its gate (spec-1) failed. [gpu-results/spec-1.md](gpu-results/spec-1.md)

- **DeepInfer** ranks every draft position in one survival list (`scheduler.hpp:1239-1258`).
- **HyperQwen.** Confidences come from the drafter's candidate scores *before* `_apply_lookup` overwrites positions (`patches/dflash2-lookup-drafting.patch:1053-1062`), and tail positions 8–15 have none. The long block runs on a 2-step, batch-unanimous heuristic with sync scheduling (lookup patch:850-1000). Real-traffic acceptance at width 15: 0.79–0.96 for the drafter's 7 positions, a flat ~0.15 tail (`docs/gotchas.md:1159-1162`). At C ≥ 2 one non-copying request vetoes the long block for everyone.
- **Mechanism.** Survival for lookup positions from an online estimate per match-length bucket (the same logistic as `acceptance_estimator.py`, with match length as the feature), ranked with the drafter positions under spec-1's budget.
- **Gain.** C1 0 to +3% (the heuristic already takes +55% on copies, `docs/optimizations.md:301-303`). C2–C4 on mixed traffic: positive, not estimated. Needs spec-1. **Effort** M; lossless; the KVarN caveat applies.
- **GPU validation.** `labd_bench` six-task suite at C1 and C4 against the heuristic. Win: no task −2% at C1, ≥+5% on the mixed C4 suite.

### spec-5: expected-acceptance telemetry (for the evals lane)

**Measured (2026-10-21):** INFO, telemetry exact at T=0 (survival 3.486 = realised 3.486 tok/step), but the variance across prompts drops only 3–7% (ratio 1.03–1.07). [gpu-results/spec-5.md](gpu-results/spec-5.md)

DeepInfer logs accept_probs = Σ min(p, q) per position (`tilelang_ops/fused_sample.py:195-248`; `common.py:53`) and uses it as the label for confidence error (`sampler.py:440-560`). HyperQwen counts realised accepts, which are noisy: "The 16% DFlash2 acceptance regression … was invisible on one README prompt" (`docs/gotchas.md:1163-1170`). The Rao-Blackwellised count has lower variance, so 1–4% effects like spec-1 and spec-2 become measurable. Effort S: a counter in the rejection-sampler path exported as a Prometheus metric. See [evals.md](evals.md).

## Why DFlash2 loses 2:1 past 64k

From `docs/long-context.md:199-218` and `docs/optimizations.md:349-356`, three parts:
1. **Acceptance.** 2.17 against 2.68–2.75 tokens per step on prose at 112k, from the 2,048-token drafter window. DeepInfer doesn't fix this; its drafter also sees only the main model's SWA window (`models/mtp/dflash.py:139-144`). See [drafter.md](drafter.md).
2. **Step cost.** 62.7 against 37.7 ms per step at 112k (34.6 tok/s ÷ 2.17 vs 72.9 ÷ 2.75). Partly the backend (since improved: FA2-fp8 +27% at 100k, `docs/long-context.md:190`), partly 8 verify positions against MTP's 4, partly the 5-layer drafter. spec-0 and spec-2 address the verify part; attn-1 in [attention-kv.md](attention-kv.md) addresses the kernel.
3. **Residency.** 1+7 state slots cost 0.88 GiB, which is why DFlash2 reaches 114k against MTP's 150k. spec-3 addresses it.

## Other findings

- **Per-step host overhead is already small.** A native profile shows a 23.49 ms step against 23.09 ms of summed GPU components (`docs/long-context.md:534-537`, 3080 TP=2), so ≤0.4 ms (~2%) is host and launch gaps **(inferred)**. Lookup's forced sync scheduling costs <1% (`docs/optimizations.md:297-299`). DeepInfer's whole-drafter graph and fused input copy (`worker/dflash_cuda_graph_runner.py`) are worth ≤2%.
- **MTP k=4–5 at 150k.** The simulation puts k=4 at +4.1% and k=5 at +5.2% over k=3 at D (D′ rows, assuming ~0.7 ms per MTP draft pass). PLAN T3/B2 (:421) has k=4 stable on WSL2 with speed undecided. k=4 keeps T=5, inside pthfa's unpadded decode kernel; k=5 doesn't.
- **Lossless T>0 is already shipped** (`draft_sample_method=probabilistic`; lookup point-mass keeps it exact, lookup patch:1071-1076).
- **A "few users" mode** (C2–C8, DFlash2 or MTP, spec-1 plus spec-3) is the realistic batch-speculation target. Mode A can't keep speculation until residency is fixed.

## Already in HyperQwen or the pin

- `DYNAMIC_K` batch-size schedule `[[1,3,7],[4,512,3]]`: +7.7, +8.0 and +9.2% at 4 streams (`start_qwen.sh:678-693`).
- Two-level varlen by hand: the lookup adaptive long block (lookup patch:850-1000).
- Adaptive verification, tried and refused (PLAN:1139-1215, row B5 :424, :662).
- The GDN conv-offset bound for a shrinking verify length (`patches/vllm-pr50021-gdn-spec-bounds.patch`).
- pthfa reads device q_len (`vllm_backend.py:38-42`).
- The fused drafter context-KV projection (`vllm@v0.31.0:.../qwen3_dflash.py:505-545`), equivalent to DeepInfer's `main_kv_proj` (`dflash.py:116-123`).
- Probabilistic draft sampling at T>0 (`docs/optimizations.md:389`).

## Rejected

- DSpark's confidence head and Markov bigram logits (`models/mtp/dflash.py:46-79, 165-176`): drafter lane, and the DSpark drafter already lost on a 3090 (`docs/long-context.md:615-640`).
- FLy entropy-gated deferred verification (`vllm@main:.../fly.py`): lossy.
- YOCO/compact truncation with window-plus-backtrack keep: V4 SWA-specific; HyperQwen's align mode is the GDN analogue.
- DeepInfer's async scheduler with stale-confidence GPU recompute as a whole: vLLM async plus spec-1/2 cover it.
- `DRAFT_PROB_PENALTY` e^−5e-5 (`common.py:53`): a numeric guard for top-k residuals; no measured issue here.
- DP attention, EP, MoE, SM90/SM100 kernels and Ascend ops: not applicable to one sm86 GPU.
- Whole-drafter CUDA graph plus fused input copy: ≤2% measured host gap.
- Writing the drafter's KV inside the target forward: vLLM already precomputes it fused.

## CPU checks

numpy only; run from `docs/research/deepinfer/cpu-checks/`.

| Script | Command | What it does |
|---|---|---|
| `spec_varlen_sim.py` | `python3 -I spec_varlen_sim.py` (~60 s) | fixed k vs vLLM's adaptive policy, stale and fresh, for B, D, D′, E and C4; self-checks that a flat cost verifies all drafts and zero confidence verifies none |
| `spec_calib.py` | `python3 -E spec_calib.py` (imports its sibling, so not `-I`) | the grid search that picked the acceptance parameters |

Model: a per-step AR(1) latent (ρ = 0.8) shifts the logit of each position's Beta-distributed conditional acceptance; the estimator sees the true probability plus N(0, σ) logit noise; the policy is copied from `adaptive_verification.py:282-350`; cost is F + c_tok × tokens (+0.7 ms per MTP draft pass). Calibration: position 0 acceptance 0.766 and 3.16–3.21 tokens per step at k=7 (measured 3.14/3.34); C4 c_tok = 0.7 ms gives fixed k=3 +7.9% over k=7 (measured `DYNAMIC_K` +7.7 to +9.2%). The lead re-ran the sim from the repo copy; the output matches the committed `.out` byte for byte. These are synthetic, not measurements.

## Open questions

1. Does the FA2-fp8 target backend (D's default) accept a device q_len below the host bound, with a varlen graph bound? spec-1's boot log answers it.
2. What is the real cost per verify width at 150k and 240k on pthfa, FA2-fp8 and KVarN? It sets the sign of spec-0 and spec-2, and the stairs matter more than the average.
3. How well does vLLM's online logistic estimator fit DFlash2's candidate scores? The sim assumes σ = 1.
4. How autocorrelated is acceptance step to step at one stream? It decides how much stale versus fresh matters.
5. What is the root cause of the KVarN corruption on a verify-length change (`start_qwen.sh:399-428`)? It blocks spec-1, spec-2 and spec-4 at E.
