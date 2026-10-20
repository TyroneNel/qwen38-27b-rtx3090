# Runtime: scheduler, CUDA graphs, determinism, structured outputs

Lane: engine runtime (Sonnet, high effort), verified by the lead. Part of the [DeepInfer expedition](README.md).

Citation forms: HyperQwen paths are repo-relative. `DeepInfer/...` refers to deepseek-ai/DeepInfer at `774ce24`. `vllm@v0.31.0:...` refers to the vLLM tag HyperQwen pins, and `vllm@main` to upstream at `4bd75e952a` (v0.32.1rc0+68).

## Verdict

HyperQwen does not lose meaningful time in the runtime layer. Host overhead is under 2% of a step, and mode A's wall clock is fully explained by decode and prefill arithmetic. DeepInfer's zero-overhead scheduler has nothing to hide here.

The one valuable DeepInfer runtime idea is confidence-scheduled variable-length verification, and **it already exists in vLLM 0.31** as "adaptive verification". It is switched off for this model because the GDN backend assumes the CPU knows the query lengths. Unblocking it pays only with several concurrent speculating users, which no current mode targets.

## Evidence: there is no host overhead to recover

| Measurement | Value | Source |
|---|---|---|
| Kernel sum vs step wall, 8k, target only | 0.878 + 0.746 + 21.47 = 23.09 ms vs 23.49 ms: **1.7% gap** | `docs/long-context.md:536-545` |
| GPU idle in a 4k prefill | 15 ms of 2.05 s, **~0.7%** | same section |
| Async-scheduling cost at batch 1 | under 1% | `docs/optimizations.md:126-130, 289-297` |
| Mode A, 256 requests × 512 out | 131,072 / 1,094 tok/s = 119.8 s decode, plus 32,768 / 1,812 = 18.1 s prefill, gives ≈ 950 tok/s predicted vs **942 measured** | `batch/README.md:18` |

Async scheduling is on in batch mode (`batch/start_qwen.sh:311`). Single-user mode turns it off (`ASYNC_SCHED=0`, `single-user/start_qwen.sh:435, 845-852`) when the long DFlash2 verify block needs the lookup patch to feed back how many draft tokens to put up next. That costs under 1%.

DeepInfer's ZOS (`DeepInfer/csrc/engine/zos.cpp`) hides schedule(N+1) and post-process(N−1) under forward(N). It takes matched rows' tokens from the previous step's sampled ids on the GPU and corrects sequence lengths on the GPU. That design exists for 60+ sequences per rank on MoE, where GPU steps are very short. A 20+ ms dense step at batch 1–4 leaves nothing to hide.

## Candidates

### runtime-1: device-side varlen metadata for GDN, then turn on vLLM adaptive verification for DFlash2

- **DeepInfer.** `DeepInfer/csrc/engine/mtp_varlen.hpp` (MTPVarlen::schedule plus the runtime estimators) chooses the total number of verified tokens that maximizes expected accepted tokens per ms. It reads that off a decode cost curve profiled at boot (`DeepInfer/deepinfer/worker/model_runner.py`, cached by `decode_perf_cache.py`). It sorts draft slots by predicted acceptance and verifies the best prefix. Under async scheduling the CPU fixes only the total budget, and a TileLang kernel recomputes the per-request lengths on the GPU (`DeepInfer/doc/inference_deployment_qa.md:239-264`).
- **DeepInfer's own numbers** (same doc, table at 253-260): verified length drops 5 → 3.2–4.4 and accepted length barely moves (3.29 → 3.20). Throughput change by concurrency per rank: **−1.5% at 16**, +1.6% at 32, +3.6% at 48, +5.4% at 60, +6.7% at 72, +11.0% at 96. The gain grows with batch, and the method *loses* at low concurrency.
- **vLLM 0.31 already has it.** `vllm@v0.31.0:vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` (docstring: "Adaptive verification for DSpark speculative decoding") does a global top-k over survival probabilities against a profiled cost curve. `acceptance_estimator.py` is an online logistic estimator for drafters that have no confidence head. It's gated by `supports_device_cpu_query_lens_mismatch()`, which returns `not cls.is_ssm()` (`vllm@v0.31.0:vllm/v1/attention/backend.py:216-228`). GDN reports `is_ssm()` true (`vllm@v0.31.0:vllm/v1/attention/backends/gdn_attn.py:37`) and builds its spec masks from CPU query lengths. Upstream added device-side varlen decode for Mamba2 only (#53036).
- **HyperQwen today.** The lookup patch carries per-request adaptive and sticky draft lengths, plus a hook for vLLM's manager (`patches/dflash2-lookup-drafting.patch:157, 1053`). There is no budget shared across requests. Modes B–E are single-stream, and mode A runs no speculation.
- **Estimated gain.** Zero in A, B, D and E: at N=1 the best global prefix *is* the best uniform k. In C at 2–4 concurrent users it's **(inferred)** +0–5%, and DeepInfer's −1.5% at 16 seqs/rank warns that low concurrency can go negative. The lane's illustrative simulation (`cpu-checks/runtime_topk_sim.py`: per-request geometric survival with a ~ Beta(6,4); step cost 1 + r·(extra tokens) with r *unmeasured*) gives +0.0% at N=1, +2.0 to +5.7% at N=4 and +4.1 to +7.0% at N=16. It assumes a linear cost slope and ignores the drafter's own cost, so treat it as a ceiling.
- **Effort.** M. **Risk:** high-impact correctness. A GDN state-index or conv-state error on rows with variable accepted length corrupts output silently, and it has to interact correctly with mamba-align prefix caching.
- **Porting plan.** A new patch `patches/gdn-device-varlen-spec.patch` against `vllm/v1/attention/backends/gdn_attn.py` that derives the spec/non-spec split on the device, following #53036 for Mamba2, and flips the mismatch flag for GDN. Make the lookup patch's hook yield to vLLM's manager. Add a launcher knob near the speculative-config block in `single-user/start_qwen.sh`.
- **GPU validation.** `bench/conc_ladder.py` at C=2, 4, 8 and 16 with `SPEC=dflash2`, adaptive on vs off, same seeds. Win: +3% at C ≥ 4, an unchanged greedy output hash at C1, and `bench/api_smoke.py` passing.
- **Recommendation.** Do this only if multi-seat speculation becomes a target (see open questions). Run runtime-2 first, because it measures the cost slope r that decides the whole payoff.

### runtime-2: step-cost curve tool (ms per step vs verified tokens on the 3090)

- **DeepInfer.** Profiles graph size against ms at boot and caches the curve on disk (`DeepInfer/deepinfer/worker/decode_perf_cache.py`).
- **HyperQwen.** `DFLASH_TOKENS=15` (mode C) and the draft counts were tuned by end-to-end sweeps. There is no recorded curve of ms per step against total query tokens.
- **Value.** It supplies the unmeasured slope for runtime-1 and the spec-lane policies, and it puts any `DFLASH_TOKENS` retune on a curve instead of a sweep. There's no runtime change.
- **Effort.** S. **Risk:** none (bench only).
- **Plan.** `bench/step_curve.py`: under each mode's launcher with CUDA graphs on, replay decode forwards at total_q = 1…32 and report ms per total_q at 8k and 64k context. Confirmation: if the slope is under 5% of the base step per extra token, runtime-1's ceiling holds.

### runtime-3: bound the tail of forced tool calls

- **HyperQwen.** A forced tool call sometimes keeps decoding whitespace inside the `qwen3_coder` XML until `max_tokens`. This happened in 2 of 20 seeds on 0.31.0 batch, 1 of 20 single-user and 5 of 20 on 0.30.0, and xgrammar's `disable_any_whitespace` (JSON only) doesn't reach it. The doc says "Not fixed here" (`docs/vllm-0.31.md:161-165`).
- **DeepInfer.** Its structured-output matcher has trigger tokens plus a bound token, so it caps the tokens allowed after the grammar completes (`DeepInfer/deepinfer/structure_output/matcher.py:58-134`). vLLM already gates grammar on the end of reasoning. The bounded tail is the piece vLLM lacks.
- **Gain.** No throughput change. It removes a 5–10% chance that a forced tool call runs to `max_tokens`, which saves latency and tokens for agents.
- **Effort.** S–M: an EBNF/structural tag that ends at `</tool_call>`, or failing that a small patch in `vllm/v1/structured_output` that terminates once the matcher accepts the closing tag. **Risk:** a stricter grammar may reject valid calls with unusual whitespace. The lane did not reproduce the failure.
- **Validation.** The 20-seed forced `get_time` test from `docs/vllm-0.31.md`. Win: 0 runaways in 40 seeds, with every valid call still accepted.

## Already in HyperQwen or vLLM 0.31

- Async scheduling wherever it's compatible (above).
- Chunked prefill mixed with decode under a 2,048-token budget (`batch/start_qwen.sh:312`). On one GPU this beats DeepInfer, which never mixes prefill and decode in one step because its prefill voting exists for data-parallel lockstep (`DeepInfer/csrc/engine/scheduler.hpp`).
- Greedy and seeded determinism checks (`bench/api_smoke.py`). Gotcha 13 states the same preconditions as DeepInfer's README (batch invariance on, varlen MTP off, fixed seed).
- Structured output through xgrammar (pinned 0.2.7), `--tool-strict-level parameter`, the spec-terminated xgrammar fix, grammar gated on reasoning, and an enforced thinking budget.
- CUDA graphs: vLLM pads spec decode to captured sizes. DeepInfer captures one graph per padded total_q and a separate DFlash drafter graph per sequence count (`DeepInfer/deepinfer/worker/cuda_graph_runner.py`, `dflash_cuda_graph_runner.py`), which is the same shape **(inferred)**.

## Rejected

- **ZOS port:** 0.7–1.7% host overhead, nothing to hide. It also conflicts with the sync scheduling that lookup drafting needs.
- **Prefill/decode separation and prefill voting:** these are DP-lockstep machinery, and mixed chunked prefill is better on one GPU.
- **LCFS decode, preempt-shortest:** mode A is bounded by KV residency, and with identical synthetic requests a policy can't change tok/s.
- **`flops_estimator.hpp` as an admission budget:** it's an MFU-reporting class with MoE/MLA/FP4/FP8 coefficients (`DeepInfer/csrc/engine/flops_estimator.hpp:14-80`).
- **Full batch invariance:** GDN/mamba backends reject it (`vllm@main:vllm/v1/attention/selector.py`). HyperQwen's custom attention backends aren't batch-invariant, and Marlin's tiling varies with token count. Upstream #60524 gets batch invariance for wNa16 only by unpacking to bf16, which throws away the int4 bandwidth win.
- **Deterministic DeepGEMM/TileKernels paths:** SM90/SM100 only.
- **Multi-node, MoE, EP, PD disaggregation, autoscaling:** not applicable to one 3090.

## Open questions

1. Is multi-seat speculation (DFlash2 at 2–16 concurrent) something you want to serve? runtime-1 is worth nothing otherwise.
2. Does the forced-tool-call runaway hurt real agent traffic? If it does, runtime-3 moves up.
3. Does real batch traffic have mixed prompt lengths? Only then could a shortest-first TTFT policy be tested.
4. Are you willing to carry another GDN vLLM patch across the next pin?
