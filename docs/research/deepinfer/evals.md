# Evals, benchmarks, quality gates and API surface

Lane: evaluation and tooling (Sonnet, medium effort), verified by the lead. Part of the [DeepInfer expedition](README.md).

Citation forms:
- HyperQwen paths are repo-relative.
- `DeepInfer/...` refers to deepseek-ai/DeepInfer at `774ce24`.
- `DeepSpec/...` refers to deepseek-ai/DeepSpec at `005e03b`.
- `recipe/...` refers to deepseek-ai/deepseek-recipe at `935817c`.
- `vllm@v0.31.0:...` refers to the pinned vLLM tag.

## Verdict

1. **DeepInfer's quality gate checks the decode path deterministically.** It forces known tokens through decode (`golden_token_ids`, `DeepInfer/deepinfer/sampling_params.py:60, 83`) and compares their logprobs against a full-prefill recompute (`DeepInfer/tests/models/test_kl.py`). HyperQwen's perplexity gate only exercises prefill, and its GSM8K gate is too coarse to see a 1-point change. So the decode-only machinery (fp8/int8 KV, KVarN, split-KV verify attention, the fp16 recurrent state) has no sensitive end-to-end gate.
2. **The CUDA-sync guard is already in vLLM 0.31.** `VLLM_GPU_SYNC_CHECK=warn|error` (`vllm@v0.31.0:vllm/utils/gpu_sync_debug.py`, `envs.py:651`) is a better design than DeepInfer's arbiter, and HyperQwen never uses it. HyperQwen has already lost 5% once to a hot-path `.item()` (`patches/dflash2-lookup-drafting.patch:974-977`).
3. **Every speed and acceptance number in HyperQwen is measured with thinking off.** `bench/labd_bench.py:74, 83`, `bench/labd_accept.py:141` and `bench/conc_ladder.py:122` all send `enable_thinking: False`. The chat template's default effort is xhigh (`prepare/translate_chat_template.py:59-61`), so a default client spends most of its tokens in a regime the headline 127 tok/s never measured. This is the cheapest large unknown in the project.

HyperQwen's acceptance methodology (`bench/labd_accept.py`: teacher-forced, prefix resync, dirty-chunk accounting) is already more careful than DeepSpec's or DeepInfer's, which use free-running traces. The weaknesses are panel breadth and the missing thinking phase, not the method.

## How much the current gate can resolve

These numbers are arithmetic, not measurements:
- GSM8K at n=200 and p≈0.95 has a binomial standard error of √(0.95·0.05/200) ≈ **1.5 points**. The T4 bar is a 1.0-point drop (`bench/t4_battery.py:65`), which is two questions.
- Paired greedy arms with about 10 discordant questions (an assumption) give a paired SE ≈ √10/200 ≈ 1.6 points.
- HyperQwen already says 8 prompts × 1k tokens has ±3% spread on tokens/step (`drafter/README.md:112-118`).
- `bench/quality_battery.py:83-95` keeps one aggregate float. It drops `finish_reason`, and when "Final answer:" is missing it silently falls back to the last number in the text.

GSM8K is a smoke test for garbage output, not a gate for small quant or kernel changes. Perplexity and logprob KL average over tens of thousands of tokens, so they're the sensitive instruments.

## Proposed gate tiers

The times below are derived from the README rates (prefill ~1.8k tok/s, 127 tok/s single stream, ~1,035 tok/s at C64). Per-item token counts are **(inferred)**.

| Tier | When | Contents | Est. time on a 3090 |
|---|---|---|---|
| T0 CPU | every PR | the existing patch-integrity steps, plus the tool-parser chunk fuzz (evals-6) | < 1 min |
| T1 fast | after any kernel, quant or patch change | boot, `verify.sh`, `api_smoke.py`, perplexity on 3 corpora, a decode-vs-prefill KL probe at 4k (evals-1), a sync-check boot (evals-2), and the canary for changes meant to be neutral (evals-7) | ~5 min + boot |
| T2 medium | before merging a perf or quant change | paired GSM8K, greedy plus 3 sampled seeds (evals-3); `labd_accept.py`; needle at 100k; residue sweep; KL probe at 16k; `conc_ladder` | 15–25 min |
| T3 full | releases, new quant or drafter | `t4_battery.py` per mode, thinking-on IFBench, the thinking cohort (evals-4), KL probe at 64–100k, soak | 1–3 h |

DeepInfer runs AIME/GPQA/HLE-style 32-sample evals (`DeepInfer/eval/run_eval.py:24-26`) as release sanity checks. On one 3090 they take hours, so they aren't recommended as per-change gates.

## Candidates (ranked)

| ID | Title | Value | Effort | Confidence |
|---|---|---|---|---|
| evals-1 | Decode-vs-prefill logprob KL probe per serving mode | High | S (4–16k), M (64–100k) | High |
| evals-2 | Run vLLM's `VLLM_GPU_SYNC_CHECK` as a hot-path sync gate | High | S | High |
| evals-3 | Auditable, paired quality battery (per-item JSONL, finish_reason, sampled arm, McNemar/Wilson) | High | S | High |
| evals-4 | Thinking-on speed and acceptance cohort, split by reasoning vs answer phase | High (tests the headline) | S | Medium |
| evals-5 | Special-token injection hardening (deepseek-recipe `special_ranges`) | Medium (safety) | M | Medium-low |
| evals-6 | Streamed tool-call test with multi-token deltas, plus a tool case in `api_smoke` | Medium | S | High |
| evals-7 | Golden greedy canary keyed by (GPU, pin, mode) | Medium | S | Medium |
| evals-8 | Per-component GPU time on the serving path (nsys first, graph probe timers later) | Medium | M | Medium |
| evals-9 | Broader acceptance panel from DeepSpec's datasets, run on labd_accept's method | Medium | S | Medium |
| evals-10 | Throughput vs per-user Pareto sweep per mode | Low-medium | S | High |
| evals-11 | Automatic warmup detection | Low | S | High |

### evals-1: decode-path vs prefill-path logprob KL probe

- **DeepInfer.** `DeepInfer/tests/models/test_kl.py` forces random non-special tokens through decode, re-feeds prompt plus output with `prompt_logprobs`, and compares the two. `DeepInfer/deepinfer/testing.py:4-12` reports cosine, RMSE, max |Δ| and the k3 KL estimator `mean(exp(d) − 1 − d)` with d = logp_x − logp_y. It needs only the chosen-token logprobs. The long variant uses 16,389 input tokens (`DeepInfer/examples/_prompts.py:26-27`).
- **HyperQwen.** Nothing compares decode logprobs against prefill logprobs. `bench/test_spec_decode_attn.py` checks a kernel against a reference, not end to end. The decode-only paths, where "benchmarks great, outputs garbage" bugs hide, have no low-noise gate.
- **Plan.** `bench/decode_kl_probe.py`, built on `bench/harness.py` (stdlib only):
  1. Greedy `/v1/completions` with `logprobs: 0` and `return_tokens_as_token_ids: true` (as `labd_accept.py:158` already does), for 8 fixed prompts × 512 new tokens.
  2. Resend prompt plus generated ids with `max_tokens: 1, prompt_logprobs: 0`.
  3. Compute k3 KL plus mean and max |Δlogp| over the generated positions.

  Write JSON next to the quality battery's output and follow bench/README's exit-code contract. For a negative control, shift the tokens or turn on a known-lossy knob, and confirm the KL rises.
- **Risk.** The baseline isn't zero: decode and chunked prefill legitimately differ (≈0.9% hidden-state difference with the fp16 state, `drafter/README.md:111`). Calibrate per mode and gate at about 2× baseline. `prompt_logprobs` at 100k is heavy, so use 4–16k in the fast tiers.
- **Validation.** Two runs per mode to get the repeat spread. Then flip INT8_ACT, FA2_FP8KV and KVarN, and confirm the probe separates them in cases where GSM8K shows nothing.

### evals-2: vLLM's GPU-sync check as a gate (don't port DeepInfer's arbiter)

- **DeepInfer.** Model-forward and sampler launches run inside `forbid_cuda_sync()` (`DeepInfer/deepinfer/worker/model_runner.py:322-326, 357-361`). Because torch's sync-debug mode is process-global, `_SyncDebugArbiter` (`DeepInfer/deepinfer/utils.py:72-111`) keeps a per-thread intent stack in which "allow" beats "forbid". Its tests are CPU-only: they monkeypatch the torch switch and hammer it from threads (`DeepInfer/tests/test_cuda_sync_guards.py`).
- **vLLM 0.31.0 already does this, and better.** It uses ContextVar scoping, arms torch at "warn" with a hook that raises only on checked threads, and also catches `non_blocking=True` copies from pageable memory. It's armed after warmup around `execute_model` and `sample_tokens`, so drafter and lookup code run inside it.
- **HyperQwen.** It isn't used anywhere (a grep finds nothing). Patch scan: `prefill-attn-int8.patch:89` calls `int(seqused_k[0].item())`, but only when `max_seqlen_q > 16` on a single sequence. That's a sync per full-attention layer per prefill chunk, never in decode or verify, so it's cheap. The scan is still worth running per mode, because single-user mode runs `--no-async-scheduling` and every re-pin can move syncs.
- **Plan.** `bench/sync_check.sh` boots a mode with `VLLM_GPU_SYNC_CHECK=warn`, runs `api_smoke.py` plus 8 real prompts, extracts the unique sync sites from the log, and fails on any site missing from `bench/sync_allowlist.txt`. Legitimate syncs get upstream's `gpu_sync_allowed()` inside the patch, noted in PATCHES.md. Use it on test boots only, never as a production default, because `error` mode kills the engine.
- **Validation.** Add a `.item()` to a patched path in a throwaway venv and confirm the check names it. On the current pin, all five modes should produce a short, stable allowlist.

### evals-3: an auditable, paired quality battery

- **DeepInfer.** It keeps raw generations between generate and score, and records `coverage_complete`, `finish_reasons` and truncation warnings (`DeepInfer/eval/score.py:132-167, 222-224`). Pass@k uses the unbiased estimator (`DeepInfer/eval/common/passk.py`).
- **Plan.** Write one JSONL row per item: `{idx, correct, finish_reason, extracted_via, completion_tokens, text}`. Add `--seeds N` for a sampled arm. Mode B ships sampling, and quant effects on acceptance are larger under sampling (`drafter/README.md:89-90`). Add a stdlib `bench/compare_quality.py` that prints Wilson 95% intervals, McNemar discordant counts, an exact binomial p and the perplexity delta. The T4 GSM8K bar should fail only on a significant paired loss (p < 0.05). Warn on `finish_reason=length` or a fallback extraction.
- **Validation.** Re-run the known pairs in `docs/quality.md` (fp32 vs fp16 state, all-linear int8) and confirm the verdicts match the perplexity columns.

### evals-4: thinking-on speed and acceptance cohort

- **Evidence.** HyperQwen's harnesses hard-code thinking off (above), and `docs/benchmarks.md` says so. DeepSpec also evaluates thinking-off (`DeepSpec/deepspec/eval/base_evaluator.py:538`, with the True line commented out). DeepInfer's own evals run thinking-mode at temperature 1.0. The drafter data (`drafter/gen_data.py`) already covers both.
- **Plan.** Add `--think` to `bench/conc_ladder.py` and to the single-user path of `run_benchmarks.sh`. In a copy of `labd_accept.py`, split the per-position counters at the `</think>` position. Report tok/s, tokens/step and the fraction of tokens spent reasoning.
- **Why it ranks high.** It's cheap, it confirms or corrects the headline number, and it gives the drafter lane a per-phase acceptance target. Reasoning text may draft better or worse than answers, and nobody has measured which.

### evals-5: special-token injection hardening

- **deepseek-recipe.** The renderer records the byte ranges of the control tokens it inserted, and only those encode as special. Any spelling of them in user, tool-result or tool-argument text encodes as plain text (`recipe/docs/tokenizer.md:77-85`; also `DeepInfer/doc/inference_deployment_qa.md`). The lane confirmed the stock behaviour with recipe's own tokenizer: a typed control token comes back as the control id.
- **HyperQwen.** No mitigation exists, and vLLM 0.31.0 has no `split_special_tokens` (git grep is empty). Qwen's `<|im_start|>`, `<|im_end|>` and tool markers are added tokens, so a coding agent pasting a chat template or tokenizer config into a tool result **(inferred)** can end a turn or open a fake system turn.
- **Plan.** Probe first: `bench/special_token_probe.py` posts content containing `<|im_end|>` and checks with `/tokenize` (served via the `tokenize-v1-route` patch) whether the control id appears. Only if it reproduces, write `patches/serve-escape-special-tokens.patch` against vLLM's HF renderer.
- **Risk.** Escaping changes how affected prompts tokenize, and so changes their prefix-cache keys.

### evals-6: streamed tool calls under speculative decoding

- Upstream's qwen3_coder parser test streams one token per delta (`vllm@v0.31.0:tests/tool_parsers/test_qwen3coder_tool_parser.py:173-202`). Under speculation the server emits up to K+1 tokens per step, so deltas that span several tokens and split markers are realistic, and nothing tests them. deepseek-recipe re-chunks its tests down to one character per chunk and compares against the unchunked parse (`recipe/deepseek-recipe/src/stream/processor.rs:564-567`).
- **Plan.** Add a CPU test, `bench/test_tool_parser_chunks.py`: 5 canned outputs with 1–3 calls and nested JSON arguments, split at random sizes 1–8 over 200 seeds, asserting that the streamed `tool_calls` equal the one-shot parse. Add `t_tool_call_stream` to `api_smoke.py`, which has no tool case today. Gotcha 36 records an earlier tool-call failure under a speculator.

### evals-7: golden greedy canary

DeepInfer compares decode perplexity against a stored golden with `==` (`DeepInfer/tests/models/test_offline_inference.py:154-158`). A deliberate refresh script is the only way to change the golden (`DeepInfer/examples/refresh_offline_decode_ppl.py`).

HyperQwen's equivalent is ad hoc: "7 of 9 long greedy prompts come back token-identical" (`docs/optimizations.md:329`). The proposal is `bench/canary.py`: 8 prompts × 256 greedy tokens at C1 (~16 s of decode). Store the ids and top-1 logprobs keyed by (GPU, pin, mode) and provide a `--refresh` flag. Treat a mismatch as "inspect", not an automatic fail, because Triton, torch and driver versions can move bits.

### evals-8: per-component GPU time on the serving path

DeepInfer records `cudaEventRecord` nodes inside captured graphs and re-points them before each replay (`DeepInfer/deepinfer/timer.py:76-200`, `DeepInfer/csrc/graph_timer/`). HyperQwen has no profiling tooling in the repo: step times are end to end, and attribution comes from microbenchmarks. The table at `docs/long-context.md:536-545` came from a one-off torch profile.

Start with no new code: an `nsys` capture of a 10 s C1 run per mode, with the breakdown written into the docs. That tells the priority-1 speed work where the 7.9 ms per DFlash2 step goes. Graph probe timers (a patch) are worth building only if continuous attribution is needed.

### evals-9 to evals-11

- **evals-9.** DeepSpec ships nine evaluation sets (gsm8k, math500, aime25, humaneval, mbpp, livecodebench, mt-bench, alpaca, arena-hard-v2; `DeepSpec/eval.py:11-27`). Taking 50 items per set and running them through `labd_accept.py`'s chunk-resync loop gives acceptance by domain. Keep one free-running sampled arm. The dataset licences are unchecked, so keep a local, uncommitted copy. Note that DeepSpec's "acceptance length" is accepted + 1 per proposal, so check the definition before comparing it with HyperQwen's tokens/step.
- **evals-10.** Extend `bench/conc_ladder.py` to N ∈ {1, 2, 4, 8, 16, 32, 64} with JSONL output and an optional Pareto plot, modelled on `DeepInfer/examples/sweep/run_sweep.sh`. The 8–64 range has no curve today.
- **evals-11.** Replace "run it twice after a restart, keep the second" (`bench/run_benchmarks.sh:10-12`) with a probe that repeats until two runs agree within 3%.

## Already in HyperQwen

- Teacher-forced acceptance with resync, dirty-chunk detection, per-position counters and block-bootstrap deltas (`bench/labd_accept.py`, `docs/long-context.md`).
- Exit-code contract and negative controls (bench/README.md, `mq3d_capacity_property.py --mutate`, `test_t4_battery.py`).
- Reasoning-effort aliases (`prepare/translate_chat_template.py`). Upstream already maps `reasoning_effort="none"` to thinking off, and Anthropic `/v1/messages` and the Responses API are upstream in 0.31.
- Perplexity, GSM8K, IFBench, residue, needle, soak, verbatim and the T4 battery.

The lane ran HyperQwen's CPU tests (`test_t4_battery.py`, `verbatim.py`, `mq3d_capacity_property.py`) and all of them pass. `test_harness.py` needs `bench/` on `sys.path`, so it fails under `python -I` only.

## Rejected

- Exact `==` golden decode perplexity as the main gate: it needs batch-invariant kernels HyperQwen doesn't have (see [runtime.md](runtime.md)). Only the canary form (evals-7) is kept.
- AIME/GPQA/HLE as per-change gates: they take hours on one 3090.
- LLM-judge scoring: GSM8K and IFBench have deterministic scorers.
- deepseek-recipe's Rust conversion layer and DSML tool format: HyperQwen serves Qwen's XML format through vLLM's parser.
- The crash-header excepthook (`DeepInfer/deepinfer/logger.py:121-146`): vLLM's process model plus `verify.sh` already detect server death.
- EP/DP/EPLB, disaggregation, KV storage nodes, autoscaling, NPU markers: none apply to one 3090.

## Open questions

1. Does Qwen3.8's tokenizer turn a typed `<|im_end|>` into the control id under vLLM 0.31's chat path? There's no Qwen tokenizer on the research box to check.
2. Does default client traffic really run thinking at xhigh? If so, evals-4 is the biggest unknown behind the headline numbers.
3. Does `prompt_logprobs` in 0.31 bypass the prefix cache, and what is its memory ceiling at 64–100k with this vocabulary? That limits evals-1's range.
4. Can DeepSpec's `eval_datasets` be redistributed inside HyperQwen?
