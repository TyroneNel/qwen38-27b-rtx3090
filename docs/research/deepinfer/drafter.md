# Drafters: models, training recipe, data and eval

Lane: drafter (Opus, high effort), verified by the lead. Part of the [DeepInfer expedition](README.md).

Citation forms:
- HyperQwen paths are repo-relative.
- `DeepInfer/...` refers to deepseek-ai/DeepInfer at `774ce24`.
- `DeepSpec/...` refers to deepseek-ai/DeepSpec at `005e03b`.
- `vllm@v0.31.0:...` refers to the pinned vLLM tag. `qwen3_dflash2.py` and `qwen3_dspark.py` are `vllm/model_executor/models/` at that tag.
- "Estimate" marks arithmetic, not a measurement. **(inferred)** marks a number or fact not seen directly in code or docs.

## Verdict

1. **Keep DFlash2.** It already does a cheaper, richer version of DSpark's semi-autoregressive correction, and it won the head-to-head on this 3090: 3.83 vs 3.37 tok/step, 150 vs 114 tok/s (`docs/long-context.md:619-631`). DeepSpec publishes no acceptance numbers to argue otherwise.
2. **Borrow two pieces of DSpark, not the drafter:** the confidence head, which drives a per-step verify trim at long context (drafter-1), and the TV-dominant loss, which targets the int4 DFlash2's documented ~5% sampling gap (drafter-3).
3. **Build the offline gate first** (drafter-2). Drafter A/Bs today sit inside ±3–5% run noise. A deterministic tau_prob makes drafter-3 and any fine-tune measurable.
4. **Mode D stays on MTP.** DFlash2's long-context loss is mostly verify cost and a larger fixed step, not training. Even with perfect trimming, DFlash2 reaches about 41 tok/s on QA-like work at 112k against MTP's 62–73 (check c2).
5. **A DeepSpec fine-tune of DFlash2 is possible but not cheap.** DeepSpec trains DFlash and DSpark, not DFlash2. It needs a training forward written from vLLM's inference module, a 0.4–3 TB cache and a rented 80 GB GPU (drafter-4). Gate it on drafter-2.

## Answers to the brief

### Q1. DSpark vs DFlash vs DFlash2 vs EAGLE3 vs MTP

| | DFlash (DeepSpec) | DSpark (DeepSpec, DeepInfer) | DFlash2 (incoai, shipped) | EAGLE3 (DeepSpec) | Qwen MTP (shipped for D) |
|---|---|---|---|---|---|
| Backbone | 5 target-shaped layers. One parallel block of 7: the anchor plus mask tokens (`DeepSpec/config/dflash/dflash_qwen3_8b.py`). Non-causal inside the block, causal to context (`DeepSpec/deepspec/modeling/dspark/common.py:78-106`). | Same as DFlash (`DeepSpec/config/dspark/dspark_qwen3_8b.py:10-57`, block_size 7, 5 layers) | 5 Qwen3 layers: H 5120, 32Q/8KV×128, MLP 17408, sliding window 2048, non-causal. Adds dynamic grouped convs in attention and MLP (`qwen3_dflash2.py:144-287`). | 1 layer, chained 7 steps (`DeepSpec/config/eagle3/eagle3_qwen3_8b.py:11-17`) | 1 layer, chained, 3–4 drafts |
| Conditioning | `fc` over 5 target layers, then `hidden_norm`; these become the context K/V (`DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py:43-151, 240-246`) | Same. Taps `[1,9,17,25,33]` for 8B and `[1,10,19,28,37]` for 14B (`dspark_qwen3_14b.py:15`). | `fc` over target layers 5/19/33/47/61 (`drafter/README.md:64-66`) | Same tap ids | Target last hidden plus the token embedding |
| Intra-block dependency | None (fully parallel) | **Markov head**: low-rank bigram `W2·W1[prev]` added to the logits and sampled sequentially (`DeepSpec/deepspec/modeling/dspark/markov_head.py:8-90`; `DeepInfer/deepinfer/model_executor/models/mtp/dflash.py:279-345`) | **CandidateSelector**: top-16 candidates per slot from the shared lm_head, then a bilinear predecessor × successor edge score modulated by the drafter hidden state (`qwen3_dflash2.py:290-358, 405-412`) | Chain | Chain |
| Extra head | None | **Confidence head** `Linear(H + rank → 1)` (`DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py:254-267`; `common.py:43-49`; `DeepInfer/.../mtp/dflash.py:46-79, 155-186`) | None | None | None |
| Loss | CE only (`DeepSpec/config/dflash/dflash_qwen3_8b.py:18-27`) | 0.1·CE + 0.9·L1(p_d, p_t), positions weighted exp(−k/4), plus BCE(conf, 1 − TV) (`DeepSpec/deepspec/modeling/dspark/loss.py:25-37, 60-87, 152-163, 248-252`) | Unknown (third party) | Step-decayed CE (0.8) | Qwen's own |
| Acceptance reported | **None.** DeepSpec's README has no table; `assets/` holds only `dspark.drawio`. | DeepInfer: accepted length 3.29–3.30 at 5 drafts, fixed verify (`DeepInfer/doc/inference_deployment_qa.md:253-260`) | 3.14 / 3.34 tok/step sampling / greedy at k=7 (`docs/optimizations.md:396`) | None | 2.8–3.0 (same table) |
| Measured on this 3090 | – | **RadixArk DSpark: 3.37 vs DFlash2's 3.83 tok/step, 114 vs 150 tok/s** (`docs/long-context.md:619-631`) | Shipped | None | Shipped |

**Draft cost per step on a 3090** (check a; estimates, lower bounds at 936 GB/s):
- DFlash2 W4A16 reads about 1.0 GB of drafter (`drafter/quant_dflash2.py:11`) plus the 0.65 GB int4 lm_head (`single-user/start_qwen.sh:16`) for candidate top-k, once per step: at least 1.76 ms. The selector gathers 16 codebook rows per slot, which is negligible.
- DSpark's Markov W2 is 256 × 248,320 = 63.6M params, read once per sequential draft position. In bf16 that is 7 × 127 MB = 0.89 GB, at least 0.95 ms per step; in int4, 0.24 ms.
- MTP-4 reads (215 MB + 107 MB) × 4 sequential passes = 1.29 GB.
- Backbone plus `fc` is 1.73B, so the selector and convs are about 190M of the 1.92B card. That fits a selector rank near 384 **(inferred**; `selector_rank` isn't mirrored locally).

EAGLE3 is a single-layer chain drafter, which is MTP's niche, and HyperQwen already showed Qwen's MTP sits at the ceiling for that shape (`drafter/README.md:29-37`).

### Q2. Can HyperQwen train its own DSpark or DFlash with DeepSpec?

- **Cache size.** DeepSpec caches bf16 hidden states for each tapped layer plus the last hidden: `(n_taps + 1)·H·2` bytes per token. bf16 is asserted (`DeepSpec/deepspec/data/target_cache_dataset.py:24, 154`). For Qwen3-4B that is 30,720 B/token, so DeepSpec's 38 TB is about 1.24B tokens. For Qwen3.8-27B with 5 taps it is **61,440 B/token** (check a′):

  | Tokens | 5 taps | 3 taps | 5 taps, no last hidden |
  |---|---|---|---|
  | 7M (HyperQwen's corpus size) | 0.43 TB | 0.29 TB | 0.36 TB |
  | 50M | 3.1 TB | 2.0 TB | 2.6 TB |
  | 1.24B (DeepSpec scale) | 76 TB | 51 TB | 63 TB |

- **Ways to shrink it.** Fewer tokens is the big one. Fewer taps and dropping `target_last_hidden` are both suggested upstream (`DeepSpec/scripts/data/README.md:115-121`); the last hidden is only needed for the L1 and confidence terms. Replacing the last hidden (10,240 B) with top-64 target logits (384 B) is not implemented; it needs a patch to `DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py:448-465`, which takes logits from `lm_head(last_hidden)`. fp8 storage would halve everything, but DeepSpec asserts bf16.
- **Compute** (estimate). With 512 anchors × 7 positions per 4,096-token sequence, training costs about 1.4e10 FLOPs per token: backbone 6 × 1.6e9 × 3,584, context path 6 × 0.18e9 × 4,096, frozen lm_head forward plus input-grad 4 × 1.27e9 × 3,584. A 50M-token epoch is about 7e17 FLOPs, roughly 0.65 H100-hours at an assumed 300 TFLOP/s effective. DeepSpec's 10 epochs × 1.24B tokens is about 160 H100-hours.
- **Memory.** Full Adam on 1.92B params is about 31 GB plus the frozen embed and lm_head. That does not fit a 3090 but fits one 80 GB GPU **(inferred)**. Training reads only the cache; the target is loaded on CPU just to copy the frozen embed and lm_head (`DeepSpec/deepspec/trainer/base_trainer.py:250-279`).
- **Data.** Regenerate responses with the served W4A16 target, as DeepSpec does with its own target (`DeepSpec/scripts/data/README.md:47-87`; non-thinking, temperature 0.7, 4,096 max tokens at :78-83). HyperQwen's draft vocab gained 10% single-stream from counting the model's own outputs (`docs/optimizations.md:84-89`). `drafter/gen_data.py` makes 5.4M tokens in 2.2 h on the 3090 (`drafter/README.md:45`), so 50M tokens is about 20 h (estimate).
- **Hidden states.** DeepSpec's `prepare_target_cache.py` hooks an HF bf16 `AutoModel` (around :253-258): 54 GB of weights for 27B, so not on the 3090. Capture instead through HyperQwen's in-process vLLM hook on the W4A16 target (`drafter/capture.py`; 74 GB in 1.7 h, `drafter/README.md:46`), extended from 1 layer to 5 taps plus the last hidden. The W4A16-vs-bf16 hidden mismatch is probably minor **(inferred)**: decode-vs-prefill hidden states differ by about 0.9% and training on either gives the same drafter (`drafter/README.md:111`).
- **Blocker.** DeepSpec trains DFlash and DSpark, **not DFlash2**: no grouped convs, no candidate selector, and the selector's training loss isn't public. Fine-tuning the shipped drafter means first writing the DFlash2 training forward from `qwen3_dflash2.py`.
- **Would it help where the MTP fine-tune didn't?** Possibly; unproven.
  - The MTP fine-tune left top-1 on response tokens at 0.685 → 0.685 (`drafter/README.md:29-37`). That was Qwen's jointly trained head on Qwen's own distribution.
  - DFlash2 is third party, trained on unknown data, then int4-quantized, and it has one documented gap: int4 loses about 5% at default sampling, 3.2 vs 3.4 (`drafter/README.md:90`).
  - DeepSpec recommends re-fine-tuning per domain, "especially if the target model is expected to run in thinking mode" (`DeepSpec/README.md:65`), yet regenerates with `--disable-thinking` (`DeepSpec/scripts/data/README.md:83`). 45% of HyperQwen's drafter prompts have thinking on (`drafter/README.md:43-44`).

### Q3. Is DFlash2's long-context weakness a training problem?

Mostly not. Every DeepSpec config uses `max_length=4096` (`DeepSpec/config/dspark/dspark_qwen3_8b.py`) with full-attention draft layers (`DeepSpec/deepspec/modeling/dspark/qwen3/config.py:15`). DeepInfer's drafter is sliding-window and asserts its window equals the main model's (`DeepInfer/.../mtp/dflash.py:139-146`). There's no long-context drafter recipe to borrow.

Check c2 decomposes the measured 112k table (`docs/long-context.md:199-207`) with step = (tok/step) / (tok/s):

| Arm | Task | ms/step |
|---|---|---|
| DFlash2 | reproduce (lookup fills 16) | 91.7 |
| DFlash2 | list commands | 76.2 |
| DFlash2 | quote / summarize / answer (verify block 8) | 62.4–62.7 |
| MTP | quote / summarize / answer | 37.5–37.8 |

The lookup patch alternates the verify block between 8 and 16 (`patches/dflash2-lookup-drafting.patch:898-997`, default on at `patches/speed-knobs-envs.patch:45`). Attributing the whole reproduce-vs-QA gap to those 8 extra tokens gives **about 3.6 ms per verify token and a 33.5 ms fixed step at 112k (inferred)**. The patch's own comment agrees in shape: an 8 → 16 block costs +6% per step at 1.5k context and +27% at 25k (`patches/dflash2-lookup-drafting.patch:958-966`).

**Lead caveat on the 3.6 ms slope.** It is a two-point difference that loads every other difference between reproduce and QA steps (lookup work, more accepted tokens written) onto verify width, and it was measured on the Triton int8 arm (`docs/long-context.md:194-197`), not the current FA2-fp8 default. Treat it as an upper bound. The spec lane's bracket for pthfa at 128k is about 1.2 ms per position ([spec-decode.md](spec-decode.md), scaled from a 3080 measurement). The two disagree by 3×; runtime-2's `step_curve` ([runtime.md](runtime.md)) settles it.

DFlash2's 2:1 loss on QA-type work at 112k then splits into: about 4 extra verify tokens versus MTP (≤ 15 ms); a larger fixed part (Triton backend, the drafter's 5 layers, the int8 padded-stride penalty at `docs/long-context.md:216-221`); and lower acceptance, 2.0–2.2 vs 2.3–2.75 tok/step. A long-context retrain only touches the third. Even at the best fixed verify width (3–4), DFlash2 QA reaches about 41 tok/s against MTP's 62–73 (c2). **D stays on MTP.**

### Q4. Draft vocab, confidence head and other tricks

- **Draft vocab.** Neither DeepSpec nor DeepInfer reduces it; both use full-vocab frozen embeddings and lm_head (`DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py:270-283`). HyperQwen's 40,960-row own-output vocab is ahead of both. Applying it to DFlash2's candidate top-k doesn't pay (check c1): DFlash2 reads the lm_head once per step, so the slice saves at least 0.58 ms (2.2% of a 26.5 ms step), while the ~2.5% out-of-vocab target tokens become forced rejections costing 0–4.7% of tau. Net −2.6% to +2.2%.
- **What the confidence head predicts.** The per-position probability that the draft token is accepted, `sigmoid(w·[h_k ; W1[prev_k]] + b)`, trained with BCE against the soft label 1 − ½‖p_draft − p_target‖₁ (`DeepSpec/deepspec/modeling/dspark/loss.py:60-70, 152-163`). That label is exactly the speculative-sampling acceptance probability; check b gives Monte-Carlo 0.6426 vs 1 − TV 0.6417. At inference DeepSpec cuts the block at the first position below a threshold (`DeepSpec/deepspec/eval/dspark/draft_ops.py:82-93`). DeepInfer applies a per-step temperature (`DeepInfer/deepinfer/environ.py:129-143`), takes a cumprod and verifies the globally best prefix under a profiled cost curve (`DeepInfer/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py:11-80`).
- **The loss shape matters for quantized drafters.** DSpark puts 0.9 of the weight on TV distance and weights positions by exp(−k/4); position 6 gets 6% of the weight (check b). Sampled acceptance depends on TV; greedy acceptance depends only on the argmax. That is exactly the int4 DFlash2's symptom: greedy holds, sampling loses about 5% (`drafter/README.md:89-91`). See drafter-3.

## Candidates (ranked)

| ID | Title | Modes | Gain | Effort | Confidence |
|---|---|---|---|---|---|
| drafter-1 | Confidence-gated verify length for single-stream DFlash2 | `SPEC=dflash2 CTX=long`; E with `PREFIX_CACHE=0`; B near 64k | 112k: +4 to +20% (ceiling, linear-cost model); ≤25k: ~0 or negative | S (stage 0), M (stage 1) | Low-medium |
| drafter-2 | Offline drafter gate: tau_prob per domain | all DFlash2 modes (indirect) | turns ±5% A/B noise into a repeatable number | M | High (as a tool) |
| drafter-3 | TV-loss repair of the W4A16 DFlash2 | B, C, E at default sampling | ceiling +6%, realistic ~+3%; 0 at greedy | M | Medium |
| drafter-4 | Own-distribution and thinking-mode fine-tune of DFlash2 | B, C, E | unknown | L, plus rented 80 GB GPU | Low |

### drafter-1: confidence-gated verify length for single-stream DFlash2

**Measured (2026-10-22):** drafter-1a LOSS, +1.8–2.4% TOTAL at 100k in D′ (bar +8%); copy −4–6% from synchronous scheduling. drafter-1b skipped on its gate. [gpu-results/drafter-1a.md](gpu-results/drafter-1a.md)

- **DeepInfer and DeepSpec.** The confidence head (Q4; `DeepSpec/deepspec/modeling/dspark/common.py:43-49`, `loss.py:152-163`, `DeepInfer/.../mtp/dflash.py:46-79`). DeepInfer verifies the cost-optimal prefix (`DeepInfer/doc/inference_deployment_qa.md:241-264`; `schedule_mtp_varlen.py:11-80`). Its numbers: verified length 5 → 3.2–4.4, accepted length 3.29 → 3.20, **−1.5% at 16 seqs/rank, +5.4% at 60**.
- **HyperQwen today.** The lookup patch already picks a per-step verify length on the host, `next_num_draft_tokens` (`patches/dflash2-lookup-drafting.patch:948-997`), from stale flags without a sync, with two choices: 8 (drafter block) or 16 (lookup tail). vLLM 0.31's adaptive verification and online estimator exist, and the DFlash2 speculator calls `_maybe_predict_acceptance` (`vllm@v0.31.0:vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:246-251`; patch :1053-1054), but that path is gated off for GDN ([runtime.md](runtime.md), runtime-1).
- **The gap.** Nothing ever verifies fewer than the drafter's 8 tokens, and nothing uses a confidence signal.
- **Correction to runtime-1.** runtime-1 originally said the gain is zero at N=1 because the best global prefix is the best uniform k. That holds for a budget fixed from stale rates. Trimming per step on fresh confidence is DeepInfer's idea applied over time instead of across requests, and it pays when verify cost per token is large, which is the long-context case. [runtime.md](runtime.md) is amended to match.
- **Mechanism.** A low-confidence step (QA, summarizing; survival about 0.53 per position) wastes most of an 8-token block. Keep draft k while `cumprod(conf)_k > c × (tokens per ms)`, the cost-aware threshold.
- **Gain** (estimate, checks c3–c5, linear verify cost). Calibrated = noise 0 / 0.5 on the logit; noisy = 1.5:

  | Context | Slope used | Calibrated | Noisy |
  |---|---|---|---|
  | ≤25k | 0.89 ms/token | +0.7% / −0.5% | −8% |
  | ~64k | 2.0 ms/token (interpolated) | +4.8% / +3.8% | −3.8% |
  | 112k, chat-like | 3.6 ms/token | +11% | +4% |
  | 112k, QA only | 3.6 ms/token | +20% | +14% |
  | 112k, QA plus reproduce mix | 3.6 ms/token | +12–13% | +10% |

  By mode: A 0 (no speculation). B about 0 for typical chats, up to +4% near 64k. C 0 at short context. D unchanged: DFlash2 QA rises from about 34 to 40 tok/s, still below MTP's 62–73. E probably more than the 112k rows, because the slope grows with context, but blocked (risk below).
- **Lead caveats.**
  - Every row uses the 3.6 ms slope, which is an upper bound (Q3). If the real slope is nearer the spec lane's 1.2 ms, the 112k rows shrink toward the ~64k row.
  - Real costs come in stairs, not a line ([spec-decode.md](spec-decode.md), [attention-kv.md](attention-kv.md)). Marlin is flat up to 16 query tokens. On pthfa, T=6–10 share one padded 64-row tile, so a trim from 8 to 6 or 7 saves no attention; only T ≤ 5 (k ≤ 4) does. Inside the stair only the GDN state writes shrink (about 0.08 ms per position). The KVarN Triton kernel's tiling, which E uses, hasn't been read.
  - **Overlaps spec-2** ([spec-decode.md](spec-decode.md)), which computes the same trim on the device from fresh survival inside a fixed-width graph. Treat them as one feature: spec-2 owns the mechanism, drafter-1 stage 1 supplies the confidence signal.
- **Effort.** S for stage 0, M for stage 1.
  - **Stage 0 (no training).** Extend `next_num_draft_tokens` to return 2–8. Use the previous step's emitted count (`last_num_emitted`, already tracked) or DFlash2's own walk probabilities (`candidate_sampler.scores`, which vLLM's estimator already reads). The hook already has a context gate (`draft_max_seq_len > CHEAP_CTX`).
  - **Stage 1.** A DSpark-style head, `Linear(5120 → 1)` with 5,121 params, on DFlash2's final hidden state, trained with BCE on 1 − TV, plus DeepInfer's per-position temperature. It trains on the 3090 from captured drafter hidden states and target top-k logits.
- **Risk.**
  - Lossless: only the number of verified drafts changes.
  - **Blocked in E today.** Any step-to-step change in verify length corrupts a prefix-cache hit on KVarN (`single-user/start_qwen.sh:399-428`, measured 38/793 wrong); DYNAMIC_K is refused there for the same reason (`:676-692`). Realizable today only at `SPEC=dflash2 CTX=long` or E with `PREFIX_CACHE=0`.
  - Greedy outputs can shift across verify widths: HyperQwen notes a ±3% greedy spread from rounding (`drafter/README.md:116-120`). Single request only, like STICKY (patch :921-935).
  - A noisy predictor goes negative at short context (−8% at ≤25k), so gate it on context length.
- **Porting plan.**
  - Stage 0: `next_num_draft_tokens` in `patches/dflash2-lookup-drafting.patch`; envs `VLLM_DFLASH2_CONF_TRIM` and `VLLM_DFLASH2_VERIFY_MS_PER_TOKEN` in `patches/speed-knobs-envs.patch`; a launcher knob in `single-user/start_qwen.sh`, refused at `CTX=huge` with prefix caching, as DYNAMIC_K is.
  - Stage 1: new `drafter/train_conf_head.py`, reusing `drafter/capture_dflash2.py` to dump the drafter's last hidden state and selector distribution and `drafter/capture.py` for the target's top-k probabilities. Ship the weights as an extra tensor beside the drafter.
- **GPU validation.** runtime-2's `bench/step_curve.py` at 25k, 64k and 112k first, to replace the inferred slope and show the stairs. Then `bench/labd_bench.py --ctx 100000` with `SPEC=dflash2 CTX=long`, trim off vs on. Win: ≥ +8% on the all-six tok/s, unchanged greedy output hash on a C1 run, and no regression above 2% at 8k on the `docs/optimizations.md` 8-prompt protocol.

### drafter-2: offline drafter gate (DeepSpec's eval metrics)

- **DeepSpec.** Per-position `accept_rate = 1 − TV(p_draft, p_target)` and the analytic `tau_prob = 1 + Σ_k Π_{j≤k} accept_rate_j` (`DeepSpec/deepspec/modeling/dspark/loss.py:46-70`), plus acceptance length and accept-rate@position (`DeepSpec/deepspec/eval/base_evaluator.py`).
- **HyperQwen today.** Drafter decisions are made on 8 prompts × 1k tokens. The spread is ±3% greedy, wider for DFlash2 (3.1–3.6 tok/step across configs), and ±5% per run at default sampling (`drafter/README.md:116-120`). The context-KV Hessian blend nearly shipped at 7% worse because one default-sampling run looked equal (`drafter/README.md:95-99`). `train_mtp.py --eval-only` has a greedy chain simulation, but for MTP only (`drafter/README.md:36-37`).
- **The gap.** No expected-acceptance metric for DFlash2 at default sampling, which is where the int4 loss lives.
- **Mechanism.** Replay captured target hidden states (taps plus last-hidden top-k) through the drafter offline; compute the selector's per-position q against the target's p; report tau_prob (sampling) and argmax-match tau (greedy) per domain (thinking vs non-thinking, code, Danish), on response tokens after an anchor. It's deterministic, so a 1% difference is visible.
- **Gain.** Indirect. It turns every drafter or quant A/B (drafter-3, quant-3's drafter requant, any fine-tune) from ±5% noise into a repeatable number, and shows whether drafter-4 has headroom before money is spent. Complements spec-5's Σ min(p, q) serving telemetry.
- **Effort.** M. The DFlash2 forward runs outside vLLM: copy `qwen3_dflash2.py`, which needs only torch and the grouped conv's CPU path (`qwen3_dflash2.py:144-170`). Inputs come from extending `drafter/capture_dflash2.py`.
- **Risk.** None to serving. Validate the metric once against vLLM's measured acceptance, as `train_mtp.py --eval-only` was.
- **Porting plan.** New `drafter/eval_dflash2.py`: loads bf16 drafter weights or the W4A16 ones dequantized through `drafter/gptq_utils.py`, reads a capture directory (5 tap rows plus target top-64 logits per response token), prints tau_prob, accept@position and greedy tau per domain. Add a `--dump-target` mode to `capture_dflash2.py`.
- **GPU validation.** Run on the bf16 and W4A16 drafters. Win: reproduces the known 3.4 vs 3.2 sampling gap and the 3.54 vs 3.34–3.65 greedy parity (`drafter/README.md:88-91`) within ±2%, with repeat-to-repeat spread below 0.5%.

### drafter-3: TV-loss repair of the W4A16 DFlash2

- **DeepSpec.** The TV-dominant loss with position decay, 0.9·L1 + 0.1·CE, γ = 4 (`DeepSpec/deepspec/modeling/dspark/loss.py:25-37, 73-87, 248-252`; `DeepSpec/config/dspark/dspark_qwen3_8b.py`).
- **HyperQwen today.** GPTQ int4 drafter (`drafter/quant_dflash2.py:1-30`). It keeps greedy acceptance but loses about 5% at default sampling: "noise in q hurts the acceptance *probability*, not the argmax" (`drafter/README.md:88-91`). A bf16 `fc` didn't help (`:94`); the context-KV Hessian blend was worse (`:95-99`).
- **The gap.** GPTQ minimizes weight-space output error. Nothing optimizes the drafter's distribution against what sampled acceptance depends on.
- **Mechanism.** Freeze the int4 linears (they stay Marlin-compatible). Train only the parameters already in bf16 (`quant_dflash2.py:9-11`): the convs' `kernel_projection` and base kernels, the selector codebooks and `hidden_projection`, and the norms, about 190M params (check a). Loss: TV against the **bf16 drafter's** selector distribution as teacher, exp(−k/4) position weights. The ceiling is bf16 parity, and no target-logit cache is needed because the teacher sees the same tapped inputs. Memory: 1 GB int4 + 3.85 GB teacher + 190M × 16 B ≈ 3 GB Adam state, plus activations; fits the 3090 **(inferred)**.
- **Gain** (estimate). The ceiling is the documented gap, 3.2 → 3.4 tok/step at default sampling (+6% tau, step unchanged): B +6%, C +6% on non-lookup steps, E similar on non-lookup steps, A and D 0, greedy 0. Realistic: half the gap, about +3% **(inferred)**. Pairs with quant-3 ([quant-heads.md](quant-heads.md)), which shrinks the same gap from the GPTQ side; measure both with drafter-2.
- **Effort.** M. Needs the DFlash2 training forward, including a differentiable selector walk.
- **Risk.** Lossless (drafter only). The MTP null result is a weak precedent here, because the target is a measured gap, not hoped-for headroom. If the int4 noise sits in the frozen linears, the repair can't close it; then hand back to the quant lane (int8 for the most sensitive layers).
- **Porting plan.** New `drafter/repair_dflash2.py`, sharing modules with `eval_dflash2.py`. Data from `capture_dflash2.py`, which already dumps `fc` input rows (`drafter/README.md:75-80`). Output a patched W4A16 checkpoint with only the bf16 tensors changed; vLLM already loads those as bf16, so no loader change.
- **GPU validation.** drafter-2's tau_prob first: it should move from about 3.2 toward 3.4. Then the `docs/optimizations.md` 8-prompt protocol at default sampling, three repeats. Win: ≥ +3% tok/s at default sampling, greedy unchanged within ±2%.

### drafter-4: own-distribution and thinking-mode fine-tune of DFlash2

- **DeepSpec.** Target-regenerated data, the target cache and the DSpark loss, 10 epochs over perfectblend (`DeepSpec/scripts/data/README.md:47-121`). The README recommends re-fine-tuning for thinking mode (`DeepSpec/README.md:65`).
- **HyperQwen today.** DFlash2 is used as shipped (`prepare/fetch_dflash2.py`). An own-output corpus exists: 5.4M tokens, 45% thinking (`drafter/README.md:41-46`).
- **The gap.** DFlash2 has never been adapted to this target's quantized outputs or to thinking-mode text.
- **Gain.** Unknown. Don't commit before drafter-2 shows a per-domain gap, for example thinking-mode accept@position well below non-thinking. evals-4 ([evals.md](evals.md)) measures the serving side of the same question.
- **Effort.** L: the DFlash2 training forward (shared with drafter-3), a 5-tap capture on the W4A16 target, a 0.4–3 TB cache, about 0.65 H100-hours per 50M-token epoch and about 20 h of 3090 generation (Q2).
- **Risk.** Lossless. A repeat of the MTP null result is a real possibility. DFlash2's selector objective is unknown.
- **Porting plan.** `drafter/capture.py`: add `--taps 5,19,33,47,61 --topk-logits 64`. New `drafter/train_dflash2.py` with the DSpark loss terms copied from `DeepSpec/deepspec/modeling/dspark/loss.py`. Re-run `drafter/quant_dflash2.py` afterwards.
- **GPU validation.** drafter-2 per domain, then the 8-prompt protocol plus `bench/labd_bench.py`. Win: ≥ +5% tok/step at default sampling in B.

## Already in HyperQwen

- DSpark serving, a loader fix and a measured comparison: `patches/dspark-draft-quant-config.patch`; `docs/long-context.md:619-631`. DSpark lost.
- A reduced draft vocab counted over the model's own outputs (MTP): `prepare/build_draft_vocab.py`, `docs/optimizations.md:76-89`. Neither DeepSpec nor DeepInfer has one.
- A host-side adaptive verify length (rule-based 8 ↔ 16 with STICKY): `patches/dflash2-lookup-drafting.patch:898-997`. DYNAMIC_K per batch size: `single-user/start_qwen.sh:676-692`.
- vLLM 0.31's DSpark model with Markov and confidence heads, adaptive verification with a cost curve, and `OnlineAcceptanceEstimator` (logit(max q), IRLS refit every 100 steps): `vllm@v0.31.0:vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` and `acceptance_estimator.py:313-436`. In the pinned tree, gated off for GDN.
- Drafter context K/V computed from projected target features in one fused precompute, the analogue of DeepInfer's `update_main_cache` and merged `main_kv_proj` (`DeepInfer/.../mtp/dflash.py:118-123, 198-228`). Calibrated in `drafter/capture_dflash2.py:127` (`_project_context_kv`; `drafter/README.md:78-80`).
- Probabilistic draft sampling, with drafts truncated to the request's top-k/top-p (`drafter/README.md:100-104`).
- Target-regenerated data (`drafter/gen_data.py`) and hidden-state capture through vLLM (`drafter/capture.py`), the same idea as DeepSpec's pipeline.

## Rejected

- **DSpark as a replacement drafter.** Measured on this 3090: 3.37 vs 3.83 tok/step, 114 vs 150 tok/s (`docs/long-context.md:628-631`).
- **Grafting DSpark's Markov head onto DFlash2.** The selector already conditions on the predecessor with hidden modulation, and W2 would add about 0.95 ms per step in bf16 (check a).
- **EAGLE3.** No Qwen3.8-27B checkpoint, and a single-layer chain is MTP's niche, where Qwen's MTP is at the ceiling (`drafter/README.md:29-37`).
- **The 40,960-row draft-vocab slice for DFlash2's candidate top-k.** Net −2.6% to +2.2% (check c1).
- **A long-context retrain to rescue mode D.** DeepSpec only has a 4,096-token recipe, and even perfect trimming leaves DFlash2 QA at about 41 vs MTP's 62–73 tok/s (check c2).
- **Training a 27B drafter from scratch at DeepSpec scale.** 76 TB of cache and about 160 H100-hours, and the measured third-party DSpark already lost.
- **Training any full drafter on the 3090.** Adam state alone is about 31 GB.
- **fp8 target cache.** DeepSpec asserts bf16 (`DeepSpec/deepspec/data/target_cache_dataset.py:24, 154`). Only matters if drafter-4 goes ahead.
- **`DFlashNextPredictHeadHC` and `MarkovGPTHead`.** V4 hyper-connection and MLA heads (`DeepInfer/deepinfer/model_executor/models/v4_modules/heads.py:26-107`).
- **The TileLang multi-sequence varlen kernel and DeepInfer's async budget split.** Multi-sequence only; covered by runtime-1 and spec-1.

## CPU checks

One script, `cpu-checks/drafter_checks.py`, about 1 minute. It needs torch, which on the research box lives in the user site-packages, so run it from `docs/research/deepinfer/cpu-checks/` with `python3 -E -P drafter_checks.py`. The DSpark heads and loss in section b are transcribed from `DeepSpec/deepspec/modeling/dspark/*`, not imported. The lead re-ran it from the repo copy; `drafter_checks.out` matches the lane's output byte for byte.

| Section | What it computes | Result |
|---|---|---|
| a | Parameter counts and bytes per step | Selector and convs ≈ 190M; DFlash2 ≥ 1.76 ms/step; Markov W2 +0.95 ms/step bf16 |
| a′ | DeepSpec cache bytes for 27B | 61,440 B/token; 3.1 TB per 50M tokens |
| b | Toy DSpark Markov and confidence heads plus loss | MC acceptance 0.6426 vs 1 − TV 0.6417; BCE fixed point exact; position 6 gets 6% of the weight |
| c1 | Draft-vocab slice on DFlash2 | Net −2.6% to +2.2% |
| c2 | 112k step anatomy | 3.64 ms/verify token, 33.5 ms fixed (inferred); best fixed DFlash2 QA ≈ 40.7 tok/s |
| c3 | Per-request truncation, noisy confidence, QA and reproduce mix | +8 to +15% (QA), +10 to +12% (mix) |
| c4 | Chat-like, cost-aware threshold at three slopes | 25k: +0.7 / −0.5 / −8%; 64k: +4.8 / +3.8 / −3.8%; 112k: +11.4 / +11.4 / +4.0% |
| c5 | Cost-aware threshold on the c3 mixes | QA +20 / +20 / +14%; mix +13 / +12 / +10% |

All c-sections assume a linear verify cost; see the stair caveats under drafter-1.

## Open questions for the user

1. Do you want DFlash2 at 64k+ (`CTX=long`, or E)? drafter-1 pays there and almost nowhere else, and in E it first needs the KVarN variable-length bug root-caused (`single-user/start_qwen.sh:399-428`).
2. Is the 3.6 ms per verify token at 112k still true on FA2-fp8? One run of runtime-2's `step_curve` settles it.
3. How much real traffic runs in thinking mode? That decides whether drafter-4 is worth a look.
4. Would you rent one 80 GB GPU for a few hours (drafter-4)? drafter-1 to drafter-3 need only the 3090.
5. Does incoai publish DFlash2's training data or selector loss? Without them, drafter-3 and drafter-4 have to reconstruct the selector training forward.
