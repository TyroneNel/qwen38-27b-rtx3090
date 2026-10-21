# DeepInfer expedition

What HyperQwen can take from deepseek-ai/DeepInfer, its submodules and its sibling DeepSpec, ranked by return on effort. Research and documentation only: nothing here changes serving. The research box has no GPU, so every gain below is an estimate, a simulation or a number quoted from HyperQwen's own docs. The GPU test plan at the end turns them into measurements.

Six lanes wrote the appendices; the lead verified citations, re-ran every CPU check from the committed copies and corrected numbers where needed (listed under [Lead corrections](#lead-corrections)).

| Appendix | Covers |
|---|---|
| [runtime.md](runtime.md) | Scheduler, CUDA graphs, determinism, structured outputs |
| [attention-kv.md](attention-kv.md) | Verify attention, KV formats, sparse attention |
| [spec-decode.md](spec-decode.md) | Verify-length policy, adaptive verification, GDN state |
| [drafter.md](drafter.md) | DSpark vs DFlash2, DeepSpec training, confidence head |
| [quant-heads.md](quant-heads.md) | Activation scaling, int4 GPTQ, heads, logits |
| [evals.md](evals.md) | Quality gates, sync checks, thinking-mode coverage |

## Executive summary

1. **DeepInfer is built for H100/B200 MoE serving: FP8/FP4, MLA, data-parallel lockstep.** None of its kernels run on sm86. What transfers is method: cycle budgeting for attention, confidence-scheduled verify length, fine-grained activation scaling, a decode-path KL gate, and DeepSpec's TV loss and confidence head.
2. **There is no runtime overhead to win.** Host time is under 2% of a step, and mode A's 942 tok/s is fully explained by arithmetic (runtime).
3. **DFlash2 stays.** It beat DSpark on this 3090 (3.83 vs 3.37 tok/step, 150 vs 114 tok/s). Its improvements come from around it, not from replacing it:
   - **Verify length.** vLLM 0.31 already ships DeepInfer's confidence-scheduled verify. It is off for this model only because GDN refuses device-side query lengths, and upstream's KDA opt-in shows the unblock is small (spec-1, S patch). It pays at a few concurrent streams (C4 +6–7% over `DYNAMIC_K`). At one stream it pays only at long context, and only if the trim uses *fresh* confidence (spec-2, drafter-1).
   - **The int4 drafter's sampling gap.** The W4A16 DFlash2 keeps greedy acceptance but loses about 5% at default sampling. Better GPTQ (quant-3) and a TV-loss repair of its bf16 parameters (drafter-3) both target it; an offline tau_prob gate (drafter-2) measures it without ±5% run noise.
4. **Deep-context verify is bound by the tensor pipe, not DRAM.** Cheaper MMAs save about 4.6 ms per step at 150k in D′ (attn-1, inferred; ncu first). The first two stages, removing row padding at T=8 (lossless) and accumulating PV in fp16 (near-lossless), get most of it at boost clock. Moving KVarN's verify onto the same CUDA kernel is the largest item for E: about +21% at 240k (attn-2, inferred, needs a profile first).
5. **Mode A's lever is activation scaling.** Per-token × per-128 int8 activation scales cut activation error 4–7× on heavy-tailed inputs; vLLM 0.31 can already serve it through the Humming kernels behind one flag. If quality allows int8 on all layers: A up to +10.6%, single-user prefill +27–30% (quant-1).
6. **Mode D stays on MTP.** DFlash2's long-context loss is verify cost and a larger fixed step; even perfect trimming leaves it at about 41 vs MTP's 62–73 tok/s at 112k (drafter Q3).
7. **The biggest unknown behind the headline numbers is thinking mode.** Every speed and acceptance number in the repo is measured with thinking off, while the chat template defaults to xhigh effort (evals-4).
8. **Two things to check cheaply first.** The negative-scale fold in `marlin-int8-negative-scales` may drop an LSB on the largest weight of about half of all groups, not "extremely rarely"; one CPU script on the real checkpoint settles it (quant-2). And there is no recorded step-cost curve, which is the single number most of the verify-length items depend on (runtime-2).

## Roadmap, ranked by ROI

Modes: A batch; B DFlash2 64k; C quoting (T=16); D `SPEC=mtp CTX=long`; D′ `SPEC=dflash2 CTX=long PTH_FA=1`; E `SPEC=dflash2 CTX=huge` (KVarN). "Lossless" means output is unchanged in distribution.

### Do first: measurement that gates everything else

| # | ID | What | Effort | Why first |
|---|---|---|---|---|
| 1 | quant-2 (check) | `quant_real_ckpt_check.py` on the real AutoRound checkpoint | minutes, CPU | Confirms or closes quant-2 |
| 2 | runtime-2 | `bench/step_curve.py`: ms/step vs verified tokens per backend and depth | S | Settles the verify cost per token (3.6 vs 1.2 ms at 112–128k) and the stairs; spec-0/1/2, drafter-1 and runtime-1 all hinge on it |
| 3 | evals-2 | Boot every mode with `VLLM_GPU_SYNC_CHECK=warn` | S | Catches hot-path syncs; already in the pin |
| 4 | evals-1 | Decode-vs-prefill logprob KL probe per mode | S (4–16k) | The quality gate for every lossy item below |
| 5 | evals-4 | Thinking-on speed and acceptance cohort | S | Tests the headline numbers |
| 6 | evals-3 | Paired quality battery (per-item JSONL, McNemar/Wilson) | S | Makes 1-point quality changes visible |
| 7 | spec-5 | Σ min(p, q) expected-acceptance telemetry | S | Lower-variance acceptance A/Bs |

### Speed and capacity

| Rank | ID | What | Modes | Gain | Effort | Conf. | Lossless |
|---|---|---|---|---|---|---|---|
| 1 | quant-1 | Per-token × per-128 int8 activation scales (Humming route first) | A; prefill B–E | A up to +10.6%; prefill +27–30% | M | Medium | No (quality gate) |
| 2 | spec-1 = runtime-1 | Opt GDN into vLLM's adaptive verification (mirror #52988) | C at 2–4 streams; E | C4 +6–7% over `DYNAMIC_K`; E +0.7 to +15% over k=7 | S patch, M validation | Medium | Yes |
| 3 | attn-1 | pthfa verify: unpadded 48-row tiles at T=8, PV fp16 accumulate, then int8 QK | D′ (E via attn-2) | −4.6 ms/step at 150k; stages 1–2 alone reach the DRAM floor at boost clock | M | Medium | Stage 1 yes; 2 near; 3 no |
| 4 | quant-7 | int4 embedding where `EMBED_UVA` can't be used | D, E | KV pool +8.4% | S | High | No (small) |
| 5 | spec-0 | Context-aware fixed `DFLASH_TOKENS` at 240k | E | −2.5 to +12.5% | S | Medium-low | Yes |
| 6 | quant-3 | Better GPTQ for HyperQwen's own int4 requants | B, C, E (drafter); D (MTP) | DFlash2 +2–3% at default sampling | S | Medium | Drafter yes |
| 7 | attn-2 | KVarN verify on a pthfa-family CUDA kernel | E | ~+21% at 240k, ~+2% at 20k | L | Medium-low | Yes |
| 8 | drafter-3 | TV-loss repair of the W4A16 DFlash2's bf16 params | B, C, E | ~+3% (ceiling +6%) at default sampling | M | Medium | Yes |
| 9 | spec-2 + drafter-1 | Per-step verify trim on fresh confidence | D′/`CTX=long`, E | E +4 to +10%; 112k +4 to +20% (linear-cost ceiling) | S (host stage 0) to M | Low-medium | Yes |
| 10 | quant-5 | Batch-mode lm_head as W4A8 | A | +2–3% | S–M | Low-medium | No |
| 11 | attn-3 | int8-QK prefill on quantized-KV arms | D, E | cold TTFT −8 to −9% at 150k | M | Medium | No |
| 12 | spec-3 | RecoverSSM for GDN | DFlash2 residency | residents ~6 → 20+; `CTX=long` reach 114k → 132k | L | Medium | Yes |
| 13 | attn-4 | Training-free top-k sparse attention | E | up to +30% at 240k | L | Low | No |
| 14 | quant-4 | Block-Hadamard-128 rotation of linear inputs | as quant-1 | as quant-1, on unchanged Marlin | L | Medium | No |
| 15 | spec-4 | Lookup-aware confidences in the verify budget | C | C1 0 to +3% | M | Low | Yes |
| 16 | quant-6 | Fuse the int8 activation quant into its producer | prefill | ~0.5% of a 4k prefill | M | Medium | Yes |
| 17 | drafter-4 | Own-distribution, thinking-mode DFlash2 fine-tune | B, C, E | unknown | L + 80 GB rental | Low | Yes |

### Quality, robustness and tooling

| ID | What | Effort | Notes |
|---|---|---|---|
| quant-2 | Lossless negative-scale fold (load-time remap onto the existing `kS8 × kU4` kernel) | S | Only if check 1 confirms it |
| drafter-2 | Offline DFlash2 gate: tau_prob and accept@position per domain | M | Required before drafter-3 or drafter-4 |
| attn-5 | Hadamard-rotated int8/int4 per-token-head KV | S–M | Quality for D; int4 would halve D's DRAM floor |
| runtime-3 | Bound the tail of forced tool calls | S–M | 5–10% of forced calls run to `max_tokens` |
| evals-5 to evals-11 | Special-token hardening, streamed tool calls, golden canary, per-component GPU time, broader acceptance panel, Pareto sweep, warmup detection | S–M each | See [evals.md](evals.md) |

## Threads that cross lanes

- **Verify length is one feature in four pieces.** runtime-2 measures the cost → spec-0 picks a better fixed width at E → spec-1 unblocks vLLM's adaptive verification on GDN → spec-2 (device mechanism) plus drafter-1 (confidence signal) trim per step. Two open issues shape all of it:
  - The cost per verify token is uncertain by 3×: drafter's 3.6 ms at 112k comes from the Triton int8 arm and is an upper bound; spec's ~1.2 ms at 128k is scaled from a 3080. Costs also come in stairs: Marlin is flat to 16 query tokens, and pthfa is flat from T=6 to T=10, so trimming 8 → 6 saves no attention there.
    - **Measured (2026-10-21, runtime-2):** the answer depends on the backend. FA2-fp8 is flat (~0.03 ms per extra verify token at 64–150k). pthfa costs ~1.2 ms per token for q 1→4, then a 4→8 stair of 1.9 ms per token at 112k and 2.6 at 150k (trimming 8 → 4 saves 18–21% of the step; 8 → 6 saves nothing). KVarN is linear and steep: 5.8, 10.3, 13.9 and 22.3 ms per token at 64k, 112k, 150k and 240k (8 → 4 is −41% of the step at 240k). Serving pads verify width to 1, 2, 4 or 8. [gpu-results/runtime-2.md](gpu-results/runtime-2.md)
  - **Blocker for E:** any step-to-step change in verify length corrupts a KVarN prefix-cache hit (`single-user/start_qwen.sh:399-428`, 38/793 wrong). Root-causing it unblocks spec-1, spec-2, spec-4 and drafter-1 in E.
- **The int4 DFlash2 sampling gap** (3.2 vs 3.4 tok/step) is attacked from two sides, quant-3 and drafter-3, and should be measured with drafter-2 and spec-5 rather than 8-prompt runs.
- **Hadamard rotation helps integer grids and hurts float grids.** attn-5 and quant-4 reached this independently: Hadamard plus int8/int4 beats NVFP4 or MXFP8, which have no MMA on sm86 anyway.
- **Thinking mode** is unmeasured for speed (evals-4) and for drafter acceptance (drafter-4); 45% of the drafter's own training prompts are thinking-on.
- **Batch speculation** (A with a drafter) is blocked by KV residency, not policy; it needs spec-3's RecoverSSM first.
- **Measurement noise** (±3% greedy, ±5% sampled per run) hides most single-digit wins; evals-3, spec-5 and drafter-2 exist to fix that.

## GPU test plan

In order; each step names the items it decides. Commands are in the appendices.

1. **Real-checkpoint fold check** (CPU only, minutes): `cd docs/research/deepinfer/cpu-checks && python3 -E -P quant_real_ckpt_check.py <path to Qwen3.8-27B-W4A16-AutoRound>`. Decides quant-2.
2. **Step-cost curve** (runtime-2): write `bench/step_curve.py`; ms/step vs total query tokens 1–32 under each launcher with CUDA graphs on, at 8k, 25k, 64k, 112k, 150k and 240k, on FA2-fp8, pthfa and KVarN Triton. Decides spec-0, spec-1, spec-2, drafter-1 and runtime-1.
3. **Sync check** (evals-2): boot all modes with `VLLM_GPU_SYNC_CHECK=warn`; record the allowlist.
4. **Decode KL probe** (evals-1) at 4–16k per mode, two runs for spread. Becomes the gate for steps 6, 7 and 9.
5. **Thinking-on cohort** (evals-4): `labd_bench` and `labd_accept` with `enable_thinking: True`, split by reasoning vs answer phase.
6. **Activation error per layer** (quant-1 step 1): extend `bench/act_calib.py` with per-128 and Hadamard-128 modes using the quantizers in `quant_a_act_quant.py`.
7. **Humming probe** (quant-1 route A): is `humming` importable, does int8 `DynamicGroup` run on sm86, speed vs Marlin with `--linear-backend humming`, and does it capture the int8 lm_head and embedding correctly.
8. **ncu on pthfa** at 150k, T=8, on a 3090: tensor-pipe vs DRAM utilisation. Decides attn-1.
9. **E profile at 200k** (torch profile): the real attention share per step. Decides attn-2 and spec-0's sign.
10. **spec-1 boot and greedy bit-identity** on `bench/labd_accept.py`, then `bench/conc_ladder.py` at C1, C2, C4 against `DYNAMIC_K`. Also answers whether FA2-fp8 accepts device q_len below the host bound.
11. **spec-0 ladder**: `DFLASH_TOKENS=3/5/7` at 90k, 150k and 240k, three boots per arm.
12. **Drafter work** once drafter-2 exists: quant-3 requant, then drafter-3, each measured by tau_prob before any tok/s run.

## CPU checks run on the research box

All in [cpu-checks/](cpu-checks/), run from that directory. Scripts that need torch use `python3 -E -P` because torch is in the user site-packages, which `-I` hides. The lead re-ran every script from its committed copy; outputs match the lanes' byte for byte.

| Script | Lane | Checks | Command |
|---|---|---|---|
| `runtime_topk_sim.py` | runtime | Global top-k verify vs best uniform k (N=1 is 0% by construction) | `python3 -I runtime_topk_sim.py` |
| `attn_roofline.py` | attention | Cycle budget and roofline per mode | `python3 -I attn_roofline.py` |
| `attn_kvq_error.py` | attention | KV formats: int8/int4, NVFP4, MXFP8, Hadamard | `python3 -I attn_kvq_error.py` |
| `attn_mma_precision.py` | attention | PV fp16 accumulate, int8 QK error | `python3 -I attn_mma_precision.py` |
| `spec_varlen_sim.py` | spec | Fixed k vs adaptive, stale vs fresh, per mode | `python3 -I spec_varlen_sim.py` (~60 s) |
| `spec_calib.py` | spec | Grid search for the acceptance parameters | `python3 -E spec_calib.py` |
| `quant_a_act_quant.py`, `quant_a2_gauss.py` | quant | Activation quant error by scheme | `python3 -E -P <script>` |
| `quant_b_signfold.py` | quant | Negative-scale fold cost and lossless remap | `python3 -E -P quant_b_signfold.py` |
| `quant_c_logits_bf16.py` | quant | bf16 vs fp32 logits KL | `python3 -E -P quant_c_logits_bf16.py` |
| `quant_d_gptq.py` | quant | GPTQ variants, importing `drafter/gptq_utils.py` | `python3 -E -P quant_d_gptq.py` |
| `quant_real_ckpt_check.py`, `quant_mk_fake_ckpt.py` | quant | Fold check for a real checkpoint (GPU-box step 1), with a fake checkpoint to self-test | `python3 -E -P quant_real_ckpt_check.py <dir>` |
| `drafter_checks.py` | drafter | Sizes, DeepSpec cache, DSpark heads and loss, vocab slice, 112k step anatomy, confidence trim | `python3 -E -P drafter_checks.py` |

## Lead corrections

Changes the lead made to lane findings before committing them:

- **E's DRAM floor** at 240k is 3.45 ms, and E's 67 tok/s was measured at `--ctx 20000`; every E gain grows with depth from there, and E's attention cost at 200k+ on a 3090 is unmeasured (attention-kv).
- **attn-1 stage 1** (unpadded 48-row tiles at T=8) is the lead's addition; with PV fp16 accumulate it reaches D′'s DRAM floor without a lossy step.
- **spec's linear cost model ignores the stairs**; on pthfa, k=7 → 5 saves no attention, k ≤ 4 does (spec-decode).
- **spec-1 is S, not M**, following #52988, with a caveat that KDA's metadata also carries `RecoverSSMMetadata`; validate GDN bit-identity first. runtime-1 is the same item and is amended to match.
- **runtime-1's "zero at N=1"** holds only for a budget fixed from stale rates; the simulation's N=1 result is 0% by construction (runtime, amended).
- **drafter-1's 3.6 ms slope** is an upper bound from the Triton int8 arm and conflicts with spec's ~1.2 ms; runtime-2 settles it (drafter).
- **quant-2's wording**: the patch already expects ~50% negative scales; the question is whether the clamp of code −8 is rare, and under the lane's reconstruction it isn't (quant-heads).

## Rejected, across lanes

- **DeepInfer kernels and formats:** FlashMLA, DeepGEMM, DeepSelect and TileLang paths are SM90/SM100 only. MXFP8 KV equals per-tensor e4m3 in precision on sm86; NVFP4 KV has no MMA here and no edge over KVarN's Hadamard-int4. Power-of-two scales and fp8 activations lose to int8 per-128.
- **Runtime:** porting DeepInfer's zero-overhead scheduler (0.7–1.7% host time to hide), prefill/decode separation, full batch invariance (GDN backends reject it, and Marlin's tiling varies).
- **Drafters:** DSpark as a replacement (lost on this card), grafting its Markov head onto DFlash2 (+0.95 ms/step), EAGLE3 (MTP's niche), the 40,960-row vocab slice for DFlash2 (−2.6 to +2.2%), a long-context retrain to rescue D, training a full drafter on the 3090.
- **Heads and numerics:** fp32 logits (KL 1–6e-4 vs the int4 head's 2.4e-3), W8A16 or bf16 DFlash2 drafter (+1.36 ms/step), per-channel int4 lm_head, sub-4-bit body.
- **Evals:** exact golden-decode perplexity as the main gate (needs batch invariance), AIME/GPQA/HLE per change (hours per run), LLM judges.
- **attn-6**, DeepInfer's balanced split scheduler: about 0 as HyperQwen is configured.
- Everything multi-node, MoE, EP, DP, PD-disaggregated or Ascend.

Each appendix lists its own rejections with reasons.

## Open questions for you

Decisions:
1. Do you want multi-seat speculation (DFlash2 at 2–16 concurrent users)? It decides how much spec-1/runtime-1 is worth beyond C4.
2. Do you want DFlash2 at 64k+ (`CTX=long` or E)? drafter-1 and spec-2 pay there and almost nowhere else.
3. How much real traffic runs thinking mode, and at what effort? It decides evals-4's priority and whether drafter-4 is worth a look.
4. Will you carry more GDN patches across vLLM pins (spec-1, spec-3)?
5. Would you rent one 80 GB GPU for a few hours (drafter-4 only)?
6. Does the forced-tool-call runaway hurt real agent traffic? If so, runtime-3 moves up.
7. Can DeepSpec's eval datasets be redistributed inside HyperQwen (evals-9)? Does incoai publish DFlash2's training data or selector loss (drafter-3, drafter-4)?

Facts the GPU box answers (the test plan covers them): the real cost per verify token and its stairs; Humming on sm86; ncu on pthfa; E's attention share at 200k; whether FA2-fp8 takes device query lengths; how well vLLM's online estimator fits DFlash2; how autocorrelated acceptance is step to step; the KVarN verify-length root cause; `prompt_logprobs` memory at 64–100k; how Qwen3.8's tokenizer treats a typed `<|im_end|>`.

## Method and sources

| Lane | Model, effort | Appendix |
|---|---|---|
| Engine runtime | Sonnet, high | [runtime.md](runtime.md) |
| Evaluation and tooling | Sonnet, medium | [evals.md](evals.md) |
| Attention and KV cache | Opus, high | [attention-kv.md](attention-kv.md) |
| Speculative decoding | Opus, high | [spec-decode.md](spec-decode.md) |
| Quantization and heads | Opus, high | [quant-heads.md](quant-heads.md) |
| Drafters | Opus, high | [drafter.md](drafter.md) |

Lanes were read-only on all sources. Each wrote findings with file:line citations; the lead spot-checked citations, re-ran the CPU checks and reconciled conflicts between lanes.

Pinned sources:
- deepseek-ai/DeepInfer `774ce24`, with its `vendor/` submodules (DeepGEMM `1fc9c66`, FlashMLA `2e5429f`, DeepEP `93eb6eb`, DeepSelect `bfa4507`).
- deepseek-ai/DeepSpec `005e03b`.
- deepseek-ai/deepseek-recipe `935817c`.
- vLLM `v0.31.0` (HyperQwen's pin) and upstream main `4bd75e952a`.
- HyperQwen `main` at `95ac251`.

Mode numbers used throughout: A 942 tok/s batch; B 127 tok/s; C 381 tok/s; E 67 tok/s at `--ctx 20000`.
