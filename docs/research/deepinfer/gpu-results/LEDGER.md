# GPU results ledger

One row per item. The GPU agent keeps this file current and commits it with every result. Run instructions are in [../gpu-agent-prompt.md](../gpu-agent-prompt.md). Protocols and win criteria are in each item's "GPU validation" bullet in its appendix.

Status: `todo` · `running` · `done` · `blocked` · `skipped`. Verdict: `WIN` (clears the appendix's win criterion and the baseline spread) · `LOSS` · `NEUTRAL` (inside noise) · `INVALID` (correctness or lossless check failed) · `INFO` (measurement only, no win criterion).

## Phase 1: health and baselines

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| env | Box inventory: GPU, driver, power limit, WSL, Docker, models | – | done | INFO | 3090 at 250 W, driver 610.88, WSL2; bf16 DFlash2 fetched | [ENVIRONMENT.md](ENVIRONMENT.md) |
| health | `make verify` plus the repo's GPU-applicable `bench/test_*` | – | todo | | | |
| cpu-checks | Re-run every script in `cpu-checks/`; compare with committed `.out` | – | todo | | | |
| baseline | Headline numbers per mode A–E on this box | health | todo | | | |

## Phase 2: measurement tools and probes

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| quant-2-check | `quant_real_ckpt_check.py` on the real checkpoint | – | todo | | | |
| runtime-2 | `bench/step_curve.py`: ms/step vs verified tokens, per backend and depth | baseline | todo | | | |
| evals-2 | `VLLM_GPU_SYNC_CHECK=warn` boot per mode | health | todo | | | |
| evals-1 | Decode-vs-prefill KL probe per mode | health | todo | | | |
| evals-4 | Thinking-on speed and acceptance cohort | baseline | todo | | | |
| evals-3 | Paired quality battery | health | todo | | | |
| spec-5 | Σ min(p, q) acceptance telemetry | health | todo | | | |
| quant-1-calib | `bench/act_calib.py` per-128 and Hadamard-128 modes | health | todo | | | |
| attn-ncu | ncu on pthfa at 150k, T=8 (attention-kv open question 1) | health | todo | | | |
| e-profile | Torch profile of E at 200k: attention share per step | health | todo | | | |
| evals-5 | Special-token injection check (tokenizer on `<\|im_end\|>`), hardening if needed | health | todo | | | |
| evals-6 | Streamed tool-call test, tool case in `api_smoke` | health | todo | | | |
| evals-7 | Golden greedy canary keyed by (GPU, pin, mode) | health | todo | | | |
| evals-8 | Per-component GPU time (nsys) | health | todo | | | |
| evals-9 | Acceptance panel from DeepSpec's datasets (local copy, uncommitted) | health | todo | | | |
| evals-10 | Throughput vs per-user Pareto sweep per mode | baseline | todo | | | |
| evals-11 | Automatic warmup detection | health | todo | | | |

## Phase 3: config and patch experiments

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| quant-1 | Per-128 int8 activation scales, route A (Humming) | quant-1-calib, evals-1 | todo | | | |
| quant-2 | Lossless negative-scale fold, fix A | quant-2-check shows negative groups use −8 | todo | | | |
| quant-7 | int4 embedding where `EMBED_UVA` can't be used | evals-1 | todo | | | |
| quant-5 | Batch-mode lm_head as W4A8 | evals-1 | todo | | | |
| spec-0 | `DFLASH_TOKENS=3/5/7` ladder at 90k, 150k, 240k | runtime-2 | todo | | | |
| spec-1 | GDN opt-in to adaptive verification (= runtime-1) | runtime-2 | todo | | | |
| kvarn-varlen | Root-cause KVarN corruption on verify-length change (`start_qwen.sh:399-428`) | – | todo | | | |
| drafter-1a | Host-side confidence trim, stage 0 | runtime-2 | todo | | | |
| spec-2 | Fresh-confidence device trim | spec-1 | todo | | | |
| spec-4 | Lookup-aware confidences | spec-1 | todo | | | |
| runtime-3 | Bound the tail of forced tool calls | – | todo | | | |

## Phase 4: kernel prototypes

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| attn-1.1 | pthfa unpadded 48-row tiles at T=8 (lossless) | attn-ncu | todo | | | |
| attn-1.2 | PV fp16 accumulate | attn-1.1 | todo | | | |
| attn-1.3 | int8 QK | attn-1.2, evals-1 | todo | | | |
| attn-3 | int8-QK prefill on quantized-KV arms | evals-1 | todo | | | |
| quant-6 | Fuse int8 activation quant into its producer | quant-1 | todo | | | |
| attn-5 | Hadamard-rotated int8/int4 per-token-head KV | evals-1 | todo | | | |

## Phase 5: drafter tools

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| drafter-2 | Offline DFlash2 gate (`drafter/eval_dflash2.py`) | – | todo | | | |
| quant-3 | Better GPTQ for the int4 drafter and lm_head | drafter-2 | todo | | | |
| drafter-3 | TV-loss repair of the W4A16 DFlash2 | drafter-2 | todo | | | |
| drafter-1b | Learned confidence head, stage 1 | drafter-1a or spec-2 shows a win | todo | | | |

## Phase 6: gated stretch

| ID | What | Gate | Status | Verdict | Key number | Evidence |
|---|---|---|---|---|---|---|
| attn-2 | KVarN verify on a pthfa-family CUDA kernel | e-profile: attention ≥ 25% of E's step at 200k; attn-1.1 done | todo | | | |
| spec-3 | RecoverSSM for GDN | spec-1 WIN | todo | | | |
| quant-4 | Block-Hadamard-128 rotation of linear inputs | quant-1 route A blocked or slower than Marlin by > 5% | todo | | | |
| attn-4 | Training-free top-k sparse attention | attn-2 done; needle and GSM8K gates ready | todo | | | |
| attn-6 | Device-side balanced split scheduler | appendix expects ~0; probe only if attn-ncu shows split imbalance | todo | | | |
| drafter-4 | Own-distribution fine-tune | needs an 80 GB GPU: out of scope here | skipped | | | |
