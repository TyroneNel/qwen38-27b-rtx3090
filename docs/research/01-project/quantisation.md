# Quantisation and the 24 GB memory budget

Lane: quant-analyst. Scope: the weight formats, the KV-cache formats and where every gigabyte of a 24 GB RTX 3090 goes in each serving mode. No GPU was used. Every number is labelled **M** (measured, with a path to the evidence) or **E** (estimated, with the formula). Most byte counts come from safetensors headers fetched by HTTP range request, with no weights downloaded: [weights_breakdown.py](weights_breakdown.py) and its output [weights_breakdown.out](weights_breakdown.out), pinned to the Hub commits listed in the script. In this file, "`.out` §n" means section n of that output.

Mode letters follow `README.md:79-85`:

- **A**: batch.
- **B**: `SPEC=dflash2 CTX=fast`, k=7.
- **C**: B with `DFLASH_TOKENS=15`.
- **D**: `SPEC=mtp CTX=long`, k=3.
- **D′**: `SPEC=dflash2 CTX=long`.
- **E**: `SPEC=dflash2 CTX=huge`.

## Summary

- **The body is the same in every mode.** It is the AutoRound int4 checkpoint: group size 128, symmetric, with fp16 scales, about 50% of them negative. It is 11.68 GiB on disk (M, `.out` §1) and runs on Marlin as W4A16. Batch mode additionally runs W4A8-int8 on the MLP.
- **Only the heads differ between variants.**
  - **Base:** lm_head and embed are int8 RTN g128, 1.20 GiB each. MTP is int8.
  - **Fast:** lm_head and MTP are int4 GPTQ, at 0.61 GiB and 0.31 GiB. Embed stays int8.
- **The KV cache has five formats**, picked by mode: bf16 (B, C), fp8 (A, D, D′ by default), int8 per-token-head (D′ alternative, `PTH_FA`), KVarN k4v2 (E), and int4 per-token-head (batch opt-in). [review: wrong for D′. On an 8.6 GPU with `pth_flash_attn` installed (the image builds it, `Dockerfile:32-33`) and the measured launch, D′ defaults to `PTH_FA=1`, int8 per-token-head on the CUSTOM backend, and `FA2_FP8KV` stays 0 (`resolve_config.sh:95-116,130-131`). fp8 + FA2 is D′'s fallback only (no pthfa, a non-8.6 GPU, `VISION=1`, `DFLASH_TOKENS`≠7, an explicit `FA2_FP8KV`, etc., `resolve_config.sh:100-108`). fp8 is the default in A and D only.]
- **Quality evidence is thin for bf16 against fp8.** The only per-token comparison against a bf16 cache shows fp8 at 1.7× the logprob drift of int8: 0.036 vs 0.021 nats. It fails the repo's non-inferiority test, while GSM8K and perplexity cannot see the difference (M, `docs/long-context.md:470-476`).
- **The single-user budget has little slack.**
  - The DFlash2 modes load about 14.74 GiB of weights (E [review: no logged load of this configuration exists; the "match" is derived from an Uncensored-checkpoint log, see the resident-weights table]) and pin a 5.20 GiB pool, or 4.90 GiB at E.
  - CUDA graphs take about 1.2 to 1.8 GiB, and a ~0.94 GiB verify buffer comes on top.
  - The shipped configuration leaves **about 0.9 GiB** spare at B (E, derived from a measured OOM) [review: two other OOM/boot logs give ≈0.7 GiB (vision tower OOM, `single-user/start_qwen.sh:920`, desktop box) and ≈1.3 GiB (headless, 440 MiB free with the 0.858 GiB tower resident, `docs/gotchas.md:134-137`); read it as 0.7-1.3 GiB] and **about 0.15 GiB** at E (M, a KV_MEM ladder).
- **A bigger drafter must pay in KV.** At B, 1 GiB of drafter costs about 13.4k pool tokens. At D′ the same gigabyte costs about 26k tokens, and at E about 55k (E, `.out` §6). [review: these are vLLM's logged pool units (max concurrency × MAX_LEN, `~/vllm/vllm/v1/core/kv_cache_utils.py:1074-1083`), which amortise per-request state. Context one request actually loses per GiB, from page geometry: B 16,384 (65,536 B/token), D′ int8 ≈31.8k (33,792 B/token; a −1 GiB KV_MEM boot measured 382 → 308 pages, `bench/test_resident_print.sh:80-81`), D′ fp8 32,768. At B that drops the largest single request from 64k to ≈56k.]

## Target weights per variant

Checkpoint metadata (M, the config.json `quantization_config` of each repo; header bytes from `.out` §1):

| Tensor group | Base, `models/Qwen3.8-27B-W4A16-AutoRound` (dbirks@1f05c441 + `prepare/`) | Fast, `…-fast` (syvai fast-variant@124c14e7) | Kernel on sm86 |
|---|---|---|---|
| 64-layer body linears: GDN `in_proj_qkv`/`in_proj_z`/`out_proj`, attention q/k/v/o, MLP gate/up/down | int4, g128, symmetric, `pack-quantized`, actorder none, fp16 scales; 12,543,596,800 B = 11.68 GiB (M) | same shards (hardlinked, `prepare/fetch_fast_variant.py:46-52`) | Marlin W4A16 (`MarlinLinearKernel`, min capability 75, `~/vllm/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:30-33`). Batch mode adds W4A8-int8 on the MLP through `VLLM_MARLIN_INPUT_DTYPE=int8` (`patches/marlin-int8-layer-select`, `marlin-int8-negative-scales`; `PATCHES.md:29-30`) |
| Left in bf16 | GDN `in_proj_a/b` 47.2 MB, conv1d / A_log / dt / norms 5.3 MB (M); the ignore list also names `lm_head` and `mtp.fc` (215 entries, M) | GDN `in_proj_a/b`, vision (206-entry ignore list, M) | plain GEMM |
| lm_head [248,320 × 5,120] | int8 RTN g128 symmetric, fp16 scales (`prepare/quant_lm_head.py:34,57`; `prepare/quant_schema.py:93-120`): 1,291,264,000 B = 1.203 GiB (E, `.out` §3). Round-trip error 0.64% (`prepare/quant_lm_head.py:4-7`) | int4 GPTQ g128, Hessian from 300k captured hidden states (`drafter/gptq_lm_head.py`; `drafter/README.md:23-28`): 655,564,816 B = 0.611 GiB (M) | Marlin W8A16 / W4A16. Kept off the int8-activation path (`docs/optimizations.md:74`) |
| embed_tokens | int8 RTN g128, bf16 scales (`prepare/quant_embed.py:34`): 1.203 GiB (E); error 0.56% | int8 g128 (fast config `group_2`, M) | gather. `EMBED_UVA=1` moves it to pinned host RAM (`single-user/start_qwen.sh:933-938`; default 0) |
| MTP module (one decoder layer + `fc` 10240→5120) | int8 RTN g128 (`prepare/quant_mtp.py:35-37`; `fc` is quantised unless `--keep-fc`): 0.40 GiB (E) | int4 GPTQ, Hessians from the drafter's own inputs (`drafter/requant_mtp_gptq.py`): 0.204 GiB (M) | Marlin |
| `mtp.draft_lm_head` (40,960 rows of lm_head) | int8, 0.198 GiB (E) | int4, 108,134,416 B = 0.101 GiB (M) | Marlin |
| Vision tower | bf16, 921,460,192 B = 0.858 GiB (M); not loaded by default (`--language-model-only`, `single-user/start_qwen.sh:899-931`) | same | — |

**The negative-scale fold is lossy.** About 50% of the body's groups have a negative scale. Marlin int8 reads scales as unsigned, so `marlin-int8-negative-scales` folds the sign into the codes. DeepInfer's GPU check on the real checkpoint measured 50.4% negative groups, all of which use code −8, so the fold drops an LSB on the largest weight of each. The error is 0.069% of weight energy (M, `~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/quant-2-check.md`; DeepInfer item quant-2). This touches only the int8-activation path (batch MLP, `INT8_ACT=int8`). W4A16 reads the signed scales directly.

**HyperQwen's GPTQ has known weaknesses** (DeepInfer quant-3, not re-verified here): signed amax/7 scales, `--mse-clip` unused, and no act-order. They apply to the fast variant's lm_head and MTP and to the DFlash2 requant (`drafter/gptq_utils.py`).

**Resident weights per mode** (E, sums of header bytes, `.out` §4, language-model-only). Measured cross-checks:

| Mode | Loaded | Estimate | Cross-check |
|---|---|---|---|
| A | base body + int8 lm_head + int8 embed, no drafter | 14.14 GiB | — |
| D (fast) | body + int4 lm_head + int8 embed + int4 MTP | 13.85 GiB | "model loading 14.26 GiB" with `--language-model-only`, of which "non-weight overhead 0.41" (M, `docs/gotchas.md:98-103`) → 13.85 ✓ |
| D (base) | body + int8 heads + int8 MTP | 14.74 GiB | — |
| B / C / D′ / E | fast target without MTP (vLLM does not load `mtp.*` under `dflash`) + DFlash2 W4A16 1.193 GiB | 14.74 GiB | The Uncensored export "15.68 GiB of weights against the fast variant's 14.71" and "Model loading took 15.71 GiB" (M, `docs/third-party-checkpoints.md:196-202`); 15.71 − 0.97 = 14.74 ✓ [review: derived, not a logged load of B. The 14.71 is prose; and at D the load log carries 0.41 GiB of non-weight overhead (`docs/gotchas.md:101-103`), so the Uncensored 15.71 vs 15.68 split implies either ~0 overhead on the dflash2 path or that 15.68 is a file size. The 14.737 header sum (E) stands; the cross-check is weak] |
| B with the bf16 DFlash2 | — | 17.13 GiB | does not fit the shipped pins (`single-user/start_qwen.sh:521-532`) |

## KV cache formats per mode

Geometry: 16 full-attention layers (3, 7, …, 63), 4 KV heads × head_dim 256; 48 GDN layers with recurrent state (M, config + headers, `.out` §1). The DFlash2 drafter adds 5 sliding-window layers (window 2048, 8 KV heads × 128). MTP adds 1 full-attention layer.

Bytes per token per layer:

| Dtype | Bytes per token per layer | Source |
|---|---|---|
| bf16 | 4,096 | `single-user/start_qwen.sh:592-596` |
| int8 per-token-head | 2,080 (target), 2,112 (drafter) | same |
| fp8 | 2,048 | same |
| KVarN k4v2 | 840 | `kvarn/README.md:75` |

| Mode | KV dtype | Attention kernel (target / drafter) | Pool | Quality evidence |
|---|---|---|---|---|
| A batch (`KV=fp8` default) | fp8 e4m3, per-tensor scale 1.0 (the checkpoint has no `kv_cache_scheme`, M; vLLM's default `_k_scale` 1.0, `~/vllm/vllm/model_executor/layers/attention/attention.py:136-152`) | FlashInfer (`batch/start_qwen.sh:149-150`) | 242,010 tokens at `GPU_UTIL=0.93`, WSL2 (M, DeepInfer `gpu-results/baseline.md:49`); ~205-225k native (`docs/long-context.md:32`) | Batch PPL 8.044 on fp8 KV against single-user bf16-KV 8.045 on the same battery (M, `docs/quality.md:31,56`; different configurations, so this indicates no loss rather than proving it) |
| A, `KV=int4pth` | int4 per-token-head | TRITON_ATTN (`batch/start_qwen.sh:121`) | 355k at 0.93 (M, `docs/long-context.md:98`) | PPL 8.257 against fp8 8.235 (+0.3%); needle at 100k and 240k (M, `:99-100`) |
| A, `KV=kvarn` | KVarN k4v2 g128, block 128 | KVarN Triton (`batch/start_qwen.sh:125`) | 302-420k (M, `:32`) | PPL +0.16% against fp8 (8.236 vs 8.223); needle 4k…240k (M, `:34-35`) |
| B (k=7), C (k=15) | bf16 | FLASH_ATTN + split-KV verify (`single-user/start_qwen.sh:210-214`) | 5.20 GiB pinned → 69,758 / 57,669 tokens (M, `start_qwen.sh:469,509`; DeepInfer measured 68,605 / 57,669 on WSL2) [review: 68,605 is B's default, with the prefix cache on; 69,758 is with it off (`single-user/README.md:167-171`)] | reference cache |
| D (`SPEC=mtp CTX=long`), default | fp8 | FlashInfer (`start_qwen.sh:220-224`; `PTH_FA` defaults to 0 here, `start_qwen.sh:297`) | profiled at `GPU_UTIL=0.93`: 183,673-184,438 tokens at 150k (M, `docs/gotchas.md:105-107`) | Same fp8 format as A |
| D, `PTH_FA=1` | int8 per-token-head | pthfa CUSTOM on target and MTP (`start_qwen.sh:271-280`) | 206,532-207,286 (M, DeepInfer `baseline.md:49`, WSL2) | 3080 TP=2: GSM8K 0.970 vs 0.960, MTP tok/step 3.635 vs 3.637 against TRITON int8; logprob drift to bf16 32k 0.0223 vs 0.0232 nats (M, `docs/long-context.md:302-310`) |
| D′ fallback on 8.6 (`FA2_FP8KV=1`) [review: was "default"; it is the default only when the pthfa default does not fire] | fp8 | `fa2/` plugin FA2 fp8 (vLLM's own FA2 refuses fp8 below sm90: `~/vllm/vllm/v1/attention/backends/fa_utils.py:306-312`; Triton fp8 needs sm89: `triton_attn.py:512-514`) | 140,902 tokens in 5.20 GiB (M, `docs/long-context.md:190`) | T4 battery: GSM8K 0.950 vs 0.955, PPL +0.04% (M, `:190`). **Teacher-forced drift to bf16 0.0360 / 0.0338 nats at 32k / 128k, against int8's 0.0219 / 0.0194; fails non-inferiority by +0.014 nats** (M, `:470-476`) |
| D′ default [review: the image and any install with `pth_flash_attn`, measured launch, 8.6] | int8 per-token-head | pthfa CUSTOM on target and drafter (`resolve_config.sh:98-117`) | 136,429 (prefix cache) / 138,696 (M, `start_qwen.sh:501-502`) | drift 0.0207 / 0.0193 nats, passes; GSM8K 0.960-0.965; T4 PPL 8.4022 vs 8.4013 (M, `docs/long-context.md:470-484`) |
| D′, `FA2_FP8KV=0` | int8 per-token-head | TRITON_ATTN + `spec-decode-int8-kv` verify kernel | same as above | drift 0.0219 / 0.0194 |
| E | KVarN `kvarn_k4v2_g128`, block 128 | KVarN Triton (`start_qwen.sh:215-219`) | 4.90 GiB → 268,169 tokens (M, `start_qwen.sh:479-481`; DeepInfer baseline `:49`) | GSM8K 95.2% (n=600); 25k verbatim reproduction exact; needle 100k (M, `docs/long-context.md:570-575`). In MTP mode KVarN costs ~7% acceptance (2.38 vs 2.56 tok/step, M, `:52-60`) |
| GDN state, all modes | float16 SSM state (`--mamba-ssm-cache-dtype float16`, `start_qwen.sh:1030-1036`); conv state bf16 | — | per request: 48 × (20,480 × (3+k) + 1,572,864) B (`start_qwen.sh:567-568`) | "the 16-bit recurrent state is free": PPL 8.044 vs 8.045 (M, `docs/quality.md:30-31`) |

**How the pools are paged** (E, launcher formula at `single-user/start_qwen.sh:566-616`, `.out` §5; it reproduces all three logged pool/block pairs in `bench/test_resident_print.sh:83,96,99`):

- A page holds 8 layers of one KV group.
- The attention block is raised until one page covers one GDN state.
- A resident request holds 6 × (2+k) state pages before it stores any context.
- Each token of context then costs 2 pages per block. That is 65,536 B per token on bf16, 33,792 on int8 and 32,768 on fp8.

| Arm | Block (tokens) | Page | State per resident request |
|---|---|---|---|
| bf16, k=7 | 448 | 14.0 MiB | 54 pages = 0.738 GiB |
| bf16, k=15 | 480 | 15.0 MiB | 102 pages = 1.49 GiB |
| int8, k=7 | 864 | 13.9 MiB | 0.734 GiB |
| fp8, k=7 | 880 | 13.75 MiB | 0.725 GiB |

The state cost explains why C keeps only 4 seats. The docs quote 0.82 GiB and 0.88 GiB for the k=7 state (`docs/long-context.md:694-695`, `docs/optimizations.md:349-350`). Those figures predate the 6 × (2+k) align-mode count, and the formula gives 0.74 GiB.

## Memory budget per serving mode

Rows add up to the card's 24 GiB (24,576 MiB, the spec figure). [review: CUDA reports 25,272,516,608 B = 23.54 GiB total on this 3090 (`docs/gotchas.md:521-522`), so every residual below is ~0.46 GiB smaller: B ≈0.3-0.55, E ≈1.9] The "residual" row is what remains: CUDA context, cuBLAS/Marlin/Triton workspaces, the activation peak, allocator fragmentation and any KVarN staging pool. It is derived, not measured. Graph sizes are the launcher's documented figures; no per-mode `Graph capturing finished … GiB` log is in the repo.

| Row | B (dflash2 fast, k=7) | C (k=15) | D′ (dflash2 long, k=7) | E (dflash2 huge) | D (mtp long, k=3) | A (batch, 64 seats) |
|---|---|---|---|---|---|---|
| Target weights | 13.54 (E) | 13.54 (E) | 13.54 (E) | 13.54 (E) | 13.55 (E) | 14.14 (E) |
| Drafter / MTP weights | DFlash2 1.193 (M, file) | 1.193 (M) | 1.193 (M) | 1.193 (M) | MTP + draft head 0.305 (M) | — |
| = loaded, cross-check | 14.74; measured 14.74 via the Uncensored load (M, `third-party-checkpoints.md:196-202`) [review: derived (E), not a logged B load] | same | same | same | 13.85; measured 14.26 incl. 0.41 non-weight (M, `gotchas.md:101-103`) | — |
| KV pool (incl. GDN state pages) | 5.20, pinned `KV_MEM=5583457484` (M, `start_qwen.sh:519`) | 5.20 (M, `:515`) | 5.20 (M, `:505`) | 4.90, `5261334938` (M, `:497`) | profiled, not pinned: ≈ 7.6 (E: 204,336 pool tokens × 40,172 B per pool token measured by the `EMBED_UVA` A/B, `.out` §6, `single-user/README.md:443`) [review: with this doc's own D pool, 183,673-184,438 (`docs/gotchas.md:105`), it is ≈6.9; read 6.9-7.6] | profiled at `GPU_UTIL=0.94`: ≈ 7.0-7.9 (E: 0.94 × 24 − weights − 1.09 activation peak − graphs; pool tokens × ≥32,768 B) |
| CUDA graphs | ~1.2 (`docs/optimizations.md:255-257`, `gotchas.md:190`) to ~1.45 (`start_qwen.sh:496-499`) | ~1.8 (`start_qwen.sh:514-516`) | ~1.2-1.45 | ~1.45 (`:496`) | CG 32 (`start_qwen.sh:664-667`), size not documented: 0.3-0.6 (E) | CG sized for 64 seats, not documented |
| Split-KV verify buffer | 0.94 (M: "Tried to allocate 960.00 MiB", `third-party-checkpoints.md:203-205`) | ≥ 0.94 (E) | kernel-specific, not measured | not measured | n/a | n/a |
| Activation peak | inside the residual; swings ~1 GiB between starts (`gotchas.md:192`), which is why the pool is pinned | same | same | same | inside vLLM's 0.93 budget; cold compile adds 0.92 (M, `gotchas.md:26`) | 1.09 warm, 1.96 cold (M, `gotchas.md:22`) |
| Vision encoder | 0 (`--language-model-only`); 0.858 if `VISION=1 VISION_OFFLOAD=0` (M, `gotchas.md:98-103`) | 0 | 0 | 0 | 0 | 0 |
| Free after first request | ≈ 0.93 (E: 926 MiB free at an OOM with +0.97 GiB weights, M, `third-party-checkpoints.md:205`; "about a gigabyte to spare", `start_qwen.sh:522-523`) | ≈ 0.6 (E: B − 0.35 GiB extra graphs) | not measured | 0.58 after boot at `MAX_SEQS=8` (M, `start_qwen.sh:634-635`) | outside vLLM: 0.07 × 24 = 1.68 including context (E) | worst prefill clears by 134-176 MiB (M, `gotchas.md:200-201`, all-layer int8) |
| Residual (context, workspaces, activations, slack) | 24 − 14.74 − 5.20 − 1.2…1.45 − 0.94 − 0.93 ≈ 0.75-1.0 (E) | ≈ 0.7 (E) | ≈ 1.5-2.5 (E) | 24 − 14.74 − 4.90 − 1.45 − 0.58 ≈ 2.3 (E; includes `KVARN_POOL_MEM_FRAC=0.15` staging, `start_qwen.sh:219`) | — | — |

Two anchors fix the overall scale:

- On the WSL2 4090 at D′, with k=7, `MAX_LEN=81920` and `MAX_SEQS=4`, the server logged "Model loading took 15.92 GiB" and "Available KV 4.85 GiB" (M, `docs/spec-decode-scratch-token-units.md:308-312`). That run used `alternative.sh` int4, a different target build.
- "Resident memory after boot 23,270 MB" at `MAX_SEQS=32` (M, same file, `:322-324`).

## Quality evidence

| What | Number | Label, path |
|---|---|---|
| Base W4A16 (int8 heads), bf16 KV | PPL 8.045 (en 10.68 / da 10.85 / code 3.05), GSM8K 95.5% | M, `docs/quality.md:56` |
| Fast variant (int4 GPTQ lm_head + MTP) | PPL 8.095 (+0.6%), GSM8K 96.5% | M, `docs/quality.md:58`; lm_head KL to bf16 head 0.0029 vs RTN 0.0068 (M, `drafter/README.md:23-28`) |
| int4 RTN lm_head (not shipped) | PPL 8.17 (+1.5%) | M, `docs/quality.md:57` |
| Batch int8 activations: gate/up / MLP (default) / all | +0.9% / +2.2% / +3.7% PPL; GSM8K 95.5 / 95.0 / — | M, `docs/quality.md:33-36`. DeepInfer re-measured all-layer at +4.00% (M, `gpu-results/evals-3.md`) |
| fp16 GDN state | PPL 8.044 vs 8.045 | M, `docs/quality.md:30-31` |
| IFBench strict, batch default | 78.3 vs model card 79.5 | M, `docs/quality.md:18` |
| KV: fp8 / int8 / int4pth / KVarN, batch battery | 8.235 / 8.231 / 8.257 / +0.16% vs fp8 | M, `docs/long-context.md:35,99` |
| KV: drift to bf16, teacher-forced, 32k / 128k | TRITON int8 0.0219 / 0.0194; pthfa int8 0.0207 / 0.0193; FA2 fp8 0.0360 / 0.0338 nats | M, `docs/long-context.md:470-476` |
| Unquantised bf16 reference | none in the repo; the 27B bf16 does not fit one card | — |

## Headroom

"Spare" means memory a new drafter or more KV could take without lowering anything else. "Exchange rate" means the context lost per GiB if KV_MEM is lowered to make room (E, `.out` §6 = KV_MEM ÷ logged pool).

| Mode | Spare today | Exchange rate (pool tokens per GiB) | Notes |
|---|---|---|---|
| B | ≈ 0.5 GiB usable (E): ~0.93 free minus a ~0.4 GiB floor that the first batch needs (dead at 396 MiB free, fine at 436, M, `start_qwen.sh:637-639`) | 13,415 (80,040 B per pool token) | The 0.97 GiB Uncensored overshoot failed by 34 MiB (M). A bf16 DFlash2 (+2.39 GiB) needs `KV_MEM` ≈ 3.0 GB and `DFLASH_MAX_LEN=8192` (M, `start_qwen.sh:528-531`). Each extra drafter layer also adds state and graph memory, not only weights |
| C | ≈ 0.2 GiB (E) | 11,090 | The 1.49 GiB per-request state is the binding term, not the pool |
| D′ | ≈ 0.5 GiB (E, same pin and graphs as B) | 26,236 (int8) / 27,097 (fp8) | The best mode to spend weight savings on: twice B's tokens per byte |
| E | ≈ 0.15 GiB (M: KV_MEM +150 MiB ran, +200 MiB killed at `MAX_SEQS=8`, `start_qwen.sh:652-655`) | 54,728 | Defaults to `MAX_SEQS=2`, so slightly more room than the ladder shows (not measured) |
| D | No fixed spare: vLLM gives everything inside 0.93 to KV. Anything new comes straight out of the pool | ≈ 26,700 (40,172 B per pool token, M via `EMBED_UVA`) | `EMBED_UVA=1` frees 1.20 GiB → +32,143 tokens (M, `single-user/README.md:443`) |
| A | ≈ 0.15 GiB at the worst prefill (M) | ≥ 32,768 per GiB (fp8 context only, E) | Only by lowering `GPU_UTIL` |

**Levers that free memory and already exist:**

- `EMBED_UVA=1`: 1.20 GiB, measured.
- An int4 embedding (DeepInfer quant-7): ~0.6 GiB, estimated as 1.203 − 0.611.
- The fast lm_head is already int4.

A drafter that is 1 GiB larger costs about 13k tokens at B or about 26k at D′. It also raises per-step read time at about 1.07 ms per GB at 936 GB/s. That is an estimate, a lower bound.

## Open questions

1. **No per-mode memory log in the repo.** We lack "Model loading took", "Available KV cache memory", graph-capture GiB and "free after boot" for each mode on one native 3090. One boot per mode with `VLLM_LOGGING_LEVEL=DEBUG` would replace every E in the budget table with M.
2. **fp8 KV fails the logprob test, but D′ still defaults to it** wherever pthfa is not installed. [review: and wherever a gate in `resolve_config.sh:100-108` fails, e.g. `VISION=1`, which the prior box's `.env` sets (`ENVIRONMENT.md:27`).] Which bf16-drift threshold should gate a KV format? D and A also run fp8 and have no drift measurement against bf16.
3. **The D default disagrees with DeepInfer's baseline.** In this checkout `SPEC=mtp CTX=long` defaults to fp8 on FlashInfer (`PTH_FA=0`, `start_qwen.sh:297`; `resolve_config.sh:98-103` defaults `PTH_FA` only for dflash2/long). DeepInfer's baseline describes D's default arm as pthfa (`gpu-results/baseline.md:65`). Which image was that? [review: answered. `baseline.md:23`; the image is `sha-2cf50a0` (`ENVIRONMENT.md:24`), which contains upstream b4b97ec (#345, PTH_FA default on mtp/long) and our 95ac251 does not (`inventory.md:9`, `upstream-diff.md:14`). Consistent with `resolve_config.sh:98-103` here.]
4. **The resident-state figure has three values.** The launcher formula gives 0.74 GiB, the docs say 0.82 / 0.88 GiB. Re-fit gotcha 33 against the 6 × (2+k) count.
5. **The 960 MiB verify buffer** is the largest single transient at B. Does pthfa/FA2 at D′ allocate the same, and does it scale with k (C)?
6. **The GPTQ fixes from DeepInfer quant-3** (signed scales, MSE clip, act-order) have not been applied to the fast lm_head, the MTP module or DFlash2.

## Reviewer additions

[review: phase 1 review, `docs/research/00-meta/reviews/phase1-quant-heads.md`]

- **Memory budget against the CUDA-visible 23.54 GiB** (E unless marked; weights from headers, re-run identical):

  | GiB | B | D′ (PTH int8 default) | E |
  |---|---|---|---|
  | Weights (target 13.54 + DFlash2 1.19) | 14.74 | 14.74 | 14.74 |
  | KV pin (M, launcher) | 5.20 | 5.20 | 4.90 |
  | Graphs (launcher figures) | 1.2-1.45 | 1.2-1.45 | ~1.45 |
  | Split-KV verify buffer (M at B) | 0.94 | unknown (CUSTOM kernel) | n/a |
  | Free after boot | 0.7-1.3 (E, three OOM/boot logs) | not measured | 0.56-0.58 at `MAX_SEQS=8` (M, `docs/gotchas.md:515,535`) |
  | Residual: CUDA context, workspaces, non-weight load overhead | ≈0-0.75 (0.3-0.5 at 0.93 free) | — | ≈1.9 |

- **Real context per GiB of KV_MEM** (page geometry, `start_qwen.sh:610-613`): B 16,384 tokens, D′ int8 ≈31.8k (measured 74 pages per GiB, `bench/test_resident_print.sh:80-81`), fp8 32,768, E ≈80k (16 × 840 B/token, block 2048 per `kvarn/README.md:76-78`; unverified). A 1 GiB larger drafter at the shipped pins makes `MAX_LEN` 65,536 at B and 131,072 at D′ unservable for one request (≈56k and ≈109k fit).

