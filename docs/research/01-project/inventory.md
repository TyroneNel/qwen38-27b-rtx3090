# Project inventory

Branch `research/eagle4-expedition` = `origin/main` `95ac251`, a fork of syv-ai/HyperQwen. Read-only map written by the code-mapper lane (no GPU, nothing served). All paths are repo-relative. `file:line` cites our tree at `95ac251`. A cite into a `.patch` file is a line of the patch file, not of the patched vLLM file (the vLLM source is not in this repo; the tree is vLLM 0.31.0 plus `patches/series`).

Number labels: **measured** = from a repo doc or the prior GPU-box results (`~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/`, called *prior GPU box* below), with the path. **estimated** = my arithmetic, with the reasoning.

Two things to know before reading further:

1. **Our tree lags `upstream/main` by 5 commits** (`git log 95ac251..upstream/main`: `63f7581`, `2cf50a0`, `373a695`, `b4b97ec`, `433b050`). One matters for the baseline: `b4b97ec` makes `PTH_FA` default to 1 on `SPEC=mtp CTX=long` on sm86. In our tree the default exists only for `SPEC=dflash2 CTX=long` (`resolve_config.sh:99-104`; `mtp/long` is missing from that `case`). The prior GPU box ran image `2cf50a0`, so its mode D number (99.8 tok/s, `baseline.md`) is on the pthfa arm, which our tree does **not** pick by default for `SPEC=mtp CTX=long`.
2. The launcher default is `SPEC=mtp` (`single-user/start_qwen.sh:197`), but `.env.example:6` and the README present `SPEC=dflash2` as the default single-user mode. Docker users get dflash2 only through `.env`.

## 1. Directory map

| Path | One line |
|---|---|
| `README.md` (252 lines) | Entry doc: modes A-E table (`:79-85`), FA2_FP8KV/PTH_FA defaults (`:96-101`), docs index (`:213-248`) |
| `Dockerfile` (65) | Single image: CUDA 13.0.3 base, vLLM 0.31.0 venv, builds pthfa and fa2, applies patches + KVarN, runs verify and gate tests at build |
| `docker-compose.yml` (147) | Services `prepare`, `single`, `batch`, `solo` (second engine on its own card, `SOLO_PORT` 18021) |
| `.env.example` (98) | The knob list Compose passes into the container (`SPEC=dflash2` at `:6`) |
| `Makefile` | `up-single`, `up-batch`, `down`, `logs`, `ps`, `verify`, `verify-install`, `shell`, `keygen`, `doctor` |
| `docker/` | `entrypoint.sh` (prepare, then verify, then start), `prepare.sh` (model fetch and requant, flock), `requirements.txt` (pins) |
| `single-user/` | Single-user launcher `start_qwen.sh` (1046 lines), `alternative.sh` (int4-KV, 256k), `select_model.sh`, `qwen-server.sh` (warmup wrapper), systemd unit, README |
| `batch/` | Batch launcher `start_qwen.sh` (318 lines), systemd unit, README |
| `launcher_common.sh` (226) | Shared shell functions: bind host, int8 exports, bool-knob check, KV offload, prefix cache, `PRINT_ARGV=1` dry run (`qwen_exec`) |
| `resolve_config.sh` (206) | Validates CTX/SPEC/KV, resolves PTH_FA and FA2_FP8KV defaults from compute capability, prints redacted effective config |
| `resolve_api_key.sh` | API key chain (env, `api_key.txt`) |
| `verify.sh` (534) | Install/config/model self-check; `--no-server`, `--install`, `--wait N`; exit 1 on FAIL |
| `patches/` | `series` (51 patches, `:17-67`), `apply.sh` (`--fuzz 0`, `--kvarn`), `check_vllm_series.sh`, `_check_applied.py` |
| `PATCHES.md` | Generated patch table (from patch headers by `scripts/patches_md.py`) |
| `kvarn/` | KVarN 4/2-bit KV cache: `files/` (Triton kernels, config), 4 patches, `install.sh` |
| `fa2/` | FlashAttention-2 fp8-KV plugin for sm86 (`fp8_attn.cu`, `prefill.cu`, `fa2-fp8kv.patch`, python adapter, `PIN`) |
| `pthfa/` | pth-flash-attn: int8 per-token-head KV attention CUDA extension and `CUSTOM` backend plugin (`csrc/pth_attn/`, `python/`, `PIN`) |
| `prepare/` | CPU-side model prep: requant lm_head, embed, MTP; draft vocab; fetch fast variant, DFlash2, third-party |
| `drafter/` | Drafter tooling: prompt collection, data gen, capture, MTP fine-tune (did not help), GPTQ, DFlash2 requant |
| `bench/` | About 65 entries: harnesses, ladders, quality battery, kernel/knob tests (`README.md` lists them), `demo/`, `results/` |
| `scripts/` | `export-patch.sh`, `hq-doctor.sh`, `patches_md.py`, `pin-bump.py`, `port-*.py/sh` (vLLM pin-bump tooling), `check_verify_graded.py` |
| `.github/workflows/` | `docker-image.yml`, `patch-integrity.yml` |
| `docs/` | Operator and design docs (index in section 9), `reproductions/`, `media/`, and `research/` (this expedition) |
| `docs/research/` | Expedition outputs: `00-meta` (brief, plan, decisions), `01-project` (this file), `02-eagle4`, `03-vllm`, `04-benchmarks`, `05-findings`, `reviews` |
| `.claude/` | Agent definitions (untracked tooling, not report content) |

## 2. Stack and pinned versions

| Component | Pin | Where |
|---|---|---|
| Base image | `nvidia/cuda:13.0.3-base-ubuntu24.04` | `Dockerfile:14` |
| Python | 3.12, venv at `/app/venv` | `Dockerfile:18,24` |
| nvcc (system) | `cuda-nvcc-13-0`, `cuda-cudart-dev-13-0`, `libcurand-dev-13-0` | `Dockerfile:19` |
| vLLM | `0.31.0` (patched) | `docker/requirements.txt:5` |
| torch | not pinned directly; resolved by vLLM 0.31.0 (cu130). `pthfa/PIN` records `TORCH_VERSION=2.13.0` | `docker/requirements.txt:3`, `pthfa/PIN` |
| transformers / tokenizers | 5.15.0 / 0.22.2 | `docker/requirements.txt:6-7` |
| compressed-tensors | 0.17.0 | `docker/requirements.txt:8` |
| huggingface_hub / hf_transfer | 1.32.0 / 0.1.9 | `docker/requirements.txt:9-10` |
| pandas / pyarrow | 3.0.5 / 20.0.0 (bench deps) | `docker/requirements.txt:19,25` |
| CUDA 13 compiler set (FlashInfer JIT) | nvcc and crt 13.0.88, cccl 13.0.85, nvvm 13.0.88 | `docker/requirements.txt:32-35` |
| xgrammar | 0.2.7 (resolved by vLLM, not pinned here) | `docs/vllm-0.31.md:11` ("vLLM 0.31.0 ... pins `xgrammar==0.2.7`") [review: re-traced from the prior map's runtime.md to the repo doc] |
| pthfa build | `PTH_COMMIT a31d8ab22d73...`, arch list `8.0;8.6;8.9` SASS only, `VLLM_VERSION 0.31.0` | `pthfa/PIN`, built at `Dockerfile:32-33` |
| fa2 build | plugin `AntonProkopyev/fa2-fp8kv-sm86` `5c3a7d88...`, flash-attention `FA_COMMIT 9cd61de3...` (the commit vLLM 0.31.0 pins), CUTLASS `62750a2` | `fa2/PIN`, built at `Dockerfile:37-39` |
| kvarn | applied unconditionally in the image (`bash kvarn/install.sh`) | `Dockerfile:51` |
| Image tag | `ghcr.io/syv-ai/hyperqwen:latest` | `docker-compose.yml:68` |
| Container ENV | `HOME=/cache` (torch.compile, Triton and FlashInfer caches live in the volume), `VLLM_NO_USAGE_STATS=1` | `Dockerfile:61` |
| Entrypoint | `docker/entrypoint.sh`, default CMD `single`, port 18020 | `Dockerfile:63-64` |
| Served name | `qwen3.8-27b` | `single-user/start_qwen.sh:1028` |

Research box (this machine) is not the serving stack: CPU only, torch 2.12.1+cu130, no vLLM (`docs/research/00-meta/plan.md`).

## 3. Models and checkpoints

| Role | Checkpoint | Source and prep | Size | Where selected |
|---|---|---|---|---|
| Target, base | `Qwen3.8-27B-W4A16-AutoRound` (W4A16, compressed-tensors/AutoRound, 48 GDN + 16 full-attention layers [review: confirmed in HF `config.json` @`1f05c441` `text_config`: `num_hidden_layers` 64, `full_attention_interval` 4, `layer_types` = 48 `linear_attention` + 16 `full_attention` at layers 3, 7, ..., 63; 4 KV heads, head_dim 256; repo: `docs/optimizations.md:55`, `:250-251`]) | `hf download dbirks/Qwen3.8-27B-W4A16-AutoRound` (`docker/prepare.sh:16,77`); `prepare/quant_lm_head.py`, `quant_embed.py`, `quant_mtp.py` requant lm_head, embed and the MTP module to int8 g128 in place | 22 GB (measured, `ENVIRONMENT.md`) | `batch/start_qwen.sh:85`, `single-user/select_model.sh:7` |
| Target, fast variant | `Qwen3.8-27B-W4A16-AutoRound-fast` (int4-GPTQ lm_head and MTP) | `prepare/fetch_fast_variant.py:46`: Hub `syvai/qwen3.8-27b-3090-fast-variant` (about 1 GB of tensors) laid over the base; `FAST_VARIANT=0` skips (`docker/prepare.sh:66`) | 15 GB (measured, `ENVIRONMENT.md`) | `single-user/select_model.sh:4-6` prefers it when present; batch does not |
| MTP head | inside the target dir (single-layer chain drafter; bf16 850 MB shipped, int8 or GPTQ-int4 here) | `prepare/quant_mtp.py`, `drafter/requant_mtp_gptq.py` | n/a | `SPEC=mtp` (`start_qwen.sh:666`) |
| Draft vocab | 40,960-row draft head, ids in `prepare/draft_vocab_ids.json`, counted over 5.4M tokens of the model's own output (97.5% coverage vs 92% for web text; `docs/optimizations.md:86-89`) | `prepare/build_draft_vocab.py`; wired by `patches/qwen3_5-mtp-draft-vocab.patch` | n/a | `SPEC=mtp` |
| DFlash2 drafter (W4A16) | `Qwen3.8-27B-DFlash2-W4A16`, 5 Qwen3-style layers, 7-token block in one non-autoregressive pass plus candidate selector | `prepare/fetch_dflash2.py:15` (`syvai/Qwen3.8-27B-DFlash2-W4A16`), made by `drafter/capture_dflash2.py` + `quant_dflash2.py` (GPTQ, 36 matrices) | 1.19 GB W4A16 vs 3.85 GB bf16 (`docs/optimizations.md:228-234`) | `single-user/start_qwen.sh:362-363` (falls back to bf16 dir) |
| DFlash2 drafter (bf16, teacher) | `incoai/Qwen3.8-27B-DFlash2` | `fetch_dflash2.py --bf16` | 3.6 GB (measured, `ENVIRONMENT.md`) | fallback in `start_qwen.sh:362` |
| DSpark drafter (alternative) | `RadixArk/Qwen3.8-27B-DSpark` | user supplies; `DRAFT_METHOD=dspark`; needs `patches/dspark-draft-quant-config.patch` | n/a | `start_qwen.sh:394` |
| Third-party checkpoints | Orca/Swift variants | `prepare/fetch_thirdparty.py`, `docs/third-party-checkpoints.md` | 12-16 GB each | not in default paths |
| Vision tower | 0.858 GiB bf16, dropped by default (`--language-model-only`); `VISION=1` keeps it, `VISION_OFFLOAD=1` puts it in pinned host RAM | `single-user/README.md:441-442`, `patches/vision-tower-cpu-offload.patch` | n/a | `start_qwen.sh:913,929` |

## 4. Serving modes and profiles

Entry point is `docker/entrypoint.sh` (stages: `prepare.sh` unless `PREPARE=0`, `verify.sh --no-server` unless `VERIFY=0`, then the launcher chosen by the Compose profile). Single-user launcher selects a profile from `SPEC x CTX`. Batch launcher selects from `KV`.

### 4.1 README modes A-E (`README.md:79-85`)

| Mode | Launch | What it is |
|---|---|---|
| A | `batch` (profile) | 64 seats, W4A8 int8 activations on MLP, fp8 KV, no speculation, async scheduling |
| B | `single`, `SPEC=dflash2` (default in `.env.example`) | DFlash2 + lookup, bf16 KV on FLASH_ATTN, 7 drafts, 64k |
| C | B + `DFLASH_TOKENS=15` | 16-token verify block filled from context, 4 seats, 56k |
| D | `SPEC=mtp CTX=long` | MTP k=3, fp8 KV (FlashInfer) in our tree; pthfa only on upstream/main |
| D' (prior expedition's name) | `SPEC=dflash2 CTX=long` | int8 KV; fp8 FA2 plugin or pthfa depending on resolve_config |
| E | `SPEC=dflash2 CTX=huge` (needs KVarN, always installed in the image) | KVarN 4/2-bit KV, 240k context |

### 4.2 Single-user profile matrix (`single-user/start_qwen.sh`)

| Profile | Attention + KV (`ATTN_ARGS`) | Drafts k | MAX_LEN | MAX_SEQS | KV_MEM (pinned pool) | Notes | Lines |
|---|---|---|---|---|---|---|---|
| `CTX=fast SPEC=mtp` (launcher default) | `FLASH_ATTN`, bf16 KV, `VLLM_SPEC_DECODE_ATTN=1` (split-KV verify) | 4 | 65536 | 8 | unpinned unless user sets `KV_MEM` | `CG=32` | `:210-214`, `:664-668` |
| `CTX=long SPEC=mtp` | `--kv-cache-dtype fp8` (FlashInfer chosen by vLLM); `PTH_FA=1` switches to `CUSTOM` int8 | 3 (k=4 crashes FlashInfer fp8, `docs/optimizations.md` "Going deeper") | 150000 | 8 | unpinned | `:220-223` |
| `CTX=huge SPEC=mtp` | `kvarn_k4v2_g128`, `--block-size 128`, `KVARN_POOL_MEM_FRAC=0.15`, `cudagraph_mode PIECEWISE` | 3 | 200000 | 8 | unpinned | PIECEWISE for non-dflash2 only (`:833-835`) | `:215-219` |
| `SPEC=off` / `none` | per CTX | 0 | per CTX | 8 | unpinned | `CG=32`; drafter-less | `:660-663` |
| `SPEC=dflash2 CTX=fast`, k=7 | `FLASH_ATTN` bf16, split-KV verify on | 7 (`DFLASH_TOKENS`) | 65536 | 8 | 5583457484 B (5.20 GiB) | pool 69,758 tok (measured, `docs/optimizations.md:254-255`) [review: 69,758 is the `PREFIX_CACHE=0` pool on 0.31; with the default `PREFIX_CACHE=1` it is 68,605 (`single-user/README.md:170-171`, `docs/gotchas.md:142-143`; prior GPU box `baseline.md:49`)] | `:511-520`, `:382` |
| `SPEC=dflash2 CTX=fast`, k=15 (mode C) | same | 15 | 57344 | 4 | 5583457484 B | pool 57,669 tok (measured, `docs/long-context.md:129` [review: was `:~120`]); decode graphs for both block lengths ~1.8 GiB vs ~1.45 GiB; `ASYNC_SCHED=0` | `:502-509`, `:430-435` |
| `SPEC=dflash2 CTX=long` | default `FA2_FP8KV=1` on sm86 (fp8 KV on FLASH_ATTN via `fa2/` plugin); `FA2_FP8KV=0` gives `int8_per_token_head` on TRITON_ATTN; `PTH_FA=1` (default when installed on sm86 and no overrides) gives `CUSTOM` int8 [review: precedence is PTH_FA first: `resolve_config.sh:95-116` sets `PTH_FA=1` when pthfa is installed (always, in the image: `Dockerfile:33`) on 8.6 with the W4A16 drafter, and then `:130-131` leaves `FA2_FP8KV` at 0. So the Docker default on a 3090 is `CUSTOM` int8, not the FA2 fp8 arm] | 7 | 131072 | 4 | 5583457484 B | pool 136,429 tok int8 w/ prefix cache (measured, `docs/long-context.md:128` [review: was `:~125`]) | `:225-260`, `:297-359`, `:503-507`; defaults `resolve_config.sh:87-149` |
| `SPEC=dflash2 CTX=huge` | KVarN | 7 or 15 | 245760 (221184 if k>7) | 2 | 5261334938 B | pool 268,169 tok (measured, `baseline.md` "KV pools"); k>7 + prefix cache pins `VLLM_DFLASH2_LOOKUP_ADAPTIVE=0` (`:420-428`); retention interval 13056 (k=7) / 14592 (k=15) (`:742-745`) | `:261-266`, `:488-495` |

Cross-cutting single-user behaviour:

- Speculative config strings: dflash2 at `start_qwen.sh:394` (`method` defaults `dflash`, `draft_sample_method` default `probabilistic`), mtp at `:666`. `probabilistic` is not optional: unset it loses the draft denominator (2.66 vs 3.23 tok/step at k=15, measured, comment `:383-390`).
- `ASYNC_SCHED` is **not** off in single-user mode generally. The default is on (`:851-852`). It is switched off only when lookup is on and `DRAFT_TOKENS>7` (modes C, E at k=15; `:430-435`), because the adaptive verify length needs the synchronous scheduler. The prior map said single-user turns it off; that holds for k=15 only.
- `CG` (max CUDA graph capture size) = `min(64, MAX_SEQS*(k+1))` for dflash2 (`:560`), 32 for mtp/off.
- Resident-seat geometry is printed by the calculator at `:588-626`.
- Final argv at `:1025-1046`: `--max-num-batched-tokens 2048`, `--mamba-ssm-cache-dtype float16`, `--reasoning-parser qwen3`, `--compilation-config {"max_cudagraph_capture_size":CG,"custom_ops":["+rms_norm","+silu_and_mul"]...}`, tool parser `qwen3_coder` with `--tool-strict-level parameter` (`:869-877`), `VLLM_USE_FLASHINFER_SAMPLER=0` (`:1020`).
- `DYNAMIC_K` (per-batch-size draft count) is refused at `CTX=huge` with prefix caching (`:688-693`).
- `alternative.sh` (170 lines): int4 per-token-head KV on TRITON_ATTN, 256000 ctx, `GPU_UTIL 0.95`, `MAX_SEQS 1`, `SPEC=dflash2|off`, `INT4_MQ_3D`.

### 4.3 Batch profile matrix (`batch/start_qwen.sh`)

| `KV=` | Backend | MAX_LEN | GPU_UTIL | Other | Lines |
|---|---|---|---|---|---|
| `fp8` (default) | fp8 KV, vLLM picks backend | 150000 | 0.94 | pool 242,010 tok at 0.93 (measured, `baseline.md`) | `:107,130,149` |
| `kvarn` | KVarN | 262144 | 0.93 | `KVARN_POOL_MEM_FRAC=0.25` | `:122-128` [review: was `:109-120`] |
| `int4pth` | TRITON_ATTN, `int4_per_token_head` | 262144 | 0.93 | | `:118-121` [review: was `:123-124`] |
| [review] WSL2 overrides | | kvarn 131072 | fp8/int4pth 0.91, kvarn 0.88 | applied when `GPU_UTIL`/`MAX_LEN` are unset and WSL is detected; the GPU box is WSL2, so its batch default is 0.91, not 0.94 | `:105-117` |

Common: `MAX_SEQS=64` (`:87`), `INT8_ACT=int8`, `INT8_LAYERS=mlp` (`:154-155`), `--async-scheduling` (`:311`), `--max-num-batched-tokens 2048` (`:312`), `--mamba-ssm-cache-dtype float16` (`:310`), compilation config with `max_cudagraph_capture_size 64` (`:313`), no speculative decoding.

## 5. Config knobs

Defaults are those in the launcher unless noted. "Single" = `single-user/start_qwen.sh`, "Batch" = `batch/start_qwen.sh`.

| Knob | Default | Effect | Where |
|---|---|---|---|
| `CTX` | `fast` | Picks KV dtype, backend, max len, k (single only) | Single `:185`; validated `resolve_config.sh:56-64` |
| `SPEC` | `mtp` (launcher), `dflash2` (`.env.example`) | `mtp`, `dflash2`, `off`, `none`; anything else refuses | Single `:197`; `.env.example:6` |
| `KV` | `fp8` | Batch KV mode: `fp8`, `kvarn`, `int4pth` | Batch `:107` |
| `MAX_LEN` | per profile | `--max-model-len`; precedence `DFLASH_MAX_LEN` > `MAX_LEN` > profile | Single `:205`, `:491` |
| `MAX_SEQS` | 8 single (4 or 2 on some dflash2 profiles), 64 batch | Admission cap, not residency | Single `:122`, `:488-520`; Batch `:87` |
| `GPU_UTIL` | 0.93 single, 0.94 batch(fp8) [review: 0.91 batch fp8/int4pth, 0.88 batch kvarn on WSL2] | `--gpu-memory-utilization` | Single `:175`; Batch `:149`, `:113-116` |
| `KV_MEM` | 5583457484 (fast/long dflash2), 5261334938 (huge) | Pins the KV pool in bytes (`--kv-cache-memory`) for deterministic boots | Single `:548`, `:672` |
| `DRAFT_TOKENS` | 4 fast, 3 long/huge | Spec length for mtp | Single `:212`, `:215-223` |
| `DFLASH_TOKENS` | 7 | dflash2 verify block (`DRAFT_TOKENS` is set from it); 15 = "reproduction mode" | Single `:313`, `:382` |
| `DRAFT_METHOD` | `dflash` | `dspark` = alternative drafter | Single `:394` [review: was `:308`, a PTH_FA guard] |
| `DRAFT_SAMPLE` | `probabilistic` | Draft sampling; `greedy` costs about 15% at T>0 | Single `:394`, `:666` |
| `LOOKUP` | 1 | Context n-gram lookup drafting (`VLLM_DFLASH2_LOOKUP`) | Single `:370` [review: was `:319`, a PTH_FA guard] |
| `LOOKUP_ADAPTIVE` | auto | Adaptive verify block length; auto-pinned off at huge+k>7+prefix cache | Single `:422`, `:420-428` |
| `DYNAMIC_K` | unset | `num_speculative_tokens_per_batch_size` e.g. `[[1,3,7],[4,512,3]]`; refused at huge+prefix cache | Single `:683-693` [review: was `:324`, a PTH_FA guard] |
| `SPEC_ATTN` | 1 | Split-KV verify attention (`VLLM_SPEC_DECODE_ATTN`), bf16 KV only | Single `:214` |
| `VLLM_SPEC_DECODE_ATTN_QMAX` | `k+1` | Sizes split-KV partial buffers once for graphs | Single `:398` |
| `ASYNC_SCHED` | 1; 0 when lookup and k>7 | Async scheduler | Single `:435`, `:851-852` |
| `PTH_FA` | 0; resolve_config sets 1 on dflash2/long sm86 | `CUSTOM` int8 per-token-head backend (target and drafter) | Single `:297-359`; `resolve_config.sh:87-117` |
| `FA2_FP8KV` | 0; resolve_config sets 1 on dflash2/long sm86 only when the `PTH_FA` default did not fire [review: `resolve_config.sh:130-131`; in the image pthfa is installed, so on a 3090 `PTH_FA=1` wins] | fp8 KV on FLASH_ATTN via `fa2/` plugin | Single `:247-260`; `resolve_config.sh:118-149` |
| `INT8_ACT` | `` (single), `int8` (batch) | W4A8 Marlin (int8 activations) | Single `:156`; Batch `:154` |
| `INT8_LAYERS` | `mlp|linear_attn|self_attn` (single), `mlp` (batch) | Which linears get int8 activations; `mlp|linear_attn` crashes (inductor) | Single `:157`; Batch `:155` |
| `PREFILL_ATTN` | unset | `int8` = int8-QK Triton prefill attention (hd256), companion to `INT8_ACT` | Single `:171-172` |
| `PREFIX_CACHE` | 1 | `--enable-prefix-caching --mamba-cache-mode align` or off | `launcher_common.sh:166-181` |
| `PREFIX_RETENTION` | auto (13056 at k=7, 14592 at k=15, huge only) | `--prefix-cache-retention-interval` | Single `:737-753` |
| `KV_OFFLOAD_GB` | unset | CPU KV tier (`--kv-offloading-size`); turns FA2_FP8KV default off | `launcher_common.sh:156-160` |
| `CUDAGRAPH_MODE` / `CG_MODE` / `CG` | PIECEWISE for huge non-dflash2; else vLLM default | CUDA graph mode and capture size | Single `:834`, `:841`, `:560` |
| `KVARN_POOL_MEM_FRAC` | 0.15 single, 0.25 batch | KVarN side-pool fraction | Single `:219`; Batch `:109-120` |
| `KVARN_FP16_DEQUANT`, `KVARN_SHARED_VERIFY` | off | KVarN decode variants | `kvarn/README.md`, `kvarn/kvarn-fp16-dequant-0.31.0.patch` [review: was `patches/kvarn/...`; the KVarN patches live in `kvarn/`, `patches/apply.sh:48,53-54`] |
| `VISION`, `VISION_OFFLOAD` | 0, 1 | Keep the vision tower; offload it to host | Single `:913`, `:929` |
| `EMBED_UVA` | 0 | Quantized embedding in pinned host RAM (about 1.25 GiB VRAM back, +15.7% pool) | Single `:938`; `patches/qwen3_5-embed-uva.patch` |
| `TOOL_PARSER`, `TOOLS`, `TOOL_STRICT` | `qwen3_coder`, 1, `parameter` | Tool calling | Single `:869-877` |
| `REQ_METRICS`, `REQ_METRICS_DETAILED` | 0 | Per-request timing in responses | Single `:888-894` |
| `PORT`, `BIND`/`HOST` | 18020, 127.0.0.1 [review: 0.0.0.0 inside a container or with a key set, `launcher_common.sh:100-104`; Compose publishes on `${BIND:-127.0.0.1}`, `docker-compose.yml:84`] | Listener; no key plus public bind warns | Single `:121`; `launcher_common.sh:92-109` |
| `VLLM_API_KEY` / `api_key.txt` | none | `--api-key`; deny-by-default middleware | `resolve_api_key.sh`; `patches/auth-deny-default.patch` |
| `EXTRA_ARGS` | empty | Appended to `vllm serve`; shadowing warnings in `resolve_config.sh:163` | Single `:331`, `:1044` |
| `SSE_KEEP_ALIVE` | 30 [review: was "unset"; `SSE_KEEP_ALIVE=${SSE_KEEP_ALIVE-30}` at Single `:129`] | `--sse-keep-alive-interval` | Single `:129`, `:1046` |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | 1 in `.env.example` | Required on WSL2 for dflash2 | `.env.example:45`, Single `:981` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments` (WSL variants differ) | Allocator | Single `:985` |
| `PREPARE`, `VERIFY`, `FAST_VARIANT`, `DFLASH2` | 1, 1, 1, 1 | Entrypoint stages and downloads | `docker/entrypoint.sh`, `docker/prepare.sh:66` |
| `VLLM_MARLIN_TUNE`, `VLLM_MARLIN_REPACK_STAGED` | off | Locally tuned Marlin tiles (+0.4% e2e measured, `docs/optimizations.md:198-203`); staged sm80 repack | `patches/marlin-tune-table.patch`, `marlin-repack-staged-sm80.patch` |

## 6. vLLM patches (`patches/series`, apply order, `--fuzz 0`)

Order is a hard requirement (`patches/series:1-15`). "Subsystem" is my grouping. "Touches" is the vLLM file list from the patch's `+++` lines. KVarN patches follow the series via `patches/apply.sh --kvarn`.

| # | Patch | Subsystem | What it changes | Touches |
|---|---|---|---|---|
| 1 | qwen3_5-embed-quant | Model / quant | Pass `quant_config` to token embedding (target and MTP) so vLLM's int8 embedding gather runs | `models/qwen3_5.py`, `qwen3_5_mtp.py` |
| 2 | marlin-int8-layer-select | Quant | Env vars choose which layers run W4A8; keeps int8-weight lm_head off it | `envs.py`, `quantization/utils/marlin_utils.py` |
| 3 | marlin-int8-negative-scales | Quant | Fold sign of negative group scales into int4 codes (kernel reads scales unsigned) | `kernels/linear/mixed_precision/marlin.py` |
| 4 | qwen3_5-mtp-draft-vocab | Spec / model | 40,960-row draft head for MTP | `models/qwen3_5_mtp.py` |
| 5 | vllm-pr50021-gdn-spec-bounds | GDN | Bounds checks in GDN/KDA spec-decode state lookups (vLLM #50021) | `mamba/ops/causal_conv1d.py`, `mamba_ssm.py`, `fla/ops/fused_recurrent.py`, `fused_sigmoid_gating.py` |
| 6 | sampler-small-topk-fast-softmax | Sampler | Sort-free top-k/top-p for small k, multi-block row softmax, drafts sampled from truncated support | `v1/sample/*`, `spec_decode/llm_base_proposer.py`, `gpu_input_batch.py` |
| 7 | spec-decode-attn | Verify attention | Split-KV Triton verify on FLASH_ATTN with query-row tiling (`_spec_attn_partial` `:212`, `_spec_attn_combine` `:287` in the patch) | `backends/flash_attn.py`, `ops/spec_decode_attn.py`, `envs.py` |
| 8 | speed-knobs-envs | Env | Registers this repo's env knobs | `envs.py` |
| 9 | hybrid-kv-groups-v2-cudagraph | KV groups | KV group sizing when the smallest bucket is the drafter's SW layers; V2 CUDA-graph memory explicit | `v1/core/kv_cache_utils.py` |
| 10 | dflash2-lookup-drafting | DFlash2 | Context suffix lookup (`_suffix_lookup_kernel` `:346`), draft fusion (`_fuse_draft_kernel` `:417`), point-mass draft logits (`:504`), `next_num_draft_tokens` (`:948`), `_apply_lookup` (`:1071`), fused context-KV precompute | `models/qwen3_dflash.py`, `qwen3_dflash2.py`, `v1/worker/gpu/{cudagraph_utils,model_runner}.py`, `spec_decode/dflash/speculator.py`, `dflash2/{lookup,speculator}.py` |
| 11 | hybrid-sw-block-promote | KV groups | Promote a SW layer's block to a divisor of the primary block instead of padding the page | `kv_cache_utils.py` |
| 12 | spec-decode-int8-kv | Verify attention | Split-KV verify over an int8 per-token-head cache | `flash_attn.py`, `triton_attn.py`, `spec_decode_attn.py` |
| 13 | vision-tower-cpu-offload | Model | Vision tower bulk weights in host RAM | `models/qwen3_vl.py` |
| 14 | int4-kv-per-token-head | KV / attention | int4 per-token-head KV with the DFlash2 drafter | `ops/int4_per_token_head.py`, `triton_unified_attention.py` |
| 15 | marlin-repack-staged-sm80 | Quant | One grow-only staging buffer for sm80 repack (opt-in) | `marlin.py` |
| 16 | offload-dflash-eagle-groups | KV offload | OffloadingConnector under dflash flagged every KV group as draft attention | `kv_connector/v1/offloading/scheduler.py` |
| 17 | dflash2-ngram-chains | DFlash2 | Quantized candidate chains, `propose` override | `dflash2/speculator.py` |
| 18 | offload-wsl2-devptr | KV offload | CPU-tier device pointers on WSL2 | `kv_offload/cpu/gpu_worker.py` |
| 19 | spec-decode-int4-kv-mq3d | Verify attention | Multi-query 3D int4 verify path | `int4_per_token_head.py`, `triton_unified_attention.py` |
| 20 | dflash2-prewarm | DFlash2 | Compile every DFlash2 rung at boot | `dflash/speculator.py` |
| 21 | marlin-tune-table | Quant | Wiring for a locally built tunable Marlin extension (off) | `_custom_ops.py` |
| 22 | prefill-attn-int8 | Prefill | int8-QK Triton prefill attention for head_dim 256 | `flash_attn.py`, `prefill_attn_hd256.py` |
| 23 | spec-sampler-prewarm | Sampler | Compile rejection-sampler Triton kernels at boot | `warmup/kernel_warmup.py` |
| 24 | mamba-align-checkpoint-order | GDN / prefix cache | Keep reachable Mamba state snapshots alive to request end | `single_type_kv_cache_manager.py` |
| 25 | spec-decode-scratch-token-units | Verify attention | mq3d scratch sized in tokens, not sequences | `triton_attn.py`, `int4_per_token_head.py`, `triton_unified_attention.py` |
| 26 | mamba-chunked-prefill-align | GDN / prefix cache | State loss and NaN during chunked prefill | `fla/ops/chunk_o.py`, `v1/worker/mamba_utils.py` |
| 27 | spec-decode-scratch-within-budget | Verify attention | mq3d scratch allocated inside the memory budget | `triton_attn.py` |
| 28 | dflash2-z-adaptive-emitted | DFlash2 | Adaptive z counts emitted tokens, not sampling slots | `dflash/speculator.py`, `dflash2/speculator.py` |
| 29 | dspark-draft-quant-config | Spec / model | bf16 DSpark drafter beside a quantized target | `models/utils.py` |
| 30 | engine-completion-log | Engine | One log line per completed engine step | `sched/scheduler.py` |
| 31 | engine-stall-sentinel | Engine | Daemon warns when no step completes for `VLLM_ENGINE_STALL_SENTINEL_S` | `v1/engine/core.py`, `envs.py` |
| 32 | triton-spec-attn-fp8-kv | Verify attention | Split-KV verify on per-tensor fp8 KV (TRITON_ATTN, sm89+) | `flash_attn.py`, `triton_attn.py`, `spec_decode_attn.py`, `envs.py` |
| 33 | topk-honour-flashinfer-sampler-switch | Sampler | `VLLM_USE_FLASHINFER_SAMPLER=0` also covers drafter top-k | `layers/logits_processor.py` |
| 34 | memory-profile-after-warmup | Memory | `profile_run` before the profiling window so cold-compile scratch is not counted | `v1/worker/gpu_worker.py` |
| 35 | cudagraph-memory-from-allocator | Memory | Measure captured graph memory by allocator reserved bytes (WSL2 driver delta reads 0) | `gpu/cudagraph_utils.py`, `gpu/model_runner.py` |
| 36 | bench-probe-errors | Bench | `vllm bench serve` probe sends the API key and classifies failure | `benchmarks/serve.py` |
| 37 | serve-404-served-names | API | 404 lists served names (vLLM #59889) | `entrypoints/serve/engine/serving.py` |
| 38 | serve-model-path-match | API | Accept model root path or basename (#59890) | `entrypoints/openai/models/serving.py` |
| 39 | tokenize-v1-route | API | `/tokenize`, `/detokenize` also under `/v1` | `serve/tokenize/api_router.py` |
| 40 | marlin-int8-asym-zp | Quant | W4A8 accepts zero-point uint4 (asymmetric AWQ exports) | `marlin.py`, `marlin_utils.py` |
| 41 | compile-key-runtime-knobs | Compile | Runtime-only knobs stay out of `compile_factors()` | `envs.py` |
| 42 | auth-deny-default | API | `--api-key` guards every path except health/ping/load/version | `launchers/cli_args.py`, `middleware/authenticate.py` |
| 43 | spec-attn-smem-fit | Verify attention | Halve KV tile on `OutOfResources` (Turing) | `spec_decode_attn.py` |
| 44 | bench-sse-keepalive | Bench | SSE keep-alive no longer fails bench requests | `benchmarks/lib/endpoint_request_func.py` |
| 45 | sampler-warmup-cuda | Sampler | Register V2 top-k/top-p JIT warmups on CUDA (vLLM #58092) | `gpu/model_runner.py` |
| 46 | pinned-kv-empty-cache | Memory | With `KV_MEM` pinned, empty allocator cache after profile run (#59893) | `gpu_worker.py` |
| 47 | api-root-health | API | `/`, `/api/status`, `/api/show` for Ollama-style clients | `launchers/app.py`, `authenticate.py` |
| 48 | qwen3_5-embed-uva | Model / memory | `VLLM_EMBED_UVA=1` pinned-host embedding via UVA | `qwen3_5.py`, `qwen3_5_mtp.py`, `envs.py` |
| 49 | vllm-pr40371-prompt-progress | API | Opt-in `prompt_progress` on streams (#40371) | `chat_completion/*`, `completion/*`, `outputs.py`, `sched/scheduler.py`, ... |
| 50 | vllm-pr61050-lru-running-prefix | KV offload | CPU tier keeps a running request's head chunks (#61050) | `kv_offload/cpu/manager.py`, `policies/*` |
| 51 | kv-cache-seat-budget | KV / diagnostics | One INFO line with resident requests at 1k, 32k, max_len | `kv_cache_utils.py` |
| K1 | kvarn-0.31.0 | KVarN | KVarN cache dtypes, quant mode `KVQuantMode.KVARN = 11`, backend, page size | (via `kvarn/`) |
| K2 | kvarn-v2-runner-0.31.0 | KVarN | KVarN on the V2 runner with DFlash2 (SW groups, Mamba block index, selector guards) | |
| K3 | kvarn-recycled-pages-0.31.0 | KVarN | Runners hand KVarN each step's block ids so it drops pages another group took (#208) | |
| K4 | kvarn-fp16-dequant-0.31.0 | KVarN | Registers `KVARN_FP16_DEQUANT` | `envs.py` |

Retired and removed: `offload-mtp-serve`, `mamba-align-retire-null-gaps` (0.30.0); `vllm-pr54282-draft-gumbel-salt`, `xgrammar-spec-terminated`, `sse-keep-alive` (0.29.0); `dflash2-backport` (native since 0.28.0) (`PATCHES.md:90-99` [review: was `:97-112`]); `int4-mq3d-envs` retired temporarily on 0.29 (`PATCHES.md:101-106`) [review: added].

## 7. Custom kernels

| Kernel | Where | Purpose | Limits |
|---|---|---|---|
| Split-KV verify attention (bf16) | `patches/spec-decode-attn.patch` (Triton, `_spec_attn_partial`/`_combine`) | FA2 splits KV across blocks only for q=1; on a multi-query verify only 24 of 82 SMs worked [review: the doc's case is the MTP verify with five queries, 24 heads, `docs/optimizations.md:91-94`]. Replaced by Triton split-KV (23 us vs 57 us per layer at 1.5k; 120 us vs 1.3 ms at 16k, measured, `docs/optimizations.md:90-95`) | `CTX=fast` only (bf16 KV on FLASH_ATTN) |
| Split-KV verify (int8, fp8, int4 mq3d) | `spec-decode-int8-kv`, `triton-spec-attn-fp8-kv`, `spec-decode-int4-kv-mq3d` patches | Same on quantized caches. At 128k, 8 queries, per layer: 1.3 ms vs 7.4 ms (vLLM unified) vs 10.1 ms (FA2) (measured, `docs/long-context.md:145-146` [review: was `:~145`; that passage does not state the hardware, so the "3080 TP=2 context" label is unsupported and removed]) | TRITON_ATTN |
| Sort-free top-k/top-p + multi-block softmax | `sampler-small-topk-fast-softmax.patch` (`_partial_kernel` `:63`, `_final_kernel` `:84`) | Removes 248k-wide sort and one-block-per-row softmax (140 us per row, measured, `docs/optimizations.md:96-98`) | top-k <= 64 |
| DFlash2 lookup kernels | `dflash2-lookup-drafting.patch:346,417,504` | Suffix match over request history, fuse with drafter proposal, point-mass draft logits | Triton, one program per request |
| int8-QK prefill attention | `prefill-attn-int8.patch`, `backends/prefill_attn_hd256.py` | QK on int8 tensor cores for head_dim 256 prefill | single-request prefill chunks on exact serving geometry |
| Marlin W4A8 (int8 activations) | vLLM's kernel plus patches 2, 3, 40, 21 | int8 tensor cores for GEMMs | `INT8_LAYERS` set |
| pth-flash-attn | `pthfa/csrc/pth_attn/` (about 6.5k lines of CUDA: `attn_kernel.cuh` 779, `attn_decode.cuh` 565, `launch.cu` 554, `combine_kernel.cuh` 305, ...) and `pthfa/python/pth_flash_attn/` (`vllm_backend.py` 794, `planner.py` 379) | int8 per-token-head KV attention, `AttentionBackendEnum.CUSTOM` via `vllm.general_plugins` entry point (`pthfa/pyproject.toml:27`) | sm80/86/89; head dims 64/128/256; `int8_per_token_head` only; windowed/non-causal at hd128 only (`pthfa/README.md`) |
| FA2 fp8-KV | `fa2/fp8_attn.cu` (189 lines), `prefill.cu` (80), `fa2-fp8kv.patch` (39.9 KB) | fp8 e4m3 KV on FlashAttention-2 for head dims 128/256, split-KV, GQA packing | sm_86 build; only `SPEC=dflash2 CTX=long` (`resolve_config.sh:143-144`) |
| KVarN | `kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py` (1274), `triton_kvarn_sinkhorn.py` (205), `kvarn_store.py` (286), `backends/kvarn_attn.py` (2875), `layers/quantization/kvarn/config.py` (446) | Port of Huawei CSL KVarN: Hadamard rotation + Sinkhorn-style variance normalization, 4-bit K / 2-bit V per 128-token tile | `kvarn_k4v2_g128`, block size 128; verify kernel is 78% of E's step at 170k (measured, `e-profile.md`) |
| GDN bounds fixes | patch 5, patch 26 | Illegal-memory-access and chunked-prefill state fixes in DeltaNet kernels | |

## 8. Bench and verify tooling

| Tool | What |
|---|---|
| `verify.sh` | Sections: environment (`:43`), vLLM patches (`:77`), KVarN (`:114`), fp8-KV FA2 (`:148`), model (`:182`), fast variant (`:336`), tokenizers (`:344`), DFlash2 drafter (`:369`), keys/units (`:376`). `make verify` runs it in the running Compose service |
| `bench/run_benchmarks.sh` | Canonical protocol (`single`, `batch`, `--long`); seeds every call (fixes prefix-cache replay, `docs/optimizations.md:175-182`) |
| `bench/harness.py`, `conc_ladder.py` | Cohort/concurrency drivers |
| `bench/labd_bench.py` | Six-task decode bench (copy, code, edit, quote, summary, qa) at a context length; used for modes C and E |
| `bench/t4_battery.py` | T4 battery: GSM8K, perplexity, residue sweep, needles, per arm |
| `bench/quality_battery.py` | Perplexity and GSM8K from parquet (`quality-data/`) |
| `bench/residue_sweep.py`, `bugb_sweep.py`, `needle_test.py`, `seat_ttft.py`, `spec_attn_ctx_scan.py`, `prefill_ab.sh`, `warmup.sh`, `api_smoke.py` | Targeted probes |
| `bench/test_*.py|sh` (about 25) | CPU-capable unit tests for knobs, kernels (`test_lookup_kernels.py`, `test_pthfa_attn.py`, `test_spec_decode_attn.py`), quant schema, prepare crash |
| `bench/mq3d_*` | mq3d scratch oracle and property tests |
| Exit codes | 0 pass/measurement, 1 FAIL, 2 invalid run, 3 negative control failed (`bench/README.md`) |
| `scripts/` | `hq-doctor.sh` (`make doctor`), `export-patch.sh`, `pin-bump.py`, `port-*` (vLLM bump tooling), `check_verify_graded.py` |
| Expedition harness | `docs/research/04-benchmarks/harness/` (bind-mounted into the stock image, `decisions.md` D4) |

## 9. CI (`.github/workflows/`)

| Workflow / job | What it gates |
|---|---|
| `docker-image.yml` | Builds the image on PRs (quality gate = the Dockerfile's patch apply, `verify.sh --install`, CPU gate tests); pushes `ghcr.io/syv-ai/hyperqwen:latest` and `sha-<7>` on main; weekly cron Mon 04:17 |
| `patch-integrity.yml` `model-verification` | CPU knob and test scripts |
| `kvarn-torch-gate` | KVarN against the pinned torch |
| `patch-index` | `check_verify_graded.py`, `patches_md.py` freshness, `test_export_patch.sh`, `pthfa/check_vendor.sh` |
| `fa2-upstream` | `fa2/check_upstream.py` against the pinned flash-attention commit |
| `git-apply` | Series applies with `git apply` via `check_vllm_series.sh` against the pinned vLLM tag |
| `prepare-crash` | `test_quant_schema.py`, `test_prepare_crash.py` |

No GPU job: every performance claim in the docs is a manual run.

## 10. Docs index

| Doc | Lines | Subject |
|---|---|---|
| `docs/optimizations.md` | 418 | What this repo adds to stock vLLM; per-step gain tables; negative results |
| `docs/long-context.md` | 822 | CTX=long/huge, fp8/int8/KVarN, DFlash2 past 64k, concurrency and residency, 3080 TP=2 ladders |
| `docs/gotchas.md` | 1551 | Numbered operational and correctness gotchas |
| `docs/vllm-0.29.md`, `vllm-0.30.md`, `vllm-0.31.md` | 438, 239, 184 | Pin-bump port notes and known issues |
| `docs/spec-decode-scratch-token-units.md` | 383 | mq3d scratch sizing |
| `docs/multi-gpu.md` | 355 | TP/solo setups |
| `docs/wsl2-4090.md`, `ubuntu-3090.md`, `python-314.md` | 567, 154, 83 | Platform notes |
| `docs/docker.md`, `install.md`, `quickstart.md` | 261, 166, 65 | Install paths |
| `docs/third-party-checkpoints.md` | 261 | Other checkpoints |
| `docs/benchmarks.md`, `quality.md`, `clients.md`, `main-track.md` | 65, 61, 63, 56 | Bench index, quality numbers, client setup, tracking |
| `docs/reproductions/` | README 261, `native-3090` 275, `a5000-230w` 327, `cmp-170hx-64gb` 106, `5070ti-16gb-tp2` 63 | Reproductions on other boxes |
| `single-user/README.md`, `batch/README.md`, `bench/README.md`, `kvarn/README.md`, `pthfa/README.md`, `fa2/README.md`, `drafter/README.md` | | Per-area docs |
| `PATCHES.md` | 112 | Generated patch table |
| `docs/research/` | | This expedition; prior art at `~/research-src/deepinfer-ref/docs/research/deepinfer/` |

## 11. Doc and code discrepancies found

| # | Finding | Evidence |
|---|---|---|
| 1 | `SPEC` default is `mtp` in the launcher but `dflash2` in `.env.example` and README prose | `single-user/start_qwen.sh:197`, `.env.example:6` |
| 2 | Tree lacks upstream's `PTH_FA` default for `SPEC=mtp CTX=long` | `resolve_config.sh:99-104` vs `~/research-src/HyperQwen/resolve_config.sh:105-107` |
| 3 | Prior map says single-user disables async scheduling; true only for lookup with k>7 | `start_qwen.sh:430-435`, `:851-852` |
| 4 | `docs/optimizations.md:3` says "nine things" and lists ten | `docs/optimizations.md:3,11-41` |
| 5 | `single-user/README.md:126` still says the drafter's prefill adds about 15% TTFT; `docs/optimizations.md:186-189` says that figure is stale | both files |
| 6 | README "E: 240k" vs launcher `CTX=huge` default 200000 (the dflash2 branch overrides to 245760/221184) | `start_qwen.sh:216` vs `:488-495` |
| 7 | Prior map's cites `batch/start_qwen.sh:311-312`, `single-user/start_qwen.sh:435, 845-852`, `:399-428` check out against our tree | verified |

## Reviewer additions

[review: added by the phase-1 project-map review, `docs/research/00-meta/reviews/phase1-project-map.md`]

| # | Finding | Evidence |
|---|---|---|
| R1 | vLLM pin 0.31.0 is enforced in four places: `docker/requirements.txt:5`, `pthfa/PIN` (`VLLM_VERSION=0.31.0`), `fa2/PIN` (`VLLM_VERSION=0.31.0`), and CI checks requirements against `fa2/PIN` (`.github/workflows/patch-integrity.yml:96-100`). The reference tree `~/vllm` is v0.32.1rc0, one minor ahead: a line cite there is not a line of the code we ship | files named |
| R2 | `README.md:96-101` still says `SPEC=dflash2 CTX=long` defaults to the FA2 fp8 plugin. At `95ac251`, `resolve_config.sh:95-116,130-131` puts `PTH_FA=1` first when pthfa is installed (always, in the image), and `.env.example:30-32` says so. A doc-vs-code discrepancy to add to section 11 | files named |
| R3 | Compose runs a separate `prepare` service first: `single`/`batch`/`solo` `depends_on: prepare: condition: service_completed_successfully` (`docker-compose.yml:85-87,107-109`), and the entrypoint runs `prepare.sh` again (idempotent). For `single` the entrypoint also sources `single-user/select_model.sh` before `verify.sh` (`docker/entrypoint.sh:26-30`) | files named |
| R4 | Mode B's pool on 0.31 with the default `PREFIX_CACHE=1` is 68,605, not 69,758 (cache off). Both numbers are measured: `single-user/README.md:170-171`; prior GPU box `baseline.md:49` reads 68,605 too, although its prose says "all equal the documented pools" | files named |
| R5 | `docs/research/01-project/weights_breakdown.out` (safetensors headers of the HF repos, measured; resident sums are estimated) already gives resident weights per mode: A 14.14 GiB, D (fast) 13.85 GiB, B/C/E with the W4A16 drafter 14.74 GiB. This is better evidence than the 15 GB directory size used in architecture.md | `weights_breakdown.out` section 4 |
