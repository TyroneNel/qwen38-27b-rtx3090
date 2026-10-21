# EAGLE-4: build, dependencies, kernels and engine integration

Paths are relative to `~/research-src/EAGLE-4` at `7b7b5a5` unless prefixed with `~/vllm` (vLLM checkout) or "ours" (`/home/tyrone/qwen38-27b-rtx3090`). Every statement about code is **measured**, meaning read from the file cited. Nothing was installed, built or run on a GPU, so any statement about runtime behaviour on sm86 is **estimated** from the code.

## 1. Build system and packaging

| Item | Value | Source |
|---|---|---|
| Package | `eagle-harness` 0.1.0, setuptools ≥ 77, licence `LicenseRef-Eagle-4-Source-Available` | `pyproject.toml:1-11` |
| Python | ≥ 3.10. Release wheels and images use 3.12 | `pyproject.toml:12`, `docs/install.md:43-52` |
| Hard deps | `numpy`, `packaging`, `torch` (no version pins) | `pyproject.toml:13` |
| Engine extras | `vllm>=0.32.0rc1.dev283,!=0.32.0`; `sglang>=0.5.23.dev0,<0.6` + `flashinfer-python>=0.7.2`; `tokenspeed>=0.1.1,<0.2` | `pyproject.toml:15-22` |
| Engine hook status | "No host release has the hooks yet (sglang v0.5.22 and vLLM v0.32.0 are the last without them)" | `pyproject.toml:16` |
| Plugin entry points | `vllm.general_plugins` → `eagle_harness.vllm.plugin:register`; also sglang and TokenSpeed groups | `pyproject.toml:29-42` |
| Package data | `csrc/*.cu`, `_native/*/*.so`, profiles (`*.json/*.csv`) | `pyproject.toml:47-49` |
| Wheel tagging | pure `py3-none-any` unless prebuilt `.so` files exist, then `py3-none-<platform>` | `setup.py:4-6, 14-28` |
| Released engine builds | vLLM `0.32.0rc1.dev283+eagle.4`, SGLang `0.5.23.dev252+eagle.4`, TokenSpeed `0.1.1+eagle.4`, from `https://eagle-hq.github.io/EAGLE-4/whl/cu130/` | `docs/release.env:5-24` |
| Torch index | `download.pytorch.org/whl/cu130` | `docs/release.env` (`EAGLE_TORCH_CUDA_INDEX`) |
| vLLM image | "Sketch only": `FROM vllm/vllm-openai:nightly-49cb2bbf…`, then `pip install vllm==$EAGLE_VLLM_VERSION` from the eagle index | `docker/vllm.Dockerfile:5-21` |
| Load-time check | `eagle-harness-check` / `compat.py` holds the same version range and calls `adapter.missing_hooks` | `eagle_harness/compat.py:52-55` |

What this means for us:
- The engine is not stock vLLM. It is a patched vLLM fork published as `+eagle.4` wheels for CUDA 13.0.
- Our image pins vLLM 0.31.0, torch 2.13, cu130 and Triton 3.7.1, with a 50-patch series (ours `Dockerfile:2-5`). [review: `Dockerfile:1-5` gives no count; `patches/series` lists 51 entries (measured).] The CUDA major and the Python version match. The vLLM version does not (§3).

## 2. Kernels

### 2.1 Tiers (from the code)

`eagle_harness/kernels/arch.py:11-17`:

| Tier | Hardware |
|---|---|
| Triton (every kernel) | NVIDIA sm_80+ (A100, H100/H200, B200/B300, sm_120), AMD CDNA2-4 |
| Native CUDA (extras) | NVIDIA only: candidate tail from sm_80, cluster shard top-k from sm_90 (else Triton) |
| PDL, Gluon GDN | NVIDIA sm_90+ / `GLUON_GDN_SM = {103}` (B300 only) |

- `MIN_NVIDIA_SM = 80` (`arch.py:30`).
- `unsupported_reason` refuses only `sm < 80` and AMD architectures outside gfx90a/942/950 (`arch.py:96-107`).
- **sm86 passes the gate.** 86 ≥ 80, and the gate only compares numbers.

### 2.2 Inventory

Languages were counted with `grep -c triton.jit`, `gluon`, and the arch-gate calls per file (measured).

| File | Language | Role | Gate | sm86 |
|---|---|---|---|---|
| `kernels/csrc/candidate.cu` (681 lines) | CUDA C++17, BF16 (`cuda_bf16.h`), loaded via ctypes | (a) shard top-k over the low-rank scores: clusters of 8 CTAs and distributed shared memory | `arch.clusters()` sm ≥ 90 (`arch.py:132-134`); the device code `__trap()`s below sm_90 (`candidate.cu:200-201`) | **No**: the gate routes sm86 to Triton (`candidate.py:45`) |
| same file | CUDA | (b) candidate tail: softmax + top-k over 128 candidates with warp shuffles | `native_candidate_tail()` sm ≥ 80 (`arch.py:127-129`, `candidate.py:87`) | **Yes, with the sm_80 SASS or a JIT build** (§2.3) |
| `kernels/candidate_triton.py` (12 `@triton.jit`) | Triton | Triton twin of (a) and (b): "the only implementation on AMD GPUs and NVIDIA before sm_90" | none | **Yes** (estimated) |
| `kernels/row_attention.py` (4) + `online_softmax.py` | Triton | draft attention for the K tree rows: window + ancestors, one key load for all rows | none | **Yes** (estimated) |
| `kernels/tree_step.py` (6), `accept_tail.py` (2), `verify_prep.py` (1), `gather_dot.py` (1) | Triton | tree growth bookkeeping, accept tail, verify layout, the exact `W_C·h` gather-dot | none | **Yes** (estimated) |
| `kernels/gdn_tree.py` (2), `gdn_tree_conv.py` (9), `gdn_tree_recurrence.py` (10) | Triton | GDN tree conv1d and tree gated-delta recurrence for verify | `arch.pdl()` sm ≥ 90 adds `gdc_wait`/`gdc_launch_dependents` (`gdn_tree_recurrence.py:255-256, 402-403, 640, 666, 729`; `gdn_tree_conv.py:481`) | **Yes without PDL** (estimated) |
| `kernels/gdn_tree_gluon.py` (737 lines) | Triton Gluon (`triton.experimental.gluon`) | faster bitwise twin of the recurrence | `arch.gluon_gdn()`: sm == 103 only (`arch.py:36, 142-145`). It is imported lazily (`gdn_tree_recurrence.py:508`) and a mismatch falls back to Triton (`gdn_tree_gluon.py:37`) | **No**: never runs, not needed |
| `kernels/gdn_tree_records.py` (4) | Triton | record/replay of the GDN tree walk | `gdn_records_replay()`: AMD only (`arch.py:153-157`) | n/a |
| `kernels/gdn_tree_chunk_flydsl.py` | FlyDSL (AMD `rocdl`) | chunked GDN tree on gfx950, imported lazily (`gdn_tree_records.py:523`) | AMD | n/a |
| `kernels/mamba2_tree.py` (3) | Triton | Mamba2 tree verify (Nemotron targets) | none | not used for Qwen |
| `kernels/l2_prefetch.py` (2) | Triton + inline PTX `prefetch.global.L2` | warms L2 with the next GEMM's weights on a side stream | `side_stream_prelaunch()`: any NVIDIA (`arch.py:160-165`) [review: wrong gate. `l2_prefetch` is used only by `sglang/l2_warm.py:36,149`, enabled by `HarnessConfig.l2_warm` (`core/config.py:70,122`; `sglang/worker.py:231`). `arch.py:160-165` is the SGLang next-round prelaunch. Not reachable from the vLLM plugin.] | **Yes**: the PTX instruction is not Hopper-specific. Benefit on a 6 MB-L2 3090 is unknown |
| `vllm/verify_attention.py`, `vllm/draft_rows.py`, `vllm/round_commit.py`, `vllm/verify_layout.py`, `vllm/depth_stop.py`, `vllm/gdn_tree_verify.py`, `vllm/tree_accept.py` | Triton | vLLM-side tree verify attention (BLOCK 64, rows padded to 16: `verify_attention.py:25, 166`), KV compaction, state commit, depth stop, accept walk | none | **Yes** (estimated) |

- **No FP8, WGMMA, TMA or tcgen05 anywhere.** A grep across `eagle_harness/` for `float8|fp8|e4m3|wgmma|tcgen|tensor_descriptor|tma|cluster_dims` finds none of these in kernel code (measured).
- The Triton kernels use `tl.dot` on BF16 tiles of ≥ 16, plus ordinary `num_warps`/`num_stages` (for example `row_attention` uses 4 warps and 3 stages). Triton lowers this to `mma.sync` on sm80/86. No kernel requests an explicit shared-memory size that would exceed the sm86 limit of 100 KB per SM (measured: no `shared=`/smem arguments found).
- Whether every autotuned config fits sm86 shared memory is **unverified** until compiled.

### 2.3 Native library build and load

- `kernels/build.py:39-42`:
  - `SASS_ARCHS = (80, 90, 100, 103, 120)` and `PTX_ARCH = 90`.
  - Flags: `-O3 -std=c++17 -shared -Xcompiler -fPIC -cudart static -diag-suppress 177`.
  - sm_103 needs CUDA ≥ 12.9 (`build.py:61-63`).
- The JIT compiles for the visible GPUs. Release wheels carry prebuilt `_native/cu<major>/*.so` with that SASS plus sm_90 PTX, and need glibc ≥ 2.34 (`docs/kernels.md:55-56`).
- Load order (`kernels/native.py:4-8, 74-88`): explicit path → prebuilt for the CUDA major → JIT with nvcc → otherwise the Triton kernels, "which give the same bits".
- `cuda_candidate_kernels: false` disables the native library (`docs/kernels.md`).
- **On sm86 (estimated):**
  - The sm_80 SASS loads, because SASS is binary-compatible within major 8 for minor ≥ the built minor.
  - The sm_90 PTX cannot JIT down to sm86.
  - So the prebuilt candidate tail runs as sm_80 code, or a JIT build produces sm_86 when nvcc 13 is present (our image has CUDA 13).
  - Worst case is the bit-identical Triton path. **sm86 is not listed** as qualified anywhere: `docs/support-matrix.md` and `README.md:190` list B300, H-series and MI355X results only.

## 3. Engine integration (vLLM) and version

- **Mechanism.** A general plugin (`vllm/plugin.py:31-67`) registers:
  - the `eagle_harness` speculative method
  - `LlamaForCausalLMEagle4`
  - the `CUSTOM` attention backend → `TreeFlashAttentionBackend`
- **Hooks the fork adds** (`vllm/adapter.py:25-71`, `HOOK_MAP`): speculative-method registry, `SpeculativeConfig.method_config`, the `prepare_verify_inputs`/`make_rejection_sampler`/`post_accept` hooks, `LinearAttnVerifyExtension.forward_spec_rows`, `suspend_drafting`, `verify_budget`, `native_decode_batch_sizes`. A host missing any of them is refused by `missing_hooks` (`adapter.py:74-76`). [review: HOOK_MAP (`adapter.py:30-71`) has 11 paths: registration, `method_config`, configure, `num_lookahead_tokens`, `DraftModelSpeculator.propose`, `prepare_verify_inputs`, `make_rejection_sampler`, `post_accept`, `forward_spec_rows`, `uses_varlen_verify_graphs`, `CUSTOM`. `suspend_drafting`, `verify_budget` and `native_decode_batch_sizes` are not in it; they are optional (`vllm/batch_policy.py:4-17`, `vllm/config.py:242-245`). On v0.31.0, 9 of the 11 are missing (`DraftModelSpeculator` and `CUSTOM` exist).]
- **Measured against our vLLM** (`git grep` in `~/vllm` at tag `v0.31.0` and at HEAD `0.32.1rc0-68`):
  - None of these exist: `register_speculative_method`, `SpeculativeMethodSpec`, `method_config`, `prepare_verify_inputs`, `make_rejection_sampler`, `post_accept`, `LinearAttnVerifyExtension`, `suspend_drafting`, `verify_budget`, `native_decode_batch_sizes`.
  - Present: the `CUSTOM` backend enum, `num_speculative_tokens_per_batch_size`, Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER`, `vllm/envs.py:306`), `qwen_gdn_linear_attn.py` and `sample/gumbel.py`.
  - So the hooks exist only in the `+eagle.4` fork, consistent with `pyproject.toml:16`.
- **Required runtime flags** (`docs/quickstart-vllm.md:56-71`): `VLLM_USE_V2_MODEL_RUNNER=1`, `--attention-backend CUSTOM`, `--linear-backend flashinfer_cutedsl`, TP=4 in the recipe, and the line "Model Runner V2 and CUSTOM attention are required".
  - Chain mode needs the draft on `FLASH_ATTN` (`docs/configuration.md:95-98`).
  - Quantized KV caches are not qualified (`docs/quickstart-vllm.md:92`, `docs/known-limits.md:48`), and tree verify raises on them (`vllm/tree_attention.py:69-70`).
- **`flashinfer_cutedsl`** is the BF16 linear backend in the recipe. In `~/vllm` it is gated to the SM100 family (`vllm/model_executor/layers/utils.py:104-111`; `vllm/utils/flashinfer.py:194-206`, `is_sm100a_supported`). On a 3090 the default linear backend must be used.
  - Whether the fork requires this flag for correctness or only for speed is **unverified**. The quickstart says only "required" for the recipe.
- **Native EAGLE3 path.** `vllm/eagle3.py` wraps stock `Eagle3LlamaForCausalLM` for harness row forwards, so the harness can drive EAGLE-3 drafts too.
  - Stock vLLM 0.31.0 `llama_eagle3.py` already reads `fc_norm`, `norm_output` and `layer_types: sliding_attention` (`~/vllm` `git show v0.31.0:vllm/model_executor/models/llama_eagle3.py:231-240, 252, 286`; `llama.py:185-219`). The EAGLE-4 config sets `fc_norm: true` and `norm_output: true` (HF `config.json`).
  - Its loader prefixes unknown tensors with `model.` (`llama_eagle3.py:420-445`), so `efficient_lm_head*` would be unexpected names. [review: mechanism wrong, conclusion right. The rule is `elif "lm_head" not in name: name = "model." + name`, so `efficient_lm_head*` keep their names and are looked up at the top level, where no such module exists; `AutoWeightsLoader(self)` then fails. They must still be dropped.]
  - The architecture string `LlamaForCausalLMEagle4` is not in the v0.31.0 registry (`registry.py:659-673`).
  - So loading the checkpoint as a plain EAGLE-3 chain drafter with a full lm_head, no harness, needs at least an `architectures` override and dropping two tensors. This is **estimated**; nothing was loaded.

## 4. Hopper/Blackwell/AMD-only features

| Feature | Arch | Where | Effect if absent |
|---|---|---|---|
| Thread-block clusters + distributed shared memory (shard top-k) | sm_90+ | `csrc/candidate.cu:190-283, 577`; `arch.py:132-134` | Triton top-k; same bits (`native.py:4-8`) |
| Programmatic dependent launch (`griddepcontrol`) | sm_90+ | `gdn_tree_recurrence.py:255-256, 665`; `gdn_tree_conv.py:481`; `arch.py:137-139` | launches serialise without overlap; tens of µs per round at most (estimated) |
| Gluon GDN recurrence | sm_103 (B300) only | `gdn_tree_gluon.py`; `arch.py:36` | Triton recurrence (the default everywhere except B300) |
| `flashinfer_cutedsl` BF16 GEMM | SM100 family | `~/vllm/vllm/utils/flashinfer.py:194-206` | use the default linear backend |
| FlyDSL chunked GDN, records replay | gfx950 / AMD | `gdn_tree_chunk_flydsl.py`, `arch.py:153-157` | not used on NVIDIA |
| FP8 / FP4 anywhere in the harness | none | grep (measured) | n/a: the draft is BF16 and the paper reports no FP8 draft |

## 5. What would fail or be missing on one RTX 3090 with our stack

Each item is estimated from the cited code unless marked otherwise.

1. **Engine hooks are missing in our vLLM 0.31.0** (measured, §3). `missing_hooks` refuses the plugin.
   - Option one: the `+eagle.4` fork (0.32.0rc1.dev283, cu130), which drops our 50-patch series [review: 51, `patches/series`]: pthfa, KVarN, int4 lm_head, DFlash2 fixes.
   - Option two: port ~10 hooks onto 0.31.0.
2. **Model Runner V2 required.** Our launch scripts never set `VLLM_USE_V2_MODEL_RUNNER` (measured: grep is empty in ours). Our patches target the V1 runner path, so switching runners risks every patch. [review: wrong. On vLLM 0.31.0 `use_v2_model_runner` defaults to True when the env var is unset and nothing unsupported is configured (`~/vllm` `vllm/config/vllm.py:700-760`), and our patches already edit MRV2 files (`v1/worker/gpu/model_runner.py`, `gpu/spec_decode/dflash2/speculator.py`, `gpu/cudagraph_utils.py`; ours `docs/gotchas.md:189`). MRV2 is not a new risk.]
3. **Quantized KV refused for tree verify** (`tree_attention.py:69-70`). Our fp8, `int8_per_token_head` and `kvarn_k4v2_g128` modes (ours `single-user/start_qwen.sh:218-250`) would fail at load in tree mode. Only `bf16` KV (`start_qwen.sh:213`) qualifies, at a ≥ 2× KV memory cost on a 24 GB card.
4. **`CUSTOM` backend clash.** Our `PTH_FA=1` path binds `--attention-backend CUSTOM` to pth-flash-attn (ours `start_qwen.sh:346`). EAGLE-4 registers `CUSTOM` too. Only one can win.
5. **`--linear-backend flashinfer_cutedsl`** is SM100-only. Drop it.
6. **Draft memory.** About 3.64 GB BF16 resident (architecture.md §1). Add 20 KiB/token of draft KV, and a GDN state row per verify row (~144 MiB fp32 each; architecture.md §7). [review: 72 MiB each on our stack, which runs `--mamba-ssm-cache-dtype float16` (ours `single-user/start_qwen.sh:1036`).]
   - On 24 GB with a W4A16 27B target (≈ 14–15 GB of weights, estimated from 27B × ~4.25 bits + BF16 embeddings), this crowds out KV capacity unless the draft layer is quantized, which no code here supports.
7. **sm86 is unqualified.** No 3090/A-series numbers exist. Kernels pass the `sm ≥ 80` gate and fall back to Triton for clusters and PDL. Correctness should hold, but autotune configs and the greedy-parity gate (`docs/support-matrix.md:159-163`) are untested on sm86.
8. **Calibration data is GPU-specific.** `check_data_key` refuses tables measured elsewhere (`vllm/config.py:304`). We must run `eagle-harness calibrate` on the 3090 (`docs/calibration.md`) and cannot reuse the B300 `rounds/tree.json`.
9. **Pipeline parallel unsupported** (`README.md:197`). This is irrelevant on one GPU.
10. **Licence.** Any of the above that copies harness code or weights triggers the source-available terms (overview.md, licence section).

Things that would *not* break:
- CUDA 13 / Python 3.12 (matches ours)
- Triton kernels
- the absence of FP8 dependencies
- FLASH_ATTN for a chain draft (FA2 supports sm80/86)

## Candidate techniques

- **E-20** Serve the released EAGLE-4 drafter as a plain chain drafter on our vLLM 0.31.0 to get an acceptance baseline without the harness:
  - override `architectures` → `Eagle3LlamaForCausalLM`
  - drop the `efficient_lm_head*` tensors (`fc_norm`, `norm_output` and the sliding layer are already supported, §3)
  - draft on FLASH_ATTN, BF16 draft weights, our quantized target KV unaffected [review: needs explicit per-draft `"attention_backend":"FLASH_ATTN"` and `"kv_cache_dtype":"bfloat16"` in `--speculative-config`; otherwise the draft inherits the target's KV dtype (`~/vllm` `vllm/config/speculative.py:376-384, 425-430`). See Reviewer additions for further missing steps.]
  - Gate behind E-01 (licence).
- **E-21** Port the warp-shuffle candidate tail (`csrc/candidate.cu`, sm_80+) or its Triton twin (`candidate_triton.py`) as the top-k stage of a low-rank head on sm86. It is the only native kernel that applies to Ampere.
- **E-22** CUDA-graph conditional nodes (`begin_capture_to_if_node`, `vllm/tree_drafter.py:191-207`) to skip unused draft steps inside one captured graph. This needs only the CUDA 12.4+ driver feature and is not Hopper-specific. Overlaps deepinfer spec-2.
- **E-23** Side-stream L2 prefetch of the next GEMM's weights (`kernels/l2_prefetch.py`) during small draft kernels. [review: SGLang-only in the release (`sglang/l2_warm.py`); a vLLM port is new work.] The benefit on a 6 MB L2 is doubtful; benchmark before adopting.
- **E-24** Native-kernel load order "prebuilt → JIT → bit-identical Triton fallback", gated by an arch-capability table (`kernels/native.py`, `kernels/arch.py`). Use it as a pattern for our own CUDA extras, such as pthfa and KVarN, so sm86 never hard-fails.
- **E-25** Rename the EAGLE-4 tree-verify backend, or add an ancestor-bitmask variant to our pthfa/KVarN verify kernels, so tree verify can coexist with quantized KV. This is the prerequisite for E-14 on our stack. Overlaps deepinfer attn-1, attn-2 and spec-3.

## Reviewer additions

Static analysis by the phase-1 reviewer (00-meta/reviews/phase1-eagle4-architecture.md). Nothing was loaded or run; all items are **estimated** from the cited code.

- **E-20, missing steps.**
  - (a) **Aux layer ids are off by one on stock vLLM.** `qwen3_next.py:711-724` appends the hidden state *entering* layer i (`layer_idx + 1` bookkeeping), and `gpu_model_runner.py:5473-5490` uses `eagle_aux_hidden_state_layer_ids` as-is (only DFlash adds +1). The EAGLE-4 plugin adds +1 itself (`eagle_harness/vllm/config.py:199-206`). So the config's `[3, 31, 59]` must be edited to `[4, 32, 60]`, or the draft gets features from layers 2/30/58.
  - (b) **Embedding.** The checkpoint ships `embed_tokens` (2.54 GB). Unless it is dropped, vLLM keeps the draft's own copy (6.18 GB draft total) because our int8 `weight_packed` target embedding cannot pass the `torch.equal` share check; that check reads `target.weight` and may raise on the quantized layer. Dropping it is only valid if the draft embedding equals the original BF16 target embedding (unverified).
  - (c) Per-draft `kv_cache_dtype` and `attention_backend` overrides (see the inline note under E-20).
  - (d) The plain EAGLE-3 path reads the full 2.54 GB lm_head every draft step: ≈3.5 GB, ≈3.7 ms/step at 936 GB/s. E-20 is an acceptance baseline, not a speed path.
  - (e) Stock vLLM SWA counts draft tokens inside the 4096 window, unlike EAGLE-4's "W context + d chain tokens" (`eagle_harness/models/eagle4_config.py:9-12`). Matters only past 4096 tokens of context.
- **E-22 is untested on our host.** CUDA-graph conditional nodes need CUDA 12.4+ driver support; nobody has checked them on our driver and container stack.
- **Hybrid KV page sizing.** A 20 KiB/token MHA draft layer must share vLLM's unified hybrid page size with the target's 64 KiB/token attention and GDN state pages; how vLLM 0.31.0 pads this was not examined.
