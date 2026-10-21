# X-16 / H16: import probe of our CUDA-extension packages with no GPU

Label: **measured** (CPU, this box: torch 2.12.1+cu130, `cuda.is_available()` False, triton 3.7.1 CPU wheel, no vLLM). Raw output: `probe-before.out` (our `95ac251`, detached worktree `~/research-wt/x16`), `probe-after.out` (prototype patch applied), `fallback.patch`. Reproduce: `python3 probe.py [worktree]`.

## Hypothesis

H16 (EAGLE-4 E-24, `02-eagle4/build-and-kernels.md:147`): our compiled extras (pthfa, fa2, KVarN) hard-fail somewhere on a host they were not built for, where an arch-table fallback in the style of EAGLE-4's `kernels/arch.py` + `kernels/native.py` would degrade instead. X-16 pass = "a list exists" (`transferability.md:234`).

## Method

`probe.py` runs 33 cases, each in a fresh subprocess. Mode `bare` = vLLM absent. Mode `stub` = a permissive in-process `vllm.*` finder (the stub is embedded in `probe.py`, written to `$TMPDIR/x16-stub`), so what still fails is not "vLLM is missing". Stub details: `current_platform.get_device_capability()` returns the `STUB_CC` env (None = what I assume vLLM's CPU platform returns, **unverified**, I did not read `~/vllm`); `vllm.envs.<unset>` is None; `is_flash_attn_varlen_func_available()` is False; triton is real. KVarN has no package of its own: `kvarn/install.sh` copies `kvarn/files/vllm/*` into site-packages `vllm/`, so the probe maps those files in as `vllm.*` modules. `spec_decode_attn.py` (created by `patches/spec-decode-attn.patch:171`, later edited by 3 more patches) is rebuilt from the first patch's `+` lines only (case S1).

In-repo extension packages found: `pthfa/` (CUDA `_C.abi3.so`), `fa2/` (two CUDA `.so`: `fa2_fp8kv.so`, `fa2_fp8kv_prefill.so`), `kvarn/` (Triton JIT, no compiled extension), plus Triton modules created by `patches/` (`spec_decode_attn.py`, `prefill_attn_hd256.py`). No other `load_library`/`CDLL`/`cpp_extension` user outside `pthfa/setup.py`, `fa2/build.py`, tests, and `patches/offload-wsl2-devptr.patch:68` (ctypes cudart, already try/except'd there).

## Result per import path (before patch)

Class: **OK** / **SILENT** (no exception, feature off) / **HARD** (exception). Cause: V = only because vLLM is absent; G = genuine no-GPU / no-compiled-extension.

| id | import path | bare (no vLLM) | with vLLM stub | cause |
|---|---|---|---|---|
| P1/P2 | `pth_flash_attn`, `.planner` | OK | n/a | pure Python (docstring `__init__.py:4-8`) |
| P3 | `pth_flash_attn.ops` | **SILENT**: `is_available()` False, `_load_library` returns False (`ops.py:330-337`) | n/a | G, by design |
| P3b | `ops.pth_flash_attn_fwd(...)` call | **HARD** `RuntimeError "op library ... is not built"` `ops.py:388` | n/a | G, by design (only reached if backend was selected) |
| P4 | `pth_flash_attn.plugin` import | OK | OK | |
| P4b | `plugin.register()` | **HARD** `ModuleNotFoundError: vllm` `plugin.py:32` | OK | V |
| P5 | `pth_flash_attn.vllm_backend` | **HARD** `ModuleNotFoundError: vllm` `vllm_backend.py:52` | OK | V |
| P5t | `PthFlashAttnBackend.supports_compute_capability(8.6)` | n/a | **SILENT**: returns False because `ops.is_available()` False (`vllm_backend.py:562-564`) | G, an existing arch-table-plus-extension gate |
| F1 | `fa2_fp8kv_vllm` | OK | OK | |
| F2 | `fa2_fp8kv_vllm.paths.check()` | **HARD** `SystemExit "fa2_fp8kv: missing ...so"` `paths.py:25` | n/a | G, by design (launcher probe, `single-user/start_qwen.sh:248-249`) |
| F3 | `register()` with `FA2_FP8KV` unset | OK (no-op, `__init__.py:6-7`) | OK | |
| F4 | `register()` with `FA2_FP8KV=1` | **HARD** `ModuleNotFoundError: vllm` `__init__.py:8` | OK | V |
| F5/F6 | `fa2_fp8kv_vllm.backend`, `fa2_prefill` | **HARD** `ModuleNotFoundError: vllm` `fa2_prefill.py:4` (`from vllm.vllm_flash_attn import ...`) | OK | V |
| F7 | `paged_prefill` | OK | n/a | torch only |
| F5a | `backend` import, CPU platform (cc None) | n/a | OK (`backend.py:22,26` gate is False, no load) | |
| F5b/F5e | `backend`, sm80 / sm120, `FA2_FP8KV=1` | n/a | **SILENT** (`BUILT_FOR = ((8,6),(8,9))`, `backend.py:19`; no load; `extension_selected()` False, so native FLASH_ATTN) | G, existing arch table |
| **F5c** | **`backend`, sm86, `FA2_FP8KV=1`, `.so` missing** | n/a | **HARD** `OSError "Could not load this library: .../fa2_fp8kv.so"` `backend.py:27` (`torch.ops.load_library`, raised at module import) | **G, the only unguarded genuine import-time hard-fail** |
| F6s | `fa2_prefill` | n/a | OK | |
| K1-K7 | `vllm...quantization.kvarn.{sinkhorn,config}`, `...ops.{kvarn_decode,kvarn_store,triton_kvarn_sinkhorn,triton_kvarn_decode}`, `...backends.kvarn_attn` | **HARD** `ModuleNotFoundError: vllm` (K0, snippet line, no repo frame) | all OK. Real triton `@triton.jit`/`autotune` import cleanly with no GPU | V only |
| K8 | run `kvarn_sinkhorn_triton` on a CPU tensor | n/a | **HARD** `RuntimeError "0 active drivers"` at the launch `triton_kvarn_sinkhorn.py:180` (raised in `triton/runtime/driver.py:20`) | G, launch time, not import time |
| S1 | `spec_decode_attn.py` (patch-1 text) | n/a | OK | |

Count (probe-before.out): 19 OK, 4 SILENT, 10 HARD, 0 crash. Of the 10 HARD: 6 are vLLM-absent only (P4b, P5, F4, F5, F6, K0), 2 are deliberate loud refusals at call/launcher time (P3b, F2), 1 is a launch-time no-driver error (K8, Triton, needs a GPU anyway), and **1 is a genuine import-time hard-fail: F5c**.

Other findings:

- **KVarN has no arch table at all.** No `supports_compute_capability`, no capability read anywhere in `kvarn/files` or the four `kvarn/*.patch` files (grep). Triton JIT-compiles for whatever GPU is there. Prefill FlashAttention is already a silent fallback (`kvarn_attn.py:80-82`, `_HAS_FLASH_ATTN`). Shared-memory overruns on small-smem GPUs fall back via `OutOfResources` (`patches/spec-attn-smem-fit.patch:106`, the same pattern for `spec_decode_attn.py`).
- **pthfa already implements E-24's idea in miniature:** the backend advertises sm_8x only when the op library loaded (`vllm_backend.py:562-564`), `register()` never imports CUDA code (`plugin.py` docstring), and the launcher probes the `.so` file (`resolve_config.sh:42-46`, `start_qwen.sh:341-345`). It has no JIT tier and no Triton twin, so there is nothing to fall back to for int8 KV on `CUSTOM`; the "fallback" is the launcher's other arm (`PTH_FA=0`).
- **Project policy conflicts with a silent fallback.** `start_qwen.sh:244-246`: "A missing install refuses rather than falling back: a fallback would make the default arm depend on whether a build step happened." E-24's "prebuilt -> JIT -> Triton, never hard-fail" is the opposite policy. Any guard below therefore logs a warning, and the launcher refusal stays the production gate.

## Fallback design (E-24 style, minimal guards)

| hard-fail | guard | where | fallback behaviour | in patch? |
|---|---|---|---|---|
| F5c `OSError` at `torch.ops.load_library` | `paths.missing()` (list of absent `.so`); `register()` returns early with a warning; the load at import is also conditional on `not missing()` | `fa2/python/fa2_fp8kv_vllm/__init__.py:11-14` (register), `backend.py:25` (load), `paths.py:26-27` (helper); arch table `BUILT_FOR` moved to `paths.py:11` | FLASH_ATTN stays stock. Note stock FLASH_ATTN on sm86 has no fp8 KV: `--kv-cache-dtype fp8` is then rejected by vLLM at selection with its own message, instead of an `OSError` inside every worker. A real working fallback is the launcher's int8 + TRITON_ATTN arm (`FA2_FP8KV=0`) | yes (`fallback.patch`, 3 files, 22 insertions, 7 deletions) |
| P3b call without `.so` | none: already behind `supports_compute_capability` | `vllm_backend.py:562-564` | explicit `--attention-backend CUSTOM` then fails in vLLM's selection with "unsupported capability", which hides the real cause (missing `.so`). Optional guard: a `logger.warning` in `plugin.register()` when `ops.LIBRARY_PATH` is absent (do not import `ops`: the plugin deliberately avoids it) | no (design only) |
| F2 `SystemExit` | none, intentional; E-24 would turn it into a warning, which the launcher policy forbids | `paths.py:25` | keep | no |
| P4b/P5/F4/F5/F6/K0 | none: vLLM is a hard dependency of these modules (`fa2/pyproject.toml` `dependencies = ["vllm>=0.29.0"]`) | n/a | n/a | no |
| K8 Triton "0 active drivers" | none needed. If KVarN ever targets a non-sm86 host: add `supports_compute_capability` (for example major >= 8, bf16 + Triton) to `KVarNAttentionBackend` next to `supports_attn_type`, `kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:~245`, so vLLM rejects at selection instead of failing in a kernel. The right threshold is unmeasured (KVarN is measured on sm86 only) | design only, **estimated**, untested | no |

### With the patch

`probe-after.out`: 21 OK, 4 SILENT, 8 HARD (was 19/4/10). F5c is now OK (backend import no longer raises; `register()` declines). F4 (bare) now reads OK only because the missing-`.so` early return happens before the vLLM import; that is a side effect, not a fix of the vLLM-absent case. All other rows are unchanged. The worktree still carries the uncommitted patch (scratch edits only; nothing committed).

## PASS/FAIL

**PASS** against "a list exists": 10 hard-fail rows classified (table above), 1 genuine import-time failure, 1 unguarded arch-table gap (KVarN).
H16 itself, as phrased in the ranking ("no hard-fail on new hosts"), is **mostly refuted**: the repo already gates on arch + extension presence for pthfa and fa2 (`vllm_backend.py:562`, `backend.py:19-28`, `resolve_config.sh:104-149`), and the one real gap (F5c) is closed by a small guard (`fallback.patch`).

## Production relevance

| item | production (GPU box) | CI / no-GPU only |
|---|---|---|
| F5c (sm86, `FA2_FP8KV=1`, `.so` missing) | Low. The image builds both `.so` and runs gate tests at build (`Dockerfile:32-39`); `start_qwen.sh:248-249` refuses before launch. It matters only for a venv outside the image (a partial `fa2/install.sh`) or a launcher that skips the check | |
| P3b / P5t (sm86, pthfa `.so` missing, `PTH_FA=1`) | Low, same reasons (`start_qwen.sh:341-345`). The visible symptom is vLLM's generic unsupported-capability error | |
| F5b/F5e (A100 sm80, Blackwell sm120) | Already graceful, but fp8 KV has no native backend there; `resolve_config.sh:118-149` keeps `FA2_FP8KV` off off-8.6 | |
| KVarN on a non-sm86 GPU | Unknown: no table, no measurement. Untested risk if the project moves GPUs | |
| V-class rows, K8, F2 | | CI and import-only probes only |

## Open questions

1. Does real vLLM's CPU platform return None from `get_device_capability()` (F5a assumption), and is `DeviceCapability` tuple-equal to `(8, 6)` for the `in BUILT_FOR` test (`backend.py:21`)? Not checked (`~/vllm` untouched). [review: yes to both. The base `Platform.get_device_capability` returns None (vLLM v0.31.0 `vllm/platforms/interface.py:433`), CpuPlatform does not override it, and `DeviceCapability` is a NamedTuple (`interface.py:91`), so `(8, 6) in BUILT_FOR` membership works.]
2. Does the pthfa gate test (`pthfa/tests/gate_test.py`) cover "`.so` missing" on a GPU-less vLLM? It requires the `.so` (check 1), so the missing-library path is never run in CI.
3. `spec_decode_attn.py` was probed from patch-1 text only, not the fully patched file (needs the patched vLLM tree, which this box lacks).

Re-run by: phase-4 reviewer, 2026-10-21, CONFIRMED: `probe.py` re-run before (git archive of 95ac251) and after `fallback.patch`; 19/4/10 and 21/4/8 reproduced (identical after path normalisation). F5c is the only import-time hard-fail for the probed paths; a present wrong-arch `.so` and vLLM-side imports hidden by the stub were not probed. The patch leaves the launcher refusal (`start_qwen.sh:245-249`) intact.
