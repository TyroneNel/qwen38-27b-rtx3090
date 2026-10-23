# pthfa: paged int8 per-token-head attention as a vLLM backend plugin

`pthfa/` is the source of [pth-flash-attn](https://github.com/noon-at-cgn/pth-flash-attn) (Apache-2.0) at one pinned commit (`pthfa/PIN`, a commit id; here the release v0.1.0): a CUDA op library that reads vLLM's `int8_per_token_head` KV cache directly (no dequantize
pass) on sm_80/86/89, and a `vllm.general_plugins` entry point that registers it as the backend `CUSTOM`.
`PTH_FA=1` (single-user, `SPEC=mtp CTX=long` or `SPEC=dflash2 CTX=long`) selects it for the target and
the drafter (the MTP layer, or the DFlash2 drafter's sliding-window, non-causal layers), and so does the
launcher's default on `SPEC=dflash2 CTX=long` and `SPEC=mtp CTX=long` on a compute capability 8.6 GPU when
this package is installed (`PTH_FA=0` opts out; `docs/gotchas.md`, item 63); nothing else
does. It is a separate extension, not a patch to vLLM, which is why it lives here and not in `patches/`:
no vLLM file changes, and `patches/check_vllm_series.sh` is untouched by it.

## What is vendored, and from where

`PIN` names the commit. `csrc/` and `python/` are the output of `tools/export_upstream.py --out` at that
commit: the extension's translation units, every header they include, and the `pth_flash_attn` package,
with the test-only handshake hooks stripped (upstream's `tools/prove_export_sass.sh` proves the stripped copy
compiles to the same SASS). `setup.py`, `pyproject.toml`, `tools/build_flags.py`, `LICENSE` and `NOTICE` are
the files at that commit, unchanged. `SHA256SUMS` lists all 36 files (32 vendored, plus attn-1's
`csrc/pth_attn/instantiate_d256_attn1.cu` and attn-2's three KVarN sources; see the attn-1 and attn-2 sections
below); the files are never edited here.

Source rather than a prebuilt `.so`: the same reasoning as #153, where a binary was declined in favour of a
source port CI can check. `bash pthfa/check_vendor.sh` checks the files against `SHA256SUMS` and that nothing
under `pthfa/` is unlisted; `--source <pth-flash-attn checkout>` re-exports `PIN`'s commit and diffs (a clone of the repository above has the tag).

## How it is built and installed

`bash pthfa/install.sh` compiles the library against the venv's torch (`TORCH_CUDA_ARCH_LIST` from `PIN`,
SASS only, no PTX, so an sm_90+ card is refused rather than JIT-compiled) and pip-installs the package into
the venv. It needs nvcc 13 and g++, no GPU and no network. It checks the installed copy: cubins for exactly
sm_80/86/89 and no PTX, then `tests/gate_test.py`. A rerun with the same pin, arch list, torch and nvcc is a
no-op. The Dockerfile runs it in its own layer before `COPY . .`, so an edit elsewhere does not recompile it,
and runs the gate test again on the patched tree.

## What it does when nothing selects it

The entry point records a class path under `AttentionBackendEnum.CUSTOM` and imports no CUDA code; vLLM
loads general plugins at debug log level, so the boot log does not change. `CUSTOM` is in none of vLLM's
automatic-selection lists, so only an explicit `--attention-backend CUSTOM` (or `"attention_backend":"CUSTOM"`
in a speculative config) reaches it. If another plugin registered `CUSTOM` first, this one leaves it alone and
says so. `tests/gate_test.py` checks each of these.

## Limits

sm_80, sm_86 and sm_89 only; head sizes 64, 128 and 256; `--kv-cache-dtype int8_per_token_head` only;
decoder self-attention, causal at every head size, sliding-window and non-causal at head size 128 only (the
DFlash2 drafter's layers); no ALiBi, attention sinks, logits soft cap, bidirectional multimodal prefix or
R-SWA; no DCP/PCP. An explicit selection outside these fails at backend selection or model load with the
reason (`docs/gotchas.md`, item 63). The library reads no environment variables.

## attn-1 experiment knobs (research/deepinfer)

`docs/research/deepinfer/attention-kv.md` (attn-1) changes the vendored kernels in place on this branch, so
`SHA256SUMS` lists the edited files and `check_vendor.sh --source` no longer matches `PIN`'s commit until the
change is re-exported upstream as a pthfa revision. Every knob defaults off; unset, the launches are the shipped
kernels. `pth_flash_attn.ops` reads the environment once at import and sets the library's process-wide mode
(`ops.set_attn1_mode`, C entry point `pth_flash_attn_set_attn1_mode`); the op schema is unchanged.

- `PTHFA_TILE48=1` (stage 1, lossless): D = 256 row tiles of 33..48 rows (6..8 query tokens at G = 6, the
  DFlash2 verify) run on the swap-AB decode kernel with 5 or 6 n-tiles instead of the padded 64-row tile.
- `PTHFA_MMA=f32|pv16|s8pv16` (default `f32`): `pv16` (stage 2, near-lossless) accumulates the decode kernel's
  P·V of each 64-token tile in fp16 (`m16n8k16.f16.f16.f16.f16`) and adds it to the fp32 O; `s8pv16` (stage 3,
  lossy) adds QK on int8 (`m16n8k32.s32.s8.s8.s32` on the cache's int8 K, each Q row quantised by its absmax).
  Only launches on the decode kernel (D = 256 and at most 32 rows, or 48 with `PTHFA_TILE48=1`) change; the
  variants are `csrc/pth_attn/instantiate_d256_attn1.cu`. `bench/test_pthfa_attn.py` has the re-derived bound
  (stage 2) and the error statistics (stage 3).

## attn-2 build flag (research/deepinfer): KVarN verify

`docs/research/deepinfer/attention-kv.md` (attn-2) adds a second op to the library for KVarN's uniform speculative
verify (mode E): `pth_flash_attn::pth_kvarn_verify_fwd`, wrapper `pth_flash_attn.ops.kvarn_verify`, kernel
`csrc/pth_attn/attn_decode_kvarn.cuh` (its header has the design and the arithmetic budget), instantiations
`csrc/pth_attn/instantiate_kvarn.cu`, op and registration (a `STABLE_TORCH_LIBRARY_FRAGMENT`)
`csrc/pth_attn/kvarn_launch.cu`. Opt-in at BUILD time: `PTHFA_KVARN=1 bash pthfa/install.sh` adds the two
translation units (`tools/build_flags.py` `KVARN_SOURCES`); unset, the source list, the flags and so the library are
the ones above, and `ops.kvarn_is_available()` is False. The vLLM side is
`docs/research/deepinfer/experiments/patches/attn2-kvarn-pthfa.patch` (`KVARN_BACKEND=pthfa`, default off), the test
and microbench `bench/test_pthfa_kvarn.py`.

## Tests

- `pthfa/tests/gate_test.py` (CPU): the items above, against the venv's vLLM.
- `bench/test_pthfa_knob.sh` (CPU): `PTH_FA` in the launchers, against a stub `vllm`.
- `bench/test_pthfa_attn.py` (GPU): the op against an fp64 reference on the cache it reads, including a pool
  past block id 2,383 (#86/#109), CUDA-graph replay and the DFlash2 drafter's windowed, non-causal shapes;
  `SKIP(by design)` off sm_8x or without the package.
- `bench/test_pthfa_kvarn.py` (GPU; `--dry-run` on CPU): the attn-2 KVarN op against bench/test_kvarn_verify.py's
  fp64 reference, `--bench` its µs per layer; `SKIP(by design)` when the library was built without `PTHFA_KVARN=1`.

## Re-vendoring

From a pth-flash-attn checkout at the new commit: `tools/export_upstream.py --out <dir>`, copy `csrc/` and
`python/` and the five build files here, set `PTH_COMMIT` in `PIN`, regenerate `SHA256SUMS`
(`find . -path ./tests -prune -o -type f ! -name README.md ! -name PIN ! -name SHA256SUMS ! -name check_vendor.sh ! -name install.sh -print | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS`),
then `bash pthfa/check_vendor.sh --source <checkout>`.
