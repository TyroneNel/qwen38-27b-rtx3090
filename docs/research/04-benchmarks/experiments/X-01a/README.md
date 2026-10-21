# X-01a (H01): what `resolve_config.sh` resolves for an unset-`PTH_FA` launch on sm86

**Label: measured (CPU only, no GPU, no model).** Raw output: `run.out` (180 lines), produced by `run.sh`.

## Hypothesis (H01 / U-01 / I-13)

With the ci-cc86 stub `nvidia-smi` (433b050) and a faked `pthfa_installed`, `resolve_config.sh` on `upstream/main`
63f7581 resolves `SPEC=mtp CTX=long` to `PTH_FA=1`, and on our 95ac251 to `FA2_FP8KV` (the brief's expectation).
Also list the `.env` keys that block the new default.

## Setup

| What | Value |
|---|---|
| Worktrees (detached, throwaway) | `~/research-wt/x01a-upstream` = 63f7581 (contains b4b97ec #345 and 433b050 #346; checked with `merge-base --is-ancestor`), `~/research-wt/x01a-ours` = 95ac251 |
| Stub `nvidia-smi` | `stub-cc86/nvidia-smi`: the exact two-line script from `.github/workflows/patch-integrity.yml` (433b050), answers `8.6` to `*compute_cap*`, exit 9 otherwise. `stub-cc89` answers 8.9, `stub-none` is an empty dir (no nvidia-smi on this box) |
| Fake `pthfa_installed` | The real function runs (`resolve_config.sh:42` upstream) against a fake `$REPO/venv/bin/python` that exits 0 (`FAKE_PTHFA_MISSING=1` makes it exit 1). Scratch files in each worktree (untracked/ignored): `venv/bin/python`, `models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors` (empty file, needed by the dflash2 gate) |
| Environment | `env -i` with only `PATH`, `HOME`, `REPO` plus the knobs of the row. `source resolve_config.sh; resolve_effective_config single`, then print `PTH_FA` and `FA2_FP8KV` |

Command: `bash docs/research/04-benchmarks/experiments/X-01a/run.sh > run.out 2>&1` (needs the two worktrees above).

## Result (headline rows; every row is in `run.out`)

PTH_FA and FA2_FP8KV are the values exported by `resolve_effective_config`; "unset" means the resolver left the
variable alone, so the launcher falls to its arm default.

| Launch (cc8.6 stub, package present, no other knobs) | upstream 63f7581 | ours 95ac251 | run.out |
|---|---|---|---|
| `SPEC=mtp CTX=long` | **PTH_FA=1**, FA2_FP8KV unset (`[effective-config] PTH_FA=1 (default; ... pthfa/: target and MTP drafter)`) | **PTH_FA unset, FA2_FP8KV unset** (effective-config prints `FA2_FP8KV=0 (default)`; launcher `PTH_FA=${PTH_FA:-0}`, so fp8 KV on FlashInfer, `single-user/start_qwen.sh:220-223,297`) | :3 / :75 |
| `SPEC=mtp CTX=fast` | unchanged (both unset) | unchanged | :6 / :77 |
| `SPEC=dflash2 CTX=long` | PTH_FA=1, `FA2_FP8KV=0 (PTH_FA=1 overrides the 8.6 default)` | same (PTH_FA=1) | :8 / :79 |
| `SPEC=dflash2 CTX=fast`, `SPEC=mtp CTX=huge`, defaults | unchanged | unchanged | :10-16 / :77-85 |

**The brief's expectation for ours is wrong in one detail.** Ours does not resolve `FA2_FP8KV` for `SPEC=mtp CTX=long`:
the FA2 fp8 default exists only for `SPEC=dflash2 CTX=long` (`resolve_config.sh:121`, ours; `FA2_FP8KV=1` on mtp is refused,
`run.out` row "+ FA2_FP8KV=1"). On ours the mtp/long arm resolves to **neither** backend variable: PTH_FA stays 0 and the
KV cache is fp8 on FlashInfer. Upstream moves that arm to int8 on the pth_flash_attn kernel. The H01 claim "mtp/long flips
from fp8/FlashInfer to PTH_FA=1" is confirmed; "ours gives FA2_FP8KV" is refuted.

Scope: the upstream change touches only `SPEC=mtp CTX=long`. `dflash2/long` already defaulted PTH_FA=1 on ours (both rows
identical). `CTX=fast`, `CTX=huge` and the default `fast/mtp` launch do not move.

### What blocks the new default (upstream, `SPEC=mtp CTX=long`, cc8.6, package present)

Gate code: `resolve_config.sh:105-107` (mtp arm: FA2_FP8KV empty, DRAFT_TOKENS unset or 3, VISION != 1), `:109` (DYNAMIC_K, KV offload tier), `:111` (EXTRA_ARGS), `:114-116` (cc 8.6 and package).

| Knob / condition | PTH_FA resolved | Falls back to |
|---|---|---|
| none of the below | **1** | n/a |
| `VISION=1` | unset | fp8 + FlashInfer (arm default) |
| `DRAFT_TOKENS=4` (any value but unset/3) | unset | fp8 + FlashInfer |
| `DRAFT_TOKENS=3` explicit | 1 | not blocked |
| `DYNAMIC_K` set (any non-empty) | unset | fp8 + FlashInfer |
| `KV_OFFLOAD_GB=8` (non-zero) or `--kv-offloading-size` / `--kv-transfer-config` in EXTRA_ARGS | unset | fp8 + FlashInfer |
| `FA2_FP8KV=0` (set to anything) | unset (`=1` is refused on mtp) | fp8 + FlashInfer |
| `--kv-cache-dtype`, `--attention-backend` or `--speculative-config` in EXTRA_ARGS | unset | user's flags |
| `PTH_FA=0` | 0 (opt out) | fp8 + FlashInfer |
| `PTH_FA=` empty | 1 (treated as unset) | n/a |
| package missing | unset, one stderr note (`FAKE_PTHFA_MISSING=1` row) | fp8 + FlashInfer |
| GPU is cc 8.9 stub, or no `nvidia-smi` | unset, silent | fp8 + FlashInfer |
| box `.env` `CTX=fast VISION=1` | unset (CTX=fast is outside the arm; with `CTX=long SPEC=mtp` kept, VISION=1 still blocks) | fp8 or bf16 per CTX |

`.env` keys that block it: `VISION=1`, `DRAFT_TOKENS` (not 3), `DYNAMIC_K`, `KV_OFFLOAD_GB`, `FA2_FP8KV` (any value),
`EXTRA_ARGS` (three flag names), and `PTH_FA=0`. The prior box's `.env` carries `CTX=fast VISION=1`
(`ENVIRONMENT.md:27` per `quantisation.md:159`), so the default does not fire there even after a fast-forward; D10 pinning
`PTH_FA` explicitly in every arm is therefore still needed, and `[effective-config]` should be read per run. On dflash2/long
(upstream `resolve_config.sh:99-104`) the blockers are `VISION=1`, `DFLASH_TOKENS` not 7, `LOOKUP` not 1, `DRAFT`, `DRAFT_METHOD` not dflash,
`DYNAMIC_K`, `FA2_FP8KV` set, and a blocked launch falls to `FA2_FP8KV=1` (rows under "blockers on SPEC=dflash2 CTX=long",
both trees identical; `DRAFT_METHOD` not run).

### Repo launcher tests on CPU

The nine launcher tests from the CI step (`no_key_bind prefix_cache_knob kv_offload_knob dynamic_k_knob tp_size bool_knobs
resident_print fa2_fp8kv_knob pthfa_knob`) run on this box. Each tree runs its own copy of the tests, with and without the
cc8.6 stub on PATH: **36 of 36 pass** (`run.out:141-180`). Ours has no test of the mtp/long default (its `test_pthfa_knob.sh:179` section covers dflash2/long only), so a green ours run says nothing about H01.

## Verdict: PASS (with a correction)

Upstream: PASS, `SPEC=mtp CTX=long` resolves PTH_FA=1. Ours: the stated outcome "FA2_FP8KV" FAILS as written; the measured
truth is "PTH_FA and FA2_FP8KV both unset, fp8 KV on FlashInfer". The hypothesis that fast-forwarding to 63f7581 flips the mtp/long
default on sm86 is confirmed. Strictly against the pass criterion in `transferability.md` X-01a ("both as stated"), the
ours half is FAIL on wording; phase 5 should restate it as "ours: no pth, no FA2 on mtp/long". This is a resolver result: no
speed or quality claim is made (that is X-01b, GPU).

## Caveats

- `pthfa_installed` is faked (the real check needs the `_C.abi3.so` in a venv); the cc 8.6 answer is a stub. Both are what CI does.
- The resolver was exercised, not the full launcher: `single-user/start_qwen.sh` reads the exported variables afterwards
  (`:297` onwards) and was checked by reading, not by booting.

Re-run by: phase-4 reviewer, 2026-10-21, CONFIRMED: `run.sh` re-run from a scratch copy, output byte-identical to `run.out` (ours mtp/long = PTH_FA and FA2_FP8KV unset = fp8 on FlashInfer). Review: `00-meta/reviews/phase4-cpu-experiments.md`.
