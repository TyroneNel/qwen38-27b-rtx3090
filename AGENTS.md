# Repository Guidelines

## Upstream and This Fork

Two remotes, two sets of rules:

- `upstream` = `syv-ai/HyperQwen` (`upstream/main`) is the source of truth and where PRs go. The remote URL still says the pre-rename `syv-ai/qwen38-27b-rtx3090`, which redirects. Everything below describes upstream unless it carries **(fork)**. Last checked against `upstream/main @ d5e2a01`.
- `origin` = this fork. Local and `origin` `main` mirror `upstream/main` exactly: no fork commits, fast-forward only (`git sync-main`). `master` is the fork's integration branch: a strict superset of `upstream/main`, holding fork-only commits plus `--no-ff` merges of the fix branches under test.

Git config and `.git/hooks` enforce this: `main` tracks `upstream/main`; `remote.pushDefault=origin` and `push.default=current`, so a branch pulls from upstream and pushes to the fork under its own name. `pre-commit` refuses commits on `main`, and refuses a staged `AGENTS.md` on any branch but `master`. `pre-push` refuses any push to `main` that is not `upstream/main`, and any push of a ref other than `master` whose tip contains `AGENTS.md`. `git new <branch>` fetches and cuts the branch from `upstream/main`.

**(fork)** marks material that exists only on `master`. The fork-only inventory:

- Files: `AGENTS.md` (this file), `bench/canary.sh`, `bench/paired_run.sh`, `bench/measure_c1.sh`, `bench/test_fa_scratch_capacity.py`, `bench/test_paired_run_port.py`, `prepare/crash_inject_proof.sh`, `run_quant.sh`, `test_resolution.sh`, `docker-compose.override.yml`, `.github/workflows/bench-policy.yml`.
- Docs: `docs/architecture-review-*.html`, `docs/decode-perf-150k-240k-plan.md`, the seven `docs/*-remediation.md` plans, `docs/AUDIT_REPORT_*.md`.
- Edits to shared files: the `docs/*` allowlist in `.gitignore`, the extra `.gitattributes` rules, `[managed]` markers in `.env.example`, the digest pin in `Dockerfile`, the disabled publish triggers in `docker-image.yml`, the tree-reset guard in `patches/check_vllm_series.sh` plus its sentinel step in `patch-integrity.yml`, the `headroom` service in `docker-compose.yml`, the unknown-`CTX` refusal in `single-user/start_qwen.sh`, and the bench `RESULT`/manifest/`REUSE=1` conventions.
- `master` also edits both launchers, `docker/prepare.sh`, `.dockerignore`, most bench drivers, `drafter/` and two `kvarn/files` modules. Diff against `upstream/main` before you assume a shared file is upstream's.

`docs/self-improvement-loop.md` is untracked (working tree only). This file is tracked on `master` only; no branch cut from `upstream/main` contains it, so checking one out removes it from the working tree. In such a worktree, read it with `git show master:AGENTS.md`.

**Upstream work** (an issue, a fix, a PR):

1. Branch from `upstream/main` in its own worktree (`git worktree add -b fix/<topic> ../hq-<topic> upstream/main`), so fork-only files stay out of the diff.
2. Verify every claim against upstream with `git show upstream/main:<path>` / `git grep <pattern> upstream/main`; the fork's working tree answers a different question.
3. Match the upstream conventions below; keep **(fork)** conventions out of upstream diffs.
4. Write issue and PR text in ASD-STE100 structure (short sentences, active voice, one instruction per step), with before/after evidence.

## Project Overview

HyperQwen serves a quantized Qwen3.8-27B on one 24 GB consumer GPU (RTX 3090/4090) with an OpenAI-compatible API: ~127 tok/s single-stream, ~1,035 tok/s aggregate at 64 concurrent, 150k–262k context (245,760 default on `SPEC=dflash2 CTX=huge`; 221,184 at `DFLASH_TOKENS>7`). The mode READMEs label their tables as vLLM 0.27.1 baselines awaiting a 0.30 re-run.

Under the serving setup this repo is three things, and none of them is an application:

1. **A patch series against a pinned vLLM** (`patches/`, 45 files, `vllm==0.30.0` since #189) applied onto the installed `vllm` package.
2. **A model-preparation pipeline** (`prepare/`, `drafter/`) that requantizes heads and builds a draft vocabulary in place, publishing each file atomically (#195).
3. **Two bash launchers** that translate env knobs into one `vllm serve` command line, plus the experimental third, `single-user/alternative.sh` (int4 KV, 256k, `set -e`, no `resolve_config.sh`, no `select_model.sh`).

There is no Python service layer. Editing "the code" usually means editing a shell launcher, a generated patch, or a prep script — and the accompanying prose, which is load-bearing here.

## Architecture & Data Flow

**Boot path** (same skeleton in `single-user/start_qwen.sh` and `batch/start_qwen.sh`):

```
export FLASHINFER_DISABLE_VERSION_CHECK=1 ; reap stale /dev/shm/vllm_offload_* ; REPO=… ; cd $REPO
  → CUDA_HOME → venv nvidia/cu13 when the PATH nvcc is older than 13
  → source resolve_config.sh ; resolve_effective_config single|batch   # validate/refuse/warn, print [effective-config]
  → source single-user/select_model.sh                                 # single only: prefer models/…-AutoRound-fast
  → profile branch: CTX+SPEC (single: MAX_LEN/DRAFT_TOKENS/ATTN_ARGS, then KV_MEM/MAX_SEQS) or KV (batch: MAX_LEN/GPU_UTIL/KV_ARGS)
  → feature knobs into bash arrays (TOOL_ARGS, METRICS_ARGS in both; single adds SPEC_ARGS, ASYNC_ARGS)
  → source resolve_api_key.sh ; resolve_vllm_key
  → exec venv/bin/vllm serve "$MODEL" … ${EXTRA_ARGS}                  # batch: EXTRA_ARGS last; single: the SSE keep-alive flag follows it
```

Both launchers prepend their own flags to `EXTRA_ARGS`, so a user flag later on the line still wins. Single adds `--kv-cache-memory=$KV_MEM` in every SPEC mode when `KV_MEM` is set (#214). With `PREFIX_CACHE=1` it adds `--enable-prefix-caching --mamba-cache-mode align`, `--prefix-match-unit 128` (`CTX=huge`) and `--prefix-cache-retention-interval …`. Batch adds only the prefix-caching pair and has no `KV_MEM`.

**Container path** (`docker/entrypoint.sh`, `set -e`): `docker/prepare.sh` (flock-serialized; `PREPARE=0` skips) → `select_model.sh` + `export MODEL` (single only) → `verify.sh --no-server` as a hard gate (`VERIFY=0` skips) → the launcher.

**Build path** (`Dockerfile`): tag-pinned CUDA base (**(fork)** digest-pinned) → `pip -r docker/requirements.txt` → `patches/apply.sh "$SP"` (the series at `--fuzz 0`) → `kvarn/install.sh` (overlay, then `patches/apply.sh --kvarn`) → `verify.sh --install`. The build *is* the patch gate, and `docker-image.yml` runs it on every PR without pushing (**(fork)** publish triggers disabled, manual only). `apply.sh` is upstream PRs #242–#244, merged here ahead of upstream.

**Evidence path** (how a number becomes repo content): a run → `ROW` lines + raw logs in `bench/results/` → an issue (field-report template) → a row in `docs/reproductions/README.md` → a numbered entry in `docs/gotchas.md` if durable → a `PATCHES.md` row, and for a user-facing knob a line in the mode README's `## Knobs` table and in `.env.example`.

Env vars are the only interface between layers. Model dirs are produced offline and mutated **in place**.

## Key Directories

| Path | Role |
|---|---|
| `single-user/` | Latency launcher (mostly measurement-record comments), `alternative.sh` (experimental third launcher), `select_model.sh`, `qwen-server.sh`, systemd unit |
| `batch/` | Throughput launcher: 64 seats, `INT8_ACT=int8` by default, `KV=fp8` default (`KV=kvarn\|int4pth` → 262,144; WSL2 `KV=kvarn` → 131,072), no speculation |
| `patches/` | Generated diffs + `series` (apply order) + `apply.sh` (the only series parser and applier: `--list`, `DIR`, `--kvarn DIR`) + `check_vllm_series.sh` + `_check_applied.py` |
| `kvarn/` | Quantized-KV backend: file overlay + three 0.30.0 patch files (`kvarn-0.30.0`, `kvarn-v2-runner-0.30.0`, `kvarn-recycled-pages-0.30.0`), applied by `patches/apply.sh --kvarn` from `install.sh`. Installed **after** the series, not part of it |
| `prepare/` | In-place requantization (`quant_lm_head.py` → `quant_embed.py` → `quant_mtp.py` → `build_draft_vocab.py`), `quant_heads_stream.py` for single-shard/AWQ, `atomic_publish.py` (the shared write protocol, #195), the two chat-template writers (`harden_`/`translate_chat_template.py`), `fetch_*.py` downloads |
| `drafter/` | How the `-fast` variant and the DFlash2 drafter were *built*; not needed to serve. Still writes in place, not through `atomic_publish.py` |
| `bench/` | CPU tests (`test_model_verification.py`, `test_prepare_state.py` in CI; `test_prepare_crash.py`, `test_kvarn_recycled_pages.py`, `test_bench_sse_keepalive.py` by hand), GPU kernel oracles (other `test_*.py`), shell drivers (`run_benchmarks.sh`, `real_rep.sh`, `prefill_ab.sh`, `warmup.sh`), live-server probes |
| `docker/` | `entrypoint.sh` (dispatch + gate), `prepare.sh` (idempotent, flock), `requirements.txt` (the pin) |
| `docs/` | Cross-cutting prose, all tracked upstream. **(fork)** `.gitignore` ignores `docs/*` behind an allowlist |
| `scripts/` | `export-patch.sh` (the sanctioned way to (re)generate a patch) and `hq-doctor.sh` (read-only status: config, GPU, port, `/health`, last log error; exit 0/1/2). The root `Makefile` wraps these and compose (`make help`) |

## Development Commands

```bash
# Serve (native venv, Linux/WSL2)
bash single-user/start_qwen.sh                                  # CTX=fast SPEC=mtp on :18020
SPEC=dflash2 PREFIX_CACHE=1 bash single-user/start_qwen.sh      # recommended single-user profile
CTX=huge SPEC=dflash2 bash single-user/start_qwen.sh            # 240k; needs kvarn/install.sh
KV=kvarn bash batch/start_qwen.sh                               # 262k (WSL2: 131k), 64 seats

# Validate without booting anything (no GPU, no vLLM)
REPO=$PWD bash resolve_config.sh single   # prints [effective-config]; a bad knob prints "refusing" and exits 1
bash verify.sh --no-server             # install + patches + KVarN + model dir + tokenizers + drafter (the serve gate)
bash verify.sh --install               # build-time subset: no GPU, no model
bash verify.sh --wait 600              # poll a starting server; exit 2 on a bad argument
bash scripts/hq-doctor.sh              # read-only status; also `make doctor`

# Docker (knobs from .env; upstream's compose file needs no hand-edit)
docker compose run --rm prepare
docker compose --profile single up -d  # also: make up-single / up-batch / verify / keygen
docker compose run --rm single verify

# The upstream CI gate (.github/workflows/patch-integrity.yml; two jobs)
python bench/test_model_verification.py
python bench/test_prepare_state.py
tag=v$(grep -E '^vllm==' docker/requirements.txt | cut -d= -f3)   # CI: actions/checkout at $tag, depth 1
git clone --depth 1 --branch "$tag" https://github.com/vllm-project/vllm .ci/vllm
bash patches/check_vllm_series.sh .ci/vllm/vllm      # arg is the PACKAGE dir; it RESETS the whole checkout
# (fork) the script always refuses this repo, and refuses any other tree without
# .qwen-disposable-series-target at its git root unless ALLOW_TREE_RESET=1
# (fork CI stamps it: touch .ci/vllm/.qwen-disposable-series-target)

# (fork) bench-policy.yml adds: bash -n over bench/ docker/ single-user/ batch/ patches/ *.sh + verify.sh,
# compileall (bench drafter prepare kvarn/files _check_applied.py), a git apply --numstat loop
# (vision-tower-cpu-offload.patch exempt), CPU fixtures bench/verbatim.py, bench/test_fa_scratch_capacity.py,
# bench/test_paired_run_port.py, and source-text checks

# Regenerate a patch — never hand-edit one
bash scripts/export-patch.sh <vllm-fork checkout> <[qwen38] topic commit> [patches/<topic>.patch]
```

There is no linter, formatter, or package-manager step. The gate has three parts:
- `bench/test_model_verification.py` + `bench/test_prepare_state.py`.
- `patches/check_vllm_series.sh`: every series patch through `apply.sh` at `--fuzz 0`, then the three KVarN patches (`apply.sh --kvarn`), then `git apply --check` and a real `git apply` on the five contractual DFlash patches, then `git diff --check`. **(fork)** It refuses this repo and any tree without the `.qwen-disposable-series-target` sentinel (or `ALLOW_TREE_RESET=1`).
- The Docker build.

## Code Conventions & Common Patterns

**Shell (the dominant language here).**

- Launchers do **not** run under `set -e`. That is why the resolver source is guarded: `source "$REPO/resolve_config.sh" || { echo …; exit 1; }` (the `select_model.sh` source is not). Other modes:
  - `set -e`: `docker/entrypoint.sh`, `docker/prepare.sh`, `kvarn/install.sh`, `single-user/alternative.sh`.
  - `set -euo pipefail`: `patches/check_vllm_series.sh`, `bench/warmup.sh`, `single-user/qwen-server.sh`.
  - `set -eu`: `scripts/export-patch.sh`.
  - `set -u`: `bench/run_benchmarks.sh`, `bench/prefill_ab.sh`, `scripts/hq-doctor.sh`.
  - Nothing: `verify.sh`, `bench/real_rep.sh`.
  - **(fork)** `run_benchmarks.sh`, `prefill_ab.sh` and `canary.sh` are `set -euo pipefail`; `real_rep.sh`, `paired_run.sh` and `measure_c1.sh` are `set -u`.
- Conditional flags are **bash arrays**, never `$( [ … ] && echo --flag )` — the substitution exits 1 when false (killed the script under `set -e`, #59) and unquoted expansion word-splits parser names. See `SPEC_ARGS=()`, `ASYNC_ARGS=(--no-async-scheduling)` in `single-user/start_qwen.sh` and `TOOL_ARGS=()`, `METRICS_ARGS=()` in both launchers.
- `${VAR-default}` vs `${VAR:-default}` is deliberate: the no-colon form lets an explicitly empty value mean "off / drop the flag" (`SSE_KEEP_ALIVE=${SSE_KEEP_ALIVE-30}`, `INT8_ACT=${INT8_ACT-int8}`, `KV_MEM=${KV_MEM-…}`).
- Export only non-empty values: `[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT`. vLLM's `env_with_choices` rejects `""` and kills the engine at startup.
- `resolve_config.sh` and `resolve_api_key.sh` define functions and set nothing until called. `single-user/select_model.sh` is a fragment that assigns `MODEL` (unexported) when sourced. All three need `REPO` set first.
- **Refuse when the config cannot be correct, warn when it is merely untested.**
  - `resolve_config.sh` refuses unknown `CTX`/`SPEC` (single) and `KV` (batch). It warns on the other mode's knobs and on `EXTRA_ARGS`-shadowed flags.
  - `single-user/start_qwen.sh` also refuses an unknown `SPEC`, `SPEC=dflash2` with no drafter, and fp16 + a speculator (#27).
  - It warns on `KV_MEM` above the profile default, `MAX_SEQS>12` at `CTX=huge`, `DFLASH_TOKENS>7` at TP>1, `CTX=huge SPEC=mtp PREFIX_CACHE=1` (#64) and a >2 GiB drafter.
  - Batch warns on nothing.

**Python (prep + bench).**

- The prepare writers go through `prepare/atomic_publish.py` (#195, PRs #198–#203):
  - `backup_once()` keeps the first backup: `.bak`, `.bak-quant`, `.bak_embed`, `.bak-mtp`, `.bak-draft`, and `.bak-orig`, which is a hardlink.
  - Each file is written to `<path>.tmp`, fsynced and renamed.
  - The safetensors index goes last, as the commit point. A killed run leaves its step pending, and the next prepare finishes it.
- `quant_lm_head.py`, `quant_embed.py` and `quant_heads_stream.py` assert their round-trip error (`assert err < 0.01`); `quant_mtp.py` only prints it.
- Tests are `bench/test_*.py`. A docstring says what the test checks and how to run it; the script prints `OK`/`FAIL` per case and exits 0/1 (`sys.exit(0 if ok else 1)`). Designed skips print `SKIP(by design)` and exit 0. The two CPU tests CI runs use `unittest`; there is no pytest.
- Measurements are `ROW <label> | k=v | k=v` lines. **(fork)** bench scripts also end in `RESULT PASS`/`RESULT FAIL` and print `PASS `/`WARN `/`FAIL ` items. Their docstrings say how they fail closed and cite the audit finding (`Fail-closed (F02): …`).

**Checks are behavioural where they can be.**
- `verify.sh` asserts registrations against the live registry (`envs.environment_variables`, `AttentionBackendEnum.KVARN`) rather than grepping for them.
- It checks patches with `patch -R --dry-run --fuzz 0`. For overlapping hunks it falls back to `patches/_check_applied.py`, and for rewritten ones to `Supersedes:`.
- Only the patch-gated live rows (`graded()`) and the recycled-pages check grep the installed tree.
- `patches/apply.sh --kvarn` checks each KVarN patch with an exact reverse dry-run before applying it: all hunks present → skipped, so an `install.sh` rerun is a no-op; a partly applied patch fails by name. `patch -N` is not an idempotence test: it skips a whole file when that file's first hunk is present. `verify.sh` checks all three KVarN patches with the same exact reverse dry-run, not `_check_applied.py`, whose 80%-per-file rule misses one missing hunk.

**Comments are measurement records.** Most knobs carry the A/B table, issue number and hardware that justify the default. Preserve them; they are the primary documentation.

**Commits.** Loose Conventional:
- Usually a lowercase `scope:` prefix naming the touched area: a directory, file, patch topic, `dx` or `install`; sometimes `docs(<sub>):`. Pin flips and repo-wide changes may carry no scope.
- Declarative subject, no trailing period, reference in trailing parens: `patches: auth-deny-default — --api-key guards everything but the liveness paths (#169)`, `bench: warm up against the model the launcher actually serves (#113) (#136)`.
- Upstream squash-merges PRs, so `main` has no merge commits and each subject ends in the PR number.
- Branches are a bare topic (`sse-keep-alive`, `fix-196-draft-vocab-txt`), `fix/…` or `docs/…`.
- **(fork)** `chore/…`, `validity/…` and `local/…` branches, merged into `master` with `--no-ff`.

**Docs.** `# Title`, one-sentence subtitle, then `[← back to the main README](../README.md)` (`../../` one level deeper). Knobs are `| var | default | notes |` tables. Numbers always carry card + power cap + units. Corrections are written *into* the entry and name the old claim; negative results are published rather than deleted.

## Important Files

| File | Why it matters |
|---|---|
| `verify.sh` | The gate everything funnels through. `ok`/`warn`/`fail` print `PASS`/`WARN`/`FAIL`; exits 1 iff `FAILS>0` (2 on a bad `--wait` argument), so WARNs never fail a run. Its model check is a `$PY - "$MODEL" <<'EOF'` heredoc **extracted verbatim** by `bench/test_model_verification.py` — the opening line and terminating `EOF` are a parsing contract, and it must stay the first heredoc opened exactly so. |
| `docker/prepare.sh` | `set -e`, serialized by `flock` (`PREPARE_LOCK_WAIT`, 600 s). `state()` is a `python - "$BASE" <<'EOF'` heredoc. A missing or torn config, index, tokenizer or shard means `download`; `intact()` checks each shard's declared bytes (#203). Otherwise index keys decide the pending steps, so the index is each step's commit point. `bench/test_prepare_state.py` runs the heredoc. |
| `resolve_config.sh` | Refuses unknown `CTX`/`SPEC` (single) and `KV` (batch) via `_refuse` → `exit 1` from inside the sourced function. Warns on cross-mode knobs and `EXTRA_ARGS`-shadowed flags; prints a redacted `[effective-config]`. Runs standalone: `bash resolve_config.sh single`. |
| `resolve_api_key.sh` | `resolve_vllm_key` (server: no placeholder, or the server demands a key nobody set; used by all three launchers) vs `resolve_client_key` (client: `OPENAI_API_KEY` → `VLLM_API_KEY` → `api_key.txt` → literal `EMPTY`, never a bare `Bearer `). |
| `single-user/select_model.sh` | Single source of the fast-variant rule. Sourced by the single-user launcher, `docker/entrypoint.sh` (single only, after prepare) and `bench/warmup.sh`, so verify, serve and warmup agree on `MODEL`. |
| `patches/series` | Apply order and source of truth; read only through `patches/apply.sh --list`, which exits 2 when it is not set-equal to `patches/*.patch`. A later patch's `Supersedes: <basename>` header lets `verify.sh` count the earlier patch, whose lines it rewrote, as applied; both stay in the series. |
| `PATCHES.md` | One row per patch: patch \| kind \| what \| upstream \| cut against \| retires when; kinds `backport`/`fix`/`feature`/`local`/`own`. Retired patches leave the tree and move to the retired prose. One export branch, `cpuchip/vllm` `qwen38/0.30` (tag `qwen38/0.30-cut5`). |
| `.env.example` | The compose `.env` template (`SPEC=dflash2`, `PREFIX_CACHE=1`, the WSL2 pin-memory flag, commented toggles). The full knob tables are the mode READMEs' `## Knobs`. **(fork)** `[managed]` marks launcher-computed values. |
| `docs/gotchas.md` | Numbered failure archive, cited by number from shell, Python, patches and PRs. |
| `docs/reproductions/README.md` | Harness rows `card \| power \| C1 decode \| notes \| source` (plus a C64 batch table), own-client bullets, and the index of full write-ups. |

## Runtime/Tooling Preferences

- **Linux/WSL2 only** for anything that runs. The repo is frequently *edited* on Windows; `bash` on a Windows host may resolve to WSL or Git Bash with different filesystem and loopback views — a localhost test that "refuses connection" is usually that, not a bug. Run git from the Windows side in worktrees created there: WSL's git cannot resolve their `gitdir` paths.
- **LF endings are mandatory** (`.gitattributes`: `* text=auto eol=lf`; **(fork)** adds `*.sh text eol=lf`, `*.patch text diff`, binary rules). A CRLF shell script or patch breaks `bash`/`patch` inside the image. Check with `git ls-files --eol`.
- Python is the image's `/app/venv/bin/python` (3.12, as is CI) or the native `venv/bin/python` (`python3 -m venv`; 3.14 works, `docs/python-314.md`). The image installs `pip -r docker/requirements.txt`; the native path is `docs/install.md`'s own pip line. The only non-Python tooling is `bench/demo/` (Node, demo renderer).
- **Never add `vllm[bench]`** or anything that resolves vLLM to `docker/requirements.txt` — it reinstalls the wheel and silently reverts every patch. `pandas` is pinned bare to stand in for the bench extra (#22); `pyarrow` is for `bench/quality_battery.py` (#159); the four `nvidia-cuda-*`/`nvvm` 13.0 pins hold FlashInfer's JIT nvcc at the 13.0 runtime (#225) — do not unpin them.
- `--fuzz 0` in the Dockerfile, `patches/check_vllm_series.sh`, `kvarn/install.sh` and `verify.sh`. Offsets are reported and fine; fuzz is a failure. The native loops in `docs/install.md`/`docs/python-314.md` do not pass it — `bash verify.sh --install` is what catches a fuzzed apply there.

## Testing & QA

Three tiers, and only the first is automatable:

1. **CPU/CI** — `patch-integrity.yml` runs `bench/test_model_verification.py` and `bench/test_prepare_state.py`: `unittest`, `subTest` + `TemporaryDirectory`, header-only safetensors fixtures built with `struct.pack("<Q", len(header)) + header`, and shell heredocs extracted and run verbatim. A new torch-free CPU test follows this shape and gets a step there. CPU-only but venv-bound (torch/vLLM), by hand: `bench/test_prepare_crash.py` (minutes; kills each writer mid-write), `bench/test_kvarn_recycled_pages.py`, `bench/test_bench_sse_keepalive.py` (all `CUDA_VISIBLE_DEVICES= venv/bin/python …`). `bench/verbatim.py`'s self-test is upstream but uncalled by upstream CI. **(fork)** `bench-policy.yml` runs `bench/verbatim.py`, `bench/test_fa_scratch_capacity.py` and `bench/test_paired_run_port.py`.
2. **GPU box, by hand** — `venv/bin/python bench/test_spec_decode_attn.py`, `test_spec_decode_bigpool.py`, `test_spec_decode_fp8.py`, `test_prefill_attn_bigpool.py`, `test_lookup_kernels.py`, `test_marlin_int8_asym.py`. Tolerances are literals (`< 0.05`; only the int8-QK prefill path relaxes to `0.5`). Designed skips exit 0 and say so.
3. **Live server** — `bash bench/run_benchmarks.sh [single|batch] [--prefill] [--long]` (default batch), `bench/quality_battery.py <tag> [--ppl-only|--gsm-only]`, `bench/concurrent_collapse.py <label> [30] [9000]` (KVarN recycled-page regression, #208; all trials on one boot). **(fork)** `bench/canary.sh`.

**Rules that decide whether a measurement is real:**

- `--model` is the **served name**, `--tokenizer` the checkpoint dir. Passing the path as `--model` 404s the client's `/tokenize` alignment probe and the run *silently* logs `WARNING: /tokenize unavailable` and reports wrong numbers.
- Every random-dataset call needs its own `--seed`; with prefix caching on, the default seed 0 silently hands later calls a partial cache hit (4.0 s vs 11.2 s cold for one 16k prefill).
- Run the harness twice and keep the second — the first after a restart reads 30–50% low.
- Thinking is pinned off per request in the Python probes (`chat_template_kwargs`). **(fork)** the shell cohorts pin `--chat-template-kwargs '{"enable_thinking": false}'` (protocol v2; v1 rows are never comparable to v2 rows).
- `vllm bench serve` exits 0 with zeroed metrics when every request failed. Without `bench-sse-keepalive.patch` (#226) it also marks any request failed whose prefill outlasts the launchers' 30 s SSE keep-alive. **(fork)** the drivers also require `Successful requests: [1-9]` and `Failed requests: 0`.
- `prefill_ab.sh` (PORT 18021) reuses a server already on the port (`FRESH=1` refuses) and, unless `KEEP=1`, tears down with `kill $(cat server.pid)` + `pkill -f "vllm serve.*$PORT"`. The pid file outlives the run, so a reused server can still be killed. **(fork)** servers are owned: an unowned port is refused unless `REUSE=1`, and a PID must map to `start_qwen.sh` in `/proc/<pid>/cmdline` before it is signalled.
- Results are evidence for a PR body, not repo content. `bench/results/` is gitignored and `run_benchmarks.sh` writes flat logs there. `prefill_ab.sh` writes `bench/results-prefill-ab/<arm>/`, which upstream does not ignore (**(fork)** ignores it). **(fork)** `run_benchmarks.sh` writes `results/run-<YYYYmmdd-HHMMSS>/manifest.json` (commit, dirty shortstat, vLLM client version, seed base, protocol tag; never the key) unless `OUT` is set. `prefill_ab.sh` writes `manifest.txt`, and `paired_run.sh` writes `<arm>-manifest.json`.

**Expected evidence in a PR:** the `ROW` lines a performance claim rests on, the issue link, the `OK`/`FAIL` output of any test the change touches (before and after), and for kernel/quant changes a `bench/quality_battery.py` run — "benchmarks great, outputs garbage" is the failure that battery exists to catch. **(fork)** also the `RESULT` line, the manifest path and a `bench/canary.sh` verdict.

## Traps Worth Knowing Before You Edit

- **Enumerated knobs live in three places**: `resolve_config.sh`'s validation list, the launcher's `if/elif` chain, and the mode README's `## Knobs` table. Miss the resolver and the value is refused before the launcher runs. Miss the launcher and only SPEC refuses (`SPEC=$SPEC is not a mode`): an unhandled CTX boots with no profile, and an unhandled KV silently becomes fp8 (**(fork)** CTX refuses too).
- **A new env knob has five homes**:
  - The patch registers it in vLLM's `envs.py` *in its own hunk* (so it is in the torch.compile cache key) and reads it via `vllm.envs`.
  - The `PATCHES.md` row names the knob (e.g. `registers VLLM_SPEC_ATTN_DEBUG`).
  - `.env.example` gets a commented line.
  - The owning mode README's `## Knobs` table gets a row.
  - If it is enumerated, `resolve_config.sh` must learn it.
  - A knob that never reaches compiled code goes into `compile_factors()`'s `ignored_factors` instead (`patches/compile-key-runtime-knobs.patch`), so changing it does not force a cold compile.
- **A new patch touches four things**: the generated `.patch`, `patches/series`, a `PATCHES.md` row, and an `envs.py` hunk if it reads a knob. Patch files are artifacts of a `[qwen38] <topic>` commit on the vLLM fork branch — regenerate with `scripts/export-patch.sh`, never hand-edit.
- **KVarN is installed after the series and is not in it** (`kvarn/install.sh`: `kvarn-0.30.0`, `kvarn-v2-runner-0.30.0`, `kvarn-recycled-pages-0.30.0`). Rebuilding the venv or replaying `patches/` silently drops it; `CTX=huge`/`KV=kvarn` then boot into an unknown cache dtype. `verify.sh` WARNs when only the recycled-pages patch (#208) is missing.
- **Head groups must never declare zero points** (`group_1..group_3`, forced `symmetric=true`/`zp_dtype=null`, #197). A body group with `symmetric=false` is legitimate for AWQ — keep that asymmetry.
- **A tensor must live in exactly its mapped shard**: vLLM loads every key of every `.safetensors` it opens, so a hardlink-assembled dir carrying a superseded copy loads it twice.
- **A model dir without `tokenizer.json` is not an error to transformers** — it yields a 1-token vocabulary and dies much later at "failed to tokenize reasoning strings".
- **Memory constants are a coupled set** (`KV_MEM`, `GPU_UTIL`, `MAX_LEN`, `CG ≤ 64`). Since 0.29, vLLM profiles CUDA-graph memory itself; `VLLM_V2_CUDAGRAPH_MEM_MIB` was 0.28's, and nothing reads it now. An exported `KV_MEM` applies in every SPEC mode (#214) and is dropped at TP>1 unless exported. Batch under WSL2 defaults `GPU_UTIL` to 0.91 (0.88 and `MAX_LEN=131072` for `KV=kvarn`, #235). Raising one constant alone yields a server that boots, answers `/health`, and dies on the first real prompt. On WSL2 it does not error at all; it runs 5–10× slower.
- **Draft profiles always pass `--prefix-cache-retention-interval`** (#189, gotcha 60). The 0.30 default for an unset value is 0 (boundaries only); 0.29's was dense.
  - Measured values: 13056/14592 for `CTX=huge SPEC=dflash2` at 7/15 drafts, else `None`.
  - Precedence: a flag in `EXTRA_ARGS` wins, then an exported `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, then `PREFIX_RETENTION` (0 = boundaries, empty = dense).
  - The engine refuses at boot a value that is not a multiple of the scheduler block.
- **`--dtype float16` in `EXTRA_ARGS` needs `SPEC=off`**: the split-KV verify attention hardcodes bf16, and the single-user launcher refuses the pair (#27).
- **No key means an open server**: the single-user launcher binds `${HOST:-0.0.0.0}`, batch and `alternative.sh` hardcode `0.0.0.0`, compose publishes on every interface, and `verify.sh` only WARNs (#204, open).
- **Gotcha numbers are positional and have drifted.** Append and correct in place. If you must renumber, sweep the tree: references live in `verify.sh`, all three launchers, `docker/prepare.sh`, several bench scripts and most docs (**(fork)** also `.env.example`). Inside `docs/docker.md`, "gotcha N below" means that file's own local list. Known drift: the single-user launcher's PIECEWISE comment and fork `.env.example` cite gotcha 37 for the FULL-capture corruption, which is gotcha 33.
- **`EXTRA_ARGS` wins over the launcher's flags**, except `--language-model-only`. Every launcher passes it via `VISION_ARGS` whenever `VISION≠1`, and countering it from `EXTRA_ARGS` regresses silently (images accepted, answered from placeholder embeddings). Use `VISION=1`.
- **(fork)** **A new file under `docs/` is silently uncommittable** until `!docs/<name>.md` is added to `.gitignore` (`git add` warns; `git status` stays silent). The allowlist is not the upstream doc set; tracked upstream docs are unaffected.
- **(fork)** **`.github/workflows/bench-policy.yml` pins source text**, not behaviour:
  - Eight named scripts must contain `RESULT FAIL` and `sys.exit(`.
  - `run_benchmarks.sh`/`prefill_ab.sh`/`real_rep.sh` must carry `chat-template-kwargs` and `Successful requests`.
  - `run_benchmarks.sh` keeps `cohort-latency-score`.
  - `labd_bench.py` keeps `d.get(k, 0.0) + float`.
  - `docker-compose.yml` keeps `profiles: [headroom]` and no `VLLM_API_KEY:?`.
  - `check_vllm_series.sh` keeps `GIT_ROOT" = "$HERE"`.
  - Reflowing those lines reds CI in a file you did not think you changed.
