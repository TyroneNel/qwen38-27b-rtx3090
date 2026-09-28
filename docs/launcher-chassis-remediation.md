# Extract the shared launcher chassis — remediation plan

Architecture review 2026-09-26, candidate 2 (Strong). The full report is
[architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card. Every number below was measured against the
tree on 2026-09-26 (method noted where it matters); nothing is carried over
from the review on trust.

**Scope.** `single-user/start_qwen.sh`, `batch/start_qwen.sh`,
`single-user/alternative.sh`, the two systemd units, and the validation layer
(`resolve_config.sh`, `resolve_api_key.sh`, `single-user/select_model.sh`) —
plus a dry-run seam that makes all of it testable without a GPU.
**Not in scope:** merging the two `start_qwen.sh` into one MODE-parameterized
file (a possible endpoint, rejected as a separate project in the review);
changing any profile's *values* (this moves code, not measurements); the
harness-side client duplication (candidate 3); the service units' contents.

## 1. Current state, precisely

### 1.1 The census — it is three launchers, not two

| file | lines | unique substantive lines¹ | shared verbatim with single¹ | conditional statements² |
|---|---|---|---|---|
| `single-user/start_qwen.sh` | 822 | 297 | — | 47 |
| `batch/start_qwen.sh` | 271 | 110 | **76 (69%)** | 13 |
| `single-user/alternative.sh` | 133 | 81 | **40 (49%)** | 6 |

¹ Method: `comm -12` on unique non-comment, non-blank lines. With duplicates
and comments included, single∩batch share **180 lines** (156 unique).
² `grep -cE '^\s*(if|case|elif) '` — the env/platform decision points.

The exec lines share **12 of 13** literal flags (measured: the union minus
batch's hardcoded `--async-scheduling` and single's `--sse-keep-alive-interval`).
`alternative.sh` (the experimental int4-KV profile, PR #42) is the most drifted
copy — see 1.3.

There are also two systemd units, `single-user/qwen-serving.service` (21 lines)
and `batch/qwen-serving.service` (18 lines), identical except the description,
the ExecStart path, and a warmup comment — a fourth, shallow copy of "how to
boot this stack," left alone by this plan (1.5).

### 1.2 The chassis: fourteen blocks, where each lives today

| # | block | single-user | batch | alternative.sh |
|---|---|---|---|---|
| 1 | env prelude (DIR/REPO, `FLASHINFER_DISABLE_VERSION_CHECK`) | :55-59 | :26-30 | :29,31 |
| 2 | stale-`/dev/shm` offload sweep (#33) | :61-72 | :32-43 | :49-56 (3rd copy) |
| 3 | CUDA_HOME/nvcc-13 fix (#185) | :75-91 | :46-62 | **absent** |
| 4 | resolve_config boilerplate | :93-99 | :64-71 | **absent** |
| 5 | INT8 export guards (#20) | :137-140 | :113-114 + :242-248 | :44-47 |
| 6 | PREFIX_CACHE arm | :509-514 (+:515-554 single-only extension) | :116-125 | :84,89-90 |
| 7 | TOOL_ARGS (#59, qwen3_coder) | :646-665 | :127-146 | :130 (hardcoded) |
| 8 | METRICS_ARGS (#51, #59) | :667-685 | :148-162 | :97-111 (3rd copy) |
| 9 | VISION block (gotcha 9) | :687-720 | :164-197 | :76,86-87 (simplified) |
| 10 | WSL2/allocator + KV-connector + TP blocks (#2/#26, #95, #163) | :750-797 | :199-238 | :13-28 (**TP block absent**) |
| 11 | `VLLM_USE_FLASHINFER_SAMPLER=0` | :798 | :239-241 | :30 |
| 12 | ASYNC_SCHED→ASYNC_ARGS | :313 + :637-644 | :264 (hardcoded) | :83 + :92-95 |
| 13 | key resolution | :800-801 (shared resolver) | :250-252 (shared resolver) | :32 (**own flavor**) |
| 14 | exec `vllm serve` skeleton | :803-822 | :254-271 | :113-133 |

Byte-identical spot checks (verified by `sed`): the PREFIX_CACHE arm
`batch:124` == `single:514`; the ASYNC_ARGS pair `single:643-644` ==
`alternative.sh:94-95`; blocks 1, 2, 7, 9, 10 are near-verbatim between the
two `start_qwen.sh` (comment drift noted in 1.3).

`select_model.sh` (7 lines) is already the shared model selector — three
consumers (`single-user/start_qwen.sh:101`, `docker/entrypoint.sh:25`,
`bench/warmup.sh:41`) plus a CI replay (`bench/test_model_verification.py:76-79`).
It is the proof that the extract-and-source pattern works in this repo.
### 1.3 Confirmed drift between the copies

Each item verified against the files on 2026-09-26; this is the cost the
duplication is already charging.

1. **The INT8 export guards disagree** (#20's "export only when non-empty"
   fix). `batch/start_qwen.sh:247-248` exports `VLLM_MARLIN_INT8_INCLUDE_RE`
   whenever `INT8_LAYERS` is non-empty — even with `INT8_ACT` empty, which
   leaves the engine with an include-regex and no input dtype.
   `single-user/start_qwen.sh:139-140` and `alternative.sh:46-47` require both.
   Batch is the outlier; the bug is latent (its `INT8_ACT` defaults to `int8`
   at `:113`) and fires on `INT8_ACT= INT8_LAYERS=mlp bash batch/start_qwen.sh`.
2. **Batch carries spec-decode metrics flags it can never produce.**
   `batch/start_qwen.sh:154-161` sets `--per-request-spec-decode-metrics`;
   batch mode enables no speculative decoding. The block was copied from
   single (`:677-684`), where the flag is real — and the comment headers have
   already drifted apart (single's `#51`/llama-swap essay at `:667-672` vs
   batch's two-line summary at `:148-151`).
3. **Batch's VISION_OFFLOAD comment cites a mode it never runs.**
   `batch/start_qwen.sh:182-188` explains the default with "on 24 GB
   SPEC=dflash2 + VISION=1 does not boot without it … measured here …
   SPEC=dflash2, RTX 3090" — SPEC=dflash2 and the KV_MEM margin it references
   exist only in single (`:705-711`, the original). A batch reader is sent to
   concepts their launcher does not have.
4. **The `HOST` knob exists only in single.** `single-user/start_qwen.sh:805`
   has `--host ${HOST:-0.0.0.0}`; `batch/start_qwen.sh:256` hardcodes
   `--host 0.0.0.0`. Same flag position, one knob lost in the copy.
5. **Async scheduling is three shapes.** Single computes it
   (`ASYNC_SCHED` at `:313` for the long DFlash2 verify block → `ASYNC_ARGS`
   array at `:643-644`); `alternative.sh:83,94-95` duplicates that logic
   verbatim; `batch/start_qwen.sh:264` hardcodes `--async-scheduling`.
6. **The CUDA-graph capture-size formula has three homes.**
   `single-user/start_qwen.sh:442` (`${CG:-...}`-guarded),
   `alternative.sh:81` (same arithmetic, unguarded), `batch/start_qwen.sh:266`
   (hardcoded `64` in the exec line).
7. **`alternative.sh` never received two platform fixes** that upstream landed
   in both `start_qwen.sh` files: the CUDA_HOME/nvcc-13 fix (`fa97789`, #185 —
   touched `batch/start_qwen.sh` + `single-user/start_qwen.sh` only) and the
   TP>1 allocator default (`d2a5538`, #176 — same two files; `alternative.sh`
   has the WSL2 and KV-connector arms at `:13-28` but not the TP arm).
8. **`alternative.sh` resolves the API key its own way**:
   `export VLLM_API_KEY="$(cat api_key.txt)"` (`:32`) — no env precedence, no
   file check, and under its own `set -e` (`:11`) a missing `api_key.txt`
   kills the script with a bare `cat` error. The shared resolver
   (`resolve_api_key.sh`, written for #113) exists and is unused there.
9. **A stale patch filename in a comment.** `single-user/start_qwen.sh:219`
   cites `kvarn-v2-runner-0.28.0.patch`; the file on this line is
   `kvarn/kvarn-v2-runner-0.29.0.patch` (the 0.29 pin flip in #148 renamed it;
   the comment was not updated).
10. **The model decision lives in five places.** `select_model.sh:4-7` (the
    shared one), `batch/start_qwen.sh:73` (inline, base-only),
    `alternative.sh:58-59` (inline, **relative** paths — breaks outside the
    repo root, and never prefers `-fast`), `resolve_config.sh:82` (print
    default), `verify.sh:21` (check default).
11. **The comments are forking.** Batch's WSL2 arm now says "see the long note
    in single-user/start_qwen.sh" (`batch:200-201`) — the code is duplicated
    but the explanation already migrated. The repo is doing this refactor by
    hand, one comment at a time.

### 1.4 The validation layer has drifted from what it validates

`resolve_config.sh` (the F13 fix) watches the launchers from the outside:

- **Its EXTRA_ARGS shadow list (`resolve_config.sh:68`) names 11 flags.** The
  three launchers' exec lines emit 17 literal flags, plus ~13 more carried in
  `$KV_ARGS`/`$ATTN_ARGS`/`$SPEC_ARGS`/`$EXTRA_ARGS` insertions. Measured gaps
  include `--compilation-config` (both `start_qwen.sh` set it),
  `--async-scheduling`, `--max-num-batched-tokens`, `--reasoning-parser`,
  `--speculative-config`, `--enable-prefix-caching`, `--mamba-cache-mode`,
  `--kv-cache-memory`, `--prefix-cache-retention-interval`,
  `--sse-keep-alive-interval`, `--default-chat-template-kwargs`. Any of those
  in EXTRA_ARGS silently shadows the launcher — the exact failure class the
  resolver was built to make visible.
- **Its MODEL print is wrong on the native single path.** Single calls
  `resolve_effective_config single` at `:99` but selects the model at `:101`;
  `resolve_config.sh:82` defaults to the base dir, so `[effective-config]
  MODEL=` prints the base model even when the `-fast` variant will be served.
  Docker masks this (entrypoint:23-27 exports MODEL first); a native boot
  prints the wrong checkpoint. (Verified by call order.)
- **`alternative.sh` is entirely outside it** — no validation of its SPEC
  (its own `case` at `:65-71` refuses, duplicating the resolver's job).

### 1.5 Change pressure

`single-user/` + `batch/` took 19 file-touches in the last 40 upstream commits
(second only to `patches/`). The launcher-affecting ones repeatedly land in
the shared blocks: #185 and #176 each had to edit both `start_qwen.sh` files
(verified: file lists of `fa97789`, `d2a5538`), and #126's model-selection fix
touched five files for one behavior (`25bd8d2`: select_model, entrypoint,
single launcher, the CI test, docs).
## 2. The design

### 2.1 `launcher_common.sh` — the chassis, at the repo root

A new sourced library beside `resolve_config.sh` and `resolve_api_key.sh`
(the repo's existing convention for shared shell). Functions, not inline
expansion, so each block keeps its gotcha essay as the function's comment —
**the comments move with the code they explain.** One source file; both
`start_qwen.sh` and `alternative.sh` source it after `REPO` is set.

| function | absorbs (table in 1.2) | notes |
|---|---|---|
| `qwen_env_prelude` | blocks 1-3 | DIR/REPO stay with the caller; the flashinfer pin, shm sweep, and CUDA_HOME fix move verbatim, comments included. Fixes 1.3.7 for `alternative.sh` by inclusion. |
| `qwen_allocator_defaults` | block 10 | WSL2 detect + KV-connector + TP arms → sets `PYTORCH_CUDA_ALLOC_CONF`. The long WSL2 essay (single:751-760) lives here; batch's "see the long note" pointer (1.3.11) dissolves. |
| `qwen_int8_exports` | block 5 | the two-condition guard (1.3.1's fix) once; callers keep their own `INT8_ACT`/`INT8_LAYERS` *defaults*, which are genuinely per-mode (batch: `int8`/`mlp`; single: off/`all`; alternative: off/`all`). |
| `qwen_tool_args` / `qwen_metrics_args` / `qwen_vision_args` | blocks 7-9 | the array builders with their #59 comments. The spec-decode metrics flag moves behind a `mode` parameter so batch stops carrying flags it cannot produce (1.3.2) — or simpler: the flag is harmless-but-dead in batch; keep it only if the owner wants one code path. Called out in the PR, not decided here. |
| `qwen_async_args` | block 12 | `ASYNC_SCHED` → `ASYNC_ARGS`; single's `:313` setter stays in single (it is dflash2-shaped), the array build shares. |
| `qwen_serve_argv` | block 14 | the exec line as an **array builder**: fills `ARGV=(venv/bin/vllm serve …)` from the caller's mode parameters. EXTRA_ARGS still expands last and still word-splits (the documented override door — resolve_config.sh:34-35); the array conversion makes the other expansions (`$VISION_ARGS`, `$KV_ARGS`, `$ATTN_ARGS`) explicit instead of relying on unquoted splitting. |
| key resolution | block 13 | stays in `resolve_api_key.sh` (already shared); `alternative.sh` switches to it (1.3.8). |

**Deliberately not shared:** the header essays (they are mode-specific
measurement reports — single's `:24-53` context tiers, batch's `:5-24` state
economics); the profile branches (CTX×SPEC and the whole DFlash2/MTP/SPEC_CFG/
KV_MEM/residency machinery in single:191-596; KV in batch:77-110); the PREFIX_
CACHE extensions single carries (its `:515-554` retention ladder is
single-shaped today; batch's arm is the base form). Sharing those would move
complexity, not concentrate it — they fail the deletion test.

### 2.2 `PRINT_ARGV=1` — the dry-run seam

Each launcher ends, instead of a bare `exec`, with:

```bash
qwen_serve_argv            # fills ARGV
if [ "${PRINT_ARGV:-0}" = 1 ]; then
  printf '%s\n' "${ARGV[@]}"           # one argument per line
  exit 0                               # before exec, after all resolution
fi
exec "${ARGV[@]}"
```

The dry run runs *all* the resolution logic (profiles, guards, allocator
decisions, warnings) and stops only at the exec — so the platform probes that
are already read-only (nvcc detection, WSL2 detection, the EXTRA_ARGS parse)
still run, and the one mutating step (the stale-shm unlink, which only removes
files no live process maps) behaves exactly as it does on a real boot. No GPU
is ever touched: the launchers make no CUDA calls before exec today (verified
by reading: every probe is `/proc`, `nvcc --version`, `du`, or a grep).

This converts the launcher layer from "reviewable only by eye" to
"assertable in CI": `CTX=huge PRINT_ARGV=1 bash single-user/start_qwen.sh`
must print `--kv-cache-dtype kvarn_k4v2_g128` and refuse without it, etc. —
the test matrix in section 4.

### 2.3 The validation layer catches up

- **The shadow list derives from the assembled argv.** With `ARGV` in hand,
  `resolve_config.sh`'s hand-maintained 11-flag list (1.4) is replaced by a
  chassis function `qwen_shadow_warnings` that greps EXTRA_ARGS against the
  flags the *current* argv actually set — the warning can never drift from
  the exec line again. (The flag list the launchers own becomes the list the
  validator checks.)
- **The model-print bug (1.4) dies by ordering**: single sources
  `select_model.sh` *before* `resolve_effective_config single`, so the
  printed `[effective-config] MODEL=` is the checkpoint that will actually
  boot. `resolve_config.sh:82`'s default becomes a fallback, not the answer.
- **`alternative.sh` joins the validation**: it sources `resolve_config.sh`
  and validates its own `SPEC` enum (`dflash2|off|none`) through the same
  refuse shape, deleting its duplicate `case` at `:65-71`.

### 2.4 The drift fixes that fall out (not the goal — the proof)

1.3's items resolve as follows: 1 (INT8 guards) → one shared guard. 2
(spec-decode flag in batch) → mode-parameterized builder, or a documented
harmless keep. 3 (batch's dflash2 VISION comment) → the comment lives once,
in the chassis, citing the mode it belongs to. 4 (HOST knob) → both launchers
get `--host ${HOST:-0.0.0.0}` from the shared argv builder (a behavior change
for batch; flagged to the owner — smallest possible one). 5, 6 (async, CG) →
one builder each. 7 (alternative missing #185/#176) → inclusion. 8 (key
flavor) → the shared resolver. 9 (stale kvarn name) → fixed in passing, one
comment. 10 (five model defaults) → `select_model.sh` stays the one
selector; `alternative.sh:58-59` and `resolve_config.sh:82` defer to it;
`verify.sh:21` stays its own (it checks, it does not select).
11 (forking comments) → comments live in the chassis functions.
## 3. Rollout

Three PRs, ordered so the behavior-preserving one proves itself before any
behavior changes land. Each is independently revertable.

### 3.1 PR A — the chassis + the dry-run seam (no behavior change)

Add `launcher_common.sh`; move blocks 1-3, 5, 7-12 of the two `start_qwen.sh`
into it **verbatim** (code and comments); both launchers source it; both
exec lines become `qwen_serve_argv` + the `PRINT_ARGV=1` gate. `alternative.sh`
is untouched in this PR. The `dflash2-backport`-style temptation to clean up
comments while moving them is resisted: the diff must read as pure moves.

Acceptance: `bash -n` clean on all three shells' targets; **byte-identical
argv** — for each profile in the test matrix (section 4), `PRINT_ARGV=1` on
main vs the branch must diff empty; `patch-integrity`'s existing
`test_model_verification.py` still passes; the image build is green (it runs
`verify.sh --install`, not the launchers, so it is a smoke check only).

### 3.2 PR B — `alternative.sh` joins the chassis (real fixes, called out)

`alternative.sh` sources the chassis and the two resolvers; its own copies of
the WSL2/KV-connector arms, the stale-shm sweep, the INT8 guards, the
METRICS/ASYNC arrays, and the SPEC case are deleted. Explicit behavior changes,
each its own commit message line: gains the CUDA_HOME fix (1.3.7a), gains the
TP allocator arm (1.3.7b), key resolution via `resolve_api_key.sh` (1.3.8),
absolute model paths (1.3.10), the guarded CG formula (1.3.6), and a
`PRINT_ARGV` gate of its own. Also fixes 1.3.9's stale comment.

Acceptance: the dry-run argv for its two profiles matches the pre-PR argv
*except* in the enumerated fixes; a boot smoke on the int4 profile.

### 3.3 PR C — the validation layer catches up + CI assertions

`qwen_shadow_warnings` replaces the hand list in `resolve_config.sh`;
single reorders select-then-resolve (1.4); a new `bench/test_launcher_argv.py`
(in `test_model_verification.py`'s existing unittest style) runs the matrix
below against `PRINT_ARGV=1` and is wired into `patch-integrity.yml`'s
CPU job — it runs in seconds, no GPU. The two systemd units stay as they are;
their only change is optional (a comment pointing at the chassis).

Acceptance: the new test file is in the workflow and green; `grep -c`
on `resolve_config.sh`'s shadow list is gone in favor of the derived check;
the `[effective-config] MODEL=` print matches `select_model.sh` on a native
single-mode dry run.

## 4. Test plan (no GPU anywhere)

The `PRINT_ARGV=1` matrix — run on main and on the branch, argv must be
byte-identical for PR A (a smaller "known-different" set for PR B):

| env | expect in argv / behavior |
|---|---|
| defaults (single) | `--kv-cache-dtype bfloat16`, `--speculative-config` with `"method":"mtp"`, CG=32 |
| `CTX=long SPEC=mtp` | `--kv-cache-dtype fp8`, **no** `--attention-backend FLASH_ATTN` |
| `CTX=huge SPEC=dflash2 PREFIX_CACHE=1` | `--kv-cache-dtype kvarn_k4v2_g128`, `--block-size 128`, `--prefix-match-unit 128`, the retention interval |
| `SPEC=dflash2 DFLASH_TOKENS=15` | `--no-async-scheduling`, CG capped at 64 |
| `SPEC=off` | no `--speculative-config` at all |
| `EXTRA_ARGS="--tensor-parallel-size 2"` | `PYTORCH_CUDA_ALLOC_CONF` printed as `expandable_segments:False` (the warning text asserts too) |
| `EXTRA_ARGS="--compilation-config …"` | the shadow warning fires (PR C) |
| `CTX=bogus` | refusal, exit 1, nothing printed to argv |
| `INT8_ACT= INT8_LAYERS=mlp bash batch/…` | no `VLLM_MARLIN_INT8_INCLUDE_RE` exported (1.3.1) |
| `VISION=1` | the mm flags, no `--language-model-only` |
| `HOST=127.0.0.1` (batch, PR C) | `--host 127.0.0.1` |

Plus the existing `bench/test_model_verification.py` (it replays
`select_model.sh` and the entrypoint — a canary that the shared selector is
untouched) and `bash -n` over every touched file in CI.

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| The verbatim-move PR (A) subtly changes quoting/word-splitting the eye misses | medium — it is why PR A is moves-only | the byte-identical-argv acceptance gate; the array conversion is the *only* semantic change and it is where the review effort goes |
| EXTRA_ARGS' documented word-splitting is accidentally array-quoted | medium | `qwen_serve_argv` re-splits EXTRA_ARGS deliberately, with a comment citing resolve_config.sh:34-35; a test row asserts it |
| Sourcing breaks the callers' `set -e`/errexit edges (`#59` class) | low | the chassis functions are arrays-and-tests throughout, the shape #59 blessed; CI runs `bash -n` and the dry-run matrix |
| Someone edits a profile value inside the chassis thinking it is shared, changing both modes | low | profile *values* never enter the chassis (2.1's "not shared" list); the file's header says so |
| `alternative.sh`'s fixed paths/key handling surprise a user of the experimental profile | low | PR B's message enumerates the behavior changes; the profile is documented experimental |
| The dry-run becomes a way to "test" a boot that then OOMs | low | the doc is explicit: PRINT_ARGV validates *argument assembly*, never that the config fits the card — the launcher's own measured ladders keep that job |

## 6. Done when

1. `grep -c "expandable_segments" single-user/start_qwen.sh batch/start_qwen.sh`
   → 0 and 0; the string lives once, in `launcher_common.sh`.
2. `grep -c "qwen3_coder" batch/start_qwen.sh single-user/alternative.sh`
   → 0 and 0 (the parser name lives once, in the chassis).
3. `PRINT_ARGV=1` produces the full argv from all three launchers, and
   `bench/test_launcher_argv.py` runs green in `patch-integrity.yml`'s CPU job.
4. `resolve_config.sh` contains no hand-maintained flag list.
5. `comm -12 <(sort -u single-user/start_qwen.sh) <(sort -u batch/start_qwen.sh)`
   on unique substantive lines drops from 76 to near zero — what remains is
   the profile data that genuinely differs.
6. A native `PRINT_ARGV=1` single-mode run prints the `-fast` model when it
   exists (1.4's bug is gone).
