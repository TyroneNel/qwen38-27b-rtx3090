# Extract the shared launcher chassis — remediation plan

Architecture review 2026-09-26, candidate 2 (Strong). The full report is
[architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card. Every number below was measured against the
tree on 2026-09-26 (method noted where it matters); nothing is carried over
from the review on trust.

> **Re-verified 2026-09-29 against upstream/main @ d5e2a01 (vLLM 0.30.0).**
> Every count re-measured with the command stated beside it; every line cite
> re-mapped (files grew again: 838/279/142 at 2522ef9 → **842/299/160**,
> `wc -l`). Seven commits landed since 2522ef9; three touch the launcher
> scripts (#219, #232, #235; `git diff 2522ef9 HEAD --stat -- single-user batch resolve_config.sh`):
> - **Fixed upstream:** drift item 8. 8cf642e (#219, merged 2026-09-28)
>   replaced `alternative.sh`'s bare `cat api_key.txt` with the shared
>   `resolve_api_key.sh` (`alternative.sh:32-41`) and gave the script a
>   `REPO` (`:35-37`) — PR B's key-resolution step is done, and the `REPO` it
>   needs to source the chassis now exists.
> - **Changed / newly skewed:** 36936ec (#232) rewrote `alternative.sh`'s
>   retention copy (now `:100-117`): its default is `0` without a KV tier and
>   `None` with one. It is no longer "`None` only", and single's ladder still
>   falls back to `None` — so the plan's "argv-identical unless
>   `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` is exported" was false after #232;
>   `qwen_retention_args` now needs a caller-supplied fallback (2.1, 3.2).
>   The #232 PR checked its change with "a stub `vllm` on the launcher's
>   `PATH` that prints its argv" — a hand-built version of 2.2's dry-run seam,
>   and one that only works on `alternative.sh` (`exec vllm serve`, `:140`):
>   both `start_qwen.sh` exec `venv/bin/vllm` (`single:823`, `batch:282`).
> - **Newly introduced duplication:** #219 added a guarded, dflash2-only
>   `VLLM_WSL2_ENABLE_PIN_MEMORY=1` default inside single's WSL2 arm
>   (`single:779-782`); `alternative.sh:15` still exports it unconditionally
>   (new drift item 13). 1bb8e4e (#235) moved batch's WSL detection into a
>   `WSL` variable (`batch:95-96`) that now drives WSL-only `GPU_UTIL`/`MAX_LEN`
>   defaults (`:98-107`) as well as the allocator arm (`:230`), so batch's block
>   10 now depends on state set 130 lines earlier (2.1's allocator row adjusted).
> - **Unchanged:** drift items 1-7, 9-11 stand at new line numbers (batch +20
>   after `:86`; single +4 after `:778`; `alternative.sh` +9 after `:31`, +9
>   more after `:99`). Batch and `alternative.sh` still have no `KV_MEM`
>   handling (#235 added `GPU_UTIL`/`MAX_LEN`, not `KV_MEM`). `resolve_config.sh`
>   is untouched since 2522ef9 — still 11 names at `:68`. The shared-line
>   count single∩batch is still 76; percentages fell because batch grew.
> - Corrections to claims that were already wrong at 2522ef9: 1.5's "second
>   only to `patches/`" (docs/ had 38 touches, so the launchers were third);
>   1.3.5's "verbatim" async copy (it drops single's `VLLM_DFLASH2_LOOKUP=1`
>   gate); 2.1's "off/`all`" INT8 layer default (it is the literal
>   `mlp|linear_attn|self_attn`); 1.4's "~13 more" flags (16 more, 13 unwatched).

> **Previous note — re-verified 2026-09-28 against upstream/main @ 2522ef9
> (vLLM 0.30.0).** Kept as history; its line numbers are those of 2522ef9.
> All numbers re-measured; every line cite re-mapped (files grew: 838/279/142).
> - 22 upstream commits landed since 1cf8665. The pin flip to vllm==0.30.0
>   (d88544b, #189) rewrote single's retention block (:524-570) and landed a
>   **second, simplified copy in `alternative.sh` (:91-99) in the same commit** —
>   the newest duplication, and its commit message says the block was "extracted
>   and run over ten input cases (both launchers)": upstream hand-built the
>   chassis unit test this doc proposes. Batch has no retention block,
>   correct-by-design (no drafter; the commit message says so).
> - c970467 (#214) moved the `--kv-cache-memory` append out of the dflash2
>   branch to follow the SPEC chain (single:507-511) — a one-launcher fix of
>   exactly the copy-drift bug class the chassis deletes; batch and
>   `alternative.sh` have no KV_MEM handling at all.
> - 1a4bf64 (#207) touched both `start_qwen.sh` + verify.sh — the third
>   both-launcher fix after #185/#176 — and #189 touched all three launchers
>   at once. 1.5's pressure is undiminished (21 launcher file-touches in the
>   last 40 commits, was 19).
> - All 11 drift items stand (re-cited); #9 is worse — the stale kvarn patch
>   name survived a *second* pin flip (single:221 cites 0.28.0; the tree now
>   carries `kvarn-v2-runner-0.30.0.patch`). New item 12: #189's two retention
>   copies are already capability-skewed. Neither #189 nor #214 updated
>   `resolve_config.sh`'s shadow list — 1.4's predicted failure, on schedule.
> - Two original numbers did not reproduce and are corrected below: single's
>   substantive-line count (297 → it is 287 by the stated method at 1cf8665;
>   289 now) and the "byte-identical" ASYNC_ARGS pair (line 2 has always
>   differed by the `:-1` fallback — near-verbatim, never identical).

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

| file | lines | substantive lines¹ | shared verbatim with single¹ | conditional statements² |
|---|---|---|---|---|
| `single-user/start_qwen.sh` | 842 (838 at 2522ef9) | 293 (289) | — | 48 (47) |
| `batch/start_qwen.sh` | 299 (279) | 122 (110) | **76 (62%; was 76, 69%)** | 18 (13) |
| `single-user/alternative.sh` | 160 (142) | 98 (87) | **41 (42%; was 40, 46%)** | 10 (8) |

¹ Substantive = non-comment, non-blank (`grep -cE '^[[:space:]]*[^#[:space:]]'`);
the shared column is `comm -12` on the `sort -u` of those lines (unique
substantive lines: 249 / 106 / 94 now, 246 / 97 / 83 at 2522ef9), and the
percentage divides it by the file's *non-unique* substantive count (76/122,
41/98) — the method that reproduces the 2522ef9 figures (76/110 = 69%,
40/87 = 46%). With duplicates and comments included (`comm -12` on plain
`sort` of whole files), single∩batch share **184 lines** (was 182; 158 unique,
unchanged — `comm -12` on `sort -u`).
² `grep -cE '^\s*(if|case|elif) '` — the env/platform decision points. Batch's
+5 are #235's WSL `GPU_UTIL`/`MAX_LEN` defaults (`batch:96-104`); single's +1
is #219's pin-memory `if` (`:779`); `alternative.sh`'s +2 are #232's tier
`case` and `PREFIX_RETENTION` `if` (`:110`, `:114`).

The exec lines share **12 of 13** literal flags (re-measured at d5e2a01,
unchanged: every `--flag` token from the `exec` line to end of file, each
`start_qwen.sh` carries 13; a 14-flag union minus batch's hardcoded
`--async-scheduling` and single's `--sse-keep-alive-interval`). `alternative.sh`
(the experimental int4-KV profile, PR #42) is the most drifted copy — see 1.3.

There are also two systemd units, `single-user/qwen-serving.service` (21 lines)
and `batch/qwen-serving.service` (18 lines), identical except the description,
the ExecStart path, and two comment wordings (one pool line; single's 3-line
warmup block against batch's one) — a fourth, shallow copy of "how to
boot this stack," left alone by this plan (1.5).

### 1.2 The chassis: fifteen blocks, where each lives today

| # | block | single-user | batch | alternative.sh |
|---|---|---|---|---|
| 1 | env prelude (DIR/REPO, `FLASHINFER_DISABLE_VERSION_CHECK`) | :55-61 | :26-32 | :29,31 + :35-37 (`ALT_DIR`/`ALT_REPO`/`REPO`, added by #219) |
| 2 | stale-`/dev/shm` offload sweep (#33) | :63-74 | :34-45 | :58-65 (3rd copy; was :49-56) |
| 3 | CUDA_HOME/nvcc-13 fix (#185) | :77-93 | :48-64 | **absent** |
| 4 | resolve_config boilerplate | :95-101 | :66-73 | **absent** |
| 5 | INT8 export guards (#20) | :139-142 | :133-136 + :270-276 (was :113-116 + :250-256) | :53-56 (was :44-47) |
| 6 | PREFIX_CACHE arm | :515-523 (+:524-644 single-only extension) | :138-153 (was :118-133) | :93,98-99 (was :84,89-90) |
| 7 | TOOL_ARGS (#59, qwen3_coder) | :662-681 | :155-174 (was :135-154) | :157 (hardcoded; was :139) |
| 8 | METRICS_ARGS (#51, #59) | :683-701 | :176-190 (was :156-170) | :124-138 (3rd copy; was :106-120) |
| 9 | VISION block (gotcha 9) | :703-736 | :192-225 (was :172-205) | :85,95-96 (simplified; was :76,86-87) |
| 10 | WSL2/allocator + KV-connector + TP blocks (#2/#26, #95, #163) | :766-817 (was :766-813; +#219's pin-memory arm :779-782) | :227-266 (was :207-246) + WSL detect hoisted to :95-96 by #235 | :13-28 (**TP block absent**; unconditional pin-memory export :15) |
| 11 | `VLLM_USE_FLASHINFER_SAMPLER=0` | :818 (was :814) | :267-269 (was :247-249) | :30 |
| 12 | ASYNC_SCHED→ASYNC_ARGS | :315 + :653-660 | :292 (hardcoded; was :272) | :92 + :121-122 (was :83 + :103-104) |
| 13 | key resolution | :820-821 (shared resolver; was :816-817) | :278-280 (shared resolver; was :258-260) | :32-41 (shared resolver since #219; was :32, own `cat` flavor) |
| 14 | exec `vllm serve` skeleton | :823-842 (was :819-838) | :282-299 (was :262-279) | :140-160 (was :122-142) |
| 15 | PREFIX_RETENTION / `--prefix-cache-retention-interval` (#174, vllm#55760, #189, #232) | :524-570 (measured 13056/14592, else `None`) | **absent — correct: no drafter** | :100-117 (was :91-99, `None` only; since #232 `0` without a KV tier, `None` with one) |

Spot checks (verified by `sed` at d5e2a01): the PREFIX_CACHE arm's first two
lines are byte-identical — `batch:151-152` (was :131-132) == `single:519-520`
(single then extends the arm at `:521-523`); the ASYNC_ARGS pair
`single:659-660` ≈ `alternative.sh:121-122` (was :103-104) is near-verbatim,
never byte-identical (line 2 has always differed by single's `:-1` fallback);
blocks 1, 2, 7, 9 are near-verbatim between the two `start_qwen.sh` (comment
drift noted in 1.3). Block 10 no longer is: since #235 batch tests a hoisted
`$WSL` (`batch:230`) where single still inlines the `grep` (`single:777`), and
since #219 single's WSL arm carries a pin-memory default batch lacks.

`select_model.sh` (7 lines) is already the shared model selector — three
consumers (`single-user/start_qwen.sh:103`, `docker/entrypoint.sh:28` (was
:25; #219 added three echo lines above it), `bench/warmup.sh:41`) plus a CI
replay (`bench/test_model_verification.py:76-79`).
It is the proof that the extract-and-source pattern works in this repo.
### 1.3 Confirmed drift between the copies

Each item verified against the files on 2026-09-26, re-verified at d5e2a01
(2026-09-29; line numbers below are d5e2a01's, old ones named where they
moved); this is the cost the duplication is already charging.

1. **The INT8 export guards disagree** (#20's "export only when non-empty"
   fix). `batch/start_qwen.sh:275-276` (was :255-256) exports `VLLM_MARLIN_INT8_INCLUDE_RE`
   whenever `INT8_LAYERS` is non-empty — even with `INT8_ACT` empty, which
   leaves the engine with an include-regex and no input dtype.
   `single-user/start_qwen.sh:141-142` and `alternative.sh:55-56` (was :46-47)
   require both. Batch is the outlier; the bug is latent (its `INT8_ACT`
   defaults to `int8` at `:135`, was :115) and fires on
   `INT8_ACT= INT8_LAYERS=mlp bash batch/start_qwen.sh`. Unchanged by the
   seven commits since 2522ef9.
2. **Batch carries spec-decode metrics flags it can never produce.**
   `batch/start_qwen.sh:176-190` (was :156-170) sets `--per-request-spec-decode-metrics` (`:189`);
   batch mode enables no speculative decoding. The block was copied from
   single (`:683-701`), where the flag is real — and the comment headers have
   already drifted apart (single's `#51`/llama-swap essay at `:683-689` vs
   batch's two-line summary at `:176-177`, was :156-157).
3. **Batch's VISION_OFFLOAD comment cites a mode it never runs.**
   `batch/start_qwen.sh:208-216` (was :188-196) explains the default with "on 24 GB
   SPEC=dflash2 + VISION=1 does not boot without it … measured here …
   SPEC=dflash2, RTX 3090" — SPEC=dflash2 and the KV_MEM margin it references
   exist only in single (`:719-727`, the original). A batch reader is sent to
   concepts their launcher does not have.
4. **The `HOST` knob exists only in single.** `single-user/start_qwen.sh:825`
   (was :821) has `--host ${HOST:-0.0.0.0}`; `batch/start_qwen.sh:284` (was
   :264) hardcodes `--host 0.0.0.0` (so does `alternative.sh:142`). Same flag
   position, one knob lost in the copy.
5. **Async scheduling is three shapes.** Single computes it
   (`ASYNC_SCHED` at `:315` for the long DFlash2 verify block, gated on
   `VLLM_DFLASH2_LOOKUP=1 && DRAFT_TOKENS>7` at `:310` → `ASYNC_ARGS` array at
   `:659-660`); `alternative.sh:92,121-122` (was :83,103-104) copies it
   **near-verbatim, not verbatim** as this item said at 2522ef9: `:92` tests
   `DRAFT_TOKENS -gt 7` alone, so `LOOKUP=0 DFLASH_TOKENS=15` turns async off
   there and not in single (the difference predates 2522ef9);
   `batch/start_qwen.sh:292` (was :272) hardcodes `--async-scheduling`.
6. **The CUDA-graph capture-size formula has three homes.**
   `single-user/start_qwen.sh:444` (`${CG:-...}`-guarded),
   `alternative.sh:90` (same arithmetic, unguarded; was :81),
   `batch/start_qwen.sh:294` (hardcoded `64` in the exec line; was :274).
7. **`alternative.sh` never received two platform fixes** that upstream landed
   in both `start_qwen.sh` files: the CUDA_HOME/nvcc-13 fix (`fa97789`, #185 —
   touched `batch/start_qwen.sh` + `single-user/start_qwen.sh` only) and the
   TP>1 allocator default (`d2a5538`, #176 — same two launchers; `alternative.sh`
   has the WSL2 and KV-connector arms at `:13-28` but not the TP arm). Still
   true at d5e2a01: `grep -c 'CUDA_HOME\|tensor-parallel' single-user/alternative.sh` → 0.
8. **Fixed upstream by 8cf642e (#219, merged 2026-09-28).** At 2522ef9
   `alternative.sh` resolved the API key its own way —
   `export VLLM_API_KEY="$(cat api_key.txt)"` (`:32`), no env precedence, no
   file check, and under its own `set -e` (`:11`) a missing `api_key.txt`
   killed the script with a bare `cat` error. It now sets `REPO` from its own
   path (`:35-37`), sources `resolve_api_key.sh` and calls `resolve_vllm_key`
   (`:39-41`) — the same precedence as the launchers. One residual skew the
   fix introduced: `alternative.sh` refuses to boot if the source fails
   (`:39-40`), while both `start_qwen.sh` source it unguarded
   (`single:820`, `batch:279`) and run without `set -e`, so a missing
   `resolve_api_key.sh` there would print "command not found" and boot
   keyless [INFERENCE: read, not run].
9. **A stale patch filename in a comment.** `single-user/start_qwen.sh:221` (unchanged at d5e2a01)
   cites `kvarn-v2-runner-0.28.0.patch`; the tree now carries
   `kvarn/kvarn-v2-runner-0.30.0.patch`. The stale name has survived two pin
   flips (#148 renamed it for 0.29, #189 for 0.30) without an update.
10. **The model decision lives in five places.** `select_model.sh:4-7` (the
    shared one), `batch/start_qwen.sh:75` (inline, base-only),
    `alternative.sh:67-68` (was :58-59; inline, **relative** paths — breaks
    outside the repo root, and never prefers `-fast`; #219 gave the script a
    `REPO` at `:37` but `MODEL`/`DRAFT` and `PATH` (`:31`, `$PWD/venv/bin`)
    still do not use it), `resolve_config.sh:82` (print default),
    `verify.sh:40` (check default; was :21 — #219's `verify.sh` hunk `@@ -6,14 +6,33 @@` added 19 lines above it).
11. **The comments are forking.** Batch's WSL2 arm now says "see the long note
    in single-user/start_qwen.sh" (`batch:228-229`, was :208-209) — the code is duplicated
    but the explanation already migrated. The repo is doing this refactor by
    hand, one comment at a time.

12. **The two retention blocks are capability-skewed** — #189 wrote both in
    one commit, and #232 (36936ec) widened the skew in the other direction.
    Single's (`:524-570`) measures the interval (`13056` at 7 drafts, `14592`
    at 15 — `:558`), warns on unmeasured draft counts (`:560-563`, the #174
    eviction essay), honors a three-rung override ladder (the EXTRA_ARGS flag
    → `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` → `PREFIX_RETENTION`, else `None`
    — `:566-567`) under `PREFIX_CACHE=1` on any `SPEC_CFG` (`:552`), and
    unsets the env spelling afterward (`:569`). `alternative.sh:100-117` (was
    :91-99) has no measured values, no env-var rung, no warning, and fires only
    on `SPEC=dflash2` (`:106`) — but since #232 it has a default single lacks:
    `0` when no KV tier is configured, `None` when `EXTRA_ARGS` carries
    `--kv-offloading-size`/`--kv-transfer-config` (`:110-114`; the #232 PR
    body gives the measurement: two ~60K conversations, dense 0 / 0 cached, `0`
    93.5% / 93.5%). So with no override, single's ladder falls back to `None`
    and `alternative.sh` emits `0`: the two copies now disagree on the
    default, not only on capabilities. (At 2522ef9 this item said
    "`None` unless `PREFIX_RETENTION` is set".)
13. **New since 2522ef9: the WSL2 pin-memory default is three shapes.** #219
    added to single's WSL arm a default that only fires for `SPEC=dflash2` and
    only when `VLLM_WSL2_ENABLE_PIN_MEMORY` is unset, with an echo
    (`single:779-782`). `alternative.sh:15` (unchanged, predates 2522ef9)
    exports `VLLM_WSL2_ENABLE_PIN_MEMORY=1` unconditionally under WSL, so an
    explicit `VLLM_WSL2_ENABLE_PIN_MEMORY=0` is overwritten there; batch has
    neither (it runs no drafter). In the same window #235 hoisted batch's WSL
    detection into `WSL` (`batch:95-96`) for new WSL-only `GPU_UTIL`/`MAX_LEN`
    defaults (`:98-107`); single and `alternative.sh` keep the inline `grep`
    (`single:777`, `alternative.sh:13`), and `alternative.sh:82`'s comment
    ("0.93 on WSL2") names a WSL value its code never picks. Three detections,
    four WSL-dependent decisions, three shapes.

### 1.4 The validation layer has drifted from what it validates

`resolve_config.sh` (the F13 fix) watches the launchers from the outside:

- **Its EXTRA_ARGS shadow list (`resolve_config.sh:68`) names 11 flags.** The
  three launchers' exec lines emit 17 literal flags (union of `--flag` tokens
  from each `exec` line to end of file; unchanged at d5e2a01), plus **16 more**
  (was "~13") assembled outside the exec lines into
  `$KV_ARGS`/`$ATTN_ARGS`/`$SPEC_ARGS`/`$VISION_ARGS`/`$METRICS_ARGS`/`$ASYNC_ARGS`/`$PREFIX_ARGS`/`$EXTRA_ARGS`
  (method: every `--flag` in non-comment, non-`echo` lines of the three files,
  minus the exec-line flags, minus the six that are only *matched* in
  EXTRA_ARGS or probed — `--tensor-parallel-size`, `--disable-custom-all-reduce`,
  `--enforce-eager`, `--kv-offloading-size`, `--kv-transfer-config`,
  `--dtype` — and `nvcc --version`). Of the 33, the list covers 11; 13 of
  the 16 non-exec flags and 9 of the 17 exec flags are unwatched. Measured gaps
  include `--compilation-config` (both `start_qwen.sh` set it),
  `--async-scheduling`, `--max-num-batched-tokens`, `--reasoning-parser`,
  `--speculative-config`, `--enable-prefix-caching`, `--mamba-cache-mode`,
  `--kv-cache-memory`, `--prefix-cache-retention-interval`,
  `--sse-keep-alive-interval`, `--default-chat-template-kwargs`. Any of those
  in EXTRA_ARGS silently shadows the launcher — the exact failure class the
  resolver was built to make visible.
- **Its MODEL print is wrong on the native single path.** Single calls
  `resolve_effective_config single` at `:101` but selects the model at `:103`;
  `resolve_config.sh:82` defaults to the base dir, so `[effective-config]
  MODEL=` prints the base model even when the `-fast` variant will be served.
  Docker masks this (`entrypoint.sh:26-30` exports MODEL first; was :23-27);
  a native boot prints the wrong checkpoint. (Verified by call order, still
  :101 → :103 at d5e2a01.)
- **`alternative.sh` is entirely outside it** — no validation of its SPEC
  (its own `case` at `:74-80`, was :65-71, refuses, duplicating the resolver's
  job). #219 made it source `resolve_api_key.sh`, not `resolve_config.sh`.

### 1.5 Change pressure

`single-user/` + `batch/` took 24 file-touches in the last 40 upstream commits
(@d5e2a01; was 21 @2522ef9), 16 of them to the three launcher scripts
(`git log -40 --name-only --format= upstream/main | grep -cE
'^(single-user|batch)/'`, and `…/(start_qwen|alternative)\.sh`). That is
third, after `patches/` (105) and `docs/` (47) — the 2522ef9 note's "second
only to `patches/` at 103" was already wrong then (`docs/` had 38). The
launcher-affecting ones
repeatedly land in the shared blocks: #185 and #176 each had to edit both
`start_qwen.sh` files (verified: file lists of `fa97789`, `d2a5538`), #126's
model-selection fix touched six files for one behavior (`25bd8d2`: select_model,
entrypoint, single launcher, the CI test + its workflow, docs) — and since the
original measurement #207 touched both launchers + verify.sh, #189 all three
at once, #214 single's KV_MEM append alone; since 2522ef9, #219 touched
single + `alternative.sh` (+ entrypoint, verify.sh), #232 `alternative.sh`,
#235 batch — three commits, three launchers, each landing in a chassis block
(10, 13, 15, and batch's block 10 again).
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
| `qwen_detect_wsl` + `qwen_allocator_defaults` | block 10 | Split in two since #235 (re-verification 2026-09-29): batch now needs the WSL answer *before* its KV profile branch (`batch:95-107`, the WSL `GPU_UTIL`/`MAX_LEN` defaults) and again at the allocator (`:230`), so detection is its own early call that sets one variable, and the allocator function reads it. `qwen_allocator_defaults`: KV-connector + TP arms → sets `PYTORCH_CUDA_ALLOC_CONF`, plus the WSL pin-memory default in single's guarded form (`single:779-782`, #219) behind a has-drafter parameter (1.3.13). The long WSL2 essay (single:767-776) lives here; batch's "see the long note" pointer (1.3.11) dissolves. The connector match (`--kv-offloading-size`/`--kv-transfer-config`) is written twice in `alternative.sh` since #232 (`:23-24`, `:110-111`) — the function exports its answer so block 15 reuses it. |
| `qwen_int8_exports` | block 5 | the two-condition guard (1.3.1's fix) once; callers keep their own `INT8_ACT`/`INT8_LAYERS` *defaults*, which are genuinely per-mode (batch: `int8`/`mlp`, `:135-136`; single and alternative: off/`mlp\|linear_attn\|self_attn`, `single:139-140`, `alternative.sh:53-54` — this row said "off/`all`" before 2026-09-29; the literal is the three-class regex). |
| `qwen_tool_args` / `qwen_metrics_args` / `qwen_vision_args` | blocks 7-9 | the array builders with their #59 comments. The spec-decode metrics flag moves behind a `mode` parameter so batch stops carrying flags it cannot produce (1.3.2) — or simpler: the flag is harmless-but-dead in batch; keep it only if the owner wants one code path. Called out in the PR, not decided here. |
| `qwen_async_args` | block 12 | `ASYNC_SCHED` → `ASYNC_ARGS`; single's `:315` setter stays in single (it is dflash2-shaped), and so does `alternative.sh`'s `:92` setter — they differ (1.3.5), and unifying them is a behavior change for PR B to call out, not a move. The array build shares. |
| `qwen_retention_args` | block 15 | single's measured ladder (13056/14592), the override order, and the #174 warning move verbatim, with the **final fallback as a caller parameter** (single: `None`; `alternative.sh`: #232's `0` without a KV tier, `None` with one). Until #232 this row said `alternative.sh`'s `None` path "is the ladder's own fallback, argv-identical unless `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` is exported"; since 36936ec that is false — a fixed `None` fallback would regress #232's measured no-tier default (1.3.12). PR B deletes the copy for the call. Batch stays absent — no drafter. |
| `qwen_serve_argv` | block 14 | the exec line as an **array builder**: fills `ARGV=(venv/bin/vllm serve …)` from the caller's mode parameters. EXTRA_ARGS still expands last and still word-splits (the documented override door — resolve_config.sh:34-35); the array conversion makes the other expansions (`$VISION_ARGS`, `$KV_ARGS`, `$ATTN_ARGS`) explicit instead of relying on unquoted splitting. |
| key resolution | block 13 | stays in `resolve_api_key.sh` (already shared). **Done upstream for `alternative.sh`** by 8cf642e (#219) — it sources the resolver at `:39-41` (1.3.8); nothing left for the chassis but, optionally, giving both `start_qwen.sh` the refuse-on-failed-source guard `alternative.sh` has. |

**Deliberately not shared:** the header essays (they are mode-specific
measurement reports — single's `:24-53` context tiers, batch's `:5-24` state
economics); the profile branches (CTX×SPEC and the whole DFlash2/MTP/SPEC_CFG/
KV_MEM/residency machinery in single:191-596; KV in batch:79-132, was :77-110, now including #235's WSL defaults :87-107); the PREFIX_
CACHE extensions beyond block 15 that single carries (the CTX=huge match-unit
line `:521-523`; the capture-mode machinery `:571-644`; batch's arm is the
base form). Sharing those would move complexity, not concentrate it — they
fail the deletion test. The retention ladder sat in this list until #189
made it a second, skewed copy; it is chassis block 15 now (1.3.12).

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

Upstream already needed this seam once: #232 verified its retention change
with "a stub `vllm` on the launcher's `PATH` that prints its argv" (PR body,
7 cases). That trick works only because `alternative.sh` execs a bare
`vllm serve` (`:140`); both `start_qwen.sh` exec the literal path
`venv/bin/vllm` (`single:823`, `batch:282`), so a PATH stub cannot see
them — `PRINT_ARGV=1` is what makes the same check possible for the two
launchers that matter most.

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
  refuse shape, deleting its duplicate `case` at `:74-80` (was :65-71).

### 2.4 The drift fixes that fall out (not the goal — the proof)

1.3's items resolve as follows: 1 (INT8 guards) → one shared guard. 2
(spec-decode flag in batch) → mode-parameterized builder, or a documented
harmless keep. 3 (batch's dflash2 VISION comment) → the comment lives once,
in the chassis, citing the mode it belongs to. 4 (HOST knob) → both launchers
get `--host ${HOST:-0.0.0.0}` from the shared argv builder (a behavior change
for batch; flagged to the owner — smallest possible one). 5, 6 (async, CG) →
one builder each. 7 (alternative missing #185/#176) → inclusion. 8 (key
flavor) → **done upstream** (8cf642e, #219). 9 (stale kvarn name) → fixed in passing, one
comment. 10 (five model defaults) → `select_model.sh` stays the one
selector; `alternative.sh:67-68` and `resolve_config.sh:82` defer to it;
`verify.sh:40` stays its own (it checks, it does not select).
11 (forking comments) → comments live in the chassis functions.
12 (skewed retention copies) → `qwen_retention_args` with a per-caller
fallback; batch stays absent by design. 13 (pin-memory shapes, WSL
detection) → `qwen_detect_wsl` once and the guarded pin-memory default in
`qwen_allocator_defaults`; `alternative.sh` stops clobbering an explicit
`VLLM_WSL2_ENABLE_PIN_MEMORY=0` (a behavior change, PR B).
## 3. Rollout

Three PRs, ordered so the behavior-preserving one proves itself before any
behavior changes land. Each is independently revertable.

### 3.1 PR A — the chassis + the dry-run seam (no behavior change)

Add `launcher_common.sh`; move blocks 1-3, 5, 7-12, and single's 15 (batch has none) of the two `start_qwen.sh`
into it **verbatim** (code and comments); both launchers source it; both
exec lines become `qwen_serve_argv` + the `PRINT_ARGV=1` gate. `alternative.sh`
is untouched in this PR. The `dflash2-backport`-style temptation to clean up
comments while moving them is resisted: the diff must read as pure moves.
Since #235 (re-verified 2026-09-29), batch calls `qwen_detect_wsl` where its
`WSL=` line sits today (`batch:95-96`, before the KV branch), not at block 10;
single calls it at `:777`, and single's #219 pin-memory arm (`:779-782`)
moves with block 10 unchanged — so PR A stays argv- and env-identical.

Acceptance: `bash -n` clean on all three shells' targets; **byte-identical
argv** — for each profile in the test matrix (section 4), `PRINT_ARGV=1` on
main vs the branch must diff empty; `patch-integrity`'s existing
`test_model_verification.py` still passes; the image build is green (it runs
`verify.sh --install`, not the launchers, so it is a smoke check only).

### 3.2 PR B — `alternative.sh` joins the chassis (real fixes, called out)

`alternative.sh` sources the chassis and `resolve_config.sh` (it already
sources `resolve_api_key.sh` since #219); its own copies of
the WSL2/KV-connector arms, the stale-shm sweep, the INT8 guards, the
METRICS/ASYNC arrays, the #189/#232 retention copy, and the SPEC case are deleted. Explicit behavior changes,
each its own commit message line: gains the CUDA_HOME fix (1.3.7a), gains the
TP allocator arm (1.3.7b), absolute model paths (1.3.10 — now a one-liner,
`REPO` exists at `:37`), the guarded CG formula (1.3.6), the guarded WSL
pin-memory default that no longer overwrites an explicit
`VLLM_WSL2_ENABLE_PIN_MEMORY` (1.3.13), and a `PRINT_ARGV` gate of its own.
Key resolution via `resolve_api_key.sh` (1.3.8) was on this list until
8cf642e (#219) landed it upstream; it is dropped from PR B. The retention copy
being deleted is upstream's own (#189 landed it, #232 changed its default),
replaced by the `qwen_retention_args` call (1.3.12) with `alternative.sh`'s
tier-aware fallback (`0` / `None`) passed in — argv-identical to #232's
behavior unless `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` is exported (the rung
the copy never had). #232's stub-`vllm` argv table (7 cases, in the PR body)
becomes the PR B acceptance rows for retention. Also fixes 1.3.9's stale
comment.

Acceptance: the dry-run argv for its two profiles matches the pre-PR argv
*except* in the enumerated fixes; #232's seven retention cases give the same
flag as the #232 table; a boot smoke on the int4 profile.

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
| `CTX=huge SPEC=dflash2 PREFIX_CACHE=1` | `--kv-cache-dtype kvarn_k4v2_g128`, `--block-size 128`, `--prefix-match-unit 128`, `--prefix-cache-retention-interval 13056` (7-draft default, single:558/567) |
| `SPEC=dflash2 DFLASH_TOKENS=15` | `--no-async-scheduling`, CG capped at 64 |
| `KV_MEM=8000000000 SPEC=mtp` | `--kv-cache-memory=8000000000` in argv (the #214 fix: honored in every SPEC mode, not only dflash2 — single:511) |
| `SPEC=off` | no `--speculative-config` at all |
| `EXTRA_ARGS="--tensor-parallel-size 2"` | `PYTORCH_CUDA_ALLOC_CONF` printed as `expandable_segments:False` (the warning text asserts too) |
| `EXTRA_ARGS="--compilation-config …"` | the shadow warning fires (PR C) |
| `CTX=bogus` | refusal, exit 1, nothing printed to argv |
| `INT8_ACT= INT8_LAYERS=mlp bash batch/…` | no `VLLM_MARLIN_INT8_INCLUDE_RE` exported (1.3.1) |
| `VISION=1` | the mm flags, no `--language-model-only` |
| `HOST=127.0.0.1` (batch, PR C) | `--host 127.0.0.1` |
| `WSL_DISTRO_NAME=x KV=kvarn bash batch/…` | `--gpu-memory-utilization 0.88 --max-model-len 131072` (#235, `batch:98-107`); `KV=fp8` → `0.91`, `150000`; an explicit `GPU_UTIL`/`MAX_LEN` wins |
| `WSL_DISTRO_NAME=x SPEC=dflash2` (single) | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` exported (#219, `single:779-782`); with `VLLM_WSL2_ENABLE_PIN_MEMORY=0` exported it stays 0 |
| `alternative.sh`, defaults (PR B) | `--prefix-cache-retention-interval 0`; with `EXTRA_ARGS="--kv-offloading-size 8"` → `None` (#232, `alternative.sh:110-115`) |

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
   → 0 and 0 (9 and 9 at d5e2a01); the string lives once, in `launcher_common.sh`.
2. `grep -c "qwen3_coder" batch/start_qwen.sh single-user/alternative.sh`
   → 0 and 0 (3 and 1 at d5e2a01; the parser name lives once, in the chassis).
3. `PRINT_ARGV=1` produces the full argv from all three launchers, and
   `bench/test_launcher_argv.py` runs green in `patch-integrity.yml`'s CPU job.
4. `resolve_config.sh` contains no hand-maintained flag list.
5. `comm -12 <(sort -u single-user/start_qwen.sh) <(sort -u batch/start_qwen.sh)`
   on unique substantive lines drops from 76 (still 76 at d5e2a01) to near zero — what remains is
   the profile data that genuinely differs.
6. A native `PRINT_ARGV=1` single-mode run prints the `-fast` model when it
   exists (1.4's bug is gone).
