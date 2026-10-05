# Extract the shared launcher chassis — remediation plan

Architecture review 2026-09-26, candidate 2 (Strong). The full report is
[architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card. Every number below was measured against the
tree on 2026-09-26 (method noted where it matters); nothing is carried over
from the review on trust.

> **Partly implemented 2026-10-04 on upstream/main @ 10bb488 (vLLM 0.30.0); checked 2026-10-05.**
> **Status:** In review. PR A is syv-ai/HyperQwen#272 (`9fa248e`). PR B is
> #273 (`6a9922f`), stacked on #272, so #272 merges first. CI is green on
> both (4 of 4). upstream/main has no launcher diff since e371b42 (`git diff
> --stat e371b42 10bb488 -- single-user batch resolve_config.sh
> resolve_api_key.sh` is empty), so the e371b42 line cites below still hold
> on main.
> - **PR A (#272) is smaller than 3.1.** It adds `launcher_common.sh` with two
>   functions. `resolve_bind_host` moves there verbatim from
>   `resolve_api_key.sh`, with its header comment. `qwen_exec` replaces
>   `exec` as each launcher's last call. With `PRINT_ARGV=1` it prints its
>   arguments, one per line, and exits 0. The three launchers source the file
>   and refuse to boot if that fails. `bench/test_no_key_bind.sh`'s 18
>   launcher rows read `--host` from the `PRINT_ARGV=1` argv, and its stub
>   now fails a row that reaches `exec`. Blocks 1-3, 5, 7-12 and 15 stay in
>   the launchers. There is no `qwen_serve_argv` array builder: `qwen_exec`
>   takes the exec words as they are, so they expand as before.
> - **Two changes from the plan in #272.** First, `resolve_bind_host` moves to
>   the chassis. 2.1's key row kept it in `resolve_api_key.sh`, but only the
>   three launchers call it, and the bench clients also source that file.
>   Second, `alternative.sh` gets the `PRINT_ARGV=1` gate in PR A. 3.1 left
>   it untouched and gave it the gate in 3.2.
> - **PR B (#273) is drift item 1, not 3.2.** It adds `qwen_int8_exports
>   <act> <layers>`. That exports `VLLM_MARLIN_INPUT_DTYPE` when `<act>` is
>   non-empty, and `VLLM_MARLIN_INT8_INCLUDE_RE` only with it. The guard had
>   four copies, not three. `bench/prefill_ab.sh:39-40` exported the regex on
>   `INT8_LAYERS` alone, like batch, and before the single launcher's own
>   guard ran. #273 deletes that copy. The function uses `if` blocks, because
>   `alternative.sh` runs under `set -e`, and a final false `&&` list would
>   stop it. Both `start_qwen.sh` now source `launcher_common.sh` right after
>   `resolve_config.sh`. The behavior change: batch and `prefill_ab.sh` with
>   `INT8_ACT` empty and `INT8_LAYERS` set no longer export the regex. The
>   engine ignored it there, because it reads the regex only when the dtype is
>   set. But the regex is in the compile cache key, so that boot probably
>   compiles cold once [INFERENCE: read in the code, not run on a GPU].
> - **Measured (CPU only; the PR bodies have the harness).** #272: 80 stub
>   rows (30 single, 26 batch, 24 alternative). Between main and the branch,
>   argv, environment, output and exit code are byte-identical in all 80.
>   `PRINT_ARGV=1` prints the stub's argv in all 80; 3 of them are refusals.
>   `test_no_key_bind.sh` gives 22 PASS, and the launcher pid is the server
>   pid. #273: 83 rows. 81 are identical, and the 2 batch INT8 drift rows
>   each lose only `VLLM_MARLIN_INT8_INCLUDE_RE=mlp`. `PRINT_ARGV=1` is
>   identical in 83 of 83. `test_no_key_bind.sh` gives 31 PASS (18 host,
>   9 INT8, 4 `verify.sh`). Each new check was broken on purpose in a scratch
>   copy, and it failed.
> - **Found 2026-10-05: #272 makes drift item 8's residual fail open.** Both
>   `start_qwen.sh` still source `resolve_api_key.sh` with no guard
>   (`single:821`, `batch:275` at #273). I moved that file away in copies of
>   the three trees, with the key only in `api_key.txt` and a stub `vllm`.
>   On upstream/main, both launchers exec `--host --port 8000`, and plain
>   argparse refuses that (`argument --host: expected one argument`). On #272
>   and #273, `resolve_bind_host` comes from `launcher_common.sh`, so both
>   launchers boot with no key. The bind is `127.0.0.1` by default, with the
>   `no API key` warning. With `HOST=0.0.0.0` it is `0.0.0.0`, after
>   `WARNING: no API key and HOST=0.0.0.0`. The fix is the
>   refuse-on-failed-source guard that `alternative.sh:39-40` has, in both
>   `start_qwen.sh` (2.1's key row). Neither PR has it yet.
> - **Census at #273** (Python sets, 1.1's definitions; the same script gives
>   e371b42's 78/43 and 250/107/95). `wc -l` is 844/296/163 (843/300/161 on
>   main). single∩batch is 80 (78 on main; +2 for the two `source
>   launcher_common.sh` lines). single∩`alternative.sh` is 42 (43 on main).
> - **Done when (6) at #273.** Items 1, 2, 4 and 6 are unchanged: 9 and 9, 3
>   and 1, the 11-flag list at `resolve_config.sh:68`, and the model print.
>   Item 5 rose from 78 to 80. Item 3 is half done: `PRINT_ARGV=1` prints the
>   full argv from all three launchers, and #271 runs `test_no_key_bind.sh`
>   in CI. `bench/test_launcher_argv.py` does not exist.
> - **Still to do.** The guard above. The shared blocks move into the chassis
>   one at a time (3.1's moves). Then 3.2 (`alternative.sh` joins the
>   chassis, with the listed fixes) and 3.3 (the validation layer and
>   `bench/test_launcher_argv.py` in CI).

> **Re-verified 2026-10-04 against upstream/main @ e371b42 (vLLM 0.30.0).**
> **Status at e371b42:** Not started. Next candidate after #5 (see the tracker in decode-perf-150k-240k-plan.md).
> Thirteen commits landed since d5e2a01. Two touch this plan's files
> (`git diff --stat d5e2a01 e371b42 -- single-user batch resolve_config.sh resolve_api_key.sh verify.sh`):
> 177ce26 (#237) touches all three launchers, `resolve_api_key.sh` and
> `verify.sh`; e1459c7 (#242/#243/#244) touches `verify.sh` only. Each launcher
> grew by one line: 842/299/160 → **843/300/161** (`wc -l`). The body's line
> cites are re-mapped to e371b42: d5e2a01 lines from single `:822`, batch
> `:281` and `alternative.sh` `:42` on move down by one; earlier lines hold.
> - **Fixed upstream:** drift item 4. 177ce26 (#237) added `resolve_bind_host`
>   to the shared `resolve_api_key.sh` (`:53-70`). All three launchers call it
>   after `resolve_vllm_key` (`single:822`, `batch:281`, `alternative.sh:42`)
>   and pass `--host $BIND_HOST` (`single:826`, `batch:285`,
>   `alternative.sh:143`). An explicit `HOST` wins in all three. With no key and
>   no `HOST`, the bind is `127.0.0.1`; inside a container (`/.dockerenv`) it
>   stays `0.0.0.0`. #237 put the logic in a shared file and left one call per
>   launcher — the extract-and-source pattern this plan uses.
> - **Changed:** drift item 8's residual. Both `start_qwen.sh` still source
>   `resolve_api_key.sh` unguarded (`single:820`, `batch:279`). A failed source
>   now also leaves `BIND_HOST` empty, so the unquoted `--host $BIND_HOST`
>   gives `vllm` the argv `--host --port N`. argparse most likely refuses that:
>   a failed boot, not a keyless one [INFERENCE: not run].
> - **Changed:** the shared-line counts rose. single∩batch unique substantive
>   lines 76 → **78** (78/123 = 63%); single∩`alternative.sh` 41 → **43**
>   (43/99 = 43%); whole files 184 → 186 (158 → 160 unique). The +2 are the two
>   lines #237 added to both `start_qwen.sh`: `resolve_bind_host` and
>   `--host $BIND_HOST --port $PORT \`. Substantive lines 294/123/99, unique
>   250/107/95.
> - **New since d5e2a01:** `bench/test_no_key_bind.sh` (177ce26, 40 lines) is a
>   third hand-built argv capture, after #189's extracted block and #232's
>   PATH stub. It copies the checkout to a temp dir, puts a stub `vllm` at
>   `venv/bin/vllm` (`:11-14`), runs all three launchers and reads `--host`
>   from the printed argv: six rows per launcher (`:25-30`) and four `verify.sh`
>   rows (`:36-39`). No CI runs it: `patch-integrity.yml` runs
>   `test_model_verification.py`, `test_prepare_state.py`, the KVarN torch test
>   and `check_vllm_series.sh`, and `grep -rn test_no_key_bind .github Makefile`
>   finds nothing. It passes at e371b42: 22 PASS, exit 0, in 0.4 s (run for
>   the 2026-10-04 re-check of harness-verdicts-ci-remediation.md). #237 also
>   touched all three launchers in one commit, as #189 did.
> - **Unchanged:** drift items 1-3, 5-7 and 9-13 and the 1.4 findings stand at
>   the re-cited lines. `resolve_config.sh`, both systemd units,
>   `select_model.sh`, `docker/entrypoint.sh` and `bench/warmup.sh` have no
>   diff since d5e2a01 (`git diff --stat d5e2a01 e371b42 -- <those files>` is
>   empty). The shadow list still names 11 flags (`resolve_config.sh:68`); the
>   call order is still `:101` → `:103`. The exec lines still share 12 of 13
>   flags (17 in the three-file union). Conditionals stay 48/18/10. Upstream
>   has no `launcher_common.sh`, no `PRINT_ARGV` and no
>   `bench/test_launcher_argv.py`; all six Done-when checks (6) still fail.
>   Open PRs #260 (`patches/`, `PATCHES.md`) and #267 (`verify.sh`) touch no
>   launcher (`gh pr view <n> --json files`).
> - **Change pressure (1.5), same command at e371b42:** 16 `single-user/` +
>   `batch/` file-touches in the last 40 commits, 11 of them to the three
>   launchers (`patches/` 64, `docs/` 48). The 40-commit window moved, so these
>   do not compare one-to-one with the d5e2a01 figures.
> - **Census method and a tool caveat:** the counts use 1.1's definitions
>   (substantive = `^[[:space:]]*[^#[:space:]]`; shared = the intersection of
>   unique substantive lines). On this host `sort` and `comm` are uutils
>   coreutils 0.2.2. The same `comm -12 <(sort -u …) <(sort -u …)` gave 94
>   and then 14 on back-to-back runs of the same files, with "not in sorted
>   order" warnings, under `LC_ALL=C`. The numbers above come from a Python set
>   intersection of the same lines, which reproduces d5e2a01's 76/41/184/158
>   exactly. Re-measure with GNU coreutils or Python.
> - **Corrected:** 2.2 said a `vllm` stub can capture argv only from
>   `alternative.sh`. `test_no_key_bind.sh` captures it from all three by
>   planting the stub in a copied tree. A stub on `PATH` still cannot see
>   `venv/bin/vllm`. 4's `HOST=127.0.0.1 (batch, PR C)` row is now upstream
>   behavior with an upstream test; 2.4's item 4 fix is done.

> **Previous note — re-verified 2026-09-29 against upstream/main @ d5e2a01
> (vLLM 0.30.0).** Kept as history; its line numbers are those of d5e2a01.
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
| `single-user/start_qwen.sh` | 843 (842 at d5e2a01) | 294 (293) | — | 48 (48) |
| `batch/start_qwen.sh` | 300 (299) | 123 (122) | **78 (63%; was 76, 62%)** | 18 (18) |
| `single-user/alternative.sh` | 161 (160) | 99 (98) | **43 (43%; was 41, 42%)** | 10 (10) |

¹ Substantive = non-comment, non-blank (`grep -cE '^[[:space:]]*[^#[:space:]]'`);
the shared column is `comm -12` on the `sort -u` of those lines (unique
substantive lines: 250 / 107 / 95 at e371b42, 249 / 106 / 94 at d5e2a01), and the
percentage divides it by the file's *non-unique* substantive count (78/123,
43/99) — the method that reproduces the 2522ef9 figures (76/110 = 69%,
40/87 = 46%). With duplicates and comments included (`comm -12` on plain
`sort` of whole files), single∩batch share **186 lines** (184 at d5e2a01; 160
unique, was 158 — `comm -12` on `sort -u`). (2026-10-04) uutils `sort`/`comm`
on the measuring host gave unstable results; the e371b42 figures come from a
Python set intersection of the same lines (see the 2026-10-04 note).
² `grep -cE '^\s*(if|case|elif) '` — the env/platform decision points. Batch's
+5 are #235's WSL `GPU_UTIL`/`MAX_LEN` defaults (`batch:96-104`); single's +1
is #219's pin-memory `if` (`:779`); `alternative.sh`'s +2 are #232's tier
`case` and `PREFIX_RETENTION` `if` (`:111`, `:115`). #237 added none.

The exec lines share **12 of 13** literal flags (re-measured at e371b42,
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
| 2 | stale-`/dev/shm` offload sweep (#33) | :63-74 | :34-45 | :59-66 (3rd copy; :58-65 at d5e2a01) |
| 3 | CUDA_HOME/nvcc-13 fix (#185) | :77-93 | :48-64 | **absent** |
| 4 | resolve_config boilerplate | :95-101 | :66-73 | **absent** |
| 5 | INT8 export guards (#20) | :139-142 | :133-136 + :270-276 (was :113-116 + :250-256) | :54-57 (:53-56 at d5e2a01) |
| 6 | PREFIX_CACHE arm | :515-523 (+:524-644 single-only extension) | :138-153 (was :118-133) | :94,99-100 (:93,98-99 at d5e2a01) |
| 7 | TOOL_ARGS (#59, qwen3_coder) | :662-681 | :155-174 (was :135-154) | :158 (hardcoded; :157 at d5e2a01) |
| 8 | METRICS_ARGS (#51, #59) | :683-701 | :176-190 (was :156-170) | :125-139 (3rd copy; :124-138 at d5e2a01) |
| 9 | VISION block (gotcha 9) | :703-736 | :192-225 (was :172-205) | :86,96-97 (simplified; :85,95-96 at d5e2a01) |
| 10 | WSL2/allocator + KV-connector + TP blocks (#2/#26, #95, #163) | :766-817 (was :766-813; +#219's pin-memory arm :779-782) | :227-266 (was :207-246) + WSL detect hoisted to :95-96 by #235 | :13-28 (**TP block absent**; unconditional pin-memory export :15) |
| 11 | `VLLM_USE_FLASHINFER_SAMPLER=0` | :818 (was :814) | :267-269 (was :247-249) | :30 |
| 12 | ASYNC_SCHED→ASYNC_ARGS | :315 + :653-660 | :293 (hardcoded; :292 at d5e2a01) | :93 + :122-123 (:92 + :121-122 at d5e2a01) |
| 13 | key resolution + bind host (#237) | :820-822 (shared resolver; `resolve_bind_host` at :822 since #237) | :278-281 (shared resolver; `resolve_bind_host` at :281) | :32-42 (shared resolver since #219; `resolve_bind_host` at :42) |
| 14 | exec `vllm serve` skeleton | :824-843 (:823-842 at d5e2a01) | :283-300 (:282-299 at d5e2a01) | :141-161 (:140-160 at d5e2a01) |
| 15 | PREFIX_RETENTION / `--prefix-cache-retention-interval` (#174, vllm#55760, #189, #232) | :524-570 (measured 13056/14592, else `None`) | **absent — correct: no drafter** | :101-118 (:100-117 at d5e2a01; `None` only before #232; since #232 `0` without a KV tier, `None` with one) |

Spot checks (verified by `sed` at e371b42): the PREFIX_CACHE arm's first two
lines are byte-identical — `batch:151-152` (was :131-132) == `single:519-520`
(single then extends the arm at `:521-523`); the ASYNC_ARGS pair
`single:659-660` ≈ `alternative.sh:122-123` (:121-122 at d5e2a01) is near-verbatim,
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
(2026-09-29) and at e371b42 (2026-10-04; line numbers below are e371b42's,
d5e2a01's named where they moved); this is the cost the duplication is
already charging.

1. **The INT8 export guards disagree** (#20's "export only when non-empty"
   fix). `batch/start_qwen.sh:275-276` (was :255-256) exports `VLLM_MARLIN_INT8_INCLUDE_RE`
   whenever `INT8_LAYERS` is non-empty — even with `INT8_ACT` empty, which
   leaves the engine with an include-regex and no input dtype.
   `single-user/start_qwen.sh:141-142` and `alternative.sh:56-57` (:55-56 at
   d5e2a01) require both. Batch is the outlier; the bug is latent (its `INT8_ACT`
   defaults to `int8` at `:135`, was :115) and fires on
   `INT8_ACT= INT8_LAYERS=mlp bash batch/start_qwen.sh`. Unchanged by the
   seven commits since 2522ef9 and the thirteen since d5e2a01.
   (2026-10-05) #273 (in review) replaces all copies with one shared guard.
   It found a fourth copy: `bench/prefill_ab.sh:39-40` behaves like batch.
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
4. **Fixed upstream by 177ce26 (#237).** At d5e2a01 the `HOST` knob existed
   only in single: `single-user/start_qwen.sh:825` had
   `--host ${HOST:-0.0.0.0}`; `batch/start_qwen.sh:284` and
   `alternative.sh:142` hardcoded `--host 0.0.0.0`. Now all three pass
   `--host $BIND_HOST` (`single:826`, `batch:285`, `alternative.sh:143`), set
   by the shared `resolve_bind_host` (`resolve_api_key.sh:53-70`) that each
   launcher calls after `resolve_vllm_key`. An explicit `HOST` wins in all
   three; with no key the default is `127.0.0.1`. The fix went into the
   shared file, not three copies.
5. **Async scheduling is three shapes.** Single computes it
   (`ASYNC_SCHED` at `:315` for the long DFlash2 verify block, gated on
   `VLLM_DFLASH2_LOOKUP=1 && DRAFT_TOKENS>7` at `:310` → `ASYNC_ARGS` array at
   `:659-660`); `alternative.sh:93,122-123` (:92,121-122 at d5e2a01) copies it
   **near-verbatim, not verbatim** as this item said at 2522ef9: `:93` tests
   `DRAFT_TOKENS -gt 7` alone, so `LOOKUP=0 DFLASH_TOKENS=15` turns async off
   there and not in single (the difference predates 2522ef9);
   `batch/start_qwen.sh:293` (:292 at d5e2a01) hardcodes `--async-scheduling`.
6. **The CUDA-graph capture-size formula has three homes.**
   `single-user/start_qwen.sh:444` (`${CG:-...}`-guarded),
   `alternative.sh:91` (same arithmetic, unguarded; :90 at d5e2a01),
   `batch/start_qwen.sh:295` (hardcoded `64` in the exec line; :294 at d5e2a01).
7. **`alternative.sh` never received two platform fixes** that upstream landed
   in both `start_qwen.sh` files: the CUDA_HOME/nvcc-13 fix (`fa97789`, #185 —
   touched `batch/start_qwen.sh` + `single-user/start_qwen.sh` only) and the
   TP>1 allocator default (`d2a5538`, #176 — same two launchers; `alternative.sh`
   has the WSL2 and KV-connector arms at `:13-28` but not the TP arm). Still
   true at e371b42: `grep -c 'CUDA_HOME\|tensor-parallel' single-user/alternative.sh` → 0.
8. **Fixed upstream by 8cf642e (#219, merged 2026-09-28).** At 2522ef9
   `alternative.sh` resolved the API key its own way —
   `export VLLM_API_KEY="$(cat api_key.txt)"` (`:32`), no env precedence, no
   file check, and under its own `set -e` (`:11`) a missing `api_key.txt`
   killed the script with a bare `cat` error. It now sets `REPO` from its own
   path (`:35-37`), sources `resolve_api_key.sh` and calls `resolve_vllm_key`
   (`:39-41`) — the same precedence as the launchers. One residual skew the
   fix introduced: `alternative.sh` refuses to boot if the source fails
   (`:39-40`), while both `start_qwen.sh` source it unguarded
   (`single:820`, `batch:279`) and run without `set -e`. At d5e2a01 a missing
   `resolve_api_key.sh` there would print "command not found" and boot
   keyless [INFERENCE: read, not run]. (2026-10-04) Since #237 the same
   failure also skips `resolve_bind_host` (`single:822`, `batch:281`), so
   `BIND_HOST` is empty and the unquoted `--host $BIND_HOST` gives `vllm`
   the argv `--host --port N`. argparse most likely refuses that: a failed
   boot, not a keyless one [INFERENCE: not run].
   (2026-10-05) Measured with a stub `vllm`: main execs `--host --port 8000`,
   and plain argparse refuses it. #272 moves `resolve_bind_host` into
   `launcher_common.sh`, so on #272 and #273 the same failure boots keyless
   again, on `127.0.0.1` or on the `HOST` given. See the block at the top.
9. **A stale patch filename in a comment.** `single-user/start_qwen.sh:221` (unchanged at e371b42)
   cites `kvarn-v2-runner-0.28.0.patch`; the tree now carries
   `kvarn/kvarn-v2-runner-0.30.0.patch`. The stale name has survived two pin
   flips (#148 renamed it for 0.29, #189 for 0.30) without an update.
10. **The model decision lives in five places.** `select_model.sh:4-7` (the
    shared one), `batch/start_qwen.sh:75` (inline, base-only),
    `alternative.sh:68-69` (:67-68 at d5e2a01; inline, **relative** paths — breaks
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
    unsets the env spelling afterward (`:569`). `alternative.sh:101-118`
    (:100-117 at d5e2a01) has no measured values, no env-var rung, no warning, and fires only
    on `SPEC=dflash2` (`:107`) — but since #232 it has a default single lacks:
    `0` when no KV tier is configured, `None` when `EXTRA_ARGS` carries
    `--kv-offloading-size`/`--kv-transfer-config` (`:111-115`; the #232 PR
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
    (`single:777`, `alternative.sh:13`), and `alternative.sh:83`'s comment
    ("0.93 on WSL2") names a WSL value its code never picks. Three detections,
    four WSL-dependent decisions, three shapes.

### 1.4 The validation layer has drifted from what it validates

`resolve_config.sh` (the F13 fix) watches the launchers from the outside:

- **Its EXTRA_ARGS shadow list (`resolve_config.sh:68`) names 11 flags.** The
  three launchers' exec lines emit 17 literal flags (union of `--flag` tokens
  from each `exec` line to end of file; unchanged at e371b42 — #237's launcher
  hunks add no new flag token), plus **16 more**
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
  :101 → :103 at e371b42.)
- **`alternative.sh` is entirely outside it** — no validation of its SPEC
  (its own `case` at `:75-81`, :74-80 at d5e2a01, refuses, duplicating the resolver's
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
(10, 13, 15, and batch's block 10 again). (2026-10-04) Since d5e2a01, 177ce26
(#237) touched all three launchers in one commit (blocks 13 and 14), as #189
did. The same command at e371b42 gives 16 `single-user/` + `batch/`
file-touches, 11 to the three launchers (`patches/` 64, `docs/` 48); the
40-commit window moved, so these do not compare one-to-one with the d5e2a01
figures.
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
| `qwen_detect_wsl` + `qwen_allocator_defaults` | block 10 | Split in two since #235 (re-verification 2026-09-29): batch now needs the WSL answer *before* its KV profile branch (`batch:95-107`, the WSL `GPU_UTIL`/`MAX_LEN` defaults) and again at the allocator (`:230`), so detection is its own early call that sets one variable, and the allocator function reads it. `qwen_allocator_defaults`: KV-connector + TP arms → sets `PYTORCH_CUDA_ALLOC_CONF`, plus the WSL pin-memory default in single's guarded form (`single:779-782`, #219) behind a has-drafter parameter (1.3.13). The long WSL2 essay (single:767-776) lives here; batch's "see the long note" pointer (1.3.11) dissolves. The connector match (`--kv-offloading-size`/`--kv-transfer-config`) is written twice in `alternative.sh` since #232 (`:23-24`, `:111-112`) — the function exports its answer so block 15 reuses it. |
| `qwen_int8_exports` | block 5 | the two-condition guard (1.3.1's fix) once; (2026-10-05: done in #273, with `if` blocks for `alternative.sh`'s `set -e`.) callers keep their own `INT8_ACT`/`INT8_LAYERS` *defaults*, which are genuinely per-mode (batch: `int8`/`mlp`, `:135-136`; single and alternative: off/`mlp\|linear_attn\|self_attn`, `single:139-140`, `alternative.sh:54-55` — this row said "off/`all`" before 2026-09-29; the literal is the three-class regex). |
| `qwen_tool_args` / `qwen_metrics_args` / `qwen_vision_args` | blocks 7-9 | the array builders with their #59 comments. The spec-decode metrics flag moves behind a `mode` parameter so batch stops carrying flags it cannot produce (1.3.2) — or simpler: the flag is harmless-but-dead in batch; keep it only if the owner wants one code path. Called out in the PR, not decided here. |
| `qwen_async_args` | block 12 | `ASYNC_SCHED` → `ASYNC_ARGS`; single's `:315` setter stays in single (it is dflash2-shaped), and so does `alternative.sh`'s `:93` setter — they differ (1.3.5), and unifying them is a behavior change for PR B to call out, not a move. The array build shares. |
| `qwen_retention_args` | block 15 | single's measured ladder (13056/14592), the override order, and the #174 warning move verbatim, with the **final fallback as a caller parameter** (single: `None`; `alternative.sh`: #232's `0` without a KV tier, `None` with one). Until #232 this row said `alternative.sh`'s `None` path "is the ladder's own fallback, argv-identical unless `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` is exported"; since 36936ec that is false — a fixed `None` fallback would regress #232's measured no-tier default (1.3.12). PR B deletes the copy for the call. Batch stays absent — no drafter. |
| `qwen_serve_argv` | block 14 | the exec line as an **array builder**: fills `ARGV=(venv/bin/vllm serve …)` from the caller's mode parameters. EXTRA_ARGS still expands last and still word-splits (the documented override door — resolve_config.sh:34-35); the array conversion makes the other expansions (`$VISION_ARGS`, `$KV_ARGS`, `$ATTN_ARGS`) explicit instead of relying on unquoted splitting. (2026-10-04) The builder emits `--host $BIND_HOST`, as all three exec lines do today. It does not compute the host: the caller runs `resolve_vllm_key` and then `resolve_bind_host` before it, as all three launchers do since #237 (`single:821-822`, `batch:280-281`, `alternative.sh:41-42`). (2026-10-05) Not in #272: `qwen_exec "$@"` takes the exec words as they are. |
| key resolution + bind host | block 13 | stays in `resolve_api_key.sh` (already shared). **Done upstream for `alternative.sh`** by 8cf642e (#219) — it sources the resolver at `:39-41` (1.3.8). (2026-10-04) #237 added `resolve_bind_host` to the same file (`:53-70`); it stays there and the chassis does not copy it. Nothing is left for the chassis but, optionally, giving both `start_qwen.sh` the refuse-on-failed-source guard `alternative.sh` has. Since #237 a failed source there also empties `BIND_HOST` (1.3.8), so the guard now protects the bind as well as the key. (2026-10-05) #272 moves `resolve_bind_host` to `launcher_common.sh`. A failed source then no longer empties `BIND_HOST`, and the launcher boots keyless. So the guard is needed, not optional (the block at the top). |

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

Upstream already needed this seam twice. #232 verified its retention change
with "a stub `vllm` on the launcher's `PATH` that prints its argv" (PR body,
7 cases). A `PATH` stub sees only `alternative.sh`, which execs a bare
`vllm serve` (`:141`); both `start_qwen.sh` exec the literal path
`venv/bin/vllm` (`single:824`, `batch:283`).

(2026-10-04) This paragraph used to say only `PRINT_ARGV=1` could give the
two `start_qwen.sh` the same check. That is false since 177ce26 (#237):
`bench/test_no_key_bind.sh` copies the checkout to a temp dir, writes the
stub to `$T/venv/bin/vllm` (`:11-14`) and runs all three launchers to the
exec. Its header says "No GPU, no model, no network". It is the third
hand-built argv capture, and no CI job runs it. `PRINT_ARGV=1` removes the
copy and the stub: the launchers print their argv in place, so the same rows
run as one CPU test in CI (3.3).

(2026-10-05) #272 builds the gate as `qwen_exec` in `launcher_common.sh`,
not as an `ARGV` array. With `PRINT_ARGV=1`, it runs `printf '%s\n' "$@"`
and `exit 0`. Otherwise it runs `exec "$@"`. #272 also converts
`test_no_key_bind.sh` to read the `PRINT_ARGV=1` argv. The temp-dir copy
stays, so the `api_key.txt` rows never touch a real key file.

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
  refuse shape, deleting its duplicate `case` at `:75-81` (:74-80 at d5e2a01).

### 2.4 The drift fixes that fall out (not the goal — the proof)

1.3's items resolve as follows: 1 (INT8 guards) → one shared guard. 2
(spec-decode flag in batch) → mode-parameterized builder, or a documented
harmless keep. 3 (batch's dflash2 VISION comment) → the comment lives once,
in the chassis, citing the mode it belongs to. 4 (HOST knob) → **done
upstream** (177ce26, #237): all three launchers pass `--host $BIND_HOST` from
the shared `resolve_bind_host`; the argv builder only carries it (2.1,
2026-10-04). 5, 6 (async, CG) →
one builder each. 7 (alternative missing #185/#176) → inclusion. 8 (key
flavor) → **done upstream** (8cf642e, #219). 9 (stale kvarn name) → fixed in passing, one
comment. 10 (five model defaults) → `select_model.sh` stays the one
selector; `alternative.sh:68-69` and `resolve_config.sh:82` defer to it;
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

(2026-10-04) PR A takes in #237's bind. `qwen_serve_argv` emits
`--host $BIND_HOST`, and both `start_qwen.sh` keep calling `resolve_vllm_key`
and then `resolve_bind_host` before it (`single:821-822`, `batch:280-281`).
`resolve_bind_host` stays in `resolve_api_key.sh`. Its answer depends on
`HOST`, the key (`VLLM_API_KEY`, or `api_key.txt` through `resolve_vllm_key`)
and `/.dockerenv` (`resolve_api_key.sh:53-70`). So every matrix row pins the
key and `HOST`; otherwise the main-vs-branch diff depends on the machine.

Acceptance: `bash -n` clean on all three shells' targets; **byte-identical
argv** — for each profile in the test matrix (section 4), `PRINT_ARGV=1` on
main vs the branch must diff empty; `patch-integrity`'s existing
`test_model_verification.py` still passes; the image build is green (it runs
`verify.sh --install`, not the launchers, so it is a smoke check only).
(2026-10-04) Also: `bash bench/test_no_key_bind.sh` passes on the branch.
Its stub sits at `venv/bin/vllm`, which stays `ARGV[0]`.

(2026-10-05) #272 is a smaller PR A. It adds the chassis file, moves
`resolve_bind_host` into it, and adds the `PRINT_ARGV=1` gate to all three
launchers. The block moves listed above are not in it; they can follow one
at a time. Its acceptance rows passed (the block at the top).

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

(2026-10-05) Not started. The tracker's "PR B" (#273) is drift item 1 only.
`alternative.sh` gets the chassis source line, `qwen_exec` (#272) and
`qwen_int8_exports` (#273). It keeps every other copy listed above.

### 3.3 PR C — the validation layer catches up + CI assertions

`qwen_shadow_warnings` replaces the hand list in `resolve_config.sh`;
single reorders select-then-resolve (1.4); a new `bench/test_launcher_argv.py`
(in `test_model_verification.py`'s existing unittest style) runs the matrix
below against `PRINT_ARGV=1` and is wired into `patch-integrity.yml`'s
CPU job — it runs in seconds, no GPU. The two systemd units stay as they are;
their only change is optional (a comment pointing at the chassis).

(2026-10-04) `bench/test_no_key_bind.sh`'s six launcher rows (`:25-30`), run
against each of the three launchers, become rows of the `PRINT_ARGV=1` matrix
in `bench/test_launcher_argv.py` (section 4). The temp-dir copy of the
checkout and the stub go. Its four `verify.sh` rows (`:36-39`) run no
launcher; they stay in `test_no_key_bind.sh`, and PR C adds that script to
the same CPU job (`model-verification`, a plain `ubuntu-latest` runner, so
the script's `/.dockerenv` skip does not fire).

Acceptance: the new test file is in the workflow and green; `grep -c`
on `resolve_config.sh`'s shadow list is gone in favor of the derived check;
the `[effective-config] MODEL=` print matches `select_model.sh` on a native
single-mode dry run. (2026-10-04) The six bind rows pass for all three
launchers, and `test_no_key_bind.sh` runs in the CPU job.

(2026-10-05) Not started. #271 (in review) adds `test_no_key_bind.sh` to the
`model-verification` job. With #272 merged, that job runs the `PRINT_ARGV=1`
host rows; with #273, it runs the INT8 rows too.

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
| `INT8_ACT= INT8_LAYERS=mlp bash batch/…` | no `VLLM_MARLIN_INT8_INCLUDE_RE` exported (1.3.1). (2026-10-05: a row in #273's `test_no_key_bind.sh`, one of three INT8 rows per launcher) |
| `VISION=1` | the mm flags, no `--language-model-only` |
| bind host, all three launchers (2026-10-04; replaces the old `HOST=127.0.0.1` (batch, PR C) row, which #237 made upstream behavior) | the six rows of `bench/test_no_key_bind.sh:25-30`: no key, no `HOST` → `--host 127.0.0.1`; `VLLM_API_KEY=k` → `0.0.0.0`; key only in `api_key.txt` → `0.0.0.0`; no key, `HOST=0.0.0.0` → `0.0.0.0`; key, `HOST=127.0.0.1` → `127.0.0.1`; key, `HOST=10.1.2.3` → `10.1.2.3` (`resolve_api_key.sh:53-70`) |
| `WSL_DISTRO_NAME=x KV=kvarn bash batch/…` | `--gpu-memory-utilization 0.88 --max-model-len 131072` (#235, `batch:98-107`); `KV=fp8` → `0.91`, `150000`; an explicit `GPU_UTIL`/`MAX_LEN` wins |
| `WSL_DISTRO_NAME=x SPEC=dflash2` (single) | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` exported (#219, `single:779-782`); with `VLLM_WSL2_ENABLE_PIN_MEMORY=0` exported it stays 0 |
| `alternative.sh`, defaults (PR B) | `--prefix-cache-retention-interval 0`; with `EXTRA_ARGS="--kv-offloading-size 8"` → `None` (#232, `alternative.sh:111-116`) |

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
| (2026-10-05) A failed `resolve_api_key.sh` source boots keyless. Since #272 `resolve_bind_host` survives it, so the argv is valid (1.3.8) | low: the file ships with the repo | the refuse-on-failed-source guard of `alternative.sh:39-40` in both `start_qwen.sh`, with a `test_no_key_bind.sh` row that removes the file |
| The dry-run becomes a way to "test" a boot that then OOMs | low | the doc is explicit: PRINT_ARGV validates *argument assembly*, never that the config fits the card — the launcher's own measured ladders keep that job |

## 6. Done when

1. `grep -c "expandable_segments" single-user/start_qwen.sh batch/start_qwen.sh`
   → 0 and 0 (9 and 9 at e371b42); the string lives once, in `launcher_common.sh`.
   (2026-10-05: 9 and 9 at #273.)
2. `grep -c "qwen3_coder" batch/start_qwen.sh single-user/alternative.sh`
   → 0 and 0 (3 and 1 at e371b42; the parser name lives once, in the chassis).
   (2026-10-05: 3 and 1 at #273.)
3. `PRINT_ARGV=1` produces the full argv from all three launchers, and
   `bench/test_launcher_argv.py` runs green in `patch-integrity.yml`'s CPU job.
   (2026-10-04: neither exists upstream. `bench/test_no_key_bind.sh` captures
   argv through a stub in a copied tree, and no CI job runs it.)
   (2026-10-05: half done, in review. #272 adds `PRINT_ARGV=1` to all three
   launchers, and #271 runs `test_no_key_bind.sh` in CI. There is no
   `bench/test_launcher_argv.py`.)
4. `resolve_config.sh` contains no hand-maintained flag list (at e371b42 it
   still has the 11-flag list at `:68`). (2026-10-05: unchanged at #273.)
5. `comm -12 <(sort -u single-user/start_qwen.sh) <(sort -u batch/start_qwen.sh)`
   on unique substantive lines drops from 76 (78 at e371b42, +2 from #237) to near zero — what remains is
   the profile data that genuinely differs. Measure it with GNU coreutils or a
   Python set intersection; uutils `sort`/`comm` gave unstable results (see the
   2026-10-04 note). (2026-10-05: 80 at #273. The two `source
   launcher_common.sh` lines are shared.)
6. A native `PRINT_ARGV=1` single-mode run prints the `-fast` model when it
   exists (1.4's bug is gone). (2026-10-05: unchanged at #273.)
