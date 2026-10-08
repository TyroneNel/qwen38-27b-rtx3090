# One door into the patch series — remediation plan

Architecture review 2026-09-26, candidate 1 (Strong). The full report with the
other six candidates is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this document is the deep dive on that card: the current state measured line by
line, the design, and the rollout.

**Re-verified 2026-10-04 against upstream/main @ e371b42 (vLLM 0.30.0).**

**Status:** Merged upstream as e1459c7 (squash of #242/#243/#244, 2026-09-30). Done-when: 6 of 6 pass at e371b42.

**2026-10-08:** the one item left, a check that `apply.sh`'s `KVARN` array agrees with `kvarn/*.patch`
(`apply.sh --kvarn` exits 2 when they disagree), merged with #274 as `76e1a15` (upstream/main @ `9133015`).

Thirteen commits landed since d5e2a01 (`git log d5e2a01..e371b42`). e1459c7 is
this plan. Four others touch the patch series: #233 (e7a5823), #261 (69036cb),
#263 (210db97) and #262 (bd6c5e2). Section 1 now records the d5e2a01 tree,
before the merge. Each of its subsections ends with what e371b42 has.

- **Done when (section 6), item by item.** (1) `grep -rln "s/#.\*//"
  --include='*.sh' --include='Dockerfile' .` prints only `./patches/apply.sh`.
  `grep -rln "s/#\.\*//" --include='*.md' .` in the e371b42 tree finds no
  file. (2) The `git grep -l
  dflash2-backport` of 6.2 prints `PATCHES.md`, `docs/docker.md` and
  `docs/optimizations.md`. (3) There are 46 `patches/*.patch` files and 46
  series lines, and `bash patches/apply.sh --list` exits 0: the plan's 44, plus
  #233's `sampler-warmup-cuda` and #263's `pinned-kv-empty-cache`. (4) The
  `patch integrity` run on e371b42 (37189534383) logs "46 patches applied with
  exact context, 3 of them at an offset, 0 with fuzz" and "4 KVarN patches
  applied after the series; 50 in total". The run on e1459c7 (36754957337)
  logged 45 + 3 = 48. (5) `README.md:121-122` reads "series (`patches/`, one
  line each in PATCHES.md)". (6) `kvarn/install.sh:21` calls `apply.sh
  --kvarn`. The only heredoc left (`:23-32`) is the registration probe, which
  2.2 keeps. `verify.sh:110-127` checks all four KVarN patches with the exact
  reverse dry-run.
- **The merge matches the design (2.1-2.4).** `apply.sh --list` checks the
  agreement (`patches/apply.sh:62-75`). The apply mode runs `patch -p1
  --forward --fuzz 0 --no-backup-if-mismatch` (`:80`). `--kvarn` skips a patch
  that reverses exactly, else applies it strictly (`:108-118`). The Dockerfile
  (`:31-34`), `verify.sh:76-84`, `patches/check_vllm_series.sh:40-70`,
  `docs/install.md:113-114` and `docs/python-314.md:74-75` call it. The
  backport patch, its special cases and both 0.27.1 KVarN files are gone. The
  backport row is retired prose at `PATCHES.md:73-76`, and `README.md`,
  `patches/series:3-6`, `check_vllm_series.sh:8-10` and `PATCHES.md:89-90`
  carry the 2.4 fixes. Two small deviations: `docs/python-314.md:74` keeps the
  `VP=` line (2.2 said it goes; the command needs it), and
  `check_vllm_series.sh:9-10` lists the Dockerfile, `docs/install.md` and
  `kvarn/install.sh` as callers but not `docs/python-314.md`.
- **The later patches used the door.** #233 and #263 each add a file, a
  `patches/series` line (`:61`, `:62`) and a `PATCHES.md` row (`:44`, `:40`),
  and CI counts them. #233 (e7a5823) merged before e1459c7, so the merge took
  it in; #263 is the first new series patch that merged through `apply.sh`. #262 adds `kvarn/kvarn-fp16-dequant-0.30.0.patch` to the
  one `KVARN` array (`apply.sh:50-51`), a FAIL rung (`verify.sh:123-127`), a row
  (`PATCHES.md:69`), a rerun no-op check in the fast gate
  (`check_vllm_series.sh:71-82`; the e371b42 run logs "a second apply.sh
  --kvarn run: all 4 already applied") and a CPU torch job
  (`patch-integrity.yml:25-37`). All 50 patch files carry an `exported from`
  marker (`grep -L -- '--- exported from cpuchip/vllm' patches/*.patch
  kvarn/*.patch` prints nothing). Open PR #260 adds `api-root-health` the same
  way: series line, row and marker (`gh pr diff 260`).
- **New gap: the KVarN list has no agreement check.** `--list` guards the
  series. Nothing compares `KVARN` (`apply.sh:50-51`) with `kvarn/*.patch`, so
  a KVarN patch left out of the array is never applied and never gated. The
  count "four" is written in `apply.sh:10`, `check_vllm_series.sh:8` and
  `kvarn/README.md:32`.
- **Export provenance drifted. The apply path did not.** `PATCHES.md:12` names
  one fork, `cpuchip/vllm`, and the export tag `qwen38/0.30-cut5`. `git
  ls-remote https://github.com/cpuchip/vllm` shows tags up to
  `qwen38/0.30-cut9`. `gh api repos/cpuchip/vllm/compare/<cut5>...<hash>` puts
  12 of the 50 export hashes outside cut5:
  - #263's `pinned-kv-empty-cache` (bbed74b28) and #262's
    `kvarn-fp16-dequant-0.30.0` (7f8c3ef0a) are on TyroneNel/vllm (`git
    ls-remote https://github.com/TyroneNel/vllm`): branch
    `qwen38/0.30-pinned-kv-empty-cache` and tag `qwen38/0.30-pinned-kv-cut1`
    (→ bbed74b28); branch `qwen38/0.30-kvarn-fp16` and tag
    `qwen38/0.30-kvarn-fp16-cut1` (→ 8ad33d165, which contains 7f8c3ef0a).
    They are the heads of open PRs cpuchip/vllm#4 and cpuchip/vllm#3.
    `PATCHES.md:12` and `kvarn/README.md:33-35` name `cpuchip/vllm` for refs
    that exist only on TyroneNel/vllm until cpuchip merges #3 and #4.
  - #261 re-exported `dflash2-ngram-chains` (596a96779) and
    `kvarn-v2-runner-0.30.0` (757257e13). Only cpuchip's untagged branch
    `qwen38/0.30-chainfix` contains them, and no doc names that branch.
    `kvarn/README.md:33` still says the first three KVarN patches come from
    `qwen38/0.30`.
  - #233's `sampler-warmup-cuda` (92e7256fa) is the head of cpuchip's branch
    `qwen38/0.30-warmup-cut7`. The tag `PATCHES.md:12` cites,
    `qwen38/0.30-warmup-cut1`, does not contain it.
  - #234's four re-exports (`marlin-int8-asym-zp`, `marlin-repack-staged-sm80`,
    `speed-knobs-envs`, `kvarn-0.30.0`) are in `qwen38/0.30` (cut9), and
    `spec-attn-smem-fit` (ede631297e) is the cut6 commit.
  - The GitHub API resolves neither of two older 7-character hashes on either
    fork: `bench-sse-keepalive` (757723b, #226) and
    `kvarn-recycled-pages-0.30.0` (40ab8e0, #222). [INFERENCE: not run:
    whether the short hash is ambiguous or the commit is absent.]
- **`scripts/export-patch.sh:4` still cites `docs/fork-workflow.md`.** The file
  does not exist (`ls docs/fork-workflow.md`).
- **3.4, both owner items.** `docs/python-314.md`: e1459c7 took the banner
  option (`:8-13`). The banner still says nobody has re-run the page on
  0.30.0. Under the banner,
  `:20` ("applies unmodified"), `:49-55` ("fifteen") and `:71`
  (`vllm==0.27.1`, then the 0.30.0 series at `:75`) still stand. The
  `docs/MR-DRAFT.md` link is closed: `PATCHES.md:89-90` points at
  `docs/gotchas.md` alone. The same sentence says two files "still carry raw
  `diff -ruN` headers", but `grep -c '^diff -ruN'` finds 0 in both.
- **Not run at this pass:** the negative controls of section 4 and
  `verify.sh --install` in a built image [INFERENCE: not run]. The CI runs
  above cover the pristine apply and the KVarN rerun.

**Status 2026-09-30: implemented as three upstream PRs, still against
upstream/main @ d5e2a01.** PR A is syv-ai/HyperQwen#242, PR B #243 and PR C
#244, all ready for review as a stack (each branch contains the ones below
it). Upstream squash-merges, which breaks a plain stack, so each PR body
carries a merge guide: squash #244 alone to take all three, or squash one at a
time with `git rebase --onto origin/main <previous commit>` between merges
(both ways simulated; same final tree). Running the checks on a pristine v0.30.0
checkout changed two parts of this plan:

- **1.3 overstated the marker decay.** On a fresh install, `apply_kvarn`'s
  `FAILED` grep catches a bad hunk in any of the 27. The silent case is a
  *rerun over a partly applied tree*: `patch -N` skips a whole file when the
  file's first hunk is present, and prints no `FAILED`. Then only the 3 marker
  hunks are guarded.
- **PR C's check was too weak.** `_check_applied.py` passed a tree with hunk 4
  of kvarn-v2-runner's `attention.py` removed (the 80% per-file rule). So does
  `verify.sh:120`, which uses it. Each KVarN patch reverses cleanly on its own
  in a fully installed tree, so an exact `patch -R --dry-run --fuzz 0` works.
  PR C uses it in `apply.sh --kvarn`: skip if every hunk is present, else apply
  strictly, so a partly applied patch fails by name. `verify.sh` uses it for
  all three patches. `--forward` as the idempotence mechanism for KVarN (2.1)
  is dropped, because it is the same `-N` behaviour. The sections below are
  corrected in place.

Open upstream PR #233 (`sampler-warmup-cuda`) adds a 46th patch. It merges
cleanly with #242 and conflicts with #243/#244 on one sentence,
`PATCHES.md:12`; the counts below become 45/48 once both land.

**Re-verified 2026-09-29 against upstream/main @ d5e2a01 (vLLM 0.30.0).**
Seven commits landed since 2522ef9; three touch this card:

- **Partially fixed upstream — #231 (8d9848d).** `--no-backup-if-mismatch` was
  added to the three apply sites that lacked it: `docs/install.md:119`,
  `docs/python-314.md:70` and `kvarn/install.sh:16` (the Dockerfile already had
  it, `:36`). All five apply sites now share that flag; the `--fuzz 0` omission
  in both prose loops (1.4.1, 1.4.2) is untouched, so the policy still diverges
  on the flag that matters. Line numbers in all three files are unchanged.
- **Shifted, not fixed — #219 (8cf642e).** `verify.sh --wait` added 19 lines
  above the patch block: every verify.sh citation moves +19 (parse `:60`→`:79`,
  agreement check `:63-68`→`:82-87`, ladder `:80-91`→`:99-110`, backport arm
  `:82-85`→`:101-104`, KVarN block `:98-107`→`:117-126`). #219 also prepended
  9 lines to `docs/docker.md` (backport mention `:9`→`:18`).
- **No new series consumer.** #219's `Makefile` (`verify-install:` runs
  `bash verify.sh --install`, `Makefile:31-32`) and `scripts/hq-doctor.sh`
  neither read nor parse `patches/series`; `grep -rn "s/#\.\*//"` (outside
  `venv/`) still finds the parse in exactly the five places of 1.1.
- **Touched, counts unchanged — #234 (e355f9f)** edited `kvarn-0.30.0.patch`
  (export hash, one `index` line, one context comment): still 13 hunks, 0
  markers.
- **Unchanged:** 45 series entries = 45 `patches/*.patch`; the seven
  dflash2-backport touch points; the stale consumer claims; README's
  "38 files"; `PATCHES.md:83`'s dangling MR-DRAFT; the fast gate's KVarN blind
  spot; the 3-of-27 marker decay.
- **Corrected at this pass** (wrong already at 2522ef9): "27 hunks across 17
  files" counts per-patch file sections — 15 distinct files; the
  optimizations.md backport mention is `:214-215`, not `:213`; python-314.md's
  0.27.1 is three pin flips behind, not two; the CI timings (1.5, 3.3) are
  replaced with `gh run list` durations; the "byte-identical" claim (1.1) and
  the `grep -r dflash2-backport` done-when criteria (3.2, 6.2) are restated so
  they are true/reachable; the install.md row of 2.2 used python-314.md's
  `$VP` (install.md's variable is `$SP`); the heredoc range `:44-71` is
  `:38-71` with its comment. New finding: `docs/MR-DRAFT.md` was deleted by
  #131, which settles half of the 3.4 owner question; PyPI metadata for
  vllm 0.30.0 (cp38-abi3, `<3.15`) partly answers the other half.

**Previously re-verified 2026-09-28 against upstream/main @ 2522ef9 (vLLM 0.30.0)**
(line numbers in this block are as of 2522ef9):

- The pin flip the drift sections anticipate *happened*: #189 pinned
  `vllm==0.30.0`, re-exported the whole series from a new fork branch
  (`qwen38/0.30`, tag `qwen38/0.30-cut5`), and touched every consumer below
  (67 files). It also removed `offload-mtp-serve` and
  `mamba-align-retire-null-gaps` from the tree — the remove-on-retire
  precedent (1.2) applied again (`PATCHES.md:69`).
- The series is still 45 entries at `series:15-59` by churn, not stasis: two
  retired out, two added — `spec-attn-smem-fit` (`:58`, #188) and
  `bench-sse-keepalive` (`:59`, #226).
- KVarN is now three patches: `kvarn-0.30.0` (13 hunks),
  `kvarn-v2-runner-0.30.0` (12 hunks, 1 marker), `kvarn-recycled-pages-0.30.0`
  (2 hunks, 2 markers; #208/#222). install.sh applies all three
  (`:18`,`:21`,`:24`); the marker heredoc now covers 2 patches and 3 markers —
  a 3-of-27 decay, the same shape as 1.3.
- Partial upstream fixes: #191 gave install.md's prose loop the venv-python SP
  resolution (`:113`) — the fuzz-0 omission and the duplicated skip arm remain
  (`:114-120`). verify.sh checks all three KVarN patches (`:98-107` at 2522ef9,
  `:117-126` at d5e2a01: `kvarn-0.30.0` by exact reverse dry-run, the other two
  with `_check_applied.py`); the weak install-time counterpart persists.
- Still standing: five series parsers; the seven dflash2-backport touch
  points; the stale consumer claims (`series:3-4`, `check_vllm_series.sh:8-9`);
  README's "38 files" (`:122`); python-314.md's 0.27.1 premise (written "now
  two pins behind"; it is three pin flips behind — 0.28.0 `dfb31d1`, 0.29.0
  #148, 0.30.0 #189); `PATCHES.md:83`'s dangling MR-DRAFT.
- Line numbers, counts and patch names throughout updated to this tree (and
  again to d5e2a01 on 2026-09-29).

**Scope.** Everything that parses, applies, or re-lists `patches/series`; the
retired artifacts that force special cases into those consumers; the stale
references the duplication has already produced. **Not in scope:** the harness
verdict conventions (candidate 5), generating the PATCHES.md table (candidate
6), and any change to the verify.sh *checking ladder* — that ladder is state
diagnosis, a deliberately independent second look, and it stays.

## 1. State before the merge (d5e2a01), precisely

This section records the d5e2a01 tree, before e1459c7. Its line numbers are
d5e2a01's unless a citation says "at e371b42". Each subsection ends with an
**At e371b42** line that gives the current state.

### 1.1 The ordered series was parsed in five places

`patches/series` (45 entries, `series:15-59`; 45 `patches/*.patch`, and the two
sets agree — recounted at d5e2a01) is the canonical order. The same sed parse
is re-encoded in three code consumers and two prose copies (no sixth: #219's
`Makefile` and `scripts/hq-doctor.sh` do not read the series):

| # | place | form | guarded against drift? |
|---|-------|------|------------------------|
| 1 | `Dockerfile:31` | sed copy, in the build's apply loop | no |
| 2 | `verify.sh:79` (was `:60` at 2522ef9; #219's `--wait` shifted it +19) | sed copy, feeds the check ladder | partially (dir↔series agreement, `:82-87`, was `:63-68`) |
| 3 | `patches/check_vllm_series.sh:45` | sed copy, feeds CI pass 1 | same partial check (`:48-52`) |
| 4 | `docs/install.md:113-120` | prose copy of the whole loop | **drifted** |
| 5 | `docs/python-314.md:68-71` | prose copy of the whole loop | **drifted** |

The sed expression is byte-identical in all five places (`grep -ho` over the
five files: one distinct string, five hits); only the file argument differs —
`check_vllm_series.sh:45` reads `"$HERE/patches/series"`, the others
`patches/series`:

```bash
sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' patches/series
```

**At e371b42:** one place. The parse is `series()` in `patches/apply.sh:58-60`.
The five consumers call `apply.sh` (`Dockerfile:32`, `verify.sh:78`,
`check_vllm_series.sh:40`, `docs/install.md:114`, `docs/python-314.md:75`).
The series has 46 entries at `series:17-62`.

### 1.2 One retired patch had seven touch points

`dflash2-backport.patch` is retired (DFlash2 is native since vLLM 0.28.0) but
kept in the tree, so every consumer carries a special case:

| touch point | what it does |
|---|---|
| `patches/series:23` | still listed |
| `Dockerfile:34` | `case` arm: skip with a message |
| `patches/check_vllm_series.sh:37` | `SKIP=(dflash2-backport.patch)` |
| `verify.sh:101-104` (was `:82-85` at 2522ef9) | "retired" branch inside the check loop |
| `docs/install.md:117` | the same skip arm, in prose |
| `PATCHES.md:23` | a table row already marked `RETIRED` |
| `PATCHES.md:12`, `docs/docker.md:18` (was `:9` at 2522ef9; #219 prepended 9 lines, a WSL2 checklist), `docs/optimizations.md:214-215` (cited as `:213` at 2522ef9, which is the paragraph's first line; the mention is on 214-215 in both trees) | historical prose (fine to keep — see 3.2) |

The same "kept for history" pattern holds ~36 KB (36,348 bytes) of inert KVarN files:
`kvarn/kvarn-0.27.1.patch` (16,041 B) and `kvarn/kvarn-v2-runner.patch`
(20,307 B), referenced only by `kvarn/README.md:50` ("the older ports, kept
for the diff history") and by `kvarn-v2-runner.patch`'s own header (`:90-91`);
`kvarn/install.sh` applies neither. The fork
branch is the declared source of truth (`PATCHES.md:12-16`), and this repo is a
git repository — the diff history is not at risk.

This contradicts the repo's own remove-on-retire precedent
(`PATCHES.md:71-73`: the 0.29.0 retirements were *removed from the tree* —
as were the two 0.30.0 ones, `:69`).

**At e371b42:** e1459c7 deleted `dflash2-backport.patch`, its five special
cases, `kvarn/kvarn-0.27.1.patch` and `kvarn/kvarn-v2-runner.patch`. The
backport and the two old ports are retired prose at `PATCHES.md:73-76`, next
to the 0.30.0 (`:71`) and 0.29.0 (`:78-80`) retirements. `PATCHES.md:12` no
longer mentions the backport. The historical notes stay at `docs/docker.md:18`
and `docs/optimizations.md:214-215`. The fork-as-source sentence is still
`PATCHES.md:12-16`.

### 1.3 Four different answers to "is the series healthy?"

- **Dockerfile**: apply-only; health = the build succeeds (with
  `verify.sh --install` at the end).
- **check_vllm_series.sh**: pass 1 applies the whole series with GNU
  `patch --fuzz 0`; pass 2 re-checks the five contractual DFlash patches with
  `git apply`. The two-pass split is load-bearing (`:6-20`) and stays.
- **verify.sh**: a four-rung ladder per patch (`:99-110`, was `:80-91` at
  2522ef9): reverse dry-run →
  `_check_applied.py` content check → `Supersedes:` lookup → forward dry-run
  diagnosis. Each rung exists because the rung above has a named failure mode.
- **kvarn/install.sh**: greps `patch -N` output for `FAILED` (`:16-17`; #231
  added `--no-backup-if-mismatch` to `:16`, line numbers unchanged), plus a
  Python heredoc (opened at `:26`; the marker count is `:38-70`, closed `:71`)
  that counts `port(kvarn-v2)` markers per file across two of the three
  patches (`PORTS`, `:46`) as its "every hunk landed" proof.

The marker check has silently decayed — and the decay survived the 0.30 port
intact. Re-measured 2026-09-29 at d5e2a01 (`grep -c '^@@'`, `grep -c '^+++ b/'`,
and `grep '^+' | grep -v '^+++' | grep -c 'port(kvarn-v2)'` per file; unchanged
from 2026-09-28 although #234 edited `kvarn-0.30.0.patch`): the three KVarN
patches carry **27 hunks across 17 per-patch file sections (15 distinct files —
"17 files" at 2522ef9 double-counted the two files touched by more than one
patch), and the heredoc counts 3 `port(kvarn-v2)` markers in 2 of
the 3 patches** (`kvarn-v2-runner-0.30.0.patch`: 12 hunks, 7 files, 1 marker;
`kvarn-recycled-pages-0.30.0.patch`: 2 hunks, 2 files, 2 markers;
`kvarn-0.30.0.patch`: 13 hunks, 8 files, no markers) — it verifies 3 of 27
hunks and prints "port complete" (`:70`). *(Corrected 2026-09-30: this is a
rerun problem, not a fresh-install one. On a fresh install the `FAILED` grep
catches a bad hunk anywhere. On a rerun, `patch -N` skips a whole file when
its first hunk is present, so a missing later hunk in that file passes both
the grep and, unless it carries a marker, the heredoc.)* Meanwhile
`verify.sh:120-126` (was `:101-107` at 2522ef9) checks the v2-runner and
recycled-pages patches with `patches/_check_applied.py`, the content checker
whose 80%-per-file heuristic was written after PR #43. *(Measured 2026-09-30:
that heuristic is not strong enough for KVarN either. It passes a tree with
hunk 4 of v2-runner's `attention.py` removed. The exact reverse dry-run
catches it, and it works for all three KVarN patches.)*

**At e371b42:** two answers, as section 6 intends. The Dockerfile, the CI gate
and `kvarn/install.sh` all apply through `apply.sh`. `verify.sh`'s ladder
stays (`verify.sh:96-103`; `Supersedes:` lookup `:85-93`). The two-pass split
is `check_vllm_series.sh:6-22`. The marker heredoc is gone: `kvarn/install.sh`
calls `apply.sh --kvarn` (`:21`), which checks each KVarN patch with the exact
reverse dry-run (`apply.sh:108-118`). `verify.sh:110-127` uses the same check
for all four KVarN patches (#262 added the fourth).

### 1.4 Drift on the ground at d5e2a01

Every duplication above has produced at least one live inconsistency. All
confirmed against the tree on 2026-09-26 (line numbers refreshed 2026-09-28,
re-checked 2026-09-29 at d5e2a01):

1. **`docs/install.md:113-120` drops `--fuzz 0`.** The prose loop applies with
   `patch -p1 --no-backup-if-mismatch` (`:119`; it was plain `patch -p1` at
   2522ef9 — #231 added the backup flag and nothing else). A hunk whose
   context moved lands by approximate
   anchor instead of failing by name — the exact failure mode the fuzz-0 rule
   (`check_vllm_series.sh:56-59`) was written to prevent. (#191 fixed the
   loop's other drift — `:113` now asks the venv's python for the package
   directory — and #231 the `.orig` backups; both left this one standing.)
2. **`docs/python-314.md` is stale as a whole.** The 0.27.1 pin there is
   deliberate (the abi3 wheel is why 3.14 works at all, `:9-11`), but
   "Every patch in `patches/` applies unmodified" (`:13`) and "all fifteen
   land" (`:42-45`) were written against the 0.27.1-era series. At d5e2a01 the
   series was 45 patches cut against 0.30.0; the doc's loop could not produce
   a working install. Its loop (`:67-71`) also lacked both `--fuzz 0` and the
   backport skip; since #231 it does carry `--no-backup-if-mismatch` (`:70`).
3. **`README.md:122` says "38 files".** There are 45. The README split (#129)
   moved the install loop to `docs/install.md`, so…
4. **`patches/series:3-4` names "the README install loop"** as a consumer. No
   such loop exists anymore (`grep -n 'patch -p1' README.md`: no hit at d5e2a01).
5. **`patches/check_vllm_series.sh:8-9`** makes the same stale claim ("which
   the Dockerfile and the README use").
6. **`PATCHES.md:83`** sends the reader to `docs/MR-DRAFT.md`, which does not
   exist in the tree. It did: #131 (1bb9d2b, 2026-09-17, "delete the stale
   PR-body source") removed it (`git log --all -- docs/MR-DRAFT.md`), and
   `PATCHES.md:83` was never updated. Its last version mentions
   `offload-wsl2-devptr` (`:17`) but not `dflash2-z-adaptive-emitted`.

**At e371b42:** e1459c7 fixed 1, 3, 4, 5 and 6, and bannered 2.
(1) `docs/install.md:114` runs `bash patches/apply.sh "$SP"`, so `--fuzz 0`
applies; the policy text is `patches/apply.sh:21-27`. (2)
`docs/python-314.md:8-13` marks the page as a 0.27.1 record; `:20`, `:49-55`
and `:71` are unchanged under it (3.4). (3) `README.md:121-122` has no count.
(4) `patches/series:3-6` and (5) `check_vllm_series.sh:8-10` name `apply.sh`
and its callers. (6) `PATCHES.md:89-90` points at `docs/gotchas.md` alone.

### 1.5 The fast CI gate never saw the KVarN patches

`patch-integrity.yml` runs `check_vllm_series.sh` (`:45-46`), which covers the
44 non-retired series patches (45 minus the `SKIP` entry). The three KVarN
patches — which carry the `CTX=huge` correctness fixes — are exercised only by
the image build (`docker-image.yml`, which also skips docs-only changes via
`paths-ignore`). Measured wall time (`gh run list`, 2026-09-28 runs, created →
updated incl. queueing): `patch integrity` 14–46 s, successful `docker image`
runs 4–12 min. The earlier "~20-minute image build" was the *local* build
figure from `docker-image.yml:2`, not the CI job; the gap is still an order of
magnitude. Yet they are exported from the fork branch at a
position *after the whole series* (`kvarn/install.sh:13-15`, whose comment
still says "both files" — a third apply was added in #222, commit 2522ef9),
which is exactly the point pass 1 reaches: the fast gate can cover them with
no new machinery.

**At e371b42:** the fast gate covers them. `patch-integrity.yml:59-60` runs
`check_vllm_series.sh`, whose pass 1 applies the KVarN patches after the
series (`:61-70`) and checks that a rerun is a no-op (`:71-82`). The e371b42
run (37189534383, 35 s created → updated) logs "4 KVarN patches applied after
the series; 50 in total". `docker-image.yml:2` still carries the local
"20-minute" figure.

## 2. The design

One new executable, `patches/apply.sh`, owns the series parse and the apply
policy. Every consumer either calls it or shows its one command line.

### 2.1 Interface

```text
patches/apply.sh --list
    Print the ordered patch basenames, one per line. Before printing, enforce
    that the list and the patches/ directory agree exactly; on disagreement,
    name the offenders on stderr and exit 2. (At d5e2a01 that check ran only
    in verify.sh and check_vllm_series.sh; moving it here gives it to every
    consumer, including the Docker build, for free. At e371b42 it is
    apply.sh:62-75. It still prints the list on disagreement, so verify.sh can
    FAIL by name and keep checking.)

patches/apply.sh DIR
patches/apply.sh --kvarn DIR
    (As implemented in #242/#244; this was `[--forward] [--kvarn] DIR`.)
    Apply the whole series to DIR (the installed vllm package directory) with
    `patch -p1 --forward --fuzz 0 --no-backup-if-mismatch`, in series order. (At
    d5e2a01 all five apply sites passed --no-backup-if-mismatch, since #231,
    but the two prose loops lacked --fuzz 0. e1459c7 replaced both loops with
    this command.) Stop at the
    first failure; the message names the patch. Per patch, print one line:
    `== <name>` (or `== <name> (N hunks at an offset)` — offsets are benign
    and reported, fuzz is refused; the policy text written at
    check_vllm_series.sh:56-59 at d5e2a01 moved here with the code it
    describes, apply.sh:21-27 at e371b42).
    --forward is always on, so a second series run fails by name instead
    of prompting; it is not the idempotence mechanism (see --kvarn).
    --kvarn     apply kvarn/kvarn-0.30.0.patch, kvarn/kvarn-v2-runner-0.30.0.patch,
                kvarn/kvarn-recycled-pages-0.30.0.patch and (since #262)
                kvarn/kvarn-fp16-dequant-0.30.0.patch, in that order, to a
                tree that already has the series — they are exported to apply at
                exactly this point (install.sh:13-24 at d5e2a01; the order and
                its reasons are apply.sh:46-51 at e371b42). Per patch: an exact
                reverse dry-run first; all hunks present → "(already applied)",
                skipped; else a strict forward apply, so a partial patch fails
                by name. (Corrected 2026-09-30; it was "continue with ... after
                the series", idempotent through --forward.)

Exit codes: 0 success · 1 an apply failed (patch named) · 2 usage error or
series/directory disagreement.
```

The sed parse from 1.1 lives in this file, once. `patches/series` stays the
single source of truth for *order*; `apply.sh` becomes the single source of
truth for *how the order is read and applied*.
### 2.2 What each consumer becomes

e1459c7 made every row below. The "before" column cites d5e2a01; the "after"
column cites e371b42.

| consumer | before (d5e2a01) | after (e371b42) |
|---|---|---|
| `Dockerfile:28-39` (`:28-34` at e371b42) | sed loop + skip arm | `bash patches/apply.sh "$SP"; bash kvarn/install.sh; bash verify.sh --install` in one `RUN set -e` (`:31-34`; the plan wrote `&&`) |
| `verify.sh:71-87` (was `:52-68` at 2522ef9) | own sed parse + agreement check | `SERIES` from `patches/apply.sh --list` (`:76-84`); the four-rung ladder (`:96-103`) is untouched; the backport arm (`:101-104` at d5e2a01) died in step 3.2 |
| `patches/check_vllm_series.sh` | own sed parse + SKIP + GNU-patch loop | names from `--list` (`:40-41`); pass 1 delegates the apply loop to `apply.sh` (`:48`), keeping its "the tree changed" guard (`:54-57`) and the offset tally by counting `apply.sh` output lines (`:52-53`); pass 2 (the five contractual DFlash patches under `git apply`, `:84-113`) is untouched |
| `docs/install.md:113-120` | drifted prose loop | the single command: `bash patches/apply.sh "$SP"` (`:114`; the page's variable is `SP`, `:113`; this row said `"$VP"`, python-314.md's name, at 2522ef9) — `--fuzz 0` now inherited by construction; the backup flag #231 added is kept |
| `docs/python-314.md:67-71` (cited `:68-71`) | drifted prose loop | the same single command (`"$VP"`, `:75`), plus the owner note from 3.4 (the banner, `:8-13`). This row said the `VP=` line goes too; it stays (`:74`), because the command needs it |
| `kvarn/install.sh` | own `apply_kvarn` (`:16-24`) + marker heredoc (`:38-71`; cited `:44-71` at 2522ef9, which omitted its explanatory comment `:38-43`) | overlay copy (`:12`) and the registration probe (`:23-32`, which checks live behavior and stays) keep their identity; the applies become `patches/apply.sh --kvarn "$SP"` (`:21`); the marker heredoc is deleted, because `--kvarn`'s exact reverse check is the completeness check. `verify.sh:113-127` uses the same exact check instead of `_check_applied.py`, for four patches since #262. (Was: `--forward --kvarn` plus `_check_applied.py`; corrected 2026-09-30.) |

The kvarn row deserves the reasoning spelled out: the marker counter guards
3 of 27 hunks on a rerun (1.3). The first version of this plan replaced it
with `_check_applied.py`, but that check passes a tree with one v2-runner hunk
missing. The exact reverse dry-run already guarded `kvarn-0.30.0` in
`verify.sh:117` at d5e2a01 (`:110` at e371b42), and it works for the other
patches as well, so PR C uses it everywhere. This still deletes the weaker
implementation and adds none.

### 2.3 Deletions (step 3.2)

All done in e1459c7. The citations are d5e2a01's.

- `patches/dflash2-backport.patch` and its five special cases
  (`series:23`, `Dockerfile:34`, `check_vllm_series.sh:37`,
  `verify.sh:101-104` (was `:82-85` at 2522ef9), `docs/install.md:117`).
- `kvarn/kvarn-0.27.1.patch` and `kvarn/kvarn-v2-runner.patch`.
- `PATCHES.md:23`: the row moves into the "Retired" prose (`PATCHES.md:71-73`;
  it landed at `:73-76` at e371b42),
  following the precedent set there. The `:12` prose mention is reworded to
  drop "except the retired dflash2-backport" — after deletion every row is a
  fork export again. (At e371b42 every row is a fork export, but `:12` names
  only `cpuchip/vllm`, and the branches and tags of two exports exist only on
  TyroneNel/vllm. See the top status block.)

### 2.4 Reference fixes (ride along with 3.2)

All done in e1459c7. Each item gives the e371b42 line.

- `README.md:122`: drop the hardcoded count — "The vLLM patch series
  (`patches/`, one line each in PATCHES.md)". A generated count belongs to
  candidate 6; a correct-by-construction sentence needs no maintenance.
  (`README.md:121-122` at e371b42.)
- `patches/series:3-4`: name the real consumers — "patches/apply.sh (which the
  Dockerfile, docs/install.md and docs/python-314.md call), verify.sh and
  patches/check_vllm_series.sh (both read it via apply.sh --list)".
  (`patches/series:3-6` at e371b42.)
- `patches/check_vllm_series.sh:8-9`: same fix. (`:8-10` at e371b42; it
  leaves out `docs/python-314.md`.)
- `PATCHES.md:83`: `docs/MR-DRAFT.md` was deleted in #131 (1.4.6); point at
  `docs/gotchas.md` alone (this said "or recreate the file" before the
  deletion was traced) — the wording choice is flagged to the owner (3.4).
  (`PATCHES.md:89-90` at e371b42 points at `docs/gotchas.md` alone.)

### 2.5 What deliberately does not change

- **Pass 2's contractual DFlash list** (`check_vllm_series.sh:89-95` at
  e371b42; `:83-89` at d5e2a01) — the
  GNU-patch/git-apply split is documented in the file and load-bearing.
- **verify.sh's ladder** — it answers "what state is this tree in?", a
  different question from "apply the series", and its independence is a
  feature: it is the skeptical second implementation.
- **The fork branch as source of truth** — `apply.sh` changes how patches are
  consumed, not how they are produced; `scripts/export-patch.sh` is untouched
  (no commit in `d5e2a01..e371b42` touches it; its `:4` still cites the
  missing `docs/fork-workflow.md`).
- **The historical prose** in `docs/docker.md:18` (was `:9`) and
  `docs/optimizations.md:214-215` (cited `:213`)
  mentioning the backport — it describes history and stays true. (Same lines
  at e371b42.)
## 3. Rollout

Three PRs, in order, each independently revertable. The ordering is chosen so
the risky-looking change (deletions) lands only after the new door is proven
by CI, and so no PR ever leaves a special case pointing at a deleted file.

### 3.1 PR A — "one door" (pure addition + rewiring, no behavior change)

Add `patches/apply.sh`; switch the Dockerfile, `verify.sh`'s list source,
`check_vllm_series.sh`'s list source and pass-1 loop, and the two docs pages
to it. The `dflash2-backport` skip logic moves *into* apply.sh temporarily
(one documented arm), so this PR deletes nothing and changes no applied
result — the tree it produces is bit-identical to d5e2a01's. (Status at
d5e2a01: not started — no `patches/apply.sh` exists. #231 already converged
the five apply sites on `--no-backup-if-mismatch`, so PR A's only
apply-policy change to the prose loops is adding `--fuzz 0`.) **Opened
2026-09-30 as #242**: old loop vs `apply.sh` trees byte-identical on v0.30.0;
`verify.sh --install` PASS/WARN/FAIL lines identical to main's. **Merged
2026-09-30 in e1459c7**, squashed with #243 and #244.

Acceptance: `patch-integrity` job green; image build green (its
`verify.sh --install` inside the build exercises the new list source);
`git diff` of a tree patched the old way vs the new way is empty.

### 3.2 PR B — "retire the retired" (deletions + reference fixes)

Delete the backport patch and the two 0.27.1 KVarN files; remove the five
special cases (including the temporary arm in apply.sh); move the PATCHES.md
row into the retired prose; fix the README count, the series header, the
check-script comment. After this PR: 44 patch files, 44 series lines, zero
per-patch exceptions anywhere. **Opened 2026-09-30 as #243 (stacked on #242).
Merged in e1459c7.** #233 landed first, so the merge had 45 patch files and 45
series lines. #263 added one more: 46 = 46 at e371b42, with zero per-patch
exceptions.

Acceptance: both workflows green; `git grep -l dflash2-backport -- ':!docs/*-remediation.md' ':!docs/architecture-review-*.html'`
finds only `PATCHES.md` (retired prose) and the historical notes in
`docs/docker.md` and `docs/optimizations.md`. (At 2522ef9 this read "`grep -r
dflash2-backport` finds only …", which cannot pass: the plan documents
themselves name the patch, and a working tree's untracked `bench/results/*/boot.log`
captures carry verify.sh's old "retired" line — 38 files in this checkout.)

### 3.3 PR C — "KVarN under the fast gate" (coverage)

`check_vllm_series.sh` pass 1 calls `apply.sh --kvarn`, so the
`patch-integrity` job covers 47 patch files (44 series + 3 KVarN) in its
sub-minute run (14–46 s wall, `gh run list`, 2026-09-28; this said "~2
minutes") instead of leaving them to the image build (4–12 min in CI; the
"~20-minute" figure here was the local build, `docker-image.yml:2`).
`kvarn/install.sh` switches its applies to `apply.sh --kvarn` (exact reverse
check, then strict apply); the marker heredoc is deleted, and `verify.sh`'s
two `_check_applied.py` KVarN rungs become exact reverse checks. **Opened
2026-09-30 as #244 (stacked on #243)**; gate log: `3 KVarN patches applied
after the series; 47 in total`. **Merged in e1459c7.** The run on e1459c7
(36754957337) logs `3 KVarN patches applied after the series; 48 in total`
(45 series patches, with #233). The run on e371b42 (37189534383) logs `4 KVarN
patches applied after the series; 50 in total`.

Acceptance: `patch-integrity` green with the KVarN trio in its log;
a manual `bash kvarn/install.sh` re-run is a no-op and exits 0; a
hand-broken KVarN hunk fails both the fast gate and install.sh by name.
(At e371b42 the gate is green with all four KVarN patches. Since #262 it also
runs `apply.sh --kvarn` a second time and requires "already applied" for all
four, `check_vllm_series.sh:71-82`. The manual `kvarn/install.sh` rerun and
the hand-broken hunk were not re-run at this pass [INFERENCE: not run].)

### 3.4 Flagged to the owner (not decided here)

- **`docs/python-314.md`'s premise.** The abi3 reasoning still holds for
  0.27.1, but the 0.30.0 series cannot apply to 0.27.1. Either re-verify the
  page against the current pin or banner it as a historical reproduction.
  Partly answered 2026-09-29: PyPI's `vllm/0.30.0` JSON lists only
  `cp38-abi3` manylinux wheels (x86_64, aarch64) with `requires_python`
  `<3.15,>=3.10` — the same abi3 shape the page relies on for 0.27.1. That is
  wheel metadata only; whether the 0.30.0 torch/FlashInfer set resolves and
  runs on 3.14 is not checked here. PR A swaps in the single command
  either way; the page's *claims* need a human run. **Still open at
  e371b42.** e1459c7 took the banner option (`docs/python-314.md:8-13`) and
  swapped in the single command (`:75`). The banner still says nobody has
  re-run the page on 0.30.0. Under it, `:20` ("applies unmodified"), `:49-55`
  ("fifteen") and `:71` (`vllm==0.27.1`) stay.
- **`PATCHES.md:83`'s dangling `docs/MR-DRAFT.md`.** Resolved as far as
  history goes: the file was deliberately deleted in #131 (1.4.6), so
  "recreate/commit it" is off the table unless the owner wants the PR-body
  source back; the remaining choice is pointing at `docs/gotchas.md` alone or
  moving the two descriptions into the patch preambles. **Closed at
  e371b42:** e1459c7 points at `docs/gotchas.md` alone (`PATCHES.md:89-90`).
  The same sentence says `dflash2-z-adaptive-emitted` and
  `offload-wsl2-devptr` "still carry raw `diff -ruN` headers"; `grep -c
  '^diff -ruN'` finds 0 in both, so that half is now false.

## 4. Test plan (no GPU required)

| test | how | proves |
|---|---|---|
| list parity | `bash patches/apply.sh --list` on main vs branch, `diff` | the rewired consumers read the same order |
| pristine apply | the `patch-integrity` job itself | the series still applies `--fuzz 0` against the pinned tag |
| negative control | in a scratch copy, break one context line of a mid-series patch | apply.sh fails, names the patch, exits 1 |
| idempotence | `apply.sh --kvarn` twice on one tree | second run prints `(already applied)` three times, exit 0 |
| agreement | drop a scratch `zzz.patch` into `patches/` | `--list` exits 2, names the file |
| ladder parity | `verify.sh --install` in the built image, before vs after the PRs; diff the PASS lines | the check semantics are unchanged |
| partial-apply detection | on a full tree, remove one hunk with `patch -R` (v2-runner `attention.py` hunks 3 and 4; kvarn-0.30.0 `kv_cache_interface.py` and `platforms/cuda.py` hunk 3); rerun `apply.sh --kvarn` and verify.sh's KVarN rungs | every case exits 1 by name and WARNs in verify.sh. Hunk 4 is the regression case: `_check_applied.py` and the marker heredoc both pass it |

At e371b42, CI runs two of these rows on every PR: pristine apply (pass 1)
and idempotence (`check_vllm_series.sh:71-82`, now four "already applied"
lines). The other rows were not re-run at this pass [INFERENCE: not run].

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| PR B deletes a file someone's 0.27.1 install still wants (the python-314 path) | low: the 0.30 series already cannot apply there (1.4.2) | the deletion is announced in the PR and the PATCHES.md retired prose; the files remain in git history; the fork branch regenerates them if ever needed |
| `apply.sh`'s per-patch output becomes a consumed contract (check script counts it) | medium | the contract is written down in the file's header (2.1); the check script consumes exit codes + `==` line count only |
| KVarN patches behave differently against a pristine *checkout* (CI) than against *site-packages* (install) | low: same files, same GNU patch mechanism; both are `-p1` against the package dir | PR C's own CI run demonstrates it before merge |
| A new patch lands between A and B and re-adds a special case | low | the agreement check (now in `--list`) fails loudly, so it cannot land silently |
| verify.sh's ladder regresses unnoticed | low | ladder-parity row in the test plan; the ladder's diff is three lines (list source), reviewable by eye |

## 6. Done when

All six pass at e371b42 (checked 2026-10-04; the evidence is in the top
status block).

1. `grep -rln "s/#.\*//" --include='*.sh' --include='Dockerfile' .` finds the
   series parse in exactly one file: `patches/apply.sh` (at d5e2a01 it
   listed three: `Dockerfile`, `patches/check_vllm_series.sh`, `verify.sh`; the
   two prose copies are `.md` and are covered by the docs rewrite in PR A).
   **PASS at e371b42:** only `./patches/apply.sh` (`:58-60`).
2. `git grep -l dflash2-backport -- ':!docs/*-remediation.md' ':!docs/architecture-review-*.html'`
   lists only `PATCHES.md`, `docs/docker.md` and `docs/optimizations.md` (the
   retired prose and the two historical notes). At d5e2a01 it listed nine
   files, the six extra being `Dockerfile`, `docs/install.md`,
   `patches/check_vllm_series.sh`, `patches/dflash2-backport.patch`,
   `patches/series` and `verify.sh`. (Was "`grep -r dflash2-backport .`", which
   also matches the plan docs and untracked bench logs — see 3.2.)
   **PASS at e371b42:** exactly those three files.
3. `ls patches/*.patch | wc -l` = `wc -l < patches/series` (cleaned) = 44
   once PR B's deletion lands (45 = 45 at d5e2a01), and `--list` agrees.
   **PASS at e371b42:** 46 = 46 (44 + #233 + #263), and `--list` exits 0.
4. The `patch-integrity` job log shows 47 patches applied (44 + 3 KVarN).
   **PASS at e371b42:** 50 (46 + 4 KVarN; run 37189534383).
5. README.md contains no hardcoded patch count. **PASS at e371b42**
   (`README.md:121-122`).
6. `kvarn/install.sh` contains no marker-count heredoc; its completeness
   check is `apply.sh --kvarn`'s exact reverse dry-run, and `verify.sh`
   checks all three KVarN patches the same way. **PASS at e371b42:**
   `kvarn/install.sh:21`; `verify.sh:110-127` checks all four (#262 added the
   fourth).

At that point "how the series is applied" has one answer, "what is applied"
has one list, and the answer to "is it healthy?" is the fast CI gate plus
verify.sh's ladder — two implementations, where the repo at d5e2a01 ran four.
At e371b42 that holds for the series. The KVarN list has no agreement check
yet (top status block).
