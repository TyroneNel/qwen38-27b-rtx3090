# One door into the patch series — remediation plan

Architecture review 2026-09-26, candidate 1 (Strong). The full report with the
other six candidates is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this document is the deep dive on that card: the current state measured line by
line, the design, and the rollout.

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

## 1. Current state, precisely

### 1.1 The ordered series is parsed in five places

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

### 1.2 One retired patch has seven touch points

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
hunks and prints "port complete" (`:70`). Meanwhile `verify.sh:120-126` (was
`:101-107` at 2522ef9) already checks the
v2-runner and recycled-pages patches properly, with
`patches/_check_applied.py` — the content checker whose 80%-per-file heuristic
was written after PR #43 taught that lesson. The strong checker exists; the
weak one runs at install time.

### 1.4 Drift already on the ground

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
   land" (`:42-45`) were written against the 0.27.1-era series. Today the
   series is 45 patches cut against 0.30.0; the doc's loop cannot produce a
   working install. Its loop (`:67-71`) also lacks both `--fuzz 0` and the
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

### 1.5 The fast CI gate never sees the KVarN patches

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

## 2. The design

One new executable, `patches/apply.sh`, owns the series parse and the apply
policy. Every consumer either calls it or shows its one command line.

### 2.1 Interface

```text
patches/apply.sh --list
    Print the ordered patch basenames, one per line. Before printing, enforce
    that the list and the patches/ directory agree exactly; on disagreement,
    name the offenders on stderr and exit 2. (Today that check runs only in
    verify.sh and check_vllm_series.sh; moving it here gives it to every
    consumer, including the Docker build, for free.)

patches/apply.sh [--forward] [--kvarn] DIR
    Apply the whole series to DIR (the installed vllm package directory) with
    `patch -p1 --fuzz 0 --no-backup-if-mismatch`, in series order. (Since
    #231 all five apply sites already pass --no-backup-if-mismatch; --fuzz 0
    is still missing from the two prose loops.) Stop at the
    first failure; the message names the patch. Per patch, print one line:
    `== <name>` (or `== <name> (N hunks at an offset)` — offsets are benign
    and reported, fuzz is refused; the policy text already written at
    check_vllm_series.sh:56-59 moves here with the code it describes).
    --forward   pass --forward so a re-run over an applied tree is a no-op
                (kvarn/install.sh's documented idempotence)
    --kvarn     after the series, continue with kvarn/kvarn-0.30.0.patch,
                kvarn/kvarn-v2-runner-0.30.0.patch and
                kvarn/kvarn-recycled-pages-0.30.0.patch, in that order — they
                are exported to apply at exactly this point (install.sh:13-24)

Exit codes: 0 success · 1 an apply failed (patch named) · 2 usage error or
series/directory disagreement.
```

The sed parse from 1.1 lives in this file, once. `patches/series` stays the
single source of truth for *order*; `apply.sh` becomes the single source of
truth for *how the order is read and applied*.
### 2.2 What each consumer becomes

| consumer | before | after |
|---|---|---|
| `Dockerfile:28-39` | sed loop + skip arm | `RUN bash patches/apply.sh "$SP" && bash kvarn/install.sh && bash verify.sh --install` |
| `verify.sh:71-87` (was `:52-68` at 2522ef9) | own sed parse + agreement check | `SERIES` from `patches/apply.sh --list`; the four-rung ladder (`:99-110`) is untouched; the `:101-104` backport arm dies in step 3.2 |
| `patches/check_vllm_series.sh` | own sed parse + SKIP + GNU-patch loop | names from `--list`; pass 1 delegates the apply loop to `apply.sh` (keeping its "the tree changed" guard at `:72-75` and the offset tally by counting `apply.sh` output lines); pass 2 (the five contractual DFlash patches under `git apply`, `:78-110`) is untouched |
| `docs/install.md:113-120` | drifted prose loop | the single command: `bash patches/apply.sh "$SP"` (the page's variable is `SP`, `:113`; this row said `"$VP"`, python-314.md's name, at 2522ef9) — `--fuzz 0` now inherited by construction; the backup flag #231 added is kept |
| `docs/python-314.md:67-71` (cited `:68-71`; the `VP=` line `:67` goes too) | drifted prose loop | the same single command (`"$VP"`), plus the owner note from 3.4 |
| `kvarn/install.sh` | own `apply_kvarn` (`:16-24`) + marker heredoc (`:38-71`; cited `:44-71` at 2522ef9, which omitted its explanatory comment `:38-43`) | overlay copy (`:12`) and the registration probe (`:26-36`, which checks live behavior and stays) keep their identity; the three applies become `patches/apply.sh --forward --kvarn "$SP"`; the marker heredoc is replaced by `python patches/_check_applied.py <kvarn patch> "$SP"` run for both marker-carrying patches — the check `verify.sh:120-126` (was `:101-107`) already runs against both |

The kvarn row deserves the reasoning spelled out: the marker counter was
verifying 3 of 27 hunks (1.3). `_check_applied.py` is the stronger,
already-battle-tested mechanism (its per-file threshold exists because of the
PR #43 failure mode), and switching install.sh to it deletes a second,
weaker implementation rather than adding a new one.

### 2.3 Deletions (step 3.2)

- `patches/dflash2-backport.patch` and its five special cases
  (`series:23`, `Dockerfile:34`, `check_vllm_series.sh:37`,
  `verify.sh:101-104` (was `:82-85` at 2522ef9), `docs/install.md:117`).
- `kvarn/kvarn-0.27.1.patch` and `kvarn/kvarn-v2-runner.patch`.
- `PATCHES.md:23`: the row moves into the "Retired" prose (`PATCHES.md:71-73`),
  following the precedent set there. The `:12` prose mention is reworded to
  drop "except the retired dflash2-backport" — after deletion every row is a
  fork export again.

### 2.4 Reference fixes (ride along with 3.2)

- `README.md:122`: drop the hardcoded count — "The vLLM patch series
  (`patches/`, one line each in PATCHES.md)". A generated count belongs to
  candidate 6; a correct-by-construction sentence needs no maintenance.
- `patches/series:3-4`: name the real consumers — "patches/apply.sh (which the
  Dockerfile, docs/install.md and docs/python-314.md call), verify.sh and
  patches/check_vllm_series.sh (both read it via apply.sh --list)".
- `patches/check_vllm_series.sh:8-9`: same fix.
- `PATCHES.md:83`: `docs/MR-DRAFT.md` was deleted in #131 (1.4.6); point at
  `docs/gotchas.md` alone (this said "or recreate the file" before the
  deletion was traced) — the wording choice is flagged to the owner (3.4).

### 2.5 What deliberately does not change

- **Pass 2's contractual DFlash list** (`check_vllm_series.sh:83-89`) — the
  GNU-patch/git-apply split is documented in the file and load-bearing.
- **verify.sh's ladder** — it answers "what state is this tree in?", a
  different question from "apply the series", and its independence is a
  feature: it is the skeptical second implementation.
- **The fork branch as source of truth** — `apply.sh` changes how patches are
  consumed, not how they are produced; `scripts/export-patch.sh` is untouched.
- **The historical prose** in `docs/docker.md:18` (was `:9`) and
  `docs/optimizations.md:214-215` (cited `:213`)
  mentioning the backport — it describes history and stays true.
## 3. Rollout

Three PRs, in order, each independently revertable. The ordering is chosen so
the risky-looking change (deletions) lands only after the new door is proven
by CI, and so no PR ever leaves a special case pointing at a deleted file.

### 3.1 PR A — "one door" (pure addition + rewiring, no behavior change)

Add `patches/apply.sh`; switch the Dockerfile, `verify.sh`'s list source,
`check_vllm_series.sh`'s list source and pass-1 loop, and the two docs pages
to it. The `dflash2-backport` skip logic moves *into* apply.sh temporarily
(one documented arm), so this PR deletes nothing and changes no applied
result — the tree it produces is bit-identical to today's. (Status at
d5e2a01: not started — no `patches/apply.sh` exists. #231 already converged
the five apply sites on `--no-backup-if-mismatch`, so PR A's only
apply-policy change to the prose loops is adding `--fuzz 0`.)

Acceptance: `patch-integrity` job green; image build green (its
`verify.sh --install` inside the build exercises the new list source);
`git diff` of a tree patched the old way vs the new way is empty.

### 3.2 PR B — "retire the retired" (deletions + reference fixes)

Delete the backport patch and the two 0.27.1 KVarN files; remove the five
special cases (including the temporary arm in apply.sh); move the PATCHES.md
row into the retired prose; fix the README count, the series header, the
check-script comment. After this PR: 44 patch files, 44 series lines, zero
per-patch exceptions anywhere.

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
`kvarn/install.sh` switches its applies to `apply.sh --forward --kvarn` and
its completeness check to `_check_applied.py`; the marker heredoc is deleted.

Acceptance: `patch-integrity` green with the KVarN trio in its log;
a manual `bash kvarn/install.sh` re-run is a no-op and exits 0; a
hand-broken KVarN hunk fails both the fast gate and install.sh by name.

### 3.4 Flagged to the owner (not decided here)

- **`docs/python-314.md`'s premise.** The abi3 reasoning still holds for
  0.27.1, but the 0.30.0 series cannot apply to 0.27.1. Either re-verify the
  page against the current pin or banner it as a historical reproduction.
  Partly answered 2026-09-29: PyPI's `vllm/0.30.0` JSON lists only
  `cp38-abi3` manylinux wheels (x86_64, aarch64) with `requires_python`
  `<3.15,>=3.10` — the same abi3 shape the page relies on for 0.27.1. That is
  wheel metadata only; whether the 0.30.0 torch/FlashInfer set resolves and
  runs on 3.14 is not checked here. PR A swaps in the single command
  either way; the page's *claims* need a human run.
- **`PATCHES.md:83`'s dangling `docs/MR-DRAFT.md`.** Resolved as far as
  history goes: the file was deliberately deleted in #131 (1.4.6), so
  "recreate/commit it" is off the table unless the owner wants the PR-body
  source back; the remaining choice is pointing at `docs/gotchas.md` alone or
  moving the two descriptions into the patch preambles.

## 4. Test plan (no GPU required)

| test | how | proves |
|---|---|---|
| list parity | `bash patches/apply.sh --list` on main vs branch, `diff` | the rewired consumers read the same order |
| pristine apply | the `patch-integrity` job itself | the series still applies `--fuzz 0` against the pinned tag |
| negative control | in a scratch copy, break one context line of a mid-series patch | apply.sh fails, names the patch, exits 1 |
| idempotence | `apply.sh --forward --kvarn` twice on one tree | second run is a no-op, exit 0 |
| agreement | drop a scratch `zzz.patch` into `patches/` | `--list` exits 2, names the file |
| ladder parity | `verify.sh --install` in the built image, before vs after the PRs; diff the PASS lines | the check semantics are unchanged |
| content-check swap | on a KVarN-installed tree, `_check_applied.py kvarn/kvarn-v2-runner-0.30.0.patch` passes; on a tree missing one hunk, it fails | the marker counter's replacement actually detects partial application (the 3-of-27 decay, 1.3) |

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| PR B deletes a file someone's 0.27.1 install still wants (the python-314 path) | low: the 0.30 series already cannot apply there (1.4.2) | the deletion is announced in the PR and the PATCHES.md retired prose; the files remain in git history; the fork branch regenerates them if ever needed |
| `apply.sh`'s per-patch output becomes a consumed contract (check script counts it) | medium | the contract is written down in the file's header (2.1); the check script consumes exit codes + `==` line count only |
| KVarN patches behave differently against a pristine *checkout* (CI) than against *site-packages* (install) | low: same files, same GNU patch mechanism; both are `-p1` against the package dir | PR C's own CI run demonstrates it before merge |
| A new patch lands between A and B and re-adds a special case | low | the agreement check (now in `--list`) fails loudly, so it cannot land silently |
| verify.sh's ladder regresses unnoticed | low | ladder-parity row in the test plan; the ladder's diff is three lines (list source), reviewable by eye |

## 6. Done when

1. `grep -rln "s/#.\*//" --include='*.sh' --include='Dockerfile' .` finds the
   series parse in exactly one file: `patches/apply.sh` (today, at d5e2a01, it
   lists three: `Dockerfile`, `patches/check_vllm_series.sh`, `verify.sh`; the
   two prose copies are `.md` and are covered by the docs rewrite in PR A).
2. `git grep -l dflash2-backport -- ':!docs/*-remediation.md' ':!docs/architecture-review-*.html'`
   lists only `PATCHES.md`, `docs/docker.md` and `docs/optimizations.md` (the
   retired prose and the two historical notes). Today it lists nine files,
   the six extra being `Dockerfile`, `docs/install.md`,
   `patches/check_vllm_series.sh`, `patches/dflash2-backport.patch`,
   `patches/series` and `verify.sh`. (Was "`grep -r dflash2-backport .`", which
   also matches the plan docs and untracked bench logs — see 3.2.)
3. `ls patches/*.patch | wc -l` = `wc -l < patches/series` (cleaned) = 44
   once PR B's deletion lands (45 = 45 today), and `--list` agrees.
4. The `patch-integrity` job log shows 47 patches applied (44 + 3 KVarN).
5. README.md contains no hardcoded patch count.
6. `kvarn/install.sh` contains no marker-count heredoc; its completeness
   check is `_check_applied.py`.

At that point "how the series is applied" has one answer, "what is applied"
has one list, and the answer to "is it healthy?" is the fast CI gate plus
verify.sh's ladder — two implementations, where the repo today runs four.
