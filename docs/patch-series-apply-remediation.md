# One door into the patch series — remediation plan

Architecture review 2026-09-26, candidate 1 (Strong). The full report with the
other six candidates is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this document is the deep dive on that card: the current state measured line by
line, the design, and the rollout.

**Scope.** Everything that parses, applies, or re-lists `patches/series`; the
retired artifacts that force special cases into those consumers; the stale
references the duplication has already produced. **Not in scope:** the harness
verdict conventions (candidate 5), generating the PATCHES.md table (candidate
6), and any change to the verify.sh *checking ladder* — that ladder is state
diagnosis, a deliberately independent second look, and it stays.

## 1. Current state, precisely

### 1.1 The ordered series is parsed in five places

`patches/series` (45 entries, `series:15-59`) is the canonical order. The same
sed parse is re-encoded in three code consumers and two prose copies:

| # | place | form | guarded against drift? |
|---|-------|------|------------------------|
| 1 | `Dockerfile:31` | sed copy, in the build's apply loop | no |
| 2 | `verify.sh:55` | sed copy, feeds the check ladder | partially (dir↔series agreement, `:58-62`) |
| 3 | `patches/check_vllm_series.sh:45` | sed copy, feeds CI pass 1 | same partial check (`:48-52`) |
| 4 | `docs/install.md:106-113` | prose copy of the whole loop | **drifted** |
| 5 | `docs/python-314.md:68-71` | prose copy of the whole loop | **drifted** |

All three code copies are byte-identical:

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
| `verify.sh:77-79` | "retired" branch inside the check loop |
| `docs/install.md:109` | the same skip arm, in prose |
| `PATCHES.md:24` | a table row already marked `RETIRED` |
| `PATCHES.md:12`, `docs/docker.md:9`, `docs/optimizations.md:213` | historical prose (fine to keep — see 3.2) |

The same "kept for history" pattern holds ~97 KB of inert KVarN files:
`kvarn/kvarn-0.27.1.patch` and `kvarn/kvarn-v2-runner.patch`, referenced only by
`kvarn/README.md:35` ("the older ports, kept for the diff history"). The fork
branch is the declared source of truth (`PATCHES.md:12-18`), and this repo is a
git repository — the diff history is not at risk.

This contradicts the repo's own remove-on-retire precedent
(`PATCHES.md:70-72`: the 0.29.0 retirements were *removed from the tree*).

### 1.3 Four different answers to "is the series healthy?"

- **Dockerfile**: apply-only; health = the build succeeds (with
  `verify.sh --install` at the end).
- **check_vllm_series.sh**: pass 1 applies the whole series with GNU
  `patch --fuzz 0`; pass 2 re-checks the five contractual DFlash patches with
  `git apply`. The two-pass split is load-bearing (`:6-20`) and stays.
- **verify.sh**: a four-rung ladder per patch (`:75-86`): reverse dry-run →
  `_check_applied.py` content check → `Supersedes:` lookup → forward dry-run
  diagnosis. Each rung exists because the rung above has a named failure mode.
- **kvarn/install.sh**: greps `patch -N` output for `FAILED` (`:16-17`), plus a
  Python heredoc (`:42-65`) that counts `port(kvarn-v2)` markers per file as its
  "every hunk landed" proof.

The marker check has silently decayed. Measured 2026-09-26:
`kvarn/kvarn-v2-runner-0.29.0.patch` has **12 hunks across 7 files but only 1
`port(kvarn-v2)` marker** — the heredoc verifies one hunk and prints
"port complete". `kvarn/kvarn-0.29.0.patch` (13 hunks) has no markers at all.
Meanwhile `verify.sh:96-98` already checks the v2-runner patch properly, with
`patches/_check_applied.py` — the content checker whose 80%-per-file heuristic
was written after PR #43 taught that lesson. The strong checker exists; the
weak one runs at install time.
### 1.4 Drift already on the ground

Every duplication above has produced at least one live inconsistency. All
confirmed against the tree on 2026-09-26:

1. **`docs/install.md:106-113` drops `--fuzz 0`.** The prose loop applies with
   plain `patch -p1`. A hunk whose context moved lands by approximate anchor
   instead of failing by name — the exact failure mode the fuzz-0 rule
   (`check_vllm_series.sh:56-59`) was written to prevent.
2. **`docs/python-314.md` is stale as a whole.** The 0.27.1 pin there is
   deliberate (the abi3 wheel is why 3.14 works at all, `:9-11`), but
   "Every patch in `patches/` applies unmodified" (`:13`) and "all fifteen
   land" (`:42-45`) were written against the 0.27.1-era series. Today the
   series is 45 patches cut against 0.29.0; the doc's loop cannot produce a
   working install. Its loop also lacks both `--fuzz 0` and the backport skip.
3. **`README.md:119` says "38 files".** There are 45. The README split (#129)
   moved the install loop to `docs/install.md`, so…
4. **`patches/series:3-4` names "the README install loop"** as a consumer. No
   such loop exists anymore (`grep` finds no `patch -p1` in README.md).
5. **`patches/check_vllm_series.sh:8-9`** makes the same stale claim ("which
   the Dockerfile and the README use").
6. **`PATCHES.md:82`** sends the reader to `docs/MR-DRAFT.md`, which does not
   exist in the tree.

### 1.5 The fast CI gate never sees the KVarN patches

`patch-integrity.yml` runs `check_vllm_series.sh`, which covers the 44
non-retired series patches. The two KVarN patches — which carry the
`CTX=huge` correctness fixes — are exercised only by the ~20-minute image
build (`docker-image.yml`). Yet they are exported from the fork branch at a
position *after the whole series* (`kvarn/install.sh:13-14`), which is exactly
the point pass 1 reaches: the fast gate can cover them with no new machinery.

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
    `patch -p1 --fuzz 0 --no-backup-if-mismatch`, in series order. Stop at the
    first failure; the message names the patch. Per patch, print one line:
    `== <name>` (or `== <name> (N hunks at an offset)` — offsets are benign
    and reported, fuzz is refused; the policy text already written at
    check_vllm_series.sh:56-59 moves here with the code it describes).
    --forward   pass --forward so a re-run over an applied tree is a no-op
                (kvarn/install.sh's documented idempotence)
    --kvarn     after the series, continue with kvarn/kvarn-0.29.0.patch and
                kvarn/kvarn-v2-runner-0.29.0.patch, in that order — they are
                exported to apply at exactly this point (install.sh:13-14,20)

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
| `verify.sh:52-62` | own sed parse + agreement check | `SERIES` from `patches/apply.sh --list`; the four-rung ladder (`:75-86`) is untouched; the `:77-79` backport arm dies in step 3.2 |
| `patches/check_vllm_series.sh` | own sed parse + SKIP + GNU-patch loop | names from `--list`; pass 1 delegates the apply loop to `apply.sh` (keeping its "the tree changed" guard at `:72-75` and the offset tally by counting `apply.sh` output lines); pass 2 (the five contractual DFlash patches under `git apply`, `:78-107`) is untouched |
| `docs/install.md:106-113` | drifted prose loop | the single command: `bash patches/apply.sh "$VP"` (with `--fuzz 0` now inherited by construction) |
| `docs/python-314.md:68-71` | drifted prose loop | the same single command, plus the owner note from 3.4 |
| `kvarn/install.sh` | own `apply_kvarn` (`:16-21`) + marker heredoc (`:42-65`) | overlay copy (`:12`) and the registration probe (`:23-33`, which checks live behavior and stays) keep their identity; the two applies become `patches/apply.sh --forward --kvarn "$SP"`; the marker heredoc is replaced by `python patches/_check_applied.py kvarn/kvarn-v2-runner-0.29.0.patch "$SP"` — the check `verify.sh:96` already runs against the same patch |

The kvarn row deserves the reasoning spelled out: the marker counter was
verifying 1 of 12 hunks (1.3). `_check_applied.py` is the stronger,
already-battle-tested mechanism (its per-file threshold exists because of the
PR #43 failure mode), and switching install.sh to it deletes a second,
weaker implementation rather than adding a new one.

### 2.3 Deletions (step 3.2)

- `patches/dflash2-backport.patch` and its five special cases
  (`series:23`, `Dockerfile:34`, `check_vllm_series.sh:37`,
  `verify.sh:77-79`, `docs/install.md:109`).
- `kvarn/kvarn-0.27.1.patch` and `kvarn/kvarn-v2-runner.patch`.
- `PATCHES.md:24`: the row moves into the "Retired" prose (`PATCHES.md:70-72`),
  following the precedent set there. The `:12` prose mention is reworded to
  drop "except the retired dflash2-backport" — after deletion every row is a
  fork export again.

### 2.4 Reference fixes (ride along with 3.2)

- `README.md:119`: drop the hardcoded count — "The vLLM patch series
  (`patches/`, one line each in PATCHES.md)". A generated count belongs to
  candidate 6; a correct-by-construction sentence needs no maintenance.
- `patches/series:3-4`: name the real consumers — "patches/apply.sh (which the
  Dockerfile, docs/install.md and docs/python-314.md call), verify.sh and
  patches/check_vllm_series.sh (both read it via apply.sh --list)".
- `patches/check_vllm_series.sh:8-9`: same fix.
- `PATCHES.md:82`: `docs/MR-DRAFT.md` does not exist; either point at
  `docs/gotchas.md` alone or recreate the file — flagged to the owner (3.4).

### 2.5 What deliberately does not change

- **Pass 2's contractual DFlash list** (`check_vllm_series.sh:83-89`) — the
  GNU-patch/git-apply split is documented in the file and load-bearing.
- **verify.sh's ladder** — it answers "what state is this tree in?", a
  different question from "apply the series", and its independence is a
  feature: it is the skeptical second implementation.
- **The fork branch as source of truth** — `apply.sh` changes how patches are
  consumed, not how they are produced; `scripts/export-patch.sh` is untouched.
- **The historical prose** in `docs/docker.md:9` and `docs/optimizations.md:213`
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
result — the tree it produces is bit-identical to today's.

Acceptance: `patch-integrity` job green; image build green (its
`verify.sh --install` inside the build exercises the new list source);
`git diff` of a tree patched the old way vs the new way is empty.

### 3.2 PR B — "retire the retired" (deletions + reference fixes)

Delete the backport patch and the two 0.27.1 KVarN files; remove the five
special cases (including the temporary arm in apply.sh); move the PATCHES.md
row into the retired prose; fix the README count, the series header, the
check-script comment. After this PR: 44 patch files, 44 series lines, zero
per-patch exceptions anywhere.

Acceptance: both workflows green; `grep -r dflash2-backport` finds only
PATCHES.md's retired prose and the historical docs notes.

### 3.3 PR C — "KVarN under the fast gate" (coverage)

`check_vllm_series.sh` pass 1 calls `apply.sh --kvarn`, so the
`patch-integrity` job covers 46 patch files (44 series + 2 KVarN) in its
~2 minutes instead of leaving them to the ~20-minute image build.
`kvarn/install.sh` switches its applies to `apply.sh --forward --kvarn` and
its completeness check to `_check_applied.py`; the marker heredoc is deleted.

Acceptance: `patch-integrity` green with the KVarN pair in its log;
a manual `bash kvarn/install.sh` re-run is a no-op and exits 0; a
hand-broken KVarN hunk fails both the fast gate and install.sh by name.

### 3.4 Flagged to the owner (not decided here)

- **`docs/python-314.md`'s premise.** The abi3 reasoning still holds for
  0.27.1, but the 0.29.0 series cannot apply to 0.27.1. Either re-verify the
  page against the current pin (does 0.29.0 publish a 3.14-compatible wheel?)
  or banner it as a historical reproduction. PR A swaps in the single command
  either way; the page's *claims* need a human run.
- **`PATCHES.md:82`'s dangling `docs/MR-DRAFT.md`.** Point at gotchas.md alone,
  or the file exists somewhere and should be committed.

## 4. Test plan (no GPU required)

| test | how | proves |
|---|---|---|
| list parity | `bash patches/apply.sh --list` on main vs branch, `diff` | the rewired consumers read the same order |
| pristine apply | the `patch-integrity` job itself | the series still applies `--fuzz 0` against the pinned tag |
| negative control | in a scratch copy, break one context line of a mid-series patch | apply.sh fails, names the patch, exits 1 |
| idempotence | `apply.sh --forward --kvarn` twice on one tree | second run is a no-op, exit 0 |
| agreement | drop a scratch `zzz.patch` into `patches/` | `--list` exits 2, names the file |
| ladder parity | `verify.sh --install` in the built image, before vs after the PRs; diff the PASS lines | the check semantics are unchanged |
| content-check swap | on a KVarN-installed tree, `_check_applied.py kvarn/kvarn-v2-runner-0.29.0.patch` passes; on a tree missing one hunk, it fails | the marker counter's replacement actually detects partial application (the 1-of-12 decay, 1.3) |

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| PR B deletes a file someone's 0.27.1 install still wants (the python-314 path) | low: the 0.29 series already cannot apply there (1.4.2) | the deletion is announced in the PR and the PATCHES.md retired prose; the files remain in git history; the fork branch regenerates them if ever needed |
| `apply.sh`'s per-patch output becomes a consumed contract (check script counts it) | medium | the contract is written down in the file's header (2.1); the check script consumes exit codes + `==` line count only |
| KVarN patches behave differently against a pristine *checkout* (CI) than against *site-packages* (install) | low: same files, same GNU patch mechanism; both are `-p1` against the package dir | PR C's own CI run demonstrates it before merge |
| A new patch lands between A and B and re-adds a special case | low | the agreement check (now in `--list`) fails loudly, so it cannot land silently |
| verify.sh's ladder regresses unnoticed | low | ladder-parity row in the test plan; the ladder's diff is three lines (list source), reviewable by eye |

## 6. Done when

1. `grep -rn "s/#.\*//" --include='*.sh' --include='Dockerfile' .` finds the
   series parse in exactly one file: `patches/apply.sh`.
2. `grep -r dflash2-backport .` finds only the PATCHES.md retired prose and
   the two historical docs notes.
3. `ls patches/*.patch | wc -l` = `wc -l < patches/series` (cleaned) = 44,
   and `--list` agrees.
4. The `patch-integrity` job log shows 46 patches applied (44 + 2 KVarN).
5. README.md contains no hardcoded patch count.
6. `kvarn/install.sh` contains no marker-count heredoc; its completeness
   check is `_check_applied.py`.

At that point "how the series is applied" has one answer, "what is applied"
has one list, and the answer to "is it healthy?" is the fast CI gate plus
verify.sh's ladder — two implementations, where the repo today runs four.
