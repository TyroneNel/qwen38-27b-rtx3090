# Generate the patch index from the patches — remediation plan

Architecture review 2026-09-26, candidate 6 (Worth exploring). The full
report is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree
on 2026-09-26.

**Re-verified 2026-09-29 against upstream/main @ d5e2a01 (vLLM 0.30.0).**
Seven commits since 2522ef9; only #234 (e355f9f) and #219 (8cf642e) touch
anything this plan cites.

- **Unchanged (recounted):** `PATCHES.md` is still 83 lines, 50 pipe-rows =
  2 header + 48 data, all 45 `patches/series` names + the 3
  `kvarn/*0.30.0*.patch` rows, no duplicates (`comm -3` of the row names
  against series + kvarn is empty). README still says "38 files" at `:122`
  (45 in `patches/`); `patches/series:3-4` and `check_vllm_series.sh:8-9`
  still name the README; `PATCHES.md:82-83` (the `diff -ruN` claim and
  `docs/MR-DRAFT.md`) unchanged; zero `PATCHES` matches in `.github/`;
  44 of 45 patch files carry the export marker; the one `Supersedes:`
  producer is still `spec-decode-scratch-within-budget.patch:49`.
- **#234 (e355f9f)** made `marlin-repack-staged-sm80` opt-in everywhere and
  edited its `PATCHES.md:37` row by hand (the "what" cell gained "opt-in with
  `VLLM_MARLIN_REPACK_STAGED=1`"), while re-exporting that patch plus
  `marlin-int8-asym-zp`, `speed-knobs-envs` and `kvarn/kvarn-0.30.0` — the
  same fact edited in two homes by hand, which is exactly the duplication
  this plan removes. It is also a ninth `PATCHES.md` touch in the 40-commit
  window (see below).
- **Newly introduced (#234):** the four re-exported files name fork hashes
  (`f11865100`, `2ba5c626e`, `ec0ae4a23`, `6a60dbc2c`) that are reachable only
  from the fork tag `qwen38/0.30-cut8` (`gh api
  repos/cpuchip/vllm/compare/<tag>...<hash>` → `behind` for cut8, `diverged`
  for cut5), but `PATCHES.md:12` still names `qwen38/0.30-cut5` as the export
  point. 40 of the 47 export-marker hashes (44 patches + 3 kvarn) are
  reachable from cut5 (44 of 47 before #234); `spec-attn-smem-fit`
  (`ede631297e`) only from cut6/cut7; `bench-sse-keepalive` (`757723b`) and
  `kvarn-recycled-pages-0.30.0` (`40ab8e0`) resolve in no tag —
  `gh api repos/cpuchip/vllm/commits/<hash>` returns 422 "No commit found"
  (those three predate 2522ef9; the previous pass did not check). The
  "tag defuses the rewrite risk" claim below and in §5 is corrected.
- **Line shifts (#219, 8cf642e — `verify.sh --wait` added 19 header lines
  and 15 live-section lines):** the `Supersedes:` consumer `verify.sh:72` →
  `:91` (the `superseded_by()` function is `:88`, its caller `:107`);
  `graded()` `:342` → `:376`, rationale `:332-341` → `:369-375`, call sites
  `:408/:414/:420` → `:442/:448/:454`. #219 also added `scripts/hq-doctor.sh`,
  so `scripts/` is no longer "only `export-patch.sh`" (still no
  `pin-bump.py`).
- **Corrected count:** "8 touches in the last 40 commits" was wrong at
  2522ef9 too — `git log -40 --format= --name-only 2522ef9 | sort | uniq -c`
  gives 9. At d5e2a01 the same window (`git log -40 --format= --name-only
  upstream/main`, commits f98de95..d5e2a01) still gives **9** (e355f9f
  entered, 13b30ea left); runners-up at 7: `verify.sh`, `patches/series`,
  `single-user/start_qwen.sh`, `docs/gotchas.md`.
- **Wording fixed:** "zero `diff -ruN` lines anywhere in the tree" →
  zero in any patch file (the string occurs only in `PATCHES.md:82` itself
  and in the review/plan docs quoting it).
- **Newly surfaced (present at 2522ef9, not caught):** `export-patch.sh:4`
  cites `docs/fork-workflow.md` for "rule 2", which does not exist (the
  rule's text is `PATCHES.md:14-15`); 2.2's "a hand-edited patch already
  fails `check_vllm_series.sh`" holds only for hunk headers in the five
  `git apply` DFlash patches (`check_vllm_series.sh:13-15`); 3.2's and 6's
  bare `grep -rn 'MR-DRAFT' .` acceptance could never pass (the review and
  this plan quote it). All three corrected in place; 3.2 and 6 now also
  require `PATCHES.md:12`'s tag to carry every marker hash.

**Earlier re-verification: 2026-09-28 against upstream/main @ 2522ef9.**

- #189 (the 0.30 port) re-exported every patch and rewrote the fork prose:
  one branch, `qwen38/0.30`, export point tagged `qwen38/0.30-cut5`
  (`PATCHES.md:12`) — no per-row hashes anymore, and the tag exists so a
  branch rewrite cannot orphan the hashes the files name. The
  history-rewrite risk in §5 is upstream-defused, not just mitigated.
  *(2026-09-29: only partly true — the fork has since been re-cut to
  `qwen38/0.30-cut8` and #234 exported from it without moving `:12`'s tag;
  see above.)*
- The hand-cut exception is gone: `marlin-int8-asym-zp.patch` was imported
  into the fork and re-exported in #189; the standard export marker is now
  at `:24`. 44 of 45 files carry it; the 45th is the retired
  `dflash2-backport.patch` (its own RETIRED header — also parseable).
- The table kept pace by hand: 83 lines, 50 pipe-rows = 2 header + 48 data
  (all 45 series names + 3 kvarn rows, comm-verified; new rows for #188,
  #226, #222). The load grew with it: PATCHES.md is now the single
  most-touched file in the last 40 commits (8 touches;
  `git log -40 --name-only`). *(2026-09-29: miscounted — the same window
  at 2522ef9 gives 9; see above.)*
- Still unguarded (zero CI references) and stale in the same places:
  README's "38 files" (`:122`), the consumer claims at `series:3-4` and
  `check_vllm_series.sh:8-9`, the `diff -ruN` mechanism claim and the
  dangling `docs/MR-DRAFT.md` (`PATCHES.md:82-83`).
- New drift, postdating the review: #189's commit message says the
  pins were "moved by scripts/pin-bump.py" — no such file is in `scripts/`
  (only `export-patch.sh`; since #219 also `hq-doctor.sh`, still no
  `pin-bump.py`); tooling referenced but never committed.
- Line numbers, counts and file names throughout updated to this tree.

**Scope.** `PATCHES.md` and its relationship to `patches/series`, the patch
preambles, `scripts/export-patch.sh`, and `verify.sh`'s per-patch markers.
**Not in scope:** the apply path (`patches/apply.sh` — candidate 1's plan;
this plan assumes series order stays the single source of truth for
*application*, and generates only the *index*).

## 1. Current state, precisely

### 1.1 The index is hand-maintained — and today, accidentally complete

`PATCHES.md` is 83 lines: the kinds taxonomy (`:6-10`), the fork/export
rules (`:12-16`), a 6-column table (`:18-19`: patch · kind · what ·
upstream · cut against · retires when), and retired/notes prose (`:69-83`).
Re-verified 2026-09-29 (d5e2a01; unchanged since 2522ef9): 50 lines start
with `|` = 2 header + **48 data rows** — all 45 series names present
(checked with `comm -3` against `patches/series` + `kvarn/*0.30.0*.patch`:
empty), plus the 3 kvarn rows (`:65-67`), no duplicates. So the table is complete *today* — the
problem is that nothing keeps it that way, and its surroundings have
already rotted:

- It is the **most-touched file** upstream: 9 touches in the last 40
  commits (`git log -40 --format= --name-only upstream/main | sort | uniq
  -c`, window f98de95..d5e2a01; was given as 8 at 2522ef9, which was a
  miscount — that window also gives 9). Next are `verify.sh`,
  `patches/series`, `single-user/start_qwen.sh`, `docs/gotchas.md` at 7.
  The newest touch, #234 (e355f9f), hand-edited the
  `marlin-repack-staged-sm80` row (`:37`) to say what its re-exported
  preamble already says — the maintenance load is real and recurring.
- `PATCHES.md:12` names the export point `qwen38/0.30-cut5`, but the fork
  has been re-cut through `qwen38/0.30-cut8` (`git ls-remote
  https://github.com/cpuchip/vllm 'refs/tags/qwen38/*'`), and only 40 of
  the 47 export-marker hashes are reachable from cut5 (`gh api
  repos/cpuchip/vllm/compare/<tag>...<hash>`): #234's four re-exports
  (`marlin-int8-asym-zp`, `marlin-repack-staged-sm80`, `speed-knobs-envs`,
  `kvarn-0.30.0`) only from cut8, `spec-attn-smem-fit` only from
  cut6/cut7, and `bench-sse-keepalive` (`757723b`) /
  `kvarn-recycled-pages-0.30.0` (`40ab8e0`) from no tag (the commits API
  answers 422 "No commit found"). The export tag is one more hand-kept fact
  with no check. (New finding 2026-09-29.)
- `README.md:122` says the series is "**38 files**" — there are 45
  (verified). No mechanism connects the two.
- `patches/series:3-4` names "the README install loop" as a consumer — the
  loop moved to `docs/install.md` in #129; the header wasn't updated.
- `patches/check_vllm_series.sh:8-9` repeats the same stale "which the
  Dockerfile and the README use".
- `PATCHES.md:83` sends readers to `docs/MR-DRAFT.md` — a file that does
  not exist anywhere in the tree (verified: `git grep -n MR-DRAFT` finds
  only `PATCHES.md:83` plus the review and remediation docs quoting it).
- `PATCHES.md:82-83`'s own mechanism claim is stale: it says
  `dflash2-z-adaptive-emitted` and `offload-wsl2-devptr` "still carry raw
  `diff -ruN` headers with timestamps instead of a preamble". Zero
  `diff -ruN` lines exist in any patch file (`git grep -n 'diff -ruN'`
  matches only `PATCHES.md:82` itself and the docs quoting it); the 0.30
  re-export refreshed both files' markers again without touching the gap:
  `dflash2-z-adaptive-emitted.patch:1-4` is blank lines + the standard
  export marker (no prose preamble — the *gap* the note describes is real),
  and `offload-wsl2-devptr.patch:1` has a one-line preamble.
- **Nothing in CI references `PATCHES.md`** (verified: zero matches in
  `.github/workflows/`).

### 1.2 The machine-read conventions that already exist (and work)

- **The export marker**: every patch file carries
  `--- exported from cpuchip/vllm <short-hash> (<topic>); regenerate with
  scripts/export-patch.sh, do not edit ---` (produced by
  `export-patch.sh:19`). Provenance is machine-readable in 44 of 45 files;
  the 45th is the retired `dflash2-backport.patch`, whose RETIRED prose
  header plays the same role. (The review's second exception is gone:
  `marlin-int8-asym-zp.patch` was hand-cut with its own marker at `:24`;
  #189 imported it into the fork and re-exported it — standard marker now
  at `:24`, re-exported again by #234 as `f11865100`, still at `:24`.)
  Machine-readable is not the same as resolvable: 7 of the 47 hashes are
  not on the `PATCHES.md:12` tag (1.1).
- **The preamble**: `export-patch.sh:15` copies the fork commit *body*
  into the file above the marker (stripping `^Source:` lines and stray
  diff-syntax lines). Whatever structure the commit body has, the preamble
  has — no export change needed to start passing headers through.
- **`Supersedes:`**: one producer (`spec-decode-scratch-within-budget.patch:49`
  declares `Supersedes: spec-decode-scratch-token-units.patch`), one consumer
  (`verify.sh:91`, the `grep` in `superseded_by()` at `:88`, called at
  `:107` in the check ladder; was `:72` at 2522ef9, shifted +19 by #219's
  `--wait` header). A
  working `Key: value` convention, consumed in production — currently
  carrying exactly one fact.
- **The `graded()` verify markers**: per-patch metadata as bash triples in
  `verify.sh:376` (definition; rationale comment `:369-375`) and call sites
  at `:442`
  (`serve-404-served-names` → `entrypoints/serve/engine/serving.py` /
  `Served models:`), `:448` (`auth-deny-default` → `…/authenticate.py` /
  `UNGUARDED_PATHS`), `:454` (`tokenize-v1-route` → `…/tokenize/api_router.py`
  / `prefix="/v1"`) — were `:342`, `:332-341`, `:408/:414/:420` at 2522ef9,
  shifted +34 by #219. A third home for per-patch facts, outside both the
  table and the preambles.

### 1.3 Where each fact lives today

| fact | home(s) | checked? |
|---|---|---|
| order | `patches/series` (+5 re-parsers — candidate 1) | dir↔series agreement |
| kind / upstream / retires-when | `PATCHES.md` table only | **nothing** |
| description ("what") | `PATCHES.md` table + preamble prose (duplicated, drift-prone) | nothing |
| provenance (commit, branch) | the export marker in every file + the export tag at `PATCHES.md:12` | nothing (tag stale: 7 of 47 hashes off cut5) |
| supersedes | preamble header (1 file) | `verify.sh:91` |
| verify marker (file+string) | `verify.sh` triples (3 patches) | `verify.sh` itself |
| file count | `README.md:122` | nothing (stale: 38 vs 45) |
## 2. The design

### 2.1 Headers in the preamble, table generated from them

Extend the working `Supersedes:` pattern to three more keys, living in the
fork commit bodies (still the source of truth — `export-patch.sh` passes
bodies through verbatim today, so **no export-tooling change is required**):

```text
Kind: backport | fix | feature | local | own        (the taxonomy at PATCHES.md:6-10)
Upstream: vllm #58028 | fork #57 | none             (free-form, as today)
Retires-when: upstream PR | the pin that carries #55450 | stays | rides with mq3d
```

A new stdlib-only generator, `scripts/patches_md.py`:

1. reads `patches/series` for order (the one canonical list — candidate 1's
   `--list` when it exists),
2. parses each patch file's preamble for `Kind:`/`Upstream:`/`Retires-when:`,
   takes the **description from the preamble's first prose paragraph** (the
   same text the table duplicates by hand today — one home, in the patch),
3. emits the table section of `PATCHES.md`, leaving the hand-written parts
   (kinds taxonomy, export rules, retired prose, port notes) untouched
   between markers like `<!-- table:begin -->` / `<!-- table:end -->`.

A row whose patch lacks a header fails the generator by name — adding a
patch without its metadata stops being possible to do silently.

### 2.2 CI: freshness as a diff check

One step in `patch-integrity.yml`: run `python scripts/patches_md.py`,
then `git diff --exit-code PATCHES.md`. The table can no longer rot: a
preamble edit without regeneration fails the build, the way a hand-edited
hunk header in one of the five contractual DFlash patches already fails
`check_vllm_series.sh`'s `git apply` pass (`check_vllm_series.sh:13-15`;
corrected 2026-09-29 — earlier text said any hand-edited patch fails it,
but pass 1 is GNU `patch` and only checks that hunks apply). Also fixed in passing (each its own
line): the `README.md:122` count sentence drops the number; the
`patches/series:3-4` and `check_vllm_series.sh:8-9` consumer lists are
corrected; `PATCHES.md:83`'s dangling `docs/MR-DRAFT.md` reference is
repointed (and `:82-83` rewritten — the mechanism it describes no longer
exists; `dflash2-z-adaptive-emitted` still needs its prose preamble,
written as a fork commit body per rule 2, never hand-edited); and
`PATCHES.md:12`'s export tag is moved to the tag the files' hashes are
actually on (1.1 — ideally the generator checks it). "Rule 2" is
`export-patch.sh:4`'s citation of `docs/fork-workflow.md`, which is not in
the tree either (verified 2026-09-29: `ls docs/fork-workflow.md` fails; the
rule's substance is `PATCHES.md:14-15`, "do not edit the files by hand") —
a third dangling reference to repoint in the same pass.

### 2.3 The graded() markers — flagged, not decided

The three `verify.sh` marker triples (1.2) could become `Verify:` preamble
headers consumed by `verify.sh` the way it already consumes `Supersedes:`.
Worth doing **after** the header convention lands — it is the same shape
of win (metadata beside the patch) but touches verify.sh's server-check
section, so it is sequenced as a follow-up, not bundled.

### 2.4 What does not change

`patches/series` stays the source of truth for order; the fork branch for
content; `export-patch.sh` byte-for-byte; `PATCHES.md`'s hand-written prose
(the retired history, the port notes — the parts that are genuinely prose);
`docs/optimizations.md`'s narrative per optimization (the table links to
it; it does not generate from it).
## 3. Rollout

Two PRs. The first is the whole point; the second is housekeeping that
rides the same diff boundary.

### 3.1 PR A — headers, generator, CI freshness check

Backfill the three headers into the fork commit bodies (one pass over the
existing topic branches; bodies carry the metadata, subjects unchanged,
so the branch hashes the series cares about — commit order and diffs —
are untouched). Re-export the affected files with `export-patch.sh`
(unchanged tooling). Add `scripts/patches_md.py` + the `patch-integrity.yml`
step. The generator's first run should reproduce the current table
near-verbatim — that parity *is* the acceptance test (4.1).

Acceptance: CI green with the new step; `git diff` of the generated table
vs the pre-existing one contains only the deliberate fixes (2.2).

### 3.2 PR B — the reference fixes + the preamble gap

`README.md:122` loses its count; `patches/series:3-4` and
`check_vllm_series.sh:8-9` name the real consumers; `PATCHES.md:82-83`
is rewritten (and its `docs/MR-DRAFT.md` reference repointed);
`export-patch.sh:4`'s `docs/fork-workflow.md` citation is repointed;
`PATCHES.md:12` names the export tag the current hashes are on (cut8 or a
new cut; two hashes, `757723b` and `40ab8e0`, need a push or a re-export
first, since no tag carries them — 1.1);
`dflash2-z-adaptive-emitted` gets its prose preamble via a fork commit
body edit + re-export (never a hand edit — rule 2). The `Verify:` marker
migration (2.3) is proposed as a follow-up issue, not this PR.

Acceptance: the generator still reproduces PATCHES.md (CI);
`git grep -n 'MR-DRAFT' -- ':!docs/architecture-review-*' ':!docs/*remediation*.md'`
finds nothing (a bare `grep -rn 'MR-DRAFT' .` always matches the review
and this plan, which quote it — corrected 2026-09-29); README has no
patch count.

## 4. Test plan

| test | how | proves |
|---|---|---|
| generation parity | run the generator on the backfilled tree; `git diff PATCHES.md` shows only the deliberate 2.2 fixes | the parser reads what the table said, nothing lost |
| freshness gate | scratch-edit one `Kind:` header without regenerating; CI must fail `git diff --exit-code` | the table cannot rot silently again |
| missing-header tripwire | add a scratch patch with no `Kind:` | the generator names the file and exits nonzero |
| export round-trip | re-export one patched topic on a scratch fork checkout; the headers land in the preamble verbatim | the fork→file path needs no tooling change |
| consumer sanity | `bash patches/check_vllm_series.sh` (CI) and `verify.sh --install` (image build) unchanged and green | the index work didn't touch the apply/check path |

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| Backfilling fork commit bodies rewrites topic-branch history the patch files' export markers point at (each names a short-hash; `PATCHES.md:12` itself no longer names hashes — since #189 it names the branch plus the `qwen38/0.30-cut5` export tag, meant to keep a rewrite from orphaning those hashes). **Status 2026-09-29:** the fork has already been rewritten through `qwen38/0.30-cut8` and #234 re-exported four files from it without moving `:12`'s tag, so 7 of 47 hashes are off cut5 and 2 are on no tag (1.1) — the tag defuses the risk only if every re-export moves it | low (backfill) / already happened (tag drift) | do the backfill on the *next* natural re-export and cut a new export tag in the same PR — the cadence #189 set: re-export, tag, update `:12`; have the generator check that every marker hash is an ancestor of the tag `:12` names (needs network or a fork checkout, so a warn-only local step, not the CI freshness gate) |
| The generator becomes a second place the description is edited | low | the preamble is the only source; the generated block is marker-fenced with "do not edit" in the fence comment |
| Free-form `Upstream:`/`Retires-when:` values drift in style | low | the taxonomy line for `Kind:` is validated; the other two are prose by design (as the table is today) |
| The `Verify:` migration tempts bundling | medium | explicitly sequenced as a follow-up (2.3); this plan closes without touching verify.sh |
| A patch that deliberately has no preamble (`dflash2-z-adaptive-emitted` today) blocks the generator | low | the generator emits the row with an empty description cell and warns — blocking only on missing `Kind:`, never on missing prose |

## 6. Done when

1. `python scripts/patches_md.py && git diff --exit-code PATCHES.md` is a
   green step in `patch-integrity.yml`.
2. Every patch's kind/upstream/retires-when/description is readable in its
   own preamble; deleting `PATCHES.md`'s table loses no information.
3. `README.md` contains no patch count; `patches/series`'s header names
   the consumers that exist; `git grep -n 'MR-DRAFT'` outside the review
   and remediation docs finds nothing; `PATCHES.md:12`'s tag carries every
   marker hash.
4. `PATCHES.md`'s hand-written remainder is prose that is genuinely prose
   (retired history, port notes) — the most-touched file becomes one
   of the least-touched.
