# Generate the patch index from the patches — remediation plan

Architecture review 2026-09-26, candidate 6 (Worth exploring). The full
report is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree
on 2026-09-26.

**Scope.** `PATCHES.md` and its relationship to `patches/series`, the patch
preambles, `scripts/export-patch.sh`, and `verify.sh`'s per-patch markers.
**Not in scope:** the apply path (`patches/apply.sh` — candidate 1's plan;
this plan assumes series order stays the single source of truth for
*application*, and generates only the *index*).

## 1. Current state, precisely

### 1.1 The index is hand-maintained — and today, accidentally complete

`PATCHES.md` is 82 lines: the kinds taxonomy (`:6-10`), the fork/export
rules (`:12-18`), a 6-column table (`:20-21`: patch · kind · what ·
upstream · cut against · retires when), and retired/notes prose (`:70-82`).
Verified 2026-09-26: 49 lines start with `|` = 2 header + **47 data rows** —
all 45 series names present (checked with `comm` against `patches/series`),
plus the 2 kvarn rows, no duplicates. So the table is complete *today* — the
problem is that nothing keeps it that way, and its surroundings have
already rotted:

- It is the **5th-most-touched file** upstream (9 touches in the last 40
  commits) — the maintenance load is real and recurring.
- `README.md:119` says the series is "**38 files**" — there are 45
  (verified). No mechanism connects the two.
- `patches/series:3-4` names "the README install loop" as a consumer — the
  loop moved to `docs/install.md` in #129; the header wasn't updated.
- `patches/check_vllm_series.sh:8-9` repeats the same stale "which the
  Dockerfile and the README use".
- `PATCHES.md:82` sends readers to `docs/MR-DRAFT.md` — a file that does
  not exist anywhere in the tree (verified: only reference in the repo).
- `PATCHES.md:81-82`'s own mechanism claim is stale: it says
  `dflash2-z-adaptive-emitted` and `offload-wsl2-devptr` "still carry raw
  `diff -ruN` headers with timestamps instead of a preamble". Both files
  have since been re-exported: zero `diff -ruN` lines in either (verified);
  `dflash2-z-adaptive-emitted.patch:1-5` is blank lines + the standard
  export marker (no prose preamble — the *gap* the note describes is real),
  and `offload-wsl2-devptr.patch:1` now has a one-line preamble.
- **Nothing in CI references `PATCHES.md`** (verified: zero matches in
  `.github/workflows/`).

### 1.2 The machine-read conventions that already exist (and work)

- **The export marker**: every patch file carries
  `--- exported from cpuchip/vllm <short-hash> (<topic>); regenerate with
  scripts/export-patch.sh, do not edit ---` (produced by
  `export-patch.sh:19`). Provenance is already machine-readable in 44 of 45
  files (the hand-cut `marlin-int8-asym-zp.patch:24` carries its own
  "cut by hand" marker — also parseable).
- **The preamble**: `export-patch.sh:15` copies the fork commit *body*
  into the file above the marker (stripping `^Source:` lines and stray
  diff-syntax lines). Whatever structure the commit body has, the preamble
  has — no export change needed to start passing headers through.
- **`Supersedes:`**: one producer (`spec-decode-scratch-within-budget.patch:49`
  declares `Supersedes: spec-decode-scratch-token-units.patch`), one consumer
  (`verify.sh:67`, in the `superseded_by` rung of the check ladder). A
  working `Key: value` convention, consumed in production — currently
  carrying exactly one fact.
- **The `graded()` verify markers**: per-patch metadata as bash triples in
  `verify.sh:333-336` (definition) and call sites at `:399`
  (`serve-404-served-names` → `entrypoints/serve/engine/serving.py` /
  `Served models:`), `:405` (`auth-deny-default` → `…/authenticate.py` /
  `UNGUARDED_PATHS`), `:411` (`tokenize-v1-route` → `…/tokenize/api_router.py`
  / `prefix="/v1"`). A third home for per-patch facts, outside both the
  table and the preambles.

### 1.3 Where each fact lives today

| fact | home(s) | checked? |
|---|---|---|
| order | `patches/series` (+5 re-parsers — candidate 1) | dir↔series agreement |
| kind / upstream / retires-when | `PATCHES.md` table only | **nothing** |
| description ("what") | `PATCHES.md` table + preamble prose (duplicated, drift-prone) | nothing |
| provenance (commit, branch) | the export marker in every file | by inspection |
| supersedes | preamble header (1 file) | `verify.sh:67` |
| verify marker (file+string) | `verify.sh` triples (3 patches) | `verify.sh` itself |
| file count | `README.md:119` | nothing (stale: 38 vs 45) |
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
preamble edit without regeneration fails the same way a hand-edited patch
already fails `check_vllm_series.sh`. Also fixed in passing (each its own
line): the `README.md:119` count sentence drops the number; the
`patches/series:3-4` and `check_vllm_series.sh:8-9` consumer lists are
corrected; `PATCHES.md:82`'s dangling `docs/MR-DRAFT.md` reference is
repointed (and `:81-82` rewritten — the mechanism it describes no longer
exists; `dflash2-z-adaptive-emitted` still needs its prose preamble,
written as a fork commit body per rule 2, never hand-edited).

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

`README.md:119` loses its count; `patches/series:3-4` and
`check_vllm_series.sh:8-9` name the real consumers; `PATCHES.md:81-82`
is rewritten (and its `docs/MR-DRAFT.md` reference repointed);
`dflash2-z-adaptive-emitted` gets its prose preamble via a fork commit
body edit + re-export (never a hand edit — rule 2). The `Verify:` marker
migration (2.3) is proposed as a follow-up issue, not this PR.

Acceptance: the generator still reproduces PATCHES.md (CI);
`grep -rn 'MR-DRAFT' .` finds nothing dangling; README has no patch count.

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
| Backfilling fork commit bodies rewrites topic-branch history the series' prose points at (`PATCHES.md:12-13` names exact branch hashes) | medium | bodies-only edits change commit hashes — so do it on the *next* natural re-export (the pin-flip cadence the repo already has), or accept one documented rewrite of the hq/hq2 lines; either way the PR names the new hashes in the same edit that updates `:12-13` |
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
   the consumers that exist; `grep -rn 'MR-DRAFT' .` finds nothing.
4. `PATCHES.md`'s hand-written remainder is prose that is genuinely prose
   (retired history, port notes) — the 5th-most-touched file becomes one
   of the least-touched.
