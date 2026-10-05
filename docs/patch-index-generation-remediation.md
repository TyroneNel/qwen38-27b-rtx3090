# Generate the patch index from the patches — remediation plan

Architecture review 2026-09-26, candidate 6 (Worth exploring). The full
report is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree
on 2026-09-26.

**Implemented 2026-10-04 and 2026-10-05 on upstream/main @ 10bb488 (vLLM 0.30.0).**

**Status:** In review. PR B is syv-ai/HyperQwen#274 (`c485c94`). PR A is
#275 (`17b1551`), stacked on #274, so #274 merges first. CI is green on
both, and #275 also runs the new `patch-index` job.

- **12 of 50 hashes were off cut5.** The 2026-10-04 pass below checked only
  the five new or changed hashes. #274 checked all 50 with `git merge-base
  --is-ancestor` against the fetched fork refs. 12 marker hashes were not
  on cut5, the tag that `PATCHES.md:12` named for all of them. By ref: 39
  on cut7, 6 on cut9 only, 2 on `qwen38/0.30-chainfix` only, and 1 on each
  of the two `TyroneNel/vllm` tags. `kvarn-recycled-pages`' `40ab8e0` was
  on no fork ref. No fork commit exports to that file: its commits on cut7,
  cut9 and chainfix also carry the `kvarn_attn.py` half, which this repo
  keeps in `kvarn/files`.
- **PR B (#274).** `PATCHES.md` names the fork ref behind every exported
  commit ("Where the commits are"). Two files are re-exported so that a tag
  holds their hash: `sampler-warmup-cuda` (`92e7256fa` → `ba3b5d8e6`, cut9)
  and `bench-sse-keepalive` (`757723b` → `bebdd65c5`, cut9). The upstream
  cells cite vllm #59892, #59888, #59890, #59893 and #59889 (merged as
  `7867d6c52d`, no release yet). `tokenize-v1-route` stays `feature`, not
  `local` as the tracker said, because `local` means hardware or
  environment. Its upstream cell is "none" and its retires-when is "stays".
  The end note names the six files with a preamble of one line or none, not
  two. `export-patch.sh:4` points at the `PATCHES.md` export rules.
  `apply.sh --kvarn` exits 2 and names each offender when `KVARN` and
  `kvarn/*.patch` disagree. That closes candidate 1's last item.
- **PR A (#275).** Each patch file ends its preamble with five headers:
  `Kind`, `What`, `Upstream`, `Cut-against` and `Retires-when`.
  `scripts/patches_md.py` (stdlib only) writes the table between markers in
  `PATCHES.md`, in apply order, from `apply.sh --list` and the new
  `apply.sh --list --kvarn`. The `patch-index` job in `patch-integrity.yml`
  runs it, then `git diff --exit-code PATCHES.md`. `git merge-tree` finds
  no conflict with #271.
- **Two changes from §3.1.** First, five headers, not three. The "what"
  cell matched the preamble's first prose paragraph in only 4 of 50 files,
  and `cut against` had no source. Second, a new cut, not new bodies on the
  old topic branches. Those branches are on `cpuchip/vllm`, where this work
  has no push access. The cut is tag `qwen38/0.30-index-cut1` on
  `TyroneNel/vllm` (branch `qwen38/0.30-index`, `7c013fc0e`): v0.30.0 plus
  one `[qwen38] <topic>` commit per file, in apply order. Each body is the
  old preamble, the headers, and a `Source:` line that `export-patch.sh`
  drops. The marker still says `cpuchip/vllm`, because `export-patch.sh` is
  unchanged.
- **Measured on #275 (2026-10-05).** All 50 hunk bodies are unchanged
  (compared after the marker, without index lines and @@ numbers). Index
  lines changed in 11 files. The @@ numbers changed in the 6 files whose 7
  hunks applied at an offset, and the series check now reports 0 offsets.
  The `vllm/` tree after the series and the KVarN patches is byte-identical
  to upstream/main's. The generated rows equal #274's 50 rows as a set; only
  the order changed. The 50 marker hashes, in apply order, equal `git
  rev-list --reverse v0.30.0..qwen38/0.30-index-cut1`. Each bad-header case
  exits 1 and names the file. A file missing from `patches/series` exits 2.
- **Not done.** The `auth-deny-default` Cut-against cell says a hunk "lands
  at an offset", but it applies at none on upstream/main and on #275. The
  fix is a fork commit body change and a re-export. `verify.sh:108-127`
  still names the four KVarN patches. The `Verify:` header (§2.3) is still
  a follow-up. The generator does not check that the marker hashes are on
  the tag (§5); #275 checked it once, by hand. `dflash2-z-adaptive-emitted`
  and `dflash2-prewarm` still have no prose above their headers (§3.2 item
  3), but their `What:` header now gives each one a description.
  cpuchip/vllm can take the cut with a PR from `qwen38/0.30-index`.

**Re-verified 2026-10-04 against upstream/main @ e371b42 (vLLM 0.30.0).**

**Status at e371b42:** Partly done. Candidate 1 (e1459c7) closed most of PR B's reference items. PR A (headers, generator, CI freshness) not started.

Thirteen commits since d5e2a01. Five touch what this plan cites: e7a5823
(#233, new `sampler-warmup-cuda`), e1459c7 (#242/#243/#244: `patches/apply.sh`,
retires `dflash2-backport` and the two 0.27.1 KVarN files), 69036cb (#261,
re-exports `dflash2-ngram-chains`), 210db97 (#263, new `pinned-kv-empty-cache`)
and bd6c5e2 (#262, new `kvarn/kvarn-fp16-dequant-0.30.0`). No patch file
has a `Kind:`, `Upstream:` or `Retires-when:` line (`grep -rln
'^\(Kind\|Upstream\|Retires-when\):' patches kvarn` is empty).

- **Index (recounted):** `PATCHES.md` is 90 lines (was 83). 52 lines start
  with `|` = 2 header + **50 data rows** (was 48). `patches/series` has 46
  names (was 45: −`dflash2-backport`, +`sampler-warmup-cuda`,
  +`pinned-kv-empty-cache`). `kvarn/*0.30.0*.patch` has 4 files (was 3:
  +`kvarn-fp16-dequant-0.30.0`). `comm -3` of the row names against series +
  kvarn is empty, with no duplicates. The table is still complete, but only
  by hand: e7a5823, 210db97 and bd6c5e2 each added their row in the same
  commit, and e1459c7 deleted the `dflash2-backport` row. Nothing checks it.
- **Touch count:** `git log -40 --format= --name-only e371b42 | sort | uniq
  -c` (window 2f9b1e6..e371b42) gives `PATCHES.md` 9. It is now second, after
  `docs/reproductions/README.md` at 10. Next are `verify.sh`, `patches/series`
  and `docs/install.md` at 6.
- **Fixed by e1459c7:** `README.md:122` has no count. No other patch count is
  in `README.md`, `docs/install.md`, `PATCHES.md`, `patches/`, `kvarn/` or
  `verify.sh` (`git grep -nE '\b(3[0-9]|4[0-9]|50) (patches|files)\b'`).
  `patches/series:3-6` names `patches/apply.sh` and its callers (the
  Dockerfile, `docs/install.md:114`, `docs/python-314.md:75`) and the two
  `--list` readers. `check_vllm_series.sh:8-10` names `apply.sh` and its
  callers. `PATCHES.md:90` no longer names `docs/MR-DRAFT.md` (`git grep -n
  MR-DRAFT` at e371b42 is empty).
- **Still present:** `PATCHES.md:89-90` (was `:82-83`) still claims two
  files carry raw `diff -ruN` headers. `git grep -n 'diff -ruN'` matches only
  `:89`. `dflash2-z-adaptive-emitted.patch:1-3` is still two blank lines and the
  marker; `offload-wsl2-devptr.patch:1` is a one-line preamble.
  `export-patch.sh:4` still cites `docs/fork-workflow.md`, which is not in
  the tree. `.github/` still has zero `PATCHES` matches. bd6c5e2 added a
  `kvarn-torch-gate` job, so `patch-integrity.yml` has three jobs:
  `model-verification`, `kvarn-torch-gate`, `git-apply`.
- **Export markers:** 50 of 50 files carry the marker (46 in `patches/`, 4 in
  `kvarn/`; was 44 of 45 + 3). The retired-file exception left with the file.
  The one `Supersedes:` producer is still
  `spec-decode-scratch-within-budget.patch:49`.
- **Export refs (`PATCHES.md:12`):** the line now names three tags, all as
  `cpuchip/vllm` refs: `qwen38/0.30-cut5`, `qwen38/0.30-warmup-cut1`
  (`sampler-warmup-cuda`) and `qwen38/0.30-pinned-kv-cut1`
  (`pinned-kv-empty-cache`). Checked with `git ls-remote` and `gh api
  repos/<owner>/vllm/compare/<ref>...<hash>`:
  - `qwen38/0.30-pinned-kv-cut1` (→ `bbed74b28`, the marker hash) and branch
    `qwen38/0.30-pinned-kv-empty-cache` exist only on `TyroneNel/vllm`. That
    branch is the head of the open cpuchip/vllm#4.
  - `kvarn-fp16-dequant-0.30.0`'s `7f8c3ef0a` is reachable from
    `TyroneNel/vllm` tag `qwen38/0.30-kvarn-fp16-cut1` (→ `8ad33d165`). Its
    branch `qwen38/0.30-kvarn-fp16` is the head of the open cpuchip/vllm#3.
    `:12` does not name this tag.
  - `sampler-warmup-cuda`'s `92e7256fa` is not on `qwen38/0.30-warmup-cut1`
    (→ `ee1832bec`; compare says `diverged`). It is the head of cpuchip branch
    `qwen38/0.30-warmup-cut7`, which has no tag.
  - `dflash2-ngram-chains`' `596a96779` (#261) and `kvarn-v2-runner-0.30.0`'s
    `757257e13` (e1459c7) are reachable only from cpuchip branch
    `qwen38/0.30-chainfix` (`688a0bd17`), which has no tag.
  - The fork now has `qwen38/0.30-cut9` (= `qwen38/0.30` head `ba3b5d8e6`).
    None of these five hashes is on cut5, cut8 or cut9.

  No hash is orphaned: each one is on a branch. The defect is that `:12`
  names `cpuchip/vllm` for two refs that exist only on `TyroneNel/vllm` until
  cpuchip merges #3/#4, omits the kvarn-fp16 tag, and three hashes sit on
  untagged branches. The other 45 hashes are unchanged since d5e2a01; this
  pass did not re-check them against cut9.
- **Upstream cells (new evidence for the `Upstream:` header):**
  `PATCHES.md:20` (`auth-deny-default` → vllm #58028), `:21`
  (`bench-probe-errors` → #58024), `:53` (`serve-404-served-names` → #58025),
  `:54` (`serve-model-path-match` → #58026) and `:55` (`tokenize-v1-route` →
  #58027) cite vllm-project/vllm PRs. All five are closed. On 2026-10-03 the
  user reopened them with the same titles and head branches as #59892,
  #59888, #59889, #59890 and #59891 (`gh pr view`).
  - (Update 2026-10-04 20:45 UTC) #59889 (`serve-404-served-names`) merged
    into vLLM main as `7867d6c52d`. It is not in v0.30.0 or in any release
    yet (`gh api .../compare/v0.30.0...7867d6c52d` says `diverged`). The
    upstream 404 says `Valid aliases: ...`, and the patch says
    `Served models: ...`. So the patch stays in the series until the pin
    carries #59889. When the pin moves, drop the patch and the
    `"Served models:"` row in `verify.sh:452` together.
  - #59891 is closed. vLLM member DarkLight1337 said the `/v1` prefix is
    only for official OpenAI endpoints, and the user closed the PR. So
    `tokenize-v1-route` stays a local patch.
  - On #59892, DarkLight1337 said the `--api-key` scope limit is intentional
    unless simon-mo wants to change it. Reviewer liulanze found a defect: with
    `--root-path /`, `removeprefix()` turns `/health` into `health`, and the
    request gets 401. `auth-deny-default.patch:118` carries the same line as
    context (`url_path = scope["path"].removeprefix(root_path)`), and the
    patch's new allowlist check (`:123`, `:129`) reads that path, so the local
    patch has the same defect. liulanze's second point (the weight-sync client
    sends no key) was not checked against 0.30.0.
  - `PATCHES.md:40` (`pinned-kv-empty-cache`) says "none yet". Its upstream PR
    is vllm-project/vllm#59893 (open).

  Six of the 50 upstream cells are stale today, and no check reads them.
- **Line shifts:** `verify.sh` `superseded_by()` `:88` → `:85`, its `grep`
  `:91` → `:88`, its caller `:107` → `:100`. `graded()` `:376` → `:386`,
  rationale `:369-375` → `:379-385`, call sites `:442/:448/:454` →
  `:452/:458/:464`. The `git apply` hunk-header note in
  `check_vllm_series.sh` `:13-15` → `:15-17`.
- **Plan changes (marked "(2026-10-04)" below):** the generator reads the
  order from `bash patches/apply.sh --list` (§2.1). `apply.sh` gains
  `--list --kvarn`, because the KVarN order is a hard-coded array at
  `apply.sh:50-51` (§2.1). The freshness check is its own Python-only CI job
  (§2.2). `PATCHES.md:12` names the right repo and one tag per ref (§2.2,
  §3.2). PR B shrinks to four items (§3.2). §6 marks what is done.

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
*application*, and generates only the *index*). (2026-10-04: candidate 1
landed `patches/apply.sh` in e1459c7. This plan reads its `--list` and adds
one small arm to it, `--list --kvarn`; it does not change how it applies.)

## 1. Current state, precisely

### 1.1 The index is hand-maintained — and today, accidentally complete

`PATCHES.md` is 90 lines: the kinds taxonomy (`:6-10`), the fork/export
rules (`:12-16`), a 6-column table (`:18-19`: patch · kind · what ·
upstream · cut against · retires when), and retired/notes prose (`:71-90`).
Re-verified 2026-10-04 (e371b42; 83 lines and 48 data rows at d5e2a01): 52
lines start with `|` = 2 header + **50 data rows** — all 46 series names
present (checked with `comm -3` against `patches/series` +
`kvarn/*0.30.0*.patch`: empty), plus the 4 kvarn rows (`:66-69`), no
duplicates. So the table is complete *today* — the problem is that nothing
keeps it that way, and its surroundings have already rotted. (2026-10-04:
three new patches and one retirement since d5e2a01, each row added or
deleted by hand in its own commit; nothing checked any of them.)

- (2026-10-04) It is the **second most-touched file** upstream: 9 touches
  in the last 40 commits (window 2f9b1e6..e371b42), after
  `docs/reproductions/README.md` at 10; next are `verify.sh`,
  `patches/series` and `docs/install.md` at 6. History of this bullet:
  at d5e2a01 it was the **most-touched file**: 9 touches in the last 40
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
  **(2026-10-04)** `:12` now names three tags as `cpuchip/vllm` refs: cut5,
  `qwen38/0.30-warmup-cut1` and `qwen38/0.30-pinned-kv-cut1`. The fork is at
  `qwen38/0.30-cut9`. `qwen38/0.30-pinned-kv-cut1` exists only on
  `TyroneNel/vllm` (head of the open cpuchip/vllm#4). `kvarn-fp16-dequant`'s
  hash is on `TyroneNel/vllm` tag `qwen38/0.30-kvarn-fp16-cut1` (head of the
  open cpuchip/vllm#3), which `:12` does not name. `sampler-warmup-cuda`'s
  hash is not on `warmup-cut1`; it is the head of the untagged cpuchip branch
  `qwen38/0.30-warmup-cut7`. `dflash2-ngram-chains` and
  `kvarn-v2-runner-0.30.0` are only on the untagged cpuchip branch
  `qwen38/0.30-chainfix`. No hash is orphaned; the repo and tag names are
  wrong (full evidence in the 2026-10-04 block above).
- **(2026-10-04) The upstream cells are stale.** `:20`, `:21`, `:53`, `:54`
  and `:55` cite vllm #58028, #58024, #58025, #58026 and #58027. All five are
  closed. The user reopened them on 2026-10-03 as #59892, #59888, #59889,
  #59890 and #59891, and #59891 is closed again: `/v1` is only for official
  OpenAI endpoints, so `tokenize-v1-route` stays local. #59889 merged into
  vLLM main on 2026-10-04 (`7867d6c52d`, no release yet), so `:53`'s
  retire-when becomes "the pin carries vllm #59889". `:40`
  (`pinned-kv-empty-cache`) says "none yet", but vllm #59893 is open. Six of 50
  cells are wrong, and no check reads them. The review on #59892 also found a
  `--root-path /` 401 on `/health` that `auth-deny-default.patch:118` shares
  (2026-10-04 block above).
- ~~`README.md:122` says the series is "**38 files**" — there are 45
  (verified). No mechanism connects the two.~~ **Fixed 2026-10-04** (e1459c7):
  `README.md:122` has no count.
- ~~`patches/series:3-4` names "the README install loop" as a consumer — the
  loop moved to `docs/install.md` in #129; the header wasn't updated.~~
  **Fixed 2026-10-04** (e1459c7): `patches/series:3-6` names `apply.sh` and
  its real callers.
- ~~`patches/check_vllm_series.sh:8-9` repeats the same stale "which the
  Dockerfile and the README use".~~ **Fixed 2026-10-04** (e1459c7):
  `check_vllm_series.sh:8-10`.
- ~~`PATCHES.md:83` sends readers to `docs/MR-DRAFT.md` — a file that does
  not exist anywhere in the tree (verified: `git grep -n MR-DRAFT` finds
  only `PATCHES.md:83` plus the review and remediation docs quoting it).~~
  **Fixed 2026-10-04** (e1459c7): `git grep -n MR-DRAFT` at e371b42 is empty.
- **Still present 2026-10-04, now `PATCHES.md:89-90`.** `PATCHES.md:82-83`'s own mechanism claim is stale: it says
  `dflash2-z-adaptive-emitted` and `offload-wsl2-devptr` "still carry raw
  `diff -ruN` headers with timestamps instead of a preamble". Zero
  `diff -ruN` lines exist in any patch file (`git grep -n 'diff -ruN'`
  matches only `PATCHES.md:82` itself and the docs quoting it); the 0.30
  re-export refreshed both files' markers again without touching the gap:
  `dflash2-z-adaptive-emitted.patch:1-4` is blank lines + the standard
  export marker (no prose preamble — the *gap* the note describes is real),
  and `offload-wsl2-devptr.patch:1` has a one-line preamble. (2026-10-04:
  `git grep -n 'diff -ruN'` matches only `PATCHES.md:89`;
  `dflash2-z-adaptive-emitted.patch:1-3` is two blank lines and the marker.)
- **Nothing in CI references `PATCHES.md`** (verified: zero matches in
  `.github/workflows/`; re-verified 2026-10-04 — `patch-integrity.yml` now
  has three jobs, `model-verification`, `kvarn-torch-gate` and `git-apply`,
  and none reads `PATCHES.md`).

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
  not on the `PATCHES.md:12` tag (1.1). (2026-10-04: 50 of 50 files carry
  the marker — 46 in `patches/`, 4 in `kvarn/`; e1459c7 deleted
  `dflash2-backport.patch`. Five new or changed hashes are on no tag that
  `:12` names as a cpuchip ref; see 1.1.)
- **The preamble**: `export-patch.sh:15` copies the fork commit *body*
  into the file above the marker (stripping `^Source:` lines and stray
  diff-syntax lines). Whatever structure the commit body has, the preamble
  has — no export change needed to start passing headers through.
- **`Supersedes:`**: one producer (`spec-decode-scratch-within-budget.patch:49`
  declares `Supersedes: spec-decode-scratch-token-units.patch`), one consumer
  (`verify.sh:91`, the `grep` in `superseded_by()` at `:88`, called at
  `:107` in the check ladder; was `:72` at 2522ef9, shifted +19 by #219's
  `--wait` header; 2026-10-04: now `:88`, `:85` and `:100`, after
  e1459c7, 177ce26 and bd6c5e2 edited `verify.sh`). A
  working `Key: value` convention, consumed in production — currently
  carrying exactly one fact.
- **The `graded()` verify markers**: per-patch metadata as bash triples in
  `verify.sh:376` (definition; rationale comment `:369-375`) and call sites
  at `:442`
  (`serve-404-served-names` → `entrypoints/serve/engine/serving.py` /
  `Served models:`), `:448` (`auth-deny-default` → `…/authenticate.py` /
  `UNGUARDED_PATHS`), `:454` (`tokenize-v1-route` → `…/tokenize/api_router.py`
  / `prefix="/v1"`) — were `:342`, `:332-341`, `:408/:414/:420` at 2522ef9,
  shifted +34 by #219. (2026-10-04: now `:386`, `:379-385` and
  `:452/:458/:464`.) A third home for per-patch facts, outside both the
  table and the preambles.

### 1.3 Where each fact lives today

| fact | home(s) | checked? |
|---|---|---|
| order | `patches/series`, read only by `patches/apply.sh --list` (2026-10-04, e1459c7; was +5 re-parsers). KVarN order: the hard-coded `KVARN` array at `apply.sh:50-51` | dir↔series agreement (`apply.sh --list` exits 2 and names each offender). KVarN list: nothing (2026-10-05, #274: `apply.sh --kvarn` and `--list --kvarn` check it) |
| kind / upstream / retires-when | `PATCHES.md` table only | **nothing** (2026-10-04: 6 of 50 upstream cells stale — 5 cite closed vllm PRs, `:40` misses #59893; 1.1). (2026-10-05: #274 fixed the six cells. #275 moves them into headers in each patch file, and the `patch-index` job checks the table) |
| description ("what") | `PATCHES.md` table + preamble prose (duplicated, drift-prone) | nothing (2026-10-05, #275: the `What:` header, checked as above) |
| provenance (commit, branch) | the export marker in every file + the export refs at `PATCHES.md:12` | nothing (2026-10-04: `:12` names cpuchip for refs that exist only on `TyroneNel/vllm`, omits the kvarn-fp16 tag, and 3 hashes are on untagged branches; was: 7 of 47 hashes off cut5). (2026-10-05: 12 of 50 were off cut5 (#274). #275 puts all 50 on `TyroneNel/vllm` tag `qwen38/0.30-index-cut1`) |
| supersedes | preamble header (1 file) | `verify.sh:88` (was `:91`) |
| verify marker (file+string) | `verify.sh` triples (3 patches) | `verify.sh` itself |
| file count | none (2026-10-04: e1459c7 removed it from `README.md:122`) | not needed |
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

1. reads the order from `bash patches/apply.sh --list` (2026-10-04:
   candidate 1 landed it in e1459c7; it is the only `patches/series`
   parser, and it exits 2 when the series and `patches/` disagree, so the
   generator fails on that too). The KVarN rows need their order as well.
   It lives in the hard-coded `KVARN` array at `apply.sh:50-51`, and
   `--list` does not print it. Add `apply.sh --list --kvarn` (one more
   `case` arm) and read it from there, so the generator does not become
   another place the four KVarN names are written (`apply.sh:50-51`,
   `verify.sh:110-126` and `PATCHES.md:66-69` already are),
2. parses each patch file's preamble for `Kind:`/`Upstream:`/`Retires-when:`,
   takes the **description from the preamble's first prose paragraph** (the
   same text the table duplicates by hand today — one home, in the patch),
3. emits the table section of `PATCHES.md`, leaving the hand-written parts
   (kinds taxonomy, export rules, retired prose, port notes) untouched
   between markers like `<!-- table:begin -->` / `<!-- table:end -->`.

A row whose patch lacks a header fails the generator by name — adding a
patch without its metadata stops being possible to do silently.

(2026-10-04) The upstream column is already stale. Six of its 50 cells are
wrong at e371b42: `PATCHES.md:20/:21/:53/:54/:55` cite closed vllm PRs (the
live ones are vllm #59892, #59888 and #59890; vllm #59889 merged into vLLM
main on 2026-10-04 as `7867d6c52d`, no release yet; vllm #59891 is closed
again), and `:40` says "none yet" while vllm #59893 is open (1.1). With the header in the fork
commit body, the PR number changes in the patch that the PR is about, at
the next re-export. The example line above still cites #58028 for format
only.

### 2.2 CI: freshness as a diff check

One step in `patch-integrity.yml`: run `python scripts/patches_md.py`,
then `git diff --exit-code PATCHES.md`. (2026-10-04) Make it its own job,
like `model-verification` and `kvarn-torch-gate`: checkout, setup-python,
the two commands. It needs no vLLM checkout, so it does not belong in the
`git-apply` job, which clones vLLM at the pin. The table can no longer rot: a
preamble edit without regeneration fails the build, the way a hand-edited
hunk header in one of the five contractual DFlash patches already fails
`check_vllm_series.sh`'s `git apply` pass (`check_vllm_series.sh:15-17`
since e1459c7, was `:13-15`;
corrected 2026-09-29 — earlier text said any hand-edited patch fails it,
but pass 1 is GNU `patch` and only checks that hunks apply; since e1459c7
pass 1 runs through `patches/apply.sh`). (2026-10-04: e1459c7 already did
the README count, the series and check-script consumer lists, and the
`MR-DRAFT` reference below. Still open: the `:89-90` rewrite, the
z-adaptive preamble, `export-patch.sh:4`, and `:12`.) Also fixed in passing (each its own
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

(2026-10-04) The `:12` fix is now about repos as well as tags. Name each
ref with the repo it is on. `qwen38/0.30-pinned-kv-cut1` and
`qwen38/0.30-kvarn-fp16-cut1` are on `TyroneNel/vllm` until cpuchip merges
cpuchip/vllm#4 and #3. Tag the three hashes that sit only on untagged
branches (`92e7256fa` on `qwen38/0.30-warmup-cut7`, `596a96779` and
`757257e13` on `qwen38/0.30-chainfix`). The end state is one
`cpuchip/vllm` tag that carries all 50 marker hashes. cut9 carries none of
the five new ones, and they sit on branches that diverge from it, so that
needs a re-export from one cut. The generator's tag check (§5) reads the
repo from `:12`.

### 2.3 The graded() markers — flagged, not decided

The three `verify.sh` marker triples (1.2) could become `Verify:` preamble
headers consumed by `verify.sh` the way it already consumes `Supersedes:`.
Worth doing **after** the header convention lands — it is the same shape
of win (metadata beside the patch) but touches verify.sh's server-check
section, so it is sequenced as a follow-up, not bundled.

### 2.4 What does not change

`patches/series` stays the source of truth for order (2026-10-04: read
through `patches/apply.sh --list`, not parsed again); the fork branch for
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
(2026-10-04) Three changes since candidate 1 landed: the generator reads
`bash patches/apply.sh --list`; PR A adds `apply.sh --list --kvarn` for the
four KVarN rows (§2.1); the CI step is its own Python-only job (§2.2). The
backfill sets `Upstream:` to the live PR numbers (vllm #59892,
vllm #59888, vllm #59889, vllm #59890 and vllm #59893). `tokenize-v1-route`
gets `none`, because vllm #59891 is closed. `serve-404-served-names` keeps
vllm #59889 and records it as merged (`7867d6c52d`), and its retire-when
becomes "the pin carries vllm #59889". So the parity diff shows those
six cells changing on purpose.

Acceptance: CI green with the new step; `git diff` of the generated table
vs the pre-existing one contains only the deliberate fixes (2.2).

(2026-10-05) Implemented as #275, with the two changes in the block at the
top: five headers, and a new cut on `TyroneNel/vllm`. The six cell fixes
landed in #274, before the headers. So #275's parity check is exact: the
same 50 rows, in apply order.

### 3.2 PR B — the reference fixes + the preamble gap

(2026-10-04) PR B is smaller now. e1459c7 already removed the README
count, fixed the `patches/series` and `check_vllm_series.sh` consumer
lists, and dropped `docs/MR-DRAFT.md` from `PATCHES.md`. Four items remain:

1. rewrite `PATCHES.md:89-90` (the `diff -ruN` claim);
2. repoint `export-patch.sh:4`'s `docs/fork-workflow.md`;
3. give `dflash2-z-adaptive-emitted` its prose preamble (fork commit body +
   re-export);
4. fix `PATCHES.md:12`: name each ref with its repo, add the kvarn-fp16 tag,
   and tag or re-export the three untagged hashes (§2.2). It needs no
   generator, so it can ship alone, ahead of PR A.

(2026-10-05) Implemented as #274: items 1, 2 and 4. For item 4 it re-exports
two files from cut9 and lists the ref behind each hash, in place of new
cpuchip tags (no push access). #275 then put all 50 on one tag. Item 3 is
not done (see the block at the top).

The original text follows as history.

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
| Backfilling fork commit bodies rewrites topic-branch history the patch files' export markers point at (each names a short-hash; `PATCHES.md:12` itself no longer names hashes — since #189 it names the branch plus the `qwen38/0.30-cut5` export tag, meant to keep a rewrite from orphaning those hashes). **Status 2026-09-29:** the fork has already been rewritten through `qwen38/0.30-cut8` and #234 re-exported four files from it without moving `:12`'s tag, so 7 of 47 hashes are off cut5 and 2 are on no tag (1.1) — the tag defuses the risk only if every re-export moves it. **Status 2026-10-04:** the fork is at cut9; five new or changed hashes are on none of cut5/cut8/cut9; `:12` names two `TyroneNel/vllm` refs as `cpuchip/vllm`, and three hashes sit on untagged branches (1.1, §2.2). **Status 2026-10-05:** #275 re-exports all 50 files from one new tag, `qwen38/0.30-index-cut1` on `TyroneNel/vllm`, and `PATCHES.md` names it. The ancestor check below is not built | low (backfill) / already happened (tag drift) | do the backfill on the *next* natural re-export and cut a new export tag in the same PR — the cadence #189 set: re-export, tag, update `:12`; have the generator check that every marker hash is an ancestor of the tag `:12` names (needs network or a fork checkout, so a warn-only local step, not the CI freshness gate) |
| The generator becomes a second place the description is edited | low | the preamble is the only source; the generated block is marker-fenced with "do not edit" in the fence comment |
| Free-form `Upstream:`/`Retires-when:` values drift in style | low | the taxonomy line for `Kind:` is validated; the other two are prose by design (as the table is today) |
| The `Verify:` migration tempts bundling | medium | explicitly sequenced as a follow-up (2.3); this plan closes without touching verify.sh |
| A patch that deliberately has no preamble (`dflash2-z-adaptive-emitted` today) blocks the generator | low | the generator emits the row with an empty description cell and warns — blocking only on missing `Kind:`, never on missing prose. (2026-10-05: #275 requires all five headers. Every file has a `What:`, so no row has an empty cell) |

## 6. Done when

Status at e371b42 (2026-10-04) is in brackets. A second bracket gives the status on 2026-10-05.

1. `python scripts/patches_md.py && git diff --exit-code PATCHES.md` is a
   green step in `patch-integrity.yml`, in its own Python-only job
   (2026-10-04). [Not done: no generator, zero `PATCHES` matches in
   `.github/`.] [2026-10-05: done in #275, in review.]
2. Every patch's kind/upstream/retires-when/description is readable in its
   own preamble; deleting `PATCHES.md`'s table loses no information. The
   generator reads the order from `patches/apply.sh --list` and
   `--list --kvarn` (2026-10-04). [Not done: no patch has a `Kind:` line;
   `--list --kvarn` does not exist.] [2026-10-05: done in #275, in review.]
3. `README.md` contains no patch count; `patches/series`'s header names
   the consumers that exist; `git grep -n 'MR-DRAFT'` outside the review
   and remediation docs finds nothing; `PATCHES.md:12`'s tag carries every
   marker hash, and `:12` names the repo each ref is on (2026-10-04).
   [Partly done: e1459c7 did the first three. `:12` is not done (1.1).]
   [2026-10-05: done. #274 names the ref behind each hash, and #275 puts
   all 50 on one tag. Both are in review.]
4. `PATCHES.md`'s hand-written remainder is prose that is genuinely prose
   (retired history, port notes) — the most-touched file becomes one
   of the least-touched. [Not done: 9 touches in the last 40 commits.]
   [2026-10-05: not measurable until #275 merges.]
