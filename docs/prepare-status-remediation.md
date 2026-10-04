# Let each preparation step report its own done-ness — remediation plan

Architecture review 2026-09-26, candidate 7 (Worth exploring). The full report
is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree on
2026-09-26.

**Re-verified 2026-10-04 against upstream/main @ e371b42 (vLLM 0.30.0).**

**Status:** Not started. Worth exploring; not urgent. Start PR A after upstream PR #267 merges (it shifts verify.sh again).

Thirteen commits since d5e2a01. Four touch this plan's surface: 177ce26
(#237), e1459c7 (#242/#243/#244), bd6c5e2 (#240/#262) and accc8cf (#246).

- **Not started:** `scripts/verify_model.py`, `prepare/prepared_names.py`
  and `prepare/pipeline_core.py` do not exist at e371b42. No prepare script
  takes `--status`. None of the five §6 items holds.
- **Unchanged:** `git diff d5e2a01 e371b42` is empty for
  `docker/prepare.sh`, `bench/test_model_verification.py`,
  `bench/test_prepare_state.py`, `build_draft_vocab.py`,
  `fetch_fast_variant.py`, `fetch_dflash2.py` and `atomic_publish.py`.
  Every `prepare.sh` cite and every `test_model_verification.py` cite
  still holds. `patch-integrity.yml:21` and `:23` hold too: the new
  `kvarn-torch-gate` job starts at `:25`.
- **verify.sh shifted +1, content the same:** e1459c7 removes 4 lines above
  the model block (`patches/apply.sh --list` replaces the series loop, the
  dflash2-backport skip goes, the KVarN checks become exact). bd6c5e2 adds
  5 (the fp16-dequant check at `:123-127`). The model block has no diff.
  New lines: guard `:129` → `:130`, section `:130-280` → `:131-281`,
  marker `:132` → `:133`, argv `:134` → `:135`, imports
  `:133`/`:176`/`:238`/`:263` → `:134`/`:177`/`:239`/`:264`, comment
  `:168-171` → `:169-172`, `group_of` `:187-197` → `:188-198`, zero-point
  `:145-224` → `:146-225` (helpers `:146-199`, checks `:200-225`),
  absent-key comment `:173-175` → `:174-176`, `.get("symmetric", True)`
  `:184` → `:185`, lm_head geometry `:236-247` → `:237-248`, duplicate
  scan `:263-278` → `:264-279`, `-d …-fast` `:285` → `:286`. #237 adds 9
  lines below the block (the key check, now `:331-339`), so the other
  `$PY - "$MODEL" …` heredocs move `:293`/`:388`/`:399` →
  `:294`/`:398`/`:409`. The split marker first matches at `:133`. Ran
  `python3 bench/test_model_verification.py` (2 tests OK) and
  `python3 bench/test_prepare_state.py` (1 test OK) in the e371b42 tree.
- **Other shifts:** `Dockerfile:39` → `:34` (e1459c7 replaces the inline
  apply loop with `bash patches/apply.sh`). #246 adds a docstring paragraph,
  the `quant_schema` import and the `load_config(d)` call to the writers,
  and drops their old config read. `quant_lm_head.py:37` → `:42` and
  `:74-111` → `:82-117`. `quant_embed.py:77-97` → `:85-103`.
  `quant_mtp.py:35,37` → `:40,42` and `:85-109` → `:94-116`. #246 also
  adds a `reject` step to `bench/test_prepare_crash.py`, so its split
  moves `:366-367` → `:431-432`.
- **#246 adds no fourth author:** `prepare/quant_schema.py` (71 lines,
  new) checks whether a dir *can* be prepared. `load_config()` requires
  `quant_method` to be compressed-tensors when set (`:52-58`), an `ignore`
  list (`:60-62`) and `config_groups.group_0` (`:64-67`). It never reads
  the index. A raw base dir and a prepared one both pass. It does not say
  which steps are done, so §1.2's three authors stand. Three of the six
  dispatched writers call it (`quant_lm_head.py:45`, `quant_embed.py:43`,
  `quant_mtp.py:56`), and `quant_heads_stream.py:74` does too.
- **A gap #246 leaves:** `quant_embed.py:85-86` runs `backup_once` and
  `save_tensors`, and then `:89` reads `qc["config_groups"]["group_1"]`.
  `quant_lm_head.py:109` creates `group_1`. `quant_schema.py:65` checks
  only `group_0`. So `quant_embed.py` on a dir without the lm_head step
  still raises `KeyError` after the shard write. `prepare.sh` dispatches
  lm_head first (`:83`, embed at `:84`), so the pipeline does not hit it.
  The embed step's real precondition is "lm_head done", which a probe can
  state (§2.1).
- **#237 adds a fourth split site:** `bench/test_no_key_bind.sh:35`
  extracts verify.sh's key check (`:331-339`) with `sed -n` and runs it
  with `eval`. It is a different block from the model check. No workflow
  runs the test (`grep -rn no_key .github` finds nothing).
- **PR #267 impact:** the PR is open (head 9ab86f8, `gh pr view 267`). It
  changes only verify.sh, +7/-1, and replaces the power-limit line at
  `:55`. That is above the model block, so when it merges every verify.sh
  cite after `:55` moves +6 more: guard `:136`, section `:137-287`, marker
  `:139`. Since the PR's base (e1459c7), main changed verify.sh only at
  bd6c5e2's hunk at `:120`, so the two do not touch. The CI split matches
  the marker text, not a line number, so the test is not affected.
- **Plan changes in this pass:** body cites re-made at e371b42 in §1.2,
  §1.4, §2.2, §3.1 and §5. §2.1 says where `--status` runs relative to
  `load_config` and sets exit codes 0/1/2. §2.2 names `quant_schema.py` as
  an option for the names module. §1.4 counts `test_no_key_bind.sh` as the
  fourth split site. §1.3 notes that it still holds.

**Previously re-verified 2026-09-29 against upstream/main @ d5e2a01 (vLLM 0.30.0).**
Seven commits since 2522ef9; only #219 (8cf642e) touches this plan's surface.

- **Unchanged:** `docker/prepare.sh`, `prepare/`, `bench/` and
  `.github/workflows/patch-integrity.yml` have no diff in
  `git diff 2522ef9 upstream/main --stat`, so every `prepare.sh` line
  cited here (`:23-24`, `:26-71`, `:31-36`, `:41-61`, `:62-69`, `:74-94`,
  `:95-122`, `:123-124`) was re-read and still holds. None of the plan's
  steps has landed: `scripts/verify_model.py`, `prepare/prepared_names.py`
  and `prepare/pipeline_core.py` do not exist, and no prepare script takes
  `--status`.
- **Shifted, not changed:** #219 added `verify.sh --wait` (19 lines of
  argument parsing above the model block, and a `/health` poll in the live
  section). The model block moves down by exactly 19 lines and its content
  is the same: `:111-261` → `:130-280`, marker `:113` → `:132`, imports
  `:114`/`:157`/`:219`/`:244` → `:133`/`:176`/`:238`/`:263`. The
  `$PY - "$MODEL" <<'EOF'` split in `test_model_verification.py:19` still
  hits that block first (the other `$PY - "$MODEL" …` heredocs at `:293`,
  `:388`, `:399` have extra argv before `<<`). Ran
  `python3 bench/test_model_verification.py` (2 tests OK) and
  `python3 bench/test_prepare_state.py` (1 test OK).
- **No fifth author:** `scripts/hq-doctor.sh` (new in #219, 80 lines)
  checks config, `.env`/key presence, GPU, the port, `/health`, and docker
  log errors. It never opens a model dir. The Makefile (new) calls
  `verify.sh` or compose. The `docker/entrypoint.sh` hunk only adds three
  stage-banner `echo`s. The other model-dir checks outside the three
  authors predate 2522ef9 and are unchanged. `single-user/select_model.sh:4`
  (#126) and `verify.sh:285` treat `-d …-fast` (the dir exists) as "the
  fast variant is there", while `state()` (`prepare.sh:66`) wants its
  index file.
- **Corrected in this pass (was wrong at 2522ef9 too):** `atomic_publish.py`
  is used by five of the six dispatched scripts. `fetch_dflash2.py` writes
  through `snapshot_download` and imports nothing from it. The claim was
  "all six writers". Several §1.2/§1.3 ranges were also tightened (see
  those entries). The §4 "no string-split" grep never matched and has been
  fixed. §3.1's `verify.sh --install` gate turns out to skip the model block.

**Previously re-verified 2026-09-28 against upstream/main @ 2522ef9 (vLLM 0.30.0).**

- d7c6572 (#195/#203) made `state()` *deeper*, not shallower: a torn index,
  config or shard now routes to `download` instead of crashing, via a new
  `intact()` shard-header check (`docker/prepare.sh:41-53`) — a fourth
  implementation of "is it done", living only in the heredoc.
- The #195 saga landed `prepare/atomic_publish.py` (74 lines). The index is
  published last, and the docstring declares it the commit point `state()`
  reads. *(Corrected 2026-09-29: this said "all six writers now publish
  through it". Five of the six dispatched scripts do; `fetch_dflash2.py`
  does not. `quant_heads_stream.py` and the two template passes also
  import it, for 8 importers in total.)* The publish contract is finally
  written down; the done-*predicates* still live in shell.
- The string-split anti-pattern **spread**: `bench/test_prepare_state.py:37`
  (new, in CI at `patch-integrity.yml:23`) splits prepare.sh's heredoc the
  same way, and the unwired `bench/test_prepare_crash.py:366-367` (cited as
  `:365-367`; the split starts at `:366`) makes a
  third call site. The extract-to-a-file fix below now pays off threefold.
- The writers remain probe-less (hand-parsed `sys.argv`, no `--status`); verify.sh's
  model block is still a torch-free heredoc (`:111-261` at 2522ef9;
  `:130-280` at d5e2a01). Every line map below re-made against this tree.

**Scope.** The "is step X done?" protocol of the model-preparation pipeline:
`docker/prepare.sh`'s `state()`, the six prepare scripts, `verify.sh`'s model
checks, and the CI test that exercises them. **Not in scope:** the quant
mechanisms themselves (candidate 4's plan — its `pipeline_core.py` constants
are the natural home for the names this plan's probes report on; the two
plans are sequenced so either can land first).

## 1. Current state, precisely

### 1.1 What "prepared" means today (all verified lines)

`docker/prepare.sh:26-71` — a Python heredoc inside the shell script —
decides which steps remain:

| check | line | what it looks for |
|---|---|---|
| base files | `:34-36` | config.json, model.safetensors.index.json, **tokenizer.json**, tokenizer_config.json (tokenizer.json is load-bearing: without it the dir stays servable-looking until "failed to tokenize reasoning strings" at startup, per the comment at `:31-33`) |
| shards intact | `:54-61` | config.json and the index parse as JSON (`:54-59`), **and** every shard the index's weight_map names holds every byte its header declares (`:60-61`, via the `intact()` checker at `:41-53`, #195/#203). A torn index, config or shard routes to `download`, not a crash. *(Wording corrected 2026-09-29: this said every weight_map file "parses as JSON". Shards are checked through their safetensors header.)* |
| `lm_head` | `:62` | `"lm_head.weight_packed" in idx` |
| `embed` | `:63` | any key `endswith("embed_tokens.weight_packed")` |
| `mtp` | `:64` | `"mtp.layers.0.mlp.down_proj.weight_packed" in idx` |
| `draft` | `:65` | `"mtp.draft_lm_head.weight_packed" in idx` **and** `mtp_draft_vocab_ids.pt` exists |
| `fast` | `:66-67` | sibling `-fast/model.safetensors.index.json` (FAST_VARIANT=0 opts out) |
| `dflash2` | `:68-69` | `Qwen3.8-27B-DFlash2-W4A16/model.safetensors` beside the base dir (optional; DFLASH2=0 drops the step, and a failed fetch only warns, `:90-92`) |

Dispatch is `prepare.sh:74-94`; the re-check after running is `:123-124`
(`LEFT=$(state | sed 's/\bdflash2\b//')` — dflash2 exempted from the final
gate, everything else must be clean).

### 1.2 The three authors of one protocol

1. **The writers** — the six prepare scripts know the real tensor names
   because they write them (`quant_lm_head.py:82-117`, `quant_embed.py:85-103`,
   `quant_mtp.py:94-116`, `build_draft_vocab.py:141-150`, `fetch_fast_variant.py:49-52`,
   `fetch_dflash2.py:18`). *(2026-10-04: #246 shifted the three quant
   ranges. At d5e2a01 they were `:74-111`, `:77-97` and `:85-109`.)* *(2026-09-29: `build_draft_vocab.py` was cited
   `:144-150` and the tensor names start at `:141`. `fetch_fast_variant.py`
   was cited `:18`, which is only its `from atomic_publish import publish`;
   the index-last copy loop is at `:49-52`.)* Since the #195 saga five of the six
   write through `prepare/atomic_publish.py` (temp file, fsync, rename; the
   safetensors index **last**, and the docstring at `:12-15` declares the
   index the commit point `state()` reads). `fetch_dflash2.py` is the
   exception: it calls `snapshot_download` straight into the target dir
   (`:18`). This was "every write" before 2026-09-29. So the *publish*
   contract is now written down in a module the heredoc does not import.
   Their CLI surface is hand-parsed `sys.argv`: a positional dir
   (`quant_lm_head.py:42`: `d = sys.argv[1]…`) plus a few ad-hoc flags
   (`quant_mtp.py:40,42` `--bits`/`--keep-fc`, `build_draft_vocab.py:38-40`
   `--n`/`--corpus`/`--ids`, `fetch_dflash2.py:13` `--bf16`). There is no
   argparse and no way to *ask* one anything. It was called
   "positional-only" before 2026-09-29. *(2026-10-04: these were
   `quant_lm_head.py:37` and `quant_mtp.py:35,37` at d5e2a01.)*
2. **`prepare.sh`'s `state()`** — re-derives "done" by pattern-matching
   those names out of the index, in a language the writers don't run
   (a heredoc), in a file none of them can see.
3. **`verify.sh`'s model section (`:131-281`; `:130-280` at d5e2a01,
   `:111-261` at 2522ef9; #219's `--wait` parsing shifted it +19, and
   e1459c7/bd6c5e2 shifted it +1)** — a deliberately skeptical
   *third* implementation, and much deeper than name-matching: it re-derives
   vLLM's group resolution (`group_of` at `:188-198`, implementing
   `find_matched_target` semantics per the comment at `:169-172`), checks
   symmetry/zero-point consistency (`:146-225`: rules and helpers
   `:146-199`, checks `:200-225`, including the absent-key-means-symmetric
   rule, commented at `:174-176` and coded as `.get("symmetric", True)` at
   `:185`), and reads shard *headers*: the lm_head geometry is compared
   declared-vs-stored at `:237-248`, and every shard's keys are scanned for
   duplicates at `:264-279`. *(2026-10-04: every number in this item moved
   +1 from d5e2a01.)* *(Corrections 2026-09-29 against the old
   2522ef9 numbers. `group_of` was `:168-179`, but the function was
   `:168-178`. Zero-point was `:126-179`, which stopped at the helpers
   before the checks. The absent-key rule was `:153-155`, one line early.
   Headers were `:216-225`, which cut off the comparison.)*

The drift risk is precise: rename a tensor in a writer (say
`weight_packed` → a new layout) and you must edit three files in two
languages — writer, heredoc, verify block. Miss the heredoc and
`prepare.sh` silently re-runs or skips a step *and reports success*
(`:123-124` only catches steps `state()` still knows how to name).

### 1.3 The CI test parses the parser

`bench/test_model_verification.py` — the first of two bench tests in CI
(`patch-integrity.yml:21`; the second is 1.4's problem) — exercises
verify.sh's model block by
**string-splitting it out of the shell source** (`:18-19`):

```python
source = (REPO / "verify.sh").read_text(encoding="utf-8")
block = source.split('$PY - "$MODEL" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
```

It then runs `block` as a subprocess against synthetic fixture dirs
(`:27-51`: tempfile model dirs with hand-built safetensors headers) and
asserts exit codes plus an empty-stderr guard (`:52-55`: "A crash must not
masquerade as a successful rejection case"). The test is good; its coupling
is not: any reformat of verify.sh's heredoc (indentation, marker style)
fails CI with an `IndexError`, not a test failure. The fixture style,
however, is exactly right and is what §2's probes are tested with.
*(2026-10-04: still true at e371b42. The file has no diff since d5e2a01,
the split first matches `verify.sh:133`, and the test passes.)*

### 1.4 The split has spread — and the same file shows the way out

The same file's second test (`:59-91`) already proves the
extract-and-replay remediation shape. It copies `docker/entrypoint.sh`
(rewriting only its `/app` root, `:73-74`) and
`single-user/select_model.sh` into a fixture tree and executes *them*:
files, not string-splits. Meanwhile the split pattern **grew two new call
sites** since the review: `bench/test_prepare_state.py:37` (#195/#203; now
the second gated test, `patch-integrity.yml:23`) splits `docker/prepare.sh`'s
`state()` heredoc out by its `python - "$BASE" <<'EOF'` marker, and the
unwired `bench/test_prepare_crash.py:431-432` (was `:366-367` at d5e2a01,
cited as `:365-367` before that) does the same to crash-test
the writers. Three parsers-of-parsers, each one `IndexError`-fragile to a
heredoc reformat — the extract-to-a-file fix (§2.2, §3.1) now pays off
threefold.

*(2026-10-04)* #237 adds a fourth split site. `bench/test_no_key_bind.sh:35`
extracts verify.sh's key check (`:331-339`) with
`sed -n "/^if \[ -s api_key.txt \] || \[ -n/,/^fi\$/p"` and runs it with
`eval`. It splits a different block from the model check, and no workflow
runs it. This plan's fix does not remove it. If the first line of the key
check changes, `sed` prints nothing, and all four `vk` checks fail.

## 2. The design, in two halves

### 2.1 `--status` on every prepare script

Each of the six scripts gains a probe mode:

```text
python prepare/quant_lm_head.py DIR --status
    exit 0 + one stdout line if DIR already has this step's result
    ("lm_head.weight_packed present in index"),
    exit 1 + the reason if not ("no lm_head.weight_packed in weight_map").
```

The predicate lives **in the file that performs the step** — the writer
reports on its own contract, so a tensor rename is a one-file edit (1.2).
`prepare.sh`'s `state()` becomes a loop over the probes and keeps only what
is genuinely orchestration: the flock (`:23-24`), the dispatch table
(`:81-94`), the download check, the harden/translate passes (`:95-122`),
and the final re-check (`:123-124`, now also probe-driven). The heredoc,
and with it every tensor name in shell, is deleted.

One depth decision is new since the review: `state()`'s `intact()`
(`:41-53`) now goes *deeper* than the index — it re-reads shard headers for
torn writes. `atomic_publish.py`'s docstring blesses the index as the
commit point, so a `--status` probe can trust it; the torn-file depth
belongs to the download branch's own check (or a shared `intact()` the
probes import), not to every predicate.

Probe map (writer → its predicate, from 1.1):

| script | --status checks |
|---|---|
| `quant_lm_head.py` | `lm_head.weight_packed` in the index |
| `quant_embed.py` | an `embed_tokens.weight_packed` key exists |
| `quant_mtp.py` | `mtp.layers.0.mlp.down_proj.weight_packed` in the index |
| `build_draft_vocab.py` | `mtp.draft_lm_head.weight_packed` in the index **and** `mtp_draft_vocab_ids.pt` present |
| `fetch_fast_variant.py` | sibling `-fast/` index exists |
| `fetch_dflash2.py` | the drafter's `model.safetensors` exists, as `prepare.sh:68` checks (stays optional). The old text said "the drafter dir exists", and the heredoc checks the file. |

*(2026-10-04)* `--status` and #246's `load_config`. Three of the probed
scripts now call `load_config(d)` at module level, right after they read
the dir from argv (`quant_lm_head.py:42`/`:45`, `quant_embed.py:40`/`:43`,
`quant_mtp.py:53`/`:56`). `load_config` prints one line to stdout on
success (`quant_schema.py:69`). It refuses with `sys.exit(message)`, which
exits 1 (`:29`, `:43`, `:45`, `:49`). Both clash with the contract above:
"one stdout line", and exit 1 means "not done". So:

- `--status` runs and answers before `load_config(d)`.
- Exit codes: 0 done, 1 not done, 2 cannot be done here (the schema check
  fails). Exit 2 needs a form of the check that returns instead of
  exiting. `quant_schema.py` has none today.
- The `quant_embed.py` probe can also report its real precondition:
  `:89` clones `config_groups.group_1`, which `quant_lm_head.py:109`
  creates, and `load_config` checks only `group_0` (`quant_schema.py:65`).

### 2.2 The verify block becomes a file

The deeper fix for 1.3: verify.sh's embedded model-check Python
(`verify.sh:131-281`; `:130-280` at d5e2a01, `:111-261` at 2522ef9) moves to a real module,
`scripts/verify_model.py` (stdlib+torch-free: it reads headers and JSON —
verified: the block imports only `json`, `os`, `sys` (`:134`), `re` (`:177`) and
`struct` (`:239`, `:264`). At 2522ef9 these were `:114`, `:157`, `:219`,
`:244`; #219 shifted them by +19 and e1459c7/bd6c5e2 by +1). `verify.sh`
calls `$PY scripts/verify_model.py "$MODEL"`; the CI test runs the *file*
against its fixtures instead of string-splitting shell (`:52` becomes a
subprocess of the script path). The heredoc dies; the skeptical second
implementation (1.2.3) is **kept** — `verify_model.py` does not consume
`--status`; it re-derives semantics, per the review's own caution that
verify.sh's value is independence. (Its `import re` at `:177` and
header-only reads confirm no torch dependency: the file runs anywhere
Python does, no venv needed — a small, real portability win for the CI job.)

Name constants (`lm_head.weight_packed`, `mtp.draft_lm_head.*`, …) get one
home: candidate 4's `pipeline_core.py` if it exists by then, else a small
`prepare/prepared_names.py` that both the probes and `verify_model.py`
import — the two plans share exactly one module and nothing else.

*(2026-10-04)* `prepare/quant_schema.py` (#246) is an option for that home
instead of a new file. It imports only `json` and `sys`. Three of the six
dispatched writers already import it (`quant_lm_head.py:34`,
`quant_embed.py:34`, `quant_mtp.py:35`). `build_draft_vocab.py`,
`fetch_fast_variant.py` and `fetch_dflash2.py` do not, so they would gain
one import. `verify_model.py` would import the constants only, never
`load_config`.

### 2.3 What does not change

verify.sh's *semantics* (every check it runs today, byte-for-byte — the
extraction is mechanical and the acceptance gate is CI parity); the
harden/translate passes (they are post-steps, not state probes);
`verify.sh`'s patch ladder (candidate 1's surface); the docker entrypoint's
ordering (prepare → select_model → verify).

## 3. Rollout

### 3.1 PR A — extract `scripts/verify_model.py`

Move the block verbatim (adjusting only argv intake); `verify.sh` calls it;
`test_model_verification.py:52` points at the file; the string-split at
`:18-19` is deleted. Acceptance: CI green with zero test-logic change —
the same fixtures, the same expected codes; `verify.sh --install` green in
the image build (`Dockerfile:34`). That gate only proves the rest of
verify.sh still runs: `--install` skips the model block entirely
(`INSTALL=1` fails the `if [ $INSTALL = 0 ]` guard at `verify.sh:130`),
so model-check parity rests on the CI test alone. *(Noted 2026-09-29.
2026-10-04: these were `Dockerfile:39` and `verify.sh:129` at d5e2a01.)*

### 3.2 PR B — the probes

`--status` on all six scripts; `prepare.sh` probe-driven; probe tests added
to `test_model_verification.py` in its existing fixture style (per script:
a done fixture → exit 0; a not-done fixture → exit 1 with the reason
string). Acceptance: `bash docker/prepare.sh` on a prepared model copy
prints "model ready" in seconds (idempotence preserved); a model dir with
one step deleted re-runs exactly that step.

## 4. Test plan

| test | how | proves |
|---|---|---|
| extraction parity | the existing CI test, unchanged expectations, now against the file | 2.2 changed plumbing, not semantics |
| probe matrix | per script: done / not-done fixtures (the `:27-51` style) | predicates live in their writers and work |
| prepare.sh end-to-end | a staged model dir missing one artifact; `prepare.sh` runs only that step | the orchestration loop consumes probes correctly |
| re-check | `prepare.sh:123-124` with a probe forced to fail | the final gate still refuses |
| negative: no string-split | `grep -n 'split..\$PY - "\$MODEL' bench/test_model_verification.py` → nothing | the parser-of-a-parser is gone. *(Fixed 2026-09-29. The old pattern `'split..PY - .MODEL'` never matched, even against today's `:19` (grep rc=1), because it has no slot for the `$` before `PY` or the `"$` before `MODEL`. The check passed vacuously. The pattern here matches `:19` today, rc=0.)* |
| no-torch check | run `scripts/verify_model.py` under the system python (no venv) on a fixture | the CI job needs no venv for this test |

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| verify.sh starts trusting `--status` and loses its skepticism | low | written into the plan (2.2): `verify_model.py` re-derives; a code-review tripwire is the absence of any `subprocess … --status` in it |
| The heredoc extraction is not byte-faithful (shell interpolation, argv) | low | the heredoc uses a quoted `<<'EOF'` marker (no interpolation; verified at `verify.sh:133`, which was `:132` at d5e2a01 and `:113` at 2522ef9) and reads the model dir from `sys.argv[1]` (`:135`), which the file takes identically; CI parity is the acceptance gate |
| A probe drifts from its writer anyway (two functions, one file) | low | co-location is the mitigation: the predicate sits next to the write that produces it; PR review checks they move together; the fixture tests pin the contract |
| prepare.sh's re-check (`:123`) becomes redundant | low | it stays — orchestration should distrust its own dispatch (a probe crash mid-list must not read as success) |
| Sequencing with candidate 4 (whose module holds the names) | medium | either lands first: this plan's fallback `prepare/prepared_names.py` folds into `pipeline_core.py` when candidate 4 lands; no rework, one import-line change. `prepare/atomic_publish.py` (#195) is now the in-tree precedent for the same-dir shared module either way. *(2026-10-04: `prepare/quant_schema.py`, #246, is a second one; §2.2.)* |

## 6. Done when

1. `docker/prepare.sh` contains no Python heredoc and no tensor names.
2. `bench/test_model_verification.py` contains no `split(` on verify.sh —
   it executes `scripts/verify_model.py` as a file.
3. Each of the six prepare scripts answers `--status` in fixture tests.
4. `verify.sh`'s model section is one call; every check it ran before the
   move still runs (CI parity).
5. Tensor names have one home, shared by writers, probes, and the
   independent verifier.
