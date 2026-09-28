# Let each preparation step report its own done-ness — remediation plan

Architecture review 2026-09-26, candidate 7 (Worth exploring). The full report
is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree on
2026-09-26.

**Re-verified 2026-09-28 against upstream/main @ 2522ef9 (vLLM 0.30.0).**

- d7c6572 (#195/#203) made `state()` *deeper*, not shallower: a torn index,
  config or shard now routes to `download` instead of crashing, via a new
  `intact()` shard-header check (`docker/prepare.sh:41-53`) — a fourth
  implementation of "is it done", living only in the heredoc.
- The #195 saga landed `prepare/atomic_publish.py` (74 lines): all six
  writers now publish through it, the safetensors index last — its docstring
  declares the index the commit point `state()` reads. The publish contract
  is finally written down; the done-*predicates* still live in shell.
- The string-split anti-pattern **spread**: `bench/test_prepare_state.py:37`
  (new, in CI at `patch-integrity.yml:23`) splits prepare.sh's heredoc the
  same way, and the unwired `bench/test_prepare_crash.py:365-367` makes a
  third call site. The extract-to-a-file fix below now pays off threefold.
- The writers remain probe-less (positional argv, no `--status`); verify.sh's
  model block is still a torch-free heredoc (`:111-261`). Every line map
  below re-made against this tree.

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
| shards intact | `:54-61` | every file in the index's weight_map parses as JSON **and** holds every byte its header declares — the `intact()` checker at `:41-53` (#195/#203); a torn index, config or shard routes to `download`, not a crash |
| `lm_head` | `:62` | `"lm_head.weight_packed" in idx` |
| `embed` | `:63` | any key `endswith("embed_tokens.weight_packed")` |
| `mtp` | `:64` | `"mtp.layers.0.mlp.down_proj.weight_packed" in idx` |
| `draft` | `:65` | `"mtp.draft_lm_head.weight_packed" in idx` **and** `mtp_draft_vocab_ids.pt` exists |
| `fast` | `:66-67` | sibling `-fast/model.safetensors.index.json` (FAST_VARIANT=0 opts out) |
| `dflash2` | `:68-69` | the DFlash2 drafter dir (optional; DFLASH2=0 silences) |

Dispatch is `prepare.sh:74-94`; the re-check after running is `:123-124`
(`LEFT=$(state | sed 's/\bdflash2\b//')` — dflash2 exempted from the final
gate, everything else must be clean).

### 1.2 The three authors of one protocol

1. **The writers** — the six prepare scripts know the real tensor names
   because they write them (`quant_lm_head.py:74-111`, `quant_embed.py:77-97`,
   `quant_mtp.py:85-109`, `build_draft_vocab.py:144-150`, `fetch_fast_variant.py:18`,
   `fetch_dflash2.py`). Since the #195 saga every write goes through
   `prepare/atomic_publish.py` (temp file, fsync, rename; the safetensors
   index **last** — its docstring declares the index the commit point
   `state()` reads), so the *publish* contract is now written down in a
   module the heredoc cannot import. Their CLI surface today is
   positional-only (`quant_lm_head.py:37`: `d = sys.argv[1]`) — no way to
   *ask* one anything.
2. **`prepare.sh`'s `state()`** — re-derives "done" by pattern-matching
   those names out of the index, in a language the writers don't run
   (a heredoc), in a file none of them can see.
3. **`verify.sh`'s model section (`:111-261`)** — a deliberately skeptical
   *third* implementation, and much deeper than name-matching: it re-derives
   vLLM's group resolution (`group_of` at `:168-179`, implementing
   `find_matched_target` semantics per the comment at `:149-152`), checks
   symmetry/zero-point consistency (`:126-179`, including the
   absent-key-means-symmetric rule at `:153-155`), and reads shard
   *headers* for declared-vs-stored geometry (`:216-225`).

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

### 1.4 The split has spread — and the same file shows the way out

The same file's second test (`:59-91`) already proves the
extract-and-replay remediation shape: it copies `docker/entrypoint.sh` and
`single-user/select_model.sh` into a fixture tree and executes *them* —
files, not string-splits. Meanwhile the split pattern **grew two new call
sites** since the review: `bench/test_prepare_state.py:37` (#195/#203; now
the second gated test, `patch-integrity.yml:23`) splits `docker/prepare.sh`'s
`state()` heredoc out by its `python - "$BASE" <<'EOF'` marker, and the
unwired `bench/test_prepare_crash.py:365-367` does the same to crash-test
the writers. Three parsers-of-parsers, each one `IndexError`-fragile to a
heredoc reformat — the extract-to-a-file fix (§2.2, §3.1) now pays off
threefold.

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
| `fetch_dflash2.py` | the drafter dir exists (stays optional) |

### 2.2 The verify block becomes a file

The deeper fix for 1.3: verify.sh's embedded model-check Python
(`verify.sh:111-261`) moves to a real module,
`scripts/verify_model.py` (stdlib+torch-free: it reads headers and JSON —
verified: the block imports only `json`, `os`, `sys`, `re` (`:157`) and
`struct` (`:219`, `:244`)). `verify.sh`
calls `$PY scripts/verify_model.py "$MODEL"`; the CI test runs the *file*
against its fixtures instead of string-splitting shell (`:52` becomes a
subprocess of the script path). The heredoc dies; the skeptical second
implementation (1.2.3) is **kept** — `verify_model.py` does not consume
`--status`; it re-derives semantics, per the review's own caution that
verify.sh's value is independence. (Its `import re` at `:157` and
header-only reads confirm no torch dependency: the file runs anywhere
Python does, no venv needed — a small, real portability win for the CI job.)

Name constants (`lm_head.weight_packed`, `mtp.draft_lm_head.*`, …) get one
home: candidate 4's `pipeline_core.py` if it exists by then, else a small
`prepare/prepared_names.py` that both the probes and `verify_model.py`
import — the two plans share exactly one module and nothing else.

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
the image build.

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
| negative: no string-split | `grep -n 'split..PY - .MODEL' bench/test_model_verification.py` → nothing | the parser-of-a-parser is gone |
| no-torch check | run `scripts/verify_model.py` under the system python (no venv) on a fixture | the CI job needs no venv for this test |

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| verify.sh starts trusting `--status` and loses its skepticism | low | written into the plan (2.2): `verify_model.py` re-derives; a code-review tripwire is the absence of any `subprocess … --status` in it |
| The heredoc extraction is not byte-faithful (shell interpolation, argv) | low | the heredoc uses a quoted `<<'EOF'` marker (no interpolation — verified at `verify.sh:113`) and reads the model dir from `sys.argv[1]`, which the file takes identically; CI parity is the acceptance gate |
| A probe drifts from its writer anyway (two functions, one file) | low | co-location is the mitigation: the predicate sits next to the write that produces it; PR review checks they move together; the fixture tests pin the contract |
| prepare.sh's re-check (`:123`) becomes redundant | low | it stays — orchestration should distrust its own dispatch (a probe crash mid-list must not read as success) |
| Sequencing with candidate 4 (whose module holds the names) | medium | either lands first: this plan's fallback `prepare/prepared_names.py` folds into `pipeline_core.py` when candidate 4 lands; no rework, one import-line change. `prepare/atomic_publish.py` (#195) is now the in-tree precedent for the same-dir shared module either way |

## 6. Done when

1. `docker/prepare.sh` contains no Python heredoc and no tensor names.
2. `bench/test_model_verification.py` contains no `split(` on verify.sh —
   it executes `scripts/verify_model.py` as a file.
3. Each of the six prepare scripts answers `--status` in fixture tests.
4. `verify.sh`'s model section is one call; every check it ran before the
   move still runs (CI parity).
5. Tensor names have one home, shared by writers, probes, and the
   independent verifier.
