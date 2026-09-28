# Deepen the preparation pipeline's shared core — remediation plan

Architecture review 2026-09-26, candidate 4 (Worth exploring). The full report
is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree on
2026-09-26.

**Re-verified 2026-09-28 against upstream/main @ 2522ef9 (vLLM 0.30.0).**

- The #195 atomic-publish saga (2f9b1e6 #198, fba2bd2 #199, a2a904c #200,
  850d2ad #201, 570ec05 #202, d7c6572 #203, 263a3ed #212) landed
  `prepare/atomic_publish.py` (74 lines: `backup_once`/`publish`/`write_json`/
  `write_text`/`save_tensors` — temp file + rename + fsync, the safetensors
  index last) and rewired all eight `prepare/` writers to it. `drafter/` was
  not touched. Section 1.2 is rewritten accordingly: the I/O mechanism is now
  shared, the write-tail *contract* is not.
- Standing, with line citations updated: the five RTN copies and the
  1e-10/1e-8 epsilon split (1.1), the fp16/bf16 encodings (1.3), the six-site
  MTP list (1.4), the two draft-head writers (1.5 — now with an atomicity
  asymmetry), the four variant-dir assemblies, the five drafter shims (1.7).
- The backup census is now six suffixes over one shared `backup_once` (1.6);
  `quant_heads_stream.py`'s `.bak-orig` stays a hardlink by design.
- `drafter/gptq_lm_head.py:83-86`'s inode-trap comment now describes a writer
  that no longer truncates in place — kept below as history, with the drift
  marked (1.5).
- New per-script duplication the saga added: the killed-run resume check
  ("already packed … completing an interrupted run") exists in five copies
  (1.2); `quant_embed.py` writes config/index with no backup of its own,
  relying on `quant_lm_head.py`'s `.bak-quant` (1.6).

**Scope.** `prepare/` (the one-time model-preparation scripts) and `drafter/`
(the drafter-training and requantization tooling) — the shared quantization
mechanisms they re-implement. **Not in scope:** the GPTQ implementation itself
(`drafter/gptq_utils.py` — a real module with real math, kept), the capture/
training research scripts, the done-ness protocol (candidate 7's plan, which
shares this plan's constants module).

## 1. Current state, precisely

### 1.1 The RTN quant math: five copies, two invisible variants

The same two lines, byte-identical except where noted:

| site | lines | variant |
|---|---|---|
| `prepare/quant_lm_head.py` | `:61-62` | QMAX=127, clamp min **1e-10**, keepdim |
| `prepare/quant_embed.py` | `:61-62` | identical to quant_lm_head (verified) |
| `prepare/quant_mtp.py` | `:74-75` | identical |
| `prepare/quant_heads_stream.py` | `:84-85` | chunked (per-block), lowercase qmax/s |
| `drafter/export_mtp.py` | `:112-113` | the draft-head RTN, identical to quant_lm_head |

A shared `rtn_quantize` **already exists** at `drafter/gptq_utils.py:77-84` —
but (a) it lives in `drafter/`, which `prepare/` scripts cannot import (no
path shim reaches it; only drafter files import `gptq_utils`), and (b) it is
not actually identical: clamp min **1e-8** (vs 1e-10, at `:82`; the same
floor in `gptq_quantize` at `:51,:55`), no keepdim, parameterized bits. The
epsilon discrepancy is exactly the class of difference that is invisible
until a produced checkpoint differs in the third decimal of a scale — today
there is no place where that difference is even written down.

### 1.2 The compressed-tensors write tail: shared I/O, per-script contract

Since the #195 saga (2026-09-27) the I/O half of the tail is one module,
`prepare/atomic_publish.py` (74 lines): `publish` (fsync, rename, dir-fsync,
`:36-41`), `write_json`/`write_text`/`save_tensors` (`:44-64`), `backup_once`
(first-wins, atomically copied, `:67-74`), the index-last ordering written
down in its docstring (`:12-15`; the consumer is `docker/prepare.sh`'s
`state()`, which reads only the index, `:26,:62-69`, and sends torn files to
re-download — `intact()`, `:41-53`, #203). All eight `prepare/` writers
import it (`quant_lm_head.py:29`, `quant_embed.py:29`, `quant_mtp.py:30`,
`quant_heads_stream.py:47`, `build_draft_vocab.py:35`,
`fetch_fast_variant.py:18`, `harden_chat_template.py:30`,
`translate_chat_template.py:40`); each still sequences the calls itself
("The index is the commit point, so it goes last" — `quant_lm_head.py:106-111`
and twins). `prepare/README.md:62-66` documents the protocol and its
crash-injection test (`bench/test_prepare_crash.py`).

The contract half — `pack_to_int32(q, BITS)` → `scale.to(dtype)` →
`weight_shape = tensor([out_f, in_f], int64)` → `weight_map` edit →
`config.json` group clone → ignore-list edit — is still assembled per script,
nine sites in eight files (`export_mtp.py` carries two), plus one slice
assembly:

| site | pack/scale/shape | weight_map | config group | ignore edit |
|---|---|---|---|---|
| `prepare/quant_lm_head.py` | `:69-72` | `:108-110` | clone `group_0`→`group_1`, `:96-103` | drop `lm_head`, `:81` (+ MTP list, 1.4) |
| `prepare/quant_embed.py` | `:69-72` | `:94-96` | clone `group_1`→`group_2`, `:83-90` | — |
| `prepare/quant_mtp.py` | `:79-81` | `:105-108` | clone `group_0`→`group_3`, `:93-100` | `:92` |
| `prepare/quant_heads_stream.py` | `:205-208` (heads), `:249-251` (MTP) | `:218-222`, `:262-265` | `group()` factory `:274-286` → `group_1..3`, `:290-294` | `:289` |
| `drafter/export_mtp.py` (GPTQ MTP) | `:84-86` | `:87-89` | `group_3`, `:98-101` | `:97` |
| `drafter/export_mtp.py` (draft head) | `:122-124` | `:127-128` | `group_4` when HBITS≠BITS, `:136-138` | — |
| `drafter/requant_mtp_gptq.py` | `:51-53` | (index copied verbatim, `:31`) | `group_3` num_bits in place, `:60-62` | — |
| `drafter/gptq_lm_head.py` | `:100-102` | index rewritten, `:106` | `group_1` num_bits in place, `:108` | — |
| `drafter/quant_dflash2.py` | `:79-81` | (single-file model, no index) | groups **fabricated**, not cloned, `:103-116` | built fresh, `:100-102` |
| `prepare/build_draft_vocab.py` (slice; no pack call) | `:141-143` | `:148-149` | — | — |

Ten files name `weight_packed` (grep): the nine writers above plus
`drafter/train_mtp.py:203-204`, which only reads the packed entries to find
shards. Three details the table hides: the group ordinals are hardcoded per
script and chained (`quant_embed.py:83` clones the `group_1` that
`quant_lm_head.py` wrote — an ordering dependency); `drafter/` writes are
still non-atomic (`save_file`/`json.dump` in place: `export_mtp.py:91-93,102,
125,129`, `requant_mtp_gptq.py:57,63`, `gptq_lm_head.py:105-109`,
`quant_dflash2.py:94,117`); and the killed-run resume check ("already packed
… completing an interrupted run") that the atomic protocol needs is itself a
per-script copy, five times: `quant_lm_head.py:52-56`, `quant_embed.py:52-56`,
`quant_mtp.py:64-68`, `quant_heads_stream.py:187-193` and `:238-244`.

### 1.3 The one difference that matters is invisible

Scale dtype: fp16 for lm_head, bf16 for embed_tokens. Encoded as:
- `prepare/quant_lm_head.py:70-71` — a **comment** ("linear layers use fp16
  scales in this checkpoint") and `.to(torch.float16)`;
- `prepare/quant_embed.py:70-71` — a **comment** ("the embedding path creates
  scales in params_dtype (bf16), unlike the linears") and `.to(torch.bfloat16)`;
- `prepare/quant_heads_stream.py:178` — as **data**:
  `for key, scale_dtype in ((lm_key, torch.float16), (emb_key, torch.bfloat16))`
  (plus a fourth restatement as a comment at `:206-207`).

Three encodings of one fact across three scripts (four statements; only the
data tuple is machine-checkable). Get it wrong and the checkpoint still loads
— vLLM casts — with a silent quality cost per group scale.

### 1.4 The MTP tensor list: six sites, four spellings

| site | form |
|---|---|
| `prepare/quant_lm_head.py:84-93` | inline tuple, inside the **ignore-list repair** — the script that quants lm_head secretly owns adding `mtp.*` to the ignore list, with the comment "The MTP draft head is stored in bf16 but missing from the ignore list, which breaks loading when speculative decoding is enabled (single-user mode)" (`:82-83`; a side job in the wrong file) |
| `prepare/quant_mtp.py:38-46` | `MTP_LINEARS` (`mtp.fc` drops out under `--keep-fc`) |
| `prepare/quant_heads_stream.py:56-64` | `MTP_LINEARS` (same `--keep-fc` conditional) |
| `drafter/export_mtp.py:67-69` | `MTP_LINEARS` |
| `drafter/requant_mtp_gptq.py:20-22` | named **`LIN`** |
| `drafter/train_mtp.py:397-404` | dict literal mapping the same names to modules |

Same eight names (`mtp.fc`, three MLP projections, four attention
projections — no `eh_proj`, no `draft_lm_head`), six sites, four spellings.

### 1.5 The draft-head contract has two writers, and history lives in comments

The contract — `mtp.draft_lm_head.weight_packed/weight_scale/weight_shape` in
`model_extra_tensors.safetensors` + `mtp_draft_vocab_ids.pt` + weight_map
entries — is consumed by `patches/qwen3_5-mtp-draft-vocab.patch:33-42`
(reads the ids file, builds `ParallelLMHead(numel)`). It is written twice:
- `prepare/build_draft_vocab.py:114-150` — slices rows out of the **already
  packed** lm_head (slice `:114-127`; the canonical writer; fused to a
  corpus-counting data tool, `:45-112`, counting plain-text corpora since
  bc6d9e7 #213, `:63-71`). Since 850d2ad #201 it writes **atomically**:
  `backup_once(..., ".bak-draft")` + `save_tensors` (`:140-144`), the ids by
  `torch.save` to `.tmp` + `publish` (`:145-146`), the weight_map edit and
  the index **last** (`:147-150`),
- `drafter/export_mtp.py:108-130` — requantizes a **trained bf16** draft head
  with its own RTN copy + write tail (the fifth RTN site, 1.1), still with
  plain `save_file` + `json.dump` in place (`:125,:129`) — **not** atomic;
  `drafter/` never got atomic_publish,
and `export_mtp.py:104-106` *also* subprocesses the first one when the
checkpoint lacks a trained head. Two writers with a real semantic difference
(slice quantized rows vs quantize a bf16 head) and, since #201, an atomicity
asymmetry (only the slice writer survives a kill mid-write), neither
documented next to the contract.

The variant-dir assembly (hardlink shards + copy config family) exists 4×:
`prepare/fetch_fast_variant.py:31-39` (copies atomic since 570ec05 #202),
`drafter/export_mtp.py:24-33`, `drafter/gptq_lm_head.py:77-93`,
`drafter/requant_mtp_gptq.py:24-32`.

The inode-truncation trap — `save_file`/`torch.save` truncate in place, so
through a hardlink a variant rewrite silently rewrote the **source** — has
its bug story in `drafter/gptq_lm_head.py:83-86`, and that comment is now
**stale as a description of current behavior**: it says
`build_draft_vocab.py` rewrites the extras "with save_file (in place …)" and
the ids "with torch.save (in place …)", but since #201 both go through
tmp+rename (`build_draft_vocab.py:144-146`; `publish` "never changes a
hardlinked copy", `atomic_publish.py:11`). The copy-not-link code the comment
justifies (`gptq_lm_head.py:87-93`) stays load-bearing as long as *any*
writer truncates in place — and drafter's still do (`export_mtp.py:125,129`,
`requant_mtp_gptq.py:57,63`, `gptq_lm_head.py:106,109`). The guardrail itself
is transmitted as one comment (`export_mtp.py:53`,
`os.remove(D+f)  # never truncate a hardlink`) and two uncommented unlinks
(`requant_mtp_gptq.py:55-56`, `gptq_lm_head.py:103-104`). A bug that already
bit once is a guardrail only where someone remembered to write it — and its
own retelling has already drifted from the code.

### 1.6 The backup-suffix protocol

Six suffixes, each invented by its script, now over one shared mechanism —
`backup_once` (first-wins, atomically copied, `atomic_publish.py:67-74`) —
except where noted:

| suffix | written | read |
|---|---|---|
| `.bak` | `quant_lm_head.py:74` (its shard) | `gptq_lm_head.py:26-30`, `train_mtp.py:210-213` |
| `.bak-quant` | `quant_lm_head.py:79,107`; `quant_heads_stream.py:270,298` | — |
| `.bak_embed` | `quant_embed.py:77` (why not `.bak`: `:74-76`) | `train_mtp.py:205-211` |
| `.bak-mtp` | `quant_mtp.py:85,90,104`; `export_mtp.py:90,92,95` (hand-rolled `shutil.copy`, **not** first-wins) | `export_mtp.py:34-37`, `requant_mtp_gptq.py:34`, `train_mtp.py:186` |
| `.bak-draft` | `build_draft_vocab.py:140` | — |
| `.bak-orig` | `quant_heads_stream.py:148-153` — an `os.link`, deliberately not `backup_once` (no copying 18.6 GB) | `gptq_lm_head.py:26-30` |

A filename protocol crossing a dozen files, still never written down in one
place. Two cross-script couplings to know: `quant_embed.py` backs up only its
shard — config.json and the index get no new backup, relying on
`quant_lm_head.py`'s `.bak-quant` (run order enforced only by the docstring,
`quant_embed.py:2`); and `export_mtp.py:34-35` still restores "pre-MTP-quant"
state by copying `.bak-mtp` files back over the live ones.

### 1.7 The import shims

`sys.path.insert` ×5, all in `drafter/` (unchanged: `capture_dflash2.py:23`,
`export_mtp.py:63`, `gptq_lm_head.py:13`, `quant_dflash2.py:24`,
`requant_mtp_gptq.py:13`) — each script hand-rolling its access to
`gptq_utils`. `prepare/` has no shim and needs none: the eight
`from atomic_publish import ...` lines work because the interpreter puts the
script's own directory on `sys.path` — the pattern §2 relies on, already in
production.

## 2. The design: `prepare/pipeline_core.py`

One stdlib+torch module in `prepare/`, the sibling of `atomic_publish.py` —
the #195 saga landed that module in exactly this shape (74 lines,
stdlib-only, safetensors imported lazily at `:60`, no classes) and answered
the placement/import question: `prepare/` scripts import it bare (`from
atomic_publish import ...`, `quant_lm_head.py:29`) because the interpreter
puts the script's own directory on `sys.path`, and `docker/prepare.sh:83-87`
invokes them as `python prepare/<script>.py`. Eight production importers and
a crash-injection test (`bench/test_prepare_crash.py`) say the
extract-a-shared-module pattern is accepted in this codebase. `drafter/`
scripts replace their five shims with the one documented two-liner that adds
the repo root (a pattern with a comment, not five accidents). It is a bag of
functions and constants — no framework, no classes, no pipeline abstraction:

```text
QMAX/BITS/GROUP constants (per-call overridable)
MTP_LINEARS: the eight names, once (1.4)

rtn_quantize(W, bits, group, *, eps, keepdim=True)
    the 1.1 math with the eps EXPLICIT — each migrated caller passes its
    current value (1e-10 or 1e-8), so outputs are bit-identical per script
    (the acceptance gate in §4); the signature documents the difference

write_packed(tensors, key, q, scale, bits, scale_dtype)
    the 1.2 contract assembly (pack → dtype → shape → weight_map → group
    clone → ignore edit) over atomic_publish's save_tensors/write_json/
    backup_once (imported, not re-implemented), with the 1.3 dtype
    difference as a required argument, not a comment

resume_packed(names, suffixes)
    the killed-run "already packed, completing an interrupted run" check
    now copied five times (1.2)

repair_ignore_list(config, keys) / clone_group(qc, src, name, targets, bits)
    the ignore-list repair moves out of quant_lm_head.py (1.4) and the
    group_0→group_N clone becomes one function with the ordinal naming
    visible

assemble_variant(src, dst, hardlink=[patterns], copy=[patterns])
    the 1.5 assembly, once

fresh_write(path, save_fn)
    unlink-before-write, for drafter/'s remaining in-place writers only
    (publish() already never truncates a hardlink, atomic_publish.py:11;
    the live in-place sites: export_mtp.py:125,129,
    requant_mtp_gptq.py:57,63, gptq_lm_head.py:106,109,
    quant_dflash2.py:94,117) — gptq_lm_head.py:83-86's bug can never be
    reintroduced silently

write_draft_head(model_dir, ids_or_head, *, source="sliced"|"trained")
    the contract's single writer (1.5); the slicing path calls
    build_draft_vocab's logic, the trained path the export_mtp logic —
    the semantic difference becomes a named parameter. Atomic on both
    paths: the two current writers diverged on exactly this (1.5)

BACKUP suffix constants for the six suffixes (1.6: .bak, .bak-quant,
.bak_embed, .bak-mtp, .bak-draft, .bak-orig) + a protocol section in
the module docstring; backup_once itself is imported from atomic_publish
```

**Deliberately not in the module:** the I/O primitive itself —
`atomic_publish.py` stays its own module, the core builds on it; the two
loaders (in-RAM whole-shard vs
`quant_heads_stream.py`'s streaming rewriter — both are load-bearing,
sharing internals not I/O); anything GPTQ (`gptq_utils.py` keeps
`gptq_quantize`, `accumulate_hessian`, and loses `rtn_quantize`/`dequant` to
the core it can now import); the corpus counting in `build_draft_vocab.py`
(a data tool, stays with its script); training/capture scripts.

## 3. Rollout

### 3.1 PR A — the core + the prepare/ migration

Add `prepare/pipeline_core.py` and `bench/test_pipeline_core.py` (§4).
Migrate the four `prepare/` quant scripts + `build_draft_vocab.py`'s
slicing writer. The shape is pre-approved by precedent: #195's
`atomic_publish.py` landed as exactly such a module (74 lines, eight
importers, its own crash-injection test at `bench/test_prepare_crash.py`).
**Bit-identical acceptance:** run each migrated script on a copy of a small
prepared fixture model, before and after, and `diff` the resulting
shard/index/config bytes — the eps-per-caller rule (2.1) exists to make this
gate pass.

### 3.2 PR B — the drafter/ migration

`gptq_lm_head.py`, `requant_mtp_gptq.py`, `quant_dflash2.py`,
`export_mtp.py` onto the core (RTN sites, write tails, variant assembly,
fresh_write); `gptq_utils.py` slims to GPTQ + Hessian accumulation and
imports the core's RTN/dequant; the five shims become the one documented
pattern. `train_mtp.py`'s dict literal becomes `MTP_LINEARS` (its only
change — research code otherwise untouched).

### 3.3 PR C — one draft-head writer + the ignore-list's proper home

`write_draft_head` becomes the only writer of the contract;
`build_draft_vocab.py` keeps corpus counting and calls it;
`export_mtp.py:108-130` calls it with `source="trained"`. The ignore-list
repair (currently quant_lm_head's side job, 1.4) moves into the core and
runs as its own step in `docker/prepare.sh`'s sequence (a new explicit
step, not a hidden one). The module docstring's protocol section (backup
suffixes, group ordinals) lands with the code.

## 4. Test plan

`bench/test_pipeline_core.py` (CPU, seconds, wired into
`patch-integrity.yml` next to `test_model_verification.py`):

| test | proves |
|---|---|
| RTN round-trip: `dequant(rtn_quantize(W)) ≈ W` within 2^-bits per group, for bits=4 and 8 | the unified math is correct on both call shapes |
| pack layout: `write_packed` output keys/shapes/dtypes against a golden tiny safetensors | the write tail's contract is pinned |
| scale-dtype: lm_head→fp16, embed→bf16 (the 1.3 fact as an assertion) | the invisible difference is now tested, not commented |
| `fresh_write` on a hardlinked pair: source inode's bytes unchanged after variant rewrite | the 1.5 bug is a regression test, not a comment |
| ignore-list repair idempotent and complete over the 8 MTP names | the 1.4 side job, now explicit |
| fixture end-to-end: `quant_lm_head.py` on a tiny synthetic model dir, then `--status`-style name checks | the scripts work through the core (fixture style per `test_model_verification.py:27-51`) |

Plus PR A's bit-identical gate (3.1) run once per migrated script.

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| Unifying the RTN changes produced checkpoints (eps, keepdim) | medium | per-caller explicit eps (2.1) + the bit-identical gate (3.1); a script whose output changes fails its own migration PR |
| Over-unification: merging the two loaders "because they both read safetensors" | medium | explicitly out of scope (2.1) — the streaming rewriter is genuinely different I/O; the plan shares math and protocol, not I/O |
| The core becomes a kitchen sink (the "framework" failure mode) | medium | the "not in the module" list is written down (2.1); review rule: constants, math, one-writer contracts — no control flow that a caller owns |
| drafter/ scripts gain a new import failure mode (repo-root shim) | low | one documented pattern replacing five ad-hoc ones; CI's test file imports the core both ways |
| `export_mtp.py`'s dual nature (delegates AND re-implements) hides a third draft-head path | low | PR C makes the writers one function with a named `source=` parameter; the subprocess call at `:104-106` remains as orchestration |

## 6. Done when

1. `grep -rn 'round(g / scale)' prepare/ drafter/` → one site (the core).
2. `grep -rn 'MTP_LINEARS\|"mtp.fc"' prepare/ drafter/` → one definition,
   imported everywhere (the dict in `train_mtp.py` and the tuple in
   `quant_lm_head.py` are gone).
3. `grep -c 'sys.path.insert' drafter/*.py` → 0 ad-hoc shims (one shared
   pattern with a comment, if any).
4. `write_draft_head` is the only writer of `mtp.draft_lm_head.*`;
   `grep -rn 'draft_lm_head.weight_packed' prepare/ drafter/` → core +
   consumers.
5. `bench/test_pipeline_core.py` green in CI; PR A's bit-identical diffs
   empty for every migrated script.
