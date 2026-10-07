# Deepen the preparation pipeline's shared core — remediation plan

Architecture review 2026-09-26, candidate 4 (Worth exploring). The full report
is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree on
2026-09-26.

> **Merged 2026-10-06 (upstream/main @ 7af097b).** D1 merged as #276 (`7796b64`).
>
> **Updated 2026-10-05 on upstream/main @ 10bb488.** The group_1 gap (D1 in the
> tracker) is in review as syv-ai/HyperQwen#276 (`f56f0d5`). The fix is not the
> `requires=` check that the 2026-10-04 block, §2 and §3.1 name.
> - `quant_embed.py:91` clones `group_0`, as `quant_mtp.py` and
>   `quant_heads_stream.py` do. `quant_lm_head.py:102-109` builds `group_1` as
>   `group_0` with `targets`, `num_bits`, `symmetric` and `zp_dtype` set, and
>   `quant_embed.py` sets the same four fields on its copy. So `group_2` does
>   not change. Run first, `quant_embed.py` now completes. The recommended
>   order stays. Only in that order does `quant_lm_head.py`'s `.bak-quant`
>   hold the pristine `config.json` and index.
> - Measured on CPU, on the `test_prepare_crash.py` fixture. The
>   `docker/prepare.sh` sequence leaves the same 34 files, byte for byte, on
>   main and on #276. On main, `quant_embed.py` alone exits 1 with
>   `KeyError: 'group_1'` after it replaced the shard and wrote `.bak_embed`.
> - The test is a new `order` step, not a `BAD_CONFIGS` case. Both orders
>   must exit 0 and leave the same files. It fails with main's
>   `quant_embed.py`. `group_2` must also equal `group_1` except for
>   `targets`, as on main. All steps: 169 cases, 0 failures.
> - For this plan, `load_config(d, requires=...)` is not needed, and PR A does
>   not start with it. Every `prepare/` script now clones `group_0`, which
>   `load_config` already checks. The one clone of another group is
>   `drafter/export_mtp.py:136`, from the `group_3` that the same script
>   writes at `:101`.

**Re-verified 2026-10-04 against upstream/main @ e371b42 (vLLM 0.30.0).**

**Status:** Not started. Worth exploring; not urgent. Build on prepare/quant_schema.py (#246), not a new module.

- Not landed. There is no `pipeline_core.py` and no PR A, B or C. The §6
  greps return the d5e2a01 results: `torch.round(g / s` six sites,
  `MTP_LINEARS\|"mtp.fc"` six definition sites, `sys.path.insert` 5 in
  `drafter/`, `draft_lm_head.weight_packed` two writers
  (`build_draft_vocab.py:141`, `export_mtp.py:122`). No open PR touches
  `prepare/` or `drafter/` (`gh pr list -R syv-ai/HyperQwen --state open`).
- Of the 13 commits since d5e2a01, only accc8cf (#246) touches `prepare/`.
  `drafter/`, `docker/prepare.sh`, `prepare/atomic_publish.py`,
  `prepare/build_draft_vocab.py`, `prepare/fetch_fast_variant.py` and
  `patches/qwen3_5-mtp-draft-vocab.patch` are unchanged
  (`git diff --stat d5e2a01 e371b42 -- <paths>` is empty for them).
- What #246 changed. It adds `prepare/quant_schema.py` (71 lines, stdlib
  only, one public function). `load_config(d)` (`:34-71`) reads
  `config.json` and returns `(c, qc)`. It exits before any write unless
  `quant_method` is compressed-tensors (`:52-58`), `ignore` is a list
  (`:60-62`) and `config_groups.group_0` exists (`:64-67`). The four quant
  scripts call it before they read a shard (`quant_lm_head.py:45`,
  `quant_embed.py:43`, `quant_mtp.py:56`, `quant_heads_stream.py:74`) and
  drop their inline `json.load` + `qc = c["quantization_config"]` pairs.
  `bench/test_prepare_crash.py` gains a `reject` step (`BAD_CONFIGS`
  `:359-371`, `reject_case`/`reject` `:374-412`): each script must exit
  non-zero on three bad configs and write nothing. `prepare/README.md:9-36`
  documents the accepted schema.
- What it means for this plan. `load_config` takes over the config read at
  the head of the 1.2 write tail (four inline copies become one). It also
  checks the inputs that §2's `clone_group` and `repair_ignore_list` take.
  It does not touch 1.1, 1.3, 1.4, 1.5 or 1.6, and it conflicts with
  nothing: §2 can consume its `(c, qc)`. It adds no copy of the MTP list or
  of the RTN math. It restates three facts: every added group is a clone of
  `group_0` (`:67`), the scripts rewrite the ignore list (`:62`), and the
  shard is written before config.json and the index (docstring `:7-8`, also
  `atomic_publish.py:12-15`). The same four-line docstring paragraph and
  two-line comment are pasted into all four scripts.
- The group_1 gap. `load_config` checks only `group_0`
  (`quant_schema.py:65`). `quant_embed.py` clones `group_1`, which
  `quant_lm_head.py` writes. Run `quant_embed.py` by hand before
  `quant_lm_head.py`: it passes the check, runs `backup_once` and
  `save_tensors` on the shard (`:85-86`), then reads
  `qc["config_groups"]["group_1"]` (`:89`) and raises KeyError. That is the
  #241 failure mode (a KeyError after the shard is replaced) through another
  key. `docker/prepare.sh:62-63,83-84` runs the scripts in order, so only a
  manual run hits it. The three `BAD_CONFIGS` cover the method, `group_0`
  and `ignore`, not `group_1`. The fix is `load_config(d,
  requires=("group_1",))` in `quant_embed.py` (§2, §3.1). (2026-10-05:
  replaced. #276 clones `group_0` instead; see the top block.)
- Recounts, d5e2a01 → e371b42: RTN copies 5 → 5; MTP list 6 sites /
  4 spellings → 6 / 4; killed-run resume check 5 → 5; write-tail sites 9 in
  8 files → 9 in 8 files; inline config reads in the quant scripts 4 → 1;
  `drafter/` non-atomic and in-place write lists unchanged; shared
  `prepare/` modules 1 → 2; `prepare/*.py` files 11 → 12; `.py` files that
  name `weight_packed` 10 → 10; `.py` files that name `.bak` 10 → 10.
- 1.7 import count. Bare imports in `prepare/` go from 8 lines to 12: eight
  `from atomic_publish import`, four `from quant_schema import load_config`
  (`grep -n 'from atomic_publish\|from quant_schema' prepare/*.py
  drafter/*.py`). No `drafter/` file imports either module. `drafter/`
  shims stay at 5. `bench/test_prepare_crash.py` puts `prepare/` on
  `sys.path` at three sites, not two (`:91`, `:233`, and `:382` in the new
  `reject_case`).
- Line cites. #246 shifts the four quant scripts by +4 to +9 lines (docstring
  +4, import +1, `load_config` call +3 or +4, inline config read −2). Every
  stale cite in §1.1–1.7 is re-cited in place below, each re-read at
  e371b42. `verify.sh:269` is now `:270` (177ce26). The older blocks keep
  their cites as history.
- CI (§4). `patch-integrity.yml` now has a CPU-torch job,
  `kvarn-torch-gate` (`:25-37`, bd6c5e2). The `model-verification` job
  (`:13-23`), where §4 puts the new test, installs no torch.
  `bench/test_prepare_crash.py` is still not in CI.

**Re-verified 2026-09-29 against upstream/main @ d5e2a01 (vLLM 0.30.0).**

- Unchanged: none of the seven commits since 2522ef9 (8d9848d #231, e355f9f
  #234, 36936ec #232, 1bb8e4e #235, d2ff871 #230, 8cf642e #219, d5e2a01 #236)
  touches `prepare/`, `drafter/`, `docker/prepare.sh`,
  `patches/qwen3_5-mtp-draft-vocab.patch`, `bench/` or `.github/`
  (`git diff --stat 2522ef9 upstream/main -- prepare drafter docker/prepare.sh
  patches/qwen3_5-mtp-draft-vocab.patch bench .github` is empty). Every line
  citation below was re-read at d5e2a01 and holds; nothing was fixed upstream,
  nothing new was introduced. No open PR touches `prepare/` or `drafter/`
  (`gh pr list -R syv-ai/HyperQwen --state open`).
- Corrected (imprecise at 2522ef9 too, same bytes): the drafter non-atomic
  write list gains `export_mtp.py:54,139` (1.2); the in-place-truncation list
  gains `export_mtp.py:91,93,102,139` and drops `requant_mtp_gptq.py:57`,
  which follows its own unlink (1.5, §2); "ten files name `weight_packed`" is
  ten `.py` files, eleven with `drafter/README.md` (1.2); the plain-text
  branch of `build_draft_vocab.py` is `:63-75`, not `:63-71` (1.5); the
  backup protocol's "dozen files" is ten `.py` files plus `verify.sh:269` and
  `prepare/README.md:58` (1.6); Done-when 1's grep `round(g / scale)` matched
  4 of the 6 RTN sites (it misses `quant_heads_stream.py:85`'s `g / s` and
  `gptq_utils.py:83`'s `scale[..., None]`) — replaced (§6), and every §6 grep
  now carries `--include='*.py'`; Done-when 4 expected "core + consumers" in
  `prepare/ drafter/`, which hold no consumer — now "the core only" (§6);
  `quant_embed.py`'s run order after `quant_lm_head.py` is enforced by
  `docker/prepare.sh:62-63,83-84`, not "only by the docstring" (1.6); the
  `export_mtp.py:53` quote is now verbatim (`os.remove(D + f)   # …`, 1.5);
  §4's `--status`-style check names a flag no script has yet (candidate 7);
  the plan's "(2.1)" cross-references pointed at a section that does not
  exist — now "§2".
- Unchanged, re-read line by line: the five RTN copies and the 1e-10 vs
  1e-8 / keepdim split (`gptq_utils.py:77-84`, clamp at `:82`), the nine
  write-tail sites, `atomic_publish.py` (74 lines, eight importers), the
  fp16/bf16 encodings (`quant_lm_head.py:70-71`, `quant_embed.py:70-71`,
  `quant_heads_stream.py:178`), the six-site/four-spelling MTP list, the five
  shims, the two draft-head writers and the patch consumer (`:33-42`), the
  `gptq_lm_head.py:83-86` comment, the six suffixes; #195 closed and
  #198–#203, #212, #213 merged (`gh`).
- Added: `drafter/train_mtp.py:395` imports `gptq_utils` bare, with no shim —
  evidence the five `drafter/` shims are redundant when run as scripts (1.7);
  the two `prepare/` scripts outside the eight importers
  (`fetch_dflash2.py`, `fetch_thirdparty.py`) only call `snapshot_download`
  (1.2); `bench/test_prepare_crash.py` puts `prepare/` on `sys.path` itself
  because it runs the scripts through `runpy` (1.7).

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
| `prepare/quant_lm_head.py` | `:69-70` | QMAX=127, clamp min **1e-10**, keepdim |
| `prepare/quant_embed.py` | `:69-70` | identical to quant_lm_head (verified) |
| `prepare/quant_mtp.py` | `:83-84` | identical |
| `prepare/quant_heads_stream.py` | `:92-93` | chunked (per-block), lowercase qmax/s |
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
import it (`quant_lm_head.py:33`, `quant_embed.py:33`, `quant_mtp.py:34`,
`quant_heads_stream.py:51`, `build_draft_vocab.py:35`,
`fetch_fast_variant.py:18`, `harden_chat_template.py:30`,
`translate_chat_template.py:40`; of the other `prepare/*.py`,
`fetch_dflash2.py:18` and `fetch_thirdparty.py:35` write only through
`snapshot_download`, and `quant_schema.py` (#246, 2026-10-04) writes
nothing); each still sequences the calls itself
("The index is the commit point, so it goes last" — `quant_lm_head.py:112-117`
and twins). `prepare/README.md:91-95` documents the protocol and its
crash-injection test (`bench/test_prepare_crash.py`). Since #246 the config
read before the tail is shared too: `load_config` (`quant_schema.py:34-71`)
replaces the four inline reads (2026-10-04).

The contract half — `pack_to_int32(q, BITS)` → `scale.to(dtype)` →
`weight_shape = tensor([out_f, in_f], int64)` → `weight_map` edit →
`config.json` group clone → ignore-list edit — is still assembled per script,
nine sites in eight files (`export_mtp.py` carries two), plus one slice
assembly:

| site | pack/scale/shape | weight_map | config group | ignore edit |
|---|---|---|---|---|
| `prepare/quant_lm_head.py` | `:77-80` | `:114-116` | clone `group_0`→`group_1`, `:102-109` | drop `lm_head`, `:87` (+ MTP list, 1.4) |
| `prepare/quant_embed.py` | `:77-80` | `:100-102` | clone `group_1`→`group_2`, `:89-96` (`group_0` after #276) | — |
| `prepare/quant_mtp.py` | `:88-90` | `:112-115` | clone `group_0`→`group_3`, `:100-107` | `:99` |
| `prepare/quant_heads_stream.py` | `:213-216` (heads), `:257-259` (MTP) | `:226-230`, `:270-273` | `group()` factory `:280-292` → `group_1..3`, `:296-300` | `:295` |
| `drafter/export_mtp.py` (GPTQ MTP) | `:84-86` | `:87-89` | `group_3`, `:98-101` | `:97` |
| `drafter/export_mtp.py` (draft head) | `:122-124` | `:127-128` | `group_4` when HBITS≠BITS, `:136-138` | — |
| `drafter/requant_mtp_gptq.py` | `:51-53` | (index copied verbatim, `:31`) | `group_3` num_bits in place, `:60-62` | — |
| `drafter/gptq_lm_head.py` | `:100-102` | index rewritten, `:106` | `group_1` num_bits in place, `:108` | — |
| `drafter/quant_dflash2.py` | `:79-81` | (single-file model, no index) | groups **fabricated**, not cloned, `:103-116` | built fresh, `:100-102` |
| `prepare/build_draft_vocab.py` (slice; no pack call) | `:141-143` | `:148-149` | — | — |

Ten `.py` files name `weight_packed` (`git grep -l weight_packed -- prepare
drafter`; eleven files with `drafter/README.md`, corrected 2026-09-29): the
nine writers above plus
`drafter/train_mtp.py:203-204`, which only reads the packed entries to find
shards. Three details the table hides: the group ordinals are hardcoded per
script and chained (`quant_embed.py:89` clones the `group_1` that
`quant_lm_head.py` wrote — an ordering dependency that #246's `load_config`
does not check, see the 2026-10-04 block; #276 removes it, see the 2026-10-05
block); `drafter/` writes are
still non-atomic (`save_file`/`json.dump` straight to the final path:
`export_mtp.py:54,91-93,102,125,129,139` — `:54` and `:139` were missing from
this list at 2522ef9 — `requant_mtp_gptq.py:57,63`, `gptq_lm_head.py:105-109`,
`quant_dflash2.py:94,117`); and the killed-run resume check ("already packed
… completing an interrupted run") that the atomic protocol needs is itself a
per-script copy, five times: `quant_lm_head.py:60-64`, `quant_embed.py:60-64`,
`quant_mtp.py:73-77`, `quant_heads_stream.py:195-201` and `:246-252`.

### 1.3 The one difference that matters is invisible

Scale dtype: fp16 for lm_head, bf16 for embed_tokens. Encoded as:
- `prepare/quant_lm_head.py:78-79` — a **comment** ("linear layers use fp16
  scales in this checkpoint") and `.to(torch.float16)`;
- `prepare/quant_embed.py:78-79` — a **comment** ("the embedding path creates
  scales in params_dtype (bf16), unlike the linears") and `.to(torch.bfloat16)`;
- `prepare/quant_heads_stream.py:186` — as **data**:
  `for key, scale_dtype in ((lm_key, torch.float16), (emb_key, torch.bfloat16))`
  (plus a fourth restatement as a comment at `:214-215`).

Three encodings of one fact across three scripts (four statements; only the
data tuple is machine-checkable). Get it wrong and the checkpoint still loads
— vLLM casts — with a silent quality cost per group scale.

### 1.4 The MTP tensor list: six sites, four spellings

| site | form |
|---|---|
| `prepare/quant_lm_head.py:90-99` | inline tuple, inside the **ignore-list repair** — the script that quants lm_head secretly owns adding `mtp.*` to the ignore list, with the comment "The MTP draft head is stored in bf16 but missing from the ignore list, which breaks loading when speculative decoding is enabled (single-user mode)" (`:88-89`; a side job in the wrong file) |
| `prepare/quant_mtp.py:43-51` | `MTP_LINEARS` (`mtp.fc` drops out under `--keep-fc`) |
| `prepare/quant_heads_stream.py:61-69` | `MTP_LINEARS` (same `--keep-fc` conditional) |
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
  bc6d9e7 #213, `:63-75` — was cited `:63-71`, which stops mid-branch).
  Since 850d2ad #201 it writes **atomically**:
  `backup_once(..., ".bak-draft")` + `save_tensors` (`:140-144`), the ids by
  `torch.save` to `.tmp` + `publish` (`:145-146`), the weight_map edit and
  the index **last** (`:147-150`),
- `drafter/export_mtp.py:108-130` (plus the `group_4` config write when
  HBITS≠BITS, `:131-139`) — requantizes a **trained bf16** draft head
  with its own RTN copy + write tail (the fifth RTN site, 1.1), still with
  plain `save_file` + `json.dump` in place (`:125,:129,:139`) — **not** atomic;
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
writer truncates in place — and drafter's still do (`export_mtp.py:91,93,
102,125,129,139`, `requant_mtp_gptq.py:63`, `gptq_lm_head.py:106,109`;
corrected 2026-09-29: at 2522ef9 this list read `export_mtp.py:125,129`,
`requant_mtp_gptq.py:57,63` — but `:57` writes a path its own `:55-56` just
unlinked, and `export_mtp.py:91,93,102,139` open an existing file — the
`:54` extras, the `:35` index copy, the `:34` config copy — for writing).
The guardrail itself
is transmitted as one comment (`export_mtp.py:53`,
`os.remove(D + f)   # never truncate a hardlink`) and two uncommented unlinks
(`requant_mtp_gptq.py:55-56`, `gptq_lm_head.py:103-104`). A bug that already
bit once is a guardrail only where someone remembered to write it — and its
own retelling has already drifted from the code.

### 1.6 The backup-suffix protocol

Six suffixes, each invented by its script, now over one shared mechanism —
`backup_once` (first-wins, atomically copied, `atomic_publish.py:67-74`) —
except where noted:

| suffix | written | read |
|---|---|---|
| `.bak` | `quant_lm_head.py:82` (its shard) | `gptq_lm_head.py:26-30`, `train_mtp.py:210-213` |
| `.bak-quant` | `quant_lm_head.py:86,113`; `quant_heads_stream.py:277,304` | — |
| `.bak_embed` | `quant_embed.py:85` (why not `.bak`: `:82-84`) | `train_mtp.py:205-211` |
| `.bak-mtp` | `quant_mtp.py:94,98,111`; `export_mtp.py:90,92,95` (hand-rolled `shutil.copy`, **not** first-wins) | `export_mtp.py:34-37`, `requant_mtp_gptq.py:34`, `train_mtp.py:186` |
| `.bak-draft` | `build_draft_vocab.py:140` | — |
| `.bak-orig` | `quant_heads_stream.py:156-161` — an `os.link`, deliberately not `backup_once` (no copying 18.6 GB) | `gptq_lm_head.py:26-30` |

A filename protocol crossing a dozen files — ten `prepare/`+`drafter/` `.py`
files (`git grep -l '\.bak' -- 'prepare/*.py' 'drafter/*.py'`, including
`fetch_thirdparty.py:42`'s message naming `.bak-orig`), plus `verify.sh:270`
(skips `.bak` shards) and `prepare/README.md:87` (`.bak*`) — still never
written down in one place. Two cross-script couplings to know:
`quant_embed.py` backs up only its
shard — config.json and the index get no new backup, relying on
`quant_lm_head.py`'s `.bak-quant` (run order enforced by `docker/prepare.sh`'s
step order, `:62-63` then `:83-84`, and for a manual run only by the
docstring, `quant_embed.py:2`; at 2522ef9 this read "enforced only by the
docstring"); and `export_mtp.py:34-35` still restores "pre-MTP-quant"
state by copying `.bak-mtp` files back over the live ones.

### 1.7 The import shims

`sys.path.insert` ×5, all in `drafter/` (unchanged: `capture_dflash2.py:23`,
`export_mtp.py:63`, `gptq_lm_head.py:13`, `quant_dflash2.py:24`,
`requant_mtp_gptq.py:13`) — each script hand-rolling its access to
`gptq_utils`. `prepare/` has no shim and needs none: the eight
`from atomic_publish import ...` lines and, since #246, the four
`from quant_schema import load_config` lines (`quant_lm_head.py:34`,
`quant_embed.py:34`, `quant_mtp.py:35`, `quant_heads_stream.py:52`;
2026-10-04) work because the interpreter puts the
script's own directory on `sys.path` — the pattern §2 relies on, already in
production. The same holds in `drafter/`: every one of the five shims inserts
the script's own directory (`os.path.dirname(os.path.abspath(__file__))`, or
`HERE`, defined that way at `capture_dflash2.py:17`), which is already
`sys.path[0]` under `python drafter/<script>.py`, and `train_mtp.py:395`
imports `gptq_utils` with no shim at all (added 2026-09-29). The one caller
that does not get the script directory for free is `runpy`:
`bench/test_prepare_crash.py` adds `prepare/` to `sys.path` itself
(`:91`, `:233`, and since #246 `:382`) before `runpy.run_path` (`:92-93`,
`:240`, `:383`).

## 2. The design: `prepare/pipeline_core.py`

**Adjusted 2026-10-04: build on `prepare/quant_schema.py`, not a new
module.** #246 landed a second shared module in `prepare/`. Its
`load_config` already does the config read that `write_packed`,
`clone_group` and `repair_ignore_list` start from (2026-10-04 block). So:

- `quant_schema.py` is the home of the core. It takes the config and
  contract facts first: `MTP_LINEARS`, `clone_group`, `repair_ignore_list`,
  the group ordinals, the per-key scale dtypes and the backup-suffix
  constants. The quant mechanics (`rtn_quantize`, `write_packed`,
  `resume_packed`) and the `drafter/` pieces (`assemble_variant`,
  `fresh_write`, `write_draft_head`) go there too. Split them into one new
  sibling only if `quant_schema.py` grows past about 150 lines.
- `load_config(d, requires=("group_0",))` names the config groups a script
  clones. `quant_embed.py` passes `("group_1",)`. This closes the group_1
  gap: today a manual run out of order replaces the shard and then raises
  KeyError (`quant_schema.py:65`; `quant_embed.py:85-86`, then `:89`).
  **2026-10-05:** not needed. #276 makes `quant_embed.py` clone
  `group_0`, the group `load_config` already checks (top block).
- `atomic_publish.py` stays the I/O layer. The core imports it and does
  not re-implement it.

Read `pipeline_core` below as "the core", in `quant_schema.py`. The
original design follows unchanged.

One stdlib+torch module in `prepare/`, the sibling of `atomic_publish.py` —
the #195 saga landed that module in exactly this shape (74 lines,
stdlib-only, safetensors imported lazily at `:60`, no classes) and answered
the placement/import question: `prepare/` scripts import it bare (`from
atomic_publish import ...`, `quant_lm_head.py:33`) because the interpreter
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
    (the acceptance gate in 3.1, run again per §4); the signature documents the difference

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
    the live in-place sites, recounted 2026-09-29 (1.5):
    export_mtp.py:91,93,102,125,129,139, requant_mtp_gptq.py:63,
    gptq_lm_head.py:106,109, quant_dflash2.py:94,117 — at 2522ef9 this
    list read export_mtp.py:125,129 and requant_mtp_gptq.py:57,63; the
    unlink-guarded writes export_mtp.py:54, requant_mtp_gptq.py:57 and
    gptq_lm_head.py:105 already follow this pattern by hand) —
    gptq_lm_head.py:83-86's bug can never be
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

**Adjusted 2026-10-04.** Extend `prepare/quant_schema.py` instead of
adding `pipeline_core.py` (§2). #246's `quant_schema.py` is a second
precedent: 71 lines, four importers, its own `reject` step in
`bench/test_prepare_crash.py` (`:359-412`). Land `load_config(d,
requires=...)` first, as its own small PR: `quant_embed.py` passes
`("group_1",)`, and `BAD_CONFIGS` gains a case with `group_0` but no
`group_1` for `quant_embed.py`. The rest of PR A follows on top of it.
**2026-10-05:** #276 closes the gap with no `requires=` and no new
`BAD_CONFIGS` case (top block), so PR A starts with the core itself.
**Bit-identical acceptance:** run each migrated script on a copy of a small
prepared fixture model, before and after, and `diff` the resulting
shard/index/config bytes — the eps-per-caller rule (§2) exists to make this
gate pass.

### 3.2 PR B — the drafter/ migration

`gptq_lm_head.py`, `requant_mtp_gptq.py`, `quant_dflash2.py`,
`export_mtp.py` onto the core (RTN sites, write tails, variant assembly,
fresh_write); `gptq_utils.py` slims to GPTQ + Hessian accumulation and
imports the core's RTN/dequant; the five shims become the one documented
pattern. `train_mtp.py`'s dict literal becomes `MTP_LINEARS` (its only
change — research code otherwise untouched).

**Adjusted 2026-10-04.** The core now lives in `prepare/quant_schema.py`,
so the one documented `drafter/` import pattern has to reach `prepare/`.
No `drafter/` file imports from `prepare/` today (no `atomic_publish` or
`quant_schema` import in `drafter/*.py`).

### 3.3 PR C — one draft-head writer + the ignore-list's proper home

`write_draft_head` becomes the only writer of the contract;
`build_draft_vocab.py` keeps corpus counting and calls it;
`export_mtp.py:108-130` calls it with `source="trained"`. The ignore-list
repair (currently quant_lm_head's side job, 1.4) moves into the core and
runs as its own step in `docker/prepare.sh`'s sequence (a new explicit
step, not a hidden one). The module docstring's protocol section (backup
suffixes, group ordinals) lands with the code. (2026-10-04: the core is
`quant_schema.py`, §2. Its docstring already states the group_0 clone rule
and the shard-before-config order, `:3-9`.)

## 4. Test plan

`bench/test_pipeline_core.py` (CPU, seconds, wired into
`patch-integrity.yml` next to `test_model_verification.py`; 2026-10-04:
that job, `model-verification` `:13-23`, installs no torch, and the new
`kvarn-torch-gate` job `:25-37` installs CPU torch only — the core's
tests also need `safetensors` and `compressed_tensors`, which the quant
scripts import, `quant_lm_head.py:30-31`):

| test | proves |
|---|---|
| RTN round-trip: `dequant(rtn_quantize(W)) ≈ W` within 2^-bits per group, for bits=4 and 8 | the unified math is correct on both call shapes |
| pack layout: `write_packed` output keys/shapes/dtypes against a golden tiny safetensors | the write tail's contract is pinned |
| scale-dtype: lm_head→fp16, embed→bf16 (the 1.3 fact as an assertion) | the invisible difference is now tested, not commented |
| `fresh_write` on a hardlinked pair: source inode's bytes unchanged after variant rewrite | the 1.5 bug is a regression test, not a comment |
| ignore-list repair idempotent and complete over the 8 MTP names | the 1.4 side job, now explicit |
| fixture end-to-end: `quant_lm_head.py` on a tiny synthetic model dir, then `--status`-style name checks (no script has `--status` at d5e2a01 — `git grep -- --status -- prepare drafter` is empty; it is candidate 7's proposal, `prepare-status-remediation.md` §2.1 — so until that lands, the checks read the index directly) | the scripts work through the core (fixture style per `test_model_verification.py:27-51`) |

Plus PR A's bit-identical gate (3.1) run once per migrated script.

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| Unifying the RTN changes produced checkpoints (eps, keepdim) | medium | per-caller explicit eps (§2) + the bit-identical gate (3.1); a script whose output changes fails its own migration PR |
| Over-unification: merging the two loaders "because they both read safetensors" | medium | explicitly out of scope (§2) — the streaming rewriter is genuinely different I/O; the plan shares math and protocol, not I/O |
| The core becomes a kitchen sink (the "framework" failure mode) | medium | the "not in the module" list is written down (§2); review rule: constants, math, one-writer contracts — no control flow that a caller owns |
| drafter/ scripts gain a new import failure mode (repo-root shim) | low | one documented pattern replacing five ad-hoc ones; CI's test file imports the core both ways |
| `export_mtp.py`'s dual nature (delegates AND re-implements) hides a third draft-head path | low | PR C makes the writers one function with a named `source=` parameter; the subprocess call at `:104-106` remains as orchestration |

## 6. Done when

1. `grep -rn --include='*.py' 'torch.round(g / s' prepare/ drafter/` → one
   site (the core). Today: six — the five RTN copies and
   `gptq_utils.py:83`. (At 2522ef9 this item read `grep -rn 'round(g /
   scale)'`, which matches only four of the six: it misses
   `quant_heads_stream.py:85`'s `g / s` and `gptq_utils.py:83`'s
   `g / scale[..., None]`.)
2. `grep -rn --include='*.py' 'MTP_LINEARS\|"mtp.fc"' prepare/ drafter/` →
   one definition, imported everywhere (the dict in `train_mtp.py` and the
   tuple in `quant_lm_head.py` are gone). `--include='*.py'` added
   2026-09-29 on all greps here: `drafter/data/` (gitignored corpora) holds
   `.jsonl` text that matches, e.g. `sys.path.insert`.
3. `grep -c 'sys.path.insert' drafter/*.py` → 0 ad-hoc shims (one shared
   pattern with a comment, if any). Today: 5.
4. `write_draft_head` is the only writer of `mtp.draft_lm_head.*`;
   `grep -rn --include='*.py' 'draft_lm_head.weight_packed' prepare/ drafter/`
   → the core only (today: the two writers, `build_draft_vocab.py:141`,
   `export_mtp.py:122`; the consumers — `docker/prepare.sh:65`, the vLLM
   patch — live outside these two directories).
5. `bench/test_pipeline_core.py` green in CI; PR A's bit-identical diffs
   empty for every migrated script.
