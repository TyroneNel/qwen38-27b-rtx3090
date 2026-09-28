# Deepen the preparation pipeline's shared core — remediation plan

Architecture review 2026-09-26, candidate 4 (Worth exploring). The full report
is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, every claim re-verified against the tree on
2026-09-26.

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
| `prepare/quant_lm_head.py` | `:46-47` | QMAX=127, clamp min **1e-10**, keepdim |
| `prepare/quant_embed.py` | `:46-47` | identical to quant_lm_head (verified) |
| `prepare/quant_mtp.py` | `:61-62` | identical |
| `prepare/quant_heads_stream.py` | `:75-76` | chunked (per-block), lowercase qmax |
| `drafter/export_mtp.py` | `:112-113` | the draft-head RTN, identical to quant_lm_head |

A shared `rtn_quantize` **already exists** at `drafter/gptq_utils.py:77-84` —
but (a) it lives in `drafter/`, which `prepare/` scripts cannot import (no
path shim reaches it), and (b) it is not actually identical: clamp min
**1e-8** (vs 1e-10), no keepdim, parameterized bits. The epsilon discrepancy
is exactly the class of difference that is invisible until a produced
checkpoint differs in the third decimal of a scale — today there is no place
where that difference is even written down.

### 1.2 The compressed-tensors write tail: eight implementations

The sequence `weight_packed = pack_to_int32(q, BITS)` → `weight_scale =
scale.to(dtype)` → `weight_shape = tensor([out_f, in_f], int64)` → backup →
`save_file` → `weight_map` edit → `config.json` group clone → ignore-list
edit, verified at: `prepare/quant_lm_head.py:54-69`,
`prepare/quant_embed.py:54-58` (+ analogous tail), `prepare/quant_mtp.py`
(same shape), `prepare/quant_heads_stream.py:148-152` (the only one with the
dtype difference as data — see 1.3), `drafter/export_mtp.py:84-93` (MTP
linears) **and** `:122-129` (draft head — two in one file),
`drafter/requant_mtp_gptq.py:51-57`, `drafter/gptq_lm_head.py`,
`drafter/quant_dflash2.py`. Ten files contain `weight_packed` writes
(verified by grep).

### 1.3 The one difference that matters is invisible

Scale dtype: fp16 for lm_head, bf16 for embed_tokens. Encoded as:
- `prepare/quant_lm_head.py:55-56` — a **comment** ("linear layers use fp16
  scales in this checkpoint") and `.to(torch.float16)`;
- `prepare/quant_embed.py:55-56` — a **comment** ("the embedding path creates
  scales in params_dtype (bf16), unlike the linears") and `.to(torch.bfloat16)`;
- `prepare/quant_heads_stream.py:152` — as **data**:
  `for key, scale_dtype in ((lm_key, torch.float16), (emb_key, torch.bfloat16))`.

Three encodings of one fact; only the third is machine-checkable. Get it
wrong and the checkpoint still loads — vLLM casts — with a silent quality
cost per group scale.

### 1.4 The MTP tensor list: six sites, three forms

| site | form |
|---|---|
| `prepare/quant_lm_head.py:74-83` | inline tuple, inside the **ignore-list repair** — the script that quants lm_head secretly owns adding `mtp.*` to the ignore list, with the comment "The MTP draft head is stored in bf16 but missing from the ignore list, which breaks loading when speculative decoding is enabled" (a side job in the wrong file) |
| `prepare/quant_mtp.py:32` | `MTP_LINEARS` |
| `prepare/quant_heads_stream.py:47` | `MTP_LINEARS` |
| `drafter/export_mtp.py:67-69` | `MTP_LINEARS` |
| `drafter/requant_mtp_gptq.py:20-22` | named **`LIN`** |
| `drafter/train_mtp.py:397-404` | dict literal mapping the same names to modules |

Same eight names (`mtp.fc`, three MLP projections, four attention
projections), six times, three spellings.

### 1.5 The draft-head contract has two writers, and history lives in comments

The contract — `mtp.draft_lm_head.weight_packed/weight_scale/weight_shape` in
`model_extra_tensors.safetensors` + `mtp_draft_vocab_ids.pt` + weight_map
entries — is consumed by `patches/qwen3_5-mtp-draft-vocab.patch:33-41`
(reads the ids file, builds `ParallelLMHead(numel)`). It is written twice:
- `prepare/build_draft_vocab.py:91-126` — slices rows out of the **already
  packed** lm_head (the canonical writer; fused to a corpus-counting data
  tool, `:38-89`),
- `drafter/export_mtp.py:108-129` — requantizes a **trained bf16** draft head
  with its own RTN copy + write tail (the fifth RTN site, 1.1),
and `export_mtp.py:104-106` *also* subprocesses the first one when the
checkpoint lacks a trained head. Two writers with a real semantic difference
(slice quantized rows vs quantize a bf16 head), neither documented next to
the contract.

The variant-dir assembly (hardlink shards + copy config family) exists 4×:
`prepare/fetch_fast_variant.py:23`, `drafter/export_mtp.py:24-29`,
`drafter/gptq_lm_head.py:75-82`, `drafter/requant_mtp_gptq.py:24-29`. The
inode-truncation trap — `save_file`/`torch.save` truncate in place, so
through a hardlink a variant rewrite silently rewrote the **source** (the
bug story is in `drafter/gptq_lm_head.py:84-88`) — is currently transmitted
as comments: `export_mtp.py:52-53` (`os.remove(D+f)  # never truncate a
hardlink`), `requant_mtp_gptq.py:55-56`. A bug that already bit once is a
guardrail only where someone remembered to write it.

### 1.6 The backup-suffix protocol

`.bak` (`quant_lm_head.py:59`), `.bak-quant` (`:62,69`), `.bak-mtp`
(`export_mtp.py:34-35,90-95`; **read** by `requant_mtp_gptq.py:34`),
`.bak-draft` (`build_draft_vocab.py:117-118`). A filename protocol crossing
six files, never written down in one place — `export_mtp.py:34-35` restores
"pre-MTP-quant" state by copying `.bak-mtp` files back over the live ones.

### 1.7 The import shims

`sys.path.insert` ×5, all in `drafter/` (`capture_dflash2.py:23`,
`export_mtp.py:63`, `gptq_lm_head.py:13`, `quant_dflash2.py:24`,
`requant_mtp_gptq.py:13`) — each script hand-rolling its access to
`gptq_utils`. `prepare/` has no access at all.

## 2. The design: `prepare/pipeline_core.py`

One stdlib+torch module in `prepare/`. `prepare/` scripts import it bare
(same directory — no shim). `drafter/` scripts replace their five shims with
the one documented two-liner that adds the repo root (a pattern with a
comment, not five accidents). It is a bag of functions and constants — no
framework, no classes, no pipeline abstraction:

```text
QMAX/BITS/GROUP constants (per-call overridable)
MTP_LINEARS: the eight names, once (1.4)

rtn_quantize(W, bits, group, *, eps, keepdim=True)
    the 1.1 math with the eps EXPLICIT — each migrated caller passes its
    current value (1e-10 or 1e-8), so outputs are bit-identical per script
    (the acceptance gate in §4); the signature documents the difference

write_packed(tensors, key, q, scale, bits, scale_dtype)
    the write tail (1.2) with the 1.3 dtype difference as a required
    argument, not a comment

repair_ignore_list(config, keys) / clone_group(qc, src, name, targets, bits)
    the ignore-list repair moves out of quant_lm_head.py (1.4) and the
    group_0→group_N clone becomes one function with the ordinal naming
    visible

assemble_variant(src, dst, hardlink=[patterns], copy=[patterns])
    the 1.5 assembly, once

fresh_write(path, save_fn)
    unlink-before-write — the inode-truncation guardrail as code
    (gptq_lm_head.py:84-88's bug can never be reintroduced silently)

write_draft_head(model_dir, ids_or_head, *, source="sliced"|"trained")
    the contract's single writer (1.5); the slicing path calls
    build_draft_vocab's logic, the trained path the export_mtp logic —
    the semantic difference becomes a named parameter

BACKUP suffix constants (.bak, .bak-quant, .bak-mtp, .bak-draft) + a
protocol section in the module docstring (1.6)
```

**Deliberately not in the module:** the two loaders (in-RAM whole-shard vs
`quant_heads_stream.py`'s streaming rewriter — both are load-bearing,
sharing internals not I/O); anything GPTQ (`gptq_utils.py` keeps
`gptq_quantize`, `accumulate_hessian`, and loses `rtn_quantize`/`dequant` to
the core it can now import); the corpus counting in `build_draft_vocab.py`
(a data tool, stays with its script); training/capture scripts.

## 3. Rollout

### 3.1 PR A — the core + the prepare/ migration

Add `prepare/pipeline_core.py` and `bench/test_pipeline_core.py` (§4).
Migrate the four `prepare/` quant scripts + `build_draft_vocab.py`'s
slicing writer. **Bit-identical acceptance:** run each migrated script on a
copy of a small prepared fixture model, before and after, and `diff` the
resulting shard/index/config bytes — the eps-per-caller rule (2.1) exists
to make this gate pass.

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
`export_mtp.py:108-129` calls it with `source="trained"`. The ignore-list
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
| `export_mtp.py`'s dual nature (delegates AND re-implements) hides a third draft-head path | low | PR C makes the writers one function with a named `source=` parameter; the subprocess call at `:105-106` remains as orchestration |

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
