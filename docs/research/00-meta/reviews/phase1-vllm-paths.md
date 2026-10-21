# Review: `docs/research/03-vllm/integration-paths.md`

Reviewer: phase-1 reviewer (Opus, read-only, no GPU). Date 2026-10-21.

Method. I rebuilt both trees under my scratch dir with `git -C ~/vllm archive` (no worktree, so there was nothing to remove). `[P]` = v0.31.0 + `bash patches/apply.sh` + `kvarn/files` + `apply.sh --kvarn`. All 51 + 4 patches applied at fuzz 0 (measured; offsets only). `[H]` = HEAD `4bd75e952a`, plus a pristine v0.31.0 copy. `git -C ~/vllm status` was clean at the start and at the end. "measured" = command output I ran. "estimated" = my reasoning.

## Verdict

**PASS WITH CORRECTIONS.**

The load-bearing conclusions hold:
- There is no EAGLE-4 support in vLLM 0.31.0 or HEAD.
- The host is `074c83a09a` + `eagle-hooks`.
- 18/33 is right.
- There is a `CUSTOM` clash with pthfa.
- Tree verify needs unquantised KV.
- The effort ratings are defensible.

I made nineteen inline corrections. Most fix line anchors or version facts. Three are substantive:
- 11 of the 33 HEAD failures are stack-dependent, so they are not proven drift.
- The int4 per-token-head KV dtype is already upstream in 0.31.0.
- Row 5 says to share `lm_head`, but EAGLE-4's drafter has its own.

Two items affect effort but were missing: `CUSTOM` is the **default** on the 3090, and the 24 GB memory cost of the drafter's own embedding and lm_head.

## Checked claims

| # | Claim | Doc section | Evidence | Result | Note |
|---|---|---|---|---|---|
| 1 | Pinned vLLM 0.31.0 | §0 | `docker/requirements.txt:5` | CORRECT | Line was `:6`, fixed |
| 2 | 51 series patches + 4 KVarN patches | §0 | `grep -v '^#' patches/series \| grep -c .` → 51; `patches/apply.sh:49-50` KVARN list; `Dockerfile:50-51` | CORRECT | Rebuild reproduced at fuzz 0 |
| 3 | HEAD is 1262 commits ahead of 0.31.0; 1411 files, +132,523/-43,893 | §1 | `git rev-list --count v0.31.0..HEAD` = 1262; `git diff --shortstat` identical | CORRECT, with a nuance | v0.31.0 is **not** an ancestor of HEAD: 27 release-branch commits are missing from HEAD (`git rev-list --count HEAD..v0.31.0`), merge-base `3d5f4d4cd5`. Note added |
| 4 | `git describe` gives `v0.32.1rc0-68` | §1 | Plain `git describe` → `proto-v0.6.0-328-g4bd75e952a`; `--tags --match 'v*'` → `v0.32.1rc0-68-g4bd75e952a` | WRONG (command) | Fixed |
| 5 | 18 of 51 apply to HEAD, 33 fail | §1, §6 | All 51 checked with **both** `git apply --check -p1` and `patch --dry-run -F0` on the HEAD export. Identical: 18 ok / 33 FAIL. The per-patch ok/FAIL column matches. Hunk lines for six patches re-checked: dflash2-lookup-drafting (306, 2228), spec-decode-attn (1447), hybrid-sw-block-promote, qwen3_5-embed-uva (187, 1559), kv-cache-seat-budget (2664), sampler-warmup-cuda (444) | CORRECT | — |
| 6 | "failures here are context drift, not cascades" | §1 | 11 of the 33 also fail alone on pristine **v0.31.0**, because they need context from earlier series patches (list inline in the doc) | WRONG | Only 22 are proven HEAD drift. Fixed |
| 7 | No EAGLE-4 support in HEAD or 0.31.0; the hooks are absent | §7, §1 | `grep -rl` for `register_speculative_method`, `SpeculativeMethodSpec`, `prepare_verify_inputs`, `make_rejection_sampler`, `post_accept`, `uses_varlen_verify_graphs`, `LinearAttnVerifyExtension`, `method_config`, `verify_budget`, `suspend_drafting`, `eagle4\|eagle_harness` → 0 files in both `[P]` and `[H]`. `git grep` at `074c83a09a` → 0 | CORRECT | Hook list: `EAGLE-4/eagle_harness/vllm/adapter.py:25-71` |
| 8 | EAGLE-4 needs `eagle-hooks` on top of `074c83a09a`, which is 155 behind HEAD and in history | §1, §7 | Stated at `EAGLE-4/eagle_harness/compat.py:54` and `eagle_harness/vllm/__init__.py:7`, `plugin.py:12`. `git cat-file -t 074c83a09a` = commit (2026-10-16, #61932). It is an ancestor of HEAD and not of v0.31.0. `rev-list` 074c83a09a..HEAD = 155; v0.31.0..074c83a09a = 1107 | CORRECT | Branch-list description was wrong (515 refs, many `origin/*`). None is `eagle-hooks`, and no commit message mentions it. Fixed |
| 9 | Tree verify needs unquantised KV | §7, §8 row 3 | `EAGLE-4/docs/known-limits.md:48`; `eagle_harness/vllm/tree_attention.py:69-70` raises `NotImplementedError` | CORRECT | — |
| 10 | EAGLE-4 binds the `CUSTOM` slot that pthfa also uses | §4, §8 | `EAGLE-4/eagle_harness/vllm/plugin.py:58` (unconditional `register_backend`) vs `pthfa/python/pth_flash_attn/plugin.py:37-45` (yields if already bound). `[P]` `registry.py:155` `CUSTOM = None` | CORRECT | Missed: `CUSTOM` is the **default** launch on 8.6 (see Missed items) |
| 11 | The spec-decode Mermaid anchors resolve | §2 | 30 `[P]` anchors resolved by `sed -n` in the rebuilt tree: speculative.py:425/429/492, vllm.py:701, model_runner.py:370/484/741/751/1578/1627/1698/2079/2233, rejection_sampler.py:237/359, dflash2/speculator.py:29/152/175/222/462/726/784/798, dflash/speculator.py:91/155/211/280/366/759, qwen3_dflash2.py:415, lookup.py:236/283, models/utils.py:962, qwen3_dflash.py:468, qwen3_5.py:403-406, scheduler.py:292-296/1446-1448 | CORRECT | Small slack only: `:2247` is the line before the `draft_tokens` assignment (`:2248`); `:713` is `draft_logits_spec`; `dflash/utils.py:15` is blank (the function is at `:16`) |
| 12 | Launcher anchors (`start_qwen.sh:686-696` JSON, `:697-701` mtp, `~672-675` +7.7%) | §2 | JSON is at `:394` (dflash2) and `:666` (mtp); `DYNAMIC_K` at `:683-694`; the +7.7% comment at `:679` | WRONG | Fixed. `:283`, `:1040` and `~818-838` are fine |
| 13 | `[H]` speculator factory "adds LiLiCorr"; `__init__.py:13-110` | §1 | LiLiCorr is already at `[P]` `__init__.py:18-23`. `[H]` file has 88 lines | WRONG | Fixed |
| 14 | "int4 is our patch"; `int4_per_token_head.py` is "new" | §1, §6.2 | Pristine v0.31.0 has `int4/int8/fp8_per_token_head` (`config/cache.py:53-55`, `triton_attn.py:301-303`) and the ops file. Our patch adds 17 lines (boot-blocker fixes) | WRONG | Fixed in both places |
| 15 | `prefill_attn_hd256.py` under `ops/` | §6.2 | The patch creates `v1/attention/backends/prefill_attn_hd256.py` | WRONG (path) | Fixed |
| 16 | Split-KV kernel is linear-causal, at `spec_decode_attn.py:33-35` | §4, row 3 | The claim is true, but it sits at `:21-22`; `QMAX_TOKENS` at `:53` and restrictions at `:40-41` are correct | WRONG (anchor) | Fixed in both places |
| 17 | V2 refuses ngram…custom_class `[P]`; only suffix/custom_class `[H]` | §1 | `[P]` `config/vllm.py:3073-3081`; `[H]` `:3030` | CORRECT | — |
| 18 | Stock FA fp8 KV closed on sm86; Triton dtype list `[H]` | §4 | `[H]` `fa_utils.py:282-312`, `flash_attn.py:433-440`, `triton_attn.py:324-334` | CORRECT | — |
| 19 | Tree attention removed in #42121 `b1728c1e66`, before 0.31.0 | §1 | `git merge-base --is-ancestor` → yes; 0 `tree_attn` files in either tree | CORRECT | — |
| 20 | Row 1 chain-mode effort **M** | §8 | The code needed is small: `method=eagle3` already routes to the V2 `EagleSpeculator` (`[P]` `__init__.py:57-62`), and `Eagle3LlamaForCausalLM` already handles `fc_norm`/`norm_output`/`layer_types`/`midlayer` (`llama_eagle3.py:231-320`). EAGLE-4's class subclasses it (`EAGLE-4/eagle_harness/vllm/models.py:43,145`) | CORRECT (rating), WRONG (rationale) | M is justified by memory and loader work, not by a factory branch (none is needed). "None specific to sm86" was wrong: the 24 GB budget blocks it. Annotated |
| 21 | Row 5 drafter quantisation **S** | §8 | `get_draft_quant_config` reaches the layers and `lm_head` of the Eagle3 Llama class (`llama_eagle3.py:108,177,332` `[P]`) | CORRECT (rating) | Anchor pointed at `qwen3_eagle3.py`, the wrong class. "Keep lm_head shared" is wrong for EAGLE-4 (own trained lm_head, tie false). Both fixed |
| 22 | Row 2 KV dtype **S** (config only) | §8 | DFlash: `dflash/utils.py:32-39` and `dflash/speculator.py:146-148`. EAGLE: `apply_draft_overrides` at `eagle/utils.py:79`, but `attn_vllm_config` is not overridden (`speculator.py:269-272`) | PARTLY SUPPORTED | S holds for DFlash; for EAGLE it is unverified. See Reviewer additions |
| 23 | Row 6 KV dtype **M** per dtype | §8 | Patch sizes (measured `grep -c '^+[^+]'`): spec-decode-int8-kv 87, triton-spec-attn-fp8-kv 116; KVarN 155+86 patch lines + 4,758 lines of new files | CORRECT for a variant | A new format is L. Note added |
| 24 | Row 3 tree verify **L** | §8 | Four coupled changes; the plugin is 4,011 lines (`wc -l`, measured); hooks are absent | CORRECT | The compaction anchor said "after propose"; it must run before (`EAGLE-4 speculator.py:17-18`). Fixed. **Estimated**: well above the 4-day L floor |
| 25 | Required host, wheel version, plugin entry point | §7 | `compat.py:51-56`, `docs/release.env:21`, `pyproject.toml:37`, `plugin.py:31-67` | CORRECT | — |
| 26 | Kernel arch (sm_80+ Triton, sm_103 Gluon, SASS list, no sm86 results) | §7 | `docs/kernels.md:18,20,21,55`; `kernels/arch.py:14,31`; grep for `sm_86\|3090\|Ampere` in `docs/` → 0 | CORRECT | — |
| 27 | `num_speculative_tokens_per_batch_size >= 0` at `[H]` `dynamic/utils.py:~54` | §2, row 4 | `[H]` `v1/spec_decode/dynamic/utils.py:52-55` | CORRECT | — |
| 28 | Quant path (`get_marlin_input_dtype:720`, fp8 sm89/sm12x only, exclude default `lm_head\|mtp`, `_POSSIBLE_KERNELS` order) | §3 | `[P]` `marlin_utils.py:720,735-742`; `envs.py:200-202`; `kernels/linear/__init__.py:500-504` | CORRECT | — |

Tally: 28 claims checked. 17 correct, 2 correct with a nuance or rationale problem, 1 partly supported, 8 wrong. Of the 8 wrong, 5 are anchors, paths or commands (#4, #12, #13, #15, #16), and 3 are substantive (#6, #14, and the row 5 lm_head advice in #21).

## Missed items

1. **pthfa `CUSTOM` is the default long-context launch on the 3090.** `resolve_config.sh:87-113` sets `PTH_FA=1` on `SPEC=dflash2 CTX=long` on cc 8.6, which means `CUSTOM` + `int8_per_token_head` KV (`start_qwen.sh:346-351`). EAGLE-4 tree mode needs both `CUSTOM` and bf16 KV, so it conflicts with the shipped default, not an option. This raises the cost of row 3. Added to Reviewer additions.
2. **The drafter's own embedding and lm_head on 24 GB.** The EAGLE-4 Qwen drafter has its own `lm_head` and `embed_tokens`, 2.54 GB BF16 each (from `02-eagle4/architecture.md` sec. 1, measured header shapes). V2 shares them only on `torch.equal` (`eagle/utils.py:15-27,38-70,100-103` `[P]`). Our target embedding is quantised (`qwen3_5-embed-quant`), so sharing probably fails (**estimated**). This is the real row 1 blocker.
3. **The `+eagle.4` build is not public yet and carries non-hook changes.** Sources: `docs/release.env:4` ("Publication is a separate release gate"), `install.md:21-22` (private index), `support-matrix.md:151` (vLLM image "Pending"), and `compatibility.md:31-33` (adds KV cache group sizing, a decode GEMM and native target decoding). None of the hook source is in the EAGLE-4 repo. Annotated on open question 2.
4. **The plugin is all-or-nothing.** `register()` refuses when **any** `HOOK_MAP` hook is missing (`plugin.py:36-37`, `adapter.py:74-76`). So even EAGLE-4's "chain mode, only the registry" path needs the full hook set. Chain mode on our stack has to use stock vLLM classes, not the plugin.
5. **The plugin also registers `Eagle3ForHarness`** (`plugin.py:64-65`). This is a second draft arch for plain EAGLE-3 checkpoints. It is not mentioned in the doc. It does not matter for our target.
6. **A 0.32 rebase is necessary but not sufficient.** §1's reading ("rebase … is a prerequisite") omits that tree mode also needs the unpublished hooks. Annotated on open question 1.

## Corrections made in `integration-paths.md`

Every edit is marked inline `[review: …]` (18 lines touched, 19 edits):

1. §0: `requirements.txt:6` → `:5`.
2. §1 Release row: `git describe` needs `--tags --match 'v*'`.
3. §1 Distance row: v0.31.0 is not an ancestor of HEAD (27 commits, merge-base `3d5f4d4cd5`).
4. §1 Speculator factory: `[H]` `:13-88`; LiLiCorr is already in `[P]` (`:18-23`).
5. §1 Triton KV dtypes: all three per-token-head dtypes are upstream in 0.31.0.
6. §1 Our series on HEAD: re-measured; 11 of the 33 failures are stack-dependent.
7. §2 intro: launcher JSON anchors `:394`/`:666`/`:683-694`.
8. §2 note 5: the +7.7% comment is at `:679`.
9. §4: causal addressing at `:21-22`.
10. §6.2 int4 row: the upstream file is patched, not new.
11. §6.2 prefill row: the path is `backends/`.
12. §7 Required host: branch-list facts; `074c83a09a` has no hooks.
13. §8 row 1 blockers: the 24 GB embedding/lm_head cost.
14. §8 row 3: causal anchor.
15. §8 row 3: compaction sits before `propose`.
16. §3 drafter quantisation: the right class is `llama_eagle3.py:108,177,332`.
17. §8 row 5: do not share `lm_head` for EAGLE-4.
18. §9 Q1: hooks are also needed.
19. §9 Q2: wheel index, publication status and non-hook changes.

I also added a short **Reviewer additions** section at the end of the doc. It covers the `CUSTOM` default, the chain-mode code scope, the EAGLE drafter KV-dtype caveat, the row 6 calibration and the drafter memory figures (estimated).

Evidence files (scratch, not committed): `head-apply.txt`, `head-gitapply.txt` and `edit_doc.py` under `/tmp/claude-1000/-home-tyrone-qwen38-27b-rtx3090/420ee804-24fe-46ef-9987-fed233d9db09/scratchpad/`.
