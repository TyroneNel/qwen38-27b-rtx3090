# Phase 1 review: project map (inventory.md, architecture.md)

Reviewer: Opus, high effort. Date: 2026-10-21. No GPU and no git operations were used. All references were read-only.
Evidence paths are relative to the repo root unless they start with `~/`. `gpu-results/` means `~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/`.

## Verdicts

| File | Verdict | Summary |
|---|---|---|
| `docs/research/01-project/inventory.md` | **PASS WITH CORRECTIONS** | The patch series, pin, layer split, knobs and measured numbers hold. 17 wrong or approximate cites and values were fixed in place. The main fix: the D' KV-path precedence was wrong. |
| `docs/research/01-project/architecture.md` | **PASS WITH CORRECTIONS** | The analysis holds. The Mermaid launch chain had the wrong edges. One headline number was stale, the B pool ignored the prefix cache, and several `~` and off-by-few cites were fixed. |

Totals: 46 claims checked (24 in inventory, 22 in architecture). 23 CORRECT (some were incomplete and have been completed), 20 WRONG (about half are wrong cites or paths with correct values), 2 UNSUPPORTED, and 1 open question answered. Every WRONG and UNSUPPORTED item has been fixed or relabelled in the doc.

## Primary evidence fetched

- HF `config.json`, `dbirks/Qwen3.8-27B-W4A16-AutoRound` at commit `1f05c441`. Metadata only, saved to the scratchpad and not committed:
  - `num_hidden_layers` 64 and `full_attention_interval` 4.
  - `layer_types` has 48 `linear_attention` and 16 `full_attention`, at layers 3, 7, ..., 63.
  - `num_key_value_heads` 4, `head_dim` 256, `num_attention_heads` 24.
  - `mtp_num_hidden_layers` 1 and `mamba_ssm_dtype` float32.
- `patches/series`: 51 non-comment entries at `:17-67`. There are 51 `.patch` files with none missing and none extra. The order equals the inventory table. The KVarN patches are in `kvarn/`, not `patches/kvarn/`.

## Checked claims

### inventory.md

| # | Claim | Doc location | Evidence | Status | Note |
|---|---|---|---|---|---|
| 1 | vLLM pinned to 0.31.0 | pin row | `docker/requirements.txt:5`; `fa2/PIN`, `pthfa/PIN`; CI check `.github/workflows/patch-integrity.yml:96-100` | CORRECT | The `~/vllm` reference is v0.32.1rc0, so code there may not match the pin (added as R1) |
| 2 | 51 patches, applied in `series` order with `--fuzz 0` | patch table | `patches/series:17-67`; `patches/apply.sh` | CORRECT | The order matches the table row for row |
| 3 | 48 GDN + 16 full-attention layers | target base row | HF config above; `docs/optimizations.md:55`, `:250-251`; `docs/multi-gpu.md:16` | CORRECT | Config evidence added to the row |
| 4 | Launcher default `SPEC=mtp`; `.env.example` sets dflash2 | SPEC knob | `single-user/start_qwen.sh:197`; `resolve_config.sh:57`; `.env.example:6`, `:41` | CORRECT | `docker/entrypoint.sh:3` comment also says MTP |
| 5 | xgrammar pin source | deps row | `docs/vllm-0.31.md:11` | WRONG (cite) | Re-cited |
| 6 | dflash2 fast k=7 pool 69,758 | profile table | `docs/optimizations.md:254-255`; `single-user/README.md:170-171` | WRONG (incomplete) | 68,605 with the default prefix cache. Note added |
| 7 | Mode C pool 57,669 at `long-context.md:~120` | profile table | `docs/long-context.md:129` | WRONG (cite) | Value correct, cite fixed |
| 8 | D' default is `FA2_FP8KV=1` | D' row | `resolve_config.sh:95-116`, `:130-131`; `Dockerfile:33` | WRONG | PTH_FA takes precedence, so the default is CUSTOM int8 in the image |
| 9 | D' pool 136,429 at `:~125` | D' row | `docs/long-context.md:128` | WRONG (cite) | Value correct |
| 10 | Batch kvarn / int4pth line ranges | batch table | `batch/start_qwen.sh:122-128`, `:118-121` | WRONG (cite) | Fixed. WSL2 override row added (`:105-117`) |
| 11 | `GPU_UTIL` 0.94 batch / 0.93 single | knob table | `batch/start_qwen.sh:149`, `:113-116`; `single-user/start_qwen.sh:175` | CORRECT (incomplete) | WSL values added |
| 12 | `DRAFT_METHOD`, `LOOKUP`, `DYNAMIC_K` cites | knob table | `single-user/start_qwen.sh:394`, `:370`, `:683-693` | WRONG (cite) | Three cites fixed |
| 13 | KVarN patch path `patches/kvarn/...` | KVarN row | `kvarn/kvarn-fp16-dequant-0.31.0.patch` | WRONG | Path fixed |
| 14 | `PORT`/`BIND` default host | knob table | `launcher_common.sh:92-109` (0.0.0.0 in a container, `:102-104`) | WRONG (incomplete) | Container note added |
| 15 | `SSE_KEEP_ALIVE` default unset | knob table | `single-user/start_qwen.sh:129` | WRONG | Default is 30 |
| 16 | Retired patches list | retired section | `PATCHES.md:90-99`, `:101-106` | WRONG (incomplete) | int4-mq3d-envs added; the file is at the repo root |
| 17 | Split-KV verify "8-16 queries" | optimizations table | `docs/optimizations.md:91-94` | WRONG | MTP verify is five queries |
| 18 | 1.3 / 7.4 / 10.1 ms "on 3080 TP=2" | optimizations table | `docs/long-context.md:145-146` | UNSUPPORTED | The passage names no hardware. Label removed, cite fixed |
| 19 | Entrypoint runs prepare -> verify -> launcher | launch section | `docker/entrypoint.sh:21-34` | CORRECT (incomplete) | `select_model.sh` runs for single at `:26-30` |
| 20 | `verify.sh` section line cites | verify section | `verify.sh:43,77,114,148,182,336,344,369,376` | CORRECT | |
| 21 | Dockerfile build steps | build section | `Dockerfile:14,18-19,24,32-33,37-39,49-56,61,63-65` | CORRECT | |
| 22 | Single-user per-profile `MAX_SEQS`/`MAX_LEN`/`KV_MEM` | profile table | `single-user/start_qwen.sh:478-520` | CORRECT | `KV_MEM` 5261334938 at `:495` |
| 23 | Adaptive-length corruption 38/793; probabilistic 2.66 vs 3.23 | knob notes | `single-user/start_qwen.sh:399-428`, `:383-390` | CORRECT | |
| 24 | Carried-over prior-box numbers (A-E headline, pools, E profile, spec-5, LEDGER) | measured tables | `gpu-results/baseline.md:23,35-49,58,65,69`; `e-profile.md:10,31-54`; `spec-5.md:32-51`; `LEDGER.md:24,25,35,48`; `quant-2-check.md:26,35`; `ENVIRONMENT.md:35-38` | CORRECT | All re-traced to primary gpu-results files |

### architecture.md

| # | Claim | Doc location | Evidence | Status | Note |
|---|---|---|---|---|---|
| 1 | Mermaid: entrypoint -> resolve_config -> launcher_common -> launchers | pipeline diagram | `single-user/start_qwen.sh:103,111,120`; `batch/start_qwen.sh:66,75`; `docker/entrypoint.sh:34` | WRONG | The entrypoint execs the launcher, which sources the libs. Edges fixed; Compose prepare and select_model added |
| 2 | Compose runs prepare before serve | diagram | `docker-compose.yml:85-87`, `:107-109` | UNSUPPORTED (missing) | Added as node A0 |
| 3 | Launcher ends in `qwen_exec vllm serve` | diagram | `single-user/start_qwen.sh:1027-1046`; `batch/start_qwen.sh:301`; `launcher_common.sh:120` | CORRECT | |
| 4 | "same target weights, int8 embed/lm_head" across modes | mode table preamble | `single-user/select_model.sh:4-7`; `prepare/fetch_fast_variant.py:2`; `single-user/README.md:295` | WRONG | The lm_head is GPTQ-int4 on the fast variant (B-E) |
| 5 | B pool 69,758 | mode table | `single-user/README.md:170-171`; `baseline.md:49` | WRONG | 68,605 with the default prefix cache |
| 6 | A `GPU_UTIL` 0.94 | mode table | `batch/start_qwen.sh:113-116`, `:149` | CORRECT (incomplete) | 0.91 on WSL2 |
| 7 | A headline 942 e2e | headline row | `README.md:108` (948); `batch/README.md:18,21` (942 on 0.27.1) | WRONG | Stale; the current tree says 948 |
| 8 | B 131.2/127.1, C 381, D 100.3, E 67/164 | headline row | `single-user/README.md:74`, `README.md:110`, `single-user/README.md:51`, `README.md:85` | CORRECT | Cites added |
| 9 | 950 tok/s prediction "measured via `batch/README.md:18`" | throughput table | `batch/README.md:18,21`; `deepinfer/runtime.md:20` | WRONG (label) | Only 1,094/942 are measured. 950 and 1,812 are estimated arithmetic from the prior map |
| 10 | Embed UVA saving at `patches/PATCHES.md` row 48 | memory table | `PATCHES.md:75` | WRONG (cite) | |
| 11 | CUDA graph 1.45 / 1.8 GiB; upstream reports 0 | memory table | `docs/optimizations.md:255-257`; `single-user/start_qwen.sh:496-499` | WRONG (cite, stale) | "Reports 0" is 0.28-era; 0.29+ profiles graphs |
| 12 | Activation swing ~1 GiB | memory table | `docs/optimizations.md:258`, `:195-197` | CORRECT (cite fixed) | Was `:262-268` |
| 13 | bf16 KV 64 KiB/token | KV table | `docs/long-context.md:121`, `:137`; HF config | CORRECT | `~` cite resolved; 16 x 2 x 4 x 256 x 2 B = 65,536 B (estimated) |
| 14 | int8 KV 138,696 / 136,429 | KV table | `docs/long-context.md:128` | CORRECT | `~` cite resolved |
| 15 | State pages 0.88 / 1.66 / 0.44 GiB; 15.8% | KV table | `docs/long-context.md:694-697`; `docs/gotchas.md:317-330` | CORRECT | `~` cite resolved; 15.8% is live, 0.88 is fitted |
| 16 | Seats 7 / 5 / 2; k=15 gives 3 | KV table | `docs/long-context.md:696-697`, `:762` | CORRECT | `~` cite resolved |
| 17 | State pages scale with `num_speculative_tokens` | spec section | `single-user/start_qwen.sh:679-681` | CORRECT (cite fixed) | Was `:677-679` |
| 18 | Split-KV query tiling at `spec-decode-attn.patch:117-171` | open question Q2 | `patches/spec-decode-attn.patch:180-194,206-207,226-239` | WRONG (cite) | `:117-171` is the enable gate and file header |
| 19 | Q2: unknown whether tree masks work | open question Q2 | `patches/spec-decode-attn.patch:84-89`, `:239`, `:268` | Answered | Causal-only routing and a positional mask, so a tree mask needs a kernel change |
| 20 | DFlash2 lookup drafting cites | spec section | `patches/dflash2-lookup-drafting.patch:346,417,504,948,1071` | CORRECT | |
| 21 | Sampler patch cites | spec section | sampler patch `:63`, `:84` | CORRECT | |
| 22 | Prior-box D' resolved to PTH_FA=1, numbers 55.7 / 48.8 | baseline section | `gpu-results/baseline.md:23`, `:35-47` | CORRECT | Consistent with the precedence fix in inventory #8 |

## Missed items

1. **The PTH_FA default beats FA2_FP8KV for D'** (`resolve_config.sh:95-116`, `:130-131`; pthfa is built into the image, `Dockerfile:33`). The docs implied FA2 fp8 for D'. `README.md:96-101` is stale on the same point.
2. **The upstream HyperQwen PTH_FA default also covers `mtp`/`long`** (`~/research-src/HyperQwen/resolve_config.sh:105-107`). This repo does not have it, so mode D stays on FlashInfer fp8.
3. **The Compose `prepare` service, and `select_model.sh` in the entrypoint**, are missing from the launch chain (`docker-compose.yml:85-87`, `:107-109`; `docker/entrypoint.sh:26-30`).
4. **The B pool depends on the prefix cache**: 68,605 by default vs 69,758 with the cache off (`single-user/README.md:170-171`).
5. **WSL2 batch overrides** (`batch/start_qwen.sh:105-117`) change A's `GPU_UTIL` and the kvarn `MAX_LEN`. The prior GPU box was WSL2.
6. **`weights_breakdown.out` exists but neither doc cites it.** It gives measured header bytes; resident sums are A 14.136, D-fast 13.849, B/C/E 14.737 GiB (estimated).
7. **The fast variant changes lm_head precision** (GPTQ-int4), so "same target weights" across modes is false.
8. **The `~/vllm` reference tree is v0.32.1rc0, not 0.31.0.** Line cites into it need a pin check.
9. **The k=4 crash claim is 0.28.0-era** (`docs/optimizations.md:405-410`). It is not re-verified on 0.31.
10. **`docs/optimizations.md:3` says "Nine" optimizations, but the list at `:11-41` has ten.** This is a repo doc nit; nothing was edited.

## Corrections made

All corrections are marked inline with `[review: ...]`.

### inventory.md

1. xgrammar row: re-cited to `docs/vllm-0.31.md:11`.
2. Target base row: added the HF config evidence for 48 + 16.
3. dflash2 fast k=7 row: pool cite set to `docs/optimizations.md:254-255`, with a note that the pool is 68,605 with the prefix cache.
4. Mode C pool: `long-context.md:~120` changed to `:129`.
5. D' row: added a note on PTH_FA precedence, and changed `:~125` to `:128`.
6. Batch table: kvarn cite `:122-128`, int4pth cite `:118-121`, and a new WSL2 overrides row `:105-117`.
7. `GPU_UTIL` knob: WSL values added.
8. `DRAFT_METHOD` cite changed to `:394`.
9. `LOOKUP` cite changed to `:370`.
10. `DYNAMIC_K` cite changed to `:683-693`.
11. `FA2_FP8KV` knob: precedence note added.
12. KVarN path changed to `kvarn/kvarn-fp16-dequant-0.31.0.patch`.
13. `PORT`/`BIND`: container 0.0.0.0 note added; cite `launcher_common.sh:92-109`.
14. `SSE_KEEP_ALIVE` default changed from unset to 30 (`:129`).
15. Retired patches: cite `PATCHES.md:90-99`, plus int4-mq3d-envs (`:101-106`).
16. Split-KV row: "8-16 queries" changed to the MTP five-query case (`docs/optimizations.md:91-94`).
17. `long-context.md:~145` changed to `:145-146`, and the unsupported "3080 TP=2" label removed.
18. New "Reviewer additions" section with R1-R5: pin locations vs `~/vllm`, the stale README FA2 text, Compose prepare and select_model, the B pool with cache, and the weights_breakdown sums.

### architecture.md

1. Mermaid launch chain: the A2 text now lists select_model, and a Compose prepare node (A0) was added. The edges now read: entrypoint execs the launcher (single or batch), and each launcher sources launcher_common and resolve_config. A `%%` review comment was added.
2. Mode preamble: corrected "int8 embed/lm_head" (the lm_head is GPTQ-int4 on the fast variant).
3. B pool changed to 68,605 with the default prefix cache (69,758 with it off).
4. A `GPU_UTIL`: WSL2 0.91 added.
5. A headline changed from 942 to 948 e2e (`README.md:108`), and cites added for B, C, D and E. The D' FA2 arm numbers are now attributed to `README.md:101`.
6. 950 tok/s throughput row relabelled. Only 1,094/942 are measured (vLLM 0.27.1); 950 and 1,812 are estimated, from `deepinfer/runtime.md:20`.
7. `patches/PATCHES.md` row 48 changed to `PATCHES.md:75`.
8. CUDA graph cite changed to `docs/optimizations.md:255-257` and `start_qwen.sh:496-497`. "Upstream reports 0" is flagged as 0.28-era.
9. Activation cite changed from `:262-268` to `:258`.
10. `long-context.md:~117` changed to `:121`/`:137`, with the HF config agreement noted.
11. `:~122-130` changed to `:128`.
12. `:~695` changed to `:694-697`, and the gotchas cite to `:319-320`; the live vs fitted distinction is noted.
13. `:~697` changed to `:696-697`.
14. `start_qwen.sh:677-679` changed to `:679-681`.
15. Q2 answered: the tiling cite is fixed, and the patch is shown to be causal-only (`spec-decode-attn.patch:87-89`, `:239`, `:268`).
16. New "Reviewer additions" section: weights_breakdown sums, the B pool with cache, the D' PTH_FA default, and the WSL2 batch defaults.

## Open questions for the orchestrator

- Should the baseline arm for mode D be re-measured with the upstream `mtp`/`long` PTH_FA default, or kept as FlashInfer fp8 to match this repo?
- `README.md:96-101` is stale on the D' default. That file is out of scope here, so it is flagged and not edited.
