# Findings: bugs, fixes and patches from the decode-speed work

The inventory the batch records imply but never collected: what was found, what holds it, what has been done about
it, and what is still open. Sourced from batches 1-3 on the reference 3090 plus the landscape survey.

**How to use this file.** Every item has an id, the evidence that established it, a status, and the next action.
Change a status only with evidence — a commit, a result path, or a quoted command output — and put the pointer in
the item. "Open" means nobody has started; "verified" means we checked and it does not affect us. Related pages:
[landscape](decode-speed-landscape.md) · [batch 2](decode-speed-batch2.md) · [batch 3](decode-speed-batch3.md) ·
[batch 4](decode-speed-batch4.md) · [PATCHES.md](../PATCHES.md) (the patch ledger — what we ship, not what we
found).

## O — bugs in our own code

| id | what | evidence | status | next action |
|---|---|---|---|---|
| O-1 | **The `RESIDENT` constant is 16-23% low.** `single-user/start_qwen.sh` budgets `104988089` bytes per (k+2) unit; measured cost is 81-88 MB, so the boot print says "about 5 seats" where 7 actually run | batch 3: pool shares 14.2 / 7.45 / 4.37% at k=7/3/1 and `Running:` 7/13/16 | **open** | one-line constant fix; calibrate against a state-slot counter first if O-5 lands |
| O-2 | **`.dockerignore` excluded `.env` but not `.env.bench`/`.env.hf`/`.env.long`**, so `COPY . .` baked the owner's env files — and any keys in them — into local images | batch 2 preflight; the built image contained them | **fixed** `9cbf00f` (now `.env*`) | none; confirm on the next build that no `.env*` lands in the image |
| O-3 | **The spill guard took its baseline after the level had risen**, so it read a spilled boot as clean and missed ~16 MiB on three uncapped boots | batch 3 caveats; `../bench/results/2026-10-14-batch3-tools/spillwatch.sh` | **partly fixed** — batch 2's mid-write CSV false positive is fixed (sub-1 MiB ignored, two consecutive samples required); the baseline-order hole is not | batch 4 Q4: take the floor between the engine's KV-cache line and the first request |
| O-4 | **`bench/quality_battery.py` cites "gotcha 10 in the README"** for perplexity headroom, and gotcha 10 is about DeltaNet kernels. The spill is also an order of magnitude bigger than recorded (1,002 MiB vs 410) | batch 3 Q2; [gotchas.md](gotchas.md) 60 | **partly fixed** — gotcha 60 carries the real magnitude and the guard's blind spot; the docstring reference is still stale | point the docstring at gotcha 60 and gotcha 53 |
| O-5 | **No way to read the state pool.** No boot line reports mamba blocks or state slots, so the `RESIDENT` constant cannot be calibrated against anything | batch 3 Q4 | **open** | batch 4 Q1: check `/metrics` for a per-group or state gauge; if none, a log line is the patch |
| O-6 | **The launcher warns about seats but not about the profiling transient** that spills when `MAX_SEQS` is raised above what the pool supports | batch 3 Q4: 16 MiB spilled at `MAX_SEQS=16`, 32 MiB at 24 | **open** | extend the existing `RESIDENT` warning to cover the transient, or document the ceiling |
| O-7 | **The `upstream` remote's push URL is enabled on the GPU checkout** (only the sandbox had it disabled) — one `git push upstream` from touching a repo we do not own | batch 2 preflight | **open** | make it a repo-level convention, not a local setting |

## U — bugs and limitations in vLLM 0.31.0

| id | upstream | what it is | what we hold | status |
|---|---|---|---|---|
| U-1 | [#48905](https://github.com/vllm-project/vllm/issues/48905) | Marlin W4A8 reads group scales as **unsigned**; AutoRound exports negative ones, so every such group is garbage — and it still benchmarks fine | `patches/marlin-int8-negative-scales.patch`, found independently | issue **open** → upstream PR candidate, the fix is written |
| U-2 | [#51581](https://github.com/vllm-project/vllm/issues/51581) | DFlash fused-KV projection calls `F.linear` on a sliced `qkv_proj` weight → breaks or silently corrupts any weight-quantized drafter | `_dense_kv_rows()` inside `patches/dflash2-lookup-drafting.patch` | issue **open** → upstream PR candidate; club-3090 carries a duplicate fix |
| U-3 | [#48904](https://github.com/vllm-project/vllm/issues/48904) | `VLLM_MARLIN_INPUT_DTYPE=int8` silently ignored for auto-round (INC) checkpoints | nothing needed | **verified not ours**: our checkpoints load as `quantization=compressed-tensors` (batch-2 boot log), not INC, and v0.31.0's INC scheme file has no `input_dtype` reference at all |
| U-4 | [#48375](https://github.com/vllm-project/vllm/pull/48375) / [#43559](https://github.com/vllm-project/vllm/issues/43559) | `MambaManager.find_longest_cache_hit` receives `drop_eagle_block` and never reads it | nothing specific — `mamba-align-checkpoint-order.patch` is in the same function but is a different fix | **unverified**: on our pin the coordinator sets `manager.drop_eagle_checkpoint_block` under `use_eagle`, so it may be handled. Needs a probe: MTP + prefix caching, acceptance against `--no-enable-prefix-caching` |
| U-5 | — | **FlashInfer + spec decode downgrades cudagraphs to PIECEWISE** (`compilation.py:1462`, `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`) | — | measured, 26-29% slower on our workload; an upstream limitation to report rather than work around |

## X — unexplained behaviour worth chasing

| id | what | evidence | why it matters |
|---|---|---|---|
| X-1 | **`CG=40` changes greedy acceptance**: 3.34 → 3.48 tokens/step at *unchanged* step time (28.9 vs 28.7-29.1 ms), every run | batch 2, "seen that the sessions did not ask for" | a CUDA-graph **capture size** should not change sampled output. Either a numerics difference between capture modes or something subtler — and it is why E5 was never a speed result |
| X-2 | **`GPU KV cache size` falls with `max_model_len` at a fixed `KV_MEM`**: 27,714 tokens at 8,192 against 63,872 at 57,344, same 5,315,022,028 bytes | batch 2, same section | it quietly undermines any capacity claim made from a short-context row (S3a's, for instance) |

Both are isolatable with `bench/results/2026-10-14-batch3-tools/workload_accept.py` (teacher-forced, so the
trajectory is not a variable) plus a two-arm boot. Neither is in any batch yet.

## M — mistakes we made in the briefs (so the next one does not repeat them)

| id | what | cost | status |
|---|---|---|---|
| M-1 | Batch 1 E1's dynamic-K schedule `[[1,8,7],[9,32,3],[33,512,0]]` can never engage at `MAX_SEQS=8` — the pool holds ~5 requests, so a bucket starting at 9 is unreachable | one arm of batch 1 measured nothing | **fixed in docs**; the run re-cut the knee at 4 and that became a result |
| M-2 | Batch 3's probe as specified cannot discriminate: `MAX_SEQS=8` admits eight and k=7 already holds seven, so the literal rule reads "Conclusion B" when the truth is A | nearly cost the batch its conclusion; the agent added uncapped `MAX_SEQS=16` boots | **fixed in docs**; every seat question now lifts the cap and states the cap it ran under |
| M-3 | The batch-3 rule said the launcher's formula over-predicts; it errs **low** | a wrong direction in a spec that was about to be executed | **fixed in place** in the brief and the batch doc |
| M-4 | The batch-3 brief asserted there is no ReplaySSM flag on the pin | batch 1's probe missed it; `--use-replayssm` is in `--help=all` | **fixed in docs**; it still does not help (non-speculative only) |


## T — instruments this work produced (reuse, do not rebuild)

All under `bench/results/`, all committed, all run inside the container:

| tool | what it does | where |
|---|---|---|
| `running_probe.py` | fires a fixed concurrent load and samples the engine's own `Running:`/`Waiting:` line plus the KV-usage gauge — **the** seat metric, and what made batch 3 conclusive | `2026-10-14-batch3-tools/` |
| `workload_accept.py` | teacher-forced acceptance on arbitrary prompts (prefix-resync against a frozen continuation), so trajectory divergence is not a variable | `2026-10-14-batch3-tools/`, first used in `2026-10-13-tools/` |
| `spillwatch.sh` | watches the Windows Shared Usage counter against a loaded-engine floor (see O-3 for its blind spot) | `2026-10-14-batch3-tools/` |
| `memsum.sh`, `sampler.ps1`, `arm.sh` | floor summary, counter sampling, per-arm boot/measure wrapper | `2026-10-14-tools/`, `2026-10-14-batch2-tools/` |

## C — patches we already hold that belong upstream

Not found by this work — the standing backlog in [PATCHES.md](../PATCHES.md), where the `Upstream` column reads
"none yet". Listed here because two of them overlap **open** upstream issues (U-1, U-2), which makes them the
cheapest possible PR: the fix is written and somebody else has already filed the report.

`qwen3_5-embed-quant` · `marlin-int8-negative-scales` (**= U-1**) · `hybrid-kv-groups-v2-cudagraph` ·
`hybrid-sw-block-promote` · `offload-dflash-eagle-groups` · `dflash2-prewarm` · `spec-sampler-prewarm` ·
`mamba-chunked-prefill-align` · `dflash2-z-adaptive-emitted` · `dspark-draft-quant-config` ·
`topk-honour-flashinfer-sampler-switch` · plus `mamba-align-checkpoint-order`, tied to unmerged vllm#45238.

## Recommended order

1. **Open the two upstream PRs** (U-1, U-2) — the fixes exist in our tree, the issues are open, and nobody has to
   design anything. Highest value per hour by a wide margin.
2. **O-1**, the `RESIDENT` constant — one line, and it currently tells operators they have ~5 seats when they
   have 7. Better after O-5/Q1 tells us whether a counter exists to calibrate against.
3. **X-1**, the `CG=40` acceptance shift — if a capture size changes sampled output, that is a silent-correctness
   class of bug and it deserves a teacher-forced A/B before `CG` is used as a speed knob. Candidate for a fifth
   question in batch 4.
4. Then batch 4 as written: Q2's policy table is the deliverable, Q1 answers O-5, Q3/Q4 close the ReplaySSM and
   `MAX_SEQS=24` cells.

