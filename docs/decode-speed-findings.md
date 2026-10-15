# Findings: bugs, fixes and patches from the decode-speed work

The inventory the batch records imply but never collected: what was found, what holds it, what has been done about
it, and what is still open. Sourced from batches 1-3 on the reference 3090 plus the landscape survey.

**What this file is for.** Speed first, precision second: an item that trades a measured amount of quality for
throughput belongs here and is worth considering, and one that trades nothing is worth more. The inventory is
therefore **S — speed levers** (the worklist), **R — refuted** (so nobody re-chases them), then the defects,
upstream limitations and open questions found on the way. **Patch management is not tracked here** — the changes we
carry with no upstream reference are rows in [PATCHES.md](../PATCHES.md), neither defects nor speed work, and
re-cutting them at a pin bump is routine.

**How to use it.** Every item has an id, the evidence that established it, a status, and the next action. Change a
status only with evidence — a commit, a result path, or a quoted command output — and put the pointer in the item.
"Open" means nobody has started; "verified" means we checked and it does not affect us. Related pages:
[landscape](decode-speed-landscape.md) · [batch 2](decode-speed-batch2.md) · [batch 3](decode-speed-batch3.md) ·
[batch 4](decode-speed-batch4.md) · [PATCHES.md](../PATCHES.md) (the patch ledger — what we ship, not what we
found).

## S — speed levers, ranked

Measured means a number from the reference 3090 at 250 W; estimated means the gain is inferred from a mechanism or
from someone else's measurement. "Precision cost" is the second axis, not a blocker.

| id | lever | gain | kind | effort | precision cost |
|---|---|---|---|---|---|
| S-1 | **Ship k as a concurrency control**: a k=3 (and k=1) profile instead of k=7 for every load | **measured**: N=4 aggregate +9.2%; N=8 350.1 against 178.5 (k=7's tail row); preemptions 2 → 0; C1 pays 112.1 → 94.1 tok/s | config | low | none |
| S-2 | **`TRITON_ATTN` + `int8_per_token_head` as the `CTX=fast` default** | **measured**: pool 68,605 → 116,440 tokens (+69.7%), N=8 preemptions 23 → 1, C1 a tie (111.2 vs 111.0) | config | low | none measured (GSM8K n=200 within one question, perplexity within 0.07%) |
| S-3 | **`CG=40`** | **measured**: +3.3% C1, +4.1% with `CUDAGRAPH_MODE=FULL` — and the mechanism is *acceptance*, 3.34 → 3.48 tokens/step at unchanged step time | config | low | unknown; the mechanism is X-1 |
| S-4 | **`INT8_ACT=int8` for prefill** | **measured**: prefill 1,437 → 1,845 tok/s at 1k and 1,200 → 1,423 at 51k (+27-30%, +19% at depth), decode unchanged | config | low | +0.9-3.7% perplexity, documented in [optimizations.md](optimizations.md) |
| S-5 | **Make the state reservation follow the per-step k** (or a ReplaySSM-style rollback that works under speculation) | **estimated**: removes the seats-versus-C1 trade entirely — k=7 speed at 13+ seats | engine work | high | none |
| S-6 | **DFlash2 selector headroom** | **estimated**: the published oracle acceptance is 6.79 against the achieved 4.27 — up to +59% tokens per verify step | engine work | high | none (lossless by construction) |
| S-7 | **GQA-packed split-KV verify attention** (24 → 64 CTAs) | **estimated**: 13.9-15.6× per-layer at 38k/100k on A100; our kernel is the current one | kernel work | med | none |
| S-8 | **Workload-aware k** — per-endpoint rather than one global value | **measured** that the spread is real: gsm8k 5.80, code 4.78, chat 3.50 tokens/step; k=7 is a compromise for all three | config | med | none |
| S-9 | **FlashInfer cudagraph support for the spec-decode verify shape** | **estimated**: recovers the measured 26-29% PIECEWISE penalty on that path (it may only tie `TRITON_ATTN`) | upstream | med | none |
| S-10 | **int8-KV prefill attention** (the e4m3→half scale-fold route) | **estimated**: prefill attention 41 → 72 TFLOPS on the b70 port; our 51k prefill row is where the 16 full-attention layers dominate | kernel work | med | none |
| S-11 | **The unadvertised mamba knobs** (`--mamba-ssu-algorithm`, `--mamba-backend`) | unknown — never tested | config | low | unknown |

S-1 to S-4 are one item type: knobs we already have, measured, not yet shipped. S-5 is the one with no ceiling —
the only lever that would let one server be fast for one user *and* hold thirteen.

## R — refuted, do not re-chase

| what | why not | evidence |
|---|---|---|
| `--max-num-batched-tokens` above 2048 | costs pool (2.4% at 4096) and 8192 refuses to boot; the win belongs to other engines | our [gotchas.md](gotchas.md) 7 and 43 |
| bf16 recurrent state | brings the fused GDN kernel back and buys nothing: +0.1% C1, 1.9% slower `ms/pass` at N=1 | batch 3 Q2 |
| `--enable-mamba-shared-prefix-checkpoint`, `--mamba-block-size 8192` | neither changed the block size, the page padding, the pool or the seat count | batch 3 Q3 |
| a per-step k schedule as a **seat** lever | state is sized for the configured maximum k, so varying k per step frees nothing | batch 3 |
| FlashInfer as the fast path on this workload | 26-29% slower than the shipped path, entirely from the cudagraph downgrade | batch 2 E2 |
| fp8 KV for decode speed | a context buy, not a speed one — the A/Bs cost decode | our [reproductions](reproductions/README.md), club-3090 |
| a 4 GiB bf16 DFlash2 drafter | int4 keeps greedy acceptance (3.34-3.65 against 3.54) and costs ~5% at default sampling | our [drafter/README.md](../drafter/README.md) |

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

## C — patches we hold that belong upstream

Not tracked here. They are rows in [PATCHES.md](../PATCHES.md) — 42 with no upstream reference (17 `fix`, 14
`feature`, 6 `local`, 5 `own`) — and managing them is routine. Two `fix` rows happen to overlap open upstream
issues (U-1, U-2 below), which is the only reason they are named on this page at all.

## Recommended order

1. **S-1 and S-2 together, as one change.** Both are measured, both are config, neither costs precision: a k=3
   profile plus the int8-KV/TRITON `CTX=fast` default is the difference between a server that preempts at eight
   concurrent users and one that holds thirteen without preempting. This is the cheapest real speed on the board.
2. **S-3, after X-1.** `CG=40` is +3-4% C1 and we do not know why. Confirm the mechanism on a teacher-forced
   target first — a capture size that changes sampled output is worth understanding before it becomes a default.
3. **S-4** for any prefill-heavy workload; it is already implemented and documented, just not on by default.
4. **S-5 or S-6** if the effort is available — the two levers with no ceiling. S-5 removes the seats-versus-C1
   trade (the thing S-1 can only choose a point on); S-6 raises acceptance rather than step time, and acceptance is
   where our single-stream numbers live.
5. **S-7 to S-11** as fill-in work, cheapest first (S-11 is one boot).
