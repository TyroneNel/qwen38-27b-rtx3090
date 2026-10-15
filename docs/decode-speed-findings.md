# Decode-speed programme: findings, open work, and how to pick it up

**What this is.** The running record of a measured effort to make this stack serve Qwen3.8-27B faster on one RTX
3090 at 250 W. Four batches have run on a real card. Every number here is either a measurement whose raw logs are
committed under [`bench/results/`](../bench/results), or an item explicitly marked as unmeasured. Speed comes
first and precision second: a lever that trades a measured amount of quality for throughput belongs here, and one
that trades nothing is worth more.

**Where it stands, in three lines.**

- **The speed levers that need no code are measured and ready**: a k=7 / k=3 / `SPEC=off` band table by
  concurrency (S-1) on an int8-KV `CTX=fast` base (S-2), worth +41% aggregate at N=4 and +91% at N=8. Published as
  "How many concurrent users" in [single-user/README.md](../single-user/README.md).
- **The levers that remain need engine work**, and the list is short: S-5 (journal-and-replay state under
  speculation) and S-6 (drafter acceptance on chat) are the only two with real headroom. What looked promising and
  is not is in **R — refuted** below.
- **Acceptance is a property of the workload, not our drafter**: 5.80 tokens/step on GSM8K, 4.78 code, 3.50 chat.
  The deficiency this programme started from was our benchmark protocol, not the stack.

## Start here, if you are a new agent

**What is asked of you.** One item from *Open work* below, run as a batch: arms, a metric, and a pass/fail rule
decided *before* the run. Batches 2-5 are the template — [the plan](decode-speed-batch5.md) and
[the paste-ready brief](gpu-agent-brief-batch5.md).

**Read in this order.** This page's **S** and **R** sections (what is known, and what not to re-chase), then the
batch plan you are running, then the brief, then `bench/results/` for the arm you are repeating.

**The rules that make a result usable.** Not style preferences — each has cost a batch something.

1. **`MAX_SEQS` hides the seat effect.** At 8 the launcher admits eight requests and k=7 already holds seven, so a
   seat question measured at `MAX_SEQS=8` reads as "no difference" when the difference is real. Run seat and
   concurrency work at `MAX_SEQS=16` and state the cap in the report.
2. **Discard the first measurement after every boot** (JIT warmup reads 30-50% low), and **one boot per arm**
   unless the effect is inside 10% — then interleave A-B-A. Boot-to-boot spread is 0-3.2% for C1 T=0 and ~10% for
   free-run decode.
3. **Prefer the noise-free metric.** Teacher-forced tokens/step on a frozen target (`workload_accept.py`)
   reproduces to three decimals between passes; free-run decode does not. C1 T=0 is 0.0-0.1% *within* a boot.
4. **A spilled run keeps its tokens/step and loses its tok/s.** Watch the Windows Shared Usage counter against the
   loaded-engine floor (84-92 MiB) — and take that floor *between* the engine's KV-cache line and the first
   request, or a boot rise will be read as clean.
5. **Preempted ladder rows are tail rows, not steady state.** `conc_ladder.py` says so itself ("no instant at
   which all N streams were decoding"); quote it as a queue measurement or not at all.
6. **A knob that changes no log line did not apply.** Quote the engine's args line and its kernel/block lines
   before believing any arm.
7. **No default flips from an experiment.** A batch produces evidence and a one-line recommendation; a default
   change is a reviewed edit afterwards.
8. **Results-only commits, to `cline/gf59cdmh`.** No PRs, issues, comments, watermarks or co-author trailers; one
   author, the repo's configured identity. Push after each batch. Raw logs beside every number.

**The map.** Which file is what, so you do not have to infer it.

| file | what it is |
|---|---|
| this page | the worklist: speed levers, refuted ideas, defects, open items |
| [decode-speed-landscape.md](decode-speed-landscape.md) | the survey that started it: what other engines get, gap by gap, with each gap's measured outcome |
| [batch2](decode-speed-batch2.md) … [batch5](decode-speed-batch5.md) | the plans: arms, metrics, pass rules |
| [gpu-agent-brief\*.md](gpu-agent-brief-batch5.md) | the paste-ready briefs, one per batch |
| [`bench/results/*-SUMMARY.md`](../bench/results) | what each batch actually measured, raw logs beside it |
| [PATCHES.md](../PATCHES.md) | the patch ledger — what we ship, unrelated to the items here |

## Open work, ranked

1. **Ship the bands (S-1 + S-2).** Measured end to end; the recipe is in
   [single-user/README.md](../single-user/README.md). What remains is whether the 2-8-stream band becomes a
   *default* somewhere — a product call.
2. **O-1**, the launcher seat print: implemented on this branch as the measured function. The remaining work is
   the state-slot counter that would make it exact rather than fitted (O-5).
3. **X-3**, the 16-seat C1 loss: a `MAX_SEQS=16` boot costs k=7 7.8% of single-stream speed, and one
   `MAX_SEQS`-sized buffer doubles in the logs. If it can be sized independently, band 2 gets cheaper.
4. **Batch 5**, for the cheap unknowns it still holds: the unadvertised mamba knobs (four boots), the `CG=40`
   mechanism (X-1), and the IFBench gate band 2 should have before it is recommended rather than measured.
5. **S-5 and S-6** — the two levers with no ceiling, both engine projects. S-5 is now specified precisely; the
   per-step-k framing is dead and journal-and-replay is the route (see the note under the S table).

## S — speed levers, ranked

Measured means a number from the reference 3090 at 250 W; estimated means the gain is inferred from a mechanism or
from someone else's measurement. "Precision cost" is the second axis, not a blocker.

| id | lever | gain | kind | effort | precision cost |
|---|---|---|---|---|---|
| S-1 | **Ship k as a concurrency control**: k=7 at 1 stream, **k=3 at 2-8, `SPEC=off` above ~9** — k=1 wins no concurrency | **measured** (batch 4's policy table): N=4 263.6 against 239.1 (k=7) and 143.2 (off); N=8 **337.1** against 176.8 (k=7, tail) and 250.7 (off); N=16 only `SPEC=off` seats all 16 (369.2). C1 pays 103.4 → 93.5 → 40.7 | config | low | none |
| S-2 | **`TRITON_ATTN` + `int8_per_token_head` as the `CTX=fast` default** | **measured**: pool 68,605 → 116,440 tokens (+69.7%), N=8 preemptions 23 → 1, C1 a tie (111.2 vs 111.0) | config | low | none measured (GSM8K n=200 within one question, perplexity within 0.07%) |
| S-3 | **`CG=40`** | **measured**: +3.3% C1, +4.1% with `CUDAGRAPH_MODE=FULL` — and the mechanism is *acceptance*, 3.34 → 3.48 tokens/step at unchanged step time | config | low | unknown; the mechanism is X-1 |
| S-4 | **`INT8_ACT=int8` for prefill** | **measured**: prefill 1,437 → 1,845 tok/s at 1k and 1,200 → 1,423 at 51k (+27-30%, +19% at depth), decode unchanged | config | low | +0.9-3.7% perplexity, documented in [optimizations.md](optimizations.md) |
| S-5 | **Journal-and-replay the state instead of holding k+1 pages per request** (ReplaySSM's shape, made spec-decode-safe) | **estimated**: removes the seats-versus-C1 trade entirely — k=7 speed at 13+ seats. **Reframed: "make the reservation follow the per-step k" is impossible** — see the note under this table | engine work | high | none |
| S-6 | **DFlash2 selector headroom** | **corrected, and much smaller than first written** — the published oracle 6.79-vs-4.27 gap is a 5-layer Qwen3-4B drafter on GSM8K, not ours; our own teacher-forced acceptance on GSM8K is **5.80**, already above the published DFlash2 5.46. The headroom that exists is on *chat* (3.50) | engine work | high | none (lossless by construction) |
| S-7 | **GQA-packed split-KV verify attention** (24 → 64 CTAs) | **estimated**: 13.9-15.6× per-layer at 38k/100k on A100; our kernel is the current one | kernel work | med | none |
| S-8 | **Workload-aware k** — per-endpoint rather than one global value | **measured** that the spread is real: gsm8k 5.80, code 4.78, chat 3.50 tokens/step; k=7 is a compromise for all three | config | med | none |
| S-9 | **FlashInfer cudagraph support for the spec-decode verify shape** | **estimated**: recovers the measured 26-29% PIECEWISE penalty on that path (it may only tie `TRITON_ATTN`) | upstream | med | none |
| S-10 | **int8-KV prefill attention** (the e4m3→half scale-fold route) | **estimated**: prefill attention 41 → 72 TFLOPS on the b70 port; our 51k prefill row is where the 16 full-attention layers dominate | kernel work | med | none |
| S-11 | **The unadvertised mamba knobs** (`--mamba-ssu-algorithm`, `--mamba-backend`) | unknown — never tested | config | low | unknown |
| S-12 | **`CG=40` default, once X-1 is understood** | see S-3 | config | low | unknown |

S-1 to S-4 are one item type: knobs we already have, measured, not yet shipped. S-5 is the one with no ceiling —
the only lever that would let one server be fast for one user *and* hold thirteen.

**The formula's farthest validated point** (batch 4 Q4): k=1 at `MAX_SEQS=24` seats **21** —
`Running: 21 reqs, Waiting: 3 reqs` — which is 18.0 blocks per request against the formula's 21.8, so it holds at
24 as well. That boot's 90.2 → 122.2 MiB Shared Usage step is batch 3's "32 MiB spill" reproduced with 256 MiB
more headroom, which confirms it is the `MAX_SEQS`-scaled boot step (8 / 16 / 32 MiB at 8 / 16 / 24) and not
memory pressure: batch 3's "could not measure" was that step, and the pre-cut neither removed nor needed to
remove it.

### S-1 and S-2, in shippable form

Both are configuration, so "shipping" them means publishing the recipes rather than changing the launchers. This
table is **batch 4's policy measurement**, not a proposal: every seat count, aggregate and C1 figure was measured
on the reference 3090 at 250 W with `KV_MEM` pinned and `MAX_SEQS=16` on every arm.

| band | profile | C1 T=0 | seats at ~4k prompts | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|---|
| **1 stream** | as shipped (k=7) | **103.4** | 5 | 239.1 | 176.8 (tail) | 156.1 (tail) |
| **2-8 streams** | `DFLASH_TOKENS=3` + `EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head'` | 93.5 | 9 | **263.6** | **337.1** | 300.4 (tail) |
| **9+ streams** | `SPEC=off` (batch mode's shape: no speculation) | 40.7 | 16 | 143.2 | 250.7 | **369.2** |
| *(k=1)* | `DFLASH_TOKENS=1` | 63.3 | 13-14 | 202.3 | 294.9 | 180.8 (tail) |

Three things the table decides that a single number cannot:

- **The band boundary is where an arm's own aggregate stops rising**, which is the first N above its seat count:
  4 → 8 for k=7 (5 seats), 8 → 16 for k=3 (9 seats). Above N=8 no speculating row is steady state.
- **k=1 wins nothing at any concurrency.** It is a strictly worse k=3: fewer tokens per step, more seats than any
  load needs. It was in this table as "7-16 users" after batch 3 — that was wrong, and the measurement is what
  corrected it.
- **Above ~9 concurrent, the answer is to stop speculating** (N=16: `SPEC=off` 369.2 against k=3's 300.4 tail).
  Shorter prompts move that crossover up, because seats rise as prompts shorten.

```dotenv
# 2-8 concurrent streams: the seat/speed point batch 4 measured
SPEC=dflash2
PREFIX_CACHE=1
CTX=fast
DFLASH_TOKENS=3
KV_MEM=5583457484          # pin the pool in bytes; GPU_UTIL moves with the KV dtype
MAX_LEN=65536
MAX_SEQS=16                # the admission cap: 8 hides the seat effect entirely
EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head'
```

### S-5 is not what it looked like (source-level answer, 2026-10-14)

Batch 5 Q2 asked for a feasibility verdict from the pinned source. Reading vLLM 0.31.0 answers it, and it refutes
the framing that was on this page:

- The per-request state cost is **static and sized from the configured k**:
  `MambaSpec.max_memory_usage_bytes` returns `page_size_bytes * (2 + num_speculative_blocks +
  num_prefill_checkpoint_blocks)` in `align` mode, and `page_size_bytes * (1 + num_speculative_blocks)` in `none`
  (`vllm/v1/kv_cache_interface.py:1048`, `:1083-1094`).
- The block-table row is likewise a fixed `1 + num_speculative_blocks` positions
  (`vllm/v1/attention/backends/utils.py:1176-1195`): the kernel gathers the last `1 + num_speculative_blocks`
  entries of a position-indexed table.
- `num_speculative_blocks` is a field of the spec, set once when the groups are built. **Nothing in that path
  knows about a per-step schedule** — which is the source-level explanation of batch 3's measurement that a
  per-step k frees no seats. Two independent confirmations, one empirical and one structural.

So a schedule cannot lower the reservation, and it should not: the reservation is a **peak** requirement, because
every step that verifies k+1 positions needs k+1 state pages live to write them into. Covering the largest k any
step will use is the only correct thing to do — which is exactly what the code does.

The lever that remains is therefore not "follow the per-step k" but **stop keeping k+1 full state pages resident
per request**: journal the recurrent-state *inputs* for the in-flight block and replay them, writing one
checkpoint back only on a flush. That is ReplaySSM's design (`use_replayssm`, `replayssm_buffer_len`), and its
docstring limits it to non-speculative decode. Batch 4 Q3 closed the shortcut: the flag is **refused outright
for this architecture** (`--use-replayssm is not supported for architecture 'Qwen3_5ForConditionalGeneration'`),
so ReplaySSM cannot be enabled for this checkpoint at all — not even in batch mode. The mechanism therefore has
to be **built**, not switched on: journal the state inputs for the in-flight block, replay them, write one
checkpoint on flush, and make that correct when the k+1 positions are known-but-unverified draft tokens. That is
a real project with a real payoff (k=7 speed at 13+ seats), and it is the only route to that payoff.

**Verdict: attemptable, but it is a redesign, not a wiring change** — and its shape is ReplaySSM-under-speculation,
not a per-step reservation.

## R — refuted, do not re-chase

| what | why not | evidence |
|---|---|---|
| `--max-num-batched-tokens` above 2048 | costs pool (2.4% at 4096) and 8192 refuses to boot; the win belongs to other engines | our [gotchas.md](gotchas.md) 7 and 43 |
| bf16 recurrent state | brings the fused GDN kernel back and buys nothing: +0.1% C1, 1.9% slower `ms/pass` at N=1 | batch 3 Q2 |
| `--enable-mamba-shared-prefix-checkpoint`, `--mamba-block-size 8192` | neither changed the block size, the page padding, the pool or the seat count | batch 3 Q3 |
| a per-step k schedule as a **seat** lever | state is sized for the configured maximum k, so varying k per step frees nothing | batch 3 |
| `--use-replayssm` | the flag reaches the engine (`'use_replayssm': True` in the args line) and validation then refuses it: `Value error, --use-replayssm is not supported for architecture 'Qwen3_5ForConditionalGeneration'`. The engine never boots — **the last upstream state knob is closed for this checkpoint** | batch 4 Q3 |
| FlashInfer as the fast path on this workload | 26-29% slower than the shipped path, entirely from the cudagraph downgrade | batch 2 E2 |
| fp8 KV for decode speed | a context buy, not a speed one — the A/Bs cost decode | our [reproductions](reproductions/README.md), club-3090 |
| a 4 GiB bf16 DFlash2 drafter | int4 keeps greedy acceptance (3.34-3.65 against 3.54) and costs ~5% at default sampling | our [drafter/README.md](../drafter/README.md) |

## O — bugs in our own code

| id | what | evidence | status | next action |
|---|---|---|---|---|
| O-1 | **The `RESIDENT` constant is wrong in principle, not just in value.** `single-user/start_qwen.sh` budgets `104988089` bytes per (k+2) unit. The measured cost is 87.6 / 82.8 / 81.1 MB per (k+2) — 20-29% lower — **and it grows with context**: k=7 costs 54 / 64 / 91 / 129 blocks per request at ~1k / 4k / 16k / 32k prompts, 2.4× from end to end. The state term is exactly `6(k+2)` blocks; the rest is `2·⌈ctx/block⌉` | batch 3 seats, then batch 4 Q1's block accounting | **fixed on this branch** — the print is now the measured function (`6(k+2) + 2*ceil(ctx/864)` pages of a 14,601,709 B pool page), printing both ends (a ~1k prompt and `MAX_LEN`), and `CTX=huge` is skipped because its block and page are unmeasured. Covered by `bench/test_resident_print.sh` (17 CPU checks, no GPU) | the exact version still needs a state-slot counter (O-5); the current fit is calibrated at k=7 and reads ~10-20% high at k=3/1 because `num_gpu_blocks` is 364/385/393 across those arms rather than `KV_MEM/page` |
| O-2 | **`.dockerignore` excluded `.env` but not `.env.bench`/`.env.hf`/`.env.long`**, so `COPY . .` baked the owner's env files — and any keys in them — into local images | batch 2 preflight; the built image contained them | **fixed** `9cbf00f` (now `.env*`) | none; confirm on the next build that no `.env*` lands in the image |
| O-3 | **The guard's baseline window cannot see the boot rise, and the rise is not a spill.** Memory profiling runs *before* the "GPU KV cache size" line, so the KV-line-to-first-request window starts above the rise. The step follows `MAX_SEQS` (8.0 MiB at 8, 16.0 at 16, 32.0 at 24), not memory pressure — a `KV_MEM` 1 GiB lower and `SPEC=off` both step identically | batch 3 caveats, then batch 4's `guard-test.txt` replay | **fixed by batch 4**: the guard keeps the brief's floor for measurement spills and adds the pre-profiling level to report the boot step | none — but the boot step should be reported, not treated as a spill |
| O-4 | **`bench/quality_battery.py` cites "gotcha 10 in the README"** for perplexity headroom, and gotcha 10 is about DeltaNet kernels. The spill is also an order of magnitude bigger than recorded (1,002 MiB vs 410) | batch 3 Q2; [gotchas.md](gotchas.md) 60 | **partly fixed** — gotcha 60 carries the real magnitude and the guard's blind spot; the docstring reference is still stale | point the docstring at gotcha 60 and gotcha 53 |
| O-5 | **No state gauge on `/metrics`** — but blocks-in-use is obtainable anyway | batch 4 Q1: no state or per-group gauge exists, while `vllm:kv_cache_usage_perc` × (`num_gpu_blocks` − 1) is an exact integer on all 11 dumps, with `num_gpu_blocks` from `vllm:cache_config_info` | **answered** (workaround, not a gauge) | use the product for seat accounting; a first-class state gauge is still the cleaner patch |
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
| X-3 | **A 16-seat boot costs k=7 single-stream speed:** C1 T=0 is 103.4 on a `MAX_SEQS=16` boot against batch 3's 112.1 on an 8-seat boot, a 7.8% loss, with the N=1 step at 32.7 against 29.5 ms/pass. k=3 (−0.6%) and k=1 (−2.2%) barely move. The one `MAX_SEQS`-sized buffer in the boot log doubles (`int4 3D scratch: capacity=256 query tokens` against 128) | batch 4, "where this contradicts batch 3" | the band table above pays it: band 2's C1 (93.5) carries this loss as well as the k=3 loss, so the real k=3 cost may be nearer 4% than 16% once the boot is matched |

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
| `test_resident_print.sh` | CPU test for the launcher seat print (17 checks, no GPU): the formula against the measurement it came from, both ends, the silent cases, and that `test_bool_knobs.sh` still passes | `bench/` |

## C — patches we hold that belong upstream

Not tracked here. They are rows in [PATCHES.md](../PATCHES.md) — 42 with no upstream reference (17 `fix`, 14
`feature`, 6 `local`, 5 `own`) — and managing them is routine. Two `fix` rows happen to overlap open upstream
issues (U-1, U-2 below), which is the only reason they are named on this page at all.

## Measured next: [batch 5](decode-speed-batch5.md)

batch 5 covers the levers on this list that cannot be settled by reading — S-11 (four boots), the S-5
reconnaissance, X-1 if batch 4 skipped it, and the IFBench gate for the band-2 profile.
