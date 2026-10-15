# Batch 4: turn the seat curve into a policy, and make seats measurable

Batch 3 answered what caps residency: the **configured** verify block. Seats go 7 → 13 → 16 as `DFLASH_TOKENS`
goes 7 → 3 → 1, preemptions fall to zero from k=3 down, and the per-request share of the pool scales as (k+2)
predicts (14.2% / 7.45% / 4.37%). Single-stream speed pays for it: C1 T=0 is 112.1 / 94.1 / 64.7 tok/s.
[Summary](../bench/results/2026-10-14-batch3-SUMMARY.md) · [landscape G7](decode-speed-landscape.md).

That leaves two things, and this batch is them: **which k each profile should ship**, and **a way to read seats
without inferring them**. Neither needs a new mechanism — both are measurement.

**The lesson to carry in: `MAX_SEQS` hides the seat effect.** Batch 3's probe as specified read Conclusion B
because `MAX_SEQS=8` admits eight requests and k=7 already holds seven. Every seat question in this batch lifts
the cap, and every report states the cap it ran under.

## Q1 — can seats be read directly? (about 30 minutes)

No boot line reports the state pool (batch 3 Q4). Two things might change that, and both are cheap to check:

1. **Metrics, not logs.** Query `/metrics` during a held load and look for anything per-group or state-shaped —
   block counts, `kv_cache_max_concurrency`, mamba/state gauges, `num_gpu_blocks` by group. If a gauge exists,
   it becomes the primary metric for every future seat question and the launcher constant can be calibrated
   against it. Report the exact metric names and values, or state that none exist.
2. **Is the per-request cost a constant?** Batch 3 measured 81-88 MB per (k+2) unit at ~4k prompts. Repeat the
   seat probe at k=7 with **~32k-token prompts** and compare the share. If it is context-independent, the
   launcher's `104988089` is a constant that is simply 16-23% high and can be corrected in one line. If it grows
   with context, a constant is wrong in principle and the print should be removed or made a function of the
   prompt length.

**Conclusion for Q1, either way:** "a gauge exists (names, values)" or "none exists, the counter is a patch";
plus "the share is constant across depth" or "it is not, here is the curve".

## Q2 — the policy table: which k for which concurrency (about 60 minutes)

The question the shipped profiles cannot answer today. `single` runs k=7 at any load; `batch` sets speculation
aside entirely. In between — 2 to 16 concurrent users, which is most of the traffic an agent front-end actually
sees — there is nothing, and batch 3 says the middle is exactly where the seat curve pays.

Four arms, one boot each, **identical `KV_MEM` and `MAX_SEQS=16`** so the cap cannot bind and the pool is not a
variable:

| arm | configuration |
|---|---|
| Q2-a | `SPEC=dflash2 DFLASH_TOKENS=7` (shipped single-user) |
| Q2-b | `SPEC=dflash2 DFLASH_TOKENS=3` |
| Q2-c | `SPEC=dflash2 DFLASH_TOKENS=1` |
| Q2-d | `SPEC=off` (no speculation — what batch mode does) |

Per arm: `bench/conc_ladder.py --n 1,2,4,8,16 --json`, the held 8-request `Running:` probe, C1 T=0, preemptions,
`kv%`, and `ms/pass`. Then the table: **aggregate decode tok/s and preemptions per arm per C**, with C1 T=0 as
the cost column.

- **Decide:** for each C in {1, 2, 4, 8, 16}, which arm wins on aggregate, and where the crossover sits between
  the speculating arms and `SPEC=off`. Batch 3's tail rows suggest it is somewhere around C=8-16 (k=3 read 350.1
  against k=1's 299.4 at N=8, and every k=7 row preempted).
- **Caveat to state:** if a row preempts, it is a tail row and does not settle a crossover on its own. Say which
  rows are steady and which are not.
- This table is the deliverable. It is what turns "speculation wins below ~8 concurrent users" from a rule of
  thumb into a documented profile choice.

## Q3 — ReplaySSM, in the one place its docstring allows it (about 20 minutes)

`--use-replayssm` exists on this pin (batch 3 Q4) and is restricted to Mamba2 **without** speculative decoding —
which is batch mode, where the 1,035 tok/s aggregate lives. Its mechanism (skip the per-step full-state store,
write the checkpoint back on flush) is aimed at the state traffic that costs us seats and steps elsewhere.

- A/B in a non-speculative profile, one boot each: `--use-replayssm` off vs on. Read `ms/pass` at C8 and C16,
  aggregate decode, seats (`Running:`), and the boot log for whether it engaged.
- Pass = at least 3% on aggregate at C8 or C16 with no quality change. Fail = no change or it did not engage.
  Either way it closes the last of the upstream state knobs.

## Q4 — finish the seats-above-16 question, and fix the guard's blind spot (about 20 minutes)

- k=1 at `MAX_SEQS=24` was the one point batch 3 could not measure: that boot spilled 32 MiB. Re-run it with
  `KV_MEM` pre-cut (batch 2's trick) so the margin holds, and report seats, preemptions and the trace.
- **The guard needs its floor taken before the rise.** Batch 3's uncapped boots spilled ~16 MiB at memory
  profiling and the guard missed it because it sampled its baseline *after* the level had already risen. Take the
  floor from the samples between the engine's KV-cache line and the first request, and require two consecutive
  samples above it before calling a spill. Note in the report whether the guard was fixed or worked around.

## Budget and order

Q1 ≈ 30 min, Q2 ≈ 60 min, Q3 ≈ 20 min, Q4 ≈ 20 min. Q2 is the deliverable; if time runs short, drop Q3 before
Q4 (Q4 closes an open cell and fixes an instrument). No new image is needed — nothing here changes engine code.

## What not to do

No engine patches and no default flips (Q2 produces the table that decides a default; it does not set one). No
new image. No EXL3/TurboQuant/SGLang/FlashInfer work. No bf16-state re-run — batch 3 closed it. No checkpoint-flag
re-runs — batch 3 closed those too. No soak, no `verify.sh`, no quality claim from n=50.

