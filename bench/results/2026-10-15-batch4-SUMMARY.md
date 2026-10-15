Recommendation: serve one stream at k=7, 2-8 concurrent streams at k=3, and more than about 9 concurrent 4k-token streams with `SPEC=off`; k=1 wins no concurrency.

## The policy table (Q2, `2026-10-15-Q2-{k7,k3,k1,off}/report.md`)

**Setup:**
- **Arms:** one boot each, int8 `CTX=fast`, `KV_MEM=5315022028` and **`MAX_SEQS=16` on every arm**.
- **Metric:** `bench/conc_ladder.py --n 1,2,4,8,16` decode aggregate (tok/s, 4k-token prompts, 256 out).
- **Row marks:** **TAIL** = the server ran fewer streams than were sent (`run` in brackets), so the row is a queue, not
  steady state. Preemptions were 0 in every ladder row.

| N | k=7 | k=3 | k=1 | `SPEC=off` | winner |
|---|---|---|---|---|---|
| 1 | **105.4** | 100.9 | 60.9 | 40.1 | k=7 |
| 2 | 167.4 | **170.0** | 115.7 | 77.5 | k=3 by 1.6%: a tie with k=7 within the 3.2% cross-boot floor |
| 4 | 239.1 | **263.6** | 202.3 | 143.2 | k=3 |
| 8 | 176.8 TAIL (5) | **337.1** | 294.9 | 250.7 | k=3 |
| 16 | 156.1 TAIL (5) | 300.4 TAIL (9) | 180.8 TAIL (13) | **369.2** (16) | `SPEC=off` |
| **cost: C1 T=0** | 103.4 | 93.5 | 63.3 | 40.7 | |
| seats at ~4k prompts (`Running:`) | 5 | 9 | 13-14 | 16 | |
| seats at ~1k prompts | 6 (6.7 by blocks) | ≥8 (12.8 by blocks) | ≥8 (21.8 by blocks) | ≥8 (39.6 by blocks) | |

**Where the crossover sits:**
- **The rule:** each speculating arm's aggregate stops rising at the first N above its seat count. That is 4 → 8 for
  k=7 (5 seats), 8 → 16 for k=3 (9 seats) and 8 → 16 for k=1 (13-14 seats).
- **Against `SPEC=off`:** the crossover sits between N=8 and N=16, where N passes k=3's 9 seats.
- **Precisely:** above N=8 no speculating row is steady. At N=8, k=3 leads `SPEC=off` by 34.5% (337.1 vs 250.7). At
  N=16, `SPEC=off` is the only arm that seats all 16.
- **Shorter prompts:** they move the crossover up, because seats rise as prompts shorten (Q1).
- **This table is the recommendation.** No default is set.

## Q1: seats can be read directly, in blocks; the cost is not a constant (`2026-10-15-Q1-seat-gauge/report.md`)

- **No state or per-group gauge exists on `/metrics`.** But `vllm:kv_cache_usage_perc` × (`num_gpu_blocks` − 1) is an
  exact integer on all 11 dumps, with `num_gpu_blocks` taken from `vllm:cache_config_info`. That is blocks in use,
  directly; divided by Running it is blocks per request.
- **The cost is affine in context:** k=7 costs **54 / 64 / 91 / 129 blocks** per request at ~1k / 4k / 16k / 32k
  prompts. That fits 52.9 + 2.01 blocks per 864 prompt tokens, and is 2.4× from 1k to 32k.
- **The state term is exactly 6(k+2):** 54 / 30 / 18 blocks at k=7/3/1, which is 87.6 / 82.8 / 81.1 MB per (k+2).
- **The launcher's `104988089` is wrong in principle.** It is 20-29% above the short-prompt cost and has no context term.
  The evidence supports a function, seats ≈ (`num_gpu_blocks` − 1) ÷ (6(k+2) + 2·⌈ctx / block⌉), not a corrected
  constant.

## Q3 and Q4

Pending at the time of the Q1+Q2 push; see the sections below once they are added.

## Where this contradicts batch 3, the docs or the launcher, with numbers

- **k=7 C1 T=0 on a 16-seat boot is 103.4, 7.8% below batch 3's 8-seat boot (112.1).** The N=1 step is 32.7 against
  29.5 ms/pass. k=3 (−0.6%) and k=1 (−2.2%) barely move. The boot log's one `MAX_SEQS`-sized buffer doubles (`int4 3D
  scratch: capacity=256 query tokens` against 128). The cause is not isolated.
- **Batch 3 called the `MAX_SEQS=16` boot rise a spill; it is not one.**
  - **The test:** the k=7 boot with `KV_MEM` a further 1 GiB lower (`q2k7cut1g`, 2,465 MiB free against 1,437) steps the
    same 86.2 → 102.2 MiB. So does `SPEC=off`, with 3,381 MiB free.
  - **What the step follows:** `MAX_SEQS` (8.0 MiB at 8, 16.0 at 16, 32.0 at 24 in batch 3), not memory pressure.
- **The brief's guard window cannot see that rise anyway.** Memory profiling runs before the "GPU KV cache size" line, so
  the KV-line → first-request window starts above it.
  - **Replay:** on batch 3's traces, that window gives 102.2 on the uncapped boots and 92.2 on the capped ones
    (`2026-10-15-batch4-tools/guard-test.txt`).
  - **The fix:** the guard keeps the brief's floor (F_kv) for measurement spills and adds the pre-profiling level
    (F_ctx) to report the boot step.
- **"k=7 holds 7"** (batch 3, landscape G7) holds at the full 5,583 MB pin only. On this 256 MiB smaller pool it holds 6
  at ~1k and 5 at ~4k. The per-request bytes are the same (788 vs 793 MB).
- **The launcher comment, "six resident, the seventh preempted":** on ~1k prompts and this pool it is 6 resident and the
  rest waiting; at ~4k it is 5.

## Method notes

- **Image:** `hyperqwen:exp-f1a033e`, reused. The Step 1 diff printed `README.md` only (+2 lines of prose), and the
  brief forbids a new image (`2026-10-15-batch4-preflight/image-reuse.txt`).
- **`KV_MEM` pre-cut:** 256 MiB on every arm, decided before any batch-4 boot from batch 3's uncapped-boot rise.
- **Guard:** `2026-10-15-batch4-tools/spillwatch.sh`.
  - **What it fixes:** the floor is taken as the brief specifies, and a spill needs two consecutive samples.
  - **The boot step:** recorded per boot and not blocking.
  - **The k=7 run:** its first attempt was stopped by the earlier, blocking version of that rule. It is kept in
    `2026-10-15-Q2-k7/attempt1-bootrise-abort/` with no measurement.
- **`VLLM_LOG_STATS_INTERVAL=2`** on every arm.
- **Env files:** each arm's is `.env.<label>`, not `.env`.
