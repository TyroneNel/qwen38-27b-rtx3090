# Batch 4 Q2: which speculation setting wins at each concurrency (2026-10-15, RTX 3090 at 250 W)

This report appears unchanged in `2026-10-15-Q2-k7`, `-Q2-k3`, `-Q2-k1` and `-Q2-off` (one boot each, labels q2k7,
q2k3, q2k1, q2off).

**Verdict:**
- **One stream:** k=7 wins.
- **2-8 streams:** k=3 wins. At 2 the lead over k=7 is within noise.
- **16 streams:** `SPEC=off` wins, and is the only arm that seats all 16.
- **The crossover:** it sits between 8 and 16 streams, at the point where k=3 runs out of seats: 9 resident at 4k-token
  prompts.
- **k=1:** wins no column.

## Configuration, identical except the speculation setting

| | value |
|---|---|
| base | int8 `CTX=fast`: `EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head'`, `PREFIX_CACHE=1`, `MAX_LEN=65536`, `SPEC=dflash2` (k arms) |
| `MAX_SEQS` | **16 on every arm** (the cap the brief requires; it binds only for `SPEC=off` at N=16, which seats all 16) |
| `KV_MEM` | **5315022028 on every arm**: the shipped 5583457484 pin pre-cut by 256 MiB, chosen before the run because batch 3's `MAX_SEQS=16` boots rose at boot (see "Memory") |
| arms | `DFLASH_TOKENS=7` / `3` / `1`, and `SPEC=off` (`DFLASH_TOKENS` line removed) |
| also | `VLLM_LOG_STATS_INTERVAL=2` on every arm; the 3090 pinned by UUID |
| env file | `.env.<label>` in each directory (the file copied to `.env`), not a file named `.env` |
| image | `hyperqwen:exp-f1a033e`, sha256:c06b97c7c9eb… (`image.txt`) |

## The table (metric: `bench/conc_ladder.py --n 1,2,4,8,16` decode aggregate, tok/s; 4k-token prompts, 256 out, 2 reps)

| N | k=7 | k=3 | k=1 | `SPEC=off` | winner |
|---|---|---|---|---|---|
| 1 | **105.4** | 100.9 | 60.9 | 40.1 | k=7 (+4.5% over k=3) |
| 2 | 167.4 | **170.0** | 115.7 | 77.5 | k=3 by 1.6%: within the 3.2% cross-boot floor, so a tie with k=7 |
| 4 | 239.1 | **263.6** | 202.3 | 143.2 | k=3 (+10.2%) |
| 8 | 176.8 TAIL (run 5) | **337.1** | 294.9 | 250.7 | k=3 (+14.3% over k=1, +34.5% over off) |
| 16 | 156.1 TAIL (run 5) | 300.4 TAIL (run 9) | 180.8 TAIL (run 13) | **369.2** | `SPEC=off`: its only row with all 16 running |
| C1 T=0 (cost column, run 2) | 103.4 | 93.5 | 63.3 | 40.7 | |
| C8 cohort T=0 (real prompts, run 2) | 323.8 | **421.9** | 373.1 | 293.0 | |

**Steady state vs tail rows:**
- **Preemptions:** none in any ladder row.
- **TAIL rows:** a row is marked TAIL when the server ran fewer streams than were sent (`run` < N). Those rows are a
  queue, not steady state.
- **Steady rows:** every N≤4 row, k=3/k=1/off at N=8, and `SPEC=off` at N=16.
- **What settles the N=16 crossover:** the k-arm N=16 rows are all tails, so the comparison is not decided by them. It
  is decided by `SPEC=off` being the only arm that holds 16 streams at all, plus the seat counts below.

`ms/pass` and tok/step, same rows:

| N | k=7 | k=3 | k=1 | off |
|---|---|---|---|---|
| 1 | 32.7 ms, 3.45 | 29.4, 2.98 | 29.7, 1.81 | 25.0, n/a (no drafts) |
| 8 | 55.2, 3.46 (TAIL) | 63.9, 2.79 | 49.1, 1.81 | 31.9, n/a |
| 16 | 44.3, 3.58 (TAIL) | 58.0, 2.88 (TAIL) | 36.4, 1.82 (TAIL) | 43.3, n/a |

## Seats: the held `Running:` probe (`2026-10-15-batch4-tools/running_probe.py`, cap `MAX_SEQS=16`)

Last HELD line of each wave, verbatim, from `raw-running-<label>-w<w>.log`:

| arm | 8 requests, ~1k-token prompts (w1) | 16 requests, ~4k-token prompts (w2) | blocks per request (w1 / w2) | pool |
|---|---|---|---|---|
| k=7 | `Running: 6 reqs, Waiting: 2 reqs, GPU KV cache usage: 89.3%` | `Running: 5 reqs, Waiting: 11 reqs, GPU KV cache usage: 89.8%` | 54 / 64 of 364 | 110,953 tokens |
| k=3 | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 62.5%` | `Running: 9 reqs, Waiting: 7 reqs, GPU KV cache usage: 92.2%` | 30 / 39.3 of 385 | 125,529 |
| k=1 | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 36.7%` | `Running: 14 reqs, Waiting: 2 reqs, GPU KV cache usage: 98.0%` (42 preemptions in the wave) | 18 / 27.4 of 393 | 134,846 |
| off | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 20.2%` | `Running: 16 reqs, Waiting: 0 reqs, GPU KV cache usage: 72.7%` | 5 / 9 of 199 | 148,200 |

**Blocks per request:**
- **How it is computed:** `vllm:kv_cache_usage_perc` × (`num_gpu_blocks` − 1) ÷ `vllm:num_requests_running`, read from
  the /metrics text each wave saved at its peak (`raw-metrics-<label>-w<w>.log`). The product is an exact integer on
  all 11 dumps (Q1, `../2026-10-15-Q1-seat-gauge/report.md`).
- **Short prompts:** the k arms cost 54 / 30 / 18 blocks at ~1k-token prompts, exactly 6(k+2).
- **The crossover follows seats:** the ~4k probe (the ladder's prompt depth) seats 5 / 9 / 13-14 / 16. Each k arm's
  aggregate stops rising at the N where N first exceeds its seats.

## Memory (`mem.txt`, guard `2026-10-15-batch4-tools/spillwatch.sh`)

| arm | free VRAM after boot | Shared Usage F_ctx → F_kv (floor) | measurement samples above F_kv+16 MiB |
|---|---|---|---|
| k=7 | 1,437 MiB | 86.2 → 102.2 MiB | none |
| k=3 | 1,501 MiB | 86.2 → 102.2 | none |
| k=1 | 1,661 MiB | 86.2 → 102.2 | none |
| off | 3,381 MiB | 86.2 → 102.2 | none |

Every `MAX_SEQS=16` boot steps 16.0 MiB at memory profiling, against 8.0 MiB on `MAX_SEQS=8` boots. **This is not a
spill:**
- **The discriminator:** `q2k7cut1g` is the k=7 boot again with `KV_MEM` a further 1 GiB lower (4241280204; 88,397
  tokens; 2,465 MiB free). It steps the same 86.2 → 102.2 MiB.
- **What the step tracks:** it does not shrink with 1 GiB more free VRAM, and `SPEC=off` with 3,381 MiB free steps the
  same. So it follows `MAX_SEQS`, not memory pressure.
- **The guard:** records this step (`BOOTRISE-<label>`) and does not stop on it. It stops only on a measurement spill
  against F_kv. None occurred.
- **The first k=7 attempt:** it stopped on this step before the evidence existed. It is kept in
  `2026-10-15-Q2-k7/attempt1-bootrise-abort/`; no measurement was taken on it.

## Contradictions and caveats, with numbers

- **k=7 C1 T=0 is 7.8% below batch 3's k=7 boot:** 103.4 against 112.1, beyond the 3.2% cross-boot floor.
  - **Where it shows:** the N=1 step is 32.7 against 29.5 ms/pass in the ladder, and tok/step is 3.17 against 3.34 in the
    C1 cohort. The same comparison for k=3 is 93.5 vs 94.1 (−0.6%) and for k=1 63.3 vs 64.7 (−2.2%).
  - **What changed:** `MAX_SEQS` 8 → 16 and the 256 MiB pre-cut. The boot log shows one `MAX_SEQS`-sized buffer:
    `int4 3D scratch: capacity=256 query tokens` here against `128` at `MAX_SEQS=8`.
  - **Cause:** not isolated. So the k=7 C1 cost of a 16-seat boot is a measured 7.8%, but which knob causes it is not
    verified.
  - **Effect on the table:** the N=1 column is from 16-seat boots. A one-stream deployment at the shipped `MAX_SEQS=8`
    reads 112.1 (batch 3).
- **Seats at k=7, ~1k prompts:**
  - **Pool:** the pool is smaller here (pre-cut, 364 blocks), and k=7 holds **6**, not batch 3's 7.
  - **Same cost in bytes:** 54.2 blocks × 14.60 MB = 791 MB per request, against batch 3's 14.2% × 5,583 MB = 793 MB.
  - **The batch-4 docs' number is pool-dependent:** they quote "7 seats" for k=7. That figure holds at the full pin
    only.
- **The landscape's "speculation wins below ~8 concurrent users":**
  - **Holds for k=3:** it wins at 8 (337.1, steady) and loses at 16.
  - **Fails for k=7:** k=7 loses from N=8 on 4k-token prompts (176.8 at 5 running).
  - **The right line:** the break is a seat count per k, not one N.

## Verified vs assumed

**Verified:**
- Every number above is quoted from the raw logs in these four directories.
- `analyze.py` (in `2026-10-15-batch4-tools/`) only reshapes them.

**Assumed, marked here:**
- **The per-group reading of "6(k+2)".** That six of the nine KV-cache groups are the 48 GDN layers (`kv cache group
  sizes [864 × 9]`; the config has 48 `linear_attention`, 16 `full_attention` and 5 drafter `sliding_attention` layers,
  grouped by 8) is [INFERENCE]. The block counts themselves are measured.
- **No quality claim:** none was measured and none is made.
