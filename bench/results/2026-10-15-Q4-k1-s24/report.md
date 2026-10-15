# Batch 4 Q4: k=1 above 16 seats, and the guard's baseline (2026-10-15, RTX 3090 at 250 W)

**Verdict:**
- **k=1 seats 21 requests at ~1k-token prompts** under `MAX_SEQS=24`, which closes batch 3's open cell. It seats 13-14
  at ~4k, with preemptions.
- **No measurement spilled.**
- **The guard is fixed.** It takes the brief's floor, and also reports the boot step the brief's window cannot see.

## Configuration

- **Base:** int8 `CTX=fast`, `DFLASH_TOKENS=1`, `MAX_SEQS=24`, `KV_MEM=5315022028`. That is the shipped pin pre-cut by
  256 MiB (batch 2's trick), the same value as every Q2 arm, so this boot differs from `../2026-10-15-Q2-k1/` only in
  `MAX_SEQS` (24 vs 16).
- **Env file:** `.env.q4k1s24`.
- **Pool:** `GPU KV cache size: 134,846 tokens`, `num_gpu_blocks="393"`, identical to Q2-k1.
- **Free VRAM after boot:** 1,455 MiB.

## Seats (`running_probe.py`, cap `MAX_SEQS=24`)

| wave | requests × prompt | HELD lines (count × line) | /metrics at peak | preemptions in wave |
|---|---|---|---|---|
| w1 | 24 × ~1,050 | 9 × `Running: 21 reqs, Waiting: 3 reqs, GPU KV cache usage: 96.4%` | usage 0.9642857142857143 × 392 = 378.0 blocks / 21 = **18.0 blocks per request** | 0 |
| w2 | 24 × ~4,130 | 9 × `Running: 13 reqs, Waiting: 11 reqs, GPU KV cache usage: 89.5%`, 7 × `Running: 14 reqs, Waiting: 10 reqs, GPU KV cache usage: 98.0%` | 384.0 blocks / 14 = 27.4 | 42 |

- **The Q1 formula predicted both rows:** 392 / 18 = 21.8 at ~1k (21 held) and 392 / 27.4 = 14.3 at ~4k (13-14 held,
  oscillating with preemptions).
- **Batch 3's pool-share figure:** "22.9 seats" at k=1 came from a 4.37% share on a 5,583 MB pool. This 256 MiB smaller
  pool holds 21.

Ladder (`raw-ladder-q4k1s24.json`, 4k-token prompts):

| N | decode agg | preempt | kv% | run | ms/pass | row |
|---|---|---|---|---|---|---|
| 8 | 299.0 | 0 | 60.7 | 8 | 48.1 | steady |
| 16 | 172.5 | 0 | 97.4 | 13 | 36.2 | TAIL |
| 24 | 299.4 | 0 | 97.4 | 13 | 60.3 | TAIL |

**On the ladder's 4k prompts the cap is not what binds.** Raising it from 16 to 24 seats no more streams: 13 running in
both. N=8 reads 299.0 against Q2-k1's 294.9 (+1.4%, same config apart from `MAX_SEQS`).

## Memory and the guard (`mem.txt`, `../2026-10-15-batch4-tools/spillwatch.sh`)

- **The floors:**

```
guard floors: FLOORS F_ctx=90.2 F_kv=122.2 n_kv=4 (window: KV-cache line 2026-10-15T13:21:13 -> first request 2026-10-15T13:21:52, local time)
BOOT RISE (recorded, not blocking): ... the boot rose 32.0 MiB above the CUDA-context level before the first request
measurements: no sample after the first request above F_kv+16 MiB: no spill seen
```

- **The step matches batch 3's spilled boot:** 90.2 → 122.2 MiB, the same as batch 3's `MAX_SEQS=24` boot at the full
  pin (`q1xk1s24`). This boot had 256 MiB more KV headroom. With the Q2 discriminator (`q2k7cut1g`, the same 16 MiB step
  at 1 GiB more free VRAM), the step is 8 / 16 / 32 MiB at `MAX_SEQS` 8 / 16 / 24 regardless of free VRAM. It is not
  memory pressure.
  - **What that does to batch 3:** its "spilled 32 MiB, could not measure" was this step. The pre-cut did not remove it
    and did not need to.
- **Guard status: fixed, not worked around.**
  - **Measurement spills:** called against F_kv, the minimum Shared Usage between the engine's `GPU KV cache size` line
    and the first request. That is the brief's floor, with two consecutive samples required. The timestamps come from
    the container log (`.kvline-<label>`) and the driver (`.firstreq-<label>`).
  - **Boot step:** F_ctx, the pre-profiling level, is kept beside it to report the boot step, because profiling runs
    before the KV-cache line and the brief's window starts above it.
  - **Replay:** on batch 3's traces this flags all four `MAX_SEQS` > 8 boots and neither `MAX_SEQS=8` boot
    (`guard-test.txt`).

## Verified vs assumed

**Verified:** every number is from this directory's raw files.

**Assumed:** "per-seat host memory" as the source of the step is an [INFERENCE]. What is measured is that it scales with
`MAX_SEQS` and not with free VRAM.
