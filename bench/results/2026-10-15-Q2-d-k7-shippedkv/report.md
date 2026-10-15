# Batch 5 Q2: the launcher seat print on hardware (2026-10-15, RTX 3090 at 250 W)

This report is identical in `2026-10-15-Q2-{a-k7,b-k3,c-k7-cut1g,d-k7-shippedkv}/`. Each directory holds its own
arm's raw files.

**Verdict:**
- **The long end holds.** On all four arms the `MAX_LEN` figure is at or below the seats observed at ~32k, and equal to
  the seats observed at ~61k.
- **The short end fails on the band profiles.** On two of four arms the ~1k figure is *below* the observed seats: by 1
  at k=7 (printed 6, observed 7) and by 2 at k=3 (printed 11, observed 13).
- **The bias is one-sided and has two named causes:**
  - The formula charges 4 attention pages at ~1k that the engine's own accounting does not show.
  - The page constant is only right at k=7 on int8 KV. The engine picks an 816-token block at k=3 and a 448-token
    block on the shipped KV, not 864.
- **The print tracks the pool at k=7.** A 1 GiB cut moves it from 382 to 308 pages, exactly the engine's
  `num_gpu_blocks`.

## The arms

- **Common to all:** image `hyperqwen:exp-844cd90` (the rebuilt launcher, `../2026-10-15-batch5-image/`),
  `MAX_LEN=65536`, `MAX_SEQS=16`, one boot each.
- **Env files:** `.env` in each directory.

| arm | profile | `KV_MEM` |
|---|---|---|
| Q2-a | int8 `CTX=fast` (`EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head'`), k=7 | 5,583,457,484 |
| Q2-b | the same, k=3 (band 2) | 5,583,457,484 |
| Q2-c | the same as Q2-a, `KV_MEM` cut 1 GiB | 4,509,715,660 |
| Q2-d | **supplementary**: k=7 on the shipped KV (no `EXTRA_ARGS`), what a default user boots | 5,583,457,484 |

- **Why int8 for a-c:** the brief's Q2 rows name only `KV_MEM`, k, `MAX_SEQS` and `MAX_LEN`. The batch doc's
  expectations ("observed 6 at ~1k (batch 4)") and the launcher's constants come from int8 `CTX=fast` boots, so a-c run
  there.
- **Why Q2-d:** it is added because the launcher prints the same figure whatever the KV dtype.

**Method:**
- **Probe:** `running_probe.py` (batch 4 copy), three held-load waves per boot:
  - w1: 16 × ~1,050-token prompts (16 is the cap, above every printed figure).
  - w2: 6 × 32,817.
  - w3: 3 × 61,461 (the print's own `MAX_LEN` end, leaving room for 512 output tokens).
- **Seats:** the most the engine's own `Running:` line shows inside the held window (`raw-running-*.log`, lines marked
  `HELD`).
- **Pool:** `num_gpu_blocks` comes from `vllm:cache_config_info` in the `/metrics` text saved at each wave's peak
  (`raw-metrics-*.log`).
- **Blocks in use:** `vllm:kv_cache_usage_perc` × (`num_gpu_blocks` − 1).
- **Table:** `raw-q2-table.log` (`../2026-10-15-batch5-tools/q2_analyze.py`).

## Printed against observed

The printed lines, verbatim from each boot log (`raw-bootlines-*.log`):

```
Q2-a [start_qwen] note: the pinned pool holds about 6 resident requests on short (~1k-token) prompts and about 1 at MAX_LEN=65536, at DFLASH_TOKENS=7 (one request needs 54+2*ceil(ctx/864) of the pool's 382 pages, ...
Q2-b ... about 11 resident requests on short (~1k-token) prompts and about 2 at MAX_LEN=65536, at DFLASH_TOKENS=3 (one request needs 30+2*ceil(ctx/864) of the pool's 382 pages, ...
Q2-c ... about 5 resident requests on short (~1k-token) prompts and about 1 at MAX_LEN=65536, at DFLASH_TOKENS=7 (one request needs 54+2*ceil(ctx/864) of the pool's 308 pages, ...
Q2-d ... about 6 resident requests on short (~1k-token) prompts and about 1 at MAX_LEN=65536, at DFLASH_TOKENS=7 (one request needs 54+2*ceil(ctx/864) of the pool's 382 pages, ...
```

| arm | printed ~1k / `MAX_LEN` | engine `num_gpu_blocks` | observed ~1k | observed ~32k | observed ~61k | error at ~1k (printed − observed) | error at depth vs ~32k / ~61k | bound |
|---|---|---|---|---|---|---|---|---|
| Q2-a k=7 | 6 / 1 | 382 | **7** | 2 | 1 | **−1 (fail)** | −1 / 0 (hold) | **fails at ~1k** |
| Q2-b k=3 | 11 / 2 | **404** | **13** | 3 | 2 | **−2 (fail)** | −1 / 0 (hold) | **fails at ~1k** |
| Q2-c k=7, −1 GiB | 5 / 1 | 308 | 5 | 2 | 1 | 0 | −1 / 0 (hold) | holds |
| Q2-d k=7, shipped KV | 6 / 1 | 380 | 6 | 1 | 1 | 0 | 0 / 0 (hold) | holds |

The `Running:` lines behind the observed column (the peak line, then its count of 2 s samples):

| arm | ~1k | ~32k | ~61k |
|---|---|---|---|
| a | `Running: 7 reqs, Waiting: 9 reqs, GPU KV cache usage: 99.2%` ×3 | `Running: 2 reqs, Waiting: 4 reqs, GPU KV cache usage: 68.2%` ×4 | `Running: 1 reqs, Waiting: 2 reqs, GPU KV cache usage: 51.2%` ×3 |
| b | `Running: 13 reqs, Waiting: 3 reqs, GPU KV cache usage: 96.8%` ×6 | `Running: 3 reqs, Waiting: 3 reqs, GPU KV cache usage: 81.9%` ×3 | `Running: 2 reqs, Waiting: 1 reqs, GPU KV cache usage: 88.8%` ×3 |
| c | `Running: 5 reqs, Waiting: 11 reqs, GPU KV cache usage: 87.9%` ×2 | `Running: 2 reqs, Waiting: 4 reqs, GPU KV cache usage: 84.7%` ×4 | `Running: 1 reqs, Waiting: 2 reqs, GPU KV cache usage: 63.5%` ×2 |
| d | `Running: 6 reqs, Waiting: 10 reqs, GPU KV cache usage: 93.4%` ×1 | `Running: 1 reqs, Waiting: 5 reqs, GPU KV cache usage: 53.8%` ×2 | `Running: 1 reqs, Waiting: 2 reqs, GPU KV cache usage: 87.1%` ×1 |

- No wave preempted (`vllm:num_preemptions_total` delta 0 on all 12) and no request errored.
- No measurement spill on any arm (`mem.txt`).

## Where the error comes from: pages per request, measured against the print's formula

Blocks in use ÷ running at each wave's peak (`raw-q2-table.log`):

| arm | engine block (boot log) | ~1k | ~32k | ~61k | the print charges ~1k / ~32k / ~61k |
|---|---|---|---|---|---|
| a | `Setting attention block size to 864 tokens` | **54.00** | 129.00 | 195.00 | 58 / 130 / 198 |
| b | `Setting attention block size to 816 tokens` | **30.00** | 109.00 | 179.00 | 34 / 106 / 174 |
| c | 864 | **54.00** | 129.00 | 195.00 | 58 / 130 / 198 |
| d | `Setting attention block size to 448 tokens` | 57.50 | **204.00** | **332.00** | 58 / 130 / 198 |

The print charges 6(k+2) + 2·⌈ctx/864⌉, with ctx = 1,024 for the short end and the wave's prompt length (32,817 / 61,461) for the other two columns.

1. **At ~1k the engine shows exactly the state term**, 6(k+2) = 54 and 30, with no attention pages, at ~1,050-token
   prompts. The formula's `2*ceil(1024/864)` adds 4. At k=7 that is 381/58 = 6.6 → 6 against 381/54 = 7.06 → 7, the
   whole of Q2-a's miss.
2. **At k=3 the pool is not `KV_MEM / 14,601,709`.**
   - The engine's block is 816 tokens there, and the pool is **404 pages**, not the 382 printed (the batch-4 pre-cut
     boots read 385 against 364, the same direction).
   - Together with point 1, that is 403/30 = 13.4 → 13 against 381/34 = 11.2 → 11.
   - **The launcher's comment has the direction wrong.** It says reading the pool as `KV_MEM`/page "is exact at k=7 and
     a little high below it". Measured, it is **low** below k=7.
3. **On the shipped KV the block is 448 tokens, not 864**, so the context term is about twice the print's.
   - 204 = 54 + 2·⌈32,817/448⌉ (= 54 + 150) and 332 = 54 + 2·⌈61,461/448⌉ (= 54 + 278), both exact.
   - The print does not know the KV dtype. At `MAX_LEN=65536` it still lands on the observed 1, because 1 is also the
     floor.
   - **Computed risk** (not booted): the same arithmetic at `MAX_LEN=32768` prints `(382-1)//(54+2*38)` = **2**, while
     Q2-d's measured 204 pages per 32.8k-token request fits **1** in 379. That would be a long-end miss in the unsafe
     direction for a default-KV user who lowers `MAX_LEN`.
4. **The pool tracks `KV_MEM` at k=7.** Q2-c's printed pool (308) equals its `num_gpu_blocks` (308), and Q2-a's (382)
   equals its own. The print is not a constant in disguise.

**Cross-check the launcher's comment cites.** The engine's `Maximum concurrency for 65,536 tokens per request` read
1.78x / 2.01x / 1.43x / 1.05x on a / b / c / d. The unfloored print at `MAX_LEN` is 381/206 = 1.85, 381/182 = 2.09,
307/206 = 1.49 and 381/206 = 1.85.

## What the numbers say about the formula (not tuned here; the correction belongs in the launcher and its test)

- **Short end:**
  - **Under-states, conservative direction:** −1 at k=7 and −2 at k=3 on the full pin, 0 at the 1 GiB cut and on the
    shipped KV.
  - **Source:** the ~1k context charge (+4 pages that the engine does not show at ~1k), plus, at k=3, the page constant
    (404 against 382 pages).
- **Long end:** holds on all four. It is exact at ~61k, and at ~32k it is below the observed by 0-1, as a bound at 64k
  should be.
- **Not covered by its constants:**
  - k ≠ 7 (the engine's block becomes 816 at k=3).
  - Any non-int8 KV (block 448: the context term doubles).

  The comment's "calibrated at k=7" is accurate. What it does not say is that the int8 KV dtype is an equal part of the
  calibration.

## Memory (`mem.txt` per arm)

| arm | free VRAM after boot | guard floors (Shared Usage) | measurement spill |
|---|---|---|---|
| a | 1,185 MiB | `FLOORS F_ctx=86.2 F_kv=102.2` | none |
| b | 1,253 MiB | 86.2 → 102.2 | none |
| c | 2,215 MiB | 86.2 → 102.2 | none |
| d | 1,333 MiB | 86.2 → 102.2 | none |

- **Boot step:** the 16 MiB boot step is the known `MAX_SEQS=16` step (batch 4 O-3), recorded and not blocking.
- **The floor:** F_kv is the brief's floor: the minimum between the engine's `GPU KV cache size` line and the first
  request.

## Verified vs assumed

**Verified:** every number above is quoted from the boot logs, `/metrics` dumps and `Running:` lines in these four
directories.

**Assumed:**
- The `MAX_LEN=32768` shipped-KV miss is computed from the launcher's arithmetic and Q2-d's measured pages per
  request, not booted.
- Why the engine shows no attention pages at ~1k is not explained here.
