# Batch 4 Q1: can seats be read directly, and is the per-request cost a constant? (2026-10-15, RTX 3090 at 250 W)

**Answers:**
- **No state gauge exists.** Seats can still be read exactly, in blocks: `vllm:kv_cache_usage_perc` ×
  (`num_gpu_blocks` − 1), where `num_gpu_blocks` is a label on `vllm:cache_config_info`, gives the blocks in use as an
  exact integer.
- **The per-request cost is not a constant.** It is affine in context, about 53 + 2 blocks per 864 prompt tokens at k=7,
  and 6(k+2) blocks at short prompts.
- **So the launcher constant is wrong in principle.** The print has to become a function of k and the prompt length.

## Where it ran

- **The boot:** the probes ran on Q2's k=7 boot, `../2026-10-15-Q2-k7/` (label q2k7). Raw files are there:
  `raw-running-q2k7-w1..5.log`, `raw-probe-q2k7-w1..5.json` and `raw-metrics-q2k7-w1..5.log`. Excerpts are in
  `raw-metrics-excerpt.log` here.
- **The config:** int8 `CTX=fast`, `DFLASH_TOKENS=7`, `KV_MEM=5315022028` (the same on all Q2 arms), `MAX_SEQS=16`.
- **The probe:** `../2026-10-15-batch4-tools/running_probe.py`. Each wave sends N requests with distinct salted prompts,
  512 tokens out, released together. At each wave's peak `Running` count it saves the full `/metrics` text.

## 1. Metrics: what `/metrics` has

The full list of `vllm:` metric families on this pin is in `raw-metrics-excerpt.log`:
- **State-shaped, per group:** nothing. No family names mamba, state, slot or KV-cache group. The only "state" hit is
  `vllm:engine_sleep_state`.
- **Pool-shaped:**

| metric | value (q2k7) | kind |
|---|---|---|
| `vllm:cache_config_info{… num_gpu_blocks="364" …}` | label, static | the pool in blocks. k=3 385, k=1 393, `SPEC=off` 199 |
| `vllm:cache_config_info{… kv_cache_max_concurrency="1.6930232558139535" …}` | label, static | the engine's 65,536-token-request concurrency; says nothing about short requests |
| `vllm:cache_config_info{… mamba_block_size="864", mamba_cache_mode="align", block_size="864" …}` | label, static | |
| `vllm:kv_cache_usage_perc` | gauge | fraction of blocks in use |
| `vllm:num_requests_running`, `vllm:num_requests_waiting` | gauges | |

**`kv_cache_usage_perc` is block-exact.** On all 11 dumps of the batch (4 arms), usage × (`num_gpu_blocks` − 1) is an
integer to four decimals (`2026-10-15-Q2-*/raw-metrics-*.log`):

| dump | usage | × (nb − 1) | running | blocks per request |
|---|---|---|---|---|
| q2k7 w1 (~1k prompts) | 0.8925619834710744 | 324.0000 | 6 | 54.0 |
| q2k7 w2 (~4k) | 0.8815426997245179 | 320.0000 | 5 | 64.0 |
| q2k7 w4 (~16k) | 0.7520661157024793 | 273.0000 | 3 | 91.0 |
| q2k7 w5 (~32k) | 0.7107438016528926 | 258.0000 | 2 | 129.0 |
| q2k3 w1 (~1k) | 0.625 | 240.0000 | 8 | 30.0 |
| q2k1 w1 (~1k) | 0.36734693877551017 | 144.0000 | 8 | 18.0 |
| q2off w1 (~1k) | 0.202020202020202 | 40.0000 | 8 | 5.0 |

**Meaning:**
- **What it gives:** a direct reading of **blocks in use**. Divided by Running, it gives blocks per resident request
  with no inference about bytes.
- **What it does not give:** it is pool-wide, not per group. Which part of a request's blocks is recurrent state and
  which is attention KV cannot be read off it.
- **The patch that would close that:** a per-group counter is still the candidate patch.
- **For future seat questions:** this pair is the primary metric. Seats = (`num_gpu_blocks` − 1) ÷ blocks per request.

## 2. Is the per-request share constant across depth? No

k=7, `MAX_SEQS=16`. Running/Waiting are from the engine's own stats lines (HELD = taken while every request of the wave
was in the server). Blocks are from the `/metrics` dump at the wave's peak:

| wave | requests × prompt tokens (median) | last HELD line | blocks / request | MB / request (× 14.60 MB page) | seats = 363 / blocks |
|---|---|---|---|---|---|
| w1 | 8 × 1,050 | `Running: 6 reqs, Waiting: 2 reqs, GPU KV cache usage: 89.3%` | 54 | 788 | 6.7 |
| w3 | 16 × 1,050 | `Running: 6 reqs, Waiting: 10 reqs, GPU KV cache usage: 89.3%` | 54 | 788 | 6.7 |
| w2 | 16 × 4,130 | `Running: 5 reqs, Waiting: 11 reqs, GPU KV cache usage: 89.8%` | 64 | 935 | 5.7 |
| w4 | 6 × 16,406 | `Running: 3 reqs, Waiting: 3 reqs, GPU KV cache usage: 75.2%` | 91 | 1,329 | 4.0 |
| w5 | 4 × 32,818 | `Running: 2 reqs, Waiting: 2 reqs, GPU KV cache usage: 71.1%` | 129 | 1,884 | 2.8 |

**A least-squares line through the four depths:** blocks = **52.9 + 2.01 per 864 prompt tokens** (fitted 55.3 / 62.5 /
91.0 / 129.1 against 54 / 64 / 91 / 129). The per-request share grows **2.4×** from ~1k to ~32k prompts. A constant is
wrong in principle.

**The two terms, and what they correspond to:**
- **The slope:** 2.01 blocks per 864 tokens is two full-attention KV groups' worth. The boot log has `kv cache group
  sizes [864, 864, 864, 864, 864, 864, 864, 864, 864]`, nine groups. The model config has 16 `full_attention`, 48
  `linear_attention` and 5 drafter `sliding_attention` layers, grouped by 8 (`using group_size 8`). [INFERENCE] That
  is 2 attention groups, 6 GDN groups and 1 drafter group.
- **The intercept is the recurrent state, and it scales exactly as 6(k+2):** at ~1k prompts the k arms cost **54 / 30 /
  18 blocks** at k=7/3/1, against 6 × 9 / 6 × 5 / 6 × 3 = 54 / 30 / 18.
- **In bytes:** 788 / 414 / 243 MB per request, i.e. **87.6 / 82.8 / 81.1 MB per (k+2)**. That reproduces batch 3's
  81-88 MB from the other direction.

**What a corrected launcher print would be:**
- **The formula:** seats ≈ (`num_gpu_blocks` − 1) ÷ (6(k+2) + 2·⌈ctx / block⌉).
  - **Measured at k=7:** 363/54 = 6.7 at ~1k (6 held) and 363/129 = 2.8 at 32k (2 held).
  - **Predicted at ~4k:** 384/39.3 = 9.8 at k=3 (9 held) and 392/27.4 = 14.3 at k=1 (13-14 held, with 42 preemptions).
- **The current print:** `KV_MEM / ((k+2) × 104988089)` uses a constant 20-29% above the measured short-prompt cost
  (104.99 against 81.1-87.6 MB). It also has no context term, so it is right at about one prompt length and wrong at
  every other.
- **What this batch does not do:** the brief's alternatives are "corrected in one line" if constant, else "removed or
  made a function". The function above is the evidence for the second. No launcher edit is made here (rule 1).

## Contradictions, with numbers

- **"A constant simply 16-23% high"** (batch-4 doc, Q1). That is right at ~1k prompts only. Per (k+2) the launcher
  constant is 20-29% above the measured cost, and the gap closes and then reverses as the prompt grows: at 32k, k=7 costs
  1,884 MB = 209 MB per (k+2).
- **The launcher comment's "six resident, the seventh preempted" (15.8% per request).** On this pool and ~1k prompts,
  k=7 holds 6 and the rest wait (14.88% per request). At ~4k prompts it holds 5.

## Verified vs assumed

**Verified:** every count and value above is quoted from the raw files named.

**Assumed, marked:**
- **The group mapping** (which of the nine groups are attention, GDN and drafter) is [INFERENCE] from the config's
  layer counts and `group_size 8`.
- **`SPEC=off`'s 5 blocks** at ~1k are not decomposed. Its boot has 4 groups (`kv cache group sizes [800, 800, 800,
  800]`), so the k-arm formula does not apply to it.
