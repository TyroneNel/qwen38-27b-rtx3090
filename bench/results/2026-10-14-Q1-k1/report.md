# Batch 3 Q1: does residency scale with k? (2026-10-14, RTX 3090 at 250 W)

The same report is in each Q1 arm directory: `2026-10-14-Q1-k7`, `-Q1-k3`, `-Q1-k1`. The supplementary uncapped
probe is in `2026-10-14-Q1x-uncapped`.

**Conclusion A: the model holds.** Residency is set by k, through the recurrent-state pages. The `RESIDENT` formula
has the right shape, but its constant is wrong. It **under**-predicts: it prints 5 seats at k=7, and the pool holds 7.

## Arms

All three arms share one config except `DFLASH_TOKENS`: the int8 `CTX=fast` base (`TRITON_ATTN` +
`int8_per_token_head`), `KV_MEM=5583457484`, `MAX_LEN=65536`, `MAX_SEQS=8`, `PREFIX_CACHE=1`, the 3090 by UUID,
image `hyperqwen:exp-f1a033e` (sha256:c06b97c7…). They also share `VLLM_LOG_STATS_INTERVAL=2`, so the engine's
stats line prints every 2 s instead of every 10 s and a 512-token wave yields several `Running:` samples. Each arm
had one boot, in the order k=7, 3, 1. The engine's args line shows `num_spec_tokens=7/3/1` (`raw-bootlines-*.log`).

## Primary metric: the engine's `Running: N reqs, Waiting: M reqs` line under a held 8-request load

The probe is `bench/results/2026-10-14-batch3-tools/running_probe.py`, run through `arm.sh probe`. It sends 8
concurrent requests at once on a barrier. The prompts are distinct and about 1.1k tokens, salted per request,
wave and arm. Each request asks for 512 tokens with `ignore_eos`. The `Running:` lines come from the container log
over each wave's window and are marked `HELD` when they fall between the last submission and the first stream
finishing. The table gives the steady pair, which is the last `HELD` value of each wave, verbatim. Every
`HELD` line is in `raw-running-<label>-w{1,2,3}.log`.

| arm | wave 1 | wave 2 | wave 3 |
|---|---|---|---|
| Q1-a k=7 | `Running: 7 reqs, Waiting: 1 reqs, GPU KV cache usage: 99.2%` | `Running: 7 reqs, Waiting: 1 reqs, GPU KV cache usage: 99.2%` | `Running: 7 reqs, Waiting: 1 reqs, GPU KV cache usage: 99.2%` |
| Q1-b k=3 | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 59.6%` | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 59.6%` | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 59.6%` |
| Q1-c k=1 | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 35.0%` | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 35.0%` | `Running: 8 reqs, Waiting: 0 reqs, GPU KV cache usage: 35.0%` |

The earlier lines of each wave are the admission ramp: 3/5, 4/4, 6/2 and so on, while the 8 prompts prefill under
the 2,048-token batch budget. At k=7 the eighth request waits with the pool at 99.2%. At k=3 and k=1, all 8 run
with 40% and 65% of the pool free.

**The literal rule cannot fire on this probe.** Running went 7 → 8 → 8, which is "within ±1". It cannot rise by 3
from k=7 because k=7 already holds 7, and `MAX_SEQS=8` caps every arm at 8. At k=3 and k=1 the cap binds, not the
pool: `Waiting: 0` with 59.6% and 35.0% used. So the batch doc's Conclusion B case ("5-6 regardless of k") is not
what happened.

### Supplementary: the same probe with the admission cap lifted (`2026-10-14-Q1x-uncapped`)

These boots are identical to Q1 except `MAX_SEQS=16` and the probe wave size N=16. They are outside the Q1 arm
definition, which fixes `MAX_SEQS=8`, and exist only to take the cap out of the discriminating metric. There were
three waves per boot. The steady pair, the same in every wave:

| k | `MAX_SEQS` | steady `Running:` line (all three waves) | GPU KV cache size line |
|---|---|---|---|
| 7 | 16 | `Running: 7 reqs, Waiting: 9 reqs, GPU KV cache usage: 99.2%` | 116,440 tokens |
| 3 | 16 | `Running: 13 reqs, Waiting: 3 reqs, GPU KV cache usage: 96.8%` | 131,724 tokens |
| 1 | 16 | `Running: 16 reqs, Waiting: 0 reqs, GPU KV cache usage: 69.9%` | 141,708 tokens |

**From k=7 to k=3, Running rises by 6 (7 → 13), which is the Conclusion A threshold (≥3) met on the uncapped
probe.** At k=1 the cap binds again, at 16. A k=1 boot at `MAX_SEQS=24` was attempted first. It was not measured:
Shared Usage rose from 90.2 to 122.2 MiB at the engine's memory-profiling peak and stayed there, the guard flagged
it, and the probe was refused (`cmd.txt`, `SPILL-q1xk1s24`, `mem.txt`). It was rerun at `MAX_SEQS=16`.

## Seats per k

| k | launcher `RESIDENT` (printed) | `KV_MEM / ((k+2) × 104988089)` | Running at `MAX_SEQS=8` | Running uncapped | pool used per running request | seats the pool holds (100% / per-request) |
|---|---|---|---|---|---|---|
| 7 | 5 ("holds about 5 resident requests") | 5.91 | 7 (Waiting 1) | **7** (Waiting 9) | 99.2 / 7 = 14.2% | 7.1 |
| 3 | not printed (10 ≥ `MAX_SEQS`) | 10.64 | 8 (Waiting 0, cap) | **13** (Waiting 3) | 96.8 / 13 = 7.4% | 13.4 |
| 1 | not printed (17 ≥ `MAX_SEQS`) | 17.73 | 8 (Waiting 0, cap) | ≥16 (Waiting 0, cap) | 69.9 / 16 = 4.4% | 22.9 |

The pool used per request comes from the engine's own `GPU KV cache usage` and is the same at both caps: k=3
gives 59.6/8 = 7.45% and 96.8/13 = 7.45%, and k=1 gives 35.0/8 = 4.38% and 69.9/16 = 4.37%. The share scales close
to (k+2): 14.2 : 7.4 : 4.4 = 1 : 0.53 : 0.31, against 9 : 5 : 3 = 1 : 0.56 : 0.33. So the form `KV_MEM / (k+2)` holds.
The constant does not. Per (k+2) unit, a request here costs 5,583,457,484 × 14.2% / 9 = 88 MB at k=7, 83 MB at k=3
and 81 MB at k=1. That includes each request's ~1.6k tokens of context. The launcher uses 105 MB, so it
under-predicts seats by 16-23%: 5.9 against 7.1, 10.6 against 13.4, 17.7 against 22.9. **[INFERENCE]** These are the
probe's ~1.6k-token requests. With longer prompts the per-request share rises: the ladder's 4k-token prompts used
16.6% per request at k=7 (99.7% with 6 running) and 10.4% at k=3 (82.9% with 8 running), as the launcher's "fewer
with long prompts" says.

## Secondary metric: `bench/conc_ladder.py --n 1,2,4,8,16` (4k-token prompts, 256 out, 2 reps)

| N | k=7 decode agg / run / preempt / kv% | k=3 | k=1 |
|---|---|---|---|
| 1 | 114 / 1 / 0 / 19.7 | 100 / 1 / 0 / 13.4 | 65 / 1 / 0 / 10.2 |
| 4 | 239 / 4 / 0 / 70.3 | 268 / 4 / 0 / 43.2 | 204 / 4 / 0 / 30.6 |
| 8 | **178.5\*** / 6 / **2** / 99.7 | **350.1** / 8 / **0** / 82.9 | **299.4** / 8 / **0** / 57.8 |
| 16 | **213.8\*** / 6 / **29** / 100.0 | **333.7\*** / 8 / **0** / 83.9 | **291.7\*** / 8 / **0** / 57.8 |

`*` marks the ladder's own tail flag: there was no instant at which all N streams were decoding, so the row is a
tail measurement. At N=16 every arm is tail, because `MAX_SEQS=8` queues half the streams. At k=7, N=8 is also
tail: 2 preemptions, 6 running. The ladder's `run` column agrees with the log line in direction. At k=7 it reads
6, not 7, because of the longer 4k-token prompts. **Preemptions fall to 0 from k=7 to k=3 (N=8: 2 → 0; N=16:
29 → 0), and the N=8 aggregate rises 178.5\* → 350.1 (+96%).** All three Conclusion A conditions hold, with the
first one read on the uncapped probe.

Per-pass cost and acceptance at N=1: `tok/step` 3.36 / 2.91 / 1.83, `ms/pass` 29.5 / 28.9 / 28.1.

## C1 T=0 (`bench/run_benchmarks.sh single`, second run of each boot, `raw-rb-<label>-2.log`)

| k | C1 T=0 decode tok/s | tok/step | C8 T=0 decode tok/s |
|---|---|---|---|
| 7 | 112.1 | 3.34 | 371.7 |
| 3 | 94.1 (−16.1%) | 2.66 | 439.3 |
| 1 | 64.7 (−42.3%) | 1.78 | 383.0 |

Seats cost single-stream speed. k=3 gives up 16% at C1 to hold 13 requests instead of 7. Its C8 cohort row is the
best of the three.

## Batch 2's contradiction is explained: its "k=3" arm allocated state for k=7

Batch 2's S2a arm was not k=3. It was the dynamic-K schedule `[[1,3,7],[4,512,3]]` on `num_speculative_tokens=7`.
Its boot log (`../2026-10-14-S2a-dynamic-k/raw-boot-Bdk1.log`) shows `num_spec_tokens=7` and an attention block of
864 tokens, the same as Q1-a's k=7 and unlike k=3's 816. Its own `Running:` lines show the k=7 per-request share:
`Running: 6 reqs, Waiting: 1 reqs, GPU KV cache usage: 84.3%` is 14.05% per request, against 14.2% at k=7 here, and
`Running: 7 reqs, Waiting: 1 reqs, GPU KV cache usage: 98.3%` five times. State pages are reserved for the
configured maximum k, not for the k the schedule runs at. So dynamic K buys step time but no seats. Only a lower
configured `DFLASH_TOKENS` buys seats.

## Boot-log evidence per arm (`raw-bootlines-<label>.log`)

| k | `RESIDENT` note | attention block size | mamba page padding | GPU KV cache size |
|---|---|---|---|---|
| 7 | `[start_qwen] note: the pinned pool holds about 5 resident requests at DFLASH_TOKENS=7 (state pages), fewer with long prompts; MAX_SEQS=8 admits more than that and the rest queue.` | 864 | 1.09% | 116,440 tokens |
| 3 | (none: 10 ≥ `MAX_SEQS`) | 816 | 0.09% | 131,724 tokens |
| 1 | (none: 17 ≥ `MAX_SEQS`) | 800 | 0.56% | 141,708 tokens |

Every arm prints `using group_size 8` and `GDN decode kernel: triton`, and none prints a count of mamba blocks or
state slots (see Q4).

## Memory

Free VRAM after boot was 1,365 MiB (k=7), 1,533 MiB (k=3) and 1,595 MiB (k=1), from `raw-smi-*.log`. On the three
Q1 boots, the 3090's Shared Usage stayed at the 84.2-92.2 MiB loaded floor: "no spill seen" in each `mem.txt`.

**The uncapped boots spilled a little at boot.** On all three `MAX_SEQS=16` boots, Shared Usage went from 86.2 to
102.2 MiB just after the engine's memory-profiling step, before the boot was healthy, and stayed there. The trace is
in `../2026-10-14-Q1x-uncapped/mem-trace-*.csv`. The `MAX_SEQS=24` boot did the same, from 90.2 to 122.2 MiB. The
guard flagged only the 24 case. On the 16 boots its loaded-engine floor was taken after the rise (102.2 MiB), so
the +16 MiB never showed against it. That is a blind spot: a rise during profiling, before dedicated passes
20,000 MiB, sets the floor instead of breaking it. The uncapped numbers are scheduler counts from the engine's
block accounting, not timings. The k=3 per-request share is 7.45% at both `MAX_SEQS=8` (no spill) and
`MAX_SEQS=16` (spill), and k=1's is 4.38% vs 4.37%. So the spill did not move what they report. The Q1 arms
proper, at `MAX_SEQS=8`, did not spill.

## Verified vs assumed

Verified: every `Running:` pair, ladder row, C1 row and boot line is quoted from the raw logs in these directories.
The seats-per-k ratios are arithmetic on those quotes. Assumed and marked **[INFERENCE]**: that the per-request
share at other prompt lengths follows the ladder's two points. The k=1 wave 2 was cut off by the driver's 3600 s
shell limit and rerun on the same boot. The empty file is kept as `raw-probe-q1k1-w2-interrupted.*` (`cmd.txt`).
