# E1: load-aware K (dynamic speculative decoding)

RTX 3090, 250 W (`power.limit 250.00 W` in every `raw-smi-*.log`), container boot.

## Did the engine get the knob?

Yes. No launcher change was needed. The launcher passes its own `--speculative-config`. A second one in
`EXTRA_ARGS` comes later on the same command line (`raw-cmdline-B.log` shows both), and the later one wins.
The engine's `non-default args` line in `raw-boot-B.log` shows:

```
'speculative_config': {'method': 'dflash', 'model': '/app/models/Qwen3.8-27B-DFlash2-W4A16', 'num_speculative_tokens': 7, 'draft_sample_method': 'probabilistic', 'num_speculative_tokens_per_batch_size': [[1, 8, 7], [9, 32, 3], [33, 512, 0]]}
```

`raw-boot-B2.log` shows `'num_speculative_tokens_per_batch_size': [[1, 3, 7], [4, 512, 3]]`. The field exists in
the image's `vllm/config/speculative.py:492`, and `vllm/v1/core/sched/scheduler.py` consumes it
(`../2026-10-13-image/raw-dynamic-k-source.log`). The `DYNAMIC_K` launcher knob from the brief was therefore not
added, and no file outside `bench/results/` changed.

## Arms

All arms use `SPEC=dflash2 PREFIX_CACHE=1 CTX=fast DFLASH_TOKENS=7 KV_MEM=5583457484 MAX_LEN=65536`, with
`MAX_SEQS` at the profile default of 8 (`--max-num-seqs 8` in `raw-cmdline-*.log`).

- **A, A2**: shipped. A2 is a second boot of A, for A-B-A.
- **B**: the experiment list's schedule, `[[1,8,7],[9,32,3],[33,512,0]]`.
- **B2** (added): `[[1,3,7],[4,512,3]]`. A's ladder showed that the schedule in B cannot change anything at
  `MAX_SEQS=8`: `run` (resident requests) never goes above 5 and preemptions start at N=8. A batch of 9 or more
  therefore never forms, and B always runs K=7. B2 puts the knee where A's ladder says the batch actually is.
  The landscape page (G1) asks for exactly this: "with the knee read off a C1-C8-C32 ladder".

Every boot reported the same pool: `GPU KV cache size: 68,605 tokens`. Graph capture `took 0.13 GiB`.

## Results

Metrics:

- **decode agg**: `bench/conc_ladder.py --n 1,2,4,8,16,32` (out 256, ~4k-token distinct prompts, 2 reps),
  tok/s.
- **C1 T=0**: `bench/run_benchmarks.sh single` row `cohort C1 real prompts T=0`, decode(C/meanTPOT). I kept the
  second run of each boot.

`*` means the ladder printed its own warning: "no instant at which all N streams were decoding: the pool could
not hold them, so that row is a tail measurement, not a steady state".

| metric | A | A2 | B | B2 | B2 vs mean(A,A2) |
|---|---|---|---|---|---|
| C1 T=0 decode, tok/s | 109.4 | 109.4 | 109.5 | 109.2 | -0.2% |
| C1 T=default decode, tok/s | 99.6 | 107.2 | 99.5 | 106.2 | +2.7% (A/A2 differ by 7.6%) |
| ladder N=1 decode agg | 102.7 | 116.0 | 107.2 | 100.3 | -8.3% (A/A2 differ by 13%) |
| ladder N=4 decode agg | 243.0 | 241.5 | 235.1 | 261.6 | **+8.0%** |
| ladder N=4 ms/pass | 57.0 | 56.1 | 57.3 | 43.2 | -23.6% |
| ladder N=4 tok/step | 3.50 | 3.49 | 3.47 | 2.95 | -15.6% |
| ladder N=8 decode agg | 213.4* | 242.0* | 234.1* | 247.1* | +8.5% (A/A2 differ by 13%) |
| ladder N=16 decode agg | 240.0* | 225.2* | 212.6* | 246.8* | +6.1% |
| ladder N=32 decode agg | 221.4* | 216.6* | 192.9* | 247.6* | +13.1% |
| preemptions N=8/16/32 | 20/74/164 | 21/63/155 | 19/69/195 | 24/73/176 | (pool-bound in every arm) |
| cohort C4 T=0 decode | 257.9 | 276.6 | 276.6 | 272.7 | +2.0% (A/A2 differ by 7.3%) |

B vs A: C1 T=0 +0.1%. N=4 -3.0%, which is inside A/A2's own spread on the other rows. The B schedule's
K=3 bucket starts at 9, and the scheduler never holds more than 5 resident requests.

## Verdict

- **B (schedule as specified): fail.** It cannot engage at `MAX_SEQS=8` with this pool. The resident set
  tops out at 5 (`run` column), so every step uses the K=7 bucket. N=16 and N=32 did not improve: -8.6% and
  -11.9% against mean(A, A2). Both are tail rows.
- **B2 (knee at 4): inconclusive-noise against the experiment's pass rule.** Its C1 holds (-0.2% on T=0).
  The rule's N=16/N=32 rows are tail measurements whose A-to-A2 spread is up to 13%, and N=16 (+6.1%) is
  below the 10% bar. The one steady-state row the ladder can produce, N=4, gave A 243.0, B2 261.6 and A2
  241.5. That is +8.0% with the B2 value above both A boots, and the mechanism is visible in the same rows:
  ms/pass falls from 56-57 to 43.2 while tok/step falls from 3.5 to 2.95. The cohort C4 rows (7% A/A2 spread)
  do not confirm it.

Noise floor, measured here: C1 T=0 0.0% boot to boot (109.4/109.4, and 109.5/109.4 run to run within A).
C1 T=default 7.6%. Ladder N=1 13%. Ladder tail rows N≥8 up to 13%.

## Memory

Free VRAM after boot, from host `nvidia-smi` (the engine logs no post-boot free figure; it logs
`Initial free memory 22.76 GiB` before the load): A 1,471 MiB, B 1,477 MiB, B2 1,477 MiB, A2 1,431 MiB. All
are at or above 1.3 GB. After the measurements, free VRAM read 921 MiB, and 423 MiB after the ladder. That is
allocator growth at runtime, and Shared Usage did not move. Shared Usage on the 3090's adapter
(`luid_..114dd`) peaked at 92.2 MiB in every boot, which equals the loaded-engine floor. `mem.txt` has no
sample above floor+16 MiB, so no spill was seen.

## Verified vs assumed

Verified: the engine received each schedule (boot log), the pool size, the counter traces, and every number
above (`raw-rb-*-2.log`, `raw-ladder-*.log`). Assumed, not verified: that the scheduler's per-step K followed
the B2 schedule exactly. I inferred it from the tok/step and ms/pass drop, not from a per-step log.
