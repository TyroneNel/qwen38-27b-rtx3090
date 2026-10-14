# S2c: is the cohort C8/C16 row a measurement on B's base?

RTX 3090, 250 W, container boot, image `hyperqwen:exp-f1a033e`. Run on the A3 boot of
`../2026-10-14-S2a-dynamic-k/`, which is S1's B base at `KV_MEM=5315022028` (pool 110,953 tokens), `MAX_SEQS=8`.
Tool: `../2026-10-14-batch2-tools/cohort_gate.py`. It runs the `run_benchmarks.sh` cohort row (custom dataset
`bench/prompts_real.jsonl`, 1024 output tokens, T=0) three times per concurrency in one boot, and diffs
`vllm:num_preemptions_total` and samples peak `vllm:num_requests_running` around each run. Raw logs:
`raw-cohort-A3.log`, `raw-cohort-A3/`.

| row | rep 1 | rep 2 | rep 3 | within-boot spread | preemptions per rep | peak running | tok/step |
|---|---|---|---|---|---|---|---|
| C8, decode(C/meanTPOT) | 346.9 | 347.4 | 346.9 | **0.1%** | 4 / 4 / 4 | 7 / 7 / 7 | 3.36 |
| C16, decode(C/meanTPOT) | 664.5 | 708.3 | 694.7 | **6.6%** | 11 / 9 / 10 | 7 / 7 / 7 | 3.46-3.55 |

**Verdict.**

- **C8: passes the spread gate (0.1%), and is quotable as a repeatable number on this base, with a caveat.**
  Every repetition preempts exactly 4 times and peaks at 7 running requests, so the row is a stable but
  pool-limited state, not 8 streams decoding together. On the brief's playbook rule ("preemptions in a ladder
  row → tail measurement"), it is not a steady-state C8 number. Quote it as "C8 cohort, 347 tok/s, 4
  preemptions" or not at all. For comparison, batch 1's 37% spread and S1's A-arm bimodality (250 / 250 / 351)
  are gone on the bigger pool. S1's B arm read 361.7 / 362.6 / 372.1 at the full 5,583,457,484-byte pin.
- **C16: not a measurement, whatever its spread.** The 6.6% is under the 10% bar, but the metric is ill-defined
  here. `decode = C × 1000 / meanTPOT` assumes 16 concurrent streams, and `MAX_SEQS=8` caps the server at 8 (peak
  7). The 664-708 is the 8-stream TPOT multiplied by 16, double-counting the queue. The row also needs 16 prompts
  from an 8-prompt file: `vllm bench serve` logs `Oversampled requests to reach 16 total samples`, so it carries
  repeated prompts that share prefix-cache blocks. Stop calling C16 a decode measurement on the single-user
  profile.

## Memory

The A3 boot's Shared Usage stayed at the 84.2-92.2 MiB floor through all six cohort runs
(`../2026-10-14-S2a-dynamic-k/mem.txt`); no spill flagged. Free VRAM after boot 1,617 MiB.

## Verified vs assumed

Verified: every number above from the raw logs and the server's counters. Assumed, not verified: that the 4
preemptions per C8 rep are the state-page limit (`start_qwen.sh`'s resident note), not the token pool. Peak
`kv%` was not sampled by this tool.
