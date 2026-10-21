# GPU queue

One GPU job at a time. The research box has no GPU (decisions D1), so every slot below is a request for the RTX 3090 box. The GPU agent works the queue in order and fills in the result path.

| # | Requested by | Job | Granted | Status | Result path |
|---|---|---|---|---|---|
| bench-1 | bench-engineer | env snapshot (container + WSL host), pinned PTH_FA recorded | | queued for GPU box | docs/research/04-benchmarks/results/env-*/ |
| bench-2 | bench-engineer | harness smoke run on mode B (REPS=1; confirms /metrics names, nvidia-smi parse) | | queued for GPU box | docs/research/04-benchmarks/results/smoke-*/ |
| bench-3 | bench-engineer | prior-timing-check: read-only grep of the box's bind-mounted bench/labd_bench.py for perf_counter | | queued for GPU box | 04-benchmarks/LEDGER.md (row prior-timing-check) |
| bench-4 | bench-engineer | baseline re-run mode B (SPEC=dflash2 PTH_FA=0), full suite | | queued for GPU box | docs/research/04-benchmarks/results/base-B-*/ |
| bench-5 | bench-engineer | baseline re-run mode D (SPEC=mtp CTX=long PTH_FA=1), full suite to 64k | | queued for GPU box | docs/research/04-benchmarks/results/base-D-*/ |
| bench-6 | bench-engineer | baseline re-run mode E (SPEC=dflash2 CTX=huge), full suite to 120k | | queued for GPU box | docs/research/04-benchmarks/results/base-E-*/ |
| bench-7 | bench-engineer | greedy T=0 equality, SPEC=off vs dflash2 vs mtp | | queued for GPU box | docs/research/04-benchmarks/results/greedy-*/ |
| bench-8 | bench-engineer | quality guard on stock B (ppl and GSM8K vs evals-3 reference) | | queued for GPU box | docs/research/04-benchmarks/results/quality-*/ |
| bench-9 | bench-engineer | PTH_FA=0 vs 1 interleaved, mtp and dflash2 CTX=long | | queued for GPU box | docs/research/04-benchmarks/results/pthfa-ab-*/ |
| bench-10 | bench-engineer | profile boot of B for draft vs verify time | | queued for GPU box | docs/research/04-benchmarks/results/profile-dv-*/ |
| bench-11 | bench-engineer | baseline re-run modes D', C, A (lower priority, after bench-4 to 6) | | queued for GPU box | docs/research/04-benchmarks/results/base-{Dp,C,A}-*/ |
| bench-12 | reviewer (DFlash) | `SPEC=off` fast-variant C1 decode run (same prompts as bench-4) to settle speculation overhead per step (estimated 4.5–6.5 ms) | | queued for GPU box | docs/research/04-benchmarks/results/specoff-C1-*/ |
