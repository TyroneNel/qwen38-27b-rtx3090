# GPU queue

One GPU job at a time, in the table order (bench-11a runs between bench-8 and bench-9, bench-11b between bench-10 and bench-12, then bench-13 to bench-25 in number order and parts in letter order: the phase-4 experiments, each row at most 2.0 h and a resumable part (`xpart.sh` in the prompt; the verdict reads across the parts of one experiment), whose gating (X-11 on X-03, X-13 on X-03 near break-even, blocked rows without tooling or inputs) is in `04-benchmarks/gpu-agent-prompt.md` "Experiments"). The research box has no GPU (decisions D1), so every slot below is a request for the RTX 3090 box. The GPU agent works the queue in order and fills in the result path.

| # | Requested by | Job | Granted | Status | Result path |
|---|---|---|---|---|---|
| bench-1 | bench-engineer | env snapshot (container + WSL host), pinned PTH_FA recorded | | queued for GPU box | docs/research/04-benchmarks/results/env-*/ |
| bench-2 | bench-engineer | harness smoke run on mode B (SMOKE=1 REPS=1; confirms /metrics names, nvidia-smi parse) | | queued for GPU box | docs/research/04-benchmarks/results/smoke-*/ |
| bench-3 | bench-engineer | prior-timing-check: read-only grep of the box's bind-mounted bench/labd_bench.py for perf_counter | | queued for GPU box | 04-benchmarks/LEDGER.md (row prior-timing-check) |
| bench-4 | bench-engineer | baseline re-run mode B (SPEC=dflash2 PTH_FA=0), full suite, two boots (D16) | | queued for GPU box | docs/research/04-benchmarks/results/base-B-*/ |
| bench-5 | bench-engineer | baseline re-run mode D (SPEC=mtp CTX=long PTH_FA=1), full suite to 64k, two boots (D16) | | queued for GPU box | docs/research/04-benchmarks/results/base-D-*/ |
| bench-6 | bench-engineer | baseline re-run mode E (SPEC=dflash2 CTX=huge), full suite to 120k, two boots (D16) | | queued for GPU box | docs/research/04-benchmarks/results/base-E-*/ |
| bench-7 | bench-engineer | greedy T=0 equality, SPEC=off vs dflash2 vs mtp | | queued for GPU box | docs/research/04-benchmarks/results/greedy-*/ |
| bench-8 | bench-engineer | quality guard on stock B (ppl and GSM8K vs evals-3 reference) | | queued for GPU box | docs/research/04-benchmarks/results/quality-*/ |
| bench-11a | bench-engineer | baseline re-run mode D' (SPEC=dflash2 CTX=long PTH_FA=1), two boots; queued ahead of bench-9 because bench-9 is gated on base-Dp (LEDGER.md) | | queued for GPU box | docs/research/04-benchmarks/results/base-Dp-*/ |
| bench-9 | bench-engineer | PTH_FA=0 vs 1 interleaved, mtp and dflash2 CTX=long | | queued for GPU box | docs/research/04-benchmarks/results/pthfa-ab-*/ |
| bench-10 | bench-engineer | profile boot of B for draft vs verify time | | queued for GPU box | docs/research/04-benchmarks/results/profile-dv-*/ |
| bench-11b | bench-engineer | baseline re-run modes C and A, two boots each (lower priority, after bench-4 to 6; runs after bench-10) | | queued for GPU box | docs/research/04-benchmarks/results/base-{C,A}-*/ |
| bench-12 | reviewer (DFlash) | `SPEC=off` fast-variant C1 decode run (same prompts as bench-4) to settle speculation overhead per step (estimated 4.5–6.5 ms) | | queued for GPU box | docs/research/04-benchmarks/results/specoff-C1-*/ |
| bench-13a | bench-engineer | X-17 part 1 (1.25 h): build variant image `hyperqwen:sha-2cf50a0-spectiming` (rule 2 lifted), boots S and U | | queued for GPU box | docs/research/04-benchmarks/results/X-17-*/13a/ |
| bench-13b | bench-engineer | X-17 part 2 (1.6 h): boots T, T, S, then overhead gate; runs iff 13a done. Gate PASS writes x17.ok | | queued for GPU box | docs/research/04-benchmarks/results/X-17-*/13b/ |
| bench-13c | bench-engineer | X-17 part 3 (0.7 h): draft/verify timing boot on D; runs iff x17.ok | | queued for GPU box | docs/research/04-benchmarks/results/X-17-*/13c/ |
| bench-14a | bench-engineer | X-08 part 1 (1.0-1.4 h): near-tie classifier, NS / DF2 / MTP boots recorded twice (replay boot iff exit 3); add-on to bench-7 | | queued for GPU box | docs/research/04-benchmarks/results/X-08-*/14a/ |
| bench-14b | bench-engineer | X-08 part 2 (1.2 h): KVarN adaptive + prefix case, 2 boots | | queued for GPU box | docs/research/04-benchmarks/results/X-08-*/14b/ |
| bench-15a | bench-engineer | X-01b round 1 (1.4 h): `SPEC=mtp CTX=long` PTH_FA=0 then 1 at `GPU_UTIL=0.89`, labd about 66k and 93k prompt tokens (nominal `--ctx 58000`, `82000`) plus C1 | | queued for GPU box | docs/research/04-benchmarks/results/X-01b-*/15a/ |
| bench-15b | bench-engineer | X-01b round 2 (1.4 h); runs iff 15a done and valid | | queued for GPU box | docs/research/04-benchmarks/results/X-01b-*/15b/ |
| bench-15c | bench-engineer | X-01b round 3 (1.4 h) plus ratio gate over 3 rounds; runs iff 15b done and valid | | queued for GPU box | docs/research/04-benchmarks/results/X-01b-*/15c/ |
| bench-15d | bench-engineer | X-01b GSM8K, one boot per arm (1.5 h); conditional: runs iff the 15c decode gate (ratio >= 1.10 at both cells, tok/step within 2%) is met | | queued for GPU box | docs/research/04-benchmarks/results/X-01b-*/15d/ |
| bench-16a | bench-engineer | X-04': int8 per-token-head KV via TRITON vs fp8 on mode B, boots A X (1.4 h) | | queued for GPU box | docs/research/04-benchmarks/results/X-04p-*/16a/ |
| bench-16b | bench-engineer | X-04': boots X A (1.4 h) plus pool and C1 gate; runs iff 16a done | | queued for GPU box | docs/research/04-benchmarks/results/X-04p-*/16b/ |
| bench-16c | bench-engineer | X-04': GSM8K quality boot of X (0.7 h); conditional: runs iff pool >= 1.6x and C1 within 2% after 16b | | queued for GPU box | docs/research/04-benchmarks/results/X-04p-*/16c/ |
| bench-17a | bench-engineer | X-03 part 1 (1.0 h): download EAGLE-4 drafter (6.2 GB, rule 5 lifted for this model; evaluation-only licence D11) with disk check, SHA record, conversion to EAGLE-3 layout; load-only pre-gate | | queued for GPU box | docs/research/04-benchmarks/results/X-03-*/17a/ |
| bench-17b | bench-engineer | X-03 part 2 (1.85 h): V3 and DFlash2 (lookup on) boots, T=0 and T=0.7; runs iff 17a pre-gate passed; fixes the X-03 verdict | | queued for GPU box | docs/research/04-benchmarks/results/X-03-*/17b/ |
| bench-17c | bench-engineer | X-03 part 3 (0.9 h): DFlash2 lookup-off boot (INFO) | | queued for GPU box | docs/research/04-benchmarks/results/X-03-*/17c/ |
| bench-18a | bench-engineer | X-02b': fitted DYNAMIC_K table vs existing at N=3 with N=1 guard, rounds 1-2 (1.8 h) | | queued for GPU box | docs/research/04-benchmarks/results/X-02b-*/18a/ |
| bench-18b | bench-engineer | X-02b': round 3 (0.9 h); runs iff 18a done and valid | | queued for GPU box | docs/research/04-benchmarks/results/X-02b-*/18b/ |
| bench-18c | bench-engineer | X-02b': rounds 4-5 (1.8 h); conditional: runs iff the median N=3 gain over rounds 1-3 is +3% to below +5% | | queued for GPU box | docs/research/04-benchmarks/results/X-02b-*/18c/ |
| bench-19 | bench-engineer | X-09: MTP full head vs 40,960-row slice on Chinese, 4 boots S F S F (1.6 h); blocked unless `zh_prompts.jsonl` is supplied | | queued for GPU box | docs/research/04-benchmarks/results/X-09-*/19/ |
| bench-20a | bench-engineer | X-10 part 1 (1.8 h): mode A fp8 (F), int8 per-token-head (I) and bf16 drift reference (R) boots | | queued for GPU box | docs/research/04-benchmarks/results/X-10-*/20a/ |
| bench-20b | bench-engineer | X-10 part 2 (1.2 h): boots I and F; runs iff 20a done and valid; verdict reads both parts | | queued for GPU box | docs/research/04-benchmarks/results/X-10-*/20b/ |
| bench-21 | bench-engineer | X-11: W4A16 V4 drafter vs bf16 V3, 2 boots (1.5 h); gated on X-03 17b WIN and V4 weights supplied | | queued for GPU box | docs/research/04-benchmarks/results/X-11-*/21/ |
| bench-22 | bench-engineer | X-12: SVD-init rank-128 recall@128 on lm_head (CPU only, 0.7 h, on the box with the weights and hidden states) | | queued for GPU box | docs/research/04-benchmarks/results/X-12-*/22/ |
| bench-23a | bench-engineer | X-13 pilot (0.9 h): capture 250k tokens and train the EAGLE-4 body for one epoch (hard 1.5 h timeout); gated on X-03 within 10% of break-even and tooling under `experiments/X-13/` | | queued for GPU box | docs/research/04-benchmarks/results/X-13-*/23a/ |
| bench-23b | bench-engineer | X-13 evaluation (1.5 h): adapted V3A vs fresh DFlash2 boot; runs iff 23a finished | | queued for GPU box | docs/research/04-benchmarks/results/X-13-*/23b/ |
| bench-24a | bench-engineer | X-14 pilot (0.6 h): train MTP depth 2 for 0.5 epoch (about 110 steps, 3.6M tokens, 1 h timeout) and export; blocked without `hidden.npy` and the model dir | | queued for GPU box | docs/research/04-benchmarks/results/X-14-*/24a/ |
| bench-24b | bench-engineer | X-14 pilot (0.9 h): train MTP depth 4, same budget (1.5 h timeout), export; runs iff 24a done | | queued for GPU box | docs/research/04-benchmarks/results/X-14-*/24b/ |
| bench-24c | bench-engineer | X-14 serving (1.5 h): boots T2, T4 and stock D reference on mode D | | queued for GPU box | docs/research/04-benchmarks/results/X-14-*/24c/ |
| bench-24d | bench-engineer | X-14 second serving pair (1.0 h); conditional: runs iff the T4/T2 tok/step ratio after 24c is 1.00 to 1.05 | | queued for GPU box | docs/research/04-benchmarks/results/X-14-*/24d/ |
| bench-25a | bench-engineer | X-06 stage 1 (0.3 h): conditional-node capture probe; blocked unless `experiments/X-06/` tooling exists; failure stops 25b and 25c | | queued for GPU box | docs/research/04-benchmarks/results/X-06-*/25a/ |
| bench-25b | bench-engineer | X-06 stage 2 (1.5 h): boots S P S (stock D, patched D, stock D); runs iff 25a passed | | queued for GPU box | docs/research/04-benchmarks/results/X-06-*/25b/ |
| bench-25c | bench-engineer | X-06 stage 2 (1.0 h): boots P S; runs iff 25b valid; verdict over all five boots | | queued for GPU box | docs/research/04-benchmarks/results/X-06-*/25c/ |
