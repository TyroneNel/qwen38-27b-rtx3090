# Agent log

Every spawn. Type is the agent type actually used (see decisions D5).

| # | Phase | Agent | Type used | Model | Effort | Task | Output | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | code-mapper | general-purpose | Sonnet | high | Inventory + architecture map | 01-project/inventory.md, architecture.md | done |
| 2 | 1 | specdec-analyst | general-purpose | Opus | high | DFlash / spec-decode deep-dive | 01-project/dflash-specdec.md | done |
| 3 | 1 | quant-analyst | general-purpose | Opus | high | Quants, memory budget, heads/drafters | 01-project/quantisation.md, heads-drafters.md | done |
| 4 | 1 | upstream-differ | general-purpose | Sonnet | high | HyperQwen vs fork | 01-project/upstream-diff.md | done |
| 5 | 1 | eagle-analyst A | general-purpose | Opus | high | EAGLE-4 architecture, inference, build, licence | 02-eagle4/overview.md, architecture.md, build-and-kernels.md | done |
| 6 | 1 | eagle-analyst B | general-purpose | Opus | high | EAGLE-4 training recipe and data | 02-eagle4/training-recipe.md | done |
| 7 | 1 | vllm-tracer | general-purpose | Sonnet | high | vLLM spec-decode/quant/attention/KV paths | 03-vllm/integration-paths.md | done |
| 8 | 2 | bench-engineer | general-purpose | Sonnet | high | Harness, prior-run baseline, GPU ledger | 04-benchmarks/harness/, baseline.md, LEDGER.md, raw/prior-3090/ | done |
| 9 | 1 | reviewer (upstream diff) | general-purpose | Opus | high | Verify upstream-diff.md | reviews/phase1-upstream-diff.md | done: PASS WITH CORRECTIONS (12 checked, 1 wrong, 3 partly off) |
| 10 | 1 | reviewer (EAGLE-4 training) | general-purpose | Opus | high | Verify training-recipe.md | reviews/phase1-eagle4-training.md | done: PASS WITH CORRECTIONS (15 checked, 1 wrong, 3 fixed) |
| 11 | 1 | reviewer (project map) | general-purpose | Opus | high | Verify inventory.md + architecture.md | reviews/phase1-project-map.md | done: PASS WITH CORRECTIONS (46 checked, 20 wrong — mostly cites — all fixed) |
| 12 | 2 | reviewer (harness + baseline) | general-purpose | Opus | high | Re-derive baseline, re-run self-check, audit harness | reviews/phase2-harness-baseline.md | done: PASS WITH CORRECTIONS (~40 numbers re-derived, 0 disagree; 2 medium bugs fixed) |
| 13 | 1 | reviewer (DFlash) | general-purpose | Opus | high | Verify dflash-specdec.md | reviews/phase1-dflash-specdec.md | done: PASS WITH CORRECTIONS (24 checked, 5 wrong, 1 unsupported, 1 circular) |
| 14 | 1 | reviewer (EAGLE-4 architecture) | general-purpose | Opus | high | Verify overview/architecture/build-and-kernels | reviews/phase1-eagle4-architecture.md | done: PASS WITH CORRECTIONS (30 checked, 7 wrong, 1 unsupported) |
| 15 | 1 | reviewer (quant + heads) | general-purpose | Opus | high | Verify quantisation.md + heads-drafters.md | reviews/phase1-quant-heads.md | done: PASS WITH CORRECTIONS (34 checked, 2 wrong, 1 unsupported) |
| 16 | 1 | reviewer (vLLM paths) | general-purpose | Opus | high | Verify integration-paths.md | reviews/phase1-vllm-paths.md | done: PASS WITH CORRECTIONS (28 checked, 8 wrong: 5 anchors, 3 substantive) |
| 8b | 2 | bench-engineer (resumed) | general-purpose | Sonnet | high | Apply reviewer recommendations: WDDM spill as VRAM source, greedy in smoke, GPU index from env | 04-benchmarks/harness/ | done (selfcheck PASS; spill parser assumes MiB in last field, smoke run must confirm) |
