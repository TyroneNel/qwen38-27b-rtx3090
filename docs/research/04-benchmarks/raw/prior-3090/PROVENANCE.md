# raw/prior-3090: provenance

Verbatim copies of the prior RTX 3090 measured results. Nothing here was edited, re-run or re-computed by this lane.

| Field | Value |
|---|---|
| Source branch | `origin/research/deepinfer` |
| Source commit | `86ce6ed` |
| Local export used for the copy | `~/research-src/deepinfer-ref/` (read-only) |
| Measured on | the RTX 3090 box described in `ENVIRONMENT.md` (250 W, driver 610.88, WSL2 + Docker Desktop, stock image `sha-2cf50a0`), 2026-10-20 to 2026-10-21 |
| Copied on | 2026-10-21, by the bench-engineer lane |

| File here | Original path (in the source commit) | sha256 (first 16) |
|---|---|---|
| ENVIRONMENT.md | `docs/research/deepinfer/gpu-results/ENVIRONMENT.md` | aab220d4988fb851 |
| baseline.md | `docs/research/deepinfer/gpu-results/baseline.md` | c65323581d8c1b77 |
| spec-5.md | `docs/research/deepinfer/gpu-results/spec-5.md` | 99190b8ec71fdc52 |
| e-profile.md | `docs/research/deepinfer/gpu-results/e-profile.md` | 8d861e3e299570b9 |
| evals-3.md | `docs/research/deepinfer/gpu-results/evals-3.md` | bcfde86280c0d3a7 |
| evals-4.md | `docs/research/deepinfer/gpu-results/evals-4.md` | d1521bd26b3cb6ee |
| evals-10.md | `docs/research/deepinfer/gpu-results/evals-10.md` | 70744b72cbd32dfa |
| evals-11.md | `docs/research/deepinfer/gpu-results/evals-11.md` | 58de9bb33b0ce5ec |
| LEDGER.md | `docs/research/deepinfer/gpu-results/LEDGER.md` | d5aebd88b7b78d69 |
| gpu-agent-prompt.md | `docs/research/deepinfer/gpu-agent-prompt.md` | f2b4aca8fe03b3a7 |

Verify: `sha256sum <file> | cut -c1-16`.

## What is NOT here

The raw per-run data (`ladder.jsonl`, `gpu.csv`, `gpumem.csv`, container logs, `summary.txt`, `breakdown.json`, profiler traces) were never committed. They stayed on the box at `G:\dev\hq-results\deepinfer-20261020\<item-id>\` (paths are given in each file's "Raw data" row). The numbers in `../../baseline.md` therefore trace to these write-ups, not to the underlying jsonl. If those directories still exist, copying them here would upgrade the evidence (open question in `../../LEDGER.md`).
