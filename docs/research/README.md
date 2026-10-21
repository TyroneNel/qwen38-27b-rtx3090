# EAGLE-4 research expedition

Can ideas from EAGLE-4 make Qwen3.8-27B (W4A16) faster or better on one RTX 3090? Phases 0-6 ran on a CPU-only research box. Every GPU measurement is packaged for the RTX 3090 box and has not run yet.

**Start here: [REPORT.md](REPORT.md).** It has the top 5 recommendations, the findings, the roadmap and the decisions that need the human.

## Evidence labels

Every number carries one of three labels ([D19](00-meta/decisions.md)):

- **measured**: the raw files are in this repo, and a second agent re-derived the number.
- **measured (reported)**: a real GPU run whose raw files we do not hold (upstream PRs, fork branches, the EAGLE-4 paper).
- **estimated**: our own arithmetic.

No speedup from this expedition is measured yet.

## Layout

| Folder | Contents |
|---|---|
| [00-meta/](00-meta/) | the [brief](00-meta/brief.md), [plan and ranked hypotheses](00-meta/plan.md), [decisions D1-D21](00-meta/decisions.md), the [agent log](00-meta/agent-log.md), the [GPU queue](00-meta/gpu-queue.md), and [reviews/](00-meta/reviews/) (one independent review per phase output) |
| [01-project/](01-project/) | a map of our project: [inventory](01-project/inventory.md), [architecture and modes](01-project/architecture.md), [DFlash and spec decode](01-project/dflash-specdec.md), [quantisation](01-project/quantisation.md), [heads and drafters](01-project/heads-drafters.md), and the [upstream diff](01-project/upstream-diff.md) |
| [02-eagle4/](02-eagle4/) | EAGLE-4: [overview and licence](02-eagle4/overview.md), [architecture](02-eagle4/architecture.md), [training recipe](02-eagle4/training-recipe.md), [build and kernels](02-eagle4/build-and-kernels.md), and [transferability to the 3090](02-eagle4/transferability.md) with its calculator |
| [03-vllm/](03-vllm/) | the [vLLM integration paths](03-vllm/integration-paths.md) for spec decode, quant, attention and KV |
| [04-benchmarks/](04-benchmarks/) | the [baseline](04-benchmarks/baseline.md) re-derived from [prior 3090 runs](04-benchmarks/raw/prior-3090/), the [harness](04-benchmarks/harness/), the CPU [experiments](04-benchmarks/experiments/) X-01a, X-02a, X-08, X-15, X-16 and X-17, the [GPU agent prompt](04-benchmarks/gpu-agent-prompt.md), and the [ledger](04-benchmarks/LEDGER.md) |
| [05-findings/](05-findings/) | findings by priority: [P1 speed](05-findings/p1-speed-inference.md), [P2 quants and heads](05-findings/p2-quants-heads.md), [P3 other](05-findings/p3-other.md), and the [roadmap](05-findings/roadmap.md) |

## Running the GPU queue (RTX 3090 box)

Steps (decision D17):

1. On the GPU box, fetch the `research/eagle4-expedition` branch from `origin`.
2. Give an agent [04-benchmarks/gpu-agent-prompt.md](04-benchmarks/gpu-agent-prompt.md).

The prompt holds everything the agent needs:

- the step-0 checks (D13), which record the commit;
- the box facts and rules;
- the bench-1 to bench-25 commands.

How the agent works:

- It runs [00-meta/gpu-queue.md](00-meta/gpu-queue.md) in order, one GPU job at a time. Each row is at most 2 h.
- It records status and result paths in [04-benchmarks/LEDGER.md](04-benchmarks/LEDGER.md).

Rules it follows:

- Never modify the stock image or `patches/series`. The timing patch lives only in a variant image (D12).
- Pin `PTH_FA` in every arm (D10).
- Use the D15 spill and abort rule.

Cost: about 27 box-hours expected, 39 upper bound (estimated).

A speedup becomes **measured** only after a second agent re-runs it from the raw results.
