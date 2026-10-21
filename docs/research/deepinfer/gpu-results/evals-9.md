# evals-9: acceptance panel from DeepSpec's evaluation sets

| Field | Value |
|---|---|
| Date | 2026-10-21 12:00–12:40 SAST |
| Commit | in this result's commit: `bench/accept_panel.py`, `bench/accept_panel_data.py` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88; no spill |
| `.env` knobs | mode B: `SPEC=dflash2 PREFIX_CACHE=1` (k=7, lookup on), thinking off |
| Data | DeepSpec's own eval JSONLs at pinned commit `005e03b`, with DeepSpec's selection (first turn, seeded shuffle, cap), then the first 50 per set (aime25 has 30): 430 items. Local, uncommitted copy at `G:\dev\hq-results\deepinfer-20261020\evals-9\data\panel.jsonl` (+ sha256 meta) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\evals-9\B\panel.{txt,json}`, frozen targets in `…\B\targets\` |

## Commands

```bash
python bench/accept_panel_data.py                     # once, downloads and selects (stdlib)
venv/bin/python bench/accept_panel.py B_panel --data /shared/panel.jsonl --sampled --out /out/panel.json
```

- **forced**: labd_accept's teacher-forced method. A greedy 256-token target is captured once and replayed in 128-token chunks; dirty chunks are dropped.
- **sampled**: one free-running request per item at T=1.0 / top-p 1 (DeepSpec's plain sampling); acceptance from `/metrics`.

tok/step here equals DeepSpec's "acceptance length"; DeepSpec drops the +1 on at most one proposal per item.

## Numbers (B, k=7; tok/step with bootstrap 95% CI)

| Domain | n | forced tok/step | acc@0 / @3 | sampled T=1 tok/step | decode tok/s (forced) |
|---|---|---|---|---|---|
| humaneval | 50 | **7.40** [7.24, 7.53] | 0.987 / 0.909 | 7.09 | 239 |
| mbpp | 50 | 6.10 [5.85, 6.36] | 0.929 / 0.716 | 5.63 | 200 |
| gsm8k | 50 | 5.90 [5.75, 6.05] | 0.920 / 0.690 | 5.84 | 194 |
| math500 | 50 | 5.83 [5.67, 5.98] | 0.920 / 0.672 | 5.61 | 186 |
| livecodebench | 50 | 5.59 [5.36, 5.84] | 0.902 / 0.635 | 4.93 | 173 |
| aime25 | 30 | 5.51 [5.28, 5.72] | 0.892 / 0.634 | 5.15 | 175 |
| mt-bench | 50 | 4.12 [3.72, 4.54] | 0.823 / 0.378 | 3.87 | 132 |
| alpaca | 50 | 3.44 [3.21, 3.74] | 0.773 / 0.275 | 3.33 | 110 |
| arena-hard-v2 | 50 | **3.36** [3.05, 3.71] | 0.698 / 0.281 | 2.99 | 105 |
| **pooled** | 430 | **4.89** [4.76, 5.02] | 0.847 / 0.519 | 4.51 | 156 |
| macro (unweighted) | 430 | 5.25 [5.17, 5.34] | 0.871 / 0.577 | 4.94 | 168 |

Clean chunks: 702 of 791 (89%); arena-hard and livecodebench have the most dirty chunks, where greedy replay diverged within a chunk.

## Reading

- **Acceptance depends on the domain by a factor of 2.2.** Code (HumanEval 7.4 of a possible 8) and math (5.5–5.9) draft far better than open chat (alpaca 3.4, arena-hard 3.4). The repo's 8 real prompts (3.36 tok/step, baseline.md) and labd's long-context tasks (2.0–3.4, evals-4) sit at the chat end. **The headline B number therefore understates code and math traffic by up to ~2× in tok/step,** and decode speed follows (105 → 239 tok/s).
- **Sampling at T=1 costs 2–12% of tok/step per domain** (pooled 4.89 → 4.51, −8%). The loss is largest on livecodebench (−12%) and smallest on gsm8k (−1%).
- These are per-domain targets for the drafter lane. The thinking-on split (evals-4) and the int4 sampling gap (drafter-2) should be read against this panel rather than the 8 prompts.

## Verdict

INFO. The broader acceptance panel exists. It runs in 40 minutes on B and separates domains with tight CIs (±0.1–0.4 tok/step at n=50). The headline number sits at the hard end of the range.
