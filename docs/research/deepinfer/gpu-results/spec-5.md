# spec-5: expected-acceptance telemetry Σ min(p, q)

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | in this result's commit: `docs/research/deepinfer/experiments/patches/spec5-accept-telemetry.patch`, `bench/spec5_accept.py`, `bench/test_spec5_telemetry.py`, `experiments/docker/Dockerfile.patch` |
| Image | `hq:exp-spec5` = stock `ghcr.io/syv-ai/hyperqwen:latest` + `spec5-accept-telemetry.patch` (inert unless `SPEC5_TELEMETRY_DIR` is set) |
| GPU | RTX 3090, 250 W, driver 610.88; busy median SM clock 1035 MHz; no spill |
| `.env` knobs | `SPEC=dflash2 PREFIX_CACHE=1` (mode B), `SPEC5_TELEMETRY_DIR=/tmp/spec5` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\spec-5\B\spec5_*.{txt,json}` |

## Why a server patch

The client never sees the draft distribution q, the rejected draft token, or that token's target probability, so Σ min(p, q) cannot be computed from `/v1` responses. The patch adds one Triton kernel after vLLM's MRv2 rejection kernel. Per draft position it records the overlap Σₓ min(p, q), the survival product ∏ min(1, p(d)/q(d)) (the expected accept count given the drafts), the drafts and the realised accepts. It copies them to a JSON file without a sync, one step behind. It does nothing unless `SPEC5_TELEMETRY_DIR` is set.

## Commands

```bash
docker build -f docs/research/deepinfer/experiments/docker/Dockerfile.patch --build-arg PATCHES=spec5-accept-telemetry.patch -t hq:exp-spec5 .
# CPU test (Triton interpreter, in the image): SPEC5TEST PASS cases=5 patch=applies
venv/bin/python bench/test_spec5_telemetry.py
# in the container (needs the telemetry file)
venv/bin/python bench/spec5_accept.py B_T0 --temperature 0 --limit 8 --max-tokens 256
venv/bin/python bench/spec5_accept.py B_T07 --temperature 0.7 --limit 8 --max-tokens 512
venv/bin/python bench/spec5_accept.py B_T07think --temperature 0.7 --think --limit 8 --max-tokens 1024
```

## Numbers

| Arm | Steps | tok/step, realised | tok/step, survival (Rao-Blackwellised) | Σ min(p,q) at pos 0 | SE across prompts, realised / survival | Variance ratio |
|---|---|---|---|---|---|---|
| T=0, thinking off | 597 | 3.486 | **3.486** (identity holds) | 0.695 | 0.496 / 0.496 | 1.00 |
| T=0.7, thinking off | 1,117 | 3.611 | 3.597 | 0.690 | 0.450 / 0.444 | 1.03 |
| T=0.7, thinking on | 2,464 | 3.341 | 3.368 | 0.742 | 0.236 / 0.228 | 1.07 |

Per position at T=0.7 (thinking off):

| pos | realised accept | survival | realised given the previous was accepted | Σ min(p,q) |
|---|---|---|---|---|
| 0 | 0.687 | 0.687 | 0.687 | 0.690 |
| 1 | 0.511 | 0.513 | 0.744 | 0.618 |
| 2 | 0.404 | 0.403 | 0.790 | 0.577 |
| 3 | 0.325 | 0.320 | 0.805 | 0.542 |
| 6 | 0.190 | 0.186 | 0.841 | 0.502 |

The `/metrics` counters agree with the file (590 vs 597 steps at T=0; the difference is one step at each edge of the window).

## Reading

- **The telemetry is correct.** At T=0, survival equals the realised count exactly, which is the patch's built-in self-check. At T>0 the two agree to within 0.4%.
- **The variance claim does not hold at the level that matters for A/Bs.** The appendix expected the Rao-Blackwellised count to make 1–4% effects measurable. Across prompts the standard error drops by only 1–7% (variance ratio 1.03–1.07). Acceptance varies far more from prompt to prompt (1.58 to 5.55 tok/step at T=0.7) than from the acceptance coin within a prompt. More prompts or a paired design (drafter-2's offline tau, evals-9's panel) reduce the noise; this estimator alone does not.
- **What it does add is a per-position picture without coin noise.** Conditional acceptance rises along the block (0.69 → 0.84), while Σ min(p, q) falls (0.69 → 0.50). Later positions are reached only on easy text, and on that text the drafter is nearly always right. That is useful input for spec-1/spec-2's confidence models.

## Verdict

INFO. The telemetry works and is exact at T=0, but it buys only a 3–7% variance reduction across prompts, not the lower-variance A/B instrument the appendix hoped for. Keep the patch as an experiment; don't add it to `patches/series` for that purpose.
