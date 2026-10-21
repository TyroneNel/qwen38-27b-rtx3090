# quant-1: per-128 int8 activation scales, route A (Humming)

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Evidence | [quant-1-calib.md](quant-1-calib.md): `bench/act_calib.py --modes all` and `bench/humming_probe.py` on the stock image, RTX 3090 at 250 W, driver 610.88 |
| Gates | quant-1-calib done (quality gate passes); evals-1 done |

## Decision: no end-to-end serving run

The appendix's win criterion for route A needs both of these:
- `run_benchmarks.sh` at 64 concurrency ≥ 942 tok/s with all layers on Humming;
- Humming prefill within 5% of Marlin INT8_ACT.

The kernel probe already fixes the decode side. A 64-stream decode step runs the linears at M ≤ 64, and there Humming int8 g128 is 2.5–3.8× slower than the Marlin W4A8-int8 kernel mode A uses today (0.26× / 0.41× / 0.40× at M = 1 / 16 / 64, geomean over the model's six linear shapes). Mode A's step is ~60% linear GEMMs (evals-8), so an all-Humming A would lose well over half its throughput **(inferred)**. A serving boot would also need two fixes first:
- a patch to stop `check_and_fallback_input_schema` from silently turning int8 g128 into W4A16;
- an `LD_LIBRARY_PATH` fix for Humming's NVRTC.

Judgement call: the result can't come out positive, so no boot was spent on it.

| | Humming int8 g128 vs Marlin W4A8-int8 |
|---|---|
| M = 1 / 16 / 64 (decode) | 0.26× / 0.41× / 0.40× |
| M = 256 / 2048 (prefill) | 1.19× / 1.19× |
| As shipped (`--linear-backend humming`) | resolves to W4A16 (bf16 activations): no int8 at all |

## Verdict

**LOSS** for route A in mode A. Humming is 2.5–3.8× slower than the current Marlin int8 path at decode sizes. Prefill alone gains +19% at the kernel level, short of the +27–30% the appendix hoped for, and it would need two vLLM-side fixes.

The quality case for per-128 scales stands: down_proj and GDN in_proj errors fall 5–6×. That case moves to quant-4, which gets the same scales on the unchanged Marlin kernel (its gate, "route A blocked or slower than Marlin by > 5%", now passes).
