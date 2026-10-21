# evals-8: per-component GPU time on the serving path

| Field | Value |
|---|---|
| Date | 2026-10-21 12:42–13:10 SAST (B, D, D′, E at 20k); E at ~170k is in [e-profile.md](e-profile.md) |
| Commit | `bench/trace_breakdown.py` (committed with e-profile) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88 |
| `.env` knobs | per mode, plus `EXTRA_ARGS=--profiler-config {"profiler":"torch","torch_profiler_dir":"/out/prof","max_iterations":80,"ignore_frontend":true}` |
| Context | labd `qa` task, warm prefix: B 23k, D and D′ ~95k (`--ctx 85000`), E 23k (`--ctx 20000`) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\evals-8\{B,D,Dp,E20k}\prof\`, `breakdown.{txt,json}` |

## Method

The appendix proposed nsys, but this image has no nsys and ncu can't read the counters under WSL2 here (attn-ncu). vLLM's own torch profiler gives the same attribution:
1. A cold labd pass fills the prefix cache.
2. `/start_profile`, then a warm 256-token pass, then `/stop_profile`.
3. `bench/trace_breakdown.py` buckets every kernel by its launching Python frame (drafter, lm_head, sampler) or by name, per decode step.

Host timings inside a profiled run are inflated by the profiler (the client saw a few tok/s), so only GPU-side numbers are used: kernel time per step, and GPU idle within the step span.

## Numbers: ms per decode step (share of step wall)

| Category | B @23k (DFlash2, FA) | D @95k (MTP, pthfa) | D′ @95k (DFlash2, pthfa) | E @23k (DFlash2, KVarN) | E @170k (e-profile) |
|---|---|---|---|---|---|
| Linear GEMMs (Marlin) | **18.9 (58%)** | **20.1 (59%)** | **18.6 (45%)** | **17.8 (38%)** | 17.7 (12%) |
| Full attention | 4.0 (12%) | 4.8 (14%) | 11.6 (28%) | 14.2 (30%) | **117.7 (78%)** |
| Drafter (incl. its GEMMs, lookup) | 3.7 (11%) | 2.4 (7%) | 3.2 (8%) | 3.3 (7%) | 3.6 (2%) |
| Norms / elementwise | 2.1 | 2.0 | 2.1 | 2.2 | 2.2 |
| GDN (48 layers) | 1.6 | 1.0 | 1.6 | 1.3 | 1.3 |
| lm_head | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| Other | 0.2 | 0.1 | 0.1 | 2.8 (spiky; median 0.13) | 1.7 |
| **Step wall (mean)** | **32.7** | **34.2** | **41.1** | **47.1** | **150.8** |
| GPU idle inside the step | 1.3 (3.9%) | 2.9 (8.4%) | 3.1 (7.5%) | **4.6 (9.7%)** | 6.0 (4.0%) |
| Query tokens per step | 8 | 4 | 8 | 8 | 8 |

## Reading

- **The appendix's "7.9 ms per DFlash2 step" question:** at B's shape, the drafter is 3.7 ms per step (11%), of which 2.6 ms is its own int4 GEMMs. The remaining ~4 ms of overhead beyond target forward is norms, GDN, lm_head and idle.
- **At shallow depth the target's Marlin GEMMs dominate** (58–59% in B and D). They are flat in context and depend on weight bytes only, so they set the floor every mode shares: ~18–20 ms per step, about 50 tok/s per tok/step.
- **Attention is the variable part.** It is 12% in B and 14% in D (pthfa at 95k with 4 query tokens), rising to 28% in D′ (pthfa, 8 query tokens, the padded stair of runtime-2) and 30% in E at only 23k. In E it reaches 78% at 170k.
- **Host idle is small except in E.** It is 1.3 ms (4%) in B. E has 4.6 ms (10%) of GPU idle per step at 23k, consistent with KVarN's per-step metadata syncs (evals-2). Removing them is worth up to ~10% in E at shallow depth (inferred: idle time is an upper bound).
- runtime.md's "host time under 2%" holds for B-like steps on native Linux; here, under WSL2, it is 4% in B and 8–10% in the pthfa and KVarN modes.

## Verdict

INFO. Per-step GPU attribution now exists for every speculative mode. Marlin GEMMs (~18–20 ms) are the shared floor. Attention is the depth- and backend-dependent part (12% in B, 30% in E at 23k, 78% at 170k). Host idle is ≤ 10%, worst in E.
