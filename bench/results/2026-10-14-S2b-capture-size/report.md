# S2b: CUDA graph capture size 64 → 40 on B's base

RTX 3090, 250 W, container boot, image `hyperqwen:exp-f1a033e`.

## Arms and what the engine got

Base: S1's B, `KV_MEM=5315022028` (pre-cut 256 MiB, `knob-changes.txt`). A = the shipped `CG` (resolves to 64).
A1, A2 and A3 are the shared A boots in `../2026-10-14-S2a-dynamic-k/`. B = `CG=40` (`.env.B`, label Bcg1),
booted between A2 and A3. The boot logs confirm the narrower set:

| | `cudagraph_capture_sizes` (engine config) | graph line | free VRAM after boot | pool |
|---|---|---|---|---|
| A (A2) | `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]` | `took 0.13 GiB (driver free-memory delta 0.32 GiB)` | 1,617 MiB | 110,953 tokens |
| B (Bcg1) | `[1, 2, 4, 8, 16, 24, 32, 40]` | `took 0.12 GiB (driver free-memory delta 0.25 GiB)` | 1,687 MiB | 110,953 tokens |

Bcg1 took 311 s to become healthy, against 134-144 s for the A boots. It captured a graph set that was not in
the compile cache.

## Results

| metric | A1 | A2 | **Bcg1** | A3 | B vs mean(A2, A3) |
|---|---|---|---|---|---|
| C1 T=0 decode (run 2) | 114.9 | 115.3 | **120.5** | 116.3 | **+4.1%** (+4.7% vs mean(A1, A2)) |
| C1 T=0 tok/step | 3.34 | 3.34 | **3.48** | 3.34 | **+4.2%** |
| C1 T=0 ms per step (= tok/step ÷ decode) | 29.1 | 29.0 | 28.9 | 28.7 | -0.3% |
| C1 T=default decode / tok/step | 115.9 / 3.35 | 110.4 / 3.17 | 109.9 / 3.15 | 113.3 / 3.28 | inside A's spread (110.4-115.9) |
| N=4 decode agg / ms/pass | 235.4 / 56.6 | 241.6 / 56.3 | 241.0 / 56.4 | 247.1 / 56.0 | -1.4% / +0.4% |

**Verdict: inconclusive. The C1 gain is not a capture-size effect.** On the rule's letter, B clears it: C1 is
+4.1-4.7% (rule ≥3%) with the narrower set confirmed in the boot log. But all of it is tokens per step: 3.48
against 3.34 on every A boot, run 1 and run 2 alike. The time per step is unchanged (28.9 ms against 28.7-29.1
ms), and so are ms/pass at N=4 (56.4 vs 56.0-56.6) and the N=4 aggregate.

A capture-size change cannot change what the drafter proposes or what greedy accepts, unless the verify step
runs on different kernels and its numerics move the greedy trajectory on these 8 prompts. Speed alone cannot
move tok/step. So the +4% is an acceptance shift on this prompt set, not a faster step. Whether it is real on
other text needs a teacher-forced comparison on a frozen target (S3-style), which this session did not run.
Batch 1's +3.3% lead (no A-B-A) is not confirmed as a speed effect either. Batch 1's report did not record its
tok/step.

Graph memory: 0.13 → 0.12 GiB reported, driver delta 0.32 → 0.25 GiB, free VRAM after boot +70 MiB. The pool is
byte-pinned, so it does not grow.

## Memory

Shared Usage on the 3090 stayed at the 84.2-92.2 MiB floor on Bcg1 (`mem.txt`); no spill flagged. Peak dedicated
23,543.8 MiB.

## Verified vs assumed

Verified: the capture sizes, graph lines, and every number above from the raw logs and counter traces. Assumed,
not verified: the mechanism for the tok/step shift (different verify kernels changing the greedy trajectory). No
output text was kept to diff the trajectories.
