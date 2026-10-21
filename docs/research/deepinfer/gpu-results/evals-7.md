# evals-7: golden greedy canary keyed by (GPU, pin, mode)

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | `bench/canary.py`, added in this result's commit |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock; one shared compile/autotune cache volume for every boot |
| GPU | RTX 3090, 250 W, driver 610.88 |
| `.env` knobs | B: `SPEC=dflash2 PREFIX_CACHE=1`; D: `SPEC=mtp CTX=long` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\evals-7\B_fresh{1,2}\`, `…\probes\<mode>\canary*.txt`, `…\evals-4\{B,D}\canary.txt`, golden file `…\shared\canary_golden.json` |

## Commands

```bash
venv/bin/python bench/canary.py --mode B_fresh --golden /shared/canary_golden.json --refresh   # boot 1, first traffic after warmup.sh
venv/bin/python bench/canary.py --mode B_fresh --golden /shared/canary_golden.json             # boot 2, first traffic after warmup.sh; then once more in the same boot
```

8 prompts × 256 greedy tokens at C1; ids and top-1 logprobs are compared exactly (`--lp-tol 0`).

## Numbers

| Protocol | Result |
|---|---|
| Fresh boot 2 against fresh boot 1, canary as the first traffic after `warmup.sh` | **MATCH 8/8, max Δlogp 0.000** |
| Same boot, canary repeated | MATCH 8/8 (both boots) |
| Golden written after a probe battery (sync check, KL probe, …); compared in another boot after a 20-minute ladder/acceptance session (B) | INSPECT: 3/8 token mismatches (first at token 33, 63, 201); the other 5 differ in logprobs by up to 0.11 |
| Same, mode D (MTP) | INSPECT: 3/8 mismatches (first at token 13, 77, 155) |

The decode KL probe in the probe boots gave the same reading: two back-to-back runs in one boot were bit-identical in every mode.

## Reading

- **The stack is bit-reproducible across boots under a fixed protocol.** Same image, GPU and compile/autotune cache, with the canary as the first traffic after warmup: identical ids and logprobs.
- **What breaks it is what ran earlier in the same boot,** not the boot itself. After a long session of other traffic, greedy text diverges within 13–201 tokens, in MTP as well as DFlash2. So it is not only the DFlash2 lookup controller (kvarn-varlen's hypothesis H1). The likely common cause is engine state that the traffic history shapes, for example runtime Triton autotune picks on shapes first seen mid-session. That is inferred, not isolated.
- **So the canary works as a gate only in this protocol:** fresh boot, `warmup.sh`, canary first, exact compare. Within that protocol a mismatch means the bits changed (kernel, pin, driver or cache). Run in the middle of a session it needs `--lp-tol` and reads "inspect".

## Verdict

INFO. The canary is exact and reproducible across boots when it runs first; it drifts after unrelated traffic in the same boot. Every equivalence check in this expedition (spec-1, drafter-1a) therefore uses fresh boots and puts its check first.
