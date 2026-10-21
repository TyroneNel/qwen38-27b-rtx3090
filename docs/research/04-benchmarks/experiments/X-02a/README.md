# X-02a: fitted k-per-concurrency table vs DYNAMIC_K (H02′, E-13/E-04)

CPU only. Files: `fit.py` (stdlib, assert self-check; `python3 -I fit.py > fit.out`), `fit.out`, `inputs.md` (every input with source and M/E label), `raw/` (verbatim copies of the source logs). Every number below is **estimated (E)** unless it is quoted as an input (M, see `inputs.md`).

## Hypothesis

H02′: a table fitted by maximising `E(k) / (V(N,k) + k·D(N))`, with a native-OFF band (k=0), beats the existing DYNAMIC_K table `[[1,3,7],[4,512,3]]` (`single-user/start_qwen.sh:678,692`) at the same boot by >=5% aggregate at N=4 or N=8.

## Method

- `V(N,k)`: measured ms/pass cells at k in {0,1,3,7} and N in {1,2,4,8} (batch 3 Q1 for k>=1, batch 4 Q2-off for k=0), linear between cells in k and in N. Tail rows (not all N streams decoding) are excluded; N=8 k=7 and N>=16 k>=1 are extrapolated.
- `D`: 0. DFlash proposes a fixed 7-token block in one pass and ms/pass already includes drafting, so `k·D` would double count. D>0 shown separately: it does not move N=4/8.
- `E(k)`: two profiles. (1) spec-5 per-position `a_i` as specified (T=0.7). (2) ladder-matched: pooled measured tok/step at k=1/3/7 from boots other than the A/B being reproduced (1.811 / 2.862 / 3.508), increments split by spec-5's shape.
- Aggregate = `1000·N·E/V·c`, c = 0.99 (measured ragged-batch factor).

## (a) Back-test: PASS with ladder-matched acceptance, FAIL with spec-5 as specified

Target: DYNAMIC_K gain at N=4, +7.7–9.2% reported (raw A/B: 241.4 → 260.0 = +7.7%). Pass window = reported range ±3 pts = [4.7, 12.2]. V(4,3) is held out: it is interpolated from the measured (4,1) and (4,7) cells.

| acceptance | V(4,3) predicted (measured: Q1 42.3, A/B 43.15) | E3/E7 | predicted gain | result |
|---|---|---|---|---|
| spec-5 per-position `a_i` (as specified) | 42.5 ms | 0.721 | **−3.3%** | **FAIL** (11–12 pts low) |
| ladder-matched | 42.5 ms | 0.816 | **+9.5%** | **PASS** |
| ladder-matched, V from Q2 (16-seat boot) | 44.0 ms (meas. 42.2) | 0.816 | +11.9% | PASS |

The cost model is fine (V(4,3) lands within 0.5–2% of three measured cells). The miss is the acceptance vector: spec-5 is T=0.7 with 8 short prompts, while the DYNAMIC_K gain was measured at T=0 on 4k-token prompts [review: temperature is not the cause: spec-5's own T=0 row (`raw/prior-3090/spec-5.md:32`) has tok/step 3.486 (ladder 3.508) and pos-0 0.695 (ladder 0.81), so the gap is prompts/config, not T], where acceptance is front-loaded (a0 0.81 vs 0.69; tail positions 3–6 sum to 0.65 vs 1.0). Even with the A/B's own measured V ratio, spec-5 `a_i` predicts −5.4%. So **spec-5's vector cannot be used to fit a table for the ladder workload**; the ladder-matched profile is used below. The README step curve (25.9/32.8/49.1) is from a faster, older boot (0.86–0.89x of the A/B boot); its scale cancels in the ratio, so it is a consistency check only. Caveat: the ladder-matched profile reproduces the gain partly by construction (E3/E7 is its input); the independent part is V(4,3). [review: the back-test is mostly a consistency check: the k=3 vs k=7 gain depends only on E3/E7 and V3/V7, so how the tail mass is split over positions 3–6 is untested by it.]

## (b) Fitted table vs the DYNAMIC_K table (ladder-matched; `fit.out` has spec-5 too)

| N | fitted k | DYNAMIC_K k | fitted vs DYNAMIC_K |
|---|---|---|---|
| 1 | 7 | 7 | 0.0% |
| 2 | 5 | 7 | +0.8% (noise level) |
| 3 | 4 | 7 | **+4.9%** (N=3 never measured; V interpolated between N=2 and 4) |
| 4 | 3 | 3 | 0.0% |
| 5–8 | 3 | 3 | 0.0% |
| 12, 16 | 3 | 3 | 0.0% (extrapolated) |

Fitted DYNAMIC_K: `[[1,1,7],[2,2,5],[3,3,4],[4,512,3]]`. The existing table already sits at the k=3 optimum for N>=4. The only real difference is moving the k=7→3 boundary down from N=4 to N=2–3. With spec-5 acceptance the optimum shifts to k=5 at N=4–5 (+4.6% / +2.7%), but that profile failed the back-test.

**Native-OFF (k=0) never wins** at N<=16 (432 grid cases, 0 argmax; with V0 x0.8 it appears at N>=15, extrapolated). At N=8 the model gives k=3 at 365 vs off 251 tok/s (measured: 350.1 / 337.1 vs 250.7). The measured N=16 OFF win (369 vs a 300 tail row) is a seat effect: state pages are sized for the boot's `num_speculative_tokens`, and a per-step k cannot buy seats.

## (c) Predicted aggregates, tok/s (E; DFLASH_TOKENS=7 boot, matched-boot base)

| N | fixed k=7 | DYNAMIC_K | fitted | fitted vs DYNAMIC_K | measured cells for comparison (M) |
|---|---|---|---|---|---|
| 1 | 117 | 117 (k=7) | 117 (k=7) | 0.0% | 118.8 (A/B), 113.9 / 105.4 (Q1/Q2) |
| 2 | 180 | 180 (k=7) | 182 (k=5) | +0.8% | 179.6 / 167.4 k=7, 175.9 / 170.0 k=3 |
| 3 | 218 | 218 (k=7) | 228 (k=4) | +4.9% | none |
| 4 | 243 | 267 (k=3) | 267 (k=3) | 0.0% | k=7 239–241; k=3 260–268 |
| 8 | 303 | 365 (k=3) | 365 (k=3) | 0.0% | k=3 boot 337–350; k=7 boot: 5 resident, tail 177–194, no steady state |

N=8 on a `DFLASH_TOKENS=7` boot is seat-limited whichever table runs (`MAX_SEQS=8` holds 5–7 residents at 4k prompts), so the N=8 row is a compute-only figure; both tables would pick k=3 and would tie. On a `DFLASH_TOKENS=3` boot both tables reduce to k<=3, and the fit picks k=3 (365 vs off 251). Fitted vs fixed k=7 is +9.9% at N=4 and +20% at N=8 (compute-only), well below the old "+41% / +91%", which were never a k-band effect (review `phase3-transferability.md`).

## Sensitivity (`fit.out`)

Grid: 6 acceptance profiles x V base (Q1/Q2) x k-shape between measured cells (linear/convex/concave) x V0 x0.8/1.0/1.3 (the cost of k=0 inside a spec boot is unmeasured). "Consistent" = acceptance profiles that pass the back-test.

| | N=2 | N=3 | N=4 | N=8 |
|---|---|---|---|---|
| median gain over DYNAMIC_K, consistent profiles | +4.7% | +10.8% | 0.0% | 0.0% |
| max gain, consistent profiles | +11.4% | +18.1% | **+5.7%** | **+4.9%** |
| max gain, all profiles incl. spec-5 | +13.2% | +19.9% | +12.9% | +9.7% |

- N=4/8 only reach the bar at a corner (a convex V between k=3 and 7, i.e. a cost stair, plus a tail 10% heavier than measured: fitted k=4). The median is 0.
- Decisive unmeasured quantity: marginal value of positions 4–6 at N=4. The fit needs the tail mass `E7−E3` to be >=1.5x the measured 0.65 before k>=5 pays (+2.1%), and 2x for +8.4%. The measured cell-to-cell spread of E7 (3.36–3.67) is about ±25% of that tail.
- Not an H02′ point but the largest, most robust difference: N=3 (and N=2 at k=5). It rests on interpolation across N, so a stair between N=2 and N=4 could erase it.
- A native-OFF band needs a V0 more than 20% below the measured SPEC=off cost and N>=15, and a seat budget a k=7 boot does not have.

## Verdict on H02′

**Not plausible at N=4 or N=8.** Base case 0.0% at both (E) [review: the exact 0.0% follows from the cliff in spec-5's shape (a3/a2 = 0.45). With a smooth tail (constant conditional acceptance from pos 3) the fitted table gains 0 to +2.2% at N=4 and ≈0% at N=8 for slopes ≥ 6.8 ms/k (+0.5–6.3% only if V(8,k>3) is much flatter than measured at N=4); `00-meta/reviews/phase4-x02a-tailshape.out`, estimated. Report 0–2% (E), not 0.0%]; the best back-test-consistent corner of the sensitivity grid is +5.7% at N=4 and +4.9% at N=8, and the median is 0. DYNAMIC_K already sits at the cost-model optimum for N>=4, and native-OFF never wins within seat limits. Expected gain over DYNAMIC_K at the H02′ points: 0–2% (E), consistent with the review's 0–5%. The only credible improvement, about +5% at N=3 (E), lies outside the N=1/2/4/8 bar. Demote H02′ as the review already did; X-02b′ is a cheap confirmation, not a likely win.

## X-02b′ (GPU box)

Config: `SPEC=dflash2 CTX=fast PREFIX_CACHE=1 DFLASH_TOKENS=7 PTH_FA=0` (PTH_FA=1 refuses DYNAMIC_K, `start_qwen.sh:324-327`), `MAX_SEQS=8`, `bench/conc_ladder.py --n 1,2,3,4,8 --salt <fixed> --reps 6 --out 1024`, ABBA boots, same image for all arms. Arms (env values, single-quoted):

| arm | `DYNAMIC_K` | purpose |
|---|---|---|
| A (baseline, same boot as the others) | `'[[1,3,7],[4,512,3]]'` | the comparator |
| B (fitted) | `'[[1,1,7],[2,2,5],[3,3,4],[4,512,3]]'` | the table to try |
| C (boundary only) | `'[[1,2,7],[3,512,3]]'` | isolates the N=3 effect cheaply |

Predictions (E, tok/s): N=3 A 218, B 228, C 228; N=2 A 180, B 182; N=1, 4, 8 identical across arms. Decision rule: H02′ wins only if B beats A by >=5% at N=4 or N=8 with C1 within 2%; expect N=4 and N=8 to tie, and N=3 to gain ~5%. Optional extra: one boot at `DYNAMIC_K='[[1,3,7],[4,512,5]]'` to measure k=5 directly at N=4 (the spec-5 profile predicts +4.6%, the ladder profile predicts a loss). Each distinct k adds a CUDA graph set at boot (`vllm/v1/worker/gpu/cudagraph_utils.py:289`), so check boot time and memory. k=0 inside a spec boot is untested (legal per `dynamic/utils.py:52-55`); do not include it in X-02b′ since the model never selects it.

## Open questions

1. Per-position acceptance at T=0 and at T>0 on the ladder prompts (spec-5 only has T=0.7 on short prompts); the gain at N=4 depends on temperature.
2. V(N=3, k) and V(N, 4..6) are interpolated, never measured.
3. Context-length dependence: all cells are 4k prompts (EAGLE-4 keys costs per 1024-token bucket); long context raises per-position cost and would lower the optimal k.
4. The +8.0% / +9.2% runs behind the "+7.7–9.2%" range have no raw data in this tree (only the +7.7% salted run does).

Re-run by: phase-4 reviewer, 2026-10-21, CONFIRMED WITH CORRECTIONS: `fit.py` re-run, output identical to `fit.out`; comparator `[[1,3,7],[4,512,3]]` matches `start_qwen.sh:678,692`. Corrections: temperature attribution, smooth-tail sensitivity (0–2%, not 0.0%), back-test is a consistency check. Recommendation: reword X-02b′ to an N=3 boundary check plus per-position acceptance capture.
