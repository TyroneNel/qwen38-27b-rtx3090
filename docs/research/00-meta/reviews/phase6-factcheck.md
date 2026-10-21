# Phase 6 fact check: REPORT.md

**Verdict: PASS WITH CORRECTIONS**

Claims checked: about 130 (64 links, 12 checklist items, about 55 table and prose numbers). Errors: 2 HIGH, 3 MED, 2 LOW.

## Findings

| # | Sev | REPORT line | Problem | Evidence |
|---|---|---|---|---|
| 1 | HIGH | 40 | "reads about 33-58% low" has the direction wrong. 63-75 tok/s against about 100 is 25-37% low. The 33-58% is how much higher PTH_FA=1 reads. | `04-benchmarks/raw/prior-3090/baseline.md:65` (63-75 vs 95-100); `05-findings/p1-speed-inference.md` F-P1-13 review ("about +33-58%") |
| 2 | HIGH | 304 | "43 ok" is wrong. `selfcheck.out` has 18 lines starting `ok` (43 is a substring count of "ok" in e.g. "tokens"). `SELFCHECK PASS` is correct. | `04-benchmarks/harness/selfcheck.out` (18 `^ok`; `selfcheck.sh` has 18 `chk` calls); `05-findings/p3-other.md:77`; `00-meta/reviews/phase4-cpu-experiments.md:14,156` all say 18 |
| 3 | MED | 225 | The break-even 3.75-4.40 tok/step (1.117-1.31x) is the W4A16 V4 row (int4 own head, 6.75 / 12.37 ms draft). V3 is the bf16 E-20 path, which needs 5.92 / 8.37 (the next bullet, line 226). The two bullets contradict. | `02-eagle4/transferability.md:112-119`; `transferability-calc.out` break-even block |
| 4 | MED | 28 | "about 40 numbers" is not supported. The phase-2 re-derivation table has 24 rows, "24 of 24 agree, 0 disagree". | `00-meta/reviews/phase2-harness-baseline.md:50` |
| 5 | MED | 298 | "+7.7% measured" is a bare "measured". The +7.7% A/B is from the cline branch with no raw file in this repo, so under D19 it is measured (reported). | `04-benchmarks/experiments/X-02a/README.md:18,91` |
| 6 | LOW | 235 | 72 MiB x 16 = 1.1 GiB. About 1.2 GiB includes align mode's 2+k slots. | `02-eagle4/architecture.md:249` [review] |
| 7 | LOW | 432 | "From plan.md" is true for 5 rows. `plan.md` has no HF drafter row and no "serving base 0.31.0". The drafter name and commit come from `02-eagle4/overview.md:6,206`. | `00-meta/plan.md:5-13` |

## Checklist results

1. Links: all 64 relative link targets exist. The `#5-eagle-4-lessons` anchor matches `## 5. EAGLE-4 lessons` (line 185). PASS.
2. Drafter name: the evidence supports `EAGLE-Corp/Qwen3.8-27B-EAGLE4` @ `f58a598`, 6,178,152,384 B.
   - Sources: `02-eagle4/overview.md:6,206-208`; `training-recipe.md:60` (the brief's `EAGLE-corp/EAGLE4-Qwen3.8-27B` redirects, sha `f58a5985`); `gpu-agent-prompt.md:555,885-891`; `phase5-findings.md:34`.
   - `architecture.md` does not name the repo, and `plan.md` has no drafter row (finding 7).
   - One doc still uses the brief's name: `03-vllm/integration-paths.md:233`. It says the weights were not inspected, so it is a harmless redirect alias. `reviews/phase1-eagle4-training-calc.py:10` also uses it, in a URL.
   - The docs do not otherwise disagree. The report is right.
3. Open question 5: PASS.
   - bench-2 (`gpu-agent-prompt.md:290`) and bench-7 (`:393`) pass `--near-tie 0.01`.
   - `greedy_equal.py:84` default is 0.1.
   - `classify_divergence.py:4,46` `--ulp-band` default is 0.25.
   - X-08 README reports 14/14 (`test.out`: "ALL 14 PASS").
4. Self-check: see finding 2.
5. Box hours: PASS. P1 = 3.6+2.4+4.2+2.8+3.75+2.7 = 19.45. P2 = 1.6+3.0+0+3.0 = 7.6. Total 27.05 (about 26.3 GPU-h). Conditional = 11.7, upper bound 38.75 (about 39). X-14 5.2 GPU-h; X-13 2.5 GPU-h and 41 GB (`gpu-agent-prompt.md:1155-1179`).
6. Counts: PASS.
   - 45 queue rows.
   - 31 findings (13 P1 / 12 P2 / 6 P3).
   - Roadmap 6+6+6; corrections 7 applied (one "applied, partly") / 2 deferred / 4 rejected = 13.
   - 7 phase-1 reviews, all PASS WITH CORRECTIONS.
   - X-16 19/4/10 -> 21/4/8; X-17 51/51 plus 4 KVarN; LEDGER has 28 rows, all `todo`.
7. Baseline table: all numbers match `baseline.md` and `raw/prior-3090/baseline.md`, `evals-10.md`, `spec-5.md`, `e-profile.md`. That covers A/B/C/D/D′/E headlines, KV pools, aggregates, B acceptance, quality, E profile and D with PTH_FA=0. Native figures 942/127.1/381/100.3/67 match (948 is the README.md:108 value, annotated in plan.md:93). The 6-14% gap and the win-bar rules match.
8. Top-5 and roadmap: gains and labels match `p1`, `p2`, `roadmap.md` summary tables, including the [review] values (N4, Q2, Q4, Q6).
9. Memory gate: V1 5.75, V3 3.33, V4 0.86, V7 0.25, +4.56/67k, +2.13/27k, -0.34, padding 20,480 -> 1,835,008, 16.4k tokens per GiB all match. The 3.75-4.40 and 1.117-1.31x numbers match too, but see finding 3 for the label.
10. Pinned commits: PASS. The first five rows match `plan.md:9-13`. The stock image matches `plan.md:20`. The drafter row is absent from plan.md (finding 7).
11. LEDGER: no row other than `todo`. PASS.
12. Spot checks: X-02a numbers (+9.5%, -3.3%, +4.9%, fitted table), D15/D17/D18 text, phase verdicts, paper numbers, 74 GB `hidden.npy` and X-17 0.4-0.7% all match.

## Fix pairs

1. old: `reads about 33-58% low at C1 on the box (measured (reported) magnitude, F-P1-13)`
   new: `reads about 25-37% low at C1 on the box (63-75 against about 100 tok/s, so the pinned run reads 33-58% higher; measured (reported) magnitude, F-P1-13)`
2. old: `Harness self-check: 43 ok,`
   new: `Harness self-check: 18 ok,`
3. old: `a V3 drafter needs 3.75-4.40 tok/step`
   new: `a W4A16 V4 drafter chain needs 3.75-4.40 tok/step`
4. old: `Phase 2 re-derived about 40 numbers with 0 disagreements`
   new: `Phase 2 re-derived 24 baseline rows (several numbers each) with 0 disagreements`
5. old: `(+9.5% predicted vs +7.7% measured)`
   new: `(+9.5% predicted vs +7.7% measured (reported))`
6. old: `about 1.2 GiB for 16 rows, estimated`
   new: `about 1.1 GiB for 16 rows, about 1.2 GiB with the 2+k slots, estimated`
7. old: `From [00-meta/plan.md](00-meta/plan.md). All sources were opened read-only.`
   new: `From [00-meta/plan.md](00-meta/plan.md); the drafter row is from [02-eagle4/overview.md](02-eagle4/overview.md). All sources were opened read-only.`
