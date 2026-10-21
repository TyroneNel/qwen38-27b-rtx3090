# Phase 5 review: findings and roadmap

Reviewer: Opus, high effort. Scope: `05-findings/{p1-speed-inference,p2-quants-heads,p3-other,roadmap}.md`. No git, no GPU. `gpu-agent-prompt.md`, `gpu-queue.md`, `LEDGER.md` and `harness/` were read but not edited.

## Verdict

**PASS WITH CORRECTIONS.**

- Every D19 label I traced is correct. No finding calls a speedup **measured** without second-agent raw files. The only **measured** numbers are the prior box baseline (814.3 tok/s mode A, PTH side finding) and CPU logic results.
- The top 5 and the quick wins hold by gain × confidence ÷ effort. No quick win is licence-gated (D11). Nothing is out of scope (D3: DeepInfer is cited as context only).
- Errors fixed:
  - F-P2-01 overstated its floor and missed that the full-head setting already ships.
  - F-P1-08 mixed two mechanisms and blamed the wrong doc.
  - F-P3-02 and roadmap Q6/decision 4 asked for a policy decision that is not needed.
  - F-P1-13/Q4 understated the PTH swing at C1.
  - Q2 is GPU-dependent and is now marked "adopt after bench-16".
  - Stale cites and stale GPU-hour totals.
- Of the 13 "Corrections noticed": 7 applied (one of them partly), 2 deferred to the GPU hand-off, 4 rejected.

## Claims checked (23)

| # | Finding | Claim | Source traced | Label | Result |
|---|---|---|---|---|---|
| 1 | F-P1-01 (top 5, Q1) | +12-22% D decode vs our checkout | `01-project/upstream-diff.md:28` (+12.2/+14.0%, 2 boots, native), `:126` (+21.3/+22.1%, 3 boots, WSL2) | measured (reported) | OK |
| 2 | F-P1-01 | 0 vs stock image; GSM8K 0.9507 vs 0.9522; pool −1.5% | `upstream-diff.md:126` | measured (reported) | OK |
| 3 | F-P1-01 | FF has 0 conflicts, 25 files | `upstream-diff.md:9,18` (+444/−140) | fact | OK |
| 4 | F-P1-01 | our checkout resolves mtp/long to fp8 FlashInfer, not FA2_FP8KV | `experiments/X-01a/run.out:3,75`; upstream `resolve_config.sh:105-116` | measured (CPU) | OK |
| 5 | F-P1-01 | box `.env` CTX=fast VISION=1 means the default never fires | `upstream-diff.md:129`; `gpu-agent-prompt.md:78`; `raw/prior-3090/baseline.md:10`; deepinfer-ref `ENVIRONMENT.md:27` | — | **Corrected**: true only for a boot that inherits the box `.env`. Every queue arm strips SPEC/CTX/PTH_FA/VISION, sets `VISION=0` and pins PTH_FA. It does not block bench-15. |
| 6 | F-P1-02 (top 5, Q2) | pool 68,605 → 116,440 (1.70×), N=8 preemptions 23 → 1, C1 111.2 vs 111.0 | `reviews/phase3-transferability.md:66-80` (cline S-2) | measured (reported) | OK. The queue cite `gpu-agent-prompt.md:778-780` is stale (bench-16 is now at :816). |
| 7 | F-P1-02 | N=8 row is pool-limited (236 tok/s, 19 preemptions) | `04-benchmarks/baseline.md:37,64` | measured | OK |
| 8 | F-P1-03 (top 5, Q3) | bands near-tie < 0.01, replay 0.01-0.25, real > 0.25; CPU 14/14 | D18 (`decisions.md:22`); `X-08/test_classify_divergence.py` (re-run: 14 passed) | measured (logic) | OK. The script default was 0.13; fixed to 0.25. |
| 9 | F-P1-03/F-P3-04 | `greedy_equal.py` near-tie 0.1 | `harness/greedy_equal.py:84`, `harness/README.md:85` | fact | OK (deferred, harness is out of scope) |
| 10 | F-P1-05 (top 5, N6) | EAGLE-4 as EAGLE-3 break-even 1.12-1.31× | `phase3-transferability.md:87`; arithmetic re-done, matches | estimated | OK |
| 11 | F-P1-05 | drafter 3.09B params, 6,178,152,384 B, HF `EAGLE-Corp/Qwen3.8-27B-EAGLE4` f58a598 | HF metadata (the brief's id redirects) | fact | OK |
| 12 | F-P1-06 (top 5, N3) | about +5% at N=3 only; 0-2% at N=4/8 | `reviews/phase4-cpu-experiments.md` (X-02a: a3 ≥ 0.404, tail 0.247-0.317, +4.9% N=3, +9.5% vs +7.7% back-test, −3.3% spec-5); `phase4-x02a-tailshape.out` | estimated | OK |
| 13 | F-P1-04 (N2) | spec-timing overhead ≤ 0.7%, 10-20 µs; 51/51 patches apply | `reviews/phase4-x17-timing.md` (READY FOR GPU, F1-F12); `X-17/README.md:193` | estimated | OK |
| 14 | F-P1-07 / L5 | H05 +1.7-3.1% B C1 | `phase3-transferability.md:119-124` | estimated | OK |
| 15 | F-P1-08 | mechanism; `architecture.md:134-140` is wrong | `02-eagle4/architecture.md:134-140`; EAGLE-4 `tree_drafter.py:186-207`; `phase3-transferability.md:150-153` | estimated | **Corrected**: the EAGLE-4 feature is a pre-verify confidence stop, not a skip after a position-0 reject. architecture.md is right; phase 3 called H06's *hypothesis* wrong. |
| 16 | F-P1-13 (Q4) | prevents a 9-22% wrong comparison | `baseline.md:54,60-61`; `raw/prior-3090/baseline.md:63-65` | measured (reported) | **Understated**: at C1 on the box, PTH_FA=0 read 63-75 vs about 100 tok/s, +33-58% (non-interleaved). 9-22% is the long-context range. |
| 17 | F-P2-01 (N4) | Chinese +100-127%; 6.4% coverage; 1.06 vs 2.23 tok/step; 35.5 vs 80.6 tok/s | `01-project/heads-drafters.md:151` → `docs/gotchas.md:1436-1472` (gotcha 61, #196, one user's 3090, CTX=long k=3) | measured (reported) endpoints, estimated switch | **Corrected**, see below |
| 18 | F-P2-02 (N5) | mode A fp8 814.3 tok/s e2e | `baseline.md:33` | measured | OK, see mode A below |
| 19 | F-P2-02 | fp8 drift 0.0360 vs int8 0.0207/0.0219 at 32k, 1.7× | `docs/long-context.md:466-476`; `01-project/quantisation.md:132` | measured (reported) | OK, but the source run is TP=2 2×3080 PTH, CTX=long. Applying it to mode A is an extrapolation (annotated). |
| 20 | F-P2-03 (L1) | −0.34 GiB before padding, ≈ +0.6 GiB net after | `phase3-transferability.md:90,103-109` (0.86 GiB; page padding) | estimated | OK (+0.5-0.6 GiB) |
| 21 | F-P2-10 | KV sharing GSM8K 0.94 → 0.56-0.64 for a 1.52× pool | `upstream-diff.md:86` (closed PR #96) | measured (reported) | OK |
| 22 | F-P3-02 (Q6) | 19/4/10 → 21/4/8 with `fallback.patch`; F5c at `backend.py:27`; conflicts with refuse | `X-16/README.md`; `phase4-cpu-experiments.md:141-147` | measured (CPU) | Numbers OK. **Policy claim wrong**: register() declines, launcher `check()` still refuses (`start_qwen.sh:244-249`). |
| 23 | F-P3-03 (Q5) | docs say NSTRONG 8 / AGREE 2, code 6/0 | `docs/optimizations.md:303-304` | fact | OK |

Also checked: X-15 (`resolve_config.sh:57`, review-annotated), the MTP fine-tune being flat (`drafter/README.md:29-37`), and the harness self-check (`harness/selfcheck.out`, now 43 ok and SELFCHECK PASS; F-P3-04's "18 ok" is stale because another reviewer is extending it).

## Specific points

**Mode A: 948 vs 814.3.** Both are right, for different configs.
- 948 e2e (about 1,035 decode) is the repo's native-Linux, single-card figure on vLLM 0.31 with batch defaults (`README.md:108`; `docs/multi-gpu.md:170-175`: KV fp8, INT8_ACT int8 on mlp, MAX_SEQS 64, MAX_LEN 150000, no spec, no prefix cache; 64 concurrent, 128/512). It is **measured (reported)**.
- 814.3 e2e (805.6-822.9; 904 decode by median TPOT) is the prior GPU-box run. That run was WSL2, stock sha-2cf50a0, at GPU_UTIL 0.93 for rounds 1-2 and 0.91 after (`baseline.md:33`). It is **measured** and is the A/B reference for the box.
- 942 is the stale vLLM 0.27.1 figure (`batch/README.md:18,21`) that the prior baseline compared against (−13.6%).
- `plan.md:93` is annotated. F-P2-02 already uses 814.3 correctly.

**Box `CTX=fast VISION=1`.** It does not block the mtp/long default in the benchmarks.
- Upstream's `mtp/long)` arm requires `VISION != 1`, `DRAFT_TOKENS` 3 and no `FA2_FP8KV`.
- The queue builds each arm from a base with SPEC/CTX/PTH_FA/VISION stripped (`gpu-agent-prompt.md:78`). Every arm_env sets `VISION=0` and pins PTH_FA.
- The prior baseline did the same ("the box `.env` minus VISION/CTX/SPEC"), so its mode D resolved PTH_FA=1.
- The box default only matters for everyday serving on that box and for user installs with `VISION=1`. Roadmap decision 3 is reduced to that.

**F-P2-01 (Chinese full head).**
- It is internal (I-10, gotcha 61), not from EAGLE-4. The only EAGLE-4 link is the T-01 factor head, which is D11-gated.
- The endpoints are one user's reported run.
- "+100%" has no derivation. The claim's own ≥ 2.0 tok/step gives 2.0 / 1.06 = +89%, and +127% is 80.6 / 35.5.
- More important: the full-head endpoint is already a documented setting. `MTP_DRAFT_VOCAB=0` falls back to the shared full lm_head (`patches/qwen3_5-mtp-draft-vocab.patch:8,40`), and `single-user/README.md:438` recommends it for Chinese. A Chinese-only deployment gets the gain today with no code.
- The coverage switch only adds value for mixed-language traffic on one boot, and that gain is not quantified. N4 stays in Next sprint, but its headline is now "+89-127% vs the default slice, already reachable via the env var".
- The memory risk is mostly answered: the fallback head is the shared copy (not verified at load).

## Ranking critique

- **Top 5** (H01, H04′, H08, H03, H02′ = F-P1-01, -02, -03, -05, -06): agreed.
  - H01 has the best gain × confidence (three same-direction reported runs, S effort).
  - H04′ is a capacity win with a measured (reported) C1 tie.
  - H08 is a gate.
  - H03 is an information gate whose speed value is 0 by itself.
  - H02′ is a narrow +5% at N=3 only.
- **Quick wins:**
  - Q1: agreed.
  - Q2: GPU-dependent. Its evidence is another agent's run, and its C1 claim is inside the 1-2% spread. It stays a quick win in effort terms, but adoption now waits on bench-16 (annotated).
  - Q3 and Q4 are prerequisites, not wins. That is fine as long as they are not counted as gains, and the roadmap says "gain: none".
  - Q5: docs only, agreed.
  - Q6: needs no policy decision (annotated in F-P3-02, Q6 and decision 4).
  - No quick win is licence-gated. L1, L3 and L5 carry D11 correctly.
- **Next sprint / larger bets:**
  - N4 is now framed as above. Its rank inside the tier holds because the GPU cost is small (1.6 h). Without the switch arm it can only re-confirm a known setting.
  - L2 (F-P2-07, 0-3%) and L3 (0-10%, low prior) are correctly last-tier.
  - L6 is correctly a dropped item kept for completeness.
- **Missing fields** (Priority / Impact / Evidence / Effort / Risks), not filled by this review:
  - Most gaps are on dropped or no-go items, where it matters little: F-P1-09, -10, -11, -12 (Risks; -11 and -12 also Effort), F-P2-09, -10, -11, F-P3-06.
  - Four actionable items lack Risks: **F-P2-07, F-P2-12, F-P3-04, F-P3-05**. F-P3-05 also lacks Evidence. The author should add them.

## Completeness

- H01-H16, including the unranked H05, all map to a finding.
- Every no-go has a finding and a roadmap "deferred" row:
  - tree verify;
  - the `+eagle.4` fork;
  - Hopper/Blackwell and E-23;
  - KV sharing;
  - from-scratch training;
  - the DFlash2 window retrain;
  - the cline k-bands.
- CPU experiments X-01a, X-02a, X-08, X-15, X-16 and X-17 are all represented, including the negative X-02a N=4/8 result and X-16 refuting H16.
- GPU rows bench-13 to bench-25 (X-01b, X-03, X-04′, X-02b′, X-06, X-09 to X-14, X-17) each appear in a tier row.
- The decisions for the human are complete. Two are now marked moot or reduced: 3 (box defaults) and 4 (import policy). Decision 5 is the real blocker list.

## Corrections noticed (roadmap.md 1-13)

| # | Verdict | Action |
|---|---|---|
| 1 | rejected | Already annotated in `transferability.md` (X-01a row). `brief.md` has no FA2_FP8KV claim. |
| 2 | applied (partly) | `X-08/classify_divergence.py:4,15-17,46,176` default 0.13 → 0.25. The tests re-ran: 14 passed. `X-08/README.md:43` annotated. `record_top2.py` has no `--ulp-band`. |
| 3 | deferred | `harness/greedy_equal.py:84` is out of edit scope. See the hand-off. |
| 4 | applied | `01-project/dflash-specdec.md:471` |
| 5 | applied | `02-eagle4/transferability.md:225`. The bar is `plan.md:36,53` plus the queue's ≥ 5-boot rule, not D16. |
| 6 | rejected | Already annotated (X-15 README `[review: :57]`). |
| 7 | applied | `upstream-diff.md:74,109`; `transferability.md` E-13 row. Line 28 was already annotated. |
| 8 | deferred | It is in `gpu-agent-prompt.md`. See the hand-off. |
| 9 | applied | `04-benchmarks/baseline.md:23`. The harness README makes no such claim. |
| 10 | applied | `00-meta/plan.md:79` |
| 11 | applied | `00-meta/plan.md:93` (mode A answer above) |
| 12 | rejected | `architecture.md:134-140` is correct. F-P1-08 was fixed instead. |
| 13 | rejected | Cosmetic. 297 is the default and 299-306 the refusal block; both point into the same block. |

## For the GPU hand-off

These are for the reviewer who owns `gpu-agent-prompt.md` / `gpu-queue.md` / `LEDGER.md` / `harness/`.

1. `harness/greedy_equal.py:84` `--near-tie` default is 0.1, and `harness/README.md:85` repeats it. D18 says 0.01. The queue should pass `--near-tie 0.01` explicitly, or fix the default.
2. The bench-23 gate (`gpu-agent-prompt.md`, bench-23a/23b, "0.90 x 3.75 = 3.375 … r < 1.12") uses an absolute tok/step bar. The reviewed X-03 bar is a same-run ratio: "within 10%" is r ≥ 0.90 × 1.12 = 1.008, with r < 1.12. The two disagree whenever DFlash2's same-run tok/step is not 3.36.
3. `gpu-agent-prompt.md` (return-format tail, about line 1187) says `record_top2.py` has a 0.13 `--ulp-band` default. It has no such argument (its args are label, `--sets`, `--max-tokens`, `--top-n`, `--passes`, `--doc-ctx`, `--corpus`, `--out`). `classify_divergence.py`'s default is now 0.25, so explicit `--ulp-band 0.25` remains correct but redundant.
4. `LEDGER.md:64` (X-06) already notes +3% vs +2%. Nothing to change there; `transferability.md:225` is now annotated.
5. Optional: the allow-list cite `start_qwen.sh:299-304` in the prompt and the queue. The block is 297 (default) and 299-306 (refusal).

## Every edit

Findings files:
- `05-findings/p1-speed-inference.md`:
  - line 10: win-bar attribution;
  - summary row F-P1-13: magnitude;
  - F-P1-01: box-defaults risk;
  - F-P1-02: stale queue cite and adopt-after-bench-16;
  - F-P1-03: ulp-band risk now fixed;
  - F-P1-08: mechanism and the architecture.md cite;
  - F-P1-13: C1 swing.
- `05-findings/p2-quants-heads.md`:
  - summary row F-P2-01;
  - F-P2-01: impact (+89% floor, `MTP_DRAFT_VOCAB=0` exists), memory risk, internal origin;
  - F-P2-02: drift provenance.
- `05-findings/p3-other.md`:
  - summary row F-P3-02;
  - F-P3-02: risk and status (consistent with refuse);
  - F-P3-04: ulp-band fixed, harness README claim.
- `05-findings/roadmap.md`:
  - line 7: GPU hours;
  - summary rows Q2, Q4, Q6, N4;
  - Q1: box defaults;
  - Q2: cite, adopt after bench-16;
  - Q3: what was done and what was handed off;
  - Q6: First and Kill;
  - N4: gain;
  - decisions 3 and 4;
  - "Corrections noticed": intro and items 1-13 marked applied/rejected/deferred.

Source docs (each with a `[review (phase 5): …]` mark):
- `00-meta/plan.md:79,93`
- `01-project/dflash-specdec.md:471`
- `01-project/upstream-diff.md:74,109`
- `02-eagle4/transferability.md`: X-06 row (225) and E-13 row (151)
- `04-benchmarks/baseline.md:23`
- `04-benchmarks/experiments/X-08/README.md:43`

Code:
- `04-benchmarks/experiments/X-08/classify_divergence.py`: `--ulp-band` default 0.13 → 0.25 in the usage line, the docstring, `classify_pair()` and argparse. The tests re-ran with `PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -p no:cacheprovider`: 14 passed.

New file:
- this review.
