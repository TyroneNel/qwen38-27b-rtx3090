# Roadmap

Built from the findings in `p1-speed-inference.md` (F-P1-xx), `p2-quants-heads.md` (F-P2-xx) and `p3-other.md` (F-P3-xx). Labels follow D19:
- **measured (reported)**: someone else's GPU run;
- **estimated**: our arithmetic.

No speedup is **measured** yet. Queue rows are bench-N base numbers in `00-meta/gpu-queue.md`, which are being split into ≤ 2 h parts. The GPU-hour estimate is P1 21.2 h and P2 8.1 h, ≈29 h in total with an upper bound ≈38 h (**estimated**, `04-benchmarks/gpu-agent-prompt.md:1082`). [review (phase 5): the current hour table reads P1 19.45 h, P2 7.6 h, about 27 h expected (GPU-hours about 26.3), 11.7 h conditional and gated, upper bound about 39 h (`gpu-agent-prompt.md`, budget table after bench-25). That file is still being edited; take the totals from it.]

Within each tier, items are ordered by gain × confidence ÷ effort. Licence-gated items (D11) are never in the quick-win tier.

## Summary

| Tier | Item | Findings | Expected gain (label) | Rows |
|---|---|---|---|---|
| Quick win | Q1 Fast-forward to upstream/main (done, D22) | F-P1-01, F-P1-13, F-P3-04 | +12-22% D decode at 66-93k vs the old checkout `95ac251` (measured (reported)) | bench-15, bench-9, bench-5 |
| Quick win | Q2 int8 TRITON KV on mode B | F-P1-02, F-P2-12 | pool 1.70×, N=8 preemptions 23→1, C1 tie (measured (reported)) [review: adopt after bench-16, not before] | bench-16 |
| Quick win | Q3 Correctness gate | F-P1-03, F-P3-04 | gate, no speed | bench-7, bench-14 |
| Quick win | Q4 Benchmark-validity lifts | F-P1-13 | prevents a 9-22% (long context) to ≈33-58% (C1 on the box) wrong comparison (measured (reported) magnitude) [review] | none (code); bench-2 smoke |
| Quick win | Q5 Doc drift fixes | F-P3-01, F-P3-03 | docs only | none |
| Quick win | Q6 F5c import guard [review: no policy choice needed; consistent with "refuse"] | F-P3-02 | robustness only | none |
| Next sprint | N1 Re-baseline on the post-FF tree | F-P1-13, F-P3-05, F-P1-12 | reference arms; TTFT and peak VRAM | bench-1 to bench-12 |
| Next sprint | N2 Spec-timing variant | F-P1-04 | tool; ≤0.7% overhead (estimated) | bench-13 (+ bench-10, bench-12) |
| Next sprint | N3 Fitted k table at N=3 | F-P1-06 | about +5% at N=3 only (estimated) | bench-18 |
| Next sprint | N4 Chinese full-head switch | F-P2-01 | Chinese +89-127% vs the default slice (estimated from measured (reported) endpoints) [review: already reachable with `MTP_DRAFT_VOCAB=0`; the switch only adds value for mixed traffic, unquantified] | bench-19 |
| Next sprint | N5 Mode A int8 KV | F-P2-02 | drift 1.7× lower (measured (reported)); speed ±3% (estimated) | bench-20 |
| Next sprint | N6 EAGLE-4 as EAGLE-3 probe | F-P1-05, F-P2-04, F-P2-08 | 0 by itself; information gate (estimated break-even 1.12-1.31×) | bench-17 |
| Larger bet | L1 W4A16 EAGLE-4 drafter | F-P2-03 | servability; likely net memory loss after padding (estimated) | bench-21 |
| Larger bet | L2 MTP TTT unroll 2→4 | F-P2-07 | 0-3% D tok/step (estimated) | bench-24 |
| Larger bet | L3 Body-only adaptation | F-P2-06 | 0-10% tok/step (estimated) | bench-23 |
| Larger bet | L4 DFlash2 window retrain | F-P2-08 | up to the −12% at 12-36k back (measured (reported) loss) | none queued |
| Larger bet | L5 SVD factors, then low-rank candidate head | F-P2-05, F-P1-07 | +1.7-3.1% B C1 (estimated) | bench-22, then X-05 (not queued) |
| Larger bet | L6 Device-side depth stop | F-P1-08 | ≤1-2% D C1 (estimated) | bench-25 |

## Tier 1: quick wins

These are small and reversible, and need no GPU or only one confirmation run.

**Q1 Fast-forward to `upstream/main` (63f7581). Done (D22).**
- **Findings:** F-P1-01, F-P1-13, F-P3-04.
- **Gain:** +12-22% D decode at 66-93k vs the old checkout `95ac251` (**measured (reported)**, 3 runs, same direction). 0 vs the stock image. It also brings `perf_counter` timing and the mtp/long launcher test.
- **First:** the human approved the fast-forward, and it ran `--ff-only` with 0 conflicts (D22; `01-project/upstream-diff.md:9`). Run `bash pthfa/install.sh` on the box. Mode-D arms set `VISION=0` and pin `PTH_FA`, because the box `.env` sets `CTX=fast VISION=1` and the default never fires there. [review (phase 5): the queue already does this for every arm (`gpu-agent-prompt.md:78`, each arm_env), as the prior baseline did, so the box defaults do not block bench-15. They matter only for everyday use of the box.]
- **Rows:** bench-15 (X-01b, 3 interleaved boots per arm at `GPU_UTIL=0.89`), with bench-9 and bench-5 as the long-context baselines.
- **Kill:** ratio < 1.10 at either cell keeps the FF but drops the speed claim from docs. Any of these pins `PTH_FA=0` on mtp/long: ratio < 1.00, a GSM8K drop > 1 pt, or a boot refusal at `MAX_LEN` 150000.

**Q2 int8 per-token-head KV (TRITON) on mode B.**
- **Findings:** F-P1-02, F-P2-12.
- **Gain:** pool 1.70×, N=8 preemptions 23→1, C1 111.2 vs 111.0 (**measured (reported)**, cline S-2). The N=8 aggregate gain is unquantified, but our N=8 row is pool-limited.
- **First:** adopt PR 359 so PPL stays comparable. Then run arm X from `gpu-agent-prompt.md:778-780` [review (phase 5): now bench-16a/16b/16c, `gpu-agent-prompt.md:816` ff.] (`EXTRA_ARGS` only, no launcher edit). [review (phase 5): this is a quick win to apply only after bench-16 passes. The 1.70× and 23→1 come from another agent's run; the C1 tie is inside the 1-2% spread.]
- **Rows:** bench-16.
- **Kill:** pool < 1.6×, or N=8 preemptions > 1, or C1 outside ±2%, or GSM8K down > 1 pt.

**Q3 Correctness gate (X-08).**
- **Findings:** F-P1-03, F-P3-04.
- **Gain:** none. It is the gate for every speed claim.
- **First:** set the `--ulp-band` default to 0.25 and `greedy_equal.py` near-tie to 0.01 (D18). [review (phase 5): `classify_divergence.py` default set to 0.25 (14 tests pass). `record_top2.py` has no `--ulp-band`. `harness/greedy_equal.py:84` (near-tie 0.1) is outside this review's edit scope; it is in the GPU hand-off list of `00-meta/reviews/phase5-findings.md`.]
- **Rows:** bench-7, then bench-14.
- **Kill:** if the KVarN adaptive + prefix case is not flagged `real`, or a non-near-tie divergence does not reproduce on the same boot, redesign the classifier and hold all speed claims.

**Q4 Benchmark-validity lifts.**
- **Findings:** F-P1-13.
- **Gain:** none. It prevents invalid rows.
- **First:**
  - cherry-pick PRs 354 (SSE `[DONE]`) and 353 (LABD corpus);
  - lift `bench/paired_run.sh` and `bench/canary.sh` from `origin/master` (check them against `bench/harness.py`);
  - require `PTH_FA` pinned and `[effective-config]` read in every arm (D10).
- **Rows:** bench-2 (smoke) after the change.
- **Kill:** revert any lift whose test fails against the current harness.

**Q5 Doc drift fixes, after Q1.**
- **Findings:** F-P3-01, F-P3-03.
- **Gain:** docs only.
- **First:** apply the six X-15 items, plus the mtp/long PTH clause. Check overlap with PR 366's doc edits.
- **Rows:** none.
- **Kill:** n/a.

**Q6 F5c import guard.**
- **Findings:** F-P3-02.
- **Gain:** robustness only.
- **First:** a human decision on "refuse" vs "fall back loudly". If fall back, apply `X-16/fallback.patch` (3 files). [review (phase 5): no decision is needed. The patch only makes `register()` decline, and launcher `check()` still refuses (`start_qwen.sh:244-249`, `phase4-cpu-experiments.md:146-147`). Apply it under the current refuse policy.]
- **Rows:** none.
- **Kill:** drop it if the policy stays "refuse" (`start_qwen.sh:244-246`). [review (phase 5): superseded; kill only if a launcher test shows a backend accepted that `check()` should refuse.]

## Tier 2: next sprint

These need the GPU queue or M effort.

**N1 Re-baseline on the post-FF tree.**
- **Findings:** F-P1-13, F-P3-05, F-P1-12.
- **Gain:** none. It sets the reference arms and adds TTFT and peak VRAM.
- **First:** bench-1 (env snapshot with `PTH_FA` recorded).
- **Rows:** bench-1 to bench-12. bench-10 and bench-12 also settle the 4.5-6.5 ms speculation overhead (**estimated**).
- **Kill:** if the boot-to-boot spread is > 3% or CV > 5%, add boots before any A/B uses the row.

**N2 Spec-timing variant image (X-17).**
- **Findings:** F-P1-04.
- **Gain:** a tool. It splits draft and verify time and decides whether N6, L5 or L6 has room.
- **First:** build `hyperqwen:sha-2cf50a0-spectiming` (D12).
- **Rows:** bench-13, read with bench-10 and bench-12.
- **Kill:** variant < 0.97 of stock means the timing data is not used. Fall back to the bench-10 profile.

**N3 Fitted k table at N=3 (X-02b′).**
- **Findings:** F-P1-06.
- **Gain:** about +5% at N=3 only (**estimated**). 0-2% at N=4/8, so no N=4/8 claim.
- **First:** run the fitted table vs DYNAMIC_K at one boot each, with an N=1 guard and per-position acceptance capture.
- **Rows:** bench-18.
- **Kill:** ratio < 1.05 at N=3, or N=1 outside 2%. Then keep DYNAMIC_K and close H02′.

**N4 Chinese full-head switch (X-09).**
- **Findings:** F-P2-01.
- **Gain:** Chinese +100-127% (**estimated** from **measured (reported)** endpoints 35.5 → 80.6 tok/s). English 0. [review (phase 5): +89% (2.0 / 1.06) to +127%; the +100% floor has no derivation. Both endpoints exist today: `MTP_DRAFT_VOCAB=0` is the documented full-head setting for Chinese (`single-user/README.md:438`). A Chinese-only deployment gets the gain now with no code. The switch only helps mixed-language traffic on one boot, by an amount nobody has measured. Source is internal (gotcha 61), not EAGLE-4.]
- **First:** the human supplies `zh_prompts.jsonl`. Implement the coverage-triggered switch arm (M); it does not exist.
- **Rows:** bench-19.
- **Kill:** full head < 2.0 tok/step on Chinese, or English D C1 outside 2%, or a resident second head that does not fit in D's free memory.

**N5 Mode A int8 KV (X-10).**
- **Findings:** F-P2-02.
- **Gain:** drift 1.7× lower (**measured (reported)**). Speed ±3% (**estimated**).
- **First:** use the TRITON int8 arm. The batch launcher has no `PTH_FA`.
- **Rows:** bench-20.
- **Kill:** 64-concurrency e2e < −3% vs fp8, or drift ≥ 0.025 nats.

**N6 EAGLE-4 drafter as EAGLE-3 (X-03).**
- **Findings:** F-P1-05, F-P2-04, F-P2-08.
- **Gain:** 0 by itself. It is an information gate with a low prior. It needs 1.117-1.31× DFlash2 tok/step to pay (**estimated**). It also reports tok/step per context bucket, which tells whether the 4096 window beats DFlash2's 2048 at 12-36k.
- **First:** a load-only pre-gate:
  - config arch edit to `LlamaForCausalLMEagle3`;
  - aux [4,32,60];
  - drop `embed_tokens` and `efficient_lm_head*`;
  - check `torch.equal` and the page size.
  Pin `PTH_FA` in both arms.
- **Rows:** bench-17. Evaluation only, under licence §2(a).
- **Kill:** LOSS at r < 1.00 closes the EAGLE-4 drafter lane (L1, L3). r between 1.00 and 1.12 keeps only L3, and only if within 10% of break-even.

## Tier 3: larger bets

These are L effort, licence-gated, or need training.

**L1 W4A16 EAGLE-4 drafter (X-11).**
- **Findings:** F-P2-03.
- **Gain:** servability only. Page padding (≈0.85-0.9 GiB) likely turns the −0.34 GiB saving into a net loss (**estimated**).
- **First:** only after an N6 WIN. Then the human supplies V4 weights or a quant tool.
- **Rows:** bench-21.
- **Kill:** tok/step < −3% vs bf16 V3, or net pool after padding below DFlash2's. Shipping is **blocked on D11**.

**L2 MTP TTT unroll 2→4 (X-14).**
- **Findings:** F-P2-07.
- **Gain:** 0-3% D tok/step (**estimated**, low prior).
- **First:** provide `hidden.npy` (≈74 GB) and the model dir on the box.
- **Rows:** bench-24.
- **Kill:** < +3% tok/step, or D C1 below the +3% / 5-boot bar.

**L3 Body-only domain adaptation (X-13).**
- **Findings:** F-P2-06.
- **Gain:** 0-10% tok/step (**estimated**, low prior).
- **First:** only if N6 is within 10% of break-even. Build the `experiments/X-13/` tooling.
- **Rows:** bench-23.
- **Kill:** the post-adaptation ratio is still < 1.12 in the same run. Shipping derived weights is **blocked on D11**.

**L4 DFlash2 window retrain.**
- **Findings:** F-P2-08.
- **Gain:** up to the −12% tok/step at 12-36k (**measured (reported)** loss).
- **First:** only if N6's per-bucket data shows a wider window helps on this target, and a DFlash2 training recipe exists. There is none today.
- **Rows:** none queued.
- **Kill:** no recipe, or N6 shows no bucket effect.

**L5 SVD factors (X-12), then a low-rank candidate head (H05).**
- **Findings:** F-P2-05, F-P1-07.
- **Gain:** +1.7-3.1% B C1, ≤ 1.3% at N=4 (**estimated**).
- **First:** **blocked on D11**. If it clears, run the CPU recall check on the box.
- **Rows:** bench-22, then X-05 (not queued).
- **Kill:** recall@128 < 91.5% after SVD init, or N2 shows the lm_head share of the draft step saves < 0.5 ms.

**L6 Device-side depth stop (X-06).**
- **Findings:** F-P1-08.
- **Gain:** ≤ 1-2% D C1 (**estimated**), below the bar.
- **First:** build the conditional-node capture probe tooling; none exists.
- **Rows:** bench-25.
- **Kill:** capture fails under WSL2/WDDM, or < +3% over ≥ 5 boots.

DeepInfer overlap, for context only (D3):
- L5 touches DI quant-5 and evals-8.
- L6 touches spec-2.
- L4 touches drafter-4.
- Tree verify (skipped) touches spec-2 and spec-3.
- GDN residency (I-03) touches spec-1, spec-3 and runtime-1.

## Decisions needed from the human

1. **D11:** does shipping a low-rank candidate head, an EAGLE-4 draft, or weights derived from it (adapted or quantised) in HyperQwen fall under the licence's "Competing Offering" clause? It decides L1, L3 and L5. Local evaluation (N6) is allowed under §2(a).
2. **Fast-forward: resolved (D22).** `main` (local and origin) and `research/eagle4-expedition` now sit on `upstream/main` 63f7581, which removes the image/checkout skew.
3. **Box defaults:** the box `.env` is `CTX=fast VISION=1`. Keep it and set `VISION=0` per mode-D arm, or change the box default? [review (phase 5): not needed for the benchmarks; every queue arm already strips these keys and sets `VISION=0`. This is only about the box's everyday serving default.]
4. **Import policy:** "refuse" or "fall back loudly" when a kernel package cannot load (Q6)? [review (phase 5): moot for Q6; the patch is compatible with refuse.]
5. **Inputs:**
   - a Chinese prompt set, and whether to build the switch arm (N4);
   - V4 weights or a quant tool (L1);
   - `hidden.npy` plus disk for L2;
   - confirm the 6.2 GB EAGLE-4 download for bench-17.
6. **Optional:** a scratch-branch allow-list edit (`start_qwen.sh:297-304`) for the PTH sub-arm on `CTX=fast` (≤ +5% C1, **estimated**).
7. **Optional:** record the PR #96 KV-sharing negative as an upstream gotcha. It lives only in the closed PR.

## Deferred / skipped

| Item | Findings | Reason |
|---|---|---|
| Verify attention at long context (78% of the E step at 170k) | F-P1-12 | The largest E-mode lever, but not EAGLE-4's. It belongs to a separate attention lane (D3). |
| Tree verify (E-05/E-14/E-25; E-15/16/17 for chain) | F-P1-09 | ≈1.2 GiB GDN state per 16 rows, causal-only kernels, tree attention removed from vLLM; 0 for a chain. |
| `+eagle.4` vLLM fork | F-P1-10 | 33 of 51 patches fail on HEAD; the build is unpublished. |
| Hopper/Blackwell paths, E-23 L2 prefetch | F-P1-11 | Not sm86; the 6 MB L2 and SGLang-only path give ≈0. |
| Fitted k table at N=4/8 | F-P1-06 | 0-2% (**estimated**), below the +5% bar under both tail shapes. |
| H05 as a near-term item | F-P1-07 | L effort, D11, and inside the spread. It survives only as L5. |
| Training a drafter from scratch | F-P2-09 | ≈53 h per 50M tokens (**estimated**). |
| Cross-layer KV sharing | F-P2-10 | GSM8K 0.94 → 0.56-0.64 (**measured (reported)**). |
| Sampled-acceptance fix for int4 DFlash2 | F-P2-11 | Same rule as ours for a chain; 0 gain. |
| Shared-head V6/V7 as "EAGLE-4" | F-P2-04 | Changes the model; not the released drafter. |
| Upstream hygiene PRs 355-358, 360-366 | F-P3-06 | Take them through upstream; no speed effect. |
| PR 364 (drafter JSONL tail repair) | F-P2-12 | Only if we generate drafter data. |
| Re-running the cline +41%/+91% bands | F-P1-06 | +41% does not reconcile (+10.2% from its own row). +91% came from boot-time k=3 plus int8, which Q2 tests directly. |
| Landing `ctx-profile-sweep` | F-P1-13 | Useful for E-mode work later; not EAGLE-4. |

## Corrections noticed

These are errors or inconsistencies in other docs. They are listed here and not edited (task constraint). [review (phase 5): each item is verified and marked below; details in `00-meta/reviews/phase5-findings.md`.]

1. `02-eagle4/transferability.md` (X-01a row and summary) and the brief say our mtp/long launch resolves to `FA2_FP8KV`. It resolves to fp8 on FlashInfer with piecewise graphs (`X-01a/run.out:75`). The review annotates this, but the original text is still there. **[rejected]** already annotated in `transferability.md` (X-01a row) by the phase-4 review; `brief.md` makes no FA2_FP8KV claim.
2. `04-benchmarks/experiments/X-08/classify_divergence.py:4,46` and `record_top2.py` keep `--ulp-band` at 0.13. D18 and the review say 0.25. The queue passes 0.25 explicitly; manual runs do not. **[applied, partly]** `classify_divergence.py:4,15,46,176` default now 0.25, tests pass, X-08 README annotated. `record_top2.py` has no `--ulp-band` argument, so nothing to fix there; `gpu-agent-prompt.md`'s claim that it has a 0.13 default goes to the GPU hand-off.
3. `greedy_equal.py` near-tie default is 0.1. D18 says 0.01. **[deferred]** `harness/greedy_equal.py:84` and `harness/README.md:85` are outside this review's edit scope; listed under "For the GPU hand-off".
4. `01-project/dflash-specdec.md:471` reads "38/793" as 38 wrong self-hits. It means 38 of 793 characters tracked (`00-meta/reviews/phase4-cpu-experiments.md:126`). **[applied]** `dflash-specdec.md:471` annotated (source `single-user/start_qwen.sh:399-411`).
5. `04-benchmarks/LEDGER.md:64` (X-06) notes the plan's +3% vs the index's +2%. `transferability.md:225` still says ≥ +2%, while `plan.md:53` and D16 say ≥ +3% over ≥ 5 boots. **[applied]** `transferability.md:225` annotated with the +3% / ≥ 5-boot bar from `plan.md:36,53`. The bar is not D16 (D16 is two boots per baseline). `LEDGER.md` not touched.
6. The X-15 README cites `resolve_config.sh:55-56`; the right line is `:57` (phase-4 review). **[rejected]** already annotated in the X-15 README (`[review: :57]`).
7. `transferability.md:29` ("five most promising" #2) and `upstream-diff.md:74,109` still quote the cline +41%/+91% bands. +41% does not match its own row (`phase3-transferability.md:32`). **[applied]** `upstream-diff.md:74,109` and `transferability.md` E-13 row annotated; `transferability.md:28` already carried the review note.
8. `gpu-agent-prompt.md:1101` (bench-23 gate) uses the absolute bar 3.375 tok/step. The reviewed X-03 bar is a same-run ratio (≥ 1.12×; ≥ 1.008× for "within 10%"). The two can disagree when DFlash2's same-run tok/step is not 3.36. **[deferred]** verified; the bench-23 gate is in `gpu-agent-prompt.md`, so it goes to the GPU hand-off (ratio gate r ≥ 0.90 × 1.12 = 1.008 and r < 1.12, same-run).
9. `04-benchmarks/baseline.md:23` and the harness README say the harness records WSL-vs-host clock drift. `snapshot.sh` does not (harness gap 2). **[applied]** `baseline.md:23` annotated. The harness README makes no such claim; harness gap 2 already records the gap.
10. `00-meta/plan.md:79` lists "H02" as CPU-now, but it is now H02′. The pre-review ranking (H02 +8-40%, H04 1.8×) sits in a collapsed "Original phase-3 ranking" block (`plan.md:57-75`), which is correct but easy to cite by mistake. **[applied]** `plan.md:79` annotated.
11. `plan.md:93` says mode A is 948 tok/s e2e. `baseline.md:33` gives 814.3 e2e (904 by median TPOT). Check which arm 948 refers to. **[applied]** `plan.md:93` annotated: 948 is the native-Linux README figure (`README.md:108`), 814.3 is the box's WSL2 measured reference. Both are right for their configs.
12. `02-eagle4/architecture.md:134-140` describes the H06 mechanism wrongly. The feature is `begin_capture_to_if_node` (`phase3-transferability.md:153`). **[rejected]** `architecture.md:134-140` is correct (pre-verify confidence stop via `begin_capture_to_if_node`). The phase-3 review called the H06 *hypothesis* wrong, not architecture.md. F-P1-08 fixed instead.
13. `transferability.md:223` and `gpu-agent-prompt.md:1105` cite the PTH allow-list as `start_qwen.sh:299-304`, while the reviews cite `:297-304`. The `PTH_FA` default is at 297 and the refusal at 299-305, so both are near but not identical. **[rejected]** cosmetic: the default is at 297 and the refusal `if/case` at 299-306; both ranges point into the same block.
