# Phase-3 review: transferability and ranked hypotheses

Reviewer: phase-3 adversarial challenge (Opus, effort high), 2026-10-21. No GPU on this box. References were opened read-only: the EAGLE-4 repo, the DeepInfer notes, vLLM v0.31.0 via `git show`, and the `cline/gf59cdmh` branch via `git show`.

I re-ran `02-eagle4/transferability-calc.py`, and its output is identical to `transferability-calc.out`. The arithmetic is correct for its inputs. The problems are in those inputs and in how the hypotheses are framed.

Labels: **M** = measured (the path is given), **E** = estimated (the reasoning is given). Branch-reported numbers come from `origin/cline/gf59cdmh:docs/decode-speed-findings.md` ("cline findings"). They are measured on the branch but not re-verified here.

## Verdict: ACCEPT WITH CHANGES

The memory gate, the break-even arithmetic and the no-go list hold up. But the ranking has four problems:

- It credits the expedition with gains that already exist. These are H01 (in the stock image), H02 (DYNAMIC_K), and H04 (measured on the cline branch).
- It uses a fixed absolute bar where only a same-run ratio is valid (H03).
- It misses a page-size padding cost that hits the EAGLE-4 draft (H03 and H11).
- It ranks two items whose estimates are below their own win bar (H05 and H06).

No ranked item needs FP8 tensor cores, TMA, clusters, PDL, Gluon or flashinfer_cutedsl.

## 1. H02 baseline strawman, and E-13/E-04 vs DeepInfer spec-1

**Evidence**
- **DYNAMIC_K already exists and is measured.** The table `[[1,3,7],[4,512,3]]` lifted 4-stream decode by +7.7%, +8.0% and +9.2% in three A/Bs, with single-stream within 1%. It "buys step time, not seats" (M, `single-user/start_qwen.sh:678-694`, `single-user/README.md:434`). A 4-boot interleaved A/B also exists (`01-project/upstream-diff.md:70`).
- **Seat bands are already measured at batch 4** (cline findings, S-1 row, line 83). In tok/s aggregate:

  | Setting | C1 | N=4 | N=8 | N=16 |
  |---|---|---|---|---|
  | k=7 | 103.4 | 239.1 | 176.8 (tail) | 156.1 |
  | k=3 (`DFLASH_TOKENS=3` + int8 TRITON) | 93.5 | 263.6 | 337.1 | 300.4 |
  | `SPEC=off` | 40.7 | 143.2 | 250.7 | 369.2 |

- **The "+41% at N=4" headline (cline findings line 12) does not match its own row.** 263.6 / 239.1 = +10.2%. The +41% is presumably measured against a different baseline (E: 263.6 / 1.41 ≈ 187, which looks like a bf16 default with preemptions). Do not cite +41% as a k-band gain.
- **A per-step k cannot buy seats.** State pages are sized for the configured maximum k (cline findings section R; `kv_cache_interface.py:1048,1083-1094` as cited there; `01-project/architecture.md:222`). The N=8 +91% comes from a *boot-time* `DFLASH_TOKENS=3` plus the int8 pool, not from a schedule.
- **DeepInfer spec-1 is confidence-scheduled adaptive verification, a different mechanism** (`deepinfer-ref/.../spec-decode.md:17,25,59`). Its simulation claims +6–7% over DYNAMIC_K at C4 and about 0 at one stream (`spec-decode.md:64,129`). D3 keeps it out of scope as an idea. It is also not a valid comparator for an EAGLE-4 hypothesis.

**Conclusion**
- H02 as written ("≥25% at N=8 over fixed k=7") is a strawman. Within one boot it is also not achievable, because seats are fixed at boot.
- What EAGLE-4 adds is narrow:
  - a cost table fitted from measured round costs;
  - a native-OFF band (width 0, allowed by `integration-paths.md:76`);
  - both inside one boot, compared against today's DYNAMIC_K table.
- Expected gain over DYNAMIC_K: 0–3% at N=4 (E, because the measured table already sits at the k=3 optimum the fit would find). There may be more at N≥9 if the off band engages, but that needs a 16-seat boot, which costs k=7 C1 −3.9% (cline X-3, M).
- **Reframe H02:** compare against the DYNAMIC_K table at the same boot, with a win at ≥ +5% at N=4 or N=8, and C1 within 2%. **Demote it to rank 5.** X-02a's pass rule (reproduce +7.7–9.2%) stays. X-02b's "DFLASH_TOKENS=3 to test seats" arm is already measured (cline S-1); drop it.

## 2. H01: real box config, framing, and what changes for users

**Evidence**
- **The stock image already resolves `PTH_FA=1` for D.** The baseline D row is 99.8 tok/s with PTH_FA=1 on the stock image (M, `04-benchmarks/baseline.md:44`). The PTH_FA=0 arm is 63–75 (`baseline.md:54`). The plan already says "0 vs stock image" (`plan.md:40`).
- **The gain is already measured on this GPU type:** +21.3% at ~66k and +22.1% at ~93k, 3 boots per arm, WSL2 3090. tok/step was equal within 1%, GSM8K was 0.9507 vs 0.9522, and the pool was −1.5% (M, `01-project/upstream-diff.md:126`).
- **The box's `.env` is `CTX=fast`,** where the default never fires. The bench arms set `CTX=long` explicitly.

**Conclusion**
- H01 is an adopt/sync item, not a hypothesis. It changes nothing for a stock-image user. It changes only fork-checkout launches of `SPEC=mtp CTX=long`, which flip from fp8 FlashInfer to int8 pthfa: −1.5% pool, and pthfa must be installed.
- Keep it at rank 1 because it is cheap and certain. But:
  - relabel the gain as **M (upstream-diff.md:126), against our checkout only**;
  - make X-01a (CPU resolution check) the only experiment;
  - X-01b repeats the existing bench-9 / pthfa-ab boots (`upstream-diff.md:126`), so one confirmation boot is enough.

## 3. H04: PTH_FA refusal, int8 at short context, and the 1.8× pool

**Evidence**
- **The launcher refuses `PTH_FA=1` outside `mtp/long|dflash2/long`** with exit 1 (M, `start_qwen.sh:297-304`). X-04 therefore needs:
  - an allow-list edit;
  - a `resolve_config` change;
  - a `test_pthfa_knob.sh` update.
- **The cline branch already measured the same lever with no launcher change** (cline S-2, M; `EXTRA_ARGS` int8 TRITON on `CTX=fast`):
  - pool 68,605 → 116,440 tokens (**1.70×**);
  - N=8 preemptions 23 → 1;
  - C1 tie, 111.2 vs 111.0;
  - GSM8K within one question;
  - PPL within 0.07%.
- **The 1.8× criterion would fail against the measured 1.70×.** The "16.4k → 31.8k per GiB" figure (1.94×) is derived from D′ geometry, not B's. Mamba state pages and the drafter KV do not halve with the target KV.
- **There is no 3090 short-context PTH evidence.** Every PTH_FA measurement is 2×3080, TP=2 (`docs/long-context.md:261-500`). At 8k, PTH vs TRITON int8 is 1.07× decode [CI 0.95–1.10] (`long-context.md:412`), and PTH's gain grows with context. So the extra gain of PTH over TRITON int8 at B's ≤64k is small (E, ≤ +5% at C1, CI includes 0).

**Conclusion**
- H04 is internal, not EAGLE-4. Its main effect (the seats) is **already measured**.
- **Reframe it as H04′:** "adopt cline S-2 int8 TRITON on B". The gate is one confirmation boot: pool ≥ 1.6×, N=8 preemptions ≤ 1, C1 within 2%.
- PTH_FA on fast becomes an optional sub-arm: needs the allow-list edit, for ≤ +5% E.
- It stays at rank 2.

## 4. H03/H11: recompute the break-even, memory gate, and probe runnability on vLLM 0.31

**Which number.**
- The 3.36 tok/step is greedy C1 on the `run_benchmarks` cohort with LOOKUP=1 hits included (M, `baseline.md:35`; lookup fills 4.3% of positions, `01-project/dflash-specdec.md:226`). spec-5 gives 3.486 at T=0 and 3.611 at T=0.7 (M, `baseline.md:39`).
- Acceptance is a workload property: 5.80 on GSM8K, 4.78 on code and 3.50 on chat (cline S-8, branch-reported).
- An absolute 3.75 cut-off on "the B prompt set" therefore depends on the prompt set and the mode.
- The recomputed formula is need = 3.36 × (29.4 − 3.3 + t_draft) / 29.4. It gives 3.75 at the 6.75 ms floor and 4.40 at ×1.83 (calc output, re-run, matches). As ratios that is **≥ 1.117× (floor) to ≥ 1.31× (×1.83) of DFlash2's tok/step on the same prompts in the same run.**
- The 3.3 ms DFlash2 draft cost includes lookup ("drafter + lookup 3.6", `dflash-specdec.md:362`). An EAGLE-4 chain arm must state whether lookup stays on.

**The 0.86 GiB figure** is correct: (435.2M + 78.6M + 1271.4M) × 0.515625 B = 0.857 GiB (calc re-run).

**Probe runnability on vLLM 0.31.0** (from tag source; E):
- **No target whitelist.** `config/speculative.py` has none for eagle3. `qwen3_5.py:294-298,368-373` (`SupportsEagle3`) and `qwen3_vl.py:124,1798` accept it.
- **Aux ids [4,32,60] are correct.** `eagle3_utils.get_eagle3_aux_layers_from_config` uses `eagle_aux_hidden_state_layer_ids` as given. The repo's [3,31,59] are 0-based and count after the layer, hence +1.
- **The architecture change is a config edit.** `registry.py:659-667` maps `LlamaForCausalLMEagle3` to `llama_eagle3`, so editing `config.json` "architectures" is enough. Draft vocab equals target vocab, so no d2t is needed.
- **Drop `embed_tokens`** (and `efficient_lm_head*`) from the draft safetensors. Otherwise `eagle/utils.py:15-27` `_should_share` runs `torch.equal` against a target embedding that is int8/UVA and may have no `.weight`.
- **Untested risks:**
  - `EagleSpeculator` on a GDN-hybrid target. MTP uses a different class, so no evidence exists either way.
  - M-RoPE target positions against the draft's 1-D rope (θ 1e7).
  - Whether the draft KV dtype override reaches EAGLE layers (`integration-paths.md` row 1, reviewer additions).
- So **no code patch is expected, but X-03 needs a 30-minute load-only pre-gate.**

**Memory gate: page-size padding is missing.**
- vLLM 0.31.0 `kv_cache_utils.py:1383-1445` unifies page sizes. Mamba pages are padded to the maximum page, and a non-dividing attention spec is padded as well.
- The draft layer is 20,480 B per token per layer, 5× the target's 4,096 (`patches/hybrid-sw-block-promote.patch` header).
- At block 16 its page is 327,680 B. The primary page is 1,835,008 B (448 × 4096). The ratio, 5.6, is not an integer.
- **With the promote patch** (rounds up to a divisor of 448 that is a multiple of 16): it picks 112 tokens → 2,293,760 B. That is bigger than the primary page, so every target and GDN page pads +25%, and the pool shrinks by about 20% (E).
- **Without promotion:** the window costs cdiv(4095 + 4096, 16) + 1 ≈ 513 blocks × 1.75 MiB ≈ 0.9 GiB per request (E, the patch's own bound formula).
- Either way this is a pool cost of about 1 GiB that the gate does not count. B's pool is about 4.2 GiB of target attention alone (E: 68,605 tokens × 16 layers × 4,096 B, from cline S-2), so 20% is about 0.85 GiB plus the GDN pages. That erases H11's −0.34 GiB "saving" (E).
- The earlier worry that the draft KV is "not freed → 1.25/3.24 GiB" is probably moot. The same patch shows windowed allocation is bounded at that formula.

**Conclusion**
- H03 stays P1 but at rank 4. Its bar becomes the same-run ratio (≥ 1.12× of DFlash2 tok/step at T=0, also reported at T=0.7, LOOKUP stated). It gets a load-only pre-gate and a page-size check: read the boot-log page size and pool tokens.
- H11 adds the page-padding cost to its success criterion: net pool must be ≥ DFlash2's, not just weight bytes.

## 5. H05: is it worth L effort and a licence gate?

**Evidence**
- The +2.2% floor estimate is consistent: 0.63 ms saved of a 29.4 ms step (calc re-run).
- The realistic range is 1.7–3.1% at C1. At N=4 it is ≤1.3%, because the step is 49.1 ms (E, `dflash-specdec.md:426-431` step curve).
- The win bar is +3% at C1, and X-05 itself says a clean win is "unlikely to be resolvable" (`transferability.md:222`).
- It also needs L effort, D11 (licence), and factors trained on DFlash2 rows (H12 first).

**Conclusion:** the expected value is below the bar after L effort plus a licence gate. **Move H05 to not ranked (estimate-only).** Revisit only if H12 reaches recall ≥ 98% for another reason.

## 6. Win threshold: is the 1–2% spread measured, and where?

**Evidence**
- "Spread 1–2% round to round" comes from the prior `baseline.md:31,69` via `04-benchmarks/baseline.md:13`. It is two boots per mode, and the raw jsonl was never committed.
- The cline batches measured **0–3.2% between boots for C1 at T=0, and about 10% for free-run decode**. Teacher-forced acceptance reproduces to 3 decimals (cline findings, M branch-reported).
- There is also a thermal settle of −9.4% (`baseline.md:41`).

**Conclusion:** with 3 boots, a +3% C1 win sits inside a 0–3.2% spread, so it is marginal. Edit `transferability.md:11-14` to:
- require ≥5 interleaved boots for any C1 claim below +5%; or
- split the claim into teacher-forced tok/step (deterministic) × step time (ms/pass, from logs). This is preferred, because it removes the free-run variance.

The N≥4 bar of +5% is fine against the measured spread.

## 7. No-gos wrongly dismissed, and non-Ampere features

**Tree verify (E-05/E-14..17/E-25).**
- The "unquantised KV halves context" argument does not apply in B and C, which already run bf16 KV with FLASH_ATTN (M, `start_qwen.sh:210-214`). B also does not occupy the CUSTOM backend slot.
- The no-go still holds for other reasons:
  - GDN tree state rows (72 MiB/row fp16);
  - a causal-only kernel;
  - tree attention removed in 0.31;
  - L effort, D11, and no `+eagle.4` hooks.
- The cited reason should be corrected.

**H06 / E-12: the mechanism is wrong.**
- EAGLE-4's depth stop is decided *before* verify, from draft-front confidence and a cost share. It does not wait for a pos-0 verify reject (`02-eagle4/architecture.md:134-140`). The current hypothesis can only act on the next step.
- The ≤1–2% estimate (`transferability.md:148`) is below the +3% bar.
- The feature is PyTorch `torch.cuda.CUDAGraph.begin_capture_to_if_node` (`EAGLE-4/eagle_harness/vllm/tree_drafter.py:191-207`). It needs CUDA 12.4+ conditional nodes and is not Hopper-only, but it is untested on WDDM/WSL2.
- **Demote H06 to the bottom of the P1/P2 block, with the mechanism fixed.**

**Non-Ampere audit**
- No ranked item depends on FP8 MMA, TMA, clusters, PDL (sm90+), Gluon (the GDN path is sm_103) or flashinfer_cutedsl (`03-vllm/integration-paths.md` §7).
- **H10:** fp8 KV in A on sm86 works only through FlashInfer, as dequant with no fp8 MMA (`start_qwen.sh:28-39`).
- H10's int8 arm needs either TRITON int8 or PTH outside the allow-list (`start_qwen.sh:297-304`).
- The prebuilt pthfa library has sm_80 SASS, not sm_86. That runs on sm86 but should be noted (`integration-paths.md` §7).

**DFlash2 window retraining (E-11)** stays out (L, no recipe). Draft-KV sliding window is fine.

## 8. Scope: ideas from outside EAGLE-4

- **H04, H09, H10 and H15 are internal carry-forwards.** H01 is upstream. That is allowed by the plan, but they should be labelled "internal" so they are not counted as EAGLE-4 transfers.
- **H02 overlaps** DeepInfer spec-0/1/2 and cline S-1. Only the fitted cost table with native-OFF is EAGLE-4 (E-13/E-04). DeepInfer's +6–7% must not be used as a target or an idea (D3). Nothing in the docs imports a DeepInfer mechanism; the overlap is only in the comparator.
- **H08** (E-18 near-tie gate) is EAGLE-4. In scope.

## Proposed corrected ranking

| Rank | ID | Hypothesis (falsifiable) | Prio | Expected gain (E) | Exp | Where | Effort | Licence-gated? |
|---|---|---|---|---|---|---|---|---|
| 1 | H01 (U-01) — adopt | Fast-forwarding to `upstream/main` (#345) makes an unset-PTH_FA `SPEC=mtp CTX=long` launch from our checkout resolve to PTH_FA=1 | P1 | +21–22% D decode at 66–93k vs our checkout (**M**, `upstream-diff.md:126`); 0 vs stock image | X-01a (+1 confirm boot) | **CPU-now** + GPU | S | no |
| 2 | H04′ (internal) — adopt | int8 per-token-head KV (TRITON, via `EXTRA_ARGS`) on `SPEC=dflash2 CTX=fast` reproduces pool ≥1.6×, N=8 preemptions ≤1, C1 within 2% | P1 | pool 1.70×, preemptions 23→1, C1 tie (**M** branch, cline S-2); PTH sub-arm ≤ +5% C1 (E) | X-04′ | GPU | S | no |
| 3 | H08 (E-18) | Near-tie + fp32 re-score classifier separates benign from real greedy divergences | P1 | Correctness gate | X-08 | **CPU-now** + GPU | S | no |
| 4 | H03 (E-20/E-02) | EAGLE-4 drafter as EAGLE-3 (aux [4,32,60], embed dropped) loads on 0.31 and reaches ≥1.12× DFlash2's tok/step on the same prompts, same run, at T=0 | P1 | Information gate (prior low) | X-03 (+ load-only pre-gate, page-size check) | GPU | M | no for eval; yes to ship |
| 5 | H02′ (E-13/E-04) | A fitted cost table with a native-OFF band beats the existing DYNAMIC_K table at the same boot by ≥5% at N=4 or N=8, C1 within 2% | P1 | 0–5% (E); the k=3 optimum is already in DYNAMIC_K | X-02a, X-02b′ | **CPU-now** + GPU | S/M | no (own tables) |
| 6 | H09 (internal) | unchanged | P2 | unchanged | X-09 | GPU | M | no |
| 7 | H10 (internal) | unchanged; int8 arm via TRITON or allow-list edit | P2 | unchanged | X-10 | GPU | M | no |
| 8 | H11 (E-20) | If H03 passes: W4A16 layer + int4 own head keeps tok/step within −3% **and** net pool (after page-size padding) ≥ DFlash2's | P2 | Servability; page padding ≈1 GiB (E) likely erases the −0.34 GiB | X-11 | GPU | S/M | no for eval; yes to ship |
| 9 | H12 | unchanged | P2 | unchanged | X-12 | CPU w/ weights | S/M | **yes (D11)** |
| 10 | H13 | unchanged (bar becomes the H03 ratio) | P2 | unchanged | X-13 | GPU | M | no for eval |
| 11 | H14 | unchanged | P2 | unchanged | X-14 | GPU | M | no |
| 12 | H06 (E-12) | A confidence-gated, pre-verify depth stop via a CUDA-graph conditional node captures on the box's driver; D C1 ≥ +3% (≥5 boots) | P2 | ≤1–2% (E), below bar | X-06 | GPU | M | no for CUDA; yes if ported |
| 13 | H15 | unchanged | P3 | Docs | X-15 | **CPU-now** | S | no |
| 14 | H16 | unchanged | P3 | Robustness | X-16 | **CPU-now** | M | no |

**Moved rows**
- **H01** keeps rank 1, relabelled as an M adopt item: the gain is already measured, and it is 0 vs the stock image.
- **H04 → H04′, rank 2:** the seat gain is already measured (cline S-2); 1.8× → 1.70×; no allow-list edit needed.
- **H08** goes from 4 to 3: it moves up as the others are reframed.
- **H03** goes from 5 to 4: absolute 3.75 → same-run ratio; adds the page-size and load pre-gate.
- **H02** goes from 2 to 5: the baseline becomes DYNAMIC_K, not k=7; per-step k cannot buy seats; the gain drops to 0–5% E.
- **H06** goes from 6 to 12 (P1 → P2): the mechanism was wrong, and the gain is below the bar.
- **H05** goes from 7 to not ranked: +2.2% E is below the +3% bar, with L effort plus D11.
- **H11** goes from 10 to 8: adds the page-padding term.

**Add**
- A load-only pre-gate for X-03.
- A boot-log page-size check for any new drafter.
- An "internal / EAGLE-4" scope column or tag.
- A "C1 claims via teacher-forced tok/step × ms/pass" rule.

**Drop**
- X-02b's seats arm (already measured).
- X-01b as a separate bench (one confirmation boot instead).
- The "+41% at N=4" citation.

## Edits made (all marked `[review: …]`)

`docs/research/02-eagle4/transferability.md`:
1. Baseline spread (lines ~11–14): cline 0–3.2% / ~10% spread; ≥5 boots or teacher-forced split.
2. Summary item 1 (H01): M label, 0 vs stock image.
3. Summary item 2 (H02): +41% does not reconcile; per-step k does not buy seats; the comparator is DYNAMIC_K.
4. Summary item 3 (H04): measured 1.70× (cline S-2), not 1.8×/1.94×.
5. Memory gate, draft KV: page-size padding cost (+25% pages or ≈0.9 GiB per request), not freed → probably moot.
6. Break-even "What could clear the bar": ratio framing, LOOKUP and T caveats.
7. E-12 row: mechanism is pre-verify confidence, not post-reject.
8. X-02a, X-02b, X-03, X-04, X-05 rows: corrected bars.
9. Five clearest no-gos, item 1 (tree verify): the bf16-KV reason does not apply in B/C; the no-go stands on the other reasons.

`docs/research/00-meta/plan.md`:
- The ranked table was replaced with the corrected one above.
- The original is kept below it as a collapsed "Original phase-3 ranking" list.
