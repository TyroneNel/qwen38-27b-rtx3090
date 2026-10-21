# P1 findings: speed and inference

Phase 5. Sources: the reviewed phase 1-4 docs. A `[review: ...]` mark wins over the text it annotates. Labels follow D19 (`00-meta/decisions.md:23`):
- **measured**: raw files in this repo, with numbers re-derived by a second agent.
- **measured (reported)**: a real GPU run whose raw files we do not hold.
- **estimated**: our arithmetic.

No speedup from this expedition is **measured** yet. CPU experiments prove logic only, never speed. Queue rows use their bench-N base number (`00-meta/gpu-queue.md`).

Win bars (plan [review (phase 5): `plan.md:36` and the queue's ≥ 5-boot rule; D16 is the two-boots-per-baseline decision, not the bar]): at C1, ≥ +3% median over ≥ 3 interleaved boots, and ≥ 5 boots when the gain is below +5%. At N ≥ 4, ≥ +5%. The boot-to-boot spread is 1-2% (C1 T=0 0-3.2%, cline). Thermal settling costs −9.4% over about 4 min (`04-benchmarks/baseline.md:41`).

## Summary

| ID | Claim | Expected impact (label) | Effort | Status | Maps to |
|---|---|---|---|---|---|
| F-P1-01 | Fast-forwarding to `upstream/main` turns on int8 PTH KV for `SPEC=mtp CTX=long` | +12-22% D decode at 66-93k vs our checkout (measured (reported)); 0 vs the stock image | S | adopt now, confirm with X-01b | H01, U-01, X-01a/b |
| F-P1-02 | int8 per-token-head KV (TRITON) on mode B via `EXTRA_ARGS` | pool 1.70×, N=8 preemptions 23→1, C1 tie (measured (reported)) | S | needs GPU run X-04′ | H04′, X-04′ |
| F-P1-03 | A near-tie + replay classifier separates benign greedy divergences from real ones | correctness gate, no speed | S | needs GPU run X-08 | H08, E-18, I-08 |
| F-P1-04 | The spec-timing variant image measures draft vs verify per step | tool; ≤0.7% overhead (estimated) | S | needs GPU run X-17 | X-17, I-04, I-06 |
| F-P1-05 | The EAGLE-4 drafter loaded as EAGLE-3 is an information gate with a low prior | 0 speed by itself; break-even needs 1.12-1.31× DFlash2 tok/step (estimated) | M | needs GPU run X-03; shipping blocked on D11 | H03, E-20, E-02 |
| F-P1-06 | A fitted per-concurrency k table does not beat DYNAMIC_K at N=4/8 (negative) | 0-2% at N=4/8 (estimated); about +5% at N=3 (estimated) | S | N=4/8 dropped; N=3 needs GPU run X-02b′ | H02′, X-02a/b′ |
| F-P1-07 | A low-rank candidate head in DFlash2 is below the bar and licence-gated (negative) | +1.7-3.1% B C1, ≤1.3% N=4 (estimated) | L | blocked on D11; dropped as a speed item | H05, E-10, E-21, I-04 |
| F-P1-08 | A device-side depth stop (CUDA-graph conditional node) is below the bar (negative) | ≤1-2% D C1 (estimated) | M + tooling | dropped (X-06 has no tooling) | H06, E-12, E-22 |
| F-P1-09 | Tree verify on our KV formats is a no-go | 0 for chain parts (estimated); tree needs ≈1.2 GiB GDN state per 16 rows | L | dropped | E-05, E-14, E-15, E-16, E-17, E-25 |
| F-P1-10 | Moving to the `+eagle.4` vLLM fork is a no-go | n/a; 33 of 51 patches fail on HEAD | L | dropped | integration path |
| F-P1-11 | Hopper/Blackwell paths and E-23 L2 prefetch do not apply to sm86 | ≈0 (estimated) | — | dropped | E-23, sm90+/SM100 paths |
| F-P1-12 | Long-context E decode is verify-attention bound; no drafter change moves it much | ≤2.4% ceiling for any drafter gain at 170k (estimated from measured profile) | — | out of EAGLE-4 scope | I-02 |
| F-P1-13 | Benchmark validity: baseline drift, PTH_FA pinning, timing and harness fixes | prevents a 9-22% (long context) to ≈33-58% (C1, box) apples-to-oranges error (measured (reported) magnitude) [review] | S-M | adopt now (code); baselines need GPU runs | D10, D16, U-02, U-03, U-08 |

---

## F-P1-01 Fast-forward to upstream/main: PTH_FA default on mtp/long

- **Priority:** P1.
- **Claim:** after the fast-forward (b4b97ec, PR #345), an unset-`PTH_FA` `SPEC=mtp CTX=long` launch on cc 8.6 resolves to int8 KV on the pth kernel instead of fp8 on FlashInfer, and it decodes faster at long context.
- **Expected impact:** +12-22% D decode at 66-93k vs our checkout (**measured (reported)**). The runs:
  - +21.3% / +22.1% on WSL2, 3 boots per arm, `GPU_UTIL=0.89`;
  - +12.2% / +14.0% native, 2 boots;
  - +9% native, 1 boot.
  The gain is **0 vs the stock image** `sha-2cf50a0`, which already contains b4b97ec. Tok/step is equal within 1%, GSM8K is 0.9507 vs 0.9522, and the pool is −1.5% (**measured (reported)**).
- **Evidence:**
  - the WSL2 3-boot run: `01-project/upstream-diff.md:126`;
  - the native runs: `upstream-diff.md:28`;
  - the gate code: `resolve_config.sh:105-121` (upstream);
  - our checkout resolves to fp8 FlashInfer, not FA2_FP8KV: `04-benchmarks/experiments/X-01a/run.out:3,75` and `00-meta/reviews/phase4-cpu-experiments.md:9,41-45`;
  - the FF has 0 conflicts: `upstream-diff.md:9,18`.
- **Effort:** S. A `--ff-only` merge of 5 commits (25 files) plus `bash pthfa/install.sh` on any box without the package.
- **Risks:**
  - The gain mixes the kernel with FULL vs PIECEWISE CUDA graphs, so it is not the kernel alone.
  - These settings silently fall back to the old path: `VISION=1`, `DRAFT_TOKENS≠3`, `DYNAMIC_K`, KV offload, any `FA2_FP8KV`, and the listed `EXTRA_ARGS` flags (`X-01a/README.md`). The box `.env` sets `CTX=fast VISION=1`, so the default never fires there (`upstream-diff.md:129`). [review (phase 5): this only affects a boot that inherits the box `.env` unchanged. Every queue arm strips SPEC/CTX/PTH_FA/VISION from the base, sets `VISION=0` and pins `PTH_FA` (`gpu-agent-prompt.md:78` and each arm_env), and the prior baseline did the same (`raw/prior-3090/baseline.md:10`, `ENVIRONMENT.md:27`). It does not block bench-15; it matters for everyday use of the box and for user installs with `VISION=1`.]
  - The ≈1.9% smaller int8 pool can make vLLM refuse to boot at `MAX_LEN` 150000 near the edge (`upstream-diff.md:127`).
- **Status:** **adopt now**, because the change is S and reversible with `PTH_FA=0`. The fast-forward itself is a decision for the human (roadmap). The size on our box needs GPU run **X-01b** (bench-15), gate ratio ≥ 1.10 at both cells.
- **Maps to:** H01, U-01, I-13, I-09; X-01a (CPU, confirmed), X-01b.

## F-P1-02 int8 TRITON KV on mode B

- **Priority:** P1.
- **Claim:** int8 per-token-head KV through `EXTRA_ARGS` (TRITON, no launcher edit) on `SPEC=dflash2 CTX=fast` grows the pool and removes the pool-limited N=8 preemptions at no C1 cost.
- **Expected impact (all measured (reported), cline S-2):**
  - pool 68,605 → 116,440 tokens (1.70×);
  - N=8 preemptions 23 → 1;
  - C1 111.2 vs 111.0;
  - GSM8K within one question, PPL within 0.07%.
  The N=8 aggregate gain is not stated alone in our inputs. Our N=8 row (236 tok/s, 19 preemptions) is pool-limited, not compute-limited (`baseline.md:37,64`). An optional PTH sub-arm adds ≤ +5% C1 (**estimated**, CI includes 0) and needs a `start_qwen.sh:297-304` allow-list edit.
- **Evidence:**
  - the S-2 numbers: `00-meta/reviews/phase3-transferability.md:66-80`;
  - the queue spec: `04-benchmarks/gpu-agent-prompt.md:778-780` [review (phase 5): now bench-16a/16b/16c at `gpu-agent-prompt.md:816` ff.; the file is being edited, cite by bench id].
- **Effort:** S. An env change, one ABBA plus one quality boot (≈3.4 h, split into ≤ 2 h parts). [review (phase 5): S to apply, but adoption waits on bench-16; the pool and preemption numbers are another agent's run, and the C1 tie is inside the 1-2% spread.]
- **Risks:**
  - The 1.70× is B-specific: Mamba state pages and drafter KV do not halve.
  - The quality guard must run on the same boot set.
  - It interacts with any EAGLE-4 page padding (F-P2-03).
- **Status:** **needs GPU run X-04′** (bench-16). Gate: pool ≥ 1.6×, N=8 preemptions ≤ 1, C1 within 2%, GSM8K within 1 pt.
- **Maps to:** H04′ (was H04), carry-forward from cline.

## F-P1-03 Greedy-divergence classifier (correctness gate)

- **Priority:** P1.
- **Claim:** a near-tie (< 1e-2 nats) plus replay band (1e-2 to 0.25) plus real (> 0.25) classifier labels every `SPEC=off` vs dflash2/mtp greedy divergence. It must flag the KVarN adaptive + prefix case as real (D18).
- **Expected impact:** no speed. It is the gate under every speed claim in this file.
- **Evidence:**
  - CPU 14/14 pass, **measured** for logic only (`04-benchmarks/experiments/X-08/README.md`);
  - the band review: `00-meta/reviews/phase4-cpu-experiments.md:112-118`;
  - the queue runs pass `--ulp-band 0.25` explicitly (`gpu-agent-prompt.md:725-753`).
- **Effort:** S. The tools exist (`classify_divergence.py`, `record_top2.py`). Running them needs one GPU slot.
- **Risks:**
  - The script default `--ulp-band` is still 0.13 (`classify_divergence.py:4,46`). A manual run without the flag gives false "real" results for gaps of 0.13-0.25. [review (phase 5): fixed; the default is now 0.25 and the 14 tests pass. `harness/greedy_equal.py` near-tie 0.1 is still open (GPU hand-off).]
  - fp32 re-scoring of the 27B (≈108 GB) is impossible on the box, so replay uses `SPEC=off` bf16.
  - bench-7 as specified cannot see the KVarN case. bench-14 adds it.
  - Record ≥ 2 document lengths ("38/793" means 38 of 793 characters tracked).
- **Status:** **needs GPU run X-08** (bench-14, after bench-7).
- **Maps to:** H08, E-18, I-08.

## F-P1-04 Spec-timing variant image

- **Priority:** P1 (measurement).
- **Claim:** six event marks per step (verify / logits / accept / other / draft / gap) under `HQ_SPEC_TIMING`, in a variant image only (D12), measure what the 4.5-6.5 ms speculation overhead (**estimated**) is made of.
- **Expected impact:** a tool, with no speed of its own. Overhead upper bound 0.4-0.7% (**estimated**), ≈10-20 µs on the launch path after fix F2. It decides whether any drafter-side lever (F-P1-05, F-P1-07, F-P1-08) has room.
- **Evidence:**
  - the review verdict READY FOR GPU, fixes F1-F4: `00-meta/reviews/phase4-x17-timing.md`;
  - the patch set: `04-benchmarks/experiments/X-17-timing/README.md` (51/51 series patches applied).
- **Effort:** S. Built and reviewed; it needs one build and an S U T T S overhead A/B.
- **Risks:**
  - acc/draft read low on E keys (F6);
  - coarse percentiles (F11);
  - WDDM query flush (F12);
  - Model Runner V2 only.
  Upstream `StepTimingCollector` has no draft metric to backport.
- **Status:** **needs GPU run X-17** (bench-13). Gate: variant ≥ 0.97 of stock. It pairs with bench-10 (profile) and bench-12 (`SPEC=off` C1), which settle the overhead.
- **Maps to:** X-17, I-04, I-06.

## F-P1-05 EAGLE-4 drafter as EAGLE-3 (V3): information gate

- **Priority:** P1.
- **Claim:** the released Qwen3.8-27B EAGLE-4 drafter loads in vLLM 0.31 as EAGLE-3 and can be compared with DFlash2 on the same prompts, in the same run, at T=0. The load needs:
  - a `config.json` architecture edit to `LlamaForCausalLMEagle3` (`registry.py:659-667`);
  - aux layers [4,32,60];
  - dropping `embed_tokens` and `efficient_lm_head*`.
  There is no target whitelist.
- **Expected impact:** 0 speed on its own. To pay off it needs **1.117-1.31× DFlash2 tok/step** (**estimated**: need = 3.36 × (29.4 − 3.3 + t_draft) / 29.4, which gives 3.75 at the byte floor and 4.40 at ×1.83). The bar is ≥ 1.12× in the same run. The prior is low. The TTFT cost of the draft prefill is < 2% (**estimated**).
- **Evidence:**
  - break-even: `02-eagle4/transferability.md:108-125`;
  - load path: `00-meta/reviews/phase3-transferability.md:93-95`;
  - memory gate: `transferability.md:62-106`;
  - the queue: `gpu-agent-prompt.md` bench-17 (HF `EAGLE-Corp/Qwen3.8-27B-EAGLE4`, commit f58a598, 6,178,152,384 B).
- **Effort:** M. A 6.2 GB download, the conversion, a load-only pre-gate and a page-size check. The run uses lowered `KV_MEM` and `MAX_LEN` 16k.
- **Risks:**
  - The draft page (20,480 B/token) does not divide the primary page (1,835,008 B). The cost is ≈ +25% pages (≈ −20% pool) with promote, or ≈ 0.9 GiB per request without (`phase3-transferability.md:105-109`).
  - V2 embedding sharing needs `torch.equal`, which our int8 embedding fails (`transferability.md:240`).
  - `DRAFT_TOKENS≠3` drops the PTH default, so pin `PTH_FA` in both arms (D10).
  - The `MAX_LEN` 16k pool is not the production pool.
- **Status:** **needs GPU run X-03** (bench-17). WIN r ≥ 1.12, LOSS r < 1.00. Local evaluation is allowed under licence §2(a). Shipping the draft or derived weights is **blocked on D11**.
- **Maps to:** H03, E-20, E-02, T-10, I-01. It decides F-P2-03 (H11) and F-P2-05 (H13).

## F-P1-06 Fitted k table vs DYNAMIC_K (negative at N=4/8)

- **Priority:** P1.
- **Claim:** a k-per-concurrency table fitted to measured acceptance and step costs (`[[1,1,7],[2,2,5],[3,3,4],[4,512,3]]`) does **not** beat the shipped DYNAMIC_K table (`[[1,3,7],[4,512,3]]`, `start_qwen.sh:678,692`) at N=4 or N=8. Native-OFF never wins at N ≤ 16.
- **Expected impact:**
  - **0-2% at N=4/8** (**estimated**) under both a cliff and a smooth acceptance tail. The +5% bar needs a3 ≥ 0.404, but the smooth tail gives 0.25-0.32.
  - **About +4.9% at N=3** (**estimated**, interpolated).
  - The fitted table vs a fixed k=7 gives +9.9% at N=4 (**estimated**), but DYNAMIC_K already takes that gain.
- **Evidence:**
  - the fit: `04-benchmarks/experiments/X-02a/README.md`;
  - the tail check: `00-meta/reviews/phase4-x02a-tailshape.out`;
  - the review: `phase4-cpu-experiments.md:80-87`;
  - why the cline +41%/+91% headlines do not transfer: `phase3-transferability.md:29-32`. +41% does not match its own row (263.6/239.1 = +10.2%). +91% at N=8 came from boot-time `DFLASH_TOKENS=3` plus int8, and a per-step k cannot buy seats.
- **Effort:** S. One table in the launcher.
- **Risks:**
  - The back-test passed ladder-matched (+9.5% vs measured +7.7%) but failed on the spec-5 profile (−3.3%).
  - The +8.0%/+9.2% DYNAMIC_K runs have no raw data in the tree.
- **Status:** the N=4/8 framing is **dropped**. The N=3 boundary check **needs GPU run X-02b′** (bench-18), gate ≥ 1.05 with an N=1 guard.
- **Maps to:** H02′ (was H02), E-13, E-04, U-04, I-05.

## F-P1-07 Low-rank candidate head in DFlash2 (negative)

- **Priority:** P1 (negative).
- **Claim:** rank-128 retrieval plus exact rescoring of 128 rows in DFlash2 `compute_candidates` saves part of the drafter's lm_head bytes. Those bytes are ≈39% of a ≈1.80 ms drafter floor.
- **Expected impact:**
  - +2.2% B C1 (**estimated**);
  - realistic range 1.7-3.1% at C1 and ≤ 1.3% at N=4 (49.1 ms step);
  - +0.4% at E 170k.
  This is inside the 1-2% spread unless measured over ≥ 5 boots.
- **Evidence:**
  - `phase3-transferability.md:120`;
  - `transferability.md:148,199`;
  - `plan.md:79` (moved out of the ranking).
- **Effort:** L. It needs X-12 recall ≥ 91.5% first, then a kernel (E-21 warp-shuffle tail) and a factor head.
- **Risks:**
  - DFlash2 already has a 16×16 low-rank edge score, so the baseline is not a plain chain drafter.
  - Licence: factor-head technology derived from EAGLE-4 is gated.
- **Status:** **blocked on D11**. Even if D11 clears it, it is dropped as a speed item unless X-12 shows ≥ 98% recall and X-17 shows the drafter share is larger than estimated.
- **Maps to:** H05, E-10, E-03, E-21, I-04.

## F-P1-08 Device-side depth stop (negative)

- **Priority:** P1 (negative).
- **Claim:** a confidence-gated, pre-verify stop through a CUDA-graph conditional node (`torch.cuda.CUDAGraph.begin_capture_to_if_node`, `EAGLE-4/eagle_harness/vllm/tree_drafter.py:191-207`) skips MTP draft passes after a position-0 reject. [review (phase 5): mixed mechanism. EAGLE-4 stops drafting *before* verify when draft confidence drops (`architecture.md:134-140`, `tree_drafter.py:186-207`); it does not react to a position-0 reject, which is only known after verify.]
- **Expected impact:** ≤ 1-2% D C1 (**estimated**), below the +3% bar. Conditional nodes cannot change the verify length, because graphs are captured per q.
- **Evidence:**
  - `phase3-transferability.md:150-153` [review (phase 5): that review found H06's *hypothesis* text (pos-0 reject) wrong and cites `02-eagle4/architecture.md:134-140` as the correct description; architecture.md is right];
  - `transferability.md:150,159,201`.
- **Effort:** M, plus tooling that does not exist (`experiments/X-06/`).
- **Risks:**
  - It needs CUDA 12.4+ conditional nodes and is untested under WSL2/WDDM.
  - Porting `tree_drafter.py` would be licence-gated; the generic CUDA feature is not.
- **Status:** **dropped**. bench-25 is blocked on tooling, and the expected gain is below the bar even if it ran.
- **Maps to:** H06, E-12, E-22, I-06.

## F-P1-09 Tree verify (no-go)

- **Priority:** P1 (negative).
- **Claim:** cost-sized tree verify does not transfer to our stack now.
  - GDN recurrence would need parent-indexed state: fp16, 72 MiB per row, ≈ 1.2 GiB per 16 rows.
  - The pthfa/KVarN verify kernels are causal-only.
  - vLLM removed tree attention before 0.31 (`b1728c1e66`).
  - E-15/E-16/E-17 (target-sample-then-match, parent-first order, row attention) give 0 for a single chain.
- **Expected impact:** paper tree vs DFlash2 is 1.28× at T=0 (**measured (reported)**, paper, not our hardware). The memory cost alone wipes out B's 0.7-1.3 GiB headroom (**estimated**).
- **Evidence:**
  - `transferability.md:36,147,152-155,162`;
  - `plan.md:79,87,94,96`.
  The phase-3 review found the "bf16 KV only" reason invalid for B/C. The memory and kernel reasons still hold.
- **Effort:** L. A re-introduction of tree attention, new GDN tree recurrence, and ancestor masks in two kernels.
- **Status:** **dropped**.
- **Maps to:** E-05, E-14, E-15, E-16, E-17, E-25.

## F-P1-10 The `+eagle.4` vLLM fork (no-go)

- **Priority:** P1 (negative).
- **Claim:** EAGLE-4 needs an out-of-tree hooks branch, and its `+eagle.4` vLLM build is unpublished. 33 of our 51 patches fail on vLLM HEAD (22 real drift, 11 only dependent). Moving to the fork means re-porting most of the stack.
- **Expected impact:** n/a. It avoids an L port.
- **Evidence:** `plan.md:79,92`; `03-vllm/integration-paths.md`.
- **Effort:** L (avoided).
- **Status:** **dropped**. The chain-mode probe (F-P1-05) is the cheap path.
- **Maps to:** integration path; E-20 is the alternative.

## F-P1-11 Hopper/Blackwell paths and L2 prefetch (no-go)

- **Priority:** P1 (negative).
- **Claim:** Gluon GDN (sm_103), PDL and clusters (sm_90+) and `flashinfer_cutedsl` (SM100) do not run on sm86, and the Triton twins already cover sm86. E-23 side-stream L2 prefetch is SGLang-only, and the 3090 has a 6 MB L2.
- **Expected impact:** ≈ 0 (**estimated**).
- **Evidence:** `transferability.md:42-46,160`; `integration-paths.md:207`.
- **Status:** **dropped**.
- **Maps to:** E-23 and the sm90+/SM100 items.

## F-P1-12 Long context is verify-attention bound (scope note)

- **Priority:** P1 (negative for EAGLE-4).
- **Claim:** at ≈170k in mode E, attention is 117.7 ms of a 151 ms step (78%), and the drafter is 3.6 ms (2.4%). No drafter-side idea from EAGLE-4 can move E by more than that share. Trees add verify rows and make it worse.
- **Expected impact:** ≤ 2.4% ceiling for drafter levers at 170k (**estimated** from a **measured** profile). The real lever is verify attention, with KVarN stage 1 ≈ 40× its floor. That is outside EAGLE-4 (D3).
- **Evidence:** `baseline.md:53`; `transferability.md:197-198`.
- **Status:** **dropped** for this expedition. It is listed under deferred in the roadmap as the largest E-mode lever.
- **Maps to:** I-02, I-03.

## F-P1-13 Benchmark validity and baseline drift

- **Priority:** P1 (method).
- **Claim:** several gaps would make an A/B in this repo compare the wrong things.
  - Image/checkout skew: the box runs `sha-2cf50a0` launchers baked in, while our checkout at 95ac251 lacks b4b97ec and 373a695.
  - Silent PTH_FA fall-outs (F-P1-01).
  - The pre-373a695 `time.time()` WSL2 clock step (≈2.3 s every ≈28 s).
  - SSE streams accepted without `[DONE]` (PR 354).
  - A blank LABD corpus off `~/qwen-serving` (PR 353).
  - Two-boot baselines with uncommitted raw jsonl.
- **Expected impact:** no speed. It prevents a 9-22% apples-to-oranges error (**measured (reported)** magnitude of the PTH swing). [review (phase 5): at C1 on the box the swing was larger: `PTH_FA=0` read 63-75 tok/s vs about 100 with `PTH_FA=1`, about +33-58%, non-interleaved (`baseline.md:54,60-61`; `raw/prior-3090/baseline.md:63-65`). 9-22% is the long-context decode range.]
- **Evidence:**
  - `upstream-diff.md:90-93,103-110,128-129`;
  - `baseline.md:29,62` (TTFT and peak VRAM never recorded).
- **Effort:**
  - S for code: the fast-forward, cherry-picking PRs 353/354, and lifting `bench/paired_run.sh` and `bench/canary.sh` from `origin/master` (M if they need porting to `bench/harness.py`).
  - GPU time for baselines.
- **Risks:** `bench/README.md` one-row conflicts (354 × 358, `ctx-profile-sweep` × 354/358).
- **Status:** the code changes are **adopt now**. The re-baselines **need GPU runs** bench-1 to bench-12 (bench-9 is the interleaved PTH_FA=0/1 A/B). Pin `PTH_FA` in every arm (D10) and read the `[effective-config]` line.
- **Maps to:** D10, D12, D16, U-02, U-03, U-08.
