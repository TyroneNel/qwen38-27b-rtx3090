# EAGLE-4 expedition: final report

Qwen3.8-27B W4A16 on one RTX 3090 (sm86, 24 GB, WSL2). Can ideas from EAGLE-4 make it faster or better?

Phase 6, 2026-10-21. Branch `research/eagle4-expedition`. The research studied our project at `95ac251` (D4). After phase 6, `main` was fast-forwarded and this branch rebased onto `upstream/main` `63f7581` (D22). All paths below are relative to `docs/research/`.

**How to read the numbers (D19, [00-meta/decisions.md](00-meta/decisions.md)).** Every number carries one of three labels:

- **measured**: the raw files are in this repo, and a second agent re-derived the number;
- **measured (reported)**: a real GPU run whose raw files we do not hold (upstream PRs, the `cline/gf59cdmh` branch, the EAGLE-4 paper);
- **estimated**: our own arithmetic or a model.

**This expedition produced no measured speedup.** The research box has no GPU (D1). Every "measured" GPU number below comes from a prior run on the GPU box (D2). Where a reviewer wrote a `[review: …]` correction, this report uses the corrected value.

---

## 1. Executive summary

### What we did

- **Phases 0-5 ran on a CPU-only research box** (6 vCPU, 19 GB RAM, no CUDA; [00-meta/plan.md](00-meta/plan.md)). The work was:
  - map our project, upstream, EAGLE-4 and vLLM;
  - re-derive a baseline from the prior measured 3090 runs;
  - test transferability against the 24 GB memory gate;
  - run six CPU experiments;
  - write 31 findings and a roadmap.
- **Every GPU job is packaged for the GPU box.** That is 45 queue rows of at most 2 h each, bench-1 to bench-25 ([00-meta/gpu-queue.md](00-meta/gpu-queue.md), [04-benchmarks/gpu-agent-prompt.md](04-benchmarks/gpu-agent-prompt.md)). None has run yet. Every row in [04-benchmarks/LEDGER.md](04-benchmarks/LEDGER.md) is `todo`.
- **Each phase passed an independent review** ([00-meta/reviews/](00-meta/reviews/)). Phase 1 had 7 reviews, all PASS WITH CORRECTIONS. Phase 2 re-derived 24 baseline rows (several numbers each) with 0 disagreements. Phase 3 got ACCEPT WITH CHANGES. Phase 4 got CONFIRMED or READY, with fixes. Phase 5 got PASS WITH CORRECTIONS.

### Top 5 recommendations

| # | Recommendation | Expected gain (label) | Finding / roadmap | Queue rows |
|---|---|---|---|---|
| 1 | Fast-forward to `upstream/main` `63f7581`: **done (D22)** for `main` and this branch. This turns on int8 PTH KV by default for `SPEC=mtp CTX=long` | **+12-22% D decode at 66-93k vs the old checkout `95ac251`** (measured (reported), upstream PR #345); **0 vs the stock image**, which already has it | [F-P1-01](05-findings/p1-speed-inference.md), Q1 | bench-15, bench-9, bench-5 |
| 2 | Use int8 per-token-head KV (TRITON) on mode B through `EXTRA_ARGS`. Adopt it only after bench-16 confirms it | **KV pool 1.70×, N=8 preemptions 23→1, C1 a tie** (measured (reported), `cline/gf59cdmh` S-2) | [F-P1-02](05-findings/p1-speed-inference.md), Q2 | bench-16 |
| 3 | Gate correctness with a near-tie classifier on greedy divergences, using the EAGLE-4 method plus replay | **no speed**; a gate that keeps benign fp near-ties from being reported as real divergences | [F-P1-03](05-findings/p1-speed-inference.md), Q3 | bench-7, bench-14 |
| 4 | Load the EAGLE-4 drafter as EAGLE-3 (V3) and measure its acceptance. This is an information gate | **0 speed by itself**. Serving it pays off only at ≥ 1.12-1.31× DFlash2 tok/step (estimated break-even); low prior. Shipping is blocked on D11 | [F-P1-05](05-findings/p1-speed-inference.md), N6 | bench-17 |
| 5 | Run a fitted per-concurrency k table, checked only at the N=3 boundary | **about +5% at N=3 only** (estimated); 0-2% at N=4/8, which is a negative result (estimated) | [F-P1-06](05-findings/p1-speed-inference.md), N3 | bench-18 |

Items 1 and 2 are the only ones with a credible speed or capacity gain, and neither comes from EAGLE-4. Both are KV-format changes that already exist in upstream or on a fork branch. Item 1 also removed a trap: a baseline measured on the old checkout without `PTH_FA` pinned reads about 25-37% low at C1 on the box (63-75 against about 100 tok/s, so the pinned run reads 33-58% higher; measured (reported) magnitude, F-P1-13).

### Bottom line on EAGLE-4 for this box

**No EAGLE-4 technique is a measured win, or a near-certain one, on one RTX 3090.** The reasons are in [section 5](#5-eagle-4-lessons):

- **Tree verify is a no-go.** Each tree branch needs its own GDN state, about 1.2 GiB per 16 rows. Our kernels are causal-only. EAGLE-4's vLLM tree verify needs unquantised KV. vLLM has removed tree attention.
- **The `+eagle.4` vLLM fork is unpublished.** Moving to vLLM HEAD breaks 33 of our 51 patches.
- **The drafter fits only as a replacement for DFlash2, and only quantised.** Page padding takes back the memory it saves.
- **The low-rank head is worth only +1.7-3.1% (estimated) and is licence-gated.**
- **Long-context decode is bound by verify attention.** The drafter is 2.4% of the step at 170k (measured).

What does carry over is method: the near-tie gate, cost-table thinking about k, an arch-table fallback check, and one cheap acceptance probe (bench-17).

### Decisions needed from the human

From [05-findings/roadmap.md](05-findings/roadmap.md), "Decisions needed from the human":

1. **D11 licence.** Does shipping a low-rank candidate head, the EAGLE-4 draft, or weights derived from it count as a "Competing Offering"? This decides L1, L3 and L5. Local evaluation under §2(a) is allowed.
2. **Fast-forward target: resolved (D22).** `main` (local and origin) and `research/eagle4-expedition` now sit on `upstream/main` `63f7581`.
3. **Box `.env` default** (`CTX=fast VISION=1`). This matters for everyday serving only, because every queue arm already overrides it.
4. **Import policy** (refuse or fall back loudly). This is now moot for Q6, because the F5c patch works with "refuse".
5. **Inputs:**
   - a Chinese prompt set and a decision on the switch arm (bench-19);
   - V4 weights or a quant tool (bench-21);
   - `hidden.npy` (about 74 GB) and disk space (bench-24);
   - confirmation of the 6.2 GB EAGLE-4 download (bench-17).
6. *Optional:* the scratch allow-list edit for the PTH sub-arm on `CTX=fast` (≤ +5% C1, estimated).
7. *Optional:* record the closed PR #96 KV-sharing result as an upstream gotcha.

---

## 2. Current state

Full map: [01-project/architecture.md](01-project/architecture.md) (§2 has the mode table). Also see [01-project/dflash-specdec.md](01-project/dflash-specdec.md), [01-project/quantisation.md](01-project/quantisation.md), [01-project/heads-drafters.md](01-project/heads-drafters.md) and [01-project/inventory.md](01-project/inventory.md).

- **Model.** Qwen3.8-27B is a hybrid of 48 GDN (linear-attention) layers and 16 full-attention layers. The body is W4A16 Marlin. GDN state is fp16, 72 MiB per request row (estimated from the shapes, [02-eagle4/architecture.md](02-eagle4/architecture.md) §7).
- **Serving stack.**
  - Image: `ghcr.io/syv-ai/hyperqwen:sha-2cf50a0`.
  - Engine: vLLM 0.31.0 plus 51 patches.
  - Host: Windows 10 + WSL2 + Docker Desktop, 250 W, driver 610.88 ([00-meta/plan.md](00-meta/plan.md)).
- **Speculative decoding.**
  - **DFlash2:** a 5-layer block drafter that proposes 7 drafts plus lookup. It uses a 16×16 rank-256 edge score and a selector walk.
  - **MTP:** k=3, with a 40,960-row draft vocabulary.
  - **DYNAMIC_K:** `[[1,3,7],[4,512,3]]`.

| Mode | Launch | KV / attention | Notes |
|---|---|---|---|
| A | batch, 64 seats | fp8 KV, W4A8 | throughput mode, no spec |
| B | `SPEC=dflash2 CTX=fast` | bf16 KV, FLASH_ATTN | 7 drafts + lookup, 64k |
| C | B + `DFLASH_TOKENS=15` | as B | copy-heavy workloads |
| D | `SPEC=mtp CTX=long` | int8 pthfa (PTH_FA=1) on the stock image, upstream and this branch; the old checkout `95ac251` gave fp8 FlashInfer | skew closed by D22, see F-P1-01 |
| D′ | `SPEC=dflash2 CTX=long` | int8 PTH (default) | |
| E | `SPEC=dflash2 CTX=huge` | KVarN K4V2 | 240k context |

**Upstream state** ([01-project/upstream-diff.md](01-project/upstream-diff.md)):

- Upstream was 5 commits ahead of `95ac251`: `433b050`, `b4b97ec`, `373a695`, `2cf50a0` and `63f7581`. After phase 6, `main` and this branch moved to `63f7581` (D22).
- The fast-forward had 0 conflicts and touched 25 files (+444/−140).
- `b4b97ec` (#345) is the PTH_FA default that recommendation 1 adopts.
- Open PRs 353-366: 353/354 matter for benchmark validity (P1), 359 for quality (P2), and 366 re-vendors pthfa.

---

## 3. Baseline

Source: [04-benchmarks/baseline.md](04-benchmarks/baseline.md), re-derived from the byte-identical prior-run files in [04-benchmarks/raw/prior-3090/](04-benchmarks/raw/prior-3090/). Reviewer: [00-meta/reviews/phase2-harness-baseline.md](00-meta/reviews/phase2-harness-baseline.md). All values below are **measured** on the prior GPU box, with the stock image on WSL2, unless marked otherwise.

| Mode | Headline | Other measured values | Raw file |
|---|---|---|---|
| A | **814.3 tok/s e2e** (805.6-822.9), 64 conc | decode 904; KV pool 242,010 tok at `GPU_UTIL=0.93` | `raw/prior-3090/baseline.md`, `evals-10.md` |
| B | **114.4 tok/s C1 greedy decode**, 3.36 tok/step | e2e 110.4; KV pool 68,605 tok; aggregate N=1/2/4/8 = 118 / 182 / 253 / 236 (N=8 had 19 preemptions and is pool-limited) | `baseline.md`, `evals-10.md`, `evals-4.md` |
| B acceptance | T=0.7, positions 0/1/2/3/6 = 0.687 / 0.511 / 0.404 / 0.325 / 0.190 | 3.611 tok/step at T=0.7; 3.486 at T=0 | `spec-5.md` |
| B quality | ppl (all) 8.401 | GSM8K greedy n=200 95.5% | `evals-3.md` |
| C | **357.6 tok/s** labd `copy` at 20k, 14.22 tok/step | KV pool 57,669 tok | `baseline.md` |
| D | **99.8 tok/s C1**, 2.64 tok/step (resolved PTH_FA=1) | KV pool about 207k; aggregate N=1/2/4/8 = 105 / 192 / 308 / 423 | `baseline.md`, `evals-10.md` |
| D′ | **55.7 tok/s** at about 60k; **48.8** at about 95k | KV pool 136,429 tok | `baseline.md` |
| E | **57.8 tok/s** labd at 20k, 2.71 tok/step | KV pool 268,169 tok | `baseline.md` |
| E at 170k | **11.5 tok/s** cold, 1.86 tok/step | step 150.8 ms, of which attention 117.7 ms (78%) and drafter 3.6 ms (2.4%) | `e-profile.md` |
| D with PTH_FA=0 | **63-75 tok/s C1** on fp8 FlashInfer, host-bound | GPU 58-64% busy | `baseline.md` |

Reading notes, all from [04-benchmarks/baseline.md](04-benchmarks/baseline.md):

- **Native-Linux doc figures** (measured (reported); no raw files): A 942 (948 per the README), B 127.1, C 381, D 100.3, E 67. The 6-14% gap is put down to WSL2.
- **The prior D baseline already ran with PTH_FA=1**, because the stock image contains `b4b97ec`. A boot from the old checkout (`95ac251`) without `PTH_FA` lands at about 63-75. This is why D10 pins `PTH_FA` in every arm.
- **Thermal settling** cost −9.4% over about 4 min (62 → 83 °C). Cold short runs read high.
- **The boot-to-boot spread is about 1-2%.** This sets the win bars:
  - C1 needs ≥ +3% median over ≥ 3 interleaved boots, and ≥ 5 boots when the gain is under +5%;
  - N≥4 needs ≥ +5%.
- **Never measured, for any mode:** TTFT, peak VRAM (only KV pool tokens were recorded), decode at 8k/32k, and greedy spec-vs-non-spec equality. The re-baseline (bench-1 to bench-12) fills these gaps (F-P3-05).

---

## 4. Findings

There are 31 findings across three files. The tables below are summaries; claims, evidence and risks live in the linked files. Effort: S = small, M = medium, L = large.

### P1: speed and inference ([05-findings/p1-speed-inference.md](05-findings/p1-speed-inference.md))

| ID | Short claim | Expected impact (label) | Status |
|---|---|---|---|
| F-P1-01 | Fast-forward turns on int8 PTH for mtp/long | +12-22% D at 66-93k vs the old checkout (measured (reported)); 0 vs the stock image | done (D22); confirm with bench-15 |
| F-P1-02 | int8 TRITON KV on mode B | pool 1.70×, N=8 preemptions 23→1, C1 tie (measured (reported)) | bench-16 |
| F-P1-03 | Near-tie + replay divergence classifier | correctness gate, no speed | bench-7, bench-14 |
| F-P1-04 | Spec-timing variant image | tool; ≤ 0.7% overhead (estimated) | bench-13 |
| F-P1-05 | EAGLE-4 drafter as EAGLE-3 (V3) | 0 by itself; break-even 1.12-1.31× (estimated) | bench-17; D11 for shipping |
| F-P1-06 | Fitted k table vs DYNAMIC_K | 0-2% at N=4/8 (estimated, negative); about +5% at N=3 (estimated) | bench-18 (N=3 only) |
| F-P1-07 | Low-rank candidate head in DFlash2 | +1.7-3.1% B C1, ≤ 1.3% N=4 (estimated, negative) | dropped as speed; D11 |
| F-P1-08 | Device-side depth stop (pre-verify confidence stop) | ≤ 1-2% D C1 (estimated, negative) | dropped; X-06 has no tooling |
| F-P1-09 | Tree verify on our KV formats | no-go; about 1.2 GiB GDN state per 16 rows (estimated) | dropped |
| F-P1-10 | `+eagle.4` vLLM fork | no-go; 33 of 51 patches fail on HEAD | dropped |
| F-P1-11 | Hopper/Blackwell paths, L2 prefetch | ≈ 0 on sm86 (estimated) | dropped |
| F-P1-12 | Long-context E is verify-attention bound | ≤ 2.4% ceiling for any drafter gain at 170k (estimated from a measured profile) | scope note |
| F-P1-13 | Benchmark validity: drift, PTH_FA pinning, timing | avoids a 9-22% (long context) to about 33-58% (C1, box) wrong comparison (measured (reported) magnitude) | adopt (code) |

### P2: quants, heads and drafters ([05-findings/p2-quants-heads.md](05-findings/p2-quants-heads.md))

| ID | Short claim | Expected impact (label) | Status |
|---|---|---|---|
| F-P2-01 | MTP full head for Chinese | Chinese +89-127% vs the default slice (estimated from measured (reported) endpoints, 35.5 → 80.6 tok/s). **Already available today with `MTP_DRAFT_VOCAB=0`**; an auto-switch adds value only for mixed traffic (unquantified). The idea comes from an internal gotcha, not EAGLE-4 | bench-19 (blocked: no zh prompts) |
| F-P2-02 | int8 PTH KV in mode A | drift 1.7× lower (measured (reported), on TP=2 2×3080, so this is an extrapolation to mode A); speed ±3% (estimated) | bench-20 |
| F-P2-03 | W4A16 EAGLE-4 drafter (V4) | −0.34 GiB before page padding, about +0.6 GiB net cost after (estimated) | bench-21 (gated on bench-17); D11 |
| F-P2-04 | Shared-head / shared-embedding variants | design constraint | adopt as a constraint |
| F-P2-05 | SVD rank-128 lm_head recall gate | gate for F-P1-07 | bench-22; D11 |
| F-P2-06 | Body-only adaptation of the EAGLE-4 drafter | 0-10% tok/step (estimated, low prior) | bench-23 (gated); D11 |
| F-P2-07 | MTP TTT unroll 2 → 4 | 0-3% D tok/step (estimated, low prior) | bench-24 (needs `hidden.npy`) |
| F-P2-08 | DFlash2 2048 draft window | −12% tok/step at 12-36k (measured (reported)); recovery needs a retrain | deferred |
| F-P2-09 | Train an EAGLE-4-style drafter from scratch | about 53 h per 50M tokens (estimated) | dropped |
| F-P2-10 | Cross-layer KV sharing (closed PR #96) | GSM8K 0.94 → 0.56-0.64 for a 1.52× pool (measured (reported)) | dropped |
| F-P2-11 | EAGLE-4 sampled-acceptance fixes on int4 DFlash2 | 0 for a chain (estimated); int4 DFlash2 −5% under sampling (measured (reported)) | dropped |
| F-P2-12 | Quality-battery corpus fix (PR 359) | no speed; keeps the quant gate valid | adopt with upstream |

### P3: docs, robustness and tooling ([05-findings/p3-other.md](05-findings/p3-other.md))

| ID | Short claim | Impact | Status |
|---|---|---|---|
| F-P3-01 | Six doc-vs-code drift items (X-15) | docs only | adopt |
| F-P3-02 | One import hard-fail (F5c) without a GPU (X-16) | robustness; H16 mostly refuted | adopt (low priority); consistent with "refuse" |
| F-P3-03 | The SPEC default differs between launcher (`mtp`) and `.env.example` (`dflash2`) | none, intended | dropped |
| F-P3-04 | Harness and tool defaults lag the reviewed method | prevents false "real" rows and wrong-temperature rows | adopt |
| F-P3-05 | TTFT and peak VRAM never recorded | fills baseline gaps | bench-4 to bench-6, bench-11 |
| F-P3-06 | Upstream hygiene PRs 355-366 | none | take through upstream |

---

## 5. EAGLE-4 lessons

Sources:

- [02-eagle4/overview.md](02-eagle4/overview.md), [02-eagle4/architecture.md](02-eagle4/architecture.md), [02-eagle4/training-recipe.md](02-eagle4/training-recipe.md), [02-eagle4/build-and-kernels.md](02-eagle4/build-and-kernels.md);
- [02-eagle4/transferability.md](02-eagle4/transferability.md) and its calculator [02-eagle4/transferability-calc.py](02-eagle4/transferability-calc.py) / [.out](02-eagle4/transferability-calc.out);
- [03-vllm/integration-paths.md](03-vllm/integration-paths.md);
- review: [00-meta/reviews/phase3-transferability.md](00-meta/reviews/phase3-transferability.md).

### What EAGLE-4 claims

All of these are measured (reported) by the paper, on 4×B300. **The paper has no Ampere results.**

- Dynamic tree vs DFlash2: 1.28× at T=0 and 1.09× at T=1 (Table 2, p. 9).
- At BS1: 1.62× vs MTP. At 512 concurrency: 1.70× vs DFlash2. Overall: up to 3.59×.
- The Efficient LM Head (rank-128 retrieval followed by exact rescoring of K=128) cuts draft overhead by more than 40%.

The released drafter for our model is `EAGLE-Corp/Qwen3.8-27B-EAGLE4` (HF commit `f58a598`, 6,178,152,384 B):

- 3.09B parameters, 6.18 GB in bf16;
- embedding plus lm_head are 82.3% of the parameters;
- 4096-token sliding window.

### Why the headline does not transfer to Ampere / 24 GB

**1. The memory gate** (estimated; transferability.md memory-gate table).

| Variant | GiB | Effect at B |
|---|---|---|
| V1 (bf16 drafter added next to DFlash2) | 5.75 | +4.56 GiB over budget, or about 67k tokens of context lost |
| V3 (bf16, replaces DFlash2) | 3.33 | +2.13 GiB, about 27k tokens lost |
| V4 (W4A16, replaces DFlash2) | 0.86 | −0.34 GiB before padding |
| V7 (smallest variant) | 0.25 | changes the model or fails the loader (F-P2-04) |

- B has only 0.7-1.3 GiB free, and E about 0.15 GiB. 1 GiB of drafter costs about 16.4k tokens of context at B (estimated, [00-meta/plan.md](00-meta/plan.md)).
- **Page padding is the cost the phase-3 review found.** The draft page is 20,480 B/token, but it is padded to the 1,835,008 B primary page. That means +25% pages, about −20% pool, or about 0.9 GiB per request (estimated).
- Net effect: V4 becomes about +0.6 GiB net cost (estimated, F-P2-03). The drafter fits only as a quantised replacement, and even then it loses memory.

**2. Break-even** (estimated).

- To pay for itself at B, a W4A16 V4 drafter chain needs 3.75-4.40 tok/step against DFlash2's measured 3.36, i.e. a ratio of 1.117-1.31×.
- The bf16 E-20 path needs 5.92 / 8.37 tok/step, so it is not a speed path.
- The paper's own tree-vs-DFlash2 margin at T=1 (1.09×) is below this bar, and that margin is for the tree, which we cannot use.
- So bench-17 is an information gate with a low prior (F-P1-05).

**3. BF16-KV tree verify** (F-P1-09; integration-paths §7-8).

- EAGLE-4's vLLM tree verify needs unquantised KV (`known-limits.md:48`, `tree_attention.py:69-70`).
- That conflicts with fp8 (A), int8 PTH (D, D′) and KVarN (E).
- B and C already run bf16 KV, so "bf16 KV only" is not what blocks them. The no-go holds for all modes for four other reasons:
  - every tree row needs its own GDN state (72 MiB per row; about 1.1 GiB for 16 rows, about 1.2 GiB with the 2+k slots, estimated);
  - our attention kernels are causal-only;
  - vLLM has removed tree attention;
  - the effort is large, and the result is gated by D11.

**4. The unpublished `+eagle.4` fork** (F-P1-10).

- The EAGLE-4 engine needs vLLM `0.32.0rc1.dev283+eagle.4` with `eagle-hooks`. That build is not published.
- The `6465fb4` hooks are missing from the 0.1.0 engine release.
- Neither vLLM 0.31 nor HEAD `4bd75e9` supports EAGLE-4.
- Moving to HEAD fails 33 of our 51 patches.
- The fork ships no sm86 SASS (sm80 SASS runs).
- It also clashes with pthfa over the CUSTOM backend slot. That clash is moot under PTH_FA, which forces int8 KV.
- The one path that works on our stack is chain mode, loading the drafter as EAGLE-3 through the registry. That is bench-17.

**5. The licence (D11)** ([02-eagle4/overview.md](02-eagle4/overview.md), licence section).

- §2(a) allows free evaluation.
- §1(e) covers the technology, not only the weights.
- §3(a)-(b) is the "Competing Offering" clause, with a US$20M revenue cap.
- §6(a) gives no patent grant for a re-implementation.
- Consequences: shipping the draft, its quantised or adapted weights (L1, L3), or a low-rank head modelled on EAGLE-4 (L5) all need a human decision. Licence-gated items are never quick wins (D11).

**6. Where the time goes** (F-P1-12).

- At 170k in mode E, the drafter is 3.6 ms of a 150.8 ms step (measured, `e-profile.md`), while attention is 78%.
- No drafter change can move long-context decode by more than about 2.4% (estimated).

### What we learned anyway

- **The near-tie / ulp-band gate** (EAGLE-4's `fp32_arbiter`, 2-ulp tolerance) became our divergence classifier (X-08). D18 sets the bands: near-tie < 1e-2 nats, replay 1e-2 to 0.25, real > 0.25.
- **Cost-table thinking** shows that DYNAMIC_K already sits at the cost-model optimum for N ≥ 4 (X-02a). The only gap left is the N=3 boundary.
- **Asking "would this hard-fail on a new arch?"** led to X-16. It found one real gap (F5c) and a 22-line guard for it.
- **The draft-window cost is real for us too.** DFlash2's 2048 window costs −12% tok/step at 12-36k (measured (reported), F-P2-08). bench-17 measures EAGLE-4's 4096 window per context bucket.
- **The tree-vs-chain gap shrinks under sampling** (1.28× at T=0 to 1.09× at T=1, measured (reported)). Our chain-only stack therefore loses less than the headline suggests.
- **Verdict counts** for E/T/U/I items are in the verdict tables in [02-eagle4/transferability.md](02-eagle4/transferability.md).

---

## 6. Upstream gaps

Details: [01-project/upstream-diff.md](01-project/upstream-diff.md), reviewed by [00-meta/reviews/phase1-upstream-diff.md](00-meta/reviews/phase1-upstream-diff.md).

| Gap | Evidence (label) | Action |
|---|---|---|
| The old checkout `95ac251` lacked the #345 PTH_FA default (`b4b97ec`) | upstream: +9% native at 1 boot; +12.2% / +14.0% native at 2 boots; +21.3% / +22.1% WSL2 at 3 boots; GSM8K 0.9507 vs 0.9522; pool −1.5% (all measured (reported)) | closed by the Q1 fast-forward (D22); bench-15 confirms on the box |
| The image was ahead of the checkout (image `2cf50a0`, checkout `95ac251`). Since D22 the checkout is `63f7581`, which differs from the image in docs only | every prior measured baseline came from the image (D2) | closed; D10 still pins PTH_FA; N1 re-baselines on the fast-forwarded tree |
| The int8 pool can refuse `MAX_LEN` 150000 | upstream-diff divergences | queue rows set context explicitly |
| Open PRs 353/354 (benchmark validity), 359 (corpus), 366 (pthfa re-vendor) | upstream-diff, open-PR table | take with upstream (F-P1-13, F-P2-12, F-P3-06) |
| Fork branches: `dynamic-k-results`, `cline/gf59cdmh` (the int8 TRITON on B evidence), `ctx-profile-sweep`, `master` (`paired_run.sh`, `canary.sh`) | read-only via `git show` | the S-2 rows feed Q2 |
| Closed PR #96 KV sharing | GSM8K drop (measured (reported)) | record as a gotcha (decision 7) |

---

## 7. Experiments

### CPU experiments (done on the research box)

These results are measured only for logic, resolver behaviour or code reading. None is a speed measurement. Review of the X-01a, X-02a, X-08, X-15 and X-16 rows: [00-meta/reviews/phase4-cpu-experiments.md](00-meta/reviews/phase4-cpu-experiments.md). Review of X-17: [00-meta/reviews/phase4-x17-timing.md](00-meta/reviews/phase4-x17-timing.md).

| ID | What | Result | Verdict |
|---|---|---|---|
| [X-01a](04-benchmarks/experiments/X-01a/README.md) | Resolve `SPEC=mtp CTX=long` on upstream `63f7581` vs ours `95ac251`, with stub GPUs (cc86, cc89, none) | upstream resolves `PTH_FA=1`. Ours leaves both `PTH_FA` and `FA2_FP8KV` unset, i.e. fp8 KV on FlashInfer, not FA2_FP8KV as first assumed (`run.out`) | PASS with a correction; review CONFIRMED (byte-identical re-run). Speed is a separate question: bench-15 |
| [X-02a](04-benchmarks/experiments/X-02a/README.md) | Fit a per-concurrency k table from a cost model and back-test it against measured DYNAMIC_K | the back-test passes with the ladder-matched acceptance profile (+9.5% predicted vs +7.7% measured (reported)) and fails with the spec-5 profile (−3.3%). Fitted table `[[1,1,7],[2,2,5],[3,3,4],[4,512,3]]`: 0-2% at N=4/8 after the smooth-tail review (estimated); +4.9% at N=3 (estimated, interpolated) | H02′ not plausible at N=4/8; review CONFIRMED WITH CORRECTIONS (temperature is not the cause; the back-test is a consistency check). bench-18 tests only N=3 |
| [X-08](04-benchmarks/experiments/X-08/README.md) | Near-tie + replay divergence classifier (`classify_divergence.py`, `record_top2.py`, `rescore_replay.py`) | 14/14 CPU tests pass (`test.out`); the ulp-band default is now 0.25, per D18 and EAGLE-4's arbiter | review CONFIRMED WITH CORRECTIONS; GPU run is bench-14 |
| [X-15](04-benchmarks/experiments/X-15/README.md) | Doc-vs-code drift | six items confirmed with file:line (lookup NSTRONG/AGREE 6/0; the README FA2 default; the SPEC default difference; GDN state 0.738 / 1.494 GiB and 2+k slots; MTP 0.475 / 0.381; the docker-compose header and other fp8 lines) | review CONFIRMED WITH CORRECTIONS; the 15.8% vs 14.2% figure is still open |
| [X-16](04-benchmarks/experiments/X-16/README.md) | Probe 33 import paths without a GPU for hard-fails | before the patch: 19 OK / 4 SILENT / 10 HARD. One genuine import-time hard-fail, F5c (`fa2 backend.py:27`, missing `.so`). After `fallback.patch` (3 files, +22/−7): 21 / 4 / 8 | review CONFIRMED; H16 mostly refuted |
| [X-17](04-benchmarks/experiments/X-17-timing/README.md) | Spec-timing variant image: six marks behind `HQ_SPEC_TIMING`, plus a parser | 51/51 patches apply, plus the 4 KVarN patches; overhead 0.4-0.7% (estimated; about 10-20 µs per step) | review READY FOR GPU after fixes F1-F3 and the README fix F4; GPU run is bench-13 |

Harness self-check: 18 ok, `SELFCHECK PASS` ([04-benchmarks/harness/selfcheck.out](04-benchmarks/harness/selfcheck.out)).

### GPU experiments (queued, not run)

The GPU agent gets [04-benchmarks/gpu-agent-prompt.md](04-benchmarks/gpu-agent-prompt.md). The order is in [00-meta/gpu-queue.md](00-meta/gpu-queue.md), and status goes into [04-benchmarks/LEDGER.md](04-benchmarks/LEDGER.md). There are 45 rows, each at most 2 h. Several rows are split into lettered parts; this report names rows by their bench number only.

**Box-hour estimate** (estimated, gpu-agent-prompt.md "Total box-hour estimate"):

| Block | Box hours (estimated) |
|---|---|
| P1 unconditional (bench-13 to bench-18) | 19.45 |
| P2 unconditional (bench-19, 20, 22, 24) | 7.6 |
| **Expected total** | **about 27** (about 26.3 GPU-hours) |
| Conditional and gated (parts of 15, 16, 18, 24; and 21, 23, 25) | 11.7 |
| **Upper bound** | **about 39** |
| Not queued (proposals) | X-14 full training about 5.2 GPU-h; X-13 full run about 2.5 GPU-h and 41 GB of disk |

The re-baseline rows bench-1 to bench-12 come first and are not in the table above.

| Rows | Job | Closes |
|---|---|---|
| bench-1, 2, 3 | environment snapshot, smoke test, prior-timing check | D13 box facts; timing validity |
| bench-4, 5, 6, 11 | B, D, E, D′, C and A baselines, two boots each (D16) | N1; TTFT and VRAM gaps |
| bench-7, 8 | greedy equality, quality | Q3 input; quality gate |
| bench-9 | PTH_FA 0 vs 1 interleaved, mtp and dflash2 at long | gap 7; feeds Q1 |
| bench-10 | profile of D′ | drafter share at depth |
| bench-12 | spec-off C1 | reference arm |

| Row | Experiment | Hypothesis / finding | Gate or blocker |
|---|---|---|---|
| bench-13 | X-17 spec-timing | F-P1-04 | overhead gate |
| bench-14 | X-08 classifier, main + KVarN | F-P1-03 | none |
| bench-15 | X-01b PTH_FA=1 vs 0 on mtp/long at depth | F-P1-01 | GSM8K boots only if the decode gate is met |
| bench-16 | X-04′ int8 TRITON on B | F-P1-02 | quality boot only if the pool and C1 gates are met |
| bench-17 | X-03 EAGLE-4 as EAGLE-3 | F-P1-05 | 6.2 GB download; pre-gate |
| bench-18 | X-02b′ k table at N=3 | F-P1-06 | rounds 4-5 only in the 3-5% band |
| bench-19 | X-09 MTP full head vs slice, Chinese | F-P2-01 | blocked: no zh prompts |
| bench-20 | X-10 int8 vs fp8 in mode A | F-P2-02 | none |
| bench-21 | X-11 V4 vs V3 | F-P2-03 | gated on a bench-17 WIN and V4 weights |
| bench-22 | X-12 SVD recall (CPU on the GPU box) | F-P2-05 | D11 for any use |
| bench-23 | X-13 body adaptation pilot | F-P2-06 | gated; blocked without tooling. Bar is a same-run ratio, 1.008 ≤ r < 1.12 |
| bench-24 | X-14 TTT unroll pilot | F-P2-07 | needs `hidden.npy` |
| bench-25 | X-06 depth stop | F-P1-08 | blocked: no tooling |

Gate rules apply to every row ([00-meta/decisions.md](00-meta/decisions.md)):

- **D15 spill rule:** flag growth over 64 MiB. Abort on OOM, on growth over 256 MiB, or when C1 is more than 10% below the prior.
- **D10:** pin PTH_FA in every arm.
- **D12:** the timing patch lives only in a variant image.
- **D17:** the GPU box fetches the branch after the phase-6 push.

The GPU hand-off review ([00-meta/reviews/phase4-gpu-handoff.md](00-meta/reviews/phase4-gpu-handoff.md)) returned READY WITH FIXES. Its three HIGH fixes covered the X-08 paths, the X-12 bf16 lm_head `.bak`, and the X-14 paths.

---

## 8. Roadmap

Full roadmap with dependencies and the deferred table: [05-findings/roadmap.md](05-findings/roadmap.md).

### Quick wins

| Item | Gain (label) | Rows |
|---|---|---|
| Q1 Fast-forward to `63f7581`: **done (D22)** | +12-22% D at 66-93k vs the old checkout (measured (reported)); 0 vs the stock image | bench-15, 9, 5 |
| Q2 int8 TRITON KV on mode B, adopted after bench-16 | pool 1.70×, N=8 preemptions 23→1, C1 tie (measured (reported)) | bench-16 |
| Q3 Correctness gate | gate, no speed | bench-7, 14 |
| Q4 Benchmark-validity fixes (PTH_FA pinning, timing, harness defaults) | avoids a 9-22% to about 33-58% wrong comparison (measured (reported) magnitude) | code; bench-2 smoke |
| Q5 Doc drift fixes | docs only | none |
| Q6 F5c import guard | robustness only | none |

### Next sprint

| Item | Gain (label) | Rows |
|---|---|---|
| N1 Re-baseline on the fast-forwarded tree | reference arms; TTFT and peak VRAM | bench-1 to 12 |
| N2 Spec-timing variant | tool; ≤ 0.7% overhead (estimated) | bench-13 |
| N3 Fitted k table at N=3 | about +5% at N=3 only (estimated) | bench-18 |
| N4 Chinese full-head switch | +89-127% on Chinese vs the slice (estimated); already reachable with `MTP_DRAFT_VOCAB=0` | bench-19 |
| N5 Mode A int8 KV | drift 1.7× lower (measured (reported)); speed ±3% (estimated) | bench-20 |
| N6 EAGLE-4 as EAGLE-3 probe | 0 by itself; break-even 1.12-1.31× (estimated) | bench-17 |

### Larger bets

| Item | Gain (label) | Rows | Gate |
|---|---|---|---|
| L1 W4A16 EAGLE-4 drafter | likely a net memory loss after padding (estimated) | bench-21 | bench-17 WIN; D11 |
| L2 MTP TTT unroll 2 → 4 | 0-3% D tok/step (estimated) | bench-24 | `hidden.npy` |
| L3 Body-only adaptation | 0-10% tok/step (estimated) | bench-23 | tooling; D11 |
| L4 DFlash2 window retrain | could win back up to the −12% at 12-36k (measured (reported) loss) | none queued | no DFlash2 recipe |
| L5 SVD factors, then a low-rank candidate head | +1.7-3.1% B C1 (estimated) | bench-22, then X-05 (not queued) | D11 |
| L6 Device-side depth stop | ≤ 1-2% D C1 (estimated) | bench-25 | tooling |

Roadmap counts: 6 quick wins, 6 next-sprint items, 6 larger bets. Of the 13 corrections noticed in phase 5, 7 were applied, 2 deferred and 4 rejected (roadmap.md, "Corrections noticed").

---

## 9. Risks and open questions

**Measurement risks**

- **The box may not reproduce the prior baseline.** D15 aborts a row when C1 is more than 10% below the prior. WSL2 runs 6-14% below native, and cold runs read up to about 9% high, so a careless comparison can create or hide any gain on the roadmap (F-P1-13).
- **Checkout/image skew (closed by D22).** Before the fast-forward, a "D" boot could mean fp8 FlashInfer (63-75 tok/s) or int8 pthfa (about 100 tok/s), depending on the source tree. A box checkout older than `63f7581` still has this skew, so D10 still pins `PTH_FA`, and every report row must record it.
- **Small gains sit near the noise.** The boot spread is 1-2%. N3 (about +5%), L2 (0-3%), L5 (+1.7-3.1%) and L6 (≤ 1-2%) are at or below the win bars. Expect NEUTRAL verdicts.
- **N3 rests on interpolation.** N=3 was never measured, and V(N=3, k) is interpolated between N=2 and N=4. A step between them could erase the gain (X-02a).
- **Timing.** Since D22 our checkout includes `373a695`, so its `labd_bench.py` uses `perf_counter`; the old `95ac251` copy used `time.time()`. Whether the prior box's copy used `perf_counter` is still open; bench-3 settles this.

**Evidence risks**

- **Recommendations 1 and 2 rest on measured (reported) numbers whose raw files we do not hold** (upstream #345; `cline/gf59cdmh` S-2). The N=8 aggregate tok/s gain for int8 TRITON on B is not quantified; only the preemption and pool numbers are.
- **F-P2-02 drift** comes from a TP=2 2×3080 run. Applying it to mode A on one 3090 is an extrapolation.
- **The Chinese gain** is derived from endpoints, not from an A/B on this box. The value of the auto-switch for mixed traffic is unquantified.

**Open questions**

1. D11, the licence. This blocks shipping L1, L3 and L5, and any use of bench-22's output.
2. Will the EAGLE-4 drafter reach ≥ 1.12× DFlash2 tok/step on our prompts? Only bench-17 can say. The prior is low.
3. How large are TTFT and peak VRAM per mode? Never measured; filled by N1.
4. X-15: the 15.8% vs 14.2% figure is still unresolved.
5. The `greedy_equal.py` default near-tie threshold is still 0.1. The GPU prompt passes `--near-tie 0.01` (D18) on bench-2 and bench-7, and the classify default is already 0.25 (F-P3-04).
6. The drafter's share of the step at D′ depth (bench-10). This decides whether any drafter work is worth doing at 60-95k.
7. Missing inputs: zh prompts, V4 weights or a quant tool, `hidden.npy` plus disk, and tooling for X-06/X-13. Until they arrive, bench-19, 21, 23, 24 and 25 stay blocked or gated.

---

## 10. Appendix

### Sources and pinned commits

From [00-meta/plan.md](00-meta/plan.md); the drafter row is from [02-eagle4/overview.md](02-eagle4/overview.md). All sources were opened read-only.

| Source | Commit |
|---|---|
| our project (HyperQwen checkout), as studied | `95ac251bc3b71017c883e83004cf4fb188d066a4`; branch rebased onto `63f7581` after phase 6 (D22) |
| upstream `main` | `63f7581804afe05f23f240c2f584a00ac18c488b` |
| EAGLE-4 repo | `7b7b5a522d7a7284b9eeb9889d03c8c2038c112f` (tag v0.1.0 = `75436712`) |
| EAGLE-4 drafter (HF) | `EAGLE-Corp/Qwen3.8-27B-EAGLE4` @ `f58a598` |
| vLLM | `4bd75e952a9f423742223c97c8cc9fc6db362950` (v0.32.1rc0-68); serving base 0.31.0 |
| deepinfer-ref (prior runs; context only, D3) | `86ce6edb4d2b5c5d6e1b09cd9e2f9fde73070a1f` |
| stock image | `ghcr.io/syv-ai/hyperqwen:sha-2cf50a0` |

Outside ideas came from EAGLE-4 only. DeepInfer notes served as context (D3); the overlap IDs are listed in [05-findings/roadmap.md](05-findings/roadmap.md). Prior-run provenance and hashes: [04-benchmarks/raw/prior-3090/PROVENANCE.md](04-benchmarks/raw/prior-3090/PROVENANCE.md).

Environments:

- **Research box:** Linux 6.17.13-2-pve, 6 vCPU, 19 GB RAM, Python 3.13.7, torch 2.12.1+cu130 (CPU only).
- **GPU box:** RTX 3090, driver 610.88, Windows 10 + WSL2.

### Method

- **Agents.** Every agent was spawned as `general-purpose` with an explicit role (D5), and effort was set per call (D9). Sonnet ran mapping, diffing, tracing, bench engineering and the experiments. Opus ran analysis and every review. Spawn log: [00-meta/agent-log.md](00-meta/agent-log.md).
- **Reviews.** Each phase output was reviewed by an independent agent that re-ran the scripts and traced the claims ([00-meta/reviews/](00-meta/reviews/)):
  - phase 1: 7 reviews;
  - phase 2: re-derivation, 0 disagreements;
  - phase 3: adversarial challenge;
  - phase 4: CPU-experiment, X-17 and GPU hand-off reviews;
  - phase 5: 23 claims traced.

  `[review: …]` marks win over the text they annotate.
- **Decisions.** D1-D22 are in [00-meta/decisions.md](00-meta/decisions.md). The ones that shape this report:
  - D1/D2: no GPU, so the baseline is the prior runs;
  - D3: DeepInfer is context only;
  - D10: pin PTH_FA;
  - D11: licence decisions go to the human;
  - D15: spill and abort rule;
  - D16: two boots per baseline;
  - D18: near-tie bands;
  - D19: evidence labels.
- **Hypothesis ranking** (phase 3, reviewed): H01, H04′, H08, H03, H02′, H09, H10, H11, H12, H13, H14, H06, H15, H16. H05 is unranked ([00-meta/plan.md](00-meta/plan.md)).

### Evidence labels (D19)

| Label | Meaning | Example in this report |
|---|---|---|
| measured | the raw files are in this repo and a second agent re-derived the value | B 114.4 tok/s C1 (`raw/prior-3090/baseline.md`) |
| measured (reported) | a real GPU run whose raw files we do not hold | upstream #345 +12-22%; cline S-2 pool 1.70×; EAGLE-4 paper 1.28× |
| estimated | our arithmetic or model | memory gate, break-even 1.12-1.31×, X-02a +5% at N=3 |

CPU experiment results are measured for logic and code behaviour only. No GPU speed number in this report was produced by this expedition.
