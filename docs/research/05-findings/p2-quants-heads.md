# P2 findings: quants, heads and drafters

Labels and win bars as in `p1-speed-inference.md` (D19, `00-meta/decisions.md:23`). No speedup here is **measured**. D11 (the licence "Competing Offering" question) gates shipping anything derived from the EAGLE-4 draft or its factor-head technology. Local evaluation under licence §2(a) is allowed.

Memory context (`00-meta/plan.md:89`, reviewed):
- usable CUDA total 23.54 GiB;
- B has 0.7-1.3 GiB free, E ≈ 0.15 GiB;
- 1 GiB of drafter costs ≈ 16.4k tokens of context at B and ≈ 32k at D′;
- DFlash2 W4A16 is 1.19 GiB, MTP 0.30 GiB.

## Summary

| ID | Claim | Expected impact (label) | Effort | Status | Maps to |
|---|---|---|---|---|---|
| F-P2-01 | Switching MTP to the full head on low draft-vocab coverage fixes Chinese decode | Chinese +89-127% vs the default slice (estimated from measured (reported) endpoints) [review: already available today via `MTP_DRAFT_VOCAB=0`; the switch's own gain is only for mixed traffic, unquantified]; English 0 | M + tooling | needs GPU run X-09 (blocked: no zh prompts, no switch arm) | H09, I-10 |
| F-P2-02 | int8 per-token-head KV in mode A matches fp8 speed with less drift | drift 1.7× lower (measured (reported)); speed ±3% (estimated) | M | needs GPU run X-10 | H10, I-09 |
| F-P2-03 | A W4A16 EAGLE-4 drafter (V4) saves memory on paper, but page padding takes it back | −0.34 GiB vs DFlash2 before padding, ≈ +0.6 GiB net cost after (estimated) | S/M + tooling | needs GPU run X-11 (gated on X-03 WIN); shipping blocked on D11 | H11, E-20 |
| F-P2-04 | Shared-head and shared-embedding variants change the model or fail the loader | n/a (design constraint) | S | adopt now as a constraint on X-03/X-11 | E-20, V2/V6/V7 |
| F-P2-05 | SVD rank-128 factors of our int4 lm_head: recall@128 gate | gate for F-P1-07 and a no-hole MTP head | S/M | blocked on D11 (X-12) | H12, T-01, T-03, T-04, T-09 |
| F-P2-06 | Body-only domain adaptation of the EAGLE-4 drafter | 0-10% tok/step (estimated, low prior) | M + tooling | needs GPU run X-13 (gated, no tooling); shipping blocked on D11 | H13, T-05 |
| F-P2-07 | Raising the MTP TTT unroll from 2 to 4 | 0-3% D tok/step (estimated, low prior) | M | needs GPU run X-14 (blocked: no `hidden.npy`) | H14, T-06 |
| F-P2-08 | The DFlash2 2048 draft window costs tok/step at 12-36k | −12% tok/step at 12-36k (measured (reported)); recovery needs a retrain | L | deferred (no DFlash2 recipe); X-03 measures the 4096 window per bucket | I-12, E-11, T-08 |
| F-P2-09 | Training an EAGLE-4-style drafter from scratch (no-go) | ≈53 h per 50M tokens (estimated) | L | dropped | T-xx (from scratch) |
| F-P2-10 | Cross-layer KV sharing (closed PR #96, no-go) | GSM8K 0.94 → 0.56-0.64 for a 1.52× pool (measured (reported)) | — | dropped | U-07 |
| F-P2-11 | Sampled-acceptance fixes from EAGLE-4 do not help int4 DFlash2 (negative) | 0 for a chain (estimated); int4 DFlash2 −5% under sampling (measured (reported)) | — | dropped | I-11, E-15 |
| F-P2-12 | Quality-battery corpus fix (PR 359) keeps "all" PPL a fixed mixture | no speed; quant-gate validity | S | adopt now (with upstream) | U-06 |

---

## F-P2-01 MTP full head on Chinese

- **Priority:** P2.
- **Claim:** the MTP 40,960-row draft vocab covers 6.4% of Chinese tokens, which gives 1.06 tok/step. Switching to the full head when coverage drops brings it back to ≥ 2.0 tok/step, with English D C1 within 2%.
- **Expected impact:** Chinese decode +100-127%: 35.5 → 80.6 tok/s at the endpoints, **measured (reported)** (`heads-drafters.md:151`). The gain of a switch is **estimated** from those two endpoints. English is 0.
  [review (phase 5): (1) The lower bound has no derivation; from the claim's own ≥ 2.0 tok/step it is 2.0 / 1.06 = +89%, and +127% is 80.6 / 35.5 (`01-project/heads-drafters.md:151`, from `docs/gotchas.md:1436-1472`, one user's 3090, issue #196, CTX=long k=3). (2) The full-head endpoint already ships: `MTP_DRAFT_VOCAB=0` falls back to the shared full lm_head (`patches/qwen3_5-mtp-draft-vocab.patch:8,40`) and is documented for Chinese (`single-user/README.md:438`). So +89-127% is the gain of a documented setting over the default, available now for Chinese-only traffic. What the switch adds is only for mixed-language traffic on one boot, and that gain is not quantified anywhere.]
- **Evidence:** `02-eagle4/transferability.md:205`; `00-meta/plan.md:47`; `04-benchmarks/gpu-agent-prompt.md:997-1014,1162`.
- **Effort:** M, plus two missing pieces:
  - a Chinese prompt set (`$HQ/zh_prompts.jsonl`, which the harness must not write itself);
  - the coverage-triggered switch arm, which exists nowhere.
- **Risks:**
  - It is not known whether the full head is the target's shared copy or a second resident copy, so the memory cost is open. [review (phase 5): the patch comment says the fallback is the shared full lm_head (`qwen3_5-mtp-draft-vocab.patch:8`), so no second copy is expected; not verified at load.]
  - English must not regress.
- **Status:** **needs GPU run X-09** (bench-19). Blocked until zh prompts exist. Without the switch arm it can confirm the two endpoints but never WIN.
- **Maps to:** H09, I-10. Internal, not EAGLE-4. [review (phase 5): confirmed; the source is gotcha 61, not the EAGLE-4 paper or repo. The only EAGLE-4 link (T-01 factor head) is D11-gated.]

## F-P2-02 Mode A int8 vs fp8 KV

- **Priority:** P2.
- **Claim:** int8 per-token-head KV in batch mode A keeps the fp8 throughput (814.3 tok/s e2e, **measured**, `04-benchmarks/baseline.md:33`) within 3% and drifts less.
- **Expected impact:**
  - Quality: fp8 drifts 0.0360 nats vs int8 0.0207/0.0219 at 32k, 1.7× less (**measured (reported)**). fp8 fails the repo's non-inferiority test (`docs/long-context.md:466-476`).
  - Speed: ±3% (**estimated**).
- **Evidence:** `transferability.md:204`; `plan.md:48,90`. [review (phase 5): the drift figures come from `docs/long-context.md:466-476`, a TP=2 2×3080 PTH context run at CTX=long, not mode A; applying them to mode A is an extrapolation, which bench-20 measures.]
- **Effort:** M.
  - The int8 arm must use TRITON. The batch launcher has no `PTH_FA`, and the single-user launcher refuses PTH outside its allow-list (`start_qwen.sh:297-304`).
  - The drift tool is glue (`gpu-agent-prompt.md`, bench-20).
- **Risks:** the int8 pool is −1.5-1.9%, so check the boot near `GPU_UTIL` edges.
- **Status:** **needs GPU run X-10** (bench-20). Kill: speed < −3%, or drift ≥ 0.025 nats.
- **Maps to:** H10, I-09. Internal.

## F-P2-03 W4A16 EAGLE-4 drafter (V4) and page padding

- **Priority:** P2.
- **Claim:** if X-03 wins, a W4A16 EAGLE-4 layer with an int4 own head (≈0.86 GiB) keeps tok/step within −3% of bf16. On paper it is 0.34 GiB lighter than DFlash2 at B/D′/E.
- **Expected impact:** −0.34 GiB before padding (**estimated**). The draft page, 20,480 B/token per layer (5× the target's), does not divide the 1,835,008 B primary page. That costs ≈ +25% pages (≈ −20% pool, ≈0.85 GiB at B) with promote, or ≈0.9 GiB per request without. Net, V4 likely costs ≈0.5-0.6 GiB more than DFlash2 (**estimated**).
- **Evidence:** `00-meta/reviews/phase3-transferability.md:105-109`; `transferability.md:62-106`; `plan.md:49`.
- **Effort:** S/M for the run. No V4 quant tool exists (`gpu-agent-prompt.md` harness gaps).
- **Risks:**
  - The quantised draft head may lose more than 3% tok/step.
  - It interacts with the F-P1-02 int8 KV page sizes.
- **Status:** **needs GPU run X-11** (bench-21). It is gated on an X-03 WIN and on V4 weights plus a quant tool. Shipping is **blocked on D11**. Kill: X-03 LOSS, or net pool below DFlash2's after padding.
- **Maps to:** H11, E-20.

## F-P2-04 Shared-head and shared-embedding variants

- **Priority:** P2.
- **Claim:**
  - The EAGLE-4 draft trains its own lm_head (`tie_word_embeddings: false`). The shared-head variants (V6/V7) therefore change the model and are not the released drafter.
  - V2/V3 embedding sharing needs `torch.equal` against our int8 `weight_packed` embedding, which cannot pass. So V3 drops `embed_tokens` and needs a loader path, or the draft costs the full 6.18 GB.
  - Embedding + lm_head are 82.3% of the 3.09B params.
- **Expected impact:** none. This is a design constraint that saves a wrong experiment.
- **Evidence:** `transferability.md:97,240`; `plan.md:84-85,95-96`.
- **Effort:** S. A load-only pre-gate inside X-03.
- **Status:** **adopt now** as a constraint on X-03/X-11. Do not run V6/V7 as "EAGLE-4".
- **Maps to:** E-20, memory-gate variants V2/V6/V7.

## F-P2-05 SVD recall@128 on our lm_head

- **Priority:** P2.
- **Claim:** SVD-initialised rank-128 factors of our int4 lm_head reach recall@128 ≥ 91.5% (the paper's SVD bar) on captured states, and ≥ 98% after training.
- **Expected impact:** a gate. It decides F-P1-07 (+1.7-3.1% C1, **estimated**) and a no-hole MTP head.
- **Evidence:** `plan.md:50`; `transferability.md:143` (E-01 legal gate).
- **Effort:** S/M. CPU only, but it needs the weights and captured hidden states, so it runs on the box.
- **Risks:** the factor head is EAGLE-4 technology. Building or shipping it is the D11 question.
- **Status:** **blocked on D11** (X-12, bench-22). It is never a quick win.
- **Maps to:** H12, T-01, T-03, T-04, T-09.

## F-P2-06 Body-only domain adaptation

- **Priority:** P2.
- **Claim:** if X-03 lands within 10% of break-even, adapting only the drafter body (P-c, ≈25-40 min, **estimated**) lifts tok/step to the X-03 ratio bar.
- **Expected impact:** 0-10% tok/step (**estimated**, low prior). Our own MTP fine-tune was flat (**measured (reported)**, `drafter/README.md:29-37`).
- **Evidence:** `transferability.md:172`; `plan.md:51`; `gpu-agent-prompt.md:1101`.
- **Effort:** M. No tooling exists under `experiments/X-13/`.
- **Risks:** derived weights are covered by D11.
- **Status:** **needs GPU run X-13** (bench-23). Gated on X-03 being within 10% of break-even and on tooling. Shipping is **blocked on D11**.
- **Maps to:** H13, T-05.

## F-P2-07 MTP TTT unroll 2 → 4

- **Priority:** P2.
- **Claim:** a deeper test-time-training unroll in the MTP trainer raises D tok/step by ≥ 3%.
- **Expected impact:** 0-3% (**estimated**, low prior; the depth-2 result was flat).
- **Evidence:** `transferability.md:173`; `plan.md:52`.
- **Effort:** M: train, export, serve.
- **Status:** **needs GPU run X-14** (bench-24). Blocked without `hidden.npy` (≈74 GB) and the model dir.
- **Maps to:** H14, T-06.

## F-P2-08 DFlash2 draft window (2048)

- **Priority:** P2.
- **Claim:** DFlash2's 2048-token draft window loses tok/step at mid context. EAGLE-4 uses a 4096 window.
- **Expected impact:** −12% tok/step at 12-36k (**measured (reported)**, `dflash-specdec.md:459`). Recovering it needs a DFlash2 retrain at the served window.
- **Evidence:** `transferability.md:57,149,175,207`.
- **Effort:** L, and there is no DFlash2 training recipe.
- **Status:** **deferred**. X-03 already records tok/step per context bucket (1.5k, 8k, 12-16k), so it shows whether EAGLE-4's 4096 window closes this gap at no extra cost.
- **Maps to:** I-12, E-11, T-08.

## F-P2-09 Training a drafter from scratch (no-go)

- **Priority:** P2 (negative).
- **Claim:** an EAGLE-4-style drafter trained on the 3090 costs ≈53 h per 50M tokens at 40 TFLOP/s effective. That breaks the ≤ 2 h job rule many times over, and our MTP fine-tune was flat.
- **Expected impact:** n/a (**estimated** cost).
- **Evidence:** `transferability.md:43`; `plan.md:86`.
- **Status:** **dropped**.
- **Maps to:** from-scratch training (plan "not ranked").

## F-P2-10 Cross-layer KV sharing (no-go)

- **Priority:** P2 (negative).
- **Claim:** sharing KV across layers buys pool at a large quality cost on this model.
  - off: pool 57,669, PPL 8.24, GSM8K 0.94;
  - `group:2`: pool 87,499 (1.52×), PPL 11.95, GSM8K 0.56-0.64;
  - `group:4`: PPL 19.57, GSM8K 0.01-0.04.
- **Expected impact:** all numbers **measured (reported)** (PR #96, three boxes, n=100, `CTX=fast` DFlash2 k=15, 0.28-era stack).
- **Evidence:** `01-project/upstream-diff.md:86`; `transferability.md:189`.
- **Risks:** the result lives only in the closed PR, and no gotcha landed upstream (`upstream-diff.md:130`).
- **Status:** **dropped**. Any EAGLE-4 or other KV-sharing idea must clear this bar first.
- **Maps to:** U-07.

## F-P2-11 Sampled acceptance on int4 DFlash2 (negative)

- **Priority:** P2 (negative).
- **Claim:** int4 DFlash2 loses about 5% under sampling. EAGLE-4's target-sample-then-match rule (E-15) uses the same acceptance rule as ours for a single chain, so it recovers none of that loss.
- **Expected impact:** 0 (**estimated**). The −5% is **measured (reported)**.
- **Evidence:** `transferability.md:153,206`.
- **Status:** **dropped**.
- **Maps to:** I-11, E-15.

## F-P2-12 Quality-battery corpus (PR 359)

- **Priority:** P2 (measurement).
- **Claim:** with a venv outside `lib/python3.12/site-packages`, the quality battery silently drops its code cohort, and "all" PPL becomes a different mixture. PR 359 resolves the corpora from the active vLLM install and refuses a missing code corpus.
- **Expected impact:** no speed. It keeps PPL comparable across the int8/fp8 and drafter arms (F-P1-02, F-P2-02, F-P2-03).
- **Evidence:** `upstream-diff.md:47,106` (the authors' claim, not traced to a line).
- **Effort:** S. It merges clean; `research/deepinfer` also edits `quality_battery.py` and auto-merges.
- **Status:** **adopt now**, from upstream once merged or by cherry-pick. PR 364 (drafter JSONL tail repair) matters only if we generate drafter data, so it is deferred.
- **Maps to:** U-06.
