# EAGLE-4 transferability to one RTX 3090

Phase 3 (transfer-analyst). This doc judges every EAGLE-4 candidate (E-01..E-25, T-01..T-10), every upstream gain, and every internal bottleneck or inconsistency found in phase 1. The judgement is for our stack on one 24 GB RTX 3090 (sm86). The ranked list for phase 4 is in `00-meta/plan.md` under "Ranked hypotheses".

## Labels and sources

- **M** means measured. The source path is given next to the number. All M numbers come from the prior WSL2 3090 runs in `04-benchmarks/baseline.md`, from project docs, or from the docs cited by the phase-1 lanes. Nothing in this doc was run on a GPU (D1, `00-meta/decisions.md:5`).
- **E** means estimated. The reasoning is given next to the number. Arithmetic is in `02-eagle4/transferability-calc.py`, with its output in `transferability-calc.out` (CPU, stdlib only). Section numbers below such as "calc §2" refer to the blocks of that output.
- **"Paper"** means the EAGLE-4 paper, with its page. **"repo"** means `~/research-src/EAGLE-4` at `7b7b5a52`.
- **Baseline spread.** All win criteria below are set against this spread (`04-benchmarks/baseline.md`, "Notes" 6, and phase-2 review):
  - 1–2% between boots for C1 decode.
  - ±3% greedy non-determinism.
  - A thermal settle of −9.4% over ~4 minutes.
  - So a **C1 win** needs a median ≥ +3% over ≥3 interleaved boots per arm, after the warm-up probe. [review: the 1–2% is 2 boots per mode, raw jsonl never committed; the cline batches measured 0–3.2% between boots for C1 T=0 and ~10% for free-run decode (`origin/cline/gf59cdmh:docs/decode-speed-findings.md`). A C1 claim below +5% needs ≥5 interleaved boots, or teacher-forced tok/step × ms/pass from logs.] A **N≥4 aggregate win** needs ≥ +5%. Anything smaller is reported as "not resolved".
- **Reference step times (E from M).**
  - B C1: 114.4 tok/s ÷ 3.36 tok/step = 29.4 ms per step.
  - D C1: 99.8 ÷ 2.64 = 26.5 ms per step.
  - E at ~170k: 150.8 ms per step (M, `e-profile.md:34`).

## Summary

**Five most promising** (ranked as in plan.md):

1. **Adopt upstream #345** (PTH_FA default on `SPEC=mtp CTX=long`). +12–22% decode at long context on a 3090 against our checkout's fp8 default (M, upstream and user boots, `01-project/upstream-diff.md` Reviewer additions). Effort S, no licence gate. [review: M, `upstream-diff.md:126` (+21.3%/+22.1%, 3 boots); 0 against the stock image, which already resolves PTH_FA=1 (`04-benchmarks/baseline.md:44`). An adopt item, not a hypothesis.] [review (phase 4): the PR #345 A/B ran image `sha-95ac251` with #345's launchers, PTH_FA=0 (fp8, FlashInfer, piecewise graphs) vs PTH_FA=1; our checkout's unset mtp/long launch is that PTH_FA=0 arm (X-01a), so the gain applies to our users. +21–22% is WSL2 at 3 boots; native boxes +12–14% (2 boots).]
2. **k per concurrency band, from our own measured tables** (E-13/E-04 shape, built on DYNAMIC_K). This is EAGLE-4's "native-OFF above break-even" applied to B.
   - Gain: +7.7–9.2% at 4 streams (M, project-reported, three A/Bs, `single-user/README.md:434`), and up to +41% at N=4 and +91% at N=8 with k bands (branch-reported, `cline/gf59cdmh`).
   - A cost-table fit is CPU-testable now.
   - [review: the comparator must be DYNAMIC_K at the same boot, not fixed k=7. A per-step k cannot buy seats (state sized for the configured max k, cline section R); the N=8 +91% is boot-time `DFLASH_TOKENS=3` + int8 pool. The "+41% at N=4" does not reconcile with cline S-1's own row (263.6 / 239.1 = +10.2%). Expected gain over DYNAMIC_K: 0–5% (E).]
3. **PTH_FA int8 KV in B (`CTX=fast`).** It doubles context per GiB (16,384 → ≈31.8k tokens) and should remove B's pool-limited N=8 preemptions. C1 is expected neutral (E). [review: already measured as int8 TRITON on `CTX=fast` via `EXTRA_ARGS` (cline S-2): pool 68,605 → 116,440 = **1.70×** (not 1.94×; GDN and drafter pages do not halve), N=8 preemptions 23 → 1, C1 111.2 vs 111.0. PTH_FA itself is refused here (`start_qwen.sh:297-304`).]
4. **EAGLE-4 drafter chain acceptance probe** (E-20/E-02, evaluation use under §2(a)). This is the one measurement that decides whether any EAGLE-4 drafter work pays off on our box.
   - Break-even against DFlash2 at B C1 is 3.75–4.40 tok/step for a W4A16 layer with its own int4 head, against DFlash2's 3.36 (E, calc §4).
5. **Near-tie and fp32-arbiter parity gate** (E-18) for the greedy-equality bench. It separates real speculative-decode bugs (KVarN adaptive + prefix cache) from fp16 ties. CPU-buildable now.

**Five clearest no-gos:**

1. **Tree verify on our KV formats** (E-05/E-14/E-25, with E-15/E-16/E-17).
   - Needs unquantised KV (`tree_attention.py:69-70`), which halves context at D′ and E. [review: B and C already run bf16 KV with FLASH_ATTN (`single-user/start_qwen.sh:210-214`), so this reason does not apply there. The no-go still holds in B/C because of GDN tree state rows, the causal-only kernel, tree attention removed in 0.31, L effort and D11.]
   - Needs ≈1.2 GiB of fp16 GDN state for 16 rows.
   - Needs the CUSTOM backend slot that pthfa holds.
   - Needs L-size kernel work, and is licence-gated.
2. **Moving to the `+eagle.4` vLLM fork.** The hooks are unpublished, and 33 of our 51 patches fail on HEAD (`03-vllm/integration-paths.md:198,227-228`).
3. **Hopper/Blackwell-only paths.** Gluon GDN (sm_103), PDL and clusters (sm_90+), `flashinfer_cutedsl` (SM100). The Triton twins cover sm86, so nothing is lost (`integration-paths.md:207`).
4. **Training an EAGLE-4-style drafter from scratch.** ≈53 h per 50M tokens at 40 TFLOP/s (E, plan carry-forward). Our MTP fine-tune was flat (M, `drafter/README.md:29-37`).
5. **Cross-layer KV sharing (closed PR #96) and E-23 L2 prefetch.**
   - #96 was measured negative: GSM8K 0.94 → 0.56–0.64.
   - E-23 is SGLang-only, and the 3090 has a 6 MB L2.

**CPU-testable now:** 5 of the 15 ranked hypotheses have a part that runs here (H01 resolution, H02 cost-table fit, H08 gate code, H15 drift list, H16 fallback probe), plus this doc's memory arithmetic. No speed claim can be settled on this box.

## What DFlash2 already does (count gains against DFlash2, not a plain chain)

| EAGLE-4 element | DFlash2 / our stack today | Net novelty for us |
|---|---|---|
| Efficient LM Head: low-rank (r=128) retrieval of K=128 candidates over the full vocab, then exact rescoring (paper §3.2 pp. 5–6; E-10) | DFlash2 gets its top-16 per slot from the **full** 248k int4 target lm_head on 7 rows: 0.656 GB per step, ≈39% of drafter bytes (`dflash-specdec.md:102-104,488-490`). Its own low-rank part is the **rank-256 bilinear edge score** over the 16×16 candidates (`qwen3_dflash2.py:297-365`, `dflash-specdec.md:105-107`). That scores *transitions*, not vocabulary retrieval. | Only the retrieval half is new. It saves ≈0.63 ms per step, **+2.2% at B C1** (E, calc §5), because DFlash2's five-layer body dominates and the head is only 39%. That is far from the paper's −55.6% draft forward (Fig. 2 p. 4), which holds for a one-layer drafter whose head is 65.4% of the forward. |
| Adaptive depth/width by cost model ΔA/ΔC ≥ τ (paper §3.1 p. 5, Algorithm 1 p. 6; E-04/E-12/E-13) | **Long block 7→15** from lookup confidence (`dflash2/speculator.py:404-457`). **DYNAMIC_K** per batch size (`start_qwen.sh:678-692`). The acceptance estimator `_maybe_predict_acceptance` (`dflash-specdec.md:118`). | Only two things are new: (a) a *measured cost table* replacing hand bands, and (b) the native-OFF range. Context-length buckets (EAGLE-4 SGLang keys costs per 1024-token interval, `architecture.md:220-223`) are also new. The mechanism already exists. |
| Chain walk over candidates | `_selector_walk_kernel`: Markov walk with Gumbel argmax, one chain of 7 (`dflash-specdec.md:108-112`) | None for chain mode. EAGLE-4 at C≥8 also runs a chain (paper p. 7; `architecture.md:233-236`). |
| Sliding-window draft attention W=4096 (E-11) | DFlash2 drafter window 2048 (`dflash-specdec.md:79`). It costs ≈−12% tok/step at 12–36k (M, `dflash-specdec.md:459`). | A wider window needs a retrained DFlash2. The EAGLE-4 drafter ships 4096 natively, which H03 can measure per context bucket. |
| Device-side depth stop (E-12/E-22) | The long-block flag is read from a pinned host copy one step stale. Reading it synchronously costs 5% (M, `dflash-specdec.md:152-153`). | Partly new. The conditional node can only skip *draft* steps, but DFlash2 drafts in one pass, so only MTP's k=3 loop has steps to skip. |
| Multi-tap target features (E-11/T-07) | DFlash2 conditions on target hidden states through its context-KV projection (`dflash-specdec.md:73-95`). MTP uses the last layer. | Matters only for a new drafter we would train (T-07). |
| Lossless sampling without draft probabilities (E-15) | The chain rejection sampler uses q from the 16-candidate logits, or a point mass for lookup tokens (`dflash-specdec.md:113-116,146-148`). | For a single chain this is the same acceptance rule. The gain exists only for trees. |

## Memory gate

Inputs:
- Weights: from the header counts (`architecture.md` §1). The total is 3,089,075,200 params.
  - Vocab 248,320 × 5120 = 1,271,398,400 params each for `embed_tokens` and `lm_head`.
  - `fc` is 78.6M params.
  - The rank-128 factors are 32.4M params.
  - The layer is 435.2M params.
  - int4 g128 is 0.5156 B/param, which matches the target's int4 head (0.656 GB for 1.271B params).
- Spare GiB per mode and context tokens per GiB of `KV_MEM`: `01-project/quantisation.md` (Headroom and Reviewer additions).
  - B and C: 16,384 per GiB (M geometry).
  - D′ and D int8: ≈31.8k per GiB (M, 74 pages per GiB).
  - A fp8: 32,768 per GiB.
  - E: ≈80k per GiB (unverified there).
  - Spare: A 0.15, B 0.5, C 0.2, D 0 (the pool takes everything), D′ 0.5, E 0.15 GiB.
- The draft **replaces** DFlash2 (1.193 GiB) in B, C, D′ and E, and MTP (0.305 GiB) in D. It is **added** in A.

Cell format: **Δ GiB vs the drafter replaced / context one request loses** (E, calc §2). The context loss is (Δ − spare) × tokens per GiB. "0" means the variant fits inside today's spare or frees memory.

| Variant (resident draft) | GiB | A | B | C | D | D′ | E |
|---|---|---|---|---|---|---|---|
| V1 bf16 as released (own embed + own head + factors) | 5.75 | +5.75 / 184k | +4.56 / 67k | +4.56 / 71k | +5.45 / 173k | +4.56 / 129k | +4.56 / 353k |
| V2 bf16, `embed_tokens` dropped (needs a share that our int8 embedding fails, `integration-paths.md:216`) | 3.39 | +3.39 / 106k | +2.19 / 28k | +2.19 / 33k | +3.08 / 98k | +2.19 / 54k | +2.19 / 163k |
| V3 = V2 without `efficient_lm_head*` (the E-20 load path) | 3.33 | +3.33 / 104k | +2.13 / 27k | +2.13 / 32k | +3.02 / 96k | +2.13 / 52k | +2.13 / 159k |
| V4 W4A16 layer + fc, int4 **own** lm_head, no embed | 0.86 | +0.86 / 23k | −0.34 / 0 | −0.34 / 0 | +0.55 / 18k | −0.34 / 0 | −0.34 / 0 |
| V5 = V4 + bf16 factors (efficient head kept; licence-gated if shipped) | 0.92 | +0.92 / 25k | −0.28 / 0 | −0.28 / 0 | +0.61 / 20k | −0.28 / 0 | −0.28 / 0 |
| V6 bf16 layer + fc, **shared** target int4 lm_head | 0.96 | +0.96 / 26k | −0.24 / 0 | −0.24 / 0 | +0.65 / 21k | −0.24 / 0 | −0.24 / 0 |
| V7 W4A16 layer + fc, shared target lm_head | 0.25 | +0.25 / 3k | −0.95 / 0 | −0.95 / 0 | −0.06 / 0 | −0.95 / 0 | −0.95 / 0 |

Reading:

- **The draft fits only as a replacement, and only quantised (V4/V5) or with a shared head (V6/V7).**
  - V2/V3, the cheapest load path, costs B ≈27k tokens of its 65,536 `MAX_LEN`. That is acceptable for a short-context *acceptance probe* (H03) but not for serving.
  - In D it costs 96k of the context.
  - In E it costs 159k, which is more than E has.
- **The shared-head variants (V6/V7) change the model.** The draft trains its own lm_head (`tie_word_embeddings: false`, `integration-paths.md:220`). V2 sharing also requires `torch.equal`, which our int8 `weight_packed` embedding cannot pass (`build-and-kernels.md:156`).
  - Acceptance loss from serving V6/V7 against our int4 target head is unknown.
  - Recovering it needs a head-frozen fine-tune (T-05 inverted), which is gated as a derivative for shipping.
- **Draft KV is the hidden term.** MHA 40 × 128 × K,V × bf16 is 20,480 B per token (calc §3).
  - With the 4096 window freed out of window: ≈80 MiB per request.
  - If vLLM 0.31.0's hybrid allocator keeps out-of-window draft blocks (unexamined, `build-and-kernels.md:162`): 1.25 GiB at 64k and 3.24 GiB at 170k. That alone rules out E and D′.
  - The 20 KiB draft page must also fit the hybrid page-size unification with the 64 KiB target page and the GDN state pages (unexamined).
  - [review: examined. The SW allocation is bounded (`patches/hybrid-sw-block-promote.patch` bound cdiv(window−1+4096,16)+1), so "not freed" is probably moot. The padding is the real cost: 20,480 B/token at block 16 = 327,680 B, against a 1,835,008 B primary page (ratio 5.6). The promote patch picks 112 tokens → 2,293,760 B, which pads every target/GDN page by +25% (≈ −20% pool, E). Without promotion, ≈513 blocks × 1.75 MiB ≈ 0.9 GiB per request (E). vLLM 0.31.0 `kv_cache_utils.py:1383-1445`.]
- **A (batch, 64 seats)** has no drafter to replace, and EAGLE-4 itself goes native-OFF at high batch (repo width 0 above break-even, `architecture.md:229-236`). A is out of scope for any draft.
- **TTFT.** The EAGLE-4 draft prefill is ≈1.1 GFLOP per prompt token (2 × 546M params), ≈2% of the target's ≈54 GFLOP per token (E: 2 × 27B). The expected TTFT effect is under 2% either way. TTFT was never measured (`baseline.md`, "Gaps").

## Draft cost and break-even against DFlash2

From calc §4. B C1 step is 29.4 ms. DFlash2's realised draft cost is 3.3 ms, against a 1.80 ms floor, so the realised/floor ratio is 1.83. For a chain of 7 sequential passes, launch and latency overhead is likely *worse* than for one block pass, so read the ×1.83 column as optimistic.

| EAGLE-4 k=7 chain | Bytes per step | Floor per round | Realised (×1.83) | tok/step needed at B C1 (floor / ×1.83) |
|---|---|---|---|---|
| E-20 path: bf16 layer + bf16 own head | 3.41 GB | 25.7 ms | 47.1 ms | 5.92 / 8.37: **not a speed path** |
| bf16 layer + efficient head | 0.94 GB | 7.2 ms | 13.2 ms | 3.80 / 4.49 |
| W4A16 layer + int4 own head (V4) | 0.88 GB | 6.8 ms | 12.4 ms | 3.75 / 4.40 |
| W4A16 layer + efficient head (V5, gated) | 0.29 GB | 2.3 ms | 4.3 ms | 3.25 / 3.47 |

**What could clear the bar.** DFlash2 today gets 3.36 tok/step greedy and 3.61 at T=0.7. A non-gated V4 chain must reach ≈3.75–4.40 tok/step, i.e. +12–31%. [review: use the ratio, not the absolute number. 3.36 is greedy C1 on the `run_benchmarks` cohort with LOOKUP hits (spec-5: 3.486 T=0, 3.611 T=0.7; workload range 3.50–5.80, cline S-8). The bar is ≥1.117× (floor) to ≥1.31× (×1.83) of DFlash2's tok/step on the same prompts in the same run, with LOOKUP stated. The 3.3 ms draft cost includes lookup (`01-project/dflash-specdec.md:362`).]

The paper only bounds this from the side:
- EAGLE-4 *tree* against DFlash2 on 4×B300 is 1.28× at T=0 C1 and 1.09× at T=1 (paper Table 2 p. 9). That is a tree, and with a cheaper draft per byte than we would have.
- The paper reports no acceptance lengths (p. 7).

The prior that a chain on our box beats DFlash2 by ≥12% is therefore low (E). The value of H03 is information: one M-size run tells us whether the rest of the EAGLE-4 drafter line (H11, H12) is worth doing.

## Verdict tables

Column legend:
- **Prio**: P1, P2 or P3.
- **Applies?**: yes, partly or no.
- **sm86/24 GB**: memory, kernels and compute. Hopper, Blackwell, FP8, cluster or PDL needs are flagged.
- **Gain**: estimated, against B/D/E decode at C1/C4 or TTFT.
- **Effort**: prototype/adopt as S/M/L. "IP row" cites the rows of `03-vllm/integration-paths.md` §8.
- **D11**: licence gate.
- **DI**: DeepInfer cross-reference IDs only (D3).
- **Exp**: cheapest experiment. "CPU" means here; "GPU" means the GPU box. Each gives the metric and the win criterion.

### EAGLE-4 inference and build items (E-xx)

| id | what | Prio | Applies? | sm86 / 24 GB | Gain (E unless M) | Effort (IP row) | Risk | D11 | DI | Exp |
|---|---|---|---|---|---|---|---|---|---|---|
| E-01 | Legal gate (§1(e), §3(a)–(b), §6(a)) | — | yes | n/a | Enables E-03..E-05 and T-01 if cleared | human decision | — | **is** the gate | — | none; D11 is open |
| E-02 | Released drafter as an acceptance baseline | P1 | partly: evaluation only | Fits only as V3 at short context (+2.13 GiB at B, −27k context) | No direct speed gain (calc §4); decides H11/H12 | M/— (IP 1 chain via `method=eagle3`) | Low; wrong aux ids give acceptance ≈1 (T-10) | no for eval (§2(a)); yes to ship | drafter-4 | = H03 |
| E-03 | Efficient head on our MTP or DFlash2 | P1 (DFlash2) / P2 (MTP) | partly | +0.065 GB factors; Triton top-k on sm80+ | DFlash2 +2.2% B C1, +0.4% E 170k; MTP +0.5% speed but no vocab hole (calc §5–6) | L/M (train factors P-a, patch `compute_candidates` or the MTP head) | Medium: recall < 98.2% loses acceptance; the SVD init alone costs 0.48 tok per verify (paper p. 6) | **yes** | quant-5, drafter-1 | = H05 / H12 |
| E-04 | Cost-table depth and width per batch size and context | P1 | yes (k only) | Fine; DYNAMIC_K exists | See E-13 | S–M | See E-13 | partly: our own tables on DYNAMIC_K = no; porting Algorithm 1 or harness code = yes | spec-0/1/2 | = H02 |
| E-05 | Cost-sized tree verify on GDN | — | no (now) | bf16 KV only; 72 MiB per row of fp16 GDN state, ≈1.2 GiB per 16 rows (`architecture.md:247`); CUSTOM clash | Paper tree vs DFlash2 is 1.28× at T=0, 1.09× at T=1 on B300 (Table 2 p. 9). On our box it is eaten by bf16 KV (½ context at D′/E) and state rows (E) | L (IP 3) | High | **yes** | spec-3, attn-1/2 | estimate only; not scheduled |
| E-10 | Low-rank candidate head in front of our head | P1 | partly (see E-03) | Fine | +2.2% at B C1 on DFlash2. Under the 1–2% spread unless ≥5 boots | L | Medium | **yes** | quant-5 | GPU after T-04 recall ≥ 98% |
| E-11 | 3-tap per-tap-norm fusion + SWA 4096 draft | P2 | partly: DFlash2 already windowed (2048) | Fine | DFlash2 loses −12% at 12–36k from its window (M, `dflash-specdec.md:459`). Recovering it needs a retrained DFlash2 (E) | L (retrain) | High (no recipe for DFlash2) | no (generic EAGLE-3/SWA; `overview.md:256`) | drafter-4 | fold into H03: acceptance per context bucket |
| E-12 | Device-side depth stop (GPU flag + conditional node) | P1 | partly: only MTP's k=3 loop | Driver feature only, not Hopper; untested on our CUDA/WDDM stack (`build-and-kernels.md:160`) | ≤1–2% on D C1 (E: skipping ≤2 of 3 MTP passes at 0.34 ms floor each when pos 0 rejects, ~14% of steps, `baseline.md` D accept@0 0.86) [review: EAGLE-4's stop is decided before verify, from draft-front confidence and a cost share (`02-eagle4/architecture.md:134-140`), not after a pos-0 reject. Below the +3% bar either way.] | M | Medium (graph capture) | yes (harness technology) | spec-2, drafter-1 | = H06 |
| E-13 | Per-batch verify width from measured round tables, incl. native-OFF | P1 | yes | Fine; but `PTH_FA=1` + dflash2 refuses DYNAMIC_K (`start_qwen.sh:324-327`), huge + prefix refuses it, and state pages scale with `num_speculative_tokens`, not k (`01-project/architecture.md:222`) | B 4 streams +7.7–9.2% (M, project-reported, `single-user/README.md:434`); k/seat bands +41% at N=4, +91% at N=8 (branch-reported) [review (phase 5): not k-band gains; +41% does not reconcile (+10.2% from its own row), +91% is boot-time k=3 plus int8 (`00-meta/reviews/phase3-transferability.md:32-33`)] | S (CPU fit) / M (IP 4 "k-only") | Low–medium: a variable verify length is unsafe on KVarN + prefix cache | partly (see E-04) | spec-0/1, runtime-2 | = H02 |
| E-14 | GDN tree recurrence from the parent state + commit | — | no | +W state rows per request; fp16 OK (`gdn_tree_verify.py:245`) | Only with E-05 | L (IP 3b, 3d) | High | **yes** | spec-3 | none |
| E-15 | Target-sample-then-match acceptance | — | no for chain | — | 0 for a single chain (same acceptance rule); helps only trees | M | — | yes | — | none |
| E-16 | Parent-first row order | — | no (a chain prefix is always valid) | — | 0 | — | — | yes | spec-2 | none |
| E-17 | Row attention, one KV load per K draft rows | — | no (tree drafting only) | — | 0 for chain | L | — | yes | — | none |
| E-18 | Near-tie + fp32-arbiter greedy parity gate | P1 (method) | yes | CPU logic; fp32 re-check on GPU | No speed gain. Separates ±3% greedy noise from real wrong-token bugs (KVarN adaptive + prefix, 38/793 wrong, M, `dflash-specdec.md:470-473`) | S | Low | no (generic test method; write our own) | evals-8 | = H08 |
| E-20 | Load the EAGLE-4 drafter as EAGLE-3 on stock 0.31 | P1 | yes for eval | V3 at B: +2.13 GiB, −27k context; 3.41 GB per step bf16 = 3.65 ms (calc §4) | Acceptance only; speed is negative (needs 5.9 tok/step) | M (IP 1 + Reviewer additions: arch alias, loader filter, aux ids [4,32,60], per-draft `kv_cache_dtype`/`attention_backend`) | Medium: the embed share may raise on the quantised layer; stock SWA counts draft tokens past 4096 | no for eval | drafter-4 | = H03 |
| E-21 | Warp-shuffle candidate tail (sm_80+) | P2 | partly: needed only with E-10 | sm80 SASS runs on sm86 (E, `integration-paths.md:207`) | Part of E-10's 0.63 ms | S once E-10 exists | Low | **yes** | quant-5 | after E-10 |
| E-22 | CUDA-graph conditional nodes | P1 | partly (as E-12) | CUDA 12.4+ driver; untested under WSL2/WDDM | As E-12 | M | Medium | no for the generic CUDA feature; yes if `tree_drafter.py` is ported | spec-2 | = H06 |
| E-23 | Side-stream L2 prefetch | — | no | 6 MB L2; SGLang-only | ≈0 (E) | M | — | yes | — | **no-go** |
| E-24 | prebuilt → JIT → Triton fallback by arch table | P3 | yes | Fine | 0 speed; no hard-fail on new hosts | M | Low | no (generic pattern) | — | = H16 |
| E-25 | Ancestor-bitmask variant in pthfa/KVarN verify kernels | — | no (now) | Prerequisite of E-05; kernels are causal-only (`integration-paths.md:218`) | Only with E-05 | L | High | yes if derived from `tree_attention.py` | attn-1/2, spec-3 | none |

### EAGLE-4 training items (T-xx)

| id | what | Prio | Applies? | sm86 / 24 GB | Gain | Effort | Risk | D11 | DI | Exp |
|---|---|---|---|---|---|---|---|---|---|---|
| T-01 | Rank-128 retrieval head + K=128 rescoring for MTP, replacing the 40,960 slice | P2 | yes | 0.065 GB factors vs 0.108 GB slice | Speed +0.5% on D (calc §6). Coverage: removes the vocab hole. Chinese 1.06 → ≈2.2 tok/step (E: the full head's 2.23, M, `gotchas.md:1436-1472`) | M (P-a ≈11–17 min per epoch) | Medium (recall) | **yes** | quant-5 | = H12 (gated). The non-gated alternative is H09 |
| T-02 | Train factors with the drafter frozen | P2 | yes, with T-01 | Fits (P-a) | As T-01 | M | Low | **yes** | drafter-4 | with H12 |
| T-03 | Frequency-weighted SVD init | P2 | yes | CPU-feasible with the head weights and token counts | Gate for T-01; paper SVD 91.5% vs trained 98.2% recall (p. 6) | S | Low | **yes** (part of the head technology) | — | = H12 (needs the lm_head file, not on this box) |
| T-04 | Recall@K offline gate | P2 | yes | CPU | Cheap gate before any served run | S | Low | no (generic metric) | evals-8 | with H12 |
| T-05 | Domain-adapt the released drafter (body only) | P2 | yes, if H03 is promising | P-c ≈25–40 min (E, training lane) | Raises H03's tok/step toward break-even. Our MTP fine-tune was flat (M, `drafter/README.md:29-37`), so the prior is low | M | Medium | no for eval; **yes** to ship derived weights (§4(c)) | drafter-4 | = H13 |
| T-06 | Longer TTT unroll in the MTP trainer | P2 | partly | Fits | Low prior: depth-2 result flat | M | Low | no (EAGLE-3 TTT is prior work) | drafter-4 | = H14 |
| T-07 | 3-tap features with per-tap norm | P3 | no (only for a new drafter) | — | — | L | — | no | — | none |
| T-08 | Train at the served window | P2 | partly (DFlash2 2048) | — | Up to the −12% at 12–36k (M loss), only with a retrain | L | High | no | — | none (no DFlash2 recipe) |
| T-09 | Two-stage dense → factors | P2 | with T-01 | — | — | — | — | **yes** | — | with H12 |
| T-10 | Export sanity: header shapes, acceptance ≈ 1 alarm | P2 | yes | CPU (headers) | Prevents a wasted GPU slot | S | — | no | — | pre-check inside H03 |

### Upstream gains (U-xx, from `01-project/upstream-diff.md` §7)

| id | what | Prio | Applies? | sm86 / 24 GB | Gain | Effort | Risk | D11 | DI | Exp |
|---|---|---|---|---|---|---|---|---|---|---|
| U-01 | Fast-forward to `upstream/main` 63f7581: b4b97ec #345 PTH_FA on mtp/long, 373a695 perf_counter, 2cf50a0, 433b050 | P1 | yes | int8 pool −1.5–1.9% vs fp8; may refuse `MAX_LEN` 150000 near the edge (upstream-diff Reviewer additions) | D long-context decode **+21–22%** WSL2 3090, 3 boots (M, user's PR #345 comment); +9% native, 1 boot; +12–14% at 2 boots. Against the stock image `sha-2cf50a0`: 0 (already live). | S (0 conflicts) | Low; the box's `.env` has `CTX=fast VISION=1`, so the default does not fire there | no | — | = H01 |
| U-02 | Interleaved PTH_FA=0/1 re-baseline | P1 (method) | yes | — | Avoids a 9–14% apples-to-oranges error | M | Low | no | — | bench-9 (queued) |
| U-03 | PRs 353 (LABD corpus) and 354 (SSE truncation) | P1 (method) | yes | — | Validity, no speed | S | Low | no | — | phase-4 harness |
| U-04 | Mine `cline/gf59cdmh` k bands and `dynamic-k-results` | P1 | yes | — | Input to H02 | M (reading) | Low | no | spec-0/1 | inside H02 (CPU) |
| U-05 | `ctx-profile-sweep` harness | P1 (method) | yes | — | Context-bucket sweeps for H02/H03 | S | Low | no | runtime-2 | phase-4 harness |
| U-06 | PR 359 (PPL corpus), PR 366 (pthfa re-vendor), PR 364 | P2 | yes | — | Hygiene | S | Low | no | — | none |
| U-07 | Cross-layer KV sharing (#96) | — | no | — | Measured negative: GSM8K 0.94 → 0.56–0.64 | — | — | no | — | **no-go** |
| U-08 | `paired_run.sh`, `canary.sh` from `master` | P1 (method) | yes | — | ABBA boots cut the ~10% free-run spread (branch-reported) | M | Low–medium | no | — | phase-4 harness |

### Internal bottlenecks and inconsistencies (I-xx, phase 1)

| id | what | Prio | EAGLE-4 technique that applies | sm86 / 24 GB | Gain | Effort | Risk | D11 | DI | Exp |
|---|---|---|---|---|---|---|---|---|---|---|
| I-01 | Accepted length at short and mid context (`dflash-specdec.md:455-467`) | P1 | E-02/E-20 (a better drafter?), T-05 | See the memory gate | +10% tok/step ≈ +10% tok/s (M scaling, ibid.) | — | — | — | drafter-3/4 | H03, H11, H13 |
| I-02 | Verify attention at long context: 78% of the step at 170k; KVarN stage 1 ≈40× its floor | P1 | **none**: E-17 is draft-side, and trees add verify rows | — | Largest E-mode lever, but not EAGLE-4's | — | — | — | attn-2, runtime-2 | out of scope (D3); bench-10 profiles it |
| I-03 | GDN state residency: 0.74–0.88 GiB per resident, ~7 ms per resident, 0.6 ms explained | P1 | none; E-14 makes it worse (W rows) | — | — | — | — | — | spec-3, spec-1, runtime-1 | bench-10 |
| I-04 | Fixed draft cost 3–3.6 ms; lm_head ≈39% of bytes | P1 | E-10/E-21 | Fine | +2.2% at B C1 (calc §5) | L | Medium | **yes** | quant-5, evals-8 | H05 |
| I-05 | Graph stairs: Marlin at B·q = 16 (≈8 ms), ragged batch −8% piecewise | P1 | E-13's measured per-(B, width) cost tables would encode the stair | — | Part of H02 | — | — | — | runtime-2 | inside H02 |
| I-06 | Host gap 5.7 ms at 170k; the long-block flag is one step stale (sync read −5%) | P1 | E-12/E-22 (partly) | Conditional nodes cannot change the verify length (graphs are per q) | ≤1–2% (E) | M | Medium | partly | spec-2 | H06 |
| I-07 | O(context) lookup scan | P3 | none (EAGLE-4 has no lookup) | — | < 1 ms at 170k (E) | — | — | — | spec-4 | none |
| I-08 | KVarN + adaptive long block + prefix cache gives wrong tokens (pinned ADAPTIVE=0 on huge) | P1 (correctness) | E-18 gate detects this class of bug | — | Correctness | S (gate) | — | no | evals-8 | H08 |
| I-09 | fp8 KV drifts 1.7× more than int8 (0.0360 vs 0.0207 nats at 32k, M, `quantisation.md` quality table) and fails non-inferiority; default in A, in D on our checkout, and as D′ fallback | P2 | none (EAGLE-4 runs bf16 KV) | int8 pool −1.5–1.9% | Quality; speed +21% at D (= U-01); A unknown | S (D via U-01) / M (A: `start_qwen.sh:299-304` refuses PTH_FA outside mtp/dflash2 long) | Medium for A (64-seat decode on pthfa untested) | no | quant-2/3 | H01 (D), H10 (A) |
| I-10 | MTP 40,960-row vocab: Chinese 6.4% coverage → 1.06 tok/step, 35.5 vs 80.6 tok/s (M, `heads-drafters.md:151`) | P2 | T-01 (gated) or a plain fallback to the full head | Memory: unverified whether the full head is the target's shared copy or a second 1.20 GiB copy (open question 4) | Chinese +100–127% (M endpoints); English 0 (E) | M | Low–medium (detector) | no for the fallback | quant-5 | H09 |
| I-11 | int4 DFlash2 −5% under sampling | P2 | none (E-15 no gain for chain) | — | — | — | — | — | drafter-3 | none |
| I-12 | DFlash2 2048 window −12% at 12–36k | P2 | E-11/T-08 (retrain) | — | — | L | — | no | drafter-4 | H03 measures EAGLE-4's 4096 window per bucket |
| I-13 | Image vs checkout skew: D resolves PTH_FA=1 in `sha-2cf50a0` but fp8 on `95ac251`; the box's `.env` `CTX=fast VISION=1` blocks the default | P1 | — | — | Removes a 12–40% measurement trap (M, `baseline.md` Notes 1–2) | S | Low | no | — | H01 |
| I-14 | Doc drift: `docs/optimizations.md:303-304` NSTRONG/AGREE (code 6/0); `README.md:96-101`; SPEC default `single-user/start_qwen.sh:197` mtp vs `.env.example:6` dflash2; GDN state 0.74 vs 0.82/0.88 GiB | P3 | — | — | Docs | S | None | no | — | H15 |
| I-15 | TTFT and peak VRAM never measured | P1 (method) | — | — | Baseline gap; every TTFT claim here is E | S | — | no | — | bench-1..12 harness |

## Experiment index (for phase 4)

The experiment IDs are new (`X-xx`). GPU items still need a `bench-N` slot in `00-meta/gpu-queue.md`, which phase 4 owns. Win criteria follow the spread rule in "Labels and sources".

| Exp | Hypothesis | Where | Metric and win criterion |
|---|---|---|---|
| X-01a | H01 | CPU | With the ci-cc86 stub `nvidia-smi` (433b050) and a fake `pthfa_installed`, `resolve_config.sh` on `upstream/main` resolves `SPEC=mtp CTX=long` to `PTH_FA=1`, and on `95ac251` to `FA2_FP8KV`. [review: wrong for ours: `95ac251` leaves PTH_FA and FA2_FP8KV both unset, i.e. fp8 KV on FlashInfer (`resolve_config.sh:121` gives the FA2 default to dflash2/long only; `start_qwen.sh:220-223,297`; X-01a `run.out:75`, re-run by the phase-4 review).] Pass = both as stated. Also list which `.env` keys block it (`VISION=1`). |
| X-01b | H01 | GPU (bench-9) | D C1 and labd TOTAL at ~66k and ~93k, PTH_FA=0 vs 1 interleaved, ≥3 boots per arm. Win: ≥ +10% decode, tok/step within ±2%, GSM8K within 1 pt. |
| X-02a | H02 | CPU | Fit a k-per-N table by maximising (1 + Σa_i)/(V(N,k) + k·D(N)). Inputs: spec-5 per-position acceptance (M), B/D/A ladders (M, `evals-10.md`), and B's 25.9/32.8/49.1 ms step curve at 1/2/4 streams (M, native, `single-user/README.md:112-117` via `dflash-specdec.md:426-431`). Output: a predicted table and predicted N=4/N=8 aggregates. Pass = the prediction reproduces the measured DYNAMIC_K +7.7–9.2% at 4 streams within ±3 pts. [review: then compare the fitted table against the DYNAMIC_K table, not k=7.] |
| X-02b | H02 | GPU | `conc_ladder` N = 1/2/4/8 on B (PTH_FA=0, where DYNAMIC_K is allowed) with the fitted table vs fixed k=7, ≥3 boots. Win: N=4 ≥ +10% aggregate, N=8 ≥ +25%, C1 within 2%. Also run at `DFLASH_TOKENS=3` to test seats, since state pages follow `num_speculative_tokens`. [review: the baseline arm is the DYNAMIC_K table at the same boot; win ≥ +5% at N=4 or N=8. The seats arm is already measured (cline S-1: k=3 N=8 337.1 vs k=7 176.8); drop it.] |
| X-03 | H03 | GPU | E-20 load (V3), `KV_MEM` lowered, `MAX_LEN` 16k. Measure tok/step per position and per context bucket (1.5k, 8k, 12–16k) on the B prompt set, T=0 and T=0.7, against DFlash2 on the same prompts. **Continue the EAGLE-4 line only if tok/step ≥ 3.75 at T=0** (the V4 floor break-even, calc §4). [review: the bar is ≥1.12× DFlash2's tok/step, same prompts, same run. Add a load-only pre-gate: config `architectures` → `LlamaForCausalLMEagle3`; drop `embed_tokens`/`efficient_lm_head*` from safetensors (`_should_share` `torch.equal`); log page size and pool tokens. Untested: EagleSpeculator on a GDN target, M-RoPE vs 1-D draft rope.] Pre-check: T-10 header shapes and aux ids [4,32,60]. |
| X-04 | H04 | GPU | Launcher: let `PTH_FA=1` through on `SPEC=dflash2 CTX=fast` (a `start_qwen.sh:299-304` allow-list edit on a scratch branch). B C1 decode; N = 1/2/4/8 aggregate and preemptions; pool tokens. Win: C1 within 2%, N=8 preemptions 0 and aggregate ≥ +15%, pool ≥ 1.8×. Quality: GSM8K within 1 pt. [review: measured 1.70× with int8 TRITON (cline S-2), so a 1.8× bar would fail. Primary arm: int8 TRITON via `EXTRA_ARGS`, with no launcher edit; gate pool ≥1.6×, N=8 preemptions ≤1, C1 within 2%. PTH arm optional, ≤ +5% E (2×3080 8k ratio 1.07 [0.95–1.10], `docs/long-context.md:412`).] |
| X-05 | H05 | GPU (after X-12 recall) | DFlash2 with a factor head in `compute_candidates`. B C1, ≥5 interleaved boots. Win: ≥ +2% decode with tok/step ≥ −1%. Estimated at +2.2%, so a clean win is unlikely to be resolvable. [review: below the +3% bar after L effort plus D11; moved to not ranked.] |
| X-06 | H06 | GPU | Micro-probe that a `torch.cuda` graph with a conditional node captures and replays on the box's driver/WDDM. Then MTP k=3 with a device-side stop at a pos-0 reject. Win: D C1 ≥ +2%. [review (phase 5): the bar is D C1 ≥ +3% over ≥ 5 interleaved boots (`plan.md:36,53`), and the mechanism is a pre-verify confidence stop, not a post-reject skip (`architecture.md:134-140`).] |
| X-08 | H08 | CPU then GPU | Build a classifier for bench-7 (greedy T=0 equality): at each first divergence, fetch top-2 logprobs and flag near-ties (margin < 1e-2 nats, or the fp32 re-score flips). CPU: unit test on synthetic logprob streams. GPU: run on bench-7. Pass = every non-near-tie divergence is reproducible, and the KVarN adaptive + prefix case is flagged as real. |
| X-09 | H09 | GPU | Chinese prompt set, D C1: slice vs full head (`MTP_DRAFT_VOCAB=0`) vs a coverage-triggered switch. Win: switch ≥ 2.0 tok/step on Chinese, English D C1 within 2% of the slice. |
| X-10 | H10 | GPU | A with int8 per-token-head KV (TRITON or pthfa) vs fp8: 64-conc e2e throughput and drift (teacher-forced nats). Win: throughput within 3%, drift ≤ 0.025 nats. |
| X-11 | H11 | GPU (if X-03 passes) | V4 W4A16 drafter (AutoRound or GPTQ on the draft) vs bf16 V3. Win: tok/step within −3% at T=0 and T=0.7. |
| X-12 | H12 | CPU on a box with `lm_head` + capture (not here: no weights) | SVD init (T-03) recall@128 of the full head's top-1 on our captured hidden states. Report against the paper's 91.5% SVD / 98.2% trained (p. 6). Licence-gated for shipping. |
| X-13 | H13 | GPU (if X-03 is within 10% of break-even) | P-c domain adaptation, then X-03 again. Win: tok/step reaches break-even. Gated to ship. |
| X-14 | H14 | GPU | MTP TTT unroll depth 2 → 4. Win: D tok/step ≥ +3%. |
| X-15 | H15 | CPU | Diff list with file:line for each drift item, handed to phase 5 (other docs are not edited in this phase). |
| X-16 | H16 | CPU | Import pthfa/KVarN packages with no GPU and record which hard-fail rather than fall back. Pass = a list; design the fallback table from E-24. |

## Open questions

1. **D11** is still open. H05 and H12 (and any E-04 code port) stay blocked for shipping. Local evaluation (H03, H11, X-12) is §2(a) use.
2. **Draft KV out-of-window freeing.** Does vLLM 0.31.0 free EAGLE-3 sliding-window draft blocks under our hybrid allocator, and how does the 20 KiB per token page unify with the 64 KiB target page? This decides D′/E feasibility for any EAGLE-4 draft (memory gate).
3. **Embedding share on the int8 target embedding.** Does `torch.equal` raise on `weight_packed`? If so, the V2/V3 load needs a loader patch (`build-and-kernels.md:156`).
4. **Is the full MTP head (`MTP_DRAFT_VOCAB=0`) a second resident copy** (1.20 GiB per `heads-drafters.md:150`) or the target's shared head? This decides H09's memory cost.
5. **`PTH_FA` with `SPEC=off` or `CTX=fast`.** The launcher refuses both (`start_qwen.sh:299-304`). Does pthfa itself support non-windowed bf16-replacement decode at 64 seats? X-04 and X-10 assume yes.
6. **DYNAMIC_K's measured gain** (+7.7–9.2% at 4 streams) and cline's +41%/+91% bands: against which fixed-k arm, and on which image? X-02a needs the raw rows (`dynamic-k-results`, `cline/gf59cdmh`).
