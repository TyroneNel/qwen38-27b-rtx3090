# EAGLE-4 expedition: plan and status

Goal: make `qwen38-27b-rtx3090` faster and better on one RTX 3090 (24 GB, sm86). Priorities: P1 speed and inference (DFlash, speculative decoding), P2 quants and heads, P3 everything else. Outside ideas come from EAGLE-4 only.

## Sources (pinned)

| Source | Location | Commit |
|---|---|---|
| Our project | `~/qwen38-27b-rtx3090`, branch `research/eagle4-expedition` from `origin/main` | `95ac251bc3b71017c883e83004cf4fb188d066a4` |
| Upstream HyperQwen | `~/research-src/HyperQwen` (`upstream/main`) | `63f7581804afe05f23f240c2f584a00ac18c488b` |
| EAGLE-4 | `~/research-src/EAGLE-4` (`main`; tag `v0.1.0` = `75436712`) | `7b7b5a522d7a7284b9eeb9889d03c8c2038c112f` |
| vLLM reference | `~/vllm` (`v0.32.1rc0-68-g4bd75e952a`) | `4bd75e952a9f423742223c97c8cc9fc6db362950` |
| Prior research (context only) | `~/research-src/deepinfer-ref` (export of `origin/research/deepinfer`) | `86ce6edb4d2b5c5d6e1b09cd9e2f9fde73070a1f` |

## Environment

| Box | Facts |
|---|---|
| Research box (this one) | Linux 6.17.13-2-pve, 6 vCPU, 19 GB RAM, **no GPU** (no `/dev/nvidia*`, no `nvidia-smi`), Python 3.13.7, torch 2.12.1+cu130 (CPU only), vLLM not installed, Docker 29.7.2, 47 GB free on `/` at start |
| GPU box (prior expedition) | RTX 3090 24 GB sm86 at 250 W, driver 610.88, Windows 10 + WSL2 + Docker Desktop, stock image `ghcr.io/syv-ai/hyperqwen:sha-2cf50a0`. Full record: `~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/ENVIRONMENT.md` |

## Phases

| Phase | What | Status |
|---|---|---|
| 0 Setup | Branch, reference clones, scaffold, environment | done |
| 1 Understand | Project map, DFlash, quants and heads, upstream diff, EAGLE-4 ×2, vLLM paths; review each | done: 9 docs, 7 reviews, all PASS WITH CORRECTIONS |
| 2 Baseline | Harness + GPU handoff; baseline from measured 3090 runs; independent re-derivation | done: re-derived, 0 disagreements (reviews/phase2-harness-baseline.md); GPU re-run queued (bench-1..12) |
| 3 Transferability | `02-eagle4/transferability.md`, ranked hypotheses below; independent review | done: ACCEPT WITH CHANGES (reviews/phase3-transferability.md), ranking corrected |
| 4 Experiments | CPU-feasible experiments here; GPU experiments packaged for the GPU box | CPU done: X-01a, X-02a, X-08, X-15, X-16, X-17 re-run by reviewers (reviews/phase4-*.md); GPU packaged as bench-13..25, rows ≤2 h, hand-off READY WITH FIXES (reviews/phase4-gpu-handoff.md) |
| 5 Findings | `05-findings/*`, roadmap; review | done: 31 findings, roadmap 6/6/6, PASS WITH CORRECTIONS (reviews/phase5-findings.md) |
| 6 Report | `REPORT.md`, fact check, README, commit, push branch to fork | running |

## Ranked hypotheses

Source: `02-eagle4/transferability.md` (verdict tables, experiment index X-xx; arithmetic in `transferability-calc.out`). Gains are **estimated** unless marked M. A win needs ≥ +3% at C1 or ≥ +5% at N≥4 against the 1–2% boot spread. **CPU-now** = a part runs on this box now. Within each priority, ordered by gain × confidence ÷ effort.

[review: table replaced by the phase-3 review's corrected ranking (`00-meta/reviews/phase3-transferability.md`). The changes: H01 is an M adopt item; H04′ adopts the measured cline S-2 int8 TRITON on B (pool 1.70×); H03's bar is a same-run ratio of ≥1.12× DFlash2 tok/step; H02′ is compared against DYNAMIC_K, not k=7; H06 is demoted (mechanism corrected, below bar); H05 is not ranked (+2.2% E, below the +3% bar, L + D11). Rows the review did not change keep their original text.]

| Rank | ID | Hypothesis (falsifiable) | Prio | Expected gain (E) | Exp | Where | Effort | Licence-gated? |
|---|---|---|---|---|---|---|---|---|
| 1 | H01 (U-01) — adopt | Fast-forwarding to `upstream/main` (#345) makes an unset-PTH_FA `SPEC=mtp CTX=long` launch from our checkout resolve to PTH_FA=1 | P1 | +21–22% D decode at 66–93k vs our checkout (**M**, `upstream-diff.md:126`); 0 vs stock image [review: +12–22% (**M**, upstream-reported, PR #345 thread): +21.3/+22.1% is WSL2, 3 boots/arm, `GPU_UTIL=0.89`; native boxes gave +12–14% (2 boots) and +9% (1 boot). Comparator = `PTH_FA=0` = fp8 KV on FlashInfer with piecewise graphs only, on image `sha-95ac251`, which is exactly what our checkout's unset mtp/long launch resolves to (X-01a `run.out:75`; not `FA2_FP8KV`). So the gain applies to our checkout's users, at the box-dependent size; part of it is FULL vs PIECEWISE graphs, not the kernel alone (phase-4 review)] | X-01a (+1 confirm boot) | **CPU-now** + GPU | S | no |
| 2 | H04′ (internal) — adopt | int8 per-token-head KV (TRITON, via `EXTRA_ARGS`) on `SPEC=dflash2 CTX=fast` reproduces pool ≥1.6×, N=8 preemptions ≤1, C1 within 2% | P1 | pool 1.70×, preemptions 23→1, C1 tie (**M** branch, cline S-2); PTH sub-arm ≤ +5% C1 (E) | X-04′ | GPU | S | no |
| 3 | H08 (E-18) | Near-tie + fp32 re-score classifier separates benign from real greedy divergences | P1 | Correctness gate | X-08 | **CPU-now** + GPU | S | no |
| 4 | H03 (E-20/E-02) | EAGLE-4 drafter as EAGLE-3 (aux [4,32,60], embed dropped) loads on 0.31 and reaches ≥1.12× DFlash2's tok/step on the same prompts, same run, at T=0 | P1 | Information gate (prior low) | X-03 (+ load-only pre-gate, page-size check) | GPU | M | no for eval; yes to ship |
| 5 | H02′ (E-13/E-04) | A fitted cost table with a native-OFF band beats the existing DYNAMIC_K table at the same boot by ≥5% at N=4 or N=8, C1 within 2% | P1 | 0–5% (E) [review: X-02a: 0–2% at N=4/8 (E) under both a cliff and a smooth acceptance tail (`reviews/phase4-x02a-tailshape.out`); the bar is not reached. Reword X-02b′ to an N=3 boundary check (fitted vs DYNAMIC_K, ~+5% E) plus a per-position acceptance capture on the ladder prompts; drop the N=4/8 win framing] ; the k=3 optimum is already in DYNAMIC_K | X-02a, X-02b′ | **CPU-now** + GPU | S/M | no (own tables) |
| 6 | H09 (I-10) (internal) | Switching MTP to the full head when draft-vocab coverage drops restores ≥2.0 tok/step on Chinese (from 1.06) with English D C1 within 2% | P2 | Chinese +100–127% (M endpoints 35.5 → 80.6 tok/s); English 0 | X-09 | GPU | M | no |
| 7 | H10 (I-09) (internal) | int8 per-token-head KV in mode A matches fp8's 814 tok/s e2e within 3% and keeps drift ≤0.025 nats (fp8 0.036) [review: int8 arm needs TRITON int8 or a `start_qwen.sh:297-304` allow-list edit] | P2 | Quality (1.7× less drift, M); speed ±3% | X-10 | GPU | M | no |
| 8 | H11 (E-20) | If H03 passes: W4A16 layer + int4 own head keeps tok/step within −3% **and** net pool (after page-size padding) ≥ DFlash2's | P2 | Servability; page padding ≈1 GiB (E) likely erases the −0.34 GiB | X-11 | GPU | S/M | no for eval; yes to ship |
| 9 | H12 (T-01/T-03/T-04/T-09) | SVD-initialised rank-128 factors of our int4 lm_head reach recall@128 ≥ 91.5% (paper SVD bar) on our captured states; ≥98% after training | P2 | Gate for H05 and a no-hole MTP head | X-12 | CPU on a box with weights (not here) | S/M | **yes (D11)** |
| 10 | H13 (T-05) | If H03 is within 10% of break-even: body-only domain adaptation (P-c) lifts EAGLE-4 tok/step to the H03 ratio bar (≥1.12× DFlash2 tok/step, same run) | P2 | 0–10% tok/step (prior low; our MTP fine-tune was flat, M) | X-13 | GPU | M | no for eval; yes to ship derived weights |
| 11 | H14 (T-06) | Raising the MTP TTT unroll from depth 2 to 4 raises D tok/step ≥3% | P2 | 0–3% (prior low) | X-14 | GPU | M | no |
| 12 | H06 (E-12) | A confidence-gated, pre-verify depth stop via a CUDA-graph conditional node captures on the box's driver; D C1 ≥ +3% (≥5 boots) | P2 | ≤1–2% (E), below bar | X-06 | GPU | M | no for CUDA; yes if ported |
| 13 | H15 (I-14) (internal) | Each listed drift item (NSTRONG/AGREE, `README.md:96-101`, SPEC default, GDN state size) is confirmed against code with file:line | P3 | Docs only | X-15 | **CPU-now** | S | no |
| 14 | H16 (E-24) | With no GPU, importing pthfa/KVarN packages hard-fails rather than falling back; an arch-table fallback removes every hard-fail | P3 | 0 speed; robustness | X-16 | **CPU-now** (probe) | M | no |

<details><summary>Original phase-3 ranking</summary>

| Rank | ID | Hypothesis (falsifiable) | Prio | Expected gain (E) | Exp | Where | Effort | Licence-gated? |
|---|---|---|---|---|---|---|---|---|
| 1 | H01 (U-01, I-13, I-09) | After fast-forwarding to `upstream/main` (#345), an unset-PTH_FA `SPEC=mtp CTX=long` launch resolves to PTH_FA=1 and decodes ≥10% faster at ~66k than the same launch on `95ac251` | P1 | +12–22% D long-context decode vs our checkout (M upstream/user boots); 0 vs stock image | X-01a, X-01b (bench-9) | **CPU-now** (resolution) + GPU | S | no |
| 2 | H02 (E-13/E-04, U-04, I-05) | A k-per-concurrency table fitted from our measured acceptance and step costs (incl. low-k/off bands) lifts B aggregate ≥10% at N=4 and ≥25% at N=8 over fixed k=7, with C1 within 2% | P1 | +8–40% at N≥4 (anchors: +7.7–9.2% M project-reported; +41/+91% branch-reported) | X-02a, X-02b | **CPU-now** (fit) + GPU | S/M | no if built on DYNAMIC_K with own tables; yes if harness code is ported |
| 3 | H04 (carry-forward) | PTH_FA int8 KV on `SPEC=dflash2 CTX=fast` keeps B C1 within 2% of bf16 KV, grows the pool ≥1.8×, and removes the N=8 preemptions (aggregate ≥ +15%) | P1 | C1 0 ±2%; N=8 +15–40%; context per GiB 16.4k → 31.8k | X-04 | GPU | S | no |
| 4 | H08 (E-18, I-08) | A near-tie + fp32 re-score classifier flags every greedy SPEC=off vs dflash2 divergence as either a near-tie or reproducible, and flags the KVarN adaptive + prefix case as real | P1 | No speed; correctness gate for H02–H05 | X-08 | **CPU-now** (code + synthetic test) + GPU | S | no |
| 5 | H03 (E-20/E-02, T-10, I-01) | The released EAGLE-4 drafter loaded as EAGLE-3 (aux [4,32,60]) reaches ≥3.75 tok/step at k=7, T=0, on the B prompt set (the W4A16 break-even vs DFlash2's 3.36) | P1 | Information gate; speed only via H11 (prior low: needs +12–31% tok/step) | X-03 | GPU | M | no for local eval (§2(a)); yes to ship |
| 6 | H06 (E-12/E-22, I-06) | A CUDA-graph conditional node that skips MTP draft passes after a pos-0 reject captures on the box's driver and raises D C1 ≥2% | P1 | ≤1–2% D C1 | X-06 | GPU | M | no for the CUDA feature; yes if `tree_drafter.py` is ported |
| 7 | H05 (E-10/E-03/E-21, I-04) | Rank-128 retrieval + exact rescoring of 128 rows in DFlash2 `compute_candidates` cuts ≥0.5 ms per step and raises B C1 ≥2% with tok/step ≥ −1% | P1 | +2.2% B C1, +0.4% E at 170k | X-05 (after X-12) | GPU | L | **yes (D11)** |
| 8 | H09 (I-10) | Switching MTP to the full head when draft-vocab coverage drops restores ≥2.0 tok/step on Chinese (from 1.06) with English D C1 within 2% | P2 | Chinese +100–127% (M endpoints 35.5 → 80.6 tok/s); English 0 | X-09 | GPU | M | no |
| 9 | H10 (I-09) | int8 per-token-head KV in mode A matches fp8's 814 tok/s e2e within 3% and keeps drift ≤0.025 nats (fp8 0.036) | P2 | Quality (1.7× less drift, M); speed ±3% | X-10 | GPU | M | no |
| 10 | H11 (E-20 + IP row 5) | If H03 passes: a W4A16 EAGLE-4 layer with an int4 own head (0.86 GiB) keeps tok/step within −3% of the bf16 draft | P2 | Makes H03 servable: −0.34 GiB vs DFlash2 at B/D′/E | X-11 | GPU | S/M | no for eval; yes to ship |
| 11 | H12 (T-01/T-03/T-04/T-09) | SVD-initialised rank-128 factors of our int4 lm_head reach recall@128 ≥ 91.5% (paper SVD bar) on our captured states; ≥98% after training | P2 | Gate for H05 and a no-hole MTP head | X-12 | CPU on a box with weights (not here) | S/M | **yes (D11)** |
| 12 | H13 (T-05) | If H03 is within 10% of break-even: body-only domain adaptation (P-c) lifts EAGLE-4 tok/step to ≥3.75 | P2 | 0–10% tok/step (prior low; our MTP fine-tune was flat, M) | X-13 | GPU | M | no for eval; yes to ship derived weights |
| 13 | H14 (T-06) | Raising the MTP TTT unroll from depth 2 to 4 raises D tok/step ≥3% | P2 | 0–3% (prior low) | X-14 | GPU | M | no |
| 14 | H15 (I-14) | Each listed drift item (NSTRONG/AGREE, `README.md:96-101`, SPEC default, GDN state size) is confirmed against code with file:line | P3 | Docs only | X-15 | **CPU-now** | S | no |
| 15 | H16 (E-24) | With no GPU, importing pthfa/KVarN packages hard-fails rather than falling back; an arch-table fallback removes every hard-fail | P3 | 0 speed; robustness | X-16 | **CPU-now** (probe) | M | no |

</details>

Not ranked: these are no-gos or estimate-only; reasons are in transferability.md. [review: H05 (E-10/E-03/E-21) added here: +2.2% E, below the +3% bar, L effort plus D11.] They are tree verify (E-05/E-14/E-15/E-16/E-17/E-25), the `+eagle.4` fork, Hopper/Blackwell paths, E-23 L2 prefetch, KV sharing (U-07), from-scratch drafter training, and DFlash2 window retraining (E-11/T-08). CPU-now items: H01, H02, H08, H15, H16 (5 of 15). [review: 5 of 14 after H05 left the ranking.] [review (phase 5): read "H02" as H02′, the reviewed fitted-k-table row; the H02 +8-40% figure is the superseded pre-review ranking.] The top speed items still need the GPU box.

## Carry-forward notes for phases 3-5

- Upstream #345 (PTH_FA default on `SPEC=mtp CTX=long`, sm86): the user's own WSL2 3090 A/B on the PR measured +21–22% decode, three boots per arm (PR comment, see `01-project/upstream-diff.md` and `00-meta/reviews/phase1-upstream-diff.md`). The GPU box runs `CTX=fast`, where the default does not fire. Candidate quick win: adopt upstream main, and test PTH_FA in the other modes.
- The EAGLE-4 Qwen3.8-27B draft is 3.09B params (training-recipe lane, header count). Memory fit next to the W4A16 target on 24 GB is the first transferability gate.
- EAGLE-4 draft memory: 3,089,075,200 bf16 params = 6.18 GB, of which embedding + lm_head are 82.3% (body ≈ 0.55B). Sharing the target's embedding and quantising the draft lm_head are the obvious levers for fit; the repo says the draft trains its own lm_head, so sharing it changes the model (reviews/phase1-eagle4-training.md).
- Training-cost central rate: 40 TFLOP/s effective on the 3090 (review recomputation); from-scratch ≈ 53 h per 50M tokens, estimated. Out of scope to run (>2 h rule); proposal only.
- DFlash2 already has a low-rank 16×16 edge score over top-16 candidates and a chain walk; EAGLE-4's low-rank candidate head must be compared against it, not against a plain chain drafter. Verify is chain-only in vLLM 0.31 and HEAD; a tree needs parent-indexed GDN state (dflash-specdec.md §8).
- Docs/code drift: `docs/optimizations.md:303-304` says lookup NSTRONG=8 / AGREE=2; the code defaults to 6/0 (P3 doc fix, pending review confirmation).
- Memory headroom (quant lane, reviewed): usable CUDA total 23.54 GiB; B free 0.7–1.3 GiB, E ≈0.15 GiB; 1 GiB of drafter costs ≈16.4k tokens of one request's context at B, ≈32k at D′. DFlash2 W4A16 is 1.19 GiB, MTP 0.30 GiB. An EAGLE-4 draft (≈3.6 GB bf16 resident, estimated) only fits as a replacement and probably only quantised or with a shared lm_head.
- fp8 KV (default in A and D; D′ defaults to int8 PTH) drifts 1.7× more than int8 against bf16 and fails the repo's non-inferiority test (`docs/long-context.md:466-476`, reviewed). Candidate P2 item: int8 PTH where fp8 is still the default.
- Acceptance (reviewed): DFlash2 3.14/3.34 tok/step; MTP k=4 CTX=fast 2.8/2.9–3.0; mode D measured 2.64 greedy. DFlash2 selector rank 256 = the 16×16 top-k edge score (consistent).
- vLLM (reviewed): no EAGLE-4 support in 0.31.0 or HEAD; it needs an out-of-tree hooks branch; 33 of our 51 patches fail on HEAD (22 are real drift, 11 only depend on earlier series patches), so "move to the EAGLE-4 fork" means re-porting most of the stack. Cheapest probe: chain-mode drafter (M). Drafter quantisation (S) and KV dtype (S/M) stand alone. Tree verify (L) needs bf16 KV and the CUSTOM backend slot pthfa already uses.
- Project map (reviewed): D′ defaults to PTH_FA (CUSTOM int8 KV); A is 948 tok/s e2e [review (phase 5): 948 e2e / ~1,035 decode is the repo's native-Linux single-card figure (`README.md:108`, batch defaults, 64 concurrent, 128/512, measured (reported)). The A/B reference on the GPU box is 814.3 e2e / 904 decode (`04-benchmarks/baseline.md:33`, WSL2, stock sha-2cf50a0, measured). Use 814.3 for box comparisons.]; B pool 68,605 tokens with default prefix cache; the verify kernel is causal-only, so a tree mask needs a kernel change. Docs drift: `README.md:96-101` stale; SPEC default is mtp in `single-user/start_qwen.sh:197` vs dflash2 in `.env.example:6` (P3).
- DFlash (reviewed): DFlash2 core is upstream vLLM 0.31.0; ours are the lookup, chain, adaptive and prewarm patches. 3090 weight-byte floor 13.25 GB / 936 GB/s = 14.2 ms per step (estimated); speculation overhead 4.5–6.5 ms (estimated; bench-12 settles it). Drafter byte floor ≈1.80 ms, lm_head ≈39% of it. vLLM removed tree attention before 0.31 (`b1728c1e66`), so tree verify is a re-introduction, not a flag.
- vLLM review: per-token-head int4/int8/fp8 KV is already upstream in 0.31.0; the EAGLE-4 draft has its own trained lm_head (sharing the target's changes the model); its embedding + lm_head ≈5 GB bf16 (estimated); the `+eagle.4` vLLM build is unpublished; pthfa holds the CUSTOM slot in the default 3090 launch (`resolve_config.sh:87-113`).
- EAGLE-4 architecture (reviewed): the candidate head cuts draft traffic 70.7% (estimated). Our GDN state is fp16, 72 MiB/row, ≈1.2 GiB for tree width 16. Embedding sharing needs `torch.equal`, which our int8 embedding cannot pass. MRV2 is already the default and patched. E-20 (load as EAGLE-3) also needs aux ids [4,32,60] and dropping `embed_tokens`, else 6.18 GB.
