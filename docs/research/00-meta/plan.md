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
| 3 Transferability | `02-eagle4/transferability.md`, ranked hypotheses below; independent review | pending |
| 4 Experiments | CPU-feasible experiments here; GPU experiments packaged for the GPU box | pending |
| 5 Findings | `05-findings/*`, roadmap; review | pending |
| 6 Report | `REPORT.md`, fact check, README, commit, push branch to fork | pending |

## Ranked hypotheses

Filled in phase 3.

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
- Project map (reviewed): D′ defaults to PTH_FA (CUSTOM int8 KV); A is 948 tok/s e2e; B pool 68,605 tokens with default prefix cache; the verify kernel is causal-only, so a tree mask needs a kernel change. Docs drift: `README.md:96-101` stale; SPEC default is mtp in `single-user/start_qwen.sh:197` vs dflash2 in `.env.example:6` (P3).
- DFlash (reviewed): DFlash2 core is upstream vLLM 0.31.0; ours are the lookup, chain, adaptive and prewarm patches. 3090 weight-byte floor 13.25 GB / 936 GB/s = 14.2 ms per step (estimated); speculation overhead 4.5–6.5 ms (estimated; bench-12 settles it). Drafter byte floor ≈1.80 ms, lm_head ≈39% of it. vLLM removed tree attention before 0.31 (`b1728c1e66`), so tree verify is a re-introduction, not a flag.
- vLLM review: per-token-head int4/int8/fp8 KV is already upstream in 0.31.0; the EAGLE-4 draft has its own trained lm_head (sharing the target's changes the model); its embedding + lm_head ≈5 GB bf16 (estimated); the `+eagle.4` vLLM build is unpublished; pthfa holds the CUSTOM slot in the default 3090 launch (`resolve_config.sh:87-113`).
- EAGLE-4 architecture (reviewed): the candidate head cuts draft traffic 70.7% (estimated). Our GDN state is fp16, 72 MiB/row, ≈1.2 GiB for tree width 16. Embedding sharing needs `torch.equal`, which our int8 embedding cannot pass. MRV2 is already the default and patched. E-20 (load as EAGLE-3) also needs aux ids [4,32,60] and dropping `embed_tokens`, else 6.18 GB.
