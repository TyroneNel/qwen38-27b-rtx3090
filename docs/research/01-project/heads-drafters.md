# Draft heads and drafters

Lane: quant-analyst. Scope: the drafters this repo serves today, what they cost in memory, how often their drafts are accepted and what `drafter/` can train on one 24 GB card. No GPU was used. Every number is labelled **M** (measured, with a path to the evidence) or **E** (estimated, with the formula).

- Parameter and byte counts come from safetensors headers and `config.json` fetched by range request at pinned commits: [weights_breakdown.py](weights_breakdown.py) → [weights_breakdown.out](weights_breakdown.out), cited as "`.out` §n".
- DeepInfer items are cited by ID (drafter-n, quant-n). Their GPU results are cited by file under `~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/`.
- EAGLE-4 is covered by another lane and is not analysed here.

Mode letters are as in [quantisation.md](quantisation.md): B is dflash2 fast k=7, C is k=15, D is mtp long k=3, D′ is dflash2 long, E is dflash2 huge.

## Summary

- **Two drafters ship.**
  - **MTP** is Qwen's own one-layer head: 425M params, int4 GPTQ, 0.30 GiB with its 40,960-row draft head. It drafts 3 tokens in mode D.
  - **DFlash2** is incoai's 5-layer block drafter: 1.92B params, requantised by this repo to W4A16, 1.19 GiB. It drafts 7 tokens per step in B, C, D′ and E and reads target layers 5/19/33/47/61.
- **Two other drafting paths exist.** Context lookup drafting is always on with DFlash2. DSpark was tested and rejected.
- **Acceptance at C1, thinking off, short prompts** (M, `docs/optimizations.md:384-398`):

  | Drafter | tok/step default sampling | tok/step greedy | tok/s |
  |---|---|---|---|
  | MTP k=3 [review: k=4 at `CTX=fast` (the row builds on `docs/optimizations.md:390,392`; split-KV verify is bf16/`CTX=fast` only, `start_qwen.sh:198-199,212`), not mode D. Mode D itself (k=3, `CTX=long`, pthfa) measured 2.64 tok/step greedy at C1 on WSL2, DeepInfer `baseline.md:42`] | 2.8 | 2.9-3.0 | ~114 / 118-124 |
  | DFlash2 | 3.14 | 3.34 | 118 / 126 |

- **Thinking on lowers acceptance on 4k prompts** (M, DeepInfer `evals-4.md`): B goes 3.49 → 3.14 tok/step, about −15% tok/s. Reasoning text in long-context tasks drafts *better* (4.42 tok/step).
- **Size against acceptance.**
  - DFlash2 int4 keeps greedy acceptance, loses about 5% under sampling, and saves 2.4 GiB and ~3.4 ms per step. That net gain is what makes DFlash2 worth shipping.
  - Both "spend more bytes" variants tried gave nothing: a bf16 `fc` (+0.26 GB), and a 49k-row draft head instead of 41k.
- **On one 3090, `drafter/` can** regenerate data, capture hidden states, GPTQ the heads and fine-tune MTP. The full pipeline takes about 4-5 h and 74 GB of disk (M). It **cannot** train or fine-tune DFlash2. Full Adam needs about 31 GB (E, DeepInfer drafter-4), and no DFlash2 training forward exists.

## The drafters

### MTP (mode D; also loaded in A only if speculation is turned on)

| Property | Value | Label, source |
|---|---|---|
| Architecture | `fc` concatenates the embedding of the next token with the target's last hidden state (10240 → 5120). Then one full-attention decoder layer: q 12,288 (24 heads × 256 × 2 for the output gate), k/v 1,024 each (4 heads × 256), MLP 17,408. Then a shared final norm and a 40,960-row draft head | M, header shapes, `.out` §2; `weights_breakdown.py:110` |
| Target layers read | the last layer's hidden state only (Qwen MTP); the head's input embeddings are the target's own `embed_tokens` | M, header (no `mtp.embed_tokens` tensor) |
| Parameters | 424,699,392: linears 372,244,480, fc 52,428,800, plus norms. The draft head adds 40,960 × 5,120 = 209.7M (shares lm_head's rows) | M, `.out` §2; E for the head |
| On disk / in VRAM, fast (shipped default) | int4 GPTQ: linears 191,938,672 + fc 27,033,616 + draft head 108,134,416 B → **0.305 GiB** (M, `.out` §1/§4) | M |
| Base variant | int8 RTN linears + fc + draft head: 0.600 GiB (E, `.out` §3: rows×cols + rows×cols/128×2) | E |
| bf16 as Qwen ships it | 744,488,960 + 104,857,600 B = 0.79 GiB without a draft head ("850 MB, ~3 ms per draft", `docs/optimizations.md:77-83`) | M |
| KV it adds | one attention layer: 4 KV heads × 256 × 2 (K, V) × 1 B = 2,048 B/token at fp8, 4,096 at bf16. That is +1/16 on top of the target's 16 layers | E, `single-user/start_qwen.sh:592-596` |
| How it was trained | Qwen trained it jointly with the target; this repo has no training record for it. Our changes: int8 RTN (`prepare/quant_mtp.py:35-37`); later int4 GPTQ with Hessians from the drafter's own inputs (`drafter/train_mtp.py --dump-hessians`, `drafter/requant_mtp_gptq.py`, `drafter/README.md:23-28`); draft vocabulary of 40,960 rows counted over the model's own outputs (`prepare/build_draft_vocab.py`, `drafter/README.md:14-22`) | M |

### DFlash2 (modes B, C, D′, E)

| Property | Value | Label, source |
|---|---|---|
| Architecture | `fc` (5 taps × 5,120 → 5,120). Then 5 Qwen3 layers, all sliding-window 2,048, non-causal over a block of 8 (7 drafts + 1 anchor); 32 Q heads / 8 KV heads × 128; MLP 17,408; a "dynamic conv" (group 16, kernel 2) before attention and before the MLP of each layer; mask token 248,070. Then a **candidate selector**: two [248,320 × 256] codebooks with `selector_top_k` 16, so the output is a low-rank scoring over the full vocabulary, walked to 16 candidates per position. No lm_head or embedding of its own (uses the target's) | M, `config.json` @015e7956 (`dflash_config`: block_size 8, selector_rank 256, selector_top_k 16, conv_group_size 16, conv_kernel_size 2); header shapes `.out` §2 |
| Target layers read | 5, 19, 33, 47, 61 (`target_layer_ids`), one every 14 layers | M, config.json |
| Parameters | 1,924,404,480: mlp 1,336,934,400; attention 262,145,280; fc 131,072,000; codebooks 63,569,920 each; dynamic convs 32,870,400 each; hidden_projection 1,310,720 | M, `.out` §2 |
| On disk, bf16 (incoai@015e7956) | 3,848,808,960 B = **3.584 GiB**. The repo's "3.85 GB" is decimal | M, `.out` §1 |
| On disk / in VRAM, W4A16 (syvai@4d30ec73, shipped) | 1,280,617,536 B = **1.193 GiB**. 1,730,150,400 params int4 (35 layer matrices + fc); 194,254,080 bf16 (candidate selector, `hidden_projection`, `kernel_projection`, norms; ignore list `drafter/quant_dflash2.py:100`) | M, `.out` §1/§2 |
| KV it adds | 5 sliding layers × 8 KV heads × 128 × 2 × 2 B = 20,480 B/token in bf16, capped at 2,048 tokens per request → ≤ 40 MiB per request (E). Its per-token row (2,112 B/layer at int8) sets the page size through `max(tgt, dft)` (`single-user/start_qwen.sh:592-604`) | E |
| How it was trained | By incoai; data and selector loss not published (DeepInfer `drafter.md` open question 5). This repo only requantises: Hessians from the drafter's own inputs over 400 prompts (`drafter/capture_dflash2.py:1-15`), GPTQ int4 g128 (`drafter/quant_dflash2.py:28-31`), compressed-tensors export | M (our steps); unknown (training) |

DeepInfer's `drafter.md` comparison table (around :24-38) infers a selector rank of about 384. The checkpoint config says 256 (M). [review: confirmed by re-fetching `config.json` @015e7956 and by the codebook shape 63,569,920 / 248,320 = 256. The DFlash lane's "16×16" (`dflash-specdec.md:106-108`) is a different quantity: `selector_top_k` 16 candidates per slot give a K×K = 16×16 edge-score matrix between neighbouring slots, each score a rank-256 bilinear product of the two codebook rows. The two docs agree; DeepInfer's 384 (`drafter.md:38`, labelled inferred) is wrong.]

### Context lookup drafting (DFlash2 only)

`patches/dflash2-lookup-drafting.patch`, `VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}` (`single-user/start_qwen.sh:367-370`). When the output reproduces the context, draft positions come from the request's own tokens rather than the drafter. With `DFLASH_TOKENS=15` (mode C) the verify block grows to 16, and the extra 8 positions are filled only from lookup. The block length adapts and goes long only while the lookup fires (`:371-381`). The drafter itself always proposes 7.

It costs no weights. It costs graph memory (~1.8 vs ~1.45 GiB, `start_qwen.sh:514-516`) and state pages (1.49 vs 0.74 GiB per request, [quantisation.md](quantisation.md)).

### DSpark (tested, rejected)

RadixArk/Qwen3.8-27B-DSpark, bf16, 7 drafts per step. It needs `patches/dspark-draft-quant-config.patch` and an architecture rename (`docs/long-context.md:619-631`; launcher `DRAFT_METHOD=dspark`, `single-user/start_qwen.sh:391-393`).

Measured on a 3090 against the shipped DFlash2 W4A16 on the same requests: **3.37 vs 3.83 tok/step, 114 vs 150 tok/s** (M, `docs/long-context.md:629-631`). The run used `KV_MEM=3000000000` because the drafter is bf16. Its parameter count was not fetched here.

## Measured acceptance

"tok/step" means accepted drafts + 1 (bonus token). "pos0" means the share of first-position drafts accepted.

**By drafter and sampling, C1, thinking off, 8 prompts × 1k tokens.** Spread is ±3% greedy and ±5% sampled per run (`drafter/README.md:111-120`).

| Config | tok/s default / greedy | tok/step default / greedy | pos0 default / greedy | Source |
|---|---|---|---|---|
| no speculation | 46 / 46 | 1.0 | — | M, `docs/optimizations.md:387` |
| MTP-2 bf16, full head | 66 / 79 | 2.1 / 2.4 | 65% / 80% | M, `:388` |
| MTP fast variant (D's head, shipped) [review: measured at k=4, `CTX=fast`, not k=3; see the summary note] | ~114 / 118-124 | 2.8 / 2.9-3.0 | 74% / 77% | M, `:395` |
| DFlash2 W4A16, k=7 (B) | 118 / 126 | 3.14 / 3.34 | ~75% / ~78% | M, `:396` |
| + lookup (default) | 130 at C1; up to 259 reproducing context | 3.3-7.8 | | M, `:397` |
| + `DFLASH_TOKENS=15` (C) | 133 at C1; up to 381 | 3.4-15.0 | | M, `:398` |
| DFlash2 bf16 vs W4A16 | | greedy 3.54 vs 3.34-3.65; default 3.4 vs 3.2 | | M, `drafter/README.md:89-93` |

**Greedy against sampled, B** (M, DeepInfer `gpu-results/spec-5.md`):

| Condition | tok/step | Σmin(p,q) at pos0 |
|---|---|---|
| T=0 | 3.486 | 0.695 |
| T=0.7, thinking off | 3.611 | 0.690 |
| T=0.7, thinking on | 3.341 | 0.742 |

- Per-position realised acceptance: 0.687, 0.511, 0.404, 0.325, …, 0.190 at pos 6.
- Sampled is *not* worse than greedy on this protocol. That contradicts the 3.14 / 3.34 split above, which came from different prompts and an older build.

**Thinking on against off, C1** (M, DeepInfer `gpu-results/evals-4.md:28-52`):

| Mode | ~4k ladder prompts: tok/step off → on | tok/s change | Long-context tasks: off (20k, answer) vs on (8k, reasoning only) |
|---|---|---|---|
| B (DFlash2) | 3.49 → 3.14 | 117.6 → 100.2 (−15%) | 3.02 vs 4.42; accept at pos 0..3 with thinking on 0.82 / 0.65 / 0.52 / 0.44; math reasoning 4.93, qa 3.18 |
| D (MTP) | 2.80 → 2.67 | −9.7% | 3.02 vs 3.22 |

`evals-4.md:9` labels D's arm "pthfa default". In this checkout `SPEC=mtp CTX=long` defaults to `PTH_FA=0` with fp8 FlashInfer (see quantisation.md open question 3).

**Per mode, long context and KV format:**

| Mode / condition | tok/step | Source |
|---|---|---|
| MTP, 112k context, fp8 KV | 2.56 (68.1 tok/s) | M, `docs/long-context.md:50-58` |
| MTP, 112k context, KVarN KV | 2.38 (32.0 tok/s) | M, same |
| MTP, pthfa int8 vs TRITON int8 (3080 TP=2) | 3.642 vs 3.631 | M, `docs/long-context.md:274-276` |
| DFlash2 D′, GSM8K window, three KV arms | accepted length 4.789-4.838 of 7 | M, `docs/long-context.md:416,449` |
| E bare-metal / copy task | 3.15 (67 tok/s) / 7.83 (164 tok/s) | M, `docs/long-context.md:556-575` |
| DFlash2 at 12-36k context | 2.3-2.6, vs MTP 2.6-3.0 | M, `docs/optimizations.md:350-354` |
| MTP draft vocab on Chinese output (6.4% coverage) | 1.06 (35.5 tok/s) vs 2.23 (80.6) with `MTP_DRAFT_VOCAB=0` | M, `docs/gotchas.md:1436-1472` |

At concurrency 8, D reaches 423 tok/s against B's 236 (M, DeepInfer `evals-10.md`). The larger drafter's fixed cost dominates once the batch fills.

## What `drafter/` can train today on one 24 GB GPU

Pipeline (`drafter/README.md:43-57`). All timings M on the reference 3090:

1. `collect_prompts.py`: 6.8k prompts from UltraChat, Magicoder, syvai/da-instruction, syvai/reasoning-v1, skolegpt-instruct and GSM8K; 45% thinking on.
2. `gen_data.py`: the served target writes its own responses. Thinking capped at 2,048 tokens, non-thinking at 1,024 (`drafter/gen_data.py:1-20`). Takes **2.2 h** for 5.4M tokens.
3. `capture.py`: an in-process vLLM V1 hook dumps last-layer hidden states [T, 5120] bf16 to a memmap (`drafter/capture.py:1-12`). Takes **1.7 h** and **74 GB** of disk.
4. `gptq_lm_head.py`: int4 GPTQ of lm_head from 300k calibration rows. Then `build_draft_vocab.py` writes the 40,960-row int4 draft head.
5. `train_mtp.py`: self-distillation of the MTP module on the target's own distribution, CE/KL over the draft vocabulary with unrolled depths (`drafter/train_mtp.py:1-18`).
   - Defaults: 2 epochs, lr 2e-5 (head 1e-5), warmup 100, 32,768 tokens per step in micro-batches of 8,192, AdamW (0.9, 0.95) (`:30-55, 425`).
   - Takes **~30 min per epoch at 4k tok/s** (M). Writes `mtp_bf16.safetensors` (`:452`).
   - `--eval-only` runs a greedy chain simulation that reproduces vLLM to 1%. `--dump-hessians` feeds GPTQ.
6. DFlash2 requantisation only: `capture_dflash2.py` (~20 min, 400 prompts, ~56 GB scratch) and `quant_dflash2.py` (~40 s; needs the GPU free for the 25,600² `fc` Hessian) (`drafter/README.md:64-84`).

Memory and feasibility:

- **MTP fine-tune fits.** 425M params × (2 B weight + 2 B grad + 8 B Adam + 4 B fp32 master) ≈ 6.8 GB (E), plus the frozen 40,960-row head and activations. It has run on the card (M, `drafter/README.md:29-37`).
  - **Result: negative.** Top-1 stayed at 0.685 → 0.685, and the chain simulation went 2.5 vs 2.6.
  - The fine-tune is kept "for the record" and is not shipped.
- **DFlash2 fine-tune does not fit.** 1.92B params × 16 B ≈ 31 GB for Adam alone (E; DeepInfer drafter-4 says it needs an 80 GB GPU).
  - No training forward exists either (the selector loss is unpublished; DeepInfer drafter-3 and drafter-4).
  - LoRA or selector-only training would fit by arithmetic, but neither is built (E).
- **Disk is the binding cost for wider captures.** DeepInfer drafter-4 would need 5 taps plus top-k logits. That is 6 × the current 74 GB per 5.4M tokens ≈ 0.44 TB (E, from 74 GB × 6), consistent with DeepInfer's 0.4-3 TB.

## Size against acceptance

| Change | Bytes | Acceptance effect | Speed effect | Source |
|---|---|---|---|---|
| MTP bf16 → int8 RTN → int4 GPTQ | 0.79 → 0.60 → 0.30 GiB | int4 RTN costs ~2% acceptance; GPTQ keeps it | −1.8 ms per step (108.6 → 118.8 tok/s greedy with GPTQ lm_head + MTP) | M, `drafter/README.md:23-28` |
| Draft head: full 248k rows → 40,960 | 1.20 → 0.10 GiB (int4) | coverage 92.1 → 97.5% of the model's own tokens; 49k rows no faster | 98.0 → 108.6 greedy, 90.0 → 107.4 default tok/s | M, `drafter/README.md:14-22` |
| Same head, Chinese output | — | 6.4% coverage; 2.23 → 1.06 tok/step | 80.6 → 35.5 tok/s | M, `docs/gotchas.md:1436-1472` |
| DFlash2 bf16 → W4A16 | 3.58 → 1.19 GiB | greedy kept (3.34-3.65 vs 3.54); default −5% (3.2 vs 3.4); relative weight error 0.147 | 31.4 → 28 ms per step (26.5 with the fast target); pool 21k → 69,758 tokens at B | M, `drafter/README.md:89-105`; `docs/optimizations.md:228-244` |
| DFlash2 `fc` int4 → bf16 | +0.26 GB | 3.17 vs 3.17 | — | M, `drafter/README.md:94` |
| DFlash2 k/v Hessian with context-KV blend | 0 | −7% greedy (3.34 → 3.12) | 126 → 118 tok/s | M, `drafter/README.md:95-99` |
| MTP (0.30 GiB) → DFlash2 (1.19 GiB) at C1 | +0.89 GiB | +12% tok/step (2.8 → 3.14 default) | +3.5% default, +4% greedy tok/s; costs ~12k pool tokens at B's exchange rate (E, `.out` §6) | M, `docs/optimizations.md:395-396` |
| Same, at 12-36k context or N=8 | +0.89 GiB | DFlash2 2.3-2.6 vs MTP 2.6-3.0 | D 423 vs B 236 tok/s at N=8 | M, `docs/optimizations.md:350-354`; DeepInfer `evals-10.md` |
| DSpark bf16 instead of DFlash2 W4A16 | larger (bf16; not fetched) | 3.37 vs 3.83 | 114 vs 150 tok/s | M, `docs/long-context.md:629-631` |
| MTP k=4 → k=5 | 0 | — | 106 / 105 tok/s vs higher at k=4 | M, `docs/optimizations.md:405` |

Read-time cost of drafter bytes: at 936 GB/s, every GB read per step costs at least ~1.07 ms (E). DFlash2 W4A16 is therefore ≥ 1.4 ms per step against measured +5 ms for bf16 (`docs/optimizations.md:228-238`). Most of the drafter's step cost is not weight bandwidth.

Memory cost: each extra GiB of drafter takes about 13.4k pool tokens at B, about 26k at D′ and about 55k at E (E, `.out` §6; see quantisation.md headroom). [review: logged-pool units; the context one request loses is ≈16.4k at B and ≈32k at D′, see quantisation.md Reviewer additions.]

## Open questions

1. **Whether greedy or sampled acceptance is better differs between sources.** `docs/optimizations.md:396` shows default 3.14 < greedy 3.34. DeepInfer `spec-5.md` shows T=0.7 3.61 > T=0 3.49. Which protocol should gate drafter decisions?
2. **DFlash2 training data and selector loss are unpublished.** Without them a DFlash2 retrain (DeepInfer drafter-3/4) must reconstruct the selector forward. Is a TV-loss repair of the int4 drafter (drafter-3, ~+3%, estimated) worth that work?
3. **The int4 DFlash2 loses ~5% at default sampling but not greedy.** Would GPTQ fixes (DeepInfer quant-3), or keeping the attention or selector in int8, recover it, and what would that cost in bytes and pool tokens?
4. **The draft vocabulary's Chinese failure (gotcha 61)** is a 2× slowdown with no automatic fallback. Should the vocabulary be language-aware, or the full head used when coverage drops?
5. **DFlash2's drafter KV at long context** is estimated here at ≤ 40 MiB per request. The page-size coupling `max(tgt, dft)` was not measured separately.
6. **Thinking on vs off** was measured at different context lengths (20k answer vs 8k reasoning, `evals-4.md:41`). A paired same-context measurement is missing.

## Reviewer additions

[review: phase 1 review, `docs/research/00-meta/reviews/phase1-quant-heads.md`]

- Mode D's own C1 acceptance (k=3, `CTX=long`) is lower than the shipped-MTP row above: 2.64 tok/step greedy on the pthfa arm (M, DeepInfer `baseline.md:38,42`, WSL2) and 2.5 / 2.4 on the older fp8 FlashInfer arm (M, `docs/optimizations.md:391`). Against B's 3.36 on the same box (`baseline.md:38`) the DFlash2 lead at C1 is ≈+27% tok/step, not +12%.
- The DeepInfer runs cited here (`evals-4`, `evals-10`, `spec-5`, `baseline`) all ran image `sha-2cf50a0` (`ENVIRONMENT.md:24`), where mode D defaults to `PTH_FA=1`; in this checkout it does not.

