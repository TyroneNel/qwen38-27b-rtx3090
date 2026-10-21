# EAGLE-4: training recipe and data, and what one 3090 can do

Role: eagle-analyst (training recipe and data). Sources pinned in `docs/research/00-meta/plan.md`. EAGLE-4 checkout `~/research-src/EAGLE-4` @ `7b7b5a5`; paper `EAGLE4_Paper.pdf` (12 pages, no appendix; page numbers below are the PDF page = printed footer). HF metadata fetched 2026-10-21.

Labels: **paper** = reported in the EAGLE-4 paper; **repo** = stated in EAGLE-4 code or docs; **HF** = HF API, card, `config.json` or safetensors header (header read with an HTTP range request, no weights downloaded); **ours-doc** = stated in our repo docs, not re-measured here; **estimated** = my arithmetic, formula given. Nothing in this file was measured on a GPU.

## Summary

- **The paper does not describe how the drafter is trained.** It has no training-data section, no hyperparameters, no schedule, no training hardware and no GPU-hours (pp. 1–12, checked in full). The only training facts are in §2.2 (p. 5: inherits EAGLE-3's token-only objective and training-time test, and EAGLE-3.1's norms), §3.2 (p. 6: the low-rank candidate head is trained for retrieval in a second stage with the rest of the drafter frozen), and §3.4 (p. 6: MoE drafts cost more training; the FFN type follows the target's).
- **There is no training code in the repo.** `eagle_harness/` and `tools/` hold serving, calibration and test code only (a grep for optimizer/backward/loss finds only the cost model's "depth loss"). `docs/new-model.md:32-40` points to an external project, TorchSpec, as a "source bundle … from the training maintainer"; "a reproducible repository revision … is not available with this release" and the recipe is "not a verified end-to-end training path".
- What the docs do give is a two-stage recipe: (1) train a dense single-layer drafter (3 aux layers, full vocabulary, rotary) [review: `docs/new-model.md:54-62` does not state these; three aux layers come from :49, full vocabulary from :105, RoPE from :27-28 and HF `rope_parameters`]; (2) freeze it and train rank-128 low-rank candidate factors with `loss_type: forward_kl`, `efficient_lm_top_k: 128` (`docs/new-model.md:54-86`).
- Released: the Qwen3.8-27B drafter weights (HF, 3.09B BF16 params counting the 1.27B embedding and 1.27B own lm_head; 0.44B in the layer), plus an SVD initialiser for the candidate head (`eagle_harness/models/low_rank_head.py:63-98`). Data, training configs and training logs are not released.
- On one 24 GB 3090, **training a full EAGLE-4-style drafter from scratch is borderline**: with lm_head frozen, about 33 h of training compute per 50M tokens at 7 unrolled depths (estimated), plus about 20 h of data generation and a 2 TB activation cache that has to be streamed. **Training only the candidate head is cheap**: about 17 min per epoch over our existing 7.2M-token capture (estimated). Fine-tuning the released drafter's body on 1M of our own tokens fits in under 2 h (estimated).

## Training data

| Question | Answer | Source |
|---|---|---|
| Sources | **Not stated.** The paper names only evaluation sets (SPEED-Bench qualitative split, HumanEval, GSM8K, MATH-500, MT-Bench), pp. 7–9. The HF card has no `datasets` tag and no training section. | paper pp. 7–10; HF card (`cardData` has only license and tags) |
| Size | **Not stated.** | — |
| Regenerated with the target? | **Not stated.** The TorchSpec YAML takes `dataset.train_data_path`, `dataset.chat_template` and `dataset.prompt_key` ("training conversations in the target's template") and **separate inference and training GPU counts** (`inference.inference_num_gpus`, `training.training_num_gpus_per_node`). That suggests TorchSpec runs the target online to produce hidden states and possibly responses (**inferred**, not stated). | `docs/new-model.md:46-52` |
| Thinking-mode data | **Not stated.** The Qwen evaluation in Fig. 3 runs "with thinking enabled" (p. 8), so the drafter is at least used in thinking mode. | paper p. 8, Fig. 3 caption |
| Filtering | **Not stated.** | — |
| Teacher signal | Target hidden states from three layers. The released config names output layers `[3, 31, 59]` of Qwen3.8-27B; vLLM taps the input of layers 4/32/60 (`eagle_harness/vllm/config.py:199-207`). | HF `config.json` `eagle_config.eagle_aux_hidden_state_layer_ids` |
| Token frequency | The optional `weighted_low_rank` head type reads `token_frequency.npy` ("int64 per-token counts of a reference corpus") to weight the SVD. The released Qwen drafter uses plain `low_rank` with trained factors, so its corpus is still unknown. | `eagle_harness/models/low_rank_head.py:44-60`; HF `config.json` |

## Objectives and losses

| Item | What is known | Source |
|---|---|---|
| Base objective | Token prediction only, no feature regression (the EAGLE-3 change). EAGLE-4 "builds upon this autoregressive drafting architecture". | paper p. 5, §2.2 |
| Training-time test / multi-step unrolling | Inherited from EAGLE-3: the drafter's own outputs are fed back into later training steps so it sees its own recurrent inputs. The **unroll depth is not stated**. Serving drafts up to depth 12–15 (p. 7), so the trained depth matters, but it is unreported. | paper p. 5, §2.2; p. 7 |
| EAGLE-3.1 norms | Post-norm of drafter hidden states and per-hidden-state RMSNorm on the captured target states. The released config has `fc_norm: true` and `norm_output: true`; the header has `fc_norm.{0,1,2}`, `layers.0.hidden_norm`. | paper p. 5; HF config/header |
| Sliding-window attention | Window 4096, and the window is applied in training ("Sliding-window semantics, as trained: at draft depth d a query attends to at most W context keys ending at it plus the d draft tokens of its own chain"). | `eagle_harness/models/eagle4_config.py:9-12`; HF `sliding_window: 4096` |
| Dense-stage loss | **Not stated.** The doc only says to train the dense drafter first and keep a checkpoint. | `docs/new-model.md:54-62` |
| Candidate-head stage | Train only the two factors A ∈ R^{V×r}, B ∈ R^{r×d}, rest frozen. The objective is aimed at retrieval: "focuses the limited capacity of the low-rank head on high-probability candidates rather than … the full output distribution" (p. 4). The config is `efficient_lm: true`, `efficient_lm_rank: 128`, `efficient_lm_top_k: 128`, `loss_type: forward_kl`. TorchSpec refuses to start without a pretrained dense drafter. | paper pp. 4, 6; `docs/new-model.md:64-86` |
| Candidate-head result | At rank 128: SVD init costs 0.48 accepted tokens per verify; the trained head's 128 candidates contain the full head's top-1 in **98.2%** of steps (SVD: 91.5%). Acceptance is unchanged within noise, and draft time per cycle falls from 9.1 to 4.8 ms at bs=1. **paper-reported** | paper p. 6, §3.2.1 |
| SVD initialisation | Frequency-weighted truncated SVD: `D^(1/2) W` with `D = freq/mean(freq) + smooth_factor` (default 10), weighting undone in the scores. Used at load time when a checkpoint lacks the factors. | `eagle_harness/models/low_rank_head.py:63-98`; `eagle_harness/sglang/models/llama_eagle4.py:487-505` |
| Distribution matching / acceptance-aware loss | **None described** beyond the forward-KL candidate stage. The acceptance-aware part of EAGLE-4 is at inference (the EAGLE-Harness cost rule ΔA/ΔC ≥ τ, p. 5), not in training. | paper p. 5 |
| MoE drafts | MoE "improves generalization across draft steps and context lengths, but increases memory requirements and training overhead"; the FFN type follows the target's (dense for Qwen). The config schema accepts MoE training provenance keys (`load_balance_loss_alpha`, `seq_aux_loss_alpha`, `moe_initialization`, `moe_mtp_layer_index`, `moe_mtp_source_revision`). The last three suggest MoE drafts are initialised from the target's MTP layer (**inferred**). | paper p. 6, §3.4; `eagle_harness/models/eagle4_config.py:126-142` |
| Evaluation advice | "A low training loss alone does not establish a useful speculative draft"; check held-out candidate recall and served acceptance. Acceptance near one suggests a wrong export, wrong hidden layers or a bad head mapping. | `docs/new-model.md:88-89, 178-181` |

## Schedule and hyperparameters

None are reported: learning rate, optimizer, batch size, sequence length, epochs and steps, TTT depth, warm-up and precision are all missing from the paper, the docs and the HF card. The only numbers are the candidate-stage example values (rank 128, top-k 128), which the docs call "an explicit example, not a tuned choice for your target" (`docs/new-model.md:78-79`). The released config also has rank 128 and 128 candidates (HF `efficient_lm_head_config`). The paper says it uses K ∈ {128, 256, 512} (p. 6).

## Hardware and cost reported

**None.** The paper reports inference hardware only: 4× B300 (pp. 7–9), 1× MI350X (p. 8), AWS c5.xlarge CPU (p. 9) and a Snapdragon 8 Elite phone (p. 10). It gives no training GPUs, GPU-hours or wall time. The only cost-related remark is that MoE drafts raise "training overhead" (p. 6).

## Released artefacts vs paper-only

| Artefact | Status | Where |
|---|---|---|
| Serving plugin (SGLang, vLLM, TokenSpeed) | released, source-available | `eagle_harness/` |
| Training code | **not released** (external TorchSpec bundle from the maintainer; no reproducible revision) | `docs/new-model.md:32-40` |
| Export procedure (`tools/convert_to_hf.py` in TorchSpec) | documented, not shipped | `docs/new-model.md:91-120` |
| Candidate-head SVD initialiser | released | `eagle_harness/models/low_rank_head.py` |
| Training data, configs, logs | **not released** | — |
| Qwen3.8-27B drafter | released, HF `EAGLE-Corp/Qwen3.8-27B-EAGLE4` (`EAGLE-corp/EAGLE4-Qwen3.8-27B` redirects), sha `f58a5985`, created 2026-10-07, last modified 2026-10-19, `torchspec_version 0.1.0`, 6,178,152,384 bytes BF16 | HF API |
| Gemma-4-E2B and Nemotron-3.5 drafters | paper only (p. 2: 134 MB Gemma head; pp. 7–8: Nemotron). The docs mention an "inspected Nemotron draft" (`docs/new-model.md:164-167`). | — |

Drafter tensors (HF safetensors header; counts computed from the shapes): `layers.0.*` 435.2M (q/k/v 5120×10240 each, o 5120², MLP 3×5120×16384, matching Table 1's 0.44B, p. 3); `fc` 5120×15360 = 78.6M; `lm_head` 248320×5120 = 1.27B (the draft's own head: `eagle_harness/vllm/models.py:146`); `embed_tokens` 1.27B (the target's, shared in vLLM, same line); candidate factors `efficient_lm_head` 248320×128 + `down_proj` 128×5120 = 32.4M. 40 heads, 40 KV heads, head_dim 128.

## What is feasible on one 24 GB 3090

### Assumptions (all estimated)

- Effective training throughput **25 TFLOP/s** in bf16. The 3090 peak with fp32 accumulate is about 71 TFLOP/s, so this is about 35% MFU, at the 250 W cap the GPU box runs (`~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/ENVIRONMENT.md:11`). Cross-check: our MTP trainer is documented at about 4k tok/s for a depth-2 run (`drafter/README.md:57`, ours-doc). With its per-token work (about 11 GFLOP: MTP body, full-vocab teacher logits per depth, 40,960-row head) that implies about 44 TFLOP/s. So 25 is conservative, and times below may be up to about 1.8× too high.
- Training FLOPs per token per unrolled depth: `6·P_body + 4·P_lm` (body forward and backward; frozen lm_head forward plus input-grad), plus `2·P_lm` once per token for the target-logit teacher. With P_body = 0.514B (layer + fc) and P_lm = 1.27B: **8.17 GFLOP per depth + 2.54 GFLOP**. At K = 7 depths that is **59.7 GFLOP/token**; K = 3 gives 27.1; K = 1 gives 10.7. Window attention (≤ 4096 keys, about 0.25 GFLOP/token/depth for training) is left out (≤ 3%).
- Memory: fp32 master weights + grads + Adam = 16 B/param. Body only: 0.514B × 16 = **8.2 GB**, plus 5.1 GB of frozen bf16 embedding and lm_head, plus activations with a chunked vocab loss. That fits. [review: the teacher needs the target's own lm_head as well (2.54 GB bf16) unless teacher logits are precomputed, so frozen weights are 7.6 GB and the total about 15.8 GB plus activations; still fits, with less headroom for K = 7 activations.] Training lm_head too: 1.79B × 16 = **28.6 GB**, which does not fit without 8-bit optimizer state or paging. The target (W4A16 fast variant 15 GB, ENVIRONMENT.md:36) cannot sit beside training, so hidden states have to be captured offline first.
- Cache size per token: 3 aux layers + final hidden in bf16 = 4 × 10,240 B ≈ **41 KB/token** (the final hidden gives the teacher logits through the target lm_head, as our trainer does: `drafter/train_mtp.py:292` [review: was :291]).
- Data generation on the 3090: 5.4M output tokens in 2.2 h, so **≈ 680 tok/s** (`drafter/README.md:45`, ours-doc). Capture: 74 GB (≈ 7.2M rows of 5120 bf16) in 1.7 h (`drafter/README.md:46`). Assume the same time for a 4-tap capture if it is prefill-bound (estimated).

### Pilots (< 2 h of GPU each)

| ID | Pilot | Work | Estimate |
|---|---|---|---|
| P-a | **Low-rank candidate head for our MTP drafter.** Freeze MTP; train rank-128 factors over the full 248,320 vocab with forward-KL against the full lm_head (EAGLE-4 stage 2 transplanted). Measure recall@128 of the target's actual next token on response tokens, and compare with the 40,960-row slice's 97.5% coverage (`drafter/README.md:14-22`). Reuses `train_mtp.py`'s data path and MTP forward. | `2·0.42 (MTP) + 2·1.27 (teacher) + 6·0.032 (factors)` ≈ 3.6 GFLOP/token; 7.2M tokens/epoch ÷ 25 TF/s | **≈ 17 min/epoch**; memory < 3 GB beyond the frozen heads. Needs `drafter/data/hidden.npy` still on the GPU box (open question). |
| P-b | **Zero-train SVD baseline** for P-a: frequency-weighted SVD (`low_rank_head.py:63-98`) with token counts from `drafter/data/gen.jsonl`. Needs only the lm_head matrix. Doing an eigendecomposition of the 5120² matrix `WᵀDW` instead of a full SVD makes it a CPU job on a box that has the weights. | minutes | Paper: SVD alone gives 91.5% recall and −0.48 acceptance (p. 6), so expect it as the floor. |
| P-c | **Domain-adapt the released EAGLE-4 drafter** to our traffic (45% thinking, Danish, code: `drafter/collect_prompts.py:128-129`). Body only, lm_head and embed frozen, K = 3–7. | capture 4 taps on 1M tokens ≈ 14 min (1.7 h × 1/7.2) + 41 GB disk; train 1M × 59.7 GFLOP ÷ 25 TF/s | **≈ 40 min (K = 7)**, ≈ 1 h total. Only worth it after (i) our stack can serve the EAGLE-4 drafter (architecture lane) and (ii) its acceptance on our prompts is measured as a baseline. |
| P-d | **Candidate-head retrain for the EAGLE-4 drafter** at a different rank/K matched to 3090 costs (e.g., r = 64). Drafter frozen. | ≈ 3.8 GFLOP/token; 7.2M tokens | **≈ 18 min/epoch**; needs P-c's 4-tap capture. [review: P-c captures only 1M tokens; a 7.2M-token 4-tap cache is 295 GB, above the 130 GB free on the GPU box (ENVIRONMENT.md:17). Either run P-d on P-c's 1M tokens (≈ 2.5 min/epoch) or cache only the frozen drafter's output hidden (10 KB/token, 74 GB).] |

### Full training from scratch (dense stage + candidate stage)

Formula: `GPU-h = D_tokens × epochs × (K·8.17 + 2.54) GFLOP ÷ (25 TFLOP/s × 3600)`. EAGLE-4's own data size is unknown, so D is a scenario, not EAGLE-4's number.

| D (tokens, 1 epoch, K = 7) | 3090 train h | + data gen h (680 tok/s) | + capture h | Cache | 3090 total h (est.) | H100 train h at 300 TF/s (est.) |
|---|---:|---:|---:|---:|---:|---:|
| 7.2M (our current set) | 4.8 | 0 (exists) / 2.9 | 1.7 | 0.3 TB | ≈ 6.5–9.4 [review: was ≈ 7–9; 4.8 + 1.7 = 6.5 with existing data, 4.8 + 2.9 + 1.7 = 9.4 regenerated] | 0.4 |
| 50M | 33 | 20 | 12 | 2.0 TB | ≈ 65 | 2.8 |
| 200M | 133 | 81 | 47 | 8.2 TB | ≈ 260 | 11 |
| 1B | 664 | 410 | 236 | 41 TB | ≈ 1,300 | 55 |

The candidate stage adds about 3.8 GFLOP/token, about 2 h per 50M tokens on the 3090 (estimated).

Constraints that make anything above about 10M tokens impractical locally: the GPU box had 130 GB free (ENVIRONMENT.md:17), so caches must be generated, trained on and deleted in chunks of about 3M tokens (one epoch per chunk). lm_head has to stay frozen or use 8-bit Adam. This matches DeepInfer's rejection of "training any full drafter on the 3090" (`deepinfer-ref/docs/research/deepinfer/drafter.md:185`) for multi-layer drafters. The single-layer EAGLE-4 body is small enough that the frozen-lm_head variant is the exception.

**Cloud estimate** (estimated; prices are placeholders, not checked against a live price list): `cost = GPU-h × $/h`. For 200M tokens with an H100 80 GB at $2–3/h: 11 h of training plus about 14 h of bf16 generation at an assumed 4k tok/s, so about 25 h, **≈ $50–75**. An 80 GB card also allows a trainable lm_head and online target states, which removes the 8 TB cache. A rented 3090 at $0.2–0.4/h for the 260 h path costs **≈ $50–100** but takes about 11 days.

## Comparison with our `drafter/` pipeline

| | Our MTP fine-tune (`drafter/`) | EAGLE-4 (paper + docs) |
|---|---|---|
| Drafter | Qwen's built-in MTP layer, 1 layer, 0.37B layer params (paper Table 1, p. 3) | new single layer, 0.44B layer params + 78.6M fc (HF header) |
| Input taps | final hidden of the target, 1 tap (`drafter/capture.py:1-10`) | 3 aux layers (3/31/59) + per-tap RMSNorm (HF config) |
| Output head | 40,960-row int4 slice of lm_head (`drafter/README.md:14-22`) | own full-vocab lm_head + rank-128 retrieval head, 128 exact-scored candidates (p. 6) |
| Data | 6.8k prompts (UltraChat, Magicoder, Danish sets, GSM8K; 45% thinking), regenerated by the served W4A16 target at default sampling: 5.4M output tokens (`drafter/README.md:43-45`; `drafter/gen_data.py:2`) | not stated |
| Loss | soft CE to the target distribution over the draft vocab (= forward KL + const), weighted by the target's in-vocab mass, response tokens only (`drafter/train_mtp.py:281-307, 338`) | dense: not stated; candidate stage: forward KL, retrieval-focused (`docs/new-model.md:68-75`; p. 4) |
| Unrolling | yes, depth 2 by default in the recipe, feeding its own hidden (`drafter/train_mtp.py:155-177`, `hid = z_prev` at :166) | yes (EAGLE-3 TTT, p. 5); depth not stated |
| Schedule | AdamW β = (0.9, 0.95), wd 0, warm-up 100, cosine to 10% (`drafter/train_mtp.py:425, 434-438`); 1 epoch, lr 3e-5 (`drafter/README.md:56-57`) | not stated |
| Hardware/cost | 3090, about 6 h end to end; about 30 min/epoch (`drafter/README.md:10, 57`, ours-doc) | not stated |
| Outcome | fine-tuning did **not** move acceptance (top-1 0.685 → 0.685); the gains came from the draft vocab and GPTQ (`drafter/README.md:29-37`) | 3.59× vanilla at C = 1, T = 0 on 4× B300 (p. 9, Table 2) [review: added T = 0; at T = 1 it is 2.59×] |

Lesson carried over from our negative result: judge any EAGLE-4-style fine-tune by response-token true-next-token acceptance and served tok/s, not by KL. EAGLE-4's docs say the same (`docs/new-model.md:88-89`).

## Licence constraints on training and derivatives

Not legal advice; flagged for the human.

- The licensed "Software" explicitly includes "model architecture, model weights and parameters, … configuration files, inference and training code" (`LICENSE:90-93`). **Fine-tuning the released drafter creates a Modification.**
- Research, evaluation, personal and internal noncommercial use, including making derivative works, is granted (`LICENSE:97-106`; "Commercial Use does not include genuine internal evaluation, academic research …", `LICENSE:55-58`). Commercial use is allowed only below US $20M aggregate revenue and only if it is not a Competing Offering (`LICENSE:133-141`).
- **Competing Offering** covers anything that "incorporates material Eagle 4 technology into a model, inference, training, acceleration, or serving framework that competes with Eagle 4" (`LICENSE:61-65`). "Eagle 4" includes "EAGLE-Harness and EAGLE Efficient LM Head technology" (`LICENSE:73-75`), and evasion by "rewriting, porting … or by placing material Eagle 4 technology in another framework" is barred (`LICENSE:127-131`). Commercial agreement §8 repeats this. **Risk:** shipping the low-rank candidate head (T-01) in HyperQwen, a serving framework, may fall under this even if we write the code ourselves. Merely interoperating or using outputs is excluded (`LICENSE:67-70`).
- Distributing a fine-tuned drafter requires LICENSE + NOTICE, a modification notice, the complete corresponding source "including … weight deltas" under the same licence (`LICENSE:180-185`), the attribution sentence (`LICENSE:192-197`) and a distinguishable name (`LICENSE:206-210`).
- Patent licence covers only the unmodified Software and permitted uses (`LICENSE:254-262`).
- The licences of TorchSpec and of the Qwen3.8-27B weights are separate. TorchSpec's was not inspected (not in the checkout).

## Candidate techniques (for the transferability lane)

- **T-01** Low-rank (rank 128) retrieval head over the full 248k vocab + exact rescoring of K = 128 gathered lm_head rows, replacing the 40,960-row draft-vocab slice for MTP. Per step it reads about 65 MB bf16 of factors versus about 105 MB for the int4 slice (estimated) and has no vocab hole (p. 6; `docs/new-model.md:122-134`). Licence risk above.
- **T-02** Train the candidate factors with the drafter frozen, forward-KL/retrieval loss, on our existing capture (pilot P-a, about 17 min/epoch, estimated) (p. 6; `docs/new-model.md:64-86`).
- **T-03** Frequency-weighted SVD initialisation of the candidate head, with token counts from our own generations (`eagle_harness/models/low_rank_head.py:63-98`).
- **T-04** Recall@K of the full head's top-1 as the offline gate for candidate heads (98.2% at K = 128 is the paper's bar, p. 6), before any served run.
- **T-05** Domain-adapt the released EAGLE-4 Qwen3.8-27B drafter on our self-distilled thinking/Danish/code data, body only, lm_head frozen (pilot P-c; about 1 h, estimated).
- **T-06** Longer TTT unroll in our MTP trainer (depth 2 today, `drafter/train_mtp.py:155-177`) to match the deeper chains EAGLE-4 drafts (12–15, p. 7). Low prior given our flat depth-2 result (`drafter/README.md:29-37`).
- **T-07** Three-tap target features (3/31/59) with per-tap RMSNorm and post-norm output (EAGLE-3.1, p. 5; HF config). Only applies if we train our own non-MTP drafter.
- **T-08** Train the drafter with the sliding window it is served with (W = 4096) so long-context drafting costs stay flat (`eagle4_config.py:9-12`; p. 6 §3.3.1). Relevant to long-context mode; overlaps deepinfer's long-context drafter notes (`drafter.md:17`).
- **T-09** Two-stage recipe: dense drafter first, then candidate factors from the frozen dense checkpoint (`docs/new-model.md:54-86`).
- **T-10** Export sanity gate: acceptance ≈ 1 means wrong layers or head mapping; check tensor shapes from the safetensors header before serving (`docs/new-model.md:122-162, 178-181`).

## Open questions

1. What data trained the Qwen3.8-27B drafter: sources, token count, target-regenerated or not, thinking share? Paper, docs and HF card are silent.
2. Dense-stage loss, TTT depth, LR/batch/steps and training GPU-hours: unreported. Is the TorchSpec bundle (and its `configs/recipes/`, `docs/new-model.md:40`) obtainable, and under what licence?
3. Was the drafter's own `lm_head` initialised from the target's and then trained, or trained from scratch? Comparing a few rows against the target's lm_head (a range request on the GPU box) would answer it.
4. Do `drafter/data/hidden.npy` (74 GB) and `gen.jsonl` still exist on the GPU box? P-a and P-b depend on them.
5. Can our pinned vLLM load `LlamaForCausalLMEagle4` and tap layers 4/32/60? This is for the architecture/inference lane, and P-c and P-d depend on it.
6. Does putting a self-written low-rank candidate head into HyperQwen count as "Competing Offering" use of "EAGLE Efficient LM Head technology"? This needs a human or legal decision before T-01 ships.
7. The effective 3090 training TFLOP/s is not measured: estimates use 25, while our trainer's documented 4k tok/s implies about 44. One timed P-a epoch would settle it.

## Reviewer additions

Added by the phase-1 reviewer; details and arithmetic in `docs/research/00-meta/reviews/phase1-eagle4-training.md` and `phase1-eagle4-training-calc.out`.

- **Open question 7, effective throughput (estimated).** Recomputing our trainer's depth-2 work per token: 6·0.42B MTP body, plus 4·0.21B frozen 40,960-row head and 2·1.27B full-vocab teacher, both **per depth** (`drafter/train_mtp.py:292`, called per depth at :345). That gives 11.8 GFLOP/token, or 10.1 if head and teacher run only on the 75% response rows. At 4k tok/s this implies **41–47 TFLOP/s** (57–67% of the 71 TFLOP/s dense peak). Use **40 TFLOP/s as the central case** and keep 25 as the pessimistic bound. At 40: P-a ≈ 11 min/epoch, P-c ≈ 25 min of training, and 50M tokens from scratch ≈ 21 h of training, ≈ 53 h in total. Caveats: "4k tok/s" and "~30 min/epoch" are the same rounded claim (7.2M ÷ 4000 = 30 min), not two independent measurements. The README run was by upstream at an unknown power limit. One timed P-a epoch still settles it.
- **Teacher recomputation.** The FLOP model computes the teacher once per token. A trainer written like ours recomputes it at every depth, so K = 7 costs 75.0 GFLOP/token instead of 59.7 (+26%).
- **The frozen lm_head departs from EAGLE-4.** The repo says "The draft trains its own lm_head" (`eagle_harness/vllm/models.py:146`). That partly answers open question 3: the head was trained, and its initialisation is still unknown. The faithful variant, with a trainable lm_head, costs 77.5 GFLOP/token at K = 7 and 28.6 GB of fp32 optimizer state. So "from scratch with lm_head frozen" is our adaptation, not the EAGLE-4 recipe.
- **Licence (not legal advice).** Two clauses add to the T-01 risk. First, §3.a bars anyone who would "develop, offer, sell, license, host, provide, or otherwise commercialize a Competing Offering", "regardless of Aggregate Revenue" (`LICENSE:122-125`). Second, "Eagle Files" are "files or other materials that contain, reproduce, modify, implement, port, or are derived from any portion of the Software" (`LICENSE:77-78`), and moving such code into another framework keeps the licence's obligations (`LICENSE:216-221`). A self-written low-rank candidate head in HyperQwen could therefore be an Eagle File even under noncommercial use.
