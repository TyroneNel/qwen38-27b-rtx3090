# Review: `docs/research/02-eagle4/training-recipe.md` (phase 1, EAGLE-4 training)

Reviewer role: P2 heads/drafters. Sources: `~/research-src/EAGLE-4` @ `7b7b5a5` (paper read in full: text extracted with `pdftotext`, pp. 3–6 also read as images), HF `EAGLE-corp/EAGLE4-Qwen3.8-27B` (API, `config.json`, card, safetensors header via range request on 2026-10-21; no weights downloaded), and our `drafter/`. No GPU was used. Recomputation script: `phase1-eagle4-training-calc.py`, output in `phase1-eagle4-training-calc.out` (both in this directory).

## Verdict

**PASS WITH CORRECTIONS.** I checked 15 claims: 14 are correct (3 of them needed a small citation, context or completeness fix) and 1 is wrong (an internal inconsistency). Every number in the cost tables reproduces to the stated rounding. The corrections are small: a cache-size conflict in pilot P-d, a missing frozen tensor in the memory budget, one table cell, one line number and two citation/context fixes. The doc's own open question on throughput (25 vs 44 TFLOP/s) is resolved: the implied rate is 41–47 TFLOP/s, so its times run about 1.6× high. The largest gaps were missed rather than wrong: the frozen draft lm_head is not EAGLE-4's recipe, and two licence clauses widen the T-01 risk.

## Checked claims

| # | Claim | Doc location | Evidence | Result | Note |
|---|---|---|---|---|---|
| 1 | Paper has 12 pages, no appendix, and no training data, hyperparameters, hardware or GPU-hours | l. 3, 9, 45, 49 | `pdfinfo`: 12 pages. "train" occurs only on pp. 4–6 plus reference titles on pp. 11–12. p. 12 is references only. HF card (90 lines) has Results/Resources/License/Citation and no training section | CORRECT | The training facts are exactly the three the doc lists: p. 5 §2.2, p. 6 §3.2.1 and p. 6 §3.4 |
| 2 | No training code in the repo. `docs/new-model.md:32-40` points to an external TorchSpec bundle with no reproducible revision, "not a verified end-to-end training path" | l. 10, 56 | `docs/new-model.md:34-40` quoted exactly. A grep for optimizer/backward/loss/TorchSpec in `eagle_harness tools tests` finds only `depth_loss` in the cost model and provenance keys in `eagle4_config.py:128-132` | CORRECT | |
| 3 | Two-stage recipe: a dense drafter first, then rank-128 factors trained with everything else frozen, `forward_kl`, top-k 128 | l. 11, 36 | `docs/new-model.md:54-86`. Paper p. 6: "we train the two factors for candidate retrieval alone, with the rest of the drafter frozen" | CORRECT, one citation fixed | "(3 aux layers, full vocabulary, rotary)" is not in :54-62. It comes from :49, :105, :27-28 and the HF `rope_parameters`. Note that `forward_kl` comes from the docs only; the paper never names the loss |
| 4 | 98.2% vs 91.5% recall of the full head's top-1 within 128 candidates; SVD init costs 0.48 accepted tokens; 9.1 → 4.8 ms at bs = 1; acceptance unchanged within noise | l. 37 | Paper p. 6, §3.2.1, right column | CORRECT | paper-reported |
| 5 | 3.09B BF16 params: 1.27B embedding, 1.27B own lm_head, 0.44B layer | l. 12, 60, 63 | Header (1,976 B, `torchspec_version 0.1.0`): 19 tensors, all BF16, **3,089,075,200** params. 6,178,150,400 data bytes + 8 + 1,976 = **6,178,152,384** = HF `content-length`. `embed_tokens` 1,271.40M (41.2%), `lm_head` 1,271.40M (41.2%), `layers.0` 435.17M (14.1%), `fc` 78.64M (2.5%), candidate factors 32.44M (1.1%), norms 0.03M | CORRECT | measured from the header. Embedding plus lm_head is **82.3%** of the checkpoint. `q/k/v_proj` are `[5120, 10240]` (in = 2H, embedding ⊕ hidden), `o_proj` `[5120, 5120]`, MLP 3 × 5120 × 16384 |
| 6 | Aux taps `[3, 31, 59]`; vLLM taps the input of layers 4/32/60 | l. 24 | HF `config.json` (identical to `eagle_harness/core/profiles/mi355x-qwen3.8-27b-tp1/draft_config.json`); `eagle_harness/vllm/config.py:199-207` (`layer + 1`) | CORRECT | |
| 7 | SVD init `D^(1/2) W`, `D = freq/mean + smooth` (10) | l. 38 | `eagle_harness/models/low_rank_head.py:63-98`; `sglang/models/llama_eagle4.py:487-505` | CORRECT | |
| 8 | "Sliding-window semantics, as trained" | l. 34 | `eagle_harness/models/eagle4_config.py:9-12`; config `sliding_window: 4096` | CORRECT | |
| 9 | Serving depth 12–15 | l. 32 | Paper p. 7: 12-4-16 at C = 1, 15-1-16 at C = 512 | CORRECT | |
| 10 | 3.59× vanilla at C = 1 | l. 113 | Paper p. 9, Table 2 | CORRECT, context added | This is at T = 0. At T = 1 it is 2.59×, and MTP is 2.22×/1.57× |
| 11 | Our trainer: 4k tok/s, depth 2, soft CE weighted by in-vocab mass, AdamW (0.9, 0.95) | l. 69, 109–112 | `drafter/README.md:56-57`; `drafter/train_mtp.py:281-322, 425, 434-438` | CORRECT | The teacher line is `:292`, not `:291` (fixed) |
| 12 | Cost estimates: 17 min/epoch (P-a), ~1 h (P-c), ~65 h per 50M tokens, and the full table | l. 79-95 | Recomputed in `phase1-eagle4-training-calc.out` | CORRECT (arithmetic) | Every cell reproduces. One exception: the 7.2M-row total is 6.5–9.4 h, not "≈ 7–9" (fixed) |
| 13 | P-d at 7.2M tokens "needs P-c's 4-tap capture" | l. 82 | P-c captures 1M tokens. 7.2M × 40,960 B = 295 GB, above the 130 GB free (`deepinfer-ref/.../ENVIRONMENT.md:17`) | WRONG (inconsistent) | Fixed inline with two workable alternatives |
| 14 | Memory: 8.2 GB body optimizer plus 5.1 GB frozen bf16 "fits"; trainable lm_head 28.6 GB does not | l. 71 | Recomputed: 8.2 / 5.09 / 28.6 GB | CORRECT but incomplete | The teacher logits need the **target's** lm_head (separate from the draft's own head, `vllm/models.py:146`), another 2.54 GB, so frozen weights total 7.6 GB. It still fits. Fixed inline |
| 15 | Licence: Software includes weights and training code; noncommercial use granted; the $20M cap; the Competing Offering definition, the anti-evasion clause and the distribution duties | l. 121-125 | Every cited `LICENSE` line range checked (55-58, 61-65, 67-70, 73-75, 90-93, 97-106, 127-131, 133-141, 180-185, 192-197, 206-210, 254-262). `COMMERCIAL-LICENSE-AGREEMENT.md` §8 (l. 83-89) repeats the clause | CORRECT | See the exact quote and the missed clauses below |

The exact Competing Offering clause, `LICENSE:61-65`: "“Competing Offering” means a product or service whose principal functionality substantially substitutes for the principal functionality of Eagle 4, or that incorporates material Eagle 4 technology into a model, inference, training, acceleration, or serving framework that competes with Eagle 4. A separate commercial license is required for every Competing Offering." `LICENSE:73-75`: "“Eagle 4” means the Software and the technology embodied in it, including EAGLE-Harness and EAGLE Efficient LM Head technology, but excluding independently developed technology that merely interoperates with the Software."

## Recomputed estimates

All values are **estimated** unless marked measured. Formulas follow the doc: `6·P_body + 4·P_lm` per depth and `2·P_lm` once for the teacher. P_body = 0.5138B, P_lm = 1.2714B and the factors 32.44M, all measured from the header.

| Item | Doc | Recomputed at 25 TF/s | At 40 TF/s (reviewer central) |
|---|---|---|---|
| GFLOP/token, K = 1/3/7 | 10.7 / 27.1 / 59.7 | 10.71 / 27.05 / 59.72 | same |
| P-a (head only, 7.2M rows) | 3.6 GFLOP, ≈ 17 min/epoch | 3.58 GFLOP, 17.2 min | 10.7 min |
| P-c (1M tokens, K = 7) | ≈ 40 min + 14 min capture, ≈ 1 h | 39.8 + 14.2 min | 24.9 + 14.2 min |
| P-d (7.2M rows) | 3.8 GFLOP, ≈ 18 min | 3.77 GFLOP, 18.1 min (needs a 295 GB cache) | 11.3 min |
| 50M tokens from scratch: train / gen / capture / total | 33 / 20 / 12 / ≈ 65 h | 33.2 / 20.4 / 11.8 / 65.4 h | 20.7 / 20.4 / 11.8 / 52.9 h |
| 200M / 1B total | ≈ 260 / ≈ 1,300 h | 261.4 / 1,307 h | 211.7 / 1,058 h |
| Cache per token / 50M | 41 KB / 2.0 TB | 40,960 B / 2.05 TB | same |
| H100 at 300 TF/s, 50M | 2.8 h | 2.77 h | same |
| Generation rate | ≈ 680 tok/s | 5.4M ÷ 7,920 s = 682 tok/s | same |
| T-01 bytes per step | 65 vs 105 MB | 64.9 MB bf16 factors vs 104.9 MB int4 slice (+ ~3 MB scales) | same |

**Throughput question (the doc's open question 7).** Our trainer's depth-2 work per token:

- 6 × 0.42B for the MTP body: 0.37B layer (paper Table 1, p. 3) plus 52M `fc`.
- 4 × 0.21B for the frozen 40,960-row head: forward plus input-grad. The head is not trainable without `--train-head`, `train_mtp.py:228`.
- 2 × 1.27B for the full-vocab teacher: `train_mtp.py:292`, recomputed per depth through `head_loss` at :345.

That totals 11.83 GFLOP/token, or 10.14 if head and teacher run only on response rows (the `valid` mask at :337-339; 5.4M of 7.2M rows). At 4,000 tok/s this implies **40.6–47.3 TFLOP/s**, 57–67% of the 3090's 71 TFLOP/s dense bf16 peak. The doc's "about 44" sits inside that range, and its arithmetic holds. The EAGLE-4 workload per depth is even more GEMM-dominated (62% lm_head), so the rate should transfer. **Resolution:** use 40 TFLOP/s as the central case and 25 as the pessimistic bound. Times in the doc are about 1.6× high at the central case.

Three caveats:

1. "~30 min/epoch at 4k tok/s" is one rounded claim (7.2M ÷ 4000 = 30.0 min), not two measurements, and it is ours-doc, not a log.
2. The run was by upstream (`git log drafter/README.md`: mhenrichsen, 2026-08-19/20) at an unknown power limit, possibly not 250 W.
3. K = 7 activations force smaller micro-batches than our depth-2 run.

## Missed items

1. **The frozen lm_head is not EAGLE-4's recipe.** `eagle_harness/vllm/models.py:146` says "The draft trains its own lm_head". The doc's from-scratch plan freezes it, which is an adaptation and should be labelled as one. The faithful variant costs 77.5 GFLOP/token at K = 7 (+30%) and needs 28.6 GB of fp32 optimizer state. This also answers half of the doc's open question 3: the head was trained, and its initialisation is unknown. A 10 KB range read of one `lm_head` row from each checkpoint would settle the initialisation. I did not do it, because it goes beyond header-only metadata.
2. **The teacher-once assumption.** The FLOP model computes the teacher once per token. A trainer shaped like ours recomputes it per depth (`train_mtp.py:292`, :345): 75.0 instead of 59.7 GFLOP/token at K = 7 (+26%). It is cheap to avoid, but it should be stated as an assumption.
3. **Memory: the target lm_head for the teacher.** It adds 2.54 GB (fixed inline).
4. **The P-d cache exceeds disk** (fixed inline).
5. **Licence clauses not cited.**
   - §3.a: "Licensee may not develop, offer, sell, license, host, provide, or otherwise commercialize a Competing Offering under this License. Any such activity requires a separate written commercial license from Licensor before the activity begins, regardless of Aggregate Revenue." (`LICENSE:122-125`). "Develop" reaches past commercial use.
   - §1.f: "Eagle Files" are files that "contain, reproduce, modify, implement, port, or are derived from any portion of the Software" (`LICENSE:77-78`). §4.g keeps every obligation when code is moved into another framework (`LICENSE:216-221`). Exhibit E names "EAGLE Efficient LM Head" (`LICENSE:417-419`).
   - Together these make a self-written candidate head in HyperQwen plausibly an Eagle File even without commercial use. Not legal advice; the decision stays with the human (doc open question 6).
6. **Metric mismatch.** The paper's 98.2% is the recall of the full head's top-1 (p. 6). P-a measures the recall of the true next token and compares it with the 97.5% vocabulary coverage (`drafter/README.md:14-22`). Those are three different quantities, so they should not be read against one another without saying so.
7. **Response-row share.** The 7.2M rows include about 1.8M prompt tokens. Response-only factor training in P-a is about 14 min/epoch at 25 TF/s, not 17. This is small and was not edited.

## Corrections made in `training-recipe.md`

Each edit is marked inline with `[review: …]`:

1. l. 11: citation for "(3 aux layers, full vocabulary, rotary)" redirected to `new-model.md:49, :105, :27-28` and HF `rope_parameters`.
2. l. 71: memory budget now includes the target lm_head for the teacher (+2.54 GB; total ≈ 15.8 GB plus activations; still fits).
3. l. 72: `drafter/train_mtp.py:291` → `:292`.
4. l. 82 (P-d): the 295 GB cache conflicts with 130 GB of free disk and with P-c's 1M-token capture. Two alternatives given.
5. l. 90: "≈ 7–9" → "≈ 6.5–9.4".
6. l. 113: "3.59×" qualified as T = 0 (2.59× at T = 1).
7. Appended a "Reviewer additions" section (4 bullets): the throughput resolution, teacher recomputation, the frozen-lm_head deviation with the partial answer to open question 3, and the two licence clauses.

The analysis, tables and pilot list were not otherwise changed.
