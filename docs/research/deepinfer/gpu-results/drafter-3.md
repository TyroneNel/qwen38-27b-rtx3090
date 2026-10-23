# drafter-3: TV-loss repair of the W4A16 DFlash2

| Field | Value |
|---|---|
| Date | 2026-10-23 20:12–22:02 SAST |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`sha256:0cf71b42d301…`, main `ce9615c`), stock |
| Tool | `drafter/train_dflash2_tv.py`: int4 linears frozen; the 45 bf16 tensors (convs, selector, norms) trained with TV against the bf16 drafter's selector distribution, exp(−k/4) position weights, cosine schedule; best checkpoint by validation tau against the teacher |
| Training data | `capture_dflash2.py --modes sample --skip 12 --n 64` on 6 domains (code, danish, chat, think, gsm8k_think, gsm8k_nothink): 384 sequences, 247,785 rows. It does not overlap drafter-2's eval capture (first 12 prompts per domain); real_*/labd are held out entirely |
| Run | 1,290 steps at 2.5 s/step (0.93 h). The plan was 3.5 h; it was cut to 1 h before training because the teacher is the bf16 drafter, which drafter-2 measured only 0.6% ahead of W4A16 on the target. The appendix names bf16 parity as the ceiling |
| Output | `G:\models\Qwen3.8-27B-DFlash2-W4A16-tv` (W4A16 checkpoint, bf16 tensors replaced; loads unchanged) |
| Measurement | drafter-2's offline gate on its capture, paired per sequence against the shipped W4A16 |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\drafter-3\{tvcap,probe,train,eval}\` |

## Training

Validation tau against the teacher went from 6.888 at the start to a best of 6.924 at step 1,200 (+0.52%); it sat at 6.86–6.90 for the first 500 steps. Per-position TV against the teacher at the end: 0.025 (k=1) to 0.080 (k=7).

## Results against the target (drafter-2 gate)

| Arm | tau_chain greedy | tau_chain sampling | paired ratio vs W4A16, sampling (CI95) | greedy (CI95) |
|---|---|---|---|---|
| W4A16 (shipped) | 3.618 | 3.394 | 1 | 1 |
| tv | 3.618 | 3.400 | **1.0017 [1.0013, 1.0021]** | 1.0001 [0.9969, 1.0033] |

Per domain at sampling: 1.0006 (code) to 1.0037 (danish), every domain ≥ 1 and 8 of 9 significant, including the held-out real_think (1.0015), real_nothink (1.0010) and labd (1.0013). Greedy is unchanged within ±1.4% in every domain.

## Verdict: LOSS (against the bar)

The repair works in the right direction and generalises to held-out domains, but it recovers +0.17%, about 30% of the 0.6% gap drafter-2 measured. The bar was ≥ +3% tok/s at default sampling with greedy within ±2%. The appendix's "+3% (ceiling +6%)" rested on the 3.2 vs 3.4 gap from single live runs, which doesn't exist on paired data. No serving A/B: the gate needs an offline ratio ≥ 1.02.
