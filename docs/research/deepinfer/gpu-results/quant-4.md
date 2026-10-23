# quant-4: block-Hadamard-128 rotation of linear inputs (go/no-go)

| Field | Value |
|---|---|
| Date | 2026-10-23 17:05–17:55 SAST |
| Commit | `bench/act_calib.py --dump-inputs`, `experiments/quant4_h128down.py --report-only` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`sha256:0cf71b42d301…`, main `ce9615c`), stock |
| Data | Real `down_proj` inputs from the served `-fast` checkpoint: 8,192 rows per layer for all 64 layers, from 96 prompts of `drafter/data/prompts.jsonl` (GPU dump, 18 GB). The report scores 1,024 rows per layer on CPU. |
| Gate (appendix precondition met: quant-1 is a LOSS) | median over the 64 `down_proj` layers of a_pt / tot ≥ 1.5, where a_pt is today's per-token int8 activation error and tot is the rotated arm's total error, including the second int4 rounding that the requant from the int4 checkpoint adds (no bf16 source on the box) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\quant-4\{dump,report}\` |

## Result

The report completed layers 0–47 and then stopped at layer 48: the checkpoint has no `weight_scale` tensor under that name for layers 48+ (a different storage layout for the last 16 layers). The 48 completed layers decide the gate on their own:

| | Value |
|---|---|
| layers scored | 48 of 64 |
| median a_pt / tot | **0.89** |
| range | 0.55 – 2.53 |
| bound on the 64-layer median, even if all 16 missing layers scored +∞ | **0.99** |
| gate | 1.5 |

A ratio below 1 means the rotated, requantized arm has *more* total error than today's unrotated per-token int8. The rotation does cut the activation error itself (a_h ≈ 0.012–0.013 against a_pt 0.075–0.093 on the layers shown in the log), but the extra int4 rounding from requantizing an already-int4 checkpoint (w ≈ 0.08–0.09) swamps it.

## Verdict: NO-GO (closed)

The appendix's CPU estimate (1.3–2.2×) assumed the activation-error term dominates; on real `down_proj` inputs the requant's weight error dominates. A requant from the 54 GB bf16 source would remove the second rounding and could change this; that source is not on the box (not downloaded: ~54 GB). Building the `-h128down` model and serving it is skipped.
