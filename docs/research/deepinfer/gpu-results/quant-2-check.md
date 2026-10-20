# quant-2-check: negative-scale fold on the real AutoRound checkpoint

| Field | Value |
|---|---|
| Date | 2026-10-20 |
| Commit | `3ceb21f` (script unchanged since `b768b26`) |
| Image | none: CPU only, repo venv in WSL (torch 2.13 CPU path) |
| GPU / power / clocks | not used |
| Checkpoints | `G:\models\Qwen3.8-27B-W4A16-AutoRound` and `…-AutoRound-fast` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\quant-2-check\` (per-tensor tables) |

## Command

```bash
cd docs/research/deepinfer/cpu-checks
python3 -E -P quant_real_ckpt_check.py /mnt/g/models/Qwen3.8-27B-W4A16-AutoRound
python3 -E -P quant_real_ckpt_check.py /mnt/g/models/Qwen3.8-27B-W4A16-AutoRound-fast
```

Two shards (`model-00006/7`, `model_extra_tensors`) are mode 0600 root in WSL's metadata, so the check ran as WSL root (read-only). 187–199 s each.

## Numbers

| Checkpoint | int4 groups | Negative-scale groups | Negative groups with a −8 code | Clamped elements per negative group | Fold error energy / weight energy |
|---|---|---|---|---|---|
| AutoRound | 187,965,440 | 50.4% | **100.0%** | 1.45–1.73 per tensor (mean ≈ 1.57) | **0.0688%** |
| AutoRound-fast | 190,873,600 | 49.7% | **100.0%** | same body; the GPTQ MTP layers have 0% negative scales | 0.0674% |

Every tensor in the body has 49.6–50.6% negative groups, and every one of those groups uses code −8 at least once. The int16 requant error of `marlin_act_int8_process_scales` (last column of the raw table) is 0.04–0.22% per tensor.

## Verdict

INFO, and the gate for quant-2 **passes**: negative groups do use −8, in every case, not "extremely rarely" (`patches/marlin-int8-negative-scales.patch:36-38`). The lane's reconstruction in `quant_b_signfold.py` is confirmed on the real checkpoint.

Size: the fold adds 0.069% of the weight energy as error. Against the 1–2% weight-error energy that int4 itself adds (the script's reference), that is +3.4 to +6.9% more weight error on every `INT8_ACT` layer, in line with the appendix's +5–9% estimate. It applies only where `INT8_ACT=int8` routes a layer through the W4A8 kernel (mode A's MLP by default); W4A16 Marlin handles signed scales natively and is not affected.

Contradictions with the appendix: none. The MTP head of the fast variant is GPTQ-quantized by HyperQwen's own `gptq_utils.py` with amax/7 positive scales, so it has no negative groups at all; that is the quant-3 observation (code −8 unused) seen from the other side.
