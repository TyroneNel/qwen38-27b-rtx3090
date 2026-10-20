# cpu-checks: re-run of every script in `cpu-checks/`

| Field | Value |
|---|---|
| Date | 2026-10-20 |
| Commit | `88549e7` |
| Interpreter | repo venv in WSL: Python 3.12, torch 2.13.0 (CPU), numpy from the venv; `CUDA_VISIBLE_DEVICES=` |
| GPU | not used |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\cpu-checks\` (`<script>.out`, `.err`, `.diff`, `summary.txt`) |

## Commands

From `docs/research/deepinfer/cpu-checks/`, each script's output diffed against its committed `.out`:

```bash
python -E -P <script>.py > <out>/<script>.out     # every script except spec_calib.py
python -E spec_calib.py > <out>/spec_calib.out    # imports its sibling, so not -P (README table)
```

Runner: `G:\dev\hq-results\deepinfer-20261020\tools\cpu_checks.sh` and `q2check.sh`.

## Results

| Script | Exit | Seconds | Against the committed `.out` |
|---|---|---|---|
| `runtime_topk_sim.py` | 0 | 1 | identical |
| `attn_roofline.py` | 0 | 0 | identical |
| `attn_kvq_error.py` | 0 | 11 | identical |
| `attn_mma_precision.py` | 0 | 3 | identical |
| `spec_varlen_sim.py` | 0 | 49 | identical |
| `spec_calib.py` (`-E`) | 0 | 1 | identical |
| `quant_a_act_quant.py` | 0 | 16 | identical |
| `quant_a2_gauss.py` | 0 | 15 | identical |
| `quant_b_signfold.py` | 0 | 15 | identical |
| `quant_c_logits_bf16.py` | 0 | 18 | identical |
| `quant_d_gptq.py` | 0 | 44 | **12 lines differ in the 4th–5th significant digit** (e.g. `--mse-clip` 6.7677e-03 → 6.7623e-03, −27.1% → −27.2%) |
| `drafter_checks.py` | 0 | 34 | identical |
| `quant_mk_fake_ckpt.py` + `quant_real_ckpt_check.py` on the fake checkpoint | 0, 0 | – | self-test runs |

`quant_d_gptq.py` runs float32 GPTQ (Cholesky, column updates) whose rounding depends on the BLAS build and thread count; this box's torch 2.13 differs from the research box's. Every ranking and every percentage agrees to within 0.1 point, so no conclusion in quant-heads.md moves.

The first pass ran `spec_calib.py` with `-P` too and it failed on its sibling import (`ModuleNotFoundError: spec_varlen_sim`); that is the documented `-E`-only case, and the re-run with `-E` matches byte for byte.

## Verdict

INFO: the CPU evidence reproduces on this box. 11 of 12 scripts are byte-identical; `quant_d_gptq.py` differs only in float noise.
