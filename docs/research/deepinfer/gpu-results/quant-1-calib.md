# quant-1-calib: activation error per layer, and the Humming probe

| Field | Value |
|---|---|
| Date | 2026-10-21 10:16–10:29 SAST |
| Commit | tools in this result's commit: `bench/act_calib.py` (`--modes`), `bench/act_quant.py`, `bench/humming_probe.py` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88 |
| Model | `Qwen3.8-27B-W4A16-AutoRound-fast` (the served checkpoint), real activations from `bench/prompts_real.jsonl` |
| Environment note | Both tools need `/app/venv/bin` on `PATH` (act_calib's engine calls `ninja`) and `…/site-packages/nvidia/cu13/lib` on `LD_LIBRARY_PATH` (Humming's NVRTC needs `libnvrtc-builtins.so.13.0`). The first run without them failed; the serving launcher sets neither, so `--linear-backend humming` would hit the same NVRTC error in a stock container (inferred from the same failure, not booted). |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\quant-1-calib\{act_calib,humming}\` |

## Commands

```bash
/app/venv/bin/python /app/bench/act_calib.py /app/models/Qwen3.8-27B-W4A16-AutoRound-fast /out/act_calib_modes.json --modes all
/app/venv/bin/python /app/bench/humming_probe.py --out /out/humming_probe.json
```

## act_calib: relative output error added by int8 activations, 256 linear layers

| Layer type (n) | per-token (today) median | per-128 median, ratio, layers ≥ 3× | Hadamard-128 median, ratio, ≥ 3× | Hadamard-128 + g128 median, ratio, ≥ 3× |
|---|---|---|---|---|
| mlp.gate_up_proj (64) | 0.0449 | 0.0101, 4.4×, 52/64 | 0.0092, 5.0×, 55/64 | 0.0055, 8.8×, 63/64 |
| **mlp.down_proj (64)** | 0.0729 | 0.0142, **5.0×, 64/64** | 0.0116, 6.2×, 64/64 | 0.0056, 13.1×, 64/64 |
| **linear_attn.in_proj_qkvz (48)** | 0.0912 | 0.0161, **5.7×, 48/48** | 0.0119, 7.8×, 48/48 | 0.0042, 21.3×, 48/48 |
| linear_attn.out_proj (48) | 0.0426 | 0.0128, 3.5×, 40/48 | 0.0139, 3.4×, 33/48 | 0.0058, 8.5×, 48/48 |
| self_attn.qkv_proj (16) | 0.0759 | 0.0128, 5.7×, 16/16 | 0.0088, 8.0×, 16/16 | 0.0038, 21.5×, 16/16 |
| self_attn.o_proj (16) | 0.0285 | 0.0118, 2.7×, 7/16 | 0.0143, 2.5×, 3/16 | 0.0063, 5.5×, 16/16 |
| **All (256)** | | **227/256 ≥ 3×** | 219/256 | 255/256 |

The bf16 output-rounding floor of the measurement is 0.0023. Power-of-two per-128 scales come out at 134/256 ≥ 3×, worse than fp32 scales, as the appendix said.

## humming_probe: route A on sm86

| Question | Answer |
|---|---|
| `humming` importable | yes, `humming-kernels` 0.1.16 |
| Routing with `--linear-backend humming` | `HummingLinearKernel` (auto picks Marlin) |
| What vLLM resolves int8 g128 activations to | **`bf16/none/g0`**: `check_and_fallback_input_schema` "prefers wna16 for groupwise weight", so with the flag as shipped Humming silently runs W4A16 |
| int8 g128 with the fallback forced off | runs on sm86; correct (error vs its own fp32 reference 0.0017–0.0025; heavy-tailed rows 0.021 vs per-token's 0.10–0.13 against bf16 activations) |

Speed, geomean over the six linear shapes of t(Marlin) / t(Humming); above 1 means Humming is faster:

| Humming arm vs | M=1 | M=16 | M=64 | M=256 | M=2048 |
|---|---|---|---|---|---|
| int8 g128 vs Marlin W4A16 | 0.24× | 0.44× | 0.85× | 2.20× | 1.98× |
| **int8 g128 vs Marlin W4A8-int8 (today's INT8_ACT)** | **0.26×** | **0.41×** | **0.40×** | 1.19× | 1.19× |
| int8 per-token vs Marlin W4A8-int8 | 0.37× | 0.39× | 0.37× | 1.01× | 1.29× |
| W4A16 vs Marlin W4A16 | 0.18× | 0.37× | 0.60× | 1.05× | 1.07× |

## Verdict

INFO, in two parts.

**Quality: the gate passes.** Per-128 scales cut the error of `down_proj` (64/64 layers) and of GDN's `in_proj_qkvz` (48/48) by ≥ 3×, at median ratios of 5.0× and 5.7×. Hadamard-128 with per-128 groups would cut every layer type 5–21×.

**Route A speed: Humming loses at decode sizes.** At M ≤ 64, the regime of every decode step including mode A's 64-stream batch, Humming int8 g128 runs at 0.26–0.41× the speed of today's Marlin W4A8-int8, so it is 2.5–4× slower. It wins only at prefill sizes (M ≥ 256, +19%). On top of that, the flag as shipped falls back to W4A16, and the container lacks the NVRTC library path.

So route A cannot deliver quant-1's mode-A gain. At most it is a prefill-only option, and it would need a patch to stop the fallback plus a library-path fix. That opens quant-4's gate ("route A blocked or slower than Marlin by > 5%"): quant-4 is the per-128 / Hadamard quality gain on the unchanged Marlin kernel.
