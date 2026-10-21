# evals-1: decode-path vs prefill-path logprob KL, per mode

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | `bench/decode_kl_probe.py`, added in this result's commit |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88 |
| `.env` knobs | per mode as baseline.md (A at `GPU_UTIL=0.91`); probe boots also had `VLLM_GPU_SYNC_CHECK=warn` (logging only) |
| Context | 8 prompts at ~4k tokens × 512 greedy tokens (two runs), and 4 prompts at ~16k × 512 |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\probes\<mode>\kl_*.{txt,json}`, `…\evals-1\{B_int8all,Dp_fa2fp8}\` |

## Method

The probe works through `/tokenize` with the chat template (thinking off) and `/v1/completions` on the token ids:

1. Generate greedily with `return_tokens_as_token_ids`, taking the decode-path logprob of each chosen token.
2. Resend prompt + output with `prompt_logprobs=0`, `max_tokens=1`. In vLLM 0.31 a `prompt_logprobs` request skips reading the prefix cache, so this is a fresh prefill.
3. Compare the two logprobs per generated position: k3 KL `mean(exp(d) − 1 − d)`, mean and max |Δ|, cosine, and the fraction of positions where prefill would not have picked the decoded token (`rank≠1`).

The negative control (prefill logprobs shifted by one position) must raise k3 at least 2×; it raised it 26–720× in every mode.

```bash
venv/bin/python bench/decode_kl_probe.py --ctx 4000 --negative-control --label B_4k_r1 --out /out/kl_4k_r1.json
venv/bin/python bench/decode_kl_probe.py --ctx 16000 --prompts 4 --label B_16k --out /out/kl_16k.json
```

## Numbers

| Mode (decode path) | k3 @4k | k3 @16k | mean \|Δ\| @4k | max \|Δ\| @4k | rank≠1 @4k | × B |
|---|---|---|---|---|---|---|
| B (DFlash2, bf16 KV, FA) | 1.30e-4 | 7.5e-5 | 0.0057 | 0.14 | 1.7% | 1.0 |
| C (B, k=15) | 1.53e-4 | 8.0e-5 | 0.0060 | 0.32 | 1.6% | 1.2 |
| D (MTP, int8 KV on pthfa) | 1.38e-4 | 9.9e-5 | 0.0062 | 0.12 | 1.5% | 1.1 |
| D′ (DFlash2, int8 KV on pthfa) | 1.42e-4 | 1.05e-4 | 0.0063 | 0.13 | 1.5% | 1.1 |
| D′ with `PTH_FA=0 FA2_FP8KV=1` (fp8 KV on FA2) | 1.62e-4 | 1.33e-4 | 0.0069 | 0.15 | 1.3% | 1.25 |
| E (KVarN k4v2) | **1.25e-3** | 6.0e-4 | 0.017 | 1.14 | 2.4% | **9.6** |
| A (batch: fp8 KV, int8 MLP activations) | **4.34e-3** | 3.4e-3 | 0.042 | 1.14 | 4.2% | **33** |
| B with `INT8_ACT=int8 INT8_LAYERS=.` (all-linear int8) | **1.14e-2** | 7.0e-3 | 0.049 | 2.58 | 4.2% | **88** |

**Repeatability.** The two 4k runs in each boot were bit-identical in every mode (spread 0). Across boots, run the probe first after warmup (evals-7); mid-session drift is possible.

**Memory.** The 16k `prompt_logprobs` pass spilled VRAM to system RAM in the tight-pool modes. WDDM non-local usage grew by 172 MiB (B), 834 MiB (C) and 640 MiB (E). This slows the probe but does not change its numbers. For speed runs, never mix this probe into the measured window.

## Reading

- **The probe separates decode-only lossy paths that GSM8K cannot.** All-linear int8 activations (+4.0% perplexity, and no GSM8K difference in evals-3) read 88× the baseline here. KVarN reads ~10×, A's fp8-KV plus int8-MLP batch profile 33×.
- **int8 KV on pthfa is close to bf16 KV:** D and D′ sit at 1.1× B. fp8 KV on FA2 is slightly worse (1.25×), consistent with the attention lane's int8-beats-e4m3 finding (attn_kvq_error).
- **Gate for the lossy items** (attn-1.3, attn-3, quant-7, quant-5, quant-1): measure k3 at 4k on the item's mode, two runs. A change that keeps k3 within 2× of its mode's stock value here passes; above that, it needs the quality battery to argue it in. The thresholds by mode: B, C, D, D′ 3e-4; E 2.5e-3; A 8.7e-3.

## Verdict

INFO. The decode KL probe works, is deterministic within a boot, and has per-mode baselines plus a calibrated 2× gate. It sees all-linear int8 activations (88×) and KVarN (~10×) clearly where GSM8K does not.
