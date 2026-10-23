# attn-5: Hadamard-rotated int8 per-token-head KV (mode D)

| Field | Value |
|---|---|
| Date | 2026-10-22 23:30 – 2026-10-23 02:07 SAST |
| Commit | `55ec644` + `experiments/patches/attn5-kv-hadamard.patch`, `bench/test_kv_hadamard.py`, `bench/kv_ref_kl.py`, `cpu-checks/attn5_hadamard_kv.py` |
| Image | `hq:exp-attn5` (`sha256:4b093108c07a…`, `Dockerfile.patch` on `ghcr.io/syv-ai/hyperqwen:latest` `sha256:0cf71b42d301…`, main `ce9615c`) |
| GPU | RTX 3090, 250 W enforced, driver 610.88; busy median SM clock 1035–1065 MHz |
| Knob (default off) | `VLLM_KV_HADAMARD=1`: K and V rotated by a 256-point Hadamard at cache write, q rotated before attention, output rotated back; 16 full-attention layers plus the MTP layer |
| Mode | D (`SPEC=mtp CTX=long`, int8 per-token-head KV on pthfa) |
| Spill | Speed boots: PCIe job-window median 43–46 MB/s, no second above 500 MB/s. The bf16 reference boot (`ref_B`) moved logits to the host (median 213 MB/s); it carries no timings. |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\attn-5\` (`test_hadamard`, `ref_B`, `D_base`, `D_H`, `D_{base,H}_q`, `D_{base,H}_s{1,2}`) |

## What the appendix expected

The appendix's CPU check (`attn_kvq_error.out`, isotropic activations) predicted int8 per-token-head output error 0.069 → 0.027 with rotation, and that int4 per-token-head would become usable for D. Two corrections from preparing the experiment:
- vLLM's int4 per-token-head mode already applies a randomized Hadamard (`int4_per_token_head.py`, `single_rht`), so the appendix's "int4 is unrotated" is wrong and only int8 was patched.
- On the real checkpoint's q/k/v projections and norm gains (`cpu-checks/attn5_hadamard_kv.py`), k_norm leaves K nearly flat. Rotation moves int8 output rel-L2 only 0.016 → 0.015, and the rotated arm wins in 7 of 16 layers.

## Correctness

`bench/test_kv_hadamard.py`: 18/18 OK. The cache holds rotated K, a frame mismatch is detected (1.000 ≥ 10 × 0.0074), and outlier channels at L=4096 drop from 0.164 to 0.039 rel-L2. The first GPU run crashed on a test bug (a fresh cache per length, while vLLM keeps its scale views bound to the first cache); fixed by sharing one 131k cache.

## Quality against a bf16-KV reference (`bench/kv_ref_kl.py`, 10 prompts, 64,218 scored tokens; reference = mode B, bf16 KV)

| | Stock int8 KV | Hadamard int8 KV |
|---|---|---|
| top-k KL | 1.632e-2 | 1.509e-2 (−7.5%) |
| top-1 agreement | 98.34% | 98.50% |
| perplexity ratio vs bf16 | ×0.9941 | ×1.0123 |
| KL by depth quarter | 1.00 / 1.19 / 2.12 / 2.22 e-2 | 0.92 / 1.42 / 1.75 / 1.95 e-2 |

Mixed: the logit distance shrinks a little at depth, but perplexity against the bf16 arm worsens by 1.8%. The decode KL probe (self-consistency, not a KV-format check) stays within the gate: 2.0e-4 at 4k, 1.15e-4 at 16k, against stock's 1.2e-4 / 1.4e-4.

| Check (separate numerics boots) | Stock | Hadamard |
|---|---|---|
| perplexity, all | 8.4018 | 8.4019 |
| GSM8K greedy / seed 1, n=200 each | 0.965 / 0.940 | 0.970 / 0.935 |
| needle at 100k, depths 0.1 / 0.5 / 0.9 | 3/3 | 3/3 |

## Speed (`run_benchmarks.sh single`, second run of each boot, two boots per arm, spill-free)

| | Stock | Hadamard | Change |
|---|---|---|---|
| C1 greedy tok/s | 98.3 / 95.4 | 96.1 / 91.3 | −3.2% (median) |
| C4 greedy tok/s | 270.1 / 261.6 | 252.6 / 238.8 | −7.0% |
| tokens per step, greedy | 2.64 / 2.64 | 2.57 / 2.57 | −2.7% |
| KV pool | 207,286 | 205,778 | −0.7% |

The MTP drafter reads the same rotated cache, and its acceptance drops 2.7% in both boots; the FWHTs add the rest.

## Verdict: LOSS

There is no quality gain on the real model, because k_norm already flattens K, and decode is 3–7% slower. The int4 half of the item was already in vLLM. For reference, vLLM's already-rotated int4 per-token-head KV (`single-user/alternative.sh`, `attn-5\int4_alt`) against the same bf16 reference: top-k KL 9.3e-2 (6× int8), perplexity ratio ×1.162, top-1 agreement 94.75%, needles 3/3. Moving D to int4 KV for a 2× pool would cost ~16% perplexity, so that half of the appendix's idea is closed too.
