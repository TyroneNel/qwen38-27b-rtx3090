# Batch 3 Q2: bf16 recurrent state (2026-10-14, RTX 3090 at 250 W)

The same report is in `2026-10-14-Q2-fp16` (arm A, boots qA1 and qA2) and `2026-10-14-Q2-bf16` (arm B, boot qB).

**Verdict: B loses.** The fused GDN CUDA kernel engages on bf16 and not on fp16, as predicted, but it buys no step
time. C1 T=0 moves +0.1% and N=1 `ms/pass` moves +1.9% (slower), against required gains of +3% and −3%. Quality is
unchanged.

## Arms

Both arms use the Q1 config at k=7: int8 `CTX=fast` base, `KV_MEM=5583457484`, `MAX_LEN=65536`, `MAX_SEQS=8`,
`PREFIX_CACHE=1`, `VLLM_LOG_STATS_INTERVAL=2`, the 3090 by UUID, image `hyperqwen:exp-f1a033e`.
- **A**: the shipped `--mamba-ssm-cache-dtype float16`, which the launcher passes.
- **B**: the same, plus `--mamba-ssm-cache-dtype bfloat16` at the end of `EXTRA_ARGS`.

The engine's own non-default-args line shows which value took effect: `'mamba_ssm_cache_dtype': 'float16'` on qA1
and qA2, `'mamba_ssm_cache_dtype': 'bfloat16'` on qB. The boots were interleaved qA1 → qB → qA2.

## The GDN-path line (`raw-bootlines-<label>.log`)

| boot | line |
|---|---|
| qA1 (fp16) | `Falling back to the Triton GDN decode path: the fused CUDA kernel requires a BF16 GDN model with K=V=128, SiLU or sigmoid gating, non-interleaved GQA layout, BF16 convolution cache, BF16 or FP32 recurrent state, and a GPU with compute capability 8.0+` then `GDN decode kernel: triton` |
| qB (bf16) | no fallback line; `GDN decode kernel: cuda` |
| qA2 (fp16) | the same fallback line as qA1, then `GDN decode kernel: triton` |

The whitelist entry is about this flag: bf16 state alone is enough to get the fused kernel on this model config.

## Speed (metric: C1 T=0 from `bench/run_benchmarks.sh single`, second run; `ms/pass` from `conc_ladder.py --n 1,8`)

| metric | qA1 (fp16) | qB (bf16) | qA2 (fp16) | B vs mean(A) |
|---|---|---|---|---|
| C1 T=0 decode tok/s, run 2 | 114.2 | 114.4 | 114.3 | +0.1% |
| C1 T=0 decode tok/s, run 1 | 114.0 | 114.5 | 114.2 | +0.4% |
| C1 T=0 tok/step, run 2 | 3.34 | 3.35 | 3.34 | flat |
| ladder N=1 `ms/pass` | 29.5 | 30.2 | 29.8 | +1.9% (slower) |
| ladder N=1 tok/step | 3.40 | 3.51 | 3.43 | +2.8% |
| ladder N=8 `ms/pass` | 43.8\* | 45.2\* | 47.1\* | tail rows |
| ladder N=8 decode agg | 196.3\* | 165.1\* | 169.2\* | tail rows |
| C8 T=0 decode tok/s, run 2 | 376.6 | 358.7 | 378.6 | −5.0% |

`*` marks the ladder's tail flag. Every N=8 row had 1 preemption with 6 running at 99.7% of the pool. That is the
Q1 k=7 seat limit, so these rows are not steady state and are not used for the verdict. Batch 2 measured the
cross-boot C1 floor at up to 3.2%, and the A boots here agree to 0.1%. B sits inside that: no win. The C8 T=0
cohort row reads 5.0% lower on B. It is a single boot, the C8 row runs with the pool full, and it is not part of
the rule. Recorded, not concluded.

## Footprint and the `Running:` probe

| | qA1 | qB | qA2 |
|---|---|---|---|
| GPU KV cache size | 116,440 tokens | 116,440 tokens | 116,440 tokens |
| attention block / mamba padding | 864 / 1.09% | 864 / 1.09% | 864 / 1.09% |
| free VRAM after boot | 1,365 MiB | 1,385 MiB | 1,365 MiB |
| steady `Running:` pair, waves 1/2/3 | `Running: 7 reqs, Waiting: 1 reqs, GPU KV cache usage: 99.2%` ×3 | the same ×3 | the same ×3 |

The footprint is identical: bf16 is the same two bytes as fp16, as the batch doc predicted. Residency does not move.

## Quality (`bench/quality_battery.py`, the same commands as batch 2 S1)

| | qA1 (fp16) | qB (bf16) | delta |
|---|---|---|---|
| GSM8K n=200, thinking off, greedy | acc=0.960 (192) | acc=0.965 (193) | +1 question |
| perplexity en | 10.7605 | 10.7574 | −0.03% |
| perplexity da | 10.9105 | 10.911 | +0.005% |
| perplexity code | 3.6008 | 3.599 | −0.05% |

Both are within one GSM8K question and 1% perplexity. The fp16 arm's perplexity matches batch 2 S1's int8 arm to
four decimals (10.7605 / 10.9105 / 3.6008).

### The spill counter during perplexity

The spill guard ran in watch-only mode for the perplexity step only, the last step on each boot (`cmd.txt`).
**The floor did not hold on either arm.**

| boot | loaded floor | Shared Usage during perplexity | step window (UTC) |
|---|---|---|---|
| qA1 | 92.2 MiB | 632.2 then **1,002.2 MiB** | 23:21:09-23:21:48 |
| qB | 84.2 MiB | 568.2 then **938.2 MiB** | 23:51:23-23:52:01 |

Both spills started inside the perplexity step and lasted until the boot went down (`mem.txt`, `SPILL-qA1`,
`SPILL-qB`). They are larger than gotcha 60's 410 MiB for int8. GSM8K ran before perplexity on each boot and stayed
on the floor. All timing ran before both, on the floor. qA2 ran no quality and never left its 84.2-92.2 MiB floor.
Perplexity numbers do not depend on speed, so the spill leaves them valid.

## Verified vs assumed

Verified: the args line, the GDN lines, every timing, probe and quality number, and the counter trace, all quoted
from the raw files in the two Q2 directories. Not verified: why the fused kernel gives no step-time gain at C1.
No kernel profile was taken.
