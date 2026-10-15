Conclusion A — the model holds

| k (`DFLASH_TOKENS`) | launcher `RESIDENT` (printed) | `KV_MEM/((k+2)×104988089)` | `Running:` at `MAX_SEQS=8` | `Running:` uncapped (`MAX_SEQS=16`) | pool per running request | seats the pool holds |
|---|---|---|---|---|---|---|
| 7 | 5 | 5.91 | 7, Waiting 1 (99.2%) | **7**, Waiting 9 (99.2%) | 14.2% | 7.1 |
| 3 | not printed | 10.64 | 8, Waiting 0 (59.6%), capped | **13**, Waiting 3 (96.8%) | 7.45% | 13.4 |
| 1 | not printed | 17.73 | 8, Waiting 0 (35.0%), capped | 16, Waiting 0 (69.9%), capped | 4.37% | 22.9 |

# Batch 3, 2026-10-14: what caps residency (G7)

RTX 3090 at 250 W (`power.limit 250.00 W`), image `hyperqwen:exp-f1a033e` reused (sha256:c06b97c7…; no code
change since, `2026-10-14-batch3-preflight/image-reuse.txt`). Int8 `CTX=fast` base, `KV_MEM=5583457484`,
`MAX_LEN=65536`, `PREFIX_CACHE=1`, the 3090 by UUID. Tools: `2026-10-14-batch3-tools/`. The `Running:` probe is
`running_probe.py`.

## Q1: residency scales with k (`2026-10-14-Q1-k{7,3,1}/report.md`)

- **The primary metric.** On the brief's `MAX_SEQS=8` probe, Running went 7 → 8 → 8. The literal ±1 band is
  met, but only because the cap binds: k=3 and k=1 run all 8 with `Waiting: 0` and 40-65% of the pool free. With
  the cap lifted (`2026-10-14-Q1x-uncapped`, `MAX_SEQS=16`), Running goes **7 → 13** from k=7 to k=3, +6. That
  meets the Conclusion A threshold. At k=1 it reaches the 16 cap with 30% of the pool still free.
- **Preemptions fall to 0** (ladder N=8: 2 → 0 → 0; N=16: 29 → 0 → 0). **The N=8 aggregate rises:** 178.5\*
  (tail) → 350.1 → 299.4 tok/s. At N=16 every arm is a tail row under `MAX_SEQS=8`: 213.8\* / 333.7\* / 291.7\*.
- **The formula's shape holds and its constant is wrong.** The per-request share scales with (k+2): 14.2 : 7.4 :
  4.4 against 9 : 5 : 3. A request costs 81-88 MB per (k+2) unit with its ~1.6k tokens of context, not 105 MB. So
  the launcher's print **under**-predicts seats by 16-23%. It says "about 5" at k=7 and the pool holds 7. This
  contradicts the batch doc's Conclusion B text, which expected an over-prediction.
- **Batch 2's contradiction is explained.** Its "k=3" arm was dynamic K on `num_speculative_tokens=7`. Its boot
  shows `num_spec_tokens=7`, an attention block of 864 (as k=7), and 14.05% of the pool per request (as k=7). The
  state pages follow the configured maximum k, so dynamic K buys step time and no seats.
- **Seats cost single-stream speed.** C1 T=0 reads 112.1 / 94.1 / 64.7 tok/s at k=7/3/1 (−16% at k=3), and C8 T=0
  reads 371.7 / 439.3 / 383.0.
  reads 371.7 / 439.3 / 383.0.

## Q2: bf16 recurrent state, B loses (`2026-10-14-Q2-{fp16,bf16}/report.md`)

- **The fused kernel appears on B only.** fp16 (qA1, qA2) prints `Falling back to the Triton GDN decode path: …`
  and `GDN decode kernel: triton`. bf16 (qB) prints `GDN decode kernel: cuda` and no fallback line. The args line
  confirms `'mamba_ssm_cache_dtype': 'bfloat16'`.
- **It buys nothing.** C1 T=0 reads 114.2 / 114.4 / 114.3 (A-B-A), +0.1%. N=1 `ms/pass` reads 29.5 / 30.2 /
  29.8, +1.9% (slower). The rule needed +3% or −3%. The N=8 ladder rows are tail rows on every boot (1
  preemption, 6 running).
- **Quality and footprint are identical.** GSM8K reads 192 vs 193/200, and perplexity is within 0.05%. Every boot
  has a 116,440-token pool, block 864 and `Running: 7 reqs, Waiting: 1 reqs, GPU KV cache usage: 99.2%`.
- **Perplexity spilled on both arms**, with the guard in watch-only mode for that step: Shared Usage went from
  92.2 to 1,002.2 MiB on fp16 and from 84.2 to 938.2 MiB on bf16. Gotcha 60 records 410 MiB for int8. All timing
  and GSM8K ran before it, on the floor.
## What could not be established

- The k=1 seat count above 16. `MAX_SEQS=24` spilled at boot: Shared Usage went 90.2 → 122.2 MiB at the
  profiling step. The 22.9 figure is the pool share arithmetic, not a held count.

## Seen that the batch did not ask for

- **`MAX_SEQS` above 8 spills at boot.** At `MAX_SEQS=16`, Shared Usage went from 86.2 to 102.2 MiB at the
  memory-profiling step on all three boots, and at `MAX_SEQS=24` from 90.2 to 122.2 MiB. The spill guard
  missed the 16 case. Its floor is taken from loaded samples, and these rose before dedicated memory passed
  20,000 MiB, so the raised level became the floor.
- **The engine's GPU KV cache size in tokens rises as k falls** at fixed bytes: 116,440 / 131,724 / 141,708. The
  attention block goes 864 / 816 / 800 at the same time.
