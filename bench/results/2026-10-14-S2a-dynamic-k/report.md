# S2a: dynamic K on B's base (`num_speculative_tokens_per_batch_size` [[1,3,7],[4,512,3]])

RTX 3090, 250 W (`power.limit 250.00 W` in every `raw-smi-*.log`), container boot, image `hyperqwen:exp-f1a033e`.

## Arms and what the engine got

Base for both arms is S1's B, with the pool pre-cut by 256 MiB (`knob-changes.txt`). That is `SPEC=dflash2
PREFIX_CACHE=1 CTX=fast DFLASH_TOKENS=7 KV_MEM=5315022028 MAX_LEN=65536 MAX_SEQS=8` plus `--attention-backend
TRITON_ATTN --kv-cache-dtype int8_per_token_head`. B (`.env.B`, label Bdk1) adds a second `--speculative-config`.
The engine's `non-default args` in `raw-boot-Bdk1.log` show the later one won:

```
'speculative_config': {'method': 'dflash', 'model': '/app/models/Qwen3.8-27B-DFlash2-W4A16', 'num_speculative_tokens': 7, 'draft_sample_method': 'probabilistic', 'num_speculative_tokens_per_batch_size': [[1, 3, 7], [4, 512, 3]]}
```

Boot order: A1, Bdk1, A2, Bcg1 (S2b's B), A3. A1, A2 and A3 are shared with S2b. Every boot reported `GPU KV
cache size: 110,953 tokens`. Graphs: A `took 0.13 GiB (driver free-memory delta 0.32 GiB)`, Bdk1 `took 0.13 GiB
(driver free-memory delta 0.37 GiB)`.

## Results

C1 T=0 = `bench/run_benchmarks.sh single`, row `cohort C1 real prompts T=0`, run 2. Ladder = `bench/conc_ladder.py
--n 4,8`. `*` = the ladder's tail-row flag.

| boot | C1 T=0 | C1 tok/step | N=4 decode agg | N=4 ms/pass | N=4 tok/step | N=4 preempt | N=8 decode agg / preempt / run |
|---|---|---|---|---|---|---|---|
| A1 | 114.9 | 3.34 | 235.4 | 56.6 | 3.40 | 0 | 191.2* / 0 / 5 |
| **Bdk1** | **115.6** | 3.34 | **260.5** | **43.8** | **2.90** | 0 | 193.6* / 0 / 5 |
| A2 | 115.3 | 3.34 | 241.6 | 56.3 | 3.47 | 0 | 180.2* / 0 / 5 |
| A3 | 116.3 | 3.34 | 247.1 | 56.0 | 3.52 | 0 | 188.1* / 0 / 5 |

| metric | A (A1, A2 flanking) | B | delta | A3 included |
|---|---|---|---|---|
| N=4 decode aggregate | 238.5 | 260.5 | **+9.2%** | +7.9% vs mean(A1..A3) 241.4 |
| C1 T=0 | 115.1 | 115.6 | **+0.4%** | +0.1% vs 115.5 |
| N=4 ms/pass | 56.5 | 43.8 | -22.4% | |
| N=4 tok/step | 3.44 | 2.90 | -15.6% | |

**Verdict: pass.** The rule is N=4 aggregate ≥5% above A with C1 within 1%. B is +9.2% against its flanking A
boots and above all three A boots (the A boots spread 235.4-247.1, 5.0%). C1 is +0.4%. Batch 1's lead was +8.0%,
ms/pass 57 → 43 and tok/step 3.50 → 2.95. All three reproduce here: 56.5 → 43.8 ms/pass and 3.44 → 2.90
tok/step. The trade is the expected one: fewer accepted tokens per step, bought back by a 22% cheaper step at
K=3. At N=8, B and A both still read as tail rows (`run` caps at 5), with 0 preemptions in every boot. Those rows
carry no verdict.

## Memory

Free VRAM after boot (`nvidia-smi`): A1/A2/A3 1,617 MiB, Bdk1 1,571 MiB. Shared Usage on the 3090 stayed at
84.2-92.2 MiB on every boot (`mem.txt`), and `spillwatch.sh` flagged nothing. Peak dedicated: A 23,615.8 MiB,
Bdk1 23,661.8 MiB.

## Verified vs assumed

Verified: the schedule in the engine's args, the pool, every number above from the raw logs, and the counter
traces. Assumed, not verified: that the per-step K followed the schedule exactly. It is inferred from the
tok/step and ms/pass drop, not from a per-step log.
