# S1: the capacity gate, `TRITON_ATTN` + `int8_per_token_head` on `CTX=fast`

RTX 3090, 250 W (`power.limit 250.00 W` in every `raw-smi-*.log`), container boot, image `hyperqwen:exp-f1a033e`
(`image.txt`).

## Arms and what the engine got

Both arms: `.env.A` / `.env.B`, byte-identical except B's `EXTRA_ARGS` line: `SPEC=dflash2 PREFIX_CACHE=1 CTX=fast
DFLASH_TOKENS=7 KV_MEM=5583457484 MAX_LEN=65536 MAX_SEQS=8 PORT=18020 PREPARE=0 VERIFY=0`, the 3090 pinned by UUID.

| arm | asked for | selected backend (`cuda.py:532`, every boot) | `kv_cache_dtype` (non-default args) | `GPU KV cache size` | graphs |
|---|---|---|---|---|---|
| A | shipped: `FLASH_ATTN` + bf16 | `Using AttentionBackendEnum.FLASH_ATTN backend.` | `'bfloat16'` | **68,605 tokens** | took 0.13 GiB |
| B | `--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head` | `Using AttentionBackendEnum.TRITON_ATTN backend.` | `'int8_per_token_head'` | **116,440 tokens** | took 0.13 GiB |

Ten boots: quality qA, qB, qB2, qA2, then timing tA1, tB1, tA2, tB2, tA3, tB3, in that order. Every A boot and
every B boot reported the same pool and graph line.

## Gate 1: quality (n=200, interleaved A B B A)

| probe | A | B | delta | rule |
|---|---|---|---|---|
| GSM8K n=200, `quality_battery.py`, boot 1 (`raw-gsm-qA.log`, `raw-gsm-qB.log`) | 0.965 (193) | 0.955 (191) | -2 questions | |
| GSM8K n=200, `quality_battery.py`, boot 2 (`raw-gsm-qA2.log`, `raw-gsm-qB2.log`) | 0.955 (191) | 0.960 (192) | +1 question | |
| GSM8K n=200, per-question tool, boot 2 (`raw-gsmq-*.log/.json`) | 191 | 191 | 0 | |
| **GSM8K, all runs** | 193 / 191 / 191 | 191 / 192 / 191 | mean 191.7 vs 191.3, -0.3 questions | gap of 2 not reproduced: **pass** |
| perplexity en (wikitext-2, 11,175 tok) | 10.7656 | 10.7605 | -0.05% | within 1%: **pass** |
| perplexity da (fineweb-2, 13,917 tok) | 10.9105 | 10.9105 | 0.00% | **pass** |
| perplexity code (vLLM source, 7,553 tok) | 3.5983 | 3.6008 | +0.07% | **pass** |
| needle, ~60k tokens at 90% depth (`raw-needle-q*.log`) | RETRIEVED | RETRIEVED | same | **pass** |

The first pair read a 2-question gap, which would have been a fail if it had reproduced. The second pair ran B
first, and it reversed: B 192, A 191. A alone moved 2 questions between its two boots (193, 191), so a 2-question
gap is inside this lane's run-to-run spread. The per-question run shows why. Both arms miss 9 questions, and 7 of
the 9 end with `finish_reason=length` at the lane's 768-token cap. The two arms disagree on exactly one question
each: B misses #153 (truncated), A misses #87 (truncated). The verdict is the brief's: no reproducible gap of 2.

**Spill on the quality boots.** The first pair spilled. Shared Usage rose from the 84-92 MiB floor to 156.2 MiB
(qA) and 410.2 MiB (qB) at the perplexity step (`prompt_logprobs`), and stayed there for the rest of the boot
(`mem.txt`). A spill moves memory, not arithmetic, so the accuracies and perplexities stand. The wall times of
those two boots (GSM8K 205 s vs 318 s, needle 70.9 s vs 127.5 s) are void and are not compared here. The repeat
pair skipped perplexity and stayed at the floor (92.2 MiB). The live spill guard added after this
(`../2026-10-14-batch2-tools/spillwatch.sh`) was running on tB2-tB3, and no timing boot spilled.

## Gate 2: speed and capacity, A-B-A interleaved (three boots per arm)

Metrics: C1 T=0 = `bench/run_benchmarks.sh single`, row `cohort C1 real prompts T=0`, decode(C/meanTPOT), second
run of the boot (`raw-rb-<label>-2.log`). Ladder = `bench/conc_ladder.py --n 4,8` (`raw-ladder-<label>.log`).
`*` marks a row the ladder flags as a tail measurement.

| boot | C1 T=0 run 1 / **run 2** | C1 tok/step | N=4 decode agg / ms/pass / preempt | N=8 decode agg / preempt / resident (`run`) | cohort C8 T=0 run 2 | free VRAM after boot / after ladder |
|---|---|---|---|---|---|---|
| tA1 | 110.3 / **109.8** | 3.33 | 235.5 / 57.6 / 0 | 218.3* / 23 / 5 | 250.2 | 1,477 / 557 MiB |
| tB1 | 110.1 / **110.0** | 3.34 | 238.9 / 59.1 / 0 | 165.3* / 1 / 6 | 361.7 | 1,365 / 463 MiB |
| tA2 | 110.1 / **110.0** | 3.33 | 235.0 / 56.1 / 0 | 222.5* / 23 / 5 | 250.2 | 1,477 / 557 MiB |
| tB2 | 109.9 / **109.4** | 3.34 | 234.6 / 58.9 / 0 | 171.5* / 1 / 6 | 362.6 | 1,365 / 463 MiB |
| tA3 | 110.9 / **113.3** | 3.33 | 250.5 / 54.8 / 0 | 216.0* / 23 / 5 | 350.9 | 1,477 / 557 MiB |
| tB3 | 114.3 / **114.3** | 3.34 | 241.3 / 57.3 / 0 | 187.7* / 2 / 6 | 372.1 | 1,365 / 463 MiB |

| metric | A (mean of 3) | B (mean of 3) | delta |
|---|---|---|---|
| C1 T=0 decode, tok/s | 111.0 | 111.2 | **+0.2%** (pairs: +0.2%, -0.5%, +0.9%) |
| GPU KV cache size | 68,605 tokens | 116,440 tokens | **+69.7%** |
| N=8 preemptions | 23 / 23 / 23 | 1 / 1 / 2 | **-94%** (B ≤ 1/12 of A) |
| N=8 resident requests | 5 / 5 / 5 | 6 / 6 / 6 | +1 |
| N=4 decode agg (steady state, 0 preemptions) | 240.3 | 238.3 | -0.9% |

**Verdict: pass.** C1 is within 2% (+0.2%, a tie), the pool is up 69.7% (rule: ≥50%), and N=8 preemptions are
1-2 against 23 (rule: ≤¼). Gate 1 passed.

**Proposal (not applied):** make `--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head` the
`CTX=fast` default in `single-user/start_qwen.sh`. It holds 116,440 KV tokens against 68,605 in the same pinned
5,583,457,484 bytes and preempts 1-2 times against 23 at N=8, at C1 T=0 111.2 vs 111.0 tok/s. GSM8K n=200 is
within one question (191.3 vs 191.7 mean of three runs) and perplexity is within 0.07% on all three texts.

## Things this run shows that the gate did not ask

- **B is still pool-bound at N=8.** `run` stops at 6 of 8 and `kv%` reads 99.7-100.0. The N=8 decode aggregate of
  B (165-188) reads *below* A's (216-222) in all three pairs. Both are tail rows, but the direction repeats in all
  three pairs, as it did in batch 1 (181.9 vs 213/242). Fewer preemptions did not buy aggregate throughput at
  N=8 on this ladder. N=4 is a tie.
- **The cohort C8 row splits by arm.** A reads 250.2 / 250.2 / 350.9 (the same bimodality batch 1 saw: 248.8 vs
  340.9). B reads 361.7 / 362.6 / 372.1, a 2.9% spread. The 37% spread batch 1 recorded on C8 is an A-arm
  (small-pool) property. S2c tests this on B's base.
- **The cross-boot C1 floor here is 3%, not 2%.** A read 109.8, 110.0 and 113.3. The last pair (tA3, tB3) ran
  ~3% faster in both arms, and tA3 moved 2.2% between its own two runs (110.9 → 113.3). Batch 1's "0.0-0.1%
  within a boot" holds for 5 of the 6 boots. Pairing removes the drift: per-pair deltas are -0.5% to +0.9%.
- **Free VRAM during load** falls to 557 MiB (A) and 463 MiB (B) after the ladder, with Shared Usage flat. B's
  463 MiB is ~30-70 MiB above the 396-436 MiB band where `start_qwen.sh`'s comments (gotcha 39, measured at
  `CTX=huge`) record first-batch deaths. Sessions 2 and 3 therefore start with `KV_MEM` 256 MiB lower (their
  `knob-changes.txt`).

## Memory

Free VRAM after boot: A 1,477 MiB, B 1,365 MiB (host `nvidia-smi`), both above the 1.3 GB rule. Shared Usage on
the 3090 (`luid_..11456`, identified in `../2026-10-14-batch2-preflight/`) stayed at the loaded floor
(84.2-92.2 MiB) on every timing boot and on qB2/qA2. Peak dedicated: A 23,773.8 MiB, B 23,867.8 MiB. qA and qB
spilled (above). Full traces: `mem-trace-<label>.csv`; per-boot summary: `mem.txt`.

## Verified vs assumed

Verified: the args, backend and KV dtype each engine received, the pool and graph lines, every number above
from the committed raw logs, and the counter traces. Assumed, not verified: that the 3% upward step on tA3/tB3
is a machine-state effect (thermal or clock). Temperatures read 80-81 °C after every ladder, so it is not
visibly thermal. Its cause is unknown.
