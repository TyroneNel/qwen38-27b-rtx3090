# Batch 3: settle what caps residency (G7)

One question, and it has two possible answers. Batch 2 showed that doubling the KV pool bought seats and
concurrency nothing: `int8_per_token_head` took the pool from 68,605 to 116,440 tokens and residency moved 5 → 6,
while the N=8 aggregate came out *lower* than the small-pool arm. The launcher carries a model that explains
this and predicts something we have not seen yet; this batch tests that model directly and reports whichever way
it falls.

**The model** (`single-user/start_qwen.sh`, printed at boot):

> Every RESIDENT request needs 1+k recurrent-state slots out of the same pool before it stores one token of
> context: 0.88 GiB at `DFLASH_TOKENS=7` and 1.66 GiB at 15 (gotcha 33), i.e. ~0.098 GiB per (k+2) … the ramp in
> `bench/conc_ladder.py` confirms live (15.8% of the `CTX=fast` pool per request, six resident, the seventh
> preempted).

`RESIDENT = KV_MEM / ((k+2) × 104988089)`, so at the shipped pin: **~6 seats at k=7, ~10 at k=3, ~17 at k=1**
(capped at `MAX_SEQS=8`).

**The contradiction batch 2 left.** On the S2a base — k=3 at N≥4, same pinned pool — the ladder's `run` column
still read 5. Under the model it should have read 8. So either the model over-predicts the k=3 gain, or residency
is capped by something else entirely (the `align`-mode checkpoint pages, the 2048-token batched-token budget, or
the graph/runner allocations the launcher warns about).

**Outcome (2026-10-14): Conclusion A — the model holds**, with two corrections to this design that the run had
to make. (1) The probe as specified cannot discriminate: `MAX_SEQS=8` caps every arm at 8 and k=7 already holds
7, so the literal rule reads as Conclusion B while the truth is A. The run added uncapped `MAX_SEQS=16` boots to
get past the cap — they were not in the arms below, and they are where the conclusion comes from. (2) The formula
errs **low**, not high: a request costs 81-88 MB per (k+2) unit against the launcher's 105 MB, so it prints
"about 5" seats where 7 fit. Results:
[../bench/results/2026-10-14-batch3-SUMMARY.md](../bench/results/2026-10-14-batch3-SUMMARY.md).

**Why this is worth a batch.** Every concurrency number in the repo is downstream of this: preemptions, the
ladder's tail rows, the C8/C16 spread, and the choice of k itself. If the model holds, k is a *concurrency
control* and the knee should be chosen for seats rather than for compute. If it does not, the launcher's
`RESIDENT` print is wrong and has to be corrected, and the next suspect gets its own test.

## The primary metric: the engine's own `Running:` line

Not the ladder's `run` column. `docs/optimizations.md` records the original state-pool finding as a log line —
`Running: 37 reqs, Waiting: 27` — and that line is the direct measurement. Fire a fixed load of exactly
`MAX_SEQS` concurrent requests (distinct prompts, 512 tokens out), hold it, and grep the engine log for
`Running:` / `Waiting:` while it is in flight. Report the pair verbatim. The ladder's `run` column is the
secondary metric; when the two disagree, the log line wins and both go in the report.

## Q1 — does residency scale with k? (the discriminating test)

Fixed on every arm: `KV_MEM=5583457484`, `MAX_LEN=65536`, `MAX_SEQS=8`, int8 `CTX=fast` base
(`TRITON_ATTN` + `int8_per_token_head`), `PREFIX_CACHE=1`, the 3090 pinned by UUID, the spill guard running.

| arm | `DFLASH_TOKENS` | the model predicts |
|---|---|---|
| Q1-a | 7 (shipped) | ~6 seats |
| Q1-b | 3 | ~10 → capped at 8 |
| Q1-c | 1 | ~17 → capped at 8 |

One boot per arm, same order every time, with the 8-request probe and `bench/conc_ladder.py --n 1,2,4,8,16`.

- **Conclusion A — the model holds.** `Running:` rises by 3 or more from Q1-a to Q1-b, preemptions fall toward 0,
  and the N=8/N=16 aggregate rises with it. Then k is a concurrency control, the `RESIDENT` formula is sound, and
  the knee belongs to seats. Report the seats-per-k curve.
- **Conclusion B — the model is wrong.** `Running:` stays within ±1 across all three arms (5-6 regardless of k).
  Then residency is not k-limited: the `RESIDENT` print is wrong and must be fixed, and the report names the
  next suspect with the evidence that points at it (the `align` checkpoint lines, the batched-token budget, the
  graph budget). Note the direction — batch 3 found the constant 16-23% **low**, i.e. it prints fewer seats than
  the pool holds.
- **Inconclusive is allowed** only if the probe cannot hold the load — say why (prompts too long, prefill
  interleave, port).

Do **not** change `KV_MEM` between the Q1 arms: that is the variable batch 2 already spent, and moving it re-opens
the byte-versus-token question instead of answering the seat one.

## Q2 — bf16 recurrent state (independent, and cheap)

The batch-2 boot log says which kernel we are not getting:

> Falling back to the Triton GDN decode path: the fused CUDA kernel requires … BF16 or FP32 recurrent state …

We pass `--mamba-ssm-cache-dtype float16`. bf16 is the same two bytes and is on that whitelist, so the footprint
should be identical and the fused path should come back.

- **A** = `--mamba-ssm-cache-dtype float16` (shipped), **B** = `bfloat16`. Everything else identical, k=7,
  interleaved A-B-A.
- Read: the GDN-path log line on every boot (which kernel engaged), C1 T=0, `ms/pass`, tokens/step, free VRAM
  after boot, and the `Running:` probe.
- Quality: GSM8K n=200 and the three perplexity texts on both arms. **Perplexity runs with the spill guard
  watching** — gotcha 60: perplexity alone spills this box at the shipped pin.
- **Conclusion:** B wins if the log shows the fused kernel on B and not on A, with C1 T=0 at least 3% up or
  `ms/pass` at least 3% down, tokens/step within noise, and quality within one GSM8K question and 1% perplexity.
  B loses if the fused kernel does not appear (the whitelist may be about the model config rather than this flag)
  or if quality moves. Either way, the boot-log line is the finding.

## Q3 — checkpoint pressure, as a side question

Only if Q1 and Q2 are done. Two flags aimed at the same pool, one boot each, k=7, A/B against the Q1-a boot:

- `--enable-mamba-shared-prefix-checkpoint` on (registers an `align` checkpoint at the shared-prefix junction
  where an EAGLE/MTP sibling resumes — exactly our setup).
- `--mamba-block-size 8192` (E6 tried this and could not prove it applied: the attention block stayed 448).

For both, the finding is the **boot-log evidence**, not a timing: grep every boot for the mamba/block/page lines
and quote them. The ones already known to exist are `Setting attention block size to … tokens to ensure that
attention page size is >= mamba page size`, `Padding mamba page size by …%`, and the group-size line. A flag that
changes none of them did not apply, and that is the result.

## Q4 — the instrumentation question

Answer it in the report, with evidence: **can the state pool be seen directly on this pin?** The batch-1 probe
listed the exposed mamba knobs (`--mamba-backend`, `--mamba-block-size`, `--mamba-cache-dtype`,
`--mamba-cache-mode`, `--mamba-ssu-algorithm`, `--enable-mamba-shared-prefix-checkpoint`,
`--enable-mamba-cache-stochastic-rounding`, `--mamba-cache-philox-rounds`) and noted that `use_replayssm` exists
in the config schema. Report whether `--help=all` exposes a ReplaySSM flag (batch 1 found none, and its docstring
says non-speculative decode only), and whether any boot line reports the number of mamba blocks or state slots. If
nothing does, say so — that is why the `Running:` probe is the primary metric, and a counter becomes a candidate
patch.

## Budget and order

Q1 ≈ 40 minutes (three boots, probe plus ladder each). Q2 ≈ 45 minutes (two boots, plus the quality pair with the
guard). Q3 ≈ 20 minutes. Q4 is reading, not running. Stop after Q2 if the conclusion is clear and time is short;
Q3 and Q4 are for the report's completeness, not for the answer.

