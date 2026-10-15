# Batch 2: ship the capacity win, then re-measure on it

Batch 1 (2026-10-13) is in [../bench/results/2026-10-13-SUMMARY.md](../bench/results/2026-10-13-SUMMARY.md) and
digested in [decode-speed-landscape.md](decode-speed-landscape.md#what-the-3090-measured-2026-10-13). It left two
things behind: a capacity win that needs a quality gate before it can ship, and three arms whose verdicts were
blocked by the small pool. This file is the next three sessions. The operating rules and the push discipline are
unchanged — [gpu-agent-brief.md](gpu-agent-brief.md) still governs, and
[gpu-agent-brief-batch2.md](gpu-agent-brief-batch2.md) is the paste-ready prompt for this batch.

## Why this order

The measured fact that matters: `TRITON_ATTN` + `int8_per_token_head` holds **116,440 KV tokens against 68,605
in the same pinned bytes** at the same C1 speed, with **1 preemption against 20** at N=8. That is a smaller KV
format, not more VRAM. It is also not a new component — `SPEC=dflash2 CTX=long` already ships exactly this
attention/KV stack, so only the *combination* is new.

And the pool is why batch 1's tail rows were unusable: at K=7 only ~5 requests stay resident, so N≥8 rows get
preempted and their A-to-A spread (up to 13%) is wider than any effect. Session 1 buys the pool; session 2
spends it on the questions that could not be answered.

## The noise floors batch 1 measured — decide the verdict before you run

| metric | observed spread | what it means for a verdict |
|---|---|---|
| C1 T=0 decode, within one boot | 0.0-0.1% | decisive for any effect of 1% or more |
| C1 T=0 decode, across boots | two shipped boots read 109.4 / 109.4; a different backend read +1.8% | treat 2% as the practical floor for a cross-boot C1 row |
| teacher-forced tokens/step on a frozen target | identical to three decimals between passes | the noise-free metric — prefer it to free-run |
| free-run decode, same prompt, same boot | ~10% | never call an effect under 10% from free-run alone |
| ladder rows where the pool preempts | up to 13% A-to-A | unusable — that is what Session 1 fixes |
| cohort C8 protocol | 37% within one boot (248.8 vs 340.9) | unusable as written; S2c gates it before it may be quoted |
| ladder N=4 steady state | A/A2 agree to 0.6% | an 8% effect there is outside noise |

A verdict of "inconclusive — noise" is a legitimate result, and is what batch 1 recorded three times. Do not
upgrade one to a win.

## Session 1 — the capacity gate (about 90 minutes, one image, two arms)

**A** = shipped `CTX=fast` (`FLASH_ATTN` + bf16 KV). **B** = the same boot with
`EXTRA_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"`. Both arms pin
`KV_MEM=5583457484`, `MAX_LEN=65536`, `MAX_SEQS=8`, `PORT=18020`, and the 3090 by GPU UUID in both
`CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` — byte for byte identical between arms.

**Gate 1 — quality, before any timing is believed (n=200, not n=50).** Batch 1's smoke read 0.980 against 0.960,
which is one question and cannot decide anything.

| probe | pass | fail |
|---|---|---|
| GSM8K, 200 questions, greedy, thinking off | within 1 question of A | a reproducible gap of 2 or more questions |
| perplexity, the three quality texts | within 1% of A | more than 1% worse on any text |
| long-context needle, one rung at ~64k | the same retrievals as A | any retrieval A makes and B misses |

A needle fail parks B for long context while leaving the fast-profile case open — say that rather than calling
the whole arm failed. Run the probes on both arms with interleaved boots so a thermal or cache effect cannot
land on one arm.

**Gate 2 — speed and capacity, A-B-A interleaved (three boots per arm, alternating).** Metrics: C1 T=0 from
`bench/run_benchmarks.sh single` (second run of each boot), `bench/conc_ladder.py --n 4,8` for the decode
aggregate, ms/pass, preemptions and resident-request count, and the engine's own `GPU KV cache size` line.

Pass = C1 within 2% of A (a tie is the expected result) **and** at least one of: pool tokens up 50% or more, or
N=8 preemptions at most a quarter of A's.

**Decision.** If both gates pass, the outcome is a one-line proposal to make B the `CTX=fast` default, written
up with these numbers. **This batch does not flip a default itself** — see the standing rules.

## Session 2 — the parked arms, on the bigger pool (about 90 minutes)

Prerequisite: Session 1 passed. Every arm here runs on **B's base config**, because that is the pool that makes
the tail rows measurable. If Session 1 failed, stop and report; S2's arms inherit the preemption problem.

**S2a — dynamic K.** A = fixed `DFLASH_TOKENS=7`. B = the same boot plus a second `--speculative-config` in
`EXTRA_ARGS` carrying `"num_speculative_tokens_per_batch_size":[[1,3,7],[4,512,3]]`. Batch 1 verified this
override works and that no launcher change is needed, so there is no `DYNAMIC_K` knob to add — the recipe is the
knob. Keep the full JSON in the arm's `.env` so the arm stays reproducible.

- Interleaved A-B-A at N=4 and N=8, plus C1 T=0.
- Pass = N=4 decode aggregate at least 5% above A with C1 within 1% (the C1 floor is 0.0-0.1% within a boot, so
  1% is a fair band). Batch 1 read +8.0% with ms/pass 57 → 43 ms; this run exists to move that from "lead" to
  "result" or to "noise".
- Record tokens/step as well: batch 1 saw it fall (3.50 → 2.95) while the aggregate rose, which is the expected
  trade and worth showing.

**What this arm did not buy (found in batch 3).** B booted with `num_spec_tokens=7` — the schedule varies k per
step, but the recurrent-state reservation is sized for the *configured* maximum, so the arm bought no extra
seats. Its +9.2% at N=4 was a compute win. Lowering the configured `DFLASH_TOKENS` at boot is the only lever
that moves seats, and that is [batch 3](decode-speed-batch3.md).

**S2b — graph capture size.** A = shipped `CG` (resolves to 64). B = `CG=40`. Interleaved A-B-A.
- Pass = at least 3% on C1 T=0 with the narrower capture set confirmed in the boot log, and graph MiB recorded.
  Batch 1 read +3.3% once, without A-B-A.

**S2c — make the C8/C16 rows usable (a measurement gate, not a speed gate).** Run the cohort ladder at C8 and
C16 on B's base with preemptions and resident count recorded.
- Pass = within-boot spread at or under 10%. Only then may the row be quoted. If it is still wide, report the
  spread and the preemption count and stop calling C8 a measurement.


## Session 3 — a publishable comparison, and the effort gate (about 60 minutes)

**S3a — the r0b0tlab-matched row.** Batch 1 measured 196.0 / 196.8 tok/s C1 and 5.795 tokens/step on GSM8K
greedy, thinking off, against r0b0tlab's published 162.9 and 5.66 — but their protocol is **max 512 new tokens
at ctx 8192** and ours was 1024 at `MAX_LEN=65536`, so the two cannot be quoted side by side yet. Re-run our
stack at their protocol: GSM8K greedy, 512 new tokens, `MAX_LEN=8192`, thinking off, same card at 250 W.
- Report C1 decode (two reps), tokens/step, and the engine's `GPU KV cache size`.
- This row is the whole point: it either makes "we are ahead on the like-for-like workload" a quotable claim or
  it retracts it. Their power limit is unpublished, so say that in the row rather than assuming it matches.

**S3b — `reasoning_effort=low` quality gate.** Batch 1 showed the served template accepts `low` (omitted =
`xhigh`) and that `low` raises acceptance: code +21.9%, chat +14.8% free-run, gsm8k +5.2%. What was not checked
is whether `low` answers as well.
- Arms: default (xhigh) and `low`, **thinking on**, interleaved boots.
- Metrics: GSM8K n=200 quality, IFBench if the image has it, and teacher-forced tokens/step on each arm's own
  frozen continuation (the effort setting changes the prompt, so each arm must be scored against its own — batch
  1 established this and it still holds).
- Pass = `low` within 1 question of xhigh on GSM8K and not worse on IFBench. Then the outcome is a documented
  recommendation to serve agentic traffic at `low`, with the acceptance numbers beside it.

## Standing rules for this batch

1. **No default flips.** Sessions 1-3 produce evidence and a proposal; changing a launcher default is a reviewed
   one-line change made afterwards, not part of an experiment run. Batch 1 followed this and it is why its
   results are trustworthy.
2. **Reuse the image** unless the branch has changed something outside `docs/` and `bench/results/` since the
   image was built. Check with `git diff --stat <image-commit>..HEAD -- . ':(exclude)docs' ':(exclude)bench/results'`
   and record the answer; do not rebuild out of habit.
3. **Interleave** every arm pair that is expected to land inside 10%. Batch 1's single-boot arms are why three
   verdicts are stuck at "inconclusive".
4. **n=200 or say it is a lead.** No quality claim from n=50.
5. **A spilled or preempted run is not a timing.** Keep the Shared Usage trace and the preemption count beside
   every row, as in batch 1.
6. Everything else from [gpu-agent-brief.md](gpu-agent-brief.md) stands: results-only commits, push after each
   session, no PRs or issues or comments, no co-authored commits, no watermarks, no soak, no assumptions.

## What batch 2 is not

No EXL3, no TurboQuant, no SGLang, no FlashInfer graph chase, no GQA-packed split-KV port, no second-GPU work.
Each of those was considered in [decode-speed-landscape.md](decode-speed-landscape.md) and parked with a reason;
none of them gets faster because it is retried. Batch 2 exists to bank the two cheap wins and to unblock the
three questions the small pool has been hiding.

