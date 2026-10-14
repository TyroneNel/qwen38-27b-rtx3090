# Batch 2, 2026-10-14: summary

One RTX 3090 at 250 W (verified `power.limit 250.00 W`), Docker Desktop WSL2, container boot, branch
`cline/gf59cdmh`. The image `hyperqwen:exp-f1a033e` (sha256:c06b97c7…) is a local rebuild of the same code tree as
batch 1, because batch 1's image was no longer on the machine (`2026-10-14-batch2-image/image.txt`). Every arm pins
the GPU by UUID and the pool in bytes. A live spill guard (`2026-10-14-batch2-tools/spillwatch.sh`) watches the
3090's Shared Usage counter on every boot from tB2 onward.

| arm | A | B | delta | verdict | noise floor for that row |
|---|---|---|---|---|---|
| S1 Gate 1, GSM8K n=200 thinking off | 193 / 191 / 191 | 191 / 192 / 191 | -0.3 questions (mean) | pass: a 2-question gap in pair 1 reversed in pair 2 | A alone 193 vs 191 between boots; 7 of 9 misses are 768-token truncations |
| S1 Gate 1, perplexity en / da / code | 10.7656 / 10.9105 / 3.5983 | 10.7605 / 10.9105 / 3.6008 | -0.05% / 0.00% / +0.07% | pass | not measured (one run per arm) |
| S1 Gate 1, needle ~60k | RETRIEVED | RETRIEVED | same | pass | — |
| S1 Gate 2, C1 T=0 (A-B-A x3) | 109.8 / 110.0 / 113.3 | 110.0 / 109.4 / 114.3 | +0.2% (pairs +0.2, -0.5, +0.9%) | pass (tie) | cross-boot 3.2% (A1 vs A3); in-pair ≤0.9% |
| S1 Gate 2, KV pool | 68,605 tokens | 116,440 tokens | +69.7% | pass (≥50%) | identical every boot |
| S1 Gate 2, N=8 preemptions | 23 / 23 / 23 | 1 / 1 / 2 | -94% | pass (≤¼) | — |
| S2a dynamic K [[1,3,7],[4,512,3]], N=4 decode agg | 235.4 / 241.6 (flanking), A3 247.1 | 260.5 | +9.2% (ms/pass 56.5 → 43.8, tok/step 3.44 → 2.90) | pass (≥5%) | A boots spread 5.0% at N=4 |
| S2a dynamic K, C1 T=0 | 114.9 / 115.3 | 115.6 | +0.4% | pass (within 1%) | in-session A spread 1.2% |
| S2b `CG=40`, C1 T=0 | 115.3 / 116.3 (flanking) | 120.5 | +4.1% | inconclusive: all of it is tok/step (3.34 → 3.48); ms per step 28.9 vs 28.7-29.1 | step time flat to 0.3% |
| S2c cohort C8 (3 reps, one boot) | — | 346.9 / 347.4 / 346.9 | spread 0.1% | passes the spread gate; 4 preemptions every rep, peak 7 running | — |
| S2c cohort C16 | — | 664.5 / 708.3 / 694.7 | spread 6.6% | not a measurement: `MAX_SEQS=8` caps running at 7, so C×1000/TPOT double-counts the queue; prompts oversampled 8 → 16 | — |

**Proposal (S1, not applied):** make `TRITON_ATTN` + `int8_per_token_head` the `CTX=fast` default. It holds 116,440
KV tokens against 68,605 in the same pinned bytes and preempts 1-2 times against 23 at N=8, at an unchanged C1
(111.2 vs 111.0 tok/s), with GSM8K n=200 within one question and perplexity within 0.07%.

**Proposal (S2a, not applied):** ship the dynamic-K schedule `[[1,3,7],[4,512,3]]` on the int8 `CTX=fast` base. It
raised the N=4 decode aggregate +9.2% (above all three A boots) at an unchanged C1 (+0.4%), reproducing batch 1's
+8.0%. There is no launcher knob for it today; the recipe is a second `--speculative-config` in `EXTRA_ARGS`.

## What is now known

- The capacity win holds a quality gate at n=200: no reproducible GSM8K gap, perplexity identical to ±0.07%, and
  the ~60k needle is retrieved by both arms.
- Dynamic K at knee 4 is a measured +9.2% at N=4 with A-B-A, moving batch 1's lead to a result.
- On B's base with `KV_MEM` 256 MiB below the pin (pool 110,953 tokens), N=8 ladder rows preempt 0 times in all
  five S2 boots (S1 B at the full pin: 1-2; S1 A: 23), yet the ladder still flags them as tail rows (`run` 5).
- The C8 cohort row is repeatable to 0.1% on the int8 pool, so batch 1's 37% spread was a small-pool artefact.
- Spill control: Sessions 2 and 3 run with `KV_MEM` pre-cut by 256 MiB and a live spill guard. No S2 boot left the
  84-92 MiB Shared Usage floor; free VRAM after boot rose from 1,365 to 1,571-1,687 MiB.

## What could not be established

- S2b: whether `CG=40` is faster at all. Its +4.1% C1 is a tok/step shift (3.34 → 3.48) at unchanged step time;
  a teacher-forced run on a frozen target would separate acceptance from speed and was not done.
- S2c: a C16 decode number on the single-user profile (`MAX_SEQS=8` makes it a queue measurement).

## Seen that the sessions did not ask for

- **Perplexity (`prompt_logprobs`) spills this box at the shipped pin.** Both first quality boots went from the
  84-92 MiB floor to 156 MiB (A) and 410 MiB (B) Shared Usage at the perplexity step and stayed there. Timing
  boots, which do not run it, never spilled.
- **B is still pool-bound at N=8** (`run` 6, `kv%` ~100). Its N=8 ladder aggregate reads lower than A's in all
  three pairs (165-188 vs 216-222, tail rows).
- **The cohort C8 row is bimodal on A (250 / 250 / 351) and steady on B (362 / 363 / 372).**
- Machine facts (`2026-10-14-batch2-preflight/report.md`): the 3090's counter instance changed to `luid_..11456`;
  the `upstream` push URL is *not* disabled on this checkout; the local image contains the owner's untracked
  `.env.*` files, because `.dockerignore` excludes only `.env`.
