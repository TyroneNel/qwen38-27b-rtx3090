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

**Proposal (S1, not applied):** make `TRITON_ATTN` + `int8_per_token_head` the `CTX=fast` default. It holds 116,440
KV tokens against 68,605 in the same pinned bytes and preempts 1-2 times against 23 at N=8, at an unchanged C1
(111.2 vs 111.0 tok/s), with GSM8K n=200 within one question and perplexity within 0.07%.

## What is now known

- The capacity win holds a quality gate at n=200: no reproducible GSM8K gap, perplexity identical to ±0.07%, and
  the ~60k needle is retrieved by both arms.

## What could not be established

- (filled in as sessions complete)

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
