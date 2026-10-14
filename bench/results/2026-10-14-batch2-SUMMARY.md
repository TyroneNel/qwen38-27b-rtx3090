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
| S3a r0b0tlab protocol, their raw prompt (40 GSM8K, greedy, 512 new, ctx 8192, C1) | r0b0tlab published: 162.9 tok/s, acceptance 5.657 | ours: 174.7 / 177.7 tok/s, acceptance 5.619 | +7.2% / +9.1% tok/s, -0.7% acceptance | quotable as "ahead on their protocol", their power limit unpublished | our rep-to-rep 1.7-4.5% |
| S3a, chat template thinking off (the brief's row) | — | 199.9 / 190.9 tok/s, acceptance 5.893 | — | our own number, not like-for-like (their prompt leaves the think block open) | rep-to-rep 4.5% |
| S3b GSM8K n=200 thinking on, xhigh vs low (2 boots) | 194 / 194 | 193 / 194 | -0.5 questions | pass (within 1) | same arm flips up to 4 questions between boots |
| S3b IFBench prompt-level strict, 300 prompts | 63.3% official; 79.4% on the 223 both answered | 47.0% official; 57.4% on the 223 | -16.3 / -22.0 pts; disagreements 55 vs 6 | **fail**: low is worse at instruction following | one sampled run per arm; a 55-to-6 split is far outside it |
| S3b teacher-forced tok/step, own targets | gsm8k 5.292, code 3.636, chat 2.854 | 5.424, 4.770, 3.316 | +2.5% / +31.2% / +16.2% | low is faster per step (as batch 1) | clean chunks as few as 1/3 on xhigh |

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
- Spill control: Sessions 2 and 3 run with `KV_MEM` pre-cut by 256 MiB and a live spill guard. No S2 or S3 boot
  left the 84-92 MiB Shared Usage floor; free VRAM after boot rose from 1,365 to 1,571-1,729 MiB.
- **On their exact protocol, our stack is ahead of r0b0tlab's ExLlamaV3 DFlash2 row:** 174.7-177.7 against 162.9
  tok/s at level acceptance (5.619 vs 5.657), at 250 W. The like-for-like claim is quotable in that form.
- **`reasoning_effort=low` is not a free speedup.** It keeps GSM8K (193-194 vs 194) and is 2.5-31% faster per
  step, but it loses 22 points of IFBench strict prompt-level on the prompts both efforts answered. No
  recommendation to serve agentic traffic at `low`.

## What could not be established

- S2b: whether `CG=40` is faster at all. Its +4.1% C1 is a tok/step shift (3.34 → 3.48) at unchanged step time;
  a teacher-forced run on a frozen target would separate acceptance from speed and was not done.
- S2c: a C16 decode number on the single-user profile (`MAX_SEQS=8` makes it a queue measurement).
- S3b: an IFBench xhigh score comparable to `docs/quality.md`'s 78.3. 63 of 300 xhigh answers ran out of the
  16,384-token budget while still thinking. On the 237 it answered, xhigh scores 80.2%. The page does not record
  the `max_tokens` its number used.
- S3a: whether r0b0tlab's power limit matches our 250 W (unpublished).

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
- **xhigh IFBench is a 7,136-token mean response.** At ~4 resident requests per 24 GB card (the recurrent-state
  pages, not the KV tokens, are the limit), the full run takes ~2.4 h. The first attempt with 32 client threads
  preempted 1,017 times on the bf16 base. Client concurrency has to match the resident limit.
- **The engine's `GPU KV cache size` falls with `max_model_len` at a fixed `KV_MEM`:** 27,714 tokens at 8,192
  against 63,872 at 57,344 (same 5,315,022,028 bytes, bf16). Not investigated.
- **`CG=40` changes greedy acceptance on the real-prompt set** (3.34 → 3.48 tok/step at C1 T=0, every run), at
  unchanged step time.
- **Spill guard false positive, fixed.** At 17:41Z the live guard read a CSV row pair mid-write: dedicated was
  written, shared not yet, and the missing value counted as 0. That made the floor 0.0 and flagged a clean boot.
  The auto-remediation cut `KV_MEM` and rebooted, which killed the IFBench low run; xhigh had already been
  scored. The cut was reverted and the guard now ignores sub-1 MiB readings and needs two consecutive samples. No
  real spill occurred in any S2 or S3 boot.
- `docs/decode-speed-landscape.md` and batch 1 contradictions: batch 1's "0.0-0.1% within a boot" C1 floor held
  on 5 of 6 S1 boots, but tA3 moved 2.2% run to run and the cross-boot floor was 3.2%, not 2%. Batch 1's C8
  "37% within one boot" is a small-pool (bf16) property: on int8 it is 0.1%.
