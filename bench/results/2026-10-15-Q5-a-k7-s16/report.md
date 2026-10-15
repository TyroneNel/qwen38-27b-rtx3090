# Batch 5 Q5: what the 16-seat boot costs k=7 (X-3) (2026-10-15, RTX 3090 at 250 W)

This report is identical in `2026-10-15-Q5-a-k7-s16/` and `2026-10-15-Q5-a2-k7-s16-wacc/`.

**Verdict: the loss reproduces, smaller than batch 4 measured, and it is not a slower step.**

| | numbers |
|---|---|
| C1 T=0 decode, `MAX_SEQS=16` vs `MAX_SEQS=8` (two boots each) | 102.8 / 106.8 vs 108.6 / 109.6, **−3.9% on the means** |
| Free-run tokens/step | 3.17 vs 3.34 on all four runs per setting |
| `ms/pass` at N=1 | 32.6 vs 32.3 |

- **The loss is almost all acceptance on the free-run trajectory**, the same shape as Q3's `CG` effect.
- **On a shared frozen target the gap mostly closes:** GSM8K takes 39 steps on both, and real4 takes 187 vs 182 (−2.5%).
- **The buffer that doubles is named:** the int4 3D attention scratch, `capacity=128` → `256` query tokens, 48.38 + 64.50
  → 96.75 + 129.00 MiB. It is the only `MAX_SEQS`-scaled allocation that differs in the boot logs.
- **Pass rule:** "the loss reproduces and the buffer that doubles is named". **Pass**, with the size corrected from 7.8%
  to 3.9% and the mechanism re-attributed (below).

## Arms

- **Common to both:** int8 `CTX=fast`, `DFLASH_TOKENS=7`, `KV_MEM=5315022028`, `MAX_LEN=65536`, `CG` unset (the
  launcher's min(64, `MAX_SEQS`×8) = 64 in both, `cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]`
  in both). Image `hyperqwen:exp-844cd90`.

| | A: `MAX_SEQS=16` | B: `MAX_SEQS=8` |
|---|---|---|
| boots | `Q5-a-k7-s16` (q5a); `Q5-a2-k7-s16-wacc` (q5w, a second boot, adds the frozen-target replay) | `../2026-10-15-Q3-a-cg64/` (q3a), the Q3 A arm; its first boot is in `attempt1-no-harness/` |
| pool | `GPU KV cache size: 110,953 tokens` | `GPU KV cache size: 110,953 tokens` |
| free VRAM after boot | 1,437 / 1,437 MiB | 1,617 MiB |
| spill | none on either boot | none |

B is Q3's A arm: the same env with `MAX_SEQS=8`. That reuse is why Q5 cost one extra boot, plus one for the replay.

## C1 and step time (`raw-rb-*.log` run 2, `raw-ladder-*.json`)

| boot | `MAX_SEQS` | C1 T=0 decode (run 2) | e2e | tok/step | ladder N=1 `ms/pass` | ladder N=1 tok/step |
|---|---|---|---|---|---|---|
| q5a | 16 | 102.8 | 99.17 | 3.17 | 32.6 | 3.27 |
| q5w | 16 | 106.8 | 102.91 | 3.17 | — | — |
| q3a, first boot | 8 | 108.6 | 104.58 | 3.34 | 32.3 | 3.49 |
| q3a | 8 | 109.6 | 105.81 | 3.34 | 32.3 | 3.43 |

- **Means:** 104.8 against 109.1, −3.9%.
- **Spread:** within a setting, boot-to-boot is 3.9% at 16 and 0.9% at 8, so the effect is about one spread wide.
- **Tokens/step:** 3.17 against 3.34 (−5.1%), on every run.
- **Step time:** `ms/pass` at N=1 is +0.9%.

## The same tokens: teacher-forced on Q3's frozen targets (`../2026-10-15-Q5-a2-k7-s16-wacc/raw-wacc-q5w.log` vs `../2026-10-15-Q3-a-cg64/raw-wacc-q3a.log`)

The two tasks that replayed fully clean on both boots:

| task | steps `MAX_SEQS=8` / 16 | tok/step | accept |
|---|---|---|---|
| gsm8k (226 tokens) | 39 / 39 | 5.795 / 5.795 | 0.685 / 0.685 |
| real4 (1,024 tokens) | 182 / 187 | 5.709 / 5.567 | 0.673 / 0.652 |

On every other task the boots fork from the frozen continuation at different points: `code` none vs 382, `real2` 687
vs 170, `real5` none vs 274. So the comparison holds only on these two rows.

**Reading:**
- **The free-run gap is mainly the trajectory.** The 16-seat boot's greedy numerics fork the continuation, and the
  forked text accepts less.
- **A residual survives on identical tokens.** It is smaller (0 and −2.5%) and in the same direction.

## Every `MAX_SEQS`-scaled allocation (`raw-maxseqs-bootdiff.log`)

- **The filter:** the boot logs of q3a (8) and q5a (16), filtered for `max_num_seqs|capacity|scratch|buffer|persistent|
  cudagraph_capture_sizes|graph|KV cache size|Model loading took|torch.compile|memory profiling|Free memory`.
- **What differs:** after stripping timestamps, only the following lines differ, apart from the AOT compile-cache paths
  (three different graph hashes per boot):

```
MAX_SEQS=8 : int4 3D scratch: capacity=128 query tokens (min of max_num_batched_tokens=2048, min(max_num_seqs=8, seq_threshold_3D=32) * max_query_len_3d=16); 396288 B/row, 48.38 MiB, allocated at model load (inside the memory profile)
MAX_SEQS=8 : int4 3D scratch: capacity=128 query tokens (...); 528384 B/row, 64.50 MiB, allocated at model load (inside the memory profile)
MAX_SEQS=16: int4 3D scratch: capacity=256 query tokens (min of max_num_batched_tokens=2048, min(max_num_seqs=16, seq_threshold_3D=32) * max_query_len_3d=16); 396288 B/row, 96.75 MiB, allocated at model load (inside the memory profile)
MAX_SEQS=16: int4 3D scratch: capacity=256 query tokens (...); 528384 B/row, 129.00 MiB, allocated at model load (inside the memory profile)
MAX_SEQS=8 : Model loading took 15.44 GiB memory      MAX_SEQS=16: Model loading took 15.55 GiB memory
```

- **The doubling accounts for the load difference:** +112.87 MiB, against 15.55 − 15.44 = 0.11 GiB.
- **What is identical in both:** the KV pool, the CUDA graph sizes and memory (`Graph capturing finished in 5 secs,
  took 0.13 GiB`), and the `Initial free memory 22.76 GiB` line.

## Against batch 4's X-3 and the findings doc

- **Size of the loss.** X-3 recorded C1 T=0 103.4 on a 16-seat boot against batch 3's 112.1 on an 8-seat boot (−7.8%).
  - The 16-seat side reproduces: 102.8 and 106.8 here.
  - The 8-seat reference does not: 108.6 and 109.6 on this image, not 112.1.
  - Matched on one image and one `KV_MEM`, the loss is 3.9%.
- **Step time.** X-3 recorded an N=1 step of 32.7 against 29.5 ms/pass. Matched, it is 32.6 against 32.3 ms (+0.9%).
  The "slower step" in X-3 is not reproduced. Its 29.5 came from a different boot and pool, and was not re-measured.
- **The band table.** It reads band 2's C1 cost as "carrying" this loss. The loss is acceptance on the C1 prompts'
  trajectory, so it is prompt-set-specific, like Q3's `CG` effect, rather than a fixed tax on every request.

## Verified vs assumed

**Verified:** every number is from the boot logs, `run_benchmarks` rows, ladder JSON and replay logs named above.

**Assumed / not verified:**
- That the scratch capacity is *why* the numerics move. It is the only scaled allocation that differs, and both
  boots also compiled different AOT graphs.
- A run that varies only the scratch capacity at a fixed `MAX_SEQS` would isolate it. That needs a code change, which
  is out of scope.
