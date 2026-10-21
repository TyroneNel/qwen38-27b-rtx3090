# runtime-2: step cost against verified tokens, per backend and depth

| Field | Value |
|---|---|
| Date | 2026-10-21 10:03–10:14 SAST (rev 2 run; the rev 1 run of 03:40 is superseded, see below) |
| Commit | `bench/step_curve.py` rev 2, added in this result's commit |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock; the tool boots each backend's engine in its own process, using the launcher's own argv and environment (`single-user/start_qwen.sh` with `PRINT_ARGV=1`) |
| GPU | RTX 3090, 250 W, driver 610.88; busy median SM clock 1590 MHz; WDDM non-local flat (86 → 94 MiB) |
| Backends | `fa2` = `SPEC=dflash2 CTX=long FA2_FP8KV=1 PTH_FA=0` (FA2 fp8 KV); `pthfa` = `SPEC=dflash2 CTX=long PTH_FA=1` (D′'s default); `kvarn` = `SPEC=dflash2 CTX=huge` (E) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\runtime-2\serve-rev2\step_curve-rev2.{jsonl,csv,plot.txt}` |

## Method

```bash
docker run ... --entrypoint /app/venv/bin/python ghcr.io/syv-ai/hyperqwen:latest \
  /app/bench/step_curve.py --backend fa2,pthfa,kvarn --out /out/step_curve-rev2.jsonl
```

- vLLM's own V2 dummy step with a context length, run with the backend's real graph dispatch.
- Each point is a GDN verify of one request with q−1 drafts, at depths 8k–150k (240k on KVarN only).
- It reports target forward, drafter and lm_head separately, plus their sum ("step"). The sum leaves out rejection sampling and host scheduling.
- Self-check: from 8k to 150k, the q=8 step must rise by at least the DRAM time to read the extra KV once. All three backends PASS.

Serving captures FULL graphs only at q=8 (k+1) and PIECEWISE graphs at 1, 2, 4 and 8. Any other width is padded up to the next of those sizes, so in serving **the only reachable verify widths are 1, 2, 4 and 8**, and the curve is reported at those widths. `seq1pw` is q=8 forced to PIECEWISE, so the slope from 1 to 8 is read in a single graph mode.

**Rev 1 is invalid.** Two dummy-step bugs, both found and fixed before this run:
1. FULL-graph capture metadata reset pthfa's `seq_lens` to 1, so pthfa never saw the context.
2. Widths other than 8 ran GDN's chunked-prefill path instead of its verify path.

The rev 1 rows are kept in `serve/` for the record and are not used.

## Numbers: step ms (target forward + drafter + lm_head), one request, PIECEWISE

| Backend | Depth | q=1 | q=2 | q=4 | q=8 | q=8 FULL (served) | ms per extra token, q 4→8 | ms per extra token, q 1→4 |
|---|---|---|---|---|---|---|---|---|
| fa2 | 8k | 27.2 | 32.6 | 32.6 | 33.2 | 27.5 | 0.15 | 1.8 |
| fa2 | 64k | 40.0 | 42.6 | 43.1 | 43.2 | 40.5 | 0.03 | 1.0 |
| fa2 | 112k | 47.8 | 50.5 | 51.0 | 51.2 | 48.3 | 0.04 | 1.1 |
| fa2 | 150k | 53.0 | 55.7 | 56.2 | 56.3 | 53.6 | 0.03 | 1.1 |
| pthfa | 8k | 24.6 | 32.2 | 32.2 | 32.5 | 25.0 | 0.08 | 2.5 |
| pthfa | 64k | 28.5 | 31.6 | 32.0 | 35.8 | 33.4 | 0.95 | 1.2 |
| pthfa | 112k | 31.8 | 34.2 | 35.3 | 42.8 | 40.4 | **1.88** | 1.1 |
| pthfa | 150k | 34.4 | 36.8 | 38.1 | 48.5 | 45.9 | **2.60** | 1.2 |
| kvarn | 8k | 34.1 | 40.7 | 40.3 | 41.0 | 31.1 | 0.17 | 2.1 |
| kvarn | 64k | 34.3 | 42.6 | 54.6 | 77.7 | 74.7 | **5.8** | 6.8 |
| kvarn | 112k | 40.7 | 53.7 | 74.5 | 115.6 | 112.6 | **10.3** | 11.3 |
| kvarn | 150k | 44.4 | 61.3 | 89.6 | 145.4 | 142.6 | **13.9** | 15.1 |
| kvarn | 240k | 57.7 | 83.2 | 127.8 | 217.1 | 214.0 | **22.3** | 23.4 |

The drafter is flat at 3.2 ms and lm_head at 0.77 ms per step at every depth on every backend; all the growth is in the target forward. Several requests (2–4 × 8 tokens, FULL) scale as expected: on pthfa at 64k, 42.2 / 61.2 / 68.4 ms for 2 / 3 / 4 requests.

## Reading against the appendices

- **The 3.6 vs 1.2 ms per verify token question has a three-way answer, one per backend:**
  - **FA2-fp8 is flat.** Above q=2 an extra verify token costs ~0.03 ms at 64–150k: the stair is one wide tile. Trimming verify width saves nothing on FA2.
  - **pthfa has the stair the appendix predicted.** q 1→4 costs ~1.1–1.2 ms per token (spec's ~1.2 ms holds). The step from 4 to 8 costs 1.9 ms per token at 112k and 2.6 at 150k, because T=8 moves from the unpadded decode kernel to the padded 64-row tile. Trimming 8 → 4 saves 7.5 ms (18%) at 112k and 10.4 ms (21%) at 150k. Trimming 8 → 6 would save nothing, since 5–8 all pad to 8.
  - **KVarN is linear and steep.** It costs 5.8 ms per verify token at 64k, 13.9 at 150k and 22.3 at 240k; drafter's 3.6 ms is too low here, not too high. At 240k, verifying 4 tokens instead of 8 cuts the step from 217 to 128 ms (−41%).
- **runtime-1's confirmation test** ("slope under 5% of the base step per extra token ⇒ the ceiling holds") passes on FA2 (≈0.1%) and on pthfa below T=4 (~3–4%). It fails badly on KVarN (10% at 240k).
- **The stairs settle spec-0's sign.** On KVarN, a lower fixed `DFLASH_TOKENS` saves 10–22 ms per dropped token at 112–240k, far more than the simulation's 1–2 ms. spec-0 is now the most promising cheap item for E at depth. On pthfa (D′), only a trim to k ≤ 3 (q=4) pays, as the appendix's stair caveat said.
- **Corrections to the appendices:**
  - KVarN's per-token verify cost is 5–20× the 1–2 ms the spec lane simulated.
  - FA2-fp8 has no per-token cost at all above q=2.
  - drafter-1's 3.6 ms "upper bound" is exceeded by KVarN at 112k and above, and it is 2× too high for pthfa below the stair.

## Verdict

INFO. The step-cost curve exists and backs every verify-length item:

| Backend | Effect of trimming verify width |
|---|---|
| FA2-fp8 | Saves nothing |
| pthfa | Pays only at T ≤ 5 (−18 to −21% per step at 112–150k for 8 → 4) |
| KVarN | Pays linearly, about −41% per step at 240k for 8 → 4 |

The KVarN slope plus e-profile's 78% attention share make mode E at depth the place where every verify-length idea (spec-0, spec-1, spec-2, drafter-1) has the most to gain. kvarn-varlen is the blocker for all of them except spec-0.
