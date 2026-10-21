# X-02a inputs

Every number `fit.py` takes in. **M** = measured (raw evidence path given), **E** = estimated (reasoning given).

Paths: `R/` = this directory's `raw/`, a verbatim copy of the first lines of each source log. The sources live only on remote-tracking branches, so I read them with read-only `git show` (no checkout, commit or branch switch):
`C` = `origin/cline/gf59cdmh` (batches 3-4, "Decode-speed programme"), `S` = `origin/dynamic-k-results`. Line numbers below are the line in the `R/` copy (the ladder logs keep their original line numbers 1-10).

Serena note: Serena has only a Python language server here and rejects `.sh` files ("Cannot extract symbols ... Active language servers: ['python']"), so the launcher lines were read with Read/grep, not Serena.

## 1. The existing DYNAMIC_K table (comparator)

| Item | Value | Source | Label |
|---|---|---|---|
| Table | `[[1,3,7],[4,512,3]]` = k=7 for N=1-3, k=3 for N>=4 | `single-user/start_qwen.sh:678` (comment), `:692` (injected as `num_speculative_tokens_per_batch_size`), `single-user/README.md:434`, `bench/test_dynamic_k_knob.sh:16` | M (code) |
| It is a documented example, not a built-in default | `DYNAMIC_K` is unset by default; format is inclusive `[first,last,k]` ranges | `single-user/README.md:434` | M (code) |
| Buys step time, not seats | state pages stay sized for `num_speculative_tokens` | `single-user/start_qwen.sh:679-681` | M (code) |
| Refused with `PTH_FA=1` on dflash2 | | `single-user/start_qwen.sh:324-327` | M (code) |
| Refused at `CTX=huge` + prefix cache | | `single-user/start_qwen.sh:686-690` | M (code) |
| k may be 0 (native OFF) | `num_speculative_tokens < 0` raises, so 0 is legal | `~/vllm/vllm/v1/spec_decode/dynamic/utils.py:52-55` | M (code) |
| Reported gain at 4 streams | +7.7%, +8.0%, +9.2% (three A/Bs), single stream unchanged | `single-user/start_qwen.sh:679`, `single-user/README.md:434` | M, project-reported. Only the +7.7% run has raw data here (below); the +8.0/+9.2% runs are in the original `launcher-dynamic-k` branch comment (`dynamic-k-results:single-user/start_qwen.sh:518` lists +8.0%/+9.2% in "two A/Bs"), not re-read |

## 2. DYNAMIC_K A/B (back-test target), bf16 FLASH_ATTN, `MAX_SEQS=8`, `DFLASH_TOKENS=7`, 4k distinct salted prompts, out=1024, 6 reps, ABBA boots

Source: `R/dynamic-k-ab-salted-report.md` (original `S:bench/results/2026-10-16-dynamic-k-ab-salted/report.md`), env `R/dynk-salted.env.B` (`DYNAMIC_K='[[1,3,7],[4,512,3]]'`), raw ladders `R/dynk-salted-ladder-{A1,B1,B2,A2}.log`.

| Cell | k=7 arm (A) | DYNAMIC_K arm (B) | Line | Label |
|---|---|---|---|---|
| N=4 decode agg tok/s | 241.4 | 260.0 (**+7.7%**) | :54 | M |
| N=4 ms/pass | 56.63 | 43.15 (-23.8%) | :58 | M |
| N=4 tok/step | 3.43 | 2.81 (-18.0%) | :56 | M |
| N=1 ms/pass | 28.95 | 29.22 | :52 | M |
| N=1 tok/step | 3.45 | 3.45 | :50 | M |

## 3. Step curve in the task brief

| Item | Value | Source | Label |
|---|---|---|---|
| ms per forward pass at N=1/2/4, k=7 | 25.9 / 32.8 / 49.1 | `single-user/README.md:116`; also `docs/research/01-project/dflash-specdec.md:426-431` | M (project-reported; raw data not in repo, per the review at `dflash-specdec.md:429-431`) |
| aggregate decode tok/s | 137 / 225 / 309 | `single-user/README.md:115` | M (project-reported) |
| per-stream decode tok/s | 137 / 97 / 46 | `single-user/README.md:114` | M. Note 97 x 2 = 194 and 46 x 4 = 184 do not equal the aggregates 225 and 309: per-stream is over a different window than the aggregate. Not used |
| Implied tok/step (agg x ms / N) | 3.55 / 3.69 / 3.79 | derived in `fit.out` | E |

Cross-base check: these step times are 0.88x / 0.85x / 0.86x of the `batch3 Q1` k=7 cells at N=1/2/4, and 0.89x / 0.87x of the DYNAMIC_K A/B k=7 cells (N=1/4); that A/B (same bf16 default config, later image `sha256:e99df7e6...`) reads 28.95 / 56.63 ms at N=1/4. So **the README step curve is from an older/faster boot than the A/B whose gain we back-test**. A constant scale cancels in a gain ratio; `fit.py` therefore uses the matched-boot cells in section 4 for V and keeps the README curve only as a consistency check.

## 4. Verify-step cost V(N,k) and measured acceptance E(k) per boot (ms/pass, tok/step, decode agg)

Common: `bench/conc_ladder.py`, distinct ~4096-token prompts, out=256, 2 reps (best kept), greedy, 250 W. `*` = tail row (not all N streams decoding), excluded from fits.

**Q1 (batch 3, int8 CTX=fast, `MAX_SEQS=8`, boot-time `DFLASH_TOKENS`=k)**: `R/batch3-Q1-k7-ladder.log`, `-k3-`, `-k1-`; rows at lines 4-8 (N=1,2,4,8,16).

| N | k=7 ms / tok-step / agg | k=3 | k=1 |
|---|---|---|---|
| 1 | 29.5 / 3.36 / 113.9 | 28.9 / 2.91 / 100.4 | 28.1 / 1.83 / 65.3 |
| 2 | 38.4 / 3.46 / 179.6 | 32.0 / 2.85 / 175.9 | 30.0 / 1.80 / 119.8 |
| 4 | 57.0 / 3.51 / 239.2 | 42.3 / 2.89 / 268.4 | 35.2 / 1.80 / 203.9 |
| 8 | 45.9 / 3.45 / 178.5 * (6 of 8 running) | 61.8 / 2.80 / 350.1 | 48.2 / 1.82 / 299.4 |

**Q2 (batch 4, int8 CTX=fast, `MAX_SEQS=16`)**: `R/batch4-Q2-{k7,k3,k1,off}-ladder.log`, rows at lines 4-8. `MAX_SEQS=16` costs a larger scratch buffer (cline X-3, `findings-S1-row.txt` context; k=7 N=1 is 32.7 vs 29.5 ms in Q1), so Q1 is the base for k>=1.

| N | k=7 | k=3 | k=1 | k=0 (SPEC=off) |
|---|---|---|---|---|
| 1 | 32.7 / 3.45 / 105.4 | 29.4 / 2.98 / 100.9 | 29.7 / 1.81 / 60.9 | 25.0 / - / 40.1 |
| 2 | 42.5 / 3.60 / 167.4 | 33.2 / 2.86 / 170.0 | 30.9 / 1.80 / 115.7 | 25.8 / - / 77.5 |
| 4 | 60.3 / 3.67 / 239.1 | 42.2 / 2.82 / 263.6 | 35.8 / 1.82 / 202.3 | 27.9 / - / 143.2 |
| 8 | tail * (5 running) | 63.9 / 2.79 / 337.1 | 49.1 / 1.81 / 294.9 | 31.9 / - / 250.7 |
| 16 | tail * | tail * | tail * | 43.3 / - / 369.2 |

All **M**. The k=0 row exists only in Q2 (`R/batch4-Q2-off-ladder.log:4-8`); it is a SPEC=off boot, so the cost of k=0 *inside* a spec boot is unmeasured (sensitivity knob V0 x0.8-1.5).

Pooled E (computed in `fit.py`, **M** averages): k=1 1.811 (8 cells), k=3 2.862 (8 cells, N<=8), k=7 3.508 (6 cells, N<=4). The A/B boots in section 2 are *not* in the pool, so the back-test's acceptance is independent of the experiment being reproduced.

Measured S-1 band row (cross-check only, int8 TRITON 16-seat): `R/findings-S1-row.txt` (= `C:docs/decode-speed-findings.md:83`): k=7 N=4 239.1, k=3 N=4 263.6 (+10.2%), k=3 N=8 337.1, SPEC=off N=8 250.7, N=16 369.2. The branch's "+41% at N=4" does not reconcile with these (as in `reviews/phase3-transferability.md`); not used.

## 5. Per-position acceptance (spec-5)

Source: `~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/spec-5.md`, B (DFlash2), T=0.7, thinking off, 8 prompts, <=512 tokens, `DFLASH_TOKENS=7`.

| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | Line | Label |
|---|---|---|---|---|---|---|---|---|---|
| realised accept | 0.687 | 0.511 | 0.404 | 0.325 | - | - | 0.190 | :40-44 | M |
| E7 (tok/step) | 3.611 realised, 3.597 survival | | | | | | | :33 | M |
| a4, a5 | 0.267, 0.228 | | | | | | | `fit.py` | E: unpublished; a4 = a3 x 0.82 (conditional trend 0.805-0.841, `:41-44`), a5 = remainder so that sum = E7 - 1 |
| T=0 total | 3.486 | | | | | | | :32 | M (per-position at T=0 not published) |

**Mismatch found (the central finding).** spec-5 implies E3/E7 = 0.721. The ladder boots measure E3/E7 = 0.816 pooled and 0.819 in the DYNAMIC_K A/B itself (T=0, 4k-token prompts). The ladder's a0 is 0.81 (k=1 boots give 1.81 tok/step) against spec-5's 0.687. Spec-5's per-position vector does not describe the ladder workload, so `fit.py` carries two acceptance profiles: `spec-5` (as specified) and `ladder-matched` (E1, E3, E7 from section 4; increments split across positions by spec-5's within-group shape: **E**).

## 6. Other

| Item | Value | Source | Label |
|---|---|---|---|
| Per-resident cost | ~7 ms per extra resident at k=7 | `docs/optimizations.md:358-366`, `dflash-specdec.md:426-431` | M (project-reported) |
| Per verify position | ~1 ms attention at 25k ctx | `docs/optimizations.md:288` | M (long context only; not used) |
| Marlin stair at B*q = 16 (~8 ms) | | `transferability.md` I-05 row (line 200) | E, review claim; the Q1/Q2 per-position slopes (N=4: 3.5-3.7 ms per k) show no isolated step, so no separate stair term is fitted |
| Drafter proposes a fixed 7-token block regardless of verify k | | `single-user/README.md:430` (`SPEC` row: "the drafter always proposes the 7 it was trained for") | M (code docs). Hence D(N) per position = 0 in the base case: ms/pass already includes drafting |
| Seat limits: k=7 holds ~5-7 residents at 4k prompts, k=3 holds ~13 | | `C:bench/results/2026-10-14-batch3-SUMMARY.md` Q1 table | M (branch-reported) |
| DYNAMIC_K at N=8 on a k=7 boot | 191.2* / 193.6* tok/s, 5 running (tail) | `C:bench/results/2026-10-14-S2a-dynamic-k/raw-ladder-{A1,Bdk1}.log:5-6`, not copied | M: DYNAMIC_K buys no N=8 steady state on a k=7 boot |
