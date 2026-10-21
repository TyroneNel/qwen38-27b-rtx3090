# Phase 1 review: quantisation.md and heads-drafters.md

Reviewer: Opus, high effort. No GPU. `weights_breakdown.py` was re-run with fresh header range requests to the four pinned Hub commits. Its output is **byte-identical** to the committed `weights_breakdown.out` (`diff` empty). The DFlash2 `config.json` @015e7956 was re-fetched.

## Verdicts

| File | Verdict |
|---|---|
| `01-project/quantisation.md` | **PASS WITH CORRECTIONS**. The byte arithmetic is sound. The D′ KV default is wrong. The B load "match" is derived, not logged. The budget is set against 24 GiB, not the CUDA-visible 23.54. The exchange rates are in logged-pool units. |
| `01-project/heads-drafters.md` | **PASS WITH CORRECTIONS**. The parameter, byte and config claims are correct. The "MTP k=3" acceptance row is actually MTP k=4 at `CTX=fast`, not mode D. |
| `01-project/weights_breakdown.py` / `.out` | **PASS**. Reproduces exactly. The categorisation logic is correct. The pin line cites in its §6 header are 1-2 lines off (`:495,506,513,519`, not `:497,505,515,519`). |

## Checked claims: quantisation.md

| # | Claim | Doc loc | Evidence | Result | Note |
|---|---|---|---|---|---|
| 1 | Body int4 g128 = 12,543,596,800 B = 11.68 GiB | l.16, 34 | re-run `.out` §1 line 3 | CORRECT | Measured from headers. "Body" here means linears only; with the GDN/norm bf16 tensors it is 11.73 (`.out` l.72). |
| 2 | Resident 13.85 GiB (D) matches a logged load | l.51 | `docs/gotchas.md:98-103`: "15.13 GiB against 14.26 with `--language-model-only` … non-weight overhead 0.42 against 0.41" at 150k/fp8 (l.105), so `SPEC=mtp CTX=long`, fast variant | CORRECT | 14.26 − 0.41 = 13.85 = header sum 13.849. A good cross-check. |
| 3 | Resident 14.74 GiB (B, C, D′, E) "matches a measured load" | l.23, 53, 107 | `docs/third-party-checkpoints.md:195-205` is a log of the **Uncensored** checkpoint. "14.71" is prose. | UNSUPPORTED | No load of B is logged. At D the log includes 0.41 GiB of non-weight overhead, so 15.71 − 0.97 is not a pure weight figure. The 14.737 header sum (E) is fine. The rows are relabelled. |
| 4 | Target drops `mtp.*` under dflash2 | l.53 | `~/vllm/vllm/model_executor/models/qwen3_5.py:328,478` (`"mtp.": None`) | CORRECT | |
| 5 | KV formats: bf16 B/C; fp8 A, D, **D′ default**; int8 PTH as the D′ alternative; KVarN E | l.20, 77-78 | `start_qwen.sh:210-224` (fast bf16, huge KVarN, long fp8); `resolve_config.sh:95-116,130-131` (PTH_FA defaults to 1 on dflash2/long and suppresses FA2_FP8KV); `Dockerfile:32-33` (pthfa built into the image); `resolve_config.sh:151` (batch fp8) | **WRONG for D′** | In the image on 8.6 with the measured launch, D′ = int8 per-token-head on CUSTOM. fp8 + FA2 is the fallback. A and D: fp8 is correct (D: `start_qwen.sh:223,297`). Fixed inline. |
| 6 | fp8 vs int8 drift 0.036 vs 0.021 nats; fails the non-inferiority test | l.21, 77, 132 | `docs/long-context.md:466-476`: FA2 fp8 0.0360/0.0338, TRITON int8 0.0219/0.0194, PTH int8 0.0207/0.0193. The test is defined in prose at `:472-475` (margin max(10%, 0.002 nats), block bootstrap). fp8 "fails against both int8 arms, by +0.014 to +0.015 nats, CIs excluding 0". | CORRECT | The ratio is 1.64× at 32k and 1.74× at 128k. The test is a documented protocol, not a script in the repo. |
| 7 | Free ≈0.9 GiB at B | l.25, 113 | `third-party-checkpoints.md:204` (926 MiB free at +0.97 GiB, so ≈0.94) | CORRECT but narrow | Two logs the doc missed: 787.5 MiB free with the 0.858 GiB tower (`start_qwen.sh:920`) gives ≈0.69. A headless boot with the tower at 440 MiB free (`gotchas.md:134-137`) gives ≈1.3. Range 0.7-1.3. Fixed inline. |
| 8 | ≈0.15 GiB spare at E | l.25, 144 | `gotchas.md:533-537`: stock 576 MiB free; +146 MiB ran, +196 MiB died, at `MAX_SEQS=8` | CORRECT | Measured at 8 seats; the default is 2. |
| 9 | 1 GiB ≈ 13.4k tokens at B, 26k at D′, 55k at E | l.26, 141-144 | `.out` §6 = KV_MEM ÷ logged pool. The logged pool is concurrency × MAX_LEN (`~/vllm/vllm/v1/core/kv_cache_utils.py:1074-1083`), so it amortises state. | CORRECT in logged units | The context one request actually loses is 16,384 (B) and ≈31.8k (D′ int8; measured 382 → 308 pages at −1 GiB, `bench/test_resident_print.sh:80-81`). B's default pool is 68,605 (prefix cache on), so 13,193/GiB. Noted inline. |
| 10 | KV_MEM pins | l.108 | `start_qwen.sh:495` (E), `:506` (D′), `:513` (C), `:519` (B) | CORRECT (cites ±2 lines) | |
| 11 | Page geometry: block 448/864/880; state 54 pages = 0.738 GiB | l.90-95 | `start_qwen.sh:610-614`; `bench/test_resident_print.sh:83,96,99` | CORRECT | Re-run matches all three. |
| 12 | Vision tower 0.858 GiB, not loaded by default | l.40 | `.out` l.6; `start_qwen.sh:899-931`; `gotchas.md:96-104` | CORRECT | |
| 13 | lm_head int8 fp16 scales; embed bf16 scales | l.36-37 | `prepare/quant_lm_head.py:55-57`; `prepare/quant_embed.py:57-58` | CORRECT | The embed cite `:34` should be `:57-58`. |
| 14 | Fast shards hardlinked | l.34 | `prepare/fetch_fast_variant.py:5,35` | CORRECT | The cite `:46-52` is the copy of shard 7 and the extras. |
| 15 | Negative-scale fold: 50.4%, 0.069% energy | l.42 | DeepInfer `gpu-results/quant-2-check.md:26,35` | CORRECT | |
| 16 | Marlin min capability 75; FA2 refuses fp8 below sm90; Triton fp8 needs sm89; fp8 scale 1.0 | l.34, 71, 77 | `~/vllm/.../marlin.py:30-33`, `fa_utils.py:306-312`, `triton_attn.py:512-514`, `attention.py:136-152` | CORRECT | These are against ~/vllm 0.32-dev, not the project's pinned 0.31. |
| 17 | fp16 GDN state PPL 8.044 vs 8.045 | l.81, 129 | `docs/quality.md:32-33` | CORRECT (cite `:30-31` is the header) | |
| 18 | DeepInfer all-layer +4.00% | l.128 | DeepInfer `evals-3.md:22,28` | CORRECT, but mis-paired | That run is single-user all-layer (docs +4.1%), not batch +3.7%. |
| 19 | D pool ≈7.6 GiB | l.108 | 204,336 is from the EMBED_UVA A/B (`single-user/README.md:443`). The doc's own D pool is 183,673-184,438 (`gotchas.md:105`). | CORRECT as a range | ≈6.9-7.6. Noted inline. |
| 20 | Open Q3: D defaults to `PTH_FA=0` in this checkout | l.160 | `resolve_config.sh:98-103` has only `dflash2/long`; `start_qwen.sh:297` | CORRECT | Consistent with the upstream-diff finding: b4b97ec (#345) is in `sha-2cf50a0` (`ENVIRONMENT.md:24`) and not in 95ac251 (`inventory.md:9`). The best cite is `baseline.md:23`, not `:65`. |

## Checked claims: heads-drafters.md

| # | Claim | Doc loc | Evidence | Result | Note |
|---|---|---|---|---|---|
| 1 | MTP 424,699,392 params; 0.305 GiB int4 incl. 40,960-row head | l.14, 38-39 | `.out` l.45, 76 | CORRECT | |
| 2 | DFlash2 1,924,404,480 params; W4A16 1,280,617,536 B = 1.193 GiB | l.15, 51-53 | `.out` l.42, 56 | CORRECT | |
| 3 | Taps 5/19/33/47/61; selector rank 256, top-k 16; block 8; window 2048; 32/8 heads × 128 | l.49-50 | re-fetched `config.json` @015e7956 | CORRECT | |
| 4 | Selector rank 256 vs DeepInfer ~384 vs the DFlash lane's 16×16 | l.57 | `dflash-specdec.md:106-108`; DeepInfer `drafter.md:38` ("inferred") | CORRECT; reconciled | 256 is the bilinear rank (codebook width: 63,569,920 / 248,320 = 256). 16×16 is top-k × top-k edge scores between adjacent slots. DeepInfer's 384 is wrong. Note added. |
| 5 | Acceptance 3.14/3.34 (DFlash2) vs MTP 2.8/2.9-3.0 "MTP k=3, D's head", C1, thinking off | l.17-22, 81 | `docs/optimizations.md:390-396`: the shipped-MTP row builds on the `CTX=fast` k=4 row (split-KV verify is `CTX=fast` only, `start_qwen.sh:198-199`; k=4 default `:212`). The k=3 `CTX=long` row is `:391` (2.5/2.4). | **WRONG (mode/k)** | The numbers are right, but they are MTP k=4 at `CTX=fast`. Mode D measured 2.64 greedy (DeepInfer `baseline.md:42`). "Default" means the model's default sampling (probabilistic); greedy is T=0. Fixed inline. |
| 6 | Thinking on costs B ≈15% | l.24, 102 | DeepInfer `evals-4.md:30` (117.6 → 100.2, −14.8%, 3.49 → 3.14) | CORRECT | |
| 7 | spec-5: T=0 3.486, T=0.7 3.611 / 3.341 | l.91-93 | DeepInfer `spec-5.md:32-34,40,44` | CORRECT | |
| 8 | DFlash2 Adam ≈31 GB | l.28, 140 | 1.92B × 16 B = 30.8 GB; DeepInfer `drafter.md:54` | CORRECT (E) | DeepInfer adds the frozen embed and lm_head on top. |
| 9 | MTP fine-tune ≈6.8 GB; negative result 0.685 → 0.685 | l.137-138 | 425M × 16 B; `drafter/README.md:29-37` | CORRECT | |
| 10 | int4 DFlash2: greedy kept, −5% sampled, 31.4 → 28 ms | l.152 | `drafter/README.md:89-93`; `docs/optimizations.md:236-238` | CORRECT | |
| 11 | DFlash2 drafter KV 20,480 B/token, ≤40 MiB/request | l.54 | config arithmetic | CORRECT (E) | The page-padded cost is larger (`max(tgt,dft)` pages); that is open Q5. |
| 12 | N=8: D 423 vs B 236 | l.119 | DeepInfer `evals-10.md:29,38` | CORRECT | |
| 13 | ≥1.4 ms read floor at 936 GB/s | l.160 | 1.28 GB / 936 GB/s = 1.37 ms | CORRECT (E) | |
| 14 | `evals-4.md:9` labels D "pthfa default" | l.105 | `evals-4.md:7,9`; image = `sha-2cf50a0` (`ENVIRONMENT.md:24`) | CORRECT | |

Totals: 34 claims checked. 2 are WRONG (D′ KV default; MTP row mode/k), 1 is UNSUPPORTED (B load match), and the rest are CORRECT, with minor cite drift on 6 of them.

## Recomputed memory budget (GiB)

Weights are header sums (E). The pins are measured launcher constants. Total = 23.54 GiB as CUDA reports it (`gotchas.md:521-522`: total 25,272,516,608 B), not 24.

| | B | D′ (PTH int8, default) | E |
|---|---|---|---|
| Weights | 14.74 | 14.74 | 14.74 |
| KV pin | 5.20 | 5.20 | 4.90 |
| Graphs | 1.2-1.45 | 1.2-1.45 | ~1.45 |
| Verify buffer | 0.94 (M) | not measured (CUSTOM) | n/a |
| Free after boot | 0.7-1.3 (E) | not measured | 0.56-0.58 (M, 8 seats) |
| Residual | ≈0-0.75 | — | ≈1.9 |

Exchange rates, context tokens per GiB of KV_MEM:

| Mode | Page geometry (E) | Logged-pool units (doc) |
|---|---|---|
| B bf16 | 16,384 | 13,193-13,415 |
| D′ int8 | ≈31.8k (measured −74 pages per GiB) | 26,236 |
| D′ fp8 | 32,768 | 27,097 |
| E KVarN | ≈80k (unverified geometry) | 54,728 |

GDN state is a fixed 54 pages (0.74 GiB) per resident request at k=7, so it does not change the marginal rate. It does set the cliff: with 1 GiB less pool, one request fits only ≈56k at B (MAX_LEN 64k) and ≈109k at D′ (MAX_LEN 128k).

## Missed items

1. The image builds pthfa (`Dockerfile:32-33`), so D′'s real default is int8 PTH. `README.md:96-101` is stale. The box `.env` sets `VISION=1` (`ENVIRONMENT.md:27`), which silently drops D′ back to fp8 FA2 (`resolve_config.sh:101`).
2. The CUDA-visible total is 23.54 GiB, not 24.
3. Two more free-memory logs at B (787.5 MiB with the tower, desktop; 440 MiB headless with the tower).
4. `bench/test_resident_print.sh:80-81` holds a measured −1 GiB KV_MEM boot (382 → 308 pages). It is the direct evidence for the drafter-vs-KV exchange rate, and the docs did not use it.
5. The MTP acceptance the docs quote is not mode D's. Mode D at C1 is ≈2.64 tok/step, so DFlash2's C1 lead is ≈+27%, not +12%.
6. B's default pool is 68,605 (prefix cache on).

## Corrections made (all marked `[review: …]`)

quantisation.md:
- Summary KV formats bullet: D′ default.
- Summary 14.74 "measured load" bullet.
- Summary 0.9 GiB B-spare bullet: the range.
- Summary exchange-rate bullet: units and real rates.
- Resident table B/C/D′/E cross-check.
- B/C pool cell: 68,605 vs 69,758.
- D′ rows: "default" and "fallback" labels swapped.
- Budget intro: 23.54 GiB.
- Budget "loaded" row: relabelled E.
- D KV pool cell: range 6.9-7.6.
- Open Q2: the VISION gate.
- Open Q3: answered.
- New "Reviewer additions" section.

heads-drafters.md:
- Summary acceptance table MTP row: k=4 `CTX=fast`.
- Measured-acceptance MTP row: the same.
- Selector reconciliation note.
- Memory-cost line: units.
- New "Reviewer additions" section: mode D's 2.64; the image used by the DeepInfer runs.

Not edited (minor cite drift, listed above): `.out` §6 pin lines, `quant_embed.py:34`, `fetch_fast_variant.py:46-52`, `quality.md:30-31`, `baseline.md:65`.
