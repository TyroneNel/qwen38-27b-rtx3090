# Phase-4 review: CPU experiments X-01a, X-15, X-02a, X-08, X-16 and harness

Reviewer: phase-4 reviewer (Opus), 2026-10-21. CPU only, no commits. All re-runs were done from copies in the session
scratchpad, or in place with bytecode and pytest cache turned off (X-08), so the repo gained no files. My outputs were
diffed against the committed `.out` files. X-17-timing was not touched.

| Exp | Verdict | Re-run result |
|---|---|---|
| X-01a | **CONFIRMED** (the phase-3 docs were wrong, not X-01a) | `run.out` byte-identical |
| X-15 | **CONFIRMED WITH CORRECTIONS** (1 cite) | `gdn-state-calc.out` identical |
| X-02a | **CONFIRMED WITH CORRECTIONS** (attribution, sensitivity) | `fit.out` identical; new tail-shape check |
| X-08 | **CONFIRMED WITH CORRECTIONS** (ulp band vs EAGLE-4, caveats) | 14/14 pass |
| X-16 | **CONFIRMED** (open Q1 answered) | 19/4/10 before, 21/4/8 after, identical |
| Harness | **CONFIRMED** | `SELFCHECK PASS`, 18 ok |

## X-01a: PTH_FA default resolution (H01)

**Re-ran.** I copied the experiment to scratch and ran `run.sh` with the ci-cc86 stub `nvidia-smi` and a faked
`pthfa_installed`. The output is byte-identical to `run.out` (`diff` empty).

**Claims checked.**
- `resolve_config.sh:121`: the FA2 fp8 default arm is `SPEC=dflash2 CTX=long` only.
- `resolve_config.sh:105-107`: the upstream mtp arm (FA2_FP8KV empty, DRAFT_TOKENS unset or 3, VISION != 1).
- `single-user/start_qwen.sh:220-223,297`: `PTH_FA=${PTH_FA:-0}`, so an unset launch gets fp8 KV on FlashInfer.
- `run.out:3` (upstream gives PTH_FA=1) and `run.out:75` (ours leaves both unset).
- `04-benchmarks/baseline.md:44`: mode D in the stock image already resolves PTH_FA=1.
- The upstream diff adds only the mtp/long arm. The `start_qwen.sh` diff between 95ac251 and upstream/main is comments
  only, so upstream's `PTH_FA=0` argv is the same as ours with PTH_FA unset.

**Confirmed.** On 95ac251, `SPEC=mtp CTX=long` resolves to PTH_FA and FA2_FP8KV both unset, which means fp8 on
FlashInfer. The brief's expectation (and `transferability.md` X-01a) that ours resolves to `FA2_FP8KV` is wrong. The
X-01a README already said this; the phase-3 docs did not.

**H01 framing.**
- The source of `upstream-diff.md:126` is the PR #345 thread (`gh pr view 345 --repo syv-ai/HyperQwen --comments`).
- That A/B ran image `sha-95ac251` with #345's launchers bind-mounted:
  - arm A = `PTH_FA=0`: fp8 KV on FlashInfer, piecewise cudagraphs only under spec decode;
  - arm B = `PTH_FA=1`: int8 on CUSTOM pth-flash-attn, full graphs.
- Arm A is exactly what our checkout's unset mtp/long launch runs. So the gain does apply to a user of our checkout,
  but its size depends on the box:
  - +21.3% / +22.1% (labd TOTAL at about 66k / 93k): WSL2, 3 boots per arm, `GPU_UTIL=0.89`, because fp8 spilled at
    0.93;
  - +12.2% / +14.0%: native card, 2 boots;
  - +9%: #325, native, 1 boot.
- All three are measured, upstream-reported. Part of the gain is FULL vs PIECEWISE graphs, not the kernel alone. Against
  the stock image the gain is 0.
- plan.md H01 now carries "+12–22% (M, upstream-reported)" with this comparator.

## X-15: drift and GDN state

**Re-ran** `gdn-state-calc.py`. Output identical to `gdn-state-calc.out`.

**Claims checked.**
- `start_qwen.sh:573,608-614`: 6 GDN groups × (2+k) pages, page 14,680,064 B, giving 0.738 GiB at k=7 (estimated).
- `bench/test_resident_print.sh:5,71-72`: measured 54 / 30 / 18 state pages (the k=7/3/1 rows), which is 6 × (2+k).
- vLLM 0.31.0 align-mode Mamba allocation (`single_type_kv_cache_manager.py`) agrees with 2+k.
- `gotchas.md:320` (k+2).
- `.env.example:41`.

**Corrections.**
- The `spec=${spec:-mtp}` default is at `resolve_config.sh:57`, not `:55-56`.
- The 15.8% vs 14.2% gap is still open.
- "2+k slots" is correct; the docs' 1+k is wrong.

## X-02a: k-per-N cost-model fit (H02′)

**Re-ran** `fit.py`. Output identical to `fit.out`.

**Is the substitution legitimate?** Yes.
- a0 = 0.811 comes from the measured k=1 boots.
- The pooled E1/E3/E7 come from boots other than the A/B being reproduced.
- But the back-test is mostly a consistency check. The k=3 vs k=7 gain depends only on E3/E7 and V3/V7, so how the tail
  mass is split over positions 3–6 is never tested.

**Temperature attribution is wrong.** The README blames spec-5's T=0.7. spec-5's own T=0 row
(`04-benchmarks/raw/prior-3090/spec-5.md:32`) gives:
- tok/step 3.486 (ladder: 3.508);
- pos-0 acceptance 0.695 (ladder: 0.81).

So the front-loading comes from prompts or config, not temperature. The ladder itself is greedy
(`bench/conc_ladder.py:120`, `temperature 0.0`).

**Sensitivity.** I wrote `reviews/phase4-x02a-tailshape.py` (output `phase4-x02a-tailshape.out`). It spreads the measured
tail E7−E3 smoothly (a constant conditional acceptance from position 3) instead of using spec-5's cliff, where
a3/a2 = 0.45. Results:
- N=4: the best k is 3 or 4, with a gain of +0.0 / +1.1 / +2.2% at E7 = 3.36 / 3.508 / 3.67. A +5% gain needs
  a3 ≥ 0.404; the smooth tail gives 0.25–0.32.
- N=8: ≈0% when the ms cost per extra k is ≥ 6.8, as at N=4. It reaches +2.5–6.3% only when the slope is 3.7 ms/k,
  which is unmeasured.

The exact "0.0%" is an artefact of the cliff. The conclusion holds: report 0–2% (E), below the 5% bar.

**Comparator.** `[[1,3,7],[4,512,3]]` is the right comparator:
- `start_qwen.sh:678` (the documented example);
- `start_qwen.sh:692` (DYNAMIC_K unset by default; the user sets it);
- vLLM `dynamic/utils.py:52-55` (k ≥ 0 legal, inclusive ranges).

**Recommendation.** Reword H02′; do not drop it. Make X-02b′ an N=3 boundary check (fitted vs DYNAMIC_K, about +5% E)
plus a per-position acceptance capture on the ladder prompts. Drop the N=4/8 win framing. plan.md was updated to match.

## X-08: greedy-divergence classifier (H08, D18)

**Re-ran.** `PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -p no:cacheprovider` in place: 14 passed. The plain-assert run
gives the same result. From a scratch copy, the end-to-end test fails with `ModuleNotFoundError: harness`. The cause is
that the test builds PYTHONPATH relative to its own file location. This is not a defect.

**D18 bands vs the EAGLE-4 source.**
- `EAGLE-4/tools/greedy_parity.py:171` passes gaps ≤ 0.13 as near-ties.
- `tools/fp32_arbiter.py:8-12,40,69-70,85` sends larger gaps to fp32 arbitration. It passes d ≤ 2 ulp, and the ulp
  scales with magnitude (2^(floor(log2|x|)−7), 0.125 on [16, 32)).
- X-08 instead calls any gap > 0.13 `real` with no replay (`classify_divergence.py:69-70`). Its own `judge --tol` is
  0.25 (`:179`).
- So X-08/D18 is stricter than EAGLE-4 on both sides:
  - gaps of 0.01–0.13 go to replay, where EAGLE-4 passes them;
  - gaps of 0.13–0.25 fail outright, where EAGLE-4 arbitrates them.
- The low side is safe. The high side can give false `real` results. Recommendation: make `--ulp-band` default to 0.25
  so it matches `--tol`. The D18 entry in `decisions.md` (not mine to edit) should say "stricter than EAGLE-4" or adopt
  0.25.

**Superset.** `record_top2.py` output is a superset of `greedy_equal.py` output for default flags (same keys plus `top`
and `messages`). With `--passes 2` (`@p2` keys) or `--doc-ctx`, the prompt keys differ, so such files cannot be mixed with
plain `greedy_equal` files in one compare.

**KVarN adaptive + prefix.** The reasoning holds: a deterministic, reproducible, large-gap divergence is flagged `real`.
Two corrections from the source (`start_qwen.sh:399-411`):
- "38/793" means turn 2 tracks the source for about 38 of 793 characters, then diverges. It does not mean 38 wrong
  self-hits; `01-project/dflash-specdec.md:471` misreads it the same way.
- The bug broke at five of six residues, so the GPU arm should record at least 2 document lengths.

`run_suite.sh:27` → `:28` (the `PYTHONPATH` export).

## X-16: off-sm86 import paths

**Re-ran** `probe.py` twice:
- before: on a `git archive` of 95ac251;
- after: in `~/research-wt/x16` with `fallback.patch` applied (uncommitted, left as found).

The counts are 19 OK / 4 SILENT / 10 HARD before and 21 / 4 / 8 after. Both are identical to the committed outputs once
`~` abbreviation and truncation are normalised.

**"One genuine hard-fail" holds** for the probed paths: F5c, the backend import on sm86 with FA2_FP8KV=1 and the `.so`
missing. Caveats:
- The permissive stub hides vLLM-side import failures.
- A wrong-arch `.so` that is present was not probed.

**Policy.** The fallback only makes `register()` decline. Launcher `check()` still refuses before launch
(`start_qwen.sh:244-249`), and stock FLASH_ATTN refuses fp8 on sm86. So the patch is consistent with refuse-by-policy.

**Open Q1 answered.**
- vLLM v0.31.0 `Platform.get_device_capability` returns None (`vllm/platforms/interface.py:433-444`).
- `CpuPlatform` does not override it.
- `DeviceCapability` is a NamedTuple (`:91`), so `in BUILT_FOR` tuple membership works.

## Harness

`selfcheck.sh` re-run: `SELFCHECK PASS`, 18 ok. The smoke gate (`sweep.py` ~:157) and `LC_CELLS`
(`run_suite.sh` ~:31) work as described. The stdev fallback is at `sweep.py:138`.

## Edits made

- `00-meta/plan.md`: H01 row (comparator and +12–22% M note); H02′ row (0–2% E, reword to an N=3 check).
- `02-eagle4/transferability.md`:
  - X-01a row: the FA2_FP8KV claim is wrong; ours is fp8 FlashInfer.
  - Five-most-promising #1: comparator and gain-applies note.
- `04-benchmarks/experiments/X-01a/README.md`: Re-run line.
- `X-15/README.md`: `:57` cite and Re-run line.
- `X-16/README.md`: open Q1 answer and Re-run line.
- `X-02a/README.md`: temperature attribution, back-test caveat, smooth-tail 0–2%, and Re-run line.
- `X-08/README.md`: real band vs EAGLE-4 arbiter, superset caveat, 38/793 and residues, `:28` cite, and Re-run line.
- New: `00-meta/reviews/phase4-x02a-tailshape.py` and `.out`, and this file.
