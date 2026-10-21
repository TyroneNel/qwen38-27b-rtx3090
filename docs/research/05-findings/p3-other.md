# P3 findings: docs, robustness and tooling

Labels as in `p1-speed-inference.md` (D19). Nothing here changes speed. The CPU results below are **measured** for logic or code reading only.

## Summary

| ID | Claim | Expected impact (label) | Effort | Status | Maps to |
|---|---|---|---|---|---|
| F-P3-01 | Six doc-vs-code drift items are confirmed against code | docs only | S | adopt now | H15, I-14, X-15 |
| F-P3-02 | Without a GPU, only one package import hard-fails (F5c); an arch-table fallback is mostly not needed | robustness only; H16 mostly refuted | S (patch exists) | adopt now (low priority) [review: consistent with refuse-by-policy, no decision needed] | H16, E-24, X-16 |
| F-P3-03 | The SPEC default differs between launcher (mtp) and `.env.example` (dflash2) on purpose | none | — | dropped (documented, no fix) | I-14, X-15 item 3 |
| F-P3-04 | Harness and tool defaults lag the reviewed method | prevents false "real" and wrong-temperature rows | S | adopt now | X-08, X-02a, harness gaps |
| F-P3-05 | TTFT and peak VRAM were never recorded for any mode | fills baseline gaps | S per mode (GPU) | needs GPU runs (baseline rows bench-4 to bench-6, bench-11) | baseline gaps |
| F-P3-06 | Upstream hygiene PRs (355-366) need no action from us | none | S each | deferred (take via upstream) | U-03, U-06 |

---

## F-P3-01 Doc drift (X-15)

- **Priority:** P3.
- **Claim:** each item below was checked against code with file:line.
  1. `docs/optimizations.md:303-304` says lookup NSTRONG 8 / AGREE 2. The code defaults to 6 / 0 (`patches/speed-knobs-envs.patch:39-40,78-79`).
  2. `README.md:96-101` says "by default FA2_FP8KV=1". That is wrong: `PTH_FA=1` takes precedence (`resolve_config.sh:57`).
  3. The SPEC default difference is intended (F-P3-03).
  4. GDN state size and slots:
     - 4a. The formula gives 0.738 GiB at k=7 and 1.494 GiB at k=15. The docs say 0.88 / 0.82 / 1.66.
     - 4b. The code uses 2+k slots; the docs say 1+k (`single-user/README.md:103-104`, `docs/long-context.md:694-695`, `docs/optimizations.md:349-350`, `docs/gotchas.md:333`).
     - 4c. MTP state is 0.475 GiB at k=4 and 0.381 at k=3. The docs say 0.44.
  5. `docker-compose.yml:11-12` is stale.
  6. Several other lines still call fp8-on-FA2 the default.
  After the fast-forward (F-P1-01), the PTH_FA wording also needs an mtp/long clause.
- **Expected impact:** docs only.
- **Evidence:**
  - `04-benchmarks/experiments/X-15/README.md`;
  - `gdn-state-calc.out`, re-derived identically;
  - the review verdict "confirmed with corrections": `00-meta/reviews/phase4-cpu-experiments.md:10`.
- **Effort:** S. Text edits in about 8 files.
- **Risks:**
  - Make the edits after the fast-forward, or the PTH wording drifts again.
  - PR 366 also corrects "fp8 default" lines, so check for overlap.
  - Open: 15.8% vs 14.2% (X-15 README).
- **Status:** **adopt now**.
- **Maps to:** H15, I-14, X-15.

## F-P3-02 No-GPU import fallback (X-16)

- **Priority:** P3.
- **Claim:** of 33 import cases, one genuinely hard-fails: F5c, an `OSError` at `load_library` (`fa2 backend.py:27`). The tally moves from 19 / 4 / 10 to 21 / 4 / 8 with `fallback.patch` (3 files, +22/−7). KVarN has no arch table at all.
- **Expected impact:** robustness only, with low production relevance. H16 ("every import hard-fails") is mostly refuted.
- **Evidence:** `04-benchmarks/experiments/X-16/README.md`; `phase4-cpu-experiments.md:13`.
- **Effort:** S. The patch exists.
- **Risks:** policy conflict. The launcher deliberately refuses rather than falls back (`start_qwen.sh:244-246`). A silent fallback could hide a wrong backend in an A/B, the same failure as F-P1-13. [review (phase 5): not a conflict. The patch only makes `register()` decline; the launcher `check()` still refuses before launch (`start_qwen.sh:244-249`), and stock FLASH_ATTN refuses fp8 on sm86 (`00-meta/reviews/phase4-cpu-experiments.md:146-147`). The remaining risk is an in-process caller that bypasses the launcher.]
- **Status:** **adopt now** only if the human chooses "fall back with a loud line" over "refuse". Otherwise **dropped**. [review (phase 5): no policy choice is needed; the patch is compatible with refuse. Adopt as a robustness fix, low priority.]
- **Maps to:** H16, E-24, X-16.

## F-P3-03 SPEC default difference

- **Priority:** P3.
- **Claim:** `single-user/start_qwen.sh:197` defaults to `SPEC=mtp`, and `.env.example:6` sets dflash2. `.env.example:41` documents the difference.
- **Expected impact:** none.
- **Evidence:** `X-15/README.md` item 3.
- **Status:** **dropped**. No fix is needed. Benchmarks must still set `SPEC` explicitly.
- **Maps to:** I-14.

## F-P3-04 Harness and tool defaults

- **Priority:** P3.
- **Claim:** several defaults lag the reviewed method:
  - `classify_divergence.py` / `record_top2.py` `--ulp-band` defaults to 0.13. D18 and the review set it to 0.25 (`classify_divergence.py:4,46`). [review (phase 5): fixed in `classify_divergence.py` (default 0.25, tests pass); `record_top2.py` has no `--ulp-band` argument.]
  - `greedy_equal.py` uses near-tie 0.1 (vs 0.01), records no top-2 ids, runs a single pass and skips `long_context` (`X-08/README.md`).
  - `sweep.py:77` hard-codes temperature 0.
  - There is no Chinese prompt set.
  - There is no V4 quant tool.
  - There is no mtp/long PTH default test on our checkout (`test_pthfa_knob.sh:179`). b4b97ec adds 83 lines of it, so the fast-forward fixes this one.
  - `snapshot.sh` does not record clock drift, although the harness README and `baseline.md:23` say it does. [review (phase 5): only `baseline.md:23` says so (now annotated); the harness README does not.]
- **Expected impact:** no speed. It prevents false "real" verdicts and wrong-temperature rows.
- **Evidence:** `04-benchmarks/gpu-agent-prompt.md` harness gaps 1-15; `phase4-cpu-experiments.md:112-118`. The harness self-check passes (18 ok), **measured** for logic.
- **Effort:** S per item. The zh prompts and the V4 tool are inputs from the human (roadmap).
- **Status:** **adopt now** for the code-only defaults. The queue already passes `--ulp-band 0.25` (`gpu-agent-prompt.md:725-753`), so only manual runs are exposed.
- **Maps to:** X-08, X-02a, harness gaps.

## F-P3-05 TTFT and peak VRAM gaps

- **Priority:** P3.
- **Claim:** no prior 3090 file records TTFT or peak VRAM for any mode. The KV pool comes from the engine log, which is not peak VRAM (`04-benchmarks/baseline.md:29,62,71`).
- **Expected impact:** fills baseline gaps. EAGLE-4's draft prefill adds < 2% TTFT (**estimated**, `02-eagle4/transferability.md`), which cannot be checked without a TTFT baseline.
- **Effort:** S per mode. `sweep.py` records mean TTFT with a salted cold prefix cache.
- **Status:** **needs GPU runs**: the baseline rows bench-4 to bench-6 and bench-11.
- **Maps to:** baseline gaps.

## F-P3-06 Upstream hygiene PRs

- **Priority:** P3.
- **Claim:** PRs 355-358 and 360-365 are CPU-only hygiene: Vision, doctor, template hardening, port tools, offload verdicts, pin-bump validation, `PATCHES.md`.
  - PR 366 re-vendors pthfa v0.1.0 with SASS-identical kernels (author's claim). Note that the prebuilt pthfa has sm_80 SASS, not sm_86, which runs on sm86 (`00-meta/reviews/phase3-transferability.md:160`).
  - PR 355 has the widest launcher footprint. Rebase any EAGLE-4 launcher edit around it, not across it.
- **Expected impact:** none.
- **Evidence:** `01-project/upstream-diff.md:40-53,113`.
- **Status:** **deferred**. Take them through upstream. PRs 353/354 are P1 and covered in F-P1-13; PR 359 is covered in F-P2-12.
- **Maps to:** U-03, U-06.
