# Raising single-user decode tok/s at 150k–240k: ranked, evidence-backed plan

**Scope.** Static analysis + research only. No code, benchmarks, or profilers were run for this
document. Analyzed tree: `main` @ `1cf86656c26b7725743c41a0ad7b9de99f5d7844` (identical to
`upstream/main`, syv-ai/HyperQwen; the fork tip `f6a5436` differs only by a docs/auth merge).

**Branch tracker:** [§0](#0-branch-tracker-sequence-and-progress) holds the sequence and progress
for every architecture candidate (C1–C7) and every T-item on this branch. Update it first.

**Re-verified 2026-10-04 against upstream/main @ `e371b42`** (13 commits after `d5e2a01`).
This pass still runs nothing on a GPU. It adds the first measured E rows at depth, which come
from PR #262, and fixes the claims they overturn. What changed:
- **Per item.**
  - **T1: advanced, not done.** #262 (bd6c5e2, merged 2026-10-04) adds a second opt-in knob,
    `KVARN_FP16_DEQUANT`. It runs the dequant math of the three fused KVarN kernels in fp16,
    and the dots still accumulate in fp32 (`kvarn/README.md:82-98`).
    `kvarn/kvarn-fp16-dequant-0.30.0.patch` registers it in `vllm/envs.py`, so it is part of
    the compile key. With it on, the split-K stage1 and verify kernels autotune only over
    configs that do not spill (`_f16_no_spill_configs`, `triton_kvarn_decode.py:60-77`). #263
    (210db97, `patches/pinned-kv-empty-cache.patch`) empties the allocator cache before the
    pinned KV pool is allocated. `KVARN_SHARED_VERIFY` is still off by default
    (`kvarn-0.30.0.patch:49,78`). The MTP-corruption comment above the gate is unchanged
    (`triton_kvarn_decode.py:995-1002`, gate on line 1003; was 939–946 and 947), and
    `kvarn/README.md:99-101` repeats it.
  - **T2: unchanged.** No commit touches adaptive verification or the spec config
    (`start_qwen.sh:274,500` hold). e7a5823 only adds a sampler JIT warmup at boot.
  - **T3: unchanged in code.** The pins are still `vllm==0.30.0` and
    `flashinfer-cubin==0.6.18.post1` (`docs/install.md:29-30`), so the sm86 graph limit holds.
    #34 is now CLOSED as NOT_PLANNED (2026-10-04T11:27Z). The maintainer closed it "for
    inactivity, not because it is solved" and asks for a reopen if it recurs on 0.30. The k=4
    soak still decides.
  - **T4: unchanged.** #153 is OPEN, with no activity since 2026-09-22. #261 (69036cb) fixes a
    first-request crash of the opt-in `VLLM_DFLASH2_CHAIN=1` path on 0.30. Neither T4 nor T2
    depends on it.
  - **T5: unchanged.** `alternative.sh` gains one line (`resolve_bind_host`, 177ce26), which
    moves its cites down by one. Its "if T1 stalls" premise is weaker now that T1 has
    measured numbers.
- **#262 vs T1.**
  - The two knobs stack. Shared verify cuts the tile visits by QLEN. FP16 halves the register
    cost of the working tiles in all three kernels. The per-token fallback gets FP16 too: it
    launches with the same `common` dict that carries `F16` (`triton_kvarn_decode.py:987,
    1067-1077`).
  - **Register spills, not only the repeated dequant, were the main cost.** At the serving
    shape (100k context, QLEN=8), the best fp32 shared verify config spills 102 registers and
    takes 2.5 ms per call. The fp16 pick spills 4 and takes 0.98 ms [REPO-MEASURED: PR #262
    body, section 3]. So shared verify alone gave only +33% on a fresh boot and +1% on a cached
    boot (31.2 / 24.3 vs 23.5 / 24.0 tok/s) [REPO-MEASURED: PR #240, maintainer A/B, Run A].
  - Autotune picks vary per fresh boot. One fresh boot picked a `maxnreg=96` verify config
    (262 spills, 4.1 ms per call) and decoded at 16.5 tok/s. #262's prune (2ee2c39) removes
    that config.
  - On WSL2, a fresh boot over-committed the card by ~0.8 GiB, and the driver moved ~380 MiB
    of GPU memory to system RAM. 2 of 5 fresh boots then ran at ~25 tok/s. #263 fixes it. The
    PR author expects native Linux to fail the allocation instead, but did not check it.
  - The knobs were only measured together. FP16 alone is unmeasured end to end. The only
    FP16-alone number is a stage1 microbenchmark, 0.88 → 0.50 ms per call (PR #240 thread).
    FP16 alone is the only lever for `SPEC=mtp CTX=huge`, because shared verify still
    corrupts MTP.
- **New field data.**
  - **E at depth** [REPO-MEASURED: PR #262 body, sections 2–3]. One RTX 3090 at 250 W, Docker
    Desktop on WSL2, vLLM 0.30.0, default checkpoint, `CTX=huge SPEC=dflash2 PREFIX_CACHE=1`.
    `bench/labd_bench.py --ctx 94000`: 6 tasks, a ~103k-token prompt, 512 max tokens. Each
    arm has 5 fresh boots with private caches. Every boot keeps the 268,169-token KV pool.
    - `main`: 23.0, 23.0, 23.7, 23.2, 22.5 tok/s (median 23.0), 2.59–2.71 tok/step.
    - #262, knobs off: 23.9, 23.7, 23.3, 23.0, 23.7 tok/s (median 23.7), 2.65–2.73 tok/step.
    - #262 at 2ee2c39 with #263, both knobs on: 52.2, 50.2, 52.0, 51.9, 50.1 tok/s (2.2×).
    - `bench/concurrent_collapse.py`: 0 of 30 trials collapsed on each of the three soaked
      boots.
  - **The #240 author's run** [REPO-MEASURED: PR #240 body]. A ~105k prompt, DFlash2, KVarN
    with an OffloadingConnector tier: 23.4 → 31.6 (shared verify) → 52.9 tok/s (+ FP16),
    2.57–2.59 tok/step. GSM8K n=100 95.0%, PPL en 10.77 / da 10.92, against the huge baseline
    of 10.77 / 10.91 (`docs/vllm-0.29.md:123`). The checkpoint is a different fine-tune, so this
    is not a controlled quality A/B.
  - **Short prompts only (C1); they do not test §2.4.** RTX 5090 at 400 W, E: 212.8 tok/s at
    3.34 tok/step (`docs/reproductions/README.md:73`, #247). L40S at 350 W: E 133.3 at 3.26, D
    106.8 at 2.66 (`:75`, #257). RTX PRO 4000 at 70 W, D: 60.1 (`:76`, #266). 2× RTX 3060
    TP=2 at 170 W, D: 64.5 (`:77`, #255).
  - **#107** (comment of 2026-10-01, vLLM 0.29.0, one 3090, WSL2): removing
    `--enable-prefix-caching` makes the wedge go away. The trigger is ~34k-token agent traffic
    with a partial prefix hit on every turn.
  - **#160 follow-up** (`docs/multi-gpu.md:285-298`): a TP=2 box (3090 + A4000, no P2P) ran
    MTP k=4 on fp8/FlashInfer, on a vLLM 0.29-based recipe, in production for two weeks with
    no repetition or corruption. This is weak support for T3: the TP differs, and the report
    gives no Xid count.
  - **Effect on the baseline** [ESTIMATE]. At ~103.6k (the prompt length the #240 A/B reports
    for this bench), the plan's linear model (73.7 @ 25k,
    38.6 @ 90k, native 3090) predicts ~35 tok/s. The WSL2 harness measured 23.0 with the
    knobs off. Platform, task mix and harness differ, so this is not a clean test of the
    model, but it points one way. Assume a fixed ~20 ms per step (~16 GB of weights, state
    and drafter at 85% of 936 GB/s) and a context term linear in depth. Then 115 ms at 103.6k
    (2.65 tok/step) gives ~0.92 ms per +1k tokens and ~240 ms per step at 240k: **~11 tok/s
    (range ~10–12) on this WSL2 harness**, not ~18. With both knobs on, 52 ms at 103.6k (2.71
    tok/step, from the #240 maintainer arm, because #262 does not state knob-on tok/step)
    gives ~95 ms at 240k: **~28 tok/s**. Native Linux may read higher.
- **Open questions now partly answered.**
  - Q1: DFlash2 k=7 with async on and shared verify on shows no collapse in any 30-trial
    `concurrent_collapse` soak (two knob-on boots in #262, both models in the #240 maintainer
    A/B, three runs by the #240 author), and tok/step does not change (2.71 vs 2.70, #240
    Run A). The MTP side is still untested.
  - Q2: E at ~103k is measured, on WSL2 only. There are no 150k or 240k cells and no D depth
    curve.
  - Q5: the verify kernel is spill-limited. In fp16 it takes ~0.98 ms per call at 100k,
    against ~0.09 ms of byte time (100,000 × 840 B ÷ 936 GB/s), so ~11× headroom remains. 16
    layers × 0.98 ms is ~16 ms of a ~52 ms step [ESTIMATE].
  - Q8: in #107's field report, prefix caching off removes the wedge on 0.29.
  - Q3, Q4, Q6 and Q7 are unanswered.
- **Corrections in this pass.**
  - F1's mechanism is incomplete. Sharing the dequant alone did not give ~2.4×, because the
    fp32 kernel spills. The measured 2.2× needs both knobs. F1 predicted ~5 ms of verify
    attention per step after sharing; the fp16 kernel still takes ~16 ms at 100k [ESTIMATE].
  - "#34 still OPEN" (§1 row 3, F8, T3 step 3, Q4) is false. #208 is also CLOSED
    (2026-09-30, completed); the 2026-09-29 block below still says OPEN.
  - The E baseline at 240k is likely ~11 tok/s on the WSL2 harness, not ~18–25 [ESTIMATE].
  - T1's "same boot" A/B cannot work. The knobs are read at boot, and `KVARN_FP16_DEQUANT` is
    in the compile key. Each arm needs at least 2 (ideally 5) fresh boots.
  - Line-number drift fixed in place: `triton_kvarn_decode.py` (bd6c5e2 edits it directly; no
    `kvarn/*.patch` touches it), `alternative.sh`, `kvarn/README.md`, `docs/multi-gpu.md`,
    `prepare/README.md`. Each fix names the old number. The `start_qwen.sh`,
    `kvarn_attn.py:425-428`, `docs/long-context.md`, `docs/vllm-0.29.md`,
    `docs/vllm-0.30.md:82-91`, `docs/gotchas.md` and `docs/optimizations.md` cites hold.

**Re-verified 2026-09-29 against upstream/main @ `d5e2a01`** (7 commits after `2522ef9`; still
no new measurements — every change below is source-level or a citation fix). What changed:
- **Upstream since `2522ef9`: unchanged for D and E.** None of the seven commits touches the
  D/E decode path. #219 (8cf642e) exports `VLLM_WSL2_ENABLE_PIN_MEMORY=1` only under WSL2
  with `SPEC=dflash2` (`single-user/start_qwen.sh:779-782`); the reference 3090 is native,
  so no repo-measured row moves. #234 (e355f9f) makes `marlin-repack-staged-sm80` opt-in; its
  old default was on for compute capability 8.0 only, never sm86. #232 (36936ec) makes
  `alternative.sh`'s prefix retention default 0 without a KV tier (T5's path; affects prefix
  reuse, not decode). #235 is batch-only; #230, #236 and #231 are docs/install.
  `KVARN_SHARED_VERIFY` is still default-off (`kvarn-0.30.0.patch:49,78`, unmoved by
  e355f9f's re-export), and the comment behind the gate is unchanged
  (`triton_kvarn_decode.py:939-946`, gate on line 947).
- **Errors this pass found that were already false at `2522ef9`, corrected in place below:**
  - `CTX=huge` does **not** run `--no-async-scheduling` by default. The launcher sets
    `ASYNC_SCHED=0` only when the lookup is on **and** `DFLASH_TOKENS>7`
    (`start_qwen.sh:310-316`), else it passes `--async-scheduling` (`:659-660`). So the
    default k=7 DFlash2 profile and every `SPEC=mtp CTX=huge` boot run async. T1's "the
    shipped profile may already be safe" argument is withdrawn (F1, F6, T1, Q1).
  - Adaptive verification (vllm#52228) is opt-in in 0.30.0 (`enable_adaptive_verification:
    bool = False`, `vllm/config/speculative.py:539`). Reading the source (not booted), it
    refuses this model: the backend check covers every KV-cache group, GDN included
    (`vllm/v1/worker/gpu/model_runner.py:621-628`, `adaptive_verification.py:464-497`),
    and SSM backends opt out (`vllm/v1/attention/backend.py:212-224`). T2's headline lever
    needs a patch; a flag alone will not turn it on.
  - FlashInfer on sm86 still reports `UNIFORM_SINGLE_TOKEN_DECODE` in 0.30.0
    (`vllm/v1/attention/backends/flashinfer.py:977-1012`), so D's verify stays PIECEWISE
    whatever the #34 soak shows. T3's graph half is not a soak question.
  - vllm#55095 is a graph-mode fallback for *non-compiled* models, vllm#54646 speeds up
    capture/boot, and vllm#54374 is a correctness fix. None of them raises decode rate for
    this compiled model. Suffix decoding already ships in 0.30.0 (`method="suffix"`, needs
    `arctic-inference`), so T8 needs no backport.
  - The "15.9 GiB loaded" weight figure comes from a `SPEC=dflash2` boot, so it includes the
    1.19 GB drafter and the ~1.27 GB int8 embedding table. Per-step target weights are
    ~14.6 GB [ESTIMATE], and §2.2–2.4 are recomputed: both ceilings are ~121 tok/s, not
    ~107.
  - `int4_per_token_head` *has* a rotation, and its batch-mode PPL (8.257, +0.3%) and 240k
    needle are already published (`docs/long-context.md:88,98-99`). T5's quality-risk text
    is corrected.
  - The 240k extrapolation is ~18 tok/s (linear, independent of tok/step), not ~19–25. The
    F1 per-tile cost is ~0.46 µs, not 0.33.
  - Gotcha 43: 4096 batched tokens boots (−2.4% pool); only 8192 refuses (R12).
  - Issue states: #208 is still OPEN (its fix, #222, merged). #196 has maintainer replies
    (2026-09-27/28: #213 fixed the `.txt` corpus bug, gotcha 61 was added), and the
    full-vs-truncated head A/B is still owed. #107's thread never mentions vllm#50021, and
    the repo has carried that backport since 9850338 (2026-08-17), before #107 was filed.
  - Line-number drift fixed throughout (start_qwen.sh, optimizations.md, vllm-0.29.md,
    long-context.md, gotchas 40/57, multi-gpu.md, kvarn/README.md). Each fix names the old
    number.

**Re-verified 2026-09-28 against upstream/main @ `2522ef9`** (22 commits later; still no new
measurements — the deltas below are source-level). What landed since the analysis:
- **PR #189 merged (d88544b): vLLM is now pinned to 0.30.0** — opportunity 2 / T2 is no
  longer "land the port" but "switch on and validate its spec-decode wins" (updated below).
  The 0.30 image is the shipped image, so Q3–Q4's soak experiments need no special build.
- **#208 fixed via #222** (`kvarn-recycled-pages-0.30.0.patch`; 2026-09-29: #222 MERGED, but
  #208 itself is still OPEN, awaiting reporter confirmation): the "!!!!" collapse — a
  *different* KVarN corruption (late page flush across KV-cache groups onto another
  request's mamba state), with a new GPU reproducer `bench/concurrent_collapse.py`.
  T1's target corruption (shared-verify ↔ async scheduling) is untouched: the comment above
  the gate survives verbatim (`triton_kvarn_decode.py:939-946`, was "941-945" and called the
  docstring) and `KVARN_SHARED_VERIFY` is still
  default-off (`kvarn-0.30.0.patch:49,78`) — but the hunt now has a proven template.
- **Appendix A's "most useful A/B left" is measured**: 2× 3090 TP=2, int8/TRITON vs
  fp8/FlashInfer MTP — 156.0 vs 90.9 tok/s (1.72×, #217; `docs/multi-gpu.md:228`, was ":226").
- `spec-attn-smem-fit` (#188) and `bench-sse-keepalive` (#226) joined the series;
  `offload-mtp-serve` and `mamba-align-retire-null-gaps` retired out. The launchers now
  pin the prefix-cache retention interval explicitly (#189; 0.30's unset default is 0 —
  `vllm/config/cache.py:148` in 0.30.0 — and vllm#55760's dense-for-Mamba+EAGLE resolution
  is *not* in 0.30.0: the launcher says it was on the 0.29 release branch only,
  `start_qwen.sh:542-544`; 2026-09-29 correction of "0.30's unset default is 0,
  vllm#55760") — T2's retention caveat is already handled in-tree.

**Number labels.** Every figure is one of:
- **[REPO-MEASURED]** — quoted from this repo's docs/issues, with the path.
- **[EXTERNAL-MEASURED]** — another project's published result, with link.
- **[ESTIMATE]** — my arithmetic, with the calculation shown. Never a measurement.

**Target.** The brief left `[TARGET] tok/s at [CONTEXT LENGTH]` as a placeholder. This plan is
written against an illustrative target of **≥ 115 tok/s at 150k** and **≥ 60 tok/s at 240k**
(mixed prose, single stream, one 24 GB card), with the copy/quote profile at 240k ≥ 250 tok/s.
Substitute the real target; the ranking does not change.

**Where the repo contradicts the brief, the repo wins.** The brief quotes Setup D as
"~95–100 tok/s at ~150k" and Setup E as "~67 tok/s mixed at ~240k". The repo's own
depth-resolved numbers say those are **short-prompt cohort figures measured on servers
configured for 150k/240k**, not decode rates at that depth:
- D (`SPEC=mtp CTX=long`, fp8): 92.6–101.8 tok/s on the C1 cohort of ~1–2k prompts
  [REPO-MEASURED: docs/vllm-0.29.md lines 334 (92.6, native 3090) and 273 (101.8, WSL2
  4090); was "lines 271, 332"], but **68.1 tok/s at 112k depth**
  [REPO-MEASURED: docs/long-context.md lines 48–56 (value on 54), issue #11] and 80.3 @ 25k /
  70.7 @ 60k [REPO-MEASURED: docs/gotchas.md gotcha 40, lines 831–836 (was "825–830"), vLLM
  0.28.0, salted prompts]. At true 150k depth expect ~60–70 [ESTIMATE: extrapolation of those
  two series].
- E (`SPEC=dflash2 CTX=huge` — `.env.example:6` sets `SPEC=dflash2`, README.md:85 adds
  `CTX=huge` — KVarN + DFlash2): 93.6–107.2 tok/s C1 [REPO-MEASURED: docs/vllm-0.29.md lines
  335 (93.6, native 3090) and 292 (107.2, WSL2 4090); was "lines 290, 333"], **73.7 tok/s at
  25k, 38.6 at 90k** [REPO-MEASURED: docs/vllm-0.29.md lines 126–127], 32.0 at 112k (MTP-3,
  not DFlash2) [REPO-MEASURED: docs/long-context.md line 54; was "line 53"]. The "67 mixed"
  is the six-task mix at `--ctx 20000` (3.15 tok/step) [REPO-MEASURED: docs/long-context.md
  line 220], a moderate-depth number. At true 240k depth the same curve reads **~18 tok/s**
  [ESTIMATE, recomputed 2026-09-29: at 2.4 tok/step, 25k = 32.6 ms and 90k = 62.2 ms per
  step, so the slope is 0.456 ms/step per +1k tokens (was "~0.43"), giving ~131 ms/step at
  240k (was "~127") and 18.4 tok/s. A linear extrapolation gives 18.4 at *any* constant
  tok/step, so the old "~19–25" upper end only holds if the slope flattens with depth]. The
  "164 while quoting" figure (README.md:85) is the *copy* task at 20k (164 tok/s, 7.83
  tok/step; the `quote` task reads 58) [REPO-MEASURED: docs/long-context.md lines 217–219],
  so it too depends on depth and task.

So the honest baseline is: **~65–75 tok/s at true 150k depth (D), ~18–25 tok/s at true 240k
depth (E)**, and the ceiling analysis below is done against those, with the short-context
numbers noted where they are what the repo publishes. **2026-10-04:** E now has a measured
point at depth: 23.0 tok/s (median of 5 fresh boots) at a ~103k-token prompt, on a 3090 at
250 W under WSL2 [REPO-MEASURED: PR #262 body]. Extrapolated from that point, E at 240k is
~11 tok/s (range ~10–12) on that harness, not ~18–25 [ESTIMATE; the calculation and the WSL2
caveat are in the top block]. D still has no measured point past 112k.

---

## 0. Branch tracker: sequence and progress

**Updated 2026-10-04 against upstream/main @ `e371b42`.** The C1, A1, A2, A3, A4, A5, A6 and A7 rows and D1 were updated
on 2026-10-05, at `10bb488`, and D3 was added and opened. On 2026-10-06, A4 was reworked, and A1, A2, A3,
D1 and D3 merged (upstream/main @ `7af097b`). On 2026-10-08, A4 and A5 merged (upstream/main @ `9133015`).
This section tracks the whole
`docs/decode-perf-150k-240k-plan` branch. It covers two tracks:

- **Track A** is the seven architecture candidates in
  [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html). It needs no GPU.
- **Track B** is this plan's T-items. It needs an RTX 3090 and a native Linux host.

The two tracks do not block each other. Each row names the doc that holds the evidence.

**Progress at `e371b42`:**

- Track A: 1 of 7 candidates is merged (C1). C6 is partly done.
- (2026-10-05, `10bb488`) Track A: five PRs are in review. They are C5 PR B (#271), C2 PR A and
  PR B (#272, #273), and C6 PR B and PR A (#274, #275). All of C6's planned PRs are open.
- (2026-10-05, `10bb488`) Defect D1 is in review as #276.
- (2026-10-05, `10bb488`) C5 PR B's second part is in review as #277. It runs `test_prepare_crash.py` in CI.
- (2026-10-05, `10bb488`) C5 PR B's third part is in review as #278. It runs `mq3d_scratch_pool_test.py` in the
  image build. All of PR B is in review.
- (2026-10-05, `10bb488`) C3's `_created` guards are dropped. The bash pattern cannot match a `_created` line.
- (2026-10-05, `10bb488`) C5 PR A is in review as #279. All of C5's planned PRs are open.
- (2026-10-06, `7af097b`) Track A: C5 PR A and PR B (#279; #271, #277, #278) and C2 PR A and PR B (#272, #273)
  are merged. All of C5's planned PRs are merged. Defects D1 (#276) and D3 (#280) are merged. Two PRs are in
  review: C6 PR B and PR A (#274, #275).
- (2026-10-08, `9133015`) Track A: C6 PR B and PR A (#274, #275) are merged. All of C6's planned PRs are merged,
  and no Track A PR is in review. C1, C5 and C6 are merged in full. C2 has PR A and PR B merged.
- Track B: T1 is measured but is not the default yet. No T-item ships by default.

### 0.1 Track A: architecture candidates (no GPU)

| Order | Item | Status @ `e371b42` | Next action | Depends on | Doc |
|---|---|---|---|---|---|
| done | **C1** One door into the patch series | **Merged** as e1459c7 (#242/#243/#244, 2026-09-30). Done-when 6 of 6 pass. | None. The one item left was: an `apply.sh` check that the `KVARN` array agrees with `kvarn/*.patch`. It ships with A4. (2026-10-05) It is in #274 (A4), in review. (2026-10-08) It merged with #274 as `76e1a15`. | — | [patch-series-apply-remediation.md](patch-series-apply-remediation.md) |
| A1 | **C5** PR B: wire the CPU tests into CI | **Merged** 2026-10-06: syv-ai/HyperQwen#271 as `4fad697`, #277 as `a79dbc7`, #278 as `3334c8b`. #271 (opened 2026-10-04 on `10bb488`) adds `test_no_key_bind.sh`, `mq3d_capacity_property.py` (with `--mutate seq-rows`) and `verbatim.py` to the `model-verification` job. CI on #271 runs all four green; the bind test does not skip on the runner. #277 (`77b4695`, opened 2026-10-05 on `10bb488`, CI green) adds a `prepare-crash` job: CPU torch, the image's pins for the four prepare libraries, `safetensors` and `psutil`, then `test_prepare_crash.py`. The job takes 1m35s on the runner, 47 s of it the test. #278 (`389e607`, opened 2026-10-05 on `10bb488`, CI green) runs `mq3d_scratch_pool_test.py` in the image build, at the end of the Dockerfile `RUN` that runs `verify.sh --install`. It adds about 8 s to build-push (5m21s on the runner). Without the patch's exclusive guard, the #278 build fails at the test, and main's Dockerfile builds green. | None. All of PR B is merged (#271, #277, #278). | C1 (done) | [harness-verdicts-ci-remediation.md](harness-verdicts-ci-remediation.md) |
| A2 | **C5** PR A: the exit-code convention | **Merged** 2026-10-06 as `94b9dd4`: syv-ai/HyperQwen#279 (`7216e19`, opened 2026-10-05 on `10bb488`, CI green). Eleven `bench/` scripts printed a failing verdict and exited 0, not the six the plan named. Ten now exit 1 on that verdict, and `seat_ttft.py` exits 2 when every request fails. `api_smoke.py` exits 2 when no request reaches a server, and `prefix_alternation.py` exits 2 when it checked no turn. `bench/README.md` writes down the exit convention (0, 1, 2, 3) and has 45 rows: the 44 entries in `bench/` and the kvarn test. Both verdicts files stay, because they are a 4090 run and a 3090 rerun, not a copy. The oracle still writes beside itself by default. Checked on a host with no GPU, on the branch and on main: 20 stub-server cases, 3 SSE cases in the published image, and 6 fake-kernel cases for the two spec-decode tests. The two exit-2 cases got 10 more stub cases. | None. PR A is merged. | — (#279 does not need A1) | same |
| A3 | **C2** Launcher chassis | **PR A and PR B merged** 2026-10-06: #272 as `631748f`, #273 as `53557bc` (opened 2026-10-04 on `10bb488`, CI green). PR A is syv-ai/HyperQwen#272: `launcher_common.sh` with `resolve_bind_host` (moved from `resolve_api_key.sh`) and `qwen_exec`, the `PRINT_ARGV=1` dry run. A normal boot keeps the same argv and environment. PR B is #273, stacked on #272: one INT8 export guard (drift item 1). It changes batch and `bench/prefill_ab.sh` only when `INT8_ACT` is empty and `INT8_LAYERS` is not: `VLLM_MARLIN_INT8_INCLUDE_RE` is then no longer exported, so that boot probably compiles cold once (not measured). (2026-10-05) On #272's first commit, a failed source of `resolve_api_key.sh` boots keyless. #272's second commit, `75d285e`, makes both `start_qwen.sh` refuse to boot, as `alternative.sh` does. #273 merges it in at `7227b0b`. `test_no_key_bind.sh` gains one refusal row per launcher: 25 PASS on #272 and 34 on #273. Without the guard, both `start_qwen.sh` boot keyless on `127.0.0.1`, and the test exits 1. CI green on both. | Move the blocks that both `start_qwen.sh` copy, one at a time (#272 left them in place). Then the plan's §3.2 (`alternative.sh` joins the chassis) and §3.3 (`bench/test_launcher_argv.py` in CI). | A1: the `test_no_key_bind.sh` rows become the `PRINT_ARGV` matrix | [launcher-chassis-remediation.md](launcher-chassis-remediation.md) |
| A4 | **C6** PR B: fork and upstream references | **Merged** 2026-10-08 as `76e1a15`: syv-ai/HyperQwen#274 (`91b0321`, opened 2026-10-04 on `10bb488`, reworked 2026-10-06, CI green). 12 of the 50 marker hashes were not on cut5, the tag `:12` named. The first revision named the fork ref behind each hash and re-exported two files from cut9. cpuchip proposed instead that the patch files are the source and no fork is named, so the rework drops the ref list and the re-exports. `export-patch.sh` writes `--- exported from <hash> (<topic>) ...`, and `PATCHES.md:12` and `kvarn/README.md` name no fork. One commit removes `cpuchip/vllm ` from all 50 markers: the series check passes, and the installed tree is the same git tree (`35a640355e`) before and after. The upstream cells cite #59892, #59888, #59890 (root path only, not its basename), #59893 and #59889 (merged as `7867d6c52d`; retires when the pin carries it). `tokenize-v1-route` stays `feature`, not `local`, because `local` means hardware or environment. Its upstream cell is "none" and its retires-when is "stays". It fixes the end note (six files have a preamble of one line or none) and `export-patch.sh:4`. `apply.sh --kvarn` exits 2 when `KVARN` and `kvarn/*.patch` disagree. | None. PR B is merged. No fork tag is needed: cpuchip closed cpuchip/vllm#3 and #4, and their commits are in his `qwen38/0.30` cut10 (`626f7da75`). | — (any time) | [patch-index-generation-remediation.md](patch-index-generation-remediation.md) |
| A5 | **C6** PR A: the index generator | **Merged** 2026-10-08 as `c5a6e78`: syv-ai/HyperQwen#275 (`17b1551`, stacked on #274, opened 2026-10-05, CI green with the new `patch-index` job). After #274 merged, @mhenrichsen rebased #275 onto main himself (`560dd47`). He kept #275's headers, #274's marker form (no repo name) and #274's `PATCHES.md` prose, added a paragraph for the table, dropped the "Where the commits are" paragraph and regenerated the table. That settles the preamble question: `PATCHES.md:20-21` says to change a row, change the headers in the patch file and run the script. The table now has 51 rows, with #260's `api-root-health`. Each patch file ends its preamble with five headers: `Kind`, `What`, `Upstream`, `Cut-against` and `Retires-when`. The plan had three, but only 4 of 50 first paragraphs matched the "what" cell. `scripts/patches_md.py` writes the table in apply order, and the `patch-index` job fails when it is not current. `apply.sh --list --kvarn` is new. All 50 files are re-exported from tag `qwen38/0.30-index-cut1` on TyroneNel/vllm (`7c013fc0e`). The hunk bodies are unchanged, the installed tree is byte-identical, and the 7 hunks that applied at an offset now apply exactly. | Follow-ups, each in its own PR, all off main `9133015`, local and not pushed yet (2026-10-08). (1) `fix/stale-cut-against` `5466da6`: the `Cut-against:` headers of `auth-deny-default` and `spec-attn-smem-fit` claimed offsets the series no longer has; both now say `0.30.0`, the hunks are unchanged. (2) `fix/export-keeps-preamble` `09dd6c2`: a re-export into an existing file keeps the preamble above the marker, so `PATCHES.md:15` and `export-patch.sh:4` agree with `PATCHES.md:20-21`. `scripts/test_export_patch.sh` is a round trip in a throwaway repo, run by the `patch-index` job; it fails against main's `export-patch.sh`. (3) `fix/verify-kvarn-list` `aa87c98`: the KVarN block in `verify.sh` reads `apply.sh --list --kvarn` and FAILs when the list and `kvarn/` disagree. The three merge together cleanly. Reviewed 2026-10-08 for completeness, simplicity and blast radius. Fixes: (1) `cd1c3d4` drops the `compile-key-runtime-knobs` line that claimed a hand cut against 0.29.0. (2) `79dfbb3` corrects `patches_md.py`'s docstring and `kvarn/README.md`, which still said to re-export to change a header, and shortens the `export-patch.sh` comment. (3) needs no change. Merge (2) before or with (1): on main, a re-export would put back the `Cut-against:` text that (1) removes. (4) The `Verify:` header is dropped. Proposed instead: a CI check that each `graded()` triple in `verify.sh` still names a patch in the series, a file in that patch and a marker in its added lines. Not started; waiting on the owner. | A4 | same |
| A6 | **C3** Harness client | Not started | Start with PR A. Do not ship the two `_created` guards: prometheus_client writes `<name>_created`, which the bash `spec()` pattern cannot match, because it needs `_total` right after the name. Measured on a stub with prometheus_client 0.18.0 and 0.26.0. | — | [harness-client-remediation.md](harness-client-remediation.md) |
| A7 | **C4** Pipeline core | Not started | Build on `prepare/quant_schema.py` (#246), not on a new module. Do not add `load_config(d, requires=...)`. After D1 (#276), every `prepare/` script clones `group_0`, and `load_config` already checks it. | — | [pipeline-core-remediation.md](pipeline-core-remediation.md) |
| A8 | **C7** `prepare --status` | Not started | PR A: move the model checks to `scripts/verify_model.py`. Give `--status` exit codes 0, 1 and 2, and run it before `load_config`. | Upstream PR #267, merged 2026-10-08 as `bb3c579` (7 lines in `verify.sh`). A7 for the shared names. | [prepare-status-remediation.md](prepare-status-remediation.md) |

**Small defects that this review found.** Fix each one in its own PR, at any time.

| ID | Defect | Status | Fix | Doc |
|---|---|---|---|---|
| D1 | `prepare/quant_embed.py:85-86` writes the shard, then `:89` reads `group_1`. `load_config` checks only `group_0` (`quant_schema.py:65`). | **Merged** 2026-10-06 as `7796b64`: syv-ai/HyperQwen#276 (`f56f0d5`, opened 2026-10-05 on `10bb488`, CI green). | `quant_embed.py` clones `group_0`, as `quant_mtp.py` and `quant_heads_stream.py` do. The planned `group_1` check was not needed: `group_1` is `group_0` with four fields set, and `quant_embed.py` sets the same four. Run first, `quant_embed.py` now completes, and the prepared fixture is byte-identical (34 files). `test_prepare_crash.py` gains an `order` step, which also requires `group_2` to equal `group_1` except for `targets`: 169 cases, 0 failures. | [pipeline-core-remediation.md](pipeline-core-remediation.md) |
| D2 | `patches/auth-deny-default.patch:118` uses `removeprefix(root_path)`. With `--root-path /`, `/health` gets 401. A reviewer found the same defect in vllm-project/vllm#59892. | Open | Use `get_route_path`. | [patch-index-generation-remediation.md](patch-index-generation-remediation.md) |
| D3 | `bench/real_rep.sh:23` writes `/tmp/rr_$TAG_$i.log`. Bash reads `$TAG_`, a variable that is not set, so every tag writes `/tmp/rr_<i>.log`. A later run overwrites the logs of an earlier run. | **Merged** 2026-10-06 as `95180ec`: syv-ai/HyperQwen#280 (`ccd14bb`, opened 2026-10-05 on `10bb488`, CI green). Found during A2. | `${TAG}_$i` in the five places (`:23`, `:25`). With a stub `vllm` and two tags, main leaves `rr_1.log` and `rr_2.log`, both with the second tag's output. The branch leaves four logs, one for each tag and repeat. The REP lines are the same on both. No other `*.sh` file on main reads past a name this way (a scan of the 20 files). | [harness-verdicts-ci-remediation.md](harness-verdicts-ci-remediation.md) |

### 0.2 Track B: decode performance (GPU)

The recommended order (2026-10-04) is **T1, T3, T4, T5, T2**. The T labels keep their original numbers.

| Order | Item | Status @ `e371b42` | Next action | Depends on |
|---|---|---|---|---|
| B1 | **T1** KVarN shared verify + fp16 dequant | Moved forward. #262 and #263 merged. `KVARN_SHARED_VERIFY` and `KVARN_FP16_DEQUANT` are both off by default. Measured at about 103k (3090, 250 W, WSL2), median tok/s: main 23.0, knobs off 23.7, knobs on 51.9 (runs 50–52). The 240k pass bar (≥45 tok/s) looks out of reach: about 28 tok/s [ESTIMATE], against a baseline of about 11 [ESTIMATE]. See T1 "Status 2026-10-04". | Do a native Linux run with 150k and 240k cells, plus PPL and needle checks. Measure fp16 alone for MTP. Then turn both knobs on for `SPEC=dflash2 CTX=huge`. | GPU host |
| B2 | **T3** MTP k=4 at `CTX=long` | Unchanged. Issue #34 is closed as not planned. | Run the k=4 soak and the #121 check on 0.30. | GPU host |
| B3 | **T4** DFlash2 + fp8 at 150k | Blocked | Wait for the #153 adapter. #261 lets n-gram chains stack on top of it. | #153 |
| B4 | **T5** int4 KV hedge (lossy) | Unchanged. The measured T1 result makes it weaker. | Do it only if T1 stalls. | B1 result |
| B5 | **T2** Adaptive verification | Unchanged. It gives about 0 gain as shipped. | It needs a GDN and backend patch first. | — |

### 0.3 Progress log

| Date | Upstream @ | Event |
|---|---|---|
| 2026-09-25 | `1cf8665` | This plan was written (static analysis only). |
| 2026-09-26 | — | The architecture review named seven candidates, and one remediation plan was written for each. |
| 2026-09-28 | `2522ef9` | All plans re-verified: 22 commits, including the vLLM 0.30.0 flip (#189). |
| 2026-09-29 | `d5e2a01` | All plans re-verified: 7 commits. |
| 2026-09-30 | `d5e2a01` | C1 was implemented as #242/#243/#244 and merged upstream as e1459c7. |
| 2026-10-04 | `e371b42` | All plans re-verified: 13 commits. #262 and #263 merged, and T1 was measured at 2.2× at about 103k. #237 fixed launcher drift item 4. #246 added `prepare/quant_schema.py`. This tracker was added. |
| 2026-10-04 | `10bb488` | A1 opened as syv-ai/HyperQwen#271 (four CPU tests in `model-verification`; CI green). |
| 2026-10-04 | `10bb488` | vllm #59889 (`serve-404-served-names`) merged into vLLM main as `7867d6c52d`; not in a release yet. A4 updated. |
| 2026-10-04 | `10bb488` | A3 opened: C2 PR A as #272, and PR B as #273, stacked on #272. CI green. |
| 2026-10-04 | `10bb488` | A4 opened as #274. 12 of the 50 marker hashes were not on cut5. CI green. |
| 2026-10-05 | `10bb488` | A5 opened as #275, stacked on #274. All 50 patch files are re-exported from TyroneNel/vllm tag `qwen38/0.30-index-cut1` (`7c013fc0e`). CI green, with the new `patch-index` job. |
| 2026-10-05 | `10bb488` | D1 opened as #276. `quant_embed.py` clones `group_0`, not `group_1`, so the C4 plan's `requires=` check is not needed. The C4 and C7 docs record it. CI green. |
| 2026-10-05 | `10bb488` | A6's `_created` guards dropped, with no PR. The defect in C3 1.5.1 does not exist. The C3 doc records the measurement. |
| 2026-10-05 | `10bb488` | C5 PR B part 2 opened as #277. compressed-tensors 0.17.0 needs `psutil` and does not declare it. The job takes 1m35s on the runner, 47 s of it the test. CI green. |
| 2026-10-05 | `10bb488` | C5 PR B part 3 opened as #278. `mq3d_scratch_pool_test.py` runs in the image build, after `verify.sh --install`. A patch without the exclusive guard fails the #278 build, and main's Dockerfile builds it green. CI green. |
| 2026-10-05 | `10bb488` | A2 opened as #279. Eleven `bench/` scripts printed a failing verdict and exited 0, not six. The two verdicts files are two runs, not a copy, so both stay. D3 (`real_rep.sh` log names) was found on the way. CI green. |
| 2026-10-05 | `10bb488` | A3: #272's `75d285e` makes both `start_qwen.sh` refuse to boot when `resolve_api_key.sh` cannot be sourced, and #273 merges it in at `7227b0b`. The A3 row and the C2 doc had it as still to do. CI green. |
| 2026-10-05 | `10bb488` | D3 opened as #280. `real_rep.sh` writes `/tmp/rr_${TAG}_$i.log`, so each tag keeps its own logs. CI green. |
| 2026-10-05 | `10bb488` | A2: #279's `7216e19` fixes its two gaps. `api_smoke.py` exits 2 when no request reaches a server, and `prefix_alternation.py` exits 2 when it checked no turn. CI green. |
| 2026-10-06 | `10bb488` | A4 reworked as #274 `91b0321`, as cpuchip proposed: the patch files are the source, and no fork is named. All 50 markers drop `cpuchip/vllm`, with the same installed tree. cpuchip closed cpuchip/vllm#3 and #4. A5 waits for #274 and for @mhenrichsen on hand-edited preambles. CI green. |
| 2026-10-06 | `7af097b` | Eight PRs merged: A1 (#271, #277, #278), A2 (#279), A3 (#272, #273), D1 (#276) and D3 (#280). All of C5's planned PRs are merged. |
| 2026-10-08 | `9133015` | A4 (#274 as `76e1a15`) and A5 (#275 as `c5a6e78`) merged. @mhenrichsen rebased #275 onto main after #274 and resolved the preamble and `PATCHES.md` conflicts himself. All of C6's planned PRs are merged. Also merged: #267 (A8's dependency, as `bb3c579`), #282 (cpuchip's port checks, 2026-10-06, as `cda8925`), #260 (`api-root-health`, as `978d271`) and #284 (compose publishes on `127.0.0.1` by default, as `9133015`; it closes issue #204). Issue #206 closed: the 0.30 series does not carry the hunk. |
| 2026-10-08 | `9133015` | A5 follow-ups (1) to (3) built as three local branches off main: `fix/stale-cut-against` `5466da6`, `fix/export-keeps-preamble` `09dd6c2`, `fix/verify-kvarn-list` `aa87c98`. Each is verified (series check OK, round-trip test, a KVarN block harness against main). Follow-up (4), the `Verify:` header, is replaced by a proposed `graded()` triple check in CI. Review before the PRs open. Review done the same day. Two fixes, `cd1c3d4` on (1) and `79dfbb3` on (2). (3) needs no change. Merge (2) first. |

**How to update.** When an item changes state, update its row and add one log line. Each
re-verification pass adds one log line here, and one dated block at the top of each doc.

## 1. Executive summary — top 5 opportunities

| # | Opportunity | Expected gain [ESTIMATE] | Quality risk | Effort |
|---|---|---|---|---|
| 1 | **Enable KVarN's shared-dequant verify kernel** (`KVARN_SHARED_VERIFY`, still default-off over an unresolved corruption — a *second*, unrelated KVarN corruption, #208's "!!!!" page-flush, was isolated and fixed in #222 since this analysis) + batch-1 kernel-efficiency pass on the KVarN decode path. The fallback re-dequantizes the whole KV cache **once per query token** (4× at MTP k=3, 8× at DFlash2 k=7) every step. **2026-10-04:** #262 adds a second opt-in knob, `KVARN_FP16_DEQUANT`, and with #263 both knobs on measure 2.2× at a ~103k prompt on E: 23.0 → 50.1–52.2 tok/s over 5 fresh boots each, WSL2 [REPO-MEASURED: PR #262 body]. Shared verify alone gave +1% to +33%, because the fp32 kernel spills registers (top block). | **240k: +80–200%** (step ~131 → ~35–45 ms; that step range alone is +190–270%, so the band is conservative); 112–150k KVarN: +40–90%. 2026-10-04: +126% measured at ~103k (medians 51.9 vs 23.0); ~11 → ~28 tok/s at 240k [ESTIMATE, top block], so the band holds but the absolute step times do not (~52 ms already at 103k) | none for shared verify (kernel math unchanged; validated in isolation already). **Low** for `KVARN_FP16_DEQUANT`: it changes numerics. Its torch gate passes, and one PPL run on another checkpoint matches the huge baseline; PPL and needles on the default checkpoint are owed | **M** |
| 2 | **Switch on and validate the 0.30.0 spec-decode wins — the port has LANDED** (PR #189 merged 2026-09-28, `vllm==0.30.0` pinned): adaptive verification (vllm#52228), DFlash AOT-schedule drop (vllm#54374), gc-freeze graph capture (vllm#54646), `FULL_DECODE_ONLY` fallback (vllm#55095), Mamba-state-at-EAGLE-resume (vllm#53945). **2026-09-29:** reading the 0.30.0 source, adaptive verification is opt-in and refuses GDN-hybrid models and non-`ALWAYS` graph backends (FlashInfer on sm86, KVarN), and the other four are correctness/boot fixes, not decode levers (§5 T2). | **~0 as shipped**; +5–15% at both depths only if adaptive verification is patched to run on GDN + the D/E backends | none (spec decode is exact) | **M** (a patch, not just validation; was S) |
| 3 | **Re-enable MTP k=4 on the fp8/FlashInfer 150k path; FULL CUDA graphs there are not reachable on sm86** (PIECEWISE costs −6.6% step [REPO-MEASURED: gotcha 40, lines 807–816]; k=4 is worth ~+7% [REPO-MEASURED: docs/optimizations.md lines 399–405, was "397–403"]). k=4 is gated on the #34 FlashInfer crash, which the 0.30 pin + FlashInfer 0.6.18.post1 may already fix; a soak test decides. (#34 was closed as NOT_PLANNED on 2026-10-04 for inactivity, unresolved; the soak is still the gate.) FULL graphs: 0.30.0's FlashInfer still reports `UNIFORM_SINGLE_TOKEN_DECODE` on sm86 (`vllm/v1/attention/backends/flashinfer.py:977-1012`), so no soak turns them on. | **150k: ~+7%** (k=4 only; was +8–14% with the graph half) | none | **S–M** |
| 4 | **DFlash2 + fp8 at 150k via the FA2-fp8 plugin geometry relaxation** (issue #153: TP=1 needs `(256,4)`/`(128,8)` admitted to the adapter's tested set). Gives DFlash2's acceptance + the lookup lane at 150k with FULL graphs, instead of the TRITON/int8 tier that decays with depth (gotcha 40) or KVarN's per-token verify. | **150k copy/quote: +30–100%**; mixed prose at depth: ~0 (MTP keeps that workload) | none | **M** |
| 5 | **int4-per-token-head KV + MQ-3D split-KV verify as the hedge 240k path** (`single-user/alternative.sh`, which already defaults `VLLM_INT4_MQ_3D=1`, line 48, was "line 47"): measured 38–46 tok/s at 88k on the 3090 [REPO-MEASURED: docs/spec-decode-scratch-token-units.md lines 366–378] vs KVarN's 38.6 at 90k, pool 314,915 tokens, GSM8K 96.0 / 100k needle retrieved [REPO-MEASURED: docs/long-context.md lines 442–444, 457–463]. | **240k: 1.5–2.5× current E** if #1 stalls | **medium** (PPL at 100k+ depth and a needle on this serving path are not yet published; the 33k-token batch PPL is +0.3% and a batch-mode 240k needle passed, docs/long-context.md:98–99 — 2026-09-29 correction of "240k needle not yet published") | **M** |

**Recommended order (2026-10-04): T1, T3, T4, T5, T2.** The T labels map to rows 1–5 above
(§4) and stay fixed.
1. **T1** — it is the only row with a measured gain at depth (2.2× at ~103k). What remains is
   a native-Linux run, 150k/240k cells, PPL and needles with FP16 on, and the MTP side.
2. **T3** — it is the cheapest: a k=4 soak plus the #121 check on the shipped 0.30 image is the
   whole gate.
3. **T4** — its copy-work gain at 150k still stands, but it waits on the #153 adapter geometry.
4. **T5** — it is lossy, and it now trails T1's measured rate (38–46 tok/s at 88k against
   ~51 at ~103k with both knobs on; different harnesses).
5. **T2** — it gives ~0 as shipped and needs a GDN and backend patch first.

**Read of the whole table:** per-step time at batch 1 is ~72–76% **weight bytes** (a fixed
~14.6 GB of target weights, §2.1; was "~70–80%" of "17.1 GB") and the rest is KV-cache
read+dequant and kernel efficiency. Lossless weight
reduction does not exist (the body is already W4A16 Marlin near its roofline; the heads are
already int4-GPTQ). Therefore the ranking is dominated by (a) **not re-reading/re-dequantizing
KV per draft token**, and (b) **tokens per step** (drafting quality), with kernel-efficiency
recovery third. Lossy ideas (int4 KV beyond KVarN, W3A16 weights, lower-precision state) are
ranked below the lossless ones and each carries a quality budget check against Hard
constraint 2.

**Two cheap honorable mentions** (not top-5 because they do not move the stated benchmark):
- **Rebuild the MTP draft vocabulary over a multilingual corpus** (issue #196: the shipped
  40,960-id list covers ~23% of one reporter's real outputs — 5–8% on Chinese prose — and
  acceptance collapses to 1.06 tok/step there). Effort **S**, lossless, huge for non-en/da/code
  users; ~0 for the repo's en/da/code cohort (its 97.5% coverage is already measured
  [REPO-MEASURED: drafter/README.md lines 17–22]).
- **Multilingual-or-full head A/B** (`MTP_DRAFT_VOCAB=0`): the same reporter measured the full
  lm_head *faster* than even a rebuilt truncated head at `CTX=long` (80.6 vs 61–64 tok/s). That
  contradicts repo doctrine, and it is one box. The maintainer has since replied (2026-09-27/28:
  the list is language-specific, the `.txt` corpus bug is fixed in #213, gotcha 61 added) and
  still owes the `CTX=long` k=3 full-vs-truncated A/B; was "no maintainer reply". Open question Q3.


---

## 2. Decode hot-path map for setups D and E, with roofline math

### 2.1 Model constants (for all math below)

From the checkpoint config (`text_config` of `dbirks/Qwen3.8-27B-W4A16-AutoRound`,
[EXTERNAL-MEASURED: huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/raw/main/config.json])
and the repo's own figures:

- 64 layers: **48 Gated DeltaNet (GDN) + 16 full attention** (every 4th layer;
  `full_attention_interval=4`). hidden 5120; attention: 24 q heads, **4 KV heads, head_dim
  256**; GDN: 16 key heads ×128, 48 value heads ×128; vocab 248,320;
  `max_position_embeddings` 262,144.
- **Weights per step (read once per forward):** ~15.9 GiB is `Model loading took` on a
  `SPEC=dflash2` boot (RTX 4090 WSL2, `alternative.sh` int4, `DFLASH_TOKENS=15`)
  [REPO-MEASURED: docs/spec-decode-scratch-token-units.md line 342, was "line 343"; boot
  config on lines 330–332] ≈ 17.1 GB. **2026-09-29 correction:** that figure includes the
  1.19 GB DFlash2 drafter and the int8 embedding table (248,320 × 5,120 × 1 B ≈ 1.27 GB, one
  row gathered per step). So the target weights actually read per step are ≈ 17.1 − 1.19 −
  1.27 ≈ **14.6 GB** [ESTIMATE]. The earlier text used 17.1 GB as target-only and added the
  drafter on top, counting it twice for E. Of that: int4-GPTQ lm_head 248,320 × 5,120 × 0.5 B
  ≈ 0.64 GB [ESTIMATE, arithmetic; not a repo figure]; the W4A16 Marlin body is the rest,
  ~13.9 GB (the old "~13.2 GB" had no repo source); int4-GPTQ lm_head and MTP module
  [REPO-MEASURED: docs/quality.md lines 49–58 (precision only, no byte counts)].
- **Recurrent state per request:** 48 × (48×128×128) elements = 37.7 M elements = **151 MB
  fp32 / 75.5 MB fp16**, read+written every step [REPO-MEASURED: docs/optimizations.md lines
  54–58 ("~150 MB per request" fp32); fp16 state is the shipped default, docs/quality.md
  line 33].
- **KV bytes per token** (16 attention layers × 4 heads × 256 dims × K+V):
  bf16 65,536 B [REPO-MEASURED: gotcha 30]; fp8 32,768 B [REPO-MEASURED: docs/long-context.md
  line 11 — its "2 KB per token" is per layer; × 16 layers]; int8-per-token-head ~33,280 B
  [REPO-MEASURED: gotcha 39, 2080 B/layer, line 736; also gotcha 28 line 308]; KVarN
  k4v2_g128 840 B/layer → **13,440 B** [REPO-MEASURED: docs/long-context.md line 15,
  kvarn/README.md line 70 (was "line 66", earlier "line 51")]; int4-per-token-head ~17,100 B [ESTIMATE, redone
  2026-09-29: the old "6.96 GiB pool ÷ 405,948 tokens = ~17,900" divided the fp8 pool at 0.95
  by the int4pth pool at 0.93 (docs/vllm-0.29.md lines 413, 422) and is 18,409 by its own
  numbers. Scaling fp8's 0.93 pool instead (6.48 GiB / 204,896 tokens, line 410 = 33,958
  B/token) by the token ratio 204,896 / 405,948 gives ~17,100].
- **Drafter weights per step:** DFlash2 W4A16 1.19 GB [REPO-MEASURED: docs/optimizations.md
  lines 222–228, was "220–227"]; MTP int4 ~0.45 GB [ESTIMATE from "~850 MB bf16 → int8",
  prepare/README.md line 45, was "line 16", then int4 GPTQ fast variant].
- **Card:** RTX 3090, 936 GB/s spec, 82 SMs, sm86, 250 W cap. Sustained effective bandwidth
  ~85–90% at these read sizes [REPO-MEASURED: gotcha 10 — the GDN decode kernel "already runs
  at ~85% of the 3090's memory bandwidth"; docs/optimizations.md lines 410–412 (was
  "405–410") — the Marlin gap "is the memory system's ramp on 16–92 MB reads, not the kernel"].


### 2.2 Setup D — `SPEC=mtp CTX=long` (fp8 KV, FlashInfer, k=3, MAX_LEN 150k)

Launcher mapping [REPO-MEASURED: single-user/start_qwen.sh lines 203–206, was "202–204"]:
`--kv-cache-dtype fp8`, `DRAFT_TOKENS=3`, MAX_LEN 150,000; async scheduling on (`:659-660`).
fp8 on sm86 has exactly one backend, FlashInfer: FLASH_ATTN refuses fp8 KV (needs FA3/SM90+)
and TRITON refuses fp8e4nv below SM89 [REPO-MEASURED: gotcha 40, lines 783–788, was
"777–787"]. FlashInfer's spec-decode path is single-token-decode-only, so the verify step
runs **PIECEWISE** (no FULL CUDA graph) [REPO-MEASURED: patches/triton-spec-attn-fp8-kv.patch
preamble, lines 6–7]. That is still true on the 0.30.0 pin: without trtllm-gen (SM90/SM100+
only), `get_cudagraph_support` returns `UNIFORM_SINGLE_TOKEN_DECODE`
(`/tmp/vllm-0.30.0` `vllm/v1/attention/backends/flashinfer.py:977-1012`,
`vllm/utils/flashinfer.py:592-607`).

Per decode step, in order:
1. **MTP drafter, 3 chained single-token forwards.** Each: one drafter layer + the truncated
   40,960-row lm_head slice (`patches/qwen3_5-mtp-draft-vocab.patch`). The drafter's own
   1-layer KV grows with context; 3 serial passes per step.
2. **Target verify forward, 4 query tokens, one pass:** 64 layers — W4A16 Marlin GEMMs
   (~14.6 GB), GDN chunk-scan on fp16 state, and 16 layers of fp8 FlashInfer multi-query
   attention over the full context (KV read once per step).
3. **Rejection sampler** (sort-free top-k/top-p, `patches/sampler-small-topk-fast-softmax.patch`;
   kernels prewarmed by `patches/spec-sampler-prewarm.patch`).
4. On accept, k+1 tokens commit and the GDN state rolls forward; on reject it restores from
   the checkpointed position (backstopped by `patches/vllm-pr50021-gdn-spec-bounds.patch`).

Step-time budget at 150k [ESTIMATE, bandwidth-only, 100% efficiency; recomputed 2026-09-29
with the ~14.6 GB target-weight figure from §2.1]: weights 14.6 + fp8 KV 150,000×32,768 B =
4.92 + state 0.15 + drafter ~0.5 = **20.2 GB → 21.5 ms → 46.4 tok/s unspeculated ceiling**
(was 22.7 GB → 24.2 ms → 41.3). At the measured ~2.6 tok/step [REPO-MEASURED:
docs/vllm-0.29.md lines 273 (2.61) and 334 (2.66), was "line 271"] → **speculated ceiling ≈
121 tok/s** (was ≈ 107). Measured: 93–102 at C1 (short), ~68–80 at depth → **~56–66% of the
byte roofline at depth, ~77–84% short** (was "~63–77% / ~85%"; the at-depth rows are 25k–112k
measurements against a 150k roofline). The at-depth gap is the PIECEWISE downgrade (−6.6%
step [REPO-MEASURED: gotcha 40, lines 807–816, was "803–810"]), the FlashInfer fp8 batch-1
decode kernel, and per-step host/sampler overhead.

### 2.3 Setup E — `SPEC=dflash2 CTX=huge` (KVarN k4v2 KV + DFlash2, MAX_LEN 245,760)

Launcher mapping [REPO-MEASURED: single-user/start_qwen.sh lines 198–201 and 220–225, was
"197–199, 219–223"; DFlash2 k=7 default on line 262; MAX_LEN 245,760 at k≤7 on line 377]:
`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`, DFlash2 k=7 (QLEN=8 verify). FULL graphs
are restored for DFlash2 (residue fix `a75ee4b`/`b356e31`); MTP stays PIECEWISE (gotcha 33,
lines 435–446; `start_qwen.sh:641-643`). **Async scheduling is ON at the default k=7.**
`--no-async-scheduling` is set only for `DFLASH_TOKENS>7` with the lookup on
(`start_qwen.sh:310-316`), because only the long adaptive block needs per-step draft-count
feedback [REPO-MEASURED: docs/optimizations.md lines 281–291, was "285–289"]. The earlier
text said CTX=huge always runs `--no-async-scheduling`; that was false at every tree this
doc has been checked against.

Per decode step:
1. **DFlash2 drafter, one non-autoregressive pass** (5 layers, 2,048-token sliding window —
   cost independent of context) + optional lookup fill from the request's own history
   (`patches/dflash2-lookup-drafting.patch`).
2. **Target verify forward, 8 query tokens:** weights (~14.6 GB), GDN state, and 16 layers of
   KVarN attention. As shipped (`KVARN_SHARED_VERIFY` off), the per-token path launches
   `_kvarn_fused_decode_stage1` over an `(NQ, Hk, SPLITS)` grid, i.e. one program per query
   token (`triton_kvarn_decode.py` lines 1064–1077, was "1008–1021"; kernel 568–706, was
   "529–654"). Each program walks
   128-token tiles: load packed 4-bit keys / 2-bit values + scales, dequantize,
   Hadamard-rotated q·k, online softmax. The shared kernel that is gated off is
   `_kvarn_fused_verify_stage1`, lines 1110–1274 (was "1053–1202"); the old citation
   "1053–1120" pointed at it as if it were the path that runs. Since #262 (2026-10-04),
   `KVARN_FP16_DEQUANT=1` runs the dequant in fp16 in both kernels; it is also default-off.
3. Sampler + commit, as D.

Step-time budget at 240k [ESTIMATE, bandwidth-only; recomputed 2026-09-29]: weights 14.6 +
KVarN KV 240,000×13,440 B = 3.23 + state 0.15 + drafter 1.28 = **19.3 GB → 20.6 ms → 48.6 tok/s
unspeculated ceiling** (was 21.7 GB → 23.2 ms → 43). At 2.5 tok/step → **speculated ceiling ≈
121 tok/s** (was ≈ 107); the 20k six-task row reads 3.15 tok/step (docs/long-context.md line
220), so 2.5 is conservative. In copy mode at ~8–15 tok/step (lookup engaged
[REPO-MEASURED: docs/optimizations.md lines 309–316, was "307–314"]) the ceiling is 250–400+
tok/s. Measured 73.7 @ 25k, 38.6 @ 90k, ~18 extrapolated @ 240k → **~61% / ~32% / ~15% of the
240k ceiling** (73.7 is ~52% of its own 25k roofline, ~143 tok/s). The earlier "~35–55% at 25k"
did not follow from its own numbers (73.7 / 107 = 69%); "~20% at depth" is now ~15%. That gap is
finding F1, and it is enormous.

### 2.4 What the roofline says, plainly

| term | 150k fp8 (D) | 240k KVarN (E) |
|---|---|---|
| weights | 14.6 GB (72%) | 14.6 GB (76%) |
| KV read | 4.92 GB (24%) | 3.23 GB (17%) |
| state + drafter | 0.65 GB (3%) | 1.43 GB (7%) |
| **step bytes** | **20.2 GB** | **19.3 GB** |
| ceiling @2.5–2.6 tok/step | ~121 tok/s | ~121 tok/s |

(2026-09-29: recomputed from the ~14.6 GB target-weight figure; the 2522ef9 table read 17.1 GB
weights, 22.7 / 21.7 GB step bytes, 75%/79% weights, ~107 tok/s ceilings.)

Three consequences:
1. **Weights dominate everywhere.** Even a perfect (free) KV cache caps D and E near
   ~145–160 tok/s at current acceptance [ESTIMATE: D 14.6 + 0.65 = 15.25 GB → 16.3 ms/step →
   61 steps/s × 2.6 ≈ 160; E 14.6 + 1.43 = 16.03 GB → 17.1 ms/step → 58 steps/s × 2.5 ≈ 146.
   The old "~110–125" did not follow from its own 17.1 GB either, which gives 126–137].
   Lossless gains must come from *efficiency* (closing the
   measured-vs-roofline gap) and *tokens per step*.
2. **KVarN has already won the byte war at 240k** (KV is 17% of step bytes). The 240k
   problem is not capacity and not bytes: the kernel *executes* far above its byte time
   (finding F1).
3. **To beat ~120–160 tok/s at any depth you need more tokens per step**, not faster
   kernels. That is why drafting quality (opportunities 3, 4, and the lookup/suffix family)
   is a first-class lever even though kernels look like the story.


---

## 3. Bottleneck findings (code evidence + confidence)

**F1 — The 240k bottleneck: KVarN verify re-dequantizes the whole KV cache once per query
token, every step (and the shared-dequant fix already exists but is default-off).**
`kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py` `kvarn_verify_attention` (lines
932–1087; was "877–1032", earlier "877–960") documents two modes: a UNIFORM shared-dequant path where "the
request's QLEN tokens SHARE each block's dequant, so KV bytes and dequant ALU match
single-token decode", and a per-token fallback with "**QLEN-x redundant dequant**"
(docstring, lines 949–954; was "894–899", earlier "~889–895"). The shared path is gated behind
`envs.KVARN_SHARED_VERIFY` (line 1003, was "947"), **default OFF**, because "serving with it corrupts the
MTP drafter's proposals (invalid [-1,...] spec tokens, embedding index asserts at
temperature>0, degenerate greedy output) through a mechanism not yet isolated — suspicion is
an interaction with async scheduling / drafter metadata rather than kernel math" (a code
comment above the gate, lines 995–1002; was "939–946", earlier "~936–946" and called the docstring). The kernel
was "numerically validated in isolation (matches the per-token kernel within fp32 reduction
noise on live inputs)". So at HEAD, every verify step on `CTX=huge` runs per-token: **4×
redundant dequant at MTP k=3, 8× at DFlash2 k=7**, growing linearly with context.
Quantified [ESTIMATE, per-tile cost recomputed 2026-09-29]: at 90k the per-token path visits
8 × (90,000/128) × 16 ≈ 90,000 tile-dequants/step. The repo's own 25k/90k numbers
[REPO-MEASURED: docs/vllm-0.29.md lines 126–127] give a slope of (62.2 − 32.6) ms over 65,000
extra visits ≈ **0.46 µs/tile-visit** at 2.4 tok/step (was "~0.33", which did not follow from
those numbers). That is ~41 ms of a ~62 ms step (was "~30 ms of a ~52–62 ms step"); sharing
the dequant cuts it to ~5 ms → step ~26 ms → **~2.4× at 90k** (the "~2–2.5×" claim stands).
This also explains the observed steepening of the KVarN tax with context [REPO-MEASURED:
docs/long-context.md lines 36 and 42 (1.22× batch at 100k), 54 and 58–60 (2.13× single-user
at 112k, ~1.98× of it step time), summary on 71; issue #11; was "lines 57–59"]: the tax is
per query token, and single-user verifies 4–8 queries.
**Confidence: HIGH** that this is the dominant 240k cost (code + repo's own depth slope
agree); MEDIUM-LOW that enabling the shared kernel is easy. The corruption mechanism is
unresolved. **2026-09-29 correction:** the earlier text argued "the launcher already runs
`--no-async-scheduling` at CTX=huge", so enabling might be a validation exercise. That is
false: async scheduling is on for the default k=7 DFlash2 profile and for every `SPEC=mtp
CTX=huge` boot (`start_qwen.sh:310-316, 659-660`), so the named suspect is live on the
shipped profile. **2026-09-28 note, re-checked 2026-09-29:** the gate and the comment are
unchanged on 0.30 (`kvarn-0.30.0.patch:49,78`; `triton_kvarn_decode.py:995-1002`, was
"939-946", earlier "941-945"; no `kvarn/*.patch` touches this file, but bd6c5e2 (#262) edits it
directly, which moved the lines). #222 proved the bisect-and-fix loop on a
*different* KVarN corruption (#208's cross-group page flush, fixed by telling the runner which
pages moved groups each step; reproducer `bench/concurrent_collapse.py`). Neither fix touches
this path. One field data point since then, not repo-measured and not a check of the MTP
symptom above: in #208 (2026-09-28) a single-3090 reporter ran `bench/concurrent_collapse.py`
with `KVARN_SHARED_VERIFY=1 SPEC=dflash2 CTX=huge PREFIX_CACHE=1` (async on, k=7). Before
#222, 0/30 trials collapsed but `corrupted_requests_total` rose by 1 on the second run; after
#222, three runs were clean.
**2026-10-04: the speed-up arrived, but this finding's mechanism was incomplete.** Register
spills, not only the QLEN-x repeated dequant, were the main cost. At 100k context and QLEN=8,
the best fp32 config of the shared verify kernel spills 102 registers and takes 2.5 ms per
call; the fp16 pick spills 4 and takes 0.98 ms [REPO-MEASURED: PR #262 body, section 3]. So
shared verify alone gave only +33% (fresh boot) and +1% (cached boot) on E at ~103k
[REPO-MEASURED: PR #240, maintainer A/B]. Shared verify plus `KVARN_FP16_DEQUANT` (#262) plus
#263 gave 2.2×, close to the ~2.4× estimated above. The predicted ~5 ms of verify attention per
step after sharing does not hold: 16 layers × 0.98 ms is ~16 ms at 100k [ESTIMATE], ~11× the
kernel's byte time (100,000 × 840 B ÷ 936 GB/s ≈ 0.09 ms per call). Confidence that KVarN
verify is the dominant 240k cost stays HIGH.

**F2 — fp8 verify on sm86 is structurally downgraded: PIECEWISE graphs and k=3 (not 4).**
Two independent constraints on D: (a) FlashInfer's spec-decode path is single-token-only →
PIECEWISE, measured −6.6% step (27.2 → 25.4 ms) [REPO-MEASURED: gotcha 40 lines 807–816,
was "803–810"; patches/triton-spec-attn-fp8-kv.patch preamble lines 6–7]. Note that this A/B
changed dtype and backend together (fp8/FlashInfer/PIECEWISE vs int8/TRITON/FULL); (b) k=4 on
FlashInfer crashed with an illegal memory access on 0.28.0 ("n=4 eventually dies, n=3
stable"), so CTX=long gives up ~7% [REPO-MEASURED: docs/optimizations.md lines 399–405, was
"397–403"]. The repo's Triton fp8 split-KV verify kernel exists but is sm89+ [REPO-MEASURED:
gotcha 57, lines 1291–1305, was "1285–1299"].
**Confidence: HIGH** (all repo-measured). **2026-09-29:** gate (a) is structural on the 0.30.0
pin, not version luck. FlashInfer's `get_cudagraph_support` returns
`UNIFORM_SINGLE_TOKEN_DECODE` on any GPU without trtllm-gen (SM90/SM12x/SM100+ only;
`vllm/v1/attention/backends/flashinfer.py:977-1012`, `vllm/utils/flashinfer.py:592-607` in
`/tmp/vllm-0.30.0`), so a GPU soak can only answer (b) (open question Q4).


**F3 — DFlash2's drafter sees only a 2,048-token window, so its acceptance edge collapses on
non-copy text at depth; it wins at long context only when the lookup lane fills from the
prompt.** Repo doctrine: "DFlash2 past 64k is worth it only for context reproduction and
loses to SPEC=mtp CTX=long roughly 2:1 on everything else" [REPO-MEASURED: .env.example
lines 18–21]; 128k acceptance divergence measured in issue #60. The lookup lane restores it
for copy/quote (164 tok/s on the copy task at `--ctx 20000`, bare-metal 3090; the `quote` task
reads 58 [REPO-MEASURED: docs/long-context.md lines 217–219; README.md:85 calls it "164 while
quoting"; the old citation "single-user/README.md" does not contain the figure]).
**Confidence: HIGH.** Consequence: DFlash2-at-150k (opportunity 4) is a *copy-workload*
play, not a mixed-prose play.

**F4 — Weights are ~72–76% of step bytes (was "75–79%", §2.4) and are already near their
roofline.** Marlin retuning for the decode shapes (M ≤ 16 on sm86) measured "3-7% per GEMM in
isolation, nothing measurable end to end — the remaining gap to peak bandwidth is the memory
system's ramp on 16–92 MB reads, not the kernel" [REPO-MEASURED: docs/optimizations.md lines
407–412; was "405–410"]. The "+0.4% end-to-end" figure the old text attributed to W4A16 is the
W4A8 tile table at M=2048, a prefill/batch shape [REPO-MEASURED: docs/optimizations.md lines
192–197; was "190–195"]. lm_head/MTP/drafter are already int4-GPTQ [REPO-MEASURED:
docs/quality.md lines 49–58]. **Confidence: HIGH.** Consequence: no lossless weight lever
remains; W3A16 is the only further weight cut and is lossy (table row T7).

**F5 — The GDN/recurrent path is not the decode bottleneck.** The DeltaNet decode kernel
already runs at ~85% of memory bandwidth and every tuning variant lands within 3%
[REPO-MEASURED: gotcha 10, lines 149–153]; fp16 state read+write is ~0.16 ms/step
[ESTIMATE: 151 MB ÷ 936 GB/s]. State size bounds *concurrency* (pages scale with the verify
block, not slots [REPO-MEASURED: gotcha 29]), not single-user speed. **Confidence: HIGH.**
Consequence: a fused GDN decode kernel (the external pattern — FlashInfer SM100
`fused_kda_decode`, 1.33× the vLLM fused kernel at one row on B200 [EXTERNAL-MEASURED:
flashinfer v0.6.18 release notes, re-read 2026-09-29; it is Kimi Delta Attention, not GDN];
vLLM #53835 SM110, merged 2026-09-05) buys little on sm86 here; deprioritized (rejected idea
R6).

**F6 — CPU/host overhead is small: PIECEWISE on the fp8 path (F2), and `--no-async-scheduling`,
where it is used (`DFLASH_TOKENS>7` with the lookup, not the default CTX=huge profile — see
§2.3), costs <1% at batch 1** [REPO-MEASURED: docs/optimizations.md lines 281–291, was
"285–289"]. The sampler is already patched; the DFlash draft pass is a captured graph
[REPO-MEASURED: gotcha 20]. **Confidence: HIGH.**

**F7 — The int8-QK prefill kernel and Marlin int8 GEMM tunes are gated to exact geometry /
known to misfire on this checkpoint, but those are prefill/batch concerns, not single-user
decode** [REPO-MEASURED: patches/prefill-attn-int8.patch, marlin-int8-negative-scales.patch;
int8 activations buy nothing at batch 1, docs/quality.md line 49, and may cost ~9% decode
via acceptance — issue #62, unconfirmed]. **Confidence: HIGH** (not on the D/E decode path).

**F8 — Reliability cliffs on the exact D/E profile: the #34 FlashInfer+MTP Xid-31 at 28–34k
(cause unattributed [REPO-MEASURED: gotcha 40, lines 789–793]; #34 CLOSED as NOT_PLANNED on
2026-10-04 for inactivity, not solved; was "still OPEN, last maintainer status 2026-09-15"), the #107 engine stepping-stalls at ~190k MTP/fp8 (#107 still OPEN), and Bug B
residue corruption at CTX=huge (mitigated: PIECEWISE for MTP, FULL for DFlash2 [REPO-MEASURED:
gotcha 33]).** The earlier text called vllm#50021 (still OPEN upstream) the "upstream
candidate" for #107. That link is not in #107's thread, and the repo has shipped the backport
(`patches/vllm-pr50021-gdn-spec-bounds.patch`) since 9850338 (2026-08-17), before #107 was filed
on 2026-09-13. #50021 is therefore not an unapplied candidate fix for #107 [INFERENCE: assumes
the reporter ran the repo series]. No steady-state tok/s cost, but any change touching the
verify path must re-run their sweeps (`bench/residue_sweep.py`, `bench/verbatim.py`).
**Confidence: HIGH.**


---

## 4. Ranked opportunity table

Gains are [ESTIMATE] at true depth (150k / 240k) vs the honest baselines (~65–75 tok/s D,
~18–25 tok/s E; was "~19–25") unless noted. 2026-10-04: the WSL2 data puts E at 240k nearer
~11 tok/s [ESTIMATE, top block]. Quality risk is against Hard constraint 2. Evidence labels as
defined at the top.

| rank | idea | mechanism | evidence | gain @150k | gain @240k | quality risk (predicted impact) | effort | files / deps |
|---|---|---|---|---|---|---|---|---|
| **T1** | Enable + fix KVarN shared-dequant verify (`KVARN_SHARED_VERIFY`), then batch-1 kernel pass (splits/tile sweep) | stop re-dequantizing KV per query token; share dequant across QLEN | [REPO-MEASURED: triton_kvarn_decode.py 932–1087 (was 877–1032, earlier 877–960); docs/vllm-0.29.md 126–127; 2026-10-04: PR #262, 2.2× at ~103k on E with `KVARN_FP16_DEQUANT` + #263, WSL2] + [ESTIMATE §3/F1] | n/a (KVarN not the 150k tier) | **+80–200%** | none — kernel math unchanged; needle re-check only. `KVARN_FP16_DEQUANT`: low — it changes numerics; PPL and needles on the default checkpoint owed | M | `kvarn/files/.../triton_kvarn_decode.py`, `kvarn_attn.py`; async-scheduling hypothesis test (async is ON in the shipped profile) |
| **T2** | 0.30.0 is landed (PR #189); make adaptive verification (#52228) run on this model. #54374 (DFlash AOT drop), #54646 (gc-freeze), #55095 (FULL_DECODE_ONLY fallback for non-compiled models) and #53945 (Mamba-resume) are correctness/boot fixes already in the pin | fewer wasted drafts when acceptance dips | [EXTERNAL-MEASURED: vLLM v0.30.0 release notes; PR #189 verified identical pools on 3090, docs/vllm-0.30.md 82–91] + [0.30.0 source: opt-in flag, refuses GDN and non-`ALWAYS` backends, §5 T2] | ~0 as shipped; +5–15% if patched | ~0 as shipped; +5–15% if patched | none (spec decode exact; IFBench/PPL unchanged by construction) | M (was S–M) | patch GDN varlen-verify support + D/E backend graph support; `"enable_adaptive_verification":true` in the spec config; re-run acceptance |
| **T3** | MTP k=4 at CTX=long (post-soak); FULL graphs on the fp8/FlashInfer verify are not reachable on sm86 in 0.30.0 | deepen drafts | [REPO-MEASURED: gotcha 40; optimizations.md 399–405 (was 397–403)] + [0.30.0 source: flashinfer.py 977–1012] | ~+7% (was +8–14%) | n/a | none | S–M | `start_qwen.sh` DRAFT_TOKENS (line 205); gated on #34 soak |
| **T4** | DFlash2 + fp8 at 150k (FA2-fp8 geometry relaxation, #153) | DFlash2 acceptance + lookup lane at 150k, FULL graphs | [EXTERNAL-MEASURED: club-3090 TP=2 103/192 tok/s, quoted in issue #153] + [REPO-MEASURED: gotcha 40; #194] | copy +30–100%; mixed ~0 | n/a | none | M | plugin adapter gate `(256,4)`/`(128,8)`; port to the 0.30 image; battery |
| **T5** | int4-per-token-head KV + MQ-3D (`VLLM_INT4_MQ_3D=1`, already `alternative.sh`'s default) as hedge 240k path | proper split-KV verify over int4 cache; pool 314,915 tokens | [REPO-MEASURED: spec-decode-scratch doc 366–378; long-context.md 88–115, 435–466; issue #60] | n/a | 1.5–2.5× current E | **medium** — PPL at 100k+ depth and a 150k/240k needle on this serving path unpublished; 33k PPL +0.3%, batch 240k needle, GSM8K 96.0 & 100k needle OK | M | `alternative.sh` + mq3d patches; quality battery |
| T6 | Multilingual draft-vocab rebuild (#196) | acceptance on non-en/da/code traffic | [REPO-MEASURED: issue #196] | ~0 on cohort; large for multilingual users | same | none | S | `prepare/build_draft_vocab.py --corpus` |
| T7 | W3A16 (~3bpw) body, EXL3-style, mixed W3/W4 | cut the dominant ~14.6 GB weight term (was "17.1 GB") ~20–25% | [EXTERNAL-MEASURED: ExLlamaV3 Qwen3.8-27B@4bpw 48.6 t/s 3090Ti; Flash-Next@3bpw 62.8 t/s] + [ESTIMATE] | +15–20% | +15–20% | **high** — likely exceeds PPL +2% budget; fallback W3-on-MLP-only | M–L | new quant in `prepare/`; Marlin W3 or EXL3 kernel |
| T8 | Suffix/prompt-lookup proposer at 150k. vLLM 0.30.0 already ships `method="suffix"` (no backport; was "backport vLLM main"), but it is its own speculator, so it *replaces* MTP, and it forces the V1 runner (`vllm/config/vllm.py:2839-2850`) | copy-mode drafts for the MTP tier (today lookup is DFlash2-only) | [EXTERNAL-MEASURED: SuffixDecoding ~5.3× on agentic/copy; vLLM spec-decode docs] | copy +20–60% (vs MTP only if MTP+suffix were combined, which stock 0.30 cannot do) | n/a | none | M | `pip install arctic-inference==0.1.1` (`vllm/config/speculative.py`); an MTP+suffix hybrid needs code; conflicts with DFlash2 lane — pick per profile |
| T9 | KVARN_NUM_KV_SPLITS / tile-size autotune at batch 1 | occupancy of the fused kernel | [REPO-MEASURED: triton_kvarn_decode.py 36–97, was 37–78; 2026-10-04: with `KVARN_FP16_DEQUANT` on, #262 already limits the split-K autotune to non-spilling configs, lines 60–77] | n/a | +10–30% (stacked on T1) | none | S | env sweep only |
| T10 | int8 recurrent state | halve state bytes | [ESTIMATE: 0.08 ms/step] | +0.3% | +0.3% | medium (state precision) for ~0 gain | — | rejected (see R5) |

**Order to work the top five (2026-10-04): T1, T3, T4, T5, T2.** The rank labels above stay as
they are, because other docs cite them.
1. T1 — the only row with a measured gain at depth (2.2× at ~103k).
2. T3 — the cheapest: a k=4 soak plus the #121 check is the whole gate.
3. T4 — its 150k copy-work gain stands, but it waits on the #153 adapter.
4. T5 — lossy, and it now trails T1's measured rate.
5. T2 — ~0 as shipped; it needs a GDN and backend patch first.


---

## 5. Top 5 — implementation sketches and validation plans

**Harness note (2026-09-29).** `bench/labd_bench.py` takes one integer `--ctx` per run
(`CTX = int(arg("--ctx", 20000))`, line 38), so every comma list below
(`--ctx 25000,90000,...`) means one run per depth. Past ~65k it also needs
`--corpus ~/bench/labd_corpus_long.txt` (usage, lines 11–19), or it silently measures a
shorter prompt. Whether that corpus reaches 150k/240k tokens is not verified here.
`bench/residue_sweep.py` takes a label as its first argument (line 26).
**(2026-10-04)** Compare arms over fresh boots, at least 2 per arm and ideally 5, each with
private, empty caches (`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`). Two
fresh boots of one image pick differently for 5 to 13 of 38 Inductor kernels and for 1 to 5 of
the 8 or 9 `@triton.autotune` kernels, and greedy output differs between them; a boot on saved
caches repeats its original output [REPO-MEASURED: PR #262 body, section 2]. One cached boot per arm can
therefore hide or invent a gain (the shared-verify-only arm read +33% fresh and +1% cached in
the #240 A/B).

### T1 — KVarN shared-dequant verify (the 240k fix)

**Mechanism.** At HEAD, `CTX=huge` verify runs the per-token fallback kernel: the whole
quantized KV cache is dequantized once per query token (4× MTP, 8× DFlash2) every step. The
shared-dequant kernel (`_kvarn_fused_verify_stage1` + stage2 combine) exists, is numerically
validated in isolation, and is gated off by `KVARN_SHARED_VERIFY` default-0 because serving
with it corrupted the MTP drafter's proposals "through a mechanism not yet isolated —
suspicion is an interaction with async scheduling / drafter metadata"
(triton_kvarn_decode.py lines 995–1002, the comment above the gate; was "939–946", earlier
"~877–960").

**Status 2026-10-04: advanced, not done.** #262 (bd6c5e2) adds `KVARN_FP16_DEQUANT`, default
off. It runs the dequant math of the three fused kernels in fp16 with fp32 accumulation, which
halves the working tiles' register cost (`kvarn/README.md:82-98`). With it on, the split-K
stage1 and verify kernels autotune only over BLOCK_N 16/32, num_warps=4 and no maxnreg
(`_f16_no_spill_configs`, triton_kvarn_decode.py lines 60–77). #263 (210db97) stops a fresh
boot from over-committing VRAM under WSL2. Measured on one 3090 at 250 W under WSL2, `CTX=huge
SPEC=dflash2 PREFIX_CACHE=1`, `bench/labd_bench.py --ctx 94000` (~103k-token prompt), 5 fresh
boots per arm, KV pool 268,169 tokens on every boot [REPO-MEASURED: PR #262 body]:

| arm | decode tok/s | median |
|---|---|---|
| `main` | 23.0, 23.0, 23.7, 23.2, 22.5 | 23.0 |
| #262, both knobs off | 23.9, 23.7, 23.3, 23.0, 23.7 | 23.7 |
| #262 at 2ee2c39 + #263, `KVARN_SHARED_VERIFY=1 KVARN_FP16_DEQUANT=1` | 52.2, 50.2, 52.0, 51.9, 50.1 | 51.9 |

The main cost was register spills, not only the repeated dequant (F1, 2026-10-04 note). The
fp32 shared kernel spills 102 registers and takes 2.5 ms per call at 100k; the fp16 pick takes
0.98 ms. So shared verify alone gave +33% on a fresh boot and +1% on a cached one (#240
maintainer A/B), and the 2.2× needs both knobs. Before #263, 2 of 5 fresh knob-on boots ran at
~25 tok/s, because the KV pool landed on top of compile scratch and the WSL2 driver moved
~380 MiB of GPU memory to system RAM. Before 2ee2c39, a `maxnreg=96` autotune pick ran at 16.5 tok/s. The soak
(`bench/concurrent_collapse.py`, 30 trials) showed 0 collapses on each soaked boot. What is
still open:
- The MTP corruption. `KVARN_SHARED_VERIFY` stays default-off, and steps 1–3 below still apply.
  For `SPEC=mtp CTX=huge`, FP16 alone is the only lever today, and nobody has measured it end
  to end.
- A native-Linux run. The #263 effect is WSL2-specific; the PR author expects native Linux to
  fail the allocation instead and did not check it.
- 150k and 240k cells. On the WSL2 data, 240k reads ~28 tok/s with both knobs on against ~11
  off [ESTIMATE, top block].
- FP16 quality on the default checkpoint: PPL and needles. The torch numerics gate passes
  (`kvarn/tests/test_kvarn_fp16_dequant_torch.py`). The #240 author's PPL (en 10.77 / da 10.92)
  matches the huge baseline (10.77 / 10.91, docs/vllm-0.29.md:123), but on another checkpoint.
- Step 4 (T9) is partly done: the F16 prune fixes the tile configs with the knob on. The
  `KVARN_NUM_KV_SPLITS` sweep is not done.

**Implementation sketch.**
1. Reproduce the corruption minimally: boot `SPEC=mtp CTX=huge` with
   `KVARN_SHARED_VERIFY=1`, async scheduling ON (the launcher default) vs OFF
   (`ASYNC_SCHED=0`, honoured at `start_qwen.sh:659-660`), and run
   `bench/residue_sweep.py <label>` + a tool-calling soak. The named suspect is async
   scheduling. **2026-09-29 correction:** the earlier text said single-user `CTX=huge`
   already forces `--no-async-scheduling` for the DFlash2 lookup, so "there is a real chance
   the shared kernel is already safe on the shipped profile". It does not. The launcher turns
   async off only for `DFLASH_TOKENS>7` with the lookup on (`start_qwen.sh:310-316`), so the
   shipped k=7 profile runs async, and the async-off arm is a real experiment rather than a
   confirmation. The #208 field run (F1) used the shared path on async k=7 DFlash2 and saw
   one corrupted request before #222 and none after. That is weak evidence the DFlash2 side
   may be safe; it does not test the MTP symptom.
2. If the corruption is async-only: gate the shared path on `not async_scheduling` (or on
   the V2-runner uniform-decode shape) rather than on a debug env var; promote the env to a
   registered knob with a safe default.
3. If it corrupts even without async scheduling: bisect the vq plan (`Seq_lens_ptr` CPU-side
   build vs the device `seq_lens` — the builder's own comment says the device tensor "can
   disagree" under async spec decode; that disagreement is the likely invalid-spec-token
   source). Pin the plan from the same snapshot the scheduler used for the step.
4. Then the batch-1 efficiency pass (T9): sweep `KVARN_NUM_KV_SPLITS` (16/32/64) and the
   stage-1 tile `BLOCK_N` at 90k/150k/240k; check SM occupancy of the `(B, Hk, splits)` grid
   at batch 1 (4 KV heads × splits on 82 SMs). The unset default is already
   context-adaptive: 32 splits up to 256 blocks, 64 above (`adaptive_num_kv_splits`,
   triton_kvarn_decode.py lines 80–97, was "61–78"), so at 90k+ the sweep starts from 64.
   `BLOCK_N` is autotuned over {16, 32, 64} (lines 51–57, was "50–58"). With
   `KVARN_FP16_DEQUANT=1`, the split-K stage1 and verify kernels keep only BLOCK_N 16/32,
   num_warps=4, no maxnreg (`_f16_no_spill_configs`, lines 60–77), so the tile half of this
   sweep is done for the fp16 path.

Diff outline: in `kvarn_verify_attention`, replace the `envs.KVARN_SHARED_VERIFY` clause with
a predicate on scheduler mode + validated geometry; in `kvarn_attn.py` plumb the flag;
document in `kvarn/README.md`. ~50–150 lines + tests.

**Benchmark + quality validation (Setup E, `CTX=huge SPEC=dflash2 PREFIX_CACHE=1`).**
- Decode vs depth: `bench/labd_bench.py <tag> --ctx 25000,90000,150000,240000 --tasks
  qa,summary` plus a verbatim-copy cell (`bench/labd_bench.py <tag> --ctx 20000` with the
  document-copy task). Arms: knobs off vs on, at least 2 (ideally 5) fresh boots each with
  private caches (harness note; was "flag off vs on, same boot", which cannot work: the knobs
  are read at boot and `KVARN_FP16_DEQUANT` is in the compile key). **Pass:** step time at 90k drops
  ≥ 30% (target decode ≥ 55 tok/s at 90k, ≥ 45 at 240k from the ~18–25 baseline); no arm
  slower than baseline by >3%. 2026-10-04: the ≥ 30% drop is met at ~103k on WSL2 (2.2×). The
  240k bar of ≥ 45 looks out of reach on that data (~28 [ESTIMATE, top block]).
- Correctness/quality: `bench/quality_battery.py huge-shared --gsm-n 200` (GSM8K ≥ baseline
  −1.0 pt → ≥ 95.0–95.5); perplexity vs E baseline (8.236 → ≤ +2% → ≤ 8.40);
  `bench/needle_test.py 150000 0.9` and `bench/needle_test.py 240000 0.9` retrieved;
  `bench/residue_sweep.py` + `bench/verbatim.py` clean at all 128 residues (Bug B/F8);
  30-min `bench/labd_soak.py` without degenerate output.
- IFBench (299 prompts, per docs/quality.md) on the best arm if the kernel path changed any
  numerics: ≥ 77.3 (baseline 78.3 − 1.0).

**Strongest reason it might NOT help:** the corruption may not be async-scheduling but a
real kernel/metadata bug that only shows under live drafter traffic, and the fix could
require re-plumbing the verify plan — in which case T1 slips from M to L. Even then, the
T9 sweep alone (splits/tile) is a cheap partial.

---

### T2 — vLLM 0.30.0 (PR #189) + adaptive verification

**Mechanism.** The port PR **landed** on 2026-09-28 (d88544b: `vllm==0.30.0` pinned, the
series re-exported, four apply-invisible regressions fixed; verified on the reference 3090
with identical KV pools on every shipped mode [REPO-MEASURED: PR #189 description, merged
2026-09-28T11:54Z; docs/vllm-0.30.md lines 82–91]). What remains of this row is the
enablement half. 0.30.0 carries (all confirmed in the v0.30.0 release notes and in
`/tmp/vllm-0.30.0`):
- acceptance estimation for adaptive verification (vllm#52228, merged 2026-09-14;
  `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py`): the engine shortens the draft
  when predicted acceptance is low, which matters most where verify cost is highest (long
  context);
- DFlash drafters dropping FlashAttention's AOT schedule (vllm#54374, merged 2026-09-04;
  `spec_decode/dflash/speculator.py:189-200`). This is a correctness fix: acceptance
  collapsed to 1.0 under FULL graphs when the target was not on FlashAttention;
- gc-freeze during V2 graph capture (vllm#54646, merged 2026-09-01; `vllm/utils/gc_utils.py`).
  This is a capture/boot-time and capture-safety fix, not a decode-rate change;
- a `FULL_DECODE_ONLY` fallback for *non-compiled* models (vllm#55095, merged 2026-09-11;
  "non-compiled models fall back to `FULL_DECODE_ONLY` graphs" in the release notes). The
  earlier "`FULL_DECODE_ONLY` graphs" read it as a new graph mode; `FULL_DECODE_ONLY` already
  existed in 0.29.0, and this compiled model is not the fallback's target;
- Mamba state cached at the EAGLE resume position (vllm#53945, merged 2026-09-08; a
  prefix-cache correctness fix). It is listed in the 0.30.0 notes, but its
  `shared_prefix_boundary` symbol also appears in the local 0.29.0 venv's
  `v1/core/kv_cache_manager.py`, and no repo patch adds it. So at least part of it predates
  the 0.30 pin [unresolved which part].

**What the 0.30.0 source says about enabling adaptive verification (2026-09-29, static
reading, not booted).** It is opt-in: `enable_adaptive_verification: bool = False`
(`vllm/config/speculative.py:539`), set via the speculative-config JSON. The launcher does not
set it (`start_qwen.sh:274, 500`). With it on, `maybe_create_adaptive_verification_manager`
(`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:448-497`) raises `ValueError` in
two cases:
(1) any checked layer's backend does not support a device/CPU query-length mismatch. The
checked set is every KV-cache-group layer minus the drafter's
(`vllm/v1/worker/gpu/model_runner.py:621-628`), which includes the 48 GDN layers, and SSM
backends opt out (`vllm/v1/attention/backend.py:212-224`; `gdn_attn.py:37` `is_ssm` → True).
(2) any target attention builder reports less than `AttentionCGSupport.ALWAYS`: FlashInfer on
sm86 reports `UNIFORM_SINGLE_TOKEN_DECODE` (D), and KVarN reports `UNIFORM_BATCH`
(`kvarn/files/vllm/v1/attention/backends/kvarn_attn.py:425-428`) (E).
So on this GDN hybrid, neither D nor E can turn it on without patching GDN varlen-verify
support and the attention backends' graph support. Also, #52228 is Model Runner V2 only
(`vllm/config/vllm.py:2894-2895` lists it as unsupported on V1). The field's own docstring
says "Currently only supported for method=\"dspark\"" (`vllm/config/speculative.py:540-541`),
but config validation does not enforce that: the only non-dspark check rejects it together
with `use_local_argmax_reduction` (`speculative.py:1564-1573`), and the DFlash2 speculator
reads the flag (`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:216`). The DFlash2
profiles run V2; whether `SPEC=mtp` resolves to V2 on this model
(`vllm/config/vllm.py:675-723`) is not checked here.

**Implementation sketch.** ~~Land PR #189 under the repo's two-box bar~~ — done
(2026-09-28). Remaining: (a) patch the two refusals above (GDN state planning from device
query lengths; `ALWAYS`-class varlen graph capture for FlashInfer-on-sm86 or the int8/TRITON
tier, `triton_attn.py:100` already reports `ALWAYS`, and for KVarN), then set
`"enable_adaptive_verification":true` for the D and E profiles. This is a real patch, not a
flag flip, and the D half may only be reachable on the int8/TRITON tier. (b) Keep the
launcher's explicit prefix-cache retention (already shipped in #189 —
`single-user/start_qwen.sh:524-570` pins 13056/14592 against 0.30's unset default of 0,
`vllm/config/cache.py:148`; vllm#55760's dense-for-Mamba+EAGLE resolution is not in 0.30.0,
see the header; was "the 0.30 default change, vllm#55760"). (c) Re-run the mode acceptance
tables.

**Benchmark + quality validation (Setups B, D, E).** `bash bench/run_benchmarks.sh single`
twice per mode (keep second run), C1–C8 + tok/step, 0.29 vs 0.30 on the same box in one
session; plus `bench/labd_bench.py --ctx 90000,150000` on D and E for the depth rows.
**Pass:** no mode regresses >3% decode; D or E gains ≥ 5%; `bench/quality_battery.py` per
mode within the constraint-2 budget (GSM8K ≥ −1.0 pt, PPL ≤ +2%); Bug-B and needle sweeps
clean on E. **Fallback:** if adaptive verification measurably cuts tok/step (estimator
mispredicts at depth), keep 0.30 but leave it off. The other 0.30 items are correctness/boot
fixes that are already in the pin.

**Strongest reason it might NOT help:** the refusals above mean nothing can be measured
until the GDN/backend patch exists, and that patch may be the whole cost of the row. Past
that, adaptive verification's estimator could under-draft at depth (acceptance at long
context is lower and spikier), trading verify cost for tok/step and netting ~0. The pools
being identical means the landed port itself cost nothing.


---

### T3 — MTP k=4 on the fp8 150k path (FULL CUDA graphs: not reachable on sm86)

**Mechanism.** D runs its verify PIECEWISE because FlashInfer's spec-decode path is
single-token-only (−6.6% step [REPO-MEASURED: gotcha 40 lines 807–816, was "803–810"]) and at
k=3 because k=4 crashed on 0.28.0's FlashInfer ("n=4 eventually dies, n=3 stable", −~7%
[REPO-MEASURED: docs/optimizations.md lines 399–405, was "397–403"]). The earlier text called
both gates version-sensitive because the 0.30 pin brings FlashInfer 0.6.18.post1
[REPO-MEASURED: PR #189 description, "Dependencies"]. **2026-09-29:** only the k=4 gate is.
The graph gate is structural on the pinned vLLM: 0.30.0's
`FlashInferMetadataBuilder.get_cudagraph_support` returns `UNIFORM_SINGLE_TOKEN_DECODE`
whenever trtllm-gen is unavailable, which is every GPU but SM90/SM12x/SM100+
(`vllm/v1/attention/backends/flashinfer.py:977-1012`, `vllm/utils/flashinfer.py:592-607`).
A multi-token verify batch therefore never gets a FULL graph on sm86, whatever
`cudagraph_mode` says.

**Implementation sketch.**
1. On the 0.30 image: soak `SPEC=mtp CTX=long DRAFT_TOKENS=4` with the F8 watch items
   (concurrent-garbage check from #121, still OPEN; long multi-turn traffic; `dmesg` watch
   for Xid). The crash signature was "one request finishes while another is mid-generation",
   so the soak must mix finishing/starting requests at 28–34k context.
2. If stable: ship `DRAFT_TOKENS=4` at CTX=long (`start_qwen.sh:205`); measure step time and
   tok/step. ~~test `--cudagraph-mode FULL` (or FULL_DECODE_ONLY from #55095) on the verify
   path~~. Withdrawn: FULL is unreachable on FlashInfer/sm86 (above), and #55095 is a fallback
   for non-compiled models. The only FULL-graph route for fp8-class KV on sm86 stays the
   int8/TRITON escape (gotcha 40, which reports `ALWAYS`: `triton_attn.py:100` in 0.30.0).
3. If k=4 still crashes: keep k=3, and reopen #34 with the `--no-async-scheduling` result, the
   driver version and the first engine log lines before the fault. That is the maintainer's
   ask from closing #34 as NOT_PLANNED on 2026-10-04 (for inactivity, not solved); was "the #34
   issue (still OPEN) stays the tracker".

**Field note (2026-10-04).** #160's reporter ran MTP k=4 on fp8/FlashInfer at TP=2 (3090 +
A4000, no P2P, a vLLM 0.29-based recipe) in production for two weeks with no repetition or
corruption (`docs/multi-gpu.md:285-298`). That is weak support for k=4 on FlashInfer: the TP
differs, and the report gives no Xid count. It does not replace the soak.

**Validation (Setup D).** `bench/run_benchmarks.sh single` (C1/C2 at default + greedy,
tok/step), plus `bench/labd_bench.py --ctx 60000,112000,150000 --tasks qa,summary` for the
depth curve; 4 h stability soak with request churn. **Pass:** decode +≥ 5% at C1 and at 112k
(was +≥ 8%, which assumed the graph half), zero Xid/EngineDeadError, quality within budget
(`bench/quality_battery.py long-k4 --gsm-n 200`: GSM8K ≥ 95.5; PPL ≤ +2%). **Fallback:**
`DRAFT_TOKENS=3` (was "+ FULL graphs only", not reachable).

**Strongest reason it might NOT help:** the #34 crash is unattributed (FlashInfer workspace
vs async-scheduling window) and may reproduce on 0.30 too. The earlier fallback, "only the
smaller graph-mode gain (~+6.6% step) survives", does not exist on FlashInfer/sm86. That gain
is only available by switching to the int8/TRITON tier, where int8 KV gives ~4% back in
acceptance (gotcha 40, lines 812–816).

---

### T4 — DFlash2 + fp8 at 150k (FA2-fp8 geometry relaxation, issue #153)

**Mechanism.** Today DFlash2 at >64k must take the TRITON/int8 tier (decays with depth:
−34% decode at 60k [REPO-MEASURED: gotcha 40 line 833, was "lines 825–832"]) or KVarN (F1).
The FA2-fp8 plugin proved DFlash2+fp8 on sm86 at TP=2: 103/192 tok/s [EXTERNAL-MEASURED:
club-3090 #1274 on vLLM 0.29.0, 2× 3090 PCIe, quoted in issue #153's body; was labelled
REPO-MEASURED]. Its adapter refuses TP=1 geometries: the model needs `(256,4)` target /
`(128,8)` drafter, and the adapter tested `(256,1/2)` and `(128,4)`. The maintainer's read:
the per-head work at TP=1 is identical (GQA ratios 6 and 4 at every TP), so this is "a gate
relaxation plus a correctness run, not new cubins" [REPO-MEASURED: issue #153 maintainer
comment, 2026-09-22; #153 still OPEN, and that comment marked it blocked on #148, which has
since landed, and on the adapter's geometry set].

**Implementation sketch.** Patch the plugin adapter's geometry set to admit `(256,4)` /
`(128,8)`; run its `check_full.py` on one 3090 at those shapes; port the plugin to the
0.30 image (it builds against a pinned 0.27.1 image); serve `SPEC=dflash2` +
`--kv-cache-dtype fp8` on FLASH_ATTN with FULL graphs at MAX_LEN 150000. Then measure.
This creates a "Setup D-prime": DFlash2 acceptance + the lookup lane at 150k.

**Validation (Setup D-prime, copy-focused).** `bench/run_benchmarks.sh single` (C1 cohort),
plus `bench/labd_bench.py --ctx 60000,112000,150000` with the document-copy/verbatim tasks
(the workload this tier is for) and qa/summary for the mixed check. **Pass:** copy-task
decode ≥ MTP-at-150k +30% with identical verbatim fidelity (`bench/verbatim.py` coverage
≥ baseline); mixed-task decode ≥ MTP − 5%; quality per budget
(`bench/quality_battery.py dflash2-fp8 --gsm-n 200`; PPL — fp8 KV is already the D-tier
dtype with published PPL parity, docs/long-context.md line 34, was "line 33").
**Fallback:** if the adapter author will not take the shapes, maintain the gate change as a
repo patch against the plugin with CI verifying against pinned upstream. This is close to
option (a) in issue #153's body (source-port `fa2-fp8kv.patch`, checked by
`check_upstream.py`), which the maintainer's comment favours over the prebuilt binary; the
old text called it "option (a) in the maintainer's comment".

**Strongest reason it might NOT help:** DFlash2's advantage at depth is concentrated on copy
work (F3); on mixed prose at 150k it may still lose to MTP ~2:1, so D-prime could end up a
second niche tier rather than the 150k default. The plugin also self-describes incomplete
soak/quality validation, so the correctness run is real work, not a formality.

---

### T5 — int4-per-token-head KV + MQ-3D split-KV verify (hedge 240k path) — LOSSY

**Mechanism.** vLLM's stock `int4_per_token_head` cache now works with DFlash2 on this repo
(314,915-token pool at 256k [REPO-MEASURED: docs/long-context.md lines 442–444, was
"441–449"]). The MQ-3D multi-query 3D split-KV verify kernel
(`patches/spec-decode-int4-kv-mq3d.patch` + `VLLM_INT4_MQ_3D=1`, which `alternative.sh` has
defaulted to since before this analysis, line 48, was "line 47") is the difference between ~9 and 94–108
tok/s at ~128k [a commenter's attribution in issue #60, not a controlled A/B; that issue is
CLOSED].
The captured-path A/B on the reference 3090: 2D→3D at 24k/49k/88k = 119→205 /
75→160 / 19→46 tok/s degenerate-repeat and 44→79 / 35→47 / 15→38 prose [REPO-MEASURED:
docs/spec-decode-scratch-token-units.md lines 366–378], with PPL/GSM8K parity between the
two kernel orders (8.2079/94.5 vs 8.2087/95.5, same file lines 358–364). KVarN at the same
90k reads 38.6 [REPO-MEASURED: docs/vllm-0.29.md line 127] — i.e., int4+MQ-3D is already at
KVarN's depth rate, with a flatter slope (2.42–2.5× split-KV win held at 88k) and 1.23× the
pool. If T1 stalls, this is the 240k path. (2026-10-04: T1 now has a measured ~51 tok/s at
~103k with both knobs on, WSL2, so T5 competes against that, not against the knob-off ~23.
The harnesses differ, so the two rates are not a controlled comparison.)

**Implementation sketch.** `bash single-user/alternative.sh` (`VLLM_INT4_MQ_3D=1` is its
default), `DFLASH_TOKENS=7` (15 does not fit at 256k — gotcha 47, lines 1081–1084),
MAX_LEN 240000 (its default is 256000, line 84, was "line 83"). Since #232 (36936ec) it passes
`--prefix-cache-retention-interval 0` when no KV tier is configured (lines 101–116, was
"100–115"). That
matters for multi-turn prefix reuse, not for these decode cells. The work is validation, not
code: produce the missing quality-at-depth evidence, then decide KVarN vs int4 as the shipped
CTX=huge default on quality grounds. **2026-09-29 correction:** the earlier "int4 pth has no
Hadamard/rotation" is false. The repo documents it as "dynamic per-token, per-head scales; the
int4 one with a rotation and asymmetric zero-points" (docs/long-context.md line 88). Whether
its rotation matches KVarN's Hadamard + variance normalization on outlier channels is not
established; per-token-head scaling is still coarser than KVarN's per-tile scheme.

**Validation (Setup E-prime).**
- Quality first (it gates everything): `bench/quality_battery.py int4kv-240 --gsm-n 200`
  (GSM8K ≥ 95.0 vs the 96.0–96.5 band → within −1.0 pt); perplexity vs the E baseline
  (KVarN reads 8.236 [REPO-MEASURED: docs/long-context.md line 34, was "line 33"]; pass ≤ +2%
  → ≤ 8.40. int4 KV PPL on the 33k-token battery is published, 8.257 (+0.3%) in batch mode
  (docs/long-context.md line 98), but PPL at 100k+ depth and on this DFlash2/MQ-3D serving
  path is not, so this is a measurement, not a formality); `bench/needle_test.py 150000 0.9`
  and `240000 0.9` retrieved (100k already retrieved on this path [REPO-MEASURED:
  docs/long-context.md lines 457–463]; a 240k needle already passed in batch mode, line 99);
  IFBench ≥ 77.3 if the PPL
  move is > 1%.
- Then speed: `bench/labd_bench.py --ctx 90000,150000,240000 --tasks qa,summary` + copy
  cells. **Pass:** decode ≥ KVarN-at-HEAD ×1.5 at 150k+ with the quality gates above met.
- **Fallback if quality exceeds budget:** keep KVarN for ≤128k and int4 only >128k (a
  context-tiered default the launcher can already express), or hold int4 K and raise V to
  8-bit (k4v8-style mixed mode — values are the 2-bit weak link; KVarN's own ablations
  point at V [EXTERNAL-MEASURED: KVarN repo/paper, github.com/huawei-csl/KVarN]).

**Strongest reason it might NOT help:** the unpublished perplexity at depth may fail the +2%
budget (int4 pth quantizes per token per head; its rotation is documented but not compared
with KVarN's; long-context retrieval leans on outlier keys). In that case the whole path is a
capacity feature, not a speed feature, and T1 (lossless) remains the only 240k route.

---

## 6. Ideas rejected, and why (including maintainer-rejected)

- **R1 — Marlin tile tuning / int8 activations at batch 1.** Measured, not assumed: the
  decode-shape retune (M ≤ 16, sm86) was "nothing measurable end to end" ("the remaining gap
  to peak bandwidth is the memory system's ramp on 16–92 MB reads, not the kernel")
  [REPO-MEASURED: docs/optimizations.md lines 407–412, was "405–410"], and the W4A8 tile table
  at M=2048 was +0.4% end-to-end [lines 192–197, was "190–195"];
  int8 activations buy nothing at batch size 1 [REPO-MEASURED: docs/quality.md line 49] and
  may cost ~9% decode via acceptance (issue #62, CLOSED: 127.7 → 116.0 tok/s, "not claiming as
  a result").
- **R2 — Fine-tuning the MTP head.** Done and rejected with data: KL halves, greedy top-1
  unchanged; Qwen's head is "already at the ceiling of a single-layer chain drafter"
  [REPO-MEASURED: drafter/README.md lines 29–37 (quote on 33–34); docs/optimizations.md lines
  407–409, was "405–407"].
- **R3 — Skipping the drafter when the lookup overwrites all its proposals.** Tried twice,
  loses net 6% ("the drafter is covering the positions past the end of the match")
  [REPO-MEASURED: gotcha 26].
- **R4 — n-gram chains at sampling temperature.** Fundamental (−8% C1): a point-mass
  proposal accepts with p(token); greedy-only gate shipped [REPO-MEASURED: issue #38].
- **R5 — int8 / lower-precision recurrent state.** Speed gain ~0.08 ms/step (state is ~0.7–0.8%
  of step bytes: 0.15 / 20.2 GB) for a real precision risk — backwards by any measure.
- **R6 — Fused single-launch GDN decode kernel (the SM100/SM110 pattern).** The GDN decode
  kernel already runs at ~85% of memory bandwidth; variants land within 3%
  [REPO-MEASURED: gotcha 10]. Launch overhead is hidden by CUDA graphs on the shipped
  profiles. The external 1.33× figure is Kimi Delta Attention on SM100 (B200), where the baseline is different.
- **R7 — TurboQuant KV backend (in vLLM 0.30).** The backend exists in 0.30.0
  (`vllm/v1/attention/backends/turboquant_attn.py`). KVarN's own comparison: ~2.4×
  TurboQuant's throughput, and vLLM's blog numbers show 40–52% lower throughput for the capacity
  [EXTERNAL-MEASURED: KVarN upstream README; vllm.ai/blog/2026-05-11-turboquant — not
  re-verified 2026-09-29, and this repo's `kvarn/README.md` does not carry the comparison, so
  the old "kvarn README references" label is corrected]. KVarN already wins this slot.
- **R8 — Eviction-based long-context methods (SnapKV/H2O etc.).** Forbidden by Hard
  constraint 2/3 (they drop tokens from the context). Not evaluated further.
- **R9 — Approximate/sparse attention (training-free).** Retrieval-at-depth is a hard
  requirement; training-free sparse attention routinely degrades needles, and the KV read is
  only 17–24% of step bytes (§2.4, was "15–22%") at 150k fp8 / 240k KVarN anyway, so the
  upside is capped. High risk, low ceiling.
- **R10 — EAGLE-3 / Medusa retrain.** Needs a training pipeline and Mamba-state-at-resume
  (vllm#53945, listed in the 0.30.0 release notes; see T2 for a sign that part of it was
  already in 0.29.0; was "only fixed in 0.30"). DFlash2 already occupies the block-drafter slot
  with better measured acceptance (4.80 vs 4.28 tok/step on bf16 [REPO-MEASURED:
  docs/optimizations.md lines 209–211, was "205–208"; a figure DFlash2's authors report]).
  Revisit only if T2 frees the hybrid-EAGLE path.
- **R11 — More draft depth at k=5+.** Measured: "going deeper (k=5) loses again: 106 / 105.
  k=4 is the knee" [REPO-MEASURED: docs/optimizations.md line 399, was "line 397"].
- **R12 — Bigger prefill chunks / TTFT work.** Out of scope (prefill, not decode). The earlier
  "measured not to boot above 2048" is wrong: 4096 boots at −2.4% pool, and only 8192 refuses
  under the pinned `KV_MEM` [REPO-MEASURED: gotcha 43, lines 896–915; `docs/optimizations.md`
  lines 189–191 still says 4096 fails, which gotcha 43 corrects].



---

## 7. Open questions only a real GPU run can answer (as specific experiments)

- **Q1 (T1 enabler).** Boot `SPEC=mtp CTX=huge KVARN_SHARED_VERIFY=1` under
  `ASYNC_SCHED=0` and under async (the shipped default, `start_qwen.sh:659-660`), one request
  at a time, `bench/residue_sweep.py <label>` + 30-min tool-call soak. Does the corruption
  reproduce with async off? If not, T1 is a validation-and-gating task, but the gate must then
  turn async off for CTX=huge or the shared path. The earlier "already off" premise is false
  (§2.3). If yes, the vq-plan snapshot bisect in T1 step 3 is the next experiment.
  **2026-10-04, partly answered:** on DFlash2 k=7 with async on, shared verify shows no
  collapse in any 30-trial `concurrent_collapse` soak (#262, #240), and tok/step does not
  change. The MTP question above is still open.
- **Q2 (the honest depth curve).** What are D and E decode tok/s at *true* 150k and 240k on
  the reference 3090 today, at fixed tok/step? Run `bench/labd_bench.py --ctx
  25000,60000,90000,112000,150000` (D) and `--ctx 90000,150000,200000,240000` (E) on one
  boot each, reporting step ms and tok/step separately (one `--ctx` per run, long corpus past
  65k — §5 harness note). The published 95–100 / 67–164 are short/moderate-depth numbers (see
  the baseline note at the top; the 67/164 pair is `--ctx 20000`); every target in this plan
  should be re-baselined on
  this curve. **2026-10-04, partly answered:** E at a ~103k prompt reads 23.0 tok/s knobs off
  and 51.9 with both knobs on (medians of 5 fresh boots, WSL2, 250 W; PR #262). There are still
  no 150k or 240k cells and no D depth curve. Use fresh boots per arm, not one boot (§5
  harness note).
- **Q3 (#196 contradiction).** At `CTX=long`, A/B `MTP_DRAFT_VOCAB=1` (shipped list) vs a
  rebuilt multilingual list vs `=0` (full head) on the reference 3090, en/da/code cohort
  plus a Chinese-prose cell. The reporter measured full head *faster* than the truncated
  head at 150k (80.6 vs 61–64 tok/s) — if that reproduces, the draft-vocab truncation is a
  pessimization at long context and the launcher should stop shipping it there; if it does
  not, the fix is just the corpus rebuild (T6). Status 2026-09-29: #196 OPEN; the maintainer
  replied 2026-09-27/28 and still owes exactly this `CTX=long` k=3 comparison. Status
  2026-10-04: still OPEN and still owed. A commenter proposed a third arm on 2026-10-02 (the
  truncated head plus whole scripts); the maintainer asked for it as an opt-in variant PR and
  will run the `CTX=long` MTP arm when the GPU is free.
- **Q4 (T3 enabler).** On the 0.30 image (now the *shipped* image — #189 merged, so no
  special build is needed) with the FlashInfer set `vllm==0.30.0` resolves: does
  `DRAFT_TOKENS=4` at `CTX=long` still Xid/IMA under request churn at 28–34k (#34, CLOSED as
  NOT_PLANNED on 2026-10-04 for inactivity, unresolved; reopen it on a recurrence), and does
  it pass the concurrent-garbage check from #121 (still OPEN; stock D passed 9/9 + 3/3 on the
  reference 3090 on 0.29, maintainer comment 2026-09-23)? Boot-and-soak, no code. The earlier
  second half, "does FULL (or FULL_DECODE_ONLY) graph mode on the fp8 verify pass", is dropped:
  FlashInfer/sm86 cannot capture it in 0.30.0 (F2).
- **Q5 (KVarN kernel occupancy).** Profile one decode step at 90k and 240k on E
  (`KVARN_SPEC_DEBUG=1` + nsys): what fraction of step time is the verify attention
  (as shipped: `_kvarn_fused_decode_stage1`/`_stage2` on the per-token grid; with the flag,
  `_kvarn_fused_verify_stage1` + stage-2; the old text named only the latter) vs the drafter
  vs the target GEMMs, and what is the achieved bytes/tile-visit vs the 0.114 µs/tile byte
  time (128 × 840 B ÷ 936 GB/s = 0.115 µs)? This decides whether T1 alone suffices or the T9
  tile/split retune is also needed. **2026-10-04, partly answered:** the shared verify kernel is
  spill-limited. At 100k and QLEN=8 it takes 2.5 ms per call in fp32 (102 spills) and 0.98 ms
  in fp16 (4 spills) [REPO-MEASURED: PR #262 body]. That is ~1.25 µs per tile visit in fp16
  (0.98 ms ÷ 781 tiles), ~11× the byte time, and ~16 ms of a ~52 ms step [ESTIMATE]. A whole-step
  nsys breakdown is still missing.
- **Q6 (T2 scope).** Rewritten 2026-09-29: first, does a boot with
  `"enable_adaptive_verification":true` in the speculative config refuse, as the 0.30.0 source
  implies (GDN layers are SSM, and FlashInfer/KVarN are not `ALWAYS`; §5 T2)? If it does, the
  question becomes the cost of a GDN varlen-verify patch. Only then: does it engage for MTP
  and DFlash2 at batch 1 (log the per-step draft count under a mixed workload), and does it
  help or hurt tok/step at 112k depth?
- **Q7 (int4 KV quality).** Perplexity of the int4-per-token-head cache at 100k+ depth, and
  needles at 150k/240k on the DFlash2/MQ-3D serving path (T5's gate). The 33k-token PPL (+0.3%)
  and a batch-mode 240k needle are published (docs/long-context.md lines 98–99); the at-depth
  PPL and the serving-path needles are not. The answer decides KVarN vs int4 as the shipped
  CTX=huge dtype.
- **Q8 (#107 stalls).** Do the stepping-stalls at ~190k MTP/fp8 reproduce on 0.30 (#107 still
  OPEN; the last maintainer note, 2026-09-23, asked for a no-async 0.29 run as the bisect
  step)? The earlier "does the still-open upstream vllm#50021 (or the repo's backport) cover
  the sawtooth shape" is moot: the backport has been in the series since 2026-08-17, so #107's
  stalls were seen with it applied (F8).
  Reliability, not throughput, but it gates any "run at 190k+" recommendation.
  **2026-10-04, partly answered:** a second reporter (2026-10-01, vLLM 0.29.0, one 3090, WSL2)
  hit the same wedge under ~34k-token agent traffic with a partial prefix hit on every turn.
  Removing `--enable-prefix-caching` made it go away. That points at the prefix-cache path; it
  is not yet run on 0.30 or at ~190k.

---

## Appendix A — multi-GPU (outside the primary single-card plan)

Not part of the plan (Hard constraint 3), recorded because the repo has measured it:
- **2× 3090 TP=2:** +16–35% decode at batch 1 with P2P/PCIe [REPO-MEASURED: issue #40];
  168.6–172.7 tok/s C1 on 2× 3090 NVLink [REPO-MEASURED: issue #159; the figures are not in
  #164, which is the batch-harness report]; TP=2 **−24% without P2P** on 2× RTX 4090 (not
  3090) [REPO-MEASURED: issue #190; docs/multi-gpu.md lines 193–205 (was "179–191") read
  151.1 → 118.7 C1, −21%]; 182.8 tok/s C1 after `expandable_segments:False` [REPO-MEASURED:
  docs/multi-gpu.md lines 155–161, was "141–147", earlier "122–128"]. TP=3 is invalid for this model (4 KV heads, 64 layers)
  [REPO-MEASURED: docs/multi-gpu.md lines 15–29].
- **MTP at TP>1:** int8 KV on TRITON_ATTN beats fp8/FlashInfer 1.43–1.89× on sm120
  [REPO-MEASURED: docs/multi-gpu.md lines 233–241, was "219–227", earlier "156–164"];
  **measured on Ampere since this analysis**: 2× 3090 TP=2 (patched-driver P2P, no NVLink)
  156.0 vs 90.9 tok/s, 1.72× [REPO-MEASURED: docs/multi-gpu.md line 242, was "line 228",
  earlier "line 226"; issue #217, CLOSED] — the same
  shape as sm120. TP1 int8/TRITON also passes the #121 concurrent-garbage check on the reference
  3090 (9/9 + 3/3 on vLLM 0.29, #121 comment 2026-09-23).
- **Pipeline parallelism (new since 2522ef9, d5e2a01/#236):** one field report (#160, vLLM
  0.28.0, 3090 + A4000, no P2P) found PP=2 + MTP a net loss: 19.8 tok/s single stream against
  32.9 with `SPEC=off`, and MTP acceptance 45.6% → ~8% [REPO-MEASURED: docs/multi-gpu.md lines
  263–304, was "249–277"]. Not a lever for this plan. The reporter's 2026-10 follow-up gives a
  TP=2 baseline on the same two cards: fp8/FlashInfer, MTP k=4, ~54.6 tok/s single stream and
  99–123 tok/s streaming decode across 16K–190K prompts, two weeks in production without
  repetition or corruption (lines 285–298; T3 field note).
- The second card buys a second memory system, which is exactly what batch-1 decode is
  bound by (§2.4) — hence the outsize gains; but it is not the 24 GB plan.

## Appendix B — sources

Repo (this tree): README.md; PATCHES.md; docs/{optimizations,long-context,benchmarks,
quality,gotchas,vllm-0.29,vllm-0.30,spec-decode-scratch-token-units,main-track,multi-gpu}.md;
single-user/README.md; single-user/start_qwen.sh; single-user/alternative.sh; .env.example;
kvarn/README.md; kvarn/kvarn-0.30.0.patch; kvarn/kvarn-fp16-dequant-0.30.0.patch (added
2026-10-04); patches/pinned-kv-empty-cache.patch (added 2026-10-04);
kvarn/tests/test_kvarn_fp16_dequant_torch.py; docs/reproductions/README.md; docs/install.md;
kvarn/files/vllm/v1/attention/ops/triton_kvarn_decode.py;
kvarn/files/vllm/v1/attention/backends/kvarn_attn.py; drafter/README.md;
prepare/README.md; patches/spec-decode-attn.patch; patches/triton-spec-attn-fp8-kv.patch;
bench/{quality_battery.py,run_benchmarks.sh,labd_bench.py,residue_sweep.py}; model config from
huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/raw/main/config.json (re-read 2026-09-29:
64 layers, interval 4, 24/4 heads × 256, GDN 16×128 / 48×128, vocab 248,320, 262,144
positions). Pinned vLLM source: v0.30.0 checkout (`/tmp/vllm-0.30.0`, commit ced6857) for
every `vllm/...` path cited in this pass.

GitHub (syv-ai/HyperQwen): issues #11, #25, #34, #38, #40, #52, #57, #60, #62, #73, #86,
#103, #105, #107, #121, #153, #159, #160, #164, #174, #190, #192, #194, #196; PR #42, #46,
#148, #188, #189 (merged 2026-09-28). Post-analysis: issues #195, #208, #213, #216–#218,
#221 and PRs #198–#203, #207, #212, #214, #215, #220, #222–#226 (see the re-verified note
at the top); since 2522ef9: #219, #230, #231, #232, #234, #235, #236. States as of
2026-09-29: OPEN #34, #107, #121, #153, #196, #208; CLOSED #11, #38, #40, #60, #62, #86, #159,
#164, #190, #194, #217; MERGED #189, #222. Upstream vLLM PRs: #50021 (still open), #52228
(merged 2026-09-14), #53945 (09-08), #54374 (09-04), #54646 (09-01), #55095 (09-11), #55760
(09-08, not in 0.30.0), #53835 (09-05), #55041, #55450, #58024–#58028 (open), #54282, #52789
(the last six not re-checked 2026-09-29); vLLM v0.30.0 release notes (published 2026-09-22).
Since d5e2a01 (2026-10-04 pass): PRs #240 (closed unmerged; carried by #262), #261, #262, #263
(all merged 2026-10-04); field reports #247, #255, #257, #266 (all closed 2026-10-04); the #107
comment of 2026-10-01; the #160 follow-up; the #196 comments of 2026-10-02 and 2026-10-04.
States as of 2026-10-04: OPEN #107, #121, #153, #196; CLOSED #34 (NOT_PLANNED, 2026-10-04),
#208 (completed, 2026-09-30); MERGED #189, #222, #261, #262, #263.

External: FlashInfer v0.6.18/v0.7.0 release notes (flashinfer-ai/flashinfer); SGLang
releases (sgl-project/sglang); llama.cpp (ggml-org/llama.cpp) and BeeLlama
(Godl1nk/beellama.cpp, 3090-measured KV-quant KLD ladder); ExLlamaV3
(turboderp-org/exllamav3: Qwen3.8-27B@4bpw 48.6 t/s 3090Ti; Flash-Next@3bpw+ngram 62.8
t/s); TensorRT-LLM (NVIDIA/TensorRT-LLM); LMDeploy (InternLM/lmdeploy); KVarN
(huawei-csl/KVarN); SuffixDecoding; vllm.ai/blog/2026-05-11-turboquant; ninfer-3090
(Don-Chad/ninfer-3090, comparison baseline).

