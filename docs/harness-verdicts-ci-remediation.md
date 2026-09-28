# Harness verdict conventions + CI wiring — remediation plan

Architecture review 2026-09-26, candidate 5 (Worth exploring). The full report
is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, with every claim re-verified against the
tree on 2026-09-26 (and one of the review's sub-claims softened where the
evidence didn't support it — see 1.2).

**Re-verified 2026-09-28 against upstream/main @ 2522ef9 (vLLM 0.30.0).**
22 commits landed since 1cf8665; what they changed in this doc's area:

- CI now gates on **two** bench tests: `test_prepare_state.py` joined
  `test_model_verification.py` (`patch-integrity.yml:21,23`, #195/#203) —
  and it string-splits `docker/prepare.sh`'s `state()` heredoc (`:37`), so
  the parser-of-a-parser pattern this plan criticizes has **spread**.
- Three new gate-able tests are **unwired**: `test_prepare_crash.py`
  (crash-injection, pure stdlib, proper exit; #195/#198),
  `test_kvarn_recycled_pages.py` (torch-CPU, asserts; #208/#222),
  `test_bench_sse_keepalive.py` (aiohttp, no GPU; #226).
- `concurrent_collapse.py` (GPU + live server; #208/#222) joined the
  reproducers — and prints its `COLLAPSED` verdict behind an exit 0.
- `demo_render.py` (411 lines) is gone, replaced by `bench/demo/`'s
  JS/mjs renderer (#220); `demo_capture.py` got a docstring-only change.
- The dishonest four, the exit-code inventory, and the rot inventory are
  unchanged — every line cite below re-verified against 2522ef9.

**Scope.** `bench/`'s 39 `*.py`/`*.sh` (+3 data files, +`bench/demo/`):
which are tests, which are measurements,
what their exit codes mean, and which of them gate anything in
`.github/workflows/`. **Not in scope:** the HTTP-client duplication
(candidate 3's plan), the launcher shell layer (candidate 2), the offline
GPU tools' contents.

## 1. Current state, precisely

### 1.1 The taxonomy (all 39 files + demo/, classified by reading each)

| kind | files | verdict? |
|---|---|---|
| gate-able CPU tests, **in CI** | `test_model_verification.py`, `test_prepare_state.py` (unittest, `unittest.main():66`; #195/#203) | yes |
| gate-able, **unwired** | `mq3d_capacity_property.py` (pure stdlib: argparse/itertools/random/sys — verified imports), `verbatim.py` (`_selftest` at `:100`, `__main__` at `:141-143`), `mq3d_scratch_pool_test.py` (torch-CPU; docstring: "Runs inside the image, no GPU"), `test_prepare_crash.py` (pure stdlib — builtins/hashlib/runpy/subprocess/tempfile, no torch — proper `sys.exit(1 if fails else 0)`), `test_kvarn_recycled_pages.py` (torch-CPU, assert-based, prints `OK`), `test_bench_sse_keepalive.py` (aiohttp only, no GPU) | yes, gating nothing |
| kernel tests (GPU, script-style) | `test_lookup_kernels.py`, `test_marlin_int8_asym.py:85`, `test_prefill_attn_bigpool.py:56`, `test_spec_decode_bigpool.py:71`, `test_spec_decode_fp8.py` (designed skips `:10,14`), `mq3d_layer2_oracle.py:511` | proper 0/1 exits |
| **broken as a test** | `test_spec_decode_attn.py` | prints `FAIL` (`:77`), **no `sys.exit` anywhere** |
| benchmark drivers | `run_benchmarks.sh`, `real_rep.sh`, `prefill_ab.sh`, `conc_ladder.py`, `labd_bench.py`, `labd_accept.py`, `spec_attn_ctx_scan.py`, `tune_gdn.py`, `act_calib.py` | 0 by design (measurement) |
| reproducers / issue oracles | `bugb_sweep.py`, `residue_sweep.py`, `seat_ttft.py`, `needle_test.py`, `needle_reuse.py`, `prefix_alternation.py`, `interleave_dose.py`, `replay_offload_serve.py`, `labd_soak.py`, `concurrent_collapse.py` (GPU + live server; #208) | mixed — see 1.2 |
| quality measurement | `api_smoke.py`, `quality_battery.py` | print only |
| demo/media | `demo_capture.py` (+ `bench/demo/`: `build_data.py` + a JS/mjs canvas renderer — outside the Python taxonomy; #220 replaced `demo_render.py`) | n/a |
| data / inputs | `make_long_corpus.py`, `prompts_real.jsonl`, `mq3d_layer2_verdicts{,-3090}.jsonl` | — |
| production lifecycle | `warmup.sh` (run by `single-user/qwen-server.sh:80`) | proper 0/1 |

### 1.2 The exit-code inventory, measured

- **Proper verdicts already exist** (0/1): `labd_soak.py:135`,
  `mq3d_layer2_oracle.py:511`, `needle_reuse.py:132`,
  `test_lookup_kernels.py:143`, `test_marlin_int8_asym.py:85`,
  `test_prefill_attn_bigpool.py:56`, `test_spec_decode_bigpool.py:71`,
  `verbatim.py:143` (all re-verified at the same lines, 2026-09-28).
  Upstream added three more of the right shape: `test_prepare_state.py`
  (`unittest.main():66`), `test_prepare_crash.py`
  (`sys.exit(1 if fails else 0)`), `test_kvarn_recycled_pages.py`
  (asserts, prints `OK`).
- **A four-value convention exists in embryo, undocumented**:
  `mq3d_capacity_property.py:109-131` returns 0 (holds), 1 (violated),
  2 ("REFUSING TO PASS: zero eligible vectors"), 3 (negative control
  failed). `replay_offload_serve.py:121` exits 2 on INVALID-TIER-OVERFLOW.
  The review card called these two inconsistent; on re-reading, both use 2
  for "the run is invalid, no verdict" — **they agree**; what is missing is
  the written convention and any other script following it.
- **Verdict-printing scripts that always exit 0** (the dishonest four):
  `test_spec_decode_attn.py` (prints `FAIL` at `:77`, no `sys.exit`),
  `prefix_alternation.py:294,304` (prints "this arm REPRODUCES the defect.",
  then `return 0`), `api_smoke.py:106` (prints `SUMMARY`, exits 0 even at
  0/12), `needle_test.py:68` (prints `MISSED`, exits 0). All four
  re-verified unchanged at the same lines, 2026-09-28 — and upstream
  added two more of the same shape: `concurrent_collapse.py` (prints
  `COLLAPSED`/`DONE`, no `sys.exit` anywhere) and
  `test_bench_sse_keepalive.py` (prints `success=False` on a stock vLLM,
  exits 0 anyway — no assert, no exit).
- **Measurement scripts that return 0 unconditionally, correctly**:
  `interleave_dose.py:154-165` (both arms — a dose-response curve has no
  verdict), the drivers above.
- **Designed skips**: `test_spec_decode_fp8.py:10,14` exits 0 with
  "SKIP(by design)" on sm<89 or a kernel without the fp8 path.

### 1.3 What gates CI today (verified)

- `patch-integrity.yml` gates on **two** bench tests:
  `python bench/test_model_verification.py` (`:21`) and, since #195/#203,
  `python bench/test_prepare_state.py` (`:23`). The first reaches into
  `verify.sh` by string-splitting on a literal heredoc marker
  (`test_model_verification.py:18-19`:
  `source.split("$PY - \"$MODEL\" <<'EOF'\n", 1)[1].split("\nEOF", 1)[0]`);
  a verify.sh reformat breaks CI with an IndexError, not a test failure.
  And the pattern has **spread**: `test_prepare_state.py:37` splits
  `docker/prepare.sh`'s `state()` heredoc the same way, and the unwired
  `test_prepare_crash.py:366-367` makes a third call site — three
  parsers-of-parsers where the review found one. That strengthens the
  case for healing it at the source (candidate 7's `--status` probes /
  candidate 1's apply door — still flagged there, not re-solved here).
- `patch-integrity.yml`'s second job runs `patches/check_vllm_series.sh`
  (candidate 1's surface). `docker-image.yml` builds the image; the build's
  `verify.sh --install` (`Dockerfile:39`) is the only install gate.

### 1.4 Rot inventory (all verified)

- `bench/test_lookup_kernels.py:3` tells the reader to run
  `python test_lookup_v2.py` — a renamed-away file.
- `bench/tune_gdn.py` has no docstring at all (starts with imports; only
  `docs/gotchas.md` explains it).
- `bench/mq3d_layer2_verdicts.jsonl` (11,438 B) and
  `bench/mq3d_layer2_verdicts-3090.jsonl` (11,439 B) — recorded oracle
  output, checked in twice, one byte apart (a card-specific duplicate).
- `bench/mq3d_layer2_oracle.py:50` writes its verdicts JSONL into `bench/`
  by default — running the oracle dirties the checkout.
- `bench/prefill_ab.sh:19` writes to `bench/results-prefill-ab/`, which is
  **not** in `.gitignore` (`.gitignore` covers `bench/results/` only).

## 2. The design

### 2.1 One verdict convention, written down

Codify the scheme `mq3d_capacity_property.py` already implements — it is
the right shape, it is just currently private to one script:

```text
exit 0   the check ran and passed — or the script is a measurement
         (declared so in its docstring AND in the manifest; a measurement
         that cannot run still fails)
exit 1   the check ran and the verdict is FAIL
exit 2   the run itself was invalid (no server, no data, zero eligible
         vectors, INVALID-TIER) — refuse to pass, not "fail"
exit 3   the test's own negative control failed — no verdict is trustworthy
```

Per-script changes (small, no measurement logic touched):

- `test_spec_decode_attn.py`: `sys.exit(1)` if any correctness row FAILs
  (timing section stays informational, printed).
- `api_smoke.py`: exit 1 unless all 12 feature checks pass — it is the
  feature smoke for the serving API; a regression should be loud.
- `needle_test.py`: exit 1 on MISSED.
- `prefix_alternation.py`: exit 1 when the arm REPRODUCES; the docstring
  gains the "clean dense arm proves nothing" caveat already printed at
  `:297-299`.
- `test_bench_sse_keepalive.py` (new upstream, #226): exit 1 when the
  request fails — today it prints `success=False` and exits 0 anyway.
- `concurrent_collapse.py` (new upstream, #208): exit 1 when any trial
  COLLAPSED — the printout already says so.
- `interleave_dose.py`, `conc_ladder.py`, `labd_bench.py`, the drivers:
  stay 0 — docstring line "measurement, not a gate" each.

### 2.2 CI wiring (seconds, no GPU)

- `patch-integrity.yml`'s model-verification job adds:
  `python bench/mq3d_capacity_property.py` (exit 0 = property holds),
  `python bench/mq3d_capacity_property.py --mutate seq-rows` (exit 0 via
  its NEGATIVE CONTROL OK path at `:112-118` — proves the test can still
  detect the bug it was written for), and `python bench/verbatim.py`
  (exit 1 if its `_selftest` misclassifies any of the 9 canned shapes).
  All pure stdlib, seconds.
- `test_prepare_crash.py` is the other natural addition (pure stdlib,
  seconds, proper exit) — but it string-splits `prepare.sh` exactly the
  way `test_prepare_state.py:37` does, so wiring it doubles down on the
  heredoc-split; note the coupling in the manifest.
- `mq3d_scratch_pool_test.py` needs torch + the patched vLLM but no GPU —
  it runs in the image build (either a `RUN` line after `verify.sh
  --install` in the Dockerfile, or a post-build step in
  `docker-image.yml`; choice flagged to the owner — the Dockerfile line is
  the same gate, the workflow step is easier to skip).
- Hardening the heredoc-split tests is **not** in this plan: the
  string-split (now three call sites — `test_model_verification.py:18-19`,
  `test_prepare_state.py:37`, `test_prepare_crash.py:366-367`) is healed
  by candidate 7's `--status` probes / candidate 1's apply door; flagged
  there, not re-solved here.

### 2.3 The manifest — one table, human-maintained

`bench/README.md` (new, short): one row per script — **script · kind ·
needs (CPU / image / GPU / server / data) · exit semantics · one-liner**.
The taxonomy in 1.1 is the seed; 39 rows (+ one `demo/` line), most of
them one line. This is the answer to "is this a test?" being re-derived
from 39 docstrings by every reader (both explorer agents independently spent most of their time
on exactly that re-derivation). A lint test that checks every `.py`/`.sh`
in `bench/` appears in the table is cheap and optional; the table alone is
the 80%.

### 2.4 Hygiene (rides along, each its own line in the PR)

- Fix `test_lookup_kernels.py:3`'s stale name.
- `tune_gdn.py` gains a 3-line docstring (what it sweeps, what a row means).
- Delete `mq3d_layer2_verdicts-3090.jsonl` (the duplicate), move the
  remaining verdicts file next to the oracle's documentation, or leave and
  mark in the manifest — owner pick; default: delete the dup, keep one.
- `mq3d_layer2_oracle.py`'s default output goes to `bench/results/`
  (gitignored) instead of the tree (`:50`).
- Add `bench/results-prefill-ab/` to `.gitignore`.
## 3. Rollout

Two PRs, both cheap and independently revertable.

### 3.1 PR A — the convention + honest exits + manifest

Write the convention into `bench/README.md` (with the manifest seeded from
1.1); change the six dishonest scripts' exits (2.1 — the original four
plus #226's and #208's); the hygiene lines (2.4). No CI change — the
repo's behavior is unchanged except that six scripts now tell the truth
to `$?`.

Acceptance: `bash bench/test_spec_decode_attn.py` (on a GPU box) exits 1
when its correctness check FAILs; `python bench/api_smoke.py` exits 1 at
<12/12 and 0 at 12/12; the manifest covers all 39 files (+ `demo/`).

### 3.2 PR B — wire the CPU tests into CI

`patch-integrity.yml` gains the three stdlib runs (2.2);
`mq3d_scratch_pool_test.py` goes into the image gate. After this PR, five
bench tests gate instead of the two that do today (#203 added the second
upstream) — and each gates with its negative control intact.

Acceptance: the workflow log shows all three new runs green in seconds;
a scratch PR that breaks the capacity property (edit the formula in
`mq3d_capacity_property.py:25-27`) fails the job — the negative control
proves the gate bites.

## 4. Test plan

| test | how | proves |
|---|---|---|
| convention self-check | run each changed script in its pass and fail shapes on a live box once (runbook in the PR) | the new exits match the printed verdicts |
| negative controls | `mq3d_capacity_property.py --mutate seq-rows` must exit 0 via NEGATIVE CONTROL OK; edit verbatim.py's classifier to misclassify one shape, `_selftest` must exit 1 | the tests can still detect their bugs |
| CI | the workflow run on the PR itself | seconds-level, no GPU |
| no measurement regressions | `labd_bench.py`, `conc_ladder.py`, `interleave_dose.py` still exit 0 after printing | drivers stay drivers |
| manifest coverage | optional lint: every `bench/*.py`/`*.sh` named in `bench/README.md` | the table can't silently rot |

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| `api_smoke.py` as a gate flakes on intentional feature changes | medium | it is wired as a *manual* gate first (documented in the manifest), CI promotion only after a few clean weeks; the exit-code change itself is the PR, the CI wiring is a later decision |
| The convention is ignored by the next new script | medium | the manifest's "exit semantics" column makes the convention the path of least resistance; the optional lint test catches an unlisted script |
| Quarantining the verdicts JSONL loses a reproduction artifact | low | one file is kept; `docs/reproductions/` is the repo's existing home for these |
| Exit-code changes break a wrapper that ignored failures on purpose | low | the only production consumer of a bench script is `qwen-server.sh:80` → `warmup.sh`, which already has proper exits and is untouched |
| The manifest rots | medium | it starts minimal (39 one-line rows); the lint test is the cheap guard; without it, rot is no worse than today's 39-docstring archaeology |

## 6. Done when

1. `bench/README.md` lists every file in `bench/` with kind, needs, and
   exit semantics.
2. The six dishonest scripts' exits match their printed verdicts (the
   review's four, plus #226's and #208's).
3. `patch-integrity.yml` runs `test_model_verification.py` and
   `test_prepare_state.py` (both already gate, #203), plus
   `mq3d_capacity_property.py` (+ `--mutate seq-rows`) and `verbatim.py`;
   the image build runs `mq3d_scratch_pool_test.py`.
4. `.gitignore` covers `bench/results-prefill-ab/`; the duplicate verdicts
   file is gone; `test_lookup_kernels.py` names itself.
5. Five bench tests gate CI where two did (`test_prepare_state.py`
   joined upstream via #203) — and every gate carries its negative
   control.
