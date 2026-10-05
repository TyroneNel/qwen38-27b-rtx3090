# Harness verdict conventions + CI wiring — remediation plan

Architecture review 2026-09-26, candidate 5 (Worth exploring). The full report
is [architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card, with every claim re-verified against the
tree on 2026-09-26 (and one of the review's sub-claims softened where the
evidence didn't support it — see 1.2).

> **Updated 2026-10-05 on upstream/main @ 10bb488.** PR A is in review as #279,
> and PR B is in review in three parts. #271 adds `test_no_key_bind.sh`, `mq3d_capacity_property.py` (with
> `--mutate seq-rows`) and `verbatim.py` to `model-verification`. #277
> (`77b4695`) adds a `prepare-crash` job that runs `test_prepare_crash.py`.
> #278 (`389e607`) runs `mq3d_scratch_pool_test.py` in the image build.
> - The 2.2 pip set was incomplete. #277 installs CPU torch, then
>   transformers, tokenizers, compressed-tensors and huggingface_hub at the
>   pins in `docker/requirements.txt`, then `safetensors` and `psutil`.
>   compressed-tensors 0.17.0 imports `psutil` at load time
>   (`compressed_tensors/offload/load.py:11`) but does not declare it. The
>   image gets it from vLLM. Without it the test exits 1 in 7 s.
> - pip keeps the CPU wheel: torch 2.14.1+cpu, no nvidia packages. In a fresh
>   Python 3.12.13 venv the install took 46 s, and the test exited 0 with 168
>   cases and 0 failures in 481 s on 6 cores. The job took 1m35s on the #277
>   runner, 47 s of it in the test step, CI green.
> - `prepare-crash` is a job of its own, not a step in `kvarn-torch-gate`, so
>   a prepare failure does not stop the kvarn test. It merges cleanly with
>   #271 and #275.
> - A scratch copy whose `write_json` writes in place fails the `mtp` step
>   (exit 1, 2 failures). The unchanged `mtp` step passes 24 cases.
> - #278 adds `venv/bin/python bench/mq3d_scratch_pool_test.py` at the end of
>   the Dockerfile `RUN` that runs `verify.sh --install` (`Dockerfile:38`).
>   Under `set -e` a failed assertion stops the build. Of the two places in
>   2.2, #278 takes the Dockerfile line: it runs in local builds too. With
>   #278, `verify.sh --install` moves `Dockerfile:34`→`:37`.
> - On the #278 runner, build-push took 5m21s, CI green. The test printed
>   `POOL UNIT TEST OK: 9 assertions` about 8 s after `verify: OK (0
>   failures)`.
> - A scratch copy of the patch without the exclusive guard in
>   `mq3d_scratch_acquire` (`if fits:`) fails the #278 build at the test's
>   exclusive assertion (exit 1). Main's Dockerfile builds the same tree,
>   and `verify.sh` reports 0 failures. Both builds ran on a host with no
>   GPU, with build-push's registry cache.
> - The test's `NEGATIVE CONTROL` line prints and does not assert. The
>   assertion at `:36` checks the same case.
> - All of PR B is in review.
> - PR A is #279 (`450e507`, CI green). It changes no CI, and no open PR
>   touches a file it changes.
> - Eleven scripts printed a failing verdict and exited 0 on main, not six.
>   A check of each script in `bench/` found five more:
>   `test_spec_decode_fp8.py` (a FAIL row), `bugb_sweep.py` (a broken
>   length), `residue_sweep.py` (a broken residue), `replay_offload_serve.py`
>   (NOT-SERVED) and `seat_ttft.py` (every request failed). #279 makes ten
>   of them exit 1 on that verdict, and `seat_ttft.py` exit 2.
> - `bench/README.md` has the exit table, four rules and 45 rows: the 44
>   entries in `bench/` (`demo/` and the data files included) and the kvarn
>   test.
> - The two verdicts files are not a copy, so #279 keeps both. They have
>   the same rows and verdicts, but they are two runs. The first is the 4090
>   run (patch sha `c2b5a002…`). The second is the 3090 rerun, with the
>   shipped patch's sha (`562f7e28…`). Their `int4_per_token_head.py` hashes
>   differ, and so do three `ref_max_abs_2d` and two `ref_max_abs_3d`
>   values. `docs/spec-decode-scratch-token-units.md` cites the first at
>   `:145` and the second at `:191`. The README says which is which.
> - The oracle still writes beside itself by default. `ORACLE_OUT` already
>   sets another path (`mq3d_layer2_oracle.py:46-50`).
>   `docs/spec-decode-scratch-token-units.md:157-161` says the default
>   rewrites the verdict file beside the oracle, and that these lines are
>   the only change from the 4090 author's script, apart from four wording
>   edits. A move to `bench/results/` would make both statements false.
> - The drivers get no "measurement, not a gate" docstring line. The
>   README's kind column says it for each one. `tune_gdn.py`'s new
>   docstring and `seat_ttft.py:19` also say it.
> - The checks ran on a host with no GPU, on the branch and on main. A
>   stdlib stub server ran 20 cases of the server scripts. The SSE test ran
>   3 cases in `ghcr.io/syv-ai/hyperqwen:latest`. A CPU harness with a fake
>   kernel ran 6 cases of the two spec-decode tests. It checks the exit
>   wiring, not the kernels. Each fail case exits 0 on main and non-zero on
>   the branch. Each pass case exits 0 on both, and INVALID-TIER-OVERFLOW
>   exits 2 on both.
> - No caller reads the exits that #279 changes. `verify.sh:405` and
>   `single-user/start_qwen.sh:598` name `bugb_sweep.py` and
>   `residue_sweep.py` in comments only.
> - Known gaps, not fixed in #279. `prefix_alternation.py` exits 0 when it
>   checked no turn (`--rounds 1`, or the budget runs out before round 2).
>   With no server, `api_smoke.py` exits 1, not 2, because each check
>   reports the connection error as a FAIL. `real_rep.sh:23` writes
>   `/tmp/rr_$TAG_$i.log`. Bash reads `$TAG_`, which is not set, so every
>   tag writes `/tmp/rr_<i>.log`. The tracker lists it as D3.
> - Corrected below for PR A: the Status line, the `test_spec_decode_fp8.py`
>   entry in 1.1, a new 1.2 note, the two 1.4 lines on the verdicts files
>   and the oracle, 2.1, 2.4, 3.1, 4, and done-when items 1, 2 and 4.
> - Corrected below: the Status line, the two 2026-10-04 notes in 2.2, the
>   2.2 image-build note, the 3.2 revised scope and acceptance, and
>   done-when items 3 and 5.

**Re-verified 2026-10-04 against upstream/main @ e371b42 (vLLM 0.30.0).**

**Status:** Not started. Next in sequence (the cheapest card; urgency up). Ship PR B first, with test_no_key_bind.sh. (2026-10-05: PR B is in review as #271, #277 and #278, and PR A as #279; see the top block.)

Thirteen commits landed since d5e2a01. Four of them touch this doc's area
(`git diff --stat d5e2a01 e371b42 -- .github bench kvarn/tests .gitignore
Dockerfile docker/prepare.sh`; `git log` per path): bd6c5e2 (#262) adds a
CI job and `kvarn/tests/`; 177ce26 (#237) adds `bench/test_no_key_bind.sh`;
accc8cf (#246) extends `test_prepare_crash.py`; e1459c7 replaces the
Dockerfile's apply loop with `patches/apply.sh`. `.gitignore` and
`docker/prepare.sh` do not change. Nothing from PR A or PR B landed: there
is no `bench/README.md`, the six dishonest exits do not change, and CI runs
none of this plan's tests.

- **CI gates at e371b42** (`patch-integrity.yml`): `:21`
  `test_model_verification.py` and `:23` `test_prepare_state.py`
  (unchanged). **NEW:** job `kvarn-torch-gate` (`:25-37`, bd6c5e2/#262)
  installs the CPU torch wheel (`:35`), then runs
  `kvarn/tests/test_kvarn_fp16_dequant_torch.py` (`:37`). The `git-apply`
  job runs `patches/check_vllm_series.sh` at `:60`. The image's install
  gate is `verify.sh --install` at `Dockerfile:34` (was `:39`). It now runs
  after `bash patches/apply.sh` (e1459c7). Three tests gate CI, not two.
- **Taxonomy:** 39 → 40 `*.py`/`*.sh` in `bench/`. The 3 data files and
  `bench/demo/` do not change (`git ls-tree e371b42 bench/`). The new file
  is `test_no_key_bind.sh` (#237). It is a gate-able CPU test, and it is
  unwired. It needs bash, python3 and tar, and it puts a stub `vllm` in a
  temp copy of the checkout. It needs no GPU and no torch. It ran green here
  in 0.4 s (22 PASS lines, exit 0). It exits 1 on any FAIL. Under
  `/.dockerenv` it prints `skip:` and exits 0 (`:10`). Outside `bench/`,
  the new `kvarn/tests/test_kvarn_fp16_dequant_torch.py` is a CPU-torch
  test in CI. It exits 0 or 1 (`:220-229`).
- **Exit inventory:** unchanged. The six dishonest exits are at the same
  lines. The exit-2 cites (`mq3d_capacity_property.py:109-131`,
  `replay_offload_serve.py:121`) hold. Both new tests follow 2.1's 0/1
  split. Their skip-as-0 matches `test_spec_decode_fp8.py:10,14`. Neither
  uses exit 2: `test_no_key_bind.sh` reports a launcher that prints no
  `--host` as FAIL (1), not as an invalid run. Neither has a negative
  control.
- **Rot inventory:** all five items are unchanged, at the same lines.
- **Moved cites (re-cited in place below):** `test_prepare_crash.py` gains
  a `reject` step (#246, +75/−6). Its imports move `:356-358`→`:421-423`,
  its heredoc split `:366-367`→`:431-432`, its exit `:377`→`:446`.
  `verify.sh` changes in three commits (+29/−19). The heredoc that
  `test_model_verification.py:19` splits moves `:132-280`→`:133-281`.
  `fi  # INSTALL` moves `:338`→`:348`. `Dockerfile:39` moves to `:34`.
- **New parse-another-script site:** `test_no_key_bind.sh:33-35` cuts
  `verify.sh`'s key-check block (`verify.sh:331`) out with
  `sed -n "/^if \[ -s api_key.txt \] || \[ -n/,/^fi\$/p"` and `eval`s it.
  That makes four sites (1.3).
- **Re-run here (CPU, Python 3.13, e371b42):** `mq3d_capacity_property.py`
  exits 0 (3.9 s). `--mutate seq-rows` exits 0 through `NEGATIVE CONTROL
  OK` (4.0 s). `verbatim.py` exits 0 (0.1 s). Both CI bench tests exit 0
  (0.3 s each). `test_prepare_crash.py` and the kvarn test were not run.

**Re-verified 2026-09-29 against upstream/main @ d5e2a01 (vLLM 0.30.0).**
Seven commits landed since 2522ef9 (#219, #230, #231, #232, #234, #235,
#236); none touches
`bench/`, `.github/`, `.gitignore`, `Dockerfile` or `docker/prepare.sh`
(`git diff --stat 2522ef9 HEAD -- bench/ .github/ .gitignore` is empty;
each commit's `git show --stat` read):

- **Unchanged upstream:** the two CI gates (`patch-integrity.yml:21,23`),
  the 39 `*.py`/`*.sh` + 3 data files + `bench/demo/` count (`git ls-tree
  HEAD bench/`), the six dishonest exits, the exit-2 convention, and the
  whole rot inventory — every line cite below re-read at d5e2a01 and
  still at the same line.
- **#219 (8cf642e) adds no test and no CI step.** It adds a `Makefile`
  (`verify-install` runs `bash verify.sh --install`, nothing in `bench/`),
  `scripts/hq-doctor.sh` (no bench call), and `verify.sh --wait`. The
  `--wait` hunks sit in the header/arg-parse and in the live section after
  `fi  # INSTALL` (`verify.sh:338`); the heredoc that
  `test_model_verification.py:19` splits (`verify.sh:132-280`) is still
  the first `$PY - "$MODEL" <<'EOF'` and is untouched, so no claim here
  changes (it only shifted 19 lines down, from `:113`).
- **Corrected at this pass (errors carried since 2522ef9, not upstream
  changes):** `test_prepare_crash.py` is **not** pure stdlib — `main()`
  imports torch, safetensors, tokenizers, huggingface_hub,
  compressed_tensors and transformers (`:356-358`) and its docstring says
  "CPU and Linux only, minutes"; `test_kvarn_recycled_pages.py` loads
  `kvarn_attn.py`, which imports vLLM (`kvarn_attn.py:47`);
  `test_bench_sse_keepalive.py` imports vLLM's
  `endpoint_request_func` by default (`:34`). All three are image-gate
  candidates, not stdlib-CI candidates (1.1, 2.2 fixed). `.gitignore`
  also covers `bench/quality-data/` (1.4 fixed). The 3.1 acceptance line
  ran a Python file with `bash` (fixed).

**Re-verified 2026-09-28 against upstream/main @ 2522ef9 (vLLM 0.30.0).**
22 commits landed since 1cf8665; what they changed in this doc's area:

- CI now gates on **two** bench tests: `test_prepare_state.py` joined
  `test_model_verification.py` (`patch-integrity.yml:21,23`, #195/#203) —
  and it string-splits `docker/prepare.sh`'s `state()` heredoc (`:37`), so
  the parser-of-a-parser pattern this plan criticizes has **spread**.
- Three new gate-able tests are **unwired**: `test_prepare_crash.py`
  (crash-injection, proper exit; #195/#198 — said "pure stdlib" at
  2522ef9; it needs torch + the prepare stack, see the 2026-09-29 note),
  `test_kvarn_recycled_pages.py` (torch-CPU + vLLM import, asserts;
  #208/#222), `test_bench_sse_keepalive.py` (aiohttp + vLLM's bench
  module, no GPU; #226).
- `concurrent_collapse.py` (GPU + live server; #208/#222) joined the
  reproducers — and prints its `COLLAPSED` verdict behind an exit 0.
- `demo_render.py` (411 lines) is gone, replaced by `bench/demo/`'s
  JS/mjs renderer (#220); `demo_capture.py` got a docstring-only change.
- The dishonest four, the exit-code inventory, and the rot inventory are
  unchanged — every line cite below re-verified against 2522ef9 (and
  again at d5e2a01, above).

**Scope.** `bench/`'s 40 `*.py`/`*.sh` (+3 data files, +`bench/demo/`;
39 at d5e2a01, `test_no_key_bind.sh` joined at e371b42), plus
`kvarn/tests/` (one file, new at e371b42) (2026-10-04):
which are tests, which are measurements,
what their exit codes mean, and which of them gate anything in
`.github/workflows/`. **Not in scope:** the HTTP-client duplication
(candidate 3's plan), the launcher shell layer (candidate 2), the offline
GPU tools' contents.

## 1. Current state, precisely

### 1.1 The taxonomy (all 40 files + demo/, classified by reading each; 39 at d5e2a01)

| kind | files | verdict? |
|---|---|---|
| gate-able CPU tests, **in CI** | `test_model_verification.py`, `test_prepare_state.py` (unittest, `unittest.main():66`; #195/#203); outside `bench/`: `kvarn/tests/test_kvarn_fp16_dequant_torch.py` (CPU torch, no vLLM; `patch-integrity.yml:37`; #262, 2026-10-04) | yes |
| gate-able, **unwired** | `test_no_key_bind.sh` (2026-10-04, #237: bash + python3 + tar, stub `vllm`, no GPU, no torch; exit 1 on FAIL, `skip:` + exit 0 under `/.dockerenv` at `:10`; ran green here in 0.4 s), `mq3d_capacity_property.py` (pure stdlib: argparse/itertools/random/sys — verified imports), `verbatim.py` (no top-level imports, `import sys` only under `__main__`; `_selftest` at `:100`, `__main__` at `:141-143`), `mq3d_scratch_pool_test.py` (torch-CPU + `import vllm`; docstring: "Runs inside the image, no GPU"), `test_prepare_crash.py` (proper `sys.exit(1 if fails else 0)` at `:446`, was `:377`; **not** stdlib — said "pure stdlib … no torch" at 2522ef9, but `main()` imports torch, safetensors, tokenizers, huggingface_hub, compressed_tensors, transformers at `:421-423`, was `:356-358`; no vLLM import; docstring: "CPU and Linux only, minutes"), `test_kvarn_recycled_pages.py` (torch-CPU, loads `kvarn_attn.py` which imports vLLM; assert-based, prints `OK` at `:63`), `test_bench_sse_keepalive.py` (aiohttp + vLLM's `vllm.benchmarks.lib.endpoint_request_func` by default, `:34`; no GPU — said "aiohttp only" at 2522ef9) | yes, gating nothing |
| kernel tests (GPU, script-style) | `test_lookup_kernels.py`, `test_marlin_int8_asym.py:85`, `test_prefill_attn_bigpool.py:56`, `test_spec_decode_bigpool.py:71`, `test_spec_decode_fp8.py` (designed skips `:10,14`; 2026-10-05: not a proper exit, it printed FAIL rows and exited 0, fixed in #279), `mq3d_layer2_oracle.py:511` | proper 0/1 exits |
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
- (2026-10-05) The inventory missed five. `test_spec_decode_fp8.py`
  prints a FAIL row and exits 0. `bugb_sweep.py` and `residue_sweep.py`
  print a broken length or residue and exit 0. `replay_offload_serve.py`
  exits 2 on INVALID-TIER-OVERFLOW, but 0 on NOT-SERVED. `seat_ttft.py`
  exits 0 when every request failed. That makes eleven, and #279 fixes
  all eleven (see the top block).
- **Measurement scripts that return 0 unconditionally, correctly**:
  `interleave_dose.py:154-165` (both arms — a dose-response curve has no
  verdict), the drivers above.
- **Designed skips**: `test_spec_decode_fp8.py:10,14` exits 0 with
  "SKIP(by design)" on sm<89 or a kernel without the fp8 path.

### 1.3 What gates CI today (verified)

- `patch-integrity.yml` gates on **three** tests (2026-10-04; it said
  "two" through d5e2a01): two bench tests,
  `python bench/test_model_verification.py` (`:21`) and, since #195/#203,
  `python bench/test_prepare_state.py` (`:23`), and, since #262, the
  `kvarn-torch-gate` job (`:25-37`). That job installs the CPU torch wheel
  (`:35`) and runs `kvarn/tests/test_kvarn_fp16_dequant_torch.py` (`:37`).
  The first bench test reaches into
  `verify.sh` by string-splitting on a literal heredoc marker
  (`test_model_verification.py:18-19`:
  `source.split("$PY - \"$MODEL\" <<'EOF'\n", 1)[1].split("\nEOF", 1)[0]`;
  the heredoc is `verify.sh:133-281` at e371b42, was `:132-280`);
  a verify.sh reformat breaks CI with an IndexError, not a test failure.
  And the pattern has **spread**: `test_prepare_state.py:37` splits
  `docker/prepare.sh`'s `state()` heredoc the same way, and the unwired
  `test_prepare_crash.py:431-432` (was `:366-367`) makes a third call
  site. (2026-10-04) The unwired `test_no_key_bind.sh:33-35` makes a
  fourth: it cuts the key-check block (`verify.sh:331`) out of `verify.sh`
  with a `sed -n` range and `eval`s it. Four parsers-of-parsers where the
  review found one. That strengthens the
  case for healing it at the source (candidate 7's `--status` probes /
  candidate 1's apply door — still flagged there, not re-solved here).
- `patch-integrity.yml`'s `git-apply` job (the third job since #262; it
  was the second) runs `patches/check_vllm_series.sh` (`:60`)
  (candidate 1's surface). `docker-image.yml` builds the image; the build's
  `verify.sh --install` (`Dockerfile:34`, was `:39`; it now runs after
  `bash patches/apply.sh`, e1459c7) is the only install gate.

### 1.4 Rot inventory (all verified)

- `bench/test_lookup_kernels.py:3` tells the reader to run
  `python test_lookup_v2.py` — a renamed-away file.
- `bench/tune_gdn.py` has no docstring at all (starts with imports; only
  `docs/gotchas.md` explains it).
- `bench/mq3d_layer2_verdicts.jsonl` (11,438 B) and
  `bench/mq3d_layer2_verdicts-3090.jsonl` (11,439 B) — recorded oracle
  output, checked in twice, one byte apart (a card-specific duplicate).
  (2026-10-05) Not a duplicate. They are the 4090 run and the 3090 rerun.
  Their patch and helper hashes differ, and so do five reference values.
  See the top block.
- `bench/mq3d_layer2_oracle.py:50` writes its verdicts JSONL into `bench/`
  by default — running the oracle dirties the checkout. (2026-10-05) This
  is by design, and #279 keeps it. See the top block.
- `bench/prefill_ab.sh:19` writes to `bench/results-prefill-ab/`, which is
  **not** in `.gitignore` (`.gitignore:3-4,16` covers `bench/quality-data/`
  and `bench/results/`; said "`bench/results/` only" at 2522ef9 —
  `bench/quality-data/` has been there since ee4d48f, 2026-08-18).

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
- (2026-10-04) The two new tests need no change. `test_no_key_bind.sh`
  and `kvarn/tests/test_kvarn_fp16_dequant_torch.py` exit 0 on pass and 1
  on fail. `test_no_key_bind.sh`'s container skip exits 0, as
  `test_spec_decode_fp8.py:10,14` does. Neither uses exit 2 or exit 3.
- (2026-10-05) #279 also fixes the five scripts that 1.2 now names. Ten
  of the eleven exit 1 on the verdict. `seat_ttft.py` exits 2, because
  every request failing is an invalid run. The drivers get no docstring
  line. The manifest's kind column says it for each one.

### 2.2 CI wiring (seconds, no GPU)

- `patch-integrity.yml`'s model-verification job adds:
  `python bench/mq3d_capacity_property.py` (exit 0 = property holds),
  `python bench/mq3d_capacity_property.py --mutate seq-rows` (exit 0 via
  its NEGATIVE CONTROL OK path at `:112-118` — proves the test can still
  detect the bug it was written for), and `python bench/verbatim.py`
  (exit 1 if its `_selftest` misclassifies any of the 9 canned shapes).
  All pure stdlib, seconds. (2026-10-04) All three ran green here at
  e371b42 (CPU, Python 3.13): 3.9 s, 4.0 s and 0.1 s.
- (2026-10-04) The same job adds `bash bench/test_no_key_bind.sh` (#237).
  It is the regression test for the no-key bind to 127.0.0.1 (#204). It
  needs bash, python3 and tar, and no GPU, torch or vLLM. It ran green
  here in 0.4 s (22 PASS lines). A GitHub runner has no `/.dockerenv`, so
  the test runs there and does not skip [INFERENCE: not run on a runner].
  (2026-10-05: on the #271 runner it ran and did not skip.)
  It is the fourth parse-another-script site (1.3); note the coupling to
  `verify.sh:331` in the manifest.
- (2026-10-04) `test_prepare_crash.py` moves out of the image gate and
  into the CPU-torch job (`kvarn-torch-gate`, `patch-integrity.yml:25-37`,
  or a sibling job). It imports no vLLM. Its imports are torch,
  safetensors, tokenizers, huggingface_hub, compressed_tensors and
  transformers (`:421-423`, was `:356-358`). The `draft` step passes
  `--ids`, so `prepare/build_draft_vocab.py:48`'s lazy pyarrow import does
  not run. The job's step is the existing CPU torch install (`:35`), then
  `pip install safetensors tokenizers huggingface_hub compressed-tensors
  transformers`, then `python bench/test_prepare_crash.py`
  [INFERENCE: not run — the pip set, that pip keeps the CPU torch wheel,
  and the runtime on a runner are unchecked; the docstring says "minutes"].
  (2026-10-05: checked in #277. The set also needs `psutil`, and pins come
  from `docker/requirements.txt`. pip keeps the CPU wheel. The job takes
  1m35s on the runner. See the top block.)
  It also string-splits `prepare.sh` (`:431-432`, was `:366-367`) exactly
  the way `test_prepare_state.py:37` does, so wiring it doubles down on the
  heredoc-split; note the coupling in the manifest.
- `test_kvarn_recycled_pages.py` and `test_bench_sse_keepalive.py` stay
  image-gate candidates: both import vLLM (`kvarn_attn.py:47`, still at
  that line; `test_bench_sse_keepalive.py:34`). The
  CPU torch wheel does not unblock them (2026-10-04).
- `mq3d_scratch_pool_test.py` needs torch + the patched vLLM but no GPU —
  it runs in the image build (either a `RUN` line after `verify.sh
  --install` in the Dockerfile, or a post-build step in
  `docker-image.yml`; choice flagged to the owner — the Dockerfile line is
  the same gate, the workflow step is easier to skip). (2026-10-05: #278
  takes the Dockerfile line, at `Dockerfile:38`; see the top block.)
- Hardening the heredoc-split tests is **not** in this plan: the
  string-split (now four call sites (2026-10-04) — `test_model_verification.py:18-19`,
  `test_prepare_state.py:37`, `test_prepare_crash.py:431-432`,
  `test_no_key_bind.sh:33-35`) is healed
  by candidate 7's `--status` probes / candidate 1's apply door; flagged
  there, not re-solved here.

### 2.3 The manifest — one table, human-maintained

`bench/README.md` (new, short): one row per script — **script · kind ·
needs (CPU / image / GPU / server / data) · exit semantics · one-liner**.
The taxonomy in 1.1 is the seed; 40 rows (+ one `demo/` line; 39 at
d5e2a01), most of them one line. (2026-10-04) Add one row for
`kvarn/tests/test_kvarn_fp16_dequant_torch.py`. It lives outside `bench/`,
but it is a CI gate, so the table names it. This is the answer to "is this a test?" being re-derived
from 40 docstrings by every reader (both explorer agents independently spent most of their time
on exactly that re-derivation). A lint test that checks every `.py`/`.sh`
in `bench/` appears in the table is cheap and optional; the table alone is
the 80%.

### 2.4 Hygiene (rides along, each its own line in the PR)

- Fix `test_lookup_kernels.py:3`'s stale name.
- `tune_gdn.py` gains a 3-line docstring (what it sweeps, what a row means).
- Delete `mq3d_layer2_verdicts-3090.jsonl` (the duplicate), move the
  remaining verdicts file next to the oracle's documentation, or leave and
  mark in the manifest — owner pick; default: delete the dup, keep one.
  (2026-10-05) Neither file is deleted. They are two runs (1.4). #279
  keeps both, and the manifest says which run each one is.
- `mq3d_layer2_oracle.py`'s default output goes to `bench/results/`
  (gitignored) instead of the tree (`:50`). (2026-10-05) Not done. #279
  keeps the default. See the top block.
- Add `bench/results-prefill-ab/` to `.gitignore`.
- (2026-10-05) #279 has the `.gitignore` line, the `test_lookup_kernels.py`
  name and the `tune_gdn.py` docstring.
## 3. Rollout

Two PRs, both cheap and independently revertable.

(2026-10-04) Ship PR B first. It changes no exit code: every test it
wires already exits 0/1. It also wires `test_no_key_bind.sh`, a security
regression test that has no gate today.

### 3.1 PR A — the convention + honest exits + manifest

Write the convention into `bench/README.md` (with the manifest seeded from
1.1); change the six dishonest scripts' exits (2.1 — the original four
plus #226's and #208's); the hygiene lines (2.4). No CI change — the
repo's behavior is unchanged except that six scripts now tell the truth
to `$?`.

Acceptance: `venv/bin/python bench/test_spec_decode_attn.py` (on a GPU
box; its docstring's invocation — this line said `bash` at 2522ef9) exits 1
when its correctness check FAILs; `python bench/api_smoke.py` exits 1 at
<12/12 and 0 at 12/12; the manifest covers all 40 files (+ `demo/`; 39
at d5e2a01) and names `kvarn/tests/test_kvarn_fp16_dequant_torch.py`
(2026-10-04).

(2026-10-05) In review as #279. It fixes eleven scripts, not six (1.2).
The manifest has 45 rows: the 44 entries in `bench/` and the kvarn test.
Against a stub server, `api_smoke.py` exits 1 at 11/12 and 0 at 12/12.
With a fake kernel on CPU, `test_spec_decode_attn.py` exits 1 on a FAIL
row and on a NaN row. A GPU run of the two spec-decode tests is not done.
On a GPU, both should exit 0, with 13 and 19 OK rows.

### 3.2 PR B — wire the CPU tests into CI

`patch-integrity.yml` gains the three stdlib runs (2.2);
`mq3d_scratch_pool_test.py` goes into the image gate. After this PR, five
bench tests gate instead of the two that do today (#203 added the second
upstream) — and each gates with its negative control intact.

(2026-10-04) Revised scope. The stdlib job also runs
`bash bench/test_no_key_bind.sh` (2.2). `test_prepare_crash.py` goes into
the CPU-torch job with its pip deps (2.2; [INFERENCE: not run]; 2026-10-05:
#277 puts it in a sibling job, `prepare-crash`, see the top block), not into
the image gate. `mq3d_scratch_pool_test.py` still goes into the image
gate (2026-10-05: #278). Counts: `patch-integrity.yml` runs three tests today (two bench
tests and the kvarn gate). After PR B it runs six
(`mq3d_capacity_property.py`, `verbatim.py`, `test_no_key_bind.sh`
added), or seven with `test_prepare_crash.py`. The image build adds
`mq3d_scratch_pool_test.py`. Not every gate has a negative control:
`test_no_key_bind.sh` and the kvarn gate have none. The capacity-property
gate keeps its (`--mutate seq-rows`).

Acceptance: the workflow log shows all three new runs green in seconds;
a scratch PR that breaks the capacity property (edit the formula in
`mq3d_capacity_property.py:25-27`) fails the job — the negative control
proves the gate bites. (2026-10-04) The log also shows
`test_no_key_bind.sh` green, and `test_prepare_crash.py` green in the
CPU-torch job if PR B takes it. (2026-10-05: both green, on #271 and #277.
The image build runs `mq3d_scratch_pool_test.py` green on #278.)

## 4. Test plan

| test | how | proves |
|---|---|---|
| convention self-check | run each changed script in its pass and fail shapes on a live box once (runbook in the PR) | the new exits match the printed verdicts |
| negative controls | `mq3d_capacity_property.py --mutate seq-rows` must exit 0 via NEGATIVE CONTROL OK; edit verbatim.py's classifier to misclassify one shape, `_selftest` must exit 1 | the tests can still detect their bugs |
| CI | the workflow run on the PR itself | seconds-level, no GPU |
| no measurement regressions | `labd_bench.py`, `conc_ladder.py`, `interleave_dose.py` still exit 0 after printing | drivers stay drivers |
| manifest coverage | optional lint: every `bench/*.py`/`*.sh` named in `bench/README.md` | the table can't silently rot |

(2026-10-05) #279 ran the convention self-check against a stub server,
the published image and a CPU harness, not on a live box (top block). Its
PR body gives the commands for a GPU box and a live server. The drivers
did not change, so the no-regression row was not run. #279 has no lint. A
one-time check found 44 rows for the 44 entries in `bench/`.

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| `api_smoke.py` as a gate flakes on intentional feature changes | medium | it is wired as a *manual* gate first (documented in the manifest), CI promotion only after a few clean weeks; the exit-code change itself is the PR, the CI wiring is a later decision |
| The convention is ignored by the next new script | medium | the manifest's "exit semantics" column makes the convention the path of least resistance; the optional lint test catches an unlisted script |
| Quarantining the verdicts JSONL loses a reproduction artifact | low | one file is kept; `docs/reproductions/` is the repo's existing home for these |
| Exit-code changes break a wrapper that ignored failures on purpose | low | the only production consumer of a bench script is `qwen-server.sh:80` → `warmup.sh`, which already has proper exits and is untouched |
| The manifest rots | medium | it starts minimal (40 one-line rows; 39 at d5e2a01); the lint test is the cheap guard; without it, rot is no worse than today's 40-docstring archaeology |

## 6. Done when

Status at e371b42 (2026-10-04): none of items 1-5 is met.

1. `bench/README.md` lists every file in `bench/` with kind, needs, and
   exit semantics. (2026-10-04) It also lists
   `kvarn/tests/test_kvarn_fp16_dequant_torch.py`.
   (2026-10-05: met when #279 merges, with 45 rows.)
2. The six dishonest scripts' exits match their printed verdicts (the
   review's four, plus #226's and #208's). (2026-10-05: eleven scripts,
   not six, see 1.2. Met when #279 merges.)
3. `patch-integrity.yml` runs `test_model_verification.py` and
   `test_prepare_state.py` (both already gate, #203), plus
   `mq3d_capacity_property.py` (+ `--mutate seq-rows`) and `verbatim.py`;
   the image build runs `mq3d_scratch_pool_test.py`. (2026-10-04) It also
   runs `test_no_key_bind.sh` in the stdlib job. The `kvarn-torch-gate`
   job keeps its kvarn test and may add `test_prepare_crash.py`.
   (2026-10-05: met when #271, #277 and #278 merge. #277 runs
   `test_prepare_crash.py` in a job of its own, `prepare-crash`.)
4. `.gitignore` covers `bench/results-prefill-ab/`; the duplicate verdicts
   file is gone; `test_lookup_kernels.py` names itself. (2026-10-05: the
   two verdicts files are two runs, not a duplicate. Both stay, and the
   manifest says which run each one is. With that change, met when #279
   merges.)
5. (2026-10-04) `patch-integrity.yml` runs six tests where it runs three
   today (seven with `test_prepare_crash.py`), and the image build runs
   `mq3d_scratch_pool_test.py`. The capacity-property gate carries its
   negative control. (Through d5e2a01 this item read "five bench tests
   where two did, every gate with its negative control". The kvarn gate
   (#262) made today's count three, and `test_no_key_bind.sh` and the kvarn
   gate have no negative control.) (2026-10-05: met when #271, #277 and #278
   merge: seven tests in `patch-integrity.yml`, and the image build runs
   `mq3d_scratch_pool_test.py`.)
