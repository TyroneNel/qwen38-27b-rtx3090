# Phase-2 review: benchmark harness and baseline

Reviewer: phase-2 independent re-run (Opus, effort high), 2026-10-21. No GPU on this box (D1, D2). So the "re-run" here means three things: re-deriving the baseline from the original files, re-running the CPU self-check, and auditing the harness code.

## Verdict: PASS WITH CORRECTIONS

- Every baseline number I checked matches its original. The copies are byte-identical to the originals.
- The self-check reproduces.
- I found one measurement bug: the decode-rate window ignored the finish chunk. I fixed it and added a regression check.
- I found one non-runnable README command: the profile `EXTRA_ARGS` line. Fixed.
- I fixed one wrong checksum in `vendor/PROVENANCE.md` and five citation or label errors in `baseline.md`.
- Nothing I found would invalidate a GPU number already planned. The remaining findings are Medium/Low recommendations, listed below.

## 1. Copies and links

- **Byte identity.** All 10 files in `docs/research/04-benchmarks/raw/prior-3090/` have the same sha256 as their originals under `~/research-src/deepinfer-ref/docs/research/deepinfer/`. For `gpu-agent-prompt.md` the original sits one directory up, not in `gpu-results/`, as PROVENANCE.md:24 says. The first-16 hashes in PROVENANCE.md:15-24 are all correct. I hashed the copy and the original directly; `cmp` also passes on `gpu-agent-prompt.md`.
- **Links.** Every markdown link in `baseline.md`, `LEDGER.md` and `harness/README.md` resolves. The cited line ranges point at the stated content, with two imprecise ones corrected below (e-profile 151 ms, evals-4 thinking-on).

## 2. Re-derivation table

All values below are **measured (prior, GPU box)**. I read each one from the original, not the copy. Paths are relative to `~/research-src/deepinfer-ref/docs/research/deepinfer/gpu-results/`.

| # | Metric | baseline.md value (line) | My value, original file:line | Agree? |
|---|---|---|---|---|
| 1 | A 64conc 128/512 e2e, median (range) | 814.3 (805.6-822.9) (:33) | 814.3 (805.6-822.9), baseline.md:35 | yes |
| 2 | A decode C/medTPOT | 904 (:33) | 912 / 896 → 904, baseline.md:36 | yes |
| 3 | A KV pool at 0.93 / 0.91 | 242,010 / 227,319 (:33) | same, baseline.md:49 | yes |
| 4 | A WDDM non-local at 0.93 / 0.91 | 370→698 / 370-378 MiB (:33) | same, baseline.md:58 | yes |
| 5 | B C1 greedy decode, tok/step | 114.4, 3.36 (:35) | 113.1 / 115.6 → 114.4, 3.36, baseline.md:38 | yes |
| 6 | B C1 e2e | 110.4 (109.2-111.6) (:35) | same, baseline.md:37 | yes |
| 7 | B C1 default sampling | 109.0 (:36) | 109.0, baseline.md:39 | yes |
| 8 | C copy decode, tok/step | 357.6 (353.2/361.9), 14.22 (:43) | same, baseline.md:40 | yes |
| 9 | D C1 greedy decode, tok/step | 99.8 (99.3/100.2), 2.64 (:44) | same, baseline.md:42 | yes |
| 10 | D′ TOTAL ~60k / ~95k | 55.7 / 48.8 (:49-50) | 55.7 (55.4/56.0) / 48.8 (48.4/49.2), baseline.md:44-45 | yes |
| 11 | E TOTAL, tok/step, copy | 57.8, 2.71, 158.6 (:51) | same, baseline.md:46-47 | yes |
| 12 | D `PTH_FA=0` C1, GPU busy, pool at 0.90 | 63-75, 58-64%, 189,795 (:54) | same, baseline.md:65 | yes |
| 13 | Ladder A per-stream N=1..32 | 44.6/40.2/30.1/20.0/12.2/6.8 (:34) | same, evals-10.md:26-31 | yes |
| 14 | Ladder B aggregate N=1..8; N=8 preemptions | 118/182/253/236; 19 (:37) | same, evals-10.md:26-29 | yes |
| 15 | Ladder D aggregate N=1..8 | 105/192/308/423 (:45) | same, evals-10.md:26-29 | yes |
| 16 | B thinking-on C1, Δ, tok/step | 100.2, −14.8%, 3.14 (:38) | same, evals-4.md:30 | yes |
| 17 | D teacher-forced accept@pos 0-2 | 0.86/0.73/0.62 (:47) | same, evals-4.md:44 | yes |
| 18 | B accept by position at T=0.7 (pos 0/1/2/3/6) | 0.687/0.511/0.404/0.325/0.190 (:39) | same, spec-5.md:40-44 | yes |
| 19 | B tok/step T=0 / T=0.7 / think | 3.486 / 3.611 / 3.341 (:39) | same, spec-5.md:32-34 | yes |
| 20 | B warm-up C1 runs 1-4 | 114.4/109.8/105.7/103.6 (:41) | same, evals-11.md:24 | yes |
| 21 | Stock quality: ppl all (en/da/code); GSM8K | 8.401 (10.77/10.91/3.60); 95.5% (:42) | same, evals-3.md:28-29 | yes |
| 22 | E at ~170k: decode cold / profiled | 11.5 (1.86) / 12.5 (2.01) (:52) | same, e-profile.md:31-32 | yes |
| 23 | E per-step drafter / attention | 3.6 ms / 117.7 ms (78.0%) (:53) | same, e-profile.md:41,43 | yes |
| 24 | Env: driver, Engine, power | 610.88, 29.8.2, 250 W (:9) | same, ENVIRONMENT.md:11,13,16 | yes |

Result: 24 of 24 agree, 0 disagree. I also checked the native-doc figures at baseline.md:56 (942 / 127.1 / 381 / 100.3 / 67) against baseline.md:35-46: they agree. `quality_ref_prior.json` also matches evals-3.md:28-29.

### Factual corrections made in baseline.md (each marked `[review: …]`)

1. **Timing validity (:18). I upgraded the claim from "estimated from commit order" to "git-verified".**
   - The commits the prior baseline records (88549e7 / e6a3ca0, original baseline.md:6) contain 373a695: `git merge-base --is-ancestor 373a695 88549e7` is true.
   - `git show e6a3ca0:bench/labd_bench.py` uses `perf_counter` at lines 85, 95 and 97.
   - The image commit 2cf50a0 also contains 373a695.
   - Still open: whether the box's working tree matched that commit. That is ledger row `prior-timing-check`.
2. **The "8.3% in `bench/README.md`" citation (:22) points at the wrong copy.** The sentence is in upstream's `bench/README.md:8` (`~/research-src/HyperQwen/bench/README.md:8`). Our 95ac251 `bench/README.md` lacks it. I fixed the same citation in harness/README.md:31.
3. **Two values sat under "Throughput" that are decode rates.** Mode C "six-task TOTAL 95.8" (:43) and mode E "`copy` 158.6" (:51) are labelled now.
4. **Mode B teacher-forced, thinking on (:40).** The original is 4 tasks, reasoning phase only (evals-4.md:41). The row implied the same 6 tasks.
5. **E per-step "151 ms wall" (:53).** This comes from e-profile.md:34 (mean 150.8 ms), not from :41-48. I added the citation.

## 3. Self-check

- **Re-run of the original scripts.** I ran `bash harness/selfcheck.sh` on this box with the scripts unchanged: exit 0, `SELFCHECK PASS`.
- **Diff against the saved `selfcheck.out`.** After normalising the temp paths, only two kinds of lines differ:
  - timestamps;
  - mock timing values that move with sleep jitter (C1 decode 650-664 tok/s, agg ±1%).

  The deterministic values all match: tok/step 2.18 / 5.33, accept rate 0.169 / 0.619, 3/3 reps, the greedy 16/16 / 16 real / 16 near-tie verdicts, quality_guard PASS / FAIL / INVALID, and vram `None`. **Reproduced.**
- **Re-run after my script fixes.** Exit 0, `SELFCHECK PASS`, now with 11 `ok` checks, including the new 1c. This output is saved over `harness/selfcheck.out`, with a header saying it is the reviewer's re-run.
  - The mock's tok/step now reads 3.29 / 0.327, because the mock's step RNG now draws n-1 tokens. That is a mock artefact, not a stack number.
- **Negative control for 1c.** The pre-fix `sweep.py` logic against the new mock reads 331.1 tok/s at a mock rate of 300, which is +17% and fails the check. The fixed code reads 284.9, which passes. Script and output are in the scratchpad only, not the repo.

## 4. Audit findings

### Checked and correct

| Item | Evidence |
|---|---|
| TTFT runs from just before the request is sent to the first chunk whose `content`/`reasoning`/`reasoning_content` is non-empty. The role-only and empty chunks are skipped. | `t0` is set at sweep.py:81, before the generator. `harness.stream` only sends on the first `next()` (bench/harness.py:67). The first-token test is at sweep.py:90-92. |
| Decode tok/s excludes TTFT and counts tokens from `usage.completion_tokens`, not from chunks. | sweep.py:86 and :98 compute `(ntok-1)/(last-first)`, with `include_usage` set at :77. |
| Concurrency is real: one thread per request behind a `Semaphore(conc)`; urllib blocks in sockets and releases the GIL. A round is `max(conc, len(set))` requests. | sweep.py:103-115 |
| Warm-up rounds are excluded from the statistics. Every round gets its own salt, so no warm-up warms the prefix cache for a timed round. | sweep.py:168-174, :109 |
| At least 3 repeats are enforced, with mean, sd, min, max and CV. | sweep.py:155-156, :131-137 |
| The `!` check for cached prompt tokens is live, because the launchers pass `--enable-prompt-tokens-details`. | single-user/start_qwen.sh:1042, batch/start_qwen.sh:315 |
| Greedy check: compares token ids (via `return_tokens_as_token_ids`), at T=0, thinking off, C1, on the same prompt set. It refuses different `max_tokens` and reports the first divergence index plus both top-1/top-2 margins. No seed is needed at T=0. | greedy_equal.py:34-45, :55-71 |
| Acceptance counters: `vllm:spec_decode_num_drafts_total`, `…num_draft_tokens_total`, `…num_accepted_tokens_total` and `…num_accepted_tokens_per_pos_total{position}`, summed over engines. tok/step is `1 + Δaccepted/Δdrafts`, matching vLLM's own `mean_acceptance_length` (~/vllm/vllm/v1/spec_decode/metrics.py:114). Per-position rate is `Δpos/Δdrafts` (metrics.py:194-195). They are **deltas over each round's window** (s0 before threads start, s1 after join), not cumulative since boot. | sweep.py:35-50, :110, :116-129 |
| PTH_FA (D10): `.env` reaches the container through `env_file` (docker-compose.yml:72-74). `run_suite.sh` exits 2 when it is unset, and `snapshot.sh` records it. The self-check exercises the refusal. | run_suite.sh:41-45, snapshot.sh:19 |
| All harness timing uses `time.perf_counter`. The image's `bench/` (2cf50a0) contains 373a695. | sweep.py:81,91,94,110,116 |
| `import harness` always resolves to `bench/harness.py` under run_suite's `PYTHONPATH="$BENCH:$HARNESS"`, from any cwd. I tested cwd `/`, the repo root, `04-benchmarks/` and `harness/`. A regular module beats a namespace directory even when that directory comes earlier on the path. Without `PYTHONPATH` the import fails cleanly with `ModuleNotFoundError`; it never silently binds to the directory. | tested on this box |
| The image's `bench/harness.py` and `bench/conc_ladder.py` are identical to our checkout's (`git diff --stat 95ac251 2cf50a0 -- bench/` lists neither), so the self-check against our `bench/` stands for the image. `quality_battery.py` writes `result_<tag>.json` with `ppl[k][0]` and `gsm8k.acc`, which is what `quality_guard.py` reads. Its ppl and greedy GSM8K code equals the prior branch's version that produced evals-3. | `git show 2cf50a0:bench/quality_battery.py`:108-112 |
| Vendor: `warmup_probe.py` and `trace_breakdown.py` are byte-identical to `~/research-src/deepinfer-ref/bench/` (full sha256 compare and `cmp`). Only the PROVENANCE hash for trace_breakdown had a typo (fixed). | vendor/PROVENANCE.md:7-8 |

### Findings

| # | Severity | Where | Finding | Fix |
|---|---|---|---|---|
| F1 | **Medium (fixed)** | sweep.py:90-95 | **Decode rate overcounts.** When EOS is the first token of a step, vLLM streams a separate chunk with an empty delta and `finish_reason` (~/vllm/vllm/entrypoints/openai/chat_completion/serving.py:745-755). `completion_tokens` counts that EOS, but `last` stopped at the previous content chunk. So the rate is `(ntok-1)` over one step too few. The bias is about 1/steps per request: up to ~1.3% at 256 tokens and 3.4 tok/step, more on short answers. It hits a random subset of requests, so it adds noise as well as bias. This is the same order as the 1-2% baseline spread. | `last` is now also stamped on the finish chunk. The mock now streams vLLM's shape (n-1 content tokens, then an EOS-only finish chunk). Self-check 1c fails the old logic (+17%) and passes the new. The repo's `conc_ladder.py:129-134` and the vendored `warmup_probe.py` keep the old convention, and I left them alone (references and vendor stay unchanged). So compare sweep numbers with sweep numbers, and ladder numbers with ladder numbers. |
| F2 | **Medium (fixed)** | README.md:97 | **The profile boot's `.env` line was not runnable.** It read `EXTRA_ARGS={"profiler":…}`, without the `--profiler-config` flag, so `vllm serve` would get a bare JSON argument. The inline comment also sat on the value line. | Now `EXTRA_ARGS=--profiler-config {…}`, as in e-profile.md:9 and run_suite.sh:71, with the comment on its own line. |
| F3 | Medium | vram.py, sweep.py:166 | **Peak VRAM via nvidia-smi says little per cell.** vLLM preallocates the KV pool, so `memory.used` is about weights + pool + graphs and nearly flat across cells. Under WSL2/WDDM it is device-wide (no per-process numbers) and excludes memory demoted to system RAM, which is the spill the prior run found at `GPU_UTIL=0.93` (baseline.md:58). The docstring states the spill caveat (vram.py:6-8), but the gap-#2 "peak VRAM per cell" will be a near-constant. | Recommended, not done (a feature, not a bug): poll `vllm:kv_cache_usage_perc` and take the per-cell peak, as conc_ladder.py:146 already does. Keep the host-side WDDM non-local sampler (ENVIRONMENT.md:28) in the GPU agent prompt for spill. |
| F4 | Low | vram.py:23 | **The GPU index defaults to 0.** On WSL2, nvidia-smi inside the container sees every GPU. That is why `resolve_config.sh:85` selects `-i "${CUDA_VISIBLE_DEVICES%%,*}"`. Index 0 is right on this box only because the 3090 enumerates first (ENVIRONMENT.md:9). Separately, spawning one `nvidia-smi` every 0.5 s during host-bound arms (`PTH_FA=0` mtp, 58-64% GPU busy, baseline.md:65) could perturb them, which matters for `pthfa-ab`. | Recommended: default `HARNESS_GPU` to the first `CUDA_VISIBLE_DEVICES` entry. Use one long-lived `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500` process, or do an on/off check in the smoke run. |
| F5 | Low | LEDGER.md:14 (smoke) | **Smoke `PHASES="snap probe sweep"` leaves out `greedy`.** So `logprobs` + `top_logprobs` + `return_tokens_as_token_ids` under dflash2's V2 model runner is first exercised at bench-7. | Recommended: add `greedy` to the smoke phases. I did not edit LEDGER.md, because it is outside this review's edit scope. |
| F6 | Low (doc fixed) | README.md:83 | **Greedy spec-vs-non-spec comparisons can be confounded by KV dtype.** A `PTH_FA=1` arm uses int8 KV and has no non-spec twin (`start_qwen.sh:299-304` refuses `PTH_FA=1` off mtp/dflash2 `CTX=long`). Its divergences against `SPEC=off` mix KV-dtype effects with speculation effects. | Added a `[review]` note: compare at the same `CTX` with `PTH_FA=0` on both arms. |
| F7 | Low | README.md:42-50 | **The `.env` heredoc depends on Compose's dotenv details.** It appends keys the box `.env` already has (`MODELS_DIR`, `VISION`), so it relies on Compose taking the last value, and it puts `# comments` on value lines. The box's `CTX=fast` equals the launcher default (`start_qwen.sh:185`), so mode B itself is unaffected. | Recommended: replace the lines (`sed -i`) rather than append, and keep comments on their own lines. The GPU agent should check the `[effective-config]` line, which the README already asks for. |
| F8 | Low | docker-compose.harness.yml:8-9, README.md:61 | **`/results` may end up root-owned.** If the checkout lives on the WSL ext4 filesystem, the container creates `/results/*` as root, and the host-side `snapshot.sh … host` may then fail to write. On `/mnt/g` (drvfs) this is fine. Separately, `/results/prof` is shared across profile boots, so traces accumulate. | Step-0 check for the GPU agent. Use a per-run `PROF_DIR`. |
| F9 | Info | sweep.py:110-121 | **Spec counters can miss the last step of a window.** They are recorded after the outputs reach the client, so a round's last step can land in the next window. That is at most about one step per window, and it carries into the next round rather than being lost. spec-5.md:46 saw 590 vs 597 steps. | None needed at ≥ 500 steps per round. |
| F10 | Info | sweep.py:98 | **The rate assumes the first content chunk holds one token.** That holds at C1. At high N with a slow client, vLLM coalesces deltas, which inflates `ntok-1` slightly. | None. |
| F11 | Info | sweep.py:101-115 | **TTFT at conc > 1 is TTFT under load.** All requests in a round start together, so it includes queueing behind the other prefills. This matters most for long-context cells at N = 2-4. | Read it as TTFT under load. |
| F12 | Info | run_suite.sh:61-62 | **`conc_ladder` prints the best rep; `sweep` prints mean ± sd.** conc_ladder.py:225-245 keeps the best rep; all reps are in the JSON. | Do not mix the two in one table. |
| F13 | Info | – | **The metric names were checked against `~/vllm` (0.32.1rc0), not against the image's vLLM 0.31.0** (docker/requirements.txt:5). | Already a smoke-row item. |
| F14 | typing only | sweep.py:43,83 / :98 / :129,:172 | Pyright's static errors are not runtime bugs: <br>• `harness.samples`/`stream` "not attributes": Pyright resolves `harness` to the directory; at runtime it is `bench/harness.py` (see "Checked" above). <br>• `last - first` "None arithmetic": `first` and `last` are set together (:91-92), and `first is None` raises at :95. <br>• Dict value types: mixed-type result dicts, by design. | None. |

## 5. README runnable on the WSL2 box?

Yes, after F2. The checks behind that:

- `COMPOSE_FILE` chains the box's untracked `docker-compose.override.yml`, which is assumed to exist (ENVIRONMENT.md:24, step-0 check). Relative paths resolve against the first file's directory, which is the repo root.
- The service is named `single`. The server listens on `PORT` inside the container (docker-compose.yml:84,100), so `curl 127.0.0.1:18020` from `exec` works, and the image has `curl` (Dockerfile:20).
- `MODELS_DIR=/mnt/g/models` follows ENVIRONMENT.md:26. `/app/resolve_api_key.sh` exists. `QUALITY_DATA` sits under the read-write models mount.
- The chain leaves out `docker-compose.override.local.yml`, so `/app/bench` is the image's copy, which has the perf_counter fix. That is what the README intends.

Remaining caveats are F7 and F8.

## 6. Every edit made

| File | Edit |
|---|---|
| `04-benchmarks/harness/sweep.py:93-94` | F1: stamp `last` on the finish chunk too (`# review:` comment). |
| `04-benchmarks/harness/mock_server.py:105,112-113` | Mock streams n-1 content tokens, then an EOS-only finish chunk, as vLLM does (`# review:` comments). |
| `04-benchmarks/harness/selfcheck.sh` (new section 1c) | Regression check for F1. |
| `04-benchmarks/harness/selfcheck.out` | Replaced with the reviewer's post-fix run (header says so). |
| `04-benchmarks/harness/README.md` | Changes, all marked `[review: …]`: <br>• :31 the 8.3% citation; <br>• :97 F2; <br>• :83 F6; <br>• :111 renumbered perf_counter lines; <br>• :114 the image also has 373a695; <br>• :125 self-check 1c. |
| `04-benchmarks/harness/vendor/PROVENANCE.md:8` | trace_breakdown sha256 typo: `…8f` → `…86`. |
| `04-benchmarks/baseline.md` | The five `[review: …]` corrections in section 2. |

Not edited: `bench/`, the references, `LEDGER.md` and `gpu-queue.md`. The bench-* rows in gpu-queue.md are consistent with the ledger.
