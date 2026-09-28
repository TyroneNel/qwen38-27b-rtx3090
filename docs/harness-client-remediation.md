# One client module for the harness — remediation plan

Architecture review 2026-09-26, candidate 3 (Strong). The full report is
[architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card. Every claim below was re-verified against
the tree on 2026-09-26; where the review's numbers were wrong, they are
corrected here and in the report.

**Re-verified 2026-09-28 against upstream/main @ 2522ef9 (vLLM 0.30.0).**

- `bench-sse-keepalive.patch` (#226, series:59) fixes the upstream `vllm
  bench serve` request functions: stripping each network chunk deleted the
  blank line between SSE messages, gluing a server's `: keep-alive` comment
  to every message after it — so the launchers' `--sse-keep-alive-interval
  30` failed every prefill over 30 s (`run_benchmarks.sh --long` read zeros;
  #216 lost its 48k+ rows). It patches vLLM's
  `benchmarks/lib/endpoint_request_func.py`, not this repo's scripts: the
  seven in-repo SSE parsers below are byte-unchanged since 1cf8665 (`git
  diff 1cf8665..HEAD -- bench/` touches no parser). The edge is now proven
  upstream; the repo's own copies remain unfixed.

- `bench-probe-errors.patch` (vllm #58024, in since the 0.30 port d88544b)
  touches only `benchmarks/serve.py`: the `/tokenize` alignment probe now
  sends the Bearer and classifies its failure (404 route-or-name vs 401 vs
  unreachable vs timeout), and the two `/metrics` fetchers send the key. The
  benchmark requests themselves still present `OPENAI_API_KEY` alone and
  still report 0.00 in every field on a keyed server with only
  `VLLM_API_KEY` set — the `docs/python-314.md:55-56` claim quoted in 1.1
  stands (partial fix: probe + scrapes, not the request path).

- `bench/concurrent_collapse.py` (new; the #208 "!!!!" GPU reproducer) is
  the 20th harness script: 11th `_key()` copy, 5th `def post(`, a sixth
  PORT-env URL, and a second model-env spelling (`MODEL`, not `VLLM_MODEL`).
  It does not stream, so the SSE-parser census stays at 7.
- `bench/demo_render.py` was deleted (#220 — the README gif is now a
  plain-JS film under `bench/demo/`); it drops out of the offline-tools
  list below.
- `demo_capture.py` gained one docstring line (#220), shifting its later
  cites +1. Every other census re-ran clean; stale lines are fixed in place.


**Scope.** The 20 harness scripts that talk to the serving API — 16 Python
(`bench/api_smoke.py`, `bugb_sweep.py`, `conc_ladder.py`,
`concurrent_collapse.py`, `demo_capture.py`,
`interleave_dose.py`, `labd_accept.py`, `labd_bench.py`, `labd_soak.py`,
`needle_reuse.py`, `needle_test.py`, `prefix_alternation.py`,
`quality_battery.py`, `replay_offload_serve.py`, `residue_sweep.py`,
`seat_ttft.py`) and 4 bash (`bench/run_benchmarks.sh`, `real_rep.sh`,
`prefill_ab.sh`, `warmup.sh`) — and the five mechanisms they re-implement:
API-key resolution, server URL, model name, the request/stream helpers, and
the `/metrics` scrape. **Not in scope:** verdict/exit-code conventions
(candidate 5), the bench manifest (candidate 5), the offline tools
(`act_calib.py`, `mq3d_*`, `spec_attn_ctx_scan.py`, `tune_gdn.py`), and
launcher-side logic (candidate 2's plan).

## 1. Current state, precisely

### 1.1 The API key: four flavors, and the canonical one is the least used

The canonical chain lives in `resolve_api_key.sh:32-42`
(`resolve_client_key`: OPENAI_API_KEY → VLLM_API_KEY → `$REPO/api_key.txt` →
`"EMPTY"`), written for #113. It is used by exactly the **4 bash scripts**
(`run_benchmarks.sh:25-26`, `real_rep.sh:12-13`, `prefill_ab.sh:24-25`,
`warmup.sh:29-30`). Everything else re-derives it, and **no Python script
checks OPENAI_API_KEY at all** (verified: `grep -c OPENAI_API_KEY bench/*.py`
is 0 everywhere):

| flavor | shape | scripts (verified lines) |
|---|---|---|
| A: env-or-file | `VLLM_API_KEY` or `$REPO/api_key.txt` via a local `_key()` | `api_smoke.py:15`, `bugb_sweep.py:34`, `conc_ladder.py:68`, `concurrent_collapse.py:24`, `interleave_dose.py:42`, `needle_reuse.py:43`, `needle_test.py:29`, `prefix_alternation.py:48`, `quality_battery.py:35`, `residue_sweep.py:39`, `seat_ttft.py:32` — **11 scripts**, each with its own `def _key(` (verified ×11) |
| B: hardcoded home path | `open(expanduser("~/qwen-serving/api_key.txt"))` — unguarded, `FileNotFoundError` anywhere else | `demo_capture.py:28`, `labd_accept.py:91`, `labd_bench.py:28`, `labd_soak.py:46` — **4 scripts** |
| C: env only | `os.environ.get("VLLM_API_KEY", "")` | `replay_offload_serve.py:33` |
| canonical | `resolve_api_key.sh` + `resolve_client_key` | the 4 bash scripts |

The failure this produces is exactly the one #113 fixed on the bash side: a
client that silently presents nothing (or the wrong thing) to a keyed server.
`docs/python-314.md:55-56` documents the symmetric version for `vllm bench
serve` (which presents `OPENAI_API_KEY`): "with only the latter set it
receives silent 401s and reports 0.00 in every field rather than failing."
That sentence still describes the benchmark request path at 0.30.0 —
`bench-probe-errors` (vllm #58024) fixed only the `/tokenize` probe and the
`/metrics` scrapes (header block). The repo's own Python scripts are the
third surface, still un-fixed.

### 1.2 The server URL: five conventions

| convention | scripts |
|---|---|
| `PORT` env, `f"http://127.0.0.1:{PORT}/v1/…"` | `api_smoke.py:16-18` |
| `"http://127.0.0.1:" + PORT` | `bugb_sweep.py:35`, `residue_sweep.py:40`, `seat_ttft.py:33`, `concurrent_collapse.py:32` (the 6th `PORT`-env reader) |
| `API = f"http://127.0.0.1:{PORT}"` | `conc_ladder.py:53-54` |
| **`VLLM_API** env**, default `…:18020/v1` | `interleave_dose.py:43`, `needle_reuse.py:44`, `needle_test.py:30`, `prefix_alternation.py:49`, `quality_battery.py:36` |
| **hardcoded** `http://127.0.0.1:18020` (no override) | `labd_bench.py:29`, `labd_soak.py:47`; `DEMO_BASE` env in `demo_capture.py:28`; `--base` arg in `labd_accept.py:105` |

Five scripts even picked a *different environment variable* (`VLLM_API`) from
the rest (`PORT`), and two allow no override at all.

### 1.3 The model name: hardcoded almost everywhere

`"qwen3.8-27b"` is a string literal in 15 call sites
(`api_smoke.py:31,75,84`, `bugb_sweep.py:66`, `conc_ladder.py:126`,
`demo_capture.py:56`, `labd_bench.py:83,96`, `labd_soak.py:84`,
`needle_test.py:59`, `quality_battery.py:70,99`,
`replay_offload_serve.py:53`, `residue_sweep.py:55`, `seat_ttft.py:52`).
Three scripts accept a
`VLLM_MODEL` env with the same default (`interleave_dose.py:44`,
`needle_reuse.py:45`, `prefix_alternation.py:50`), `concurrent_collapse.py:33`
reads a second env spelling (`MODEL`, not `VLLM_MODEL`), one takes `--model`
(`labd_accept.py:106`). **No script asks the server** — `/v1/models` appears
nowhere in `bench/`. A served-name change (the
`serve-model-path-match`/`serve-404-served-names` patches exist precisely
because this bites) is a 16-file edit.

The bash side has its own drift: `run_benchmarks.sh:28` and `real_rep.sh:14`
default `MODEL` to the base checkpoint, `prefill_ab.sh:26` to `-fast`, and
only `warmup.sh:41` uses the shared `select_model.sh` (the candidate-2 plan
covers the launcher-side selector).
### 1.4 The mechanisms, counted

Measured against the 16 Python scripts (the review said "~1,500 lines of
copy-paste" — that was an overestimate; the honest numbers):

- **164 lines** match the narrow glue idiom grep (`def _key|def post|def
  metrics|KEY =|BASE =|urllib…|Bearer|data: |[DONE]|include_usage|ttft`)
  across the 16 files (re-measured 2026-09-28: 155 on the original fifteen,
  +9 in `concurrent_collapse.py`); counting the surrounding blocks
  (preambles, inline request builders, stream loops, scrape functions, TTFT
  math) puts the re-implemented total at **~420 lines**. The duplication is
  broad rather than deep — which is exactly why it drifted: no single copy is big enough
  to look like a module.
- `def _key(` — **11 copies** (the file-read-with-fallback helper).
- `def post(` — **5** (`api_smoke.py:21-27`, `quality_battery.py:43-46`,
  `needle_test.py`, `labd_accept.py`, `concurrent_collapse.py:41-48`); the
  other 11 scripts inline the same
  `urllib.request.Request` + headers block at each call site (e.g.
  `labd_bench.py:88-91` and `:101-103`, twice in one file).
- SSE stream parsers (`data:` / `[DONE]` / `include_usage` / first-token
  timing) — **7**: `api_smoke.py:74-78`, `conc_ladder.py:124-156`,
  `demo_capture.py` (~:61-98), `labd_accept.py:167-`, `labd_bench.py:109-125`,
  `replay_offload_serve.py:53-74`, `seat_ttft.py`. The TTFT idiom
  `(t_first or t_end) - t0` recurs in at least 5
  (`conc_ladder.py:154`, `demo_capture.py:90`, `labd_bench.py:130`,
  `replay_offload_serve.py:71`, `labd_accept.py:195`).
- `def metrics(` — **7 Python** (`bugb_sweep.py`, `conc_ladder.py:114-121`,
  `labd_accept.py`, `labd_bench.py:42-50`, `labd_soak.py`,
  `replay_offload_serve.py:41-46`, `residue_sweep.py`) **+ 3 bash**
  (`run_benchmarks.sh:38-39`, `real_rep.sh:18-19`, `prefill_ab.sh:56-57`).

### 1.5 The un-propagated fixes — the proof the glue is where bugs live

1. **The `_created` guard exists in 1 of 3 bash spec-scrapers.** vLLM's
   prometheus client emits `…_total_created` lines (documented in-repo:
   `labd_accept.py:135` "prometheus_client also emits _created lines"). The
   scrape pattern `^vllm:spec_decode_num_(drafts|accepted_tokens)_total`
   matches those lines too. `real_rep.sh:19` filters them
   (`grep -v created`); `run_benchmarks.sh:39` and `prefill_ab.sh:57` do
   not — and their tok/step arithmetic is **positional** (`awk '{print $2}'`
   into a split list, `run_benchmarks.sh:47-51`), so a `_created` line in
   the stream shifts a Unix timestamp into the token-count slots. Whether a
   given run reads as garbage or as a silently plausible number depends on
   the server's emission order; the guard makes the question moot, and two
   of the three scrapers lack it. (The Python scrapers are safe by accident:
   `conc_ladder.py:119` requires `{` right after the metric name;
   `labd_bench.py:47` requires `" "` or `"{"` — `_created` matches neither.)
2. **The array lesson, learned in one file only.** `warmup.sh:43-46` carries
   the comment "Build the command as an array, not a string: an unquoted $B
   re-splits and re-globs" — and `run_benchmarks.sh:34`, `real_rep.sh:17`,
   `prefill_ab.sh:30` still build `B=` as a string and expand it unquoted.
3. **`prefill_ab.sh:39-40` re-implements the launcher's INT8 export guards
   in the drifted shape** (exports `VLLM_MARLIN_INT8_INCLUDE_RE` whenever
   `INT8_LAYERS` is non-empty, `INT8_ACT` or not — the candidate-2 plan's
   1.3.1 bug shape, in a fourth file).
4. **Model defaults split three ways** on the bash side (1.3): base, `-fast`,
   and the one correct consumer of `select_model.sh`.

None of these is a big bug. All of them are the same bug: a fix lands in the
copy the reporter was running, and the harness has no place to put it for
everyone.
## 2. The design: `bench/harness.py`

One stdlib-only module (every script today runs on the venv's bare Python —
no `requests`, and adding a dependency for this would be a regression).
Scripts import it as `import harness`: each runs as
`venv/bin/python bench/<script>.py` (verified in their docstrings), which
puts `bench/` on `sys.path` — no shim, no package surgery, no
`sys.path.insert` (the anti-pattern candidate 4 flags in `drafter/`).

```text
client_key()  -> str
    The resolve_api_key.sh resolve_client_key chain, exactly:
    OPENAI_API_KEY > VLLM_API_KEY > $REPO/api_key.txt > "EMPTY".
    One implementation, kept honest by a test that runs both sides (4.1).

base_url()  -> str
    One convention: VLLM_API if set (full base), else
    http://127.0.0.1:${PORT:-18020}. No per-script env-var inventions.

model()  -> str
    VLLM_MODEL if set, else the first id of GET {base}/v1/models (sends
    client_key(); this stack's auth-deny-default patch guards /v1/models).
    The server is the source of truth; the 15 hardcoded literals die.

post(path, payload, timeout=1200) -> dict
    JSON POST to base_url()+path with the key header. Replaces 5 defs and
    11 inline copies.

stream_chat(payload, timeout=1800) -> (ttft_s, decode_s, ntok, usage, text)
    The SSE parser once: data: lines, [DONE], usage capture, first-content
    timing, the (first or end) - t0 idiom with the documented caveat.
    Replaces 7 parsers.

metrics(*names) -> dict[str, float]
    GET {base}/metrics with the key; sums label-variant lines per name;
    _created lines excluded by construction (1.5.1 can never recur).

spec_delta(m0, m1) -> (steps, tok_per_step)
    The drafts/accepted arithmetic by NAME, not position (1.5.1).
```

Bash side: the three `spec()`/`metrics()` helpers keep `curl` (they are five
lines) but gain the `_created` filter — or, tidier, call
`venv/bin/python bench/harness.py --metrics name1 name2` (the module is
executable). The `B=` strings become arrays, propagating `warmup.sh:43-46`'s
comment with the change. The model-default drift (1.3) defers to
`select_model.sh` — one line each, riding candidate 2's convention.

**Deliberately not in the module:** workload logic (prompts, doses, tasks —
that is each script's actual content); anything offline (no server, no GPU
keys); the `vllm bench serve` invocation itself (a different client, owned
by the run_benchmarks family); verdict/exit-code policy (candidate 5).
## 3. Rollout

Three PRs, each independently revertable. Migration is all-or-nothing per
mechanism — a half-migrated mechanism is worse than none (two conventions
again), so each PR owns its mechanism end to end.

### 3.1 PR A — the module + key/URL/post migration

Add `bench/harness.py` with `client_key`, `base_url`, `post`, and its test
file (4.1). Migrate all 16 Python scripts: delete every `_key`/`KEY`/
`BASE`/`API`/`URL` preamble and every `def post(` / inline request block.
Flavor-B scripts (`labd_*`, `demo_capture`) gain working key resolution on
any checkout — a behavior change, called out in the message (today they
hard-fail outside `~/qwen-serving`).

Acceptance: `grep -c 'def _key(|api_key.txt' bench/*.py` → 0 outside
harness.py; `bench/test_harness.py` green in CI; `api_smoke.py` output
against a live server byte-identical to pre-migration (modulo timings).

### 3.2 PR B — stream/metrics migration + the bash guards

Add `stream_chat`, `metrics`, `spec_delta`. Migrate the 7 SSE parsers and 7
Python scrapers. Bash: add the `_created` filter to `run_benchmarks.sh:39`
and `prefill_ab.sh:57` (matching `real_rep.sh:19` — one line each, zero risk
to measurement), and convert the three `B=` strings to arrays with
`warmup.sh:43-46`'s comment attached.

Acceptance: `grep -l '\[DONE\]' bench/*.py` → only harness.py and
test_harness.py; the two bash `spec()` functions carry the filter;
`tok/step` on a live dflash2 server unchanged (the filter is a no-op when
no `_created` lines exist).

### 3.3 PR C — the model name + docs

Add `model()`; replace the 15 literals and the `VLLM_MODEL`/`MODEL`
duplications;
the three bash scripts take their model from `select_model.sh` (candidate 2
alignment). Update the affected docstrings (several document
`VLLM_API_KEY … PORT=` on their usage lines, e.g. `api_smoke.py:5`).

Acceptance: `grep -c '"qwen3.8-27b"' bench/*.py` → 0 outside harness tests;
renaming the served model is a one-line change.

## 4. Test plan

### 4.1 `bench/test_harness.py` (CPU, seconds, wired into `patch-integrity.yml`)

In `test_model_verification.py`'s existing unittest style, against a stub
`http.server`:

| test | proves |
|---|---|
| `client_key` precedence table (only OPENAI set / only VLLM set / only file / none → `"EMPTY"`) run **twice**: once in Python, once by sourcing `resolve_api_key.sh` in a subprocess — the results must match | the port cannot drift from the canonical chain (the #113 failure mode) |
| stub `/metrics` containing `…_total` and `…_total_created` lines | `metrics()` excludes `_created`; `spec_delta` computes by name |
| stub SSE transcript (chunks, `[DONE]`, a `usage` frame) | `stream_chat` parses ttft/ntok/usage correctly, including the no-first-token edge |
| `model()` with `VLLM_MODEL` set, and against a stub `/v1/models` (key header asserted on the stub side) | override + server-as-source-of-truth both work |
| `base_url()` matrix: `VLLM_API` set / `PORT` set / neither | one URL convention |

### 4.2 Migration parity (runbook for the PRs, needs a live server once)

For each migrated script: run against a live server pre- and post-migration
and diff the output modulo timing fields (`api_smoke.py`'s SUMMARY line,
`labd_bench.py`'s LABD rows, `conc_ladder.py`'s table). This is a manual
gate, listed per PR in section 3 — the harness measures a GPU server, so
full automation belongs to candidate 5's manifest, not this plan.

## 5. Risks

| risk | likelihood | mitigation |
|---|---|---|
| Half-migration leaves two conventions again | medium | the rollout is mechanism-per-PR (3.1-3.3), each with a grep-based acceptance gate; no partial-mechanism merges |
| `import harness` breaks under an exotic invocation (`python -m`, symlinked script) | low | the documented invocation (`venv/bin/python bench/x.py`) puts `bench/` on `sys.path`; the test file imports it the same way; a fallback two-liner is acceptable if a real caller breaks |
| Flavor-B scripts silently change behavior for their one known user | medium | called out in PR A's message; the change is strictly "works where it used to crash" |
| `model()` adds a `/v1/models` round-trip per invocation | low | cached per process; the endpoint is cheap and already hit by `verify.sh`'s server checks |
| The `_created` filter changes a published number | low | it is a no-op unless `_created` lines exist; if they do, the unfiltered number was already wrong (1.5.1) |
| Scope creep into verdict conventions (candidate 5) | medium | exit-code policy is explicitly out of scope (section 2's "not in the module") |

## 6. Done when

1. `grep -c 'def _key(' bench/*.py` → 0 everywhere except `harness.py`'s
   interior.
2. `grep -l 'OPENAI_API_KEY' bench/*.py` → `harness.py` and
   `test_harness.py` only — and the precedence table is test-pinned.
3. `grep -l '\[DONE\]' bench/*.py` → `harness.py` / `test_harness.py`
   (`test_bench_sse_keepalive.py`'s fixture transcript aside).
4. `run_benchmarks.sh:39` and `prefill_ab.sh:57` carry the `_created`
   filter; `real_rep.sh:19` has it but no why comment — add one while there.
5. `grep -c '"qwen3.8-27b"' bench/*.py` → 0 outside tests.
6. `bench/test_harness.py` runs in `patch-integrity.yml` in seconds, green.

At that point "how the harness talks to the server" has one answer per
mechanism, the next #113 lands in one file, and the harness's ~420 lines of
re-implemented glue is a module the tests actually cover.
