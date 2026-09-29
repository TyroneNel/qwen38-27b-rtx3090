# One client module for the harness — remediation plan

Architecture review 2026-09-26, candidate 3 (Strong). The full report is
[architecture-review-20260926-194530.html](architecture-review-20260926-194530.html);
this is the deep dive on that card. Every claim below was re-verified against
the tree on 2026-09-26; where the review's numbers were wrong, they are
corrected here and in the report.

**Re-verified 2026-09-29 against upstream/main @ d5e2a01 (vLLM 0.30.0).**
Every census below was re-run; the command is named next to each count.

- **Unchanged:** `bench/` is byte-identical to 2522ef9 (`git diff --stat
  2522ef9 upstream/main -- bench/` is empty), and so is
  `resolve_api_key.sh` (`resolve_client_key` still `:32-42`, OPENAI →
  VLLM → `$REPO/api_key.txt` → `EMPTY`). All counts hold: 20 scripts, 11
  `def _key(`, 5 `def post(`, 7 SSE parsers, 7 + 3 `metrics()`, 15
  literals, 164 glue-grep lines. None of the seven upstream commits since
  2522ef9 touches a harness client.
- **#219 (8cf642e) adds no harness client and no client key flavor.**
  `scripts/hq-doctor.sh:56`, the `Makefile` `ps` target and `verify.sh
  --wait` only `GET /health`, which is unauthenticated
  (`patches/auth-deny-default.patch:72`, `UNGUARDED_PATHS`), and send no
  key; hq-doctor reads `VLLM_API_KEY` from `.env` presence-only
  (`:29-31`). The census stays 20. The key change in #219 is server-side:
  `single-user/alternative.sh:39-41` now sources `resolve_api_key.sh` and
  calls `resolve_vllm_key` (was `cat api_key.txt`), so the resolver is
  sourced by 7 files (3 launchers + the 4 bench bash scripts; `git grep -n
  resolve_api_key -- ':!docs'`). Worth knowing for `client_key()`: #219's
  quickstart puts the key in `.env` (`make keygen`), and neither
  `resolve_client_key` nor any Python flavor reads `.env`; a host-side
  bench run against the Docker server still needs the key exported. That
  is unchanged behavior, not a new flavor, and this plan does not add a
  `.env` reader.
- **Corrections to claims that were already wrong at 2522ef9** (the bench
  files did not change, the earlier pass mis-read them):
  - The #226 bullet below said "the repo's own copies remain unfixed". They
    were never broken by that defect: all 7 in-repo parsers read the
    response line by line (`for raw in r:` / `splitlines()`) and skip any
    line not starting `data:`, so a `: keep-alive` line is dropped. A
    throwaway stub server sending `: keep-alive\n\n` before three chunks
    returned all three tokens and the usage frame through the
    `labd_bench.py:110-125` loop shape. #226 is a vLLM-client fix only.
  - TTFT idiom: `(first or end) - t0` is in **4** files, not "at least 5":
    `conc_ladder.py:154` is a divergent variant (`(first - t0) if first
    else None`).
  - URL conventions: **six**, not five. `replay_offload_serve.py:32` takes
    `PORT` as positional argv (`sys.argv[2]`), missed before.
  - Line cites: `concurrent_collapse.py` key `:24` → `:31` (`:24` is its
    `def _key(`); `demo_capture.py` `DEMO_BASE` `:28` → `:29`;
    `replay_offload_serve.py` `metrics()` `:41-46` → `:41-49`;
    `docs/python-314.md` quote `:55-56` → `:55-57`.
  - `bench-probe-errors.patch` has been in the series since 3ae9171 (#165,
    2026-09-22), not "since d88544b"; d88544b (#189) re-exported it for
    0.30. vllm #58024 is still open (`gh pr view 58024 -R
    vllm-project/vllm`).
  - The model name is also hardcoded outside the 15 literals: the four
    bash `vllm bench serve` commands (`run_benchmarks.sh:34`,
    `real_rep.sh:17`, `prefill_ab.sh:30`, `warmup.sh:50`) and
    `replay_offload_serve.py:73`'s label strip. `real_rep.sh:17-18` also
    hardcodes port 18020. Added to 1.2, 1.3 and PR C.
  - `verify.sh` does not call `/v1/models` (`git grep -n v1/models --
    ':!docs'` hits only patches); the risk row that said it did is fixed.
  - Invocation: 10 of the 16 docstrings say `venv/bin/python bench/…`, 5
    say `python bench/…`, `replay_offload_serve.py:12` says `python
    replay.py` (a stale name). Either form puts the script's directory on
    `sys.path`; section 2 is corrected.
  - PR A's acceptance grep lacked `-E` (without it `|` is literal and the
    count is 0 today, proving nothing); PR B's `[DONE]` gate ignored
    `test_bench_sse_keepalive.py`. Both fixed.

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
  upstream. (Corrected 2026-09-29: this said "the repo's own copies remain
  unfixed"; they never had the defect — they parse line by line and skip
  non-`data:` lines, so `: keep-alive` is dropped. See the 2026-09-29 note.)

- `bench-probe-errors.patch` (vllm #58024, still open upstream; in the
  series since 3ae9171 (#165) and re-exported by the 0.30 port d88544b —
  this said "in since the 0.30 port d88544b")
  touches only `benchmarks/serve.py`: the `/tokenize` alignment probe now
  sends the Bearer and classifies its failure (404 route-or-name vs 401 vs
  unreachable vs timeout), and the two `/metrics` fetchers send the key. The
  benchmark requests themselves still present `OPENAI_API_KEY` alone and
  still report 0.00 in every field on a keyed server with only
  `VLLM_API_KEY` set (`/tmp/vllm-0.30.0`
  `vllm/benchmarks/lib/endpoint_request_func.py:155` reads only
  `OPENAI_API_KEY`) — the `docs/python-314.md:55-57` claim quoted in 1.1
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
`prefill_ab.sh`, `warmup.sh`) (census 2026-09-29: `grep -lE
"urllib|http\.client|curl |127\.0\.0\.1|localhost" bench/*.py bench/*.sh`
gives these 20 plus two tests that only name 127.0.0.1 for their own stub
or a `torch.distributed` init address, `test_bench_sse_keepalive.py:22,26` and
`test_marlin_int8_asym.py:31`) — and the five mechanisms they re-implement:
API-key resolution, server URL, model name, the request/stream helpers, and
the `/metrics` scrape. **Not in scope:** verdict/exit-code conventions
(candidate 5), the bench manifest (candidate 5), the offline tools
(`act_calib.py`, `make_long_corpus.py`, `mq3d_*`, `spec_attn_ctx_scan.py`,
`tune_gdn.py`, the `verbatim.py` library, the `test_*.py` files, and
`bench/demo/` — whose `render.mjs:35` server is its own local static
server, not the API; the first-pass list named only four of these), and
launcher-side logic (candidate 2's plan). `scripts/hq-doctor.sh` and the
`Makefile` (#219) are also out: they only probe `/health` (2026-09-29 note).

## 1. Current state, precisely

### 1.1 The API key: four flavors, and the canonical one is the least used

The canonical chain lives in `resolve_api_key.sh:32-42`
(`resolve_client_key`: OPENAI_API_KEY → VLLM_API_KEY → `$REPO/api_key.txt` →
`"EMPTY"`), written for #113. It is used by exactly the **4 bash scripts**
(`run_benchmarks.sh:25-26`, `real_rep.sh:12-13`, `prefill_ab.sh:24-25`,
`warmup.sh:29-30`). Everything else re-derives it, and **no Python script
checks OPENAI_API_KEY at all** (verified: `grep -c OPENAI_API_KEY bench/*.py`
is 0 everywhere; flavors re-counted 2026-09-29 with `grep -nE
"api_key\.txt|VLLM_API_KEY|OPENAI_API_KEY" bench/*.py` and `grep -n "def
_key(" bench/*.py`):

| flavor | shape | scripts (verified lines) |
|---|---|---|
| A: env-or-file | `VLLM_API_KEY` or `$REPO/api_key.txt` via a local `_key()` | `api_smoke.py:15`, `bugb_sweep.py:34`, `conc_ladder.py:68`, `concurrent_collapse.py:31` (was cited `:24`, its `def _key(` line), `interleave_dose.py:42`, `needle_reuse.py:43`, `needle_test.py:29`, `prefix_alternation.py:48`, `quality_battery.py:35`, `residue_sweep.py:39`, `seat_ttft.py:32` — **11 scripts**, each with its own `def _key(` (verified ×11) |
| B: hardcoded home path | `open(expanduser("~/qwen-serving/api_key.txt"))` — unguarded, `FileNotFoundError` anywhere else | `demo_capture.py:28`, `labd_accept.py:91`, `labd_bench.py:28`, `labd_soak.py:46` — **4 scripts** |
| C: env only | `os.environ.get("VLLM_API_KEY", "")` | `replay_offload_serve.py:33` |
| canonical | `resolve_api_key.sh` + `resolve_client_key` | the 4 bash scripts |

The failure this produces is exactly the one #113 fixed on the bash side: a
client that silently presents nothing (or the wrong thing) to a keyed server.
`docs/python-314.md:55-57` (was cited `:55-56`; the quoted sentence ends on
`:57`) documents the symmetric version for `vllm bench
serve` (which presents `OPENAI_API_KEY`): "with only the latter set it
receives silent 401s and reports 0.00 in every field rather than failing."
That sentence still describes the benchmark request path at 0.30.0 —
`bench-probe-errors` (vllm #58024) fixed only the `/tokenize` probe and the
`/metrics` scrapes (header block). The repo's own Python scripts are the
third surface, still un-fixed. (Unchanged at d5e2a01: #219 moved
`single-user/alternative.sh` onto `resolve_vllm_key`, the server side, and
its `scripts/hq-doctor.sh` sends no key — see the 2026-09-29 note.)

### 1.2 The server URL: six conventions (five before 2026-09-29; one was missed)

| convention | scripts |
|---|---|
| `PORT` env, `f"http://127.0.0.1:{PORT}/v1/…"` | `api_smoke.py:16-18` |
| `"http://127.0.0.1:" + PORT` | `bugb_sweep.py:35`, `residue_sweep.py:40`, `seat_ttft.py:33`, `concurrent_collapse.py:32` (the 6th `PORT`-env reader) |
| `API = f"http://127.0.0.1:{PORT}"` | `conc_ladder.py:53-54` |
| **`VLLM_API** env**, default `…:18020/v1` | `interleave_dose.py:43`, `needle_reuse.py:44`, `needle_test.py:30`, `prefix_alternation.py:49`, `quality_battery.py:36` |
| **hardcoded** `http://127.0.0.1:18020` (no override) | `labd_bench.py:29`, `labd_soak.py:47`; `DEMO_BASE` env in `demo_capture.py:29` (was cited `:28`, the key line); `--base` arg in `labd_accept.py:105` |
| **positional argv** `PORT` (`sys.argv[2]`), host hardcoded in each f-string | `replay_offload_serve.py:32` (URLs `:42`, `:56`) — missing from the first two passes |

Five scripts even picked a *different environment variable* (`VLLM_API`) from
the rest (`PORT`), two allow no override at all, and one takes the port on
its command line. Census: `grep -nE
"PORT|VLLM_API|127\.0\.0\.1|DEMO_BASE|--base" bench/<the 16>.py`.

The bash side is not uniform either: `run_benchmarks.sh:27` and
`warmup.sh:32-33` read `HOST`/`PORT` (default 18020), `prefill_ab.sh:18`
reads `PORT` defaulting to **18021**, and `real_rep.sh:17-18` hardcodes
`127.0.0.1:18020` in both its bench command and its `metrics()`.

### 1.3 The model name: hardcoded almost everywhere

`"qwen3.8-27b"` is a string literal in 15 call sites
(`api_smoke.py:31,75,84`, `bugb_sweep.py:66`, `conc_ladder.py:126`,
`demo_capture.py:56`, `labd_bench.py:83,96`, `labd_soak.py:84`,
`needle_test.py:59`, `quality_battery.py:70,99`,
`replay_offload_serve.py:53`, `residue_sweep.py:55`, `seat_ttft.py:52`;
`grep -nF '"qwen3.8-27b"' bench/*.py` gives 20 matches in 16 files: these
15 plus the 5 env/arg defaults below).
Three scripts accept a
`VLLM_MODEL` env with the same default (`interleave_dose.py:44`,
`needle_reuse.py:45`, `prefix_alternation.py:50`), `concurrent_collapse.py:33`
reads a second env spelling (`MODEL`, not `VLLM_MODEL`), one takes `--model`
(`labd_accept.py:106`). **No script asks the server** — `/v1/models` appears
nowhere in tracked `bench/` (`git grep -n v1/models -- bench` is empty; the
hits under `bench/results/` are git-ignored campaign scripts). Two more
hardcodings the literal grep misses (added 2026-09-29):
`replay_offload_serve.py:73` strips the label
`{engine="0",model_name="qwen3.8-27b",` from metric keys, and all four bash
scripts pass `--model qwen3.8-27b … --served-model-name qwen3.8-27b` to
`vllm bench serve` (`run_benchmarks.sh:34`, `real_rep.sh:17`,
`prefill_ab.sh:30`, `warmup.sh:50`). A served-name change (the
`serve-model-path-match`/`serve-404-served-names` patches exist precisely
because this bites) is a 20-file edit (16 Python + 4 bash; was "16-file").

The bash side has its own drift: `run_benchmarks.sh:28` and `real_rep.sh:14`
default `MODEL` to the base checkpoint, `prefill_ab.sh:26` to `-fast`, and
only `warmup.sh:41` uses the shared `select_model.sh` (the candidate-2 plan
covers the launcher-side selector).
### 1.4 The mechanisms, counted

Measured against the 16 Python scripts (the review said "~1,500 lines of
copy-paste" — that was an overestimate; the honest numbers):

- **164 lines** match the narrow glue idiom grep (`def _key|def post|def
  metrics|KEY =|BASE =|urllib|Bearer|data: |\[DONE\]|include_usage|ttft`,
  `cat <the 16> | grep -cE`; 163 if `urllib` is narrowed to
  `urllib\.request`) across the 16 files (re-measured 2026-09-28 and again
  2026-09-29: 155 on the original fifteen, +9 in `concurrent_collapse.py`).
  Counting the surrounding blocks puts the re-implemented total at
  **~420 lines**. That figure was never tied to a method; 2026-09-29 pinned
  it with a throwaway `ast` walk over the 16 files: the union of the
  `_key`/`post`/`metrics` function spans, module-level
  `KEY`/`BASE`/`API`/`URL`/`URLC`/`PORT`/`MODEL`/`H` assignments, and every
  non-compound statement (a `with` block counts whole) that calls
  `urlopen`/`Request` is **308 lines**; adding request-payload statements
  (a dict with a `model` or `stream_options` key) and assignments using
  `t0`/`first`/`t_first`/`t_end` is **437**. The ~420 sits in that bracket;
  the payload dicts carry each script's prompts, so the true glue count is
  nearer the lower bound than the upper one.
  The duplication is broad rather than deep — which is exactly why it drifted: no single copy is big enough
  to look like a module.
- `def _key(` — **11 copies** (the file-read-with-fallback helper).
- `def post(` — **5** (`api_smoke.py:21-27`, `quality_battery.py:43-46`,
  `needle_test.py`, `labd_accept.py`, `concurrent_collapse.py:41-48`); the
  other 11 scripts inline the same
  `urllib.request.Request` + headers block at each call site (e.g.
  `labd_bench.py:88-91` and `:101-103`, twice in one file).
- SSE stream parsers (`data:` / `[DONE]` / `include_usage` / first-token
  timing) — **7** (`grep -l '\[DONE\]' bench/*.py`, less the
  `test_bench_sse_keepalive.py` fixture): `api_smoke.py:74-77` (was
  `:74-78`; `:78` is the next def), `conc_ladder.py:124-156`,
  `demo_capture.py` (~:61-98), `labd_accept.py:166-195`,
  `labd_bench.py:109-125`, `replay_offload_serve.py:53-74`,
  `seat_ttft.py:49-77`. The TTFT idiom `(t_first or t_end) - t0` recurs in
  **4** (`demo_capture.py:90`, `labd_bench.py:130`,
  `replay_offload_serve.py:71`, `labd_accept.py:195`); this said "at least
  5" and listed `conc_ladder.py:154`, which is a divergent fifth shape
  (`(first - t0) if first else None`) — drift, not a copy.
- `def metrics(` — **7 Python** (`bugb_sweep.py`, `conc_ladder.py:114-121`,
  `labd_accept.py`, `labd_bench.py:42-50`, `labd_soak.py`,
  `replay_offload_serve.py:41-49` (was cited `:41-46`), `residue_sweep.py`)
  **+ 3 bash**
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
   the comment "Build the command as an array, not a string: an unquoted "$B"
   re-splits and re-globs" (verbatim at d5e2a01, `$B` in quotes; earlier
   passes dropped them) — and `run_benchmarks.sh:34`, `real_rep.sh:17`,
   `prefill_ab.sh:30` still build `B=` as a string and expand it unquoted
   (`grep -n '"\$B"' bench/{run_benchmarks,real_rep,prefill_ab}.sh` is
   empty).
3. **`prefill_ab.sh:39-40` re-implements the launcher's INT8 export guards
   in the drifted shape** (exports `VLLM_MARLIN_INT8_INCLUDE_RE` whenever
   `INT8_LAYERS` is non-empty, `INT8_ACT` or not — the candidate-2 plan's
   1.3.1 bug shape, in a fourth file; at d5e2a01 the batch copy is
   `batch/start_qwen.sh:276`, `[ -n "$INT8_LAYERS" ] && export …`; the
   two single-user copies carry the two-condition guard,
   `single-user/start_qwen.sh:142` and `single-user/alternative.sh:56`).
   Its own comment, `prefill_ab.sh:36-37`, says it copies batch because the
   "single-user script has no wiring yet" — stale: it boots
   `single-user/start_qwen.sh`, which has that wiring.
4. **Model defaults split three ways** on the bash side (1.3): base, `-fast`,
   and the one correct consumer of `select_model.sh`.

None of these is a big bug. All of them are the same bug: a fix lands in the
copy the reporter was running, and the harness has no place to put it for
everyone.
## 2. The design: `bench/harness.py`

One stdlib-only module (every script today runs on the venv's bare Python —
no `requests`, and adding a dependency for this would be a regression).
Scripts import it as `import harness`: each runs as `python
bench/<script>.py` — 10 docstrings say `venv/bin/python bench/…`, 5 say
`python bench/…`, and `replay_offload_serve.py:12` says `python replay.py`
(stale name; fix it in PR C). Every form runs a file path, which puts the
script's directory on `sys.path` — no shim, no package surgery, no
`sys.path.insert` (the anti-pattern candidate 4 flags in `drafter/`). (This
said "`venv/bin/python bench/<script>.py` (verified in their docstrings)";
6 of 16 do not say that.) Precedent in `bench/` already: `bugb_sweep.py:23`
and `residue_sweep.py:23` do `sys.path.insert` before `from verbatim
import …` — redundant under the same rule; PR A may drop them.

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
    The server is the source of truth; the 15 hardcoded literals, the 5
    env/arg defaults and replay_offload_serve.py:73's label strip die.

post(path, payload, timeout=1200) -> dict
    JSON POST to base_url()+path with the key header. Replaces 5 defs and
    11 inline copies.

stream_chat(payload, timeout=1800) -> (ttft_s, decode_s, ntok, usage, text)
    The SSE parser once: data: lines, [DONE], usage capture, first-content
    timing, the (first or end) - t0 idiom with the documented caveat.
    Replaces 7 parsers. Not a keep-alive fix: the 7 are already immune to
    the #226 defect (2026-09-29 note); this is consolidation only.
    conc_ladder.py:154 returns None when no token arrives (the other 4
    fall back to t_end) — keep that distinguishable in the return value.

metrics(*names) -> dict[str, float]
    GET {base}/metrics with the key; sums label-variant lines per name;
    _created lines excluded by construction (1.5.1 can never recur).

spec_delta(m0, m1) -> (steps, tok_per_step)
    The drafts/accepted arithmetic by NAME, not position (1.5.1).
```

Bash side: the three `spec()`/`metrics()` helper pairs keep `curl` (two
one-line functions per script, e.g. `run_benchmarks.sh:38-39`; this said
"five lines") but gain the `_created` filter — or, tidier, call
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
hard-fail outside `~/qwen-serving`). `replay_offload_serve.py` takes its
port as `argv[2]` (1.2's sixth convention); moving it to `base_url()`
changes its command line (`TAG PORT DEPTH N` → `PORT=… TAG DEPTH N`), so
call that out too, or keep the positional and export it into `PORT` before
the first `base_url()` call.

Acceptance: `grep -cE 'def _key\(|api_key\.txt' bench/*.py` → 0 outside
harness.py (15 files match today; this said `grep -c 'def _key(|api_key.txt'`,
which without `-E` matches the `|` literally and already reads 0 everywhere);
`bench/test_harness.py` green in CI; `api_smoke.py` output
against a live server byte-identical to pre-migration (modulo timings).

### 3.2 PR B — stream/metrics migration + the bash guards

Add `stream_chat`, `metrics`, `spec_delta`. Migrate the 7 SSE parsers and 7
Python scrapers. Bash: add the `_created` filter to `run_benchmarks.sh:39`
and `prefill_ab.sh:57` (matching `real_rep.sh:19` — one line each, zero risk
to measurement), and convert the three `B=` strings to arrays with
`warmup.sh:43-46`'s comment attached. `real_rep.sh:17` is one of those
lines; while there, take the port from `PORT` like `run_benchmarks.sh:27`
instead of the hardcoded 18020 at `:17-18` (1.2).

Acceptance: `grep -l '\[DONE\]' bench/*.py` → only harness.py,
test_harness.py and `test_bench_sse_keepalive.py`'s fixture transcript (its
`:18`; this gate forgot it, section 6 did not); the two bash `spec()`
functions carry the filter;
`tok/step` on a live dflash2 server unchanged (the filter is a no-op when
no `_created` lines exist).

### 3.3 PR C — the model name + docs

Add `model()`; replace the 15 literals, the `VLLM_MODEL`/`MODEL`
duplications, and `replay_offload_serve.py:73`'s hardcoded
`model_name="qwen3.8-27b"` label strip;
the three bash scripts take their model from `select_model.sh` (candidate 2
alignment), and all four take the served name for `--model …
--served-model-name …` (`run_benchmarks.sh:34`, `real_rep.sh:17`,
`prefill_ab.sh:30`, `warmup.sh:50`) from `VLLM_MODEL` or `harness.py
--model` instead of the literal. Update the affected docstrings (several
document `VLLM_API_KEY … PORT=` on their usage lines, e.g. `api_smoke.py:5`;
`replay_offload_serve.py:12` still calls itself `replay.py`).

Acceptance: `grep -c '"qwen3.8-27b"' bench/*.py` → 0 outside harness tests
(20 matches in 16 files today) and `grep -n 'qwen3\.8-27b' bench/*.sh` → 0;
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
| `import harness` breaks under an exotic invocation (`python -m`, symlinked script) | low | the documented invocations (`venv/bin/python bench/x.py` or `python bench/x.py`, section 2) put `bench/` on `sys.path`; the test file imports it the same way; a fallback two-liner is acceptable if a real caller breaks |
| Flavor-B scripts silently change behavior for their one known user | medium | called out in PR A's message; the change is strictly "works where it used to crash" |
| `model()` adds a `/v1/models` round-trip per invocation | low | cached per process; the endpoint is cheap. (This said it is "already hit by `verify.sh`'s server checks"; it is not — `verify.sh` hits `/health`, `/v1/chat/completions`, `/tokenize` and `/v1/tokenize` only (`grep -noE 'PORT/[a-zA-Z0-9/_.-]+' verify.sh`), and `git grep -n v1/models -- ':!docs'` hits only patches.) |
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
5. `grep -c '"qwen3.8-27b"' bench/*.py` → 0 outside tests, and `grep -n
   'qwen3\.8-27b' bench/*.sh` → 0 (the four bash bench commands, 1.3).
6. `bench/test_harness.py` runs in `patch-integrity.yml` in seconds, green.

At that point "how the harness talks to the server" has one answer per
mechanism, the next #113 lands in one file, and the harness's ~420 lines of
re-implemented glue is a module the tests actually cover.
