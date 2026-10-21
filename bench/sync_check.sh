#!/bin/bash
# Hot-path GPU<->CPU sync gate (DeepInfer evals-2). vLLM 0.31's VLLM_GPU_SYNC_CHECK=warn arms torch's
# sync-debug mode around execute_model and sample_tokens once warmup is done (vllm/utils/gpu_sync_debug.py),
# so every synchronizing call on the hot path -- and every non_blocking copy from pageable memory -- prints a
# Python warning naming its file and line:
#   <file>.py:<line>: UserWarning: called a synchronizing CUDA operation (Triggered internally at ...)
#   <file>.py:<line>: UserWarning: Tensor.copy_: a non_blocking H2D copy that may be silently synchronous (...)
# This script drives traffic through a server booted with it, collects the unique sites from the server log
# and fails on any site missing from bench/sync_allowlist.txt.
#
#   bash bench/sync_check.sh <server log> [--allowlist PATH] [--tokens N] [--settle S]
#   bash bench/sync_check.sh <server log> --parse-only        # no traffic; just the log against the list
#   bash bench/sync_check.sh <server log> --record [--mode M] # append the log's new sites to the list, exit 0
#
# The server: booted with VLLM_GPU_SYNC_CHECK=warn (never error: error kills the engine at the first site),
# reached as every bench script reaches it (bench/harness.py: VLLM_API or PORT, key chain). The log: a file
# that holds the server's stdout+stderr and is still being written (e.g. the launcher under `2>&1 | tee`, or
# `docker logs -f <c> > file 2>&1 &` on the host); every warning in it counts, from boot on. Traffic:
# bench/api_smoke.py (its 13 features, including a 20k prompt and a streamed tool call), then the 8 prompts in
# bench/prompts_real.jsonl at the server's default sampling, thinking off, N tokens each (default 256).
#
# Run inside the serving container, it also checks /proc that the vllm process was started with the knob
# (exit 2 if not); from outside it says the armed state is unverified.
#
# Allowlist: one site per line, `<path>:<line> [kind]  # why`, paths relative to site-packages (vllm/...) or to
# the repo; no kind matches every kind at that line. A line moves on every re-pin, so a moved site shows as new
# and is re-reviewed. Kinds: sync (torch's sync check), copy_-H2D, to-H2D, cuda-H2D, copy_-D2H (the implicit
# non_blocking copy check).
#
# Exit: 0 every site allowlisted (or --record); 1 a new site, or an error-mode raise in the log; 2 invalid run:
# no log, no request reached the server, no real prompt succeeded, or the server is visibly not armed.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
export PATH="$REPO/venv/bin:$PATH"
PY=$(command -v python3 || command -v python) || { echo "sync_check: no python3" >&2; exit 2; }

LOG=; ALLOW="$HERE/sync_allowlist.txt"; PARSE_ONLY=0; RECORD=0; MODE_TAG=; TOKENS=256; SETTLE=3
while [ $# -gt 0 ]; do
  case "$1" in
    --allowlist) ALLOW=$2; shift 2 ;;
    --parse-only) PARSE_ONLY=1; shift ;;
    --record) RECORD=1; shift ;;
    --mode) MODE_TAG=$2; shift 2 ;;
    --tokens) TOKENS=$2; shift 2 ;;
    --settle) SETTLE=$2; shift 2 ;;
    -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "sync_check: unknown option $1" >&2; exit 2 ;;
    *) [ -z "$LOG" ] && LOG=$1 || { echo "sync_check: one log path only" >&2; exit 2; }; shift ;;
  esac
done
[ -n "$LOG" ] || { echo "usage: bash bench/sync_check.sh <server log> [--allowlist PATH] [--parse-only] [--record]" >&2; exit 2; }
[ -f "$LOG" ] || { echo "sync_check: INVALID: no log file at $LOG" >&2; exit 2; }

if [ "$PARSE_ONLY" = 0 ] && [ "$RECORD" = 0 ]; then
  # Armed? Visible only from inside the server's container (or host, for a native install).
  "$PY" - <<'PY'
import glob, sys
seen = armed = 0
for d in glob.glob("/proc/[0-9]*"):
    try:
        cmd = open(d + "/cmdline", "rb").read().replace(b"\0", b" ")
        if b"vllm" not in cmd or (b"serve" not in cmd and b"EngineCore" not in cmd):
            continue
        env = open(d + "/environ", "rb").read().split(b"\0")
    except OSError:
        continue
    seen += 1
    armed += any(e in (b"VLLM_GPU_SYNC_CHECK=warn", b"VLLM_GPU_SYNC_CHECK=error") for e in env)
if not seen:
    print("sync_check: no vllm process visible here; cannot confirm VLLM_GPU_SYNC_CHECK=warn (unverified)")
elif not armed:
    print("sync_check: INVALID: the vllm process was not started with VLLM_GPU_SYNC_CHECK=warn; a clean log would mean nothing")
    sys.exit(2)
else:
    print("sync_check: server armed (VLLM_GPU_SYNC_CHECK set in the vllm process environment)")
PY
  rc=$?; [ $rc = 0 ] || exit $rc

  echo "sync_check: [1/2] api_smoke.py"
  (cd "$HERE" && "$PY" api_smoke.py); SMOKE=$?
  case $SMOKE in
    0) ;;
    2) echo "sync_check: INVALID: api_smoke reached no server"; exit 2 ;;
    *) echo "sync_check: api_smoke had failures (exit $SMOKE); the sync verdict below does not depend on them" ;;
  esac

  echo "sync_check: [2/2] 8 real prompts x $TOKENS tokens"
  (cd "$HERE" && "$PY" - "$TOKENS" <<'PY'
import json, sys, time
import harness
n_ok = 0
for i, line in enumerate(open("prompts_real.jsonl")):
    p = json.loads(line)["prompt"]
    t0 = time.time()
    try:
        r = harness.post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}],
                         "max_tokens": int(sys.argv[1]), "chat_template_kwargs": {"enable_thinking": False}},
                         timeout=900)
        n = r["usage"]["completion_tokens"]
        n_ok += 1
        print(f"  prompt {i}: {n} tokens in {time.time() - t0:.1f} s")
    except Exception as e:  # noqa: BLE001
        print(f"  prompt {i}: FAILED {type(e).__name__}: {str(e)[:200]}")
sys.exit(0 if n_ok else 2)
PY
  ); [ $? = 0 ] || { echo "sync_check: INVALID: no real prompt succeeded"; exit 2; }
  sleep "$SETTLE"   # let a `docker logs -f > file` writer catch up
fi

"$PY" - "$LOG" "$ALLOW" "$RECORD" "$MODE_TAG" "$REPO" <<'PY'
import os, re, sys, time
log, allow, record, mode, repo = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4], sys.argv[5]
# A Python warning line, anywhere in the line: docker timestamps and "(EngineCore pid=N)" prefixes pass. Both
# torch's sync warning and gpu_sync_debug's copy warning are UserWarnings.
WARN = re.compile(r"(?P<path>(?:[A-Za-z]:)?[^\s:'\"]+\.py):(?P<line>\d+): UserWarning: (?P<msg>.*)")
COPY = re.compile(r"Tensor\.(?P<op>to|cuda|copy_): a non_blocking (?P<dir>H2D|D2H) copy")
ERRMODE = "GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()"
roots = [r.rstrip("/") + "/" for r in {repo, "/app"} if r]


def rel(path):
    path = path.replace("\\", "/")
    if "site-packages/" in path:
        return path.split("site-packages/", 1)[1]
    for r in roots:
        if path.startswith(r):
            return path[len(r):]
    return path


sites, raises = {}, 0
with open(log, errors="replace") as f:
    for text in f:
        if ERRMODE in text and "RuntimeError" in text:
            raises += 1
        m = WARN.search(text)
        if not m:
            continue
        msg = m.group("msg")
        if "called a synchronizing CUDA operation" in msg:
            kind = "sync"
        elif "Implicit GPU<->CPU sync detected" in msg:
            c = COPY.search(msg)
            kind = f"{c.group('op')}-{c.group('dir')}" if c else "implicit"
        else:
            continue
        key = (rel(m.group("path")), int(m.group("line")), kind)
        n, first = sites.get(key, (0, msg))
        sites[key] = (n + 1, first)

allowed = set()
if os.path.exists(allow):
    for raw in open(allow):
        entry = raw.split("#", 1)[0].split()
        if not entry:
            continue
        loc = entry[0]
        path, _, line = loc.rpartition(":")
        if not path or not line.isdigit():
            print(f"sync_check: allowlist line not understood, ignored: {raw.strip()}")
            continue
        allowed.add((path, int(line), entry[1] if len(entry) > 1 else None))


def is_allowed(path, line, kind):
    return (path, line, None) in allowed or (path, line, kind) in allowed


new = []
for (path, line, kind), (n, msg) in sorted(sites.items()):
    ok = is_allowed(path, line, kind)
    new += [] if ok else [(path, line, kind)]
    print(f"  {'allow' if ok else 'NEW  '} {path}:{line} {kind}  x{n}  {msg[:120]}")
if raises:
    print(f"sync_check: {raises} error-mode raise(s) in the log (VLLM_GPU_SYNC_CHECK=error); boot with warn to name the sites")
if record and new:
    with open(allow, "a") as f:
        f.write(f"# recorded {time.strftime('%Y-%m-%d')}{' ' + mode if mode else ''} from {os.path.basename(log)}\n")
        for path, line, kind in new:
            f.write(f"{path}:{line} {kind}\n")
    print(f"sync_check: recorded {len(new)} site(s) in {allow}")
verdict = "RECORDED" if record else ("FAIL" if new or raises else "PASS")
print(f"SYNC_CHECK {verdict} sites={len(sites)} new={len(new)} allowlisted={len(sites) - len(new)} "
      f"warnings={sum(n for n, _ in sites.values())} error_raises={raises} log={log}")
sys.exit(0 if record or not (new or raises) else 1)
PY
