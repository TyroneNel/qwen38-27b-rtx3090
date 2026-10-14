"""Batch 3's Running: probe, one wave: exactly N concurrent requests (default 8 = MAX_SEQS), distinct prompts,
512 output tokens each (ignore_eos, so every stream runs its full length), released together on a barrier.

  venv/bin/python /tmp/running_probe.py <label> <wave> [--n 8] [--out 512] [--ctx-tokens 1024]

Runs inside the container next to bench/harness.py (it adds /app/bench to sys.path). It prints one JSON line:
the wave's UTC window, each request's submit / first-token / last-token time, and the scheduler gauges polled from
/metrics every 0.5 s (vllm:num_requests_running / _waiting, vllm:num_preemptions_total). The primary metric is not
here: it is the engine's own `Running: N reqs, Waiting: M reqs` log line, which arm.sh `probe` pulls from the
container log for this wave's window afterwards. `held` is the window in which the load is held: from the last
submission to the first stream finishing; a log line inside it was taken while all N requests were in the server.

Prompts are ~ctx-tokens long (bench/conc_ladder.py's FILLER, 44 tokens per repeat) with a per-request, per-wave,
per-label salt at the front, so no request is served from the prefix cache and the KV pool is not the limit
(8 x ~1.6k tokens against a pool of >100k). The probe measures seats, not context.
"""
import datetime, json, sys, threading, time

sys.path.insert(0, "/app/bench")
import harness  # noqa: E402


def _arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


LABEL, WAVE = sys.argv[1], int(sys.argv[2])
N = _arg("--n", 8)
OUT = _arg("--out", 512)
CTX = _arg("--ctx-tokens", 1024)
FILLER = ("The RTX 3090 has 24 GB of GDDR6X and 82 streaming multiprocessors. "
          "Memory bandwidth is 936 GB/s, which is what decode is bound by. ")
REPEATS = max(1, round(CTX / 44))


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def prompt(i):
    salt = f"[probe {LABEL} wave {WAVE} stream {i}] "
    return (salt + FILLER * REPEATS + f"\nStream {i} of wave {WAVE}: continue this text in your own words, "
            "at length, without repeating it.\n")


res = [dict(i=i) for i in range(N)]
barrier = threading.Barrier(N)


def run(i):
    r = res[i]
    payload = {"prompt": prompt(i), "max_tokens": OUT, "min_tokens": OUT, "ignore_eos": True, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    barrier.wait()
    r["submit"] = time.time()
    try:
        for frame in harness.stream("/v1/completions", payload):
            now = time.time()
            if frame.get("choices") and frame["choices"][0].get("text"):
                r.setdefault("first", now)
                r["last"] = now
            if frame.get("usage"):
                r["completion_tokens"] = frame["usage"]["completion_tokens"]
                r["prompt_tokens"] = frame["usage"]["prompt_tokens"]
    except Exception as e:  # recorded, not raised: a failed stream is evidence about the cap
        r["error"] = repr(e)
    r["end"] = time.time()


gauges, stop = [], threading.Event()


def poll():
    names = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_preemptions_total")
    while not stop.is_set():
        try:
            m = harness.metrics(*names)
            gauges.append([round(time.time(), 2)] + [m.get(n) for n in names])
        except Exception as e:
            gauges.append([round(time.time(), 2), repr(e)])
        stop.wait(0.5)


p = threading.Thread(target=poll, daemon=True)
p.start()
t0 = time.time()
threads = [threading.Thread(target=run, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
t1 = time.time()
stop.set()
p.join()
held = (max(r["submit"] for r in res), min(r.get("last", r["end"]) for r in res))
for r in res:
    for k in ("submit", "first", "last", "end"):
        if k in r:
            r[k] = iso(r[k])
print(json.dumps({
    "label": LABEL, "wave": WAVE, "n": N, "out": OUT, "ctx_tokens": CTX,
    "t0": iso(t0), "t1": iso(t1), "held_from": iso(held[0]), "held_to": iso(held[1]),
    "requests": res,
    "gauges_running_waiting_preemptions": gauges,
}))
