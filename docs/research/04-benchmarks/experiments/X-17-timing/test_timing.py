#!/usr/bin/env python3
"""CPU unit test: SpecTimer (from spec-timing.patch) with fake event times, and parse_timing.py.

    python3 test_timing.py        # prints OK and exits 0; needs torch (CPU build is fine)

The module under test is extracted from the '+' lines of the new-file hunk in spec-timing.patch,
so this checks the exported patch, not the scratch tree.
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_spec_timing():
    lines, on = [], False
    for ln in open(os.path.join(HERE, "spec-timing.patch")):
        if ln.startswith("diff --git") and ln.rstrip().endswith("spec_timing.py"):
            on = True
        elif ln.startswith("diff --git"):
            on = False
        elif on and ln.startswith("+") and not ln.startswith("+++"):
            lines.append(ln[1:])
    d = tempfile.mkdtemp()
    p = os.path.join(d, "spec_timing.py")
    open(p, "w").write("".join(lines))
    spec = importlib.util.spec_from_file_location("spec_timing", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


st = load_spec_timing()
sys.path.insert(0, HERE)
import parse_timing as pt  # noqa: E402


class Dev:
    """Fake GPU: events get the time the test sets; query() is true once the device passed it."""

    def __init__(self):
        self.t = 0.0
        self.done = -1.0
        self.syncs = 0


class Ev:
    def __init__(self, dev):
        self.dev, self.t = dev, None

    def record(self):
        self.t = self.dev.t

    def query(self):
        return self.t is not None and self.t <= self.dev.done

    def elapsed_time(self, other):
        assert self.query() and other.query(), "elapsed_time on an unfinished event would sync"
        return other.t - self.t

    def synchronize(self):
        self.dev.syncs += 1
        self.dev.done = max(self.dev.done, self.t)


class T(st.SpecTimer):
    def _ensure_bufs(self, dtype):  # no pinned memory on a CPU torch
        if self._buf_dtype is None:
            self._buf_dtype = dtype
            for sl in self._slots:
                sl.buf = torch.empty(self._max_reqs, dtype=dtype)


def mk(nslots=8, every=1000, pin=True, capturing=lambda: False):
    dev, out = Dev(), []
    t = T(lambda: Ev(dev), out.append, every=every, nslots=nslots, pin=pin,
          clock=lambda: 1000.0 + dev.t / 1000, capturing=capturing)
    return dev, out, t


def step(dev, t, t0, d, drafts=7, reqs=1, emitted=(3,), graph=True, draft=True):
    """One step: marks at t0 and t0+cumulative d = (verify, logits, accept, other, draft)."""
    dev.t = t0
    t.begin(reqs, reqs * (drafts + 1), drafts, graph)
    dev.t += d[0]; t.mark(1)
    dev.t += d[1]; t.mark(2)
    dev.t += d[2]; t.sampled(torch.tensor(emitted, dtype=torch.int32))
    dev.t += d[3]
    if draft:
        t.mark(4)
        dev.t += d[4]; t.mark(5)
    t.end()
    return dev.t


def close(a, b):
    assert abs(a - b) < 1e-9, (a, b)


# 1. phases, gap, keys, tokens, histogram; nothing resolves until the device passes the marks
dev, out, t = mk()
e1 = step(dev, t, 0.0, (10, 1, 1.5, 0.5, 3), emitted=(3,))        # verify 10 logits 1 accept 1.5 other .5 draft 3
e2 = step(dev, t, e1 + 1.0, (12, 1, 1.5, 0.5, 4), emitted=(5,))  # gap 1.0 after step 1
assert t.steps == 0, "no step may resolve while its marks are still on the device"
dev.done = e2 + 100
dev.t = e2 + 2.0
t.end()                      # end() drains the ring without a sync
assert t.steps == 2 and dev.syncs == 0, (t.steps, dev.syncs)
k = t.keys["spec/r1/G"]
assert k["n"] == 2 and k["gap_n"] == 1 and k["emit"] == 8 and k["drafts"] == 14, k
close(k["ms"]["verify"], 22.0); close(k["ms"]["draft"], 7.0); close(k["ms"]["accept"], 3.0)
close(k["ms"]["gap"], 1.0); close(k["ms"]["total"], 16.0 + 19.0)
h = t.hist["spec"]["verify"]  # 10 -> bucket (7.5,10], 12 -> (10,15]
assert h[st.EDGES_MS.index(10)] == 1 and h[st.EDGES_MS.index(15)] == 1 and sum(h) == 2, h

# 2. a plain step (no drafts verified) gets its own key; with no speculator (SPEC=off) no draft phase
dev, out, t = mk()
e = step(dev, t, 0.0, (20, 2, 0.5, 0.2, 0), drafts=0, reqs=2, emitted=(1, 1), graph=False, draft=False)
dev.done = 1e9; dev.t = e + 5; t.end()
p = t.keys["plain/r2/E"]
assert "draft" not in p["ms"] and p["emit"] == 2, p
close(p["ms"]["total"], 22.7)            # 0 -> last mark (4, closed by end())

# 3. full ring: the new step is dropped and counted, never blocks or syncs
dev, out, t = mk(nslots=3)
x = 0.0
for _ in range(5):
    x = step(dev, t, x + 1, (1, 1, 1, 1, 1))
assert t.dropped >= 1 and dev.syncs == 0, (t.dropped, dev.syncs)
dev.done = 1e9; dev.t = x + 1; t.end()
assert t.steps >= 1

# 3b. (review) no gap is measured across a dropped step
dev, out, t = mk(nslots=2)
x = step(dev, t, 0.0, (1, 1, 1, 1, 1))       # s1 0..5
x = step(dev, t, x + 1, (1, 1, 1, 1, 1))     # s2 6..11
x = step(dev, t, x + 1, (1, 1, 1, 1, 1))     # s3: ring full -> dropped
assert t.dropped == 1
dev.done = x
x = step(dev, t, x + 50, (1, 1, 1, 1, 1))    # s4: full ring -> begin() resolves s1, s2 first
dev.done = 1e9; t.end()
k = t.keys["spec/r1/G"]
assert k["n"] == 3 and k["gap_n"] == 1, k    # s1->s2 only; s2->s4 spans the dropped s3
close(k["ms"]["gap"], 1.0)

# 3c. (review) begin() does not resolve while the ring has room; end() does (the step's own
#     work is still queued then, so the bookkeeping does not delay the next forward launch)
dev, out, t = mk()
e = step(dev, t, 0.0, (1, 1, 1, 1, 1))
dev.done = 1e9; dev.t = e + 1
t.begin(1, 8, 7, True)
assert t.steps == 0, "begin() resolved on the launch path"
t.end()
assert t.steps == 2, t.steps

# 4. a step that began and never reached end() is abandoned, its slot is reused
dev, out, t = mk()
dev.t = 1.0; t.begin(1, 8, 7, True)
dev.t = 2.0; t.begin(1, 8, 7, True)
assert t.abandoned == 1 and len(t._free) == len(t._slots) - 1

# 5. capture guard: nothing recorded while a stream is capturing
dev, out, t = mk(capturing=lambda: True)
t.begin(1, 8, 7, True); t.mark(1); t.end()
assert t._cur is None and not t._pending

# 6. no pinned memory -> no token counts, no copy
dev, out, t = mk(pin=False)
e = step(dev, t, 0.0, (1, 1, 1, 1, 1)); dev.done = 1e9; dev.t = e + 1; t.end()
assert "emit" not in t.keys["spec/r1/G"] and t.snapshot()["tokens"] is False

# 7. log lines -> parser: cadence, prefixes, windows, cumulative diff, stats
dev, out, t = mk(every=2)
x = 0.0
for i in range(6):
    x = step(dev, t, x + 1.0, (10, 1, 1, 1, 4), emitted=(4,))
    dev.done = x
x += 1; dev.t = x; dev.done = 1e9; t.end()
assert len(out) == 3 and all(l.startswith("HQ_SPEC_TIMING {") for l in out), len(out)
log = ["INFO noise", "HQ_SPEC_TIMING_START {\"v\":1}"] + [f"(EngineCore pid=7) INFO 10-21 12:00:0{i} {l}" for i, l in enumerate(out)]
f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False); f.write("\n".join(log)); f.close()
runs = pt.read_lines([f.name])
assert list(runs) == [t.run] and len(runs[t.run]) == 3
base, end = pt.pick(runs[t.run], None, None)
assert base is None and end["steps"] == 6
s = pt.summarise(pt.diff_snap(base, end), pt.DEFAULT_EDGES)
b = s["kinds"]["spec"]
close(b["phases"]["verify"]["mean_ms"], 10.0); close(b["phases"]["draft"]["mean_ms"], 4.0)
close(b["phases"]["gap"]["mean_ms"], 1.0)            # 5 gaps of 1.0 (step 1 has no predecessor)
close(b["tok_per_step"], 4.0)
close(b["accept_rate"], (4 - 1) / 7)  # 3 accepted of 7 drafts per request-step
close(b["phases"]["draft"]["share_of_total"], 4 / 17)
assert b["phases"]["verify"]["p50_ms"] == 10 and b["phases"]["draft"]["p95_ms"] == 4
# implied rate: 24 tokens over 6*17 ms + 5 gaps
close(b["implied_tok_s"], 24 / ((6 * 17 + 5) / 1000))
# window by time: baseline = last line at/before the 1st line's ts -> remaining 4 steps
first = runs[t.run][0]
base, end = pt.pick(runs[t.run], first["ts"], None)
d = pt.diff_snap(base, end)
assert d["steps"] == 4 and d["keys"]["spec/r1/G"]["emit"] == 16, d["steps"]
assert len(list(pt.windows(runs[t.run]))) == 3
# CLI end to end, incl. --out and the exit code on empty input
outj = f.name + ".json"
buf = io.StringIO()
sys.argv = ["parse_timing.py", f.name, "--label", "T", "--out", outj]
with redirect_stdout(buf):
    assert pt.main() == 0
assert "spec (all)" in buf.getvalue() and json.load(open(outj))[t.run]["label"] == "T"
sys.argv = ["parse_timing.py", os.devnull]
with redirect_stderr(io.StringIO()):
    assert pt.main() == 2  # empty input: exit 2, message on stderr
assert pt.DEFAULT_EDGES == list(st.EDGES_MS), "parser and patch histogram edges differ"

# 8. off by default: make_timer() is None without importing torch/vllm
os.environ.pop("HQ_SPEC_TIMING", None)
assert st.make_timer() is None
os.environ["HQ_SPEC_TIMING"] = "0"
assert st.make_timer() is None

# 9. flush() syncs once (exit-time only) and emits the tail
dev, out, t = mk(every=1000)
e = step(dev, t, 0.0, (1, 1, 1, 1, 1))
t.flush()
assert dev.syncs == 1 and t.steps == 1 and len(out) == 1

print("OK")
