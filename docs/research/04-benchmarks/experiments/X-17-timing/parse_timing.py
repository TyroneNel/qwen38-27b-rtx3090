#!/usr/bin/env python3
"""Turn "HQ_SPEC_TIMING {json}" log lines (spec-timing.patch) into a draft/verify/accept summary.

The lines carry CUMULATIVE sums, so a run's numbers are (last line) - (baseline line):
  docker compose logs single | python3 parse_timing.py --label B-pth0
  python3 parse_timing.py container.log --t0 1761050000 --t1 1761050600 --out spectiming_B-pth0.json
  python3 parse_timing.py container.log --windows          # one row per emitted window
--t0/--t1 are unix seconds (the lines' "ts"; the WSL2 wall clock steps by ~2 s, windows are
quantised to HQ_SPEC_TIMING_EVERY steps anyway). Without them the whole boot is summarised,
warm-up included. Stdlib only.
"""
import argparse
import json
import math
import sys

MARK = "HQ_SPEC_TIMING "
PHASES = ("gap", "verify", "logits", "accept", "other", "draft", "total")
STEP_PARTS = ("verify", "logits", "accept", "other", "draft")  # sum to total; gap is separate
DEFAULT_EDGES = [0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, 100, 250]


def read_lines(files):
    """-> {run: [snapshot, ...]} in file order; non-matching lines are skipped."""
    runs = {}
    for f in files or ["-"]:
        fh = sys.stdin if f == "-" else open(f, errors="replace")
        for line in fh:
            i = line.find(MARK)  # START lines are "HQ_SPEC_TIMING_START", no space: skipped
            if i < 0:
                continue
            try:
                d = json.loads(line[i + len(MARK):])
            except ValueError:
                continue
            if d.get("v") == 1 and "keys" in d:
                runs.setdefault(d["run"], []).append(d)
    return runs


def pick(snaps, t0, t1):
    """(baseline or None, final) for the window: last snapshot at/before t0, last at/before t1."""
    def last_le(t):
        c = [s for s in snaps if s["ts"] <= t]
        return c[-1] if c else None
    end = last_le(t1) if t1 is not None else snaps[-1]
    base = last_le(t0) if t0 is not None else None
    return base, end


def diff_snap(a, b):
    """b - a as {keys, hist, steps, dropped, abandoned}; a may be None (zero)."""
    keys = {}
    for k, v in b["keys"].items():
        z = (a or {}).get("keys", {}).get(k, {})
        d = {f: v.get(f, 0) - z.get(f, 0) for f in ("n", "gap_n", "tok", "drafts", "emit")}
        d["ms"] = {p: x - z.get("ms", {}).get(p, 0.0) for p, x in v["ms"].items()}
        d["sq"] = {p: x - z.get("sq", {}).get(p, 0.0) for p, x in v["sq"].items()}
        if d["n"] > 0:
            keys[k] = d
    hist = {}
    for kind, ph in b["hist"].items():
        for p, h in ph.items():
            z = (a or {}).get("hist", {}).get(kind, {}).get(p, [0] * len(h))
            hist.setdefault(kind, {})[p] = [x - y for x, y in zip(h, z)]
    g = lambda f: b.get(f, 0) - (a or {}).get(f, 0)
    return {"keys": keys, "hist": hist, "steps": g("steps"), "dropped": g("dropped"),
            "abandoned": g("abandoned"), "tokens": b.get("tokens", False)}


def pct(h, q, edges):
    """Upper edge (ms) of the bucket holding the q-quantile; inf for the overflow bucket."""
    n = sum(h)
    if n <= 0:
        return None
    c = 0
    for i, x in enumerate(h):
        c += x
        if c >= q * n:
            return edges[i] if i < len(edges) else math.inf
    return math.inf


def stats(keys, phase, with_gap=False):
    n = sum((v["gap_n"] if phase == "gap" else v["n"]) for v in keys)
    s = sum(v["ms"].get(phase, 0.0) for v in keys)
    q = sum(v["sq"].get(phase, 0.0) for v in keys)
    if n <= 0 or s == 0.0:
        return None
    m = s / n
    return {"sum_ms": s, "mean_ms": m, "sd_ms": math.sqrt(max(0.0, q / n - m * m)), "n": n}


def summarise(d, edges):
    """Per kind (spec/plain) and per key: mean ms of every phase, tokens/step, implied tok/s."""
    out = {"steps": d["steps"], "dropped": d["dropped"], "abandoned": d["abandoned"], "kinds": {}, "keys": {}}

    def block(keys, hist=None, names=()):
        n = sum(v["n"] for v in keys)
        b = {"n": n, "phases": {}}
        for p in PHASES:
            st = stats(keys, p)
            if st:
                if hist and p in hist:
                    st["p50_ms"] = pct(hist[p], 0.5, edges)
                    st["p95_ms"] = pct(hist[p], 0.95, edges)
                b["phases"][p] = st
        tot = b["phases"].get("total", {}).get("sum_ms")
        if tot:
            for p in STEP_PARTS:
                if p in b["phases"]:
                    b["phases"][p]["share_of_total"] = b["phases"][p]["sum_ms"] / tot
        if d["tokens"] and n:
            emit = sum(v["emit"] for v in keys)
            b["emitted_tokens"] = emit
            b["tok_per_step"] = emit / n
            b["drafts_per_step"] = sum(v["drafts"] for v in keys) / n
            rs = [int(k.split("/")[1][1:]) for k in names]
            if names and max(rs) < 17 and b["drafts_per_step"]:  # r17 is a cap: reqs unknown
                req_steps = sum(r * v["n"] for r, v in zip(rs, keys))
                # emitted = accepted drafts + one target token per request-step
                b["accept_rate"] = (emit - req_steps) / sum(v["drafts"] for v in keys)
            wall = (tot or 0.0) + sum(v["ms"].get("gap", 0.0) for v in keys)
            if wall:
                # decode rate implied by the event timeline: compare with sweep.py's tok/s
                b["implied_tok_s"] = emit / (wall / 1000.0)
        return b

    for kind in sorted({k.split("/")[0] for k in d["keys"]}):
        ks = [v for k, v in d["keys"].items() if k.split("/")[0] == kind]
        out["kinds"][kind] = block(ks, d["hist"].get(kind), [k for k in d["keys"] if k.split("/")[0] == kind])
    for k, v in sorted(d["keys"].items()):
        out["keys"][k] = block([v], None, [k])
    return out


def fmt(x, w=7, p=2):
    return f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-"


def render(run, pid, d, s, label):
    L = [f"== {label or 'run'} {run} pid {pid}: {s['steps']} timed steps "
         f"(dropped {s['dropped']}, abandoned {s['abandoned']})"]
    cols = ("gap", "verify", "logits", "accept", "other", "draft", "total")
    L.append(f"{'':16}{'n':>7}" + "".join(f"{c:>8}" for c in cols) + f"{'tok/step':>9}{'acc/draft':>10}{'impl tok/s':>11}")
    rows = [(f"{k} (all)", v) for k, v in s["kinds"].items()] + list(s["keys"].items())
    for name, b in rows:
        ph = b["phases"]
        L.append(f"{name:16}{b['n']:>7}" + "".join(fmt(ph.get(c, {}).get('mean_ms'), 8) for c in cols)
                 + fmt(b.get("tok_per_step"), 9) + fmt(b.get("accept_rate"), 10, 3) + fmt(b.get("implied_tok_s"), 11, 1))
    L.append("   mean ms per step; key = kind/r<requests>/<G full graph | E not>; gap = idle between steps;")
    L.append("   draft = whole speculator.propose; step = total + gap")
    for kind, b in s["kinds"].items():
        ph = b["phases"]
        parts = " ".join(f"{p} {100 * ph[p]['share_of_total']:.0f}%" for p in STEP_PARTS
                         if p in ph and "share_of_total" in ph[p])
        q = {p: (ph[p].get("p50_ms"), ph[p].get("p95_ms")) for p in ("verify", "accept", "draft") if p in ph}
        L.append(f"   {kind}: {parts}; p50/p95 <= " +
                 ", ".join(f"{p} {a}/{b_}" for p, (a, b_) in q.items()))
    return "\n".join(L)


def windows(snaps):
    prev = None
    for s in snaps:
        yield prev, s
        prev = s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="log files (default stdin)")
    ap.add_argument("--label", default="")
    ap.add_argument("--t0", type=float, help="baseline: last line at/before this unix ts")
    ap.add_argument("--t1", type=float, help="end: last line at/before this unix ts")
    ap.add_argument("--windows", action="store_true", help="one summary per emitted window")
    ap.add_argument("--out", help="write the summary JSON here")
    a = ap.parse_args()
    runs = read_lines(a.files)
    if not runs:
        print("no HQ_SPEC_TIMING lines found (variant image? HQ_SPEC_TIMING=1 in the container env?)",
              file=sys.stderr)
        return 2
    result = {}
    for run, snaps in runs.items():
        edges = DEFAULT_EDGES
        pairs = list(windows(snaps)) if a.windows else [pick(snaps, a.t0, a.t1)]
        for base, end in pairs:
            if end is None or (base is not None and base is end):
                continue
            d = diff_snap(base, end)
            if d["steps"] <= 0:
                continue
            s = summarise(d, edges)
            tag = run if not a.windows else f"{run}@{end['ts']}"
            result[tag] = {"pid": end["pid"], "label": a.label, "t0": base["ts"] if base else None,
                           "t1": end["ts"], **s}
            print(render(tag, end["pid"], d, s, a.label), end="\n\n")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
