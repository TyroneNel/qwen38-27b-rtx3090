"""Batch 4 tables from the raw files (host Python, stdlib only). Prints markdown; nothing here is measured, it only
reads what arm.sh wrote.

  python bench/results/2026-10-15-batch4-tools/analyze.py <dir>:<label> ...

Per boot:
  - pool: the engine's "GPU KV cache size: N tokens" line (raw-boot-<label>.log)
  - per probe wave (raw-running-<label>-w<w>.log, HELD lines only = taken while every request of the wave was in the
    server): the Running/Waiting pairs, the modal Running count, and the per-request share of the pool,
    kv% / Running, over the HELD lines at the modal count; prompt tokens from raw-probe-<label>-w<w>.json
  - ladder rows (raw-ladder-<label>.json): decode agg, preempt, kv%, run, ms/pass, tok/step. A row is a TAIL row when
    it preempted or ran fewer streams than it sent (run < n).
  - C1 T=0: the "cohort C1 real prompts T=0" decode figure of the second run_benchmarks.sh (raw-rb-<label>-2.log)
"""
import collections, glob, json, os, re, statistics, sys

RUN = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")


def boot(d, l):
    pool = None
    p = os.path.join(d, f"raw-boot-{l}.log")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
            if m:
                pool = int(m.group(1).replace(",", ""))
    return pool


def waves(d, l):
    out = []
    for p in sorted(glob.glob(os.path.join(d, f"raw-running-{l}-w*.log")), key=lambda s: int(re.search(r"-w(\d+)\.log", s).group(1))):
        w = int(re.search(r"-w(\d+)\.log", p).group(1))
        head = open(p, encoding="utf-8", errors="replace").readline()
        n = re.search(r"n=(\d+)", head); ctx = re.search(r"ctx~(\d+)", head)
        held, hts = [], []
        for x in open(p, encoding="utf-8", errors="replace"):
            if x.startswith("HELD") and RUN.search(x):
                held.append(tuple(map(float, RUN.search(x).groups())))
                hts.append(re.search(r"\d{4}-\d\d-\d\dT[\d:.]+Z", x).group(0)[:23])
        pj = os.path.join(d, f"raw-probe-{l}-w{w}.json")
        ptoks = []
        if os.path.exists(pj):
            j = json.loads(open(pj).read() or "{}")
            ptoks = [r.get("prompt_tokens") for r in j.get("requests", []) if r.get("prompt_tokens")]
            firsts = sorted(r["first"][:23] for r in j.get("requests", []) if r.get("first"))
            errs = [r["error"] for r in j.get("requests", []) if "error" in r]
        else:
            errs, firsts = ["no probe json"], []
        rec = dict(w=w, n=int(n.group(1)) if n else None, ctx=int(ctx.group(1)) if ctx else None, held=held,
                   prompt=statistics.median(ptoks) if ptoks else None, errors=errs)
        if held:
            mode = collections.Counter(int(h[0]) for h in held).most_common(1)[0][0]
            at = [h for h in held if int(h[0]) == mode]
            rec.update(mode=mode, waiting=[int(h[1]) for h in at],
                       share=statistics.mean(h[2] / h[0] for h in at if h[0]), kv=[h[2] for h in at])
            # decode phase: HELD lines at the modal count taken after the mode-th request had its first token, i.e.
            # once every running request had finished its prefill
            if len(firsts) >= mode:
                dec = [h for h, t in zip(held, hts) if int(h[0]) == mode and t >= firsts[mode - 1]]
                if dec:
                    rec.update(dec_n=len(dec), dec_share=max(h[2] / h[0] for h in dec))
        out.append(rec)
    return out


def blocks(d, l):
    for p in sorted(glob.glob(os.path.join(d, f"raw-metrics-{l}-w*.log"))):
        m = re.search(r'num_gpu_blocks="(\d+)"', open(p, encoding="utf-8", errors="replace").read())
        if m:
            return int(m.group(1))
    return None


def ladder(d, l):
    p = os.path.join(d, f"raw-ladder-{l}.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p))["rows"]


def c1(d, l):
    p = os.path.join(d, f"raw-rb-{l}-2.log")
    if not os.path.exists(p):
        return None
    for line in open(p, encoding="utf-8", errors="replace"):
        if line.startswith("ROW cohort C1 real prompts T=0 "):
            return float(re.search(r"decode\(C/meanTPOT\)=([\d.]+)", line).group(1))
    return None


for spec in sys.argv[1:]:
    d, l = spec.rsplit(":", 1)
    pool = boot(d, l); nb = blocks(d, l)
    print(f"\n### {l} ({d})  pool={pool} tokens  num_gpu_blocks={nb}  C1 T=0 (run 2)={c1(d, l)}\n")
    print("| wave | n | ctx | prompt tok (median) | HELD Running/Waiting @ kv% | modal Running | share/request (mean, all HELD at mode) | decode-phase share (max) | blocks/request (decode) | seats = 100/decode share | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in waves(d, l):
        pairs = " ".join(f"{int(a)}/{int(b)}@{c:.1f}" for a, b, c in r["held"]) or "none HELD"
        if "share" in r:
            ds = r.get("dec_share")
            print(f"| w{r['w']} | {r['n']} | {r['ctx']} | {r['prompt']} | {pairs} | {r['mode']} | {r['share']:.2f}% | "
                  f"{f'{ds:.2f}% ({r['dec_n']} lines)' if ds else '-'} | "
                  f"{f'{ds / 100 * nb:.1f}' if ds and nb else '-'} | {f'{100 / ds:.1f}' if ds else '-'} | {len(r['errors'])} |")
        else:
            print(f"| w{r['w']} | {r['n']} | {r['ctx']} | {r['prompt']} | {pairs} | - | - | - | - | - | {len(r['errors'])} |")
    rows = ladder(d, l)
    if rows:
        print("\n| N | decode agg | preempt | kv% | run | ms/pass | tok/step | row |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            tail = r["preempt"] > 0 or r["running"] < r["n"]
            print(f"| {r['n']} | {r['decode_agg']:.1f} | {r['preempt']:.0f} | {r['kv']:.1f} | {r['running']:.0f} | "
                  f"{r['ms_pass']:.1f} | {r['tok_step']:.2f} | {'TAIL' if tail else 'steady'} |")
