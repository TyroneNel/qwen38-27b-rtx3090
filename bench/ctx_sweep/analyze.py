#!/usr/bin/env python3
"""The context-profile sweep's decisions, read from the run directory D. Standard library only. sweep.sh runs it
inside the image on a copy of the files it needs, so Windows needs no Python.
  analyze.py arms                    one line per arm: name, profile and env lines, tab-separated
  analyze.py dryrun <D>              every arm's dry run against its expectations, and each profile's arms told apart
  analyze.py plan <D> <arm> <pass>   the arm's next step: BOOT <label> <profile> and its env lines, DONE, BROKEN or STOP
  analyze.py cells <D> <label>       the boot's cells, name and command, tab-separated
  analyze.py boot <D> <label>        the boot's RECORD line, then its verdict line
  analyze.py report <D>              summary.md on stdout
"""
import json, math, os, re, sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = {"GPU_UTIL": "--gpu-memory-utilization", "KV_MEM": "--kv-cache-memory", "MAX_LEN": "--max-model-len"}
MAX_DEDICATED, MAX_RISE, MAX_USED_BEFORE, MIN_TRACE = 23400, 24.0, 300, 20000
MEM_STEPS, GPU_UTIL_STEP, KV_MEM_STEP = 2, 0.02, 256 << 20
PPL_TOL, GSM_MIN, CTX_CAP, CLOCK_TOL = 1.03, 0.90, 262144, 0.15
REF_ARM = "f-mtp4"
GPU_NAME = "NVIDIA GeForce RTX 3090, 8.6, 250.00 W"
LABEL = re.compile(r"(.+)-p(\d+)-(\d+)")
TASK = re.compile(r"^LABD \S+ (\w+)\s+prompt=\s*(\d+) out=\s*(\d+) steps=\s*(\d+) tok/step=\s*([\d.]+) "
                  r"decode=\s*([\d.]+) e2e=\s*([\d.]+) ttft=\s*(-?[\d.]+)s", re.M)
TOTAL = re.compile(r"^LABD \S+ TOTAL out=(\d+) steps=(\d+) tok/step=([\d.]+) decode=([\d.]+) tok/s", re.M)
DEATH = re.compile(r"EngineDeadError|Engine core proc \S+ died|EngineCore.*(died|encountered an issue|fatal)")


def read(*p):
    p = os.path.join(*p)
    return open(p, errors="replace").read().replace("\r", "") if os.path.exists(p) else None


def unquote(v):
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"" else v


def envdict(lines):
    return {k: unquote(v) for k, _, v in (x.partition("=") for x in lines if "=" in x and not x.startswith("#"))}


def base():
    return envdict(read(HERE, "base.env").splitlines())


def arms():
    out = {}
    for line in open(os.path.join(HERE, "arms.txt")):
        if line.startswith("#") or not line.strip():
            continue
        prof, name, env, exp = (x.strip() for x in line.split("|"))
        out[name] = dict(name=name, profile=prof, env=[e.strip() for e in env.split(";") if e.strip()],
                         expect=[e.strip() for e in exp.split(";") if e.strip()])
    return out


def launch(text):
    argv, env = [], []
    for line in (text or "").splitlines():
        if line.startswith("ARG "):
            argv.append(line[4:])
        elif line.startswith("ENV "):
            env.append(line[4:])
    return argv, env


def flag(argv, f):
    v = None
    for i, a in enumerate(argv):
        if a == f and i + 1 < len(argv):
            v = argv[i + 1]
        elif a.startswith(f + "="):
            v = a[len(f) + 1:]
    return v


def expectations(arm, env):
    return arm["expect"] + [f"{FIT[k]} {v}" for k, v in env.items() if k in FIT]


def check(exp, argv, env, model):
    bad = [] if argv[:2] == ["serve", model] else [f"argv does not start with serve {model}"]
    for e in exp:
        m = re.fullmatch(r"(--[\w-]+)[ =](\S+)", e)
        if m:
            got = flag(argv, m.group(1))
            if got != m.group(2):
                bad.append(f"{m.group(1)} is {got}, want {m.group(2)}")
        elif re.fullmatch(r"[A-Z][A-Z0-9_]*=\S*", e):
            if e not in env:
                bad.append(f"env lacks {e}")
        elif not any(e in a for a in argv):
            bad.append(f"argv lacks {e}")
    return bad


def dryrun(D):
    A, model, fails = arms(), base()["MODEL"], 0
    runs = {}
    for name, a in A.items():
        argv, env = launch(read(D, "dryrun", name + ".out"))
        runs[name] = (argv, env)
        bad = check(expectations(a, envdict(a["env"])), argv, env, model)
        fails += bool(bad)
        print(f"DRY {name} " + ("ok" if not bad else "FAIL: " + "; ".join(bad)))
    for x in A:
        for y in A:
            if x < y and A[x]["profile"] == A[y]["profile"]:
                tell = check(expectations(A[x], envdict(A[x]["env"])), *runs[y], model) or \
                       check(expectations(A[y], envdict(A[y]["env"])), *runs[x], model)
                if not tell:
                    fails += 1
                    print(f"DRY {x} and {y} FAIL: neither arm's expectations reject the other's launch")
    print("DRYRUN_OK" if not fails else f"DRYRUN_FAIL {fails}")
    return 0 if not fails else 1


def category(v):
    t = (v or "").split()
    if not t:
        return "other"
    return {"BOOT_CLEAN": "clean", "BOOT_SETUP": "setup", "RETRY_MAX_LEN": "maxlen"}.get(t[0]) or \
        ("mem" if t[:2] == ["BOOT_UNCLEAN", "mem"] else "other")


def boots(D, arm):
    out = []
    for n in os.listdir(D):
        m = LABEL.fullmatch(n)
        if m and m.group(1) == arm and os.path.isdir(os.path.join(D, n)):
            out.append(dict(label=n, p=int(m.group(2)), n=int(m.group(3)), verdict=(read(D, n, "verdict") or "").strip()))
    return sorted(out, key=lambda h: (h["p"], h["n"]))


def plan(D, arm, p):
    a = arms()[arm]
    env = envdict(a["env"])
    dry, _ = launch(read(D, "dryrun", arm + ".out"))
    pin, util = flag(dry, "--kv-cache-memory"), env.get("GPU_UTIL") or flag(dry, "--gpu-memory-utilization")
    hist = boots(D, arm)
    fit, mem, maxlen, others, clean = {}, 0, False, {}, set()
    for h in hist:
        c = category(h["verdict"])
        if c == "setup":
            return [f"STOP {h['label']}: {h['verdict']}"]
        if c == "clean":
            clean.add(h["p"])
        elif c == "mem":
            mem += 1
            if mem <= MEM_STEPS:
                fit.update({"KV_MEM": str(int(pin) - mem * KV_MEM_STEP)} if pin else
                           {"GPU_UTIL": f"{float(util) - GPU_UTIL_STEP * mem:.2f}"})
        elif c == "maxlen" and not maxlen:
            maxlen = True
            fit["MAX_LEN"] = h["verdict"].split()[1]
        else:
            others[h["p"]] = others.get(h["p"], 0) + 1
    if p in clean:
        return [f"DONE {arm} pass {p}"]
    if any(q not in clean for q in range(1, p)):
        return [f"BROKEN {arm} pass {p}: no clean boot in an earlier pass"]
    if mem > MEM_STEPS:
        return [f"BROKEN {arm} pass {p}: still unclean on memory after {MEM_STEPS} steps down"]
    if others.get(p, 0) >= 2:
        return [f"BROKEN {arm} pass {p}: two boots unclean or failed for reasons other than memory"]
    n = 1 + max([int(m.group(1)) for x in os.listdir(D) for m in [re.fullmatch(re.escape(arm) + rf"-p{p}-(\d+)(-aborted)?", x)] if m], default=0)
    lines = [f"{k}={fit[k]}" if k in fit else line for line in a["env"] for k in [line.partition("=")[0]]]
    return [f"BOOT {arm}-p{p}-{n} {a['profile']}"] + lines + [f"{k}={v}" for k, v in fit.items() if k not in env]


def cells(D, L):
    arm, p = LABEL.fullmatch(L).group(1), int(LABEL.fullmatch(L).group(2))
    ml = int(flag(launch(read(D, L, "live.txt"))[0], "--max-model-len") or 0)
    if not ml:
        return []
    lb = lambda c, ctx: (c, f"venv/bin/python bench/labd_bench.py sw-{L}-{c} --ctx {ctx} "
                            "--corpus ~/bench/labd_corpus_long.txt")
    smoke = ("smoke", "venv/bin/python bench/api_smoke.py")
    needle = ("needle", f"venv/bin/python bench/needle_test.py {ml * 9 // 10} 0.9")
    qual = ("qual", f"QUALITY_DATA=/cache/quality-data venv/bin/python bench/quality_battery.py sw-{L} "
                    + ("--gsm-n 200" if p == 1 else "--ppl-only"))
    prof = arms()[arm]["profile"]
    # The deepest labd cell runs first: a shallower document is a prefix of it, so only the first copy task is cold.
    if prof == "fast":
        return [smoke, lb("labd48", 42000), lb("labd16", 14000),
                ("ladder", "venv/bin/python bench/conc_ladder.py --n 1,2,4 --out 256 --reps 2 --ctx-tokens 4096 --salt ctxsweep"),
                qual]
    if prof == "long":
        return [smoke, lb("labd113", 100000), lb("labd66", 58000), needle, qual]
    return [smoke, needle] + ([lb("labd113", 100000)] if ml >= 120000 else []) + [qual]


def exit0(t):
    return t is not None and re.search(r"^# end \S+ exit=0$", t, re.M) is not None


def num(s):
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


def clock(t):
    c = []
    for line in (t or "").splitlines():
        if line.startswith("CLOCK "):
            try:
                c.append(tuple(float(x) for x in line.split()[1:4]))
            except ValueError:
                pass
    if len(c) < 2 or len(c[0]) < 3 or len(c[-1]) < 3 or c[-1][1] <= c[0][1]:
        return None
    (a1, v1, c1), (a2, v2, c2) = c[0], c[-1]
    return ((a2 + c2) - (a1 + c1)) / 2 / (v2 - v1)


def gpu(t):
    cols = {"sm": 2, "power": 4, "temp": 5, "util": 6}
    vals = {k: [] for k in cols}
    for line in (t or "").splitlines():
        if line.startswith("GPU, "):
            f = [x.strip() for x in line[5:].split(",")]
            for k, i in cols.items():
                v = num(f[i]) if len(f) > i else None
                if v is not None:
                    vals[k].append(v)
    out = {k: mean(v) if v else None for k, v in vals.items()}
    out["tmax"], out["n"] = max(vals["temp"], default=None), len(vals["sm"])
    return out


def parse_cell(name, t, r):
    """One cell's numbers on the host's clock: VM-clock rates divided by r, durations multiplied by r."""
    t = t or ""
    if name == "smoke":
        m = re.search(r"^SUMMARY (\d+) / (\d+) passed", t, re.M)
        return dict(ok=bool(m), passed=int(m.group(1)) if m else None, total=int(m.group(2)) if m else None)
    if name.startswith("labd"):
        m, tasks = TOTAL.search(t), {g[0]: g for g in TASK.findall(t)}
        if not (exit0(t) and m and len(tasks) == 6):
            return dict(ok=False)
        cp = tasks["copy"]
        ttft = float(cp[7]) * r
        return dict(ok=True, decode=float(m.group(4)) / r, tps=float(m.group(3)), prompt=int(cp[1]), ttft=ttft,
                    prefill=int(cp[1]) / ttft if ttft > 0 else None)
    if name == "ladder":
        rows = {m.group(1): [num(x) for x in m.group(2).split("|")] for m in re.finditer(r"^\| ([^|]+?) \|(.*)\|$", t, re.M)}
        ns = [int(x) for x in rows.get("N", []) if x is not None]
        per = dict(zip(ns, [x / r if x else None for x in rows.get("per-stream tok/s", [])]))
        return dict(ok=exit0(t) and bool(per.get(1)), per=per,
                    tok_step=dict(zip(ns, rows.get("tok/step", []))), preempt=dict(zip(ns, rows.get("preemptions", []))))
    if name == "needle":
        m = re.search(r"^elapsed ([\d.]+)s", t, re.M)
        st = "RETRIEVED" if re.search(r"^RETRIEVED$", t, re.M) else "MISSED" if re.search(r"^MISSED$", t, re.M) else "ERROR"
        c = re.search(r"^context ~(\d+) tokens", t, re.M)
        return dict(ok=True, status=st, elapsed=float(m.group(1)) * r if m else None, tokens=int(c.group(1)) if c else None)
    if name == "qual":
        m = re.search(r"^\S+ PPL \{(.*)\}", t, re.M)
        ppl = {k: float(v) for k, v in re.findall(r"'(\w+)': \(([\d.]+),", m.group(1))} if m else {}
        g = re.search(r"^\S+ GSM8K n=(\d+) acc=([\d.]+)", t, re.M)
        return dict(ok=exit0(t) and bool(ppl), ppl=ppl, gsm=float(g.group(2)) if g else None)
    return dict(ok=False)


def memtrace(B, L, luid):
    kvt, frt = ((read(B, f".{x}-{L}") or "").strip() for x in ("kvline", "firstreq"))
    d, s = {}, {}
    for line in (read(B, f"mem-trace-{L}.csv") or "").splitlines()[1:]:
        f = line.split(",")
        if len(f) == 4 and f[1] == luid:
            try:
                (d if f[2] == "dedicated usage" else s)[f[0]] = float(f[3]) / 1048576
            except ValueError:
                pass
    shared = [t for t in s if s[t] > 1]
    fctx = min((s[t] for t in shared if kvt and t < kvt and d.get(t, 0) > 1000), default=None)
    fkv = min((s[t] for t in shared if kvt and frt and kvt <= t < frt), default=None)
    return dict(maxd=max(d.values(), default=None), fctx=fctx, fkv=fkv,
                rise=fkv - fctx if fctx is not None and fkv is not None else None)


def boot(D, L):
    B, arm = os.path.join(D, L), LABEL.fullmatch(L).group(1)
    a, bs, run = arms()[arm], base(), envdict((read(D, "run.env") or "").splitlines())
    env = {k: v for k, v in envdict((read(B, "env") or "").splitlines()).items() if k not in bs}
    r = dict(label=L, arm=arm, profile=a["profile"], p=int(LABEL.fullmatch(L).group(2)), env=env,
             setup=[], mem=[], other=[], flags=[])
    status = (read(B, "boot.status") or "FAILED no boot.status").strip()
    log = read(B, "boot.log") or ""
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    r["pool"] = int(m.group(1).replace(",", "")) if m else None
    img = (read(B, "image.txt") or "").split()
    if not img or img[0] != run.get("IMG_ID"):
        r["setup"].append(f"container image {img[:1]} is not run.env's IMG_ID {run.get('IMG_ID')}")
    smi = re.findall(r"^(\d+) MiB, (\d+) MiB, (\d+) MiB, ([\d.]+) W", read(B, "smi.log") or "", re.M)
    r["used_before"] = int(smi[0][1]) if smi else None
    r["free_after"] = int(smi[1][2]) if len(smi) > 1 else None
    if r["used_before"] is None or r["used_before"] > MAX_USED_BEFORE:
        r["setup"].append(f"the 3090 held {r['used_before']} MiB before boot (limit {MAX_USED_BEFORE})")
    if status != "OK":
        est = re.findall(r"estimated maximum model length is (\d+)", log)
        if est and not r["setup"]:
            n = int(est[-1]) // 1024 * 1024
            return r, f"RETRY_MAX_LEN {n} {L}: the KV pool holds about {est[-1]} tokens"
        if re.search(r"CUDA out of memory|OutOfMemoryError", log):
            r["mem"].append("boot ran out of GPU memory")
        else:
            r["other"].append(f"boot {status}")
        return r, verdict(r)
    argv, envl = launch(read(B, "live.txt"))
    r["argv"], r["maxlen"] = argv, int(flag(argv, "--max-model-len") or 0)
    r["setup"] += ["live " + x for x in check(expectations(a, env), argv, envl, bs["MODEL"])]
    g = read(B, "gpu.txt") or ""
    if f"GPU 0: NVIDIA GeForce RTX 3090 (UUID: {bs['NVIDIA_VISIBLE_DEVICES']})" not in g or GPU_NAME not in g:
        r["setup"].append(f"the container's GPU 0 is not the 3090 at {GPU_NAME}")
    r.update(memtrace(B, L, run.get("LUID")))
    if r["maxd"] is None or r["maxd"] <= MIN_TRACE:
        r["setup"].append(f"no 3090 counter sample above {MIN_TRACE} MiB (wrong LUID?)")
    elif r["maxd"] > MAX_DEDICATED:
        r["mem"].append(f"peak Dedicated {r['maxd']:.0f} MiB above {MAX_DEDICATED}")
    if r["rise"] is None:
        r["other"].append("no spill-guard floors")
    elif r["rise"] > MAX_RISE:
        r["mem"].append(f"boot rise {r['rise']:.1f} MiB above {MAX_RISE:.0f}")
    spill = (read(B, f"SPILL-{L}") or "").strip()
    if spill:
        r["mem"].append("SPILL " + spill)
    rs = dict(re.findall(r"^(after boot|before down): (.*)$", read(B, "restarts.txt") or "", re.M))
    if not (len(rs) == 2 and rs["after boot"] == rs["before down"] and rs["after boot"].startswith("0 ")):
        r["other"].append(f"restarted ({rs})")
    if DEATH.search(read(B, "serverlog.log") or "EngineDeadError (no server log)"):
        r["other"].append("engine death in the server log, or no server log")
    r["cells"], r["gpu"], r["r"] = {}, {}, {}
    for name, _ in cells(D, L):
        t = read(B, f"cell-{name}.log")
        k = clock(t)
        r["r"][name] = k
        if k is None or abs(k - 1) > CLOCK_TOL:
            r["flags"].append(f"{name}: host/VM clock ratio {k}, numbers on the VM clock")
            k = 1.0
        r["cells"][name], r["gpu"][name] = parse_cell(name, t, k), gpu(t)
        if not r["cells"][name]["ok"]:
            r["other"].append(f"{name} did not finish")
    if not cells(D, L):
        r["other"].append("no cells: no --max-model-len in the live argv")
    elif not sum(x["n"] for x in r["gpu"].values()):
        r["setup"].append("no in-container GPU telemetry in any cell")
    return r, verdict(r)


def verdict(r):
    L = r["label"]
    if r["setup"]:
        return f"BOOT_SETUP {L}: " + "; ".join(r["setup"])
    for k in ("mem", "other"):
        if r[k]:
            return f"BOOT_UNCLEAN {k} {L}: " + "; ".join(r[k])
    return f"BOOT_CLEAN {L}"


# ---- report ----

def gmean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def avg(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def metrics(b):
    c = b["cells"]
    g = lambda cell, k: (c.get(cell) or {}).get(k)
    m = dict(d16=g("labd16", "decode"), d48=g("labd48", "decode"), d66=g("labd66", "decode"),
             d113=g("labd113", "decode"), pf48=g("labd48", "prefill"), pf113=g("labd113", "prefill"),
             ttft48=g("labd48", "ttft"), ttft113=g("labd113", "ttft"), tps48=g("labd48", "tps"),
             tps113=g("labd113", "tps"), pool=b["pool"], maxlen=b["maxlen"],
             ctx=min(b["pool"], CTX_CAP) if b["pool"] else None, needle=g("needle", "status"),
             needle_s=g("needle", "elapsed"), gsm=g("qual", "gsm"), ppl=g("qual", "ppl") or {},
             smoke=(g("smoke", "passed"), g("smoke", "total")), maxd=b["maxd"], rise=b["rise"],
             free_after=b["free_after"])
    per = g("ladder", "per") or {}
    for n in (1, 2, 4):
        m[f"c{n}"] = per.get(n, per.get(str(n)))
    m["speed"] = gmean([m["d66"], m["d113"]]) if m["d66"] and m["d113"] else None
    gs = [x for x in b["gpu"].values() if x["n"]]
    for k in ("sm", "power", "util"):
        m[k] = avg([x[k] for x in gs])
    m["r"] = avg([x for x in b["r"].values() if x])
    return m


def records(D):
    out = {}
    for n in sorted(os.listdir(D)):
        if LABEL.fullmatch(n):
            m = re.search(r"^RECORD (\{.*\})$", read(D, n, "analysis.txt") or "", re.M)
            if m:
                rec = json.loads(m.group(1))
                rec["verdict"] = (read(D, n, "verdict") or "").strip()
                out.setdefault(rec["arm"], []).append(rec)
    for v in out.values():
        v.sort(key=lambda b: (b["p"], int(LABEL.fullmatch(b["label"]).group(3))))
    return out


SCORE_KEYS = {"fast": ["d16", "d48", "c1", "pf48"], "long": ["speed", "ctx"]}


def score(prof, m, best):
    if prof == "huge":
        return (m["maxlen"] or 0, m["d113"] or 0, m["pool"] or 0)
    ks = SCORE_KEYS[prof]
    if any(not m.get(k) for k in ks):
        return None
    return gmean([m[k] / best[k] for k in ks])


def report(D):
    A, run, recs = arms(), envdict((read(D, "run.env") or "").splitlines()), records(D)
    arm_m, used, notes = {}, {}, {}
    for name in A:
        clean = [b for b in recs.get(name, []) if b["verdict"].startswith("BOOT_CLEAN")]
        if not clean:
            continue
        fit = clean[-1]["env"]
        used[name] = [b for b in clean if b["env"] == fit]
        notes[name] = ["fit moved"] if len(used[name]) < len(clean) else []
        ms, qs = [metrics(b) for b in used[name]], [metrics(b) for b in clean]
        am = {k: avg([x[k] for x in ms]) for k in ms[0] if all(isinstance(x[k], (int, float, type(None))) for x in ms)}
        # Quality does not move with the memory fit, so its bars take every clean boot: GSM8K runs on pass 1 only.
        am["ppl"] = {k: avg([x["ppl"].get(k) for x in qs]) for k in set().union(*(x["ppl"] for x in qs))}
        am["needles"] = [x["needle"] for x in qs]
        am["smokes"] = [x["smoke"] for x in qs]
        am["gsm"] = avg([x["gsm"] for x in qs])
        am["boots"] = ms
        arm_m[name] = am
    ref = arm_m.get(REF_ARM, {}).get("ppl") or {}
    if not ref:
        corpora = set().union(*(m["ppl"] for m in arm_m.values())) if arm_m else set()
        ref = {k: min(m["ppl"][k] for m in arm_m.values() if m["ppl"].get(k)) for k in corpora}
    why = {}
    for name, m in arm_m.items():
        w = []
        if any(s[0] is None or s[0] != s[1] for s in m["smokes"]):
            w.append("smoke " + ", ".join(f"{s[0]}/{s[1]}" for s in m["smokes"]))
        for k, v in sorted(ref.items()):
            if m["ppl"].get(k) is None or m["ppl"][k] > PPL_TOL * v:
                w.append(f"PPL {k} {m['ppl'].get(k)} above {PPL_TOL} x {v:.4f}")
        if m["gsm"] is None or m["gsm"] < GSM_MIN:
            w.append(f"GSM8K {m['gsm']} below {GSM_MIN}")
        if A[name]["profile"] != "fast" and any(x != "RETRIEVED" for x in m["needles"]):
            w.append("needle " + ", ".join(map(str, m["needles"])))
        why[name] = w
    out = [f"# Context-profile sweep", "",
           f"Image `{run.get('IMG')}` ({run.get('IMG_ID')}), 3090 counters `{run.get('LUID')}`, {run.get('PASSES')} passes. "
           f"Numbers are on the host's clock (`r` is host seconds per VM second). A score is normalized to the best "
           f"eligible arm of its profile. CLEAR means the winner's worst boot beats the runner-up's best boot.", "",
           "## Recommendations", "", "| profile | arm | margin | env on top of base.env |", "|---|---|---|---|"]
    tables = []
    for prof in ("fast", "long", "huge"):
        names = [n for n in A if A[n]["profile"] == prof]
        elig = [n for n in names if n in arm_m and not why[n]]
        best = {k: max((arm_m[n][k] or 0) for n in elig) for k in SCORE_KEYS.get(prof, [])} if elig else {}
        sc = {n: score(prof, arm_m[n], best) for n in elig}
        ranked = sorted([n for n in elig if sc[n] is not None], key=lambda n: sc[n], reverse=True)
        if ranked:
            w = ranked[0]
            margin = "only eligible arm"
            if len(ranked) > 1:
                ru = ranked[1]
                if prof == "huge":
                    clear = arm_m[w]["maxlen"] > arm_m[ru]["maxlen"] or \
                        min(b["d113"] or 0 for b in arm_m[w]["boots"]) > max(b["d113"] or 0 for b in arm_m[ru]["boots"])
                else:
                    clear = min(score(prof, b, best) or 0 for b in arm_m[w]["boots"]) > \
                        max(score(prof, b, best) or 0 for b in arm_m[ru]["boots"])
                margin = f"CLEAR over {ru}" if clear else f"within boot spread of {ru}"
            out.append(f"| {prof} | {w} | {margin} | `{' '.join(f'{k}={v}' for k, v in used[w][-1]['env'].items())}` |")
        else:
            out.append(f"| {prof} | none | no eligible arm | |")
        f1 = lambda v, s="{:.1f}": "n/a" if v is None else s.format(v)
        cols = {"fast": [("d16", "decode ~16k"), ("d48", "decode ~48k"), ("c1", "C1"), ("c2", "C2/stream"),
                         ("c4", "C4/stream"), ("pf48", "prefill ~48k tok/s"), ("ttft48", "TTFT ~48k s"), ("tps48", "tok/step")],
                "long": [("d66", "decode ~66k"), ("d113", "decode ~113k"), ("pool", "pool"), ("pf113", "prefill ~113k tok/s"),
                         ("ttft113", "TTFT ~113k s"), ("tps113", "tok/step"), ("needle_s", "needle s")],
                "huge": [("maxlen", "MAX_LEN"), ("pool", "pool"), ("d113", "decode ~113k"), ("needle_s", "needle s"),
                         ("ttft113", "TTFT ~113k s"), ("tps113", "tok/step")]}[prof]
        t = [f"## {prof}", "", "| arm | boots | score | eligible | " + " | ".join(c[1] for c in cols) + " | PPL | GSM8K |",
             "|---|---|---|---|" + "---|" * (len(cols) + 2)]
        for n in names:
            if n not in arm_m:
                last = (recs.get(n) or [{}])[-1].get("verdict", "never booted")
                t.append(f"| {n} | 0 clean | | no: {last} |" + " |" * (len(cols) + 2))
                continue
            m = arm_m[n]
            s = sc.get(n)
            s = "n/a" if s is None else (f"{s[0]}, {f1(s[1])}, {s[2]}" if prof == "huge" else f"{s:.3f}")
            cells_ = [f1(m[c], "{:.0f}" if c in ("pool", "maxlen") else "{:.1f}") for c, _ in cols]
            t.append(f"| {n} | {len(used[n])}{' (' + ', '.join(notes[n]) + ')' if notes[n] else ''} | {s} | "
                     f"{'yes' if not why[n] else 'no: ' + '; '.join(why[n])} | " + " | ".join(cells_) +
                     f" | {' '.join(f'{k} {v:.3f}' for k, v in sorted(m['ppl'].items()) if v)} | {f1(m['gsm'], '{:.3f}')} |")
        tables += t + [""]
    out += [""] + tables
    out += ["## Fit, memory and clocks", "",
            "| arm | env on top of base.env | pool | free after boot MiB | peak Dedicated MiB | rise MiB | SM MHz | W | util % | r | flags |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    f1 = lambda v, s="{:.1f}": "n/a" if v is None else s.format(v)
    for n, m in arm_m.items():
        fl = sorted({f for b in used[n] for f in b["flags"]})
        out.append(f"| {n} | `{' '.join(f'{k}={v}' for k, v in used[n][-1]['env'].items())}` | {f1(m['pool'], '{:.0f}')} | "
                   f"{f1(m['free_after'], '{:.0f}')} | {f1(m['maxd'], '{:.0f}')} | {f1(m['rise'])} | {f1(m['sm'], '{:.0f}')} | "
                   f"{f1(m['power'], '{:.0f}')} | {f1(m['util'], '{:.0f}')} | {f1(m['r'], '{:.3f}')} | {'; '.join(fl)} |")
    out += ["", "## Every boot", "", "| label | verdict |", "|---|---|"]
    for n in A:
        for b in recs.get(n, []):
            out.append(f"| {b['label']} | {b['verdict']} |")
    print("\n".join(out))
    return 0


def main(a):
    if a[:1] == ["arms"]:
        for x in arms().values():
            print("\t".join([x["name"], x["profile"]] + x["env"]))
        return 0
    if a[:1] == ["dryrun"]:
        return dryrun(a[1])
    if a[:1] == ["plan"]:
        print("\n".join(plan(a[1], a[2], int(a[3]))))
        return 0
    if a[:1] == ["cells"]:
        for c in cells(a[1], a[2]):
            print("\t".join(c))
        return 0
    if a[:1] == ["boot"]:
        r, v = boot(a[1], a[2])
        print("RECORD " + json.dumps(r))
        print(v)
        return 0
    if a[:1] == ["report"]:
        return report(a[1])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
