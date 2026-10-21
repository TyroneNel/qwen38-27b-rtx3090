"""spec-5: expected acceptance, sum_x min(p, q), against the realised accept counts, per prompt and per draft
position. It reads the telemetry file that docs/research/deepinfer/experiments/patches/spec5-accept-telemetry.patch
makes the server write, so it needs that patch in the image and SPEC5_TELEMETRY_DIR in the server's environment,
and it runs where it can read that directory (inside the serving container).

  /app/venv/bin/python bench/spec5_accept.py [tag] [--dir /tmp/spec5] [--prompts bench/prompts_real.jsonl]
                       [--limit 8] [--max-tokens 256] [--temperature 0.7] [--seed 0] [--think]
                       [--watch SECONDS] [--out f.json]

Why the server has to compute it: the client never sees the draft distribution q, nor the rejected draft token,
so no logprobs request can give sum_x min(p, q) or p(d)/q(d) for a position that was rejected.

What the file holds, per draft position i, summed over every verify step since boot:
  overlap   sum_x min(p_i, q_i): the chance position i accepts a fresh draft given the drafted prefix
            (DeepInfer's accept_probs). Its per-step mean is a conditional rate, comparable with
            accepted_i / accepted_(i-1).
  survival  prod_(j<=i) min(1, p_j(d_j) / q_j(d_j)): E[position i accepted | the drafts]. Unbiased for the
            realised count with the accept coins integrated out, so 1 + sum_i survival_i / steps is
            tokens/step with less variance than the realised 1 + sum_i accepted_i / steps.
  drafts    verified positions; drafts_0 is the number of verify steps.
  accepted  realised accepts.
Greedy requests (--temperature 0) have no coins, so survival must equal accepted exactly: this tool exits 1 if it
does not, which makes a T=0 run the patch's self-check on the GPU. At T > 0 the two agree in expectation and the
per-prompt spread shows what the variance reduction buys (`var ratio` = var(real) / var(rb) across prompts).

Window edges: the server publishes each step's totals at the NEXT verify step, so the file lags by one step. A
prompt's window therefore holds the previous request's last step and misses its own; over a run that is one
step in and one out, and the greedy identity holds regardless (both rows lag together). /metrics' spec-decode
counters are diffed over the whole run as a cross-check and printed, not gated.

--watch S measures somebody else's traffic instead: snapshot, sleep S, snapshot, report the totals.

Exit: 0 measurement; 1 greedy identity broken (survival != accepted at T=0); 2 invalid run (no telemetry file,
the file did not advance, or no verify step in the window).
"""
import glob
import json
import math
import os
import sys
import time

import harness

ARGV = sys.argv[1:]
TAG = ARGV[0] if ARGV and not ARGV[0].startswith("--") else "run"


def arg(name, default, cast=str):
    return cast(ARGV[ARGV.index(name) + 1]) if name in ARGV else default


DIR = arg("--dir", os.environ.get("SPEC5_TELEMETRY_DIR", "/tmp/spec5"))
PROMPTS = arg("--prompts", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_real.jsonl"))
LIMIT = arg("--limit", 8, int)
MAXTOK = arg("--max-tokens", 256, int)
TEMP = arg("--temperature", 0.7, float)
SEED = arg("--seed", 0, int)
THINK = "--think" in ARGV
WATCH = arg("--watch", 0.0, float)
OUT = arg("--out", "")
ROWS = ("overlap", "survival", "drafts", "accepted")
PROM = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total")


def invalid(msg):
    print(f"SPEC5 {TAG} INVALID {msg}")
    sys.exit(2)


def snapshot():
    """The newest telemetry file in DIR (a restarted server leaves the old pid's file behind)."""
    best = None
    for path in glob.glob(os.path.join(DIR, "spec5-*.json")):
        try:
            with open(path) as f:
                body = json.load(f)
        except (OSError, ValueError):
            continue  # os.replace makes a torn read unlikely; a stale tmp name never matches the glob
        if best is None or body["updated"] > best["updated"]:
            best = body
    return best


def delta(a, b):
    return {r: [y - x for x, y in zip(a[r], b[r])] for r in ROWS} | {"calls": b["calls"] - a["calls"]}


def summarise(d):
    steps = d["drafts"][0] if d["drafts"] else 0.0
    if steps <= 0:
        return None
    k = len(d["drafts"])
    real = 1 + sum(d["accepted"]) / steps
    rb = 1 + sum(d["survival"]) / steps
    per = []
    for i in range(k):
        n = d["drafts"][i]
        prev = d["accepted"][i - 1] if i else n
        per.append({"pos": i, "drafts": n,
                    "real": d["accepted"][i] / n if n else float("nan"),
                    "rb": d["survival"][i] / n if n else float("nan"),
                    "overlap": d["overlap"][i] / n if n else float("nan"),
                    "real_cond": d["accepted"][i] / prev if prev else float("nan")})
    return {"steps": steps, "calls": d["calls"], "real": real, "rb": rb, "per_pos": per,
            "identity_gap": max((abs(s - a) for s, a in zip(d["survival"], d["accepted"])), default=0.0)}


def prom():
    m = harness.metrics(*PROM)
    return [m.get(n, 0.0) for n in PROM]


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else float("nan")
    return mu, math.sqrt(var / n) if n > 1 else float("nan"), var


def request(prompt, seed):
    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
               "temperature": TEMP, "seed": seed,
               "chat_template_kwargs": {"enable_thinking": THINK}}
    return harness.post("/v1/chat/completions", payload, timeout=3600)


first = snapshot()
if first is None:
    invalid(f"no spec5-*.json in {DIR}: the patch is not in the image, SPEC5_TELEMETRY_DIR is not set in the "
            "server's environment (or names another directory), the server runs the V1 model runner, or it has "
            "no speculator")

per_prompt = []
if WATCH > 0:
    p0, s0 = prom(), snapshot()
    time.sleep(WATCH)
    p1, s1 = prom(), snapshot()
    mode = f"watch={WATCH:g}s"
else:
    prompts = [json.loads(line)["prompt"] for line in open(PROMPTS) if line.strip()][:LIMIT]
    if not prompts:
        invalid(f"no prompts in {PROMPTS}")
    # One short request first, so the previous traffic's last step is published before the first window opens.
    request("Say hi.", SEED)
    p0, s0 = prom(), snapshot()
    prev = s0
    for i, text in enumerate(prompts):
        r = request(text, SEED + i)
        cur = snapshot()
        s = summarise(delta(prev, cur))
        usage = r.get("usage") or {}
        if s:
            s.update(idx=i, completion_tokens=usage.get("completion_tokens"),
                     finish_reason=(r.get("choices") or [{}])[0].get("finish_reason"))
            per_prompt.append(s)
            print(f"SPEC5 {TAG} prompt={i} steps={s['steps']:.0f} tok/step real={s['real']:.3f} "
                  f"rb={s['rb']:.3f} tokens={usage.get('completion_tokens')}", flush=True)
        else:
            print(f"SPEC5 {TAG} prompt={i} no verify step in its window", flush=True)
        prev = cur
    p1, s1 = prom(), prev
    mode = f"prompts={len(prompts)} T={TEMP:g} think={int(THINK)} max_tokens={MAXTOK}"

if s1["calls"] <= s0["calls"]:
    invalid(f"the telemetry file did not advance (calls {s0['calls']} -> {s1['calls']}): no verify step ran, or "
            "the file belongs to a dead server")
tot = summarise(delta(s0, s1))
if tot is None:
    invalid("no verify step with a draft in the window")

print(f"{'pos':>3} {'drafts':>8} {'real':>7} {'rb':>7} {'real|prev':>9} {'overlap':>8}")
for p in tot["per_pos"]:
    print(f"{p['pos']:>3} {p['drafts']:>8.0f} {p['real']:>7.3f} {p['rb']:>7.3f} {p['real_cond']:>9.3f} "
          f"{p['overlap']:>8.3f}")
dp_drafts, dp_acc = p1[0] - p0[0], p1[1] - p0[1]
print(f"/metrics over the run: drafts={dp_drafts:.0f} accepted={dp_acc:.0f} "
      f"(file: steps={tot['steps']:.0f} accepted={sum(p['real'] * p['drafts'] for p in tot['per_pos']):.0f}; "
      f"one step differs at each edge)")

mr, ser, vr = mean_se([p["real"] for p in per_prompt])
mb, seb, vb = mean_se([p["rb"] for p in per_prompt])
ratio = vr / vb if vb > 1e-12 else float("nan")
greedy = WATCH <= 0 and TEMP == 0.0
ok = not greedy or tot["identity_gap"] < 1e-6
verdict = "" if ok else f" FAIL greedy identity gap={tot['identity_gap']:.3g}"
print(f"SPEC5 {TAG} {mode} steps={tot['steps']:.0f} tok/step real={tot['real']:.3f} rb={tot['rb']:.3f} "
      f"overlap0={tot['per_pos'][0]['overlap']:.3f} se real={ser:.3f} rb={seb:.3f} var_ratio={ratio:.2f}"
      + verdict)
if OUT:
    with open(OUT, "w") as f:
        json.dump({"tag": TAG, "mode": mode, "temperature": TEMP, "think": THINK, "seed": SEED,
                   "max_tokens": MAXTOK, "total": tot, "prompts": per_prompt,
                   "metrics_delta": {"drafts": dp_drafts, "accepted": dp_acc},
                   "mean_se": {"real": [mr, ser], "rb": [mb, seb]}, "var_ratio": ratio}, f, indent=1)
    print(f"wrote {OUT}")
sys.exit(0 if ok else 1)
