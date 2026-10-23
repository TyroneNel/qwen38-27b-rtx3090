#!/usr/bin/env python3
"""kvarn-varlen: reproduce, and take apart, the "adaptive verify length corrupts a KVarN
prefix-cache hit" result (single-user/start_qwen.sh, the VLLM_DFLASH2_LOOKUP_ADAPTIVE
comment: "self-hit 38/793 WRONG").

Same measurement as that comment and bench/residue_sweep.py: the residue_sweep prompt
(document + "Gengiv ordret de forste 60 linjer ..." + " x" * pad, so one pad is one prompt
token and --residues picks the templated length mod 128), max_tokens 300, temperature 0,
thinking off. One request arms the cache cold, the next ones are full self-hits; each
answer is reported as prefix/chars, prefix = longest answer prefix found in the document
(verbatim.prefix_match), so "38/793" means the same thing it meant there. A hit is WRONG
when its sha differs from the cold answer's. Coverage (verbatim.coverage) is printed next
to it: one wrong character pins the prefix at 38 however good the rest is, so coverage is
what separates a greedy divergence from a collapse (verbatim.py:27-31).

Every (arm, residue) uses its own cache_salt, so arms and residues never hit each other
and one server runs them all; a cold request is a real cold prefill.

Arms (--arms, comma-separated, in order):
  base    cold, then --measured self-hits back to back.          The original protocol.
  reset   cold, a short unrelated prose request, then one self-hit. Client-only test of
          hypothesis H1 (the hit inherits the previous request's adaptive-length state);
          needs no patch.
  nohit   a ~1.5k-token copy prompt cold, a long copy request, then the short prompt cold
          again under a new salt. Prefill is one step both times and nothing is a cache
          hit: if the two short answers differ, the cache is not needed at all.
  replay  needs experiments/patches/kvarn-varlen-sched.patch with VLLM_KVV_TRACE and
          VLLM_KVV_REPLAY pointing into --kvv-dir, and this script running where the
          server's filesystem is (docker exec / docker cp). cold (schedule s1 traced),
          hit (s2), a NEW-salt cold run forced to s2, and the hit forced to s1.
            cold@s2 == hit   -> the deviation is the schedule, reproduced without any hit
            hit@s1  == cold  -> the hit path is clean under the cold schedule

Traces: with --kvv-dir holding sched.jsonl (VLLM_KVV_TRACE) and/or fp.jsonl (VLLM_KVV_FP,
experiments/patches/kvarn-varlen-fp.patch), each request's slice is attached to its
result: its verify-length schedule, the H2-H5 invariant flags, and for every hit the
first committed position whose KV / GDN fingerprint differs from its cold run (only
meaningful when the two schedules agree up to that position; the report says whether
they do).

  python bench/kvarn_varlen_repro.py --label base --residues 16,64,0 --arms base,reset
  python bench/kvarn_varlen_repro.py --label replay --residues 16 --arms replay --kvv-dir /tmp/kvv

Exit status: 0 when every measured hit matched its cold answer, 1 when any was WRONG,
2 on a setup error. --json writes everything (texts, schedules, fingerprint diffs).
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from verbatim import coverage, prefix_match  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ASK = "\n\nGengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."
PROSE = "Write two plain sentences about the weather in Copenhagen in October."
RUN = f"{time.time_ns():x}"


def load_doc(path, chars):
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return f.read()[:chars]


def content_for(doc, pad):
    return "Dokument:\n\n" + doc + ASK + " x" * pad


def msgs(content):
    return [{"role": "user", "content": content}]


def prompt_len(content):
    d = harness.post("/tokenize", {"messages": msgs(content), "add_generation_prompt": True,
                                   "chat_template_kwargs": {"enable_thinking": False}},
                     timeout=300)
    return int(d["count"])


def pad_for(doc, residue):
    """A pad that puts the templated prompt at `residue` mod 128. One " x" is one token
    here (residue_sweep.py), so one correction step normally suffices."""
    pad = (residue - prompt_len(content_for(doc, 0))) % 128
    for _ in range(4):
        n = prompt_len(content_for(doc, pad))
        if n % 128 == residue:
            return pad, n
        pad += (residue - n) % 128
    raise RuntimeError(f"could not reach residue {residue}: last templated length {n}")


class Trace:
    """Line-offset brackets over the server's kvv trace files (sched.jsonl, fp.jsonl)."""

    def __init__(self, d):
        self.d = d
        self.paths = {k: os.path.join(d, f) for k, f in (("sched", "sched.jsonl"), ("fp", "fp.jsonl"))} if d else {}

    def mark(self):
        return {k: self._count(p) for k, p in self.paths.items()}

    @staticmethod
    def _count(p):
        if not os.path.exists(p):
            return 0
        with open(p) as f:
            return sum(1 for _ in f)

    def since(self, mark):
        out = {}
        for k, p in self.paths.items():
            if not os.path.exists(p):
                continue
            with open(p) as f:
                lines = f.readlines()[mark.get(k, 0):]
            out[k] = [json.loads(x) for x in lines if x.strip()]
        return out

    def replay_path(self):
        return os.path.join(self.d, "replay.txt") if self.d else None


def chat(content, salt, max_tokens, trace, replay=None):
    payload = {"messages": msgs(content), "max_tokens": max_tokens, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}, "cache_salt": salt}
    rp = trace.replay_path()
    if replay is not None:
        with open(rp, "w") as f:
            f.write(",".join(str(-1 if x is None else x) for x in replay))
    m = trace.mark()
    d0 = harness.spec()
    t0 = time.perf_counter()
    try:
        r = harness.post("/v1/chat/completions", payload, timeout=1800)
    finally:
        if replay is not None and os.path.exists(rp):
            os.remove(rp)
    secs = time.perf_counter() - t0
    d1 = harness.spec()
    dn = d1[0] - d0[0]
    ans = r["choices"][0]["message"].get("content") or ""
    u = r.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return {"ans": ans, "sha": hashlib.sha1(ans.encode()).hexdigest()[:12],
            "prompt_tokens": u.get("prompt_tokens"), "cached": det.get("cached_tokens"),
            "gen": u.get("completion_tokens"), "finish": r["choices"][0].get("finish_reason"),
            "tps": (d1[1] - d0[1]) / dn + 1 if dn else float("nan"), "secs": round(secs, 1),
            "trace": trace.since(m)}


# ----------------------------------------------------------------------------- traces
def req_id_of(tr):
    """The engine request id that dominates a request's trace slice."""
    counts = {}
    for rec in tr.get("fp", []):
        if rec.get("t") == "sched":
            for q in rec["reqs"]:
                counts[q["id"]] = counts.get(q["id"], 0) + 1
    for rec in tr.get("sched", []):
        for i in rec.get("ids", []):
            counts[i] = counts.get(i, 0) + 1
    return max(counts, key=counts.get) if counts else None


def schedule(tr):
    """((num_computed, scheduled tokens) per step, draft count decided per emitting call;
    verify length = draft count + 1)."""
    rid = req_id_of(tr)
    ran, decided = [], []
    for rec in tr.get("sched", []):
        if rid not in rec["ids"]:
            continue
        i = rec["ids"].index(rid)
        em = rec["emitted"][i] if rec.get("emitted") else None
        if em is not None and em >= 1:
            decided.append(rec["out"])
    steps = [q for rec in tr.get("fp", []) if rec.get("t") == "sched"
             for q in rec["reqs"] if q["id"] == rid]
    ran = [(q["nc"], q["q"]) for q in steps]
    return ran, decided


def flags(tr):
    out = {}
    for rec in tr.get("fp", []):
        for k in ("H2_back", "H2_reloc", "H2_null", "H3", "H4", "H5"):
            if k in rec:
                out.setdefault(k, []).append({"step": rec.get("step"), "t": rec.get("t"),
                                              "committed": rec.get("committed"), "v": rec[k]})
    return out


def _by_committed(tr, kind):
    rid = req_id_of(tr)
    nc_of = {}
    for rec in tr.get("fp", []):
        if rec.get("t") == "sched":
            for q in rec["reqs"]:
                if q["id"] == rid:
                    nc_of[rec["step"]] = q["nc"]
    out = {}
    for rec in tr.get("fp", []):
        if rec.get("t") != kind:
            continue
        if kind == "kv":
            if rec.get("id", rid) != rid:
                continue
            c = rec["committed"]
        else:
            if rec.get("id") != rid or rec["step"] not in nc_of:
                continue
            c = nc_of[rec["step"]]
        out.setdefault(c, rec)
    return out


def fp_diff(cold_tr, hit_tr):
    """First committed position where the hit READ different KV or GDN state than its
    cold run, and whether the two schedules agreed up to there."""
    res = {}
    for kind, field in (("kv", "fp"), ("gdn", "ssm")):
        a, b = _by_committed(cold_tr, kind), _by_committed(hit_tr, kind)
        common = sorted(set(a) & set(b))
        first = None
        for c in common:
            if a[c].get(field) != b[c].get(field):
                diff_layers = [i for i, (x, y) in enumerate(zip(a[c][field], b[c][field])) if x != y]
                first = {"committed": c, "layers": diff_layers}
                if kind == "kv" and a[c].get("tiles0") and b[c].get("tiles0"):
                    t = [k for k, (x, y) in enumerate(zip(a[c]["tiles0"], b[c]["tiles0"])) if x != y]
                    first["tiles0_first"] = t[:8]
                break
        res[kind] = {"compared": len(common), "first_diff": first}
    diffs = [v["first_diff"]["committed"] for v in res.values() if v["first_diff"]]
    if diffs:
        lim = min(diffs)
        sa = [x for x in schedule(cold_tr)[0] if x[0] < lim]
        sb = [x for x in schedule(hit_tr)[0] if x[0] < lim]
        start = min((x[0] for x in sb), default=0)
        # Steps the hit skipped (its prefill started at the hit length) are not compared.
        res["schedules_agree_before_diff"] = [x for x in sa if x[0] >= start] == sb
    return res


# ----------------------------------------------------------------------------- report
def frac(r, doc):
    return f"{prefix_match(r['ans'], doc)}/{len(r['ans'])}"


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def show(label, arm, res, tag, r, doc, ref=None, ref_name="cold"):
    s = (f"[{label}] {arm:6s} res={res:3d} {tag:10s} {frac(r, doc):>9s} cov={coverage(r['ans'], doc):.2f} "
         f"sha={r['sha']} tok/step={r['tps']:.2f} prompt={r['prompt_tokens']} cached={r['cached']} "
         f"gen={r['gen']} {r['secs']}s")
    if ref is not None:
        same = r["sha"] == ref["sha"]
        s += f"  == {ref_name}" if same else ("  WRONG" if ref_name == "cold" else f"  != {ref_name}")
        if not same:
            i = first_div(ref["ans"], r["ans"])
            r["first_div"] = i
            s += (f"\n      diverges at char {i}: {ref_name:5s} {ref['ans'][max(0, i - 25):i + 25]!r}"
                  f"\n                         this  {r['ans'][max(0, i - 25):i + 25]!r}")
    ran, decided = schedule(r["trace"])
    if decided:
        s += f"\n      draft counts decided (verify = +1): {decided[:24]}{' ...' if len(decided) > 24 else ''}"
    elif ran:
        s += f"\n      steps (num_computed, q): {ran[:16]}"
    fl = flags(r["trace"])
    if fl:
        s += "\n      FLAGS: " + ", ".join(f"{k} x{len(v)} first={v[0]}" for k, v in fl.items())
    print(s, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--label", default="kvv")
    ap.add_argument("--residues", default="16,64,0",
                    help="templated prompt length mod 128 (the 2026-08 runs broke at 16/32/64/96/124, not 0)")
    ap.add_argument("--arms", default="base,reset", help="base,reset,nohit,replay")
    ap.add_argument("--measured", type=int, default=2, help="self-hits after the cold arm request (base arm)")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", os.path.join(REPO, "docs", "gotchas.md")))
    ap.add_argument("--chars", type=int, default=72000, help="document prefix length, as residue_sweep.py")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--kvv-dir", default=os.environ.get("KVV_DIR", ""),
                    help="directory of the server's VLLM_KVV_TRACE / VLLM_KVV_FP files (sched.jsonl, fp.jsonl) "
                         "and of VLLM_KVV_REPLAY (replay.txt); must be the server's filesystem")
    ap.add_argument("--json", default="", help="write all results here")
    a = ap.parse_args()

    arms = [x for x in a.arms.split(",") if x]
    bad_arm = [x for x in arms if x not in ("base", "reset", "nohit", "replay")]
    if bad_arm:
        sys.exit(f"unknown arm(s): {bad_arm}")
    trace = Trace(a.kvv_dir)
    if "replay" in arms and not (a.kvv_dir and os.path.isdir(a.kvv_dir)):
        print("replay arm needs --kvv-dir on the server's filesystem (VLLM_KVV_TRACE=<dir>/sched.jsonl, "
              "VLLM_KVV_REPLAY=<dir>/replay.txt)", file=sys.stderr)
        return 2
    doc = load_doc(a.corpus, a.chars)
    results, wrong = [], 0

    def salt(*parts):
        return "kvv-" + RUN + "-" + "-".join(str(p) for p in parts)

    for ri, res in enumerate(int(x) for x in a.residues.split(",") if x != ""):
        pad, n = pad_for(doc, res)
        content = content_for(doc, pad)
        print(f"[{a.label}] residue {res}: pad={pad} templated length {n}", flush=True)
        for arm in arms:
            if arm == "nohit" and ri:
                continue                       # the nohit prompt has no pad: run it once
            rows = []
            if arm == "base":
                s = salt(arm, res)
                cold = chat(content, s, a.max_tokens, trace)
                show(a.label, arm, res, "cold", cold, doc)
                rows.append(("cold", cold))
                for i in range(a.measured):
                    hit = chat(content, s, a.max_tokens, trace)
                    show(a.label, arm, res, f"self-hit{i + 1}", hit, doc, cold)
                    hit["fp_vs_cold"] = fp_diff(cold["trace"], hit["trace"]) if hit["trace"].get("fp") else None
                    if hit["fp_vs_cold"]:
                        print(f"      fp vs cold: {hit['fp_vs_cold']}", flush=True)
                    wrong += hit["sha"] != cold["sha"]
                    rows.append((f"hit{i + 1}", hit))
            elif arm == "reset":
                s = salt(arm, res)
                cold = chat(content, s, a.max_tokens, trace)
                show(a.label, arm, res, "cold", cold, doc)
                inter = chat(PROSE, salt(arm, res, "prose"), 48, trace)
                print(f"[{a.label}] {arm:6s} res={res:3d} interposer gen={inter['gen']} {inter['ans'][:60]!r}", flush=True)
                hit = chat(content, s, a.max_tokens, trace)
                show(a.label, arm, res, "self-hit", hit, doc, cold)
                hit["fp_vs_cold"] = fp_diff(cold["trace"], hit["trace"]) if hit["trace"].get("fp") else None
                if hit["fp_vs_cold"]:
                    print(f"      fp vs cold: {hit['fp_vs_cold']}", flush=True)
                wrong += hit["sha"] != cold["sha"]
                rows += [("cold", cold), ("interposer", inter), ("hit", hit)]
            elif arm == "nohit":
                short = content_for(doc[:6000], 0)
                inter = chat(PROSE, salt(arm, res, "prose"), 48, trace)
                c1 = chat(short, salt(arm, res, "a"), a.max_tokens, trace)
                show(a.label, arm, res, "short-1", c1, doc)
                lng = chat(content, salt(arm, res, "long"), a.max_tokens, trace)
                show(a.label, arm, res, "long-copy", lng, doc)
                c2 = chat(short, salt(arm, res, "b"), a.max_tokens, trace)
                show(a.label, arm, res, "short-2", c2, doc, c1, "short1")
                wrong += c2["sha"] != c1["sha"]
                rows += [("prose", inter), ("short1", c1), ("long", lng), ("short2", c2)]
            elif arm == "replay":
                s = salt(arm, res)
                cold = chat(content, s, a.max_tokens, trace)
                show(a.label, arm, res, "cold", cold, doc)
                hit = chat(content, s, a.max_tokens, trace)
                show(a.label, arm, res, "self-hit", hit, doc, cold)
                wrong += hit["sha"] != cold["sha"]
                s1, s2 = schedule(cold["trace"])[1], schedule(hit["trace"])[1]
                if not s1 or not s2:
                    print("replay: no schedule traced (is VLLM_KVV_TRACE=<kvv-dir>/sched.jsonl set on the server?)",
                          file=sys.stderr)
                    return 2
                print(f"      s1 (cold) {s1[:24]}\n      s2 (hit)  {s2[:24]}\n      schedules "
                      f"{'IDENTICAL' if s1 == s2 else 'DIFFER at decode step ' + str(next((i for i, (x, y) in enumerate(zip(s1, s2)) if x != y), min(len(s1), len(s2))))}",
                      flush=True)
                cold2 = chat(content, salt(arm, res, "cold-s2"), a.max_tokens, trace, replay=s2)
                show(a.label, arm, res, "cold@s2", cold2, doc, hit, "hit")
                hit1 = chat(content, s, a.max_tokens, trace, replay=s1)
                show(a.label, arm, res, "hit@s1", hit1, doc, cold)
                hit1["fp_vs_cold"] = fp_diff(cold["trace"], hit1["trace"]) if hit1["trace"].get("fp") else None
                if hit1["fp_vs_cold"]:
                    print(f"      fp hit@s1 vs cold: {hit1['fp_vs_cold']}", flush=True)
                verdict = ("schedule-only (no corruption): cold@s2 reproduces the hit, hit@s1 reproduces cold"
                           if cold2["sha"] == hit["sha"] and hit1["sha"] == cold["sha"] else
                           "hit-path corruption: the hit differs from cold even under the cold schedule"
                           if hit1["sha"] != cold["sha"] else
                           "mixed: hit@s1 is clean but cold@s2 does not reproduce the hit")
                print(f"[{a.label}] replay res={res} VERDICT: {verdict}", flush=True)
                rows += [("cold", cold), ("hit", hit), ("cold@s2", cold2), ("hit@s1", hit1),
                         ("verdict", {"text": verdict, "s1": s1, "s2": s2})]
            results.append({"residue": res, "arm": arm, "pad": pad, "len": n,
                            "rows": [{"tag": t, **r} for t, r in rows]})
    print(f"[{a.label}] DONE: {wrong} measured answer(s) differ from their cold reference", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"label": a.label, "corpus": a.corpus, "run": RUN, "results": results}, f, indent=1)
    return 1 if wrong else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"kvarn_varlen_repro: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
