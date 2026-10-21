#!/usr/bin/env python3
"""X-08 / H08 divergence classifier for the greedy T=0 equality gate (bench-7). CPU only, stdlib only.

  classify_divergence.py classify REF.json CAND.json [CAND2.json ...] [--near-tie 0.01] [--ulp-band 0.25]
                                  [--out report.json] [--rescore-out requests.json]
  classify_divergence.py judge report.json replays.json [--tol 0.25] [--out final.json]

Input: greedy_equal.py `record` JSON ({"items": {pid: {"ids", "margins", ...}}}), or the superset written by
record_top2.py (adds "top" = [[[token_id, logprob], ...] per position] and "messages"). REF is the SPEC=off run.

At the first differing position k (or the end of the shorter run), the gap is how far the *other* run's token sits below
each run's own top-1 there. Both runs saw the same prefix, so a numerical flip shows a small gap on both sides.
  identical     same ids, same length
  near_tie      gap < --near-tie on BOTH sides, other token is the runner-up. Numerical, accepted.
  ulp_band      gap <= --ulp-band (0.25 = D18 replay ceiling, two bf16 ulps for logits in [16,32), EAGLE-4 tools/fp32_arbiter.py:7-12), or
                evidence is one-sided / a lower bound. Needs the replay (judge) to decide. NOT a pass yet.
  real          gap > --ulp-band: a defect.
  unknown       no logprobs at k in either run. Fail closed.
judge adds replay_tie / replay_argmax (pass) or real (fail) to ulp_band entries from rescore_replay.py output.
Exit: 0 gate passes, 1 any real, 2 invalid input or unknown, 3 no real but ulp_band entries still unjudged.
"""
import argparse
import json
import sys
from typing import Any


def gap_at(item, k, other):
    """(gap, kind) of `other` token below this run's top-1 at k.
    kind: exact (other is in the saved top-n), lower_bound (it is not: true gap >= gap), assumed (margins only,
    other taken to be the runner-up), None (no logprobs at k)."""
    top = (item.get("top") or [None] * (k + 1))[k] if k < len(item.get("ids", [])) else None
    if top and len(top) > 1:
        lp = {t: p for t, p in top}
        if other is None:  # the other run had stopped: its EOS is not in ids, take the runner-up
            return round(top[0][1] - top[1][1], 6), "assumed"
        if other in lp:
            return round(top[0][1] - lp[other], 6), "exact"
        return round(top[0][1] - top[-1][1], 6), "lower_bound"
    m = item.get("margins") or []
    if k < len(m) and m[k] is not None and k < len(item["ids"]):
        return m[k], "assumed"
    return None, None


def classify_pair(a, b, near_tie=0.01, ulp_band=0.25) -> dict[str, Any]:
    ia, ib = a["ids"], b["ids"]
    k = next((i for i, (p, q) in enumerate(zip(ia, ib)) if p != q), None)
    if k is None and len(ia) == len(ib):
        return dict(status="identical", n=len(ia))
    length = k is None
    if length:
        k = min(len(ia), len(ib))
    ta = ia[k] if k < len(ia) else None  # None: that run had already stopped
    tb = ib[k] if k < len(ib) else None
    sides = {}
    for name, run, other in (("ref", a, tb), ("cand", b, ta)):
        g, kind = gap_at(run, k, other) if run["ids"][k:k + 1] else (None, None)
        sides[name] = dict(gap=g, kind=kind)
    known = [s for s in sides.values() if s["kind"]]
    e = dict(k=k, lengths=[len(ia), len(ib)], ref_token=ta, cand_token=tb, sides=sides,
             notes=[n for n, c in (("token0", k == 0), ("last_token", k == min(len(ia), len(ib)) - 1),
                                   ("length_divergence", length), ("one_sided", len(known) == 1)) if c])
    if not known:
        return dict(e, status="unknown", reason="no logprobs at the divergence position in either run")
    gap = max(s["gap"] for s in known)
    soft = any(s["kind"] != "exact" for s in known)  # lower bound / assumed runner-up
    e["gap"] = gap
    if gap > ulp_band:
        return dict(e, status="real")
    if gap < near_tie and len(known) == 2 and not length and not any(s["kind"] == "lower_bound" for s in known):
        return dict(e, status="near_tie")
    return dict(e, status="ulp_band", needs_rescore=True,
                reason="one-sided/lower-bound/length evidence" if soft or len(known) < 2 or length else "inside one bf16 ulp")


def rescore_request(pid, e, ref_item):
    return dict(id=pid, k=e["k"], prefix_ids=ref_item["ids"][:e["k"]], ref_token=e["ref_token"],
                cand_token=e["cand_token"], messages=ref_item.get("messages"))


def judge_one(e, replay, tol=0.25) -> dict[str, Any]:
    """Decide an ulp_band entry from a SPEC=off replay's top-n at k: [[token_id, logprob], ...] best first.
    Pass when the two tokens are within `tol` nats (2 x 0.125 bf16 ulp) or the candidate is the replay's top-1.
    A token missing from the list sits at or below the list's last logprob, which gives a lower bound on d."""
    top = replay.get("top") or []
    if len(top) < 2:
        return dict(e, status="unknown", reason="replay returned no top logprobs", needs_rescore=False)
    lp = {t: p for t, p in top}
    ref, cand = e["ref_token"], e["cand_token"]
    out: dict[str, Any] = dict(e, needs_rescore=False, replay=dict(top1=top[0][0], tol=tol))
    if ref is None or cand is None:  # length divergence: the stop token is the assumed runner-up
        d = top[0][1] - top[1][1]
        return dict(out, status="replay_tie" if d <= tol else "real", replay=dict(out["replay"], d=round(d, 6)))
    if top[0][0] == cand:
        return dict(out, status="replay_argmax")
    floor = top[-1][1]
    lo = abs(lp.get(ref, floor) - lp.get(cand, floor)) if (ref in lp) != (cand in lp) else None
    if ref in lp and cand in lp:
        d = abs(lp[ref] - lp[cand])
        return dict(out, status="replay_tie" if d <= tol else "real", replay=dict(out["replay"], d=round(d, 6)))
    if lo is not None and lo > tol:
        return dict(out, status="real", replay=dict(out["replay"], d_min=round(lo, 6)))
    return dict(out, status="unknown", reason="token outside the replay's top-n and gap not provable")


def summarize(report):
    c = {}
    for e in report["entries"].values():
        c[e["status"]] = c.get(e["status"], 0) + 1
    return c


def exit_code(counts):
    if counts.get("real"):
        return 1
    if counts.get("unknown"):
        return 2
    return 3 if counts.get("ulp_band") else 0


def worse(a, b):
    sev = {0: 0, 3: 1, 2: 2, 1: 3}  # real > unknown/invalid > pending > pass
    return a if sev[a] >= sev[b] else b


def cmd_classify(args):
    ref = json.load(open(args.ref))
    out, reqs, code = {}, [], 0
    for path in args.cands:
        cand = json.load(open(path))
        if ref["items"].keys() != cand["items"].keys() or ref.get("max_tokens") != cand.get("max_tokens"):
            print(f"INVALID {path}: different prompt sets or max_tokens"); code = 2; continue
        entries = {}
        for pid in ref["items"]:
            e = classify_pair(ref["items"][pid], cand["items"][pid], args.near_tie, args.ulp_band)
            entries[pid] = e
            if e.get("needs_rescore"):
                reqs.append(dict(rescore_request(pid, e, ref["items"][pid]), cand_label=cand["label"]))
            extra = f" at {e['k']}/{e['lengths']} gap {e.get('gap')} {e['notes']}" if e["status"] != "identical" else ""
            print(f"{ref['label']} vs {cand['label']} {pid}: {e['status'].upper()}{extra}")
        rep = dict(ref=ref["label"], cand=cand["label"], near_tie=args.near_tie, ulp_band=args.ulp_band, entries=entries)
        out[cand["label"]] = rep
        n = summarize(rep)
        print(f"SUMMARY {ref['label']} vs {cand['label']}: {n} -> exit {exit_code(n)}")
        code = worse(code, exit_code(n))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    if args.rescore_out:
        json.dump(reqs, open(args.rescore_out, "w"))
        print(f"{len(reqs)} rescore requests -> {args.rescore_out}")
    return code


def cmd_judge(args):
    rep = json.load(open(args.report))
    replays = {(r["cand_label"], r["id"]): r for r in json.load(open(args.replays))}
    code = 0
    for label, r in rep.items():
        for pid, e in r["entries"].items():
            if e["status"] == "ulp_band" and (label, pid) in replays:
                r["entries"][pid] = judge_one(e, replays[(label, pid)], args.tol)
        n = summarize(r)
        print(f"SUMMARY judged {r['ref']} vs {label}: {n} -> exit {exit_code(n)}")
        code = worse(code, exit_code(n))
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("ref"); c.add_argument("cands", nargs="+")
    c.add_argument("--near-tie", type=float, default=0.01); c.add_argument("--ulp-band", type=float, default=0.25)
    c.add_argument("--out"); c.add_argument("--rescore-out")
    j = sub.add_parser("judge")
    j.add_argument("report"); j.add_argument("replays"); j.add_argument("--tol", type=float, default=0.25)
    j.add_argument("--out")
    a = ap.parse_args(argv)
    return cmd_classify(a) if a.cmd == "classify" else cmd_judge(a)


if __name__ == "__main__":
    sys.exit(main())
