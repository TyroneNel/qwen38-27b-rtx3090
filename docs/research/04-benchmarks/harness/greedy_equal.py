#!/usr/bin/env python3
"""Greedy (temperature 0) equality of two boots on the same prompts: spec vs non-spec, or any A/B that must be lossless.
Speculative decoding is exact in distribution, so at T=0 the token ids should match; this finds where they don't.

  greedy_equal.py record LABEL [--sets short_chat,code] [--max-tokens 256] [--out DIR]
  greedy_equal.py compare A.json B.json [--near-tie 0.1]

record: one boot, concurrency 1 (no batch-shape numerics), fixed prompts, T=0, thinking off, token ids via
return_tokens_as_token_ids, plus the top-2 logprob margin at every position. Writes OUT/greedy_<LABEL>.json.
compare: per prompt, first differing token index and the top-1/top-2 margin there in each run. A divergence where
both margins are < --near-tie nats is a numerical near-tie (different kernel shapes round differently), not a
speculation bug; a divergence with a large margin is a real defect. Exit 0 all identical, 1 any divergence
(verdict line says near-tie or real), 2 invalid (different prompt sets, request failed, no token ids).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def record(args):
    import harness
    sets = {}
    for s in args.sets.split(","):
        for l in open(os.path.join(HERE, "prompts", s + ".jsonl")):
            r = json.loads(l)
            if "prompt" in r:
                sets[f"{s}/{r['id']}"] = r["prompt"]
    out = {"label": args.label, "max_tokens": args.max_tokens, "items": {},
           "env": {k: os.environ[k] for k in ("SPEC", "CTX", "DFLASH_TOKENS", "PREFIX_CACHE", "PTH_FA") if k in os.environ}}
    for pid, prompt in sets.items():
        r = harness.post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens, "temperature": 0,
            "logprobs": True, "top_logprobs": 2, "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}})
        ch = r["choices"][0]
        content = (ch.get("logprobs") or {}).get("content")
        if not content:
            sys.exit(f"invalid: no logprobs for {pid}; the server did not return token ids")
        ids = [int(c["token"].split(":")[1]) for c in content]
        margins = [round(c["top_logprobs"][0]["logprob"] - c["top_logprobs"][1]["logprob"], 4)
                   if len(c.get("top_logprobs") or []) > 1 else None for c in content]
        out["items"][pid] = dict(ids=ids, margins=margins, finish=ch["finish_reason"], text=ch["message"]["content"])
        print(f"{pid}: {len(ids)} tokens, finish={ch['finish_reason']}", flush=True)
    path = os.path.join(args.out, f"greedy_{args.label}.json")
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(path, "w"))
    print("wrote", path)


def compare(args):
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    if a["items"].keys() != b["items"].keys() or a["max_tokens"] != b["max_tokens"]:
        print("INVALID: different prompt sets or max_tokens"); return 2
    div = []
    for pid in a["items"]:
        x, y = a["items"][pid], b["items"][pid]
        k = next((i for i, (p, q) in enumerate(zip(x["ids"], y["ids"])) if p != q), None)
        if k is None and len(x["ids"]) != len(y["ids"]):
            k = min(len(x["ids"]), len(y["ids"]))
        if k is None:
            print(f"{pid}: IDENTICAL ({len(x['ids'])} tokens)")
            continue
        ma = x["margins"][k] if k < len(x["margins"]) else None
        mb = y["margins"][k] if k < len(y["margins"]) else None
        tie = ma is not None and mb is not None and max(ma, mb) < args.near_tie
        div.append(tie)
        print(f"{pid}: DIVERGES at token {k}/{len(x['ids'])} (A margin {ma}, B margin {mb}) -> "
              f"{'near-tie' if tie else 'REAL'}")
    n = len(a["items"])
    print(f"SUMMARY greedy_equal {a['label']} vs {b['label']}: {n - len(div)}/{n} identical, "
          f"{sum(div)} near-tie, {len(div) - sum(div)} real")
    return 1 if div else 0


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("record")
r.add_argument("label"); r.add_argument("--sets", default="short_chat,code")
r.add_argument("--max-tokens", type=int, default=256); r.add_argument("--out", default=".")
c = sub.add_parser("compare")
c.add_argument("a"); c.add_argument("b"); c.add_argument("--near-tie", type=float, default=0.1)
a = ap.parse_args()
if a.cmd == "record":
    record(a)
else:
    sys.exit(compare(a))
