#!/usr/bin/env python3
"""X-08 recorder: what greedy_equal.py `record` stores, plus what the classifier needs. Run on the GPU box (needs a server).

  PYTHONPATH=/app/bench record_top2.py LABEL [--sets short_chat,code] [--max-tokens 256] [--top-n 5] [--passes 2]
                                      [--doc-ctx 8000 --corpus PATH] [--out DIR]

Writes OUT/greedy_<LABEL>.json in greedy_equal.py's schema (ids, margins, finish, text) and adds per item:
  top       top-n [token_id, logprob] per position (greedy_equal.py keeps only the margin, not the ids)
  messages  the request, so rescore_replay.py can replay a prefix
Differences from greedy_equal.py record (the harness file is untouched):
  --passes 2   sends every prompt again; the repeat is stored as `<id>@p2` and is a prefix-cache hit when the server
               has PREFIX_CACHE on. This is the only way bench-7 can reach the KVarN adaptive + prefix case.
  --doc-ctx N  also records the long_context set as "Dokument:\\n\\n" + corpus[:3.6*N chars] + question (as sweep.py),
               including the `copy` task, which is the self-hit case. greedy_equal.py skips this set (rows have no `prompt`).
Concurrency 1, T=0, thinking off, like greedy_equal.py.
"""
import argparse
import json
import os
import sys

HARNESS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "harness"))


def prompts(args):
    out = {}
    for s in filter(None, args.sets.split(",")):
        for l in open(os.path.join(HARNESS, "prompts", s + ".jsonl")):
            r = json.loads(l)
            out[f"{s}/{r['id']}"] = r["prompt"]
    if args.doc_ctx:
        path = args.corpus or os.environ.get("HARNESS_CORPUS") or "/cache/bench/labd_corpus_long.txt"
        doc = open(os.path.expanduser(path)).read()[: int(args.doc_ctx * 3.6)]
        for l in open(os.path.join(HARNESS, "prompts", "long_context.jsonl")):
            r = json.loads(l)
            out[f"long{args.doc_ctx}/{r['id']}"] = "Dokument:\n\n" + doc + "\n\n" + r["question"]
    return out


def to_item(ch, messages):
    content = (ch.get("logprobs") or {}).get("content")
    if not content:
        return None
    ids, top, margins = [], [], []
    for c in content:
        ids.append(int(c["token"].split(":")[1]))
        t = [[int(x["token"].split(":")[1]), x["logprob"]] for x in c.get("top_logprobs") or []]
        top.append(t)
        margins.append(round(t[0][1] - t[1][1], 4) if len(t) > 1 else None)
    return dict(ids=ids, margins=margins, top=top, finish=ch["finish_reason"], text=ch["message"]["content"], messages=messages)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("label"); ap.add_argument("--sets", default="short_chat,code")
    ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--passes", type=int, default=1); ap.add_argument("--doc-ctx", type=int, default=0)
    ap.add_argument("--corpus"); ap.add_argument("--out", default=".")
    a = ap.parse_args()
    import harness  # bench/harness.py (PYTHONPATH=/app/bench, as run_suite.sh sets)
    out = {"label": a.label, "max_tokens": a.max_tokens, "items": {},
           "env": {k: os.environ[k] for k in ("SPEC", "CTX", "DFLASH_TOKENS", "PREFIX_CACHE", "PTH_FA",
                                              "VLLM_DFLASH2_LOOKUP_ADAPTIVE") if k in os.environ}}
    for p in range(1, a.passes + 1):
        for pid, prompt in prompts(a).items():
            msgs = [{"role": "user", "content": prompt}]
            r = harness.post("/v1/chat/completions", {
                "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0, "logprobs": True,
                "top_logprobs": a.top_n, "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}})
            item = to_item(r["choices"][0], msgs)
            if item is None:
                sys.exit(f"invalid: no logprobs for {pid}")
            key = pid if p == 1 else f"{pid}@p{p}"
            out["items"][key] = item
            print(f"{key}: {len(item['ids'])} tokens, finish={item['finish']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"greedy_{a.label}.json")
    json.dump(out, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    main()
