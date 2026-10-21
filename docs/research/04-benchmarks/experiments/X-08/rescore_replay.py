#!/usr/bin/env python3
"""X-08 replay step for the GPU agent. Boot SPEC=off with the SAME CTX / KV dtype / PTH_FA pin as the reference run
(PREFIX_CACHE may be on or off), then:

  PYTHONPATH=/app/bench rescore_replay.py requests.json --out replays.json [--top-n 5]
  classify_divergence.py judge report.json replays.json --out final.json

requests.json comes from `classify_divergence.py classify ... --rescore-out`. For each divergence this rebuilds the
exact common prefix (chat-template prompt tokens from POST /tokenize, then the reference's generated ids up to k),
sends it to /v1/completions as token ids with max_tokens=1 and reads the top-n logprobs of the next token.
Only a prefill-shaped forward at SPEC=off is available on this box: this is a THIRD bf16 vote, not an fp32 oracle
(the 27B target in fp32 is ~108 GB). It separates a fragile argmax from a corrupted forward by gap size, nothing finer.
"""
import argparse
import json
import sys


def replay(r, top_n, harness):
    if not r.get("messages"):
        return dict(r, top=[], error="no messages in request (record with record_top2.py)")
    body = {"messages": r["messages"], "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    prompt = harness.post("/tokenize", body)["tokens"] + r["prefix_ids"]
    c = harness.post("/v1/completions", {"prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": top_n,
                                         "return_tokens_as_token_ids": True})["choices"][0]
    d = c["logprobs"]["top_logprobs"][0]  # {"token_id:N": logprob}
    top = sorted(([int(t.split(":")[1]), lp] for t, lp in d.items()), key=lambda x: -x[1])
    return dict(r, top=top)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("requests"); ap.add_argument("--out", required=True); ap.add_argument("--top-n", type=int, default=5)
    a = ap.parse_args()
    import harness
    res = []
    for r in json.load(open(a.requests)):
        res.append(replay(r, a.top_n, harness))
        print(f"{r['cand_label']} {r['id']} k={r['k']}: top={res[-1]['top'][:2]} {res[-1].get('error', '')}", flush=True)
    json.dump(res, open(a.out, "w"))
    sys.exit(1 if any(x.get("error") for x in res) else 0)


if __name__ == "__main__":
    main()
