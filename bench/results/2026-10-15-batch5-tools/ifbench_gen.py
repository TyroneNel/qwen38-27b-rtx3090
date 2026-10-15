"""S3b: IFBench generation against the running server, one reasoning_effort per call.

docs/quality.md's protocol: AllenAI IFBench (299 prompts, official eval scripts), thinking enabled, model-default
sampling (no sampling fields in the request, so the server's generation config applies). The response scored is
message.content, i.e. after the qwen3 reasoning parser has split the thinking off, which is what the IFBench README
asks for ("process the output to extract the answer without the reasoning chains"). finish_reason "length" is
counted: a truncated response is scored as whatever content it has.

Writes <out>.jsonl in the {"prompt", "response"} format run_eval.py reads. Scoring runs separately with the pinned
IFBench checkout's own run_eval.py (ifbench_setup.sh).

  venv/bin/python ifbench_gen.py <tag> --data IFBench_test.jsonl --out /tmp/ifb-<tag> [--effort low]
        [--max-tokens 16384] [--conc 32]
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/app/bench")
import harness  # noqa: E402


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


TAG = sys.argv[1]
DATA = arg("--data", "")
OUT = arg("--out", f"/tmp/ifb-{TAG}")
EFFORT = arg("--effort", "")
MAXTOK = int(arg("--max-tokens", 16384))
CONC = int(arg("--conc", 32))
kw = {"enable_thinking": True}
if EFFORT:
    kw["reasoning_effort"] = EFFORT

rows = [json.loads(l) for l in open(DATA)]


def one(r):
    resp = harness.post("/v1/chat/completions", {"messages": [{"role": "user", "content": r["prompt"]}],
                                                 "max_tokens": MAXTOK, "chat_template_kwargs": kw}, timeout=7200)
    ch = resp["choices"][0]
    return {"key": r["key"], "prompt": r["prompt"], "response": ch["message"].get("content") or "",
            "finish": ch.get("finish_reason"), "completion_tokens": resp["usage"]["completion_tokens"]}


t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(one, rows))
with open(OUT + ".jsonl", "w") as f:
    for r in res:
        f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")
json.dump(res, open(OUT + "-full.json", "w"), indent=1)
print(f"IFBGEN {TAG} effort={EFFORT or '<template default>'} n={len(res)} "
      f"truncated={sum(1 for r in res if r['finish'] == 'length')} "
      f"empty={sum(1 for r in res if not r['response'].strip())} "
      f"mean_completion_tokens={sum(r['completion_tokens'] for r in res)/len(res):.0f} max_tokens={MAXTOK} "
      f"{time.time()-t0:.0f}s", flush=True)
