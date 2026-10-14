"""GSM8K exact match per question, thinking on (S3b) or off (S1's repeat pair).

bench/quality_battery.py's GSM8K lane hardcodes enable_thinking false and max_tokens 768 and keeps only the
accuracy, so it can neither score the effort setting nor say which questions two arms disagree on. This is the
same lane (same first N questions, same instruction, same answer extraction, greedy, 32 client threads) with
--think 0|1, an optional reasoning_effort (omitted = the template default, xhigh), a token budget, and the
per-question result kept. The answer is read from message.content, i.e. after the server's qwen3 reasoning
parser has split the thinking off. finish_reason "length" is counted: a truncated answer is wrong.
--think 0 --max-tokens 768 reproduces quality_battery.py's lane exactly.

  venv/bin/python gsm_effort.py <tag> [--think 1] [--effort low] [--n 200] [--max-tokens 16384] [--conc 32]
"""
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/app/bench")
import harness  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


TAG = sys.argv[1]
EFFORT = arg("--effort", "")
N = int(arg("--n", 200))
MAXTOK = int(arg("--max-tokens", 16384))
CONC = int(arg("--conc", 32))
kw = {"enable_thinking": arg("--think", "1") == "1"}
if EFFORT:
    kw["reasoning_effort"] = EFFORT


def extract_num(s):  # quality_battery.extract_num
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "") if m else None


def one(row):
    i, (q, a) = row
    gold = a.split("####")[-1].strip().replace(",", "")
    r = harness.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": q + "\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."}],
        "max_tokens": MAXTOK, "temperature": 0, "chat_template_kwargs": kw}, timeout=7200)
    ch = r["choices"][0]
    txt = ch["message"].get("content") or ""
    m = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", txt)
    pred = (m.group(1).replace(",", "") if m else extract_num(txt))
    try:
        ok = abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        ok = False
    return {"i": i, "ok": ok, "pred": pred, "gold": gold, "finish": ch.get("finish_reason"),
            "completion_tokens": r["usage"]["completion_tokens"]}


t = pq.read_table("/app/models/quality-data/gsm8k/main/test-00000-of-00001.parquet")
rows = list(enumerate(zip(t.column("question").to_pylist(), t.column("answer").to_pylist())))[:N]
t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(one, rows))
correct = sum(r["ok"] for r in res)
trunc = sum(1 for r in res if r["finish"] == "length")
toks = sum(r["completion_tokens"] for r in res) / len(res)
print(f"GSMEFFORT {TAG} think={kw['enable_thinking']} effort={EFFORT or '<template default>'} n={len(res)} "
      f"correct={correct} acc={correct/len(res):.3f} truncated={trunc} mean_completion_tokens={toks:.0f} "
      f"max_tokens={MAXTOK} {time.time()-t0:.0f}s", flush=True)
print("WRONG", json.dumps([(r["i"], r["pred"], r["gold"], r["finish"]) for r in res if not r["ok"]]), flush=True)
json.dump(res, open(f"/tmp/gsm_effort_{TAG}.json", "w"), indent=1)
