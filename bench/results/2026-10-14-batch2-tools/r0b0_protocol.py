"""S3a: our stack at r0b0tlab's GSM8K acceptance protocol, one request at a time (C1).

Their scripts/acceptance_check.py (r0b0tlab/qwen38-exl3-dflash2, fetched 2026-10-14): the first --n GSM8K test
questions, greedy, max_new_tokens 512, ctx 8192, prompt "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n",
and two per-request means:
  acceptance length = new_tokens / (new_tokens - accepted_draft_tokens)
  tok_per_s        = new_tokens / wall time of the generate call (prefill included)
Here a request's drafts and accepted tokens are the vllm:spec_decode_* counter deltas around it, which is exact
at C1. vLLM's per-step tokens = 1 + accepted/drafts is printed beside their acceptance length, and decode tok/s
(first token to last, prefill excluded) beside their tok_per_s.

  --template chat  the served chat template with enable_thinking false (the brief's "thinking off")
  --template raw   their literal prompt string through /v1/completions (no think block, as they ran it)

  venv/bin/python r0b0_protocol.py <tag> [--n 40] [--max-tokens 512] [--template chat|raw] [--reps 2]
"""
import glob, json, sys, time

sys.path.insert(0, "/app/bench")
import harness  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


TAG = sys.argv[1]
N = int(arg("--n", 40))
MAXTOK = int(arg("--max-tokens", 512))
TEMPLATE = arg("--template", "chat")
REPS = int(arg("--reps", 2))
SC = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total")

qs = pq.read_table(glob.glob("/app/models/quality-data/gsm8k/main/test-*.parquet")[0]).column("question").to_pylist()[:N]


def prompt_ids(q):
    if TEMPLATE == "raw":
        r = harness.post("/tokenize", {"prompt": f"<|im_start|>user\n{q.strip()}<|im_end|>\n<|im_start|>assistant\n",
                                       "add_special_tokens": False})
    else:
        r = harness.post("/tokenize", {"messages": [{"role": "user", "content": q}], "add_generation_prompt": True,
                                       "chat_template_kwargs": {"enable_thinking": False}})
    return r["tokens"]


def counters():
    m = harness.metrics(*SC)
    return tuple(m.get(k, 0.0) for k in SC)


def generate(ids):
    payload = {"prompt": ids, "max_tokens": MAXTOK, "temperature": 0, "logprobs": 0,
               "return_tokens_as_token_ids": True, "stream": True}
    n, t_first = 0, None
    t0 = time.time()
    for ev in harness.stream("/v1/completions", payload):
        for ch in ev.get("choices", []):
            toks = (ch.get("logprobs") or {}).get("tokens") or []
            if toks and t_first is None:
                t_first = time.time()
            n += len(toks)
    t1 = time.time()
    return n, t0, t_first or t1, t1


ids = [prompt_ids(q) for q in qs]
for _ in range(2):  # warm the path the measured requests take
    generate(ids[0][:16])
print(f"# {TAG} template={TEMPLATE} n={N} max_tokens={MAXTOK} reps={REPS}", flush=True)
summ = []
for rep in range(1, REPS + 1):
    rows = []
    for i, p in enumerate(ids):
        a = counters()
        n, t0, tf, t1 = generate(p)
        b = counters()
        drafts, acc = b[0] - a[0], b[1] - a[1]
        rows.append({"i": i, "prompt_tokens": len(p), "new_tokens": n, "drafts": drafts, "accepted": acc,
                     "al_theirs": n / max(1, n - acc) if n else 0.0, "tok_per_step": 1 + acc / max(1, drafts),
                     "tps_theirs": n / (t1 - t0) if n else 0.0,
                     "decode_tps": (n - 1) / (t1 - tf) if n > 1 and t1 > tf else 0.0, "ttft": tf - t0})
    u = [r for r in rows if r["new_tokens"] > 0]
    mean = lambda k: sum(r[k] for r in u) / max(1, len(u))  # noqa: E731
    tot_n = sum(r["new_tokens"] for r in u)
    agg_step = 1 + sum(r["accepted"] for r in u) / max(1, sum(r["drafts"] for r in u))
    agg_dec = (sum(r["new_tokens"] - 1 for r in u if r["decode_tps"])
               / max(1e-9, sum((r["new_tokens"] - 1) / r["decode_tps"] for r in u if r["decode_tps"])))
    s = {"rep": rep, "n": len(rows), "hit_cap": sum(1 for r in rows if r["new_tokens"] >= MAXTOK),
         "mean_al_theirs": mean("al_theirs"), "mean_tps_theirs": mean("tps_theirs"),
         "mean_decode_tps": mean("decode_tps"), "mean_ttft_ms": 1000 * mean("ttft"),
         "agg_tok_per_step": agg_step, "agg_decode_tps": agg_dec, "new_tokens": tot_n}
    summ.append({"summary": s, "per_request": rows})
    print(f"R0B0 {TAG} rep={rep} n={s['n']} hit_cap={s['hit_cap']} mean_accept_len(theirs)={s['mean_al_theirs']:.3f} "
          f"mean_tok/s(theirs,incl.prefill)={s['mean_tps_theirs']:.1f} mean_decode_tok/s={s['mean_decode_tps']:.1f} "
          f"agg_tok/step={agg_step:.3f} agg_decode_tok/s={agg_dec:.1f} mean_ttft={s['mean_ttft_ms']:.0f}ms "
          f"new_tokens={tot_n}", flush=True)
json.dump(summ, open(f"/tmp/r0b0_{TAG}.json", "w"), indent=1)
