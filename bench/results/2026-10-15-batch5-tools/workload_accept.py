"""Teacher-forced tokens per step per workload (E3) and per reasoning effort (E4).

Batch 5 copy (Q3, the CG=40 mechanism): unchanged except that the eight bench/prompts_real.jsonl prompts are added
as tasks real0..real7 - the prompts behind run_benchmarks.sh's C1 T=0 row, where batch 2 saw 3.34 -> 3.48 tokens/step.

bench/labd_accept.py implements prefix-resync teacher forcing but its six tasks are fixed
(Danish instructions over a document corpus built from ~/qwen-serving, which the image does not
have), so it cannot score a GSM8K, code or chat continuation. This is the same method with the
prompts as parameters, kept in bench/results because the brief allows no product changes:

  capture  one greedy continuation per workload (/v1/completions on the chat-templated prompt
           ids, return_tokens_as_token_ids), frozen to <targets>/<name>.json; an existing file is
           reused, never recaptured.
  replay   walk the frozen ids in chunks of --chunk tokens; chunk j sends prompt + T[:j*C] with
           max_tokens=C at temperature 0, diffs the vllm:spec_decode_* counters around it, and
           keeps the chunk only if the returned ids equal T[j*C:(j+1)*C] (dirty chunks reported).
  decode   free-running greedy decode tok/s of the capture request (first token to last), C1.

  venv/bin/python workload_accept.py <tag> --targets DIR [--tasks gsm8k,code,chat]
        [--max-tokens 1024] [--chunk 128] [--think 0|1] [--effort low|...] [--decode-reps 2]
"""
import glob, json, os, sys, time

sys.path.insert(0, "/app/bench")
import harness  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


TAG = sys.argv[1]
TARGETS = arg("--targets", "/tmp/targets")
MAXTOK = int(arg("--max-tokens", 1024))
CHUNK = int(arg("--chunk", 128))
THINK = arg("--think", "0") == "1"
EFFORT = arg("--effort", "")
REPS = int(arg("--decode-reps", 2))
WANT = arg("--tasks", "gsm8k,code,chat").split(",")
os.makedirs(TARGETS, exist_ok=True)

gsm = pq.read_table(glob.glob("/app/models/quality-data/gsm8k/main/test-*.parquet")[0]).column("question").to_pylist()
chat = [json.loads(l)["prompt"] for l in open("/app/bench/prompts_real.jsonl")]
PROMPTS = {
    "gsm8k": gsm[0] + "\nSolve it step by step, showing every calculation, then give the final answer on its own line.",
    "code": ("Write one long Python function, parse_access_log(path), that reads an nginx access log in the "
             "combined format, parses every field with a compiled regular expression, skips and counts malformed "
             "lines, aggregates requests per status code, per hour and per client IP, computes p50/p95/p99 response "
             "sizes, and returns a dataclass with the results. Comment every block of the function in detail. "
             "Output only the code."),
    "chat": chat[1],
}
PROMPTS.update({f"real{i}": p for i, p in enumerate(chat)})

kw = {"enable_thinking": THINK}
if EFFORT:
    kw["reasoning_effort"] = EFFORT

SC = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
      "vllm:spec_decode_num_accepted_tokens_total")


def counters():
    m = harness.metrics(*SC)
    return tuple(m.get(k, 0.0) for k in SC)


def generate(ids, n):
    payload = {"prompt": ids, "max_tokens": n, "temperature": 0, "logprobs": 0,
               "return_tokens_as_token_ids": True, "stream": True,
               "stream_options": {"include_usage": True}}
    toks, t_first = [], None
    t0 = time.time()
    for ev in harness.stream("/v1/completions", payload):
        for ch in ev.get("choices", []):
            lp = ch.get("logprobs") or {}
            if lp.get("tokens"):
                if t_first is None:
                    t_first = time.time()
                toks.extend(lp["tokens"])
    t1 = time.time()
    assert all(t.startswith("token_id:") for t in toks), "server ignored return_tokens_as_token_ids"
    return [int(t[9:]) for t in toks], (t_first or t1) - t0, t1 - (t_first or t1)


r = harness.post("/tokenize", {"messages": [{"role": "user", "content": "warm up"}],
                               "add_generation_prompt": True, "chat_template_kwargs": kw})
for _ in range(2):
    generate(r["tokens"], 64)

print(f"# {TAG} think={THINK} effort={EFFORT or '<template default>'} max_tokens={MAXTOK} chunk={CHUNK}", flush=True)
out = []
for name in WANT:
    r = harness.post("/tokenize", {"messages": [{"role": "user", "content": PROMPTS[name]}],
                                   "add_generation_prompt": True, "chat_template_kwargs": kw})
    pids = r["tokens"]
    path = os.path.join(TARGETS, f"{name}.json")
    if not os.path.exists(path):
        ids, _, _ = generate(pids, MAXTOK)
        json.dump({"task": name, "captured_by": TAG, "chat_template_kwargs": kw, "prompt": PROMPTS[name],
                   "prompt_ids": pids, "ids": ids}, open(path, "w"))
        print(f"WACC {TAG} {name} CAPTURED n={len(ids)} -> {path}", flush=True)
    tgt = json.load(open(path))
    assert tgt["prompt_ids"] == pids, f"{name}: prompt ids differ from the frozen target's"
    T = tgt["ids"]
    # free-running C1 decode on the same prompt, counters diffed for free-running tok/step
    dec = []
    for _ in range(REPS):
        a = counters()
        ids, ttft, d = generate(pids, MAXTOK)
        b = counters()
        dec.append((len(ids) - 1) / d if d > 0 else 0.0)
        fr_tps = 1 + (b[2] - a[2]) / max(b[0] - a[0], 1)
    st = sl = ac = forced = 0.0
    clean = chunks = 0
    first_bad = None
    for off in range(0, len(T), CHUNK):
        want = T[off: off + CHUNK]
        a = counters()
        got, _, _ = generate(pids + T[:off], len(want))
        b = counters()
        chunks += 1
        if got == want:
            clean += 1
            st += b[0] - a[0]; sl += b[1] - a[1]; ac += b[2] - a[2]; forced += len(got)
        elif first_bad is None:
            first_bad = off + next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), len(got))
    tps = 1 + ac / max(st, 1)
    acc = ac / max(sl, 1)
    row = {"task": name, "prompt_tokens": len(pids), "target_tokens": len(T), "chunks_clean": clean,
           "chunks": chunks, "steps": st, "tok_per_step": tps, "accept_rate": acc,
           "freerun_tok_per_step_last": fr_tps, "decode_tok_s": dec, "first_bad": first_bad}
    out.append(row)
    print(f"WACC {TAG} {name:6s} prompt={len(pids)} target={len(T)} chunks={clean}/{chunks} steps={st:.0f} "
          f"tok/step={tps:.3f} accept={acc:.3f} freerun_tok/step={fr_tps:.3f} "
          f"decode_c1={'/'.join(f'{x:.1f}' for x in dec)} tok/s firstbad={first_bad}", flush=True)
json.dump(out, open(os.path.join(TARGETS, f"result_{TAG}.json"), "w"), indent=1)
