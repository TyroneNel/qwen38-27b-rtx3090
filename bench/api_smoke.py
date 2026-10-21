"""API feature smoke test against the running server: the request-level features a different
model runner could break (logprobs, n, stop, seeds, structured outputs, penalties, streaming,
thinking, prompt_logprobs, a 20k-token prompt, a streamed tool call). Prints PASS/FAIL per feature,
and exits 1 unless all 13 pass. It exits 2 when no request reaches a server.

  venv/bin/python bench/api_smoke.py          # server and key from bench/harness.py (PORT=18020 by default)
"""
import json, sys, time, urllib.request

import harness


def chat(msg, **kw):
    p = {"messages": [{"role": "user", "content": msg}], "max_tokens": 64,
         "chat_template_kwargs": {"enable_thinking": False}}
    p.update(kw)
    return harness.post("/v1/chat/completions", p, timeout=600)


results, unreached = [], []
def check(name, fn):
    try:
        ok, info = fn()
    except Exception as e:  # noqa
        ok, info = False, f"{type(e).__name__}: {str(e)[:200]}"
        if type(e) is urllib.error.URLError:  # its subclass HTTPError means a server answered
            unreached.append(name)
    results.append((name, ok, info)); print(("PASS " if ok else "FAIL ") + name + " — " + str(info)[:200], flush=True)


def t_greedy_det():
    a = chat("Nævn tre danske byer, kun navnene.", temperature=0)["choices"][0]["message"]["content"]
    b = chat("Nævn tre danske byer, kun navnene.", temperature=0)["choices"][0]["message"]["content"]
    return a == b, a[:80]
def t_seed():
    a = chat("Skriv et digt på fire linjer om regn.", temperature=1.0, seed=1234, max_tokens=48)["choices"][0]["message"]["content"]
    b = chat("Skriv et digt på fire linjer om regn.", temperature=1.0, seed=1234, max_tokens=48)["choices"][0]["message"]["content"]
    return a == b, a[:80]
def t_logprobs():
    r = chat("Hvad er 2+2?", temperature=0, logprobs=True, top_logprobs=5, max_tokens=8)
    lp = r["choices"][0]["logprobs"]["content"]
    return len(lp) > 0 and len(lp[0]["top_logprobs"]) == 5 and all(x["logprob"] <= 0 for x in lp), f"{len(lp)} tokens, first={lp[0]['token']!r}"
def t_n2():
    r = chat("Giv mig et tilfældigt tal mellem 1 og 1000.", temperature=1.0, n=2, max_tokens=16)
    return len(r["choices"]) == 2, [c["message"]["content"][:30] for c in r["choices"]]
def t_stop():
    r = chat("Tæl fra 1 til 20, ét tal per linje.", temperature=0, stop=["5"], max_tokens=64)
    c = r["choices"][0]
    return "5" not in c["message"]["content"] and c["finish_reason"] == "stop", c["message"]["content"][:40].replace("\n", "/")
def t_json_schema():
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"]}
    r = chat("Giv Danmarks hovedstad og dens indbyggertal.", temperature=0,
             response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}, max_tokens=64)
    d = json.loads(r["choices"][0]["message"]["content"])
    return isinstance(d.get("population"), int) and isinstance(d.get("city"), str), d
def t_min_tokens_penalty():
    r = chat("Sig hej.", temperature=0.7, min_tokens=30, presence_penalty=1.2, frequency_penalty=0.3, max_tokens=48)
    return r["usage"]["completion_tokens"] >= 30, r["usage"]
def t_stream():
    n = sum(1 for _ in harness.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": "Skriv to sætninger om vejret."}],
                                         "max_tokens": 48, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600))
    return n > 5, f"{n} chunks"
def t_thinking():
    r = chat("Hvad er 17*23? Svar kort.", temperature=0.6, max_tokens=512, chat_template_kwargs={"enable_thinking": True})
    m = r["choices"][0]["message"]
    return ("391" in (m.get("content") or "")) and bool(m.get("reasoning_content") or m.get("reasoning")), (m.get("content") or "")[:60]
def t_completions_echo_logprobs():
    r = harness.post("/v1/completions", {"prompt": "København er hovedstaden i", "max_tokens": 4,
                                         "temperature": 0, "echo": True, "logprobs": 1}, timeout=600)
    lp = r["choices"][0]["logprobs"]
    return len(lp["tokens"]) > 4 and lp["token_logprobs"][0] is None, lp["tokens"][:6]
def t_long_ctx_greedy():
    doc = ("Danmark er et land i Nordeuropa. " * 40 + "\n") * 60   # ~20k tokens
    t0 = time.perf_counter()
    r = chat(doc + "\nHvor mange gange står ordet Nordeuropa i teksten ovenfor, cirka? Svar kort.", temperature=0, max_tokens=32)
    return r["usage"]["prompt_tokens"] > 12000, f"prompt={r['usage']['prompt_tokens']} tok, {time.perf_counter()-t0:.1f}s, {r['choices'][0]['message']['content'][:50]!r}"
def t_thinking_budget():
    # vLLM 0.30's model runner enforces thinking_token_budget (v1/worker/gpu/sample/thinking_budget.py),
    # so the old expectation of a 400 is gone. A control without the budget shows the cap binds.
    def reasoning(**kw):
        r = chat("Hvad er 17*23? Svar kort.", temperature=0, max_tokens=512,
                 chat_template_kwargs={"enable_thinking": True}, **kw)
        details = r["usage"].get("completion_tokens_details") or {}
        return details.get("reasoning_tokens"), (r["choices"][0]["message"].get("content") or "").strip()
    free, _ = reasoning()
    capped, answer = reasoning(thinking_token_budget=16)
    ok = free is not None and capped is not None and capped <= 16 < free and bool(answer)
    return ok, f"reasoning {free} tokens unbudgeted, {capped} with budget 16, answer {answer[:20]!r}"
def t_tool_call_stream():
    # tool_choice=required: with VLLM_ENFORCE_STRICT_TOOL_CALLING (on by default) the qwen3_coder parser takes
    # required like auto under a structural tag, so the XML parser streams the call. Under a speculator a delta
    # can carry several tokens and split the markup (evals-6; gotcha 36).
    tools = [{"type": "function", "function": {"name": "get_time", "description": "Current time in an IANA timezone.",
              "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}}]
    calls, finish, frames = {}, None, 0
    for ev in harness.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hvad er klokken i København lige nu?"}],
                             "tools": tools, "tool_choice": "required", "max_tokens": 128, "temperature": 0, "stream": True,
                             "chat_template_kwargs": {"enable_thinking": False}}, timeout=600):
        frames += 1
        for ch in ev.get("choices", []):
            finish = ch.get("finish_reason") or finish
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                c["name"] += fn.get("name") or ""
                c["arguments"] += fn.get("arguments") or ""
    args = [json.loads(c["arguments"]) for c in calls.values()]  # a reassembly that does not parse raises: FAIL
    ok = len(calls) == 1 and calls[min(calls)]["name"] == "get_time" and isinstance(args[0], dict) and bool(args[0].get("timezone"))
    return ok, f"{frames} frames, finish={finish}, calls={[(c['name'], a) for c, a in zip(calls.values(), args)]}"

for name, fn in [("greedy determinism", t_greedy_det), ("seeded sampling determinism", t_seed), ("logprobs/top_logprobs", t_logprobs),
                 ("n=2", t_n2), ("stop strings", t_stop), ("json_schema structured output", t_json_schema),
                 ("min_tokens + penalties", t_min_tokens_penalty), ("streaming", t_stream), ("thinking mode", t_thinking),
                 ("completions echo+logprobs (prompt_logprobs)", t_completions_echo_logprobs), ("20k-token prompt", t_long_ctx_greedy),
                 ("thinking_token_budget enforced", t_thinking_budget), ("streamed tool call", t_tool_call_stream)]:
    check(name, fn)
print("SUMMARY", sum(1 for _, ok, _ in results if ok), "/", len(results), "passed")
if len(unreached) == len(results):
    print(f"INVALID: no server answered at {harness.base_url()}, so there is no verdict.")
    sys.exit(2)
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
