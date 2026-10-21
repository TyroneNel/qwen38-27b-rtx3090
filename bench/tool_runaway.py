#!/usr/bin/env python3
"""Forced tool-call runaway test (runtime-3; docs/vllm-0.31.md, "What is still unproven"): a forced call to a
tool with no parameters, get_time, once per seed 0..N-1, max_tokens 2048. On 0.31.0 a call sometimes keeps
decoding whitespace inside the qwen3_coder XML after <function=get_time> until max_tokens (2 of 20 seeds batch,
1 of 20 single-user, 5 of 20 on 0.30.0).

A seed is a RUNAWAY when finish_reason is "length", or the output holds more than --ws-tokens whitespace-only
tokens in a row inside the call (between <function=...> and </function>, or to the end when the call never
closes) or after the last </tool_call>. The raw output is rebuilt from the per-token logprobs, which cover the
tool-call text the parser strips. A seed is a VALID CALL when finish_reason is "tool_calls" or "stop" and the
response carries one or more tool calls, each naming an offered tool with arguments that parse to a JSON
object meeting its schema (get_time: exactly {}), and with --choice named, every call is get_time.

Exit 0 when no seed ran away and every call was valid, 1 when any seed ran away, 3 when none ran away but
a call was invalid, 2 when no request reached a server.

  venv/bin/python bench/tool_runaway.py                            # 40 seeds, tool_choice=required, 1 tool
  venv/bin/python bench/tool_runaway.py --tools 2 --choice named   # get_time plus get_weather offered, named
  venv/bin/python bench/tool_runaway.py --json out.jsonl           # one line per seed, plus the summary
Server and key come from bench/harness.py (VLLM_API, else PORT=18020).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import urllib.error

import harness

GET_TIME = {"type": "function", "function": {"name": "get_time", "description": "Get the current local time.",
            "parameters": {"type": "object", "properties": {}}}}
GET_WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather in a city.",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def valid_args(name, args):
    if name == "get_time":
        return args == {}
    return name == "get_weather" and set(args) == {"city"} and isinstance(args["city"], str) and bool(args["city"])


def ws_runs(tokens):
    """(longest run of whitespace-only tokens inside a call, whitespace-only tokens after the last </tool_call>)."""
    inside = run = best = 0
    text = ""
    for tok in tokens:
        text += tok
        opened = text.rfind("<function=")
        closed = text.rfind("</function>")
        inside = opened > closed and ">" in text[opened:]
        if inside and tok and not tok.strip():
            run += 1
            best = max(best, run)
        elif tok.strip():
            run = 0
    tail = 0
    end = text.rfind("</tool_call>")
    if end >= 0:
        for tok in reversed(tokens):
            if tok.strip():
                break
            tail += 1
    return best, tail


def one(seed, args, tools):
    payload = {"messages": [{"role": "user", "content": args.prompt}], "tools": tools, "max_tokens": args.max_tokens,
               "seed": seed, "logprobs": True,
               "tool_choice": {"type": "function", "function": {"name": "get_time"}} if args.choice == "named"
               else "required"}
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.thinking != "default":
        payload["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    r = harness.post("/v1/chat/completions", payload, timeout=args.timeout)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    lp = (ch.get("logprobs") or {}).get("content") or []
    tokens = [t.get("token", "") for t in lp]
    in_call, tail = ws_runs(tokens) if tokens else ws_runs(list(msg.get("content") or ""))
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            a = json.loads(fn.get("arguments") or "")
        except ValueError:
            a = None
        calls.append((fn.get("name"), a))
    offered = {t["function"]["name"] for t in tools}
    valid = (ch.get("finish_reason") in ("tool_calls", "stop") and bool(calls)
             and all(n in offered and isinstance(a, dict) and valid_args(n, a) for n, a in calls)
             and (args.choice != "named" or all(n == "get_time" for n, _ in calls)))
    finish = ch.get("finish_reason")
    runaway = finish == "length" or in_call > args.ws_tokens or tail > args.ws_tokens
    raw = "".join(tokens) if tokens else (msg.get("content") or "")
    # Where the output ended: still reasoning or before any call (a length finish here is a thinking budget, not
    # the runaway this item is about), inside an open call, or after the last closed call.
    stage = ("in_call" if raw.rfind("<function=") > raw.rfind("</function>") else
             "after_call" if "</tool_call>" in raw else "before_call")
    return {"seed": seed, "finish": finish, "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "ws_in_call": in_call, "ws_tail": tail, "stage": stage, "logprobs": bool(tokens), "calls": calls,
            "valid": valid, "runaway": runaway, "raw_tail": repr(raw[-80:])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=40, help="seeds 0..N-1 (docs/vllm-0.31.md used 20)")
    ap.add_argument("--tools", type=int, choices=(1, 2), default=1, help="1: get_time; 2: get_time and get_weather")
    ap.add_argument("--choice", choices=("required", "named"), default="required",
                    help='tool_choice "required", or named {"function": {"name": "get_time"}}')
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--ws-tokens", type=int, default=32,
                    help="a whitespace-only token run longer than this inside the call or after it is a runaway")
    ap.add_argument("--temperature", type=float, default=None, help="unset: the server's generation-config default")
    ap.add_argument("--thinking", choices=("default", "on", "off"), default="default",
                    help="chat_template_kwargs.enable_thinking; default sends none (the launcher's default)")
    ap.add_argument("--prompt", default="What time is it right now?")
    ap.add_argument("--concurrency", type=int, default=1, help="seeds in flight at once (1 = one after another)")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--json", help="write one JSON line per seed, then the summary, to this file")
    args = ap.parse_args()
    tools = [GET_TIME] if args.tools == 1 else [GET_TIME, GET_WEATHER]

    rows, unreached = [], 0
    def run(seed):
        try:
            return one(seed, args, tools)
        except urllib.error.HTTPError as e:
            return {"seed": seed, "error": f"HTTP {e.code}: {e.read()[:200]!r}", "runaway": False, "valid": False}
        except urllib.error.URLError as e:
            return {"seed": seed, "error": f"unreachable: {e.reason}", "runaway": False, "valid": False,
                    "unreached": True}
        except Exception as e:  # noqa: BLE001 - a malformed response is a failed seed, not a crash
            return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:200]}", "runaway": False, "valid": False}
    with ThreadPoolExecutor(max(1, args.concurrency)) as pool:
        for row in pool.map(run, range(args.seeds)):
            rows.append(row)
            unreached += bool(row.get("unreached"))
            if "error" in row:
                print(f"seed {row['seed']:3d} ERROR {row['error']}", flush=True)
                continue
            tag = "RUNAWAY" if row["runaway"] else ("ok" if row["valid"] else "INVALID")
            print(f"seed {row['seed']:3d} {tag:7s} finish={row['finish']} tokens={row['completion_tokens']} "
                  f"ws_in_call={row['ws_in_call']} ws_tail={row['ws_tail']} calls={row['calls']}"
                  + (f" stage={row['stage']} tail={row['raw_tail']}" if row["runaway"] else ""), flush=True)

    n = len(rows)
    runaways = [r["seed"] for r in rows if r["runaway"]]
    valid = sum(r["valid"] for r in rows)
    no_lp = sum(1 for r in rows if "error" not in r and not r["logprobs"])
    stages = {s: sum(1 for r in rows if r["runaway"] and r.get("stage") == s)
              for s in ("before_call", "in_call", "after_call")}
    summary = {"seeds": n, "tools": args.tools, "choice": args.choice, "max_tokens": args.max_tokens,
               "runaways": len(runaways), "runaway_seeds": runaways, "runaway_stages": stages, "valid_calls": valid,
               "valid_rate": round(valid / n, 4) if n else 0.0, "errors": sum("error" in r for r in rows),
               "no_logprobs": no_lp}
    print(f"\nrunaways {len(runaways)}/{n} {runaways} {stages}  valid calls {valid}/{n}  errors {summary['errors']}"
          + (f"  ({no_lp} responses without logprobs: whitespace measured on content only)" if no_lp else ""))
    if args.json:
        with open(args.json, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")
    if n and unreached == n:
        sys.exit(2)
    sys.exit(1 if runaways else 0 if valid == n else 3)


if __name__ == "__main__":
    main()
