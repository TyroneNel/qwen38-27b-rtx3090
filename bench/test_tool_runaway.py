#!/usr/bin/env python3
"""CPU-only check for tool_runaway.py against a stub server that answers each seed with a canned forced-call
response: a clean call, a call that decodes newlines inside <function=get_time> until max_tokens, a call that
closes after 40 whitespace tokens (finish tool_calls, still a runaway), and a call with invented arguments.
The prompt picks the stub's script, so each run checks one exit code: 0 clean, 1 any runaway, 3 invalid only,
2 no server.

  python bench/test_tool_runaway.py
"""
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

SCRIPT = Path(__file__).resolve().parent / "tool_runaway.py"
seen = []

CALL = ["<tool_call>", "\n", "<function=", "get_time", ">\n", "\n", "</function>", "\n", "</tool_call>"]


def response(kind, seed):
    if kind == "ok":
        toks, args, finish = CALL, "{}", "tool_calls"
    elif kind == "length":
        toks, args, finish = CALL[:5] + ["\n\n"] * 2043, None, "length"
    elif kind == "ws40":
        toks, args, finish = CALL[:5] + ["  "] * 39 + CALL[5:], "{}", "tool_calls"  # 39 + the "\n" before </function>
    else:  # invented arguments
        toks, args, finish = CALL, '{"hour": "14"}', "tool_calls"
    msg = {"role": "assistant", "content": None if args else "".join(toks), "tool_calls": [] if args is None else [
        {"id": f"c{seed}", "type": "function", "function": {"name": "get_time", "arguments": args}}]}
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish,
                         "logprobs": {"content": [{"token": t, "logprob": -0.1} for t in toks]}}],
            "usage": {"completion_tokens": len(toks)}}


SCRIPTS = {"clean": ["ok"], "mixed": ["ok", "length", "ws40", "bad"], "invalid": ["ok", "bad"]}


class Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append(p)
        script = SCRIPTS[p["messages"][0]["content"]]
        body = json.dumps(response(script[p["seed"] % len(script)], p["seed"])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(port, *argv):
    env = {**os.environ, "VLLM_API": f"http://127.0.0.1:{port}", "OPENAI_API_KEY": "stub"}
    env.pop("VLLM_MODEL", None)
    p = subprocess.run([sys.executable, str(SCRIPT), *argv], env=env, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    fails = []
    def check(name, ok, out=""):
        print(("PASS " if ok else "FAIL ") + name, flush=True)
        if not ok:
            fails.append(name)
            print(out)

    code, out = run(port, "--prompt", "clean", "--seeds", "5")
    check("clean: exit 0, 0 runaways, 5/5 valid", code == 0 and "runaways 0/5" in out and "valid calls 5/5" in out, out)
    p = seen[-1]
    check("payload: required, max_tokens 2048, logprobs, seed 4, no temperature or template kwargs",
          p["tool_choice"] == "required" and p["max_tokens"] == 2048 and p["logprobs"] is True and p["seed"] == 4
          and "temperature" not in p and "chat_template_kwargs" not in p
          and [t["function"]["name"] for t in p["tools"]] == ["get_time"], json.dumps(p)[:300])

    code, out = run(port, "--prompt", "mixed", "--seeds", "8", "--concurrency", "3")
    check("mixed: exit 1, seeds 1 2 5 6 run away, 4/8 valid",
          code == 1 and "runaways 4/8 [1, 2, 5, 6]" in out and "valid calls 4/8" in out, out)
    check("mixed: ws40 seed measured inside the call", "seed   2 RUNAWAY finish=tool_calls" in out
          and "ws_in_call=40" in out, out)

    code, out = run(port, "--prompt", "mixed", "--seeds", "4", "--ws-tokens", "64")
    check("--ws-tokens 64: the 40-token seed is no longer a runaway", code == 1 and "runaways 1/4 [1]" in out, out)

    code, out = run(port, "--prompt", "invalid", "--seeds", "4", "--choice", "named", "--tools", "2",
                    "--thinking", "off", "--temperature", "0.7")
    p = seen[-1]
    check("invalid only: exit 3", code == 3 and "runaways 0/4" in out and "valid calls 2/4" in out, out)
    check("payload: named get_time, two tools, thinking off, temperature 0.7",
          p["tool_choice"] == {"type": "function", "function": {"name": "get_time"}} and len(p["tools"]) == 2
          and p["chat_template_kwargs"] == {"enable_thinking": False} and p["temperature"] == 0.7, json.dumps(p)[:300])

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    code, out = run(dead, "--seeds", "2")
    check("no server: exit 2", code == 2, out)

    srv.shutdown()
    print(f"{len(fails)} failed" if fails else "all passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
