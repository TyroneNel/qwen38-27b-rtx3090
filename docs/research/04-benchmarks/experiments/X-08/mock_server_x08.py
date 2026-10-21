#!/usr/bin/env python3
"""X-08 COPY of harness/mock_server.py (original untouched). Additions: runner-up token id = id+1 (so --perturb flips to
the runner-up, a faithful near-tie), --tie-margin, --perturb-from (every position >= IDX differs, like a corrupted
forward), POST /tokenize, POST /v1/completions with token-id prompts (the replay endpoints, always the clean stream).
Original docstring follows.
Tiny OpenAI-compatible stand-in for the vLLM server, for the harness's CPU self-check (selfcheck.sh). Stdlib only.
It serves /health, /v1/models, /metrics (the vllm: counters the bench scripts read), /v1/chat/completions (stream and
not, with logprobs and token ids). Output is a deterministic function of the prompt (greedy-like), so a "spec" and
a "non-spec" mock produce identical tokens at different speeds, which is what greedy_equal.py must see.

  mock_server.py [--port 18999] [--rate 400] [--spec 7 --tps 3.5] [--perturb IDX] [--tie-at IDX]

--rate tok/s non-spec; --spec K emulates K drafts per step with --tps tokens/step (counters move like vLLM's).
--perturb IDX changes the token at IDX (a fake divergence); --tie-at IDX sets the top-2 margin there to 0.01.
It measures nothing real: numbers from it prove the plumbing, never the stack.
"""
import argparse
import hashlib
import http.server
import json
import random
import threading
import time

A = None
LOCK = threading.Lock()
M = dict(gen=0.0, drafts=0.0, draft_tokens=0.0, accepted=0.0, pos={}, iters=0.0, iter_tok=0.0, running=0)


PROMPTS = {}


def tokens(prompt, n, clean=False):
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:12], 16)
    ri, rm = random.Random(seed), random.Random(seed + 1)  # separate streams: position i is independent of n
    ids = [ri.randrange(1000, 50000) for _ in range(n)]
    margins = [round(0.5 + rm.random() * 3, 4) for _ in range(n)]
    alts = [t + 1 for t in ids]
    if A.tie_at is not None and A.tie_at < n:
        margins[A.tie_at] = A.tie_margin
    if clean:
        return ids, margins, alts
    for i in range(n):
        if (A.perturb is not None and i == A.perturb) or (A.perturb_from is not None and i >= A.perturb_from):
            ids[i], alts[i] = alts[i], ids[i]  # runner-up wins; the old top-1 becomes the runner-up
    return ids, margins, alts


def steps(n):
    """Token counts per forward step: 1 each without spec, 1 + accepted with it (mean ~A.tps)."""
    out, rng = [], random.Random(n)
    while sum(out) < n:
        if not A.spec:
            out.append(1); continue
        acc = 0
        while acc < A.spec and rng.random() < 1 - 1 / A.tps:
            acc += 1
        out.append(1 + acc)
        with LOCK:
            M["drafts"] += 1; M["draft_tokens"] += A.spec; M["accepted"] += acc
            for p in range(acc):
                M["pos"][p] = M["pos"].get(p, 0) + 1
    return out


class H(http.server.BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self.send(200, b"")
        elif self.path == "/v1/models":
            self.send(200, {"data": [{"id": "mock"}]})
        elif self.path == "/metrics":
            with LOCK:
                L = ['vllm:generation_tokens_total{engine="0"} %s' % M["gen"],
                     'vllm:spec_decode_num_drafts_total{engine="0"} %s' % M["drafts"],
                     'vllm:spec_decode_num_draft_tokens_total{engine="0"} %s' % M["draft_tokens"],
                     'vllm:spec_decode_num_accepted_tokens_total{engine="0"} %s' % M["accepted"],
                     'vllm:iteration_tokens_total_sum{engine="0"} %s' % M["iter_tok"],
                     'vllm:iteration_tokens_total_count{engine="0"} %s' % M["iters"],
                     'vllm:num_preemptions_total{engine="0"} 0', 'vllm:kv_cache_usage_perc{engine="0"} 0.1',
                     'vllm:num_requests_running{engine="0"} %s' % M["running"], 'vllm:num_requests_waiting{engine="0"} 0']
                L += ['vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="%d"} %s' % (p, v)
                      for p, v in sorted(M["pos"].items())]
            self.send(200, "\n".join(L).encode() + b"\n", "text/plain")
        else:
            self.send(404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":  # one marker id stands for the whole templated prompt
            text = req["messages"][0]["content"]
            marker = 900000 + int(hashlib.sha256(text.encode()).hexdigest()[:5], 16)
            PROMPTS[marker] = text
            return self.send(200, {"tokens": [marker], "count": 1})
        if self.path == "/v1/completions":  # next-token top-n after [marker] + generated ids
            p = req["prompt"]
            k = len(p) - 1
            ids, margins, alts = tokens(PROMPTS[p[0]], k + 1, clean=True)
            if p[1:] != ids[:k]:
                return self.send(400, {"error": "prefix is not this server's clean greedy stream"})
            top = {f"token_id:{ids[k]}": -0.1, f"token_id:{alts[k]}": round(-0.1 - margins[k], 4)}
            for j in range(2, int(req.get("logprobs") or 2)):
                top[f"token_id:{ids[k] + 10 + j}"] = round(-0.1 - margins[k] - j, 4)
            return self.send(200, {"choices": [{"text": "x", "logprobs": {"top_logprobs": [top]}}]})
        prompt = req["messages"][0]["content"]
        n = int(req.get("max_tokens", 16))
        ids, margins, alts = tokens(prompt, n)
        ptok = len(prompt.split())
        with LOCK:
            M["running"] += 1
        try:
            if not req.get("stream"):
                for sz in steps(n):
                    time.sleep(sz / A.rate if not A.spec else 1 / A.rate)
                content = [{"token": f"token_id:{t}", "logprob": -0.1,
                            "top_logprobs": [{"token": f"token_id:{t}", "logprob": -0.1},
                                             {"token": f"token_id:{al}", "logprob": round(-0.1 - m, 4)}] +
                                            [{"token": f"token_id:{t + 10 + j}", "logprob": round(-0.1 - m - j, 4)}
                                             for j in range(2, int(req.get("top_logprobs") or 2))]}
                           for t, m, al in zip(ids, margins, alts)]
                ch = {"message": {"content": " ".join(f"w{t}" for t in ids)}, "finish_reason": "length"}
                if req.get("logprobs"):
                    ch["logprobs"] = {"content": content}
                self.send(200, {"choices": [ch], "usage": {"prompt_tokens": ptok, "completion_tokens": n}})
                return
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            i = 0
            time.sleep(0.002 * (1 + ptok / 2000))  # "prefill"
            for sz in steps(n - 1):  # review: n-1 content tokens, then the EOS (token n) alone in the last step
                time.sleep(1 / A.rate if A.spec else sz / A.rate)
                text = "".join(f" w{t}" for t in ids[i:i + sz]); i += sz
                with LOCK:
                    M["gen"] += sz; M["iters"] += 1; M["iter_tok"] += sz
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": text}}]}).encode() + b"\n\n")
                self.wfile.flush()
            time.sleep(1 / A.rate)  # review: the EOS-only last step, streamed as vLLM does: empty delta + finish_reason
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode() + b"\n\n")
            u = {"prompt_tokens": ptok, "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": 0}}
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": u}).encode() + b"\n\ndata: [DONE]\n\n")
        finally:
            with LOCK:
                M["running"] -= 1

    def log_message(self, *a):
        pass


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--port", type=int, default=18999); ap.add_argument("--rate", type=float, default=400)
ap.add_argument("--spec", type=int, default=0); ap.add_argument("--tps", type=float, default=3.5)
ap.add_argument("--perturb", type=int); ap.add_argument("--tie-at", type=int)
ap.add_argument("--tie-margin", type=float, default=0.005); ap.add_argument("--perturb-from", type=int)
A = ap.parse_args()
http.server.ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
