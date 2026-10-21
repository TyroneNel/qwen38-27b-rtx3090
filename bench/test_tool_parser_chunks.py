#!/usr/bin/env python3
"""Streamed tool calls with multi-token deltas against the one-shot parse (evals-6).

Upstream's qwen3_coder test streams one token per delta. Under speculative decoding the server emits up to K+1
tokens per step, so a delta can span several tokens and split `<tool_call>`, `<function=`, `<parameter=` and
`</parameter>` anywhere. This feeds five canned model outputs (1-3 calls, nested JSON arguments, a value holding
`<` and `>`, one thinking-on output) through the parser the server builds (`--tool-call-parser qwen3_coder
--reasoning-parser qwen3 --tool-strict-level parameter`, via the serving layer's `Parser.parse_delta`), cut at
random sizes of 1-8 tokens over 200 seeds each, and asserts the reassembled `tool_calls` (names, and arguments
as JSON) equal what `Parser.parse` gives for the whole output. A 1-token-per-delta stream runs first per case,
so a failure says whether streaming breaks at all or only under multi-token deltas. Content that differs is
counted and printed, not failed.

Needs the image's vLLM (no GPU) and the model's tokenizer (the parser matches its markers by token id):

  docker run --rm --entrypoint /app/venv/bin/python -v /mnt/g/dev/HyperQwen/bench:/app/bench \\
    -v /mnt/g/models:/app/models:ro ghcr.io/syv-ai/hyperqwen:latest \\
    /app/bench/test_tool_parser_chunks.py [--tokenizer DIR] [--seeds 200] [--max-chunk 8] [--out f.jsonl]

--tokenizer defaults to /app/models/Qwen3.8-27B-W4A16-AutoRound-fast (TOKENIZER env overrides).
Exit: 0 every stream matches, 1 a stream differs, 2 a case's one-shot parse is not the call list it was
written to produce (no verdict), 3 the negative control (a stream with one argument token dropped) still
matched, so the comparison cannot see a broken stream.
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

TOOLS = [
    {"type": "function", "function": {"name": "get_time", "description": "Current time in a timezone.",
     "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}},
    {"type": "function", "function": {"name": "search_docs", "description": "Search the documentation.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"},
         "filters": {"type": "object", "properties": {"city": {"type": "string"},
                                                      "tags": {"type": "array", "items": {"type": "string"}}}},
         "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
]


def call(name, **params):
    """The Qwen3 XML the model emits for one call: one `<parameter=k>` block per argument, objects as JSON."""
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n"
                   for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


CODE = 'def clamp(a, lo, hi):\n    if a < lo:\n        return lo\n    return hi if a > hi else a\n# <b>not a tag</b>\n'
# (name, thinking, model output, the function names the one-shot parse must find)
CASES = [
    ("one_call", False, call("get_time", timezone="Europe/Copenhagen"), ["get_time"]),
    ("content_then_nested", False,
     "Jeg slår det op i dokumentationen.\n\n" + call(
         "search_docs", query="prefix cache rød grød", filters={"city": "Aarhus", "tags": ["kv", "spec"],
                                                                "nested": {"deep": [1, {"x": None}]}}, limit=5),
     ["search_docs"]),
    ("two_calls", False,
     call("get_time", timezone="UTC") + "\n" + call("search_docs", query="dflash2 acceptance",
                                                    filters={"tags": []}, limit=10),
     ["get_time", "search_docs"]),
    ("three_calls_code", False,
     call("write_file", path="src/clamp.py", content=CODE) + "\n"
     + call("search_docs", query="<tool_call> in a value", filters={"city": "Odense", "tags": ["a<b", "c>d"]})
     + "\n" + call("get_time", timezone="America/New_York"),
     ["write_file", "search_docs", "get_time"]),
    ("thinking_on", True,
     "The user wants the time in Tokyo and a doc search. I will call both tools.\n</think>\n\n"
     + call("get_time", timezone="Asia/Tokyo") + "\n"
     + call("search_docs", query="int8 activations", filters={"tags": ["quant", "w4a8"]}, limit=3),
     ["get_time", "search_docs"]),
]


def load(tokenizer_dir):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.parser_manager import ParserManager
    from vllm.tokenizers import get_tokenizer
    tok = get_tokenizer(tokenizer_dir)
    parser_cls = ParserManager.get_parser(tool_parser_name="qwen3_coder", reasoning_parser_name="qwen3",
                                          enable_auto_tools=True, tool_strict_level="parameter")
    return tok, parser_cls, ChatCompletionRequest


class Case:
    def __init__(self, env, name, thinking, text, want):
        tok, self.parser_cls, Request = env
        self.tok, self.name, self.text, self.want = tok, name, text, want
        self.ctk = {"enable_thinking": thinking}
        messages = [{"role": "user", "content": "Use the tools."}]
        self.request = Request(model="m", messages=messages, tools=TOOLS, tool_choice="auto",
                               chat_template_kwargs=self.ctk)
        prompt = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=thinking)
        self.prompt_ids = tok.encode(prompt, add_special_tokens=False)
        self.ids = tok.encode(text, add_special_tokens=False)

    def parser(self):
        p = self.parser_cls(self.tok, self.request.tools, chat_template_kwargs=self.ctk, model_config=None)
        self.request = p.adjust_request(self.request)
        return p

    def one_shot(self):
        _, content, calls = self.parser().parse(self.text, self.request, enable_auto_tools=True,
                                                model_output_token_ids=self.ids)
        return [(c.name, json.loads(c.arguments)) for c in calls or []], (content or "")

    def bounds(self, sizes):
        """Delta end offsets from a size sequence. A delta whose decoded text ends inside a UTF-8 sequence takes
        the next token too, as vLLM's incremental detokenizer holds such text back."""
        ends, pos, it = [], 0, iter(sizes)
        while pos < len(self.ids):
            end = min(len(self.ids), pos + next(it))
            while end < len(self.ids) and self.tok.decode(self.ids[:end], skip_special_tokens=False).endswith("\ufffd"):
                end += 1
            ends.append(end)
            pos = end
        return ends

    def stream(self, ends, drop=None):
        """Feed the deltas ending at `ends` through parse_delta; `drop` removes the token at that index (the
        negative control). Returns the reassembled calls and content."""
        p, ids = self.parser(), list(self.ids)
        if drop is not None:
            del ids[drop]
            ends = [e - (e > drop) for e in ends]
            ends = sorted(set(e for e in ends if e > 0)) or [len(ids)]
        calls, content, prev_text, pos = {}, [], "", 0
        for end in ends:
            text = self.tok.decode(ids[:end], skip_special_tokens=False)
            delta_text, prev_text = text[len(prev_text):], text
            d = p.parse_delta(delta_text, ids[pos:end], self.request, self.prompt_ids, finished=end == len(ids))
            pos = end
            if d is None:
                continue
            content.append(d.content or "")
            for tc in d.tool_calls or []:
                c = calls.setdefault(tc.index, {"name": None, "args": ""})
                if tc.function is not None:
                    if tc.function.name:
                        if c["name"] not in (None, tc.function.name):
                            raise AssertionError(f"call {tc.index} renamed {c['name']!r} -> {tc.function.name!r}")
                        c["name"] = tc.function.name
                    c["args"] += tc.function.arguments or ""
        out = []
        for i in sorted(calls):
            try:
                args = json.loads(calls[i]["args"])
            except ValueError:
                args = {"<unparseable>": calls[i]["args"]}
            out.append((calls[i]["name"], args))
        return out, "".join(content)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER", "/app/models/Qwen3.8-27B-W4A16-AutoRound-fast"))
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--max-chunk", type=int, default=8)
    ap.add_argument("--out", help="JSONL: one row per failing stream, then one per case")
    a = ap.parse_args()
    env = load(a.tokenizer)
    rows, fails, invalid = [], 0, 0
    cases = [Case(env, *c) for c in CASES]
    for case in cases:
        want, want_content = case.one_shot()
        if [n for n, _ in want] != case.want:
            print(f"INVALID {case.name}: one-shot parse found {[n for n, _ in want]}, written for {case.want}")
            invalid += 1
            continue
        bad, content_diff, first = 0, 0, None
        runs = [("1tok", case.bounds(iter(lambda: 1, 0)))]
        for seed in range(a.seeds):
            rng = random.Random(seed)
            runs.append((f"seed{seed}", case.bounds(iter(lambda: rng.randint(1, a.max_chunk), 0))))
        for label, ends in runs:
            try:
                got, content = case.stream(ends)
            except Exception as e:  # noqa: BLE001 - a parser crash mid-stream is a failing stream
                got, content = [("<exception>", f"{type(e).__name__}: {e}")], ""
            if content.strip() != want_content.strip():
                content_diff += 1
            if got != want:
                bad += 1
                sizes = [e - s for s, e in zip([0] + ends, ends)]
                row = {"case": case.name, "run": label, "sizes": sizes, "got": got, "want": want}
                rows.append(row)
                first = first or row
        fails += bad
        print(f"{'FAIL' if bad else 'PASS'} {case.name}: {len(case.ids)} tokens, {len(want)} call(s), "
              f"{bad}/{len(runs)} streams differ, {content_diff} with different content", flush=True)
        if first:
            print(f"  first: {first['run']} sizes={first['sizes'][:24]}...\n  got  {json.dumps(first['got'])[:300]}"
                  f"\n  want {json.dumps(first['want'])[:300]}")
        rows.append({"case": case.name, "tokens": len(case.ids), "calls": len(want), "streams": len(runs),
                     "differ": bad, "content_differ": content_diff})

    # Negative control: drop one token from inside the first call's first argument value; the reassembled
    # arguments must then differ from the one-shot parse of the intact output.
    nc = cases[0]
    value_tok = len(nc.tok.encode(nc.text[:nc.text.index("Copenhagen") + 4], add_special_tokens=False)) - 1
    got, _ = nc.stream(nc.bounds(iter(lambda: 1, 0)), drop=value_tok)
    control_ok = got != nc.one_shot()[0]
    print(f"negative control (token {value_tok} dropped from {nc.name}): {'detected' if control_ok else 'NOT detected'} "
          f"-> {json.dumps(got)[:160]}")

    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_streams = sum(r.get("streams", 0) for r in rows)
    verdict = "NEGATIVE-CONTROL-FAILED" if not control_ok else "INVALID" if invalid else "FAIL" if fails else "PASS"
    print(f"SUMMARY tool_parser_chunks {verdict} cases={len(cases) - invalid}/{len(cases)} streams={n_streams} "
          f"differ={fails} max_chunk={a.max_chunk} seeds={a.seeds}")
    sys.exit(3 if not control_ok else 2 if invalid else 1 if fails else 0)


if __name__ == "__main__":
    main()
