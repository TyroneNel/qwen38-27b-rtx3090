#!/usr/bin/env python3
"""CPU-only check for docs/research/deepinfer/experiments/patches/runtime3-tool-call-bound.patch (runtime-3).
Runs inside the image's venv with the patch applied; bench/test_tool_call_bound.sh does that in a throwaway
container without a GPU. It builds the qwen_3_coder structural tag the way the server does for a forced call
(--tool-strict-level parameter) and checks, with the knob off, that the tag is byte-identical to xgrammar's
builtin and still accepts the runaway (2,000 newlines inside the call), and with VLLM_TOOL_CALL_BOUND=1 that:
whitespace inside the call stops at VLLM_TOOL_CALL_MAX_WS; nothing but <= MAX_WS whitespace may follow a
required call; parallel calls joined by "\\n", indented code in a string parameter and a pretty-printed JSON
argument still match; tool_choice=auto is unchanged; and on a compiled grammar the matcher accepts EOS after the
call and only EOS once the whitespace tail is used up.

  /app/venv/bin/python bench/test_tool_call_bound.py            # knob read from the environment
"""
import json
import os
import sys

import xgrammar as xgr
from xgrammar import get_model_structural_tag as xg_tag
from xgrammar.testing import _is_grammar_accept_string as accepts

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

ON = os.environ.get("VLLM_TOOL_CALL_BOUND") == "1"
MAX_WS = int(os.environ.get("VLLM_TOOL_CALL_MAX_WS", "16"))

GET_TIME = {"type": "function", "function": {"name": "get_time", "description": "Current time.",
            "parameters": {"type": "object", "properties": {}}}}
WRITE = {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                     "opts": {"type": "object"}}, "required": ["path", "content"]}}}
NAMED = {"type": "function", "function": {"name": "get_time"}}

CALL = "<tool_call>\n<function=get_time>\n\n</function>\n</tool_call>"
def inner(ws):
    return "<tool_call>\n<function=get_time>\n" + ws + "\n</function>\n</tool_call>"
CODE = "def f(x):\n        if x:\n                return  1\n\n\n\n    return 2\n"
def wcall(opts):
    return ("<tool_call>\n<function=write_file>\n<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=content>\n"
            + CODE + "\n</parameter>\n<parameter=opts>\n" + opts + "\n</parameter>\n</function>\n</tool_call>")
CODE_CALL = wcall('{"a": {"b": 1}}')                                  # 17 spaces of indentation inside a string
PRETTY_CALL = wcall('{\n    "a": {\n        "b": 1\n    }\n}')        # a 9-character whitespace run in JSON

fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f" — {info}" if info else ""), flush=True)
    if not ok:
        fails.append(name)


def server_tag(tools, choice):
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], tools=tools, tool_choice=choice)
    return get_model_structural_tag("qwen_3_coder", req.tools, req.tool_choice, False,
                                    strict_level=ToolStrictLevel.PARAMETER)


def grammar(tag):
    return xgr.Grammar.from_structural_tag(json.dumps(tag.model_dump()))


def strict(tool):
    return {"type": "function", "function": {**tool["function"], "strict": True}}


print(f"VLLM_TOOL_CALL_BOUND={'1' if ON else 'off'} VLLM_TOOL_CALL_MAX_WS={MAX_WS}")
for label, tools, choice in [("required/1", [GET_TIME], "required"), ("required/2", [GET_TIME, WRITE], "required"),
                             ("named", [GET_TIME, WRITE], NAMED), ("auto", [GET_TIME, WRITE], "auto")]:
    tag = server_tag(tools, choice)
    g = grammar(tag)
    builtin = xg_tag("qwen_3_coder", tools=[strict(t) for t in tools], tool_choice=choice, reasoning=False)
    same = json.dumps(tag.model_dump()) == json.dumps(builtin.model_dump())
    check(f"{label}: one valid call accepted", accepts(g, CALL))
    if label in ("required/2", "auto"):
        check(f"{label}: indented code in a string parameter accepted", accepts(g, CODE_CALL))
        # The cap applies to JSON whitespace too, so a pretty-printed argument needs MAX_WS >= 9 (the documented
        # cost of the knob: the grammar then forces the model to a compact form, it does not reject the call).
        want = not ON or label == "auto" or MAX_WS >= 9
        check(f"{label}: pretty-printed JSON argument {'accepted' if want else 'rejected'}",
              accepts(g, PRETTY_CALL) == want)
    if not ON or label == "auto":
        check(f"{label}: tag identical to xgrammar's builtin", same)
        if label != "auto":
            check(f"{label}: runaway (2000 newlines in the call) still accepted", accepts(g, inner("\n" * 2000)))
        continue
    check(f"{label}: tag differs from the builtin", not same)
    check(f"{label}: {MAX_WS} whitespace inside the call accepted", accepts(g, inner("\n" * MAX_WS)))
    check(f"{label}: {MAX_WS + 1} whitespace inside the call rejected", not accepts(g, inner("\n" * (MAX_WS + 1))))
    check(f"{label}: 2000 newlines inside the call rejected", not accepts(g, inner("\n" * 2000)))
    check(f"{label}: {MAX_WS + 1} whitespace after the call rejected", not accepts(g, CALL + "\n" * (MAX_WS + 1)))
    check(f"{label}: text after the call rejected", not accepts(g, CALL + "\nDone."))
    if label.startswith("required"):
        check(f"{label}: {MAX_WS} whitespace after the call accepted", accepts(g, CALL + " \n" * (MAX_WS // 2)))
        check(f"{label}: two calls joined by \\n accepted", accepts(g, CALL + "\n" + CALL))

    # A compiled grammar over a byte vocabulary: after the call the matcher must accept EOS, and once the
    # whitespace tail is used up, EOS must be the only legal token.
    vocab = [bytes([b]).decode("latin-1") for b in range(128)] + ["<eos>"]
    eos = len(vocab) - 1
    info = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, stop_token_ids=[eos])
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(json.dumps(tag.model_dump()))
    m = xgr.GrammarMatcher(compiled)
    for ch in CALL:
        assert m.accept_token(ord(ch)), ch
    mask = xgr.allocate_token_bitmask(1, info.vocab_size)
    m.fill_next_token_bitmask(mask)
    check(f"{label}: EOS legal right after </tool_call>", bool(mask[0, eos // 32] >> (eos % 32) & 1))
    if label.startswith("required"):
        for _ in range(MAX_WS):
            assert m.accept_token(ord("\n"))
        m.fill_next_token_bitmask(mask)
        legal = [t for t in range(info.vocab_size) if mask[0, t // 32] >> (t % 32) & 1]
        check(f"{label}: only EOS legal after {MAX_WS} trailing newlines", legal == [eos], f"legal={legal}")
    else:
        m.fill_next_token_bitmask(mask)
        legal = [t for t in range(info.vocab_size) if mask[0, t // 32] >> (t % 32) & 1]
        check(f"{label}: only EOS legal after the call", legal == [eos], f"legal={legal}")

print(f"{len(fails)} failed" if fails else "all passed")
sys.exit(1 if fails else 0)
