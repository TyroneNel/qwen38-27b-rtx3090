# runtime-3: bound the tail of forced tool calls

| Field | Value |
|---|---|
| Date | 2026-10-21 17:22–18:20 SAST |
| Commit | in this result's commit: `docs/research/deepinfer/experiments/patches/runtime3-tool-call-bound.patch`, `bench/tool_runaway.py`, `bench/test_tool_runaway.py`, `bench/test_tool_call_bound.{py,sh}` |
| Image | `hq:exp-runtime3` = stock `ghcr.io/syv-ai/hyperqwen:latest` + `runtime3-tool-call-bound.patch` (inert unless `VLLM_TOOL_CALL_BOUND=1`); both arms on this image |
| GPU | RTX 3090, 250 W, driver 610.88 |
| `.env` knobs | batch: `GPU_UTIL=0.91`; single: `SPEC=dflash2 PREFIX_CACHE=1`; on-arms add `VLLM_TOOL_CALL_BOUND=1` (whitespace cap `VLLM_TOOL_CALL_MAX_WS` default 16) |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\runtime-3\{batch,single}_{off,on}\rt3_*.{txt,jsonl}` |

## Cause, from the image's code

Forced calls (`tool_choice=required` or a named tool, with `VLLM_ENFORCE_STRICT_TOOL_CALLING` on, the default) are constrained by xgrammar's built-in `qwen_3_coder` structural tag, which vLLM builds without `max_whitespace_cnt`:
- For a tool with no parameters, the argument grammar is `[ \n\r\t]*`, so any amount of whitespace is grammatical.
- With `required` and parallel calls left on, free text after `</tool_call>` is also allowed until end of sequence.

The patch changes only `vllm/tool_parsers/structural_tag_registry.py`, and only for required and named calls. It caps every whitespace run inside arguments at 16 characters. After a required call it allows only `\n` + another call, or ≤16 whitespace characters then end of sequence. `auto` is untouched.

CPU checks: `bench/test_tool_call_bound.sh` passes (off: byte-identical to xgrammar's tag and still accepts a 2,000-newline runaway; on: 16 accepted, 17 rejected, 16-space code indentation inside a string parameter accepted), and `bench/test_tool_runaway.py` passes.

## Commands

```bash
# in each arm's container
venv/bin/python bench/tool_runaway.py --seeds 40 --json /out/rt3_t1.jsonl              # forced get_time, 1 tool
venv/bin/python bench/tool_runaway.py --seeds 40 --tools 2 --json /out/rt3_t2.jsonl    # + get_weather
venv/bin/python bench/tool_runaway.py --seeds 40 --choice named --json /out/rt3_named.jsonl
```

Runaway: `finish_reason=length` (max_tokens 2048) or >32 whitespace-only tokens in a row inside or after the call. Valid: finish `tool_calls`/`stop`, every call names an offered tool, and its arguments match the schema.

## Numbers (40 seeds each, thinking at the server default)

| Mode | Case | Off: runaways (stage) | Off: valid calls | On: runaways | On: valid calls |
|---|---|---|---|---|---|
| A (batch) | 1 tool, required | 6/40 (in call) | 34/40 | **0/40** | **40/40** |
| A | 2 tools, required | 10/40 (6 in call, 4 after) | 34/40 | **0/40** | **40/40** |
| A | named | 6/40 (in call) | 34/40 | **0/40** | **40/40** |
| B (single) | 1 tool, required | 10/40 (9 in, 1 after) | 31/40 | **0/40** | **40/40** |
| B | 2 tools, required | 8/40 (7 in, 1 after) | 33/40 | **0/40** | **40/40** |
| B | named | 10/40 (9 in, 1 after) | 31/40 | **0/40** | **40/40** |

No request errored in any arm, and none ran away before the call (thinking never consumed the budget).

## Against the appendix

- The runaway rate on this box is higher than `docs/vllm-0.31.md` recorded: 15–25% of forced calls (6–10 of 40) against 2/20 in batch and 1/20 in single there. The appendix's "5–10%" understates it.
- About a fifth of the runaways happen *after* the call (free text following `</tool_call>` under `required`), a second path the appendix did not name.

## Verdict

**WIN.** The appendix's criterion is "0 runaways in 40 seeds, with every valid call still accepted", and the result is 0/40 runaways and 40/40 valid calls in all 6 cells (two modes × three cases), down from 6–10/40 runaways. It is a grammar-only change and needs no speed run (the same structural-tag path, a tighter grammar).
