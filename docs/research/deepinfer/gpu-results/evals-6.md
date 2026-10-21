# evals-6: streamed tool calls under speculative decoding

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | tools added in this result's commit (`bench/test_tool_parser_chunks.py`, `bench/api_smoke.py` case 13) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88 (api_smoke part); the parser test runs on CPU |
| `.env` knobs | per mode (baseline.md); probe boots also had `VLLM_GPU_SYNC_CHECK=warn` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\health\newtests\test_tool_parser_chunks.log`, `…\probes\<mode>\sync_check.txt` |

## Commands

```bash
# CPU, in the image without a GPU, real tokenizer mounted:
docker run --rm -v $REPO/bench:/app/bench -v /mnt/g/models:/app/models:ro --entrypoint /app/venv/bin/python \
  ghcr.io/syv-ai/hyperqwen:latest /app/bench/test_tool_parser_chunks.py --seeds 200
# GPU, inside each mode's server (run by bench/sync_check.sh, which calls api_smoke first):
venv/bin/python bench/api_smoke.py
```

## Results

| Check | Result |
|---|---|
| `test_tool_parser_chunks.py`: served `qwen3_coder` + `qwen3` reasoning parser, 5 canned outputs (1–3 calls, nested JSON, a value containing `<`/`>`, a literal `<tool_call>` in a query, thinking on), deltas of 1–8 tokens over 200 seeds | **PASS**: 5/5 cases, 1,005 streams, 0 differ from the one-shot parse; the dropped-token negative control is caught |
| `api_smoke.py` `t_tool_call_stream` (`tool_choice=required`, `get_time`, reassembled and parsed) in A, B, C, D, D′, E | **PASS in all six modes**; 13/13 api_smoke features pass everywhere. Frames per call: 8 in A (no speculation), 5 under DFlash2 and MTP (multi-token deltas) |

## Verdict

INFO / pass. Streaming tool calls survive speculative decoding's multi-token deltas, in the parser and end to end in every mode. No fix needed. The new test is CPU-only (the image plus the tokenizer), so it can join the patch-integrity CI job if the user wants it there.
