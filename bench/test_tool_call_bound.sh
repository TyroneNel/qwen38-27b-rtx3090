#!/bin/bash
# CPU-only check for runtime3-tool-call-bound.patch (runtime-3): in a throwaway container of the stock image (no
# GPU), dry-run then apply the patch to the installed vLLM, and run bench/test_tool_call_bound.py with the knob
# off and on. Exits non-zero if the patch does not apply at --fuzz 0 or either run fails.
#   bash bench/test_tool_call_bound.sh            # IMAGE=ghcr.io/syv-ai/hyperqwen:latest by default
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-ghcr.io/syv-ai/hyperqwen:latest}
docker run --rm -v "$REPO:/repo:ro" --entrypoint bash "$IMAGE" -c '
set -e
SP=/app/venv/lib/python3.12/site-packages/vllm
P=/repo/docs/research/deepinfer/experiments/patches/runtime3-tool-call-bound.patch
patch -p1 --forward --fuzz 0 --dry-run -d $SP -i $P
patch -p1 --forward --fuzz 0 --quiet -d $SP -i $P
cd /repo/bench
/app/venv/bin/python test_tool_call_bound.py
VLLM_TOOL_CALL_BOUND=1 /app/venv/bin/python test_tool_call_bound.py
VLLM_TOOL_CALL_BOUND=1 VLLM_TOOL_CALL_MAX_WS=4 /app/venv/bin/python test_tool_call_bound.py
' 2>&1 | grep -v -e "pin_memory=False" -e "No CUDA runtime"
