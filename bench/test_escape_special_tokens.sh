#!/bin/bash
# CPU-only check for serve-escape-special-tokens.patch (evals-5): in a throwaway container of the stock image (no
# GPU), record the stock vLLM's chat ids for the test corpus, dry-run then apply the patch to the installed vLLM, and
# run bench/test_escape_special_tokens.py against that record (it toggles VLLM_ESCAPE_SPECIAL_TOKENS itself).
# Exits non-zero if the patch does not apply at --fuzz 0 or any check fails.
#   bash bench/test_escape_special_tokens.sh     # IMAGE=ghcr.io/syv-ai/hyperqwen:latest, MODELS=/mnt/g/models
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-ghcr.io/syv-ai/hyperqwen:latest}
MODELS=${MODELS:-/mnt/g/models}
docker run --rm --cpus 4 -v "$REPO:/repo:ro" -v "$MODELS:/app/models:ro" --entrypoint bash "$IMAGE" -c '
set -e
SP=/app/venv/lib/python3.12/site-packages/vllm
P=/repo/docs/research/deepinfer/experiments/patches/serve-escape-special-tokens.patch
cd /repo/bench
nice -n 19 /app/venv/bin/python test_escape_special_tokens.py --dump /tmp/stock.json
patch -p1 --forward --fuzz 0 --dry-run -d $SP -i $P
patch -p1 --forward --fuzz 0 --quiet -d $SP -i $P
nice -n 19 /app/venv/bin/python test_escape_special_tokens.py --stock /tmp/stock.json
' 2>&1 | grep -v -e "pin_memory=False" -e "No CUDA runtime" -e "vllm._C" -e "_deepselect_C" -e "is part of Qwen3VLVideoProcessorInitKwargs" \
          -e "Model Runner V2 requires Triton"
