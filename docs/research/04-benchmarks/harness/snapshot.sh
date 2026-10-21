#!/bin/bash
# GPU + environment snapshot, to OUTDIR/snapshot_<TAG>.txt. Run it inside the serving container (nvidia-smi,
# env knobs, vLLM and server identity) AND on the WSL host (docker image, compose .env, KV pool log lines);
# every part is skipped when its tool is absent, so it never fails on a box without a GPU.
#   bash snapshot.sh OUTDIR [pre|post]
OUT=${1:?usage: snapshot.sh OUTDIR [TAG]}; TAG=${2:-snap}; mkdir -p "$OUT"; F="$OUT/snapshot_${TAG}_$(uname -n).txt"
BENCH=${BENCH:-/app/bench}; PY=${PY:-$(command -v /app/venv/bin/python || command -v python3)}
sec() { echo; echo "## $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
{
sec date; date -u +%FT%TZ; uname -n; uname -r
sec nvidia-smi
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,uuid,driver_version,power.limit,power.draw,clocks.max.sm,clocks.sm,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
  nvidia-smi -q -d PERFORMANCE,POWER 2>/dev/null | grep -E "Clocks Event|Throttle|SW Power|HW Slowdown|Thermal|Power Limit|Enforced|Current Power" | head -20
else echo "nvidia-smi: not found (no GPU on this box)"; fi
sec "env knobs (keys redacted)"
env | grep -E '^(SPEC|CTX|DFLASH|DRAFT|LOOKUP|DYNAMIC_K|PREFIX_CACHE|PTH_FA|FA2|KV|MAX_|GPU_UTIL|VISION|VLLM_|EXTRA_ARGS|MODEL|PORT|INT8|SPEC_ATTN|HARNESS|QUALITY|CUDA_VISIBLE|NVIDIA_VISIBLE)' | grep -viE 'KEY|TOKEN|SECRET' | sort
echo "PTH_FA=${PTH_FA-<UNSET>}  (pinned explicitly? an unset value follows the launcher copy's default: upstream b4b97ec)"
sec "server"
if curl -sf -m 5 -o /dev/null "http://127.0.0.1:${PORT:-18020}/health"; then
  VLLM_API="http://127.0.0.1:${PORT:-18020}" PYTHONPATH="$BENCH" "$PY" -c "import harness;print('model:',harness.model());print('spec counters:',harness.spec())" 2>&1
  PYTHONPATH="$BENCH" "$PY" -c "
import harness
for n,l,v in harness.samples():
    if n in ('vllm:cache_config_info','vllm:kv_cache_usage_perc') or n.startswith('vllm:spec_decode_num_accepted_tokens_per_pos'): print(n,l,v)" 2>&1 | cut -c1-600
else echo "no server on 127.0.0.1:${PORT:-18020}"; fi
sec "vLLM / image"
[ -x /app/venv/bin/python ] && /app/venv/bin/python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)" 2>&1 | tail -1
[ -f /app/patches/series ] && { echo "patches/series sha256: $(sha256sum /app/patches/series | cut -c1-16)  count: $(grep -vc '^#\|^$' /app/patches/series)"; }
sec "repo (host side)"
have git && git rev-parse --show-toplevel >/dev/null 2>&1 && { git log -1 --format='%H %s'; git status --short | head -10; }
sec "docker (host side)"
if have docker; then
  docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>&1 | head
  for c in $(docker ps -q 2>/dev/null | head -3); do docker inspect --format '{{.Name}} image={{.Image}} cfg={{.Config.Image}}' "$c"; done
  [ -f .env ] && { echo "--- .env (keys redacted)"; sed -E 's/^([A-Za-z_]*(KEY|TOKEN|SECRET)[A-Za-z_]*)=.*/\1=<redacted>/' .env; }
  for s in single batch; do docker compose logs --tail 2000 "$s" 2>/dev/null | grep -iE "effective-config|GPU KV cache size|Maximum concurrency|KV cache" | tail -8; done
fi
have wsl.exe && wsl.exe -l -v 2>/dev/null | tr -d '\0'
true
} > "$F" 2>&1
echo "wrote $F"
