#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locations derive from .env (single source of truth). MODELS_DIR there is a
# Windows-style path (e.g. G:/models, used by docker-compose); convert it for WSL.
set -a
# shellcheck disable=SC1091
[ -f "$SCRIPT_DIR/.env" ] && . "$SCRIPT_DIR/.env"
set +a
MODELS_DIR="${MODELS_DIR:-./models}"
MODELS_DIR="${MODELS_DIR//\\//}" # G:\models -> G:/models
case "$MODELS_DIR" in
  # shellcheck disable=SC1133
  [A-Za-z]:/*) _drv="${MODELS_DIR:0:1}"; MODELS_DIR="/mnt/${_drv,}${MODELS_DIR:2}" ;;
esac
MODEL_SRC="${1:-${QUANT_MODEL:-}}"
if [ -z "$MODEL_SRC" ]; then
  echo "ERROR: no source model. Usage: ./run_quant.sh <model_name_or_path> [output_dir]" \
       "  (or set QUANT_MODEL; examples: Swift-Qwen3.8-27B-Uncensored-BF16," \
       "G:/models/SomeModel-BF16, hf-org/repo-id)" >&2
  exit 2
fi
# Bare directory name (relative to MODELS_DIR) or absolute path or HF repo id.
MODEL_SRC="${MODEL_SRC//\\//}"                # G:\models\Foo -> G:/models/Foo
case "$MODEL_SRC" in
  /*|[A-Za-z]:/*|*/*) ;;                       # already a path or repo id
  *) MODEL_SRC="$MODELS_DIR/$MODEL_SRC" ;;     # name within MODELS_DIR
esac
case "$MODEL_SRC" in
  [A-Za-z]:/*) _drv="${MODEL_SRC:0:1}"; MODEL_SRC="/mnt/${_drv,}${MODEL_SRC:2}" ;;
esac
OUTPUT_DIR="${2:-${QUANT_OUTPUT:-}}"
if [ -z "$OUTPUT_DIR" ]; then
  case "$MODEL_SRC" in
    */*) OUTPUT_DIR="$MODELS_DIR/$(basename "$MODEL_SRC" | sed 's/-BF16$//' | sed 's/-bf16$//')-W4A16" ;;
    *) OUTPUT_DIR="$MODELS_DIR/$(basename "$MODEL_SRC")-W4A16" ;;
  esac
fi
OUTPUT_DIR="${OUTPUT_DIR//\\//}"
case "$OUTPUT_DIR" in
  [A-Za-z]:/*) _drv="${OUTPUT_DIR:0:1}"; OUTPUT_DIR="/mnt/${_drv,}${OUTPUT_DIR:2}" ;;
esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Both GPUs: 3080 Ti 12 GB (cuda:0) + 3090 24 GB (cuda:1). accelerate packs
# the first-listed device first, so "1,0" prioritizes the 3090 (~21.6 GB of
# layers at the 0.9 ratio) and spills the remainder to the 3080 Ti.
export CUDA_VISIBLE_DEVICES=0,1
DEVICE_MAP="${DEVICE_MAP:-1,0}"

# Preflight: fail fast if the primary quant GPU is short on free VRAM.
# accelerate packs the FIRST device in DEVICE_MAP first; if that card is
# already occupied, layers silently spill onto the next card and OOM there
# (2026-09-17: an idle server held 23.4 GiB on the 3090, so everything
# landed on the 12 GB 3080 Ti and died on block 0).
map_cuda_to_phys() { # logical cuda idx -> physical GPU idx via CUDA_VISIBLE_DEVICES
  local i="${1:-0}"
  local -a vis=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a vis <<< "$CUDA_VISIBLE_DEVICES"
    echo "${vis[$i]:-$i}"
  else
    echo "$i"
  fi
}
PRIMARY_PHYS="$(map_cuda_to_phys "${DEVICE_MAP%%,*}")"
PRIMARY_FREE_MIB="$(nvidia-smi -i "$PRIMARY_PHYS" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')"
PRIMARY_MIN_FREE_MIB="${PRIMARY_MIN_FREE_MIB:-21000}"
if [ -n "$PRIMARY_FREE_MIB" ] && [ "$PRIMARY_FREE_MIB" -lt "$PRIMARY_MIN_FREE_MIB" ]; then
  echo "ERROR: primary quant GPU (physical $PRIMARY_PHYS) has ${PRIMARY_FREE_MIB} MiB free; need >= ${PRIMARY_MIN_FREE_MIB}." >&2
  echo "       Free that GPU first, or override DEVICE_MAP / PRIMARY_MIN_FREE_MIB." >&2
  exit 1
fi
echo "Preflight: primary quant GPU (physical $PRIMARY_PHYS) free=${PRIMARY_FREE_MIB:-unknown} MiB (threshold ${PRIMARY_MIN_FREE_MIB})."
export AR_DISK_STREAM_MODEL=1
export AR_WORK_SPACE="$MODELS_DIR/.quant-work/ar_work_space"
export AR_RESUME_DIR="$MODELS_DIR/.quant-work/.quant_checkpoint"
# WSL root (/tmp, /usr) is currently read-only: keep every temp/compile
# cache on the writable NVMe volume instead of the defaults under /tmp, ~/.
export TMPDIR="$MODELS_DIR/.quant-work/tmp"
export TORCHINDUCTOR_CACHE_DIR="$MODELS_DIR/.quant-work/torchinductor"
export TRITON_CACHE_DIR="$MODELS_DIR/.quant-work/triton"
mkdir -p "$AR_WORK_SPACE/offload/compressor_resume" "$AR_RESUME_DIR" prepare/logs \
  "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "Starting AutoRound on 3080 Ti + 3090; models: $MODELS_DIR workspace: $AR_WORK_SPACE"
venv/bin/auto_round quantize "$MODEL_SRC" \
  --scheme W4A16 \
  --bits 4 \
  --group_size 128 \
  --format auto_round:llm_compressor \
  --device_map "$DEVICE_MAP" \
  --low_gpu_mem_usage \
  --nsamples 128 \
  --seqlen 2048 \
  --batch_size 4 \
  --output_dir "$OUTPUT_DIR" \
  --no-quant_lm_head

printf '\nAutoRound finished successfully.\n'
