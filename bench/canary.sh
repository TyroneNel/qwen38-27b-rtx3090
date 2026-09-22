#!/usr/bin/env bash
# Fast health canary: is this boot decoding at full rate, or is it degraded?
#
# The full harness takes ~15 minutes and eight cohorts to say what one request
# says in four seconds. The discriminator is per-token latency, not throughput:
# a degraded boot on this box does identical work (same draft tokens, same
# acceptance, same tok/step) at ~2.6x the time per token.
#
#   healthy   TPOT ~7 ms    decode >110 tok/s
#   degraded  TPOT ~19 ms   decode <60 tok/s
#
# Run it after every boot, before trusting any number from that server.
#
#   bash bench/canary.sh                  # against $HOST:$PORT
#   bash bench/canary.sh 300              # 300 output tokens instead of 200
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18020}
NTOK=${1:-200}
MODEL_NAME=${MODEL_NAME:-qwen3.8-27b}

if [ -f "$REPO/resolve_api_key.sh" ]; then
  # shellcheck source=/dev/null
  source "$REPO/resolve_api_key.sh"
  resolve_client_key
else
  export OPENAI_API_KEY=${OPENAI_API_KEY:-${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}}
fi
AUTH=()
[ -n "${OPENAI_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $OPENAI_API_KEY")

BASE="http://$HOST:$PORT"
if ! curl -fsS --max-time 5 "$BASE/health" >/dev/null 2>&1; then
  echo "canary: server not answering on $BASE/health" >&2
  exit 2
fi

# Greedy, fixed length, ignore_eos: the generation is the same work every run,
# so the only thing that varies between boots is how fast it comes out.
REQ=$(printf '{"model":"%s","prompt":"Count slowly and describe each number.","max_tokens":%d,"min_tokens":%d,"temperature":0,"ignore_eos":true,"stream":false}' \
  "$MODEL_NAME" "$NTOK" "$NTOK")

# A fresh server does not reach steady decode on one request. Measured on a
# single boot with no config change between passes:
#
#   pass1 28.35 ms   pass2 15.55 ms   pass3 7.89 ms   pass4 8.10 ms   pass5 8.38 ms
#
# One short warmup is not enough -- a single timed pass after it samples a
# random point on that curve and will call a healthy server degraded. Measure
# repeatedly instead and report the best, which is the steady state: warmup
# only ever makes a pass slower, never faster, so the minimum is the honest
# number and no averaging can recover it once a slow pass is in the mean.
PASSES=${PASSES:-4}

best_wall=""; best_tok=0
for _pass in $(seq 1 "$PASSES"); do
  START=$(date +%s.%N)
  RESP=$(curl -fsS --max-time 180 "${AUTH[@]}" -H 'Content-Type: application/json' \
    -d "$REQ" "$BASE/v1/completions") || { echo "canary: request failed" >&2; exit 2; }
  END=$(date +%s.%N)

  OUT_TOK=$(printf '%s' "$RESP" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+$' | tail -1)
  OUT_TOK=${OUT_TOK:-0}
  [ "$OUT_TOK" -gt 0 ] || { echo "canary: no tokens generated" >&2; exit 2; }

  WALL=$(awk -v s="$START" -v e="$END" 'BEGIN{printf "%.4f", e-s}')
  if [ -z "$best_wall" ] || awk -v a="$WALL" -v b="$best_wall" 'BEGIN{exit !(a<b)}'; then
    best_wall=$WALL; best_tok=$OUT_TOK
  fi
done

OUT_TOK=$best_tok
read -r WALL TPS TPOT <<<"$(awk -v w="$best_wall" -v n="$OUT_TOK" \
  'BEGIN{ printf "%.2f %.1f %.2f", w, n/w, 1000*w/n}')"

# Report the serving card, not whichever GPU happens to have the most free: on
# a mixed box the desktop card would mask the server's headroom entirely.
GPU_SEL=${GPU_UUID:+-i $GPU_UUID}
FREE=$(nvidia-smi ${GPU_SEL:-} --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)

printf 'canary: %s tokens in %ss  decode=%s tok/s  TPOT=%s ms  vram_free=%s MiB\n' \
  "$OUT_TOK" "$WALL" "$TPS" "$TPOT" "${FREE:-?}"

# Thresholds measured on this box, short-request canary (200 tokens):
#   healthy   9.9-10.9 ms TPOT   (KV pool leaving >2 GiB VRAM free)
#   degraded 18.9-30.6 ms TPOT   (pool pinned to ~800 MiB free, gotcha 43)
# The dead band between 13 and 18 keeps a merely-busy host from reading as
# degraded. A full-harness C1 runs faster than this (~7 ms) because it reuses a
# warm prefix; do not compare the two numbers directly.
VERDICT=$(awk -v t="$TPOT" 'BEGIN{ if (t<=13) print "HEALTHY"; else if (t>=18) print "DEGRADED"; else print "MARGINAL" }')
case "$VERDICT" in
  HEALTHY)  echo "canary: HEALTHY — decode at full rate"; exit 0;;
  MARGINAL) echo "canary: MARGINAL — between known states; re-run or check host load"; exit 1;;
  DEGRADED) echo "canary: DEGRADED — this boot decodes at ~1/3 rate; restart the server before benching"; exit 1;;
esac
