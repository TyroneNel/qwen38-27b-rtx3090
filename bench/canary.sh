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

# A fresh server decodes at about half speed per step until it has done some
# work, then drops to full speed in a single step. Idle time does not help: a
# swift-base boot left idle 3 minutes was still slow on its first request.
# Same prompt, same tok/step (3.39) on every pass:
#
#   pass1 19.14   pass2 17.70   pass3 16.74   pass4 13.33   pass5+ 8.81-8.90 ms
#
# The drop coincides with the server allocating ~430 MiB (2118 -> 1688 MiB
# free) and nothing in the server log. Slow requests before the drop, observed
# 2026-09-23: 1 on the -fast variants, 4, 7 and 10 on base checkpoints.
#
# The slow phase is a plateau, not a curve (~57 ms/step held for 8 passes), so
# it cannot be detected by waiting for passes to agree -- a stuck-slow server
# looks settled. A fixed best-of-4 or settle-on-stability both called healthy
# base servers MARGINAL/DEGRADED. Instead do WARMUP untimed requests, above the
# worst case seen, then time PASSES and report the best.
WARMUP=${WARMUP:-15}
PASSES=${PASSES:-3}

for _w in $(seq 1 "$WARMUP"); do
  curl -fsS --max-time 180 "${AUTH[@]}" -H 'Content-Type: application/json' \
    -d "$REQ" "$BASE/v1/completions" >/dev/null || { echo "canary: warmup request failed" >&2; exit 2; }
done

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

printf 'canary: %s tokens in %ss  decode=%s tok/s  TPOT=%s ms  vram_free=%s MiB  (after %s warmup requests)\n' \
  "$OUT_TOK" "$WALL" "$TPS" "$TPOT" "${FREE:-?}" "$WARMUP"

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
