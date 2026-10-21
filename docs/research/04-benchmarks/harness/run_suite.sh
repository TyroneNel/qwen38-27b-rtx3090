#!/bin/bash
# One boot's measurement suite. Run INSIDE the serving container (README.md has the commands), after the
# server answers /health. Each phase logs to OUT/<phase>.log; a failed phase is recorded and the rest still run.
#
#   bash /harness/run_suite.sh LABEL                    # default phases
#   PHASES="snap sweep greedy" bash /harness/run_suite.sh LABEL
#   PHASES="snap quality" ...                           # quality needs GPU_UTIL headroom (README)
#
# Phases (default: snap warm probe sweep ladder greedy spill):
#   snap     nvidia-smi + env snapshot, pre and post                 (snapshot.sh)
#   warm     the repo's bench/warmup.sh
#   probe    warmup_probe until two runs agree within 3% (vendor/)   -> WARM/NOT-WARM verdict
#   sweep    TTFT, decode, throughput, acceptance, peak VRAM on the fixed prompt sets (sweep.py)
#   ladder   bench/conc_ladder.py (reused unchanged): N-stream scaling, ms/pass, preemptions, KV%
#   greedy   record T=0 token ids for the spec-vs-non-spec check   (greedy_equal.py compare happens off-box)
#   quality  bench/quality_battery.py + quality_guard.py            (not in the default set)
#   spill    WDDM non-local usage summary of $GPUMEM_CSV (README "VRAM and spill"); skipped if GPUMEM_CSV is unset
#   profile  torch profile + vendor/trace_breakdown.py: draft vs verify GPU time (needs the profiler boot, README)
# PTH_FA must be set (0 or 1) in the booted container; run_suite.sh exits 2 otherwise.
# Sizing knobs (defaults are for a ~68k-token KV pool, mode B; raise them only in CTX=long/huge modes):
#   (LC_CELLS="" skips the long-context cells; SMOKE=1 lets REPS go below 3, smoke runs only)
#   REPS=3 WARMUP=1 MAXTOK=256 CONC=1,2,4,8 LC_CELLS="8000:1,2,4 32000:1" LADDER_CELLS="200:1,2,4,8 8000:1,2,4"
#   SALT=$LABEL (same salt on every boot of an A/B; a second run on the same boot needs a new one)
LABEL=${1:?usage: run_suite.sh LABEL}
BENCH=${BENCH:-/app/bench}; HARNESS=${HARNESS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUT=${OUT:-/results/$LABEL-$(date +%Y%m%d-%H%M%S)}; mkdir -p "$OUT"
PY=${PY:-$(command -v /app/venv/bin/python || command -v python3)}
export PYTHONPATH="$BENCH:$HARNESS"
PHASES=${PHASES:-snap warm probe sweep ladder greedy spill}
REPS=${REPS:-3}; WARMUP=${WARMUP:-1}; MAXTOK=${MAXTOK:-256}; CONC=${CONC:-1,2,4,8}; SALT=${SALT:-$LABEL}
LC_CELLS=${LC_CELLS-"8000:1,2,4 32000:1"}; LADDER_CELLS=${LADDER_CELLS:-"200:1,2,4,8 8000:1,2,4"}
QUALITY_DATA=${QUALITY_DATA:-/app/models/quality-data}; export QUALITY_DATA
PORT=${PORT:-18020}; fails=0
has() { [[ " $PHASES " == *" $1 "* ]]; }
run() { # phase cmd...   (never aborts the suite)
  local p=$1; shift; echo "[$(date +%T)] $p: $*" | tee -a "$OUT/suite.log"
  "$@" > "$OUT/$p.log" 2>&1; local rc=$?
  echo "$p rc=$rc" >> "$OUT/status.txt"; tail -n "${TAIL:-6}" "$OUT/$p.log"; [ $rc -ne 0 ] && fails=$((fails+1)); return 0; }
echo "label=$LABEL out=$OUT phases=$PHASES" | tee "$OUT/suite.log"
# PTH_FA must be pinned in the arm's .env: upstream b4b97ec (#345, in the stock image, not in our 95ac251 launchers) makes an
# unset PTH_FA resolve to 1 for SPEC=mtp CTX=long on cc 8.6 (~+9% decode at ~66k), so an unpinned arm depends on which
# launcher copy booted. Refuse to measure an unpinned arm (ALLOW_UNSET_PTH_FA=1 overrides, and the run is marked).
if [ -z "${PTH_FA:-}" ]; then
  if [ "${ALLOW_UNSET_PTH_FA:-0}" = 1 ]; then echo "WARNING: PTH_FA unset: arm depends on the launcher's default" | tee -a "$OUT/suite.log"
  else echo "PTH_FA is unset in this container: pin PTH_FA=0 or 1 in .env and reboot (README, 'Arm rules')" >&2; exit 2; fi
fi
echo "PTH_FA=${PTH_FA:-<UNSET>} SPEC=${SPEC:-<unset>} CTX=${CTX:-<unset>}" | tee -a "$OUT/suite.log"
curl -sf -m 10 -o /dev/null "http://127.0.0.1:$PORT/health" || { echo "no server on :$PORT" >&2; exit 1; }

has snap && bash "$HARNESS/snapshot.sh" "$OUT" pre
has warm && run warm bash "$BENCH/warmup.sh"
has probe && run probe "$PY" "$HARNESS/vendor/warmup_probe.py" --tokens "${PROBE_TOKENS:-256}" --out "$OUT/warmup_probe.json"
if has sweep; then
  for s in short_chat code; do
    run "sweep_$s" "$PY" "$HARNESS/sweep.py" "$LABEL" --set $s --conc "$CONC" --reps "$REPS" --warmup "$WARMUP" \
        --max-tokens "$MAXTOK" --salt "$SALT" --out "$OUT"; done
  for cell in $LC_CELLS; do
    run "sweep_long_${cell%%:*}" "$PY" "$HARNESS/sweep.py" "$LABEL-${cell%%:*}" --set long_context --ctx "${cell%%:*}" \
        --conc "${cell##*:}" --reps "$REPS" --warmup "$WARMUP" --max-tokens "${LC_MAXTOK:-128}" --salt "$SALT" --out "$OUT"; done
fi
if has ladder; then
  for cell in $LADDER_CELLS; do c=${cell%%:*}
    run "ladder_$c" "$PY" "$HARNESS/vram.py" "$OUT/ladder_${c}_vram.json" -- "$PY" "$BENCH/conc_ladder.py" --n "${cell##*:}" \
        --ctx-tokens "$c" --out "${LADDER_OUT:-256}" --reps "$REPS" --salt "$SALT" --label "$LABEL-$c" --json "$OUT/ladder_$c.json"; done
fi
has greedy && run greedy "$PY" "$HARNESS/greedy_equal.py" record "$LABEL" --max-tokens "${GREEDY_TOKENS:-256}" --out "$OUT"
if has quality; then
  run quality "$PY" "$BENCH/quality_battery.py" "$LABEL"
  run quality_guard "$PY" "$HARNESS/quality_guard.py" "$QUALITY_DATA/result_$LABEL.json"
  cp "$QUALITY_DATA/result_$LABEL.json" "$OUT/" 2>/dev/null
fi
if has profile; then
  # the server must have been booted with EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/results/prof"}'
  source /app/resolve_api_key.sh 2>/dev/null && resolve_client_key
  AUTH=(-H "Authorization: Bearer ${OPENAI_API_KEY:-EMPTY}")
  curl -sf -X POST "${AUTH[@]}" "http://127.0.0.1:$PORT/start_profile" > "$OUT/profile_start.log" 2>&1
  run profile_load "$PY" "$HARNESS/sweep.py" "$LABEL-prof" --set short_chat --conc 1 --reps 3 --warmup 0 --max-tokens "$MAXTOK" --salt "$SALT-prof" --out "$OUT"
  curl -sf -X POST "${AUTH[@]}" "http://127.0.0.1:$PORT/stop_profile" >> "$OUT/profile_start.log" 2>&1
  run profile_breakdown "$PY" "$HARNESS/vendor/trace_breakdown.py" "${PROF_DIR:-/results/prof}" --json "$OUT/breakdown.json"
fi
if has spill; then
  if [ -r "${GPUMEM_CSV:-}" ]; then cp "$GPUMEM_CSV" "$OUT/gpumem.csv"; run spill "$PY" "$HARNESS/gpumem_summary.py" "$OUT/gpumem.csv" "$OUT/spill.json"
  else echo "spill: GPUMEM_CSV unset or unreadable: no WDDM spill reading for this boot (README 'VRAM and spill')" | tee -a "$OUT/suite.log"; fi
fi
has snap && bash "$HARNESS/snapshot.sh" "$OUT" post
echo "done: $fails failed phase(s); results in $OUT" | tee -a "$OUT/suite.log"
[ $fails -eq 0 ]
