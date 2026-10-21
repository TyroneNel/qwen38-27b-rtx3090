#!/bin/bash
# CPU self-check of the harness against mock_server.py (no GPU, no model). Exercises run_suite.sh end to end
# (snapshot, warmup_probe, sweep x3 sets, the repo's conc_ladder.py, greedy record), then greedy_equal compare
# (identical, near-tie, real divergence) and quality_guard (pass and fail). Prints SELFCHECK PASS or exits 1.
#   bash selfcheck.sh            # from anywhere; output saved to selfcheck.out by the caller (tee)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../../../.." && pwd)"
T=$(mktemp -d); trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$T"' EXIT
export PORT=18999 OPENAI_API_KEY=k PTH_FA=0 PYTHONPATH="$REPO/bench:$HERE"; unset VLLM_API VLLM_MODEL
PY=${PY:-python3}; fail=0
chk() { local want=$1 got=$2 what=$3; if [ "$want" = "$got" ]; then echo "ok   $what (exit $got)"; else echo "FAIL $what: want exit $want, got $got"; fail=1; fi; }
mock() { kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; "$PY" "$HERE/mock_server.py" --port $PORT "$@" & for i in $(seq 50); do curl -sf -o /dev/null localhost:$PORT/health && return; sleep 0.1; done; echo "mock did not start"; exit 1; }
head -c 300000 "$REPO"/docs/*.md > "$T/corpus.txt"   # stand-in for the frozen labd corpus

echo "== 1. run_suite.sh against the mock (spec on: acceptance counters move)"
printf "ts,nonlocal_mib\n1,370\n2,372\n3,378\n" > "$T/gpumem.csv"; printf "1,370\n2,698\n" > "$T/gpumem_spill.csv"
mock --spec 7 --tps 3.5 --rate 300
HARNESS_CORPUS="$T/corpus.txt" BENCH="$REPO/bench" HARNESS="$HERE" OUT="$T/suite" PHASES="snap probe sweep ladder greedy spill" GPUMEM_CSV="$T/gpumem.csv" \
  PROBE_TOKENS=24 GREEDY_TOKENS=24 REPS=3 MAXTOK=24 LC_MAXTOK=16 CONC=1,2 LC_CELLS="1000:1,2" LADDER_CELLS="44:1,2" LADDER_OUT=16 TAIL=4 \
  bash "$HERE/run_suite.sh" selfcheck-spec; chk 0 $? "run_suite.sh (all phases)"
cat "$T/suite/status.txt"
echo "-- snapshot (no GPU here, must degrade gracefully):"; head -12 "$T"/suite/snapshot_pre_*.txt
echo "-- sweep json: acceptance and spread are present:"
"$PY" - "$T/suite" <<'PYEOF'
import json, sys, glob
for f in sorted(glob.glob(sys.argv[1] + "/sweep_*short_chat.json")):
    c = json.load(open(f))["cells"][1]
    s = c["summary"]
    print(f"conc={c['conc']} ttft_mean_ms={s['ttft']['mean']:.1f} sd={s['ttft']['sd']:.1f} reps={s['ttft']['n']} "
          f"tok_step={s['tok_step']['mean']:.2f} accept_rate={s['accept_rate']['mean']:.3f} vram={c['vram']}")
    assert s["ttft"]["n"] >= 3 and 1.5 < s["tok_step"]["mean"] < 7 and s["accept_rate"]["mean"] > 0 and c['vram']['peak_mib'] is None
PYEOF
chk 0 $? "sweep json: n>=3, tok/step and accept rate read from /metrics, vram None without nvidia-smi"
cp "$T/suite"/greedy_selfcheck-spec.json "$T/spec.json"

echo "== 1b. run_suite.sh refuses an arm with PTH_FA unpinned"
env -u PTH_FA BENCH="$REPO/bench" HARNESS="$HERE" OUT="$T/unpinned" PHASES="snap" bash "$HERE/run_suite.sh" unpinned 2>&1 | tail -1; chk 2 ${PIPESTATUS[0]} "unpinned PTH_FA is refused"
grep -h "^PTH_FA" "$T"/suite/snapshot_pre_*.txt | head -1
echo "== 1c. sweep decode rate keeps the EOS-only last step in its window (non-spec mock, 300 tok/s; review fix)"
mock --rate 300; "$PY" "$HERE/sweep.py" ratecheck --set short_chat --conc 1 --max-tokens 8 --out "$T/rate" >/dev/null
"$PY" - "$T/rate/sweep_ratecheck_short_chat.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))["cells"][0]["summary"]["rate"]["mean"]
print(f"decode {r:.1f} tok/s/stream vs mock 300 (without the fix: ~+17%, 7 tokens over 6 steps)")
assert 240 < r < 315, r
PYEOF
chk 0 $? "decode rate = (ntok-1) over first content -> finish chunk"
echo "== 2. greedy_equal: non-spec vs spec (must be identical)"
mock --rate 300; "$PY" "$HERE/greedy_equal.py" record nospec --max-tokens 24 --out "$T" >/dev/null
"$PY" "$HERE/greedy_equal.py" compare "$T/greedy_nospec.json" "$T/spec.json" | tail -1; chk 0 ${PIPESTATUS[0]} "identical spec vs non-spec"
echo "== 3. greedy_equal: injected divergence at token 5"
mock --rate 300 --perturb 5; "$PY" "$HERE/greedy_equal.py" record real --max-tokens 24 --out "$T" >/dev/null
"$PY" "$HERE/greedy_equal.py" compare "$T/greedy_nospec.json" "$T/greedy_real.json" | tail -2; chk 1 ${PIPESTATUS[0]} "real divergence is flagged"
mock --rate 300 --tie-at 5; "$PY" "$HERE/greedy_equal.py" record tieA --max-tokens 24 --out "$T" >/dev/null
mock --rate 300 --tie-at 5 --perturb 5; "$PY" "$HERE/greedy_equal.py" record tieB --max-tokens 24 --out "$T" >/dev/null
"$PY" "$HERE/greedy_equal.py" compare "$T/greedy_tieA.json" "$T/greedy_tieB.json" | tail -2; chk 1 ${PIPESTATUS[0]} "near-tie divergence is flagged and classified"

echo "== 4. quality_guard on synthetic battery results (not real measurements)"
echo '{"ppl":{"all":[8.401,1],"en":[10.77,1],"da":[10.91,1],"code":[3.60,1]},"gsm8k":{"acc":0.95}}' > "$T/q_ok.json"
echo '{"ppl":{"all":[8.737,1],"en":[11.04,1],"da":[11.41,1],"code":[3.78,1]},"gsm8k":{"acc":0.96}}' > "$T/q_bad.json"
"$PY" "$HERE/quality_guard.py" "$T/q_ok.json" | tail -1; chk 0 ${PIPESTATUS[0]} "guard passes a lossless result"
"$PY" "$HERE/quality_guard.py" "$T/q_bad.json" | tail -1; chk 1 ${PIPESTATUS[0]} "guard fails +4% ppl (the prior all-linear-int8 pair)"
echo '{}' > "$T/q_inv.json"; "$PY" "$HERE/quality_guard.py" "$T/q_inv.json" | tail -1; chk 2 ${PIPESTATUS[0]} "guard: invalid input"

echo "== 4b. gpumem_summary: flat vs SPILL (synthetic samples)"
grep -q '"flat"' "$T/suite/spill.json"; chk 0 $? "suite wrote spill.json, flat"
"$PY" "$HERE/gpumem_summary.py" "$T/gpumem_spill.csv" "$T/s.json"; chk 1 $? "370 -> 698 MiB is SPILL (the prior A at GPU_UTIL=0.93)"
echo "== 4c. vram.py takes the GPU from CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=GPU-abc,GPU-def "$PY" -c "import vram; assert vram.gpu()=='GPU-abc'; print('gpu', vram.gpu())"; chk 0 $? "first CUDA_VISIBLE_DEVICES entry"
echo "== 5. vram.py wrapper without nvidia-smi"
"$PY" "$HERE/vram.py" "$T/v.json" -- true; chk 0 $? "vram.py exits with the wrapped command's status"; cat "$T/v.json"; echo
[ $fail = 0 ] && echo "SELFCHECK PASS" || { echo "SELFCHECK FAIL"; exit 1; }
