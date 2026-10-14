#!/bin/bash
# spillwatch.sh <dir> <label>: live spill guard for one boot, started by arm.sh boot, ends with the sampler.
# Batch 2's guard (../2026-10-14-batch2-tools/spillwatch.sh) unchanged except for the watch-only switch below.
# Every 10 s it reads <dir>/mem-trace-<label>.csv. The loaded-engine floor is the minimum Shared Usage over 3090
# samples with Dedicated > 20,000 MiB and Shared > 1 MiB. A timestamp read while the sampler has written its dedicated
# row but not yet its shared row has no shared value (it counts as 0); that read race made the floor 0.0 at
# 2026-10-14T17:41Z and flagged a clean boot. A spill is two consecutive loaded samples above floor+16 MiB.
# On a spill it writes <dir>/SPILL-<label> (the samples and the floor) and kills the benchmark clients in the
# container, so no further minutes are spent measuring a spilled engine; arm.sh refuses to start any measurement
# while that file exists, and the drivers stop the boot.
# Watch-only: while <dir>/.watch-only-<label> exists, a spill is recorded (SPILL-<label>, cmd.txt) but nothing is
# killed. Batch 3 sets it for the perplexity step only, which runs last on its boot: perplexity is known to spill
# this box at the shipped pin (gotcha 60), its numbers do not depend on speed, and the brief asks whether the
# floor held during it rather than for the step to be cut short.
set -u
dir=$1; label=$2; L=${LUID:-luid_0x00000000_0x00011456_phys_0}
csv="$dir/mem-trace-$label.csv"; flag="$dir/SPILL-$label"; stop="$dir/.sampler-stop-$label"
while [ ! -e "$stop" ]; do
  sleep 10
  [ -f "$csv" ] || continue
  hit=$(awk -F, -v L="$L" '
    $2==L {t=$1; if ($3=="dedicated usage") d[t]=$4/1048576; else s[t]=$4/1048576; ts[t]=1}
    END { floor=-1; n=asorti(ts,k)
      for (i=1;i<=n;i++) { t=k[i]; if (d[t]>20000 && s[t]>1 && (floor<0 || s[t]<floor)) floor=s[t] }
      prev=""
      for (i=1;i<=n;i++) { t=k[i]
        if (floor>=0 && d[t]>20000 && s[t]>floor+16) {
          if (prev!="") { printf "%s and %s shared=%.1f MiB dedicated=%.1f MiB floor=%.1f MiB\n", prev, t, s[t], d[t], floor; exit }
          prev=t
        } else prev="" } }' "$csv")
  if [ -n "$hit" ]; then
    echo "SPILL $label $hit" > "$flag"
    if [ -e "$dir/.watch-only-$label" ]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$label] SPILL detected (watch-only, nothing killed): $hit" >> "$dir/cmd.txt"
      exit 0
    fi
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$label] SPILL detected: $hit; killing benchmark clients" >> "$dir/cmd.txt"
    MSYS_NO_PATHCONV=1 docker compose --profile single exec -T single pkill -f 'run_benchmarks|vllm bench|conc_ladder|quality_battery|needle_test|/tmp/[a-z0-9_]+\.py' >/dev/null 2>&1
    exit 0
  fi
done
