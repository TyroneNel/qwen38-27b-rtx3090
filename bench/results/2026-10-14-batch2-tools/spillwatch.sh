#!/bin/bash
# spillwatch.sh <dir> <label>: live spill guard for one boot, started by arm.sh boot, ends with the sampler.
# Every 10 s it applies memsum.sh's rule to <dir>/mem-trace-<label>.csv: the loaded-engine floor is the minimum
# Shared Usage over 3090 samples with Dedicated > 20,000 MiB, and a sample more than 16 MiB above it is a spill.
# On the first spill it writes <dir>/SPILL-<label> (the sample and the floor) and kills the benchmark clients in
# the container, so no further minutes are spent measuring a spilled engine; arm.sh refuses to start any
# measurement while that file exists, and the drivers stop the boot.
set -u
dir=$1; label=$2; L=${LUID:-luid_0x00000000_0x00011456_phys_0}
csv="$dir/mem-trace-$label.csv"; flag="$dir/SPILL-$label"; stop="$dir/.sampler-stop-$label"
while [ ! -e "$stop" ]; do
  sleep 10
  [ -f "$csv" ] || continue
  hit=$(awk -F, -v L="$L" '
    $2==L {t=$1; if ($3=="dedicated usage") d[t]=$4/1048576; else s[t]=$4/1048576; ts[t]=1}
    END { floor=-1; n=asorti(ts,k)
      for (i=1;i<=n;i++) { t=k[i]; if (d[t]>20000 && (floor<0 || s[t]<floor)) floor=s[t] }
      for (i=1;i<=n;i++) { t=k[i]; if (floor>=0 && s[t]>floor+16) { printf "%s shared=%.1f MiB dedicated=%.1f MiB floor=%.1f MiB\n", t, s[t], d[t], floor; exit } } }' "$csv")
  if [ -n "$hit" ]; then
    echo "SPILL $label $hit" > "$flag"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$label] SPILL detected: $hit; killing benchmark clients" >> "$dir/cmd.txt"
    MSYS_NO_PATHCONV=1 docker compose --profile single exec -T single pkill -f 'run_benchmarks|vllm bench|conc_ladder|quality_battery|needle_test|/tmp/[a-z0-9_]+\.py' >/dev/null 2>&1
    exit 0
  fi
done
