#!/bin/bash
# spillwatch.sh <dir> <label>: the live spill guard for one boot, from the 2026-10-15 batch 4 tools with the logic
# unchanged. sweep.sh starts it right after compose up, and it ends when the sampler's stop file appears.
# On WSL2 an over-committed 3090 does not fail. It spills to system memory and runs 3-6x slower, and the Windows
# adapter counters are the only place that shows it.
#   F_kv  = min Shared Usage between the engine's "GPU KV cache size" line (<dir>/.kvline-<label>) and the first
#           measurement request (<dir>/.firstreq-<label>). A spill is two consecutive samples after the first request
#           with Dedicated > 20,000 MiB and Shared > F_kv + 16 MiB.
#   F_ctx = min Shared Usage before the KV-cache line with Dedicated > 1,000 MiB, the CUDA-context level. F_kv above
#           F_ctx + 8 MiB is recorded as BOOTRISE-<label> and does not block. analyze.py gates the rise itself.
# On a spill it writes <dir>/SPILL-<label> and kills the bench clients in the container (the batch 4 list plus
# labd_bench and api_smoke, which this sweep runs), unless
# <dir>/.watch-only-<label> exists. sweep.sh skips every later cell of a boot with a SPILL file.
set -u
dir=$1; label=$2; L=${LUID:-luid_0x00000000_0x00011456_phys_0}
csv="$dir/mem-trace-$label.csv"; stop="$dir/.sampler-stop-$label"
kvf="$dir/.kvline-$label"; frf="$dir/.firstreq-$label"
bootchecked=0
note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$label] $*" >> "$dir/cmd.txt"; }
while [ ! -e "$stop" ]; do
  sleep 10
  [ -f "$csv" ] && [ -f "$kvf" ] && [ -f "$frf" ] || continue
  out=$(awk -F, -v L="$L" -v KV="$(cat "$kvf")" -v FR="$(cat "$frf")" '
    $2==L {t=$1; if ($3=="dedicated usage") d[t]=$4/1048576; else s[t]=$4/1048576; ts[t]=1}
    END { n=asorti(ts,k); fc=-1; fk=-1; nk=0
      for (i=1;i<=n;i++) { t=k[i]; if (!(s[t]>1)) continue
        if (t<KV) { if (d[t]>1000 && (fc<0 || s[t]<fc)) fc=s[t] }
        else if (t<FR) { nk++; if (fk<0 || s[t]<fk) fk=s[t] } }
      printf "FLOORS F_ctx=%.1f F_kv=%.1f n_kv=%d\n", fc, fk, nk
      if (fk<0) exit
      prev=""
      for (i=1;i<=n;i++) { t=k[i]
        if (t>=FR && d[t]>20000 && s[t]>fk+16) {
          if (prev!="") { printf "SPILL %s and %s shared=%.1f MiB dedicated=%.1f MiB F_kv=%.1f MiB\n", prev, t, s[t], d[t], fk; exit }
          prev=t
        } else prev="" } }' "$csv")
  floors=$(echo "$out" | grep '^FLOORS')
  hit=$(echo "$out" | grep '^SPILL')
  if [ "$bootchecked" = 0 ] && echo "$floors" | grep -q 'F_kv=[0-9]'; then
    bootchecked=1
    note "guard floors: $floors (window: KV-cache line $(cat "$kvf") -> first request $(cat "$frf"), local time)"
    rise=$(echo "$floors" | awk '{split($2,a,"="); split($3,b,"="); if (a[2]>=0 && b[2]>a[2]+8) printf "%.1f", b[2]-a[2]}')
    if [ -n "$rise" ]; then
      echo "BOOTRISE $label $floors: the boot rose ${rise} MiB above the CUDA-context level before the first request" > "$dir/BOOTRISE-$label"
      note "BOOT RISE (recorded, not blocking): $(cat "$dir/BOOTRISE-$label")"
    fi
  fi
  if [ -n "$hit" ]; then
    echo "$hit ($floors)" > "$dir/SPILL-$label"
    if [ -e "$dir/.watch-only-$label" ]; then
      note "SPILL detected (watch-only, nothing killed): $hit"
      exit 0
    fi
    note "SPILL detected: $hit; killing benchmark clients"
    MSYS_NO_PATHCONV=1 docker compose --profile single exec -T single pkill -f 'run_benchmarks|vllm bench|conc_ladder|quality_battery|needle_test|labd_bench|api_smoke|/tmp/[a-z0-9_]+\.py' >/dev/null 2>&1
    exit 0
  fi
done
