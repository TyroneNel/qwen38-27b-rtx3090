#!/bin/bash
# spillwatch.sh <dir> <label>: live spill guard for one boot, started by arm.sh boot right after compose up, ends with
# the sampler. Batch 3's guard (../2026-10-14-batch3-tools/spillwatch.sh) with the floor taken where the batch 4 brief
# says, plus the reference that would have caught batch 3's blind spot.
#
# Batch 3's floor was the minimum Shared Usage over every loaded sample (Dedicated > 20,000 MiB). Its three MAX_SEQS=16
# boots rose 86.2 -> 102.2 MiB at the memory-profiling step, before Dedicated passed 20,000 MiB, so every "loaded" sample
# was already at the raised level and the rise became the floor.
#
# Floors here (3090 samples only, Shared > 1 MiB so a CSV row read mid-write, batch 2's false positive, never counts):
#   F_kv  = min Shared over the samples between the engine's "GPU KV cache size" line (<dir>/.kvline-<label>) and the
#           first measurement request (<dir>/.firstreq-<label>). This is the brief's floor, and the floor every
#           measurement spill is called against: two consecutive samples after the first request above F_kv + 16 MiB.
#   F_ctx = min Shared over the samples before the KV-cache line with Dedicated > 1,000 MiB: the CUDA-context level once
#           weights load, before memory profiling (the first context sample can read lower: 82.2 vs 86.2 MiB on q2k7).
#           The profiling step runs BEFORE the KV-cache line is printed, so a rise there sits inside F_kv and the
#           brief's window alone cannot see it. If F_kv > F_ctx + 8 MiB the boot rose more than a MAX_SEQS=8 boot does:
#           BOOTRISE-<label>, RECORDED, NOT BLOCKING. Every clean batch-3 boot (MAX_SEQS=8) steps exactly 8.0 MiB at
#           profiling (84.2 -> 92.2); the MAX_SEQS=16 boots stepped 16.0 (86.2 -> 102.2), which a +16 threshold would
#           have missed by equality (guard-test.txt).
#           Why it does not block: q2k7 (KV_MEM 5315022028, 1,437 MiB free after boot) and q2k7cut1g (the same boot with
#           KV_MEM 1 GiB lower, 2,465 MiB free) both step exactly 86.2 -> 102.2 MiB. A step that does not shrink with
#           1 GiB more free VRAM is not memory pressure; it follows MAX_SEQS (8 -> 8.0, 16 -> 16.0, 24 -> 32.0 MiB in
#           batch 3), i.e. host memory the engine maps per seat [INFERENCE: pinned per-request buffers].
# On a measurement spill it writes <dir>/SPILL-<label> and kills the benchmark clients in the container (unless
# <dir>/.watch-only-<label> exists); arm.sh refuses to start a measurement while SPILL-<label> exists.
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
    MSYS_NO_PATHCONV=1 docker compose --profile single exec -T single pkill -f 'run_benchmarks|vllm bench|conc_ladder|quality_battery|needle_test|/tmp/[a-z0-9_]+\.py' >/dev/null 2>&1
    exit 0
  fi
done
