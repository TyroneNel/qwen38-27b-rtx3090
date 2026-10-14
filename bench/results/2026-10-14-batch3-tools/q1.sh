#!/bin/bash
# Batch 3 Q1: three boots on the int8 CTX=fast base, identical except DFLASH_TOKENS (7, 3, 1), in that order.
# Per boot: the boot-log lines the brief names, three Running: probe waves, conc_ladder --n 1,2,4,8,16, then
# run_benchmarks.sh single twice (the second run is the C1 T=0 number), memory evidence, down.
#   bash bench/results/2026-10-14-batch3-tools/q1.sh [k ...]     (default: 7 3 1)
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-14-batch3-tools
A="bash $T/arm.sh"
for k in ${@:-7 3 1}; do
  d=bench/results/2026-10-14-Q1-k$k; l=q1k$k
  mkdir -p "$d"
  sed "s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=$k/" $T/env.q1-base > "$d/.env.$l"
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "Q1 k=$k boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'resident requests|GPU KV cache size|mamba|block size|page size|GDN|non-default args' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2000 > "$d/raw-bootlines-$l.log"
  for w in 1 2 3; do $A probe "$d" "$l" $w; done
  $A ladder "$d" "$l" --n 1,2,4,8,16
  $A bench "$d" "$l" 1
  $A bench "$d" "$l" 2
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q1_DONE
