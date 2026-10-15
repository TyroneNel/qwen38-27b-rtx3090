#!/bin/bash
# Batch 4 Q1+Q2: four boots on the int8 CTX=fast base, KV_MEM=5315022028 and MAX_SEQS=16 on all of them, identical except
# the speculation setting: DFLASH_TOKENS=7, 3, 1, then SPEC=off. Per boot: the boot-log lines, the held 8-request Running:
# probe (~1k-token prompts, batch 3's shape), a 16-request probe at ~4k (the ladder's prompt depth, cap not binding),
# conc_ladder --n 1,2,4,8,16, run_benchmarks.sh single twice (the second run is the C1 T=0 number), memory evidence, down.
# The k=7 boot also carries Q1: the same seat probe at ~1k, ~16k and ~32k prompts, each wave saving the full /metrics
# text at its peak.
#   bash bench/results/2026-10-15-batch4-tools/q2.sh [arm ...]     arms: k7 k3 k1 off (default: all four, in that order)
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch4-tools
A="bash $T/arm.sh"
for arm in ${@:-k7 k3 k1 off}; do
  d=bench/results/2026-10-15-Q2-$arm; l=q2$arm
  mkdir -p "$d"
  case $arm in
    off) sed -e 's/^SPEC=.*/SPEC=off/' -e '/^DFLASH_TOKENS=/d' $T/env.base > "$d/.env.$l" ;;
    k*)  sed "s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=${arm#k}/" $T/env.base > "$d/.env.$l" ;;
  esac
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "Q2 $arm boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'resident requests|GPU KV cache size|mamba|block size|page size|GDN|non-default args|speculative|effective-config' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
  $A probe "$d" "$l" 1 8 1024
  $A probe "$d" "$l" 2 16 4096
  if [ "$arm" = k7 ]; then
    $A probe "$d" "$l" 3 16 1024
    $A probe "$d" "$l" 4 6 16384
    $A probe "$d" "$l" 5 4 32768
  fi
  $A ladder "$d" "$l" --n 1,2,4,8,16
  $A bench "$d" "$l" 1
  $A bench "$d" "$l" 2
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q2_DONE
