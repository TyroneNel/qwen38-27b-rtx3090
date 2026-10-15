#!/bin/bash
# Batch 5 Q2: the launcher seat print on hardware. One boot per arm, image hyperqwen:exp-844cd90 (the rebuilt launcher).
# Per boot: the launcher's printed pair (the `[start_qwen] note:` line), the boot-log pool lines, three held-load
# Running: waves with running_probe.py (each saving the full /metrics text at its peak):
#   w1  16 requests at ~1k-token prompts   (the print's short end; 16 = the cap, above every predicted seat count)
#   w2   6 requests at ~32k                (the brief's depth)
#   w3   3 requests at ~61k                (MAX_LEN=65536 minus room for 512 output tokens: the print's long end itself)
# then down and the memory summary.
#   bash bench/results/2026-10-15-batch5-tools/q2.sh [arm ...]   arms: a b c d (default all, in that order)
#   a  int8 CTX=fast  KV_MEM=5583457484  k=7      b  same, k=3
#   c  int8 CTX=fast  KV_MEM=4509715660 (5583457484 - 1 GiB)  k=7
#   d  the shipped KV (no EXTRA_ARGS: what a default user boots)  KV_MEM=5583457484  k=7   [supplementary arm]
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch5-tools
A="bash $T/arm.sh"
for arm in ${@:-a b c d}; do
  case $arm in
    a) id=Q2-a-k7;      e='s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=7/' ;;
    b) id=Q2-b-k3;      e='s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=3/' ;;
    c) id=Q2-c-k7-cut1g; e='s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=7/; s/^KV_MEM=.*/KV_MEM=4509715660/' ;;
    d) id=Q2-d-k7-shippedkv; e='s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=7/; /^EXTRA_ARGS=/d' ;;
  esac
  d=bench/results/2026-10-15-$id; l=q2$arm
  mkdir -p "$d"
  sed -e "$e" $T/env.base > "$d/.env"
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "Q2 $arm boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'start_qwen\] note|resident requests|GPU KV cache size|Maximum concurrency|mamba_block_size|block size|page size|GDN|non-default args|kv_cache_dtype|attention backend|scratch' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
  $A probe "$d" "$l" 1 16 1024
  $A probe "$d" "$l" 2 6 32768
  $A probe "$d" "$l" 3 3 61440
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q2_DONE
