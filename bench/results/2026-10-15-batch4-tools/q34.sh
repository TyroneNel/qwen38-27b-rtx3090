#!/bin/bash
# Batch 4 Q3 and Q4, one boot each, same base as Q2 (int8 CTX=fast, KV_MEM=5315022028).
#   q3: SPEC=off + --use-replayssm (EXTRA_ARGS). Its A arm is the Q2 SPEC=off boot (2026-10-15-Q2-off), which differs
#       only by this flag. Boot lines (incl. every ReplaySSM/replay mention), a 16-request probe at ~4k, conc_ladder
#       --n 1,2,4,8,16. If the engine refuses the flag the boot fails and the log is the result.
#   q4: DFLASH_TOKENS=1 MAX_SEQS=24: the Running: probe at 24 requests (~1k and ~4k prompts), conc_ladder --n 8,16,24.
#   bash bench/results/2026-10-15-batch4-tools/q34.sh [q3] [q4]
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch4-tools
A="bash $T/arm.sh"
for q in ${@:-q3 q4}; do
  case $q in
    q3) d=bench/results/2026-10-15-Q3-replayssm; l=q3rssm; mkdir -p "$d"
        sed -e 's/^SPEC=.*/SPEC=off/' -e '/^DFLASH_TOKENS=/d' \
            -e "s/^EXTRA_ARGS=.*/EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head --use-replayssm'/" \
            $T/env.base > "$d/.env.$l" ;;
    q4) d=bench/results/2026-10-15-Q4-k1-s24; l=q4k1s24; mkdir -p "$d"
        sed -e 's/^DFLASH_TOKENS=.*/DFLASH_TOKENS=1/' -e 's/^MAX_SEQS=.*/MAX_SEQS=24/' $T/env.base > "$d/.env.$l" ;;
  esac
  if ! $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK; then
    echo "$q boot failed"; grep -aiE 'replay|error|Error' "$d/raw-boot-$l.log" | sed 's/^single-1  | //' | cut -c1-600 | tail -40 > "$d/raw-bootfail-$l.log"
    $A down "$d" "$l"; bash $T/memsum.sh "$d" "$l"; continue
  fi
  grep -aiE 'resident requests|GPU KV cache size|mamba|replay|block size|page size|GDN|non-default args|speculative|effective-config' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
  if [ $q = q3 ]; then
    $A probe "$d" "$l" 1 16 4096
    $A ladder "$d" "$l" --n 1,2,4,8,16
  else
    $A probe "$d" "$l" 1 24 1024
    $A probe "$d" "$l" 2 24 4096
    $A ladder "$d" "$l" --n 8,16,24
  fi
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q34_DONE
