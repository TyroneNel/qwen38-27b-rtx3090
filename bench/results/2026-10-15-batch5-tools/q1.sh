#!/bin/bash
# Batch 5 Q1: the unadvertised mamba knobs on the band-2 profile (int8 CTX=fast, DFLASH_TOKENS=3, KV_MEM=5583457484,
# MAX_SEQS=16). One variable per boot, appended to EXTRA_ARGS. Baseline first and last (A-B-A):
#   a1  (none)                                b   --mamba-ssu-algorithm horizontal     c   --mamba-ssu-algorithm vertical
#   d   --mamba-backend flashinfer            a2  (none, the closing baseline)
#   bf  --mamba-backend flashinfer --mamba-ssu-algorithm horizontal   } only if b/c refuse to boot and d boots:
#   cf  --mamba-backend flashinfer --mamba-ssu-algorithm vertical     } config/mamba.py rejects an SSU algorithm
#                                                                       without the FlashInfer backend
# Full arm: boot-log kernel lines, the held-load Running: probe (16 x ~1k), conc_ladder --n 1,4,8,16, run_benchmarks
# single twice (run 2 is the C1 T=0 number), down, memory summary. bf/cf (the add-on pair): kernel lines and the two
# benchmark runs only.
#   bash bench/results/2026-10-15-batch5-tools/q1.sh [arm ...]   (default: a1 b c d bf cf a2)
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch5-tools
A="bash $T/arm.sh"
BASE="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
booted() { [ -f "bench/results/2026-10-15-$1/.booted" ]; }
for arm in ${@:-a1 b c d bf cf a2}; do
  case $arm in
    a1) id=Q1-a1-baseline;     x="" ;;
    a2) id=Q1-a2-baseline;     x="" ;;
    b)  id=Q1-b-ssu-horizontal; x="--mamba-ssu-algorithm horizontal" ;;
    c)  id=Q1-c-ssu-vertical;   x="--mamba-ssu-algorithm vertical" ;;
    d)  id=Q1-d-backend-flashinfer; x="--mamba-backend flashinfer" ;;
    bf) id=Q1-bf-flashinfer-horizontal; x="--mamba-backend flashinfer --mamba-ssu-algorithm horizontal" ;;
    cf) id=Q1-cf-flashinfer-vertical;   x="--mamba-backend flashinfer --mamba-ssu-algorithm vertical" ;;
  esac
  if [ "$arm" = bf ] || [ "$arm" = cf ]; then
    if booted Q1-b-ssu-horizontal || booted Q1-c-ssu-vertical || ! booted Q1-d-backend-flashinfer; then
      echo "Q1 $arm skipped: b/c booted or d did not"; continue; fi
  fi
  d=bench/results/2026-10-15-$id; l=q1$arm
  mkdir -p "$d"
  sed -e 's/^DFLASH_TOKENS=.*/DFLASH_TOKENS=3/' -e "s|^EXTRA_ARGS=.*|EXTRA_ARGS='$BASE${x:+ $x}'|" $T/env.base > "$d/.env"
  if ! $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK; then
    echo "Q1 $arm boot failed"; $A down "$d" "$l"
    grep -aE "Error|error:|refus|not supported|Traceback" "$d/raw-boot-$l.log" | sed 's/^single-1  | //' | cut -c1-600 | tail -12 > "$d/raw-bootfail-$l.log"
    continue
  fi
  touch "$d/.booted"
  grep -aiE 'non-default args|GDN decode kernel|mamba|ssu|flashinfer|warmup|GPU KV cache size|Maximum concurrency|scratch|start_qwen\] note|fla|kernel' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
  if [ "$arm" != bf ] && [ "$arm" != cf ]; then
    $A probe "$d" "$l" 1 16 1024
    $A ladder "$d" "$l" --n 1,4,8,16
  fi
  $A bench "$d" "$l" 1
  $A bench "$d" "$l" 2
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q1_DONE
