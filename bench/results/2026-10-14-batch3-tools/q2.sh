#!/bin/bash
# Batch 3 Q2: recurrent-state dtype, A = --mamba-ssm-cache-dtype float16 (shipped), B = bfloat16, interleaved A-B-A
# on the Q1 int8 base at DFLASH_TOKENS=7 (same KV_MEM, MAX_LEN, MAX_SEQS, stats interval as Q1).
# Per boot: the boot-log lines (GDN decode path, args), three Running: probe waves, conc_ladder --n 1,8 (ms/pass and
# tok/step at N=1 and N=8), run_benchmarks.sh single twice (the second run is C1 T=0). On qA1 and qB only, after
# all timing: GSM8K n=200 (quality_battery.py, thinking off, as batch 2 S1), then the three perplexity texts with
# the spill guard in watch-only mode, last on the boot.
#   bash bench/results/2026-10-14-batch3-tools/q2.sh [label ...]   (default: qA1 qB qA2)
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-14-batch3-tools
A="bash $T/arm.sh"
QD='mkdir -p /tmp/qd && ln -sfn /app/models/quality-data/gsm8k /app/models/quality-data/wikitext /app/models/quality-data/fineweb2 /tmp/qd/ && cd /app && QUALITY_DATA=/tmp/qd'
for l in ${@:-qA1 qB qA2}; do
  case $l in qA*) d=bench/results/2026-10-14-Q2-fp16;; qB*) d=bench/results/2026-10-14-Q2-bf16;; esac
  mkdir -p "$d"
  case $l in
    qA*) cp $T/env.q1-base "$d/.env.$l" ;;
    qB*) sed "s/^EXTRA_ARGS='\(.*\)'$/EXTRA_ARGS='\1 --mamba-ssm-cache-dtype bfloat16'/" $T/env.q1-base > "$d/.env.$l" ;;
  esac
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "Q2 $l boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'resident requests|GPU KV cache size|mamba|block size|page size|GDN|non-default args' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2000 > "$d/raw-bootlines-$l.log"
  grep -aE 'GDN decode|Falling back' "$d/raw-bootlines-$l.log"
  for w in 1 2 3; do $A probe "$d" "$l" $w | grep -c HELD; done
  $A ladder "$d" "$l" --n 1,8
  $A bench "$d" "$l" 1
  $A bench "$d" "$l" 2
  case $l in qA1|qB)
    $A exec "$d" "gsm-$l" "$QD venv/bin/python bench/quality_battery.py q2-$l --gsm-only --gsm-n 200" | tail -3
    $A watchonly "$d" "$l" on
    $A exec "$d" "ppl-$l" "$QD venv/bin/python bench/quality_battery.py q2-$l --ppl-only" | tail -6
    sleep 25  # two more counter samples after the step, so the guard and mem.txt see its tail
    $A watchonly "$d" "$l" off ;;
  esac
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q2_DONE
