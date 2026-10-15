#!/bin/bash
# Batch 5 Q3 (the CG=40 mechanism, X-1) and Q5 (X-3, the 16-seat C1 loss). Batch 2 S2b's base for both: int8 CTX=fast,
# DFLASH_TOKENS=7, KV_MEM=5315022028 (the value batch 2 S2b and batch 4 ran at), MAX_LEN=65536.
#   q3a  MAX_SEQS=8, CG unset (the launcher's min(64, 8*(7+1)) = 64)  -> also Q5's B arm (MAX_SEQS=8)
#   q3b  MAX_SEQS=8, CG=40
#   q5a  MAX_SEQS=16, CG unset (min(64, 16*8) = 64)                     -> Q5's A arm
# q3a/q3b: run_benchmarks single (run 1 = warm-up and the guard's first-request marker), workload_accept.py on 11
# prompts (gsm8k, code, chat, the 8 prompts_real.jsonl C1 prompts) against ONE frozen target set captured on q3a
# and replayed unchanged on q3b, run_benchmarks single again (run 2 = C1 T=0), conc_ladder --n 1 (ms/pass at N=1).
# q5a: run_benchmarks single twice and conc_ladder --n 1.
#   bash bench/results/2026-10-15-batch5-tools/q35.sh [arm ...]   (default: q3a q3b q5a)
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch5-tools
A="bash $T/arm.sh"
TGT=/app/bench/results/2026-10-15-Q3-a-cg64/targets
TASKS=gsm8k,code,chat,real0,real1,real2,real3,real4,real5,real6,real7
for l in ${@:-q3a q3b q5a}; do
  case $l in
    q3a) id=Q3-a-cg64; e='s/^MAX_SEQS=.*/MAX_SEQS=8/' ;;
    q3b) id=Q3-b-cg40; e='s/^MAX_SEQS=.*/MAX_SEQS=8/; $a CG=40' ;;
    q5a) id=Q5-a-k7-s16; e='s/^MAX_SEQS=.*/MAX_SEQS=16/' ;;
  esac
  d=bench/results/2026-10-15-$id
  mkdir -p "$d"
  sed -e 's/^DFLASH_TOKENS=.*/DFLASH_TOKENS=7/' -e 's/^KV_MEM=.*/KV_MEM=5315022028/' -e "$e" $T/env.base > "$d/.env"
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "$l boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'non-default args|cudagraph|capture|max_num_seqs|max_num_batched|scratch|capacity|buffer|GPU KV cache size|Maximum concurrency|GDN decode kernel|start_qwen\]' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
  $A bench "$d" "$l" 1
  if [ "$l" != q5a ]; then
    $A exec "$d" "wacc-$l" "cd /app && venv/bin/python /app/$T/workload_accept.py $l --targets $TGT --tasks $TASKS --max-tokens 1024 --chunk 128 --decode-reps 2"
  fi
  $A bench "$d" "$l" 2
  $A ladder "$d" "$l" --n 1
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q35_DONE
