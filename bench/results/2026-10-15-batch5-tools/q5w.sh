#!/bin/bash
# Batch 5 Q5 add-on: the teacher-forced replay on a MAX_SEQS=16 boot. Q5-a's free-run C1 T=0 fell with tokens/step
# (3.17 against Q3-a's 3.34 at MAX_SEQS=8) at the same N=1 ms/pass, so the X-3 loss looks like an acceptance shift,
# the same shape as X-1. This boot (Q5-a's env, a second boot) replays Q3's frozen targets (captured on the MAX_SEQS=8
# CG=64 boot) with workload_accept.py, so MAX_SEQS=16 and 8 are compared on identical token sequences.
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch5-tools
A="bash $T/arm.sh"
DC=(docker compose --profile single)
TASKS=gsm8k,code,chat,real0,real1,real2,real3,real4,real5,real6,real7
TD=bench/results/2026-10-15-Q3-a-cg64/targets
d=bench/results/2026-10-15-Q5-a2-k7-s16-wacc; l=q5w
[ -d "$TD" ] || { echo "q5w: no frozen targets at $TD"; exit 1; }
mkdir -p "$d"; cp bench/results/2026-10-15-Q5-a-k7-s16/.env "$d/.env"
$A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "$l boot failed"; $A down "$d" "$l"; exit 1; }
grep -aiE 'non-default args|cudagraph|max_num_seqs|scratch|capacity|GPU KV cache size|GDN decode kernel|start_qwen\]' "$d/raw-boot-$l.log" \
  | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
$A bench "$d" "$l" 1
"${DC[@]}" cp $T/workload_accept.py single:/tmp/workload_accept.py
"${DC[@]}" cp "$TD" single:/tmp/q3-targets
$A exec "$d" "wacc-$l" "cd /app && venv/bin/python /tmp/workload_accept.py $l --targets /tmp/q3-targets --tasks $TASKS --max-tokens 1024 --chunk 128 --decode-reps 2"
"${DC[@]}" cp single:/tmp/q3-targets/result_$l.json "$d/raw-wacc-result-$l.json"
$A bench "$d" "$l" 2
$A down "$d" "$l"
bash $T/memsum.sh "$d" "$l"
echo Q5W_DONE
