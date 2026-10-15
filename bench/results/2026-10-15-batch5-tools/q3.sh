#!/bin/bash
# Batch 5 Q3 rerun. q35.sh's first Q3 pass ran workload_accept.py from /app/bench/results/..., which does not exist in
# the container: compose does not load docker-compose.override.local.yml (the ./bench bind mount), so the image's own
# /app/bench has no results/ tree and both wacc runs exited at "can't open file" (attempt1-no-harness/). This pass
# copies the harness in with docker compose cp, keeps the frozen targets in /tmp/q3-targets in the container, copies
# them out after q3a (the capture) and back in before q3b, so both arms replay the SAME target ids.
# Same arms, env and steps as q35.sh: bench 1, workload_accept, bench 2, conc_ladder --n 1.
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch5-tools
A="bash $T/arm.sh"
DC=(docker compose --profile single)
TASKS=gsm8k,code,chat,real0,real1,real2,real3,real4,real5,real6,real7
TD=bench/results/2026-10-15-Q3-a-cg64/targets
for l in ${@:-q3a q3b}; do
  case $l in q3a) id=Q3-a-cg64;; q3b) id=Q3-b-cg40;; esac
  d=bench/results/2026-10-15-$id
  mkdir -p "$d/attempt1-no-harness"
  for f in "$d"/* "$d"/.firstreq-* "$d"/.kvline-*; do case "$f" in *attempt1-no-harness|*/.env|*/report.md) ;; *) [ -e "$f" ] && mv "$f" "$d/attempt1-no-harness/";; esac; done
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "$l boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'non-default args|cudagraph|capture|max_num_seqs|max_num_batched|scratch|capacity|buffer|GPU KV cache size|Maximum concurrency|GDN decode kernel|start_qwen\]' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
  $A bench "$d" "$l" 1
  "${DC[@]}" cp $T/workload_accept.py single:/tmp/workload_accept.py
  if [ "$l" = q3b ]; then "${DC[@]}" cp "$TD" single:/tmp/q3-targets; fi
  $A exec "$d" "wacc-$l" "cd /app && venv/bin/python /tmp/workload_accept.py $l --targets /tmp/q3-targets --tasks $TASKS --max-tokens 1024 --chunk 128 --decode-reps 2"
  if [ "$l" = q3a ]; then rm -rf "$TD"; "${DC[@]}" cp single:/tmp/q3-targets "$TD"; else "${DC[@]}" cp single:/tmp/q3-targets/result_$l.json "$d/raw-wacc-result-$l.json"; fi
  $A bench "$d" "$l" 2
  $A ladder "$d" "$l" --n 1
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q3_DONE
