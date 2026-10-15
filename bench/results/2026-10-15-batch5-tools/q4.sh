#!/bin/bash
# Batch 5 Q4: IFBench prompt-level strict (all 299 prompts of the pinned AllenAI checkout), docs/quality.md protocol
# (thinking on, reasoning_effort field omitted = the template default xhigh, model-default sampling, max_tokens
# 16384), 4 client threads, on two boots at the shipped pin KV_MEM=5583457484:
#   q4s  the shipped profile: SPEC=dflash2, CTX=fast, DFLASH_TOKENS=7, no EXTRA_ARGS, MAX_SEQS unset (launcher default 8)
#   q4b  band 2: DFLASH_TOKENS=3 + EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head',
#        MAX_SEQS=16
# IFBench is not in the image: installed per boot into /tmp in the container by ifbench_setup.sh (commit pinned there).
#   bash bench/results/2026-10-15-batch5-tools/q4.sh [arm ...]   (default: q4b q4s)
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-15-batch5-tools
A="bash $T/arm.sh"
DC=(docker compose --profile single)
IFD=/tmp/IFBench/ifbench/data/IFBench_test.jsonl
for l in ${@:-q4b q4s}; do
  case $l in
    q4s) id=Q4-shipped; e='s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=7/; /^EXTRA_ARGS=/d; /^MAX_SEQS=/d; /^# the cap lifted/d' ;;
    q4b) id=Q4-band2;   e='s/^DFLASH_TOKENS=.*/DFLASH_TOKENS=3/' ;;
  esac
  d=bench/results/2026-10-15-$id
  mkdir -p "$d"
  sed -e "$e" $T/env.base > "$d/.env"
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "$l boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'non-default args|GPU KV cache size|Maximum concurrency|kv_cache_dtype|start_qwen\]' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2500 > "$d/raw-bootlines-$l.log"
  # the guard's first-request marker under the boot label (exec writes it under its own name)
  date +%Y-%m-%dT%H:%M:%S > "$d/.firstreq-$l"; echo "$(date -u +%FT%TZ) [$l] first measurement request at $(cat "$d/.firstreq-$l") local (IFBench)" >> "$d/cmd.txt"; sleep 22
  "${DC[@]}" cp $T/ifbench_setup.sh single:/tmp/ifbench_setup.sh
  "${DC[@]}" cp $T/ifbench_gen.py single:/tmp/ifbench_gen.py
  $A exec "$d" "ifbench-setup-$l" "bash /tmp/ifbench_setup.sh"
  curl -s http://127.0.0.1:18020/metrics | grep -E "^vllm:(num_preemptions_total|generation_tokens_total|request_success_total)" > "$d/raw-ifbench-metrics-before-$l.log"
  $A exec "$d" "ifbench-gen-$l" "cd /app && venv/bin/python /tmp/ifbench_gen.py $l --data $IFD --out /tmp/ifb-$l --max-tokens 16384 --conc 4"
  curl -s http://127.0.0.1:18020/metrics | grep -E "^vllm:(num_preemptions_total|generation_tokens_total|request_success_total)" > "$d/raw-ifbench-metrics-after-$l.log"
  $A exec "$d" "ifbench-eval-$l" "cd /tmp/IFBench && NLTK_DATA=/tmp/nltk_data /tmp/ifb/bin/python -m run_eval --input_data=$IFD --input_response_data=/tmp/ifb-$l.jsonl --output_dir=/tmp/ifb-eval-$l"
  "${DC[@]}" cp single:/tmp/ifb-$l-full.json "$d/raw-ifbench-$l.json"
  "${DC[@]}" cp single:/tmp/ifb-eval-$l "$d/raw-ifbench-eval-$l"
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q4_DONE
