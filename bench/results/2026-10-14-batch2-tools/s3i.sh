#!/bin/bash
# S3b IFBench boot, after the first attempt on the bf16 base thrashed (b1: 32 client threads against ~4 resident
# requests, KV pool 97% full, 1,017 preemptions, ~80/299 prompts in 55 min; see ../2026-10-14-S3b-effort/cmd.txt).
#   s3i.sh <label> <first>   one boot of 2026-10-14-S3b-effort/.env.I: S2a's B (TRITON_ATTN + int8_per_token_head,
#                            dynamic K [[1,3,7],[4,512,3]], KV_MEM=5315022028). IFBench 299 prompts per effort, <first>
#                            first, client concurrency 4 (the resident limit, so nothing queues into preemption).
# Speculative decoding with probabilistic draft sampling is lossless, and S1 showed the int8 KV format
# quality-equivalent at n=200, so the base change moves speed, not what is scored.
set -u
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-14-batch2-tools
D=bench/results/2026-10-14-S3b-effort
label=$1; first=${2:-xhigh}; second=low; [ "$first" = low ] && second=xhigh
fin() { bash $T/arm.sh down $D $label; bash $T/memsum.sh $D $label; }
chk() { bash $T/arm.sh spilled $D $label && { echo "SPILL $label: $(cat $D/SPILL-$label)"; fin; exit 3; }; return 0; }
cpout() { MSYS_NO_PATHCONV=1 docker compose --profile single cp "single:$1" "$2"; }
bash $T/arm.sh boot $D $label I | tail -1 | grep -q BOOT_OK || { echo "BOOT FAILED $label"; bash $T/arm.sh down $D $label; exit 1; }
sleep 25; chk
bash $T/arm.sh cpin $D $label $T/ifbench_setup.sh
bash $T/arm.sh cpin $D $label $T/ifbench_gen.py
bash $T/arm.sh exec $D ifbench-setup-$label "bash /tmp/ifbench_setup.sh"; chk
IFD=/tmp/IFBench/ifbench/data/IFBench_test.jsonl
for e in $first $second; do
  eff=""; [ "$e" = low ] && eff="--effort low"   # xhigh = the template default, i.e. the field omitted
  bash $T/arm.sh exec $D ifbench-gen-$e-$label "cd /app && venv/bin/python /tmp/ifbench_gen.py $label-$e --data $IFD --out /tmp/ifb-$e $eff --max-tokens 16384 --conc 4"; chk
  bash $T/arm.sh exec $D ifbench-eval-$e-$label "cd /tmp/IFBench && NLTK_DATA=/tmp/nltk_data /tmp/ifb/bin/python -m run_eval --input_data=$IFD --input_response_data=/tmp/ifb-$e.jsonl --output_dir=/tmp/ifb-eval-$e"
  cpout /tmp/ifb-$e-full.json $D/raw-ifbench-$e-$label.json
  cpout /tmp/ifb-eval-$e $D/raw-ifbench-eval-$e-$label
  curl -s http://127.0.0.1:18020/metrics | grep -E "^vllm:(num_preemptions_total|generation_tokens_total|request_success_total)" > $D/raw-ifbench-metrics-$e-$label.log
done
fin
