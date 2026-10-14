#!/bin/bash
# Session 1 driver. Run from the repo root in Git Bash.
#   s1.sh quality <label> <env>   one boot: perplexity (3 texts), GSM8K n=200 thinking off, needle ~60k, down
#   s1.sh timing  <label> <env>   one boot: run_benchmarks single x2 (keep run 2), conc_ladder --n 4,8, down
#   s1.sh gsmrep  <label> <env>   one boot: GSM8K n=200 thinking off twice, the battery and gsm_effort.py --think 0
#                                 (same lane, per-question results kept), down
# quality_battery.py writes result_<tag>.json into QUALITY_DATA; pointing it at a /tmp dir of symlinks keeps that
# write inside the container instead of in G:/models/quality-data.
set -u
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-14-batch2-tools
D=bench/results/2026-10-14-S1-capacity
mode=$1; label=$2; env=$3
bash $T/arm.sh boot $D $label $env | tail -1 | grep -q BOOT_OK || { echo "BOOT FAILED $label"; bash $T/arm.sh down $D $label; exit 1; }
QD='mkdir -p /tmp/qd && ln -sfn /app/models/quality-data/gsm8k /app/models/quality-data/wikitext /app/models/quality-data/fineweb2 /tmp/qd/ && cd /app && QUALITY_DATA=/tmp/qd'
if [ "$mode" = quality ]; then
  bash $T/arm.sh exec $D ppl-$label "$QD venv/bin/python bench/quality_battery.py s1-$label --ppl-only"
  bash $T/arm.sh exec $D gsm-$label "$QD venv/bin/python bench/quality_battery.py s1-$label --gsm-only --gsm-n 200"
  bash $T/arm.sh exec $D needle-$label "cd /app && venv/bin/python bench/needle_test.py 60000"
elif [ "$mode" = gsmrep ]; then
  bash $T/arm.sh exec $D gsm-$label "$QD venv/bin/python bench/quality_battery.py s1-$label --gsm-only --gsm-n 200"
  bash $T/arm.sh cpin $D $label $T/gsm_effort.py
  bash $T/arm.sh exec $D gsmq-$label "cd /app && venv/bin/python /tmp/gsm_effort.py s1-$label --think 0 --max-tokens 768 --n 200"
  MSYS_NO_PATHCONV=1 docker compose --profile single cp single:/tmp/gsm_effort_s1-$label.json $D/raw-gsmq-$label.json
else
  bash $T/arm.sh bench $D $label 1
  bash $T/arm.sh bench $D $label 2
  bash $T/arm.sh ladder $D $label --n 4,8
fi
bash $T/arm.sh down $D $label
bash $T/memsum.sh $D $label
