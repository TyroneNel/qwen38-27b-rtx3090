#!/bin/bash
# Session 3 driver. Run from the repo root in Git Bash.
#   s3.sh a <label>                 S3a: one boot of 2026-10-14-S3a-r0b0tlab/.env.A (shipped CTX=fast, MAX_LEN=8192):
#                                   r0b0_protocol.py chat template x2 reps (the brief's thinking-off row), then their
#                                   literal raw prompt x2 reps, then workload_accept.py gsm8k at 512 new tokens.
#   s3.sh b <label> <first> [wacc]  S3b: one boot of 2026-10-14-S3b-effort/.env.A; GSM8K n=200 thinking on at both
#                                   efforts, <first> (xhigh|low) first; with "wacc", then teacher-forced tok/step for
#                                   both on each arm's own frozen targets (deterministic, so one boot carries it).
#                                   reasoning_effort is a request field, not a boot knob, so the interleave is by
#                                   order inside the boot and across two boots.
# After every step a spill flagged by spillwatch.sh ends the boot at once (exit 3).
set -u
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-14-batch2-tools
mode=$1; label=$2; first=${3:-xhigh}; wacc=${4:-}
D=bench/results/2026-10-14-S3a-r0b0tlab; [ "$mode" = a ] || D=bench/results/2026-10-14-S3b-effort
fin() { bash $T/arm.sh down $D $label; bash $T/memsum.sh $D $label; }
chk() { bash $T/arm.sh spilled $D $label && { echo "SPILL $label: $(cat $D/SPILL-$label)"; fin; exit 3; }; return 0; }
cpout() { MSYS_NO_PATHCONV=1 docker compose --profile single cp "single:$1" "$2"; }
bash $T/arm.sh boot $D $label A | tail -1 | grep -q BOOT_OK || { echo "BOOT FAILED $label"; bash $T/arm.sh down $D $label; exit 1; }
sleep 25; chk
if [ "$mode" = a ]; then
  bash $T/arm.sh cpin $D $label $T/r0b0_protocol.py
  bash $T/arm.sh cpin $D $label $T/workload_accept.py
  bash $T/arm.sh exec $D r0b0-chat-$label "cd /app && venv/bin/python /tmp/r0b0_protocol.py $label-chat --template chat --n 40 --max-tokens 512 --reps 2"; chk
  bash $T/arm.sh exec $D r0b0-raw-$label "cd /app && venv/bin/python /tmp/r0b0_protocol.py $label-raw --template raw --n 40 --max-tokens 512 --reps 2"; chk
  bash $T/arm.sh exec $D wacc-$label "cd /app && venv/bin/python /tmp/workload_accept.py $label --targets /tmp/targets-s3a --tasks gsm8k --max-tokens 512"; chk
  cpout /tmp/r0b0_$label-chat.json $D/raw-r0b0-chat-$label.json
  cpout /tmp/r0b0_$label-raw.json $D/raw-r0b0-raw-$label.json
  cpout /tmp/targets-s3a $D/targets-$label
else
  second=low; [ "$first" = low ] && second=xhigh
  bash $T/arm.sh cpin $D $label $T/gsm_effort.py
  bash $T/arm.sh cpin $D $label $T/workload_accept.py
  for e in $first $second; do
    eff=""; [ "$e" = low ] && eff="--effort low"   # xhigh = the template default, i.e. the field omitted
    bash $T/arm.sh exec $D gsm-$e-$label "cd /app && venv/bin/python /tmp/gsm_effort.py $label-$e --think 1 $eff --n 200 --max-tokens 16384"; chk
    cpout /tmp/gsm_effort_$label-$e.json $D/raw-gsm-$e-$label.json
  done
  if [ "$wacc" = wacc ]; then
    for e in $first $second; do
      eff=""; [ "$e" = low ] && eff="--effort low"
      bash $T/arm.sh exec $D wacc-$e-$label "cd /app && venv/bin/python /tmp/workload_accept.py $label-$e --targets /tmp/targets-$e --think 1 $eff --tasks gsm8k,code,chat"; chk
      cpout /tmp/targets-$e $D/targets-$e-$label
    done
  fi
fi
fin
