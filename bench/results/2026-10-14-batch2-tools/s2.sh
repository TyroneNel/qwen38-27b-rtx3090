#!/bin/bash
# Session 2 driver. Run from the repo root in Git Bash.
#   s2.sh <dir> <label> <env> [cohort]
# One boot from <dir>/.env.<env>: run_benchmarks single x2 (keep run 2), conc_ladder --n 4,8; with "cohort",
# also cohort_gate.py at C8 and C16, three repetitions each, logged into the S2c directory. Then down + memsum.
# The A boots are shared between S2a and S2b: A1 Bdk A2 Bcg A3 puts an A boot on each side of each B.
set -u
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-14-batch2-tools
C=bench/results/2026-10-14-S2c-cohort-gate
fin() { bash $T/arm.sh down $dir $label; bash $T/memsum.sh $dir $label; }
# After every step: a spill flagged by spillwatch.sh ends the boot at once (exit 3); the caller re-runs the
# session's boots with a smaller KV_MEM in BOTH arms' .env, so the pair stays byte-identical.
chk() { bash $T/arm.sh spilled $dir $label && { echo "SPILL $label: $(cat $dir/SPILL-$label)"; fin; exit 3; }; return 0; }
bash $T/arm.sh boot $dir $label $env | tail -1 | grep -q BOOT_OK || { echo "BOOT FAILED $label"; bash $T/arm.sh down $dir $label; exit 1; }
sleep 25; chk   # two sampler ticks after health: a spill at boot ends it here, before any measurement
bash $T/arm.sh bench $dir $label 1; chk
bash $T/arm.sh bench $dir $label 2; chk
bash $T/arm.sh ladder $dir $label --n 4,8; chk
if [ "$extra" = cohort ]; then
  bash $T/arm.sh cpin $C $label $T/cohort_gate.py
  # the cohort log lives in the S2c directory but the boot (and its spill flag) in $dir
  bash $T/arm.sh exec $C cohort-$label "cd /app && venv/bin/python /tmp/cohort_gate.py s2c-$label --conc 8,16 --reps 3 --out-dir /tmp/cohort"
  MSYS_NO_PATHCONV=1 docker compose --profile single cp single:/tmp/cohort $C/raw-cohort-$label; chk
fi
fin
