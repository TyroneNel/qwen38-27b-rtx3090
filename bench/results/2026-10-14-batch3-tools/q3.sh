#!/bin/bash
# Batch 3 Q3 (+ Q4's --help=all check): one boot per flag on the Q1-a config (int8 base, DFLASH_TOKENS=7), compared
# with the Q1-a boot (bench/results/2026-10-14-Q1-k7). The finding is the boot log: the non-default args line, the
# attention block size / mamba page padding / group size / KV cache size lines. One Running: probe wave per boot
# rides along so a flag that changed the pool shows it in the same metric as Q1.
#   bash bench/results/2026-10-14-batch3-tools/q3.sh [label ...]   (default: q3ckpt q3blk)
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
T=bench/results/2026-10-14-batch3-tools
A="bash $T/arm.sh"
for l in ${@:-q3ckpt q3blk}; do
  case $l in
    q3ckpt) d=bench/results/2026-10-14-Q3-shared-prefix-ckpt; f='--enable-mamba-shared-prefix-checkpoint' ;;
    q3blk)  d=bench/results/2026-10-14-Q3-mamba-block-8192;   f='--mamba-block-size 8192' ;;
  esac
  mkdir -p "$d"
  sed "s/^EXTRA_ARGS='\(.*\)'$/EXTRA_ARGS='\1 $f'/" $T/env.q1-base > "$d/.env.$l"
  $A boot "$d" "$l" | tail -1 | grep -q BOOT_OK || { echo "Q3 $l boot failed"; $A down "$d" "$l"; continue; }
  grep -aiE 'resident requests|GPU KV cache size|mamba|block size|page size|group_size|GDN|non-default args' "$d/raw-boot-$l.log" \
    | sed 's/^single-1  | //' | cut -c1-2000 > "$d/raw-bootlines-$l.log"
  for w in 1 2; do $A probe "$d" "$l" $w | grep HELD | tail -2; done
  [ "$l" = q3ckpt ] && $A exec "$d" "help-all-$l" \
    'cd /app && echo "## vllm serve --help=all | grep -inE replay|mamba|ssm"; venv/bin/vllm serve --help=all 2>&1 | grep -inE "replay|mamba|ssm"; echo "## count of --help=all lines matching replay:"; venv/bin/vllm serve --help=all 2>&1 | grep -ic replay; echo "## replayssm in the installed vllm source"; P=$(venv/bin/python -c "import vllm,os;print(os.path.dirname(vllm.__file__))"); grep -rn -A3 "use_replayssm:" $P --include=*.py | head -20'
  $A down "$d" "$l"
  bash $T/memsum.sh "$d" "$l"
done
echo Q3_DONE
