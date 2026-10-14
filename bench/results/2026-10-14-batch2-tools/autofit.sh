#!/bin/bash
# autofit.sh <session-dir>[,<dir>...] <label> <min-max-len> -- <driver command...>
# Runs one driver boot and, if spillwatch.sh ended it with a spill (driver exit 3) or the engine refused to start
# because MAX_LEN no longer fits the pool, shrinks the memory knobs in EVERY .env.* of <session-dir> (so the arms of
# a pair stay byte-identical) and re-runs the boot, at most 3 times:
#   spill        KV_MEM -= 256 MiB (268,435,456 bytes)
#   pool < len   MAX_LEN -= 8192, never below <min-max-len> (the longest request the session's tests send)
# Every change is appended to <session-dir>/cmd.txt and to <session-dir>/knob-changes.txt with the reason.
# Boots measured before a cut are not comparable with boots after it: the caller re-runs the whole pair.
# Extra comma-separated directories get the same cut: they hold boots compared with these (S2's shared A boots
# live in S2a, its B boots in S2a and S2b, and S2c rides on S2a's A3). The first directory holds the boot logs.
set -u
cd "$(git rev-parse --show-toplevel)"
dirs=${1//,/ }; dir=${dirs%% *}; label=$2; minlen=$3; shift 4
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { echo "$(ts) [autofit] $*" | tee -a "$dir/cmd.txt" "$dir/knob-changes.txt"; }
setk() { local f d; for d in $dirs; do for f in "$d"/.env.*; do sed -i "s/^$1=.*/$1=$2/" "$f"; done; done; }
getk() { grep -h "^$1=" "$dir"/.env.* | head -1 | cut -d= -f2; }
for try in 1 2 3 4; do
  "$@"; rc=$?
  [ $rc -eq 3 ] || [ $rc -eq 1 ] || exit $rc
  boot="$dir/raw-boot-$label.log"
  if [ $rc -eq 1 ] && grep -aqE "KV cache is needed|estimated maximum model length" "$boot" 2>/dev/null; then
    len=$(getk MAX_LEN); est=$(grep -aoE "estimated maximum model length is [0-9]+" "$boot" | head -1 | grep -oE "[0-9]+$")
    new=$(( ${est:-$((len - 8192))} / 1024 * 1024 ))   # vLLM's own estimate for this pool, floored to 1k
    [ $new -lt "$minlen" ] && { note "MAX_LEN $len cannot drop below $minlen for this session's tests; stopping"; exit 1; }
    note "boot $label refused: pool smaller than MAX_LEN=$len ($(grep -aoE 'ValueError.{0,200}' "$boot" | head -1)); MAX_LEN $len -> $new"
    setk MAX_LEN $new
  elif [ $rc -eq 3 ]; then
    kv=$(getk KV_MEM); new=$((kv - 268435456))
    note "boot $label spilled ($(cat "$dir/SPILL-$label")); KV_MEM $kv -> $new in every .env.* of: $dirs"
    setk KV_MEM $new
  else
    exit $rc
  fi
  [ $try -eq 4 ] && { note "gave up after 3 knob changes"; exit 1; }
done
