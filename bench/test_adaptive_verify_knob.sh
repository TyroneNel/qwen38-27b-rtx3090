#!/bin/bash
# CPU test for ADAPTIVE_VERIFY (single-user/start_qwen.sh, the spec-1 / runtime-1 experiment): unset, empty or 0
# changes nothing, argv and exported environment byte for byte, also against the launcher at git HEAD (or
# BASE_LAUNCHER=<file>) when one is available; 1 under SPEC=dflash2 adds "enable_adaptive_verification":true to the
# --speculative-config JSON with every other key unchanged and exports VLLM_GDN_ADAPTIVE_VERIFY=1 and
# VLLM_FA2_VARLEN_VERIFY=1 (an explicit VLLM_FA2_VARLEN_VERIFY=0 is kept); it composes with DYNAMIC_K; SPEC=off
# ignores it with a warning; SPEC=mtp, CTX=huge with prefix caching on, DFLASH_TOKENS>7 with the lookup on, and a
# value other than 0/1 refuse it. Copies the checkout to a temp dir and runs the launcher against a stub vllm that
# prints its environment and argv. No GPU, no model, no network.
#
#   bash bench/test_adaptive_verify_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
# The stub prints the exported environment (sorted, "ENV " prefixed), then the argv one per line.
printf '#!/bin/sh\nenv | LC_ALL=C sort | grep -v "^_=" | sed "s/^/ENV /"\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"
chmod +x "$T/venv/bin/vllm"
# The launcher to compare the unset knob against: BASE_LAUNCHER, else git HEAD's, else none.
BASE=""
if [ -n "${BASE_LAUNCHER:-}" ]; then BASE=$BASE_LAUNCHER
elif git -C "$REPO" show HEAD:single-user/start_qwen.sh >"$T/base_launcher.sh" 2>/dev/null; then BASE="$T/base_launcher.sh"
fi
FAILS=0
KEY=enable_adaptive_verification; D=DRAFT=/app/models/stub-drafter; K='[[1,3,7],[4,512,3]]'
UNSET="-u ADAPTIVE_VERIFY -u VLLM_GDN_ADAPTIVE_VERIFY -u VLLM_FA2_VARLEN_VERIFY -u DYNAMIC_K -u EXTRA_ARGS -u PRINT_ARGV
  -u PREFIX_CACHE -u DFLASH_TOKENS -u DRAFT_METHOD -u LOOKUP -u LOOKUP_ADAPTIVE -u VLLM_DFLASH2_LOOKUP
  -u VLLM_DFLASH2_LOOKUP_ADAPTIVE -u CONF_TRIM -u CTX -u SPEC -u PTH_FA -u FA2_FP8KV -u KV_OFFLOAD_GB -u CUDA_VISIBLE_DEVICES"
# launch [@base] <env assignments...>: the stub's output on stdout (minus the ADAPTIVE_VERIFY assignment itself,
# which the stub inherits from this test); stderr in $T/err, exit code in $T/rc.
# shellcheck disable=SC2086
launch() { local sc=single-user/start_qwen.sh
  if [ "${1:-}" = @base ]; then shift; cp "$BASE" "$T/single-user/start_qwen.base.sh"; sc=single-user/start_qwen.base.sh; fi
  (cd "$T" && env $UNSET VLLM_API_KEY=k "$@" bash "$sc" 2>"$T/err" </dev/null; echo $? >"$T/rc") | grep -v '^ENV ADAPTIVE_VERIFY='; }
cfg() { launch "$@" | sed -n '/^--speculative-config$/{n;p;}'; }
envv() { launch "${@:2}" | sed -n "s/^ENV $1=//p" | grep . || echo unset; }
# rest: the stub output without the speculative config value and without the two vLLM knobs.
rest() { launch "$@" | sed '/^--speculative-config$/{n;d;}' | grep -v -e '^ENV VLLM_GDN_ADAPTIVE_VERIFY=' -e '^ENV VLLM_FA2_VARLEN_VERIFY=' -e '^ENV ADAPTIVE_VERIFY='; }
field() { python3 -I -c 'import json, sys
try: d = json.loads(sys.stdin.read())
except ValueError: print("invalid"); sys.exit()
k = sys.argv[1]; print(json.dumps(d[k], separators=(",", ":")) if k in d else "absent")' "$1"; }
# same <without> <with>: True when the two JSON configs are equal once the second loses $KEY.
same() { printf '%s\n%s\n' "$1" "$2" | python3 -I -c 'import json, sys
a, b = map(json.loads, sys.stdin.read().splitlines()); b.pop(sys.argv[1], None); print(a == b)' "$KEY" 2>/dev/null; }
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$what" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
eq() { [ "$1" = "$2" ] && echo yes || echo no; }
# started <stub output>: yes when the stub vllm ran (it prints the environment); launcher notes alone are not a start.
started() { grep -q '^ENV ' <<<"$1" && echo yes || echo no; }

ARMS=("SPEC=dflash2 CTX=fast $D" "SPEC=dflash2 CTX=huge PREFIX_CACHE=0 $D" "SPEC=dflash2 CTX=fast $D DFLASH_TOKENS=15"
      "SPEC=mtp CTX=fast" "SPEC=off CTX=fast")
echo "== unset, empty and 0 change nothing (argv and environment)"
for arm in "${ARMS[@]}"; do
  # shellcheck disable=SC2086
  u=$(launch $arm); urc=$(cat "$T/rc")
  check 0 "$urc" "$arm: unset starts"
  # shellcheck disable=SC2086
  check yes "$(eq "$u" "$(launch $arm ADAPTIVE_VERIFY=)")"  "$arm: ADAPTIVE_VERIFY= as unset"
  # shellcheck disable=SC2086
  check yes "$(eq "$u" "$(launch $arm ADAPTIVE_VERIFY=0)")" "$arm: ADAPTIVE_VERIFY=0 as unset"
  if [ -n "$BASE" ]; then
    # shellcheck disable=SC2086
    b=$(launch @base $arm)
    check yes "$(eq "$u" "$(sed 's#start_qwen.base.sh#start_qwen.sh#g' <<<"$b")")" "$arm: unset as the base launcher"
  fi
done
[ -n "$BASE" ] || echo "  (no git HEAD and no BASE_LAUNCHER: base-launcher comparison skipped)"

echo "== SPEC=dflash2 CTX=fast, ADAPTIVE_VERIFY=1"
c0=$(cfg SPEC=dflash2 CTX=fast $D)
c1=$(cfg SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1)
check absent "$(field $KEY <<<"$c0")"                 "unset: $KEY"
check true   "$(field $KEY <<<"$c1")"                 "ADAPTIVE_VERIFY=1: $KEY"
check '"dflash"' "$(field method <<<"$c1")"           "ADAPTIVE_VERIFY=1: method"
check 7      "$(field num_speculative_tokens <<<"$c1")" "ADAPTIVE_VERIFY=1: num_speculative_tokens"
check True   "$(same "$c0" "$c1")"                    "ADAPTIVE_VERIFY=1: every other key as unset"
check 1      "$(envv VLLM_GDN_ADAPTIVE_VERIFY SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1)" "ADAPTIVE_VERIFY=1: VLLM_GDN_ADAPTIVE_VERIFY"
check 1      "$(envv VLLM_FA2_VARLEN_VERIFY SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1)"   "ADAPTIVE_VERIFY=1: VLLM_FA2_VARLEN_VERIFY"
check 0      "$(envv VLLM_FA2_VARLEN_VERIFY SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1 VLLM_FA2_VARLEN_VERIFY=0)" \
  "ADAPTIVE_VERIFY=1 VLLM_FA2_VARLEN_VERIFY=0: kept"
check unset  "$(envv VLLM_GDN_ADAPTIVE_VERIFY SPEC=dflash2 CTX=fast $D)" "unset: VLLM_GDN_ADAPTIVE_VERIFY"
check yes    "$(eq "$(rest SPEC=dflash2 CTX=fast $D)" "$(rest SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1)")" \
  "ADAPTIVE_VERIFY=1: argv and env otherwise as unset"
c2=$(cfg SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1 DYNAMIC_K="$K")
check true   "$(field $KEY <<<"$c2")"                 "with DYNAMIC_K: $KEY"
check "$K"   "$(field num_speculative_tokens_per_batch_size <<<"$c2")" "with DYNAMIC_K: num_speculative_tokens_per_batch_size"
c3=$(cfg SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1 DFLASH_TOKENS=15 LOOKUP=0)
check true   "$(field $KEY <<<"$c3")"                 "DFLASH_TOKENS=15 LOOKUP=0: $KEY"

echo "== refusals"
out=$(launch SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=1 DFLASH_TOKENS=15)
check 1   "$(cat "$T/rc")"                            "DFLASH_TOKENS=15 (lookup on): exit code"
check yes "$(grep -q 'ADAPTIVE_VERIFY refused' "$T/err" && echo yes || echo no)" "DFLASH_TOKENS=15: stderr names ADAPTIVE_VERIFY"
check no "$(started "$out")"                          "DFLASH_TOKENS=15: vllm started"
out=$(launch SPEC=dflash2 CTX=huge PREFIX_CACHE=1 $D ADAPTIVE_VERIFY=1)
check 1   "$(cat "$T/rc")"                            "CTX=huge PREFIX_CACHE=1: exit code"
check yes "$(grep -q 'ADAPTIVE_VERIFY refused' "$T/err" && echo yes || echo no)" "CTX=huge PREFIX_CACHE=1: stderr names ADAPTIVE_VERIFY"
check no "$(started "$out")"                          "CTX=huge PREFIX_CACHE=1: vllm started"
out=$(launch SPEC=dflash2 CTX=huge $D ADAPTIVE_VERIFY=1)
check 1   "$(cat "$T/rc")"                            "CTX=huge PREFIX_CACHE unset: exit code"
check no "$(started "$out")"                          "CTX=huge PREFIX_CACHE unset: vllm started"
check true "$(cfg SPEC=dflash2 CTX=huge PREFIX_CACHE=0 $D ADAPTIVE_VERIFY=1 | field $KEY)" "CTX=huge PREFIX_CACHE=0: $KEY"
out=$(launch SPEC=mtp CTX=fast ADAPTIVE_VERIFY=1)
check 1   "$(cat "$T/rc")"                            "SPEC=mtp: exit code"
check no "$(started "$out")"                          "SPEC=mtp: vllm started"
out=$(launch SPEC=dflash2 CTX=fast $D ADAPTIVE_VERIFY=yes)
check 1   "$(cat "$T/rc")"                            "ADAPTIVE_VERIFY=yes: exit code"
check no "$(started "$out")"                          "ADAPTIVE_VERIFY=yes: vllm started"

echo "== SPEC=off"
out=$(launch SPEC=off CTX=fast ADAPTIVE_VERIFY=1)
check 0  "$(cat "$T/rc")"                             "starts"
check no "$(grep -qx -- --speculative-config <<<"$out" && echo yes || echo no)" "--speculative-config in argv"
check yes "$(grep -q ADAPTIVE_VERIFY "$T/err" && echo yes || echo no)" "stderr names ADAPTIVE_VERIFY"
check unset "$(envv VLLM_GDN_ADAPTIVE_VERIFY SPEC=off CTX=fast ADAPTIVE_VERIFY=1)" "VLLM_GDN_ADAPTIVE_VERIFY"
echo; [ $FAILS = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
