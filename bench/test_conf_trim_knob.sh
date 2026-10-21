#!/bin/bash
# CPU test for CONF_TRIM (single-user/start_qwen.sh; research experiment drafter-1a, docs/research/deepinfer/drafter.md):
# unset or empty, every single-user arm gives the argv, VLLM_* environment and stderr of the same launcher with the
# CONF_TRIM block cut out, byte for byte; CONF_TRIM=1|2 on SPEC=dflash2 exports VLLM_DFLASH2_CONF_TRIM and the threshold,
# switches to --no-async-scheduling and changes nothing else in the argv; a bad value, another SPEC, CTX=huge with
# prefix caching on, DYNAMIC_K alongside, or no threshold are refused. Copies the checkout to a temp dir and runs the
# launcher against a stub vllm (prints its argv, then "@@ENV" and its VLLM_* environment). No GPU, no model, no network.
#
#   bash bench/test_conf_trim_knob.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
stage() {
  mkdir -p "$2"
  (cd "$1" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$2"
  mkdir -p "$2/venv/bin" "$2/models/Qwen3.8-27B-DFlash2-W4A16"
  : > "$2/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"
  printf '#!/bin/sh\nprintf "%%s\\n" "$@"\necho @@ENV\nenv | grep "^VLLM_" | sort\n' > "$2/venv/bin/vllm"
  chmod +x "$2/venv/bin/vllm"
}
stage "$REPO" "$T/new"
# The base launcher: the same file without the CONF_TRIM block (its first comment line through "fi # CONF_TRIM"). Both
# run from the same tree, swapped into place, so paths in the argv and stderr match.
L="$T/new/single-user/start_qwen.sh"
cp "$L" "$T/launcher.new"
sed '/^# CONF_TRIM (drafter-1a/,/^fi # CONF_TRIM$/d' "$T/launcher.new" > "$T/launcher.base"
check 1 "$(cmp -s "$T/launcher.new" "$T/launcher.base" && echo 0 || echo 1)" "the base launcher really lacks the block"
check 0 "$(grep -c CONF_TRIM "$T/launcher.base")" "no CONF_TRIM left in the base launcher"
mkdir -p "$T/smi"; printf '#!/bin/sh\nexit 9\n' > "$T/smi/nvidia-smi"; chmod +x "$T/smi/nvidia-smi"
# run <new|base> <env...>: argv and VLLM_* environment on stdout, stderr in $T/err, exit code in $T/rc.
run() { cp "$T/launcher.$1" "$L"; shift
  (cd "$T/new" && env -u PRINT_ARGV -u CONF_TRIM -u CONF_TRIM_THR -u VERIFY_MS_PER_TOKEN -u ASYNC_SCHED -u KV_OFFLOAD_GB \
     -u EXTRA_ARGS -u SPEC -u CTX -u PYTORCH_CUDA_ALLOC_CONF -u WSL_DISTRO_NAME -u CUDA_HOME -u FA2_FP8KV -u PTH_FA \
     -u DRAFT_METHOD -u DFLASH_TOKENS -u LOOKUP -u DYNAMIC_K -u ADAPTIVE_VERIFY -u DRAFT -u PREFIX_CACHE -u CUDA_VISIBLE_DEVICES \
     PATH="$T/smi:$PATH" VLLM_API_KEY=k FA2_FP8KV=0 "$@" bash single-user/start_qwen.sh 2>"$T/err" </dev/null)
  echo $? > "$T/rc"; }
rc() { cat "$T/rc"; }
ARMS=("SPEC=mtp CTX=fast" "SPEC=mtp CTX=long" "SPEC=mtp CTX=huge" "SPEC=dflash2 CTX=fast" "SPEC=dflash2 CTX=long"
      "SPEC=dflash2 CTX=huge" "SPEC=dflash2 CTX=huge PREFIX_CACHE=0" "SPEC=off CTX=long")

echo "== unset and empty = the launcher without the block (argv, VLLM_* environment, stderr), every arm"
for arm in "${ARMS[@]}"; do
  # shellcheck disable=SC2086
  run base $arm > "$T/a"; cp "$T/err" "$T/a.err"; ra=$(rc)
  # shellcheck disable=SC2086
  run new $arm > "$T/b"; cp "$T/err" "$T/b.err"
  # shellcheck disable=SC2086
  run new $arm CONF_TRIM= > "$T/c"; cp "$T/err" "$T/c.err"
  same=1; cmp -s "$T/a" "$T/b" && cmp -s "$T/a" "$T/c" && cmp -s "$T/a.err" "$T/b.err" && cmp -s "$T/a.err" "$T/c.err" || same=0
  check "0 1" "$ra $same" "$arm: boots, unset = empty = base"
done

echo "== CONF_TRIM=1 on SPEC=dflash2 CTX=long"
run new SPEC=dflash2 CTX=long > "$T/off"
run new SPEC=dflash2 CTX=long CONF_TRIM=1 VERIFY_MS_PER_TOKEN=2.0 > "$T/on"
check 0 "$(rc)" "boots"
check 1 "$(grep -cx 'VLLM_DFLASH2_CONF_TRIM=1' "$T/on")" "exports VLLM_DFLASH2_CONF_TRIM=1"
check 1 "$(grep -cx 'VLLM_DFLASH2_VERIFY_MS_PER_TOKEN=2.0' "$T/on")" "exports VLLM_DFLASH2_VERIFY_MS_PER_TOKEN"
check 1 "$(grep -cx -- '--no-async-scheduling' "$T/on")" "--no-async-scheduling"
norm() { sed -e '/^VLLM_DFLASH2_CONF_TRIM=/d' -e '/^VLLM_DFLASH2_VERIFY_MS_PER_TOKEN=/d' -e '/^VLLM_DFLASH2_CONF_TRIM_THR=/d' \
  -e 's/^--no-async-scheduling$/--async-scheduling/' "$1"; }
check yes "$([ "$(norm "$T/off")" = "$(norm "$T/on")" ] && echo yes || echo no)" "nothing else changes"
run new SPEC=dflash2 CTX=long CONF_TRIM=2 CONF_TRIM_THR=0.3 > "$T/on2"
check "0 1 1" "$(rc) $(grep -cx 'VLLM_DFLASH2_CONF_TRIM=2' "$T/on2") $(grep -cx 'VLLM_DFLASH2_CONF_TRIM_THR=0.3' "$T/on2")" \
  "CONF_TRIM=2 CONF_TRIM_THR=0.3: boots, exports both"
run new SPEC=dflash2 CTX=long CONF_TRIM=1 VERIFY_MS_PER_TOKEN=2.0 ASYNC_SCHED=1 > "$T/on3"
check "0 1 1" "$(rc) $(grep -cx -- '--async-scheduling' "$T/on3") $(grep -c 'async scheduling ignores' "$T/err")" \
  "explicit ASYNC_SCHED=1 is honoured, with a warning"
run new SPEC=dflash2 CTX=huge PREFIX_CACHE=0 CONF_TRIM=1 VERIFY_MS_PER_TOKEN=3.6 > /dev/null
check 0 "$(rc)" "CTX=huge PREFIX_CACHE=0: accepted"

echo "== refusals"
refused() { local what=$1 pat=$2; shift 2; local out; out=$(run new "$@")
  check "1 yes" "$(rc) $([ -z "$out" ] && grep -q "$pat" "$T/err" && echo yes || echo no)" "$what"; }
refused "CONF_TRIM=3"                     "CONF_TRIM=3"       SPEC=dflash2 CTX=long CONF_TRIM=3 VERIFY_MS_PER_TOKEN=2
refused "SPEC=mtp"                        "SPEC=dflash2 set"  SPEC=mtp CTX=long CONF_TRIM=1 VERIFY_MS_PER_TOKEN=2
refused "CTX=huge, prefix caching on"     "CONF_TRIM refused" SPEC=dflash2 CTX=huge CONF_TRIM=1 VERIFY_MS_PER_TOKEN=2
refused "with DYNAMIC_K"                  "DYNAMIC_K"         SPEC=dflash2 CTX=long CONF_TRIM=1 VERIFY_MS_PER_TOKEN=2 DYNAMIC_K='[[1,4,7]]'
refused "no threshold"                    "needs VERIFY_MS"   SPEC=dflash2 CTX=long CONF_TRIM=1
refused "with ADAPTIVE_VERIFY=1"          "ADAPTIVE_VERIFY"   SPEC=dflash2 CTX=long CONF_TRIM=1 VERIFY_MS_PER_TOKEN=2 ADAPTIVE_VERIFY=1
echo; [ "$FAILS" = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }
