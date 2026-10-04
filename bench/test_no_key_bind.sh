#!/bin/bash
# CPU test for #204: which --host each launcher passes to `vllm serve`, with and without a key.
# Copies the checkout to a temp dir, so the api_key.txt rows never touch the real one, and runs the three
# launchers with PRINT_ARGV=1 (launcher_common.sh), which prints the argv instead of starting vLLM. No GPU,
# no model, no network. In a container (/.dockerenv) the default differs, so this is skipped there.
#
#   bash bench/test_no_key_bind.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
[ -f /.dockerenv ] && { echo "skip: /.dockerenv exists, the container default is 0.0.0.0 by design"; exit 0; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
# A launcher that ignores PRINT_ARGV=1 reaches exec. This stub then fails the row, and no real vllm starts.
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\necho "exec reached: PRINT_ARGV=1 did not stop the launcher" >&2; exit 99\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
FAILS=0
# host_of <launcher> [@keyfile] <env assignments...>: the argument after --host in the PRINT_ARGV=1 argv.
# @keyfile puts the key only in api_key.txt.
host_of() { local sc=$1; shift
  rm -f "$T/api_key.txt"; [ "${1:-}" = @keyfile ] && { echo filekey > "$T/api_key.txt"; shift; }
  (cd "$T" && env -u VLLM_API_KEY -u HOST PRINT_ARGV=1 "$@" bash "$sc" 2>/dev/null </dev/null) | sed -n '/^--host$/{n;p;}'; }
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s -> %s\n' "$what" "$got"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "${got:-<none>}" "$want"; FAILS=$((FAILS+1)); fi; }
for sc in single-user/start_qwen.sh batch/start_qwen.sh single-user/alternative.sh; do
  echo "== $sc"
  check 127.0.0.1 "$(host_of $sc)"                              "no key, no HOST"
  check 0.0.0.0   "$(host_of $sc VLLM_API_KEY=k)"               "key in the environment"
  check 0.0.0.0   "$(host_of $sc @keyfile)"                     "key only in api_key.txt"
  check 0.0.0.0   "$(host_of $sc HOST=0.0.0.0)"                 "no key, HOST=0.0.0.0 (explicit)"
  check 127.0.0.1 "$(host_of $sc HOST=127.0.0.1 VLLM_API_KEY=k)" "key, HOST=127.0.0.1"
  check 10.1.2.3  "$(host_of $sc HOST=10.1.2.3 VLLM_API_KEY=k)"  "key, HOST=10.1.2.3"
done
echo "== verify.sh key check (the section only, as the launcher env would set it)"
vk() { (cd "$T" && rm -f api_key.txt; env -u VLLM_API_KEY -u HOST "$@" bash -c '
  ok(){ echo "PASS $1"; }; warn(){ echo "WARN $1"; }; fail(){ echo "FAIL $1"; }
  eval "$(sed -n "/^if \[ -s api_key.txt \] || \[ -n/,/^fi\$/p" verify.sh)"' 2>&1 | cut -c1-4); }
[ "$(vk)" = WARN ] && echo "  PASS  no key, no HOST -> WARN" || { echo "  FAIL  no key, no HOST"; FAILS=$((FAILS+1)); }
[ "$(vk HOST=127.0.0.1)" = WARN ] && echo "  PASS  no key, HOST=127.0.0.1 -> WARN" || { echo "  FAIL  no key, HOST=127.0.0.1"; FAILS=$((FAILS+1)); }
[ "$(vk HOST=0.0.0.0)" = FAIL ] && echo "  PASS  no key, HOST=0.0.0.0 -> FAIL" || { echo "  FAIL  no key, HOST=0.0.0.0"; FAILS=$((FAILS+1)); }
[ "$(vk VLLM_API_KEY=k HOST=0.0.0.0)" = PASS ] && echo "  PASS  key, HOST=0.0.0.0 -> PASS" || { echo "  FAIL  key, HOST=0.0.0.0"; FAILS=$((FAILS+1)); }
echo; [ $FAILS = 0 ] && echo "all bind checks passed" || { echo "$FAILS bind checks FAILED"; exit 1; }
