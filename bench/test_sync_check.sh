#!/bin/bash
# CPU test for bench/sync_check.sh's log side (--parse-only, --record): a synthetic server log in the formats
# vLLM 0.31's VLLM_GPU_SYNC_CHECK=warn produces (Python warnings from torch's sync-debug mode and from
# vllm/utils/gpu_sync_debug.py's non_blocking copy checks), with docker timestamps, "(EngineCore pid=N)"
# prefixes, a repeated site, a repo-path site, the source-line continuations, and lines that must not count
# (another warning category, another UserWarning, a traceback frame). Checks the unique sites and their kinds,
# the allowlist match (with and without a kind), --record, the error-mode raise, a clean log and a missing log.
# No GPU, no server.
#
#   bash bench/test_sync_check.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
SP=/app/venv/lib/python3.12/site-packages
cat > "$T/server.log" <<EOF
(APIServer pid=1) INFO 10-20 19:00:00 [api_server.py:100] Starting vLLM API server
2026-10-20T19:00:01.000000000Z (EngineCore pid=88) $SP/vllm/v1/worker/gpu/model_runner.py:1234: UserWarning: called a synchronizing CUDA operation (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:154.)
  n = int(x.item())
(EngineCore pid=88) $SP/vllm/v1/worker/gpu/model_runner.py:1234: UserWarning: called a synchronizing CUDA operation (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:154.)
  n = int(x.item())
(EngineCore pid=88) $SP/vllm/v1/attention/backends/kvarn_attn.py:77: UserWarning: Tensor.copy_: a non_blocking H2D copy that may be silently synchronous (the CPU tensor is not pinned). Implicit GPU<->CPU sync detected - use a pinned, densely laid out CPU tensor, or wrap with gpu_sync_allowed()
  buf.copy_(cpu, non_blocking=True)
/app/pthfa/pth_flash_attn/launch.py:42: UserWarning: called a synchronizing CUDA operation (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:154.)
(EngineCore pid=88) $SP/vllm/v1/worker/gpu/model_runner.py:999: DeprecationWarning: called a synchronizing CUDA operation is not this
(EngineCore pid=88) $SP/vllm/model_executor/layers/foo.py:12: UserWarning: some other warning
  File "$SP/vllm/v1/worker/gpu/model_runner.py", line 1234, in execute_model
(APIServer pid=1) INFO 10-20 19:00:05 [loggers.py:200] Engine 000: Avg prompt throughput: 1.0 tokens/s
EOF
printf 'INFO only\nno warnings here\n' > "$T/clean.log"
printf '%s\n' "(EngineCore pid=88) RuntimeError: GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()" > "$T/raise.log"
FAILS=0
check() { if [ "$2" = "$1" ]; then printf '  PASS  %s\n' "$3"; else printf '  FAIL  %s -> %s (want %s)\n' "$3" "$2" "$1"; FAILS=$((FAILS+1)); fi; }
sc() { bash "$HERE/sync_check.sh" "$@" > "$T/out" 2>&1; echo $?; }

: > "$T/allow"
check 1 "$(sc "$T/server.log" --parse-only --allowlist "$T/allow")" "new sites against an empty allowlist exit 1"
check 3 "$(grep -c '^  NEW ' "$T/out")" "three unique sites (repeats, other categories and other warnings ignored)"
grep -q 'NEW   vllm/v1/worker/gpu/model_runner.py:1234 sync  x2' "$T/out"; check 0 $? "site-packages path made relative, repeat counted x2"
grep -q 'NEW   vllm/v1/attention/backends/kvarn_attn.py:77 copy_-H2D  x1' "$T/out"; check 0 $? "implicit copy site has kind copy_-H2D"
grep -q 'NEW   pthfa/pth_flash_attn/launch.py:42 sync' "$T/out"; check 0 $? "repo path made repo-relative"
grep -q '^SYNC_CHECK FAIL sites=3 new=3 allowlisted=0 warnings=4 error_raises=0' "$T/out"; check 0 $? "summary line"

check 0 "$(sc "$T/server.log" --record --mode dflash2-long --allowlist "$T/allow")" "--record exits 0"
check 3 "$(grep -vc '^#' "$T/allow")" "--record appended the three sites"
check 0 "$(sc "$T/server.log" --parse-only --allowlist "$T/allow")" "recorded allowlist passes"
grep -q '^SYNC_CHECK PASS sites=3 new=0 allowlisted=3' "$T/out"; check 0 $? "PASS summary"

printf 'vllm/v1/worker/gpu/model_runner.py:1234  # any kind\nvllm/v1/attention/backends/kvarn_attn.py:77 sync\npthfa/pth_flash_attn/launch.py:42 sync\n' > "$T/allow2"
check 1 "$(sc "$T/server.log" --parse-only --allowlist "$T/allow2")" "a kind mismatch is a new site"
check 1 "$(grep -c '^  NEW ' "$T/out")" "only the kind-mismatched site is new; a kindless entry matches"

check 0 "$(sc "$T/clean.log" --parse-only --allowlist "$T/allow2")" "clean log passes"
check 1 "$(sc "$T/raise.log" --parse-only --allowlist "$T/allow2")" "error-mode raise fails"
check 2 "$(sc "$T/nope.log" --parse-only)" "missing log is an invalid run"

if [ "$FAILS" = 0 ]; then echo "test_sync_check: all passed"; exit 0; fi
echo "test_sync_check: $FAILS failed"; exit 1
