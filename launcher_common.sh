#!/bin/bash
# launcher_common.sh - code shared by the three server launchers
# (single-user/start_qwen.sh, batch/start_qwen.sh, single-user/alternative.sh).
#
# Sourced after REPO is set. It defines functions and sets nothing by itself.
#
#   resolve_bind_host   after resolve_vllm_key (resolve_api_key.sh). Sets BIND_HOST, the
#                       --host the launchers pass. An explicit HOST wins. Else a
#                       server with a key listens on every interface (0.0.0.0,
#                       as before), and one without a key listens on 127.0.0.1
#                       only, because with no key nothing on the network is
#                       protected (#204). Inside a container the default stays
#                       0.0.0.0, since a published port cannot reach a loopback
#                       bind; there the port mapping is what limits exposure.
#
#   qwen_exec           the launcher's last call, in place of exec. With PRINT_ARGV=1 it
#                       prints the argv, one argument per line, and exits 0 instead
#                       of starting vLLM. Every check, default and warning before it
#                       has already run, so this is a dry run of the launcher that
#                       needs no GPU. The environment is not printed: it holds
#                       VLLM_API_KEY.

resolve_bind_host() {
  if [ -n "${HOST:-}" ]; then
    BIND_HOST=$HOST
    if [ -z "${VLLM_API_KEY:-}" ]; then
      case "$HOST" in 127.*|localhost|::1) ;; *)
        echo "WARNING: no API key and HOST=$HOST: anything that can reach this port can use the server. Set VLLM_API_KEY or api_key.txt (openssl rand -hex 24)." >&2 ;;
      esac
    fi
  elif [ -n "${VLLM_API_KEY:-}" ]; then
    BIND_HOST=0.0.0.0
  elif [ -f /.dockerenv ]; then
    BIND_HOST=0.0.0.0
    echo "WARNING: no API key: this container listens on 0.0.0.0, so whatever the published port reaches is open. Set VLLM_API_KEY (make keygen) or publish the port on 127.0.0.1 only." >&2
  else
    BIND_HOST=127.0.0.1
    echo "no API key: binding 127.0.0.1 only. To serve other machines, set VLLM_API_KEY (openssl rand -hex 24 > api_key.txt) or HOST=0.0.0.0." >&2
  fi
}

qwen_exec() {
  if [ "${PRINT_ARGV:-0}" = 1 ]; then
    printf '%s\n' "$@"
    exit 0
  fi
  exec "$@"
}
