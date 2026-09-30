#!/usr/bin/env bash
# The one door into patches/series: this file reads the apply order and applies it.
# Every other consumer (the Dockerfile, verify.sh, patches/check_vllm_series.sh and the
# install pages in docs/) calls this file instead of parsing patches/series itself.
#
#   bash patches/apply.sh --list   print the series in apply order, one basename per line
#   bash patches/apply.sh DIR      apply the series to DIR, the vllm PACKAGE directory
#                                  (site-packages/vllm, not site-packages)
#
# Exit codes: 0 success; 1 a patch did not apply (the message names it); 2 a usage error,
# or patches/series and the patches/ directory disagree.
#
# --list always prints the series. When the series and the directory disagree, it also
# names each offender on stderr and exits 2: a patch that is not in the series is never
# applied, and a name in the series with no file is a typo. The apply mode refuses to start
# on that disagreement.
#
# Apply policy: GNU `patch -p1 --forward --fuzz 0 --no-backup-if-mismatch`, in series order,
# stop at the first patch that fails. An offset means the context matched exactly and the
# file only grew around it: that is benign, and the output reports it. Fuzz means the
# context did NOT match and GNU patch accepted an approximate anchor: that is a patch cut
# against a tree that no longer exists, so it fails here by name instead of landing by guess.
# --forward makes a patch that is already applied fail by name, instead of a question on
# the terminal.
#
# Output contract (patches/check_vllm_series.sh counts these lines), one line per patch:
#   == <name>
#   == <name> (<N> hunk(s) at an offset)
# On a failure, the patch output goes to stderr, indented, after a FAILED: line.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  echo "usage: bash patches/apply.sh --list | DIR  (DIR = the installed vllm package directory)" >&2
  exit 2
}

series() {
  sed -e 's/#.*//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//' -e '/^$/d' "$HERE/series"
}

list() {
  local names on_disk in_series extra missing
  names=$(series)
  printf '%s\n' "$names"
  on_disk=$(for f in "$HERE"/*.patch; do [ -e "$f" ] && basename "$f"; done | sort)
  in_series=$(printf '%s\n' "$names" | sort)
  [ "$on_disk" = "$in_series" ] && return 0
  echo "ERROR: patches/series and the patches/ directory disagree:" >&2
  extra=$(comm -23 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$in_series"))
  missing=$(comm -13 <(printf '%s\n' "$on_disk") <(printf '%s\n' "$in_series"))
  [ -z "$extra" ] || printf '    not in patches/series (never applied): %s\n' $extra >&2
  [ -z "$missing" ] || printf '    in patches/series, no such file (or listed twice): %s\n' $missing >&2
  return 2
}

apply() {
  local dir=$1 names name out n
  [ -d "$dir" ] || { echo "ERROR: $dir is not a directory" >&2; usage; }
  names=$(list) || exit 2
  while IFS= read -r name; do
    if ! out=$(patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -d "$dir" -i "$HERE/$name" 2>&1); then
      {
        echo "FAILED: $name does not apply to $dir with exact context. Possible causes:"
        echo "    the patch is already applied (bash verify.sh --no-server says which patches are),"
        echo "    DIR is not the vllm package directory, or the vLLM version is not the pin in"
        echo "    docker/requirements.txt (then regenerate the patch against the pin)."
        printf '%s\n' "$out" | sed 's/^/    /'
      } >&2
      exit 1
    fi
    n=$(printf '%s\n' "$out" | grep -c 'offset' || true)
    if [ "$n" -gt 0 ]; then echo "== $name ($n hunk(s) at an offset)"; else echo "== $name"; fi
  done <<<"$names"
}

case "${1:-}" in
  --list) [ $# -eq 1 ] || usage; list ;;
  ""|-*) usage ;;
  *) [ $# -eq 1 ] || usage; apply "$1" ;;
esac
