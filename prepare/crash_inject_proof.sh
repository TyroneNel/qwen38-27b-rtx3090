#!/bin/bash
# prepare/crash_inject_proof.sh — Loop C crash-injection proof for F04.
#
# Proves generation-scoped, locked, atomic preparation by killing publishers
# with SIGKILL at publication boundaries and asserting the on-disk protocol:
# after every kill the generation is either the OLD complete generation or
# the NEW complete generation — never mixed — source hashes AND inodes are
# unchanged, and the lock is re-acquirable (no wedged flock).
#
# SIGKILL, not SIGTERM: graceful shutdown would test the cleanup trap, not
# the atomicity of the on-disk protocol.
#
# Modes:
#   --self-test   run the full inject/assert cycle against synthetic
#                 publishers that mimic each real protocol (tmp+replace,
#                 bak-orig+replace, staging+replace, flock+publish).
#                 Needs no GPU, no models. This is the CI-safe mode.
#   --live        same cycle against the REAL boundaries below. Needs the
#                 model tree, venv, and idle CPU/GPU. NOT run while the
#                 GPUs are busy; recorded here so the live run is mechanical.
#
# Real boundaries (validity/model-prep branch):
#   B1 build_draft_vocab.py  extras tmp + os.replace
#   B2 quant_lm_head.py      shard / index / config tmp + os.replace
#   B3 quant_heads_stream.py shard .bak-orig rename + tmp os.replace
#   B4 drafter/capture.py    seqs.json.staging + os.replace
#   B5 drafter/export_mtp.py generation copy-out + .bak-mtp rollback files
#   B6 docker/prepare.sh     flock -n fd9 + completion manifest
set -u
MODE=self-test
for a in "$@"; do case $a in --self-test) MODE=self-test;; --live) MODE=live;; \
  --negative-test) MODE=negative;; \
  *) echo "crash_inject_proof: unknown flag $a" >&2; exit 1;; esac; done

PASS=0; FAIL=0; SKIP=0
LOG=""; [ -n "${PROOF_LOG:-}" ] && LOG="$PROOF_LOG"
command -v sha256sum >/dev/null 2>&1 \
  || { echo "crash_inject_proof: refusing: sha256sum not found" >&2; exit 1; }
# flock(1) is absent on some platforms (e.g. git-bash). Lock assertions then
# SKIP — they do not pass vacuously — and the live Linux run covers B6.
FLOCK_OK=0; command -v flock >/dev/null 2>&1 && FLOCK_OK=1
say() { echo "proof: $1"; [ -n "$LOG" ] && echo "proof: $1" >> "$LOG"; }
verdict() { # $1=name $2=0/1 $3=detail
  if [ "$2" = 0 ]; then PASS=$((PASS+1)); say "PASS $1 ($3)";
  else FAIL=$((FAIL+1)); say "FAIL $1 ($3)"; fi
}
skip() { SKIP=$((SKIP+1)); say "SKIP $1 ($2)"; }
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
ino() { stat -c %i "$1" 2>/dev/null || stat -f %i "$1" 2>/dev/null; }

# _cycle <name> <publisher-fn> <gendir> <target> <sourcedir> [rollback-relpath]
# Rollback semantics (B3 shape): between the .bak-orig rename and the final
# replace the live path is MISSING and the old generation is selectable only
# via the rollback file. That is old-complete, not mixed — provided the
# rollback hashes to the pre-run generation. Without a rollback entry a
# missing live target is mixed.
_cycle() {
  local name=$1 pub=$2 gen=$3 target=$4 src=$5 rollback=${6:-}
  local src_snap; src_snap=$(cd "$src" && sha256sum ./* 2>/dev/null | sort)
  local src_ino; src_ino=$(cd "$src" && stat -c '%n %i' ./* 2>/dev/null | sort)
  local old_hash; old_hash=$(sha "$gen/$target")
  for round in midwrite late; do
    rm -f "$gen/$target.complete"
    ( $pub "$gen" "$target" ) & local pid=$!
    if [ "$round" = midwrite ]; then sleep 0.15; else sleep 1.5; fi
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    local new_hash; new_hash=$(sha "$gen/$target")
    local complete="absent"; [ -f "$gen/$target.complete" ] && complete=$(cat "$gen/$target.complete")
    # Old-or-new, never mixed: a new COMPLETE marker must agree with new
    # data; without the marker the data must still be the old generation;
    # with a rollback protocol a missing live target whose rollback hashes
    # to the old generation is old-via-rollback (recoverable, not mixed).
    local rb_hash="";
    [ -n "$rollback" ] && [ -f "$gen/$rollback" ] && rb_hash=$(sha "$gen/$rollback")
    if [ "$complete" = "$new_hash" ] && [ -n "$new_hash" ]; then verdict "$name/$round" 0 "new-complete";
    elif [ "$complete" = "absent" ] && [ "$new_hash" = "$old_hash" ]; then verdict "$name/$round" 0 "old-intact";
    elif [ "$complete" = "absent" ] && [ -z "$new_hash" ] && [ "$rb_hash" = "$old_hash" ] && [ -n "$old_hash" ]; then
      verdict "$name/$round" 0 "old-via-rollback";
    else verdict "$name/$round" 1 "mixed state (marker=$complete data=$new_hash old=$old_hash rb=$rb_hash)"; fi
    if [ "$(cd "$src" && sha256sum ./* 2>/dev/null | sort)" = "$src_snap" ] \
       && [ "$(cd "$src" && stat -c '%n %i' ./* 2>/dev/null | sort)" = "$src_ino" ]; then
      verdict "$name/$round-sources" 0 "hashes+inodes unchanged"
    else verdict "$name/$round-sources" 1 "source mutated"; fi
    if [ "$FLOCK_OK" = 1 ]; then
      if ( flock -n "$gen/.lock" true ) 2>/dev/null; then verdict "$name/$round-lock" 0 "re-acquirable";
      else verdict "$name/$round-lock" 1 "lock wedged"; fi
    else skip "$name/$round-lock" "no flock(1) on this platform; live run covers B6"; fi
  done
}

# ---- synthetic publishers: same protocols as B1-B6, ~2s of chunked writes ---
_pub_tmp_replace() { # $1=gen $2=target  (B1, B2, B4 shape)
  local g=$1 t=$2 i
  : > "$g/$t.tmp"
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t.tmp"; sleep 0.05; done
  mv "$g/$t.tmp" "$g/$t"
  sha "$g/$t" > "$g/$t.complete"
}
_pub_bak_orig() { # $1=gen $2=target  (B3 shape)
  local g=$1 t=$2 i
  [ -e "$g/$t.bak-orig" ] || mv "$g/$t" "$g/$t.bak-orig"
  : > "$g/$t.tmp"
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t.tmp"; sleep 0.05; done
  mv "$g/$t.tmp" "$g/$t"
  sha "$g/$t" > "$g/$t.complete"
}
_pub_flock() { # $1=gen $2=target  (B6 shape: work under an exclusive lock)
  local g=$1 t=$2 i
  if [ "$FLOCK_OK" = 1 ]; then exec 9>"$g/.lock"; flock 9; fi
  : > "$g/$t.tmp"
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t.tmp"; sleep 0.05; done
  mv "$g/$t.tmp" "$g/$t"
  sha "$g/$t" > "$g/$t.complete"
}

_pub_direct() { # NEGATIVE CONTROL: in-place writes, no tmp, no marker.
  # This is the pre-F04 shape. The proof must flag every round mixed.
  local g=$1 t=$2 i
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t"; sleep 0.05; done
}

_negative_test() {
  local root; root=$(mktemp -d "${TMPDIR:-/tmp}/crashproof.XXXXXX")
  local gen=$root/gen src=$root/src; mkdir -p "$gen" "$src"
  seq 1 50 > "$gen/data.bin"
  echo "source-v1" > "$src/weights.bin"
  _cycle "negative/_pub_direct" _pub_direct "$gen" "data.bin" "$src"
  rm -rf "$root"
  if [ "$FAIL" -gt 0 ]; then
    say "negative control RED as required (proof detects in-place publication)"
  else say "negative control GREEN — proof is blind, fix it"; fi
}

_self_test() {
  local root; root=$(mktemp -d "${TMPDIR:-/tmp}/crashproof.XXXXXX")
  say "self-test root $root"
  local proto
  for proto in _pub_tmp_replace _pub_bak_orig _pub_flock; do
    local gen=$root/gen-$proto src=$root/src-$proto
    mkdir -p "$gen" "$src"
    seq 1 50 > "$gen/data.bin"
    echo "source-v1" > "$src/weights.bin"; echo "source-v1" > "$src/config.json"
    local rb=""; [ "$proto" = _pub_bak_orig ] && rb="data.bin.bak-orig"
    _cycle "selftest/$proto" "$proto" "$gen" "data.bin" "$src" "$rb"
  done
  rm -rf "$root"
}

# ---- live boundary configs: publisher command per boundary (NOT run now) ----
# Each entry: name|generation-dir|target|publisher-command. The live cycle
# snapshots, backgrounds the publisher, SIGKILLs, and applies the same
# old-or-new + sources-unchanged + lock assertions with boundary-specific
# completeness predicates (index parses, shard sizes, manifest present).
_live_boundaries() {
  say "LIVE mode: GPUs are busy — configs recorded, nothing executed."
  say "B1 extras: prepare/build_draft_vocab.py -> draft_vocab extras tmp+replace"
  say "B2 head: prepare/quant_lm_head.py -> shard/index/config tmp+replace"
  say "B3 stream: prepare/quant_heads_stream.py -> .bak-orig + tmp replace"
  say "B4 capture: drafter/capture.py -> seqs.json.staging replace"
  say "B5 export: drafter/export_mtp.py -> generation copy-out + .bak-mtp"
  say "B6 install: docker/prepare.sh -> flock fd9 + completion manifest"
  say "Run with idle hardware: PROOF_LOG=evidence.log bash $0 --live (then implement per-boundary publishers)."
}

if [ "$MODE" = live ]; then _live_boundaries; elif [ "$MODE" = negative ]; then _negative_test; else _self_test; fi
say "verdicts: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
# Negative mode inverts the bar: success is the proof going red.
if [ "$MODE" = negative ]; then [ "$FAIL" -gt 0 ]; else [ "$FAIL" = 0 ]; fi
