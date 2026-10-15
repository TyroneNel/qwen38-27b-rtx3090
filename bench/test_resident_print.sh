#!/bin/bash
# CPU test for the resident-request note in single-user/start_qwen.sh. The note used to be a
# constant (KV_MEM / ((k+2) * 104988089)) with no context term; it is now the function measured in
# bench/results/2026-10-15-Q1-seat-gauge/report.md:
#
#   one resident request = 6*(k+2) + 2*ceil(ctx/864) pool pages, pool = KV_MEM / 14,601,709
#
# Copies the checkout to a temp dir, stubs vllm, and dry-runs the launcher with PRINT_ARGV=1 --
# no GPU, no model, no network. Checks the printed numbers, that the note stays silent when
# MAX_SEQS fits, and that CTX=huge is skipped (its block and page differ and are unmeasured).
#
#   bash bench/test_resident_print.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
FAILS=0
check() { local want=$1 got=$2 what=$3
  if [ "$got" = "$want" ]; then printf '  PASS  %s\n' "$what"; else printf '  FAIL  %s -> %s (want %s)\n' "$what" "$got" "$want"; FAILS=$((FAILS+1)); fi; }

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$REPO" && tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=models -cf - .) | tar -xf - -C "$T"
mkdir -p "$T/venv/bin"
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > "$T/venv/bin/vllm"; chmod +x "$T/venv/bin/vllm"
# The dflash2 branch refuses to start without a drafter directory; a marker file is enough for a dry run.
mkdir -p "$T/models/Qwen3.8-27B-DFlash2-W4A16" && : > "$T/models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"

# run <env assignments...>: the launcher's dry-run output on stdout, note lines only.
run() {
  (cd "$T" && env -u PRINT_ARGV -u DFLASH_TOKENS -u KV_MEM -u MAX_SEQS -u MAX_LEN \
     SPEC=dflash2 PREFIX_CACHE=0 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" "$@" \
     bash single-user/start_qwen.sh 2>/dev/null </dev/null) | grep -o 'the pinned pool holds about .*resident requests on short' || true
}
# seats <env...>: the printed range as "lo:hi" (a single number prints as "n:n").
seats() { run "$@" | sed -E 's/.*about ([0-9]+) to ([0-9]+) resident.*/\1:\2/; s/.*about ([0-9]+) resident.*/\1:\1/'; }
# note <env...>: the whole note, so the parts beyond the seat counts can be checked.
note() {
  (cd "$T" && env -u PRINT_ARGV -u DFLASH_TOKENS -u KV_MEM -u MAX_SEQS -u MAX_LEN \
     SPEC=dflash2 PREFIX_CACHE=0 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" "$@" \
     bash single-user/start_qwen.sh 2>/dev/null </dev/null) | grep -o '\[start_qwen\] note: the pinned pool.*' || true
}

echo "== the printed range brackets every measured arm (batch 5 Q2, 2026-10-15)"
# The pool is ENGINE-CHOSEN (382 pages at k=7 on int8 KV, 404 at k=3 -- the block becomes 816), so
# each KV_MEM below is the one whose pool matched that boot's num_gpu_blocks. The claim under test
# is the pass rule: the observed seat count lies inside the printed bracket.
#   a    pool 382, k=7, observed 7   -> bracket 6:7
#   b    pool 404, k=3, observed 13  -> bracket 11:13
#   c    pool 308, k=7, observed 5   -> bracket 5:5
#   d    pool 364, k=7, observed 6   -> bracket 6:6   (batch 4's arm)
check "6:7"   "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" "k=7, pool 382: bracket 6:7 contains the observed 7 (batch 5 a)"
check "11:13" "$(seats KV_MEM=5899090436 DFLASH_TOKENS=3 MAX_SEQS=16 MAX_LEN=65536)" "k=3, pool 404: bracket 11:13 contains the observed 13 (batch 5 b)"
check "5:5"   "$(seats KV_MEM=4509715660 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" "k=7, pool 308: bracket 5:5 contains the observed 5 (batch 5 c)"
check "6:6"   "$(seats KV_MEM=5315022028 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" "k=7, pool 364: bracket 6:6 contains the observed 6 (batch 4)"

echo "== the state term is 6*(k+2) pages, and MAX_LEN no longer enters the print"
check "6:7"   "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=1024)"  "the printed range does not depend on MAX_LEN"
check "6:7"   "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" "  ... nor at 65536"
check "11:13" "$(seats KV_MEM=5899090436 DFLASH_TOKENS=3 MAX_SEQS=16 MAX_LEN=65536)" "k=3: state 30 -> 13 upper, 11 lower"
check "15:21" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=1 MAX_SEQS=24 MAX_LEN=65536)" "k=1: state 18 -> 21 upper, 15 lower; the observed 21 is inside (batch 4 Q4)"

echo "== silent when MAX_SEQS fits the pool"
check "" "$(run KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=7 MAX_LEN=65536)"  "MAX_SEQS=7 == the upper bound: no note"
check "" "$(run KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=4 MAX_LEN=65536)"  "MAX_SEQS=4: no note"
check "" "$(run KV_MEM=4900000000 DFLASH_TOKENS=3 MAX_SEQS=16 CTX=huge)"      "CTX=huge: skipped (KVarN block and page are unmeasured)"
[ -n "$(run KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 CTX=long)" ] \
  && check 1 1 "CTX=long still prints" || check 1 0 "CTX=long still prints"

echo "== the note stays auditable"
n="$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=8 MAX_LEN=65536)"
case "$n" in *"reserves 54 of the"*) check 1 1 "the note names the state term (54 at k=7)";; *) check 1 0 "the note names the state term";; esac
case "$n" in *"382 pages for recurrent state"*) check 1 1 "the note names the pool in pages";; *) check 1 0 "the note names the pool in pages";; esac
case "$n" in *"DFLASH_TOKENS=7"*) check 1 1 "the note names k";; *) check 1 0 "the note names k";; esac
case "$n" in *"GPU KV cache size"*) check 1 1 "the note points at the engine's own authority";; *) check 1 0 "the note points at the engine's authority";; esac
case "$n" in *"before any"*) check 1 1 "the note separates state from attention pages";; *) check 1 0 "the note separates state from attention";; esac

echo "== nothing else about the launcher moved"
# The pool/stub makes a full dry run cheap; the exit code must stay 0.
(cd "$T" && env -u PRINT_ARGV SPEC=dflash2 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" \
   bash single-user/start_qwen.sh >/dev/null 2>&1 </dev/null); check 0 "$?" "dry run still exits 0"
bash "$HERE/test_bool_knobs.sh" >"$T/knobs.out" 2>&1; check 0 "$?" "bench/test_bool_knobs.sh still passes"

echo; [ "$FAILS" = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }