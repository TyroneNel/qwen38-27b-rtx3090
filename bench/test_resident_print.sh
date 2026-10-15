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
     bash single-user/start_qwen.sh 2>/dev/null </dev/null) | grep -o 'the pinned pool holds about [0-9]* resident requests on short (~1k-token) prompts and about [0-9]* at MAX_LEN=[0-9]*' || true
}
# seats <env...>: just the two numbers, "1k:len".
seats() { run "$@" | sed -E 's/.*about ([0-9]+) resident.*about ([0-9]+) at.*/\1:\2/'; }
# note <env...>: the whole note, so the parts beyond the seat counts can be checked.
note() {
  (cd "$T" && env -u PRINT_ARGV -u DFLASH_TOKENS -u KV_MEM -u MAX_SEQS -u MAX_LEN \
     SPEC=dflash2 PREFIX_CACHE=0 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" "$@" \
     bash single-user/start_qwen.sh 2>/dev/null </dev/null) | grep -o '\[start_qwen\] note: the pinned pool.*' || true
}

echo "== the formula matches the measurement it came from"
# batch 4's k=7 boot: KV_MEM=5315022028, 364 pages, 6(k+2)=54, +2*ceil(1024/864)=4 -> 58; 363/58=6
# at MAX_LEN=65536: 54 + 2*76 = 206; 363/206 = 1
check "6:1" "$(seats KV_MEM=5315022028 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" \
      "k=7 at batch 4's pin: 6 at ~1k, 1 at 65536 (measured 6 held; the engine's own label reads 1.69)"
check "6:1" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=8 MAX_LEN=65536)" \
      "shipped pin, MAX_SEQS=8: still prints (8 > 6)"

echo "== the context term is what moves it, and the note says so"
# pool 382 pages at the shipped pin; 6(k+2)=54
check "6:1" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" "65536 -> 54+152=206 -> 1 seat"
check "6:2" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=32768)"  "32768 -> 54+76=130 -> 2 seats (measured 2)"
check "6:5" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=4096)"   "4096 -> 54+10=64 -> 5 seats (measured 5)"
check "6:6" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=1024)"   "1024 -> 54+4=58 -> 6 seats"

echo "== k changes the state term: 6*(k+2) pages"
check "17:2" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=1 MAX_SEQS=20 MAX_LEN=65536)" "k=1: 18+4=22 -> 17 at ~1k; 18+152=170 -> 2 at 65536"
check "11:2" "$(seats KV_MEM=5583457484 DFLASH_TOKENS=3 MAX_SEQS=16 MAX_LEN=65536)" "k=3: 30+4=34 -> 11 at ~1k"
check "6:1"  "$(seats KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536)" "k=7: 54+4=58 -> 6 at ~1k"

echo "== silent when MAX_SEQS already fits, and where the fit is unmeasured"
check "" "$(run KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=4 MAX_LEN=65536)" \
      "MAX_SEQS=4 <= 6 seats: no note"
check "" "$(run KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=6 MAX_LEN=65536)" \
      "MAX_SEQS=6 == 6 seats: no note (the boundary is 'admits more than')"
check "" "$(run KV_MEM=4900000000 DFLASH_TOKENS=3 MAX_SEQS=16 CTX=huge)" \
      "CTX=huge: skipped (KVarN block and page are unmeasured)"
[ -n "$(run KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536 CTX=long)" ] \
  && check 1 1 "CTX=long still prints" || check 1 0 "CTX=long still prints"

echo "== the note stays auditable"
n="$(note KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=8 MAX_LEN=65536)"
case "$n" in *"54+2*ceil(ctx/864)"*) check 1 1 "the note names the per-request formula";; *) check 1 0 "the note names the per-request formula";; esac
case "$n" in *"the pool's 382 pages"*) check 1 1 "the note names the pool in pages (5583457484/14601709=382)";; *) check 1 0 "the note names the pool in pages (382)";; esac
case "$n" in *"DFLASH_TOKENS=7"*) check 1 1 "the note names k";; *) check 1 0 "the note names k";; esac
case "$n" in *"MAX_LEN=65536"*) check 1 1 "the note names the context it used";; *) check 1 0 "the note names the context";; esac

echo "== nothing else about the launcher moved"
# The pool/stub makes a full dry run cheap; the exit code must stay 0.
(cd "$T" && env -u PRINT_ARGV SPEC=dflash2 PRINT_ARGV=1 VLLM_API_KEY=k HOME="$T" \
   bash single-user/start_qwen.sh >/dev/null 2>&1 </dev/null); check 0 "$?" "dry run still exits 0"
bash "$HERE/test_bool_knobs.sh" >"$T/knobs.out" 2>&1; check 0 "$?" "bench/test_bool_knobs.sh still passes"

echo; [ "$FAILS" = 0 ] && echo "all checks passed" || { echo "$FAILS checks FAILED"; exit 1; }