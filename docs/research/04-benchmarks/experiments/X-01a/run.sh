#!/bin/bash
# X-01a: what does resolve_config.sh resolve for unset-PTH_FA launches on sm86?
# CPU only. Needs two detached worktrees (upstream/main 63f7581, ours 95ac251) with a fake venv/bin/python
# (exit 0 = pth_flash_attn installed) and an empty models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors.
# The cc86 stub is the CI stub from 433b050 (.github/workflows/patch-integrity.yml), byte for byte.
HERE=$(cd "$(dirname "$0")" && pwd)
WT=${WT:-$HOME/research-wt}
row() {  # row <label> <env assignments...>
  local tree=$1 stub=$2 label=$3; shift 3
  local repo=$WT/x01a-$tree
  local out
  out=$(cd "$repo" && env -i HOME="$HOME" PATH="$HERE/$stub:/usr/bin:/bin" REPO="$repo" "$@" \
    bash -c 'source "$REPO/resolve_config.sh"; resolve_effective_config single; rc=$?
             echo "RESULT rc=$rc PTH_FA=${PTH_FA:-unset} FA2_FP8KV=${FA2_FP8KV:-unset}"' 2>&1)
  local res; res=$(printf '%s\n' "$out" | grep '^RESULT' )
  [ -n "$res" ] || res="RESULT refused (exit before print)"
  printf '%-9s | %-48s | %s\n' "$tree" "$label" "${res#RESULT }"
  printf '%s\n' "$out" | grep -v '^RESULT' | grep -v 'effective-config\] \(MODE\|MODEL\|MAX_LEN\|VLLM_API\)' | sed 's/^/            > /'
}
cases() {  # cases <tree>
  local t=$1
  echo "---- scope: SPEC x CTX, cc8.6 stub, package present, no other knobs"
  row $t stub-cc86 "SPEC=mtp CTX=long"      SPEC=mtp CTX=long
  row $t stub-cc86 "SPEC=mtp CTX=fast"      SPEC=mtp CTX=fast
  row $t stub-cc86 "SPEC=dflash2 CTX=long"  SPEC=dflash2 CTX=long
  row $t stub-cc86 "SPEC=dflash2 CTX=fast"  SPEC=dflash2 CTX=fast
  row $t stub-cc86 "SPEC=mtp CTX=huge"      SPEC=mtp CTX=huge
  row $t stub-cc86 "defaults (CTX/SPEC unset)" 
  echo "---- blockers on SPEC=mtp CTX=long"
  row $t stub-cc86 "+ VISION=1"             SPEC=mtp CTX=long VISION=1
  row $t stub-cc86 "+ DRAFT_TOKENS=4"       SPEC=mtp CTX=long DRAFT_TOKENS=4
  row $t stub-cc86 "+ DRAFT_TOKENS=3 (explicit, = default)" SPEC=mtp CTX=long DRAFT_TOKENS=3
  row $t stub-cc86 "+ DYNAMIC_K=1"          SPEC=mtp CTX=long DYNAMIC_K=1
  row $t stub-cc86 "+ KV_OFFLOAD_GB=8"      SPEC=mtp CTX=long KV_OFFLOAD_GB=8
  row $t stub-cc86 "+ FA2_FP8KV=0"          SPEC=mtp CTX=long FA2_FP8KV=0
  row $t stub-cc86 "+ FA2_FP8KV=1"          SPEC=mtp CTX=long FA2_FP8KV=1
  row $t stub-cc86 "+ EXTRA_ARGS=--kv-cache-dtype fp8"  SPEC=mtp CTX=long "EXTRA_ARGS=--kv-cache-dtype fp8"
  row $t stub-cc86 "+ EXTRA_ARGS=--attention-backend X" SPEC=mtp CTX=long "EXTRA_ARGS=--attention-backend X"
  row $t stub-cc86 "+ EXTRA_ARGS=--speculative-config {}" SPEC=mtp CTX=long "EXTRA_ARGS=--speculative-config {}"
  row $t stub-cc86 "+ PTH_FA=0 (opt out)"   SPEC=mtp CTX=long PTH_FA=0
  row $t stub-cc86 "+ PTH_FA=1 (explicit)"  SPEC=mtp CTX=long PTH_FA=1
  row $t stub-cc86 "+ PTH_FA= (empty)"      SPEC=mtp CTX=long PTH_FA=
  row $t stub-cc86 "+ package missing"      SPEC=mtp CTX=long FAKE_PTHFA_MISSING=1
  row $t stub-cc89 "+ cc 8.9 stub"          SPEC=mtp CTX=long
  row $t "stub-none" "+ no nvidia-smi (CI as shipped)" SPEC=mtp CTX=long
  row $t stub-cc86 "box .env: CTX=fast VISION=1" CTX=fast VISION=1
  row $t stub-cc86 "box .env + CTX=long (VISION=1 kept)" CTX=long VISION=1 SPEC=mtp
  echo "---- blockers on SPEC=dflash2 CTX=long"
  row $t stub-cc86 "+ VISION=1"             SPEC=dflash2 CTX=long VISION=1
  row $t stub-cc86 "+ DFLASH_TOKENS=15"     SPEC=dflash2 CTX=long DFLASH_TOKENS=15
  row $t stub-cc86 "+ LOOKUP=0"             SPEC=dflash2 CTX=long LOOKUP=0
  row $t stub-cc86 "+ DYNAMIC_K=1"          SPEC=dflash2 CTX=long DYNAMIC_K=1
  row $t stub-cc86 "+ DRAFT=x"              SPEC=dflash2 CTX=long DRAFT=x
  row $t stub-cc86 "+ FA2_FP8KV=0"          SPEC=dflash2 CTX=long FA2_FP8KV=0
}
mkdir -p "$HERE/stub-none"
for t in upstream ours; do
  echo "================ $t: $(git -C "$WT/x01a-$t" log --oneline -1)"
  cases $t
done

# ---- the repo's own launcher tests, CI-style: stub-less pass, then cc8.6-stub pass (CI step in 433b050)
# Run in a pristine copy of the tests' own fixtures: the tests build their own fake REPOs, but run from the
# worktree root. Upstream's nine-test list is used for both trees (ours lacks nothing: all nine files exist).
for t in upstream ours; do
  for mode in nostub cc86; do
    echo "================ launcher tests: $t / $mode"
    P=/usr/bin:/bin; [ $mode = cc86 ] && P="$HERE/stub-cc86:$P"
    for n in no_key_bind prefix_cache_knob kv_offload_knob dynamic_k_knob tp_size bool_knobs resident_print fa2_fp8kv_knob pthfa_knob; do
      ( cd "$WT/x01a-$t" && PATH="$P" timeout 300 bash "bench/test_$n.sh" > "$HERE/.t.log" 2>&1 ); rc=$?
      printf '%-9s %-6s test_%-18s rc=%s  %s\n' $t $mode $n $rc "$(tail -n1 "$HERE/.t.log" | cut -c1-110)"
      [ $rc = 0 ] || sed 's/^/      | /' "$HERE/.t.log" | tail -n 12
    done
  done
done
rm -f "$HERE/.t.log"
