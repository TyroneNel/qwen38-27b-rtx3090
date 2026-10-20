#!/bin/bash
# The context-profile sweep on the 3090 PC (Windows, WSL2, Docker Desktop). Run it from Git Bash at the repo root.
#   sweep.sh init   <D> <image> <luid>   write <D>/run.env and docker-compose.override.yml, back up .env
#   sweep.sh dryrun <D>                  every arm's launch through the stub vllm in a throwaway container
#   sweep.sh run    <D>                  every arm, every pass, until analyze.py's plan says done; resumes after a crash
#   sweep.sh boot   <D> <label>          one measured boot of <D>/<label>/env (run calls this)
#   sweep.sh report <D>                  <D>/summary.md
# <D> is relative to the repo root. analyze.py makes every decision; this file moves files and processes.
# docker compose up and run go through PowerShell: from Git Bash they fail on the G:/models bind mount.
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)" || exit 2
T=bench/ctx_sweep
CT=hyperqwen-single-1
DC=(docker compose --profile single)
GPU=$(sed -n 's/^NVIDIA_VISIBLE_DEVICES=//p' $T/base.env)
Q="--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
op=${1:-}; D=${2:-}
case $D in ""|/*|*:*|*..*) sed -n '2,8s/^# \{0,1\}//p' "$0"; exit 2 ;; esac
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" | tee -a "$D/sweep.log"; }
ps1() { powershell.exe -NoProfile -Command "Set-Location '$(cygpath -w "$PWD")'; $1" < /dev/null; }
smi() { nvidia-smi -i "$GPU" --query-gpu=memory.total,memory.used,memory.free,power.limit,power.draw,temperature.gpu,clocks.sm --format=csv; }
clk() { local a v c; a=$(date +%s.%N); v=$(docker exec $CT cut -d' ' -f1 /proc/uptime); c=$(date +%s.%N); echo "$a $v $c"; }
# an [lean] <paths> -- <args>: analyze.py inside the image on a copy of bench/ctx_sweep and <paths>.
an() {
  local x=() p=()
  [ "$1" = lean ] && { x=(--exclude='*.csv' --exclude='*.log'); shift; }
  while [ "$1" != -- ]; do p+=("$1"); shift; done; shift
  tar -c "${x[@]}" $T "${p[@]}" | docker run -i --rm --entrypoint bash "$IMG" -c \
    'mkdir -p /tmp/w && cd /tmp/w && tar -x && exec /app/venv/bin/python bench/ctx_sweep/analyze.py "$@"' an "$@" | tr -d '\r'
}
kvline() {
  [ -f "$B/.kvline-$L" ] && return
  local kv
  kv=$("${DC[@]}" logs --no-color --timestamps single 2>/dev/null | grep -m1 'GPU KV cache size' | awk '{print $3}')
  [ -n "$kv" ] && date -d "$kv" +%Y-%m-%dT%H:%M:%S > "$B/.kvline-$L"
}
cell() {
  local n=$1 rc
  [ -f "$B/SPILL-$L" ] && { log "[$L] skipped $n: $(cat "$B/SPILL-$L")"; return; }
  log "[$L] cell $n: $2"
  {
    echo "CLOCK $(clk)"
    echo "# start $(ts)"
    "${DC[@]}" exec -T single bash -lc "nvidia-smi -i $GPU $Q -l 1 > /tmp/gpu-$n.csv 2>&1 & g=\$!; $2; s=\$?; kill \$g; sed 's/^/GPU, /' /tmp/gpu-$n.csv; exit \$s" < /dev/null 2>&1
    rc=$?
    echo "# end $(ts) exit=$rc"
    echo "CLOCK $(clk)"
  } > "$B/cell-$n.log"
}
[ "$op" = init ] || . "$D/run.env" || exit 2
case $op in
init)
  IMG=${3:?image}; LUID=${4:?luid}
  [ -e "$D" ] && { echo "$D exists"; exit 2; }
  [ -e docker-compose.override.yml ] && { echo "docker-compose.override.yml exists: move it aside first"; exit 2; }
  id=$(docker image inspect -f '{{.Id}}' "$IMG") || exit 2
  mkdir -p "$D"
  printf 'IMG=%s\nIMG_ID=%s\nLUID=%s\nPASSES=2\n' "$IMG" "$id" "$LUID" > "$D/run.env"
  printf 'services:\n  single:\n    image: %s\n    pull_policy: never\n    volumes:\n      - ./%s:/sweep:ro\n' "$IMG" $T > docker-compose.override.yml
  if [ -f .env ] && [ ! -e .env.ctx-sweep-backup ]; then cp -p .env .env.ctx-sweep-backup; fi
  cat "$D/run.env" docker-compose.override.yml ;;
dryrun)
  mkdir -p "$D/dryrun"
  mapfile -t A < <(an -- arms)
  [ ${#A[@]} -gt 0 ] || { echo "analyze.py arms printed nothing"; exit 2; }
  for a in "${A[@]}"; do
    name=${a%%$'\t'*}; envs=${a#*$'\t'}; envs=${envs#*$'\t'}
    { cat $T/base.env; printf '%s\n' "$envs" | tr '\t' '\n'; } > .env
    ps1 "docker compose --profile single run --rm -T --no-deps --entrypoint bash single -c 'cp /sweep/vllm_stub venv/bin/vllm && exec bash docker/entrypoint.sh single'" \
      > "$D/dryrun/$name.out" 2> "$D/dryrun/$name.err"
    log "dryrun $name: $(grep -c '^ARG ' "$D/dryrun/$name.out") args"
  done
  an "$D/dryrun" -- dryrun "$D" > "$D/dryrun.txt"
  cat "$D/dryrun.txt"
  [ "$(tail -1 "$D/dryrun.txt")" = DRYRUN_OK ] || exit 1 ;;
boot)
  L=${3:?label}; B=$D/$L
  [ -f "$B/env" ] || { echo "no $B/env"; exit 2; }
  cp "$B/env" .env
  log "[$L] boot: $(grep -v -x -F -f $T/base.env "$B/env" | tr '\n' ' ')"
  powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','$(cygpath -w $T/sampler.ps1)','-Out','$(cygpath -w "$B/mem-trace-$L.csv")','-Stop','$(cygpath -w "$B/.sampler-stop-$L")'" < /dev/null
  { echo "# before boot $(ts)"; smi; } > "$B/smi.log"
  ps1 "docker compose --profile single up -d --force-recreate --no-deps single 2>&1 | Out-String" >> "$B/cmd.txt"
  LUID=$LUID nohup bash $T/spillwatch.sh "$B" "$L" > /dev/null 2>&1 &
  t0=$(date +%s); status=
  while [ -z "$status" ]; do
    kvline
    if curl -sf -o /dev/null http://127.0.0.1:18020/health; then status=OK; break; fi
    st=$(docker inspect -f '{{.RestartCount}} {{.State.Status}}' $CT 2>&1)
    case $st in "0 running"|"0 created") ;; *) status="FAILED container state: $st" ;; esac
    [ $(( $(date +%s) - t0 )) -gt 2400 ] && status="FAILED no /health after 2400 s"
    [ -n "$status" ] || sleep 5
  done
  kvline
  echo "$status" > "$B/boot.status"
  log "[$L] boot $status after $(( $(date +%s) - t0 )) s"
  "${DC[@]}" logs --no-color single > "$B/boot.log" 2>&1
  { echo "# after boot $(ts)"; smi; } >> "$B/smi.log"
  docker inspect -f '{{.Image}}' $CT > "$B/image.txt" 2>&1
  if [ "$status" = OK ]; then
    "${DC[@]}" exec -T single bash -c 'p=$(pgrep -f "[v]llm serve" | head -1)
      tr "\0" "\n" < /proc/$p/cmdline | sed -n "/^serve\$/,\$p" | sed "s/^/ARG /"
      tr "\0" "\n" < /proc/$p/environ | grep -E "^(VLLM_|KVARN_|PYTORCH_CUDA)" | grep -v KEY | sort | sed "s/^/ENV /"' \
      < /dev/null > "$B/live.txt" 2>&1
    "${DC[@]}" exec -T single bash -c 'nvidia-smi -L; nvidia-smi --query-gpu=name,compute_cap,power.limit --format=csv,noheader' \
      < /dev/null > "$B/gpu.txt" 2>&1
    docker inspect -f 'after boot: {{.RestartCount}} {{.State.StartedAt}}' $CT >> "$B/restarts.txt"
    sleep 30   # the spill guard's floor window runs from the KV-cache line to the first request
    date +%Y-%m-%dT%H:%M:%S > "$B/.firstreq-$L"
    sleep 22
    mapfile -t C < <(an "$B/live.txt" -- cells "$D" "$L")
    for c in "${C[@]}"; do cell "${c%%$'\t'*}" "${c#*$'\t'}"; done
    docker inspect -f 'before down: {{.RestartCount}} {{.State.StartedAt}}' $CT >> "$B/restarts.txt"
  fi
  "${DC[@]}" logs --no-color single > "$B/serverlog.log" 2>&1
  "${DC[@]}" down >> "$B/cmd.txt" 2>&1
  touch "$B/.sampler-stop-$L"
  sleep 12
  an "$D/run.env" "$B" -- boot "$D" "$L" > "$B/analysis.txt"
  tail -1 "$B/analysis.txt" > "$B/verdict"
  grep -qE '^(BOOT_|RETRY_MAX_LEN )' "$B/verdict" || { log "[$L] analyze.py boot printed no verdict"; rm -f "$B/verdict"; exit 2; }
  log "$(cat "$B/verdict")" ;;
run)
  [ "$(tail -1 "$D/dryrun.txt" 2>/dev/null)" = DRYRUN_OK ] || { echo "sweep.sh dryrun must end DRYRUN_OK first"; exit 2; }
  for b in "$D"/*-p*-*/; do
    b=${b%/}
    case $b in *-aborted|*'*'*) continue ;; esac
    [ -f "$b/verdict" ] && continue
    log "recovery: $b has no verdict, stopping its sampler and renaming it $b-aborted"
    touch "$b/.sampler-stop-${b##*/}"; sleep 12
    mv "$b" "$b-aborted"
    "${DC[@]}" down >> "$D/sweep.log" 2>&1
  done
  mapfile -t A < <(an -- arms)
  names=(); for a in "${A[@]}"; do names+=("${a%%$'\t'*}"); done
  for p in $(seq 1 "$PASSES"); do
    order=("${names[@]}")
    if [ $(( p % 2 )) = 0 ]; then order=(); for (( i=${#names[@]}-1; i>=0; i-- )); do order+=("${names[i]}"); done; fi
    for arm in "${order[@]}"; do
      while :; do
        mapfile -t P < <(an lean "$D" -- plan "$D" "$arm" "$p")
        case ${P[0]:-} in
          BOOT\ *)
            L=$(echo "${P[0]}" | cut -d' ' -f2)
            mkdir -p "$D/$L"
            { cat $T/base.env; printf '%s\n' "${P[@]:1}"; } > "$D/$L/env"
            bash $T/sweep.sh boot "$D" "$L" || exit 2 ;;
          DONE\ *|BROKEN\ *) log "${P[0]}"; break ;;
          *) log "plan: ${P[*]:-no output}"; exit 2 ;;
        esac
      done
    done
  done
  bash $T/sweep.sh report "$D" ;;
report)
  an lean "$D" -- report "$D" > "$D/summary.md"
  cat "$D/summary.md" ;;
*) sed -n '2,8s/^# \{0,1\}//p' "$0"; exit 2 ;;
esac
