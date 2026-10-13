#!/bin/bash
# Per-boot driver for the 2026-10-13 decode-speed arms. Run from the repo root in Git Bash.
#   arm.sh boot   <dir> <label>          copy <dir>/.env.<label> to ./.env, start the counter sampler, up, wait for /health
#   arm.sh bench  <dir> <label> <n>      bench/run_benchmarks.sh single -> <dir>/raw-rb-<label>-<n>.log (+ cohort logs)
#   arm.sh ladder <dir> <label> <args..> bench/conc_ladder.py <args> -> <dir>/raw-ladder-<label>.log (+ json)
#   arm.sh exec   <dir> <name> <cmd>     run <cmd> in the container (bash -lc) -> <dir>/raw-<name>.log
#   arm.sh down   <dir> <label>          stop the sampler, compose down
set -u
export MSYS_NO_PATHCONV=1
cd "$(git rev-parse --show-toplevel)"
TOOLS=bench/results/2026-10-13-tools
DC=(docker compose --profile single)
op=$1; dir=$2; label=$3; shift 3
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" | tee -a "$dir/cmd.txt"; }
smi() { nvidia-smi -i 0 --query-gpu=memory.total,memory.used,memory.free,power.limit,power.draw,temperature.gpu,clocks.sm --format=csv; }

winp() { local p=${1//\//\\}; echo "G:\\dev\\HyperQwen\\$p"; }
case $op in
boot)
  cp "$dir/.env.$label" .env
  rm -f "$dir/.sampler-stop-$label"
  log "[$label] cp $dir/.env.$label .env"
  log "[$label] sampler: $TOOLS/sampler.ps1 -> $dir/mem-trace-$label.csv"
  powershell.exe -NoProfile -Command "Start-Process -WindowStyle Hidden powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','$(winp $TOOLS/sampler.ps1)','-Out','$(winp "$dir/mem-trace-$label.csv")','-Stop','$(winp "$dir/.sampler-stop-$label")'"
  { echo "# before boot $(ts)"; smi; } > "$dir/raw-smi-$label.log"
  # compose up from Git Bash's bash.exe fails here with "invalid volume specification: 'G:/models:/app/models:rw'";
  # the same command from PowerShell creates the bind mount, so up runs through PowerShell.
  log "[$label] (PowerShell) docker compose --profile single up -d --force-recreate --no-deps single"
  powershell.exe -NoProfile -Command "Set-Location G:\\dev\\HyperQwen; docker compose --profile single up -d --force-recreate --no-deps single 2>&1 | Out-String" | tee -a "$dir/cmd.txt"
  t0=$(date +%s)
  while :; do
    if curl -sf -o /dev/null http://127.0.0.1:18020/health; then break; fi
    rc=$(docker inspect -f '{{.RestartCount}} {{.State.Status}}' hyperqwen-single-1 2>/dev/null)
    case "$rc" in 0\ running|0\ created) ;; *) log "[$label] container state: $rc"; "${DC[@]}" logs --no-color single > "$dir/raw-boot-$label.log" 2>&1; "${DC[@]}" stop single; echo BOOT_FAILED; exit 1;; esac
    [ $(( $(date +%s) - t0 )) -gt 2400 ] && { log "[$label] health timeout"; "${DC[@]}" logs --no-color single > "$dir/raw-boot-$label.log" 2>&1; echo BOOT_TIMEOUT; exit 1; }
    sleep 10
  done
  log "[$label] healthy after $(( $(date +%s) - t0 )) s"
  "${DC[@]}" logs --no-color single > "$dir/raw-boot-$label.log" 2>&1
  { echo "# after boot, before any request $(ts)"; smi; } >> "$dir/raw-smi-$label.log"
  "${DC[@]}" exec -T single bash -lc 'tr "\0" " " < /proc/$(pgrep -f "vllm serve" | head -1)/cmdline; echo; nvidia-smi -L; env | grep -E "^(CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|VLLM_|MODEL=)" | sort' > "$dir/raw-cmdline-$label.log" 2>&1
  docker image inspect "$(docker inspect -f '{{.Image}}' hyperqwen-single-1)" --format '{{.Id}} {{json .RepoTags}}' > "$dir/raw-image-$label.log"
  echo BOOT_OK ;;
bench)
  n=$1
  log "[$label] ${DC[*]} exec -T single env OUT=/tmp/rb-$label-$n bash bench/run_benchmarks.sh single"
  { echo "# start $(ts)"; "${DC[@]}" exec -T single env OUT=/tmp/rb-$label-$n bash bench/run_benchmarks.sh single 2>&1; echo "# end $(ts) exit=$?"; } > "$dir/raw-rb-$label-$n.log"
  "${DC[@]}" cp single:/tmp/rb-$label-$n "$dir/raw-rb-$label-$n" >/dev/null 2>&1
  { echo "# after bench $label-$n $(ts)"; smi; } >> "$dir/raw-smi-$label.log"
  grep '^ROW' "$dir/raw-rb-$label-$n.log" ;;
ladder)
  log "[$label] ${DC[*]} exec -T single venv/bin/python bench/conc_ladder.py $* --json /tmp/ladder-$label.json"
  { echo "# start $(ts)"; "${DC[@]}" exec -T single venv/bin/python bench/conc_ladder.py "$@" --json /tmp/ladder-$label.json 2>&1; echo "# end $(ts) exit=$?"; } > "$dir/raw-ladder-$label.log"
  "${DC[@]}" cp single:/tmp/ladder-$label.json "$dir/raw-ladder-$label.json" >/dev/null 2>&1
  { echo "# after ladder $label $(ts)"; smi; } >> "$dir/raw-smi-$label.log"
  cat "$dir/raw-ladder-$label.log" ;;
exec)
  log "[$label] ${DC[*]} exec -T single bash -lc '$*'"
  { echo "# start $(ts)"; "${DC[@]}" exec -T single bash -lc "$*" 2>&1; echo "# end $(ts) exit=$?"; } > "$dir/raw-$label.log"
  cat "$dir/raw-$label.log" ;;
down)
  "${DC[@]}" logs --no-color single > "$dir/raw-serverlog-$label.log" 2>&1
  log "[$label] ${DC[*]} down"
  "${DC[@]}" down 2>&1 | tee -a "$dir/cmd.txt"
  touch "$dir/.sampler-stop-$label"
  sleep 12; rm -f "$dir/.sampler-stop-$label"
  echo DOWN_OK ;;
esac
