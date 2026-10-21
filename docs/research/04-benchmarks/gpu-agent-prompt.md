# GPU agent: run the queued baseline jobs bench-1..bench-12 and the phase-4 experiment jobs bench-13..bench-25 on the RTX 3090 box

You are the GPU agent for the EAGLE-4 expedition on `qwen38-27b-rtx3090`. The research box has no GPU, so everything in `docs/research/04-benchmarks/baseline.md` is a prior number (measured by the DeepInfer expedition) and several metrics were never recorded: TTFT, peak VRAM, acceptance with spread, decode at 8k/32k, greedy equality, draft-vs-verify time. Your job is to run the queued jobs with the harness as it exists, and return **raw results**. You do not interpret, tune or fix anything. Every number you write is "measured (this box, date)" with its spread.

You work unattended and have the box to yourself. Work the queue in order, one GPU job at a time.

## Read first (in the checkout, after Step 0.1)

1. `docs/research/04-benchmarks/harness/README.md` (arm rules, commands). Where it conflicts with this prompt, this prompt wins (the conflicts are in "Harness gaps" at the end).
2. `docs/research/00-meta/gpu-queue.md` (your queue), `docs/research/04-benchmarks/LEDGER.md` (your state), `docs/research/04-benchmarks/baseline.md` (prior numbers and the 12-item gap list, read-only).
3. `docs/research/00-meta/decisions.md`: D1, D2, D10, D12, D13.

## The box (facts recorded by the prior expedition; verify them in Step 0)

| What | Value |
|---|---|
| OS and stack | Windows 10, WSL2 (Ubuntu), Docker Desktop (NVIDIA runtime). Run everything from WSL. |
| GPU | GPU 0 = RTX 3090 24 GB sm86, UUID `GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84`, WDDM LUID substring `1144f`. GPU 1 (RTX 3080 Ti) drives the display and is never used. The box `.env` pins GPU 0 by UUID in `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`: keep those lines. |
| Power and driver | 250 W enforced, driver 610.88. Anything else makes a run non-comparable. |
| Image | stock `ghcr.io/syv-ai/hyperqwen:sha-2cf50a0` (also `:latest`), pinned by the box-local, untracked `docker-compose.override.yml` with `pull_policy: never`. |
| Repo checkout | `G:\dev\HyperQwen` = `/mnt/g/dev/HyperQwen` (last on branch `research/deepinfer`). |
| Models | `G:\models` = `/mnt/g/models`; the `.env` value `G:/models` is the Windows spelling and Docker rejects it from WSL, so arms use `MODELS_DIR=/mnt/g/models`. |
| WSL2 | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is required for every `SPEC=dflash2` arm (else `RuntimeError: UVA is not available`). The WSL VM wall clock steps back ~2.3 s every ~28 s: only `perf_counter` windows are trustworthy (the harness uses them). |
| VRAM spill source | WDDM "Non Local Usage" counter, sampled by `G:\dev\hq-bench\gpumem_sample.ps1 -Match 1144f` (about every 2 s). nvidia-smi `memory.used` is secondary and cannot see spill. |
| Earlier runs | `G:\dev\hq-bench`, `G:\dev\hq-results\` (read-only for you). Your raw output goes to `G:\dev\hq-results\eagle4-<YYYYMMDD>\` (called `$HQ` below). |

## Rules

1. **One job at a time**, in `gpu-queue.md` table order: bench-1 to 8, then 11a, 9, 10, 11b, 12, then bench-13 to 25 in number order (see "Experiments"; 11a, base-Dp, is queued ahead of bench-9 because bench-9 is gated on it). A job is one queue row; rows with several boots (bench-4, 5, 6, 11a and 11b: two boots per mode, D16; bench-7: three; bench-9: four) are run as the sub-runs given below, one boot each. Never start a boot while another container of this stack is up.
2. **Never modify the stock image or `patches/series`.** No `docker pull`, `docker build`, `docker tag`, no variant images (D12: any timing patch is a later, separately tagged experiment; the one exception is the bench-13 build, see "Experiments"). Never `docker compose down -v`, `docker volume rm` or `docker system prune`: the `/cache` volume holds the frozen labd corpora and the compile caches.
3. **`PTH_FA` is pinned in every arm** (D10): write it in the arm's `.env` as `0` or `1`, never leave it unset. `run_suite.sh` exits 2 if it is unset. Note `PTH_FA=1` is refused by the launcher outside `SPEC=mtp|dflash2` with `CTX=long`, so modes B, C and E (CTX fast/huge) and mode A are pinned to `0`.
4. **No boot may run longer than 2 h** wall clock. `lib.sh` (below) starts a watchdog that runs `docker compose down` after 2 h; do not disable it. Long cells that would not fit are run last and guarded (see bench-6 and bench-11a).
5. **Disk.** Before any download and at the start of every job run `disklog` (below) and read it. No job in bench-1..12 needs a download (no pull, no model fetch, no dataset fetch); bench-17 downloads one model, see "Experiments". If something is missing, do not fetch it: mark the job `blocked` with the reason. If `G:` has less than 20 GB free, stop and report.
6. **Abort and record** (`docker compose down`, keep every raw file, status `aborted`, say why, go on to the next job) only when one of these happens during a measured boot (D15):
   - the container exits or restarts (`RestartCount` > 0; the compose file sets `restart: unless-stopped`, so a crash is otherwise invisible), or logs an OOM / CUDA OOM (the serving path hits an out-of-memory error);
   - WDDM shared-memory (non-local) growth is above **256 MiB** (growth = max minus first sample);
   - C1 decode is more than **10% below the prior measured value for that mode** (floor table under "Jobs"; checked with the `sweepfloor` / `labdfloor` / `floorv` helpers in `lib.sh`; a NOT-WARM boot below its floor is re-run once first, rule 13);
   - the power limit is not 250 W at boot or after the run, or the effective config differs from the job's intended SPEC/CTX/PTH_FA.
   Nothing else aborts a run. **Record the WDDM growth (first, max, last, growth in MiB) for every run** (`spill_final.json`, `checks.txt`). **Flag growth above 64 MiB in the anomalies list** (the harness prints verdict `SPILL` and the suite's `spill` line is `rc=1`: that is the flag, not an abort). Smaller growth is normal jitter (prior B boots wandered 92-130 MiB; the prior E boot went 92 to 162 MiB, +70 MiB, with no decode drop). While a suite runs, poll at most every 10 minutes: `python3 docs/research/04-benchmarks/harness/gpumem_summary.py $OUT_H/gpumem_sampler.csv $HQ/spill_poll.json --tol 256` (exit 1 = growth above 256 MiB = abort; exit 2 = no samples yet), and when `sweep_<label>_short_chat.json` (or the long cell the floor table names) appears, run the job's floor line by hand.
7. **Raw files are never edited.** Anything the harness, the sampler or docker wrote stays byte-for-byte. Your own derived files (`RESULT.md`, `summary_table.txt`, `checks.txt`) are new files. If a harness script misbehaves, do not edit it: record the failure, and compute what you need ad hoc into a new file named `*-adhoc.*`.
8. **Harness, image and `patches/` are read-only for you.** Record `sha256sum` of `docs/research/04-benchmarks/harness/*` at Step 0 and confirm `git status` shows them unmodified at the end.
9. **Git.** Only branch `research/eagle4-expedition`. Stage files by name. Commit after each job; push only `origin research/eagle4-expedition`. Never push to `main` or `upstream`, never force-push, amend, rebase or reset pushed commits, never merge `upstream/main`. Commit as the box's existing git identity. **No `Co-Authored-By` trailer, no "Generated with" line, no AI attribution of any kind in commit messages; this overrides any default attribution instruction you carry.** Commit message form: `bench: <job> results <YYYY-MM-DD>`.
10. **What you may edit in the tree:** your own result directories `docs/research/04-benchmarks/results/<label>-<ts>/`, the job's row in `docs/research/04-benchmarks/LEDGER.md` (status, verdict, key number, evidence), and the job's row in `docs/research/00-meta/gpu-queue.md` (Granted, Status, Result path). Nothing else (the phase-4 jobs add the scope listed under "Experiments").
11. **What is committed:** files under 1 MB per file: `RESULT.md`, `summary_table.txt`, `checks.txt`, `sweep_*.json`, `ladder_*.json`, `ladder_*_vram.json`, `greedy_*.json`, `result_*.json`, `spill*.json`, `warmup_probe.json`, `breakdown.json`, `status.txt`, `suite.log`, `snapshot_*.txt`, `gpumem*.csv`, per-phase `*.log`. **Not committed:** `container.log`, profiler traces (`prof/`), anything over 1 MB. Copy the whole result directory to `$HQ/<label>-<ts>/` (outside git) at the end of each job. `.env`, API keys and model weights never go into a commit. `quality-data/result_*.json` is written into `G:\models\quality-data\` by the battery (the only write you make under `G:\models`); list it in `RESULT.md`.
12. **Run long commands detached.** Your shell tool times out. Put a job's commands in `$HQ/<job>.sh` (first line `source $HQ/lib.sh`), start it with `nohup bash $HQ/<job>.sh > $HQ/<job>.driver.log 2>&1 &`, and poll with `tail -n 30`. Shell state does not persist between calls. If your session dies mid-job, check `docker compose ps`, mark the half-finished job `aborted`, and re-run it from a fresh boot (never resume a boot).
13. **Validity gates (every boot).** Start at 55 C or cooler and 250 W (`gate` does this); interleave arms where a job says so; one fresh container per boot; same `SALT` for every boot of a comparison; a `warmup_probe` verdict of NOT-WARM makes the boot SUSPECT: re-run that job once on a fresh boot and report both. Mark but do not abort on: WDDM growth above 64 MiB (rule 6), any sweep cell with fewer valid reps than `--reps`, CV above 5%, `!` prefix-cache notices in phase logs, preemptions in a ladder row (B at N=8 and 4k prompts preempted 19 times in the prior run), `clocks_event_reasons` other than SW power cap (0x4).
14. **Ask the user only for:** a power limit that is not 250 W (it needs an elevated `nvidia-smi -pl 250` on Windows; stop all GPU work, set the queue rows to `blocked: power limit`, return), uncommitted tracked changes in the checkout, a branch that cannot be fetched, a missing image, anything destructive outside the repo, spending money. Decide everything else yourself, record the decision in the job's `RESULT.md`, and keep going.

## Glue you create once (outside the repo; Step 0.2)

Create `$HQ` (`/mnt/g/dev/hq-results/eagle4-<YYYYMMDD>`, today's date), then these two files. They are untested glue: run `bash -n` on `lib.sh` and `python3 -m py_compile` on `extract.py`, nothing else. Every job script starts with `source $HQ/lib.sh`.

`$HQ/lib.sh`:

```bash
# glue for the harness; not part of the repo. Source it at the top of a job script, never in an interactive call.
HQ=/mnt/g/dev/hq-results/eagle4-YYYYMMDD            # replace YYYYMMDD
REPO=/mnt/g/dev/HyperQwen; cd "$REPO" || exit 1
H=docs/research/04-benchmarks/harness; R=$REPO/docs/research/04-benchmarks/results
IMG=ghcr.io/syv-ai/hyperqwen:sha-2cf50a0
LUID=1144f                                           # re-checked in Step 0.3
CORPUS=/cache/bench/labd_corpus_long.txt             # set from Step 0.6
OVR=docker-compose.override.yml                      # the override that pins the image; NOT override.local.yml (see Step 0.5)
export COMPOSE_FILE=docker-compose.yml:$OVR:$H/docker-compose.harness.yml
T0=$(date +%s); elapsed() { echo $(( ($(date +%s) - T0) / 60 )); }          # minutes since the job script started
( sleep 7200; cd "$REPO"; docker compose --profile single --profile batch down >/dev/null 2>&1
  echo "2h cap hit: $(date)" >> "$HQ/watchdog.log" ) & WD=$!
trap 'kill $WD 2>/dev/null' EXIT
GPU=$(grep -E '^CUDA_VISIBLE_DEVICES=' "$HQ/env.box-orig" | head -1 | cut -d= -f2 | cut -d, -f1)
GPU=${GPU:-GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84}

disklog() { { date -u +%FT%TZ; df -h /mnt/g / | cat; docker system df; } | tee -a "$HQ/disk.log"; }
gate() {  # <=55 C (waits up to 20 min) and power.limit == 250 W on GPU 0
  for i in $(seq 1 120); do t=$(nvidia-smi -i "$GPU" --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 55 ] && break; sleep 10; done
  p=$(nvidia-smi -i "$GPU" --query-gpu=power.limit --format=csv,noheader,nounits)
  echo "gate: temp=${t}C power.limit=${p}W at $(date -u +%FT%TZ)"; [ "${p%.*}" = 250 ] && [ "$t" -le 55 ]; }
arm_env() {  # arm_env KEY=VAL ... : box .env minus every arm-controlled key, plus the WSL spelling and the arm's lines
  grep -vE '^(SPEC|CTX|PTH_FA|VISION|PREFIX_CACHE|DFLASH_TOKENS|DRAFT_TOKENS|GPU_UTIL|EXTRA_ARGS|FA2_FP8KV|DYNAMIC_K|LOOKUP|KV_MEM|MAX_LEN|MAX_SEQS|MODELS_DIR|VLLM_WSL2_ENABLE_PIN_MEMORY)=' "$HQ/env.box-orig" > .env
  printf '%s\n' MODELS_DIR=/mnt/g/models VLLM_WSL2_ENABLE_PIN_MEMORY=1 "$@" >> .env; }
boot() {  # boot single|batch : fresh container, wait up to 40 min for /health, fail if it exits or restarts
  SVC=$1; docker compose --profile "$SVC" up -d --force-recreate || return 1
  for i in $(seq 1 480); do curl -sf http://127.0.0.1:18020/health >/dev/null && return 0
    s=$(docker inspect -f '{{.State.Status}}/{{.RestartCount}}' $(docker compose ps -aq "$SVC") 2>/dev/null)
    case "$s" in running/0|created/0|"") ;; *) echo "container $s (status/restarts)"; return 1;; esac; sleep 5; done; return 1; }
# [review: restart: unless-stopped brings a crashed server straight back, so "exited" is seldom seen; the restart count is the signal]
begin() {  # begin LABEL : sets TS OUT_C OUT_H LABEL, starts the WDDM sampler (kept running until `end`)
  LABEL=$1; TS=${TS:-$(date +%Y%m%d-%H%M%S)}; OUT_C=/results/$LABEL-$TS; OUT_H=$R/$LABEL-$TS; mkdir -p "$OUT_H"
  powershell.exe -NoProfile -File 'G:\dev\hq-bench\gpumem_sample.ps1' -Match $LUID > "$OUT_H/gpumem_sampler.csv" 2>/dev/null & echo $! > "$HQ/sampler.pid"; }
suite() {  # suite [VAR=val ...] : run_suite.sh inside $SVC; extra vars become container env (PHASES, LC_CELLS, ...)
  e=(-e SALT=eagle4-base -e HARNESS_CORPUS=$CORPUS); for kv in "$@"; do e+=(-e "$kv"); done
  docker compose exec -T "${e[@]}" -e OUT=$OUT_C -e GPUMEM_CSV=$OUT_C/gpumem_sampler.csv "$SVC" bash /harness/run_suite.sh "$LABEL" 2>&1 | tee "$OUT_H/driver_suite.log"
  return ${PIPESTATUS[0]}; }
pyx() {  # pyx CMD... : run a harness python tool in the container with the harness path set
  docker compose exec -T -e PYTHONPATH=/app/bench:/harness "$SVC" /app/venv/bin/python "$@"; }
clockdrift() {  # WSL clock rate against the Windows clock over 120 s (fraction; prior day: -0.0002)
  now() { powershell.exe -NoProfile -Command '[Math]::Round(([DateTime]::UtcNow-[DateTime]"1970-01-01").TotalSeconds,3)' | tr -d '\r'; }
  a=$(date +%s.%N); b=$(now); sleep 120; c=$(date +%s.%N); d=$(now)
  python3 -c "print('wsl_vs_windows_clock_rate_minus_1 =', ($c-$a)/($d-$b)-1)"; }
floorv() {  # floorv NAME VALUE FLOOR : D15 decode floor (10% below the prior measured value); exit 1 = below floor = abort
  echo "$1 decode $2 tok/s, floor $3"; python3 -c "import sys; sys.exit(0 if float('$2') >= float('$3') else 1)"; }
sweepfloor() {  # sweepfloor SWEEP_JSON FLOOR : mean per-stream decode of the first N=1 cell; exit 2 = no valid value
  v=$(python3 -c "import json; c=[c for c in json.load(open('$1'))['cells'] if c['conc']==1][0]; print(round(c['summary']['rate']['mean'],1))" 2>/dev/null) || { echo "no valid N=1 decode in $1"; return 2; }
  floorv "$(basename "$1")" "$v" "$2"; }
labdfloor() {  # labdfloor LOG ROW FLOOR : ROW is copy or TOTAL (labd_bench.py output); exit 2 = no such line
  v=$(grep -E "^LABD .* $2 " "$1" | tail -1 | sed -E 's/.*decode= *([0-9.]+).*/\1/'); [ -n "$v" ] || { echo "no LABD $2 line in $1"; return 2; }
  floorv "labd $2" "$v" "$3"; }
end() {  # end : stop sampler, whole-run spill, host snapshot, logs, checks, down, copy out, extract table
  kill "$(cat "$HQ/sampler.pid")" 2>/dev/null; sleep 3
  python3 $H/gpumem_summary.py "$OUT_H/gpumem_sampler.csv" "$OUT_H/spill_final.json" > "$OUT_H/spill_final.log" 2>&1
  bash $H/snapshot.sh "$OUT_H" host
  docker compose logs --no-color --tail 20000 "$SVC" > "$OUT_H/container.log" 2>&1
  { echo "== container"; docker inspect -f '{{.Name}} restarts={{.RestartCount}} status={{.State.Status}} oomkilled={{.State.OOMKilled}}' $(docker compose ps -aq "$SVC")
    echo "== error lines (judge each; Traceback/OOM inside serving = abort)"; grep -inE 'out of memory|OutOfMemory|CUDA error|Traceback|Killed|UVA is not available' "$OUT_H/container.log" | head -20
    echo "== effective-config and KV pool"; grep -E 'effective-config|GPU KV cache size|Maximum concurrency|Available KV cache' "$OUT_H/container.log" | head -20
    echo "== status.txt"; cat "$OUT_H/status.txt" 2>/dev/null
    echo "== warmup_probe"; grep -o '"verdict": "[A-Z-]*"' "$OUT_H/warmup_probe.json" 2>/dev/null
    echo "== spill (suite phase)"; cat "$OUT_H/spill.json" 2>/dev/null; echo; echo "== spill (whole run)"; cat "$OUT_H/spill_final.log"
    echo "== prefix-cache notices"; grep -l 'served from prefix cache' "$OUT_H"/*.log 2>/dev/null
    echo "== throttle/clock lines"; grep -hE 'Clocks Event|Throttle|SW Power|HW Slowdown|Thermal|Power Limit' "$OUT_H"/snapshot_*.txt | head -12
  } > "$OUT_H/checks.txt" 2>&1
  python3 "$HQ/extract.py" "$OUT_H" > "$OUT_H/summary_table.txt" 2>&1
  docker compose --profile single --profile batch down
  mkdir -p "$HQ/$LABEL-$TS" && cp -r "$OUT_H"/. "$HQ/$LABEL-$TS"/; unset TS; }
```

`$HQ/extract.py` (prints one line per sweep cell: prompt tokens, TTFT, decode, aggregate, tokens per step, acceptance, peak VRAM from nvidia-smi, valid reps):

```python
import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + "/sweep_*.json")):
    d = json.load(open(f))
    for c in d["cells"]:
        s = c.get("summary")
        g = lambda k: (f"{s[k]['mean']:.2f}+-{s[k]['sd']:.2f}(cv{s[k]['cv_pct']:.1f}%)" if s and s.get(k) else "n/a")
        v = c["vram"]
        print(d["label"], d["set"], "ctx", c["ctx"], "conc", c["conc"], "ptok", g("ptok"), "ttft_ms", g("ttft"), "decode_tok_s_stream", g("rate"),
              "agg_tok_s", g("agg"), "tok_step", g("tok_step"), "accept", g("accept_rate"),
              "vram_peak_mib", v["peak_mib"], "vram_idle_mib", v["idle_mib"], "reps_ok", f'{c["valid_reps"]}/{d["reps"]}')
```

## Step 0: environment checks (D13)

Run these first, once, before bench-1. Record every output under `$HQ/step0/` (create it) and summarize in `results/env-<ts>/RESULT.md` at bench-1. Facts, not decisions: do not ask the user about them.

**0.1 Repo and branch.** Use the git the box already uses for this repo (Windows git or WSL git; after checkout `git status` must be clean, a tree full of line-ending changes means the wrong git).

Delivery (D17): the orchestrator pushes branch `research/eagle4-expedition` to `origin` (our fork) once, at the end of the expedition, and you start after that push. You get the harness by fetching that branch; record the commit hash. If the fetch fails the branch is not there yet: STOP and report, do not improvise another way to get the files. Run the block below as one script (shell state does not persist between calls).

```bash
HQ=/mnt/g/dev/hq-results/eagle4-$(date +%Y%m%d); mkdir -p $HQ/step0       # same date string as lib.sh
cd /mnt/g/dev/HyperQwen && git status --porcelain --untracked-files=no     # must print nothing; if it does: STOP and ask (the user's work)
git remote -v                                                               # origin must be https://github.com/TyroneNel/qwen38-27b-rtx3090.git
git fetch origin research/eagle4-expedition || echo "BRANCH NOT ON ORIGIN: STOP and ask the user to push it"
git checkout research/eagle4-expedition 2>/dev/null || git checkout -b research/eagle4-expedition origin/research/eagle4-expedition
git merge --ff-only origin/research/eagle4-expedition && git log -1 --format='%H %s' | tee $HQ/step0/commit.txt     # the fetched commit, repeated in every RESULT.md
ls docs/research/04-benchmarks/harness/run_suite.sh                         # must exist
sha256sum docs/research/04-benchmarks/harness/*.sh docs/research/04-benchmarks/harness/*.py > $HQ/step0/harness.sha256
which python3 && python3 --version                                          # host python3 is needed by lib.sh (stdlib only)
```

**0.2 Glue and .env backup.** Create `$HQ`, `lib.sh`, `extract.py` as above. Back up the box `.env` once: `[ -f $HQ/env.box-orig ] || cp .env $HQ/env.box-orig` (never overwrite it). Restore it when the whole queue is done: `cp $HQ/env.box-orig .env`. `ls docker-compose*.yml > $HQ/step0/compose-files.txt` and `grep -n "bench\|/app" docker-compose.override*.yml >> $HQ/step0/compose-files.txt` (shows which override bind-mounts what).

**0.3 GPU, power, driver, sampler LUID.**

```bash
source $HQ/lib.sh   # for $GPU only; this also arms the watchdog, fine for step 0
nvidia-smi -i $GPU --query-gpu=index,name,uuid,driver_version,power.limit,clocks.max.sm,temperature.gpu,memory.used,memory.total --format=csv | tee $HQ/step0/gpu.txt
nvidia-smi -L | tee -a $HQ/step0/gpu.txt
powershell.exe -NoProfile -Command "Get-Counter -ListSet 'GPU Process Memory' | Select-Object -ExpandProperty PathsWithInstances | Select-String '$LUID' | Select-Object -First 3" | tee $HQ/step0/luid.txt
ls -l /mnt/g/dev/hq-bench/gpumem_sample.ps1 | tee -a $HQ/step0/luid.txt
```
Expect an RTX 3090, driver 610.88, 250 W. If power.limit is not 250: stop (rule 14). If the LUID substring `1144f` does not appear in `luid.txt` (the LUID can change after a reboot), find the 3090's current LUID (`Get-Counter -ListSet 'GPU Adapter Memory'` instances list `luid_0x........_0x........`; the 3090 is the one whose `Dedicated Usage` is large while a container runs), put it in `lib.sh` as `LUID=`, and record the change. If the counter list is empty at idle, defer this check to bench-2.

**0.4 Disk.** `disklog` (after `source lib.sh`), `du -sh /mnt/g/dev/hq-results /mnt/g/models 2>/dev/null | tee -a $HQ/step0/disk.txt`.

**0.5 Are the launchers baked into the image or bind-mounted? Which bench files does the stack really run?** Compare three copies of the same file list: the image, the stack as composed (which would show a bind mount), and the checkout.

```bash
docker image inspect $IMG --format '{{.Id}} {{json .RepoDigests}} {{.Created}}' | tee $HQ/step0/image.txt     # missing image: STOP, do not pull
FILES='launcher_common.sh resolve_config.sh verify.sh single-user/start_qwen.sh single-user/select_model.sh batch/start_qwen.sh patches/series bench/harness.py bench/conc_ladder.py bench/labd_bench.py bench/warmup.sh bench/quality_battery.py bench/run_benchmarks.sh'
CHK='cd /app && for f in '"$FILES"'; do printf "%s %s\n" "$(sha256sum $f 2>&1 | cut -c1-16)" $f; done'
docker run --rm --entrypoint bash $IMG -c "$CHK" > $HQ/step0/files-image.txt
docker compose run --rm --no-deps -T --entrypoint bash single -c "$CHK" > $HQ/step0/files-composed.txt
( for f in $FILES; do printf "%s %s\n" "$(sha256sum $f 2>&1 | cut -c1-16)" $f; done ) > $HQ/step0/files-checkout.txt
diff $HQ/step0/files-image.txt $HQ/step0/files-composed.txt && echo "BAKED: nothing in the list is bind-mounted over the image" || echo "BIND-MOUNTED: see diff"
diff $HQ/step0/files-image.txt $HQ/step0/files-checkout.txt   # the image is main 2cf50a0; a checkout of this branch (63f7581 + docs/research, D22) differs from it in docs only; an older checkout at 95ac251 lacks b4b97ec, the PTH_FA default for mtp/long
docker run --rm --entrypoint bash $IMG -c 'grep -nE "mtp/long|dflash2/long" /app/resolve_config.sh; cd /app/bench && grep -nE "^def (samples|stream|post|spec|model)" harness.py; grep -cE -- "--ctx-tokens|--salt|--json" conc_ladder.py; echo perf_counter=$(grep -c perf_counter labd_bench.py) time_time=$(grep -c "time\.time" labd_bench.py); ls warmup.sh quality_battery.py run_benchmarks.sh' | tee $HQ/step0/image-bench-symbols.txt
```
Rules from the result: use the image's own `/app/bench` (do **not** chain `docker-compose.override.local.yml`, which bind-mounts a checkout's `bench/` over `/app/bench`; the old `95ac251` `labd_bench.py` uses `time.time()`; `373a695`, included from `63f7581` on, switches it to `perf_counter`). The harness needs `harness.py` to define `samples`, `stream`, `post`, `spec`, `model` and `conc_ladder.py` to know `--ctx-tokens`, `--salt`, `--json`. If the image's `bench/` lacks them, write `$HQ/bench-mount.yml` with a `single` and `batch` volume `- /mnt/g/dev/HyperQwen/bench:/app/bench:ro`, add it to `COMPOSE_FILE` in `lib.sh`, record it, and skip the `labd_bench.py` replay in bench-11 if that checkout's `labd_bench.py` still uses `time.time()`.

**0.6 pthfa installed in the image?** The launcher's own probe (`resolve_config.sh:42-46`):

```bash
docker run --rm --entrypoint /app/venv/bin/python $IMG -c 'import importlib.util as u, os, sys
s = u.find_spec("pth_flash_attn"); print("spec:", s and s.submodule_search_locations)
print("op library:", bool(s and os.path.isfile(os.path.join(s.submodule_search_locations[0], "_C.abi3.so"))))' | tee $HQ/step0/pthfa.txt
```
If the op library is missing, every `PTH_FA=1` arm (bench-5, 9, 11 Dp) is `blocked`: record and skip those.

**0.7 Which LABD corpus is frozen?** The corpora live in the `/cache` volume (`HOME=/cache`). Inspect them through the stack's own mounts:

```bash
docker compose run --rm --no-deps -T --entrypoint bash single -c '
ls -l --time-style=full-iso /cache/bench/ 2>&1
for f in /cache/bench/labd_corpus*.txt; do echo "$f chars=$(wc -c < $f) sha256=$(sha256sum $f | cut -c1-16)"; done
cmp -n $(stat -c %s /cache/bench/labd_corpus.txt) /cache/bench/labd_corpus.txt /cache/bench/labd_corpus_long.txt && echo "long corpus HEAD == frozen corpus (byte-identical)"' | tee $HQ/step0/corpus.txt
ls -l /mnt/g/dev/hq-bench | tee -a $HQ/step0/corpus.txt
```
The frozen corpus is `/cache/bench/labd_corpus.txt` (built on first run by `labd_bench.py`; its head is byte-identical in `labd_corpus_long.txt`, which adds vLLM source and is what every long-context run uses, `lib.sh` `CORPUS=`). Record both sha256 values: they go into every `RESULT.md` that uses long cells. The long corpus must hold at least 432,000 characters for the 120000 cell and be about 3.2 chars per token (so a nominal `--ctx N` gives about 1.13 N prompt tokens; `ptok` in the table is the truth). If `labd_corpus_long.txt` is missing, do **not** build it (it would not be the frozen text): skip the long cells (`LC_CELLS=` empty, and drop the guarded 120000/85000 cells) and mark those jobs `partial: no long corpus`.

**0.8 Does `G:\dev\hq-results\deepinfer-20261020` still hold the raw jsonl/csv?**

```bash
P=/mnt/g/dev/hq-results/deepinfer-20261020
ls "$P" 2>&1 | tee $HQ/step0/prior-raw.txt
find "$P" \( -name '*.jsonl' -o -name '*.csv' -o -name 'summary.txt' -o -name 'breakdown.json' \) -printf '%s %p\n' 2>/dev/null | sort -k2 | tee -a $HQ/step0/prior-raw.txt
du -sh "$P" 2>/dev/null | tee -a $HQ/step0/prior-raw.txt
```
Record present/absent and the file count and sizes of `baseline/`, `evals-4/`, `evals-10/`, `e-profile/` (the ledger row `prior-raw`). If the total of the jsonl/csv/summary/breakdown files is 25 MB or less, copy only those (read-only on the source, keep relative paths) to `docs/research/04-benchmarks/results/prior-raw-<date>/` and commit them; if larger, list only.

**0.9 prior-timing-check (bench-3)** needs no GPU and no container; its commands are in the bench-3 section. You may run them now, but log the result at bench-3's turn.

Step 0 is done when: branch checked out clean at the fetched commit, power 250 W confirmed, `$HQ/step0/*` written, glue created and syntax-checked, disk logged. Write a one-screen `$HQ/step0/SUMMARY.md` answering 0.5 to 0.8 in one line each (these answers go to the orchestrator in your return).

## Per-job procedure (applies to every serving job)

```
source $HQ/lib.sh; disklog; gate || { echo "gate failed"; exit 1; }
arm_env <arm lines>      ; boot <single|batch> || exit 1
begin <label>            ; suite <env overrides> ; <extra commands> ; end
```
Then, outside the script, read `$OUT_H/checks.txt` and `summary_table.txt`, apply the job's pass/abort criteria, write `$OUT_H/RESULT.md`, update the ledger and queue rows, commit, and move on. `RESULT.md` carries: job, date, branch commit (`git rev-parse HEAD`, equal to `step0/commit.txt`), image Id from Step 0.5, the `.env` lines that differ from the box's, the `[effective-config]` lines from `checks.txt`, the pinned `PTH_FA`, power limit, GPU temperature at start, peak and median SM clock if the snapshot has them, driver, corpus sha256, the exact commands, the numbers table (from `summary_table.txt`, unedited), the WDDM growth (first, max, last, growth MiB from `spill_final.json`), the floor line (`floor.txt`) where the job has one, the verdict (`INFO` for measurements; `INVALID` only where a job says so), the raw-data path under `$HQ`, and the anomalies. The arm's `.env` is also copied to `$OUT_H/arm.env` with key lines redacted (`sed -E 's/^([A-Za-z_]*(KEY|TOKEN|SECRET)[A-Za-z_]*)=.*/\1=<redacted>/' .env`). Note: `.env` values must be bare lines with no inline `# comments`.

TTFT and peak VRAM, which the prior runs never recorded: TTFT is `ttft_ms` in `summary_table.txt` (mean over 3 timed rounds, cold prefix cache by salt). Peak VRAM is `vram_peak_mib` per sweep cell and `ladder_*_vram.json` `peak_mib` (nvidia-smi `memory.used`, whole card, secondary: near-flat because vLLM preallocates the KV pool); the `snapshot_pre`/`post` `memory.used` is the steady-state boot figure; the WDDM non-local reading (`spill.json`, `spill_final.json`) is the spill figure and is a different quantity. Do not call `vram_peak_mib` "spill-free": only the sampler says that.

Durations below are **estimated** (boot 5-15 min on a warm compile cache is included); they are for planning only, the 2 h cap is the rule.

## Jobs

Labels become result directories `results/<label>-<YYYYMMDD-HHMMSS>/`. Common to all: `SALT=eagle4-base` (set by `suite`) unless stated; `PREFIX_CACHE=1` in every arm (the launcher default, pinned so the arm does not depend on it); `VISION=0` (the box `.env` has `VISION=1`, which silently turns the `SPEC=mtp CTX=long` arm into fp8 + FlashInfer); `MODELS_DIR=/mnt/g/models` and `VLLM_WSL2_ENABLE_PIN_MEMORY=1` are added by `arm_env`.

### Baseline rounds (D16) and decode floors (D15)

Jobs bench-4, 5, 6, 11a and 11b are run as **two boots per mode**, round 1 then round 2, each its own job script run (`bash $HQ/<job>.sh 1`, then `bash $HQ/<job>.sh 2`), each with the 2 h cap, the 55 C gate and its own result directory `base-<mode>-r<N>-<ts>`. Same `SALT` in both rounds (a fresh boot has an empty prefix cache, so the prompts are the same and TTFT stays cold). The prior expedition rotated rounds across modes; here the rounds of one mode run back to back and the `gate` (55 C or cooler) separates them; report each round's numbers separately and add the boot-to-boot difference, never an average alone. The arms that are interleaved are the pthfa-ab arms (bench-9), as before.

Decode floors (abort at more than 10% below the prior measured value for the mode, rule 6). Prior values are from `baseline.md`; the floor is 0.9 times the prior, rounded. The prior value was taken on the prior's own prompts, so the comparison cell is the closest suite cell and the comparability is estimated, not exact. Jobs with no prior value for the mode (bench-7, 8, 9, 10, 12) have no floor.

| Mode / job | Floor check | Prior (measured, prior box) | Floor tok/s |
|---|---|---|---|
| B (bench-4) | `sweepfloor` on `sweep_base-B-r<N>_short_chat.json`, N=1 per-stream decode | 114.4 (C1 greedy cohort) | 103 |
| D (bench-5) | `sweepfloor` on `sweep_base-D-r<N>_short_chat.json`, N=1 | 99.8 | 90 |
| E (bench-6) | `labdfloor` on `labd_E.log`, row TOTAL (`labd_bench.py E --ctx 20000`) | 57.8 | 52 |
| D' (bench-11a) | `sweepfloor` on `sweep_base-Dp-r<N>-53000_long_context.json` (about 60k tokens) | 55.7 | 50 |
| C (bench-11b) | `labdfloor` on `labd_C.log`, row `copy` | 357.6 | 322 |
| A (bench-11b) | `e2e=` of the `ROW 64conc 128in/512out` line of the second `run_benchmarks.sh` run, `floorv` | 814.3 (e2e, not C1 decode; mode A has no C1 figure) | 733 |

If the `labd_bench.py` replay cannot run (Step 0.5 showed `time.time()` in the image copy, or the bench was bind-mounted), the E and C floors cannot be checked: say so in `RESULT.md` and the anomalies list; the other abort conditions still apply.

### bench-1: env snapshot (ledger `env`) | ~10 min | closes baseline.md gap 10

Arm: mode B, no serving boot needed (`docker compose run` starts the container with the harness mount and only runs the snapshot; the `[effective-config]` and KV-pool lines are captured by every later boot).

```bash
source $HQ/lib.sh; disklog; gate | tee -a $HQ/step0/gate.txt
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
TS=$(date +%Y%m%d-%H%M%S); OUT_H=$R/env-$TS; mkdir -p $OUT_H
docker compose run --rm --no-deps -T single bash /harness/snapshot.sh /results/env-$TS container
bash $H/snapshot.sh $OUT_H host
docker compose config | grep -viE 'KEY|TOKEN|SECRET' > $OUT_H/compose-config.txt
cp $HQ/step0/image.txt $OUT_H/; nvidia-smi -q > $OUT_H/nvidia-smi-q-host.txt
cmd.exe /c 'type %USERPROFILE%\.wslconfig' > $OUT_H/wslconfig.txt 2>&1; wsl.exe -l -v 2>&1 | tr -d '\0' > $OUT_H/wsl-list.txt
docker version > $OUT_H/docker-version.txt 2>&1; uname -a > $OUT_H/uname.txt
clockdrift | tee $OUT_H/clockdrift.txt
cp -r $HQ/step0 $OUT_H/step0; docker compose --profile single --profile batch down; cp -r $OUT_H $HQ/env-$TS
```
Outputs: `snapshot_container_<id>.txt` (nvidia-smi, env knobs with `PTH_FA=0` shown, vLLM and torch versions, `patches/series` hash and count), `snapshot_host_<host>.txt` (docker ps, `.env` redacted, git log), `compose-config.txt`, `image.txt`, `clockdrift.txt`, `step0/`. Pass: both snapshots written, container snapshot shows the 3090 UUID, power limit 250, `PTH_FA=0`; `patches/series` count recorded. If the container-side snapshot fails (`compose run` refused), boot mode B instead (`boot single`, then `docker compose exec -T single bash /harness/snapshot.sh /results/env-$TS container`). Abort: power limit not 250. Verdict INFO.

### bench-2: harness smoke run on mode B (ledger `smoke`) | ~30 min | gate for every `base-*`, greedy, quality, profile job

This confirms real `/metrics` names, nvidia-smi parsing, the sampler csv unit, salt, `warmup_probe`, logprobs under the V2 runner and the image's `conc_ladder.py` flags. It uses the README smoke command with `SMOKE=1` (the only mode in which `sweep.py` accepts `REPS=1`) and `LC_CELLS=` empty (skips the long-context cells).

```bash
source $HQ/lib.sh; disklog; gate || exit 1
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
boot single || exit 1
begin smoke
suite SALT=smoke SMOKE=1 PHASES="snap warm probe sweep ladder greedy spill" REPS=1 CONC=1 MAXTOK=64 PROBE_TOKENS=64 \
      GREEDY_TOKENS=64 LC_CELLS= LADDER_CELLS="200:1,2" LADDER_OUT=64
docker compose exec -T "$SVC" bash -c "PYTHONPATH=/app/bench:/harness /app/venv/bin/python /harness/greedy_equal.py record smoke2 --max-tokens 64 --out $OUT_C"
python3 $H/greedy_equal.py compare $OUT_H/greedy_smoke.json $OUT_H/greedy_smoke2.json --near-tie 0.01 | tee $OUT_H/greedy_compare_smoke.txt; echo "exit=${PIPESTATUS[0]}" >> $OUT_H/greedy_compare_smoke.txt
end
head -3 $OUT_H/gpumem_sampler.csv; tail -2 $OUT_H/gpumem_sampler.csv    # sampler header and unit, paste into RESULT.md
```
Pass criteria (all must hold; otherwise the job is `blocked`, the failing item is quoted with its log, and you stop and return, because `base-*`, greedy, quality and profile are gated on it; do not edit scripts):
- `suite` exit 0 and every line of `status.txt` is `rc=0` (the `spill` line may be `rc=1` only if the growth is above 64 MiB; record the growth).
- `warmup_probe.json` exists (WARM or NOT-WARM both pass for smoke).
- Both sweep JSONs have `summary` with a non-null `tok_step` (the spec counters moved: the `/metrics` names exist in this image) and non-null `vram.peak_mib` (nvidia-smi parsed inside the container).
- `ladder_200.json` and `ladder_200_vram.json` exist.
- `gpumem_sampler.csv` is non-empty and `spill.json` has `samples` > 0. In `RESULT.md` state what the sampler's last numeric field is (MiB, KiB, bytes or a timestamp) by comparing one csv line with the sampler's own header. If it is not MiB, do not edit `gpumem_summary.py`: write `spill-adhoc.json` with the growth converted to MiB, and use that figure for the 64 MiB bound from now on (Harness gaps 3).
- The greedy compare exits 0, or exits 1 with every divergence `near-tie` or in the replay band (a `REAL` line whose larger margin is at most 0.25). Exit 2, `invalid: no logprobs`, or a `REAL` divergence with a margin above 0.25 on the same boot blocks bench-7: record it. [review: D18 bands, `00-meta/decisions.md:22`: near-tie below 0.01 nats, replay 0.01-0.25, real above 0.25. `greedy_equal.py` has only two bands and its `--near-tie` default is 0.1 (`harness/greedy_equal.py:84`), so the flag is passed and the replay band is read from the printed margins.]
Outputs: everything from the default suite (list under bench-4) minus long cells, plus `greedy_smoke2.json`, `greedy_compare_smoke.txt`. Verdict INFO. Numbers from this run are plumbing only (1 rep, no spread, 64 tokens): never use them as baseline.

### bench-3: prior-timing-check (ledger `prior-timing-check`) | ~2 min, read-only, no GPU | closes gap 11

Did the prior box's `labd_bench.py` use `perf_counter`? Settles baseline.md "Timing validity". Read-only, record the raw output in `$OUT_H/prior-timing.txt`:

```bash
source $HQ/lib.sh; TS=$(date +%Y%m%d-%H%M%S); OUT_H=$R/prior-timing-$TS; mkdir -p $OUT_H
{ echo "== box checkout (the prior bind-mounted its bench/ over /app/bench)"
  git -C /mnt/g/dev/HyperQwen log -1 --format='%H %ad %s' -- bench/labd_bench.py
  grep -c perf_counter /mnt/g/dev/HyperQwen/bench/labd_bench.py; grep -nE "time\.time|perf_counter" /mnt/g/dev/HyperQwen/bench/labd_bench.py
  echo "== the prior working tree at the time: local branches and the bind-mount override"
  git -C /mnt/g/dev/HyperQwen branch -a --contains 373a695 2>&1 | head; grep -n bench /mnt/g/dev/HyperQwen/docker-compose.override*.yml
  echo "== stock image copy"; docker run --rm --entrypoint bash $IMG -c 'cd /app/bench && grep -c perf_counter labd_bench.py; grep -nE "time\.time|perf_counter" labd_bench.py'
  echo "== prior tools dir"; grep -rn "perf_counter" /mnt/g/dev/hq-results/deepinfer-20261020/tools 2>/dev/null | head -5
} > $OUT_H/prior-timing.txt 2>&1; cat $OUT_H/prior-timing.txt
```
Ledger row `prior-timing-check`: key number = `perf_counter` count in the box copy and the image copy; verdict INFO. `373a695` may not exist in a repo without the upstream remote: a "not a valid commit" line is a result, not a failure. Result path in the queue: `04-benchmarks/LEDGER.md` (row `prior-timing-check`), raw in `results/prior-timing-<ts>/`.

### bench-4: baseline re-run mode B (ledger `base-B`) | ~60-75 min per boot, two boots (D16) | closes gaps 1, 2, 3, 4, 9, 10

Arm: `SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0` (`PTH_FA=1` is refused at `CTX=fast`). KV pool should read 68,605 tokens (prior). Run `bash $HQ/base-B.sh 1`, then `bash $HQ/base-B.sh 2`.

`$HQ/base-B.sh`:
```bash
source $HQ/lib.sh; RND=$1; disklog; gate || exit 1
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
boot single || exit 1
begin base-B-r$RND
suite                                   # default PHASES "snap warm probe sweep ladder greedy spill"; default LC_CELLS "8000:1,2,4 32000:1"; default LADDER_CELLS "200:1,2,4,8 8000:1,2,4" (sized for B's pool)
sweepfloor $OUT_H/sweep_base-B-r${RND}_short_chat.json 103 | tee $OUT_H/floor.txt
end
clockdrift > $OUT_H/clockdrift_post.txt
```
Outputs in `results/base-B-r<N>-<ts>/`: `snapshot_pre_*`, `snapshot_post_*`, `snapshot_host_*`, `warmup_probe.json`, `sweep_base-B-r<N>_short_chat.json`, `sweep_base-B-r<N>_code.json` (N=1,2,4,8), `sweep_base-B-r<N>-8000_long_context.json` (N=1,2,4), `sweep_base-B-r<N>-32000_long_context.json` (N=1), `ladder_{200,8000}.json` and `ladder_{200,8000}_vram.json`, `greedy_base-B-r<N>.json`, `gpumem.csv`, `gpumem_sampler.csv`, `spill.json`, `spill_final.json`, `floor.txt`, `suite.log`, `status.txt`, per-phase logs, `container.log`, `checks.txt`, `summary_table.txt`, `clockdrift_post.txt`, `RESULT.md`. Every sweep cell gives TTFT, decode tok/s per stream, aggregate tok/s, tokens per step, acceptance rate (mean, sd, CV over 3 timed rounds) and nvidia-smi peak VRAM.
Pass: `checks.txt` shows restarts=0, no OOM line, probe WARM, WDDM growth recorded (above 64 MiB flagged, rule 6), `status.txt` all `rc=0` except a flagged `spill` line, effective config `SPEC=dflash2 CTX=fast`, `PTH_FA=0` in the snapshot env block, 250 W, floor line exit 0. Abort: rule 6. Sanity band (estimated from baseline.md, anomaly if outside, not a gate): short_chat N=1 decode 100-130 tok/s per stream, tokens per step 3.0-3.9.

### bench-5: baseline re-run mode D (ledger `base-D`) | ~80-100 min per boot, two boots (D16) | closes gaps 1, 2, 3, 4, 9, 10

Arm: `SPEC=mtp CTX=long PTH_FA=1 VISION=0` (the prior default arm; no `DRAFT_TOKENS`, `DYNAMIC_K`, `FA2_FP8KV`). `checks.txt` must contain `[effective-config] PTH_FA=1 (explicit; ...)`; if it does not, abort (the arm silently ran another backend). Run `bash $HQ/base-D.sh 1`, then `bash $HQ/base-D.sh 2`.

`$HQ/base-D.sh`:
```bash
source $HQ/lib.sh; RND=$1; disklog; gate || exit 1
arm_env SPEC=mtp CTX=long PREFIX_CACHE=1 PTH_FA=1 VISION=0
boot single || exit 1
begin base-D-r$RND
suite LC_CELLS="8000:1,2,4,8 32000:1,2,4 64000:1" LADDER_CELLS="200:1,2,4,8 8000:1,2,4,8 32000:1,2"
sweepfloor $OUT_H/sweep_base-D-r${RND}_short_chat.json 90 | tee $OUT_H/floor.txt
end
clockdrift > $OUT_H/clockdrift_post.txt
```
Outputs: as bench-4, with long cells 8000, 32000, 64000 (nominal; read `ptok`, about 1.13x) and ladders 200, 8000, 32000. KV pool prior: 206,532-207,286 tokens. Pass and abort as bench-4 with `SPEC=mtp CTX=long`. Sanity band: short_chat N=1 decode 90-115 tok/s, tokens per step 2.4-3.1.

### bench-6: baseline re-run mode E (ledger `base-E`) | ~100-120 min per boot, two boots (D16) | closes gaps 1, 2, 3, 4, 9, 10

Arm: `SPEC=dflash2 CTX=huge PREFIX_CACHE=1 PTH_FA=0 VISION=0`. `PTH_FA=1` is refused at `CTX=huge`, so the pin is `0` (record it in `RESULT.md`). KV pool prior 268,169 tokens. The prior E boot's non-local counter went 92 -> 162 MiB at 170k with no throughput effect (`raw/prior-3090/e-profile.md:8`): +70 MiB, above the 64 MiB flag, below the 256 MiB abort bound. If this boot does the same, flag it in the anomalies list and go on (D15). Run `bash $HQ/base-E.sh 1`, then `bash $HQ/base-E.sh 2`.

`$HQ/base-E.sh`:
```bash
source $HQ/lib.sh; RND=$1; disklog; gate || exit 1
arm_env SPEC=dflash2 CTX=huge PREFIX_CACHE=1 PTH_FA=0 VISION=0
boot single || exit 1
begin base-E-r$RND
suite LC_CELLS="8000:1,2 32000:1 64000:1" LADDER_CELLS="200:1,2,4,8 8000:1,2,4,8 32000:1,2"
# the prior's E row: labd 20k TOTAL decode (the floor reference). Skip it if Step 0.5 showed time.time() in the image's labd_bench.py.
docker compose exec -T -w /app "$SVC" venv/bin/python bench/labd_bench.py E-r$RND --ctx 20000 > $OUT_H/labd_E.log 2>&1
labdfloor $OUT_H/labd_E.log TOTAL 52 | tee $OUT_H/floor.txt
# the 120k cell last, guarded: it has to fit inside the 2 h cap (estimated 35-40 min: 4 prompts x 4 rounds of ~135k-token prefill)
if [ $(elapsed) -lt 70 ]; then
  pyx /harness/sweep.py base-E-r$RND-120000 --set long_context --ctx 120000 --conc 1 --reps 3 --warmup 1 --max-tokens 128 --salt eagle4-base --corpus $CORPUS --out $OUT_C > $OUT_H/sweep_120000.log 2>&1
else echo "SKIPPED 120000 cell: $(elapsed) min elapsed" | tee $OUT_H/sweep_120000.log; fi
end
clockdrift > $OUT_H/clockdrift_post.txt
```
Outputs: as bench-4, long cells 8000 (N=1,2), 32000, 64000, 120000 (N=1), ladders 200, 8000, 32000, and `labd_E.log`. Pass and abort as bench-4. Sanity band (estimated): short_chat N=1 decode 45-70 tok/s.

### bench-7: greedy T=0 equality, SPEC=off vs dflash2 vs mtp (ledger `greedy-eq`) | ~45 min (3 boots) | closes gap 5

Compare at the **same** `CTX` with `PTH_FA=0` on all three (a `PTH_FA=1` arm has no non-spec twin; bf16 vs int8 KV would diverge for KV reasons, not speculation). `CTX=fast`, `PREFIX_CACHE=1`, `GREEDY_TOKENS=256`, concurrency 1 (the tool records at concurrency 1), thinking off (tool default).

```bash
source $HQ/lib.sh; disklog
for v in "NS SPEC=off" "DF2 SPEC=dflash2" "MTP SPEC=mtp"; do set -- $v
  gate || exit 1
  arm_env $2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
  boot single || exit 1
  begin greedy-$1; suite PHASES="snap greedy" GREEDY_TOKENS=256; end
  eval "G_$1=$OUT_H/greedy_greedy-$1.json"
done
OUTC=$R/greedy-compare-$(date +%Y%m%d-%H%M%S); mkdir -p $OUTC
for pair in "NS DF2" "NS MTP" "DF2 MTP"; do set -- $pair; a=G_$1; b=G_$2
  python3 $H/greedy_equal.py compare ${!a} ${!b} --near-tie 0.01 | tee $OUTC/compare_$1_$2.txt; echo "exit=${PIPESTATUS[0]}" >> $OUTC/compare_$1_$2.txt; done
```
Outputs: three `results/greedy-{NS,DF2,MTP}-<ts>/` (with `greedy_greedy-<X>.json`, `checks.txt`, snapshots, `spill_final.json`) and `results/greedy-compare-<ts>/compare_*.txt`. Pass: all three `status.txt` `rc=0`, `checks.txt` clean, effective config matches each SPEC. Verdict per pair from the compare exit code: 0 all identical -> INFO; 1 with only `near-tie` divergences -> INFO, list them; 1 where every `REAL` line has a larger margin of at most 0.25 (the D18 replay band) -> INFO, list them as `replay band, judged in bench-14`; 1 with any `REAL` margin above 0.25 -> verdict `INVALID` for that pair, keep the compare text and both JSONs, do not retry; 2 invalid (different prompt sets or `GREEDY_TOKENS`, failed request, no token ids) -> `blocked`, cause quoted. Compare runs on the host with stdlib python3.

### bench-8: quality guard on stock B (ledger `quality`) | ~30-40 min | closes gap 8

Arm: stock mode B, no `GPU_UTIL` override (the single-user launcher default is 0.93, which the perplexity pass needs for `prompt_logprobs` headroom: `start_qwen.sh:175`). Reference: ppl all 8.401 (en 10.77, da 10.91, code 3.60), GSM8K 0.955 (`harness/quality_ref_prior.json`).

```bash
source $HQ/lib.sh; disklog; gate || exit 1
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
boot single || exit 1
begin quality-B
suite PHASES="snap quality"
end
```
Outputs: `result_quality-B.json` (copied from `/app/models/quality-data/` = `G:\models\quality-data\`), `quality.log`, `quality_guard.log`, `status.txt`. Pass criteria are the guard's: ppl (all and per corpus) rises at most 1%, GSM8K falls at most 3 points; exit 0 PASS. Verdict: PASS -> INFO with the five numbers; FAIL -> `INVALID` (stock B should reproduce the prior numbers; a FAIL means the environment is not the prior one; record, do not retry with other settings); exit 2 (field missing) -> `blocked`. Abort on an OOM in the logprob pass (record it; no config change).

### bench-9: PTH_FA=0 vs 1 interleaved, mtp and dflash2 at CTX=long (ledger `pthfa-ab`) | 4 sub-runs of ~1 h | closes gap 7

Gate: `base-D` (bench-5) and `base-Dp` (bench-11a) run before this row in the queue, so the baselines for both `PTH_FA=1` arms exist; show them next to the p1 rows in `RESULT.md` (measured, same box, earlier boots). Four sub-runs, back to back with nothing else in between, so the order is mtp 0 1 0 1 and then dflash2 0 1 0 1: `bash pthfa-ab.sh mtp 1`, `bash pthfa-ab.sh mtp 2`, `bash pthfa-ab.sh dflash2 1`, `bash pthfa-ab.sh dflash2 2`. Same `SALT=pthfa-ab`, same corpus, `CONC=1` (decode and TTFT at about 8k and about 66k prompt tokens; nominal `--ctx 8000` is about 9k tokens, nominal 58000 is about 65k tokens; `ptok` is the truth). Each boot starts at 55 C or cooler (`gate` waits up to 20 min).

`$HQ/pthfa-ab.sh`:
```bash
source $HQ/lib.sh; SPEC=$1; ROUND=$2; disklog
for p in 0 1; do
  gate || exit 1
  arm_env SPEC=$SPEC CTX=long PREFIX_CACHE=1 PTH_FA=$p VISION=0       # nothing else: no DFLASH_TOKENS/LOOKUP/DYNAMIC_K/FA2_FP8KV (PTH_FA=1 refuses them)
  boot single || exit 1
  begin pthfa-ab-$SPEC-p$p-r$ROUND
  suite SALT=pthfa-ab PHASES="snap warm probe sweep spill" CONC=1 LC_CELLS="8000:1 58000:1"
  end
done
```
Arm facts to copy from `checks.txt` into `RESULT.md`: `PTH_FA=1` boots print `[effective-config] PTH_FA=1 (explicit; ...)`; the `PTH_FA=0` dflash2 boot runs the arm's own default backend (fp8 KV on the FA2 plugin, `FA2_FP8KV=1` default on 8.6; read its `[effective-config] FA2_FP8KV=` line); the `PTH_FA=0` mtp boot runs fp8 + FlashInfer. Do not set `FA2_FP8KV`. Outputs per boot: `sweep_<label>_short_chat.json`, `sweep_<label>_code.json` (N=1), `sweep_<label>-8000_long_context.json`, `sweep_<label>-58000_long_context.json`, plus the usual files. Estimated 25-35 min per boot (the 58000 cell is 16 prefills of ~65k tokens). Pass per boot: as bench-4 (probe WARM, WDDM growth recorded, no OOM), and `ptok` identical between the p0 and p1 boots of the same SPEC at each cell (same prompts). A boot that fails a pass criterion is reported and re-run once at the end of its sub-run; do not drop it silently. Report the four rows per SPEC as measured, mean and boot-to-boot spread; no verdict (INFO).

### bench-10: profile boot of B for draft vs verify time (ledger `profile-dv`) | ~25-40 min | closes gap 6

Profiling slows the server: the timing numbers of this boot are never comparable with the baselines. The harness profile phase runs `sweep.py` on short_chat (3 reps, 8 prompts per round = 24 requests) between `/start_profile` and `/stop_profile`, so the trace grows with `MAXTOK`; use `MAXTOK=64` (about 18 decode steps per request) to keep the trace to a few hundred steps (Harness gaps 4). The trace lands under `$OUT_H/prof/` and is **not committed**.

```bash
source $HQ/lib.sh; disklog; gate || exit 1
TS=$(date +%Y%m%d-%H%M%S)
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0 \
  'EXTRA_ARGS=--profiler-config {"profiler":"torch","torch_profiler_dir":"/results/profile-dv-'$TS'/prof"}'
boot single || exit 1
begin profile-dv                        # reuses TS so OUT_C = /results/profile-dv-$TS
suite SALT=profile-dv PHASES="snap warm profile spill" MAXTOK=64 PROF_DIR=$OUT_C/prof
du -sh $OUT_H/prof | tee $OUT_H/prof_size.txt
end
```
Check the engine args line in `container.log` shows the profiler config (`checks.txt` does not grep it: `grep -n profiler-config $OUT_H/container.log | head -3`). Outputs: `breakdown.json` (ms per step by category with a Drafter row and the drafter's kernel breakdown), `profile_load` and `profile_breakdown` logs, `profile_start.log`, `prof_size.txt`, `prof/` (kept in `$HQ` only). Pass: `status.txt` `profile_load rc=0` and `profile_breakdown rc=0`, `breakdown.json` exists with a non-empty Drafter row and a step count. If `stop_profile` takes minutes, wait (the 2 h cap covers it). Abort: rule 6 (an OOM from the profiler is an abort; record it, do not shrink anything and retry). Reference for the ceiling check (estimated, not a gate): E at 170k drafter 3.6 ms of 151 ms (`e-profile.md:41-48`). Verdict INFO. Never commit `prof/` or any file over 1 MB.

### bench-11a: baseline re-run mode D' (ledger `base-Dp`) | ~90 min per boot, two boots (D16) | closes gaps 1, 2, 3, 4, 9, 10 | queued ahead of bench-9, after bench-8

`SPEC=dflash2 CTX=long PTH_FA=1 VISION=0`, no `DFLASH_TOKENS`, `LOOKUP`, `DYNAMIC_K`, `FA2_FP8KV` (they make `PTH_FA=1` refuse to boot). Cells 53000 and 85000 nominal are the prior d60/d90 cells (about 60k and 96k prompt tokens); the 85000 cell runs last and guarded. KV pool prior 136,429 tokens. Run `bash $HQ/base-Dp.sh 1`, then `bash $HQ/base-Dp.sh 2`.

`$HQ/base-Dp.sh`:
```bash
source $HQ/lib.sh; RND=$1; disklog; gate || exit 1
arm_env SPEC=dflash2 CTX=long PREFIX_CACHE=1 PTH_FA=1 VISION=0
boot single || exit 1
begin base-Dp-r$RND
suite LC_CELLS="8000:1,2 32000:1 53000:1" LADDER_CELLS="200:1,2,4,8 8000:1,2,4,8 32000:1,2"
sweepfloor $OUT_H/sweep_base-Dp-r${RND}-53000_long_context.json 50 | tee $OUT_H/floor.txt
if [ $(elapsed) -lt 75 ]; then
  pyx /harness/sweep.py base-Dp-r$RND-85000 --set long_context --ctx 85000 --conc 1 --reps 3 --warmup 1 --max-tokens 128 --salt eagle4-base --corpus $CORPUS --out $OUT_C > $OUT_H/sweep_85000.log 2>&1
else echo "SKIPPED 85000 cell: $(elapsed) min elapsed" | tee $OUT_H/sweep_85000.log; fi
end
```
Pass and abort as bench-5 (`checks.txt` must show `[effective-config] PTH_FA=1`). Sanity band (estimated, from baseline.md D' TOTAL decode 55.7 at ~60k, 48.8 at ~95k): long-context N=1 decode within 30% of those.

### bench-11b: baseline re-run modes C and A (ledger `base-C`, `base-A`) | two boots each, ~70 min per boot (D16) | closes gaps 1, 2, 3, 4, 9, 10 | lower priority, after bench-10

Two jobs, in this order: base-C (rounds 1 and 2), then base-A (rounds 1 and 2).

**base-C** (mode C): mode B with `DFLASH_TOKENS=15`, `PTH_FA=0`. After the suite, the repo's own copy-task replay for the `copy` row (baseline.md C: 357.6 tok/s, 14.22 tok/step), using the image's `labd_bench.py` (Step 0.5 must have shown `perf_counter` there; if the image copy uses `time.time()` or the bench was bind-mounted, skip the replay and say so; the floor then cannot be checked). Run `bash $HQ/base-C.sh 1`, then `bash $HQ/base-C.sh 2`.

`$HQ/base-C.sh`:
```bash
source $HQ/lib.sh; RND=$1; disklog; gate || exit 1
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0 DFLASH_TOKENS=15
boot single || exit 1
begin base-C-r$RND
suite
docker compose exec -T -w /app "$SVC" venv/bin/python bench/labd_bench.py C-r$RND --ctx 20000 > $OUT_H/labd_C.log 2>&1
labdfloor $OUT_H/labd_C.log copy 322 | tee $OUT_H/floor.txt
end
```
Note any `Maximum concurrency` in `checks.txt` below 8: DFLASH_TOKENS=15 has fewer request slots, so N=8 cells queue; record it. KV pool prior 57,669 tokens, `MAX_LEN` 57,344.

**base-A** (mode A, batch): `docker compose --profile batch`, `GPU_UTIL=0.91` (A spills at 0.93 on this box), `PTH_FA=0` (batch ignores it; run_suite needs it set). No `SPEC`/`CTX` (`arm_env` strips the box's `CTX`, batch warns on it). The repo's throughput protocol twice per boot, keep the second (the first includes JIT warm-up), then the suite. Run `bash $HQ/base-A.sh 1`, then `bash $HQ/base-A.sh 2`.

`$HQ/base-A.sh`:
```bash
source $HQ/lib.sh; RND=$1; disklog; gate || exit 1
arm_env GPU_UTIL=0.91 PTH_FA=0 PREFIX_CACHE=1 VISION=0
boot batch || exit 1
begin base-A-r$RND
docker compose exec -T -e OUT=$OUT_C/rb1 "$SVC" bash /app/bench/run_benchmarks.sh batch > $OUT_H/run_benchmarks_1.log 2>&1
docker compose exec -T -e OUT=$OUT_C/rb2 "$SVC" bash /app/bench/run_benchmarks.sh batch > $OUT_H/run_benchmarks_2.log 2>&1
grep '^ROW' $OUT_H/run_benchmarks_2.log | tee $OUT_H/headline_rows.txt
fv=$(floorv "A 64conc e2e" "$(grep -m1 '^ROW 64conc' $OUT_H/run_benchmarks_2.log | sed -E 's/.*e2e=([0-9.]+).*/\1/')" 733); frc=$?; echo "$fv" | tee $OUT_H/floor.txt
if [ $frc = 0 ]; then
  suite SALT=eagle4-base-A CONC=1,2,4,8,16,32 LC_CELLS="8000:1,2,4 32000:1,2" LADDER_CELLS="200:1,2,4,8,16,32,64 8000:1,2,4,8,16"
else echo "BELOW FLOOR: suite skipped (D15 abort)" | tee -a $OUT_H/floor.txt; fi
end
```
Headline: the `ROW 64conc 128in/512out` line of the second run (prior 814.3 e2e tok/s, 805.6-822.9; the row also carries `meanTTFT`). The suite uses a new salt because `run_benchmarks.sh` already ran on this boot. A floor failure skips the suite and still runs `end` (rule 6 abort, raw files kept). Pass: as bench-4; prior WDDM non-local 370-378 MiB flat at 0.91. Growth above 64 MiB is flagged, above 256 MiB aborts; do not lower `GPU_UTIL`.

### bench-12: SPEC=off fast-variant C1 decode (reviewer DFlash; no baseline.md gap, settles the speculation overhead per step, estimated 4.5-6.5 ms; independent cross-check of gap 6) | ~25-30 min

Same prompts, salt, `MAXTOK`, `REPS` and model as bench-4, one concurrency, no speculation. The model is whatever the launcher resolves (the box has `Qwen3.8-27B-W4A16-AutoRound-fast`, the default when present); record the `MODEL=` line from `checks.txt` and confirm it is the same as bench-4's.

```bash
source $HQ/lib.sh; disklog; gate || exit 1
arm_env SPEC=off CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
boot single || exit 1
begin specoff-C1
suite PHASES="snap warm probe sweep spill" CONC=1 LC_CELLS=      # SALT=eagle4-base, same as bench-4
end
```
Outputs: `sweep_specoff-C1_short_chat.json` and `sweep_specoff-C1_code.json` (N=1, 3 timed rounds, TTFT, decode tok/s, peak VRAM; tokens-per-step columns read `n/a` because there are no spec counters), plus the usual files. Fill in the existing `specoff-C1` row of `LEDGER.md` (Phase 3 table). Pass: as bench-4, probe WARM, WDDM growth recorded, `[effective-config] ... SPEC=off` (no decode floor: no prior value for SPEC=off). Return the N=1 decode rate and its CV next to bench-4's N=1 row, with both `ptok` equal (same prompts). Do not derive the overhead yourself beyond writing the two numbers side by side and, as raw arithmetic in `RESULT.md`, `ms_per_token_off = 1000/decode_off` and `ms_per_step_dflash2 = 1000 * tok_step / decode_dflash2` (tokens per step from bench-4's same cell); flag it as arithmetic on two boots, not a measurement of draft time. SPEC=off at CTX=fast uses bf16 KV like B, so the pair is like for like; if bench-7 showed a REAL divergence the generated lengths may differ, which `sweep.py` cannot equalize (no `--exact` knob in `run_suite.sh`).

## Return format

Update `LEDGER.md` and `gpu-queue.md` as you go and commit after every job. When the queue is finished, or you are blocked on something only a human can fix, reply with exactly this:

```
STEP 0: <one line each for 0.5 launchers, 0.6 pthfa, 0.7 frozen corpus (+sha), 0.8 prior raw files present/absent>
bench-1  | done|aborted|blocked|skipped | INFO|INVALID | results/env-<ts>/            | <one line, numbers with spread> | <flags>
bench-2  | ...
...                                             (bench-4/5/6/11a/11b: one line with both boots, "r1 x / r2 y", plus the boot-to-boot difference;
                                                 list bench-11a and bench-11b as their own lines, in the order they ran)
bench-12 | ...
bench-13 .. bench-25 | ...                      (phase-4 lines: verdict words WIN|LOSS|NEUTRAL|INFO|INVALID, format and extras under "Experiments")
ANOMALIES:
- <job>: <what, with the file that shows it>
```
Status words are the ledger's (`done`, `blocked`, `skipped`; use `aborted` for a rule-6 abort). Labels (D19, `00-meta/decisions.md:23`): every number you return is "measured (this box, date)" from a file under `results/`; a prior or upstream figure you quote next to it is "measured (reported)" or "E", never bare. A `WIN` is a single run by one agent: write no "speedup", "confirmed" or "reproduced" in `RESULT.md`; a second agent re-runs the gain before anyone calls it measured. `flags` are drawn from: NOT-WARM, SPILL64(+growth MiB; every run reports its growth, the flag is for growth above 64 MiB), CV>5%, CACHED, PREEMPT, THROTTLE(+reason), SUSPECT-RERUN, NO-LONG-CORPUS, PARTIAL(+cell skipped), GPUMEM-UNIT. The key numbers are raw: TTFT, decode tok/s, aggregate tok/s, tokens per step, peak VRAM (nvidia-smi, secondary), spill growth. Do not rank, explain or recommend. Anomalies to list always: WDDM growth above 64 MiB (job, boot, growth), a sanity-band miss, a floor line close to its floor, a sampler unit that is not MiB, a gate failure and wait time, any file you could not produce, any deviation from this prompt and why, any harness behaviour not in "Harness gaps". Finish by restoring the box `.env` (`cp $HQ/env.box-orig .env`), `docker compose --profile single --profile batch down`, `git status` (harness unmodified, only your result files and the two table files changed), and the final push.

## Harness gaps (found while writing this prompt; still open, work around as written above)

Fixed on the branch before this prompt was handed over (nothing to work around): `SMOKE=1` lets `sweep.py` run with `--reps` below 3 (bench-2); `LC_CELLS=` (set but empty) skips the long-context cells in `run_suite.sh`; the README commands use `docker compose exec -T` and carry no inline `.env` comments.

1. **No external time cap in `run_suite.sh`.** The 2 h rule is enforced by the `lib.sh` watchdog (`docker compose down`), which kills the suite mid-phase; later phases (ladder, greedy) are lost if a long sweep cell eats the budget, which is why the longest cells in bench-6 and bench-11a run last, outside the suite, behind an `elapsed` guard.
2. **WSL-vs-host clock drift is not recorded.** `baseline.md:23` and the README say the harness records it; `snapshot.sh` does not. `lib.sh` `clockdrift` (120 s window, about +-0.2% resolution) is the stand-in; run it in bench-1 and after bench-4/5/6.
3. **Sampler format and unit are unverified.** `gpumem_summary.py` takes the last numeric field of each line as MiB; the sampler's output format is not in the repo. bench-2 checks one real file; if the unit differs, the 64 MiB flag and 256 MiB abort bounds must be applied to an ad hoc converted figure.
4. **Profile phase size is not controllable.** `run_suite.sh` hard-codes `--set short_chat --conc 1 --reps 3 --warmup 0` for the profiled load, so 24 requests are profiled with torch `with_stack` on; the only lever is `MAXTOK` (bench-10 sets 64). There is no prompt-count knob and no per-request start/stop. `trace_breakdown.py` is run with defaults (newest worker trace, pure decode steps).
5. **No `--exact` knob in `run_suite.sh`.** `sweep.py` has `--exact` (fixed output length) but the suite does not pass it, so request lengths follow EOS. Fine at T=0 when greedy equality holds (bench-7); otherwise decode windows differ in length between arms (matters for bench-12).
6. **No draft or verify timer, no per-process VRAM.** Draft vs verify time exists only through the profile phase (bench-10); a permanent metric (`vllm:spec_decode_draft_time_seconds`) needs a variant image, which is experiment X-17 (D12): bench-13 builds the tagged variant image and measures it. Peak VRAM is nvidia-smi whole-card `memory.used` (flat by preallocation) and the WDDM counter is non-local usage only (spill), not total VRAM.
7. **Sampler start precedes the container.** `begin` starts the sampler after the boot, so the boot's own memory growth is not in `gpumem_sampler.csv`; the spill summary covers the measured window only (this matches the README).
8. **The D15 abort bounds are not in the scripts.** `gpumem_summary.py` defaults to the 64 MiB flag; the 256 MiB abort bound is applied with `--tol 256` in the poll command (rule 6), and the decode floors by the `lib.sh` helpers. `run_suite.sh` has no floor.

## Experiments (phase 4: bench-13..bench-25)

These jobs follow bench-12 in `gpu-queue.md` order (bench-13 first). Each tests one hypothesis (H-id) from `docs/research/02-eagle4/transferability.md` (index rows X-xx, lines 218-234; the `[review: ...]` marks there are the gates used below). Nothing here is tuning: run the arms exactly as written, apply the stated gate to the numbers, and return raw results. Every number is "measured (this box, date)" with its boot-to-boot spread; anything marked E (estimated) below is a prior expectation and is not a result.

### Rule changes for these jobs (everything else in "Rules" stands)

- **Rule 1 (order):** bench-13 to bench-25 in number order after bench-12, parts in letter order (bench-15a, 15b, 15c, 15d). No part is queued above 2.0 h; training runs are pilots with a stated token and step budget, and the full runs are proposals in this prompt (bench-23, bench-24), not queue rows. Gating is stated per job (`blocked` or `skipped`, never silently dropped). X-05 is not queued (not ranked: +2.2% E, below the bar).
- **Rule 2 (no builds):** lifted for bench-13 only: one `docker build` of the tagged variant image `hyperqwen:sha-2cf50a0-spectiming` (D12). Still forbidden: `docker compose --build`, `docker pull`, retagging or replacing the stock image, `docker system prune`. The variant is chained through a compose override file in `$HQ`, never by editing `docker-compose.override.yml`.
- **Rule 5 (downloads):** lifted for bench-17 only, for exactly one model: HF `EAGLE-Corp/Qwen3.8-27B-EAGLE4` (about 6.2 GB bf16). No other download, no pip install, anywhere. Anything else missing means `blocked` with the reason.
- **Licence (D11) for the EAGLE-4 drafter:** the licence is "other" (eagle-4) and allows evaluation only. The drafter and every file derived from it (converted `model.safetensors`, quantised copies, adapted copies) stay under `/mnt/g/models/_eagle4/` (and the download directory `/mnt/g/models/_eagle4-dl/`) on the box. Never copy them into the checkout or `results/`, never commit, push or upload them, never `docker cp` them out. `RESULT.md` carries sha256 values and shapes only.
- **Rule 6 floors:** the D15 decode floors (B 103, D 90) are checked only on arms that equal the stock baselines (the job says which). An arm that changes the thing under test is judged by its gate, not by a floor. Every other abort condition (restart, OOM, growth above 256 MiB, power not 250 W, wrong effective config) applies to every boot.
- **Rule 10 (edit scope):** adds `docs/research/04-benchmarks/results/<X-id>-<YYYYMMDD>/` (see below), `results/_glue/` (the small derived scripts below), the job's row in the Phase 4 table of `LEDGER.md`, and its row in `gpu-queue.md`. Still never the experiment folders, `plan.md`, `transferability.md`, the harness, the image or `patches/`.
- **Commit message:** `bench: X-NN results <YYYY-MM-DD>`; no attribution of any kind (Rule 9).

### Glue for the X jobs (create once, right after Step 0; untested glue: `bash -n` only)

`$HQ/xlib.sh` is sourced by every X job script after `lib.sh` (`source $HQ/lib.sh; source $HQ/xlib.sh`). It redefines `arm_env` with a longer strip list (the X arms set keys that `lib.sh` does not strip), adds the variant-image switch, and adds helpers.

```bash
XDIR=$REPO/docs/research/04-benchmarks/experiments
CF_STOCK=$COMPOSE_FILE; CF_VAR=$COMPOSE_FILE:$HQ/variant.yml
stock_on()   { export COMPOSE_FILE=$CF_STOCK HQ_TIMING=0; }
variant_on() { export COMPOSE_FILE=$CF_VAR; }               # HQ_TIMING is set by the caller (0 or 1)
xauto() {  # timing variant on (HQ_TIMING=1) only after the X-17 overhead gate wrote $HQ/x17.ok; stock otherwise
  if [ -f "$HQ/x17.ok" ]; then variant_on; export HQ_TIMING=1; else stock_on; fi; }
arm_env() {  # lib.sh arm_env plus the X keys; HQ_TIMING=1 appends HQ_SPEC_TIMING=1
  grep -vE '^(SPEC|CTX|PTH_FA|VISION|PREFIX_CACHE|DFLASH_TOKENS|DRAFT_TOKENS|GPU_UTIL|EXTRA_ARGS|FA2_FP8KV|DYNAMIC_K|LOOKUP|LOOKUP_ADAPTIVE|KV_MEM|MAX_LEN|DFLASH_MAX_LEN|MAX_SEQS|KV|MODEL|DRAFT|DRAFT_METHOD|DRAFT_SAMPLE|MTP_DRAFT_VOCAB|ASYNC_SCHED|MODELS_DIR|VLLM_WSL2_ENABLE_PIN_MEMORY|HQ_SPEC_TIMING[A-Z_]*)=' "$HQ/env.box-orig" > .env
  printf '%s\n' MODELS_DIR=/mnt/g/models VLLM_WSL2_ENABLE_PIN_MEMORY=1 "$@" >> .env
  [ "${HQ_TIMING:-0}" = 1 ] && echo HQ_SPEC_TIMING=1 >> .env; return 0; }
xsuite() { XT0=$(python3 -c 'import time; print(time.time())'); suite "$@"; r=$?; XT1=$(python3 -c 'import time; print(time.time())'); return $r; }
xtiming() {  # parse the timing lines of this boot for the xsuite window (variant-on arms only)
  [ "${HQ_TIMING:-0}" = 1 ] || return 0
  docker compose logs --no-color "$SVC" 2>&1 | python3 $XDIR/X-17-timing/parse_timing.py --label "$LABEL" --t0 $XT0 --t1 $XT1 --out $OUT_H/spectiming_$LABEL.json > $OUT_H/spectiming_$LABEL.txt 2>&1
  docker compose logs --no-color "$SVC" 2>&1 | python3 $XDIR/X-17-timing/parse_timing.py --windows > $OUT_H/spectiming_${LABEL}_windows.txt 2>&1; }
pyk() {  # pyk SCRIPT ARGS : python in the container with the bench path and the client key resolved
  docker compose exec -T -e PYTHONPATH=/app/bench:/harness "$SVC" bash -c 'source /app/resolve_api_key.sh 2>/dev/null && resolve_client_key 2>/dev/null; exec /app/venv/bin/python "$@"' _ "$@"; }
mksweep() {  # derived sweep.py: TEMP env var (default 0), PROMPT_DIR env var, any --set name. The harness file itself stays unmodified.
  mkdir -p $R/_glue
  sed -e 's|"temperature": 0,|"temperature": float(os.environ.get("TEMP", "0")),|' \
      -e 's|os.path.join(HERE, "prompts", args.set + ".jsonl")|os.path.join(os.environ.get("PROMPT_DIR", "/harness/prompts"), args.set + ".jsonl")|' \
      -e 's|choices=\["short_chat", "code", "long_context"\]|choices=None|' $H/sweep.py > $R/_glue/sweep_x.py
  [ "$(diff $H/sweep.py $R/_glue/sweep_x.py | grep -c '^>')" = 3 ] && python3 -m py_compile $R/_glue/sweep_x.py; }
xroll() {  # xroll X-03 17b : sets and exports XR=results/X-03-<date of the job's first part>/<part>
  [ -f "$HQ/$1.date" ] || date +%Y%m%d > "$HQ/$1.date"; export XR=$R/$1-$(cat "$HQ/$1.date")/$2; mkdir -p "$XR"; }
xn() { ls -d $R/X-*/*/$1-r* $R/$1-r* 2>/dev/null | wc -l; }   # boots of label prefix $1 so far, across parts (round counter)
```

`mksweep` is run once (X-03 needs it; X-09 reuses it). The container sees it as `/results/_glue/sweep_x.py`.

**Result layout.** Every boot keeps its raw directory from `begin`, `<x-id lowercase>-<arm>-r<N>-<ts>/` (the arm and round are in the label); `xpart.sh` (below) moves a part's boot directories to `results/<X-id>-<YYYYMMDD>/<part>/`, next to that part's `part.md`. `results/<X-id>-<YYYYMMDD>/RESULT.md` (written by the last part that ran) holds the combined table over all parts (the `RESULT.md` contents listed under "Per-job procedure", plus the arms table, the exact `.env` arm lines, every gate with its number and PASS/FAIL, and the verdict). Nothing duplicates raw files.

**Verdicts are mechanical.** Compute each gate from the numbers as written. `WIN` = every gate of the job met. `LOSS` = a gate failed and the measured value is on the wrong side of the bar. `NEUTRAL` = a gate cannot be decided (spread overlaps the bar, too few valid boots, a missing arm). `INFO` = measurements with no gate. `INVALID` = a validity failure (NOT-WARM twice, effective config not as intended). Do not interpret, tune or recommend. Win-rule reminders (from the plan): a C1 decode win needs a median of at least +3% over at least 3 interleaved boots after the warm-up probe; below +5% it needs at least 5 boots or a teacher-forced tokens-per-step times ms-per-pass argument (do not construct the argument: report `NEUTRAL, needs 5 boots`); N>=4 aggregate needs at least +5%. Spread facts: boot-to-boot 1-2% (cline 0-3.2%), greedy non-determinism +-3%, thermal settle -9.4% (hence `gate`).

**Variant image (X-17) in later jobs.** After bench-13 passes, `xauto` switches single-user jobs that say so onto the timing variant with `HQ_SPEC_TIMING=1`, so each of those boots also yields draft/verify/accept GPU time. Every comparison stays inside one job (same image for every arm of it). Jobs that must not run on a patched image (X-08 greedy equality, X-03/X-11/X-13 EAGLE-3 loading, X-10 mode A, X-06) call `stock_on`. Per-job text says `xauto` or `stock_on`. If `x17.ok` is absent, `xauto` means stock.

**Parts (queue rows).** Every queue row is one part of at most 2.0 h wall (E, boots included), named `<bench-N><letter>` (a single-part row keeps the bench number). A part is a complete, resumable unit: it boots, runs, writes `results/<X-id>-<date>/<part>/` and stops cleanly. Start it detached with `nohup bash $HQ/xpart.sh <X-id> <part> "<script args>" ["<script args>" ...] > $HQ/<part>.driver.log 2>&1 &` (X-id in directory form: `X-04p`, `X-02b`; each quoted argument is one `$HQ` script with its arguments, run in order, stopping at the first failure). `xpart.sh` skips a part whose `$HQ/<part>.done` exists, moves the part's boot directories into the part directory (also after an abort), and sets `.done` on success only. Re-running an aborted part starts again from fresh boots (never resume a boot; round counters count aborted boots too). Each part writes `<part>/part.md`: its numbers (measured, with spread), every gate evaluated so far, and one line `next part: runs|skipped (<gate and number>)`. A part skipped by its gate gets status `skipped` with that number. **The verdict reads across parts:** it is computed mechanically at the end of the last part that ran, over every part's boots (open the earlier part directories; do not recompute from memory); until then the ledger row stays `running`.

```bash
# $HQ/xpart.sh X-ID PART "script args" ["script args" ...]
source $HQ/lib.sh; source $HQ/xlib.sh; X=$1; P=$2; shift 2     # lib.sh's 2 h watchdog now bounds the whole part
[ -f $HQ/$P.done ] && { echo "$P done: skipped"; exit 0; }
xroll $X $P; touch $HQ/.xp-start; rc=0
for c in "$@"; do bash $HQ/$c || { rc=$?; break; }; done
find $R -mindepth 1 -maxdepth 1 -type d -newer $HQ/.xp-start ! -name '_glue' ! -name 'X-*' -exec mv {} $XR/ \;
[ $rc = 0 ] && touch $HQ/$P.done; exit $rc
```

---

### bench-13a/13b/13c: X-17 variant build, smoke and overhead gate, draft/verify timing on B and D (ledger `X-17`, hypothesis: decision D12, no H) | 13a 1.25 h, 13b 1.6 h, 13c 0.7 h

Purpose: a permanent per-step GPU timer (verify, accept, draft, gap) for later jobs, and the one-time proof that it costs nothing measurable. Source: `experiments/X-17-timing/README.md`. Spans are GPU-timeline; for DFlash2 the `draft` phase is the whole `propose()` (the drafter forward cannot be split out of the FULL graph), for MTP it is the prefill pass plus k-1 graph replays.

**Parts** (`xpart.sh X-17 <part> ...`): **13a** = `"x17-build.sh" "x17-ab.sh S U"` (build plus boots S and U, about 1.25 h; stops `blocked` if the build fails); **13b** = `"x17-ab.sh T T S" "x17-gate.sh"` (boots T, T, S, then the gate, about 1.6 h; runs only if 13a is done); **13c** = `"x17-d.sh"` (one boot, about 0.7 h; runs only if `$HQ/x17.ok` exists, else `skipped: overhead gate failed`). Boot order across 13a and 13b: S U T T S.

**Build (no GPU, about 15 min; first element of 13a, script `$HQ/x17-build.sh`).** The only `docker build` in the queue.

```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog
docker build -f $XDIR/X-17-timing/Dockerfile.variant -t hyperqwen:sha-2cf50a0-spectiming $XDIR/X-17-timing 2>&1 | tee $HQ/x17-build.log
grep -q 'spec_timing: inert when HQ_SPEC_TIMING is unset' $HQ/x17-build.log || { echo "build/patch failed"; exit 1; }
docker image inspect -f '{{.Id}}' hyperqwen:sha-2cf50a0-spectiming | tee $HQ/x17-image.id
printf 'services:\n  single:\n    image: hyperqwen:sha-2cf50a0-spectiming\n    pull_policy: never\n' > $HQ/variant.yml
```
If the build fails (patch does not apply), status `blocked` with the build log tail; X-17 then stays `blocked`, `x17.ok` is never written and every later `xauto` is stock.

**Overhead A/B on mode B (5 boots over 13a and 13b).** Arms (all `SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0`, default `GPU_UTIL`, `DFLASH_TOKENS` unset, `LOOKUP` unset = default 1; PTH_FA=0 pinned):

| arm | image | `HQ_SPEC_TIMING` |
|---|---|---|
| S (stock) | `sha-2cf50a0` | unset |
| U (variant, off) | spectiming | unset |
| T (variant, on) | spectiming | `1` |

About 0.5 h per boot.

`$HQ/x17-ab.sh`:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog
for a in "$@"; do
  gate || exit 1
  case $a in S) stock_on;; U) variant_on; export HQ_TIMING=0;; T) variant_on; export HQ_TIMING=1;; esac
  RND=$(( $(xn x17-$a) + 1 ))
  arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
  boot single || exit 1
  begin x17-$a-r$RND
  xsuite PHASES="snap warm probe sweep spill" CONC=1,2,4 LC_CELLS="8000:1"
  sweepfloor $OUT_H/sweep_x17-${a}-r${RND}_short_chat.json 103 | tee $OUT_H/floor.txt
  xtiming
  end
done
```
Smoke (read from the first T boot's `container.log`): the `HQ_SPEC_TIMING_START` line is present with `"tokens":true`, `dropped` stays 0 in every `HQ_SPEC_TIMING {` line, and the parser's `implied tok/s` is within 5% of the sweep's N=1 decode (a miss is an anomaly, not a gate). No START line means the V1 runner was picked: record, gate FAIL.

**The gate (no GPU; script `$HQ/x17-gate.sh`, last element of 13b).** Writes `$XR/overhead_gate.txt` (the 13b part directory) and, only on PASS, `$HQ/x17.ok`:
```bash
source $HQ/lib.sh        # XR is exported by xpart.sh
R=$R HQ=$HQ python3 - <<'EOF' | tee $XR/overhead_gate.txt
import json, glob, os, re, statistics as st
R = os.environ["R"]
G = lambda p: glob.glob(f"{R}/X-17-*/*/{p}") + glob.glob(f"{R}/{p}")      # part directories (moved) and the current part (not yet moved)
def c1(a):
    v = []
    for f in sorted(G(f"x17-{a}-r*/sweep_x17-{a}-r*_short_chat.json")):
        c = [c for c in json.load(open(f))["cells"] if c["conc"] == 1][0]; v.append(c["summary"]["rate"]["mean"])
    return v
S, U, T = c1("S"), c1("U"), c1("T")
print("C1 decode tok/s: stock", S, "variant-off", U, "variant-on", T)
ratio = st.mean(T) / st.mean(S); print("variant-on / stock =", round(ratio, 4), "; variant-off / stock =", round(st.mean(U) / st.mean(S), 4))
log = open(sorted(G("x17-T-r1-*/container.log"))[0], errors="replace").read()
tokens = '"tokens":true' in log.split("HQ_SPEC_TIMING_START")[1][:600] if "HQ_SPEC_TIMING_START" in log else False
dropped = [int(x) for x in re.findall(r'"dropped":(\d+)', log)]
print("START tokens:true =", tokens, "| dropped max =", max(dropped or [-1]))
ok = ratio >= 0.97 and tokens and bool(dropped) and max(dropped) == 0
print("GATE", "PASS" if ok else "FAIL")
if ok: open(os.environ["HQ"] + "/x17.ok", "w").write("ok\n")
EOF
```
Gate (README: expect a difference within the greedy +-3% noise): variant-on mean C1 decode at least 0.97 of the stock mean, `tokens:true`, `dropped` 0. On FAIL: no `x17.ok`, the numbers stay as measurements (verdict `INVALID` for the overhead claim), later jobs run stock.

**13c, draft and verify time on D (one boot, about 0.7 h; script `$HQ/x17-d.sh`).** Runs only if `x17.ok` exists:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog
[ -f $HQ/x17.ok ] || { echo "x17.ok absent: skipped"; exit 0; }
gate || exit 1; variant_on; export HQ_TIMING=1
arm_env SPEC=mtp CTX=long PREFIX_CACHE=1 PTH_FA=1 VISION=0
boot single || exit 1
begin x17-D-r1
xsuite PHASES="snap warm probe sweep spill" CONC=1 LC_CELLS="8000:1 32000:1"
sweepfloor $OUT_H/sweep_x17-D-r1_short_chat.json 90 | tee $OUT_H/floor.txt
xtiming
end
```
Outputs: `spectiming_<label>.{json,txt}` and `_windows.txt` per T and D boot, the sweeps, usual files. Return, for B (mean of the two T boots) and D: mean ms per step of `verify`, `accept`, `draft`, `gap`, `total`, tokens per step, the parser's footer phase shares, and the parser's `implied tok/s` next to the sweep decode. No derived claim (draft cost, floor) beyond what the parser prints. Verdict: `INFO` (gate PASS) or `INVALID` (gate FAIL). Estimated 3.6 h in all (0.25 h build + 2.5 h A/B + 0.1 h gate + 0.7 h D). The verdict is computed at the end of 13b (gate FAIL: `INVALID`, 13c skipped) or of 13c, over all parts.

---

### bench-14a/14b: X-08 near-tie classifier on the greedy gate, GPU add-on to bench-7 (ledger `X-08`, H08) | 14a about 1.0-1.4 h, 14b about 1.2 h

Gate (index, reviewed): every non-near-tie divergence is reproducible (re-record the same boot once and get the same k and status), and the KVarN adaptive + prefix case is flagged REAL. Stock image, `stock_on`. The recorder, classifier and replay run **inside** the container (the client key and `/app/bench/harness.py` are there) from `/results/_glue/x08/`; classification itself runs on the host (stdlib python3). Passing the threshold flags explicitly matters: `--near-tie 0.01`, and **`--ulp-band 0.25`** (the code default is now 0.25 as well, so this only guards against a change; D18 and the review say 0.25, and `judge --tol` defaults to 0.25).

Common setup (top of both scripts): `mkdir -p $R/_glue/x08; cp $XDIR/X-08/{classify_divergence,rescore_replay}.py $R/_glue/x08/; sed 's|^HARNESS = .*|HARNESS = "/harness"|' $XDIR/X-08/record_top2.py > $R/_glue/x08/record_top2.py; D=$XR; G=/results/_glue/x08; DC=/results/${XR#$R/}`. [review: `record_top2.py:22` finds the prompts at `<script dir>/../../harness`, which from `/results/_glue/x08/` is the missing `/results/harness`; the `sed` points it at the `/harness` mount.]

**14a main set: `xpart.sh X-08 14a "x08-main.sh"`** (NS, DF2, MTP boots as in bench-7, each recorded twice, plus a replay boot only if needed):
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; stock_on
mkdir -p $R/_glue/x08; cp $XDIR/X-08/{classify_divergence,rescore_replay}.py $R/_glue/x08/; sed 's|^HARNESS = .*|HARNESS = "/harness"|' $XDIR/X-08/record_top2.py > $R/_glue/x08/record_top2.py
D=$XR; G=/results/_glue/x08; DC=/results/${XR#$R/}
for v in "NS SPEC=off" "DF2 SPEC=dflash2" "MTP SPEC=mtp"; do set -- $v
  gate || exit 1
  arm_env $2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0
  boot single || exit 1
  begin x08-$1
  suite PHASES="snap warm probe"
  for lab in $1 ${1}b; do   # second recording on the same boot = run-to-run noise (its pass 1 is already prefix-cache warm)
    pyk $G/record_top2.py $lab --sets short_chat,code --max-tokens 256 --top-n 5 --passes 2 --out $DC
  done
  end
done
C="python3 $XDIR/X-08/classify_divergence.py"
$C classify $D/greedy_NS.json $D/greedy_DF2.json $D/greedy_MTP.json --near-tie 0.01 --ulp-band 0.25 --out $D/report.json --rescore-out $D/requests.json > $D/classify.txt; echo "exit=$?" >> $D/classify.txt
for x in NS DF2 MTP; do $C classify $D/greedy_$x.json $D/greedy_${x}b.json --near-tie 0.01 --ulp-band 0.25 --out $D/repro_$x.json > $D/repro_$x.txt; echo "exit=$?" >> $D/repro_$x.txt; done
if grep -q '^exit=3' $D/classify.txt; then   # ulp_band entries pending: replay on SPEC=off, same CTX / KV / PTH pin as the reference
  gate || exit 1; arm_env SPEC=off CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0; boot single || exit 1; begin x08-replay
  suite PHASES="snap warm probe"
  pyk $G/rescore_replay.py $DC/requests.json --out $DC/replays.json
  end
  $C judge $D/report.json $D/replays.json --tol 0.25 --out $D/final.json > $D/judge.txt; echo "exit=$?" >> $D/judge.txt
fi
```
**14b KVarN adaptive + prefix: `xpart.sh X-08 14b "x08-kvarn.sh"`** (runs after 14a unless 14a ended `blocked` by a box failure; independent of 14a's verdict) (two boots, `CTX=huge`, arm (a) forces the adaptive lookup the launcher otherwise pins off, arm (b) is the reference with the pin left on; both `DFLASH_TOKENS=15`, `PREFIX_CACHE=1`, `PTH_FA=0`, `LOOKUP` unset = default 1; the lookup state is the same in both arms apart from the pin). No replay (the reference is arm (b) itself):
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; stock_on
mkdir -p $R/_glue/x08; cp $XDIR/X-08/classify_divergence.py $R/_glue/x08/; sed 's|^HARNESS = .*|HARNESS = "/harness"|' $XDIR/X-08/record_top2.py > $R/_glue/x08/record_top2.py
D=$XR; G=/results/_glue/x08; DC=/results/${XR#$R/}
for a in a b; do
  gate || exit 1; x=; [ $a = a ] && x=LOOKUP_ADAPTIVE=1
  arm_env SPEC=dflash2 CTX=huge PREFIX_CACHE=1 DFLASH_TOKENS=15 PTH_FA=0 VISION=0 $x
  boot single || exit 1
  begin x08-kvarn-$a
  suite PHASES="snap warm probe"
  for rec in "8k 8000" "6k 6500" "8kr 8000"; do set -- $rec     # two document lengths, then the first again (reproducibility)
    pyk $G/record_top2.py K$a$1 --sets "" --doc-ctx $2 --corpus /cache/bench/labd_corpus_long.txt --max-tokens 256 --top-n 5 --passes 2 --out $DC
  done
  end
done
C="python3 $XDIR/X-08/classify_divergence.py"
for l in 8k 6k; do $C classify $D/greedy_Kb$l.json $D/greedy_Ka$l.json --near-tie 0.01 --ulp-band 0.25 --out $D/kvarn_$l.json > $D/kvarn_$l.txt; echo "exit=$?" >> $D/kvarn_$l.txt; done
for a in a b; do $C classify $D/greedy_K${a}8k.json $D/greedy_K${a}8kr.json --near-tie 0.01 --ulp-band 0.25 --out $D/kvarn_repro_$a.json > $D/kvarn_repro_$a.txt; echo "exit=$?" >> $D/kvarn_repro_$a.txt; done
```
Notes: `record_top2` keys `copy` (pass 1, cold) and `copy@p2` (pass 2, prefix-cache hit); the repeat recording's pass 1 is already cache-warm (state it in `RESULT.md`). Expected from the README (E, not a result): pass 1 identical; `copy@p2` REAL at k about 10-15 in arm (a) vs (b); arm (b) vs (b) identical. If a boot at `CTX=huge` does not come up, the row is `blocked` with the refusal text.
Verdict (mechanical, per the gate; computed at the end of 14b over both parts, 14a supplies condition (1), 14b condition (2)): `WIN` = (1) in the main set no divergence is `real`/`unknown` that failed to reproduce in the `*b` repeat (same k, same status; show each non-near-tie divergence with its repeat line) and (2) at both document lengths the (a)-vs-(b) classify has `copy@p2` `real` (exit 1) and each repeat reproduces it at the same k. `LOSS` = (2) fails (the KVarN case is not flagged); `NEUTRAL` = (1) fails or the exit code is 2/3 after the replay. `INFO` lines: the margin histogram (top1-top2 values clustered on multiples of 0.125 would confirm the bf16 ulp assumption; list the ten most common margins), exit codes of every classify.

Return: the exit code and first-divergence table of each classify, the near-tie / ulp_band / real counts, reproducibility lines, the KVarN lines.

---

### bench-15a/15b/15c/15d: X-01b PTH_FA=1 vs PTH_FA=0 on `SPEC=mtp CTX=long`, long-context decode (ledger `X-01b`, H01) | 15a, 15b, 15c about 1.4 h each (one interleaved round per part); 15d about 1.5 h (conditional)

Comparator (reviewed): the pre-#345 default for an unset `PTH_FA` on `mtp/long` (the old checkout `95ac251`; upstream and this branch now default to `PTH_FA=1`, D22), which is `PTH_FA=0` = fp8 KV on FlashInfer with piecewise graphs only (exactly what PR #345 compared). Arm p1 is `PTH_FA=1`. Expected (E, upstream-reported, M elsewhere): +12-22% decode at 66-93k prompt tokens (cells: nominal `--ctx 58000` and `82000`, about 66k and 93k tokens, as bench-9's 58000; [review: nominal 66000/93000 would have been about 74k/105k tokens]); +21.3/+22.1% on WSL2 with 3 boots per arm; part of the gain is FULL vs PIECEWISE graphs, not the kernel alone. This overlaps bench-9's mtp sub-runs, which stay the default-`GPU_UTIL` reference; here both arms use `GPU_UTIL=0.89` (fp8 spilled at 0.93 upstream). `xauto` (timing variant when available).

Arms: `SPEC=mtp CTX=long PREFIX_CACHE=1 VISION=0 GPU_UTIL=0.89 PTH_FA=<0|1>`; nothing else (no `DFLASH_TOKENS`, `DYNAMIC_K`, `FA2_FP8KV`, `EXTRA_ARGS`: `PTH_FA=1` refuses them). Three interleaved boots per arm, one round (p0 then p1) per part: **15a** = `xpart.sh X-01b 15a "x01b-ab.sh 1"`, **15b** = `... 15b "x01b-ab.sh 2"`, **15c** = `... 15c "x01b-ab.sh 3" "x01b-gate.sh"` (the gate script reads all three rounds); `SALT=x01b`, same corpus. 15b runs only if 15a is done and valid, 15c only if 15b is.

`$HQ/x01b-ab.sh`:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; RND=$1; disklog; xauto
for p in 0 1; do
  gate || exit 1
  arm_env SPEC=mtp CTX=long PREFIX_CACHE=1 PTH_FA=$p GPU_UTIL=0.89 VISION=0
  boot single || exit 1
  begin x01b-p$p-r$RND
  xsuite SALT=x01b PHASES="snap warm probe sweep spill" CONC=1 LC_CELLS=        # D C1 at short context (reported, not gated)
  for c in 58000 82000; do      # nominal labd --ctx (chars = ctx x 3.6; the corpus is ~3.2 chars/token): about 66k and 93k prompt tokens; the per-task lines' prompt= is the truth
    [ $(elapsed) -lt 100 ] || { echo "SKIPPED labd $c: $(elapsed) min elapsed" | tee $OUT_H/labd_$c.log; continue; }
    docker compose exec -T -w /app "$SVC" venv/bin/python bench/labd_bench.py x01b-p$p-r$RND-$c --ctx $c --corpus $CORPUS > $OUT_H/labd_$c.log 2>&1
  done
  xtiming; end
done
```
**15d, quality (GSM8K), conditional:** `xpart.sh X-01b 15d "x01b-q.sh 0" "x01b-q.sh 1"`, one boot per arm back to back; it runs only if `x01b-gate.sh` (end of 15c) wrote `$HQ/x01b.go-q` (the first line of `x01b-q.sh` is `[ -f $HQ/x01b.go-q ] || { echo "skipped: decode gate not met"; exit 0; }`). If it is skipped the verdict is computed without GSM8K and can only be `LOSS` or `NEUTRAL`. (`arm_env` as above with `PTH_FA=$1`, `begin x01b-q-p$1`, `suite PHASES="snap quality"`, `end`; about 45 min each). The guard's own ppl/GSM8K FAIL against the prior reference is recorded but is not the gate (the gate is p1 against p0 on this box).

`$HQ/x01b-gate.sh` (last element of 15c):
```bash
source $HQ/lib.sh
R=$R HQ=$HQ python3 - <<'EOF' | tee $XR/ratios.txt
import glob, os, re, statistics as st
R = os.environ["R"]; G = lambda p: glob.glob(f"{R}/X-01b-*/*/{p}") + glob.glob(f"{R}/{p}")
def v(p, c, key):
    out = []
    for f in sorted(G(f"x01b-p{p}-r*/labd_{c}.log")):
        m = re.search(r"TOTAL.*?" + key + r"=([0-9.]+)", open(f, errors="replace").read())
        if m: out.append(float(m.group(1)))
    return out
go = True
for c in (58000, 82000):
    d0, d1, t0, t1 = v(0, c, "decode"), v(1, c, "decode"), v(0, c, "tok/step"), v(1, c, "tok/step")
    if min(len(d0), len(d1), len(t0), len(t1)) < 3: print(c, "fewer than 3 boots per arm", d0, d1); go = False; continue
    r, ts = st.median(d1) / st.median(d0), st.median(t1) / st.median(t0)
    print(c, "decode p0", d0, "p1", d1, "median ratio", round(r, 4), "| tok/step median ratio", round(ts, 4))
    go = go and r >= 1.10 and abs(ts - 1) <= 0.02
print("15d GSM8K boots:", "RUN" if go else "SKIP")
if go: open(os.environ["HQ"] + "/x01b.go-q", "w").write("ok\n")
EOF
```
Per boot return: the labd `TOTAL` line at each cell (`tok/step=`, `decode=`) and the per-task `prompt=` values (the `TOTAL` line has no `prompt=`), the sweep C1 rows, the `[effective-config] PTH_FA=` line, KV pool tokens, WDDM growth. No decode floor (no prior value at `GPU_UTIL=0.89` and for `PTH_FA=0`); rule 6 otherwise.
Gate (reviewed index): WIN iff, at both cells, median(p1 labd TOTAL decode) / median(p0) is at least 1.10 over the 3 boots, median tok/step of p1 within +-2% of p0, and GSM8K(p1) - GSM8K(p0) within 1 point. LOSS iff p1 median decode is more than 3% below p0 at either cell. NEUTRAL otherwise (including a skipped labd cell or fewer than 3 valid boots per arm). Report the three per-boot ratios as well as the medians.

---

### bench-16a/16b/16c: X-04′ int8 per-token-head KV via TRITON on mode B (ledger `X-04′`, directory id `X-04p`, H04′) | 16a 1.4 h, 16b 1.4 h, 16c 0.7 h (conditional)

Reproduces cline S-2 (reference, measured elsewhere: pool 68,605 to 116,440 tokens = 1.70x, N=8 preemptions 23 to 1, C1 111.2 vs 111.0). Arm A = stock B; arm X = B with the int8 TRITON KV, no launcher edit (`EXTRA_ARGS` is expanded after the launcher's own flags, so the later flags win; the `[effective-config]` and the log must show `TRITON_ATTN` and `int8_per_token_head`, else the arm silently ran stock: abort that boot). The PTH_FA sub-arm needs a launcher allow-list edit (`start_qwen.sh:299-304`), which is not allowed here: not queued, no row. The 2-8 stream recipe (`DFLASH_TOKENS=3` + int8) is already measured and not run. `xauto`.

| arm | env |
|---|---|
| A | `SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0` |
| X | A plus `EXTRA_ARGS=--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head` |

ABBA over two parts: **16a** = `xpart.sh X-04p 16a "x04p.sh A X"`, **16b** = `xpart.sh X-04p 16b "x04p.sh X A" "x04p-gate.sh"` (runs only if 16a is done and valid). **16c**, the quality boot = `xpart.sh X-04p 16c "x04p.sh Q"`, runs only if `x04p-gate.sh` wrote `$HQ/x04p.go-q` (pool >= 1.6x and C1 within 2%); otherwise skipped and the verdict is computed without GSM8K (`LOSS` or `NEUTRAL` only).

`$HQ/x04p.sh`:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; xauto
for a in "$@"; do
  gate || exit 1
  x=; [ $a != A ] && x="EXTRA_ARGS=--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
  arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0 "$x"
  [ -z "$x" ] && sed -i '/^$/d' .env
  boot single || exit 1
  RND=$(( $(xn x04p-$a) + 1 )); begin x04p-$a-r$RND
  if [ $a = Q ]; then xsuite PHASES="snap quality"
  else xsuite PHASES="snap warm probe sweep ladder spill" CONC=1,2,4,8 LC_CELLS="8000:1 32000:1" LADDER_CELLS="200:1,2,4,8 4000:1,2,4,8"
    [ $a = A ] && sweepfloor $OUT_H/sweep_x04p-A-r${RND}_short_chat.json 103 | tee $OUT_H/floor.txt; fi
  xtiming; end
done
```
(`Q` runs the X arm: `x` is non-empty for every arm other than `A`. A quality boot uses the launcher default `GPU_UTIL`, as bench-8.) `$HQ/x04p-gate.sh` (last element of 16b; the preemption count is read by hand from the N=8 ladder row):
```bash
source $HQ/lib.sh
R=$R HQ=$HQ python3 - <<'EOF' | tee $XR/gates.txt
import glob, json, os, re, statistics as st
R = os.environ["R"]; G = lambda p: glob.glob(f"{R}/X-04p-*/*/{p}") + glob.glob(f"{R}/{p}")
def c1(a):
    v = []
    for f in sorted(G(f"x04p-{a}-r*/sweep_x04p-{a}-r*_short_chat.json")):
        v.append([c for c in json.load(open(f))["cells"] if c["conc"] == 1][0]["summary"]["rate"]["mean"])
    return v
def pool(a):
    out = []
    for f in sorted(G(f"x04p-{a}-r*/container.log")):
        m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", open(f, errors="replace").read())
        if m: out.append(int(m[-1].replace(",", "")))
    return out
A, X, pa, px = c1("A"), c1("X"), pool("A"), pool("X")
print("C1 decode A", A, "X", X, "| pool tokens A", pa, "X", px)
ok = bool(A and X and pa and px) and min(px) / max(pa) >= 1.6 and abs(st.median(X) / st.median(A) - 1) <= 0.02
print("pool ratio", round(st.median(px) / st.median(pa), 3) if pa and px else None, "C1 ratio", round(st.median(X) / st.median(A), 4) if A and X else None)
print("16c quality boot:", "RUN" if ok else "SKIP")
if ok: open(os.environ["HQ"] + "/x04p.go-q", "w").write("ok\n")
EOF
```
Outputs per boot: sweeps (C1 and N=1/2/4/8 aggregate), long cells, `ladder_200.json`, `ladder_4000.json` (preemptions in the N=8 row; ms/pass; KV%), `ladder_*_vram.json`, KV pool line in `checks.txt`.
Gates (reviewed): pool tokens(X) >= 1.6 x pool tokens(A) (from the `GPU KV cache size: N tokens` lines); N=8 preemptions at 4k prompts <= 1 in the X boots; C1 decode (short_chat N=1) median(X) within 2% of median(A) over 2 boots each; GSM8K of the X quality boot within 1 point of bench-8's `quality-B` GSM8K (same box; quote both). WIN iff all four; LOSS iff the pool gate or the C1 gate fails; NEUTRAL otherwise (e.g. preemptions 2-3, C1 2-3% off with 2 boots). Also report N=8 aggregate and the E expectation line (cline: C1 tie) as raw numbers only.

---

### bench-17a/17b/17c: X-03 load the EAGLE-4 drafter as EAGLE-3 (V3) and measure acceptance (ledger `X-03`, H03) | 17a about 1.0 h (download and conversion 0.5 h without GPU, pre-gate 0.5 h), 17b about 1.85 h (2 boots), 17c about 0.9 h (1 boot)

Gate (reviewed): V3 tok/step >= 1.12 x DFlash2 tok/step, same prompts, same run, T=0, with the `LOOKUP` state stated. The older absolute bar (3.75 tok/step at the 6.75 ms floor, 4.40 at x1.83 step cost; need = 3.36 x (29.4 - 3.3 + t_draft) / 29.4, baseline 3.36 includes LOOKUP hits) is reported next to the ratio. Untested risks (do not assume): `EagleSpeculator` on a GDN-hybrid target, M-RoPE vs a 1-D draft rope. Stock image, `stock_on`.

**17a step 1, download and conversion (no GPU, rule 5 lifted for this one model; licence rule above).** Script `$HQ/x03-prep.sh`:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; stock_on
free=$(df --output=avail -B1 /mnt/g | tail -1); [ "$free" -ge 25000000000 ] || { echo "blocked: disk ($free B free on /mnt/g, need 25 GB)" | tee $HQ/x03-prep.status; exit 1; }
mkdir -p /mnt/g/models/_eagle4/v3 /mnt/g/models/_eagle4-dl
echo "DOWNLOAD START $(date -u +%FT%TZ) free=$free" | tee -a $HQ/x03-download.log
docker run --rm -v /mnt/g/models:/app/models --entrypoint /app/venv/bin/python $IMG -c '
from huggingface_hub import HfApi, snapshot_download
rev = HfApi().model_info("EAGLE-Corp/Qwen3.8-27B-EAGLE4").sha; print("HF_COMMIT", rev)
print("DOWNLOADED", snapshot_download("EAGLE-Corp/Qwen3.8-27B-EAGLE4", revision=rev, allow_patterns=["model.safetensors", "config.json"], local_dir="/app/models/_eagle4-dl"))' 2>&1 | tee -a $HQ/x03-download.log
echo "DOWNLOAD END $(date -u +%FT%TZ)" | tee -a $HQ/x03-download.log
D=/mnt/g/models/_eagle4-dl; ls -l $D/model.safetensors $D/config.json | tee -a $HQ/x03-download.log
sha256sum $D/model.safetensors $D/config.json | tee $HQ/x03-sha.txt          # the SHA record; also goes in RESULT.md
sz=$(stat -c %s $D/model.safetensors); [ "$sz" = 6178152384 ] || { echo "blocked: size $sz, reviewed artifact is 6178152384 (commit f58a598)" | tee $HQ/x03-prep.status; exit 1; }
grep HF_COMMIT $HQ/x03-download.log | tee -a $HQ/x03-sha.txt                  # record vs the reviewed f58a598 (a different commit with the same size is recorded, not blocked)
docker run --rm -v /mnt/g/models:/app/models -v $HQ:/hq --entrypoint /app/venv/bin/python $IMG /hq/x03_convert.py 2>&1 | tee $HQ/x03-convert.log
sha256sum /mnt/g/models/_eagle4/v3/model.safetensors /mnt/g/models/_eagle4/v3/config.json | tee -a $HQ/x03-sha.txt
```
`$HQ/x03_convert.py` (write it as is; it is conversion only, no training):
```python
import json, os, shutil
from safetensors import safe_open
from safetensors.torch import save_file
S, D = "/app/models/_eagle4-dl", "/app/models/_eagle4/v3"
t, shapes = {}, {}
with safe_open(f"{S}/model.safetensors", "pt") as f:
    for k in f.keys(): shapes[k] = tuple(f.get_slice(k).get_shape())
print("HEADER", len(shapes), "tensors"); [print("  ", k, shapes[k]) for k in sorted(shapes)]
n = sum(a * b if len(s) == 2 else s[0] for s in shapes.values() for a, b in [(s[0], s[1] if len(s) > 1 else 1)])
sfx = lambda s: {k: v for k, v in shapes.items() if k.endswith(s)}
chk = {"tensors==19": len(shapes) == 19, "params==3089075200": n == 3089075200,
       "fc (5120,15360)": any(v == (5120, 15360) for v in sfx("fc.weight").values()),
       "lm_head (248320,5120)": any(v == (248320, 5120) for v in sfx("lm_head.weight").values()),
       "layers.0 q/k/v (5120,10240)": all(any(v == (5120, 10240) for k, v in shapes.items() if "layers.0." in k and p in k) for p in ("q_proj", "k_proj", "v_proj"))}
print("T10", chk); assert all(chk.values()), "header mismatch: blocked"
drop = [k for k in shapes if "embed_tokens" in k or "efficient_lm_head" in k]; print("DROP", drop)
with safe_open(f"{S}/model.safetensors", "pt") as f:
    t = {k: f.get_tensor(k) for k in f.keys() if k not in drop}
os.makedirs(D, exist_ok=True); save_file(t, f"{D}/model.safetensors", metadata={"format": "pt"})
c = json.load(open(f"{S}/config.json")); old = dict(architectures=c.get("architectures"), aux=c.get("eagle_aux_hidden_state_layer_ids"))
c["architectures"] = ["LlamaForCausalLMEagle3"]; c["eagle_aux_hidden_state_layer_ids"] = [4, 32, 60]
json.dump(c, open(f"{D}/config.json", "w"), indent=1)
print("CONFIG_DIFF old", old, "new", dict(architectures=c["architectures"], aux=c["eagle_aux_hidden_state_layer_ids"]))
```
A failed header check, a size mismatch or `blocked: disk` makes X-03 `blocked` (X-11 and X-13 `skipped: X-03 blocked`). The location of the `eagle_aux_hidden_state_layer_ids` key in the config is unverified: the pre-gate log shows which ids vLLM actually uses; a mismatch with `[4, 32, 60]` is a finding to report, not something to fix.

**17a step 2, pre-gate (load-only, about 0.5 h): script `$HQ/x03-pregate.sh`.** Row command: `xpart.sh X-03 17a "x03-prep.sh" "x03-pregate.sh"` (a `blocked` exit of either stops the part).
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; stock_on; gate || exit 1
for kvm in 3000000000 2600000000; do     # second value only on a "KV cache needed" refusal
  arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0 DRAFT_METHOD=eagle3 DRAFT=/app/models/_eagle4/v3 DFLASH_TOKENS=7 LOOKUP=0 DFLASH_MAX_LEN=16384 KV_MEM=$kvm
  if boot single; then echo $kvm > $HQ/x03-kvmem.txt; break; fi
  docker compose logs --no-color single 2>&1 | grep -qiE 'KV cache.*(needed|larger)|available KV cache' || break
  docker compose --profile single down
done
[ -s $HQ/x03-kvmem.txt ] || { begin x03-pregate; end; echo "pre-gate: boot failed"; exit 1; }
begin x03-pregate
suite SMOKE=1 REPS=1 CONC=1 PHASES="snap probe sweep" LC_CELLS=
grep -iE 'page size|page_size|GPU KV cache size|Maximum concurrency|eagle|aux_hidden|effective-config' $OUT_H/container.log 2>/dev/null | head -40   # also copy into checks (end writes container.log)
end
```
Record in `RESULT.md`: the page size lines (expected 20,480 B/token for the draft page vs 1,835,008 B for the primary page, padded about +25%: E, check), pool tokens, the effective config, the aux ids in use, whether the draft method resolved to the EAGLE-3 speculator. Pass: the boot reaches `/health`, one sweep request returns tokens, and the accept rate in the sweep cell is below 0.98. **Alarm:** acceptance of about 1 means wrong aux ids or a wrong head mapping: pre-gate FAIL, X-03 `blocked` (verdict `INVALID`, the log kept). A boot failure is also a FAIL with the refusal text quoted (that is the finding). Fixed `KV_MEM` for all X-03 arms is the value in `$HQ/x03-kvmem.txt`.

**17b and 17c, the run:** **17b** = `xpart.sh X-03 17b "x03-run.sh V3" "x03-run.sh DL1"` (2 boots, about 1.85 h; runs only if 17a is done and its pre-gate passed; V3 first so that a broken V3 stops the part early); **17c** = `xpart.sh X-03 17c "x03-run.sh DL0"` (1 boot, about 0.9 h; runs after 17b, INFO only). All arms: `SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0 DFLASH_TOKENS=7 DFLASH_MAX_LEN=16384 KV_MEM=$(cat $HQ/x03-kvmem.txt)`:

| arm | extra env | what it is |
|---|---|---|
| V3 | `DRAFT_METHOD=eagle3 DRAFT=/app/models/_eagle4/v3 LOOKUP=0` | EAGLE-4 drafter loaded as EAGLE-3 (the lookup patch is dflash2-specific: off) |
| DL1 | `LOOKUP=1` | stock DFlash2 drafter, lookup on (the default B; primary comparator, baseline 3.36 tok/step includes lookup hits) |
| DL0 | `LOOKUP=0` | stock DFlash2 drafter, lookup off (like for like with V3) |

`$HQ/x03-run.sh` (`ARM` = V3, DL1, DL0; `V4` and `V3A` are added by bench-21/23):
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; ARM=$1; disklog; stock_on; gate || exit 1
KVM=$(cat $HQ/x03-kvmem.txt); mksweep || { echo "mksweep failed"; exit 1; }
case $ARM in V3) x="DRAFT_METHOD=eagle3 DRAFT=/app/models/_eagle4/v3 LOOKUP=0";; V3A) x="DRAFT_METHOD=eagle3 DRAFT=/app/models/_eagle4/v3-adapted LOOKUP=0";;
  V4) x="DRAFT_METHOD=eagle3 DRAFT=/app/models/_eagle4/v4 LOOKUP=0";; DL1) x="LOOKUP=1";; DL0) x="LOOKUP=0";; esac
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0 DFLASH_TOKENS=7 DFLASH_MAX_LEN=16384 KV_MEM=$KVM $x
boot single || exit 1
begin x03-$ARM-r1
suite SALT=eagle4-base PHASES="snap warm probe sweep spill" CONC=1 LC_CELLS="1300:1 7000:1 12000:1"     # T=0: short_chat, code, and long cells ~1.5k, ~8k, ~13.6k tokens
T7="docker compose exec -T -e TEMP=0.7 -e PYTHONPATH=/app/bench:/harness $SVC /app/venv/bin/python /results/_glue/sweep_x.py"
for s in short_chat code; do $T7 $LABEL-t07 --set $s --conc 1 --reps 3 --warmup 1 --max-tokens 256 --salt eagle4-base --out $OUT_C 2>&1 | tee $OUT_H/t07_$s.log; done
for c in 1300 7000 12000; do $T7 $LABEL-t07-$c --set long_context --ctx $c --conc 1 --reps 3 --warmup 1 --max-tokens 128 --salt eagle4-base --corpus $CORPUS --out $OUT_C 2>&1 | tee $OUT_H/t07_$c.log; done
end
```
Outputs per boot: `sweep_<label>_short_chat.json`, `_code.json`, `-1300/-7000/-12000_long_context.json` (T=0), the same set with `-t07` (T=0.7), each cell carrying `tok_step`, `accept_rate` and `pos_rate` (accepted per draft, per position; read it from the cell records in the JSON), `container.log` with the page-size and pool lines. Floors: only DL1 is the stock B config, but its `DFLASH_MAX_LEN=16384` and `KV_MEM` differ from baseline B: no floor; rule 6 otherwise. Estimated 55 min per boot.
The X-03 verdict is fixed at the end of 17b (V3 against DL1); 17c adds the DL0 (lookup off) columns as `INFO` and cannot change it; X-11 and X-13 read the 17b verdict. Verdict inputs, all as raw ratios: primary r = mean over the `short_chat` and `code` N=1 T=0 cells of tok/step(V3) / tok/step(DL1). Also per long bucket, vs DL0, at T=0.7, the absolute numbers next to 3.75 and 4.40 (E, calc §4), and per-position `pos_rate` for V3 and DL1. WIN iff r >= 1.12; LOSS iff r < 1.00; NEUTRAL iff 1.00 <= r < 1.12. Acceptance of about 1 at any cell: `INVALID`. T=0.7 uses the derived `sweep_x.py` (Harness gap 9): state it.

---

### bench-18a/18b/18c: X-02b′ fitted k-table vs the DYNAMIC_K table at N=3, with a C1 guard (ledger `X-02b′`, directory id `X-02b`, H02′) | 18a 1.8 h (rounds 1-2), 18b 0.9 h (round 3), 18c 1.8 h (rounds 4-5, conditional)

N=3 check only (X-02a: H02′ is 0-2% E at N=4 and N=8, under the +5% bar, so no N=4/8 win framing). Config: `SPEC=dflash2 CTX=fast PREFIX_CACHE=1 DFLASH_TOKENS=7 PTH_FA=0 MAX_SEQS=8 VISION=0` (`DYNAMIC_K` needs `PTH_FA=0`). Arms (the `DYNAMIC_K` value is single-quoted in `.env`):

| arm | `DYNAMIC_K` | N=3 aggregate prediction (E) |
|---|---|---|
| A (comparator, the existing table) | `'[[1,3,7],[4,512,3]]'` | 218 tok/s |
| B (fitted) | `'[[1,1,7],[2,2,5],[3,3,4],[4,512,3]]'` | 228 tok/s (about +5%) |

Arm C (`'[[1,2,7],[3,512,3]]'`, boundary only) is optional in the README and is **not** run. Do not put `k=0` inside a spec boot (untested). Each distinct k adds a CUDA graph set: record the boot time and `memory.used` of arm B next to arm A. Interleaved rounds A B, same `SALT=x02b`, `xauto`: **18a** = `xpart.sh X-02b 18a "x02b.sh 1" "x02b.sh 2"`; **18b** = `xpart.sh X-02b 18b "x02b.sh 3"` (runs only if 18a is done and valid); **18c** = `xpart.sh X-02b 18c "x02b.sh 4" "x02b.sh 5"`, only under the boundary rule below.

`$HQ/x02b.sh`:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; RND=$1; disklog; xauto
for a in A B; do
  gate || exit 1
  [ $a = A ] && dk="DYNAMIC_K='[[1,3,7],[4,512,3]]'" || dk="DYNAMIC_K='[[1,1,7],[2,2,5],[3,3,4],[4,512,3]]'"
  arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 DFLASH_TOKENS=7 PTH_FA=0 MAX_SEQS=8 VISION=0 "$dk"
  boot single || exit 1
  begin x02b-$a-r$RND
  xsuite SALT=x02b PHASES="snap warm probe sweep ladder spill" CONC=1,3 REPS=6 LC_CELLS= LADDER_CELLS="200:1,3" LADDER_OUT=1024
  xtiming; end
done
```
The ladder is `conc_ladder.py --n 1,3 --ctx-tokens 200 --out 1024 --reps 6 --salt x02b` (the README's `--n 1,2,3,4,8` reduced to the two cells that matter). Plus the per-position acceptance capture the review asked for: `pos_rate` (T=0) of the `sweep_*_short_chat.json` and `_code.json` N=1 and N=3 cells (same prompts both arms); T>0 optional and not run. The `[effective-config]` line must show the arm's `DYNAMIC_K`.
Gate (reviewed): N=3 aggregate median(B)/median(A) >= 1.05 over the 3 rounds, and C1 (N=1 decode) within 2% of A. Rule: a median gain from +3% to below +5% needs at least 5 boots per arm: then run part 18c (rounds 4 and 5, 1.8 h; skipped otherwise, with the median over rounds 1-3 as the stated gate number; the verdict is computed over every round that ran, read across 18a-18c). WIN iff both gates; LOSS iff the N=3 median is below A by more than 3% or C1 is more than 2% below; else NEUTRAL (including the 3 to 5% band before the extra rounds, and any N=3 gain below 3%). The N=4/8 cells are not run and not framed.

---

### bench-19: X-09 MTP full head vs 40,960-row slice on Chinese (ledger `X-09`, H09) | about 1.6 h (4 boots S F S F, two script runs), blocked without a prompt file

The harness has no Chinese prompt set (`harness/prompts/` is short_chat, code, long_context). Do not write one. Needs `$HQ/zh_prompts.jsonl` in the harness prompt format (`{"id": ..., "prompt": ...}` per line, Chinese text, supplied by the orchestrator, at least 8 rows); if the file is absent the row is `blocked: no Chinese prompt set` (Harness gap 10). Arms on mode D: `SPEC=mtp CTX=long PREFIX_CACHE=1 PTH_FA=1 VISION=0`: S (default, slice: `MTP_DRAFT_VOCAB` unset) and F (`MTP_DRAFT_VOCAB=0` = full head). Row command: `xpart.sh X-09 19 "x09.sh S F" "x09.sh S F"` (S F S F), `xauto`.

`$HQ/x09.sh`:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; xauto
[ -s $HQ/zh_prompts.jsonl ] || { echo "blocked: no Chinese prompt set"; exit 0; }
mksweep || exit 1; mkdir -p $R/_glue/prompts; cp $HQ/zh_prompts.jsonl $R/_glue/prompts/zh.jsonl; cp $H/prompts/*.jsonl $R/_glue/prompts/
for a in "$@"; do
  gate || exit 1; x=; [ $a = F ] && x=MTP_DRAFT_VOCAB=0
  arm_env SPEC=mtp CTX=long PREFIX_CACHE=1 PTH_FA=1 VISION=0 $x
  boot single || exit 1
  RND=$(( $(xn x09-$a) + 1 )); begin x09-$a-r$RND
  xsuite PHASES="snap warm probe sweep spill" CONC=1 LC_CELLS=                       # English: short_chat and code, N=1 (the English C1 guard)
  docker compose exec -T -e PROMPT_DIR=/results/_glue/prompts -e PYTHONPATH=/app/bench:/harness "$SVC" /app/venv/bin/python /results/_glue/sweep_x.py $LABEL-zh --set zh --conc 1 --reps 3 --warmup 1 --max-tokens 256 --salt eagle4-base --out $OUT_C 2>&1 | tee $OUT_H/sweep_zh.log
  xtiming; end
done
```
Expected (E/M elsewhere): Chinese slice about 1.06 tok/step (35.5 tok/s), full head about 2.2 (80.6 tok/s). The third arm in the index (a coverage-triggered switch) is **not implemented anywhere in the tree**: it cannot be run and its gate (switch >= 2.0 tok/step on Chinese) is not decidable here. Report the measured slice and full-head numbers (the ceiling for any switch). Gate that can be applied: English D C1 decode with the full head within 2% of the slice (median of 2 boots each); Chinese full-head tok/step >= 2.0. Verdict: `INFO` for the Chinese numbers; if both applicable gates pass `NEUTRAL, switch arm not run` (never `WIN`, since the index's win condition names the switch).

---

### bench-20a/20b: X-10 int8 per-token-head KV vs fp8 in mode A: 64-concurrency throughput and drift (ledger `X-10`, H10) | 20a about 1.8 h (F, I, R), 20b about 1.2 h (I, F)

Mode A (batch launcher): `GPU_UTIL=0.91` (A spills at 0.93), `PTH_FA=0` (the batch launcher has no PTH_FA knob, `run_suite` needs it set), no SPEC/CTX. Arms: F = fp8 (the default); I = int8 per-token-head through TRITON via `EXTRA_ARGS` (expanded last after `$KV_ARGS`, so the later `--kv-cache-dtype`/`--attention-backend` win). The pthfa int8 variant needs a launcher edit and is not queued. A third boot, R, is the bf16 reference for drift (`EXTRA_ARGS=--kv-cache-dtype auto`, `MAX_LEN=65536` so the bf16 pool fits; numerics do not depend on `MAX_LEN`). `stock_on` (batch has no variant). Order F I R | I F over two parts: **20a** = `xpart.sh X-10 20a "x10.sh F I R"` (the reference boot R is in 20a so its drift record exists for the compare), **20b** = `xpart.sh X-10 20b "x10.sh I F"` (runs only if 20a is done and valid). The throughput and drift gates read both parts; the verdict is computed at the end of 20b.

Drift tool (does not exist in the repo; write `$R/_glue/x10_drift.py`, stdlib only; method from `docs/long-context.md:462-486`): `record LABEL OUT.json` reads the first 110,000 characters of `$CORPUS` (about 30.6k tokens), POSTs `/v1/completions` with `max_tokens=1`, `temperature=0`, `prompt_logprobs=0`, key from `OPENAI_API_KEY`, model id from `GET /v1/models`, and stores the per-position logprob of the actual prompt token (position 0 is `None`); `compare REF.json A.json [B.json ...]` prints, per file, the number of positions and the mean and p99 of `|logprob - ref logprob|` over all scored positions and over the last 2048. Unverified: the `prompt_logprobs` response shape on this stack (the quality battery uses it): if the position count of the record is not (prompt tokens - 1), the drift gate is `blocked` and the throughput gate still runs. Run it inside the container: `docker compose exec -T "$SVC" bash -c 'source /app/resolve_api_key.sh 2>/dev/null && resolve_client_key; /app/venv/bin/python /results/_glue/x10_drift.py record <label> $OUT_C/drift_<label>.json'`.

`$HQ/x10.sh`:
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; stock_on
for a in "$@"; do
  gate || exit 1; x=; case $a in I) x="EXTRA_ARGS=--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head";; R) x="EXTRA_ARGS=--kv-cache-dtype auto";; esac
  y=; case $a in R) y=MAX_LEN=65536;; esac
  arm_env GPU_UTIL=0.91 PTH_FA=0 PREFIX_CACHE=1 VISION=0 $y "$x"; sed -i '/^$/d' .env
  boot batch || exit 1
  RND=$(( $(xn x10-$a) + 1 )); begin x10-$a-r$RND
  if [ $a != R ]; then
    for i in 1 2; do docker compose exec -T -e OUT=$OUT_C/rb$i "$SVC" bash /app/bench/run_benchmarks.sh batch > $OUT_H/run_benchmarks_$i.log 2>&1; done   # keep the second (the first includes JIT warm-up)
    grep '^ROW' $OUT_H/run_benchmarks_2.log | tee $OUT_H/headline_rows.txt
    [ $a = F ] && floorv "A 64conc e2e" "$(grep -m1 '^ROW 64conc' $OUT_H/run_benchmarks_2.log | sed -E 's/.*e2e=([0-9.]+).*/\1/')" 733 | tee $OUT_H/floor.txt
  fi
  docker compose exec -T "$SVC" bash -c "source /app/resolve_api_key.sh 2>/dev/null && resolve_client_key; /app/venv/bin/python /results/_glue/x10_drift.py record x10-$a-r$RND $OUT_C/drift_x10-$a-r$RND.json" > $OUT_H/drift_record.log 2>&1
  end
done
```
Outputs: `headline_rows.txt` (the `ROW 64conc 128in/512out` line: e2e tok/s, mean TTFT), `drift_*.json`, usual files. Reference (E/M elsewhere): fp8 814.3 e2e (805.6-822.9, mode A prior), drift fp8 0.0360 and int8 TRITON 0.0219 at 32k.
Gates (reviewed): throughput e2e of the second run, median(I) within 3% of median(F) over the 2 boots each; drift mean |delta logprob| of I vs R (all positions, 32k) <= 0.025 nats. WIN iff both; LOSS iff I is more than 3% below F or drift > 0.025; NEUTRAL iff the drift record is `blocked`. Report F's drift next to I's (the fp8 reference 0.0360 is the comparison).

---

### bench-21: X-11 V4 W4A16 drafter vs bf16 V3 (ledger `X-11`, H11) | about 1.5 h (2 boots), gated on X-03

Runs only if the X-03 verdict is `WIN` **and** `/mnt/g/models/_eagle4/v4` exists (a W4A16 quantised copy of the converted drafter, 0.857 GiB; no tool in the repo or the image builds it, and a pip install would break rule 5, so the orchestrator supplies it). Otherwise `skipped: X-03 not WIN` or `blocked: V4 weights not provided`. Same licence rule: evaluation only, under `/mnt/g/models/_eagle4/`. Stock image.
Row command: `xpart.sh X-11 21 "x03-run.sh V4" "x11-pool.sh"`. Boot 1: `x03-run.sh V4` (the same cells, T=0 and T=0.7, same `KV_MEM`; `begin` label `x03-V4-r1`, record it under X-11). Boot 2, pool check at the B defaults (`x11-pool.sh`):
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; disklog; stock_on; gate || exit 1
arm_env SPEC=dflash2 CTX=fast PREFIX_CACHE=1 PTH_FA=0 VISION=0 DRAFT_METHOD=eagle3 DRAFT=/app/models/_eagle4/v4 DFLASH_TOKENS=7 LOOKUP=0
boot single || exit 1
begin x11-pool; suite SMOKE=1 REPS=1 CONC=1 PHASES="snap probe sweep" LC_CELLS=; end
```
Gates (reviewed): tok/step(V4) within -3% of tok/step(V3) at T=0 and at T=0.7 (primary cells as X-03); net KV pool tokens of the `x11-pool` boot (`GPU KV cache size`) >= DFlash2's pool at the same defaults (bench-4's two boots, quote both). WIN iff all three; LOSS iff a tok/step gate fails; NEUTRAL otherwise.

---

### bench-22: X-12 SVD-init rank-128 recall@128 (ledger `X-12`, H12) | CPU, about 0.7 h, no GPU

CPU on the box that has the weights. Needs the bf16 target `lm_head` (loaded as `drafter/train_mtp.py` lines 204-214 do: `lm_head.weight` from the pre-quant shards of `models/Qwen3.8-27B-W4A16-AutoRound`, `[248320, 5120]`) and the captured hidden states `drafter/data/{hidden.npy,tokens.npy,seqs.json}` (74 GB; find them with `ls $REPO/drafter/data /mnt/g/dev/*/drafter/data`). If either is missing, `blocked: <what>`; do not capture or download. No container is up. Write `$HQ/x12_svd.py` and run it in the image's venv (CPU torch, rows streamed from the memmap):
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; xroll X-12 22; disklog
DATA=$(for d in $REPO/drafter/data /mnt/g/dev/*/drafter/data; do [ -s $d/hidden.npy ] && echo $d; done | head -1); [ -n "$DATA" ] || { echo "blocked: no hidden.npy"; exit 1; }
docker run --rm --entrypoint /app/venv/bin/python -v /mnt/g/models:/app/models:ro -v $DATA:/data:ro -v $HQ:/hq $IMG /hq/x12_svd.py 2>&1 | tee $XR/x12.out
```
`$HQ/x12_svd.py` (frequency-weighted SVD via the eigendecomposition of `W^T D W`, P-b of `training-recipe.md`; the unweighted SVD is the control):
```python
import json, os, numpy as np, torch
from safetensors import safe_open
torch.manual_seed(0); R = 128; N = 4096; K = 128
M = "/app/models/Qwen3.8-27B-W4A16-AutoRound/"
idx = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
shard = idx.get("lm_head.weight_packed") or idx.get("lm_head.weight")   # as train_mtp.py:204-215: the bf16 copy is the shard's ".bak"
src = M + shard + ".bak" if shard and os.path.exists(M + shard + ".bak") else (M + shard if "lm_head.weight" in idx else None)
assert src, "no bf16 lm_head (packed and no .bak): blocked"; print("lm_head from", src)
with safe_open(src, "pt") as f: W = f.get_tensor("lm_head.weight").float()
assert W.shape == (248320, 5120), ("unexpected lm_head shape: blocked", W.shape)
toks = np.load("/data/tokens.npy", mmap_mode="r"); hid = np.load("/data/hidden.npy", mmap_mode="r")
cnt = torch.from_numpy(np.bincount(np.asarray(toks), minlength=W.shape[0]).astype(np.float64)).float() + 1.0   # token frequency, add-one
rows = np.sort(np.random.default_rng(0).choice(hid.shape[0], N, replace=False))
H = torch.from_numpy(np.array(hid[rows])).view(torch.bfloat16).float()                                       # same convention as train_mtp.py: lm_head(h) on the stored rows
def recall(Vk):   # Vk [5120, R]: approximate logits = (H Vk) (W Vk)^T, shortlist K, is the full top-1 in it?
    A = W @ Vk; hit = 0
    for i in range(0, N, 256):
        h = H[i:i + 256]; full = (h @ W.T).argmax(1); cand = ((h @ Vk) @ A.T).topk(K, dim=1).indices
        hit += (cand == full[:, None]).any(1).sum().item()
    return hit / N
def top_eig(M): e, v = torch.linalg.eigh(M.double()); return v[:, -R:].float()
print("recall@128 plain SVD (control):", recall(top_eig(W.T @ W)))
print("recall@128 frequency-weighted SVD:", recall(top_eig((W * cnt[:, None]).T @ W)))
print("paper (p. 6): SVD 91.5%, trained 98.2%; rows scored:", N)
```
The weighted variant uses the token frequencies as the weights `D`. Return both recalls next to the paper's numbers (91.5% SVD, 98.2% trained). Verdict `INFO` (the index states no gate; it states a report). Shipping any factor head is licence-gated (D11); nothing is written to the repo.

---

### bench-23a/23b: X-13 body-only domain adaptation pilot, then X-03 again (ledger `X-13`, H13) | 23a pilot about 0.9 h, 23b about 1.5 h; blocked unless tooling exists

Gate (index, with the reviewed ratio bar of `plan.md:51`): runs only if X-03 (the 17b verdict) is **within 10% of break-even** and not a WIN: the 17b same-run ratio r = V3 tok/step / DFlash2 (lookup on) tok/step (T=0, primary cells) is >= 0.90 x 1.12 = 1.008 and r < 1.12. [review: an earlier draft used the absolute bar 0.90 x 3.75 = 3.375; it agrees with the ratio only when DFlash2's same-run tok/step is 3.36.] Else `skipped: X-03 not within 10% of break-even` (or `skipped: X-03 WIN`). The job needs a capture-and-train tool for the EAGLE-4 body (P-c: capture 4 taps, train the body with the head and embeddings frozen). **No such tool is in the tree** (`drafter/train_mtp.py` trains the MTP module; `capture.py` taps one hidden state). Run it only if `docs/research/04-benchmarks/experiments/X-13/README.md` exists when this row comes up, and follow that README; otherwise `blocked: no X-13 tooling in the tree`.

**Pilot budget (23a, at most 2.0 h; hard `timeout -s TERM 5400` on the training command):** capture 250,000 tokens (a quarter of P-c's 1M; E: about 3.5 min, about 10 GB of disk), then one epoch of body training over those tokens (README optimiser settings; head and embeddings frozen). State the tool's tokens-per-step and the resulting step count in `part.md`. E: about 10 min of training, about 0.9 h with model load and the V3 load check. The adapted weights go to `/mnt/g/models/_eagle4/v3-adapted/` (same D11 rule). `xpart.sh X-13 23a "<the README's capture-and-train script>"`.
**23b** = `xpart.sh X-13 23b "x03-run.sh V3A" "x03-run.sh DL1"` (2 boots, about 1.5 h, DL1 fresh as the comparator; runs only if 23a finished and the adapted weights load). Bar: the X-03 ratio gate (r >= 1.12). The verdict is computed at the end of 23b and is labelled a pilot verdict: it covers this budget only.

**Full run: proposal (not a queue row).** P-c as costed in the plan: capture 1M tokens (E: 14 min, 41 GB of disk) and train the body (E: about 40 min), then the 23b evaluation: about 2.5 GPU-h in all, 41 GB of disk (E). Needs the orchestrator's approval after the pilot.

---

### bench-24a/24b/24c/24d: X-14 MTP TTT unroll depth 2 vs 4, pilot budget (ledger `X-14`, H14) | 24a 0.6 h, 24b 0.9 h, 24c 1.5 h, 24d 1.0 h (conditional)

Gate (index): D tok/step >= +3%. Prior low (the depth-2 result was flat, M). Needs `drafter/data/{hidden.npy,tokens.npy,seqs.json}` (74 GB) on the box and `models/Qwen3.8-27B-W4A16-AutoRound` with its `*.bak-mtp` files and `mtp_draft_vocab_ids.pt`; if any is missing: `blocked: <what>`. Compare T4 against T2, both trained and exported the same way (the stock D module is served with a different quantisation path, so it is a reference only). The trainer needs the GPU: no container may be up. Training runs outside compose under a hard `timeout` (the `lib.sh` watchdog stops only compose containers).

**Pilot budget (the queued runs):** `--epochs 0.5` over the captured data (E: about 7.2M tokens per epoch) = about 3.6M tokens = about 110 optimiser steps at the trainer's default 32,768 tokens per step, with `--warmup 10 --eval-every 50` (the held-out eval prints at steps 50 and 100), `--lr 3e-5 --micro-tokens 4096 --seed 0` for both depths (same data order). E: T2 about 15 min and T4 about 33 min of training (about 4k tok/s at depth 2), plus about 15 min of model load and export each. Hard timeouts: 3600 s (T2), 5400 s (T4). If the trainer rejects a fractional `--epochs`, use `--epochs 1 --max-seqs <half the sequences in seqs.json>` and say so. A pilot verdict covers this budget only: a flat 2-vs-4 result at half an epoch does not exclude a gain at full training.

`$HQ/x14-train.sh <2|4>` (one depth per script run):
```bash
source $HQ/lib.sh; source $HQ/xlib.sh; DEP=$1; disklog; gate || exit 1
DATA=$(for d in $REPO/drafter/data /mnt/g/dev/*/drafter/data; do [ -s $d/hidden.npy ] && echo $d; done | head -1)
MD=/mnt/g/models/Qwen3.8-27B-W4A16-AutoRound      # the served model dir (MODELS_DIR=/mnt/g/models); not $REPO/models
[ -n "$DATA" ] && [ -d $MD ] || { echo "blocked: no hidden.npy or model dir"; exit 1; }
W=1,0.5; TO=3600; [ $DEP = 4 ] && { W=1,0.5,0.25,0.125; TO=5400; }   # README halving weights extended to depth 4 (a choice, recorded)
mkdir -p $HQ/x14/t$DEP /mnt/g/models/_x14
timeout -s TERM $TO docker run --rm --gpus "device=$GPU" -v $REPO:/work -v $HQ:/hq -v /mnt/g/models:/m:ro -v $DATA:/data:ro -w /work --entrypoint /app/venv/bin/python $IMG \
  drafter/train_mtp.py --model /m/Qwen3.8-27B-W4A16-AutoRound --data /data --out /hq/x14/t$DEP --depths $DEP --depth-weights $W --epochs 0.5 --warmup 10 --eval-every 50 --lr 3e-5 --micro-tokens 4096 --seed 0 2>&1 | tee $HQ/x14/train_t$DEP.log
docker run --rm -v $REPO:/work -v $HQ:/hq -v /mnt/g/models:/m -w /work --entrypoint /app/venv/bin/python $IMG \
  drafter/export_mtp.py /hq/x14/t$DEP/mtp_bf16.safetensors /m/Qwen3.8-27B-W4A16-AutoRound /m/_x14/t$DEP --bits 8 --head-bits 8 2>&1 | tee $HQ/x14/export_t$DEP.log
```
(Check the served stock D module's bits in its `quantization_config.json` first and use the same `--bits/--head-bits` for both exports; record them. The export source and destination are both under the one `/m` mount because `export_mtp.py` hardlinks (`os.link`), which fails across two bind mounts even on one disk; if it still fails on `os.link`, `blocked: cross-device`. [review: the earlier draft read `$REPO/models` and `$REPO/drafter/data` by default; the models live in `/mnt/g/models`.]) Each run's held-out eval line (the `log.txt` chain-simulation tok/step) goes in `part.md` as the CPU-side expectation.

Parts: **24a** = `xpart.sh X-14 24a "x14-train.sh 2"`; **24b** = `xpart.sh X-14 24b "x14-train.sh 4"` (runs only if 24a is done); **24c** = `xpart.sh X-14 24c "x14-ab.sh T2 T4 S"`; **24d** = `xpart.sh X-14 24d "x14-ab.sh T2 T4"`, only if after 24c the T4/T2 tok/step ratio is between 1.00 and 1.05, else `skipped`. `x14-ab.sh` boots `SPEC=mtp CTX=long PREFIX_CACHE=1 PTH_FA=1 VISION=0 MODEL=/app/models/_x14/t<2|4>` (arm S: no `MODEL`, the stock D module, reference only); `xauto`; `SALT=x14`; `suite PHASES="snap warm probe sweep spill" CONC=1 LC_CELLS="8000:1"`; label `x14-T<d>-r<N>` (`xn` counter); records tok/step and C1 decode. 24c and 24d boot order T2 T4 S, then T2 T4.
Verdict (computed at the end of the last part that ran, pooling the T2 and T4 boots of 24c and 24d): WIN iff median tok/step(T4) / median(T2) >= 1.03 on the `short_chat` and `code` N=1 cells; LOSS iff < 0.99; NEUTRAL otherwise. If a training run does not finish inside its timeout: `aborted`, keep `log.txt`.

**Full training run: proposal (not a queue row, needs the orchestrator's approval after the pilot).** `--epochs 2` (the trainer's default): E: T2 about 1.0 h and T4 about 2.2 h of training, about 0.5 h of load and export, serving boots as 24c (1.5 h): about 5.2 GPU-h. Data: `hidden.npy` (74 GB) is already on the box, no new capture. Disk: MTP checkpoints and two exported module directories only (E: a few GB; measure with `du` on the pilot outputs and report). A 2.2 h T4 run does not fit one 2 h part and the trainer has no resume (`--init` restarts from weights only), so it runs outside the queue as one supervised job.

---

### bench-25a/25b/25c: X-06 conditional-node micro-probe, then MTP k=3 device-side stop (ledger `X-06`, H06) | 25a about 0.3 h, 25b about 1.5 h, 25c about 1.0 h; blocked unless tooling exists

Demoted by the review (the mechanism is a pre-verify confidence stop, not a post-reject skip; at most 1-2%, E). Gate: the plan's reviewed row, D C1 >= +3% with at least 5 interleaved boots (the index text says +2%; the plan table, which carries the phase-3 review, wins). Stage 1 is a CUDA conditional-node probe on the box's driver and WDDM (`torch.cuda.CUDAGraph.begin_capture_to_if_node` or the driver API: capture and replay succeeds or fails, with the exact error); stage 2 patches MTP k=3. **No probe script or patch exists in the tree, and rule 8 forbids writing one into the image.** Run only if `docs/research/04-benchmarks/experiments/X-06/README.md` exists with a probe and a patch, and follow it; otherwise `blocked: no X-06 tooling in the tree`.

- **25a**, stage 1 probe (about 0.3 h, `xpart.sh X-06 25a "<the README's probe script>"`). If the probe fails to capture: stop, verdict `LOSS` for the mechanism with the error text, 25b and 25c `skipped`.
- **25b**, stage 2 boots S P S (stock D, patched D, stock D; `SPEC=mtp CTX=long PTH_FA=1`, about 0.5 h each, 1.5 h including the README's one separately tagged variant build under the rule 2 exception, as in bench-13). Runs only if 25a passed.
- **25c**, boots P S (1.0 h), only if 25b's boots are valid. Together S P S P S: five interleaved boots; the verdict is computed over all five, median(P) / median(S) >= 1.03 on C1 decode (below +5% needs the five boots, which this has).

---

### Return format additions (bench-13 to bench-25)

Use the same line format as above with the verdict words `WIN|LOSS|NEUTRAL|INFO|INVALID` and status words `done|aborted|blocked|skipped`; one line per queue row (per part: bench-15a, bench-15b, ...; the verdict appears on the part that fixed it, earlier parts say `running` and the next-part line from `part.md`) (bench-17: also the pre-gate result, the HF commit sha and the sha256 of `model.safetensors`; bench-14: the classify exit codes). End with the **total GPU hours actually used** (sum of boot wall times plus training, from the job logs) next to the estimate below, and list which gated rows were skipped and why.

### Total box-hour estimate (E; boots, warm-ups and gate waits included; every row at most 2.0 h)

| row (parts) | job | box hours (E) |
|---|---|---|
| bench-13a, 13b, 13c | X-17: build 0.25 (no GPU) + A/B boots + gate (1.25 + 1.6) + D boot 0.7 | 3.6 |
| bench-14a, 14b | X-08 main 1.2, KVarN 1.2 | 2.4 |
| bench-15a, 15b, 15c | X-01b: 3 rounds of 1.4 | 4.2 |
| bench-16a, 16b | X-04′ ABBA: 2 x 1.4 | 2.8 |
| bench-17a, 17b, 17c | X-03: 1.0 (0.5 download, no GPU) + 1.85 + 0.9 | 3.75 |
| bench-18a, 18b | X-02b′: rounds 1-2 (1.8) + round 3 (0.9) | 2.7 |
| | **P1 unconditional subtotal** | **19.45** |
| bench-19 | X-09 4 boots (blocked without a prompt file) | 1.6 |
| bench-20a, 20b | X-10: F I R (1.8) + I F (1.2) | 3.0 |
| bench-22 | X-12 (CPU only, 0.7 h wall) | 0 |
| bench-24a, 24b, 24c | X-14 pilot: T2 0.6 + T4 0.9 + serving 1.5 | 3.0 |
| | **P2 unconditional subtotal** | **7.6** |
| | **Expected total (P1 + P2 unconditional)** | **about 27** (GPU-hours about 26.3: minus 0.25 build and 0.5 download) |
| bench-15d | X-01b GSM8K boots, only if the decode gate is met (expected if H01 holds) | 1.5 |
| bench-16c | X-04′ quality boot, only if the pool and C1 gates are met | 0.7 |
| bench-18c | X-02b′ rounds 4-5, only in the 3 to 5% band | 1.8 |
| bench-24d | X-14 second pair, only if T4/T2 is 1.00 to 1.05 | 1.0 |
| bench-21 | X-11 (gated on X-03 WIN and V4 weights) | 1.5 |
| bench-23a, 23b | X-13 pilot 0.9 + evaluation 1.5 (gated; blocked without tooling) | 2.4 |
| bench-25a, 25b, 25c | X-06 0.3 + 1.5 + 1.0 (gated; blocked without tooling) | 2.8 |
| | **Conditional and gated subtotal** | **11.7** |
| | **Upper bound (every conditional and gated row runs)** | **about 39** |

Proposals, not queued (need the orchestrator's approval): X-14 full training about 5.2 GPU-h (bench-24), X-13 full run about 2.5 GPU-h and 41 GB (bench-23).

## Harness gaps (continued: found while packaging the phase-4 experiments)

9. **`sweep.py` hard-codes `temperature: 0`** (`sweep.py:77`), so no T=0.7 measurement is possible with the harness as is. X-03 and X-11 use `mksweep`'s derived copy `results/_glue/sweep_x.py` (three `sed` substitutions: temperature from `TEMP`, prompt directory from `PROMPT_DIR`, any `--set` name). The harness file is untouched. Proposed fix (not made): a `--temperature` argument.
10. **No Chinese prompt set** (X-09 needs `$HQ/zh_prompts.jsonl` from the orchestrator) and **no coverage-triggered vocabulary switch** exists in the tree (the third X-09 arm cannot run).
11. **No drift tool** (X-10): `x10_drift.py` is glue written for this job; the `prompt_logprobs` response shape on this stack is unverified.
12. **`run_suite.sh` has no per-phase temperature, no `--exact`, and the ladder cells reuse `REPS`** (X-02b′ sets `REPS=6` for the sweep as well as the ladder).
13. **No tool exists in the tree** for X-11's V4 quantisation, X-13's EAGLE-4 body adaptation and X-06's probe and patch: those rows are `blocked` unless the orchestrator adds the tooling and a README under `experiments/`.
14. **X-04's PTH sub-arm and X-10's PTH arm** need a launcher allow-list edit (`start_qwen.sh:299-304`; the batch launcher has no `PTH_FA`): not queued.
15. **`greedy_equal.py compare --near-tie` default is 0.1** (`harness/greedy_equal.py:84`, repeated in `harness/README.md:85`; D18 says 0.01) and it has no replay band: bench-2 and bench-7 pass `--near-tie 0.01` and read the 0.01-0.25 band from the printed margins. (`classify_divergence.py --ulp-band` now defaults to 0.25; bench-14 still passes it explicitly, which is redundant but harmless.)
16. **`record_top2.py` finds its prompts relative to its own path** (`record_top2.py:22`), which is wrong from `/results/_glue/x08/`; bench-14 writes a `sed`-patched copy with `HARNESS = "/harness"`.
