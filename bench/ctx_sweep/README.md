# The context-profile sweep

This sweep finds the best configuration for each `CTX` profile on the RTX 3090 (Windows, WSL2,
Docker Desktop), using the image built from the latest `main`.

- **fast** is ranked on speed up to about 64k tokens.
- **long** is ranked on speed and context together.
- **huge** is ranked on the longest context that boots and finds the needle.

`arms.txt` lists the arms. Each arm is `base.env` plus a few knob settings. `analyze.py` makes every
decision: the plan, each boot's verdict and the report. `sweep.sh` only moves files and processes.
`bench/test_ctx_sweep.py` tests all of it on CPU.

## How it decides

Each arm boots once per pass, with two passes. Pass 2 runs the arms in reverse order, so drift over
the night hits both ends of the list. Each boot runs its profile's cells:

| profile | cells |
|---|---|
| fast | api_smoke, labd_bench at ~48k and ~16k, conc_ladder at N=1,2,4, quality |
| long | api_smoke, labd_bench at ~113k and ~66k, needle at 0.9 x MAX_LEN, quality |
| huge | api_smoke, needle at 0.9 x MAX_LEN, labd_bench at ~113k if MAX_LEN is at least 120,000, quality |

The quality cell is perplexity on three corpora. On pass 1 it also runs 200 GSM8K items.

A boot is clean only when all of these hold:

- it did not spill to system memory;
- its peak Dedicated Usage stayed at or below 23,400 MiB;
- it rose at most 24 MiB in shared memory between the CUDA context and the first request;
- the 3090 held at most 300 MiB before the boot;
- the live vLLM argv and env match the arm's expectations in `arms.txt`;
- the container's GPU 0 is the 3090 at 250 W, and the container ran the image in `run.env`;
- the container did not restart and the engine did not die;
- every cell finished, with GPU samples in each.

If a boot is unclean on memory, the next boot of that arm runs 0.02 lower `GPU_UTIL`. An arm that pins
`KV_MEM` gets 256 MiB less instead. There are at most two steps. If a boot fails with vLLM's
"estimated maximum model length", the next boot uses that length. Pass 2 starts at the fit pass 1
found. An arm that is still unclean after its steps, or that is unclean twice for other reasons in one
pass, is `BROKEN`. The run goes on without it. A setup problem (wrong argv, wrong GPU, wrong image, a
busy card, no counter samples) stops the whole run.

An arm is eligible when it passes all of these bars:

- api_smoke passes 12 of 12 on every clean boot;
- perplexity on each corpus is at most 1.03 x that of `f-mtp4`, today's fast default;
- GSM8K is at least 0.90;
- on long and huge, the needle is RETRIEVED on every clean boot.

The ranking:

- **fast** is the geometric mean of decode at ~16k and ~48k, C1 decode and prefill at ~48k, each
  divided by the best eligible arm's.
- **long** is the geometric mean of speed (decode at ~66k and ~113k) and context (the KV pool in tokens,
  capped at 262,144), each divided by the best eligible arm's.
- **huge** ranks MAX_LEN first, then decode at ~113k, then the KV pool.

The winner's margin is CLEAR when its worst boot beats the runner-up's best boot. For huge, a larger
MAX_LEN is also CLEAR. Otherwise the margin is "within boot spread".

Every rate is on the host's clock. The WSL2 VM's clock runs at a slightly different rate from the
host's, so each cell records both clocks at its start and end, and `analyze.py` corrects for the ratio.
The bench clients time durations with `time.perf_counter()`, so the VM wall clock's steps (about 2.3 s
back every 28 s) do not reach them.

## Runbook for the agent on the 3090 PC

You are running this sweep on this PC. It takes about 15 to 20 hours. A script decides every verdict.
You run the steps, follow the rules below, and report what the script printed.

### Machine facts

- Repo: `G:\dev\HyperQwen`. Models: `G:\models`, mounted at `/app/models`.
- Run the sweep from **Git Bash** at the repo root. `docker compose up` and `run` go through
  PowerShell, because from Git Bash they fail on the `G:/models` bind mount. `sweep.sh` does that itself.
- The 3090 is headless and is UUID `GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84`. A 3080 Ti drives the
  display. Never run on the 3080 Ti.
- The 3090's power limit should read 250 W. Record it. Do not change it.
- Never run `prepare` or `docker compose build`.
- Pull exactly one image, the `sha-` tag of the `main` commit you test, and only if it is not already
  present. Never pull, retag or remove `latest` or any other image.
- On WSL2 an over-committed GPU does not fail. It spills to system memory and runs 3-6x slower. That
  is why every boot is checked for a clean fit.

### Rules

- Do not change git config. `git config user.name` must already print `TyroneNel`. If not, stop and report.
- No Co-Authored-By, Signed-off-by, tool or agent names, or generated-by banners in commits.
- Never touch `main` or any upstream remote. Never force-push. Never amend a pushed commit.
- Do not comment on, review or edit any GitHub PR or issue. Your results go in your final message only.
- Do not edit anything in `bench/ctx_sweep/`, the launcher or the compose file. If something blocks,
  stop and report it instead of working around it. The one fix you may make is in "When the run stops".
- While the sweep runs, send nothing to port 18020 yourself and run nothing else on the 3090. Only
  one `sweep.sh` runs at a time.
- `git add` names every path. Untracked owner files (`.env`, `.env.*` at the repo root,
  `docker-compose.override.local.yml`) stay as they are. The exception is `.env`, which the sweep
  overwrites and you restore at the end.
- Leave the machine as you found it. No containers up, no sampler processes, the original branch
  checked out, no stray files in the repo.

### 0. Preflight (PowerShell, at `G:\dev\HyperQwen`)

Record the output of each in your notes. Stop and report if any check fails.

```powershell
Set-Location G:\dev\HyperQwen
git remote -v            # the remote for https://github.com/TyroneNel/qwen38-27b-rtx3090.git is <fork> below
git status -sb           # note the current branch (<orig>); no modified tracked files
git config user.name
$env:COMPOSE_FILE        # must be empty
Test-Path docker-compose.override.yml    # must be False (sweep.sh init creates it)
Test-Path .env.ctx-sweep-backup          # must be False (sweep.sh init creates it)
nvidia-smi --query-gpu=index,name,uuid,power.limit,memory.total,memory.used,display_active --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv    # nothing on the 3090
docker ps -a --format '{{.Names}} {{.Status}}'    # no hyperqwen container running
docker volume inspect hyperqwen_qwen-cache --format '{{.Name}} {{.CreatedAt}}'    # must exist
docker system df
Get-ChildItem G:\models | Select-Object -ExpandProperty Name
    # must include Qwen3.8-27B-W4A16-AutoRound-fast and Qwen3.8-27B-DFlash2-W4A16
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
    # the AC setting must be 0 (never sleep); if not, stop and report, do not change it
Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' `
  -SampleInterval 5 -MaxSamples 3 | ForEach-Object { $_.CounterSamples } |
  ForEach-Object { "{0} {1:N1} MiB" -f $_.Path, ($_.CookedValue/1MB) }
```

If a hyperqwen container is up and you did not start it, stop and report.

**The commit under test.** It is the head of upstream `main` when you start:

```powershell
git ls-remote https://github.com/syv-ai/HyperQwen.git refs/heads/main    # record the full SHA as <main>
```

The image is `ghcr.io/syv-ai/hyperqwen:sha-<first 7 characters of main>`, written `<img>` below.
If `docker images <img>` lists nothing, pull it. If the tag does not exist yet, CI is still building
it. Wait up to an hour and try again, then stop and report.

```powershell
docker pull <img>
docker image inspect <img> `
  --format '{{.Id}} {{json .RepoDigests}} {{.Created}} {{index .Config.Labels "org.opencontainers.image.revision"}}'
```

The revision label must be `<main>`. If it is not, stop.

**Find the 3090's adapter LUID.** The counter instances are named like
`luid_0x00000000_0x0001144f_phys_0`, and the LUID changes across reboots. The headless 3090 idles near
0 MiB Dedicated Usage while the display card holds hundreds of MiB or more. If that does not single out
one instance, allocate 2 GiB on the 3090 and see which instance's Dedicated Usage rises by about 2 GiB:

```powershell
docker run --rm --gpus "device=GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84" --entrypoint /app/venv/bin/python <img> `
  -c "import torch,time; x=torch.empty(2<<30,dtype=torch.uint8,device='cuda'); time.sleep(30)"
# sample Get-Counter in a second PowerShell window while it sleeps
```

Write the LUID down as `<luid>`. The spill guard and the fit checks are blind with the wrong one.

### 1. Set up (Git Bash, at the repo root)

```bash
cd /g/dev/HyperQwen
git fetch <fork> ctx-profile-sweep
git rev-parse <fork>/ctx-profile-sweep    # must be the SHA your prompt gives; if not, stop
git fetch https://github.com/syv-ai/HyperQwen.git main
git switch -c ctx-sweep-results <fork>/ctx-profile-sweep    # if the branch exists, stop and report
git merge --no-edit <main>    # "Already up to date." if main has not moved; any conflict means stop
git log -1 --format=%H        # record it
```

Pick the run directory once, with today's date, and use that literal path for `D` in every later
command, even in a fresh shell:

```bash
D=bench/results/$(date +%F)-ctx-sweep; echo $D
export MSYS_NO_PATHCONV=1
bash bench/ctx_sweep/sweep.sh init $D <img> <luid>
```

`init` writes `$D/run.env` and `docker-compose.override.yml`, and copies an existing `.env` to
`.env.ctx-sweep-backup`. It prints both files. The override sets the image, `pull_policy: never` and
the read-only mount of `bench/ctx_sweep` at `/sweep`. Check it from PowerShell:
`docker compose --profile single config` must show `<img>`, `pull_policy: never` and the `/sweep` mount,
beside the models and `qwen-cache` mounts.

### 2. Data on the qwen-cache volume (Git Bash)

The bench tools read their corpora from `~/bench` and the quality data from `/cache/quality-data`.
Inside the container `~` is `/cache`, the `hyperqwen_qwen-cache` volume. Earlier runs built all of this
on this PC, so it should all be there. This step runs in throwaway containers with no GPU.

```bash
export MSYS_NO_PATHCONV=1; IMG=<img>
docker run --rm -v hyperqwen_qwen-cache:/cache --entrypoint bash $IMG -c \
  'ls -la /cache/bench; sha256sum /cache/bench/labd_corpus*.txt; find /cache/quality-data -name "*.parquet" | sort'
docker run --rm -v hyperqwen_qwen-cache:/cache --entrypoint /app/venv/bin/python $IMG -c \
  'b = open("/cache/bench/labd_corpus.txt").read(); l = open("/cache/bench/labd_corpus_long.txt").read(); print(len(b), len(b.split()), len(l), l[:len(b)] == b)'
    # the first above 200,000 chars and 20,000 words, the long one above 400,000 chars, and True
    # (a corpus built where the docs were not found is 200,000 newlines: 0 words)
```

These three parquet files must be listed:

- `/cache/quality-data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet`
- `/cache/quality-data/gsm8k/main/test-00000-of-00001.parquet`
- `/cache/quality-data/fineweb2/data/dan_Latn/test/000_00000.parquet`

Never rebuild or replace a corpus that exists. If a corpus exists but fails its length check, stop and
report. If something is missing, build only that piece:

```bash
# labd_corpus.txt missing: the bench writes it from the repo docs, then fails on the dead port (expected)
docker run --rm -v hyperqwen_qwen-cache:/cache -e HOME=/cache -e VLLM_API=http://127.0.0.1:9 \
  --entrypoint bash $IMG -c 'cd /app && [ ! -e /cache/bench/labd_corpus.txt ] && ln -sfn /app /cache/qwen-serving \
  && { venv/bin/python bench/labd_bench.py corpus-init --ctx 1000 --tasks qa; rm -f /cache/qwen-serving; }; ls -la /cache/bench'
# labd_corpus_long.txt missing
docker run --rm -v hyperqwen_qwen-cache:/cache -e HOME=/cache --entrypoint bash $IMG -c \
  'cd /app && { [ -e /cache/bench/labd_corpus_long.txt ] || venv/bin/python bench/make_long_corpus.py; }'
# a parquet set missing
docker run --rm -v hyperqwen_qwen-cache:/cache -e HOME=/cache --entrypoint bash $IMG -c '
  cd /cache && H=/app/venv/bin/hf
  [ -d quality-data/wikitext ] || $H download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir quality-data/wikitext
  [ -d quality-data/gsm8k ] || $H download openai/gsm8k --repo-type dataset --include "main/test-*" --local-dir quality-data/gsm8k
  [ -d quality-data/fineweb2 ] || $H download HuggingFaceFW/fineweb-2 --repo-type dataset --include "data/dan_Latn/test/000_00000.parquet" --local-dir quality-data/fineweb2'
```

Then run the two checks again. Record the corpora's sha256 and the parquet list.

### 3. Dry run (Git Bash)

```bash
export MSYS_NO_PATHCONV=1; D=bench/results/<date>-ctx-sweep
bash bench/ctx_sweep/sweep.sh dryrun $D; echo "exit=$?"
```

This launches every arm through the image's real entrypoint, with a stub `vllm` that prints its argv
and env instead of serving. It loads no model and takes a few minutes. It must end with
`DRYRUN_OK` and `exit=0`. Each `DRY <arm> FAIL` line names what the launcher did differently from the
arm's expectations. That means `main` changed a knob since the sweep was written. Stop and report the
whole output. Do not edit `arms.txt`.

### 4. The run (Git Bash)

```bash
export MSYS_NO_PATHCONV=1; D=bench/results/<date>-ctx-sweep
nohup bash -c 'bash bench/ctx_sweep/sweep.sh run "$0"; echo "RUN_EXIT $?"' $D > $D/run.log 2>&1 &
```

Poll it every 20 to 30 minutes with `tail -n 3 $D/sweep.log`. A boot takes 20 to 50 minutes. The run is
over when `$D/run.log` ends with `RUN_EXIT <n>`. The lines you will see:

- `[<label>] boot: ...` and `[<label>] cell <name>: ...` while a boot runs. Labels are
  `<arm>-p<pass>-<n>`.
- `BOOT_CLEAN <label>`, `BOOT_UNCLEAN mem|other <label>: <reasons>` or `RETRY_MAX_LEN <n> <label>: ...`
  after each boot. The plan acts on these by itself.
- `DONE <arm> pass <p>` or `BROKEN <arm> pass <p>: <why>` when an arm is through a pass.

If the PC or Docker Desktop restarted, or the run process died without `RUN_EXIT`, start Docker Desktop
if it is down and wait until `docker version` works. After a PC restart, find the LUID again as in
step 0 and change the `LUID=` line in `$D/run.env`, because it changes across reboots. Then run the
same `nohup` command again. The run resumes. It renames the boot it was in to `<label>-aborted` and boots that arm again. Note every such
restart for the report.

### When the run stops

`RUN_EXIT 0` means every arm is through both passes and `$D/summary.md` is written. Any other exit
means the run stopped. The line before it says why.

- `plan: STOP <label>: BOOT_SETUP <label>: no 3090 counter sample above 20000 MiB (wrong LUID?)`
  means the counters were read for the wrong adapter. This is the one fix you may make. Find the LUID
  again as in step 0, change the `LUID=` line in `$D/run.env`, rename `$D/<label>` to
  `$D/<label>-aborted`, and run the `nohup` command again.
- Any other `STOP`, `analyze.py boot printed no verdict`, or a Python traceback means stop and report.
  Include the stop line, `$D/<label>/analysis.txt` and the last 80 lines of `$D/<label>/boot.log`.
  Then run `bash bench/ctx_sweep/sweep.sh report $D` and include `$D/summary.md` too.

A `BROKEN` arm is a result, not a stop. Report it as it stands. Do not rerun it or change its knobs.

### 5. Record and push

Commit and push the results only. `bench/results/` is in `.gitignore`, so the add needs `-f`.

```bash
D=bench/results/<date>-ctx-sweep
git add -f $D
git status --short    # only $D is staged
git commit -m "bench: context-profile sweep on the 3090, image sha-<first 7 of main>"
git push -u <fork> ctx-sweep-results
```

This branch is temporary and will be deleted. Nothing may depend on it later, so your final message
must carry the results itself.

### 6. Clean up

```bash
export MSYS_NO_PATHCONV=1
docker ps --format '{{.Names}}' | grep hyperqwen && echo "STILL UP"    # must print nothing
docker run --rm -v hyperqwen_qwen-cache:/cache --entrypoint bash <img> -c \
  'rm -fv /cache/bench/results/labd_sw-*.json /cache/quality-data/result_sw-*.json'    # this run's result files only
rm -f docker-compose.override.yml
[ -f .env.ctx-sweep-backup ] && mv -f .env.ctx-sweep-backup .env || rm -f .env    # the owner's .env exactly, or none
git switch <orig>
git status -sb    # as in preflight
```

From PowerShell, confirm no sampler is still running.
`Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object CommandLine -match 'sampler.ps1'`
must print nothing.

Keep the image and the cache volume's corpora and quality data.

### Final message

The results branch will be deleted, so do not point at it for anything. Put everything in the message:

- the full `$D/summary.md`, verbatim, including all of its tables;
- `<main>`, the image tag, ID, RepoDigests and revision label, and the `ctx-sweep-results` head before
  the results commit;
- the 3090's power limit, `<luid>`, the corpora's sha256 and the parquet list;
- the run's start and end time, and every restart or LUID fix;
- the pushed results commit SHA;
- anything that did not go as written above.
