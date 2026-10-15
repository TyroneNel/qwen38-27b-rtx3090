# GPU agent brief, batch 4 — the seat curve becomes a policy

Paste the whole of this file as the task for the agent on the Windows box. The arms and the decision rules live in
[decode-speed-batch4.md](decode-speed-batch4.md), which you must read before Step 0. Batches 1-3 are in
[`bench/results/`](../bench/results) (`2026-10-13-SUMMARY.md`, `2026-10-14-batch2-SUMMARY.md`,
`2026-10-14-batch3-SUMMARY.md`).

Read in this order: the G7 section of [decode-speed-landscape.md](decode-speed-landscape.md) (what is already
settled), then [decode-speed-batch4.md](decode-speed-batch4.md) (the arms), then this file.

## Role

Same machine, same rules. Batch 3 settled what caps residency: the **configured** verify block, 7 → 13 → 16 seats
as `DFLASH_TOKENS` goes 7 → 3 → 1, with C1 T=0 paying 112.1 → 94.1 → 64.7 tok/s. Two things follow and this batch
is them: **which k each profile should ship** at a given concurrency (Q2, the deliverable), and **a way to read
seats without inferring them** (Q1). Q3 and Q4 close the two loose ends batch 3 left open. Commit results and
reports to `cline/gf59cdmh` and push.

## The hard rules

1. **This batch measures and decides; it does not ship.** No default flips, no engine patches, no edits outside
   `bench/results/` and `docs/`.
2. **Speed is everything.** No soak, no `verify.sh`, no acceptance battery. Q1's second probe and Q2's ladders are
   the only long steps.
3. **No assumptions.** Every fact is a command's output, quoted. "Not verified" is a complete answer.
4. **No PRs, issues, comments or reviews.** No `gh pr`, `gh issue`, no `gh api` write call.
5. **No co-authored commits, no watermarks, and no invented identity.** One author — the repo's configured
   identity — with no `Co-Authored-By`, no `Signed-off-by`, no tool or agent name, no banners. If `git config
   user.name` is unset on this checkout, set the repo-local identity (`TyroneNel
   <71038642+TyroneNel@users.noreply.github.com>`) rather than committing as a placeholder: a non-routable
   address leaves the commits unattributed and an agent name in the author field is the same watermark this rule
   forbids elsewhere.
6. **Only `cline/gf59cdmh`.** Never `main`, never the `upstream` remote, never force-push, never amend a pushed
   commit.
7. **Leave the machine as you found it.** Containers down, no stray `docker run`, no files outside the repo.

## What batches 1-3 established — verify, do not re-derive

- The 3090 is headless (the 3080 Ti holds the display); pin it by UUID in `CUDA_VISIBLE_DEVICES` and
  `NVIDIA_VISIBLE_DEVICES`. `power.limit` reads 250.00 W. Models under `G:/models` are prepared — never run
  `prepare`. `docker compose up` must run from PowerShell, not Git Bash.
- **`MAX_SEQS` hides the seat effect.** Batch 3's probe read "Conclusion B" purely because `MAX_SEQS=8` admits
  eight requests and k=7 already holds seven. Every seat question in this batch runs with the cap lifted
  (`MAX_SEQS=16`), and every report states the cap it ran under.
- Seats are set by the **configured** `DFLASH_TOKENS`, not by a per-step schedule (batch 2's dynamic-K arm booted
  with `num_spec_tokens=7` and bought none). The per-request share of the pool scales as (k+2): 14.2% / 7.45% /
  4.37% at k=7/3/1.
- The launcher's `RESIDENT` constant (`104988089`) is **16-23% low** — it prints about 5 seats where 7 fit.
- The spill oracle is the counter, not free VRAM: idle Shared Usage 0.25 MiB, loaded-engine floor 84-92 MiB.
  Perplexity spills this box badly (up to 1,002 MiB — gotcha 60). Reuse
  `bench/results/2026-10-14-batch3-tools/spillwatch.sh`, and **take its floor from the samples between the
  engine's KV-cache line and the first request** — batch 3's version sampled the baseline after the rise and
  missed a ~16 MiB spill on three uncapped boots.
- Noise floors: C1 T=0 within a boot 0.0-0.1%, cross-boot up to 3.2%; teacher-forced tokens/step is noise-free;
  free-run decode ~10%; preempted ladder rows are tail rows, not steady state; the C8 cohort spread is 0.1% on the
  int8 pool.
- The image to reuse is `hyperqwen:exp-f1a033e` unless Step 1 says the branch has changed code.

## Step 0 — preflight (10 minutes)

```powershell
cd G:\dev\HyperQwen
git fetch origin; git status -sb; git log --oneline -1
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv
nvidia-smi --query-gpu=uuid,display_active --format=csv
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | Select-String hyperqwen
Get-ChildItem G:\models | Select-Object Name
Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 10 -MaxSamples 6
```

Into `bench/results/<date>-batch4-preflight/preflight.txt`. Dirty tracked tree: stop and report the diff.
Untracked owner files (`.env.*`, `docker-compose.override.local.yml`) stay untouched; name every path in
`git add`.

## Step 1 — image reuse check (5 minutes)

```powershell
git diff --stat <image-commit>..HEAD -- . ':(exclude)docs' ':(exclude)bench/results' ':(exclude).dockerignore'
```

Nothing printed means reuse `hyperqwen:exp-f1a033e`; record its digest, do not rebuild. The `.dockerignore` change
on this branch affects only what a *new* build would exclude, so it does not invalidate the existing image.

## Step 2 — Q1, can seats be read directly? (about 30 minutes)

1. **Metrics.** During a held load, fetch `/metrics` and look for anything per-group or state-shaped: block
   counts, `kv_cache_max_concurrency`, mamba/state gauges, per-group `num_gpu_blocks`. Report exact metric names
   and values, or state plainly that none exist. A gauge here becomes the primary metric for every future seat
   question.
2. **Is the share constant across context depth?** Re-run the seat probe at k=7 with ~4k prompts (the batch-3
   shape) and with **~32k prompts**, `MAX_SEQS=16`, same `KV_MEM`, and compare the per-request share. If it is
   context-independent, the launcher constant can be corrected in one line; if it grows with depth, a constant is
   wrong in principle and the print should be removed or made a function.

## Step 3 — Q2, the policy table (about 60 minutes) — this is the deliverable

Four boots, **identical `KV_MEM` and `MAX_SEQS=16`**: `DFLASH_TOKENS=7`, `3`, `1`, and `SPEC=off`. Per arm:
`bench/conc_ladder.py --n 1,2,4,8,16 --json`, the held 8-request `Running:` probe, C1 T=0, preemptions, `kv%`,
`ms/pass`.

Then produce the table — aggregate decode tok/s and preemptions per arm per C, with C1 T=0 as the cost column —
and answer: for each C in {1, 2, 4, 8, 16}, which arm wins, and where the crossover sits between the speculating
arms and `SPEC=off`. Say which rows are steady state and which preempted. Do not set a default; the table is the
recommendation.

## Step 4 — Q3 and Q4 (about 40 minutes)

- **Q3, ReplaySSM where its docstring allows it:** A/B in a non-speculative profile, one boot each,
  `--use-replayssm` off vs on. Read `ms/pass` at C8 and C16, aggregate decode, seats, and whether the boot log
  shows it engaged. Pass = 3%+ aggregate at C8 or C16, no quality change.
- **Q4, the last open cell and the guard:** re-run k=1 at `MAX_SEQS=24` with `KV_MEM` pre-cut (batch 2's trick) so
  the margin holds — batch 3 spilled 32 MiB there and could not measure it. And fix the guard's baseline: take
  the floor between the engine's KV-cache line and the first request, with two consecutive samples required before
  calling a spill. Report whether you fixed it or worked around it.

## Step 5 — record and push (after Q1+Q2, and again after Q3+Q4)

One directory per arm under `bench/results/<YYYY-MM-DD>-<arm-id>/`: `.env` (or `.env.<label>` — say which), `cmd.txt`,
`image.txt` (tag, digest, commit, boot class), `mem.txt` (free VRAM after boot, the Shared Usage trace, and the
floor you took it against), `raw-*.log` verbatim including the `Running:` samples and any `/metrics` excerpts,
`report.md` with the arms side by side, the metric named, the raw numbers, the delta, the verdict, and one line on
what was verified versus assumed. Then `bench/results/<date>-batch4-SUMMARY.md`, whose **first section is the
policy table** and whose first line names the recommendation in one sentence.

```powershell
git add bench/results/<arm-dir>          # name paths explicitly; never git add -A
git commit -m "exp: batch4 <question> <arm> (250W, <date>)"
git push origin cline/gf59cdmh
```

Push after Q1+Q2 and again after Q3+Q4. An unpushed result did not happen.

## Step 6 — report back

Lead with the policy table and the one-sentence recommendation, then Q1's answer (gauge or no gauge; constant or
not), then Q3/Q4. If anything contradicts batch 3, the landscape page or the launcher's own comment, say so with
the number — a corrected model is worth more than a confirmed one.

## Failure playbook

| symptom | do this |
|---|---|
| a ladder row preempts | mark it a tail row; it cannot settle a crossover alone |
| Shared Usage above the floor | first check the guard's baseline is not taken after the rise (batch 3's blind spot); if real, cut `KV_MEM` identically on all arms and record it |
| a spill on an uncapped boot | pre-cut `KV_MEM` for that arm and say so; do not report a spilled timing |
| `/metrics` has no state gauge | say so — that is the Q1 answer, and it makes the counter a candidate patch |
| a knob is absent from the engine's args line | the arm's result is "knob not applied"; quote the args line and continue |
| `RuntimeError: UVA is not available` | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` missing from `.env` |
| bash errors with `\r` | CRLF tree: `git config core.autocrlf false`, re-checkout |
| an arm past twice its budget | record what you have, mark "incomplete — time-boxed", move on |

## Do not

No default flips, no engine patches, no new image, no EXL3/TurboQuant/SGLang/FlashInfer work, no bf16-state or
checkpoint-flag re-runs (batch 3 closed both), no `prepare`, no soak, no `verify.sh`, no quality claim from n=50,
no PR, no issue, no comment, no watermark, no co-author trailer.

