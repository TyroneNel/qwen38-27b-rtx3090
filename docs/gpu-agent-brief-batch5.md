# GPU agent brief, batch 5 — the speed levers that need a GPU

Paste the whole of this file as the task for the agent on the Windows box. The arms and the pass rules live in
[decode-speed-batch5.md](decode-speed-batch5.md), which you must read before Step 0. Batches 1-4 are in
[`bench/results/`](../bench/results).

Read in this order: the **S — speed levers** section of [decode-speed-findings.md](decode-speed-findings.md) (what
is already known, and its **R — refuted** table so you do not re-run anything in it), then
[decode-speed-batch5.md](decode-speed-batch5.md), then this file.

## Role

Same machine, same rules. Batches 1-4 measured what the stack does; batch 5 is the residue that cannot be settled
by reading — four questions: the unadvertised Mamba knobs (never measured), whether the state reservation could
follow the per-step k (the one lever with no ceiling), the `CG=40` mechanism if batch 4 skipped it, and one
precision number for the profile about to be recommended. Commit results and reports to `cline/gf59cdmh` and push.

## The hard rules

1. **This batch measures; it does not ship.** No default flips, no engine patches, no edits outside
   `bench/results/` and `docs/`.
2. **Speed is everything**, with one exception: Q4's IFBench run and the n=50 smoke on a winning Q1 arm are the
   precision half and they are bounded.
3. **No assumptions.** Every fact is a command's output, quoted. "Not verified" is a complete answer.
4. **No PRs, issues, comments or reviews.** No `gh pr`, `gh issue`, no `gh api` write call.
5. **One author, no watermarks.** The repo's configured identity — `user.name` is set to `TyroneNel` on this
   checkout; if it is unset, set the repo-local identity rather than committing as a placeholder. No
   `Co-Authored-By`, no `Signed-off-by`, no tool or agent name, no banners.
6. **Only `cline/gf59cdmh`.** Never `main`, never the `upstream` remote, never force-push, never amend a pushed
   commit.
7. **Leave the machine as you found it.** Containers down, no stray `docker run`, no files outside the repo.

## What batches 1-4 established — verify, do not re-derive

- The 3090 is headless (the 3080 Ti holds the display); pin it by UUID in `CUDA_VISIBLE_DEVICES` and
  `NVIDIA_VISIBLE_DEVICES`. `power.limit` reads 250.00 W. Models under `G:/models` are prepared — never run
  `prepare`. `docker compose up` must run from PowerShell, not Git Bash.
- **`MAX_SEQS` hides the seat effect**: 8 admits eight requests and k=7 already holds seven. Every concurrency
  question runs at `MAX_SEQS=16` and states the cap it ran under.
- Seats follow the **configured** k: k=7 → 7 seats at 112.1 tok/s C1, k=3 → 13 at 94.1, k=1 → 16 at 64.7. The
  per-request pool share is 14.2 / 7.45 / 4.37%, and the launcher's `RESIDENT` constant is 16-23% low.
- The band-2 profile (`DFLASH_TOKENS=3` + `EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype
  int8_per_token_head'`, `KV_MEM=5583457484`, `MAX_SEQS=16`) is Q1's baseline: pool 116,440 tokens, preemptions 0,
  N=8 aggregate 350.1. Its quality gate so far is GSM8K n=200 and perplexity; Q4 adds IFBench.
- The spill oracle is the counter, not free VRAM: idle Shared Usage 0.25 MiB, loaded-engine floor 84-92 MiB.
  Perplexity spills this box badly (up to 1,002 MiB). Reuse `bench/results/2026-10-14-batch3-tools/spillwatch.sh`
  and **take its floor between the engine's KV-cache line and the first request** — the batch-3 version sampled
  after the rise and missed a spill.
- Noise floors: C1 T=0 within a boot 0.0-0.1%, cross-boot up to 3.2%; teacher-forced tokens/step is noise-free;
  free-run decode ~10%; preempted ladder rows are tail rows, not steady state.
- Instruments to reuse, not rebuild: `running_probe.py`, `workload_accept.py`, `memsum.sh`/`sampler.ps1`/`arm.sh`
## Step 0 — preflight (10 minutes)

```powershell
cd G:\dev\HyperQwen
git fetch origin; git status -sb; git log --oneline -1
git config user.name; git config user.email          # must be TyroneNel / the noreply address
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv
nvidia-smi --query-gpu=uuid,display_active --format=csv
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | Select-String hyperqwen
Get-ChildItem G:\models | Select-Object Name
Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 10 -MaxSamples 6
```

Into `bench/results/<date>-batch5-preflight/preflight.txt`. Dirty tracked tree: stop and report the diff.
Untracked owner files (`.env.*`, `docker-compose.override.local.yml`) stay untouched; name every path in `git add`.

## Step 1 — image reuse check (5 minutes)

```powershell
git diff --stat <image-commit>..HEAD -- . ':(exclude)docs' ':(exclude)bench/results'
```

Nothing printed means reuse the batch-3 image; record its digest, do not rebuild. Anything printed means rebuild
once, tag `hyperqwen:exp-<short-sha>`, record the digest.

## Step 2 — Q1, the Mamba knobs (about 50 minutes)

Baseline is the **band-2 profile**. Four boots, one variable each: nothing, `--mamba-ssu-algorithm horizontal`,
`--mamba-ssu-algorithm vertical`, `--mamba-backend flashinfer` (each appended to `EXTRA_ARGS`).

Per arm: C1 T=0 (second run of the boot), `bench/conc_ladder.py --n 1,4,8,16 --json`, the held-load `Running:`
probe, `ms/pass`, and **the kernel-selection lines from the boot log** (`GDN decode kernel: triton|cuda`, the mamba
warmup lines). A knob that changes no kernel line did not apply — quote the lines and say so. Order the arms so the
baseline runs first and last (cheap A-B-A), and run `quality_battery.py --gsm-only --gsm-n 50` on any winner before
calling it one.

## Step 3 — Q2, the reconnaissance (no GPU)

Read the image's source for where the recurrent-state pool is sized from `num_speculative_tokens` — start at
`vllm/v1/core/kv_cache_utils.py`, `kv_cache_planning.py` and the `MambaManager` in
`single_type_kv_cache_manager.py` — and answer the three questions in the batch doc with **quoted file, line and
expression**. Then state the verdict: attemptable, attemptable with X, or needs a redesign because Y. Do not write
the patch; this is a feasibility answer.

## Step 4 — Q3 and Q4 (about 60 minutes)

- **Q3** (`CG=40` mechanism) only if batch 4 skipped it: two boots, `CG=64` and `CG=40`, teacher-forced with a
  **shared frozen target**; report tokens/step and accept rate on the same sequence.
- **Q4** (precision for the band-2 profile): IFBench prompt-level strict, 300 prompts, band-2 profile against the
  shipped one, **4 client threads** (batch 2 preempted 1,017 times by over-subscribing). Install IFBench per boot
  the way batch 2 did if it is not in the image, and record the commit you pinned.

## Step 5 — record and push (after Q1, and again after Q3+Q4)

One directory per arm under `bench/results/<YYYY-MM-DD>-<arm-id>/`: `.env` (with `KV_MEM`, `MAX_LEN`,
`DFLASH_TOKENS`, `MAX_SEQS` and `EXTRA_ARGS`), `cmd.txt`, `image.txt` (tag, digest, commit, boot class), `mem.txt`
(free VRAM after boot, the Shared Usage trace, and the floor you took it against), `raw-*.log` verbatim including
the `Running:` samples and the kernel lines, `report.md` with the arms side by side, the metric named, the raw
numbers, the delta, the verdict, and one line on what was verified versus assumed. Q2's answer goes in
`bench/results/<date>-batch5-S5-recon/` as prose with the quoted code. Then
`bench/results/<date>-batch5-SUMMARY.md`, whose first line is the one-sentence answer to Q1.

```powershell
git add bench/results/<arm-dir>          # name paths explicitly; never git add -A
git commit -m "exp: batch5 <question> <arm> (250W, <date>)"
git push origin cline/gf59cdmh
```

Push after Q1 and again after Q3+Q4. An unpushed result did not happen.

## Step 6 — report back

Q1's answer first (did any knob engage *and* help — the two are separate answers), then Q2's verdict with its
quoted code, then Q3/Q4. If anything contradicts [decode-speed-findings.md](decode-speed-findings.md) — including
its own **R — refuted** table — say so with the number: a corrected model is worth more than a confirmed one.

## Failure playbook

| symptom | do this |
|---|---|
| a knob does not appear in the engine's args line | the arm's result is "knob not applied"; quote the args line and continue |
| a knob appears but no kernel line changes | same verdict — it applied to nothing; that is a finding |
| `--mamba-backend flashinfer` refuses to boot | record the refusal verbatim; it is the answer |
| a ladder row preempts | mark it a tail row; it cannot settle a crossover alone |
| Shared Usage above the floor | check the guard's baseline is not taken after the rise; if real, cut `KV_MEM` identically across arms and record it |
| `RuntimeError: UVA is not available` | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` missing from `.env` |
| bash errors with `\r` | CRLF tree: `git config core.autocrlf false`, re-checkout |
| an arm past twice its budget | record what you have, mark "incomplete — time-boxed", move on |

## Do not

No default flips, no engine patches, no new image unless the non-docs files changed, no re-runs of anything in the
**R — refuted** table (batched-token budget above 2048, bf16 state, the checkpoint flags, a per-step k schedule as
a seat lever, FlashInfer as the fast path, the 4 GiB bf16 drafter), no `prepare`, no soak, no `verify.sh`, no
quality claim from n=50, no PR, no issue, no comment, no watermark, no co-author trailer.
  under `bench/results/2026-10-14-batch3-tools/` and `2026-10-14-batch2-tools/`.
- Image: `hyperqwen:exp-f1a033e` unless Step 1 says the branch's non-docs files changed.