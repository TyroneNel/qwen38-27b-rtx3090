# GPU agent brief, batch 2 — capacity gate, then re-measure on it

Paste the whole of this file as the task for the agent on the Windows box. It supersedes
[gpu-agent-brief.md](gpu-agent-brief.md) for this batch; the arm definitions live in
[decode-speed-batch2.md](decode-speed-batch2.md), which you must read before Step 0.

## Role

You operate one Windows workstation with an RTX 3090 (24 GB, 250 W) and an RTX 3080 Ti (the display card),
Docker Desktop on the WSL2 backend, and PowerShell. Batch 1 ran seven arms on this box and its raw logs are in
`bench/results/2026-10-13-*`. Your job is batch 2: three sessions that bank the capacity win batch 1 measured and
then re-answer the three questions the small KV pool has been hiding. You commit results and reports to
`cline/gf59cdmh` and push. Nothing else.

## The hard rules

1. **Speed is everything.** No soak, no `verify.sh` full run, no acceptance battery. Session 1's quality probes
   are the only long steps and they are bounded: GSM8K n=200, three perplexity texts, one needle rung.
2. **No assumptions.** Every fact in a report is the output of a command you ran, quoted verbatim. "Not verified"
   is a complete answer.
3. **No pull requests, issues, comments or reviews.** Never run `gh pr`, `gh issue`, or a `gh api` write call.
   Commit and push to one branch.
4. **No co-authored commits, no watermarks.** One author, the repo's configured identity, no `Co-Authored-By`,
   no `Signed-off-by`, no tool or agent name, no banners or badges in any file.
5. **No default flips.** Batch 2 produces evidence and a one-line proposal. It does not change a launcher
   default, a compose file, or anything under `single-user/`, `batch/`, `patches/`, `prepare/` or `kvarn/`.
6. **Only `cline/gf59cdmh`.** Never commit to `main`, never push to the `upstream` remote (its push URL is
   disabled on purpose), never force-push, never amend a pushed commit.
7. **Leave the machine as you found it.** Containers down, no stray `docker run`, no files outside the repo.

## What batch 1 established about this box — verify, do not re-derive

- The 3090 is **headless**; the 3080 Ti holds the display. Pin the card by UUID in both
  `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES`; Docker Desktop does not honour `--gpus device=N`.
- `power.limit` reads **250.00 W** (default 350). Record it again in the preflight; do not change it.
- Models are already prepared under `G:/models` (`Qwen3.8-27B-W4A16-AutoRound-fast`,
  `Qwen3.8-27B-DFlash2-W4A16`, `quality-data`). **Do not run the `prepare` profile.**
- `docker compose up` must run from **PowerShell**; from Git Bash it rejects the `G:/models` bind.
- The spill oracle is the counter, not free VRAM: idle Shared Usage on the 3090 is **0.25 MiB**, the
  loaded-engine floor is **84-92 MiB**, and free VRAM after boot (1.4-1.5 GB) falls to 423-921 MiB under load
  with Shared Usage flat — that is allocator growth, not a spill. Any sample more than 16 MiB above the loaded
  floor is a spill.
- The image from batch 1 is `hyperqwen:exp-e7fb509`. Reuse it unless Step 1 says the branch has changed code.
- C1 T=0 is the decisive metric (0.0-0.1% within a boot); teacher-forced tokens/step on a frozen target is
  noise-free; free-run decode moves ~10%; preempted ladder rows move up to 13%; the C8 cohort protocol moved 37%
  within one boot.

## Step 0 — preflight (10 minutes)

Re-verify, do not copy batch 1's answers:

```powershell
cd G:\dev\HyperQwen
git fetch origin; git status -sb; git log --oneline -1
nvidia-smi --query-gpu=index,name,memory.total,power.limit,power.default_limit --format=csv
nvidia-smi --query-gpu=uuid,display_active --format=csv
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | Select-String hyperqwen
Get-ChildItem G:\models | Select-Object Name
Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -SampleInterval 10 -MaxSamples 6
```

Record into `bench/results/<date>-batch2-preflight/preflight.txt`. If the branch is dirty with tracked
modifications, **stop** and report the diff — batch 1 found the owner's untracked local files (`.env.*`,
`docker-compose.override.local.yml`, some `docs/*`) and left them alone, naming every path in `git add`. Do the
same.

## Step 1 — image reuse check (5 minutes)

```powershell
git diff --stat <image-commit>..HEAD -- . ':(exclude)docs' ':(exclude)bench/results'
```

If that prints nothing, the batch 1 image is still the code under test: reuse it, record its digest, and do not
rebuild. If it prints anything, rebuild once, tag `hyperqwen:exp-<short-sha>`, and record the digest. Either way
`docker image inspect <tag> --format '{{.Id}}'` goes in every arm directory.

## Step 2 — Session 1, the capacity gate (about 90 minutes)

Read `docs/decode-speed-batch2.md` Session 1. In short: A is shipped `CTX=fast`, B adds
`EXTRA_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"`, both with
`KV_MEM=5583457484 MAX_LEN=65536 MAX_SEQS=8` and the 3090 pinned by UUID.

1. **Quality first, interleaved boots**: GSM8K n=200 greedy thinking off, perplexity on the three texts, one
   needle rung at ~64k. Report A and B side by side with the deltas. A gap of two questions on GSM8K or 1% on
   perplexity is a fail; one question is a pass.
2. **Then timing, A-B-A interleaved** (three boots per arm, alternating): C1 T=0 second run of each boot,
   `bench/conc_ladder.py --n 4,8`, the engine's `GPU KV cache size` line, preemptions, resident count, free VRAM
   after boot, Shared Usage trace.
3. Write the verdict against the rule in the batch doc (C1 within 2% **and** pool +50% or preemptions ≤ ¼ of A's)
   and, if it passes, a one-line proposal to make B the `CTX=fast` default. **Do not make the change.**

If Gate 1 fails, say so, write the report, push it, and **stop** — Session 2 inherits the preemption problem and
is not worth running without the pool.

## Step 3 — Session 2, the parked arms on the bigger pool (about 90 minutes)

Only if Session 1 passed; every arm runs on B's base. Read the batch doc for the pass rules.

- **S2a dynamic K**: A = `DFLASH_TOKENS=7`; B = A plus a second `--speculative-config` in `EXTRA_ARGS` carrying
  `"num_speculative_tokens_per_batch_size":[[1,3,7],[4,512,3]]` (batch 1 verified the later config wins; no
  launcher knob exists and none is needed). Interleaved A-B-A at N=4 and N=8, plus C1 T=0 and tokens/step.
- **S2b capture size**: A = default `CG` (64); B = `CG=40`. Interleaved A-B-A, C1 T=0 and graph MiB.
- **S2c the C8/C16 rows**: run the cohort ladder at C8 and C16 on B's base and report the within-boot spread and
  preemptions. This is a measurement gate — a row is only quotable at ≤10% spread.

## Step 4 — Session 3, publishable comparison and the effort gate (about 60 minutes)

- **S3a**: re-run our stack at r0b0tlab's protocol — GSM8K greedy, **max 512 new tokens**, `MAX_LEN=8192`,
  thinking off, 250 W. Report C1 decode (two reps), tokens/step, pool tokens. This is the row that either makes
  "we are ahead on the like-for-like workload" quotable or retracts it.
- **S3b**: `reasoning_effort` default (xhigh) vs `low`, **thinking on**, interleaved boots. GSM8K n=200 quality,
  IFBench if present, and teacher-forced tokens/step on **each arm's own** frozen continuation (the effort
  setting changes the prompt, so a shared target measures neither arm). Pass = `low` within one question of
  xhigh and not worse on IFBench.


## Step 5 — record and push (after each session)

Same layout as batch 1, one directory per arm under `bench/results/<YYYY-MM-DD>-<arm-id>/`: `.env` (with
`KV_MEM` and `MAX_LEN`), `cmd.txt`, `image.txt` (tag, digest, commit, boot class), `mem.txt` (free VRAM after
boot and the Shared Usage trace), `raw-*.log` verbatim, `report.md` with the arms side by side, the metric named,
raw numbers, delta in percent, the verdict (pass / fail / inconclusive-noise / knob not applied / spilled /
incomplete), free VRAM after boot, and one line on what was verified versus assumed.

```powershell
git add bench/results/<arm-dir>          # name paths explicitly; never `git add -A`
git commit -m "exp: batch2 <session> <arm> (250W, <date>)"
git push origin cline/gf59cdmh
```

Then a `bench/results/<date>-batch2-SUMMARY.md`: one row per arm, and the three things worth reading — what is
now known, what could not be established and why, and anything seen that the sessions did not ask for.

Push after **each session**, not at the end. An unpushed result did not happen.

## Step 6 — report back

In your final message: the summary table (arm, A, B, delta, verdict, noise floor for that metric), then the
proposal sentence for any default flip you are recommending, then the caveats. If an arm contradicted
`docs/decode-speed-landscape.md` or batch 1, say so with the number — both pages are expected to be partly
wrong, and correcting them is worth more than confirming them.

## Failure playbook

| symptom | do this |
|---|---|
| `RuntimeError: UVA is not available` | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` missing from `.env` |
| decode suddenly 2-6× slower, no error | check the Shared Usage counter against the loaded floor, cut `KV_MEM`/`MAX_LEN`, re-run, and record what you changed |
| Shared Usage above the floor on a FlashInfer boot | expected from batch 1 (a fixed host-side mapping, 172-220 MiB). Not a spill, but say which boot it was |
| preemptions in a ladder row | that row is a tail measurement, not a steady state — mark it unusable rather than reporting it |
| a knob is absent from the engine's args line | the arm's result is "knob not applied" — record the exact args line and continue |
| bash errors with `\r` | CRLF working tree: `git config core.autocrlf false`, re-checkout |
| OOM at boot | record the engine's memory line, raise `GPU_UTIL` headroom per `docs/gotchas.md` |
| boot fails, error not recognized | grep `docs/gotchas.md` with `Select-String`, report the gotcha it matched |
| an arm exceeds twice its budget | record what you have, mark it "incomplete — time-boxed", move to the next arm |

## Do not

No EXL3, no TurboQuant, no SGLang, no FlashInfer graph chase, no GQA-packed split-KV, no second-GPU work, no
launcher default flips, no `prepare`, no soak, no `verify.sh`, no quality claim from n=50, no PR, no issue, no
comment, no watermark, no co-author trailer.

