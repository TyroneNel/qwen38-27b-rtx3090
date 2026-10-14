# GPU agent brief, batch 3 — settle what caps residency

Paste the whole of this file as the task for the agent on the Windows box. The arm definitions and the decision
rules live in [decode-speed-batch3.md](decode-speed-batch3.md), which you must read before Step 0. Batches 1 and 2
are in [../bench/results/2026-10-13-SUMMARY.md](../bench/results/2026-10-13-SUMMARY.md) and
[../bench/results/2026-10-14-batch2-SUMMARY.md](../bench/results/2026-10-14-batch2-SUMMARY.md).

Read in this order: the G7 section of [decode-speed-landscape.md](decode-speed-landscape.md) (why this test
exists, and the numbers it has to explain), then [decode-speed-batch3.md](decode-speed-batch3.md) (the arms and
the decision rules), then this file. The batch-1 brief carries a superseded banner; use this one.

## Role

Same machine, same rules. Batch 2 showed that doubling the KV pool bought no seats: residency moved 5 → 6 and the
N=8 aggregate came out lower. The launcher predicts residency from the verify block
(`RESIDENT = KV_MEM / ((k+2) × 104988089)` → ~6 seats at k=7, ~10 at k=3), but batch 2's k=3 arm still read 5. Your
job is to settle that with one discriminating test, report **Conclusion A or Conclusion B** explicitly, and test
one independent lever (bf16 recurrent state) that may be worth a step-time win. The landscape page's G7 section
carries the seat model and the batch-2 contradiction in full, and the backlog row 13 records this test — cite
those numbers rather than re-deriving them. Commit results and reports to `cline/gf59cdmh` and push.

## The hard rules

1. **This batch answers a question; it does not ship anything.** No default flips, no edits under `single-user/`,
   `batch/`, `patches/`, `prepare/` or `kvarn/`, nothing outside `bench/results/` and `docs/`.
2. **Speed is everything.** No soak, no `verify.sh`. The only long steps are Q2's quality pair (GSM8K n=200 and
   the three perplexity texts) and they are bounded.
3. **No assumptions.** Every fact is a command's output, quoted. "Not verified" is a complete answer.
4. **No PRs, issues, comments or reviews.** No `gh pr`, `gh issue`, no `gh api` write call.
5. **No co-authored commits, no watermarks.** One author, no `Co-Authored-By`, no `Signed-off-by`, no tool or agent
   name, no banners.
6. **Only `cline/gf59cdmh`.** Never `main`, never the `upstream` remote, never force-push, never amend a pushed
   commit.
7. **Leave the machine as you found it.** Containers down, no stray `docker run`, no files outside the repo.

## What batches 1-2 established — verify, do not re-derive

- The 3090 is headless (the 3080 Ti holds the display); pin it by UUID in `CUDA_VISIBLE_DEVICES` and
  `NVIDIA_VISIBLE_DEVICES`. `power.limit` reads 250.00 W. Models under `G:/models` are prepared — never run
  `prepare`.
- `docker compose up` must run from PowerShell, not Git Bash.
- The spill oracle is the counter, not free VRAM: idle Shared Usage 0.25 MiB, loaded-engine floor 84-92 MiB. A
  live guard exists from batch 2 (`bench/results/2026-10-14-batch2-tools/spillwatch.sh`) — reuse it, and note its
  known false positive (a CSV row read mid-write made the floor 0.0 and triggered a spurious remediation; the
  guard now ignores sub-1 MiB readings and needs two consecutive samples). Batch 2 pre-cut `KV_MEM` by 256 MiB for
  its timing sessions to keep the guard quiet; do the same only if you keep it identical across the arms you
  compare.
- **Perplexity alone spills this box at the shipped pin** (gotcha 60: 156 MiB bf16 / 410 MiB int8 against the
  84-92 MiB floor). Watch the counter during Q2's perplexity step and say whether the floor held.
- The int8 base is validated: `TRITON_ATTN` + `int8_per_token_head`, GSM8K n=200 within one question, perplexity
  within 0.07%, pool 116,440 tokens against 68,605. Use it as Q1's base.
- Noise floors: C1 T=0 within a boot 0.0-0.1%, cross-boot up to 3.2%; teacher-forced tokens/step is noise-free;
  free-run decode ~10%; preempted ladder rows up to 13%; the C8 cohort spread is 0.1% on the int8 pool.
- The image from batch 2 is `hyperqwen:exp-f1a033e`.

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

Into `bench/results/<date>-batch3-preflight/preflight.txt`. If the branch is dirty with tracked modifications,
stop and report the diff. Untracked owner files (`.env.*`, `docker-compose.override.local.yml`) are left alone;
name every path in `git add`.

## Step 1 — image reuse check (5 minutes)

```powershell
git diff --stat <image-commit>..HEAD -- . ':(exclude)docs' ':(exclude)bench/results'
```

Nothing printed means reuse the batch-2 image; record its digest and do not rebuild. Anything printed means
rebuild once, tag `hyperqwen:exp-<short-sha>`, record the digest.

## Step 2 — Q1, the discriminating test (about 40 minutes)

Three boots on the **int8 base**, identical except `DFLASH_TOKENS`: **7**, **3**, **1**. Everything else fixed:
`KV_MEM=5583457484`, `MAX_LEN=65536`, `MAX_SEQS=8`, `PREFIX_CACHE=1`, the 3090 by UUID. Do not vary `KV_MEM`.

For each arm, in this order:

1. Boot, wait for health, and record from the boot log: the launcher's own `RESIDENT` note (it prints when
   `MAX_SEQS` exceeds what the pool holds), the `GPU KV cache size` line, and every line mentioning
   `mamba`/`block size`/`page`.
2. **The `Running:` probe** — the primary metric. Fire exactly 8 concurrent requests with distinct prompts and
   512 output tokens, hold the load, and while all 8 are mid-generation sample the engine log for its
   `Running: N reqs, Waiting: M` line. Take three samples per arm and report every pair verbatim. Whatever script
   you use lives under `bench/results/` (batch 2 set that precedent with `workload_accept.py`); name it in the
   report.
3. `bench/conc_ladder.py --n 1,2,4,8,16 --json` — the `run` column is the secondary metric, and its
   preemptions/`kv%`/`ms/pass`/`tok/step` come along.
4. C1 T=0 from `bench/run_benchmarks.sh single`, second run of the boot.

Then write the conclusion, and write it as one of the two the batch doc defines:

- **Conclusion A — the model holds:** `Running:` rises by 3 or more from k=7 to k=3, preemptions fall toward 0,
  and the N=8/N=16 aggregate rises. Include the seats-per-k curve.
- **Conclusion B — the model is wrong:** `Running:` stays within ±1 across all three arms. Say so plainly, name
  the next suspect the evidence points at, and note that the launcher's `RESIDENT` print over-predicts and should
  be corrected.
- "Inconclusive" only if the probe could not hold the load — then say exactly what prevented it.

If the two metrics disagree, the log line wins and both go in the report.

## Step 3 — Q2, bf16 recurrent state (about 45 minutes)

Arms: A = `--mamba-ssm-cache-dtype float16` (shipped), B = `bfloat16`, k=7, int8 base, interleaved A-B-A.

- Per boot: the GDN-path line (`Falling back to the Triton GDN decode path…` or the fused-kernel equivalent),
  C1 T=0, `ms/pass` at N=1 and N=8, tokens/step, free VRAM after boot, and the `Running:` probe.
- Quality, on both arms: GSM8K n=200 and the three perplexity texts, **with the spill guard watching** and the
  counter reported next to the perplexity numbers.
- Conclusion: B wins if the fused kernel appears on B and not on A **and** (C1 T=0 ≥ +3% or `ms/pass` ≥ −3%) with
  tokens/step within noise and quality within one GSM8K question and 1% perplexity. Otherwise say why B lost —
  including "the fused kernel did not appear", which would mean the whitelist entry is about the model config
  rather than this flag.

## Step 4 — Q3 and Q4 (about 20 minutes, optional if time is short)

- **Q3:** one boot each with `--enable-mamba-shared-prefix-checkpoint` and with `--mamba-block-size 8192`, A/B
  against the Q1-a boot. The finding is the **boot-log evidence**: quote the block-size and page-padding lines.
  A flag that changes none of them did not apply — that is the result.
- **Q4:** confirm the instrumentation, do not hunt for it. From the image, report whether
  `vllm serve --help=all` exposes a ReplaySSM flag, and whether any boot line reports mamba blocks or state
  slots. Batch 1 already established there is **no** ReplaySSM flag on this pin, and that the config field's
  docstring restricts it to non-speculative decode — so its absence is expected: record it and move on rather
  than looking for a way to enable it. If no boot line reports the state pool, say that plainly; it is why the
  `Running:` probe is the primary metric, and a counter becomes a candidate patch.


## Step 5 — record and push (after Q1, and again after Q2)

One directory per arm under `bench/results/<YYYY-MM-DD>-<arm-id>/`: `.env` (with `KV_MEM`, `MAX_LEN`,
`DFLASH_TOKENS`, the mamba dtype), `cmd.txt`, `image.txt` (tag, digest, commit, boot class), `mem.txt` (free VRAM
after boot and the Shared Usage trace), `raw-*.log` verbatim including the `Running:` samples, `report.md` with
the arms side by side, the metric named, the raw numbers, the delta, the verdict, and one line on what was
verified versus assumed. Then `bench/results/<date>-batch3-SUMMARY.md`, whose **first line is the conclusion**:
"Conclusion A — the model holds" or "Conclusion B — the model is wrong", followed by the seats-per-k table.

```powershell
git add bench/results/<arm-dir>          # name paths explicitly; never git add -A
git commit -m "exp: batch3 <question> <arm> (250W, <date>)"
git push origin cline/gf59cdmh
```

Push after Q1 and again after Q2. An unpushed result did not happen.

## Step 6 — report back

Lead with the conclusion, not the table. Then the evidence for it (the `Running:` pairs, the launcher's
`RESIDENT` estimate, preemptions, the N=8/N=16 aggregates), then Q2's outcome with the GDN-path lines, then the
caveats. If the answer contradicts `docs/decode-speed-batch3.md` or the launcher's own comment, say so with the
number — a corrected model is worth more than a confirmed one.

## Failure playbook

| symptom | do this |
|---|---|
| the 8-request probe cannot hold 8 streams | report what `Running:` read and why (prompt length, prefill interleave, OOM) — that is itself evidence about the cap |
| `Running:` never appears in the log | say so, use the ladder's `run` column as secondary, and note the missing instrumentation in Q4 |
| Shared Usage above the loaded floor | check the guard is not reading a mid-write CSV row (batch 2's false positive); if real, cut `KV_MEM` identically on all arms and record it |
| a knob is absent from the engine's args line | the arm's result is "knob not applied"; quote the args line and continue |
| preemptions in a ladder row | mark the row a tail measurement, do not report it as steady state |
| `RuntimeError: UVA is not available` | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` missing from `.env` |
| bash errors with `\r` | CRLF tree: `git config core.autocrlf false`, re-checkout |
| an arm past twice its budget | record what you have, mark "incomplete — time-boxed", move on |

## Do not

No default flips, no engine patches, no EXL3/TurboQuant/SGLang/FlashInfer work, no `prepare`, no soak, no
`verify.sh`, no quality claim from n=50, no PR, no issue, no comment, no watermark, no co-author trailer.

