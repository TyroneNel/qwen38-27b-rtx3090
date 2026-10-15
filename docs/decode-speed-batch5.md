# Batch 5: the speed levers that need a GPU

Batches 1-4 measured what the stack does. Batch 5 is the residue: the levers on the
[speed worklist](decode-speed-findings.md) that cannot be settled by reading, plus one reconnaissance question
that decides whether the biggest open lever is worth attempting. Everything here is config or measurement — no
engine patches.

## Q1 — the unadvertised mamba knobs (S-11), the cheapest unknown on the list

The engine exposes more Mamba controls than the docs use (batch 1's probe):
`--mamba-ssu-algorithm {auto,horizontal,simple,vertical,None}`, `--mamba-backend`, plus the three already refuted.
None has ever been measured on our workload. One boot each, everything else fixed at the **band-2 profile**
(`DFLASH_TOKENS=3` + `EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head'`,
`MAX_SEQS=16`, `KV_MEM` pinned):

| arm | `EXTRA_ARGS` addition |
|---|---|
| Q1-a | *(none — the baseline for this batch)* |
| Q1-b | `--mamba-ssu-algorithm horizontal` |
| Q1-c | `--mamba-ssu-algorithm vertical` |
| Q1-d | `--mamba-backend flashinfer` |

Per arm: C1 T=0, `bench/conc_ladder.py --n 1,4,8,16`, the held-load `Running:` probe, `ms/pass`, and the boot log's
kernel-selection lines (`GDN decode kernel: triton|cuda`, the mamba warmup lines). **Report which kernel engaged** —
a knob that changes no kernel line did not apply, and that is the result.

- Pass = 3% or more on C1 T=0 or on the N=8 aggregate, with the kernel line confirming it engaged.
- A quality smoke (GSM8K n=50) on any arm that wins, before it is called a win.

## Q2 — is the state reservation per-step-k-able? (S-5 reconnaissance)

S-5 is the only lever with no ceiling: make the recurrent-state reservation follow the *per-step* k instead of the
configured maximum, which would give k=7 speed at 13 seats. Before anyone attempts it, answer three questions
**from the image's source and logs, not from guesswork**:

1. **Where does the reservation come from?** Find the code that sizes the mamba/state pool from
   `num_speculative_tokens` (start at `vllm/v1/core/kv_cache_utils.py` / `kv_cache_planning.py`, and the
   `MambaManager` in `single_type_kv_cache_manager.py`). Quote the file, line and expression.
2. **Is the allocation per-request static or per-step?** Batch 3 established that a per-step schedule frees
   nothing, so something is sized up front. Identify it precisely, and say whether it *could* follow the step's k
   without a redesign — one paragraph, with the code.
3. **Does upstream have a mechanism for it?** `use_replayssm` / `replayssm_buffer_len` and
   `enable_mamba_shared_prefix_checkpoint` are the two candidates. Report what ReplaySSM's kernel would need to
   become spec-decode-safe (its docstring says non-speculative only) — from its docstring and any tests, not by
   running it.

- Output is a written feasibility answer with quoted code, not a benchmark. **Verdict: "attemptable", "attemptable
  with X", or "needs a redesign because Y".** Do not write the patch.

## Q3 — finish X-1 if batch 4 did not (the `CG=40` mechanism)

Batch 4 Q5 covers this; run it here only if batch 4 skipped it. Two boots, `CG=64` (default) and `CG=40`, on the
teacher-forced harness with a **shared frozen target**. Metric: tokens/step and accept rate on the same sequence.
If the acceptance difference survives a shared target, `CG` is not a speed knob and that is worth an upstream
report; if it vanishes, batch 2's E5 was measuring trajectory drift.

## Q4 — one cheap precision answer to go with the speed

The band-2 profile is the recommended shape for 2-6 users and its quality gate has only been GSM8K n=200 and
perplexity (batch 2 S1). For a profile that is about to be recommended rather than measured, add the one metric we
have not run on it: **IFBench prompt-level strict**, 300 prompts, on the band-2 profile against the shipped one.
Batch 2 established that this benchmark has the resolution to settle a real difference (55-against-6
disagreements on `reasoning_effort`), so it will answer "is the int8-KV/TRITON profile as precise" with a number
rather than an argument.

Client concurrency must match the resident limit — batch 2 preempted 1,017 times by pointing 32 threads at a
## Budget and order

Q1 ≈ 50 minutes (four boots, one ladder each). Q2 is reading plus one page of writing, no GPU — do it first if the
machine is busy. Q3 ≈ 20 minutes. Q4 ≈ 40 minutes. If time is short: Q2 (it decides a high-effort lever) and Q1
(four boots for four answers) before Q4.

## What not to do

No engine patches. No default flips — the profiles in
[decode-speed-findings.md](decode-speed-findings.md) are recommendations, and this batch only adds evidence to
them. No re-running of anything in that file's **R — refuted** table. No new image unless the branch's non-docs
files changed. No soak, no `verify.sh`.
4-request server. Use 4 threads at `MAX_SEQS=16`.