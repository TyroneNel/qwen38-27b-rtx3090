# Batch 5: the speed levers that need a GPU

Batches 1-4 measured what the stack does. Batch 5 is what is left: the levers that cannot be settled by reading, the `CG=40`
question batch 4 skipped, one precision gate for the profile that is now recommended to users, and the first
hardware run of the launcher seat print this branch changed.
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

## Q2 — the launcher seat print, on hardware (new product code, never booted)

`single-user/start_qwen.sh` no longer prints a constant. It prints the function batch 4 Q1 measured —
one resident request is `6(k+2) + 2*ceil(ctx/864)` pool pages of a 14,601,709 B page, evaluated at a ~1k prompt and
at `MAX_LEN` — and it skips `CTX=huge`, whose block and page differ. The arithmetic is covered by
`bench/test_resident_print.sh` (CPU only), but **the print has never run on a real boot**, and its two fitted
constants (`6` and `14,601,709`) come from four points at k=7 on one pool. Verify it where it will be used.

**Arms** — the band profiles from [single-user/README.md](../single-user/README.md), one boot each, reporting for
each: the launcher's printed pair, and the **observed** seats at ~1k and at ~32k prompts.

| arm | env | what to check |
|---|---|---|
| Q2-a | `KV_MEM=5583457484 DFLASH_TOKENS=7 MAX_SEQS=16 MAX_LEN=65536` | printed 6 and 1; observed 6 at ~1k (batch 4), 2 at 32k |
| Q2-b | as above with `DFLASH_TOKENS=3` | printed 11 and 2; observed 9 at ~1k (batch 4) |
| Q2-c | `DFLASH_TOKENS=7` + a `KV_MEM` cut ~1 GiB | does the print track the pool, or is it a constant in disguise |

- **Method:** the held-load probe (`running_probe.py`) at ~1k and at ~32k prompts, plus the engine's own
  `Running:` line; the shortest and longest prompt are what the print claims to bound, so measure at both.
- **Pass = the printed pair brackets the observation**, per arm: the ~1k figure at or above the observed seats at
  ~1k, and the `MAX_LEN` figure at or below the observed seats at depth. It is a bound, not a point estimate.
- **Report the error**, not just pass/fail: printed minus observed at each end, per arm. A consistent bias is
  worth more than a pass.
- **If it is wrong**, say by how much and at which end, and quote the observed `Running:` and `num_gpu_blocks`.
  The arithmetic is in `single-user/start_qwen.sh` and the test is the place to encode the correction.
- **Do not** tune the constants to fit — they are calibrated at k=7 and the divergence at k=3 is expected. What
  this arm establishes is whether the *bound* holds and how wide it is.


## Q3 — the `CG=40` mechanism (X-1) — batch 4 skipped this

Batch 4 did not run it, so this is the run. Two boots, `CG=64` (default) and `CG=40`, on the
teacher-forced harness with a **shared frozen target**. Metric: tokens/step and accept rate on the same sequence.
If the acceptance difference survives a shared target, `CG` is not a speed knob and that is worth an upstream
report; if it vanishes, batch 2's E5 was measuring trajectory drift.

## Q4 — the precision gate band 2 needs before it is recommended

The band-2 profile is the recommended shape for 2-6 users and its quality gate has only been GSM8K n=200 and
perplexity (batch 2 S1). For a profile that is about to be recommended rather than measured, add the one metric we
have not run on it: **IFBench prompt-level strict**, 300 prompts, on the band-2 profile against the shipped one.
Batch 2 established that this benchmark has the resolution to settle a real difference (55-against-6
disagreements on `reasoning_effort`), so it will answer "is the int8-KV/TRITON profile as precise" with a number
rather than an argument.

Client concurrency must match the resident limit — batch 2 preempted 1,017 times by pointing 32 threads at a
## Q5 — X-3: what the 16-seat boot costs k=7 (optional, if time remains)

Batch 4 found the band table's cost column carries something that is not k's doing: **C1 T=0 is 103.4 on a
`MAX_SEQS=16` boot against batch 3's 112.1 on an 8-seat boot**, a 7.8% loss, while k=3 (−0.6%) and k=1 (−2.2%)
barely move. The boot log's one `MAX_SEQS`-sized buffer doubles (`int4 3D scratch: capacity=256 query tokens`
against 128), and the cause is not isolated.

- **A** = `MAX_SEQS=16` (the band profile), **B** = `MAX_SEQS=8`, **both at k=7**, one boot each, C1 T=0.
- Read: C1 T=0, `ms/pass` at N=1, and every `MAX_SEQS`-scaled allocation in the boot log — quote them side by side.
- **Pass = the 7.8% loss reproduces and the buffer that doubles is named.** If it does not reproduce, say so; that
  would mean batch 4's figure was a single-boot artifact and the band table's C1 cost is cleaner than recorded.

## Budget and order

Q2 ≈ 60 minutes (three boots, probed at two prompt depths each) — run it first: it verifies product code this
branch changed. Q1 ≈ 50 minutes (four boots, one ladder each). Q3 ≈ 20 minutes. Q4 ≈ 40 minutes. Q5 ≈ 20 minutes.
If time runs short: Q2, then Q1 (four boots, four answers), then Q4 (precision for a profile users are told to
use); Q3 and Q5 are closes.

## Already answered — do not re-run

- **The S-5 reconnaissance.** Reading the 0.31.0 source answered it: the state reservation is static, sized from
  the configured k (`MambaSpec.max_memory_usage_bytes`), and a per-step schedule cannot lower it — nor should it,
  since k+1 positions need k+1 pages live. See the note in [decode-speed-findings.md](decode-speed-findings.md).
- **ReplaySSM, anywhere.** Batch 4 Q3 showed the engine refuses it for this architecture
  (`--use-replayssm is not supported for architecture 'Qwen3_5ForConditionalGeneration'`), so no profile can
  engage it.

## What not to do

No engine patches. No default flips — the profiles in
[decode-speed-findings.md](decode-speed-findings.md) are recommendations, and this batch only adds evidence to
them. No re-running of anything in that file's **R — refuted** table. No new image unless the branch's non-docs
files changed. No soak, no `verify.sh`.
4-request server. Use 4 threads at `MAX_SEQS=16`.