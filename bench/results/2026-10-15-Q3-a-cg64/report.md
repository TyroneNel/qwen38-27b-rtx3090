# Batch 5 Q3: the `CG=40` mechanism (X-1) on a shared frozen target (2026-10-15, RTX 3090 at 250 W)

This report is identical in `2026-10-15-Q3-a-cg64/` and `2026-10-15-Q3-b-cg40/`.

**Verdict: the acceptance difference does not survive a shared target. X-1 is trajectory drift, so `CG` is not an
acceptance knob and there is nothing to report upstream about acceptance.**

| finding | numbers |
|---|---|
| Batch 2's free-run gap reproduces exactly | C1 T=0 tok/step 3.34 → 3.48 on all four `run_benchmarks` runs per arm (two boots each) |
| Step time is the same | `ms/pass` at N=1: 32.3 → 32.1 ms |
| The gap vanishes when both arms replay the same frozen tokens | 39 against 39 steps on GSM8K; 182 against 184 on real4, i.e. 5.709 vs 5.647 tok/step, CG=40 1.1% *lower* |
| Free-run on the harness's chat-templated prompts shows no gain either | mean 4.59 vs 4.57 tok/step |

**What CG does change is the greedy numerics.** The two boots fork from the frozen continuation at different
positions:
- `code`: no fork vs 382.
- `real5`: none vs 29.
- `real6`: 362 vs 47.

A forked continuation is different text with a different acceptance. That is the 3.34 → 3.48.

## Arms

- **Base:** batch 2 S2b's, int8 `CTX=fast`, `DFLASH_TOKENS=7`, `KV_MEM=5315022028`, `MAX_LEN=65536`, `MAX_SEQS=8`.
- **Image:** `hyperqwen:exp-844cd90`. Env files are `.env`.

| | A (`Q3-a-cg64`) | B (`Q3-b-cg40`) |
|---|---|---|
| `CG` | unset: the launcher's min(64, 8×8) = 64 | `CG=40` |
| boot log | `'max_cudagraph_capture_size': 64`, `cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]` | `'max_cudagraph_capture_size': 40`, `cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40]` |
| free VRAM after boot | 1,617 MiB | 1,693 MiB |
| spill | none (floor `F_kv=92.2`) | none (`F_kv=92.2`) |

**Method:**
- **Harness:** `workload_accept.py`, batch 2's teacher-forced harness. The batch 5 copy only adds the eight
  `bench/prompts_real.jsonl` prompts (the C1 T=0 cohort's) as tasks `real0`-`real7`.
- **Shared target:** A captured one greedy continuation per prompt (thinking off, ≤1,024 tokens), frozen to
  `Q3-a-cg64/targets/*.json`. Both A and B replayed those exact ids: 128-token chunks of prompt + target prefix, the
  spec-decode counters diffed around each chunk, and a chunk kept only if the boot's greedy output equals the frozen
  chunk.
- **Free-run figures:** `run_benchmarks.sh single` twice per boot (C1 T=0 = run 2) and `conc_ladder --n 1`.

## Teacher-forced on the shared target (`raw-wacc-q3a.log`, `raw-wacc-q3b.log`)

| task | clean chunks A / B | steps A / B | tok/step A / B | accept A / B | first fork from the frozen target A / B |
|---|---|---|---|---|---|
| **gsm8k** (226 tokens) | 2/2 / 2/2 | **39 / 39** | **5.795 / 5.795** | 0.685 / 0.685 | none / none |
| **real4** | 8/8 / 8/8 | **182 / 184** | **5.709 / 5.647** | 0.673 / 0.664 | none / none |
| code | 8/8 / 7/8 | 220 / 199 | 4.741 / 4.568 | 0.534 / 0.510 | none / 382 |
| real0 | 6/8 / 8/8 | 164 / 205 | 4.774 / 5.073 | 0.539 / 0.582 | 706 / none |
| real1 (= chat) | 5/8 / 6/8 | 172 / 218 | 3.715 / 3.541 | 0.388 / 0.363 | 252 / 514 |
| real2 | 7/8 / 5/8 | 476 / 337 | 1.882 / 1.902 | 0.126 / 0.129 | 687 / 359 |
| real3 | 7/8 / 6/8 | 181 / 160 | 4.956 / 4.850 | 0.565 / 0.550 | 480 / 243 |
| real5 | 8/8 / 6/8 | 176 / 132 | 5.932 / 5.939 | 0.705 / 0.706 | none / 29 |
| real6 | 6/8 / 6/8 | 192 / 193 | 4.068 / 4.036 | 0.438 / 0.434 | 362 / 47 |
| real7 (439 tokens) | 3/4 / 3/4 | 76 / 80 | 4.132 / 3.875 | 0.447 / 0.411 | 272 / 272 |

- **The like-for-like rows are bold:** the only tasks where both arms replayed every chunk cleanly, so the same tokens
  were scored.
  - GSM8K is identical to the step.
  - real4 is 1.1% lower at `CG=40`.
- **Every other row scores a different subset of chunks per arm**, because a chunk is dropped when that boot's greedy
  output forks. Those rows are not a comparison.
- **Even A replaying its own capture forks:** `chat` at 252, `real0` at 706. A chunked re-prefill and a continuous
  speculative decode do not produce bit-identical greedy tokens on the same boot.

## Free-running (`raw-rb-*.log`, `raw-ladder-*.json`, and the first attempt's `attempt1-no-harness/raw-rb-*.log`)

| | A `CG=64` | B `CG=40` |
|---|---|---|
| C1 T=0 tok/step, attempt-1 boot r1 / r2 | 3.34 / 3.34 | 3.48 / 3.48 |
| C1 T=0 tok/step, this boot r1 / r2 | 3.34 / 3.34 | 3.48 / 3.48 |
| C1 T=0 decode, this boot r2 | 109.6 | 115.2 (+5.1%) |
| ladder N=1 `ms/pass` | 32.3 | 32.1 |
| harness free-run tok/step, mean of 10 tasks (chat-templated, thinking off) | 4.59 | 4.57 |

**Reading:** the `run_benchmarks` gap is perfectly repeatable, the same on two boots per arm. It is still a property
of the trajectory and not of the verifier:
- **Its prompts are deterministic per arm**, so each arm repeats its own continuation.
- **The two arms' continuations differ** (the fork column).
- **On a shared continuation the acceptance is equal** (GSM8K) or slightly lower at `CG=40` (real4).
- **The C1 gain stays on this one prompt set:** the +5.1% C1 decode is real on these eight prompts and does not carry
  over to the harness's own free-run prompts (4.59 vs 4.57).

## First attempt

- **What happened:** `q35.sh` first ran the harness from `/app/bench/results/...`, which does not exist in the
  container. The `./bench` bind mount lives only in the untracked `docker-compose.override.local.yml`, which compose
  does not load. Both replays exited `can't open file` (`attempt1-no-harness/raw-wacc-*.log`).
- **The rerun:** this boot pair (`q3.sh`) copies the harness and the frozen targets in with `docker compose cp`.
- **What is kept:** the attempt-1 `run_benchmarks` and ladder logs. They reproduce the same 3.34/3.48.

## Verified vs assumed

**Verified:** every number is from the raw logs here.

**Not verified:**
- The harness keeps only per-task aggregates, so clean chunks cannot be matched across arms inside a task. The
  like-for-like evidence is the two fully clean tasks: 221 against 223 steps on identical tokens.
- The kernel-level cause of the fork (which op's numerics move with the capture set) is not isolated.
