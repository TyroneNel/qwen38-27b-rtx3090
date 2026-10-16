# DYNAMIC_K A/B on the 3090, launcher default dflash2 config (2026-10-16)

**Verdict: INCONCLUSIVE.** B's N=4 decode aggregate is +12.4% over the mean of A and A2, and ms/pass (-28.2%) and
tok/step (-17.8%) both drop, so the schedule took effect and raised N=4 throughput. It is not PASS because B's N=1
per-stream tok/s is +10.3% over the A/A2 mean, outside the 3% band. It is not FAIL because N=4 is well above +1%.
A2 was the one extra boot the brief allows; nothing further was run.

## Setup

| | |
|---|---|
| Branch commit | `a5d807e` single-user: DYNAMIC_K forwards vLLM's per-batch-size draft count (branch `launcher-dynamic-k`) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest`, `sha256:e99df7e6dc73208971ae0d1f1063deb100c997308248f253174176cf0514cb9e`, created 2026-10-11T14:12:13Z, vLLM 0.31.0 (`num_speculative_tokens_per_batch_size` grep count 2) |
| Why this image | `hyperqwen:exp-f1a033e` and `hyperqwen:exp-844cd90` were not on the machine; `:latest` was pulled with the owner's approval |
| Image vs branch | every file in `single-user/`, `patches/`, `launcher_common.sh`, `resolve_config.sh`, `resolve_api_key.sh`, `verify.sh`, `bench/conc_ladder.py` is byte-identical to the branch except `single-user/start_qwen.sh` (bind-mounted from the branch) and `single-user/README.md` |
| Launcher in container | sha256 `54fff7ef…ca19e2` in all three boots, equal to the host file; DYNAMIC_K count 6 |
| 3090 | `GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84`, power limit 250.00 W (set by the owner before the run; preflight read 350 W) |
| 3090 adapter LUID | `luid_0x00000000_0x0001144f_phys_0` (a 2 GiB allocation on the 3090 raised it by 2,303 MiB) |
| Argv | `vllm serve /app/models/Qwen3.8-27B-W4A16-AutoRound-fast … --max-model-len 65536 --max-num-seqs 8 --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 … --kv-cache-memory=5583457484`; in-container `nvidia-smi -L` lists the 3090 as GPU 0 (the 3080 Ti also appears as GPU 1; `CUDA_VISIBLE_DEVICES` pins the 3090) |
| A vs B argv | identical except B's `--speculative-config` adds `"num_speculative_tokens_per_batch_size":[[1,3,7],[4,512,3]]`; A2's argv is identical to A's |
| Ladder | `bench/conc_ladder.py --n 1,4`: distinct ~4096-token prompts, out=256, reps=2, temperature 0 |

## Boots

| Arm | Healthy after | Free VRAM after boot | F_ctx | F_kv | Max Dedicated | Max Shared | Spill | BOOTRISE |
|---|---|---|---|---|---|---|---|---|
| A  | 141 s | 1,477 MiB | 84.2 MiB | 92.2 MiB | 23,609.8 MiB | 92.2 MiB | none (checked afterwards, see below) | none (step 8.0 MiB) |
| B  | 143 s | 1,431 MiB | 84.2 MiB | 92.2 MiB | 23,591.8 MiB | 92.2 MiB | none (live guard) | none (step 8.0 MiB) |
| A2 | 138 s | 1,477 MiB | 84.2 MiB | 92.2 MiB | 23,609.8 MiB | 92.2 MiB | none (live guard) | none (step 8.0 MiB) |

`raw-boot-B.log` has no traceback and no validation error.

**A's live guard was blind.** Arm A's tools were started with `bash` from a shell where `bash` resolved to
`C:\Windows\system32\bash.exe` (WSL), not Git Bash. WSL does not inherit the exported `LUID`, so `spillwatch.sh` and
the first `memsum.sh` used the script default `luid_0x00000000_0x00011456_phys_0` and saw no samples (the first A
block in `mem.txt`, `samples=0`). The container, the ladder and the counter CSV were unaffected: `compose up` and the
sampler go through PowerShell, and the CSV records every adapter. `memsum.sh` was then re-run for A from Git Bash
with the right LUID (the second A block in `mem.txt`): no sample after the first request above F_kv + 16 MiB. B and
A2 ran from Git Bash with the live guard on the right LUID.

## Results

| Row | A | A2 | B | B vs mean(A, A2) |
|---|---|---|---|---|
| N=1 decode aggregate tok/s | 126.4 | 122.9 | 137.0 | +9.9% |
| N=1 per-stream tok/s | 127.1 | 122.4 | 137.6 | +10.3% |
| N=1 tok/step | 3.49 | 3.47 | 3.82 | +9.8% |
| N=1 tok/pass | 3.5 | 3.4 | 3.8 | +10.1% |
| N=1 ms/pass | 27.6 | 27.7 | 27.8 | +0.5% |
| N=1 preemptions | 0 | 0 | 0 | = |
| N=1 kv% | 23.7 | 23.7 | 23.7 | = |
| N=4 decode aggregate tok/s | 264.4 | 263.6 | 296.8 | +12.4% |
| N=4 per-stream tok/s | 44.5 | 44.0 | 48.1 | +8.7% |
| N=4 tok/step | 3.56 | 3.64 | 2.96 | -17.8% |
| N=4 tok/pass | 14.1 | 14.1 | 11.4 | -19.1% |
| N=4 ms/pass | 53.2 | 53.5 | 38.3 | -28.2% |
| N=4 preemptions | 0 | 0 | 0 | = |
| N=4 kv% | 83.4 | 83.4 | 83.4 | = |

B vs A alone: N=4 decode aggregate +12.3% (296.8 vs 264.4), ms/pass -28.0%, tok/step -16.9%; N=1 per-stream +8.3%.

## Reading

- N=4 meets every PASS condition against both A and the A/A2 mean: decode aggregate ≥ +5%, ms/pass and tok/step
  lower, 0 preemptions in every arm.
- N=1 is what blocks PASS. N=1 is inside the schedule's first range (k=7 in both arms), and its ms/pass is flat
  (27.6 / 27.7 / 27.8). The whole N=1 gain comes from tok/step (3.82 vs 3.48), i.e. more tokens accepted per step.
  [INFERENCE] This is likely prompt-dependent acceptance variance: the ladder salts prompts with the wall clock, so
  each boot gets different text, and the drafter samples probabilistically. But A and A2 agree to 0.6% on tok/step,
  so a single B boot cannot separate variance from a real effect at N=1.
- The single-stream concern was a regression. B is faster at N=1, not slower. The rule asks for "unchanged within
  3%", and B did not meet it.
- Both A boots land well above the E1 reference (~243 tok/s at N=4 on exp-e7fb509). This run used a different image
  (`:latest`). There was no spill (Shared stayed at F_kv), and the GPU and model were confirmed, so the higher
  baseline is not a measurement fault.
