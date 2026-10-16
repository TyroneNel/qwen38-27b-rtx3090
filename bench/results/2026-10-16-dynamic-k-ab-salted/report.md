# DYNAMIC_K A/B, fixed prompts, 4 interleaved boots (2026-10-16)

Follow-up to `../2026-10-16-dynamic-k-ab/` (INCONCLUSIVE: B was +12.4% at N=4 but also +10.3% at N=1, where both arms
run k=7).

**Verdict: PASS.** Compared on the mean of every rep: B's N=4 decode aggregate is **+7.7%** over A (260.0 vs 241.4 tok/s),
and B's N=4 ms/pass (-23.8%) and tok/step (-18.0%) are both lower. B's N=1 per-stream tok/s is **-1.0%** (117.7 vs 118.8),
inside the 3% band. Preemptions are 0 in every rep of every arm. The N=4 rep ranges do not overlap: A spans
230.3-251.8 and B spans 253.9-271.7.

## What changed from the first run

| | First run | This run |
|---|---|---|
| Prompts | salted with the wall clock, so different text every boot | `--salt dynk1016`: the same text in every boot (still distinct per stream, rep and N, so no prefix-cache hits) |
| Ladder | `--n 1,4`, reps 2, out 256; best rep kept | `--n 1,4 --reps 6 --out 1024`; every rep kept in the JSON (`reps`) and averaged here |
| Boots | A, B, A2 | A1, B1, B2, A2 (interleaved A B B A) |
| Tools run from | WSL bash for A (blind live guard), Git Bash after | Git Bash throughout; live guard on the right LUID for every boot |

The `--salt` option and the per-rep JSON are commit `53122c7` on this branch (`bench/conc_ladder.py`). With no
`--salt`, the ladder behaves as before. The patched ladder was bind-mounted into the container next to the branch
launcher, because the image's copy predates it. In-container hashes matched the host in every boot: `start_qwen.sh`
`54fff7ef…ca19e2`, `conc_ladder.py` `cdfbf110…3fe7` (`host-hashes.txt`, `raw-mount-*.log`).

## Setup

Same as the first run: branch `a5d807e` launcher; image `ghcr.io/syv-ai/hyperqwen:latest`
`sha256:e99df7e6dc73208971ae0d1f1063deb100c997308248f253174176cf0514cb9e` (vLLM 0.31.0); 3090
`GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84` at 250.00 W; LUID `luid_0x00000000_0x0001144f_phys_0` (unchanged, no reboot);
`.env.A` and `.env.B` copied unchanged from the first run. The argvs match the brief (`--max-model-len 65536
--max-num-seqs 8 --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 --kv-cache-memory=5583457484`). A1 = A2 and
B1 = B2 byte for byte, and A and B differ only by `"num_speculative_tokens_per_batch_size":[[1,3,7],[4,512,3]]`. The
boot logs have no traceback or validation error.

## Boots

| Boot | Env | Healthy after | Free VRAM after boot | F_ctx / F_kv | Max Dedicated | Max Shared | Spill | BOOTRISE |
|---|---|---|---|---|---|---|---|---|
| A1 | A | 148 s | 1,477 MiB | 84.2 / 92.2 MiB | 23,609.8 MiB | 92.2 MiB | none | none |
| B1 | B | 136 s | 1,431 MiB | 84.2 / 92.2 MiB | 23,591.8 MiB | 92.2 MiB | none | none |
| B2 | B | 137 s | 1,431 MiB | 84.2 / 92.2 MiB | 23,591.8 MiB | 92.2 MiB | none | none |
| A2 | A | 131 s | 1,477 MiB | 84.2 / 92.2 MiB | 23,609.8 MiB | 92.2 MiB | none | none |

## Results: mean over 6 reps per boot, 12 per arm

| Row | A1 | B1 | B2 | A2 | A (12 reps) | B (12 reps) | B vs A |
|---|---|---|---|---|---|---|---|
| N=1 decode aggregate tok/s | 120.2 | 117.7 | 117.6 | 117.3 | 118.8 [111.4-125.5] | 117.6 [111.8-124.2] | -0.9% |
| N=1 per-stream tok/s | 120.3 | 117.7 | 117.6 | 117.4 | 118.8 [111.5-125.6] | 117.7 [111.9-124.1] | -1.0% |
| N=1 tok/step | 3.45 | 3.45 | 3.45 | 3.45 | 3.45 | 3.45 | 0.0% |
| N=1 tok/pass | 3.44 | 3.44 | 3.44 | 3.44 | 3.44 | 3.44 | 0.0% |
| N=1 ms/pass | 28.60 | 29.22 | 29.22 | 29.30 | 28.95 | 29.22 | +0.9% |
| N=1 preemptions / kv% | 0 / 23.8 | 0 / 23.8 | 0 / 23.8 | 0 / 23.8 | 0 / 23.8 | 0 / 23.8 | = |
| N=4 decode aggregate tok/s | 241.8 | 261.2 | 258.8 | 241.1 | 241.4 [230.3-251.8] | 260.0 [253.9-271.7] | **+7.7%** |
| N=4 per-stream tok/s | 51.9 | 54.9 | 54.5 | 51.6 | 51.7 | 54.7 | +5.6% |
| N=4 tok/step | 3.41 | 2.83 | 2.79 | 3.45 | 3.43 | 2.81 | -18.0% |
| N=4 tok/pass | 13.59 | 11.29 | 11.14 | 13.75 | 13.67 | 11.22 | -17.9% |
| N=4 ms/pass | 56.23 | 43.24 | 43.05 | 57.04 | 56.63 | 43.15 | -23.8% |
| N=4 preemptions / kv% | 0 / 84.8 | 0 / 84.0 | 0 / 83.8 | 0 / 84.6 | 0 / 84.7 | 0 / 83.9 | = |
| N=4 end-to-end aggregate tok/s | 123.7 | 129.0 | 127.9 | 123.4 | 123.6 | 128.5 | +4.0% |

Brackets are the min-max over that arm's 12 reps. The best-rep tables printed by the ladder (`raw-ladder-*.log`) agree:
N=4 A 251.8 / 250.6 against B 268.4 / 271.7.

Boot-to-boot noise with the same config: A1 vs A2 is -2.4% at N=1 and -0.3% at N=4. B1 vs B2 is 0.0% at N=1 and -0.9%
at N=4. The N=4 effect is about 8x the larger same-config gap.

## Reading

- **The first run's N=1 gap was prompt content.** With fixed prompts, the six N=1 reps give the same tok/step sequence
  in all four boots: 3.324, 3.324, 3.596, 3.445, 3.582, 3.421. Acceptance on this stack is a property of the prompt,
  not of the boot or the arm. The first run compared different prompts, and its +10.3% at N=1 came entirely from
  tok/step.
- **N=1 is unchanged.** tok/step is identical, and ms/pass is +0.9%. A1 is the outlier at 28.60 ms; the other three
  boots read 29.22-29.30. [INFERENCE] A1 ran first on the coolest GPU (53 °C after boot, against 59-60 °C).
- **N=4 matches E1.** E1 (image exp-e7fb509) measured +8.0% decode aggregate and -23.6% ms/pass. This run gets +7.7%
  and -23.8% on the newer image. Dropping to k=3 at N≥4 lowers tok/step by 18% but cuts the verify pass by 24%, so net
  throughput rises.
- The N=4 baseline here is ~241 tok/s, against ~264 in the first run. Each run used different prompts and output
  lengths (1024 against 256 tokens; longer outputs decode at greater KV depth), so absolute numbers do not carry
  between the two runs. Only the within-run A/B does.
