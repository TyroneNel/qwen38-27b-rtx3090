# E3: acceptance by workload

RTX 3090, 250 W, container boot. One server with the shipped config (`.env.A`: `SPEC=dflash2 PREFIX_CACHE=1
CTX=fast DFLASH_TOKENS=7 KV_MEM=5583457484 MAX_LEN=65536`, the same as E1-A). Free VRAM after boot and the
Shared Usage trace are in `mem.txt`; no sample went above the loaded-engine floor.

## Method

`bench/labd_accept.py` cannot run these workloads. Its six tasks are fixed Danish instructions over a corpus it
builds from `~/qwen-serving/*.md`, which the image does not have. `../2026-10-13-tools/workload_accept.py`
applies the same prefix-resync teacher forcing to three prompts. The brief allows no product change, so it
lives under `bench/results/`. For each prompt it:

1. Captures one greedy continuation of up to 1,024 tokens, thinking off (`enable_thinking: false`, as
   `labd_accept.py` does). It freezes the token ids to `targets/<task>.json`.
2. Replays the frozen ids in 128-token chunks at temperature 0 and diffs the `vllm:spec_decode_*` counters
   around each chunk. It counts a chunk only when the server reproduced it exactly.
3. Measures free-running greedy C1 decode on the same prompt twice.

The prompts:

- gsm8k: the first test question of `quality-data/gsm8k`, plus a step-by-step instruction.
- code: one long commented Python function, with the prompt text in the script.
- chat: line 2 of `bench/prompts_real.jsonl`, the SSD flash-translation-layer explanation.

The first pass (`raw-wacc-A-1.log`) captured the targets and is discarded. The second pass
(`raw-wacc-A-2.log`) replayed the same frozen files and is kept.

## Results (kept pass, `raw-wacc-A-2.log`)

| workload | target tokens | clean chunks | steps | tok/step | accept rate (of 7 drafts) | free-run tok/step | C1 decode, tok/s (2 reps) | tok/step vs chat |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 226 | 2/2 | 39 | **5.795** | 0.685 | 5.795 | 196.8 / 196.0 | **1.66x** |
| code | 1,024 | 8/8 | 218 | **4.784** | 0.541 | 4.725 | 157.2 / 156.6 | **1.37x** |
| chat | 1,024 | 4/8 (first divergence at token 497) | 147 | **3.497** | 0.357 | 3.589 | 105.7 / 118.3 | 1.00x |

The discarded pass replayed the same targets and gave the same tok/step to the third decimal. Its C1 decode
read 214.5/215.2 on gsm8k, 145.4/170.1 on code and 110.2/123.3 on chat. The run-to-run spread on decode is
therefore about 10%, while tok/step on a frozen sequence does not move.

## Verdict: pass (the table exists)

Two workloads are at or above 1.3x the chat row:

- **gsm8k at 1.66x** (5.80 tokens per step, ~196 tok/s single-stream).
- **code at 1.37x** (4.78, ~157 tok/s).

This stack should be quoted and tuned on those two, with the workload named. Our chat protocol's ~3.3-3.5 is
the low end, not a property of the drafter. On GSM8K this drafter reaches 5.80 greedy, thinking off, against
the published 5.46 for `z-lab/Qwen3.8-27B-DFlash2`, which ran with thinking on, temperature 1.0 and block 8
(`docs/decode-speed-landscape.md` G2). The two numbers come from different sampling, so they are not a like
comparison. They show only that the gap the landscape page describes ("ours ~3.1-3.65") is the workload, not
the W4A16 drafter.

## Limits

- The gsm8k target stopped on its own at 226 tokens, so its row rests on 39 steps.
- The chat row counts 4 of 8 chunks: greedy ties broke differently from token 497 on. As
  `labd_accept.py`'s own notes say, the clean-only number is biased slightly high.
- One prompt per workload. This is a lead, not a benchmark.

## Verified vs assumed

Verified: every number above (raw logs and the frozen targets in `targets/`). Not verified: that one GSM8K
question, one code prompt and one chat prompt represent their classes.
