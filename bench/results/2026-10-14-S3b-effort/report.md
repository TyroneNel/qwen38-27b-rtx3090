# S3b: `reasoning_effort=low` against the default (xhigh), thinking on

RTX 3090, 250 W, container boot, image `hyperqwen:exp-f1a033e`.

## Arms and boots

`reasoning_effort` is a request field. Omitted means the template default, xhigh, as batch 1 established. It is not
a boot knob, so the arms share boots and the interleave is by order:

| boot | config | what ran, in order |
|---|---|---|
| b1 | `.env.A`: shipped `CTX=fast` (FLASH_ATTN + bf16), `KV_MEM=5315022028`, `MAX_LEN=57344` (`knob-changes.txt`) | GSM8K xhigh, GSM8K low, teacher-forced tok/step xhigh then low |
| b2 | `.env.A` | GSM8K low, GSM8K xhigh |
| i1 | `.env.I` = S2a's B: TRITON_ATTN + int8 KV + dynamic K, `KV_MEM=5315022028`, `MAX_LEN=65536` | IFBench xhigh (300 prompts) |
| i2 | `.env.I` | IFBench low (300 prompts) |

IFBench is not in the image. It was installed per boot into a venv under `/tmp` in the container, from AllenAI
IFBench at the pinned commit `1c40f0c` (`../2026-10-14-batch2-tools/ifbench_setup.sh`; `raw-ifbench-setup-*.log`
shows `Version: 0.2.0`). The image, the serving venv and the host were not touched. Generation:
`ifbench_gen.py`, the `docs/quality.md` protocol (thinking on, model-default sampling), `max_tokens` 16,384,
scoring `message.content` after the qwen3 reasoning parser. Scoring: the checkout's own `run_eval.py`.

IFBench moved to the int8 base after a first attempt on b1 thrashed. With 32 client threads against ~4 resident
requests, the bf16 pool sat at 97% with 28 queued and recorded 1,017 preemptions; ~80 of 299 prompts were done in
55 min. The owner chose the int8 base with client concurrency 4. Dynamic K's sampling is lossless and S1 showed int8
KV quality-equivalent, so the base moves speed, not what is scored. i1 and i2 each recorded 0 preemptions
(`raw-ifbench-metrics-*.log`). i2 exists because the spill guard falsely flagged i1 at 17:41Z, after xhigh had
finished and been scored (a read race, now fixed; `cmd.txt`, `knob-changes.txt`). The flag killed the low run;
low then ran alone on a fresh boot of the same config.

## GSM8K, n=200, thinking on, greedy (`gsm_effort.py`, `raw-gsm-*.log/.json`)

| boot | xhigh correct | low correct | xhigh mean tokens / truncated | low mean tokens / truncated |
|---|---|---|---|---|
| b1 (xhigh first) | 194 | 193 | 767 / 1 | 364 / 0 |
| b2 (low first) | 194 | 194 | 693 / 0 | 356 / 0 |

Low is within one question of xhigh on both boots: **pass on GSM8K**. Low uses about half the completion tokens.
Per-question, the arms disagree on 1 question in b1 and 6 in b2 (3 each way). The same arm also flips between
boots: xhigh 4 questions, low 1. That is the spread of greedy decoding under 32-way batching, not an effort effect.

## IFBench, 300 prompts (`raw-ifbench-*`, paired scoring `raw-ifbench-pair.log`)

| | xhigh | low | delta |
|---|---|---|---|
| truncated at 16,384 tokens (still thinking, empty answer) | 63 | 17 | |
| empty answer with `finish_reason=stop` | 0 | 17 | |
| mean completion tokens | 7,136 | 4,402 | -38% |
| **prompt-level strict, official (all 300)** | **63.3%** (190) | **47.0%** (141) | **-16.3 pts** |
| prompt-level loose, official | 69.3% | 56.0% | -13.3 pts |
| prompt-level strict, answered by that arm | 80.2% (190/237) | 53.0% (141/266) | -27.2 pts |
| **prompt-level strict, answered by both (223 prompts)** | **79.4%** (177) | **57.4%** (128) | **-22.0 pts** |
| prompt-level loose, answered by both | 87.4% | 68.6% | -18.8 pts |
| instruction-level strict, answered by both | 81.5% | 61.7% | -19.8 pts |
| answered-by-both disagreements (strict) | passes 55 that low fails | passes 6 that xhigh fails | |

**Verdict: fail. `low` is clearly worse at instruction following.** The rule is low within one GSM8K question of
xhigh *and* not worse on IFBench. GSM8K passes. IFBench fails by 16 points on the official score and by 22 points
on the 223 prompts both arms answered, so the drop is not a truncation artefact: when both answer, low fails 55
prompts that xhigh passes, and xhigh fails only 6 that low passes. One sampled run per arm cannot resolve a point
or two, but a 55-to-6 split is far outside that. No recommendation to serve agentic traffic at `low`. Batch 1's
acceptance gain at low (code +21.9%, chat +14.8%) buys speed, and this run shows that speed costs instruction
following.

**The official xhigh score (63.3%) is not the published 78.3% and must not be compared to it.** 63 of 300 xhigh
answers are empty because thinking ran past the 16,384-token cap. On the 237 it answered, xhigh scores 80.2%, in
line with `docs/quality.md`'s 78.3. That page does not record its `max_tokens`. It needs one well above 16k for
xhigh, or its number cannot be reproduced.

Low has 17 empty answers that finished with `stop`, not `length`. The model ended its turn without content after
the reasoning parser split off the thinking. Not investigated.

## Teacher-forced tokens per step on each arm's own frozen target (b1, `raw-wacc-*-b1.log`, `targets-*-b1/`)

| task | xhigh tok/step (clean chunks) | low tok/step (clean chunks) | low vs xhigh |
|---|---|---|---|
| gsm8k | 5.292 (1/3) | 5.424 (2/2) | +2.5% |
| code | 3.636 (4/8) | 4.770 (8/8) | +31.2% |
| chat | 2.854 (2/8) | 3.316 (4/8) | +16.2% |

The direction matches batch 1 (code +21.9%, chat +14.8%, gsm8k +5.2%). The "clean chunks" column counts chunks
whose replay reproduced the frozen target exactly. Several replays diverged partway, so the xhigh rows rest on as
few as 1-4 chunks. Batch 1 called this metric identical to three decimals between passes, but that only holds on
the chunks that replay cleanly.

## Memory

Free VRAM after boot: b1 1,723 MiB, b2 1,729 MiB, i1/i2 1,571 MiB. Shared Usage on the 3090 stayed at the
84.2-92.2 MiB floor on b1, b2, i1 and i2 (`mem.txt`). No real spill. The i1 flag at 17:41Z was a guard false
positive (above).

## Verified vs assumed

Verified: every number above from the raw logs and the paired-scoring output. The IFBench version and commit come
from the setup log. Assumed, not verified: that `docs/quality.md`'s 78.3 used a larger token budget. The page does
not say, and the 80.2% answered-only score is the evidence. The xhigh generation summary line was read from
`raw-ifbench-gen-xhigh-i1.log` at 17:27Z, before the restarted run truncated that file. It is quoted in `cmd.txt`.
