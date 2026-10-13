# E4: reasoning effort

RTX 3090, 250 W, container boot. One server with the shipped config (`.env.A`, the same as E1-A). Both
effort levels ran on that one boot. `mem.txt` shows Shared Usage peaking at 92.2 MiB, the loaded-engine floor,
so no spill.

## Does the served template accept it?

Yes. `raw-template-check.log` tokenizes one prompt with thinking on, at three effort settings, and prints the
template's effort line:

```
None 53 ['Reasoning effort is set to xhigh. Please think carefully through the task, ...']
low 41 ['Reasoning effort is set to low. Keep your thinking brief and focused, ...']
xhigh 53 ['Reasoning effort is set to xhigh. Please think carefully through the task, ...']
chat status ok, finish_reason stop
```

With no effort set, the template renders the same 53-token prompt as an explicit `xhigh`. So arm A ("default")
is the xhigh arm. A chat request with `reasoning_effort: low` returned 200 (`finish_reason stop`). The effort
line exists only with thinking on (`enable_thinking` undefined or true in `chat_template.jinja`), so both arms
run with thinking on, unlike E3.

## Method

The method is E3's (`../2026-10-13-tools/workload_accept.py`), with `--think 1`, the same three workloads, and
1,024-token greedy targets in 128-token chunks.

The experiment list asks for "the same teacher-forced sequences as E3". That cannot be done here. The effort
setting changes the prompt (53 against 41 tokens on "hi"), and it changes the output it is meant to change.
Forcing low-effort servers onto E3's thinking-off text would measure neither. Each arm therefore captured and
froze its own continuation (`targets-default/`, `targets-low/`) and is scored against it.

Run order on the boot: default-1, low-1 (captures, discarded), default-2, low-2 (kept). Each arm's tok/step
was identical to the third decimal between its two passes.

## Results (kept passes, `raw-wacc-default-2.log`, `raw-wacc-low-2.log`)

| workload | metric | A: default = xhigh | B: low | delta |
|---|---|---|---|---|
| gsm8k | teacher-forced tok/step (clean chunks) | 4.887 (2/3) | 5.143 (2/2) | +5.2% |
| gsm8k | free-run tok/step | 4.728 | 5.176 | +9.5% |
| gsm8k | target length (model stopped itself) | 383 tokens | 176 tokens | -54% |
| gsm8k | C1 decode, tok/s (2 reps) | 157.8 / 108.8 | 175.9 / 175.2 | n/a |
| code | teacher-forced tok/step | 3.992 (4/8) | 4.866 (7/8) | **+21.9%** |
| code | free-run tok/step | 3.559 | 4.858 | **+36.5%** |
| code | C1 decode, tok/s | 118.6 / 118.7 | 163.5 / 136.8 | +26% (mean) |
| chat | teacher-forced tok/step | 2.682 (5/8) | 3.629 (**1/8**) | +35.3%, on 35 steps only |
| chat | free-run tok/step | 2.649 | 3.042 | +14.8% |
| chat | C1 decode, tok/s | 88.9 / 89.1 | 102.1 / 92.2 | +9% (mean) |

## Verdict: pass on acceptance; the quality condition is not verified

`low` raises tokens per step by more than 10% on code (+21.9% teacher-forced, +36.5% free-run) and on chat
free-run (+14.8%). gsm8k rose 5.2% teacher-forced and 9.5% free-run. The direction matches club-3090's 5.362
against 3.872 (`docs/decode-speed-landscape.md` G2), at a smaller size on gsm8k.

The pass rule also requires "unchanged output quality on the same sequence". No quality smoke ran, because
this arm touches no kernel, dtype or backend (hard rule 1). Whether `low` answers as well as xhigh is
therefore **not verified**. On the one GSM8K question, the low arm wrote a 176-token answer against 383.
Whether both answers are right was not checked.

The chat teacher-forced row rests on 1 clean chunk out of 8 (35 steps). Read the chat free-run row instead.

## Seen but not asked

Thinking on (xhigh) costs acceptance against E3's thinking-off rows, same prompts and same boot config:

- gsm8k 4.887 against 5.795 (-15.7%).
- code 3.992 against 4.784 (-16.6%).
- chat 2.682 against 3.497 (-23.3%).

The single-user benchmarks are the slow end of the drafter whenever thinking is left at the template default.

## Verified vs assumed

Verified: the template renders and accepts `low` and `xhigh`, the default is xhigh, and every number above
comes from the raw logs and the frozen targets. Not verified: output quality at `low`. That one prompt per
workload is representative is assumed.
