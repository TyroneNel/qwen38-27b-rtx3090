# drafter-2: offline DFlash2 gate (`drafter/eval_dflash2.py`)

| Field | Value |
|---|---|
| Date | 2026-10-23 18:04–19:04 SAST |
| Tools | `drafter/capture_dflash2.py` (target taps from the `-fast` W4A16 target, in-process vLLM), `drafter/dflash2_ref.py` (reference DFlash2 forward, one code path for both drafters), `drafter/eval_dflash2.py` (tau_chain/tau_anchor/acc@k, paired ratio with CI, `--check-known`, negative control, half-batch repeat) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`sha256:0cf71b42d301…`, main `ce9615c`), stock |
| Capture | 9 domains (code, danish, chat, think, gsm8k_think, gsm8k_nothink, labd, real_think, real_nothink), 12 prompts each, greedy and default sampling: 188 sequences, 161,358 rows, 7.7 GiB; greedy top-1 against the reference 0.9965 |
| Drafters | `Qwen3.8-27B-DFlash2` (bf16, HF revision `015e7956…`) and the shipped `Qwen3.8-27B-DFlash2-W4A16` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\drafter-2\{cap,known,nothink}\`; provenance study `drafter-2\bf16_gap.md` |

## Check against the known numbers (real_think, `drafter/README.md`)

| Check | Measured | Known window | |
|---|---|---|---|
| bf16 tau_chain, default sampling | 3.165 | 3.332–3.468 ("3.4") | FAIL |
| W4A16 tau_chain, default sampling | 3.145 | 3.136–3.264 ("3.2") | ok |
| bf16 tau_chain, greedy | 3.305 | 3.469–3.611 ("3.54") | FAIL |
| W4A16 tau_chain, greedy | 3.295 | 3.273–3.723 | ok |
| sampling gap W4A16/bf16 | 0.9936 (CI95 0.9918–0.9951) | 0.922–0.960 | FAIL |
| greedy parity W4A16/bf16 | 0.9970 | 0.925–1.052 | ok |
| repeat spread (half batch) | 0.21% | < 0.5% | ok |
| negative control (target rows shifted by one) | tau 1.040 vs 3.165 | must collapse | ok |

real_nothink (informative; the known numbers are think-mode): bf16 3.646 / W4A16 3.628 sampling, ratio 0.9949 (CI95 0.9937–0.9962); greedy parity 0.9898; repeat spread 0.62%.

## Paired W4A16/bf16 tau_chain ratio over all domains

| Domain | Greedy | Default sampling |
|---|---|---|
| ALL | 0.9970 [0.9925, 1.0014] | 0.9944 [0.9936, 0.9952] |
| chat | 0.9952 | 0.9938 |
| code | 0.9828 | 0.9948 |
| danish | 1.0057 | 0.9948 |
| gsm8k_nothink | 0.9887 | 0.9947 |
| gsm8k_think | 0.9847 | 0.9926 |
| labd | 1.0000 | 0.9981 |

## Why the bf16 numbers miss

The tool can't pull one drafter down on its own:
- Both drafters run through one forward (`dflash2_ref.py:516-552`); only the weight load differs (bf16 loaded directly, W4A16 dequantized), and the loader is strict about names and shapes.
- The W4A16 checkpoint is a GPTQ of exactly this bf16 checkpoint. All 45 unquantized tensors are byte-identical, and the dequantized int4 linears sit at cosine 0.986–0.992 with relative error 0.13–0.17 (plain g128 rounding gives 0.12).

The weak point is the reference. The bf16 "3.4 / 3.54" numbers and the 0.941 gap are single, unpaired live vLLM runs from 2026-08-19 (commit `e4546d0`), and they were never re-measured. The repo itself puts single-run default sampling at ±5% (`drafter/README.md:164-168`), and W4A16 alone has read 3.17–3.4 in live runs. A ratio of two unpaired single runs carries about ±7%, so 0.941 and 0.994 differ by less than one standard deviation.

## Verdict: INFO

The tool is validated where it matters: W4A16 matches its published numbers, the negative control collapses, and repeats agree within 0.21%. The published bf16 gap does **not** reproduce. Paired on one capture, the W4A16 drafter costs 0.6% tau_chain at default sampling and 0.3% at greedy, not ~6%. Paired comparisons against W4A16 (quant-3, drafter-3) are valid with this tool. The roadmap's drafter-3 motivation ("close a ~5% gap", +3% bar, +6% ceiling) does not hold: matching bf16 is worth about +0.6%.
