# S3a: our stack at r0b0tlab's GSM8K protocol

RTX 3090, 250 W (`power.limit 250.00 W` in `raw-smi-a1.log`), container boot, image `hyperqwen:exp-f1a033e`.

## Their protocol, as published

From `r0b0tlab/qwen38-exl3-dflash2` (`scripts/acceptance_check.py` and `notes/ACCEPTANCE.md`, read 2026-10-14):

- The first `--n 40` GSM8K test questions, greedy (argmax), `max_new_tokens 512`, ctx 8192.
- The prompt is the literal string `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n`, with no think block.
- `acceptance length = new_tokens / (new_tokens - accepted_draft_tokens)` and `tok/s = new_tokens / wall time of
  the generate call` (prefill included), both averaged per request.
- Published: **5.657 acceptance, 162.9 tok/s, n=40, 5/40 hit the cap.** Their power limit is not published.

## Our run

One boot of the shipped `CTX=fast` stack: `SPEC=dflash2 PREFIX_CACHE=1 DFLASH_TOKENS=7`, FLASH_ATTN + bf16 KV,
`MAX_LEN=8192`, `KV_MEM=5315022028` (pinned 256 MiB below the shipped value; see `knob-changes.txt`). Engine:
`max_model_len': 8192`, `'kv_cache_memory_bytes': 5315022028`, `GPU KV cache size: 27,714 tokens`. Tool:
`../2026-10-14-batch2-tools/r0b0_protocol.py`. Same first 40 questions, greedy, 512 max new tokens, one request at
a time. Their two means are computed per request from the server's spec-decode counters, which is exact at C1.
Two templates, two reps each:

| prompt | rep | their tok/s (incl. prefill) | decode tok/s (first→last token, mean) | their acceptance length | vLLM tok/step (aggregate) | hit cap |
|---|---|---|---|---|---|---|
| chat template, `enable_thinking: false` (the brief's row) | 1 | 199.9 | 218.1 | 5.893 | 5.817 | 6/40 |
| | 2 | 190.9 | 208.1 | 5.893 | 5.817 | 6/40 |
| **their literal raw prompt** | 1 | **174.7** | 187.7 | **5.619** | 5.575 | 9/40 |
| | 2 | **177.7** | 190.9 | **5.619** | 5.575 | 9/40 |
| r0b0tlab, published | — | 162.9 | — | 5.657 | — | 5/40 |

Both reps produced the same token count (chat 12,847, raw 13,453), so the outputs are identical and the rep-to-rep
difference (-4.5% chat, +1.7% raw) is timing noise on the same work.

Batch 1's single-question method (`workload_accept.py`, GSM8K question 0) at 512 tokens: `tok/step=5.795`,
identical to batch 1's 1024-token reading. The 226-token answer never reaches either cap. The free-run reps read
97.2 and 202.9 tok/s (`raw-wacc-a1.log`). The first is an outlier with no explanation in the log.

## Verdict

**On their exact prompt and metric, our stack reads 174.7 / 177.7 tok/s against their 162.9: +7.2% to +9.1%,
at 250 W.** Acceptance is level: 5.619 against 5.657 (-0.7%). The "ahead on the like-for-like workload" claim is
quotable in that form, with two caveats stated beside it:

- Their power limit is unpublished. Ours is 250 W, against a 350 W default.
- Both n=40 runs are one model and one prompt set. The rep-to-rep spread here is 1.7-4.5%, so the margin is
  ~2-5x our own noise, not more.

The brief's row (chat template, thinking off) reads higher, 190.9-199.9 tok/s and acceptance 5.89. It is not
like-for-like, because their raw prompt leaves the think block open and 9/40 answers hit the cap, against 6/40 on
the chat template. Quote the raw-prompt row against theirs, and the chat-template row only as our own number.

## Seen, not asked

- **`GPU KV cache size: 27,714 tokens` at `MAX_LEN=8192`**, against 63,872 tokens at `MAX_LEN=57344` for the same
  5,315,022,028 pinned bytes and the same bf16 stack (S3b's `raw-boot-b1.log`). The token count the engine reports
  falls with `max_model_len`. Not investigated. The run needs ≤ 8,192 tokens per request at C1, so the pool does
  not constrain it.
- Boot took 313 s (new `max_model_len`, so the compile cache missed).

## Memory

Free VRAM after boot 1,723 MiB. Shared Usage on the 3090 stayed at 82.2 MiB, its loaded floor, through the run
(`mem.txt`). Peak dedicated 22,761.8 MiB. No spill.

## Verified vs assumed

Verified: their protocol from their published script and notes, our engine's settings, and every number above
from `raw-r0b0-*-a1.log/.json` and `raw-wacc-a1.log`. Assumed, not verified: that their tok/s and ours time the
same span. Theirs times `gen.generate()` in-process; ours times an HTTP stream, which adds client and server
overhead. If anything, that biases against us.
