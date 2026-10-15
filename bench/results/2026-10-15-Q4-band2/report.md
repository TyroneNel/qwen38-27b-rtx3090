# Batch 5 Q4: IFBench, band 2 against the shipped profile (2026-10-15): stopped, no result

**Verdict: no result. The run was stopped on the owner's instruction, and nothing was scored.**

The owner's instruction, verbatim: "the IfBench is hitting token limit. results are void, stop skip IFBench".

## What ran

- **Band-2 boot (`q4b`):** `DFLASH_TOKENS=3`, `EXTRA_ARGS='--attention-backend TRITON_ATTN --kv-cache-dtype
  int8_per_token_head'`, `KV_MEM=5583457484`, `MAX_SEQS=16`, image `hyperqwen:exp-844cd90`.
- **IFBench install:** per boot from AllenAI IFBench at the pinned commit `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`
  (`raw-ifbench-setup-q4b.log`).
- **Generation:** docs/quality.md's protocol (thinking on, `reasoning_effort` omitted = template default xhigh,
  model-default sampling, `max_tokens` 16384), 4 client threads (`cmd.txt`).

The generation was killed at 2026-10-15T19:01Z (`cmd.txt`), 1 h 23 min after it started at 17:38Z. The counters at the stop
(`raw-ifbench-metrics-at-stop-q4b.log`):

```
vllm:num_preemptions_total 0.0
vllm:request_success_total{finished_reason="stop"} 180.0
vllm:request_success_total{finished_reason="length"} 24.0
vllm:generation_tokens_total 1.325558e+06
```

- **Truncation:** 24 of 204 finished requests (11.8%) ended at the 16,384-token cap. The responses were held in the
  generator's memory and written only at the end, so none were saved or scored.
- **Shipped arm:** the shipped-profile boot (`Q4-shipped`) never started.
- **Preemptions:** none, at 4 client threads on the band-2 pool.

## What is left open

The precision gate for band 2 is still GSM8K n=200 and perplexity only (batch 2 S1). A rerun needs a token budget that
xhigh thinking does not hit, or a lower effort on both arms. That is a protocol decision, not something this batch
decided.

## Verified vs assumed

**Verified:** the counters and the stop time are quoted from this directory.

**Not verified:** any IFBench score.
