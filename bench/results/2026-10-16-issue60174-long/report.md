# vllm#60174 check on the int8 `CTX=long` profile

**Verdict: NOT AFFECTED.** Turns 2-5 of every warm pass had real prefix-cache hits: 19,008 or 19,872 cached tokens,
which is 22 or 23 blocks of 864. Every reply is coherent. No reply showed the collapse: the longest single-token run
is 2, and the lowest distinct-word ratio on a cache-hit turn is 0.627. Warm acceptance length is within 10.1% of
cold on every pass:

| pass | acceptance length, all turns | vs cold-T0 |
|---|---|---|
| warm-T0 | 4.18 | -0.2% |
| warm-T1 | 3.76 | -10.1% |
| warm-T0-repeat | 4.23 | +1.0% |
| cold-T0 | 4.19 | - |

Warm-T0 and its cold-T0 twin gave identical replies on 4 of 5 turns. On turn 4 the two replies differ from
character 0, and both are coherent.

This follows the bf16 run in `../2026-10-16-issue60174-prefix-cache/`. That run's brief, rules and replay are reused
here; only the profile changed.

## Preflight

| fact | value |
|---|---|
| original branch | `cline/gf59cdmh` (behind origin by 1), untracked owner files only |
| test branch | `issue-60174-long` from `origin/issue-60174-check` |
| git user.name | `TyroneNel` |
| image | `sha256:e99df7e6dc73208971ae0d1f1063deb100c997308248f253174176cf0514cb9e` `["ghcr.io/syv-ai/hyperqwen:latest"]` (vLLM 0.31.0) |
| GPU | RTX 3090 `GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84`, **250 W** limit (not changed), 0 MiB used, no compute apps |
| 3090 adapter LUID | `luid_0x00000000_0x0001144f_phys_0` (idle 4.0 MiB) |
| containers | no hyperqwen container present |
| env | `.env.L` = `../2026-10-16-dynamic-k-ab/.env.A` with `CTX=fast` changed to `CTX=long`. `KV_MEM`, `MAX_LEN` and `MAX_SEQS` are unset. |
| replay script | `replay_60174.py` is byte-identical to the bf16 run's copy, which includes the `list(messages)` fix. The profile forced no change. |

## Boot facts (label L)

- `docker compose up` at 11:02:56Z. KV-cache line at 11:05:07Z. **Healthy after 142 s.**
- Model loading took 15.39 GiB and 92.8 s. Graph capture took 2 s and 0.10 GiB.
- **Free VRAM after boot, before any request: 1,587 MiB** (22,740 MiB used).
- **Spill:** F_ctx = 84.2 MiB and F_kv = 92.2 MiB, an 8.0 MiB step, so no boot rise. Max shared is 92.2 MiB and
  max dedicated is 23,293.8 MiB, above 20,000. No sample after the first request was above F_kv + 16 MiB. There is no
  `SPILL-L` file, and `arm.sh spilled` returned 1. **Not spilled.**

### argv (`raw-cmdline-L.log`): every expected argument is present

`--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head`, `--max-model-len 131072`,
`--max-num-seqs 4`, `--kv-cache-memory=5583457484`, `--enable-prefix-caching --mamba-cache-mode align`,
`--speculative-config {"method":"dflash","model":"/app/models/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`.

### Boot-log lines (`raw-boot-L.log`, line numbers in that file)

- 79: `Setting attention block size to 864 tokens to ensure that attention page size is >= mamba page size.`
- 119: `kv cache group sizes [864, 864, 864, 864, 864, 864, 864, 864, 864]`
- 120: `kv lcm block sizes 864`
- 96: `Not promoting draft block sizes: no divisor of the primary block 864 is >= layer language_model.model.layers.3.self_attn.attn's covering block 1728.`
- 50 and 70: `int4 3D scratch: capacity=64 query tokens (min of max_num_batched_tokens=2048, min(max_num_seqs=4, seq_threshold_3D=32) * max_query_len_3d=16)`.
  Line 50 reports 396288 B/row, 24.19 MiB. Line 70 reports 528384 B/row, 32.25 MiB. Lines 101-104 read
  `capacity=64 query tokens adopted by the metadata builder`.
- 100: `GPU KV cache size: 136,429 tokens, Maximum concurrency for 131,072 tokens per request: 1.04x`

### The drafter's KV dtype

**The log does not state the drafter's KV cache dtype.** The only dtype lines are engine-wide:
`kv_cache_dtype=int8_per_token_head` in the engine config (line 34) and `'kv_cache_dtype': 'int8_per_token_head'` in
the non-default args (line 18). The lines about the drafter:

- 26: `Resolved architecture: DFlash2DraftModel`
- 28: `Overriding draft model max model len from 262144 to 131072`
- 34: `speculative_config=SpeculativeConfig(method='dflash', model='/app/models/Qwen3.8-27B-DFlash2-W4A16', num_spec_tokens=7)`
- 40: `DFlash2 lookup-augmented drafting on (k=7 nmin=6 nmax=12 nstrong=6 agree=0 nmin_tail=4 longmin=6 search=1073741824)`
- 69-70: `Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN']`, then the second `int4 3D scratch`
  line at 528384 B/row. These come right after the target's weights load and right before the 1.19 GiB drafter
  checkpoint loads, so they are probably the drafter's attention `[INFERENCE]`. Neither line names a dtype.
- 96: the `Not promoting draft block sizes` line above.
- 97: `Sliding-window bucket(s) are the smallest; using group_size 8 (pads the sliding-window group instead of the full-attention/mamba layers)`
- 99: `WARNING … Speculative decoding (method=dflash) is enabled but no KV cache group could be identified as the draft model's.`
- 111: `Warmed rejection-sampler kernels (k=7, draft_logits=True)`
- 114: `Capturing model for DFlash2 speculator...`

## Replay

This is the same client and the same conversation as the bf16 run: a 44-section synthetic spec of 20,264 tokens and
5 turns per pass, with `max_tokens` 256, thinking off, non-streaming and one request at a time. Each pass has its own
document header. cold-T0 replayed warm-T0's exact bodies in the normal pass order, each with a unique `cache_salt`.
The prompt token counts match warm-T0 on every turn. The raw data is in `raw-replay-L.json` and `raw-replay-L.log`.

| pass | turn | prompt tok | cached tok | acc len | longest token run | longest word run | distinct ratio | finish | first diff vs cold |
|---|---|---|---|---|---|---|---|---|---|
| warm-T0 | 1 | 20,391 | 0 | 4.53 | 1 | 1 | 0.469 | stop | identical |
| warm-T0 | 2 | 20,645 | 19,008 | 4.53 | 2 | 1 | 0.725 | stop | identical |
| warm-T0 | 3 | 20,855 | 19,008 | 4.58 | 1 | 1 | 0.860 | stop | identical |
| warm-T0 | 4 | 21,002 | 19,872 | 4.49 | 1 | 1 | 0.627 | length | char 0 |
| warm-T0 | 5 | 21,299 | 19,872 | 3.09 | 1 | 1 | 0.824 | stop | identical |
| cold-T0 | 1 | 20,391 | 0 | 4.53 | 1 | 1 | 0.469 | stop | - |
| cold-T0 | 2 | 20,645 | 0 | 4.53 | 2 | 1 | 0.725 | stop | - |
| cold-T0 | 3 | 20,855 | 0 | 4.58 | 1 | 1 | 0.860 | stop | - |
| cold-T0 | 4 | 21,002 | 0 | 4.69 | 2 | 1 | 0.829 | length | - |
| cold-T0 | 5 | 21,299 | 0 | 2.98 | 1 | 1 | 0.824 | stop | - |
| warm-T1 | 1 | 20,389 | 0 | 3.95 | 1 | 1 | 0.443 | stop | - |
| warm-T1 | 2 | 20,679 | 19,008 | 3.60 | 2 | 1 | 0.640 | stop | - |
| warm-T1 | 3 | 20,941 | 19,008 | 3.90 | 1 | 1 | 0.847 | stop | - |
| warm-T1 | 4 | 21,100 | 19,872 | 4.62 | 2 | 1 | 0.821 | length | - |
| warm-T1 | 5 | 21,397 | 19,872 | 2.78 | 1 | 1 | 0.828 | stop | - |
| warm-T0-repeat | 1 | 20,391 | 0 | 4.46 | 1 | 1 | 0.472 | stop | - |
| warm-T0-repeat | 2 | 20,646 | 19,008 | 4.57 | 2 | 1 | 0.750 | stop | - |
| warm-T0-repeat | 3 | 20,899 | 19,008 | 4.70 | 1 | 1 | 0.816 | stop | - |
| warm-T0-repeat | 4 | 21,045 | 19,872 | 4.48 | 1 | 1 | 0.763 | length | - |
| warm-T0-repeat | 5 | 21,342 | 19,872 | 3.14 | 1 | 1 | 0.781 | stop | - |

- Warm-T0 turn 4 starts with "Section 34 does not describe a single, unified step-by-step…". Its cold twin starts
  with "Based on Section 34, the document specifies the following fa…". Both are coherent lists of the section's
  rules, and both hit the 256-token limit. Their acceptance lengths are 4.49 and 4.69 (-4%). Every warm/cold pair is
  within ±25%.
- On this profile the cached prefix is shorter than in the bf16 run (19,008/19,872 against 19,712-20,608 tokens),
  because blocks are 864 tokens here and hits only count whole blocks.
- Prefill is slower than on bf16, as expected. A cold 20k turn took 27.4-30.3 s (bf16: 20.4-22.9 s). A turn with a
  cache hit took 3.7-5.0 s.

## Against the bf16 run

The bf16 values come from `origin/issue-60174-check:bench/results/2026-10-16-issue60174-prefix-cache/raw-replay-P.json`.

| pass | bf16 acc all | int8 acc all | bf16 acc t2-5 | int8 acc t2-5 | bf16 cached t2-5 | int8 cached t2-5 | bf16 min distinct (hit turns) | int8 min distinct (hit turns) |
|---|---|---|---|---|---|---|---|---|
| warm-T0 | 4.22 | 4.18 | 4.16 | 4.08 | 19,712 / 20,160 / 20,160 / 20,160 | 19,008 / 19,008 / 19,872 / 19,872 | 0.750 | 0.627 |
| cold-T0 | 4.60 | 4.19 | 4.65 | 4.09 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | - (no hits) | - (no hits) |
| warm-T1 | 4.09 | 3.76 | 4.17 | 3.71 | 19,712 / 20,160 / 20,608 / 20,608 | 19,008 / 19,008 / 19,872 / 19,872 | 0.576 | 0.640 |
| warm-T0-repeat | 4.42 | 4.23 | 4.41 | 4.16 | 19,712 / 20,160 / 20,160 / 20,160 | 19,008 / 19,008 / 19,872 / 19,872 | 0.605 | 0.750 |

Note on the bf16 cold-T0 row: the bf16 cold pass is a rerun made after a client bug. Its value of 4.60 is high mainly
because its turn 4 took a different wording (6.24). On int8, warm and cold are about equal, so the hits cost nothing
in acceptance.

## The trigger and this result

The issue thread puts the trigger in the drafter's fp8 KV: fp8 has no scales, so they default to 1.0 and the values
saturate. `int8_per_token_head` computes an fp32 scale per token and per head, so that saturation mechanism has no
equivalent here `[INFERENCE]`. The result fits that: with a quantized KV cache on this profile and real cache hits on
every follow-up turn, the drafter kept accepting normally (lowest 2.78, about the same as cold) and every reply was
coherent. This run cannot show that the drafter itself used int8 KV. The log never names the drafter's KV dtype, and
line 99 says no KV cache group was identified as the draft model's.

The vllm#55601 unit bug could not trigger here: every group's block (864) equals the attention/mamba block. The brief
also states the image already carries the fix (`mamba_hybrid.py:123`). This run did not check that.

## Deviations from the brief

- No deviations in the run. One boot. The four passes ran in their normal order, and cold-T0 used warm-T0's exact
  bodies, with matching prompt token counts.
- Prompts grew from 20.4k to 21.3k tokens, not to the brief's ~21/22/23/24k. The conversation construction is the
  same as in the bf16 run, as the brief requires, and that construction cannot add ~1k tokens per turn.
- `arm.sh exec "$D" L 'true'` set the guard's first-request marker before the replay ran. `.firstreq-replay-L` is
  left over from the `replay-L` exec call.
