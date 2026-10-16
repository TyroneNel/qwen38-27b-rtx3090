# vllm#60174 check: DFlash2 + prefix caching on the default config

**Verdict: NOT AFFECTED.** Turns 2-5 of all three warm passes got real prefix-cache hits (19,712-20,608 cached
tokens). Every reply is coherent. No repeated-token collapse showed up: the longest single-token run is 2 and the
lowest warm distinct-word ratio on a cache-hit turn is 0.576. Pass-level warm acceptance length is 4.09-4.42 against
cold 4.60, a 4-11% gap, inside ±25%. Warm-T0 and its cold-T0 twin produced identical replies on 4 of 5 turns; turn 4
forked at character 75.

[vllm#60174](https://github.com/vllm-project/vllm/issues/60174): after a prefix-cache hit, DFlash2 emits one
repeated token (`duct`) and accepts nothing (`Mean acceptance length: 1.00`) on vLLM 0.30/0.31.

## Preflight

| fact | value |
|---|---|
| original branch | `cline/gf59cdmh` (behind origin by 1), untracked owner files only |
| test branch | `issue-60174-check` from `origin/dynamic-k-results` |
| git user.name | `TyroneNel` |
| image | `ghcr.io/syv-ai/hyperqwen:latest` = `sha256:e99df7e6dc73208971ae0d1f1063deb100c997308248f253174176cf0514cb9e` (vLLM 0.31.0, `raw-image-P.log`) |
| GPU | RTX 3090 `GPU-2e8e4d29-ecb0-5fbf-7a84-d2370311be84`, power limit **250 W** (recorded, not changed), 0 MiB used, no compute apps |
| 3090 adapter LUID | `luid_0x00000000_0x0001144f_phys_0` (idle 4.0 MiB; display 3080 Ti `…0x0001246f…` at 534 MiB) |
| containers | no hyperqwen container present |
| env | `.env.P` = `2026-10-16-dynamic-k-ab/.env.A`: `SPEC=dflash2 PREFIX_CACHE=1 CTX=fast DFLASH_TOKENS=7`, `MODEL=/app/models/Qwen3.8-27B-W4A16-AutoRound-fast`, port 18020 |
| override | `docker-compose.override.yml`: image pinned, `pull_policy: never`, branch `start_qwen.sh` mounted read-only (checked with `docker compose --profile single config`) |

## Boot facts (label P)

- Up at 10:44:08Z. KV-cache line at 10:46:19Z (`GPU KV cache size: 68,605 tokens, Maximum concurrency for 65,536
  tokens per request: 1.05x`). **Healthy after 142 s.**
- Model loading took 15.33 GiB. Graph capture took 5 s and 0.13 GiB.
- **Free VRAM after boot, before any request: 1,477 MiB** (22,850 MiB used of 24,576; `raw-smi-P.log`).
- **Spill:** floors F_ctx = 84.2 MiB and F_kv = 92.2 MiB. That is an 8.0 MiB step, the normal MAX_SEQS=8 step, so no
  boot rise. Max shared = 92.2 MiB. Max dedicated = 23,415.8 MiB, above 20,000. No sample after the first request rose
  above F_kv + 16 MiB. There is no `SPILL-P` and `arm.sh spilled` returned 1. **Not spilled.**

### argv checks (`raw-cmdline-P.log`, `raw-boot-P.log`)

- `--kv-cache-dtype bfloat16` (with `--attention-backend FLASH_ATTN`): yes.
- `--enable-prefix-caching --mamba-cache-mode align`: yes.
- `--speculative-config {"method":"dflash","model":"/app/models/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`: yes.
- `--max-model-len 65536`, `--max-num-seqs 8`, `--enable-prompt-tokens-details`, `--kv-cache-memory=5583457484`.
- Boot log: `kv cache group sizes [448, 448, 448, 448, 448, 448, 448, 448, 448]` and `Setting attention block size
  to 448 tokens to ensure that attention page size is >= mamba page size`. Every prefix-cacheable group's block
  equals the 448-token mamba block, so the unit bug fixed in vllm#55601 (group block smaller than `mamba_block_size`)
  cannot occur here.

## Replay

`replay_60174.py` ran in the container through `arm.sh exec` and `bench/harness.py`. `.env.P` sets no
`VLLM_API_KEY` and there is no `api_key.txt`, so the server bound no key and the harness sent `EMPTY`. The document
is a synthetic storage-system spec: 44 sections, 20,264 tokens, no sentence repeated. Each chat has 5 turns: system
prompt + document + question, then on each later turn the previous messages, the model's reply and a new question.
Requests were sent one at a time with `max_tokens` 256, `enable_thinking: false`, non-streaming. Each pass had its
own document header. The cold-T0 pass resent warm-T0's exact bodies, each with a unique `cache_salt`; the server
accepted `cache_salt`. Its prompt token counts match warm-T0 on every turn. Acceptance length = Δaccepted/Δdrafts + 1
from `/metrics` around each request. The two run columns give the longest run of one repeated token id (tokenized
reply) and of one repeated word. Distinct ratio = unique words / words. Raw data: `raw-replay-P.json`,
`raw-replay-P.log` (run 1), `raw-replay-P-cold.log` (cold rerun).

| pass | turn | prompt tok | cached tok | acc len | longest token run | longest word run | distinct ratio | finish | first diff vs cold |
|---|---|---|---|---|---|---|---|---|---|
| warm-T0 | 1 | 20,386 | 0 | 4.46 | 1 | 1 | 0.472 | stop | identical |
| warm-T0 | 2 | 20,641 | 19,712 | 5.00 | 2 | 1 | 0.750 | stop | identical |
| warm-T0 | 3 | 20,894 | 20,160 | 4.70 | 1 | 1 | 0.816 | stop | identical |
| warm-T0 | 4 | 21,040 | 20,160 | 4.20 | 2 | 1 | 0.766 | length | char 75 |
| warm-T0 | 5 | 21,337 | 20,160 | 3.12 | 1 | 1 | 0.800 | stop | identical |
| cold-T0 | 1 | 20,386 | 0 | 4.46 | 1 | 1 | 0.472 | stop | - |
| cold-T0 | 2 | 20,641 | 0 | 5.00 | 2 | 1 | 0.750 | stop | - |
| cold-T0 | 3 | 20,894 | 0 | 4.70 | 1 | 1 | 0.816 | stop | - |
| cold-T0 | 4 | 21,040 | 0 | 6.24 | 1 | 1 | 0.605 | stop | - |
| cold-T0 | 5 | 21,337 | 0 | 3.19 | 1 | 1 | 0.800 | stop | - |
| warm-T1 | 1 | 20,389 | 0 | 3.88 | 1 | 1 | 0.418 | length | - |
| warm-T1 | 2 | 20,686 | 19,712 | 4.43 | 2 | 1 | 0.576 | length | - |
| warm-T1 | 3 | 20,981 | 20,160 | 4.42 | 1 | 1 | 0.830 | stop | - |
| warm-T1 | 4 | 21,134 | 20,608 | 5.66 | 1 | 1 | 0.606 | stop | - |
| warm-T1 | 5 | 21,407 | 20,608 | 2.78 | 1 | 1 | 0.794 | stop | - |
| warm-T0-repeat | 1 | 20,390 | 0 | 4.46 | 1 | 1 | 0.472 | stop | - |
| warm-T0-repeat | 2 | 20,645 | 19,712 | 4.67 | 2 | 1 | 0.750 | stop | - |
| warm-T0-repeat | 3 | 20,898 | 20,160 | 4.32 | 1 | 1 | 0.816 | stop | - |
| warm-T0-repeat | 4 | 21,044 | 20,160 | 6.42 | 1 | 1 | 0.605 | stop | - |
| warm-T0-repeat | 5 | 21,297 | 20,160 | 2.92 | 1 | 1 | 0.791 | stop | - |

| pass | acceptance length, all turns | turns 2-5 | vs cold-T0 (all turns) |
|---|---|---|---|
| warm-T0 | 4.22 | 4.16 | -8% |
| cold-T0 | 4.60 | 4.65 | - |
| warm-T1 | 4.09 | 4.17 | -11% |
| warm-T0-repeat | 4.42 | 4.41 | -4% |

- Cached tokens are whole 448-token blocks: 44, 45 or 46 blocks. A hit stops at the last block boundary of the
  earlier prefix, as `--mamba-cache-mode align` implies.
- Warm-T0 turn 4 forked at character 75 ("…failure handling **and operational behaviors**…" against cold's
  "…failure handling **steps:**…"). The warm reply then ran to the 256-token limit, and both replies are coherent.
  That turn's acceptance gap, 4.20 against 6.24 (-33%), compares two different texts. It is the only per-request pair
  outside ±25%. Warm-T0-repeat took cold's branch on turn 4 (same text, 6.42). A T=0 fork after a hit is the
  numerics-level divergence the brief allows.
- The lowest distinct ratios are on turn 1 (0.418-0.472), which has no cache hit. Those replies quote document
  sentences back and are coherent. The lowest acceptance length in the run is 2.78, far from the issue's 1.00.

## The issue's trigger and this result

According to the brief's summary of the thread, the trigger is the DFlash2 drafter inheriting
`--kv-cache-dtype fp8` with no scales. Setting only the drafter's KV back to bf16 makes the output clean. This boot
runs bf16 KV for target and drafter alike (`--kv-cache-dtype bfloat16`), with 448-token blocks in every group. It got
real cache hits on every follow-up turn and stayed coherent with normal acceptance. That fits the fp8-drafter-KV
diagnosis: without fp8 KV, the failure does not appear on this config. It does not prove that diagnosis, because no
fp8 arm was run.

**Not tested:** the int8 `CTX=long` profile (`--kv-cache-dtype int8_per_token_head`, TRITON_ATTN). The drafter may
inherit a quantized KV dtype there, as in the issue, and that profile was not run.

## Deviations from the brief

- **The cold-T0 control ran twice; the first run was invalid.** A client bug saved every request body with a
  reference to the one growing message list. The first cold pass therefore sent the final turn-5 conversation five
  times (21,498 prompt tokens each; kept as `cold-T0-invalid-run1` in the JSON and visible in `raw-replay-P.log`). I
  fixed the bug (`list(messages)`) and reran only cold-T0 in the same boot with `--cold-rerun`, after
  warm-T0-repeat. That run rebuilt warm-T0's exact bodies (prompt tokens equal on every turn) and sent each with a new
  unique `cache_salt`, so the order change cannot create cache hits. No second boot.
- **Prompt growth:** prompts grew about 250-300 tokens per turn, from 20.4k to 21.3k, not to ~21/22/23/24k. The
  conversation was built exactly as specified (previous messages + a reply of at most 256 tokens + a short question),
  and that construction cannot add ~1k tokens per turn.
- **Spill-guard marker:** `arm.sh exec "$D" replay-P` writes `.firstreq-replay-P`, not `.firstreq-P`. I ran
  `arm.sh exec "$D" P 'true'` first so the guard's P marker was set at 12:47:29 local, before any replay request.
  `.firstreq-replay-P` and `.firstreq-replay-P-cold` are left over from the two exec calls.
