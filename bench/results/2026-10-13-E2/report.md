# E2: attention backend x KV dtype at equal context

RTX 3090, 250 W (`power.limit 250.00 W` in every `raw-smi-*.log`), container boot.

## Arms and what the engine got

Every arm uses `SPEC=dflash2 PREFIX_CACHE=1 CTX=fast DFLASH_TOKENS=7 MAX_LEN=65536 KV_MEM=5583457484`. The
arms differ only in `EXTRA_ARGS` (see `.env.<label>`). Arm A is the shipped `CTX=fast` path. Its numbers are
E1's two shipped boots, `../2026-10-13-E1/` A and A2, which used the identical `.env` minus `VERIFY=0`.
Running a third identical boot would have repeated them. The A smoke ran on the E3 boot, which used the same
shipped `.env` (`../2026-10-13-E3/raw-gsm-A.log`).

| arm | asked for | engine's args (`non-default args`) | selected backend (boot log) | CUDA graphs captured | KV pool |
|---|---|---|---|---|---|
| A | shipped: `FLASH_ATTN` + bf16 | n/a | `Using AttentionBackendEnum.FLASH_ATTN backend.` | PIECEWISE + FULL + dflash2 FULL | 68,605 tokens |
| B | `TRITON_ATTN` + `int8_per_token_head` | `'attention_backend': 'TRITON_ATTN'`, `'kv_cache_dtype': 'int8_per_token_head'` | `Using AttentionBackendEnum.TRITON_ATTN backend.` | PIECEWISE + FULL + dflash2 FULL | 116,440 tokens |
| C | `FLASHINFER` + fp8 | `'FLASHINFER'`, `'fp8'` | `Using AttentionBackendEnum.FLASHINFER backend.` | **PIECEWISE only** | 84,260 tokens |
| D | `FLASHINFER` + bf16 | `'FLASHINFER'`, `'bfloat16'` | target `FLASHINFER`; the second selection is `Using FLASH_ATTN attention backend out of potential backends: [...]` | **PIECEWISE only** | 68,605 tokens |

In D, the second selection (`cuda.py:592`) picked `FLASH_ATTN`. In A, B and C it matched the requested
backend. The target layers ran FlashInfer. That the second selection is the drafter's is an inference I did
not verify.

**Silent config downgrade, C and D.** Both boot logs carry this line:

```
WARNING ... [compilation.py:1462] CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE
```

A and B capture 13 PIECEWISE, 10 FULL and 3 dflash2 FULL graph progress bars. C and D capture the 13
PIECEWISE only. The backend engaged, and the decode step lost its full graph.

## Results

Metrics:

- **C1 decode**: `bench/run_benchmarks.sh single`, row `cohort C1 real prompts T=0`, decode(C/meanTPOT), second
  run of the boot.
- **tok/step**: the same row's `tok/step`, from the server's spec-decode counters, i.e. `bench/harness.py spec`.
- **ms/pass**: `bench/conc_ladder.py --n 1,8`.

| metric | A (E1 A / A2) | B triton+int8 | C flashinfer+fp8 | D flashinfer+bf16 |
|---|---|---|---|---|
| C1 decode T=0, tok/s | 109.4 / 109.4 | 111.4 | 78.2 | 80.6 |
| delta vs A | n/a | **+1.8%** | **-28.5%** | **-26.3%** |
| C1 tok/step T=0 | 3.33 / 3.33 | 3.34 (+0.3%) | 3.31 (-0.6%) | 3.25 (-2.4%) |
| C1 decode T=default | 99.6 / 107.2 | 103.8 | 73.2 | 75.8 |
| ladder N=1 ms/pass | 31.8 / 31.4 | 31.7 | 41.7 | 41.1 |
| ladder N=8 ms/pass | 52.6* / 54.4* | 46.1* | 43.8* | 53.5* |
| ladder N=8 decode agg | 213.4* / 242.0* | 181.9* | 149.6* | 239.6* |
| ladder N=8 preemptions / resident | 20 / 5, 21 / 5 | 1 / 6 | 0 / 3 | 23 / 5 |
| cohort C8 decode T=0 | 248.8 / 249.9 | 368.8 | 311.0 | 332.0 |
| GSM8K smoke, n=50 | 0.980 (`raw-gsm-A.log` in E3) | 0.960 (`raw-gsm-Bq.log`) | not run | not run |
| free VRAM after boot (`nvidia-smi`) | 1,471 / 1,431 MiB | 1,359 MiB | 1,631 MiB | 2,101 MiB |

`*` marks a ladder tail row ("no instant at which all N streams were decoding"). Those rows are not steady
state. A/A2 differ by 13% at N=8. The cohort C8 row has a 37% spread inside one shipped boot (E1 A run 1 read
340.9, run 2 read 248.8), so the C8 column is not evidence for anything.

## Verdicts

- **B: no difference on C1** (+1.8%, under 3%). tok/step is within 2% and the pass bar is 5% on C1, so B is
  not a winner. It holds a 70% larger pool in the same bytes (116,440 against 68,605 tokens), and at N=8 it
  preempted once against 20 times. That is capacity, not decode speed. The GSM8K smoke read 0.960 against
  0.980. That is one question in 50 (2 points), above the 1-point bound, and n=50 cannot resolve a 1-point
  difference. Not verified either way.
- **C: fail.** -28.5% C1. ms/pass at N=1 rises from 31.8 to 41.7 with the same tok/step. Cause seen in the
  log: cudagraph mode downgraded to PIECEWISE.
- **D: fail.** -26.3% C1. Same downgrade.

## Shared Usage on the FlashInfer boots

The shipped and Triton boots hold the 3090's Shared Usage at 84-92 MiB. C and D jumped at the end of the
boot, before any request, and stayed flat through every measurement:

- C: 220.2 MiB.
- D: 172.2 MiB.

`mem.txt` flags every such sample as a possible spill. The brief says not to measure in that state, but to cut
`KV_MEM` until it is clean. I re-booted C with `KV_MEM=4509715660` (1 GiB less) as arm **C2**:

- Free VRAM after boot rose to 2,687 MiB. Shared Usage still went to 220.2 MiB and stayed there.
- C1 T=0 read 81.6 tok/s against C's 78.2 (+4%). ms/pass at N=1 read 42.0 against 41.7.

With 2.7 GB of the card free, the 128 MiB is not overflow from the working set. It is a fixed host-side
allocation of the FlashInfer path. FlashInfer page-locks a host workspace, but I have not verified that this
is that allocation. C's -28.5% is the graph downgrade, not a spill. Strictly, by the brief's rule, C and D
were measured with Shared Usage above baseline. I record them as "fail" with this caveat rather than as
"spilled", because cutting 1 GiB of pool did not move the counter.

## Contradicts the landscape page

`docs/decode-speed-landscape.md` G3 reads the 2x3090 reproduction as a 1.7x gain: 90.9 tok/s on the
fp8/FlashInfer path against 156.0 with `TRITON_ATTN` + `int8_per_token_head`. It calls this "a 1.7× gap between
two configurations we already ship". On one 3090 at 250 W, the ordering of those two holds: Triton+int8 at
111.4 against FlashInfer+fp8 at 78.2 is 1.42x. But the shipped single-user default (`FLASH_ATTN` + bf16, 109.4)
is already level with Triton+int8 (+1.8%). There is no 1.4-1.7x left to collect on `CTX=fast`. The gap is
FlashInfer's spec-decode cudagraph downgrade, not a fast path we are missing.

## Verified vs assumed

Verified: the args each engine received, the backend selection lines, the downgrade warning, the graph
progress bars, the pool sizes, every number above from the committed raw logs, and the Shared Usage traces.
Assumed, not verified: that the second backend selection in D is the drafter's, and that the FlashInfer
shared-memory step is its pinned workspace.
