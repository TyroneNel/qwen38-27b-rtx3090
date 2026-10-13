# E5: CUDA-graph capture size and mode

RTX 3090, 250 W, container boot. All arms `SPEC=dflash2 PREFIX_CACHE=1 CTX=fast DFLASH_TOKENS=7
KV_MEM=5583457484 MAX_LEN=65536`.

## Arms and what the engine got

- **A**: shipped. `CG` unset, resolved by the launcher to `min(MAX_SEQS*(DRAFT_TOKENS+1), 64) = 64`
  (`--compilation-config {"max_cudagraph_capture_size":64,...}`, `raw-cmdline-A.log` in E1). A's numbers are
  E1's two shipped boots A and A2, identical `.env`, same image.
- **B** (`CG=40`): `--compilation-config {"max_cudagraph_capture_size":40,...}`, `raw-cmdline-B.log`.
- **C** (`CG=40 CUDAGRAPH_MODE=FULL`): the same plus `"cudagraph_mode":"FULL"`. The GDN layers cannot run FULL,
  so the engine logged `CUDAGraphMode.FULL is not supported with GDNAttentionBackend backend (support:
  AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_AND_PIECEWISE` — the same mode A ends up with.

Graph progress bars at boot: A 13 PIECEWISE + 10 FULL + 3 dflash2 FULL; B 10 + 7 + 2; C 10 + 7 + 2. The
smaller capture set engaged.

## Results

| metric | A (E1 A / A2) | B (CG=40) | C (CG=40 FULL) |
|---|---|---|---|
| graph memory at boot | 0.13 GiB (delta 0.32/0.32) | 0.12 GiB (delta 0.25) | 0.12 GiB (delta 0.16) |
| model load | 15.33 GiB | 15.33 GiB | 15.33 GiB |
| KV pool | 68,605 tokens | 68,605 tokens | 68,605 tokens |
| free VRAM after boot (nvidia-smi) | 1,471 / 1,431 MiB | 1,547 MiB | 1,637 MiB |
| C1 decode T=0, tok/s | 109.4 / 109.4 | 113.0 | 113.9 |
| delta vs A | n/a | **+3.3%** | **+4.1%** |
| C1 decode T=default | 99.6 / 107.2 | 107.5 | 106.0 |
| cohort C8 decode T=0 | 248.8 / 249.9 | 278.6 | 357.9 |
| ladder N=1 ms/pass | 31.8 / 31.4 | 31.0 | 30.6 |
| ladder N=8 decode agg / preempt | 213.4* / 20, 242.0* / 21 | 229.2* / 23 | 222.6* / 24 |
| cudagraph mode | full+piecewise (no warning) | same (no warning) | FULL requested, FULL_AND_PIECEWISE applied |

`*` = ladder tail row, not steady state. A/A2 differ by 13% there.

Noise check: A was measured on two separate boots, 109.4 and 109.4 (0.0% spread; run-to-run within a boot
0.1%). Both A boots ran earlier the same day, not bracketing B and C, so this is A-A-B-C, not a strict A-B-A.
The noise rule's "3-10% needs A-B-A before it is called" therefore applies in the strict sense.

## Verdict

Against the pass rule (>5% decode at C1 or C8, with a larger pool, no mode downgrade): **fail**. The C1 gain
is 3.3-4.1%, not above 5%. The C8 rows (278.6/357.9 against 248.8/249.9) are inside the C8 protocol's own
37% spread measured in E1 (248.8 against 340.9 within one boot), so they do not count. **The pool did not
grow**: `KV_MEM` pins it in bytes, so the smaller graphs cannot buy KV tokens on this configuration. The
landscape's expected "0.3 GiB vs 3.7 GiB of graph memory" is far off too: this stack's graphs are 0.12-0.13
GiB either way, not 3.7 GiB, and cutting the capture set frees about 80-170 MiB of boot residency, not
gigabytes.

What B/C did do, measured: +3.3%/+4.1% C1 decode over two shipped boots, ms/pass 31.8-31.4 → 31.0/30.6, free
VRAM after boot +76/+166 MiB. In the 3-10% band, without the strict A-B-A, this is a lead, not a result. No
mode downgrade occurred in either arm.

## Verified vs assumed

Verified: the compilation-config each engine received, the graph bars, the pool size, free VRAM, and every
number above from the raw logs. Assumed, not verified: that the 0.01-GiB graph-memory delta is the cause of
the +3.3% and not some unrelated cold-boot effect — the two shipped boots say C1 T=0 is repeatable to 0.0-0.1%,
not why B is faster.
