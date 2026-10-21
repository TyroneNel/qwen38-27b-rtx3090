# attn-ncu: pthfa verify at 150k, T=8 — tensor pipe or DRAM?

| Field | Value |
|---|---|
| Date | 2026-10-21 10:30 (microbenchmark), 10:33 and 13:13 (ncu attempts) SAST |
| Commit | in this result's commit: `bench/pthfa_verify_bench.py`, `bench/ncu_pthfa.sh`, `experiments/docker/Dockerfile.prof` |
| Image | microbenchmark: stock `ghcr.io/syv-ai/hyperqwen:latest`; ncu: `hq:exp-prof` (stock + Nsight Compute 2025.3.1 + Nsight Systems 2025.3.2) |
| GPU | RTX 3090, 250 W, driver 610.88; busy median SM clock 1605 MHz, 74 °C |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\attn-ncu\{microbench,ncu}\` |

## ncu: blocked

`ncu` fails in the container with `ERR_NVGPUCTRPERM` (no access to GPU performance counters). The user enabled "Allow access to the GPU performance counters to all users" in the NVIDIA Control Panel. Docker Desktop and WSL were then restarted (`tools/restart_stack.ps1`), and the second attempt still failed the same way. The registry value the setting normally writes (`RmProfilingAdminOnly`) does not appear under `nvlddmkm\Global\NVTweak` either. Counter access through the WSL2 paravirtual driver could not be obtained on this box, so the tensor-pipe vs DRAM question is answered from the microbenchmark below, not ncu.

## Microbenchmark (the substitute evidence)

`bench/pthfa_verify_bench.py`: one full-attention layer of the pthfa CUSTOM backend's call (paged int8 per-token-head KV, Hq 24 / Hkv 4 / D 256), 16 distinct layer caches replayed in one CUDA graph so KV never sits in L2; batch 1. Columns: µs per layer, achieved DRAM GB/s, % of 936 GB/s, the DRAM floor.

```bash
docker run ... ghcr.io/syv-ai/hyperqwen:latest /app/bench/pthfa_verify_bench.py --out /out/pthfa_verify_b1 --tag stock
```

| T (query tokens) | kernel | 64k µs (% DRAM) | 112k | 128k | 150k µs (% DRAM) | floor @150k |
|---|---|---|---|---|---|---|
| 1 | decode | 263 (54%) | 457 (55%) | 525 (54%) | 616 (54%) | 333 |
| 4 | decode | 305 (47%) | 524 (48%) | 602 (47%) | 716 (47%) | 333 |
| 5 | decode | 357 (40%) | 618 (40%) | 724 (39%) | 858 (39%) | 333 |
| **6** | **tile16 (64-row padded)** | 581 (25%) | 1014 (25%) | 1175 (24%) | 1382 (24%) | 333 |
| **8** | tile16 | 586 (24%) | 1026 (24%) | 1180 (24%) | **1398 (24%)** | 333 |
| 10 | tile16 | 597 (24%) | 1041 (24%) | 1198 (24%) | 1406 (24%) | 333 |
| 12 / 16 | tile16 (2 tiles) | 1072 / 1088 (13%) | 1895 / 1925 | 2189 / 2212 | 2595 / 2624 (13%) | 333 |

## Reading (answers attention-kv open question 1 without counters)

- **At T=8 the kernel is not DRAM-bound.** It moves KV at 24% of peak, 4.2× its floor at 150k. The decode kernel on the same bytes reaches 47–54% at T ≤ 4. With identical KV bytes, the 2.0× gap at T=5 vs T=6 can only be on-chip work: the padded 64-row tile does 64/48 = 1.33× the MMA rows plus the tile's own overheads. Cost is flat from T=6 to T=10 (one tile) and doubles at T=12 (two tiles).
- These are exactly the stairs attn-1's stages target. Stage 1 moves T=6–8 onto the unpadded decode kernel; if it lands near the T=5 decode figure, 1,398 → ~860 µs at 150k is 1.6× per layer (inferred), above the ≥ 1.3× win line. The pipe-level split (tensor pipe vs issue vs L1) stays unmeasured without counters.
- runtime-2's end-to-end step stair at 150k (q 4→8: +10.4 ms) matches this table: (1,398 − 716) × 16 layers = 10.9 ms.

## Verdict

**blocked** for ncu itself: performance-counter access is unavailable in WSL2 containers on this box after the Control Panel setting and a full Docker/WSL restart. The question it was meant to answer is answered by the microbenchmark: pthfa at T=8 / 150k runs at 24% of DRAM bandwidth (on-chip-bound, a padded-tile stair), not at the DRAM floor. Decision: treat that as meeting attn-1.1's gate. The gate exists to confirm that the T=8 verify is not DRAM-bound before investing in cheaper MMAs, and the 24% vs 54% contrast confirms it.
