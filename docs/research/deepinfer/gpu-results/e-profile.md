# e-profile: mode E's per-step GPU time at ~170k

| Field | Value |
|---|---|
| Date | 2026-10-21 07:54 SAST |
| Commit | `16b50c8` (tools: `bench/trace_breakdown.py`, added in this result's commit) |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock |
| GPU | RTX 3090, 250 W, driver 610.88; busy median SM clock 1455 MHz, 82 °C; VRAM non-local 92 → 162 MiB (no measurable spill effect) |
| `.env` knobs | `SPEC=dflash2 CTX=huge PREFIX_CACHE=1`, `EXTRA_ARGS=--profiler-config {"profiler":"torch","torch_profiler_dir":"/out/prof"}` |
| Context | prompt 169,895 tokens (`labd_bench --ctx 170000 --tasks qa`, long corpus), pool 268,169 tokens |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\e-profile\E\` (`prof/rank0.*.pt.trace.json.gz`, `breakdown.{txt,json}`, `eprof_*.txt`) |

## Commands

```bash
# in the container, after warmup: a cold pass fills the prefix cache, a warm pass is profiled
venv/bin/python bench/labd_bench.py E_cold --ctx 170000 --tasks qa --max-tokens 64  --corpus /cache/bench/labd_corpus_long.txt
curl -X POST localhost:18020/start_profile
venv/bin/python bench/labd_bench.py E_prof --ctx 170000 --tasks qa --max-tokens 256 --corpus /cache/bench/labd_corpus_long.txt
curl -X POST localhost:18020/stop_profile
# CPU, afterwards:
python bench/trace_breakdown.py prof --json breakdown.json
```

The planned depth was 200k. `--ctx 170000` gave 169,895 prompt tokens (labd converts ctx to characters at 3.6/token, and this corpus tokenizes denser). That is still inside the 150–240k range the plan asks about.

## Numbers

| | Value |
|---|---|
| Decode, unprofiled cold pass | **11.5 tok/s** at 1.86 tok/step (≈ 160 ms/step) |
| Decode, profiled warm pass | 12.5 tok/s at 2.01 tok/step |
| Profiled decode steps | 152 (8 query tokens each) |
| Step wall (mean / median) | 150.8 / 167.9 ms |
| GPU busy within the window | 95.0%; host gap between `execute_model` ranges 5.7 ms median |

Per step, by category (mean ms, % of step wall):

| Category | ms/step | % wall | Main kernel |
|---|---|---|---|
| **Full attention (target)** | **117.7** | **78.0%** | `_kvarn_fused_decode_stage1` 117.3 ms over 16 layers (7.3 ms/layer), stage 2 0.33 ms |
| Linear GEMMs (target) | 17.7 | 11.7% | Marlin, 256 calls |
| Drafter (DFlash2 + lookup) | 3.6 | 2.4% | |
| Norms / elementwise | 2.2 | 1.4% | |
| GDN | 1.3 | 0.9% | |
| lm_head | 0.8 | 0.5% | |
| Other | 1.7 | 1.1% | |
| KV writes, sampling, bookkeeping | < 0.1 | | |

## Against the appendix

- **The gate for attn-2 passes by a wide margin.** It asked for attention ≥ 25% of E's step at 200k; the measurement is 78% at 170k.
- **KVarN's verify kernel runs about 40× above its DRAM floor here.** int4 K and V for 4 KV heads × 256 dims is about 1 KB per token per layer before scales. At 169,895 tokens × 16 layers that is ~2.8 GB, or ~3.0 ms at 936 GB/s, against 117 ms measured. The README's lead correction gives E's floor at 240k as 3.45 ms. The kernel is nowhere near DRAM-bound, so attn-2's estimate ("~+21% at 240k") is far too small. If a pthfa-family kernel got KVarN's verify within 2× of the floor, E's step at 170k would fall from ~150 ms to ~40 ms, about 3.5× faster decode. That is an inference from this profile, not a measurement.
- **E's deep-context speed is an order of magnitude below its 20k headline.** 11.5 tok/s at 170k against 57.8 at ~25k (baseline.md). runtime-2's step curve agrees: the KVarN step at q=8 grows from 31 ms (8k) to 144 ms (150k) and 215 ms (240k) (runtime-2 serve pass, FULL-graph points). The drafter and GEMMs are flat in depth, so the attention kernel is the only part that grows.
- The host gap (5.7 ms, 4% of the step here) includes the 13 per-step KVarN metadata syncs that evals-2 found. At 20k depth, where the step is ~47 ms, the same gap would be ~12%; evals-8's E20k profile measures it.

## Verdict

INFO. Attention is 78% of E's step at ~170k, almost all of it one Triton kernel (`_kvarn_fused_decode_stage1`) running ~40× over its DRAM floor. This is the largest lever measured so far. It re-ranks attn-2 from "+21% at 240k" to "up to ~3× at 150k+" (inferred), and it opens attn-2's gate in the ledger.
