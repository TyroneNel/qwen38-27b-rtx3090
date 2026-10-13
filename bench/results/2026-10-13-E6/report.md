# E6: mamba block size

RTX 3090, 250 W, container boot. Both arms `SPEC=dflash2 PREFIX_CACHE=1 CTX=fast DFLASH_TOKENS=7
KV_MEM=5583457484 MAX_LEN=65536`. B adds `EXTRA_ARGS='--mamba-block-size 8192'`.

## Does the flag exist?

Yes. `venv/bin/vllm serve --help=all` inside the image lists `--mamba-block-size MAMBA_BLOCK_SIZE`
(`../2026-10-13-image/raw-engine-probe.log`). The engine accepted it: B's `non-default args` line contains
`'mamba_block_size': 8192`, and the validator allows it only with prefix caching, which this config has
(`raw-mamba-block-size-source.log`, `config/vllm.py` `validate_mamba_block_size`).

## What changed in the boot: nothing visible

A and B printed byte-identical pool geometry (`raw-boot-*.log`):

```
Setting attention block size to 448 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 3.23% to ensure that mamba page size and attention page size are exactly equal.
GPU KV cache size: 68,605 tokens, Maximum concurrency for 65,536 tokens per request: 1.05x
```

Free VRAM after boot: 1,477 MiB in both. No line mentions 8192. The 0.31.0 source explains why the user's
value may not survive: `platforms/interface.py` overwrites it (`cache_config.mamba_block_size =
attn_block_size` after `chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)`), and this repo
patches `interface.py` (`hybrid-kv-groups` patches). Whether 8192 was honoured at a level the log does not
print is **not verified**.

The pool could not have grown in any case: `KV_MEM` pins it in bytes, so 68,605 tokens is arithmetic, not a
knob effect.

## Results

| metric | A (shipped) | B (`--mamba-block-size 8192`) |
|---|---|---|
| `non-default args` | no mamba_block_size | `'mamba_block_size': 8192` |
| KV pool | 68,605 tokens | 68,605 tokens |
| page-geometry lines | 448 / 3.23% / group_size 8 | 448 / 3.23% / group_size 8 |
| two long concurrent (~24k ctx, `--n 2 --reps 1`) | 0 preemptions, run 2, TTFT 36,020 ms | 0 preemptions, run 2, TTFT 35,946 ms |
| ladder N=16 decode agg / preemptions | 258.5* / 73 | 228.0* / 70 |
| follow-up TTFT probe (`--n 1 --shared --reps 3`, 24k ctx) | best rep TTFT 23,852 ms | best rep TTFT 23,810 ms |

`*` = tail row, not steady state (A/A2 spread at N=16 was 225-240 tok/s in E1). The -11.8% there is inside
the tail rows' own noise.

The follow-up TTFT probe is not interpretable. `conc_ladder.py` keeps the rep with the best steady-state
decode aggregate, so the printed TTFT is whichever rep it kept, not the cached-rep TTFT, and the reps are not
listed individually. Cold/cached TTFT at block size 448 vs 8192 was therefore **not measured**. G10's claim is
a prefix-cache TTFT effect, and the one thing this arm ran that could show it cannot be read from these logs.

## Verdict: knob applied, no measurable effect found (inconclusive)

The engine took the flag. The two visible knob effects — reported pool tokens and preemptions — are identical
to A, the first because the pool is pinned in bytes, the second because the pool is the same size. The
checkpoint-cadence effect G10 is about produces no line in this engine's log, and the TTFT probe was not
designed well enough to read it. On this pin, this arm needs a checkpoint-counting metric (or a `--metrics`
histogram) to answer, not a speed re-run.

## Verified vs assumed

Verified: the flag exists, the engine received it, the geometry and pool lines, the ladder and TTFT numbers
above. Not verified: whether 8192 actually changed the checkpoint cadence (the log does not say), and the
cold/cached TTFT difference (probe not interpretable).
