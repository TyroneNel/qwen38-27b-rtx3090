# Batch 3 Q3 and Q4: checkpoint flags, and can the state pool be seen? (2026-10-14, RTX 3090 at 250 W)

The same report is in `2026-10-14-Q3-shared-prefix-ckpt` (boot q3ckpt) and `2026-10-14-Q3-mamba-block-8192` (boot
q3blk). Both boots use the Q1-a config (int8 base, k=7, `KV_MEM=5583457484`, `MAX_SEQS=8`) with one flag appended
to `EXTRA_ARGS`. They are compared with the Q1-a boot `../2026-10-14-Q1-k7/raw-bootlines-q1k7.log`.

## Q3: neither flag changed the pool. Both were accepted, and neither applied to any line the batch asks about.

| boot-log line | Q1-a (no flag) | `--enable-mamba-shared-prefix-checkpoint` | `--mamba-block-size 8192` |
|---|---|---|---|
| engine args line | — | `'enable_mamba_shared_prefix_checkpoint': True` | `'mamba_block_size': 8192` |
| attention block size | `Setting attention block size to 864 tokens …` | 864 | 864 |
| mamba page padding | `Padding mamba page size by 1.09% …` | 1.09% | 1.09% |
| group size | `… using group_size 8 …` | 8 | 8 |
| draft block promotion | `Not promoting draft block sizes: no divisor of the primary block 864 is >= … covering block 1728.` | same | same |
| GPU KV cache size | 116,440 tokens | 116,440 tokens | 116,440 tokens |
| kv lcm block sizes | 864 | 864 | 864 |
| free VRAM after boot | 1,365 MiB | 1,365 MiB | 1,365 MiB |
| steady `Running:` pair (2 waves) | `Running: 7 reqs, Waiting: 1 reqs, GPU KV cache usage: 99.2%` | the same | the same |

**Result:** both flags reach the engine, because the args line carries them. Neither changes the attention block,
the page padding, the group size, the pool or the seats. `--mamba-block-size 8192` does not apply on this config,
as E6 found: the attention block stays at 864 tokens. The shared-prefix checkpoint registers at a prefix junction,
which the probe's distinct, unshared prompts never exercise. So the probe's "no change" for that flag says nothing
about shared-prefix traffic. Neither boot left the 84.2-92.2 MiB Shared Usage floor (`mem.txt`).

## Q4: the instrumentation

**`vllm serve --help=all` does expose ReplaySSM on this pin.** That contradicts batch 1 and the landscape page's
G7 ("no CLI flag for them on this pin"). From `raw-help-all-q3ckpt.log`, run in the q3ckpt container:

```
1126:  --replayssm-buffer-len REPLAYSSM_BUFFER_LEN
1127:                        ReplaySSM logical history length B for Mamba2. Triton
1136:  --use-replayssm, --no-use-replayssm
1137:                        Use the ReplaySSM Mamba2 decode kernel: cache recent
```

The field's docstring (`raw-q4-replayssm.log`, `vllm/config/cache.py:213`) keeps the restriction batch 1 recorded:

```
    use_replayssm: bool = False
    """Use the ReplaySSM Mamba2 decode kernel: cache recent SSM inputs and skip
    the per-step full-state store, writing the checkpoint back only on flush.
    Requires mamba_cache_mode 'none' or 'align' (prefix caching) and the Triton
    or FlashInfer mamba backend; standard (non-speculative) decode only. …"""
```

It is a Mamba2 kernel for non-speculative decode, and `vllm/config/vllm.py:3453` rejects any architecture without
`supports_replayssm`. So the flag existing changes nothing for DFlash2 serving: it is still not a lever for this
configuration. Recorded, not pursued, as the brief directs. Whether this model's class sets `supports_replayssm`
was not checked.

**No boot line reports the state pool.** None of the 12 batch-3 boot logs (`2026-10-14-Q*/raw-boot-*.log`)
matches `num_gpu_blocks`, `mamba blocks`, `state slots` or `blocks for mamba`. The pool appears only as a token
count (`GPU KV cache size`) and as the `GPU KV cache usage` percentage in the stats line. That percentage is what
measured the seats in Q1 (14.2% of the pool per request at k=7). A direct counter of state slots in use and free
would remove that inference step. It is the candidate patch.

## Verified vs assumed

Verified: every line above is quoted from the boot logs, the `--help=all` grep and the source grep in these
directories. One extra command ran outside a boot: a one-shot `docker run --rm` of the image, with no GPU and no
model, to print the `cache.py` docstring. It is recorded in `cmd.txt`. Its attempt to print the help entries
through `sed` printed nothing. The help lines quoted above come from the in-container grep on the q3ckpt boot.
