# Batch 4 Q3: ReplaySSM in a non-speculative profile (2026-10-15, RTX 3090 at 250 W)

**Verdict: fail. The engine refuses `--use-replayssm` for this model's architecture, so it never engages.** This closes
the last upstream state knob for this checkpoint on this pin.

## The arms

| | A (off) | B (on) |
|---|---|---|
| boot | `../2026-10-15-Q2-off/` (q2off) | this directory (q3rssm) |
| config | int8 `CTX=fast`, `SPEC=off`, `KV_MEM=5315022028`, `MAX_SEQS=16`, `PREFIX_CACHE=1` (`mamba_cache_mode=align`) | the same plus `--use-replayssm` in `EXTRA_ARGS` |
| env file | `.env.q2off` | `.env.q3rssm` |

- **The A arm:** Q2's `SPEC=off` boot, which differs from B by this flag only. This is the docstring's allowed case:
  non-speculative decode and `mamba_cache_mode` `align`.
- **The flag reached the engine.** The args line reads `'use_replayssm': True` (`raw-boot-q3rssm.log`).
- **The engine rejected it at config validation**, before any weight loaded (`raw-boot-q3rssm.log`, `raw-bootfail-q3rssm.log`):

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
  Value error, --use-replayssm is not supported for architecture 'Qwen3_5ForConditionalGeneration' [type=value_error, ...]
```

- **The container:** it restarted, the driver saw `container state: 1 running` and stopped it. It never answered
  `/health`.
- **The counter:** stayed at the idle 0.2 MiB with 4.1 MiB dedicated (`mem.txt`); no CUDA context was created.

## Reading it

| metric the brief asks for | A (q2off) | B |
|---|---|---|
| ms/pass C8 / C16 | 31.9 / 43.3 | no boot |
| decode agg C8 / C16 | 250.7 / 369.2 | no boot |
| seats (`Running:` at 16 × ~4k) | 16 | no boot |
| engaged? | — | no: refused for `Qwen3_5ForConditionalGeneration` |

**Why it is refused:**
- **The check:** `vllm/config/vllm.py:3453` rejects any model whose class does not set `supports_replayssm` (batch 3 Q4,
  `../2026-10-14-Q3-shared-prefix-ckpt/raw-q4-replayssm.log`).
- **This model:** its class does not set it. The docstring says "Mamba2 decode kernel", and this model's recurrent layers
  are GDN, not Mamba2.
- **So:** there is no profile, speculative or not, in which this flag applies to this checkpoint on this pin.

## Verified vs assumed

**Verified:** the args line and the error are quoted from the boot log.

**Assumed:** nothing. "GDN, not Mamba2" is the batch-3 docstring reading; the refusal itself does not depend on it.
