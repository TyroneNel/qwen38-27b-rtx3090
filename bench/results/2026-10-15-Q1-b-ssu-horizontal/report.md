# Batch 5 Q1: the unadvertised mamba knobs (S-11) on the band-2 profile (2026-10-15, RTX 3090 at 250 W)

This report is identical in the seven `2026-10-15-Q1-*` directories of this batch (`a1-baseline`, `b-ssu-horizontal`,
`c-ssu-vertical`, `d-backend-flashinfer`, `bf-flashinfer-horizontal`, `cf-flashinfer-vertical`, `a2-baseline`). It
does not cover batch 4's `Q1-seat-gauge`.

**Verdict:**
- **Engaged? No knob engaged.**
  - `--mamba-ssu-algorithm` alone (horizontal, vertical) is refused at config validation.
  - `--mamba-backend flashinfer` (alone or with either algorithm) reaches the engine's args and changes no
    kernel-selection line. `GDN decode kernel: triton` and the 132-key JIT warmup are identical to the baseline.
- **Helped? No.** No arm moved C1 T=0 or the N=8 aggregate by the 3% the pass rule asks:
  - C1 T=0 decode is 91.5-91.8 on all five booted arms.
  - N=8 is 342.5 on flashinfer against 336.9 / 338.4 on the bracketing baselines (+1.4% against their mean). That is
    within the baseline-to-baseline spread and comes from tokens/step, not step time.
- **GSM8K n=50:** not run, because there is no winner to smoke.

## Arms

- **Common to all:** the band-2 profile at the full pin: int8 `CTX=fast`, `DFLASH_TOKENS=3`, `KV_MEM=5583457484`,
  `MAX_LEN=65536`, `MAX_SEQS=16`, image `hyperqwen:exp-844cd90`.
- **What varies:** one addition to `EXTRA_ARGS` per boot.
- **Env files:** `.env` in each directory.
- **Order:** a1 → b → c → d → bf → cf → a2, so the baseline runs first and last (A-B-A).
- **bf and cf are not in the brief.** The driver adds them only when b and c are refused and d boots. The refusal
  message itself asks for `--mamba-backend flashinfer` alongside the algorithm, so this pair is the only way the SSU
  knob can be applied at all.

| arm | `EXTRA_ARGS` addition | boot |
|---|---|---|
| a1 | (none) | ok |
| b | `--mamba-ssu-algorithm horizontal` | **refused** |
| c | `--mamba-ssu-algorithm vertical` | **refused** |
| d | `--mamba-backend flashinfer` | ok |
| bf | `--mamba-backend flashinfer --mamba-ssu-algorithm horizontal` | ok |
| cf | `--mamba-backend flashinfer --mamba-ssu-algorithm vertical` | ok |
| a2 | (none) | ok |

## Did the knob apply? (boot logs)

**b and c** (`raw-boot-q1b.log`, `raw-boot-q1c.log`), verbatim:

```
ValueError: Mamba SSU algorithm selection is only supported with the FlashInfer backend. Please set `--mamba-backend flashinfer`, or omit `--mamba-ssu-algorithm`.
```

**d, bf and cf**:
- **The knobs reached the engine.** Its args read `'mamba_backend': 'flashinfer'`, plus `'mamba_ssu_algorithm':
  'horizontal'` and `'vertical'` on bf and cf.
- **No line changed.** Against a1's kernel and pool lines, with timestamps, pids, durations and rates stripped, bf and
  a2 are identical. d and cf differ only in a tqdm progress-bar render of the same `JIT kernel warmup (132 compile
  keys)`. Every arm reports `GDN decode kernel: triton` (`../2026-10-15-Q1-d-backend-flashinfer/
  raw-bootdiff-vs-baseline.log`).
- **Pool:** 131,724 tokens and `num_gpu_blocks=404` on every arm, with 1,253 MiB free after boot on every arm.

**Why** (source, `../2026-10-15-batch5-tools/src-mamba-knobs.txt`, vLLM 0.31.0 in this image):
- **Who reads the backend:** `mamba_config.backend` and the SSU dispatcher (`layers/mamba/ops/ssu_dispatch.py`) are read
  by `mamba_mixer2.py` (Mamba2 layers) and the ReplaySSM warmup only.
- **Not the GDN layers:** a grep of `layers/mamba/gdn/*.py` and `models/qwen3_5*.py` for
  `mamba_config|ssu|MambaBackend` finds nothing that reads them.
- **The validation:** `config/mamba.py:76` raises the error above when an algorithm is given without the FlashInfer
  backend.

This model's recurrent layers are GDN, so the knobs have nothing to act on here.

## Speed

Metric: `run_benchmarks.sh single` C1 T=0, run 2 of each boot (run 1 discarded as warmup), and `conc_ladder.py --n
1,4,8,16` (`raw-ladder-*.json`, ~4k-token prompts).

| arm | C1 T=0 decode (run 2) | C1 T=0 e2e r1 / r2 | tok/step | ladder N=1 | N=4 | **N=8** | N=16 | ms/pass N=1 / N=8 |
|---|---|---|---|---|---|---|---|---|
| a1 | 91.6 | 89.26 / 89.12 | 2.70 | 101.0 | 261.7 | **336.9** | 303.2 TAIL (10 running, 4 preempt) | 29.3 / 63.1 |
| d | 91.6 | 89.40 / 89.05 | 2.70 | 97.3 | 264.8 | **342.5** | 306.6 TAIL (10, 2) | 29.5 / 63.8 |
| bf | 91.8 | 89.27 / 89.39 | 2.70 | — | — | — | — | — |
| cf | 91.5 | 89.68 / 89.01 | 2.70 | — | — | — | — | — |
| a2 | 91.7 | 89.23 / 89.16 | 2.70 | 100.3 | 261.5 | **338.4** | 301.8 TAIL (10, 5) | 29.2 / 63.5 |

- **Delta, d against the mean of a1 and a2:**
  - C1 T=0 decode: 0.0%.
  - N=8: +1.4% (342.5 against 337.65). Its ms/pass is 63.8, *higher* than the baselines' 63.1/63.5. The gain is
    tokens/step in a free-running ladder row (2.80 against 2.75/2.74), the ~10% noise class, not a faster step.
  - N=1 ladder: −3.3% (97.3 against 100.65), also a tokens/step move (2.90 against 2.98/2.94) at the same ms/pass.
- **Baseline spread:** a1 against a2 is 0.1% on C1 decode and 0.4% on N=8.
- **Held-load probe** (16 × ~1k prompts, `raw-running-*-w1.log`): the peak engine line on a1, d and a2 together is
  `Running: 13 reqs, Waiting: 3 reqs, GPU KV cache usage: 96.8%`, 16 HELD samples. Seats are unchanged, as the
  identical pool predicts.
- **Spills:** none, measurement or otherwise, on any booted arm (`mem.txt`). The 16 MiB `MAX_SEQS=16` boot step was
  recorded and is the known one.

## Verified vs assumed

**Verified:** the refusals, the args lines, the identical kernel lines and every speed figure are quoted from this
batch's raw files.

**Assumed:** that the knobs would act on a Mamba2 model is read from the source, not measured. No Mamba2 model was
booted.
