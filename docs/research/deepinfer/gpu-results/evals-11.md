# evals-11: automatic warmup detection

| Field | Value |
|---|---|
| Date | 2026-10-21 |
| Commit | tool in this result's commit: `bench/warmup_probe.py` |
| Image | `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`), stock; compile caches warm (`di-cache` volume) |
| GPU | RTX 3090, 250 W, driver 610.88; the probe started on a GPU cooled to ≤ 55 °C (driver gate), 62 °C at the first request |
| `.env` knobs | B: `SPEC=dflash2 PREFIX_CACHE=1`; D: `SPEC=mtp CTX=long` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\evals-4\{B,D}\warmup_probe.{txt,json}`, `gpu.csv` |

## Command

Run first thing after `/health`, before `bench/warmup.sh` and before any other traffic:

```bash
venv/bin/python bench/warmup_probe.py --out /out/warmup_probe.json   # 8 prompts x 256 tokens at C1 and C4, repeat until two runs agree within 3%
```

## Numbers

| Mode | Run 1 C1 / C4 per-stream tok/s | Run 2 | Run 3 | Run 4 | Verdict |
|---|---|---|---|---|---|
| B | 114.4 / 61.4 | 109.8 / 60.0 (−4.0%) | 105.7 / 58.9 (−3.7%) | 103.6 / 58.2 (−2.0%) | WARM after 4 runs |
| D | 103.6 / 77.6 | 101.2 / 74.5 (−2.3% / −4.0%) | 98.5 / 73.3 (−2.7% / −1.6%) | – | WARM after 3 runs |

GPU during B's probe: temperature 62 → 83 °C over the first ~5 minutes; power at the 250 W cap the whole time, with throttle reason `SW power cap` only.

## Reading

- **On this box the first run is the fastest, not the slowest.** `run_benchmarks.sh` assumes the first run after a start "includes JIT warmup and reads 30–50% low" and keeps the second. With warm compile caches and the launcher's own warmup, there is no cold penalty left. What remains is heat-soak: as the card heats from ~60 to ~83 °C at a fixed 250 W, the sustainable clock drops and C1 decode falls by 9.4% (B) over about 4 minutes before it levels off.
- The probe's "two consecutive runs within 3%" rule detects the end of that settling correctly. For A/B work this matters more than JIT. Every arm in this expedition starts from a GPU cooled to ≤ 55 °C and runs the same sequence, so the drift falls equally on both arms. Absolute numbers from a short run on a cold card read high by up to ~9%.

## Verdict

INFO. The tool works and replaces "keep the second run" with a measured criterion. The settling it detects on this box is thermal (≈ −9% over 4 minutes at C1 in B), not JIT.
