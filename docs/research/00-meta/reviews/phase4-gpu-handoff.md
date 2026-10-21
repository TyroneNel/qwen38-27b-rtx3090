# Phase 4 review: GPU hand-off package

Reviewer: Opus, effort high, P1. Date: 2026-10-21. Scope: `04-benchmarks/gpu-agent-prompt.md` (the prompt), `00-meta/gpu-queue.md` (the queue), `04-benchmarks/LEDGER.md`, checked against the harness, `experiments/*`, `drafter/*`, the launcher, `02-eagle4/transferability.md` (the index), `00-meta/plan.md`, `00-meta/decisions.md` and the phase 3/4 reviews. No GPU, no git, no model loaded. Line numbers are after the edits below.

## Verdict: READY WITH FIXES (fixes applied)

The review found three errors that would fail a run or report a false `blocked`: X-08 recorder path, X-12 lm_head, X-14 paths. It also found two that would give wrong numbers or miss an abort: X-01b context cells, `boot()` restart detection. All five are fixed in the prompt. After these edits nothing known blocks the hand-off. The remaining notes are low-risk, and the agent can handle them as written.

## Findings

| # | Sev | Where | Finding | Status |
|---|---|---|---|---|
| 1 | HIGH | prompt 708, 713, 740; `experiments/X-08/record_top2.py:22` | `record_top2.py` finds its prompts at `<script dir>/../../harness`. The prompt ran it from `/results/_glue/x08/`, so it looked in `/results/harness/prompts/`, which does not exist. Both 14a and 14b would fail on the first recording with `FileNotFoundError`. I reproduced this on CPU (below). | Fixed: each setup line writes a `sed` copy with `HARNESS = "/harness"` (the harness compose file mounts it read-only). Gap 16 added. |
| 2 | HIGH | prompt 1073-1081 (x12_svd.py) | The script read `lm_head.weight` / `lm_head.weight_packed` from the live shard. In the W4A16 tree that tensor is packed int4, and the bf16 copy is in `<shard>.bak` (`drafter/train_mtp.py:205,214-215`). The shape assert therefore always failed, giving a false "no bf16 lm_head: blocked". | Fixed: read `<shard>.bak` the way `train_mtp.py` does, and fall back to an unpacked `lm_head.weight` in the live shard. |
| 3 | HIGH | prompt 1121-1129 (x14-train.sh) | The script checked and trained from `$REPO/models/...` and `$REPO/drafter/data` (the defaults at `train_mtp.py:31-32`). The models are in `/mnt/g/models` (prompt line 22, `arm_env`), so the check gave a false `blocked`. The export also hardlinked between two bind mounts (`/work/models` to `/m/_x14`). `os.link` (`export_mtp.py:27,29`) fails with EXDEV across two mounts even on one disk. | Fixed: find `DATA` with the same `ls` search as X-12, mount `/mnt/g/models` at `/m`, pass `--model /m/... --data /data`, and export from `/m/...` to `/m/_x14/...`. The X-12 data mount uses the same `DATA` search. |
| 4 | MEDIUM | prompt 80-85 (`boot()`) | The service has `restart: unless-stopped`, so a server that crashes during startup comes straight back. `ps --status exited` is then seldom true, and the loop waits 40 min while the server crash-loops. `end()` sees `restarts=` only after the run. | Fixed: `docker inspect` Status/RestartCount. Any state other than `running/0` or `created/0` fails the boot. I checked the template and the `case` on this box. |
| 5 | MEDIUM | prompt 766, 779, 801, 811; queue 25 | X-01b used nominal labd `--ctx 66000/93000`. labd writes ctx x 3.6 characters, and the frozen corpus is about 3.2 chars/token (prompt 207; `baseline.md:50`: `--ctx 85000` gives about 95k). The prompts would have been about 74k and 105k tokens, not the 66-93k that H01 and PR #345 compare. Also, the "per boot return" asked for `prompt=` from the `TOTAL` line, which has no `prompt=` field. | Fixed: cells 58000/82000 (about 66k/93k tokens, the same as bench-9's 58000 to about 65k) in the loop, gate tuple, comment and queue row. The return now takes `prompt=` from the per-task lines. |
| 6 | LOW | prompt 556 | The D11 licence rule named only `/mnt/g/models/_eagle4/`. The raw download sits in `/mnt/g/models/_eagle4-dl/` (prompt 881-900). | Fixed. |
| 7 | LOW | prompt 1192 | Gap 15 said `record_top2.py --ulp-band`. The flag belongs to `classify_divergence.py classify` (default 0.13, `classify_divergence.py:4`). | Fixed. |
| 8 | LOW | prompt 515 | "Add a row `specoff-C1`". `LEDGER.md:44` already has that row, so the agent would have added a duplicate. | Fixed: "Fill in the existing row". |
| 9 | LOW | prompt 532 | The return format had no D19 rule (`decisions.md:23`). A single-agent `WIN` could have been called a speedup. | Fixed: one labelling sentence added. |
| 10 | NOTE | prompt X-03 pre-gate | The retry grep for "available KV cache" can also trigger a retry after a boot fails for a reason other than KV. That costs one boot and gives no wrong number. | Not changed. |
| 11 | NOTE | prompt X-17 build | The build-log grep for the patch step finds nothing on a fully cached rebuild. That gives a false build failure, but only on a second build. | Not changed. |
| 12 | NOTE | `lib.sh` watchdog | A 2 h watchdog `down` during a labd loop happens before `end()`, so `container.log` for that boot is lost. The `aborted` status still applies. | Not changed. |
| 13 | NOTE | queue 23 vs prompt hours table | bench-14a is "1.0-1.4 h" in the queue and the prompt header, and 1.2 in the hours table. The table uses the midpoint, which is consistent. | Not changed. |

## Checks found correct (no change)

- **Commands vs scripts (more than 15 lines checked):**
  - `sweep.py` flags and JSON fields (`summary.rate.mean`, `valid_reps`, `vram.peak_mib`) used by `sweepfloor` and `extract.py`
  - `run_suite.sh` env knobs (`PHASES`, `LC_CELLS=` empty, `SMOKE`, `GPUMEM_CSV`, `MAXTOK`)
  - `conc_ladder.py --n --ctx-tokens --out --reps --salt --label --json`
  - labd line format and the `labdfloor` / X-01b regexes
  - `parse_timing.py --label --t0 --t1 --out --windows`
  - `classify_divergence.py classify|judge` arguments and exit codes 0/1/2/3
  - `rescore_replay.py REQ --out`
  - `record_top2.py --sets --doc-ctx --corpus --passes --top-n --out`
  - `train_mtp.py` flags: `--epochs` is a float, plus `--depths --depth-weights --warmup --eval-every --lr --micro-tokens --seed --model --data`
  - `export_mtp.py ck src dst --bits --head-bits`
  - launcher profiles: fast 65536 bf16, huge 200000 kvarn, long 150000 fp8 (`start_qwen.sh:210-224`). The new 82000 cell fits.
  - `mksweep`: three `sed` hits on `sweep.py`
  - `xroll` / `xpart` / `xn` layout against the X-01b gate globs
- **Gates:** every X gate matches the index `[review: …]` marks and the plan table. That includes X-06 (+3% over 5 boots, plan over index, as the prompt states), X-08 (`--ulp-band 0.25`, D18) and X-03 (1.12x, lookup on).
- **Rules:**
  - one job at a time; every part at most 2.0 h, with the watchdog
  - training only as pilots with `timeout` budgets (X-13, X-14)
  - `PTH_FA` pinned in every `arm_env` call (D10)
  - D15 floors applied only to stock-equal arms; growth 64 MiB flag and 256 MiB abort
  - two boots per baseline mode (D16)
  - D17 delivery: commit after every job, final push, `.env` restored
  - `disklog` before every job; the X-03 download log, revision sha and sha256
  - D11 is evaluation only; HF id `EAGLE-Corp/Qwen3.8-27B-EAGLE4` (6,178,152,384 B, commit f58a598)
  - Rule 9: no attribution; harness, `experiments/` and raw files never edited
- **Consistency:** queue IDs bench-1..25c, parts, durations and result paths match the prompt and the LEDGER rows. The hours table sums check out: 19.45 + 7.6 = about 27, conditional rows 11.7, upper bound about 39.

## Dry runs (CPU, this box)

- `bash -n` on all 42 bash blocks and `py_compile` on all 3 python blocks, extracted from the edited prompt: all pass.
- **X-08 recorder.** I ran the `sed`-patched `record_top2.py` from a copy outside its directory against `mock_server_x08.py`: it wrote `greedy_T.json`, rc 0. The unpatched copy in the same place failed with `FileNotFoundError .../harness/prompts/short_chat.jsonl`.
- **X-01b gate.** I ran the gate heredoc on synthetic labd logs in the real layout (rounds 1-2 under `X-01b-<date>/15a|15b/`, round 3 still in `results/`). It found 3 boots per arm per cell and read the `TOTAL` decode, not the per-task line. It printed the ratios and wrote `x01b.go-q`.
- **`boot()`.** The `docker inspect` template returned `exited/0` on a local container. The `case` passes `running/0`, `created/0` and an empty value, and fails `restarting/1`, `running/2` and `exited/0`.
- The harness was not touched, so `selfcheck.sh` was not re-run.

## Edits made

`04-benchmarks/gpu-agent-prompt.md`:
1. 80-85 `boot()`: restart and exit detection through `docker inspect`, with a `[review]` line.
2. 708, 713, 740: X-08 setup writes a `sed`-patched `record_top2.py`, with a `[review]` note on 708.
3. 515: specoff-C1 "Fill in the existing row".
4. 532: D19 labelling sentence in the return format.
5. 556: licence rule includes `/mnt/g/models/_eagle4-dl/`.
6. 766, 779, 801, 811: X-01b cells changed to 58000/82000, the comment, the token note with a `[review]` mark, and the `prompt=` source.
7. 1068: X-12 `DATA` search and mount.
8. 1073-1081: X-12 reads the bf16 `lm_head` from `.bak`.
9. 1121-1129, 1132: X-14 `DATA`/`MD`, `/m` mounts, `--model`/`--data`, export on one mount, with a `[review]` note.
10. 1192-1193: gap 15 names `classify_divergence.py`; gap 16 added.

`00-meta/gpu-queue.md`:
11. Line 25 (bench-15a): "labd about 66k and 93k prompt tokens (nominal `--ctx 58000`, `82000`)".

`LEDGER.md` and the harness: no change.

### Follow-up: deferred items from `reviews/phase5-findings.md` "For the GPU hand-off"

I verified each item against its source before applying it.

- **Item 1, greedy near-tie: verified.** `harness/greedy_equal.py:84` defaults `--near-tie` to 0.1, and `harness/README.md:85` repeats that value. D18 (`decisions.md:22`) puts the near-tie band below 0.01, the replay band at 0.01-0.25, and real above 0.25. `greedy_equal` has only two bands. I tested it on CPU with a synthetic pair whose margins are 0.05 and 0.04: at the default the divergence is `near-tie`, and with `--near-tie 0.01` it is `REAL`. With the flag alone, bench-7 would therefore call replay-band divergences `INVALID`. The harness is unchanged, so `selfcheck.sh` was not re-run.
- **Item 2, bench-23 gate: verified.** The reviewed bar in `plan.md:51` is the H03 ratio (≥1.12x DFlash2, same run). `transferability.md:231` says "within 10% of break-even". The prompt's absolute 3.375 bar agrees with that ratio only when DFlash2 runs at 3.36 tok/step.
- **Item 3, stale 0.13: verified.** `classify_divergence.py:4,46` now defaults `--ulp-band` to 0.25. The prompt still said "still 0.13" (line 706) and gap 15 quoted 0.13.

Edits:
12. Prompt 290 (bench-2) and 393 (bench-7): `greedy_equal.py compare ... --near-tie 0.01`.
13. Prompt 300 (bench-2 pass) and 395 (bench-7 verdict): a `REAL` line whose larger margin is at most 0.25 counts as the D18 replay band. bench-2 is not blocked by it, and bench-7 records it as INFO, "judged in bench-14". Only a margin above 0.25 blocks bench-2 or gives `INVALID` in bench-7. Line 300 has a `[review]` mark.
14. Prompt 1104 (bench-23): the gate is now the 17b same-run ratio, 1.008 ≤ r < 1.12, replacing the absolute 3.375 bar. The line has a `[review]` mark.
15. Prompt 706: the `--ulp-band` default is now 0.25. Gap 15 (1192) is rewritten for the `greedy_equal` near-tie default.

Not applied: item 4 (nothing to change) and item 5 (the optional `start_qwen.sh:299-304` cite; the line range is off by two lines and affects nothing that runs). The queue and `LEDGER.md` already state bench-23 as "within 10% of break-even", so they need no change.
