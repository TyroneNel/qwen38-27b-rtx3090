# Run state: paused 2026-10-21 20:25 SAST (user asked to stop and save)

GPU queue stopped cleanly: no containers running, GPU 0 at 0 MiB and 250 W. Everything measured so far is committed; the ledger is current. To resume, read this file, then [LEDGER.md](LEDGER.md).

## Box tooling (not in the repo)

Driver scripts live in `G:\dev\hq-results\deepinfer-20261020\tools\`:

| Script | Use |
|---|---|
| `di.sh` (`di_boot`) | One server boot per arm: gate, samplers, job via `docker exec`, `bootsum.py` |
| `run_tool.sh` (`di_tool`) | Standalone GPU tool containers |
| `phase*.sh` | Queued plans, resumable: boots whose dir has `rc=0` are skipped |
| `job_*.sh` | In-container jobs |
| `restart_stack.ps1` | Docker Desktop + WSL restart |

Run every script from WSL as `wsl -d Ubuntu -e bash /mnt/g/dev/hq-results/deepinfer-20261020/tools/<script>`. Inline `wsl -- …` commands get their `$` re-expanded, so put logic in script files.

Variant images built locally:
- `hq:exp-spec5`, `hq:exp-quant2`, `hq:exp-runtime3`, `hq:exp-spec1`, `hq:exp-drafter1a`
- `hq:exp-kvv`, `hq:exp-negctl`, `hq:exp-escape`
- `hq:exp-prof` (ncu/nsys), `hq:exp-pthfa`, `hq:exp-pthfa-prof`
- `hq:exp-attn2`, `hq:exp-quant4`

Rebuild any of them with `tools/build_variants.sh` or the Dockerfiles in `experiments/docker/`. The base for all of them is the stock `ghcr.io/syv-ai/hyperqwen:latest` (`c696d22e80b4…`); re-pull it first.

## Where the queue stopped

Phase 3 was running `phase3b.sh` (spec-0 round 3) when stopped. spec-0 is already decided (WIN, two rounds).

Remaining work, in order:
1. **`phase3c.sh all`**: drafter-1a (D′ arms A0, T1, T1s ×2 + A1), quant-7 (D and E, `-fast` vs `-fast-emb4`; that model dir is built), quant-5 (A1 vs A2 ×3).
2. **`phase3x.sh`**:
   - kvarn-varlen full 0–127 residue sweeps (default corpus, then the frozen labd corpus). Inside the remaining ~3 h of its timebox.
   - quant-2 spill-free throughput re-run (×3 per arm).
   - spec-0 at 20k (k = 7/3/5 ×2).
3. **Phase-end steps:** update the ledger, `MODELS_DIR=/mnt/g/models make verify`, commit and push, re-check the power limit.
4. **Phase 4: `phase4a.sh`** (attn-1 tests per stage, then the microbenchmark arms on `hq:exp-pthfa`). Then attn-3, quant-6 and attn-5 (no code prepared yet). The attn-1.1 gate is treated as met by the microbenchmark (attn-ncu.md).
5. **Phase 5:** drafter-2 capture and eval (`drafter/capture_dflash2.py --dump-target`, `drafter/eval_dflash2.py --check-known`), then quant-3 (Hessian capture + `quant_dflash2.py --q3` into a new dir) and drafter-3 (`drafter/train_dflash2_tv.py`, 1-day timebox), each scored by drafter-2 first. drafter-1b depends on the drafter-1a or spec-2 result.
6. **Phase 6:**
   - **`phase6a.sh` (attn-2):** `KVARN_VERIFY_SHARED` Triton kernel — correctness, then microbenchmark, then end to end plus KL. The pthfa CUDA variant is behind `PTHFA_KVARN=1`; it compiles but has never run.
   - **quant-4:** go/no-go capture first. The CPU estimate is only a 1.3–2.2× error gain, because the requant has to start from the int4 checkpoint.
   - **attn-4:** gate needs attn-2 done.
   - **attn-6:** probe only if split imbalance is seen.
7. **Finish:** README `## GPU results` section and the `Measured` columns, final `make verify` on a stock build of the branch, the report.

## Uncommitted code in the working tree (written by delegated agents; committed with each item's result)

| Files | Item | Status |
|---|---|---|
| `pthfa/` (attn-1 stages 1–3 behind `PTHFA_TILE48` / `PTHFA_MMA`; attn-2 CUDA kernel behind `PTHFA_KVARN=1`) | attn-1, attn-2 | default-off; built into `hq:exp-pthfa` |
| `docs/research/deepinfer/experiments/docker/Dockerfile.pthfa`, `experiments/patches/attn2-*.patch`, `bench/test_kvarn_verify.py`, `bench/kvarn_verify_bench.py`, `bench/test_pthfa_kvarn.py` | attn-2 | not yet run |
| `experiments/patches/drafter1a-conf-trim.patch` (launcher `CONF_TRIM` block is committed) | drafter-1a | not yet run |
| `experiments/patches/quant2-lossless-fold.patch`, `bench/test_marlin_int8_fold.py`, fixed `bench/test_marlin_int8_asym.py` | quant-2 | GPU tests passed (bit-exact remap) |
| `experiments/patches/kvarn-varlen-{sched,fp}.patch`, `bench/kvarn_varlen_repro.py` | kvarn-varlen | used in steps 1–4 |
| `experiments/quant7_emb4.py`, `experiments/quant5_head_kl.py` | quant-7, quant-5 | artefact `G:\models\Qwen3.8-27B-W4A16-AutoRound-fast-emb4` built |
| `experiments/patches/quant4-act-hadamard.patch`, `experiments/quant4_h128down.py`, `bench/test_act_hadamard.py`, `cpu-checks/quant4_requant.*`, `bench/act_calib.py --dump-inputs` | quant-4 | go/no-go pending |
| `drafter/dflash2_ref.py`, `drafter/eval_dflash2.py`, `drafter/capture_dflash2.py` (`--dump-target`), `drafter/gptq_utils.py`, `drafter/quant_dflash2.py` (q3 flags), `drafter/train_dflash2_tv.py`, `drafter/README.md`, `bench/test_gptq_q3.py` | drafter-2, quant-3, drafter-3 | CPU self-tests pass |

## Open decisions found so far (for the user)

- **spec-0:** make `DFLASH_TOKENS=3` the `CTX=huge` default above some depth (pending the 20k run).
- **runtime-3:** carry the tool-call bound patch.
- **quant-2:** carry the lossless fold, pending throughput.
- **evals-5:** whether to carry the escape patch (it removes the control-token ids, but the spelled-out injection still works).
- **attn-2:** the KVarN verify kernel is the largest lever measured: 78% of E's step at 170k, ~40× its DRAM floor.
