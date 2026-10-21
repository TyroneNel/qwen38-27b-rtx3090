# spec-1 (= runtime-1): opt GDN into vLLM 0.31's adaptive verification

| Field | Value |
|---|---|
| Date | 2026-10-21 18:12–18:45 SAST |
| Commit | in this result's commit: `docs/research/deepinfer/experiments/patches/gdn-adaptive-verify-optin.patch`, the `ADAPTIVE_VERIFY` launcher knob in `single-user/start_qwen.sh` (default off), `bench/test_adaptive_verify_knob.sh` |
| Image | `hq:exp-spec1` = stock `ghcr.io/syv-ai/hyperqwen:latest` + the patch + the repo launcher. All three arms on this image |
| GPU | RTX 3090, 250 W, driver 610.88 |
| `.env` knobs | `SPEC=dflash2 PREFIX_CACHE=1` (mode B, `CTX=fast`, k=7); on-arm adds `ADAPTIVE_VERIFY=1`, which sets `"enable_adaptive_verification":true` plus `VLLM_GDN_ADAPTIVE_VERIFY=1` and `VLLM_FA2_VARLEN_VERIFY=1` |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\spec-1\{id_off1,id_on,id_off2}\` (`job.log`, `labd/labdacc_*.json`, `container.log`) |

## The patch

The patch is gated by environment knobs:
- **GDN backend:** reports that it accepts device query lengths below the host bound for pure spec-decode batches.
- **Varlen graph bound:** declares `num_spec+1`.
- **Profiler rows:** filters the profiler's over-long dummy rows out of spec decode.
- **FA2 (B's target backend):** also declares a varlen bound of k+1 under its own knob. Without it, vLLM refuses adaptive verification on `FLASH_ATTN`.

The patch dry-runs at fuzz 0, and the launcher knob test passes: unset, empty or 0 give byte-identical argv and env in 5 arms.

## Protocol: greedy bit-identity before any speed run (appendix: "Validate it with the greedy bit-identity check below before any speed run")

`bench/labd_accept.py` is teacher-forced. The off-arm captures greedy targets (`--recapture`) and freezes them; each later arm replays the same targets in 128-token chunks and reports, per task, the first position where the server's greedy token departs from the target (`firstbad`) and how many chunks stayed clean. Each arm is a fresh boot with the check as the first traffic after `warmup.sh` (evals-7).

```bash
venv/bin/python bench/labd_accept.py av-off  --ctx 20000 --recapture   # boot 1, adaptive off
venv/bin/python bench/labd_accept.py av-on   --ctx 20000               # boot 2, ADAPTIVE_VERIFY=1
venv/bin/python bench/labd_accept.py av-off2 --ctx 20000               # boot 3, adaptive off (control)
```

## Numbers

| Arm | Clean chunks | `firstbad` (copy, code, edit, quote, summary, qa) | Teacher-forced tok/step | Server-reported mean acceptance length |
|---|---|---|---|---|
| off (boot 1) | 18/23 | –, –, –, 255, 186, 156 | 3.359 | – |
| **on (boot 2)** | **0/23** | **1, 0, 0, 0, 0, 0** | 1.000 (nothing countable) | **1.75**, per-position 0.75 |
| off (boot 3, control) | 18/23 | –, –, –, 255, 186, 156 | 3.359 | – |

Off-arm behaviour is bit-identical across the two boots (same clean chunks, same first-bad positions, same 688 steps). With adaptive verification on, **every task departs from the reference greedy text at token 0 or 1**, and the drafter's acceptance collapses: 1,694 drafted for 182 accepted in a 12-second window. The server reports no error; the engine shut down only when the container was stopped.

## Verdict

**INVALID.** The lossless item fails its greedy bit-identity check from the first generated token, so per the protocol the item stops here and no speed run was made. The appendix's lead caveat applies: KDA's opt-in carries RecoverSSM metadata that Qwen's GDN backend lacks. Trimming a row's verify length with GDN's one-state-slot-per-token layout and `num_accepted_tokens` indexing is evidently not safe as patched. The earliest token is already wrong, which points at the state or conv-window read for a trimmed row, or at FA2's varlen path, rather than at rounding (inferred).

Consequences for the ledger (gates as written):
- **spec-2** (gate: spec-1) is skipped.
- **spec-4** (gate: spec-1) is skipped.
- **spec-3** (gate: spec-1 WIN) is skipped.

The finding stands for the user: the "S patch" estimate was optimistic. A correct GDN opt-in needs the device-side state handling the KDA path has, which is M–L work, not a flag flip.
