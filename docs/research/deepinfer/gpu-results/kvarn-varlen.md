# kvarn-varlen: KVarN prefix-cache corruption on verify-length change

| Field | Value |
|---|---|
| Date | 2026-10-21 (steps 1–4) and 2026-10-23 12:10–14:52 SAST (full sweeps) |
| Commit | `bench/kvarn_varlen_repro.py`; trace and replay hooks `experiments/patches/kvarn-varlen-{sched,fp}.patch` (step 2 only) |
| Image | steps 1, 4: `ghcr.io/syv-ai/hyperqwen:latest` `c696d22e80b4…` (main `2cf50a0`); steps 2–3: `hq:exp-kvv`; sweeps: `latest` `0cf71b42d301…` (main `ce9615c`) |
| `.env` knobs | `SPEC=dflash2 CTX=huge DFLASH_TOKENS=15 LOOKUP_ADAPTIVE=1` (the configuration the launcher pins: adaptive 8 ↔ 16 verify length, prefix cache on, KVarN) |
| Protocol | Per prompt residue (prompt length mod 128): one cold run and two cache-hit runs of the same prompt in one boot, 300 greedy tokens, compared by output sha. A correctness test: the stock E pool paged over PCIe during these boots (median 232 MB/s), which costs speed but cannot change outputs. |
| Raw data | `G:\dev\hq-results\deepinfer-20261020\kvarn-varlen\s{1..6}_*\kvv.txt` |
| Timebox | 4 h of GPU time: ~1 h (steps 1–4) + ~2.5 h (two sweeps) |

## What the launcher says

`single-user/start_qwen.sh:403-430` pins the verify block for `DFLASH_TOKENS>7 + CTX=huge + prefix caching`, because the adaptive length "corrupts the second turn over a shared prefix on the KVarN cache": 38/793 wrong cache hits at five of six residues tested, clean with the length pinned or prefix caching off. Root cause recorded as open.

## Results

| Step | What | Result |
|---|---|---|
| 1 `s1_stock` | 6 residues (0, 16, 32, 64, 96, 124), arms: plain hit, hit after a reset/interposer request, no-hit | every cache hit `== cold` |
| 2 `s2_trace` | schedule and KV/GDN fingerprint traces, replay | a hit inherits the previous request's verify schedule (15 drafts at step 0 instead of 7), so the schedules differ at decode step 0; replaying the cold run under the hit's schedule reproduces the hit, and vice versa. KV and GDN fingerprints first differ exactly where the schedules do. **Verdict: schedule-only, no corruption.** |
| 3 `s3_resetfix` | reset the schedule on each new request (`VLLM_KVV_RESET_ON_NEW_REQ=1`), 2 measured hits per residue | 0 of 12 differ |
| 4 `s4_pinned` | launcher's pinned length, control | 0 differ |
| 5 `s5_sweep_gotchas` | **all residues 0–127**, default corpus (prompt ~21.2k tokens, ~17k cached) | **128/128 hits identical to cold, 0 wrong** |
| 6 `s6_sweep_labd` | **all residues 0–127**, frozen labd corpus (prompt ~23.0k, ~19.5k cached) | **128/128 identical, 0 wrong** |

## Verdict: INFO (not reproduced)

On the current image, with the adaptive verify length on, a KVarN prefix-cache hit reproduces its cold run at all 256 residues tested, across two corpora. The one difference found (step 2) is schedule inheritance: a hit starts with the previous request's verify width, which changes only acceptance at the first steps, not the target's output. The 38/793 corruption the launcher guards against does not reproduce. A KVarN fix landed after the pin was written (for example `kvarn-recycled-pages` for #208's late-flush corruption), which may be the cure **(inference, not bisected)**.

**Decision for the user:** the pin in `start_qwen.sh:425-430` (`VLLM_DFLASH2_LOOKUP_ADAPTIVE=0` for this configuration) can be lifted on this evidence. The pin's own comment records pinned as the faster setting anyway (14.71 vs 14.29 tok/step), so lifting it buys correctness headroom, not speed. spec-1 (adaptive verification for GDN) stays blocked on its own INVALID result, independent of this.
